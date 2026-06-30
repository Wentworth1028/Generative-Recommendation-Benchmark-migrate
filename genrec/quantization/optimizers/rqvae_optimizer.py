
import torch
import torch.nn.functional as F
from genrec.quantization.optimizers.base_optimizer import AbstractTokenizerOptimizer



class RQVAETokenizerOptimizer(AbstractTokenizerOptimizer):

    def __init__(self, config: dict, tokenizer: torch.nn.Module):
        super().__init__(config)
        self.tokenizer = tokenizer
        
        self.quant_loss_weight = self.config['quant_loss_weight']
        self.popularity_balance_weight = float(self.config.get('popularity_balance_weight', 0.0))
        self.popularity_softmax_temperature = float(self.config.get('popularity_softmax_temperature', 1.0))
        self.popularity_weight_transform = str(self.config.get('popularity_weight_transform', 'nominal')).lower()
        self.popularity_nominal_gamma = float(self.config.get('popularity_nominal_gamma', 0.5))
        if self.popularity_nominal_gamma <= 0.0:
            raise ValueError("popularity_nominal_gamma must be positive.")
        self.popularity_balance_eps = float(self.config.get('popularity_balance_eps', 1e-8))
        self.popularity_nominal_clip_value = self._resolve_nominal_clip_value()
        self.popularity_nominal_denominator = self._compute_nominal_denominator()
        self.popularity_balance_top_k = int(self.config.get('popularity_balance_top_k', 0) or 0)
        self.popularity_balance_log_distribution = self._as_bool(
            self.config.get('popularity_balance_log_distribution', False)
        )
        self.popularity_balance_disabled_layers = self._parse_disabled_layers(
            self.config.get('popularity_balance_disabled_layers', [])
        )
        learning_rate = self.config['learning_rate']
        
        # self.torch_optimizer = torch.optim.Adagrad(self.tokenizer.parameters(), lr=learning_rate)
        weight_decay = self.config.get('weight_decay', 0.1)

        decay_params = []
        no_decay_params = []
        for name, param in self.tokenizer.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim == 1 or "bn" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]


        self.torch_optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=learning_rate
        )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def zero_grad(self):
        self.torch_optimizer.zero_grad()

    def _popularity_counts_tensor(self) -> torch.Tensor:
        item_popularity = self.config.get('item_popularity', {})
        if not item_popularity:
            return torch.empty(0, dtype=torch.float32)
        counts = [float(value) for value in item_popularity.values()]
        return torch.tensor(counts, dtype=torch.float32).clamp_min(0.0)

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return float(value)

    def _resolve_nominal_clip_value(self):
        explicit_clip = self._optional_float(self.config.get('popularity_nominal_clip_value'))
        if explicit_clip is not None:
            return max(0.0, explicit_clip)

        clip_quantile = self._optional_float(self.config.get('popularity_nominal_clip_quantile', 0.995))
        if clip_quantile is None or clip_quantile <= 0.0 or clip_quantile >= 1.0:
            return None

        counts = self._popularity_counts_tensor()
        positive_counts = counts[counts > 0.0]
        if positive_counts.numel() == 0:
            return None
        return float(torch.quantile(positive_counts, clip_quantile).item())

    def _compute_nominal_denominator(self) -> float:
        counts = self._popularity_counts_tensor()
        if counts.numel() == 0:
            return 1.0
        if self.popularity_nominal_clip_value is not None:
            counts = counts.clamp_max(self.popularity_nominal_clip_value)
        denominator = counts.pow(self.popularity_nominal_gamma).sum().clamp_min(
            self.popularity_balance_eps
        )
        return float(denominator.item())
    
    def _transform_popularity_weights(self, popularity_weights: torch.Tensor):
        popularity_weights = popularity_weights.float().clamp_min(0.0)
        if self.popularity_weight_transform == 'log1p':
            return torch.log1p(popularity_weights)
        if self.popularity_weight_transform == 'nominal':
            if self.popularity_nominal_clip_value is not None:
                popularity_weights = popularity_weights.clamp_max(self.popularity_nominal_clip_value)
            denominator = popularity_weights.new_tensor(self.popularity_nominal_denominator)
            return popularity_weights.pow(self.popularity_nominal_gamma) / denominator
        raise ValueError(f"Unsupported popularity_weight_transform: {self.popularity_weight_transform}")

    def _parse_disabled_layers(self, value):
        if value is None or value == "":
            return set()
        if isinstance(value, str):
            text = value.strip()
            if text in ("[]", ""):
                return set()
            text = text.strip("[]")
            return {int(item.strip()) for item in text.split(",") if item.strip()}
        if isinstance(value, (list, tuple, set)):
            return {int(item) for item in value}
        return {int(value)}

    def _codebook_squared_l2_scale(self, codebooks: torch.Tensor) -> torch.Tensor:
        codebook_l2 = codebooks.detach().norm(p=2, dim=-1)
        return codebook_l2.mean(dim=-1).pow(2).clamp_min(self.popularity_balance_eps)

    def _normalize_popularity_distances(
        self,
        distances: torch.Tensor,
        codebooks: torch.Tensor,
    ) -> torch.Tensor:
        scale = self._codebook_squared_l2_scale(codebooks).view(1, -1, 1)
        return distances / scale

    def popularity_balance_entropy_floor(self) -> float:
        codebook_size = int(self.config.get('codebook_size', 256))
        if self.popularity_balance_top_k > 0:
            support_size = min(max(1, self.popularity_balance_top_k), codebook_size)
        else:
            support_size = codebook_size
        return -float(torch.log(torch.tensor(float(support_size))))

    def _compute_candidate_token_mass(
        self,
        distances: torch.Tensor,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.popularity_balance_top_k > 0 and self.popularity_balance_top_k < distances.size(-1):
            top_k = self.popularity_balance_top_k
            _, topk_indices = torch.topk(
                distances.detach(),
                k=top_k,
                dim=-1,
                largest=False,
            )
            topk_logits = logits.gather(dim=-1, index=topk_indices)
            topk_assignment = torch.softmax(topk_logits, dim=-1)

            soft_assignment = torch.zeros_like(logits)
            soft_assignment.scatter_(dim=-1, index=topk_indices, src=topk_assignment)

            candidate_mask_per_item = torch.zeros_like(logits, dtype=torch.bool)
            candidate_mask_per_item.scatter_(
                dim=-1,
                index=topk_indices,
                src=torch.ones_like(topk_indices, dtype=torch.bool),
            )
            candidate_mask = candidate_mask_per_item.any(dim=0)
        else:
            soft_assignment = torch.softmax(logits, dim=-1)
            candidate_mask = torch.ones(
                logits.size(1),
                logits.size(2),
                device=logits.device,
                dtype=torch.bool,
            )

        weighted_assignment = soft_assignment * weights.view(-1, 1, 1)
        token_mass = weighted_assignment.sum(dim=0)
        return token_mass, candidate_mask

    def _popularity_distribution(
        self,
        distances: torch.Tensor,
        popularity_weights: torch.Tensor,
        codebooks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = max(self.popularity_softmax_temperature, self.popularity_balance_eps)
        normalized_distances = self._normalize_popularity_distances(distances, codebooks)
        logits = -normalized_distances / temperature
        weights = self._transform_popularity_weights(popularity_weights).to(
            device=distances.device,
            dtype=distances.dtype,
        )
        token_mass, candidate_mask = self._compute_candidate_token_mass(distances, logits, weights)
        token_mass = token_mass.masked_fill(~candidate_mask, 0.0)
        token_distribution = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(
            self.popularity_balance_eps
        )
        token_distribution = token_distribution.masked_fill(~candidate_mask, 0.0)
        return token_distribution, token_mass, candidate_mask, normalized_distances, logits

    def _compute_popularity_balance_loss(
        self,
        distances: torch.Tensor,
        popularity_weights: torch.Tensor,
        codebooks: torch.Tensor,
    ):
        if self.popularity_balance_weight <= 0.0:
            return distances.new_tensor(0.0)
        if distances is None or popularity_weights is None:
            return distances.new_tensor(0.0) if distances is not None else torch.tensor(0.0)

        token_distribution, _, candidate_mask, _, _ = self._popularity_distribution(
            distances,
            popularity_weights,
            codebooks,
        )
        entropy_objective = (
            token_distribution * torch.log(token_distribution.clamp_min(self.popularity_balance_eps))
        ).sum(dim=-1)
        if not self.popularity_balance_disabled_layers:
            return entropy_objective.mean()

        enabled_mask = torch.ones(
            entropy_objective.size(0),
            device=entropy_objective.device,
            dtype=torch.bool,
        )
        for layer_idx in self.popularity_balance_disabled_layers:
            if 0 <= layer_idx < enabled_mask.size(0):
                enabled_mask[layer_idx] = False

        if not enabled_mask.any():
            return distances.new_tensor(0.0)
        return entropy_objective[enabled_mask].mean()

    def compute_popularity_distribution_metrics(
        self,
        tokenizer_output: tuple,
        popularity_weights=None,
    ) -> dict[str, dict]:
        if not self.popularity_balance_log_distribution:
            return {}
        if popularity_weights is None:
            return {}

        distances = tokenizer_output[3] if len(tokenizer_output) > 3 else None
        quantization_context = tokenizer_output[4] if len(tokenizer_output) > 4 else {}
        codebooks = quantization_context.get("codebooks") if isinstance(quantization_context, dict) else None
        if distances is None or codebooks is None:
            return {}

        with torch.no_grad():
            regularization_distances = distances.detach()
            regularization_codebooks = codebooks.detach()
            popularity_weights = popularity_weights.detach()
            (
                token_distribution,
                token_mass,
                candidate_mask,
                normalized_distances,
                logits,
            ) = self._popularity_distribution(
                regularization_distances,
                popularity_weights,
                regularization_codebooks,
            )

            scalars: dict[str, float] = {}
            histograms: dict[str, torch.Tensor] = {}
            n_layers = token_distribution.size(0)
            codebook_size = token_distribution.size(1)
            nominal_support = (
                min(max(1, self.popularity_balance_top_k), codebook_size)
                if self.popularity_balance_top_k > 0
                else codebook_size
            )
            nominal_floor = -float(torch.log(token_distribution.new_tensor(float(nominal_support))).cpu())

            mean_values: dict[str, list[float]] = {
                "entropy": [],
                "entropy_loss": [],
                "entropy_gap_to_support": [],
                "entropy_gap_to_nominal": [],
                "effective_tokens": [],
                "p_max": [],
                "p_std": [],
                "top1_share": [],
                "top5_share": [],
                "top10_share": [],
                "candidate_support": [],
                "distance_std_mean": [],
                "distance_range_mean": [],
                "codebook_squared_l2_scale": [],
                "normalized_distance_range_mean": [],
                "logit_range_mean": [],
            }

            for layer_idx in range(n_layers):
                layer_mask = candidate_mask[layer_idx]
                support = int(layer_mask.sum().item())
                if support <= 0:
                    continue

                p_values = token_distribution[layer_idx][layer_mask]
                entropy = -(
                    p_values * torch.log(p_values.clamp_min(self.popularity_balance_eps))
                ).sum()
                entropy_loss = -entropy
                support_floor = -torch.log(p_values.new_tensor(float(support)))
                sorted_p = torch.sort(p_values, descending=True).values
                top5_count = min(5, sorted_p.numel())
                top10_count = min(10, sorted_p.numel())

                raw_distances = regularization_distances[:, layer_idx, :]
                layer_normalized_distances = normalized_distances[:, layer_idx, :]
                layer_logits = logits[:, layer_idx, :]
                codebook_squared_l2_scale = self._codebook_squared_l2_scale(
                    regularization_codebooks[layer_idx : layer_idx + 1]
                )[0]
                distance_range = raw_distances.max(dim=-1).values - raw_distances.min(dim=-1).values
                normalized_distance_range = (
                    layer_normalized_distances.max(dim=-1).values
                    - layer_normalized_distances.min(dim=-1).values
                )
                logit_range = layer_logits.max(dim=-1).values - layer_logits.min(dim=-1).values

                layer_scalars = {
                    "entropy": float(entropy.cpu()),
                    "entropy_loss": float(entropy_loss.cpu()),
                    "entropy_gap_to_support": float((entropy_loss - support_floor).cpu()),
                    "entropy_gap_to_nominal": float((entropy_loss - nominal_floor).cpu()),
                    "effective_tokens": float(torch.exp(entropy).cpu()),
                    "p_max": float(p_values.max().cpu()),
                    "p_min": float(p_values.min().cpu()),
                    "p_std": float(p_values.std(unbiased=False).cpu()),
                    "top1_share": float(sorted_p[:1].sum().cpu()),
                    "top5_share": float(sorted_p[:top5_count].sum().cpu()),
                    "top10_share": float(sorted_p[:top10_count].sum().cpu()),
                    "candidate_support": float(support),
                    "distance_std_mean": float(raw_distances.std(dim=-1, unbiased=False).mean().cpu()),
                    "distance_range_mean": float(distance_range.mean().cpu()),
                    "codebook_squared_l2_scale": float(codebook_squared_l2_scale.cpu()),
                    "normalized_distance_range_mean": float(normalized_distance_range.mean().cpu()),
                    "logit_range_mean": float(logit_range.mean().cpu()),
                }

                for name, value in layer_scalars.items():
                    scalars[f"layer_{layer_idx}/{name}"] = value
                    if name in mean_values:
                        mean_values[name].append(value)

                histograms[f"layer_{layer_idx}/p"] = p_values.detach().float().cpu()
                histograms[f"layer_{layer_idx}/mass"] = token_mass[layer_idx][layer_mask].detach().float().cpu()

            for name, values in mean_values.items():
                if values:
                    scalars[f"mean/{name}"] = float(sum(values) / len(values))

            scalars["config/temperature"] = float(self.popularity_softmax_temperature)
            scalars["config/top_k"] = float(self.popularity_balance_top_k)
            scalars["config/nominal_support"] = float(nominal_support)
            scalars["config/popularity_nominal_gamma"] = float(self.popularity_nominal_gamma)
            scalars["config/popularity_nominal_denominator"] = float(self.popularity_nominal_denominator)
            if self.popularity_nominal_clip_value is not None:
                scalars["config/popularity_nominal_clip_value"] = float(self.popularity_nominal_clip_value)
            return {"scalars": scalars, "histograms": histograms}

    def compute_loss(self, original_embeddings: torch.Tensor, tokenizer_output: tuple, popularity_weights=None):
        quantized_embeddings, _, commit_loss, distances = tokenizer_output[:4]
        quantization_context = tokenizer_output[4] if len(tokenizer_output) > 4 else {}
        codebooks = quantization_context.get("codebooks") if isinstance(quantization_context, dict) else None
        
        reconstruction_loss = F.mse_loss(quantized_embeddings, original_embeddings)
        if codebooks is None:
            popularity_balance_loss = distances.new_tensor(0.0)
        else:
            popularity_balance_loss = self._compute_popularity_balance_loss(
                distances,
                popularity_weights,
                codebooks,
            )
        total_loss = (
            reconstruction_loss
            + self.quant_loss_weight * commit_loss
            + self.popularity_balance_weight * popularity_balance_loss
        )
        
        return total_loss, reconstruction_loss, commit_loss, popularity_balance_loss

    def step(self):
        self.torch_optimizer.step()
    
    def move_optimizer_state_to_device(self, device):
        for state in self.torch_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
