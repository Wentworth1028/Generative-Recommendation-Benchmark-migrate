import torch
import torch.nn.functional as F
from genrec.quantization.optimizers.base_optimizer import AbstractTokenizerOptimizer


class RQVAETokenizerOptimizer(AbstractTokenizerOptimizer):

    def __init__(self, config: dict, tokenizer: torch.nn.Module):
        super().__init__(config)
        self.tokenizer = tokenizer

        self.quant_loss_weight = self.config['quant_loss_weight']
        self.popularity_balance_weight = float(self._config_get('balance_weight', 'popularity_balance_weight', 0.0))
        self.popularity_balance_mode = str(self._config_get('balance_mode', 'popularity_balance_mode', 'batch_entropy')).lower()
        if self.popularity_balance_mode not in {'batch_entropy', 'ema_item'}:
            raise ValueError(
                f"Unsupported popularity_balance_mode: {self.popularity_balance_mode}. "
                "Expected 'batch_entropy' or 'ema_item'."
            )
        self.popularity_softmax_temperature = float(self._config_get('softmax_temperature', 'popularity_softmax_temperature', 1.0))
        self.popularity_weight_transform = str(self._config_get('weight_transform', 'popularity_weight_transform', 'nominal')).lower()
        self.popularity_nominal_gamma = float(self._config_get('nominal_gamma', 'popularity_nominal_gamma', 0.5))
        if self.popularity_nominal_gamma <= 0.0:
            raise ValueError("popularity_nominal_gamma must be positive.")
        self.popularity_balance_eps = float(self._config_get('balance_eps', 'popularity_balance_eps', 1e-8))
        self.popularity_nominal_clip_value = self._resolve_nominal_clip_value()
        self.popularity_nominal_denominator = self._compute_nominal_denominator()
        self.popularity_transformed_weight_mean = self._compute_transformed_popularity_mean()
        self.popularity_balance_top_k = int(self._config_get('top_k', 'popularity_balance_top_k', 0) or 0)
        self.popularity_balance_log_distribution = self._as_bool(
            self._config_get('log_distribution', 'popularity_balance_log_distribution', False)
        )
        self.token_balance_enabled = self._as_bool(
            self._config_get('token_balance_enabled', 'popularity_token_balance_enabled', True)
        )
        self.popularity_prefix_balance_enabled = self._as_bool(
            self._config_get('prefix_balance_enabled', 'popularity_prefix_balance_enabled', False)
        )
        self.popularity_ema_half_life_epochs = float(self._config_get('ema_half_life_epochs', 'popularity_ema_half_life_epochs', 1.0))
        self.popularity_ema_half_life_items = self._resolve_ema_half_life_items()
        self.popularity_ema_normalize_item_weights = self._as_bool(
            self._config_get('ema_normalize_item_weights', 'popularity_ema_normalize_item_weights', True)
        )
        self.popularity_ema_mass: torch.Tensor | None = None
        self.popularity_prefix_ema_mass: dict[int, torch.Tensor] = {}
        self._pending_popularity_ema_observation: torch.Tensor | None = None
        self._pending_popularity_prefix_ema_observation: dict[int, torch.Tensor] | None = None
        self._pending_popularity_ema_count = 0
        self._last_popularity_layer_loss: torch.Tensor | None = None
        self._last_popularity_layer_enabled_mask: torch.Tensor | None = None
        self.popularity_balance_disabled_layers = self._parse_disabled_layers(
            self._config_get('disabled_layers', 'popularity_balance_disabled_layers', [])
        )
        self.popularity_balance_layer_weights = self._parse_layer_weights(
            self._config_get('layer_weights', 'popularity_balance_layer_weights', [])
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
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        self.torch_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _config_get(self, key: str, legacy_key: str | None = None, default=None):
        if key in self.config:
            return self.config.get(key)
        if legacy_key is not None and legacy_key in self.config:
            return self.config.get(legacy_key)
        return default

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
        explicit_clip = self._optional_float(self._config_get('nominal_clip_value', 'popularity_nominal_clip_value'))
        if explicit_clip is not None:
            return max(0.0, explicit_clip)

        clip_quantile = self._optional_float(self._config_get('nominal_clip_quantile', 'popularity_nominal_clip_quantile', 0.995))
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
        denominator = counts.pow(self.popularity_nominal_gamma).sum().clamp_min(self.popularity_balance_eps)
        return float(denominator.item())

    def _compute_transformed_popularity_mean(self) -> float:
        counts = self._popularity_counts_tensor()
        if counts.numel() == 0:
            return 1.0
        transformed = self._transform_popularity_weights(counts)
        return float(transformed.mean().clamp_min(self.popularity_balance_eps).item())

    def _resolve_ema_half_life_items(self) -> float:
        explicit_items = self._optional_float(self._config_get('ema_half_life_items', 'popularity_ema_half_life_items'))
        if explicit_items is not None and explicit_items > 0.0:
            return explicit_items
        item_count = max(1, len(self.config.get('item_popularity', {})))
        return max(1.0, self.popularity_ema_half_life_epochs * item_count)

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

    def _transform_ema_item_weights(self, popularity_weights: torch.Tensor) -> torch.Tensor:
        weights = self._transform_popularity_weights(popularity_weights)
        if self.popularity_ema_normalize_item_weights:
            weights = weights / max(self.popularity_transformed_weight_mean, self.popularity_balance_eps)
        return weights

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

    def _parse_layer_weights(self, value):
        if value is None or value == "":
            return []
        if isinstance(value, str):
            text = value.strip()
            if text in ("[]", ""):
                return []
            text = text.strip("[]")
            return [float(item.strip()) for item in text.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return [float(value)]

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

    def _compute_candidate_assignment(
        self,
        distances: torch.Tensor,
        logits: torch.Tensor,
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

        return soft_assignment, candidate_mask

    def _compute_candidate_token_mass(
        self,
        distances: torch.Tensor,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        soft_assignment, candidate_mask = self._compute_candidate_assignment(distances, logits)
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
        token_distribution = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.popularity_balance_eps)
        token_distribution = token_distribution.masked_fill(~candidate_mask, 0.0)
        return token_distribution, token_mass, candidate_mask, normalized_distances, logits

    def _enabled_layer_mask(self, n_layers: int, device: torch.device) -> torch.Tensor:
        weights = self._layer_weight_tensor(n_layers, device=device, dtype=torch.float32)
        return weights > 0.0

    def _layer_weight_tensor(self, n_layers: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.popularity_balance_layer_weights:
            if len(self.popularity_balance_layer_weights) == n_layers - 1:
                # RQ-VAE produces four SID tokens, but the last quantizer is
                # Sinkhorn-balanced and mainly serves as a supplemental bucket.
                # Three weights mean "effective semantic layers only"; the
                # final supplemental layer is intentionally not regularized.
                layer_weights = [*self.popularity_balance_layer_weights, 0.0]
            elif len(self.popularity_balance_layer_weights) == n_layers:
                layer_weights = self.popularity_balance_layer_weights
            else:
                raise ValueError(
                    "popularity_balance_layer_weights must have either one value per effective "
                    "semantic layer or one value per quantizer: "
                    f"got {len(self.popularity_balance_layer_weights)} values for {n_layers} quantizers."
                )
            weights = torch.tensor(layer_weights, device=device, dtype=dtype)
        else:
            weights = torch.ones(n_layers, device=device, dtype=dtype)
            if n_layers > 1:
                weights[-1] = 0.0
        weights = weights.clamp_min(0.0)
        for layer_idx in self.popularity_balance_disabled_layers:
            if 0 <= layer_idx < weights.size(0):
                weights[layer_idx] = 0.0
        return weights

    def _weighted_layer_mean(self, layer_values: torch.Tensor) -> torch.Tensor:
        layer_weights = self._layer_weight_tensor(
            layer_values.size(0),
            device=layer_values.device,
            dtype=layer_values.dtype,
        )
        denominator = layer_weights.sum()
        if denominator <= self.popularity_balance_eps:
            return layer_values.new_tensor(0.0)
        return (layer_values * layer_weights).sum() / denominator

    def _compute_batch_entropy_popularity_balance_loss(
        self,
        distances: torch.Tensor,
        popularity_weights: torch.Tensor,
        codebooks: torch.Tensor,
    ):
        token_distribution, _, candidate_mask, _, _ = self._popularity_distribution(
            distances,
            popularity_weights,
            codebooks,
        )
        entropy_objective = (
            token_distribution * torch.log(token_distribution.clamp_min(self.popularity_balance_eps))
        ).sum(dim=-1)
        return self._weighted_layer_mean(entropy_objective)

    def _ensure_popularity_ema_mass(self, distances: torch.Tensor) -> torch.Tensor:
        n_layers = distances.size(1)
        codebook_size = distances.size(2)
        expected_shape = (n_layers, codebook_size)
        if (
            self.popularity_ema_mass is None
            or tuple(self.popularity_ema_mass.shape) != expected_shape
            or self.popularity_ema_mass.device != distances.device
            or self.popularity_ema_mass.dtype != distances.dtype
        ):
            self.popularity_ema_mass = distances.new_full(expected_shape, 1.0 / codebook_size)
        return self.popularity_ema_mass

    def _popularity_ema_distribution(self, distances: torch.Tensor | None = None) -> torch.Tensor | None:
        if self.popularity_ema_mass is None:
            if distances is None:
                return None
            self._ensure_popularity_ema_mass(distances)
        mass = self.popularity_ema_mass
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(self.popularity_balance_eps)

    def _prefix_depths(self, n_layers: int, device: torch.device) -> list[int]:
        layer_weights = self._layer_weight_tensor(n_layers, device=device, dtype=torch.float32)
        return [
            depth
            for depth in range(2, n_layers + 1)
            if layer_weights[depth - 1].item() > 0.0
        ]

    def _ensure_popularity_prefix_ema_mass(
        self,
        depth: int,
        codebook_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_prefixes = codebook_size ** depth
        mass = self.popularity_prefix_ema_mass.get(depth)
        if (
            mass is None
            or mass.numel() != num_prefixes
            or mass.device != device
            or mass.dtype != dtype
        ):
            mass = torch.full(
                (num_prefixes,),
                1.0 / float(num_prefixes),
                device=device,
                dtype=dtype,
            )
            self.popularity_prefix_ema_mass[depth] = mass
        return mass

    def _popularity_prefix_ema_distribution(
        self,
        depth: int,
        codebook_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mass = self._ensure_popularity_prefix_ema_mass(depth, codebook_size, device, dtype)
        return mass / mass.sum().clamp_min(self.popularity_balance_eps)

    def _compute_prefix_ema_item_popularity_balance_loss(
        self,
        residuals: torch.Tensor | None,
        codebooks: torch.Tensor,
        item_weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.popularity_prefix_balance_enabled
            or residuals is None
            or self.popularity_balance_top_k <= 0
        ):
            return codebooks.new_tensor(0.0)

        n_layers = min(residuals.size(1), codebooks.size(0))
        prefix_depths = self._prefix_depths(n_layers, device=codebooks.device)
        if not prefix_depths:
            return codebooks.new_tensor(0.0)

        batch_size = residuals.size(0)
        codebook_size = codebooks.size(1)
        beam_width = min(max(1, self.popularity_balance_top_k), codebook_size)
        temperature = max(self.popularity_softmax_temperature, self.popularity_balance_eps)
        distance_scales = self._codebook_squared_l2_scale(codebooks).to(
            device=codebooks.device,
            dtype=codebooks.dtype,
        )

        latent = residuals[:, 0, :]
        first_distances = ((latent.unsqueeze(1) - codebooks[0].unsqueeze(0)) ** 2).sum(dim=-1)
        first_logits = -(first_distances / distance_scales[0]) / temperature
        _, first_indices = torch.topk(first_distances.detach(), k=beam_width, dim=-1, largest=False)
        first_probs = torch.softmax(first_logits.gather(dim=-1, index=first_indices), dim=-1)

        beam_keys = first_indices
        beam_scores = first_probs
        first_code_vectors = codebooks[0].index_select(0, first_indices.reshape(-1)).view(
            batch_size,
            beam_width,
            -1,
        )
        beam_residuals = latent.unsqueeze(1) - first_code_vectors

        prefix_losses = []
        prefix_observations: dict[int, torch.Tensor] = {}
        max_depth = max(prefix_depths)

        for layer_idx in range(1, max_depth):
            current_codebook = codebooks[layer_idx]
            child_distances = (
                (beam_residuals.unsqueeze(2) - current_codebook.view(1, 1, codebook_size, -1)) ** 2
            ).sum(dim=-1)
            child_logits = -(child_distances / distance_scales[layer_idx]) / temperature
            _, child_indices = torch.topk(child_distances.detach(), k=beam_width, dim=-1, largest=False)
            child_probs = torch.softmax(child_logits.gather(dim=-1, index=child_indices), dim=-1)

            candidate_scores = beam_scores.unsqueeze(-1) * child_probs
            flat_scores = candidate_scores.reshape(batch_size, -1)
            kept_scores, kept_flat_indices = torch.topk(flat_scores, k=beam_width, dim=-1, largest=True)

            flat_child_indices = child_indices.reshape(batch_size, -1)
            selected_children = flat_child_indices.gather(dim=-1, index=kept_flat_indices)
            parent_slot = kept_flat_indices.div(beam_width, rounding_mode='floor')
            parent_keys = beam_keys.gather(dim=-1, index=parent_slot)
            beam_keys = parent_keys * codebook_size + selected_children
            beam_scores = kept_scores / kept_scores.sum(dim=-1, keepdim=True).clamp_min(self.popularity_balance_eps)

            parent_residuals = beam_residuals.gather(
                dim=1,
                index=parent_slot.unsqueeze(-1).expand(-1, -1, beam_residuals.size(-1)),
            )
            selected_code_vectors = current_codebook.index_select(0, selected_children.reshape(-1)).view(
                batch_size,
                beam_width,
                -1,
            )
            beam_residuals = parent_residuals - selected_code_vectors

            depth = layer_idx + 1
            if depth not in prefix_depths:
                continue

            prefix_distribution = self._popularity_prefix_ema_distribution(
                depth,
                codebook_size,
                device=codebooks.device,
                dtype=codebooks.dtype,
            ).detach()
            uniform = codebooks.new_tensor(1.0 / float(codebook_size ** depth))
            prefix_bias = torch.log(
                (prefix_distribution + self.popularity_balance_eps)
                / (uniform + self.popularity_balance_eps)
            )
            selected_bias = prefix_bias.gather(dim=0, index=beam_keys.reshape(-1)).view(batch_size, beam_width)
            item_prefix_loss = (beam_scores * selected_bias * item_weights.view(-1, 1)).sum(dim=-1)
            prefix_losses.append((depth, item_prefix_loss.mean()))

            observation = codebooks.new_zeros(codebook_size ** depth)
            weighted_scores = (beam_scores.detach() * item_weights.detach().view(-1, 1)).reshape(-1)
            observation.scatter_add_(dim=0, index=beam_keys.detach().reshape(-1), src=weighted_scores)
            prefix_observations[depth] = observation / float(batch_size)

        self._pending_popularity_prefix_ema_observation = prefix_observations or None
        if not prefix_losses:
            return codebooks.new_tensor(0.0)

        layer_weights = self._layer_weight_tensor(n_layers, device=codebooks.device, dtype=codebooks.dtype)
        weighted_loss = codebooks.new_tensor(0.0)
        weight_sum = codebooks.new_tensor(0.0)
        for depth, prefix_loss in prefix_losses:
            weight = layer_weights[depth - 1]
            weighted_loss = weighted_loss + prefix_loss * weight
            weight_sum = weight_sum + weight
        return weighted_loss / weight_sum.clamp_min(self.popularity_balance_eps)

    def _compute_ema_item_popularity_balance_loss(
        self,
        distances: torch.Tensor,
        popularity_weights: torch.Tensor,
        codebooks: torch.Tensor,
        residuals: torch.Tensor | None = None,
    ):
        temperature = max(self.popularity_softmax_temperature, self.popularity_balance_eps)
        normalized_distances = self._normalize_popularity_distances(distances, codebooks)
        logits = -normalized_distances / temperature
        soft_assignment, _ = self._compute_candidate_assignment(distances, logits)
        item_weights = self._transform_ema_item_weights(popularity_weights).to(
            device=distances.device,
            dtype=distances.dtype,
        )

        ema_distribution = self._popularity_ema_distribution(distances).detach()
        codebook_size = distances.size(-1)
        uniform = distances.new_tensor(1.0 / codebook_size)
        ema_bias = torch.log((ema_distribution + self.popularity_balance_eps) / (uniform + self.popularity_balance_eps))

        per_item_layer_loss = (soft_assignment * ema_bias.unsqueeze(0) * item_weights.view(-1, 1, 1)).sum(dim=-1)
        layer_loss = per_item_layer_loss.mean(dim=0)
        enabled_mask = self._enabled_layer_mask(layer_loss.size(0), layer_loss.device)
        self._last_popularity_layer_loss = layer_loss.detach()
        self._last_popularity_layer_enabled_mask = enabled_mask.detach()
        if self.token_balance_enabled:
            loss = self._weighted_layer_mean(layer_loss)
        else:
            loss = distances.new_tensor(0.0)

        hard_indices = distances.detach().argmin(dim=-1)
        hard_assignment = F.one_hot(
            hard_indices,
            num_classes=distances.size(-1),
        ).to(device=distances.device, dtype=distances.dtype)
        observation = (hard_assignment * item_weights.detach().view(-1, 1, 1)).mean(dim=0)
        self._pending_popularity_ema_observation = observation.detach()
        self._pending_popularity_ema_count = int(distances.size(0))

        prefix_loss = self._compute_prefix_ema_item_popularity_balance_loss(
            residuals,
            codebooks,
            item_weights,
        )
        return loss + prefix_loss

    def _compute_popularity_balance_loss(
        self,
        distances: torch.Tensor,
        popularity_weights: torch.Tensor,
        codebooks: torch.Tensor,
        residuals: torch.Tensor | None = None,
    ):
        if distances is None or popularity_weights is None:
            return distances.new_tensor(0.0) if distances is not None else torch.tensor(0.0)

        if self.popularity_balance_weight <= 0.0:
            return distances.new_tensor(0.0)

        if self.popularity_balance_mode == 'ema_item':
            return self._compute_ema_item_popularity_balance_loss(
                distances,
                popularity_weights,
                codebooks,
                residuals=residuals,
            )
        return self._compute_batch_entropy_popularity_balance_loss(
            distances,
            popularity_weights,
            codebooks,
        )

    def pop_pending_ema_observation(self):
        if self._pending_popularity_prefix_ema_observation:
            return {
                "token": self._pending_popularity_ema_observation,
                "prefix": self._pending_popularity_prefix_ema_observation,
            }, self._pending_popularity_ema_count
        return self._pending_popularity_ema_observation, self._pending_popularity_ema_count

    def clear_pending_popularity_ema_observation(self):
        self._pending_popularity_ema_observation = None
        self._pending_popularity_prefix_ema_observation = None
        self._pending_popularity_ema_count = 0

    def apply_popularity_ema_update(self, observation, item_count: int):
        if self.popularity_balance_mode != 'ema_item' or observation is None or item_count <= 0:
            self.clear_pending_popularity_ema_observation()
            return
        if isinstance(observation, dict):
            token_observation = observation.get("token")
            prefix_observation = observation.get("prefix") or {}
        else:
            token_observation = observation
            prefix_observation = {}
        if token_observation is None:
            self.clear_pending_popularity_ema_observation()
            return
        if self.popularity_ema_mass is None:
            self.popularity_ema_mass = token_observation.detach().clone().clamp_min(0.0)
        token_observation = token_observation.to(
            device=self.popularity_ema_mass.device,
            dtype=self.popularity_ema_mass.dtype,
        ).clamp_min(0.0)
        rho = 2.0 ** (-float(item_count) / max(self.popularity_ema_half_life_items, 1.0))
        with torch.no_grad():
            self.popularity_ema_mass.mul_(rho).add_(token_observation, alpha=1.0 - rho)
            self.popularity_ema_mass.clamp_min_(0.0)
            for depth, depth_observation in prefix_observation.items():
                depth = int(depth)
                if depth_observation is None:
                    continue
                depth_observation = depth_observation.detach().clamp_min(0.0)
                prefix_mass = self.popularity_prefix_ema_mass.get(depth)
                if (
                    prefix_mass is None
                    or prefix_mass.shape != depth_observation.shape
                    or prefix_mass.device != depth_observation.device
                    or prefix_mass.dtype != depth_observation.dtype
                ):
                    prefix_mass = depth_observation.clone()
                    self.popularity_prefix_ema_mass[depth] = prefix_mass
                else:
                    prefix_mass.mul_(rho).add_(depth_observation, alpha=1.0 - rho)
                    prefix_mass.clamp_min_(0.0)
        self.clear_pending_popularity_ema_observation()

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
                entropy = -(p_values * torch.log(p_values.clamp_min(self.popularity_balance_eps))).sum()
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
                    layer_normalized_distances.max(dim=-1).values - layer_normalized_distances.min(dim=-1).values
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

            ema_distribution = self._popularity_ema_distribution()
            if ema_distribution is not None:
                ema_mean_values: dict[str, list[float]] = {
                    "entropy": [],
                    "effective_tokens": [],
                    "p_max": [],
                    "p_std": [],
                    "top1_share": [],
                    "top5_share": [],
                    "top10_share": [],
                }
                for layer_idx in range(ema_distribution.size(0)):
                    p_values = ema_distribution[layer_idx]
                    entropy = -(p_values * torch.log(p_values.clamp_min(self.popularity_balance_eps))).sum()
                    sorted_p = torch.sort(p_values, descending=True).values
                    top5_count = min(5, sorted_p.numel())
                    top10_count = min(10, sorted_p.numel())
                    layer_scalars = {
                        "entropy": float(entropy.cpu()),
                        "effective_tokens": float(torch.exp(entropy).cpu()),
                        "p_max": float(p_values.max().cpu()),
                        "p_std": float(p_values.std(unbiased=False).cpu()),
                        "top1_share": float(sorted_p[:1].sum().cpu()),
                        "top5_share": float(sorted_p[:top5_count].sum().cpu()),
                        "top10_share": float(sorted_p[:top10_count].sum().cpu()),
                    }
                    for name, value in layer_scalars.items():
                        scalars[f"ema/layer_{layer_idx}/{name}"] = value
                        ema_mean_values[name].append(value)
                    histograms[f"ema/layer_{layer_idx}/p"] = p_values.detach().float().cpu()
                for name, values in ema_mean_values.items():
                    if values:
                        scalars[f"ema/mean/{name}"] = float(sum(values) / len(values))

            layer_loss = self._last_popularity_layer_loss
            enabled_mask = self._last_popularity_layer_enabled_mask
            if layer_loss is not None and enabled_mask is not None:
                if layer_loss.size(0) == token_distribution.size(0):
                    enabled_losses: list[float] = []
                    all_losses: list[float] = []
                    for layer_idx in range(layer_loss.size(0)):
                        loss_value = float(layer_loss[layer_idx].detach().cpu())
                        enabled_value = bool(enabled_mask[layer_idx].detach().cpu())
                        scalars[f"ema/layer_{layer_idx}/loss_contribution"] = loss_value
                        scalars[f"ema/layer_{layer_idx}/enabled"] = 1.0 if enabled_value else 0.0
                        scalars[f"ema/layer_{layer_idx}/regularization_weight"] = float(
                            self._layer_weight_tensor(
                                layer_loss.size(0),
                                device=layer_loss.device,
                                dtype=layer_loss.dtype,
                            )[layer_idx]
                            .detach()
                            .cpu()
                        )
                        all_losses.append(loss_value)
                        if enabled_value:
                            enabled_losses.append(loss_value)
                    if all_losses:
                        scalars["ema/mean/loss_contribution_all_layers"] = float(sum(all_losses) / len(all_losses))
                    if enabled_losses:
                        scalars["ema/mean/loss_contribution_enabled_layers"] = float(
                            sum(enabled_losses) / len(enabled_losses)
                        )

            scalars["config/temperature"] = float(self.popularity_softmax_temperature)
            scalars["config/top_k"] = float(self.popularity_balance_top_k)
            scalars["config/nominal_support"] = float(nominal_support)
            scalars["config/nominal_gamma"] = float(self.popularity_nominal_gamma)
            scalars["config/nominal_denominator"] = float(self.popularity_nominal_denominator)
            scalars["config/ema_half_life_items"] = float(self.popularity_ema_half_life_items)
            scalars["config/token_balance_enabled"] = 1.0 if self.token_balance_enabled else 0.0
            scalars["config/prefix_balance_enabled"] = 1.0 if self.popularity_prefix_balance_enabled else 0.0
            if self.popularity_balance_layer_weights:
                for layer_idx, layer_weight in enumerate(self.popularity_balance_layer_weights):
                    scalars[f"config/layer_weight_{layer_idx}"] = float(layer_weight)
            if self.popularity_nominal_clip_value is not None:
                scalars["config/nominal_clip_value"] = float(self.popularity_nominal_clip_value)
            return {"scalars": scalars, "histograms": histograms}

    def compute_loss(self, original_embeddings: torch.Tensor, tokenizer_output: tuple, popularity_weights=None):
        quantized_embeddings, _, commit_loss, distances = tokenizer_output[:4]
        quantization_context = tokenizer_output[4] if len(tokenizer_output) > 4 else {}
        codebooks = quantization_context.get("codebooks") if isinstance(quantization_context, dict) else None
        residuals = quantization_context.get("residuals") if isinstance(quantization_context, dict) else None

        reconstruction_loss = F.mse_loss(quantized_embeddings, original_embeddings)
        if codebooks is None:
            popularity_balance_loss = distances.new_tensor(0.0)
        else:
            popularity_balance_loss = self._compute_popularity_balance_loss(
                distances,
                popularity_weights,
                codebooks,
                residuals=residuals,
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
