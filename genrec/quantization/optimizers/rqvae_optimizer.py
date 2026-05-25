
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
        self.popularity_weight_transform = self.config.get('popularity_weight_transform', 'log1p')
        self.popularity_balance_eps = float(self.config.get('popularity_balance_eps', 1e-8))
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
    def zero_grad(self):
        self.torch_optimizer.zero_grad()
    
    def _transform_popularity_weights(self, popularity_weights: torch.Tensor):
        popularity_weights = popularity_weights.float().clamp_min(0.0)
        if self.popularity_weight_transform == 'none':
            return popularity_weights
        if self.popularity_weight_transform == 'log1p':
            return torch.log1p(popularity_weights)
        if self.popularity_weight_transform == 'sqrt':
            return torch.sqrt(popularity_weights)
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

    def _compute_popularity_balance_loss(self, distances: torch.Tensor, popularity_weights: torch.Tensor):
        if self.popularity_balance_weight <= 0.0:
            return distances.new_tensor(0.0)
        if distances is None or popularity_weights is None:
            return distances.new_tensor(0.0) if distances is not None else torch.tensor(0.0)

        temperature = max(self.popularity_softmax_temperature, self.popularity_balance_eps)
        soft_assignment = torch.softmax(-distances / temperature, dim=-1)

        weights = self._transform_popularity_weights(popularity_weights).to(
            device=distances.device,
            dtype=distances.dtype,
        )
        weighted_assignment = soft_assignment * weights.view(-1, 1, 1)
        token_mass = weighted_assignment.sum(dim=0)
        token_distribution = token_mass / token_mass.sum(dim=-1, keepdim=True).clamp_min(self.popularity_balance_eps)

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

    def compute_loss(self, original_embeddings: torch.Tensor, tokenizer_output: tuple, popularity_weights=None):
        quantized_embeddings, _, commit_loss, distances = tokenizer_output
        
        reconstruction_loss = F.mse_loss(quantized_embeddings, original_embeddings)
        popularity_balance_loss = self._compute_popularity_balance_loss(distances, popularity_weights)
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
