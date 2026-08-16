import torch
import torch.nn.functional as F
from genrec.quantization.optimizers.base_optimizer import AbstractTokenizerOptimizer


class RQVAETokenizerOptimizer(AbstractTokenizerOptimizer):
    """The unmodified RQ-VAE reconstruction and commitment objective."""

    is_debias_optimizer = False

    def __init__(self, config: dict, tokenizer: torch.nn.Module):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.quant_loss_weight = self.config['quant_loss_weight']

        learning_rate = self.config['learning_rate']
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

        self.torch_optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=learning_rate,
        )

    def zero_grad(self):
        self.torch_optimizer.zero_grad()

    def compute_loss(self, original_embeddings: torch.Tensor, tokenizer_output: tuple, popularity_weights=None):
        quantized_embeddings, _, commit_loss = tokenizer_output[:3]
        reconstruction_loss = F.mse_loss(quantized_embeddings, original_embeddings)
        total_loss = reconstruction_loss + self.quant_loss_weight * commit_loss
        return total_loss, reconstruction_loss, commit_loss

    def step(self):
        self.torch_optimizer.step()

    def move_optimizer_state_to_device(self, device):
        for state in self.torch_optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
