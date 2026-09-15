
import torch
import torch.nn.functional as F
from genrec.quantization.debias.popularity_optimizer import PopularityRQVAETokenizerOptimizer



class LETTERRQVAETokenizerOptimizer(PopularityRQVAETokenizerOptimizer):

    def __init__(self, config: dict, tokenizer: torch.nn.Module):
        super().__init__(config, tokenizer)
        self.cf_alpha = float(self.config['cf_alpha'])

    def CF_loss(self, dense_quantized_rep: torch.Tensor, cf_rep: torch.Tensor):
        batch_size = dense_quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=dense_quantized_rep.device)
        similarities = torch.matmul(dense_quantized_rep, cf_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss
    def compute_loss(self, original_embeddings: torch.Tensor, cf_embeddings: torch.Tensor, tokenizer_output: tuple, popularity_weights=None):
        base_total_loss, reconstruction_loss, commit_loss, popularity_balance_loss = super().compute_loss(
            original_embeddings,
            tokenizer_output,
            popularity_weights=popularity_weights,
        )
        dense_quantized_embeddings = tokenizer_output[5]

        # Compute CF loss
        cf_loss = self.CF_loss(dense_quantized_embeddings, cf_embeddings)

        total_loss = base_total_loss + self.cf_alpha * cf_loss
        return total_loss, reconstruction_loss, commit_loss, cf_loss, popularity_balance_loss
