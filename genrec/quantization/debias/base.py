"""Small shared contract for tokenizer debias optimizers."""

from typing import Protocol


class TokenizerDebiasOptimizer(Protocol):
    """Behavior consumed by the generic RQ-VAE trainer."""

    is_debias_optimizer: bool

    def pop_pending_ema_observation(self): ...

    def apply_popularity_ema_update(self, observation, item_count: int): ...

    def compute_popularity_distribution_metrics(self, tokenizer_output: tuple, popularity_weights=None): ...
