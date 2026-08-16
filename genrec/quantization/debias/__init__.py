"""Tokenizer debias objectives.

This package is separate from the baseline RQ-VAE optimizer so that new
debias methods do not require editing the standard tokenizer path.
"""

from .popularity_optimizer import PopularityRQVAETokenizerOptimizer
from .trainer import PopularityDebiasController

__all__ = ["PopularityDebiasController", "PopularityRQVAETokenizerOptimizer"]
