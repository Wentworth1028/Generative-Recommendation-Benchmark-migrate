from __future__ import annotations

from typing import Optional

from transformers import T5Config

from genrec.models.GHOST.GHOST import GhostLETTER, GhostTIGER


def _build_ghost_config(vocab_size: int, model_config: dict, *, include_tau: bool = False) -> T5Config:
    config = T5Config(
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        d_kv=model_config["d_kv"],
        d_ff=model_config["d_ff"],
        num_layers=model_config["num_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        num_heads=model_config["num_heads"],
        dropout_rate=model_config["dropout_rate"],
        tie_word_embeddings=model_config["tie_word_embeddings"],
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    if include_tau:
        config.tau = float(model_config.get("tau", 1.0))
    config.ghost_variant = str(model_config.get("ghost_variant", "ghost"))
    config.ghost_collection_path = model_config.get("ghost_collection_path")
    config.auo_alpha = float(model_config.get("auo_alpha", 0.1))
    config.auo_ka = int(model_config.get("auo_ka", 200))
    config.auo_kb = int(model_config.get("auo_kb", 5))
    config.auo_temperature = float(model_config.get("auo_temperature", 1.0))
    return config


def create_ghost_tiger_model(vocab_size: int, model_config: dict) -> GhostTIGER:
    config = _build_ghost_config(vocab_size, model_config, include_tau=False)
    return GhostTIGER(config)


def create_ghost_letter_model(vocab_size: int, model_config: dict) -> GhostLETTER:
    config = _build_ghost_config(vocab_size, model_config, include_tau=True)
    return GhostLETTER(config)
