from __future__ import annotations

from typing import Optional

from transformers import T5Config

from genrec.models.GENPLUGIN.GENPLUGIN import GenPluginDualT5


def _build_genplugin_config(vocab_size: int, model_config: dict, *, include_tau: bool = False) -> T5Config:
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
    config.sid_length = int(model_config.get("sid_length", model_config.get("tokens_per_item", 4)))
    config.genplugin_stage = model_config.get("genplugin_stage", "pretrain")
    config.text_embedding_dim = int(model_config.get("text_embedding_dim", 4096))
    config.text_projection_hidden_dim = int(model_config.get("text_projection_hidden_dim", 2048))
    config.kl_temperature = float(model_config.get("kl_temperature", 0.85))
    config.item_temperature = float(model_config.get("item_temperature", 0.9))
    config.kl_weight = float(model_config.get("kl_weight", 0.5))
    config.item_weight = float(model_config.get("item_weight", 0.1))
    config.semantic_substitution_start_epoch = int(model_config.get("semantic_substitution_start_epoch", 10))
    config.semantic_substitution_outer_prob = float(model_config.get("semantic_substitution_outer_prob", 0.5))
    config.semantic_substitution_position_prob = float(model_config.get("semantic_substitution_position_prob", 0.4))
    config.semantic_substitution_top_k = int(model_config.get("semantic_substitution_top_k", 5))
    return config


def create_genplugin_tiger_model(
    vocab_size: int,
    model_config: dict,
    text_embedding_path: Optional[str] = None,
) -> GenPluginDualT5:
    config = _build_genplugin_config(vocab_size, model_config, include_tau=False)
    return GenPluginDualT5(
        config,
        backbone_type="tiger",
        text_embedding_path=text_embedding_path,
        text_embedding_dim=int(model_config.get("text_embedding_dim", 4096)),
        text_projection_hidden_dim=int(model_config.get("text_projection_hidden_dim", 2048)),
        sid_length=int(model_config.get("sid_length", model_config.get("tokens_per_item", 4))),
        kl_temperature=float(model_config.get("kl_temperature", 0.85)),
        item_temperature=float(model_config.get("item_temperature", 0.9)),
        kl_weight=float(model_config.get("kl_weight", 0.5)),
        item_weight=float(model_config.get("item_weight", 0.1)),
        semantic_substitution_start_epoch=int(model_config.get("semantic_substitution_start_epoch", 10)),
        semantic_substitution_outer_prob=float(model_config.get("semantic_substitution_outer_prob", 0.5)),
        semantic_substitution_position_prob=float(model_config.get("semantic_substitution_position_prob", 0.4)),
        semantic_substitution_top_k=int(model_config.get("semantic_substitution_top_k", 5)),
    )


def create_genplugin_letter_model(
    vocab_size: int,
    model_config: dict,
    text_embedding_path: Optional[str] = None,
) -> GenPluginDualT5:
    config = _build_genplugin_config(vocab_size, model_config, include_tau=True)
    return GenPluginDualT5(
        config,
        backbone_type="letter",
        text_embedding_path=text_embedding_path,
        text_embedding_dim=int(model_config.get("text_embedding_dim", 4096)),
        text_projection_hidden_dim=int(model_config.get("text_projection_hidden_dim", 2048)),
        sid_length=int(model_config.get("sid_length", model_config.get("tokens_per_item", 4))),
        kl_temperature=float(model_config.get("kl_temperature", 0.85)),
        item_temperature=float(model_config.get("item_temperature", 0.9)),
        kl_weight=float(model_config.get("kl_weight", 0.5)),
        item_weight=float(model_config.get("item_weight", 0.1)),
        semantic_substitution_start_epoch=int(model_config.get("semantic_substitution_start_epoch", 10)),
        semantic_substitution_outer_prob=float(model_config.get("semantic_substitution_outer_prob", 0.5)),
        semantic_substitution_position_prob=float(model_config.get("semantic_substitution_position_prob", 0.4)),
        semantic_substitution_top_k=int(model_config.get("semantic_substitution_top_k", 5)),
    )

