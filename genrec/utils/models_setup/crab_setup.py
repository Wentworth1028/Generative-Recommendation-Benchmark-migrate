from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """A small native LoRA wrapper that keeps the base Linear state key."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.weight = linear.weight
        if linear.bias is not None:
            self.bias = linear.bias
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.rank,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        update = F.linear(F.linear(self.lora_dropout(inputs), self.lora_A), self.lora_B)
        return base + update * self.scaling

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        with torch.no_grad():
            merged.weight.copy_(self.weight + delta.to(dtype=self.weight.dtype))
            if self.bias is not None:
                merged.bias.copy_(self.bias)
        merged.weight.requires_grad_(self.weight.requires_grad)
        if merged.bias is not None and self.bias is not None:
            merged.bias.requires_grad_(self.bias.requires_grad)
        return merged


def _replace_attention_linears(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
) -> list[str]:
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        is_attention_projection = (
            isinstance(child, nn.Linear)
            and not isinstance(child, LoRALinear)
            and child_name in {"q", "k", "v", "o"}
            and ("SelfAttention" in full_name or "EncDecAttention" in full_name)
        )
        if is_attention_projection:
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            replaced.append(full_name)
        else:
            replaced.extend(
                _replace_attention_linears(child, rank, alpha, dropout, full_name)
            )
    return replaced


def inject_attention_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    if rank < 0:
        raise ValueError("LoRA rank must be non-negative.")
    if rank == 0:
        return []
    replaced = _replace_attention_linears(model, rank, alpha, dropout)
    if not replaced:
        raise RuntimeError("No T5 attention q/k/v/o projections were found for CRAB LoRA.")
    return replaced


def merge_attention_lora(module: nn.Module) -> int:
    merged = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, child_name, child.merged_linear())
            merged += 1
        else:
            merged += merge_attention_lora(child)
    return merged


def load_token_initialization(path: str | Path) -> dict[int, int]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"CRAB token initialization manifest does not exist: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = payload.get("new_token_to_source_token")
    if not isinstance(mapping, dict):
        raise ValueError(
            f"CRAB manifest {manifest_path} misses new_token_to_source_token."
        )
    return {int(token): int(source) for token, source in mapping.items()}


def _copy_new_token_rows(
    model: nn.Module,
    old_vocab_size: int,
    new_vocab_size: int,
    mapping: dict[int, int],
) -> None:
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None:
        raise RuntimeError("CRAB requires a model with input token embeddings.")

    for token_id, source_id in mapping.items():
        if not 0 <= source_id < old_vocab_size:
            raise ValueError(
                f"Source token {source_id} is outside pretrained vocabulary [0, {old_vocab_size})."
            )
        if not 0 <= token_id < new_vocab_size:
            raise ValueError(
                f"New token {token_id} is outside resized vocabulary [0, {new_vocab_size})."
            )

    with torch.no_grad():
        for token_id, source_id in mapping.items():
            input_embedding.weight[token_id].copy_(input_embedding.weight[source_id])
            if (
                output_embedding is not None
                and output_embedding.weight.data_ptr() != input_embedding.weight.data_ptr()
            ):
                output_embedding.weight[token_id].copy_(output_embedding.weight[source_id])


def _set_crab_trainable_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    input_embedding.weight.requires_grad_(True)
    if output_embedding is not None:
        output_embedding.weight.requires_grad_(True)

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)


def prepare_crab_model(
    model: nn.Module,
    pretrained_model_path: str | Path,
    token_initialization_path: str | Path,
    vocab_size: int,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the backbone and prepare CRAB's embedding/attention adaptation."""
    pretrained_path = Path(pretrained_model_path)
    if not (pretrained_path / "config.json").is_file():
        raise FileNotFoundError(
            f"CRAB pretrained backbone is not a Hugging Face checkpoint: {pretrained_path}"
        )

    loaded_model = model.__class__.from_pretrained(str(pretrained_path))
    old_vocab_size = int(loaded_model.get_input_embeddings().num_embeddings)
    mapping = load_token_initialization(token_initialization_path)

    loaded_model.resize_token_embeddings(int(vocab_size), mean_resizing=False)
    _copy_new_token_rows(
        loaded_model,
        old_vocab_size=old_vocab_size,
        new_vocab_size=int(vocab_size),
        mapping=mapping,
    )
    lora_modules = inject_attention_lora(
        loaded_model,
        rank=int(lora_rank),
        alpha=float(lora_alpha),
        dropout=float(lora_dropout),
    )
    _set_crab_trainable_parameters(loaded_model)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in loaded_model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in loaded_model.parameters())
    loaded_model.config.crab_lora_rank = int(lora_rank)
    loaded_model.config.crab_lora_alpha = float(lora_alpha)
    loaded_model.config.crab_lora_dropout = float(lora_dropout)
    loaded_model.config.crab_pretrained_vocab_size = old_vocab_size

    report = {
        "pretrained_model_path": str(pretrained_path),
        "token_initialization_path": str(token_initialization_path),
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": int(vocab_size),
        "initialized_new_tokens": len(mapping),
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_module_count": len(lora_modules),
        "lora_modules": lora_modules,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }
    return loaded_model, report
