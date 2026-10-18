"""SPRINT regularization adapted to TIGER's T5 encoder-decoder backbone.

The implementation follows the supplied GenRec SASRec-SPRINT code while
leaving TIGER's tokenizer, model, collator, and constrained decoding intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.trainers.generative.tiger_trainer import TigerTrainer


ATTENTION_SCOPES = frozenset({"encoder_self", "decoder_self", "decoder_cross"})
FFN_SCOPES = frozenset({"encoder", "decoder"})


def _register_power_iteration_buffers(
    module: nn.Module,
    weight: torch.Tensor,
    name: str,
) -> None:
    if weight.ndim != 2:
        raise ValueError(f"SPRINT spectral norm expects a matrix, got {tuple(weight.shape)}.")
    rows, columns = weight.shape
    u_name, v_name = f"{name}_u", f"{name}_v"
    if not hasattr(module, u_name):
        u = F.normalize(torch.randn(rows, dtype=torch.float32), dim=0)
        module.register_buffer(u_name, u, persistent=False)
    if not hasattr(module, v_name):
        v = F.normalize(torch.randn(columns, dtype=torch.float32), dim=0)
        module.register_buffer(v_name, v, persistent=False)


def sprint_power_iteration(
    module: nn.Module,
    weight: torch.Tensor,
    *,
    name: str,
    num_iterations: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate a matrix spectral norm while retaining gradients to ``weight``."""
    if num_iterations < 1:
        raise ValueError("SPRINT num_iterations must be at least one.")
    if eps <= 0:
        raise ValueError("SPRINT eps must be positive.")
    _register_power_iteration_buffers(module, weight, name)

    compute_weight = weight.float() if weight.dtype in (torch.float16, torch.bfloat16) else weight
    u_buffer = getattr(module, f"{name}_u")
    v_buffer = getattr(module, f"{name}_v")
    u = u_buffer.to(device=weight.device, dtype=compute_weight.dtype).detach().clone()
    v = v_buffer.to(device=weight.device, dtype=compute_weight.dtype).detach().clone()

    with torch.no_grad():
        detached = compute_weight.detach()
        for _ in range(num_iterations):
            u = F.normalize(detached @ v, dim=0, eps=eps)
            v = F.normalize(detached.transpose(0, 1) @ u, dim=0, eps=eps)
        u_buffer.copy_(u.to(device=u_buffer.device, dtype=u_buffer.dtype))
        v_buffer.copy_(v.to(device=v_buffer.device, dtype=v_buffer.dtype))

    return torch.linalg.vector_norm(compute_weight @ v.detach())


def sprint_attention_column_proxy(
    attention: torch.Tensor,
    *,
    temperature: float,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth the maximum attention-column sum independently for each head.

    ``attention`` is ``[batch, heads, query_length, key_length]``. Flattening
    batch and key positions is equivalent to applying the maximum-column bound
    to the block-diagonal attention matrix formed by all user sequences.
    """
    if attention.ndim != 4:
        raise ValueError("SPRINT attention proxy expects [batch, heads, query, key], " f"got {tuple(attention.shape)}.")
    if temperature <= 0:
        raise ValueError("SPRINT attention temperature must be positive.")

    batch_size, _, query_length, key_length = attention.shape
    weights = attention.abs()
    if query_mask is not None:
        if tuple(query_mask.shape) != (batch_size, query_length):
            raise ValueError(
                f"SPRINT query mask {tuple(query_mask.shape)} does not match "
                f"attention queries {(batch_size, query_length)}."
            )
        weights = weights * query_mask[:, None, :, None].to(weights.dtype)

    column_sums = weights.sum(dim=-2)
    if key_mask is None:
        valid_keys = torch.ones((batch_size, key_length), device=attention.device, dtype=torch.bool)
    else:
        if tuple(key_mask.shape) != (batch_size, key_length):
            raise ValueError(
                f"SPRINT key mask {tuple(key_mask.shape)} does not match " f"attention keys {(batch_size, key_length)}."
            )
        valid_keys = key_mask.to(device=attention.device, dtype=torch.bool)
    if not bool(valid_keys.any()):
        raise ValueError("SPRINT attention proxy received no valid key positions.")

    flattened = column_sums.permute(1, 0, 2).flatten(start_dim=1)
    flattened = flattened.masked_fill(~valid_keys.flatten().unsqueeze(0), -torch.inf)
    return torch.logsumexp(flattened * float(temperature), dim=-1) / float(temperature)


class SPRINTTigerTrainer(TigerTrainer):
    """TIGER NLL plus attention/projection SPRINT regularization."""

    sprint_backbone_name = "TIGER"
    sprint_adaptation = "GenRec SASRec-SPRINT to TIGER T5 encoder-decoder"

    def __init__(
        self,
        *args,
        sprint_attention_weight: float = 1.0,
        sprint_ffn_weight: float = 1e-3,
        sprint_embedding_weight: float = 0.1,
        sprint_attention_temperature: float = 4.0,
        sprint_spectral_norm_iters: int = 1,
        sprint_eps: float = 1e-12,
        sprint_log_transform: str = "log1p",
        sprint_layer_reduction: str = "mean",
        sprint_attention_scopes: Sequence[str] = (
            "encoder_self",
            "decoder_self",
            "decoder_cross",
        ),
        sprint_ffn_scopes: Sequence[str] = ("encoder", "decoder"),
        **kwargs,
    ) -> None:
        self.sprint_attention_weight = float(sprint_attention_weight)
        self.sprint_ffn_weight = float(sprint_ffn_weight)
        self.sprint_embedding_weight = float(sprint_embedding_weight)
        self.sprint_attention_temperature = float(sprint_attention_temperature)
        self.sprint_spectral_norm_iters = int(sprint_spectral_norm_iters)
        self.sprint_eps = float(sprint_eps)
        self.sprint_log_transform = str(sprint_log_transform).lower()
        self.sprint_layer_reduction = str(sprint_layer_reduction).lower()
        self.sprint_attention_scopes = tuple(str(scope) for scope in sprint_attention_scopes)
        self.sprint_ffn_scopes = tuple(str(scope) for scope in sprint_ffn_scopes)
        self._validate_config()

        model = kwargs.get("model", args[0] if args else None)
        if model is None:
            raise ValueError(f"SPRINT {self.sprint_backbone_name} trainer requires a model.")
        self._initialize_power_iteration_buffers(model)
        super().__init__(*args, **kwargs)
        self.last_sprint_components: dict[str, float] = {}
        self._write_sprint_config(model)

    def _validate_config(self) -> None:
        for name, value in (
            ("attention_weight", self.sprint_attention_weight),
            ("ffn_weight", self.sprint_ffn_weight),
            ("embedding_weight", self.sprint_embedding_weight),
        ):
            if value < 0:
                raise ValueError(f"SPRINT {name} must be non-negative, got {value}.")
        if self.sprint_attention_temperature <= 0:
            raise ValueError("SPRINT attention temperature must be positive.")
        if self.sprint_spectral_norm_iters < 1:
            raise ValueError("SPRINT spectral_norm_iters must be at least one.")
        if self.sprint_eps <= 0:
            raise ValueError("SPRINT eps must be positive.")
        if self.sprint_log_transform not in {"log", "log1p"}:
            raise ValueError("SPRINT log_transform must be 'log' or 'log1p'.")
        if self.sprint_layer_reduction not in {"sum", "mean"}:
            raise ValueError("SPRINT layer_reduction must be 'sum' or 'mean'.")
        unknown_attention = set(self.sprint_attention_scopes) - ATTENTION_SCOPES
        unknown_ffn = set(self.sprint_ffn_scopes) - FFN_SCOPES
        if unknown_attention:
            raise ValueError(f"Unknown SPRINT attention scopes: {sorted(unknown_attention)}.")
        if unknown_ffn:
            raise ValueError(f"Unknown SPRINT FFN scopes: {sorted(unknown_ffn)}.")

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        while hasattr(model, "module"):
            model = model.module
        return model

    @staticmethod
    def _attention_module(block: nn.Module, scope: str) -> nn.Module:
        if scope in {"encoder_self", "decoder_self"}:
            return block.layer[0].SelfAttention
        if scope == "decoder_cross":
            return block.layer[1].EncDecAttention
        raise ValueError(f"Unsupported attention scope: {scope}.")

    @staticmethod
    def _ffn_module(block: nn.Module) -> nn.Module:
        return block.layer[-1].DenseReluDense

    @staticmethod
    def _ffn_input_projections(ffn: nn.Module) -> list[nn.Linear]:
        if hasattr(ffn, "wi"):
            return [ffn.wi]
        inputs = [projection for name in ("wi_0", "wi_1") if (projection := getattr(ffn, name, None)) is not None]
        if not inputs:
            raise TypeError(f"Unsupported T5 FFN module: {type(ffn).__name__}.")
        return inputs

    def _iter_attention_modules(self, model: nn.Module) -> Iterable[tuple[str, int, nn.Module]]:
        if self.sprint_attention_weight > 0 and "encoder_self" in self.sprint_attention_scopes:
            for index, block in enumerate(model.encoder.block):
                yield "encoder_self", index, self._attention_module(block, "encoder_self")
        if self.sprint_attention_weight > 0 and "decoder_self" in self.sprint_attention_scopes:
            for index, block in enumerate(model.decoder.block):
                yield "decoder_self", index, self._attention_module(block, "decoder_self")
        if self.sprint_attention_weight > 0 and "decoder_cross" in self.sprint_attention_scopes:
            for index, block in enumerate(model.decoder.block):
                yield "decoder_cross", index, self._attention_module(block, "decoder_cross")

    def _iter_ffn_modules(self, model: nn.Module) -> Iterable[tuple[str, int, nn.Module]]:
        if "encoder" in self.sprint_ffn_scopes:
            for index, block in enumerate(model.encoder.block):
                yield "encoder", index, self._ffn_module(block)
        if "decoder" in self.sprint_ffn_scopes:
            for index, block in enumerate(model.decoder.block):
                yield "decoder", index, self._ffn_module(block)

    def _initialize_power_iteration_buffers(self, model: nn.Module) -> None:
        model = self._unwrap_model(model)
        if self.sprint_attention_weight > 0:
            for _, _, attention in self._iter_attention_modules(model):
                _register_power_iteration_buffers(attention.o, attention.o.weight, "sprint")
                head_dim = int(attention.key_value_proj_dim)
                for head in range(int(attention.n_heads)):
                    start, end = head * head_dim, (head + 1) * head_dim
                    _register_power_iteration_buffers(
                        attention,
                        attention.v.weight[start:end],
                        f"sprint_v_head_{head}",
                    )
        if self.sprint_ffn_weight > 0:
            for _, _, ffn in self._iter_ffn_modules(model):
                for projection in [*self._ffn_input_projections(ffn), ffn.wo]:
                    _register_power_iteration_buffers(projection, projection.weight, "sprint")
        if self.sprint_embedding_weight > 0:
            output_embedding = model.get_output_embeddings()
            if output_embedding is None or not hasattr(output_embedding, "weight"):
                raise ValueError(f"SPRINT requires {self.sprint_backbone_name}'s output embedding matrix.")
            _register_power_iteration_buffers(output_embedding, output_embedding.weight, "sprint")

    def _transform(self, value: torch.Tensor) -> torch.Tensor:
        if self.sprint_log_transform == "log1p":
            return torch.log1p(value.clamp_min(0.0))
        return torch.log(value.clamp_min(self.sprint_eps))

    def _reduce(self, components: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
        if not components:
            return zero
        stacked = torch.stack(components)
        return stacked.mean() if self.sprint_layer_reduction == "mean" else stacked.sum()

    def _projection_norm(self, module: nn.Module, weight: torch.Tensor, name: str = "sprint") -> torch.Tensor:
        return sprint_power_iteration(
            module,
            weight,
            name=name,
            num_iterations=self.sprint_spectral_norm_iters,
            eps=self.sprint_eps,
        )

    def _attention_component(
        self,
        attention_module: nn.Module,
        attention_weights: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        column_proxy = sprint_attention_column_proxy(
            attention_weights,
            temperature=self.sprint_attention_temperature,
            query_mask=query_mask,
            key_mask=key_mask,
        )
        head_dim = int(attention_module.key_value_proj_dim)
        value_norms = []
        for head in range(int(attention_module.n_heads)):
            start, end = head * head_dim, (head + 1) * head_dim
            value_norms.append(
                self._projection_norm(
                    attention_module,
                    attention_module.v.weight[start:end],
                    name=f"sprint_v_head_{head}",
                )
            )
        value_norms_tensor = torch.stack(value_norms)
        output_norm = self._projection_norm(attention_module.o, attention_module.o.weight)
        combined = (
            torch.sqrt((column_proxy * value_norms_tensor.square()).sum().clamp_min(self.sprint_eps)) * output_norm
        )
        return self._transform(combined)

    def _ffn_component(self, ffn: nn.Module) -> torch.Tensor:
        norms = [
            self._projection_norm(projection, projection.weight)
            for projection in [*self._ffn_input_projections(ffn), ffn.wo]
        ]
        product = torch.stack(norms).prod()
        return self._transform(product)

    def _decoder_mask(self, inputs: dict, labels: torch.Tensor) -> torch.Tensor:
        decoder_mask = inputs.get("decoder_attention_mask")
        if decoder_mask is not None:
            return decoder_mask.to(dtype=torch.bool)
        return labels.ne(-100)

    def _regularizers(self, model: nn.Module, outputs, inputs: dict, labels: torch.Tensor):
        zero = outputs.logits.new_zeros(())
        source_mask = inputs["attention_mask"].to(dtype=torch.bool)
        target_mask = self._decoder_mask(inputs, labels)
        attention_components: list[torch.Tensor] = []

        if self.sprint_attention_weight > 0 and "encoder_self" in self.sprint_attention_scopes:
            if outputs.encoder_attentions is None:
                raise RuntimeError(f"SPRINT requires {self.sprint_backbone_name} encoder attentions.")
            for block, weights in zip(model.encoder.block, outputs.encoder_attentions):
                attention_components.append(
                    self._attention_component(
                        self._attention_module(block, "encoder_self"),
                        weights,
                        source_mask,
                        source_mask,
                    )
                )
        if self.sprint_attention_weight > 0 and "decoder_self" in self.sprint_attention_scopes:
            if outputs.decoder_attentions is None:
                raise RuntimeError(f"SPRINT requires {self.sprint_backbone_name} decoder attentions.")
            for block, weights in zip(model.decoder.block, outputs.decoder_attentions):
                attention_components.append(
                    self._attention_component(
                        self._attention_module(block, "decoder_self"),
                        weights,
                        target_mask,
                        target_mask,
                    )
                )
        if self.sprint_attention_weight > 0 and "decoder_cross" in self.sprint_attention_scopes:
            if outputs.cross_attentions is None:
                raise RuntimeError(f"SPRINT requires {self.sprint_backbone_name} cross-attentions.")
            for block, weights in zip(model.decoder.block, outputs.cross_attentions):
                attention_components.append(
                    self._attention_component(
                        self._attention_module(block, "decoder_cross"),
                        weights,
                        target_mask,
                        source_mask,
                    )
                )

        ffn_components = (
            [self._ffn_component(ffn) for _, _, ffn in self._iter_ffn_modules(model)]
            if self.sprint_ffn_weight > 0
            else []
        )
        attention_loss = self._reduce(attention_components, zero)
        ffn_loss = self._reduce(ffn_components, zero)
        embedding_loss = zero
        if self.sprint_embedding_weight > 0:
            output_embedding = model.get_output_embeddings()
            embedding_loss = self._transform(self._projection_norm(output_embedding, output_embedding.weight))
        return attention_loss, ffn_loss, embedding_loss

    def _write_sprint_config(self, model: nn.Module) -> None:
        args = getattr(self, "args", None)
        if args is not None and int(getattr(args, "process_index", 0)) != 0:
            return
        output_dir = Path(getattr(args, "output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "adaptation": self.sprint_adaptation,
            "backbone": self.sprint_backbone_name,
            "attention_weight": self.sprint_attention_weight,
            "ffn_weight": self.sprint_ffn_weight,
            "embedding_weight": self.sprint_embedding_weight,
            "attention_temperature": self.sprint_attention_temperature,
            "spectral_norm_iters": self.sprint_spectral_norm_iters,
            "eps": self.sprint_eps,
            "log_transform": self.sprint_log_transform,
            "layer_reduction": self.sprint_layer_reduction,
            "attention_scopes": list(self.sprint_attention_scopes),
            "ffn_scopes": list(self.sprint_ffn_scopes),
            "attention_proxy": "per-head logsumexp of valid attention column sums",
            "attention_projection": "sqrt(sum_h(proxy_h * ||Wv_h||_2^2)) * ||Wo||_2",
            "ffn_projection": "product of T5 input/output projection spectral norms",
            "embedding_matrix": "lm_head.weight",
            "reference_defaults": {
                "attention_weight": 1.0,
                "ffn_weight": 0.001,
                "embedding_weight": 0.1,
                "attention_temperature": 4.0,
                "spectral_norm_iters": 1,
                "log_transform": "log1p",
                "layer_reduction": "mean",
            },
            "paper_summary_alternative": {
                "embedding_weight": 0.0,
                "attention_temperature": 1.0,
                "log_transform": "log",
                "layer_reduction": "sum",
            },
        }
        (output_dir / "sprint_loss_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_mask = inputs.pop("loss_mask", None)
        labels = inputs.get("labels")
        enabled = any(
            weight > 0
            for weight in (
                self.sprint_attention_weight,
                self.sprint_ffn_weight,
                self.sprint_embedding_weight,
            )
        )
        need_attentions = self.sprint_attention_weight > 0 and bool(self.sprint_attention_scopes)
        outputs = model(
            **inputs,
            output_attentions=need_attentions,
            use_cache=False if enabled else None,
        )
        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        if loss_mask is not None:
            masked_logits = outputs.logits.masked_fill(loss_mask == 0.0, -1e9)
            rec_loss = F.cross_entropy(
                masked_logits.reshape(-1, masked_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            rec_loss = outputs.loss

        if not enabled:
            return (rec_loss, outputs) if return_outputs else rec_loss

        attr_model = self._unwrap_model(model)
        attention_loss, ffn_loss, embedding_loss = self._regularizers(attr_model, outputs, inputs, labels)
        weighted_attention = self.sprint_attention_weight * attention_loss
        weighted_ffn = self.sprint_ffn_weight * ffn_loss
        weighted_embedding = self.sprint_embedding_weight * embedding_loss
        loss = rec_loss + weighted_attention + weighted_ffn + weighted_embedding
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite SPRINT loss: "
                f"rec={float(rec_loss.detach())}, attention={float(attention_loss.detach())}, "
                f"ffn={float(ffn_loss.detach())}, embedding={float(embedding_loss.detach())}."
            )

        self.last_sprint_components = {
            "rec_loss": float(rec_loss.detach()),
            "attention_loss": float(attention_loss.detach()),
            "ffn_loss": float(ffn_loss.detach()),
            "embedding_loss": float(embedding_loss.detach()),
            "weighted_attention_loss": float(weighted_attention.detach()),
            "weighted_ffn_loss": float(weighted_ffn.detach()),
            "weighted_embedding_loss": float(weighted_embedding.detach()),
            "total_loss": float(loss.detach()),
        }
        outputs.loss = loss
        outputs.loss_rec = rec_loss.detach()
        outputs.loss_sprint_attention = attention_loss.detach()
        outputs.loss_sprint_ffn = ffn_loss.detach()
        outputs.loss_sprint_embedding = embedding_loss.detach()
        return (loss, outputs) if return_outputs else loss


class SPRINTLetterTrainer(SPRINTTigerTrainer):
    """LETTER NLL plus SPRINT, preserving LETTER logit temperature scaling.

    LETTER uses the same T5 encoder-decoder modules as TIGER, so the SPRINT
    regularizers are shared. Its tokenizer, collaborative signal, model
    factory, and ``logits / tau`` forward behavior remain LETTER-specific.
    """

    sprint_backbone_name = "LETTER"
    sprint_adaptation = "GenRec SASRec-SPRINT to LETTER T5 encoder-decoder"
