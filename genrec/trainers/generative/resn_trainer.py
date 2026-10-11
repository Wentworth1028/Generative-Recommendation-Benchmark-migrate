"""ReSN regularization adapted to TIGER semantic-ID generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.trainers.generative.tiger_trainer import TigerTrainer


def resn_regularizer(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Approximate the squared spectral norm of ``user_emb @ item_emb.T``.

    This is the right-to-left computation used by ReSN. It never materializes
    the user-item score matrix and keeps gradients through both embeddings.
    """
    if user_emb.ndim != 2 or item_emb.ndim != 2:
        raise ValueError("ReSN expects user_emb [n_users, dim] and item_emb [n_items, dim].")
    if user_emb.size(1) != item_emb.size(1):
        raise ValueError(
            f"ReSN embedding dimensions differ: {user_emb.size(1)} != {item_emb.size(1)}."
        )
    if user_emb.size(0) == 0 or item_emb.size(0) == 0:
        raise ValueError("ReSN requires at least one user and one item embedding.")

    compute_dtype = (
        torch.float32
        if user_emb.dtype in (torch.float16, torch.bfloat16)
        or item_emb.dtype in (torch.float16, torch.bfloat16)
        else torch.promote_types(user_emb.dtype, item_emb.dtype)
    )
    users = user_emb.to(dtype=compute_dtype)
    items = item_emb.to(dtype=compute_dtype)

    a = users.sum(dim=0)
    b = items @ a
    c = items.transpose(0, 1) @ b
    z = users @ c

    numerator = z.square().sum()
    denominator = b.square().sum()
    return numerator / (denominator + float(eps))


class ReSNTigerTrainer(TigerTrainer):
    """TIGER trainer with NLL plus ReSN spectral regularization.

    The attached sequential implementation averages model outputs into user
    representations and uses the full item embedding table. TIGER predicts SID
    tokens rather than item IDs, so this adaptation uses decoder hidden states
    as users and pools the corresponding rows of ``lm_head.weight`` for every
    valid item SID. The latter is the embedding table that actually scores
    decoder states, including when TIGER does not tie input/output embeddings.
    """

    def __init__(
        self,
        *args,
        resn_weight: float = 1e-4,
        eps: float = 1e-8,
        item_pooling: str = "mean",
        exclude_eos_from_user: bool = True,
        item2tokens: Optional[Dict] = None,
        **kwargs,
    ):
        self.resn_weight = float(resn_weight)
        self.resn_eps = float(eps)
        self.resn_item_pooling = str(item_pooling).lower()
        self.resn_exclude_eos_from_user = bool(exclude_eos_from_user)
        if self.resn_weight < 0:
            raise ValueError(f"resn_weight must be non-negative, got {self.resn_weight}.")
        if self.resn_eps <= 0:
            raise ValueError(f"eps must be positive, got {self.resn_eps}.")
        if self.resn_item_pooling not in {"mean", "sum"}:
            raise ValueError("item_pooling must be either 'mean' or 'sum'.")

        clean_item2tokens = self._normalize_item2tokens(item2tokens or {})
        model = kwargs.get("model", args[0] if args else None)
        if model is None:
            raise ValueError("ReSNTigerTrainer requires a model.")
        sid_ids, sid_mask, item_ids = self._build_item_sid_cache(
            clean_item2tokens,
            int(model.config.vocab_size),
        )
        model.register_buffer("resn_item_sid_ids", sid_ids, persistent=False)
        model.register_buffer("resn_item_sid_mask", sid_mask, persistent=False)
        model.register_buffer("resn_item_ids", item_ids, persistent=False)

        super().__init__(*args, item2tokens=clean_item2tokens, **kwargs)
        self._write_resn_config(model)
        self.last_resn_components: dict[str, float] = {}

    @staticmethod
    def _normalize_item2tokens(item2tokens: Dict) -> Dict[int, tuple[int, ...]]:
        return {
            int(item_id): tuple(int(token) for token in tokens)
            for item_id, tokens in item2tokens.items()
        }

    @staticmethod
    def _build_item_sid_cache(
        item2tokens: Dict[int, tuple[int, ...]],
        vocab_size: int,
    ) -> tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
        if not item2tokens:
            raise ValueError("ReSN requires a non-empty item2tokens mapping.")

        sorted_items = sorted(item2tokens.items())
        max_sid_length = max(len(tokens) for _, tokens in sorted_items)
        if max_sid_length == 0:
            raise ValueError("ReSN item SIDs must not be empty.")

        sid_ids = torch.zeros((len(sorted_items), max_sid_length), dtype=torch.long)
        sid_mask = torch.zeros_like(sid_ids, dtype=torch.bool)
        item_ids = torch.empty(len(sorted_items), dtype=torch.long)
        for row, (item_id, tokens) in enumerate(sorted_items):
            if not tokens:
                raise ValueError(f"Item {item_id} has an empty SID.")
            if min(tokens) < 0 or max(tokens) >= vocab_size:
                raise ValueError(
                    f"Item {item_id} has SID token outside [0, {vocab_size}): {tokens}."
                )
            length = len(tokens)
            sid_ids[row, :length] = torch.tensor(tokens, dtype=torch.long)
            sid_mask[row, :length] = True
            item_ids[row] = item_id
        return sid_ids, sid_mask, item_ids

    def _write_resn_config(self, model: nn.Module) -> None:
        args = getattr(self, "args", None)
        if args is not None and int(getattr(args, "process_index", 0)) != 0:
            return
        output_dir = Path(getattr(args, "output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "resn_weight": self.resn_weight,
            "eps": self.resn_eps,
            "user_representation": "mean_valid_decoder_hidden",
            "item_representation": f"{self.resn_item_pooling}_lm_head_rows_for_item_sid",
            "exclude_eos_from_user": self.resn_exclude_eos_from_user,
            "num_items": int(model.resn_item_sid_ids.size(0)),
            "max_sid_length": int(model.resn_item_sid_ids.size(1)),
            "uses_full_item_catalog": True,
            "user_scope": "per_process_training_batch",
            "output_temperature": float(getattr(model, "temperature", 1.0)),
        }
        (output_dir / "resn_loss_config.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        return unwrapped.module if hasattr(unwrapped, "module") else unwrapped

    def _item_representations(self, model: nn.Module) -> torch.Tensor:
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ValueError("ReSN requires a model output embedding weight.")

        sid_ids = model.resn_item_sid_ids
        sid_mask = model.resn_item_sid_mask
        token_vectors = F.embedding(sid_ids, output_embeddings.weight)
        item_vectors = (token_vectors * sid_mask.unsqueeze(-1)).sum(dim=1)
        if self.resn_item_pooling == "mean":
            item_vectors = item_vectors / sid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return item_vectors

    def _user_representations(
        self,
        model: nn.Module,
        outputs,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        if not outputs.decoder_hidden_states:
            raise RuntimeError("ReSN requires decoder hidden states from TIGER.")
        decoder_hidden = outputs.decoder_hidden_states[-1]
        valid = labels.ne(-100)
        if self.resn_exclude_eos_from_user and self.eos_token_id is not None:
            valid = valid & labels.ne(int(self.eos_token_id))
        if not valid.any(dim=1).all():
            raise ValueError("Every ReSN sample must have at least one valid non-EOS target token.")

        user_vectors = (decoder_hidden * valid.unsqueeze(-1)).sum(dim=1)
        user_vectors = user_vectors / valid.sum(dim=1, keepdim=True).clamp_min(1)
        if bool(getattr(model.config, "tie_word_embeddings", False)):
            user_vectors = user_vectors * (float(model.model_dim) ** -0.5)
        temperature = float(getattr(model, "temperature", 1.0))
        if temperature <= 0:
            raise ValueError(f"Model output temperature must be positive, got {temperature}.")
        user_vectors = user_vectors / temperature
        return user_vectors

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_mask = inputs.pop("loss_mask", None)
        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits
        if loss_mask is not None:
            logits = logits.masked_fill(loss_mask == 0.0, -1e9)
            rec_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            rec_loss = outputs.loss

        attr_model = self._unwrap_model(model)
        users = self._user_representations(attr_model, outputs, labels)
        items = self._item_representations(attr_model)
        regularizer = resn_regularizer(users, items, eps=self.resn_eps)
        loss = rec_loss + self.resn_weight * regularizer
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite ReSN loss: "
                f"rec={float(rec_loss.detach())}, regularizer={float(regularizer.detach())}."
            )

        self.last_resn_components = {
            "rec_loss": float(rec_loss.detach()),
            "resn_loss": float(regularizer.detach()),
            "weighted_resn_loss": float((self.resn_weight * regularizer).detach()),
            "total_loss": float(loss.detach()),
        }
        outputs.loss = loss
        outputs.loss_rec = rec_loss.detach()
        outputs.loss_resn = regularizer.detach()
        outputs.loss_resn_weighted = (self.resn_weight * regularizer).detach()
        return (loss, outputs) if return_outputs else loss


class ReSNLetterTrainer(ReSNTigerTrainer):
    """LETTER specialization using its temperature-scaled output scores."""
