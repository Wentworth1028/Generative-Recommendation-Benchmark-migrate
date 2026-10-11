"""DROS robust item-level loss adapted to TIGER generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.trainers.generative.tiger_trainer import TigerTrainer


def dros_robust_loss(
    squared_errors: torch.Tensor,
    nominal_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute beta * log(sum_i p_i exp(loss_i / beta)) stably."""
    if squared_errors.shape != nominal_weights.shape:
        raise ValueError(
            "DROS losses and nominal weights must have the same shape, got "
            f"{tuple(squared_errors.shape)} and {tuple(nominal_weights.shape)}."
        )
    if squared_errors.numel() == 0:
        raise ValueError("DROS requires at least one sequence-item pair.")
    if temperature <= 0:
        raise ValueError(f"DROS temperature must be positive, got {temperature}.")
    if bool((nominal_weights < 0).any().detach().item()):
        raise ValueError("DROS nominal weights must be non-negative.")

    compute_dtype = (
        torch.float32
        if squared_errors.dtype in (torch.float16, torch.bfloat16)
        or nominal_weights.dtype in (torch.float16, torch.bfloat16)
        else torch.promote_types(squared_errors.dtype, nominal_weights.dtype)
    )
    losses = squared_errors.to(dtype=compute_dtype).reshape(-1)
    weights = nominal_weights.to(dtype=compute_dtype).reshape(-1)
    positive = weights > 0
    if not bool(positive.any().detach().item()):
        raise ValueError("DROS nominal weights must have a positive sum.")

    losses = losses[positive]
    log_weights = weights[positive].log()
    beta = torch.as_tensor(temperature, dtype=compute_dtype, device=losses.device)
    return beta * (
        torch.logsumexp(log_weights + losses / beta, dim=0)
        - torch.logsumexp(log_weights, dim=0)
    )


class DROSTigerTrainer(TigerTrainer):
    """TIGER NLL plus DROS over sampled item-level positive/negative pairs.

    The source trainer scores users and item-ID embeddings with a dot product.
    TIGER has no item-ID table, so an item is represented by pooling its SID
    rows from ``lm_head.weight`` and a user by pooling valid decoder states.
    This keeps DROS at item level rather than applying it independently to SID
    token frequencies.
    """

    def __init__(
        self,
        *args,
        dros_temperature: float = 0.5,
        dros_weight: float = 0.1,
        popularity_temperature: float = 0.05,
        num_negative_samples: int = 5,
        score_transform: str = "sigmoid",
        popularity_scope: str = "model_window",
        item_pooling: str = "mean",
        exclude_eos_from_user: bool = True,
        item2tokens: Optional[Dict] = None,
        **kwargs,
    ):
        self.dros_temperature = float(dros_temperature)
        self.dros_weight = float(dros_weight)
        self.dros_popularity_temperature = float(popularity_temperature)
        self.dros_num_negative_samples = int(num_negative_samples)
        self.dros_score_transform = str(score_transform).lower()
        self.dros_popularity_scope = str(popularity_scope).lower()
        self.dros_item_pooling = str(item_pooling).lower()
        self.dros_exclude_eos_from_user = bool(exclude_eos_from_user)

        if self.dros_temperature <= 0:
            raise ValueError("dros_temperature must be positive.")
        if self.dros_weight < 0:
            raise ValueError("dros_weight must be non-negative.")
        if self.dros_popularity_temperature < 0:
            raise ValueError("popularity_temperature must be non-negative.")
        if self.dros_num_negative_samples <= 0:
            raise ValueError("num_negative_samples must be positive.")
        if self.dros_score_transform not in {"sigmoid", "identity"}:
            raise ValueError("score_transform must be 'sigmoid' or 'identity'.")
        if self.dros_popularity_scope not in {"model_window", "full_train_split"}:
            raise ValueError("popularity_scope must be 'model_window' or 'full_train_split'.")
        if self.dros_item_pooling not in {"mean", "sum"}:
            raise ValueError("item_pooling must be either 'mean' or 'sum'.")

        clean_item2tokens = self._normalize_item2tokens(item2tokens or {})
        model = kwargs.get("model", args[0] if args else None)
        train_dataset = kwargs.get("train_dataset")
        if model is None:
            raise ValueError("DROSTigerTrainer requires a model.")
        if train_dataset is None:
            raise ValueError("DROSTigerTrainer requires a training dataset for item popularity.")

        sid_ids, sid_mask, item_ids, item_id_to_row = self._build_item_sid_cache(
            clean_item2tokens,
            int(model.config.vocab_size),
        )
        popularity = self._build_item_popularity(train_dataset, item_ids)
        eligible_rows = torch.nonzero(popularity > 0, as_tuple=False).flatten()
        if eligible_rows.numel() < 2:
            raise ValueError("DROS negative sampling requires at least two training items.")

        model.register_buffer("dros_item_sid_ids", sid_ids, persistent=False)
        model.register_buffer("dros_item_sid_mask", sid_mask, persistent=False)
        model.register_buffer("dros_item_ids", item_ids, persistent=False)
        model.register_buffer("dros_item_id_to_row", item_id_to_row, persistent=False)
        model.register_buffer("dros_item_popularity", popularity, persistent=False)
        model.register_buffer("dros_eligible_item_rows", eligible_rows, persistent=False)

        super().__init__(*args, item2tokens=clean_item2tokens, **kwargs)
        self._write_dros_config(model)
        self.last_dros_components: dict[str, float] = {}

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
    ) -> tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor, torch.LongTensor]:
        if not item2tokens:
            raise ValueError("DROS requires a non-empty item2tokens mapping.")

        sorted_items = sorted(item2tokens.items())
        max_sid_length = max(len(tokens) for _, tokens in sorted_items)
        if max_sid_length == 0:
            raise ValueError("DROS item SIDs must not be empty.")

        sid_ids = torch.zeros((len(sorted_items), max_sid_length), dtype=torch.long)
        sid_mask = torch.zeros_like(sid_ids, dtype=torch.bool)
        item_ids = torch.empty(len(sorted_items), dtype=torch.long)
        max_item_id = max(item_id for item_id, _ in sorted_items)
        item_id_to_row = torch.full((max_item_id + 1,), -1, dtype=torch.long)
        for row, (item_id, tokens) in enumerate(sorted_items):
            if item_id < 0:
                raise ValueError(f"DROS item IDs must be non-negative, got {item_id}.")
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
            item_id_to_row[item_id] = row
        return sid_ids, sid_mask, item_ids, item_id_to_row

    def _build_item_popularity(self, train_dataset, item_ids: torch.LongTensor) -> torch.Tensor:
        if not hasattr(train_dataset, "user_seqs"):
            raise ValueError("DROS training dataset must expose user_seqs.")

        counts: Counter[int] = Counter()
        max_items = int(train_dataset.config["max_seq_len"])
        for sequence in train_dataset.user_seqs.values():
            train_sequence = list(sequence[:-2])
            if self.dros_popularity_scope == "model_window":
                train_sequence = train_sequence[-max_items:]
            counts.update(int(item_id) for item_id in train_sequence)

        popularity = torch.tensor(
            [float(counts.get(int(item_id), 0)) for item_id in item_ids.tolist()],
            dtype=torch.float32,
        )
        if float(popularity.sum()) <= 0:
            raise ValueError("DROS found no training interactions for nominal popularity.")
        return popularity

    def _write_dros_config(self, model: nn.Module) -> None:
        args = getattr(self, "args", None)
        if args is not None and int(getattr(args, "process_index", 0)) != 0:
            return
        output_dir = Path(getattr(args, "output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        popularity = model.dros_item_popularity
        payload = {
            "dros_temperature": self.dros_temperature,
            "dros_weight": self.dros_weight,
            "popularity_temperature": self.dros_popularity_temperature,
            "num_negative_samples": self.dros_num_negative_samples,
            "negative_sampling": "uniform_over_train_support_excluding_target_and_history",
            "score_transform": self.dros_score_transform,
            "popularity_scope": self.dros_popularity_scope,
            "nominal_weight": "(item_count / total_count) ** popularity_temperature",
            "positive_negative_reduction": "separate_logsumexp_then_sum",
            "user_representation": "mean_valid_decoder_hidden",
            "item_representation": f"{self.dros_item_pooling}_lm_head_rows_for_item_sid",
            "exclude_eos_from_user": self.dros_exclude_eos_from_user,
            "num_items": int(model.dros_item_ids.numel()),
            "num_items_with_positive_popularity": int((popularity > 0).sum()),
            "train_interaction_count": int(popularity.sum()),
            "uses_item_level_scores": True,
            "output_temperature": float(getattr(model, "temperature", 1.0)),
        }
        (output_dir / "dros_loss_config.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        return unwrapped.module if hasattr(unwrapped, "module") else unwrapped

    def _rows_for_item_ids(self, model: nn.Module, item_ids: torch.LongTensor) -> torch.LongTensor:
        if bool((item_ids < 0).any()) or bool((item_ids >= model.dros_item_id_to_row.numel()).any()):
            raise ValueError("DROS batch contains an item ID outside the tokenizer catalog.")
        rows = model.dros_item_id_to_row[item_ids]
        if bool((rows < 0).any()):
            raise ValueError("DROS batch contains an item ID missing from item2tokens.")
        return rows

    def _item_representations(self, model: nn.Module, rows: torch.LongTensor) -> torch.Tensor:
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ValueError("DROS requires a model output embedding weight.")

        sid_ids = model.dros_item_sid_ids[rows]
        sid_mask = model.dros_item_sid_mask[rows]
        token_vectors = F.embedding(sid_ids, output_embeddings.weight)
        item_vectors = (token_vectors * sid_mask.unsqueeze(-1)).sum(dim=-2)
        if self.dros_item_pooling == "mean":
            item_vectors = item_vectors / sid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return item_vectors

    def _user_representations(
        self,
        model: nn.Module,
        outputs,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        if not outputs.decoder_hidden_states:
            raise RuntimeError("DROS requires decoder hidden states from TIGER.")
        decoder_hidden = outputs.decoder_hidden_states[-1]
        valid = labels.ne(-100)
        if self.dros_exclude_eos_from_user and self.eos_token_id is not None:
            valid = valid & labels.ne(int(self.eos_token_id))
        if not valid.any(dim=1).all():
            raise ValueError("Every DROS sample must have at least one valid non-EOS target token.")

        user_vectors = (decoder_hidden * valid.unsqueeze(-1)).sum(dim=1)
        user_vectors = user_vectors / valid.sum(dim=1, keepdim=True).clamp_min(1)
        if bool(getattr(model.config, "tie_word_embeddings", False)):
            user_vectors = user_vectors * (float(model.model_dim) ** -0.5)
        temperature = float(getattr(model, "temperature", 1.0))
        if temperature <= 0:
            raise ValueError(f"Model output temperature must be positive, got {temperature}.")
        user_vectors = user_vectors / temperature
        return user_vectors

    def _sample_negative_rows(
        self,
        model: nn.Module,
        positive_rows: torch.LongTensor,
        history_item_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        eligible = model.dros_eligible_item_rows
        positive_matches = eligible.unsqueeze(0).eq(positive_rows.unsqueeze(1))
        if not bool(positive_matches.any(dim=1).all()):
            raise ValueError("DROS positive targets must have positive training popularity.")

        sample_shape = (positive_rows.size(0), self.dros_num_negative_samples)
        draws = torch.randint(
            low=0,
            high=eligible.numel(),
            size=sample_shape,
            device=positive_rows.device,
        )
        negative_rows = eligible[draws]

        valid_history = history_item_ids.ge(0)
        safe_history_ids = history_item_ids.clamp_min(0)
        in_catalog = safe_history_ids.lt(model.dros_item_id_to_row.numel())
        safe_history_ids = safe_history_ids.clamp_max(model.dros_item_id_to_row.numel() - 1)
        history_rows = model.dros_item_id_to_row[safe_history_ids]
        valid_history = valid_history & in_catalog & history_rows.ge(0)

        for _ in range(100):
            in_history = (
                negative_rows.unsqueeze(-1).eq(history_rows.unsqueeze(1))
                & valid_history.unsqueeze(1)
            ).any(dim=-1)
            invalid = negative_rows.eq(positive_rows.unsqueeze(1)) | in_history
            if not bool(invalid.any()):
                return negative_rows
            redraws = torch.randint(
                low=0,
                high=eligible.numel(),
                size=(int(invalid.sum().item()),),
                device=positive_rows.device,
            )
            negative_rows[invalid] = eligible[redraws]
        raise RuntimeError("DROS negative sampling could not avoid target/history items.")

    def _score_transform(self, scores: torch.Tensor) -> torch.Tensor:
        if self.dros_score_transform == "sigmoid":
            return torch.sigmoid(scores)
        return scores

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_mask = inputs.pop("loss_mask", None)
        history_item_ids = inputs.pop("history_item_ids", None)
        labels = inputs.get("labels")
        target_item_ids = inputs.get("label_id")
        outputs = model(
            **inputs,
            output_hidden_states=self.dros_weight > 0.0,
            use_cache=False,
        )
        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss
        if target_item_ids is None:
            raise ValueError("DROS requires label_id target item IDs from TigerDataCollator.")
        if history_item_ids is None:
            raise ValueError("DROS requires history_item_ids from DROSTigerDataCollator.")

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

        if self.dros_weight == 0.0:
            robust_positive = rec_loss.new_zeros(())
            robust_negative = rec_loss.new_zeros(())
        else:
            attr_model = self._unwrap_model(model)
            positive_rows = self._rows_for_item_ids(attr_model, target_item_ids.long())
            negative_rows = self._sample_negative_rows(
                attr_model,
                positive_rows,
                history_item_ids.long(),
            )

            users = self._user_representations(attr_model, outputs, labels)
            positive_items = self._item_representations(attr_model, positive_rows)
            negative_items = self._item_representations(attr_model, negative_rows)
            positive_scores = (users * positive_items).sum(dim=-1)
            negative_scores = torch.einsum("bd,bnd->bn", users, negative_items)
            positive_predictions = self._score_transform(positive_scores)
            negative_predictions = self._score_transform(negative_scores)

            positive_errors = (1.0 - positive_predictions).square()
            negative_errors = negative_predictions.square()
            popularity = attr_model.dros_item_popularity
            total_popularity = popularity.sum()
            positive_weights = (
                popularity[positive_rows] / total_popularity
            ).pow(self.dros_popularity_temperature)
            negative_weights = (
                popularity[negative_rows] / total_popularity
            ).pow(self.dros_popularity_temperature)

            robust_positive = dros_robust_loss(
                positive_errors,
                positive_weights,
                self.dros_temperature,
            )
            robust_negative = dros_robust_loss(
                negative_errors,
                negative_weights,
                self.dros_temperature,
            )

        robust_loss = robust_positive + robust_negative
        weighted_robust = self.dros_weight * robust_loss
        loss = rec_loss + weighted_robust
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite DROS loss: "
                f"rec={float(rec_loss.detach())}, robust={float(robust_loss.detach())}."
            )

        self.last_dros_components = {
            "rec_loss": float(rec_loss.detach()),
            "dro_positive_loss": float(robust_positive.detach()),
            "dro_negative_loss": float(robust_negative.detach()),
            "dro_loss": float(robust_loss.detach()),
            "weighted_dro_loss": float(weighted_robust.detach()),
            "total_loss": float(loss.detach()),
        }
        outputs.loss = loss
        outputs.loss_rec = rec_loss.detach()
        outputs.loss_dros = robust_loss.detach()
        outputs.loss_dros_weighted = weighted_robust.detach()
        return (loss, outputs) if return_outputs else loss


class DROSLetterTrainer(DROSTigerTrainer):
    """LETTER specialization using its temperature-scaled output scores."""
