import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.quantization.data.dataset.rqvae_dataset import ItemEmbeddingDataset
from genrec.trainers.generative.tiger_trainer import TigerTrainer


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class TWMTLTigerTrainer(TigerTrainer):
    """TIGER trainer with Token-Weighted Multi-Target Learning.

    The tokenizer, decoding trie, beam search, and evaluation are inherited from
    TIGER. Only the training loss is replaced by the TWMTL objective.
    """

    def __init__(
        self,
        *args,
        sid_length: int = 4,
        beta: float = 0.99,
        curriculum_c: float = 2e-5,
        eps: float = 1e-8,
        initial_lambda_fg: float = 1.0,
        initial_lambda_fr: float = 1.0,
        initial_lambda_or: float = 1.0,
        repair_duplicate_sids: bool = True,
        item2tokens: Optional[Dict] = None,
        train_dataset=None,
        eval_dataset=None,
        **kwargs,
    ):
        self.sid_length = int(sid_length)
        self.beta = float(beta)
        self.curriculum_c = float(curriculum_c)
        self.twmtl_eps = float(eps)
        self.initial_lambda_fg = float(initial_lambda_fg)
        self.initial_lambda_fr = float(initial_lambda_fr)
        self.initial_lambda_or = float(initial_lambda_or)
        self.repair_duplicate_sids = bool(repair_duplicate_sids)

        clean_item2tokens = self._normalize_item2tokens(item2tokens or {})
        if self.repair_duplicate_sids:
            clean_item2tokens = self._repair_duplicate_sids(clean_item2tokens, kwargs.get("vocab_size"))
        self._apply_item2tokens_to_datasets(clean_item2tokens, train_dataset, eval_dataset)
        model_for_params = kwargs.get("model", args[0] if args else None)
        if model_for_params is not None:
            self._attach_lambda_parameters(model_for_params)

        super().__init__(
            *args,
            item2tokens=clean_item2tokens,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        sid_cache = self._build_sid_cache(clean_item2tokens, self.sid_length)
        freq_counts = self._build_frequency_counts(train_dataset, clean_item2tokens, self.sid_length, self.vocab_size)
        fg_weights, fg_stats = self._build_front_greater_weights(train_dataset, clean_item2tokens, self.sid_length)

        self.model.register_buffer("twmtl_sid_cache", sid_cache, persistent=False)
        self.model.register_buffer("twmtl_freq_counts", freq_counts, persistent=False)
        self.model.register_buffer("twmtl_fg_weights", fg_weights, persistent=False)
        self._write_twmtl_debug(fg_weights, fg_stats, freq_counts)

    def _normalize_item2tokens(self, item2tokens: Dict) -> Dict[int, tuple[int, ...]]:
        return {
            int(item_id): tuple(int(token) for token in tokens)
            for item_id, tokens in item2tokens.items()
            if int(item_id) != 0
        }

    def _repair_duplicate_sids(self, item2tokens: Dict[int, tuple[int, ...]], vocab_size: Optional[int]):
        if not item2tokens or vocab_size is None:
            return item2tokens

        repaired = {item_id: list(tokens) for item_id, tokens in item2tokens.items()}
        tuple_to_items = defaultdict(list)
        for item_id, tokens in repaired.items():
            tuple_to_items[tuple(tokens)].append(item_id)

        duplicate_groups = [items for items in tuple_to_items.values() if len(items) > 1]
        if not duplicate_groups:
            return {item_id: tuple(tokens) for item_id, tokens in repaired.items()}

        used_sids = {tuple(tokens) for tokens in repaired.values()}
        last_position_tokens = {tokens[-1] for tokens in repaired.values()}
        min_last_token = min(last_position_tokens)
        candidate_tokens = [
            token for token in range(min_last_token, int(vocab_size)) if token not in last_position_tokens
        ]
        rng = np.random.default_rng(2026)
        rng.shuffle(candidate_tokens)
        candidate_iter = iter(candidate_tokens)

        for items in duplicate_groups:
            for item_id in sorted(items)[1:]:
                prefix = tuple(repaired[item_id][:-1])
                for new_last_token in candidate_iter:
                    candidate_sid = prefix + (int(new_last_token),)
                    if candidate_sid not in used_sids:
                        repaired[item_id][-1] = int(new_last_token)
                        used_sids.add(candidate_sid)
                        break
        return {item_id: tuple(tokens) for item_id, tokens in repaired.items()}

    def _apply_item2tokens_to_datasets(self, item2tokens, *datasets) -> None:
        tokens2item = {tuple(tokens): item_id for item_id, tokens in item2tokens.items()}
        for dataset in datasets:
            if dataset is not None and hasattr(dataset, "tokenizer"):
                dataset.tokenizer.item2tokens = item2tokens
                dataset.tokenizer.tokens2item = tokens2item

    def _attach_lambda_parameters(self, model: nn.Module) -> None:
        init = {
            "fg": self.initial_lambda_fg,
            "fr": self.initial_lambda_fr,
            "or": self.initial_lambda_or,
        }
        for name, value in init.items():
            param_name = f"twmtl_eta_{name}"
            if not hasattr(model, param_name):
                eta = torch.tensor(inverse_softplus(float(value)), dtype=torch.float32)
                model.register_parameter(param_name, nn.Parameter(eta))

    def _build_sid_cache(self, item2tokens: Dict[int, tuple[int, ...]], sid_length: int) -> torch.LongTensor:
        max_item_id = max(item2tokens.keys(), default=0)
        sid_cache = torch.zeros((max_item_id + 1, sid_length), dtype=torch.long)
        for item_id, tokens in item2tokens.items():
            sid_cache[item_id, : min(sid_length, len(tokens))] = torch.tensor(tokens[:sid_length], dtype=torch.long)
        return sid_cache

    def _build_frequency_counts(
        self,
        train_dataset,
        item2tokens: Dict[int, tuple[int, ...]],
        sid_length: int,
        vocab_size: Optional[int],
    ) -> torch.LongTensor:
        if vocab_size is None:
            vocab_size = max(max(tokens) for tokens in item2tokens.values()) + 1
        counts = torch.zeros((sid_length, int(vocab_size)), dtype=torch.long)
        if train_dataset is None:
            return counts

        for sample in getattr(train_dataset, "samples", []):
            item_id = int(sample["target_item"])
            tokens = item2tokens.get(item_id)
            if tokens is None:
                continue
            for pos, token_id in enumerate(tokens[:sid_length]):
                if 0 <= token_id < vocab_size:
                    counts[pos, token_id] += 1
        return counts

    def _build_front_greater_weights(self, train_dataset, item2tokens, sid_length: int):
        embeddings_by_item = self._load_item_embeddings(train_dataset)
        item_ids = [item_id for item_id in sorted(item2tokens) if item_id in embeddings_by_item]
        if not item_ids:
            return torch.ones(sid_length, dtype=torch.float32), {"fallback": "no_item_embeddings"}

        embeddings = np.asarray([embeddings_by_item[item_id] for item_id in item_ids], dtype=np.float64)
        token_rows = [item2tokens[item_id][:sid_length] for item_id in item_ids]
        mu = []
        for prefix_len in range(sid_length + 1):
            groups = defaultdict(list)
            for row_idx, tokens in enumerate(token_rows):
                key = () if prefix_len == 0 else tuple(tokens[:prefix_len])
                groups[key].append(row_idx)
            mu.append(self._weighted_partition_dispersion(embeddings, groups))

        deltas = np.maximum(np.asarray(mu[:-1]) - np.asarray(mu[1:]), 0.0)
        if float(deltas.sum()) > 0:
            weights = deltas / deltas.sum() * sid_length
        else:
            weights = np.ones(sid_length, dtype=np.float64)
        stats = {
            "mu": [float(x) for x in mu],
            "delta": [float(x) for x in deltas],
            "num_items": len(item_ids),
        }
        return torch.tensor(weights, dtype=torch.float32), stats

    def _load_item_embeddings(self, train_dataset) -> Dict[int, np.ndarray]:
        if train_dataset is None:
            return {}
        tokenizer = getattr(train_dataset, "tokenizer", None)
        tokenizer_config = dict(getattr(tokenizer, "config", {}) or {})
        data_text_files = getattr(train_dataset, "data_text_files", None)
        if not tokenizer_config or data_text_files is None:
            return {}

        tokenizer_config["data_text_files"] = data_text_files
        dataset = ItemEmbeddingDataset(
            data_text_files=data_text_files,
            config=tokenizer_config,
            text_encoder_model=tokenizer_config["text_encoder_model"],
            embedding_extraction_strategy=tokenizer_config.get("embedding_strategy", "mean_pooling"),
            device="cpu",
        )
        return {int(item_id): np.asarray(embedding, dtype=np.float32) for item_id, embedding in dataset.item_embeddings.items()}

    def _weighted_partition_dispersion(self, embeddings: np.ndarray, groups: Dict[tuple, list[int]]) -> float:
        total = embeddings.shape[0]
        weighted = 0.0
        for indices in groups.values():
            group_embeddings = embeddings[indices]
            centroid = group_embeddings.mean(axis=0, keepdims=True)
            dispersion = ((group_embeddings - centroid) ** 2).sum(axis=1).mean()
            weighted += len(indices) / total * dispersion
        return float(weighted)

    def _write_twmtl_debug(self, fg_weights: torch.Tensor, fg_stats: dict, freq_counts: torch.Tensor) -> None:
        if not self.is_world_process_zero():
            return
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sid_length": self.sid_length,
            "beta": self.beta,
            "curriculum_c": self.curriculum_c,
            "front_greater_weights": [float(x) for x in fg_weights.cpu().tolist()],
            "front_greater_stats": fg_stats,
            "frequency_nonzero": int((freq_counts > 0).sum().item()),
        }
        with (output_dir / "twmtl_loss_config.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_mask = inputs.pop("loss_mask", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits
        if loss_mask is not None:
            logits = logits.masked_fill(loss_mask == 0.0, -1e9)

        vocab_size = logits.size(-1)
        ce_all = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        mask_all = labels.ne(-100)
        loss_or = (ce_all * mask_all).sum() / mask_all.sum().clamp_min(1)

        sid_len = min(self.sid_length, labels.size(1))
        sid_labels = labels[:, :sid_len]
        ce_sid = ce_all[:, :sid_len]
        mask_sid = sid_labels.ne(-100)
        if self.eos_token_id is not None:
            mask_sid = mask_sid & sid_labels.ne(int(self.eos_token_id))

        fg_weights = model.twmtl_fg_weights[:sid_len].to(device=logits.device, dtype=ce_sid.dtype)
        loss_fg = (ce_sid * fg_weights.unsqueeze(0) * mask_sid).sum() / mask_sid.sum().clamp_min(1)

        freq_counts = model.twmtl_freq_counts[:sid_len].to(logits.device)
        safe_labels = sid_labels.clamp_min(0).clamp_max(freq_counts.size(1) - 1)
        pos_ids = torch.arange(sid_len, device=logits.device).unsqueeze(0).expand_as(safe_labels)
        counts = freq_counts[pos_ids, safe_labels].clamp_min(1).to(dtype=ce_sid.dtype)
        beta = torch.tensor(self.beta, device=logits.device, dtype=ce_sid.dtype)
        effective_num = (1.0 - torch.pow(beta, counts)) / max(1.0 - self.beta, self.twmtl_eps)
        raw_fr = (1.0 / effective_num.clamp_min(self.twmtl_eps)) * mask_sid
        fr_denom = raw_fr.sum(dim=1, keepdim=True).clamp_min(self.twmtl_eps)
        fr_weights = raw_fr / fr_denom * self.sid_length
        loss_fr = (ce_sid * fr_weights * mask_sid).sum() / mask_sid.sum().clamp_min(1)

        eta_fg = getattr(model, "twmtl_eta_fg")
        eta_fr = getattr(model, "twmtl_eta_fr")
        eta_or = getattr(model, "twmtl_eta_or")
        lambda_fg = F.softplus(eta_fg) + self.twmtl_eps
        lambda_fr = F.softplus(eta_fr) + self.twmtl_eps
        lambda_or = F.softplus(eta_or) + self.twmtl_eps
        step = float(getattr(self.state, "global_step", 0))
        gamma = math.exp(-self.curriculum_c * step)
        scaled_fg = gamma * lambda_fg
        scaled_or = gamma * lambda_or
        scaled_fr = (1.0 - gamma) * lambda_fr
        alpha_sum = scaled_fg + scaled_fr + scaled_or + self.twmtl_eps
        alpha_fg = scaled_fg / alpha_sum
        alpha_fr = scaled_fr / alpha_sum
        alpha_or = scaled_or / alpha_sum

        loss = alpha_fg * loss_fg + alpha_fr * loss_fr + alpha_or * loss_or
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss
