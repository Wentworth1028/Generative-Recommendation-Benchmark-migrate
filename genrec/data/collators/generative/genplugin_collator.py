from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
import numpy as np

from genrec.genplugin.data_utils import load_json


@dataclass
class GenPluginDataCollator:
    max_item_seq_len: int
    sid_length: int
    pad_token_id: int
    eos_token_id: int
    text_pad_token_id: int
    label_pad_token_id: int = -100
    stage: str = "pretrain"
    retrieval_top_k: int = 30
    user_embedding_path: Optional[Any] = None
    reranked_user_index_path: Optional[Any] = None

    def __post_init__(self) -> None:
        self.stage = str(self.stage).lower()
        self.user_embedding_cache = {}
        self.reranked_user_index = {}
        if self.user_embedding_path:
            self.user_embedding_cache = self._load_split_mapping(self.user_embedding_path, expect_tensor=True)
        if self.reranked_user_index_path:
            self.reranked_user_index = self._load_split_mapping(self.reranked_user_index_path, expect_tensor=False)

    def _load_split_mapping(self, value, expect_tensor: bool):
        if isinstance(value, dict):
            result = {}
            for split, path in value.items():
                if path is None:
                    continue
                result[str(split)] = self._load_single_path(path, expect_tensor=expect_tensor)
            return result
        return {"default": self._load_single_path(value, expect_tensor=expect_tensor)}

    def _load_single_path(self, value, expect_tensor: bool):
        path = Path(value)
        if expect_tensor:
            if path.suffix == ".npy":
                loaded = np.load(path, allow_pickle=True)
            else:
                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, dict) and "embeddings" in loaded:
                    loaded = loaded["embeddings"]
            return torch.as_tensor(loaded, dtype=torch.float32)
        return load_json(path)

    def _pad_left(self, values: List[int], target_len: int, pad_value: int) -> List[int]:
        values = values[-target_len:]
        padding = target_len - len(values)
        if padding > 0:
            values = [pad_value] * padding + values
        return values

    def _pad_right(self, values: List[int], target_len: int, pad_value: int) -> List[int]:
        values = values[:target_len]
        padding = target_len - len(values)
        if padding > 0:
            values = values + [pad_value] * padding
        return values

    def _resolve_split(self, feature: Dict[str, Any]) -> str:
        return str(feature.get("split", "default"))

    def _build_retrieval_tensor(self, sample_index: int, split: str) -> Optional[torch.Tensor]:
        if split not in self.user_embedding_cache or split not in self.reranked_user_index:
            return None
        user_embedding_cache = self.user_embedding_cache[split]
        reranked_user_index = self.reranked_user_index[split]
        if sample_index >= len(reranked_user_index):
            return None
        candidate_indices = [int(index) for index in reranked_user_index[sample_index][: self.retrieval_top_k]]
        if not candidate_indices:
            return torch.zeros(
                self.retrieval_top_k,
                user_embedding_cache.size(-1),
                dtype=user_embedding_cache.dtype,
            )
        candidate_tensor = user_embedding_cache.index_select(0, torch.as_tensor(candidate_indices, dtype=torch.long))
        if candidate_tensor.size(0) < self.retrieval_top_k:
            padding = torch.zeros(
                self.retrieval_top_k - candidate_tensor.size(0),
                candidate_tensor.size(1),
                dtype=candidate_tensor.dtype,
            )
            candidate_tensor = torch.cat([candidate_tensor, padding], dim=0)
        return candidate_tensor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_id_len = self.max_item_seq_len * self.sid_length + 1
        max_text_len = self.max_item_seq_len + 1

        id_input_ids = []
        id_attention_mask = []
        text_input_ids = []
        text_attention_mask = []
        labels = []
        sample_indices = []
        item_ids = []
        user_ids = []
        user_rag_embs = []

        for feature in features:
            source_tokens = list(feature["source_tokens"])
            source_tokens = self._pad_left(source_tokens, self.max_item_seq_len * self.sid_length, self.pad_token_id)
            source_tokens = source_tokens + [self.pad_token_id]
            id_input_ids.append(source_tokens)
            id_attention_mask.append([1 if token != self.pad_token_id else 0 for token in source_tokens])

            history_items = list(feature["text_input_ids"])
            history_items = self._pad_left(history_items, self.max_item_seq_len, self.text_pad_token_id)
            history_items = history_items + [self.text_pad_token_id]
            text_input_ids.append(history_items)
            text_attention_mask.append([1 if token != self.text_pad_token_id else 0 for token in history_items])

            target_tokens = list(feature["target_tokens"]) + [self.eos_token_id]
            labels.append(target_tokens)

            sample_index = int(feature["index"])
            split = self._resolve_split(feature)
            sample_indices.append(sample_index)
            item_ids.append(int(feature["item_id"]))
            user_ids.append(int(feature.get("user_id", -1)))

            retrieval_tensor = self._build_retrieval_tensor(sample_index, split)
            if retrieval_tensor is None:
                if split in self.user_embedding_cache:
                    cache = self.user_embedding_cache[split]
                    retrieval_tensor = torch.zeros(
                        self.retrieval_top_k,
                        cache.size(-1),
                        dtype=cache.dtype,
                    )
                else:
                    retrieval_tensor = None
            if retrieval_tensor is not None:
                user_rag_embs.append(retrieval_tensor)

        max_label_len = max(len(label) for label in labels) if labels else 0
        labels = [self._pad_right(label, max_label_len, self.label_pad_token_id) for label in labels]

        batch = {
            "lm_inputs": {
                "input_ids": torch.tensor(id_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(id_attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            },
            "input_ids": torch.tensor(id_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(id_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "text_input_ids": torch.tensor(text_input_ids, dtype=torch.long),
            "text_attention_mask": torch.tensor(text_attention_mask, dtype=torch.long),
            "index": torch.tensor(sample_indices, dtype=torch.long),
            "item_id": torch.tensor(item_ids, dtype=torch.long),
            "label_id": torch.tensor(item_ids, dtype=torch.long),
            "user_id": torch.tensor(user_ids, dtype=torch.long),
        }

        if user_rag_embs:
            batch["user_rag_emb"] = torch.stack(user_rag_embs, dim=0)

        return batch
