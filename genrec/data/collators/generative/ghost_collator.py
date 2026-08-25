from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GhostDataCollator:
    max_seq_len: int
    pad_token_id: int
    eos_token_id: int
    mode: str = "train"
    label_pad_token_id: int = -100

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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        sample_indices: List[int] = []
        item_ids: List[int] = []
        user_ids: List[int] = []
        target_is_tail: List[int] = []

        truncated_sources: List[List[int]] = []
        for feature in features:
            source_tokens = list(feature["source_tokens"])
            source_tokens = source_tokens[-self.max_seq_len :]
            truncated_sources.append(source_tokens)

        max_source_len = max((len(tokens) for tokens in truncated_sources), default=0)
        for feature, source_tokens in zip(features, truncated_sources):
            padded_source = self._pad_left(source_tokens, max_source_len, self.pad_token_id)
            input_ids_list.append(padded_source)

            target_tokens = list(feature["target_tokens"])
            target_tokens.append(self.eos_token_id)
            labels_list.append(target_tokens)

            sample_indices.append(int(feature["index"]))
            item_ids.append(int(feature["item_id"]))
            user_ids.append(int(feature.get("user_id", -1)))
            target_is_tail.append(int(feature.get("target_is_tail", 0)))

        max_label_len = max((len(label) for label in labels_list), default=0)
        labels_list = [self._pad_right(label, max_label_len, self.label_pad_token_id) for label in labels_list]

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        labels = torch.tensor(labels_list, dtype=torch.long)

        batch = {
            "lm_inputs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            },
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "index": torch.tensor(sample_indices, dtype=torch.long),
            "item_id": torch.tensor(item_ids, dtype=torch.long),
            "label_id": torch.tensor(item_ids, dtype=torch.long),
            "user_id": torch.tensor(user_ids, dtype=torch.long),
            "target_is_tail": torch.tensor(target_is_tail, dtype=torch.long),
        }
        return batch

