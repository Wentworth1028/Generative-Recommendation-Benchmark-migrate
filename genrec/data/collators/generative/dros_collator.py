"""TIGER collator adapter that adds padded item-level histories for DROS."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from genrec.data.collators.generative.tiger_collator import TigerDataCollator


@dataclass
class DROSTigerDataCollator(TigerDataCollator):
    history_pad_item_id: int = -1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        max_history_length = max(len(feature["history_items"]) for feature in features)
        history_rows = []
        for feature in features:
            history = list(feature["history_items"])
            padding = [self.history_pad_item_id] * (max_history_length - len(history))
            history_rows.append(padding + history)
        batch["history_item_ids"] = torch.tensor(history_rows, dtype=torch.long)
        return batch
