from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import pickle

import pandas as pd

from genrec.data.datasets.base_dataset import BaseSeqRecDataset
from genrec.genplugin.data_utils import build_leave_one_out_samples, iteratively_filter_interactions


class GhostDataset(BaseSeqRecDataset):
    """Leave-one-out dataset for GHOST / SKT variable-length semantic IDs."""

    def __init__(
        self,
        data_interaction_files: str,
        data_text_files: str,
        tokenizer,
        config: dict,
        mode: str = "train",
        device: Optional[str] = None,
    ) -> None:
        config = dict(config)
        config.setdefault("use_user_tokens", False)
        self.min_user_interactions = int(config.get("min_user_interactions", 5))
        self.min_item_interactions = int(config.get("min_item_interactions", 5))
        self.max_history_items = int(config.get("max_seq_len", 20))
        super().__init__(
            data_interaction_files=data_interaction_files,
            data_text_files=data_text_files,
            tokenizer=tokenizer,
            config=config,
            mode=mode,
            device=device,
        )
        self.use_user_tokens = False
        self.max_sid_length = int(getattr(self.tokenizer, "total_sid_length", max(len(tokens) for tokens in self.tokenizer.item2tokens.values())))
        self.max_token_len = self.max_history_items * (self.max_sid_length + 1)

    def _load_user_seqs(self) -> Dict[int, List[int]]:
        with open(self.data_interaction_files, "rb") as f:
            interaction_frame = pickle.load(f)
        if not isinstance(interaction_frame, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in {self.data_interaction_files}, got {type(interaction_frame)}")

        filtered_frame = iteratively_filter_interactions(
            interaction_frame,
            min_user_interactions=self.min_user_interactions,
            min_item_interactions=self.min_item_interactions,
        )
        if filtered_frame.empty:
            return {}

        user_seqs: Dict[int, List[int]] = defaultdict(list)
        for _, row in filtered_frame.iterrows():
            user_id = int(row["UserID"])
            user_seqs[user_id] = [int(item) for item in row["ItemID"]]
        return user_seqs

    def _create_samples(self) -> List[Dict[str, Any]]:
        return build_leave_one_out_samples(
            self.user_seqs,
            self.mode,
            max_history_items=self.max_history_items,
        )

    def _get_item_tokens(self, item_id: int) -> List[int]:
        if hasattr(self.tokenizer, "get_item_tokens"):
            return list(self.tokenizer.get_item_tokens(item_id))
        return list(super()._get_item_tokens(item_id))

    def __getitem__(self, index: int) -> Dict[str, Union[int, List[int]]]:
        sample = self.samples[index]
        history_items = list(sample["history_items"])
        target_item = int(sample["target_item"])
        user_id = int(sample["user_id"])

        source_tokens: List[int] = []
        for item_id in history_items:
            source_tokens.extend(self._get_item_tokens(item_id))
            source_tokens.append(int(self.tokenizer.eos_token))

        target_tokens = self._get_item_tokens(target_item)
        target_is_tail = int(getattr(self.tokenizer, "is_tail_item", lambda _: False)(target_item))
        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "target_id": target_item,
            "item_id": target_item,
            "target_is_tail": target_is_tail,
            "user_id": user_id,
            "history_item_ids": history_items,
            "index": int(sample["index"]),
            "split": self.mode,
        }
