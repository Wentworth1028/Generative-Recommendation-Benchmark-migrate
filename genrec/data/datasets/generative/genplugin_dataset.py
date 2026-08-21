from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Any, Union, Optional
import pickle

import pandas as pd

from genrec.data.datasets.base_dataset import BaseSeqRecDataset
from genrec.genplugin.data_utils import build_leave_one_out_samples, iteratively_filter_interactions


class GenPluginDataset(BaseSeqRecDataset):
    """Leave-one-out dataset for GENPLUGIN pretraining / RAR fine-tuning."""

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
        self.sid_length = int(config.get("sid_length", config.get("tokens_per_item", 4)))
        super().__init__(
            data_interaction_files=data_interaction_files,
            data_text_files=data_text_files,
            tokenizer=tokenizer,
            config=config,
            mode=mode,
            device=device,
        )
        self.use_user_tokens = False
        self.max_token_len = self.tokens_per_item * self.max_history_items

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

    def __getitem__(self, index: int) -> Dict[str, Union[int, List[int]]]:
        sample = self.samples[index]
        history_items = list(sample["history_items"])
        target_item = int(sample["target_item"])
        user_id = int(sample["user_id"])

        source_tokens: List[int] = []
        for item_id in history_items:
            source_tokens.extend(self._get_item_tokens(item_id))

        target_tokens = self._get_item_tokens(target_item)
        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "text_input_ids": history_items,
            "index": int(sample["index"]),
            "item_id": target_item,
            "target_id": target_item,
            "user_id": user_id,
            "history_item_ids": history_items,
            "split": self.mode,
        }
