from __future__ import annotations

import csv
import numpy as np
from pathlib import Path

from genrec.quantization.data.dataset.rqvae_dataset import ItemEmbeddingDataset
from genrec.quantization.tokenizers.ghost_tokenizer import GhostSKTTokenizer
from genrec.utils.popularity_metrics import compute_train_item_popularity


class GhostSKTTrainingPipeline:
    """Offline pipeline for GHOST/SKT tokenizer construction."""

    def __init__(self, config, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        self.dataset = None
        self.tokenizer: GhostSKTTokenizer | None = None
        self.item_popularity: dict[int, int] = {}

    def _is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    def _load_item_embeddings(self) -> tuple[list[int], np.ndarray]:
        dataset = ItemEmbeddingDataset(
            data_text_files=self.config["data_text_files"],
            config=self.config,
            text_encoder_model=self.config.get("text_encoder_model", "sentence-transformers/sentence-t5-base"),
            embedding_extraction_strategy=self.config.get("embedding_strategy", "mean_pooling"),
            device=self.config.get("device"),
        )
        self.dataset = dataset
        item_ids = [int(item_id) for item_id in dataset.item_ids]
        embeddings = np.asarray([dataset.item_embeddings[item_id] for item_id in item_ids], dtype=np.float32)
        return item_ids, embeddings

    def _compute_item_popularity(self) -> dict[int, int]:
        return compute_train_item_popularity(self.config["interaction_files"], shift_item_id=0)

    def _ensure_tokenizer_loaded_from_json(self) -> GhostSKTTokenizer:
        if self.tokenizer is None or not self.tokenizer.item2tokens:
            self.tokenizer = GhostSKTTokenizer.load(self.config)
        return self.tokenizer

    def _export_item_popularity_tokens_csv(self) -> None:
        if not self._is_main_process():
            return

        tokenizer = self._ensure_tokenizer_loaded_from_json()
        output_path = Path(self.config["save_path"].replace(".json", "_item_popularity_tokens.csv"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_item_ids = sorted(set(self.item_popularity.keys()) | set(tokenizer.item2tokens.keys()))
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["item_id", "popularity", "token_ids"])
            for item_id in all_item_ids:
                token_ids = tokenizer.item2tokens.get(item_id, [])
                writer.writerow(
                    [
                        int(item_id),
                        int(self.item_popularity.get(item_id, 0)),
                        str([int(token_id) for token_id in token_ids]),
                    ]
                )

        print(f"Saved item-popularity-token CSV to: {output_path}")

    def export_existing_popularity_token_csv(self) -> None:
        self.item_popularity = self._compute_item_popularity()
        self._ensure_tokenizer_loaded_from_json()
        self._export_item_popularity_tokens_csv()

    def run(self) -> None:
        if self._is_main_process():
            print("=== Starting GHOST SKT Tokenizer Pipeline ===")

        self.item_popularity = self._compute_item_popularity()
        item_ids, embeddings = self._load_item_embeddings()

        if self._is_main_process():
            print(f"[GHOST] loaded {len(item_ids)} item embeddings")
            self.tokenizer = GhostSKTTokenizer(self.config)
            self.tokenizer.finalize_tokenization((item_ids, embeddings), self.item_popularity)
            self._export_item_popularity_tokens_csv()
            print(f"[GHOST] tokenizer saved to: {self.config['save_path']}")

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
