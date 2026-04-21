import os
import csv
import json
import numpy as np
import pickle
from collections import Counter

from genrec.quantization.tokenizers.rqkmeans_tokenizer import RQKmeansTokenizer
from genrec.quantization.data.dataset.rqvae_dataset import create_item_dataloader

class RQKmeansPipeline:
    def __init__(self, config, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        self.tokenizer = None
        self.dataset = None
        self.item_popularity = {}

    def _is_main_process(self):
        return self.accelerator is None or self.accelerator.is_main_process

    def _compute_item_popularity(self):
        interaction_path = self.config['interaction_files']
        with open(interaction_path, 'rb') as f:
            user2item_data = pickle.load(f)

        counter = Counter()
        for _, row in user2item_data.iterrows():
            item_seq = row['ItemID']
            counter.update(int(item_id) for item_id in item_seq)

        if self._is_main_process():
            print(f"Computed popularity for {len(counter)} items from: {interaction_path}")
        return dict(counter)

    def _ensure_tokenizer_loaded_from_json(self):
        if self.tokenizer is None or not self.tokenizer.item2tokens:
            self.tokenizer = RQKmeansTokenizer.load(self.config)
        return self.tokenizer

    def _export_item_popularity_tokens_csv(self):
        if not self._is_main_process():
            return

        tokenizer = self._ensure_tokenizer_loaded_from_json()
        output_path = self.config['save_path'].replace('.json', '_item_popularity_tokens.csv')
        all_item_ids = sorted(set(self.item_popularity.keys()) | set(tokenizer.item2tokens.keys()))

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["item_id", "popularity", "token_ids"])
            for item_id in all_item_ids:
                token_ids = tokenizer.item2tokens.get(item_id, [])
                writer.writerow([
                    int(item_id),
                    int(self.item_popularity.get(item_id, 0)),
                    json.dumps([int(token_id) for token_id in token_ids], ensure_ascii=False),
                ])

        print(f"Saved item-popularity-token CSV to: {output_path}")

    def export_existing_popularity_token_csv(self):
        self.item_popularity = self._compute_item_popularity()
        self._ensure_tokenizer_loaded_from_json()
        self._export_item_popularity_tokens_csv()

    def _prepare_data(self):
        print("\n--- Creating Dataset ---")
        global_batch_size = self.config.get('batch_size', 256)
        
        if self.accelerator is not None:
            num_processes = self.accelerator.num_processes
            per_device_batch_size = max(1, global_batch_size // num_processes)
            with self.accelerator.main_process_first():
                dataset, _, _ = create_item_dataloader(
                    data_text_files=self.config['data_text_files'],
                    config=self.config,
                    batch_size=per_device_batch_size,
                    text_encoder_model=self.config["text_encoder_model"],
                    embedding_strategy=self.config.get("embedding_strategy", "mean_pooling"),
                )
        else:
            dataset, _, _ = create_item_dataloader(
                data_text_files=self.config['data_text_files'],
                config=self.config,
                batch_size=global_batch_size,
                text_encoder_model=self.config["text_encoder_model"],
                embedding_strategy=self.config.get("embedding_strategy", "mean_pooling"),
                num_workers=self.config.get("num_workers", 4)
            )
        self.dataset = dataset

    def _run_kmeans_and_finalize(self):
        is_main_process = (self.accelerator is None) or self.accelerator.is_main_process

        if is_main_process:
            print("\n--- [Main Process] Initializing RQKmeans Tokenizer ---")
            self.tokenizer = RQKmeansTokenizer(self.config)
            
            item_ids_list = self.dataset.item_ids
            embeddings_array = np.array([self.dataset.item_embeddings[item_id] for item_id in item_ids_list])
            
            user2item_path = self.config['interaction_files']
            with open(user2item_path, 'rb') as f:
                user2item_data = pickle.load(f)
            all_user_ids = user2item_data['UserID'].tolist()
            
            print("\n--- [Main Process] Running RQ-KMeans (on CPU) & Saving JSON ---")
            self.tokenizer.finalize_tokenization((item_ids_list, embeddings_array), all_user_ids)
            self._export_item_popularity_tokens_csv()
        else:
            print(f"\n--- [Worker Process] Waiting for Main Process to finish KMeans ---")

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def run(self):
        if self.accelerator is None or self.accelerator.is_main_process:
            print(f"=== Starting RQ-KMeans Training Pipeline ===")

        self.item_popularity = self._compute_item_popularity()
        self._prepare_data()
        
        self._run_kmeans_and_finalize()
        
        if self.accelerator is None or self.accelerator.is_main_process:
            print("\n=== RQ-KMeans Training Pipeline Finished Successfully ===")
