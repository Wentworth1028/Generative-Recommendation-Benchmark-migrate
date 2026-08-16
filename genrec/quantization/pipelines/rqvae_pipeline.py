import os
import csv
import json
import torch
import numpy as np
import pickle
from collections import Counter

from genrec.quantization.tokenizers.rqvae_tokenizer import RQVAETokenizer
from genrec.quantization.optimizers.rqvae_optimizer import RQVAETokenizerOptimizer
from genrec.quantization.debias.popularity_optimizer import PopularityRQVAETokenizerOptimizer
from genrec.quantization.trainers.rqvae_trainer import RQVAETrainer
from genrec.quantization.data.dataset.rqvae_dataset import create_item_dataloader

class DataloaderWrapper:
    def __init__(self, dl):
        self.dl = dl
    def __iter__(self):
        for batch in self.dl:
            yield batch['item_ids'], batch['embeddings']

    def __len__(self):
        return len(self.dl)


class RQVAETrainingPipeline:
    """
    A pipeline that encapsulates the complete training process for an RQ-VAE Tokenizer.
    
    By calling the .run() method, the following steps can be completed in one go:
    1. Data loading and preprocessing
    2. Initialization of the model, optimizer, and trainer
    3. Codebook initialization using K-Means
    4. Model training
    5. Finalization and verification of the Tokenizer
    """
    def __init__(self, config, accelerator=None):
        """
        Initializes the pipeline.
        
        Args:
            config (dict): A configuration dictionary containing all model and training parameters.
                           It must include valid data and output paths.
        """
        self.config = config
        self.accelerator = accelerator
        self.tokenizer = None
        self.optimizer = None
        self.trainer = None
        self.dataset = None
        self.train_dataloader = None
        self.item_popularity = {}
        self.final_model_decision = self.config.get('final_model_decision', 'save_final').lower()
        if self.final_model_decision not in ['save_final', 'save_best']:
            raise ValueError(f"Invalid 'final_model_decision' value: {self.final_model_decision}. "
                             f"Must be 'save_final' or 'save_best'.")

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
            self.tokenizer = RQVAETokenizer.load(self.config)
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
        """Creates the dataset and dataloader."""
        print("\n--- Creating Dataset and Dataloader ---")
        required_paths = ['data_text_files', 'interaction_files','save_path', 'checkpoint_path', 'text_encoder_model']
        for path_key in required_paths:
            if not self.config.get(path_key):
                raise ValueError(f"Configuration error: '{path_key}' must be specified in the config.")
        global_batch_size = self.config['batch_size']
        if self.accelerator is not None:
            num_processes = self.accelerator.num_processes
            per_device_batch_size = max(1, global_batch_size // num_processes)
            
            if self.accelerator.is_main_process and global_batch_size % num_processes != 0:
                print(f"Warning: Global batch size {global_batch_size} is not perfectly divisible by {num_processes} GPUs. "
                      f"Using per-device batch size of {per_device_batch_size} (Actual global will be {per_device_batch_size * num_processes}).")
        else:
            per_device_batch_size = global_batch_size
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                dataset, train_dataloader, valid_dataloader = create_item_dataloader(
                    data_text_files=self.config['data_text_files'],
                    config=self.config,
                    batch_size=per_device_batch_size,
                    text_encoder_model=self.config["text_encoder_model"],
                    embedding_strategy=self.config.get("embedding_strategy", "mean_pooling"),
                )
        else:
            dataset, train_dataloader, valid_dataloader = create_item_dataloader(
                data_text_files=self.config['data_text_files'],
                config=self.config,
                batch_size=per_device_batch_size,
                text_encoder_model=self.config["text_encoder_model"],
                embedding_strategy=self.config.get("embedding_strategy", "mean_pooling"),
                num_workers=self.config.get("num_workers", 4)
            )
        self.dataset = dataset
        self.train_dataloader = DataloaderWrapper(train_dataloader)
        self.valid_dataloader = DataloaderWrapper(valid_dataloader)
        print("Dataset and Dataloader created successfully.")

        print("\n--- Running a definitive data check ---")
        try:
            _, check_embeddings = next(iter(self.train_dataloader))
            print(f"Batch shape of embeddings from dataloader: {check_embeddings.shape}")
            assert len(check_embeddings.shape) == 2, "Embeddings should be a 2D tensor."
            assert check_embeddings.shape[1] == self.config['sent_emb_dim'], "Embedding dimension mismatch."
            print("--- Data check passed. ---")
        except Exception as e:
            print(f"Data check failed: {e}")
            raise

    def _initialize_components(self):
        """Initializes the model, optimizer, and trainer."""
        print("\n--- Initializing Model, Optimizer, and Trainer ---")
        self.config['item_popularity'] = self.item_popularity
        self.tokenizer = RQVAETokenizer(self.config)
        if self._popularity_debias_enabled():
            optimizer_class = PopularityRQVAETokenizerOptimizer
        else:
            optimizer_class = RQVAETokenizerOptimizer
        self.optimizer = optimizer_class(self.config, self.tokenizer)
        print(f"Using tokenizer optimizer: {optimizer_class.__name__}")
        self.trainer = RQVAETrainer(self.config, self.tokenizer, self.optimizer, accelerator=self.accelerator)
        print("Initialization complete.")

    def _popularity_debias_enabled(self) -> bool:
        """Select the debias optimizer without changing legacy YAML names."""
        def as_bool(value):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        balance_weights = [
            float(self.config.get('balance_weight', 0.0) or 0.0),
            float(self.config.get('popularity_balance_weight', 0.0) or 0.0),
        ]
        prefix_enabled = any(
            as_bool(self.config.get(key, False))
            for key in ('prefix_balance_enabled', 'popularity_prefix_balance_enabled')
        )
        log_distribution = any(
            as_bool(self.config.get(key, False))
            for key in ('log_distribution', 'popularity_balance_log_distribution')
        )
        return max(balance_weights) > 0.0 or prefix_enabled or log_distribution

    def _initialize_codebooks(self):
        """Initializes the RQ-VAE codebooks using K-Means (Main Process Only)."""
        if self.accelerator is None or self.accelerator.is_main_process:
            print("\n--- Initializing RQ-VAE codebooks with K-Means ---")
            all_embeddings = np.vstack([self.dataset.item_embeddings[item_id] for item_id in self.dataset.item_ids])
            print(f"Total embeddings shape for K-Means: {all_embeddings.shape}")
            self.tokenizer.initialize_rqvae(all_embeddings)
            print("Codebook initialization complete.")
            
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _train(self):
        """Executes the training loop."""
        print("\n--- Starting Tokenizer Training ---")
        self.trainer.fit(self.train_dataloader,self.valid_dataloader)
        print("Training finished.")

    @staticmethod
    def _optional_path(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return str(value)

    def _load_rqvae_checkpoint(self, checkpoint_path: str, purpose: str = "checkpoint"):
        device = self.config.get('device', 'cpu')
        original_state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(original_state_dict, dict) and "state_dict" in original_state_dict:
            original_state_dict = original_state_dict["state_dict"]

        new_state_dict = {}
        for key, value in original_state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            if new_key.startswith('rq_vae.'):
                new_key = new_key[len('rq_vae.'):]
            new_state_dict[new_key] = value

        self.tokenizer.rq_vae.load_state_dict(new_state_dict)
        self.tokenizer.rq_vae.to(device)
        print(f"--- RQ-VAE model loaded successfully from {purpose}: {checkpoint_path} ---")

    def _finalize_and_verify(self):
        """Finalizes the Tokenizer and verifies its functionality."""
        print("\n--- Finalizing and Testing Tokenizer ---")
        user2item_path = self.config['interaction_files']
        with open(user2item_path, 'rb') as f:
            user2item_data = pickle.load(f)
        user_id_column = user2item_data['UserID']
        all_user_ids = user_id_column.tolist()
        print(f"Extracted {len(all_user_ids)} unique user IDs.")
        item_ids_list = self.dataset.item_ids
        embeddings_array = np.array([self.dataset.item_embeddings[id] for id in item_ids_list])

        self.tokenizer.finalize_tokenization((item_ids_list, embeddings_array), all_user_ids)
        self._export_item_popularity_tokens_csv()
        print(f"Tokenizer finalized. Item to token map saved to: {self.config['save_path']}")

    def run(self):
        """
        Executes the complete training pipeline in sequence.
        """
        print(f"--- Starting RQ-VAE Training Pipeline ---")
        print(f"Using device: {self.config.get('device', 'cpu')}")

        self.item_popularity = self._compute_item_popularity()
        self._prepare_data()
        self._initialize_components()
        checkpoint_path = self._optional_path(self.config.get('checkpoint_path'))
        finetune_from_checkpoint = self._optional_path(self.config.get('finetune_from_checkpoint'))
        if finetune_from_checkpoint:
            if not os.path.exists(finetune_from_checkpoint):
                raise FileNotFoundError(f"Fine-tune source checkpoint does not exist: {finetune_from_checkpoint}")
            print(f"\n--- Fine-tuning RQ-VAE from checkpoint '{finetune_from_checkpoint}'. ---")
            print("--- Loading source model and continuing tokenizer training. ---")
            self._load_rqvae_checkpoint(finetune_from_checkpoint, purpose="fine-tune source")
            self._train()
            if checkpoint_path and os.path.exists(checkpoint_path) and self.final_model_decision == 'save_best':
                print(f"\n--- Best fine-tuned checkpoint found at '{checkpoint_path}'. ---")
                try:
                    self._load_rqvae_checkpoint(checkpoint_path, purpose="best fine-tuned checkpoint")
                except Exception as e:
                    print(f"--- Error loading checkpoint: {e} ---")
            self._finalize_and_verify()
            print("\n--- RQ-VAE Training Pipeline Finished Successfully ---")
            return

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n--- Checkpoint found at '{checkpoint_path}'. ---")
            print("--- Loading model from checkpoint and skipping training. ---")
            try:
                self._load_rqvae_checkpoint(checkpoint_path)
            except Exception as e:
                print(f"--- Error loading checkpoint: {e} ---")
                print("--- Proceeding with full training pipeline instead. ---")
                self._initialize_codebooks()
                self._train()
                if checkpoint_path and os.path.exists(checkpoint_path) and self.final_model_decision == 'save_best':
                    print(f"\n--- Best usage checkpoint found at '{checkpoint_path}'. ---")
                    try:
                        self._load_rqvae_checkpoint(checkpoint_path, purpose="best usage checkpoint")
                    except Exception as e:
                        print(f"--- Error loading checkpoint: {e} ---")
        else:
            print("\n--- No checkpoint found. Proceeding with full training. ---")
            self._initialize_codebooks()
            self._train()
            if checkpoint_path and os.path.exists(checkpoint_path) and self.final_model_decision == 'save_best':
                print(f"\n--- Best usage checkpoint found at '{checkpoint_path}'. ---")
                try:
                    self._load_rqvae_checkpoint(checkpoint_path, purpose="best usage checkpoint")
                except Exception as e:
                    print(f"--- Error loading checkpoint: {e} ---")
        

        # self._initialize_codebooks()
        # self._train()
        self._finalize_and_verify()
        
        print("\n--- RQ-VAE Training Pipeline Finished Successfully ---")
