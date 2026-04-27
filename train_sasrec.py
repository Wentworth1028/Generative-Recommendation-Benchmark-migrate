import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments
)

import pandas as pd
import numpy as np
import pickle
import os
import time
import argparse
import random
import json
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

from disrec.sasrec.sasrec4hf import SASRecConfig,SASRec4HF
from disrec.datasets.model_dataset import SASRecDataset
from disrec.datasets.data_collator import SASRecDataCollator
from genrec.utils.popularity_metrics import compute_prediction_popularity_metrics, compute_train_item_popularity

from transformers import TrainerCallback,EarlyStoppingCallback
import numpy as np


class EvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n_epochs=5):
        self.n_epochs = n_epochs
        self.last_eval_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch ) % self.n_epochs == 0:
            control.should_evaluate = True
            self.last_eval_epoch = state.epoch
        else:
            control.should_evaluate = False
            
    def on_evaluate(self, args, state, control, metrics, **kwargs):

        control.should_save = state.epoch == self.last_eval_epoch
def compute_metrics(eval_pred):
    top_k_indices, labels = eval_pred.predictions, eval_pred.label_ids
    
    valid_mask = (labels != -100)

    last_item_indices = valid_mask.shape[1] - 1 - np.argmax(np.fliplr(valid_mask), axis=1)
    
    batch_indices = np.arange(labels.shape[0])
    true_labels = labels[batch_indices, last_item_indices] # [Batch]
    
    results = {}
    K_VALUES = [1, 5, 10]
    true_labels_reshaped = true_labels[:, np.newaxis] # [Batch, 1]
    
    for k in K_VALUES:
        top_k_preds = top_k_indices[:, :k] # [Batch, k]

        hits = (true_labels_reshaped == top_k_preds).any(axis=1)
        results[f"Hit@{k}"] = np.mean(hits)

        hit_mask = (true_labels_reshaped == top_k_preds)
        hit_positions = np.where(hit_mask) 
        
        ranks = hit_positions[1] 
        
        if len(ranks) > 0:
            dcg = 1.0 / np.log2(ranks + 2)
            ndcg_at_k = np.sum(dcg) / len(true_labels)
        else:
            ndcg_at_k = 0.0
            
        results[f"NDCG@{k}"] = ndcg_at_k

    return results

def load_data_and_get_vocab_size(file_path: str) -> int:

    print(f"load from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Not Found: {file_path}")
    
    df = pd.read_pickle(file_path)

    

    if 'ItemID' not in df.columns or df['ItemID'].apply(len).sum() == 0:
        raise ValueError("Error: 'ItemID' Not Found")
        

    max_item_id = df['ItemID'].explode().max()
    
    vocab_size = int(max_item_id) + 2
    
    print(f"Max ItemID: {max_item_id}")
    print(f"vocab_size: {vocab_size}")
    
    return vocab_size
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    _, indices = torch.topk(logits, k=10, dim=-1) 
    return indices
def print_model_parameters(model: nn.Module):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 50)
    print("model params:")
    print(f"  - total_params:  {total_params:,}")
    print(f"  - trainable_params:   {trainable_params:,}")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SASRec and export CF item embeddings.")
    parser.add_argument("--data_file", type=str, default="./data/Beauty/user2item.pkl")
    parser.add_argument("--output_dir", type=str, default="./sasrec")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--num_neg_samples", type=int, default=4)
    parser.add_argument("--norm_emb", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1024)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    REAL_DATA_FILE = args.data_file
    OUTPUT_DIR = args.output_dir
    MAX_SEQ_LEN = args.max_seq_len
    EMBEDDING_DIM = args.embedding_dim
    EVAL_EVERY_N_EPOCHS = args.eval_every_n_epochs
    VOCAB_SIZE = load_data_and_get_vocab_size(REAL_DATA_FILE)
    
    config = SASRecConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        hidden_size=EMBEDDING_DIM,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout_prob=args.hidden_dropout_prob,
        pad_token_id=0,
        norm_emb=args.norm_emb,
        num_neg_samples=args.num_neg_samples,
    )
    model = SASRec4HF(config)
    print_model_parameters(model)
    train_dataset = SASRecDataset(
        data_interaction_files=REAL_DATA_FILE,
        config=config.to_dict(),
        mode='train'
    )
    eval_dataset = SASRecDataset(
        data_interaction_files=REAL_DATA_FILE,
        config=config.to_dict(),
        mode='valid'
    )
    test_dataset = SASRecDataset(
        data_interaction_files=REAL_DATA_FILE,
        config=config.to_dict(),
        mode='test'
    )
    data_collator = SASRecDataCollator(pad_token_id=config.pad_token_id, max_seq_len=MAX_SEQ_LEN)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="NDCG@10", 
        greater_is_better=True,       
        report_to="none",
        learning_rate=args.learning_rate,
        ddp_find_unused_parameters=False,
        weight_decay=args.weight_decay,
        # warmup_steps=500, 
    )
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        EvaluateEveryNEpochsCallback(n_epochs=EVAL_EVERY_N_EPOCHS)
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks = callbacks,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    

    trainer.train()

    print("--- Test Set Evaluation ---")
    test_predictions = trainer.predict(test_dataset)
    test_results = test_predictions.metrics
    test_results_renamed = {f"test_{k}": v for k, v in test_results.items()}
    sasrec_predictions = []
    for row in test_predictions.predictions:
        seen = set()
        items = []
        for item_id in row.tolist():
            raw_item_id = int(item_id) - 1
            if raw_item_id < 0 or raw_item_id in seen:
                continue
            seen.add(raw_item_id)
            items.append(raw_item_id)
        sasrec_predictions.append(items)
    train_item_popularity = compute_train_item_popularity(REAL_DATA_FILE, shift_item_id=0)
    popularity_metrics = compute_prediction_popularity_metrics(
        sasrec_predictions,
        train_item_popularity,
        k_list=(1, 5, 10),
        quantiles=(0.1, 0.2),
    )
    test_results_renamed.update({f"test_{key}": value for key, value in popularity_metrics.items()})
    print("Result:", test_results_renamed)
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    final_metrics = {
        "model": "sasrec",
        "data_file": os.path.abspath(REAL_DATA_FILE),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_model_dir": os.path.abspath(best_model_dir),
        "best_metric": trainer.state.best_metric,
        "metrics": test_results_renamed,
        "popularity_metrics": popularity_metrics,
        "config": {
            "max_seq_len": args.max_seq_len,
            "embedding_dim": args.embedding_dim,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "hidden_dropout_prob": args.hidden_dropout_prob,
            "num_neg_samples": args.num_neg_samples,
            "norm_emb": args.norm_emb,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "eval_every_n_epochs": args.eval_every_n_epochs,
            "save_total_limit": args.save_total_limit,
            "seed": args.seed,
        },
    }
    final_metrics_path = os.path.join(OUTPUT_DIR, "final_metrics.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    print(f"save final metrics to: {final_metrics_path}")

    print("--- Inducing CF Embedding ---")
    
    final_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    final_model.eval()
    
    with torch.no_grad():

        cf_embeddings = final_model.get_input_embeddings().weight.detach().cpu()

    cf_emb_save_path = os.path.join(OUTPUT_DIR, "sasrec_cf_embedding.pt")
    
    torch.save(cf_embeddings, cf_emb_save_path)
    
    print(f"save CF Embedding to: {cf_emb_save_path}")
    print(f"Embedding shape: {cf_embeddings.shape}") # [VOCAB_SIZE, EMBEDDING_DIM]
