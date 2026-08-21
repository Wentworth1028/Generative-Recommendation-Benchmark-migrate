from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import EarlyStoppingCallback, TrainingArguments

from genrec.data.collators.generative.genplugin_collator import GenPluginDataCollator
from genrec.data.datasets.generative.genplugin_dataset import GenPluginDataset
from genrec.genplugin.data_utils import save_json
from genrec.genplugin.tokenizer_utils import load_semantic_id_tokenizer
from genrec.genplugin.retrieval_utils import rerank_candidates_by_cosine
from genrec.trainers.generative.genplugin_trainer import GenPluginTrainer
from genrec.utils.common_utils import set_seed
from genrec.utils.logging_utils import setup_logging
from genrec.utils.metrics import compute_metrics
from genrec.utils.models_setup.genplugin_setup import create_genplugin_letter_model, create_genplugin_tiger_model
from genrec.utils.popularity_metrics import (
    compute_dataset_item_popularity,
    compute_prediction_popularity_metrics,
    compute_token_popularity_metrics,
)


def _recursive_to_device(batch: Any, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _recursive_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_recursive_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_recursive_to_device(value, device) for value in batch)
    return batch


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        path = path / "pytorch_model.bin"
    if not path.exists():
        raise FileNotFoundError(f"Pretrained state dict not found: {path}")
    return torch.load(path, map_location="cpu")


def _build_datasets(tokenizer, data_interaction_files, data_text_files, model_config):
    dataset_kwargs = dict(
        data_interaction_files=data_interaction_files,
        data_text_files=data_text_files,
        tokenizer=tokenizer,
        config=model_config,
    )
    return (
        GenPluginDataset(mode="train", **dataset_kwargs),
        GenPluginDataset(mode="valid", **dataset_kwargs),
        GenPluginDataset(mode="test", **dataset_kwargs),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GENPLUGIN with RAR.")
    parser.add_argument("--data_interaction_files", required=True)
    parser.add_argument("--data_text_files", required=True)
    parser.add_argument("--item2tokens_path", required=True)
    parser.add_argument("--text_embedding_path", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--retrieval_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", choices=["tiger", "letter"], default="tiger")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--max_gen_length", type=int, default=5)
    parser.add_argument("--max_k", type=int, default=20)
    parser.add_argument("--inference_mode", default="FastCBS")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(output_dir / "logs"))
    logger.info(f"output_dir: {output_dir}")

    tokenizer = load_semantic_id_tokenizer(args.item2tokens_path)
    model_config = {
        "d_model": 128,
        "d_kv": 64,
        "d_ff": 1024,
        "num_layers": 4,
        "num_decoder_layers": 4,
        "num_heads": 6,
        "dropout_rate": 0.1,
        "tie_word_embeddings": True,
        "max_seq_len": args.max_seq_len,
        "sid_length": tokenizer.digits,
        "text_embedding_dim": 4096,
        "text_projection_hidden_dim": 2048,
        "genplugin_stage": "rar",
        "kl_temperature": 0.85,
        "item_temperature": 0.9,
        "kl_weight": 0.5,
        "item_weight": 0.1,
        "semantic_substitution_start_epoch": 10,
        "semantic_substitution_outer_prob": 0.5,
        "semantic_substitution_position_prob": 0.4,
        "semantic_substitution_top_k": 5,
    }

    train_dataset, valid_dataset, test_dataset = _build_datasets(
        tokenizer=tokenizer,
        data_interaction_files=args.data_interaction_files,
        data_text_files=args.data_text_files,
        model_config=model_config,
    )

    if args.backbone == "tiger":
        model = create_genplugin_tiger_model(
            vocab_size=tokenizer.vocab_size,
            model_config=model_config,
            text_embedding_path=args.text_embedding_path,
        )
    else:
        model = create_genplugin_letter_model(
            vocab_size=tokenizer.vocab_size,
            model_config=model_config,
            text_embedding_path=args.text_embedding_path,
        )

    state_dict = _load_state_dict(Path(args.pretrained_model_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded pretrained weights. missing={len(missing)}, unexpected={len(unexpected)}")
    model.freeze_for_rar()

    cache_root = Path(args.cache_dir) if args.cache_dir else Path(args.pretrained_model_path).parent.parent / "caches"
    retrieval_root = Path(args.retrieval_dir) if args.retrieval_dir else output_dir / "retrieval"

    user_embedding_paths = {
        split: cache_root / split / "user_emb_mean.npy" for split in ("train", "valid", "test")
    }
    reranked_paths = {
        split: retrieval_root / split / "reranked_user_index.json" for split in ("train", "valid", "test")
    }

    train_collator = GenPluginDataCollator(
        max_item_seq_len=args.max_seq_len,
        sid_length=tokenizer.digits,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        text_pad_token_id=model.text_sentinel_index,
        stage="rar",
        retrieval_top_k=30,
        user_embedding_path=user_embedding_paths,
        reranked_user_index_path=reranked_paths,
    )

    train_batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_train_batch_size = max(1, train_batch_size // world_size)
    per_device_eval_batch_size = max(1, eval_batch_size // world_size)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "generation_model"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="ndcg@10",
        greater_is_better=True,
        logging_strategy="epoch",
        logging_dir=str(output_dir / "logs"),
        report_to=[],
        warmup_ratio=args.warmup_ratio,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = GenPluginTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        compute_metrics=lambda pred: compute_metrics(pred, tokens_to_item_map=tokenizer.tokens2item),
        generation_params={
            "max_gen_length": args.max_gen_length,
            "num_beams": args.num_beams,
            "max_k": args.max_k,
        },
        item2tokens=tokenizer.item2tokens,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        vocab_size=tokenizer.vocab_size,
        inference_mode=args.inference_mode,
        do_generate=True,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "generation_model"))
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
    torch.save(unwrapped_model.state_dict(), output_dir / "generation_model" / "pytorch_model.bin")

    test_results = trainer.predict(test_dataset)
    metrics = test_results.metrics

    predictions_tensor = torch.from_numpy(test_results.predictions)
    batch_size = predictions_tensor.shape[0]
    num_beams = predictions_tensor.shape[1]
    generated_ids_reshaped = predictions_tensor.view(batch_size, num_beams, -1)[:, :, 1:]
    predictions = []
    for user_sequences in generated_ids_reshaped:
        seen = set()
        item_ids = []
        for seq in user_sequences:
            tokens_tuple = tuple(seq.tolist())
            item_id = tokenizer.tokens2item.get(tokens_tuple, None)
            if item_id is None or item_id in seen:
                continue
            seen.add(item_id)
            item_ids.append(int(item_id))
        predictions.append(item_ids)

    item_popularity = compute_dataset_item_popularity(args.data_interaction_files, shift_item_id=0)
    popularity_metrics = compute_prediction_popularity_metrics(
        predictions,
        item_popularity,
        k_list=[1, 5, 10],
    )
    token_popularity_metrics = compute_token_popularity_metrics(
        tokenizer.item2tokens,
        item_popularity,
    )
    metrics.update({f"test_{key}": value for key, value in popularity_metrics.items()})
    metrics.update({f"test_{key}": value for key, value in token_popularity_metrics.items()})

    save_json(
        {
            "model": args.backbone,
            "dataset": Path(args.data_interaction_files).parent.name,
            "output_dir": str(output_dir.resolve()),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "metrics": {key: float(value) for key, value in metrics.items()},
            "popularity_metrics": {key: float(value) for key, value in popularity_metrics.items()},
            "token_popularity_metrics": {key: float(value) for key, value in token_popularity_metrics.items()},
            "config": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "num_train_epochs": args.num_train_epochs,
                "num_beams": args.num_beams,
                "max_gen_length": args.max_gen_length,
                "max_k": args.max_k,
                "seed": args.seed,
            },
        },
        output_dir / "final_metrics.json",
    )


if __name__ == "__main__":
    main()
