from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments

from genrec.data.collators.generative.genplugin_collator import GenPluginDataCollator
from genrec.data.datasets.generative.genplugin_dataset import GenPluginDataset
from genrec.genplugin.data_utils import masked_mean, save_json
from genrec.genplugin.tokenizer_utils import load_semantic_id_tokenizer
from genrec.trainers.generative.genplugin_trainer import GenPluginTrainer
from genrec.utils.common_utils import set_seed
from genrec.utils.logging_utils import setup_logging
from genrec.utils.models_setup.genplugin_setup import create_genplugin_letter_model, create_genplugin_tiger_model


class UpdateEpochCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            target = model.module if hasattr(model, "module") else model
            target.current_epoch = int(state.epoch or 0)


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


def _build_datasets(
    tokenizer,
    data_interaction_files: str,
    data_text_files: str,
    model_config: dict,
):
    dataset_kwargs = dict(
        data_interaction_files=data_interaction_files,
        data_text_files=data_text_files,
        tokenizer=tokenizer,
        config=model_config,
    )
    train_dataset = GenPluginDataset(mode="train", **dataset_kwargs)
    valid_dataset = GenPluginDataset(mode="valid", **dataset_kwargs)
    test_dataset = GenPluginDataset(mode="test", **dataset_kwargs)
    return train_dataset, valid_dataset, test_dataset


def _export_split_cache(
    model,
    dataset,
    collator,
    output_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> None:
    output_dir = output_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    max_index = len(dataset)
    hidden_dim = model.id_model.config.d_model
    user_cache = np.zeros((max_index, hidden_dim), dtype=np.float32)
    text_cache = np.zeros((max_index, hidden_dim), dtype=np.float32)
    decoder_cache = np.zeros((max_index, hidden_dim), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = _recursive_to_device(batch, device)
            sample_index = batch["index"].detach().cpu().numpy()
            id_outputs = model.encode_id(batch["input_ids"], batch["attention_mask"])
            text_features = model._maybe_lookup_text_features(
                text_input_ids=batch["text_input_ids"],
            )
            text_outputs = model.encode_text(text_features, batch["text_attention_mask"])

            decoder_input_ids = model._shift_right(batch["labels"])
            decoder_attention_mask = decoder_input_ids.ne(model.config.pad_token_id).long()
            decoder_outputs = model.decode(
                target_tokens=decoder_input_ids,
                encoder_memory=id_outputs.last_hidden_state,
                encoder_attention_mask=batch["attention_mask"],
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            user_mean = masked_mean(
                id_outputs.last_hidden_state[:, :-1, :],
                batch["attention_mask"][:, :-1],
            )
            text_mean = masked_mean(
                text_outputs.last_hidden_state[:, :-1, :],
                batch["text_attention_mask"][:, :-1],
            )
            decoder_hidden = decoder_outputs.decoder_hidden_states[-1][:, 1:, :]
            decoder_mask = batch["labels"][:, :-1].ne(-100)
            decoder_mean = masked_mean(decoder_hidden, decoder_mask)

            for offset, index in enumerate(sample_index.tolist()):
                user_cache[index] = user_mean[offset].detach().cpu().numpy()
                text_cache[index] = text_mean[offset].detach().cpu().numpy()
                decoder_cache[index] = decoder_mean[offset].detach().cpu().numpy()

    np.save(output_dir / "user_emb_mean.npy", user_cache)
    np.save(output_dir / "text_user_emb_mean.npy", text_cache)
    np.save(output_dir / "decoder_item_emb_mean.npy", decoder_cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain GENPLUGIN.")
    parser.add_argument("--data_interaction_files", required=True)
    parser.add_argument("--data_text_files", required=True)
    parser.add_argument("--item2tokens_path", required=True)
    parser.add_argument("--text_embedding_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", choices=["tiger", "letter"], default="tiger")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
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
        "genplugin_stage": "pretrain",
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

    if model.text_sentinel_index is None:
        raise ValueError("The text embedding table must be available to create the text sentinel token.")

    train_batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_train_batch_size = max(1, train_batch_size // world_size)
    per_device_eval_batch_size = max(1, eval_batch_size // world_size)

    train_collator = GenPluginDataCollator(
        max_item_seq_len=args.max_seq_len,
        sid_length=tokenizer.digits,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        text_pad_token_id=model.text_sentinel_index,
        stage="pretrain",
    )

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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            UpdateEpochCallback(),
        ],
        compute_metrics=None,
        generation_params={},
        item2tokens=tokenizer.item2tokens,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        vocab_size=tokenizer.vocab_size,
        inference_mode=None,
        do_generate=False,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "generation_model"))
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
    torch.save(unwrapped_model.state_dict(), output_dir / "generation_model" / "pytorch_model.bin")

    device = trainer.accelerator.device if hasattr(trainer, "accelerator") else next(model.parameters()).device
    export_collator = GenPluginDataCollator(
        max_item_seq_len=args.max_seq_len,
        sid_length=tokenizer.digits,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        text_pad_token_id=model.text_sentinel_index,
        stage="pretrain",
    )
    _export_split_cache(unwrapped_model, train_dataset, export_collator, cache_dir, "train", per_device_eval_batch_size, device)
    _export_split_cache(unwrapped_model, valid_dataset, export_collator, cache_dir, "valid", per_device_eval_batch_size, device)
    _export_split_cache(unwrapped_model, test_dataset, export_collator, cache_dir, "test", per_device_eval_batch_size, device)

    save_json(
        {
            "backbone": args.backbone,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_train_epochs": args.num_train_epochs,
            "warmup_ratio": args.warmup_ratio,
            "early_stopping_patience": args.early_stopping_patience,
            "max_seq_len": args.max_seq_len,
            "text_embedding_path": args.text_embedding_path,
            "item2tokens_path": args.item2tokens_path,
        },
        output_dir / "pretrain_config.json",
    )


if __name__ == "__main__":
    main()
