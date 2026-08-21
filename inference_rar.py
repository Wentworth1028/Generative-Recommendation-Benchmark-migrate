from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments

from genrec.data.collators.generative.genplugin_collator import GenPluginDataCollator
from genrec.data.datasets.generative.genplugin_dataset import GenPluginDataset
from genrec.genplugin.data_utils import save_json
from genrec.genplugin.tokenizer_utils import load_semantic_id_tokenizer
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


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.is_dir():
        path = path / "pytorch_model.bin"
    return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GENPLUGIN RAR inference.")
    parser.add_argument("--data_interaction_files", required=True)
    parser.add_argument("--data_text_files", required=True)
    parser.add_argument("--item2tokens_path", required=True)
    parser.add_argument("--text_embedding_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--retrieval_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", choices=["tiger", "letter"], default="tiger")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
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

    dataset_kwargs = dict(
        data_interaction_files=args.data_interaction_files,
        data_text_files=args.data_text_files,
        tokenizer=tokenizer,
        config=model_config,
    )
    test_dataset = GenPluginDataset(mode="test", **dataset_kwargs)

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

    state_dict = _load_state_dict(Path(args.checkpoint_path))
    model.load_state_dict(state_dict, strict=False)
    model.freeze_for_rar()

    cache_root = Path(args.cache_dir)
    retrieval_root = Path(args.retrieval_dir)
    collator = GenPluginDataCollator(
        max_item_seq_len=args.max_seq_len,
        sid_length=tokenizer.digits,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        text_pad_token_id=model.text_sentinel_index,
        stage="rar",
        retrieval_top_k=30,
        user_embedding_path={"test": cache_root / "test" / "user_emb_mean.npy"},
        reranked_user_index_path={"test": retrieval_root / "test" / "reranked_user_index.json"},
    )

    trainer = GenPluginTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "prediction_tmp"),
            per_device_eval_batch_size=args.batch_size,
            report_to=[],
            do_train=False,
            do_eval=False,
            remove_unused_columns=False,
        ),
        train_dataset=None,
        eval_dataset=test_dataset,
        data_collator=collator,
        callbacks=[],
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
            "metrics": {key: float(value) for key, value in metrics.items()},
            "popularity_metrics": {key: float(value) for key, value in popularity_metrics.items()},
            "token_popularity_metrics": {key: float(value) for key, value in token_popularity_metrics.items()},
            "config": {
                "num_beams": args.num_beams,
                "max_gen_length": args.max_gen_length,
                "max_k": args.max_k,
                "seed": args.seed,
            },
        },
        output_dir / "final_metrics.json",
    )
    save_json(predictions, output_dir / "predictions.json")


if __name__ == "__main__":
    main()
