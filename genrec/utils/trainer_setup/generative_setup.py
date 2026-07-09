# genrec/utils/trainer_setup/generative/generative_setup.py

from typing import Optional, Dict, List
from collections import Counter
from functools import partial
from transformers import TrainingArguments, EarlyStoppingCallback
from hydra.utils import instantiate
from omegaconf import DictConfig

from genrec.utils.metrics import compute_metrics
from genrec.utils.callbacks.generative.generative_callback import (
    GenerativeLoggingCallback,
    EvaluateEveryNEpochsCallback,
    DelayedEvaluateEveryNEpochsCallback,
)


def compute_train_target_item_popularity(train_dataset) -> Dict[int, int]:
    counter = Counter()
    for sample in getattr(train_dataset, "samples", []):
        target_item = sample.get("target_item")
        if target_item is not None:
            counter[int(target_item)] += 1
    return dict(counter)


def setup_training(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    model_config,
    generative_config: DictConfig,
    output_dirs,
    logger,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    train_data_collator,
    vocab_size: Optional[int] = None,
):

    training_args = TrainingArguments(
        lr_scheduler_type="cosine",
        output_dir=output_dirs['model'],
        num_train_epochs=model_config['num_epochs'],
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=output_dirs['logs'],
        logging_strategy="epoch",
        report_to=[],
        warmup_ratio=model_config["warmup_ratio"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        metric_for_best_model="ndcg@10",
        greater_is_better=True,
        seed=model_config.get("seed", 42),
        data_seed=model_config.get("seed", 42),
    )

    tokens_to_item_map = tokenizer.tokens2item
    compute_metrics_with_map = partial(compute_metrics, tokens_to_item_map=tokens_to_item_map)

    num_beams = model_config.get('num_beams', 10)
    max_gen_length = model_config.get('max_gen_length', 5)
    k_list = model_config.get('k_list', [5, 10, 20])
    max_k = k_list[-1] if k_list else 10

    generation_params = {
        'max_gen_length': max_gen_length,
        'num_beams': num_beams,
        'max_k': max_k,
        'prefix_popularity_penalty': model_config.get('prefix_popularity_penalty', 0.0),
        'prefix_popularity_transform': model_config.get('prefix_popularity_transform', 'log1p'),
        'prefix_popularity_positive_only': model_config.get('prefix_popularity_positive_only', True),
        'prefix_popularity_max_length': model_config.get('prefix_popularity_max_length', None),
        'prefix_popularity_eps': model_config.get('prefix_popularity_eps', 1e-8),
    }
    item_popularity = compute_train_target_item_popularity(train_dataset)

    # ===== CallBacks =====
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=model_config.get("early_stop_upper_steps", 1000)),
        GenerativeLoggingCallback(logger),
        # start_epoch means when to start evaluate
        DelayedEvaluateEveryNEpochsCallback(n_epochs=model_config.get("evaluation_epoch", 5), start_epoch=40),
    ]

    trainer_partial = instantiate(generative_config.trainer)

    trainer = trainer_partial(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=train_data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics_with_map,
        generation_params=generation_params,
        item2tokens=tokenizer.item2tokens,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        vocab_size=vocab_size,
        inference_mode=model_config["inference_mode"],
        item_popularity=item_popularity,
    )

    return trainer
