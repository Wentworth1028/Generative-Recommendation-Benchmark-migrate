# train_with_generative.py

import os
import torch
import json
from torch.utils.data import DataLoader
from datetime import datetime
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf

from genrec.quantization.pipelines.rqvae_pipeline import RQVAETrainingPipeline
from genrec.quantization.tokenizers.rqvae_tokenizer import RQVAETokenizer
from genrec.data.collators.generative.tiger_collator import TigerDataCollator
from genrec.utils.nni_utils import get_nni_params, update_config_with_nni
from genrec.utils.common_utils import set_seed
from genrec.utils.logging_utils import setup_logging
from genrec.utils.factory import get_model_factory, get_dataset_class, get_collator_class, get_pipeline_class
from genrec.utils.trainer_setup.generative_setup import setup_training
from genrec.utils.popularity_metrics import compute_dataset_item_popularity, compute_prediction_popularity_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_output_directories(base_output_dir: str = "./output"):

    if "NNI_PLATFORM" in os.environ:
        nni_output_dir = os.environ["NNI_OUTPUT_DIR"]
        dirs = {
            'base': base_output_dir,
            'tokenizer': os.path.join(base_output_dir, 'tokenizer_model'),
            'model': os.path.join(base_output_dir, 'generation_model'),
            'checkpoints': os.path.join(base_output_dir, 'checkpoints'),
            'logs': os.path.join(nni_output_dir, 'logs'),
        }
    else:
        dirs = {
            'base': base_output_dir,
            'tokenizer': os.path.join(base_output_dir, 'tokenizer_model'),
            'model': os.path.join(base_output_dir, 'generation_model'),
            'checkpoints': os.path.join(base_output_dir, 'checkpoints'),
            'logs': os.path.join(base_output_dir, 'logs'),
        }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def stage1_train_tokenizer(
    rqvae_config: dict, output_dirs: dict, gen_type: str, force_retrain: bool = False, accelerator=None
):
    print("\n" + "=" * 60)
    print("RQ-VAE Tokenizer")
    print("=" * 60)

    tokenizer_checkpoint = rqvae_config['checkpoint_path']
    item2tokens_path = rqvae_config['save_path']

    if not force_retrain and os.path.exists(item2tokens_path):
        print(f"exist tokenizer checkpoint: {tokenizer_checkpoint}")
        print("skip tokenizer training...")
        return True

    required_files = [rqvae_config['data_text_files'], rqvae_config['interaction_files']]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"not exist: {file_path}")
            return False

    try:
        PipelineClass = get_pipeline_class(gen_type)
        pipeline = PipelineClass(rqvae_config, accelerator=accelerator)
        pipeline.run()
        return True
    except Exception as e:
        print(f"RQ-VAE tokenizer train false: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def stage2_train_generation_model(
    model_config, rqvae_config, generative_config: DictConfig, output_dirs, accelerator, logger, force_retrain=False
):

    if accelerator.is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("training generation model")
        logger.info("=" * 60)

    model_save_path = model_config['model_save_path']
    do_inference_only = (not force_retrain) and os.path.exists(model_save_path)

    if do_inference_only and accelerator.is_main_process:
        logger.info(f"found existing model_save_path: {model_save_path}")
        logger.info("will load model and run inference only (skip training).")

    tokenizer_items2tokens_path = os.path.join(output_dirs['tokenizer'], 'item2tokens.json')
    if not os.path.exists(tokenizer_items2tokens_path):
        if accelerator.is_main_process:
            logger.info(f"Error: Not Found: {tokenizer_items2tokens_path}")
        return False

    if accelerator.is_main_process:
        logger.info("-" * 40)
        logger.info("🚀 parameters:")
        logger.info(f"   - Learning Rate: {model_config.get('learning_rate')}")
        logger.info(f"   - Weight Decay:  {model_config.get('weight_decay')}")
        logger.info(f"   - Batch Size:    {model_config.get('batch_size')}")
        logger.info(f"   - Num Epochs:    {model_config.get('num_epochs')}")
        logger.info(f"   - Seed:          {model_config.get('seed')}")
        logger.info(f"   - Inference:     {model_config.get('inference_mode')}")
        logger.info("-" * 40)
    if accelerator.is_main_process:
        logger.info(f"loading Tokenizer...")
    tokenizer = RQVAETokenizer.load(rqvae_config)
    if accelerator.is_main_process:
        logger.info(f"total {len(tokenizer.item2tokens)} item")
        logger.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    gen_type = generative_config.type
    use_user_tokens = model_config['use_user_tokens']
    if accelerator.is_main_process:
        logger.info(f"generative mode type: {gen_type}")
        logger.info(f"use user tokens: {use_user_tokens}")

    create_model_fn = get_model_factory(gen_type)
    vocab_size = tokenizer.vocab_size if use_user_tokens else tokenizer.vocab_size - tokenizer.num_user_tokens
    if do_inference_only:
        if accelerator.is_main_process:
            logger.info(f"loading hf model from dir: {output_dirs['model']}")
        # model = create_model_fn(vocab_size=vocab_size, model_config=model_config)
        # model.load_state_dict(torch.load(model_save_path, map_location='cpu'), strict=False)
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(output_dirs['model'])
    else:
        model = create_model_fn(vocab_size=vocab_size, model_config=model_config)
    # if use_user_tokens:
    #     model = create_tiger_model(
    #     vocab_size=tokenizer.vocab_size,
    #     model_config=model_config,
    #     )
    # else:
    #     model = create_tiger_model(
    #         vocab_size=tokenizer.vocab_size - tokenizer.num_user_tokens,
    #         model_config=model_config,
    #     )

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"model parameters: {total_params:,}")

    DatasetClass = get_dataset_class(gen_type)

    train_dataset = DatasetClass(
        data_interaction_files=model_config['data_interaction_files'],
        data_text_files=model_config['data_text_files'],
        tokenizer=tokenizer,
        config=model_config,
        mode='train',
    )
    valid_dataset = DatasetClass(
        data_interaction_files=model_config['data_interaction_files'],
        data_text_files=model_config['data_text_files'],
        tokenizer=tokenizer,
        config=model_config,
        mode='valid',
    )
    test_dataset = DatasetClass(
        data_interaction_files=model_config['data_interaction_files'],
        data_text_files=model_config['data_text_files'],
        tokenizer=tokenizer,
        config=model_config,
        mode='test',
    )

    CollatorClass = get_collator_class(gen_type)

    train_data_collator = CollatorClass(
        max_seq_len=train_dataset.max_token_len,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        mode="train",
    )
    test_data_collator = CollatorClass(
        max_seq_len=train_dataset.max_token_len,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        mode="test",
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=model_config['test_batch_size'], shuffle=False, collate_fn=test_data_collator
    )

    test_dataloader = accelerator.prepare(test_dataloader)

    train_batch_size = model_config['batch_size']
    test_batch_size = model_config['test_batch_size']
    num_devices = accelerator.num_processes

    if train_batch_size % num_devices != 0 or test_batch_size % num_devices != 0:
        if accelerator.is_main_process:
            logger.error(f"Error:  {train_batch_size} or {test_batch_size} can not divide by{num_devices}")
        return False

    per_device_train_batch_size = train_batch_size // num_devices
    per_device_eval_batch_size = test_batch_size // num_devices

    if accelerator.is_main_process:
        logger.info(f"Batch Size setting (total {num_devices} devices)")
        logger.info(f"  - training global {train_batch_size} -> one device {per_device_train_batch_size}")
        logger.info(f"  - evaluation: global {test_batch_size} -> one device {per_device_eval_batch_size}")

    trainer = setup_training(
        model,
        tokenizer,
        train_dataset,
        valid_dataset,
        model_config,
        generative_config,
        output_dirs,
        logger,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        train_data_collator=train_data_collator,
        vocab_size=vocab_size,
    )
    model.config.use_cache = False
    if do_inference_only:
        if accelerator.is_main_process:
            logger.info("skip training and run inference only.")
    else:
        trainer.train()
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("predict test set...")
    test_results = trainer.predict(test_dataset)
    if accelerator.is_main_process:
        metrics = test_results.metrics
        predictions_tensor = torch.from_numpy(test_results.predictions)
        batch_size = predictions_tensor.shape[0]
        num_beams = predictions_tensor.shape[1]
        generated_ids_reshaped = predictions_tensor.view(batch_size, num_beams, -1)[:, :, 1:]
        tiger_predictions = []
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
            tiger_predictions.append(item_ids)

        item_popularity = compute_dataset_item_popularity(model_config['data_interaction_files'], shift_item_id=0)
        popularity_metrics = compute_prediction_popularity_metrics(
            tiger_predictions,
            item_popularity,
            k_list=model_config.get("k_list", [1, 5, 10]),
        )
        metrics.update({f"test_{key}": value for key, value in popularity_metrics.items()})

        k_values = sorted(
            list(
                set(
                    int(key.split("@")[-1])
                    for key in metrics.keys()
                    if key.startswith("test_hit@") or key.startswith("test_ndcg@")
                )
            )
        )

        logger.info("=" * 30 + " test results " + "=" * 30)

        for k in k_values:
            hit_val = metrics.get(f"test_hit@{k}", 0.0)
            ndcg_val = metrics.get(f"test_ndcg@{k}", 0.0)

            logger.info(f"Hit@{k}: {hit_val:.4f}, NDCG@{k}: {ndcg_val:.4f}")
        for key, value in popularity_metrics.items():
            logger.info(f"{key}: {value:.4f}")

        logger.info("=" * 75)
        final_metrics = {
            "model": gen_type,
            "dataset": model_config.get("dataset_name"),
            "output_dir": os.path.abspath(output_dirs['base']),
            "model_dir": os.path.abspath(output_dirs['model']),
            "tokenizer_dir": os.path.abspath(output_dirs['tokenizer']),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "inference_mode": model_config.get("inference_mode"),
            "metrics": {key: float(value) for key, value in metrics.items()},
            "popularity_metrics": {key: float(value) for key, value in popularity_metrics.items()},
            "config": {
                "learning_rate": model_config.get("learning_rate"),
                "weight_decay": model_config.get("weight_decay"),
                "batch_size": model_config.get("batch_size"),
                "test_batch_size": model_config.get("test_batch_size"),
                "num_epochs": model_config.get("num_epochs"),
                "num_beams": model_config.get("num_beams"),
                "max_gen_length": model_config.get("max_gen_length"),
                "k_list": model_config.get("k_list"),
                "seed": model_config.get("seed"),
                "d_model": model_config.get("d_model"),
                "d_kv": model_config.get("d_kv"),
                "d_ff": model_config.get("d_ff"),
                "num_layers": model_config.get("num_layers"),
                "num_decoder_layers": model_config.get("num_decoder_layers"),
                "num_heads": model_config.get("num_heads"),
                "dropout_rate": model_config.get("dropout_rate"),
                "tie_word_embeddings": model_config.get("tie_word_embeddings"),
            },
        }
        final_metrics_path = os.path.join(output_dirs['base'], "final_metrics.json")
        with open(final_metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Final metrics saved to: {final_metrics_path}")

    if (not do_inference_only) and ("NNI_PLATFORM" not in os.environ):
        trainer.save_model(output_dirs['model'])

    if accelerator.is_main_process:
        logger.info("Evaluation Finish!")

    return True


@hydra.main(version_base=None, config_path="config", config_name="generative")
def main(cfg: DictConfig):

    seed = getattr(cfg, 'seed', 42)
    set_seed(seed)

    if "NNI_PLATFORM" in os.environ:
        nni_params = get_nni_params()
        cfg = update_config_with_nni(cfg, nni_params)

    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device
    logger = None

    output_dirs = setup_output_directories(cfg.output_dir)
    if accelerator.is_main_process:
        logger = setup_logging(output_dirs['logs'])
        logger.info(f"output_dirs: {output_dirs['base']}")
        logger.info(f"dataset: {cfg.dataset}")
        logger.info(f"output_dir: {cfg.output_dir}")
        logger.info(f"{accelerator.num_processes} processes")
        logger.info(f"start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{output_dirs['logs']}")

    success = True

    rqvae_config = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    rqvae_config['device'] = device
    rqvae_config['tokenizer_path'] = os.path.join(output_dirs['tokenizer'], 'tokenizer.pkl')
    rqvae_config['save_path'] = os.path.join(output_dirs['tokenizer'], 'item2tokens.json')
    rqvae_config['checkpoint_path'] = os.path.join(output_dirs['tokenizer'], 'tokenizer_checkpoint.pth')

    if not cfg.skip_tokenizer:
        tokenizer_success = stage1_train_tokenizer(
            rqvae_config,
            output_dirs,
            gen_type=cfg.tokenizer_type,
            force_retrain=cfg.force_retrain_tokenizer,
            accelerator=accelerator,
        )
        if not tokenizer_success:
            if accelerator.is_main_process:
                logger.info("Tokenizer train error")
            return
        success = success and tokenizer_success
        accelerator.wait_for_everyone()
    elif accelerator.is_main_process:
        logger.info("skip tokenizer training")

    if not cfg.skip_model and success:

        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model_config['device'] = device
        model_config['dataset_name'] = cfg.dataset
        # model_config['model_save_path'] = os.path.join(output_dirs['model'], f"{cfg.dataset}_final_model.pt")
        model_config['model_save_path'] = output_dirs['model']
        model_config['checkpoint_dir'] = output_dirs['checkpoints']
        model_config['seed'] = seed

        model_success = stage2_train_generation_model(
            model_config,
            rqvae_config,
            cfg.generative,
            output_dirs,
            accelerator,
            force_retrain=cfg.force_retrain_model,
            logger=logger,
        )
        success = success and model_success
    elif cfg.skip_model and accelerator.is_main_process:
        logger.info("skip train")

    if accelerator.is_main_process:
        logger.info("\n" + "=" * 60)
        if success:
            logger.info("Finish Train!")
            logger.info(f"checkpoint : {output_dirs['base']}")
        else:
            logger.info("Error")
        logger.info(f"Finish Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
    accelerator.wait_for_everyone()

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
