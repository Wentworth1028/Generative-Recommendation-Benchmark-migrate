from __future__ import annotations

from typing import Optional, Dict, List, Any, Union
import math

import torch
import torch.nn as nn
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedModel

from genrec.generation.trie import Trie, prefix_allowed_tokens_fn
from genrec.trainers.generative.base_trainer import BaseGenerativeTrainer


class FastTrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, vocab_size: int):
        self.trie = trie
        self.vocab_size = vocab_size
        self.tensor_mask_cache = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seqs_cpu = input_ids.tolist()
        current_device = scores.device

        for i, seq in enumerate(seqs_cpu):
            seq_tuple = tuple(seq)
            if seq_tuple not in self.tensor_mask_cache:
                allowed_tokens = self.trie.get(seq)
                node_mask = torch.full((self.vocab_size,), -math.inf, device=current_device)
                if allowed_tokens:
                    node_mask[allowed_tokens] = 0.0
                self.tensor_mask_cache[seq_tuple] = node_mask
            scores[i, :] += self.tensor_mask_cache[seq_tuple]
        return scores


class GenPluginTrainer(BaseGenerativeTrainer):
    """Trainer for GENPLUGIN pretraining and RAR fine-tuning."""

    def __init__(
        self,
        model,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        callbacks=None,
        compute_metrics=None,
        generation_params: Optional[Dict] = None,
        item2tokens: Optional[Dict] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        optimizers=(None, None),
        vocab_size: Optional[int] = None,
        inference_mode: Optional[str] = None,
        do_generate: bool = True,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            generation_params=generation_params,
            item2tokens=item2tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            optimizers=optimizers,
            vocab_size=vocab_size,
            inference_mode=inference_mode,
        )
        self.do_generate = do_generate
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            if self.inference_mode == "CBS":
                self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
            if self.inference_mode == "FastCBS":
                trie_processor = FastTrieLogitsProcessor(self.candidate_trie, self.vocab_size)
                self.processors = LogitsProcessorList([trie_processor])
        else:
            self.candidate_trie = None
            self.prefix_allowed_fn = None

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("item_id", inputs.get("label_id"))

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.mean().detach() if getattr(outputs, "loss", None) is not None else torch.tensor(0.0)

        if prediction_loss_only or not self.do_generate:
            return (loss, None, None)

        device = self.accelerator.device if hasattr(self, "accelerator") else next(model.parameters()).device
        encoder_input_ids = inputs["input_ids"].to(device)
        encoder_attention_mask = inputs["attention_mask"].to(device)
        user_rag_emb = inputs.get("user_rag_emb")
        if user_rag_emb is not None:
            user_rag_emb = user_rag_emb.to(device)

        gen_kwargs = {
            "max_length": self.generation_params.get("max_gen_length", 5),
            "num_beams": self.generation_params.get("num_beams", 10),
            "num_return_sequences": self.generation_params.get("max_k", 10),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }

        if hasattr(self, "accelerator"):
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model

        generate_kwargs = {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "user_rag_emb": user_rag_emb,
            **gen_kwargs,
        }
        if self.inference_mode == "FastCBS":
            generated_sequences = unwrapped_model.generate(
                logits_processor=self.processors,
                **generate_kwargs,
            )
        elif self.inference_mode == "CBS":
            generated_sequences = unwrapped_model.generate(
                prefix_allowed_tokens_fn=self.prefix_allowed_fn,
                **generate_kwargs,
            )
        else:
            generated_sequences = unwrapped_model.generate(**generate_kwargs)

        batch_size = encoder_input_ids.shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        return (loss, generated_ids_reshaped, labels)

