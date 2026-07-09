# genrec/trainers/generative/tiger_trainer.py

from typing import Optional, Dict, List, Any, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .base_trainer import BaseGenerativeTrainer
from genrec.generation.trie import Trie, prefix_allowed_tokens_fn
import math
from transformers import LogitsProcessor
from transformers import LogitsProcessorList

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


class PrefixPopularityLogitsProcessor(LogitsProcessor):
    """Apply an inference-time penalty to over-popular SID prefixes."""

    def __init__(
        self,
        trie,
        prefix_bias: Dict[tuple, float],
        penalty: float,
        positive_only: bool = True,
    ):
        self.trie = trie
        self.prefix_bias = prefix_bias
        self.penalty = float(penalty)
        self.positive_only = bool(positive_only)
        self.tensor_penalty_cache = {}

    def _candidate_penalties(self, seq: List[int], device):
        seq_tuple = tuple(seq)
        cache_key = (seq_tuple, str(device))
        if cache_key in self.tensor_penalty_cache:
            return self.tensor_penalty_cache[cache_key]

        allowed_tokens = self.trie.get(seq)
        if not allowed_tokens:
            self.tensor_penalty_cache[cache_key] = (None, None)
            return None, None

        penalties = []
        kept_tokens = []
        for token in allowed_tokens:
            candidate_prefix = seq_tuple + (int(token),)
            bias = float(self.prefix_bias.get(candidate_prefix, 0.0))
            if self.positive_only:
                bias = max(bias, 0.0)
            penalty = self.penalty * bias
            if penalty != 0.0:
                kept_tokens.append(int(token))
                penalties.append(penalty)

        if not kept_tokens:
            self.tensor_penalty_cache[cache_key] = (None, None)
            return None, None

        token_tensor = torch.tensor(kept_tokens, dtype=torch.long, device=device)
        penalty_tensor = torch.tensor(penalties, dtype=torch.float32, device=device)
        self.tensor_penalty_cache[cache_key] = (token_tensor, penalty_tensor)
        return token_tensor, penalty_tensor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty <= 0.0 or not self.prefix_bias:
            return scores

        current_device = scores.device
        for i, seq in enumerate(input_ids.tolist()):
            token_tensor, penalty_tensor = self._candidate_penalties(seq, current_device)
            if token_tensor is not None:
                scores[i, token_tensor] -= penalty_tensor.to(dtype=scores.dtype)
        return scores

def build_prefix_popularity_bias(
    item2tokens: Dict[int, tuple],
    item_popularity: Optional[Dict[int, float]],
    transform: str = "log1p",
    max_prefix_length: Optional[int] = None,
    eps: float = 1e-8,
) -> Dict[tuple, float]:
    """Build log-ratio popularity bias for every observed SID prefix.

    Prefix keys include the decoder BOS/root token, matching the trie sequences:
    (0, token_1, ..., token_l).
    """

    if not item2tokens or not item_popularity:
        return {}

    prefix_mass_by_depth: Dict[int, Dict[tuple, float]] = {}

    def transform_popularity(value: float) -> float:
        value = max(float(value), 0.0)
        if transform == "none":
            return value
        if transform == "sqrt":
            return math.sqrt(value)
        if transform == "log1p":
            return math.log1p(value)
        raise ValueError(f"Unknown prefix popularity transform: {transform}")

    for item_id, tokens in item2tokens.items():
        if item_id not in item_popularity:
            continue
        mass = transform_popularity(item_popularity[item_id])
        if mass <= 0.0:
            continue

        token_list = list(tokens)
        effective_length = len(token_list)
        if max_prefix_length is not None and max_prefix_length > 0:
            effective_length = min(effective_length, int(max_prefix_length))

        for depth in range(1, effective_length + 1):
            prefix = tuple([0] + token_list[:depth])
            depth_mass = prefix_mass_by_depth.setdefault(depth, {})
            depth_mass[prefix] = depth_mass.get(prefix, 0.0) + mass

    prefix_bias = {}
    for depth_mass in prefix_mass_by_depth.values():
        if not depth_mass:
            continue
        mean_mass = sum(depth_mass.values()) / max(len(depth_mass), 1)
        for prefix, mass in depth_mass.items():
            prefix_bias[prefix] = math.log((mass + eps) / (mean_mass + eps))

    return prefix_bias


class TigerTrainer(BaseGenerativeTrainer):
    """
    Tiger Trainer for Generative Recommendation.
    
    Supports constrained beam search generation during evaluation.
    """
    
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
        item_popularity: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize Tiger Trainer.
        
        Args:
            model: T5 model for generation
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            callbacks: List of callbacks
            compute_metrics: Metrics computation function
            generation_params: Generation parameters (max_gen_length, num_beams, max_k)
            item2tokens: Item to tokens mapping (for constrained generation)
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID
            optimizers: Optimizer and scheduler tuple
        """
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
            inference_mode=inference_mode
        )
        
        # Build Trie for constrained generation
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            prefix_penalty = float(self.generation_params.get("prefix_popularity_penalty", 0.0) or 0.0)
            prefix_processors = []
            if self.inference_mode == "CBS":
                self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
            if self.inference_mode == "FastCBS":
                trie_processor = FastTrieLogitsProcessor(self.candidate_trie,self.vocab_size)
                prefix_processors.append(trie_processor)
            if prefix_penalty > 0.0:
                prefix_bias = build_prefix_popularity_bias(
                    self.item2tokens,
                    item_popularity,
                    transform=self.generation_params.get("prefix_popularity_transform", "log1p"),
                    max_prefix_length=self.generation_params.get("prefix_popularity_max_length"),
                    eps=float(self.generation_params.get("prefix_popularity_eps", 1e-8)),
                )
                prefix_processors.append(
                    PrefixPopularityLogitsProcessor(
                        self.candidate_trie,
                        prefix_bias,
                        penalty=prefix_penalty,
                        positive_only=bool(self.generation_params.get("prefix_popularity_positive_only", True)),
                    )
                )
            if prefix_processors:
                self.processors = LogitsProcessorList(prefix_processors)
        else:
            self.candidate_trie = None
            self.prefix_allowed_fn = None
    
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        
        loss_mask = inputs.pop("loss_mask", None)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits 
        loss = None
        if labels is not None:
            #loss_mask [batch_size, seq_len, vocab_size]
            if loss_mask is not None:
                
                masked_logits = logits.masked_fill(loss_mask == 0.0, -1e9)

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                unwrapped_model = self.accelerator.unwrap_model(model)
                loss = loss_fct(
                    masked_logits.view(-1, unwrapped_model.config.vocab_size),
                    labels.view(-1)
                )
            else:
                loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
