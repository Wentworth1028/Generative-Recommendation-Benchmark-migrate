from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from genrec.models.LETTER.LETTER import LETTERT5ForConditionalGeneration
from genrec.models.TIGER.TIGER import TIGER


class GhostAuoMixin:
    """Shared AUO helpers for the GHOST variants."""

    def _init_ghost_mixin(self, config) -> None:
        self.ghost_variant = str(getattr(config, "ghost_variant", "ghost")).lower()
        self.auo_alpha = float(getattr(config, "auo_alpha", 0.1))
        self.auo_ka = int(getattr(config, "auo_ka", 200))
        self.auo_kb = int(getattr(config, "auo_kb", 5))
        self.auo_temperature = float(getattr(config, "auo_temperature", 1.0))
        self.ghost_collection_path = getattr(config, "ghost_collection_path", None)
        self.ghost_collection = self._load_ghost_collection(self.ghost_collection_path)

    @staticmethod
    def _load_ghost_collection(path: Optional[str]) -> dict[str, Any]:
        if not path:
            return {}
        file_path = Path(path)
        if not file_path.exists():
            return {}
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = float(getattr(self, "temperature", 1.0))
        if temperature != 1.0:
            logits = logits / temperature
        return logits

    def _decode_logits_from_labels(
        self,
        *,
        encoder_memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        decoder_input_ids = self._shift_right(labels)
        decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=encoder_memory,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=None,
        )
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(sequence_output)
        return self._scale_logits(lm_logits)

    def _candidate_sequences_for_batch(
        self,
        item_ids: Optional[torch.Tensor],
        target_is_tail: Optional[torch.Tensor],
    ) -> tuple[list[int], list[list[int]]]:
        if item_ids is None:
            return [], []

        batch_indices: list[int] = []
        candidate_sequences: list[list[int]] = []
        item_ids_list = item_ids.detach().to("cpu").tolist()
        tail_mask = (
            target_is_tail.detach().to("cpu").tolist()
            if target_is_tail is not None
            else [1] * len(item_ids_list)
        )

        for batch_index, (item_id, is_tail) in enumerate(zip(item_ids_list, tail_mask)):
            if not bool(is_tail):
                continue
            entry = self.ghost_collection.get(str(int(item_id))) or self.ghost_collection.get(int(item_id))
            if not entry:
                continue
            head_sids = entry.get("head_sids", [])
            for sid in head_sids[: self.auo_kb]:
                sequence = [int(token) for token in sid]
                sequence.append(int(self.config.eos_token_id))
                batch_indices.append(batch_index)
                candidate_sequences.append(sequence)
        return batch_indices, candidate_sequences

    def _pad_candidate_labels(self, candidate_sequences: list[list[int]], device: torch.device) -> torch.LongTensor:
        if not candidate_sequences:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        max_len = max(len(sequence) for sequence in candidate_sequences)
        padded = torch.full((len(candidate_sequences), max_len), -100, dtype=torch.long, device=device)
        for row_idx, sequence in enumerate(candidate_sequences):
            padded[row_idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        return padded

    def _compute_auo_loss(
        self,
        *,
        encoder_memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        item_ids: Optional[torch.Tensor],
        target_is_tail: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.auo_alpha <= 0.0 or not self.ghost_collection:
            return encoder_memory.new_tensor(0.0)

        batch_indices, candidate_sequences = self._candidate_sequences_for_batch(item_ids, target_is_tail)
        if not candidate_sequences:
            return encoder_memory.new_tensor(0.0)

        index_tensor = torch.tensor(batch_indices, dtype=torch.long, device=encoder_memory.device)
        expanded_memory = encoder_memory.index_select(0, index_tensor)
        expanded_mask = encoder_attention_mask.index_select(0, index_tensor)
        candidate_labels = self._pad_candidate_labels(candidate_sequences, encoder_memory.device)
        logits = self._decode_logits_from_labels(
            encoder_memory=expanded_memory,
            encoder_attention_mask=expanded_mask,
            labels=candidate_labels,
        )
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(dim=-1, index=candidate_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        unlikelihood = -torch.log1p(-target_probs.clamp(max=1.0 - 1e-7))
        mask = candidate_labels.ne(-100)
        if not mask.any():
            return encoder_memory.new_tensor(0.0)
        return (unlikelihood * mask).sum() / mask.sum().clamp_min(1)

    def _augment_with_auo(
        self,
        outputs: Seq2SeqLMOutput,
        *,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        item_id: Optional[torch.LongTensor] = None,
        target_is_tail: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        if labels is None:
            return outputs
        if attention_mask is None and input_ids is not None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        if attention_mask is None:
            raise ValueError("Ghost models require attention_mask when labels are provided.")

        base_loss = outputs.loss
        auo_loss = self._compute_auo_loss(
            encoder_memory=outputs.encoder_last_hidden_state,
            encoder_attention_mask=attention_mask,
            item_ids=item_id,
            target_is_tail=target_is_tail,
        )
        total_loss = base_loss + self.auo_alpha * auo_loss if base_loss is not None else self.auo_alpha * auo_loss
        outputs.loss = total_loss
        outputs.loss_nll = base_loss.detach() if base_loss is not None else None
        outputs.loss_auo = auo_loss.detach()
        outputs.loss_total = total_loss.detach()
        return outputs

    def _expand_for_beam_search(self, tensor: torch.Tensor, beam_count: int) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(1).expand(-1, beam_count, -1).reshape(tensor.size(0) * beam_count, tensor.size(1))
        if tensor.dim() == 3:
            return tensor.unsqueeze(1).expand(-1, beam_count, -1, -1).reshape(
                tensor.size(0) * beam_count, tensor.size(1), tensor.size(2)
            )
        raise ValueError(f"Unsupported tensor rank for beam expansion: {tensor.dim()}")

    @torch.no_grad()
    def _ghost_generate(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Any] = None,
        logits_processor=None,
        prefix_allowed_tokens_fn=None,
        max_length: int = 5,
        num_beams: int = 10,
        num_return_sequences: int = 10,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = attention_mask
        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, "last_hidden_state") else encoder_outputs[0]
            encoder_attention_mask = attention_mask

        device = input_ids.device
        batch_size = input_ids.size(0)
        bos_id = int(self.config.decoder_start_token_id)
        pad_token_id = int(self.config.pad_token_id if pad_token_id is None else pad_token_id)
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        num_beams = max(1, int(num_beams))
        num_return_sequences = min(max(1, int(num_return_sequences)), num_beams)
        max_steps = max(1, int(max_length) - 1)

        sequences = torch.full((batch_size, 1, 1), bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros((batch_size, 1), dtype=torch.float, device=device)
        finished = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

        for _ in range(max_steps):
            beam_count = sequences.size(1)
            flat_sequences = sequences.reshape(batch_size * beam_count, -1)
            flat_memory = self._expand_for_beam_search(encoder_hidden_states, beam_count)
            flat_mask = self._expand_for_beam_search(encoder_attention_mask, beam_count)
            decoder_input_ids = flat_sequences
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=None,
                inputs_embeds=None,
                past_key_values=None,
                encoder_hidden_states=flat_memory,
                encoder_attention_mask=flat_mask,
                head_mask=None,
                cross_attn_head_mask=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=None,
            )
            sequence_output = decoder_outputs[0]
            if self.config.tie_word_embeddings:
                sequence_output = sequence_output * (self.model_dim**-0.5)
            step_logits = self.lm_head(sequence_output)
            step_scores = F.log_softmax(self._scale_logits(step_logits)[:, -1, :], dim=-1)

            finished_flat = finished.reshape(-1)
            if (~finished_flat).any():
                active_indices = torch.nonzero(~finished_flat, as_tuple=False).flatten()
                active_sequences = flat_sequences.index_select(0, active_indices)
                active_scores = step_scores.index_select(0, active_indices)
                if prefix_allowed_tokens_fn is not None:
                    constrained = torch.full_like(active_scores, -math.inf)
                    for local_idx, sequence in enumerate(active_sequences):
                        flat_index = int(active_indices[local_idx].item())
                        batch_id = flat_index // beam_count
                        allowed_tokens = prefix_allowed_tokens_fn(batch_id, sequence)
                        if not allowed_tokens:
                            allowed_tokens = [eos_token_id]
                        constrained[local_idx, allowed_tokens] = active_scores[local_idx, allowed_tokens]
                    active_scores = constrained
                if logits_processor is not None:
                    active_scores = logits_processor(active_sequences, active_scores)
                step_scores = torch.full_like(step_scores, -math.inf)
                step_scores.index_copy_(0, active_indices, active_scores)

            if finished_flat.any():
                step_scores[finished_flat] = -math.inf
                step_scores[finished_flat, pad_token_id] = 0.0

            total_scores = step_scores + beam_scores.reshape(-1, 1)
            total_scores = total_scores.view(batch_size, beam_count * total_scores.size(-1))
            next_scores, next_tokens = torch.topk(total_scores, k=num_beams, dim=-1)
            vocab_size = step_scores.size(-1)
            next_beam_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_token_ids = torch.remainder(next_tokens, vocab_size)

            gathered_sequences = sequences.gather(
                dim=1,
                index=next_beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1)),
            )
            sequences = torch.cat([gathered_sequences, next_token_ids.unsqueeze(-1)], dim=-1)
            beam_scores = next_scores
            finished = finished.gather(dim=1, index=next_beam_indices) | next_token_ids.eq(eos_token_id)

            if finished.all():
                break

        return sequences[:, :num_return_sequences, :].reshape(batch_size * num_return_sequences, -1)


class GhostTIGER(GhostAuoMixin, TIGER):
    def __init__(self, config):
        super().__init__(config)
        self._init_ghost_mixin(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        item_id: Optional[torch.LongTensor] = None,
        target_is_tail: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = TIGER.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        if not (return_dict if return_dict is not None else self.config.use_return_dict):
            return outputs
        return self._augment_with_auo(
            outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            item_id=item_id,
            target_is_tail=target_is_tail,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self._ghost_generate(*args, **kwargs)


class GhostLETTER(GhostAuoMixin, LETTERT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self._init_ghost_mixin(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        item_id: Optional[torch.LongTensor] = None,
        target_is_tail: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = LETTERT5ForConditionalGeneration.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        if not (return_dict if return_dict is not None else self.config.use_return_dict):
            return outputs
        return self._augment_with_auo(
            outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            item_id=item_id,
            target_is_tail=target_is_tail,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self._ghost_generate(*args, **kwargs)
