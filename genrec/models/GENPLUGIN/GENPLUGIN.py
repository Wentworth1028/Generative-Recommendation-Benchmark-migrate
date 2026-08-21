from __future__ import annotations

import copy
import math
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from genrec.genplugin.data_utils import load_dense_embedding_matrix, masked_mean


class GenPluginDualT5(nn.Module):
    """GENPLUGIN wrapper built on top of the existing TIGER/LETTER backbone."""

    def __init__(
        self,
        config: T5Config,
        *,
        backbone_type: str = "tiger",
        text_embedding_path: Optional[str] = None,
        text_embedding_dim: int = 4096,
        text_projection_hidden_dim: int = 2048,
        sid_length: int = 4,
        kl_temperature: float = 0.85,
        item_temperature: float = 0.9,
        kl_weight: float = 0.5,
        item_weight: float = 0.1,
        semantic_substitution_start_epoch: int = 10,
        semantic_substitution_outer_prob: float = 0.5,
        semantic_substitution_position_prob: float = 0.4,
        semantic_substitution_top_k: int = 5,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone_type = str(backbone_type).lower()
        self.sid_length = int(sid_length)
        self.kl_temperature = float(kl_temperature)
        self.item_temperature = float(item_temperature)
        self.kl_weight = float(kl_weight)
        self.item_weight = float(item_weight)
        self.semantic_substitution_start_epoch = int(semantic_substitution_start_epoch)
        self.semantic_substitution_outer_prob = float(semantic_substitution_outer_prob)
        self.semantic_substitution_position_prob = float(semantic_substitution_position_prob)
        self.semantic_substitution_top_k = int(semantic_substitution_top_k)
        self.current_epoch = 0
        self.temperature = float(getattr(config, "tau", 1.0))

        self.id_model = T5ForConditionalGeneration(config)
        self.text_model = copy.deepcopy(self.id_model)
        self.text_model.decoder = self.id_model.decoder
        self.text_model.lm_head = self.id_model.lm_head
        self.text_model.shared = self.id_model.shared

        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, text_projection_hidden_dim),
            nn.Linear(text_projection_hidden_dim, config.d_model),
        )

        self._text_embedding_table = None
        self.text_sentinel_index = None
        if text_embedding_path:
            self._text_embedding_table, self.text_sentinel_index = load_dense_embedding_matrix(text_embedding_path)
            if self._text_embedding_table.size(1) != text_embedding_dim:
                raise ValueError(
                    f"text embedding dim mismatch: expected {text_embedding_dim}, "
                    f"got {self._text_embedding_table.size(1)}"
                )

    def get_input_embeddings(self):
        return self.id_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.id_model.set_input_embeddings(value)
        self.text_model.set_input_embeddings(value)

    def _shift_right(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        return self.id_model._shift_right(input_ids)

    def freeze_for_rar(self) -> None:
        for param in self.id_model.encoder.parameters():
            param.requires_grad = False
        for param in self.text_model.encoder.parameters():
            param.requires_grad = False
        for param in self.text_projection.parameters():
            param.requires_grad = False

    @property
    def text_embedding_table(self) -> Optional[torch.Tensor]:
        return self._text_embedding_table

    def _resolve_lm_inputs(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        lm_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[Optional[torch.LongTensor], Optional[torch.Tensor], Optional[torch.LongTensor]]:
        if lm_inputs:
            input_ids = lm_inputs.get("input_ids", input_ids)
            attention_mask = lm_inputs.get("attention_mask", attention_mask)
            labels = lm_inputs.get("labels", labels)
        return input_ids, attention_mask, labels

    def _lookup_text_features(self, text_input_ids: torch.LongTensor) -> torch.FloatTensor:
        if self._text_embedding_table is None:
            raise ValueError("text_embedding_path must be provided to use GENPLUGIN text encoding.")
        flat_ids = text_input_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        features = self._text_embedding_table.index_select(0, flat_ids)
        return features.view(*text_input_ids.shape, -1)

    def _maybe_lookup_text_features(
        self,
        history_item_features: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        if history_item_features is not None:
            if torch.is_floating_point(history_item_features):
                return history_item_features
            return self._lookup_text_features(history_item_features)
        if text_input_ids is None:
            raise ValueError("text_input_ids or history_item_features must be provided.")
        return self._lookup_text_features(text_input_ids)

    def _fuse_user_rag_memory(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        user_rag_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = attention_mask.to(device=hidden_states.device)
        if user_rag_emb is None:
            return hidden_states, attention_mask
        if user_rag_emb.dim() == 2:
            user_rag_emb = user_rag_emb.unsqueeze(1)
        if user_rag_emb.size(0) != hidden_states.size(0):
            raise ValueError(
                f"user_rag_emb batch mismatch: {user_rag_emb.size(0)} vs {hidden_states.size(0)}"
            )
        rag_mask = torch.ones(
            user_rag_emb.size(0),
            user_rag_emb.size(1),
            device=hidden_states.device,
            dtype=attention_mask.dtype,
        )
        fused_hidden = torch.cat([user_rag_emb.to(hidden_states.device), hidden_states], dim=1)
        fused_mask = torch.cat([rag_mask, attention_mask], dim=1)
        return fused_hidden, fused_mask

    def encode_id(
        self,
        history_sid_tokens: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutput:
        return self.id_model.encoder(
            input_ids=history_sid_tokens,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def encode_text(
        self,
        history_item_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutput:
        projection_device = self.text_projection[0].weight.device
        projected = self.text_projection(
            history_item_features.to(device=projection_device, dtype=self.text_projection[0].weight.dtype)
        )
        attention_mask = attention_mask.to(device=projected.device)
        return self.text_model.encoder(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def _decode(
        self,
        target_tokens: Optional[torch.LongTensor],
        encoder_memory: torch.Tensor,
        optional_decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Seq2SeqLMOutput:
        decoder_outputs = self.id_model.decoder(
            input_ids=target_tokens if optional_decoder_inputs_embeds is None else None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=optional_decoder_inputs_embeds,
            past_key_values=None,
            encoder_hidden_states=encoder_memory,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            use_cache=bool(use_cache) if use_cache is not None else False,
            output_attentions=bool(output_attentions) if output_attentions is not None else False,
            output_hidden_states=bool(output_hidden_states) if output_hidden_states is not None else False,
            return_dict=True,
            cache_position=None,
        )

        sequence_output = decoder_outputs[0]
        if self.id_model.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.id_model.model_dim**-0.5)
        lm_logits = self.id_model.lm_head(sequence_output)
        if self.temperature != 1.0:
            lm_logits = lm_logits / self.temperature

        return Seq2SeqLMOutput(
            loss=None,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_memory,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def decode(
        self,
        target_tokens: Optional[torch.LongTensor],
        encoder_memory: torch.Tensor,
        optional_decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Seq2SeqLMOutput:
        return self._decode(
            target_tokens=target_tokens,
            encoder_memory=encoder_memory,
            optional_decoder_inputs_embeds=optional_decoder_inputs_embeds,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def _maybe_apply_semantic_substitution(
        self,
        decoder_inputs_embeds: torch.Tensor,
        text_logits: torch.Tensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        if not self.training:
            return decoder_inputs_embeds
        if self.current_epoch < self.semantic_substitution_start_epoch:
            return decoder_inputs_embeds
        if torch.rand((), device=decoder_inputs_embeds.device) >= self.semantic_substitution_outer_prob:
            return decoder_inputs_embeds

        token_logits = text_logits[:, :-1, :]
        topk_k = min(self.semantic_substitution_top_k, token_logits.size(-1))
        values, indices = torch.topk(token_logits, k=topk_k, dim=-1)
        denom = values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = values / denom
        topk_embeds = self.get_input_embeddings()(indices)
        predicted_embeds = (weights.unsqueeze(-1) * topk_embeds).sum(dim=-2)

        gold_embeds = decoder_inputs_embeds[:, 1:, :]
        valid_mask = labels[:, 1:].ne(-100)
        position_mask = torch.rand(valid_mask.shape, device=decoder_inputs_embeds.device) < self.semantic_substitution_position_prob
        replace_mask = valid_mask & position_mask
        gold_embeds = torch.where(replace_mask.unsqueeze(-1), predicted_embeds, gold_embeds)
        return torch.cat([decoder_inputs_embeds[:, :1, :], gold_embeds], dim=1)

    def _compute_item_contrastive_loss(
        self,
        id_hidden_states: torch.Tensor,
        id_attention_mask: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        id_hidden_states = id_hidden_states * id_attention_mask.unsqueeze(-1).to(dtype=id_hidden_states.dtype)
        text_hidden_states = text_hidden_states * text_attention_mask.unsqueeze(-1).to(dtype=text_hidden_states.dtype)
        id_hidden_states = id_hidden_states[:, :-1, :]
        id_attention_mask = id_attention_mask[:, :-1]
        text_hidden_states = text_hidden_states[:, :-1, :]
        text_attention_mask = text_attention_mask[:, :-1]

        if id_hidden_states.size(1) % self.sid_length != 0:
            raise AssertionError(
                f"Trimmed id_length={id_hidden_states.size(1)} is not divisible by sid_length={self.sid_length}"
            )
        text_len = text_hidden_states.size(1)
        if id_hidden_states.size(1) != text_len * self.sid_length:
            raise AssertionError(
                f"Trimmed id_length={id_hidden_states.size(1)} != sid_length * text_length={self.sid_length * text_len}"
            )

        batch_size, _, hidden_dim = id_hidden_states.shape
        id_item_states = id_hidden_states.reshape(batch_size, text_len, self.sid_length, hidden_dim).sum(dim=2)
        valid_mask = text_attention_mask.to(dtype=torch.bool)
        if not valid_mask.any():
            return id_hidden_states.new_tensor(0.0)

        id_flat = id_item_states[valid_mask]
        text_flat = text_hidden_states[valid_mask]
        if id_flat.size(0) <= 1:
            return id_hidden_states.new_tensor(0.0)

        similarity = torch.matmul(id_flat, text_flat.t()) / self.item_temperature
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss_id = F.cross_entropy(similarity, labels)
        loss_text = F.cross_entropy(similarity.t(), labels)
        return loss_id + loss_text

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        lm_inputs: Optional[Dict[str, torch.Tensor]] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_inputs_embeds: Optional[torch.FloatTensor] = None,
        user_rag_emb: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Any] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        mode: Optional[str] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        input_ids, attention_mask, labels = self._resolve_lm_inputs(input_ids, attention_mask, labels, lm_inputs)

        if input_ids is None or attention_mask is None:
            raise ValueError("GENPLUGIN requires input_ids and attention_mask.")
        if labels is None:
            raise ValueError("GENPLUGIN requires labels during training/evaluation.")

        stage = str(mode or getattr(self.config, "genplugin_stage", "pretrain")).lower()
        if user_rag_emb is not None:
            stage = "rar"

        if stage == "rar":
            id_outputs = self.encode_id(input_ids, attention_mask)
            fused_memory, fused_mask = self._fuse_user_rag_memory(
                id_outputs.last_hidden_state,
                attention_mask,
                user_rag_emb=user_rag_emb,
            )
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()
            decoder_outputs = self._decode(
                target_tokens=decoder_input_ids,
                encoder_memory=fused_memory,
                encoder_attention_mask=fused_mask,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
            )
            logits = decoder_outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            output = Seq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.decoder_hidden_states,
                decoder_attentions=decoder_outputs.decoder_attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=fused_memory,
                encoder_hidden_states=id_outputs.hidden_states,
                encoder_attentions=id_outputs.attentions,
            )
            output.loss_id = loss.detach()
            return output if return_dict else (loss, logits)

        if text_input_ids is None and text_inputs_embeds is None:
            raise ValueError("Pretraining requires text_input_ids or text_inputs_embeds.")
        if text_attention_mask is None:
            raise ValueError("Pretraining requires text_attention_mask.")

        id_outputs = self.encode_id(input_ids, attention_mask)
        text_features = self._maybe_lookup_text_features(text_inputs_embeds, text_input_ids)
        text_outputs = self.encode_text(text_features, text_attention_mask)

        decoder_input_ids = self._shift_right(labels)
        decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).long()

        text_decoder_outputs = self._decode(
            target_tokens=decoder_input_ids,
            encoder_memory=text_outputs.last_hidden_state,
            encoder_attention_mask=text_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        text_logits = text_decoder_outputs.logits
        text_loss = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), labels.view(-1), ignore_index=-100)

        id_decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)
        id_decoder_inputs_embeds = self._maybe_apply_semantic_substitution(
            id_decoder_inputs_embeds,
            text_logits,
            labels,
        )
        id_decoder_outputs = self._decode(
            target_tokens=None,
            encoder_memory=id_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            optional_decoder_inputs_embeds=id_decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        id_logits = id_decoder_outputs.logits
        id_loss = F.cross_entropy(id_logits.view(-1, id_logits.size(-1)), labels.view(-1), ignore_index=-100)

        id_logits_for_kl = id_logits[:, :-1, :]
        text_logits_for_kl = text_logits[:, :-1, :]
        id_log_prob = F.log_softmax(id_logits_for_kl / self.kl_temperature, dim=-1)
        text_log_prob = F.log_softmax(text_logits_for_kl / self.kl_temperature, dim=-1)
        id_prob = id_log_prob.exp()
        text_prob = text_log_prob.exp()
        kl_loss = F.kl_div(id_log_prob, text_prob, reduction="batchmean") + F.kl_div(
            text_log_prob, id_prob, reduction="batchmean"
        )

        item_loss = self._compute_item_contrastive_loss(
            id_outputs.last_hidden_state,
            attention_mask,
            text_outputs.last_hidden_state,
            text_attention_mask,
        )

        total_loss = id_loss + text_loss + self.kl_weight * kl_loss + self.item_weight * item_loss
        output = Seq2SeqLMOutput(
            loss=total_loss,
            logits=id_logits,
            past_key_values=id_decoder_outputs.past_key_values,
            decoder_hidden_states=id_decoder_outputs.decoder_hidden_states,
            decoder_attentions=id_decoder_outputs.decoder_attentions,
            cross_attentions=id_decoder_outputs.cross_attentions,
            encoder_last_hidden_state=id_outputs.last_hidden_state,
            encoder_hidden_states=id_outputs.hidden_states,
            encoder_attentions=id_outputs.attentions,
        )
        output.loss_id = id_loss.detach()
        output.loss_text = text_loss.detach()
        output.loss_kl = kl_loss.detach()
        output.loss_item = item_loss.detach()
        return output if return_dict else (total_loss, id_logits)

    def _expand_for_beam_search(
        self,
        tensor: torch.Tensor,
        beam_count: int,
    ) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(1).expand(-1, beam_count, -1).reshape(tensor.size(0) * beam_count, tensor.size(1))
        if tensor.dim() == 3:
            return tensor.unsqueeze(1).expand(-1, beam_count, -1, -1).reshape(
                tensor.size(0) * beam_count, tensor.size(1), tensor.size(2)
            )
        raise ValueError(f"Unsupported tensor rank for beam expansion: {tensor.dim()}")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Any] = None,
        user_rag_emb: Optional[torch.Tensor] = None,
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
            encoder_outputs = self.encode_id(input_ids, attention_mask)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = attention_mask
        else:
            if isinstance(encoder_outputs, BaseModelOutput):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]
            encoder_attention_mask = attention_mask

        if user_rag_emb is not None:
            encoder_hidden_states, encoder_attention_mask = self._fuse_user_rag_memory(
                encoder_hidden_states, encoder_attention_mask, user_rag_emb=user_rag_emb
            )

        device = input_ids.device
        batch_size = input_ids.size(0)
        bos_id = int(self.config.decoder_start_token_id)
        pad_token_id = int(self.config.pad_token_id if pad_token_id is None else pad_token_id)
        num_beams = max(1, int(num_beams))
        num_return_sequences = min(max(1, int(num_return_sequences)), num_beams)
        max_steps = max(1, int(max_length) - 1)

        sequences = torch.full((batch_size, 1, 1), bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros((batch_size, 1), dtype=torch.float, device=device)

        for _ in range(max_steps):
            beam_count = sequences.size(1)
            flat_sequences = sequences.reshape(batch_size * beam_count, -1)
            flat_memory = self._expand_for_beam_search(encoder_hidden_states, beam_count)
            flat_mask = self._expand_for_beam_search(encoder_attention_mask, beam_count)
            decoder_outputs = self.decode(
                target_tokens=flat_sequences,
                encoder_memory=flat_memory,
                encoder_attention_mask=flat_mask,
                return_dict=True,
            )
            step_scores = F.log_softmax(decoder_outputs.logits[:, -1, :], dim=-1)
            if prefix_allowed_tokens_fn is not None:
                constrained = torch.full_like(step_scores, -math.inf)
                for flat_index, sequence in enumerate(flat_sequences):
                    batch_id = flat_index // beam_count
                    allowed_tokens = prefix_allowed_tokens_fn(batch_id, sequence)
                    if allowed_tokens:
                        constrained[flat_index, allowed_tokens] = step_scores[flat_index, allowed_tokens]
                step_scores = constrained
            if logits_processor is not None:
                step_scores = logits_processor(flat_sequences, step_scores)
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

        return sequences[:, :num_return_sequences, :].reshape(batch_size * num_return_sequences, -1)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Any] = None,
        user_rag_emb: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "input_ids": kwargs.get("input_ids"),
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "user_rag_emb": user_rag_emb,
        }
