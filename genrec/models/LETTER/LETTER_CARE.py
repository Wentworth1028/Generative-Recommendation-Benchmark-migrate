from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from genrec.models.LETTER.LETTER import LETTERT5ForConditionalGeneration


class LETTERCARE(LETTERT5ForConditionalGeneration):
    """CARE variant for LETTER semantic-ID generation.

    The tokenizer and semantic IDs stay unchanged. CARE inserts learned query
    embeddings at each generation stage and predicts each SID token from the
    last query hidden state. LETTER's temperature scaling is preserved.
    """

    def __init__(self, config):
        super().__init__(config)
        self.care_sid_length = int(getattr(config, "care_sid_length", 4))
        query_numbers = getattr(config, "care_query_numbers", [1, 1, 4, 4])
        if isinstance(query_numbers, str):
            query_numbers = [int(item.strip()) for item in query_numbers.strip("[]").split(",") if item.strip()]
        self.care_query_numbers = [int(n) for n in query_numbers][: self.care_sid_length]
        if len(self.care_query_numbers) < self.care_sid_length:
            fill_value = self.care_query_numbers[-1] if self.care_query_numbers else 1
            self.care_query_numbers.extend([fill_value] * (self.care_sid_length - len(self.care_query_numbers)))
        self.care_alpha = float(getattr(config, "care_alpha", 0.7))

        self.care_queries = nn.ParameterList()
        for n_queries in self.care_query_numbers:
            query = nn.Parameter(torch.empty(n_queries, config.d_model))
            nn.init.normal_(query, mean=0.0, std=0.02)
            self.care_queries.append(query)

    def _progressive_attention_mask(
        self,
        attention_mask: torch.Tensor,
        stage_idx: int,
    ) -> torch.Tensor:
        nonpad = attention_mask.to(dtype=torch.bool)
        token_rank = nonpad.long().cumsum(dim=-1) - 1
        within_item_position = torch.remainder(token_rank.clamp_min(0), self.care_sid_length)
        visible = within_item_position <= stage_idx
        return (nonpad & visible).to(dtype=attention_mask.dtype)

    def _decoder_prefix_embeds(self, generated_ids: torch.LongTensor, stage_idx: int) -> torch.Tensor:
        embed = self.get_input_embeddings()
        pieces = [embed(generated_ids[:, :1])]
        batch_size = generated_ids.size(0)
        for query_stage in range(stage_idx + 1):
            query = self.care_queries[query_stage].to(device=generated_ids.device)
            pieces.append(query.unsqueeze(0).expand(batch_size, -1, -1))
            if query_stage < stage_idx:
                pieces.append(embed(generated_ids[:, query_stage + 1 : query_stage + 2]))
        return torch.cat(pieces, dim=1)

    def _stage_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        generated_ids: torch.LongTensor,
        stage_idx: int,
    ) -> torch.FloatTensor:
        stage_attention_mask = self._progressive_attention_mask(attention_mask, stage_idx)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=stage_attention_mask,
            return_dict=True,
        )
        decoder_inputs_embeds = self._decoder_prefix_embeds(generated_ids, stage_idx)
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=stage_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        query_hidden = decoder_outputs.last_hidden_state[:, -1, :]
        if self.config.tie_word_embeddings:
            query_hidden = query_hidden * (self.model_dim**-0.5)
        return self.lm_head(query_hidden) / self.temperature

    def _diversity_loss(self) -> torch.Tensor:
        queries = torch.cat([query for query in self.care_queries], dim=0)
        if queries.size(0) <= 1:
            return queries.new_tensor(0.0)
        normalized = F.normalize(queries, p=2, dim=-1)
        cosine = normalized @ normalized.t()
        off_diag = cosine.sum() - torch.diagonal(cosine).sum()
        return off_diag / (queries.size(0) * queries.size(0) - queries.size(0))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            raise ValueError("CARE requires input_ids.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()

        batch_size = input_ids.size(0)
        device = input_ids.device
        sid_len = self.care_sid_length
        if labels is None:
            generated_ids = torch.full(
                (batch_size, 1),
                int(self.config.decoder_start_token_id),
                dtype=torch.long,
                device=device,
            )
            logits = []
            for stage_idx in range(sid_len):
                stage_logits = self._stage_logits(input_ids, attention_mask, generated_ids, stage_idx)
                logits.append(stage_logits)
                next_token = stage_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            lm_logits = torch.stack(logits, dim=1)
            return Seq2SeqLMOutput(loss=None, logits=lm_logits) if return_dict else (lm_logits,)

        sid_labels = labels[:, :sid_len].to(device)
        bos = torch.full(
            (batch_size, 1),
            int(self.config.decoder_start_token_id),
            dtype=torch.long,
            device=device,
        )
        logits = []
        for stage_idx in range(sid_len):
            generated_ids = torch.cat([bos, sid_labels[:, :stage_idx].clamp_min(0)], dim=1)
            logits.append(self._stage_logits(input_ids, attention_mask, generated_ids, stage_idx))
        lm_logits = torch.stack(logits, dim=1)

        loss_rec = F.cross_entropy(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            sid_labels.reshape(-1),
            ignore_index=-100,
        )
        loss_div = self._diversity_loss().to(loss_rec.device)
        loss = loss_rec + self.care_alpha * loss_div

        output = Seq2SeqLMOutput(loss=loss, logits=lm_logits)
        output.loss_rec = loss_rec.detach()
        output.loss_div = loss_div.detach()
        return output if return_dict else (loss, lm_logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        logits_processor=None,
        max_length: int = 5,
        num_beams: int = 10,
        num_return_sequences: int = 10,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        batch_size = input_ids.size(0)
        device = input_ids.device
        bos_id = int(self.config.decoder_start_token_id)
        num_beams = max(1, int(num_beams))
        num_return_sequences = min(max(1, int(num_return_sequences)), num_beams)
        sid_len = min(self.care_sid_length, max(1, int(max_length) - 1))

        sequences = torch.full((batch_size, 1, 1), bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros((batch_size, 1), dtype=torch.float, device=device)

        for stage_idx in range(sid_len):
            beam_count = sequences.size(1)
            flat_sequences = sequences.reshape(batch_size * beam_count, -1)
            flat_input_ids = input_ids.unsqueeze(1).expand(-1, beam_count, -1).reshape(batch_size * beam_count, -1)
            flat_attention_mask = attention_mask.unsqueeze(1).expand(-1, beam_count, -1).reshape(
                batch_size * beam_count, -1
            )
            stage_logits = self._stage_logits(flat_input_ids, flat_attention_mask, flat_sequences, stage_idx)
            token_scores = F.log_softmax(stage_logits, dim=-1)
            if logits_processor is not None:
                token_scores = logits_processor(flat_sequences, token_scores)
            vocab_size = token_scores.size(-1)
            total_scores = token_scores + beam_scores.reshape(-1, 1)
            total_scores = total_scores.view(batch_size, beam_count * vocab_size)
            next_scores, next_indices = torch.topk(total_scores, k=num_beams, dim=-1)
            next_beam_indices = torch.div(next_indices, vocab_size, rounding_mode="floor")
            next_tokens = torch.remainder(next_indices, vocab_size)

            gathered_sequences = sequences.gather(
                dim=1,
                index=next_beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1)),
            )
            sequences = torch.cat([gathered_sequences, next_tokens.unsqueeze(-1)], dim=-1)
            beam_scores = next_scores

        return sequences[:, :num_return_sequences, :].reshape(batch_size * num_return_sequences, -1)
