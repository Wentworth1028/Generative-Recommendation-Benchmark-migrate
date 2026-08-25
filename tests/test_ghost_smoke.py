from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import T5Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genrec.generation.trie import Trie
from genrec.models.GHOST.GHOST import GhostLETTER, GhostTIGER
from genrec.trainers.generative.tiger_trainer import FastTrieLogitsProcessor
from genrec.utils.common_utils import tokens_to_item_id


def _build_config(include_tau: bool = False) -> T5Config:
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    if include_tau:
        config.tau = 1.0
    config.ghost_variant = "ghost"
    config.ghost_collection_path = None
    config.auo_alpha = 0.1
    config.auo_ka = 200
    config.auo_kb = 5
    config.auo_temperature = 1.0
    return config


@torch.no_grad()
def _make_collection():
    return {
        "7": {
            "head_sids": [
                [8, 9, 10, 11],
                [8, 9, 10, 12],
                [8, 9, 10, 13],
                [8, 9, 10, 14],
                [8, 9, 10, 15],
            ]
        }
    }


def _run_forward_smoke(model):
    model.train()
    model.ghost_collection = _make_collection()
    model.auo_alpha = 0.1
    model.auo_kb = 5

    input_ids = torch.tensor([[2, 3, 1, 0], [4, 5, 1, 0]], dtype=torch.long)
    attention_mask = input_ids.ne(0).long()
    labels = torch.tensor([[8, 9, 10, 11, 1], [8, 9, 10, 12, 1]], dtype=torch.long)
    item_id = torch.tensor([6, 7], dtype=torch.long)
    target_is_tail = torch.tensor([0, 1], dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        item_id=item_id,
        target_is_tail=target_is_tail,
    )

    assert torch.isfinite(outputs.loss)
    assert outputs.logits.shape[0] == 2
    assert outputs.encoder_last_hidden_state.shape[:2] == input_ids.shape

    outputs.loss.backward()
    assert model.lm_head.weight.grad is not None


def test_ghost_tiger_forward_smoke():
    torch.manual_seed(0)
    model = GhostTIGER(_build_config(include_tau=False))
    _run_forward_smoke(model)


def test_ghost_letter_forward_smoke():
    torch.manual_seed(0)
    model = GhostLETTER(_build_config(include_tau=True))
    _run_forward_smoke(model)


def test_ghost_trie_eos_fallback_and_token_normalization():
    trie = Trie({1: (2, 3), 2: (2, 3, 4)})
    processor = FastTrieLogitsProcessor(trie, vocab_size=8, fallback_token_id=1)

    scores = torch.zeros((1, 8), dtype=torch.float)
    masked = processor(torch.tensor([[0, 2, 3, 4]], dtype=torch.long), scores)
    allowed = torch.isfinite(masked[0]).nonzero(as_tuple=False).flatten().tolist()
    assert allowed == [1]

    assert tokens_to_item_id([2, 3, 1, 0, 0], {(2, 3): 42}) == 42


if __name__ == "__main__":
    test_ghost_tiger_forward_smoke()
    test_ghost_letter_forward_smoke()
    test_ghost_trie_eos_fallback_and_token_normalization()
