from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def load_semantic_id_tokenizer(item2tokens_path: str | Path):
    path = Path(item2tokens_path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    item2tokens = {int(key): [int(token) for token in value] for key, value in loaded.items()}
    if not item2tokens:
        raise ValueError(f"item2tokens file is empty: {path}")

    first_tokens = next(iter(item2tokens.values()))
    digits = len(first_tokens)
    max_token_id = max(max(tokens) for tokens in item2tokens.values())

    tokenizer = SimpleNamespace()
    tokenizer.item2tokens = item2tokens
    tokenizer.tokens2item = {tuple(tokens): item_id for item_id, tokens in item2tokens.items()}
    tokenizer.pad_token = 0
    tokenizer.eos_token = 1
    tokenizer.ignored_label = -100
    tokenizer.reserve_tokens = 0
    tokenizer.n_codebooks = 1
    tokenizer.codebook_size = max_token_id + 1
    tokenizer.digits = digits
    tokenizer.num_user_tokens = 0
    tokenizer.user_token_start_idx = tokenizer.n_codebooks * tokenizer.codebook_size
    tokenizer.vocab_size = tokenizer.user_token_start_idx

    def _get_user_token(user_id):
        raise NotImplementedError("GENPLUGIN tokenizer does not use user tokens.")

    tokenizer.get_user_token = _get_user_token
    return tokenizer

