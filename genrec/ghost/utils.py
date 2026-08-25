from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def normalize_token_sequence(
    sequence: Sequence[int] | Iterable[int],
    *,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    decoder_start_token_id: int = 0,
) -> list[int]:
    cleaned: list[int] = []
    for token in sequence:
        token_id = int(token)
        if token_id == eos_token_id:
            break
        if token_id in {pad_token_id, decoder_start_token_id}:
            continue
        cleaned.append(token_id)
    return cleaned


def tokens_key(
    sequence: Sequence[int] | Iterable[int],
    *,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    decoder_start_token_id: int = 0,
) -> tuple[int, ...]:
    return tuple(
        normalize_token_sequence(
            sequence,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )
    )


def longest_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if int(left[idx]) != int(right[idx]):
            return idx
    return limit


def l2_normalize(array: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(array, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return array / denom


def infer_tokenizer_kind(generative_type: str) -> str:
    normalized = str(generative_type).lower()
    if normalized.startswith("ghost"):
        return "ghost"
    if normalized.endswith("_ghost") or normalized.endswith("_skt"):
        return "ghost"
    if normalized.startswith("letter"):
        return "letter"
    if normalized.startswith("tiger"):
        return "tiger"
    return normalized
