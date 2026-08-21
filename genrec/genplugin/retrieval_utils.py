from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import re

import numpy as np
import torch


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> list[str]:
    if text is None:
        return []
    return _TOKEN_PATTERN.findall(str(text).lower())


@dataclass
class BM25Index:
    documents: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in self.documents]
        self.doc_len = np.asarray([len(doc) for doc in self.documents], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.idf: dict[str, float] = {}
        all_terms = set(term for doc in self.documents for term in doc)
        doc_count = len(self.documents)
        for term in all_terms:
            freq = sum(1 for doc in self.doc_freqs if term in doc)
            self.idf[term] = math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not query_tokens or not len(self.documents):
            return scores

        query_freq = Counter(query_tokens)
        for doc_index, doc_freq in enumerate(self.doc_freqs):
            doc_len = float(self.doc_len[doc_index])
            denom_norm = self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
            score = 0.0
            for term, qf in query_freq.items():
                if term not in doc_freq:
                    continue
                tf = float(doc_freq[term])
                idf = self.idf.get(term, 0.0)
                score += idf * (tf * (self.k1 + 1.0)) / (tf + denom_norm)
            scores[doc_index] = score
        return scores

    def top_k(self, query_tokens: list[str], k: int = 1001, exclude_index: int | None = None) -> list[int]:
        scores = self.score(query_tokens)
        if exclude_index is not None and 0 <= exclude_index < len(scores):
            scores[exclude_index] = -np.inf
        if len(scores) == 0:
            return []
        k = min(k, len(scores))
        top_indices = np.argpartition(-scores, kth=k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [int(index) for index in top_indices.tolist()]


def build_history_document(history_items: Iterable[int], item_text_by_id: dict[int, str]) -> list[str]:
    tokens: list[str] = []
    for item_id in history_items:
        tokens.extend(normalize_text(item_text_by_id.get(int(item_id), "")))
    return tokens


def cosine_similarities(query_vector: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
    query_norm = query_vector.norm(p=2).clamp_min(1e-9)
    candidate_norm = candidate_vectors.norm(p=2, dim=-1).clamp_min(1e-9)
    return (candidate_vectors @ query_vector) / (candidate_norm * query_norm)


def rerank_candidates_by_cosine(
    query_index: int,
    sparse_candidates: list[int],
    collaborative_candidates: list[int],
    user_embeddings: torch.Tensor,
    top_k: int = 50,
) -> list[int]:
    sparse_candidates = [int(index) for index in sparse_candidates]
    collaborative_candidates = [int(index) for index in collaborative_candidates]
    collaborative_lookup = set(collaborative_candidates)

    ordered: list[int] = []
    seen: set[int] = set()
    for candidate in sparse_candidates:
        if candidate == query_index or candidate in seen:
            continue
        if candidate in collaborative_lookup:
            ordered.append(candidate)
            seen.add(candidate)
            if len(ordered) >= top_k:
                return ordered[:top_k]

    remaining = []
    for candidate in sparse_candidates:
        if candidate == query_index or candidate in seen:
            continue
        remaining.append(candidate)
        seen.add(candidate)
    for candidate in collaborative_candidates:
        if candidate == query_index or candidate in seen:
            continue
        remaining.append(candidate)
        seen.add(candidate)

    if not remaining:
        return ordered[:top_k]

    query_vector = user_embeddings[query_index]
    candidate_tensor = user_embeddings[remaining]
    similarities = cosine_similarities(query_vector, candidate_tensor)
    sort_indices = torch.argsort(similarities, descending=True).tolist()
    for sort_index in sort_indices:
        ordered.append(int(remaining[sort_index]))
        if len(ordered) >= top_k:
            break
    return ordered[:top_k]

