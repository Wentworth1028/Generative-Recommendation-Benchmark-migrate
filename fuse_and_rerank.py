from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from genrec.genplugin.data_utils import load_json, save_json
from genrec.genplugin.retrieval_utils import rerank_candidates_by_cosine


def _load_user_embeddings(path: str) -> torch.Tensor:
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        return torch.as_tensor(np.load(path_obj, allow_pickle=True), dtype=torch.float32)
    table = torch.load(path_obj, map_location="cpu")
    return torch.as_tensor(table, dtype=torch.float32)


def rerank_split(
    sparse_index_path: Path,
    collaborative_index_path: Path,
    user_embedding_path: str,
    output_path: Path,
    fusion_top_k: int,
) -> None:
    sparse_index = load_json(sparse_index_path)
    collaborative_index = load_json(collaborative_index_path)
    user_embeddings = _load_user_embeddings(user_embedding_path)

    reranked = []
    for query_index, (sparse_candidates, collaborative_candidates) in enumerate(
        zip(sparse_index, collaborative_index)
    ):
        reranked.append(
            rerank_candidates_by_cosine(
                query_index=query_index,
                sparse_candidates=list(sparse_candidates)[:fusion_top_k],
                collaborative_candidates=list(collaborative_candidates)[:fusion_top_k],
                user_embeddings=user_embeddings,
                top_k=fusion_top_k,
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(reranked, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse sparse and collaborative retrieval results for GENPLUGIN.")
    parser.add_argument("--sparse_index_path", required=True)
    parser.add_argument("--collaborative_index_path", required=True)
    parser.add_argument("--user_embedding_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--fusion_top_k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rerank_split(
        sparse_index_path=Path(args.sparse_index_path),
        collaborative_index_path=Path(args.collaborative_index_path),
        user_embedding_path=args.user_embedding_path,
        output_path=Path(args.output_path),
        fusion_top_k=args.fusion_top_k,
    )


if __name__ == "__main__":
    main()

