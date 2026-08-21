from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from genrec.genplugin.data_utils import build_leave_one_out_samples, load_filtered_user_sequences, save_json


def _pad_histories(histories: list[list[int]], max_history_items: int) -> torch.LongTensor:
    padded = []
    for history in histories:
        shifted = [int(item) + 1 for item in history[-max_history_items:]]
        padding = max_history_items - len(shifted)
        padded.append([0] * padding + shifted)
    return torch.tensor(padded, dtype=torch.long)


def _load_embedding_table(path: str) -> torch.Tensor:
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        return torch.as_tensor(np.load(path_obj, allow_pickle=True), dtype=torch.float32)
    table = torch.load(path_obj, map_location="cpu")
    return torch.as_tensor(table, dtype=torch.float32)


def _sequence_embeddings_from_item_table(histories: list[list[int]], item_table: torch.Tensor, max_history_items: int) -> torch.Tensor:
    padded_histories = _pad_histories(histories, max_history_items)
    item_embeddings = item_table[padded_histories]
    mask = padded_histories.ne(0).unsqueeze(-1).to(dtype=item_embeddings.dtype)
    summed = (item_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-9)
    return summed / denom


def _sequence_embeddings_from_sasrec_model(
    histories: list[list[int]],
    sasrec_model_dir: str,
    max_history_items: int,
) -> torch.Tensor:
    from disrec.sasrec.sasrec4hf import SASRec4HF

    model = SASRec4HF.from_pretrained(sasrec_model_dir)
    model.eval()
    padded_histories = _pad_histories(histories, max_history_items)
    attention_mask = padded_histories.ne(0).long()
    with torch.no_grad():
        sequence_hidden = model.SASRec(log_seqs=padded_histories)
        mask = attention_mask.unsqueeze(-1).to(dtype=sequence_hidden.dtype)
        summed = (sequence_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-9)
        return summed / denom


def build_split_index(
    interaction_path: str,
    output_dir: Path,
    split: str,
    max_history_items: int,
    top_k: int,
    min_user_interactions: int,
    min_item_interactions: int,
    sasrec_model_dir: str | None = None,
    sasrec_item_embedding_path: str | None = None,
) -> None:
    user_sequences = load_filtered_user_sequences(
        interaction_path,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    samples = build_leave_one_out_samples(user_sequences, split, max_history_items=max_history_items)
    if not samples:
        return

    histories = [sample["history_items"] for sample in samples]
    if sasrec_model_dir:
        user_embeddings = _sequence_embeddings_from_sasrec_model(histories, sasrec_model_dir, max_history_items)
    elif sasrec_item_embedding_path:
        item_table = _load_embedding_table(sasrec_item_embedding_path)
        user_embeddings = _sequence_embeddings_from_item_table(histories, item_table, max_history_items)
    else:
        raise ValueError("Provide either --sasrec_model_dir or --sasrec_item_embedding_path.")

    normalized_embeddings = torch.nn.functional.normalize(user_embeddings, p=2, dim=-1)
    candidate_lists = []
    chunk_size = 512
    total_queries = normalized_embeddings.size(0)
    for start in range(0, total_queries, chunk_size):
        end = min(total_queries, start + chunk_size)
        query_chunk = normalized_embeddings[start:end]
        similarities = query_chunk @ normalized_embeddings.t()
        for row_offset, query_index in enumerate(range(start, end)):
            similarities[row_offset, query_index] = -torch.inf
        candidate_k = min(top_k, similarities.size(1))
        top_indices = torch.topk(similarities, k=candidate_k, dim=-1).indices.tolist()
        candidate_lists.extend([[int(index) for index in row] for row in top_indices])

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    save_json(candidate_lists, split_dir / "item_retrival.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build collaborative retrieval candidates for GENPLUGIN.")
    parser.add_argument("--data_interaction_files", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "valid", "test"])
    parser.add_argument("--max_history_items", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=1001)
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--min_item_interactions", type=int, default=5)
    parser.add_argument("--sasrec_model_dir", default=None)
    parser.add_argument("--sasrec_item_embedding_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        build_split_index(
            interaction_path=args.data_interaction_files,
            output_dir=output_dir,
            split=split,
            max_history_items=args.max_history_items,
            top_k=args.top_k,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            sasrec_model_dir=args.sasrec_model_dir,
            sasrec_item_embedding_path=args.sasrec_item_embedding_path,
        )


if __name__ == "__main__":
    main()
