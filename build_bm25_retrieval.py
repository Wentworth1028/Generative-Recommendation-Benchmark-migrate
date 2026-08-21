from __future__ import annotations

import argparse
from pathlib import Path

from genrec.genplugin.data_utils import build_leave_one_out_samples, load_filtered_user_sequences, load_item_title_map, save_json
from genrec.genplugin.retrieval_utils import BM25Index, build_history_document


def build_split_index(
    interaction_path: str,
    text_path: str,
    output_dir: Path,
    split: str,
    max_history_items: int,
    top_k: int,
    min_user_interactions: int,
    min_item_interactions: int,
) -> None:
    user_sequences = load_filtered_user_sequences(
        interaction_path,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    samples = build_leave_one_out_samples(user_sequences, split, max_history_items=max_history_items)
    if not samples:
        return

    title_map = load_item_title_map(text_path)
    documents = [build_history_document(sample["history_items"], title_map) for sample in samples]
    bm25 = BM25Index(documents)

    candidate_lists = []
    for sample in samples:
        query_tokens = build_history_document(sample["history_items"], title_map)
        candidate_lists.append(bm25.top_k(query_tokens, k=top_k, exclude_index=sample["index"]))

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    save_json(candidate_lists, split_dir / "sparse_user_index.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 retrieval candidates for GENPLUGIN.")
    parser.add_argument("--data_interaction_files", required=True)
    parser.add_argument("--data_text_files", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "valid", "test"])
    parser.add_argument("--max_history_items", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=1001)
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--min_item_interactions", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        build_split_index(
            interaction_path=args.data_interaction_files,
            text_path=args.data_text_files,
            output_dir=output_dir,
            split=split,
            max_history_items=args.max_history_items,
            top_k=args.top_k,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )


if __name__ == "__main__":
    main()

