import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import contextlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.quantization.data.dataset.rqvae_dataset import ItemEmbeddingDataset
from genrec.quantization.tokenizers.rqvae_tokenizer import RQVAETokenizer


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def default_log_file(args: argparse.Namespace) -> Path:
    return Path(args.output_tokenizer_dir).parent / "crab_lite.log"


def run_with_log(args: argparse.Namespace) -> None:
    log_file = Path(args.log_file) if args.log_file else default_log_file(args)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append_log else "w"
    with log_file.open(mode, encoding="utf-8") as log:
        log.write("=" * 60 + "\n")
        log.write("CRAB-lite rebalance\n")
        log.write(f"Output: {Path(args.output_tokenizer_dir).parent}\n")
        log.write("=" * 60 + "\n")
        log.flush()
        with contextlib.redirect_stdout(TeeStream(sys.__stdout__, log)), contextlib.redirect_stderr(
            TeeStream(sys.__stderr__, log)
        ):
            run_crab_lite(args)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_item2tokens(path: Path) -> dict[int, tuple[int, ...]]:
    raw = load_json(path)
    return {int(item_id): tuple(int(token) for token in tokens) for item_id, tokens in raw.items()}


def load_user_ids_and_popularity(interaction_file: Path) -> tuple[list[int], dict[int, int], dict[int, list[int]]]:
    with interaction_file.open("rb") as f:
        user2item_data = pickle.load(f)

    item_popularity = Counter()
    user_sequences = {}
    user_ids = []
    for _, row in user2item_data.iterrows():
        user_id = int(row["UserID"])
        seq = [int(item_id) for item_id in row["ItemID"]]
        user_ids.append(user_id)
        user_sequences[user_id] = seq
        item_popularity.update(seq)
    return user_ids, dict(item_popularity), user_sequences


def hash_user_id(user_id: int, start_idx: int, num_user_tokens: int) -> int:
    hash_object = hashlib.md5(str(user_id).encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return start_idx + (hash_int % num_user_tokens)


def gini(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - (n + 1) / n)


def entropy(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 0.0
    prob = arr / total
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob)).sum())


def top_ratio_exposure(values: list[float], ratio: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if arr.size == 0 or total <= 0:
        return 0.0
    top_n = max(1, int(math.ceil(arr.size * ratio)))
    return float(np.sort(arr)[::-1][:top_n].sum() / total)


def token_popularity_by_layer(
    item2tokens: dict[int, tuple[int, ...]],
    item_popularity: dict[int, int],
    n_codebooks: int,
    layer_token_ids: list[list[int]] | None = None,
) -> list[dict[int, int]]:
    result = [defaultdict(int) for _ in range(n_codebooks)]
    if layer_token_ids is not None:
        for layer, token_ids in enumerate(layer_token_ids):
            for token_id in token_ids:
                result[layer][int(token_id)] += 0
    for item_id, tokens in item2tokens.items():
        pop = int(item_popularity.get(item_id, 0))
        for layer, token_id in enumerate(tokens):
            result[layer][int(token_id)] += pop
    return [dict(layer_pop) for layer_pop in result]


def summarize_layer_popularity(layer_popularity: list[dict[int, int]], top_ratio: float) -> dict[str, Any]:
    summary = {}
    for layer, pop_map in enumerate(layer_popularity):
        values = list(pop_map.values())
        summary[str(layer)] = {
            "num_tokens": len(values),
            "total_popularity": int(sum(values)),
            "mean_popularity": float(np.mean(values)) if values else 0.0,
            "gini": gini(values),
            "entropy": entropy(values),
            f"top_{top_ratio:.2f}_exposure_ratio": top_ratio_exposure(values, top_ratio),
        }
    return summary


def semantic_ids_from_tokens(
    item2tokens: dict[int, tuple[int, ...]],
    reserve_tokens: int,
    codebook_size: int,
    n_codebooks: int,
) -> dict[int, tuple[int, ...]]:
    item2sem = {}
    for item_id, tokens in item2tokens.items():
        if len(tokens) != n_codebooks:
            raise ValueError(f"Item {item_id} has {len(tokens)} tokens, expected {n_codebooks}.")
        sem_ids = []
        for layer, token_id in enumerate(tokens):
            sem_id = int(token_id) - reserve_tokens - codebook_size * layer
            if sem_id < 0 or sem_id >= codebook_size:
                raise ValueError(
                    f"Item {item_id} token {token_id} is outside the original layer-{layer} "
                    f"range. CRAB-lite expects an original RQ-VAE tokenizer as input."
                )
            sem_ids.append(sem_id)
        item2sem[item_id] = tuple(sem_ids)
    return item2sem


def load_rqvae_tokenizer(config: dict, tokenizer_dir: Path, device: str) -> RQVAETokenizer:
    tokenizer_config = dict(config)
    tokenizer_config["save_path"] = str(tokenizer_dir / "item2tokens.json")
    tokenizer_config["checkpoint_path"] = str(tokenizer_dir / "tokenizer_checkpoint.pth")
    tokenizer_config["tokenizer_path"] = str(tokenizer_dir / "tokenizer.pkl")
    tokenizer_config["device"] = device
    tokenizer = RQVAETokenizer(tokenizer_config)

    checkpoint_path = tokenizer_dir / "tokenizer_checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing tokenizer checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state = {}
    for key, value in state_dict.items():
        clean_state[key[len("rq_vae.") : ] if key.startswith("rq_vae.") else key] = value
    tokenizer.rq_vae.load_state_dict(clean_state)
    tokenizer.rq_vae.to(device)
    tokenizer.rq_vae.eval()
    return tokenizer


def export_residuals(
    tokenizer: RQVAETokenizer,
    item_ids: list[int],
    embeddings: np.ndarray,
    item2sem: dict[int, tuple[int, ...]],
    device: str,
    batch_size: int,
) -> tuple[list[np.ndarray], list[torch.Tensor]]:
    n_codebooks = tokenizer.n_codebooks
    codebooks = [layer.get_codebook().detach().cpu().clone() for layer in tokenizer.rq_vae.rq.vq_layers]
    residuals = [[] for _ in range(n_codebooks)]

    with torch.no_grad():
        for start in range(0, len(item_ids), batch_size):
            end = start + batch_size
            batch_item_ids = item_ids[start:end]
            batch_embeddings = torch.from_numpy(embeddings[start:end]).float().to(device)
            encoded = tokenizer.rq_vae.encoder(batch_embeddings)
            residual = encoded
            for layer in range(n_codebooks):
                residuals[layer].append(residual.detach().cpu().numpy())
                sem_ids = torch.tensor(
                    [item2sem[item_id][layer] for item_id in batch_item_ids],
                    dtype=torch.long,
                    device=device,
                )
                codeword = tokenizer.rq_vae.rq.vq_layers[layer].embedding(sem_ids)
                residual = residual - codeword

    residual_arrays = [np.concatenate(parts, axis=0) for parts in residuals]
    return residual_arrays, codebooks


def initial_kmeans(vectors: np.ndarray, weights: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if len(vectors) == n_clusters:
        return np.arange(n_clusters, dtype=np.int64)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    try:
        return kmeans.fit_predict(vectors, sample_weight=weights)
    except TypeError:
        return kmeans.fit_predict(vectors)


def repair_empty_clusters(labels: np.ndarray, pop: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = labels.copy()
    counts = np.bincount(labels, minlength=n_clusters)
    for empty_cluster in np.where(counts == 0)[0]:
        movable = [idx for idx in range(len(labels)) if counts[labels[idx]] > 1]
        if not movable:
            continue
        donor = max(movable, key=lambda idx: pop[idx])
        counts[labels[donor]] -= 1
        labels[donor] = empty_cluster
        counts[empty_cluster] += 1
    return labels


def update_centers(vectors: np.ndarray, sizes: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centers = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float64)
    for cluster in range(n_clusters):
        mask = labels == cluster
        if not np.any(mask):
            continue
        centers[cluster] = np.average(vectors[mask], axis=0, weights=sizes[mask])
    return centers


def balanced_refinement(
    vectors: np.ndarray,
    sizes: np.ndarray,
    popularity: np.ndarray,
    n_clusters: int,
    balance_weight: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    pop_norm = popularity / max(float(popularity.sum()), 1e-12)
    target_pop = 1.0 / n_clusters
    labels = initial_kmeans(vectors, sizes, n_clusters, seed)
    labels = repair_empty_clusters(labels, pop_norm, n_clusters)
    centers = update_centers(vectors, sizes, labels, n_clusters)

    order = np.argsort(-pop_norm)
    for _ in range(max_iter):
        new_labels = np.full_like(labels, fill_value=-1)
        cluster_pop = np.zeros(n_clusters, dtype=np.float64)
        for group_idx in order:
            distances = ((centers - vectors[group_idx]) ** 2).sum(axis=1)
            costs = sizes[group_idx] * distances + balance_weight * (cluster_pop + pop_norm[group_idx] - target_pop) ** 2
            cluster = int(np.argmin(costs))
            new_labels[group_idx] = cluster
            cluster_pop[cluster] += pop_norm[group_idx]
        new_labels = repair_empty_clusters(new_labels, pop_norm, n_clusters)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = update_centers(vectors, sizes, labels, n_clusters)

    return labels, centers.astype(np.float32)


def resolve_duplicate_item_tokens(
    item2tokens: dict[int, list[int]],
    item_popularity: dict[int, int],
    residuals: list[np.ndarray],
    item_index: dict[int, int],
    codebooks_by_layer_token: list[dict[int, np.ndarray]],
    next_token_id: int,
) -> tuple[int, list[dict[str, Any]]]:
    duplicate_resolutions = []
    tuple_to_items = defaultdict(list)
    for item_id, tokens in item2tokens.items():
        tuple_to_items[tuple(tokens)].append(item_id)

    last_layer = len(residuals) - 1
    for token_tuple, items in tuple_to_items.items():
        if len(items) <= 1:
            continue
        items = sorted(items, key=lambda item: (-item_popularity.get(item, 0), item))
        kept_item = items[0]
        for item_id in items[1:]:
            old_token = item2tokens[item_id][last_layer]
            new_token = next_token_id
            next_token_id += 1
            item2tokens[item_id][last_layer] = new_token
            codebooks_by_layer_token[last_layer][new_token] = residuals[last_layer][item_index[item_id]].astype(np.float32)
            duplicate_resolutions.append(
                {
                    "kept_item": int(kept_item),
                    "reassigned_item": int(item_id),
                    "old_last_layer_token": int(old_token),
                    "new_last_layer_token": int(new_token),
                }
            )
    return next_token_id, duplicate_resolutions


def rebuild_samples(
    user_sequences: dict[int, list[int]],
    item2tokens: dict[int, list[int]],
    max_seq_len: int,
    output_dir: Path,
) -> None:
    modes = {"train": output_dir / "rebuilt_train.jsonl", "valid": output_dir / "rebuilt_valid.jsonl", "test": output_dir / "rebuilt_test.jsonl"}
    files = {mode: path.open("w", encoding="utf-8") for mode, path in modes.items()}
    try:
        for user_id, item_seq in user_sequences.items():
            train_seq = item_seq[-(max_seq_len + 2) : -2]
            for idx in range(1, len(train_seq)):
                history = train_seq[:idx]
                target = train_seq[idx]
                write_rebuilt_sample(files["train"], user_id, history, target, item2tokens)

            if len(item_seq) >= 3:
                history = item_seq[:-2][-max_seq_len:]
                write_rebuilt_sample(files["valid"], user_id, history, item_seq[-2], item2tokens)
            if len(item_seq) >= 2:
                history = item_seq[:-1][-max_seq_len:]
                write_rebuilt_sample(files["test"], user_id, history, item_seq[-1], item2tokens)
    finally:
        for f in files.values():
            f.close()


def write_rebuilt_sample(f, user_id: int, history: list[int], target: int, item2tokens: dict[int, list[int]]) -> None:
    if target not in item2tokens:
        return
    source_tokens = []
    for item_id in history:
        if item_id in item2tokens:
            source_tokens.extend(item2tokens[item_id])
    row = {
        "user_id": int(user_id),
        "history_items": [int(item_id) for item_id in history],
        "target_item": int(target),
        "source_tokens": [int(token) for token in source_tokens],
        "target_tokens": [int(token) for token in item2tokens[target]],
    }
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_crab_lite(args: argparse.Namespace) -> None:
    source_tokenizer_dir = Path(args.tokenizer_dir)
    output_tokenizer_dir = Path(args.output_tokenizer_dir)
    if source_tokenizer_dir.resolve() == output_tokenizer_dir.resolve():
        raise ValueError("--output-tokenizer-dir must be different from --tokenizer-dir.")
    output_dir = output_tokenizer_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    crab_config = config.get("crab", {})
    args.split_ratio = args.split_ratio if args.split_ratio is not None else crab_config.get("split_ratio", 0.10)
    args.analysis_top_ratio = (
        args.analysis_top_ratio if args.analysis_top_ratio is not None else crab_config.get("analysis_top_ratio", 0.05)
    )
    args.max_splits = args.max_splits if args.max_splits is not None else crab_config.get("max_splits", 3)
    args.balance_weight = (
        args.balance_weight if args.balance_weight is not None else crab_config.get("balance_weight", 1.0)
    )
    args.kmeans_max_iter = (
        args.kmeans_max_iter if args.kmeans_max_iter is not None else crab_config.get("kmeans_max_iter", 50)
    )
    args.seed = args.seed if args.seed is not None else crab_config.get("random_seed", 2026)
    config["data_text_files"] = args.data_text_files
    config["interaction_files"] = args.interaction_file
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    item2tokens_original = load_item2tokens(source_tokenizer_dir / "item2tokens.json")
    user_ids, item_popularity, user_sequences = load_user_ids_and_popularity(Path(args.interaction_file))

    tokenizer = load_rqvae_tokenizer(config, source_tokenizer_dir, device)
    reserve_tokens = tokenizer.reserve_tokens
    num_user_tokens = tokenizer.num_user_tokens
    n_codebooks = int(config["n_codebooks"])
    codebook_size = int(config["codebook_size"])
    item2sem = semantic_ids_from_tokens(item2tokens_original, reserve_tokens, codebook_size, n_codebooks)

    dataset = ItemEmbeddingDataset(
        data_text_files=args.data_text_files,
        config=config,
        text_encoder_model=config["text_encoder_model"],
        embedding_extraction_strategy=config.get("embedding_strategy", "mean_pooling"),
        device=device,
    )
    item_ids = [int(item_id) for item_id in dataset.item_ids if int(item_id) in item2tokens_original]
    embeddings = np.asarray([dataset.item_embeddings[item_id] for item_id in item_ids], dtype=np.float32)
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    residuals, original_codebooks = export_residuals(
        tokenizer=tokenizer,
        item_ids=item_ids,
        embeddings=embeddings,
        item2sem=item2sem,
        device=device,
        batch_size=args.batch_size,
    )

    item2tokens = {item_id: list(tokens) for item_id, tokens in item2tokens_original.items()}
    codebooks_by_layer_token = []
    for layer in range(n_codebooks):
        layer_map = {}
        for sem_id in range(codebook_size):
            token_id = reserve_tokens + codebook_size * layer + sem_id
            layer_map[token_id] = original_codebooks[layer][sem_id].numpy().astype(np.float32)
        codebooks_by_layer_token.append(layer_map)

    original_layer_token_ids = [
        [reserve_tokens + codebook_size * layer + sem_id for sem_id in range(codebook_size)]
        for layer in range(n_codebooks)
    ]
    before_popularity = token_popularity_by_layer(
        item2tokens_original,
        item_popularity,
        n_codebooks,
        layer_token_ids=original_layer_token_ids,
    )
    dump_json(
        {
            "token_popularity": [{str(k): int(v) for k, v in layer.items()} for layer in before_popularity],
            "summary": summarize_layer_popularity(before_popularity, args.analysis_top_ratio),
        },
        output_dir / "token_popularity_before.json",
    )

    selected_records = []
    split_mapping = {}
    next_token_id = max(max(tokens) for tokens in item2tokens_original.values()) + 1

    for layer in range(n_codebooks):
        layer_pop = before_popularity[layer]
        mean_popularity = float(np.mean(list(layer_pop.values()))) if layer_pop else 0.0
        top_n = max(1, int(math.ceil(len(layer_pop) * args.split_ratio)))
        selected_tokens = sorted(layer_pop.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

        for old_token_id, token_popularity in selected_tokens:
            if mean_popularity <= 0:
                continue
            desired_splits = min(args.max_splits, max(2, int(math.ceil(token_popularity / mean_popularity))))
            affected_items = [item_id for item_id in item_ids if item2tokens[item_id][layer] == old_token_id]
            if layer < n_codebooks - 1:
                groups = defaultdict(list)
                for item_id in affected_items:
                    groups[item2tokens[item_id][layer + 1]].append(item_id)
            else:
                groups = {item_id: [item_id] for item_id in affected_items}

            group_items = list(groups.values())
            split_count = min(desired_splits, len(group_items))
            if split_count < 2:
                continue

            vectors = []
            sizes = []
            popularity = []
            for items in group_items:
                indices = [item_index[item_id] for item_id in items]
                vectors.append(residuals[layer][indices].mean(axis=0))
                sizes.append(len(items))
                popularity.append(sum(item_popularity.get(item_id, 0) for item_id in items))

            labels, centers = balanced_refinement(
                vectors=np.asarray(vectors, dtype=np.float64),
                sizes=np.asarray(sizes, dtype=np.float64),
                popularity=np.asarray(popularity, dtype=np.float64),
                n_clusters=split_count,
                balance_weight=args.balance_weight,
                max_iter=args.kmeans_max_iter,
                seed=args.seed,
            )

            split_token_ids = [int(old_token_id)]
            for _ in range(1, split_count):
                split_token_ids.append(int(next_token_id))
                next_token_id += 1

            codebooks_by_layer_token[layer][old_token_id] = centers[0]
            for cluster_idx in range(1, split_count):
                codebooks_by_layer_token[layer][split_token_ids[cluster_idx]] = centers[cluster_idx]

            cluster_popularity = defaultdict(int)
            for group_idx, items in enumerate(group_items):
                new_token_id = split_token_ids[int(labels[group_idx])]
                for item_id in items:
                    item2tokens[item_id][layer] = new_token_id
                    cluster_popularity[int(labels[group_idx])] += int(item_popularity.get(item_id, 0))

            record = {
                "layer": int(layer),
                "old_token_id": int(old_token_id),
                "popularity": int(token_popularity),
                "mean_layer_popularity": mean_popularity,
                "num_affected_items": len(affected_items),
                "num_groups": len(group_items),
                "num_splits": int(split_count),
                "split_token_ids": split_token_ids,
                "cluster_popularity": {str(k): int(v) for k, v in sorted(cluster_popularity.items())},
            }
            selected_records.append(record)
            split_mapping[str(old_token_id)] = record

    next_token_id, duplicate_resolutions = resolve_duplicate_item_tokens(
        item2tokens=item2tokens,
        item_popularity=item_popularity,
        residuals=residuals,
        item_index=item_index,
        codebooks_by_layer_token=codebooks_by_layer_token,
        next_token_id=next_token_id,
    )

    user_token_start_idx = max(max(tokens) for tokens in item2tokens.values()) + 1
    user2tokens = {user_id: hash_user_id(user_id, user_token_start_idx, num_user_tokens) for user_id in set(user_ids)}
    tokens2item = {tuple(tokens): item_id for item_id, tokens in item2tokens.items()}
    if len(tokens2item) != len(item2tokens):
        raise RuntimeError("Duplicate semantic IDs remain after duplicate resolution.")

    if output_tokenizer_dir.exists() and args.overwrite:
        shutil.rmtree(output_tokenizer_dir)
    output_tokenizer_dir.mkdir(parents=True, exist_ok=True)

    dump_json({str(k): [int(x) for x in v] for k, v in item2tokens.items()}, output_tokenizer_dir / "item2tokens.json")
    dump_json({str(k): int(v) for k, v in user2tokens.items()}, output_tokenizer_dir / "item2tokens_users.json")
    dump_json({str(k): int(v) for k, v in tokens2item.items()}, output_tokenizer_dir / "item2tokens_tokens2item.json")
    dump_json({str(k): [int(x) for x in v] for k, v in item2tokens.items()}, output_dir / "updated_item_sids.json")
    dump_json(selected_records, output_dir / "selected_overpopular_tokens.json")
    dump_json(split_mapping, output_dir / "split_mapping.json")
    dump_json(duplicate_resolutions, output_dir / "duplicate_sid_resolutions.json")

    after_popularity = token_popularity_by_layer(
        {k: tuple(v) for k, v in item2tokens.items()},
        item_popularity,
        n_codebooks,
        layer_token_ids=[sorted(layer_map) for layer_map in codebooks_by_layer_token],
    )
    dump_json(
        {
            "token_popularity": [{str(k): int(v) for k, v in layer.items()} for layer in after_popularity],
            "summary": summarize_layer_popularity(after_popularity, args.analysis_top_ratio),
        },
        output_dir / "token_popularity_after.json",
    )

    codebook_payload = {
        "layer_token_ids": [sorted(layer_map) for layer_map in codebooks_by_layer_token],
        "codebooks": [
            torch.from_numpy(np.asarray([layer_map[token_id] for token_id in sorted(layer_map)], dtype=np.float32))
            for layer_map in codebooks_by_layer_token
        ],
        "user_token_start_idx": user_token_start_idx,
        "num_user_tokens": num_user_tokens,
    }
    torch.save(codebook_payload, output_dir / "updated_codebooks.pt")
    for layer, layer_map in enumerate(codebooks_by_layer_token):
        np.save(output_dir / f"updated_codebook_layer_{layer}.npy", np.asarray([layer_map[t] for t in sorted(layer_map)], dtype=np.float32))

    rebuilt_dir = output_dir / "rebuilt_data"
    rebuilt_dir.mkdir(parents=True, exist_ok=True)
    rebuild_samples(user_sequences, item2tokens, args.max_seq_len, rebuilt_dir)

    with (output_tokenizer_dir / "item2tokens_item_popularity_tokens.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "popularity", "token_ids"])
        for item_id in sorted(item2tokens):
            writer.writerow([item_id, int(item_popularity.get(item_id, 0)), json.dumps(item2tokens[item_id])])

    tokenizer_config = dict(config)
    tokenizer_config["save_path"] = str(output_tokenizer_dir / "item2tokens.json")
    tokenizer_config["checkpoint_path"] = str(output_tokenizer_dir / "tokenizer_checkpoint.pth")
    tokenizer_config["tokenizer_path"] = str(output_tokenizer_dir / "tokenizer.pkl")
    rebalanced_tokenizer = RQVAETokenizer(tokenizer_config)
    rebalanced_tokenizer.item2tokens = {k: tuple(v) for k, v in item2tokens.items()}
    rebalanced_tokenizer.tokens2item = tokens2item
    rebalanced_tokenizer.user2tokens = user2tokens
    rebalanced_tokenizer.user_token_start_idx = user_token_start_idx
    with (output_tokenizer_dir / "tokenizer.pkl").open("wb") as f:
        pickle.dump(rebalanced_tokenizer, f)

    report = {
        "source_tokenizer_dir": str(source_tokenizer_dir),
        "output_tokenizer_dir": str(output_tokenizer_dir),
        "num_items": len(item2tokens),
        "num_selected_tokens": len(selected_records),
        "num_duplicate_resolutions": len(duplicate_resolutions),
        "semantic_vocab_size": user_token_start_idx,
        "vocab_size_with_user_tokens": user_token_start_idx + num_user_tokens,
        "crab": {
            "split_ratio": args.split_ratio,
            "analysis_top_ratio": args.analysis_top_ratio,
            "max_splits": args.max_splits,
            "balance_weight": args.balance_weight,
            "kmeans_max_iter": args.kmeans_max_iter,
            "seed": args.seed,
        },
    }
    dump_json(report, output_dir / "crab_lite_report.json")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRAB-lite codebook rebalancing for TIGER RQ-VAE tokenizers.")
    parser.add_argument("--config", default="config/tokenizer/rqvae.yaml")
    parser.add_argument("--tokenizer-dir", required=True, help="Source tokenizer_model directory.")
    parser.add_argument("--output-tokenizer-dir", required=True, help="Destination rebalanced tokenizer_model directory.")
    parser.add_argument("--data-text-files", required=True)
    parser.add_argument("--interaction-file", required=True)
    parser.add_argument("--split-ratio", type=float, default=None)
    parser.add_argument("--analysis-top-ratio", type=float, default=None)
    parser.add_argument("--max-splits", type=int, default=None)
    parser.add_argument("--balance-weight", type=float, default=None)
    parser.add_argument("--kmeans-max-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path for CRAB-lite stdout/stderr log. Defaults to <output-tokenizer-dir>/../crab_lite.log.",
    )
    parser.add_argument("--append-log", action="store_true", help="Append to --log-file instead of replacing it.")
    return parser.parse_args()


if __name__ == "__main__":
    run_with_log(parse_args())
