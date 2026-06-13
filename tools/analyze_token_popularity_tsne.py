#!/usr/bin/env python3
"""Empirical token-popularity binding analysis.

Experiments implemented:
1. Codebook-vector t-SNE for baseline tokenizers.
2. Item residual t-SNE for popular/unpopular item groups.
3. Raw item embedding / RQ-VAE encoder latent t-SNE.
4. Top-K candidate coverage and margin analysis.

The analysis intentionally writes to an undated output directory because these
artifacts are exploratory diagnostics rather than experiment runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.quantization.data.dataset.rqvae_dataset import ItemEmbeddingDataset
from tools.crab_lite_rebalance import load_item2tokens, load_rqvae_tokenizer


RESERVE_TOKENS = 100
CODEBOOK_SIZE = 256
N_CODEBOOKS = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_path(path_text: str, repo: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if path.is_absolute() and "output" in path.parts:
        return repo.joinpath(*path.parts[path.parts.index("output") :])
    return repo / path_text


def load_interactions(path: Path) -> tuple[dict[int, int], dict[int, list[int]]]:
    with path.open("rb") as f:
        frame = pickle.load(f)
    popularity = Counter()
    user_sequences = {}
    for _, row in frame.iterrows():
        user = int(row["UserID"])
        seq = [int(item) for item in row["ItemID"]]
        user_sequences[user] = seq
        popularity.update(seq)
    return dict(popularity), user_sequences


def popular_items_by_interaction_half(item_popularity: dict[int, int]) -> set[int]:
    total = sum(item_popularity.values())
    target = total * 0.5
    running = 0
    popular = set()
    for item_id, count in sorted(item_popularity.items(), key=lambda kv: (-kv[1], kv[0])):
        if running >= target:
            break
        popular.add(int(item_id))
        running += int(count)
    return popular


def sem_ids_from_tokens(tokens: tuple[int, ...], n_codebooks: int = N_CODEBOOKS) -> tuple[int, ...]:
    sem_ids = []
    for layer in range(n_codebooks):
        sem_ids.append(int(tokens[layer]) - RESERVE_TOKENS - CODEBOOK_SIZE * layer)
    return tuple(sem_ids)


def load_item_embeddings(data_text_file: Path, config: dict[str, Any], device: str) -> dict[int, np.ndarray]:
    dataset = ItemEmbeddingDataset(
        data_text_files=str(data_text_file),
        config=config,
        text_encoder_model=config["text_encoder_model"],
        embedding_extraction_strategy=config.get("embedding_strategy", "mean_pooling"),
        device=device,
    )
    return {int(item_id): np.asarray(vec, dtype=np.float32) for item_id, vec in dataset.item_embeddings.items()}


def reduce_for_tsne(vectors: np.ndarray, seed: int) -> np.ndarray:
    if vectors.shape[1] <= 50:
        return vectors
    n_components = min(50, vectors.shape[0] - 1, vectors.shape[1])
    return PCA(n_components=n_components, random_state=seed).fit_transform(vectors)


def run_tsne(vectors: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    if vectors.shape[0] < 3:
        return np.zeros((vectors.shape[0], 2), dtype=np.float32)
    effective_perplexity = min(perplexity, max(2.0, (vectors.shape[0] - 1) / 3))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(reduce_for_tsne(vectors, seed))


def sample_items(
    item_ids: list[int],
    popular_items: set[int],
    max_items: int,
    seed: int,
) -> list[int]:
    if len(item_ids) <= max_items:
        return item_ids
    rng = np.random.default_rng(seed)
    popular = [item for item in item_ids if item in popular_items]
    unpopular = [item for item in item_ids if item not in popular_items]
    half = max_items // 2
    n_pop = min(len(popular), half)
    n_unpop = min(len(unpopular), max_items - n_pop)
    if n_pop + n_unpop < max_items:
        n_pop = min(len(popular), max_items - n_unpop)
    sampled = []
    if n_pop:
        sampled.extend(rng.choice(popular, size=n_pop, replace=False).tolist())
    if n_unpop:
        sampled.extend(rng.choice(unpopular, size=n_unpop, replace=False).tolist())
    return sorted(int(item) for item in sampled)


def token_stats(
    item2sem: dict[int, tuple[int, ...]],
    item_popularity: dict[int, int],
    popular_items: set[int],
    method: str,
    dataset: str,
) -> list[dict[str, Any]]:
    rows = []
    for layer in range(N_CODEBOOKS):
        usage = Counter()
        mass = Counter()
        pop_mass = Counter()
        unpop_mass = Counter()
        for item_id, sem_ids in item2sem.items():
            token_id = int(sem_ids[layer])
            item_mass = int(item_popularity.get(item_id, 0))
            usage[token_id] += 1
            mass[token_id] += item_mass
            if item_id in popular_items:
                pop_mass[token_id] += item_mass
            else:
                unpop_mass[token_id] += item_mass
        for token_id in range(CODEBOOK_SIZE):
            total_mass = int(mass[token_id])
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "layer": layer,
                    "token_id": token_id,
                    "item_count": int(usage[token_id]),
                    "popularity_mass": total_mass,
                    "popular_mass": int(pop_mass[token_id]),
                    "unpopular_mass": int(unpop_mass[token_id]),
                    "popular_mass_ratio": float(pop_mass[token_id] / total_mass) if total_mass > 0 else 0.0,
                }
            )
    return rows


def plot_codebook_tsne(
    output_dir: Path,
    dataset: str,
    method: str,
    layer: int,
    codebook: np.ndarray,
    stats_rows: list[dict[str, Any]],
    seed: int,
    perplexity: float,
) -> None:
    figure_dir = output_dir / dataset / method / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("log_popularity_mass", "viridis"),
        ("popular_mass_ratio", "coolwarm"),
    ]
    if all((figure_dir / f"codebook_tsne_layer{layer}_{name}.png").exists() for name, _ in plot_specs):
        return

    coords = run_tsne(codebook.astype(np.float32), seed, perplexity)
    layer_rows = [row for row in stats_rows if row["layer"] == layer]
    mass = np.asarray([float(row["popularity_mass"]) for row in layer_rows], dtype=np.float32)
    pop_ratio = np.asarray([float(row["popular_mass_ratio"]) for row in layer_rows], dtype=np.float32)

    for values, name, cmap in [
        (np.log1p(mass), "log_popularity_mass", "viridis"),
        (pop_ratio, "popular_mass_ratio", "coolwarm"),
    ]:
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=values, s=24, cmap=cmap, alpha=0.9)
        plt.colorbar(scatter, label=name)
        plt.title(f"{dataset} {method} layer {layer}: codebook t-SNE")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(figure_dir / f"codebook_tsne_layer{layer}_{name}.png", dpi=180)
        plt.close()


def plot_residual_tsne(
    output_dir: Path,
    dataset: str,
    method: str,
    layer: int,
    item_ids: list[int],
    residuals: np.ndarray,
    codebook: np.ndarray,
    assignments: np.ndarray,
    item_popularity: dict[int, int],
    popular_items: set[int],
    seed: int,
    perplexity: float,
) -> dict[str, Any]:
    is_pop = np.asarray([item in popular_items for item in item_ids], dtype=bool)
    popularity_values = np.asarray([item_popularity.get(item, 0) for item in item_ids], dtype=np.float32)

    figure_dir = output_dir / dataset / method / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"residual_tsne_layer{layer}_item_pop_group.png"
    if not figure_path.exists():
        vectors = np.vstack([residuals, codebook.astype(np.float32)])
        coords = run_tsne(vectors, seed, perplexity)
        item_coords = coords[: len(item_ids)]
        code_coords = coords[len(item_ids) :]

        plt.figure(figsize=(7, 6))
        plt.scatter(
            item_coords[~is_pop, 0],
            item_coords[~is_pop, 1],
            s=8,
            c="#4C78A8",
            alpha=0.45,
            label="unpopular item",
        )
        plt.scatter(
            item_coords[is_pop, 0],
            item_coords[is_pop, 1],
            s=10,
            c="#E45756",
            alpha=0.65,
            label="popular item",
        )
        plt.scatter(code_coords[:, 0], code_coords[:, 1], s=34, c="black", alpha=0.55, marker="x", label="codeword")
        plt.title(f"{dataset} {method} layer {layer}: residual t-SNE")
        plt.legend(loc="best", frameon=False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(figure_path, dpi=180)
        plt.close()

    quantized = codebook[assignments]
    quant_error = np.linalg.norm(residuals - quantized, axis=1)
    pop_error = quant_error[is_pop]
    unpop_error = quant_error[~is_pop]
    pop_centroid = residuals[is_pop].mean(axis=0) if is_pop.any() else np.zeros(residuals.shape[1])
    unpop_centroid = residuals[~is_pop].mean(axis=0) if (~is_pop).any() else np.zeros(residuals.shape[1])
    silhouette = np.nan
    if is_pop.any() and (~is_pop).any() and len(item_ids) >= 10:
        try:
            sample_vectors = residuals
            labels = is_pop.astype(int)
            if len(item_ids) > 5000:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(item_ids), size=5000, replace=False)
                sample_vectors = residuals[idx]
                labels = labels[idx]
            silhouette = float(silhouette_score(sample_vectors, labels))
        except ValueError:
            silhouette = np.nan

    return {
        "dataset": dataset,
        "method": method,
        "layer": layer,
        "num_items_plotted": len(item_ids),
        "popular_items_plotted": int(is_pop.sum()),
        "unpopular_items_plotted": int((~is_pop).sum()),
        "popular_mean_log_interactions": float(np.log1p(popularity_values[is_pop]).mean()) if is_pop.any() else np.nan,
        "unpopular_mean_log_interactions": float(np.log1p(popularity_values[~is_pop]).mean()) if (~is_pop).any() else np.nan,
        "popular_mean_quant_error": float(pop_error.mean()) if pop_error.size else np.nan,
        "unpopular_mean_quant_error": float(unpop_error.mean()) if unpop_error.size else np.nan,
        "pop_unpop_centroid_distance": float(np.linalg.norm(pop_centroid - unpop_centroid)),
        "popular_unpopular_silhouette": silhouette,
    }


def plot_vector_popularity_tsne(
    output_dir: Path,
    dataset: str,
    method: str,
    name: str,
    item_ids: list[int],
    vectors: np.ndarray,
    item_popularity: dict[int, int],
    popular_items: set[int],
    seed: int,
    perplexity: float,
) -> dict[str, Any]:
    is_pop = np.asarray([item in popular_items for item in item_ids], dtype=bool)
    popularity_values = np.asarray([item_popularity.get(item, 0) for item in item_ids], dtype=np.float32)

    figure_dir = output_dir / dataset / method / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"{name}_tsne_item_pop_group.png"
    if not figure_path.exists():
        coords = run_tsne(vectors.astype(np.float32), seed, perplexity)
        plt.figure(figsize=(7, 6))
        plt.scatter(
            coords[~is_pop, 0],
            coords[~is_pop, 1],
            s=8,
            c="#4C78A8",
            alpha=0.45,
            label="unpopular item",
        )
        plt.scatter(
            coords[is_pop, 0],
            coords[is_pop, 1],
            s=10,
            c="#E45756",
            alpha=0.65,
            label="popular item",
        )
        plt.title(f"{dataset} {method}: {name.replace('_', ' ')} t-SNE")
        plt.legend(loc="best", frameon=False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(figure_path, dpi=180)
        plt.close()

    pop_centroid = vectors[is_pop].mean(axis=0) if is_pop.any() else np.zeros(vectors.shape[1])
    unpop_centroid = vectors[~is_pop].mean(axis=0) if (~is_pop).any() else np.zeros(vectors.shape[1])
    silhouette = np.nan
    if is_pop.any() and (~is_pop).any() and len(item_ids) >= 10:
        try:
            sample_vectors = vectors
            labels = is_pop.astype(int)
            if len(item_ids) > 5000:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(item_ids), size=5000, replace=False)
                sample_vectors = vectors[idx]
                labels = labels[idx]
            silhouette = float(silhouette_score(sample_vectors, labels))
        except ValueError:
            silhouette = np.nan

    return {
        "dataset": dataset,
        "method": method,
        "vector_name": name,
        "num_items_plotted": len(item_ids),
        "popular_items_plotted": int(is_pop.sum()),
        "unpopular_items_plotted": int((~is_pop).sum()),
        "popular_mean_log_interactions": float(np.log1p(popularity_values[is_pop]).mean()) if is_pop.any() else np.nan,
        "unpopular_mean_log_interactions": float(np.log1p(popularity_values[~is_pop]).mean()) if (~is_pop).any() else np.nan,
        "pop_unpop_centroid_distance": float(np.linalg.norm(pop_centroid - unpop_centroid)),
        "popular_unpopular_silhouette": silhouette,
    }


def compute_topk_candidate_metrics(
    dataset: str,
    method: str,
    layer: int,
    residuals: np.ndarray,
    codebook: np.ndarray,
    assignments: np.ndarray,
    top_k_values: list[int],
    temperature: float,
) -> list[dict[str, Any]]:
    distances = ((residuals[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
    order = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, order, axis=1)
    ranks = np.empty(len(assignments), dtype=np.int64)
    for row_idx, token_id in enumerate(assignments):
        matches = np.flatnonzero(order[row_idx] == int(token_id))
        ranks[row_idx] = int(matches[0]) + 1 if len(matches) else codebook.shape[0] + 1

    top1 = sorted_distances[:, 0]
    top2 = sorted_distances[:, 1] if sorted_distances.shape[1] > 1 else sorted_distances[:, 0]
    eps = 1e-8
    margin = top2 - top1
    relative_margin = margin / (top1 + eps)
    assigned_distance = distances[np.arange(len(assignments)), assignments]

    logits = -distances / max(float(temperature), eps)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), eps)

    rows = []
    for top_k in top_k_values:
        k = min(int(top_k), codebook.shape[0])
        topk_ids = order[:, :k]
        coverage = (ranks <= k).mean()
        topk_mass = np.take_along_axis(probs, topk_ids, axis=1).sum(axis=1)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "layer": layer,
                "top_k": k,
                "coverage_rate": float(coverage),
                "mean_hard_rank": float(ranks.mean()),
                "p95_hard_rank": float(np.percentile(ranks, 95)),
                "top1_match_rate": float((ranks == 1).mean()),
                "mean_top1_top2_margin": float(margin.mean()),
                "mean_relative_margin": float(relative_margin.mean()),
                "mean_assigned_distance": float(assigned_distance.mean()),
                "mean_top1_distance": float(top1.mean()),
                "mean_topk_soft_mass": float(topk_mass.mean()),
                "p05_topk_soft_mass": float(np.percentile(topk_mass, 5)),
            }
        )
    return rows


def extract_vae_codebooks_and_residuals(
    tokenizer_dir: Path,
    config: dict[str, Any],
    item_ids: list[int],
    embeddings: np.ndarray,
    item2sem: dict[int, tuple[int, ...]],
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    tokenizer = load_rqvae_tokenizer(config, tokenizer_dir, device, "tiger")
    codebooks = [
        layer.embedding.weight.detach().cpu().numpy().astype(np.float32)
        for layer in tokenizer.rq_vae.rq.vq_layers
    ]
    sem_matrix = np.asarray([item2sem[item] for item in item_ids], dtype=np.int64)
    residuals_by_layer = []
    latent_parts = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(item_ids), batch_size):
            batch_embeddings = torch.from_numpy(embeddings[start : start + batch_size]).to(device)
            batch_sem = torch.from_numpy(sem_matrix[start : start + batch_size]).to(device)
            latent = tokenizer.rq_vae.encoder(batch_embeddings)
            latent_parts.append(latent.detach().cpu().numpy().astype(np.float32))
            residual = latent
            batch_residuals = []
            for layer, quantizer in enumerate(tokenizer.rq_vae.rq.vq_layers):
                batch_residuals.append(residual.detach().cpu().numpy().astype(np.float32))
                code = quantizer.embedding(batch_sem[:, layer])
                residual = residual - code
            if not residuals_by_layer:
                residuals_by_layer = [[] for _ in range(len(batch_residuals))]
            for layer, batch_residual in enumerate(batch_residuals):
                residuals_by_layer[layer].append(batch_residual)
    residuals_by_layer = [np.vstack(parts) for parts in residuals_by_layer]
    return codebooks, residuals_by_layer, sem_matrix, np.vstack(latent_parts)


def extract_rqkmeans_codebooks_and_residuals(
    item_ids: list[int],
    embeddings: np.ndarray,
    item2sem: dict[int, tuple[int, ...]],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    sem_matrix = np.asarray([item2sem[item] for item in item_ids], dtype=np.int64)
    residual = embeddings.astype(np.float32).copy()
    codebooks = []
    residuals_by_layer = []
    for layer in range(N_CODEBOOKS):
        residuals_by_layer.append(residual.copy())
        centers = np.zeros((CODEBOOK_SIZE, residual.shape[1]), dtype=np.float32)
        for token_id in range(CODEBOOK_SIZE):
            mask = sem_matrix[:, layer] == token_id
            if mask.any():
                centers[token_id] = residual[mask].mean(axis=0)
        codebooks.append(centers)
        residual = residual - centers[sem_matrix[:, layer]]
    return codebooks, residuals_by_layer, sem_matrix


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_token_layers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    keys = sorted({(row["dataset"], row["method"], row["layer"]) for row in rows})
    for dataset, method, layer in keys:
        layer_rows = [
            row for row in rows
            if row["dataset"] == dataset and row["method"] == method and row["layer"] == layer
        ]
        masses = [float(row["popularity_mass"]) for row in layer_rows]
        used_rows = [row for row in layer_rows if float(row["popularity_mass"]) > 0]
        ratios = [float(row["popular_mass_ratio"]) for row in used_rows]
        total_mass = sum(masses)
        top_n = max(1, int(len(masses) * 0.05))
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "layer": layer,
                "used_tokens": len(used_rows),
                "top5pct_token_mass_share": (
                    sum(sorted(masses, reverse=True)[:top_n]) / total_mass if total_mass > 0 else 0.0
                ),
                "tokens_popular_ratio_ge_0.9": sum(1 for ratio in ratios if ratio >= 0.9),
                "tokens_popular_ratio_le_0.1": sum(1 for ratio in ratios if ratio <= 0.1),
                "mean_popular_mass_ratio_used_tokens": float(np.mean(ratios)) if ratios else 0.0,
            }
        )
    return summary_rows


def find_rqkmeans_dir(root: Path, dataset: str) -> Path | None:
    candidates = [
        root / dataset / "original" / "tokenizer_model",
        root / dataset.capitalize() / "original" / "tokenizer_model",
        root / dataset.upper() / "original" / "tokenizer_model",
    ]
    for candidate in candidates:
        if (candidate / "item2tokens.json").exists():
            return candidate
    return None


def analyze_dataset(
    dataset: str,
    tokenizer_dir: Path,
    interaction_file: Path,
    text_file: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    status_rows: list[dict[str, Any]],
    token_stat_rows: list[dict[str, Any]],
    residual_metric_rows: list[dict[str, Any]],
    vector_metric_rows: list[dict[str, Any]],
    topk_metric_rows: list[dict[str, Any]],
) -> None:
    item_popularity, _ = load_interactions(interaction_file)
    popular_items = popular_items_by_interaction_half(item_popularity)
    item_embeddings = load_item_embeddings(text_file, config, args.device)

    methods = [("rqvae", tokenizer_dir)]
    rqkmeans_dir = find_rqkmeans_dir(Path(args.rqkmeans_root), dataset)
    if rqkmeans_dir is not None:
        methods.append(("rqkmeans", rqkmeans_dir))
    else:
        status_rows.append({"dataset": dataset, "method": "rqkmeans", "status": "skipped_missing_tokenizer"})

    for method, method_tokenizer_dir in methods:
        item2tokens = load_item2tokens(method_tokenizer_dir / "item2tokens.json")
        item2sem = {
            item_id: sem_ids_from_tokens(tokens)
            for item_id, tokens in item2tokens.items()
            if item_id in item_embeddings
        }
        item_ids_all = sorted(item2sem)
        sampled_ids = sample_items(item_ids_all, popular_items, args.max_items, args.seed)
        embeddings = np.asarray([item_embeddings[item_id] for item_id in sampled_ids], dtype=np.float32)
        sampled_item2sem = {item_id: item2sem[item_id] for item_id in sampled_ids}

        stats_rows = token_stats(item2sem, item_popularity, popular_items, method, dataset)
        token_stat_rows.extend(stats_rows)

        if method == "rqvae":
            codebooks, residuals_by_layer, sem_matrix, encoder_latent = extract_vae_codebooks_and_residuals(
                method_tokenizer_dir,
                config,
                sampled_ids,
                embeddings,
                sampled_item2sem,
                args.device,
            )
        else:
            codebooks, residuals_by_layer, sem_matrix = extract_rqkmeans_codebooks_and_residuals(
                sampled_ids,
                embeddings,
                sampled_item2sem,
            )
            encoder_latent = None

        vector_metric_rows.append(
            plot_vector_popularity_tsne(
                Path(args.output_dir),
                dataset,
                method,
                "item_embedding",
                sampled_ids,
                embeddings,
                item_popularity,
                popular_items,
                args.seed,
                args.perplexity,
            )
        )
        if method == "rqvae":
            vector_metric_rows.append(
                plot_vector_popularity_tsne(
                    Path(args.output_dir),
                    dataset,
                    method,
                    "encoder_latent",
                    sampled_ids,
                    encoder_latent,
                    item_popularity,
                    popular_items,
                    args.seed,
                    args.perplexity,
                )
            )

        for layer in range(N_CODEBOOKS):
            plot_codebook_tsne(
                Path(args.output_dir),
                dataset,
                method,
                layer,
                codebooks[layer],
                stats_rows,
                args.seed,
                args.perplexity,
            )
            metric_row = plot_residual_tsne(
                Path(args.output_dir),
                dataset,
                method,
                layer,
                sampled_ids,
                residuals_by_layer[layer],
                codebooks[layer],
                sem_matrix[:, layer],
                item_popularity,
                popular_items,
                args.seed,
                args.perplexity,
            )
            residual_metric_rows.append(metric_row)
            topk_metric_rows.extend(
                compute_topk_candidate_metrics(
                    dataset,
                    method,
                    layer,
                    residuals_by_layer[layer],
                    codebooks[layer],
                    sem_matrix[:, layer],
                    args.top_k_values,
                    args.candidate_temperature,
                )
            )

        status_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "status": "ok",
                "tokenizer_dir": str(method_tokenizer_dir),
                "num_items_total": len(item_ids_all),
                "num_items_sampled": len(sampled_ids),
                "num_popular_items_total": len(popular_items),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", default="output/20260609_baseline_hparams/best_by_dataset.csv")
    parser.add_argument("--output-dir", default="output/analysis/token_popularity_binding")
    parser.add_argument("--rqkmeans-root", default="output/rqkmeans")
    parser.add_argument("--config", default="config/tokenizer/rqvae.yaml")
    parser.add_argument("--datasets", nargs="*", default=["electronic", "movielens", "sports", "toy"])
    parser.add_argument("--max-items", type=int, default=4000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k-values", nargs="*", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--candidate-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = REPO_ROOT
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    baseline_rows = {row["dataset"]: row for row in read_csv(resolve_path(args.baseline_csv, repo))}
    status_rows: list[dict[str, Any]] = []
    token_stat_rows: list[dict[str, Any]] = []
    residual_metric_rows: list[dict[str, Any]] = []
    vector_metric_rows: list[dict[str, Any]] = []
    topk_metric_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        row = baseline_rows.get(dataset)
        if row is None:
            status_rows.append({"dataset": dataset, "method": "rqvae", "status": "skipped_missing_baseline"})
            continue
        output_dir = resolve_path(row["output_dir"], repo)
        tokenizer_dir = output_dir / "tokenizer_model"
        interaction_file = repo / "data" / dataset / "user2item.pkl"
        text_file = repo / "data" / dataset / "item2title.pkl"
        dataset_config = dict(config)
        dataset_config["data_text_files"] = str(text_file)
        dataset_config["interaction_files"] = str(interaction_file)
        print(f"[analyze] dataset={dataset} tokenizer={tokenizer_dir}")
        analyze_dataset(
            dataset,
            tokenizer_dir,
            interaction_file,
            text_file,
            dataset_config,
            args,
            status_rows,
            token_stat_rows,
            residual_metric_rows,
            vector_metric_rows,
            topk_metric_rows,
        )

    output_dir = Path(args.output_dir)
    write_rows(output_dir / "token_codebook_stats.csv", token_stat_rows)
    write_rows(output_dir / "token_layer_summary.csv", summarize_token_layers(token_stat_rows))
    write_rows(output_dir / "residual_group_metrics.csv", residual_metric_rows)
    write_rows(output_dir / "embedding_latent_group_metrics.csv", vector_metric_rows)
    write_rows(output_dir / "topk_candidate_metrics.csv", topk_metric_rows)
    write_rows(output_dir / "run_status.csv", status_rows)
    with (output_dir / "analysis_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"[done] wrote analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
