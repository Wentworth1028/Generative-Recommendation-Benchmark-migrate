from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from genrec.ghost.utils import l2_normalize, longest_common_prefix_length
from genrec.quantization.tokenizers.base_tokenizer import AbstractTokenizer


class GhostSKTTokenizer(AbstractTokenizer):
    """Skeleton-founded tokenizer used by GHOST and SKT-only variants."""

    ARTIFACT_VERSION = 2

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = dict(config)

        self.pad_token = 0
        self.eos_token = 1
        self.bos_token = self.pad_token
        self.reserve_tokens = int(self.config.get("reserve_tokens", 2))
        self.num_user_tokens = 0
        self.ignored_label = -100

        self.head_ratio = float(self.config.get("head_ratio", 0.2))
        self.skeleton_length = int(self.config.get("skeleton_length", 4))
        self.tail_extra_length = int(self.config.get("tail_extra_length", 2))
        self.semantic_dim = int(self.config.get("semantic_dim", 32))
        self.codebook_size = int(self.config.get("codebook_size", 256))
        # The shared dataset/backbone contract expects the maximum number of
        # codebook positions, while individual GHOST items may be shorter.
        self.n_codebooks = self.skeleton_length + self.tail_extra_length
        self.digits = self.n_codebooks
        self.user_token_start_idx = self._item_vocab_size()
        self.projection_method = str(self.config.get("projection_method", "pca")).lower()
        self.random_state = int(self.config.get("seed", self.config.get("random_state", 42)))
        self.kmeans_n_init = self.config.get("kmeans_n_init", "auto")
        self.kmeans_max_iter = int(self.config.get("kmeans_max_iter", 100))
        self.collision_repair_enabled = bool(self.config.get("collision_repair_enabled", True))
        self.collision_candidates_per_layer = int(self.config.get("collision_candidates_per_layer", 8))
        self.collision_beam_width = int(self.config.get("collision_beam_width", 256))
        self.auo_ka = int(self.config.get("auo_ka", 200))
        self.auo_kb = int(self.config.get("auo_kb", 5))

        if not 0.0 < self.head_ratio < 1.0:
            raise ValueError(f"GHOST head_ratio must be in (0, 1), got {self.head_ratio}.")
        if self.skeleton_length <= 0 or self.tail_extra_length <= 0:
            raise ValueError("GHOST skeleton_length and tail_extra_length must be positive.")
        if self.semantic_dim <= 0 or self.codebook_size <= 1:
            raise ValueError("GHOST semantic_dim must be positive and codebook_size must exceed one.")
        if self.reserve_tokens < 2:
            raise ValueError("GHOST must reserve at least PAD and EOS tokens.")
        if self.auo_ka <= 0 or self.auo_kb <= 0 or self.auo_kb > self.auo_ka:
            raise ValueError("GHOST requires 0 < auo_kb <= auo_ka.")
        if self.collision_candidates_per_layer <= 0 or self.collision_beam_width <= 0:
            raise ValueError("GHOST collision search widths must be positive.")

        self.item2tokens: Dict[int, list[int]] = {}
        self.tokens2item: Dict[tuple[int, ...], int] = {}
        self.item_lengths: Dict[int, int] = {}
        self.item_groups: Dict[int, str] = {}
        self.nearest_head: Dict[int, int] = {}
        self.head_item_ids: list[int] = []
        self.tail_item_ids: list[int] = []
        self.head_codebooks: np.ndarray | None = None
        self.tail_codebooks: np.ndarray | None = None
        self.semantic_embeddings: np.ndarray | None = None
        self.projector_state: dict[str, Any] = {}
        self.undesired_collection: dict[str, Any] = {}

        self.save_path = self.config["save_path"]
        self.metadata_path = self._metadata_path(self.save_path)
        self.tokens2item_save_path = self.config.get(
            "tokens2item_save_path",
            self.save_path.replace(".json", "_tokens2item.json"),
        )
        self.head_items_path = self.save_path.replace(".json", "_head_items.json")
        self.tail_items_path = self.save_path.replace(".json", "_tail_items.json")
        self.undesired_collection_path = self.save_path.replace(".json", "_undesired_collection.json")
        self.semantic_embedding_path = self.save_path.replace(".json", "_semantic_embeddings.npy")
        self.head_codebook_path = self.save_path.replace(".json", "_head_codebooks.npy")
        self.tail_codebook_path = self.save_path.replace(".json", "_tail_codebooks.npy")
        self.projector_component_path = self.save_path.replace(".json", "_projector_components.npy")
        self.projector_mean_path = self.save_path.replace(".json", "_projector_mean.npy")

    @staticmethod
    def _metadata_path(save_path: str) -> str:
        return save_path.replace("item2tokens.json", "ghost_metadata.json")

    @property
    def total_sid_length(self) -> int:
        return self.skeleton_length + self.tail_extra_length

    def _item_vocab_size(self) -> int:
        return self.reserve_tokens + self.total_sid_length * self.codebook_size

    def _codes_to_tokens(self, codes: Iterable[int], *, start_layer: int = 0) -> list[int]:
        return [
            int(code) + self.reserve_tokens + (start_layer + layer_idx) * self.codebook_size
            for layer_idx, code in enumerate(codes)
        ]

    def _find_unique_codes(
        self,
        vector: np.ndarray,
        codebooks: np.ndarray,
        preferred_codes: Iterable[int],
        used_codes: set[tuple[int, ...]],
        *,
        valid_centers: int,
        context: str,
    ) -> np.ndarray:
        """Find the nearest unused residual-quantization path with a small beam.

        RQ-KMeans does not guarantee unique code tuples. GHOST needs an
        invertible SID-to-item map, so only colliding items are reassigned to
        the closest available path under the already fitted codebooks.
        """
        preferred = tuple(int(code) for code in preferred_codes)
        if preferred not in used_codes:
            return np.asarray(preferred, dtype=np.int64)
        if not self.collision_repair_enabled:
            raise ValueError(f"Duplicate GHOST SID in {context}: {preferred}.")

        valid_centers = max(1, min(int(valid_centers), self.codebook_size))
        candidate_widths = []
        width = min(self.collision_candidates_per_layer, valid_centers)
        while width < valid_centers:
            candidate_widths.append(width)
            width = min(valid_centers, width * 2)
        candidate_widths.append(valid_centers)

        for candidate_width in candidate_widths:
            beams: list[tuple[float, tuple[int, ...], np.ndarray]] = [
                (float(np.dot(vector, vector)), tuple(), vector.astype(np.float32, copy=True))
            ]
            for layer_idx in range(codebooks.shape[0]):
                centers = codebooks[layer_idx, :valid_centers]
                expanded: list[tuple[float, tuple[int, ...], np.ndarray]] = []
                for _, prefix, residual in beams:
                    distances = ((centers - residual[None, :]) ** 2).sum(axis=1)
                    if candidate_width < valid_centers:
                        nearest = np.argpartition(distances, candidate_width - 1)[:candidate_width]
                    else:
                        nearest = np.arange(valid_centers)
                    nearest = nearest[np.argsort(distances[nearest], kind="stable")]
                    for code in nearest:
                        next_residual = residual - centers[int(code)]
                        score = float(np.dot(next_residual, next_residual))
                        expanded.append((score, prefix + (int(code),), next_residual))
                expanded.sort(key=lambda entry: (entry[0], entry[1]))
                beams = expanded[: self.collision_beam_width]

            for _, codes, _ in beams:
                if codes not in used_codes:
                    return np.asarray(codes, dtype=np.int64)

        raise ValueError(
            f"Unable to assign a unique GHOST SID in {context}; increase "
            "codebook_size/tail_extra_length or collision_beam_width."
        )

    def _deduplicate_codes(
        self,
        vectors: np.ndarray,
        codebooks: np.ndarray,
        codes: np.ndarray,
        *,
        valid_centers: int,
        context: str,
        group_keys: list[tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        repaired = codes.astype(np.int64, copy=True)
        used_by_group: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
        for row_idx in range(repaired.shape[0]):
            group = group_keys[row_idx] if group_keys is not None else tuple()
            used = used_by_group.setdefault(group, set())
            repaired[row_idx] = self._find_unique_codes(
                vectors[row_idx],
                codebooks,
                repaired[row_idx],
                used,
                valid_centers=valid_centers,
                context=f"{context} group={group}",
            )
            used.add(tuple(int(code) for code in repaired[row_idx]))
        return repaired

    @property
    def vocab_size(self) -> int:
        return self._item_vocab_size() + self.num_user_tokens

    @property
    def max_token_seq_len(self) -> int:
        return self.total_sid_length

    def log(self, message: str) -> None:
        print(message)

    def _save_json(self, payload: Any, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_tokens2item(self) -> None:
        payload = {str(tuple(tokens)): int(item_id) for tokens, item_id in self.tokens2item.items()}
        self._save_json(payload, self.tokens2item_save_path)

    @staticmethod
    def _load_json_if_exists(path: str) -> Any | None:
        file_path = Path(path)
        if not file_path.exists():
            return None
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_item2tokens(self, path: str) -> dict[int, list[int]]:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return {int(item_id): [int(token) for token in tokens] for item_id, tokens in loaded.items()}

    def _load_tokens2item(self, path: str) -> dict[tuple[int, ...], int]:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        tokens2item: dict[tuple[int, ...], int] = {}
        for key, value in loaded.items():
            try:
                tokens = ast.literal_eval(key)
            except Exception:
                cleaned = key.strip("()").replace(" ", "")
                tokens = tuple(int(part) for part in cleaned.split(",") if part)
            tokens2item[tuple(int(token) for token in tokens)] = int(value)
        return tokens2item

    def _load_array(self, path: str) -> np.ndarray | None:
        file_path = Path(path)
        if not file_path.exists():
            return None
        return np.load(file_path, allow_pickle=True)

    def _fit_projection(self, embeddings: np.ndarray) -> np.ndarray:
        if self.projection_method in {"identity", "none"}:
            projected = embeddings.astype(np.float32, copy=True)
            if projected.shape[1] < self.semantic_dim:
                padded = np.zeros((projected.shape[0], self.semantic_dim), dtype=np.float32)
                padded[:, : projected.shape[1]] = projected
                projected = padded
            elif projected.shape[1] > self.semantic_dim:
                projected = projected[:, : self.semantic_dim]
            self.projector_state = {"method": "identity", "input_dim": int(embeddings.shape[1])}
            return l2_normalize(projected)

        n_components = min(self.semantic_dim, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components, random_state=self.random_state, svd_solver="randomized")
        projected = pca.fit_transform(embeddings).astype(np.float32)
        if projected.shape[1] < self.semantic_dim:
            padded = np.zeros((projected.shape[0], self.semantic_dim), dtype=np.float32)
            padded[:, : projected.shape[1]] = projected
            projected = padded
        self.projector_state = {
            "method": "pca",
            "input_dim": int(embeddings.shape[1]),
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        }
        return l2_normalize(projected)

    def _fit_residual_codebooks(
        self,
        vectors: np.ndarray,
        num_layers: int,
        seed_offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if vectors.size == 0:
            empty_codebooks = np.zeros((num_layers, self.codebook_size, self.semantic_dim), dtype=np.float32)
            empty_tokens = np.zeros((0, num_layers), dtype=np.int64)
            return empty_codebooks, empty_tokens

        residuals = vectors.astype(np.float32, copy=True)
        codes = np.zeros((vectors.shape[0], num_layers), dtype=np.int64)
        codebooks = np.zeros((num_layers, self.codebook_size, vectors.shape[1]), dtype=np.float32)

        for layer_idx in range(num_layers):
            n_clusters = min(self.codebook_size, max(1, residuals.shape[0]))
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=self.kmeans_n_init,
                max_iter=self.kmeans_max_iter,
                random_state=self.random_state + seed_offset + layer_idx,
            )
            labels = kmeans.fit_predict(residuals)
            centers = kmeans.cluster_centers_.astype(np.float32)
            codebooks[layer_idx, :n_clusters] = centers
            codes[:, layer_idx] = labels
            residuals = residuals - centers[labels]
        return codebooks, codes

    def _encode_from_codebooks(self, vectors: np.ndarray, codebooks: np.ndarray, num_layers: int) -> np.ndarray:
        if vectors.size == 0:
            return np.zeros((0, num_layers), dtype=np.int64)
        residuals = vectors.astype(np.float32, copy=True)
        codes = np.zeros((vectors.shape[0], num_layers), dtype=np.int64)
        for layer_idx in range(num_layers):
            layer_centers = codebooks[layer_idx]
            distances = ((residuals[:, None, :] - layer_centers[None, :, :]) ** 2).sum(axis=-1)
            labels = distances.argmin(axis=1)
            codes[:, layer_idx] = labels
            residuals = residuals - layer_centers[labels]
        return codes

    def _build_head_tail_split(
        self,
        item_ids: list[int],
        item_popularity: dict[int, int],
    ) -> tuple[list[int], list[int]]:
        ranked = sorted(
            ((int(item_id), int(item_popularity.get(int(item_id), 0))) for item_id in item_ids),
            key=lambda pair: (-pair[1], pair[0]),
        )
        if not ranked:
            return [], []
        head_count = int(len(ranked) * self.head_ratio)
        head_count = max(1, min(head_count, len(ranked) - 1 if len(ranked) > 1 else 1))
        head_items = [item_id for item_id, _ in ranked[:head_count]]
        tail_items = [item_id for item_id, _ in ranked[head_count:]]
        if not tail_items and len(ranked) > 1:
            tail_items = [head_items.pop()]
        return head_items, tail_items

    def _build_undesired_collection(
        self,
        head_items: list[int],
        tail_items: list[int],
        projected_embeddings: np.ndarray,
        item_index: dict[int, int],
    ) -> dict[str, Any]:
        if not head_items or not tail_items:
            return {}

        head_matrix = np.asarray([projected_embeddings[item_index[item_id]] for item_id in head_items], dtype=np.float32)
        head_matrix = l2_normalize(head_matrix)
        tail_matrix = np.asarray([projected_embeddings[item_index[item_id]] for item_id in tail_items], dtype=np.float32)
        tail_matrix = l2_normalize(tail_matrix)
        similarity = tail_matrix @ head_matrix.T

        collection: dict[str, Any] = {}
        for tail_pos, tail_item_id in enumerate(tail_items):
            tail_sids = self.item2tokens[tail_item_id]
            candidate_scores = similarity[tail_pos]
            top_k = min(self.auo_ka, len(head_items))
            candidate_indices = np.argsort(-candidate_scores)[:top_k]
            scored_candidates = []
            for candidate_idx in candidate_indices:
                head_item_id = head_items[int(candidate_idx)]
                head_sids = self.item2tokens[head_item_id]
                lcp = longest_common_prefix_length(tail_sids, head_sids)
                scored_candidates.append(
                    {
                        "head_item_id": int(head_item_id),
                        "head_sids": [int(token) for token in head_sids],
                        "similarity": float(candidate_scores[int(candidate_idx)]),
                        "lcp": int(lcp),
                    }
                )
            scored_candidates.sort(key=lambda entry: (entry["lcp"], -entry["similarity"], entry["head_item_id"]))
            selected = scored_candidates[: min(self.auo_kb, len(scored_candidates))]
            collection[str(int(tail_item_id))] = {
                "head_item_ids": [entry["head_item_id"] for entry in selected],
                "head_sids": [entry["head_sids"] for entry in selected],
                "lcp": [entry["lcp"] for entry in selected],
                "similarity": [entry["similarity"] for entry in selected],
            }
        return collection

    def finalize_tokenization(
        self,
        item_embeddings_data: tuple[list[int], np.ndarray],
        item_popularity: dict[int, int],
        user_ids: list[int] | None = None,
    ) -> None:
        if Path(self.save_path).exists() and Path(self.metadata_path).exists() and Path(self.undesired_collection_path).exists():
            self.log(f"[GHOST] Loading existing tokenizer artifacts from {self.save_path}")
            self.item2tokens = self._load_item2tokens(self.save_path)
            self.tokens2item = self._load_tokens2item(self.tokens2item_save_path) if Path(self.tokens2item_save_path).exists() else {
                tuple(tokens): item_id for item_id, tokens in self.item2tokens.items()
            }
            metadata = self._load_json_if_exists(self.metadata_path) or {}
            self._restore_metadata(metadata)
            collection = self._load_json_if_exists(self.undesired_collection_path) or {}
            self.undesired_collection = collection
            self._validate_tokenization()
            self._validate_undesired_collection()
            return

        item_ids, embeddings = item_embeddings_data
        if len(item_ids) != int(embeddings.shape[0]):
            raise ValueError("Number of item IDs does not match the number of embeddings.")

        item_ids = [int(item_id) for item_id in item_ids]
        item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        projected_embeddings = self._fit_projection(np.asarray(embeddings, dtype=np.float32))
        self.semantic_embeddings = projected_embeddings

        head_items, tail_items = self._build_head_tail_split(item_ids, item_popularity)
        self.head_item_ids = head_items
        self.tail_item_ids = tail_items
        self.item_groups = {item_id: "head" for item_id in head_items}
        self.item_groups.update({item_id: "tail" for item_id in tail_items})

        head_vectors = np.asarray([projected_embeddings[item_index[item_id]] for item_id in head_items], dtype=np.float32)
        head_codebooks, head_codes = self._fit_residual_codebooks(
            head_vectors,
            self.skeleton_length,
            seed_offset=0,
        )
        head_codes = self._deduplicate_codes(
            head_vectors,
            head_codebooks,
            head_codes,
            valid_centers=min(self.codebook_size, len(head_items)),
            context="head",
        )
        self.head_codebooks = head_codebooks

        self.item2tokens = {}
        self.item_lengths = {}
        head_code_map: dict[int, np.ndarray] = {}
        for pos, item_id in enumerate(head_items):
            tokens = head_codes[pos].tolist()
            self.item2tokens[item_id] = self._codes_to_tokens(tokens)
            self.item_lengths[item_id] = self.skeleton_length
            head_code_map[item_id] = head_codes[pos]

        tail_vectors = np.asarray([projected_embeddings[item_index[item_id]] for item_id in tail_items], dtype=np.float32)
        if tail_items:
            tail_prefix_codes = []
            nearest_head_items = []
            head_matrix = l2_normalize(head_vectors)
            tail_matrix = l2_normalize(tail_vectors)
            similarities = tail_matrix @ head_matrix.T
            nearest_head_indices = similarities.argmax(axis=1)
            for tail_pos, nearest_idx in enumerate(nearest_head_indices):
                nearest_head_item = head_items[int(nearest_idx)]
                nearest_head_items.append(nearest_head_item)
                tail_prefix_codes.append(head_code_map[nearest_head_item])
                self.nearest_head[tail_items[tail_pos]] = int(nearest_head_item)

            prefix_residuals = []
            for tail_pos, tail_item_id in enumerate(tail_items):
                prefix_tokens = tail_prefix_codes[tail_pos]
                prefix_sum = np.zeros(self.semantic_dim, dtype=np.float32)
                for layer_idx, token_id in enumerate(prefix_tokens):
                    prefix_sum += head_codebooks[layer_idx, int(token_id)]
                prefix_residuals.append(projected_embeddings[item_index[tail_item_id]] - prefix_sum)
            tail_residuals = np.asarray(prefix_residuals, dtype=np.float32)
            tail_codebooks, tail_codes = self._fit_residual_codebooks(
                tail_residuals,
                self.tail_extra_length,
                seed_offset=1024,
            )
            tail_codes = self._deduplicate_codes(
                tail_residuals,
                tail_codebooks,
                tail_codes,
                valid_centers=min(self.codebook_size, len(tail_items)),
                context="tail",
                group_keys=[tuple(int(code) for code in prefix) for prefix in tail_prefix_codes],
            )
            self.tail_codebooks = tail_codebooks

            for pos, item_id in enumerate(tail_items):
                prefix_tokens = self._codes_to_tokens(tail_prefix_codes[pos])
                extra_tokens = self._codes_to_tokens(
                    tail_codes[pos],
                    start_layer=self.skeleton_length,
                )
                all_tokens = prefix_tokens + extra_tokens
                self.item2tokens[item_id] = all_tokens
                self.item_lengths[item_id] = self.total_sid_length
        else:
            self.tail_codebooks = np.zeros((self.tail_extra_length, self.codebook_size, self.semantic_dim), dtype=np.float32)

        self._validate_tokenization()
        self.tokens2item = {tuple(tokens): item_id for item_id, tokens in self.item2tokens.items()}
        self.undesired_collection = self._build_undesired_collection(head_items, tail_items, projected_embeddings, item_index)
        self._validate_undesired_collection()

        self._persist_artifacts(item_ids, projected_embeddings)

    def _persist_artifacts(self, item_ids: list[int], projected_embeddings: np.ndarray) -> None:
        self._save_json({str(item_id): tokens for item_id, tokens in self.item2tokens.items()}, self.save_path)
        self._save_tokens2item()
        self._save_json(self.undesired_collection, self.undesired_collection_path)
        self._save_json(self.head_item_ids, self.head_items_path)
        self._save_json(self.tail_item_ids, self.tail_items_path)
        self._save_json(self._build_metadata(), self.metadata_path)
        np.save(self.semantic_embedding_path, projected_embeddings.astype(np.float32))
        np.save(self.head_codebook_path, self.head_codebooks)
        np.save(self.tail_codebook_path, self.tail_codebooks)
        if self.projector_state.get("method") == "pca":
            np.save(self.projector_component_path, self.projector_state["components"])
            np.save(self.projector_mean_path, self.projector_state["mean"])

    def _build_metadata(self) -> dict[str, Any]:
        metadata = {
            "artifact_version": self.ARTIFACT_VERSION,
            "head_ratio": self.head_ratio,
            "skeleton_length": self.skeleton_length,
            "tail_extra_length": self.tail_extra_length,
            "semantic_dim": self.semantic_dim,
            "codebook_size": self.codebook_size,
            "auo_ka": self.auo_ka,
            "auo_kb": self.auo_kb,
            "projection_method": self.projection_method,
            "reserve_tokens": self.reserve_tokens,
            "n_codebooks": self.n_codebooks,
            "digits": self.digits,
            "num_user_tokens": self.num_user_tokens,
            "user_token_start_idx": self.user_token_start_idx,
            "vocab_size": self.vocab_size,
            "head_item_ids": self.head_item_ids,
            "tail_item_ids": self.tail_item_ids,
            "item_lengths": self.item_lengths,
            "nearest_head": self.nearest_head,
            "collision_repair": {
                "enabled": self.collision_repair_enabled,
                "candidates_per_layer": self.collision_candidates_per_layer,
                "beam_width": self.collision_beam_width,
                "strategy": "nearest unused residual path within fixed SID length",
            },
            "projector_state": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in self.projector_state.items()
                if key in {"method", "input_dim", "explained_variance_ratio"}
            },
        }
        return metadata

    def _restore_metadata(self, metadata: dict[str, Any]) -> None:
        artifact_version = int(metadata.get("artifact_version", 0))
        if artifact_version != self.ARTIFACT_VERSION and not bool(self.config.get("allow_legacy_artifacts", False)):
            raise ValueError(
                "Incompatible GHOST tokenizer artifact. Version 1 artifacts contain "
                "an invalid un-offset tail prefix and must be rebuilt; expected "
                f"artifact_version={self.ARTIFACT_VERSION}, got {artifact_version}."
            )
        self.head_ratio = float(metadata.get("head_ratio", self.head_ratio))
        self.skeleton_length = int(metadata.get("skeleton_length", self.skeleton_length))
        self.tail_extra_length = int(metadata.get("tail_extra_length", self.tail_extra_length))
        self.semantic_dim = int(metadata.get("semantic_dim", self.semantic_dim))
        self.codebook_size = int(metadata.get("codebook_size", self.codebook_size))
        self.auo_ka = int(metadata.get("auo_ka", self.auo_ka))
        self.auo_kb = int(metadata.get("auo_kb", self.auo_kb))
        self.reserve_tokens = int(metadata.get("reserve_tokens", self.reserve_tokens))
        # Recompute compatibility fields from the restored variable-length SID
        # configuration. Older GHOST artifacts do not contain these fields.
        self.n_codebooks = self.skeleton_length + self.tail_extra_length
        self.digits = self.n_codebooks
        self.num_user_tokens = int(metadata.get("num_user_tokens", 0))
        expected_user_start = self._item_vocab_size()
        stored_user_start = metadata.get("user_token_start_idx")
        self.user_token_start_idx = int(
            stored_user_start if stored_user_start is not None else expected_user_start
        )
        if self.num_user_tokens == 0:
            self.user_token_start_idx = expected_user_start
        self.head_item_ids = [int(item_id) for item_id in metadata.get("head_item_ids", [])]
        self.tail_item_ids = [int(item_id) for item_id in metadata.get("tail_item_ids", [])]
        self.item_lengths = {int(item_id): int(length) for item_id, length in metadata.get("item_lengths", {}).items()}
        self.nearest_head = {int(item_id): int(head_id) for item_id, head_id in metadata.get("nearest_head", {}).items()}
        self.projector_state = metadata.get("projector_state", {})

        if Path(self.semantic_embedding_path).exists():
            self.semantic_embeddings = np.load(self.semantic_embedding_path, allow_pickle=True)
        if Path(self.head_codebook_path).exists():
            self.head_codebooks = np.load(self.head_codebook_path, allow_pickle=True)
        if Path(self.tail_codebook_path).exists():
            self.tail_codebooks = np.load(self.tail_codebook_path, allow_pickle=True)

    def _validate_tokenization(self) -> None:
        head_items = set(self.head_item_ids)
        tail_items = set(self.tail_item_ids)
        if head_items & tail_items:
            raise ValueError("GHOST head and tail item partitions overlap.")
        if set(self.item2tokens) != head_items | tail_items:
            raise ValueError("GHOST item mapping does not match the head/tail partition.")

        seen: dict[tuple[int, ...], int] = {}
        for item_id, tokens in self.item2tokens.items():
            expected_length = self.total_sid_length if item_id in tail_items else self.skeleton_length
            if len(tokens) != expected_length:
                raise ValueError(
                    f"GHOST item {item_id} has SID length {len(tokens)}, expected {expected_length}."
                )
            for layer_idx, token in enumerate(tokens):
                lower = self.reserve_tokens + layer_idx * self.codebook_size
                upper = lower + self.codebook_size
                if not lower <= int(token) < upper:
                    raise ValueError(
                        f"GHOST item {item_id} token {token} at layer {layer_idx} "
                        f"is outside [{lower}, {upper})."
                    )
            key = tuple(int(token) for token in tokens)
            if key in seen:
                raise ValueError(f"Duplicate GHOST SID for items {seen[key]} and {item_id}: {key}.")
            seen[key] = int(item_id)

        for tail_item in tail_items:
            if tail_item not in self.nearest_head:
                raise ValueError(f"GHOST tail item {tail_item} has no nearest head item.")
            head_item = self.nearest_head[tail_item]
            if head_item not in head_items:
                raise ValueError(f"GHOST tail item {tail_item} references non-head item {head_item}.")
            if self.item2tokens[tail_item][: self.skeleton_length] != self.item2tokens[head_item]:
                raise ValueError(
                    f"GHOST tail item {tail_item} does not inherit nearest head {head_item}'s SID."
                )

    def _validate_undesired_collection(self) -> None:
        head_items = set(self.head_item_ids)
        expected_count = min(self.auo_kb, self.auo_ka, len(head_items))
        for tail_item in self.tail_item_ids:
            entry = self.undesired_collection.get(str(int(tail_item)))
            if entry is None:
                raise ValueError(f"GHOST tail item {tail_item} has no AUO collection entry.")
            candidate_ids = [int(item_id) for item_id in entry.get("head_item_ids", [])]
            candidate_sids = entry.get("head_sids", [])
            lcp_values = [int(value) for value in entry.get("lcp", [])]
            if len(candidate_ids) != expected_count:
                raise ValueError(
                    f"GHOST tail item {tail_item} has {len(candidate_ids)} AUO candidates, "
                    f"expected {expected_count}."
                )
            if not (len(candidate_sids) == len(candidate_ids) == len(lcp_values)):
                raise ValueError(f"GHOST tail item {tail_item} has inconsistent AUO metadata lengths.")
            if any(item_id not in head_items for item_id in candidate_ids):
                raise ValueError(f"GHOST tail item {tail_item} has a non-head AUO candidate.")
            actual_lcp = [
                longest_common_prefix_length(self.item2tokens[tail_item], sid)
                for sid in candidate_sids
            ]
            if actual_lcp != lcp_values:
                raise ValueError(f"GHOST tail item {tail_item} has stale AUO LCP metadata.")
            if lcp_values != sorted(lcp_values):
                raise ValueError(f"GHOST tail item {tail_item} AUO candidates are not LCP-ranked.")

    @classmethod
    def load(cls, config: dict):
        tokenizer = cls(config)
        if Path(tokenizer.save_path).exists():
            tokenizer.item2tokens = tokenizer._load_item2tokens(tokenizer.save_path)
            tokenizer.tokens2item = tokenizer._load_tokens2item(tokenizer.tokens2item_save_path) if Path(tokenizer.tokens2item_save_path).exists() else {
                tuple(tokens): item_id for item_id, tokens in tokenizer.item2tokens.items()
            }
        if Path(tokenizer.metadata_path).exists():
            metadata = tokenizer._load_json_if_exists(tokenizer.metadata_path) or {}
            tokenizer._restore_metadata(metadata)
        if Path(tokenizer.undesired_collection_path).exists():
            tokenizer.undesired_collection = tokenizer._load_json_if_exists(tokenizer.undesired_collection_path) or {}
        if tokenizer.item2tokens and tokenizer.head_item_ids and tokenizer.tail_item_ids:
            tokenizer._validate_tokenization()
            tokenizer._validate_undesired_collection()
        return tokenizer

    def get_item_tokens(self, item_id: int) -> list[int]:
        return list(self.item2tokens[int(item_id)])

    def get_item_length(self, item_id: int) -> int:
        return int(self.item_lengths.get(int(item_id), len(self.item2tokens.get(int(item_id), []))))

    def is_tail_item(self, item_id: int) -> bool:
        return int(item_id) in set(self.tail_item_ids)
