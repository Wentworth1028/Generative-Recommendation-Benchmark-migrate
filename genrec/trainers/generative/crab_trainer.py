from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch

from genrec.trainers.generative.tiger_trainer import TigerTrainer


def hierarchical_semantic_alignment(
    embedding: torch.Tensor,
    child_groups: list[tuple[int, ...]],
) -> torch.Tensor:
    if not child_groups:
        return embedding.new_zeros(())

    child_ids, group_ids, group_counts = build_hsa_index(child_groups)
    return indexed_hierarchical_semantic_alignment(
        embedding,
        child_ids.to(embedding.device),
        group_ids.to(embedding.device),
        group_counts.to(embedding.device),
    )


def build_hsa_index(
    child_groups: list[tuple[int, ...]],
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    child_ids = []
    group_ids = []
    group_counts = []
    for group_id, children in enumerate(child_groups):
        child_ids.extend(int(child) for child in children)
        group_ids.extend([group_id] * len(children))
        group_counts.append(len(children))
    return (
        torch.tensor(child_ids, dtype=torch.long),
        torch.tensor(group_ids, dtype=torch.long),
        torch.tensor(group_counts, dtype=torch.long),
    )


def indexed_hierarchical_semantic_alignment(
    embedding: torch.Tensor,
    child_ids: torch.LongTensor,
    group_ids: torch.LongTensor,
    group_counts: torch.LongTensor,
) -> torch.Tensor:
    if child_ids.numel() == 0:
        return embedding.new_zeros(())

    child_embeddings = embedding.index_select(0, child_ids)
    num_groups = int(group_counts.numel())
    counts = group_counts.to(dtype=embedding.dtype)
    centroid_sums = embedding.new_zeros((num_groups, embedding.shape[1]))
    centroid_sums.index_add_(0, group_ids, child_embeddings)
    centroids = centroid_sums / counts.unsqueeze(1)
    centered = child_embeddings - centroids.index_select(0, group_ids)
    squared_distance = centered.square().sum(dim=-1)
    group_squared_distance = embedding.new_zeros((num_groups,))
    group_squared_distance.index_add_(0, group_ids, squared_distance)
    # CRAB Equation 9 averages within each sibling group and sums over parents
    # and hierarchy levels.
    return (group_squared_distance / counts).sum()


class CrabTigerTrainer(TigerTrainer):
    """TIGER/LETTER trainer with CRAB's tree-structured token regularizer."""

    def __init__(
        self,
        *args,
        gamma: float = 1e-5,
        sid_length: int = 4,
        dominance_warning_ratio: float = 10.0,
        item2tokens: Optional[Dict] = None,
        **kwargs,
    ):
        self.crab_gamma = float(gamma)
        self.crab_sid_length = int(sid_length)
        self.crab_dominance_warning_ratio = float(dominance_warning_ratio)
        self._crab_dominance_warned = False
        self._crab_dominance_checked = False
        self._last_crab_component_tensors: dict[str, torch.Tensor] = {}
        self.last_crab_components: dict[str, float] = {}
        self.crab_child_groups = self._build_child_groups(item2tokens or {})
        model_for_buffers = kwargs.get("model", args[0] if args else None)
        if model_for_buffers is not None:
            child_ids, group_ids, group_counts = build_hsa_index(self.crab_child_groups)
            model_for_buffers.register_buffer(
                "crab_hsa_child_ids", child_ids, persistent=False
            )
            model_for_buffers.register_buffer(
                "crab_hsa_group_ids", group_ids, persistent=False
            )
            model_for_buffers.register_buffer(
                "crab_hsa_group_counts", group_counts, persistent=False
            )
        super().__init__(*args, item2tokens=item2tokens, **kwargs)
        self._write_crab_loss_config()

    def _build_child_groups(self, item2tokens: Dict) -> list[tuple[int, ...]]:
        relations = [defaultdict(set) for _ in range(max(0, self.crab_sid_length - 1))]
        for tokens in item2tokens.values():
            row = [int(token) for token in tokens[: self.crab_sid_length]]
            for level in range(min(len(row) - 1, len(relations))):
                relations[level][row[level]].add(row[level + 1])
        return [
            tuple(sorted(children))
            for level_relations in relations
            for children in level_relations.values()
            if children
        ]

    def _tree_regularizer(self, model) -> torch.Tensor:
        unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        embedding = unwrapped.get_input_embeddings().weight
        return indexed_hierarchical_semantic_alignment(
            embedding,
            unwrapped.crab_hsa_child_ids,
            unwrapped.crab_hsa_group_ids,
            unwrapped.crab_hsa_group_counts,
        )

    def _write_crab_loss_config(self) -> None:
        if not self.is_world_process_zero():
            return
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "gamma": self.crab_gamma,
            "sid_length": self.crab_sid_length,
            "child_group_count": len(self.crab_child_groups),
            "hsa_reduction": "mean_within_parent_sum_across_parents_and_levels",
            "dominance_warning_ratio": self.crab_dominance_warning_ratio,
        }
        (output_dir / "crab_loss_config.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and not any(key.startswith("eval_") for key in logs):
            self.last_crab_components = {
                key: float(value)
                for key, value in self._last_crab_component_tensors.items()
            }
            logs = {**logs, **self.last_crab_components}
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base_result = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        rec_loss, outputs = base_result
        tree_loss = self._tree_regularizer(model) if self.crab_gamma > 0.0 else rec_loss.new_tensor(0.0)
        weighted_tree_loss = self.crab_gamma * tree_loss
        loss = rec_loss + weighted_tree_loss

        self._last_crab_component_tensors = {
            "crab_rec_loss": rec_loss.detach(),
            "crab_hsa_loss": tree_loss.detach(),
            "crab_weighted_hsa_loss": weighted_tree_loss.detach(),
        }
        if not self._crab_dominance_checked:
            rec_value = float(rec_loss.detach())
            tree_value = float(tree_loss.detach())
            weighted_value = float(weighted_tree_loss.detach())
            if (
                rec_value > 0.0
                and weighted_value > self.crab_dominance_warning_ratio * rec_value
            ):
                warnings.warn(
                    "CRAB weighted HSA loss dominates recommendation loss: "
                    f"rec={rec_value:.6g}, hsa={tree_value:.6g}, "
                    f"gamma*hsa={weighted_value:.6g}. Retune gamma for this backbone.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._crab_dominance_warned = True
            self._crab_dominance_checked = True

        outputs.loss = loss
        outputs.loss_rec = rec_loss.detach()
        outputs.loss_tree = tree_loss.detach()
        outputs.loss_tree_weighted = weighted_tree_loss.detach()
        return (loss, outputs) if return_outputs else loss
