from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import torch

from genrec.trainers.generative.tiger_trainer import TigerTrainer


class CrabTigerTrainer(TigerTrainer):
    """TIGER/LETTER trainer with CRAB's tree-structured token regularizer."""

    def __init__(
        self,
        *args,
        gamma: float = 0.2,
        sid_length: int = 4,
        item2tokens: Optional[Dict] = None,
        **kwargs,
    ):
        self.crab_gamma = float(gamma)
        self.crab_sid_length = int(sid_length)
        self.crab_child_groups = self._build_child_groups(item2tokens or {})
        super().__init__(*args, item2tokens=item2tokens, **kwargs)

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
        if not self.crab_child_groups:
            return embedding.new_tensor(0.0)

        group_losses = []
        for children in self.crab_child_groups:
            child_ids = torch.as_tensor(children, dtype=torch.long, device=embedding.device)
            child_embeddings = embedding.index_select(0, child_ids)
            centroid = child_embeddings.mean(dim=0, keepdim=True)
            group_losses.append((child_embeddings - centroid).square().sum(dim=-1).mean())
        # Equation 9 sums the per-parent child variance over levels and parents.
        return torch.stack(group_losses).sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base_result = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        rec_loss, outputs = base_result
        tree_loss = self._tree_regularizer(model) if self.crab_gamma > 0.0 else rec_loss.new_tensor(0.0)
        loss = rec_loss + self.crab_gamma * tree_loss
        outputs.loss = loss
        outputs.loss_rec = rec_loss.detach()
        outputs.loss_tree = tree_loss.detach()
        return (loss, outputs) if return_outputs else loss
