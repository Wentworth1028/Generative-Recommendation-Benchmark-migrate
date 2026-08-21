from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable
import json
import math
import pickle
import re

import numpy as np
import pandas as pd
import torch


def load_interaction_frame(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        frame = pickle.load(f)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame in {path}, got {type(frame)}")
    return frame


def load_item_title_map(path: str | Path, title_field: str = "Title") -> dict[int, str]:
    with open(path, "rb") as f:
        frame = pickle.load(f)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame in {path}, got {type(frame)}")
    if "ItemID" not in frame.columns:
        raise ValueError(f"Item text file must contain ItemID column: {path}")
    if title_field not in frame.columns:
        raise ValueError(f"Item text file must contain {title_field} column: {path}")
    title_map: dict[int, str] = {}
    for _, row in frame.iterrows():
        item_id = int(row["ItemID"])
        title = row.get(title_field, "")
        title_map[item_id] = "" if pd.isna(title) else str(title)
    return title_map


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sort_interaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort per-user interaction histories by timestamp when available."""
    if "Timestamp" not in df.columns:
        return df.copy().reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        items = list(row["ItemID"])
        timestamps = list(row["Timestamp"])
        paired = sorted(zip(timestamps, items), key=lambda pair: pair[0])
        if paired:
            timestamps_sorted, items_sorted = zip(*paired)
        else:
            timestamps_sorted, items_sorted = (), ()
        rows.append(
            {
                "UserID": int(row["UserID"]),
                "ItemID": [int(item) for item in items_sorted],
                "Timestamp": [int(ts) for ts in timestamps_sorted],
            }
        )
    return pd.DataFrame(rows)


def iteratively_filter_interactions(
    df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """Iteratively remove users/items below the minimum interaction threshold."""
    work_df = sort_interaction_frame(df)
    if work_df.empty:
        return work_df

    changed = True
    while changed:
        changed = False

        user_lengths = work_df["ItemID"].apply(len)
        keep_user_mask = user_lengths >= min_user_interactions
        if keep_user_mask.sum() != len(work_df):
            work_df = work_df.loc[keep_user_mask].reset_index(drop=True)
            changed = True

        if work_df.empty:
            break

        item_counter: Counter[int] = Counter()
        for items in work_df["ItemID"]:
            item_counter.update(int(item) for item in items)

        keep_items = {item_id for item_id, count in item_counter.items() if count >= min_item_interactions}
        if len(keep_items) != len(item_counter):
            changed = True

        if changed:
            filtered_rows = []
            for _, row in work_df.iterrows():
                items = []
                timestamps = []
                for item, ts in zip(row["ItemID"], row["Timestamp"]):
                    if int(item) in keep_items:
                        items.append(int(item))
                        timestamps.append(int(ts))
                if len(items) >= min_user_interactions:
                    filtered_rows.append({"UserID": int(row["UserID"]), "ItemID": items, "Timestamp": timestamps})
            work_df = pd.DataFrame(filtered_rows)

    if work_df.empty:
        return work_df
    return sort_interaction_frame(work_df).reset_index(drop=True)


def load_filtered_user_sequences(
    path: str | Path,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> dict[int, list[int]]:
    frame = load_interaction_frame(path)
    filtered_frame = iteratively_filter_interactions(
        frame,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    if filtered_frame.empty:
        return {}
    user_sequences: dict[int, list[int]] = {}
    for _, row in filtered_frame.iterrows():
        user_sequences[int(row["UserID"])] = [int(item) for item in row["ItemID"]]
    return user_sequences


def build_leave_one_out_samples(
    user_sequences: dict[int, list[int]],
    mode: str,
    max_history_items: int = 20,
) -> list[dict[str, Any]]:
    """Build prefix samples that match the current TIGER/LETTER leave-one-out flow."""
    mode = mode.lower()
    if mode not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported mode: {mode}")

    samples: list[dict[str, Any]] = []
    sample_index = 0
    for user_id in sorted(user_sequences):
        sequence = [int(item) for item in user_sequences[user_id]]
        if mode == "train":
            train_sequence = sequence[-(max_history_items + 2) : -2]
            if len(train_sequence) < 2:
                continue
            for idx in range(1, len(train_sequence)):
                history = train_sequence[:idx]
                target = train_sequence[idx]
                samples.append(
                    {
                        "index": sample_index,
                        "user_id": int(user_id),
                        "history_items": [int(item) for item in history],
                        "target_item": int(target),
                    }
                )
                sample_index += 1
        elif mode == "valid":
            if len(sequence) < 3:
                continue
            history = sequence[:-2]
            target = sequence[-2]
            if len(history) > max_history_items:
                history = history[-max_history_items:]
            samples.append(
                {
                    "index": sample_index,
                    "user_id": int(user_id),
                    "history_items": [int(item) for item in history],
                    "target_item": int(target),
                }
            )
            sample_index += 1
        else:
            if len(sequence) < 2:
                continue
            history = sequence[:-1]
            target = sequence[-1]
            if len(history) > max_history_items:
                history = history[-max_history_items:]
            samples.append(
                {
                    "index": sample_index,
                    "user_id": int(user_id),
                    "history_items": [int(item) for item in history],
                    "target_item": int(target),
                }
            )
            sample_index += 1
    return samples


def load_dense_embedding_matrix(
    embedding_path: str | Path,
    append_sentinel: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load a dense embedding matrix and optionally append a zero sentinel row."""
    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file does not exist: {path}")

    suffix = "".join(path.suffixes).lower()
    raw: Any
    if suffix.endswith(".npy"):
        raw = np.load(path, allow_pickle=True)
    elif suffix.endswith(".npz"):
        loaded = np.load(path, allow_pickle=True)
        if len(loaded.files) == 1:
            raw = loaded[loaded.files[0]]
        else:
            raw = {key: loaded[key] for key in loaded.files}
    elif suffix.endswith(".pt") or suffix.endswith(".pth"):
        raw = torch.load(path, map_location="cpu")
    elif suffix.endswith(".pkl") or suffix.endswith(".pickle"):
        with open(path, "rb") as f:
            raw = pickle.load(f)
    else:
        raise ValueError(f"Unsupported embedding file format: {path}")

    if isinstance(raw, dict):
        keys = sorted(int(key) for key in raw.keys())
        if not keys:
            raise ValueError(f"Embedding dictionary is empty: {path}")
        first_value = raw[keys[0]]
        first_array = np.asarray(first_value, dtype=np.float32)
        matrix = np.zeros((keys[-1] + 1, first_array.shape[-1]), dtype=np.float32)
        for key in keys:
            matrix[int(key)] = np.asarray(raw[int(key)], dtype=np.float32)
    elif isinstance(raw, pd.DataFrame):
        if "ItemID" not in raw.columns:
            raise ValueError("DataFrame embedding file must contain ItemID column.")
        vector_columns = [column for column in raw.columns if column != "ItemID"]
        if len(vector_columns) == 1 and raw[vector_columns[0]].map(lambda value: isinstance(value, (list, tuple, np.ndarray))).all():
            first_dim = len(np.asarray(raw[vector_columns[0]].iloc[0], dtype=np.float32))
            matrix = np.zeros((int(raw["ItemID"].max()) + 1, first_dim), dtype=np.float32)
            for _, row in raw.iterrows():
                matrix[int(row["ItemID"])] = np.asarray(row[vector_columns[0]], dtype=np.float32)
        else:
            matrix = np.zeros((int(raw["ItemID"].max()) + 1, len(vector_columns)), dtype=np.float32)
            for _, row in raw.iterrows():
                matrix[int(row["ItemID"])] = row[vector_columns].to_numpy(dtype=np.float32)
    else:
        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Embedding matrix must be 2D, got shape {matrix.shape}")

    if append_sentinel:
        if matrix.size == 0:
            raise ValueError(f"Embedding matrix is empty: {path}")
        if not np.allclose(matrix[-1], 0.0):
            matrix = np.concatenate([matrix, np.zeros((1, matrix.shape[1]), dtype=matrix.dtype)], axis=0)

    tensor = torch.as_tensor(matrix, dtype=torch.float32)
    sentinel_index = tensor.size(0) - 1
    return tensor, sentinel_index


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(eps)
    return summed / denom
