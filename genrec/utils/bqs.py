"""Balanced Quality Score utilities.

The implementation follows Coppolillo et al. (2024) and the authors' public
reference code. Popularity classes are derived from training interactions;
BQS itself compares one candidate run with an explicit baseline run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import savgol_filter


DEFAULT_BQS_PENALTIES = {"hit": 10.0, "ndcg": 2.0}
DEFAULT_BQS_TEMPERATURE = 0.1


@dataclass(frozen=True)
class PopularityClasses:
    low: frozenset[int]
    medium: frozenset[int]
    high: frozenset[int]
    low_threshold: int
    high_threshold: int
    inflection_index: int
    low_elbow_index: int
    high_elbow_index: int
    smoothing_window: int

    @property
    def item_count(self) -> int:
        return len(self.low) + len(self.medium) + len(self.high)

    def to_context(self) -> dict:
        low_ids = ",".join(str(item_id) for item_id in sorted(self.low))
        return {
            "grouping_method": "paper_savgol_rotor",
            "popularity_source": "train_interactions_leave_last_two_out",
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "catalog_item_count": self.item_count,
            "low_item_count": len(self.low),
            "medium_item_count": len(self.medium),
            "high_item_count": len(self.high),
            "low_item_fraction": len(self.low) / self.item_count if self.item_count else 0.0,
            "inflection_index": self.inflection_index,
            "low_elbow_index": self.low_elbow_index,
            "high_elbow_index": self.high_elbow_index,
            "smoothing_window": self.smoothing_window,
            "low_item_ids_sha256": hashlib.sha256(low_ids.encode("ascii")).hexdigest(),
        }


def _valid_savgol_window(item_count: int) -> int:
    if item_count < 3:
        return item_count
    window = max(5, item_count // 5)
    return max(3, min(window, item_count))


def _rotor_elbow_index(points: np.ndarray) -> int:
    """Return the elbow using the algorithm from kneebow.rotor.Rotor."""
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("Rotor expects a non-empty [N, 2] array.")
    if len(points) <= 2:
        return 0

    mins = points.min(axis=0)
    spans = points.max(axis=0) - mins
    spans[spans == 0.0] = 1.0
    scaled = (points - mins) / spans
    theta = math.atan2(
        scaled[-1, 1] - scaled[0, 1],
        scaled[-1, 0] - scaled[0, 0],
    )
    cosine = math.cos(theta)
    sine = math.sin(theta)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    rotated = scaled.dot(rotation)
    return int(np.argmin(rotated[:, 1]))


def build_popularity_classes(item_popularity: Mapping[int, int]) -> PopularityClasses:
    """Build paper-style low, medium, and high popularity item classes.

    The official notebook smooths the ascending popularity curve, finds the
    minimum absolute second derivative, and applies Rotor to either side. This
    version preserves that behavior while making the window and segment bounds
    valid for small or even-sized catalogs.
    """
    if not item_popularity:
        raise ValueError("Cannot build BQS popularity classes from an empty catalog.")
    if any(int(count) < 0 for count in item_popularity.values()):
        raise ValueError("Item popularity counts must be non-negative.")

    sorted_pairs = sorted(
        ((int(item_id), int(count)) for item_id, count in item_popularity.items()),
        key=lambda pair: (pair[1], pair[0]),
    )
    popularity = np.asarray([count for _, count in sorted_pairs], dtype=np.float64)
    item_count = len(popularity)

    if item_count < 5:
        low_index = 0
        high_index = max(0, item_count - 2)
        inflection_index = max(1, item_count // 2)
        window = item_count
    else:
        window = _valid_savgol_window(item_count)
        smoothed = savgol_filter(popularity, window_length=window, polyorder=2)
        second_derivative = np.diff(np.diff(smoothed))
        raw_inflection = int(np.argmin(np.abs(second_derivative)))

        # Both Rotor calls need a meaningful segment. The notebook does not
        # guard this edge case because its benchmark catalogs are large.
        inflection_index = min(max(raw_inflection, 2), item_count - 2)
        x = np.arange(item_count, dtype=np.float64)
        left_points = np.column_stack((x[:inflection_index], -smoothed[:inflection_index]))
        right_points = np.column_stack((x[inflection_index:], smoothed[inflection_index:]))
        low_index = _rotor_elbow_index(left_points)
        high_index = inflection_index + _rotor_elbow_index(right_points)

    low_threshold = int(popularity[low_index])
    high_threshold = int(popularity[high_index])
    if high_threshold < low_threshold:
        low_threshold, high_threshold = high_threshold, low_threshold

    low = frozenset(item_id for item_id, count in sorted_pairs if count <= low_threshold)
    medium = frozenset(
        item_id for item_id, count in sorted_pairs if low_threshold < count <= high_threshold
    )
    high = frozenset(item_id for item_id, count in sorted_pairs if count > high_threshold)
    if (low | medium | high) != set(item_popularity) or (low & medium) or (low & high) or (medium & high):
        raise AssertionError("BQS popularity classes must be disjoint and cover the catalog.")

    return PopularityClasses(
        low=low,
        medium=medium,
        high=high,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        inflection_index=inflection_index,
        low_elbow_index=low_index,
        high_elbow_index=high_index,
        smoothing_window=window,
    )


def compute_low_popularity_quality_metrics(
    predictions: Sequence[Sequence[int]],
    labels: Sequence[int],
    low_items: set[int] | frozenset[int],
    k_list: Iterable[int] = (1, 5, 10),
) -> tuple[Dict[str, float], dict]:
    """Compute HR_L and NDCG_L over samples whose test target is low-popular."""
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions and labels have different lengths: {len(predictions)} != {len(labels)}."
        )

    low_samples = [
        (int(label), predicted_items)
        for label, predicted_items in zip(labels, predictions)
        if int(label) in low_items
    ]
    if not low_samples:
        raise ValueError("The test split contains no low-popularity targets; BQS is undefined.")

    metrics: Dict[str, float] = {}
    for k in k_list:
        if int(k) <= 0:
            raise ValueError(f"BQS cutoff must be positive, got {k}.")
        hits = 0.0
        ndcg = 0.0
        for label, predicted_items in low_samples:
            top_k = list(predicted_items[: int(k)])
            if label not in top_k:
                continue
            hits += 1.0
            rank = top_k.index(label) + 1
            ndcg += 1.0 / math.log2(rank + 1)
        metrics[f"low_hit@{int(k)}"] = hits / len(low_samples)
        metrics[f"low_ndcg@{int(k)}"] = ndcg / len(low_samples)

    context = {
        "test_target_count": len(labels),
        "low_test_target_count": len(low_samples),
        "low_test_target_fraction": len(low_samples) / len(labels) if labels else 0.0,
    }
    return metrics, context


def bqs_gain_loss(delta: float, penalty: float) -> float:
    if penalty <= 1.0:
        raise ValueError(f"BQS penalty must be greater than 1, got {penalty}.")
    delta = float(delta)
    return delta if delta >= 0.0 else -((penalty * delta) ** 2) + delta


def compute_bqs(
    candidate_quality: float,
    baseline_quality: float,
    candidate_low_quality: float,
    baseline_low_quality: float,
    penalty: float,
    temperature: float = DEFAULT_BQS_TEMPERATURE,
) -> dict:
    if temperature <= 0.0:
        raise ValueError(f"BQS temperature must be positive, got {temperature}.")
    global_delta = float(candidate_quality) - float(baseline_quality)
    low_delta = float(candidate_low_quality) - float(baseline_low_quality)
    transformed_global = bqs_gain_loss(global_delta, penalty)
    transformed_low = bqs_gain_loss(low_delta, penalty)
    score_input = transformed_global + transformed_low
    scaled_score_input = score_input / float(temperature)
    if score_input >= 0.0:
        unscaled_score = 1.0 / (1.0 + math.exp(-score_input))
    else:
        exp_input = math.exp(score_input)
        unscaled_score = exp_input / (1.0 + exp_input)
    if scaled_score_input >= 0.0:
        score = 1.0 / (1.0 + math.exp(-scaled_score_input))
    else:
        exp_input = math.exp(scaled_score_input)
        score = exp_input / (1.0 + exp_input)
    return {
        "score": score,
        "global_delta": global_delta,
        "low_delta": low_delta,
        "transformed_global_delta": transformed_global,
        "transformed_low_delta": transformed_low,
        "sigmoid_input": score_input,
        "unscaled_score": unscaled_score,
        "scaled_sigmoid_input": scaled_score_input,
        "temperature": float(temperature),
        "penalty": float(penalty),
    }


def _metric_value(final_metrics: Mapping, key: str) -> float:
    metrics = final_metrics.get("metrics", final_metrics)
    if key not in metrics:
        raise KeyError(f"Required BQS metric is missing: {key}")
    return float(metrics[key])


def validate_bqs_context(candidate: Mapping, baseline: Mapping) -> None:
    candidate_context = candidate.get("bqs_context")
    baseline_context = baseline.get("bqs_context")
    if not candidate_context or not baseline_context:
        raise ValueError("Both candidate and baseline must contain bqs_context metadata.")

    comparable_fields = (
        "grouping_method",
        "popularity_source",
        "catalog_item_count",
        "low_item_count",
        "low_threshold",
        "low_item_ids_sha256",
        "test_target_count",
        "low_test_target_count",
    )
    mismatches = [
        field
        for field in comparable_fields
        if candidate_context.get(field) != baseline_context.get(field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}={candidate_context.get(field)!r}/{baseline_context.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"Candidate and baseline BQS contexts differ: {details}")


def compute_bqs_metrics(
    candidate: Mapping,
    baseline: Mapping,
    k_list: Iterable[int] = (1, 5, 10),
    penalties: Mapping[str, float] = DEFAULT_BQS_PENALTIES,
    temperature: float = DEFAULT_BQS_TEMPERATURE,
    validate_context: bool = True,
) -> tuple[Dict[str, float], dict]:
    """Compute HR-based and NDCG-based BQS values from final-metrics records."""
    if candidate.get("dataset") != baseline.get("dataset"):
        raise ValueError(
            f"BQS dataset mismatch: {candidate.get('dataset')!r} != {baseline.get('dataset')!r}."
        )
    if validate_context:
        validate_bqs_context(candidate, baseline)

    metrics: Dict[str, float] = {}
    details = {}
    for quality_name in ("hit", "ndcg"):
        penalty = float(penalties[quality_name])
        for k in k_list:
            result = compute_bqs(
                candidate_quality=_metric_value(candidate, f"test_{quality_name}@{int(k)}"),
                baseline_quality=_metric_value(baseline, f"test_{quality_name}@{int(k)}"),
                candidate_low_quality=_metric_value(candidate, f"test_low_{quality_name}@{int(k)}"),
                baseline_low_quality=_metric_value(baseline, f"test_low_{quality_name}@{int(k)}"),
                penalty=penalty,
                temperature=temperature,
            )
            metric_name = f"bqs_{quality_name}@{int(k)}"
            metrics[metric_name] = result["score"]
            details[metric_name] = result
    return metrics, details


def load_final_metrics(path: str | Path) -> dict:
    metrics_path = Path(path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "final_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
