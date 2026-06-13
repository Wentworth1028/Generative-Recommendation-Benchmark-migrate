import pickle
from collections import Counter
from typing import Dict, Iterable, List, Sequence


def compute_train_item_popularity(data_interaction_files: str, shift_item_id: int = 0) -> Dict[int, int]:
    with open(data_interaction_files, "rb") as f:
        user2item_data = pickle.load(f)

    counter = Counter()
    for _, row in user2item_data.iterrows():
        item_seq = row["ItemID"]
        train_seq = item_seq[:-2]
        counter.update(int(item_id) + shift_item_id for item_id in train_seq)
    return dict(counter)


def compute_dataset_item_popularity(data_interaction_files: str, shift_item_id: int = 0) -> Dict[int, int]:
    with open(data_interaction_files, "rb") as f:
        user2item_data = pickle.load(f)

    counter = Counter()
    for _, row in user2item_data.iterrows():
        item_seq = row["ItemID"]
        counter.update(int(item_id) + shift_item_id for item_id in item_seq)
    return dict(counter)


def get_top_popular_items(item_popularity: Dict[int, int], quantile: float) -> set:
    if not item_popularity:
        return set()

    sorted_items = sorted(item_popularity.items(), key=lambda x: (-x[1], x[0]))
    top_k = max(1, int(len(sorted_items) * quantile))
    return {item_id for item_id, _ in sorted_items[:top_k]}


def get_pop_items_by_interaction_share(item_popularity: Dict[int, int], target_share: float = 0.5) -> set:
    if not item_popularity:
        return set()

    sorted_items = sorted(item_popularity.items(), key=lambda x: (-x[1], x[0]))
    total_interactions = sum(count for _, count in sorted_items)
    if total_interactions <= 0:
        return set()

    cutoff = total_interactions * target_share
    cumulative = 0
    pop_items = set()
    for item_id, count in sorted_items:
        pop_items.add(item_id)
        cumulative += count
        if cumulative >= cutoff:
            break
    return pop_items


def _gini(values: Sequence[float]) -> float:
    clean_values = sorted(float(value) for value in values if value >= 0)
    n = len(clean_values)
    if n == 0:
        return 0.0
    total = sum(clean_values)
    if total <= 0:
        return 0.0
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(clean_values))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _popularity_groups_by_item_count(item_popularity: Dict[int, int], num_groups: int = 5) -> List[set]:
    if not item_popularity:
        return []
    sorted_items = [item_id for item_id, _ in sorted(item_popularity.items(), key=lambda x: (-x[1], x[0]))]
    groups = []
    n_items = len(sorted_items)
    for group_idx in range(num_groups):
        start = round(group_idx * n_items / num_groups)
        end = round((group_idx + 1) * n_items / num_groups)
        groups.append(set(sorted_items[start:end]))
    return groups


def compute_prediction_popularity_metrics(
    predictions: Sequence[Sequence[int]],
    item_popularity: Dict[int, int],
    k_list: Iterable[int] = (1, 5, 10),
    quantiles: Iterable[float] = (0.1, 0.2),
) -> Dict[str, float]:
    metrics = {}
    pop_items = get_pop_items_by_interaction_share(item_popularity, target_share=0.5)
    quantile_pop_items = {
        quantile: get_top_popular_items(item_popularity, quantile)
        for quantile in quantiles
    }
    popularity_groups = _popularity_groups_by_item_count(item_popularity, num_groups=5)
    all_items = set(item_popularity)

    for k in k_list:
        total_predictions = 0
        pop_predictions = 0
        arp_sum = 0.0
        exposure_counter = Counter()
        quantile_hits = {quantile: 0 for quantile in quantile_pop_items}

        for predicted_items in predictions:
            top_k_items = [item for item in predicted_items[:k] if item is not None]
            total_predictions += len(top_k_items)
            pop_predictions += sum(1 for item in top_k_items if item in pop_items)
            arp_sum += sum(item_popularity.get(item, 0) for item in top_k_items)
            exposure_counter.update(top_k_items)
            all_items.update(top_k_items)
            for quantile, items in quantile_pop_items.items():
                quantile_hits[quantile] += sum(1 for item in top_k_items if item in items)

        pop_share = pop_predictions / total_predictions if total_predictions > 0 else 0.0
        metrics[f"pop_item_share@{k}"] = pop_share
        metrics[f"unpop_item_share@{k}"] = 1.0 - pop_share if total_predictions > 0 else 0.0
        metrics[f"arp@{k}"] = arp_sum / total_predictions if total_predictions > 0 else 0.0

        for quantile, hit_count in quantile_hits.items():
            quantile_pct = int(round(quantile * 100))
            share = hit_count / total_predictions if total_predictions > 0 else 0.0
            metrics[f"pop{quantile_pct}_item_share@{k}"] = share
            metrics[f"unpop{quantile_pct}_item_share@{k}"] = (
                1.0 - share if total_predictions > 0 else 0.0
            )

        exposure_values = [exposure_counter.get(item, 0) for item in sorted(all_items)]
        metrics[f"gini@{k}"] = _gini(exposure_values)

        group_utilities = []
        num_users = len(predictions)
        for group in popularity_groups:
            if not group or num_users <= 0:
                continue
            group_exposure = sum(exposure_counter.get(item, 0) for item in group)
            group_utilities.append(group_exposure / (len(group) * num_users))
        metrics[f"mgu@{k}"] = min(group_utilities) if group_utilities else 0.0
    return metrics


def compute_token_popularity_metrics(
    item2tokens: Dict[int, Sequence[int]],
    item_popularity: Dict[int, int],
    codebook_size: int = 256,
    reserve_tokens: int = 100,
) -> Dict[str, float]:
    if not item2tokens:
        return {}

    num_layers = len(next(iter(item2tokens.values())))
    metrics = {}
    for layer in range(num_layers):
        masses = [0.0 for _ in range(codebook_size)]
        item_counts = [0 for _ in range(codebook_size)]
        extra_mass = Counter()
        for item_id, tokens in item2tokens.items():
            if layer >= len(tokens):
                continue
            token_id = int(tokens[layer])
            sem_id = token_id - reserve_tokens - codebook_size * layer
            mass = float(item_popularity.get(int(item_id), 0))
            if 0 <= sem_id < codebook_size:
                masses[sem_id] += mass
                item_counts[sem_id] += 1
            else:
                extra_mass[token_id] += mass

        values = masses + [float(value) for value in extra_mass.values()]
        total_mass = sum(values)
        top10_n = max(1, int(len(values) * 0.1))
        top20_n = max(1, int(len(values) * 0.2))
        entropy = 0.0
        if total_mass > 0:
            for value in values:
                if value > 0:
                    probability = value / total_mass
                    entropy -= probability * __import__("math").log(probability)
            max_entropy = __import__("math").log(len(values)) if len(values) > 1 else 1.0
            entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            entropy_norm = 0.0

        prefix = f"token_layer{layer}"
        metrics[f"{prefix}_popularity_gini"] = _gini(values)
        metrics[f"{prefix}_popularity_entropy_norm"] = entropy_norm
        metrics[f"{prefix}_top10_popularity_share"] = (
            sum(sorted(values, reverse=True)[:top10_n]) / total_mass if total_mass > 0 else 0.0
        )
        metrics[f"{prefix}_top20_popularity_share"] = (
            sum(sorted(values, reverse=True)[:top20_n]) / total_mass if total_mass > 0 else 0.0
        )
        metrics[f"{prefix}_used_token_count"] = sum(1 for value in values if value > 0)
        metrics[f"{prefix}_item_count_gini"] = _gini(item_counts)
    return metrics
