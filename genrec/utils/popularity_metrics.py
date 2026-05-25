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


def compute_prediction_popularity_metrics(
    predictions: Sequence[Sequence[int]],
    item_popularity: Dict[int, int],
    k_list: Iterable[int] = (1, 5, 10),
    quantiles: Iterable[float] = (0.1,),
) -> Dict[str, float]:
    del quantiles
    metrics = {}
    pop_items = get_pop_items_by_interaction_share(item_popularity, target_share=0.5)
    for k in k_list:
        total_predictions = 0
        pop_predictions = 0
        for predicted_items in predictions:
            top_k_items = [item for item in predicted_items[:k] if item is not None]
            total_predictions += len(top_k_items)
            pop_predictions += sum(1 for item in top_k_items if item in pop_items)

        pop_share = pop_predictions / total_predictions if total_predictions > 0 else 0.0
        metrics[f"pop_item_share@{k}"] = pop_share
        metrics[f"unpop_item_share@{k}"] = 1.0 - pop_share if total_predictions > 0 else 0.0
    return metrics
