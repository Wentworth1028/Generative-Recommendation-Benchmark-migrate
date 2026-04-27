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


def get_top_popular_items(item_popularity: Dict[int, int], quantile: float) -> set:
    if not item_popularity:
        return set()

    sorted_items = sorted(item_popularity.items(), key=lambda x: (-x[1], x[0]))
    top_k = max(1, int(len(sorted_items) * quantile))
    return {item_id for item_id, _ in sorted_items[:top_k]}


def compute_prediction_popularity_metrics(
    predictions: Sequence[Sequence[int]],
    item_popularity: Dict[int, int],
    k_list: Iterable[int] = (1, 5, 10),
    quantiles: Iterable[float] = (0.1,),
) -> Dict[str, float]:
    metrics = {}
    for quantile in quantiles:
        popular_items = get_top_popular_items(item_popularity, quantile)
        total_train_popularity = sum(item_popularity.values())
        popular_train_popularity = sum(item_popularity.get(item_id, 0) for item_id in popular_items)
        train_popularity_share = (
            popular_train_popularity / total_train_popularity if total_train_popularity > 0 else 0.0
        )
        for k in k_list:
            total_predictions = 0
            popular_predictions = 0
            for predicted_items in predictions:
                top_k_items = [item for item in predicted_items[:k] if item is not None]
                total_predictions += len(top_k_items)
                popular_predictions += sum(1 for item in top_k_items if item in popular_items)

            key = f"popularity@{k}-{quantile:g}"
            prediction_share = popular_predictions / total_predictions if total_predictions > 0 else 0.0
            metrics[key] = prediction_share
            metrics[f"{key}_train_share"] = train_popularity_share
            metrics[f"{key}_amplification"] = (
                prediction_share / train_popularity_share if train_popularity_share > 0 else 0.0
            )
    return metrics
