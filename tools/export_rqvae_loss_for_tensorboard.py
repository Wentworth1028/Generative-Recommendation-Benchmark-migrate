#!/usr/bin/env python3
"""Export RQ-VAE loss histories for TensorBoard-style inspection.

The trainer writes tokenizer_model/rqvae_loss_history.csv.  This script
collects selected runs, summarizes whether the popularity regularizer changes,
and optionally emits TensorBoard event files when tensorboard is installed.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path


LOSS_COLUMNS = [
    "total_loss",
    "rq_loss_without_pop",
    "recon_loss",
    "commit_loss",
    "popularity_balance_loss",
    "weighted_popularity_balance_loss",
    "effective_popularity_balance_weight",
]


def maybe_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        return None
    return SummaryWriter


def resolve_path(path_text: str, repo: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if path.is_absolute():
        candidate = repo / path.relative_to(path.anchor)
        if candidate.exists():
            return candidate
        parts = path.parts
        if "output" in parts:
            output_index = parts.index("output")
            candidate = repo.joinpath(*parts[output_index:])
            if candidate.exists():
                return candidate
    candidate = repo / path_text
    return candidate


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def summarize_loss(rows: list[dict[str, str]]) -> dict[str, float]:
    first = rows[0]
    last = rows[-1]
    tail = rows[-min(100, len(rows)) :]
    pop_values = [to_float(row["popularity_balance_loss"]) for row in rows]
    weighted_values = [to_float(row["weighted_popularity_balance_loss"]) for row in rows]

    return {
        "epochs_logged": float(len(rows)),
        "pop_balance_first": to_float(first["popularity_balance_loss"]),
        "pop_balance_last": to_float(last["popularity_balance_loss"]),
        "pop_balance_min": min(pop_values),
        "pop_balance_delta_last_minus_first": to_float(last["popularity_balance_loss"])
        - to_float(first["popularity_balance_loss"]),
        "weighted_pop_first": to_float(first["weighted_popularity_balance_loss"]),
        "weighted_pop_last": to_float(last["weighted_popularity_balance_loss"]),
        "weighted_pop_min": min(weighted_values),
        "rq_no_pop_first": to_float(first["rq_loss_without_pop"]),
        "rq_no_pop_last": to_float(last["rq_loss_without_pop"]),
        "recon_first": to_float(first["recon_loss"]),
        "recon_last": to_float(last["recon_loss"]),
        "commit_first": to_float(first["commit_loss"]),
        "commit_last": to_float(last["commit_loss"]),
        "tail100_pop_balance_mean": sum(
            to_float(row["popularity_balance_loss"]) for row in tail
        )
        / len(tail),
        "tail100_weighted_pop_mean": sum(
            to_float(row["weighted_popularity_balance_loss"]) for row in tail
        )
        / len(tail),
        "effective_weight_last": to_float(last["effective_popularity_balance_weight"]),
    }


def write_tensorboard(run_dir: Path, rows: list[dict[str, str]], SummaryWriter) -> None:
    writer = SummaryWriter(log_dir=str(run_dir))
    for row in rows:
        step = int(row["epoch"])
        for column in LOSS_COLUMNS:
            writer.add_scalar(column, to_float(row[column]), step)
    writer.flush()
    writer.close()


def write_scalar_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *LOSS_COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ["epoch", *LOSS_COLUMNS]})


def collect_current_best(args, repo: Path) -> list[dict[str, str]]:
    rows = read_csv(resolve_path(args.current_best_csv, repo))
    selected = []
    dataset_filter = set(args.datasets)
    for row in rows:
        dataset = row.get("dataset", "")
        if dataset_filter and dataset not in dataset_filter:
            continue
        output_dir = resolve_path(row["output_dir"], repo)
        selected.append(
            {
                "dataset": dataset,
                "run": row.get("selection_group", "current_best"),
                "output_dir": str(output_dir),
                "popularity_balance_weight": row.get("popularity_balance_weight", ""),
                "popularity_softmax_temperature": row.get("popularity_softmax_temperature", ""),
                "popularity_balance_disabled_layers": row.get(
                    "popularity_balance_disabled_layers", ""
                ),
                "test_ndcg@5": row.get("test_ndcg@5", ""),
                "test_pop_item_share@5": row.get("test_pop_item_share@5", ""),
                "test_unpop_item_share@5": row.get("test_unpop_item_share@5", ""),
            }
        )
    return selected


def write_fairness_comparison(args, repo: Path, output_dir: Path) -> None:
    baseline_path = resolve_path(args.baseline_csv, repo)
    current_path = resolve_path(args.current_best_csv, repo)
    if not baseline_path.exists() or not current_path.exists():
        return

    baseline = {row["dataset"]: row for row in read_csv(baseline_path)}
    current = {row["dataset"]: row for row in read_csv(current_path)}
    rows = []
    for dataset in sorted(set(baseline) & set(current)):
        base = baseline[dataset]
        cur = current[dataset]
        base_pop = to_float(base.get("test_pop_item_share@5", ""))
        cur_pop = to_float(cur.get("test_pop_item_share@5", ""))
        rows.append(
            {
                "dataset": dataset,
                "baseline_ndcg@5": base.get("test_ndcg@5", ""),
                "current_best_ndcg@5": cur.get("test_ndcg@5", ""),
                "baseline_pop_item_share@5": base.get("test_pop_item_share@5", ""),
                "current_best_pop_item_share@5": cur.get("test_pop_item_share@5", ""),
                "delta_pop_item_share@5": f"{cur_pop - base_pop:.12e}",
                "fairness_improved": str(cur_pop < base_pop),
                "current_output_dir": cur.get("output_dir", ""),
            }
        )

    with (output_dir / "fairness_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-best-csv",
        default="output/20260610_baseline_vs_current_best/current_best_setting_all_metrics.csv",
    )
    parser.add_argument(
        "--baseline-csv",
        default="output/20260610_baseline_vs_current_best/baseline_all_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="output/20260610_rqvae_loss_tensorboard",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["electronic", "sports"],
        help="Datasets to export. Empty means all rows in current-best CSV.",
    )
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output_dir = resolve_path(args.output_dir, repo)
    output_dir.mkdir(parents=True, exist_ok=True)

    SummaryWriter = maybe_summary_writer()
    if SummaryWriter is None:
        (output_dir / "README_tensorboard.txt").write_text(
            "tensorboard is not installed in this environment.\n"
            "Install it, then rerun this script to create event files:\n"
            "  poetry run pip install tensorboard\n"
            "  poetry run python tools/export_rqvae_loss_for_tensorboard.py\n"
            "After events are generated:\n"
            f"  poetry run tensorboard --logdir {output_dir / 'tensorboard'} --port 6006\n",
            encoding="utf-8",
        )

    summary_rows = []
    for selected in collect_current_best(args, repo):
        loss_path = Path(selected["output_dir"]) / "tokenizer_model" / "rqvae_loss_history.csv"
        if not loss_path.exists():
            selected["status"] = f"missing loss history: {loss_path}"
            summary_rows.append(selected)
            continue

        rows = read_csv(loss_path)
        run_name = f"{selected['dataset']}_{Path(selected['output_dir']).name}"
        run_dir = output_dir / "tensorboard" / run_name
        scalar_csv = output_dir / f"{run_name}_scalars.csv"
        write_scalar_csv(scalar_csv, rows)
        if SummaryWriter is not None:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            write_tensorboard(run_dir, rows, SummaryWriter)

        selected.update({key: f"{value:.12e}" for key, value in summarize_loss(rows).items()})
        selected["loss_history_csv"] = str(loss_path)
        selected["scalar_csv"] = str(scalar_csv)
        selected["tensorboard_dir"] = str(run_dir) if SummaryWriter is not None else ""
        selected["status"] = "ok"
        summary_rows.append(selected)

    fieldnames = sorted({key for row in summary_rows for key in row})
    with (output_dir / "loss_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_fairness_comparison(args, repo, output_dir)
    print(f"Wrote: {output_dir / 'loss_summary.csv'}")
    print(f"Wrote: {output_dir / 'fairness_comparison.csv'}")
    if SummaryWriter is None:
        print(f"TensorBoard events not written; see {output_dir / 'README_tensorboard.txt'}")
    else:
        print(f"TensorBoard logdir: {output_dir / 'tensorboard'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
