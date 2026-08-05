from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPOSITORY_URL = "https://github.com/SakanaAI/SearchCast.git"
REPOSITORY_COMMIT = "9a12b22525d787c0e0f919b2bd5b26fec5d64d03"
DATASET_SHA256 = "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842"
PAPER_MSE = {96: 0.081, 192: 0.167, 336: 0.305, 720: 0.811}
SERIES_GROUP_SIZES = (1, 2, 4, 8)
HORIZONS = tuple(PAPER_MSE)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPOSITORY = ROOT / ".tools" / "SearchCast"
DATASET = OFFICIAL_REPOSITORY / "data" / "exchange_rate.csv"
OUTPUT_ROOT = (
    ROOT
    / "data"
    / "training"
    / "runs"
    / "searchcast-exchange-official-9a12b225"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce and summarize SearchCast's Exchange-Rate benchmark."
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run all four official series-group sweeps even when outputs exist.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not clone or train; summarize already-complete outputs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subprocess_run(command: list[str], cwd: Path | None = None) -> None:
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def ensure_official_repository() -> None:
    if not OFFICIAL_REPOSITORY.exists():
        OFFICIAL_REPOSITORY.parent.mkdir(parents=True, exist_ok=True)
        subprocess_run(
            ["git", "clone", "--no-checkout", REPOSITORY_URL, str(OFFICIAL_REPOSITORY)]
        )
        subprocess_run(
            ["git", "checkout", "--detach", REPOSITORY_COMMIT],
            cwd=OFFICIAL_REPOSITORY,
        )

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=OFFICIAL_REPOSITORY, text=True
    ).strip()
    if head != REPOSITORY_COMMIT:
        raise RuntimeError(
            f"Expected SearchCast commit {REPOSITORY_COMMIT}, found {head}."
        )
    actual_hash = sha256(DATASET)
    if actual_hash != DATASET_SHA256:
        raise RuntimeError(
            f"Exchange dataset hash mismatch: expected {DATASET_SHA256}, found {actual_hash}."
        )


def benchmark_path(series_group_size: int) -> Path:
    return OUTPUT_ROOT / f"sgs{series_group_size}" / "benchmark_comparison.csv"


def run_official_sweep(series_group_size: int) -> None:
    output_directory = OUTPUT_ROOT / f"sgs{series_group_size}"
    output_directory.mkdir(parents=True, exist_ok=True)
    subprocess_run(
        [
            sys.executable,
            "optuna_ridge.py",
            "--input_csv",
            "data/exchange_rate.csv",
            "--output_dir",
            str(output_directory),
            "--scaler_scope",
            "local",
            "--scaler_method",
            "mean",
            "--local_horizon_group_size",
            "24",
            "--local_series_group_size",
            str(series_group_size),
            "--n_folds",
            "3",
            "--n_trials",
            "20",
            "--pool_series",
            "--instance_norm",
        ],
        cwd=OFFICIAL_REPOSITORY,
    )


def load_benchmark(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row[""]): {
                "mse": float(row["Local MSE"]),
                "mae": float(row["Local MAE"]),
                "global_mse": float(row["Global MSE"]),
                "global_mae": float(row["Global MAE"]),
            }
            for row in rows
        }


def load_exchange_values(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=range(1, 9))
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError(f"Unexpected Exchange dataset shape or values: {values.shape}.")
    return values


def standardize_like_official(values: np.ndarray) -> np.ndarray:
    train_rows = int(values.shape[0] * 0.7)
    train = values[:train_rows]
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[scales == 0.0] = 1.0
    return ((values - means) / scales).astype(np.float32)


def persistence_metrics(
    standardized_values: np.ndarray, horizon: int, context: int = 720
) -> dict[str, float | int]:
    rows = standardized_values.shape[0]
    train_rows = int(rows * 0.7)
    test_rows = int(rows * 0.2)
    validation_rows = rows - train_rows - test_rows
    test_start = train_rows + validation_rows
    test_with_context = standardized_values[test_start - context :]
    windows = np.lib.stride_tricks.sliding_window_view(
        test_with_context, context + horizon, axis=0
    ).transpose(1, 0, 2)
    actual = windows[:, :, context:]
    prediction = np.broadcast_to(windows[:, :, context - 1 : context], actual.shape)
    error = prediction - actual
    return {
        "windows_per_series": int(actual.shape[1]),
        "mse": float(np.mean(np.square(error), dtype=np.float64)),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
    }


def build_summary() -> dict[str, Any]:
    values = load_exchange_values(DATASET)
    standardized_values = standardize_like_official(values)
    persistence = {
        horizon: persistence_metrics(standardized_values, horizon)
        for horizon in HORIZONS
    }
    sweeps: dict[int, dict[str, Any]] = {}
    for series_group_size in SERIES_GROUP_SIZES:
        benchmark = load_benchmark(benchmark_path(series_group_size))
        cells = {}
        for horizon in HORIZONS:
            metrics = benchmark[horizon]
            baseline_mse = float(persistence[horizon]["mse"])
            cells[horizon] = {
                **metrics,
                "persistence_mse": baseline_mse,
                "persistence_mae": float(persistence[horizon]["mae"]),
                "mse_skill_vs_persistence_percent": 100.0
                * (1.0 - metrics["mse"] / baseline_mse),
                "paper_mse": PAPER_MSE[horizon],
                "difference_from_paper_mse": metrics["mse"] - PAPER_MSE[horizon],
            }
        mean_mse = float(np.mean([cells[h]["mse"] for h in HORIZONS]))
        mean_persistence_mse = float(
            np.mean([cells[h]["persistence_mse"] for h in HORIZONS])
        )
        sweeps[series_group_size] = {
            "mean_mse": mean_mse,
            "mean_persistence_mse": mean_persistence_mse,
            "mean_mse_skill_vs_persistence_percent": 100.0
            * (1.0 - mean_mse / mean_persistence_mse),
            "horizons": cells,
        }

    best_group_size = min(sweeps, key=lambda group: sweeps[group]["mean_mse"])
    return {
        "benchmark": "SearchCast Exchange-Rate",
        "official_repository": REPOSITORY_URL,
        "official_commit": REPOSITORY_COMMIT,
        "dataset_sha256": DATASET_SHA256,
        "dataset_rows": int(values.shape[0]),
        "dataset_series": int(values.shape[1]),
        "paper_mean_mse": float(np.mean(list(PAPER_MSE.values()))),
        "best_series_group_size": best_group_size,
        "sweeps": sweeps,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("sgs  H=96       H=192      H=336      H=720      mean")
    for group_size in SERIES_GROUP_SIZES:
        sweep = summary["sweeps"][group_size]
        cells = sweep["horizons"]
        values = "  ".join(f'{cells[h]["mse"]:.6f}' for h in HORIZONS)
        marker = " *" if group_size == summary["best_series_group_size"] else ""
        print(f"{group_size:<3}  {values}  {sweep['mean_mse']:.6f}{marker}")
    best = summary["sweeps"][summary["best_series_group_size"]]
    print(f"Paper mean MSE:       {summary['paper_mean_mse']:.6f}")
    print(f"Best replicated MSE:  {best['mean_mse']:.6f}")
    print(f"Persistence mean MSE: {best['mean_persistence_mse']:.6f}")
    print(
        "Skill vs persistence: "
        f"{best['mean_mse_skill_vs_persistence_percent']:+.6f}%"
    )


def main() -> None:
    args = parse_arguments()
    if not args.summary_only:
        ensure_official_repository()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for group_size in SERIES_GROUP_SIZES:
            if args.rerun or not benchmark_path(group_size).exists():
                run_official_sweep(group_size)
    elif not DATASET.exists():
        raise FileNotFoundError(
            "The pinned official dataset is missing; run without --summary-only first."
        )

    missing = [str(benchmark_path(group)) for group in SERIES_GROUP_SIZES if not benchmark_path(group).exists()]
    if missing:
        raise FileNotFoundError("Missing completed benchmark outputs: " + ", ".join(missing))

    summary = build_summary()
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
