"""Reproduce the transferable SearchCast analysis on canonical BTCUSDT candles.

The paper "How Good Can Linear Models Be for Time-Series Forecasting?" studies
how Ridge preprocessing choices reveal dataset-specific structure.  This module
adapts that protocol to one log-price series at six project-native candle
resolutions and writes reviewable Markdown reports, charts, and JSON results.

The adaptation is deliberately conservative:

* canonical spot BTCUSDT candles are the only input;
* 1h and 1d candles are derived from 1m candles, while 1M and 3M are complete
  UTC calendar months and quarters;
* the target is log close, so errors have a proportional interpretation;
* every scale uses a chronological 80/20 development/test split;
* preprocessing search uses three expanding development folds where the sample
  count permits it and the paper's 21-value inner Ridge-alpha grid;
* one deterministic day per calendar month is used at 1s to keep the study
  reproducible and bounded without pretending that disjoint days are adjacent.

Run from the repository root with ``npm run analysis:searchcast-btc``.
"""

from __future__ import annotations

import argparse
import calendar
import gc
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trading_storage import candle_times, read_candle_column


PAPER_URL = "https://arxiv.org/html/2606.27282v1"
COMMON_START = "2021-07-25"
COMMON_END = "2026-07-24"
ALPHAS = np.logspace(-6.0, 3.0, 21, dtype=np.float64)
EPSILON = 1e-12
SEED = 260627282
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class ScaleSpec:
    key: str
    label: str
    periods_per_year: float
    horizons: tuple[int, ...]
    lookbacks: tuple[int, ...]
    project_lookback: int | None
    train_samples: int
    validation_samples: int
    test_samples: int


SCALE_SPECS = (
    ScaleSpec("1s", "one second", 365.25 * 86_400, (1, 5, 15, 30, 60),
              (8, 16, 32, 64, 128, 256), 64, 3_000, 900, 1_800),
    ScaleSpec("1m", "one minute", 365.25 * 1_440, (1, 5, 15, 30, 60),
              (8, 16, 32, 64, 128, 256), 64, 3_000, 900, 1_800),
    ScaleSpec("1h", "one hour", 365.25 * 24, (1, 3, 6, 12, 24),
              (6, 12, 24, 48, 96, 168, 336), 32, 3_000, 900, 1_800),
    ScaleSpec("1d", "one day", 365.25, (1, 3, 7, 14, 30),
              (3, 7, 14, 30, 60, 90, 180, 365), 32, 1_200, 400, 600),
    ScaleSpec("1w", "one UTC calendar week", 365.25 / 7.0, (1, 2, 4, 8, 13, 26),
              (2, 4, 8, 13, 16, 26, 52), None, 1_000, 300, 400),
    ScaleSpec("1M", "one calendar month", 12.0, (1, 2, 3, 6),
              (2, 3, 6, 12, 18, 24), 16, 10_000, 10_000, 10_000),
    ScaleSpec("3M", "one calendar quarter", 4.0, (1, 2),
              (2, 3, 4, 6, 8), 16, 10_000, 10_000, 10_000),
)


@dataclass
class CandleFrame:
    scale: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    segment_bounds: list[tuple[int, int]]
    source_rows: int
    expected_source_rows: int
    sampling_note: str

    def model_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        log_close = np.log(self.close)
        return [
            (self.timestamps[start:end], log_close[start:end])
            for start, end in self.segment_bounds
            if end - start >= 3
        ]


@dataclass(frozen=True)
class Trial:
    lookback: int
    normalization_scope: str
    normalization_method: str
    local_ratio: float
    augmentation: str
    augmentation_sigma: float


@dataclass
class WindowBatch:
    x: np.ndarray
    y: np.ndarray
    target_times: np.ndarray


@dataclass
class NormalizationState:
    means: np.ndarray
    scales: np.ndarray
    reference_scale: float


@dataclass
class FittedModel:
    trial: Trial
    alpha: float
    coefficients: np.ndarray
    global_center: float | None
    global_scale: float | None
    reference_scale: float


@dataclass
class TrialScore:
    trial: Trial
    alpha: float
    cv_mse: float
    fold_mse: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/searchcast-btc-2026-08-04"),
        help="Report directory relative to the repository root.",
    )
    parser.add_argument(
        "--scales",
        default=",".join(spec.key for spec in SCALE_SPECS),
        help="Comma-separated subset of 1s,1m,1h,1d,1w,1M,3M.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Preprocessing trials per forecast horizon (paper default: 20).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small deterministic smoke run for development.",
    )
    parser.add_argument(
        "--clean-stale-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output).resolve()
    requested = [value.strip() for value in args.scales.split(",") if value.strip()]
    known = {spec.key: spec for spec in SCALE_SPECS}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"unknown scales: {', '.join(unknown)}")
    if args.trials < 4:
        raise ValueError("--trials must be at least 4")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "charts").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    remove_legacy_case_collisions(output_root)
    if args.clean_stale_only:
        return

    results: list[dict[str, object]] = []
    minute_frame: CandleFrame | None = None

    if "1s" in requested:
        print("Loading stratified 1s BTCUSDT sample...", flush=True)
        second_frame = load_second_sample(repo_root)
        results.append(run_scale(
            second_frame,
            known["1s"],
            output_root,
            trials=6 if args.quick else args.trials,
            quick=args.quick,
        ))
        del second_frame
        gc.collect()

    derived_requested = any(scale in requested for scale in ("1m", "1h", "1d", "1w", "1M", "3M"))
    if derived_requested:
        print("Loading continuous common-window 1m BTCUSDT candles...", flush=True)
        minute_frame = load_minute_common_window(repo_root)
        frames: dict[str, CandleFrame] = {"1m": minute_frame}
        if any(scale in requested for scale in ("1h", "1d", "1w", "1M", "3M")):
            frames["1h"] = aggregate_fixed(minute_frame, 60, "1h")
            frames["1d"] = aggregate_fixed(minute_frame, 1_440, "1d")
            frames["1w"] = aggregate_complete_weeks(frames["1d"])
            frames["1M"] = aggregate_complete_months(frames["1d"])
            frames["3M"] = aggregate_complete_quarters(frames["1M"])
        for scale in requested:
            if scale == "1s":
                continue
            results.append(run_scale(
                frames[scale],
                known[scale],
                output_root,
                trials=6 if args.quick else args.trials,
                quick=args.quick,
            ))
            if scale != "1m":
                del frames[scale]
            gc.collect()

    results.sort(key=lambda item: list(known).index(str(item["scale"])))
    write_overview(results, output_root, args)
    manifest = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper": PAPER_URL,
        "commonWindow": {"start": COMMON_START, "end": COMMON_END},
        "seed": SEED,
        "trialsPerHorizon": 6 if args.quick else args.trials,
        "quick": bool(args.quick),
        "scales": results,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(output_root / "results" / "all-scales.json", manifest)
    print(f"Wrote SearchCast BTC reports to {output_root}", flush=True)


def load_second_sample(repo_root: Path) -> CandleFrame:
    root = candle_reference_root(repo_root, "1s")
    files = [
        file for file in sorted(root.glob("*.json"))
        if COMMON_START <= file.stem <= COMMON_END
    ]
    grouped: dict[str, list[Path]] = {}
    for file in files:
        grouped.setdefault(file.stem[:7], []).append(file)
    selected = [
        min(month_files, key=lambda file: abs(int(file.stem[-2:]) - 15))
        for month_files in grouped.values()
    ]
    selected.sort()
    return load_reference_files(
        selected,
        scale="1s",
        expected_source_rows=len(files) * 86_400,
        sampling_note=(
            f"Deterministic stratified sample: {len(selected)} complete UTC days, "
            "one day nearest the 15th of every represented calendar month. "
            "Windows never cross the gaps between sampled days."
        ),
    )


def load_minute_common_window(repo_root: Path) -> CandleFrame:
    root = candle_reference_root(repo_root, "1m")
    files = [
        file for file in sorted(root.glob("*.json"))
        if COMMON_START <= file.stem <= COMMON_END
    ]
    start = datetime.fromisoformat(COMMON_START).date()
    end = datetime.fromisoformat(COMMON_END).date()
    expected_days = (end - start).days + 1
    if len(files) != expected_days:
        raise RuntimeError(
            f"common 1m window is not day-complete: {len(files)} files, "
            f"expected {expected_days}"
        )
    return load_reference_files(
        files,
        scale="1m",
        expected_source_rows=expected_days * 1_440,
        sampling_note="All canonical one-minute rows in the common continuous window.",
    )


def candle_reference_root(repo_root: Path, interval: str) -> Path:
    root = (
        repo_root / "data" / "market" / "immutable" / "refs" / "candles"
        / "spot-btcusdt" / "btcusdt" / interval
    )
    if not root.is_dir():
        raise FileNotFoundError(f"canonical candle directory is missing: {root}")
    return root


def load_reference_files(
    files: Sequence[Path],
    *,
    scale: str,
    expected_source_rows: int,
    sampling_note: str,
) -> CandleFrame:
    if not files:
        raise RuntimeError(f"no canonical {scale} candle references selected")
    chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in (*PRICE_COLUMNS, "volume")
    }
    timestamp_chunks: list[np.ndarray] = []
    segment_bounds: list[tuple[int, int]] = []
    offset = 0
    for index, file in enumerate(files):
        timestamp = candle_times(file).astype(np.int64, copy=False)
        values = {
            name: read_candle_column(file, name).astype(np.float64, copy=False)
            for name in chunks
        }
        size = timestamp.size
        if size == 0 or any(value.size != size for value in values.values()):
            raise RuntimeError(f"invalid candle shard dimensions: {file}")
        if not np.all(np.diff(timestamp) > 0):
            raise RuntimeError(f"non-monotone candle shard: {file}")
        if any(not np.isfinite(value).all() for value in values.values()):
            raise RuntimeError(f"non-finite candle shard: {file}")
        if any(bool((values[name] <= 0).any()) for name in PRICE_COLUMNS):
            raise RuntimeError(f"non-positive price in {file}")
        timestamp_chunks.append(timestamp)
        for name, value in values.items():
            chunks[name].append(value)
        segment_bounds.append((offset, offset + size))
        offset += size
        if (index + 1) % 250 == 0:
            print(f"  loaded {index + 1}/{len(files)} {scale} shards", flush=True)
    timestamps = np.concatenate(timestamp_chunks)
    fields = {name: np.concatenate(values) for name, values in chunks.items()}
    return CandleFrame(
        scale=scale,
        timestamps=timestamps,
        open=fields["open"],
        high=fields["high"],
        low=fields["low"],
        close=fields["close"],
        volume=fields["volume"],
        segment_bounds=segment_bounds if scale == "1s" else [(0, offset)],
        source_rows=offset,
        expected_source_rows=expected_source_rows,
        sampling_note=sampling_note,
    )


def aggregate_fixed(frame: CandleFrame, factor: int, scale: str) -> CandleFrame:
    if len(frame.segment_bounds) != 1:
        raise ValueError("fixed aggregation requires a continuous source frame")
    usable = frame.close.size - frame.close.size % factor
    if usable < factor:
        raise ValueError(f"not enough rows to aggregate {scale}")
    shape = (-1, factor)
    timestamps = frame.timestamps[:usable].reshape(shape)[:, 0]
    opening = frame.open[:usable].reshape(shape)[:, 0]
    high = frame.high[:usable].reshape(shape).max(axis=1)
    low = frame.low[:usable].reshape(shape).min(axis=1)
    close = frame.close[:usable].reshape(shape)[:, -1]
    volume = frame.volume[:usable].reshape(shape).sum(axis=1)
    rows = close.size
    return CandleFrame(
        scale=scale,
        timestamps=timestamps,
        open=opening,
        high=high,
        low=low,
        close=close,
        volume=volume,
        segment_bounds=[(0, rows)],
        source_rows=frame.source_rows,
        expected_source_rows=frame.expected_source_rows,
        sampling_note=(
            f"Derived from all common-window 1m candles in complete {scale} UTC bins."
        ),
    )


def aggregate_complete_months(daily: CandleFrame) -> CandleFrame:
    dates = daily.timestamps.astype("datetime64[ms]").astype("datetime64[D]")
    months = dates.astype("datetime64[M]")
    unique_months = np.unique(months)
    rows: list[tuple[int, float, float, float, float, float]] = []
    for month in unique_months:
        indices = np.flatnonzero(months == month)
        month_text = str(month)
        year, month_number = map(int, month_text.split("-"))
        expected = calendar.monthrange(year, month_number)[1]
        if indices.size != expected:
            continue
        selected_dates = dates[indices]
        if str(selected_dates[0]) != f"{month_text}-01":
            continue
        rows.append((
            int(daily.timestamps[indices[0]]),
            float(daily.open[indices[0]]),
            float(np.max(daily.high[indices])),
            float(np.min(daily.low[indices])),
            float(daily.close[indices[-1]]),
            float(np.sum(daily.volume[indices])),
        ))
    return frame_from_rows(
        "1M",
        rows,
        source_rows=daily.source_rows,
        expected_source_rows=daily.expected_source_rows,
        sampling_note="Complete UTC calendar months derived from common-window 1m candles.",
    )


def aggregate_complete_weeks(daily: CandleFrame) -> CandleFrame:
    """Aggregate complete Monday-Sunday UTC weeks from a continuous daily frame."""
    dates = daily.timestamps.astype("datetime64[ms]").astype("datetime64[D]")
    day_numbers = dates.astype(np.int64)
    weekdays = (day_numbers + 3) % 7  # 1970-01-01 was a Thursday (Monday=0).
    week_starts = dates - weekdays.astype("timedelta64[D]")
    rows: list[tuple[int, float, float, float, float, float]] = []
    for week_start in np.unique(week_starts):
        indices = np.flatnonzero(week_starts == week_start)
        if indices.size != 7 or weekdays[indices[0]] != 0 or weekdays[indices[-1]] != 6:
            continue
        if not np.all(np.diff(day_numbers[indices]) == 1):
            continue
        rows.append((
            int(daily.timestamps[indices[0]]),
            float(daily.open[indices[0]]),
            float(np.max(daily.high[indices])),
            float(np.min(daily.low[indices])),
            float(daily.close[indices[-1]]),
            float(np.sum(daily.volume[indices])),
        ))
    return frame_from_rows(
        "1w",
        rows,
        source_rows=daily.source_rows,
        expected_source_rows=daily.expected_source_rows,
        sampling_note=(
            "Complete Monday-Sunday UTC calendar weeks derived from common-window 1m candles."
        ),
    )


def aggregate_complete_quarters(monthly: CandleFrame) -> CandleFrame:
    month_values = monthly.timestamps.astype("datetime64[ms]").astype("datetime64[M]")
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, month in enumerate(month_values):
        year, month_number = map(int, str(month).split("-"))
        grouped.setdefault((year, (month_number - 1) // 3), []).append(index)
    rows: list[tuple[int, float, float, float, float, float]] = []
    for (_, _), indices in sorted(grouped.items()):
        if len(indices) != 3:
            continue
        months = [int(str(month_values[index]).split("-")[1]) for index in indices]
        if months[0] not in (1, 4, 7, 10) or months != list(range(months[0], months[0] + 3)):
            continue
        rows.append((
            int(monthly.timestamps[indices[0]]),
            float(monthly.open[indices[0]]),
            float(np.max(monthly.high[indices])),
            float(np.min(monthly.low[indices])),
            float(monthly.close[indices[-1]]),
            float(np.sum(monthly.volume[indices])),
        ))
    return frame_from_rows(
        "3M",
        rows,
        source_rows=monthly.source_rows,
        expected_source_rows=monthly.expected_source_rows,
        sampling_note="Complete UTC calendar quarters derived from complete monthly candles.",
    )


def frame_from_rows(
    scale: str,
    rows: Sequence[tuple[int, float, float, float, float, float]],
    *,
    source_rows: int,
    expected_source_rows: int,
    sampling_note: str,
) -> CandleFrame:
    if not rows:
        raise RuntimeError(f"aggregation produced no {scale} candles")
    matrix = np.asarray(rows, dtype=np.float64)
    count = len(rows)
    return CandleFrame(
        scale=scale,
        timestamps=matrix[:, 0].astype(np.int64),
        open=matrix[:, 1],
        high=matrix[:, 2],
        low=matrix[:, 3],
        close=matrix[:, 4],
        volume=matrix[:, 5],
        segment_bounds=[(0, count)],
        source_rows=source_rows,
        expected_source_rows=expected_source_rows,
        sampling_note=sampling_note,
    )


def run_scale(
    frame: CandleFrame,
    spec: ScaleSpec,
    output_root: Path,
    *,
    trials: int,
    quick: bool,
) -> dict[str, object]:
    print(f"Analyzing {spec.key} ({frame.close.size:,} candles)...", flush=True)
    profile = describe_frame(frame, spec)
    segments = frame.model_segments()
    split = chronological_split(segments)
    horizon_results: list[dict[str, object]] = []
    fitted_models: dict[int, FittedModel] = {}
    forecast_examples: dict[int, dict[str, np.ndarray]] = {}

    horizons = spec.horizons[:2] if quick else spec.horizons
    for horizon in horizons:
        print(f"  searching H={horizon} candles", flush=True)
        outcome = analyze_horizon(
            segments,
            spec,
            horizon,
            split,
            trial_count=trials,
            quick=quick,
        )
        horizon_results.append(outcome["result"])
        fitted_models[horizon] = outcome["model"]
        forecast_examples[horizon] = outcome["forecast"]

    power_law = fit_lookback_power_law(horizon_results)
    result: dict[str, object] = {
        "scale": spec.key,
        "label": spec.label,
        "profile": profile,
        "split": split,
        "projectLookback": spec.project_lookback,
        "horizons": horizon_results,
        "lookbackPowerLaw": power_law,
        "limitations": scale_limitations(frame, spec, horizon_results),
    }
    chart_paths = render_scale_charts(
        frame,
        spec,
        result,
        fitted_models,
        forecast_examples,
        output_root / "charts",
    )
    result["charts"] = chart_paths
    write_json(output_root / "results" / f"{artifact_slug(spec.key)}.json", result)
    write_scale_report(result, output_root)
    return result


def chronological_split(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    all_times = np.concatenate([timestamps for timestamps, _ in segments])
    if all_times.size < 12:
        raise RuntimeError("at least 12 observations are required")
    all_times.sort()
    indices = {
        "start": 0,
        "fold1TrainEnd": max(1, int(math.floor(all_times.size * 0.50))),
        "fold1ValidationEnd": max(2, int(math.floor(all_times.size * 0.60))),
        "fold2ValidationEnd": max(3, int(math.floor(all_times.size * 0.70))),
        "developmentEnd": max(4, int(math.floor(all_times.size * 0.80))),
        "end": all_times.size,
    }
    for name, index in list(indices.items()):
        if name == "end":
            continue
        indices[name] = min(index, all_times.size - 1)
    end_exclusive = int(all_times[-1]) + infer_step_ms(all_times)
    split = {
        "startMs": int(all_times[0]),
        "fold1TrainEndMs": int(all_times[indices["fold1TrainEnd"]]),
        "fold1ValidationEndMs": int(all_times[indices["fold1ValidationEnd"]]),
        "fold2ValidationEndMs": int(all_times[indices["fold2ValidationEnd"]]),
        "developmentEndMs": int(all_times[indices["developmentEnd"]]),
        "endMs": end_exclusive,
        "developmentRows": indices["developmentEnd"],
        "testRows": all_times.size - indices["developmentEnd"],
    }
    return split


def analyze_horizon(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    spec: ScaleSpec,
    horizon: int,
    split: dict[str, object],
    *,
    trial_count: int,
    quick: bool,
) -> dict[str, object]:
    folds = fold_ranges(split)
    lookbacks = valid_lookbacks(
        segments,
        spec.lookbacks[:3] if quick else spec.lookbacks,
        horizon,
        folds,
        spec.train_samples,
    )
    if not lookbacks:
        raise RuntimeError(f"no valid lookbacks for {spec.key} H={horizon}")
    max_lookback = max(lookbacks)
    fold_batches = []
    for fold_index, (train_start, train_end, validation_start, validation_end) in enumerate(folds):
        train = sample_windows(
            segments,
            train_start,
            train_end,
            max_lookback,
            horizon,
            spec.train_samples if not quick else min(500, spec.train_samples),
        )
        validation = sample_windows(
            segments,
            validation_start,
            validation_end,
            max_lookback,
            horizon,
            spec.validation_samples if not quick else min(200, spec.validation_samples),
        )
        if train.x.shape[0] < max(4, max_lookback + 2) or validation.x.shape[0] < 1:
            continue
        fold_batches.append((train, validation, fold_index))
    if not fold_batches:
        raise RuntimeError(f"no valid expanding folds for {spec.key} H={horizon}")

    trials = generate_trials(lookbacks, trial_count, seed=SEED + horizon * 97 + len(spec.key))
    scores = [score_trial(trial, fold_batches, horizon) for trial in trials]
    scores.sort(key=lambda score: (score.cv_mse, score.trial.lookback))
    best = scores[0]

    baseline_reference = spec.project_lookback if spec.project_lookback is not None else 16
    fixed_lookback = min(lookbacks, key=lambda value: abs(value - baseline_reference))
    baseline_trials = {
        "global_ridge": Trial(
            fixed_lookback, "global", "standard", 1.0, "none", 0.0,
        ),
        "local_full_ridge": Trial(
            fixed_lookback, "local", "standard", 1.0, "none", 0.0,
        ),
    }
    baseline_scores = {
        name: score_trial(trial, fold_batches, horizon)
        for name, trial in baseline_trials.items()
    }

    development = sample_windows(
        segments,
        int(split["startMs"]),
        int(split["developmentEndMs"]),
        max_lookback,
        horizon,
        spec.train_samples if not quick else min(700, spec.train_samples),
    )
    test = sample_windows(
        segments,
        int(split["developmentEndMs"]),
        int(split["endMs"]),
        max_lookback,
        horizon,
        spec.test_samples if not quick else min(300, spec.test_samples),
    )
    if development.x.shape[0] < 3 or test.x.shape[0] < 1:
        raise RuntimeError(f"insufficient final windows for {spec.key} H={horizon}")

    tuned_model = fit_model(development, best.trial, best.alpha, seed=SEED + horizon)
    tuned_prediction = predict_model(tuned_model, test.x[:, -best.trial.lookback:])
    models: dict[str, dict[str, object]] = {
        "persistence": evaluate_prediction(
            test,
            np.repeat(test.x[:, -1:], horizon, axis=1),
            development,
        ),
        "tuned_ridge": evaluate_prediction(test, tuned_prediction, development),
    }
    baseline_predictions: dict[str, np.ndarray] = {}
    for name, trial in baseline_trials.items():
        score = baseline_scores[name]
        model = fit_model(development, trial, score.alpha, seed=SEED + horizon + 13)
        prediction = predict_model(model, test.x[:, -trial.lookback:])
        models[name] = evaluate_prediction(test, prediction, development)
        baseline_predictions[name] = prediction

    persistence_mse = float(models["persistence"]["normalizedMse"])
    tuned_mse = float(models["tuned_ridge"]["normalizedMse"])
    models["tuned_ridge"]["gainVsPersistencePercent"] = (
        100.0 * (1.0 - tuned_mse / persistence_mse) if persistence_mse > 0 else 0.0
    )
    near_best = [
        score for score in scores
        if score.cv_mse <= best.cv_mse * 1.01 + 1e-12
    ]
    result = {
        "horizonCandles": horizon,
        "cvFolds": len(fold_batches),
        "validLookbacks": list(lookbacks),
        "fixedBaselineLookback": fixed_lookback,
        "trainWindows": int(development.x.shape[0]),
        "testWindows": int(test.x.shape[0]),
        "bestTrial": trial_dict(best),
        "nearBestLookbackIqr": percentile_triplet(
            np.asarray([score.trial.lookback for score in near_best], dtype=np.float64)
        ),
        "models": models,
        "baselineCv": {
            name: trial_dict(score) for name, score in baseline_scores.items()
        },
        "topTrials": [trial_dict(score) for score in scores[: min(10, len(scores))]],
    }
    example_index = test.x.shape[0] - 1
    forecast = {
        "context": test.x[example_index, -best.trial.lookback:].copy(),
        "truth": test.y[example_index].copy(),
        "tuned": tuned_prediction[example_index].copy(),
        "global": baseline_predictions["global_ridge"][example_index].copy(),
        "time": np.asarray([test.target_times[example_index]], dtype=np.int64),
    }
    return {"result": result, "model": tuned_model, "forecast": forecast}


def fold_ranges(split: dict[str, object]) -> list[tuple[int, int, int, int]]:
    start = int(split["startMs"])
    fold1_train = int(split["fold1TrainEndMs"])
    fold1_validation = int(split["fold1ValidationEndMs"])
    fold2_validation = int(split["fold2ValidationEndMs"])
    development_end = int(split["developmentEndMs"])
    return [
        (start, fold1_train, fold1_train, fold1_validation),
        (start, fold1_validation, fold1_validation, fold2_validation),
        (start, fold2_validation, fold2_validation, development_end),
    ]


def valid_lookbacks(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    candidates: Sequence[int],
    horizon: int,
    folds: Sequence[tuple[int, int, int, int]],
    maximum_samples: int,
) -> tuple[int, ...]:
    valid = []
    for lookback in candidates:
        if lookback < 2:
            continue
        train_start, train_end, validation_start, validation_end = folds[0]
        train = sample_windows(
            segments, train_start, train_end, lookback, horizon,
            min(maximum_samples, max(lookback + 2, 32)),
        )
        validation = sample_windows(
            segments, validation_start, validation_end, lookback, horizon, 8,
        )
        if train.x.shape[0] >= max(4, lookback + 2) and validation.x.shape[0] >= 1:
            valid.append(lookback)
    return tuple(valid)


def sample_windows(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    start_ms: int,
    end_ms: int,
    lookback: int,
    horizon: int,
    maximum_samples: int,
) -> WindowBatch:
    ranges: list[tuple[int, int, int]] = []
    total = 0
    for segment_index, (timestamps, values) in enumerate(segments):
        if values.size != timestamps.size:
            raise ValueError("timestamp/value segment mismatch")
        target_left = max(lookback, int(np.searchsorted(timestamps, start_ms, side="left")))
        end_index = int(np.searchsorted(timestamps, end_ms, side="left"))
        target_right = min(values.size - horizon + 1, end_index - horizon + 1)
        count = max(0, target_right - target_left)
        if count:
            ranges.append((segment_index, target_left, target_right))
            total += count
    if total == 0:
        return WindowBatch(
            np.empty((0, lookback), dtype=np.float64),
            np.empty((0, horizon), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    requested = min(maximum_samples, total)
    flat = np.unique(np.linspace(0, total - 1, requested, dtype=np.int64))
    counts = np.asarray([right - left for _, left, right in ranges], dtype=np.int64)
    cumulative = np.cumsum(counts)
    segment_choices = np.searchsorted(cumulative, flat, side="right")
    previous = np.concatenate((np.asarray([0], dtype=np.int64), cumulative[:-1]))
    x = np.empty((flat.size, lookback), dtype=np.float64)
    y = np.empty((flat.size, horizon), dtype=np.float64)
    target_times = np.empty(flat.size, dtype=np.int64)
    lag_offsets = np.arange(lookback, 0, -1, dtype=np.int64)
    horizon_offsets = np.arange(horizon, dtype=np.int64)
    for output_index, (flat_index, range_index) in enumerate(zip(flat, segment_choices, strict=True)):
        segment_index, left, _ = ranges[int(range_index)]
        position = left + int(flat_index - previous[int(range_index)])
        timestamps, values = segments[segment_index]
        x[output_index] = values[position - lag_offsets]
        y[output_index] = values[position + horizon_offsets]
        target_times[output_index] = timestamps[position]
    return WindowBatch(x, y, target_times)


def generate_trials(
    lookbacks: Sequence[int],
    count: int,
    *,
    seed: int,
) -> list[Trial]:
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []
    for index, lookback in enumerate(lookbacks):
        trials.append(Trial(lookback, "global", "standard", 1.0, "none", 0.0))
        if len(trials) >= count:
            return trials
        ratio = float(10.0 ** np.linspace(-2.5, 0.0, len(lookbacks))[index])
        trials.append(Trial(lookback, "local", "standard", ratio, "none", 0.0))
        if len(trials) >= count:
            return trials
    augmentation_cycle = ("time", "frequency", "none")
    while len(trials) < count:
        lookback = int(rng.choice(lookbacks))
        local = bool(rng.random() < 0.82)
        method = "robust" if rng.random() < 0.20 else "standard"
        augmentation = augmentation_cycle[len(trials) % len(augmentation_cycle)]
        sigma = 0.0 if augmentation == "none" else float(10.0 ** rng.uniform(-3.0, math.log10(0.5)))
        trials.append(Trial(
            lookback=lookback,
            normalization_scope="local" if local else "global",
            normalization_method=method,
            local_ratio=float(10.0 ** rng.uniform(-2.5, 0.0)) if local else 1.0,
            augmentation=augmentation,
            augmentation_sigma=sigma,
        ))
    return trials


def score_trial(
    trial: Trial,
    fold_batches: Sequence[tuple[WindowBatch, WindowBatch, int]],
    horizon: int,
) -> TrialScore:
    per_alpha: list[list[float]] = [[] for _ in ALPHAS]
    for train, validation, fold_index in fold_batches:
        train_slice = slice(-trial.lookback, None)
        x_train = train.x[:, train_slice]
        x_validation = validation.x[:, train_slice]
        design_train, normalized_y, train_state, global_parameters = normalize_training(
            x_train,
            train.y,
            trial,
            seed=SEED + horizon * 101 + fold_index * 17 + trial.lookback,
        )
        design_validation, validation_state = normalize_evaluation(
            x_validation,
            trial,
            global_parameters,
        )
        coefficients = ridge_alpha_grid(design_train, normalized_y, ALPHAS)
        for alpha_index, coefficient in enumerate(coefficients):
            normalized_prediction = design_validation @ coefficient
            prediction = denormalize(normalized_prediction, validation_state)
            mse = float(np.mean(np.square(prediction - validation.y)))
            normalized_mse = mse / max(train_state.reference_scale ** 2, EPSILON)
            per_alpha[alpha_index].append(normalized_mse)
    means = np.asarray([
        np.mean(values) if values else math.inf for values in per_alpha
    ], dtype=np.float64)
    best_index = int(np.argmin(means))
    return TrialScore(
        trial=trial,
        alpha=float(ALPHAS[best_index]),
        cv_mse=float(means[best_index]),
        fold_mse=[float(value) for value in per_alpha[best_index]],
    )


def fit_model(batch: WindowBatch, trial: Trial, alpha: float, *, seed: int) -> FittedModel:
    x = batch.x[:, -trial.lookback:]
    design, normalized_y, state, global_parameters = normalize_training(
        x, batch.y, trial, seed=seed,
    )
    coefficient = ridge_alpha_grid(
        design, normalized_y, np.asarray([alpha], dtype=np.float64)
    )[0]
    global_center, global_scale = global_parameters
    return FittedModel(
        trial=trial,
        alpha=alpha,
        coefficients=coefficient,
        global_center=global_center,
        global_scale=global_scale,
        reference_scale=state.reference_scale,
    )


def predict_model(model: FittedModel, x: np.ndarray) -> np.ndarray:
    design, state = normalize_evaluation(
        x,
        model.trial,
        (model.global_center, model.global_scale),
    )
    return denormalize(design @ model.coefficients, state)


def normalize_training(
    x: np.ndarray,
    y: np.ndarray,
    trial: Trial,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, NormalizationState, tuple[float | None, float | None]]:
    reference_scale = float(np.std(x))
    if not math.isfinite(reference_scale) or reference_scale < EPSILON:
        reference_scale = 1.0
    if trial.normalization_scope == "global":
        center, scale = center_scale(x.reshape(-1), trial.normalization_method)
        means = np.full((x.shape[0], 1), center, dtype=np.float64)
        scales = np.full((x.shape[0], 1), scale, dtype=np.float64)
        normalized_x = (x - center) / scale
        normalized_y = (y - center) / scale
        design = np.column_stack((normalized_x, np.ones(x.shape[0], dtype=np.float64)))
        global_parameters: tuple[float | None, float | None] = (center, scale)
    else:
        count = effective_local_count(x.shape[1], trial.local_ratio)
        local = x[:, -count:]
        means, scales = row_center_scale(local, trial.normalization_method)
        normalized_x = (x - means) / scales
        normalized_y = (y - means) / scales
        design = np.column_stack((normalized_x, scales[:, 0]))
        global_parameters = (None, None)
    design = augment_design(design, x.shape[1], trial, seed=seed)
    return (
        design,
        normalized_y,
        NormalizationState(means, scales, reference_scale),
        global_parameters,
    )


def normalize_evaluation(
    x: np.ndarray,
    trial: Trial,
    global_parameters: tuple[float | None, float | None],
) -> tuple[np.ndarray, NormalizationState]:
    reference_scale = float(np.std(x))
    if not math.isfinite(reference_scale) or reference_scale < EPSILON:
        reference_scale = 1.0
    if trial.normalization_scope == "global":
        center, scale = global_parameters
        if center is None or scale is None:
            raise ValueError("global normalization parameters are missing")
        means = np.full((x.shape[0], 1), center, dtype=np.float64)
        scales = np.full((x.shape[0], 1), scale, dtype=np.float64)
        design = np.column_stack(((x - center) / scale, np.ones(x.shape[0])))
    else:
        count = effective_local_count(x.shape[1], trial.local_ratio)
        means, scales = row_center_scale(x[:, -count:], trial.normalization_method)
        design = np.column_stack(((x - means) / scales, scales[:, 0]))
    return design, NormalizationState(means, scales, reference_scale)


def center_scale(values: np.ndarray, method: str) -> tuple[float, float]:
    if method == "robust":
        center = float(np.median(values))
        q25, q75 = np.quantile(values, (0.25, 0.75))
        scale = float(q75 - q25)
    else:
        center = float(np.mean(values))
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale < EPSILON:
        scale = 1.0
    return center, scale


def row_center_scale(values: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "robust":
        means = np.median(values, axis=1, keepdims=True)
        quartiles = np.quantile(values, (0.25, 0.75), axis=1)
        scales = (quartiles[1] - quartiles[0])[:, None]
    else:
        means = np.mean(values, axis=1, keepdims=True)
        scales = np.std(values, axis=1, keepdims=True)
    scales = np.where(np.isfinite(scales) & (scales >= EPSILON), scales, 1.0)
    return means, scales


def effective_local_count(lookback: int, ratio: float) -> int:
    return min(lookback, max(2, int(math.ceil(lookback * ratio))))


def augment_design(
    design: np.ndarray,
    history_columns: int,
    trial: Trial,
    *,
    seed: int,
) -> np.ndarray:
    if trial.augmentation == "none" or trial.augmentation_sigma <= 0:
        return design
    output = design.copy()
    history = output[:, :history_columns]
    rng = np.random.default_rng(seed)
    if trial.augmentation == "time":
        history += rng.normal(0.0, trial.augmentation_sigma, size=history.shape)
    elif trial.augmentation == "frequency":
        spectrum = np.fft.rfft(history, axis=1)
        perturbation = (
            rng.normal(size=spectrum.shape) + 1j * rng.normal(size=spectrum.shape)
        ) / math.sqrt(2.0)
        spectrum *= 1.0 + trial.augmentation_sigma * perturbation
        history[:] = np.fft.irfft(spectrum, n=history_columns, axis=1)
    else:
        raise ValueError(f"unknown augmentation: {trial.augmentation}")
    return output


def ridge_alpha_grid(
    x: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
) -> list[np.ndarray]:
    gram = x.T @ x
    cross = x.T @ y
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ cross
    return [
        eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
        for alpha in alphas
    ]


def denormalize(prediction: np.ndarray, state: NormalizationState) -> np.ndarray:
    return prediction * state.scales + state.means


def evaluate_prediction(
    test: WindowBatch,
    prediction: np.ndarray,
    development: WindowBatch,
) -> dict[str, object]:
    error = prediction - test.y
    reference_scale = float(np.std(development.x))
    reference_scale = reference_scale if reference_scale >= EPSILON else 1.0
    mse = float(np.mean(np.square(error)))
    endpoint_actual = np.sign(test.y[:, -1] - test.x[:, -1])
    endpoint_prediction = np.sign(prediction[:, -1] - test.x[:, -1])
    nonzero = endpoint_actual != 0
    direction = float(np.mean(endpoint_prediction[nonzero] == endpoint_actual[nonzero])) \
        if bool(nonzero.any()) else math.nan
    return {
        "normalizedMse": mse / (reference_scale ** 2),
        "rmseBps": math.sqrt(mse) * 10_000.0,
        "maeBps": float(np.mean(np.abs(error))) * 10_000.0,
        "endpointDirectionAccuracy": direction,
    }


def describe_frame(frame: CandleFrame, spec: ScaleSpec) -> dict[str, object]:
    returns = np.concatenate([
        np.diff(np.log(frame.close[start:end]))
        for start, end in frame.segment_bounds
        if end - start >= 2
    ])
    absolute_returns = np.abs(returns)
    mean = float(np.mean(returns))
    deviation = float(np.std(returns))
    centered = returns - mean
    if deviation < EPSILON:
        skew = 0.0
        excess_kurtosis = 0.0
    else:
        skew = float(np.mean(centered ** 3) / (deviation ** 3))
        excess_kurtosis = float(np.mean(centered ** 4) / (deviation ** 4) - 3.0)
    acf_lags = autocorrelation_lags(returns, profile_lags(frame.scale, returns.size))
    absolute_acf = autocorrelation_lags(absolute_returns, profile_lags(frame.scale, returns.size))
    return {
        "candles": int(frame.close.size),
        "sourceRowsRead": int(frame.source_rows),
        "expectedSourceRows": int(frame.expected_source_rows),
        "sourceCoveragePercent": 100.0 * frame.source_rows / frame.expected_source_rows,
        "start": iso_time(int(frame.timestamps[0])),
        "end": iso_time(int(frame.timestamps[-1])),
        "samplingNote": frame.sampling_note,
        "priceMinimum": float(np.min(frame.low)),
        "priceMaximum": float(np.max(frame.high)),
        "endingClose": float(frame.close[-1]),
        "meanLogReturnBps": mean * 10_000.0,
        "medianLogReturnBps": float(np.median(returns)) * 10_000.0,
        "returnStdBps": deviation * 10_000.0,
        "meanAbsoluteReturnBps": float(np.mean(absolute_returns)) * 10_000.0,
        "annualizedVolatilityPercent": deviation * math.sqrt(spec.periods_per_year) * 100.0,
        "skewness": skew,
        "excessKurtosis": excess_kurtosis,
        "meanCandleRangeBps": float(np.mean(np.log(frame.high / frame.low))) * 10_000.0,
        "medianVolumeBtc": float(np.median(frame.volume)),
        "returnAcf": acf_lags,
        "absoluteReturnAcf": absolute_acf,
    }


def profile_lags(scale: str, size: int) -> tuple[int, ...]:
    candidates = {
        "1s": (1, 2, 5, 15, 30, 60, 300),
        "1m": (1, 2, 5, 15, 60, 360, 1_440),
        "1h": (1, 2, 6, 12, 24, 168, 336),
        "1d": (1, 2, 7, 14, 30, 90, 365),
        "1w": (1, 2, 4, 13, 26, 52),
        "1M": (1, 2, 3, 6, 12, 24),
        "3M": (1, 2, 3, 4),
    }[scale]
    return tuple(lag for lag in candidates if lag < size)


def autocorrelation_lags(values: np.ndarray, lags: Sequence[int]) -> dict[str, float]:
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered))
    if variance < EPSILON:
        return {str(lag): 0.0 for lag in lags}
    return {
        str(lag): float(np.dot(centered[:-lag], centered[lag:]) / variance)
        for lag in lags
    }


def fit_lookback_power_law(horizons: Sequence[dict[str, object]]) -> dict[str, float]:
    x = np.log(np.asarray([item["horizonCandles"] for item in horizons], dtype=np.float64))
    y = np.log(np.asarray([
        item["bestTrial"]["lookback"]  # type: ignore[index]
        for item in horizons
    ], dtype=np.float64))
    if x.size < 2 or np.allclose(x, x[0]):
        return {"a": float(np.exp(np.mean(y))), "b": 0.0, "rSquared": 0.0}
    b, log_a = np.polyfit(x, y, 1)
    fitted = log_a + b * x
    total = float(np.sum(np.square(y - np.mean(y))))
    residual = float(np.sum(np.square(y - fitted)))
    r_squared = 1.0 - residual / total if total > EPSILON else 1.0
    return {"a": float(np.exp(log_a)), "b": float(b), "rSquared": r_squared}


def scale_limitations(
    frame: CandleFrame,
    spec: ScaleSpec,
    horizons: Sequence[dict[str, object]],
) -> list[str]:
    limitations = [
        "The target is univariate log close; the paper's cross-series grouping sweep is not applicable.",
        "Search uses deterministic joint trials rather than Optuna TPE; the searched axes and 21-value alpha loop match the paper.",
        "No nonlinear baseline is trained; the comparison isolates preprocessing gains against persistence and fixed Ridge baselines.",
    ]
    if spec.key == "1s":
        limitations.append(
            "The 1s fit uses one complete UTC day per month; it spans every month but is not an exhaustive 157.8M-row fit."
        )
    if frame.close.size < 100:
        limitations.append(
            f"Only {frame.close.size} complete {spec.key} candles exist in the common window; estimates are exploratory and high variance."
        )
    if any(int(item["testWindows"]) < 30 for item in horizons):
        limitations.append(
            "At least one horizon has fewer than 30 held-out windows, so test metrics are descriptive rather than inferential."
        )
    return limitations


def trial_dict(score: TrialScore) -> dict[str, object]:
    value = asdict(score.trial)
    value.update({
        "alpha": score.alpha,
        "cvNormalizedMse": score.cv_mse,
        "foldNormalizedMse": score.fold_mse,
        "effectiveLocalPoints": effective_local_count(
            score.trial.lookback, score.trial.local_ratio
        ) if score.trial.normalization_scope == "local" else None,
    })
    return value


def percentile_triplet(values: np.ndarray) -> dict[str, float]:
    q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
    return {"q25": float(q25), "median": float(median), "q75": float(q75)}


def render_scale_charts(
    frame: CandleFrame,
    spec: ScaleSpec,
    result: dict[str, object],
    fitted_models: dict[int, FittedModel],
    forecasts: dict[int, dict[str, np.ndarray]],
    chart_root: Path,
) -> list[str]:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    paths = []
    slug = artifact_slug(spec.key)
    horizons = result["horizons"]
    assert isinstance(horizons, list)

    path = chart_root / f"{slug}-data-profile.png"
    plot_data_profile(frame, spec, result["profile"], path)
    paths.append(f"charts/{path.name}")

    path = chart_root / f"{slug}-model-comparison.png"
    plot_model_comparison(spec, horizons, path)
    paths.append(f"charts/{path.name}")

    path = chart_root / f"{slug}-lookback-horizon.png"
    plot_lookback_horizon(spec, horizons, result["lookbackPowerLaw"], path)
    paths.append(f"charts/{path.name}")

    path = chart_root / f"{slug}-hyperparameters.png"
    plot_hyperparameters(spec, horizons, path)
    paths.append(f"charts/{path.name}")

    path = chart_root / f"{slug}-weights.png"
    plot_weights(spec, fitted_models, path)
    paths.append(f"charts/{path.name}")

    maximum_horizon = max(forecasts)
    path = chart_root / f"{slug}-forecast.png"
    plot_forecast(spec, maximum_horizon, forecasts[maximum_horizon], path)
    paths.append(f"charts/{path.name}")
    return paths


def plot_data_profile(
    frame: CandleFrame,
    spec: ScaleSpec,
    profile: object,
    path: Path,
) -> None:
    assert isinstance(profile, dict)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True)
    indices = downsample_indices(frame.close.size, 2_000)
    dates = frame.timestamps[indices].astype("datetime64[ms]")
    axes[0].plot(dates, frame.close[indices], color="#2563eb", linewidth=1.0)
    axes[0].set_title(f"BTCUSDT close at {spec.key}")
    axes[0].set_ylabel("USDT")
    acf = profile["returnAcf"]
    absolute = profile["absoluteReturnAcf"]
    assert isinstance(acf, dict) and isinstance(absolute, dict)
    labels = list(acf)
    x = np.arange(len(labels))
    width = 0.38
    axes[1].bar(x - width / 2, [acf[label] for label in labels], width,
                label="return", color="#2563eb")
    axes[1].bar(x + width / 2, [absolute[label] for label in labels], width,
                label="absolute return", color="#f59e0b")
    axes[1].axhline(0.0, color="#111827", linewidth=0.7)
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("lag (candles)")
    axes[1].set_ylabel("autocorrelation")
    axes[1].set_title("Linear and volatility memory")
    axes[1].legend(frameon=False)
    save_figure(figure, path)


def plot_model_comparison(
    spec: ScaleSpec,
    horizons: Sequence[dict[str, object]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = [int(item["horizonCandles"]) for item in horizons]
    styles = {
        "persistence": ("Persistence", "#6b7280", "o"),
        "global_ridge": ("Fixed global Ridge", "#dc2626", "s"),
        "local_full_ridge": ("Fixed full-window local Ridge", "#f59e0b", "^"),
        "tuned_ridge": ("Tuned Ridge", "#2563eb", "o"),
    }
    for key, (label, color, marker) in styles.items():
        y = [float(item["models"][key]["normalizedMse"]) for item in horizons]  # type: ignore[index]
        axis.plot(x, y, label=label, color=color, marker=marker, linewidth=1.5)
    axis.set_xlabel(f"forecast horizon ({spec.key} candles)")
    axis.set_ylabel("held-out normalized MSE")
    axis.set_title("Forecast error by preprocessing choice")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(x, [str(value) for value in x])
    axis.legend(frameon=False)
    save_figure(figure, path)


def plot_lookback_horizon(
    spec: ScaleSpec,
    horizons: Sequence[dict[str, object]],
    power_law: object,
    path: Path,
) -> None:
    assert isinstance(power_law, dict)
    figure, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = np.asarray([item["horizonCandles"] for item in horizons], dtype=np.float64)
    y = np.asarray([item["bestTrial"]["lookback"] for item in horizons], dtype=np.float64)  # type: ignore[index]
    lower = np.asarray([item["nearBestLookbackIqr"]["q25"] for item in horizons], dtype=np.float64)  # type: ignore[index]
    upper = np.asarray([item["nearBestLookbackIqr"]["q75"] for item in horizons], dtype=np.float64)  # type: ignore[index]
    axis.fill_between(x, lower, upper, color="#93c5fd", alpha=0.45, label="near-best IQR")
    axis.plot(x, y, color="#2563eb", marker="o", linewidth=1.6, label="selected lookback")
    fit_x = np.geomspace(float(np.min(x)), float(np.max(x)), 100)
    fit_y = float(power_law["a"]) * fit_x ** float(power_law["b"])
    axis.plot(fit_x, fit_y, color="#111827", linestyle="--", linewidth=1.0,
              label=f"L = {float(power_law['a']):.1f} H^{float(power_law['b']):+.2f}")
    if spec.project_lookback is not None:
        axis.axhline(spec.project_lookback, color="#dc2626", linestyle=":", linewidth=1.0,
                     label=f"project default L={spec.project_lookback}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(x, [str(int(value)) for value in x])
    y_ticks = [*map(int, y)]
    if spec.project_lookback is not None:
        y_ticks.append(spec.project_lookback)
    axis.set_yticks(sorted(set(y_ticks)))
    axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.set_xlabel(f"forecast horizon ({spec.key} candles)")
    axis.set_ylabel("lookback (candles)")
    axis.set_title("Optimal context versus forecast horizon")
    axis.legend(frameon=False)
    save_figure(figure, path)


def plot_hyperparameters(
    spec: ScaleSpec,
    horizons: Sequence[dict[str, object]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 6.3), constrained_layout=True)
    x = np.asarray([item["horizonCandles"] for item in horizons], dtype=np.float64)
    trials = [item["bestTrial"] for item in horizons]
    lookback = [float(trial["lookback"]) for trial in trials]
    ratios = [float(trial["local_ratio"]) if trial["normalization_scope"] == "local" else 1.0 for trial in trials]
    alphas = [float(trial["alpha"]) for trial in trials]
    sigmas = [float(trial["augmentation_sigma"]) for trial in trials]
    axes[0, 0].plot(x, lookback, color="#2563eb", marker="o")
    axes[0, 0].set_ylabel("candles")
    axes[0, 0].set_title("Lookback L")
    axes[0, 1].plot(x, ratios, color="#7c3aed", marker="o")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("trailing fraction r")
    axes[0, 1].set_title("Normalization locality")
    axes[1, 0].plot(x, alphas, color="#059669", marker="o")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("alpha")
    axes[1, 0].set_title("Ridge regularization")
    colors = {"none": "#6b7280", "time": "#f59e0b", "frequency": "#dc2626"}
    for horizon, sigma, trial in zip(x, sigmas, trials, strict=True):
        axes[1, 1].scatter(horizon, max(sigma, 5e-4), s=45,
                           color=colors[str(trial["augmentation"])],
                           label=str(trial["augmentation"]))
    handles, labels = axes[1, 1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes[1, 1].legend(unique.values(), unique.keys(), frameon=False)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("noise sigma (none shown at floor)")
    axes[1, 1].set_title("Selected augmentation")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xticks(x, [str(int(value)) for value in x])
        axis.set_xlabel(f"horizon ({spec.key} candles)")
    save_figure(figure, path)


def plot_weights(
    spec: ScaleSpec,
    models: dict[int, FittedModel],
    path: Path,
) -> None:
    maximum = max(model.trial.lookback for model in models.values())
    matrix = np.full((len(models), maximum), np.nan, dtype=np.float64)
    labels = []
    for row, (horizon, model) in enumerate(sorted(models.items())):
        weights = np.abs(model.coefficients[:model.trial.lookback, -1])
        matrix[row, -weights.size:] = weights
        labels.append(str(horizon))
    finite = matrix[np.isfinite(matrix)]
    ceiling = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    figure, axis = plt.subplots(figsize=(10.0, 3.5), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma",
                        vmin=0.0, vmax=max(ceiling, EPSILON))
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_ylabel(f"forecast horizon ({spec.key} candles)")
    axis.set_xlabel("lag (oldest to most recent)")
    axis.set_title("Endpoint Ridge weight magnitude")
    ticks = np.linspace(0, maximum - 1, min(6, maximum), dtype=int)
    axis.set_xticks(ticks, [str(maximum - tick) for tick in ticks])
    figure.colorbar(image, ax=axis, label="absolute normalized coefficient")
    save_figure(figure, path)


def plot_forecast(
    spec: ScaleSpec,
    horizon: int,
    forecast: dict[str, np.ndarray],
    path: Path,
) -> None:
    context = forecast["context"]
    truth = forecast["truth"]
    tuned = forecast["tuned"]
    global_prediction = forecast["global"]
    origin = float(context[-1])
    context_bps = (context - origin) * 10_000.0
    truth_bps = (truth - origin) * 10_000.0
    tuned_bps = (tuned - origin) * 10_000.0
    global_bps = (global_prediction - origin) * 10_000.0
    x_context = np.arange(-context.size + 1, 1)
    x_future = np.arange(1, horizon + 1)
    figure, axis = plt.subplots(figsize=(9.5, 4.1), constrained_layout=True)
    axis.plot(x_context, context_bps, color="#6b7280", linewidth=1.0, label="context")
    axis.plot(x_future, truth_bps, color="#111827", linewidth=1.6, label="truth")
    axis.plot(x_future, tuned_bps, color="#2563eb", linewidth=1.5, label="tuned Ridge")
    axis.plot(x_future, global_bps, color="#dc2626", linewidth=1.2, label="fixed global Ridge")
    axis.axvline(0, color="#111827", linewidth=0.8, linestyle="--")
    axis.set_xlabel(f"candles relative to forecast origin ({spec.key})")
    axis.set_ylabel("log-price move from origin (bps)")
    axis.set_title(f"Held-out forecast example, H={horizon}")
    axis.legend(frameon=False)
    save_figure(figure, path)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight", metadata={"Software": "searchcast_btc_analysis.py"})
    plt.close(figure)


def write_scale_report(result: dict[str, object], output_root: Path) -> None:
    scale = str(result["scale"])
    slug = artifact_slug(scale)
    profile = result["profile"]
    horizons = result["horizons"]
    power = result["lookbackPowerLaw"]
    limitations = result["limitations"]
    assert isinstance(profile, dict)
    assert isinstance(horizons, list)
    assert isinstance(power, dict)
    assert isinstance(limitations, list)
    one_step = horizons[0]
    best = one_step["bestTrial"]
    gain = float(one_step["models"]["tuned_ridge"].get("gainVsPersistencePercent", 0.0))
    reliability = evidence_label(scale, int(profile["candles"]))

    lines = [
        f"# SearchCast-style BTCUSDT analysis: {scale}",
        "",
        f"> **Result.** At the one-candle horizon, the selected context is "
        f"**{best['lookback']} {scale} candles**, using {best['normalization_scope']} "
        f"{best['normalization_method']} normalization, alpha "
        f"**{format_number(float(best['alpha']))}**, and {best['augmentation']} augmentation. "
        f"Held-out normalized MSE changes by **{gain:+.2f}%** relative to persistence. "
        f"Evidence quality at this scale is **{reliability}**.",
        "",
        "## Scope and adaptation",
        "",
        f"This report applies the transferable analysis from [How Good Can Linear Models Be for "
        f"Time-Series Forecasting?]({PAPER_URL}) to canonical spot BTCUSDT **log close** at "
        f"{scale} resolution. The model is multi-output Ridge regression. Search covers context "
        "length, global versus trailing-window normalization, standard versus robust scaling, "
        "time/frequency/no augmentation, and the paper's 21-value alpha grid. Candidate "
        "preprocessors are scored with chronological expanding-window validation; the final "
        "20% of time is sealed until evaluation.",
        "",
        "BTC close is a single target, so cross-series grouping is not applicable. The paper's "
        "forecast-horizon grouping question is represented here by independent tuning at several "
        "horizon cutoffs and the fitted relation `L* = a H^b`.",
        "",
        "## Dataset statistics",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Candles analyzed | {int(profile['candles']):,} |",
        f"| Period | {profile['start']} to {profile['end']} |",
        f"| Source rows read | {int(profile['sourceRowsRead']):,} / {int(profile['expectedSourceRows']):,} ({float(profile['sourceCoveragePercent']):.2f}%) |",
        f"| Price range | {format_usdt(float(profile['priceMinimum']))} to {format_usdt(float(profile['priceMaximum']))} |",
        f"| Mean / median log return | {float(profile['meanLogReturnBps']):+.4f} / {float(profile['medianLogReturnBps']):+.4f} bps |",
        f"| Return standard deviation | {float(profile['returnStdBps']):.3f} bps |",
        f"| Mean absolute return | {float(profile['meanAbsoluteReturnBps']):.3f} bps |",
        f"| Annualized volatility | {float(profile['annualizedVolatilityPercent']):.2f}% |",
        f"| Skewness / excess kurtosis | {float(profile['skewness']):+.3f} / {float(profile['excessKurtosis']):+.3f} |",
        f"| Mean high-low range | {float(profile['meanCandleRangeBps']):.3f} bps |",
        f"| Median candle volume | {float(profile['medianVolumeBtc']):,.4f} BTC |",
        "",
        str(profile["samplingNote"]),
        "",
        f"![{scale} BTC data profile](charts/{slug}-data-profile.png)",
        "",
        "Return autocorrelation measures linear predictability; absolute-return autocorrelation "
        "measures volatility clustering. The latter can be persistent even when signed returns "
        "are close to serially uncorrelated.",
        "",
        "## Held-out forecasting results",
        "",
        "All MSE values are divided by the development-window log-price variance. RMSE and MAE "
        "are log-price errors in basis points.",
        "",
        "| H (candles) | Test windows | Persistence MSE | Global Ridge MSE | Local-full Ridge MSE | Tuned Ridge MSE | Tuned RMSE | Direction | Gain vs persistence |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in horizons:
        models = item["models"]
        tuned = models["tuned_ridge"]
        lines.append(
            f"| {item['horizonCandles']} | {item['testWindows']} | "
            f"{float(models['persistence']['normalizedMse']):.6g} | "
            f"{float(models['global_ridge']['normalizedMse']):.6g} | "
            f"{float(models['local_full_ridge']['normalizedMse']):.6g} | "
            f"{float(tuned['normalizedMse']):.6g} | "
            f"{float(tuned['rmseBps']):.2f} bps | "
            f"{float(tuned['endpointDirectionAccuracy']):.1%} | "
            f"{float(tuned.get('gainVsPersistencePercent', 0.0)):+.2f}% |"
        )
    lines.extend([
        "",
        f"![{scale} model comparison](charts/{slug}-model-comparison.png)",
        "",
        "A negative gain means tuned Ridge did not beat the no-change forecast on the sealed test "
        "period. Such a result is useful: it bounds how much tradable directional information this "
        "linear close-only setup exposes.",
        "",
        "## Learned dataset-specific parameters",
        "",
        "| H | L | Local/global | Method | r | Effective local points | Alpha | Augmentation | Sigma | CV MSE |",
        "|---:|---:|---|---|---:|---:|---:|---|---:|---:|",
    ])
    for item in horizons:
        trial = item["bestTrial"]
        effective = trial["effectiveLocalPoints"]
        lines.append(
            f"| {item['horizonCandles']} | {trial['lookback']} | "
            f"{trial['normalization_scope']} | {trial['normalization_method']} | "
            f"{float(trial['local_ratio']):.4f} | {effective if effective is not None else 'n/a'} | "
            f"{format_number(float(trial['alpha']))} | {trial['augmentation']} | "
            f"{format_number(float(trial['augmentation_sigma']))} | "
            f"{float(trial['cvNormalizedMse']):.6g} |"
        )
    lines.extend([
        "",
        f"The fitted relation is **L* = {float(power['a']):.2f} H^{float(power['b']):+.3f}** "
        f"with R-squared **{float(power['rSquared']):.3f}**. A positive exponent means longer "
        "forecast horizons selected more context; a negative exponent means old history became "
        "less useful as the target moved farther out.",
        "",
        f"![{scale} lookback versus horizon](charts/{slug}-lookback-horizon.png)",
        "",
        f"![{scale} selected preprocessing parameters](charts/{slug}-hyperparameters.png)",
        "",
        "## What the model uses",
        "",
        f"![{scale} Ridge weight magnitudes](charts/{slug}-weights.png)",
        "",
        "The heatmap shows endpoint-forecast coefficients after preprocessing. Bright recent lags "
        "indicate local momentum/mean-reversion structure; isolated older bands suggest recurring "
        "phase anchors. White/blank left regions are outside the selected context.",
        "",
        f"![{scale} held-out forecast example](charts/{slug}-forecast.png)",
        "",
        "The forecast plot is one deterministic held-out example at the longest evaluated horizon. "
        "It is diagnostic, not a hand-picked claim about average performance; the table above is "
        "the aggregate test result.",
        "",
        "## Implications for later studies",
        "",
    ])
    lines.extend(implication_lines(result))
    lines.extend([
        "",
        "The selected parameters should be treated as search priors, not fixed production truth. "
        "A later model should center its context and normalization search near these values, while "
        "retaining neighboring candidates and walk-forward validation.",
        "",
        "## Limitations",
        "",
    ])
    lines.extend([f"- {value}" for value in limitations])
    lines.extend([
        "",
        "Fees, leverage, slippage, and position transitions are intentionally absent. Forecast "
        "accuracy is not trading profitability, especially at 1s and 1m where errors can be smaller "
        "than execution costs.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        f"npm run analysis:searchcast-btc -- --scales {scale}",
        "```",
        "",
        f"Machine-readable details, all CV folds, and top trials are in "
        f"[`results/{slug}.json`](results/{slug}.json).",
        "",
    ])
    write_text(output_root / f"{slug}.md", "\n".join(lines))


def implication_lines(result: dict[str, object]) -> list[str]:
    horizons = result["horizons"]
    power = result["lookbackPowerLaw"]
    assert isinstance(horizons, list) and isinstance(power, dict)
    trials = [item["bestTrial"] for item in horizons]
    local_count = sum(trial["normalization_scope"] == "local" for trial in trials)
    robust_count = sum(trial["normalization_method"] == "robust" for trial in trials)
    augmented_count = sum(trial["augmentation"] != "none" for trial in trials)
    exponent = float(power["b"])
    lookback_statement = (
        "grows with horizon" if exponent > 0.10 else
        "shrinks with horizon" if exponent < -0.10 else
        "is approximately horizon-invariant"
    )
    project_lookback = result["projectLookback"]
    if project_lookback is None:
        project_comparison = (
            f"- No existing project default is defined at this scale; the one-step search selects "
            f"{trials[0]['lookback']} candles."
        )
    else:
        project_comparison = (
            f"- The one-step selected lookback is {trials[0]['lookback']} candles versus the "
            f"project's current nominal {project_lookback} candles at this scale."
        )
    lines = [
        f"- Selected lookback {lookback_statement} (`b={exponent:+.3f}`).",
        f"- Local normalization wins {local_count}/{len(trials)} horizon cells; robust scaling wins "
        f"{robust_count}/{len(trials)}.",
        f"- Noise augmentation wins {augmented_count}/{len(trials)} horizon cells.",
        project_comparison,
    ]
    boundary_horizons = [
        item["horizonCandles"]
        for item in horizons
        if item["bestTrial"]["lookback"] in (
            min(item["validLookbacks"]), max(item["validLookbacks"])
        )
    ]
    if boundary_horizons:
        lines.append(
            "- The selected lookback touches a searched boundary at H="
            + ", ".join(map(str, boundary_horizons))
            + "; those L values are censored optima, not precise interior estimates."
        )
    return lines


def write_overview(
    results: Sequence[dict[str, object]],
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    if not results:
        raise RuntimeError("no scale results were generated")
    figure, axis = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    scales = [str(item["scale"]) for item in results]
    exponents = [float(item["lookbackPowerLaw"]["b"]) for item in results]  # type: ignore[index]
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in exponents]
    axis.bar(scales, exponents, color=colors)
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_ylabel("power-law exponent b in L* = a H^b")
    axis.set_xlabel("candle scale")
    axis.set_title("How optimal BTC context changes with forecast horizon")
    save_figure(figure, output_root / "charts" / "all-scales-lookback-exponent.png")

    lines = [
        "# SearchCast-style BTCUSDT study across seven candle scales",
        "",
        f"This directory reproduces the transferable parts of [How Good Can Linear Models Be for "
        f"Time-Series Forecasting?]({PAPER_URL}) on this repository's canonical BTCUSDT data. "
        "It contains one document per requested scale, checked-in charts, and JSON with every "
        "selected parameter and held-out metric.",
        "",
        "## Cross-scale result",
        "",
        "| Scale | Candles | One-step L | Project L | Normalization | Alpha | One-step test MSE | Gain vs persistence | L/H exponent b | Evidence |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        profile = result["profile"]
        one_step = result["horizons"][0]
        trial = one_step["bestTrial"]
        model = one_step["models"]["tuned_ridge"]
        evidence = evidence_label(str(result["scale"]), int(profile["candles"]))
        project_lookback = result["projectLookback"]
        project_cell = str(project_lookback) if project_lookback is not None else "n/a"
        lines.append(
            f"| [{result['scale']}]({artifact_slug(str(result['scale']))}.md) | {int(profile['candles']):,} | "
            f"{trial['lookback']} | {project_cell} | "
            f"{trial['normalization_scope']} {trial['normalization_method']} | "
            f"{format_number(float(trial['alpha']))} | {float(model['normalizedMse']):.6g} | "
            f"{float(model.get('gainVsPersistencePercent', 0.0)):+.2f}% | "
            f"{float(result['lookbackPowerLaw']['b']):+.3f} | {evidence} |"
        )
    lines.extend([
        "",
        "## Main findings",
        "",
        *cross_scale_findings(results),
        "",
        "![Cross-scale lookback exponent](charts/all-scales-lookback-exponent.png)",
        "",
        "The exponent is a compact stationarity diagnostic. Positive values mean longer targets "
        "benefit from more history; negative values mean distant history increasingly hurts. "
        "Monthly and quarterly exponents should not drive architecture decisions without a longer "
        "historical corpus.",
        "",
        "## Documents",
        "",
    ])
    lines.extend([
        f"- [{item['scale']} report]({artifact_slug(str(item['scale']))}.md)"
        for item in results
    ])
    lines.extend([
        "",
        "## Method contract",
        "",
        "- Source: immutable canonical spot BTCUSDT candle references.",
        f"- Common interval: {COMMON_START} through {COMMON_END} UTC.",
        "- Target: future log close path.",
        "- Outer split: first 80% development, final 20% held out.",
        "- Inner selection: three chronological expanding folds when sample count allows.",
        "- Ridge alpha: 21 log-spaced values from `1e-6` through `1e3`.",
        f"- Preprocessing trials per horizon: {6 if args.quick else args.trials}.",
        "- 1s fit sampling: one complete day nearest the 15th of every calendar month; no windows "
        "cross sampled-day gaps.",
        "- 1w: complete Monday-Sunday UTC calendar weeks only.",
        "- 1M/3M: complete UTC calendar months/quarters only.",
        "",
        "## Reproduce everything",
        "",
        "```powershell",
        "npm run analysis:searchcast-btc",
        "```",
        "",
        "The consolidated machine-readable artifact is "
        "[`results/all-scales.json`](results/all-scales.json).",
        "",
    ])
    write_text(output_root / "README.md", "\n".join(lines))


def infer_step_ms(timestamps: np.ndarray) -> int:
    differences = np.diff(timestamps)
    positive = differences[differences > 0]
    return int(np.median(positive)) if positive.size else 1


def artifact_slug(scale: str) -> str:
    """Return a case-insensitive-filesystem-safe artifact name."""
    return {
        "1s": "1s",
        "1m": "1min",
        "1h": "1h",
        "1d": "1d",
        "1w": "1w",
        "1M": "1month",
        "3M": "3month",
    }[scale]


def evidence_label(scale: str, candles: int) -> str:
    if candles < 100:
        return "exploratory"
    if scale == "1s":
        return "sampled"
    return "usable"


def cross_scale_findings(results: Sequence[dict[str, object]]) -> list[str]:
    horizons = [
        horizon
        for result in results
        for horizon in result["horizons"]  # type: ignore[index]
    ]
    wins = sum(
        float(horizon["models"]["tuned_ridge"]["gainVsPersistencePercent"]) > 0  # type: ignore[index]
        for horizon in horizons
    )
    augmented = sum(
        horizon["bestTrial"]["augmentation"] != "none"  # type: ignore[index]
        for horizon in horizons
    )
    local = sum(
        horizon["bestTrial"]["normalization_scope"] == "local"  # type: ignore[index]
        for horizon in horizons
    )
    reliable_directions = [
        float(horizon["models"]["tuned_ridge"]["endpointDirectionAccuracy"])  # type: ignore[index]
        for result in results
        if str(result["scale"]) in ("1s", "1m", "1h", "1d", "1w")
        for horizon in result["horizons"]  # type: ignore[index]
    ]
    return [
        f"- Tuned Ridge beats persistence in **{wins}/{len(horizons)}** sealed-test horizon cells. "
        "The no-change forecast is the stronger close-only baseline across this corpus.",
        f"- On 1s through 1w, endpoint direction accuracy spans "
        f"**{min(reliable_directions):.1%} to {max(reliable_directions):.1%}**; no stable "
        "directional edge appears.",
        f"- Local normalization is selected in **{local}/{len(horizons)}** cells and noise "
        f"augmentation in **{augmented}/{len(horizons)}**. BTC log close usually prefers the "
        "simpler global/no-noise path, unlike the paper's benchmark aggregate.",
        "- Minute and second contexts contract sharply beyond the one-step target, while the "
        "one-day target can hit very long contexts. Boundary hits are hypotheses for a wider "
        "search, not proof that maximum history is intrinsically best.",
        "- Monthly and quarterly estimates are too data-limited to set production parameters; "
        "they mainly show that the nominal 16-candle contexts cannot be validated from five years "
        "of local history.",
    ]


def remove_legacy_case_collisions(output_root: Path) -> None:
    """Remove artifacts from the pre-slug layout created by this generator.

    On a case-insensitive filesystem the old ``1m`` and ``1M`` names referred
    to the same files.  The explicit list is intentionally narrow so reruns do
    not delete unrelated documentation.
    """
    legacy = [
        output_root / "1M.md",
        output_root / "3M.md",
        output_root / "results" / "1M.json",
        output_root / "results" / "3M.json",
        *(
            output_root / "charts" / f"{scale}-{suffix}.png"
            for scale in ("1M", "3M")
            for suffix in (
                "data-profile",
                "model-comparison",
                "lookback-horizon",
                "hyperparameters",
                "weights",
                "forecast",
            )
        ),
    ]
    for path in legacy:
        path.unlink(missing_ok=True)


def downsample_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, maximum, dtype=np.int64))


def iso_time(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1_000, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def format_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 10_000:
        return f"{value:.3e}"
    return f"{value:.5g}"


def format_usdt(value: float) -> str:
    return f"${value:,.2f}"


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
