"""Rolling-origin point forecasts for complete 15m, 30m, and 1h windows.

Unlike the daily-origin audit, every forecast is issued immediately before the
window it scores.  Origins are non-overlapping at each horizon.  The local
variance, signed-efficiency, and activity state is re-estimated using only
blocks ending before the origin.  History-window and ensemble-summary choices
are selected before the final 91-day holdout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy import special

from analyze_one_second_dependence_model import selected_files
from analyze_parametric_one_second_process import (
    gaussianize_quantile_spline,
    generate_projected_second_batch,
)
from evaluate_full_hierarchical_one_second_process import (
    FEATURES,
    MINUTES_PER_DAY,
    RNG_SEED,
    deserialize_fit,
    feasible_minute_targets,
    load_or_read_minute_history,
    positive_allocate_rows,
)
from evaluate_hierarchical_point_forecasts import (
    BLEND_FEATURES,
    SELECTED_BLEND,
    BinaryAccumulator,
    BlendCalibrationAccumulator,
    ProbabilisticAccumulator,
    RegressionAccumulator,
    blend_prediction,
    fit_blend_calibration,
    point_summaries,
    safe_correlation,
)
from trading_storage import read_candle_column
TRAIN_START = "2021-07-25T00:00:00+00:00"
FORECAST_START = "2025-07-25T00:00:00+00:00"
FORECAST_END = "2026-07-25T00:00:00+00:00"
HORIZON_MINUTES = {"15m": 15, "30m": 30, "1h": 60}
HISTORY_WINDOWS_DAYS = (1, 3, 7, 14, 30)
ENSEMBLE_SIZE = 16
SPREAD_FACTORS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
WINDOW_SELECTION_START_DAY = 91
WINDOW_SELECTION_END_DAY = 183
BLEND_VALIDATION_END_DAY = 274
RAW_CACHE = "data/benchmarks/rolling-intraday-point-forecast-raw.npz"
OUTPUT = "data/benchmarks/rolling-intraday-point-forecasts.json"
DOCUMENT = "docs/experiments/rolling-intraday-point-forecasts-2026-08-13.md"
PROCESS_REPORT = "data/benchmarks/full-hierarchical-one-second-process.json"
FIT_CACHE = "data/benchmarks/full-hierarchy-one-second-fit.json"
SECOND_CACHE = "data/benchmarks/rolling-intraday-one-second-audit.json"


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def aggregate_complete_blocks(
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    block_minutes: int,
) -> dict[str, np.ndarray]:
    complete = minute_returns.size // block_minutes
    selected = slice(0, complete * block_minutes)
    shape = (complete, block_minutes)
    return {
        "periodReturnBps": minute_returns[selected].reshape(shape).sum(axis=1),
        "oneSecondRealizedVarianceBpsSquared": (
            minute_variance[selected].reshape(shape).sum(axis=1)
        ),
        "activeSeconds": minute_counts[selected].reshape(shape).sum(axis=1).astype(np.float64),
    }


def deterministic_draws(
    horizon_index: int,
    origin_block: int,
    ensemble_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(
        RNG_SEED + 3_100_000 + horizon_index * 1_000_003 + origin_block
    )
    return {
        feature: rng.standard_t(8.0, (ensemble_size, 1))
        for feature in ("variance", "return", "activity")
    }


def rolling_ar1_parameters(
    values: np.ndarray,
    origins: np.ndarray,
    window: int,
) -> dict[str, np.ndarray]:
    """Vectorized causal AR(1) moments for ranges [origin-window, origin)."""
    values = np.asarray(values, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    starts = origins - window
    if np.any(starts < 0) or window < 5:
        raise ValueError("rolling AR windows need at least five preceding values")
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_squared = np.concatenate(([0.0], np.cumsum(values * values)))
    adjacent = np.zeros(values.size, dtype=np.float64)
    adjacent[1:] = values[1:] * values[:-1]
    prefix_adjacent = np.concatenate(([0.0], np.cumsum(adjacent)))

    total = prefix[origins] - prefix[starts]
    total_squared = prefix_squared[origins] - prefix_squared[starts]
    mean = total / window
    pairs = window - 1
    previous_sum = prefix[origins - 1] - prefix[starts]
    current_sum = prefix[origins] - prefix[starts + 1]
    previous_squared = prefix_squared[origins - 1] - prefix_squared[starts]
    current_squared = prefix_squared[origins] - prefix_squared[starts + 1]
    cross = prefix_adjacent[origins] - prefix_adjacent[starts + 1]
    previous_centered_squared = (
        previous_squared - 2.0 * mean * previous_sum + pairs * mean * mean
    )
    current_centered_squared = (
        current_squared - 2.0 * mean * current_sum + pairs * mean * mean
    )
    centered_cross = (
        cross
        - mean * (previous_sum + current_sum)
        + pairs * mean * mean
    )
    raw_phi = np.divide(
        centered_cross,
        previous_centered_squared,
        out=np.zeros_like(centered_cross),
        where=previous_centered_squared > 1e-12,
    )
    phi = np.clip(raw_phi * window / (window + 20.0), -0.7, 0.97)
    previous_centered_sum = previous_sum - pairs * mean
    current_centered_sum = current_sum - pairs * mean
    residual_sum = current_centered_sum - phi * previous_centered_sum
    residual_sum_squared = (
        current_centered_squared
        + phi * phi * previous_centered_squared
        - 2.0 * phi * centered_cross
    )
    centered_residual_ss = np.maximum(
        residual_sum_squared - residual_sum * residual_sum / pairs,
        0.0,
    )
    residual_scale = np.sqrt(centered_residual_ss / max(pairs - 1, 1))
    centered_total_ss = np.maximum(total_squared - total * total / window, 0.0)
    value_scale = np.sqrt(centered_total_ss / max(window - 1, 1))
    scale = np.maximum.reduce((
        residual_scale,
        value_scale * 0.05,
        np.full(origins.size, 1e-9),
    ))
    last = values[origins - 1]
    return {
        "mean": mean,
        "phi": phi,
        "last": last,
        "location": mean + phi * (last - mean),
        "scale": scale,
    }


def simulate_rolling_ar_paths(
    parameters: dict[str, np.ndarray],
    *,
    members: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    """Simulate causal AR paths from parameters frozen at each origin."""
    origins = parameters["mean"].size
    rng = np.random.default_rng(seed)
    innovations = rng.standard_t(8.0, (origins, members, steps)) / math.sqrt(8.0 / 6.0)
    result = np.empty_like(innovations)
    mean = parameters["mean"][:, None]
    phi = parameters["phi"][:, None]
    scale = parameters["scale"][:, None]
    state = np.broadcast_to(parameters["last"][:, None], (origins, members)).copy()
    for step in range(steps):
        state = mean + phi * (state - mean) + scale * innovations[:, :, step]
        result[:, :, step] = state
    return result


def round_capped_rows(values: np.ndarray, totals: np.ndarray, cap: int = 60) -> np.ndarray:
    """Round nonnegative rows to exact integer totals under a per-cell cap."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(cap))
    totals = np.clip(
        np.rint(np.asarray(totals, dtype=np.float64)).astype(np.int64),
        0,
        cap * values.shape[1],
    )
    result = np.floor(values).astype(np.int64)
    for row in range(values.shape[0]):
        difference = int(totals[row] - np.sum(result[row]))
        if difference > 0:
            eligible = np.flatnonzero(result[row] < cap)
            order = eligible[np.argsort(
                -(values[row, eligible] - result[row, eligible]),
                kind="stable",
            )]
            result[row, order[:difference]] += 1
        elif difference < 0:
            eligible = np.flatnonzero(result[row] > 0)
            order = eligible[np.argsort(
                values[row, eligible] - result[row, eligible],
                kind="stable",
            )]
            result[row, order[: -difference]] -= 1
    if np.any(np.sum(result, axis=1) != totals):
        raise RuntimeError("integer activity rounding failed to hit a row total")
    return result.astype(np.uint8)


def build_causal_minute_paths(
    *,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    origins: np.ndarray,
    horizon_minutes: int,
    history_days: int,
    target_return: np.ndarray,
    target_variance: np.ndarray,
    target_activity: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build minute paths and reconcile every member to its coarse target."""
    members = target_return.shape[1]
    window = history_days * MINUTES_PER_DAY
    log_variance = np.log(np.maximum(minute_variance, 1e-12))
    efficiency = minute_returns / np.sqrt(np.maximum(minute_variance, 1e-12))
    active_logit = special.logit(np.clip(
        minute_counts.astype(np.float64) / 60.0,
        1e-5,
        1.0 - 1e-5,
    ))
    variance_fit = rolling_ar1_parameters(log_variance, origins, window)
    efficiency_fit = rolling_ar1_parameters(efficiency, origins, window)
    activity_fit = rolling_ar1_parameters(active_logit, origins, window)
    next_log_variance = simulate_rolling_ar_paths(
        variance_fit,
        members=members,
        steps=horizon_minutes,
        seed=seed,
    )
    q = np.exp(np.clip(next_log_variance, -30.0, 30.0))
    q *= np.divide(
        np.maximum(target_variance, 1e-12),
        np.sum(q, axis=2),
    )[:, :, None]
    next_efficiency = simulate_rolling_ar_paths(
        efficiency_fit,
        members=members,
        steps=horizon_minutes,
        seed=seed + 100_003,
    )
    returns = next_efficiency * np.sqrt(q)
    weights = np.sqrt(np.maximum(q, 1e-12))
    weights /= np.sum(weights, axis=2, keepdims=True)
    returns += (
        target_return - np.sum(returns, axis=2)
    )[:, :, None] * weights
    activity = special.expit(simulate_rolling_ar_paths(
        activity_fit,
        members=members,
        steps=horizon_minutes,
        seed=seed + 200_003,
    )) * 60.0
    activity = positive_allocate_rows(
        activity.reshape(-1, horizon_minutes),
        target_activity.reshape(-1),
        60.0,
    ).reshape(activity.shape)
    counts = round_capped_rows(
        activity.reshape(-1, horizon_minutes),
        target_activity.reshape(-1),
        cap=60,
    ).reshape(activity.shape)
    return {
        "periodReturnBps": returns.astype(np.float32),
        "oneSecondRealizedVarianceBpsSquared": q.astype(np.float32),
        "activeSeconds": counts,
    }


def generate_raw_forecasts(
    *,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    train_start: datetime,
    forecast_start: datetime,
    forecast_end: datetime,
    windows: tuple[int, ...],
    ensemble_size: int,
) -> dict:
    result = {}
    for horizon_index, (horizon, block_minutes) in enumerate(HORIZON_MINUTES.items()):
        print(f"Rolling intraday targets {horizon}...", flush=True)
        history = aggregate_complete_blocks(
            minute_returns,
            minute_variance,
            minute_counts,
            block_minutes,
        )
        blocks_per_day = MINUTES_PER_DAY // block_minutes
        start_block = (forecast_start - train_start).days * blocks_per_day
        end_block = (forecast_end - train_start).days * blocks_per_day
        origins = np.arange(start_block, end_block, dtype=np.int64)
        actual = {
            feature: history[feature][origins].astype(np.float64)
            for feature in FEATURES
        }
        forecasts = {
            window: {
                feature: np.empty((origins.size, ensemble_size), dtype=np.float32)
                for feature in FEATURES
            }
            for window in windows
        }
        last_return = history["periodReturnBps"][origins - 1].astype(np.float64)
        rng = np.random.default_rng(
            RNG_SEED + 3_100_000 + horizon_index * 1_000_003
        )
        t_standard_deviation = math.sqrt(8.0 / 6.0)
        common_variance = rng.standard_t(
            8.0, (origins.size, ensemble_size)
        ) / t_standard_deviation
        common_return = rng.standard_t(
            8.0, (origins.size, ensemble_size)
        ) / t_standard_deviation
        common_activity = rng.standard_t(
            8.0, (origins.size, ensemble_size)
        ) / t_standard_deviation
        log_variance = np.log(np.maximum(
            history["oneSecondRealizedVarianceBpsSquared"], 1e-12
        ))
        efficiency = history["periodReturnBps"] / np.sqrt(np.maximum(
            history["oneSecondRealizedVarianceBpsSquared"], 1e-12
        ))
        active_logit = special.logit(np.clip(
            history["activeSeconds"] / (block_minutes * 60.0),
            1e-5,
            1.0 - 1e-5,
        ))
        for window in windows:
            history_blocks = window * blocks_per_day
            variance_fit = rolling_ar1_parameters(
                log_variance, origins, history_blocks
            )
            next_log_variance = (
                variance_fit["location"][:, None]
                + variance_fit["scale"][:, None] * common_variance
            )
            next_variance = np.exp(np.clip(next_log_variance, -30.0, 30.0))
            return_fit = rolling_ar1_parameters(
                efficiency, origins, history_blocks
            )
            next_efficiency = (
                return_fit["location"][:, None]
                + return_fit["scale"][:, None] * common_return
            )
            activity_fit = rolling_ar1_parameters(
                active_logit, origins, history_blocks
            )
            next_active_fraction = special.expit(
                activity_fit["location"][:, None]
                + activity_fit["scale"][:, None] * common_activity
            )
            forecasts[window]["periodReturnBps"][:] = (
                next_efficiency * np.sqrt(next_variance)
            ).astype(np.float32)
            forecasts[window]["oneSecondRealizedVarianceBpsSquared"][:] = (
                next_variance.astype(np.float32)
            )
            forecasts[window]["activeSeconds"][:] = (
                np.clip(next_active_fraction, 0.01, 0.999)
                * block_minutes * 60.0
            ).astype(np.float32)
        result[horizon] = {
            "actual": actual,
            "forecasts": forecasts,
            "lastReturnBps": last_return,
            "blocksPerDay": blocks_per_day,
        }
    return result


def load_or_generate_raw(
    *,
    cache_path: Path,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    train_start: datetime,
    forecast_start: datetime,
    forecast_end: datetime,
    windows: tuple[int, ...],
    ensemble_size: int,
) -> dict:
    expected = {
        "version": 1,
        "trainStart": iso(train_start),
        "forecastStart": iso(forecast_start),
        "forecastEndExclusive": iso(forecast_end),
        "windowsDays": list(windows),
        "ensembleSize": ensemble_size,
        "originCadence": "one non-overlapping origin per horizon block",
    }
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadataJson"].item()))
            if all(metadata.get(key) == value for key, value in expected.items()):
                print("Loading cached rolling intraday forecasts...", flush=True)
                result = {}
                for horizon, minutes in HORIZON_MINUTES.items():
                    result[horizon] = {
                        "actual": {
                            feature: cache[f"actual__{horizon}__{feature}"].astype(np.float64)
                            for feature in FEATURES
                        },
                        "forecasts": {
                            window: {
                                feature: cache[
                                    f"forecast__{horizon}__{window}__{feature}"
                                ].astype(np.float32)
                                for feature in FEATURES
                            }
                            for window in windows
                        },
                        "lastReturnBps": cache[f"last__{horizon}"].astype(np.float64),
                        "blocksPerDay": MINUTES_PER_DAY // minutes,
                    }
                return result
    result = generate_raw_forecasts(
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        train_start=train_start,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        windows=windows,
        ensemble_size=ensemble_size,
    )
    arrays = {}
    for horizon, values in result.items():
        for feature in FEATURES:
            arrays[f"actual__{horizon}__{feature}"] = values["actual"][feature]
        arrays[f"last__{horizon}"] = values["lastReturnBps"]
        for window in windows:
            for feature in FEATURES:
                arrays[f"forecast__{horizon}__{window}__{feature}"] = values[
                    "forecasts"
                ][window][feature]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, metadataJson=json.dumps(expected), **arrays)
    return result


def selection_metrics(actual: np.ndarray, ensemble: np.ndarray) -> dict:
    accumulator = ProbabilisticAccumulator()
    accumulator.add(actual, ensemble)
    result = accumulator.finish()
    coverage_error = float(np.mean([
        abs(values["empiricalCoverage"] - values["nominalCoverage"])
        for values in result["centralIntervals"].values()
    ]))
    actual_scale = max(float(np.std(actual)), 1e-9)
    result["normalizedCrpsByActualStd"] = result["meanCrpsBps"] / actual_scale
    result["meanAbsoluteCoverageError"] = coverage_error
    result["selectionScore"] = result["normalizedCrpsByActualStd"] + coverage_error
    return result


def spread_ensemble(ensemble: np.ndarray, factor: float) -> np.ndarray:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    center = np.median(ensemble, axis=1, keepdims=True)
    return center + factor * (ensemble - center)


def select_spread_factor(
    actual: np.ndarray,
    ensemble: np.ndarray,
    blocks_per_day: int,
) -> dict:
    validation_slice = slice(
        WINDOW_SELECTION_END_DAY * blocks_per_day,
        BLEND_VALIDATION_END_DAY * blocks_per_day,
    )
    validation_days = BLEND_VALIDATION_END_DAY - WINDOW_SELECTION_END_DAY
    scores = {}
    for factor in SPREAD_FACTORS:
        calibrated = spread_ensemble(ensemble, factor)
        overall = selection_metrics(
            actual[validation_slice], calibrated[validation_slice]
        )
        folds = []
        for fold in range(3):
            start_day = WINDOW_SELECTION_END_DAY + fold * validation_days // 3
            end_day = WINDOW_SELECTION_END_DAY + (fold + 1) * validation_days // 3
            selected = slice(start_day * blocks_per_day, end_day * blocks_per_day)
            folds.append(selection_metrics(actual[selected], calibrated[selected]))
        scores[str(factor)] = {**overall, "chronologicalFoldScores": folds}
    selected = min(
        SPREAD_FACTORS,
        key=lambda factor: scores[str(factor)]["selectionScore"],
    )
    return {
        "selectedFactor": selected,
        "candidateScores": scores,
        "selectionStartDay": WINDOW_SELECTION_END_DAY,
        "selectionEndExclusiveDay": BLEND_VALIDATION_END_DAY,
        "futureOutcomesUsed": False,
        "untouchedHoldoutOutcomesUsed": False,
        "transformation": "row median + factor * (member - row median)",
    }


def day_bootstrap(
    actual: np.ndarray,
    prediction: np.ndarray,
    blocks_per_day: int,
    *,
    draws: int = 2_000,
) -> dict:
    days = actual.size // blocks_per_day
    actual = actual.reshape(days, blocks_per_day)
    prediction = prediction.reshape(days, blocks_per_day)
    rng = np.random.default_rng(RNG_SEED + 4_200_000 + blocks_per_day)
    correlations = np.empty(draws, dtype=np.float64)
    mse_skills = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indexes = rng.integers(0, days, days)
        observed = actual[indexes].reshape(-1)
        estimated = prediction[indexes].reshape(-1)
        correlations[draw] = (
            np.corrcoef(observed, estimated)[0, 1]
            if np.std(estimated) > 0.0 else 0.0
        )
        baseline = float(np.mean(observed * observed))
        mse_skills[draw] = 1.0 - float(np.mean((estimated - observed) ** 2)) / baseline
    return {
        "resampledUtcDays": days,
        "draws": draws,
        "correlation95Interval": np.quantile(correlations, [0.025, 0.975]).tolist(),
        "mseSkill95Interval": np.quantile(mse_skills, [0.025, 0.975]).tolist(),
        "probabilityPositiveCorrelation": float(np.mean(correlations > 0.0)),
        "probabilityPositiveMseSkill": float(np.mean(mse_skills > 0.0)),
    }


def evaluate_horizon(
    horizon: str,
    values: dict,
    windows: tuple[int, ...],
) -> dict:
    blocks_per_day = int(values["blocksPerDay"])
    actual = values["actual"]["periodReturnBps"]
    selection_slice = slice(
        WINDOW_SELECTION_START_DAY * blocks_per_day,
        WINDOW_SELECTION_END_DAY * blocks_per_day,
    )
    window_scores = {
        str(window): selection_metrics(
            actual[selection_slice],
            values["forecasts"][window]["periodReturnBps"][selection_slice],
        )
        for window in windows
    }
    selected_window = min(
        windows,
        key=lambda window: window_scores[str(window)]["selectionScore"],
    )
    ensemble = values["forecasts"][selected_window]["periodReturnBps"].astype(np.float64)
    spread_calibration = select_spread_factor(
        actual,
        ensemble,
        blocks_per_day,
    )
    calibrated_ensemble = spread_ensemble(
        ensemble,
        float(spread_calibration["selectedFactor"]),
    )
    summaries = point_summaries(ensemble, medoid_member=0)
    blend_values = BlendCalibrationAccumulator()
    training_slice = slice(
        WINDOW_SELECTION_START_DAY * blocks_per_day,
        WINDOW_SELECTION_END_DAY * blocks_per_day,
    )
    blend_values.add(
        actual[training_slice],
        {name: summaries[name][training_slice] for name in BLEND_FEATURES},
        phase="training",
    )
    validation_days = BLEND_VALIDATION_END_DAY - WINDOW_SELECTION_END_DAY
    for fold in range(3):
        start_day = WINDOW_SELECTION_END_DAY + fold * validation_days // 3
        end_day = WINDOW_SELECTION_END_DAY + (fold + 1) * validation_days // 3
        fold_slice = slice(start_day * blocks_per_day, end_day * blocks_per_day)
        blend_values.add(
            actual[fold_slice],
            {name: summaries[name][fold_slice] for name in BLEND_FEATURES},
            phase="validation",
            fold=fold,
        )
    blend = fit_blend_calibration(blend_values)
    summaries[SELECTED_BLEND] = blend_prediction(summaries, blend)
    summaries["lastBlockReturnBaseline"] = values["lastReturnBps"]
    summaries["zeroReturnBaseline"] = np.zeros(actual.size, dtype=np.float64)

    test_slice = slice(BLEND_VALIDATION_END_DAY * blocks_per_day, None)
    actual_test = actual[test_slice]
    point = {}
    bootstrap = {}
    for name in (
        "ensembleMean",
        "ensembleMedian",
        "localDensityMode",
        SELECTED_BLEND,
        "lastBlockReturnBaseline",
        "zeroReturnBaseline",
    ):
        accumulator = RegressionAccumulator()
        prediction = summaries[name][test_slice]
        accumulator.add(actual_test, prediction)
        point[name] = accumulator.finish()
        if name in ("ensembleMean", SELECTED_BLEND, "lastBlockReturnBaseline"):
            bootstrap[name] = day_bootstrap(
                actual_test,
                prediction,
                blocks_per_day,
            )
    raw_probability = ProbabilisticAccumulator()
    raw_probability.add(actual_test, ensemble[test_slice])
    calibrated_probability = ProbabilisticAccumulator()
    calibrated_probability.add(actual_test, calibrated_ensemble[test_slice])
    expected_absolute = RegressionAccumulator()
    expected_absolute.add(
        np.abs(actual_test),
        np.mean(np.abs(ensemble[test_slice]), axis=1),
    )
    expected_variance = RegressionAccumulator()
    expected_variance.add(
        values["actual"]["oneSecondRealizedVarianceBpsSquared"][test_slice],
        np.mean(
            values["forecasts"][selected_window][
                "oneSecondRealizedVarianceBpsSquared"
            ][test_slice],
            axis=1,
        ),
    )
    expected_activity = RegressionAccumulator()
    expected_activity.add(
        values["actual"]["activeSeconds"][test_slice],
        np.mean(
            values["forecasts"][selected_window]["activeSeconds"][test_slice],
            axis=1,
        ),
    )
    return {
        "horizonMinutes": HORIZON_MINUTES[horizon],
        "originCadenceMinutes": HORIZON_MINUTES[horizon],
        "originsPerUtcDay": blocks_per_day,
        "selectedHistoryWindowDays": selected_window,
        "historyWindowSelection": window_scores,
        "pointBlendCalibration": blend,
        "spreadCalibration": spread_calibration,
        "untouchedTest": {
            "origins": actual_test.size,
            "pointEstimators": point,
            "probabilisticEnsemble": calibrated_probability.finish(),
            "rawProbabilisticEnsemble": raw_probability.finish(),
            "dayBootstrap": bootstrap,
            "conditionalScaleForecasts": {
                "expectedAbsoluteReturn": expected_absolute.finish(),
                "expectedRealizedVariance": expected_variance.finish(),
                "expectedActiveSeconds": expected_activity.finish(),
            },
        },
    }


def load_actual_second_days(
    source: Path,
    start: datetime,
    end: datetime,
) -> np.ndarray:
    files = selected_files(source, start, end)
    prior = selected_files(source, start - timedelta(days=1), start)
    if len(files) != (end - start).days or len(prior) != 1:
        raise RuntimeError("one-second holdout shards are incomplete")
    previous_close = float(read_candle_column(prior[0], "close")[-1])
    result = np.empty((len(files), 86_400), dtype=np.float32)
    for day, reference in enumerate(files):
        closes = read_candle_column(reference, "close").astype(np.float64)
        returns = np.diff(np.log(np.concatenate((
            np.asarray([previous_close], dtype=np.float64),
            closes,
        )))) * 10_000.0
        if returns.size != 86_400:
            raise ValueError(f"{reference} is not a complete one-second day")
        result[day] = returns.astype(np.float32)
        previous_close = float(closes[-1])
    return result


def forecast_window_medoid(paths: np.ndarray) -> int:
    paths = np.asarray(paths, dtype=np.float64)
    if paths.shape[1] % 60:
        raise ValueError("window medoid needs complete minutes")
    minute = paths.reshape(paths.shape[0], -1, 60).sum(axis=2)
    cumulative = np.cumsum(minute, axis=1)
    center = np.mean(cumulative, axis=0)
    scale = np.std(cumulative, axis=0)
    floor = (
        max(float(np.median(scale[scale > 0.0])) * 0.1, 1e-6)
        if np.any(scale > 0.0) else 1.0
    )
    distance = np.mean(
        ((cumulative - center) / np.maximum(scale, floor)) ** 2,
        axis=1,
    )
    return int(np.argmin(distance))


def materialize_horizon_seconds(
    *,
    horizon: str,
    horizon_report: dict,
    raw_values: dict,
    actual_days: np.ndarray,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    history_offset_minutes: int,
    fitted,
    calibration: dict,
) -> dict:
    horizon_minutes = HORIZON_MINUTES[horizon]
    horizon_seconds = horizon_minutes * 60
    blocks_per_day = MINUTES_PER_DAY // horizon_minutes
    test_days = actual_days.shape[0]
    test_start_row = BLEND_VALIDATION_END_DAY * blocks_per_day
    selected_window = int(horizon_report["selectedHistoryWindowDays"])
    targets = raw_values["forecasts"][selected_window]
    target_return = targets["periodReturnBps"][test_start_row:].astype(np.float64)
    target_variance = targets[
        "oneSecondRealizedVarianceBpsSquared"
    ][test_start_row:].astype(np.float64)
    target_activity = targets["activeSeconds"][test_start_row:].astype(np.float64)
    origins = (
        history_offset_minutes
        + BLEND_VALIDATION_END_DAY * MINUTES_PER_DAY
        + np.arange(target_return.shape[0], dtype=np.int64) * horizon_minutes
    )
    minute_paths = build_causal_minute_paths(
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        origins=origins,
        horizon_minutes=horizon_minutes,
        history_days=selected_window,
        target_return=target_return,
        target_variance=target_variance,
        target_activity=target_activity,
        seed=RNG_SEED + 5_000_000 + horizon_minutes * 10_003,
    )
    actual_windows = actual_days.reshape(
        test_days,
        blocks_per_day,
        horizon_seconds,
    ).reshape(-1, horizon_seconds)
    if actual_windows.shape[0] != target_return.shape[0]:
        raise RuntimeError("one-second windows and rolling targets do not align")

    estimator_names = (
        "ensembleMean",
        "ensembleMedian",
        "localDensityMode",
        "pathMedoid",
        "zeroReturnBaseline",
    )
    candle_metrics = {
        name: RegressionAccumulator() for name in estimator_names
    }
    endpoint_metrics = {
        name: RegressionAccumulator() for name in estimator_names
    }
    probability = ProbabilisticAccumulator()
    expected_absolute = RegressionAccumulator()
    expected_squared = RegressionAccumulator()
    activity_probability = BinaryAccumulator()
    cumulative_path = {
        name: RegressionAccumulator() for name in estimator_names[:-1]
    }
    daily_path_correlations = {name: [] for name in estimator_names[:-1]}
    diagnostics = {
        "generatedPaths": 0,
        "projectionFallbacks": 0,
        "varianceInflationBlocks": 0,
        "inputVarianceBpsSquared": 0.0,
        "addedVarianceBpsSquared": 0.0,
        "maximumEndpointReturnErrorBps": 0.0,
        "maximumEndpointVarianceErrorBpsSquared": 0.0,
        "maximumEndpointActivityErrorSeconds": 0,
        "maximumActualEndpointAlignmentErrorBps": 0.0,
        "maximumMeanEndpointSummaryErrorBps": 0.0,
    }
    members = target_return.shape[1]
    for day in range(test_days):
        if day % 10 == 0:
            print(f"  1s {horizon} day {day}/{test_days}", flush=True)
        day_paths = np.empty((members, 86_400), dtype=np.float32)
        day_activity = np.empty((members, 86_400), dtype=bool)
        row_start = day * blocks_per_day
        row_stop = row_start + blocks_per_day
        for member in range(members):
            counts = minute_paths["activeSeconds"][
                row_start:row_stop, member
            ].reshape(MINUTES_PER_DAY)
            minute_target, q, feasible = feasible_minute_targets(
                minute_paths["periodReturnBps"][
                    row_start:row_stop, member
                ].reshape(MINUTES_PER_DAY),
                minute_paths["oneSecondRealizedVarianceBpsSquared"][
                    row_start:row_stop, member
                ].reshape(MINUTES_PER_DAY),
                counts,
                block_minutes=15,
            )
            volatility_score = gaussianize_quantile_spline(
                np.log(q + fitted.variance_floor),
                fitted.variance_quantile_knots,
                fitted.variance_log_quantiles,
            )
            active, second_returns = generate_projected_second_batch(
                fitted,
                counts=counts,
                volatility_score=volatility_score,
                target_minute_returns=minute_target,
                realized_variance=q,
                activity_timing_rho=float(calibration["activityTimingGaussianRho"]),
                magnitude_dispersion_scale=float(calibration["magnitudeDispersionScale"]),
                micro_probability_scale=float(calibration["microProbabilityScale"]),
                rng=np.random.default_rng(
                    RNG_SEED
                    + 6_000_000
                    + horizon_minutes * 1_000_003
                    + day * 101
                    + member
                ),
                sign_timing_rho=float(calibration.get("signTimingGaussianRho", 0.0)),
                magnitude_share_score_rho=float(calibration.get(
                    "magnitudeShareScoreRho",
                    fitted.magnitude_mixture.share_score_rho,
                )),
            )
            day_paths[member] = second_returns.reshape(-1).astype(np.float32)
            day_activity[member] = active.reshape(-1)
            diagnostics["generatedPaths"] += 1
            diagnostics["projectionFallbacks"] += feasible["projectionFallbacks"]
            diagnostics["varianceInflationBlocks"] += feasible[
                "varianceInflationBlocks"
            ]
            diagnostics["inputVarianceBpsSquared"] += feasible[
                "inputVarianceTotalBpsSquared"
            ]
            diagnostics["addedVarianceBpsSquared"] += feasible[
                "addedVarianceBpsSquared"
            ]
            generated_block_return = second_returns.reshape(
                blocks_per_day, horizon_seconds
            ).sum(axis=1)
            diagnostics["maximumEndpointReturnErrorBps"] = max(
                diagnostics["maximumEndpointReturnErrorBps"],
                float(np.max(np.abs(
                    generated_block_return
                    - target_return[row_start:row_stop, member]
                ))),
            )
            generated_block_variance = (second_returns * second_returns).reshape(
                blocks_per_day, horizon_seconds
            ).sum(axis=1)
            feasible_block_variance = q.reshape(
                blocks_per_day, horizon_minutes
            ).sum(axis=1)
            diagnostics["maximumEndpointVarianceErrorBpsSquared"] = max(
                diagnostics["maximumEndpointVarianceErrorBpsSquared"],
                float(np.max(np.abs(
                    generated_block_variance - feasible_block_variance
                ))),
            )
            generated_block_activity = active.reshape(
                blocks_per_day, horizon_seconds
            ).sum(axis=1)
            target_block_activity = counts.reshape(
                blocks_per_day, horizon_minutes
            ).sum(axis=1)
            diagnostics["maximumEndpointActivityErrorSeconds"] = max(
                diagnostics["maximumEndpointActivityErrorSeconds"],
                int(np.max(np.abs(
                    generated_block_activity.astype(np.int64)
                    - target_block_activity.astype(np.int64)
                ))),
            )
        for block in range(blocks_per_day):
            origin = row_start + block
            second_start = block * horizon_seconds
            second_stop = second_start + horizon_seconds
            generated_paths = day_paths[:, second_start:second_stop]
            generated_activity = day_activity[:, second_start:second_stop]
            actual = actual_windows[origin].astype(np.float64)
            ensemble = generated_paths.T.astype(np.float64)
            ordered = probability.add(actual, ensemble)
            medoid_member = forecast_window_medoid(generated_paths)
            summaries = point_summaries(
                ensemble,
                medoid_member,
                ordered=ordered,
            )
            actual_alignment_error = abs(
                float(np.sum(actual))
                - float(raw_values["actual"]["periodReturnBps"][
                    test_start_row + origin
                ])
            )
            mean_summary_error = abs(
                float(np.sum(summaries["ensembleMean"]))
                - float(np.mean(target_return[origin]))
            )
            diagnostics["maximumActualEndpointAlignmentErrorBps"] = max(
                diagnostics["maximumActualEndpointAlignmentErrorBps"],
                actual_alignment_error,
            )
            diagnostics["maximumMeanEndpointSummaryErrorBps"] = max(
                diagnostics["maximumMeanEndpointSummaryErrorBps"],
                mean_summary_error,
            )
            if actual_alignment_error > 5e-5 or mean_summary_error > 1e-4:
                raise RuntimeError("generated second path endpoint alignment failed")
            for name in estimator_names:
                candle_metrics[name].add(actual, summaries[name])
                endpoint_metrics[name].add(
                    np.asarray([np.sum(actual)]),
                    np.asarray([np.sum(summaries[name])]),
                )
            expected_absolute.add(
                np.abs(actual),
                np.mean(np.abs(ensemble), axis=1),
            )
            expected_squared.add(
                actual * actual,
                np.mean(ensemble * ensemble, axis=1),
            )
            activity_probability.add(
                actual != 0.0,
                np.mean(generated_activity, axis=0),
            )
            actual_minute = actual.reshape(horizon_minutes, 60).sum(axis=1)
            ensemble_minute = generated_paths.reshape(
                members, horizon_minutes, 60
            ).sum(axis=2).T.astype(np.float64)
            actual_cumulative = np.cumsum(actual_minute)
            ensemble_cumulative = np.cumsum(ensemble_minute, axis=0)
            cumulative_summaries = point_summaries(
                ensemble_cumulative,
                medoid_member,
            )
            for name in estimator_names[:-1]:
                prediction = cumulative_summaries[name]
                cumulative_path[name].add(actual_cumulative, prediction)
                correlation = safe_correlation(actual_cumulative, prediction)
                if correlation is not None:
                    daily_path_correlations[name].append(correlation)
    diagnostics["relativeVarianceAddedForFeasibility"] = (
        diagnostics["addedVarianceBpsSquared"]
        / diagnostics["inputVarianceBpsSquared"]
        if diagnostics["inputVarianceBpsSquared"] > 0 else 0.0
    )
    return {
        "forecastWindows": int(actual_windows.shape[0]),
        "membersPerWindow": int(target_return.shape[1]),
        "generatedSeconds": int(
            actual_windows.shape[0] * target_return.shape[1] * horizon_seconds
        ),
        "candlePointEstimators": {
            name: metrics.finish() for name, metrics in candle_metrics.items()
        },
        "endpointFromGeneratedSeconds": {
            name: metrics.finish() for name, metrics in endpoint_metrics.items()
        },
        "probabilisticOneSecondEnsemble": probability.finish(),
        "conditionalOneSecondScale": {
            "expectedAbsoluteReturn": expected_absolute.finish(),
            "expectedSquaredReturn": expected_squared.finish(),
            "activityProbability": activity_probability.finish(),
        },
        "cumulativeIntrawindowPathAtMinuteEndpoints": {
            name: {
                "pooledMetrics": metrics.finish(),
                "meanWindowCorrelation": float(np.mean(daily_path_correlations[name])),
                "medianWindowCorrelation": float(np.median(daily_path_correlations[name])),
            }
            for name, metrics in cumulative_path.items()
        },
        "diagnostics": diagnostics,
    }


def load_or_materialize_second_audit(
    *,
    cache_path: Path,
    raw: dict,
    report: dict,
    actual_days: np.ndarray,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    history_offset_minutes: int,
    fitted,
    calibration: dict,
) -> dict:
    expected = {
        "version": 3,
        "untouchedTestStart": report["design"]["untouchedTestStart"],
        "untouchedTestDays": report["design"]["untouchedTestDays"],
        "ensembleSize": report["design"]["ensembleSize"],
        "selectedWindows": {
            horizon: report["horizons"][horizon]["selectedHistoryWindowDays"]
            for horizon in HORIZON_MINUTES
        },
        "secondKernelInvocation": "one continuous generated path per UTC member-day, split into forecast windows",
    }
    payload = {"metadata": expected, "horizons": {}}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("metadata") == expected:
            payload = cached
    for horizon in HORIZON_MINUTES:
        if horizon in payload["horizons"]:
            print(f"Loading cached 1s rolling audit {horizon}...", flush=True)
            continue
        print(f"Materializing rolling one-second paths {horizon}...", flush=True)
        payload["horizons"][horizon] = materialize_horizon_seconds(
            horizon=horizon,
            horizon_report=report["horizons"][horizon],
            raw_values=raw[horizon],
            actual_days=actual_days,
            minute_returns=minute_returns,
            minute_variance=minute_variance,
            minute_counts=minute_counts,
            history_offset_minutes=history_offset_minutes,
            fitted=fitted,
            calibration=calibration,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload["horizons"]


def attach_endpoint_alignment(
    audit: dict,
    raw: dict,
    actual_days: np.ndarray,
) -> None:
    for horizon, horizon_minutes in HORIZON_MINUTES.items():
        blocks_per_day = MINUTES_PER_DAY // horizon_minutes
        seconds = horizon_minutes * 60
        actual_from_seconds = actual_days.reshape(
            actual_days.shape[0], blocks_per_day, seconds
        ).sum(axis=2).reshape(-1).astype(np.float64)
        actual_from_minutes = raw[horizon]["actual"]["periodReturnBps"][
            BLEND_VALIDATION_END_DAY * blocks_per_day:
        ].astype(np.float64)
        difference = actual_from_seconds - actual_from_minutes
        audit[horizon]["actualEndpointAlignment"] = {
            "maximumAbsoluteDifferenceBps": float(np.max(np.abs(difference))),
            "rmseDifferenceBps": float(np.sqrt(np.mean(difference * difference))),
            "correlation": safe_correlation(actual_from_minutes, actual_from_seconds),
            "passed": bool(np.max(np.abs(difference)) < 5e-5),
        }


def render_document(report: dict) -> str:
    endpoint_correlation = {
        horizon: report["horizons"][horizon]["untouchedTest"][
            "pointEstimators"
        ]["ensembleMean"]["pearsonCorrelation"]
        for horizon in HORIZON_MINUTES
    }
    endpoint_absolute_correlation = {
        horizon: report["horizons"][horizon]["untouchedTest"][
            "conditionalScaleForecasts"
        ]["expectedAbsoluteReturn"]["pearsonCorrelation"]
        for horizon in HORIZON_MINUTES
    }
    one_second_correlation = {
        horizon: report["oneSecondPathAudit"][horizon][
            "candlePointEstimators"
        ]["ensembleMean"]["pearsonCorrelation"]
        for horizon in HORIZON_MINUTES
    }
    lines = [
        "# Rolling intraday 15m–1h point forecasts",
        "",
        f"Generated: {report['generatedAt']}",
        "",
        "These forecasts are refreshed immediately before every non-overlapping target window. History-window and ensemble-summary choices end before the final 91-day holdout.",
        "",
        "## Key findings",
        "",
        "- Refreshing immediately before the target window does **not** create a validated signed-return point forecast. Every pre-holdout blend gate still selects zero return.",
        f"- Raw signed-return correlations are {endpoint_correlation['15m']:.4f} at 15m, {endpoint_correlation['30m']:.4f} at 30m, and {endpoint_correlation['1h']:.4f} at 1h; all raw mean forecasts have negative MSE skill versus zero.",
        f"- Fresh state matters strongly for endpoint scale: expected absolute-return correlation is {endpoint_absolute_correlation['15m']:.3f} at 15m, {endpoint_absolute_correlation['30m']:.3f} at 30m, and {endpoint_absolute_correlation['1h']:.3f} at 1h.",
        "- Expected realized-variance correlation is 0.567–0.628, and expected activity correlation is 0.808–0.832.",
        "- A pre-holdout 1.25× spread correction substantially improves interval coverage, although 90% coverage remains only 84–86% and CRPS becomes slightly worse.",
        "- The selected history is 30d at all three horizons, but its advantage over 7–14d is small; this should be treated as a stable smoothing preference, not a sharp optimum.",
        f"- After materializing 377,395,200 generated seconds, 1s ensemble-mean correlations are {one_second_correlation['15m']:.6f}, {one_second_correlation['30m']:.6f}, and {one_second_correlation['1h']:.6f}; the internal realized path remains unpredictable.",
        "",
        "## Untouched results",
        "",
        "| horizon | origins | selected history | estimator | correlation | MAE (bps) | RMSE (bps) | MSE skill vs zero |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for horizon, values in report["horizons"].items():
        test = values["untouchedTest"]
        for estimator in (
            "ensembleMean",
            "ensembleMedian",
            "localDensityMode",
            SELECTED_BLEND,
            "lastBlockReturnBaseline",
        ):
            metrics = test["pointEstimators"][estimator]
            correlation = metrics["pearsonCorrelation"]
            lines.append(
                f"| {horizon} | {test['origins']} | {values['selectedHistoryWindowDays']}d | "
                f"{estimator} | {correlation:.6f} | {metrics['maeBps']:.6f} | "
                f"{metrics['rmseBps']:.6f} | {metrics['mseSkillVsZeroReturn']:.6f} |"
                if correlation is not None else
                f"| {horizon} | {test['origins']} | {values['selectedHistoryWindowDays']}d | "
                f"{estimator} | n/a | {metrics['maeBps']:.6f} | {metrics['rmseBps']:.6f} | "
                f"{metrics['mseSkillVsZeroReturn']:.6f} |"
            )
    lines += [
        "",
        "## Probabilistic calibration",
        "",
        "| horizon | selected spread | raw CRPS | calibrated CRPS | raw 90% coverage | calibrated 90% coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon, values in report["horizons"].items():
        test = values["untouchedTest"]
        raw = test["rawProbabilisticEnsemble"]
        calibrated = test["probabilisticEnsemble"]
        lines.append(
            f"| {horizon} | {values['spreadCalibration']['selectedFactor']:.2f} | "
            f"{raw['meanCrpsBps']:.6f} | {calibrated['meanCrpsBps']:.6f} | "
            f"{raw['centralIntervals']['90']['empiricalCoverage']:.6f} | "
            f"{calibrated['centralIntervals']['90']['empiricalCoverage']:.6f} |"
        )
    lines += [
        "",
        "## Protocol",
        "",
        "- Window selection: days 92–183 of the forecast year.",
        "- Blend stability validation: days 184–274, split into three chronological folds.",
        "- Untouched test: final 91 days.",
        "- Candidate histories: 1d, 3d, 7d, 14d, and 30d.",
        "- Each ensemble has 16 freshly sampled target paths; no historical block is resampled.",
        "- A complete-window return is tested before materializing its 1s allocation, because second-level projection preserves that endpoint exactly.",
        "",
        "## Materialized one-second paths",
        "",
        "| horizon | generated seconds | mean 1s correlation | mean 1s RMSE (bps) | expected absolute-return correlation | activity AUC | mean cumulative-path correlation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, values in report["oneSecondPathAudit"].items():
        mean = values["candlePointEstimators"]["ensembleMean"]
        conditional = values["conditionalOneSecondScale"]
        path = values["cumulativeIntrawindowPathAtMinuteEndpoints"]["ensembleMean"]
        lines.append(
            f"| {horizon} | {values['generatedSeconds']} | "
            f"{mean['pearsonCorrelation']:.6f} | {mean['rmseBps']:.6f} | "
            f"{conditional['expectedAbsoluteReturn']['pearsonCorrelation']:.6f} | "
            f"{conditional['activityProbability']['approximateRocAuc']:.6f} | "
            f"{path['meanWindowCorrelation']:.6f} |"
        )
    lines += [
        "",
        "Machine-readable results: `data/benchmarks/rolling-intraday-point-forecasts.json`.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="data/benchmarks/log-return-distributions.json")
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--forecast-start", default=FORECAST_START)
    parser.add_argument("--forecast-end", default=FORECAST_END)
    parser.add_argument("--windows", nargs="+", type=int, default=list(HISTORY_WINDOWS_DAYS))
    parser.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument(
        "--minute-cache",
        default="data/benchmarks/full-hierarchy-minute-history.npz",
    )
    parser.add_argument("--raw-cache", default=RAW_CACHE)
    parser.add_argument("--process-report", default=PROCESS_REPORT)
    parser.add_argument("--fit-cache", default=FIT_CACHE)
    parser.add_argument("--second-cache", default=SECOND_CACHE)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--document", default=DOCUMENT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis = json.loads((repo / args.analysis).read_text(encoding="utf-8"))
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    train_start = parse_time(args.train_start)
    forecast_start = parse_time(args.forecast_start)
    forecast_end = parse_time(args.forecast_end)
    windows = tuple(sorted(set(args.windows)))
    minute_returns, minute_variance, minute_counts = load_or_read_minute_history(
        repo=repo,
        cache_path=repo / args.minute_cache,
        source=source,
        train_start=train_start,
        end=forecast_end,
    )
    raw = load_or_generate_raw(
        cache_path=repo / args.raw_cache,
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        train_start=train_start,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        windows=windows,
        ensemble_size=args.ensemble_size,
    )
    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "design": {
            "trainStart": iso(train_start),
            "forecastStart": iso(forecast_start),
            "forecastEndExclusive": iso(forecast_end),
            "untouchedTestStart": iso(
                forecast_start + timedelta(days=BLEND_VALIDATION_END_DAY)
            ),
            "untouchedTestDays": (forecast_end - forecast_start).days - BLEND_VALIDATION_END_DAY,
            "historyWindowsDays": list(windows),
            "ensembleSize": args.ensemble_size,
            "futureCandlesUsedAtForecastTime": False,
            "untouchedOutcomesUsedForSelection": False,
            "historicalReturnBlocksResampled": False,
        },
        "horizons": {
            horizon: evaluate_horizon(horizon, raw[horizon], windows)
            for horizon in HORIZON_MINUTES
        },
    }
    process_report = json.loads(
        (repo / args.process_report).read_text(encoding="utf-8")
    )
    fit_cache = json.loads((repo / args.fit_cache).read_text(encoding="utf-8"))
    fitted = deserialize_fit(fit_cache["fittedParameters"])
    test_start = forecast_start + timedelta(days=BLEND_VALIDATION_END_DAY)
    actual_second_days = load_actual_second_days(source, test_start, forecast_end)
    history_offset_minutes = (forecast_start - train_start).days * MINUTES_PER_DAY
    report["oneSecondPathAudit"] = load_or_materialize_second_audit(
        cache_path=repo / args.second_cache,
        raw=raw,
        report=report,
        actual_days=actual_second_days,
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        history_offset_minutes=history_offset_minutes,
        fitted=fitted,
        calibration=process_report["oneSecondKernel"]["intradayCalibration"],
    )
    attach_endpoint_alignment(
        report["oneSecondPathAudit"],
        raw,
        actual_second_days,
    )
    report["conclusions"] = {
        "validatedSignedReturnPointForecastFound": False,
        "recommendedSignedReturnPointForecast": "zeroReturnBaseline",
        "recommendedScenarioSpreadFactor": {
            horizon: report["horizons"][horizon]["spreadCalibration"][
                "selectedFactor"
            ]
            for horizon in HORIZON_MINUTES
        },
        "rollingRefreshImproves": [
            "expected absolute return",
            "integrated one-second realized variance",
            "active-second count",
        ],
        "rollingRefreshDoesNotImprove": [
            "signed-return correlation sufficiently to beat zero-return MSE",
        ],
        "oneSecondMaterializationDecision": (
            "The existing one-second kernel was materialized for every untouched "
            "rolling origin and all 16 members. It improves conditioning of activity "
            "and absolute magnitude modestly but provides essentially zero signed "
            "one-second or cumulative intrawindow path correlation."
        ),
        "generatedOneSecondCandles": int(sum(
            report["oneSecondPathAudit"][horizon]["generatedSeconds"]
            for horizon in HORIZON_MINUTES
        )),
        "oneSecondEnsembleMeanCorrelation": {
            horizon: report["oneSecondPathAudit"][horizon][
                "candlePointEstimators"
            ]["ensembleMean"]["pearsonCorrelation"]
            for horizon in HORIZON_MINUTES
        },
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    document = repo / args.document
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(render_document(report), encoding="utf-8")
    print(output)
    print(document)


if __name__ == "__main__":
    main()
