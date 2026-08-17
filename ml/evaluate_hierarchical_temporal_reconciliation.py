"""Evaluate coherent 1d -> 8h -> 4h probabilistic target forecasts."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import special, stats
from scipy.optimize import linear_sum_assignment

from analyze_one_second_dependence_model import DAY_SECONDS, selected_files
from calibrate_rolling_forecasts import (
    CANDIDATE_METHODS,
    calibrated_ensemble,
    calibration_assessment,
    calibration_selection_score,
    fit_calibrator,
)
from evaluate_parametric_next_day_process import ensemble_metrics
from evaluate_rolling_refit_next_day_process import aggregate_history_blocks, fit_ar1
from trading_storage import read_candle_column


TRAIN_START = "2021-07-25T00:00:00+00:00"
TEST_START = "2025-07-25T00:00:00+00:00"
TEST_END = "2026-07-25T00:00:00+00:00"
DEFAULT_WINDOWS = (7, 30, 90, 365, 730)
DEFAULT_ENSEMBLE_SIZE = 64
DEFAULT_CALIBRATION_DAYS = 183
DEFAULT_ONLINE_CALIBRATION_DAYS = 90
MINUTES_PER_DAY = 1_440
LEVEL_MINUTES = {
    "1d": 1_440,
    "8h": 480,
    "4h": 240,
}
FEATURES = (
    "periodReturnBps",
    "oneSecondRealizedVarianceBpsSquared",
    "activeSeconds",
)
RECONCILIATION_SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
RECONCILIATION_METHODS = (
    "hardTopDownUncoupled",
    "hardTopDownRankCoupled",
    "errorWeightedUncoupled",
    "errorWeightedRankCoupled",
)
COHERENT_CALIBRATED_METHOD = "selectedHierarchyCoherentResidual"
RNG_SEED = 0x4849_4552


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis = read_json(repo / args.analysis)
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    train_start = parse_time(args.train_start)
    test_start = parse_time(args.test_start)
    test_end = parse_time(args.test_end)
    windows = tuple(sorted(set(args.windows)))
    if not train_start < test_start < test_end:
        raise ValueError("expected train-start < test-start < test-end")

    prior_files = selected_files(source, train_start - timedelta(days=1), train_start)
    files = selected_files(source, train_start, test_end)
    expected_days = (test_end - train_start).days
    if len(files) != expected_days:
        raise RuntimeError(f"expected {expected_days} complete daily shards, found {len(files)}")
    previous_close = (
        float(read_candle_column(prior_files[0], "close")[-1])
        if len(prior_files) == 1
        else float(read_candle_column(files[0], "close")[0])
    )
    minute_returns, minute_variance, minute_counts = read_minute_history(
        files,
        previous_close,
    )
    test_day_offset = (test_start - train_start).days
    test_days = (test_end - test_start).days
    test_minute_start = test_day_offset * MINUTES_PER_DAY
    test_minute_stop = test_minute_start + test_days * MINUTES_PER_DAY
    test_returns = minute_returns[test_minute_start:test_minute_stop].reshape(
        test_days,
        MINUTES_PER_DAY,
    )
    test_variance = minute_variance[test_minute_start:test_minute_stop].reshape(
        test_days,
        MINUTES_PER_DAY,
    )
    test_counts = minute_counts[test_minute_start:test_minute_stop].reshape(
        test_days,
        MINUTES_PER_DAY,
    )
    actual = actual_hierarchy(test_returns, test_variance, test_counts)
    forecasts = generate_raw_forecasts(
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        history_offset=test_minute_start,
        test_days=test_days,
        windows=windows,
        ensemble_size=args.ensemble_size,
        rng=np.random.default_rng(RNG_SEED),
    )
    report = evaluate_hierarchy(
        actual=actual,
        forecasts=forecasts,
        windows=windows,
        calibration_days=args.calibration_days,
        online_calibration_days=args.online_calibration_days,
        ensemble_size=args.ensemble_size,
    )
    report.update({
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "design": {
            "historyStart": train_start.isoformat().replace("+00:00", "Z"),
            "forecastOriginStart": test_start.isoformat().replace("+00:00", "Z"),
            "forecastOriginEndExclusive": test_end.isoformat().replace("+00:00", "Z"),
            "forecastOrigins": test_days,
            "calibrationOrigins": args.calibration_days,
            "untouchedTestOrigins": test_days - args.calibration_days,
            "untouchedTestStart": (
                test_start + timedelta(days=args.calibration_days)
            ).isoformat().replace("+00:00", "Z"),
            "historyWindowsDays": list(windows),
            "ensembleSize": args.ensemble_size,
            "onlineCalibrationHistoryDays": args.online_calibration_days,
            "levels": {
                level: {
                    "minutes": minutes,
                    "nodesPerDay": MINUTES_PER_DAY // minutes,
                }
                for level, minutes in LEVEL_MINUTES.items()
            },
            "features": list(FEATURES),
            "futureDataUsedAtForecastTime": False,
            "testOutcomesUsedForWindowOrWeightFit": False,
            "rawForecast": (
                "At each UTC-day origin, horizon-specific shrinkage AR(1)+Student-t "
                "target processes are fitted inside the trailing window and simulated "
                "sequentially through all 1d, 8h, and 4h nodes of the next day."
            ),
            "hardTopDown": (
                "The sampled 1d target is fixed; 8h children are adjusted to it and 4h "
                "children are adjusted to their 8h parent."
            ),
            "errorWeighted": (
                "All ten node forecasts are projected onto the coherent hierarchy using "
                "a shrunk forecast-error second-moment matrix fitted only on calibration "
                "origins."
            ),
            "calibratedBaseline": (
                "Each selected level/feature forecast is recalibrated at every test origin "
                "from only the latest completed forecast/outcome pairs. Reconciliation is "
                "compared against this stronger independent baseline."
            ),
            "coherentResidualFollowup": (
                "A causal 90-day residual-vector correction is applied at the six 4h "
                "bottom nodes, bridged back to the already selected 1d member target, and "
                "then aggregated upward. Its architecture was added after inspection of "
                "the initial reconciliation test and therefore remains exploratory."
            ),
            "coherentResidualArchitectureChosenAfterInitialTestInspection": True,
        },
    })
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(compact_summary(report), indent=2))


def read_minute_history(
    files: list[Path],
    previous_close: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minute_count = len(files) * MINUTES_PER_DAY
    minute_returns = np.empty(minute_count, dtype=np.float64)
    minute_variance = np.empty(minute_count, dtype=np.float64)
    minute_counts = np.empty(minute_count, dtype=np.uint8)
    cursor = 0
    last_close = previous_close
    for index, reference in enumerate(files):
        if index % 100 == 0:
            print(f"Reading hierarchy source {index}/{len(files)}...", file=sys.stderr)
        closes = read_candle_column(reference, "close")
        if closes.shape != (DAY_SECONDS,) or np.any(~np.isfinite(closes)) or np.any(closes <= 0):
            raise ValueError(f"invalid one-second closes: {reference}")
        returns = np.diff(np.log(np.concatenate((
            np.asarray([last_close], dtype=np.float64),
            closes,
        )))) * 10_000.0
        rows = returns.reshape(MINUTES_PER_DAY, 60)
        stop = cursor + MINUTES_PER_DAY
        minute_returns[cursor:stop] = np.sum(rows, axis=1)
        minute_variance[cursor:stop] = np.sum(rows * rows, axis=1)
        minute_counts[cursor:stop] = np.sum(rows != 0.0, axis=1, dtype=np.uint8)
        cursor = stop
        last_close = float(closes[-1])
    return minute_returns, minute_variance, minute_counts


def actual_hierarchy(
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    days = minute_returns.shape[0]
    for level, block_minutes in LEVEL_MINUTES.items():
        blocks = MINUTES_PER_DAY // block_minutes
        result[level] = {
            "periodReturnBps": np.sum(
                minute_returns.reshape(days, blocks, block_minutes),
                axis=2,
            ),
            "oneSecondRealizedVarianceBpsSquared": np.sum(
                minute_variance.reshape(days, blocks, block_minutes),
                axis=2,
            ),
            "activeSeconds": np.sum(
                minute_counts.reshape(days, blocks, block_minutes),
                axis=2,
            ).astype(np.float64),
        }
    return result


def generate_raw_forecasts(
    *,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    history_offset: int,
    test_days: int,
    windows: tuple[int, ...],
    ensemble_size: int,
    rng: np.random.Generator,
) -> dict[tuple[int, str, str], np.ndarray]:
    forecasts = {
        (window, level, feature): np.empty(
            (
                test_days,
                MINUTES_PER_DAY // block_minutes,
                ensemble_size,
            ),
            dtype=np.float64,
        )
        for window in windows
        for level, block_minutes in LEVEL_MINUTES.items()
        for feature in FEATURES
    }
    for day in range(test_days):
        if day % 25 == 0:
            print(f"Hierarchy raw forecast {day}/{test_days}...", flush=True)
        history_end = history_offset + day * MINUTES_PER_DAY
        for level, block_minutes in LEVEL_MINUTES.items():
            steps = MINUTES_PER_DAY // block_minutes
            common_draws = {
                name: rng.standard_t(8.0, (ensemble_size, steps))
                for name in ("variance", "return", "activity")
            }
            for window in windows:
                history_start = history_end - window * MINUTES_PER_DAY
                block_return, block_variance, block_active = aggregate_history_blocks(
                    minute_returns[history_start:history_end],
                    minute_variance[history_start:history_end],
                    minute_counts[history_start:history_end],
                    block_minutes,
                )
                targets = sample_local_period_target_paths(
                    block_return,
                    block_variance,
                    block_active,
                    common_draws,
                )
                forecasts[(window, level, "periodReturnBps")][day] = targets[
                    "return"
                ].T
                forecasts[(
                    window,
                    level,
                    "oneSecondRealizedVarianceBpsSquared",
                )][day] = targets["variance"].T
                forecasts[(window, level, "activeSeconds")][day] = (
                    targets["activeFraction"].T * block_minutes * 60.0
                )
    return forecasts


def sample_local_period_target_paths(
    returns: np.ndarray,
    variance: np.ndarray,
    active_fraction: np.ndarray,
    common_draws: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    shapes = {np.asarray(value).shape for value in common_draws.values()}
    if len(shapes) != 1:
        raise ValueError("all common draw arrays must have the same shape")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("sequential common draws must have shape (paths, steps)")
    log_variance = np.log(np.maximum(variance, 1e-12))
    variance_fit = fit_ar1(log_variance)
    next_log_variance = draw_fitted_ar1_paths(
        variance_fit,
        common_draws["variance"],
    )
    next_variance = np.exp(next_log_variance)

    efficiency = returns / np.sqrt(np.maximum(variance, 1e-12))
    next_efficiency = draw_fitted_ar1_paths(
        fit_ar1(efficiency),
        common_draws["return"],
    )
    next_return = next_efficiency * np.sqrt(next_variance)

    active_logit = special.logit(np.clip(active_fraction, 1e-5, 1.0 - 1e-5))
    centered_log_variance = log_variance - np.mean(log_variance)
    denominator = float(np.sum(centered_log_variance * centered_log_variance))
    raw_slope = (
        float(np.sum(
            centered_log_variance * (active_logit - np.mean(active_logit))
        )) / denominator
        if denominator > 1e-12
        else 0.0
    )
    slope = raw_slope * returns.size / (returns.size + 30.0)
    residual = active_logit - (
        np.mean(active_logit) + slope * centered_log_variance
    )
    next_activity_residual = draw_fitted_ar1_paths(
        fit_ar1(residual),
        common_draws["activity"],
    )
    next_active_fraction = special.expit(
        np.mean(active_logit)
        + slope * (next_log_variance - np.mean(log_variance))
        + next_activity_residual
    )
    return {
        "variance": next_variance,
        "return": next_return,
        "activeFraction": np.clip(next_active_fraction, 0.01, 0.999),
    }


def draw_fitted_ar1_paths(fit: dict[str, float], common_t8: np.ndarray) -> np.ndarray:
    common_t8 = np.asarray(common_t8, dtype=np.float64)
    uniforms = stats.t.cdf(common_t8, df=8.0)
    innovations = stats.t.ppf(uniforms, df=fit["degrees"])
    standard_deviation = math.sqrt(fit["degrees"] / (fit["degrees"] - 2.0))
    innovations = fit["scale"] * innovations / standard_deviation
    result = np.empty_like(innovations)
    state = np.full(common_t8.shape[0], fit["last"], dtype=np.float64)
    for step in range(common_t8.shape[1]):
        state = fit["mean"] + fit["phi"] * (state - fit["mean"]) + innovations[:, step]
        result[:, step] = state
    return result


def hierarchy_matrix() -> np.ndarray:
    matrix = np.zeros((10, 6), dtype=np.float64)
    matrix[0] = 1.0
    matrix[1, 0:2] = 1.0
    matrix[2, 2:4] = 1.0
    matrix[3, 4:6] = 1.0
    matrix[4:] = np.eye(6)
    return matrix


def rank_couple_hierarchy(raw: np.ndarray) -> np.ndarray:
    """Pair scale-specific members while preserving every node's marginal ensemble."""
    result = np.asarray(raw, dtype=np.float64).copy()
    paths = raw.shape[2]
    for day in range(raw.shape[0]):
        daily_order = np.argsort(raw[day, 0])
        eight_source_order = np.argsort(np.sum(raw[day, 1:4], axis=0))
        result[day, 1:4, daily_order] = raw[day, 1:4, eight_source_order]

        target = result[day, 1:4].T
        source = np.stack((
            np.sum(raw[day, 4:6], axis=0),
            np.sum(raw[day, 6:8], axis=0),
            np.sum(raw[day, 8:10], axis=0),
        ), axis=1)
        scale = np.maximum(
            0.5 * (np.std(target, axis=0) + np.std(source, axis=0)),
            1e-9,
        )
        cost = np.sum(
            ((target[:, None, :] - source[None, :, :]) / scale[None, None, :]) ** 2,
            axis=2,
        )
        target_indexes, source_indexes = linear_sum_assignment(cost)
        if target_indexes.size != paths:
            raise RuntimeError("hierarchy path assignment is incomplete")
        result[day, 4:10, target_indexes] = raw[day, 4:10, source_indexes]
    return result


def stack_actual_nodes(actual: dict, feature: str) -> np.ndarray:
    return np.concatenate(
        [actual[level][feature] for level in ("1d", "8h", "4h")],
        axis=1,
    )


def stack_forecast_nodes(
    forecasts: dict,
    selected_windows: dict,
    feature: str,
) -> np.ndarray:
    return np.concatenate(
        [
            forecasts[(selected_windows[level][feature], level, feature)]
            for level in ("1d", "8h", "4h")
        ],
        axis=1,
    )


def select_calibration_methods(
    actual: dict,
    forecasts: dict,
    selected_windows: dict,
    *,
    fit_days: int,
    selection_end: int,
    ensemble_size: int,
) -> tuple[dict, dict]:
    selected = {level: {} for level in LEVEL_MINUTES}
    scores = {level: {} for level in LEVEL_MINUTES}
    for level in LEVEL_MINUTES:
        for feature in FEATURES:
            raw = forecasts[(selected_windows[level][feature], level, feature)]
            actual_values = actual[level][feature]
            model_feature, actual_fit, raw_fit = calibration_representation(
                level,
                feature,
                actual_values[:fit_days],
                raw[:fit_days],
            )
            _, actual_selection, raw_selection = calibration_representation(
                level,
                feature,
                actual_values[fit_days:selection_end],
                raw[fit_days:selection_end],
            )
            actual_fit = actual_fit.reshape(-1)
            raw_fit = raw_fit.reshape(-1, ensemble_size)
            actual_selection = actual_selection.reshape(-1)
            raw_selection = raw_selection.reshape(-1, ensemble_size)
            scores[level][feature] = {}
            for method in CANDIDATE_METHODS:
                if method == "raw":
                    prediction = raw_selection
                else:
                    model = fit_calibrator(
                        model_feature,
                        actual_fit,
                        raw_fit,
                        method,
                    )
                    prediction, _ = calibrated_ensemble(
                        model_feature,
                        raw_selection,
                        model,
                        ensemble_size,
                    )
                metrics = ensemble_metrics(actual_selection, prediction)
                scores[level][feature][method] = {
                    "selectionScore": calibration_selection_score(metrics),
                    "meanCrps": metrics["meanCrps"],
                    "maximumAbsoluteCoverageError": calibration_assessment(metrics)[
                        "maximumAbsoluteCoverageError"
                    ],
                    "pitKsStatistic": metrics["pit"]["ksStatistic"],
                }
            selected[level][feature] = min(
                CANDIDATE_METHODS,
                key=lambda method: scores[level][feature][method]["selectionScore"],
            )
    return selected, scores


def stack_online_calibrated_nodes(
    actual: dict,
    forecasts: dict,
    selected_windows: dict,
    selected_methods: dict,
    *,
    feature: str,
    start: int,
    history_days: int,
    output_members: int,
) -> np.ndarray:
    calibrated = []
    for level in ("1d", "8h", "4h"):
        raw = forecasts[(selected_windows[level][feature], level, feature)]
        calibrated.append(online_calibrate_level(
            level,
            feature,
            actual[level][feature],
            raw,
            start=start,
            history_days=history_days,
            method=selected_methods[level][feature],
            output_members=output_members,
        ))
    return np.concatenate(calibrated, axis=1)


def online_calibrate_level(
    level: str,
    feature: str,
    actual: np.ndarray,
    ensemble: np.ndarray,
    *,
    start: int,
    history_days: int,
    method: str,
    output_members: int,
) -> np.ndarray:
    result = ensemble.copy()
    if method == "raw":
        return result
    model_feature = calibration_feature_name(feature)
    for origin in range(start, actual.shape[0]):
        history_start = max(0, origin - history_days)
        _, history_actual, history_ensemble = calibration_representation(
            level,
            feature,
            actual[history_start:origin],
            ensemble[history_start:origin],
        )
        model = fit_calibrator(
            model_feature,
            history_actual.reshape(-1),
            history_ensemble.reshape(-1, ensemble.shape[-1]),
            method,
        )
        _, _, current = calibration_representation(
            level,
            feature,
            actual[origin:origin + 1],
            ensemble[origin:origin + 1],
        )
        prediction, _ = calibrated_ensemble(
            model_feature,
            current.reshape(-1, ensemble.shape[-1]),
            model,
            output_members,
        )
        prediction = prediction.reshape(ensemble.shape[1], output_members)
        if feature == "activeSeconds":
            prediction *= LEVEL_MINUTES[level] * 60.0
            prediction = np.clip(
                prediction,
                0.0,
                LEVEL_MINUTES[level] * 60.0,
            )
        result[origin] = prediction
    return result


def calibration_feature_name(feature: str) -> str:
    return "activeSecondFraction" if feature == "activeSeconds" else feature


def calibration_representation(
    level: str,
    feature: str,
    actual: np.ndarray,
    ensemble: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    if feature == "activeSeconds":
        scale = LEVEL_MINUTES[level] * 60.0
        return (
            "activeSecondFraction",
            np.asarray(actual, dtype=np.float64) / scale,
            np.asarray(ensemble, dtype=np.float64) / scale,
        )
    return (
        feature,
        np.asarray(actual, dtype=np.float64),
        np.asarray(ensemble, dtype=np.float64),
    )


def select_windows(
    actual: dict,
    forecasts: dict,
    windows: tuple[int, ...],
    calibration_days: int,
) -> tuple[dict, dict]:
    selected = {level: {} for level in LEVEL_MINUTES}
    scores = {level: {} for level in LEVEL_MINUTES}
    for level in LEVEL_MINUTES:
        for feature in FEATURES:
            actual_values = actual[level][feature][:calibration_days].reshape(-1)
            scores[level][feature] = {}
            for window in windows:
                ensemble = forecasts[(window, level, feature)][:calibration_days]
                ensemble = ensemble.reshape(-1, ensemble.shape[-1])
                metrics = ensemble_metrics(actual_values, ensemble)
                scores[level][feature][str(window)] = {
                    "meanCrps": metrics["meanCrps"],
                    "normalizedCrpsByActualStd": metrics[
                        "normalizedCrpsByActualStd"
                    ],
                    "maximumAbsoluteCoverageError": calibration_assessment(metrics)[
                        "maximumAbsoluteCoverageError"
                    ],
                }
            selected[level][feature] = min(
                windows,
                key=lambda window: scores[level][feature][str(window)]["meanCrps"],
            )
    return selected, scores


def hard_top_down_reconcile(
    raw: np.ndarray,
    feature: str,
    error_second_moment: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(raw)
    result[:, 0] = raw[:, 0]
    variances = np.maximum(np.diag(error_second_moment), 1e-12)
    for day in range(raw.shape[0]):
        for member in range(raw.shape[2]):
            parent = float(raw[day, 0, member])
            eight_raw = raw[day, 1:4, member]
            if feature == "periodReturnBps":
                eight = additive_bridge(eight_raw, parent, variances[1:4])
            else:
                cap = 28_800.0 if feature == "activeSeconds" else None
                eight = positive_allocation(eight_raw, parent, cap)
            result[day, 1:4, member] = eight
            for block in range(3):
                child_slice = slice(4 + 2 * block, 6 + 2 * block)
                four_raw = raw[day, child_slice, member]
                if feature == "periodReturnBps":
                    four = additive_bridge(
                        four_raw,
                        float(eight[block]),
                        variances[child_slice],
                    )
                else:
                    cap = 14_400.0 if feature == "activeSeconds" else None
                    four = positive_allocation(four_raw, float(eight[block]), cap)
                result[day, child_slice, member] = four
    return result


def additive_bridge(values: np.ndarray, target: float, error_variances: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(error_variances, dtype=np.float64), 1e-12)
    weights /= np.sum(weights)
    return np.asarray(values, dtype=np.float64) + weights * (target - np.sum(values))


def positive_allocation(
    values: np.ndarray,
    target: float,
    cap: float | None,
) -> np.ndarray:
    target = max(float(target), 0.0)
    positive = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    if cap is None:
        return positive * (target / np.sum(positive))
    target = min(target, cap * positive.size)
    result = np.zeros_like(positive)
    remaining = np.ones(positive.size, dtype=bool)
    remaining_target = target
    while np.any(remaining):
        weights = positive[remaining]
        proposed = remaining_target * weights / np.sum(weights)
        overflow = proposed > cap
        indexes = np.flatnonzero(remaining)
        if not np.any(overflow):
            result[indexes] = proposed
            break
        result[indexes[overflow]] = cap
        remaining[indexes[overflow]] = False
        remaining_target = target - float(np.sum(result))
    return result


def fit_error_weighted_projection(
    raw: np.ndarray,
    actual: np.ndarray,
    calibration_days: int,
    fit_start: int = 0,
    shrinkage: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(raw[fit_start:calibration_days], axis=2)
    errors = median - actual[fit_start:calibration_days]
    second_moment = errors.T @ errors / errors.shape[0]
    diagonal = np.diag(np.diag(second_moment))
    shrunk = (1.0 - shrinkage) * second_moment + shrinkage * diagonal
    ridge = max(float(np.trace(shrunk)) / shrunk.shape[0] * 1e-8, 1e-10)
    shrunk += np.eye(shrunk.shape[0]) * ridge
    structure = hierarchy_matrix()
    precision = np.linalg.pinv(shrunk)
    bottom_projection = np.linalg.pinv(
        structure.T @ precision @ structure
    ) @ structure.T @ precision
    return bottom_projection, shrunk


def error_weighted_reconcile(
    raw: np.ndarray,
    feature: str,
    bottom_projection: np.ndarray,
) -> tuple[np.ndarray, int]:
    bottom = np.einsum("bn,dnp->dbp", bottom_projection, raw)
    before = bottom.copy()
    if feature == "oneSecondRealizedVarianceBpsSquared":
        bottom = np.maximum(bottom, 1e-12)
    elif feature == "activeSeconds":
        bottom = np.clip(bottom, 0.0, 14_400.0)
    clips = int(np.count_nonzero(bottom != before))
    reconciled = np.einsum("nb,dbp->dnp", hierarchy_matrix(), bottom)
    return reconciled, clips


def select_projection_shrinkage(
    raw: np.ndarray,
    actual: np.ndarray,
    *,
    feature: str,
    fit_start: int,
    fit_end: int,
    selection_end: int,
) -> tuple[float, dict[str, dict[str, float]]]:
    scores = {}
    scales = np.maximum(np.std(actual[fit_start:fit_end], axis=0, ddof=1), 1e-9)
    selection_actual = actual[fit_end:selection_end]
    for shrinkage in RECONCILIATION_SHRINKAGES:
        projection, _ = fit_error_weighted_projection(
            raw,
            actual,
            fit_end,
            fit_start=fit_start,
            shrinkage=shrinkage,
        )
        reconciled, clips = error_weighted_reconcile(raw, feature, projection)
        selection_ensemble = reconciled[fit_end:selection_end]
        joint_energy = energy_score(
            selection_actual,
            selection_ensemble,
            scales,
        )
        marginal = ensemble_metrics(
            selection_actual.reshape(-1),
            selection_ensemble.reshape(-1, selection_ensemble.shape[-1]),
        )
        score = joint_energy + marginal["normalizedCrpsByActualStd"]
        scores[str(shrinkage)] = {
            "selectionScore": float(score),
            "jointEnergyScore": joint_energy,
            "normalizedAllNodeCrps": marginal["normalizedCrpsByActualStd"],
            "positiveBottomClipsAcrossAllOrigins": clips,
        }
    selected = min(
        RECONCILIATION_SHRINKAGES,
        key=lambda value: scores[str(value)]["selectionScore"],
    )
    return selected, scores


def coherence_errors(values: np.ndarray) -> dict[str, float]:
    daily_vs_eight = values[:, 0] - np.sum(values[:, 1:4], axis=1)
    daily_vs_four = values[:, 0] - np.sum(values[:, 4:10], axis=1)
    eight_vs_four = np.stack(
        [
            values[:, 1 + block] - np.sum(
                values[:, 4 + 2 * block:6 + 2 * block],
                axis=1,
            )
            for block in range(3)
        ],
        axis=1,
    )
    return {
        "meanAbsoluteDailyVsEightHour": float(np.mean(np.abs(daily_vs_eight))),
        "meanAbsoluteDailyVsFourHour": float(np.mean(np.abs(daily_vs_four))),
        "meanAbsoluteEightHourVsFourHour": float(np.mean(np.abs(eight_vs_four))),
        "maximumAbsoluteConstraintError": float(max(
            np.max(np.abs(daily_vs_eight)),
            np.max(np.abs(daily_vs_four)),
            np.max(np.abs(eight_vs_four)),
        )),
    }


def energy_score(
    actual: np.ndarray,
    ensemble: np.ndarray,
    scales: np.ndarray,
) -> float:
    normalized_actual = actual / scales[None, :]
    normalized_ensemble = ensemble / scales[None, :, None]
    first = np.mean(np.linalg.norm(
        normalized_ensemble - normalized_actual[:, :, None],
        axis=1,
    ), axis=1)
    second = np.empty(actual.shape[0], dtype=np.float64)
    for day in range(actual.shape[0]):
        members = normalized_ensemble[day].T
        differences = members[:, None, :] - members[None, :, :]
        second[day] = 0.5 * float(np.mean(np.linalg.norm(differences, axis=2)))
    return float(np.mean(first - second))


def evaluate_hierarchy(
    *,
    actual: dict,
    forecasts: dict,
    windows: tuple[int, ...],
    calibration_days: int,
    online_calibration_days: int,
    ensemble_size: int,
) -> dict:
    test_days = next(iter(actual["1d"].values())).shape[0]
    if not 40 <= calibration_days <= test_days - 40:
        raise ValueError("calibration split must leave at least 40 test origins")
    selected_windows, window_scores = select_windows(
        actual,
        forecasts,
        windows,
        calibration_days,
    )
    method_selection_days = calibration_days // 2
    selected_calibration_methods, calibration_method_scores = (
        select_calibration_methods(
            actual,
            forecasts,
            selected_windows,
            fit_days=method_selection_days,
            selection_end=calibration_days,
            ensemble_size=ensemble_size,
        )
    )
    methods = {
        "independentRaw": {},
        "independentCalibrated": {},
        "rankCoupledIndependent": {},
        **{method: {} for method in RECONCILIATION_METHODS},
    }
    reconciliation = {}
    all_actual = {}
    reconciliation_selection_start = (
        method_selection_days + calibration_days
    ) // 2
    for feature in FEATURES:
        actual_nodes = stack_actual_nodes(actual, feature)
        raw = stack_forecast_nodes(forecasts, selected_windows, feature)
        calibrated = stack_online_calibrated_nodes(
            actual,
            forecasts,
            selected_windows,
            selected_calibration_methods,
            feature=feature,
            start=method_selection_days,
            history_days=online_calibration_days,
            output_members=ensemble_size,
        )
        coupled = rank_couple_hierarchy(calibrated)
        reconciliation_fit_end = reconciliation_selection_start
        all_actual[feature] = actual_nodes
        methods["independentRaw"][feature] = raw
        methods["independentCalibrated"][feature] = calibrated
        methods["rankCoupledIndependent"][feature] = coupled
        reconciliation[feature] = {}
        for coupling, candidate in (
            ("uncoupled", calibrated),
            ("rankCoupled", coupled),
        ):
            selected_shrinkage, shrinkage_scores = select_projection_shrinkage(
                candidate,
                actual_nodes,
                feature=feature,
                fit_start=method_selection_days,
                fit_end=reconciliation_fit_end,
                selection_end=calibration_days,
            )
            projection, error_second_moment = fit_error_weighted_projection(
                candidate,
                actual_nodes,
                calibration_days,
                fit_start=method_selection_days,
                shrinkage=selected_shrinkage,
            )
            hard = hard_top_down_reconcile(
                candidate,
                feature,
                error_second_moment,
            )
            weighted, clips = error_weighted_reconcile(
                candidate,
                feature,
                projection,
            )
            suffix = "RankCoupled" if coupling == "rankCoupled" else "Uncoupled"
            methods[f"hardTopDown{suffix}"][feature] = hard
            methods[f"errorWeighted{suffix}"][feature] = weighted
            reconciliation[feature][coupling] = {
                "errorSecondMoment": error_second_moment.tolist(),
                "bottomProjection": projection.tolist(),
                "positiveBottomClips": clips,
                "selectedCovarianceShrinkage": selected_shrinkage,
                "shrinkageSelectionFitStart": method_selection_days,
                "shrinkageSelectionFitEndExclusive": reconciliation_fit_end,
                "shrinkageSelectionEndExclusive": calibration_days,
                "shrinkageSelectionScores": shrinkage_scores,
            }

    selected_hierarchy_methods, hierarchy_method_scores = select_hierarchy_methods(
        methods,
        all_actual,
        start=reconciliation_selection_start,
        end=calibration_days,
    )
    methods[COHERENT_CALIBRATED_METHOD] = {
        feature: coherent_bottom_residual_calibration(
            all_actual[feature],
            methods[selected_hierarchy_methods[feature]][feature],
            feature=feature,
            start=calibration_days,
            history_days=online_calibration_days,
        )
        for feature in FEATURES
    }

    node_slices = {
        "1d": slice(0, 1),
        "8h": slice(1, 4),
        "4h": slice(4, 10),
        "allNodes": slice(0, 10),
    }
    method_report = {}
    for method, feature_values in methods.items():
        method_report[method] = {
            "features": {},
            "coherence": {},
        }
        for feature, ensemble in feature_values.items():
            actual_nodes = all_actual[feature]
            feature_report = {"levels": {}}
            for level, node_slice in node_slices.items():
                actual_test = actual_nodes[calibration_days:, node_slice].reshape(-1)
                ensemble_test = ensemble[calibration_days:, node_slice, :].reshape(
                    -1,
                    ensemble_size,
                )
                metrics = ensemble_metrics(actual_test, ensemble_test)
                feature_report["levels"][level] = {
                    "probabilistic": metrics,
                    "calibrationAssessment": calibration_assessment(metrics),
                }
            scales = np.maximum(
                np.std(actual_nodes[:calibration_days], axis=0, ddof=1),
                1e-9,
            )
            feature_report["jointTenNodeEnergyScore"] = energy_score(
                actual_nodes[calibration_days:],
                ensemble[calibration_days:],
                scales,
            )
            method_report[method]["features"][feature] = feature_report
            method_report[method]["coherence"][feature] = coherence_errors(
                ensemble[calibration_days:]
            )

    for method in (*RECONCILIATION_METHODS, COHERENT_CALIBRATED_METHOD):
        for feature in FEATURES:
            independent = method_report["independentCalibrated"]["features"][feature]
            candidate = method_report[method]["features"][feature]
            candidate["jointEnergySkillVsCalibratedIndependent"] = float(
                1.0
                - candidate["jointTenNodeEnergyScore"]
                / independent["jointTenNodeEnergyScore"]
            )
            for level in node_slices:
                raw_crps = independent["levels"][level]["probabilistic"]["meanCrps"]
                candidate_crps = candidate["levels"][level]["probabilistic"]["meanCrps"]
                candidate["levels"][level]["crpsSkillVsCalibratedIndependent"] = float(
                    1.0 - candidate_crps / raw_crps
                )

    return {
        "selectedHistoryWindows": selected_windows,
        "calibrationWindowScores": window_scores,
        "selectedCalibrationMethods": selected_calibration_methods,
        "calibrationMethodScores": calibration_method_scores,
        "selectedHierarchyMethods": selected_hierarchy_methods,
        "hierarchyMethodSelectionScores": hierarchy_method_scores,
        "selectedMethodUntouchedTest": {
            feature: selected_method_test_summary(
                selected_hierarchy_methods[feature],
                feature,
                method_report,
            )
            for feature in FEATURES
        },
        "coherentResidualCalibrationChronologicalTest": {
            feature: selected_method_test_summary(
                COHERENT_CALIBRATED_METHOD,
                feature,
                method_report,
            )
            for feature in FEATURES
        },
        "reconciliationFits": reconciliation,
        "methods": method_report,
        "summary": build_summary(method_report),
    }


def select_hierarchy_methods(
    methods: dict,
    actual: dict,
    *,
    start: int,
    end: int,
) -> tuple[dict[str, str], dict[str, dict]]:
    candidates = (
        "independentCalibrated",
        "rankCoupledIndependent",
        *RECONCILIATION_METHODS,
    )
    selected = {}
    scores = {}
    for feature in FEATURES:
        actual_values = actual[feature]
        scales = np.maximum(np.std(actual_values[:start], axis=0, ddof=1), 1e-9)
        selection_actual = actual_values[start:end]
        scores[feature] = {}
        for method in candidates:
            selection_ensemble = methods[method][feature][start:end]
            joint = energy_score(selection_actual, selection_ensemble, scales)
            marginal = ensemble_metrics(
                selection_actual.reshape(-1),
                selection_ensemble.reshape(-1, selection_ensemble.shape[-1]),
            )
            score = joint + marginal["normalizedCrpsByActualStd"]
            scores[feature][method] = {
                "selectionScore": float(score),
                "jointEnergyScore": joint,
                "normalizedAllNodeCrps": marginal["normalizedCrpsByActualStd"],
                "exactlyCoherent": is_coherent_method(method),
            }
        selected[feature] = min(
            candidates,
            key=lambda method: scores[feature][method]["selectionScore"],
        )
    return selected, scores


def selected_method_test_summary(
    method: str,
    feature: str,
    method_report: dict,
) -> dict:
    candidate = method_report[method]["features"][feature]
    baseline = method_report["independentCalibrated"]["features"][feature]
    return {
        "method": method,
        "exactlyCoherent": is_coherent_method(method),
        "jointEnergySkillVsCalibratedIndependent": float(
            1.0
            - candidate["jointTenNodeEnergyScore"]
            / baseline["jointTenNodeEnergyScore"]
        ),
        "crpsSkillVsCalibratedIndependentByLevel": {
            level: float(
                1.0
                - candidate["levels"][level]["probabilistic"]["meanCrps"]
                / baseline["levels"][level]["probabilistic"]["meanCrps"]
            )
            for level in ("1d", "8h", "4h", "allNodes")
        },
        "pitPassByLevel": {
            level: candidate["levels"][level]["calibrationAssessment"][
                "pitUniformAtFivePercent"
            ]
            for level in ("1d", "8h", "4h", "allNodes")
        },
    }


def coherent_bottom_residual_calibration(
    actual: np.ndarray,
    coherent_ensemble: np.ndarray,
    *,
    feature: str,
    start: int,
    history_days: int,
) -> np.ndarray:
    result = coherent_ensemble.copy()
    members = coherent_ensemble.shape[2]
    for origin in range(start, actual.shape[0]):
        history_start = max(0, origin - history_days)
        actual_bottom = bottom_forward_transform(
            feature,
            actual[history_start:origin, 4:10],
        )
        predicted_bottom = bottom_forward_transform(
            feature,
            np.median(
                coherent_ensemble[history_start:origin, 4:10],
                axis=2,
            ),
        )
        residuals = actual_bottom - predicted_bottom
        if feature == "oneSecondRealizedVarianceBpsSquared":
            residuals = np.clip(residuals, -2.5, 2.5)
        indexes = np.minimum(
            (
                (np.arange(members, dtype=np.float64) + 0.5)
                * residuals.shape[0]
                / members
            ).astype(np.int64),
            residuals.shape[0] - 1,
        )
        current_center = bottom_forward_transform(
            feature,
            np.median(coherent_ensemble[origin, 4:10], axis=1),
        )
        bottom = bottom_inverse_transform(
            feature,
            current_center[:, None] + residuals[indexes].T,
        )
        parent_targets = coherent_ensemble[origin, 0]
        for member in range(members):
            if feature == "periodReturnBps":
                bottom[:, member] = additive_bridge(
                    bottom[:, member],
                    float(parent_targets[member]),
                    np.ones(6, dtype=np.float64),
                )
            else:
                cap = 14_400.0 if feature == "activeSeconds" else None
                bottom[:, member] = positive_allocation(
                    bottom[:, member],
                    float(parent_targets[member]),
                    cap,
                )
        result[origin] = hierarchy_matrix() @ bottom
    return result


def bottom_forward_transform(feature: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if feature == "oneSecondRealizedVarianceBpsSquared":
        return np.log(np.maximum(values, 1e-12))
    if feature == "activeSeconds":
        return special.logit(np.clip(values / 14_400.0, 1e-6, 1.0 - 1e-6))
    return values.copy()


def bottom_inverse_transform(feature: str, values: np.ndarray) -> np.ndarray:
    if feature == "oneSecondRealizedVarianceBpsSquared":
        return np.exp(np.clip(values, -40.0, 40.0))
    if feature == "activeSeconds":
        return special.expit(values) * 14_400.0
    return np.asarray(values, dtype=np.float64).copy()


def is_coherent_method(method: str) -> bool:
    return method in RECONCILIATION_METHODS or method == COHERENT_CALIBRATED_METHOD


def build_summary(method_report: dict) -> dict:
    result = {}
    for method in RECONCILIATION_METHODS:
        result[method] = {}
        for feature in FEATURES:
            values = method_report[method]["features"][feature]
            result[method][feature] = {
                "jointEnergySkillVsCalibratedIndependent": values[
                    "jointEnergySkillVsCalibratedIndependent"
                ],
                "crpsSkillVsCalibratedIndependentByLevel": {
                    level: values["levels"][level][
                        "crpsSkillVsCalibratedIndependent"
                    ]
                    for level in ("1d", "8h", "4h", "allNodes")
                },
                "pitPassByLevel": {
                    level: values["levels"][level]["calibrationAssessment"][
                        "pitUniformAtFivePercent"
                    ]
                    for level in ("1d", "8h", "4h", "allNodes")
                },
            }
    return result


def compact_summary(report: dict) -> dict:
    return {
        "design": report["design"],
        "selectedHistoryWindows": report["selectedHistoryWindows"],
        "selectedCalibrationMethods": report["selectedCalibrationMethods"],
        "selectedHierarchyMethods": report["selectedHierarchyMethods"],
        "selectedMethodUntouchedTest": report["selectedMethodUntouchedTest"],
        "coherentResidualCalibrationChronologicalTest": report[
            "coherentResidualCalibrationChronologicalTest"
        ],
        "summary": report["summary"],
        "coherence": {
            method: report["methods"][method]["coherence"]
            for method in report["methods"]
        },
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="data/benchmarks/log-return-distributions.json")
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--test-start", default=TEST_START)
    parser.add_argument("--test-end", default=TEST_END)
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    parser.add_argument("--calibration-days", type=int, default=DEFAULT_CALIBRATION_DAYS)
    parser.add_argument(
        "--online-calibration-days",
        type=int,
        default=DEFAULT_ONLINE_CALIBRATION_DAYS,
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/hierarchical-temporal-reconciliation.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
