"""Leakage-safe hierarchical probabilistic forecasts resolved to one second.

The rolling target model is fitted independently at 1d, 8h, 4h, 2h, 1h,
30m, 15m, and 1m.  History windows and marginal calibration methods are
selected on the first half of a chronological calibration period.  A second
calibration slice selects how much information each parent level contributes
to coherent minute leaves.  The untouched final slice is used exactly once.

The component-free fitted one-second kernel then turns selected minute targets
for return, integrated variance, and activity into new second candles.  It does
not resample historical candles, and every generated minute exactly realizes
its feasible target triple.  All coarser values are direct sums of that same
one-second path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

from analyze_one_second_dependence_model import (
    AcfAccumulator,
    dense_histogram,
    full_histogram,
    histogram_edges,
    jensen_shannon_bits,
    selected_files,
)
from analyze_parametric_one_second_process import (
    DAY_SECONDS,
    MINUTES_PER_DAY,
    ConditionalMagnitudeMixture,
    DailyReturnFit,
    DailyVarianceFit,
    FactorMixture,
    FittedProcess,
    SeasonalFit,
    bounded_histogram_counts,
    fit_process,
    gaussianize_quantile_spline,
    generate_projected_second_batch,
    measure_source,
    project_returns_to_daily_targets,
    serialize_fit,
)
from calibrate_rolling_forecasts import (
    CANDIDATE_METHODS,
    calibrated_ensemble,
    fit_calibrator,
)
from evaluate_hierarchical_temporal_reconciliation import (
    read_minute_history,
    sample_local_period_target_paths,
)
from evaluate_parametric_next_day_process import (
    calibrate_intraday_generation,
    ensemble_metrics,
    histogram_from_counts,
)
from evaluate_rolling_refit_next_day_process import aggregate_history_blocks
from trading_storage import read_candle_column


TRAIN_START = "2021-07-25T00:00:00+00:00"
FORECAST_START = "2025-07-25T00:00:00+00:00"
FORECAST_END = "2026-07-25T00:00:00+00:00"
CALIBRATION_DAYS = 183
HIERARCHY_SELECTION_DAYS = 91
ENSEMBLE_SIZE = 16
SECOND_PATHS_PER_DAY = 4
RNG_SEED = 0x3153_4849
FEATURES = (
    "periodReturnBps",
    "oneSecondRealizedVarianceBpsSquared",
    "activeSeconds",
)
LEVEL_MINUTES = {
    "1d": 1_440,
    "8h": 480,
    "4h": 240,
    "2h": 120,
    "1h": 60,
    "30m": 30,
    "15m": 15,
    "1m": 1,
}
BOTTOM_UP_LEVELS = ("15m", "30m", "1h", "2h", "4h", "8h", "1d")
DIRECT_CHILD_LEVEL = {
    "15m": "1m",
    "30m": "15m",
    "1h": "30m",
    "2h": "1h",
    "4h": "2h",
    "8h": "4h",
    "1d": "8h",
}
WINDOWS_BY_LEVEL = {
    "1d": (30, 90, 365, 730),
    "8h": (30, 90, 365),
    "4h": (7, 30, 90, 365),
    "2h": (7, 30, 90),
    "1h": (7, 30, 90),
    "30m": (7, 30, 90),
    "15m": (7, 30),
    "1m": (7, 30),
}
ONLINE_HISTORY_DAYS = {
    "1d": 90,
    "8h": 90,
    "4h": 90,
    "2h": 60,
    "1h": 30,
    "30m": 14,
    "15m": 14,
    "1m": 7,
}
PARENT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MINIMUM_RECONCILIATION_GAIN = 0.002
KNOWN_HISTOGRAM_LEVELS = ("1m", "15m", "1h", "4h", "1d")
ACF_LAGS = (1, 2, 5, 15, 60, 300, 900)
GAP_CAP_SECONDS = 300
SIGN_RHO_CANDIDATES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97)
SIGN_CALIBRATION_DAYS = 14
MAGNITUDE_RHO_CANDIDATES = (
    0.135,
    0.25,
    0.4,
    0.55,
    0.7,
    0.82,
    0.9,
    0.96,
    0.98,
    0.99,
    0.995,
    0.999,
)
VOLATILITY_COPULA_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
VOLATILITY_ACF_LAGS_MINUTES = (1, 5, 15, 60, 300, 900)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis = read_json(repo / args.analysis)
    histograms = read_json(repo / args.histograms)
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    train_start = parse_time(args.train_start)
    forecast_start = parse_time(args.forecast_start)
    forecast_end = parse_time(args.forecast_end)
    calibration_days = args.calibration_days
    hierarchy_selection_days = args.hierarchy_selection_days
    if not train_start < forecast_start < forecast_end:
        raise ValueError("expected train-start < forecast-start < forecast-end")
    forecast_days = (forecast_end - forecast_start).days
    untouched_start = calibration_days + hierarchy_selection_days
    if calibration_days < 80 or hierarchy_selection_days < 40:
        raise ValueError("calibration and hierarchy-selection periods are too short")
    if forecast_days - untouched_start < 80:
        raise ValueError("nested selection must leave at least 80 untouched days")

    minute_returns, minute_variance, minute_counts = load_or_read_minute_history(
        repo=repo,
        cache_path=repo / args.minute_cache,
        source=source,
        train_start=train_start,
        end=forecast_end,
    )
    history_offset = (forecast_start - train_start).days * MINUTES_PER_DAY
    actual = actual_hierarchy(
        minute_returns[history_offset:],
        minute_variance[history_offset:],
        minute_counts[history_offset:],
    )

    method_fit_days = calibration_days // 2
    (
        calibrated,
        selected_windows,
        window_scores,
        selected_calibrators,
        calibrator_scores,
    ) = load_or_build_calibrated_forecasts(
        repo=repo,
        cache_path=repo / args.forecast_cache,
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        history_offset=history_offset,
        actual=actual,
        forecast_days=forecast_days,
        calibration_days=calibration_days,
        ensemble_size=args.ensemble_size,
        train_start=train_start,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )

    coherent_leaves, reconciliation = reconcile_all_features(
        actual,
        calibrated,
        selection_start=calibration_days,
        selection_end=untouched_start,
    )
    generator_fit_end = forecast_start + timedelta(days=untouched_start)
    fitted, intraday_calibration, fit_metadata = load_or_fit_second_kernel(
        repo=repo,
        cache_path=repo / args.fit_cache,
        source=source,
        train_start=train_start,
        fit_end=generator_fit_end,
        one_second_histogram=full_histogram(histograms, "1s"),
    )
    sign_start = untouched_start - SIGN_CALIBRATION_DAYS
    sign_files = selected_files(
        source,
        generator_fit_end - timedelta(days=SIGN_CALIBRATION_DAYS),
        generator_fit_end,
    )
    sign_previous = float(read_candle_column(
        selected_files(
            source,
            generator_fit_end - timedelta(days=SIGN_CALIBRATION_DAYS + 1),
            generator_fit_end - timedelta(days=SIGN_CALIBRATION_DAYS),
        )[-1],
        "close",
    )[-1])
    kernel_target = measure_actual_seconds(
        sign_files,
        sign_previous,
        histogram_edges(full_histogram(histograms, "1s")),
        acf_days=SIGN_CALIBRATION_DAYS,
    )
    sign_target_acf = kernel_target["acf"]
    sign_timing_rho, sign_calibration = calibrate_sign_timing_rho(
        fitted=fitted,
        calibration=intraday_calibration,
        minute_returns=actual["1m"]["periodReturnBps"][sign_start:untouched_start],
        minute_variance=actual["1m"][
            "oneSecondRealizedVarianceBpsSquared"
        ][sign_start:untouched_start],
        minute_counts=actual["1m"]["activeSeconds"][sign_start:untouched_start],
        target_acf=sign_target_acf,
    )
    intraday_calibration = {
        **intraday_calibration,
        "signTimingGaussianRho": sign_timing_rho,
    }
    fit_metadata["intradayCalibration"] = intraday_calibration
    fit_metadata["signTimingCalibration"] = sign_calibration
    (
        coherent_leaves["oneSecondRealizedVarianceBpsSquared"],
        volatility_copula,
    ) = calibrate_volatility_copula(
        coherent_leaves["oneSecondRealizedVarianceBpsSquared"],
        actual["1m"]["oneSecondRealizedVarianceBpsSquared"],
        fitted.volatility_factors,
        selection_start=calibration_days,
        selection_end=untouched_start,
    )
    reconciliation["oneSecondRealizedVarianceBpsSquared"][
        "volatilityCopula"
    ] = volatility_copula
    coherent_leaves, cross_feature_coupling = joint_couple_leaf_features(
        coherent_leaves
    )
    reconciliation["crossFeaturePathCoupling"] = cross_feature_coupling
    magnitude_timing_rho, magnitude_timing_calibration = (
        calibrate_magnitude_timing_rho(
            fitted=fitted,
            calibration=intraday_calibration,
            minute_returns=coherent_leaves["periodReturnBps"][
                sign_start:untouched_start
            ],
            minute_variance=coherent_leaves[
                "oneSecondRealizedVarianceBpsSquared"
            ][sign_start:untouched_start],
            minute_counts=coherent_leaves["activeSeconds"][
                sign_start:untouched_start
            ],
            target_acf=sign_target_acf,
            target_histogram_counts=kernel_target["histogramCounts"],
            histogram_edges_value=histogram_edges(full_histogram(histograms, "1s")),
            paths_per_day=2,
        )
    )
    intraday_calibration["magnitudeShareScoreRho"] = magnitude_timing_rho
    fit_metadata["intradayCalibration"] = intraday_calibration
    fit_metadata["magnitudeTimingCalibration"] = magnitude_timing_calibration
    coherent = {
        feature: hierarchy_from_leaves(coherent_leaves[feature])
        for feature in FEATURES
    }
    hierarchy_report = evaluate_target_forecasts(
        actual,
        calibrated,
        coherent,
        test_start=untouched_start,
    )
    untouched_files = selected_files(source, generator_fit_end, forecast_end)
    if len(untouched_files) != forecast_days - untouched_start:
        raise RuntimeError("untouched second-level holdout is incomplete")
    actual_second = measure_actual_seconds(
        untouched_files,
        float(read_candle_column(
            selected_files(
                source,
                generator_fit_end - timedelta(days=1),
                generator_fit_end,
            )[-1],
            "close",
        )[-1]),
        histogram_edges(full_histogram(histograms, "1s")),
        acf_days=args.acf_days,
    )

    independent_leaves = {
        feature: calibrated["1m"][feature][untouched_start:].astype(np.float64)
        for feature in FEATURES
    }
    hierarchical_leaves = {
        feature: coherent_leaves[feature][untouched_start:].astype(np.float64)
        for feature in FEATURES
    }
    second_models = {}
    for model_index, (model_id, leaves) in enumerate((
        ("independentMinute", independent_leaves),
        ("hierarchicalCoherent", hierarchical_leaves),
    )):
        second_models[model_id] = simulate_second_holdout(
            fitted=fitted,
            calibration=intraday_calibration,
            leaves=leaves,
            one_second_edges=histogram_edges(full_histogram(histograms, "1s")),
            paths_per_day=args.second_paths_per_day,
            acf_days=args.acf_days,
            rng_seed=RNG_SEED + 50_000 * (model_index + 1),
        )

    distribution_report = evaluate_return_distributions(
        actual=actual,
        calibrated=calibrated,
        coherent=coherent,
        test_start=untouched_start,
        histograms=histograms,
        actual_second=actual_second,
        second_models=second_models,
    )
    second_report = evaluate_second_diagnostics(actual_second, second_models)
    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "design": {
            "historyStart": iso(train_start),
            "forecastOriginStart": iso(forecast_start),
            "forecastOriginEndExclusive": iso(forecast_end),
            "forecastOrigins": forecast_days,
            "calibrationOrigins": calibration_days,
            "hierarchyArchitectureSelectionOrigins": hierarchy_selection_days,
            "hierarchyArchitectureSelectionStart": iso(
                forecast_start + timedelta(days=calibration_days)
            ),
            "untouchedTestOrigins": forecast_days - untouched_start,
            "untouchedTestStart": iso(generator_fit_end),
            "ensembleSize": args.ensemble_size,
            "secondPathsPerTestDay": args.second_paths_per_day,
            "levels": {
                level: {
                    "minutes": minutes,
                    "nodesPerDay": MINUTES_PER_DAY // minutes,
                }
                for level, minutes in LEVEL_MINUTES.items()
            },
            "futureDataUsedAtForecastTime": False,
            "untouchedOutcomesUsedForSelection": False,
            "historicalCandlesResampledDuringGeneration": False,
            "oneSecondKernelFitEndExclusive": iso(generator_fit_end),
            "reconciliationInformationFlow": (
                "Each independently forecast parent is rank-coupled to the coherent child "
                "sum. A validation-selected parent weight adjusts the minute leaves, so "
                "all levels remain exact sums of the same leaf path. Weight zero is the "
                "automatic fallback."
            ),
            "marginalCalibrationCopula": (
                "Calibrated marginal quantiles are reassigned to each row using the raw "
                "ensemble's member ranks. This preserves each simulated path's temporal "
                "copula instead of aligning low/high quantiles across every candle."
            ),
            "oneSecondGeneration": (
                "A fitted conditional activity/magnitude/sign kernel generates fresh "
                "seconds and projects them to the reconciled minute return, variance, "
                "and integer activity targets."
            ),
        },
        "selectedHistoryWindows": selected_windows,
        "historyWindowSelection": window_scores,
        "selectedCalibrationMethods": selected_calibrators,
        "calibrationMethodSelection": calibrator_scores,
        "hierarchicalReconciliation": reconciliation,
        "targetForecastAudit": hierarchy_report,
        "oneSecondKernel": fit_metadata,
        "returnDistributionAudit": distribution_report,
        "secondProcessAudit": second_report,
        "acceptance": build_acceptance(hierarchy_report, distribution_report, second_report),
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(compact_summary(report), indent=2))


def actual_hierarchy(
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    if minute_returns.size % MINUTES_PER_DAY:
        raise ValueError("minute history must contain complete UTC days")
    days = minute_returns.size // MINUTES_PER_DAY
    result: dict[str, dict[str, np.ndarray]] = {}
    for level, minutes in LEVEL_MINUTES.items():
        nodes = MINUTES_PER_DAY // minutes
        shape = (days, nodes, minutes)
        result[level] = {
            "periodReturnBps": minute_returns.reshape(shape).sum(axis=2),
            "oneSecondRealizedVarianceBpsSquared": minute_variance.reshape(shape).sum(axis=2),
            "activeSeconds": minute_counts.reshape(shape).sum(axis=2).astype(np.float64),
        }
    return result


def load_or_read_minute_history(
    *,
    repo: Path,
    cache_path: Path,
    source: Path,
    train_start: datetime,
    end: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = {
        "trainStart": iso(train_start),
        "endExclusive": iso(end),
    }
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadataJson"].item()))
            if all(metadata.get(key) == value for key, value in expected.items()):
                print("Loading cached minute history...", flush=True)
                return (
                    cache["minuteReturns"].astype(np.float64),
                    cache["minuteVariance"].astype(np.float64),
                    cache["minuteCounts"].astype(np.uint8),
                )
    prior_files = selected_files(
        source,
        train_start - timedelta(days=1),
        train_start,
    )
    files = selected_files(source, train_start, end)
    if len(files) != (end - train_start).days:
        raise RuntimeError("one or more complete daily one-second shards are missing")
    previous_close = (
        float(read_candle_column(prior_files[-1], "close")[-1])
        if prior_files
        else float(read_candle_column(files[0], "close")[0])
    )
    values = read_minute_history(files, previous_close)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        metadataJson=json.dumps(expected),
        minuteReturns=values[0],
        minuteVariance=values[1],
        minuteCounts=values[2],
    )
    return values


def load_or_build_calibrated_forecasts(
    *,
    repo: Path,
    cache_path: Path,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    history_offset: int,
    actual: dict,
    forecast_days: int,
    calibration_days: int,
    ensemble_size: int,
    train_start: datetime,
    forecast_start: datetime,
    forecast_end: datetime,
) -> tuple[dict, dict, dict, dict, dict]:
    expected = {
        "version": 2,
        "trainStart": iso(train_start),
        "forecastStart": iso(forecast_start),
        "forecastEndExclusive": iso(forecast_end),
        "forecastDays": forecast_days,
        "calibrationDays": calibration_days,
        "ensembleSize": ensemble_size,
        "windowsByLevel": {key: list(value) for key, value in WINDOWS_BY_LEVEL.items()},
    }
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadataJson"].item()))
            if all(metadata.get(key) == value for key, value in expected.items()):
                print("Loading cached calibrated hierarchy forecasts...", flush=True)
                calibrated = {
                    level: {
                        feature: cache[f"calibrated__{level}__{feature}"]
                        for feature in FEATURES
                    }
                    for level in LEVEL_MINUTES
                }
                return (
                    calibrated,
                    metadata["selectedHistoryWindows"],
                    metadata["historyWindowSelection"],
                    metadata["selectedCalibrationMethods"],
                    metadata["calibrationMethodSelection"],
                )
    selected_windows, window_scores = select_history_windows(
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        history_offset=history_offset,
        actual=actual,
        forecast_days=forecast_days,
        calibration_days=calibration_days,
        ensemble_size=ensemble_size,
    )
    raw = generate_selected_forecasts(
        minute_returns=minute_returns,
        minute_variance=minute_variance,
        minute_counts=minute_counts,
        history_offset=history_offset,
        forecast_days=forecast_days,
        selected_windows=selected_windows,
        ensemble_size=ensemble_size,
    )
    fit_days = calibration_days // 2
    selected_calibrators, calibrator_scores = select_calibration_methods(
        actual,
        raw,
        fit_days=fit_days,
        selection_end=calibration_days,
        ensemble_size=ensemble_size,
    )
    calibrated = online_calibrate_forecasts(
        actual,
        raw,
        selected_calibrators,
        start=fit_days,
        ensemble_size=ensemble_size,
    )
    metadata = {
        **expected,
        "selectedHistoryWindows": selected_windows,
        "historyWindowSelection": window_scores,
        "selectedCalibrationMethods": selected_calibrators,
        "calibrationMethodSelection": calibrator_scores,
    }
    arrays = {
        f"calibrated__{level}__{feature}": calibrated[level][feature]
        for level in LEVEL_MINUTES
        for feature in FEATURES
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, metadataJson=json.dumps(metadata), **arrays)
    return (
        calibrated,
        selected_windows,
        window_scores,
        selected_calibrators,
        calibrator_scores,
    )


def deterministic_draws(
    *,
    day: int,
    level: str,
    paths: int,
    steps: int,
) -> dict[str, np.ndarray]:
    level_index = tuple(LEVEL_MINUTES).index(level)
    rng = np.random.default_rng(RNG_SEED + level_index * 100_003 + day)
    return {
        name: rng.standard_t(8.0, (paths, steps))
        for name in ("variance", "return", "activity")
    }


def generate_level_forecast(
    *,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    history_offset: int,
    forecast_days: int,
    level: str,
    window: int,
    ensemble_size: int,
) -> dict[str, np.ndarray]:
    block_minutes = LEVEL_MINUTES[level]
    steps = MINUTES_PER_DAY // block_minutes
    result = {
        feature: np.empty((forecast_days, steps, ensemble_size), dtype=np.float32)
        for feature in FEATURES
    }
    for day in range(forecast_days):
        history_end = history_offset + day * MINUTES_PER_DAY
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
            deterministic_draws(
                day=day,
                level=level,
                paths=ensemble_size,
                steps=steps,
            ),
        )
        result["periodReturnBps"][day] = targets["return"].T
        result["oneSecondRealizedVarianceBpsSquared"][day] = targets["variance"].T
        result["activeSeconds"][day] = (
            targets["activeFraction"].T * block_minutes * 60.0
        )
    return result


def select_history_windows(
    *,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    history_offset: int,
    actual: dict,
    forecast_days: int,
    calibration_days: int,
    ensemble_size: int,
) -> tuple[dict, dict]:
    selected = {level: {} for level in LEVEL_MINUTES}
    scores = {level: {feature: {} for feature in FEATURES} for level in LEVEL_MINUTES}
    for level in LEVEL_MINUTES:
        for window in WINDOWS_BY_LEVEL[level]:
            print(f"Window audit {level} {window}d...", flush=True)
            forecast = generate_level_forecast(
                minute_returns=minute_returns,
                minute_variance=minute_variance,
                minute_counts=minute_counts,
                history_offset=history_offset,
                forecast_days=calibration_days,
                level=level,
                window=window,
                ensemble_size=ensemble_size,
            )
            for feature in FEATURES:
                metrics = fast_ensemble_metrics(
                    actual[level][feature][:calibration_days].reshape(-1),
                    forecast[feature].reshape(-1, ensemble_size),
                )
                scores[level][feature][str(window)] = metrics
            del forecast
        for feature in FEATURES:
            selected[level][feature] = min(
                WINDOWS_BY_LEVEL[level],
                key=lambda value: scores[level][feature][str(value)]["selectionScore"],
            )
    return selected, scores


def generate_selected_forecasts(
    *,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    history_offset: int,
    forecast_days: int,
    selected_windows: dict,
    ensemble_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    result = {level: {} for level in LEVEL_MINUTES}
    for level in LEVEL_MINUTES:
        for window in sorted(set(selected_windows[level].values())):
            print(f"Selected forecast {level} {window}d...", flush=True)
            forecast = generate_level_forecast(
                minute_returns=minute_returns,
                minute_variance=minute_variance,
                minute_counts=minute_counts,
                history_offset=history_offset,
                forecast_days=forecast_days,
                level=level,
                window=window,
                ensemble_size=ensemble_size,
            )
            for feature in FEATURES:
                if selected_windows[level][feature] == window:
                    result[level][feature] = forecast[feature]
            del forecast
    return result


def fast_ensemble_metrics(actual: np.ndarray, ensemble: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    ensemble = np.asarray(ensemble, dtype=np.float64)
    sorted_values = np.sort(ensemble, axis=1)
    members = ensemble.shape[1]
    first = np.mean(np.abs(ensemble - actual[:, None]), axis=1)
    coefficients = 2.0 * np.arange(1, members + 1) - members - 1.0
    second = np.sum(sorted_values * coefficients[None, :], axis=1) / (members * members)
    crps = float(np.mean(first - second))
    scale = max(float(np.std(actual)), 1e-12)
    errors = []
    for coverage in (0.5, 0.8, 0.9):
        tail = 0.5 * (1.0 - coverage)
        lower = np.quantile(ensemble, tail, axis=1)
        upper = np.quantile(ensemble, 1.0 - tail, axis=1)
        errors.append(abs(float(np.mean((actual >= lower) & (actual <= upper))) - coverage))
    normalized = crps / scale
    coverage_mae = float(np.mean(errors))
    return {
        "meanCrps": crps,
        "normalizedCrpsByActualStd": normalized,
        "coverageMeanAbsoluteError": coverage_mae,
        "maximumAbsoluteCoverageError": float(max(errors)),
        "selectionScore": normalized + coverage_mae,
    }


def calibration_representation(
    level: str,
    feature: str,
    actual: np.ndarray,
    ensemble: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    if feature == "activeSeconds":
        scale = LEVEL_MINUTES[level] * 60.0
        return "activeSecondFraction", actual / scale, ensemble / scale
    return feature, actual, ensemble


def select_calibration_methods(
    actual: dict,
    raw: dict,
    *,
    fit_days: int,
    selection_end: int,
    ensemble_size: int,
) -> tuple[dict, dict]:
    selected = {level: {} for level in LEVEL_MINUTES}
    scores = {level: {feature: {} for feature in FEATURES} for level in LEVEL_MINUTES}
    for level in LEVEL_MINUTES:
        print(f"Calibration selection {level}...", flush=True)
        for feature in FEATURES:
            model_feature, fit_actual, fit_ensemble = calibration_representation(
                level,
                feature,
                actual[level][feature][:fit_days],
                raw[level][feature][:fit_days],
            )
            _, selection_actual, selection_ensemble = calibration_representation(
                level,
                feature,
                actual[level][feature][fit_days:selection_end],
                raw[level][feature][fit_days:selection_end],
            )
            fit_actual = fit_actual.reshape(-1)
            fit_ensemble = fit_ensemble.reshape(-1, ensemble_size)
            selection_actual = selection_actual.reshape(-1)
            selection_ensemble = selection_ensemble.reshape(-1, ensemble_size)
            for method in CANDIDATE_METHODS:
                if method == "raw":
                    prediction = selection_ensemble
                else:
                    model = fit_calibrator(model_feature, fit_actual, fit_ensemble, method)
                    prediction, _ = calibrated_ensemble(
                        model_feature,
                        selection_ensemble,
                        model,
                        ensemble_size,
                    )
                scores[level][feature][method] = fast_ensemble_metrics(
                    selection_actual,
                    prediction,
                )
            selected[level][feature] = min(
                CANDIDATE_METHODS,
                key=lambda method: scores[level][feature][method]["selectionScore"],
            )
    return selected, scores


def online_calibrate_forecasts(
    actual: dict,
    raw: dict,
    selected_methods: dict,
    *,
    start: int,
    ensemble_size: int,
) -> dict:
    result = {level: {} for level in LEVEL_MINUTES}
    for level in LEVEL_MINUTES:
        print(f"Walk-forward calibration {level}...", flush=True)
        for feature in FEATURES:
            source = raw[level][feature]
            method = selected_methods[level][feature]
            output = source.copy()
            if method != "raw":
                model_feature = calibration_representation(
                    level,
                    feature,
                    actual[level][feature][:1],
                    source[:1],
                )[0]
                for origin in range(start, source.shape[0]):
                    history_start = max(0, origin - ONLINE_HISTORY_DAYS[level])
                    _, history_actual, history_ensemble = calibration_representation(
                        level,
                        feature,
                        actual[level][feature][history_start:origin],
                        source[history_start:origin],
                    )
                    model = fit_calibrator(
                        model_feature,
                        history_actual.reshape(-1),
                        history_ensemble.reshape(-1, ensemble_size),
                        method,
                    )
                    _, _, current = calibration_representation(
                        level,
                        feature,
                        actual[level][feature][origin:origin + 1],
                        source[origin:origin + 1],
                    )
                    prediction, _ = calibrated_ensemble(
                        model_feature,
                        current.reshape(-1, ensemble_size),
                        model,
                        ensemble_size,
                    )
                    prediction = restore_member_ranks(
                        current.reshape(-1, ensemble_size),
                        prediction,
                    )
                    prediction = prediction.reshape(source.shape[1], ensemble_size)
                    if feature == "activeSeconds":
                        prediction *= LEVEL_MINUTES[level] * 60.0
                        prediction = np.clip(
                            prediction,
                            0.0,
                            LEVEL_MINUTES[level] * 60.0,
                        )
                    output[origin] = prediction
            result[level][feature] = output.astype(np.float32)
    return result


def restore_member_ranks(raw: np.ndarray, calibrated: np.ndarray) -> np.ndarray:
    """Put calibrated quantiles back on each row's original path-member ranks."""
    raw = np.asarray(raw, dtype=np.float64)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    if raw.shape != calibrated.shape:
        raise ValueError("raw and calibrated ensembles must have equal shape")
    order = np.argsort(raw, axis=1)
    result = np.empty_like(calibrated)
    np.put_along_axis(result, order, np.sort(calibrated, axis=1), axis=1)
    return result


def match_leaf_marginals(
    dependence_source: np.ndarray,
    marginal_source: np.ndarray,
) -> np.ndarray:
    """Keep one leaf ensemble's path ranks and another's row marginals."""
    shape = dependence_source.shape
    matched = restore_member_ranks(
        np.asarray(dependence_source).reshape(-1, shape[-1]),
        np.asarray(marginal_source).reshape(-1, shape[-1]),
    )
    return matched.reshape(shape).astype(np.float32)


def gaussian_member_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    members = values.shape[-1]
    order = np.argsort(values, axis=-1)
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(members), values.shape),
        axis=-1,
    )
    probabilities = (ranks.astype(np.float64) + 0.5) / members
    return stats.norm.ppf(probabilities)


def factor_copula_scores(
    fit: FactorMixture,
    *,
    days: int,
    steps: int,
    members: int,
    seed: int,
) -> np.ndarray:
    result = np.empty((days, steps, members), dtype=np.float64)
    phi = np.exp(-1.0 / fit.timescales)
    innovation_scale = np.sqrt(1.0 - phi * phi)
    observation_weights = np.sqrt(np.maximum(fit.weights, 0.0))
    for day in range(days):
        rng = np.random.default_rng(seed + day)
        state = rng.standard_normal((members, fit.timescales.size))
        for step in range(steps):
            state = (
                phi[None, :] * state
                + innovation_scale[None, :]
                * rng.standard_normal((members, fit.timescales.size))
            )
            result[day, step] = state @ observation_weights
            if fit.white_variance > 0:
                result[day, step] += math.sqrt(fit.white_variance) * rng.standard_normal(members)
    return result


def segmented_correlation(values: np.ndarray, lag: int) -> float:
    if values.ndim == 2:
        left = values[:, :-lag].reshape(-1)
        right = values[:, lag:].reshape(-1)
    elif values.ndim == 3:
        left = values[:, :-lag, :].reshape(-1)
        right = values[:, lag:, :].reshape(-1)
    else:
        raise ValueError("segmented correlation expects day/time[/member] axes")
    return float(np.corrcoef(left, right)[0, 1])


def volatility_acf_error(actual: np.ndarray, ensemble: np.ndarray) -> dict:
    actual_log = np.log(np.maximum(np.asarray(actual, dtype=np.float64), 1e-12))
    ensemble_log = np.log(np.maximum(np.asarray(ensemble, dtype=np.float64), 1e-12))
    observed = {
        str(lag): segmented_correlation(actual_log, lag)
        for lag in VOLATILITY_ACF_LAGS_MINUTES
    }
    predicted = {
        str(lag): segmented_correlation(ensemble_log, lag)
        for lag in VOLATILITY_ACF_LAGS_MINUTES
    }
    return {
        "meanAbsoluteError": float(np.mean([
            abs(observed[str(lag)] - predicted[str(lag)])
            for lag in VOLATILITY_ACF_LAGS_MINUTES
        ])),
        "actual": observed,
        "predicted": predicted,
    }


def calibrate_volatility_copula(
    leaves: np.ndarray,
    actual: np.ndarray,
    factor_fit: FactorMixture,
    *,
    selection_start: int,
    selection_end: int,
) -> tuple[np.ndarray, dict]:
    latent = factor_copula_scores(
        factor_fit,
        days=leaves.shape[0],
        steps=leaves.shape[1],
        members=leaves.shape[2],
        seed=RNG_SEED + 1_100_000,
    )
    existing = gaussian_member_ranks(leaves)
    candidates = {}
    scores = {}
    folds = chronological_folds(selection_start, selection_end)
    for weight in VOLATILITY_COPULA_WEIGHTS:
        dependence = math.sqrt(max(0.0, 1.0 - weight * weight)) * existing + weight * latent
        candidate = match_leaf_marginals(dependence, leaves)
        fold_values = []
        for fold_start, fold_end in folds:
            acf = volatility_acf_error(
                actual[fold_start:fold_end],
                candidate[fold_start:fold_end],
            )
            distribution = subtree_score(
                candidate,
                {"1m": {"oneSecondRealizedVarianceBpsSquared": actual}},
                "oneSecondRealizedVarianceBpsSquared",
                parent_level="1m",
                start=fold_start,
                end=fold_end,
            )
            fold_values.append({
                "acfMeanAbsoluteError": acf["meanAbsoluteError"],
                "minuteSelectionScore": distribution,
                "combinedScore": distribution + acf["meanAbsoluteError"],
            })
        scores[str(weight)] = fold_values
        candidates[weight] = candidate
    stable = [0.0]
    for weight in VOLATILITY_COPULA_WEIGHTS[1:]:
        if all(
            row["combinedScore"] < baseline["combinedScore"]
            for row, baseline in zip(scores[str(weight)], scores["0.0"])
        ):
            stable.append(weight)
    selected = min(
        stable,
        key=lambda weight: np.mean([
            row["combinedScore"] for row in scores[str(weight)]
        ]),
    )
    test_diagnostic = volatility_acf_error(
        actual[selection_end:],
        candidates[selected][selection_end:],
    )
    return candidates[selected], {
        "selectedWeight": selected,
        "stableCandidateWeights": stable,
        "selectionFolds": [list(value) for value in folds],
        "candidateFoldScores": scores,
        "untouchedTestDiagnostic": test_diagnostic,
        "futureDataUsedForSelection": False,
        "marginalForecastsChanged": False,
    }


def target_feasibility_cost(
    returns: np.ndarray,
    variance: np.ndarray,
    activity: np.ndarray,
    *,
    block_minutes: int = 15,
) -> float:
    counts = np.clip(np.rint(activity), 0.0, 60.0)
    available = np.sqrt(np.maximum(counts * variance, 0.0))
    needed = np.abs(returns.reshape(-1, block_minutes).sum(axis=1))
    capacity = available.reshape(-1, block_minutes).sum(axis=1)
    ratio = np.divide(
        needed,
        capacity,
        out=np.full_like(needed, 1e6),
        where=capacity > 1e-12,
    )
    violation = np.maximum(np.log(np.maximum(ratio, 1.0)), 0.0)
    return float(np.mean(violation * violation))


def joint_couple_leaf_features(
    leaves: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    """Match whole-day feature paths while preserving all feature marginals."""
    returns = leaves["periodReturnBps"].astype(np.float64)
    variance = leaves["oneSecondRealizedVarianceBpsSquared"].astype(np.float64)
    activity = leaves["activeSeconds"].astype(np.float64)
    days, _, members = returns.shape
    result_variance = np.empty_like(variance)
    result_activity = np.empty_like(activity)
    before = []
    after = []
    q_activity_rank_correlation = []
    for day in range(days):
        q_totals = np.sum(variance[day], axis=0)
        activity_totals = np.sum(activity[day], axis=0)
        q_order = np.argsort(q_totals)
        activity_order = np.argsort(activity_totals)
        activity_by_q = np.empty_like(activity[day])
        activity_by_q[:, q_order] = activity[day][:, activity_order]

        cost = np.empty((members, members), dtype=np.float64)
        for return_member in range(members):
            for q_member in range(members):
                cost[return_member, q_member] = target_feasibility_cost(
                    returns[day, :, return_member],
                    variance[day, :, q_member],
                    activity_by_q[:, q_member],
                )
        row_indexes, column_indexes = linear_sum_assignment(cost)
        assignment = np.empty(members, dtype=np.int64)
        assignment[row_indexes] = column_indexes
        result_variance[day] = variance[day][:, assignment]
        result_activity[day] = activity_by_q[:, assignment]
        before.extend(float(cost[index, index]) for index in range(members))
        after.extend(float(cost[index, assignment[index]]) for index in range(members))
        q_activity_rank_correlation.append(float(np.corrcoef(
            np.argsort(np.argsort(np.sum(result_variance[day], axis=0))),
            np.argsort(np.argsort(np.sum(result_activity[day], axis=0))),
        )[0, 1]))
    result = {
        **leaves,
        "oneSecondRealizedVarianceBpsSquared": result_variance.astype(np.float32),
        "activeSeconds": result_activity.astype(np.float32),
    }
    return result, {
        "method": "daily Hungarian assignment of variance/activity paths to return paths",
        "meanFeasibilityCostBefore": float(np.mean(before)),
        "meanFeasibilityCostAfter": float(np.mean(after)),
        "maximumFeasibilityCostAfter": float(np.max(after)),
        "meanDailyVarianceActivityRankCorrelation": float(np.mean(
            q_activity_rank_correlation
        )),
        "wholeDayFeaturePathsPermutedOnly": True,
        "marginalForecastsChanged": False,
        "futureOutcomesUsed": False,
    }


def rank_couple_parent(parent: np.ndarray, child_sum: np.ndarray) -> np.ndarray:
    if parent.shape != child_sum.shape:
        raise ValueError("parent and child aggregates must share shape")
    result = np.empty_like(parent)
    rows = parent.reshape(-1, parent.shape[-1])
    child_rows = child_sum.reshape(-1, child_sum.shape[-1])
    output = result.reshape(-1, result.shape[-1])
    for index in range(rows.shape[0]):
        child_order = np.argsort(child_rows[index])
        output[index, child_order] = np.sort(rows[index])
    return result


def aggregate_leaves(leaves: np.ndarray, level: str) -> np.ndarray:
    minutes = LEVEL_MINUTES[level]
    days, leaf_count, members = leaves.shape
    if leaf_count != MINUTES_PER_DAY:
        raise ValueError("expected one minute leaf per UTC-day minute")
    return leaves.reshape(
        days,
        MINUTES_PER_DAY // minutes,
        minutes,
        members,
    ).sum(axis=2)


def positive_allocate_rows(
    values: np.ndarray,
    targets: np.ndarray,
    cap: float | None,
) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    targets = np.maximum(np.asarray(targets, dtype=np.float64), 0.0)
    if cap is None:
        totals = np.sum(values, axis=1)
        return values * np.divide(
            targets,
            totals,
            out=np.zeros_like(targets),
            where=totals > 0,
        )[:, None]
    targets = np.minimum(targets, cap * values.shape[1])
    low = np.zeros(targets.shape, dtype=np.float64)
    high = np.maximum(
        np.divide(
            targets,
            np.maximum(np.min(values, axis=1), 1e-12),
        ),
        1.0,
    )
    for _ in range(48):
        middle = 0.5 * (low + high)
        total = np.sum(np.minimum(cap, values * middle[:, None]), axis=1)
        low = np.where(total < targets, middle, low)
        high = np.where(total < targets, high, middle)
    result = np.minimum(cap, values * (0.5 * (low + high))[:, None])
    residual = targets - np.sum(result, axis=1)
    room = cap - result
    room_total = np.sum(room, axis=1)
    result += room * np.divide(
        residual,
        room_total,
        out=np.zeros_like(residual),
        where=room_total > 1e-12,
    )[:, None]
    return np.clip(result, 0.0, cap)


def reconcile_leaf_groups(
    leaves: np.ndarray,
    parent_targets: np.ndarray,
    *,
    level: str,
    feature: str,
    return_weights: np.ndarray | None = None,
) -> np.ndarray:
    minutes = LEVEL_MINUTES[level]
    days, _, members = leaves.shape
    nodes = MINUTES_PER_DAY // minutes
    groups = (
        leaves.reshape(days, nodes, minutes, members)
        .transpose(0, 1, 3, 2)
        .reshape(-1, minutes)
    )
    targets = parent_targets.reshape(-1)
    if feature == "periodReturnBps":
        if return_weights is None:
            weights = np.ones_like(leaves, dtype=np.float64)
        else:
            weights = np.maximum(np.asarray(return_weights, dtype=np.float64), 0.0)
        weights = (
            weights.reshape(days, nodes, minutes, members)
            .transpose(0, 1, 3, 2)
            .reshape(-1, minutes)
        )
        totals = np.sum(weights, axis=1)
        weights = np.divide(
            weights,
            totals[:, None],
            out=np.full_like(weights, 1.0 / minutes),
            where=totals[:, None] > 0,
        )
        groups = groups + (targets - np.sum(groups, axis=1))[:, None] * weights
    else:
        cap = 60.0 if feature == "activeSeconds" else None
        groups = positive_allocate_rows(groups, targets, cap)
    return (
        groups.reshape(days, nodes, members, minutes)
        .transpose(0, 1, 3, 2)
        .reshape(days, MINUTES_PER_DAY, members)
    )


def reconcile_immediate_children(
    leaves: np.ndarray,
    parent_targets: np.ndarray,
    *,
    parent_level: str,
    feature: str,
) -> np.ndarray:
    """Adjust only direct child totals, preserving every child's inner shape."""
    child_level = DIRECT_CHILD_LEVEL[parent_level]
    parent_minutes = LEVEL_MINUTES[parent_level]
    child_minutes = LEVEL_MINUTES[child_level]
    ratio = parent_minutes // child_minutes
    days, _, members = leaves.shape
    child = aggregate_leaves(leaves, child_level)
    parent_nodes = MINUTES_PER_DAY // parent_minutes
    groups = (
        child.reshape(days, parent_nodes, ratio, members)
        .transpose(0, 1, 3, 2)
        .reshape(-1, ratio)
    )
    targets = parent_targets.reshape(-1)
    if feature == "periodReturnBps":
        child_targets = groups + (
            (targets - np.sum(groups, axis=1)) / ratio
        )[:, None]
    else:
        cap = child_minutes * 60.0 if feature == "activeSeconds" else None
        child_targets = positive_allocate_rows(groups, targets, cap)
    child_targets = (
        child_targets.reshape(days, parent_nodes, members, ratio)
        .transpose(0, 1, 3, 2)
        .reshape(days, MINUTES_PER_DAY // child_minutes, members)
    )
    return reconcile_leaf_groups(
        leaves,
        child_targets,
        level=child_level,
        feature=feature,
        return_weights=None,
    )


def descendant_levels(parent_level: str) -> tuple[str, ...]:
    parent_minutes = LEVEL_MINUTES[parent_level]
    return tuple(
        level for level, minutes in LEVEL_MINUTES.items()
        if minutes <= parent_minutes
    )


def subtree_score(
    leaves: np.ndarray,
    actual: dict,
    feature: str,
    *,
    parent_level: str,
    start: int,
    end: int,
) -> float:
    values = []
    for level in descendant_levels(parent_level):
        prediction = aggregate_leaves(leaves[start:end], level)
        metrics = fast_ensemble_metrics(
            actual[level][feature][start:end].reshape(-1),
            prediction.reshape(-1, prediction.shape[-1]),
        )
        values.append(metrics["selectionScore"])
    return float(np.mean(values))


def chronological_folds(start: int, end: int, count: int = 3) -> tuple[tuple[int, int], ...]:
    boundaries = np.linspace(start, end, count + 1).round().astype(int)
    return tuple(
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(count)
        if boundaries[index + 1] > boundaries[index]
    )


def select_and_apply_parent(
    leaves: np.ndarray,
    parent: np.ndarray,
    actual: dict,
    *,
    level: str,
    feature: str,
    selection_start: int,
    selection_end: int,
) -> tuple[np.ndarray, dict]:
    child_sum = aggregate_leaves(leaves, level)
    coupled = rank_couple_parent(parent.astype(np.float64), child_sum)
    scores = {}
    fold_scores = {}
    candidates = {}
    folds = chronological_folds(selection_start, selection_end)
    for weight in PARENT_WEIGHTS:
        targets = weight * coupled + (1.0 - weight) * child_sum
        candidate = reconcile_immediate_children(
            leaves,
            targets,
            parent_level=level,
            feature=feature,
        )
        score = subtree_score(
            candidate,
            actual,
            feature,
            parent_level=level,
            start=selection_start,
            end=selection_end,
        )
        scores[str(weight)] = score
        fold_scores[str(weight)] = [
            subtree_score(
                candidate,
                actual,
                feature,
                parent_level=level,
                start=fold_start,
                end=fold_end,
            )
            for fold_start, fold_end in folds
        ]
        candidates[weight] = candidate
    baseline = scores["0.0"]
    stable = [0.0]
    fold_gains = {"0.0": [0.0] * len(folds)}
    for weight in PARENT_WEIGHTS[1:]:
        gains = [
            1.0 - candidate_score / baseline_score
            for candidate_score, baseline_score in zip(
                fold_scores[str(weight)],
                fold_scores["0.0"],
            )
        ]
        fold_gains[str(weight)] = gains
        if min(gains) >= MINIMUM_RECONCILIATION_GAIN:
            stable.append(weight)
    best = min(stable, key=lambda value: scores[str(value)])
    relative_gain = 1.0 - scores[str(best)] / baseline if baseline > 0 else 0.0
    selected = best if relative_gain >= MINIMUM_RECONCILIATION_GAIN else 0.0
    return candidates[selected].astype(np.float32), {
        "selectedParentWeight": selected,
        "bestUngatedParentWeight": best,
        "relativeValidationScoreGain": relative_gain,
        "minimumRequiredGain": MINIMUM_RECONCILIATION_GAIN,
        "fallbackUsed": selected == 0.0,
        "validationScores": scores,
        "chronologicalFoldScores": fold_scores,
        "chronologicalFoldGainsVsFallback": fold_gains,
        "stableCandidateWeights": stable,
    }


def reconcile_all_features(
    actual: dict,
    calibrated: dict,
    *,
    selection_start: int,
    selection_end: int,
) -> tuple[dict, dict]:
    leaves_by_feature: dict[str, np.ndarray] = {}
    report = {feature: {} for feature in FEATURES}
    for feature in (
        "oneSecondRealizedVarianceBpsSquared",
        "activeSeconds",
        "periodReturnBps",
    ):
        print(f"Hierarchical conditioning {feature}...", flush=True)
        fallback_leaves = calibrated["1m"][feature].astype(np.float64)
        leaves = fallback_leaves
        for level in BOTTOM_UP_LEVELS:
            leaves, details = select_and_apply_parent(
                leaves,
                calibrated[level][feature],
                actual,
                level=level,
                feature=feature,
                selection_start=selection_start,
                selection_end=selection_end,
            )
            report[feature][level] = details
        folds = chronological_folds(selection_start, selection_end)
        fallback_scores = [
            subtree_score(
                fallback_leaves,
                actual,
                feature,
                parent_level="1d",
                start=fold_start,
                end=fold_end,
            )
            for fold_start, fold_end in folds
        ]
        candidate_scores = [
            subtree_score(
                leaves,
                actual,
                feature,
                parent_level="1d",
                start=fold_start,
                end=fold_end,
            )
            for fold_start, fold_end in folds
        ]
        gains = [
            1.0 - candidate / fallback
            for candidate, fallback in zip(candidate_scores, fallback_scores)
        ]
        minute_fallback_scores = [
            subtree_score(
                fallback_leaves,
                actual,
                feature,
                parent_level="1m",
                start=fold_start,
                end=fold_end,
            )
            for fold_start, fold_end in folds
        ]
        minute_candidate_scores = [
            subtree_score(
                leaves,
                actual,
                feature,
                parent_level="1m",
                start=fold_start,
                end=fold_end,
            )
            for fold_start, fold_end in folds
        ]
        minute_gains = [
            1.0 - candidate / fallback
            for candidate, fallback in zip(
                minute_candidate_scores,
                minute_fallback_scores,
            )
        ]
        full_passed = min(gains) >= MINIMUM_RECONCILIATION_GAIN
        minute_passed = min(minute_gains) >= -MINIMUM_RECONCILIATION_GAIN
        copula_leaves = match_leaf_marginals(leaves, fallback_leaves)
        copula_scores = [
            subtree_score(
                copula_leaves,
                actual,
                feature,
                parent_level="1d",
                start=fold_start,
                end=fold_end,
            )
            for fold_start, fold_end in folds
        ]
        copula_gains = [
            1.0 - candidate / fallback
            for candidate, fallback in zip(copula_scores, fallback_scores)
        ]
        if full_passed and minute_passed:
            mode = "fullParentConditioning"
        elif full_passed and min(copula_gains) >= MINIMUM_RECONCILIATION_GAIN:
            leaves = copula_leaves
            mode = "copulaOnlyParentConditioning"
        else:
            leaves = fallback_leaves.astype(np.float32)
            mode = "bottomUpMinuteFallback"
        feature_passed = mode != "bottomUpMinuteFallback"
        for level in BOTTOM_UP_LEVELS:
            report[feature][level]["effectiveParentWeight"] = (
                report[feature][level]["selectedParentWeight"]
                if feature_passed
                else 0.0
            )
        report[feature]["featureGate"] = {
            "passed": feature_passed,
            "fallbackUsed": not feature_passed,
            "selectedMode": mode,
            "chronologicalFoldFallbackScores": fallback_scores,
            "chronologicalFoldCandidateScores": candidate_scores,
            "chronologicalFoldGains": gains,
            "minuteFoldFallbackScores": minute_fallback_scores,
            "minuteFoldCandidateScores": minute_candidate_scores,
            "minuteFoldGains": minute_gains,
            "copulaOnlyFoldScores": copula_scores,
            "copulaOnlyFoldGains": copula_gains,
            "minimumRequiredGainInEveryFold": MINIMUM_RECONCILIATION_GAIN,
        }
        leaves_by_feature[feature] = leaves
    return leaves_by_feature, report


def hierarchy_from_leaves(leaves: np.ndarray) -> dict[str, np.ndarray]:
    return {level: aggregate_leaves(leaves, level) for level in LEVEL_MINUTES}


def evaluate_target_forecasts(
    actual: dict,
    independent: dict,
    coherent: dict,
    *,
    test_start: int,
) -> dict:
    report = {feature: {"levels": {}} for feature in FEATURES}
    for feature in FEATURES:
        bottom_up = hierarchy_from_leaves(
            independent["1m"][feature].astype(np.float64)
        )
        for level in LEVEL_MINUTES:
            actual_values = actual[level][feature][test_start:].reshape(-1)
            base = independent[level][feature][test_start:].reshape(
                -1,
                independent[level][feature].shape[-1],
            )
            model = coherent[feature][level][test_start:].reshape(
                -1,
                coherent[feature][level].shape[-1],
            )
            base_metrics = ensemble_metrics(actual_values, base)
            bottom_up_values = bottom_up[level][test_start:].reshape(
                -1,
                bottom_up[level].shape[-1],
            )
            bottom_up_metrics = ensemble_metrics(actual_values, bottom_up_values)
            model_metrics = ensemble_metrics(actual_values, model)
            report[feature]["levels"][level] = {
                "independentScaleSpecificOracle": base_metrics,
                "bottomUpMinuteCoherentFallback": bottom_up_metrics,
                "hierarchical": model_metrics,
                "hierarchicalCrpsSkillVsScaleSpecificOracle": (
                    1.0 - model_metrics["meanCrps"] / base_metrics["meanCrps"]
                ),
                "hierarchicalCrpsSkillVsBottomUpFallback": (
                    1.0
                    - model_metrics["meanCrps"] / bottom_up_metrics["meanCrps"]
                ),
                "hierarchicalMaximumCoverageError": maximum_coverage_error(model_metrics),
            }
        actual_nodes = np.concatenate(
            [actual[level][feature][test_start:] for level in LEVEL_MINUTES],
            axis=1,
        )
        base_nodes = np.concatenate(
            [independent[level][feature][test_start:] for level in LEVEL_MINUTES],
            axis=1,
        )
        model_nodes = np.concatenate(
            [coherent[feature][level][test_start:] for level in LEVEL_MINUTES],
            axis=1,
        )
        scales = np.concatenate([
            np.full(
                MINUTES_PER_DAY // LEVEL_MINUTES[level],
                max(float(np.std(actual[level][feature][:test_start])), 1e-9),
            )
            for level in LEVEL_MINUTES
        ])
        base_energy = energy_score(actual_nodes, base_nodes, scales)
        bottom_up_nodes = np.concatenate(
            [bottom_up[level][test_start:] for level in LEVEL_MINUTES],
            axis=1,
        )
        bottom_up_energy = energy_score(actual_nodes, bottom_up_nodes, scales)
        model_energy = energy_score(actual_nodes, model_nodes, scales)
        report[feature]["allLevelJointEnergy"] = {
            "independentScaleSpecificOracle": base_energy,
            "bottomUpMinuteCoherentFallback": bottom_up_energy,
            "hierarchical": model_energy,
            "hierarchicalSkillVsScaleSpecificOracle": 1.0 - model_energy / base_energy,
            "hierarchicalSkillVsBottomUpFallback": 1.0 - model_energy / bottom_up_energy,
        }
        report[feature]["maximumCoherenceError"] = hierarchy_coherence_error(
            coherent[feature]["1m"][test_start:],
            coherent[feature],
            start=test_start,
        )
    return report


def energy_score(actual: np.ndarray, ensemble: np.ndarray, scales: np.ndarray) -> float:
    actual_normalized = actual / scales[None, :]
    ensemble_normalized = ensemble / scales[None, :, None]
    first = np.mean(
        np.linalg.norm(ensemble_normalized - actual_normalized[:, :, None], axis=1),
        axis=1,
    )
    second = np.empty(actual.shape[0], dtype=np.float64)
    for day in range(actual.shape[0]):
        members = ensemble_normalized[day].T
        differences = members[:, None, :] - members[None, :, :]
        second[day] = 0.5 * float(np.mean(np.linalg.norm(differences, axis=2)))
    return float(np.mean(first - second))


def hierarchy_coherence_error(
    leaves: np.ndarray,
    hierarchy: dict[str, np.ndarray],
    *,
    start: int,
) -> float:
    error = 0.0
    selected_leaves = leaves
    for level in LEVEL_MINUTES:
        expected = aggregate_leaves(selected_leaves, level)
        error = max(error, float(np.max(np.abs(expected - hierarchy[level][start:]))))
    return error


def maximum_coverage_error(metrics: dict) -> float:
    return float(max(
        abs(values["coverageError"])
        for values in metrics["centralIntervals"].values()
    ))


def serialize_factor(value: dict) -> FactorMixture:
    return FactorMixture(
        timescales=np.asarray(value["timescales"], dtype=np.float64),
        weights=np.asarray(value["weights"], dtype=np.float64),
        white_variance=float(value["whiteVariance"]),
        target_lags=np.asarray(value["targetLags"], dtype=np.int64),
        target_covariance=np.asarray(value["targetCovariance"], dtype=np.float64),
        fitted_covariance=np.asarray(value["fittedCovariance"], dtype=np.float64),
    )


def deserialize_fit(value: dict) -> FittedProcess:
    variance = value["varianceQuantileSpline"]
    seasonal = value["volatilitySeasonal"]
    daily_variance = value["dailyVarianceBudget"]
    daily_return = value["dailyReturnTarget"]
    magnitude = value["conditionalMagnitudeMixture"]
    return FittedProcess(
        variance_floor=float(value["varianceFloorBpsSquared"]),
        variance_quantile_knots=np.asarray(variance["probabilities"], dtype=np.float64),
        variance_log_quantiles=np.asarray(variance["logVarianceQuantiles"], dtype=np.float64),
        volatility_seasonal=SeasonalFit(
            coefficients=np.asarray(seasonal["coefficients"], dtype=np.float64),
            daily_harmonics=int(seasonal["dailyHarmonics"]),
            weekly_harmonics=int(seasonal["weeklyHarmonics"]),
        ),
        volatility_factors=serialize_factor(value["volatilityFactors"]),
        daily_variance=DailyVarianceFit(
            floor=float(daily_variance["floorBpsSquared"]),
            quantile_knots=np.asarray(
                daily_variance["quantileSpline"]["probabilities"], dtype=np.float64
            ),
            log_quantiles=np.asarray(
                daily_variance["quantileSpline"]["logVarianceQuantiles"], dtype=np.float64
            ),
            factors=serialize_factor(daily_variance["factors"]),
            target_mean=float(daily_variance["targetMeanBpsSquared"]),
        ),
        daily_return=DailyReturnFit(
            quantile_knots=np.asarray(
                daily_return["quantileSpline"]["probabilities"], dtype=np.float64
            ),
            return_quantiles=np.asarray(
                daily_return["quantileSpline"]["returnQuantilesBps"], dtype=np.float64
            ),
            factors=serialize_factor(daily_return["factors"]),
        ),
        activity_coefficients=np.asarray(value["activityVolatilityCoefficients"], dtype=np.float64),
        activity_residual_quantile_knots=np.asarray(
            value["activityResidualQuantileSpline"]["probabilities"], dtype=np.float64
        ),
        activity_residual_quantiles=np.asarray(
            value["activityResidualQuantileSpline"]["quantiles"], dtype=np.float64
        ),
        activity_factors=serialize_factor(value["activityFactors"]),
        activity_count_probabilities=np.asarray(value["activityCountProbabilities"], dtype=np.float64),
        target_adjacent_activity_probability=float(value["targetAdjacentActivityProbability"]),
        magnitude_mixture=ConditionalMagnitudeMixture(
            activity_edges=np.asarray(magnitude["activityBinUpperEdges"], dtype=np.float64),
            volatility_edges=np.asarray(magnitude["volatilityBinUpperEdges"], dtype=np.float64),
            component_weights=np.asarray(magnitude["componentWeights"], dtype=np.float64),
            log_concentration_means=np.asarray(magnitude["logConcentrationMeans"], dtype=np.float64),
            log_concentration_stds=np.asarray(magnitude["logConcentrationStds"], dtype=np.float64),
            share_score_rho=float(magnitude["shareScoreRho"]),
            micro_probabilities=np.asarray(magnitude["microProbabilities"], dtype=np.float64),
            micro_threshold_bps=float(magnitude["microThresholdBps"]),
        ),
        positive_probability=float(value["positiveSignProbability"]),
        efficiency_quantile_knots=np.asarray(
            value["efficiencyQuantileSpline"]["probabilities"], dtype=np.float64
        ),
        efficiency_quantiles=np.asarray(
            value["efficiencyQuantileSpline"]["quantiles"], dtype=np.float64
        ),
        efficiency_volatility_coefficients=np.asarray(
            value["efficiencyVolatilityCoefficients"], dtype=np.float64
        ),
        efficiency_residual_std=float(value["efficiencyResidualStd"]),
        efficiency_factors=serialize_factor(value["efficiencyFactors"]),
        target_mean_realized_variance=float(value["targetMeanRealizedVariance"]),
        target_minute_return_variance=float(value["targetMinuteReturnVariance"]),
        target_mean_activity_count=float(value["targetMeanActivityCount"]),
    )


def load_or_fit_second_kernel(
    *,
    repo: Path,
    cache_path: Path,
    source: Path,
    train_start: datetime,
    fit_end: datetime,
    one_second_histogram: dict,
) -> tuple[FittedProcess, dict, dict]:
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached.get("fitStart") == iso(train_start) and cached.get("fitEndExclusive") == iso(fit_end):
            return (
                deserialize_fit(cached["fittedParameters"]),
                cached["intradayCalibration"],
                {
                    "cache": str(cache_path.relative_to(repo)).replace("\\", "/"),
                    "fitStart": cached["fitStart"],
                    "fitEndExclusive": cached["fitEndExclusive"],
                    "fitDays": cached["fitDays"],
                    "loadedFromCache": True,
                    "intradayCalibration": cached["intradayCalibration"],
                },
            )
    print("Fitting causal one-second kernel...", flush=True)
    files = selected_files(source, train_start, fit_end)
    edges = histogram_edges(one_second_histogram)
    measurements = measure_source(
        files,
        micro_threshold_bps=0.5 * float(one_second_histogram["binWidthBps"]),
        one_second_histogram_edges=edges,
    )
    fitted = fit_process(measurements)
    training_histogram = histogram_from_counts(
        one_second_histogram,
        measurements.one_second_histogram_counts,
    )
    volatility_score = gaussianize_quantile_spline(
        np.log(measurements.realized_variance + fitted.variance_floor),
        fitted.variance_quantile_knots,
        fitted.variance_log_quantiles,
    )
    calibration = calibrate_intraday_generation(
        fitted,
        measurements,
        volatility_score,
        training_histogram,
        np.random.default_rng(RNG_SEED + 700),
    )
    payload = {
        "version": 1,
        "fitStart": iso(train_start),
        "fitEndExclusive": iso(fit_end),
        "fitDays": len(files),
        "futureDataUsed": False,
        "historicalCandlesResampledDuringGeneration": False,
        "fittedParameters": serialize_fit(fitted),
        "intradayCalibration": calibration,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "cache": str(cache_path.relative_to(repo)).replace("\\", "/"),
        "fitStart": iso(train_start),
        "fitEndExclusive": iso(fit_end),
        "fitDays": len(files),
        "loadedFromCache": False,
        "intradayCalibration": calibration,
    }
    del measurements, volatility_score
    gc.collect()
    return fitted, calibration, metadata


def round_activity_counts(values: np.ndarray, group_minutes: int = 15) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 60.0)
    if values.size % group_minutes:
        raise ValueError("activity vector must contain complete rounding groups")
    result = np.empty(values.size, dtype=np.uint8)
    for offset in range(0, values.size, group_minutes):
        block = values[offset:offset + group_minutes]
        base = np.floor(block).astype(np.int64)
        target = int(np.clip(round(float(np.sum(block))), 0, 60 * group_minutes))
        remaining = target - int(np.sum(base))
        if remaining > 0:
            order = np.argsort(-(block - base))
            for index in order:
                if remaining == 0:
                    break
                room = 60 - base[index]
                take = min(room, remaining)
                base[index] += take
                remaining -= take
        result[offset:offset + group_minutes] = base.astype(np.uint8)
    return result


def feasible_minute_targets(
    returns: np.ndarray,
    variance: np.ndarray,
    counts: np.ndarray,
    *,
    block_minutes: int = 15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    target_returns = np.asarray(returns, dtype=np.float64).copy()
    q = np.maximum(np.asarray(variance, dtype=np.float64), 0.0).copy()
    input_variance_total = float(np.sum(q))
    counts = np.asarray(counts, dtype=np.uint8)
    target_blocks = target_returns.reshape(-1, block_minutes).sum(axis=1)
    zero = counts == 0
    single = counts == 1
    q[zero] = 0.0
    target_returns[zero] = 0.0
    target_sign = np.repeat(
        np.where(target_blocks >= 0.0, 1.0, -1.0),
        block_minutes,
    )
    target_returns[single] = target_sign[single] * np.sqrt(q[single])
    adjustable = counts >= 2
    bounds = np.sqrt(counts.astype(np.float64) * q) * (1.0 - 1e-10)
    target_returns[adjustable] = np.clip(
        target_returns[adjustable],
        -bounds[adjustable],
        bounds[adjustable],
    )
    inflation_blocks = 0
    maximum_inflation = 1.0
    for block_index, target in enumerate(target_blocks):
        start = block_index * block_minutes
        stop = start + block_minutes
        block_counts = counts[start:stop]
        block_q = q[start:stop]
        block_adjustable = block_counts >= 2
        fixed = float(np.sum(target_returns[start:stop][~block_adjustable]))
        adjustable_bound = float(np.sum(np.sqrt(
            block_counts[block_adjustable].astype(np.float64)
            * block_q[block_adjustable]
        ))) * (1.0 - 1e-10)
        needed = abs(float(target) - fixed)
        if needed > adjustable_bound + 1e-10 and np.any(block_adjustable):
            factor = (needed / max(adjustable_bound, 1e-12)) ** 2 * (1.0 + 1e-9)
            indexes = np.arange(start, stop)[block_adjustable]
            q[indexes] *= factor
            inflation_blocks += 1
            maximum_inflation = max(maximum_inflation, factor)
    projected, fallbacks = project_returns_to_daily_targets(
        target_returns,
        q,
        counts,
        target_blocks,
        aligned=False,
        block_minutes=block_minutes,
    )
    achieved = projected.reshape(-1, block_minutes).sum(axis=1)
    return projected, q, {
        "projectionFallbacks": fallbacks,
        "varianceInflationBlocks": inflation_blocks,
        "maximumVarianceInflationFactor": maximum_inflation,
        "inputVarianceTotalBpsSquared": input_variance_total,
        "feasibleVarianceTotalBpsSquared": float(np.sum(q)),
        "addedVarianceBpsSquared": float(np.sum(q) - input_variance_total),
        "maximumBlockReturnTargetErrorBps": float(np.max(np.abs(achieved - target_blocks))),
    }


def zero_gap_counts(returns: np.ndarray, cap: int = GAP_CAP_SECONDS) -> np.ndarray:
    zero = np.asarray(returns) == 0.0
    padded = np.concatenate((np.asarray([False]), zero, np.asarray([False])))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    lengths = stops - starts
    return np.bincount(np.minimum(lengths, cap), minlength=cap + 1)


def measure_acf(files: list[Path], previous_close: float) -> dict[str, list[float]]:
    accumulator = AcfAccumulator.create()
    last_close = previous_close
    for reference in files:
        closes = read_candle_column(reference, "close")
        returns = np.diff(np.log(np.concatenate((
            np.asarray([last_close], dtype=np.float64),
            closes,
        )))) * 10_000.0
        last_close = float(closes[-1])
        accumulator.add_returns(returns)
    return accumulator.finish()


def calibrate_sign_timing_rho(
    *,
    fitted: FittedProcess,
    calibration: dict,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    target_acf: dict[str, list[float]],
) -> tuple[float, dict]:
    rows = minute_returns.shape[0]
    counts_by_day = np.rint(minute_counts).astype(np.uint8)
    scores = {}
    selected_lags = (1, 2, 5, 15)
    for rho_index, rho in enumerate(SIGN_RHO_CANDIDATES):
        accumulator = AcfAccumulator.create()
        for day in range(rows):
            q = np.maximum(minute_variance[day].astype(np.float64), 0.0)
            volatility_score = gaussianize_quantile_spline(
                np.log(q + fitted.variance_floor),
                fitted.variance_quantile_knots,
                fitted.variance_log_quantiles,
            )
            _, generated = generate_projected_second_batch(
                fitted,
                counts=counts_by_day[day],
                volatility_score=volatility_score,
                target_minute_returns=minute_returns[day].astype(np.float64),
                realized_variance=q,
                activity_timing_rho=float(calibration["activityTimingGaussianRho"]),
                magnitude_dispersion_scale=float(calibration["magnitudeDispersionScale"]),
                micro_probability_scale=float(calibration["microProbabilityScale"]),
                rng=np.random.default_rng(RNG_SEED + 800_000 + rho_index * 10_000 + day),
                sign_timing_rho=rho,
            )
            accumulator.add_returns(generated.reshape(-1))
        generated_acf = accumulator.finish()
        return_error = float(np.mean([
            abs(generated_acf["return"][lag] - target_acf["return"][lag])
            for lag in selected_lags
        ]))
        scores[str(rho)] = {
            "returnAcfMeanAbsoluteError": return_error,
            "generatedReturnAcf": {
                str(lag): float(generated_acf["return"][lag])
                for lag in selected_lags
            },
        }
    selected = min(
        SIGN_RHO_CANDIDATES,
        key=lambda rho: scores[str(rho)]["returnAcfMeanAbsoluteError"],
    )
    return selected, {
        "days": rows,
        "selectedRho": selected,
        "targetReturnAcf": {
            str(lag): float(target_acf["return"][lag])
            for lag in selected_lags
        },
        "candidateScores": scores,
        "futureDataUsed": False,
    }


def calibrate_magnitude_timing_rho(
    *,
    fitted: FittedProcess,
    calibration: dict,
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    target_acf: dict[str, list[float]],
    target_histogram_counts: np.ndarray,
    histogram_edges_value: np.ndarray,
    paths_per_day: int = 1,
) -> tuple[float, dict]:
    rows = minute_returns.shape[0]
    if minute_returns.ndim == 2:
        minute_returns = minute_returns[:, :, None]
        minute_variance = minute_variance[:, :, None]
        minute_counts = minute_counts[:, :, None]
    if minute_returns.ndim != 3:
        raise ValueError("magnitude calibration targets need day/minute[/member] axes")
    members = minute_returns.shape[2]
    selected_members = np.linspace(
        0,
        members - 1,
        min(paths_per_day, members),
    ).round().astype(int)
    scores = {}
    selected_lags = (1, 2, 5, 15)
    target_probability = probability(target_histogram_counts)
    candidates = tuple(sorted(set((
        float(fitted.magnitude_mixture.share_score_rho),
        *MAGNITUDE_RHO_CANDIDATES,
    ))))
    for rho_index, rho in enumerate(candidates):
        accumulator = AcfAccumulator.create()
        histogram = np.zeros(histogram_edges_value.size - 1, dtype=np.int64)
        for day in range(rows):
            for path_index, member in enumerate(selected_members):
                counts = round_activity_counts(minute_counts[day, :, member])
                target_return, q, _ = feasible_minute_targets(
                    minute_returns[day, :, member],
                    minute_variance[day, :, member],
                    counts,
                )
                volatility_score = gaussianize_quantile_spline(
                    np.log(q + fitted.variance_floor),
                    fitted.variance_quantile_knots,
                    fitted.variance_log_quantiles,
                )
                _, generated = generate_projected_second_batch(
                    fitted,
                    counts=counts,
                    volatility_score=volatility_score,
                    target_minute_returns=target_return,
                    realized_variance=q,
                    activity_timing_rho=float(calibration["activityTimingGaussianRho"]),
                    magnitude_dispersion_scale=float(calibration["magnitudeDispersionScale"]),
                    micro_probability_scale=float(calibration["microProbabilityScale"]),
                    rng=np.random.default_rng(
                        RNG_SEED
                        + 1_300_000
                        + rho_index * 100_000
                        + day * 100
                        + path_index
                    ),
                    sign_timing_rho=float(calibration["signTimingGaussianRho"]),
                    magnitude_share_score_rho=rho,
                )
                flat = generated.reshape(-1)
                accumulator.add_returns(flat)
                histogram += bounded_histogram_counts(flat, histogram_edges_value)
        generated_acf = accumulator.finish()
        absolute_error = float(np.mean([
            abs(
                generated_acf["absoluteReturn"][lag]
                - target_acf["absoluteReturn"][lag]
            )
            for lag in selected_lags
        ]))
        histogram_js = jensen_shannon_bits(
            probability(histogram),
            target_probability,
        )
        scores[str(rho)] = {
            "absoluteReturnAcfMeanAbsoluteError": absolute_error,
            "oneSecondHistogramJsBits": histogram_js,
            "selectionScore": absolute_error + 2.0 * histogram_js,
            "generatedAbsoluteReturnAcf": {
                str(lag): float(generated_acf["absoluteReturn"][lag])
                for lag in selected_lags
            },
        }
    selected = min(
        candidates,
        key=lambda rho: scores[str(rho)]["selectionScore"],
    )
    return selected, {
        "days": rows,
        "generatedPathsPerDay": int(selected_members.size),
        "selectedRho": selected,
        "targetAbsoluteReturnAcf": {
            str(lag): float(target_acf["absoluteReturn"][lag])
            for lag in selected_lags
        },
        "candidateScores": scores,
        "selectionScore": "absolute-return ACF MAE + 2 * one-second histogram JS bits",
        "futureDataUsed": False,
    }


def simulate_second_holdout(
    *,
    fitted: FittedProcess,
    calibration: dict,
    leaves: dict[str, np.ndarray],
    one_second_edges: np.ndarray,
    paths_per_day: int,
    acf_days: int,
    rng_seed: int,
) -> dict:
    days, minutes, members = leaves["periodReturnBps"].shape
    if minutes != MINUTES_PER_DAY:
        raise ValueError("one-second generation requires full-day minute leaves")
    selected_members = np.linspace(0, members - 1, paths_per_day).round().astype(int)
    histogram = np.zeros(one_second_edges.size - 1, dtype=np.int64)
    gaps = np.zeros(GAP_CAP_SECONDS + 1, dtype=np.int64)
    acf = AcfAccumulator.create()
    diagnostics = {
        "paths": 0,
        "projectionFallbacks": 0,
        "varianceInflationBlocks": 0,
        "maximumVarianceInflationFactor": 1.0,
        "inputVarianceTotalBpsSquared": 0.0,
        "feasibleVarianceTotalBpsSquared": 0.0,
        "addedVarianceBpsSquared": 0.0,
        "maximumBlockReturnTargetErrorBps": 0.0,
        "maximumMinuteReturnErrorBps": 0.0,
        "maximumMinuteVarianceErrorBpsSquared": 0.0,
        "maximumMinuteActivityError": 0,
    }
    for day in range(days):
        if day % 20 == 0:
            print(f"Second simulation {day}/{days}...", flush=True)
        for path_index, member in enumerate(selected_members):
            counts = round_activity_counts(leaves["activeSeconds"][day, :, member])
            minute_return, q, feasible = feasible_minute_targets(
                leaves["periodReturnBps"][day, :, member],
                leaves["oneSecondRealizedVarianceBpsSquared"][day, :, member],
                counts,
            )
            volatility_score = gaussianize_quantile_spline(
                np.log(q + fitted.variance_floor),
                fitted.variance_quantile_knots,
                fitted.variance_log_quantiles,
            )
            rng = np.random.default_rng(rng_seed + day * 10_007 + path_index)
            active, second_returns = generate_projected_second_batch(
                fitted,
                counts=counts,
                volatility_score=volatility_score,
                target_minute_returns=minute_return,
                realized_variance=q,
                activity_timing_rho=float(calibration["activityTimingGaussianRho"]),
                magnitude_dispersion_scale=float(calibration["magnitudeDispersionScale"]),
                micro_probability_scale=float(calibration["microProbabilityScale"]),
                rng=rng,
                sign_timing_rho=float(calibration.get("signTimingGaussianRho", 0.0)),
                magnitude_share_score_rho=float(
                    calibration.get(
                        "magnitudeShareScoreRho",
                        fitted.magnitude_mixture.share_score_rho,
                    )
                ),
            )
            flat = second_returns.reshape(-1)
            histogram += bounded_histogram_counts(flat, one_second_edges)
            gaps += zero_gap_counts(flat)
            if day < acf_days and path_index == 0:
                acf.add_returns(flat)
            generated_return = np.sum(second_returns, axis=1)
            generated_q = np.sum(second_returns * second_returns, axis=1)
            generated_counts = np.sum(active, axis=1)
            diagnostics["paths"] += 1
            diagnostics["projectionFallbacks"] += feasible["projectionFallbacks"]
            diagnostics["varianceInflationBlocks"] += feasible["varianceInflationBlocks"]
            diagnostics["maximumVarianceInflationFactor"] = max(
                diagnostics["maximumVarianceInflationFactor"],
                feasible["maximumVarianceInflationFactor"],
            )
            diagnostics["inputVarianceTotalBpsSquared"] += feasible[
                "inputVarianceTotalBpsSquared"
            ]
            diagnostics["feasibleVarianceTotalBpsSquared"] += feasible[
                "feasibleVarianceTotalBpsSquared"
            ]
            diagnostics["addedVarianceBpsSquared"] += feasible[
                "addedVarianceBpsSquared"
            ]
            diagnostics["maximumBlockReturnTargetErrorBps"] = max(
                diagnostics["maximumBlockReturnTargetErrorBps"],
                feasible["maximumBlockReturnTargetErrorBps"],
            )
            diagnostics["maximumMinuteReturnErrorBps"] = max(
                diagnostics["maximumMinuteReturnErrorBps"],
                float(np.max(np.abs(generated_return - minute_return))),
            )
            diagnostics["maximumMinuteVarianceErrorBpsSquared"] = max(
                diagnostics["maximumMinuteVarianceErrorBpsSquared"],
                float(np.max(np.abs(generated_q - q))),
            )
            diagnostics["maximumMinuteActivityError"] = max(
                diagnostics["maximumMinuteActivityError"],
                int(np.max(np.abs(generated_counts.astype(np.int64) - counts.astype(np.int64)))),
            )
    diagnostics["relativeVarianceAddedForFeasibility"] = (
        diagnostics["addedVarianceBpsSquared"]
        / diagnostics["inputVarianceTotalBpsSquared"]
        if diagnostics["inputVarianceTotalBpsSquared"] > 0
        else 0.0
    )
    return {
        "histogramCounts": histogram,
        "zeroGapCounts": gaps,
        "acf": acf.finish(),
        "diagnostics": diagnostics,
        "generatedSeconds": int(np.sum(histogram)),
        "acfPaths": min(days, acf_days),
    }


def measure_actual_seconds(
    files: list[Path],
    previous_close: float,
    edges: np.ndarray,
    *,
    acf_days: int,
) -> dict:
    histogram = np.zeros(edges.size - 1, dtype=np.int64)
    gaps = np.zeros(GAP_CAP_SECONDS + 1, dtype=np.int64)
    acf = AcfAccumulator.create()
    last_close = previous_close
    for index, reference in enumerate(files):
        closes = read_candle_column(reference, "close")
        returns = np.diff(np.log(np.concatenate((
            np.asarray([last_close], dtype=np.float64),
            closes,
        )))) * 10_000.0
        last_close = float(closes[-1])
        histogram += bounded_histogram_counts(returns, edges)
        gaps += zero_gap_counts(returns)
        if index < acf_days:
            acf.add_returns(returns)
    return {
        "histogramCounts": histogram,
        "zeroGapCounts": gaps,
        "acf": acf.finish(),
        "seconds": int(np.sum(histogram)),
        "acfDays": min(len(files), acf_days),
    }


def probability(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return counts / np.sum(counts)


def evaluate_return_distributions(
    *,
    actual: dict,
    calibrated: dict,
    coherent: dict,
    test_start: int,
    histograms: dict,
    actual_second: dict,
    second_models: dict,
) -> dict:
    result = {
        "1s": {
            "actualObservations": actual_second["seconds"],
            "models": {},
        }
    }
    actual_probability = probability(actual_second["histogramCounts"])
    full_probability = dense_histogram(full_histogram(histograms, "1s"))
    result["1s"]["holdoutVsFullHistoryJsBits"] = jensen_shannon_bits(
        actual_probability,
        full_probability,
    )
    for model_id, model in second_models.items():
        result["1s"]["models"][model_id] = {
            "generatedObservations": model["generatedSeconds"],
            "jsBitsVsUntouchedHoldout": jensen_shannon_bits(
                probability(model["histogramCounts"]),
                actual_probability,
            ),
        }
    for level in KNOWN_HISTOGRAM_LEVELS:
        template = full_histogram(histograms, level)
        edges = histogram_edges(template)
        actual_values = actual[level]["periodReturnBps"][test_start:].reshape(-1)
        base_values = calibrated[level]["periodReturnBps"][test_start:].reshape(-1)
        model_values = coherent["periodReturnBps"][level][test_start:].reshape(-1)
        actual_counts = bounded_histogram_counts(actual_values, edges)
        result[level] = {
            "actualObservations": int(actual_values.size),
            "holdoutVsFullHistoryJsBits": jensen_shannon_bits(
                probability(actual_counts),
                dense_histogram(template),
            ),
            "models": {
                "independent": {
                    "generatedObservations": int(base_values.size),
                    "jsBitsVsUntouchedHoldout": jensen_shannon_bits(
                        probability(bounded_histogram_counts(base_values, edges)),
                        probability(actual_counts),
                    ),
                },
                "hierarchicalCoherent": {
                    "generatedObservations": int(model_values.size),
                    "jsBitsVsUntouchedHoldout": jensen_shannon_bits(
                        probability(bounded_histogram_counts(model_values, edges)),
                        probability(actual_counts),
                    ),
                },
            },
        }
    return result


def gap_summary(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=np.float64)
    total = np.sum(counts)
    survival = np.cumsum(counts[::-1])[::-1] / total
    probabilities = counts / total
    return {
        "gaps": int(total),
        "censoredAtSeconds": GAP_CAP_SECONDS,
        "probabilityAtLeast": {
            str(length): float(survival[length])
            for length in (1, 2, 5, 10, 15, 30, 60, 120, 180, 300)
        },
        "probabilities": probabilities,
    }


def evaluate_second_diagnostics(actual: dict, models: dict) -> dict:
    actual_gap = gap_summary(actual["zeroGapCounts"])
    result = {
        "acfLagsSeconds": list(ACF_LAGS),
        "actual": {
            "acf": {
                name: [float(values[lag]) for lag in ACF_LAGS]
                for name, values in actual["acf"].items()
            },
            "zeroGapSurvival": {
                key: value for key, value in actual_gap.items() if key != "probabilities"
            },
        },
        "models": {},
    }
    for model_id, model in models.items():
        model_gap = gap_summary(model["zeroGapCounts"])
        result["models"][model_id] = {
            "acf": {
                name: [float(values[lag]) for lag in ACF_LAGS]
                for name, values in model["acf"].items()
            },
            "absoluteAcfErrorBySignal": {
                name: float(np.mean(np.abs(
                    np.asarray([values[lag] for lag in ACF_LAGS])
                    - np.asarray([actual["acf"][name][lag] for lag in ACF_LAGS])
                )))
                for name, values in model["acf"].items()
            },
            "zeroGapSurvival": {
                key: value for key, value in model_gap.items() if key != "probabilities"
            },
            "zeroGapJsBits": jensen_shannon_bits(
                model_gap["probabilities"],
                actual_gap["probabilities"],
            ),
            "generationDiagnostics": model["diagnostics"],
        }
    return result


def build_acceptance(targets: dict, distributions: dict, seconds: dict) -> dict:
    hierarchical_js = distributions["1s"]["models"]["hierarchicalCoherent"][
        "jsBitsVsUntouchedHoldout"
    ]
    diagnostics = seconds["models"]["hierarchicalCoherent"]["generationDiagnostics"]
    coherent_level_gates = {}
    oracle_diagnostics = {}
    for feature in FEATURES:
        coherent_level_gates[feature] = {
            level: {
                "crpsNotWorseThanBottomUpFallback": (
                    targets[feature]["levels"][level][
                        "hierarchicalCrpsSkillVsBottomUpFallback"
                    ] >= -0.01
                ),
            }
            for level in LEVEL_MINUTES
        }
        oracle_diagnostics[feature] = {
            level: {
                "crpsSkillVsIncoherentScaleSpecificOracle": targets[feature][
                    "levels"
                ][level]["hierarchicalCrpsSkillVsScaleSpecificOracle"],
                "maximumCoverageError": targets[feature]["levels"][level][
                    "hierarchicalMaximumCoverageError"
                ],
            }
            for level in LEVEL_MINUTES
        }
    hierarchical_acf = seconds["models"]["hierarchicalCoherent"][
        "absoluteAcfErrorBySignal"
    ]
    return {
        "oneSecondReturnDistributionTargetBits": 0.005,
        "oneSecondReturnDistributionPassed": hierarchical_js < 0.005,
        "oneSecondReturnDistributionJsBits": hierarchical_js,
        "exactMinuteReturnPassed": diagnostics["maximumMinuteReturnErrorBps"] < 1e-8,
        "exactMinuteVariancePassed": diagnostics[
            "maximumMinuteVarianceErrorBpsSquared"
        ] < 1e-8,
        "exactMinuteActivityPassed": diagnostics["maximumMinuteActivityError"] == 0,
        "zeroGapJsTargetBits": 0.001,
        "zeroGapPassed": seconds["models"]["hierarchicalCoherent"][
            "zeroGapJsBits"
        ] < 0.001,
        "coherentHierarchyGates": coherent_level_gates,
        "allCoherentHierarchyGatesPassed": all(
            gate
            for feature in coherent_level_gates.values()
            for level in feature.values()
            for gate in level.values()
        ),
        "scaleSpecificOracleDiagnostics": oracle_diagnostics,
        "absoluteReturnAcfMeanAbsoluteErrorTarget": 0.05,
        "absoluteReturnAcfPassed": hierarchical_acf["absoluteReturn"] < 0.05,
        "activityAcfMeanAbsoluteErrorTarget": 0.05,
        "activityAcfPassed": hierarchical_acf["activity"] < 0.05,
    }


def compact_summary(report: dict) -> dict:
    return {
        "selectedHistoryWindows": report["selectedHistoryWindows"],
        "selectedCalibrationMethods": report["selectedCalibrationMethods"],
        "selectedParentWeights": {
            feature: {
                level: report["hierarchicalReconciliation"][feature][level][
                    "effectiveParentWeight"
                ]
                for level in BOTTOM_UP_LEVELS
            }
            for feature in FEATURES
        },
        "featureHierarchyGates": {
            feature: report["hierarchicalReconciliation"][feature]["featureGate"]
            for feature in FEATURES
        },
        "jointEnergySkill": {
            feature: {
                "vsBottomUpFallback": values["allLevelJointEnergy"][
                    "hierarchicalSkillVsBottomUpFallback"
                ],
                "vsScaleSpecificOracle": values["allLevelJointEnergy"][
                    "hierarchicalSkillVsScaleSpecificOracle"
                ],
            }
            for feature, values in report["targetForecastAudit"].items()
        },
        "oneSecondJsBits": report["returnDistributionAudit"]["1s"]["models"],
        "zeroGapJsBits": {
            model: values["zeroGapJsBits"]
            for model, values in report["secondProcessAudit"]["models"].items()
        },
        "acceptance": report["acceptance"],
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="data/benchmarks/log-return-distributions.json")
    parser.add_argument("--histograms", default="data/benchmarks/log-return-histograms.json")
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--forecast-start", default=FORECAST_START)
    parser.add_argument("--forecast-end", default=FORECAST_END)
    parser.add_argument("--calibration-days", type=int, default=CALIBRATION_DAYS)
    parser.add_argument(
        "--hierarchy-selection-days",
        type=int,
        default=HIERARCHY_SELECTION_DAYS,
    )
    parser.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument("--second-paths-per-day", type=int, default=SECOND_PATHS_PER_DAY)
    parser.add_argument("--acf-days", type=int, default=30)
    parser.add_argument(
        "--minute-cache",
        default="data/benchmarks/full-hierarchy-minute-history.npz",
    )
    parser.add_argument(
        "--forecast-cache",
        default="data/benchmarks/full-hierarchy-calibrated-forecasts.npz",
    )
    parser.add_argument(
        "--fit-cache",
        default="data/benchmarks/full-hierarchy-one-second-fit.json",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/full-hierarchical-one-second-process.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
