"""Rolling local-statistic refit and full next-day candle simulation audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy import special, stats

from analyze_one_second_dependence_model import (
    full_histogram,
    histogram_edges,
    selected_files,
)
from analyze_parametric_one_second_process import (
    DAY_SECONDS,
    MINUTES_PER_DAY,
    RNG_SEED,
    bounded_histogram_counts,
    discrete_midpoint_gaussian_scores,
    fit_process,
    gaussianize_quantile_spline,
    generate_projected_second_batch,
    inverse_discrete_gaussian_copula,
    inverse_gaussianized_quantile_spline,
    measure_source,
    project_returns_to_daily_targets,
    seasonal_values,
)
from evaluate_parametric_next_day_process import (
    GaussianPosterior,
    activity_location,
    calibrate_intraday_generation,
    ensemble_metrics,
    filter_factor_observations,
    histogram_from_counts,
    minute_path_range,
    read_daily_outcomes,
    sample_factor_paths,
    summarize_histogram_forecast,
    update_factor_posterior,
)
from trading_storage import read_candle_column


TRAIN_START = "2021-07-25T00:00:00+00:00"
TEST_START = "2025-07-25T00:00:00+00:00"
TEST_END = "2026-07-25T00:00:00+00:00"
DEFAULT_WINDOWS = (7, 30, 90, 365, 730)
HORIZON_MINUTES = {
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "8h": 480,
    "1d": 1_440,
}
DEFAULT_ENSEMBLE_SIZE = 16
FEATURE_NAMES = (
    "periodReturnBps",
    "oneSecondRealizedVarianceBpsSquared",
    "activeSecondFraction",
    "minutePathRangeBps",
    "minuteReturnQuadraticVariationBpsSquared",
    "maximumAbsoluteMinuteReturnBps",
    "minuteLag1ReturnCorrelation",
)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis = read_json(repo / args.analysis)
    histograms = read_json(repo / args.histograms)
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    train_start = parse_time(args.train_start)
    test_start = parse_time(args.test_start)
    test_end = parse_time(args.test_end)
    windows = tuple(sorted(set(args.windows)))
    horizons = tuple(args.horizons)
    if not train_start < test_start < test_end:
        raise ValueError("expected train-start < test-start < test-end")
    if min(windows) < 7:
        raise ValueError("rolling windows shorter than seven days are unsupported")

    train_files = selected_files(source, train_start, test_start)
    test_files = selected_files(source, test_start, test_end)
    one_second_histogram = full_histogram(histograms, "1s")
    one_second_edges = histogram_edges(one_second_histogram)
    measurements = measure_source(
        train_files,
        micro_threshold_bps=0.5 * float(one_second_histogram["binWidthBps"]),
        one_second_histogram_edges=one_second_edges,
    )
    fitted = fit_process(measurements)
    previous_close = float(read_candle_column(train_files[-1], "close")[-1])
    outcomes = read_daily_outcomes(test_files, previous_close, one_second_edges)
    actual_one_second_counts = read_prefix_one_second_histograms(
        test_files,
        previous_close,
        horizons,
        one_second_edges,
    )

    training_histogram = histogram_from_counts(
        one_second_histogram,
        measurements.one_second_histogram_counts,
    )
    training_volatility_score = gaussianize_quantile_spline(
        np.log(measurements.realized_variance + fitted.variance_floor),
        fitted.variance_quantile_knots,
        fitted.variance_log_quantiles,
    )
    calibration = calibrate_intraday_generation(
        fitted,
        measurements,
        training_volatility_score,
        training_histogram,
        np.random.default_rng(RNG_SEED + 700),
    )
    result, forecast_arrays = run_rolling_audit(
        fitted=fitted,
        measurements=measurements,
        outcomes=outcomes,
        windows=windows,
        horizons=horizons,
        ensemble_size=args.ensemble_size,
        calibration=calibration,
        histograms=histograms,
        actual_one_second_counts=actual_one_second_counts,
        rng=np.random.default_rng(RNG_SEED + 710),
    )
    forecasts_output = repo / args.forecasts_output
    forecasts_output.parent.mkdir(parents=True, exist_ok=True)
    forecast_arrays["metadataJson"] = np.asarray(json.dumps({
        "testStart": test_start.isoformat().replace("+00:00", "Z"),
        "testEndExclusive": test_end.isoformat().replace("+00:00", "Z"),
        "targetDays": len(test_files),
        "windows": list(windows),
        "horizons": list(horizons),
        "features": list(FEATURE_NAMES),
    }))
    np.savez_compressed(forecasts_output, **forecast_arrays)
    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "design": {
            "stableShapeFitStart": train_start.isoformat().replace("+00:00", "Z"),
            "stableShapeFitEndExclusive": test_start.isoformat().replace("+00:00", "Z"),
            "testStart": test_start.isoformat().replace("+00:00", "Z"),
            "testEndExclusive": test_end.isoformat().replace("+00:00", "Z"),
            "targetDays": len(test_files),
            "trailingWindowDays": list(windows),
            "trailingWindowCandles": [days * DAY_SECONDS for days in windows],
            "predictionHorizons": {
                horizon: {
                    "minutes": HORIZON_MINUTES[horizon],
                    "candles": HORIZON_MINUTES[horizon] * 60,
                }
                for horizon in horizons
            },
            "generatedPathsPerTargetWindowAndHorizon": args.ensemble_size,
            "fullyMaterializedSecondPathsPerTargetWindowAndHorizon": 1,
            "fullyMaterializedSecondPathCases": (
                len(test_files) * len(windows) * len(horizons)
            ),
            "futureDataUsedForAnyForecast": False,
            "historicalCandlesResampled": False,
            "forecastArchive": str(forecasts_output.relative_to(repo)).replace("\\", "/"),
            "rollingRefit": (
                "Before every target period, variance, variance-normalized signed return, "
                "and variance-conditioned active fraction are re-estimated from historical "
                "blocks of the same duration inside exactly the latest X realized days. "
                "Stable intraminute shape parameters are pre-holdout only."
            ),
            "commonRandomNumbersAcrossWindows": True,
        },
        "stableShapeCalibration": calibration,
        "windows": result,
        "ranking": rank_windows(result),
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(compact_summary(report), indent=2))


def run_rolling_audit(
    *,
    fitted,
    measurements,
    outcomes,
    windows: tuple[int, ...],
    horizons: tuple[str, ...],
    ensemble_size: int,
    calibration: dict[str, float],
    histograms: dict,
    actual_one_second_counts: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    test_days = outcomes.returns.size
    history_minute_returns = np.concatenate((
        measurements.minute_returns,
        outcomes.minute_returns.reshape(-1),
    ))
    history_minute_variance = np.concatenate((
        measurements.realized_variance,
        outcomes.minute_realized_variance.reshape(-1),
    ))
    history_minute_counts = np.concatenate((
        measurements.counts,
        outcomes.minute_activity_counts.reshape(-1),
    ))
    history_offset = measurements.minute_returns.size

    train_indexes = np.arange(measurements.counts.size, dtype=np.int64)
    train_volatility_score = gaussianize_quantile_spline(
        np.log(measurements.realized_variance + fitted.variance_floor),
        fitted.variance_quantile_knots,
        fitted.variance_log_quantiles,
    )
    train_volatility_residual = train_volatility_score - seasonal_values(
        train_indexes,
        fitted.volatility_seasonal,
    )
    count_scores = discrete_midpoint_gaussian_scores(
        fitted.activity_count_probabilities
    )
    train_activity_score = count_scores[measurements.counts]
    train_activity_residual = gaussianize_quantile_spline(
        train_activity_score
        - activity_location(train_volatility_score, fitted.activity_coefficients),
        fitted.activity_residual_quantile_knots,
        fitted.activity_residual_quantiles,
    )
    filter_minutes = min(measurements.counts.size, 180 * MINUTES_PER_DAY)
    volatility_posterior = filter_factor_observations(
        fitted.volatility_factors,
        train_volatility_residual[-filter_minutes:],
    )
    activity_posterior = filter_factor_observations(
        fitted.activity_factors,
        train_activity_residual[-filter_minutes:],
    )
    efficiency_stationary = GaussianPosterior(
        mean=np.zeros(fitted.efficiency_factors.timescales.size),
        covariance=np.eye(fitted.efficiency_factors.timescales.size),
    )

    test_q = outcomes.minute_realized_variance.reshape(-1)
    test_counts = outcomes.minute_activity_counts.reshape(-1)
    test_indexes = np.arange(
        measurements.counts.size,
        measurements.counts.size + test_q.size,
        dtype=np.int64,
    )
    test_volatility_score = gaussianize_quantile_spline(
        np.log(test_q + fitted.variance_floor),
        fitted.variance_quantile_knots,
        fitted.variance_log_quantiles,
    )
    test_volatility_residual = test_volatility_score - seasonal_values(
        test_indexes,
        fitted.volatility_seasonal,
    )
    test_activity_residual = gaussianize_quantile_spline(
        count_scores[test_counts]
        - activity_location(test_volatility_score, fitted.activity_coefficients),
        fitted.activity_residual_quantile_knots,
        fitted.activity_residual_quantiles,
    )

    stores = {}
    for window in windows:
        stores[window] = {
            horizon: {
                "ensembles": {
                    name: np.empty((test_days, ensemble_size), dtype=np.float64)
                    for name in FEATURE_NAMES
                },
                "histogramCounts": {
                    scale_id: np.zeros(
                        int(full_histogram(histograms, scale_id)["binCount"]),
                        dtype=np.int64,
                    )
                    for scale_id in ("1s", "1m")
                },
                "projectionFallbacks": 0,
                "targetClips": 0,
            }
            for horizon in horizons
        }

    maximum_horizon = max(HORIZON_MINUTES[horizon] for horizon in horizons)
    for day in range(test_days):
        if day % 25 == 0:
            print(f"Rolling refit forecast {day}/{test_days}...", flush=True)
        minute_start = day * MINUTES_PER_DAY
        minute_stop = minute_start + MINUTES_PER_DAY
        forecast_indexes = test_indexes[
            minute_start:minute_start + maximum_horizon
        ]
        volatility_latent = sample_factor_paths(
            fitted.volatility_factors,
            volatility_posterior,
            maximum_horizon,
            ensemble_size,
            rng,
        )
        base_volatility_score = (
            seasonal_values(forecast_indexes, fitted.volatility_seasonal)[None, :]
            + volatility_latent
        )
        base_log_q = inverse_gaussianized_quantile_spline(
            base_volatility_score,
            fitted.variance_quantile_knots,
            fitted.variance_log_quantiles,
        )
        base_q = np.maximum(0.0, np.exp(base_log_q) - fitted.variance_floor)

        activity_latent = sample_factor_paths(
            fitted.activity_factors,
            activity_posterior,
            maximum_horizon,
            ensemble_size,
            rng,
        )
        activity_residual = inverse_gaussianized_quantile_spline(
            activity_latent,
            fitted.activity_residual_quantile_knots,
            fitted.activity_residual_quantiles,
        )
        base_activity_score = (
            activity_location(
                base_volatility_score,
                fitted.activity_coefficients,
            )
            + activity_residual
        )
        efficiency_latent = sample_factor_paths(
            fitted.efficiency_factors,
            efficiency_stationary,
            maximum_horizon,
            ensemble_size,
            rng,
        )
        efficiency_coefficients = fitted.efficiency_volatility_coefficients
        base_efficiency_score = (
            efficiency_coefficients[0]
            + efficiency_coefficients[1] * base_volatility_score
            + efficiency_coefficients[2]
            * (base_volatility_score * base_volatility_score - 1.0)
            + fitted.efficiency_residual_std * efficiency_latent
        )
        for horizon in horizons:
            horizon_minutes = HORIZON_MINUTES[horizon]
            common_draws = {
                "variance": rng.standard_t(8.0, ensemble_size),
                "return": rng.standard_t(8.0, ensemble_size),
                "activity": rng.standard_t(8.0, ensemble_size),
            }
            for window in windows:
                history_end = history_offset + day * MINUTES_PER_DAY
                history_start = history_end - window * MINUTES_PER_DAY
                local_returns, local_variance, local_active = aggregate_history_blocks(
                    history_minute_returns[history_start:history_end],
                    history_minute_variance[history_start:history_end],
                    history_minute_counts[history_start:history_end],
                    horizon_minutes,
                )
                local_targets = sample_local_period_targets(
                    local_returns,
                    local_variance,
                    local_active,
                    common_draws,
                )
                score = base_activity_score[:, :horizon_minutes]
                counts = counts_matching_period_activity(
                    score,
                    fitted.activity_count_probabilities,
                    local_targets["activeFraction"],
                )
                q = base_q[:, :horizon_minutes].copy()
                q[counts == 0] = 0.0
                q_totals = np.sum(q, axis=1)
                q *= np.divide(
                    local_targets["variance"],
                    q_totals,
                    out=np.ones(ensemble_size),
                    where=q_totals > 0,
                )[:, None]
                efficiency = inverse_gaussianized_quantile_spline(
                    base_efficiency_score[:, :horizon_minutes],
                    fitted.efficiency_quantile_knots,
                    fitted.efficiency_quantiles,
                )
                efficiency[counts == 0] = 0.0
                maximum_efficiency = np.sqrt(counts.astype(np.float64))
                efficiency = np.clip(
                    efficiency * calibration["efficiencyScale"],
                    -maximum_efficiency * (1.0 - 1e-10),
                    maximum_efficiency * (1.0 - 1e-10),
                )
                single = counts == 1
                efficiency[single] = np.where(
                    efficiency[single] >= 0.0,
                    1.0,
                    -1.0,
                )
                minute_returns = efficiency * np.sqrt(q)
                feasible = np.sum(
                    np.sqrt(counts.astype(np.float64) * q),
                    axis=1,
                ) * (1.0 - 1e-9)
                period_targets = np.clip(
                    local_targets["return"],
                    -feasible,
                    feasible,
                )
                store = stores[window][horizon]
                store["targetClips"] += int(np.count_nonzero(
                    period_targets != local_targets["return"]
                ))
                projected, fallbacks = project_returns_to_daily_targets(
                    minute_returns.reshape(-1),
                    q.reshape(-1),
                    counts.reshape(-1),
                    period_targets,
                    aligned=False,
                    block_minutes=horizon_minutes,
                )
                minute_returns = projected.reshape(
                    ensemble_size,
                    horizon_minutes,
                )
                store["projectionFallbacks"] += fallbacks
                features = generated_features(minute_returns, q, counts)
                for name, values in features.items():
                    store["ensembles"][name][day] = values

                store["histogramCounts"]["1m"] += bounded_histogram_counts(
                    minute_returns[0],
                    histogram_edges(full_histogram(histograms, "1m")),
                )
                _, second_returns = generate_projected_second_batch(
                    fitted,
                    counts=counts[0],
                    volatility_score=base_volatility_score[0, :horizon_minutes],
                    target_minute_returns=minute_returns[0],
                    realized_variance=q[0],
                    activity_timing_rho=calibration["activityTimingGaussianRho"],
                    magnitude_dispersion_scale=calibration["magnitudeDispersionScale"],
                    micro_probability_scale=calibration["microProbabilityScale"],
                    rng=rng,
                )
                store["histogramCounts"]["1s"] += bounded_histogram_counts(
                    second_returns,
                    histogram_edges(full_histogram(histograms, "1s")),
                )

        for value in test_volatility_residual[minute_start:minute_stop]:
            volatility_posterior = update_factor_posterior(
                fitted.volatility_factors,
                volatility_posterior,
                float(value),
            )
        for value in test_activity_residual[minute_start:minute_stop]:
            activity_posterior = update_factor_posterior(
                fitted.activity_factors,
                activity_posterior,
                float(value),
            )

    result = {}
    forecast_arrays: dict[str, np.ndarray] = {}
    for window in windows:
        result[str(window)] = {
            "historyDays": window,
            "historyCandles": window * DAY_SECONDS,
            "horizons": {},
        }
        for horizon in horizons:
            horizon_minutes = HORIZON_MINUTES[horizon]
            store = stores[window][horizon]
            actual_minute_returns = outcomes.minute_returns[:, :horizon_minutes]
            actual_q = outcomes.minute_realized_variance[:, :horizon_minutes]
            actual_counts = outcomes.minute_activity_counts[:, :horizon_minutes]
            actual_features = generated_features(
                actual_minute_returns,
                actual_q,
                actual_counts,
            )
            feature_metrics = {}
            for name, actual in actual_features.items():
                ensemble = store["ensembles"][name]
                forecast_arrays[actual_archive_key(horizon, name)] = actual
                forecast_arrays[ensemble_archive_key(window, horizon, name)] = ensemble
                feature_metrics[name] = {
                    "probabilistic": ensemble_metrics(actual, ensemble),
                    "ensembleMedian": paired_metrics(
                        actual,
                        np.median(ensemble, axis=1),
                    ),
                    "singleGeneratedPath": paired_metrics(
                        actual,
                        ensemble[:, 0],
                    ),
                    "actualMean": float(np.mean(actual)),
                    "generatedMean": float(np.mean(ensemble)),
                }
            actual_minute_counts = bounded_histogram_counts(
                actual_minute_returns.reshape(-1),
                histogram_edges(full_histogram(histograms, "1m")),
            )
            distributions = {
                "1s": summarize_histogram_forecast(
                    actual_one_second_counts[horizon],
                    store["histogramCounts"]["1s"],
                    full_histogram(histograms, "1s"),
                    repetitions=200,
                ),
                "1m": summarize_histogram_forecast(
                    actual_minute_counts,
                    store["histogramCounts"]["1m"],
                    full_histogram(histograms, "1m"),
                    repetitions=400,
                ),
            }
            result[str(window)]["horizons"][horizon] = {
                "predictionMinutes": horizon_minutes,
                "predictionCandles": horizon_minutes * 60,
                "historicalBlocksUsedPerFit": (
                    window * MINUTES_PER_DAY // horizon_minutes
                ),
                "featureMetrics": feature_metrics,
                "returnDistributions": distributions,
                "projectionFallbacks": store["projectionFallbacks"],
                "targetFeasibilityClips": store["targetClips"],
            }
    return result, forecast_arrays


def actual_archive_key(horizon: str, feature: str) -> str:
    return f"actual__{horizon}__{feature}"


def ensemble_archive_key(window: int, horizon: str, feature: str) -> str:
    return f"ensemble__{window}__{horizon}__{feature}"


def aggregate_history_blocks(
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    minute_counts: np.ndarray,
    block_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    complete = minute_returns.size // block_minutes
    if complete < 5:
        raise ValueError("too few historical blocks for local fit")
    selected_returns = minute_returns[-complete * block_minutes:].reshape(
        complete,
        block_minutes,
    )
    selected_variance = minute_variance[-complete * block_minutes:].reshape(
        complete,
        block_minutes,
    )
    selected_counts = minute_counts[-complete * block_minutes:].reshape(
        complete,
        block_minutes,
    )
    return (
        np.sum(selected_returns, axis=1),
        np.sum(selected_variance, axis=1),
        np.mean(selected_counts, axis=1) / 60.0,
    )


def sample_local_period_targets(
    returns: np.ndarray,
    variance: np.ndarray,
    active_fraction: np.ndarray,
    common_draws: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if returns.size < 5:
        raise ValueError("local target fit needs at least five days")
    log_variance = np.log(np.maximum(variance, 1e-12))
    variance_fit = fit_ar1(log_variance)
    next_log_variance = draw_fitted_ar1(
        variance_fit,
        common_draws["variance"],
    )
    next_variance = np.exp(next_log_variance)

    efficiency = returns / np.sqrt(np.maximum(variance, 1e-12))
    efficiency_fit = fit_ar1(efficiency)
    next_efficiency = draw_fitted_ar1(
        efficiency_fit,
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
    activity_fit = fit_ar1(residual)
    next_activity_residual = draw_fitted_ar1(
        activity_fit,
        common_draws["activity"],
    )
    next_active_logit = (
        np.mean(active_logit)
        + slope * (next_log_variance - np.mean(log_variance))
        + next_activity_residual
    )
    next_active_fraction = special.expit(next_active_logit)
    return {
        "variance": next_variance,
        "return": next_return,
        "activeFraction": np.clip(next_active_fraction, 0.01, 0.999),
    }


def fit_ar1(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    centered = values - mean
    denominator = float(np.sum(centered[:-1] * centered[:-1]))
    raw_phi = (
        float(np.sum(centered[1:] * centered[:-1]) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    phi = float(np.clip(raw_phi * values.size / (values.size + 20.0), -0.7, 0.97))
    residual = values[1:] - (mean + phi * (values[:-1] - mean))
    scale = float(np.std(residual, ddof=1)) if residual.size > 1 else 0.0
    scale = max(scale, float(np.std(values)) * 0.05, 1e-9)
    excess = max(float(stats.kurtosis(residual, fisher=True, bias=False)), 0.0) if residual.size >= 8 else 0.0
    degrees = float(np.clip(6.0 / excess + 4.0, 4.2, 30.0)) if excess > 1e-6 else 30.0
    return {
        "mean": mean,
        "phi": phi,
        "last": float(values[-1]),
        "scale": scale,
        "degrees": degrees,
    }


def draw_fitted_ar1(fit: dict[str, float], common_t8: np.ndarray) -> np.ndarray:
    target_mean = fit["mean"] + fit["phi"] * (fit["last"] - fit["mean"])
    uniforms = stats.t.cdf(common_t8, df=8.0)
    innovations = stats.t.ppf(uniforms, df=fit["degrees"])
    standard_deviation = math.sqrt(fit["degrees"] / (fit["degrees"] - 2.0))
    return target_mean + fit["scale"] * innovations / standard_deviation


def counts_matching_period_activity(
    score: np.ndarray,
    probabilities: np.ndarray,
    target_active_fraction: np.ndarray,
) -> np.ndarray:
    paths = score.shape[0]
    target_mean = np.clip(target_active_fraction, 0.0, 1.0) * 60.0
    low = np.full(paths, -8.0)
    high = np.full(paths, 8.0)
    for _ in range(36):
        middle = 0.5 * (low + high)
        counts = inverse_discrete_gaussian_copula(
            score + middle[:, None],
            probabilities,
        )
        generated_mean = np.mean(counts, axis=1)
        low = np.where(generated_mean < target_mean, middle, low)
        high = np.where(generated_mean < target_mean, high, middle)
    return inverse_discrete_gaussian_copula(
        score + (0.5 * (low + high))[:, None],
        probabilities,
    ).astype(np.uint8)


def generated_features(
    minute_returns: np.ndarray,
    q: np.ndarray,
    counts: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "periodReturnBps": np.sum(minute_returns, axis=1),
        "oneSecondRealizedVarianceBpsSquared": np.sum(q, axis=1),
        "activeSecondFraction": np.mean(counts, axis=1) / 60.0,
        "minutePathRangeBps": minute_path_range(minute_returns),
        "minuteReturnQuadraticVariationBpsSquared": np.sum(
            minute_returns * minute_returns,
            axis=1,
        ),
        "maximumAbsoluteMinuteReturnBps": np.max(
            np.abs(minute_returns),
            axis=1,
        ),
        "minuteLag1ReturnCorrelation": row_lag1_correlation(minute_returns),
    }


def daily_zero_probability_from_counts(counts: np.ndarray) -> np.ndarray:
    offset = MINUTES_PER_DAY - 1
    complete = (counts.size - offset) // MINUTES_PER_DAY
    selected = counts[offset:offset + complete * MINUTES_PER_DAY].reshape(
        complete,
        MINUTES_PER_DAY,
    )
    return 1.0 - np.mean(selected, axis=1) / 60.0


def row_lag1_correlation(values: np.ndarray) -> np.ndarray:
    left = values[:, :-1]
    right = values[:, 1:]
    left_centered = left - np.mean(left, axis=1)[:, None]
    right_centered = right - np.mean(right, axis=1)[:, None]
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    return np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=np.zeros(values.shape[0]),
        where=denominator > 0,
    )


def paired_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "bias": float(np.mean(error)),
        "meanAbsoluteError": float(np.mean(np.abs(error))),
        "rootMeanSquaredError": float(np.sqrt(np.mean(error * error))),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
    }


def rank_windows(result: dict) -> dict:
    first_window = next(iter(result.values()))
    horizon_names = first_window["horizons"].keys()
    return {
        horizon: {
            "lowestCrpsByFeature": {
                feature: min(
                    result,
                    key=lambda window: result[window]["horizons"][horizon][
                        "featureMetrics"
                    ][feature]["probabilistic"]["meanCrps"],
                )
                for feature in first_window["horizons"][horizon][
                    "featureMetrics"
                ]
            },
            "lowestJsByCandleScale": {
                scale: min(
                    result,
                    key=lambda window: result[window]["horizons"][horizon][
                        "returnDistributions"
                    ][scale]["jsDivergenceBits"],
                )
                for scale in ("1s", "1m")
            },
        }
        for horizon in horizon_names
    }


def compact_summary(report: dict) -> dict:
    windows = {}
    for window, values in report["windows"].items():
        windows[window] = {}
        for horizon, horizon_values in values["horizons"].items():
            windows[window][horizon] = {
                "featureCrps": {
                    feature: metrics["probabilistic"]["meanCrps"]
                    for feature, metrics in horizon_values[
                        "featureMetrics"
                    ].items()
                },
                "featureMedianBias": {
                    feature: metrics["ensembleMedian"]["bias"]
                    for feature, metrics in horizon_values[
                        "featureMetrics"
                    ].items()
                },
                "jsByCandleScale": {
                    scale: metrics["jsDivergenceBits"]
                    for scale, metrics in horizon_values[
                        "returnDistributions"
                    ].items()
                },
            }
    return {
        "design": report["design"],
        "windows": windows,
        "ranking": report["ranking"],
    }


def read_prefix_one_second_histograms(
    files: list[Path],
    previous_close: float,
    horizons: tuple[str, ...],
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        horizon: np.zeros(edges.size - 1, dtype=np.int64)
        for horizon in horizons
    }
    last_close = previous_close
    for file in files:
        closes = read_candle_column(file, "close")
        returns = np.diff(np.log(np.concatenate((
            np.asarray([last_close], dtype=np.float64),
            closes,
        )))) * 10_000.0
        for horizon in horizons:
            candle_count = HORIZON_MINUTES[horizon] * 60
            result[horizon] += bounded_histogram_counts(
                returns[:candle_count],
                edges,
            )
        last_close = float(closes[-1])
    return result


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
    parser.add_argument("--histograms", default="data/benchmarks/log-return-histograms.json")
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--test-start", default=TEST_START)
    parser.add_argument("--test-end", default=TEST_END)
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument(
        "--horizons",
        nargs="+",
        choices=list(HORIZON_MINUTES),
        default=list(HORIZON_MINUTES),
    )
    parser.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    parser.add_argument(
        "--output",
        default="data/benchmarks/rolling-refit-next-day-process.json",
    )
    parser.add_argument(
        "--forecasts-output",
        default="data/benchmarks/rolling-refit-next-day-process-forecasts.npz",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
