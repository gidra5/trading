"""Chronological next-day calibration audit for the fitted return process.

The process is fitted once on a strictly earlier window. Each day in the
holdout is then scored as a one-day-ahead forecast. AR-factor posteriors are
updated only after that day's realized observation has been scored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_one_second_dependence_model import (
    full_histogram,
    full_scale,
    histogram_centers,
    histogram_edges,
    jensen_shannon_bits,
    selected_files,
)
from analyze_parametric_one_second_process import (
    ACTIVITY_CALIBRATION_MINUTES,
    DAY_SECONDS,
    MINUTES_PER_DAY,
    RNG_SEED,
    DailyReturnFit,
    DailyVarianceFit,
    FactorMixture,
    FittedProcess,
    ArFactorGenerator,
    bounded_histogram_counts,
    calibrate_activity_rho,
    calibrate_discrete_location,
    calibrate_magnitude_dispersion,
    discrete_midpoint_gaussian_scores,
    fit_process,
    gaussianize_quantile_spline,
    generate_projected_second_batch,
    inverse_gaussianized_quantile_spline,
    inverse_discrete_gaussian_copula,
    js_sampling_floor,
    measure_source,
    project_returns_to_daily_targets,
    quantiles,
    seasonal_values,
    selected_daily_correlations,
)
from trading_storage import read_candle_column


DEFAULT_TRAIN_START = "2021-07-25T00:00:00+00:00"
DEFAULT_TRAIN_END = "2025-07-25T00:00:00+00:00"
DEFAULT_TEST_END = "2026-07-25T00:00:00+00:00"
DEFAULT_ENSEMBLE_SIZE = 2_048
CALIBRATION_DRAWS = 250_000


@dataclass(frozen=True)
class GaussianPosterior:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class DailyOutcomes:
    returns: np.ndarray
    realized_variance: np.ndarray
    zero_probability: np.ndarray
    intraday_range: np.ndarray
    minute_returns: np.ndarray
    minute_realized_variance: np.ndarray
    minute_activity_counts: np.ndarray
    one_second_histogram_counts: np.ndarray


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis = read_json(repo / args.analysis)
    histograms = read_json(repo / args.histograms)
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    train_start = parse_time(args.train_start)
    train_end = parse_time(args.train_end)
    test_end = parse_time(args.test_end)
    if not train_start < train_end < test_end:
        raise ValueError("expected train-start < train-end < test-end")

    train_files = selected_files(source, train_start, train_end)
    test_files = selected_files(source, train_end, test_end)
    if len(train_files) < 365 or not test_files:
        raise RuntimeError("chronological audit needs at least one training year and one test day")

    one_second_histogram = full_histogram(histograms, "1s")
    measurements = measure_source(
        train_files,
        micro_threshold_bps=0.5 * float(one_second_histogram["binWidthBps"]),
        one_second_histogram_edges=histogram_edges(one_second_histogram),
    )
    fitted = fit_process(measurements)
    previous_close = float(read_candle_column(train_files[-1], "close")[-1])
    outcomes = read_daily_outcomes(
        test_files,
        previous_close,
        histogram_edges(one_second_histogram),
    )

    return_forecasts = forecast_daily_returns(
        fitted.daily_return,
        measurements.daily_returns,
        outcomes.returns,
        args.ensemble_size,
        np.random.default_rng(RNG_SEED + 200),
    )
    variance_forecasts = forecast_daily_variance(
        fitted.daily_variance,
        measurements.daily_realized_variance,
        outcomes.realized_variance,
        args.ensemble_size,
        np.random.default_rng(RNG_SEED + 300),
    )

    return_gaussian = gaussian_ensembles(
        outcomes.returns.size,
        args.ensemble_size,
        float(np.mean(measurements.daily_returns)),
        float(np.std(measurements.daily_returns)),
        np.random.default_rng(RNG_SEED + 400),
    )
    log_train_variance = np.log(measurements.daily_realized_variance)
    variance_lognormal = np.exp(gaussian_ensembles(
        outcomes.realized_variance.size,
        args.ensemble_size,
        float(np.mean(log_train_variance)),
        float(np.std(log_train_variance)),
        np.random.default_rng(RNG_SEED + 500),
    ))

    return_metrics = {
        name: ensemble_metrics(outcomes.returns, values)
        for name, values in {
            "fittedProcessUnconditional": return_forecasts["unconditional"],
            "fittedProcessStateConditioned": return_forecasts["conditioned"],
            "gaussianBaseline": return_gaussian,
        }.items()
    }
    variance_metrics = {
        name: ensemble_metrics(outcomes.realized_variance, values)
        for name, values in {
            "fittedProcessUnconditional": variance_forecasts["unconditional"],
            "fittedProcessStateConditioned": variance_forecasts["conditioned"],
            "lognormalBaseline": variance_lognormal,
        }.items()
    }
    attach_skill_scores(return_metrics, "gaussianBaseline")
    attach_skill_scores(variance_metrics, "lognormalBaseline")

    daily_histogram = full_histogram(histograms, "1d")
    return_distribution = distribution_comparison(
        training=measurements.daily_returns,
        actual=outcomes.returns,
        predicted=return_forecasts["conditioned"].reshape(-1),
        histogram=daily_histogram,
    )
    intraday = forecast_intraday_paths(
        fitted=fitted,
        measurements=measurements,
        outcomes=outcomes,
        return_targets=return_forecasts["conditioned"],
        variance_targets=variance_forecasts["conditioned"],
        histograms=histograms,
        analysis=analysis,
        path_count=args.intraday_ensemble_size,
        rng=np.random.default_rng(RNG_SEED + 600),
    )

    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "design": {
            "trainStart": train_start.isoformat().replace("+00:00", "Z"),
            "trainEndExclusive": train_end.isoformat().replace("+00:00", "Z"),
            "testStart": train_end.isoformat().replace("+00:00", "Z"),
            "testEndExclusive": test_end.isoformat().replace("+00:00", "Z"),
            "trainingDays": len(train_files),
            "scoredNextDays": len(test_files),
            "ensembleSizePerDay": args.ensemble_size,
            "parameterRefitsInsideHoldout": 0,
            "stateUpdateTiming": (
                "Each realized day updates the AR-factor posterior only after its forecast is scored."
            ),
            "futureDataUsedForFit": False,
            "historicalCandlesResampled": False,
        },
        "trainingFit": {
            "dailyReturnBps": describe(measurements.daily_returns),
            "dailyRealizedVarianceBpsSquared": describe(
                measurements.daily_realized_variance
            ),
            "meanActiveSecondsPerMinute": float(np.mean(measurements.counts)),
            "zeroSecondProbability": float(
                1.0 - np.mean(measurements.counts) / 60.0
            ),
        },
        "actualHoldout": {
            "dailyReturnBps": describe(outcomes.returns),
            "dailyRealizedVarianceBpsSquared": describe(outcomes.realized_variance),
            "dailyZeroProbability": describe(outcomes.zero_probability),
            "dailyIntradayRangeBps": describe(outcomes.intraday_range),
            "dailyReturnCorrelations": selected_daily_correlations(outcomes.returns),
            "dailyLogVarianceCorrelations": selected_daily_correlations(
                np.log(outcomes.realized_variance)
            ),
        },
        "dailyReturnForecast": {
            "metrics": return_metrics,
            "distributionShift": return_distribution,
            "assessment": calibration_assessment(
                return_metrics["fittedProcessStateConditioned"]
            ),
        },
        "dailyVarianceForecast": {
            "metrics": variance_metrics,
            "assessment": calibration_assessment(
                variance_metrics["fittedProcessStateConditioned"]
            ),
        },
        "intradayCandleForecast": intraday,
        "interpretation": interpretation(
            return_metrics,
            variance_metrics,
            intraday,
        ),
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(compact_summary(report), indent=2))


def forecast_daily_returns(
    fit: DailyReturnFit,
    training: np.ndarray,
    actual: np.ndarray,
    ensemble_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    training_score = gaussianize_quantile_spline(
        training,
        fit.quantile_knots,
        fit.return_quantiles,
    )
    actual_score = gaussianize_quantile_spline(
        actual,
        fit.quantile_knots,
        fit.return_quantiles,
    )
    unconditional_score, conditioned_score = rolling_score_ensembles(
        fit.factors,
        training_score,
        actual_score,
        ensemble_size,
        rng,
    )
    return {
        "unconditional": inverse_gaussianized_quantile_spline(
            unconditional_score,
            fit.quantile_knots,
            fit.return_quantiles,
        ),
        "conditioned": inverse_gaussianized_quantile_spline(
            conditioned_score,
            fit.quantile_knots,
            fit.return_quantiles,
        ),
    }


def forecast_daily_variance(
    fit: DailyVarianceFit,
    training: np.ndarray,
    actual: np.ndarray,
    ensemble_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    training_log = np.log(training + fit.floor)
    actual_log = np.log(actual + fit.floor)
    training_score = gaussianize_quantile_spline(
        training_log,
        fit.quantile_knots,
        fit.log_quantiles,
    )
    actual_score = gaussianize_quantile_spline(
        actual_log,
        fit.quantile_knots,
        fit.log_quantiles,
    )
    unconditional_score, conditioned_score = rolling_score_ensembles(
        fit.factors,
        training_score,
        actual_score,
        ensemble_size,
        rng,
    )
    scale_score = rng.normal(
        0.0,
        math.sqrt(factor_observation_variance(fit.factors)),
        CALIBRATION_DRAWS,
    )
    scale_values = np.maximum(
        0.0,
        np.exp(inverse_gaussianized_quantile_spline(
            scale_score,
            fit.quantile_knots,
            fit.log_quantiles,
        )) - fit.floor,
    )
    mean_scale = fit.target_mean / float(np.mean(scale_values))

    def transform(score: np.ndarray) -> np.ndarray:
        log_variance = inverse_gaussianized_quantile_spline(
            score,
            fit.quantile_knots,
            fit.log_quantiles,
        )
        return np.maximum(0.0, np.exp(log_variance) - fit.floor) * mean_scale

    return {
        "unconditional": transform(unconditional_score),
        "conditioned": transform(conditioned_score),
    }


def rolling_score_ensembles(
    fit: FactorMixture,
    training_score: np.ndarray,
    actual_score: np.ndarray,
    ensemble_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if ensemble_size < 32:
        raise ValueError("ensemble size must be at least 32")
    posterior = filter_factor_observations(fit, training_score)
    observation_std = math.sqrt(factor_observation_variance(fit))
    unconditional = rng.normal(
        0.0,
        observation_std,
        (actual_score.size, ensemble_size),
    )
    conditioned = np.empty_like(unconditional)
    for index, observed in enumerate(actual_score):
        mean, variance = factor_next_observation(fit, posterior)
        conditioned[index] = rng.normal(
            mean,
            math.sqrt(max(variance, 1e-12)),
            ensemble_size,
        )
        posterior = update_factor_posterior(fit, posterior, float(observed))
    return unconditional, conditioned


def filter_factor_observations(
    fit: FactorMixture,
    observations: np.ndarray,
) -> GaussianPosterior:
    state_count = fit.timescales.size
    posterior = GaussianPosterior(
        mean=np.zeros(state_count, dtype=np.float64),
        covariance=np.eye(state_count, dtype=np.float64),
    )
    for value in np.asarray(observations, dtype=np.float64):
        posterior = update_factor_posterior(fit, posterior, float(value))
    return posterior


def update_factor_posterior(
    fit: FactorMixture,
    posterior: GaussianPosterior,
    observation: float,
) -> GaussianPosterior:
    phi = np.exp(-1.0 / fit.timescales)
    observation_weights = np.sqrt(np.maximum(fit.weights, 0.0))
    predicted_mean = phi * posterior.mean
    predicted_covariance = (
        phi[:, None] * posterior.covariance * phi[None, :]
        + np.diag(1.0 - phi * phi)
    )
    projected = predicted_covariance @ observation_weights
    innovation_variance = float(
        observation_weights @ projected + fit.white_variance
    )
    if innovation_variance <= 1e-12:
        return GaussianPosterior(predicted_mean, predicted_covariance)
    gain = projected / innovation_variance
    innovation = observation - float(observation_weights @ predicted_mean)
    updated_mean = predicted_mean + gain * innovation
    updated_covariance = predicted_covariance - np.outer(gain, projected)
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return GaussianPosterior(updated_mean, updated_covariance)


def factor_next_observation(
    fit: FactorMixture,
    posterior: GaussianPosterior,
) -> tuple[float, float]:
    phi = np.exp(-1.0 / fit.timescales)
    observation_weights = np.sqrt(np.maximum(fit.weights, 0.0))
    predicted_mean = phi * posterior.mean
    predicted_covariance = (
        phi[:, None] * posterior.covariance * phi[None, :]
        + np.diag(1.0 - phi * phi)
    )
    mean = float(observation_weights @ predicted_mean)
    variance = float(
        observation_weights @ predicted_covariance @ observation_weights
        + fit.white_variance
    )
    return mean, variance


def factor_observation_variance(fit: FactorMixture) -> float:
    return float(np.sum(fit.weights) + fit.white_variance)


def forecast_intraday_paths(
    *,
    fitted: FittedProcess,
    measurements,
    outcomes: DailyOutcomes,
    return_targets: np.ndarray,
    variance_targets: np.ndarray,
    histograms: dict,
    analysis: dict,
    path_count: int,
    rng: np.random.Generator,
) -> dict:
    if path_count < 1 or path_count > return_targets.shape[1]:
        raise ValueError("intraday path count must fit inside the daily ensemble")
    training_indexes = np.arange(measurements.counts.size, dtype=np.int64)
    training_volatility_score = gaussianize_quantile_spline(
        np.log(measurements.realized_variance + fitted.variance_floor),
        fitted.variance_quantile_knots,
        fitted.variance_log_quantiles,
    )
    training_volatility_residual = training_volatility_score - seasonal_values(
        training_indexes,
        fitted.volatility_seasonal,
    )
    count_scores = discrete_midpoint_gaussian_scores(
        fitted.activity_count_probabilities
    )
    training_activity_score = count_scores[measurements.counts]
    training_activity_location = activity_location(
        training_volatility_score,
        fitted.activity_coefficients,
    )
    training_activity_residual = gaussianize_quantile_spline(
        training_activity_score - training_activity_location,
        fitted.activity_residual_quantile_knots,
        fitted.activity_residual_quantiles,
    )
    filter_minutes = min(measurements.counts.size, 180 * MINUTES_PER_DAY)
    volatility_posterior = filter_factor_observations(
        fitted.volatility_factors,
        training_volatility_residual[-filter_minutes:],
    )
    activity_posterior = filter_factor_observations(
        fitted.activity_factors,
        training_activity_residual[-filter_minutes:],
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
    test_activity_score = count_scores[test_counts]
    test_activity_residual = gaussianize_quantile_spline(
        test_activity_score
        - activity_location(test_volatility_score, fitted.activity_coefficients),
        fitted.activity_residual_quantile_knots,
        fitted.activity_residual_quantiles,
    )

    one_second_histogram = full_histogram(histograms, "1s")
    training_histogram = histogram_from_counts(
        one_second_histogram,
        measurements.one_second_histogram_counts,
    )
    calibration = calibrate_intraday_generation(
        fitted,
        measurements,
        training_volatility_score,
        training_histogram,
        rng,
    )

    scale_factors = {"1m": 1, "15m": 15, "1h": 60, "4h": 240}
    model_histogram_counts = {
        scale_id: np.zeros(
            int(full_histogram(histograms, scale_id)["binCount"]),
            dtype=np.int64,
        )
        for scale_id in ("1s", *scale_factors.keys())
    }
    feature_ensembles = {
        "minutePathRangeBps": np.empty((outcomes.returns.size, path_count)),
        "minuteReturnQuadraticVariationBpsSquared": np.empty(
            (outcomes.returns.size, path_count)
        ),
        "zeroSecondProbability": np.empty((outcomes.returns.size, path_count)),
        "maximumAbsoluteMinuteReturnBps": np.empty(
            (outcomes.returns.size, path_count)
        ),
    }
    projection_fallbacks = 0
    efficiency_stationary = GaussianPosterior(
        mean=np.zeros(fitted.efficiency_factors.timescales.size),
        covariance=np.eye(fitted.efficiency_factors.timescales.size),
    )

    for day in range(outcomes.returns.size):
        day_start = day * MINUTES_PER_DAY
        day_stop = day_start + MINUTES_PER_DAY
        indexes = test_indexes[day_start:day_stop]
        volatility_latent = sample_factor_paths(
            fitted.volatility_factors,
            volatility_posterior,
            MINUTES_PER_DAY,
            path_count,
            rng,
        )
        volatility_score = (
            seasonal_values(indexes, fitted.volatility_seasonal)[None, :]
            + volatility_latent
        )
        log_q = inverse_gaussianized_quantile_spline(
            volatility_score,
            fitted.variance_quantile_knots,
            fitted.variance_log_quantiles,
        )
        q = np.maximum(0.0, np.exp(log_q) - fitted.variance_floor)

        activity_latent = sample_factor_paths(
            fitted.activity_factors,
            activity_posterior,
            MINUTES_PER_DAY,
            path_count,
            rng,
        )
        activity_residual = inverse_gaussianized_quantile_spline(
            activity_latent,
            fitted.activity_residual_quantile_knots,
            fitted.activity_residual_quantiles,
        )
        generated_activity_score = (
            activity_location(volatility_score, fitted.activity_coefficients)
            + activity_residual
            + calibration["activityScoreOffset"]
        )
        counts = inverse_discrete_gaussian_copula(
            generated_activity_score,
            fitted.activity_count_probabilities,
        ).astype(np.uint8)
        q[counts == 0] = 0.0
        q_totals = np.sum(q, axis=1)
        q *= np.divide(
            variance_targets[day, :path_count],
            q_totals,
            out=np.ones(path_count),
            where=q_totals > 0,
        )[:, None]

        efficiency_latent = sample_factor_paths(
            fitted.efficiency_factors,
            efficiency_stationary,
            MINUTES_PER_DAY,
            path_count,
            rng,
        )
        efficiency_coefficients = fitted.efficiency_volatility_coefficients
        efficiency_score = (
            efficiency_coefficients[0]
            + efficiency_coefficients[1] * volatility_score
            + efficiency_coefficients[2]
            * (volatility_score * volatility_score - 1.0)
            + fitted.efficiency_residual_std * efficiency_latent
        )
        efficiency = inverse_gaussianized_quantile_spline(
            efficiency_score,
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
        efficiency[single] = np.where(efficiency[single] >= 0.0, 1.0, -1.0)
        minute_returns = efficiency * np.sqrt(q)
        projected, fallback_count = project_returns_to_daily_targets(
            minute_returns.reshape(-1),
            q.reshape(-1),
            counts.reshape(-1),
            return_targets[day, :path_count],
            aligned=False,
        )
        minute_returns = projected.reshape(path_count, MINUTES_PER_DAY)
        projection_fallbacks += fallback_count

        cumulative = np.concatenate((
            np.zeros((path_count, 1)),
            np.cumsum(minute_returns, axis=1),
        ), axis=1)
        feature_ensembles["minutePathRangeBps"][day] = (
            np.max(cumulative, axis=1) - np.min(cumulative, axis=1)
        )
        feature_ensembles["minuteReturnQuadraticVariationBpsSquared"][day] = (
            np.sum(minute_returns * minute_returns, axis=1)
        )
        feature_ensembles["zeroSecondProbability"][day] = (
            1.0 - np.mean(counts, axis=1) / 60.0
        )
        feature_ensembles["maximumAbsoluteMinuteReturnBps"][day] = np.max(
            np.abs(minute_returns),
            axis=1,
        )

        for scale_id, factor in scale_factors.items():
            values = (
                minute_returns.reshape(-1)
                if factor == 1
                else np.sum(
                    minute_returns.reshape(
                        path_count,
                        MINUTES_PER_DAY // factor,
                        factor,
                    ),
                    axis=2,
                ).reshape(-1)
            )
            model_histogram_counts[scale_id] += bounded_histogram_counts(
                values,
                histogram_edges(full_histogram(histograms, scale_id)),
            )

        _, second_returns = generate_projected_second_batch(
            fitted,
            counts=counts[0],
            volatility_score=volatility_score[0],
            target_minute_returns=minute_returns[0],
            realized_variance=q[0],
            activity_timing_rho=calibration["activityTimingGaussianRho"],
            magnitude_dispersion_scale=calibration["magnitudeDispersionScale"],
            micro_probability_scale=calibration["microProbabilityScale"],
            rng=rng,
        )
        model_histogram_counts["1s"] += bounded_histogram_counts(
            second_returns,
            histogram_edges(one_second_histogram),
        )

        for value in test_volatility_residual[day_start:day_stop]:
            volatility_posterior = update_factor_posterior(
                fitted.volatility_factors,
                volatility_posterior,
                float(value),
            )
        for value in test_activity_residual[day_start:day_stop]:
            activity_posterior = update_factor_posterior(
                fitted.activity_factors,
                activity_posterior,
                float(value),
            )

    actual_features = {
        "minutePathRangeBps": minute_path_range(outcomes.minute_returns),
        "minuteReturnQuadraticVariationBpsSquared": np.sum(
            outcomes.minute_returns * outcomes.minute_returns,
            axis=1,
        ),
        "zeroSecondProbability": outcomes.zero_probability,
        "maximumAbsoluteMinuteReturnBps": np.max(
            np.abs(outcomes.minute_returns),
            axis=1,
        ),
    }
    feature_metrics = {
        name: {
            **ensemble_metrics(actual_features[name], ensemble),
            "assessment": feature_calibration_assessment(
                ensemble_metrics(actual_features[name], ensemble)
            ),
        }
        for name, ensemble in feature_ensembles.items()
    }
    multiscale = {}
    for scale_id in ("1s", *scale_factors.keys()):
        if scale_id == "1s":
            actual_counts = outcomes.one_second_histogram_counts
        else:
            factor = scale_factors[scale_id]
            actual_values = (
                outcomes.minute_returns.reshape(-1)
                if factor == 1
                else np.sum(
                    outcomes.minute_returns.reshape(
                        outcomes.returns.size,
                        MINUTES_PER_DAY // factor,
                        factor,
                    ),
                    axis=2,
                ).reshape(-1)
            )
            actual_counts = bounded_histogram_counts(
                actual_values,
                histogram_edges(full_histogram(histograms, scale_id)),
            )
        multiscale[scale_id] = summarize_histogram_forecast(
            actual_counts,
            model_histogram_counts[scale_id],
            full_histogram(histograms, scale_id),
            repetitions=500 if scale_id == "1s" else 1_000,
        )

    return {
        "ensemblePathsPerDay": path_count,
        "oneSecondPathsPerDay": 1,
        "calibration": calibration,
        "dailyProjectionFallbacks": projection_fallbacks,
        "pathFeatureMetrics": feature_metrics,
        "multiscaleReturnDistributions": multiscale,
    }


def calibrate_intraday_generation(
    fitted: FittedProcess,
    measurements,
    training_volatility_score: np.ndarray,
    training_histogram: dict,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample_count = 250_000
    burn = 30 * MINUTES_PER_DAY
    volatility_generator = ArFactorGenerator(fitted.volatility_factors, rng)
    activity_generator = ArFactorGenerator(fitted.activity_factors, rng)
    volatility_score = (
        seasonal_values(np.arange(sample_count), fitted.volatility_seasonal)
        + volatility_generator.draw(sample_count + burn)[burn:]
    )
    activity_residual = inverse_gaussianized_quantile_spline(
        activity_generator.draw(sample_count + burn)[burn:],
        fitted.activity_residual_quantile_knots,
        fitted.activity_residual_quantiles,
    )
    raw_activity_score = (
        activity_location(volatility_score, fitted.activity_coefficients)
        + activity_residual
    )
    activity_offset = calibrate_discrete_location(
        raw_activity_score,
        fitted.activity_count_probabilities,
        fitted.target_mean_activity_count,
    )
    counts = inverse_discrete_gaussian_copula(
        raw_activity_score + activity_offset,
        fitted.activity_count_probabilities,
    ).astype(np.uint8)
    log_q = inverse_gaussianized_quantile_spline(
        volatility_score,
        fitted.variance_quantile_knots,
        fitted.variance_log_quantiles,
    )
    q = np.maximum(0.0, np.exp(log_q) - fitted.variance_floor)
    q[counts == 0] = 0.0
    q *= fitted.target_mean_realized_variance / np.mean(q)
    efficiency_generator = ArFactorGenerator(fitted.efficiency_factors, rng)
    efficiency_latent = efficiency_generator.draw(sample_count + burn)[burn:]
    coefficients = fitted.efficiency_volatility_coefficients
    efficiency_score = (
        coefficients[0]
        + coefficients[1] * volatility_score
        + coefficients[2] * (volatility_score * volatility_score - 1.0)
        + fitted.efficiency_residual_std * efficiency_latent
    )
    efficiency = inverse_gaussianized_quantile_spline(
        efficiency_score,
        fitted.efficiency_quantile_knots,
        fitted.efficiency_quantiles,
    )
    efficiency[counts == 0] = 0.0
    maximum = np.sqrt(counts.astype(np.float64))
    efficiency = np.clip(
        efficiency,
        -maximum * (1.0 - 1e-10),
        maximum * (1.0 - 1e-10),
    )
    single = counts == 1
    efficiency[single] = np.where(efficiency[single] >= 0.0, 1.0, -1.0)
    efficiency_scale = math.sqrt(
        fitted.target_minute_return_variance
        / float(np.var(efficiency * np.sqrt(q)))
    )
    activity_rho = calibrate_activity_rho(
        measurements.counts,
        fitted.target_adjacent_activity_probability,
        rng,
    )
    dispersion, micro, _ = calibrate_magnitude_dispersion(
        fitted,
        counts=measurements.counts,
        volatility_score=training_volatility_score,
        target_minute_returns=measurements.minute_returns,
        realized_variance=measurements.realized_variance,
        activity_timing_rho=activity_rho,
        histogram=training_histogram,
    )
    return {
        "activityScoreOffset": float(activity_offset),
        "efficiencyScale": float(efficiency_scale),
        "activityTimingGaussianRho": float(activity_rho),
        "magnitudeDispersionScale": float(dispersion),
        "microProbabilityScale": float(micro),
    }


def sample_factor_paths(
    fit: FactorMixture,
    posterior: GaussianPosterior,
    steps: int,
    paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(posterior.covariance)
    covariance_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    state = (
        posterior.mean[None, :]
        + rng.standard_normal((paths, fit.timescales.size)) @ covariance_root.T
    )
    phi = np.exp(-1.0 / fit.timescales)
    innovation_scale = np.sqrt(1.0 - phi * phi)
    observation_weights = np.sqrt(np.maximum(fit.weights, 0.0))
    result = np.empty((paths, steps), dtype=np.float64)
    for step in range(steps):
        state = (
            phi[None, :] * state
            + innovation_scale[None, :]
            * rng.standard_normal((paths, fit.timescales.size))
        )
        result[:, step] = state @ observation_weights
        if fit.white_variance > 0:
            result[:, step] += math.sqrt(fit.white_variance) * rng.standard_normal(paths)
    return result


def activity_location(score: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return (
        coefficients[0]
        + coefficients[1] * score
        + coefficients[2] * (score * score - 1.0)
    )


def histogram_from_counts(template: dict, counts: np.ndarray | None) -> dict:
    if counts is None or np.sum(counts) <= 0:
        raise ValueError("training histogram counts are required")
    probabilities = counts.astype(np.float64) / np.sum(counts)
    return {
        **template,
        "observations": int(np.sum(counts)),
        "nonzeroBins": [
            [int(index), float(value)]
            for index, value in enumerate(probabilities)
            if value > 0
        ],
    }


def minute_path_range(minute_returns: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate((
        np.zeros((minute_returns.shape[0], 1)),
        np.cumsum(minute_returns, axis=1),
    ), axis=1)
    return np.max(cumulative, axis=1) - np.min(cumulative, axis=1)


def feature_calibration_assessment(metrics: dict) -> dict:
    maximum_error = max(
        abs(values["coverageError"])
        for values in metrics["centralIntervals"].values()
    )
    return {
        "pitUniformAtFivePercent": metrics["pit"]["ksPValue"] >= 0.05,
        "maximumAbsoluteCoverageError": float(maximum_error),
        "allCentralCoverageErrorsWithinFivePoints": maximum_error <= 0.05,
    }


def summarize_histogram_forecast(
    actual_counts: np.ndarray,
    model_counts: np.ndarray,
    histogram: dict,
    *,
    repetitions: int,
) -> dict:
    actual_probability = actual_counts.astype(np.float64) / np.sum(actual_counts)
    model_probability = model_counts.astype(np.float64) / np.sum(model_counts)
    centers = histogram_centers(histogram)
    actual_mean = float(np.sum(actual_probability * centers))
    model_mean = float(np.sum(model_probability * centers))
    actual_sigma = math.sqrt(float(np.sum(
        actual_probability * (centers - actual_mean) ** 2
    )))
    model_sigma = math.sqrt(float(np.sum(
        model_probability * (centers - model_mean) ** 2
    )))
    center = np.abs(centers) < 0.25 * actual_sigma
    tail3 = np.abs(centers) >= 3.0 * actual_sigma
    tail5 = np.abs(centers) >= 5.0 * actual_sigma

    def ratio(mask: np.ndarray) -> dict:
        observed = float(np.sum(actual_probability[mask]))
        model = float(np.sum(model_probability[mask]))
        return {
            "observed": observed,
            "model": model,
            "ratioObservedOverModel": observed / model if model > 0 else None,
        }

    return {
        "actualObservations": int(np.sum(actual_counts)),
        "modelObservations": int(np.sum(model_counts)),
        "jsDivergenceBits": jensen_shannon_bits(
            actual_probability,
            model_probability,
        ),
        "varianceRatioObservedOverModel": (actual_sigma / model_sigma) ** 2,
        "centralMass": ratio(center),
        "threeSigmaTail": ratio(tail3),
        "fiveSigmaTail": ratio(tail5),
        "finiteHoldoutSamplingFloor": js_sampling_floor(
            model_probability,
            int(np.sum(actual_counts)),
            repetitions=repetitions,
        ),
    }


def read_daily_outcomes(
    files: list[Path],
    previous_close: float,
    one_second_edges: np.ndarray,
) -> DailyOutcomes:
    returns = np.empty(len(files), dtype=np.float64)
    variance = np.empty(len(files), dtype=np.float64)
    zero_probability = np.empty(len(files), dtype=np.float64)
    intraday_range = np.empty(len(files), dtype=np.float64)
    minute_returns = np.empty((len(files), MINUTES_PER_DAY), dtype=np.float64)
    minute_realized_variance = np.empty_like(minute_returns)
    minute_activity_counts = np.empty(
        (len(files), MINUTES_PER_DAY),
        dtype=np.uint8,
    )
    one_second_histogram_counts = np.zeros(
        one_second_edges.size - 1,
        dtype=np.int64,
    )
    last_close = previous_close
    for index, file in enumerate(files):
        closes = read_candle_column(file, "close")
        if closes.shape != (DAY_SECONDS,) or np.any(~np.isfinite(closes)):
            raise ValueError(f"invalid one-second closes: {file}")
        day_returns = np.diff(np.log(np.concatenate((
            np.asarray([last_close], dtype=np.float64),
            closes,
        )))) * 10_000.0
        returns[index] = float(np.sum(day_returns))
        variance[index] = float(np.sum(day_returns * day_returns))
        zero_probability[index] = float(np.mean(day_returns == 0.0))
        log_path = np.log(np.concatenate((
            np.asarray([last_close], dtype=np.float64),
            closes,
        ))) * 10_000.0
        intraday_range[index] = float(np.max(log_path) - np.min(log_path))
        minute_returns[index] = np.sum(
            day_returns.reshape(MINUTES_PER_DAY, 60),
            axis=1,
        )
        rows = day_returns.reshape(MINUTES_PER_DAY, 60)
        minute_realized_variance[index] = np.sum(rows * rows, axis=1)
        minute_activity_counts[index] = np.sum(
            rows != 0.0,
            axis=1,
            dtype=np.uint8,
        )
        one_second_histogram_counts += bounded_histogram_counts(
            day_returns,
            one_second_edges,
        )
        last_close = float(closes[-1])
    return DailyOutcomes(
        returns=returns,
        realized_variance=variance,
        zero_probability=zero_probability,
        intraday_range=intraday_range,
        minute_returns=minute_returns,
        minute_realized_variance=minute_realized_variance,
        minute_activity_counts=minute_activity_counts,
        one_second_histogram_counts=one_second_histogram_counts,
    )


def ensemble_metrics(actual: np.ndarray, ensemble: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    ensemble = np.asarray(ensemble, dtype=np.float64)
    if ensemble.shape[0] != actual.size:
        raise ValueError("one ensemble row is required per actual value")
    sorted_ensemble = np.sort(ensemble, axis=1)
    sample_count = ensemble.shape[1]
    pit = (
        np.sum(ensemble < actual[:, None], axis=1)
        + 0.5 * np.sum(ensemble == actual[:, None], axis=1)
        + 0.5
    ) / (sample_count + 1.0)
    first_crps_term = np.mean(np.abs(ensemble - actual[:, None]), axis=1)
    coefficients = 2.0 * np.arange(1, sample_count + 1) - sample_count - 1.0
    second_crps_term = np.sum(
        sorted_ensemble * coefficients[None, :],
        axis=1,
    ) / (sample_count * sample_count)
    crps = first_crps_term - second_crps_term
    intervals = {}
    for level in (0.5, 0.8, 0.9, 0.95):
        tail = 0.5 * (1.0 - level)
        lower = np.quantile(ensemble, tail, axis=1)
        upper = np.quantile(ensemble, 1.0 - tail, axis=1)
        intervals[str(level)] = {
            "targetCoverage": level,
            "observedCoverage": float(np.mean(
                (actual >= lower) & (actual <= upper)
            )),
            "coverageError": float(np.mean(
                (actual >= lower) & (actual <= upper)
            ) - level),
            "meanWidth": float(np.mean(upper - lower)),
        }
    ks = stats.kstest(pit, "uniform")
    log_scores = kernel_log_scores(actual, ensemble)
    median = np.median(ensemble, axis=1)
    return {
        "observations": int(actual.size),
        "ensembleSize": int(sample_count),
        "meanCrps": float(np.mean(crps)),
        "normalizedCrpsByActualStd": float(np.mean(crps) / np.std(actual)),
        "meanNegativeLogDensity": float(np.mean(log_scores)),
        "medianBias": float(np.mean(median - actual)),
        "meanAbsoluteMedianError": float(np.mean(np.abs(median - actual))),
        "pit": {
            "mean": float(np.mean(pit)),
            "variance": float(np.var(pit)),
            "uniformTargetVariance": 1.0 / 12.0,
            "ksStatistic": float(ks.statistic),
            "ksPValue": float(ks.pvalue),
            "decileCounts": np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))[0].tolist(),
            "lag1Correlation": correlation_at_lag(pit, 1),
        },
        "centralIntervals": intervals,
    }


def kernel_log_scores(actual: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    sample_count = ensemble.shape[1]
    row_std = np.std(ensemble, axis=1, ddof=1)
    bandwidth = 1.06 * row_std * sample_count ** (-0.2)
    bandwidth = np.maximum(bandwidth, np.maximum(row_std * 1e-4, 1e-12))
    normalized = (actual[:, None] - ensemble) / bandwidth[:, None]
    density = np.mean(np.exp(-0.5 * normalized * normalized), axis=1)
    density /= bandwidth * math.sqrt(2.0 * math.pi)
    return -np.log(np.maximum(density, 1e-300))


def attach_skill_scores(metrics: dict[str, dict], baseline_id: str) -> None:
    baseline = metrics[baseline_id]
    for values in metrics.values():
        values["crpsSkillVsBaseline"] = (
            1.0 - values["meanCrps"] / baseline["meanCrps"]
        )
        values["negativeLogDensityImprovementVsBaseline"] = (
            baseline["meanNegativeLogDensity"]
            - values["meanNegativeLogDensity"]
        )


def distribution_comparison(
    *,
    training: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    histogram: dict,
) -> dict:
    edges = histogram_edges(histogram)

    def probabilities(values: np.ndarray) -> np.ndarray:
        counts = bounded_histogram_counts(values, edges).astype(np.float64)
        return counts / np.sum(counts)

    predicted_probability = probabilities(predicted)
    actual_probability = probabilities(actual)
    training_probability = probabilities(training)
    return {
        "trainingVsHoldoutJsBits": jensen_shannon_bits(
            training_probability,
            actual_probability,
        ),
        "predictedVsHoldoutJsBits": jensen_shannon_bits(
            predicted_probability,
            actual_probability,
        ),
        "holdoutSamplingFloorAgainstPredictedPopulation": js_sampling_floor(
            predicted_probability,
            actual.size,
        ),
    }


def gaussian_ensembles(
    rows: int,
    columns: int,
    mean: float,
    standard_deviation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.normal(mean, standard_deviation, (rows, columns))


def calibration_assessment(metrics: dict) -> dict:
    coverage_errors = [
        abs(values["coverageError"])
        for values in metrics["centralIntervals"].values()
    ]
    return {
        "pitUniformAtFivePercent": metrics["pit"]["ksPValue"] >= 0.05,
        "maximumAbsoluteCoverageError": float(max(coverage_errors)),
        "allCentralCoverageErrorsWithinFivePoints": max(coverage_errors) <= 0.05,
        "positiveCrpsSkillVsGaussianFamilyBaseline": (
            metrics["crpsSkillVsBaseline"] > 0.0
        ),
    }


def interpretation(
    return_metrics: dict,
    variance_metrics: dict,
    intraday: dict,
) -> dict:
    returns = return_metrics["fittedProcessStateConditioned"]
    variance = variance_metrics["fittedProcessStateConditioned"]
    return {
        "dailyReturnCalibrated": (
            returns["pit"]["ksPValue"] >= 0.05
            and max(abs(row["coverageError"]) for row in returns["centralIntervals"].values()) <= 0.05
        ),
        "dailyVarianceCalibrated": (
            variance["pit"]["ksPValue"] >= 0.05
            and max(abs(row["coverageError"]) for row in variance["centralIntervals"].values()) <= 0.05
        ),
        "stateConditioningReturnCrpsChange": (
            1.0
            - returns["meanCrps"]
            / return_metrics["fittedProcessUnconditional"]["meanCrps"]
        ),
        "stateConditioningVarianceCrpsChange": (
            1.0
            - variance["meanCrps"]
            / variance_metrics["fittedProcessUnconditional"]["meanCrps"]
        ),
        "intradayFeaturesCalibrated": all(
            values["assessment"]["pitUniformAtFivePercent"]
            and values["assessment"]["allCentralCoverageErrorsWithinFivePoints"]
            for values in intraday["pathFeatureMetrics"].values()
        ),
        "claimLimit": (
            "Passing marginal and path calibration supports probabilistic scenario use, "
            "not prediction of the realized candle path. Trading usefulness requires a "
            "separate decision/backtest audit with costs."
        ),
    }


def compact_summary(report: dict) -> dict:
    return {
        "design": report["design"],
        "dailyReturn": {
            "assessment": report["dailyReturnForecast"]["assessment"],
            "conditioned": report["dailyReturnForecast"]["metrics"][
                "fittedProcessStateConditioned"
            ],
            "distributionShift": report["dailyReturnForecast"]["distributionShift"],
        },
        "dailyVariance": {
            "assessment": report["dailyVarianceForecast"]["assessment"],
            "conditioned": report["dailyVarianceForecast"]["metrics"][
                "fittedProcessStateConditioned"
            ],
        },
        "intradayCandleForecast": report["intradayCandleForecast"],
        "interpretation": report["interpretation"],
    }


def describe(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "standardDeviation": float(np.std(values)),
        **quantiles(values),
    }


def correlation_at_lag(values: np.ndarray, lag: int) -> float:
    if values.size <= lag:
        return 0.0
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="data/benchmarks/log-return-distributions.json")
    parser.add_argument("--histograms", default="data/benchmarks/log-return-histograms.json")
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    parser.add_argument("--intraday-ensemble-size", type=int, default=64)
    parser.add_argument(
        "--output",
        default="data/benchmarks/parametric-next-day-holdout.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
