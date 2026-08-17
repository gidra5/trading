"""Fit and validate a component-free fitted BTCUSDT one-second process.

Unlike ``analyze_one_second_dependence_model.py``, simulation in this module
never selects a historical day, minute, activity mask, sign mask, or magnitude
profile. Historical returns are used only to estimate a finite parameter set.

The process has six layers:

1. a bounded fitted quantile-spline marginal for minute realized variance driven by a causal
   mixture of Gaussian AR(1) volatility factors plus calendar harmonics;
2. a fitted daily integrated-variance budget that normalizes the minute path;
3. a fitted categorical minute activity marginal coupled to volatility by a
   Gaussian copula and a separate causal mixture of Gaussian AR(1) factors;
4. a fitted minute signed-efficiency innovation plus a generated daily signed
   target enforced through a bounded variance-weighted projection;
5. a Gaussian-copula activity mask with the generated count fixed exactly and
   state-conditional logistic-normal squared-magnitude shares;
6. a fitted nonzero micro-return layer followed by a random
   constrained projection whose 60 generated returns have exactly the generated
   minute return and realized variance.

All target scales are sums of the same generated one-second path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import optimize, signal, stats

from analyze_one_second_dependence_model import (
    AcfAccumulator,
    aggregate_aligned_minutes,
    dense_histogram,
    full_histogram,
    full_scale,
    histogram_centers,
    histogram_edges,
    jensen_shannon_bits,
    pick_lags,
    selected_files,
    selected_lags,
    summarize_raw_model,
    variance_ratio_from_acf,
)
from trading_storage import read_candle_column


DAY_SECONDS = 86_400
MINUTE_SECONDS = 60
MINUTES_PER_DAY = 1_440
MINUTES_PER_WEEK = 10_080
RNG_SEED = 0xC0FFEE_1D
SIMULATION_BATCH_MINUTES = 20_000
SIMULATED_ACF_MINUTES = 100_000
MARGINAL_FIT_SAMPLE = 250_000
DEPENDENCE_FIT_SAMPLE = 500_000
ACTIVITY_CALIBRATION_MINUTES = 40_000
MAGNITUDE_CALIBRATION_MINUTES = 20_000
MAGNITUDE_MIXTURE_COMPONENTS = 3
MAGNITUDE_MIXTURE_FIT_SAMPLE = 100_000
LONG_RUN_DAILY_DAYS = 50_000
LONG_RUN_DAILY_BATCH_DAYS = 250
ONE_SECOND_JS_TARGET = 0.005
LONG_RUN_DAILY_JS_TARGET = 0.01

MINUTE_FACTOR_TIMESCALES = np.asarray(
    [1.5, 5.0, 20.0, 90.0, 360.0, 1_440.0, 10_080.0, 43_200.0],
    dtype=np.float64,
)
SECOND_SIGN_TIMESCALES = np.asarray(
    [0.75, 2.0, 6.0, 20.0, 90.0, 600.0],
    dtype=np.float64,
)
MINUTE_DEPENDENCE_LAGS = np.asarray(
    [1, 2, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1_440,
     2_880, 7_200, 10_080, 20_160, 43_200],
    dtype=np.int64,
)
SIGN_FIT_LAGS = np.asarray([1, 2, 3, 5, 10, 15, 30, 45, 60], dtype=np.int64)
DAILY_FACTOR_TIMESCALES = np.asarray(
    [1.5, 5.0, 20.0, 90.0, 365.0, 1_095.0],
    dtype=np.float64,
)
DAILY_DEPENDENCE_LAGS = np.asarray(
    [1, 2, 3, 5, 7, 14, 30, 60, 90, 180, 365],
    dtype=np.int64,
)


@dataclass(frozen=True)
class SeasonalFit:
    coefficients: np.ndarray
    daily_harmonics: int
    weekly_harmonics: int


@dataclass(frozen=True)
class FactorMixture:
    timescales: np.ndarray
    weights: np.ndarray
    white_variance: float
    target_lags: np.ndarray
    target_covariance: np.ndarray
    fitted_covariance: np.ndarray


@dataclass(frozen=True)
class DailyVarianceFit:
    floor: float
    quantile_knots: np.ndarray
    log_quantiles: np.ndarray
    factors: FactorMixture
    target_mean: float


@dataclass(frozen=True)
class DailyReturnFit:
    quantile_knots: np.ndarray
    return_quantiles: np.ndarray
    factors: FactorMixture


@dataclass(frozen=True)
class ConditionalMagnitudeMixture:
    activity_edges: np.ndarray
    volatility_edges: np.ndarray
    component_weights: np.ndarray
    log_concentration_means: np.ndarray
    log_concentration_stds: np.ndarray
    share_score_rho: float
    micro_probabilities: np.ndarray
    micro_threshold_bps: float


@dataclass(frozen=True)
class SourceMeasurements:
    counts: np.ndarray
    realized_variance: np.ndarray
    minute_returns: np.ndarray
    positive_probability: float
    active_sign_products: np.ndarray
    active_sign_pairs: np.ndarray
    adjacent_activity_probability: float
    dirichlet_concentration: float
    minute_log_concentration: np.ndarray
    minute_micro_counts: np.ndarray
    micro_threshold_bps: float
    magnitude_share_score_rho: float
    daily_realized_variance: np.ndarray
    daily_returns: np.ndarray
    one_second_histogram_counts: np.ndarray | None
    returns: int
    days: int


@dataclass(frozen=True)
class FittedProcess:
    variance_floor: float
    variance_quantile_knots: np.ndarray
    variance_log_quantiles: np.ndarray
    volatility_seasonal: SeasonalFit
    volatility_factors: FactorMixture
    daily_variance: DailyVarianceFit
    daily_return: DailyReturnFit
    activity_coefficients: np.ndarray
    activity_residual_quantile_knots: np.ndarray
    activity_residual_quantiles: np.ndarray
    activity_factors: FactorMixture
    activity_count_probabilities: np.ndarray
    target_adjacent_activity_probability: float
    magnitude_mixture: ConditionalMagnitudeMixture
    positive_probability: float
    efficiency_quantile_knots: np.ndarray
    efficiency_quantiles: np.ndarray
    efficiency_volatility_coefficients: np.ndarray
    efficiency_residual_std: float
    efficiency_factors: FactorMixture
    target_mean_realized_variance: float
    target_minute_return_variance: float
    target_mean_activity_count: float


class ArFactorGenerator:
    """Streaming stationary Gaussian AR(1) mixture."""

    def __init__(self, fit: FactorMixture, rng: np.random.Generator) -> None:
        self.fit = fit
        self.rng = rng
        self.phi = np.exp(-1.0 / fit.timescales)
        self.state = rng.standard_normal(fit.timescales.size)

    def draw(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("factor draw count cannot be negative")
        result = np.zeros(count, dtype=np.float64)
        for index, (phi, weight) in enumerate(zip(self.phi, self.fit.weights)):
            if weight <= 0:
                continue
            innovations = self.rng.standard_normal(count)
            scale = math.sqrt(max(0.0, 1.0 - phi * phi))
            values, final = signal.lfilter(
                [scale],
                [1.0, -phi],
                innovations,
                zi=[phi * self.state[index]],
            )
            self.state[index] = float(values[-1]) if count else self.state[index]
            result += math.sqrt(weight) * values
        if self.fit.white_variance > 0:
            result += math.sqrt(self.fit.white_variance) * self.rng.standard_normal(count)
        return result


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis = read_json(repo / args.analysis)
    histograms = read_json(repo / args.histograms)
    bootstrap = read_json(repo / args.bootstrap)
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    start = datetime.fromisoformat(
        analysis["fullHistory"]["startTime"].replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(
        analysis["fullHistory"]["endTime"].replace("Z", "+00:00")
    )
    files = selected_files(source, start, end)
    if not files:
        raise RuntimeError("no complete one-second shards selected")

    one_second_histogram = full_histogram(histograms, "1s")
    measurements = measure_source(
        files,
        micro_threshold_bps=0.5 * float(one_second_histogram["binWidthBps"]),
    )
    fitted = fit_process(measurements)
    generated = simulate_process(
        fitted,
        minute_count=measurements.counts.size,
        one_second_histogram=one_second_histogram,
        one_second_sigma=float(full_scale(analysis, "1s")["standardDeviationBps"]),
    )

    validation = validate_scales(
        generated,
        histograms=histograms,
        analysis=analysis,
    )
    long_run_daily_returns = simulate_long_run_daily_returns(
        fitted,
        days=LONG_RUN_DAILY_DAYS,
        activity_score_offset=float(generated["activityScoreOffset"]),
        efficiency_scale=float(generated["efficiencyScale"]),
        rng=np.random.default_rng(RNG_SEED + 100),
    )
    daily_histogram = full_histogram(histograms, "1d")
    long_run_daily_validation = summarize_raw_model(
        "fittedStatisticalProcessLongRun",
        "Long-run direct aggregation of the fitted one-second process",
        long_run_daily_returns,
        dense_histogram(daily_histogram),
        daily_histogram,
        float(full_scale(analysis, "1d")["standardDeviationBps"]),
    )
    lags = selected_lags()
    model_acf = generated["acf"]
    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "commonEndTime": analysis["source"]["commonAnalysisEndTime"],
        "scope": {
            "fitReturns": measurements.returns,
            "fitMinutes": int(measurements.counts.size),
            "fitDays": measurements.days,
            "generatedMinutes": int(measurements.counts.size),
            "longRunGeneratedDays": LONG_RUN_DAILY_DAYS,
            "historicalComponentsSampledDuringGeneration": False,
            "separateJumpProcess": False,
        },
        "model": model_description(fitted),
        "fittedParameters": serialize_fit(fitted),
        "fitDiagnostics": {
            "observedMeanActiveSeconds": float(np.mean(measurements.counts)),
            "generatedMeanActiveSeconds": float(np.mean(generated["counts"])),
            "observedZeroSecondProbability": float(
                1.0 - np.mean(measurements.counts) / MINUTE_SECONDS
            ),
            "generatedZeroSecondProbability": float(
                1.0 - np.mean(generated["counts"]) / MINUTE_SECONDS
            ),
            "observedZeroActivityMinuteProbability": float(
                np.mean(measurements.counts == 0)
            ),
            "generatedZeroActivityMinuteProbability": float(
                np.mean(generated["counts"] == 0)
            ),
            "observedAdjacentActivityProbability": measurements.adjacent_activity_probability,
            "generatedAdjacentActivityProbability": generated["adjacentActivityProbability"],
            "activityScoreOffset": generated["activityScoreOffset"],
            "activityTimingGaussianRho": generated["activityTimingGaussianRho"],
            "observedMinuteVarianceQuantiles": quantiles(measurements.realized_variance),
            "generatedMinuteVarianceQuantiles": quantiles(generated["realizedVariance"]),
            "observedMinuteVarianceCorrelations": selected_correlations(
                np.log(measurements.realized_variance + fitted.variance_floor)
            ),
            "generatedMinuteVarianceCorrelations": selected_correlations(
                np.log(generated["realizedVariance"] + fitted.variance_floor)
            ),
            "observedDailyVarianceQuantiles": quantiles(
                measurements.daily_realized_variance
            ),
            "generatedDailyVarianceQuantiles": quantiles(
                generated["dailyVarianceBudgets"]
            ),
            "observedDailyLogVarianceCorrelations": selected_daily_correlations(
                np.log(
                    measurements.daily_realized_variance
                    + fitted.daily_variance.floor
                )
            ),
            "generatedDailyLogVarianceCorrelations": selected_daily_correlations(
                np.log(
                    generated["dailyVarianceBudgets"]
                    + fitted.daily_variance.floor
                )
            ),
            "observedDailyReturnQuantiles": quantiles(measurements.daily_returns),
            "generatedDailyReturnTargetQuantiles": quantiles(
                generated["dailyReturnTargets"]
            ),
            "observedDailyReturnCorrelations": selected_daily_correlations(
                measurements.daily_returns
            ),
            "generatedDailyReturnTargetCorrelations": selected_daily_correlations(
                generated["dailyReturnTargets"]
            ),
            "dailyProjectionFallbacks": generated["dailyProjectionFallbacks"],
            "observedActivityCountCorrelations": selected_correlations(
                measurements.counts.astype(np.float64)
            ),
            "generatedActivityCountCorrelations": selected_correlations(
                generated["counts"].astype(np.float64)
            ),
            "observedMagnitudeShareScoreRho": measurements.magnitude_share_score_rho,
            "fittedMagnitudeShareScoreRho": fitted.magnitude_mixture.share_score_rho,
            "magnitudeDispersionScale": generated["magnitudeDispersionScale"],
            "microProbabilityScale": generated["microProbabilityScale"],
            "magnitudeDispersionCalibration": generated["magnitudeCalibration"],
        },
        "validation": {
            "modelsByScale": validation,
            "longRunDaily": long_run_daily_validation,
            "acceptance": {
                "oneSecondJsTargetBits": ONE_SECOND_JS_TARGET,
                "oneSecondJsBits": validation["1s"][0]["jsDivergenceBits"],
                "oneSecondPassed": (
                    validation["1s"][0]["jsDivergenceBits"]
                    < ONE_SECOND_JS_TARGET
                ),
                "longRunDailyJsTargetBits": LONG_RUN_DAILY_JS_TARGET,
                "longRunDailyJsBits": long_run_daily_validation["jsDivergenceBits"],
                "longRunDailyPassed": (
                    long_run_daily_validation["jsDivergenceBits"]
                    < LONG_RUN_DAILY_JS_TARGET
                ),
            },
            "bootstrapBenchmarkByScale": extract_bootstrap_benchmark(bootstrap),
            "oneSecondModelProbabilities": generated["oneSecondProbabilities"].tolist(),
            "dailyJsSamplingFloor": js_sampling_floor(
                dense_histogram(full_histogram(histograms, "1d")),
                int(validation["1d"][0]["observations"]),
            ),
            "acfSampleMinutes": SIMULATED_ACF_MINUTES,
            "modelAcf": {
                "lagsSeconds": lags,
                "returnCorrelation": pick_lags(model_acf["return"], lags),
                "absoluteReturnCorrelation": pick_lags(model_acf["absoluteReturn"], lags),
                "logAbsoluteReturnCorrelation": pick_lags(
                    model_acf["logAbsoluteReturn"], lags
                ),
                "squaredReturnCorrelation": pick_lags(model_acf["squaredReturn"], lags),
                "activityCorrelation": pick_lags(model_acf["activity"], lags),
                "varianceRatioFromReturnAcf": variance_ratio_from_acf(
                    model_acf["return"], 60
                ),
            },
        },
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(compact_summary(report), indent=2))
    acceptance = report["validation"]["acceptance"]
    if not acceptance["oneSecondPassed"] or not acceptance["longRunDailyPassed"]:
        raise RuntimeError(
            "parametric process failed JS acceptance targets: "
            f"1s={acceptance['oneSecondJsBits']:.6g}, "
            f"1d-long-run={acceptance['longRunDailyJsBits']:.6g}"
        )


def measure_source(
    files: list[Path],
    *,
    micro_threshold_bps: float,
    one_second_histogram_edges: np.ndarray | None = None,
) -> SourceMeasurements:
    minute_count = len(files) * MINUTES_PER_DAY - 1
    counts = np.empty(minute_count, dtype=np.uint8)
    realized_variance = np.empty(minute_count, dtype=np.float64)
    minute_returns = np.empty(minute_count, dtype=np.float64)
    minute_log_concentration = np.full(minute_count, np.nan, dtype=np.float32)
    minute_micro_counts = np.empty(minute_count, dtype=np.uint8)
    sign_products = np.zeros(SIGN_FIT_LAGS.size, dtype=np.float64)
    sign_pairs = np.zeros(SIGN_FIT_LAGS.size, dtype=np.int64)
    positive = 0
    active_total = 0
    adjacent_active = 0
    adjacent_pairs = 0
    dirichlet_numerator = 0.0
    dirichlet_denominator = 0
    magnitude_share_cross = 0.0
    magnitude_share_pairs = 0
    cursor = 0
    previous_close: float | None = None
    source_histogram_counts = (
        np.zeros(one_second_histogram_edges.size - 1, dtype=np.int64)
        if one_second_histogram_edges is not None
        else None
    )

    for file_index, reference in enumerate(files):
        if file_index % 100 == 0:
            print(f"Reading parametric fit source {file_index}/{len(files)}...", file=sys.stderr)
        closes = read_candle_column(reference, "close")
        if closes.shape != (DAY_SECONDS,) or np.any(~np.isfinite(closes)) or np.any(closes <= 0):
            raise ValueError(f"invalid one-second closes: {reference}")
        returns = np.empty(DAY_SECONDS, dtype=np.float64)
        returns[1:] = np.diff(np.log(closes)) * 10_000.0
        returns[0] = (
            np.nan
            if previous_close is None
            else math.log(closes[0] / previous_close) * 10_000.0
        )
        previous_close = float(closes[-1])
        finite = np.isfinite(returns)
        if source_histogram_counts is not None:
            source_histogram_counts += bounded_histogram_counts(
                returns[finite],
                one_second_histogram_edges,
            )
        active = finite & (returns != 0.0)
        positive += int(np.count_nonzero(active & (returns > 0.0)))
        active_total += int(np.count_nonzero(active))
        for lag_index, lag in enumerate(SIGN_FIT_LAGS):
            both = active[lag:] & active[:-lag]
            if np.any(both):
                sign_products[lag_index] += float(np.sum(
                    np.sign(returns[lag:][both]) * np.sign(returns[:-lag][both])
                ))
                sign_pairs[lag_index] += int(np.count_nonzero(both))

        rows = returns.reshape(-1, MINUTE_SECONDS)
        valid = np.all(np.isfinite(rows), axis=1)
        rows = rows[valid]
        row_active = rows != 0.0
        n = np.sum(row_active, axis=1, dtype=np.uint8)
        q = np.sum(rows * rows, axis=1, dtype=np.float64)
        r = np.sum(rows, axis=1, dtype=np.float64)
        micro_count = np.sum(
            row_active & (np.abs(rows) < micro_threshold_bps),
            axis=1,
            dtype=np.uint8,
        )
        adjacent_active += int(np.count_nonzero(row_active[:, 1:] & row_active[:, :-1]))
        adjacent_pairs += int(rows.shape[0] * (MINUTE_SECONDS - 1))
        selected = (n > 1) & (q > 0)
        batch_log_concentration = np.full(rows.shape[0], np.nan, dtype=np.float32)
        if np.any(selected):
            selected_active = row_active[selected]
            shares = np.divide(
                rows[selected] * rows[selected],
                q[selected, None],
                out=np.zeros_like(rows[selected]),
                where=q[selected, None] > 0,
            )
            concentration = np.sum(shares * shares, axis=1)
            alpha = dirichlet_alpha_from_simpson(n[selected].astype(np.float64), concentration)
            valid_alpha = np.isfinite(alpha) & (alpha > 0)
            dirichlet_numerator += float(np.sum(np.log(alpha[valid_alpha])))
            dirichlet_denominator += int(np.count_nonzero(valid_alpha))
            selected_locations = np.flatnonzero(selected)
            batch_log_concentration[selected_locations[valid_alpha]] = np.log(
                alpha[valid_alpha]
            ).astype(np.float32)

            log_shares = np.where(
                selected_active,
                np.log(np.maximum(shares, 1e-300)),
                0.0,
            )
            selected_counts = n[selected].astype(np.float64)
            log_means = np.sum(log_shares, axis=1) / selected_counts
            centered_log_shares = (log_shares - log_means[:, None]) * selected_active
            log_variance = np.sum(
                centered_log_shares * centered_log_shares,
                axis=1,
            ) / selected_counts
            usable_rows = log_variance > 1e-12
            normalized_log_shares = np.divide(
                centered_log_shares,
                np.sqrt(log_variance)[:, None],
                out=np.zeros_like(centered_log_shares),
                where=np.sqrt(log_variance)[:, None] > 0,
            )
            adjacent_selected = (
                selected_active[:, 1:]
                & selected_active[:, :-1]
                & usable_rows[:, None]
            )
            magnitude_share_cross += float(np.sum(
                normalized_log_shares[:, 1:][adjacent_selected]
                * normalized_log_shares[:, :-1][adjacent_selected]
            ))
            magnitude_share_pairs += int(np.count_nonzero(adjacent_selected))

        end_cursor = cursor + rows.shape[0]
        counts[cursor:end_cursor] = n
        realized_variance[cursor:end_cursor] = q
        minute_returns[cursor:end_cursor] = r
        minute_log_concentration[cursor:end_cursor] = batch_log_concentration
        minute_micro_counts[cursor:end_cursor] = micro_count
        cursor = end_cursor

    if cursor != minute_count:
        raise RuntimeError(f"expected {minute_count} complete minutes, found {cursor}")
    alpha = math.exp(dirichlet_numerator / dirichlet_denominator)
    return SourceMeasurements(
        counts=counts,
        realized_variance=realized_variance,
        minute_returns=minute_returns,
        positive_probability=positive / active_total,
        active_sign_products=sign_products / sign_pairs,
        active_sign_pairs=sign_pairs,
        adjacent_activity_probability=adjacent_active / adjacent_pairs,
        dirichlet_concentration=alpha,
        minute_log_concentration=minute_log_concentration,
        minute_micro_counts=minute_micro_counts,
        micro_threshold_bps=micro_threshold_bps,
        magnitude_share_score_rho=float(np.clip(
            magnitude_share_cross / magnitude_share_pairs,
            -0.95,
            0.95,
        )),
        daily_realized_variance=aggregate_aligned_minutes(
            realized_variance,
            MINUTES_PER_DAY,
        ),
        daily_returns=aggregate_aligned_minutes(
            minute_returns,
            MINUTES_PER_DAY,
        ),
        one_second_histogram_counts=source_histogram_counts,
        returns=len(files) * DAY_SECONDS - 1,
        days=len(files),
    )


def fit_process(source: SourceMeasurements) -> FittedProcess:
    positive_q = source.realized_variance[source.realized_variance > 0]
    variance_floor = float(np.quantile(positive_q, 0.001) * 0.25)
    log_q = np.log(source.realized_variance + variance_floor)
    marginal_indexes = evenly_spaced_indexes(log_q.size, MARGINAL_FIT_SAMPLE)
    marginal_values = np.sort(log_q[marginal_indexes])
    marginal_probabilities = (
        np.arange(marginal_values.size, dtype=np.float64) + 0.5
    ) / marginal_values.size
    variance_quantile_knots = np.concatenate((
        np.geomspace(0.5 / marginal_values.size, 0.01, 128, endpoint=False),
        np.linspace(0.01, 0.99, 1_024, endpoint=False),
        1.0 - np.geomspace(0.5 / marginal_values.size, 0.01, 128)[::-1],
    ))
    variance_quantile_knots = np.unique(np.clip(
        variance_quantile_knots,
        0.5 / marginal_values.size,
        1.0 - 0.5 / marginal_values.size,
    ))
    variance_log_quantiles = np.interp(
        variance_quantile_knots,
        marginal_probabilities,
        marginal_values,
    )
    volatility_score = gaussianize_quantile_spline(
        log_q,
        variance_quantile_knots,
        variance_log_quantiles,
    )
    volatility_seasonal = fit_seasonal(volatility_score)
    volatility_residual = volatility_score - seasonal_values(
        np.arange(volatility_score.size, dtype=np.int64), volatility_seasonal
    )
    volatility_factors = fit_factor_mixture(
        volatility_residual,
        timescales=MINUTE_FACTOR_TIMESCALES,
        lags=MINUTE_DEPENDENCE_LAGS,
    )

    positive_daily_variance = source.daily_realized_variance[
        source.daily_realized_variance > 0
    ]
    daily_floor = float(np.quantile(positive_daily_variance, 0.001) * 0.25)
    daily_log_variance = np.log(source.daily_realized_variance + daily_floor)
    daily_knots, daily_log_quantiles = fit_quantile_spline(daily_log_variance)
    daily_score = gaussianize_quantile_spline(
        daily_log_variance,
        daily_knots,
        daily_log_quantiles,
    )
    daily_factors = fit_factor_mixture(
        daily_score,
        timescales=DAILY_FACTOR_TIMESCALES,
        lags=DAILY_DEPENDENCE_LAGS,
    )
    daily_variance = DailyVarianceFit(
        floor=daily_floor,
        quantile_knots=daily_knots,
        log_quantiles=daily_log_quantiles,
        factors=daily_factors,
        target_mean=float(np.mean(source.daily_realized_variance)),
    )
    daily_return_knots, daily_return_quantiles = fit_quantile_spline(
        source.daily_returns
    )
    daily_return_score = gaussianize_quantile_spline(
        source.daily_returns,
        daily_return_knots,
        daily_return_quantiles,
    )
    daily_return = DailyReturnFit(
        quantile_knots=daily_return_knots,
        return_quantiles=daily_return_quantiles,
        factors=fit_factor_mixture(
            daily_return_score,
            timescales=DAILY_FACTOR_TIMESCALES,
            lags=DAILY_DEPENDENCE_LAGS,
        ),
    )

    activity_count_probabilities = (
        np.bincount(source.counts, minlength=61).astype(np.float64) + 0.5
    )
    activity_count_probabilities /= np.sum(activity_count_probabilities)
    activity_score_by_count = discrete_midpoint_gaussian_scores(
        activity_count_probabilities
    )
    activity_score = activity_score_by_count[source.counts]
    regression_indexes = evenly_spaced_indexes(activity_score.size, DEPENDENCE_FIT_SAMPLE)
    design = np.column_stack((
        np.ones(regression_indexes.size),
        volatility_score[regression_indexes],
        volatility_score[regression_indexes] ** 2 - 1.0,
    ))
    adjusted_target = activity_score[regression_indexes]
    activity_coefficients = np.linalg.lstsq(design, adjusted_target, rcond=None)[0]
    activity_location = (
        activity_coefficients[0]
        + activity_coefficients[1] * volatility_score
        + activity_coefficients[2] * (volatility_score * volatility_score - 1.0)
    )
    activity_residual = activity_score - activity_location
    activity_residual_knots, activity_residual_quantiles = fit_quantile_spline(
        activity_residual
    )
    standardized_activity_residual = gaussianize_quantile_spline(
        activity_residual,
        activity_residual_knots,
        activity_residual_quantiles,
    )
    activity_factors = fit_factor_mixture(
        standardized_activity_residual,
        timescales=MINUTE_FACTOR_TIMESCALES,
        lags=MINUTE_DEPENDENCE_LAGS,
    )
    magnitude_mixture = fit_conditional_magnitude_mixture(
        source.counts,
        volatility_score,
        source.minute_log_concentration,
        source.magnitude_share_score_rho,
        source.minute_micro_counts,
        source.micro_threshold_bps,
    )
    valid_efficiency = source.realized_variance > 0
    efficiency = np.divide(
        source.minute_returns[valid_efficiency],
        np.sqrt(source.realized_variance[valid_efficiency]),
    )
    efficiency_knots, efficiency_quantiles = fit_quantile_spline(efficiency)
    efficiency_score = gaussianize_quantile_spline(
        efficiency,
        efficiency_knots,
        efficiency_quantiles,
    )
    efficiency_volatility = volatility_score[valid_efficiency]
    efficiency_design = np.column_stack((
        np.ones(efficiency.size),
        efficiency_volatility,
        efficiency_volatility * efficiency_volatility - 1.0,
    ))
    efficiency_indexes = evenly_spaced_indexes(
        efficiency.size, DEPENDENCE_FIT_SAMPLE
    )
    efficiency_coefficients = np.linalg.lstsq(
        efficiency_design[efficiency_indexes],
        efficiency_score[efficiency_indexes],
        rcond=None,
    )[0]
    efficiency_residual = efficiency_score - efficiency_design @ efficiency_coefficients
    efficiency_residual_std = float(np.std(efficiency_residual))
    # Above one minute the empirical variance is essentially additive. Keep the
    # signed efficiency innovation white; persistence belongs to Q and activity.
    efficiency_factors = FactorMixture(
        timescales=np.asarray([1.0]),
        weights=np.asarray([0.0]),
        white_variance=1.0,
        target_lags=np.asarray([1]),
        target_covariance=np.asarray([0.0]),
        fitted_covariance=np.asarray([0.0]),
    )
    return FittedProcess(
        variance_floor=variance_floor,
        variance_quantile_knots=variance_quantile_knots,
        variance_log_quantiles=variance_log_quantiles,
        volatility_seasonal=volatility_seasonal,
        volatility_factors=volatility_factors,
        daily_variance=daily_variance,
        daily_return=daily_return,
        activity_coefficients=activity_coefficients,
        activity_residual_quantile_knots=activity_residual_knots,
        activity_residual_quantiles=activity_residual_quantiles,
        activity_factors=activity_factors,
        activity_count_probabilities=activity_count_probabilities,
        target_adjacent_activity_probability=source.adjacent_activity_probability,
        magnitude_mixture=magnitude_mixture,
        positive_probability=source.positive_probability,
        efficiency_quantile_knots=efficiency_knots,
        efficiency_quantiles=efficiency_quantiles,
        efficiency_volatility_coefficients=efficiency_coefficients,
        efficiency_residual_std=efficiency_residual_std,
        efficiency_factors=efficiency_factors,
        target_mean_realized_variance=float(np.mean(source.realized_variance)),
        target_minute_return_variance=float(np.var(source.minute_returns)),
        target_mean_activity_count=float(np.mean(source.counts)),
    )


def fit_conditional_magnitude_mixture(
    counts: np.ndarray,
    volatility_score: np.ndarray,
    log_concentration: np.ndarray,
    share_score_rho: float,
    micro_counts: np.ndarray,
    micro_threshold_bps: float,
) -> ConditionalMagnitudeMixture:
    valid = np.isfinite(log_concentration)
    if np.count_nonzero(valid) < 10_000:
        raise RuntimeError("too few valid minutes for the magnitude mixture")
    clipped = np.asarray(log_concentration[valid], dtype=np.float64)
    lower, upper = np.quantile(clipped, [0.001, 0.999])
    clipped = np.clip(clipped, lower, upper)
    activity_edges = np.quantile(counts[valid], [0.25, 0.5, 0.75])
    volatility_edges = np.quantile(volatility_score[valid], [0.25, 0.5, 0.75])
    state_ids = magnitude_state_ids(
        counts[valid],
        volatility_score[valid],
        activity_edges,
        volatility_edges,
    )
    state_count = 16
    weights = np.empty((state_count, MAGNITUDE_MIXTURE_COMPONENTS), dtype=np.float64)
    means = np.empty_like(weights)
    stds = np.empty_like(weights)
    global_fit = fit_gaussian_mixture_1d(clipped, MAGNITUDE_MIXTURE_COMPONENTS)
    for state in range(state_count):
        values = clipped[state_ids == state]
        fitted = (
            fit_gaussian_mixture_1d(values, MAGNITUDE_MIXTURE_COMPONENTS)
            if values.size >= 1_000
            else global_fit
        )
        weights[state], means[state], stds[state] = fitted
    all_states = magnitude_state_ids(
        counts,
        volatility_score,
        activity_edges,
        volatility_edges,
    )
    micro_probabilities = np.empty(state_count, dtype=np.float64)
    for state in range(state_count):
        selected_state = all_states == state
        active_total = float(np.sum(counts[selected_state], dtype=np.float64))
        micro_total = float(np.sum(micro_counts[selected_state], dtype=np.float64))
        micro_probabilities[state] = (micro_total + 0.5) / (active_total + 1.0)
    return ConditionalMagnitudeMixture(
        activity_edges=activity_edges,
        volatility_edges=volatility_edges,
        component_weights=weights,
        log_concentration_means=means,
        log_concentration_stds=stds,
        share_score_rho=float(np.clip(share_score_rho, -0.9, 0.9)),
        micro_probabilities=micro_probabilities,
        micro_threshold_bps=micro_threshold_bps,
    )


def fit_gaussian_mixture_1d(
    values: np.ndarray,
    components: int,
    iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < components:
        raise ValueError("Gaussian mixture needs a nonempty one-dimensional sample")
    fit_values = values[evenly_spaced_indexes(
        values.size,
        MAGNITUDE_MIXTURE_FIT_SAMPLE,
    )]
    means = np.quantile(fit_values, np.linspace(0.15, 0.85, components))
    global_std = max(float(np.std(fit_values)), 0.1)
    stds = np.full(components, global_std * 0.65)
    weights = np.full(components, 1.0 / components)
    for _ in range(iterations):
        standardized = (fit_values[:, None] - means[None, :]) / stds[None, :]
        log_responsibility = (
            np.log(np.maximum(weights, 1e-12))[None, :]
            - np.log(stds)[None, :]
            - 0.5 * standardized * standardized
        )
        maximum = np.max(log_responsibility, axis=1, keepdims=True)
        responsibility = np.exp(log_responsibility - maximum)
        responsibility /= np.sum(responsibility, axis=1, keepdims=True)
        mass = np.sum(responsibility, axis=0) + 1e-9
        weights = mass / np.sum(mass)
        means = np.sum(responsibility * fit_values[:, None], axis=0) / mass
        variance = np.sum(
            responsibility * (fit_values[:, None] - means[None, :]) ** 2,
            axis=0,
        ) / mass
        stds = np.sqrt(np.maximum(variance, 0.05 ** 2))
    order = np.argsort(means)
    return weights[order], means[order], stds[order]


def magnitude_state_ids(
    counts: np.ndarray,
    volatility_score: np.ndarray,
    activity_edges: np.ndarray,
    volatility_edges: np.ndarray,
) -> np.ndarray:
    activity_bin = np.searchsorted(activity_edges, counts, side="right")
    volatility_bin = np.searchsorted(volatility_edges, volatility_score, side="right")
    return activity_bin.astype(np.int64) * 4 + volatility_bin.astype(np.int64)


def sample_magnitude_concentrations(
    mixture: ConditionalMagnitudeMixture,
    counts: np.ndarray,
    volatility_score: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    states = magnitude_state_ids(
        counts,
        volatility_score,
        mixture.activity_edges,
        mixture.volatility_edges,
    )
    result = np.empty(counts.size, dtype=np.float64)
    for state in np.unique(states):
        locations = np.flatnonzero(states == state)
        weights = mixture.component_weights[int(state)]
        component = np.searchsorted(
            np.cumsum(weights),
            rng.random(locations.size),
            side="right",
        )
        component = np.minimum(component, weights.size - 1)
        log_concentration = (
            mixture.log_concentration_means[int(state), component]
            + mixture.log_concentration_stds[int(state), component]
            * rng.standard_normal(locations.size)
        )
        result[locations] = np.exp(np.clip(log_concentration, -9.0, 5.0))
    return result


def logistic_normal_energy(
    concentrations: np.ndarray,
    counts: np.ndarray,
    rho: float,
    dispersion_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    count_values = counts.astype(np.float64)
    expected_simpson = np.divide(
        concentrations + 1.0,
        count_values * concentrations + 1.0,
        out=np.ones_like(concentrations),
        where=count_values > 0,
    )
    sigma_squared = np.log(np.maximum(1.0, count_values * expected_simpson))
    sigma = dispersion_scale * np.sqrt(sigma_squared)
    scores = correlated_minute_scores(
        counts.size,
        MINUTE_SECONDS,
        rho,
        rng,
    )
    log_energy = sigma[:, None] * scores - 0.5 * sigma_squared[:, None]
    return np.exp(np.clip(log_energy, -30.0, 30.0))


def generate_daily_variance_budgets(
    fit: DailyVarianceFit,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    burn = 10 * 365
    generator = ArFactorGenerator(fit.factors, rng)
    score = generator.draw(count + burn)[burn:]
    log_variance = inverse_gaussianized_quantile_spline(
        score,
        fit.quantile_knots,
        fit.log_quantiles,
    )
    variance = np.maximum(0.0, np.exp(log_variance) - fit.floor)
    variance *= fit.target_mean / np.mean(variance)
    return variance


def generate_daily_return_targets(
    fit: DailyReturnFit,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    burn = 10 * 365
    generator = ArFactorGenerator(fit.factors, rng)
    score = generator.draw(count + burn)[burn:]
    return inverse_gaussianized_quantile_spline(
        score,
        fit.quantile_knots,
        fit.return_quantiles,
    )


def apply_aligned_daily_variance_budgets(
    minute_variance: np.ndarray,
    daily_budgets: np.ndarray,
) -> np.ndarray:
    result = np.asarray(minute_variance, dtype=np.float64).copy()
    offset = MINUTES_PER_DAY - 1
    available = (result.size - offset) // MINUTES_PER_DAY
    if daily_budgets.size != available:
        raise ValueError("daily budget count does not match complete aligned days")
    selected = result[offset:offset + available * MINUTES_PER_DAY].reshape(
        available,
        MINUTES_PER_DAY,
    )
    totals = np.sum(selected, axis=1)
    scale = np.divide(
        daily_budgets,
        totals,
        out=np.ones_like(daily_budgets),
        where=totals > 0,
    )
    selected *= scale[:, None]
    return result


def project_returns_to_daily_targets(
    minute_returns: np.ndarray,
    minute_variance: np.ndarray,
    activity_counts: np.ndarray,
    daily_targets: np.ndarray,
    *,
    aligned: bool,
    block_minutes: int = MINUTES_PER_DAY,
) -> tuple[np.ndarray, int]:
    """Minimally tilt minute returns so every selected block has its target sum."""
    result = np.asarray(minute_returns, dtype=np.float64).copy()
    q = np.asarray(minute_variance, dtype=np.float64)
    counts = np.asarray(activity_counts, dtype=np.float64)
    if block_minutes < 1:
        raise ValueError("projection block must contain at least one minute")
    offset = block_minutes - 1 if aligned else 0
    available = (result.size - offset) // block_minutes
    if daily_targets.size != available:
        raise ValueError("daily return target count does not match complete days")
    stop = offset + available * block_minutes
    selected = result[offset:stop].reshape(available, block_minutes)
    selected_q = q[offset:stop].reshape(available, block_minutes)
    selected_counts = counts[offset:stop].reshape(available, block_minutes)
    bounds = np.sqrt(np.maximum(0.0, selected_counts * selected_q)) * (1.0 - 1e-10)
    lower = -bounds
    upper = bounds
    fixed = selected_counts < 2
    lower[fixed] = selected[fixed]
    upper[fixed] = selected[fixed]
    adjustable_q = np.where(fixed, 0.0, selected_q)
    weights = np.divide(
        adjustable_q,
        np.sum(adjustable_q, axis=1)[:, None],
        out=np.zeros_like(adjustable_q),
        where=np.sum(adjustable_q, axis=1)[:, None] > 0,
    )
    delta = daily_targets - np.sum(selected, axis=1)
    candidate = selected + delta[:, None] * weights
    feasible = np.all((candidate >= lower) & (candidate <= upper), axis=1)
    selected[feasible] = candidate[feasible]

    fallback_count = 0
    for row in np.flatnonzero(~feasible):
        row_lower = lower[row]
        row_upper = upper[row]
        target = float(np.clip(
            daily_targets[row],
            np.sum(row_lower),
            np.sum(row_upper),
        ))
        row_weights = adjustable_q[row].copy()
        movable = row_weights > 0
        low = float(np.min(
            (row_lower[movable] - selected[row, movable]) / row_weights[movable]
        ))
        high = float(np.max(
            (row_upper[movable] - selected[row, movable]) / row_weights[movable]
        ))
        for _ in range(64):
            middle = 0.5 * (low + high)
            total = float(np.sum(np.clip(
                selected[row] + middle * row_weights,
                row_lower,
                row_upper,
            )))
            if total < target:
                low = middle
            else:
                high = middle
        selected[row] = np.clip(
            selected[row] + 0.5 * (low + high) * row_weights,
            row_lower,
            row_upper,
        )
        fallback_count += 1

    return result, fallback_count


def generate_projected_second_batch(
    fitted: FittedProcess,
    *,
    counts: np.ndarray,
    volatility_score: np.ndarray,
    target_minute_returns: np.ndarray,
    realized_variance: np.ndarray,
    activity_timing_rho: float,
    magnitude_dispersion_scale: float,
    micro_probability_scale: float,
    rng: np.random.Generator,
    sign_timing_rho: float = 0.0,
    magnitude_share_score_rho: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    timing_scores = correlated_minute_scores(
        counts.size,
        MINUTE_SECONDS,
        activity_timing_rho,
        rng,
    )
    active = activity_mask_from_scores(timing_scores, counts)
    micro, micro_values = sample_micro_returns(
        fitted.magnitude_mixture,
        active=active,
        counts=counts,
        volatility_score=volatility_score,
        target_minute_returns=target_minute_returns,
        realized_variance=realized_variance,
        probability_scale=micro_probability_scale,
        rng=rng,
    )
    macro_active = active & ~micro
    concentrations = sample_magnitude_concentrations(
        fitted.magnitude_mixture,
        counts,
        volatility_score,
        rng,
    )
    energy = logistic_normal_energy(
        concentrations,
        counts,
        (
            fitted.magnitude_mixture.share_score_rho
            if magnitude_share_score_rho is None
            else magnitude_share_score_rho
        ),
        magnitude_dispersion_scale,
        rng,
    )
    sign_scores = correlated_minute_scores(
        counts.size,
        MINUTE_SECONDS,
        sign_timing_rho,
        rng,
    )
    raw_signs = np.where(
        sign_scores < stats.norm.ppf(fitted.positive_probability),
        1.0,
        -1.0,
    )
    base = raw_signs * np.sqrt(energy) * macro_active
    macro_target_returns = target_minute_returns - np.sum(micro_values, axis=1)
    macro_target_variance = realized_variance - np.sum(
        micro_values * micro_values,
        axis=1,
    )
    returns = constrained_second_returns(
        base,
        macro_active,
        macro_target_returns,
        macro_target_variance,
    )
    returns += micro_values
    return active, returns


def sample_micro_returns(
    mixture: ConditionalMagnitudeMixture,
    *,
    active: np.ndarray,
    counts: np.ndarray,
    volatility_score: np.ndarray,
    target_minute_returns: np.ndarray,
    realized_variance: np.ndarray,
    probability_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    states = magnitude_state_ids(
        counts,
        volatility_score,
        mixture.activity_edges,
        mixture.volatility_edges,
    )
    probabilities = np.clip(
        mixture.micro_probabilities[states] * probability_scale,
        0.0,
        0.98,
    )
    proposed = rng.binomial(counts.astype(np.int64), probabilities)
    efficiency_squared = np.divide(
        target_minute_returns * target_minute_returns,
        realized_variance,
        out=np.zeros_like(target_minute_returns),
        where=realized_variance > 0,
    )
    required_macro = np.maximum(2, np.ceil(efficiency_squared + 1e-9).astype(np.int64))
    required_macro = np.minimum(required_macro, counts.astype(np.int64))
    micro_counts = np.minimum(
        proposed,
        np.maximum(0, counts.astype(np.int64) - required_macro),
    )
    scores = rng.random(active.shape)
    scores[~active] = -1.0
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(active.shape[1]), active.shape),
        axis=1,
    )
    micro = active & (ranks < micro_counts[:, None])
    magnitude = mixture.micro_threshold_bps * (
        0.02 + 0.96 * rng.random(active.shape)
    )
    signs = np.where(rng.random(active.shape) < 0.5, 1.0, -1.0)
    values = signs * magnitude * micro
    micro_energy = np.sum(values * values, axis=1)
    energy_cap = 0.01 * realized_variance
    scale = np.minimum(
        1.0,
        np.divide(
            np.sqrt(np.maximum(energy_cap, 0.0)),
            np.sqrt(micro_energy),
            out=np.ones_like(micro_energy),
            where=micro_energy > 0,
        ),
    )
    values *= scale[:, None]

    macro = active & ~micro
    macro_count = np.sum(macro, axis=1).astype(np.float64)
    remaining_return = target_minute_returns - np.sum(values, axis=1)
    remaining_variance = realized_variance - np.sum(values * values, axis=1)
    infeasible = remaining_return * remaining_return > (
        macro_count * remaining_variance * (1.0 + 1e-10)
    )
    for row in np.flatnonzero(infeasible):
        for position in np.flatnonzero(micro[row]):
            micro[row, position] = False
            values[row, position] = 0.0
            macro_count[row] += 1.0
            remaining_return[row] = target_minute_returns[row] - np.sum(values[row])
            remaining_variance[row] = realized_variance[row] - np.sum(values[row] ** 2)
            if remaining_return[row] ** 2 <= macro_count[row] * remaining_variance[row] * (1.0 + 1e-10):
                break
    return micro, values


def calibrate_magnitude_dispersion(
    fitted: FittedProcess,
    *,
    counts: np.ndarray,
    volatility_score: np.ndarray,
    target_minute_returns: np.ndarray,
    realized_variance: np.ndarray,
    activity_timing_rho: float,
    histogram: dict,
) -> tuple[float, float, list[dict[str, float]]]:
    indexes = evenly_spaced_indexes(counts.size, MAGNITUDE_CALIBRATION_MINUTES)
    observed = dense_histogram(histogram)
    edges = histogram_edges(histogram)
    tried: dict[tuple[float, float], float] = {}

    def evaluate(dispersion_scale: float, micro_scale: float) -> float:
        key = (
            round(float(dispersion_scale), 8),
            round(float(micro_scale), 8),
        )
        if key in tried:
            return tried[key]
        _, returns = generate_projected_second_batch(
            fitted,
            counts=counts[indexes],
            volatility_score=volatility_score[indexes],
            target_minute_returns=target_minute_returns[indexes],
            realized_variance=realized_variance[indexes],
            activity_timing_rho=activity_timing_rho,
            magnitude_dispersion_scale=dispersion_scale,
            micro_probability_scale=micro_scale,
            rng=np.random.default_rng(RNG_SEED + 70),
        )
        probabilities = bounded_histogram_counts(returns, edges).astype(np.float64)
        probabilities /= returns.size
        divergence = jensen_shannon_bits(observed, probabilities)
        tried[key] = divergence
        return divergence

    for dispersion_scale in (0.8, 1.15, 1.5, 2.0):
        for micro_scale in (0.5, 0.65, 0.8, 0.95, 1.1):
            evaluate(dispersion_scale, micro_scale)
    best = min(tried, key=tried.get)
    for dispersion_scale in best[0] * np.asarray([0.82, 0.92, 1.0, 1.09, 1.2]):
        for micro_scale in best[1] * np.asarray([0.82, 0.92, 1.0, 1.09, 1.2]):
            evaluate(float(dispersion_scale), float(micro_scale))
    best = min(tried, key=tried.get)
    diagnostics = [
        {
            "dispersionScale": float(scales[0]),
            "microProbabilityScale": float(scales[1]),
            "jsDivergenceBits": float(divergence),
        }
        for scales, divergence in sorted(tried.items())
    ]
    return float(best[0]), float(best[1]), diagnostics


def bounded_histogram_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    indexes = np.searchsorted(edges, flat, side="right") - 1
    indexes = np.clip(indexes, 0, edges.size - 2)
    return np.bincount(indexes, minlength=edges.size - 1)


def simulate_process(
    fitted: FittedProcess,
    *,
    minute_count: int,
    one_second_histogram: dict,
    one_second_sigma: float,
) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + 1)
    burn = 30 * MINUTES_PER_DAY
    total = minute_count + burn
    volatility_generator = ArFactorGenerator(fitted.volatility_factors, rng)
    activity_generator = ArFactorGenerator(fitted.activity_factors, rng)
    volatility_latent = volatility_generator.draw(total)[burn:]
    minute_indexes = np.arange(minute_count, dtype=np.int64)
    volatility_score = (
        seasonal_values(minute_indexes, fitted.volatility_seasonal)
        + volatility_latent
    )
    log_q = inverse_gaussianized_quantile_spline(
        volatility_score,
        fitted.variance_quantile_knots,
        fitted.variance_log_quantiles,
    )
    q = np.maximum(0.0, np.exp(log_q) - fitted.variance_floor)
    q *= fitted.target_mean_realized_variance / np.mean(q)

    activity_latent = activity_generator.draw(total)[burn:]
    coefficients = fitted.activity_coefficients
    activity_residual = inverse_gaussianized_quantile_spline(
        activity_latent,
        fitted.activity_residual_quantile_knots,
        fitted.activity_residual_quantiles,
    )
    activity_score = (
        coefficients[0]
        + coefficients[1] * volatility_score
        + coefficients[2] * (volatility_score * volatility_score - 1.0)
        + activity_residual
    )
    activity_score_offset = calibrate_discrete_location(
        activity_score,
        fitted.activity_count_probabilities,
        fitted.target_mean_activity_count,
    )
    counts = inverse_discrete_gaussian_copula(
        activity_score + activity_score_offset,
        fitted.activity_count_probabilities,
    ).astype(np.uint8)
    q[counts == 0] = 0.0
    complete_daily_count = (minute_count - (MINUTES_PER_DAY - 1)) // MINUTES_PER_DAY
    daily_budgets = generate_daily_variance_budgets(
        fitted.daily_variance,
        complete_daily_count,
        rng,
    )
    q = apply_aligned_daily_variance_budgets(q, daily_budgets)
    activity_timing_rho = calibrate_activity_rho(
        counts,
        fitted.target_adjacent_activity_probability,
        np.random.default_rng(RNG_SEED + 50),
    )

    efficiency_generator = ArFactorGenerator(fitted.efficiency_factors, rng)
    efficiency_latent = efficiency_generator.draw(total)[burn:]
    efficiency_coefficients = fitted.efficiency_volatility_coefficients
    efficiency_score = (
        efficiency_coefficients[0]
        + efficiency_coefficients[1] * volatility_score
        + efficiency_coefficients[2] * (volatility_score * volatility_score - 1.0)
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
        efficiency,
        -maximum_efficiency * (1.0 - 1e-10),
        maximum_efficiency * (1.0 - 1e-10),
    )
    single = counts == 1
    efficiency[single] = np.where(efficiency[single] >= 0.0, 1.0, -1.0)
    current_minute_variance = float(np.var(efficiency * np.sqrt(q)))
    efficiency_scale = math.sqrt(
        fitted.target_minute_return_variance / current_minute_variance
    )
    efficiency *= efficiency_scale
    efficiency = np.clip(
        efficiency,
        -maximum_efficiency * (1.0 - 1e-10),
        maximum_efficiency * (1.0 - 1e-10),
    )
    efficiency[single] = np.where(efficiency[single] >= 0.0, 1.0, -1.0)
    target_minute_returns = efficiency * np.sqrt(q)
    daily_return_targets = generate_daily_return_targets(
        fitted.daily_return,
        complete_daily_count,
        rng,
    )
    target_minute_returns, daily_projection_fallbacks = (
        project_returns_to_daily_targets(
            target_minute_returns,
            q,
            counts,
            daily_return_targets,
            aligned=True,
        )
    )

    (
        magnitude_dispersion_scale,
        micro_probability_scale,
        magnitude_calibration,
    ) = calibrate_magnitude_dispersion(
        fitted,
        counts=counts,
        volatility_score=volatility_score,
        target_minute_returns=target_minute_returns,
        realized_variance=q,
        activity_timing_rho=activity_timing_rho,
        histogram=one_second_histogram,
    )

    edges = histogram_edges(one_second_histogram)
    histogram_counts = np.zeros(edges.size - 1, dtype=np.int64)
    minute_returns = np.empty(minute_count, dtype=np.float64)
    adjacent_active = 0
    acf_path: list[np.ndarray] = []

    for batch_start in range(0, minute_count, SIMULATION_BATCH_MINUTES):
        batch_end = min(minute_count, batch_start + SIMULATION_BATCH_MINUTES)
        batch_count = batch_end - batch_start
        batch_counts = counts[batch_start:batch_end]
        active, returns = generate_projected_second_batch(
            fitted,
            counts=batch_counts,
            volatility_score=volatility_score[batch_start:batch_end],
            target_minute_returns=target_minute_returns[batch_start:batch_end],
            realized_variance=q[batch_start:batch_end],
            activity_timing_rho=activity_timing_rho,
            magnitude_dispersion_scale=magnitude_dispersion_scale,
            micro_probability_scale=micro_probability_scale,
            rng=rng,
        )
        adjacent_active += int(np.count_nonzero(active[:, 1:] & active[:, :-1]))
        minute_returns[batch_start:batch_end] = np.sum(returns, axis=1)
        histogram_counts += bounded_histogram_counts(returns, edges)
        remaining = SIMULATED_ACF_MINUTES - sum(
            path.size // MINUTE_SECONDS for path in acf_path
        )
        if remaining > 0:
            take = min(remaining, batch_count)
            acf_path.append(returns[:take].reshape(-1))

    probabilities = histogram_counts.astype(np.float64) / (minute_count * MINUTE_SECONDS)
    acf = AcfAccumulator.create()
    acf.add_returns(np.concatenate(acf_path))
    return {
        "counts": counts,
        "realizedVariance": q,
        "dailyVarianceBudgets": daily_budgets,
        "dailyReturnTargets": daily_return_targets,
        "dailyProjectionFallbacks": daily_projection_fallbacks,
        "minuteReturns": minute_returns,
        "oneSecondProbabilities": probabilities,
        "oneSecondSigmaBps": float(math.sqrt(np.sum(
            probabilities * histogram_centers(one_second_histogram) ** 2
        ))),
        "oneSecondCentralMass": probability_in_region(
            probabilities,
            histogram_centers(one_second_histogram),
            lambda centers: np.abs(centers) < 0.25 * one_second_sigma,
        ),
        "oneSecondTail3": probability_in_region(
            probabilities,
            histogram_centers(one_second_histogram),
            lambda centers: np.abs(centers) >= 3.0 * one_second_sigma,
        ),
        "oneSecondTail5": probability_in_region(
            probabilities,
            histogram_centers(one_second_histogram),
            lambda centers: np.abs(centers) >= 5.0 * one_second_sigma,
        ),
        "adjacentActivityProbability": adjacent_active
        / (minute_count * (MINUTE_SECONDS - 1)),
        "activityScoreOffset": activity_score_offset,
        "activityTimingGaussianRho": activity_timing_rho,
        "magnitudeDispersionScale": magnitude_dispersion_scale,
        "microProbabilityScale": micro_probability_scale,
        "magnitudeCalibration": magnitude_calibration,
        "efficiencyScale": efficiency_scale,
        "acf": acf.finish(),
    }


def simulate_long_run_daily_returns(
    fitted: FittedProcess,
    *,
    days: int,
    activity_score_offset: float,
    efficiency_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate long-run 1d returns by summing the modeled minute-return layer.

    The constrained second-return projection is omitted here because it preserves
    every generated minute return exactly. Consequently this is mathematically the
    same 1s -> 1d aggregation law while avoiding a 4.32-billion-element second path.
    """
    if days < 1:
        raise ValueError("long-run daily simulation needs at least one day")

    daily_budgets = generate_daily_variance_budgets(
        fitted.daily_variance,
        days,
        rng,
    )
    daily_return_targets = generate_daily_return_targets(
        fitted.daily_return,
        days,
        rng,
    )
    volatility_generator = ArFactorGenerator(fitted.volatility_factors, rng)
    activity_generator = ArFactorGenerator(fitted.activity_factors, rng)
    efficiency_generator = ArFactorGenerator(fitted.efficiency_factors, rng)
    burn = 30 * MINUTES_PER_DAY
    volatility_generator.draw(burn)
    activity_generator.draw(burn)
    efficiency_generator.draw(burn)

    result = np.empty(days, dtype=np.float64)
    coefficients = fitted.activity_coefficients
    efficiency_coefficients = fitted.efficiency_volatility_coefficients
    minute_cursor = 0
    for day_start in range(0, days, LONG_RUN_DAILY_BATCH_DAYS):
        day_end = min(days, day_start + LONG_RUN_DAILY_BATCH_DAYS)
        batch_days = day_end - day_start
        minute_count = batch_days * MINUTES_PER_DAY
        minute_indexes = np.arange(
            minute_cursor,
            minute_cursor + minute_count,
            dtype=np.int64,
        )
        minute_cursor += minute_count

        volatility_score = (
            seasonal_values(minute_indexes, fitted.volatility_seasonal)
            + volatility_generator.draw(minute_count)
        )
        log_q = inverse_gaussianized_quantile_spline(
            volatility_score,
            fitted.variance_quantile_knots,
            fitted.variance_log_quantiles,
        )
        q = np.maximum(0.0, np.exp(log_q) - fitted.variance_floor)

        activity_residual = inverse_gaussianized_quantile_spline(
            activity_generator.draw(minute_count),
            fitted.activity_residual_quantile_knots,
            fitted.activity_residual_quantiles,
        )
        activity_score = (
            coefficients[0]
            + coefficients[1] * volatility_score
            + coefficients[2] * (volatility_score * volatility_score - 1.0)
            + activity_residual
            + activity_score_offset
        )
        counts = inverse_discrete_gaussian_copula(
            activity_score,
            fitted.activity_count_probabilities,
        ).astype(np.uint8)
        q[counts == 0] = 0.0

        q_by_day = q.reshape(batch_days, MINUTES_PER_DAY)
        q_totals = np.sum(q_by_day, axis=1)
        q_by_day *= np.divide(
            daily_budgets[day_start:day_end],
            q_totals,
            out=np.ones(batch_days, dtype=np.float64),
            where=q_totals > 0,
        )[:, None]

        efficiency_score = (
            efficiency_coefficients[0]
            + efficiency_coefficients[1] * volatility_score
            + efficiency_coefficients[2]
            * (volatility_score * volatility_score - 1.0)
            + fitted.efficiency_residual_std
            * efficiency_generator.draw(minute_count)
        )
        efficiency = inverse_gaussianized_quantile_spline(
            efficiency_score,
            fitted.efficiency_quantile_knots,
            fitted.efficiency_quantiles,
        )
        efficiency[counts == 0] = 0.0
        maximum_efficiency = np.sqrt(counts.astype(np.float64))
        efficiency = np.clip(
            efficiency * efficiency_scale,
            -maximum_efficiency * (1.0 - 1e-10),
            maximum_efficiency * (1.0 - 1e-10),
        )
        single = counts == 1
        efficiency[single] = np.where(efficiency[single] >= 0.0, 1.0, -1.0)
        minute_returns = efficiency * np.sqrt(q)
        minute_returns, _ = project_returns_to_daily_targets(
            minute_returns,
            q,
            counts,
            daily_return_targets[day_start:day_end],
            aligned=False,
        )
        result[day_start:day_end] = np.sum(
            minute_returns.reshape(batch_days, MINUTES_PER_DAY),
            axis=1,
        )

    return result


def validate_scales(generated: dict, *, histograms: dict, analysis: dict) -> dict:
    result: dict[str, list[dict]] = {}
    one_second_histogram = full_histogram(histograms, "1s")
    observed_probabilities = dense_histogram(one_second_histogram)
    observed_sigma = float(full_scale(analysis, "1s")["standardDeviationBps"])
    model_probabilities = generated["oneSecondProbabilities"]
    centers = histogram_centers(one_second_histogram)
    observed_center = float(np.sum(
        observed_probabilities[np.abs(centers) < 0.25 * observed_sigma]
    ))
    observed_tail3 = float(np.sum(
        observed_probabilities[np.abs(centers) >= 3.0 * observed_sigma]
    ))
    observed_tail5 = float(np.sum(
        observed_probabilities[np.abs(centers) >= 5.0 * observed_sigma]
    ))
    result["1s"] = [{
        "id": "fittedStatisticalProcess",
        "label": "Component-free fitted one-second process",
        "observations": int(generated["counts"].size * MINUTE_SECONDS),
        "sigmaBps": generated["oneSecondSigmaBps"],
        "varianceRatioObservedOverModel": (
            observed_sigma / generated["oneSecondSigmaBps"]
        ) ** 2,
        "jsDivergenceBits": jensen_shannon_bits(
            observed_probabilities, model_probabilities
        ),
        "centralMass": ratio_measure(
            observed_center, generated["oneSecondCentralMass"]
        ),
        "threeSigmaTail": ratio_measure(
            observed_tail3, generated["oneSecondTail3"]
        ),
        "fiveSigmaTail": ratio_measure(
            observed_tail5, generated["oneSecondTail5"]
        ),
    }]

    minute_returns = generated["minuteReturns"]
    for scale_id, factor in (
        ("1m", 1), ("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1_440)
    ):
        values = (
            minute_returns
            if factor == 1
            else aggregate_aligned_minutes(minute_returns, factor)
        )
        histogram = full_histogram(histograms, scale_id)
        result[scale_id] = [summarize_raw_model(
            "fittedStatisticalProcess",
            "Component-free fitted one-second process",
            values,
            dense_histogram(histogram),
            histogram,
            float(full_scale(analysis, scale_id)["standardDeviationBps"]),
        )]
    return result


def fit_seasonal(values: np.ndarray, daily_harmonics: int = 4, weekly_harmonics: int = 3) -> SeasonalFit:
    indexes = evenly_spaced_indexes(values.size, DEPENDENCE_FIT_SAMPLE)
    design = seasonal_design(indexes, daily_harmonics, weekly_harmonics)
    coefficients = np.linalg.lstsq(design, values[indexes], rcond=None)[0]
    return SeasonalFit(coefficients, daily_harmonics, weekly_harmonics)


def seasonal_design(indexes: np.ndarray, daily_harmonics: int, weekly_harmonics: int) -> np.ndarray:
    columns = [np.ones(indexes.size, dtype=np.float64)]
    for period, harmonics in (
        (MINUTES_PER_DAY, daily_harmonics),
        (MINUTES_PER_WEEK, weekly_harmonics),
    ):
        phase = 2.0 * math.pi * indexes / period
        for harmonic in range(1, harmonics + 1):
            columns.append(np.sin(harmonic * phase))
            columns.append(np.cos(harmonic * phase))
    return np.column_stack(columns)


def seasonal_values(indexes: np.ndarray, fit: SeasonalFit) -> np.ndarray:
    return seasonal_design(
        indexes,
        fit.daily_harmonics,
        fit.weekly_harmonics,
    ) @ fit.coefficients


def fit_factor_mixture(
    values: np.ndarray,
    *,
    timescales: np.ndarray,
    lags: np.ndarray,
) -> FactorMixture:
    centered = values - np.mean(values)
    target_variance = float(np.mean(centered * centered))
    usable_lags = lags[lags < values.size]
    if usable_lags.size == 0:
        return FactorMixture(
            timescales=timescales.copy(),
            weights=np.zeros(timescales.size, dtype=np.float64),
            white_variance=target_variance,
            target_lags=usable_lags.copy(),
            target_covariance=np.empty(0, dtype=np.float64),
            fitted_covariance=np.empty(0, dtype=np.float64),
        )
    covariance = np.asarray([
        float(np.mean(centered[lag:] * centered[:-lag]))
        for lag in usable_lags
    ])
    return fit_covariance_targets(
        target_variance=target_variance,
        target_lags=usable_lags,
        target_covariance=covariance,
        timescales=timescales,
    )


def fit_covariance_targets(
    *,
    target_variance: float,
    target_lags: np.ndarray,
    target_covariance: np.ndarray,
    timescales: np.ndarray,
) -> FactorMixture:
    design = np.exp(-target_lags[:, None] / timescales[None, :])
    weights, _ = optimize.nnls(design, np.maximum(target_covariance, 0.0))
    maximum_factor_variance = max(0.0, target_variance * 0.999)
    if np.sum(weights) > maximum_factor_variance:
        weights *= maximum_factor_variance / np.sum(weights)
    fitted = design @ weights
    return FactorMixture(
        timescales=timescales.copy(),
        weights=weights,
        white_variance=max(0.0, target_variance - float(np.sum(weights))),
        target_lags=target_lags.copy(),
        target_covariance=target_covariance.copy(),
        fitted_covariance=fitted,
    )


def gaussianize_quantile_spline(
    values: np.ndarray,
    probability_knots: np.ndarray,
    value_quantiles: np.ndarray,
) -> np.ndarray:
    probabilities = np.interp(
        values,
        value_quantiles,
        probability_knots,
        left=probability_knots[0],
        right=probability_knots[-1],
    )
    return stats.norm.ppf(probabilities)


def inverse_gaussianized_quantile_spline(
    values: np.ndarray,
    probability_knots: np.ndarray,
    value_quantiles: np.ndarray,
) -> np.ndarray:
    probabilities = stats.norm.cdf(np.clip(values, -7.0, 7.0))
    return np.interp(
        probabilities,
        probability_knots,
        value_quantiles,
        left=value_quantiles[0],
        right=value_quantiles[-1],
    )


def fit_quantile_spline(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.sort(values[evenly_spaced_indexes(values.size, MARGINAL_FIT_SAMPLE)])
    probabilities = (np.arange(selected.size, dtype=np.float64) + 0.5) / selected.size
    knots = np.concatenate((
        np.geomspace(0.5 / selected.size, 0.01, 128, endpoint=False),
        np.linspace(0.01, 0.99, 1_024, endpoint=False),
        1.0 - np.geomspace(0.5 / selected.size, 0.01, 128)[::-1],
    ))
    knots = np.unique(np.clip(
        knots,
        0.5 / selected.size,
        1.0 - 0.5 / selected.size,
    ))
    return knots, np.interp(knots, probabilities, selected)


def discrete_midpoint_gaussian_scores(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or np.any(probabilities < 0) or np.sum(probabilities) <= 0:
        raise ValueError("discrete probabilities must be a nonnegative vector")
    normalized = probabilities / np.sum(probabilities)
    lower = np.cumsum(normalized) - normalized
    midpoint = lower + 0.5 * normalized
    return stats.norm.ppf(np.clip(midpoint, 1e-10, 1.0 - 1e-10))


def inverse_discrete_gaussian_copula(
    scores: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    cumulative = np.cumsum(probabilities / np.sum(probabilities))
    uniforms = stats.norm.cdf(scores)
    return np.searchsorted(cumulative, uniforms, side="right")


def calibrate_discrete_location(
    scores: np.ndarray,
    probabilities: np.ndarray,
    target_mean: float,
) -> float:
    def generated_mean(offset: float) -> float:
        return float(np.mean(inverse_discrete_gaussian_copula(
            scores + offset,
            probabilities,
        )))

    left, right = -4.0, 4.0
    if not generated_mean(left) <= target_mean <= generated_mean(right):
        raise ValueError("discrete location target is outside calibration bounds")
    for _ in range(30):
        middle = (left + right) / 2.0
        if generated_mean(middle) < target_mean:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def constrained_second_returns(
    base: np.ndarray,
    active: np.ndarray,
    target_returns: np.ndarray,
    target_variance: np.ndarray,
) -> np.ndarray:
    """Project random active vectors onto exact per-row sum and squared norm."""
    if base.shape != active.shape or base.ndim != 2:
        raise ValueError("base and activity arrays must have the same 2D shape")
    if target_returns.shape != (base.shape[0],) or target_variance.shape != (base.shape[0],):
        raise ValueError("minute targets do not match the row count")
    counts = np.sum(active, axis=1).astype(np.float64)
    means = np.divide(
        target_returns,
        counts,
        out=np.zeros_like(target_returns, dtype=np.float64),
        where=counts > 0,
    )
    base_means = np.divide(
        np.sum(base, axis=1),
        counts,
        out=np.zeros_like(target_returns, dtype=np.float64),
        where=counts > 0,
    )
    centered = (base - base_means[:, None]) * active
    centered_norm = np.sqrt(np.sum(centered * centered, axis=1))
    degenerate = (counts > 1) & (centered_norm < 1e-12)
    if np.any(degenerate):
        for row in np.flatnonzero(degenerate):
            positions = np.flatnonzero(active[row])
            centered[row, positions[0]] = 1.0
            centered[row, positions[1]] = -1.0
        centered_norm = np.sqrt(np.sum(centered * centered, axis=1))
    residual_energy = np.maximum(
        0.0,
        target_variance - np.divide(
            target_returns * target_returns,
            counts,
            out=np.zeros_like(target_returns, dtype=np.float64),
            where=counts > 0,
        ),
    )
    residual_scale = np.divide(
        np.sqrt(residual_energy),
        centered_norm,
        out=np.zeros_like(target_returns, dtype=np.float64),
        where=centered_norm > 0,
    )
    return active * (means[:, None] + residual_scale[:, None] * centered)


def calibrate_activity_rho(
    counts: np.ndarray,
    target_adjacent_probability: float,
    rng: np.random.Generator,
) -> float:
    selected = counts[evenly_spaced_indexes(
        counts.size,
        ACTIVITY_CALIBRATION_MINUTES,
    )]
    innovations = rng.standard_normal((selected.size, MINUTE_SECONDS))

    def adjacent_probability(rho: float) -> float:
        scores = np.empty_like(innovations)
        scores[:, 0] = innovations[:, 0]
        scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        for position in range(1, MINUTE_SECONDS):
            scores[:, position] = rho * scores[:, position - 1] + scale * innovations[:, position]
        active = activity_mask_from_scores(scores, selected)
        return float(np.mean(active[:, 1:] & active[:, :-1]))

    low = adjacent_probability(0.0)
    high = adjacent_probability(0.995)
    if target_adjacent_probability <= low:
        return 0.0
    if target_adjacent_probability >= high:
        return 0.995
    left, right = 0.0, 0.995
    for _ in range(18):
        middle = (left + right) / 2.0
        if adjacent_probability(middle) < target_adjacent_probability:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def correlated_minute_scores(
    rows: int,
    columns: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    innovations = rng.standard_normal((rows, columns))
    scores = np.empty_like(innovations)
    scores[:, 0] = innovations[:, 0]
    scale = math.sqrt(max(0.0, 1.0 - rho * rho))
    for position in range(1, columns):
        scores[:, position] = rho * scores[:, position - 1] + scale * innovations[:, position]
    return scores


def activity_mask_from_scores(scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or counts.shape != (scores.shape[0],):
        raise ValueError("activity score/count dimensions do not match")
    if np.any(counts < 0) or np.any(counts > scores.shape[1]):
        raise ValueError("activity counts are outside the score width")
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(scores.shape[1]), scores.shape),
        axis=1,
    )
    return ranks < counts[:, None]


def dirichlet_alpha_from_simpson(counts: np.ndarray, simpson: np.ndarray) -> np.ndarray:
    denominator = simpson * counts - 1.0
    return np.divide(
        1.0 - simpson,
        denominator,
        out=np.full_like(simpson, np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )


def selected_correlations(values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for lag in (1, 5, 15, 60, 240, 1_440, 10_080, 43_200):
        if lag >= values.size:
            continue
        result[str(lag)] = float(np.corrcoef(values[lag:], values[:-lag])[0, 1])
    return result


def selected_daily_correlations(values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for lag in (1, 2, 5, 7, 14, 30, 60, 90, 180, 365):
        if lag >= values.size:
            continue
        result[str(lag)] = float(np.corrcoef(values[lag:], values[:-lag])[0, 1])
    return result


def js_sampling_floor(
    probabilities: np.ndarray,
    observations: int,
    repetitions: int = 5_000,
) -> dict[str, object]:
    normalized = probabilities / np.sum(probabilities)
    rng = np.random.default_rng(RNG_SEED + 90)
    one_sample = np.empty(repetitions, dtype=np.float64)
    two_sample = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        left = rng.multinomial(observations, normalized) / observations
        right = rng.multinomial(observations, normalized) / observations
        one_sample[index] = jensen_shannon_bits(normalized, left)
        two_sample[index] = jensen_shannon_bits(left, right)
    levels = (0.025, 0.5, 0.975)
    return {
        "observations": observations,
        "repetitions": repetitions,
        "oneSyntheticSampleVsPopulation": {
            str(level): float(np.quantile(one_sample, level)) for level in levels
        },
        "twoIndependentSamples": {
            str(level): float(np.quantile(two_sample, level)) for level in levels
        },
    }


def evenly_spaced_indexes(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def quantiles(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(values, [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
    return {
        name: float(value)
        for name, value in zip(("p001", "p01", "p10", "p50", "p90", "p99", "p999"), result)
    }


def probability_in_region(
    probabilities: np.ndarray,
    centers: np.ndarray,
    selector,
) -> float:
    return float(np.sum(probabilities[selector(centers)]))


def ratio_measure(observed: float, model: float) -> dict[str, float | None]:
    return {
        "observed": observed,
        "model": model,
        "ratioObservedOverModel": observed / model if model > 0 else None,
    }


def serialize_seasonal(fit: SeasonalFit) -> dict:
    return {
        "dailyHarmonics": fit.daily_harmonics,
        "weeklyHarmonics": fit.weekly_harmonics,
        "coefficients": fit.coefficients.tolist(),
    }


def serialize_mixture(fit: FactorMixture) -> dict:
    return {
        "timescales": fit.timescales.tolist(),
        "weights": fit.weights.tolist(),
        "whiteVariance": fit.white_variance,
        "targetLags": fit.target_lags.tolist(),
        "targetCovariance": fit.target_covariance.tolist(),
        "fittedCovariance": fit.fitted_covariance.tolist(),
    }


def serialize_daily_variance(fit: DailyVarianceFit) -> dict:
    return {
        "floorBpsSquared": fit.floor,
        "quantileSpline": {
            "probabilities": fit.quantile_knots.tolist(),
            "logVarianceQuantiles": fit.log_quantiles.tolist(),
        },
        "factors": serialize_mixture(fit.factors),
        "targetMeanBpsSquared": fit.target_mean,
    }


def serialize_daily_return(fit: DailyReturnFit) -> dict:
    return {
        "quantileSpline": {
            "probabilities": fit.quantile_knots.tolist(),
            "returnQuantilesBps": fit.return_quantiles.tolist(),
        },
        "factors": serialize_mixture(fit.factors),
    }


def serialize_magnitude_mixture(fit: ConditionalMagnitudeMixture) -> dict:
    return {
        "activityBinUpperEdges": fit.activity_edges.tolist(),
        "volatilityBinUpperEdges": fit.volatility_edges.tolist(),
        "componentWeights": fit.component_weights.tolist(),
        "logConcentrationMeans": fit.log_concentration_means.tolist(),
        "logConcentrationStds": fit.log_concentration_stds.tolist(),
        "shareScoreRho": fit.share_score_rho,
        "microProbabilities": fit.micro_probabilities.tolist(),
        "microThresholdBps": fit.micro_threshold_bps,
    }


def serialize_fit(fit: FittedProcess) -> dict:
    return {
        "varianceFloorBpsSquared": fit.variance_floor,
        "varianceQuantileSpline": {
            "probabilities": fit.variance_quantile_knots.tolist(),
            "logVarianceQuantiles": fit.variance_log_quantiles.tolist(),
        },
        "volatilitySeasonal": serialize_seasonal(fit.volatility_seasonal),
        "volatilityFactors": serialize_mixture(fit.volatility_factors),
        "dailyVarianceBudget": serialize_daily_variance(fit.daily_variance),
        "dailyReturnTarget": serialize_daily_return(fit.daily_return),
        "activityVolatilityCoefficients": fit.activity_coefficients.tolist(),
        "activityResidualQuantileSpline": {
            "probabilities": fit.activity_residual_quantile_knots.tolist(),
            "quantiles": fit.activity_residual_quantiles.tolist(),
        },
        "activityFactors": serialize_mixture(fit.activity_factors),
        "activityCountProbabilities": fit.activity_count_probabilities.tolist(),
        "targetAdjacentActivityProbability": fit.target_adjacent_activity_probability,
        "conditionalMagnitudeMixture": serialize_magnitude_mixture(
            fit.magnitude_mixture
        ),
        "positiveSignProbability": fit.positive_probability,
        "efficiencyQuantileSpline": {
            "probabilities": fit.efficiency_quantile_knots.tolist(),
            "quantiles": fit.efficiency_quantiles.tolist(),
        },
        "efficiencyVolatilityCoefficients": fit.efficiency_volatility_coefficients.tolist(),
        "efficiencyResidualStd": fit.efficiency_residual_std,
        "efficiencyFactors": serialize_mixture(fit.efficiency_factors),
        "targetMeanRealizedVariance": fit.target_mean_realized_variance,
        "targetMinuteReturnVariance": fit.target_minute_return_variance,
        "targetMeanActivityCount": fit.target_mean_activity_count,
    }


def model_description(fit: FittedProcess) -> dict:
    return {
        "class": "Component-free fitted hierarchical Gaussian-copula marked activity process",
        "volatility": (
            "Minute realized variance has a bounded fitted quantile-spline marginal. Its Gaussian copula "
            "score is the sum of daily/weekly Fourier terms, fitted stationary AR(1) factors, "
            "and white innovation. No historical variance path is selected during generation."
        ),
        "dailyVarianceBudget": (
            "A separate bounded marginal and long-memory AR-factor process generates each "
            "complete day's integrated variance. Preliminary minute variances are normalized "
            "within the day to sum exactly to this generated budget."
        ),
        "dailyReturnTarget": (
            "A bounded fitted daily-return marginal with a causal AR-factor Gaussian copula "
            "generates a fresh signed target for each day. Minute returns receive a "
            "variance-weighted bounded tilt so their direct sum equals that target."
        ),
        "activityCount": (
            "A fitted 61-category count marginal is coupled to generated volatility and a "
            "separate fitted AR-factor activity score through a Gaussian copula."
        ),
        "activityTiming": (
            "Within each minute a fitted Gaussian AR(1) score is generated and its highest N "
            "positions are active, preserving the generated count while controlling adjacency."
        ),
        "magnitude": (
            "A state-conditional three-component mixture generates a fresh per-minute "
            "dispersion parameter. Correlated logistic-normal energy scores generate the "
            "active second magnitudes, with a separate fitted nonzero micro-return layer; "
            "their global dispersion is calibrated against the post-projection one-second histogram."
        ),
        "signedMinuteInnovation": (
            "The generated minute return is sqrt(Q) times a fitted signed-efficiency quantile "
            "process, conditionally coupled to volatility. A random heavy-tailed vector is then "
            "projected so its active second returns sum exactly to R and their squares to Q."
        ),
        "aggregation": "Every slower return is a direct sum of the same generated one-second path.",
        "historicalResampling": "None during generation; only fitted scalar/vector parameters are used.",
        "seed": RNG_SEED,
    }


def extract_bootstrap_benchmark(report: dict) -> dict:
    result = {}
    for scale_id, rows in report["validation"]["modelsByScale"].items():
        selected = next(
            (row for row in rows if row["id"] == "fullNonJumpDependence"),
            None,
        )
        if selected is not None:
            result[scale_id] = selected
    return result


def compact_summary(report: dict) -> dict:
    scales = {}
    for scale_id, rows in report["validation"]["modelsByScale"].items():
        row = rows[0]
        scales[scale_id] = {
            "varianceRatio": row["varianceRatioObservedOverModel"],
            "centerRatio": row["centralMass"]["ratioObservedOverModel"],
            "tail3Ratio": row["threeSigmaTail"]["ratioObservedOverModel"],
            "tail5Ratio": row["fiveSigmaTail"]["ratioObservedOverModel"],
            "jsBits": row["jsDivergenceBits"],
        }
    return {
        "historicalComponentsSampled": report["scope"]["historicalComponentsSampledDuringGeneration"],
        "fitDiagnostics": report["fitDiagnostics"],
        "validationByScale": scales,
        "longRunDaily": {
            "varianceRatio": report["validation"]["longRunDaily"]["varianceRatioObservedOverModel"],
            "centerRatio": report["validation"]["longRunDaily"]["centralMass"]["ratioObservedOverModel"],
            "tail3Ratio": report["validation"]["longRunDaily"]["threeSigmaTail"]["ratioObservedOverModel"],
            "tail5Ratio": report["validation"]["longRunDaily"]["fiveSigmaTail"]["ratioObservedOverModel"],
            "jsBits": report["validation"]["longRunDaily"]["jsDivergenceBits"],
        },
        "acceptance": report["validation"]["acceptance"],
        "modelVarianceRatio60": report["validation"]["modelAcf"]["varianceRatioFromReturnAcf"],
    }


def read_json(file: Path) -> dict:
    return json.loads(file.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="data/benchmarks/log-return-distributions.json")
    parser.add_argument("--histograms", default="data/benchmarks/log-return-histograms.json")
    parser.add_argument(
        "--bootstrap",
        default="data/benchmarks/one-second-dependence-model.json",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/parametric-one-second-process.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
