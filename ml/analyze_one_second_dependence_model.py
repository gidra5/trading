"""Fit and validate a non-jump 1s BTCUSDT dependence model.

The model separates activity, a persistent activity/volatility state, empirical
nonzero magnitude innovations, and within-minute signed dependence.  Extreme
returns remain in the empirical innovation and volatility-state tails; there is
no separate jump process.
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
from scipy.fft import irfft, next_fast_len, rfft

from trading_storage import read_candle_column


DAY_SECONDS = 86_400
MINUTE_SECONDS = 60
MAX_LAG_SECONDS = 900
RNG_SEED = 0x5EED_1A2B
SIMULATED_ACF_MINUTES = 100_000
SIMULATION_BATCH_MINUTES = 20_000


@dataclass
class AcfAccumulator:
    cross: dict[str, np.ndarray]
    pairs: np.ndarray
    total: dict[str, float]
    total_squared: dict[str, float]
    observations: int
    segments: int

    @classmethod
    def create(cls) -> "AcfAccumulator":
        names = (
            "return",
            "absoluteReturn",
            "logAbsoluteReturn",
            "squaredReturn",
            "activity",
        )
        return cls(
            cross={name: np.zeros(MAX_LAG_SECONDS + 1) for name in names},
            pairs=np.zeros(MAX_LAG_SECONDS + 1, dtype=np.int64),
            total={name: 0.0 for name in names},
            total_squared={name: 0.0 for name in names},
            observations=0,
            segments=0,
        )

    def add_returns(self, returns_bps: np.ndarray) -> None:
        values = np.asarray(returns_bps, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size <= MAX_LAG_SECONDS:
            raise ValueError("ACF segment is shorter than the requested lag range")
        signals = {
            "return": values,
            "absoluteReturn": np.abs(values),
            "logAbsoluteReturn": np.log1p(np.abs(values)),
            "squaredReturn": values * values,
            "activity": (values != 0.0).astype(np.float64),
        }
        fft_size = next_fast_len(values.size * 2 - 1)
        for name, signal in signals.items():
            spectrum = rfft(signal, fft_size)
            autocross = irfft(spectrum * np.conjugate(spectrum), fft_size)
            self.cross[name] += autocross[: MAX_LAG_SECONDS + 1]
            self.total[name] += float(np.sum(signal, dtype=np.float64))
            self.total_squared[name] += float(np.dot(signal, signal))
        self.pairs += values.size - np.arange(MAX_LAG_SECONDS + 1)
        self.observations += int(values.size)
        self.segments += 1

    def finish(self) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for name in self.cross:
            mean = self.total[name] / self.observations
            variance = self.total_squared[name] / self.observations - mean * mean
            covariance = self.cross[name] / self.pairs - mean * mean
            result[name] = (
                covariance / variance
                if variance > 0
                else np.full_like(covariance, np.nan)
            ).tolist()
        return result


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis = read_json(repo / args.analysis)
    histograms = read_json(repo / args.histograms)
    prior_delay = read_json(repo / args.delay_aware)
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    start = datetime.fromisoformat(analysis["fullHistory"]["startTime"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(analysis["fullHistory"]["endTime"].replace("Z", "+00:00"))
    files = selected_files(source, start, end)
    if not files:
        raise RuntimeError("no complete one-second shards selected")

    minute_count = len(files) * 1_440 - 1
    counts = np.empty(minute_count, dtype=np.uint8)
    realized_variance = np.empty(minute_count, dtype=np.float64)
    minute_returns = np.empty(minute_count, dtype=np.float64)
    activity_masks = np.empty(minute_count, dtype=np.uint64)
    positive_masks = np.empty(minute_count, dtype=np.uint64)
    normalized_magnitude_profiles = np.empty(
        (minute_count, MINUTE_SECONDS),
        dtype=np.float16,
    )
    actual_acf = AcfAccumulator.create()
    cursor = 0
    previous_close: float | None = None
    positions = np.arange(MINUTE_SECONDS, dtype=np.uint64)

    for file_index, reference in enumerate(files):
        if file_index % 100 == 0:
            print(f"Reading one-second dependence {file_index}/{len(files)}...", file=sys.stderr)
        closes = read_candle_column(reference, "close")
        if closes.shape != (DAY_SECONDS,) or np.any(~np.isfinite(closes)) or np.any(closes <= 0):
            raise ValueError(f"invalid one-second closes: {reference}")
        logs = np.log(closes)
        returns = np.empty(DAY_SECONDS, dtype=np.float64)
        returns[1:] = np.diff(logs) * 10_000.0
        returns[0] = np.nan if previous_close is None else math.log(closes[0] / previous_close) * 10_000.0
        previous_close = float(closes[-1])
        actual_acf.add_returns(returns)
        rows = returns.reshape(-1, MINUTE_SECONDS)
        valid = np.all(np.isfinite(rows), axis=1)
        rows = rows[valid]
        active = rows != 0.0
        positive = rows > 0.0
        n = np.sum(active, axis=1, dtype=np.uint8)
        q = np.sum(rows * rows, axis=1, dtype=np.float64)
        r = np.sum(rows, axis=1, dtype=np.float64)
        mask = np.sum(
            np.left_shift(active.astype(np.uint64), positions[None, :]),
            axis=1,
            dtype=np.uint64,
        )
        positive_mask = np.sum(
            np.left_shift(positive.astype(np.uint64), positions[None, :]),
            axis=1,
            dtype=np.uint64,
        )
        scale = np.sqrt(q)
        normalized_magnitude = np.divide(
            np.abs(rows),
            scale[:, None],
            out=np.zeros_like(rows),
            where=scale[:, None] > 0,
        )
        end_cursor = cursor + rows.shape[0]
        counts[cursor:end_cursor] = n
        realized_variance[cursor:end_cursor] = q
        minute_returns[cursor:end_cursor] = r
        activity_masks[cursor:end_cursor] = mask
        positive_masks[cursor:end_cursor] = positive_mask
        normalized_magnitude_profiles[cursor:end_cursor] = normalized_magnitude.astype(np.float16)
        cursor = end_cursor

    if cursor != minute_count:
        raise RuntimeError(f"expected {minute_count} complete minutes, found {cursor}")

    states = fit_states(counts, realized_variance)
    transition = fit_transition(states["ids"], states["count"])
    positive_probability = positive_nonzero_probability(analysis)
    simulation = simulate_models(
        counts=counts,
        realized_variance=realized_variance,
        activity_masks=activity_masks,
        positive_masks=positive_masks,
        normalized_magnitude_profiles=normalized_magnitude_profiles,
        state_ids=states["ids"],
        positive_probability=positive_probability,
        histogram=full_histogram(histograms, "1m"),
        observed_sigma_bps=float(full_scale(analysis, "1m")["standardDeviationBps"]),
    )

    actual_acfs = actual_acf.finish()
    observed_histogram = dense_histogram(full_histogram(histograms, "1m"))
    prior_by_id = {item["id"]: item for item in prior_delay["models"]}
    model_rows = [
        normalize_prior_model(prior_by_id["iidSeconds"]),
        normalize_prior_model(prior_by_id["actualActivity"]),
        summarize_simulated_model(
            "clusteredVolatility",
            "State-dependent activity and realized variance; IID signs",
            simulation["clusteredVolatility"],
            observed_histogram,
            full_histogram(histograms, "1m"),
            float(full_scale(analysis, "1m")["standardDeviationBps"]),
        ),
        summarize_simulated_model(
            "fullNonJumpDependence",
            "Block-state activity/volatility plus joint normalized intraminute templates",
            simulation["fullNonJumpDependence"],
            observed_histogram,
            full_histogram(histograms, "1m"),
            float(full_scale(analysis, "1m")["standardDeviationBps"]),
        ),
    ]
    scale_validation = {"1m": model_rows}
    generated_minute_returns = simulation["minuteReturnsByModel"]
    for scale_id, factor in (("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1_440)):
        scale_histogram = full_histogram(histograms, scale_id)
        observed_probabilities = dense_histogram(scale_histogram)
        observed_sigma = float(full_scale(analysis, scale_id)["standardDeviationBps"])
        scale_validation[scale_id] = [
            summarize_raw_model(
                model_id,
                label,
                aggregate_aligned_minutes(values, factor),
                observed_probabilities,
                scale_histogram,
                observed_sigma,
            )
            for model_id, label, values in (
                (
                    "clusteredVolatility",
                    "State-dependent activity and realized variance; IID signs",
                    generated_minute_returns["clusteredVolatility"],
                ),
                (
                    "fullNonJumpDependence",
                    "Block-state activity/volatility plus joint normalized intraminute templates",
                    generated_minute_returns["fullNonJumpDependence"],
                ),
            )
        ]
    full_model_acf = simulation["fullModelAcf"]
    lags = selected_lags()
    state_ids = states["ids"]
    state_counts = np.bincount(state_ids, minlength=states["count"])
    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "commonEndTime": analysis["source"]["commonAnalysisEndTime"],
        "scope": {
            "returns": int(actual_acf.observations),
            "minutes": int(minute_count),
            "days": len(files),
            "separateJumpProcess": False,
            "extremeReturnTreatment": "Retained in empirical magnitude and volatility-state tails.",
        },
        "measuredDependence": {
            "acfBoundaryTreatment": (
                f"ACF pairs crossing the {actual_acf.segments} daily FFT segments are omitted; "
                f"at lag {MAX_LAG_SECONDS}s this omits less than "
                f"{100 * actual_acf.segments * MAX_LAG_SECONDS / actual_acf.observations:.4f}% of pairs."
            ),
            "lagsSeconds": lags,
            "returnCorrelation": pick_lags(actual_acfs["return"], lags),
            "absoluteReturnCorrelation": pick_lags(actual_acfs["absoluteReturn"], lags),
            "logAbsoluteReturnCorrelation": pick_lags(actual_acfs["logAbsoluteReturn"], lags),
            "squaredReturnCorrelation": pick_lags(actual_acfs["squaredReturn"], lags),
            "activityCorrelation": pick_lags(actual_acfs["activity"], lags),
            "returnCorrelationAllLagsThrough900s": actual_acfs["return"],
            "absoluteReturnCorrelationAllLagsThrough900s": actual_acfs["absoluteReturn"],
            "logAbsoluteReturnCorrelationAllLagsThrough900s": actual_acfs["logAbsoluteReturn"],
            "squaredReturnCorrelationAllLagsThrough900s": actual_acfs["squaredReturn"],
            "activityCorrelationAllLagsThrough900s": actual_acfs["activity"],
            "varianceRatioFromReturnAcf": variance_ratio_from_acf(actual_acfs["return"], 60),
        },
        "latentState": {
            "definition": (
                "A 4x4 discrete minute state: quartile of nonzero-second count crossed with "
                "quartile of RMS nonzero-return magnitude."
            ),
            "activityBinUpperEdges": states["activityEdges"].tolist(),
            "volatilityLogRmsBinUpperEdges": states["volatilityEdges"].tolist(),
            "stateCount": int(states["count"]),
            "stationaryProbability": (state_counts / minute_count).tolist(),
            "transitionMatrix": transition.tolist(),
            "sameStateProbability": float(np.mean(state_ids[1:] == state_ids[:-1])),
            "activityCountCorrelationMinutes": minute_correlations(counts.astype(np.float64)),
            "logRmsCorrelationMinutes": minute_correlations(states["logRms"]),
        },
        "model": {
            "class": "Semi-parametric block-state marked activity process",
            "minuteState": (
                "The joint minute state Z_m and realized-variance scale Q_m are sampled in "
                "a random permutation of contiguous 1,440-minute blocks without replacement. "
                "This retains slow activity/volatility persistence and the exact scale marginal; "
                "the fitted 16-state transition matrix is reported as a compact diagnostic."
            ),
            "activityEmission": (
                "Given Z_m, sample a joint empirical 60-second activity and normalized absolute-"
                "magnitude template from a historical minute in that state."
            ),
            "magnitudeEmission": (
                "Independently sample realized variance Q_m from another minute in the same state, "
                "then multiply the unit-L2 normalized magnitude template by sqrt(Q_m). This "
                "retains intraminute magnitude clustering without replaying the source returns."
            ),
            "signEmission": (
                "The full model keeps the empirical sign pattern paired with the activity/magnitude "
                "shape template, retaining within-minute signed autocorrelation and its interaction "
                "with activity. The volatility-only control replaces these signs IID."
            ),
            "return": "r_t = 0 for inactive template positions; otherwise r_t = sign_t * magnitude_t.",
            "innovationTail": "Empirical and untruncated; no separate jump indicator is fitted.",
            "seed": RNG_SEED,
            "stateBlockMinutes": 1_440,
        },
        "validation": {
            "models": model_rows,
            "modelsByScale": scale_validation,
            "fullModelAcfSampleMinutes": SIMULATED_ACF_MINUTES,
            "fullModelAcf": {
                "lagsSeconds": lags,
                "returnCorrelation": pick_lags(full_model_acf["return"], lags),
                "absoluteReturnCorrelation": pick_lags(full_model_acf["absoluteReturn"], lags),
                "logAbsoluteReturnCorrelation": pick_lags(full_model_acf["logAbsoluteReturn"], lags),
                "squaredReturnCorrelation": pick_lags(full_model_acf["squaredReturn"], lags),
                "activityCorrelation": pick_lags(full_model_acf["activity"], lags),
                "varianceRatioFromReturnAcf": variance_ratio_from_acf(full_model_acf["return"], 60),
            },
        },
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({
        "measured": {
            "varianceRatio60": report["measuredDependence"]["varianceRatioFromReturnAcf"],
            "lag1Return": report["measuredDependence"]["returnCorrelation"][0],
            "lag1Absolute": report["measuredDependence"]["absoluteReturnCorrelation"][0],
            "lag60Absolute": report["measuredDependence"]["absoluteReturnCorrelation"][lags.index(60)],
        },
        "models": [
            {
                "id": row["id"],
                "varianceRatio": row["varianceRatioObservedOverModel"],
                "centerRatio": row["centralMass"]["ratioObservedOverModel"],
                "tail3Ratio": row["threeSigmaTail"]["ratioObservedOverModel"],
                "jsBits": row["jsDivergenceBits"],
            }
            for row in model_rows
        ],
        "fullModelByScale": {
            scale_id: next(
                row for row in rows if row["id"] == "fullNonJumpDependence"
            )
            for scale_id, rows in scale_validation.items()
        },
        "modelAcf": report["validation"]["fullModelAcf"],
    }, indent=2))


def selected_files(source: Path, start: datetime, end: datetime) -> list[Path]:
    result: list[Path] = []
    for file in sorted(source.glob("????-??-??.json")):
        day = datetime.fromisoformat(file.stem).replace(tzinfo=timezone.utc)
        if start <= day and day.timestamp() + DAY_SECONDS <= end.timestamp():
            result.append(file)
    return result


def fit_states(counts: np.ndarray, realized_variance: np.ndarray) -> dict[str, object]:
    activity_edges = distinct_quantile_edges(counts.astype(np.float64), (0.25, 0.5, 0.75))
    rms = np.sqrt(realized_variance / np.maximum(counts.astype(np.float64), 1.0))
    positive_rms = rms[rms > 0]
    floor = float(np.quantile(positive_rms, 0.001))
    log_rms = np.log(np.maximum(rms, floor))
    volatility_edges = distinct_quantile_edges(log_rms, (0.25, 0.5, 0.75))
    activity_bin = np.digitize(counts, activity_edges, right=True)
    volatility_bin = np.digitize(log_rms, volatility_edges, right=True)
    activity_bins = activity_edges.size + 1
    volatility_bins = volatility_edges.size + 1
    raw_ids = activity_bin * volatility_bins + volatility_bin
    _, ids = np.unique(raw_ids, return_inverse=True)
    return {
        "ids": ids.astype(np.int16),
        "count": int(np.max(ids)) + 1,
        "activityEdges": activity_edges,
        "volatilityEdges": volatility_edges,
        "logRms": log_rms,
    }


def distinct_quantile_edges(values: np.ndarray, probabilities: tuple[float, ...]) -> np.ndarray:
    edges = np.quantile(values, probabilities, method="nearest").astype(np.float64)
    return np.unique(edges)


def fit_transition(states: np.ndarray, state_count: int, smoothing: float = 0.5) -> np.ndarray:
    matrix = np.full((state_count, state_count), smoothing, dtype=np.float64)
    np.add.at(matrix, (states[:-1], states[1:]), 1.0)
    matrix /= np.sum(matrix, axis=1, keepdims=True)
    return matrix


def positive_nonzero_probability(analysis: dict) -> float:
    one_second = next(
        scale for scale in analysis["fullHistory"]["scales"] if scale["id"] == "1s"
    )
    return float(one_second["positiveFraction"]) / (1.0 - float(one_second["zeroFraction"]))


def simulate_models(
    *,
    counts: np.ndarray,
    realized_variance: np.ndarray,
    activity_masks: np.ndarray,
    positive_masks: np.ndarray,
    normalized_magnitude_profiles: np.ndarray,
    state_ids: np.ndarray,
    positive_probability: float,
    histogram: dict,
    observed_sigma_bps: float,
) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED)
    minute_count = counts.size
    state_count = int(np.max(state_ids)) + 1
    state_pools = [np.flatnonzero(state_ids == state) for state in range(state_count)]
    state_driver_indexes = permuted_block_indexes(
        minute_count,
        block_length=1_440,
        rng=rng,
    )
    generated_states = state_ids[state_driver_indexes]

    edges = histogram_edges(histogram)
    histogram_counts = {
        "clusteredVolatility": np.zeros(edges.size - 1, dtype=np.int64),
        "fullNonJumpDependence": np.zeros(edges.size - 1, dtype=np.int64),
    }
    sums = {name: 0.0 for name in histogram_counts}
    sums_squared = {name: 0.0 for name in histogram_counts}
    central = {name: 0 for name in histogram_counts}
    tail3 = {name: 0 for name in histogram_counts}
    tail5 = {name: 0 for name in histogram_counts}
    model_path: list[np.ndarray] = []
    minute_returns_by_model = {
        name: np.empty(minute_count, dtype=np.float64)
        for name in histogram_counts
    }
    position_columns = np.arange(MINUTE_SECONDS, dtype=np.uint64)

    for batch_start in range(0, minute_count, SIMULATION_BATCH_MINUTES):
        batch_end = min(minute_count, batch_start + SIMULATION_BATCH_MINUTES)
        batch_states = generated_states[batch_start:batch_end]
        batch_size = batch_states.size
        shape_indexes = np.empty(batch_size, dtype=np.int64)
        for state in np.unique(batch_states):
            locations = np.flatnonzero(batch_states == state)
            pool = state_pools[int(state)]
            shape_indexes[locations] = rng.choice(pool, size=locations.size, replace=True)
        q = realized_variance[state_driver_indexes[batch_start:batch_end]]
        masks = activity_masks[shape_indexes]
        active_time = (
            np.right_shift(masks[:, None], position_columns[None, :]) & np.uint64(1)
        ).astype(bool)
        template_positive = (
            np.right_shift(positive_masks[shape_indexes, None], position_columns[None, :])
            & np.uint64(1)
        ).astype(bool)
        sampled_magnitudes = (
            normalized_magnitude_profiles[shape_indexes].astype(np.float64)
            * np.sqrt(q)[:, None]
        )
        iid_sign = np.where(
            rng.random((batch_size, MINUTE_SECONDS)) < positive_probability,
            1.0,
            -1.0,
        ) * active_time
        template_sign = np.where(template_positive, 1.0, -1.0) * active_time
        returns_by_model = {
            "clusteredVolatility": np.sum(iid_sign * sampled_magnitudes, axis=1),
            "fullNonJumpDependence": np.sum(template_sign * sampled_magnitudes, axis=1),
        }
        for name, values in returns_by_model.items():
            minute_returns_by_model[name][batch_start:batch_end] = values
            histogram_counts[name] += np.histogram(values, bins=edges)[0]
            sums[name] += float(np.sum(values, dtype=np.float64))
            sums_squared[name] += float(np.dot(values, values))
            central[name] += int(np.count_nonzero(np.abs(values) < 0.25 * observed_sigma_bps))
            tail3[name] += int(np.count_nonzero(np.abs(values) >= 3.0 * observed_sigma_bps))
            tail5[name] += int(np.count_nonzero(np.abs(values) >= 5.0 * observed_sigma_bps))

        remaining_path = SIMULATED_ACF_MINUTES - sum(item.shape[0] // MINUTE_SECONDS for item in model_path)
        if remaining_path > 0:
            take = min(remaining_path, batch_size)
            time_values = template_sign[:take] * sampled_magnitudes[:take]
            model_path.append(time_values.reshape(-1))

    result: dict[str, object] = {}
    for name in histogram_counts:
        result[name] = {
            "observations": minute_count,
            "probabilities": histogram_counts[name].astype(np.float64) / minute_count,
            "meanBps": sums[name] / minute_count,
            "sigmaBps": math.sqrt(
                max(0.0, sums_squared[name] / minute_count - (sums[name] / minute_count) ** 2)
            ),
            "centralMass": central[name] / minute_count,
            "tail3": tail3[name] / minute_count,
            "tail5": tail5[name] / minute_count,
        }
    model_acf_accumulator = AcfAccumulator.create()
    model_acf_accumulator.add_returns(np.concatenate(model_path))
    result["fullModelAcf"] = model_acf_accumulator.finish()
    result["minuteReturnsByModel"] = minute_returns_by_model
    return result


def permuted_block_indexes(
    length: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if length < 1 or block_length < 1 or block_length > length:
        raise ValueError("invalid stationary block dimensions")
    full_blocks, remainder = divmod(length, block_length)
    order = rng.permutation(full_blocks)
    parts = [
        np.arange(block * block_length, (block + 1) * block_length, dtype=np.int64)
        for block in order
    ]
    if remainder:
        parts.append(np.arange(full_blocks * block_length, length, dtype=np.int64))
    return np.concatenate(parts)


def summarize_simulated_model(
    model_id: str,
    label: str,
    model: dict,
    observed_probabilities: np.ndarray,
    histogram: dict,
    observed_sigma_bps: float,
) -> dict:
    centers = histogram_centers(histogram)
    observed_central = float(np.sum(observed_probabilities[np.abs(centers) < 0.25 * observed_sigma_bps]))
    observed_tail3 = float(np.sum(observed_probabilities[np.abs(centers) >= 3.0 * observed_sigma_bps]))
    observed_tail5 = float(np.sum(observed_probabilities[np.abs(centers) >= 5.0 * observed_sigma_bps]))
    probabilities = model["probabilities"]
    return {
        "id": model_id,
        "label": label,
        "sigmaBps": model["sigmaBps"],
        "varianceRatioObservedOverModel": (observed_sigma_bps / model["sigmaBps"]) ** 2,
        "jsDivergenceBits": jensen_shannon_bits(observed_probabilities, probabilities),
        "centralMass": {
            "observed": observed_central,
            "model": model["centralMass"],
            "ratioObservedOverModel": observed_central / model["centralMass"],
        },
        "threeSigmaTail": {
            "observed": observed_tail3,
            "model": model["tail3"],
            "ratioObservedOverModel": observed_tail3 / model["tail3"],
        },
        "fiveSigmaTail": {
            "observed": observed_tail5,
            "model": model["tail5"],
            "ratioObservedOverModel": observed_tail5 / model["tail5"],
        },
    }


def aggregate_aligned_minutes(values: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("aggregation factor must be positive")
    # The first modeled complete minute ends at common-start + 2 minutes, so its
    # minute index within the UTC day is 1. Advance to the next aligned endpoint.
    offset = (factor - 1) % factor
    selected = values[offset:]
    complete = selected.size // factor
    return np.sum(selected[: complete * factor].reshape(complete, factor), axis=1)


def summarize_raw_model(
    model_id: str,
    label: str,
    values: np.ndarray,
    observed_probabilities: np.ndarray,
    histogram: dict,
    observed_sigma_bps: float,
) -> dict:
    counts, _ = np.histogram(values, bins=histogram_edges(histogram))
    probabilities = counts.astype(np.float64) / values.size
    model_sigma = float(np.std(values))
    observed_centers = histogram_centers(histogram)
    observed_central = float(np.sum(
        observed_probabilities[np.abs(observed_centers) < 0.25 * observed_sigma_bps]
    ))
    observed_tail3 = float(np.sum(
        observed_probabilities[np.abs(observed_centers) >= 3.0 * observed_sigma_bps]
    ))
    observed_tail5 = float(np.sum(
        observed_probabilities[np.abs(observed_centers) >= 5.0 * observed_sigma_bps]
    ))
    model_central = float(np.mean(np.abs(values) < 0.25 * observed_sigma_bps))
    model_tail3 = float(np.mean(np.abs(values) >= 3.0 * observed_sigma_bps))
    model_tail5 = float(np.mean(np.abs(values) >= 5.0 * observed_sigma_bps))
    return {
        "id": model_id,
        "label": label,
        "observations": int(values.size),
        "sigmaBps": model_sigma,
        "varianceRatioObservedOverModel": (observed_sigma_bps / model_sigma) ** 2,
        "jsDivergenceBits": jensen_shannon_bits(observed_probabilities, probabilities),
        "centralMass": {
            "observed": observed_central,
            "model": model_central,
            "ratioObservedOverModel": observed_central / model_central,
        },
        "threeSigmaTail": {
            "observed": observed_tail3,
            "model": model_tail3,
            "ratioObservedOverModel": observed_tail3 / model_tail3,
        },
        "fiveSigmaTail": {
            "observed": observed_tail5,
            "model": model_tail5,
            "ratioObservedOverModel": (
                observed_tail5 / model_tail5 if model_tail5 > 0 else None
            ),
        },
    }


def normalize_prior_model(model: dict) -> dict:
    return {
        "id": model["id"],
        "label": model["label"],
        "sigmaBps": model["sigmaBps"],
        "varianceRatioObservedOverModel": model["varianceRatioObservedOverModel"],
        "jsDivergenceBits": model["jsDivergenceBits"],
        "centralMass": model["centralMass"],
        "threeSigmaTail": model["threeSigmaTail"],
        "fiveSigmaTail": model["fiveSigmaTail"],
    }


def full_window(report: dict, scale_id: str) -> dict:
    scale = next(item for item in report["scales"] if item["id"] == scale_id)
    return next(item for item in scale["windows"] if item["id"] == "full")


def full_scale(report: dict, scale_id: str) -> dict:
    return next(item for item in report["fullHistory"]["scales"] if item["id"] == scale_id)


def full_histogram(report: dict, scale_id: str) -> dict:
    return full_window(report, scale_id)["histogram"]


def dense_histogram(histogram: dict) -> np.ndarray:
    result = np.zeros(int(histogram["binCount"]), dtype=np.float64)
    for index, probability in histogram["nonzeroBins"]:
        result[int(index)] = float(probability)
    return result


def histogram_edges(histogram: dict) -> np.ndarray:
    return float(histogram["lowerBps"]) + np.arange(
        int(histogram["binCount"]) + 1,
        dtype=np.float64,
    ) * float(histogram["binWidthBps"])


def histogram_centers(histogram: dict) -> np.ndarray:
    edges = histogram_edges(histogram)
    return (edges[:-1] + edges[1:]) / 2.0


def jensen_shannon_bits(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("JS distributions must have equal shapes")
    midpoint = (left + right) / 2.0
    left_selected = left > 0
    right_selected = right > 0
    divergence = 0.5 * np.sum(left[left_selected] * np.log(left[left_selected] / midpoint[left_selected]))
    divergence += 0.5 * np.sum(right[right_selected] * np.log(right[right_selected] / midpoint[right_selected]))
    return float(divergence / math.log(2.0))


def variance_ratio_from_acf(acf: list[float], horizon: int) -> float:
    return 1.0 + 2.0 * sum(
        (1.0 - lag / horizon) * acf[lag]
        for lag in range(1, horizon)
    )


def selected_lags() -> list[int]:
    return list(range(1, 61)) + [90, 120, 180, 300, 600, 900]


def pick_lags(values: list[float], lags: list[int]) -> list[float]:
    return [float(values[lag]) for lag in lags]


def minute_correlations(values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for lag in (1, 5, 15, 60, 240, 1_440):
        result[str(lag)] = float(np.corrcoef(values[lag:], values[:-lag])[0, 1])
    return result


def read_json(file: Path) -> dict:
    return json.loads(file.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="data/benchmarks/log-return-distributions.json")
    parser.add_argument("--histograms", default="data/benchmarks/log-return-histograms.json")
    parser.add_argument(
        "--delay-aware",
        default="data/benchmarks/delay-aware-minute-return-distribution.json",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/one-second-dependence-model.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
