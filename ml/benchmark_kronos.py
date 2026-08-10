from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from differentiable_exposure_value_oracle import (
    DifferentiableExposureValueOracle,
    DifferentiableExposureValueOracleConfig,
)
from kronos_model_zoo import (
    KRONOS_SOURCE_COMMIT,
    KronosModelSpec,
    selected_specs,
    snapshot_dir,
    source_root,
)
from kronos_probabilistic import (
    DEFAULT_QUANTILE_LEVELS,
    empirical_ohlc_quantiles,
    generate_sample_paths,
    kline_valid_mask,
    kqsp,
    point_estimators,
)
from oracle_distribution_path import oracle_forward_kl_per_example
from trading_storage import candle_times, read_candle_column


STEP_MS = 60_000
HORIZON = 15
HORIZON_MS = HORIZON * STEP_MS
DEFAULT_LOOKBACK = 512
PROBABILITY_FLOOR = 1e-8
FORECAST_CONTRACT = "kronos-causal-15x1m-forecast-v2"
PRICE_COLUMNS = ("open", "high", "low", "close")
ALL_COLUMNS = (*PRICE_COLUMNS, "volume")
ORACLE_CONFIG = DifferentiableExposureValueOracleConfig(
    holding_period_steps=1,
    decision_delay_steps=1,
    value_horizon_steps=HORIZON,
    friction=0.00175,
    grid_size=101,
    temperature=0.01,
    min_exposure=-100,
    max_exposure=100,
    max_effective_exposure=250,
    quote_borrow_rate=0.000016658479699709332,
    asset_borrow_rate=0.000016658479699709332,
)
EXECUTION_ORACLE_CONFIG = DifferentiableExposureValueOracleConfig(
    # The bot receives one forecast every HORIZON candles and cannot rebalance
    # between them.  Hold the scored base action for that complete interval;
    # a one-candle hold would give the Bellman continuation clairvoyant
    # minute-by-minute rebalancing that the executing bot does not have.
    holding_period_steps=HORIZON,
    decision_delay_steps=HORIZON,
    value_horizon_steps=HORIZON,
    friction=0.00175,
    grid_size=101,
    temperature=0.01,
    min_exposure=-5,
    max_exposure=5,
    max_effective_exposure=12.5,
    quote_borrow_rate=0.000016658479699709332,
    asset_borrow_rate=0.000016658479699709332,
)


@dataclass(frozen=True)
class Window:
    id: str
    label: str
    group: str
    start_time: int
    end_time: int


@dataclass(frozen=True)
class Origin:
    target_start: int
    target_index: int
    window_ids: tuple[str, ...]


@dataclass(frozen=True)
class Corpus:
    times: np.ndarray
    values: np.ndarray
    references: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class MetricBatch:
    scalars: dict[str, np.ndarray]
    candle_actual: np.ndarray
    candle_predicted: np.ndarray
    close_return_actual: np.ndarray
    close_return_predicted: np.ndarray
    close_path_actual: np.ndarray
    close_path_predicted: np.ndarray
    horizon_return_actual: np.ndarray
    horizon_return_predicted: np.ndarray
    oracle_actual: np.ndarray
    oracle_predicted: np.ndarray


@dataclass(frozen=True)
class ProbabilisticBatch:
    scalars: dict[str, np.ndarray]
    raw_coverage: np.ndarray
    repaired_coverage: np.ndarray


class CorrelationAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.actual_sum = 0.0
        self.predicted_sum = 0.0
        self.actual_square_sum = 0.0
        self.predicted_square_sum = 0.0
        self.product_sum = 0.0

    def add(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        left = np.asarray(actual, dtype=np.float64).reshape(-1)
        right = np.asarray(predicted, dtype=np.float64).reshape(-1)
        if left.shape != right.shape or not np.isfinite(left).all() \
                or not np.isfinite(right).all():
            raise ValueError("correlation observations are invalid")
        self.count += left.size
        self.actual_sum += float(left.sum())
        self.predicted_sum += float(right.sum())
        self.actual_square_sum += float(np.dot(left, left))
        self.predicted_square_sum += float(np.dot(right, right))
        self.product_sum += float(np.dot(left, right))

    def result(self) -> float | None:
        if self.count < 2:
            return None
        actual_variance = (
            self.actual_square_sum - self.actual_sum**2 / self.count
        )
        predicted_variance = (
            self.predicted_square_sum - self.predicted_sum**2 / self.count
        )
        if actual_variance <= 0 or predicted_variance <= 0:
            return None
        covariance = self.product_sum - (
            self.actual_sum * self.predicted_sum / self.count
        )
        return float(covariance / math.sqrt(actual_variance * predicted_variance))

    def state(self) -> dict:
        return {
            "count": self.count,
            "actualSum": self.actual_sum,
            "predictedSum": self.predicted_sum,
            "actualSquareSum": self.actual_square_sum,
            "predictedSquareSum": self.predicted_square_sum,
            "productSum": self.product_sum,
        }

    @classmethod
    def from_state(cls, value: dict) -> CorrelationAccumulator:
        output = cls()
        output.count = int(value["count"])
        output.actual_sum = float(value["actualSum"])
        output.predicted_sum = float(value["predictedSum"])
        output.actual_square_sum = float(value["actualSquareSum"])
        output.predicted_square_sum = float(value["predictedSquareSum"])
        output.product_sum = float(value["productSum"])
        return output


class ForecastAccumulator:
    def __init__(self, action_count: int) -> None:
        self.examples = 0
        self.scalar_sums: defaultdict[str, float] = defaultdict(float)
        self.candle_correlation = CorrelationAccumulator()
        self.close_return_correlation = CorrelationAccumulator()
        self.close_path_correlation = CorrelationAccumulator()
        self.horizon_return_correlation = CorrelationAccumulator()
        self.horizon_return_actual: list[np.ndarray] = []
        self.horizon_return_predicted: list[np.ndarray] = []
        self.oracle_actual_sum = np.zeros(action_count, dtype=np.float64)
        self.oracle_predicted_sum = np.zeros(action_count, dtype=np.float64)

    def add(self, batch: MetricBatch, indexes: Iterable[int]) -> None:
        selected = np.fromiter(indexes, dtype=np.int64)
        if selected.size == 0:
            return
        self.examples += int(selected.size)
        for name, values in batch.scalars.items():
            self.scalar_sums[name] += float(values[selected].sum())
        self.candle_correlation.add(
            batch.candle_actual[selected], batch.candle_predicted[selected]
        )
        self.close_return_correlation.add(
            batch.close_return_actual[selected],
            batch.close_return_predicted[selected],
        )
        self.close_path_correlation.add(
            batch.close_path_actual[selected], batch.close_path_predicted[selected]
        )
        self.horizon_return_correlation.add(
            batch.horizon_return_actual[selected],
            batch.horizon_return_predicted[selected],
        )
        self.horizon_return_actual.append(batch.horizon_return_actual[selected])
        self.horizon_return_predicted.append(
            batch.horizon_return_predicted[selected]
        )
        self.oracle_actual_sum += batch.oracle_actual[selected].sum(axis=0)
        self.oracle_predicted_sum += batch.oracle_predicted[selected].sum(axis=0)

    def result(self, grid: np.ndarray) -> dict:
        if self.examples <= 0:
            raise RuntimeError("cannot finalize empty Kronos metrics")
        mean = {
            name: value / self.examples
            for name, value in self.scalar_sums.items()
        }
        actual_distribution = self.oracle_actual_sum / self.examples
        predicted_distribution = self.oracle_predicted_sum / self.examples
        return {
            "examples": self.examples,
            "candle": {
                "anchoredLogMse": mean["candleLogMse"],
                "persistenceAnchoredLogMse": mean["candlePersistenceMse"],
                "mseSkillVsPersistence": skill(
                    mean["candleLogMse"], mean["candlePersistenceMse"]
                ),
                "rawPriceMseUsd2": mean["rawPriceMse"],
                "anchoredLogCorrelation": self.candle_correlation.result(),
                "validOhlcFraction": mean["validOhlcFraction"],
            },
            "closeReturn": {
                "logReturnMse": mean["closeReturnMse"],
                "zeroBaselineMse": mean["closeReturnZeroMse"],
                "mseSkillVsZero": skill(
                    mean["closeReturnMse"], mean["closeReturnZeroMse"]
                ),
                "correlation": self.close_return_correlation.result(),
                "directionAccuracy": mean["directionAccuracy"],
                "cumulativeLogReturnMse": mean["cumulativeMse"],
                "zeroCumulativeMse": mean["zeroCumulativeMse"],
                "cumulativeMseSkillVsZero": skill(
                    mean["cumulativeMse"], mean["zeroCumulativeMse"]
                ),
                "horizonCorrelation": self.horizon_return_correlation.result(),
                "horizonRankCorrelation": spearman_correlation(
                    np.concatenate(self.horizon_return_actual),
                    np.concatenate(self.horizon_return_predicted),
                ),
                "horizonDirectionAccuracy": mean["horizonDirectionAccuracy"],
            },
            "closePath": {
                "anchoredLogMse": mean["closePathMse"],
                "persistenceAnchoredLogMse": mean["closePathPersistenceMse"],
                "mseSkillVsPersistence": skill(
                    mean["closePathMse"], mean["closePathPersistenceMse"]
                ),
                "correlation": self.close_path_correlation.result(),
            },
            "paperAligned": {
                "priceSeriesIc": mean["priceSeriesIc"],
                "priceSeriesRankIc": mean["priceSeriesRankIc"],
                "horizonReturnIc": self.horizon_return_correlation.result(),
                "horizonReturnRankIc": spearman_correlation(
                    np.concatenate(self.horizon_return_actual),
                    np.concatenate(self.horizon_return_predicted),
                ),
            },
            "oracle": {
                "forwardKl": mean["oracleKl"],
                "probabilityMse": mean["oracleProbabilityMse"],
                "totalVariation": mean["oracleTotalVariation"],
                "modalActionAgreement": mean["oracleModeAgreement"],
                "expectedExposureMae": mean["oracleExpectedExposureMae"],
                "predictedEntropy": mean["oraclePredictedEntropy"],
                "actualEntropy": mean["oracleActualEntropy"],
                "meanPredictedDistribution": predicted_distribution.tolist(),
                "meanActualDistribution": actual_distribution.tolist(),
                "meanPredictedExpectedExposure": float(
                    np.dot(predicted_distribution, grid)
                ),
                "meanActualExpectedExposure": float(
                    np.dot(actual_distribution, grid)
                ),
                "meanPredictedModalExposure": float(
                    grid[int(predicted_distribution.argmax())]
                ),
                "meanActualModalExposure": float(
                    grid[int(actual_distribution.argmax())]
                ),
            },
        }

    def state(self) -> dict:
        return {
            "examples": self.examples,
            "scalarSums": dict(self.scalar_sums),
            "candleCorrelation": self.candle_correlation.state(),
            "closeReturnCorrelation": self.close_return_correlation.state(),
            "closePathCorrelation": self.close_path_correlation.state(),
            "horizonReturnCorrelation": self.horizon_return_correlation.state(),
            "horizonReturnActual": self._horizon_values(
                self.horizon_return_actual
            ),
            "horizonReturnPredicted": self._horizon_values(
                self.horizon_return_predicted
            ),
            "oracleActualSum": self.oracle_actual_sum.tolist(),
            "oraclePredictedSum": self.oracle_predicted_sum.tolist(),
        }

    @staticmethod
    def _horizon_values(parts: list[np.ndarray]) -> list[float]:
        return np.concatenate(parts).astype(np.float64).tolist() if parts else []

    @classmethod
    def from_state(cls, value: dict, action_count: int) -> ForecastAccumulator:
        output = cls(action_count)
        output.examples = int(value["examples"])
        output.scalar_sums.update({
            str(name): float(total)
            for name, total in value["scalarSums"].items()
        })
        output.candle_correlation = CorrelationAccumulator.from_state(
            value["candleCorrelation"]
        )
        output.close_return_correlation = CorrelationAccumulator.from_state(
            value["closeReturnCorrelation"]
        )
        output.close_path_correlation = CorrelationAccumulator.from_state(
            value["closePathCorrelation"]
        )
        output.horizon_return_correlation = CorrelationAccumulator.from_state(
            value["horizonReturnCorrelation"]
        )
        actual = np.asarray(value["horizonReturnActual"], dtype=np.float64)
        predicted = np.asarray(
            value["horizonReturnPredicted"], dtype=np.float64
        )
        if actual.shape != predicted.shape or actual.shape != (output.examples,):
            raise ValueError("invalid forecast accumulator horizon state")
        if output.examples:
            output.horizon_return_actual.append(actual)
            output.horizon_return_predicted.append(predicted)
        output.oracle_actual_sum = np.asarray(
            value["oracleActualSum"], dtype=np.float64
        )
        output.oracle_predicted_sum = np.asarray(
            value["oraclePredictedSum"], dtype=np.float64
        )
        if output.oracle_actual_sum.shape != (action_count,) \
                or output.oracle_predicted_sum.shape != (action_count,):
            raise ValueError("invalid forecast accumulator oracle state")
        return output


class ProbabilisticAccumulator:
    def __init__(self, levels: np.ndarray) -> None:
        self.levels = np.asarray(levels, dtype=np.float64)
        self.examples = 0
        self.observations = 0
        self.scalar_sums: defaultdict[str, float] = defaultdict(float)
        self.raw_coverage_sum = np.zeros((self.levels.size, 4), dtype=np.float64)
        self.repaired_coverage_sum = np.zeros_like(self.raw_coverage_sum)

    def add(self, batch: ProbabilisticBatch, indexes: Iterable[int]) -> None:
        selected = np.fromiter(indexes, dtype=np.int64)
        if selected.size == 0:
            return
        self.examples += int(selected.size)
        self.observations += int(selected.size * HORIZON)
        for name, values in batch.scalars.items():
            self.scalar_sums[name] += float(values[selected].sum())
        self.raw_coverage_sum += batch.raw_coverage[selected].sum(axis=(0, 1))
        self.repaired_coverage_sum += batch.repaired_coverage[selected].sum(
            axis=(0, 1)
        )

    def result(self) -> dict:
        if self.examples <= 0 or self.observations <= 0:
            raise RuntimeError("cannot finalize empty probabilistic metrics")
        mean = {
            name: value / self.examples
            for name, value in self.scalar_sums.items()
        }
        raw_coverage = self.raw_coverage_sum / self.observations
        repaired_coverage = self.repaired_coverage_sum / self.observations
        expected = np.repeat(self.levels[:, None], 4, axis=1)
        return {
            "examples": self.examples,
            "samplePathCrpsAnchoredLog": mean["samplePathCrps"],
            "samplePathValidOhlcFraction": mean["samplePathValidOhlcFraction"],
            "originsWithAnyInvalidSamplePathFraction": mean[
                "originsWithAnyInvalidSamplePath"
            ],
            "rawQuantileValidOhlcFraction": mean["rawQuantileValidOhlcFraction"],
            "repairedQuantileValidOhlcFraction": mean[
                "repairedQuantileValidOhlcFraction"
            ],
            "rawAverageQuantileLossAnchoredLog": mean["rawPinball"],
            "repairedAverageQuantileLossAnchoredLog": mean["repairedPinball"],
            "rawQuantileCoverageError": float(np.abs(
                raw_coverage - expected
            ).mean()),
            "repairedQuantileCoverageError": float(np.abs(
                repaired_coverage - expected
            ).mean()),
            "rawCoverageByQuantileAndFeature": raw_coverage.tolist(),
            "repairedCoverageByQuantileAndFeature": repaired_coverage.tolist(),
            "central80Coverage": mean["central80Coverage"],
            "central80MeanWidthAnchoredLog": mean["central80Width"],
        }

    def state(self) -> dict:
        return {
            "levels": self.levels.tolist(),
            "examples": self.examples,
            "observations": self.observations,
            "scalarSums": dict(self.scalar_sums),
            "rawCoverageSum": self.raw_coverage_sum.tolist(),
            "repairedCoverageSum": self.repaired_coverage_sum.tolist(),
        }

    @classmethod
    def from_state(
        cls, value: dict, levels: np.ndarray
    ) -> ProbabilisticAccumulator:
        output = cls(levels)
        stored_levels = np.asarray(value["levels"], dtype=np.float64)
        if not np.array_equal(stored_levels, output.levels):
            raise ValueError("probabilistic accumulator quantiles changed")
        output.examples = int(value["examples"])
        output.observations = int(value["observations"])
        output.scalar_sums.update({
            str(name): float(total)
            for name, total in value["scalarSums"].items()
        })
        output.raw_coverage_sum = np.asarray(
            value["rawCoverageSum"], dtype=np.float64
        )
        output.repaired_coverage_sum = np.asarray(
            value["repairedCoverageSum"], dtype=np.float64
        )
        expected_shape = (output.levels.size, 4)
        if output.raw_coverage_sum.shape != expected_shape \
                or output.repaired_coverage_sum.shape != expected_shape:
            raise ValueError("invalid probabilistic accumulator coverage state")
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all public Kronos models on non-overlapping 15x1m "
            "forecasts throughout the non-fit inspector windows."
        )
    )
    parser.add_argument("--windows-json", required=True)
    parser.add_argument("--models", default="all", help="all or mini,small,base")
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        help="Local fine-tuned predictor checkpoint; requires exactly one model.",
    )
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=Path,
        help="Optional local fine-tuned tokenizer checkpoint.",
    )
    parser.add_argument(
        "--model-label",
        help="Report/partial identifier for a local checkpoint.",
    )
    parser.add_argument(
        "--ensemble-predictor-checkpoint",
        help="Second predictor checkpoint path, or 'pretrained', for path ensembling.",
    )
    parser.add_argument(
        "--ensemble-tokenizer-checkpoint",
        type=Path,
        help="Optional tokenizer paired with the second predictor.",
    )
    parser.add_argument(
        "--ensemble-sample-count",
        type=int,
        help="Paths allocated to the second predictor from --sample-count.",
    )
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--point-estimator",
        choices=("ensembleMean", "projectedMean", "ensembleMedian", "kqspMedian"),
        default="projectedMean",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--inference-precision",
        choices=("float32", "float16"),
        default="float32",
        help=(
            "CUDA inference precision. float16 enables mixed precision and "
            "must be validated against float32 before a final run."
        ),
    )
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/benchmarks/"
            "kronos-probabilistic-15m-inspector-windows-2026-08-06.json"
        ),
    )
    parser.add_argument(
        "--forecast-output",
        type=Path,
        help="Optional causal per-origin forecast artifact; requires one model.",
    )
    parser.add_argument(
        "--max-origins-per-window",
        type=int,
        help="Deterministic smoke-test cap; omitted for complete window coverage.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching atomic per-model partial results.",
    )
    parser.add_argument(
        "--progress-checkpoint-seconds",
        type=float,
        default=120.0,
        help="Atomic within-model checkpoint cadence; zero disables it.",
    )
    return parser.parse_args()


def parse_windows(payload: str) -> tuple[Window, ...]:
    values = json.loads(payload)
    if not isinstance(values, list):
        raise ValueError("inspector windows JSON must be a list")
    windows = tuple(Window(
        id=str(value["id"]),
        label=str(value["label"]),
        group=str(value["group"]),
        start_time=int(value["startTime"]),
        end_time=int(value["endTime"]),
    ) for value in values)
    if len(windows) != 28 or len({window.id for window in windows}) != len(windows):
        raise ValueError(f"expected 28 unique non-fit inspector windows, got {len(windows)}")
    if any(window.id == "latest" or window.id.startswith("fit-") for window in windows):
        raise ValueError("fit and latest windows must be excluded")
    if any(
        window.start_time % STEP_MS != 0
        or window.end_time % STEP_MS != 0
        or window.end_time <= window.start_time
        for window in windows
    ):
        raise ValueError("inspector windows are not complete minute ranges")
    return windows


def day_strings(start_ms: int, end_ms: int) -> Iterable[str]:
    current = datetime.fromtimestamp(start_ms / 1_000, tz=timezone.utc).date()
    final = datetime.fromtimestamp((end_ms - 1) / 1_000, tz=timezone.utc).date()
    while current <= final:
        yield current.isoformat()
        current += timedelta(days=1)


def load_corpus(root: Path, windows: tuple[Window, ...], lookback: int) -> Corpus:
    required_days: set[str] = set()
    warmup_ms = lookback * STEP_MS
    for window in windows:
        required_days.update(day_strings(window.start_time - warmup_ms, window.end_time))
    references = tuple(root / f"{day}.json" for day in sorted(required_days))
    missing = [file for file in references if not file.is_file()]
    if missing:
        preview = ", ".join(file.stem for file in missing[:8])
        raise FileNotFoundError(
            f"missing {len(missing)} canonical 1m candle days (first: {preview}); "
            "fetch the inspector history before running Kronos"
        )
    time_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    fingerprint = hashlib.sha256()
    for index, reference in enumerate(references, start=1):
        fingerprint.update(reference.name.encode("utf-8"))
        fingerprint.update(reference.read_bytes())
        times = candle_times(reference).astype(np.int64, copy=False)
        columns = [
            read_candle_column(reference, name).astype(np.float64, copy=False)
            for name in ALL_COLUMNS
        ]
        values = np.column_stack(columns)
        if times.shape != (values.shape[0],) or not np.isfinite(values).all():
            raise ValueError(f"invalid canonical candle shard: {reference}")
        time_parts.append(times)
        value_parts.append(values)
        if index % 50 == 0 or index == len(references):
            print(f"DATA loaded {index}/{len(references)} daily candle shards", flush=True)
    times = np.concatenate(time_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    if np.any(np.diff(times) <= 0):
        raise ValueError("selected canonical candle timestamps overlap or regress")
    return Corpus(
        times=times,
        values=values,
        references=tuple(str(file).replace("\\", "/") for file in references),
        fingerprint=fingerprint.hexdigest(),
    )


def build_origins(
    corpus: Corpus,
    windows: tuple[Window, ...],
    lookback: int,
    max_per_window: int | None,
) -> tuple[Origin, ...]:
    memberships: defaultdict[int, list[str]] = defaultdict(list)
    indexes: dict[int, int] = {}
    for window in windows:
        candidates = list(range(
            window.start_time,
            window.end_time - HORIZON_MS + 1,
            HORIZON_MS,
        ))
        if max_per_window is not None and len(candidates) > max_per_window:
            positions = np.linspace(
                0, len(candidates) - 1, num=max_per_window, dtype=np.int64
            )
            candidates = [candidates[int(position)] for position in positions]
        for target_start in candidates:
            target_index = int(np.searchsorted(corpus.times, target_start))
            start = target_index - lookback
            end = target_index + HORIZON
            expected = np.arange(
                target_start - lookback * STEP_MS,
                target_start + HORIZON_MS,
                STEP_MS,
                dtype=np.int64,
            )
            if start < 0 or end > corpus.times.size \
                    or not np.array_equal(corpus.times[start:end], expected):
                raise ValueError(
                    f"{window.id}: missing contiguous context/target at "
                    f"{iso_time(target_start)}"
                )
            memberships[target_start].append(window.id)
            indexes[target_start] = target_index
    return tuple(
        Origin(
            target_start=target_start,
            target_index=indexes[target_start],
            window_ids=tuple(memberships[target_start]),
        )
        for target_start in sorted(indexes)
    )


def timestamp_features(timestamps: np.ndarray) -> np.ndarray:
    flat = pd.DatetimeIndex(pd.to_datetime(
        timestamps.reshape(-1), unit="ms", utc=True
    ))
    features = np.column_stack((
        flat.minute,
        flat.hour,
        flat.weekday,
        flat.day,
        flat.month,
    )).astype(np.float32)
    return features.reshape(*timestamps.shape, 5)


def predictor_inputs(
    corpus: Corpus,
    origins: tuple[Origin, ...],
    start: int,
    end: int,
    lookback: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = origins[start:end]
    context = np.stack([
        corpus.values[origin.target_index - lookback:origin.target_index]
        for origin in rows
    ])
    target = np.stack([
        corpus.values[origin.target_index:origin.target_index + HORIZON, :4]
        for origin in rows
    ])
    amount = context[..., 4:5] * context[..., :4].mean(axis=-1, keepdims=True)
    model_context = np.concatenate((context, amount), axis=-1).astype(np.float32)
    means = model_context.mean(axis=1)
    stds = model_context.std(axis=1)
    normalized = np.clip(
        (model_context - means[:, None, :]) / (stds[:, None, :] + 1e-5),
        -5,
        5,
    )
    context_times = np.stack([
        corpus.times[origin.target_index - lookback:origin.target_index]
        for origin in rows
    ])
    target_times = np.stack([
        corpus.times[origin.target_index:origin.target_index + HORIZON]
        for origin in rows
    ])
    anchors = context[:, -1, 3].astype(np.float64)
    return (
        normalized.astype(np.float32, copy=False),
        timestamp_features(context_times),
        timestamp_features(target_times),
        means,
        stds,
        target,
        anchors,
    )


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.where(
        probabilities > 0,
        probabilities * np.log(np.clip(probabilities, np.finfo(np.float64).tiny, None)),
        0,
    ).sum(axis=-1)


def average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float | None:
    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        return None
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = math.sqrt(
        float(np.dot(left_rank, left_rank) * np.dot(right_rank, right_rank))
    )
    return float(np.dot(left_rank, right_rank) / denominator) \
        if denominator > 0 else None


def row_correlation(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(predicted, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("row correlation inputs must be matching matrices")
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(left_centered).sum(axis=1)
        * np.square(right_centered).sum(axis=1)
    )
    numerator = (left_centered * right_centered).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def paper_price_correlations(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = actual.shape[0]
    actual_rows = np.moveaxis(actual, 2, 1).reshape(-1, actual.shape[1])
    predicted_rows = np.moveaxis(predicted, 2, 1).reshape(-1, predicted.shape[1])
    ic = row_correlation(actual_rows, predicted_rows).reshape(batch_size, 4).mean(1)
    actual_rank = np.stack([average_ranks(row) for row in actual_rows])
    predicted_rank = np.stack([average_ranks(row) for row in predicted_rows])
    rank_ic = row_correlation(actual_rank, predicted_rank).reshape(
        batch_size, 4
    ).mean(1)
    return ic, rank_ic


def oracle_path_mixture(
    paths: np.ndarray,
    actual: np.ndarray,
    anchors: np.ndarray,
    oracle: DifferentiableExposureValueOracle,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = oracle_path_probabilities(paths, anchors, oracle, device)
    actual_close = np.log(
        np.asarray(actual[..., 3], dtype=np.float64)
        / anchors[:, None]
    )
    actual_returns = np.diff(
        np.column_stack((np.zeros(actual_close.shape[0]), actual_close)),
        axis=1,
    )
    with torch.no_grad():
        realized = oracle.forward_from_log_returns(torch.as_tensor(
            actual_returns,
            dtype=torch.float32,
            device=device,
        )).probabilities
    return predicted, realized.double().cpu().numpy()


def oracle_path_probabilities(
    paths: np.ndarray,
    anchors: np.ndarray,
    oracle: DifferentiableExposureValueOracle,
    device: torch.device,
) -> np.ndarray:
    mixture, _ = oracle_path_distributions(paths, anchors, oracle, device)
    return mixture


def oracle_path_distributions(
    paths: np.ndarray,
    anchors: np.ndarray,
    oracle: DifferentiableExposureValueOracle,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return oracle-vote and expected-log-utility action distributions."""
    path_close = np.log(
        np.asarray(paths[..., 3], dtype=np.float64)
        / anchors[:, None, None]
    )
    path_returns = np.diff(
        np.concatenate((
            np.zeros((*path_close.shape[:2], 1), dtype=np.float64),
            path_close,
        ), axis=2),
        axis=2,
    )
    with torch.no_grad():
        flat = torch.as_tensor(
            path_returns.reshape(-1, HORIZON),
            dtype=torch.float32,
            device=device,
        )
        output = oracle.forward_from_log_returns(flat)
        path_probabilities = output.probabilities.reshape(
            paths.shape[0], paths.shape[1], -1
        )
        mixture = path_probabilities.mean(dim=1)
        action_values = output.action_values.reshape(
            paths.shape[0], paths.shape[1], -1
        )
        expected_utility = torch.softmax(
            action_values.mean(dim=1) / oracle.config.temperature,
            dim=-1,
        )
    return (
        mixture.double().cpu().numpy(),
        expected_utility.double().cpu().numpy(),
    )


def causal_forecast_rows(
    origins: tuple[Origin, ...],
    paths: np.ndarray,
    predictions: dict[str, np.ndarray],
    predicted_oracle: np.ndarray,
    execution_oracle: np.ndarray,
    execution_utility: np.ndarray,
    anchors: np.ndarray,
) -> list[dict]:
    if paths.shape[0] != len(origins) \
            or predicted_oracle.shape[0] != len(origins) \
            or execution_oracle.shape[0] != len(origins) \
            or execution_utility.shape[0] != len(origins):
        raise ValueError("causal forecast batch dimensions do not match origins")
    horizon_returns = np.log(paths[:, :, -1, 3] / anchors[:, None])
    mean_close = predictions["ensembleMean"][..., 3]
    median_close = predictions["ensembleMedian"][..., 3]
    output = []
    for index, origin in enumerate(origins):
        returns = horizon_returns[index]
        output.append({
            # The signal becomes available when the final context candle closes,
            # one millisecond before the first target candle opens.
            "decisionTime": origin.target_start - 1,
            "targetStartTime": origin.target_start,
            "windowIds": list(origin.window_ids),
            "anchorPrice": float(anchors[index]),
            "horizonLogReturnMean": float(returns.mean()),
            "horizonLogReturnMedian": float(np.median(returns)),
            "horizonLogReturnStd": float(returns.std()),
            "horizonUpProbability": float((returns > 0).mean()),
            "horizonLogReturnP10": float(np.quantile(returns, 0.1)),
            "horizonLogReturnP90": float(np.quantile(returns, 0.9)),
            "meanCloseLogPath": np.log(
                mean_close[index] / anchors[index]
            ).tolist(),
            "medianCloseLogPath": np.log(
                median_close[index] / anchors[index]
            ).tolist(),
            "oracleProbabilities": predicted_oracle[index].tolist(),
            "executionOracleProbabilities": execution_oracle[index].tolist(),
            "executionUtilityProbabilities": execution_utility[index].tolist(),
        })
    return output


def probabilistic_batch(
    paths: np.ndarray,
    raw_quantiles: np.ndarray,
    repaired_quantiles: np.ndarray,
    actual: np.ndarray,
    anchors: np.ndarray,
    levels: np.ndarray,
) -> ProbabilisticBatch:
    anchors_expanded = anchors[:, None, None]
    actual_log = np.log(actual[..., :4] / anchors_expanded)
    path_log = np.log(paths[..., :4] / anchors[:, None, None, None])
    raw_log = np.log(raw_quantiles / anchors[:, None, None, None])
    repaired_log = np.log(
        repaired_quantiles / anchors[:, None, None, None]
    )
    target = actual_log[:, :, None, :]
    quantile_levels = levels[None, None, :, None]

    def pinball(quantiles: np.ndarray) -> np.ndarray:
        error = target - quantiles
        return np.maximum(
            quantile_levels * error,
            (quantile_levels - 1.0) * error,
        ).mean(axis=(1, 2, 3))

    sample_absolute_error = np.abs(
        path_log - actual_log[:, None, :, :]
    ).mean(axis=(1, 2, 3))
    pairwise = np.abs(
        path_log[:, :, None, :, :] - path_log[:, None, :, :, :]
    ).mean(axis=(1, 2, 3, 4))
    raw_valid = kline_valid_mask(raw_quantiles)
    repaired_valid = kline_valid_mask(repaired_quantiles)
    path_valid = kline_valid_mask(paths[..., :4])
    low = repaired_log[:, :, 0, :]
    high = repaired_log[:, :, -1, :]
    return ProbabilisticBatch(
        scalars={
            "samplePathCrps": sample_absolute_error - 0.5 * pairwise,
            "samplePathValidOhlcFraction": path_valid.mean(axis=(1, 2)),
            "originsWithAnyInvalidSamplePath": (~path_valid).any(
                axis=(1, 2)
            ).astype(np.float64),
            "rawQuantileValidOhlcFraction": raw_valid.mean(axis=(1, 2)),
            "repairedQuantileValidOhlcFraction": repaired_valid.mean(
                axis=(1, 2)
            ),
            "rawPinball": pinball(raw_log),
            "repairedPinball": pinball(repaired_log),
            "central80Coverage": (
                (actual_log >= low) & (actual_log <= high)
            ).mean(axis=(1, 2)),
            "central80Width": (high - low).mean(axis=(1, 2)),
        },
        raw_coverage=target <= raw_log,
        repaired_coverage=target <= repaired_log,
    )


def metric_batch(
    prediction: np.ndarray,
    actual: np.ndarray,
    anchors: np.ndarray,
    predicted_oracle: np.ndarray,
    actual_oracle: np.ndarray,
    grid: np.ndarray,
) -> MetricBatch:
    predicted_ohlc = np.asarray(prediction[..., :4], dtype=np.float64)
    actual_ohlc = np.asarray(actual[..., :4], dtype=np.float64)
    if not np.isfinite(predicted_ohlc).all() or np.any(predicted_ohlc <= 0):
        raise ValueError("Kronos emitted a non-positive or non-finite OHLC prediction")
    anchored_actual = np.log(actual_ohlc / anchors[:, None, None])
    anchored_predicted = np.log(predicted_ohlc / anchors[:, None, None])
    actual_close_path = anchored_actual[..., 3]
    predicted_close_path = anchored_predicted[..., 3]
    actual_close_returns = np.diff(np.column_stack((
        np.zeros(actual_close_path.shape[0]), actual_close_path
    )), axis=1)
    predicted_close_returns = np.diff(np.column_stack((
        np.zeros(predicted_close_path.shape[0]), predicted_close_path
    )), axis=1)
    horizon_actual = actual_close_returns.sum(axis=1)
    horizon_predicted = predicted_close_returns.sum(axis=1)
    price_ic, price_rank_ic = paper_price_correlations(
        anchored_actual,
        anchored_predicted,
    )
    predicted_open = predicted_ohlc[..., 0]
    predicted_high = predicted_ohlc[..., 1]
    predicted_low = predicted_ohlc[..., 2]
    predicted_close = predicted_ohlc[..., 3]
    valid_ohlc = (
        (predicted_high >= np.maximum.reduce((
            predicted_open, predicted_low, predicted_close
        )))
        & (predicted_low <= np.minimum.reduce((
            predicted_open, predicted_high, predicted_close
        )))
    )
    with torch.no_grad():
        kl = oracle_forward_kl_per_example(
            torch.as_tensor(predicted_oracle, dtype=torch.float32),
            torch.as_tensor(actual_oracle, dtype=torch.float32),
            probability_floor=PROBABILITY_FLOOR,
        ).cpu().numpy().astype(np.float64)
    predicted_expected = predicted_oracle @ grid
    actual_expected = actual_oracle @ grid
    scalars = {
        "candleLogMse": np.square(
            anchored_predicted - anchored_actual
        ).mean(axis=(1, 2)),
        "candlePersistenceMse": np.square(anchored_actual).mean(axis=(1, 2)),
        "rawPriceMse": np.square(predicted_ohlc - actual_ohlc).mean(axis=(1, 2)),
        "validOhlcFraction": valid_ohlc.mean(axis=1),
        "closeReturnMse": np.square(
            predicted_close_returns - actual_close_returns
        ).mean(axis=1),
        "closeReturnZeroMse": np.square(actual_close_returns).mean(axis=1),
        "directionAccuracy": (
            np.sign(predicted_close_returns) == np.sign(actual_close_returns)
        ).mean(axis=1),
        "cumulativeMse": np.square(
            predicted_close_returns.sum(axis=1) - actual_close_returns.sum(axis=1)
        ),
        "zeroCumulativeMse": np.square(actual_close_returns.sum(axis=1)),
        "horizonDirectionAccuracy": (
            np.sign(horizon_predicted) == np.sign(horizon_actual)
        ).astype(np.float64),
        "closePathMse": np.square(
            predicted_close_path - actual_close_path
        ).mean(axis=1),
        "closePathPersistenceMse": np.square(actual_close_path).mean(axis=1),
        "priceSeriesIc": price_ic,
        "priceSeriesRankIc": price_rank_ic,
        "oracleKl": kl,
        "oracleProbabilityMse": np.square(
            predicted_oracle - actual_oracle
        ).mean(axis=1),
        "oracleTotalVariation": 0.5 * np.abs(
            predicted_oracle - actual_oracle
        ).sum(axis=1),
        "oracleModeAgreement": (
            predicted_oracle.argmax(axis=1) == actual_oracle.argmax(axis=1)
        ).astype(np.float64),
        "oracleExpectedExposureMae": np.abs(predicted_expected - actual_expected),
        "oraclePredictedEntropy": entropy(predicted_oracle),
        "oracleActualEntropy": entropy(actual_oracle),
    }
    return MetricBatch(
        scalars=scalars,
        candle_actual=anchored_actual,
        candle_predicted=anchored_predicted,
        close_return_actual=actual_close_returns,
        close_return_predicted=predicted_close_returns,
        close_path_actual=actual_close_path,
        close_path_predicted=predicted_close_path,
        horizon_return_actual=horizon_actual,
        horizon_return_predicted=horizon_predicted,
        oracle_actual=actual_oracle,
        oracle_predicted=predicted_oracle,
    )


def evaluate_model(
    repo_root: Path,
    spec,
    model_id: str,
    corpus: Corpus,
    origins: tuple[Origin, ...],
    windows: tuple[Window, ...],
    args: argparse.Namespace,
    progress_file: Path,
    run_signature: str,
) -> dict:
    source = source_root(repo_root)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Kronos is not installed; run npm run kronos:setup first"
        ) from error
    model_path = args.model_checkpoint or snapshot_dir(
        repo_root, spec.model_repo
    )
    tokenizer_path = args.tokenizer_checkpoint or snapshot_dir(
        repo_root, spec.tokenizer_repo
    )
    if not (model_path / "model.safetensors").is_file() \
            or not (tokenizer_path / "model.safetensors").is_file():
        raise FileNotFoundError("Kronos checkpoints are missing; run npm run kronos:setup")
    if args.lookback > spec.max_context:
        raise ValueError(
            f"lookback {args.lookback} exceeds {spec.id} context {spec.max_context}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see the GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(model_path))
    predictor = KronosPredictor(
        model, tokenizer, device=str(device), max_context=args.lookback
    )
    predictor.model.eval()
    predictor.tokenizer.eval()
    ensemble_predictor = None
    ensemble_model = None
    ensemble_tokenizer = None
    ensemble_model_path = None
    ensemble_tokenizer_path = None
    if args.ensemble_predictor_checkpoint is not None:
        ensemble_model_path = (
            snapshot_dir(repo_root, spec.model_repo)
            if args.ensemble_predictor_checkpoint == "pretrained"
            else args.ensemble_predictor_checkpoint
        )
        ensemble_tokenizer_path = (
            args.ensemble_tokenizer_checkpoint
            or snapshot_dir(repo_root, spec.tokenizer_repo)
        )
        ensemble_tokenizer = KronosTokenizer.from_pretrained(
            str(ensemble_tokenizer_path)
        )
        ensemble_model = Kronos.from_pretrained(str(ensemble_model_path))
        ensemble_predictor = KronosPredictor(
            ensemble_model,
            ensemble_tokenizer,
            device=str(device),
            max_context=args.lookback,
        )
        ensemble_predictor.model.eval()
        ensemble_predictor.tokenizer.eval()
    oracle = DifferentiableExposureValueOracle(ORACLE_CONFIG).to(device)
    execution_oracle = DifferentiableExposureValueOracle(
        EXECUTION_ORACLE_CONFIG
    ).to(device) if args.forecast_output is not None else None
    grid = oracle.grid.detach().cpu().numpy().astype(np.float64)
    levels = DEFAULT_QUANTILE_LEVELS
    estimator_names = (
        "ensembleMean", "projectedMean", "ensembleMedian", "kqspMedian"
    )
    unique: dict[str, ForecastAccumulator] = {
        name: ForecastAccumulator(grid.size) for name in estimator_names
    }
    per_window: dict[str, dict[str, ForecastAccumulator]] = {
        name: {
            window.id: ForecastAccumulator(grid.size) for window in windows
        }
        for name in estimator_names
    }
    probabilistic_unique = ProbabilisticAccumulator(levels)
    probabilistic_per_window: dict[str, ProbabilisticAccumulator] = {
        window.id: ProbabilisticAccumulator(levels) for window in windows
    }
    forecast_rows: list[dict] = []
    requested_batch_size = args.batch_size or max(
        1, spec.default_batch_size // args.sample_count
    )
    batch_size = requested_batch_size
    completed = 0
    previous_duration = 0.0
    if not args.no_resume:
        progress_state = read_evaluation_progress(
            progress_file,
            run_signature,
            model_id,
            len(origins),
        )
        if progress_state is not None:
            completed = int(progress_state["completedOrigins"])
            batch_size = int(progress_state["effectiveBatchSize"])
            previous_duration = float(progress_state["durationSeconds"])
            unique = {
                name: ForecastAccumulator.from_state(
                    progress_state["estimators"][name]["unique"], grid.size
                )
                for name in estimator_names
            }
            per_window = {
                name: {
                    window.id: ForecastAccumulator.from_state(
                        progress_state["estimators"][name]["windows"][window.id],
                        grid.size,
                    )
                    for window in windows
                }
                for name in estimator_names
            }
            probabilistic_unique = ProbabilisticAccumulator.from_state(
                progress_state["probabilistic"]["unique"], levels
            )
            probabilistic_per_window = {
                window.id: ProbabilisticAccumulator.from_state(
                    progress_state["probabilistic"]["windows"][window.id],
                    levels,
                )
                for window in windows
            }
            forecast_rows = list(progress_state.get("forecastRows", []))
            if args.forecast_output is not None \
                    and len(forecast_rows) != completed:
                raise ValueError(
                    "forecast progress row count does not match completed origins"
                )
            print(
                f"RESUME MODEL {model_id} at {completed}/{len(origins)} "
                f"from {progress_file}",
                flush=True,
            )
    started = time.monotonic()
    next_progress = (math.floor(completed / len(origins) / 0.05) + 1) * 0.05
    next_checkpoint = (
        time.monotonic() + args.progress_checkpoint_seconds
        if args.progress_checkpoint_seconds > 0 else math.inf
    )
    while completed < len(origins):
        end = min(len(origins), completed + batch_size)
        try:
            (
                normalized,
                context_stamps,
                target_stamps,
                means,
                stds,
                actual,
                anchors,
            ) = predictor_inputs(
                corpus, origins, completed, end, args.lookback
            )
            seed_inference_batch(args.seed, origins[completed:end])
            primary_sample_count = (
                args.sample_count - args.ensemble_sample_count
                if ensemble_predictor is not None else args.sample_count
            )
            paths = generate_sample_paths(
                predictor,
                normalized,
                context_stamps,
                target_stamps,
                means,
                stds,
                horizon=HORIZON,
                temperature=args.temperature,
                top_p=args.top_p,
                sample_count=primary_sample_count,
                inference_precision=args.inference_precision,
            )
            if ensemble_predictor is not None:
                ensemble_paths = generate_sample_paths(
                    ensemble_predictor,
                    normalized,
                    context_stamps,
                    target_stamps,
                    means,
                    stds,
                    horizon=HORIZON,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    sample_count=args.ensemble_sample_count,
                    inference_precision=args.inference_precision,
                )
                paths = np.concatenate((paths, ensemble_paths), axis=1)
            if not np.isfinite(paths).all() or np.any(paths[..., :4] <= 0):
                raise ValueError(
                    "Kronos emitted a non-positive or non-finite sample path"
                )
            raw_quantiles = empirical_ohlc_quantiles(paths, levels)
            repaired_quantiles = kqsp(raw_quantiles)
            predictions = point_estimators(paths, repaired_quantiles, levels)
            predicted_oracle, actual_oracle = oracle_path_mixture(
                paths,
                actual,
                anchors,
                oracle,
                device,
            )
            metrics = {
                name: metric_batch(
                    prediction,
                    actual,
                    anchors,
                    predicted_oracle,
                    actual_oracle,
                    grid,
                )
                for name, prediction in predictions.items()
            }
            probabilistic_metrics = probabilistic_batch(
                paths,
                raw_quantiles,
                repaired_quantiles,
                actual,
                anchors,
                levels,
            )
            if args.forecast_output is not None:
                assert execution_oracle is not None
                (
                    execution_predicted_oracle,
                    execution_expected_utility,
                ) = oracle_path_distributions(
                    paths,
                    anchors,
                    execution_oracle,
                    device,
                )
                forecast_rows.extend(causal_forecast_rows(
                    origins[completed:end],
                    paths,
                    predictions,
                    predicted_oracle,
                    execution_predicted_oracle,
                    execution_expected_utility,
                    anchors,
                ))
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(
                f"MODEL {model_id} CUDA OOM; retrying batch size {batch_size}",
                flush=True,
            )
            continue
        indexes = range(end - completed)
        for name in estimator_names:
            unique[name].add(metrics[name], indexes)
        probabilistic_unique.add(probabilistic_metrics, indexes)
        members: defaultdict[str, list[int]] = defaultdict(list)
        for local, origin in enumerate(origins[completed:end]):
            for window_id in origin.window_ids:
                members[window_id].append(local)
        for window_id, local_indexes in members.items():
            for name in estimator_names:
                per_window[name][window_id].add(
                    metrics[name], local_indexes
                )
            probabilistic_per_window[window_id].add(
                probabilistic_metrics, local_indexes
            )
        completed = end
        progress = completed / len(origins)
        if progress >= next_progress or completed == len(origins):
            elapsed = time.monotonic() - started
            print(
                f"MODEL {model_id} {completed}/{len(origins)} "
                f"({100 * progress:.1f}%) {completed / max(elapsed, 1e-9):.2f} origins/s",
                flush=True,
            )
            next_progress += 0.05
        if time.monotonic() >= next_checkpoint and completed < len(origins):
            duration_so_far = previous_duration + time.monotonic() - started
            write_evaluation_progress(
                progress_file,
                run_signature=run_signature,
                model_id=model_id,
                total_origins=len(origins),
                completed_origins=completed,
                effective_batch_size=batch_size,
                duration_seconds=duration_so_far,
                estimator_names=estimator_names,
                unique=unique,
                per_window=per_window,
                probabilistic_unique=probabilistic_unique,
                probabilistic_per_window=probabilistic_per_window,
                forecast_rows=forecast_rows,
            )
            print(
                f"PROGRESS CHECKPOINT MODEL {model_id} "
                f"{completed}/{len(origins)} {progress_file}",
                flush=True,
            )
            next_checkpoint = time.monotonic() + args.progress_checkpoint_seconds
    duration = previous_duration + time.monotonic() - started
    estimator_results = {}
    for name in estimator_names:
        window_results = []
        for window in windows:
            window_results.append({
                "id": window.id,
                "label": window.label,
                "group": window.group,
                "startTime": window.start_time,
                "endTime": window.end_time,
                "metrics": per_window[name][window.id].result(grid),
            })
        estimator_results[name] = {
            "uniqueOrigins": unique[name].result(grid),
            "macroWindow": macro_window_metrics(window_results),
            "windows": window_results,
        }
    probabilistic_windows = [
        {
            "id": window.id,
            "label": window.label,
            "group": window.group,
            "startTime": window.start_time,
            "endTime": window.end_time,
            "metrics": probabilistic_per_window[window.id].result(),
        }
        for window in windows
    ]
    primary = estimator_results[args.point_estimator]
    result = {
        "id": model_id,
        "model": {
            "repoId": spec.model_repo,
            "revision": spec.model_revision,
            "parameters": spec.parameters,
            "tokenizerRepoId": spec.tokenizer_repo,
            "tokenizerRevision": spec.tokenizer_revision,
            "publishedMaxContext": spec.max_context,
            "predictorCheckpoint": (
                str(model_path).replace("\\", "/")
                if args.model_checkpoint else None
            ),
            "predictorCheckpointFingerprint": (
                checkpoint_fingerprint(model_path)
                if args.model_checkpoint else None
            ),
            "tokenizerCheckpoint": (
                str(tokenizer_path).replace("\\", "/")
                if args.tokenizer_checkpoint else None
            ),
            "tokenizerCheckpointFingerprint": (
                checkpoint_fingerprint(tokenizer_path)
                if args.tokenizer_checkpoint else None
            ),
            "pathEnsemble": (
                {
                    "primarySampleCount": (
                        args.sample_count - args.ensemble_sample_count
                    ),
                    "secondarySampleCount": args.ensemble_sample_count,
                    "secondaryPredictorCheckpoint": (
                        "pretrained"
                        if args.ensemble_predictor_checkpoint == "pretrained"
                        else str(ensemble_model_path).replace("\\", "/")
                    ),
                    "secondaryPredictorFingerprint": checkpoint_fingerprint(
                        ensemble_model_path
                    ),
                    "secondaryTokenizerCheckpoint": str(
                        ensemble_tokenizer_path
                    ).replace("\\", "/"),
                    "secondaryTokenizerFingerprint": checkpoint_fingerprint(
                        ensemble_tokenizer_path
                    ),
                }
                if ensemble_predictor is not None else None
            ),
        },
        "runtime": {
            "device": str(device),
            "durationSeconds": duration,
            "originsPerSecond": len(origins) / duration,
            "requestedBatchSize": requested_batch_size,
            "effectiveBatchSize": batch_size,
            "effectiveFlattenedPathBatchSize": batch_size * args.sample_count,
            "peakCudaMemoryBytes": (
                int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else None
            ),
        },
        "primaryEstimator": args.point_estimator,
        "uniqueOrigins": primary["uniqueOrigins"],
        "macroWindow": primary["macroWindow"],
        "windows": primary["windows"],
        "estimators": estimator_results,
        "probabilistic": {
            "quantileLevels": levels.tolist(),
            "uniqueOrigins": probabilistic_unique.result(),
            "windows": probabilistic_windows,
        },
    }
    if args.forecast_output is not None:
        if len(forecast_rows) != len(origins):
            raise RuntimeError("causal forecast artifact is incomplete")
        result["_forecastRows"] = forecast_rows
    del (
        oracle,
        predictor,
        model,
        tokenizer,
        ensemble_predictor,
        ensemble_model,
        ensemble_tokenizer,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def macro_window_metrics(results: list[dict]) -> dict:
    paths = {
        "candleMseSkillVsPersistence": ("candle", "mseSkillVsPersistence"),
        "candleAnchoredLogCorrelation": ("candle", "anchoredLogCorrelation"),
        "closeReturnMseSkillVsZero": ("closeReturn", "mseSkillVsZero"),
        "closeReturnCorrelation": ("closeReturn", "correlation"),
        "horizonReturnCorrelation": ("closeReturn", "horizonCorrelation"),
        "horizonReturnRankCorrelation": (
            "closeReturn", "horizonRankCorrelation"
        ),
        "closeDirectionAccuracy": ("closeReturn", "directionAccuracy"),
        "priceSeriesIc": ("paperAligned", "priceSeriesIc"),
        "priceSeriesRankIc": ("paperAligned", "priceSeriesRankIc"),
        "oracleForwardKl": ("oracle", "forwardKl"),
        "oracleProbabilityMse": ("oracle", "probabilityMse"),
        "oracleModalActionAgreement": ("oracle", "modalActionAgreement"),
    }
    output = {}
    for label, (section, name) in paths.items():
        values = [
            result["metrics"][section][name]
            for result in results
            if result["metrics"][section][name] is not None
        ]
        output[label] = float(np.mean(values)) if values else None
    return output


def skill(error: float, baseline: float) -> float | None:
    return 1.0 - error / baseline if baseline > 0 else None


def iso_time(timestamp: int) -> str:
    return datetime.fromtimestamp(
        timestamp / 1_000, tz=timezone.utc
    ).isoformat()


def main() -> None:
    args = parse_args()
    if args.lookback < 1 or args.lookback > DEFAULT_LOOKBACK:
        raise ValueError(f"--lookback must be in [1, {DEFAULT_LOOKBACK}]")
    if args.sample_count < 1:
        raise ValueError("--sample-count must be positive")
    if args.ensemble_predictor_checkpoint is None:
        if args.ensemble_tokenizer_checkpoint is not None \
                or args.ensemble_sample_count is not None:
            raise ValueError(
                "ensemble tokenizer/sample count requires "
                "--ensemble-predictor-checkpoint"
            )
    else:
        if len(selected_specs(args.models)) != 1:
            raise ValueError("path ensembling requires exactly one --models value")
        if args.ensemble_sample_count is None:
            args.ensemble_sample_count = args.sample_count // 2
        if not 0 < args.ensemble_sample_count < args.sample_count:
            raise ValueError(
                "--ensemble-sample-count must be between zero and sample count"
            )
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_origins_per_window is not None \
            and args.max_origins_per_window < 1:
        raise ValueError("--max-origins-per-window must be positive")
    if args.forecast_output is not None and len(selected_specs(args.models)) != 1:
        raise ValueError("--forecast-output requires exactly one --models value")
    if not math.isfinite(args.progress_checkpoint_seconds) \
            or args.progress_checkpoint_seconds < 0:
        raise ValueError(
            "--progress-checkpoint-seconds must be finite and non-negative"
        )
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be positive and finite")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    repo_root = Path(__file__).resolve().parents[1]
    windows = parse_windows(args.windows_json)
    history_root = (
        args.history_dir if args.history_dir.is_absolute()
        else repo_root / args.history_dir
    ).resolve()
    output = (
        args.output if args.output.is_absolute() else repo_root / args.output
    ).resolve()
    if args.forecast_output is not None:
        args.forecast_output = (
            args.forecast_output
            if args.forecast_output.is_absolute()
            else repo_root / args.forecast_output
        ).resolve()
    specs = selected_specs(args.models)
    if (args.model_checkpoint or args.tokenizer_checkpoint) and len(specs) != 1:
        raise ValueError("local checkpoints require exactly one --models value")
    if args.model_label and not (args.model_checkpoint or args.tokenizer_checkpoint):
        raise ValueError("--model-label is only valid with a local checkpoint")
    for name in ("model_checkpoint", "tokenizer_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            resolved = value if value.is_absolute() else repo_root / value
            setattr(args, name, resolved.resolve())
    if args.ensemble_predictor_checkpoint is not None \
            and args.ensemble_predictor_checkpoint.lower() != "pretrained":
        ensemble_checkpoint = Path(args.ensemble_predictor_checkpoint)
        args.ensemble_predictor_checkpoint = (
            ensemble_checkpoint
            if ensemble_checkpoint.is_absolute()
            else repo_root / ensemble_checkpoint
        ).resolve()
    elif args.ensemble_predictor_checkpoint is not None:
        args.ensemble_predictor_checkpoint = "pretrained"
    if args.ensemble_tokenizer_checkpoint is not None:
        args.ensemble_tokenizer_checkpoint = (
            args.ensemble_tokenizer_checkpoint
            if args.ensemble_tokenizer_checkpoint.is_absolute()
            else repo_root / args.ensemble_tokenizer_checkpoint
        ).resolve()
    source = source_root(repo_root)
    if not source.is_dir():
        raise FileNotFoundError("Kronos source is missing; run npm run kronos:setup")
    actual_commit = subprocess_commit(source)
    if actual_commit != KRONOS_SOURCE_COMMIT:
        raise ValueError(
            f"Kronos source is {actual_commit}; expected {KRONOS_SOURCE_COMMIT}"
        )
    corpus = load_corpus(history_root, windows, args.lookback)
    origins = build_origins(
        corpus,
        windows,
        args.lookback,
        args.max_origins_per_window,
    )
    membership_count = sum(len(origin.window_ids) for origin in origins)
    print(
        f"BENCHMARK windows={len(windows)} uniqueOrigins={len(origins)} "
        f"windowOrigins={membership_count} horizon={HORIZON}x1m "
        f"lookback={args.lookback}",
        flush=True,
    )
    run_signature = benchmark_signature(
        windows=windows,
        corpus=corpus,
        origins=origins,
        args=args,
        specs=specs,
    )
    results = []
    for spec in specs:
        model_id = args.model_label or (
            f"{spec.id}-finetuned"
            if args.model_checkpoint or args.tokenizer_checkpoint
            else spec.id
        )
        partial_file = model_partial_file(output, model_id)
        progress_file = model_progress_file(output, model_id)
        partial = read_model_partial(partial_file, run_signature, model_id) \
            if not args.no_resume else None
        if partial is not None:
            print(f"RESUME MODEL {model_id} from {partial_file}", flush=True)
            results.append(partial)
            continue
        print(
            f"START MODEL {model_id} params={spec.parameters} "
            f"tokenizer={spec.tokenizer_repo}",
            flush=True,
        )
        result = evaluate_model(
            repo_root,
            spec,
            model_id,
            corpus,
            origins,
            windows,
            args,
            progress_file,
            run_signature,
        )
        forecast_rows = result.pop("_forecastRows", None)
        if args.forecast_output is not None:
            if forecast_rows is None:
                raise RuntimeError("model evaluation did not return forecast rows")
            atomic_json({
                "version": 2,
                "contract": FORECAST_CONTRACT,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "runSignature": run_signature,
                "modelId": model_id,
                "market": "Binance spot BTCUSDT",
                "intervalMs": STEP_MS,
                "lookbackCandles": args.lookback,
                "horizonCandles": HORIZON,
                "originStrideCandles": HORIZON,
                "temperature": args.temperature,
                "topP": args.top_p,
                "sampleCount": args.sample_count,
                "oracleGrid": np.linspace(-100, 100, 101).tolist(),
                "executionOracleGrid": np.linspace(-5, 5, 101).tolist(),
                "executionOracle": EXECUTION_ORACLE_CONFIG.__dict__,
                "executionOracleAggregations": {
                    "executionOracleProbabilities": (
                        "arithmetic mean of per-path oracle action probabilities"
                    ),
                    "executionUtilityProbabilities": (
                        "softmax of mean per-path log-wealth action values"
                    ),
                },
                "rows": forecast_rows,
            }, args.forecast_output)
            print(f"WROTE FORECASTS {args.forecast_output}", flush=True)
        results.append(result)
        atomic_json({
            "version": 2,
            "contract": "kronos-probabilistic-model-partial-v2",
            "runSignature": run_signature,
            "model": result,
        }, partial_file)
        progress_file.unlink(missing_ok=True)
        print(f"CHECKPOINT MODEL {model_id} {partial_file}", flush=True)
    report = {
        "version": 2,
        "contract": (
            "kronos-probabilistic-nonfit-inspector-windows-15x1m-v2"
        ),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "kronosSource": {
            "url": "https://github.com/shiyu-coder/Kronos",
            "commit": KRONOS_SOURCE_COMMIT,
        },
        "data": {
            "market": "Binance spot BTCUSDT",
            "interval": "1m",
            "historyRoot": str(history_root).replace("\\", "/"),
            "referenceCount": len(corpus.references),
            "corpusFingerprint": corpus.fingerprint,
            "referenceFiles": list(corpus.references),
            "windows": len(windows),
            "excludedWindows": [
                "fit-full", "fit-1", "fit-2", "fit-3", "fit-4", "latest-3m"
            ],
            "horizonCandles": HORIZON,
            "lookbackCandles": args.lookback,
            "originStrideCandles": HORIZON,
            "uniqueOrigins": len(origins),
            "windowOriginMemberships": membership_count,
            "targetPolicy": (
                "consecutive non-overlapping 15-candle blocks; context ends "
                "one minute before each target and may precede the inspector window"
            ),
        },
        "sampling": {
            "temperature": args.temperature,
            "topP": args.top_p,
            "sampleCount": args.sample_count,
            "seed": args.seed,
            "batchSeedPolicy": "sha256(seed,targetStartTimes)-v2",
            "retainedPaths": True,
            "pointEstimator": args.point_estimator,
            "evaluatedPointEstimators": [
                "ensembleMean", "projectedMean", "ensembleMedian", "kqspMedian"
            ],
            "quantileLevels": DEFAULT_QUANTILE_LEVELS.tolist(),
            "constraintRepair": "KQSP",
            "inferencePrecision": args.inference_precision,
        },
        "oracle": {
            "holdingPeriodCandles": ORACLE_CONFIG.holding_period_steps,
            "decisionDelayCandles": ORACLE_CONFIG.decision_delay_steps,
            "valueHorizonCandles": ORACLE_CONFIG.value_horizon_steps,
            "friction": ORACLE_CONFIG.friction,
            "grid": np.linspace(-100, 100, 101).tolist(),
            "temperature": ORACLE_CONFIG.temperature,
            "maxEffectiveExposure": ORACLE_CONFIG.max_effective_exposure,
            "quoteBorrowRatePerCandle": ORACLE_CONFIG.quote_borrow_rate,
            "assetBorrowRatePerCandle": ORACLE_CONFIG.asset_borrow_rate,
            "probabilityFloorForKl": PROBABILITY_FLOOR,
            "pathSource": (
                "mean oracle distribution across retained stochastic Kronos "
                "close-return paths; realized oracle from the actual path"
            ),
        },
        "models": results,
    }
    report["runSignature"] = run_signature
    atomic_json(report, output)
    print(f"WROTE {output}", flush=True)


def benchmark_signature(
    *,
    windows: tuple[Window, ...],
    corpus: Corpus,
    origins: tuple[Origin, ...],
    args: argparse.Namespace,
    specs: tuple[KronosModelSpec, ...],
) -> str:
    payload = {
        "contract": (
            "kronos-probabilistic-nonfit-inspector-windows-15x1m-v2"
        ),
        "sourceCommit": KRONOS_SOURCE_COMMIT,
        "corpusFingerprint": corpus.fingerprint,
        "windows": [window.__dict__ for window in windows],
        "uniqueOrigins": len(origins),
        "windowOriginMemberships": sum(len(origin.window_ids) for origin in origins),
        "lookback": args.lookback,
        "sampleCount": args.sample_count,
        "temperature": args.temperature,
        "topP": args.top_p,
        "pointEstimator": args.point_estimator,
        "quantileLevels": DEFAULT_QUANTILE_LEVELS.tolist(),
        "seed": args.seed,
        "batchSeedPolicy": "sha256(seed,targetStartTimes)-v2",
        "requestedBatchSize": args.batch_size,
        "maxOriginsPerWindow": args.max_origins_per_window,
        "models": [
            {
                "id": spec.id,
                "modelRepo": spec.model_repo,
                "modelRevision": spec.model_revision,
                "tokenizerRepo": spec.tokenizer_repo,
                "tokenizerRevision": spec.tokenizer_revision,
                "requestedBatchSize": (
                    args.batch_size or max(
                        1, spec.default_batch_size // args.sample_count
                    )
                ),
            }
            for spec in specs
        ],
        "modelCheckpointFingerprint": (
            checkpoint_fingerprint(args.model_checkpoint)
            if args.model_checkpoint else None
        ),
        "tokenizerCheckpointFingerprint": (
            checkpoint_fingerprint(args.tokenizer_checkpoint)
            if args.tokenizer_checkpoint else None
        ),
        "modelLabel": args.model_label,
        "oracle": ORACLE_CONFIG.__dict__,
    }
    if args.forecast_output is not None:
        payload["emitForecastRows"] = True
        payload["forecastContract"] = FORECAST_CONTRACT
        payload["executionOracle"] = EXECUTION_ORACLE_CONFIG.__dict__
    inference_precision = getattr(args, "inference_precision", "float32")
    if inference_precision != "float32":
        # Preserve compatibility with progress written before this optional
        # acceleration existed. Non-default numerical modes remain isolated.
        payload["inferencePrecision"] = inference_precision
    if args.ensemble_predictor_checkpoint is not None:
        payload.update({
            "ensemblePredictorCheckpoint": (
                args.ensemble_predictor_checkpoint
                if args.ensemble_predictor_checkpoint == "pretrained"
                else checkpoint_fingerprint(args.ensemble_predictor_checkpoint)
            ),
            "ensembleTokenizerCheckpointFingerprint": (
                checkpoint_fingerprint(args.ensemble_tokenizer_checkpoint)
                if args.ensemble_tokenizer_checkpoint else None
            ),
            "ensembleSampleCount": args.ensemble_sample_count,
        })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_partial_file(output: Path, model_id: str) -> Path:
    return output.with_name(f"{output.stem}.{model_id}.partial.json")


def model_progress_file(output: Path, model_id: str) -> Path:
    return output.with_name(f"{output.stem}.{model_id}.progress.json")


def read_model_partial(
    file: Path,
    run_signature: str,
    model_id: str,
) -> dict | None:
    if not file.is_file():
        return None
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    model = value.get("model")
    if value.get("contract") != "kronos-probabilistic-model-partial-v2" \
            or value.get("runSignature") != run_signature \
            or not isinstance(model, dict) \
            or model.get("id") != model_id:
        return None
    return model


def seed_inference_batch(seed: int, origins: tuple[Origin, ...]) -> int:
    payload = ",".join(
        (str(seed), *(str(origin.target_start) for origin in origins))
    ).encode("ascii")
    batch_seed = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "little"
    ) % (2**31)
    random.seed(batch_seed)
    np.random.seed(batch_seed)
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(batch_seed)
    return batch_seed


def write_evaluation_progress(
    file: Path,
    *,
    run_signature: str,
    model_id: str,
    total_origins: int,
    completed_origins: int,
    effective_batch_size: int,
    duration_seconds: float,
    estimator_names: tuple[str, ...],
    unique: dict[str, ForecastAccumulator],
    per_window: dict[str, dict[str, ForecastAccumulator]],
    probabilistic_unique: ProbabilisticAccumulator,
    probabilistic_per_window: dict[str, ProbabilisticAccumulator],
    forecast_rows: list[dict],
) -> None:
    atomic_json({
        "version": 1,
        "contract": "kronos-probabilistic-model-progress-v1",
        "runSignature": run_signature,
        "modelId": model_id,
        "totalOrigins": total_origins,
        "completedOrigins": completed_origins,
        "effectiveBatchSize": effective_batch_size,
        "durationSeconds": duration_seconds,
        "estimators": {
            name: {
                "unique": unique[name].state(),
                "windows": {
                    window_id: accumulator.state()
                    for window_id, accumulator in per_window[name].items()
                },
            }
            for name in estimator_names
        },
        "probabilistic": {
            "unique": probabilistic_unique.state(),
            "windows": {
                window_id: accumulator.state()
                for window_id, accumulator in probabilistic_per_window.items()
            },
        },
        "forecastRows": forecast_rows,
    }, file)


def read_evaluation_progress(
    file: Path,
    run_signature: str,
    model_id: str,
    total_origins: int,
) -> dict | None:
    if not file.is_file():
        return None
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) \
            or value.get("contract") \
            != "kronos-probabilistic-model-progress-v1" \
            or value.get("runSignature") != run_signature \
            or value.get("modelId") != model_id \
            or value.get("totalOrigins") != total_origins:
        return None
    completed = value.get("completedOrigins")
    batch_size = value.get("effectiveBatchSize")
    duration = value.get("durationSeconds")
    if not isinstance(completed, int) or not 0 < completed < total_origins \
            or not isinstance(batch_size, int) or batch_size < 1 \
            or not isinstance(duration, (int, float)) \
            or not math.isfinite(duration) or duration < 0:
        return None
    return value


def atomic_json(value: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    delay = 0.05
    for attempt in range(12):
        try:
            temporary.replace(output)
            return
        except PermissionError:
            if attempt == 11:
                raise
            # Windows denies replace while a reader has the destination open.
            # Progress monitors are short-lived, so bounded backoff preserves
            # atomic publication without discarding hours of resumable work.
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def subprocess_commit(source: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    ).stdout.strip()


def checkpoint_fingerprint(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors"):
        file = checkpoint / name
        if not file.is_file():
            raise FileNotFoundError(f"invalid Kronos checkpoint: {file}")
        digest.update(name.encode("utf-8"))
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
