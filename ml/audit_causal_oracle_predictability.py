"""Read-only train/validation predictability audit for the causal oracle.

The held-out test references and payloads are deliberately never opened.  Test
filenames participate only in the existing chronological split/purge boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

from trading_storage import read_candle_column, read_shard_array
from train_joint_price_oracle import CausalSegment, purge_cross_split_windows


SECOND_MS = 1_000
MINUTE_MS = 60_000
DAY_ROWS = 1_440
DAY_SECONDS = 86_400
ACTION_COUNT = 101
CONTEXT_LENGTH = 3_601
FORECAST_HORIZON = 3_600
SOURCE_TEMPERATURE = 0.01
TEMPERATURES = (0.01, 0.02, 0.04)
VALIDATION_DAYS = 30
TEST_DAYS = 30
CALIBRATION_FRACTION = 0.2
FEATURE_NAMES = (
    "return1m",
    "return5m",
    "return15m",
    "return1h",
    "rmsVol5m",
    "rmsVol15m",
    "rmsVol1h",
    "ma60Path",
    "ma15MinusMa60",
    "ma5MinusMa15",
    "pathMinusMa5",
    "deltaMa60Path",
    "deltaMa15MinusMa60",
    "deltaMa5MinusMa15",
    "deltaPathMinusMa5",
)


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    feature_indexes: tuple[int, ...]
    bin_counts: tuple[int, ...]


REGIME_SPECS = (
    RegimeSpec("returns", (0, 1, 2, 3), (5, 5, 5, 7)),
    RegimeSpec("returns-plus-vol", (0, 1, 2, 3, 6), (5, 5, 5, 7, 5)),
    RegimeSpec(
        "returns-plus-vol-plus-fast-ma",
        (0, 1, 2, 3, 6, 10),
        (5, 5, 5, 7, 5, 5),
    ),
    RegimeSpec("trend-vol", (3, 6, 9, 10), (12, 8, 6, 6)),
)
PRIOR_STRENGTHS = (16.0, 64.0, 256.0)
LAGGED_ORACLE_CALIBRATION_FRACTION = 0.2
V18_BEST_RAW_VALIDATION_KL = 0.9698349614329325


@dataclass(frozen=True)
class LaggedTargetSplit:
    probabilities: np.ndarray
    valid: np.ndarray
    opened_target_files: frozenset[Path]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = (
        repo_root
        / "data/training/immutable/refs/oracle/1s"
        / "hindsight-bot-71391c44b323e044e6ab"
    )
    candle_root = (
        repo_root
        / "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"
    )
    target_files = sorted(target_root.glob("*.json"))
    if len(target_files) <= VALIDATION_DAYS + TEST_DAYS:
        raise ValueError("causal oracle corpus is too short")
    segments = split_and_purge(target_files)
    print(
        "Loading causal train/validation targets and summaries; test payloads "
        "remain sealed.",
        file=sys.stderr,
        flush=True,
    )
    target_cache: dict[Path, np.ndarray] = {}
    opened_target_files: set[Path] = set()
    train_features, train_targets = load_split(
        "train",
        segments["train"],
        candle_root,
        target_cache=target_cache,
        opened_target_files=opened_target_files,
    )
    validation_features, validation_targets = load_split(
        "validation",
        segments["validation"],
        candle_root,
        target_cache=target_cache,
        opened_target_files=opened_target_files,
    )
    accessible_target_files = {
        target_file.stem: target_file
        for target_file in target_files[:-TEST_DAYS]
    }
    safe_lag_minutes = minimum_safe_oracle_lag_minutes(
        FORECAST_HORIZON,
        SECOND_MS,
        MINUTE_MS,
    )
    train_lagged = load_lagged_target_split(
        segments["train"],
        accessible_target_files,
        safe_lag_minutes,
        target_cache=target_cache,
    )
    validation_lagged = load_lagged_target_split(
        segments["validation"],
        accessible_target_files,
        safe_lag_minutes,
        target_cache=target_cache,
    )
    lagged_oracle = lagged_oracle_audit(
        train_targets,
        train_lagged,
        validation_targets,
        validation_lagged,
        safe_lag_minutes=safe_lag_minutes,
    )
    opened_target_files.update(train_lagged.opened_target_files)
    opened_target_files.update(validation_lagged.opened_target_files)
    del train_lagged, validation_lagged, target_cache

    temperature_report: dict[str, dict[str, float]] = {}
    validation_targets_by_temperature: dict[float, np.ndarray] = {}
    train_prior_by_temperature: dict[float, np.ndarray] = {}
    for temperature in TEMPERATURES:
        train_target = retemper(train_targets, temperature)
        validation_target = retemper(validation_targets, temperature)
        train_prior = normalized_mean(train_target)
        temperature_report[temperature_key(temperature)] = {
            "trainTargetEntropyNats": mean_entropy(train_target),
            "validationTargetEntropyNats": mean_entropy(validation_target),
            "trainTargetEntropyFractionOfLogActions": (
                mean_entropy(train_target) / math.log(ACTION_COUNT)
            ),
            "validationTargetEntropyFractionOfLogActions": (
                mean_entropy(validation_target) / math.log(ACTION_COUNT)
            ),
            "trainPriorEntropyNats": entropy(train_prior),
            "validationKlFromTrainPrior": mean_kl(
                validation_target,
                np.broadcast_to(train_prior, validation_target.shape),
            ),
        }
        validation_targets_by_temperature[temperature] = validation_target
        train_prior_by_temperature[temperature] = train_prior

    raw_prior = train_prior_by_temperature[SOURCE_TEMPERATURE]
    univariate = univariate_audit(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        raw_prior,
    )
    selected, calibration = select_regime(
        train_features,
        train_targets,
    )
    transfer = transfer_audit(
        selected,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        validation_targets_by_temperature,
        train_prior_by_temperature,
    )

    raw_prior_kl = temperature_report["0.01"]["validationKlFromTrainPrior"]
    selected_raw_kl = transfer["0.01"]["raw01KlDirect"]
    result = {
        "schemaVersion": 1,
        "accessContract": {
            "trainTargetPayloadsOpened": len({
                segment.target_file for segment in segments["train"]
            }),
            "validationTargetPayloadsOpened": len({
                segment.target_file for segment in segments["validation"]
            }),
            "testReferencesOpened": 0,
            "testPayloadsOpened": 0,
            "gpuUsed": False,
            "uniqueTrainOrValidationTargetPayloadsOpened": len(
                opened_target_files
            ),
        },
        "corpus": {
            "targetFileCount": len(target_files),
            "trainFileCount": len(target_files) - VALIDATION_DAYS - TEST_DAYS,
            "validationFileCount": VALIDATION_DAYS,
            "heldoutTestFileCount": TEST_DAYS,
            "trainRowsAfterPurge": int(train_targets.shape[0]),
            "validationRowsAfterPurge": int(validation_targets.shape[0]),
            "actionCount": ACTION_COUNT,
            "trainDateStart": target_files[0].stem,
            "trainDateEnd": target_files[-VALIDATION_DAYS - TEST_DAYS - 1].stem,
            "validationDateStart": target_files[-VALIDATION_DAYS - TEST_DAYS].stem,
            "validationDateEnd": target_files[-TEST_DAYS - 1].stem,
        },
        "temperatures": temperature_report,
        "univariateTrainFittedBinsAtRaw01": univariate,
        "trainOnlyRegimeSelection": calibration,
        "selectedRegime": {
            "name": selected[0].name,
            "features": [FEATURE_NAMES[index] for index in selected[0].feature_indexes],
            "binCounts": list(selected[0].bin_counts),
            "priorStrength": selected[1],
            "rawValidationKl": selected_raw_kl,
            "absoluteKlReductionFromPrior": raw_prior_kl - selected_raw_kl,
            "relativeKlReductionFromPrior": (
                (raw_prior_kl - selected_raw_kl) / raw_prior_kl
            ),
        },
        "temperatureTransfer": transfer,
        "temperatureFacts": target_temperature_facts(
            validation_targets,
            validation_targets_by_temperature,
        ),
        "laggedOracle": lagged_oracle,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


def split_and_purge(
    target_files: list[Path],
) -> dict[str, list[CausalSegment]]:
    train_end = len(target_files) - VALIDATION_DAYS - TEST_DAYS
    validation_end = len(target_files) - TEST_DAYS
    raw = {split: [] for split in ("train", "validation", "test")}
    for index, target_file in enumerate(target_files):
        split = (
            "train" if index < train_end
            else "validation" if index < validation_end
            else "test"
        )
        day_start = int(datetime.combine(
            date.fromisoformat(target_file.stem),
            datetime.min.time(),
            timezone.utc,
        ).timestamp() * SECOND_MS)
        raw[split].append(CausalSegment(
            split=split,
            prediction_time_start=day_start + SECOND_MS - 1,
            count=DAY_ROWS,
            target_file=target_file,
            target_row_offset=0,
            step_ms=MINUTE_MS,
        ))
    return purge_cross_split_windows(
        raw,
        CONTEXT_LENGTH,
        FORECAST_HORIZON,
    )


def load_split(
    split: str,
    segments: list[CausalSegment],
    candle_root: Path,
    *,
    target_cache: dict[Path, np.ndarray] | None = None,
    opened_target_files: set[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    feature_cache: dict[str, np.ndarray] = {}
    for index, segment in enumerate(segments, start=1):
        day = segment.target_file.stem
        features = feature_cache.get(day)
        if features is None:
            features = daily_causal_features(candle_root, day)
            feature_cache[day] = features
        targets = read_target_day(
            segment.target_file,
            target_cache,
            opened_target_files,
        )
        start = segment.target_row_offset
        end = start + segment.count
        feature_parts.append(features[start:end])
        target_parts.append(np.asarray(targets[start:end], dtype=np.float32))
        if index % 25 == 0 or index == len(segments):
            print(
                f"{split}: {index}/{len(segments)} target days loaded",
                file=sys.stderr,
                flush=True,
            )
    features = np.concatenate(feature_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    if features.shape != (targets.shape[0], len(FEATURE_NAMES)):
        raise RuntimeError(f"{split} causal features are misaligned")
    if not np.isfinite(features).all() \
            or not np.isfinite(targets).all() \
            or bool((targets < 0).any()) \
            or not np.allclose(targets.sum(axis=1), 1, atol=2e-4, rtol=2e-4):
        raise ValueError(f"{split} corpus contains invalid values")
    return features, targets


def read_target_day(
    target_file: Path,
    cache: dict[Path, np.ndarray] | None = None,
    opened_target_files: set[Path] | None = None,
) -> np.ndarray:
    target_file = target_file.resolve()
    if cache is not None and target_file in cache:
        if opened_target_files is not None:
            opened_target_files.add(target_file)
        return cache[target_file]
    _shard, targets = read_shard_array(
        target_file,
        "<f4",
        (DAY_ROWS, ACTION_COUNT),
    )
    targets = np.asarray(targets, dtype=np.float32)
    if not np.isfinite(targets).all() \
            or bool((targets < 0).any()) \
            or not np.allclose(
                targets.sum(axis=1),
                1,
                atol=2e-4,
                rtol=2e-4,
            ):
        raise ValueError(f"oracle target contains invalid values: {target_file}")
    if cache is not None:
        cache[target_file] = targets
    if opened_target_files is not None:
        opened_target_files.add(target_file)
    return targets


def minimum_safe_oracle_lag_minutes(
    value_horizon_steps: int,
    interval_ms: int,
    target_step_ms: int,
) -> int:
    """Earliest target lag whose complete oracle price window is observable."""
    if min(value_horizon_steps, interval_ms, target_step_ms) < 1:
        raise ValueError("oracle horizon and intervals must be positive")
    horizon_ms = value_horizon_steps * interval_ms
    return math.ceil(horizon_ms / target_step_ms)


def load_lagged_target_split(
    segments: list[CausalSegment],
    target_files_by_day: dict[str, Path],
    lag_minutes: int,
    *,
    target_cache: dict[Path, np.ndarray] | None = None,
) -> LaggedTargetSplit:
    minimum_lag = minimum_safe_oracle_lag_minutes(
        FORECAST_HORIZON,
        SECOND_MS,
        MINUTE_MS,
    )
    if lag_minutes < minimum_lag:
        raise ValueError(
            f"lagged oracle target is not causal: {lag_minutes}m < "
            f"{minimum_lag}m"
        )
    if not segments:
        raise ValueError("lagged oracle split is empty")
    parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    opened: set[Path] = set()
    for segment in segments:
        if segment.step_ms != MINUTE_MS:
            raise ValueError("lagged oracle targets require one-minute rows")
        target_day = date.fromisoformat(segment.target_file.stem)
        target_rows = (
            segment.target_row_offset
            + np.arange(segment.count, dtype=np.int64)
        )
        source_rows_unwrapped = target_rows - lag_minutes
        source_day_offsets = np.floor_divide(
            source_rows_unwrapped,
            DAY_ROWS,
        )
        source_rows = np.mod(source_rows_unwrapped, DAY_ROWS)
        probabilities = np.zeros(
            (segment.count, ACTION_COUNT),
            dtype=np.float32,
        )
        valid = np.zeros(segment.count, dtype=np.bool_)
        for day_offset in np.unique(source_day_offsets):
            positions = source_day_offsets == day_offset
            source_day = (
                target_day + timedelta(days=int(day_offset))
            ).isoformat()
            source_file = target_files_by_day.get(source_day)
            if source_file is None:
                continue
            source = read_target_day(
                source_file,
                target_cache,
                opened,
            )
            probabilities[positions] = source[source_rows[positions]]
            valid[positions] = True
        parts.append(probabilities)
        valid_parts.append(valid)
    return LaggedTargetSplit(
        probabilities=np.concatenate(parts, axis=0),
        valid=np.concatenate(valid_parts),
        opened_target_files=frozenset(opened),
    )


def lagged_oracle_audit(
    train_targets: np.ndarray,
    train_lagged: LaggedTargetSplit,
    validation_targets: np.ndarray,
    validation_lagged: LaggedTargetSplit,
    *,
    safe_lag_minutes: int,
) -> dict[str, object]:
    if train_lagged.probabilities.shape != train_targets.shape \
            or validation_lagged.probabilities.shape \
            != validation_targets.shape:
        raise ValueError("lagged targets are not aligned with current targets")
    if not bool(validation_lagged.valid.all()):
        raise ValueError(
            "the full validation split does not have the minimum safe lag"
        )
    fit_end = int(
        train_targets.shape[0]
        * (1 - LAGGED_ORACLE_CALIBRATION_FRACTION)
    )
    calibration_valid = train_lagged.valid.copy()
    calibration_valid[:fit_end] = False
    fit_prior = normalized_mean(train_targets[:fit_end])
    mixture_weight, calibration_kl = select_convex_mixture_weight(
        train_targets[calibration_valid],
        train_lagged.probabilities[calibration_valid],
        fit_prior,
    )
    full_train_prior = normalized_mean(train_targets)
    prior_prediction = np.broadcast_to(
        full_train_prior,
        validation_targets.shape,
    )
    lagged_prediction = validation_lagged.probabilities
    mixture_prediction = (
        (1 - mixture_weight) * full_train_prior[None, :]
        + mixture_weight * lagged_prediction
    )
    prior_kl = mean_kl(validation_targets, prior_prediction)
    direct_kl = mean_kl(validation_targets, lagged_prediction)
    mixture_kl = mean_kl(validation_targets, mixture_prediction)
    return {
        "availabilityProof": {
            "oracleTargetAtSUsesClosesThrough": (
                "s + valueHorizonSteps * intervalMs"
            ),
            "valueHorizonSteps": FORECAST_HORIZON,
            "intervalMs": SECOND_MS,
            "targetStepMs": MINUTE_MS,
            "minimumSafeLagMinutes": safe_lag_minutes,
            "decisionDelayAndHoldAddNoExtraLag": True,
        },
        "trainRows": int(train_targets.shape[0]),
        "trainRowsWithAvailableLag": int(train_lagged.valid.sum()),
        "trainRowsWithoutStoredPredecessor": int(
            (~train_lagged.valid).sum()
        ),
        "validationRows": int(validation_targets.shape[0]),
        "validationRowsWithAvailableLag": int(
            validation_lagged.valid.sum()
        ),
        "trainInternalSelection": {
            "fitRows": fit_end,
            "calibrationRowsWithAvailableLag": int(
                calibration_valid.sum()
            ),
            "lagWeight": mixture_weight,
            "trainPriorWeight": 1 - mixture_weight,
            "calibrationKl": calibration_kl,
        },
        "raw01Validation": {
            "trainPriorKl": prior_kl,
            "directLaggedOracleKl": direct_kl,
            "trainSelectedPriorMixtureKl": mixture_kl,
            "absoluteMixtureReductionFromPrior": prior_kl - mixture_kl,
            "relativeMixtureReductionFromPrior": (
                (prior_kl - mixture_kl) / prior_kl
            ),
            "v18BestKl": V18_BEST_RAW_VALIDATION_KL,
            "mixtureKlMinusV18": mixture_kl - V18_BEST_RAW_VALIDATION_KL,
            "materiallyBeatsV18": mixture_kl \
                < V18_BEST_RAW_VALIDATION_KL - 0.01,
        },
        "sourceTargetFilesOpened": {
            "train": len(train_lagged.opened_target_files),
            "validation": len(validation_lagged.opened_target_files),
            "testReferences": 0,
            "testPayloads": 0,
        },
    }


def select_convex_mixture_weight(
    targets: np.ndarray,
    lagged: np.ndarray,
    prior: np.ndarray,
) -> tuple[float, float]:
    if targets.shape != lagged.shape \
            or targets.ndim != 2 \
            or prior.shape != (targets.shape[1],) \
            or targets.shape[0] < 1:
        raise ValueError("mixture calibration arrays are incompatible")

    def objective(weight: float) -> float:
        prediction = (
            (1 - weight) * prior[None, :]
            + weight * lagged
        )
        return mean_kl(targets, prediction)

    low = 0.0
    high = 1.0
    ratio = (math.sqrt(5) - 1) / 2
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_score = objective(left)
    right_score = objective(right)
    for _ in range(36):
        if left_score <= right_score:
            high = right
            right = left
            right_score = left_score
            left = high - ratio * (high - low)
            left_score = objective(left)
        else:
            low = left
            left = right
            left_score = right_score
            right = low + ratio * (high - low)
            right_score = objective(right)
    candidates = (
        (0.0, objective(0.0)),
        ((low + high) / 2, objective((low + high) / 2)),
        (1.0, objective(1.0)),
    )
    return min(candidates, key=lambda item: item[1])


def daily_causal_features(candle_root: Path, day_value: str) -> np.ndarray:
    current_day = date.fromisoformat(day_value)
    previous_day = (current_day - timedelta(days=1)).isoformat()
    previous = read_candle_column(candle_root / f"{previous_day}.json", "close")
    current = read_candle_column(candle_root / f"{day_value}.json", "close")
    if previous.shape != (DAY_SECONDS,) or current.shape != (DAY_SECONDS,):
        raise ValueError(f"invalid candle length around {day_value}")
    boundary_closes = np.concatenate((
        previous[DAY_SECONDS - 3_600::60],
        current[::60],
    )).astype(np.float64, copy=False)
    if boundary_closes.shape != (DAY_ROWS + 60,) \
            or bool((boundary_closes <= 0).any()):
        raise ValueError(f"invalid minute boundaries around {day_value}")
    minute_returns = np.diff(np.log(boundary_closes))
    windows = np.lib.stride_tricks.sliding_window_view(minute_returns, 60)
    if windows.shape != (DAY_ROWS, 60):
        raise RuntimeError("minute return windows are misaligned")
    path = windows.cumsum(axis=1)
    bands = final_path_bands(path)
    previous_bands = final_path_bands(path[:, :-1])
    return np.column_stack((
        windows[:, -1],
        windows[:, -5:].sum(axis=1),
        windows[:, -15:].sum(axis=1),
        windows.sum(axis=1),
        np.sqrt(np.mean(np.square(windows[:, -5:]), axis=1)),
        np.sqrt(np.mean(np.square(windows[:, -15:]), axis=1)),
        np.sqrt(np.mean(np.square(windows), axis=1)),
        bands,
        bands - previous_bands,
    )).astype(np.float32, copy=False)


def final_path_bands(path: np.ndarray) -> np.ndarray:
    slow = path.mean(axis=1)
    middle = path[:, -min(15, path.shape[1]):].mean(axis=1)
    fast = path[:, -min(5, path.shape[1]):].mean(axis=1)
    return np.column_stack((
        slow,
        middle - slow,
        fast - middle,
        path[:, -1] - fast,
    ))


def retemper(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if math.isclose(temperature, SOURCE_TEMPERATURE, rel_tol=0, abs_tol=1e-15):
        result = probabilities.copy()
        result /= result.sum(axis=1, keepdims=True)
        return result
    exponent = SOURCE_TEMPERATURE / temperature
    with np.errstate(divide="ignore"):
        logits = np.where(probabilities > 0, np.log(probabilities) * exponent, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits).astype(np.float32, copy=False)
    result /= result.sum(axis=1, keepdims=True)
    return result


def sharpen(probabilities: np.ndarray, source_temperature: float) -> np.ndarray:
    exponent = source_temperature / SOURCE_TEMPERATURE
    logits = np.log(np.clip(probabilities, np.finfo(np.float32).tiny, None))
    logits *= exponent
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits).astype(np.float32, copy=False)
    result /= result.sum(axis=1, keepdims=True)
    return result


def normalized_mean(probabilities: np.ndarray) -> np.ndarray:
    result = probabilities.mean(axis=0, dtype=np.float64)
    result /= result.sum()
    return result


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities > 0
    return float(-(probabilities[positive] * np.log(probabilities[positive])).sum())


def mean_entropy(probabilities: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0)
    return float(-terms.sum(dtype=np.float64) / probabilities.shape[0])


def mean_kl(target: np.ndarray, prediction: np.ndarray) -> float:
    prediction = np.clip(prediction, np.finfo(np.float64).tiny, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        target_log = np.where(target > 0, np.log(target), 0)
        terms = np.where(target > 0, target * (target_log - np.log(prediction)), 0)
    result = float(terms.sum(dtype=np.float64) / target.shape[0])
    if -1e-6 < result < 0:
        return 0.0
    return result


def quantile_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    edges = np.quantile(
        values.astype(np.float64, copy=False),
        np.linspace(0, 1, bin_count + 1)[1:-1],
    )
    return np.unique(edges)


def cell_ids(
    features: np.ndarray,
    spec: RegimeSpec,
    edges: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, int]:
    result = np.zeros(features.shape[0], dtype=np.int64)
    multiplier = 1
    for index, current_edges in zip(spec.feature_indexes, edges, strict=True):
        bins = np.searchsorted(current_edges, features[:, index], side="right")
        result += bins * multiplier
        multiplier *= current_edges.size + 1
    return result, multiplier


def fit_edges(features: np.ndarray, spec: RegimeSpec) -> tuple[np.ndarray, ...]:
    return tuple(
        quantile_edges(features[:, index], count)
        for index, count in zip(spec.feature_indexes, spec.bin_counts, strict=True)
    )


def sufficient_table(
    targets: np.ndarray,
    ids: np.ndarray,
    cell_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(ids, kind="stable")
    ordered_ids = ids[order]
    starts = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1,
    ))
    populated = ordered_ids[starts]
    sums = np.zeros((cell_count, targets.shape[1]), dtype=np.float64)
    sums[populated] = np.add.reduceat(
        targets[order],
        starts,
        axis=0,
        dtype=np.float64,
    )
    counts = np.bincount(ids, minlength=cell_count).astype(np.float64)
    return sums, counts


def smoothed_table(
    sums: np.ndarray,
    counts: np.ndarray,
    prior: np.ndarray,
    prior_strength: float,
) -> np.ndarray:
    return (
        sums + prior_strength * prior[None, :]
    ) / (counts[:, None] + prior_strength)


def univariate_audit(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    prior: np.ndarray,
) -> list[dict[str, float | str | int]]:
    prior_prediction = np.broadcast_to(prior, validation_targets.shape)
    prior_kl = mean_kl(validation_targets, prior_prediction)
    result: list[dict[str, float | str | int]] = []
    for feature_index, name in enumerate(FEATURE_NAMES):
        spec = RegimeSpec(name, (feature_index,), (16,))
        edges = fit_edges(train_features, spec)
        train_ids, cell_count = cell_ids(train_features, spec, edges)
        validation_ids, _ = cell_ids(validation_features, spec, edges)
        sums, counts = sufficient_table(train_targets, train_ids, cell_count)
        prediction = smoothed_table(sums, counts, prior, 64.0)[validation_ids]
        kl = mean_kl(validation_targets, prediction)
        result.append({
            "feature": name,
            "effectiveBins": cell_count,
            "validationKl": kl,
            "absoluteKlReductionFromPrior": prior_kl - kl,
            "relativeKlReductionFromPrior": (prior_kl - kl) / prior_kl,
        })
    result.sort(key=lambda item: float(item["validationKl"]))
    return result


def select_regime(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[RegimeSpec, float], list[dict[str, object]]]:
    calibration_count = int(features.shape[0] * CALIBRATION_FRACTION)
    fit_end = features.shape[0] - calibration_count
    fit_features = features[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_features = features[fit_end:]
    calibration_targets = targets[fit_end:]
    fit_prior = normalized_mean(fit_targets)
    report: list[dict[str, object]] = []
    best: tuple[float, RegimeSpec, float] | None = None
    for spec in REGIME_SPECS:
        edges = fit_edges(fit_features, spec)
        fit_ids, cell_count = cell_ids(fit_features, spec, edges)
        calibration_ids, _ = cell_ids(calibration_features, spec, edges)
        sums, counts = sufficient_table(fit_targets, fit_ids, cell_count)
        scores: dict[str, float] = {}
        for prior_strength in PRIOR_STRENGTHS:
            prediction = smoothed_table(
                sums,
                counts,
                fit_prior,
                prior_strength,
            )[calibration_ids]
            score = mean_kl(calibration_targets, prediction)
            scores[str(int(prior_strength))] = score
            candidate = (score, spec, prior_strength)
            if best is None or candidate[0] < best[0]:
                best = candidate
        report.append({
            "name": spec.name,
            "features": [FEATURE_NAMES[index] for index in spec.feature_indexes],
            "requestedBinCounts": list(spec.bin_counts),
            "effectiveCells": cell_count,
            "calibrationKlByPriorStrength": scores,
        })
    if best is None:
        raise RuntimeError("no causal regime candidate was evaluated")
    report.append({
        "selection": {
            "name": best[1].name,
            "priorStrength": best[2],
            "calibrationKl": best[0],
            "fitRows": fit_end,
            "calibrationRows": calibration_count,
            "fullTrainPriorUsedOnlyAfterSelection": True,
        }
    })
    return (best[1], best[2]), report


def transfer_audit(
    selected: tuple[RegimeSpec, float],
    train_features: np.ndarray,
    raw_train_targets: np.ndarray,
    validation_features: np.ndarray,
    raw_validation_targets: np.ndarray,
    validation_targets_by_temperature: dict[float, np.ndarray],
    prior_by_temperature: dict[float, np.ndarray],
) -> dict[str, dict[str, float]]:
    spec, prior_strength = selected
    edges = fit_edges(train_features, spec)
    train_ids, cell_count = cell_ids(train_features, spec, edges)
    validation_ids, _ = cell_ids(validation_features, spec, edges)
    result: dict[str, dict[str, float]] = {}
    for temperature in TEMPERATURES:
        train_targets = retemper(raw_train_targets, temperature)
        validation_targets = validation_targets_by_temperature[temperature]
        sums, counts = sufficient_table(train_targets, train_ids, cell_count)
        prediction = smoothed_table(
            sums,
            counts,
            prior_by_temperature[temperature],
            prior_strength,
        )[validation_ids]
        sharpened = (
            prediction
            if temperature == SOURCE_TEMPERATURE
            else sharpen(prediction, temperature)
        )
        result[temperature_key(temperature)] = {
            "sameTemperatureValidationKl": mean_kl(
                validation_targets,
                prediction,
            ),
            "raw01KlDirect": mean_kl(raw_validation_targets, prediction),
            "raw01KlAfterAnalyticSharpening": mean_kl(
                raw_validation_targets,
                sharpened,
            ),
        }
    return result


def target_temperature_facts(
    raw_validation_targets: np.ndarray,
    targets_by_temperature: dict[float, np.ndarray],
) -> dict[str, dict[str, float]]:
    raw_modes = raw_validation_targets.argmax(axis=1)
    result: dict[str, dict[str, float]] = {}
    for temperature in TEMPERATURES[1:]:
        softened = targets_by_temperature[temperature]
        recovered = sharpen(softened, temperature)
        result[temperature_key(temperature)] = {
            "raw01KlToPerfectSoftenedTarget": mean_kl(
                raw_validation_targets,
                softened,
            ),
            "raw01KlAfterExactTargetSharpening": mean_kl(
                raw_validation_targets,
                recovered,
            ),
            "modalActionAgreement": float(np.mean(
                raw_modes == softened.argmax(axis=1)
            )),
            "meanPositiveSupportSize": float(np.mean(
                np.count_nonzero(softened > 0, axis=1)
            )),
        }
    return result


def temperature_key(temperature: float) -> str:
    return f"{temperature:.2f}"


if __name__ == "__main__":
    main()
