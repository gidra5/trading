"""Leakage-safe train/validation audit of causal one-minute OHLCV features.

The oracle target at prediction time ``t`` uses future prices, but every input
here ends at or before ``t``.  In particular, a target sampled at
``hh:mm:00.999`` may use the one-minute candle that closed at
``hh:mm-1:59.999``; the candle starting at ``hh:mm`` is still open and is
excluded.  Held-out test reference metadata and payloads are never opened.

The audit is deliberately non-neural.  It asks whether train-fitted quantile
regimes expose enough validation information to justify putting OHLCV into a
long neural run.  Candidate grouping and Dirichlet shrinkage are selected on
the final chronological 20% of training data.  Validation is scored once for
the selected close-only, OHLCV-only, and joint candidates.
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
MINUTE_CLOSE_OFFSET_MS = 59_999
DAY_ROWS = 1_440
ACTION_COUNT = 101
CONTEXT_LENGTH = 3_601
FORECAST_HORIZON = 3_600
VALIDATION_DAYS = 30
TEST_DAYS = 30
CALIBRATION_FRACTION = 0.2
PRIOR_STRENGTHS = (16.0, 64.0, 256.0, 1_024.0)
V18_BEST_RAW_VALIDATION_KL = 0.9698349614329325


FEATURE_NAMES = (
    # Close-only controls, all reconstructible from the existing 1s closes.
    "return1m",
    "return5m",
    "return15m",
    "return60m",
    "rmsReturn5m",
    "rmsReturn15m",
    "rmsReturn60m",
    # Scale-free volume surprise/trend.  The 60m trend compares half-hours.
    "logVolume1mVs60m",
    "logVolume5mVs60m",
    "logVolume15mVs60m",
    "logVolumeTrend1m",
    "logVolumeTrend5m",
    "logVolumeTrend15m",
    "logVolumeTrend60m",
    # True intraminute range and wick/body geometry, absent from close-only
    # history.  Fractions use logarithmic price distances within each candle.
    "meanLogRange1m",
    "meanLogRange5m",
    "meanLogRange15m",
    "meanLogRange60m",
    "logRange5mVs60m",
    "logRange15mVs60m",
    "meanBodyFraction1m",
    "meanBodyFraction5m",
    "meanBodyFraction15m",
    "meanBodyFraction60m",
    "meanWickImbalance1m",
    "meanWickImbalance5m",
    "meanWickImbalance15m",
    "meanWickImbalance60m",
    "meanCloseLocation1m",
    "meanCloseLocation5m",
    "meanCloseLocation15m",
    "meanCloseLocation60m",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class RegimeSpec:
    family: str
    name: str
    features: tuple[str, ...]
    bin_counts: tuple[int, ...]

    @property
    def feature_indexes(self) -> tuple[int, ...]:
        return tuple(FEATURE_INDEX[name] for name in self.features)


# These are immutable audit hypotheses, not candidates created after looking at
# validation.  Each family gets its own train-only selection so the joint score
# can be compared with a fair close-only control.
REGIME_SPECS = (
    RegimeSpec(
        "close-only",
        "close-trend-vol",
        ("return15m", "return60m", "rmsReturn60m"),
        (6, 10, 8),
    ),
    RegimeSpec(
        "close-only",
        "close-multiscale",
        ("return5m", "return15m", "return60m", "rmsReturn60m"),
        (5, 5, 7, 6),
    ),
    RegimeSpec(
        "close-only",
        "close-fast-vol",
        ("return1m", "return5m", "return60m", "rmsReturn15m"),
        (5, 5, 7, 6),
    ),
    RegimeSpec(
        "ohlcv-only",
        "volume-surprise",
        (
            "logVolume1mVs60m",
            "logVolume5mVs60m",
            "logVolume15mVs60m",
            "logVolumeTrend60m",
        ),
        (8, 6, 5, 6),
    ),
    RegimeSpec(
        "ohlcv-only",
        "volume-trends",
        (
            "logVolumeTrend1m",
            "logVolumeTrend5m",
            "logVolumeTrend15m",
            "logVolumeTrend60m",
        ),
        (6, 6, 6, 6),
    ),
    RegimeSpec(
        "ohlcv-only",
        "geometry-level",
        (
            "meanLogRange1m",
            "meanLogRange15m",
            "meanLogRange60m",
            "meanBodyFraction15m",
            "meanWickImbalance15m",
        ),
        (6, 6, 7, 5, 5),
    ),
    RegimeSpec(
        "ohlcv-only",
        "geometry-shape",
        (
            "logRange5mVs60m",
            "logRange15mVs60m",
            "meanBodyFraction15m",
            "meanWickImbalance15m",
            "meanCloseLocation15m",
        ),
        (6, 6, 5, 5, 5),
    ),
    RegimeSpec(
        "ohlcv-only",
        "volume-geometry",
        (
            "logVolume5mVs60m",
            "logVolumeTrend60m",
            "logRange15mVs60m",
            "meanWickImbalance15m",
        ),
        (7, 6, 7, 6),
    ),
    RegimeSpec(
        "close-plus-ohlcv",
        "close-volume",
        (
            "return15m",
            "return60m",
            "rmsReturn60m",
            "logVolume5mVs60m",
            "logVolumeTrend60m",
        ),
        (5, 7, 5, 5, 5),
    ),
    RegimeSpec(
        "close-plus-ohlcv",
        "close-geometry",
        (
            "return15m",
            "return60m",
            "rmsReturn60m",
            "logRange15mVs60m",
            "meanWickImbalance15m",
        ),
        (5, 7, 5, 5, 5),
    ),
    RegimeSpec(
        "close-plus-ohlcv",
        "close-volume-range",
        (
            "return15m",
            "return60m",
            "rmsReturn60m",
            "logVolume5mVs60m",
            "logRange15mVs60m",
        ),
        (5, 7, 5, 5, 5),
    ),
    RegimeSpec(
        "close-plus-ohlcv",
        "close-ohlcv-compact",
        (
            "return60m",
            "rmsReturn60m",
            "logVolume5mVs60m",
            "logVolumeTrend60m",
            "logRange15mVs60m",
            "meanWickImbalance15m",
        ),
        (7, 5, 4, 4, 4, 4),
    ),
)
FAMILIES = ("close-only", "ohlcv-only", "close-plus-ohlcv")


@dataclass
class AccessLog:
    target_files: set[Path]
    candle_files: set[Path]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = (
        repo_root
        / "data/training/immutable/refs/oracle/1s"
        / "hindsight-bot-71391c44b323e044e6ab"
    )
    candle_root = (
        repo_root
        / "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"
    )
    target_files = sorted(target_root.glob("*.json"))
    if len(target_files) <= VALIDATION_DAYS + TEST_DAYS:
        raise ValueError("causal oracle corpus is too short")

    segments = split_and_purge(target_files)
    access = AccessLog(set(), set())
    candle_cache: dict[str, np.ndarray] = {}
    print(
        "Loading causal train/validation targets and completed 1m OHLCV; "
        "test references remain sealed.",
        file=sys.stderr,
        flush=True,
    )
    train_features, train_targets = load_split(
        "train",
        segments["train"],
        candle_root,
        candle_cache,
        access,
    )
    validation_features, validation_targets = load_split(
        "validation",
        segments["validation"],
        candle_root,
        candle_cache,
        access,
    )

    test_target_paths = {
        path.resolve() for path in target_files[-TEST_DAYS:]
    }
    test_day_names = {path.stem for path in target_files[-TEST_DAYS:]}
    if access.target_files & test_target_paths:
        raise RuntimeError("held-out test target payload was opened")
    if test_day_names & {path.stem for path in access.candle_files}:
        raise RuntimeError("held-out test candle reference was opened")

    prior = normalized_mean(train_targets)
    prior_kl = mean_kl_constant(validation_targets, prior)
    selected, calibration = select_each_family(
        train_features,
        train_targets,
    )
    stacked_selection = select_stacked_mixture(
        train_features,
        train_targets,
        selected["close-only"],
        selected["ohlcv-only"],
    )
    validation_scores = {}
    for family, (spec, strength) in selected.items():
        score = fit_full_and_score(
            spec,
            strength,
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            prior,
        )
        validation_scores[family] = {
            "name": spec.name,
            "features": list(spec.features),
            "binCounts": list(spec.bin_counts),
            "priorStrength": strength,
            **score,
            "absoluteKlReductionFromPrior": prior_kl - score["rawValidationKl"],
            "relativeKlReductionFromPrior": (
                (prior_kl - score["rawValidationKl"]) / prior_kl
            ),
            "klMinusV18": score["rawValidationKl"] - V18_BEST_RAW_VALIDATION_KL,
        }

    stacked_score = fit_full_stacked_mixture_and_score(
        selected["close-only"],
        selected["ohlcv-only"],
        float(stacked_selection["ohlcvWeight"]),
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        prior,
    )
    validation_scores["stacked-close-plus-ohlcv"] = {
        "closeSpec": selected["close-only"][0].name,
        "ohlcvSpec": selected["ohlcv-only"][0].name,
        "closeWeight": stacked_selection["closeWeight"],
        "ohlcvWeight": stacked_selection["ohlcvWeight"],
        **stacked_score,
        "absoluteKlReductionFromPrior": (
            prior_kl - stacked_score["rawValidationKl"]
        ),
        "relativeKlReductionFromPrior": (
            (prior_kl - stacked_score["rawValidationKl"]) / prior_kl
        ),
        "klMinusV18": (
            stacked_score["rawValidationKl"] - V18_BEST_RAW_VALIDATION_KL
        ),
    }

    close_kl = validation_scores["close-only"]["rawValidationKl"]
    joint_kl = validation_scores[
        "stacked-close-plus-ohlcv"
    ]["rawValidationKl"]
    result = {
        "schemaVersion": 1,
        "auditContract": {
            "target": "causal oracle 1h horizon / 1m delay / 1m hold / T=0.01",
            "input": "only 1m candles whose close timestamp is <= prediction t",
            "predictionTimestampWithinMinuteMs": 999,
            "minuteCandleCloseOffsetMs": MINUTE_CLOSE_OFFSET_MS,
            "rowZeroLatestInput": "previous UTC day 23:59 candle",
            "selection": "final chronological 20% of training only",
            "validationUse": "one final score per independently selected family",
            "gpuUsed": False,
        },
        "accessContract": {
            "trainTargetPayloadsOpened": len({
                segment.target_file.resolve() for segment in segments["train"]
            }),
            "validationTargetPayloadsOpened": len({
                segment.target_file.resolve()
                for segment in segments["validation"]
            }),
            "uniqueTrainOrValidationTargetPayloadsOpened": len(access.target_files),
            "uniqueCausalOneMinuteCandleReferencesOpened": len(access.candle_files),
            "testTargetReferenceMetadataOpened": 0,
            "testTargetPayloadsOpened": 0,
            "testCandleReferenceMetadataOpened": 0,
            "testCandlePayloadsOpened": 0,
        },
        "corpus": {
            "targetFilenameCount": len(target_files),
            "trainFileCount": len(target_files) - VALIDATION_DAYS - TEST_DAYS,
            "validationFileCount": VALIDATION_DAYS,
            "sealedTestFileCount": TEST_DAYS,
            "trainRowsAfterPurge": int(train_targets.shape[0]),
            "validationRowsAfterPurge": int(validation_targets.shape[0]),
            "featureCount": len(FEATURE_NAMES),
            "trainDateStart": target_files[0].stem,
            "trainDateEnd": target_files[-VALIDATION_DAYS - TEST_DAYS - 1].stem,
            "validationDateStart": target_files[-VALIDATION_DAYS - TEST_DAYS].stem,
            "validationDateEnd": target_files[-TEST_DAYS - 1].stem,
        },
        "trainPriorRawValidationKl": prior_kl,
        "v18BestRawValidationKl": V18_BEST_RAW_VALIDATION_KL,
        "trainOnlySelection": calibration,
        "trainOnlyStackedSelection": stacked_selection,
        "selectedFamilyValidation": validation_scores,
        "incrementalOhlcv": {
            "comparison": "train-selected convex close/OHLCV backoff vs close-only",
            "jointKlMinusCloseOnlyKl": joint_kl - close_kl,
            "absoluteKlReductionFromCloseOnly": close_kl - joint_kl,
            "relativeKlReductionFromCloseOnly": (close_kl - joint_kl) / close_kl,
            "beatsCloseOnly": joint_kl < close_kl,
            "beatsV18": joint_kl < V18_BEST_RAW_VALIDATION_KL,
            "materiallyBeatsV18ByAtLeast0.002": (
                joint_kl <= V18_BEST_RAW_VALIDATION_KL - 0.002
            ),
        },
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
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        day_start = utc_day_start_ms(target_file.stem)
        raw[split].append(CausalSegment(
            split=split,
            prediction_time_start=day_start + SECOND_MS - 1,
            count=DAY_ROWS,
            target_file=target_file,
            target_row_offset=0,
            step_ms=MINUTE_MS,
        ))
    return purge_cross_split_windows(raw, CONTEXT_LENGTH, FORECAST_HORIZON)


def utc_day_start_ms(day_value: str) -> int:
    return int(datetime.combine(
        date.fromisoformat(day_value),
        datetime.min.time(),
        timezone.utc,
    ).timestamp() * SECOND_MS)


def latest_closed_candle_start_ms(
    prediction_time_ms: int,
    *,
    candle_step_ms: int = MINUTE_MS,
    close_offset_ms: int = MINUTE_CLOSE_OFFSET_MS,
) -> int:
    """Start timestamp of the newest candle fully closed by prediction time."""
    if candle_step_ms < 1 or not 0 <= close_offset_ms < candle_step_ms:
        raise ValueError("invalid candle timing contract")
    return (
        (prediction_time_ms - close_offset_ms) // candle_step_ms
    ) * candle_step_ms


def load_split(
    split: str,
    segments: list[CausalSegment],
    candle_root: Path,
    candle_cache: dict[str, np.ndarray],
    access: AccessLog,
) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("the OHLCV audit may only load train or validation")
    if any(segment.split != split for segment in segments):
        raise ValueError("segment split does not match requested audit split")
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for index, segment in enumerate(segments, start=1):
        day_value = segment.target_file.stem
        features = daily_causal_ohlcv_features(
            candle_root,
            day_value,
            candle_cache,
            access.candle_files,
        )
        target_file = segment.target_file.resolve()
        _shard, targets = read_shard_array(
            target_file,
            "<f4",
            (DAY_ROWS, ACTION_COUNT),
        )
        access.target_files.add(target_file)
        targets = np.asarray(targets, dtype=np.float32)
        start = segment.target_row_offset
        end = start + segment.count
        feature_parts.append(features[start:end])
        target_parts.append(targets[start:end])
        if index % 25 == 0 or index == len(segments):
            print(
                f"{split}: {index}/{len(segments)} days loaded",
                file=sys.stderr,
                flush=True,
            )
    features = np.concatenate(feature_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    if features.shape != (targets.shape[0], len(FEATURE_NAMES)):
        raise RuntimeError(f"{split} OHLCV features are misaligned")
    if not np.isfinite(features).all() \
            or not np.isfinite(targets).all() \
            or bool((targets < 0).any()) \
            or not np.allclose(
                targets.sum(axis=1),
                1,
                atol=2e-4,
                rtol=2e-4,
            ):
        raise ValueError(f"{split} contains invalid values")
    return features, targets


def read_minute_day(
    candle_root: Path,
    day_value: str,
    cache: dict[str, np.ndarray],
    opened: set[Path],
) -> np.ndarray:
    cached = cache.get(day_value)
    if cached is not None:
        return cached
    reference = (candle_root / f"{day_value}.json").resolve()
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    opened.add(reference)
    sequence = manifest.get("sequence", {})
    constants = manifest.get("layout", {}).get("constants", {})
    if sequence != {
        "start": utc_day_start_ms(day_value),
        "step": MINUTE_MS,
        "count": DAY_ROWS,
        "unit": "unix-ms",
    }:
        raise ValueError(f"non-canonical one-minute sequence: {reference}")
    if constants.get("interval") != "1m" \
            or constants.get("closeTimeOffsetMs") != MINUTE_CLOSE_OFFSET_MS \
            or constants.get("closed") is not True:
        raise ValueError(f"non-canonical one-minute close timing: {reference}")
    values = np.column_stack(tuple(
        read_candle_column(reference, column)
        for column in ("open", "high", "low", "close", "volume")
    )).astype(np.float64, copy=False)
    if values.shape != (DAY_ROWS, 5) \
            or not np.isfinite(values).all() \
            or bool((values[:, :4] <= 0).any()) \
            or bool((values[:, 4] < 0).any()) \
            or bool((values[:, 1] < values[:, [0, 3]].max(axis=1)).any()) \
            or bool((values[:, 2] > values[:, [0, 3]].min(axis=1)).any()):
        raise ValueError(f"invalid one-minute OHLCV: {reference}")
    cache[day_value] = values
    return values


def completed_candle_windows(
    previous_day: np.ndarray,
    current_day: np.ndarray,
    length: int = 60,
) -> np.ndarray:
    """Return one strictly completed-candle window per target minute."""
    if previous_day.shape != (DAY_ROWS, 5) \
            or current_day.shape != (DAY_ROWS, 5) \
            or not 1 <= length <= DAY_ROWS:
        raise ValueError("invalid daily OHLCV arrays or history length")
    combined = np.concatenate((previous_day[-length:], current_day), axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(
        combined,
        length,
        axis=0,
    )
    # NumPy places the window axis last: [row, column, history].  The final
    # possible window includes the current day's last (still-open) candle for
    # the last target row, so exactly the first DAY_ROWS windows are retained.
    return np.moveaxis(windows[:DAY_ROWS], -1, 1)


def completed_close_windows(
    previous_day: np.ndarray,
    current_day: np.ndarray,
    transition_count: int = 60,
) -> np.ndarray:
    if previous_day.shape != (DAY_ROWS, 5) \
            or current_day.shape != (DAY_ROWS, 5) \
            or not 1 <= transition_count < DAY_ROWS:
        raise ValueError("invalid daily OHLCV arrays or transition count")
    length = transition_count + 1
    combined = np.concatenate((
        previous_day[-length:, 3],
        current_day[:, 3],
    ))
    return np.lib.stride_tricks.sliding_window_view(
        combined,
        length,
    )[:DAY_ROWS]


def daily_causal_ohlcv_features(
    candle_root: Path,
    day_value: str,
    cache: dict[str, np.ndarray] | None = None,
    opened: set[Path] | None = None,
) -> np.ndarray:
    if cache is None:
        cache = {}
    if opened is None:
        opened = set()
    current_date = date.fromisoformat(day_value)
    previous_value = (current_date - timedelta(days=1)).isoformat()
    previous = read_minute_day(candle_root, previous_value, cache, opened)
    current = read_minute_day(candle_root, day_value, cache, opened)
    candles = completed_candle_windows(previous, current)
    closes = completed_close_windows(previous, current)
    features = causal_ohlcv_features(candles, closes)

    day_start = utc_day_start_ms(day_value)
    first_prediction = day_start + SECOND_MS - 1
    last_prediction = first_prediction + (DAY_ROWS - 1) * MINUTE_MS
    if latest_closed_candle_start_ms(first_prediction) \
            != day_start - MINUTE_MS \
            or latest_closed_candle_start_ms(last_prediction) \
            != day_start + (DAY_ROWS - 2) * MINUTE_MS:
        raise RuntimeError("completed-minute timestamp alignment failed")
    return features


def causal_ohlcv_features(
    candle_windows: np.ndarray,
    close_windows: np.ndarray,
) -> np.ndarray:
    if candle_windows.ndim != 3 \
            or candle_windows.shape[1:] != (60, 5) \
            or close_windows.shape != (candle_windows.shape[0], 61):
        raise ValueError("expected [rows,60,5] candles and [rows,61] closes")
    opens = candle_windows[:, :, 0]
    highs = candle_windows[:, :, 1]
    lows = candle_windows[:, :, 2]
    closes = candle_windows[:, :, 3]
    volumes = candle_windows[:, :, 4]
    if bool((opens <= 0).any()) \
            or bool((highs <= 0).any()) \
            or bool((lows <= 0).any()) \
            or bool((closes <= 0).any()) \
            or bool((volumes < 0).any()):
        raise ValueError("OHLCV windows contain invalid values")

    returns = np.diff(np.log(close_windows), axis=1)
    volume60 = volumes.mean(axis=1)
    volume_floor = np.maximum(volume60 * 1e-9, np.finfo(np.float64).tiny)

    log_ranges = np.log(highs / lows)
    log_bodies = np.log(closes / opens)
    upper_wicks = np.log(highs / np.maximum(opens, closes))
    lower_wicks = np.log(np.minimum(opens, closes) / lows)
    range_floor = np.maximum(log_ranges, np.finfo(np.float64).eps)
    body_fraction = np.clip(log_bodies / range_floor, -1, 1)
    wick_imbalance = np.clip(
        (lower_wicks - upper_wicks) / range_floor,
        -1,
        1,
    )
    close_location = np.clip(
        (np.log(closes / lows) - np.log(highs / closes)) / range_floor,
        -1,
        1,
    )

    def trailing_mean(values: np.ndarray, length: int) -> np.ndarray:
        return values[:, -length:].mean(axis=1)

    def log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        floor = np.maximum(
            denominator * 1e-9,
            np.finfo(np.float64).tiny,
        )
        # Subtract logs rather than divide first: a transition from an all-zero
        # window to non-zero activity is large but finite and must not overflow.
        return (
            np.log(np.maximum(numerator + floor, np.finfo(np.float64).tiny))
            - np.log(np.maximum(
                denominator + floor,
                np.finfo(np.float64).tiny,
            ))
        )

    volume1 = trailing_mean(volumes, 1)
    volume5 = trailing_mean(volumes, 5)
    volume15 = trailing_mean(volumes, 15)
    volume_trend1 = log_ratio(volumes[:, -1], volumes[:, -2])
    volume_trend5 = log_ratio(
        trailing_mean(volumes, 5),
        volumes[:, -10:-5].mean(axis=1),
    )
    volume_trend15 = log_ratio(
        trailing_mean(volumes, 15),
        volumes[:, -30:-15].mean(axis=1),
    )
    volume_trend60 = log_ratio(
        volumes[:, -30:].mean(axis=1),
        volumes[:, :30].mean(axis=1),
    )
    # Keep exact zero-volume histories finite while retaining multiplicative
    # scale invariance everywhere the market traded.
    volume_surprises = tuple(
        np.log(np.maximum(
            value + volume_floor,
            np.finfo(np.float64).tiny,
        ))
        - np.log(np.maximum(
            volume60 + volume_floor,
            np.finfo(np.float64).tiny,
        ))
        for value in (volume1, volume5, volume15)
    )

    range1 = trailing_mean(log_ranges, 1)
    range5 = trailing_mean(log_ranges, 5)
    range15 = trailing_mean(log_ranges, 15)
    range60 = trailing_mean(log_ranges, 60)
    features = np.column_stack((
        returns[:, -1],
        returns[:, -5:].sum(axis=1),
        returns[:, -15:].sum(axis=1),
        returns.sum(axis=1),
        np.sqrt(np.mean(np.square(returns[:, -5:]), axis=1)),
        np.sqrt(np.mean(np.square(returns[:, -15:]), axis=1)),
        np.sqrt(np.mean(np.square(returns), axis=1)),
        *volume_surprises,
        volume_trend1,
        volume_trend5,
        volume_trend15,
        volume_trend60,
        range1,
        range5,
        range15,
        range60,
        log_ratio(range5, range60),
        log_ratio(range15, range60),
        trailing_mean(body_fraction, 1),
        trailing_mean(body_fraction, 5),
        trailing_mean(body_fraction, 15),
        trailing_mean(body_fraction, 60),
        trailing_mean(wick_imbalance, 1),
        trailing_mean(wick_imbalance, 5),
        trailing_mean(wick_imbalance, 15),
        trailing_mean(wick_imbalance, 60),
        trailing_mean(close_location, 1),
        trailing_mean(close_location, 5),
        trailing_mean(close_location, 15),
        trailing_mean(close_location, 60),
    )).astype(np.float32, copy=False)
    if features.shape != (candle_windows.shape[0], len(FEATURE_NAMES)) \
            or not np.isfinite(features).all():
        raise ValueError("derived OHLCV features are invalid")
    return features


def normalized_mean(probabilities: np.ndarray) -> np.ndarray:
    result = probabilities.mean(axis=0, dtype=np.float64)
    result /= result.sum()
    return result


def mean_kl_constant(targets: np.ndarray, prediction: np.ndarray) -> float:
    return mean_kl_indexed(
        targets,
        np.zeros(targets.shape[0], dtype=np.int64),
        prediction[None, :],
    )


def mean_kl_indexed(
    targets: np.ndarray,
    ids: np.ndarray,
    table: np.ndarray,
    *,
    batch_size: int = 8_192,
) -> float:
    if targets.ndim != 2 \
            or ids.shape != (targets.shape[0],) \
            or table.ndim != 2 \
            or table.shape[1] != targets.shape[1]:
        raise ValueError("KL arrays are incompatible")
    total = 0.0
    tiny = np.finfo(np.float64).tiny
    for start in range(0, targets.shape[0], batch_size):
        end = min(start + batch_size, targets.shape[0])
        target = targets[start:end].astype(np.float64, copy=False)
        prediction = np.clip(table[ids[start:end]], tiny, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                target > 0,
                target * (np.log(target) - np.log(prediction)),
                0,
            )
        total += terms.sum(dtype=np.float64)
    result = total / targets.shape[0]
    return 0.0 if -1e-6 < result < 0 else float(result)


def mean_kl_stacked_indexed(
    targets: np.ndarray,
    first_ids: np.ndarray,
    first_table: np.ndarray,
    second_ids: np.ndarray,
    second_table: np.ndarray,
    second_weight: float,
    *,
    batch_size: int = 8_192,
) -> float:
    if not 0 <= second_weight <= 1 \
            or first_ids.shape != (targets.shape[0],) \
            or second_ids.shape != (targets.shape[0],) \
            or first_table.shape[1:] != (targets.shape[1],) \
            or second_table.shape[1:] != (targets.shape[1],):
        raise ValueError("stacked KL arrays or weight are incompatible")
    total = 0.0
    tiny = np.finfo(np.float64).tiny
    for start in range(0, targets.shape[0], batch_size):
        end = min(start + batch_size, targets.shape[0])
        target = targets[start:end].astype(np.float64, copy=False)
        prediction = (
            (1 - second_weight) * first_table[first_ids[start:end]]
            + second_weight * second_table[second_ids[start:end]]
        )
        prediction = np.clip(prediction, tiny, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                target > 0,
                target * (np.log(target) - np.log(prediction)),
                0,
            )
        total += terms.sum(dtype=np.float64)
    result = total / targets.shape[0]
    return 0.0 if -1e-6 < result < 0 else float(result)


def quantile_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    edges = np.quantile(
        values.astype(np.float64, copy=False),
        np.linspace(0, 1, bin_count + 1)[1:-1],
    )
    return np.unique(edges)


def fit_edges(
    features: np.ndarray,
    spec: RegimeSpec,
) -> tuple[np.ndarray, ...]:
    return tuple(
        quantile_edges(features[:, index], count)
        for index, count in zip(
            spec.feature_indexes,
            spec.bin_counts,
            strict=True,
        )
    )


def cell_ids(
    features: np.ndarray,
    spec: RegimeSpec,
    edges: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, int]:
    result = np.zeros(features.shape[0], dtype=np.int64)
    multiplier = 1
    for index, current_edges in zip(
        spec.feature_indexes,
        edges,
        strict=True,
    ):
        bins = np.searchsorted(
            current_edges,
            features[:, index],
            side="right",
        )
        result += bins * multiplier
        multiplier *= current_edges.size + 1
    return result, multiplier


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


def select_each_family(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[
    dict[str, tuple[RegimeSpec, float]],
    dict[str, object],
]:
    fit_end = int(features.shape[0] * (1 - CALIBRATION_FRACTION))
    fit_features = features[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_features = features[fit_end:]
    calibration_targets = targets[fit_end:]
    fit_prior = normalized_mean(fit_targets)
    candidates: list[dict[str, object]] = []
    best: dict[str, tuple[float, RegimeSpec, float]] = {}
    for spec in REGIME_SPECS:
        edges = fit_edges(fit_features, spec)
        fit_ids, cell_count = cell_ids(fit_features, spec, edges)
        calibration_ids, _ = cell_ids(calibration_features, spec, edges)
        sums, counts = sufficient_table(fit_targets, fit_ids, cell_count)
        scores: dict[str, float] = {}
        for strength in PRIOR_STRENGTHS:
            table = smoothed_table(sums, counts, fit_prior, strength)
            score = mean_kl_indexed(
                calibration_targets,
                calibration_ids,
                table,
            )
            scores[str(int(strength))] = score
            candidate = (score, spec, strength)
            if spec.family not in best or candidate[0] < best[spec.family][0]:
                best[spec.family] = candidate
        candidates.append({
            "family": spec.family,
            "name": spec.name,
            "features": list(spec.features),
            "requestedBinCounts": list(spec.bin_counts),
            "effectiveCells": cell_count,
            "populatedFitCells": int(np.count_nonzero(counts)),
            "calibrationKlByPriorStrength": scores,
        })
    if set(best) != set(FAMILIES):
        raise RuntimeError("a regime family had no candidate")
    selected = {
        family: (value[1], value[2])
        for family, value in best.items()
    }
    report = {
        "fitRows": fit_end,
        "calibrationRows": features.shape[0] - fit_end,
        "calibrationFraction": CALIBRATION_FRACTION,
        "fitPriorUsedForCalibration": True,
        "validationUsedForSelection": False,
        "candidates": candidates,
        "selected": {
            family: {
                "name": value[1].name,
                "priorStrength": value[2],
                "calibrationKl": value[0],
            }
            for family, value in best.items()
        },
    }
    return selected, report


def fitted_regime(
    spec: RegimeSpec,
    prior_strength: float,
    features: np.ndarray,
    targets: np.ndarray,
    prior: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    edges = fit_edges(features, spec)
    ids, cell_count = cell_ids(features, spec, edges)
    sums, counts = sufficient_table(targets, ids, cell_count)
    return edges, smoothed_table(
        sums,
        counts,
        prior,
        prior_strength,
    ), counts


def select_stacked_mixture(
    features: np.ndarray,
    targets: np.ndarray,
    close_selected: tuple[RegimeSpec, float],
    ohlcv_selected: tuple[RegimeSpec, float],
) -> dict[str, float | int | str]:
    """Select an OHLCV backoff weight without consulting validation."""
    fit_end = int(features.shape[0] * (1 - CALIBRATION_FRACTION))
    fit_features = features[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_features = features[fit_end:]
    calibration_targets = targets[fit_end:]
    prior = normalized_mean(fit_targets)
    close_spec, close_strength = close_selected
    ohlcv_spec, ohlcv_strength = ohlcv_selected
    close_edges, close_table, _ = fitted_regime(
        close_spec,
        close_strength,
        fit_features,
        fit_targets,
        prior,
    )
    ohlcv_edges, ohlcv_table, _ = fitted_regime(
        ohlcv_spec,
        ohlcv_strength,
        fit_features,
        fit_targets,
        prior,
    )
    close_ids, _ = cell_ids(
        calibration_features,
        close_spec,
        close_edges,
    )
    ohlcv_ids, _ = cell_ids(
        calibration_features,
        ohlcv_spec,
        ohlcv_edges,
    )

    def objective(weight: float) -> float:
        return mean_kl_stacked_indexed(
            calibration_targets,
            close_ids,
            close_table,
            ohlcv_ids,
            ohlcv_table,
            weight,
        )

    low = 0.0
    high = 1.0
    ratio = (math.sqrt(5) - 1) / 2
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_score = objective(left)
    right_score = objective(right)
    for _ in range(32):
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
    middle = (low + high) / 2
    ohlcv_weight, calibration_kl = min(
        (0.0, objective(0.0)),
        (middle, objective(middle)),
        (1.0, objective(1.0)),
        key=lambda item: item[1],
    )
    return {
        "fitRows": fit_end,
        "calibrationRows": targets.shape[0] - fit_end,
        "closeSpec": close_spec.name,
        "ohlcvSpec": ohlcv_spec.name,
        "closeWeight": 1 - ohlcv_weight,
        "ohlcvWeight": ohlcv_weight,
        "calibrationKl": calibration_kl,
        "validationUsedForSelection": False,
    }


def fit_full_stacked_mixture_and_score(
    close_selected: tuple[RegimeSpec, float],
    ohlcv_selected: tuple[RegimeSpec, float],
    ohlcv_weight: float,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    train_prior: np.ndarray,
) -> dict[str, float | int]:
    close_spec, close_strength = close_selected
    ohlcv_spec, ohlcv_strength = ohlcv_selected
    close_edges, close_table, close_counts = fitted_regime(
        close_spec,
        close_strength,
        train_features,
        train_targets,
        train_prior,
    )
    ohlcv_edges, ohlcv_table, ohlcv_counts = fitted_regime(
        ohlcv_spec,
        ohlcv_strength,
        train_features,
        train_targets,
        train_prior,
    )
    close_ids, _ = cell_ids(
        validation_features,
        close_spec,
        close_edges,
    )
    ohlcv_ids, _ = cell_ids(
        validation_features,
        ohlcv_spec,
        ohlcv_edges,
    )
    return {
        "rawValidationKl": mean_kl_stacked_indexed(
            validation_targets,
            close_ids,
            close_table,
            ohlcv_ids,
            ohlcv_table,
            ohlcv_weight,
        ),
        "validationRowsWithEmptyCloseCell": int(np.count_nonzero(
            close_counts[close_ids] == 0,
        )),
        "validationRowsWithEmptyOhlcvCell": int(np.count_nonzero(
            ohlcv_counts[ohlcv_ids] == 0,
        )),
    }


def fit_full_and_score(
    spec: RegimeSpec,
    prior_strength: float,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    train_prior: np.ndarray,
) -> dict[str, float | int]:
    edges = fit_edges(train_features, spec)
    train_ids, cell_count = cell_ids(train_features, spec, edges)
    validation_ids, _ = cell_ids(validation_features, spec, edges)
    sums, counts = sufficient_table(train_targets, train_ids, cell_count)
    prediction = smoothed_table(
        sums,
        counts,
        train_prior,
        prior_strength,
    )
    return {
        "rawValidationKl": mean_kl_indexed(
            validation_targets,
            validation_ids,
            prediction,
        ),
        "effectiveCells": cell_count,
        "populatedTrainCells": int(np.count_nonzero(counts)),
        "validationRowsInEmptyTrainCells": int(np.count_nonzero(
            counts[validation_ids] == 0,
        )),
    }


if __name__ == "__main__":
    main()
