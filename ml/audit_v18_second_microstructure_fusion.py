"""Leakage-safe residual audit of causal 1s OHLCV microstructure beyond v18.

The frozen v18 encoder sees one-second closes, while the earlier OHLCV audit
only saw completed one-minute aggregates.  This audit derives information that
those aggregates discard: within-minute path inefficiency, variance/range
ratio, sign changes, extreme ordering, volume concentration, temporal volume
imbalance, a signed-volume proxy, and summed one-second range.  Each is
summarized over completed 1/5/15/60-minute windows; three features from the
current fully closed one-second candle are also causal at target time t.

Three fixed paired regimes compare completed-minute aggregate controls with a
joint control+microstructure table.  Regime, Dirichlet shrinkage, and a convex
joint/control backoff are selected on a chronological training holdout only.
One fusion scalar is then selected on the first validation half and scored on
the untouched second half.  Test reference JSON and payloads are never opened.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from audit_causal_oracle_ohlcv_predictability import (
    ACTION_COUNT,
    DAY_ROWS,
    FEATURE_NAMES as AGGREGATE_FEATURE_NAMES,
    causal_ohlcv_features,
    completed_candle_windows,
    completed_close_windows,
    mean_kl_indexed,
    mean_kl_stacked_indexed,
    normalized_mean,
    smoothed_table,
    split_and_purge,
    sufficient_table,
    utc_day_start_ms,
)
from audit_v18_calendar_fusion import (
    convex_probability_fusion,
    select_scalar,
    softmax,
    split_metrics,
    validate_preserved_checkpoint_without_test_access,
)
from audit_v18_ohlcv_fusion import (
    AuditedMarketCloseCache,
    AuditedOracleTargetCache,
    fusion_metrics,
    log_ratio_feature_fusion,
    validation_timestamps,
)
from audit_v18_on_v28_validation_rows import collect_sequence_core_rows
from evaluate_joint_price_oracle_actions import resolve_device
from trading_storage import (
    load_torch_checkpoint,
    read_candle_column,
    read_shard_array,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    CausalOracleDataset,
    CausalSegment,
    build_model,
    resolve,
    resolve_training_config,
    validate_plan,
)


DEFAULT_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
)
SECOND_MS = 1_000
SECOND_ROWS = 86_400
MINUTE_MS = 60_000
MINUTE_SECONDS = 60
HORIZONS = (1, 5, 15, 60)
MICRO_METRIC_NAMES = (
    "pathInefficiency",
    "varianceRangeRatio",
    "signFlipRate",
    "extremeOrder",
    "volumeHhi",
    "volumeHalfImbalance",
    "signedVolume",
    "microRangeShare",
)
MICRO_FEATURE_NAMES = tuple(
    f"{name}{horizon}m"
    for name in MICRO_METRIC_NAMES
    for horizon in HORIZONS
) + (
    "currentSecondRangeFraction",
    "currentSecondVolumeSurprise",
    "currentSecondSignedVolume",
)
FEATURE_NAMES = AGGREGATE_FEATURE_NAMES + MICRO_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
PRIOR_STRENGTHS = (64.0, 256.0, 1_024.0)
CALIBRATION_FRACTION = 0.2
EXPECTED_CHECKPOINT_EPOCH = 7
CLEAN_GATE = 0.002


@dataclass(frozen=True)
class PairedRegime:
    name: str
    control_features: tuple[str, ...]
    micro_features: tuple[str, ...]
    control_bins: tuple[int, ...]
    micro_bins: tuple[int, ...]

    @property
    def joint_features(self) -> tuple[str, ...]:
        return self.control_features + self.micro_features

    @property
    def joint_bins(self) -> tuple[int, ...]:
        return self.control_bins + self.micro_bins


REGIMES = (
    PairedRegime(
        "path-dynamics",
        ("return15m", "return60m", "rmsReturn60m"),
        ("pathInefficiency5m", "signFlipRate15m", "extremeOrder60m"),
        (5, 7, 5),
        (3, 3, 3),
    ),
    PairedRegime(
        "range-concentration",
        (
            "return60m",
            "rmsReturn60m",
            "logRange15mVs60m",
            "logVolume15mVs60m",
        ),
        (
            "varianceRangeRatio15m",
            "microRangeShare60m",
            "volumeHhi5m",
        ),
        (5, 5, 4, 4),
        (3, 3, 3),
    ),
    PairedRegime(
        "signed-volume",
        (
            "return5m",
            "return60m",
            "rmsReturn60m",
            "logVolume15mVs60m",
        ),
        (
            "volumeHalfImbalance5m",
            "signedVolume15m",
            "volumeHhi60m",
        ),
        (4, 5, 5, 4),
        (3, 3, 3),
    ),
    PairedRegime(
        "current-second",
        ("return1m", "return15m", "rmsReturn15m", "meanLogRange1m"),
        (
            "currentSecondRangeFraction",
            "currentSecondVolumeSurprise",
            "currentSecondSignedVolume",
        ),
        (4, 5, 4, 4),
        (3, 3, 3),
    ),
)


@dataclass
class AccessLog:
    target_files: set[Path]
    candle_files: set[Path]


class SecondDayCache:
    def __init__(
        self,
        root: Path,
        opened: set[Path],
        maximum_days: int = 3,
    ) -> None:
        self.root = root
        self.opened = opened
        self.maximum_days = max(2, int(maximum_days))
        self.days: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, day_value: str) -> np.ndarray:
        cached = self.days.pop(day_value, None)
        if cached is not None:
            self.days[day_value] = cached
            return cached
        values = read_second_day(self.root, day_value, self.opened)
        self.days[day_value] = values
        while len(self.days) > self.maximum_days:
            self.days.popitem(last=False)
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = audit_v18_second_microstructure_fusion(
        arguments.plan,
        requested_device=arguments.device,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


def audit_v18_second_microstructure_fusion(
    plan_file: Path,
    *,
    requested_device: str = "auto",
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    storage = training_storage_layout(repo_root)
    resolved_plan = resolve(repo_root, plan_file).resolve()
    plan = json.loads(resolved_plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    if plan.get("testPolicy", "sealed-never-load") != "sealed-never-load":
        raise ValueError("microstructure audit requires sealed-never-load")
    model_config = plan["model"]
    training = plan["training"]
    resolved_training = resolve_training_config(training)
    if not bool(resolved_training.get("policyOnly", False)):
        raise ValueError("microstructure audit requires policy-only v18")
    target_root = require_under(
        resolve(repo_root, Path(plan["targetReferenceDir"])),
        storage.immutable / "refs" / "oracle",
        "targetReferenceDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        storage.runs,
        "runDir",
    )
    target_files = sorted(target_root.glob("*.json"))
    segments = split_and_purge(target_files)
    counts = {
        split: sum(segment.count for segment in segments[split])
        for split in ("train", "validation", "test")
    }
    test_target_files = {
        segment.target_file.resolve() for segment in segments["test"]
    }
    test_days = {segment.target_file.stem for segment in segments["test"]}

    access = AccessLog(set(), set())
    cache = SecondDayCache(history_root, access.candle_files)
    print(
        "Loading causal train/validation 1s OHLCV microstructure; test stays "
        "sealed.",
        file=sys.stderr,
        flush=True,
    )
    train_features, train_targets = load_split(
        "train", segments["train"], cache, access,
    )
    validation_features, validation_targets = load_split(
        "validation", segments["validation"], cache, access,
    )
    if access.target_files & test_target_files:
        raise RuntimeError("sealed test target was opened")
    if test_days & {path.stem for path in access.candle_files}:
        raise RuntimeError("sealed test candle was opened")

    selected, selection_report = select_paired_regime(
        train_features,
        train_targets,
    )
    regime, control_strength, joint_strength = selected
    backoff = select_joint_backoff(
        train_features,
        train_targets,
        selected,
    )
    prior = normalized_mean(train_targets)
    control_edges, control_table, control_counts = fit_probability_table(
        train_features,
        train_targets,
        regime.control_features,
        regime.control_bins,
        control_strength,
        prior,
    )
    joint_edges, joint_table, joint_counts = fit_probability_table(
        train_features,
        train_targets,
        regime.joint_features,
        regime.joint_bins,
        joint_strength,
        prior,
    )
    validation_control_ids, _ = encode_cells(
        validation_features,
        regime.control_features,
        control_edges,
    )
    validation_joint_ids, _ = encode_cells(
        validation_features,
        regime.joint_features,
        joint_edges,
    )
    control_probabilities = control_table[validation_control_ids]
    joint_probabilities = joint_table[validation_joint_ids]
    residual_probabilities = convex_probability_fusion(
        control_probabilities,
        joint_probabilities,
        float(backoff["jointWeight"]),
    )

    checkpoint_file = run_dir / "checkpoints" / "best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file, map_location="cpu", weights_only=False,
    )
    validate_preserved_checkpoint_without_test_access(
        checkpoint,
        plan,
        model_config,
        resolved_training,
        counts,
        run_dir,
    )
    if int(checkpoint["epoch"]) != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("preserved best checkpoint is no longer v18 epoch 7")

    device = resolve_device(requested_device, str(training["device"]))
    dataset = CausalOracleDataset(
        history_root,
        segments,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
        target_rows_per_file=DAY_ROWS,
        action_count=int(model_config["actionCount"]),
        close_cache_days=int(training.get("closeCacheDays", 10)),
        target_cache_days=int(training.get("targetCacheDays", 3)),
        pin_memory=device.type == "cuda",
        include_future_closes=False,
    )
    inference_history: set[Path] = set()
    inference_targets: set[Path] = set()
    dataset.close_cache = AuditedMarketCloseCache(
        history_root,
        int(training.get("closeCacheDays", 10)),
        inference_history,
    )
    dataset.target_cache = AuditedOracleTargetCache(
        int(training.get("targetCacheDays", 3)),
        rows_per_file=DAY_ROWS,
        action_count=int(model_config["actionCount"]),
        pin_memory=device.type == "cuda",
        opened=inference_targets,
    )
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(
        "Running one frozen v18 validation inference pass.",
        file=sys.stderr,
        flush=True,
    )
    logits, model_targets = collect_sequence_core_rows(
        model,
        dataset,
        "validation",
        int(training["sequenceCoreTraining"]["coreRows"]),
        device,
        resolved_training,
    )
    del model, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    expected_shape = (counts["validation"], ACTION_COUNT)
    if any(value.shape != expected_shape for value in (
        logits,
        model_targets,
        validation_targets,
        control_probabilities,
        joint_probabilities,
        residual_probabilities,
    )):
        raise RuntimeError("microstructure fusion rows are misaligned")
    if not np.array_equal(model_targets, validation_targets):
        raise RuntimeError("v18 and microstructure target order differs")
    expected_validation_files = {
        segment.target_file.resolve() for segment in segments["validation"]
    }
    if inference_targets != expected_validation_files \
            or inference_targets & test_target_files \
            or test_days & {path.stem for path in inference_history}:
        raise RuntimeError("unexpected v18 inference reference access")

    probabilities = softmax(logits)
    split_at = probabilities.shape[0] // 2
    probability_weight, probability_fit_kl = select_scalar(
        lambda value: mean_kl_dense(
            validation_targets[:split_at],
            convex_probability_fusion(
                probabilities[:split_at],
                residual_probabilities[:split_at],
                value,
            ),
        ),
        0,
        1,
    )
    ratio_weight, ratio_fit_kl = select_scalar(
        lambda value: mean_kl_dense(
            validation_targets[:split_at],
            log_ratio_feature_fusion(
                probabilities[:split_at],
                residual_probabilities[:split_at],
                control_probabilities[:split_at],
                value,
            ),
        ),
        0,
        4,
    )
    probability_fused = convex_probability_fusion(
        probabilities, residual_probabilities, probability_weight,
    )
    ratio_fused = log_ratio_feature_fusion(
        probabilities,
        residual_probabilities,
        control_probabilities,
        ratio_weight,
    )
    v18_metrics = split_metrics(validation_targets, probabilities, split_at)
    probability_metrics = fusion_metrics(
        probability_weight,
        probability_fit_kl,
        split_metrics(validation_targets, probability_fused, split_at),
        v18_metrics,
    )
    ratio_metrics = fusion_metrics(
        ratio_weight,
        ratio_fit_kl,
        split_metrics(validation_targets, ratio_fused, split_at),
        v18_metrics,
    )
    probability_metrics["passesClean0.002Gate"] = bool(
        probability_metrics["secondHalfKlReductionFromV18"] >= CLEAN_GATE
    )
    ratio_metrics["passesClean0.002Gate"] = bool(
        ratio_metrics["secondHalfKlReductionFromV18"] >= CLEAN_GATE
    )
    timestamps = validation_timestamps(segments["validation"])
    expected_feature_targets = {
        segment.target_file.resolve()
        for split in ("train", "validation")
        for segment in segments[split]
    }
    if access.target_files != expected_feature_targets:
        raise RuntimeError("unexpected feature-audit target access set")
    return {
        "schemaVersion": 1,
        "audit": "v18-plus-causal-1s-ohlcv-microstructure-residual",
        "accessContract": {
            "trainTargetReferenceFilesOpened": len({
                segment.target_file.resolve() for segment in segments["train"]
            }),
            "validationTargetReferenceFilesOpened": len(
                expected_validation_files
            ),
            "causalOneSecondFeatureReferencesOpened": len(access.candle_files),
            "causalOneSecondV18ReferencesOpened": len(inference_history),
            "testTargetReferencesOpened": 0,
            "testTargetPayloadsOpened": 0,
            "testCandleReferencesOpened": 0,
            "testCandlePayloadsOpened": 0,
            "modelWeightsChanged": False,
            "trainingStarted": False,
            "inferenceDevice": str(device),
        },
        "estimator": {
            "featureCount": len(FEATURE_NAMES),
            "microstructureFeatureCount": len(MICRO_FEATURE_NAMES),
            "selectedRegime": regime.name,
            "controlFeatures": list(regime.control_features),
            "microstructureFeatures": list(regime.micro_features),
            "controlPriorStrength": control_strength,
            "jointPriorStrength": joint_strength,
            "controlWeight": backoff["controlWeight"],
            "jointWeight": backoff["jointWeight"],
            "trainInternalBackoffKl": backoff["calibrationKl"],
            "validationRowsWithEmptyControlCell": int(np.count_nonzero(
                control_counts[validation_control_ids] == 0
            )),
            "validationRowsWithEmptyJointCell": int(np.count_nonzero(
                joint_counts[validation_joint_ids] == 0
            )),
            "selection": selection_report,
        },
        "validationSplit": {
            "rows": counts["validation"],
            "scalarFitRows": split_at,
            "untouchedSecondHalfRows": counts["validation"] - split_at,
            "scalarFitTimestampStart": int(timestamps[0]),
            "scalarFitTimestampEnd": int(timestamps[split_at - 1]),
            "untouchedTimestampStart": int(timestamps[split_at]),
            "untouchedTimestampEnd": int(timestamps[-1]),
        },
        "raw01Kl": {
            "v18": v18_metrics,
            "aggregateControl": split_metrics(
                validation_targets, control_probabilities, split_at,
            ),
            "jointMicrostructure": split_metrics(
                validation_targets, joint_probabilities, split_at,
            ),
            "trainSelectedBackoff": split_metrics(
                validation_targets, residual_probabilities, split_at,
            ),
            "convexFusion": probability_metrics,
            "logRatioResidualFusion": ratio_metrics,
        },
        "selectionContract": {
            "regimeShrinkageAndBackoffSelectedOnTrainOnly": True,
            "fusionScalarSelectedOnFirstValidationHalfOnly": True,
            "untouchedSecondHalfUsedForSelection": False,
            "cleanPromotionGate": CLEAN_GATE,
            "testUsed": False,
        },
    }


def read_second_day(
    root: Path,
    day_value: str,
    opened: set[Path],
) -> np.ndarray:
    reference = (root / f"{day_value}.json").resolve()
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    opened.add(reference)
    if manifest.get("sequence") != {
        "start": utc_day_start_ms(day_value),
        "step": SECOND_MS,
        "count": SECOND_ROWS,
        "unit": "unix-ms",
    }:
        raise ValueError(f"invalid one-second sequence: {reference}")
    constants = manifest.get("layout", {}).get("constants", {})
    if constants.get("interval") != "1s" \
            or constants.get("closeTimeOffsetMs") != 999 \
            or constants.get("closed") is not True:
        raise ValueError(f"invalid one-second timing: {reference}")
    values = np.column_stack(tuple(
        read_candle_column(reference, name)
        for name in ("open", "high", "low", "close", "volume")
    )).astype(np.float64, copy=False)
    if values.shape != (SECOND_ROWS, 5) \
            or not np.isfinite(values).all() \
            or bool((values[:, :4] <= 0).any()) \
            or bool((values[:, 4] < 0).any()):
        raise ValueError(f"invalid one-second OHLCV: {reference}")
    return values


def aggregate_minutes(second_values: np.ndarray) -> np.ndarray:
    if second_values.shape != (SECOND_ROWS, 5):
        raise ValueError("one-second day has invalid shape")
    values = second_values.reshape(DAY_ROWS, MINUTE_SECONDS, 5)
    return np.column_stack((
        values[:, 0, 0],
        values[:, :, 1].max(axis=1),
        values[:, :, 2].min(axis=1),
        values[:, -1, 3],
        values[:, :, 4].sum(axis=1),
    ))


def minute_micro_metrics(
    second_values: np.ndarray,
    previous_close: float,
) -> np.ndarray:
    values = second_values.reshape(DAY_ROWS, MINUTE_SECONDS, 5)
    closes = second_values[:, 3]
    prior = np.concatenate((
        np.asarray([previous_close], dtype=np.float64),
        closes[:-1],
    ))
    returns = np.log(closes / prior).reshape(DAY_ROWS, MINUTE_SECONDS)
    highs = values[:, :, 1]
    lows = values[:, :, 2]
    volumes = values[:, :, 4]
    minute_range = np.log(highs.max(axis=1) / lows.min(axis=1))
    second_ranges = np.log(highs / lows)
    absolute_path = np.abs(returns).sum(axis=1)
    net = returns.sum(axis=1)
    sum_squares = np.square(returns).sum(axis=1)
    tiny = np.finfo(np.float64).tiny
    path_inefficiency = np.divide(
        np.maximum(absolute_path - np.abs(net), 0),
        absolute_path,
        out=np.zeros_like(absolute_path),
        where=absolute_path > 0,
    )
    variance_range = sum_squares / (
        sum_squares + np.square(minute_range) + tiny
    )
    sign_flips = (
        returns[:, 1:] * returns[:, :-1] < 0
    ).mean(axis=1)
    extreme_order = (
        np.argmax(highs, axis=1) - np.argmin(lows, axis=1)
    ) / (MINUTE_SECONDS - 1)
    total_volume = volumes.sum(axis=1)
    hhi = np.divide(
        MINUTE_SECONDS * np.square(volumes).sum(axis=1),
        np.square(total_volume),
        out=np.ones_like(total_volume),
        where=total_volume > 0,
    )
    hhi = np.clip((hhi - 1) / (MINUTE_SECONDS - 1), 0, 1)
    volume_imbalance = np.divide(
        volumes[:, 30:].sum(axis=1) - volumes[:, :30].sum(axis=1),
        total_volume,
        out=np.zeros_like(total_volume),
        where=total_volume > 0,
    )
    signed_volume = np.divide(
        (np.sign(returns) * volumes).sum(axis=1),
        total_volume,
        out=np.zeros_like(total_volume),
        where=total_volume > 0,
    )
    range_sum = second_ranges.sum(axis=1)
    micro_range_share = range_sum / (range_sum + minute_range + tiny)
    result = np.column_stack((
        path_inefficiency,
        variance_range,
        sign_flips,
        extreme_order,
        hhi,
        volume_imbalance,
        signed_volume,
        micro_range_share,
    ))
    if result.shape != (DAY_ROWS, len(MICRO_METRIC_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError("invalid minute microstructure metrics")
    return result


def completed_windows(
    previous: np.ndarray,
    current: np.ndarray,
    length: int = 60,
) -> np.ndarray:
    if previous.ndim != 2 or current.shape != previous.shape \
            or previous.shape[0] != DAY_ROWS:
        raise ValueError("minute summaries have invalid shapes")
    combined = np.concatenate((previous[-length:], current), axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(
        combined, length, axis=0,
    )[:DAY_ROWS]
    return np.moveaxis(windows, -1, 1)


def causal_second_microstructure_features(
    previous_seconds: np.ndarray,
    current_seconds: np.ndarray,
) -> np.ndarray:
    previous_minutes = aggregate_minutes(previous_seconds)
    current_minutes = aggregate_minutes(current_seconds)
    minute_windows = completed_candle_windows(
        previous_minutes, current_minutes,
    )
    close_windows = completed_close_windows(
        previous_minutes, current_minutes,
    )
    aggregate = causal_ohlcv_features(minute_windows, close_windows)
    previous_micro = minute_micro_metrics(
        previous_seconds, float(previous_seconds[0, 3]),
    )
    current_micro = minute_micro_metrics(
        current_seconds, float(previous_seconds[-1, 3]),
    )
    micro_windows = completed_windows(previous_micro, current_micro)
    summary_columns = [
        micro_windows[:, -horizon:, metric_index].mean(axis=1)
        for metric_index in range(len(MICRO_METRIC_NAMES))
        for horizon in HORIZONS
    ]
    current_indexes = np.arange(DAY_ROWS) * MINUTE_SECONDS
    current_rows = current_seconds[current_indexes]
    current_ranges = np.log(current_rows[:, 1] / current_rows[:, 2])
    recent_ranges = np.log(
        minute_windows[:, :, 1] / minute_windows[:, :, 2]
    ).mean(axis=1)
    range_fraction = current_ranges / (
        current_ranges + recent_ranges + np.finfo(np.float64).tiny
    )
    recent_second_volume = minute_windows[:, :, 4].sum(axis=1) / 3_600
    volume_floor = np.maximum(
        recent_second_volume * 1e-9,
        np.finfo(np.float64).tiny,
    )
    volume_surprise = (
        np.log(current_rows[:, 4] + volume_floor)
        - np.log(recent_second_volume + volume_floor)
    )
    previous_current_closes = np.concatenate((
        np.asarray([previous_seconds[-1, 3]]),
        current_seconds[current_indexes[1:] - 1, 3],
    ))
    current_returns = np.log(current_rows[:, 3] / previous_current_closes)
    current_signed_volume = np.sign(current_returns) * np.divide(
        current_rows[:, 4],
        current_rows[:, 4] + recent_second_volume + volume_floor,
    )
    result = np.column_stack((
        aggregate,
        *summary_columns,
        range_fraction,
        volume_surprise,
        current_signed_volume,
    )).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, len(FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError("invalid causal microstructure feature matrix")
    return result


def daily_features(day_value: str, cache: SecondDayCache) -> np.ndarray:
    current_date = date.fromisoformat(day_value)
    previous_value = (current_date - timedelta(days=1)).isoformat()
    return causal_second_microstructure_features(
        cache.load(previous_value),
        cache.load(day_value),
    )


def load_split(
    split: str,
    segments: list[CausalSegment],
    cache: SecondDayCache,
    access: AccessLog,
) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("microstructure audit only loads train or validation")
    if any(segment.split != split for segment in segments):
        raise ValueError("segment split differs from requested split")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, segment in enumerate(segments, start=1):
        day_features = daily_features(segment.target_file.stem, cache)
        target_file = segment.target_file.resolve()
        _shard, day_targets = read_shard_array(
            target_file, "<f4", (DAY_ROWS, ACTION_COUNT),
        )
        access.target_files.add(target_file)
        start = segment.target_row_offset
        end = start + segment.count
        features.append(day_features[start:end])
        targets.append(np.asarray(day_targets[start:end], dtype=np.float32))
        if index % 25 == 0 or index == len(segments):
            print(
                f"{split}: {index}/{len(segments)} days loaded",
                file=sys.stderr,
                flush=True,
            )
    feature_rows = np.concatenate(features)
    target_rows = np.concatenate(targets)
    if feature_rows.shape != (target_rows.shape[0], len(FEATURE_NAMES)) \
            or not np.isfinite(feature_rows).all() \
            or not np.isfinite(target_rows).all() \
            or bool((target_rows < 0).any()) \
            or not np.allclose(
                target_rows.sum(axis=1), 1, atol=2e-4, rtol=2e-4,
            ):
        raise ValueError(f"invalid {split} microstructure rows")
    return feature_rows, target_rows


def fit_edges(
    features: np.ndarray,
    names: tuple[str, ...],
    bins: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.unique(np.quantile(
            features[:, FEATURE_INDEX[name]].astype(np.float64, copy=False),
            np.linspace(0, 1, count + 1)[1:-1],
        ))
        for name, count in zip(names, bins, strict=True)
    )


def encode_cells(
    features: np.ndarray,
    names: tuple[str, ...],
    edges: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, int]:
    ids = np.zeros(features.shape[0], dtype=np.int64)
    multiplier = 1
    for name, current_edges in zip(names, edges, strict=True):
        ids += np.searchsorted(
            current_edges,
            features[:, FEATURE_INDEX[name]],
            side="right",
        ) * multiplier
        multiplier *= current_edges.size + 1
    return ids, multiplier


def fit_probability_table(
    features: np.ndarray,
    targets: np.ndarray,
    names: tuple[str, ...],
    bins: tuple[int, ...],
    strength: float,
    prior: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    edges = fit_edges(features, names, bins)
    ids, count = encode_cells(features, names, edges)
    sums, counts = sufficient_table(targets, ids, count)
    return edges, smoothed_table(sums, counts, prior, strength), counts


def candidate_scores(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    names: tuple[str, ...],
    bins: tuple[int, ...],
    prior: np.ndarray,
) -> tuple[float, float, dict[str, float], int, int]:
    edges = fit_edges(fit_features, names, bins)
    fit_ids, cell_count = encode_cells(fit_features, names, edges)
    calibration_ids, _ = encode_cells(calibration_features, names, edges)
    sums, counts = sufficient_table(fit_targets, fit_ids, cell_count)
    scores = {
        str(int(strength)): mean_kl_indexed(
            calibration_targets,
            calibration_ids,
            smoothed_table(sums, counts, prior, strength),
        )
        for strength in PRIOR_STRENGTHS
    }
    strength, score = min(
        ((float(name), value) for name, value in scores.items()),
        key=lambda item: (item[1], item[0]),
    )
    return strength, score, scores, cell_count, int(np.count_nonzero(counts))


def select_paired_regime(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[PairedRegime, float, float], dict[str, object]]:
    fit_end = int(features.shape[0] * (1 - CALIBRATION_FRACTION))
    fit_features = features[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_features = features[fit_end:]
    calibration_targets = targets[fit_end:]
    prior = normalized_mean(fit_targets)
    rows: list[dict[str, object]] = []
    choices: list[tuple[float, str, PairedRegime, float, float]] = []
    for regime in REGIMES:
        control = candidate_scores(
            fit_features,
            fit_targets,
            calibration_features,
            calibration_targets,
            regime.control_features,
            regime.control_bins,
            prior,
        )
        joint = candidate_scores(
            fit_features,
            fit_targets,
            calibration_features,
            calibration_targets,
            regime.joint_features,
            regime.joint_bins,
            prior,
        )
        rows.append({
            "name": regime.name,
            "controlFeatures": list(regime.control_features),
            "microstructureFeatures": list(regime.micro_features),
            "controlKlByStrength": control[2],
            "jointKlByStrength": joint[2],
            "selectedControlKl": control[1],
            "selectedJointKl": joint[1],
            "jointKlReductionFromControl": control[1] - joint[1],
            "controlCells": control[3],
            "jointCells": joint[3],
            "populatedControlCells": control[4],
            "populatedJointCells": joint[4],
        })
        choices.append((joint[1], regime.name, regime, control[0], joint[0]))
    winner = min(choices, key=lambda item: (item[0], item[1]))
    return (winner[2], winner[3], winner[4]), {
        "fitRows": fit_end,
        "calibrationRows": features.shape[0] - fit_end,
        "validationUsedForSelection": False,
        "candidates": rows,
        "selected": {
            "name": winner[2].name,
            "controlPriorStrength": winner[3],
            "jointPriorStrength": winner[4],
            "calibrationJointKl": winner[0],
        },
    }


def select_joint_backoff(
    features: np.ndarray,
    targets: np.ndarray,
    selected: tuple[PairedRegime, float, float],
) -> dict[str, float | int | bool]:
    fit_end = int(features.shape[0] * (1 - CALIBRATION_FRACTION))
    fit_features = features[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_features = features[fit_end:]
    calibration_targets = targets[fit_end:]
    prior = normalized_mean(fit_targets)
    regime, control_strength, joint_strength = selected
    control_edges, control_table, _ = fit_probability_table(
        fit_features,
        fit_targets,
        regime.control_features,
        regime.control_bins,
        control_strength,
        prior,
    )
    joint_edges, joint_table, _ = fit_probability_table(
        fit_features,
        fit_targets,
        regime.joint_features,
        regime.joint_bins,
        joint_strength,
        prior,
    )
    control_ids, _ = encode_cells(
        calibration_features, regime.control_features, control_edges,
    )
    joint_ids, _ = encode_cells(
        calibration_features, regime.joint_features, joint_edges,
    )
    weight, score = select_scalar(
        lambda value: mean_kl_stacked_indexed(
            calibration_targets,
            control_ids,
            control_table,
            joint_ids,
            joint_table,
            value,
        ),
        0,
        1,
    )
    return {
        "fitRows": fit_end,
        "calibrationRows": targets.shape[0] - fit_end,
        "controlWeight": 1 - weight,
        "jointWeight": weight,
        "calibrationKl": score,
        "validationUsedForSelection": False,
    }


def mean_kl_dense(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.shape != predictions.shape:
        raise ValueError("dense KL inputs differ")
    return mean_kl_indexed(
        targets,
        np.arange(targets.shape[0], dtype=np.int64),
        predictions,
    )


if __name__ == "__main__":
    main()
