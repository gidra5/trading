"""Audit whether completed-minute six-hour close summaries add beyond v18.

The feature tables and their empirical-Bayes shrinkage are selected on the
final chronological 20% of training.  A train-selected mixture backs the
six-hour table off to a matched one-hour close control.  Frozen v18 is then
run exactly once, and only on the 42,780 validation rows retained by v28.  The
first chronological validation half selects scalar fusion strengths; the
second half is the untouched promotion holdout.

The primary fusion is a log-ratio residual::

    p(action) = normalize(v18(action) * (long(action) / short(action)) ** w)

It therefore tests the table evidence attributable to six-hour summaries
beyond a matched one-hour table.  No neural weights are changed or trained,
and sealed test reference JSON and payload objects are never opened.
"""

from __future__ import annotations

import argparse
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
    CALIBRATION_FRACTION,
    DAY_ROWS,
    MINUTE_CLOSE_OFFSET_MS,
    MINUTE_MS,
    PRIOR_STRENGTHS,
    latest_closed_candle_start_ms,
    mean_kl_indexed,
    mean_kl_stacked_indexed,
    quantile_edges,
    smoothed_table,
    sufficient_table,
    utc_day_start_ms,
)
from audit_causal_oracle_predictability import mean_kl, normalized_mean
from audit_v18_calendar_fusion import (
    convex_probability_fusion,
    select_scalar,
    softmax,
    split_metrics,
    validate_preserved_checkpoint_without_test_access,
)
from audit_v18_ohlcv_fusion import log_ratio_feature_fusion
from audit_v18_on_v28_validation_rows import (
    collect_sequence_core_rows,
    filename_only_segments,
    load_plan,
    read_dataset_event,
    split_counts,
    validate_checkpoint_dataset_identity,
    validate_comparable_plans,
    validate_dataset_event,
)
from trading_storage import (
    load_torch_checkpoint,
    read_candle_column,
    read_shard_array,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    CausalOracleDataset,
    build_model,
    resolve,
    resolve_training_config,
)


DEFAULT_V18_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
)
DEFAULT_V28_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-long-tcn-v28-reuse.json"
)
HISTORY_MINUTES = 360
LOG_RATIO_BOUNDS = (0.0, 4.0)
PROBABILITY_MIXTURE_BOUNDS = (0.0, 1.0)
PROMOTION_GAIN = 0.002
EXPECTED_VALIDATION_ROWS = 42_780


FEATURE_NAMES = (
    "return1h",
    "return2h",
    "return3h",
    "return6h",
    "rmsReturn1h",
    "rmsReturn2h",
    "rmsReturn3h",
    "rmsReturn6h",
    "ma360Path",
    "ma180MinusMa360",
    "ma120MinusMa180",
    "ma60MinusMa120",
    "pathMinusMa60",
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


# Candidate definitions are fixed before validation inference.  The short
# family is a matched control using only summaries from v18's final hour.  The
# long family adds summaries that necessarily depend on older completed
# minutes.  Sparse high-dimensional crosses are deliberately avoided.
REGIME_SPECS = (
    RegimeSpec(
        "short-control",
        "short-return-vol",
        ("return1h", "rmsReturn1h"),
        (14, 10),
    ),
    RegimeSpec(
        "short-control",
        "short-trend-vol",
        ("return1h", "rmsReturn1h", "pathMinusMa60"),
        (10, 8, 6),
    ),
    RegimeSpec(
        "six-hour",
        "long-endpoint-vol",
        (
            "return1h",
            "return2h",
            "return6h",
            "rmsReturn1h",
            "rmsReturn2h",
            "rmsReturn6h",
        ),
        (5, 5, 8, 5, 4, 5),
    ),
    RegimeSpec(
        "six-hour",
        "long-trend-vol",
        (
            "return1h",
            "return3h",
            "return6h",
            "rmsReturn1h",
            "rmsReturn3h",
            "rmsReturn6h",
        ),
        (5, 5, 8, 5, 4, 5),
    ),
    RegimeSpec(
        "six-hour",
        "long-additive-bands",
        (
            "ma180MinusMa360",
            "ma120MinusMa180",
            "ma60MinusMa120",
            "pathMinusMa60",
            "rmsReturn1h",
            "rmsReturn6h",
        ),
        (4, 4, 4, 5, 4, 6),
    ),
    RegimeSpec(
        "six-hour",
        "long-compact-band",
        (
            "return1h",
            "return6h",
            "ma180MinusMa360",
            "ma60MinusMa120",
            "rmsReturn6h",
        ),
        (5, 8, 5, 5, 7),
    ),
)
FAMILIES = ("short-control", "six-hour")


@dataclass
class AccessLog:
    target_files: set[Path]
    candle_files: set[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-plan", type=Path, default=DEFAULT_V18_PLAN)
    parser.add_argument("--v28-plan", type=Path, default=DEFAULT_V28_PLAN)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = audit_v18_six_hour_close_fusion(
        arguments.v18_plan,
        arguments.v28_plan,
        requested_device=arguments.device,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


def audit_v18_six_hour_close_fusion(
    v18_plan_file: Path,
    v28_plan_file: Path,
    *,
    requested_device: str = "auto",
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    storage = training_storage_layout(repo_root)
    v18_plan = load_plan(repo_root, v18_plan_file)
    v28_plan = load_plan(repo_root, v28_plan_file)
    validate_comparable_plans(v18_plan, v28_plan)
    target_root = require_under(
        resolve(repo_root, Path(v18_plan["targetReferenceDir"])),
        storage.immutable / "refs" / "oracle",
        "targetReferenceDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(v18_plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    minute_history_root = history_root.parent / "1m"
    target_files = sorted(target_root.glob("*.json"))
    v18_segments = filename_only_segments(target_files, v18_plan)
    v28_segments = filename_only_segments(target_files, v28_plan)
    v18_counts = split_counts(v18_segments)
    v28_counts = split_counts(v28_segments)
    if v28_counts["validation"] != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("v28 validation count is not the sealed 42,780 rows")

    test_target_files = {
        segment.target_file.resolve() for segment in v28_segments["test"]
    }
    test_day_names = {
        segment.target_file.stem for segment in v28_segments["test"]
    }
    access = AccessLog(set(), set())
    close_cache: dict[str, np.ndarray] = {}
    print(
        "Loading train and exact v28-validation six-hour completed-minute "
        "features; test remains sealed.",
        file=sys.stderr,
        flush=True,
    )
    train_features, train_targets = load_feature_target_split(
        "train",
        v28_segments["train"],
        minute_history_root,
        close_cache,
        access,
    )
    validation_features, validation_targets = load_feature_target_split(
        "validation",
        v28_segments["validation"],
        minute_history_root,
        close_cache,
        access,
    )
    if access.target_files & test_target_files:
        raise RuntimeError("sealed test target reference or payload was opened")
    if test_day_names & {path.stem for path in access.candle_files}:
        raise RuntimeError("sealed test candle reference was opened")

    selected, train_selection = select_each_family(
        train_features,
        train_targets,
    )
    short_selected = selected["short-control"]
    long_selected = selected["six-hour"]
    backoff_selection = select_long_backoff(
        train_features,
        train_targets,
        short_selected,
        long_selected,
    )
    train_prior = normalized_mean(train_targets)
    short_edges, short_table, short_counts = fitted_regime(
        *short_selected,
        train_features,
        train_targets,
        train_prior,
    )
    long_edges, long_table, long_counts = fitted_regime(
        *long_selected,
        train_features,
        train_targets,
        train_prior,
    )
    short_ids, _ = cell_ids(
        validation_features,
        short_selected[0],
        short_edges,
    )
    long_ids, _ = cell_ids(
        validation_features,
        long_selected[0],
        long_edges,
    )
    short_probabilities = short_table[short_ids]
    long_probabilities = long_table[long_ids]
    long_weight = float(backoff_selection["longWeight"])
    backed_long_probabilities = convex_probability_fusion(
        short_probabilities,
        long_probabilities,
        long_weight,
    )

    v18_run = require_under(
        resolve(repo_root, Path(v18_plan["runDir"])),
        storage.runs,
        "v18 runDir",
    )
    v28_run = require_under(
        resolve(repo_root, Path(v28_plan["runDir"])),
        storage.runs,
        "v28 runDir",
    )
    v18_event = read_dataset_event(v18_run)
    v28_event = read_dataset_event(v28_run)
    validate_dataset_event(v18_event, v18_plan, v18_counts)
    validate_dataset_event(v28_event, v28_plan, v28_counts)
    checkpoint_file = v18_run / "checkpoints" / "best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    resolved_training = resolve_training_config(v18_plan["training"])
    validate_preserved_checkpoint_without_test_access(
        checkpoint,
        v18_plan,
        v18_plan["model"],
        resolved_training,
        v18_counts,
        v18_run,
    )
    validate_checkpoint_dataset_identity(checkpoint, v18_event, v18_plan)
    if int(checkpoint.get("epoch", -1)) != 7:
        raise ValueError("six-hour fusion audit requires frozen v18 epoch 7")

    device = resolve_device(requested_device, str(v18_plan["training"]["device"]))
    inference_segments = {
        "train": [],
        "validation": v28_segments["validation"],
        "test": [],
    }
    dataset = CausalOracleDataset(
        history_root,
        inference_segments,
        int(v18_plan["model"]["contextLength"]),
        int(v18_plan["model"]["forecastHorizon"]),
        target_rows_per_file=DAY_ROWS,
        action_count=int(v18_plan["model"]["actionCount"]),
        close_cache_days=int(v18_plan["training"].get("closeCacheDays", 10)),
        target_cache_days=int(
            v18_plan["training"].get("targetCacheDays", 3)
        ),
        pin_memory=device.type == "cuda",
        include_future_closes=False,
    )
    inference_target_files: set[Path] = set()
    inference_history_files: set[Path] = set()
    original_target_load = dataset.target_cache.load
    original_history_load = dataset.close_cache.load_day

    def tracked_target_load(file: Path) -> torch.Tensor:
        inference_target_files.add(file.resolve())
        return original_target_load(file)

    def tracked_history_load(day_value: str) -> np.ndarray:
        inference_history_files.add(
            (history_root / f"{day_value}.json").resolve()
        )
        return original_history_load(day_value)

    dataset.target_cache.load = tracked_target_load
    dataset.close_cache.load_day = tracked_history_load
    model = build_model(v18_plan["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    print(
        "Running frozen v18 once on exactly 42,780 v28 validation rows.",
        file=sys.stderr,
        flush=True,
    )
    logits, inference_targets = collect_sequence_core_rows(
        model,
        dataset,
        "validation",
        int(v18_plan["training"]["sequenceCoreTraining"]["coreRows"]),
        device,
        resolved_training,
    )
    del model, dataset, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    expected_shape = (EXPECTED_VALIDATION_ROWS, ACTION_COUNT)
    if logits.shape != expected_shape \
            or inference_targets.shape != expected_shape \
            or validation_targets.shape != expected_shape \
            or short_probabilities.shape != expected_shape \
            or long_probabilities.shape != expected_shape:
        raise RuntimeError("v18 and feature-table rows are not aligned")
    if not np.array_equal(inference_targets, validation_targets):
        raise RuntimeError("v18 target order differs from feature-table order")
    expected_validation_target_files = {
        segment.target_file.resolve()
        for segment in v28_segments["validation"]
    }
    if inference_target_files != expected_validation_target_files \
            or inference_target_files & test_target_files:
        raise RuntimeError("v18 inference opened unexpected target references")
    if test_day_names & {path.stem for path in inference_history_files}:
        raise RuntimeError("v18 inference opened sealed test candles")

    base_probabilities = softmax(logits)
    split_at = EXPECTED_VALIDATION_ROWS // 2
    residual_weight, residual_fit_kl = select_scalar(
        lambda value: mean_kl(
            validation_targets[:split_at],
            log_ratio_feature_fusion(
                base_probabilities[:split_at],
                backed_long_probabilities[:split_at],
                short_probabilities[:split_at],
                value,
            ),
        ),
        *LOG_RATIO_BOUNDS,
    )
    mixture_weight, mixture_fit_kl = select_scalar(
        lambda value: mean_kl(
            validation_targets[:split_at],
            convex_probability_fusion(
                base_probabilities[:split_at],
                backed_long_probabilities[:split_at],
                value,
            ),
        ),
        *PROBABILITY_MIXTURE_BOUNDS,
    )
    residual_probabilities = log_ratio_feature_fusion(
        base_probabilities,
        backed_long_probabilities,
        short_probabilities,
        residual_weight,
    )
    mixture_probabilities = convex_probability_fusion(
        base_probabilities,
        backed_long_probabilities,
        mixture_weight,
    )
    v18_metrics = split_metrics(
        validation_targets,
        base_probabilities,
        split_at,
    )
    residual_metrics = split_metrics(
        validation_targets,
        residual_probabilities,
        split_at,
    )
    mixture_metrics = split_metrics(
        validation_targets,
        mixture_probabilities,
        split_at,
    )
    holdout_gain = (
        v18_metrics["secondHalfKl"] - residual_metrics["secondHalfKl"]
    )
    timestamps = validation_timestamps(v28_segments["validation"])
    expected_feature_target_files = {
        segment.target_file.resolve()
        for split in ("train", "validation")
        for segment in v28_segments[split]
    }
    if access.target_files != expected_feature_target_files:
        raise RuntimeError("unexpected feature-audit target access set")

    return {
        "schemaVersion": 1,
        "audit": "frozen-v18-plus-six-hour-completed-close-regime-residual",
        "accessContract": {
            "featureTrainTargetReferenceFilesOpened": len({
                segment.target_file.resolve()
                for segment in v28_segments["train"]
            }),
            "featureValidationTargetReferenceFilesOpened": len({
                segment.target_file.resolve()
                for segment in v28_segments["validation"]
            }),
            "completedMinuteCandleReferencesOpened": len(access.candle_files),
            "v18ValidationTargetReferenceFilesOpened": len(
                inference_target_files
            ),
            "v18OneSecondCandleReferencesOpened": len(
                inference_history_files
            ),
            "testTargetReferenceMetadataOpened": 0,
            "testTargetPayloadsOpened": 0,
            "testCandleReferenceMetadataOpened": 0,
            "testCandlePayloadsOpened": 0,
            "frozenV18ValidationInferencePasses": 1,
            "modelWeightsChanged": False,
            "neuralTrainingStarted": False,
            "inferenceDevice": str(device),
        },
        "checkpoint": {
            "planId": v18_plan["id"],
            "kind": "best",
            "epoch": 7,
            "file": str(checkpoint_file),
            "datasetFingerprint": v18_event["datasetFingerprint"],
        },
        "features": {
            "historyMinutes": HISTORY_MINUTES,
            "usesOnlyFullyCompletedMinuteCandles": True,
            "scaleInvariant": True,
            "names": list(FEATURE_NAMES),
            "additivePathBandIdentity": (
                "ma360Path + ma180MinusMa360 + ma120MinusMa180 + "
                "ma60MinusMa120 + pathMinusMa60 == return6h"
            ),
        },
        "trainSelection": {
            **train_selection,
            "longBackoff": backoff_selection,
        },
        "selectedTables": {
            "shortControl": selected_table_report(
                short_selected,
                short_counts,
                short_ids,
            ),
            "sixHour": selected_table_report(
                long_selected,
                long_counts,
                long_ids,
            ),
            "trainSelectedShortBackoffWeight": (
                backoff_selection["shortWeight"]
            ),
            "trainSelectedSixHourWeight": backoff_selection["longWeight"],
        },
        "validationSplit": {
            "rows": EXPECTED_VALIDATION_ROWS,
            "scalarFitRows": split_at,
            "untouchedHoldoutRows": EXPECTED_VALIDATION_ROWS - split_at,
            "scalarFitTimestampStart": int(timestamps[0]),
            "scalarFitTimestampEnd": int(timestamps[split_at - 1]),
            "holdoutTimestampStart": int(timestamps[split_at]),
            "holdoutTimestampEnd": int(timestamps[-1]),
        },
        "raw01Kl": {
            "v18": v18_metrics,
            "shortControlOnly": split_metrics(
                validation_targets,
                short_probabilities,
                split_at,
            ),
            "sixHourOnly": split_metrics(
                validation_targets,
                long_probabilities,
                split_at,
            ),
            "sixHourWithTrainSelectedShortBackoff": split_metrics(
                validation_targets,
                backed_long_probabilities,
                split_at,
            ),
            "primaryLogRatioResidual": {
                "selectedWeight": residual_weight,
                "firstHalfSelectionKl": residual_fit_kl,
                **residual_metrics,
                "secondHalfKlReductionFromV18": holdout_gain,
                "fullValidationKlReductionFromV18": (
                    v18_metrics["fullValidationKl"]
                    - residual_metrics["fullValidationKl"]
                ),
            },
            "descriptiveConvexProbabilityMixture": {
                "selectedWeight": mixture_weight,
                "firstHalfSelectionKl": mixture_fit_kl,
                **mixture_metrics,
                "secondHalfKlReductionFromV18": (
                    v18_metrics["secondHalfKl"]
                    - mixture_metrics["secondHalfKl"]
                ),
                "fullValidationKlReductionFromV18": (
                    v18_metrics["fullValidationKl"]
                    - mixture_metrics["fullValidationKl"]
                ),
                "eligibleForPromotionGate": False,
            },
        },
        "promotionGate": {
            "metric": "primary log-ratio residual raw T=0.01 KL gain on untouched second validation half",
            "requiredGain": PROMOTION_GAIN,
            "observedGain": holdout_gain,
            "passed": holdout_gain >= PROMOTION_GAIN,
        },
        "selectionContract": {
            "featureAndRegimeCandidatesPredeclared": True,
            "regimeShrinkageAndBackoffSelectedOnTrainOnly": True,
            "fusionScalarsSelectedOnFirstValidationHalfOnly": True,
            "secondValidationHalfUsedForAnySelection": False,
            "fullValidationUsedForAnySelection": False,
            "testUsed": False,
        },
    }


def resolve_device(requested: str, configured: str) -> torch.device:
    value = configured if requested == "auto" else requested
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def read_minute_close_day(
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
    if manifest.get("sequence") != {
        "start": utc_day_start_ms(day_value),
        "step": MINUTE_MS,
        "count": DAY_ROWS,
        "unit": "unix-ms",
    }:
        raise ValueError(f"non-canonical one-minute sequence: {reference}")
    constants = manifest.get("layout", {}).get("constants", {})
    if constants.get("interval") != "1m" \
            or constants.get("closeTimeOffsetMs") \
            != MINUTE_CLOSE_OFFSET_MS \
            or constants.get("closed") is not True:
        raise ValueError(f"non-canonical one-minute close timing: {reference}")
    values = np.asarray(read_candle_column(reference, "close"), dtype=np.float64)
    if values.shape != (DAY_ROWS,) \
            or not np.isfinite(values).all() \
            or bool((values <= 0).any()):
        raise ValueError(f"invalid one-minute closes: {reference}")
    cache[day_value] = values
    return values


def completed_close_windows(
    previous_day: np.ndarray,
    current_day: np.ndarray,
    transition_count: int = HISTORY_MINUTES,
) -> np.ndarray:
    if previous_day.shape != (DAY_ROWS,) \
            or current_day.shape != (DAY_ROWS,) \
            or not 1 <= transition_count < DAY_ROWS:
        raise ValueError("invalid close days or transition count")
    length = transition_count + 1
    combined = np.concatenate((previous_day[-length:], current_day))
    return np.lib.stride_tricks.sliding_window_view(
        combined,
        length,
    )[:DAY_ROWS]


def causal_six_hour_features(close_windows: np.ndarray) -> np.ndarray:
    if close_windows.ndim != 2 \
            or close_windows.shape[1] != HISTORY_MINUTES + 1 \
            or not np.isfinite(close_windows).all() \
            or bool((close_windows <= 0).any()):
        raise ValueError("expected positive [rows,361] completed closes")
    returns = np.diff(np.log(close_windows.astype(np.float64, copy=False)), axis=1)
    path = np.cumsum(returns, axis=1)

    def trailing_sum(length: int) -> np.ndarray:
        return returns[:, -length:].sum(axis=1)

    def trailing_rms(length: int) -> np.ndarray:
        return np.sqrt(np.mean(np.square(returns[:, -length:]), axis=1))

    ma360 = path.mean(axis=1)
    ma180 = path[:, -180:].mean(axis=1)
    ma120 = path[:, -120:].mean(axis=1)
    ma60 = path[:, -60:].mean(axis=1)
    features = np.column_stack((
        *(trailing_sum(length) for length in (60, 120, 180, 360)),
        *(trailing_rms(length) for length in (60, 120, 180, 360)),
        ma360,
        ma180 - ma360,
        ma120 - ma180,
        ma60 - ma120,
        path[:, -1] - ma60,
    )).astype(np.float32, copy=False)
    if features.shape != (close_windows.shape[0], len(FEATURE_NAMES)) \
            or not np.isfinite(features).all():
        raise ValueError("six-hour close features are invalid")
    bands = features[:, 8:].sum(axis=1)
    if not np.allclose(bands, features[:, FEATURE_INDEX["return6h"]], atol=2e-7, rtol=2e-5):
        raise RuntimeError("additive path bands do not reconstruct return6h")
    return features


def daily_causal_six_hour_features(
    candle_root: Path,
    day_value: str,
    cache: dict[str, np.ndarray],
    opened: set[Path],
) -> np.ndarray:
    day = date.fromisoformat(day_value)
    previous_value = (day - timedelta(days=1)).isoformat()
    previous = read_minute_close_day(candle_root, previous_value, cache, opened)
    current = read_minute_close_day(candle_root, day_value, cache, opened)
    features = causal_six_hour_features(
        completed_close_windows(previous, current)
    )
    first_prediction = utc_day_start_ms(day_value) + 999
    last_prediction = first_prediction + (DAY_ROWS - 1) * MINUTE_MS
    if latest_closed_candle_start_ms(first_prediction) \
            != utc_day_start_ms(day_value) - MINUTE_MS \
            or latest_closed_candle_start_ms(last_prediction) \
            != utc_day_start_ms(day_value) + (DAY_ROWS - 2) * MINUTE_MS:
        raise RuntimeError("completed-minute feature alignment failed")
    return features


def load_feature_target_split(
    split: str,
    segments: list,
    candle_root: Path,
    candle_cache: dict[str, np.ndarray],
    access: AccessLog,
) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "validation"} \
            or any(segment.split != split for segment in segments):
        raise ValueError("feature audit may load only its train/validation split")
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for index, segment in enumerate(segments, start=1):
        day_value = segment.target_file.stem
        daily = daily_causal_six_hour_features(
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
        start = segment.target_row_offset
        end = start + segment.count
        feature_parts.append(daily[start:end])
        target_parts.append(np.asarray(targets[start:end], dtype=np.float32))
        if index % 50 == 0 or index == len(segments):
            print(
                f"{split}: {index}/{len(segments)} days loaded",
                file=sys.stderr,
                flush=True,
            )
    features = np.concatenate(feature_parts)
    targets = np.concatenate(target_parts)
    if features.shape != (targets.shape[0], len(FEATURE_NAMES)) \
            or not np.isfinite(targets).all() \
            or bool((targets < 0).any()) \
            or not np.allclose(targets.sum(axis=1), 1, atol=2e-4, rtol=2e-4):
        raise ValueError(f"invalid {split} feature/target corpus")
    return features, targets


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
    if len(edges) != len(spec.feature_indexes):
        raise ValueError("regime edges do not match feature indexes")
    result = np.zeros(features.shape[0], dtype=np.int64)
    multiplier = 1
    for index, current_edges in zip(
        spec.feature_indexes,
        edges,
        strict=True,
    ):
        result += np.searchsorted(
            current_edges,
            features[:, index],
            side="right",
        ) * multiplier
        multiplier *= current_edges.size + 1
    return result, multiplier


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


def select_each_family(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[dict[str, tuple[RegimeSpec, float]], dict[str, object]]:
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
            if spec.family not in best or score < best[spec.family][0]:
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
        raise RuntimeError("a six-hour regime family had no candidate")
    return {
        family: (item[1], item[2]) for family, item in best.items()
    }, {
        "fitRows": fit_end,
        "calibrationRows": features.shape[0] - fit_end,
        "calibrationFraction": CALIBRATION_FRACTION,
        "validationUsedForSelection": False,
        "candidates": candidates,
        "selected": {
            family: {
                "name": item[1].name,
                "features": list(item[1].features),
                "binCounts": list(item[1].bin_counts),
                "priorStrength": item[2],
                "calibrationKl": item[0],
            }
            for family, item in best.items()
        },
    }


def select_long_backoff(
    features: np.ndarray,
    targets: np.ndarray,
    short_selected: tuple[RegimeSpec, float],
    long_selected: tuple[RegimeSpec, float],
) -> dict[str, float | int | str | bool]:
    fit_end = int(features.shape[0] * (1 - CALIBRATION_FRACTION))
    fit_features = features[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_features = features[fit_end:]
    calibration_targets = targets[fit_end:]
    prior = normalized_mean(fit_targets)
    short_edges, short_table, _ = fitted_regime(
        *short_selected,
        fit_features,
        fit_targets,
        prior,
    )
    long_edges, long_table, _ = fitted_regime(
        *long_selected,
        fit_features,
        fit_targets,
        prior,
    )
    short_ids, _ = cell_ids(
        calibration_features,
        short_selected[0],
        short_edges,
    )
    long_ids, _ = cell_ids(
        calibration_features,
        long_selected[0],
        long_edges,
    )
    weight, calibration_kl = select_scalar(
        lambda value: mean_kl_stacked_indexed(
            calibration_targets,
            short_ids,
            short_table,
            long_ids,
            long_table,
            value,
        ),
        0.0,
        1.0,
    )
    return {
        "fitRows": fit_end,
        "calibrationRows": targets.shape[0] - fit_end,
        "shortSpec": short_selected[0].name,
        "longSpec": long_selected[0].name,
        "shortWeight": 1 - weight,
        "longWeight": weight,
        "calibrationKl": calibration_kl,
        "validationUsedForSelection": False,
    }


def selected_table_report(
    selected: tuple[RegimeSpec, float],
    counts: np.ndarray,
    validation_ids: np.ndarray,
) -> dict[str, object]:
    spec, strength = selected
    return {
        "name": spec.name,
        "features": list(spec.features),
        "binCounts": list(spec.bin_counts),
        "priorStrength": strength,
        "effectiveCells": int(counts.size),
        "populatedTrainCells": int(np.count_nonzero(counts)),
        "validationRowsInEmptyTrainCells": int(np.count_nonzero(
            counts[validation_ids] == 0
        )),
    }


def validation_timestamps(segments: list) -> np.ndarray:
    parts = [
        segment.prediction_time_start
        + np.arange(segment.count, dtype=np.int64) * segment.step_ms
        for segment in segments
    ]
    if not parts:
        raise ValueError("validation timestamp split is empty")
    result = np.concatenate(parts)
    if result.size > 1 and bool((np.diff(result) <= 0).any()):
        raise ValueError("validation timestamps must be strictly chronological")
    return result


if __name__ == "__main__":
    main()
