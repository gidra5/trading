"""Leakage-safe diagnostic fusion of preserved v18 with 1m OHLCV geometry.

The OHLCV estimator, its quantile cells, shrinkage, and its close-control
backoff are selected using chronological training rows only.  Each validation
feature row ends at the newest one-minute candle fully closed by prediction
time.  The first chronological half of validation then selects one scalar for
each of two frozen fusion forms; the untouched second half is the fusion
holdout.  No weights are trained and sealed test references are never opened.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from audit_causal_oracle_ohlcv_predictability import (
    ACTION_COUNT,
    AccessLog,
    FEATURE_NAMES,
    cell_ids,
    fitted_regime,
    load_split,
    select_each_family,
    select_stacked_mixture,
    split_and_purge,
)
from audit_causal_oracle_predictability import mean_kl, normalized_mean
from audit_v18_calendar_fusion import (
    convex_probability_fusion,
    select_scalar,
    softmax,
    split_metrics,
    validate_preserved_checkpoint_without_test_access,
)
from evaluate_joint_price_oracle_actions import (
    collect_policy_rows,
    resolve_device,
)
from trading_storage import (
    load_torch_checkpoint,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    CausalOracleDataset,
    MarketCloseCache,
    OracleTargetCache,
    build_model,
    resolve,
    resolve_training_config,
    validate_plan,
)


DEFAULT_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
)
EXPECTED_CHECKPOINT_EPOCH = 7
EXPECTED_CLOSE_SPEC = "close-trend-vol"
EXPECTED_OHLCV_SPEC = "geometry-level"
PROBABILITY_MIXTURE_BOUNDS = (0.0, 1.0)
LOG_RATIO_BIAS_BOUNDS = (0.0, 4.0)


class AuditedMarketCloseCache(MarketCloseCache):
    def __init__(
        self,
        history_root: Path,
        maximum_days: int,
        opened: set[Path],
    ) -> None:
        super().__init__(history_root, maximum_days)
        self.opened = opened

    def load_day(self, date_value: str) -> np.ndarray:
        self.opened.add(
            (self.history_root / f"{date_value}.json").resolve()
        )
        return super().load_day(date_value)


class AuditedOracleTargetCache(OracleTargetCache):
    def __init__(self, *args, opened: set[Path], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opened = opened

    def load(self, file: Path) -> torch.Tensor:
        self.opened.add(file.resolve())
        return super().load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = audit_v18_ohlcv_fusion(
        arguments.plan,
        requested_device=arguments.device,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


def audit_v18_ohlcv_fusion(
    plan_file: Path,
    *,
    requested_device: str = "auto",
    batch_size: int | None = None,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    storage = training_storage_layout(repo_root)
    resolved_plan_file = resolve(repo_root, plan_file).resolve()
    plan = json.loads(resolved_plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    if plan.get("testPolicy", "sealed-never-load") != "sealed-never-load":
        raise ValueError("OHLCV fusion audit requires sealed-never-load test policy")
    model_config = plan["model"]
    training = plan["training"]
    resolved_training = resolve_training_config(training)
    if not bool(resolved_training.get("policyOnly", False)):
        raise ValueError("v18 OHLCV fusion requires policy-only training")
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
    minute_history_root = history_root.parent / "1m"
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        storage.runs,
        "runDir",
    )

    # Filenames define the already-established chronological suffix split.  No
    # held-out JSON reference is read to derive it.
    target_files = sorted(target_root.glob("*.json"))
    segments = split_and_purge(target_files)
    counts = {
        split: sum(segment.count for segment in segments[split])
        for split in ("train", "validation", "test")
    }
    test_target_files = {
        segment.target_file.resolve() for segment in segments["test"]
    }
    test_day_names = {
        segment.target_file.stem for segment in segments["test"]
    }

    access = AccessLog(set(), set())
    candle_cache: dict[str, np.ndarray] = {}
    print(
        "Fitting frozen OHLCV geometry/backoff from train and materializing "
        "strictly completed validation candles; test remains sealed.",
        file=sys.stderr,
        flush=True,
    )
    train_features, train_targets = load_split(
        "train",
        segments["train"],
        minute_history_root,
        candle_cache,
        access,
    )
    validation_features, validation_targets = load_split(
        "validation",
        segments["validation"],
        minute_history_root,
        candle_cache,
        access,
    )
    if access.target_files & test_target_files:
        raise RuntimeError("sealed test target reference or payload was opened")
    if test_day_names & {path.stem for path in access.candle_files}:
        raise RuntimeError("sealed test one-minute candle reference was opened")

    selected, train_selection = select_each_family(
        train_features,
        train_targets,
    )
    close_selected = selected["close-only"]
    ohlcv_selected = selected["ohlcv-only"]
    if close_selected[0].name != EXPECTED_CLOSE_SPEC \
            or ohlcv_selected[0].name != EXPECTED_OHLCV_SPEC:
        raise RuntimeError(
            "train-selected OHLCV audit tuple changed; review before fusion"
        )
    backoff_selection = select_stacked_mixture(
        train_features,
        train_targets,
        close_selected,
        ohlcv_selected,
    )
    train_prior = normalized_mean(train_targets)
    close_edges, close_table, close_counts = fitted_regime(
        *close_selected,
        train_features,
        train_targets,
        train_prior,
    )
    ohlcv_edges, ohlcv_table, ohlcv_counts = fitted_regime(
        *ohlcv_selected,
        train_features,
        train_targets,
        train_prior,
    )
    validation_close_ids, _ = cell_ids(
        validation_features,
        close_selected[0],
        close_edges,
    )
    validation_ohlcv_ids, _ = cell_ids(
        validation_features,
        ohlcv_selected[0],
        ohlcv_edges,
    )
    close_probabilities = close_table[validation_close_ids]
    geometry_probabilities = ohlcv_table[validation_ohlcv_ids]
    backoff_probabilities = convex_probability_fusion(
        close_probabilities,
        geometry_probabilities,
        float(backoff_selection["ohlcvWeight"]),
    )

    checkpoint_file = run_dir / "checkpoints" / "best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
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
        raise ValueError(
            "expected preserved v18 epoch-7 best checkpoint, got epoch "
            f"{checkpoint['epoch']}"
        )

    device = resolve_device(requested_device, str(training["device"]))
    evaluation_batch_size = (
        int(training["evaluationBatchSize"])
        if batch_size is None
        else int(batch_size)
    )
    if evaluation_batch_size < 1:
        raise ValueError("batch size must be positive")
    dataset = CausalOracleDataset(
        history_root,
        segments,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
        target_rows_per_file=1_440,
        action_count=int(model_config["actionCount"]),
        close_cache_days=int(training.get("closeCacheDays", 10)),
        target_cache_days=int(training.get("targetCacheDays", 3)),
        pin_memory=device.type == "cuda",
        include_future_closes=False,
    )
    inference_history_files: set[Path] = set()
    inference_target_files: set[Path] = set()
    # CausalOracleDataset has not loaded anything during construction.  Swap in
    # audited instances before the first iterator is requested so the access
    # report covers every reference opened by v18 inference itself.
    dataset.close_cache = AuditedMarketCloseCache(
        history_root,
        int(training.get("closeCacheDays", 10)),
        inference_history_files,
    )
    dataset.target_cache = AuditedOracleTargetCache(
        int(training.get("targetCacheDays", 3)),
        rows_per_file=1_440,
        action_count=int(model_config["actionCount"]),
        pin_memory=device.type == "cuda",
        opened=inference_target_files,
    )
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(
        "Running frozen v18 epoch-7 checkpoint on validation only; no test "
        "reference or payload is opened.",
        file=sys.stderr,
        flush=True,
    )
    logits, inference_targets = collect_policy_rows(
        model,
        dataset,
        "validation",
        evaluation_batch_size,
        device,
        training,
    )
    del model, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if logits.shape != (counts["validation"], ACTION_COUNT) \
            or inference_targets.shape != logits.shape \
            or validation_targets.shape != logits.shape \
            or close_probabilities.shape != logits.shape \
            or geometry_probabilities.shape != logits.shape \
            or backoff_probabilities.shape != logits.shape:
        raise RuntimeError("v18/OHLCV fusion rows are not aligned")
    if not np.array_equal(inference_targets, validation_targets):
        raise RuntimeError("v18 inference target order differs from OHLCV order")
    expected_validation_target_files = {
        segment.target_file.resolve()
        for segment in segments["validation"]
    }
    if inference_target_files != expected_validation_target_files:
        raise RuntimeError("v18 inference opened unexpected target references")
    if inference_target_files & test_target_files:
        raise RuntimeError("v18 inference opened a sealed test target")
    if test_day_names & {path.stem for path in inference_history_files}:
        raise RuntimeError("v18 inference opened a sealed test candle")
    probabilities = softmax(logits)
    split_at = probabilities.shape[0] // 2
    if split_at < 1 or split_at == probabilities.shape[0]:
        raise RuntimeError("validation fusion split is empty")

    probability_weight, probability_fit_kl = select_scalar(
        lambda value: mean_kl(
            validation_targets[:split_at],
            convex_probability_fusion(
                probabilities[:split_at],
                backoff_probabilities[:split_at],
                value,
            ),
        ),
        *PROBABILITY_MIXTURE_BOUNDS,
    )
    log_ratio_weight, log_ratio_fit_kl = select_scalar(
        lambda value: mean_kl(
            validation_targets[:split_at],
            log_ratio_feature_fusion(
                probabilities[:split_at],
                backoff_probabilities[:split_at],
                close_probabilities[:split_at],
                value,
            ),
        ),
        *LOG_RATIO_BIAS_BOUNDS,
    )
    probability_fused = convex_probability_fusion(
        probabilities,
        backoff_probabilities,
        probability_weight,
    )
    log_ratio_fused = log_ratio_feature_fusion(
        probabilities,
        backoff_probabilities,
        close_probabilities,
        log_ratio_weight,
    )

    expected_target_files = {
        segment.target_file.resolve()
        for split in ("train", "validation")
        for segment in segments[split]
    }
    if access.target_files != expected_target_files:
        raise RuntimeError("unexpected OHLCV target-reference access set")
    timestamps = validation_timestamps(segments["validation"])
    v18_metrics = split_metrics(validation_targets, probabilities, split_at)
    probability_metrics = split_metrics(
        validation_targets,
        probability_fused,
        split_at,
    )
    log_ratio_metrics = split_metrics(
        validation_targets,
        log_ratio_fused,
        split_at,
    )
    return {
        "schemaVersion": 1,
        "audit": "preserved-v18-plus-train-selected-causal-ohlcv-fusion",
        "accessContract": {
            "trainTargetReferenceFilesOpened": len({
                segment.target_file.resolve() for segment in segments["train"]
            }),
            "validationTargetReferenceFilesOpened": len({
                segment.target_file.resolve()
                for segment in segments["validation"]
            }),
            "causalOneMinuteCandleReferencesOpened": len(access.candle_files),
            "causalOneSecondCandleReferencesOpenedByV18": len(
                inference_history_files
            ),
            "testTargetReferenceMetadataOpened": 0,
            "testTargetPayloadsOpened": 0,
            "testCandleReferenceMetadataOpened": 0,
            "testCandlePayloadsOpened": 0,
            "modelWeightsChanged": False,
            "trainingStarted": False,
            "inferenceDevice": str(device),
        },
        "checkpoint": {
            "planId": checkpoint["planId"],
            "kind": "best",
            "epoch": int(checkpoint["epoch"]),
            "globalStep": int(checkpoint["globalStep"]),
            "file": str(checkpoint_file),
            "reportedRawValidationKl": float(
                checkpoint["validation"]["klDivergence"]
            ),
        },
        "ohlcvEstimator": {
            "featureCount": len(FEATURE_NAMES),
            "closeControlSpec": close_selected[0].name,
            "closeControlFeatures": list(close_selected[0].features),
            "closeControlBinCounts": list(close_selected[0].bin_counts),
            "closeControlPriorStrength": close_selected[1],
            "geometrySpec": ohlcv_selected[0].name,
            "geometryFeatures": list(ohlcv_selected[0].features),
            "geometryBinCounts": list(ohlcv_selected[0].bin_counts),
            "geometryPriorStrength": ohlcv_selected[1],
            "closeBackoffWeight": backoff_selection["closeWeight"],
            "geometryBackoffWeight": backoff_selection["ohlcvWeight"],
            "trainInternalBackoffCalibrationKl": (
                backoff_selection["calibrationKl"]
            ),
            "trainRows": train_targets.shape[0],
            "validationRowsUsedForEstimatorSelection": 0,
            "validationRowsWithEmptyCloseCell": int(np.count_nonzero(
                close_counts[validation_close_ids] == 0
            )),
            "validationRowsWithEmptyGeometryCell": int(np.count_nonzero(
                ohlcv_counts[validation_ohlcv_ids] == 0
            )),
            "trainSelection": train_selection["selected"],
        },
        "validationSplit": {
            "rows": counts["validation"],
            "scalarFitRows": split_at,
            "fusionHoldoutRows": counts["validation"] - split_at,
            "scalarFitTimestampStart": int(timestamps[0]),
            "scalarFitTimestampEnd": int(timestamps[split_at - 1]),
            "fusionHoldoutTimestampStart": int(timestamps[split_at]),
            "fusionHoldoutTimestampEnd": int(timestamps[-1]),
        },
        "raw01Kl": {
            "v18": v18_metrics,
            "matchedCloseControl": split_metrics(
                validation_targets,
                close_probabilities,
                split_at,
            ),
            "geometryOnly": split_metrics(
                validation_targets,
                geometry_probabilities,
                split_at,
            ),
            "geometryWithTrainSelectedCloseBackoff": split_metrics(
                validation_targets,
                backoff_probabilities,
                split_at,
            ),
            "convexProbabilityMixture": fusion_metrics(
                probability_weight,
                probability_fit_kl,
                probability_metrics,
                v18_metrics,
            ),
            "logRatioGeometryBiasRelativeToCloseControl": fusion_metrics(
                log_ratio_weight,
                log_ratio_fit_kl,
                log_ratio_metrics,
                v18_metrics,
            ),
        },
        "selectionContract": {
            "ohlcvGroupingShrinkageAndBackoffSelectedOnTrainOnly": True,
            "fusionScalarSelectedOnFirstChronologicalValidationHalfOnly": True,
            "secondValidationHalfUsedForScalarSelection": False,
            "fullValidationReportedOnlyAsDescriptiveMetric": True,
            "exactCompletedMinuteAlignment": True,
            "testUsed": False,
        },
    }


def log_ratio_feature_fusion(
    base: np.ndarray,
    feature_probability: np.ndarray,
    matched_control: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Apply only the feature estimator's evidence beyond its close control."""
    if base.shape != feature_probability.shape \
            or base.shape != matched_control.shape \
            or base.ndim != 2 \
            or not math.isfinite(weight):
        raise ValueError("log-ratio feature fusion inputs are incompatible")
    tiny = np.finfo(np.float64).tiny
    logits = (
        np.log(np.clip(base, tiny, None))
        + weight * (
            np.log(np.clip(feature_probability, tiny, None))
            - np.log(np.clip(matched_control, tiny, None))
        )
    )
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    return result


def validation_timestamps(segments: list) -> np.ndarray:
    parts: list[np.ndarray] = []
    for segment in segments:
        if segment.step_ms != 60_000:
            raise ValueError("fusion audit requires one-minute validation rows")
        parts.append(
            segment.prediction_time_start
            + np.arange(segment.count, dtype=np.int64) * segment.step_ms
        )
    if not parts:
        raise ValueError("validation timestamp split is empty")
    result = np.concatenate(parts)
    if result.size > 1 and bool((np.diff(result) <= 0).any()):
        raise ValueError("validation timestamps must be strictly chronological")
    return result


def fusion_metrics(
    selected_weight: float,
    first_half_selection_kl: float,
    metrics: dict[str, float],
    v18_metrics: dict[str, float],
) -> dict[str, float | bool]:
    return {
        "selectedWeight": selected_weight,
        "firstHalfSelectionKl": first_half_selection_kl,
        **metrics,
        "secondHalfKlReductionFromV18": (
            v18_metrics["secondHalfKl"] - metrics["secondHalfKl"]
        ),
        "fullValidationKlReductionFromV18": (
            v18_metrics["fullValidationKl"] - metrics["fullValidationKl"]
        ),
        "improvesUntouchedSecondHalf": (
            metrics["secondHalfKl"] < v18_metrics["secondHalfKl"]
        ),
    }


if __name__ == "__main__":
    main()
