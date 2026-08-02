"""Leakage-safe diagnostic fusion of v18 with a train-only calendar prior.

Unlike the general checkpoint evaluator, this audit deliberately does not call
``load_causal_segments`` because that function validates every reference file,
including the sealed test references.  Split boundaries are derived from names,
only train/validation targets are opened, and inference runs on validation only.

The first chronological half of validation selects one scalar independently for
a convex probability mixture and a log-ratio calendar bias.  The second half is
the untouched fusion holdout.  No model weights are changed and no training is
performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

from audit_causal_oracle_predictability import mean_kl, normalized_mean
from audit_oracle_calendar_predictability import (
    ACTION_COUNT,
    CalendarSelection,
    CALENDAR_SPECS,
    calendar_rows_for_segments,
    encode_cells,
    fit_calendar_table,
    load_train_validation_targets,
    split_and_purge,
)
from evaluate_joint_price_oracle_actions import (
    collect_policy_rows,
    resolve_device,
)
from joint_price_oracle import parameter_count
from trading_storage import (
    load_torch_checkpoint,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    CausalOracleDataset,
    DATA_CONTRACT,
    architecture_contract_for_model_config,
    build_model,
    configuration_fingerprint,
    resolve,
    resolve_training_config,
    validate_plan,
)


DEFAULT_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
)
CALENDAR_SPEC_NAME = "hour-by-weekday"
CALENDAR_FINE_PRIOR_STRENGTH = 16.0
CALENDAR_BACKOFF_PRIOR_STRENGTH = 1_024.0
PROBABILITY_MIXTURE_BOUNDS = (0.0, 1.0)
LOG_RATIO_BIAS_BOUNDS = (0.0, 2.0)
SCALAR_SEARCH_ITERATIONS = 48


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
    report = audit_v18_calendar_fusion(
        arguments.plan,
        requested_device=arguments.device,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


def audit_v18_calendar_fusion(
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
    if plan.get("testPolicy", "sealed-never-load") \
            != "sealed-never-load":
        raise ValueError("fusion audit requires sealed-never-load test policy")
    model_config = plan["model"]
    training = plan["training"]
    resolved_training = resolve_training_config(training)
    if not bool(resolved_training.get("policyOnly", False)):
        raise ValueError("v18 fusion audit requires policy-only training")
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

    # Listing filenames establishes the existing chronological boundary without
    # reading any JSON reference in the held-out test suffix.
    target_files = sorted(target_root.glob("*.json"))
    segments = split_and_purge(target_files)
    train_validation_targets, opened = load_train_validation_targets(segments)
    test_files = {
        segment.target_file.resolve() for segment in segments["test"]
    }
    if opened & test_files:
        raise RuntimeError("sealed test target was opened")
    counts = {
        split: sum(segment.count for segment in segments[split])
        for split in ("train", "validation", "test")
    }

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

    calendar = {
        split: calendar_rows_for_segments(segments[split])
        for split in ("train", "validation")
    }
    train_prior = normalized_mean(train_validation_targets["train"])
    calendar_spec = next(
        spec for spec in CALENDAR_SPECS if spec.name == CALENDAR_SPEC_NAME
    )
    calendar_selection = CalendarSelection(
        spec=calendar_spec,
        fine_prior_strength=CALENDAR_FINE_PRIOR_STRENGTH,
        backoff_prior_strength=CALENDAR_BACKOFF_PRIOR_STRENGTH,
        calibration_kl=math.nan,
    )
    calendar_table = fit_calendar_table(
        calendar["train"].values,
        train_validation_targets["train"],
        calendar_selection.spec,
        calendar_selection.fine_prior_strength,
        calendar_selection.backoff_prior_strength,
        train_prior,
    )
    validation_calendar_ids, _ = encode_cells(
        calendar["validation"].values,
        calendar_selection.spec.fields,
    )
    calendar_probabilities = calendar_table[validation_calendar_ids]

    device = resolve_device(requested_device, str(training["device"]))
    evaluation_batch_size = (
        int(training["evaluationBatchSize"])
        if batch_size is None else int(batch_size)
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
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(
        "Running preserved v18 best checkpoint on validation only; no test "
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

    expected_targets = train_validation_targets["validation"]
    if logits.shape != (counts["validation"], ACTION_COUNT) \
            or inference_targets.shape != logits.shape \
            or calendar_probabilities.shape != logits.shape:
        raise RuntimeError("fusion rows are not aligned")
    if not np.array_equal(inference_targets, expected_targets):
        if not np.allclose(inference_targets, expected_targets, atol=0, rtol=0):
            raise RuntimeError("v18 inference target order differs from audit order")
    probabilities = softmax(logits)
    split_at = probabilities.shape[0] // 2
    if split_at < 1 or split_at == probabilities.shape[0]:
        raise RuntimeError("validation fusion split is empty")

    probability_weight, probability_fit_kl = select_scalar(
        lambda value: mean_kl(
            expected_targets[:split_at],
            convex_probability_fusion(
                probabilities[:split_at],
                calendar_probabilities[:split_at],
                value,
            ),
        ),
        *PROBABILITY_MIXTURE_BOUNDS,
    )
    log_ratio_weight, log_ratio_fit_kl = select_scalar(
        lambda value: mean_kl(
            expected_targets[:split_at],
            log_ratio_calendar_fusion(
                probabilities[:split_at],
                calendar_probabilities[:split_at],
                train_prior,
                value,
            ),
        ),
        *LOG_RATIO_BIAS_BOUNDS,
    )
    probability_fused = convex_probability_fusion(
        probabilities,
        calendar_probabilities,
        probability_weight,
    )
    log_ratio_fused = log_ratio_calendar_fusion(
        probabilities,
        calendar_probabilities,
        train_prior,
        log_ratio_weight,
    )

    opened_validation = {
        segment.target_file.resolve() for segment in segments["validation"]
    }
    opened_train = {
        segment.target_file.resolve() for segment in segments["train"]
    }
    if opened != opened_train | opened_validation:
        raise RuntimeError("unexpected target-reference access set")
    return {
        "schemaVersion": 1,
        "audit": "preserved-v18-plus-train-only-calendar-fusion",
        "accessContract": {
            "trainTargetReferenceFilesOpened": len(opened_train),
            "validationTargetReferenceFilesOpened": len(opened_validation),
            "testReferenceFilesOpened": 0,
            "testPayloadsOpened": 0,
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
        "calendar": {
            "spec": calendar_selection.spec.name,
            "fields": list(calendar_selection.spec.fields),
            "backoffFields": list(calendar_selection.spec.backoff_fields),
            "finePriorStrength": calendar_selection.fine_prior_strength,
            "backoffPriorStrength": (
                calendar_selection.backoff_prior_strength
            ),
            "fitRows": counts["train"],
            "fitUsesValidationRows": False,
        },
        "validationSplit": {
            "rows": counts["validation"],
            "scalarFitRows": split_at,
            "fusionHoldoutRows": counts["validation"] - split_at,
            "scalarFitTimestampStart": int(
                calendar["validation"].timestamps_ms[0]
            ),
            "scalarFitTimestampEnd": int(
                calendar["validation"].timestamps_ms[split_at - 1]
            ),
            "fusionHoldoutTimestampStart": int(
                calendar["validation"].timestamps_ms[split_at]
            ),
            "fusionHoldoutTimestampEnd": int(
                calendar["validation"].timestamps_ms[-1]
            ),
        },
        "raw01Kl": {
            "v18": split_metrics(
                expected_targets,
                probabilities,
                split_at,
            ),
            "calendarOnly": split_metrics(
                expected_targets,
                calendar_probabilities,
                split_at,
            ),
            "convexProbabilityMixture": {
                "selectedWeight": probability_weight,
                "firstHalfSelectionKl": probability_fit_kl,
                **split_metrics(
                    expected_targets,
                    probability_fused,
                    split_at,
                ),
            },
            "logRatioCalendarBias": {
                "selectedWeight": log_ratio_weight,
                "firstHalfSelectionKl": log_ratio_fit_kl,
                **split_metrics(
                    expected_targets,
                    log_ratio_fused,
                    split_at,
                ),
            },
        },
        "selectionContract": {
            "calendarGroupingAndShrinkageSelectedOnTrainOnly": True,
            "fusionScalarSelectedOnFirstChronologicalValidationHalfOnly": True,
            "secondValidationHalfUsedForScalarSelection": False,
            "fullValidationReportedOnlyAsDescriptiveMetric": True,
            "testUsed": False,
        },
    }


def validate_preserved_checkpoint_without_test_access(
    checkpoint: dict,
    plan: dict,
    model_config: dict,
    resolved_training: dict,
    counts: dict[str, int],
    run_dir: Path,
) -> None:
    """Validate checkpoint identity using preserved metadata, never test refs."""
    expected = {
        "planId": plan["id"],
        "modelConfig": model_config,
        "parameterCount": parameter_count(build_model(model_config)),
        "architectureContract": architecture_contract_for_model_config(
            model_config
        ),
        "dataContract": DATA_CONTRACT,
        "trainingConfigFingerprint": configuration_fingerprint(
            resolved_training
        ),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"preserved v18 checkpoint {key} mismatch")
    log_file = run_dir / "logs" / "training.jsonl"
    first_event = json.loads(
        log_file.read_text(encoding="utf-8").splitlines()[0]
    )
    if first_event.get("event") != "dataset" \
            or first_event.get("datasetFingerprint") \
            != checkpoint.get("datasetFingerprint") \
            or first_event.get("counts") != counts:
        raise ValueError("preserved v18 dataset metadata mismatch")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("softmax requires finite two-dimensional logits")
    values = values - values.max(axis=1, keepdims=True)
    result = np.exp(values)
    result /= result.sum(axis=1, keepdims=True)
    return result


def convex_probability_fusion(
    base: np.ndarray,
    calendar: np.ndarray,
    weight: float,
) -> np.ndarray:
    if base.shape != calendar.shape or not 0 <= weight <= 1:
        raise ValueError("convex fusion inputs are incompatible")
    result = (1 - weight) * base + weight * calendar
    result /= result.sum(axis=1, keepdims=True)
    return result


def log_ratio_calendar_fusion(
    base: np.ndarray,
    calendar: np.ndarray,
    train_prior: np.ndarray,
    weight: float,
) -> np.ndarray:
    if base.shape != calendar.shape \
            or train_prior.shape != (base.shape[1],) \
            or not math.isfinite(weight):
        raise ValueError("log-ratio fusion inputs are incompatible")
    tiny = np.finfo(np.float64).tiny
    logits = (
        np.log(np.clip(base, tiny, None))
        + weight * (
            np.log(np.clip(calendar, tiny, None))
            - np.log(np.clip(train_prior[None, :], tiny, None))
        )
    )
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    return result


def select_scalar(
    objective: Callable[[float], float],
    low: float,
    high: float,
) -> tuple[float, float]:
    if not math.isfinite(low) or not math.isfinite(high) or not low < high:
        raise ValueError("scalar search bounds are invalid")
    initial_low = low
    initial_high = high
    ratio = (math.sqrt(5) - 1) / 2
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_score = float(objective(left))
    right_score = float(objective(right))
    for _ in range(SCALAR_SEARCH_ITERATIONS):
        if left_score <= right_score:
            high = right
            right = left
            right_score = left_score
            left = high - ratio * (high - low)
            left_score = float(objective(left))
        else:
            low = left
            left = right
            left_score = right_score
            right = low + ratio * (high - low)
            right_score = float(objective(right))
    middle = (low + high) / 2
    candidates = (
        (initial_low, float(objective(initial_low))),
        (middle, float(objective(middle))),
        (initial_high, float(objective(initial_high))),
    )
    result = min(candidates, key=lambda item: (item[1], item[0]))
    if not math.isfinite(result[1]):
        raise ValueError("scalar search objective is non-finite")
    return result


def split_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    split_at: int,
) -> dict[str, float]:
    if targets.shape != probabilities.shape \
            or not 0 < split_at < targets.shape[0]:
        raise ValueError("fusion metric rows are incompatible")
    return {
        "firstHalfKl": mean_kl(targets[:split_at], probabilities[:split_at]),
        "secondHalfKl": mean_kl(targets[split_at:], probabilities[split_at:]),
        "fullValidationKl": mean_kl(targets, probabilities),
    }


if __name__ == "__main__":
    main()
