from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from trading_storage import load_torch_checkpoint
from train_normalized_glu_next_return import atomic_json
from train_structured_feature_process import (
    COMPARABLE_EVALUATION_SCOPE,
    FEATURE_STATE_EVALUATION_SCOPE,
    StructuredFeatureSequenceDataset,
    compact_feature_state_metrics,
    validate_plan,
)


ARCHIVE_NAME = "training.complete-feature-state-original.jsonl"
BACKFILL_CONTRACT = "structured-feature-comparable-log-backfill-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill structured feature-process epoch rows with comparable "
            "next-return metrics while preserving the original log exactly."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def sha256_file(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def target_stats(
    dataset: StructuredFeatureSequenceDataset,
    split: str,
    *,
    limit: int | None,
) -> dict[str, float | int]:
    count = dataset.counts[split]
    if limit is not None:
        count = min(count, int(limit))
    total = 0.0
    square = 0.0
    values = 0
    for start in range(0, count, 65536):
        logical = np.arange(start, min(count, start + 65536), dtype=np.int64)
        _, targets = dataset._examples(split, logical)
        channel = targets[..., 0].astype(np.float64, copy=False)
        total += float(channel.sum(dtype=np.float64))
        square += float(np.square(channel).sum(dtype=np.float64))
        values += int(channel.size)
    mean = total / values
    variance = max(0.0, square / values - mean**2)
    return {
        "examples": values,
        "targetMean": mean,
        "targetStd": math.sqrt(variance),
        "zeroBaselineMse": square / values,
    }


def comparable_metrics(
    feature_state: dict,
    *,
    output_std: float,
    target: dict[str, float | int],
) -> dict:
    normalized_values = feature_state.get("perFeatureNormalizedMse")
    correlation_values = feature_state.get("perFeatureCorrelation")
    if not isinstance(normalized_values, list) or not normalized_values:
        raise ValueError("epoch row lacks per-feature normalized MSE")
    if not isinstance(correlation_values, list) or not correlation_values:
        raise ValueError("epoch row lacks per-feature correlation")
    normalized_mse = float(normalized_values[0])
    correlation = float(correlation_values[0])
    mse = normalized_mse * output_std**2
    zero_mse = float(target["zeroBaselineMse"])
    return {
        "examples": int(feature_state["examples"]),
        "normalizedMse": normalized_mse,
        "mse": mse,
        "rmse": math.sqrt(max(0.0, mse)),
        "correlation": correlation,
        "targetMean": float(target["targetMean"]),
        "targetStd": float(target["targetStd"]),
        "zeroBaselineMse": zero_mse,
        "mseSkillVsZero": 1.0 - mse / zero_mse if zero_mse > 0 else 0.0,
        "channelIndex": 0,
        "channelId": "return-lag-0s",
        "evaluationScope": COMPARABLE_EVALUATION_SCOPE,
        "historicalMetricAvailability": (
            "mse/correlation reconstructed exactly from the original "
            "per-channel sufficient metrics; direction/MAE/prediction moments "
            "were not recorded per channel"
        ),
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    architecture = plan["architecture"]
    if int(architecture["outputSteps"]) != 1:
        raise ValueError("the comparable backfill currently requires one output step")
    run_root = (repo / plan["runDir"]).resolve()
    status = json.loads((run_root / "state/status.json").read_text(encoding="utf-8"))
    if status.get("stage") not in {"complete", "stopped", "failed"}:
        raise ValueError("refusing to rewrite the active log of a non-terminal run")

    active_log = run_root / "logs/training.jsonl"
    archive_log = run_root / "logs" / ARCHIVE_NAME
    if not active_log.is_file():
        raise FileNotFoundError(active_log)
    if not archive_log.exists():
        shutil.copyfile(active_log, archive_log)
        if sha256_file(active_log) != sha256_file(archive_log):
            archive_log.unlink(missing_ok=True)
            raise RuntimeError("original training log archive verification failed")
    source_hash = sha256_file(archive_log)

    checkpoint = load_torch_checkpoint(
        run_root / "checkpoints/last.json", map_location="cpu", weights_only=False
    )
    state = checkpoint["emaModel"]
    output_std = float(state["output_std"][0])
    if not math.isfinite(output_std) or output_std <= 0:
        raise ValueError("invalid return-channel training standard deviation")

    dataset = StructuredFeatureSequenceDataset(
        (repo / plan["datasetDir"]).resolve(),
        input_steps=int(architecture["inputSteps"]),
        output_steps=int(architecture["outputSteps"]),
        train_examples=int(plan["subset"]["examples"]),
    )
    try:
        targets = {
            "train": target_stats(
                dataset,
                "train",
                limit=int(plan["training"]["epochTrainEvaluationExamples"]),
            ),
            "validation": target_stats(dataset, "validation", limit=None),
        }
    finally:
        dataset.close()

    output_lines: list[str] = []
    epoch_rows = 0
    with archive_log.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid original JSONL at line {line_number}: {error}"
                ) from error
            if event.get("event") == "minute-return-epoch":
                original_train = event.get("train")
                original_validation = event.get("validation")
                if not isinstance(original_train, dict) or not isinstance(
                    original_validation, dict
                ):
                    raise ValueError(f"epoch row {line_number} lacks metrics")
                event["featureState"] = {
                    "evaluationScope": FEATURE_STATE_EVALUATION_SCOPE,
                    "train": compact_feature_state_metrics(original_train),
                    "validation": compact_feature_state_metrics(
                        original_validation
                    ),
                }
                event["train"] = comparable_metrics(
                    original_train, output_std=output_std, target=targets["train"]
                )
                event["validation"] = comparable_metrics(
                    original_validation,
                    output_std=output_std,
                    target=targets["validation"],
                )
                event["evaluationScope"] = COMPARABLE_EVALUATION_SCOPE
                event["checkpointSelectionScope"] = FEATURE_STATE_EVALUATION_SCOPE
                if "bestValidationMse" in event:
                    event["bestFeatureStateValidationMse"] = event.pop(
                        "bestValidationMse"
                    )
                if "bestValidationCorrelation" in event:
                    event["bestFeatureStateValidationCorrelation"] = event.pop(
                        "bestValidationCorrelation"
                    )
                event["metricsBackfilledBy"] = BACKFILL_CONTRACT
                epoch_rows += 1
            output_lines.append(json.dumps(event, separators=(",", ":")))

    if epoch_rows == 0:
        raise ValueError("original training log contains no epoch rows")
    temporary = active_log.with_suffix(active_log.suffix + ".comparable.tmp")
    temporary.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    transformed_hash = sha256_file(temporary)
    os.replace(temporary, active_log)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    audit = {
        "contract": BACKFILL_CONTRACT,
        "generatedAt": generated_at,
        "planId": plan["id"],
        "headlineEvaluationScope": COMPARABLE_EVALUATION_SCOPE,
        "featureStateEvaluationScope": FEATURE_STATE_EVALUATION_SCOPE,
        "epochRows": epoch_rows,
        "originalLog": str(archive_log.relative_to(repo)).replace("\\", "/"),
        "activeLog": str(active_log.relative_to(repo)).replace("\\", "/"),
        "originalSha256": source_hash,
        "activeSha256": transformed_hash,
        "returnChannelTrainingStd": output_std,
        "targetStatistics": targets,
    }
    atomic_json(audit, run_root / "state/comparable-metrics-backfill.json")
    print(json.dumps(audit, separators=(",", ":")))


if __name__ == "__main__":
    main()
