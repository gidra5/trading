from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch

from normalized_glu_next_return import NormalizedGluNextReturn
from trading_storage import load_torch_checkpoint
from train_feature_augmented_next_return import (
    FeatureMatrixDataset,
    evaluate,
    validate_plan,
)
from train_normalized_glu_next_return import atomic_json


CONTRACT = "stopped-feature-union-one-step-evaluation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every durable selection checkpoint from a stopped "
            "temporal feature-regression run."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model(
    plan: dict,
    checkpoint: dict,
    device: torch.device,
) -> NormalizedGluNextReturn:
    state = checkpoint["model"]
    architecture = plan["architecture"]
    model = NormalizedGluNextReturn(
        state["feature_mean"],
        state["feature_std"],
        state["target_mean"],
        state["target_std"],
        widths=tuple(int(value) for value in architecture["widths"]),
        learnable_centering=bool(architecture["learnableCentering"]),
        dropout=0,
        dropout_rate=0,
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
    )
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def main() -> None:
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    plan_file = (
        (repo / args.plan).resolve()
        if not args.plan.is_absolute()
        else args.plan.resolve()
    )
    plan = json.loads(plan_file.read_text("utf-8"))
    validate_plan(plan)
    run_root = (repo / plan["runDir"]).resolve()
    status_file = run_root / "state/status.json"
    status = json.loads(status_file.read_text("utf-8"))
    if status.get("stage") == "training":
        raise ValueError("refusing to evaluate a run still marked as training")

    union_history_dir = plan.get("unionHistoryDatasetDir")
    dataset = FeatureMatrixDataset(
        (repo / plan["datasetDir"]).resolve(),
        selection=plan.get("datasetSelection"),
        examples_by_split=plan.get("examplesBySplit"),
        union_history_root=(
            (repo / union_history_dir).resolve()
            if union_history_dir is not None else None
        ),
        feature_history_seconds=plan.get("featureHistorySeconds"),
    )
    device = torch.device(args.device)
    batch_size = int(
        args.batch_size or plan["training"]["evaluationBatchSize"]
    )
    checkpoint_specs = (
        (
            "best-validation-mse", "validation-mse",
            "best-validation-mse.json", "validation", "normalizedMse",
        ),
        (
            "best-validation-correlation", "validation-correlation",
            "best-validation-correlation.json", "validation", "correlation",
        ),
        (
            "best-train-mse", "train-mse",
            "best-train-mse.json", "train", "normalizedMse",
        ),
        (
            "best-train-correlation", "train-correlation",
            "best-train-correlation.json", "train", "correlation",
        ),
        ("stopped-last", "stopped-last", "last.json", None, None),
    )
    policies: dict[str, dict] = {}
    comparison_policies: dict[str, dict] = {}
    started = time.monotonic()
    for label, comparison_label, filename, split, metric in checkpoint_specs:
        checkpoint_file = run_root / "checkpoints" / filename
        if not checkpoint_file.is_file():
            continue
        checkpoint = load_torch_checkpoint(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        model = load_model(plan, checkpoint, device)
        target_std = float(checkpoint["model"]["target_std"])
        test_metrics = evaluate(
            model,
            dataset,
            "test",
            batch_size=batch_size,
            target_std=target_std,
            device=device,
        )
        policy = {
            "epoch": int(checkpoint["epoch"]),
            "train": checkpoint["train"],
            "validation": checkpoint["validation"],
            "test": test_metrics,
            "checkpointPolicy": label,
            "checkpoint": str(checkpoint_file.relative_to(repo)),
        }
        policy["selectionScore"] = (
            float(policy[split][metric])
            if split is not None and metric is not None else None
        )
        policies[label] = policy
        comparison_policies[comparison_label] = policy
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best = policies["best-validation-mse"]
    stopped = policies["stopped-last"]
    generated_at = utc_now()
    output_file = (
        (repo / args.output).resolve()
        if args.output is not None and not args.output.is_absolute()
        else args.output.resolve()
        if args.output is not None
        else (
            repo
            / "data/benchmarks"
            / f'{plan["id"]}-stopped-eval.json'
        ).resolve()
    )
    artifact = {
        "contract": CONTRACT,
        "generatedAt": generated_at,
        "planId": plan["id"],
        "selectionPolicy": "best-validation-mse",
        "selectionDoesNotUseTest": True,
        "stoppedAfterEpoch": stopped["epoch"],
        "examplesBySplit": {
            split: dataset.logical_count(split)
            for split in ("train", "validation", "test")
        },
        "featureCount": dataset.feature_count,
        "baseFeatureCount": dataset.manifest.get("baseFeatureCount"),
        "historySeconds": dataset.manifest.get("featureHistorySeconds"),
        "parameterCount": int(status["latest"]["parameterCount"]),
        "policies": policies,
        "seconds": time.monotonic() - started,
    }
    atomic_json(artifact, output_file)
    atomic_json(artifact, run_root / "state/stopped-evaluation.json")

    result = {
        "contract": "stopped-feature-union-one-step-evaluation-dashboard-result-v1",
        "planId": plan["id"],
        "selectionPolicy": "best-validation-mse",
        "selectionDoesNotUseTest": True,
        "examples": dataset.logical_count("train"),
        "examplesBySplit": artifact["examplesBySplit"],
        "featureCount": dataset.feature_count,
        "parameterCount": artifact["parameterCount"],
        "trainableParameterCount": artifact["parameterCount"],
        "bestEpoch": best["epoch"],
        "stoppedAfterEpoch": stopped["epoch"],
        "bestValidationScore": best["validation"]["normalizedMse"],
        "train": best["train"],
        "validation": best["validation"],
        "test": best["test"],
        "evaluationArtifact": str(output_file.relative_to(repo)).replace("\\", "/"),
    }
    atomic_json(result, run_root / "state/result.json")
    atomic_json(
        {
            "contract": "next-return-checkpoint-selection-comparison-v1",
            "policies": comparison_policies,
        },
        run_root / "state/checkpoint-selection-comparison.json",
    )
    status.pop("pid", None)
    status.update(
        {
            "stage": "complete",
            "updatedAt": generated_at,
            "evaluatedAt": generated_at,
            "message": (
                "Stopped run evaluated on the untouched one-step test split "
                "for every durable checkpoint policy; training logs preserved."
            ),
            "selectionPolicy": "best-validation-mse",
            "stoppedAfterEpoch": stopped["epoch"],
            "evaluation": {
                "artifact": result["evaluationArtifact"],
                "bestEpoch": best["epoch"],
                "bestTest": best["test"],
                "stoppedLastTest": stopped["test"],
            },
        }
    )
    atomic_json(status, status_file)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
