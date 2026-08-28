from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch

from structured_feature_process import StructuredSharedIoFeatureProcess
from trading_storage import load_torch_checkpoint
from train_normalized_glu_next_return import atomic_json
from train_structured_feature_process import (
    build_dataset,
    comparable_policy_metrics,
    evaluate,
    persist_structured_feature_evaluation,
    validate_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every durable checkpoint policy from a completed "
            "structured shared-IO feature-process run."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def model_from_state(
    plan: dict, state: dict[str, torch.Tensor], device: torch.device
) -> StructuredSharedIoFeatureProcess:
    architecture = plan["architecture"]
    model = StructuredSharedIoFeatureProcess(
        state["input_mean"],
        state["input_std"],
        state["output_mean"],
        state["output_std"],
        input_steps=int(architecture["inputSteps"]),
        output_steps=int(architecture["outputSteps"]),
        feature_width=int(architecture["featureWidth"]),
        market_width=int(architecture["marketWidth"]),
        prefix_width=int(architecture["prefixWidth"]),
        feature_distribution_width=int(architecture["featureDistributionWidth"]),
        extended_prefix_width=int(architecture["extendedPrefixWidth"]),
        next_feature_distribution_width=int(
            architecture["nextFeatureDistributionWidth"]
        ),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture["learnableCentering"]),
        linear_rank=(
            None
            if architecture.get("linearFactorization") is None
            else int(architecture["linearFactorization"]["rank"])
        ),
        layer8_attention=architecture.get("layer8Attention"),
        layer8_function_approximator=architecture.get(
            "layer8FunctionApproximator"
        ),
        recurrent_memory=architecture.get("recurrentMemory"),
    )
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def main() -> None:
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    run_root = (repo / plan["runDir"]).resolve()
    status_file = run_root / "state/status.json"
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if status.get("stage") == "training":
        raise ValueError("refusing to evaluate a run still marked as training")
    dataset = build_dataset(plan, repo)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    batch_size = int(args.batch_size or plan["training"]["evaluationBatchSize"])
    if hasattr(dataset, "rollout"):
        dataset.statistics(batch_size)
    checkpoint_specs = (
        (
            "best-validation-mse",
            run_root / "checkpoints/selections/validation-mse.json",
            "model",
        ),
        (
            "best-validation-correlation",
            run_root / "checkpoints/selections/validation-correlation.json",
            "model",
        ),
        ("last", run_root / "checkpoints/last.json", "emaModel"),
    )
    policies: dict[str, dict] = {}
    started = time.monotonic()
    parameter_count = 0
    for label, checkpoint_file, state_key in checkpoint_specs:
        checkpoint = load_torch_checkpoint(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        model = model_from_state(plan, checkpoint[state_key], device)
        parameter_count = sum(value.numel() for value in model.parameters())
        selection_score = checkpoint.get("score")
        evaluations = {
            split: evaluate(
                model, dataset, split, batch_size=batch_size, device=device
            )
            for split in ("train", "validation", "test")
        }
        policies[label] = {
            "epoch": int(checkpoint["epoch"]),
            "selectionScore": (
                None if selection_score is None else float(selection_score)
            ),
            "selectionMetric": (
                "featureState.correlation"
                if label == "best-validation-correlation"
                else "featureState.normalizedMse"
                if label == "best-validation-mse"
                else None
            ),
            "checkpointPolicy": label,
            "checkpoint": str(checkpoint_file.relative_to(repo)).replace("\\", "/"),
            **comparable_policy_metrics(evaluations),
        }
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    artifact, result = persist_structured_feature_evaluation(
        repo=repo,
        run_root=run_root,
        plan=plan,
        dataset=dataset,
        parameter_count=parameter_count,
        policies=policies,
        generated_at=generated_at,
    )
    status.pop("pid", None)
    status.update({
        "stage": "complete",
        "updatedAt": generated_at,
        "evaluatedAt": generated_at,
        "message": (
            "All durable structured feature-process checkpoints were "
            "evaluated on full train, validation, and untouched test splits."
        ),
        "selectionPolicy": result["selectionPolicy"],
        "completedAfterEpoch": artifact["completedAfterEpoch"],
        "result": result,
        "evaluation": {
            "artifact": result["evaluationArtifact"],
            "bestEpoch": result["bestEpoch"],
            "bestTest": result["test"],
            "lastTest": policies["last"]["test"],
            "seconds": time.monotonic() - started,
        },
    })
    atomic_json(status, status_file)
    dataset.close()
    print(json.dumps({
        "planId": plan["id"],
        "selectionPolicy": result["selectionPolicy"],
        "bestEpoch": result["bestEpoch"],
        "bestValidation": result["validation"],
        "bestTest": result["test"],
        "lastTest": policies["last"]["test"],
        "policies": sorted(policies),
        "evaluationArtifact": result["evaluationArtifact"],
        "seconds": time.monotonic() - started,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
