from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch

from return_knot_density import KnotDensityContract
from structured_feature_process import StructuredSharedIoFeatureProcess
from trading_storage import load_torch_checkpoint
from train_normalized_glu_next_return import atomic_json
from train_structured_feature_process import (
    build_dataset,
    comparable_policy_metrics,
    evaluate,
    evaluate_feature_embedding_density,
    evaluate_structured_return_density,
    persist_embedding_density_evaluation,
    persist_structured_return_density_evaluation,
    persist_structured_feature_evaluation,
    uses_base_support_embedding_density,
    uses_embedding_density_objective,
    uses_structured_return_density_objective,
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
    parser.add_argument(
        "--allow-stopped-training-status",
        action="store_true",
        help=(
            "Evaluate after the exact training process was externally stopped "
            "even if its last durable status still says training."
        ),
    )
    return parser.parse_args()


def model_from_state(
    plan: dict,
    state: dict[str, torch.Tensor],
    device: torch.device,
    return_density_contract: KnotDensityContract | None = None,
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
        feature_embedding_density=architecture.get("featureEmbeddingDensity"),
        return_density=architecture.get("returnDensity"),
        return_density_contract=return_density_contract,
        hindsight_conditioning=architecture.get("hindsightConditioning"),
        recurrent_activation_checkpointing=bool(
            architecture.get("recurrentActivationCheckpointing", False)
        ),
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
    if status.get("stage") == "training" \
            and not args.allow_stopped_training_status:
        raise ValueError("refusing to evaluate a run still marked as training")
    dataset = build_dataset(plan, repo)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    batch_size = int(args.batch_size or plan["training"]["evaluationBatchSize"])
    if hasattr(dataset, "rollout"):
        dataset.statistics(batch_size)
    uses_embedding_density = uses_embedding_density_objective(plan)
    uses_base_support_density = uses_base_support_embedding_density(plan)
    uses_return_density = uses_structured_return_density_objective(plan)
    return_density = (
        KnotDensityContract.load(
            (repo / plan["density"]["source"]).resolve(),
            fit=str(plan["density"]["fit"]),
        )
        if uses_return_density else None
    )
    checkpoint_specs = ((
        "best-validation-nll",
        run_root / "checkpoints/selections/validation-nll.json",
        "model",
    ), *(
        (
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
        )
        if uses_base_support_density else ()
    ), ("last", run_root / "checkpoints/last.json", "emaModel")) \
        if uses_embedding_density else (
        (
            "best-validation-nll",
            run_root / "checkpoints/selections/validation-nll.json",
            "model",
        ),
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
    ) if uses_return_density else (
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
        model = model_from_state(
            plan,
            checkpoint[state_key],
            device,
            return_density,
        )
        parameter_count = sum(value.numel() for value in model.parameters())
        selection_score = checkpoint.get("score")
        if uses_return_density:
            if return_density is None:
                raise AssertionError("structured return-density contract is missing")
            evaluations = {
                split: evaluate_structured_return_density(
                    model, dataset, split, batch_size=batch_size,
                    device=device, density=return_density,
                )
                for split in ("train", "validation", "test")
            }
        else:
            evaluator = (
                evaluate_feature_embedding_density
                if uses_embedding_density else evaluate
            )
            evaluations = {
                split: evaluator(
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
                "returnDensityExpectation.correlation"
                if uses_return_density and label == "best-validation-correlation"
                else "returnDensityExpectation.normalizedMse"
                if uses_return_density and label == "best-validation-mse"
                else "featureState.correlation"
                if label == "best-validation-correlation"
                else "featureState.normalizedMse"
                if label == "best-validation-mse"
                else "validation.negativeLogLikelihood"
                if label == "best-validation-nll"
                else None
            ),
            "checkpointPolicy": label,
            "checkpoint": str(checkpoint_file.relative_to(repo)).replace("\\", "/"),
            **(
                {
                    "density": evaluations,
                    **comparable_policy_metrics(evaluations),
                }
                if uses_return_density
                else
                {
                    "density": evaluations,
                    **comparable_policy_metrics(evaluations),
                }
                if uses_base_support_density
                else {"distribution": evaluations}
                if uses_embedding_density
                else comparable_policy_metrics(evaluations)
            ),
        }
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    persistence = (
        persist_structured_return_density_evaluation
        if uses_return_density
        else persist_embedding_density_evaluation
        if uses_embedding_density else persist_structured_feature_evaluation
    )
    artifact, result = persistence(
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
            "bestTest": (
                result["density"]["test"]
                if uses_return_density
                else result["density"]["test"]
                if uses_base_support_density
                else result["distribution"]["test"]
                if uses_embedding_density else result["test"]
            ),
            "lastTest": (
                policies["last"]["density"]["test"]
                if uses_return_density
                else policies["last"]["density"]["test"]
                if uses_base_support_density
                else policies["last"]["distribution"]["test"]
                if uses_embedding_density else policies["last"]["test"]
            ),
            "seconds": time.monotonic() - started,
        },
    })
    atomic_json(status, status_file)
    dataset.close()
    print(json.dumps({
        "planId": plan["id"],
        "selectionPolicy": result["selectionPolicy"],
        "bestEpoch": result["bestEpoch"],
        "bestValidation": (
            result["density"]["validation"]
            if uses_return_density
            else result["density"]["validation"]
            if uses_base_support_density
            else result["distribution"]["validation"]
            if uses_embedding_density else result["validation"]
        ),
        "bestTest": (
            result["density"]["test"]
            if uses_return_density
            else result["density"]["test"]
            if uses_base_support_density
            else result["distribution"]["test"]
            if uses_embedding_density else result["test"]
        ),
        "lastTest": (
            policies["last"]["density"]["test"]
            if uses_return_density
            else policies["last"]["density"]["test"]
            if uses_base_support_density
            else policies["last"]["distribution"]["test"]
            if uses_embedding_density else policies["last"]["test"]
        ),
        "policies": sorted(policies),
        "evaluationArtifact": result["evaluationArtifact"],
        "seconds": time.monotonic() - started,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
