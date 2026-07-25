from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from adaptive_curriculum import (
    ProjectedQuadratic,
    candidate_matrix,
    enumerate_weight_candidates,
    select_probe_indices,
)
from mlp_model import PolicySupport
from train_mlp import (
    FittedPolicyDataset,
    atomic_json,
    cached_training_normalization,
    cached_training_parameter_scale,
    deterministic_current_states,
    loader,
    parse_time_weighting,
    resolve_device,
    set_determinism,
    time_weighting_metadata,
    validate_dataset_manifest,
)
from train_mlp_population import PopulationRuntime


TERM_METRICS = (
    ("klDivergence", 1.0),
    ("probabilityMse", 1.0),
    ("parameterMse", 1.0),
    ("excessEntropy", 1.0),
    ("temporalMutualInformationReward", -1.0),
    ("oracleMutualInformation", -1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project all absolute MLP loss-weight tuples through a local "
            "validation-KL Hessian."
        ),
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specification = json.loads(args.spec.read_text())
    result = project_weight_candidates(specification)
    atomic_json(result, args.output)
    print(json.dumps({
        "event": "projected-loss-weight-screen-complete",
        "output": str(args.output.resolve()),
        "parent": result["parent"],
        "delaySeconds": result["delaySeconds"],
        "candidates": result["candidateCount"],
        "selectedProbes": len(result["selectedProbeKeys"]),
        "projectionBackend": result["projectionBackend"],
        "gpuMemoryMiB": result["gpuMemoryMiB"],
    }))


def project_weight_candidates(specification: dict) -> dict:
    validate_projection_specification(specification)
    set_determinism(int(specification["seed"]))
    dataset_root = Path(specification["dataset"])
    manifest = json.loads((dataset_root / "dataset.json").read_text())
    validate_dataset_manifest(manifest, dataset_root)
    plan = json.loads(Path(specification["plan"]).read_text())
    if (
        manifest.get("planId") != plan.get("id")
        or int(manifest.get("predictionDelayMs", -1))
        != int(plan.get("predictionDelayMs", -2))
    ):
        raise ValueError("projected screen dataset does not match its delay plan")
    expected_time_weighting = time_weighting_metadata(
        parse_time_weighting(json.dumps(specification["timeWeighting"]))
    )
    if manifest["exampleWeighting"]["timeWeighting"] != expected_time_weighting:
        raise ValueError("projected screen time weighting does not match the dataset")

    device = resolve_device(str(specification["device"]))
    train = FittedPolicyDataset(
        manifest, dataset_root, "train", target="teacherParameters",
    )
    validation = FittedPolicyDataset(
        manifest, dataset_root, "validation", target="teacherParameters",
    )
    feature_mean, feature_std = cached_training_normalization(
        train,
        Path(specification["featureStatisticsCache"]),
    )
    parameter_scale = cached_training_parameter_scale(
        train,
        Path(specification["targetStatisticsCache"]),
    ).to(device)
    support = PolicySupport(**manifest["policySupport"])
    actions = torch.linspace(
        support.visible_lower,
        support.visible_upper,
        int(manifest["actionCount"]),
        dtype=torch.float32,
        device=device,
    )
    current = deterministic_current_states(
        int(specification["statesPerExample"]),
        support,
        device,
        visible=True,
    )
    common = {
        "seed": int(specification["seed"]),
        "dropout": float(specification["dropout"]),
        "compile": False,
    }
    runtime = PopulationRuntime(
        1,
        feature_mean,
        feature_std,
        common,
        actions,
        current,
        support,
        parameter_scale,
        int(manifest["samplingIntervalMs"]),
        device,
    )
    runtime.reset([{
        "initializeFromCheckpoint": str(specification["parent"]),
    }])
    runtime.model.eval()
    workers = argparse.Namespace(workers=int(specification.get("workers", 0)))
    batch_size = int(specification["batchSize"])
    train_batch = first_contiguous_batch(loader(
        train,
        workers,
        shuffle=False,
        batch_size=batch_size,
    ))
    validation_batch = first_contiguous_batch(loader(
        validation,
        workers,
        shuffle=False,
        batch_size=batch_size,
    ))
    parameters = tuple(runtime.model.parameters())
    zero_weights = torch.zeros(
        (1, len(TERM_METRICS)),
        dtype=torch.float32,
        device=device,
    )
    train_metrics = objective_batch(runtime, train_batch, zero_weights)
    term_gradients: list[tuple[Tensor, ...]] = []
    for index, (metric, sign) in enumerate(TERM_METRICS):
        scalar = train_metrics[metric].sum() * sign
        gradient = torch.autograd.grad(
            scalar,
            parameters,
            retain_graph=index + 1 < len(TERM_METRICS),
            allow_unused=True,
        )
        term_gradients.append(tuple(
            (
                value.detach()
                if value is not None
                else torch.zeros_like(parameter)
            )
            for value, parameter in zip(gradient, parameters, strict=True)
        ))

    gradient_gram = pairwise_dot(term_gradients)
    del train_metrics
    validation_metrics = objective_batch(runtime, validation_batch, zero_weights)
    validation_kl = validation_metrics["klDivergence"].sum()
    validation_gradient = torch.autograd.grad(
        validation_kl,
        parameters,
        create_graph=True,
        allow_unused=True,
    )
    validation_gradient = tuple(
        (
            value
            if value is not None
            else torch.zeros_like(parameter)
        )
        for value, parameter in zip(validation_gradient, parameters, strict=True)
    )
    linear = np.asarray([
        tensor_dot(validation_gradient, direction)
        for direction in term_gradients
    ], dtype=np.float64)
    projection_backend = "validation-hessian"
    try:
        quadratic = np.empty((len(TERM_METRICS), len(TERM_METRICS)), dtype=np.float64)
        for row, direction in enumerate(term_gradients):
            directional_derivative = sum(
                (gradient * value).sum()
                for gradient, value in zip(
                    validation_gradient,
                    direction,
                    strict=True,
                )
            )
            hessian_direction = torch.autograd.grad(
                directional_derivative,
                parameters,
                retain_graph=row + 1 < len(TERM_METRICS),
                allow_unused=True,
            )
            hessian_direction = tuple(
                (
                    value.detach()
                    if value is not None
                    else torch.zeros_like(parameter)
                )
                for value, parameter in zip(
                    hessian_direction,
                    parameters,
                    strict=True,
                )
            )
            for column, other in enumerate(term_gradients):
                quadratic[row, column] = tensor_dot(
                    hessian_direction,
                    other,
                )
        quadratic = 0.5 * (quadratic + quadratic.T)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        projection_backend = "gradient-gram-fallback"
        quadratic = gradient_gram.copy()

    candidates = enumerate_weight_candidates(
        specification["absoluteWeightLevels"],
        canonicalize_global_scale=bool(
            specification.get("canonicalizeGlobalScale", True)
        ),
        fixed_loss_weights=specification.get("fixedLossWeights"),
        term_weight_levels=specification.get("termWeightLevels"),
    )
    matrix = candidate_matrix(candidates)
    projection = ProjectedQuadratic(
        base_validation=float(validation_kl.detach().cpu()),
        linear=linear,
        quadratic=quadratic,
        gradient_gram=gradient_gram,
        learning_rate=float(specification["learningRate"]),
        maximum_gradient_norm=float(specification["maximumGradientNorm"]),
    )
    scores = projection.scores(matrix)
    selected = select_probe_indices(
        matrix,
        scores,
        int(specification["initialProbeCount"]),
        exploitation_fraction=float(specification["exploitationFraction"]),
        random_fraction=float(specification["randomFraction"]),
        seed=int(specification["seed"]),
    )
    return {
        "version": 2,
        "parent": str(specification["parent"]),
        "datasetPlanId": manifest["planId"],
        "delaySeconds": int(manifest["predictionDelayMs"]) // 1_000,
        "absoluteWeightLevels": [
            float(value) for value in specification["absoluteWeightLevels"]
        ],
        "canonicalizeGlobalScale": bool(
            specification.get("canonicalizeGlobalScale", True)
        ),
        "fixedLossWeights": specification.get("fixedLossWeights", {}),
        "termWeightLevels": specification.get("termWeightLevels", {}),
        "candidateSpaceDigest": specification.get("candidateSpaceDigest"),
        "searchDesignDigest": specification.get("searchDesignDigest"),
        "candidateCount": len(candidates),
        "baseValidationKl": projection.base_validation,
        "linear": linear.tolist(),
        "quadratic": quadratic.tolist(),
        "gradientGram": gradient_gram.tolist(),
        "learningRate": projection.learning_rate,
        "maximumGradientNorm": projection.maximum_gradient_norm,
        "projectionBackend": projection_backend,
        "selectedProbeKeys": [candidates[index].key for index in selected],
        "selectedProbes": [{
            "key": candidates[index].key,
            "weights": candidates[index].as_dict(),
            "projectedValidationKl": float(scores[index]),
        } for index in selected],
        "projectedBest": [{
            "key": candidates[int(index)].key,
            "weights": candidates[int(index)].as_dict(),
            "projectedValidationKl": float(scores[int(index)]),
        } for index in np.argsort(scores)[:min(32, len(scores))]],
        "gpuMemoryMiB": (
            float(torch.cuda.max_memory_allocated() / 1_048_576)
            if device.type == "cuda"
            else 0
        ),
    }


def objective_batch(
    runtime: PopulationRuntime,
    batch,
    weight_matrix: Tensor,
) -> dict[str, Tensor]:
    features, targets, time_weights, times, _ = batch
    return runtime.objective(
        features.to(runtime.device),
        targets.to(runtime.device),
        time_weights.to(runtime.device),
        times.to(runtime.device),
        weight_matrix,
    )


def first_contiguous_batch(data):
    for batch in data:
        times = batch[3]
        if len(times) < 2:
            continue
        differences = times[1:] - times[:-1]
        if bool((differences == differences[0]).all()) and int(differences[0]) > 0:
            return batch
    raise RuntimeError("projected screen found no contiguous time block")


def pairwise_dot(
    gradients: list[tuple[Tensor, ...]],
) -> np.ndarray:
    result = np.empty((len(gradients), len(gradients)), dtype=np.float64)
    for row, left in enumerate(gradients):
        for column in range(row, len(gradients)):
            value = tensor_dot(left, gradients[column])
            result[row, column] = value
            result[column, row] = value
    return result


def tensor_dot(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=left[0].device)
    for left_value, right_value in zip(left, right, strict=True):
        total += (left_value.float() * right_value.float()).sum(dtype=torch.float64)
    return float(total.detach().cpu())


def validate_projection_specification(specification: dict) -> None:
    required = {
        "dataset",
        "plan",
        "parent",
        "absoluteWeightLevels",
        "batchSize",
        "statesPerExample",
        "dropout",
        "seed",
        "device",
        "workers",
        "timeWeighting",
        "featureStatisticsCache",
        "targetStatisticsCache",
        "learningRate",
        "maximumGradientNorm",
        "initialProbeCount",
        "exploitationFraction",
        "randomFraction",
    }
    if not isinstance(specification, dict) or not required.issubset(specification):
        raise ValueError("projected loss-weight specification is incomplete")
    for key in ("batchSize", "statesPerExample", "initialProbeCount"):
        if not isinstance(specification[key], int) or specification[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "learningRate",
        "maximumGradientNorm",
        "exploitationFraction",
        "randomFraction",
    ):
        if not math.isfinite(float(specification[key])) or float(specification[key]) < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if not Path(specification["parent"]).is_file():
        raise FileNotFoundError(f"projection parent does not exist: {specification['parent']}")


if __name__ == "__main__":
    main()
