from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from return_oracle_decoder_screen import (
    FEATURE_CONTRACT,
    PRODUCTION_ORACLE_TEMPERATURE,
    RUNNER_CONTRACT,
    SELECTION_CONTRACT,
    DecoderModel,
    LearnedRadiusShrinkingDecoder,
    build_decoder,
    canonical_plan_fingerprint,
    file_sha256,
    next_equal_entropy_step_training_temperature,
    next_stale_learning_rate,
    optimizer_parameter_groups,
    production_entropy_confidence_weights,
    smooth_oracle_probabilities,
    target_temperature,
    validate_screen_plan,
    weighted_policy_metrics,
    weighted_target_entropy_log_temperature_derivative,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)
from train_return_oracle_ce import (
    DeviceBatch,
    DeviceBatchPipeline,
    ExperimentDataset,
    RunReporter,
    atomic_json,
    frozen_cpu_copy,
    group_batches,
    iso_now,
    resolve,
    select_segments,
    validate_source_manifest,
)


ModelForwardResult = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
MetricFunction = Callable[[Tensor], ModelForwardResult]


@dataclass
class ScreenContext:
    repo_root: Path
    plan_file: Path
    plan: dict[str, Any]
    plan_fingerprint: str
    source_root: Path
    dataset_root: Path
    run_dir: Path
    feature_root: Path
    source_manifest: dict[str, Any]
    dataset_manifest: dict[str, Any]
    segments: dict[str, list[Any]]
    feature_mean: Tensor
    feature_std: Tensor


@dataclass
class MetricAccumulator:
    examples: float = 0
    sums: dict[str, float] | None = None

    def add(self, metrics: dict[str, Tensor], weight: float) -> None:
        if weight <= 0 or not math.isfinite(weight):
            raise ValueError("metric weight must be finite and positive")
        if self.sums is None:
            self.sums = {name: 0.0 for name in metrics}
        if set(metrics) != set(self.sums):
            raise ValueError("metric fields changed during an epoch")
        self.examples += weight
        for name, value in metrics.items():
            self.sums[name] += float(value.detach()) * weight

    def result(self) -> dict[str, float]:
        if self.examples <= 0 or self.sums is None:
            raise RuntimeError("cannot finalize empty screen metrics")
        return {
            name: value / self.examples for name, value in self.sums.items()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one immutable return-oracle decoder capability screen. "
            "The sealed test split is never loaded."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the immutable plan/data/model contracts without training.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help=(
            "Pause after this many completed epochs, after validation and "
            "the durable last/best checkpoint writes. Use 1 for a one-epoch "
            "canary; omit it to resume through the plan budget."
        ),
    )
    return parser.parse_args()


def validate_runtime_options(
    *,
    validate_only: bool,
    stop_after_epoch: int | None,
) -> None:
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    if validate_only and stop_after_epoch is not None:
        raise ValueError(
            "--validate-only and --stop-after-epoch are mutually exclusive"
        )


def completed_epoch_limit_reached(
    stop_after_epoch: int | None,
    last_completed_epoch: int,
) -> bool:
    """Interpret the CLI boundary as a one-based completed-epoch count."""
    if last_completed_epoch < -1:
        raise ValueError("last completed epoch index is invalid")
    return stop_after_epoch is not None \
        and last_completed_epoch + 1 >= stop_after_epoch


def early_stopping_limit_reached(
    patience: int | None,
    stale_epochs: int,
) -> bool:
    if stale_epochs < 0:
        raise ValueError("stale epoch count cannot be negative")
    if patience is None:
        return False
    if isinstance(patience, bool) or patience < 1:
        raise ValueError("early stopping patience must be positive")
    return stale_epochs >= patience


def persist_validated_epoch(
    checkpoint: dict[str, Any],
    *,
    last_checkpoint: Path,
    best_checkpoint: Path,
    improved: bool,
    stop_after_epoch: int | None,
    completed_epoch: int,
) -> bool:
    """Synchronously persist validation-bearing state before a pause decision."""
    if "validation" not in checkpoint:
        raise ValueError("completed-epoch checkpoint lacks validation metrics")
    save_torch_checkpoint(checkpoint, last_checkpoint)
    if improved:
        save_torch_checkpoint(checkpoint, best_checkpoint)
    return completed_epoch_limit_reached(stop_after_epoch, completed_epoch)


def load_context(plan_path: Path) -> ScreenContext:
    repo_root = Path(__file__).resolve().parent.parent
    plan_file = resolve(repo_root, plan_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_screen_plan(plan)
    plan_fingerprint = canonical_plan_fingerprint(plan)
    layout = training_storage_layout(repo_root)
    dataset_config = plan["dataset"]
    dataset_root = require_under(
        resolve(repo_root, Path(dataset_config["datasetDir"])).resolve(),
        layout.datasets,
        "datasetDir",
    )
    source_root = require_under(
        resolve(repo_root, Path(dataset_config["sourceDatasetDir"])).resolve(),
        layout.datasets,
        "sourceDatasetDir",
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])).resolve(),
        layout.runs,
        "runDir",
    )
    dataset_file = dataset_root / "dataset.json"
    source_file = source_root / "dataset.json"
    dataset_bytes = dataset_file.read_bytes()
    source_bytes = source_file.read_bytes()
    if file_sha256(dataset_bytes) != dataset_config["datasetSha256"] \
            or file_sha256(source_bytes) \
            != dataset_config["sourceDatasetSha256"]:
        raise ValueError("frozen decoder dataset fingerprint changed")
    dataset_manifest = json.loads(dataset_bytes)
    source_manifest = json.loads(source_bytes)
    validate_source_manifest(source_manifest)
    if float(source_manifest.get("execution", {}).get("temperature", -1)) \
            != PRODUCTION_ORACLE_TEMPERATURE:
        raise ValueError("source oracle temperature is not production 0.01")
    if dataset_manifest.get("inputTransform") != FEATURE_CONTRACT \
            or dataset_manifest.get("counts") != dataset_config["expectedCounts"] \
            or dataset_manifest.get("testSelection") \
            != "last 1,000,000 chronological source test examples":
        raise ValueError("final compact decoder dataset contract changed")
    all_segments = select_segments(source_manifest, source_root)
    counts = {
        split: sum(segment.count for segment in values)
        for split, values in all_segments.items()
    }
    if counts != dataset_config["expectedCounts"]:
        raise ValueError("selected source segment counts changed")
    # Deliberately remove the test split from the live dataset object. Its
    # metadata is fingerprinted above, but no test target payload is reachable.
    segments = {
        "train": all_segments["train"],
        "validation": all_segments["validation"],
    }
    statistics = dataset_manifest["featureStandardization"]
    feature_mean = torch.tensor(statistics["mean"], dtype=torch.float32)
    feature_std = torch.tensor(statistics["std"], dtype=torch.float32)
    feature_root = (
        layout.immutable / "refs" / "features"
        / "return-oracle-simple-returns-v1"
        / dataset_config["featureSetId"]
    )
    return ScreenContext(
        repo_root=repo_root,
        plan_file=plan_file,
        plan=plan,
        plan_fingerprint=plan_fingerprint,
        source_root=source_root,
        dataset_root=dataset_root,
        run_dir=run_dir,
        feature_root=feature_root,
        source_manifest=source_manifest,
        dataset_manifest=dataset_manifest,
        segments=segments,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def snapshot_plan(context: ScreenContext) -> None:
    snapshot_file = context.run_dir / "state" / "plan.json"
    snapshot = {
        "planSha256": context.plan_fingerprint,
        "plan": context.plan,
    }
    if snapshot_file.is_file():
        if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
            raise ValueError("run directory belongs to a different immutable plan")
        return
    atomic_json(snapshot, snapshot_file)


def build_optimizers(
    model: DecoderModel,
    training: dict[str, Any],
    device: torch.device,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    muon_parameters, adamw_parameters = optimizer_parameter_groups(model)
    optimizer = training["optimizer"]
    muon = optimizer["muon"]
    adamw = optimizer["adamw"]
    return (
        torch.optim.Muon(
            muon_parameters,
            lr=float(training["learningRate"]),
            weight_decay=float(muon["weightDecay"]),
            momentum=float(muon["momentum"]),
            nesterov=bool(muon["nesterov"]),
            ns_steps=int(muon["newtonSchulzSteps"]),
            eps=float(muon["epsilon"]),
            adjust_lr_fn=str(muon["adjustLearningRate"]),
        ),
        torch.optim.AdamW(
            adamw_parameters,
            lr=float(training["learningRate"]),
            betas=tuple(float(value) for value in adamw["betas"]),
            eps=float(adamw["epsilon"]),
            weight_decay=float(adamw["weightDecay"]),
            fused=device.type == "cuda",
        ),
    )


def build_schedulers(
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    training: dict[str, Any],
) -> tuple[torch.optim.lr_scheduler.ReduceLROnPlateau, ...]:
    schedule = training["learningRateSchedule"]
    if schedule["type"] == "decay-every-prod-kl-stale-block":
        return ()
    return tuple(
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(schedule["factor"]),
            patience=int(schedule["patience"]),
            threshold=float(schedule["threshold"]),
            threshold_mode="abs",
            min_lr=float(schedule["minimumLearningRate"]),
        )
        for optimizer in optimizers
    )


def batch_metrics(
    forward_result: ModelForwardResult,
    raw_targets: Tensor,
    weights: Tensor,
    train_temperature: float,
    regularizers: dict[str, Any] | list[Any],
    *,
    include_entropy_temperature_derivative: bool = False,
) -> dict[str, Tensor]:
    (
        logits,
        mean_penalties,
        variance_penalties,
        soft_weight_bound,
        centering_idempotence,
        centering_symmetry,
    ) = forward_result
    curriculum_targets = smooth_oracle_probabilities(
        raw_targets,
        source_temperature=PRODUCTION_ORACLE_TEMPERATURE,
        target_temperature=train_temperature,
    )
    metrics = weighted_policy_metrics(logits, raw_targets, weights)
    curriculum_metrics = weighted_policy_metrics(
        logits, curriculum_targets, weights
    )
    metrics.update({
        "trainingCrossEntropy": curriculum_metrics["rawCrossEntropy"],
        "curriculumTargetKl": curriculum_metrics["rawBaseActionKl"],
        "curriculumTargetProbabilityMse": (
            curriculum_metrics["rawProbabilityMse"]
        ),
        "curriculumTargetEntropy": curriculum_metrics["rawTargetEntropy"],
        "predictedEntropy": curriculum_metrics["rawPredictedEntropy"],
    })
    if include_entropy_temperature_derivative:
        metrics["curriculumTargetEntropyLogTemperatureDerivative"] = (
            weighted_target_entropy_log_temperature_derivative(
                raw_targets,
                curriculum_targets,
                weights,
                source_temperature=PRODUCTION_ORACLE_TEMPERATURE,
                target_temperature=train_temperature,
            )
        )
    normalized_weights = weights.float().clamp_min(0)
    weight_sum = normalized_weights.sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    mean_penalty = (
        mean_penalties.float() * normalized_weights
    ).sum() / weight_sum
    variance_penalty = (
        variance_penalties.float() * normalized_weights
    ).sum() / weight_sum
    if regularizers:
        soft_layer_norm_config = regularizers["softLayerNorm"]
        centering_config = regularizers.get("centeringConstraint")
        soft_layer_norm = (
            mean_penalty
            + float(soft_layer_norm_config["varianceWeight"])
            * variance_penalty
        )
        centering_constraint = (
            logits.new_zeros((), dtype=torch.float32)
            if centering_config is None
            else (
                float(centering_config["idempotenceWeight"])
                * centering_idempotence.float()
                + float(centering_config["symmetryWeight"])
                * centering_symmetry.float()
            )
        )
        regularization_loss = (
            float(soft_layer_norm_config["weight"]) * soft_layer_norm
            + float(regularizers["softWeightBound"]["weight"])
            * soft_weight_bound.float()
            + centering_constraint
        )
    else:
        soft_layer_norm = logits.new_zeros((), dtype=torch.float32)
        centering_constraint = logits.new_zeros((), dtype=torch.float32)
        regularization_loss = logits.new_zeros((), dtype=torch.float32)
    metrics.update({
        "softLayerNormMeanPenalty": mean_penalty,
        "softLayerNormVariancePenalty": variance_penalty,
        "softLayerNorm": soft_layer_norm,
        "softWeightBound": soft_weight_bound.float(),
        "centeringIdempotence": centering_idempotence.float(),
        "centeringSymmetry": centering_symmetry.float(),
        "centeringConstraint": centering_constraint,
        "regularizationLoss": regularization_loss,
        "loss": metrics["trainingCrossEntropy"] + regularization_loss,
    })
    return metrics


@torch.inference_mode()
def evaluate_validation(
    model: DecoderModel,
    forward: MetricFunction,
    dataset: ExperimentDataset,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    pipeline: DeviceBatchPipeline,
    train_temperature: float,
    regularizers: dict[str, Any] | list[Any],
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    parameter_penalties: tuple[Tensor, Tensor, Tensor] | None = None
    if regularizers:
        if not isinstance(model, LearnedRadiusShrinkingDecoder):
            raise TypeError("restored regularizers require learned-radius decoder")
        weight_config = regularizers["softWeightBound"]
        parameter_penalties = model.regularizer_parameter_penalties(
            desired_weight_magnitude=float(weight_config["desiredMagnitude"]),
            weight_bound_sharpness=float(weight_config["sharpness"]),
            absolute_epsilon=float(weight_config["absoluteEpsilon"]),
            include_centering_constraint=(
                "centeringConstraint" in regularizers
            ),
        )
    batches = dataset.iter_batches(
        "validation", batch_size, shuffle=False, seed=0
    )
    for features, targets, weights, original_count in pipeline.batches(batches):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            forward_result = forward(features)
            if parameter_penalties is not None:
                forward_result = (
                    *forward_result[:3],
                    *parameter_penalties,
                )
            metrics = batch_metrics(
                forward_result,
                targets,
                weights,
                train_temperature,
                regularizers,
                include_entropy_temperature_derivative=True,
            )
        accumulator.add(metrics, original_count)
    return accumulator.result()


def restore_checkpoint(
    checkpoint: dict[str, Any],
    *,
    context: ScreenContext,
    model: DecoderModel,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    schedulers: tuple[torch.optim.lr_scheduler.ReduceLROnPlateau, ...],
    model_parameters: int,
    device: torch.device,
) -> tuple[
    int, int, float, int, int, int | None, float | None, float | None, int, int
]:
    if checkpoint.get("planSha256") != context.plan_fingerprint \
            or checkpoint.get("architectureContract") \
            != model.architecture_contract \
            or checkpoint.get("selectionContract") != SELECTION_CONTRACT \
            or checkpoint.get("runnerContract") != RUNNER_CONTRACT \
            or checkpoint.get("parameterCount") != model_parameters:
        raise ValueError("resume checkpoint is incompatible with immutable plan")
    model.load_state_dict(checkpoint["model"])
    for optimizer, state in zip(
        optimizers, checkpoint["optimizers"], strict=True
    ):
        optimizer.load_state_dict(state)
        if device.type == "cuda" and isinstance(optimizer, torch.optim.AdamW):
            for group in optimizer.param_groups:
                group["fused"] = True
                group["foreach"] = None
    for scheduler, state in zip(
        schedulers, checkpoint["schedulers"], strict=True
    ):
        scheduler.load_state_dict(state)
    torch.set_rng_state(checkpoint["torchRngState"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([
            value.cpu() for value in checkpoint["cudaRngStates"]
        ])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["globalStep"]),
        float(checkpoint["bestRawValidationKl"]),
        int(checkpoint["bestEpoch"]),
        int(checkpoint.get(
            "staleEpochs",
            max(0, int(checkpoint["epoch"]) - int(checkpoint["bestEpoch"])),
        )),
        (
            None
            if checkpoint.get("curriculumGateEpoch") is None
            else int(checkpoint["curriculumGateEpoch"])
        ),
        (
            None
            if checkpoint.get("nextTrainingTargetTemperature") is None
            else float(checkpoint["nextTrainingTargetTemperature"])
        ),
        (
            None
            if checkpoint.get("curriculumStartValidationEntropy") is None
            else float(checkpoint["curriculumStartValidationEntropy"])
        ),
        int(checkpoint.get("learningRateDecaySteps", 0)),
        int(checkpoint.get("learningRateStaleEpochs", checkpoint.get(
            "staleEpochs",
            max(0, int(checkpoint["epoch"]) - int(checkpoint["bestEpoch"])),
        ))),
    )


def make_checkpoint(
    *,
    context: ScreenContext,
    model: DecoderModel,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    schedulers: tuple[torch.optim.lr_scheduler.ReduceLROnPlateau, ...],
    epoch: int,
    global_step: int,
    best_raw_validation_kl: float,
    best_epoch: int,
    stale_epochs: int,
    validation: dict[str, float],
    model_parameters: int,
    train_temperature: float,
    next_train_temperature: float,
    curriculum_gate_epoch: int | None,
    curriculum_start_validation_entropy: float | None,
    learning_rate_decay_steps: int,
    learning_rate_stale_epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    return frozen_cpu_copy({
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
        "epoch": epoch,
        "globalStep": global_step,
        "bestRawValidationKl": best_raw_validation_kl,
        "bestEpoch": best_epoch,
        "staleEpochs": stale_epochs,
        "validation": validation,
        "trainingTargetTemperature": train_temperature,
        "nextTrainingTargetTemperature": next_train_temperature,
        "curriculumGateEpoch": curriculum_gate_epoch,
        "curriculumStartValidationEntropy": curriculum_start_validation_entropy,
        "learningRateDecaySteps": learning_rate_decay_steps,
        "learningRateStaleEpochs": learning_rate_stale_epochs,
        "parameterCount": model_parameters,
        "planSha256": context.plan_fingerprint,
        "architectureContract": model.architecture_contract,
        "featureContract": FEATURE_CONTRACT,
        "selectionContract": SELECTION_CONTRACT,
        "runnerContract": RUNNER_CONTRACT,
        "torchRngState": torch.get_rng_state(),
        "cudaRngStates": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        ),
        "sealedTestEvaluated": False,
    })


def train(context: ScreenContext, stop_after_epoch: int | None) -> None:
    plan = context.plan
    training = plan["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    reporter = RunReporter(context.run_dir)
    snapshot_plan(context)
    dataset = ExperimentDataset(
        context.feature_root,
        context.segments,
        workers=int(training["workers"]),
        prefetch_factor=int(training["prefetchFactor"]),
    )
    model = build_decoder(
        plan, context.feature_mean, context.feature_std
    ).to(device)
    model_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    optimizers = build_optimizers(model, training, device)
    schedulers = build_schedulers(optimizers, training)
    start_epoch = 0
    global_step = 0
    best_raw_validation_kl = math.inf
    best_epoch = -1
    stale_epochs = 0
    curriculum_gate_epoch: int | None = None
    stateful_train_temperature: float | None = None
    curriculum_start_validation_entropy: float | None = None
    learning_rate_decay_steps = 0
    learning_rate_stale_epochs = 0
    last_checkpoint = context.run_dir / "checkpoints" / "last.json"
    best_checkpoint = context.run_dir / "checkpoints" / "best.json"
    if checkpoint_exists(last_checkpoint):
        checkpoint = load_torch_checkpoint(
            last_checkpoint, map_location=device, weights_only=False
        )
        (
            start_epoch,
            global_step,
            best_raw_validation_kl,
            best_epoch,
            stale_epochs,
            curriculum_gate_epoch,
            stateful_train_temperature,
            curriculum_start_validation_entropy,
            learning_rate_decay_steps,
            learning_rate_stale_epochs,
        ) = restore_checkpoint(
            checkpoint,
            context=context,
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            model_parameters=model_parameters,
            device=device,
        )
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    regularizers = plan["objective"]["regularizers"]
    training_weighting = plan["objective"].get("trainingExampleWeighting")
    confidence_weighting_strength = (
        None
        if training_weighting is None
        else float(training_weighting["lambda"])
    )

    def activation_forward(
        features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if regularizers:
            if not isinstance(model, LearnedRadiusShrinkingDecoder):
                raise TypeError(
                    "restored regularizers require learned-radius decoder"
                )
            logits, mean_penalties, variance_penalties = (
                model.forward_with_regularizers(features)
            )
            return logits, mean_penalties, variance_penalties
        logits = model(features)
        per_example_zero = logits.new_zeros(
            (logits.shape[0],), dtype=torch.float32
        )
        return logits, per_example_zero, per_example_zero

    def training_forward(features: Tensor) -> ModelForwardResult:
        logits, mean_penalties, variance_penalties = activation_forward(features)
        if regularizers:
            if not isinstance(model, LearnedRadiusShrinkingDecoder):
                raise TypeError(
                    "restored regularizers require learned-radius decoder"
                )
            weight_config = regularizers["softWeightBound"]
            parameter_penalties = model.regularizer_parameter_penalties(
                desired_weight_magnitude=float(
                    weight_config["desiredMagnitude"]
                ),
                weight_bound_sharpness=float(weight_config["sharpness"]),
                absolute_epsilon=float(weight_config["absoluteEpsilon"]),
                include_centering_constraint=(
                    "centeringConstraint" in regularizers
                ),
            )
        else:
            scalar_zero = logits.new_zeros((), dtype=torch.float32)
            parameter_penalties = (scalar_zero, scalar_zero, scalar_zero)
        return (
            logits,
            mean_penalties,
            variance_penalties,
            *parameter_penalties,
        )

    def validation_forward(features: Tensor) -> ModelForwardResult:
        logits, mean_penalties, variance_penalties = activation_forward(features)
        scalar_zero = logits.new_zeros((), dtype=torch.float32)
        return (
            logits,
            mean_penalties,
            variance_penalties,
            scalar_zero,
            scalar_zero,
            scalar_zero,
        )

    if bool(training.get("compile", False)):
        options = {"triton.cudagraphs": False}
        compiled_train_forward = torch.compile(
            training_forward, options=options, dynamic=True, fullgraph=False
        )
        compiled_validation_forward = torch.compile(
            validation_forward, options=options, dynamic=True, fullgraph=False
        )
    else:
        compiled_train_forward = training_forward
        compiled_validation_forward = validation_forward
    batch_size = int(training["batchSize"])
    evaluation_batch_size = int(training["evaluationBatchSize"])
    accumulation = int(training["gradientAccumulationSteps"])
    maximum_epochs = int(training["epochs"])
    early_stopping_patience = training.get("earlyStoppingPatience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
    learning_rate_schedule = training["learningRateSchedule"]
    schedule_start = int(learning_rate_schedule.get("startEpoch", 0))
    pipeline = DeviceBatchPipeline(device)
    started_at = iso_now()
    reporter.status(
        "training",
        startedAt=started_at,
        planId=plan["id"],
        startEpoch=start_epoch,
        architectureContract=model.architecture_contract,
        parameters=model_parameters,
        trainExamples=dataset.count("train"),
        validationExamples=dataset.count("validation"),
        testPolicy="sealed-never-load",
        selectionContract=SELECTION_CONTRACT,
        stopAfterEpoch=stop_after_epoch,
        earlyStoppingPatience=early_stopping_patience,
    )
    reporter.emit({
        "event": "screen-start",
        "planId": plan["id"],
        "planSha256": context.plan_fingerprint,
        "startEpoch": start_epoch,
        "epochs": maximum_epochs,
        "architectureContract": model.architecture_contract,
        "parameters": model_parameters,
        "selectionContract": SELECTION_CONTRACT,
        "curriculum": plan["curriculum"],
        "learningRateSchedule": learning_rate_schedule,
        "stopAfterEpoch": stop_after_epoch,
        "earlyStoppingPatience": early_stopping_patience,
        "sealedTestEvaluated": False,
    })
    if completed_epoch_limit_reached(stop_after_epoch, start_epoch - 1):
        event = {
            "event": "screen-paused",
            "reason": "stop-after-epoch-already-reached",
            "lastCompletedEpoch": start_epoch - 1,
            "completedEpochs": start_epoch,
            "globalStep": global_step,
            "checkpoint": str(last_checkpoint),
            "sealedTestEvaluated": False,
        }
        reporter.emit(event)
        reporter.status(
            "paused",
            startedAt=started_at,
            pausedAt=iso_now(),
            planId=plan["id"],
            latest=event,
            bestRawValidationKl=best_raw_validation_kl,
            bestEpoch=best_epoch,
            resumableCheckpoint=str(last_checkpoint),
            stopAfterEpoch=stop_after_epoch,
            testPolicy="sealed-never-load",
            message=(
                "The requested completed-epoch boundary was already reached; "
                "no training or test evaluation ran."
            ),
        )
        return
    try:
        for epoch in range(start_epoch, maximum_epochs):
            epoch_started = time.monotonic()
            per_epoch_gate = plan["curriculum"].get("gate", {}).get(
                "evaluation"
            ) == "every-epoch"
            if per_epoch_gate:
                if stateful_train_temperature is None:
                    stateful_train_temperature = float(
                        plan["curriculum"]["startTemperature"]
                    )
                train_temperature = stateful_train_temperature
            else:
                train_temperature = target_temperature(
                    plan, epoch, curriculum_gate_epoch
                )
            model.train()
            accumulator = MetricAccumulator()
            batches = dataset.iter_batches(
                "train", batch_size, shuffle=True, seed=seed + epoch
            )
            for batch_group in group_batches(
                pipeline.batches(batches), accumulation
            ):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                weighted_batch_group = []
                for features, targets, weights, original_count in batch_group:
                    training_weights = (
                        weights
                        if confidence_weighting_strength is None
                        else production_entropy_confidence_weights(
                            targets,
                            weights,
                            confidence_weighting_strength,
                        )
                    )
                    weighted_batch_group.append((
                        features,
                        targets,
                        training_weights,
                        original_count,
                        training_weights.sum(),
                    ))
                group_weight = sum(
                    batch[4] for batch in weighted_batch_group
                )
                for (
                    features,
                    targets,
                    training_weights,
                    _original_count,
                    batch_weight,
                ) in weighted_batch_group:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=device.type == "cuda",
                    ):
                        forward_result = compiled_train_forward(features)
                        metrics = batch_metrics(
                            forward_result,
                            targets,
                            training_weights,
                            train_temperature,
                            regularizers,
                        )
                    loss = metrics["loss"]
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"non-finite decoder loss at epoch {epoch}"
                        )
                    (loss * (batch_weight / group_weight)).backward()
                    accumulator.add(metrics, float(batch_weight.detach()))
                gradient_norm = clip_grad_norm_(
                    model.parameters(),
                    float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                if not math.isfinite(float(gradient_norm)):
                    raise FloatingPointError(
                        f"non-finite decoder gradient at epoch {epoch}"
                    )
                for optimizer in optimizers:
                    optimizer.step()
                global_step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_metrics = accumulator.result()
            validation = evaluate_validation(
                model,
                compiled_validation_forward,
                dataset,
                evaluation_batch_size,
                device,
                amp_dtype,
                pipeline,
                train_temperature,
                regularizers,
            )
            curriculum_kl = validation["curriculumTargetKl"]
            if curriculum_start_validation_entropy is None:
                curriculum_start_validation_entropy = validation[
                    "curriculumTargetEntropy"
                ]
            gate = plan["curriculum"].get("gate")
            gate_metric = None if gate is None else str(gate["metric"])
            gate_value = (
                None if gate_metric is None else validation[gate_metric]
            )
            gate_reached_now = False
            entropy_goal = validation["curriculumTargetEntropy"]
            if per_epoch_gate and gate_value is not None:
                next_train_temperature, next_entropy_goal = (
                    next_equal_entropy_step_training_temperature(
                        plan,
                        train_temperature,
                        gate_value,
                        validation["curriculumTargetEntropy"],
                        validation["rawTargetEntropy"],
                        curriculum_start_validation_entropy,
                        validation[
                            "curriculumTargetEntropyLogTemperatureDerivative"
                        ],
                    )
                )
                gate_reached_now = next_train_temperature < train_temperature
            else:
                next_train_temperature = train_temperature
                next_entropy_goal = entropy_goal
            if not per_epoch_gate and gate is not None \
                    and curriculum_gate_epoch is None \
                    and gate_value is not None \
                    and gate_value <= float(gate["threshold"]):
                curriculum_gate_epoch = epoch
                gate_reached_now = True
            raw_kl = validation["rawBaseActionKl"]
            if not math.isfinite(raw_kl):
                raise FloatingPointError("raw validation KL is non-finite")
            improved = raw_kl < best_raw_validation_kl
            if improved:
                best_raw_validation_kl = raw_kl
                best_epoch = epoch
                stale_epochs = 0
                learning_rate_stale_epochs = 0
            else:
                stale_epochs += 1
                learning_rate_stale_epochs += 1
            learning_rate_before = float(optimizers[0].param_groups[0]["lr"])
            if learning_rate_schedule["type"] \
                    == "decay-every-prod-kl-stale-block":
                learning_rate = next_stale_learning_rate(
                    learning_rate_schedule,
                    learning_rate_before,
                    learning_rate_stale_epochs,
                )
                if learning_rate < learning_rate_before:
                    for optimizer in optimizers:
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                    learning_rate_decay_steps += 1
            elif epoch >= schedule_start:
                for scheduler in schedulers:
                    scheduler.step(raw_kl)
                learning_rate = float(optimizers[0].param_groups[0]["lr"])
            else:
                learning_rate = learning_rate_before
            checkpoint = make_checkpoint(
                context=context,
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                epoch=epoch,
                global_step=global_step,
                best_raw_validation_kl=best_raw_validation_kl,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                validation=validation,
                model_parameters=model_parameters,
                train_temperature=train_temperature,
                next_train_temperature=next_train_temperature,
                curriculum_gate_epoch=curriculum_gate_epoch,
                curriculum_start_validation_entropy=(
                    curriculum_start_validation_entropy
                ),
                learning_rate_decay_steps=learning_rate_decay_steps,
                learning_rate_stale_epochs=learning_rate_stale_epochs,
                device=device,
            )
            pause_for_epoch_limit = persist_validated_epoch(
                checkpoint,
                last_checkpoint=last_checkpoint,
                best_checkpoint=best_checkpoint,
                improved=improved,
                stop_after_epoch=stop_after_epoch,
                completed_epoch=epoch,
            )
            event = {
                "event": "epoch",
                "epoch": epoch,
                "epochs": maximum_epochs,
                "seconds": time.monotonic() - epoch_started,
                "globalStep": global_step,
                "trainingTargetTemperature": train_temperature,
                "nextTrainingTargetTemperature": next_train_temperature,
                "curriculumTemperatureDecreased": gate_reached_now,
                "curriculumValidationKl": curriculum_kl,
                "curriculumGateMetric": gate_metric,
                "curriculumGateValue": gate_value,
                "curriculumGateReached": (
                    gate_reached_now
                    if per_epoch_gate
                    else curriculum_gate_epoch is not None
                ),
                "curriculumGateReachedNow": gate_reached_now,
                "curriculumGateEpoch": curriculum_gate_epoch,
                "nextCurriculumTargetEntropyGoal": next_entropy_goal,
                "train": train_metrics,
                "validation": validation,
                "selectionMetric": "rawBaseActionKl",
                "bestRawValidationKl": best_raw_validation_kl,
                "bestEpoch": best_epoch,
                "staleEpochs": stale_epochs,
                "improved": improved,
                "learningRate": learning_rate,
                "learningRateReduced": learning_rate < learning_rate_before,
                "learningRateDecaySteps": learning_rate_decay_steps,
                "learningRateScheduleMetric": (
                    learning_rate_schedule.get("metric")
                ),
                "learningRateScheduleValue": raw_kl,
                "learningRateStaleEpochs": learning_rate_stale_epochs,
                "learningRateStaleEpochsPerReduction": (
                    learning_rate_schedule.get("staleEpochsPerReduction")
                ),
                "sealedTestEvaluated": False,
            }
            reporter.emit(event)
            reporter.status(
                "training",
                startedAt=started_at,
                planId=plan["id"],
                latest=event,
                bestRawValidationKl=best_raw_validation_kl,
                bestEpoch=best_epoch,
                stopAfterEpoch=stop_after_epoch,
                testPolicy="sealed-never-load",
            )
            if per_epoch_gate:
                stateful_train_temperature = next_train_temperature
            # This check is intentionally after validation and the synchronous
            # canonical last/best checkpoint writes above. Returning here can
            # only expose a fully resumable epoch boundary.
            if pause_for_epoch_limit:
                pause_event = {
                    "event": "screen-paused",
                    "reason": "stop-after-epoch",
                    "lastCompletedEpoch": epoch,
                    "completedEpochs": epoch + 1,
                    "globalStep": global_step,
                    "checkpoint": str(last_checkpoint),
                    "bestCheckpoint": str(best_checkpoint),
                    "sealedTestEvaluated": False,
                }
                reporter.emit(pause_event)
                reporter.status(
                    "paused",
                    startedAt=started_at,
                    pausedAt=iso_now(),
                    planId=plan["id"],
                    latest=pause_event,
                    bestRawValidationKl=best_raw_validation_kl,
                    bestEpoch=best_epoch,
                    resumableCheckpoint=str(last_checkpoint),
                    stopAfterEpoch=stop_after_epoch,
                    testPolicy="sealed-never-load",
                    message=(
                        "Paused after validation at a durable completed-epoch "
                        "boundary; omit --stop-after-epoch to resume at the "
                        "next epoch."
                    ),
                )
                return
            if early_stopping_limit_reached(
                early_stopping_patience,
                stale_epochs,
            ):
                reporter.emit({
                    "event": "early-stop",
                    "epoch": epoch,
                    "staleEpochs": stale_epochs,
                    "patience": early_stopping_patience,
                    "bestEpoch": best_epoch,
                    "bestRawValidationKl": best_raw_validation_kl,
                    "checkpoint": str(last_checkpoint),
                    "sealedTestEvaluated": False,
                })
                break
    except KeyboardInterrupt:
        reporter.status(
            "paused",
            startedAt=started_at,
            pausedAt=iso_now(),
            planId=plan["id"],
            message=(
                "Interrupted; resume from the last completed epoch with the "
                "same immutable plan. The test split was not evaluated."
            ),
            bestRawValidationKl=(
                best_raw_validation_kl
                if math.isfinite(best_raw_validation_kl)
                else None
            ),
            bestEpoch=best_epoch,
            stopAfterEpoch=stop_after_epoch,
            testPolicy="sealed-never-load",
        )
        raise
    result = {
        "completedAt": iso_now(),
        "planId": plan["id"],
        "planSha256": context.plan_fingerprint,
        "architectureContract": model.architecture_contract,
        "epochs": maximum_epochs,
        "bestEpoch": best_epoch,
        "bestRawValidationKl": best_raw_validation_kl,
        "staleEpochs": stale_epochs,
        "earlyStoppingPatience": early_stopping_patience,
        "earlyStopped": early_stopping_limit_reached(
            early_stopping_patience,
            stale_epochs,
        ),
        "selectionContract": SELECTION_CONTRACT,
        "checkpoint": str(best_checkpoint),
        "sealedTestEvaluated": False,
        "testPolicy": "sealed-never-load",
        "interpretation": (
            "long-run raw-KL saturation attempt"
            if early_stopping_patience is not None
            else "short capability screen; not a saturation claim"
        ),
    }
    atomic_json(result, context.run_dir / "state" / "result.json")
    reporter.emit({"event": "screen-complete", **result})
    reporter.status(
        "complete",
        startedAt=started_at,
        completedAt=result["completedAt"],
        planId=plan["id"],
        latest=result,
        bestRawValidationKl=best_raw_validation_kl,
        bestEpoch=best_epoch,
        testPolicy="sealed-never-load",
    )


def main() -> None:
    args = parse_args()
    validate_runtime_options(
        validate_only=args.validate_only,
        stop_after_epoch=args.stop_after_epoch,
    )
    context = load_context(args.plan)
    model = build_decoder(
        context.plan, context.feature_mean, context.feature_std
    )
    model_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    optimizer_parameter_groups(model)
    if args.validate_only:
        print(json.dumps({
            "valid": True,
            "planId": context.plan["id"],
            "planSha256": context.plan_fingerprint,
            "architectureContract": model.architecture_contract,
            "parameterCount": model_parameters,
            "counts": context.plan["dataset"]["expectedCounts"],
            "selectionContract": SELECTION_CONTRACT,
            "testPolicy": "sealed-never-load",
        }, indent=2))
        return
    train(context, args.stop_after_epoch)


if __name__ == "__main__":
    main()
