from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from joint_price_oracle import JointLossWeights, joint_price_oracle_objective
from joint_price_oracle_actions import (
    actionable_policy_metrics_numpy,
    greedy_teacher_rollout_numpy,
    teacher_actions_at_current_exposures_numpy,
)
from trading_storage import require_under, training_storage_layout
from train_joint_price_oracle import (
    CausalOracleDataset,
    CausalSegment,
    MetricAccumulator,
    add_action_objective,
    autocast_context,
    build_model,
    load_causal_segments,
    resolve,
    resolve_action_objective_config,
    resolve_training_config,
    unpack_batch,
    validate_plan,
)


MAXIMUM_EXAMPLES = 4_096
MAXIMUM_STEPS = 10_000


@dataclass(frozen=True)
class FixedTrainingSubset:
    input_closes: Tensor
    future_closes: Tensor
    target_policy: Tensor
    teacher_current_exposures: Tensor
    reset_mask: np.ndarray

    @property
    def count(self) -> int:
        return int(self.input_closes.shape[0])


@dataclass(frozen=True)
class OverfitSettings:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    minimum_relative_improvement: float
    seed: int
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit a fixed chronological train-only subset without writing "
            "checkpoints, run state, or model artifacts."
        ),
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=0.01,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU is intentionally the safe default; CUDA must be explicit.",
    )
    return parser.parse_args()


def chronological_training_head(
    segments: list[CausalSegment],
    count: int,
) -> list[CausalSegment]:
    if count < 1:
        raise ValueError("diagnostic example count must be positive")
    remaining = count
    result: list[CausalSegment] = []
    for segment in sorted(
        segments,
        key=lambda value: value.prediction_time_start,
    ):
        if segment.split != "train":
            raise ValueError("diagnostic subset can contain only train segments")
        take = min(remaining, segment.count)
        result.append(CausalSegment(
            split="train",
            prediction_time_start=segment.prediction_time_start,
            count=take,
            target_file=segment.target_file,
            target_row_offset=segment.target_row_offset,
            step_ms=segment.step_ms,
        ))
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"train split has only {count - remaining:,}/{count:,} "
            "requested diagnostic examples"
        )
    return result


def materialize_fixed_training_subset(
    dataset: CausalOracleDataset,
    segments: list[CausalSegment],
    action_grid: np.ndarray,
    *,
    friction: float,
    temperature: float,
    execution_policy: dict[str, Any] | None = None,
) -> FixedTrainingSubset:
    """Read only selected training rows, preserving chronological resets."""
    if not segments or any(segment.split != "train" for segment in segments):
        raise ValueError("fixed diagnostic segments must be non-empty training rows")
    expected = sum(segment.count for segment in segments)
    batches = list(dataset.iter_batches(
        "train",
        expected,
        shuffle=False,
        seed=0,
        maximum_batches=None,
    ))
    if len(batches) != len(segments):
        raise RuntimeError("fixed subset batches do not match selected segments")
    inputs: list[Tensor] = []
    futures: list[Tensor] = []
    targets: list[Tensor] = []
    reset_mask = np.zeros(expected, dtype=np.bool_)
    offset = 0
    previous: CausalSegment | None = None
    for segment, batch in zip(segments, batches, strict=True):
        input_closes, future_closes, target_policy, teacher = unpack_batch(batch)
        if teacher is not None or input_closes.shape[0] != segment.count:
            raise RuntimeError("fixed subset dataset alignment is invalid")
        discontinuity = (
            previous is None
            or segment.step_ms != previous.step_ms
            or segment.prediction_time_start
            != previous.prediction_time_end + previous.step_ms
        )
        if discontinuity:
            reset_mask[offset] = True
        inputs.append(input_closes)
        futures.append(future_closes)
        targets.append(target_policy)
        offset += segment.count
        previous = segment
    input_tensor = torch.cat(inputs, dim=0).contiguous()
    future_tensor = torch.cat(futures, dim=0).contiguous()
    target_tensor = torch.cat(targets, dim=0).contiguous()
    if input_tensor.shape[0] != expected or target_tensor.shape[0] != expected:
        raise RuntimeError("fixed subset materialization lost examples")
    rollout = greedy_teacher_rollout_numpy(
        target_tensor.detach().cpu().numpy(),
        np.asarray(action_grid, dtype=np.float64),
        reset_mask=reset_mask,
        friction=friction,
        temperature=temperature,
        execution_policy=execution_policy,
    )
    teacher_exposures = torch.from_numpy(
        rollout.current_exposures.astype(np.float32, copy=True)
    )
    return FixedTrainingSubset(
        input_closes=input_tensor,
        future_closes=future_tensor,
        target_policy=target_tensor,
        teacher_current_exposures=teacher_exposures,
        reset_mask=reset_mask,
    )


def fixed_subset_source_switch_fraction(
    subset: FixedTrainingSubset,
    action_grid: Tensor,
    action_objective: dict,
    *,
    friction: float,
    temperature: float,
) -> float | None:
    """Return the fixed-subset class prior required by global weighting."""
    config = resolve_action_objective_config(action_objective)
    if config.get("switchWeighting", "batch") != "global":
        return None
    teacher_actions = teacher_actions_at_current_exposures_numpy(
        subset.target_policy.detach().cpu().numpy(),
        action_grid.detach().cpu().numpy().astype(np.float64),
        subset.teacher_current_exposures.detach().cpu().numpy(),
        friction=friction,
        temperature=temperature,
        execution_policy=config.get("executionPolicy"),
    )
    fraction = float(teacher_actions.switch_labels.mean())
    if not 0 < fraction < 1:
        raise ValueError(
            "global action switch weighting requires both switch and hold "
            "examples in the fixed diagnostic subset; increase --examples"
        )
    return fraction


def _loss_weights(training: dict) -> JointLossWeights:
    values = training["lossWeights"]
    return JointLossWeights(
        policy_cross_entropy=float(values["policyCrossEntropy"]),
        conditioned_policy_cross_entropy=float(
            values.get("conditionedPolicyCrossEntropy", 0.0)
        ),
        forecast=float(values["forecast"]),
        soft_layer_norm=float(values["softLayerNorm"]),
    )


def _batch_metrics(
    model: nn.Module,
    input_closes: Tensor,
    future_closes: Tensor,
    target_policy: Tensor,
    teacher_current_exposures: Tensor,
    action_grid: Tensor,
    training: dict,
    action_objective: dict,
    *,
    source_switch_fraction: float | None,
    friction: float,
    temperature: float,
) -> tuple[dict[str, Tensor], Tensor]:
    output = model.forward_with_forecast(input_closes)
    metrics = joint_price_oracle_objective(
        output,
        input_closes,
        future_closes,
        target_policy,
        _loss_weights(training),
        action_grid=action_grid,
        policy_friction=friction,
        policy_temperature=temperature,
        forecast_huber_delta=float(training["forecastHuberDelta"]),
        volatility_floor=float(training["volatilityFloor"]),
    )
    metrics = add_action_objective(
        metrics,
        output.policy_logits,
        target_policy,
        teacher_current_exposures,
        action_grid,
        action_objective,
        source_switch_fraction=source_switch_fraction,
        policy_friction=friction,
        policy_temperature=temperature,
    )
    return metrics, output.policy_logits


@torch.no_grad()
def evaluate_fixed_subset(
    model: nn.Module,
    subset: FixedTrainingSubset,
    action_grid: Tensor,
    training: dict,
    action_objective: dict,
    *,
    device: torch.device,
    batch_size: int,
    source_switch_fraction: float | None = None,
    friction: float,
    temperature: float,
) -> dict[str, Any]:
    resolved_action_objective = resolve_action_objective_config(
        action_objective
    )
    if source_switch_fraction is None:
        source_switch_fraction = fixed_subset_source_switch_fraction(
            subset,
            action_grid,
            resolved_action_objective,
            friction=friction,
            temperature=temperature,
        )
    model.eval()
    accumulator = MetricAccumulator()
    logits: list[np.ndarray] = []
    for start in range(0, subset.count, batch_size):
        end = min(subset.count, start + batch_size)
        with autocast_context(device, training):
            metrics, predicted_logits = _batch_metrics(
                model,
                subset.input_closes[start:end].to(device),
                subset.future_closes[start:end].to(device),
                subset.target_policy[start:end].to(device),
                subset.teacher_current_exposures[start:end].to(device),
                action_grid,
                training,
                resolved_action_objective,
                source_switch_fraction=source_switch_fraction,
                friction=friction,
                temperature=temperature,
            )
        accumulator.add(metrics, end - start)
        logits.append(predicted_logits.float().cpu().numpy())
    components = accumulator.result()
    _require_finite_components(components, "evaluation")
    predictions = np.concatenate(logits, axis=0)
    actions = actionable_policy_metrics_numpy(
        predictions,
        subset.target_policy.detach().cpu().numpy(),
        action_grid.detach().cpu().numpy().astype(np.float64),
        reset_mask=subset.reset_mask,
        friction=friction,
        temperature=temperature,
        execution_policy=resolved_action_objective.get("executionPolicy"),
    )
    return {
        "components": components,
        "actions": _json_finite(actions),
    }


def run_fixed_subset_overfit(
    model: nn.Module,
    subset: FixedTrainingSubset,
    action_grid: Tensor,
    training: dict,
    action_objective: dict,
    settings: OverfitSettings,
    *,
    friction: float,
    temperature: float,
) -> dict[str, Any]:
    _validate_settings(settings, subset.count)
    _validate_subset_finite(subset)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if settings.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostic requested but CUDA is unavailable")
        torch.cuda.manual_seed_all(settings.seed)
    model = model.to(settings.device)
    grid = action_grid.to(settings.device)
    source_switch_fraction = fixed_subset_source_switch_fraction(
        subset,
        action_grid,
        action_objective,
        friction=friction,
        temperature=temperature,
    )
    initial = evaluate_fixed_subset(
        model,
        subset,
        grid,
        training,
        action_objective,
        device=settings.device,
        batch_size=settings.batch_size,
        source_switch_fraction=source_switch_fraction,
        friction=friction,
        temperature=temperature,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    last_gradient_norm = 0.0
    for step in range(settings.steps):
        start = (step * settings.batch_size) % subset.count
        indexes = torch.arange(
            start,
            start + settings.batch_size,
            dtype=torch.long,
        ) % subset.count
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(settings.device, training):
            metrics, _predicted_logits = _batch_metrics(
                model,
                subset.input_closes.index_select(0, indexes).to(settings.device),
                subset.future_closes.index_select(0, indexes).to(settings.device),
                subset.target_policy.index_select(0, indexes).to(settings.device),
                subset.teacher_current_exposures.index_select(0, indexes).to(
                    settings.device
                ),
                grid,
                training,
                action_objective,
                source_switch_fraction=source_switch_fraction,
                friction=friction,
                temperature=temperature,
            )
        loss = metrics["loss"]
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite diagnostic loss at step {step + 1}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(gradient is None for gradient in gradients):
            raise RuntimeError("diagnostic found disconnected trainable parameters")
        assert all(gradient is not None for gradient in gradients)
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"non-finite diagnostic gradient at step {step + 1}")
        gradient_norm = clip_grad_norm_(
            model.parameters(),
            settings.gradient_clip,
            error_if_nonfinite=True,
        )
        last_gradient_norm = float(gradient_norm)
        if not math.isfinite(last_gradient_norm) or last_gradient_norm <= 0:
            raise RuntimeError(f"invalid diagnostic gradient norm at step {step + 1}")
        optimizer.step()
    final = evaluate_fixed_subset(
        model,
        subset,
        grid,
        training,
        action_objective,
        device=settings.device,
        batch_size=settings.batch_size,
        source_switch_fraction=source_switch_fraction,
        friction=friction,
        temperature=temperature,
    )
    initial_loss = float(initial["components"]["loss"])
    final_loss = float(final["components"]["loss"])
    absolute_improvement = initial_loss - final_loss
    relative_improvement = absolute_improvement / max(abs(initial_loss), 1e-12)
    if relative_improvement < settings.minimum_relative_improvement:
        raise RuntimeError(
            "fixed-subset loss failed its improvement gate: "
            f"{relative_improvement:.6f} < "
            f"{settings.minimum_relative_improvement:.6f}"
        )
    return {
        "initial": initial,
        "final": final,
        "absoluteLossImprovement": absolute_improvement,
        "relativeLossImprovement": relative_improvement,
        "lastGradientNorm": last_gradient_norm,
    }


def run_plan_diagnostic(
    plan_file: Path,
    settings: OverfitSettings,
    examples: int,
) -> dict[str, Any]:
    if examples < 1 or examples > MAXIMUM_EXAMPLES:
        raise ValueError(
            f"examples must be in [1, {MAXIMUM_EXAMPLES}]"
        )
    _validate_settings(settings, examples)
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = resolve(repo_root, plan_file)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    resolved_training = resolve_training_config(plan["training"])
    action_objective = resolve_action_objective_config(
        resolved_training.get("actionObjective", {})
    )
    storage = training_storage_layout(repo_root)
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
    model_config = plan["model"]
    segments, target_manifest, _excluded, _fingerprint = load_causal_segments(
        target_root,
        plan,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
    )
    selected = chronological_training_head(segments["train"], examples)
    dataset = CausalOracleDataset(
        history_root,
        {"train": selected},
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
        target_rows_per_file=1_440,
        action_count=int(model_config["actionCount"]),
        close_cache_days=int(resolved_training["closeCacheDays"]),
        target_cache_days=int(resolved_training["targetCacheDays"]),
        pin_memory=False,
    )
    contract = target_manifest["contract"]
    grid_numpy = np.asarray(contract["usableGrid"], dtype=np.float64)
    friction = float(contract["options"]["friction"])
    temperature = float(contract["options"]["temperature"])
    subset = materialize_fixed_training_subset(
        dataset,
        selected,
        grid_numpy,
        friction=friction,
        temperature=temperature,
        execution_policy=action_objective.get("executionPolicy"),
    )
    # Seed before construction so repeated diagnostics start from identical
    # parameters as well as identical fixed-subset update order.
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    result = run_fixed_subset_overfit(
        build_model(model_config),
        subset,
        torch.from_numpy(grid_numpy.astype(np.float32)),
        resolved_training,
        action_objective,
        settings,
        friction=friction,
        temperature=temperature,
    )
    return {
        "version": 1,
        "kind": "joint-price-oracle-fixed-subset-overfit",
        "planId": plan["id"],
        "modelVariant": model_config.get("variant", "legacy"),
        "split": "train",
        "chronological": True,
        "examples": subset.count,
        "firstPredictionTime": selected[0].prediction_time_start,
        "lastPredictionTime": selected[-1].prediction_time_end,
        "steps": settings.steps,
        "batchSize": settings.batch_size,
        "device": str(settings.device),
        "writesCheckpointsOrArtifacts": False,
        **result,
    }


def _validate_settings(settings: OverfitSettings, examples: int) -> None:
    if settings.steps < 1 or settings.steps > MAXIMUM_STEPS:
        raise ValueError(f"steps must be in [1, {MAXIMUM_STEPS}]")
    if settings.batch_size < 1 or settings.batch_size > examples:
        raise ValueError("batch size must be in [1, examples]")
    for name, value in (
        ("learning rate", settings.learning_rate),
        ("gradient clip", settings.gradient_clip),
    ):
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")
    if settings.weight_decay < 0 or not math.isfinite(settings.weight_decay):
        raise ValueError("weight decay must be finite and non-negative")
    if settings.minimum_relative_improvement < 0 \
            or not math.isfinite(settings.minimum_relative_improvement):
        raise ValueError("minimum relative improvement must be finite and non-negative")


def _require_finite_components(values: dict[str, float], label: str) -> None:
    if not values or any(not math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"{label} produced non-finite component metrics")


def _validate_subset_finite(subset: FixedTrainingSubset) -> None:
    tensors = (
        subset.input_closes,
        subset.future_closes,
        subset.target_policy,
        subset.teacher_current_exposures,
    )
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("fixed training subset contains non-finite values")


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def settings_from_arguments(
    arguments: argparse.Namespace,
    training: dict,
) -> OverfitSettings:
    return OverfitSettings(
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        learning_rate=float(
            arguments.learning_rate
            if arguments.learning_rate is not None
            else training["learningRate"]
        ),
        weight_decay=float(
            arguments.weight_decay
            if arguments.weight_decay is not None
            else training.get("weightDecay", 0.0)
        ),
        gradient_clip=float(
            arguments.gradient_clip
            if arguments.gradient_clip is not None
            else training["gradientClip"]
        ),
        minimum_relative_improvement=arguments.minimum_relative_improvement,
        seed=int(
            arguments.seed
            if arguments.seed is not None
            else training["seed"]
        ),
        device=torch.device(arguments.device),
    )


def main() -> None:
    arguments = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = resolve(repo_root, arguments.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    settings = settings_from_arguments(arguments, plan["training"])
    report = run_plan_diagnostic(plan_path, settings, arguments.examples)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
