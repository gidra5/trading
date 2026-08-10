from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from return_oracle_ce import (
    BRANCH_NORMALIZATION_INITIAL_RADIUS,
    INPUT_RETURN_COUNT,
    LearnableCenteringNorm,
    OUTPUT_ACTION_COUNT,
    centering_matrix_constraint_components,
    dropout_gate_probability,
    fused_glu,
    soft_layer_norm_components,
    soft_weight_bound_penalty,
)


SCREEN_SCHEMA_VERSION = 1
PRODUCTION_ORACLE_TEMPERATURE = 0.01
FINAL_DATASET_COUNTS = {
    "train": 33_195_600,
    "validation": 15_418_800,
    "test": 1_000_000,
}
FINAL_DATASET_SHA256 = (
    "cc5bddd8948390713e68505236a0db8b17510c4cf01b960f0fd6df19b0a4fbd8"
)
FINAL_SOURCE_DATASET_SHA256 = (
    "215888a941a8888bc3b3ceb85d979092b94ee273c844c81a179d2f0e3308f85f"
)
FEATURE_CONTRACT = (
    "train-position-standardized-completed-minute-close-only-simple-returns-v2"
)
SELECTION_CONTRACT = "raw-uncalibrated-base-action-validation-kl-at-0.01-v1"
RUNNER_CONTRACT = "compact-minute-decoder-screen-sealed-test-v1"
LEARNED_RADIUS_WIDTHS = tuple(range(512, 271, -16))
LEARNED_RADIUS_ARCHITECTURE = (
    "shrinking-fused-glu-independent-branch-centering-sqrt-"
    "learned-radius-full-a-post-bias-v16"
)
RESIDUAL_GLU_ARCHITECTURE = "pre-ln-residual-glu-8x256-v1"


@dataclass(frozen=True)
class TemperatureStage:
    start_epoch: int
    temperature: float


def canonical_plan_fingerprint(plan: dict[str, Any]) -> str:
    encoded = json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def temperature_stages(plan: dict[str, Any]) -> tuple[TemperatureStage, ...]:
    return tuple(
        TemperatureStage(
            start_epoch=int(stage["startEpoch"]),
            temperature=float(stage["temperature"]),
        )
        for stage in plan["curriculum"]["stages"]
    )


def training_temperature(plan: dict[str, Any], epoch: int) -> float:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    selected: TemperatureStage | None = None
    for stage in temperature_stages(plan):
        if stage.start_epoch > epoch:
            break
        selected = stage
    if selected is None:
        raise ValueError("temperature curriculum must start at epoch zero")
    return selected.temperature


def target_temperature(
    plan: dict[str, Any],
    epoch: int,
    gate_epoch: int | None = None,
) -> float:
    curriculum_type = plan["curriculum"].get("type")
    if curriculum_type == "target-probability-temperature-power":
        return training_temperature(plan, epoch)
    raise ValueError(f"unsupported temperature curriculum: {curriculum_type}")


def next_equal_entropy_step_training_temperature(
    plan: dict[str, Any],
    current_temperature: float,
    validation_kl: float,
    validation_target_entropy: float,
    production_target_entropy: float,
    start_target_entropy: float,
    entropy_log_temperature_derivative: float,
) -> tuple[float, float]:
    """Take one fixed validation-entropy step after the KL gate passes."""
    curriculum = plan["curriculum"]
    final_temperature = float(curriculum["finalTemperature"])
    entropy_intervals = int(curriculum["entropySchedulePoints"]) - 1
    entropy_decrease = (
        start_target_entropy - production_target_entropy
    ) / entropy_intervals
    next_target_entropy = max(
        production_target_entropy,
        validation_target_entropy - entropy_decrease,
    )
    threshold = float(curriculum["gate"]["threshold"])
    if validation_kl > threshold:
        return current_temperature, validation_target_entropy
    if next_target_entropy >= validation_target_entropy \
            or current_temperature <= final_temperature:
        return current_temperature, production_target_entropy
    minimum_derivative = float(
        curriculum["solver"]["minimumEntropyDerivative"]
    )
    derivative = max(
        entropy_log_temperature_derivative,
        minimum_derivative,
    )
    log_temperature_change = (
        next_target_entropy - validation_target_entropy
    ) / derivative
    next_temperature = current_temperature * math.exp(log_temperature_change)
    return (
        max(final_temperature, min(current_temperature, next_temperature)),
        next_target_entropy,
    )


def next_stale_learning_rate(
    schedule: dict[str, Any],
    current_learning_rate: float,
    stale_epochs: int,
) -> float:
    """Decay after each complete block of stale production-KL epochs."""
    if current_learning_rate <= 0 or not math.isfinite(current_learning_rate):
        raise ValueError("current learning rate must be finite and positive")
    stale_block = int(schedule["staleEpochsPerReduction"])
    qualifying_stale_epochs = int(schedule["qualifyingStaleEpochs"])
    if stale_epochs <= 0 or stale_epochs > qualifying_stale_epochs:
        return current_learning_rate
    if stale_epochs == qualifying_stale_epochs:
        final_block = qualifying_stale_epochs % stale_block
        if final_block == 0:
            final_block = stale_block
        decay_factor = float(schedule["perEpochDecayFactor"]) ** final_block
    elif stale_epochs % stale_block == 0:
        decay_factor = float(schedule["decayFactor"])
    else:
        return current_learning_rate
    return max(
        float(schedule["finalLearningRate"]),
        current_learning_rate * decay_factor,
    )


def smooth_oracle_probabilities(
    probabilities: Tensor,
    *,
    source_temperature: float,
    target_temperature: float,
) -> Tensor:
    """Re-temperature stored softmax probabilities without inventing support."""
    if probabilities.ndim != 2:
        raise ValueError("oracle probabilities must be [example, action]")
    if source_temperature <= 0 or target_temperature <= 0:
        raise ValueError("oracle temperatures must be positive")
    target = probabilities.float().clamp_min(0)
    row_sum = target.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(target).all()) or bool((row_sum <= 0).any()):
        raise ValueError("oracle probabilities must be finite with positive rows")
    normalized = target / row_sum
    if math.isclose(
        source_temperature,
        target_temperature,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        return normalized
    exponent = source_temperature / target_temperature
    negative_infinity = torch.full_like(normalized, -torch.inf)
    log_probabilities = torch.where(
        normalized > 0,
        normalized.log(),
        negative_infinity,
    )
    return torch.softmax(log_probabilities * exponent, dim=-1)


def production_entropy_confidence_weights(
    target_probabilities: Tensor,
    example_weights: Tensor,
    strength: float,
) -> Tensor:
    """Upweight sharper production targets without dropping examples."""
    if target_probabilities.ndim != 2:
        raise ValueError("target probabilities must be [example, action]")
    if example_weights.shape != (target_probabilities.shape[0],):
        raise ValueError("example weights must contain one value per example")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("confidence-weighting strength must be finite and nonnegative")
    target = target_probabilities.float().clamp_min(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    target_log = torch.where(
        target > 0,
        target.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(target),
    )
    target_entropy = -(target * target_log).sum(dim=-1)
    return example_weights.float().clamp_min(0) * torch.exp(
        -float(strength) * target_entropy
    )


def weighted_target_entropy_log_temperature_derivative(
    source_probabilities: Tensor,
    target_probabilities: Tensor,
    example_weights: Tensor,
    *,
    source_temperature: float,
    target_temperature: float,
) -> Tensor:
    """Derivative of mean target entropy with respect to log temperature."""
    if source_probabilities.ndim != 2 \
            or target_probabilities.shape != source_probabilities.shape:
        raise ValueError("source and target probabilities must match")
    if example_weights.shape != (source_probabilities.shape[0],):
        raise ValueError("example weights must contain one value per example")
    source = source_probabilities.float().clamp_min(0)
    source = source / source.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    target = target_probabilities.float().clamp_min(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    source_log = torch.where(
        source > 0,
        source.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(source),
    )
    expected_log = (target * source_log).sum(dim=-1, keepdim=True)
    log_variance = (
        target * (source_log - expected_log).square()
    ).sum(dim=-1)
    exponent = source_temperature / target_temperature
    per_example_derivative = exponent * exponent * log_variance
    weights = example_weights.float().clamp_min(0)
    return (per_example_derivative * weights).sum() / weights.sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )


def weighted_policy_metrics(
    predicted_logits: Tensor,
    target_probabilities: Tensor,
    example_weights: Tensor,
) -> dict[str, Tensor]:
    """Raw production-target diagnostics used for all screen selection."""
    if predicted_logits.ndim != 2 \
            or target_probabilities.shape != predicted_logits.shape:
        raise ValueError("logits and targets must have matching [example, action]")
    if predicted_logits.shape[-1] != OUTPUT_ACTION_COUNT:
        raise ValueError("decoder output must use the 255-action oracle grid")
    if example_weights.shape != (predicted_logits.shape[0],):
        raise ValueError("example weights must contain one value per example")
    target = target_probabilities.float().clamp_min(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    predicted_log = torch.log_softmax(predicted_logits.float(), dim=-1)
    predicted_probabilities = predicted_log.exp()
    cross_entropy = -(target * predicted_log).sum(dim=-1)
    target_log = torch.where(
        target > 0,
        target.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(target),
    )
    target_entropy = -(target * target_log).sum(dim=-1)
    predicted_entropy = -(predicted_probabilities * predicted_log).sum(dim=-1)
    probability_mse = (predicted_probabilities - target).square().mean(dim=-1)
    weights = example_weights.float().clamp_min(0)
    weight_sum = weights.sum().clamp_min(torch.finfo(torch.float32).tiny)

    def mean(value: Tensor) -> Tensor:
        return (value * weights).sum() / weight_sum

    return {
        "rawCrossEntropy": mean(cross_entropy),
        "rawBaseActionKl": mean(cross_entropy - target_entropy),
        "rawProbabilityMse": mean(probability_mse),
        "rawTargetEntropy": mean(target_entropy),
        "rawPredictedEntropy": mean(predicted_entropy),
    }


class ResidualGluDecoder(nn.Module):
    """Small, conventional pre-LN residual GLU capability baseline."""

    architecture_contract = RESIDUAL_GLU_ARCHITECTURE

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        *,
        width: int = 256,
        depth: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        _validate_feature_statistics(feature_mean, feature_std)
        if width <= 0 or depth <= 0:
            raise ValueError("residual GLU width and depth must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())
        self.input = nn.Linear(INPUT_RETURN_COUNT, width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.layers = nn.ModuleList([
            nn.Linear(width, 2 * width) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, OUTPUT_ACTION_COUNT)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = 1.0 / math.sqrt(depth)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.input.weight, nonlinearity="linear")
        nn.init.zeros_(self.input.bias)
        for norm in (*self.norms, self.final_norm):
            norm.reset_parameters()
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (self.input.weight, *(layer.weight for layer in self.layers))

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.input(
            (features.float() - self.feature_mean) / self.feature_std
        )
        for norm, layer in zip(self.norms, self.layers, strict=True):
            value, gate = layer(norm(hidden)).chunk(2, dim=-1)
            branch = value * torch.sigmoid(gate)
            hidden = hidden + self.residual_scale * self.dropout(branch)
        return self.output(self.final_norm(hidden))


class LearnedRadiusShrinkingDecoder(nn.Module):
    """Recovered 0.0712-KL decoder architecture with checkpoint-compatible keys."""

    architecture_contract = LEARNED_RADIUS_ARCHITECTURE

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        *,
        dropout: float = 0.05,
        dropout_rate: float = 0.5,
        initial_radius: float = BRANCH_NORMALIZATION_INITIAL_RADIUS,
        minimum_radius: float = 1e-4,
        output_count: int = OUTPUT_ACTION_COUNT,
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 \
                or feature_std.shape != feature_mean.shape \
                or feature_mean.numel() < 1:
            raise ValueError(
                "feature normalization must contain matching non-empty vectors"
            )
        if not bool(torch.isfinite(feature_mean).all()) \
                or not bool(torch.isfinite(feature_std).all()) \
                or not bool((feature_std > 0).all()):
            raise ValueError(
                "feature normalization must be finite with positive scales"
            )
        if not 0 <= dropout < 1 or not 0 <= dropout_rate <= 1:
            raise ValueError("dropout settings are invalid")
        if output_count < 1:
            raise ValueError("decoder output count must be positive")
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())
        self.dropout_rate = float(dropout_rate)
        self.dropout_gate_probability = dropout_gate_probability(
            self.dropout_rate
        )
        widths = (feature_mean.numel(), *LEARNED_RADIUS_WIDTHS)
        self.layers = nn.ModuleList([
            nn.Linear(input_width, 2 * output_width)
            for input_width, output_width in zip(
                widths[:-1], widths[1:], strict=True
            )
        ])
        self.value_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                width,
                denominator_family="sqrt",
                initial_scale=initial_radius,
                minimum_scale=minimum_radius,
            )
            for width in LEARNED_RADIUS_WIDTHS
        ])
        self.gate_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                width,
                denominator_family="sqrt",
                initial_scale=initial_radius,
                minimum_scale=minimum_radius,
            )
            for width in LEARNED_RADIUS_WIDTHS
        ])
        self.value_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(width))
            for width in LEARNED_RADIUS_WIDTHS
        ])
        self.gate_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(width))
            for width in LEARNED_RADIUS_WIDTHS
        ])
        self.value_transforms = nn.ModuleList([
            nn.Linear(width, width, bias=False)
            for width in LEARNED_RADIUS_WIDTHS
        ])
        self.gate_transforms = nn.ModuleList([
            nn.Linear(width, width, bias=False)
            for width in LEARNED_RADIUS_WIDTHS
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(LEARNED_RADIUS_WIDTHS[-1], output_count)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
            nn.init.zeros_(layer.bias)
        for normalizer in (
            *self.value_centering_normalizers,
            *self.gate_centering_normalizers,
        ):
            normalizer.reset_parameters()
        for bias in (*self.value_norm_biases, *self.gate_norm_biases):
            nn.init.zeros_(bias)
        for transform in (*self.value_transforms, *self.gate_transforms):
            nn.init.eye_(transform.weight)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            *(layer.weight for layer in self.layers),
            *(layer.weight for layer in self.value_transforms),
            *(layer.weight for layer in self.gate_transforms),
        )

    def forward_with_regularizers(
        self,
        features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = (features.float() - self.feature_mean) / self.feature_std
        intermittent_dropout = (
            self.training
            and self.dropout.p > 0
            and 0 < self.dropout_rate < 1
        )
        pass_gate = (
            torch.rand((), device=hidden.device)
            < self.dropout_gate_probability
            if intermittent_dropout
            else None
        )
        mean_penalties: list[Tensor] = []
        variance_penalties: list[Tensor] = []
        for (
            layer,
            value_normalizer,
            gate_normalizer,
            value_bias,
            gate_bias,
            value_transform,
            gate_transform,
        ) in zip(
            self.layers,
            self.value_centering_normalizers,
            self.gate_centering_normalizers,
            self.value_norm_biases,
            self.gate_norm_biases,
            self.value_transforms,
            self.gate_transforms,
            strict=True,
        ):
            hidden, raw_value, raw_gate = fused_glu(
                layer(hidden),
                value_normalizer,
                gate_normalizer,
                value_bias,
                gate_bias,
                value_transform,
                gate_transform,
            )
            branch_components = tuple(
                soft_layer_norm_components(branch)
                for branch in (raw_value, raw_gate)
            )
            mean_penalties.append(torch.stack(tuple(
                mean for mean, _variance in branch_components
            )).mean(dim=0))
            variance_penalties.append(torch.stack(tuple(
                variance for _mean, variance in branch_components
            )).mean(dim=0))
            if self.training and self.dropout.p > 0:
                if self.dropout_rate >= 1:
                    hidden = self.dropout(hidden)
                elif intermittent_dropout:
                    layer_gate = (
                        torch.rand((), device=hidden.device)
                        < self.dropout_gate_probability
                    )
                    hidden = torch.where(
                        pass_gate & layer_gate,
                        self.dropout(hidden),
                        hidden,
                    )
        return (
            self.output(hidden),
            torch.stack(mean_penalties).mean(dim=0),
            torch.stack(variance_penalties).mean(dim=0),
        )

    def regularizer_parameter_penalties(
        self,
        *,
        desired_weight_magnitude: float,
        weight_bound_sharpness: float,
        absolute_epsilon: float,
        include_centering_constraint: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bounded_weights = tuple(
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.ndim == 2
        )
        weight_bound = soft_weight_bound_penalty(
            bounded_weights,
            desired_magnitude=desired_weight_magnitude,
            sharpness=weight_bound_sharpness,
            absolute_epsilon=absolute_epsilon,
        )
        if not include_centering_constraint:
            zero = weight_bound.new_zeros(())
            return weight_bound, zero, zero
        centering_matrices = tuple(
            normalizer.weight
            for normalizer in (
                *self.value_centering_normalizers,
                *self.gate_centering_normalizers,
            )
        )
        idempotence, symmetry = centering_matrix_constraint_components(
            centering_matrices
        )
        return weight_bound, idempotence, symmetry

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_with_regularizers(features)[0]


DecoderModel = ResidualGluDecoder | LearnedRadiusShrinkingDecoder


def build_decoder(
    plan: dict[str, Any],
    feature_mean: Tensor,
    feature_std: Tensor,
) -> DecoderModel:
    architecture = plan["architecture"]
    architecture_type = architecture["type"]
    if architecture_type == "residual-glu":
        return ResidualGluDecoder(
            feature_mean,
            feature_std,
            width=int(architecture["width"]),
            depth=int(architecture["depth"]),
            dropout=float(architecture["dropout"]),
        )
    if architecture_type == "learned-radius-shrinking":
        return LearnedRadiusShrinkingDecoder(
            feature_mean,
            feature_std,
            dropout=float(architecture["dropout"]),
            dropout_rate=float(architecture["dropoutRate"]),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
        )
    raise ValueError(f"unsupported decoder architecture: {architecture_type}")


def optimizer_parameter_groups(
    model: DecoderModel,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    muon = tuple(model.muon_parameters())
    muon_ids = {id(parameter) for parameter in muon}
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    adamw = tuple(
        parameter for parameter in trainable if id(parameter) not in muon_ids
    )
    if not muon or not adamw or len(muon_ids) != len(muon):
        raise RuntimeError("invalid decoder optimizer routing")
    if {id(value) for value in (*muon, *adamw)} != {
        id(value) for value in trainable
    }:
        raise RuntimeError("decoder optimizer routing is incomplete")
    if any(parameter.ndim != 2 for parameter in muon):
        raise RuntimeError("Muon parameters must be matrices")
    return muon, adamw


def validate_screen_plan(plan: dict[str, Any]) -> None:
    if plan.get("schemaVersion") != SCREEN_SCHEMA_VERSION:
        raise ValueError("decoder screen plan schema is invalid")
    required = ("id", "label", "dataset", "runDir", "architecture",
                "curriculum", "objective", "selection", "training")
    if any(not plan.get(field) for field in required):
        raise ValueError("decoder screen plan is missing required fields")
    dataset = plan["dataset"]
    if dataset.get("featureContract") != FEATURE_CONTRACT \
            or dataset.get("expectedCounts") != FINAL_DATASET_COUNTS \
            or str(dataset.get("datasetSha256", "")).lower() \
            != FINAL_DATASET_SHA256 \
            or str(dataset.get("sourceDatasetSha256", "")).lower() \
            != FINAL_SOURCE_DATASET_SHA256 \
            or float(dataset.get("oracleTemperature", -1)) \
            != PRODUCTION_ORACLE_TEMPERATURE \
            or dataset.get("testPolicy") != "sealed-never-load":
        raise ValueError("decoder screen must use the frozen final corpus")
    selection = plan["selection"]
    if selection != {
        "split": "validation",
        "metric": "rawBaseActionKl",
        "mode": "min",
        "oracleTemperature": PRODUCTION_ORACLE_TEMPERATURE,
        "predictionCalibration": "none",
        "contract": SELECTION_CONTRACT,
    }:
        raise ValueError("decoder screen selection must be raw exact-temperature KL")
    objective = plan["objective"]
    objective_base = {
        "type": "soft-target-cross-entropy",
        "trainingTargets": "curriculum-temperature",
        "validationTargets": "stored-production-0.01",
    }
    if any(objective.get(key) != value for key, value in objective_base.items()):
        raise ValueError("decoder screen objective contract is invalid")
    training_weighting = objective.get("trainingExampleWeighting")
    if training_weighting is not None:
        strength = training_weighting.get("lambda")
        if training_weighting.get("type") \
                != "production-target-entropy-confidence" \
                or training_weighting.get("validationWeights") != "unchanged" \
                or isinstance(strength, bool) \
                or not isinstance(strength, (int, float)) \
                or not math.isfinite(float(strength)) \
                or float(strength) <= 0:
            raise ValueError("decoder training example weighting is invalid")
    regularizers = objective.get("regularizers")
    restored_regularizers = {
        "softLayerNorm": {
            "weight": 1,
            "varianceWeight": 1,
            "measurement": "raw-value-and-gate-projections",
        },
        "softWeightBound": {
            "weight": 0.01,
            "desiredMagnitude": 1,
            "sharpness": 10,
            "absoluteEpsilon": 1e-8,
        },
        "centeringConstraint": {
            "idempotenceWeight": 1,
            "symmetryWeight": 1,
            "matrices": "independent-value-and-gate",
        },
    }
    restored_without_centering = {
        key: value
        for key, value in restored_regularizers.items()
        if key != "centeringConstraint"
    }
    if regularizers not in (
        [], restored_regularizers, restored_without_centering
    ):
        raise ValueError("decoder screen regularizer contract is invalid")
    curriculum = plan["curriculum"]
    curriculum_type = curriculum.get("type")
    if curriculum_type not in {
        "target-probability-temperature-power",
        "kl-gated-equal-validation-entropy-steps-target-probability-temperature",
    } or float(curriculum.get("sourceTemperature", -1)) \
            != PRODUCTION_ORACLE_TEMPERATURE:
        raise ValueError("decoder temperature curriculum contract is invalid")
    training = plan["training"]
    if curriculum_type == "target-probability-temperature-power":
        stages = temperature_stages(plan)
        if not stages or stages[0].start_epoch != 0 \
                or any(stage.start_epoch < 0 or stage.temperature <= 0
                       for stage in stages) \
                or any(right.start_epoch <= left.start_epoch
                       for left, right in zip(stages, stages[1:])) \
                or stages[-1].temperature != PRODUCTION_ORACLE_TEMPERATURE:
            raise ValueError(
                "curriculum stages must end at production temperature"
            )
        final_temperature_epoch = stages[-1].start_epoch
    else:
        gate = curriculum.get("gate", {})
        entropy_schedule_points = curriculum.get("entropySchedulePoints")
        solver = curriculum.get("solver", {})
        if isinstance(entropy_schedule_points, bool) \
                or not isinstance(entropy_schedule_points, int) \
                or entropy_schedule_points < 2 \
                or float(curriculum.get("startTemperature", 0)) \
                <= PRODUCTION_ORACLE_TEMPERATURE \
                or float(curriculum.get("finalTemperature", -1)) \
                != PRODUCTION_ORACLE_TEMPERATURE \
                or gate.get("metric") != "curriculumTargetKl" \
                or gate.get("evaluation") != "every-epoch" \
                or not 0 < float(gate.get("threshold", 0)) < 1 \
                or solver.get("type") != "log-temperature-newton" \
                or float(solver.get("minimumEntropyDerivative", 0)) <= 0:
            raise ValueError("KL-gated temperature curriculum is invalid")
        final_temperature_epoch = entropy_schedule_points - 1
    positive = (
        "epochs", "batchSize", "evaluationBatchSize",
        "gradientAccumulationSteps", "learningRate", "gradientClip",
        "seed", "workers", "prefetchFactor",
    )
    if any(float(training.get(field, 0)) <= 0 for field in positive):
        raise ValueError("decoder screen training settings are invalid")
    if final_temperature_epoch >= int(training["epochs"]):
        raise ValueError("screen must train at production temperature")
    if training.get("mixedPrecision") != "bfloat16":
        raise ValueError("decoder screen requires bfloat16 mixed precision")
    optimizer = training.get("optimizer", {})
    if optimizer.get("type") != "hybrid-muon-adamw":
        raise ValueError("decoder screen optimizer contract is invalid")
    schedule = training.get("learningRateSchedule", {})
    schedule_type = schedule.get("type")
    if schedule_type == "reduce-on-raw-validation-kl-plateau":
        valid_schedule = (
            int(schedule.get("startEpoch", -1)) >= 0
            and 0 < float(schedule.get("factor", 0)) < 1
            and int(schedule.get("patience", 0)) >= 1
            and float(schedule.get("threshold", -1)) >= 0
            and 0 < float(schedule.get("minimumLearningRate", 0))
            <= float(training["learningRate"])
        )
    elif schedule_type == "decay-every-prod-kl-stale-block":
        qualifying_stale_epochs = int(schedule.get("qualifyingStaleEpochs", 0))
        stale_epochs_per_reduction = int(
            schedule.get("staleEpochsPerReduction", 0)
        )
        start_learning_rate = float(schedule.get("startLearningRate", 0))
        final_learning_rate = float(schedule.get("finalLearningRate", 0))
        per_epoch_decay_factor = float(schedule.get("perEpochDecayFactor", 0))
        decay_factor = float(schedule.get("decayFactor", 0))
        valid_schedule = (
            schedule.get("metric") == "rawBaseActionKl"
            and math.isclose(
                start_learning_rate,
                float(training["learningRate"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            and 0 < final_learning_rate < start_learning_rate
            and qualifying_stale_epochs >= 1
            and 1 <= stale_epochs_per_reduction <= qualifying_stale_epochs
            and 0 < per_epoch_decay_factor < 1
            and 0 < decay_factor < 1
            and math.isclose(
                start_learning_rate
                * per_epoch_decay_factor ** qualifying_stale_epochs,
                final_learning_rate,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            and math.isclose(
                per_epoch_decay_factor ** stale_epochs_per_reduction,
                decay_factor,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
    else:
        valid_schedule = False
    if not valid_schedule:
        raise ValueError("decoder screen LR schedule is invalid")
    early_stopping_patience = training.get("earlyStoppingPatience")
    if early_stopping_patience is not None \
            and (
                isinstance(early_stopping_patience, bool)
                or not isinstance(early_stopping_patience, int)
                or early_stopping_patience < 1
            ):
        raise ValueError(
            "decoder earlyStoppingPatience must be a positive integer"
        )
    architecture = plan["architecture"]
    if regularizers and architecture.get("type") != "learned-radius-shrinking":
        raise ValueError("restored regularizers require learned-radius architecture")
    if architecture.get("type") == "residual-glu":
        if architecture.get("contract") != RESIDUAL_GLU_ARCHITECTURE \
                or int(architecture.get("width", 0)) <= 0 \
                or int(architecture.get("depth", 0)) <= 0 \
                or not 0 <= float(architecture.get("dropout", -1)) < 1:
            raise ValueError("residual GLU architecture is invalid")
    elif architecture.get("type") == "learned-radius-shrinking":
        if architecture.get("contract") != LEARNED_RADIUS_ARCHITECTURE \
                or architecture.get("widths") != list(LEARNED_RADIUS_WIDTHS) \
                or not 0 <= float(architecture.get("dropout", -1)) < 1 \
                or not 0 <= float(architecture.get("dropoutRate", -1)) <= 1 \
                or float(architecture.get("minimumRadius", 0)) <= 0 \
                or float(architecture.get("initialRadius", 0)) \
                <= float(architecture.get("minimumRadius", 0)) \
                or not math.isclose(
                    float(architecture.get("initialRadius", 0)),
                    BRANCH_NORMALIZATION_INITIAL_RADIUS,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
            raise ValueError("learned-radius architecture is invalid")
    else:
        raise ValueError("decoder screen architecture is unsupported")


def _validate_feature_statistics(feature_mean: Tensor, feature_std: Tensor) -> None:
    if feature_mean.shape != (INPUT_RETURN_COUNT,) \
            or feature_std.shape != (INPUT_RETURN_COUNT,):
        raise ValueError("feature statistics must match the 60-return input")
    if not bool(torch.isfinite(feature_mean).all()) \
            or not bool(torch.isfinite(feature_std).all()) \
            or not bool((feature_std > 0).all()):
        raise ValueError("feature statistics must be finite and positive")
