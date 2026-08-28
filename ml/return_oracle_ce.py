from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


INPUT_RETURN_COUNT = 60
OUTPUT_ACTION_COUNT = 255
HIDDEN_WIDTHS = (256,) * 8
HIDDEN_LAYER_COUNT = len(HIDDEN_WIDTHS)
BRANCH_NORMALIZATION_EPSILON = 1e-5
BRANCH_NORMALIZATION_INITIAL_RADIUS = math.sqrt(
    BRANCH_NORMALIZATION_EPSILON
)


def dropout_gate_probability(application_rate: float) -> float:
    """Split one marginal rate across pass and layer Bernoulli gates."""
    if not 0 <= application_rate <= 1:
        raise ValueError("dropout application rate must be in [0, 1]")
    return application_rate ** 0.5


def soft_layer_norm_components(hidden: Tensor) -> tuple[Tensor, Tensor]:
    """Per-example squared mean and unit-variance penalties."""
    if hidden.ndim != 2:
        raise ValueError("soft LayerNorm expects [example, neuron] activations")
    variance, mean = torch.var_mean(
        hidden.float(),
        dim=-1,
        correction=0,
    )
    return mean.square(), (variance - 1.0).square()


def distribution_layer_components(hidden: Tensor) -> tuple[Tensor, Tensor]:
    """Width-normalized per-example unit-sum and non-negativity penalties."""
    if hidden.ndim != 2:
        raise ValueError(
            "distribution layer expects [example, neuron] activations"
        )
    activations = hidden.float()
    width = activations.shape[-1]
    return (
        (activations.mean(dim=-1) - 1.0 / width).square(),
        functional.relu(-activations).square().mean(dim=-1),
    )


class LearnableCenteringNorm(nn.Module):
    """Apply fixed or learnable centering and a learned-radius RMS family."""

    def __init__(
        self,
        width: int,
        *,
        denominator_family: str = "sqrt",
        initial_scale: float = BRANCH_NORMALIZATION_INITIAL_RADIUS,
        minimum_scale: float = 1e-4,
        learnable_centering: bool = True,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("soft normalization width must be positive")
        if denominator_family not in {"sqrt", "tanh"}:
            raise ValueError(
                f"unsupported soft normalization family: "
                f"{denominator_family}"
            )
        if minimum_scale <= 0 or initial_scale <= minimum_scale:
            raise ValueError(
                "soft normalization requires initial scale above its "
                "positive minimum"
            )
        self.normalized_shape = (width,)
        self.denominator_family = denominator_family
        self.initial_scale = float(initial_scale)
        self.minimum_scale = float(minimum_scale)
        self.learnable_centering = bool(learnable_centering)
        if self.learnable_centering:
            self.weight = nn.Parameter(torch.empty(width, width))
        else:
            self.register_parameter("weight", None)
        self.raw_scale = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        width = self.normalized_shape[0]
        if self.weight is not None:
            with torch.no_grad():
                self.weight.copy_(
                    torch.eye(width, device=self.weight.device)
                    - torch.full(
                        (width, width),
                        1.0 / width,
                        device=self.weight.device,
                    )
                )
        self.reset_scale()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Older fixed-centering checkpoints stored the canonical width-by-width
        # projector. It is now represented exactly by mean subtraction.
        if self.weight is None:
            state_dict.pop(f"{prefix}weight", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_scale(self) -> None:
        with torch.no_grad():
            initial_softplus = self.initial_scale - self.minimum_scale
            self.raw_scale.fill_(math.log(math.expm1(initial_softplus)))

    def scale(self) -> Tensor:
        return functional.softplus(self.raw_scale.float()) \
            + self.minimum_scale

    def inverse_denominator(
        self,
        rms_over_scale_squared: Tensor,
    ) -> Tensor:
        """Return the reciprocal of the configured positive denominator."""
        squared = rms_over_scale_squared.float()
        if self.denominator_family == "sqrt":
            return torch.rsqrt(1.0 + squared)
        small_threshold = 1e-4
        squared_small = squared.clamp_max(small_threshold)
        small_gain = (
            1.0
            - squared_small / 3.0
            + 2.0 * squared_small.square() / 15.0
        )
        safe_u = squared.clamp_min(small_threshold).sqrt()
        regular_gain = torch.tanh(safe_u) / safe_u
        return torch.where(
            squared < small_threshold,
            small_gain,
            regular_gain,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 2 \
                or hidden.shape[-1] != self.normalized_shape[0]:
            raise ValueError(
                "learnable centering soft norm expects matching "
                "[example, neuron] activations"
            )
        if self.weight is None:
            centered = hidden - hidden.mean(dim=-1, keepdim=True)
        else:
            centered = functional.linear(hidden, self.weight)
        scale = self.scale()
        rms_over_scale_squared = (
            centered.float().square().mean(dim=-1, keepdim=True)
            / scale.square()
        )
        gain = (
            self.inverse_denominator(rms_over_scale_squared)
            / scale
        )
        return centered * gain.to(dtype=centered.dtype)


def centering_matrix_constraint_components(
    matrices: Iterable[Tensor],
) -> tuple[Tensor, Tensor]:
    """Mean squared idempotence and symmetry residuals across matrices."""
    matrix_values = tuple(matrices)
    if not matrix_values:
        raise ValueError("centering constraints require at least one matrix")
    idempotence: list[Tensor] = []
    symmetry: list[Tensor] = []
    for matrix in matrix_values:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("centering matrices must be square")
        value = matrix.float()
        idempotence.append((value @ value - value).square().mean())
        symmetry.append((value.transpose(0, 1) - value).square().mean())
    return torch.stack(idempotence).mean(), torch.stack(symmetry).mean()


def fused_glu_branches(
    projected: Tensor,
    value_centering_normalizer: LearnableCenteringNorm | None = None,
    gate_centering_normalizer: LearnableCenteringNorm | None = None,
    value_norm_bias: Tensor | None = None,
    gate_norm_bias: Tensor | None = None,
    value_transform: nn.Module | None = None,
    gate_transform: nn.Module | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return raw and normalized/transformed value and gate branches."""
    if projected.ndim != 2 or projected.shape[-1] % 2 != 0:
        raise ValueError(
            "fused GLU expects [example, 2 * output_neuron] projections"
    )
    raw_value, raw_gate_logits = projected.chunk(2, dim=-1)
    if value_centering_normalizer is None \
            and gate_centering_normalizer is None:
        if value_norm_bias is not None or gate_norm_bias is not None:
            raise ValueError(
                "branch normalization biases require a centering normalizer"
            )
        value = raw_value
        gate_logits = raw_gate_logits
    else:
        if value_centering_normalizer is None \
                or gate_centering_normalizer is None \
                or value_norm_bias is None or gate_norm_bias is None:
            raise ValueError(
                "centering normalization requires a normalizer and bias "
                "for each branch"
            )
        value = value_centering_normalizer(raw_value)
        gate_logits = gate_centering_normalizer(raw_gate_logits)
    value = (
        value_transform(value)
        if value_transform is not None
        else value
    )
    gate_logits = (
        gate_transform(gate_logits)
        if gate_transform is not None
        else gate_logits
    )
    if value_norm_bias is not None:
        value = value + value_norm_bias.to(dtype=value.dtype)
    if gate_norm_bias is not None:
        gate_logits = gate_logits + gate_norm_bias.to(
            dtype=gate_logits.dtype
        )
    return raw_value, raw_gate_logits, value, gate_logits


def fused_glu(
    projected: Tensor,
    value_centering_normalizer: LearnableCenteringNorm | None = None,
    gate_centering_normalizer: LearnableCenteringNorm | None = None,
    value_norm_bias: Tensor | None = None,
    gate_norm_bias: Tensor | None = None,
    value_transform: nn.Module | None = None,
    gate_transform: nn.Module | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize and transform branches while returning their raw projections."""
    raw_value, raw_gate_logits, value, gate_logits = fused_glu_branches(
        projected,
        value_centering_normalizer,
        gate_centering_normalizer,
        value_norm_bias,
        gate_norm_bias,
        value_transform,
        gate_transform,
    )
    return value * torch.sigmoid(gate_logits), raw_value, raw_gate_logits


def soft_weight_bound_penalty(
    weights: Iterable[Tensor],
    *,
    desired_magnitude: float = 1.0,
    sharpness: float = 10.0,
    absolute_epsilon: float = 1e-8,
) -> Tensor:
    """Mean squared softplus excess beyond a smooth absolute-weight bound."""
    if desired_magnitude <= 0:
        raise ValueError("desired weight magnitude must be positive")
    if sharpness <= 0:
        raise ValueError("weight-bound sharpness must be positive")
    if absolute_epsilon <= 0:
        raise ValueError("weight-bound absolute epsilon must be positive")
    weight_tensors = tuple(weights)
    if not weight_tensors:
        raise ValueError("soft weight bound requires at least one weight tensor")
    total_penalty = torch.zeros(
        (),
        device=weight_tensors[0].device,
        dtype=torch.float32,
    )
    parameter_count = 0
    for weight in weight_tensors:
        smooth_magnitude = (
            weight.float().square() + absolute_epsilon
        ).sqrt()
        excess = smooth_magnitude - desired_magnitude
        smooth_excess = functional.softplus(
            sharpness * excess
        ) / sharpness
        total_penalty = total_penalty + smooth_excess.square().sum()
        parameter_count += weight.numel()
    return total_penalty / parameter_count


class ReturnOracleMlp(nn.Module):
    """Shrinking GLU MLP from normalized returns to oracle logits."""

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        dropout: float = 0.05,
        dropout_rate: float = 1.0,
        normalization_family: str = "sqrt",
        normalization_initial_scale: float = (
            BRANCH_NORMALIZATION_INITIAL_RADIUS
        ),
        normalization_minimum_scale: float = 1e-4,
        learnable_centering: bool = True,
    ) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if feature_mean.shape != (INPUT_RETURN_COUNT,) \
                or feature_std.shape != (INPUT_RETURN_COUNT,):
            raise ValueError(
                "feature normalization must match the 60-return input contract"
            )
        if not bool(torch.isfinite(feature_mean).all()) \
                or not bool(torch.isfinite(feature_std).all()) \
                or not bool((feature_std > 0).all()):
            raise ValueError("feature normalization must be finite and positive")
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer(
            "feature_std",
            feature_std.float().clamp_min(1e-6).clone(),
        )
        self.dropout_rate = float(dropout_rate)
        self.learnable_centering = bool(learnable_centering)
        self.dropout_gate_probability = dropout_gate_probability(
            self.dropout_rate
        )
        widths = (INPUT_RETURN_COUNT, *HIDDEN_WIDTHS)
        self.layers = nn.ModuleList([
            nn.Linear(input_width, output_width * 2)
            for input_width, output_width in zip(
                widths[:-1],
                widths[1:],
                strict=True,
            )
        ])
        self.value_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                output_width,
                denominator_family=normalization_family,
                initial_scale=normalization_initial_scale,
                minimum_scale=normalization_minimum_scale,
                learnable_centering=self.learnable_centering,
            )
            for output_width in HIDDEN_WIDTHS
        ])
        self.gate_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                output_width,
                denominator_family=normalization_family,
                initial_scale=normalization_initial_scale,
                minimum_scale=normalization_minimum_scale,
                learnable_centering=self.learnable_centering,
            )
            for output_width in HIDDEN_WIDTHS
        ])
        if self.learnable_centering:
            # Value and gate branches share one centering matrix per layer.
            for value_normalizer, gate_normalizer in zip(
                self.value_centering_normalizers,
                self.gate_centering_normalizers,
                strict=True,
            ):
                gate_normalizer.weight = value_normalizer.weight
        self.value_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(output_width))
            for output_width in HIDDEN_WIDTHS
        ])
        self.gate_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(output_width))
            for output_width in HIDDEN_WIDTHS
        ])
        self.value_transforms = nn.ModuleList([
            nn.Linear(output_width, output_width, bias=False)
            for output_width in HIDDEN_WIDTHS
        ])
        self.gate_transforms = nn.ModuleList([
            nn.Linear(output_width, output_width, bias=False)
            for output_width in HIDDEN_WIDTHS
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(HIDDEN_WIDTHS[-1], OUTPUT_ACTION_COUNT)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_normal_(
                layer.weight,
                nonlinearity="linear",
            )
            nn.init.zeros_(layer.bias)
        for normalizer in self.value_centering_normalizers:
            normalizer.reset_parameters()
        for normalizer in self.gate_centering_normalizers:
            normalizer.reset_scale()
        for bias in (
            *self.value_norm_biases,
            *self.gate_norm_biases,
        ):
            nn.init.zeros_(bias)
        for transform in (
            *self.value_transforms,
            *self.gate_transforms,
        ):
            nn.init.eye_(transform.weight)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward_with_regularizers(
        self,
        features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        model_input = (
            features.float() - self.feature_mean
        ) / self.feature_std
        hidden = model_input
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
        distribution_sum_penalties: list[Tensor] = []
        distribution_negative_penalties: list[Tensor] = []
        for (
            layer,
            value_centering_normalizer,
            gate_centering_normalizer,
            value_norm_bias,
            gate_norm_bias,
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
            hidden, value, gate_logits = fused_glu(
                layer(hidden),
                value_centering_normalizer,
                gate_centering_normalizer,
                value_norm_bias,
                gate_norm_bias,
                value_transform,
                gate_transform,
            )
            branch_components = [
                soft_layer_norm_components(branch)
                for branch in (value, gate_logits)
            ]
            mean_penalties.append(
                torch.stack([
                    mean for mean, _variance in branch_components
                ]).mean(dim=0)
            )
            variance_penalties.append(
                torch.stack([
                    variance for _mean, variance in branch_components
                ]).mean(dim=0)
            )
            sum_penalty, negative_penalty = distribution_layer_components(
                hidden
            )
            distribution_sum_penalties.append(sum_penalty)
            distribution_negative_penalties.append(negative_penalty)
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
            torch.stack(distribution_sum_penalties).mean(dim=0),
            torch.stack(distribution_negative_penalties).mean(dim=0),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_with_regularizers(features)[0]


class ReturnOracleExportMlp(nn.Module):
    """Logits-only model with input normalization folded into layer one."""

    def __init__(self, folded_model: ReturnOracleMlp) -> None:
        super().__init__()
        self.layers = folded_model.layers
        self.value_centering_normalizers = (
            folded_model.value_centering_normalizers
        )
        self.gate_centering_normalizers = (
            folded_model.gate_centering_normalizers
        )
        self.value_norm_biases = folded_model.value_norm_biases
        self.gate_norm_biases = folded_model.gate_norm_biases
        self.value_transforms = folded_model.value_transforms
        self.gate_transforms = folded_model.gate_transforms
        self.output = folded_model.output

    def forward(self, raw_features: Tensor) -> Tensor:
        hidden = raw_features.float()
        for (
            layer,
            value_centering_normalizer,
            gate_centering_normalizer,
            value_norm_bias,
            gate_norm_bias,
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
            hidden, _raw_value, _raw_gate = fused_glu(
                layer(hidden),
                value_centering_normalizer,
                gate_centering_normalizer,
                value_norm_bias,
                gate_norm_bias,
                value_transform,
                gate_transform,
            )
        return self.output(hidden)


@torch.no_grad()
def fold_input_normalization_for_export(
    model: ReturnOracleMlp,
) -> ReturnOracleExportMlp:
    """Copy a trained model and fold its frozen input affine into layer one."""
    folded_model = deepcopy(model).eval()
    first_layer = folded_model.layers[0]
    layer_weight = first_layer.weight
    mean = folded_model.feature_mean.to(
        device=layer_weight.device,
        dtype=layer_weight.dtype,
    )
    std = folded_model.feature_std.to(
        device=layer_weight.device,
        dtype=layer_weight.dtype,
    )
    export_weight = layer_weight / std.unsqueeze(0)
    export_bias = first_layer.bias - export_weight @ mean
    first_layer.weight.copy_(export_weight)
    first_layer.bias.copy_(export_bias)
    return ReturnOracleExportMlp(folded_model).eval()


def oracle_policy_cross_entropy(
    predicted_logits: Tensor,
    target_probabilities: Tensor,
    example_weights: Tensor | None = None,
    *,
    reverse_kl_prediction_mixture: float = 1e-2,
) -> dict[str, Tensor]:
    """Soft-target CE plus diagnostic forward and skew reverse KL."""
    if predicted_logits.ndim != 2 \
            or target_probabilities.shape != predicted_logits.shape:
        raise ValueError(
            "predicted logits and raw oracle probabilities must have matching "
            "[example, action] shapes"
        )
    if predicted_logits.shape[1] != OUTPUT_ACTION_COUNT:
        raise ValueError(
            f"the stored action grid must contain {OUTPUT_ACTION_COUNT} cells"
        )
    if example_weights is not None \
            and example_weights.shape != (predicted_logits.shape[0],):
        raise ValueError("example weights must contain one value per example")
    if not 0 < reverse_kl_prediction_mixture < 1:
        raise ValueError(
            "reverse-KL prediction-mixture weight must be in (0, 1)"
        )

    predicted_log = torch.log_softmax(predicted_logits.float(), dim=-1)
    predicted = predicted_log.exp()
    target = target_probabilities.float().clamp_min(0)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    target_log = torch.where(
        target > 0,
        target.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(target),
    )
    cross_entropy = -(target * predicted_log).sum(dim=-1)
    target_entropy = -(target * target_log).sum(dim=-1)
    base_kl = cross_entropy - target_entropy
    reverse_kl_reference = (
        (1 - reverse_kl_prediction_mixture) * target
        + reverse_kl_prediction_mixture * predicted
    )
    reverse_kl_reference_log = reverse_kl_reference.clamp_min(
        torch.finfo(torch.float32).tiny
    ).log()
    reverse_kl = (
        predicted * (predicted_log - reverse_kl_reference_log)
    ).sum(dim=-1)
    probability_mse = (target - predicted).square().mean(dim=-1)
    predicted_entropy = -(predicted * predicted_log).sum(dim=-1)
    entropy_gap = predicted_entropy - target_entropy
    entropy_sharpness = torch.relu(entropy_gap).square()
    if example_weights is None:
        weights = torch.ones_like(cross_entropy)
    else:
        weights = example_weights.float().clamp_min(0)
    weight_sum = weights.sum().clamp_min(torch.finfo(torch.float32).tiny)

    def weighted_mean(values: Tensor) -> Tensor:
        return (values * weights).sum() / weight_sum

    mean_cross_entropy = weighted_mean(cross_entropy)
    return {
        "loss": mean_cross_entropy,
        "crossEntropy": mean_cross_entropy,
        "baseKlDivergence": weighted_mean(base_kl),
        "reverseKlDivergence": weighted_mean(reverse_kl),
        "probabilityMse": weighted_mean(probability_mse),
        "targetEntropy": weighted_mean(target_entropy),
        "predictedEntropy": weighted_mean(predicted_entropy),
        "entropyGap": weighted_mean(entropy_gap),
        "entropySharpness": weighted_mean(entropy_sharpness),
    }


def oracle_policy_objective(
    predicted_logits: Tensor,
    target_probabilities: Tensor,
    mean_penalties: Tensor,
    variance_penalties: Tensor,
    distribution_sum_penalties: Tensor,
    distribution_negative_penalties: Tensor,
    soft_weight_bound_penalty: Tensor,
    centering_idempotence_penalty: Tensor,
    centering_symmetry_penalty: Tensor,
    example_weights: Tensor | None = None,
    *,
    cross_entropy_weight: float = 1.0,
    reverse_kl_weight: float = 0.0,
    reverse_kl_prediction_mixture: float = 1e-2,
    entropy_sharpness_weight: float = 0.0,
    reverse_kl_gate: Tensor | None = None,
    entropy_sharpness_gate: Tensor | None = None,
    reverse_kl_scale: float = 1.0,
    entropy_sharpness_scale: float = 1.0,
    soft_layer_norm_weight: float = 0.01,
    soft_weight_bound_weight: float = 0.01,
    variance_weight: float = 1.0,
    distribution_sum_weight: float = 0.01,
    distribution_negative_weight: float = 0.01,
    centering_idempotence_weight: float = 1.0,
    centering_symmetry_weight: float = 1.0,
) -> dict[str, Tensor]:
    """CE plus activation, weight, and centering-matrix regularizers."""
    if min(
        cross_entropy_weight,
        reverse_kl_weight,
        entropy_sharpness_weight,
        reverse_kl_scale,
        entropy_sharpness_scale,
        soft_layer_norm_weight,
        soft_weight_bound_weight,
        variance_weight,
        distribution_sum_weight,
        distribution_negative_weight,
        centering_idempotence_weight,
        centering_symmetry_weight,
    ) < 0:
        raise ValueError("objective weights must be non-negative")
    if not 0 < reverse_kl_prediction_mixture < 1:
        raise ValueError(
            "reverse-KL prediction-mixture weight must be in (0, 1)"
        )
    if reverse_kl_gate is not None and reverse_kl_gate.ndim != 0:
        raise ValueError("reverse-KL gate must be scalar")
    if entropy_sharpness_gate is not None \
            and entropy_sharpness_gate.ndim != 0:
        raise ValueError("entropy-sharpness gate must be scalar")
    if mean_penalties.shape != (predicted_logits.shape[0],) \
            or variance_penalties.shape != mean_penalties.shape:
        raise ValueError(
            "soft LayerNorm penalties must contain one value per example"
        )
    if distribution_sum_penalties.shape != mean_penalties.shape \
            or distribution_negative_penalties.shape != mean_penalties.shape:
        raise ValueError(
            "distribution penalties must contain one value per example"
        )
    if soft_weight_bound_penalty.ndim != 0:
        raise ValueError("soft weight-bound penalty must be scalar")
    if centering_idempotence_penalty.ndim != 0 \
            or centering_symmetry_penalty.ndim != 0:
        raise ValueError("centering-matrix penalties must be scalar")
    metrics = oracle_policy_cross_entropy(
        predicted_logits,
        target_probabilities,
        example_weights,
        reverse_kl_prediction_mixture=reverse_kl_prediction_mixture,
    )
    weights = (
        torch.ones_like(mean_penalties)
        if example_weights is None
        else example_weights.float().clamp_min(0)
    )
    weight_sum = weights.sum().clamp_min(torch.finfo(torch.float32).tiny)
    mean_penalty = (mean_penalties.float() * weights).sum() / weight_sum
    variance_penalty = (
        variance_penalties.float() * weights
    ).sum() / weight_sum
    distribution_sum_penalty = (
        distribution_sum_penalties.float() * weights
    ).sum() / weight_sum
    distribution_negative_penalty = (
        distribution_negative_penalties.float() * weights
    ).sum() / weight_sum
    soft_layer_norm = mean_penalty + variance_weight * variance_penalty
    distribution_layer = (
        distribution_sum_weight * distribution_sum_penalty
        + distribution_negative_weight * distribution_negative_penalty
    )
    centering_constraint = (
        centering_idempotence_weight
        * centering_idempotence_penalty.float()
        + centering_symmetry_weight
        * centering_symmetry_penalty.float()
    )
    metrics["softLayerNormMeanPenalty"] = mean_penalty
    metrics["softLayerNormVariancePenalty"] = variance_penalty
    metrics["softLayerNorm"] = soft_layer_norm
    metrics["distributionLayerSumPenalty"] = distribution_sum_penalty
    metrics["distributionLayerNegativePenalty"] = (
        distribution_negative_penalty
    )
    metrics["distributionLayer"] = distribution_layer
    metrics["softWeightBound"] = soft_weight_bound_penalty.float()
    metrics["centeringIdempotence"] = (
        centering_idempotence_penalty.float()
    )
    metrics["centeringSymmetry"] = centering_symmetry_penalty.float()
    metrics["centeringConstraint"] = centering_constraint
    reverse_gate = (
        predicted_logits.new_ones((), dtype=torch.float32)
        if reverse_kl_gate is None
        else reverse_kl_gate.float()
    )
    entropy_gate = (
        predicted_logits.new_ones((), dtype=torch.float32)
        if entropy_sharpness_gate is None
        else entropy_sharpness_gate.float()
    )
    metrics["reverseKlGate"] = reverse_gate
    metrics["entropySharpnessGate"] = entropy_gate
    metrics["loss"] = (
        cross_entropy_weight * metrics["crossEntropy"]
        + reverse_gate
        * reverse_kl_scale
        * reverse_kl_weight
        * metrics["reverseKlDivergence"]
        + entropy_gate
        * entropy_sharpness_scale
        * entropy_sharpness_weight
        * metrics["entropySharpness"]
        + soft_layer_norm_weight * soft_layer_norm
        + distribution_layer
        + soft_weight_bound_weight * metrics["softWeightBound"]
        + centering_constraint
    )
    return metrics


def simple_return_features(close_boundaries: Tensor) -> Tensor:
    """Convert 61 positive close boundaries into 60 adjacent simple returns."""
    if close_boundaries.ndim != 2 \
            or close_boundaries.shape[1] != INPUT_RETURN_COUNT + 1:
        raise ValueError(
            f"close boundaries must have [example, {INPUT_RETURN_COUNT + 1}] shape"
        )
    if not bool((close_boundaries > 0).all()):
        raise ValueError("close boundaries must be positive")
    return close_boundaries[:, 1:] / close_boundaries[:, :-1] - 1.0


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
