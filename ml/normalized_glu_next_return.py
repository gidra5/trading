from __future__ import annotations

import torch
from torch import Tensor, nn

from next_return_dataset import DAY_SECONDS, HISTORY_RETURN_COUNT
from return_oracle_ce import (
    BRANCH_NORMALIZATION_INITIAL_RADIUS,
    LearnableCenteringNorm,
    dropout_gate_probability,
    fused_glu,
)


DEFAULT_WIDTHS = (512,)
TRAINING_POSITION_INPUT_NORMALIZATION = "training-position"
PER_SEQUENCE_INPUT_NORMALIZATION = "per-sequence"
PER_SEQUENCE_REVERSIBLE_INPUT_NORMALIZATION = "per-sequence-reversible"
PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION = (
    "per-sequence-reversible-with-stats"
)
CAUSAL_VOLATILITY_INPUT_NORMALIZATION = "causal-volatility"
INPUT_NORMALIZATION_MODES = frozenset({
    TRAINING_POSITION_INPUT_NORMALIZATION,
    PER_SEQUENCE_INPUT_NORMALIZATION,
    PER_SEQUENCE_REVERSIBLE_INPUT_NORMALIZATION,
    PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION,
    CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
})
REVERSIBLE_SEQUENCE_INPUT_NORMALIZATION_MODES = frozenset({
    PER_SEQUENCE_REVERSIBLE_INPUT_NORMALIZATION,
    PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION,
    CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
})
MINIMUM_SEQUENCE_STD = 1e-8
ARCHITECTURE_CONTRACT = (
    "next-return-fused-glu-stack-independent-branch-centering-sqrt-"
    "learned-radius-full-a-post-bias-v2"
)


class NormalizedGluNextReturn(nn.Module):
    """A learned-radius normalized GLU stack adapted to T-return regression."""

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
        *,
        widths: tuple[int, ...] = DEFAULT_WIDTHS,
        input_normalization: str = TRAINING_POSITION_INPUT_NORMALIZATION,
        volatility_window: int | None = None,
        learnable_centering: bool = True,
        dropout: float = 0.05,
        dropout_rate: float = 0.5,
        initial_radius: float = BRANCH_NORMALIZATION_INITIAL_RADIUS,
        minimum_radius: float = 1e-4,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (HISTORY_RETURN_COUNT,) \
                or feature_std.shape != feature_mean.shape \
                or not bool(torch.isfinite(feature_mean).all()) \
                or not bool(torch.isfinite(feature_std).all()) \
                or bool((feature_std <= 0).any()):
            raise ValueError("feature normalization must contain 120 finite values")
        if target_mean.ndim > 1 or target_std.shape != target_mean.shape \
                or target_mean.numel() < 1 \
                or not bool(torch.isfinite(target_mean).all()) \
                or not bool(torch.isfinite(target_std).all()) \
                or bool((target_std <= 0).any()):
            raise ValueError("target normalization must contain T finite scales")
        if not widths or any(width < 2 for width in widths):
            raise ValueError(
                "next-return predictor requires one or more GLU layers "
                "with width at least two"
            )
        if input_normalization not in INPUT_NORMALIZATION_MODES:
            raise ValueError(
                f"unsupported input normalization: {input_normalization}"
            )
        if input_normalization == CAUSAL_VOLATILITY_INPUT_NORMALIZATION:
            if volatility_window is None \
                    or not isinstance(volatility_window, int) \
                    or isinstance(volatility_window, bool) \
                    or not 1 <= volatility_window <= DAY_SECONDS:
                raise ValueError(
                    "causal volatility window must be in [1, 86,400]"
                )
        elif volatility_window is not None:
            raise ValueError(
                "volatility window requires causal-volatility normalization"
            )
        if not 0 <= dropout < 1 or not 0 <= dropout_rate <= 1:
            raise ValueError("dropout settings are invalid")
        self.widths = tuple(int(width) for width in widths)
        self.input_normalization = input_normalization
        self.volatility_window = volatility_window
        self.learnable_centering = bool(learnable_centering)
        self.horizon_return_count = int(target_mean.numel())
        self.dropout_rate = float(dropout_rate)
        self.dropout_gate_probability = dropout_gate_probability(dropout_rate)
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())
        target_shape = () if self.horizon_return_count == 1 else (-1,)
        self.register_buffer(
            "target_mean", target_mean.float().reshape(target_shape).clone()
        )
        self.register_buffer(
            "target_std", target_std.float().reshape(target_shape).clone()
        )

        input_width = HISTORY_RETURN_COUNT + (
            2 if input_normalization
            == PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION
            else 0
        )
        all_widths = (input_width, *self.widths)
        self.layers = nn.ModuleList([
            nn.Linear(input_width, 2 * output_width)
            for input_width, output_width in zip(
                all_widths[:-1], all_widths[1:], strict=True
            )
        ])
        self.value_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                width,
                denominator_family="sqrt",
                initial_scale=initial_radius,
                minimum_scale=minimum_radius,
            )
            for width in self.widths
        ])
        self.gate_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                width,
                denominator_family="sqrt",
                initial_scale=initial_radius,
                minimum_scale=minimum_radius,
            )
            for width in self.widths
        ])
        if not self.learnable_centering:
            for normalizer in (
                *self.value_centering_normalizers,
                *self.gate_centering_normalizers,
            ):
                normalizer.weight.requires_grad_(False)
        self.value_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for width in self.widths
        ])
        self.gate_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for width in self.widths
        ])
        self.value_transforms = nn.ModuleList([
            nn.Linear(width, width, bias=False) for width in self.widths
        ])
        self.gate_transforms = nn.ModuleList([
            nn.Linear(width, width, bias=False) for width in self.widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(
            self.widths[-1], self.horizon_return_count
        )
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
        # Begin at the training-mean baseline. The sequence head learns on the
        # first update and then propagates task gradients into the GLU stack.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            *(layer.weight for layer in self.layers),
            *(layer.weight for layer in self.value_transforms),
            *(layer.weight for layer in self.gate_transforms),
        )

    def _normalize_features_with_sequence_stats(
        self,
        features: Tensor,
        causal_volatility_rms: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if features.ndim != 2 or features.shape[-1] != HISTORY_RETURN_COUNT:
            raise ValueError("normalized GLU expects [example, 120] returns")
        values = features.float()
        if self.input_normalization == CAUSAL_VOLATILITY_INPUT_NORMALIZATION:
            if self.volatility_window is None:
                raise RuntimeError("causal volatility window is missing")
            if causal_volatility_rms is None:
                if self.volatility_window > HISTORY_RETURN_COUNT:
                    raise ValueError(
                        "causal volatility windows above 120 require an "
                        "external input-only RMS scale"
                    )
                raw_volatility = values[
                    :, -self.volatility_window:
                ].square().mean(dim=-1, keepdim=True).sqrt()
            else:
                raw_volatility = causal_volatility_rms.float()
                if raw_volatility.ndim == 1:
                    raw_volatility = raw_volatility.unsqueeze(-1)
                if raw_volatility.shape != (values.shape[0], 1) \
                        or not bool(torch.isfinite(raw_volatility).all()) \
                        or bool((raw_volatility < 0).any()):
                    raise ValueError(
                        "external causal volatility must contain one finite "
                        "non-negative RMS scale per example"
                    )
            volatility_floor = self.target_std.float().mean().detach() * 0.1
            volatility = (
                raw_volatility.square() + volatility_floor.square()
            ).sqrt().clamp_min(MINIMUM_SEQUENCE_STD)
            return values / volatility, torch.zeros_like(volatility), volatility
        if self.input_normalization in {
            PER_SEQUENCE_INPUT_NORMALIZATION,
            *REVERSIBLE_SEQUENCE_INPUT_NORMALIZATION_MODES,
        }:
            sequence_mean = values.mean(dim=-1, keepdim=True)
            centered = values - sequence_mean
            sequence_std = centered.square().mean(
                dim=-1, keepdim=True
            ).sqrt().clamp_min(MINIMUM_SEQUENCE_STD)
            normalized = centered / sequence_std
            if self.input_normalization \
                    == PER_SEQUENCE_REVERSIBLE_WITH_STATS_INPUT_NORMALIZATION:
                training_mean = self.feature_mean.mean()
                training_std = self.feature_std.mean()
                side_features = torch.cat((
                    (sequence_mean - training_mean) / training_std,
                    sequence_std / training_std - 1.0,
                ), dim=-1)
                normalized = torch.cat((normalized, side_features), dim=-1)
            return normalized, sequence_mean, sequence_std
        return (
            (values - self.feature_mean) / self.feature_std,
            None,
            None,
        )

    def normalize_features(
        self, features: Tensor, causal_volatility_rms: Tensor | None = None
    ) -> Tensor:
        return self._normalize_features_with_sequence_stats(
            features, causal_volatility_rms
        )[0]

    def causal_volatility(
        self, features: Tensor, causal_volatility_rms: Tensor | None = None
    ) -> Tensor:
        if self.input_normalization != CAUSAL_VOLATILITY_INPUT_NORMALIZATION:
            raise RuntimeError("model does not use causal volatility normalization")
        _normalized, _mean, volatility = (
            self._normalize_features_with_sequence_stats(
                features, causal_volatility_rms
            )
        )
        if volatility is None:
            raise RuntimeError("causal volatility scale is missing")
        return volatility.squeeze(-1)

    def _forward_standardized_with_sequence_stats(
        self,
        features: Tensor,
        causal_volatility_rms: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        hidden, sequence_mean, sequence_std = (
            self._normalize_features_with_sequence_stats(
                features, causal_volatility_rms
            )
        )
        intermittent_dropout = (
            self.training
            and self.dropout.p > 0
            and 0 < self.dropout_rate < 1
        )
        pass_gate = (
            torch.rand((), device=hidden.device) < self.dropout_gate_probability
            if intermittent_dropout
            else None
        )
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
            hidden, _raw_value, _raw_gate = fused_glu(
                layer(hidden),
                value_normalizer,
                gate_normalizer,
                value_bias,
                gate_bias,
                value_transform,
                gate_transform,
            )
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
        return self.output(hidden), sequence_mean, sequence_std

    def forward_standardized(
        self, features: Tensor, causal_volatility_rms: Tensor | None = None
    ) -> Tensor:
        standardized, _sequence_mean, _sequence_std = (
            self._forward_standardized_with_sequence_stats(
                features, causal_volatility_rms
            )
        )
        return standardized.squeeze(-1) \
            if self.horizon_return_count == 1 else standardized

    def forward(
        self, features: Tensor, causal_volatility_rms: Tensor | None = None
    ) -> Tensor:
        standardized, sequence_mean, sequence_std = (
            self._forward_standardized_with_sequence_stats(
                features, causal_volatility_rms
            )
        )
        if self.input_normalization \
                in REVERSIBLE_SEQUENCE_INPUT_NORMALIZATION_MODES:
            if sequence_mean is None or sequence_std is None:
                raise RuntimeError("reversible sequence statistics are missing")
            prediction = standardized * sequence_std + sequence_mean
        else:
            prediction = standardized * self.target_std + self.target_mean
        return prediction.squeeze(-1) \
            if self.horizon_return_count == 1 else prediction


def optimizer_parameter_groups(
    model: NormalizedGluNextReturn,
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
        raise RuntimeError("invalid normalized GLU optimizer routing")
    if {id(value) for value in (*muon, *adamw)} != {
        id(value) for value in trainable
    } or any(parameter.ndim != 2 for parameter in muon):
        raise RuntimeError("normalized GLU optimizer routing is incomplete")
    return muon, adamw


def depth_width_parameter_assignments(
    model: NormalizedGluNextReturn,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Assign every trainable scalar to one of four depth-width quadrants."""
    if len(model.widths) % 2 != 0 \
            or len(set(model.widths)) != 1 \
            or model.widths[0] % 2 != 0:
        raise ValueError(
            "depth-width partitioning requires an even number of equal, "
            "even-width layers"
        )
    depth_boundary = len(model.widths) // 2
    width = model.widths[0]
    width_boundary = width // 2
    result: list[tuple[Tensor, Tensor]] = []
    covered: set[int] = set()

    def add(parameter: Tensor, assignment: Tensor) -> None:
        if not parameter.requires_grad:
            return
        if assignment.shape != parameter.shape \
                or assignment.dtype != torch.uint8 \
                or int(assignment.min()) < 0 \
                or int(assignment.max()) > 3 \
                or id(parameter) in covered:
            raise RuntimeError("invalid depth-width parameter assignment")
        covered.add(id(parameter))
        result.append((parameter, assignment))

    channel_half = (
        torch.arange(width, device=model.output.weight.device) >= width_boundary
    ).to(torch.uint8)
    branch_half = torch.cat((channel_half, channel_half))
    for index, (
        layer,
        value_normalizer,
        gate_normalizer,
        value_bias,
        gate_bias,
        value_transform,
        gate_transform,
    ) in enumerate(zip(
        model.layers,
        model.value_centering_normalizers,
        model.gate_centering_normalizers,
        model.value_norm_biases,
        model.gate_norm_biases,
        model.value_transforms,
        model.gate_transforms,
        strict=True,
    )):
        depth_offset = 2 if index >= depth_boundary else 0
        branch_assignment = branch_half + depth_offset
        channel_assignment = channel_half + depth_offset
        add(
            layer.weight,
            branch_assignment[:, None].expand_as(layer.weight).clone(),
        )
        add(layer.bias, branch_assignment.clone())
        for normalizer, scalar_width_half in (
            (value_normalizer, 0),
            (gate_normalizer, 1),
        ):
            add(
                normalizer.weight,
                channel_assignment[:, None]
                .expand_as(normalizer.weight).clone(),
            )
            add(
                normalizer.raw_scale,
                torch.full_like(
                    normalizer.raw_scale,
                    depth_offset + scalar_width_half,
                    dtype=torch.uint8,
                ),
            )
        for bias in (value_bias, gate_bias):
            add(bias, channel_assignment.clone())
        for transform in (value_transform, gate_transform):
            add(
                transform.weight,
                channel_assignment[:, None]
                .expand_as(transform.weight).clone(),
            )

    output_depth_offset = 2
    add(
        model.output.weight,
        (channel_half + output_depth_offset)[None, :]
        .expand_as(model.output.weight).clone(),
    )
    add(
        model.output.bias,
        torch.full_like(
            model.output.bias, output_depth_offset, dtype=torch.uint8
        ),
    )
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if covered != trainable_ids:
        raise RuntimeError("depth-width partitioning did not cover the model")
    return tuple(result)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
