from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from exact_tensor_return_density import TensorPathGluBlock
from return_oracle_ce import LearnableCenteringNorm, fused_glu


ARCHITECTURE_CONTRACT = "structured-shared-io-feature-process-v1"


class RankFactorizedLinear(nn.Module):
    """Train-from-scratch low-rank map, optionally around an identity base."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        requested_rank: int,
        *,
        bias: bool,
        identity_base: bool = False,
        right_initial_std: float = 1.0,
    ) -> None:
        super().__init__()
        if input_width <= 0 or output_width <= 0 or requested_rank <= 0:
            raise ValueError("factorized linear dimensions must be positive")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.requested_rank = int(requested_rank)
        # Asking for rank 256 on a 59-wide IO map has an effective rank ceiling
        # of 59. Capping it avoids redundant factors without reducing capacity.
        self.rank = min(
            self.requested_rank, self.input_width, self.output_width
        )
        self.identity_base = bool(identity_base)
        self.left = nn.Parameter(torch.empty(self.rank, self.input_width))
        self.right = nn.Parameter(torch.empty(self.output_width, self.rank))
        self.bias = nn.Parameter(torch.zeros(self.output_width)) if bias else None
        nn.init.normal_(self.left, std=1 / math.sqrt(self.input_width))
        nn.init.normal_(self.right, std=float(right_initial_std))

    def forward(self, values: Tensor) -> Tensor:
        result = F.linear(F.linear(values, self.left), self.right) / math.sqrt(
            self.rank
        )
        if self.identity_base:
            if self.output_width == self.input_width:
                identity = values
            elif self.output_width < self.input_width:
                identity = values[..., :self.output_width]
            else:
                identity = F.pad(values, (0, self.output_width - self.input_width))
            result = result + identity
        if self.bias is not None:
            result = result + self.bias
        return result

    def muon_parameters(self) -> tuple[Tensor, Tensor]:
        return self.left, self.right


class LowRankTensorPathGluBlock(nn.Module):
    """GNGLU whose trainable matrices are rank-factorized LoRA-style maps."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        rank: int,
        *,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        output_bias: Tensor,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.projection = RankFactorizedLinear(
            input_width, 2 * hidden_width, rank, bias=True,
        )
        self.value_centering = LearnableCenteringNorm(
            hidden_width,
            denominator_family="sqrt",
            initial_scale=initial_radius,
            minimum_scale=minimum_radius,
            learnable_centering=learnable_centering,
        )
        self.gate_centering = LearnableCenteringNorm(
            hidden_width,
            denominator_family="sqrt",
            initial_scale=initial_radius,
            minimum_scale=minimum_radius,
            learnable_centering=learnable_centering,
        )
        self.value_bias = nn.Parameter(torch.zeros(hidden_width))
        self.gate_bias = nn.Parameter(torch.zeros(hidden_width))
        self.value_transform = RankFactorizedLinear(
            hidden_width, hidden_width, rank,
            bias=False, identity_base=True, right_initial_std=1e-3,
        )
        self.gate_transform = RankFactorizedLinear(
            hidden_width, hidden_width, rank,
            bias=False, identity_base=True, right_initial_std=1e-3,
        )
        self.output = RankFactorizedLinear(
            hidden_width, output_width, rank,
            bias=True, identity_base=True, right_initial_std=1e-4,
        )
        if output_bias.shape != (output_width,):
            raise ValueError("low-rank GNGLU output bias has the wrong shape")
        with torch.no_grad():
            assert self.output.bias is not None
            self.output.bias.copy_(output_bias)

    def forward(self, values: Tensor) -> Tensor:
        hidden, _raw_value, _raw_gate = fused_glu(
            self.projection(values),
            self.value_centering,
            self.gate_centering,
            self.value_bias,
            self.gate_bias,
            self.value_transform,
            self.gate_transform,
        )
        return self.output(hidden)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        maps = (
            self.projection,
            self.value_transform,
            self.gate_transform,
            self.output,
        )
        return tuple(
            parameter for mapping in maps for parameter in mapping.muon_parameters()
        )


class DualStateGatedExchangeCell(nn.Module):
    """Diagram-faithful recurrent value layer with private hidden memory.

    ``x`` and ``x_star`` feed separately normalized value and gate paths. The
    previous hidden state feeds matching hidden-value and hidden-gate paths.
    Local hidden/value candidates are gated, then exchange candidates and two
    product gates mix information between H and V. The diagram's signed mix is
    used literally: ``mix(a,b,t) = a*t + b*(t-1)``.
    """

    def __init__(
        self,
        value_input_width: int,
        gate_input_width: int,
        value_width: int,
        hidden_width: int,
        *,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
    ) -> None:
        super().__init__()
        dimensions = (
            value_input_width,
            gate_input_width,
            value_width,
            hidden_width,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("dual-state cell dimensions must be positive")
        self.value_input_width = int(value_input_width)
        self.gate_input_width = int(gate_input_width)
        self.value_width = int(value_width)
        self.hidden_width = int(hidden_width)

        # W_hv, W_hg, W_v, and W_g from the diagram.
        self.hidden_value_projection = nn.Linear(
            self.hidden_width, self.hidden_width
        )
        self.hidden_gate_projection = nn.Linear(
            self.hidden_width, self.hidden_width
        )
        self.input_value_projection = nn.Linear(
            self.value_input_width, self.value_width
        )
        self.input_gate_projection = nn.Linear(
            self.gate_input_width, self.value_width
        )
        normalizer_options = {
            "denominator_family": "sqrt",
            "initial_scale": float(initial_radius),
            "minimum_scale": float(minimum_radius),
            "learnable_centering": bool(learnable_centering),
        }
        self.hidden_value_norm = LearnableCenteringNorm(
            self.hidden_width, **normalizer_options
        )
        self.hidden_gate_norm = LearnableCenteringNorm(
            self.hidden_width, **normalizer_options
        )
        self.input_value_norm = LearnableCenteringNorm(
            self.value_width, **normalizer_options
        )
        self.input_gate_norm = LearnableCenteringNorm(
            self.value_width, **normalizer_options
        )

        # Every affine in the drawing is independently learned.
        self.hidden_value_affine = nn.Linear(
            self.hidden_width, self.hidden_width
        )
        self.hidden_gate_affine = nn.Linear(
            self.hidden_width, self.hidden_width
        )
        self.value_affine = nn.Linear(self.value_width, self.value_width)
        self.value_gate_affine = nn.Linear(
            self.value_width, self.value_width
        )
        self.hidden_keep_gate = nn.Linear(
            self.hidden_width, self.hidden_width
        )
        self.value_write_hidden_gate = nn.Linear(
            self.value_width, self.hidden_width
        )
        self.value_keep_gate = nn.Linear(
            self.value_width, self.value_width
        )
        self.hidden_write_value_gate = nn.Linear(
            self.hidden_width, self.value_width
        )
        self.hidden_to_value = nn.Linear(
            self.hidden_width, self.value_width
        )
        self.value_to_hidden = nn.Linear(
            self.value_width, self.hidden_width
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        projections = (
            self.hidden_value_projection,
            self.hidden_gate_projection,
            self.input_value_projection,
            self.input_gate_projection,
        )
        for projection in projections:
            nn.init.normal_(
                projection.weight,
                std=1 / math.sqrt(projection.in_features),
            )
            nn.init.zeros_(projection.bias)
        for affine in (
            self.hidden_value_affine,
            self.hidden_gate_affine,
            self.value_affine,
            self.value_gate_affine,
        ):
            nn.init.eye_(affine.weight)
            nn.init.zeros_(affine.bias)
        for gate in (
            self.hidden_keep_gate,
            self.value_write_hidden_gate,
            self.value_keep_gate,
            self.hidden_write_value_gate,
        ):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        for exchange in (self.hidden_to_value, self.value_to_hidden):
            nn.init.xavier_uniform_(exchange.weight)
            nn.init.zeros_(exchange.bias)

    @staticmethod
    def signed_mix(first: Tensor, second: Tensor, gate: Tensor) -> Tensor:
        return first * gate + second * (gate - 1.0)

    def forward(
        self,
        value_input: Tensor,
        gate_input: Tensor,
        hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if value_input.ndim != 2 \
                or value_input.shape[-1] != self.value_input_width:
            raise ValueError("dual-state value input has the wrong shape")
        if gate_input.ndim != 2 \
                or gate_input.shape != (
                    value_input.shape[0], self.gate_input_width
                ):
            raise ValueError("dual-state gate input has the wrong shape")
        if hidden is None:
            hidden = value_input.new_zeros(
                value_input.shape[0], self.hidden_width
            )
        elif hidden.shape != (value_input.shape[0], self.hidden_width):
            raise ValueError("dual-state hidden memory has the wrong shape")

        hidden_value = self.hidden_value_norm(
            self.hidden_value_projection(hidden)
        )
        hidden_gate = self.hidden_gate_norm(
            self.hidden_gate_projection(hidden)
        )
        input_value = self.input_value_norm(
            self.input_value_projection(value_input)
        )
        input_gate = self.input_gate_norm(
            self.input_gate_projection(gate_input)
        )

        local_hidden = self.hidden_value_affine(hidden_value) * torch.sigmoid(
            self.hidden_gate_affine(hidden_gate)
        )
        local_value = self.value_affine(input_value) * torch.sigmoid(
            self.value_gate_affine(input_gate)
        )
        hidden_mix_gate = torch.sigmoid(
            self.hidden_keep_gate(hidden_value)
        ) * torch.sigmoid(self.value_write_hidden_gate(input_value))
        value_mix_gate = torch.sigmoid(
            self.value_keep_gate(input_value)
        ) * torch.sigmoid(self.hidden_write_value_gate(hidden_value))
        next_hidden = self.signed_mix(
            local_hidden,
            self.value_to_hidden(local_value),
            hidden_mix_gate,
        )
        next_value = self.signed_mix(
            local_value,
            self.hidden_to_value(local_hidden),
            value_mix_gate,
        )
        return next_value, next_hidden

    def muon_parameters(self) -> tuple[Tensor, ...]:
        maps = (
            self.hidden_value_projection,
            self.hidden_gate_projection,
            self.input_value_projection,
            self.input_gate_projection,
            self.hidden_value_affine,
            self.hidden_gate_affine,
            self.value_affine,
            self.value_gate_affine,
            self.hidden_keep_gate,
            self.value_write_hidden_gate,
            self.value_keep_gate,
            self.hidden_write_value_gate,
            self.hidden_to_value,
            self.value_to_hidden,
        )
        return tuple(mapping.weight for mapping in maps)


class CausalSelfAttentionLayer8(nn.Module):
    """Causal single-head attention over the generated NF-state sequence."""

    def __init__(
        self,
        input_width: int,
        query_width: int,
        key_width: int,
        value_width: int,
        output_width: int,
    ) -> None:
        super().__init__()
        dimensions = (
            input_width,
            query_width,
            key_width,
            value_width,
            output_width,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("self-attention dimensions must be positive")
        if int(query_width) != int(key_width):
            raise ValueError("self-attention query and key widths must match")
        self.input_width = int(input_width)
        self.query_width = int(query_width)
        self.key_width = int(key_width)
        self.value_width = int(value_width)
        self.output_width = int(output_width)
        self.query = nn.Linear(self.input_width, self.query_width)
        self.key = nn.Linear(self.input_width, self.key_width)
        self.value = nn.Linear(self.input_width, self.value_width)
        self.output = nn.Linear(self.value_width, self.output_width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query.weight, std=1 / math.sqrt(self.input_width))
        nn.init.normal_(self.key.weight, std=1 / math.sqrt(self.input_width))
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.output.bias)
        if self.input_width == self.value_width:
            nn.init.eye_(self.value.weight)
        else:
            nn.init.xavier_uniform_(self.value.weight)
        if self.value_width == self.output_width:
            nn.init.eye_(self.output.weight)
        else:
            nn.init.xavier_uniform_(self.output.weight)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_width:
            raise ValueError(
                "layer-8 self-attention expects [batch,steps,input_width]"
            )
        query = self.query(values).unsqueeze(1)
        key = self.key(values).unsqueeze(1)
        value = self.value(values).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        ).squeeze(1)
        return self.output(attended)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            self.query.weight,
            self.key.weight,
            self.value.weight,
            self.output.weight,
        )


class ConditionalEmbeddingGaussianMixtureAttention(nn.Module):
    """Evaluate a normalized PDF at a candidate feature embedding.

    The candidate feature state is encoded by shared layer (1) and acts as the
    query.  The current NF state produces multiple key offsets and scalar value
    logits.  The causal layer-8 expectation is the common key anchor.  Values
    are normalized with a softmax and each key is the mean of an isotropic
    Gaussian with a fixed standard deviation, so the final log-sum-exp is an
    actual normalized mixture log density rather than an unbounded attention
    score.
    """

    def __init__(
        self,
        state_width: int,
        embedding_width: int,
        component_count: int,
        *,
        fixed_standard_deviation: float,
        normalization_epsilon: float,
    ) -> None:
        super().__init__()
        if min(int(state_width), int(embedding_width), int(component_count)) <= 0:
            raise ValueError("embedding-density attention dimensions must be positive")
        if min(float(fixed_standard_deviation), float(normalization_epsilon)) <= 0:
            raise ValueError("embedding-density attention scales must be positive")
        self.state_width = int(state_width)
        self.embedding_width = int(embedding_width)
        self.component_count = int(component_count)
        self.fixed_standard_deviation = float(fixed_standard_deviation)
        self.normalization_epsilon = float(normalization_epsilon)
        self.key = nn.Linear(
            self.state_width,
            self.component_count * self.embedding_width,
        )
        self.value = nn.Linear(self.state_width, self.component_count)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Small, nonidentical offsets retain the causal-attention expectation as
        # the initial center while breaking component symmetry immediately.
        nn.init.normal_(
            self.key.weight,
            std=1e-3 / math.sqrt(self.state_width),
        )
        nn.init.normal_(self.key.bias, std=1e-3)
        nn.init.normal_(
            self.value.weight,
            std=1e-3 / math.sqrt(self.state_width),
        )
        nn.init.zeros_(self.value.bias)

    def _unit_rms(self, values: Tensor) -> Tensor:
        return F.normalize(
            values, dim=-1, eps=self.normalization_epsilon,
        ) * math.sqrt(self.embedding_width)

    def distribution_parameters(
        self,
        states: Tensor,
        anchors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if states.ndim != 3 or states.shape[-1] != self.state_width:
            raise ValueError(
                "embedding-density states must have shape "
                "[batch,steps,state_width]"
            )
        if anchors.shape != (*states.shape[:2], self.embedding_width):
            raise ValueError("embedding-density anchors have the wrong shape")
        key_offsets = self.key(states).reshape(
            *states.shape[:2], self.component_count, self.embedding_width,
        )
        keys = self._unit_rms(anchors.unsqueeze(-2) + key_offsets)
        log_weights = F.log_softmax(self.value(states), dim=-1)
        return keys, log_weights

    def forward(
        self,
        queries: Tensor,
        states: Tensor,
        anchors: Tensor,
    ) -> Tensor:
        if queries.shape != (*states.shape[:2], self.embedding_width):
            raise ValueError("embedding-density queries have the wrong shape")
        queries = self._unit_rms(queries)
        keys, log_weights = self.distribution_parameters(states, anchors)
        difference = queries.unsqueeze(-2) - keys
        variance = self.fixed_standard_deviation**2
        log_normalizer = self.embedding_width * math.log(
            self.fixed_standard_deviation * math.sqrt(2.0 * math.pi)
        )
        component_log_density = (
            -0.5 * difference.square().sum(dim=-1) / variance
            - log_normalizer
        )
        return torch.logsumexp(log_weights + component_log_density, dim=-1)

    def muon_parameters(self) -> tuple[Tensor, Tensor]:
        return self.key.weight, self.value.weight


class ConditionalEmbeddingBaseSupportGaussianMixture(nn.Module):
    """A normalized embedding density whose atoms carry base-feature values.

    The causal NF state emits mixture masses and standardized base-coordinate
    offsets around the ordinary layer-(8)/(2) point prediction. The caller
    causally rederives every base-coordinate component into a complete feature
    state and encodes that state with shared layer (1). Those encoded states
    are the Gaussian means. Consequently the same component masses define both
    a proper density in embedding space and an exact expected base state.
    """

    def __init__(
        self,
        state_width: int,
        embedding_width: int,
        base_width: int,
        component_count: int,
        *,
        fixed_standard_deviation: float,
        normalization_epsilon: float,
    ) -> None:
        super().__init__()
        dimensions = (
            state_width, embedding_width, base_width, component_count,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("base-support density dimensions must be positive")
        if min(float(fixed_standard_deviation), float(normalization_epsilon)) <= 0:
            raise ValueError("base-support density scales must be positive")
        self.state_width = int(state_width)
        self.embedding_width = int(embedding_width)
        self.base_width = int(base_width)
        self.component_count = int(component_count)
        self.fixed_standard_deviation = float(fixed_standard_deviation)
        self.normalization_epsilon = float(normalization_epsilon)
        self.base_offset = nn.Linear(
            self.state_width,
            self.component_count * self.base_width,
        )
        self.mass_logit = nn.Linear(self.state_width, self.component_count)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.base_offset.weight,
            std=1e-3 / math.sqrt(self.state_width),
        )
        nn.init.normal_(self.base_offset.bias, std=1e-3)
        nn.init.normal_(
            self.mass_logit.weight,
            std=1e-3 / math.sqrt(self.state_width),
        )
        nn.init.zeros_(self.mass_logit.bias)

    def _unit_rms(self, values: Tensor) -> Tensor:
        return F.normalize(
            values, dim=-1, eps=self.normalization_epsilon,
        ) * math.sqrt(self.embedding_width)

    def distribution_parameters(
        self,
        states: Tensor,
        standardized_base_anchors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if states.ndim != 3 or states.shape[-1] != self.state_width:
            raise ValueError(
                "base-support density states must have shape "
                "[batch,steps,state_width]"
            )
        if standardized_base_anchors.shape != (
            *states.shape[:2], self.base_width,
        ):
            raise ValueError("base-support density anchors have the wrong shape")
        offsets = self.base_offset(states).reshape(
            *states.shape[:2], self.component_count, self.base_width,
        )
        components = standardized_base_anchors.unsqueeze(-2) + offsets
        log_weights = F.log_softmax(self.mass_logit(states), dim=-1)
        return components, log_weights

    def log_density(
        self,
        queries: Tensor,
        component_embeddings: Tensor,
        log_weights: Tensor,
    ) -> Tensor:
        if queries.ndim != 3 or queries.shape[-1] != self.embedding_width:
            raise ValueError("base-support density queries have the wrong shape")
        expected_component_shape = (
            *queries.shape[:2], self.component_count, self.embedding_width,
        )
        if component_embeddings.shape != expected_component_shape:
            raise ValueError("base-support component embeddings have the wrong shape")
        if log_weights.shape != (*queries.shape[:2], self.component_count):
            raise ValueError("base-support mixture masses have the wrong shape")
        queries = self._unit_rms(queries)
        keys = self._unit_rms(component_embeddings)
        difference = queries.unsqueeze(-2) - keys
        variance = self.fixed_standard_deviation**2
        log_normalizer = self.embedding_width * math.log(
            self.fixed_standard_deviation * math.sqrt(2.0 * math.pi)
        )
        component_log_density = (
            -0.5 * difference.square().sum(dim=-1) / variance
            - log_normalizer
        )
        return torch.logsumexp(log_weights + component_log_density, dim=-1)

    def expected_base(
        self, standardized_components: Tensor, log_weights: Tensor
    ) -> Tensor:
        if standardized_components.shape[:-1] != log_weights.shape:
            raise ValueError("base-support expectation shapes have changed")
        if standardized_components.shape[-1] != self.base_width:
            raise ValueError("base-support expectation width has changed")
        return (
            log_weights.exp().unsqueeze(-1) * standardized_components
        ).sum(dim=-2)

    def muon_parameters(self) -> tuple[Tensor, Tensor]:
        return self.base_offset.weight, self.mass_logit.weight


class CausalKnotBasisGramLossLayer8(nn.Module):
    """Causal Q/K/V function approximator with a learned orthogonal DoG basis.

    Query points, knots, and knot values use the same learned projections as
    single-head attention. Aggregation weights are the raw normalized
    point/knot kernel values ``w=b``. For every causal prefix, ``G=avg(b^T b)``
    is returned to training through ``||G-I||_F^2`` rather than being inverted
    in the forward computation. Negative weights are intentional.
    """

    def __init__(
        self,
        input_width: int,
        point_width: int,
        knot_width: int,
        value_width: int,
        output_width: int,
        *,
        kernel_bandwidth: float,
        normalization_epsilon: float,
    ) -> None:
        super().__init__()
        dimensions = (
            input_width,
            point_width,
            knot_width,
            value_width,
            output_width,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("orthogonal knot-basis dimensions must be positive")
        if int(point_width) != int(knot_width):
            raise ValueError("orthogonal knot-basis point/knot widths must match")
        if min(float(kernel_bandwidth), float(normalization_epsilon)) <= 0:
            raise ValueError("knot-basis scales must be positive")
        self.input_width = int(input_width)
        self.point_width = int(point_width)
        self.knot_width = int(knot_width)
        self.value_width = int(value_width)
        self.output_width = int(output_width)
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.normalization_epsilon = float(normalization_epsilon)
        self.query = nn.Linear(self.input_width, self.point_width)
        self.key = nn.Linear(self.input_width, self.knot_width)
        self.value = nn.Linear(self.input_width, self.value_width)
        self.output = nn.Linear(self.value_width, self.output_width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query.weight, std=1 / math.sqrt(self.input_width))
        nn.init.normal_(self.key.weight, std=1 / math.sqrt(self.input_width))
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.output.bias)
        if self.input_width == self.value_width:
            nn.init.eye_(self.value.weight)
        else:
            nn.init.xavier_uniform_(self.value.weight)
        if self.value_width == self.output_width:
            nn.init.eye_(self.output.weight)
        else:
            nn.init.xavier_uniform_(self.output.weight)

    def kernel(self, points: Tensor, knots: Tensor) -> Tensor:
        """Derivative-of-Gaussian basis for normalized points and knots."""
        dot = torch.einsum("bd,bkd->bk", points, knots).clamp(-1.0, 1.0)
        distance_square = torch.clamp_min(2.0 - 2.0 * dot, 0.0)
        bandwidth_square = self.kernel_bandwidth**2
        return (1.0 - distance_square / bandwidth_square) * torch.exp(
            -distance_square / (2.0 * bandwidth_square)
        )

    def basis_weights(self, values: Tensor) -> tuple[Tensor, ...]:
        if values.ndim != 3 or values.shape[-1] != self.input_width:
            raise ValueError(
                "layer-8 knot basis expects "
                "[batch,steps,input_width]"
            )
        points = F.normalize(
            self.query(values), dim=-1, eps=self.normalization_epsilon
        )
        knots = F.normalize(
            self.key(values), dim=-1, eps=self.normalization_epsilon
        )
        result: list[Tensor] = []
        for step in range(values.shape[1]):
            knot_count = step + 1
            basis = self.kernel(
                points[:, step], knots[:, :knot_count]
            )
            result.append(basis)
        return tuple(result)

    def basis_weights_and_gram_identity_loss(
        self, values: Tensor
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        weights = self.basis_weights(values)
        newest = weights[-1]
        gram = newest.transpose(0, 1) @ newest / float(values.shape[0])
        identity = torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        )
        return weights, (gram - identity).square().mean()

    def forward_with_gram_identity_loss(
        self, values: Tensor
    ) -> tuple[Tensor, Tensor]:
        weights, gram_loss = self.basis_weights_and_gram_identity_loss(values)
        knot_values = self.value(values)
        approximations = [
            torch.einsum(
                "bk,bkv->bv", weight, knot_values[:, :step + 1]
            )
            for step, weight in enumerate(weights)
        ]
        return self.output(torch.stack(approximations, dim=1)), gram_loss

    def forward(self, values: Tensor) -> Tensor:
        weights = self.basis_weights(values)
        knot_values = self.value(values)
        approximations = [
            torch.einsum(
                "bk,bkv->bv", weight, knot_values[:, :step + 1]
            )
            for step, weight in enumerate(weights)
        ]
        return self.output(torch.stack(approximations, dim=1))

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            self.query.weight,
            self.key.weight,
            self.value.weight,
            self.output.weight,
        )


def _gnglu(
    input_width: int,
    output_width: int,
    *,
    initial_radius: float,
    minimum_radius: float,
    learnable_centering: bool,
    linear_rank: int | None,
) -> TensorPathGluBlock | LowRankTensorPathGluBlock:
    """Build one dense or trainable-low-rank expected-width GNGLU."""
    hidden_width = (int(input_width) + int(output_width)) // 2
    if linear_rank is not None:
        return LowRankTensorPathGluBlock(
            int(input_width),
            hidden_width,
            int(output_width),
            int(linear_rank),
            initial_radius=float(initial_radius),
            minimum_radius=float(minimum_radius),
            learnable_centering=bool(learnable_centering),
            output_bias=torch.zeros(int(output_width)),
        )
    block = TensorPathGluBlock(
        int(input_width),
        hidden_width,
        int(output_width),
        initial_radius=float(initial_radius),
        minimum_radius=float(minimum_radius),
        learnable_centering=bool(learnable_centering),
        output_bias=torch.zeros(int(output_width)),
    )
    # A zero final projection in every block would initially disconnect a deep
    # composition. Rectangular identity keeps every numbered path live while
    # preserving the GNGLU's normalized hidden representation.
    with torch.no_grad():
        nn.init.eye_(block.output.weight)
    return block


@dataclass(frozen=True)
class StructuredFeatureProcessTrace:
    feature_embeddings: tuple[Tensor, ...]
    market_states: tuple[Tensor, ...]
    prefix_states: tuple[Tensor, ...]
    feature_distribution_states: tuple[Tensor, ...]
    extended_prefix_states: tuple[Tensor, ...]
    next_feature_distribution_states: tuple[Tensor, ...]
    expected_feature_embeddings: tuple[Tensor, ...]
    outputs: Tensor
    layer8_gram_identity_loss: Tensor


class StructuredSharedIoFeatureProcess(nn.Module):
    """Structured K1-input/K2-output latent feature process from the diagram.

    Parenthesized diagram numbers correspond one-to-one with ``layer1`` through
    ``layer11``. Repeated calls reuse those exact module instances. The two
    unnumbered seed projections initialize market and prefix state from the
    first observed feature embedding.

    All externally supplied inputs are raw feature values. The emitted tensor
    is in per-output-feature standardized coordinates so a feature-balanced MSE
    can be optimized; ``raw_outputs`` maps it back to feature units.
    """

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        input_mean: Tensor,
        input_std: Tensor,
        output_mean: Tensor,
        output_std: Tensor,
        *,
        input_steps: int,
        output_steps: int,
        feature_width: int,
        market_width: int,
        prefix_width: int,
        feature_distribution_width: int,
        extended_prefix_width: int,
        next_feature_distribution_width: int,
        initial_radius: float,
        minimum_radius: float,
        learnable_centering: bool,
        linear_rank: int | None = None,
        layer8_attention: dict[str, int | str] | None = None,
        layer8_function_approximator: dict[str, object] | None = None,
        recurrent_memory: dict[str, object] | None = None,
        feature_embedding_density: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if input_mean.ndim != 1 or input_std.shape != input_mean.shape:
            raise ValueError("input feature statistics have incompatible shapes")
        if output_mean.ndim != 1 or output_std.shape != output_mean.shape:
            raise ValueError("output feature statistics have incompatible shapes")
        if bool((input_std <= 0).any()) or bool((output_std <= 0).any()):
            raise ValueError("feature standard deviations must be positive")
        dimensions = (
            input_steps,
            output_steps,
            feature_width,
            market_width,
            prefix_width,
            feature_distribution_width,
            extended_prefix_width,
            next_feature_distribution_width,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("structured feature-process dimensions must be positive")
        if linear_rank is not None and int(linear_rank) <= 0:
            raise ValueError("structured feature-process linear rank must be positive")

        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.input_width = int(input_mean.numel())
        self.output_width = int(output_mean.numel())
        self.feature_width = int(feature_width)
        self.market_width = int(market_width)
        self.prefix_width = int(prefix_width)
        self.feature_distribution_width = int(feature_distribution_width)
        self.extended_prefix_width = int(extended_prefix_width)
        self.next_feature_distribution_width = int(
            next_feature_distribution_width
        )
        self.linear_rank = None if linear_rank is None else int(linear_rank)
        self.layer8_attention = (
            None if layer8_attention is None else dict(layer8_attention)
        )
        self.layer8_function_approximator = (
            None
            if layer8_function_approximator is None
            else dict(layer8_function_approximator)
        )
        self.recurrent_memory = (
            None if recurrent_memory is None else dict(recurrent_memory)
        )
        self.feature_embedding_density = (
            None
            if feature_embedding_density is None
            else dict(feature_embedding_density)
        )
        if self.layer8_attention is not None \
                and self.layer8_function_approximator is not None:
            raise ValueError("layer 8 cannot be both attention and a function unit")
        recurrent_layer_numbers: frozenset[int] = frozenset()
        recurrent_hidden_width: int | None = None
        if self.recurrent_memory is not None:
            if self.recurrent_memory.get("type") \
                    != "dual-state-gated-exchange-v1":
                raise ValueError("unsupported recurrent-memory type")
            recurrent_layer_numbers = frozenset(
                int(value)
                for value in self.recurrent_memory.get("layers", ())
            )
            if recurrent_layer_numbers != frozenset((
                1, 2, 3, 4, 5, 6, 7, 9, 10, 11,
            )):
                raise ValueError(
                    "recurrent memory must cover every repeated GNGLU layer"
                )
            recurrent_hidden_width = int(
                self.recurrent_memory.get("hiddenWidth", 0)
            )
            if recurrent_hidden_width <= 0:
                raise ValueError("recurrent hidden width must be positive")
            if self.recurrent_memory.get("mix") \
                    != "a*t+b*(t-1)" \
                    or self.recurrent_memory.get("activation") != "sigmoid":
                raise ValueError("recurrent-memory cell formula changed")
            if self.linear_rank is not None:
                raise ValueError(
                    "dual-state recurrent cells do not use low-rank GNGLUs"
                )
        self.register_buffer("input_mean", input_mean.float().clone())
        self.register_buffer("input_std", input_std.float().clone())
        self.register_buffer("output_mean", output_mean.float().clone())
        self.register_buffer("output_std", output_std.float().clone())

        options = {
            "initial_radius": float(initial_radius),
            "minimum_radius": float(minimum_radius),
            "learnable_centering": bool(learnable_centering),
            "linear_rank": self.linear_rank,
        }

        def numbered_layer(
            number: int,
            input_width: int,
            output_width: int,
            *,
            value_input_width: int | None = None,
            gate_input_width: int | None = None,
        ) -> nn.Module:
            if number not in recurrent_layer_numbers:
                return _gnglu(input_width, output_width, **options)
            assert recurrent_hidden_width is not None
            return DualStateGatedExchangeCell(
                int(value_input_width or input_width),
                int(gate_input_width or input_width),
                output_width,
                recurrent_hidden_width,
                initial_radius=float(initial_radius),
                minimum_radius=float(minimum_radius),
                learnable_centering=bool(learnable_centering),
            )

        # Numbered, shared diagram layers.
        self.layer1 = numbered_layer(
            1, self.input_width, self.feature_width
        )
        self.layer2 = numbered_layer(
            2, self.feature_width, self.output_width
        )
        self.layer3 = numbered_layer(
            3,
            self.feature_width + self.market_width,
            self.feature_distribution_width,
            value_input_width=self.feature_width,
            gate_input_width=self.market_width,
        )
        self.layer4 = numbered_layer(
            4,
            self.prefix_width + self.feature_distribution_width,
            self.extended_prefix_width,
            value_input_width=self.prefix_width,
            gate_input_width=self.feature_distribution_width,
        )
        self.layer5 = numbered_layer(
            5,
            self.feature_width + self.market_width,
            self.market_width,
            value_input_width=self.feature_width,
            gate_input_width=self.market_width,
        )
        self.layer6 = numbered_layer(
            6,
            self.feature_width + self.extended_prefix_width,
            self.prefix_width,
            value_input_width=self.feature_width,
            gate_input_width=self.extended_prefix_width,
        )
        self.layer7 = numbered_layer(
            7,
            self.extended_prefix_width,
            self.next_feature_distribution_width,
        )
        if self.layer8_attention is None \
                and self.layer8_function_approximator is None:
            self.layer8 = _gnglu(
                self.next_feature_distribution_width,
                self.feature_width,
                **options,
            )
        elif self.layer8_attention is not None:
            if self.layer8_attention.get("type") \
                    != "causal-self-attention-v1":
                raise ValueError("unsupported layer-8 self-attention type")
            self.layer8 = CausalSelfAttentionLayer8(
                self.next_feature_distribution_width,
                int(self.layer8_attention["queryWidth"]),
                int(self.layer8_attention["keyWidth"]),
                int(self.layer8_attention["valueWidth"]),
                int(self.layer8_attention["outputWidth"]),
            )
            if self.layer8.output_width != self.feature_width:
                raise ValueError(
                    "layer-8 self-attention output must match feature width"
                )
        else:
            assert self.layer8_function_approximator is not None
            function = self.layer8_function_approximator
            if function.get("type") \
                    != "causal-dog-knot-basis-gram-loss-v1":
                raise ValueError("unsupported layer-8 function approximator type")
            kernel = function.get("kernel")
            orthogonalization_loss = function.get("orthogonalizationLoss")
            if not isinstance(kernel, dict) \
                    or kernel.get("type") \
                    != "normalized-distance-dog-v1" \
                    or not isinstance(orthogonalization_loss, dict) \
                    or orthogonalization_loss.get("type") \
                    != "gram-identity-mean-square-v1":
                raise ValueError("invalid layer-8 function basis configuration")
            self.layer8 = CausalKnotBasisGramLossLayer8(
                self.next_feature_distribution_width,
                int(function["pointWidth"]),
                int(function["knotWidth"]),
                int(function["valueWidth"]),
                int(function["outputWidth"]),
                kernel_bandwidth=float(kernel["bandwidth"]),
                normalization_epsilon=float(function["normalizationEpsilon"]),
            )
            if self.layer8.output_width != self.feature_width:
                raise ValueError(
                    "layer-8 function output must match feature width"
                )
        self.layer9 = numbered_layer(
            9,
            self.next_feature_distribution_width + self.market_width,
            self.market_width,
            value_input_width=self.next_feature_distribution_width,
            gate_input_width=self.market_width,
        )
        self.layer10 = numbered_layer(
            10,
            self.next_feature_distribution_width + self.extended_prefix_width,
            self.prefix_width,
            value_input_width=self.next_feature_distribution_width,
            gate_input_width=self.extended_prefix_width,
        )
        self.layer11 = numbered_layer(
            11,
            self.next_feature_distribution_width + self.market_width,
            self.feature_distribution_width,
            value_input_width=self.next_feature_distribution_width,
            gate_input_width=self.market_width,
        )
        if self.feature_embedding_density is None:
            self.embedding_density_attention = None
        else:
            density = self.feature_embedding_density
            density_type = density.get("type")
            common_valid = (
                int(density.get("queryWidth", 0)) == self.feature_width
                and int(density.get("keyWidth", 0)) == self.feature_width
                and density.get("queryNormalization") == "unit-rms"
                and density.get("targetEncoderGradient") == "stop-gradient"
            )
            legacy_valid = (
                density_type == "conditional-gaussian-mixture-attention-v1"
                and int(density.get("valueWidth", 0)) == 1
            )
            base_support_valid = (
                density_type
                == "causal-base-support-gaussian-mixture-density-v2"
                and int(density.get("valueWidth", 0)) == self.output_width
                and density.get("keyConstruction")
                == "causal-derived-feature-state-through-shared-layer1"
            )
            if not common_valid or not (legacy_valid or base_support_valid):
                raise ValueError("invalid feature-embedding density attention")
            if self.recurrent_memory is not None:
                raise ValueError(
                    "feature-embedding density requires stateless numbered layers"
                )
            if legacy_valid:
                self.embedding_density_attention = (
                    ConditionalEmbeddingGaussianMixtureAttention(
                        self.next_feature_distribution_width,
                        self.feature_width,
                        int(density["components"]),
                        fixed_standard_deviation=float(
                            density["fixedStandardDeviation"]
                        ),
                        normalization_epsilon=float(
                            density["normalizationEpsilon"]
                        ),
                    )
                )
            else:
                self.embedding_density_attention = (
                    ConditionalEmbeddingBaseSupportGaussianMixture(
                        self.next_feature_distribution_width,
                        self.feature_width,
                        self.output_width,
                        int(density["components"]),
                        fixed_standard_deviation=float(
                            density["fixedStandardDeviation"]
                        ),
                        normalization_epsilon=float(
                            density["normalizationEpsilon"]
                        ),
                    )
                )

        # Unnumbered seed blocks at the top of the diagram.
        self.initial_market = _gnglu(
            self.feature_width, self.market_width, **options
        )
        self.initial_prefix = _gnglu(
            self.feature_width, self.prefix_width, **options
        )

    def _forward_trace(
        self, inputs: Tensor, *, include_auxiliary_loss: bool = False
    ) -> StructuredFeatureProcessTrace:
        if inputs.ndim != 3 or inputs.shape[1:] != (
            self.input_steps,
            self.input_width,
        ):
            raise ValueError(
                "structured feature inputs must have shape "
                f"[batch,{self.input_steps},{self.input_width}]"
            )
        normalized = (inputs.float() - self.input_mean) / self.input_std
        recurrent_hidden: dict[int, Tensor] = {}

        def apply_numbered_layer(
            number: int,
            joined_input: Tensor,
            value_input: Tensor | None = None,
            gate_input: Tensor | None = None,
        ) -> Tensor:
            layer = getattr(self, f"layer{number}")
            if not isinstance(layer, DualStateGatedExchangeCell):
                return layer(joined_input)
            actual_value_input = (
                joined_input if value_input is None else value_input
            )
            actual_gate_input = (
                joined_input if gate_input is None else gate_input
            )
            value, hidden = layer(
                actual_value_input,
                actual_gate_input,
                recurrent_hidden.get(number),
            )
            recurrent_hidden[number] = hidden
            return value

        embeddings = tuple(
            apply_numbered_layer(1, normalized[:, step])
            for step in range(self.input_steps)
        )

        market = self.initial_market(embeddings[0])
        prefix = self.initial_prefix(embeddings[0])
        feature_distribution = apply_numbered_layer(
            3,
            torch.cat((embeddings[0], market), dim=-1),
            embeddings[0],
            market,
        )
        extended_prefix = apply_numbered_layer(
            4,
            torch.cat((prefix, feature_distribution), dim=-1),
            prefix,
            feature_distribution,
        )

        market_states = [market]
        prefix_states = [prefix]
        feature_distribution_states = [feature_distribution]
        extended_prefix_states = [extended_prefix]

        for embedding in embeddings[1:]:
            market = apply_numbered_layer(
                5,
                torch.cat((embedding, market), dim=-1),
                embedding,
                market,
            )
            prefix = apply_numbered_layer(
                6,
                torch.cat((embedding, extended_prefix), dim=-1),
                embedding,
                extended_prefix,
            )
            feature_distribution = apply_numbered_layer(
                3,
                torch.cat((embedding, market), dim=-1),
                embedding,
                market,
            )
            extended_prefix = apply_numbered_layer(
                4,
                torch.cat((prefix, feature_distribution), dim=-1),
                prefix,
                feature_distribution,
            )
            market_states.append(market)
            prefix_states.append(prefix)
            feature_distribution_states.append(feature_distribution)
            extended_prefix_states.append(extended_prefix)

        next_feature_distribution_states: list[Tensor] = []
        expected_feature_embeddings: list[Tensor] = []
        outputs: list[Tensor] = []
        layer8_gram_identity_losses: list[Tensor] = []
        next_feature_distribution: Tensor | None = None
        for output_step in range(self.output_steps):
            if output_step > 0:
                if next_feature_distribution is None:
                    raise RuntimeError("missing preceding output latent state")
                market = apply_numbered_layer(
                    9,
                    torch.cat((next_feature_distribution, market), dim=-1),
                    next_feature_distribution,
                    market,
                )
                prefix = apply_numbered_layer(
                    10,
                    torch.cat((
                        next_feature_distribution, extended_prefix,
                    ), dim=-1),
                    next_feature_distribution,
                    extended_prefix,
                )
                feature_distribution = apply_numbered_layer(
                    11,
                    torch.cat((
                        next_feature_distribution, market,
                    ), dim=-1),
                    next_feature_distribution,
                    market,
                )
                extended_prefix = apply_numbered_layer(
                    4,
                    torch.cat((prefix, feature_distribution), dim=-1),
                    prefix,
                    feature_distribution,
                )
                market_states.append(market)
                prefix_states.append(prefix)
                feature_distribution_states.append(feature_distribution)
                extended_prefix_states.append(extended_prefix)

            next_feature_distribution = apply_numbered_layer(
                7, extended_prefix
            )
            if isinstance(self.layer8, CausalSelfAttentionLayer8):
                causal_sequence = torch.stack((
                    *next_feature_distribution_states,
                    next_feature_distribution,
                ), dim=1)
                expected_embedding = self.layer8(causal_sequence)[:, -1]
            elif isinstance(self.layer8, CausalKnotBasisGramLossLayer8):
                causal_sequence = torch.stack((
                    *next_feature_distribution_states,
                    next_feature_distribution,
                ), dim=1)
                if include_auxiliary_loss:
                    approximations, gram_identity_loss = \
                        self.layer8.forward_with_gram_identity_loss(
                            causal_sequence
                        )
                    layer8_gram_identity_losses.append(gram_identity_loss)
                else:
                    approximations = self.layer8(causal_sequence)
                expected_embedding = approximations[:, -1]
            else:
                expected_embedding = self.layer8(next_feature_distribution)
            output = apply_numbered_layer(2, expected_embedding)
            next_feature_distribution_states.append(next_feature_distribution)
            expected_feature_embeddings.append(expected_embedding)
            outputs.append(output)

        return StructuredFeatureProcessTrace(
            feature_embeddings=embeddings,
            market_states=tuple(market_states),
            prefix_states=tuple(prefix_states),
            feature_distribution_states=tuple(feature_distribution_states),
            extended_prefix_states=tuple(extended_prefix_states),
            next_feature_distribution_states=tuple(
                next_feature_distribution_states
            ),
            expected_feature_embeddings=tuple(expected_feature_embeddings),
            outputs=torch.stack(outputs, dim=1),
            layer8_gram_identity_loss=(
                torch.stack(layer8_gram_identity_losses).mean()
                if layer8_gram_identity_losses
                else outputs[0].new_zeros(())
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return standardized predictions with shape ``[B,K2,O]``."""
        return self._forward_trace(inputs).outputs

    def forward_with_auxiliary_loss(
        self, inputs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return predictions and the layer-8 Gram-identity penalty."""
        trace = self._forward_trace(inputs, include_auxiliary_loss=True)
        return trace.outputs, trace.layer8_gram_identity_loss

    def feature_embedding_log_density(
        self,
        inputs: Tensor,
        raw_target_features: Tensor,
    ) -> Tensor:
        """Evaluate each realized future feature state under the NF density."""
        if self.embedding_density_attention is None:
            raise RuntimeError("feature-embedding density attention is disabled")
        if not isinstance(
            self.embedding_density_attention,
            ConditionalEmbeddingGaussianMixtureAttention,
        ):
            raise RuntimeError(
                "base-support density requires causally derived component features"
            )
        if raw_target_features.shape != (
            inputs.shape[0], self.output_steps, self.input_width,
        ):
            raise ValueError("feature-embedding density targets have the wrong shape")
        trace = self._forward_trace(inputs)
        normalized_targets = (
            raw_target_features.float() - self.input_mean
        ) / self.input_std
        # The same numbered layer (1) supplies the target query.  Stopping this
        # branch is essential: otherwise a trainable encoder can minimize NLL
        # by mapping every possible feature state to one identical point.
        queries = torch.stack(tuple(
            self.layer1(normalized_targets[:, step]).detach()
            for step in range(self.output_steps)
        ), dim=1)
        states = torch.stack(trace.next_feature_distribution_states, dim=1)
        anchors = torch.stack(trace.expected_feature_embeddings, dim=1)
        return self.embedding_density_attention(queries, states, anchors)

    def feature_embedding_base_distribution(
        self, inputs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return raw causal base support and log masses for every output step."""
        density = self.embedding_density_attention
        if not isinstance(
            density, ConditionalEmbeddingBaseSupportGaussianMixture,
        ):
            raise RuntimeError("base-support embedding density is disabled")
        trace = self._forward_trace(inputs)
        states = torch.stack(trace.next_feature_distribution_states, dim=1)
        standardized_components, log_weights = density.distribution_parameters(
            states, trace.outputs,
        )
        raw_components = (
            standardized_components * self.output_std + self.output_mean
        )
        return raw_components, log_weights

    def feature_embedding_log_density_from_base_support(
        self,
        raw_target_features: Tensor,
        raw_component_feature_states: Tensor,
        log_weights: Tensor,
    ) -> Tensor:
        """Evaluate targets after base support was causally rederived to features."""
        density = self.embedding_density_attention
        if not isinstance(
            density, ConditionalEmbeddingBaseSupportGaussianMixture,
        ):
            raise RuntimeError("base-support embedding density is disabled")
        if raw_target_features.ndim != 3 \
                or raw_target_features.shape[1:] != (
                    self.output_steps, self.input_width,
                ):
            raise ValueError("base-support density targets have the wrong shape")
        expected_components = (
            raw_target_features.shape[0], self.output_steps,
            density.component_count, self.input_width,
        )
        if raw_component_feature_states.shape != expected_components:
            raise ValueError("derived component feature states have the wrong shape")
        normalized_targets = (
            raw_target_features.float() - self.input_mean
        ) / self.input_std
        queries = torch.stack(tuple(
            self.layer1(normalized_targets[:, step]).detach()
            for step in range(self.output_steps)
        ), dim=1)
        flattened_components = raw_component_feature_states.reshape(
            -1, self.input_width,
        )
        normalized_components = (
            flattened_components.float() - self.input_mean
        ) / self.input_std
        component_embeddings = self.layer1(normalized_components).reshape(
            *expected_components[:-1], self.feature_width,
        )
        return density.log_density(
            queries, component_embeddings, log_weights,
        )

    def expected_raw_base_features(
        self, raw_components: Tensor, log_weights: Tensor
    ) -> Tensor:
        density = self.embedding_density_attention
        if not isinstance(
            density, ConditionalEmbeddingBaseSupportGaussianMixture,
        ):
            raise RuntimeError("base-support embedding density is disabled")
        if raw_components.shape != (
            *log_weights.shape, self.output_width,
        ):
            raise ValueError("raw base-support expectation shapes have changed")
        return (log_weights.exp().unsqueeze(-1) * raw_components).sum(dim=-2)

    def trace(self, inputs: Tensor) -> StructuredFeatureProcessTrace:
        return self._forward_trace(inputs)

    def raw_outputs(self, standardized_outputs: Tensor) -> Tensor:
        if standardized_outputs.ndim != 3 or standardized_outputs.shape[1:] != (
            self.output_steps,
            self.output_width,
        ):
            raise ValueError("standardized structured outputs have the wrong shape")
        return standardized_outputs * self.output_std + self.output_mean

    def standardized_targets(self, raw_targets: Tensor) -> Tensor:
        if raw_targets.ndim != 3 or raw_targets.shape[1:] != (
            self.output_steps,
            self.output_width,
        ):
            raise ValueError("raw structured targets have the wrong shape")
        return (raw_targets.float() - self.output_mean) / self.output_std

    def muon_parameters(self) -> tuple[Tensor, ...]:
        blocks = (
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
            self.layer8,
            self.layer9,
            self.layer10,
            self.layer11,
            self.initial_market,
            self.initial_prefix,
            self.embedding_density_attention,
        )
        return tuple(
            parameter
            for block in blocks
            if block is not None
            for parameter in block.muon_parameters()
        )
