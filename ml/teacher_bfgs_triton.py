from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
import triton
import triton.language as tl

from mlp_model import PolicySupport


PARAMETER_COUNT = tl.constexpr(8)
PADDED_PARAMETER_COUNT = tl.constexpr(8)
ACTION_BLOCK = tl.constexpr(64)
STATE_BLOCK = tl.constexpr(32)


@dataclass(frozen=True)
class TritonBfgsDiagnostics:
    iterations: int
    host_checks: int
    active_remaining: int
    compactions: int


@triton.jit
def _softplus_scaled(offset, kappa):
    value = offset * kappa
    return (tl.maximum(value, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(value)))) / kappa


@triton.jit
def _isfinite(value):
    return (value == value) & (tl.abs(value) < float("inf"))


@triton.jit
def _objective_gradient(
    target_pointer,
    action_pointer,
    current_pointer,
    fit_index,
    raw0,
    raw1,
    raw2,
    raw3,
    raw4,
    raw5,
    raw6,
    raw7,
    latent_lower,
    latent_upper,
    visible_lower,
    visible_upper,
    hinge_span,
    friction,
    temperature,
    parameter_mask: tl.constexpr,
    action_count: tl.constexpr,
    target_stride: tl.constexpr,
    state_count: tl.constexpr,
):
    action_offset = tl.arange(0, ACTION_BLOCK)[None, :]
    state_offset = tl.arange(0, STATE_BLOCK)[:, None]
    valid_state = state_offset < state_count
    valid_action = action_offset < action_count
    valid = valid_state & valid_action
    target_offset = (
        fit_index * state_count * target_stride
        + state_offset * target_stride
        + action_offset
    )
    target = tl.load(target_pointer + target_offset, mask=valid, other=0.0)
    action = tl.load(action_pointer + action_offset, mask=valid_action, other=0.0)
    current = tl.load(current_pointer + state_offset, mask=valid_state, other=0.0)

    visible_span = visible_upper - visible_lower
    half_visible_span = visible_span * 0.5
    visible_center = (visible_lower + visible_upper) * 0.5
    latent_span = latent_upper - latent_lower
    first_fraction = tl.sigmoid(raw0)
    c1 = latent_lower + latent_span * first_fraction
    second_fraction = tl.sigmoid(raw1)
    c2 = c1 + (latent_upper - c1) * second_fraction
    dc1_raw0 = latent_span * first_fraction * (1.0 - first_fraction)
    dc2_raw0 = dc1_raw0 * (1.0 - second_fraction)
    dc2_raw1 = (latent_upper - c1) * second_fraction * (1.0 - second_fraction)

    # The transition widths are calibrated on the usable score span, which
    # can deliberately differ from the effective/visible support.  In
    # particular, the production fitter optimizes on [-100, 100] and remaps
    # the score to [-250, 250] without widening either hinge.  Keep the queued
    # optimizer on that same objective so adaptive work does not need to be
    # repaired by the PyTorch compatibility fallback.
    kappa_c = 82.0 / hinge_span
    kappa_x = 678.0 / hinge_span
    buy_slope = friction / (1.0 - friction)
    sell_slope = friction
    beta_x = -(buy_slope + sell_slope) / temperature
    c1_offset = action - c1
    c2_offset = action - c2
    c1_feature = _softplus_scaled(c1_offset, kappa_c)
    c2_feature = _softplus_scaled(c2_offset, kappa_c)
    moving_feature = _softplus_scaled(action - current, kappa_x)
    logits = (
        raw2 / visible_span * (action - latent_lower)
        - 0.5 * raw3 / (half_visible_span * half_visible_span)
        * (action - visible_center) * (action - visible_center)
        + raw4 / visible_span * c1_feature
        + beta_x * moving_feature
        + raw5 / visible_span * c2_feature
    )
    lower_fraction = tl.sigmoid(raw6)
    upper_fraction = tl.sigmoid(raw7)
    cutoff_lower = latent_lower + (-latent_lower) * lower_fraction
    cutoff_upper = latent_upper * upper_fraction
    cutoff_lower = tl.where(raw6 <= -13.999999, latent_lower, cutoff_lower)
    cutoff_lower = tl.where(raw6 >= 13.999999, 0.0, cutoff_lower)
    cutoff_upper = tl.where(raw7 <= -13.999999, 0.0, cutoff_upper)
    cutoff_upper = tl.where(raw7 >= 13.999999, latent_upper, cutoff_upper)
    feasible_action = (
        valid_action
        & (action >= visible_lower) & (action <= visible_upper)
        & (action >= cutoff_lower) & (action <= cutoff_upper)
    )
    # Invalid state rows get a finite dummy softmax whose loss and gradient are
    # masked away. Invalid action lanes remain negative infinity.
    # Match ``torch.finfo(torch.float32).min`` used by the shared policy
    # decoder.  IEEE -inf makes the padded action's 0 * -inf loss contribution
    # a NaN before masking on some Triton lowering paths; that used to
    # deactivate every queued fit in the initializer and silently leave all
    # useful recovery work to the much heavier compatibility fallback.
    masked_logit = -3.4028234663852886e38
    logits = tl.where(
        valid_state,
        tl.where(feasible_action, logits, masked_logit),
        tl.where(action_offset == 0, 0.0, masked_logit),
    )
    maximum = tl.max(logits, axis=1)
    exponential = tl.exp(logits - maximum[:, None])
    normalizer = tl.sum(exponential, axis=1)
    probability = exponential / normalizer[:, None]
    log_normalizer = maximum + tl.log(normalizer)
    row_loss = -tl.sum(
        tl.where(valid, target * (logits - log_normalizer[:, None]), 0.0),
        axis=1,
    )
    row_loss = tl.where(tl.arange(0, STATE_BLOCK) < state_count, row_loss, 0.0)
    objective = tl.sum(row_loss, axis=0) / state_count

    residual = tl.where(valid & feasible_action, (probability - target) / state_count, 0.0)
    sigmoid_c1 = tl.sigmoid(kappa_c * c1_offset)
    sigmoid_c2 = tl.sigmoid(kappa_c * c2_offset)
    derivative0 = (
        -raw4 / visible_span * sigmoid_c1 * dc1_raw0
        -raw5 / visible_span * sigmoid_c2 * dc2_raw0
    )
    derivative1 = -raw5 / visible_span * sigmoid_c2 * dc2_raw1
    derivative2 = (action - latent_lower) / visible_span
    derivative3 = -0.5 * (
        (action - visible_center) / half_visible_span
    ) * ((action - visible_center) / half_visible_span)
    derivative4 = c1_feature / visible_span
    derivative5 = c2_feature / visible_span
    gradient0 = tl.sum(tl.sum(residual * derivative0, axis=1), axis=0)
    gradient1 = tl.sum(tl.sum(residual * derivative1, axis=1), axis=0)
    gradient2 = tl.sum(tl.sum(residual * derivative2, axis=1), axis=0)
    gradient3 = tl.sum(tl.sum(residual * derivative3, axis=1), axis=0)
    gradient4 = tl.sum(tl.sum(residual * derivative4, axis=1), axis=0)
    gradient5 = tl.sum(tl.sum(residual * derivative5, axis=1), axis=0)

    excess2 = tl.maximum(tl.abs(raw2) - 100.0, 0.0)
    excess3 = tl.maximum(tl.abs(raw3) - 100.0, 0.0)
    excess4 = tl.maximum(tl.abs(raw4) - 100.0, 0.0)
    excess5 = tl.maximum(tl.abs(raw5) - 100.0, 0.0)
    objective += 1.0e-9 * (
        excess2 * excess2 + excess3 * excess3
        + excess4 * excess4 + excess5 * excess5
    )
    gradient2 += 2.0e-9 * excess2 * tl.where(raw2 < 0.0, -1.0, 1.0)
    gradient3 += 2.0e-9 * excess3 * tl.where(raw3 < 0.0, -1.0, 1.0)
    gradient4 += 2.0e-9 * excess4 * tl.where(raw4 < 0.0, -1.0, 1.0)
    gradient5 += 2.0e-9 * excess5 * tl.where(raw5 < 0.0, -1.0, 1.0)
    if parameter_mask == 0b111100:
        gradient0 = 0.0
        gradient1 = 0.0
    return (
        objective,
        gradient0,
        gradient1,
        gradient2,
        gradient3,
        gradient4,
        gradient5,
    )


@triton.jit
def _initialize_kernel(
    target_pointer,
    action_pointer,
    current_pointer,
    raw_pointer,
    hessian_pointer,
    loss_pointer,
    gradient_pointer,
    active_pointer,
    converged_pointer,
    stable_pointer,
    latent_lower,
    latent_upper,
    visible_lower,
    visible_upper,
    hinge_span,
    friction,
    temperature,
    parameter_mask: tl.constexpr,
    action_count: tl.constexpr,
    target_stride: tl.constexpr,
    state_count: tl.constexpr,
):
    fit = tl.program_id(0)
    parameter = tl.arange(0, PADDED_PARAMETER_COUNT)
    raw_base = fit * PARAMETER_COUNT
    raw = tl.load(raw_pointer + raw_base + parameter, mask=parameter < PARAMETER_COUNT, other=0.0)
    raw0 = tl.sum(tl.where(parameter == 0, raw, 0.0), axis=0)
    raw1 = tl.sum(tl.where(parameter == 1, raw, 0.0), axis=0)
    raw2 = tl.sum(tl.where(parameter == 2, raw, 0.0), axis=0)
    raw3 = tl.sum(tl.where(parameter == 3, raw, 0.0), axis=0)
    raw4 = tl.sum(tl.where(parameter == 4, raw, 0.0), axis=0)
    raw5 = tl.sum(tl.where(parameter == 5, raw, 0.0), axis=0)
    raw6 = tl.sum(tl.where(parameter == 6, raw, 0.0), axis=0)
    raw7 = tl.sum(tl.where(parameter == 7, raw, 0.0), axis=0)
    loss, g0, g1, g2, g3, g4, g5 = _objective_gradient(
        target_pointer,
        action_pointer,
        current_pointer,
        fit,
        raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7,
        latent_lower, latent_upper, visible_lower, visible_upper, hinge_span,
        friction, temperature,
        parameter_mask=parameter_mask,
        action_count=action_count,
        target_stride=target_stride,
        state_count=state_count,
    )
    gradient = tl.where(
        parameter == 0, g0,
        tl.where(parameter == 1, g1,
        tl.where(parameter == 2, g2,
        tl.where(parameter == 3, g3,
        tl.where(parameter == 4, g4,
        tl.where(parameter == 5, g5, 0.0))))),
    )
    tl.store(loss_pointer + fit, loss)
    tl.store(
        gradient_pointer + fit * PADDED_PARAMETER_COUNT + parameter,
        gradient,
    )
    row = tl.arange(0, PADDED_PARAMETER_COUNT)[:, None]
    column = tl.arange(0, PADDED_PARAMETER_COUNT)[None, :]
    tl.store(
        hessian_pointer + fit * PADDED_PARAMETER_COUNT * PADDED_PARAMETER_COUNT
        + row * PADDED_PARAMETER_COUNT + column,
        tl.where(row == column, 1.0, 0.0),
    )
    tl.store(active_pointer + fit, _isfinite(loss).to(tl.int8))
    tl.store(converged_pointer + fit, 0)
    tl.store(stable_pointer + fit, 0)


@triton.jit
def _step_kernel(
    fit_index_pointer,
    target_pointer,
    action_pointer,
    current_pointer,
    raw_pointer,
    hessian_pointer,
    loss_pointer,
    gradient_pointer,
    active_pointer,
    converged_pointer,
    stable_pointer,
    latent_lower,
    latent_upper,
    visible_lower,
    visible_upper,
    hinge_span,
    friction,
    temperature,
    tolerance,
    parameter_mask: tl.constexpr,
    action_count: tl.constexpr,
    target_stride: tl.constexpr,
    state_count: tl.constexpr,
    line_search_candidates: tl.constexpr,
):
    fit = tl.load(fit_index_pointer + tl.program_id(0))
    active = tl.load(active_pointer + fit).to(tl.int1)
    if active:
        parameter = tl.arange(0, PADDED_PARAMETER_COUNT)
        parameter_valid = parameter < PARAMETER_COUNT
        raw_base = fit * PARAMETER_COUNT
        raw = tl.load(raw_pointer + raw_base + parameter, mask=parameter_valid, other=0.0)
        gradient = tl.load(
            gradient_pointer + fit * PADDED_PARAMETER_COUNT + parameter
        )
        gradient = tl.where(parameter_valid, gradient, 0.0)
        loss = tl.load(loss_pointer + fit)
        gradient_maximum = tl.max(tl.abs(gradient), axis=0)
        newly_converged = gradient_maximum <= tolerance
        if newly_converged:
            tl.store(active_pointer + fit, 0)
            tl.store(converged_pointer + fit, 1)
        else:
            row = tl.arange(0, PADDED_PARAMETER_COUNT)[:, None]
            column = tl.arange(0, PADDED_PARAMETER_COUNT)[None, :]
            hessian = tl.load(
                hessian_pointer + fit * PADDED_PARAMETER_COUNT * PADDED_PARAMETER_COUNT
                + row * PADDED_PARAMETER_COUNT + column
            )
            direction = -tl.sum(hessian * gradient[None, :], axis=1)
            directional_derivative = tl.sum(gradient * direction, axis=0)
            invalid_direction = (
                directional_derivative >= 0.0
                or not _isfinite(directional_derivative)
            )
            identity = tl.where(row == column, 1.0, 0.0)
            hessian = tl.where(invalid_direction, identity, hessian)
            direction = tl.where(invalid_direction, -gradient, direction)
            directional_derivative = tl.sum(gradient * direction, axis=0)
            direction_maximum = tl.max(tl.abs(direction), axis=0)
            direction_scale = tl.minimum(1.0, 4.0 / tl.maximum(direction_maximum, 1.0e-30))
            direction *= direction_scale
            directional_derivative *= direction_scale
            d0 = tl.sum(tl.where(parameter == 0, direction, 0.0), axis=0)
            d1 = tl.sum(tl.where(parameter == 1, direction, 0.0), axis=0)
            d2 = tl.sum(tl.where(parameter == 2, direction, 0.0), axis=0)
            d3 = tl.sum(tl.where(parameter == 3, direction, 0.0), axis=0)
            d4 = tl.sum(tl.where(parameter == 4, direction, 0.0), axis=0)
            d5 = tl.sum(tl.where(parameter == 5, direction, 0.0), axis=0)
            r0 = tl.sum(tl.where(parameter == 0, raw, 0.0), axis=0)
            r1 = tl.sum(tl.where(parameter == 1, raw, 0.0), axis=0)
            r2 = tl.sum(tl.where(parameter == 2, raw, 0.0), axis=0)
            r3 = tl.sum(tl.where(parameter == 3, raw, 0.0), axis=0)
            r4 = tl.sum(tl.where(parameter == 4, raw, 0.0), axis=0)
            r5 = tl.sum(tl.where(parameter == 5, raw, 0.0), axis=0)
            r6 = tl.sum(tl.where(parameter == 6, raw, 0.0), axis=0)
            r7 = tl.sum(tl.where(parameter == 7, raw, 0.0), axis=0)
            accepted = False
            next0, next1, next2 = r0, r1, r2
            next3, next4, next5 = r3, r4, r5
            next_loss = loss
            for candidate_index in tl.static_range(0, line_search_candidates):
                step = 0.5 ** candidate_index
                candidate0 = tl.minimum(14.0, tl.maximum(-14.0, r0 + step * d0))
                candidate1 = tl.minimum(14.0, tl.maximum(-14.0, r1 + step * d1))
                candidate2 = tl.minimum(1.0e4, tl.maximum(-1.0e4, r2 + step * d2))
                candidate3 = tl.minimum(1.0e4, tl.maximum(-1.0e4, r3 + step * d3))
                candidate4 = tl.minimum(1.0e4, tl.maximum(-1.0e4, r4 + step * d4))
                candidate5 = tl.minimum(1.0e4, tl.maximum(-1.0e4, r5 + step * d5))
                candidate_loss, _, _, _, _, _, _ = _objective_gradient(
                    target_pointer,
                    action_pointer,
                    current_pointer,
                    fit,
                    candidate0, candidate1, candidate2,
                    candidate3, candidate4, candidate5, r6, r7,
                    latent_lower, latent_upper, visible_lower, visible_upper, hinge_span,
                    friction, temperature,
                    parameter_mask=parameter_mask,
                    action_count=action_count,
                    target_stride=target_stride,
                    state_count=state_count,
                )
                take = (
                    not accepted
                    and _isfinite(candidate_loss)
                    and candidate_loss <= loss + 1.0e-4 * step * directional_derivative
                )
                next0 = tl.where(take, candidate0, next0)
                next1 = tl.where(take, candidate1, next1)
                next2 = tl.where(take, candidate2, next2)
                next3 = tl.where(take, candidate3, next3)
                next4 = tl.where(take, candidate4, next4)
                next5 = tl.where(take, candidate5, next5)
                next_loss = tl.where(take, candidate_loss, next_loss)
                accepted = accepted or take
            if accepted:
                (
                    next_loss,
                    ng0, ng1, ng2, ng3, ng4, ng5,
                ) = _objective_gradient(
                    target_pointer,
                    action_pointer,
                    current_pointer,
                    fit,
                    next0, next1, next2, next3, next4, next5, r6, r7,
                    latent_lower, latent_upper, visible_lower, visible_upper, hinge_span,
                    friction, temperature,
                    parameter_mask=parameter_mask,
                    action_count=action_count,
                    target_stride=target_stride,
                    state_count=state_count,
                )
                next_raw = tl.where(
                    parameter == 0, next0,
                    tl.where(parameter == 1, next1,
                    tl.where(parameter == 2, next2,
                    tl.where(parameter == 3, next3,
                    tl.where(parameter == 4, next4,
                    tl.where(parameter == 5, next5,
                    tl.where(parameter == 6, r6,
                    tl.where(parameter == 7, r7, 0.0))))))),
                )
                next_gradient = tl.where(
                    parameter == 0, ng0,
                    tl.where(parameter == 1, ng1,
                    tl.where(parameter == 2, ng2,
                    tl.where(parameter == 3, ng3,
                    tl.where(parameter == 4, ng4,
                    tl.where(parameter == 5, ng5, 0.0))))),
                )
                parameter_delta = next_raw - raw
                gradient_delta = next_gradient - gradient
                curvature = tl.sum(parameter_delta * gradient_delta, axis=0)
                hessian_gradient = tl.sum(hessian * gradient_delta[None, :], axis=1)
                gradient_hessian_gradient = tl.sum(
                    gradient_delta * hessian_gradient, axis=0
                )
                valid_curvature = _isfinite(curvature) and curvature > 1.0e-12
                safe_curvature = tl.maximum(curvature, 1.0e-12)
                updated_hessian = (
                    hessian
                    + (safe_curvature + gradient_hessian_gradient)
                    / (safe_curvature * safe_curvature)
                    * parameter_delta[:, None] * parameter_delta[None, :]
                    - (
                        hessian_gradient[:, None] * parameter_delta[None, :]
                        + parameter_delta[:, None] * hessian_gradient[None, :]
                    ) / safe_curvature
                )
                hessian = tl.where(valid_curvature, updated_hessian, identity)
                relative_improvement = (loss - next_loss) / tl.maximum(tl.abs(loss), 1.0)
                stable = tl.load(stable_pointer + fit)
                stable = tl.where(relative_improvement <= tolerance, stable + 1, 0)
                stable_convergence = (
                    stable >= 4
                    and tl.max(tl.abs(next_gradient), axis=0)
                    <= tl.maximum(tolerance * 10.0, 1.0e-5)
                )
                tl.store(
                    raw_pointer + raw_base + parameter,
                    next_raw,
                    mask=parameter_valid,
                )
                tl.store(loss_pointer + fit, next_loss)
                tl.store(
                    gradient_pointer + fit * PADDED_PARAMETER_COUNT + parameter,
                    next_gradient,
                )
                tl.store(
                    hessian_pointer + fit * PADDED_PARAMETER_COUNT * PADDED_PARAMETER_COUNT
                    + row * PADDED_PARAMETER_COUNT + column,
                    hessian,
                )
                tl.store(stable_pointer + fit, stable)
                if stable_convergence:
                    tl.store(active_pointer + fit, 0)
                    tl.store(converged_pointer + fit, 1)
            else:
                tl.store(active_pointer + fit, 0)


def triton_bfgs(
    initial_raw: Tensor,
    target: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
    maximum_iterations: int,
    tolerance: float,
    parameter_mask: Tensor,
    line_search_candidates: int,
    check_interval: int = 32,
) -> tuple[Tensor, Tensor, Tensor, TritonBfgsDiagnostics]:
    if initial_raw.ndim != 2 or initial_raw.shape[1] != PARAMETER_COUNT:
        raise ValueError("Triton BFGS needs an [examples, 8] initial parameter tensor")
    if target.ndim != 3 or target.shape[0] != initial_raw.shape[0]:
        raise ValueError("Triton BFGS target must be [examples, states, actions]")
    if target.shape[1] > STATE_BLOCK or target.shape[2] > ACTION_BLOCK:
        raise ValueError("Triton BFGS supports at most 32 states and 64 actions")
    if not all(value.is_cuda and value.dtype == torch.float32 for value in (
        initial_raw, target, actions, currents,
    )):
        raise ValueError("Triton BFGS requires contiguous CUDA Float32 tensors")
    if line_search_candidates < 1 or line_search_candidates > 12:
        raise ValueError("Triton BFGS supports 1..12 line-search candidates")
    raw = initial_raw.detach().contiguous().clone()
    target = target.detach().contiguous()
    action_count = target.shape[2]
    # Each state's probability row starts on a 256-byte boundary. Besides
    # coalescing the 64-lane loads, this keeps different fit programs from
    # sharing cache lines when the real grid has 63 sampled actions.
    if action_count < ACTION_BLOCK:
        aligned_target = torch.zeros(
            (target.shape[0], target.shape[1], ACTION_BLOCK),
            dtype=target.dtype,
            device=target.device,
        )
        aligned_target[:, :, :action_count] = target
        target = aligned_target
    actions = actions.detach().contiguous()
    currents = currents.detach().contiguous()
    example_count = raw.shape[0]
    hessian = torch.empty(
        (example_count, PADDED_PARAMETER_COUNT, PADDED_PARAMETER_COUNT),
        dtype=torch.float32,
        device=raw.device,
    )
    loss = torch.empty(example_count, dtype=torch.float32, device=raw.device)
    gradient = torch.empty(
        (example_count, PADDED_PARAMETER_COUNT), dtype=torch.float32, device=raw.device
    )
    active = torch.empty(example_count, dtype=torch.int8, device=raw.device)
    converged = torch.empty(example_count, dtype=torch.int8, device=raw.device)
    stable = torch.empty(example_count, dtype=torch.int32, device=raw.device)
    mask_values = parameter_mask.detach().to("cpu").tolist()
    mask_bits = sum((1 << index) for index, value in enumerate(mask_values) if value != 0)
    if mask_bits not in (0b00111100, 0b00111111):
        raise ValueError("Triton BFGS optimizes the linear or complete six score parameters")
    hinge_span = support.hinge_span
    if hinge_span is None:
        hinge_span = float(support.latent_upper) - float(support.latent_lower)
    common = dict(
        latent_lower=float(support.latent_lower),
        latent_upper=float(support.latent_upper),
        visible_lower=float(support.visible_lower),
        visible_upper=float(support.visible_upper),
        hinge_span=float(hinge_span),
        friction=float(support.friction),
        temperature=float(support.temperature),
        parameter_mask=mask_bits,
        action_count=action_count,
        target_stride=target.shape[2],
        state_count=target.shape[1],
        num_warps=8,
    )
    _initialize_kernel[(example_count,)](
        target,
        actions,
        currents,
        raw,
        hessian,
        loss,
        gradient,
        active,
        converged,
        stable,
        **common,
    )
    checks = 0
    compactions = 0
    iterations = 0
    active_indices = torch.arange(
        example_count, dtype=torch.int32, device=raw.device
    )
    for iteration in range(maximum_iterations):
        _step_kernel[(active_indices.numel(),)](
            active_indices,
            target,
            actions,
            currents,
            raw,
            hessian,
            loss,
            gradient,
            active,
            converged,
            stable,
            tolerance=float(tolerance),
            line_search_candidates=line_search_candidates,
            **common,
        )
        iterations = iteration + 1
        if iterations % check_interval == 0 or iterations == maximum_iterations:
            checks += 1
            active_indices = torch.nonzero(active, as_tuple=False).flatten().to(torch.int32)
            compactions += 1
            if active_indices.numel() == 0:
                break
    remaining = active_indices.numel()
    return raw, loss, converged.bool(), TritonBfgsDiagnostics(
        iterations=iterations,
        host_checks=checks,
        active_remaining=remaining,
        compactions=compactions,
    )
