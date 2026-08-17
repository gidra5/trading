from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def regression_scale_gradients(
    predictions: Tensor,
    targets: Tensor,
    *,
    target_std: float,
) -> Tensor:
    """Per-example d/d(scale) normalized MSE at scale=1."""
    if predictions.ndim != 1 or targets.shape != predictions.shape:
        raise ValueError("EIIL regression inputs must be equal one-dimensional tensors")
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("EIIL target standard deviation must be finite and positive")
    normalized_prediction = predictions.float() / float(target_std)
    normalized_residual = (
        predictions.float() - targets.float()
    ) / float(target_std)
    return 2.0 * normalized_prediction * normalized_residual


def infer_binary_environment_ids(scale_gradients: np.ndarray) -> np.ndarray:
    """Return the hard optimum of binary EIIL for a scalar scale gradient.

    EIIL maximizes the sum of squared environment IRMv1 gradients. With two
    environments and a scalar reference output scale, the objective depends
    only on sums of the per-example gradients. Its hard global optimum places
    positive and negative contributions in opposite environments.
    """
    values = np.asarray(scale_gradients, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("EIIL requires at least two finite scale gradients")
    environment_ids = (values <= 0).astype(np.uint8)
    counts = np.bincount(environment_ids, minlength=2)
    if bool((counts == 0).any()):
        # Zero-gradient examples do not affect the EIIL objective. If all
        # informative gradients share one sign, use the exact rank boundary
        # that maximizes the same binary hard-assignment objective while
        # keeping both inferred environments non-empty.
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(values[order], dtype=np.float64)
        total = float(cumulative[-1])
        objective = np.square(cumulative[:-1]) + np.square(
            total - cumulative[:-1]
        )
        split = int(np.argmax(objective)) + 1
        environment_ids.fill(1)
        environment_ids[order[:split]] = 0
    return environment_ids


def binary_eiil_objective(
    scale_gradients: np.ndarray,
    environment_ids: np.ndarray,
) -> float:
    values = np.asarray(scale_gradients, dtype=np.float64)
    groups = np.asarray(environment_ids)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("EIIL objective arrays must have matching shapes")
    if not np.isin(groups, (0, 1)).all():
        raise ValueError("EIIL environment IDs must be binary")
    # The released EIIL implementation masks each environment and retains the
    # pooled-example denominator rather than renormalizing by group size.
    pooled = float(values.size)
    return sum(
        float(values[groups == environment].sum(dtype=np.float64) / pooled) ** 2
        for environment in (0, 1)
    ) / 2.0


def irm_v1_regression_objective(
    predictions: Tensor,
    targets: Tensor,
    weights: Tensor,
    environment_ids: Tensor,
    *,
    target_std: float,
    penalty_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Uniform-environment MSE plus the IRMv1 output-scale penalty."""
    if predictions.ndim != 1 or targets.shape != predictions.shape \
            or weights.shape != predictions.shape \
            or environment_ids.shape != predictions.shape:
        raise ValueError("IRM regression batch tensors must have matching vectors")
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("IRM target standard deviation must be finite and positive")
    if not math.isfinite(penalty_weight) or penalty_weight <= 0:
        raise ValueError("IRM penalty weight must be finite and positive")
    scale = torch.ones((), dtype=predictions.dtype, device=predictions.device)
    scale.requires_grad_(True)
    per_example = (
        (predictions * scale - targets.float()) / float(target_std)
    ).square()
    environment_losses: list[Tensor] = []
    environment_penalties: list[Tensor] = []
    for environment in torch.unique(environment_ids.long(), sorted=True):
        mask = environment_ids.long() == environment
        environment_weights = weights[mask]
        denominator = environment_weights.sum()
        if not bool(denominator > 0):
            continue
        environment_loss = (
            per_example[mask] * environment_weights
        ).sum() / denominator
        gradient = torch.autograd.grad(
            environment_loss, scale, create_graph=True
        )[0]
        environment_losses.append(environment_loss)
        environment_penalties.append(gradient.square())
    if not environment_losses:
        raise RuntimeError("IRM batch contains no non-empty environments")
    risk = torch.stack(environment_losses).mean()
    penalty = torch.stack(environment_penalties).mean()
    objective = risk + float(penalty_weight) * penalty
    if penalty_weight > 1:
        objective = objective / float(penalty_weight)
    return objective, risk, penalty
