"""CUDA-accelerated point-in-time portfolio-basis selection.

The implementation is the Gram-matrix equivalent of column-pivoted modified
Gram-Schmidt used by apps/server/src/portfolio-basis.ts. It processes many
independent rolling windows together and returns only the selected columns;
large diagnostic correlation matrices are intentionally not retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BasisSelection:
    selected: np.ndarray
    sizes: np.ndarray
    eligible_counts: np.ndarray
    median_r_squared: np.ndarray
    p10_r_squared: np.ndarray
    target_reached: np.ndarray


@torch.inference_mode()
def select_basis_batch(
    return_windows: np.ndarray,
    eligible: np.ndarray,
    *,
    anchor_index: int | None,
    min_size: int = 8,
    max_size: int = 512,
    target_median_r_squared: float = 0.8,
    target_p10_r_squared: float = 0.5,
    residual_equivalence_band: float = 0.05,
    device: str | torch.device = "cuda",
) -> BasisSelection:
    """Select one coverage-driven basis for each [asset, sample] window."""

    if return_windows.ndim != 3:
        raise ValueError("return_windows must have shape [batch, assets, samples]")
    batch_size, asset_count, sample_count = return_windows.shape
    if eligible.shape != (batch_size, asset_count):
        raise ValueError("eligible must have shape [batch, assets]")
    if sample_count < 2 or asset_count == 0 or batch_size == 0:
        raise ValueError("basis batches require non-empty assets and samples")
    if not 0 <= residual_equivalence_band <= 1:
        raise ValueError("residual_equivalence_band must be in [0, 1]")

    runtime_device = torch.device(device)
    values = torch.as_tensor(
        np.ascontiguousarray(return_windows),
        dtype=torch.float32,
        device=runtime_device,
    )
    active_assets = torch.as_tensor(
        np.ascontiguousarray(eligible),
        dtype=torch.bool,
        device=runtime_device,
    )
    finite = torch.isfinite(values).all(dim=2)
    active_assets &= finite
    # Preserve the native return amplitude before centering and normalizing the
    # vectors for the correlation/coverage calculation.  Measuring this after
    # normalization would erase exactly the volatility preference this tie-break
    # is intended to express.
    priority = torch.nan_to_num(values).abs().mean(dim=2)
    values = torch.nan_to_num(values)
    values -= values.mean(dim=2, keepdim=True)
    norms = torch.linalg.vector_norm(values, dim=2, keepdim=True)
    active_assets &= norms.squeeze(2) > 1e-12
    values /= torch.clamp(norms, min=1e-12)

    gram = torch.bmm(values, values.transpose(1, 2))
    maximum_rank = min(max_size, asset_count, sample_count - 1)
    selected = torch.full(
        (batch_size, maximum_rank),
        -1,
        dtype=torch.int32,
        device=runtime_device,
    )
    selected_mask = torch.zeros_like(active_assets)
    factor = torch.zeros(
        (batch_size, asset_count, maximum_rank),
        dtype=torch.float32,
        device=runtime_device,
    )
    residual = torch.ones(
        (batch_size, asset_count),
        dtype=torch.float32,
        device=runtime_device,
    )
    eligible_counts = active_assets.sum(dim=1)
    running = eligible_counts > 0
    target_reached = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=runtime_device,
    )
    sizes = torch.zeros(
        batch_size,
        dtype=torch.int32,
        device=runtime_device,
    )
    final_median = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=runtime_device,
    )
    final_p10 = torch.zeros_like(final_median)
    batch_indexes = torch.arange(batch_size, device=runtime_device)

    for rank in range(maximum_rank):
        candidates = active_assets & ~selected_mask & running[:, None]
        candidate_residual = torch.where(candidates, residual, -1.0)
        maximum_residual = candidate_residual.max(dim=1).values

        if rank == 0 and anchor_index is not None:
            forced = (
                active_assets[:, anchor_index]
                & ~selected_mask[:, anchor_index]
                & running
            )
        else:
            forced = torch.zeros_like(running)

        equivalent = candidates & (
            candidate_residual
            >= maximum_residual[:, None] * (1 - residual_equivalence_band)
        )
        equivalent_priority = torch.where(equivalent, priority, -1.0)
        maximum_priority = equivalent_priority.max(dim=1).values
        priority_ties = equivalent & (
            equivalent_priority >= maximum_priority[:, None] - 1e-12
        )
        residual_ties = torch.where(priority_ties, residual, -1.0)
        pivot = residual_ties.argmax(dim=1)
        if anchor_index is not None:
            pivot = torch.where(
                forced,
                torch.full_like(pivot, anchor_index),
                pivot,
            )
        selected[:, rank] = torch.where(
            running,
            pivot.to(torch.int32),
            -1,
        )
        selected_mask[batch_indexes, pivot] |= running

        gram_column = gram[batch_indexes, :, pivot]
        if rank == 0:
            correction: torch.Tensor | float = 0.0
        else:
            correction = (
                factor[:, :, :rank]
                * factor[batch_indexes, pivot, :rank][:, None, :]
            ).sum(dim=2)
        pivot_residual = torch.sqrt(
            torch.clamp(residual[batch_indexes, pivot], min=1e-12),
        )
        new_factor = (gram_column - correction) / pivot_residual[:, None]
        new_factor = torch.where(
            running[:, None],
            new_factor,
            torch.zeros_like(new_factor),
        )
        factor[:, :, rank] = new_factor
        residual = torch.clamp(
            residual - new_factor.square(),
            min=0,
            max=1,
        )
        sizes += running.to(torch.int32)

        median_r_squared, p10_r_squared = _coverage_quantiles(
            residual,
            active_assets,
            eligible_counts,
        )
        final_median = torch.where(
            running,
            median_r_squared,
            final_median,
        )
        final_p10 = torch.where(running, p10_r_squared, final_p10)
        reached_now = (
            running
            & (sizes >= min_size)
            & (median_r_squared >= target_median_r_squared)
            & (p10_r_squared >= target_p10_r_squared)
        )
        target_reached |= reached_now
        exhausted = sizes >= torch.minimum(
            eligible_counts.to(torch.int32),
            torch.full_like(sizes, maximum_rank),
        )
        running &= ~reached_now & ~exhausted
        # A Python bool forces a device synchronization. Check termination in
        # small blocks instead of after every pivot; stopped rows stay masked,
        # so the selected ranks and coverage stopping points remain exact.
        if (
            (rank + 1) % 8 == 0
            or rank + 1 == maximum_rank
        ) and not bool(running.any()):
            break

    return BasisSelection(
        selected=selected.cpu().numpy(),
        sizes=sizes.cpu().numpy(),
        eligible_counts=eligible_counts.cpu().numpy(),
        median_r_squared=final_median.cpu().numpy(),
        p10_r_squared=final_p10.cpu().numpy(),
        target_reached=target_reached.cpu().numpy(),
    )


def capped_proportional_weights_batch(
    sizes: np.ndarray,
    active: np.ndarray,
    maximum_weight: float = 0.05,
) -> np.ndarray:
    """Vectorized capped proportional weighting for padded selections."""

    if sizes.shape != active.shape or sizes.ndim != 2:
        raise ValueError("sizes and active must have matching 2D shapes")
    if not 0 < maximum_weight <= 1:
        raise ValueError("maximum_weight must be in (0, 1]")
    if np.any(sizes < 0) or not np.isfinite(sizes).all():
        raise ValueError("sizes must be finite and non-negative")

    weights = np.zeros_like(sizes, dtype=np.float64)
    remaining = np.ones(sizes.shape[0], dtype=np.float64)
    uncapped = active.copy()
    for _ in range(sizes.shape[1]):
        counts = uncapped.sum(axis=1)
        if not np.any(counts):
            break
        active_size = np.where(uncapped, sizes, 0).sum(axis=1)
        proportional = np.divide(
            remaining[:, None] * sizes,
            active_size[:, None],
            out=np.zeros_like(weights),
            where=active_size[:, None] > 0,
        )
        equal = np.divide(
            remaining,
            counts,
            out=np.zeros_like(remaining),
            where=counts > 0,
        )
        proposed = np.where(
            active_size[:, None] > 0,
            proportional,
            equal[:, None],
        )
        breaches = uncapped & (proposed > maximum_weight + 1e-12)
        rows_with_breaches = breaches.any(axis=1)
        if not np.any(rows_with_breaches):
            weights = np.where(uncapped, proposed, weights)
            break
        weights[breaches] = maximum_weight
        remaining -= breaches.sum(axis=1) * maximum_weight
        uncapped &= ~breaches

    weights[~active] = 0
    return weights.astype(np.float32)


def _coverage_quantiles(
    residual: torch.Tensor,
    eligible: torch.Tensor,
    eligible_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = torch.where(eligible, residual, torch.inf)
    sorted_residual = torch.sort(masked, dim=1).values
    median_residual = _row_quantile(sorted_residual, eligible_counts, 0.5)
    p90_residual = _row_quantile(sorted_residual, eligible_counts, 0.9)
    return 1 - median_residual, 1 - p90_residual


def _row_quantile(
    sorted_values: torch.Tensor,
    counts: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    positions = torch.clamp(counts - 1, min=0).to(torch.float32) * probability
    lower = positions.floor().to(torch.int64)
    upper = positions.ceil().to(torch.int64)
    weight = positions - lower
    batch = torch.arange(sorted_values.shape[0], device=sorted_values.device)
    values = (
        sorted_values[batch, lower] * (1 - weight)
        + sorted_values[batch, upper] * weight
    )
    return torch.where(counts > 0, values, torch.zeros_like(values))
