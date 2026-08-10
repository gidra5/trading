from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import torch


DEFAULT_QUANTILE_LEVELS = np.asarray(
    (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    dtype=np.float64,
)


def generate_sample_paths(
    predictor,
    normalized_context: np.ndarray,
    context_stamps: np.ndarray,
    target_stamps: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    horizon: int,
    temperature: float,
    top_p: float,
    sample_count: int,
    inference_precision: str = "float32",
) -> np.ndarray:
    """Generate and retain Kronos paths instead of averaging them internally."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    batch_size = normalized_context.shape[0]
    if means.shape != stds.shape or means.shape[0] != batch_size:
        raise ValueError("normalization statistics do not match the context batch")
    expanded_context = np.repeat(normalized_context, sample_count, axis=0)
    expanded_context_stamps = np.repeat(context_stamps, sample_count, axis=0)
    expanded_target_stamps = np.repeat(target_stamps, sample_count, axis=0)
    if inference_precision not in ("float32", "float16"):
        raise ValueError(
            "inference_precision must be either 'float32' or 'float16'"
        )
    if inference_precision == "float16":
        device_type = torch.device(predictor.device).type
        if device_type != "cuda":
            raise ValueError("float16 Kronos inference requires a CUDA device")
        precision_context = torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
        )
    else:
        precision_context = nullcontext()
    with precision_context:
        decoded = predictor.generate(
            expanded_context,
            expanded_context_stamps,
            expanded_target_stamps,
            horizon,
            temperature,
            0,
            top_p,
            1,
            False,
        )
    expected_shape = (
        batch_size * sample_count,
        horizon,
        means.shape[1],
    )
    if decoded.shape != expected_shape:
        raise ValueError(
            f"unexpected Kronos path shape {decoded.shape}; expected {expected_shape}"
        )
    paths = decoded.reshape(
        batch_size,
        sample_count,
        horizon,
        means.shape[1],
    )
    return paths * (stds[:, None, None, :] + 1e-5) \
        + means[:, None, None, :]


def empirical_ohlc_quantiles(
    paths: np.ndarray,
    levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
) -> np.ndarray:
    """Return marginal OHLC quantiles with shape [batch, horizon, quantile, 4]."""
    values = np.asarray(paths, dtype=np.float64)
    quantile_levels = np.asarray(levels, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] < 4:
        raise ValueError("paths must have shape [batch, sample, horizon, feature]")
    if quantile_levels.ndim != 1 or quantile_levels.size == 0 \
            or np.any(np.diff(quantile_levels) <= 0) \
            or quantile_levels[0] <= 0 or quantile_levels[-1] >= 1:
        raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
    quantiles = np.quantile(
        values[..., :4],
        quantile_levels,
        axis=1,
        method="linear",
    )
    return np.moveaxis(quantiles, 0, 2)


_KLINE_CONSTRAINTS = np.asarray(
    (
        (1.0, -1.0, 0.0, 0.0),   # open <= high
        (0.0, -1.0, 0.0, 1.0),   # close <= high
        (-1.0, 0.0, 1.0, 0.0),   # low <= open
        (0.0, 0.0, 1.0, -1.0),   # low <= close
    ),
    dtype=np.float64,
)


def _kline_projection_matrices() -> tuple[np.ndarray, ...]:
    identity = np.eye(4, dtype=np.float64)
    matrices: list[np.ndarray] = []
    for mask in range(1 << _KLINE_CONSTRAINTS.shape[0]):
        active = np.flatnonzero([
            bool(mask & (1 << index))
            for index in range(_KLINE_CONSTRAINTS.shape[0])
        ])
        if active.size == 0:
            matrices.append(identity)
            continue
        constraints = _KLINE_CONSTRAINTS[active]
        gram = constraints @ constraints.T
        matrices.append(
            identity - constraints.T @ np.linalg.pinv(gram) @ constraints
        )
    return tuple(matrices)


_KLINE_PROJECTIONS = _kline_projection_matrices()


def project_kline_rows(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto low <= {open, close} <= high."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != 4 or not np.isfinite(array).all():
        raise ValueError("K-line rows must be finite OHLC vectors")
    flat = array.reshape(-1, 4)
    best = np.empty_like(flat)
    best_error = np.full(flat.shape[0], np.inf, dtype=np.float64)
    for projection in _KLINE_PROJECTIONS:
        candidate = flat @ projection.T
        feasible = np.all(
            candidate @ _KLINE_CONSTRAINTS.T <= 1e-10,
            axis=1,
        )
        error = np.square(candidate - flat).sum(axis=1)
        use = feasible & (error < best_error)
        best[use] = candidate[use]
        best_error[use] = error[use]
    if not np.isfinite(best_error).all():
        raise RuntimeError("failed to project one or more K-line rows")
    return best.reshape(array.shape)


def isotonic_projection(values: np.ndarray) -> np.ndarray:
    """Unweighted PAVA projection of one vector onto the monotone cone."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("isotonic input must be a finite vector")
    means: list[float] = []
    weights: list[int] = []
    for value in array:
        means.append(float(value))
        weights.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (
                means[-2] * weights[-2] + means[-1] * weights[-1]
            ) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
    output = np.empty_like(array)
    offset = 0
    for mean, weight in zip(means, weights, strict=True):
        output[offset:offset + weight] = mean
        offset += weight
    return output


def kqsp(quantiles: np.ndarray) -> np.ndarray:
    """Apply K-line then quantile minimum-distance projections."""
    array = np.asarray(quantiles, dtype=np.float64)
    if array.ndim != 4 or array.shape[-1] != 4:
        raise ValueError(
            "quantiles must have shape [batch, horizon, quantile, 4]"
        )
    projected = project_kline_rows(array)
    flat = projected.reshape(-1, projected.shape[-2], 4)
    for row in flat:
        for feature in range(4):
            row[:, feature] = isotonic_projection(row[:, feature])
    if not np.all(np.diff(projected, axis=2) >= -1e-10):
        raise RuntimeError("KQSP failed to remove quantile crossings")
    if not np.all(kline_valid_mask(projected)):
        raise RuntimeError("KQSP failed to remove K-line crossings")
    return projected


def kline_valid_mask(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != 4:
        raise ValueError("K-line values must end with four OHLC features")
    open_price, high, low, close = np.moveaxis(array, -1, 0)
    return (
        (high >= np.maximum(open_price, close) - 1e-10)
        & (low <= np.minimum(open_price, close) + 1e-10)
    )


def point_estimators(
    paths: np.ndarray,
    repaired_quantiles: np.ndarray,
    levels: Sequence[float] | np.ndarray = DEFAULT_QUANTILE_LEVELS,
) -> dict[str, np.ndarray]:
    quantile_levels = np.asarray(levels, dtype=np.float64)
    median_indexes = np.flatnonzero(np.isclose(quantile_levels, 0.5))
    if median_indexes.size != 1:
        raise ValueError("KQSP point estimation requires a 0.5 quantile")
    mean = np.mean(paths[..., :4], axis=1)
    median = np.median(paths[..., :4], axis=1)
    repaired_median = repaired_quantiles[:, :, int(median_indexes[0]), :]
    projected_mean = project_kline_rows(mean)
    return {
        "ensembleMean": mean,
        "projectedMean": projected_mean,
        "ensembleMedian": median,
        "kqspMedian": repaired_median,
    }
