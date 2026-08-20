"""Invertible transforms and positive multidimensional triangular densities."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.spatial import cKDTree
from scipy.stats import qmc


EPSILON = 1e-12


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def stable_logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if bool(((values <= 0) | (values >= 1)).any()):
        raise ValueError("logit input must lie strictly inside (0, 1)")
    return np.log(values) - np.log1p(-values)


@dataclass(frozen=True)
class CovarianceTransform:
    center: np.ndarray
    whitening: np.ndarray
    coloring: np.ndarray
    covariance: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, ridge: float = 1e-8) -> "CovarianceTransform":
        values = _matrix(values)
        center = np.mean(values, axis=0, dtype=np.float64)
        covariance = np.cov(values, rowvar=False, dtype=np.float64)
        covariance = np.atleast_2d(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(float(np.max(eigenvalues)) * ridge, np.finfo(np.float64).tiny)
        eigenvalues = np.maximum(eigenvalues, floor)
        whitening = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
        coloring = (eigenvectors * eigenvalues ** 0.5) @ eigenvectors.T
        return cls(center, whitening, coloring, covariance)

    def forward(self, values: np.ndarray) -> np.ndarray:
        return (_matrix(values) - self.center) @ self.whitening.T

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return _matrix(values) @ self.coloring.T + self.center


@dataclass(frozen=True)
class AsinhMatrixTransform:
    """z -> sigmoid(A @ asinh(z)), with a closed-form inverse."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transform matrix must be square")
        sign, log_determinant = np.linalg.slogdet(matrix)
        if sign == 0 or not math.isfinite(float(log_determinant)):
            raise ValueError("transform matrix must be invertible")
        object.__setattr__(self, "matrix", matrix)

    def forward(self, whitened: np.ndarray) -> np.ndarray:
        latent = np.arcsinh(_matrix(whitened)) @ self.matrix.T
        return stable_sigmoid(latent)

    def inverse(self, unit: np.ndarray) -> np.ndarray:
        inverse = np.linalg.inv(self.matrix)
        latent = stable_logit(_matrix(unit)) @ inverse.T
        return np.sinh(latent)

    def log_abs_jacobian(self, whitened: np.ndarray) -> np.ndarray:
        whitened = _matrix(whitened)
        latent = np.arcsinh(whitened) @ self.matrix.T
        _, log_determinant = np.linalg.slogdet(self.matrix)
        logistic_log_derivative = -np.logaddexp(0.0, -latent) \
            - np.logaddexp(0.0, latent)
        asinh_log_derivative = -0.5 * np.log1p(whitened * whitened)
        return float(log_determinant) + np.sum(
            logistic_log_derivative + asinh_log_derivative,
            axis=1,
        )


def factorized_matrix(
    dimension: int,
    rotation: Sequence[float],
    shear: Sequence[float],
    log_scales: Sequence[float],
) -> np.ndarray:
    """Return R H D with R in SO(d), H unit upper triangular, D positive."""
    if dimension not in (2, 3):
        raise ValueError("only two- and three-dimensional transforms are supported")
    expected_angles = dimension * (dimension - 1) // 2
    if len(rotation) != expected_angles or len(shear) != expected_angles:
        raise ValueError("rotation and shear parameter counts do not match dimension")
    if len(log_scales) != dimension:
        raise ValueError("scale parameter count does not match dimension")
    skew = np.zeros((dimension, dimension), dtype=np.float64)
    upper = np.eye(dimension, dtype=np.float64)
    index = 0
    for row in range(dimension):
        for column in range(row + 1, dimension):
            skew[row, column] = float(rotation[index])
            skew[column, row] = -float(rotation[index])
            upper[row, column] = float(shear[index])
            index += 1
    rotation_matrix = expm(skew)
    scales = np.diag(np.exp(np.asarray(log_scales, dtype=np.float64)))
    return rotation_matrix @ upper @ scales


def triangular_basis_areas(knots: np.ndarray) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64)
    if knots.ndim != 1 or knots.size < 2 or np.any(np.diff(knots) <= 0):
        raise ValueError("knots must be a strictly increasing vector")
    gaps = np.diff(knots)
    areas = np.empty(knots.size, dtype=np.float64)
    areas[0] = gaps[0] / 2.0
    areas[-1] = gaps[-1] / 2.0
    if knots.size > 2:
        areas[1:-1] = (gaps[:-1] + gaps[1:]) / 2.0
    return areas


def hat_interval_masses(edges: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Exact interval masses of normalized piecewise-linear grid hats."""
    edges = np.asarray(edges, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    areas = triangular_basis_areas(knots)
    cdf = np.empty((edges.size, knots.size), dtype=np.float64)
    for index in range(knots.size):
        left = knots[index - 1] if index > 0 else knots[index]
        center = knots[index]
        right = knots[index + 1] if index + 1 < knots.size else knots[index]
        values = np.zeros(edges.size, dtype=np.float64)
        if center > left:
            selected = (edges > left) & (edges < center)
            distance = edges[selected] - left
            values[selected] = distance * distance / (2.0 * (center - left))
            values[edges >= center] = (center - left) / 2.0
        if right > center:
            selected = (edges > center) & (edges < right)
            distance = edges[selected] - center
            values[selected] += distance - distance * distance / (2.0 * (right - center))
            values[edges >= right] += (right - center) / 2.0
        cdf[:, index] = values / areas[index]
    masses = np.diff(cdf, axis=0)
    masses[masses < 0] = 0
    return masses


def triangle_interval_masses(
    edges: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """Exact interval masses of independently boundary-normalized triangles."""
    edges = np.asarray(edges, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    if centers.shape != widths.shape or centers.ndim != 1:
        raise ValueError("centers and widths must be equal-length vectors")
    if np.any(widths <= 0):
        raise ValueError("triangle widths must be positive")
    lower = np.maximum(0.0, centers - widths)
    upper = np.minimum(1.0, centers + widths)
    raw_total = _triangle_primitive(upper, centers, widths) \
        - _triangle_primitive(lower, centers, widths)
    if np.any(raw_total <= 0):
        raise ValueError("triangles must intersect the unit interval")
    primitive = np.stack(
        [_triangle_primitive(np.full_like(centers, edge), centers, widths)
         for edge in edges],
        axis=0,
    )
    lower_primitive = _triangle_primitive(lower, centers, widths)
    cdf = np.clip((primitive - lower_primitive) / raw_total, 0.0, 1.0)
    masses = np.diff(cdf, axis=0)
    masses[masses < 0] = 0
    return masses


def rectangular_grid_bin_probabilities(
    edges: Sequence[np.ndarray],
    knots: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    dimension = len(knots)
    if len(edges) != dimension or weights.ndim != dimension:
        raise ValueError("grid dimensions are inconsistent")
    matrices = [hat_interval_masses(edges[axis], knots[axis]) for axis in range(dimension)]
    if dimension == 2:
        result = np.einsum("ai,ij,bj->ab", matrices[0], weights, matrices[1], optimize=True)
    elif dimension == 3:
        result = np.einsum(
            "ai,ijk,bj,ck->abc",
            matrices[0], weights, matrices[1], matrices[2],
            optimize=True,
        )
    else:
        raise ValueError("only two- and three-dimensional grids are supported")
    return _normalized_probabilities(result)


def point_cloud_bin_probabilities(
    edges: Sequence[np.ndarray],
    centers: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    centers = _matrix(centers)
    widths = _matrix(widths)
    weights = np.asarray(weights, dtype=np.float64)
    if centers.shape != widths.shape or centers.shape[0] != weights.size:
        raise ValueError("point-cloud arrays are inconsistent")
    dimension = centers.shape[1]
    matrices = [
        triangle_interval_masses(edges[axis], centers[:, axis], widths[:, axis])
        for axis in range(dimension)
    ]
    if dimension == 2:
        result = np.einsum(
            "am,bm,m->ab", matrices[0], matrices[1], weights, optimize=True,
        )
    elif dimension == 3:
        result = np.einsum(
            "am,bm,cm,m->abc",
            matrices[0], matrices[1], matrices[2], weights, optimize=True,
        )
    else:
        raise ValueError("only two- and three-dimensional clouds are supported")
    return _normalized_probabilities(result)


def sample_point_cloud(
    centers: np.ndarray,
    widths: np.ndarray,
    weights: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Draw a deterministic scrambled-Sobol sample from product triangles."""
    centers = _matrix(centers)
    widths = _matrix(widths)
    weights = np.asarray(weights, dtype=np.float64)
    if centers.shape != widths.shape or centers.shape[0] != weights.size:
        raise ValueError("point-cloud arrays are inconsistent")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    probabilities = _normalized_probabilities(weights.copy())
    dimension = centers.shape[1]
    exponent = int(math.ceil(math.log2(sample_count)))
    uniforms = qmc.Sobol(
        d=dimension + 1,
        scramble=True,
        seed=seed,
    ).random_base2(exponent)[:sample_count]
    component_cdf = np.cumsum(probabilities)
    component_cdf[-1] = 1.0
    selected = np.searchsorted(component_cdf, uniforms[:, 0], side="right")
    selected_centers = centers[selected]
    selected_widths = widths[selected]
    result = np.empty((sample_count, dimension), dtype=np.float64)
    for axis in range(dimension):
        center = selected_centers[:, axis]
        width = selected_widths[:, axis]
        lower = np.maximum(0.0, center - width)
        upper = np.minimum(1.0, center + width)
        lower_mass = _triangle_primitive(lower, center, width)
        upper_mass = _triangle_primitive(upper, center, width)
        raw_mass = lower_mass + uniforms[:, axis + 1] * (upper_mass - lower_mass)
        ascending = raw_mass <= width / 2.0
        coordinate = np.empty(sample_count, dtype=np.float64)
        coordinate[ascending] = center[ascending] - width[ascending] + np.sqrt(
            2.0 * width[ascending] * raw_mass[ascending],
        )
        descending_gap = np.maximum(
            0.0,
            2.0 * width[~ascending] * (width[~ascending] - raw_mass[~ascending]),
        )
        coordinate[~ascending] = center[~ascending] + width[~ascending] \
            - np.sqrt(descending_gap)
        result[:, axis] = np.clip(coordinate, lower, upper)
    return result


def sample_each_point_cloud_component(
    centers: np.ndarray,
    widths: np.ndarray,
    samples_per_component: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use shared Sobol quadrature draws inside every point-cloud component."""
    centers = _matrix(centers)
    widths = _matrix(widths)
    if centers.shape != widths.shape:
        raise ValueError("point-cloud arrays are inconsistent")
    if samples_per_component < 2 or samples_per_component & (samples_per_component - 1):
        raise ValueError("samples_per_component must be a power of two")
    count, dimension = centers.shape
    exponent = int(math.log2(samples_per_component))
    base_uniforms = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=seed,
    ).random_base2(exponent)
    selected_centers = np.repeat(centers, samples_per_component, axis=0)
    selected_widths = np.repeat(widths, samples_per_component, axis=0)
    uniforms = np.tile(base_uniforms, (count, 1))
    result = np.empty_like(selected_centers)
    for axis in range(dimension):
        center = selected_centers[:, axis]
        width = selected_widths[:, axis]
        lower = np.maximum(0.0, center - width)
        upper = np.minimum(1.0, center + width)
        lower_mass = _triangle_primitive(lower, center, width)
        upper_mass = _triangle_primitive(upper, center, width)
        raw_mass = lower_mass + uniforms[:, axis] * (upper_mass - lower_mass)
        ascending = raw_mass <= width / 2.0
        coordinate = np.empty(result.shape[0], dtype=np.float64)
        coordinate[ascending] = center[ascending] - width[ascending] + np.sqrt(
            2.0 * width[ascending] * raw_mass[ascending],
        )
        coordinate[~ascending] = center[~ascending] + width[~ascending] - np.sqrt(
            np.maximum(
                0.0,
                2.0 * width[~ascending] * (width[~ascending] - raw_mass[~ascending]),
            ),
        )
        result[:, axis] = np.clip(coordinate, lower, upper)
    return result, np.repeat(np.arange(count, dtype=np.int64), samples_per_component)


def select_nonredundant_point_cloud_centers(
    centers: np.ndarray,
    weights: np.ndarray,
    retained_count: int,
) -> np.ndarray:
    """Return indices that preserve mass and isolated coverage while pruning centers."""
    centers = _matrix(centers)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (centers.shape[0],):
        raise ValueError("point-cloud center weights are inconsistent")
    if not 1 <= retained_count <= centers.shape[0]:
        raise ValueError("retained center count is outside the available range")
    if retained_count == centers.shape[0]:
        return np.arange(centers.shape[0], dtype=np.int64)
    tree = cKDTree(centers)
    distances = tree.query(centers, k=2, workers=-1)[0][:, 1]
    positive = distances[distances > 0]
    distance_floor = (
        float(np.quantile(positive, 0.05)) * 0.1
        if positive.size else np.finfo(np.float64).eps
    )
    isolation = np.maximum(distances, distance_floor)
    normalized_weights = np.maximum(weights, np.finfo(np.float64).tiny)
    # Low-mass centers close to another center have the lowest first-order
    # removal cost. Isolated tail centers survive despite their low mass.
    removal_cost = normalized_weights * isolation * isolation
    retained = np.argpartition(removal_cost, -retained_count)[-retained_count:]
    return np.sort(retained.astype(np.int64))


def conditional_operation_metrics(
    reference_values: np.ndarray,
    approximation_values: np.ndarray,
    bins_by_context_dimension: Sequence[int] = (24, 12),
    approximation_weights: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Compare autoregressive conditional means and coordinate-wise medians.

    For target coordinate j, coordinates [0, j) define quantile conditioning
    cells. Cells and their evaluation weights always come from reference_values;
    approximation_values may be aligned reconstructions or independent model
    samples.
    """
    reference_values = _matrix(reference_values)
    approximation_values = _matrix(approximation_values)
    if reference_values.shape[1] != approximation_values.shape[1]:
        raise ValueError("reference and approximation dimensions differ")
    if approximation_weights is not None:
        approximation_weights = np.asarray(approximation_weights, dtype=np.float64)
        if approximation_weights.shape != (approximation_values.shape[0],):
            raise ValueError("approximation weights have the wrong shape")
        if np.any(approximation_weights < 0) or not np.sum(approximation_weights) > 0:
            raise ValueError("approximation weights must have positive total mass")
    dimension = reference_values.shape[1]
    if len(bins_by_context_dimension) < dimension - 1:
        raise ValueError("missing conditioning-bin counts")
    result: list[dict[str, float | int]] = []
    for target_axis in range(1, dimension):
        requested_bins = int(bins_by_context_dimension[target_axis - 1])
        edges = [
            _quantile_edges(reference_values[:, axis], requested_bins)
            for axis in range(target_axis)
        ]
        shape = tuple(edge.size - 1 for edge in edges)
        reference_cells = _cell_indices(reference_values[:, :target_axis], edges, shape)
        approximation_cells = _cell_indices(
            approximation_values[:, :target_axis], edges, shape,
        )
        cell_count = math.prod(shape)
        reference_counts = np.bincount(reference_cells, minlength=cell_count)
        approximation_counts = np.bincount(approximation_cells, minlength=cell_count)
        approximation_mass = np.bincount(
            approximation_cells,
            weights=approximation_weights,
            minlength=cell_count,
        ) if approximation_weights is not None else approximation_counts.astype(np.float64)
        included = (reference_counts > 0) & (approximation_mass > 0)
        truth_means, truth_medians = _grouped_mean_median(
            reference_cells,
            reference_values[:, target_axis],
            cell_count,
        )
        model_means, model_medians = _grouped_mean_median(
            approximation_cells,
            approximation_values[:, target_axis],
            cell_count,
            approximation_weights,
        )
        probabilities = reference_counts.astype(np.float64)
        probabilities /= np.sum(probabilities)
        covered_mass = float(np.sum(probabilities[included]))
        evaluation_weights = probabilities[included] / covered_mass
        mean_error = model_means - truth_means
        median_error = model_medians - truth_medians
        result.append({
            "targetAxis": target_axis,
            "contextDimensions": target_axis,
            "requestedBinsPerAxis": requested_bins,
            "conditioningCells": int(cell_count),
            "minimumReferenceCellObservations": int(np.min(reference_counts)),
            "minimumApproximationCellObservations": int(np.min(approximation_counts)),
            "coveredConditioningMass": covered_mass,
            "completeCoverage": bool(covered_mass >= 1.0 - 1e-12),
            "conditionalMeanRmseBps": float(np.sqrt(np.sum(
                evaluation_weights * mean_error[included] * mean_error[included],
            ))),
            "conditionalMedianMaeBps": float(np.sum(
                evaluation_weights * np.abs(median_error[included]),
            )),
            "maximumConditionalMeanAbsoluteErrorBps": float(np.max(np.abs(mean_error[included]))),
            "maximumConditionalMedianAbsoluteErrorBps": float(np.max(np.abs(median_error[included]))),
        })
    return result


def nearest_reconstruction_metrics(
    values: np.ndarray,
    decoded_centers: np.ndarray,
) -> dict[str, float]:
    values = _matrix(values)
    decoded_centers = _matrix(decoded_centers)
    if values.shape[1] != decoded_centers.shape[1]:
        raise ValueError("values and centers have different dimensions")
    tree = cKDTree(decoded_centers)
    _, indices = tree.query(values, k=1, workers=-1)
    residual = values - decoded_centers[indices]
    absolute = np.abs(residual)
    return {
        "rmseBpsPerCoordinate": float(np.sqrt(np.mean(residual * residual))),
        "maeBpsPerCoordinate": float(np.mean(absolute)),
        "p99AbsoluteErrorBps": float(np.quantile(absolute, 0.99)),
        "maximumAbsoluteErrorBps": float(np.max(absolute)),
    }


def grid_centers(knots: Sequence[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*knots, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)


def grid_component_return_means(
    knots: Sequence[np.ndarray],
    covariance: CovarianceTransform,
    transform: AsinhMatrixTransform,
    quadrature_order: int = 4,
) -> np.ndarray:
    centers = grid_centers(knots)
    lower_axes = []
    upper_axes = []
    for axis_knots in knots:
        lower_axes.append(np.concatenate((axis_knots[:1], axis_knots[:-1])))
        upper_axes.append(np.concatenate((axis_knots[1:], axis_knots[-1:])))
    lower_mesh = np.meshgrid(*lower_axes, indexing="ij")
    upper_mesh = np.meshgrid(*upper_axes, indexing="ij")
    lower = np.stack([axis.reshape(-1) for axis in lower_mesh], axis=1)
    upper = np.stack([axis.reshape(-1) for axis in upper_mesh], axis=1)
    return triangular_component_return_means(
        centers, lower, upper, covariance, transform, quadrature_order,
    )


def triangular_component_return_means(
    centers: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    covariance: CovarianceTransform,
    transform: AsinhMatrixTransform,
    quadrature_order: int = 4,
) -> np.ndarray:
    """Integrate decoded returns against normalized product-triangle components."""
    centers = _matrix(centers)
    lower = _matrix(lower)
    upper = _matrix(upper)
    if centers.shape != lower.shape or centers.shape != upper.shape:
        raise ValueError("triangle component bounds are inconsistent")
    if quadrature_order < 2:
        raise ValueError("quadrature order must be at least two")
    count, dimension = centers.shape
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    axis_nodes: list[np.ndarray] = []
    axis_weights: list[np.ndarray] = []
    for axis in range(dimension):
        coordinate_nodes = []
        coordinate_weights = []
        center = centers[:, axis]
        for side_lower, side_upper, ascending in (
            (lower[:, axis], center, True),
            (center, upper[:, axis], False),
        ):
            half = (side_upper - side_lower) / 2.0
            midpoint = (side_upper + side_lower) / 2.0
            positions = midpoint[:, None] + half[:, None] * nodes[None, :]
            side_width = side_upper - side_lower
            if ascending:
                height = np.divide(
                    positions - side_lower[:, None],
                    side_width[:, None],
                    out=np.zeros_like(positions),
                    where=side_width[:, None] > 0,
                )
            else:
                height = np.divide(
                    side_upper[:, None] - positions,
                    side_width[:, None],
                    out=np.zeros_like(positions),
                    where=side_width[:, None] > 0,
                )
            raw_weights = half[:, None] * weights[None, :] * height
            coordinate_nodes.append(positions)
            coordinate_weights.append(raw_weights)
        combined_nodes = np.concatenate(coordinate_nodes, axis=1)
        combined_weights = np.concatenate(coordinate_weights, axis=1)
        total = np.sum(combined_weights, axis=1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("triangle component has zero area")
        axis_nodes.append(combined_nodes)
        axis_weights.append(combined_weights / total)
    result = np.zeros((count, dimension), dtype=np.float64)
    choices = range(2 * quadrature_order)
    for selection in itertools.product(choices, repeat=dimension):
        unit = np.stack([
            axis_nodes[axis][:, selection[axis]] for axis in range(dimension)
        ], axis=1)
        mass = np.ones(count, dtype=np.float64)
        for axis in range(dimension):
            mass *= axis_weights[axis][:, selection[axis]]
        # Endpoint hats have a zero-area half whose formal quadrature nodes can
        # equal 0 or 1. Their product mass is zero, but the analytic logit is
        # intentionally defined only on the open strip.
        unit = np.clip(unit, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
        decoded = covariance.inverse(transform.inverse(unit))
        result += mass[:, None] * decoded
    return result


def probability_metrics(target: np.ndarray, model: np.ndarray) -> dict[str, float]:
    target = _normalized_probabilities(target)
    model = _normalized_probabilities(model)
    target_flat = target.reshape(-1)
    model_flat = np.maximum(model.reshape(-1), np.finfo(np.float64).tiny)
    positive = target_flat > 0
    midpoint = (target_flat + model_flat) / 2.0
    target_term = np.zeros_like(target_flat)
    target_term[positive] = target_flat[positive] * np.log2(
        target_flat[positive] / midpoint[positive],
    )
    model_positive = model_flat > 0
    model_term = np.zeros_like(model_flat)
    model_term[model_positive] = model_flat[model_positive] * np.log2(
        model_flat[model_positive] / midpoint[model_positive],
    )
    return {
        "jensenShannonBits": float(0.5 * np.sum(target_term + model_term)),
        "totalVariation": float(0.5 * np.sum(np.abs(target_flat - model_flat))),
    }


def _quantile_edges(values: np.ndarray, requested_bins: int) -> np.ndarray:
    if requested_bins < 2:
        raise ValueError("at least two conditioning bins are required")
    edges = np.unique(np.quantile(
        np.asarray(values, dtype=np.float64),
        np.linspace(0.0, 1.0, requested_bins + 1),
    ))
    if edges.size < 3:
        raise ValueError("conditioning coordinate has fewer than two distinct bins")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _cell_indices(
    values: np.ndarray,
    edges: Sequence[np.ndarray],
    shape: tuple[int, ...],
) -> np.ndarray:
    indices = []
    for axis, axis_edges in enumerate(edges):
        coordinate = np.searchsorted(axis_edges, values[:, axis], side="right") - 1
        indices.append(np.clip(coordinate, 0, axis_edges.size - 2))
    return np.ravel_multi_index(indices, shape)


def _grouped_mean_median(
    groups: np.ndarray,
    values: np.ndarray,
    group_count: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(
        groups,
        weights=weights,
        minlength=group_count,
    ) if weights is not None else np.bincount(groups, minlength=group_count)
    totals = np.bincount(
        groups,
        weights=values if weights is None else values * weights,
        minlength=group_count,
    )
    means = np.divide(
        totals,
        counts,
        out=np.full(group_count, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    medians = np.full(group_count, np.nan, dtype=np.float64)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    sorted_values = values[order]
    sorted_weights = None if weights is None else weights[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [groups.size]))
    for start, stop in zip(starts, stops, strict=True):
        group = int(sorted_groups[start])
        if sorted_weights is None:
            medians[group] = float(np.median(sorted_values[start:stop]))
        else:
            group_values = sorted_values[start:stop]
            group_weights = sorted_weights[start:stop]
            value_order = np.argsort(group_values, kind="stable")
            ordered_values = group_values[value_order]
            ordered_weights = group_weights[value_order]
            midpoint = 0.5 * np.sum(ordered_weights)
            median_index = int(np.searchsorted(
                np.cumsum(ordered_weights), midpoint, side="left",
            ))
            medians[group] = float(ordered_values[min(median_index, ordered_values.size - 1)])
    return means, medians


def _triangle_primitive(
    values: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    left = centers - widths
    right = centers + widths
    result = np.zeros(np.broadcast_shapes(values.shape, centers.shape), dtype=np.float64)
    values, centers, widths, left, right = np.broadcast_arrays(
        values, centers, widths, left, right,
    )
    ascending = (values > left) & (values <= centers)
    distance = values[ascending] - left[ascending]
    result[ascending] = distance * distance / (2.0 * widths[ascending])
    descending = (values > centers) & (values < right)
    distance = values[descending] - centers[descending]
    result[descending] = widths[descending] / 2.0 + distance \
        - distance * distance / (2.0 * widths[descending])
    result[values >= right] = widths[values >= right]
    return result


def _matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expected a row-major matrix")
    return values


def _normalized_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values[values < 0] = 0
    total = float(np.sum(values))
    if not total > 0:
        raise ValueError("probability array has no mass")
    return values / total
