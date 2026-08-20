"""Search density-tempered adaptive point clouds for consecutive BTC returns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

from multidimensional_return_knots import (
    AsinhMatrixTransform,
    CovarianceTransform,
    conditional_operation_metrics,
    point_cloud_bin_probabilities,
    probability_metrics,
    sample_each_point_cloud_component,
)
from return_knot_density import ReturnTransform, component_return_means
from trading_storage import read_candle_column


TransformFamily = Literal["scalar", "vector", "factorized"]
DEFAULT_PAIR_ANALYSIS = Path("data/benchmarks/consecutive-log-return-pairs.json")
DEFAULT_TRIPLE_ANALYSIS = Path("data/benchmarks/consecutive-log-return-triples.json")
DEFAULT_REFERENCE = Path("data/benchmarks/one-second-return-32-knot-fits.json")
DEFAULT_SAMPLE_CACHE = Path("data/runtime-cache/multidimensional-return-knot-samples-v1.npz")
DEFAULT_OUTPUT = Path("data/benchmarks/multidimensional-return-knot-search.json")
DEFAULT_REPORT = Path("docs/experiments/multidimensional-return-knot-search-2026-08-17.md")
CONDITIONAL_BINS = (24, 12)


@dataclass(frozen=True)
class SearchOptions:
    device: str
    transform_steps: int
    transform_sample: int
    cloud_sample: int
    conditional_draws: int
    train_stride: int
    validation_stride: int
    rebuild_cache: bool
    quick: bool
    minimum_component_draws: int = 16
    fast_point_cloud: bool = False
    compute_conditional_diagnostics: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search density-tempered multidimensional point-cloud return densities.",
    )
    parser.add_argument("--pair-analysis", type=Path, default=DEFAULT_PAIR_ANALYSIS)
    parser.add_argument("--triple-analysis", type=Path, default=DEFAULT_TRIPLE_ANALYSIS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sample-cache", type=Path, default=DEFAULT_SAMPLE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument("--validation-stride", type=int, default=32)
    parser.add_argument("--transform-sample", type=int, default=240_000)
    parser.add_argument("--cloud-sample", type=int, default=320_000)
    parser.add_argument("--conditional-draws", type=int, default=8_388_608)
    parser.add_argument("--transform-steps", type=int, default=260)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    pair_path = resolve(repo, args.pair_analysis)
    triple_path = resolve(repo, args.triple_analysis)
    reference_path = resolve(repo, args.reference)
    cache_path = resolve(repo, args.sample_cache)
    output_path = resolve(repo, args.output)
    report_path = resolve(repo, args.report)
    options = SearchOptions(
        device=args.device,
        transform_steps=80 if args.quick else args.transform_steps,
        transform_sample=min(args.transform_sample, 80_000) if args.quick else args.transform_sample,
        cloud_sample=min(args.cloud_sample, 100_000) if args.quick else args.cloud_sample,
        conditional_draws=min(args.conditional_draws, 131_072) if args.quick else args.conditional_draws,
        train_stride=args.train_stride,
        validation_stride=args.validation_stride,
        rebuild_cache=args.rebuild_cache,
        quick=args.quick,
    )
    pair_analysis = read_json(pair_path)
    triple_analysis = read_json(triple_path)
    validate_windows(pair_analysis, triple_analysis)
    samples, sample_metadata = load_or_build_samples(repo, pair_analysis, cache_path, options)
    reference = load_reference(reference_path)
    dimensions: dict[str, Any] = {}
    for dimension in (2, 3):
        print(f"\nSearching {dimension}D all-active point cloud...", flush=True)
        dimensions[str(dimension)] = search_dimension(
            dimension,
            samples[f"train{dimension}"].astype(np.float64),
            samples[f"validation{dimension}"].astype(np.float64),
            reference,
            options,
        )
    artifact: dict[str, Any] = {
        "version": 3,
        "generatedAt": utc_now(),
        "source": {
            "pairAnalysis": relative(repo, pair_path),
            "tripleAnalysis": relative(repo, triple_path),
            "oneDimensionalReference": relative(repo, reference_path),
        },
        "methodology": {
            "pipeline": "zero mask -> fitted covariance whitening -> asinh -> fitted matrix transform -> sigmoid -> density-tempered adaptive triangular-kernel point cloud",
            "zeroMasks": "Exact mask probabilities remain separate; this search fits only all-active continuous components.",
            "split": "The first four UTC years fit the representation; the final UTC year reports temporal drift. Acceptance measures in-sample representation error, matching the 1D reference experiment.",
            "densityCriterion": "Training histogram Jensen-Shannon divergence in bits per active coordinate must not exceed the direct JS-optimized 1D 32-knot reference.",
            "operationCriterion": "Every autoregressive conditional-mean RMSE and conditional-median MAE must not exceed the corresponding error produced by quantizing the target coordinate through the JS-optimized 1D 32-knot representation.",
            "conditionalQueries": "2D scores r2|r1. 3D scores both r2|r1 and r3|r1,r2. Empirical quantile cells use 24 bins for one conditioning coordinate and 12 per axis for two conditioning coordinates.",
            "cloud": "Mini-batch k-means centers, local product-triangular bandwidths, empirical component weights, and one low-weight full-support triangular background whose density vanishes at the strip boundary.",
            "beta": "Beta changes only center allocation in mapped unit space. Smoothed pilot-density weights proportional to p(u)^(beta-1) flatten the observations seen by k-means; final bandwidths and mixture weights are always refit to the untempered distribution. Beta is reselected from the incumbent and its neighboring candidates at each knot count.",
            "betaCandidates": {"2": beta_candidates(2), "3": beta_candidates(3)},
            "transformFamilies": ["shared scalar", "positive scaling vector", "factorized rotation-shear-positive-scale matrix"],
            "modelIntegration": f"Conditional operations target {options.conditional_draws:,} deterministic scrambled-Sobol quadrature draws, allocated equally within every component and then weighted by exact mixture mass.",
            "histogramResolution": {"2": 64, "3": 24},
        },
        "sample": sample_metadata,
        "reference1d": reference,
        "dimensions": dimensions,
    }
    artifact["finalModels"] = fit_final_models(dimensions, samples, reference, options)
    compact_search_parameters(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(artifact), encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


def search_dimension(
    dimension: int,
    train: np.ndarray,
    validation: np.ndarray,
    reference: dict[str, Any],
    options: SearchOptions,
) -> dict[str, Any]:
    covariance = CovarianceTransform.fit(train)
    whitened_train = covariance.forward(train)
    whitened_validation = covariance.forward(validation)
    reference_train = reference_conditional_operations(train, reference)
    reference_validation = reference_conditional_operations(validation, reference)
    print(
        f"  samples train={train.shape[0]:,}, validation={validation.shape[0]:,}; "
        f"1D32 conditional baselines {format_operations(reference_train)}",
        flush=True,
    )
    transforms: dict[str, Any] = {}
    screening: dict[str, Any] = {}
    transformed: dict[str, tuple[np.ndarray, np.ndarray, AsinhMatrixTransform]] = {}
    screen_count = 128 if options.quick else (384 if dimension == 2 else 512)
    incumbent_transform: dict[str, Any] | None = None
    for family in ("scalar", "vector", "factorized"):
        print(f"  fitting nested {family} transform on the true distribution...", flush=True)
        transform_fit = fit_tail_transform(
            whitened_train,
            whitened_validation,
            family,
            options,
            incumbent_transform,
        )
        incumbent_transform = transform_fit
        transform = AsinhMatrixTransform(
            np.asarray(transform_fit["matrix"], dtype=np.float64),
        )
        unit_train = transform.forward(whitened_train)
        unit_validation = transform.forward(whitened_validation)
        for beta in beta_candidates(dimension):
            key = transform_key(family, beta)
            print(
                f"  screening {family} allocation beta={beta:g} "
                f"with {screen_count:,} points...",
                flush=True,
            )
            cloud = fit_cloud(
                unit_train,
                unit_validation,
                screen_count,
                beta,
                transform,
                covariance,
                train,
                validation,
                reference_train,
                reference,
                options,
            )
            transforms[key] = {**transform_fit, "beta": beta}
            screening[key] = cloud
            transformed[key] = (unit_train, unit_validation, transform)
    selected_key = min(screening, key=lambda key: screening[key]["acceptanceScore"])
    selected = transforms[selected_key]
    print(
        f"  selected transform {selected['family']} beta={selected['beta']:g}; "
        f"score={screening[selected_key]['acceptanceScore']:.6g}",
        flush=True,
    )
    search = search_cloud_counts(
        candidate_cloud_counts(dimension, options.quick),
        transformed[selected_key],
        float(selected["beta"]),
        covariance,
        train,
        validation,
        reference_train,
        reference,
        options,
        screening[selected_key],
    )
    return {
        "dimension": dimension,
        "component": "all-active",
        "trainObservations": int(train.shape[0]),
        "validationObservations": int(validation.shape[0]),
        "covariance": covariance_payload(covariance),
        "referenceConditionalOperations": reference_train,
        "validationReferenceConditionalOperations": reference_validation,
        "transforms": transforms,
        "screeningKnotCount": screen_count,
        "screening": screening,
        "selectedTransform": {
            "key": selected_key,
            "family": selected["family"],
            "beta": selected["beta"],
        },
        "search": search,
    }


def fit_tail_transform(
    train: np.ndarray,
    validation: np.ndarray,
    family: TransformFamily,
    options: SearchOptions,
    initial: dict[str, Any] | None,
) -> dict[str, Any]:
    dimension = train.shape[1]
    selected = evenly_spaced_rows(train, options.transform_sample)
    device = torch.device(options.device)
    values = torch.as_tensor(np.arcsinh(selected), dtype=torch.float64, device=device)
    angle_count = dimension * (dimension - 1) // 2
    if initial is None:
        initial_scale = math.pi / math.sqrt(3.0) / max(
            float(np.mean(np.std(np.arcsinh(selected), axis=0))), 1e-6,
        )
        initial_logs = np.full(1 if family == "scalar" else dimension, math.log(initial_scale))
        initial_rotation = np.zeros(angle_count)
        initial_shear = np.zeros(angle_count)
    else:
        source_logs = np.asarray(initial["parameters"]["logScales"], dtype=np.float64)
        if family == "scalar":
            initial_logs = np.asarray([float(np.mean(source_logs))])
        elif source_logs.size == 1:
            initial_logs = np.repeat(source_logs, dimension)
        else:
            initial_logs = source_logs.copy()
        initial_rotation = np.asarray(
            initial["parameters"].get("rotation", np.zeros(angle_count)), dtype=np.float64,
        )
        initial_shear = np.asarray(
            initial["parameters"].get("shear", np.zeros(angle_count)), dtype=np.float64,
        )
    log_scales = torch.nn.Parameter(torch.as_tensor(initial_logs, dtype=torch.float64, device=device))
    parameters: list[torch.nn.Parameter] = [log_scales]
    rotation = None
    shear = None
    if family == "factorized":
        rotation = torch.nn.Parameter(torch.as_tensor(
            initial_rotation, dtype=torch.float64, device=device,
        ))
        shear = torch.nn.Parameter(torch.as_tensor(
            initial_shear, dtype=torch.float64, device=device,
        ))
        parameters.extend([rotation, shear])
    optimizer = torch.optim.Adam(parameters, lr=0.035)
    best = math.inf
    best_arrays: list[np.ndarray] = []
    for _ in range(options.transform_steps):
        optimizer.zero_grad(set_to_none=True)
        matrix = torch_transform_matrix(dimension, family, log_scales, rotation, shear)
        latent = values @ matrix.T
        energy = torch.sum(F.softplus(latent) + F.softplus(-latent), dim=1)
        loss = -torch.logdet(matrix) + torch.mean(energy)
        if family == "factorized":
            assert rotation is not None and shear is not None
            loss = loss + 2e-5 * (rotation.square().sum() + shear.square().sum())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best:
            best = value
            best_arrays = [parameter.detach().cpu().numpy().copy() for parameter in parameters]
    for parameter, best_value in zip(parameters, best_arrays, strict=True):
        parameter.data.copy_(torch.as_tensor(best_value, dtype=parameter.dtype, device=device))
    matrix = torch_transform_matrix(
        dimension, family, log_scales, rotation, shear,
    ).detach().cpu().numpy()
    result_parameters: dict[str, Any] = {
        "logScales": log_scales.detach().cpu().numpy().tolist(),
    }
    if family == "factorized":
        assert rotation is not None and shear is not None
        result_parameters["rotation"] = rotation.detach().cpu().numpy().tolist()
        result_parameters["shear"] = shear.detach().cpu().numpy().tolist()
    return {
        "family": family,
        "matrix": matrix.tolist(),
        "parameters": result_parameters,
        "uniformNllTrainPerObservation": transform_uniform_nll(train, matrix),
        "uniformNllValidationPerObservation": transform_uniform_nll(validation, matrix),
        "uniformObjective": best,
    }


def torch_transform_matrix(
    dimension: int,
    family: TransformFamily,
    log_scales: torch.Tensor,
    rotation: torch.Tensor | None,
    shear: torch.Tensor | None,
) -> torch.Tensor:
    identity = torch.eye(dimension, dtype=log_scales.dtype, device=log_scales.device)
    if family == "scalar":
        return identity * torch.exp(log_scales[0])
    diagonal = torch.diag(torch.exp(log_scales))
    if family == "vector":
        return diagonal
    if rotation is None or shear is None:
        raise AssertionError("factorized transform parameters are missing")
    skew = torch.zeros_like(identity)
    upper = torch.zeros_like(identity)
    index = 0
    for row in range(dimension):
        for column in range(row + 1, dimension):
            basis = torch.zeros_like(identity)
            basis[row, column] = 1.0
            skew = skew + rotation[index] * (basis - basis.T)
            upper = upper + shear[index] * basis
            index += 1
    return torch.matrix_exp(skew) @ (identity + upper) @ diagonal


def transform_uniform_nll(whitened: np.ndarray, matrix: np.ndarray) -> float:
    values = np.arcsinh(evenly_spaced_rows(whitened, 300_000))
    latent = values @ matrix.T
    _, log_determinant = np.linalg.slogdet(matrix)
    return float(
        -log_determinant
        + np.mean(np.sum(np.logaddexp(0.0, latent) + np.logaddexp(0.0, -latent), axis=1))
    )


def fast_weighted_lloyd(
    values: np.ndarray,
    sample_weights: np.ndarray,
    count: int,
    initial_centers: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit many low-dimensional centers with tree-based weighted Lloyd updates.

    The production joint search often has roughly two observations per center.
    Exact all-pairs mini-batch k-means is needlessly expensive in that regime;
    a cKDTree makes assignment nearly linearithmic in the center count. Full
    verification deliberately retains the original MiniBatchKMeans path.
    """
    values = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if values.ndim != 2 or sample_weights.shape != (values.shape[0],):
        raise ValueError("fast Lloyd inputs have inconsistent shapes")
    if not 1 <= count <= values.shape[0]:
        raise ValueError("fast Lloyd center count is outside the sample")
    if initial_centers is None:
        probabilities = np.maximum(sample_weights, 0.0)
        probabilities /= np.sum(probabilities)
        rng = np.random.default_rng(seed)
        selected = rng.choice(values.shape[0], size=count, replace=False, p=probabilities)
        centers = values[selected].copy()
        iterations = 10
        damping = 1.0
    else:
        centers = np.asarray(initial_centers, dtype=np.float64).copy()
        if centers.shape != (count, values.shape[1]):
            raise ValueError("fast Lloyd initial centers have the wrong shape")
        # A shallow, heavily damped warm update is cheap in isolation but
        # forces the outer fixed-point loop to repeat every covariance and
        # transform proposal while centers inch toward their centroids. Six
        # half-steps settle a transported/pruned cloud in the same refit while
        # retaining damping for assignment-boundary stability.
        iterations = 6
        damping = 0.5
    centers = np.clip(centers, 1e-8, 1.0 - 1e-8)
    previous_labels: np.ndarray | None = None
    for _ in range(iterations):
        distances, labels = cKDTree(centers).query(values, k=1, workers=-1)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        mass = np.bincount(labels, weights=sample_weights, minlength=count)
        updated = centers.copy()
        occupied = mass > np.finfo(np.float64).tiny
        for axis in range(values.shape[1]):
            totals = np.bincount(
                labels,
                weights=sample_weights * values[:, axis],
                minlength=count,
            )
            centroids = totals[occupied] / mass[occupied]
            updated[occupied, axis] += damping * (
                centroids - updated[occupied, axis]
            )
        empty = np.flatnonzero(~occupied)
        if empty.size:
            removal_score = sample_weights * np.square(distances)
            replacements = np.argpartition(removal_score, -empty.size)[-empty.size:]
            updated[empty] = values[replacements]
        previous_labels = labels
        centers = np.clip(updated, 1e-8, 1.0 - 1e-8)
    labels = cKDTree(centers).query(values, k=1, workers=-1)[1]
    return centers, labels.astype(np.int64, copy=False)


def fit_cloud(
    unit_train: np.ndarray,
    unit_validation: np.ndarray,
    count: int,
    beta: float,
    transform: AsinhMatrixTransform,
    covariance: CovarianceTransform,
    raw_train: np.ndarray,
    raw_validation: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    *,
    initial_adaptive_centers: np.ndarray | None = None,
    disabled_conditional_mean_targets: frozenset[int] = frozenset(),
    conditional_metrics_active: bool = True,
    enable_conditional_weight_calibration: bool = True,
    compute_conditional_diagnostics: bool = True,
    cloud_seed_offset: int = 0,
) -> dict[str, Any]:
    dimension = unit_train.shape[1]
    if not dimension / (dimension + 2.0) <= beta <= 1.0:
        raise ValueError("allocation beta lies outside its dimension-motivated bounds")
    adaptive_count = max(2, count - 1)
    selected = evenly_spaced_rows(unit_train, options.cloud_sample)
    allocation_weights, allocation = density_tempering_weights(selected, beta)
    if initial_adaptive_centers is not None:
        initial_adaptive_centers = np.asarray(initial_adaptive_centers, dtype=np.float64)
        if initial_adaptive_centers.shape != (adaptive_count, dimension):
            raise ValueError(
                "initial adaptive centers must match the requested cloud size and dimension",
            )
        initial_adaptive_centers = np.clip(initial_adaptive_centers, 1e-8, 1 - 1e-8)
    cloud_seed = 1701 + dimension + count + cloud_seed_offset
    if options.fast_point_cloud:
        adaptive_centers, labels = fast_weighted_lloyd(
            selected,
            allocation_weights,
            adaptive_count,
            initial_adaptive_centers,
            cloud_seed,
        )
    else:
        kmeans_init: str | np.ndarray = (
            "k-means++" if initial_adaptive_centers is None else initial_adaptive_centers
        )
        kmeans = MiniBatchKMeans(
            n_clusters=adaptive_count,
            init=kmeans_init,
            n_init=1,
            max_iter=40 if options.quick else 90,
            batch_size=min(16_384, selected.shape[0]),
            reassignment_ratio=0.002,
            random_state=cloud_seed,
        )
        kmeans.fit(selected, sample_weight=allocation_weights)
        labels = kmeans.predict(selected)
        adaptive_centers = np.clip(
            kmeans.cluster_centers_.astype(np.float64),
            1e-8,
            1 - 1e-8,
        )
    cluster_counts = np.bincount(labels, minlength=adaptive_count).astype(np.float64)
    residual = selected - adaptive_centers[labels]
    squared = np.zeros((adaptive_count, dimension), dtype=np.float64)
    for axis in range(dimension):
        np.add.at(squared[:, axis], labels, residual[:, axis] ** 2)
    standard_deviation = np.sqrt(squared / np.maximum(cluster_counts[:, None], 1.0))
    base_widths = np.maximum(
        math.sqrt(6.0) * standard_deviation,
        0.015 * adaptive_count ** (-1.0 / dimension),
    )
    histogram_bins = 64 if dimension == 2 else 24
    edges = [np.linspace(0.0, 1.0, histogram_bins + 1)] * dimension
    train_target = np.histogramdd(selected, bins=edges)[0]
    background_weight = min(0.002, 1.0 / max(count, 2))
    adaptive_weights = cluster_counts / np.sum(cluster_counts) * (1.0 - background_weight)
    weights = np.concatenate((adaptive_weights, [background_weight]))
    centers = np.vstack((adaptive_centers, np.full((1, dimension), 0.5)))
    best: tuple[float, float, np.ndarray] | None = None
    for scale in (0.85, 1.0, 1.2, 1.5, 1.9):
        adaptive_widths = np.clip(base_widths * scale, 1e-5, 2.0)
        # Width 0.5 reaches both strip boundaries but vanishes there. A wider,
        # nearly uniform background has positive boundary density; after the
        # inverse logit/sinh map that creates unnecessarily heavy return tails
        # and makes conditional means converge very slowly.
        widths = np.vstack((adaptive_widths, np.full((1, dimension), 0.5)))
        model = point_cloud_bin_probabilities(edges, centers, widths, weights)
        js = probability_metrics(train_target, model)["jensenShannonBits"]
        if best is None or js < best[0]:
            best = js, scale, widths
    if best is None:
        raise AssertionError("cloud bandwidth search failed")
    _, bandwidth_scale, widths = best
    density_weights = weights.copy()
    if enable_conditional_weight_calibration:
        calibrated_weights, calibration = calibrate_conditional_weights(
            centers,
            widths,
            weights,
            covariance,
            transform,
            raw_train,
            reference_operations,
            options,
            disabled_conditional_mean_targets,
        )
        blend_candidates: list[tuple[float, float, np.ndarray]] = []
        for blend in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            candidate_weights = (1.0 - blend) * density_weights + blend * calibrated_weights
            candidate_model = point_cloud_bin_probabilities(
                edges, centers, widths, candidate_weights,
            )
            candidate_js_per_dimension = probability_metrics(
                train_target, candidate_model,
            )["jensenShannonBits"] / dimension
            blend_candidates.append((candidate_js_per_dimension, blend, candidate_weights))
        js_threshold = float(reference["densityThresholds"]["jensenShannonBits"])
        density_eligible = [item for item in blend_candidates if item[0] <= js_threshold]
        if density_eligible:
            _, selected_blend, weights = max(density_eligible, key=lambda item: item[1])
        else:
            _, selected_blend, weights = min(blend_candidates, key=lambda item: item[0])
        calibration["selectedDensitySafeBlend"] = selected_blend
        calibration["calibratedWeightTotalVariation"] = float(
            0.5 * np.sum(np.abs(calibrated_weights - density_weights))
        )
        calibration["selectedWeightTotalVariation"] = float(
            0.5 * np.sum(np.abs(weights - density_weights))
        )
    else:
        weights = density_weights
        calibration = {
            "enabled": False,
            "reason": "Conditional operations are diagnostics under the JS-only objective.",
            "selectedDensitySafeBlend": 0.0,
            "calibratedWeightTotalVariation": 0.0,
            "selectedWeightTotalVariation": 0.0,
        }
    model = point_cloud_bin_probabilities(edges, centers, widths, weights)
    density = with_per_dimension(probability_metrics(train_target, model), dimension)
    validation_target = np.histogramdd(unit_validation, bins=edges)[0]
    validation_density = with_per_dimension(
        probability_metrics(validation_target, model), dimension,
    )
    if compute_conditional_diagnostics:
        evaluation_per_component = quadrature_samples_per_component(
            count,
            options.conditional_draws,
            maximum=2_048,
            minimum=options.minimum_component_draws,
        )
        unit_model, model_components = sample_each_point_cloud_component(
            centers,
            widths,
            seed=9109 + 101 * dimension + count,
            samples_per_component=evaluation_per_component,
        )
        unit_model = np.clip(
            unit_model,
            np.finfo(np.float64).eps,
            1.0 - np.finfo(np.float64).eps,
        )
        raw_model = covariance.inverse(transform.inverse(unit_model))
        model_sample_weights = weights[model_components] / evaluation_per_component
        operations = conditional_operation_metrics(
            raw_train,
            raw_model,
            CONDITIONAL_BINS,
            model_sample_weights,
        )
        validation_operations = conditional_operation_metrics(
            raw_validation,
            raw_model,
            CONDITIONAL_BINS,
            model_sample_weights,
        )
        operation_draws = int(unit_model.shape[0])
    else:
        evaluation_per_component = 0
        operation_draws = 0
        operations = []
        validation_operations = []
    acceptance = acceptance_metrics(
        density,
        operations,
        reference,
        reference_operations,
        disabled_conditional_mean_targets,
        conditional_metrics_active=conditional_metrics_active,
    )
    return {
        "knotCount": int(count),
        "beta": beta,
        "allocation": allocation,
        "centersUnit": centers.tolist(),
        "bandwidthsUnit": widths.tolist(),
        "componentWeights": weights.tolist(),
        "backgroundComponents": 1,
        "bandwidthScale": bandwidth_scale,
        "conditionalWeightCalibration": calibration,
        "operationQuadratureSamplesPerComponent": evaluation_per_component,
        "operationQuadratureDraws": operation_draws,
        "conditionalDiagnosticsComputed": compute_conditional_diagnostics,
        "validationHistogramBinsPerAxis": histogram_bins,
        "density": density,
        "validationDensity": validation_density,
        "conditionalOperations": operations,
        "validationConditionalOperations": validation_operations,
        "disabledConditionalMeanTargets": sorted(disabled_conditional_mean_targets),
        **acceptance,
    }


def density_tempering_weights(
    unit_values: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Estimate p(u) and return escort weights proportional to p(u)^(beta-1)."""
    unit_values = np.asarray(unit_values, dtype=np.float64)
    if unit_values.ndim != 2 or unit_values.shape[1] not in (2, 3):
        raise ValueError("density tempering supports only 2D and 3D matrices")
    dimension = unit_values.shape[1]
    lower = dimension / (dimension + 2.0)
    if not lower <= beta <= 1.0:
        raise ValueError("allocation beta lies outside its dimension-motivated bounds")
    if beta == 1.0:
        weights = np.ones(unit_values.shape[0], dtype=np.float64)
        return weights, {
            "histogramBinsPerAxis": 0,
            "gaussianSmoothingSigmaBins": 0.0,
            "effectiveSampleSize": float(weights.size),
            "effectiveSampleFraction": 1.0,
            "minimumWeight": 1.0,
            "maximumWeight": 1.0,
        }
    bins = 128 if dimension == 2 else 48
    histogram = np.histogramdd(
        unit_values,
        bins=[bins] * dimension,
        range=[(0.0, 1.0)] * dimension,
    )[0]
    smoothing_sigma = 1.25
    density = gaussian_filter(histogram, sigma=smoothing_sigma, mode="reflect")
    positive = density[density > 0]
    floor = max(float(np.quantile(positive, 0.01)) * 0.1, np.finfo(np.float64).tiny)
    density = np.maximum(density, floor)
    indices = np.clip(
        np.floor(unit_values * bins).astype(np.int64),
        0,
        bins - 1,
    )
    log_density = np.log(density[tuple(indices[:, axis] for axis in range(dimension))])
    log_weights = (beta - 1.0) * log_density
    # A tiny number of boundary cells should not dominate a center-allocation fit.
    log_weights = np.minimum(log_weights, np.quantile(log_weights, 0.999))
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.mean(weights)
    effective_sample_size = float(np.sum(weights) ** 2 / np.sum(weights * weights))
    return weights, {
        "histogramBinsPerAxis": bins,
        "gaussianSmoothingSigmaBins": smoothing_sigma,
        "effectiveSampleSize": effective_sample_size,
        "effectiveSampleFraction": effective_sample_size / weights.size,
        "minimumWeight": float(np.min(weights)),
        "maximumWeight": float(np.max(weights)),
    }


def calibrate_conditional_weights(
    centers: np.ndarray,
    widths: np.ndarray,
    initial_weights: np.ndarray,
    covariance: CovarianceTransform,
    transform: AsinhMatrixTransform,
    raw_train: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    options: SearchOptions,
    disabled_conditional_mean_targets: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Calibrate mixture weights to the conditional operations used for acceptance."""
    dimension = raw_train.shape[1]
    target_draws = 262_144 if options.quick else 1_048_576
    maximum_per_component = 256 if options.quick else 512
    available = max(options.minimum_component_draws, target_draws // centers.shape[0])
    samples_per_component = min(
        maximum_per_component,
        1 << int(math.floor(math.log2(available))),
    )
    unit_samples, component_ids = sample_each_point_cloud_component(
        centers,
        widths,
        samples_per_component,
        seed=7229 + centers.shape[0] + 31 * dimension,
    )
    unit_samples = np.clip(
        unit_samples,
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    raw_samples = covariance.inverse(transform.inverse(unit_samples))
    count = centers.shape[0]
    calibration_rows: list[dict[str, np.ndarray | float]] = []
    for target_axis, baseline in enumerate(reference_operations, start=1):
        requested_bins = CONDITIONAL_BINS[target_axis - 1]
        edges = [
            conditional_quantile_edges(raw_train[:, axis], requested_bins)
            for axis in range(target_axis)
        ]
        shape = tuple(edge.size - 1 for edge in edges)
        cell_count = math.prod(shape)
        empirical_cells = conditional_cell_indices(raw_train[:, :target_axis], edges, shape)
        sample_cells = conditional_cell_indices(raw_samples[:, :target_axis], edges, shape)
        empirical_counts = np.bincount(empirical_cells, minlength=cell_count).astype(np.float64)
        empirical_probability = empirical_counts / np.sum(empirical_counts)
        empirical_mean = np.bincount(
            empirical_cells,
            weights=raw_train[:, target_axis],
            minlength=cell_count,
        ) / empirical_counts
        empirical_median = np.asarray([
            np.median(raw_train[empirical_cells == cell, target_axis])
            for cell in range(cell_count)
        ])
        flat = sample_cells * count + component_ids
        component_mass = np.bincount(
            flat, minlength=cell_count * count,
        ).reshape(cell_count, count) / samples_per_component
        component_first = np.bincount(
            flat,
            weights=raw_samples[:, target_axis],
            minlength=cell_count * count,
        ).reshape(cell_count, count) / samples_per_component
        below = raw_samples[:, target_axis] <= empirical_median[sample_cells]
        component_below = np.bincount(
            flat,
            weights=below.astype(np.float64),
            minlength=cell_count * count,
        ).reshape(cell_count, count) / samples_per_component
        median_tolerance = float(baseline["conditionalMedianMaeBps"])
        empirical_near_median = np.asarray([
            np.mean(np.abs(raw_train[empirical_cells == cell, target_axis] - empirical_median[cell])
                    <= median_tolerance)
            for cell in range(cell_count)
        ])
        probability_tolerance = np.maximum(empirical_near_median / 2.0, 2e-3)
        calibration_rows.append({
            "targetAxis": float(target_axis),
            "mass": component_mass,
            "first": component_first,
            "below": component_below,
            "probability": empirical_probability,
            "mean": empirical_mean,
            "meanTolerance": float(baseline["conditionalMeanRmseBps"]),
            "medianProbabilityTolerance": probability_tolerance,
        })
    device = torch.device(options.device)
    dtype = torch.float64
    initial = np.maximum(np.asarray(initial_weights, dtype=np.float64), 1e-15)
    initial /= np.sum(initial)
    logits = torch.nn.Parameter(torch.as_tensor(np.log(initial), dtype=dtype, device=device))
    initial_tensor = torch.as_tensor(initial, dtype=dtype, device=device)
    rows = [
        {
            key: torch.as_tensor(value, dtype=dtype, device=device)
            if isinstance(value, np.ndarray) else value
            for key, value in row.items()
        }
        for row in calibration_rows
    ]
    optimizer = torch.optim.Adam([logits], lr=0.05)
    best_loss = math.inf
    best_weights = initial.copy()
    steps = 80 if options.quick else 220
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        loss = torch.zeros((), dtype=dtype, device=device)
        for row in rows:
            mass = row["mass"] @ weights
            safe_mass = torch.clamp(mass, min=1e-12)
            model_mean = (row["first"] @ weights) / safe_mass
            model_below = (row["below"] @ weights) / safe_mass
            probability = row["probability"]
            mean_error = (model_mean - row["mean"]) / row["meanTolerance"]
            median_error = (model_below - 0.5) / row["medianProbabilityTolerance"]
            supported = (mass > 1e-10).to(dtype)
            if int(row["targetAxis"]) not in disabled_conditional_mean_targets:
                loss = loss + torch.sum(probability * supported * mean_error.square())
            loss = loss + torch.sum(probability * supported * median_error.square())
            loss = loss + 1_000.0 * torch.sum(probability * (1.0 - supported))
        relative = torch.log(torch.clamp(weights, min=1e-300)) \
            - torch.log(torch.clamp(initial_tensor, min=1e-300))
        loss = loss + 0.5 * torch.sum(weights * relative)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 20.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_weights = weights.detach().cpu().numpy().copy()
    total_variation = float(0.5 * np.sum(np.abs(best_weights - initial)))
    relative = np.log(np.maximum(best_weights, 1e-300) / initial)
    return best_weights, {
        "samplesPerComponent": samples_per_component,
        "optimizationSteps": steps,
        "objective": best_loss,
        "weightTotalVariationFromDensityFit": total_variation,
        "weightKlFromDensityFitNats": float(np.sum(best_weights * relative)),
    }


def quadrature_samples_per_component(
    component_count: int,
    target_draws: int,
    maximum: int,
    minimum: int = 16,
) -> int:
    if minimum < 1 or maximum < minimum:
        raise ValueError("component quadrature limits are inconsistent")
    available = max(minimum, target_draws // component_count)
    return min(maximum, 1 << int(math.floor(math.log2(available))))


def conditional_quantile_edges(values: np.ndarray, requested_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, requested_bins + 1)))
    if edges.size < 3:
        raise ValueError("conditioning coordinate has fewer than two distinct bins")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def conditional_cell_indices(
    values: np.ndarray,
    edges: Sequence[np.ndarray],
    shape: tuple[int, ...],
) -> np.ndarray:
    indices = []
    for axis, axis_edges in enumerate(edges):
        coordinate = np.searchsorted(axis_edges, values[:, axis], side="right") - 1
        indices.append(np.clip(coordinate, 0, axis_edges.size - 2))
    return np.ravel_multi_index(indices, shape)


def search_cloud_counts(
    candidates: list[int],
    transformed: tuple[np.ndarray, np.ndarray, AsinhMatrixTransform],
    beta: float,
    covariance: CovarianceTransform,
    raw_train: np.ndarray,
    raw_validation: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    screened: dict[str, Any],
) -> dict[str, Any]:
    unit_train, unit_validation, transform = transformed
    fits: list[dict[str, Any]] = []
    current_beta = beta
    dimension = unit_train.shape[1]
    for count in candidates:
        if count == screened["knotCount"]:
            fit = screened
        else:
            trials: list[dict[str, Any]] = []
            for candidate_beta in neighboring_beta_candidates(dimension, current_beta):
                print(
                    f"  point-cloud count search {count:,} knots, "
                    f"allocation beta={candidate_beta:g}...",
                    flush=True,
                )
                trials.append(fit_cloud(
                    unit_train,
                    unit_validation,
                    count,
                    candidate_beta,
                    transform,
                    covariance,
                    raw_train,
                    raw_validation,
                    reference_operations,
                    reference,
                    options,
                ))
            fit = min(trials, key=lambda candidate: candidate["acceptanceScore"])
            fit["allocationBetaTrials"] = [
                {
                    "beta": candidate["beta"],
                    "acceptanceScore": candidate["acceptanceScore"],
                    "passes": candidate["passes"],
                }
                for candidate in trials
            ]
        current_beta = float(fit["beta"])
        fits.append(fit)
        if enough_confirmation(fits):
            break
    passing = [fit for fit in fits if fit["passes"]]
    smallest = min(passing, key=lambda fit: fit["knotCount"], default=None)
    best = min(fits, key=lambda fit: fit["acceptanceScore"])
    confirmed = next((
        fits[index - 1]["knotCount"]
        for index in range(1, len(fits))
        if fits[index - 1]["passes"] and fits[index]["passes"]
    ), None)
    return {
        "candidates": fits,
        "smallestPassingKnotCount": None if smallest is None else smallest["knotCount"],
        "confirmedPassingKnotCount": confirmed,
        "bestTestedKnotCount": best["knotCount"],
        "bestAcceptanceScore": best["acceptanceScore"],
    }


def fit_final_models(
    dimensions: dict[str, Any],
    samples: dict[str, np.ndarray],
    reference: dict[str, Any],
    options: SearchOptions,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    thinning = max(1, options.train_stride // options.validation_stride)
    for dimension in (2, 3):
        train = samples[f"train{dimension}"].astype(np.float64)
        validation = samples[f"validation{dimension}"].astype(np.float64)
        full = np.vstack((train, validation[::thinning]))
        covariance = CovarianceTransform.fit(full)
        whitened = covariance.forward(full)
        reference_operations = reference_conditional_operations(full, reference)
        selected = dimensions[str(dimension)]["selectedTransform"]
        family = selected["family"]
        search = dimensions[str(dimension)]["search"]
        selected_count = search["smallestPassingKnotCount"] or search["bestTestedKnotCount"]
        source_fit = next(
            candidate for candidate in search["candidates"]
            if candidate["knotCount"] == selected_count
        )
        beta = float(source_fit["beta"])
        print(
            f"Final {dimension}D: fitting {family} transform on five years; "
            f"starting allocation beta={beta:g}...",
            flush=True,
        )
        transform_fit: dict[str, Any] | None = None
        for nested_family in ("scalar", "vector", "factorized"):
            transform_fit = fit_tail_transform(
                whitened,
                whitened,
                nested_family,
                options,
                transform_fit,
            )
            if nested_family == family:
                break
        if transform_fit is None:
            raise AssertionError("nested transform fit did not produce a result")
        transform = AsinhMatrixTransform(np.asarray(transform_fit["matrix"], dtype=np.float64))
        unit = transform.forward(whitened)
        remaining = [
            count for count in candidate_cloud_counts(dimension, options.quick)
            if count >= selected_count
        ]
        fit = None
        current_beta = beta
        for count in remaining:
            trials: list[dict[str, Any]] = []
            for candidate_beta in neighboring_beta_candidates(dimension, current_beta):
                print(
                    f"Final {dimension}D point cloud at {count:,} knots, "
                    f"allocation beta={candidate_beta:g}...",
                    flush=True,
                )
                trials.append(fit_cloud(
                    unit,
                    unit,
                    count,
                    candidate_beta,
                    transform,
                    covariance,
                    full,
                    full,
                    reference_operations,
                    reference,
                    options,
                ))
            candidate = min(trials, key=lambda trial: trial["acceptanceScore"])
            current_beta = float(candidate["beta"])
            if fit is None or candidate["acceptanceScore"] < fit["acceptanceScore"]:
                fit = candidate
            if candidate["passes"]:
                fit = candidate
                break
        if fit is None:
            raise AssertionError("final point-cloud search did not produce a fit")
        result[str(dimension)] = {
            "observations": int(full.shape[0]),
            "validationThinning": thinning,
            "referenceConditionalOperations": reference_operations,
            "transformFamily": family,
            "beta": float(fit["beta"]),
            "covariance": covariance_payload(covariance),
            "postAsinhTransform": transform_fit,
            "fit": fit,
        }
    return result


def acceptance_metrics(
    density: dict[str, float],
    operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    reference_operations: list[dict[str, float | int]],
    disabled_conditional_mean_targets: frozenset[int] = frozenset(),
    *,
    conditional_metrics_active: bool = True,
) -> dict[str, Any]:
    density_ratio = density["jensenShannonBitsPerDimension"] \
        / reference["densityThresholds"]["jensenShannonBits"]
    query_ratios: list[dict[str, float | int | bool]] = []
    ratios = [density_ratio]
    ranking_ratios = [density_ratio]
    operation_pairs = (
        zip(operations, reference_operations, strict=True)
        if conditional_metrics_active or operations
        else ()
    )
    for operation, baseline in operation_pairs:
        target_axis = int(operation["targetAxis"])
        mean_ratio = float(operation["conditionalMeanRmseBps"]) \
            / max(float(baseline["conditionalMeanRmseBps"]), 1e-15)
        median_ratio = float(operation["conditionalMedianMaeBps"]) \
            / max(float(baseline["conditionalMedianMaeBps"]), 1e-15)
        coverage = float(operation["coveredConditioningMass"])
        coverage_ratio = 1.0 if coverage >= 1.0 - 1e-12 else 1_000_000.0
        mean_enabled = (
            conditional_metrics_active
            and target_axis not in disabled_conditional_mean_targets
        )
        median_enabled = conditional_metrics_active
        coverage_enabled = conditional_metrics_active
        if mean_enabled:
            ratios.append(mean_ratio)
            ranking_ratios.append(mean_ratio)
        if median_enabled:
            ratios.append(median_ratio)
            ranking_ratios.append(median_ratio)
        if coverage_enabled:
            ratios.append(coverage_ratio)
            if coverage_ratio > 1.0:
                ranking_ratios.append(coverage_ratio)
        query_ratios.append({
            "targetAxis": target_axis,
            "conditionalMeanRatioTo1d32": mean_ratio,
            "conditionalMeanEnabled": mean_enabled,
            "conditionalMedianRatioTo1d32": median_ratio,
            "conditionalMedianEnabled": median_enabled,
            "coveredConditioningMass": coverage,
            "coverageEnabled": coverage_enabled,
            "passes": (
                (not mean_enabled or mean_ratio <= 1.0)
                and (not median_enabled or median_ratio <= 1.0)
                and (not coverage_enabled or coverage_ratio <= 1.0)
            ),
        })
    score = max(ratios)
    ranking_score = float(
        np.mean(np.power(np.asarray(ranking_ratios, dtype=np.float64), 8.0)) ** (1.0 / 8.0)
    )
    return {
        "densityJsRatioTo1d32": density_ratio,
        "conditionalRatios": query_ratios,
        "acceptanceScore": score,
        "jointObjective": ranking_score,
        "disabledConditionalMeanTargets": sorted(disabled_conditional_mean_targets),
        "conditionalMetricsActive": conditional_metrics_active,
        "passes": bool(score <= 1.0),
    }


def reference_conditional_operations(
    values: np.ndarray,
    reference: dict[str, Any],
) -> list[dict[str, float | int]]:
    decoded = values.copy()
    for target_axis in range(1, values.shape[1]):
        decoded[:, target_axis] = decode_reference_returns(values[:, target_axis], reference)
    return conditional_operation_metrics(values, decoded, CONDITIONAL_BINS)


def decode_reference_returns(values: np.ndarray, reference: dict[str, Any]) -> np.ndarray:
    transform = reference["transform"]
    unit = sigmoid(
        transform["alpha"] * np.arcsinh(
            (values - transform["locationBps"]) / transform["scaleBps"],
        ),
    )
    knots = np.asarray(reference["knotsUnit"], dtype=np.float64)
    right = np.clip(np.searchsorted(knots, unit, side="left"), 1, knots.size - 1)
    left = right - 1
    selected = np.where(
        np.abs(unit - knots[right]) < np.abs(unit - knots[left]), right, left,
    )
    return np.asarray(reference["componentMeansBps"], dtype=np.float64)[selected]


def load_reference(path: Path) -> dict[str, Any]:
    artifact = read_json(path)
    fit = artifact["fits"]["js"]
    transform = ReturnTransform(
        alpha=float(artifact["transform"]["alpha"]),
        location_bps=float(artifact["transform"]["locationBps"]),
        scale_bps=float(artifact["transform"]["scaleBps"]),
    )
    knots = np.asarray(fit["knotsUnit"], dtype=np.float64)
    means_bps = component_return_means(knots, transform) * 10_000.0
    return {
        "fit": "direct JS-optimized",
        "knotCount": len(knots),
        "knotsUnit": fit["knotsUnit"],
        "componentMeansBps": means_bps.tolist(),
        "transform": artifact["transform"],
        "densityThresholds": {
            "jensenShannonBits": float(fit["metrics"]["jsBits"]),
        },
    }


def beta_candidates(dimension: int) -> list[float]:
    if dimension == 2:
        return [0.5, 0.625, 0.75, 0.875, 1.0]
    if dimension == 3:
        return [0.6, 0.7, 0.8, 0.9, 1.0]
    raise ValueError("only 2D and 3D searches are supported")


def neighboring_beta_candidates(dimension: int, incumbent: float) -> list[float]:
    candidates = beta_candidates(dimension)
    index = min(range(len(candidates)), key=lambda item: abs(candidates[item] - incumbent))
    lower = max(0, index - 1)
    upper = min(len(candidates), index + 2)
    return candidates[lower:upper]


def candidate_cloud_counts(dimension: int, quick: bool) -> list[int]:
    if quick:
        return [64, 128, 256, 512]
    if dimension == 2:
        return [32, 64, 96, 144, 256, 384, 576, 768, 1024, 1536, 2048, 3072, 4096, 6144]
    return [
        64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
        6144, 8192, 12_288, 16_384, 24_576, 32_768,
    ]


def transform_key(family: str, beta: float) -> str:
    return f"{family}-beta-{beta:g}"


def enough_confirmation(fits: list[dict[str, Any]]) -> bool:
    return len(fits) >= 2 and fits[-1]["passes"] and fits[-2]["passes"]


def with_per_dimension(metrics: dict[str, float], dimension: int) -> dict[str, float]:
    return {
        **metrics,
        "jensenShannonBitsPerDimension": metrics["jensenShannonBits"] / dimension,
    }


def covariance_payload(covariance: CovarianceTransform) -> dict[str, Any]:
    return {
        "centerBps": covariance.center.tolist(),
        "matrixBpsSquared": covariance.covariance.tolist(),
        "symmetricWhitening": covariance.whitening.tolist(),
    }


def compact_search_parameters(artifact: dict[str, Any]) -> None:
    heavy = {"centersUnit", "bandwidthsUnit", "componentWeights"}
    for dimension in artifact["dimensions"].values():
        for fit in dimension["screening"].values():
            for key in heavy:
                fit.pop(key, None)
        for fit in dimension["search"]["candidates"]:
            for key in heavy:
                fit.pop(key, None)


def render_report(artifact: dict[str, Any]) -> str:
    reference = artifact["reference1d"]
    lines = [
        "# Density-tempered multidimensional one-second return point clouds",
        "",
        f"Generated {artifact['generatedAt']}.",
        "",
        "## Acceptance criteria",
        "",
        f"The direct JS-optimized 1D 32-knot reference has JS "
        f"{reference['densityThresholds']['jensenShannonBits']:.9g} bits. The joint fit must "
        "not exceed this per active coordinate. KL is not calculated.",
        "",
        "The point cloud must also beat the 1D 32-knot quantizer on every measured conditional "
        "mean and conditional median operation. Exact zero masks remain separate mixture components.",
        "",
        "Beta affects only knot allocation after the invertible mapping. K-means observations are "
        "weighted by the smoothed mapped density raised to beta minus one. Smaller beta spends "
        "relatively fewer centers in the concentrated middle while leaving the mapping and tails "
        "geometrically unchanged. Final bandwidths and weights target the true distribution.",
        "",
        "## Exact zero-mask split",
        "",
        "| dimension | observations | all zero | any zero | all active |",
        "|---:|---:|---:|---:|---:|",
    ]
    mask_counts = artifact["sample"]["zeroMaskCounts"]
    for dimension in (2, 3):
        counts = np.asarray(mask_counts[f"train{dimension}"], dtype=np.int64) \
            + np.asarray(mask_counts[f"validation{dimension}"], dtype=np.int64)
        total = int(np.sum(counts))
        lines.append(
            f"| {dimension} | {total:,} | {counts[0] / total:.4%} | "
            f"{(total - counts[-1]) / total:.4%} | {counts[-1] / total:.4%} |"
        )
    lines.append("")
    for dimension_text, result in artifact["dimensions"].items():
        dimension = int(dimension_text)
        lines.extend([
            f"## {dimension}D all-active component",
            "",
            f"Training sample: {result['trainObservations']:,}; final-year sample: "
            f"{result['validationObservations']:,}.",
            "",
            "### 1D 32-knot conditional-operation baselines",
            "",
            "| target | conditioning coordinates/cells | mean RMSE bps | median MAE bps |",
            "|---|---:|---:|---:|",
        ])
        for operation in result["referenceConditionalOperations"]:
            lines.append(operation_row(operation))
        lines.extend([
            "",
            f"### Transform and allocation-beta screening at {result['screeningKnotCount']:,} points",
            "",
            "| family | beta | allocation ESS | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio |",
            "|---|---:|---:|---:|---|---|---:|",
        ])
        ordered = sorted(
            result["transforms"],
            key=lambda key: (result["transforms"][key]["family"], result["transforms"][key]["beta"]),
        )
        for key in ordered:
            transform = result["transforms"][key]
            fit = result["screening"][key]
            lines.append(
                f"| {transform['family']} | {transform['beta']:.3g} | "
                f"{fit['allocation']['effectiveSampleFraction']:.3%} | "
                f"{fit['density']['jensenShannonBitsPerDimension']:.8g} | "
                f"{operation_values(fit['conditionalOperations'], 'conditionalMeanRmseBps')} | "
                f"{operation_values(fit['conditionalOperations'], 'conditionalMedianMaeBps')} | "
                f"{fit['acceptanceScore']:.6g} |"
            )
        selected = result["selectedTransform"]
        lines.extend([
            "",
            f"Selected: **{selected['family']}**, beta **{selected['beta']:.3g}**.",
            "",
            "### Knot-count search",
            "",
            "| points | beta | fit JS/d | final-year JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |",
            "|---:|---:|---:|---:|---|---|---:|---:|",
        ])
        for fit in result["search"]["candidates"]:
            lines.append(
                f"| {fit['knotCount']:,} | {fit['beta']:.3g} | "
                f"{fit['density']['jensenShannonBitsPerDimension']:.8g} | "
                f"{fit['validationDensity']['jensenShannonBitsPerDimension']:.8g} | "
                f"{operation_values(fit['conditionalOperations'], 'conditionalMeanRmseBps')} | "
                f"{operation_values(fit['conditionalOperations'], 'conditionalMedianMaeBps')} | "
                f"{fit['acceptanceScore']:.6g} | {'yes' if fit['passes'] else 'no'} |"
            )
        minimum = result["search"]["smallestPassingKnotCount"]
        candidates = result["search"]["candidates"]
        confirmed = next((
            candidates[index - 1]["knotCount"]
            for index in range(1, len(candidates))
            if candidates[index - 1]["passes"] and candidates[index]["passes"]
        ), None)
        lines.extend([
            "",
            f"Smallest tested passing point cloud: **{minimum if minimum is not None else 'none'}**. "
            f"First passing count confirmed by the next tested size: **{confirmed if confirmed is not None else 'none'}**. "
            f"Best tested count: **{result['search']['bestTestedKnotCount']}** with maximum "
            f"normalized error {result['search']['bestAcceptanceScore']:.6g}.",
            "",
        ])
        final = artifact["finalModels"][dimension_text]
        fit = final["fit"]
        lines.extend([
            "### Complete-five-year refit",
            "",
            "| transform | beta | points | JS/d | conditional mean RMSEs | conditional median MAEs | max ratio | passes |",
            "|---|---:|---:|---:|---|---|---:|---:|",
            f"| {final['transformFamily']} | {final['beta']:.3g} | {fit['knotCount']:,} | "
            f"{fit['density']['jensenShannonBitsPerDimension']:.8g} | "
            f"{operation_values(fit['conditionalOperations'], 'conditionalMeanRmseBps')} | "
            f"{operation_values(fit['conditionalOperations'], 'conditionalMedianMaeBps')} | "
            f"{fit['acceptanceScore']:.6g} | {'yes' if fit['passes'] else 'no'} |",
            "",
            f"Post-asinh matrix: `{json.dumps(final['postAsinhTransform']['matrix'], separators=(',', ':'))}`",
            "",
        ])
    two = artifact["dimensions"]["2"]
    three = artifact["dimensions"]["3"]
    final_two = artifact["finalModels"]["2"]["fit"]
    final_three = artifact["finalModels"]["3"]["fit"]
    selected_two_count = two["search"]["smallestPassingKnotCount"] \
        or two["search"]["bestTestedKnotCount"]
    selected_three_count = three["search"]["smallestPassingKnotCount"] \
        or three["search"]["bestTestedKnotCount"]
    selected_two_fit = next(
        fit for fit in two["search"]["candidates"]
        if fit["knotCount"] == selected_two_count
    )
    selected_three_fit = next(
        fit for fit in three["search"]["candidates"]
        if fit["knotCount"] == selected_three_count
    )
    two_minimum = two["search"]["smallestPassingKnotCount"]
    two_summary = (
        f"the smallest tested 2D pass is {two_minimum:,} points"
        if two_minimum is not None
        else "no tested 2D count passes"
    )
    lines.extend([
        "## Conclusions",
        "",
        f"- In the four-year representation fit, {two_summary}. The first pass confirmed by "
        f"the next tested size is {two['search']['confirmedPassingKnotCount'] or 'none'}.",
        f"- No tested 3D count passes; the best is {three['search']['bestTestedKnotCount']:,} "
        f"points at {three['search']['bestAcceptanceScore']:.6g}x the strictest 1D32 threshold.",
        f"- The complete-five-year 2D refit {'passes' if final_two['passes'] else 'does not pass'} "
        f"at {final_two['knotCount']:,} points with {strictest_metric(final_two)}.",
        f"- The complete-five-year 3D refit {'passes' if final_three['passes'] else 'does not pass'} "
        f"at {final_three['knotCount']:,} points with {strictest_metric(final_three)}.",
        f"- Screening / selected-count / complete-five-year allocation betas are "
        f"{two['selectedTransform']['beta']:.3g} / {selected_two_fit['beta']:.3g} / "
        f"{artifact['finalModels']['2']['beta']:.3g} in 2D and "
        f"{three['selectedTransform']['beta']:.3g} / {selected_three_fit['beta']:.3g} / "
        f"{artifact['finalModels']['3']['beta']:.3g} in 3D. Beta does not participate in the "
        "forward or inverse mapping; it only changes finite point placement.",
        "",
        "## Interpretation",
        "",
        "- Fit JS measures representation error. Final-year JS and conditional-operation errors "
        "also contain temporal distribution drift and are reported but do not select knot count.",
        "- Conditional statistics are calculated from the continuous triangular mixture, not from "
        "nearest-center reconstructions.",
        "- Counts are the smallest tested passing configurations in the recorded search bracket, "
        "not proofs over every omitted integer.",
        "- Point-cloud fits are not nested across counts. A higher count can score worse because "
        "k-means initialization, local bandwidths, and operation-aware weight calibration are refit.",
        "- When no complete-five-year candidate passes, the final section retains the tested "
        "candidate with the lowest maximum normalized error rather than relabeling it as passing.",
        "",
        "## Reproduction",
        "",
        "```text",
        "npm run analysis:return-multidimensional-knots",
        "```",
        "",
        "The machine artifact retains covariance and transform parameters plus the final centers, "
        "bandwidths, and weights.",
        "",
    ])
    return "\n".join(lines)


def operation_row(operation: dict[str, float | int]) -> str:
    target = int(operation["targetAxis"]) + 1
    context = ",".join(f"r{axis + 1}" for axis in range(int(operation["contextDimensions"])))
    return (
        f"| r{target} | {context} ({int(operation['conditioningCells'])}) | "
        f"{float(operation['conditionalMeanRmseBps']):.8g} | "
        f"{float(operation['conditionalMedianMaeBps']):.8g} |"
    )


def operation_values(operations: list[dict[str, Any]], field: str) -> str:
    return " / ".join(f"{float(operation[field]):.7g}" for operation in operations)


def strictest_metric(fit: dict[str, Any]) -> str:
    candidates = [(float(fit["densityJsRatioTo1d32"]), "density JS")]
    for query in fit["conditionalRatios"]:
        target = int(query["targetAxis"]) + 1
        candidates.extend((
            (float(query["conditionalMeanRatioTo1d32"]), f"r{target} conditional-mean RMSE"),
            (float(query["conditionalMedianRatioTo1d32"]), f"r{target} conditional-median MAE"),
        ))
    ratio, name = max(candidates)
    return f"{name} ({ratio:.6g}x)"


def format_operations(operations: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"r{int(operation['targetAxis']) + 1}: mean={float(operation['conditionalMeanRmseBps']):.5g}, "
        f"median={float(operation['conditionalMedianMaeBps']):.5g} bps"
        for operation in operations
    )


def load_or_build_samples(
    repo: Path,
    analysis: dict[str, Any],
    cache_path: Path,
    options: SearchOptions,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    full = analysis["scales"][0]["fullHistory"]
    start = parse_iso(full["startTime"])
    end = parse_iso(full["endTime"])
    train_end = start.replace(year=start.year + 4)
    metadata_key = np.asarray([
        start.timestamp(), end.timestamp(), train_end.timestamp(),
        options.train_stride, options.validation_stride, 1,
    ], dtype=np.float64)
    if cache_path.exists() and not options.rebuild_cache:
        cached = np.load(cache_path)
        if np.array_equal(cached["metadataKey"], metadata_key):
            arrays = {
                key: cached[key]
                for key in ("train2", "validation2", "train3", "validation3")
            }
            metadata = json.loads(str(cached["metadataJson"][0]))
            return arrays, metadata
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    selected = sorted(
        file for file in source.glob("????-??-??.json")
        if start <= datetime.fromisoformat(file.stem).replace(tzinfo=timezone.utc) < end
    )
    if not selected:
        raise RuntimeError("no 1s candle references cover the analysis window")
    prior_candidates = sorted(file for file in source.glob("????-??-??.json") if file < selected[0])
    if not prior_candidates:
        raise RuntimeError("a pre-window close is required")
    previous_close = float(read_candle_column(prior_candidates[-1], "close")[-1])
    buffers: dict[str, list[np.ndarray]] = {
        "train2": [], "validation2": [], "train3": [], "validation3": [],
    }
    mask_counts = {
        "train2": np.zeros(4, dtype=np.int64),
        "validation2": np.zeros(4, dtype=np.int64),
        "train3": np.zeros(8, dtype=np.int64),
        "validation3": np.zeros(8, dtype=np.int64),
    }
    tails = {2: np.empty(0, dtype=np.float64), 3: np.empty(0, dtype=np.float64)}
    for day_index, reference in enumerate(selected):
        if day_index % 50 == 0:
            print(f"Reading multidimensional samples {day_index}/{len(selected)}...", flush=True)
        day = datetime.fromisoformat(reference.stem).replace(tzinfo=timezone.utc)
        closes = read_candle_column(reference, "close").astype(np.float64, copy=False)
        prior = np.empty(closes.size + 1, dtype=np.float64)
        prior[0] = previous_close
        prior[1:] = closes
        returns = np.log(prior[1:] / prior[:-1]) * 10_000.0
        previous_close = float(closes[-1])
        split = "train" if day < train_end else "validation"
        stride = options.train_stride if split == "train" else options.validation_stride
        for dimension in (2, 3):
            combined = np.concatenate((tails[dimension], returns))
            if combined.size >= dimension:
                windows = np.lib.stride_tricks.sliding_window_view(combined, dimension)
                code = np.zeros(windows.shape[0], dtype=np.int64)
                for axis in range(dimension):
                    code |= (windows[:, axis] != 0.0).astype(np.int64) << axis
                key = f"{split}{dimension}"
                mask_counts[key] += np.bincount(code, minlength=1 << dimension)
                eligible = np.flatnonzero(code == (1 << dimension) - 1)
                offset = ((day_index + 1) * (37 + 2 * dimension)) % stride
                chosen = eligible[offset::stride]
                if chosen.size:
                    buffers[key].append(windows[chosen].astype(np.float32))
            tails[dimension] = combined[-(dimension - 1):].copy()
    arrays = {
        key: np.concatenate(parts, axis=0) if parts else np.empty((0, int(key[-1])), dtype=np.float32)
        for key, parts in buffers.items()
    }
    metadata = {
        "startTime": start.isoformat().replace("+00:00", "Z"),
        "trainEndTime": train_end.isoformat().replace("+00:00", "Z"),
        "endTime": end.isoformat().replace("+00:00", "Z"),
        "trainStride": options.train_stride,
        "validationStride": options.validation_stride,
        "sampleCounts": {key: int(value.shape[0]) for key, value in arrays.items()},
        "zeroMaskCounts": {key: value.tolist() for key, value in mask_counts.items()},
        "maskBitOrder": "least-significant bit is the earliest return; 1 means nonzero",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        metadataKey=metadata_key,
        metadataJson=np.asarray([json.dumps(metadata)]),
        **arrays,
    )
    return arrays, metadata


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def evenly_spaced_rows(values: np.ndarray, limit: int) -> np.ndarray:
    if values.shape[0] <= limit:
        return values
    indices = np.linspace(0, values.shape[0] - 1, limit, dtype=np.int64)
    return values[indices]


def validate_windows(pair: dict[str, Any], triple: dict[str, Any]) -> None:
    pair_window = pair["scales"][0]["fullHistory"]
    triple_window = triple["scales"][0]["fullHistory"]
    if pair_window["startTime"] != triple_window["startTime"] \
            or pair_window["endTime"] != triple_window["endTime"]:
        raise ValueError("pair and triple analyses cover different windows")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


if __name__ == "__main__":
    main()
