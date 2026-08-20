"""Jointly optimize multidimensional return transforms and adaptive point clouds."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.linalg import expm

from multidimensional_return_knots import (
    AsinhMatrixTransform,
    CovarianceTransform,
    conditional_operation_metrics,
    point_cloud_bin_probabilities,
    probability_metrics,
    sample_each_point_cloud_component,
    select_nonredundant_point_cloud_centers,
)
from search_multidimensional_return_knots import (
    DEFAULT_PAIR_ANALYSIS,
    DEFAULT_REFERENCE,
    DEFAULT_SAMPLE_CACHE,
    DEFAULT_TRIPLE_ANALYSIS,
    CONDITIONAL_BINS,
    SearchOptions,
    acceptance_metrics,
    fit_cloud,
    evenly_spaced_rows,
    load_or_build_samples,
    load_reference,
    read_json,
    reference_conditional_operations,
    quadrature_samples_per_component,
    validate_windows,
    with_per_dimension,
)


DEFAULT_INITIAL = Path("data/benchmarks/multidimensional-return-knot-search.json")
DEFAULT_OUTPUT = Path("data/benchmarks/multidimensional-return-knot-joint-js-only.json")
DEFAULT_REPORT = Path(
    "docs/experiments/multidimensional-return-knot-joint-js-only-2026-08-18.md",
)
DISABLED_CONDITIONAL_MEAN_TARGETS = frozenset()
CONDITIONAL_METRICS_ACTIVE = False


@dataclass(frozen=True)
class JointState:
    empirical_covariance: CovarianceTransform
    covariance_log_shape: np.ndarray
    covariance: CovarianceTransform
    transform: AsinhMatrixTransform
    beta: float
    fit: dict[str, Any]


@dataclass(frozen=True)
class JointOptions:
    optimization: SearchOptions
    verification: SearchOptions
    maximum_sweeps_per_count: int
    maximum_prune_attempts: int
    quick: bool


def write_partial_checkpoint(
    path: Path,
    dimensions: dict[str, Any],
    in_progress: dict[str, Any] | None,
) -> None:
    """Atomically persist completed dimensions and the current prune boundary."""
    payload: dict[str, Any] = {
        "version": 2,
        "generatedAt": utc_now(),
        "complete": False,
        "dimensions": dimensions,
    }
    if in_progress is not None:
        payload["inProgress"] = in_progress
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly optimize covariance, transform, beta, and point-cloud knots.",
    )
    parser.add_argument("--pair-analysis", type=Path, default=DEFAULT_PAIR_ANALYSIS)
    parser.add_argument("--triple-analysis", type=Path, default=DEFAULT_TRIPLE_ANALYSIS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sample-cache", type=Path, default=DEFAULT_SAMPLE_CACHE)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--maximum-sweeps-per-count", type=int, default=16)
    parser.add_argument("--maximum-prune-attempts", type=int, default=30)
    parser.add_argument("--dimensions", type=int, nargs="+", choices=(2, 3), default=(2, 3))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed dimensions and in-progress pruning from the partial checkpoint.",
    )
    parser.add_argument(
        "--preserve-existing-dimensions",
        action="store_true",
        help="Keep dimensions in the output artifact that are not selected by --dimensions.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    pair_path = resolve(repo, args.pair_analysis)
    triple_path = resolve(repo, args.triple_analysis)
    reference_path = resolve(repo, args.reference)
    cache_path = resolve(repo, args.sample_cache)
    initial_path = resolve(repo, args.initial)
    output_path = resolve(repo, args.output)
    report_path = resolve(repo, args.report)
    pair_analysis = read_json(pair_path)
    triple_analysis = read_json(triple_path)
    validate_windows(pair_analysis, triple_analysis)
    options = build_options(args)
    samples, sample_metadata = load_or_build_samples(
        repo,
        pair_analysis,
        cache_path,
        options.verification,
    )
    reference = load_reference(reference_path)
    initial_artifact = read_json(initial_path)
    checkpoint_path = output_path.with_name(f"{output_path.stem}-partial.json")
    checkpoint = (
        read_json(checkpoint_path)
        if args.resume and checkpoint_path.exists()
        else {}
    )
    if checkpoint:
        dimensions: dict[str, Any] = deepcopy(checkpoint.get("dimensions", {}))
    elif args.preserve_existing_dimensions and output_path.exists():
        existing_artifact = read_json(output_path)
        selected_dimensions = set(args.dimensions)
        dimensions = {
            key: deepcopy(value)
            for key, value in existing_artifact.get("dimensions", {}).items()
            if int(key) not in selected_dimensions
        }
    else:
        dimensions = {}
    in_progress = checkpoint.get("inProgress")
    for dimension in args.dimensions:
        if str(dimension) in dimensions:
            print(f"\nSkipping completed {dimension}D result from checkpoint.", flush=True)
            continue
        print(f"\nJointly optimizing {dimension}D point cloud...", flush=True)
        resume_dimension = (
            in_progress
            if isinstance(in_progress, dict)
            and int(in_progress.get("dimension", -1)) == dimension
            else None
        )

        def save_progress(progress: dict[str, Any]) -> None:
            write_partial_checkpoint(checkpoint_path, dimensions, progress)

        dimensions[str(dimension)] = optimize_dimension(
            dimension,
            samples,
            reference,
            initial_artifact,
            options,
            resume_dimension=resume_dimension,
            progress_callback=save_progress,
        )
        in_progress = None
        write_partial_checkpoint(checkpoint_path, dimensions, None)
    artifact = {
        "version": 1,
        "generatedAt": utc_now(),
        "source": {
            "initialFit": relative(repo, initial_path),
            "pairAnalysis": relative(repo, pair_path),
            "tripleAnalysis": relative(repo, triple_path),
            "oneDimensionalReference": relative(repo, reference_path),
        },
        "methodology": {
            "objective": "Density Jensen-Shannon divergence per dimension relative to the direct JS-optimized 1D 32-knot reference; ratio <= 1 is required for acceptance.",
            "disabledLoss": "All conditional mean, median, and coverage measurements are diagnostics only and are excluded from calibration, optimization, and acceptance.",
            "covariance": "The empirical symmetric whitening is adjusted by exp(G), where G is symmetric and trace zero; this changes anisotropic covariance shape without introducing a scale gauge.",
            "transform": "The post-asinh matrix is updated by invertible exponential rotation, shear, and scale perturbations and retains an exact factorized representation.",
            "beta": "Beta remains a center-allocation parameter only and is bounded by d/(d+2) <= beta <= 1.",
            "knots": "Every parameter proposal warm-starts k-means from transported centers, then refits bandwidths and true-distribution mixture weights.",
            "searchAcceleration": "Search-fidelity cloud proposals use cKDTree weighted Lloyd updates, cached knot-stationarity, and per-prune resumable checkpoints. Final verification retains the original MiniBatchKMeans path and high-draw audit.",
            "pruning": "Low removal-cost centers are proposed first using mixture mass times squared nearest-center distance; each accepted prune is followed by another round-robin update.",
            "convergence": "At each knot count, a sweep tests every covariance-shape and post-asinh factor direction, beta, and a knot refit. Step sizes are refined twice; convergence requires a full minimum-resolution sweep with no accepted update.",
            "verification": f"Finalists use up to {options.verification.conditional_draws:,} deterministic component-stratified Sobol draws.",
        },
        "sample": sample_metadata,
        "reference1d": reference,
        "dimensions": dimensions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(artifact), encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"\nWrote {output_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


def build_options(args: argparse.Namespace) -> JointOptions:
    if args.maximum_sweeps_per_count < 1 or args.maximum_prune_attempts < 1:
        raise ValueError("joint iteration counts must be positive")
    optimization = SearchOptions(
        device=args.device,
        transform_steps=0,
        transform_sample=0,
        cloud_sample=40_000 if args.quick else 65_536,
        conditional_draws=131_072 if args.quick else 524_288,
        train_stride=128,
        validation_stride=32,
        rebuild_cache=False,
        quick=True,
        minimum_component_draws=16,
        fast_point_cloud=True,
        compute_conditional_diagnostics=False,
    )
    verification = SearchOptions(
        device=args.device,
        transform_steps=0,
        transform_sample=0,
        cloud_sample=100_000 if args.quick else 320_000,
        conditional_draws=262_144 if args.quick else 8_388_608,
        train_stride=128,
        validation_stride=32,
        rebuild_cache=False,
        quick=args.quick,
        minimum_component_draws=16,
        fast_point_cloud=False,
        compute_conditional_diagnostics=True,
    )
    return JointOptions(
        optimization=optimization,
        verification=verification,
        maximum_sweeps_per_count=(
            1 if args.quick else args.maximum_sweeps_per_count
        ),
        maximum_prune_attempts=(
            min(3, args.maximum_prune_attempts)
            if args.quick else args.maximum_prune_attempts
        ),
        quick=args.quick,
    )


def optimize_dimension(
    dimension: int,
    samples: dict[str, np.ndarray],
    reference: dict[str, Any],
    initial_artifact: dict[str, Any],
    options: JointOptions,
    resume_dimension: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    train = samples[f"train{dimension}"].astype(np.float64)
    validation = samples[f"validation{dimension}"].astype(np.float64)
    thinning = max(1, options.verification.train_stride // options.verification.validation_stride)
    raw = np.vstack((train, validation[::thinning]))
    reference_operations = reference_conditional_operations(raw, reference)
    initial = initial_state(dimension, initial_artifact, reference, reference_operations)
    print(
        f"  observations={raw.shape[0]:,}; initial points={initial.fit['knotCount']:,}, "
        f"beta={initial.beta:g}, active score={initial.fit['acceptanceScore']:.6g}, "
        f"objective={initial.fit['jointObjective']:.6g}",
        flush=True,
    )
    pruning_resume: dict[str, Any] | None = None
    if resume_dimension is not None:
        if resume_dimension.get("phase") != "pruning":
            raise ValueError("unsupported joint-optimization checkpoint phase")
        history = deepcopy(resume_dimension["prefixHistory"])
        convergence_events = deepcopy(resume_dimension["prefixConvergenceEvents"])
        state = joint_state_from_payload(resume_dimension["initialConvergedState"])
        accepted_states = [initial]
        if state.fit["passes"]:
            accepted_states.append(state)
        pruning_resume = deepcopy(resume_dimension["pruning"])
        resumed_state = joint_state_from_payload(pruning_resume["state"])
        print(
            f"  resuming prune attempt {int(pruning_resume['nextAttempt']) + 1} "
            f"from {int(resumed_state.fit['knotCount']):,} points...",
            flush=True,
        )
    else:
        history = [state_summary(initial, "stored verified initial")]
        state = evaluate_state(
            initial,
            initial.empirical_covariance,
            initial.covariance_log_shape,
            initial.covariance,
            initial.transform,
            initial.beta,
            int(initial.fit["knotCount"]),
            raw,
            reference_operations,
            reference,
            options.optimization,
            seed_offset=1,
        )
        history.append(state_summary(state, "common-fidelity warm refit"))
        accepted_states = [initial]
        if state.fit["passes"]:
            accepted_states.append(state)
        state, updates, initial_convergence = converge_state(
            state,
            raw,
            reference_operations,
            reference,
            options,
            direction_offset=0,
        )
        history.extend(updates)
        convergence_events = [initial_convergence]
        if state.fit["passes"]:
            accepted_states.append(state)

    prefix_history = deepcopy(history)
    prefix_convergence = deepcopy(convergence_events)
    initial_converged_payload = joint_state_payload(state)

    def save_pruning_progress(pruning: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        progress_callback({
            "dimension": dimension,
            "phase": "pruning",
            "prefixHistory": prefix_history,
            "prefixConvergenceEvents": prefix_convergence,
            "initialConvergedState": initial_converged_payload,
            "pruning": pruning,
        })

    state, prune_history, pruned_states, prune_convergence = prune_until_boundary(
        state,
        raw,
        reference_operations,
        reference,
        options,
        resume=pruning_resume,
        progress_callback=save_pruning_progress,
    )
    history.extend(prune_history)
    accepted_states.extend(pruned_states)
    convergence_events.extend(prune_convergence)
    finalists = unique_states_by_count([state, *reversed(accepted_states)])
    verified: list[dict[str, Any]] = []
    final_state: JointState | None = None
    for finalist_index, finalist in enumerate(finalists):
        print(
            f"  full verification at {finalist.fit['knotCount']:,} points, "
            f"beta={finalist.beta:g}...",
            flush=True,
        )
        candidate = evaluate_state(
            finalist,
            finalist.empirical_covariance,
            finalist.covariance_log_shape,
            finalist.covariance,
            finalist.transform,
            finalist.beta,
            int(finalist.fit["knotCount"]),
            raw,
            reference_operations,
            reference,
            options.verification,
            seed_offset=10_000 + finalist_index,
        )
        verified.append(state_summary(candidate, "full verification"))
        if final_state is None or candidate.fit["jointObjective"] < final_state.fit["jointObjective"]:
            final_state = candidate
        if candidate.fit["passes"]:
            final_state = candidate
            break
    if final_state is None:
        raise AssertionError("joint optimization produced no verifiable state")
    covariance_adjustment = expm(final_state.covariance_log_shape)
    return {
        "dimension": dimension,
        "observations": int(raw.shape[0]),
        "disabledConditionalMeanTargets": sorted(DISABLED_CONDITIONAL_MEAN_TARGETS),
        "referenceConditionalOperations": reference_operations,
        "initial": state_summary(initial, "stored verified initial"),
        "history": history,
        "verificationCandidates": verified,
        "convergenceEvents": convergence_events,
        "final": {
            "knotCount": int(final_state.fit["knotCount"]),
            "beta": final_state.beta,
            "passes": final_state.fit["passes"],
            "covariance": covariance_payload(final_state.covariance),
            "covarianceAdjustmentFromEmpirical": covariance_adjustment.tolist(),
            "postAsinhMatrix": final_state.transform.matrix.tolist(),
            "postAsinhFactorization": factorize_matrix(final_state.transform.matrix),
            "fit": final_state.fit,
        },
    }


def initial_state(
    dimension: int,
    artifact: dict[str, Any],
    reference: dict[str, Any],
    reference_operations: list[dict[str, float | int]],
) -> JointState:
    payload = artifact["finalModels"][str(dimension)]
    covariance_payload_data = payload["covariance"]
    whitening = np.asarray(covariance_payload_data["symmetricWhitening"], dtype=np.float64)
    coloring = np.linalg.inv(whitening)
    covariance = CovarianceTransform(
        center=np.asarray(covariance_payload_data["centerBps"], dtype=np.float64),
        whitening=whitening,
        coloring=coloring,
        covariance=coloring @ coloring.T,
    )
    transform = AsinhMatrixTransform(
        np.asarray(payload["postAsinhTransform"]["matrix"], dtype=np.float64),
    )
    fit = deepcopy(payload["fit"])
    fit.update(acceptance_metrics(
        fit["density"],
        fit["conditionalOperations"],
        reference,
        reference_operations,
        DISABLED_CONDITIONAL_MEAN_TARGETS,
        conditional_metrics_active=CONDITIONAL_METRICS_ACTIVE,
    ))
    return JointState(
        covariance,
        np.zeros((dimension, dimension), dtype=np.float64),
        covariance,
        transform,
        float(payload["beta"]),
        fit,
    )


def evaluate_state(
    source: JointState,
    empirical_covariance: CovarianceTransform,
    covariance_log_shape: np.ndarray,
    covariance: CovarianceTransform,
    transform: AsinhMatrixTransform,
    beta: float,
    count: int,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    seed_offset: int,
    warm_start: bool = True,
) -> JointState:
    initial_centers = (
        transported_pruned_centers(source, covariance, transform, count - 1)
        if warm_start else None
    )
    unit = transform.forward(covariance.forward(raw))
    fit = fit_cloud(
        unit,
        unit,
        count,
        beta,
        transform,
        covariance,
        raw,
        raw,
        reference_operations,
        reference,
        options,
        initial_adaptive_centers=initial_centers,
        disabled_conditional_mean_targets=DISABLED_CONDITIONAL_MEAN_TARGETS,
        conditional_metrics_active=CONDITIONAL_METRICS_ACTIVE,
        enable_conditional_weight_calibration=CONDITIONAL_METRICS_ACTIVE,
        compute_conditional_diagnostics=options.compute_conditional_diagnostics,
        cloud_seed_offset=seed_offset,
    )
    return JointState(
        empirical_covariance,
        covariance_log_shape,
        covariance,
        transform,
        beta,
        fit,
    )


def evaluate_fixed_cloud(
    source: JointState,
    empirical_covariance: CovarianceTransform,
    covariance_log_shape: np.ndarray,
    covariance: CovarianceTransform,
    transform: AsinhMatrixTransform,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    seed_offset: int,
) -> JointState:
    """Evaluate a transform proposal while holding every cloud parameter fixed."""
    centers = np.asarray(source.fit["centersUnit"], dtype=np.float64)
    widths = np.asarray(source.fit["bandwidthsUnit"], dtype=np.float64)
    weights = np.asarray(source.fit["componentWeights"], dtype=np.float64)
    count, dimension = centers.shape
    histogram_bins = 64 if dimension == 2 else 24
    edges = [np.linspace(0.0, 1.0, histogram_bins + 1)] * dimension
    unit_target = transform.forward(covariance.forward(raw))
    selected_target = evenly_spaced_rows(unit_target, options.cloud_sample)
    target = np.histogramdd(selected_target, bins=edges)[0]
    model = point_cloud_bin_probabilities(edges, centers, widths, weights)
    density = with_per_dimension(probability_metrics(target, model), dimension)
    if CONDITIONAL_METRICS_ACTIVE:
        samples_per_component = quadrature_samples_per_component(
            count,
            options.conditional_draws,
            maximum=256 if options.quick else 2_048,
            minimum=options.minimum_component_draws,
        )
        unit_model, component_ids = sample_each_point_cloud_component(
            centers,
            widths,
            samples_per_component,
            seed=9_109 + 101 * dimension + count,
        )
        unit_model = np.clip(
            unit_model,
            np.finfo(np.float64).eps,
            1.0 - np.finfo(np.float64).eps,
        )
        raw_model = covariance.inverse(transform.inverse(unit_model))
        sample_weights = weights[component_ids] / samples_per_component
        operations = conditional_operation_metrics(
            raw,
            raw_model,
            CONDITIONAL_BINS,
            sample_weights,
        )
        operation_draws = int(unit_model.shape[0])
    else:
        samples_per_component = 0
        operation_draws = 0
        operations = []
    acceptance = acceptance_metrics(
        density,
        operations,
        reference,
        reference_operations,
        DISABLED_CONDITIONAL_MEAN_TARGETS,
        conditional_metrics_active=CONDITIONAL_METRICS_ACTIVE,
    )
    fit = deepcopy(source.fit)
    fit.update({
        "density": density,
        "validationDensity": density,
        "conditionalOperations": operations,
        "validationConditionalOperations": operations,
        "operationQuadratureSamplesPerComponent": samples_per_component,
        "operationQuadratureDraws": operation_draws,
        "conditionalDiagnosticsComputed": bool(CONDITIONAL_METRICS_ACTIVE),
        **acceptance,
    })
    return JointState(
        empirical_covariance,
        covariance_log_shape,
        covariance,
        transform,
        source.beta,
        fit,
    )


def transported_pruned_centers(
    source: JointState,
    covariance: CovarianceTransform,
    transform: AsinhMatrixTransform,
    retained_count: int,
) -> np.ndarray:
    old_centers = np.asarray(source.fit["centersUnit"], dtype=np.float64)[:-1]
    old_weights = np.asarray(source.fit["componentWeights"], dtype=np.float64)[:-1]
    if retained_count > old_centers.shape[0]:
        raise ValueError("joint continuation only supports equal or lower knot counts")
    if retained_count < old_centers.shape[0]:
        retained = select_nonredundant_point_cloud_centers(
            old_centers,
            old_weights,
            retained_count,
        )
        old_centers = old_centers[retained]
    clipped = np.clip(old_centers, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    raw_centers = source.covariance.inverse(source.transform.inverse(clipped))
    return transform.forward(covariance.forward(raw_centers))


def converge_state(
    state: JointState,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: JointOptions,
    direction_offset: int,
) -> tuple[JointState, list[dict[str, Any]], dict[str, Any]]:
    dimension = raw.shape[1]
    coverage_sweeps = 1
    covariance_step = 0.07 if dimension == 2 else 0.05
    transform_step = 0.06 if dimension == 2 else 0.045
    beta_step = 0.125 if dimension == 2 else 0.1
    minimum_covariance_step = covariance_step / 4.0
    minimum_transform_step = transform_step / 4.0
    minimum_beta_step = beta_step / 4.0
    no_update_sweeps = 0
    history: list[dict[str, Any]] = []
    converged = False
    sweeps_completed = 0
    knots_stationary = False
    for local_sweep in range(options.maximum_sweeps_per_count):
        sweep_index = direction_offset + local_sweep
        state, updates, knots_stationary = round_robin(
            state,
            raw,
            reference_operations,
            reference,
            options.optimization,
            sweep_index,
            covariance_step,
            transform_step,
            beta_step,
            knots_stationary,
        )
        history.extend(updates)
        sweeps_completed += 1
        accepted = any(bool(row.get("acceptedUpdate")) for row in updates)
        no_update_sweeps = 0 if accepted else no_update_sweeps + 1
        if no_update_sweeps < coverage_sweeps:
            continue
        at_minimum_resolution = (
            covariance_step <= minimum_covariance_step * (1.0 + 1e-12)
            and transform_step <= minimum_transform_step * (1.0 + 1e-12)
            and beta_step <= minimum_beta_step * (1.0 + 1e-12)
        )
        if at_minimum_resolution:
            converged = True
            history.append(state_summary(state, "fixed point reached"))
            break
        covariance_step = max(minimum_covariance_step, covariance_step / 2.0)
        transform_step = max(minimum_transform_step, transform_step / 2.0)
        beta_step = max(minimum_beta_step, beta_step / 2.0)
        no_update_sweeps = 0
        history.append(state_summary(state, "refined parameter-search resolution"))
    if not converged:
        history.append(state_summary(state, "fixed-point safety cap reached"))
    return state, history, {
        "knotCount": int(state.fit["knotCount"]),
        "converged": converged,
        "sweepsCompleted": sweeps_completed,
        "coverageSweepsWithoutUpdateRequired": coverage_sweeps,
        "maximumSweeps": options.maximum_sweeps_per_count,
        "finalCovarianceStep": covariance_step,
        "finalTransformStep": transform_step,
        "finalBetaStep": beta_step,
    }


def round_robin(
    state: JointState,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    sweep_index: int,
    covariance_step: float,
    transform_step: float,
    beta_step: float,
    knots_stationary: bool,
) -> tuple[JointState, list[dict[str, Any]], bool]:
    history: list[dict[str, Any]] = []
    dimension = raw.shape[1]
    count = int(state.fit["knotCount"])
    cov_directions = covariance_directions(dimension)
    for cov_index, cov_direction in enumerate(cov_directions):
        state, row = optimize_symmetric_stage(
            f"covariance shape {cov_index + 1}/{len(cov_directions)}",
            state,
            raw,
            reference_operations,
            reference,
            options,
            count,
            cov_direction,
            covariance_step,
            100 + cov_index,
        )
        history.append(row)
    post_directions = transform_directions(dimension)
    for transform_index, transform_direction in enumerate(post_directions):
        state, row = optimize_transform_stage(
            f"post-asinh transform {transform_index + 1}/{len(post_directions)}",
            state,
            raw,
            reference_operations,
            reference,
            options,
            count,
            transform_direction,
            transform_step,
            200 + transform_index,
        )
        history.append(row)
    state, row = optimize_beta_stage(
        state,
        raw,
        reference_operations,
        reference,
        options,
        count,
        beta_step,
        300,
    )
    history.append(row)
    parameters_changed = any(bool(update.get("acceptedUpdate")) for update in history)
    if knots_stationary and not parameters_changed:
        skipped = state_summary(state, "knot refit (cached stationary)")
        skipped.update({
            "acceptedUpdate": False,
            "objectiveBefore": float(state.fit["jointObjective"]),
            "candidateObjectives": [],
            "candidateBetas": [],
            "cachedStationary": True,
        })
        history.append(skipped)
        print(
            "    knot refit: skipped; unchanged state is already knot-stationary",
            flush=True,
        )
        return state, history, True
    knot_candidate = evaluate_state(
        state,
        state.empirical_covariance,
        state.covariance_log_shape,
        state.covariance,
        state.transform,
        state.beta,
        count,
        raw,
        reference_operations,
        reference,
        options,
        seed_offset=400,
    )
    before = state
    state = better_state(state, knot_candidate)
    history.append(update_summary("knot refit", before, state, [knot_candidate]))
    return state, history, state is before


def optimize_symmetric_stage(
    name: str,
    state: JointState,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    count: int,
    direction: np.ndarray,
    step: float,
    seed_offset: int,
) -> tuple[JointState, dict[str, Any]]:
    candidates = []
    for index, sign in enumerate((-1.0, 1.0)):
        log_shape = state.covariance_log_shape + sign * step * direction
        log_shape = (log_shape + log_shape.T) / 2.0
        log_shape -= np.eye(log_shape.shape[0]) * np.trace(log_shape) / log_shape.shape[0]
        multiplier = expm(log_shape)
        whitening = multiplier @ state.empirical_covariance.whitening
        coloring = state.empirical_covariance.coloring @ np.linalg.inv(multiplier)
        covariance = CovarianceTransform(
            center=state.empirical_covariance.center,
            whitening=whitening,
            coloring=coloring,
            covariance=coloring @ coloring.T,
        )
        candidates.append(evaluate_fixed_cloud(
            state,
            state.empirical_covariance,
            log_shape,
            covariance,
            state.transform,
            raw,
            reference_operations,
            reference,
            options,
            seed_offset,
        ))
    before = state
    if candidates:
        state = better_state(state, min(candidates, key=objective_key))
    return state, update_summary(name, before, state, candidates)


def optimize_transform_stage(
    name: str,
    state: JointState,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    count: int,
    direction: np.ndarray,
    step: float,
    seed_offset: int,
) -> tuple[JointState, dict[str, Any]]:
    candidates = []
    for index, sign in enumerate((-1.0, 1.0)):
        matrix = expm(sign * step * direction) @ state.transform.matrix
        candidates.append(evaluate_fixed_cloud(
            state,
            state.empirical_covariance,
            state.covariance_log_shape,
            state.covariance,
            AsinhMatrixTransform(matrix),
            raw,
            reference_operations,
            reference,
            options,
            seed_offset,
        ))
    before = state
    if candidates:
        state = better_state(state, min(candidates, key=objective_key))
    return state, update_summary(name, before, state, candidates)


def optimize_beta_stage(
    state: JointState,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    count: int,
    step: float,
    seed_offset: int,
) -> tuple[JointState, dict[str, Any]]:
    dimension = raw.shape[1]
    lower = dimension / (dimension + 2.0)
    beta_candidates = sorted({
        max(lower, min(1.0, state.beta - step)),
        max(lower, min(1.0, state.beta + step)),
    } - {state.beta})
    candidates = [
        evaluate_state(
            state,
            state.empirical_covariance,
            state.covariance_log_shape,
            state.covariance,
            state.transform,
            beta,
            count,
            raw,
            reference_operations,
            reference,
            options,
            seed_offset,
        )
        for index, beta in enumerate(beta_candidates)
    ]
    before = state
    if candidates:
        state = better_state(state, min(candidates, key=objective_key))
    return state, update_summary("allocation beta", before, state, candidates)


def prune_until_boundary(
    state: JointState,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: JointOptions,
    resume: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[JointState, list[dict[str, Any]], list[JointState], list[dict[str, Any]]]:
    dimension = raw.shape[1]
    minimum_step = 1
    if resume is None:
        step = max(minimum_step, int(state.fit["knotCount"]) // (4 if dimension == 2 else 8))
        history: list[dict[str, Any]] = []
        accepted: list[JointState] = []
        convergence_events: list[dict[str, Any]] = []
        start_attempt = 0
    else:
        state = joint_state_from_payload(resume["state"])
        step = int(resume["step"])
        history = deepcopy(resume["history"])
        accepted = [
            joint_state_from_payload(payload)
            for payload in resume.get("acceptedStates", [])
        ]
        convergence_events = deepcopy(resume["convergenceEvents"])
        start_attempt = int(resume["nextAttempt"])

    def save(next_attempt: int) -> None:
        if progress_callback is None:
            return
        # Only the most recent higher-count fallbacks are needed if full
        # verification rejects the lowest search-fidelity state.
        fallback_states = unique_states_by_count(accepted)[-3:]
        progress_callback({
            "state": joint_state_payload(state),
            "step": step,
            "nextAttempt": next_attempt,
            "history": history,
            "acceptedStates": [joint_state_payload(item) for item in fallback_states],
            "convergenceEvents": convergence_events,
        })

    save(start_attempt)
    for attempt in range(start_attempt, options.maximum_prune_attempts):
        current_count = int(state.fit["knotCount"])
        if step < minimum_step or current_count - step < 64:
            break
        target = current_count - step
        print(
            f"  prune attempt {attempt + 1}: {current_count:,} -> {target:,} points...",
            flush=True,
        )
        candidate = evaluate_state(
            state,
            state.empirical_covariance,
            state.covariance_log_shape,
            state.covariance,
            state.transform,
            state.beta,
            target,
            raw,
            reference_operations,
            reference,
            options.optimization,
            seed_offset=1_000 + attempt * 40,
        )
        candidate, updates, convergence = converge_state(
            candidate,
            raw,
            reference_operations,
            reference,
            options,
            direction_offset=(attempt + 1) * 100,
        )
        convergence_events.append(convergence)
        if not candidate.fit["passes"]:
            print(f"    warm continuation failed; trying a cold knot restart at {target:,}...", flush=True)
            cold = evaluate_state(
                state,
                state.empirical_covariance,
                state.covariance_log_shape,
                state.covariance,
                state.transform,
                state.beta,
                target,
                raw,
                reference_operations,
                reference,
                options.optimization,
                seed_offset=2_000 + attempt * 40,
                warm_start=False,
            )
            cold, cold_updates, cold_convergence = converge_state(
                cold,
                raw,
                reference_operations,
                reference,
                options,
                direction_offset=(attempt + 1) * 100 + 50,
            )
            history.append(state_summary(cold, f"cold restart {current_count}->{target}"))
            history.extend(cold_updates)
            convergence_events.append(cold_convergence)
            if prune_key(cold) < prune_key(candidate):
                candidate = cold
        history.append(state_summary(candidate, f"prune proposal {current_count}->{target}"))
        history.extend(updates)
        if candidate.fit["passes"]:
            state = candidate
            accepted.append(candidate)
            history.append(state_summary(state, "accepted prune"))
            step = max(minimum_step, min(step, int(state.fit["knotCount"]) // 4))
        else:
            history.append(state_summary(candidate, "rejected prune"))
            step //= 2
        save(attempt + 1)
    return state, history, accepted, convergence_events


def better_state(incumbent: JointState, candidate: JointState) -> JointState:
    required = float(incumbent.fit["jointObjective"]) * (1.0 - 1e-3)
    return candidate if float(candidate.fit["jointObjective"]) < required else incumbent


def objective_key(state: JointState) -> tuple[float, float]:
    return float(state.fit["jointObjective"]), float(state.fit["acceptanceScore"])


def prune_key(state: JointState) -> tuple[bool, float, float]:
    return not bool(state.fit["passes"]), *objective_key(state)


def covariance_directions(dimension: int) -> list[np.ndarray]:
    directions = []
    for axis in range(dimension - 1):
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        matrix[axis, axis] = 1.0
        matrix[-1, -1] = -1.0
        directions.append(normalized(matrix))
    for row in range(dimension):
        for column in range(row + 1, dimension):
            matrix = np.zeros((dimension, dimension), dtype=np.float64)
            matrix[row, column] = matrix[column, row] = 1.0
            directions.append(normalized(matrix))
    return directions


def transform_directions(dimension: int) -> list[np.ndarray]:
    directions = [normalized(np.eye(dimension, dtype=np.float64))]
    for axis in range(dimension - 1):
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        matrix[axis, axis] = 1.0
        matrix[-1, -1] = -1.0
        directions.append(normalized(matrix))
    for row in range(dimension):
        for column in range(row + 1, dimension):
            rotation = np.zeros((dimension, dimension), dtype=np.float64)
            rotation[row, column] = 1.0
            rotation[column, row] = -1.0
            directions.append(normalized(rotation))
            shear = np.zeros((dimension, dimension), dtype=np.float64)
            shear[row, column] = 1.0
            directions.append(normalized(shear))
    return directions


def normalized(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix)


def update_summary(
    stage: str,
    before: JointState,
    after: JointState,
    candidates: list[JointState],
) -> dict[str, Any]:
    result = state_summary(after, stage)
    result["acceptedUpdate"] = after is not before
    result["objectiveBefore"] = float(before.fit["jointObjective"])
    result["candidateObjectives"] = [float(candidate.fit["jointObjective"]) for candidate in candidates]
    result["candidateBetas"] = [candidate.beta for candidate in candidates]
    print(
        f"    {stage}: {'accepted' if result['acceptedUpdate'] else 'kept incumbent'}; "
        f"objective {result['objectiveBefore']:.6g} -> {result['jointObjective']:.6g}, "
        f"active score={result['acceptanceScore']:.6g}",
        flush=True,
    )
    return result


def state_summary(state: JointState, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "knotCount": int(state.fit["knotCount"]),
        "beta": state.beta,
        "acceptanceScore": float(state.fit["acceptanceScore"]),
        "jointObjective": float(state.fit["jointObjective"]),
        "passes": bool(state.fit["passes"]),
        "densityJsRatioTo1d32": float(state.fit["densityJsRatioTo1d32"]),
        "conditionalRatios": state.fit["conditionalRatios"],
    }


def unique_states_by_count(states: list[JointState]) -> list[JointState]:
    best_by_count: dict[int, JointState] = {}
    for state in states:
        count = int(state.fit["knotCount"])
        incumbent = best_by_count.get(count)
        if incumbent is None or objective_key(state) < objective_key(incumbent):
            best_by_count[count] = state
    return [best_by_count[count] for count in sorted(best_by_count)]


def covariance_payload(covariance: CovarianceTransform) -> dict[str, Any]:
    return {
        "centerBps": covariance.center.tolist(),
        "effectiveCovarianceBpsSquared": (covariance.coloring @ covariance.coloring.T).tolist(),
        "whitening": covariance.whitening.tolist(),
        "coloring": covariance.coloring.tolist(),
    }


def covariance_from_payload(payload: dict[str, Any]) -> CovarianceTransform:
    whitening = np.asarray(payload["whitening"], dtype=np.float64)
    coloring = np.asarray(payload["coloring"], dtype=np.float64)
    return CovarianceTransform(
        center=np.asarray(payload["centerBps"], dtype=np.float64),
        whitening=whitening,
        coloring=coloring,
        covariance=coloring @ coloring.T,
    )


def joint_state_payload(state: JointState) -> dict[str, Any]:
    return {
        "empiricalCovariance": covariance_payload(state.empirical_covariance),
        "covarianceLogShape": state.covariance_log_shape.tolist(),
        "covariance": covariance_payload(state.covariance),
        "transformMatrix": state.transform.matrix.tolist(),
        "beta": state.beta,
        "fit": state.fit,
    }


def joint_state_from_payload(payload: dict[str, Any]) -> JointState:
    return JointState(
        empirical_covariance=covariance_from_payload(payload["empiricalCovariance"]),
        covariance_log_shape=np.asarray(payload["covarianceLogShape"], dtype=np.float64),
        covariance=covariance_from_payload(payload["covariance"]),
        transform=AsinhMatrixTransform(
            np.asarray(payload["transformMatrix"], dtype=np.float64),
        ),
        beta=float(payload["beta"]),
        fit=deepcopy(payload["fit"]),
    )


def factorize_matrix(matrix: np.ndarray) -> dict[str, Any]:
    rotation, upper = np.linalg.qr(np.asarray(matrix, dtype=np.float64))
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    rotation = rotation @ np.diag(signs)
    upper = np.diag(signs) @ upper
    if np.linalg.det(rotation) <= 0:
        raise ValueError("post-asinh matrix does not have a proper rotation factor")
    scales = np.diag(upper)
    shear = upper @ np.diag(1.0 / scales)
    reconstructed = rotation @ shear @ np.diag(scales)
    if not np.allclose(reconstructed, matrix, rtol=1e-10, atol=1e-10):
        raise AssertionError("post-asinh factorization does not reconstruct the matrix")
    return {
        "rotation": rotation.tolist(),
        "unitUpperShear": shear.tolist(),
        "positiveScales": scales.tolist(),
    }


def render_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Joint multidimensional return-knot optimization",
        "",
        f"Generated {artifact['generatedAt']}.",
        "",
        "Only density Jensen-Shannon divergence is used for optimization and acceptance. "
        "Conditional means, medians, and coverage remain in the diagnostic tables but do not "
        "affect fitting or pruning.",
        "",
    ]
    for dimension_text, result in artifact["dimensions"].items():
        initial = result["initial"]
        final = result["final"]
        fit = final["fit"]
        lines.extend([
            f"## {dimension_text}D",
            "",
            f"Initial: {initial['knotCount']:,} points, beta {initial['beta']:.6g}, active "
            f"score {initial['acceptanceScore']:.6g}, joint objective "
            f"{initial['jointObjective']:.6g}.",
            "",
            f"Final: **{final['knotCount']:,} points**, beta **{final['beta']:.6g}**, active "
            f"score **{fit['acceptanceScore']:.6g}**, joint objective "
            f"**{fit['jointObjective']:.6g}**, {'passes' if final['passes'] else 'does not pass'}.",
            "",
            "### Fixed-point checks",
            "",
            "| points | sweeps | converged | covariance step | transform step | beta step |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for event in result["convergenceEvents"]:
            lines.append(
                f"| {event['knotCount']:,} | {event['sweepsCompleted']} | "
                f"{'yes' if event['converged'] else 'safety cap'} | "
                f"{event['finalCovarianceStep']:.6g} | "
                f"{event['finalTransformStep']:.6g} | {event['finalBetaStep']:.6g} |"
            )
        lines.extend([
            "",
            "| metric | ratio to 1D32 | active | passes |",
            "|---|---:|---:|---:|",
            f"| density JS/d | {fit['densityJsRatioTo1d32']:.6g} | yes | "
            f"{'yes' if fit['densityJsRatioTo1d32'] <= 1 else 'no'} |",
        ])
        for query in fit["conditionalRatios"]:
            target = int(query["targetAxis"]) + 1
            mean_active = bool(query["conditionalMeanEnabled"])
            median_active = bool(query["conditionalMedianEnabled"])
            lines.extend([
                f"| r{target} conditional mean | {query['conditionalMeanRatioTo1d32']:.6g} | "
                f"{'yes' if mean_active else 'no'} | "
                f"{'yes' if query['conditionalMeanRatioTo1d32'] <= 1 else 'no'} |",
                f"| r{target} conditional median | {query['conditionalMedianRatioTo1d32']:.6g} | "
                f"{'yes' if median_active else 'no'} | "
                f"{'yes' if query['conditionalMedianRatioTo1d32'] <= 1 else 'no'} |",
            ])
        lines.extend([
            "",
            "### Accepted/rejected iteration trace",
            "",
            "| stage | points | beta | objective | active score | passes |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in result["history"]:
            lines.append(
                f"| {row['stage']} | {row['knotCount']:,} | {row['beta']:.6g} | "
                f"{row['jointObjective']:.6g} | {row['acceptanceScore']:.6g} | "
                f"{'yes' if row['passes'] else 'no'} |"
            )
        lines.extend([
            "",
            f"Covariance adjustment from empirical whitening: "
            f"`{json.dumps(final['covarianceAdjustmentFromEmpirical'], separators=(',', ':'))}`",
            "",
            f"Post-asinh matrix: "
            f"`{json.dumps(final['postAsinhMatrix'], separators=(',', ':'))}`",
            "",
        ])
    lines.extend([
        "## Reproduction",
        "",
        "```text",
        "node scripts/run-ml-python.mjs ml/joint_optimize_multidimensional_return_knots.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
