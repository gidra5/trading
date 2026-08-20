"""Continue JS-only knot pruning with the verification-quality cloud fitter."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from multidimensional_return_knots import (
    AsinhMatrixTransform,
    CovarianceTransform,
    conditional_operation_metrics,
    sample_each_point_cloud_component,
)
from search_multidimensional_return_knots import (
    CONDITIONAL_BINS,
    DEFAULT_PAIR_ANALYSIS,
    DEFAULT_REFERENCE,
    DEFAULT_SAMPLE_CACHE,
    DEFAULT_TRIPLE_ANALYSIS,
    SearchOptions,
    acceptance_metrics,
    load_or_build_samples,
    load_reference,
    quadrature_samples_per_component,
    read_json,
    reference_conditional_operations,
    validate_windows,
)
from joint_optimize_multidimensional_return_knots import (
    DISABLED_CONDITIONAL_MEAN_TARGETS,
    JointState,
    covariance_from_payload,
    covariance_payload,
    evaluate_state,
    factorize_matrix,
    joint_state_from_payload,
    joint_state_payload,
    render_report,
    state_summary,
)


DEFAULT_ARTIFACT = Path("data/benchmarks/multidimensional-return-knot-joint-js-only.json")
DEFAULT_REPORT = Path(
    "docs/experiments/multidimensional-return-knot-joint-js-only-2026-08-18.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine a JS-only knot boundary with high-fidelity cloud fits.",
    )
    parser.add_argument("--pair-analysis", type=Path, default=DEFAULT_PAIR_ANALYSIS)
    parser.add_argument("--triple-analysis", type=Path, default=DEFAULT_TRIPLE_ANALYSIS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sample-cache", type=Path, default=DEFAULT_SAMPLE_CACHE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dimensions", type=int, nargs="+", choices=(2, 3), default=(3,))
    parser.add_argument("--maximum-prune-attempts", type=int, default=40)
    parser.add_argument("--cold-restarts", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maximum_prune_attempts < 1 or args.cold_restarts < 1:
        raise ValueError("prune attempts and cold restarts must be positive")
    repo = Path(__file__).resolve().parents[1]
    pair_path = resolve(repo, args.pair_analysis)
    triple_path = resolve(repo, args.triple_analysis)
    reference_path = resolve(repo, args.reference)
    cache_path = resolve(repo, args.sample_cache)
    artifact_path = resolve(repo, args.artifact)
    report_path = resolve(repo, args.report)
    pair_analysis = read_json(pair_path)
    triple_analysis = read_json(triple_path)
    validate_windows(pair_analysis, triple_analysis)
    options = high_fidelity_options(args)
    samples, _ = load_or_build_samples(repo, pair_analysis, cache_path, options)
    reference = load_reference(reference_path)
    artifact = read_json(artifact_path)
    checkpoint_path = artifact_path.with_name(
        f"{artifact_path.stem}-high-fidelity-partial.json",
    )
    checkpoint = read_json(checkpoint_path) if args.resume and checkpoint_path.exists() else {}
    for dimension in args.dimensions:
        key = str(dimension)
        if key not in artifact.get("dimensions", {}):
            raise ValueError(f"artifact has no {dimension}D result to refine")
        train = samples[f"train{dimension}"].astype(np.float64)
        validation = samples[f"validation{dimension}"].astype(np.float64)
        thinning = max(1, options.train_stride // options.validation_stride)
        raw = np.vstack((train, validation[::thinning]))
        reference_operations = reference_conditional_operations(raw, reference)
        resume_payload = (
            checkpoint
            if int(checkpoint.get("dimension", -1)) == dimension
            else None
        )

        def save_progress(payload: dict[str, Any]) -> None:
            write_checkpoint(checkpoint_path, dimension, payload)

        print(f"\nHigh-fidelity refining {dimension}D point cloud...", flush=True)
        artifact["dimensions"][key] = refine_dimension(
            artifact["dimensions"][key],
            raw,
            reference_operations,
            reference,
            options,
            args.maximum_prune_attempts,
            args.cold_restarts,
            resume_payload,
            save_progress,
        )
        checkpoint = {}
    artifact["generatedAt"] = utc_now()
    artifact.setdefault("methodology", {})["highFidelityBoundaryRefinement"] = (
        "After the joint fast-search boundary, lower knot counts are fit and accepted using "
        "the 320,000-observation MiniBatchKMeans verification path. Warm and cold clouds are "
        "tested after failures, pruning continues to a one-knot rejection, and conditional "
        "operations remain diagnostics computed only for the final retained cloud."
    )
    atomic_write_json(artifact_path, artifact)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(artifact), encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"\nWrote {artifact_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


def high_fidelity_options(args: argparse.Namespace) -> SearchOptions:
    return SearchOptions(
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
        compute_conditional_diagnostics=False,
    )


def refine_dimension(
    result: dict[str, Any],
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    maximum_attempts: int,
    cold_restarts: int,
    resume: dict[str, Any] | None,
    progress_callback: Any,
) -> dict[str, Any]:
    if resume:
        state = joint_state_from_payload(resume["state"])
        step = int(resume["step"])
        start_attempt = int(resume["nextAttempt"])
        history = deepcopy(resume["history"])
        lower_failure = deepcopy(resume.get("lowerFailure"))
        initial_summary = deepcopy(resume["initial"])
        print(
            f"  resuming attempt {start_attempt + 1} from "
            f"{int(state.fit['knotCount']):,} points with step {step:,}",
            flush=True,
        )
    else:
        state = state_from_result(result)
        step = max(1, int(state.fit["knotCount"]) // 4)
        start_attempt = 0
        history: list[dict[str, Any]] = []
        lower_failure: dict[str, Any] | None = None
        initial_summary = state_summary(state, "high-fidelity refinement start")
        print(
            f"  start={int(state.fit['knotCount']):,} points, "
            f"JS ratio={float(state.fit['densityJsRatioTo1d32']):.6g}, "
            f"initial step={step:,}",
            flush=True,
        )

    def save(next_attempt: int) -> None:
        progress_callback({
            "state": joint_state_payload(state),
            "step": step,
            "nextAttempt": next_attempt,
            "history": history,
            "lowerFailure": lower_failure,
            "initial": initial_summary,
        })

    save(start_attempt)
    attempts_completed = start_attempt
    for attempt in range(start_attempt, maximum_attempts):
        current_count = int(state.fit["knotCount"])
        if step < 1 or current_count - step < 64:
            break
        target = current_count - step
        print(
            f"  high-fidelity attempt {attempt + 1}: "
            f"{current_count:,} -> {target:,} points...",
            flush=True,
        )
        warm = fit_candidate(
            state,
            target,
            raw,
            reference_operations,
            reference,
            options,
            40_000 + attempt * 20,
            warm_start=True,
        )
        candidates = [warm]
        history.append(state_summary(warm, f"high-fidelity warm {current_count}->{target}"))
        candidate = warm
        if not warm.fit["passes"]:
            print(
                f"    warm fit failed; trying {cold_restarts} high-fidelity cold fits...",
                flush=True,
            )
            for restart in range(cold_restarts):
                cold = fit_candidate(
                    state,
                    target,
                    raw,
                    reference_operations,
                    reference,
                    options,
                    50_000 + attempt * 100 + restart,
                    warm_start=False,
                )
                candidates.append(cold)
                history.append(state_summary(
                    cold,
                    f"high-fidelity cold {restart + 1}/{cold_restarts} "
                    f"{current_count}->{target}",
                ))
            candidate = min(candidates, key=objective_key)
        history.append(state_summary(candidate, f"high-fidelity proposal {current_count}->{target}"))
        if candidate.fit["passes"]:
            state = candidate
            history.append(state_summary(state, "accepted high-fidelity prune"))
            lower_failure = None
            step = max(1, min(step, int(state.fit["knotCount"]) // 4))
            print(
                f"    accepted; JS ratio={float(state.fit['densityJsRatioTo1d32']):.6g}",
                flush=True,
            )
        else:
            lower_failure = state_summary(candidate, "rejected high-fidelity prune")
            history.append(deepcopy(lower_failure))
            step //= 2
            print(
                f"    rejected; best JS ratio="
                f"{float(candidate.fit['densityJsRatioTo1d32']):.6g}; next step={step:,}",
                flush=True,
            )
        attempts_completed = attempt + 1
        save(attempts_completed)
    if step >= 1 and attempts_completed >= maximum_attempts:
        raise RuntimeError(
            "high-fidelity refinement exhausted its attempt budget before reaching a boundary",
        )
    if lower_failure is None or int(lower_failure["knotCount"]) != int(state.fit["knotCount"]) - 1:
        raise RuntimeError("high-fidelity refinement did not establish a one-knot boundary")
    final_state = attach_conditional_diagnostics(
        state,
        raw,
        reference_operations,
        reference,
        options,
    )
    refined = deepcopy(result)
    refined.setdefault("history", []).extend(history)
    refined.setdefault("verificationCandidates", []).append(
        state_summary(final_state, "high-fidelity exact-boundary verification"),
    )
    refined["highFidelityRefinement"] = {
        "initial": initial_summary,
        "history": history,
        "attemptsCompleted": attempts_completed,
        "lowerFailure": lower_failure,
        "oneKnotBoundary": True,
        "cloudSample": options.cloud_sample,
        "conditionalDiagnosticsOnlyAtFinal": True,
        "coldRestartsAfterWarmFailure": cold_restarts,
    }
    covariance_adjustment = matrix_exponential(final_state.covariance_log_shape)
    refined["final"] = {
        "knotCount": int(final_state.fit["knotCount"]),
        "beta": final_state.beta,
        "passes": bool(final_state.fit["passes"]),
        "covariance": covariance_payload(final_state.covariance),
        "covarianceAdjustmentFromEmpirical": covariance_adjustment.tolist(),
        "postAsinhMatrix": final_state.transform.matrix.tolist(),
        "postAsinhFactorization": factorize_matrix(final_state.transform.matrix),
        "fit": final_state.fit,
    }
    print(
        f"  exact high-fidelity boundary: {int(final_state.fit['knotCount']):,} passes, "
        f"{int(lower_failure['knotCount']):,} fails",
        flush=True,
    )
    return refined


def fit_candidate(
    source: JointState,
    count: int,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
    seed_offset: int,
    *,
    warm_start: bool,
) -> JointState:
    return evaluate_state(
        source,
        source.empirical_covariance,
        source.covariance_log_shape,
        source.covariance,
        source.transform,
        source.beta,
        count,
        raw,
        reference_operations,
        reference,
        options,
        seed_offset,
        warm_start=warm_start,
    )


def state_from_result(result: dict[str, Any]) -> JointState:
    final = result["final"]
    covariance = covariance_from_payload(final["covariance"])
    adjustment = np.asarray(final["covarianceAdjustmentFromEmpirical"], dtype=np.float64)
    inverse_adjustment = np.linalg.inv(adjustment)
    empirical_whitening = inverse_adjustment @ covariance.whitening
    empirical_coloring = covariance.coloring @ adjustment
    empirical = CovarianceTransform(
        center=covariance.center.copy(),
        whitening=empirical_whitening,
        coloring=empirical_coloring,
        covariance=empirical_coloring @ empirical_coloring.T,
    )
    eigenvalues, eigenvectors = np.linalg.eigh((adjustment + adjustment.T) / 2.0)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("covariance adjustment is not positive definite")
    log_shape = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
    return JointState(
        empirical_covariance=empirical,
        covariance_log_shape=log_shape,
        covariance=covariance,
        transform=AsinhMatrixTransform(np.asarray(final["postAsinhMatrix"], dtype=np.float64)),
        beta=float(final["beta"]),
        fit=deepcopy(final["fit"]),
    )


def attach_conditional_diagnostics(
    state: JointState,
    raw: np.ndarray,
    reference_operations: list[dict[str, float | int]],
    reference: dict[str, Any],
    options: SearchOptions,
) -> JointState:
    centers = np.asarray(state.fit["centersUnit"], dtype=np.float64)
    widths = np.asarray(state.fit["bandwidthsUnit"], dtype=np.float64)
    weights = np.asarray(state.fit["componentWeights"], dtype=np.float64)
    count, dimension = centers.shape
    samples_per_component = quadrature_samples_per_component(
        count,
        options.conditional_draws,
        maximum=2_048,
        minimum=options.minimum_component_draws,
    )
    unit_model, component_ids = sample_each_point_cloud_component(
        centers,
        widths,
        seed=91_009 + 101 * dimension + count,
        samples_per_component=samples_per_component,
    )
    unit_model = np.clip(
        unit_model,
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    raw_model = state.covariance.inverse(state.transform.inverse(unit_model))
    model_weights = weights[component_ids] / samples_per_component
    operations = conditional_operation_metrics(
        raw,
        raw_model,
        CONDITIONAL_BINS,
        model_weights,
    )
    acceptance = acceptance_metrics(
        state.fit["density"],
        operations,
        reference,
        reference_operations,
        DISABLED_CONDITIONAL_MEAN_TARGETS,
        conditional_metrics_active=False,
    )
    fit = deepcopy(state.fit)
    fit.update({
        "conditionalOperations": operations,
        "validationConditionalOperations": operations,
        "operationQuadratureSamplesPerComponent": samples_per_component,
        "operationQuadratureDraws": int(unit_model.shape[0]),
        "conditionalDiagnosticsComputed": True,
        **acceptance,
    })
    return JointState(
        state.empirical_covariance,
        state.covariance_log_shape,
        state.covariance,
        state.transform,
        state.beta,
        fit,
    )


def objective_key(state: JointState) -> tuple[float, float]:
    return float(state.fit["jointObjective"]), float(state.fit["acceptanceScore"])


def matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T


def write_checkpoint(path: Path, dimension: int, payload: dict[str, Any]) -> None:
    checkpoint = {
        "version": 1,
        "generatedAt": utc_now(),
        "dimension": dimension,
        **payload,
    }
    atomic_write_json(path, checkpoint)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
