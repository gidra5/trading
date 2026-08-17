from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import beta as beta_function

from trading_storage import read_candle_column


ObjectiveName = Literal["js", "kl", "cdfL2", "joint"]

DEFAULT_ANALYSIS = Path("data/benchmarks/log-return-distributions.json")
DEFAULT_FITS = Path("data/benchmarks/log-return-distribution-fits.json")
DEFAULT_OUTPUT = Path("data/benchmarks/one-second-return-64-knot-fits.json")
DEFAULT_REPORT = Path("docs/experiments/one-second-return-64-knot-fits-2026-08-14.md")
DEFAULT_CACHE = Path("data/runtime-cache/one-second-active-static-transform-histogram.npz")
LOG_2 = math.log(2.0)


@dataclass(frozen=True)
class StaticTransform:
    alpha: float
    location_bps: float
    scale_bps: float

    def forward(self, return_bps: np.ndarray) -> np.ndarray:
        latent = self.alpha * np.arcsinh(
            (return_bps - self.location_bps) / self.scale_bps,
        )
        positive = latent >= 0.0
        result = np.empty_like(latent, dtype=np.float64)
        result[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
        exp_value = np.exp(latent[~positive])
        result[~positive] = exp_value / (1.0 + exp_value)
        return result

    def inverse(self, unit: np.ndarray) -> np.ndarray:
        logit = np.log(unit) - np.log1p(-unit)
        return self.location_bps + self.scale_bps * np.sinh(logit / self.alpha)


@dataclass(frozen=True)
class TargetHistogram:
    counts: np.ndarray
    unit_edges: np.ndarray
    unit_centers: np.ndarray
    return_widths_bps: np.ndarray
    observations: int
    active_observations: int
    zero_observations: int
    active_standard_deviation_bps: float

    @property
    def probabilities(self) -> np.ndarray:
        return self.counts.astype(np.float64) / self.active_observations


@dataclass
class FitState:
    objective: ObjectiveName
    knots: np.ndarray
    component_weights: np.ndarray
    knot_density_heights: np.ndarray
    objective_value: float
    convergence: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit static triangular return-density knots to the full 1s history.",
    )
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--knots", type=int, default=64)
    parser.add_argument("--core-bins", type=int, default=131_072)
    parser.add_argument("--optimization-stride", type=int, default=4)
    parser.add_argument("--latent-limit", type=float, default=36.0)
    parser.add_argument("--adam-steps", type=int, default=400)
    parser.add_argument("--lbfgs-steps", type=int, default=60)
    parser.add_argument("--full-refinement-adam-steps", type=int, default=180)
    parser.add_argument("--full-refinement-lbfgs-steps", type=int, default=80)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.knots < 3:
        raise ValueError("At least three knots are required.")
    if args.core_bins < 1_024:
        raise ValueError("core-bins must be at least 1024.")
    repo = Path(__file__).resolve().parents[1]
    analysis_path = resolve(repo, args.analysis)
    fits_path = resolve(repo, args.fits)
    output_path = resolve(repo, args.output)
    report_path = resolve(repo, args.report)
    cache_path = resolve(repo, args.cache)
    analysis = read_json(analysis_path)
    fits = read_json(fits_path)
    transform = static_transform_from_fit(fits)
    print(
        "Static transform: "
        f"alpha={transform.alpha:.12g}, mu={transform.location_bps:.12g} bps, "
        f"scale={transform.scale_bps:.12g} bps",
        flush=True,
    )
    target = load_or_build_target(
        repo,
        analysis,
        transform,
        cache_path,
        args.core_bins,
        args.latent_limit,
        args.rebuild_cache,
    )
    print(
        f"Target: {target.observations:,} returns, {target.active_observations:,} active, "
        f"zero={target.zero_observations / target.observations:.6%}",
        flush=True,
    )
    optimization_target = coarsen_target(target, args.optimization_stride)
    fitter = KnotFitter(optimization_target, args.knots, args.device)
    auditor = KnotFitter(target, args.knots, args.device)
    coarse_states: dict[str, FitState] = {}
    for objective in ("js", "kl", "cdfL2"):
        print(f"Optimizing {objective}...", flush=True)
        coarse_states[objective] = fitter.fit(
            objective,
            args.adam_steps,
            args.lbfgs_steps,
            args.restarts,
        )
    fits_by_objective: dict[str, FitState] = {}
    for objective in ("js", "kl", "cdfL2"):
        print(f"Cross-refining {objective} on the full grid...", flush=True)
        candidates = [
            auditor.refine(
                state,
                objective,
                args.full_refinement_adam_steps,
                args.full_refinement_lbfgs_steps,
            )
            for state in (*coarse_states.values(), *fits_by_objective.values())
        ]
        fits_by_objective[objective] = min(candidates, key=lambda state: state.objective_value)
        metrics = auditor.metrics_for_state(fits_by_objective[objective])
        print(
            f"  JS={metrics['jsBits']:.9g} bits, KL={metrics['klBits']:.9g} bits, "
            f"CDF-L2={metrics['cdfL2Bps']:.9g} bps",
            flush=True,
        )
    joint_scales = {
        "js": auditor.metrics_for_state(fits_by_objective["js"])["jsBits"],
        "kl": auditor.metrics_for_state(fits_by_objective["kl"])["klBits"],
        "cdfL2": auditor.metrics_for_state(fits_by_objective["cdfL2"])["cdfL2Normalized"],
    }
    print(f"Optimizing joint relative objective with scales {joint_scales}...", flush=True)
    coarse_joint = fitter.fit(
        "joint",
        args.adam_steps,
        args.lbfgs_steps,
        args.restarts,
        joint_scales=joint_scales,
        extra_initial_states=list(fits_by_objective.values()),
    )
    joint_seed: FitState = coarse_joint
    stabilized = False
    for cycle in range(8):
        joint_scales = {
            "js": auditor.metrics_for_state(fits_by_objective["js"])["jsBits"],
            "kl": auditor.metrics_for_state(fits_by_objective["kl"])["klBits"],
            "cdfL2": auditor.metrics_for_state(fits_by_objective["cdfL2"])["cdfL2Normalized"],
        }
        print(f"Joint/full cross-refinement cycle {cycle + 1} with {joint_scales}...", flush=True)
        joint_candidates = (joint_seed, *fits_by_objective.values())
        fits_by_objective["joint"] = min(
            (
                auditor.refine(
                    state,
                    "joint",
                    args.full_refinement_adam_steps,
                    args.full_refinement_lbfgs_steps,
                    joint_scales,
                )
                for state in joint_candidates
            ),
            key=lambda state: state.objective_value,
        )
        improved = False
        for objective in ("js", "kl", "cdfL2"):
            candidate = auditor.refine(
                fits_by_objective["joint"],
                objective,
                args.full_refinement_adam_steps,
                args.full_refinement_lbfgs_steps,
            )
            if candidate.objective_value < fits_by_objective[objective].objective_value * (1.0 - 1e-5):
                fits_by_objective[objective] = candidate
                improved = True
        joint_seed = fits_by_objective["joint"]
        if not improved:
            stabilized = True
            break
    if not stabilized:
        print(
            "Reached the bounded eight-cycle candidate search; selecting the best "
            "observed solutions and performing one final joint refinement.",
            flush=True,
        )

    # Recompute the joint normalization from the best single-objective candidates
    # found by the bounded search, then give the joint objective one final pass.
    joint_scales = {
        "js": auditor.metrics_for_state(fits_by_objective["js"])["jsBits"],
        "kl": auditor.metrics_for_state(fits_by_objective["kl"])["klBits"],
        "cdfL2": auditor.metrics_for_state(fits_by_objective["cdfL2"])["cdfL2Normalized"],
    }
    final_joint_candidates = tuple(fits_by_objective.values())
    fits_by_objective["joint"] = min(
        (
            auditor.refine(
                state,
                "joint",
                args.full_refinement_adam_steps,
                args.full_refinement_lbfgs_steps,
                joint_scales,
            )
            for state in final_joint_candidates
        ),
        key=lambda state: state.objective_value,
    )

    # A joint candidate is also a valid candidate for each individual objective.
    # Relabel it when it is the best observed fit, which makes the reported
    # single-objective minima honest over the entire candidate pool.
    joint_metrics = auditor.metrics_for_state(fits_by_objective["joint"])
    metric_keys = {
        "js": "jsBits",
        "kl": "klBits",
        "cdfL2": "cdfL2Normalized",
    }
    for objective, metric_key in metric_keys.items():
        single_metrics = auditor.metrics_for_state(fits_by_objective[objective])
        if joint_metrics[metric_key] < single_metrics[metric_key]:
            joint_state = fits_by_objective["joint"]
            fits_by_objective[objective] = FitState(
                objective=objective,
                knots=joint_state.knots.copy(),
                component_weights=joint_state.component_weights.copy(),
                knot_density_heights=joint_state.knot_density_heights.copy(),
                objective_value=joint_metrics[metric_key],
                convergence={
                    **joint_state.convergence,
                    "selectedFromJointCandidatePool": True,
                },
            )

    joint_scales = {
        "js": auditor.metrics_for_state(fits_by_objective["js"])["jsBits"],
        "kl": auditor.metrics_for_state(fits_by_objective["kl"])["klBits"],
        "cdfL2": auditor.metrics_for_state(fits_by_objective["cdfL2"])["cdfL2Normalized"],
    }
    joint_metrics = auditor.metrics_for_state(fits_by_objective["joint"])
    fits_by_objective["joint"].objective_value = (
        joint_metrics["jsBits"] / joint_scales["js"]
        + joint_metrics["klBits"] / joint_scales["kl"]
        + joint_metrics["cdfL2Normalized"] / joint_scales["cdfL2"]
    ) / 3.0
    metrics_by_fit = {
        name: auditor.metrics_for_state(state)
        for name, state in fits_by_objective.items()
    }
    pairwise = pairwise_with_transform(fits_by_objective, target, transform)
    artifact = build_artifact(
        analysis_path,
        fits_path,
        cache_path,
        analysis,
        transform,
        target,
        args,
        fits_by_objective,
        metrics_by_fit,
        pairwise,
        joint_scales,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(repo)}", flush=True)
    print(f"Wrote {report_path.relative_to(repo)}", flush=True)


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def static_transform_from_fit(fits: dict[str, Any]) -> StaticTransform:
    scale_fit = next(item for item in fits["scales"] if item["id"] == "1s")
    parameters = scale_fit["parameters"]
    sigma = float(scale_fit["sigmaBps"])
    alpha = float(scale_fit["asymptoticSurvivalExponent"])
    location = float(parameters["locationSigma"]) * sigma
    generalized_t_scale = float(parameters["scaleSigma"]) * sigma
    power = float(parameters["power"])
    tail = float(parameters["tail"])
    scale = project_generalized_t_scale(
        generalized_t_scale,
        power,
        tail,
        alpha,
    )
    return StaticTransform(alpha=alpha, location_bps=location, scale_bps=scale)


def project_generalized_t_scale(
    generalized_t_scale: float,
    power: float,
    tail: float,
    alpha: float,
) -> float:
    normalization = power / (
        2.0 * beta_function(1.0 / power, tail - 1.0 / power)
    )

    def generalized_t_density(unit: float) -> float:
        return normalization * (1.0 + unit**power) ** (-tail)

    def log_cosh(value: float) -> float:
        absolute = abs(value)
        return absolute + math.log1p(math.exp(-2.0 * absolute)) - math.log(2.0)

    def cross_entropy(log_scale: float) -> float:
        scale = math.exp(log_scale)
        ratio = generalized_t_scale / scale

        def integrand(unit: float) -> float:
            standardized = ratio * unit
            latent = alpha * math.asinh(standardized)
            log_density = (
                math.log(alpha)
                - math.log(scale)
                - 0.5 * math.log1p(standardized * standardized)
                - math.log(4.0)
                - 2.0 * log_cosh(latent / 2.0)
            )
            return 2.0 * generalized_t_density(unit) * -log_density

        return quad(
            integrand,
            0.0,
            math.inf,
            epsabs=1e-10,
            epsrel=2e-9,
            limit=300,
        )[0]

    result = minimize_scalar(
        cross_entropy,
        bounds=(math.log(generalized_t_scale / 100.0), math.log(generalized_t_scale * 100.0)),
        method="bounded",
        options={"xatol": 1e-11},
    )
    if not result.success:
        raise RuntimeError(f"Static transform scale projection failed: {result.message}")
    return math.exp(float(result.x))


def histogram_geometry(
    transform: StaticTransform,
    core_bins: int,
    latent_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent_edges = np.linspace(-latent_limit, latent_limit, core_bins + 1, dtype=np.float64)
    finite_unit_edges = stable_sigmoid(latent_edges)
    unit_edges = np.concatenate(([0.0], finite_unit_edges, [1.0]))
    latent_centers = (latent_edges[:-1] + latent_edges[1:]) / 2.0
    core_unit_centers = stable_sigmoid(latent_centers)
    unit_centers = np.concatenate((
        [stable_sigmoid_scalar(-latent_limit - 1.0)],
        core_unit_centers,
        [stable_sigmoid_scalar(latent_limit + 1.0)],
    ))
    return_edges = transform.location_bps + transform.scale_bps * np.sinh(
        latent_edges / transform.alpha,
    )
    return_widths = np.zeros(core_bins + 2, dtype=np.float64)
    return_widths[1:-1] = np.diff(return_edges)
    return unit_edges, unit_centers, return_widths, latent_edges


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def stable_sigmoid_scalar(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def load_or_build_target(
    repo: Path,
    analysis: dict[str, Any],
    transform: StaticTransform,
    cache_path: Path,
    core_bins: int,
    latent_limit: float,
    rebuild: bool,
) -> TargetHistogram:
    geometry = histogram_geometry(transform, core_bins, latent_limit)
    unit_edges, unit_centers, return_widths, _ = geometry
    expected_metadata = np.asarray([
        transform.alpha,
        transform.location_bps,
        transform.scale_bps,
        float(core_bins),
        latent_limit,
    ], dtype=np.float64)
    if cache_path.exists() and not rebuild:
        cached = np.load(cache_path)
        if np.array_equal(cached["metadata"], expected_metadata):
            return TargetHistogram(
                counts=cached["counts"],
                unit_edges=unit_edges,
                unit_centers=unit_centers,
                return_widths_bps=return_widths,
                observations=int(cached["observations"][0]),
                active_observations=int(cached["active_observations"][0]),
                zero_observations=int(cached["zero_observations"][0]),
                active_standard_deviation_bps=float(cached["active_standard_deviation_bps"][0]),
            )
    full = analysis["fullHistory"]
    start = datetime.fromisoformat(full["startTime"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(full["endTime"].replace("Z", "+00:00"))
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    files = sorted(source.glob("????-??-??.json"))
    counts = np.zeros(core_bins + 2, dtype=np.int64)
    observations = 0
    zeros = 0
    active_sum = 0.0
    active_sum_squared = 0.0
    previous_close: float | None = None
    latent_step = 2.0 * latent_limit / core_bins
    selected = []
    for file in files:
        day = datetime.fromisoformat(file.stem).replace(tzinfo=timezone.utc)
        if start <= day < end:
            selected.append(file)
    for index, reference in enumerate(selected):
        if index % 50 == 0:
            print(f"Reading 1s target {index}/{len(selected)}...", flush=True)
        closes = read_candle_column(reference, "close").astype(np.float64, copy=False)
        if closes.size != 86_400:
            raise ValueError(f"Incomplete 1s shard: {reference}")
        if previous_close is None:
            returns = np.log(closes[1:] / closes[:-1]) * 10_000.0
        else:
            prior = np.empty(closes.size + 1, dtype=np.float64)
            prior[0] = previous_close
            prior[1:] = closes
            returns = np.log(prior[1:] / prior[:-1]) * 10_000.0
        previous_close = float(closes[-1])
        observations += int(returns.size)
        active_mask = returns != 0.0
        zero_count = int(returns.size - np.count_nonzero(active_mask))
        zeros += zero_count
        active = returns[active_mask]
        active_sum += float(np.sum(active, dtype=np.float64))
        active_sum_squared += float(np.dot(active, active))
        latent = transform.alpha * np.arcsinh(
            (active - transform.location_bps) / transform.scale_bps,
        )
        indices = np.floor((latent + latent_limit) / latent_step).astype(np.int64) + 1
        indices[latent < -latent_limit] = 0
        indices[latent >= latent_limit] = core_bins + 1
        counts += np.bincount(indices, minlength=core_bins + 2)
    active_observations = observations - zeros
    if int(np.sum(counts)) != active_observations:
        raise AssertionError("Active histogram count does not match active observations.")
    active_mean = active_sum / active_observations
    active_variance = (
        active_sum_squared - active_observations * active_mean * active_mean
    ) / (active_observations - 1)
    target = TargetHistogram(
        counts=counts,
        unit_edges=unit_edges,
        unit_centers=unit_centers,
        return_widths_bps=return_widths,
        observations=observations,
        active_observations=active_observations,
        zero_observations=zeros,
        active_standard_deviation_bps=math.sqrt(active_variance),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        metadata=expected_metadata,
        counts=counts,
        observations=np.asarray([observations], dtype=np.int64),
        active_observations=np.asarray([active_observations], dtype=np.int64),
        zero_observations=np.asarray([zeros], dtype=np.int64),
        active_standard_deviation_bps=np.asarray([target.active_standard_deviation_bps]),
    )
    return target


def coarsen_target(target: TargetHistogram, stride: int) -> TargetHistogram:
    if stride < 1:
        raise ValueError("optimization-stride must be positive.")
    if stride == 1:
        return target
    core_bins = target.counts.size - 2
    if core_bins % stride != 0:
        raise ValueError("optimization-stride must divide core-bins exactly.")
    core_counts = target.counts[1:-1].reshape(-1, stride).sum(axis=1)
    counts = np.concatenate((target.counts[:1], core_counts, target.counts[-1:]))
    finite_edges = target.unit_edges[1:-1]
    selected_edges = finite_edges[::stride]
    if selected_edges[-1] != finite_edges[-1]:
        selected_edges = np.concatenate((selected_edges, finite_edges[-1:]))
    unit_edges = np.concatenate(([0.0], selected_edges, [1.0]))
    source_centers = target.unit_centers[1:-1].reshape(-1, stride)
    source_latent_centers = np.log(source_centers) - np.log1p(-source_centers)
    core_centers = stable_sigmoid(np.mean(source_latent_centers, axis=1))
    unit_centers = np.concatenate((target.unit_centers[:1], core_centers, target.unit_centers[-1:]))
    core_widths = target.return_widths_bps[1:-1].reshape(-1, stride).sum(axis=1)
    return_widths = np.concatenate((
        target.return_widths_bps[:1],
        core_widths,
        target.return_widths_bps[-1:],
    ))
    if unit_edges.size != counts.size + 1 or unit_centers.size != counts.size:
        raise AssertionError("Coarsened target geometry is inconsistent.")
    return TargetHistogram(
        counts=counts,
        unit_edges=unit_edges,
        unit_centers=unit_centers,
        return_widths_bps=return_widths,
        observations=target.observations,
        active_observations=target.active_observations,
        zero_observations=target.zero_observations,
        active_standard_deviation_bps=target.active_standard_deviation_bps,
    )


class KnotFitter:
    def __init__(self, target: TargetHistogram, knot_count: int, device_name: str):
        self.target = target
        self.knot_count = knot_count
        self.device = torch.device(device_name)
        self.dtype = torch.float64
        self.target_probabilities = torch.as_tensor(
            target.probabilities,
            dtype=self.dtype,
            device=self.device,
        )
        self.unit_edges = torch.as_tensor(target.unit_edges, dtype=self.dtype, device=self.device)
        self.unit_centers = torch.as_tensor(
            target.unit_centers,
            dtype=self.dtype,
            device=self.device,
        )
        self.return_widths = torch.as_tensor(
            target.return_widths_bps,
            dtype=self.dtype,
            device=self.device,
        )
        self.target_cdf_centers = torch.cumsum(self.target_probabilities, dim=0) \
            - 0.5 * self.target_probabilities
        self.initializations = self._initializations()

    def _initializations(self) -> list[tuple[np.ndarray, np.ndarray, str]]:
        probabilities = self.target.probabilities
        cumulative = np.cumsum(probabilities)
        quantile_levels = np.arange(self.knot_count, dtype=np.float64) / (self.knot_count - 1)
        quantile_knots = np.empty(self.knot_count, dtype=np.float64)
        quantile_knots[0] = 0.0
        quantile_knots[-1] = 1.0
        for index, probability in enumerate(quantile_levels[1:-1], start=1):
            bin_index = int(np.searchsorted(cumulative, probability, side="left"))
            prior = cumulative[bin_index - 1] if bin_index > 0 else 0.0
            mass = probabilities[bin_index]
            fraction = 0.5 if mass <= 0.0 else (probability - prior) / mass
            left = self.target.unit_edges[bin_index]
            right = self.target.unit_edges[bin_index + 1]
            quantile_knots[index] = left + fraction * (right - left)
        latent_knots = stable_sigmoid(np.linspace(-18.0, 18.0, self.knot_count))
        latent_knots[0] = 0.0
        latent_knots[-1] = 1.0
        blended = 0.75 * quantile_knots + 0.25 * latent_knots
        blended[0] = 0.0
        blended[-1] = 1.0
        return [
            (quantile_knots, self._initial_component_weights(quantile_knots), "equal-mass"),
            (blended, self._initial_component_weights(blended), "blended"),
            (latent_knots, self._initial_component_weights(latent_knots), "uniform-latent"),
        ]

    def _initial_component_weights(self, knots: np.ndarray) -> np.ndarray:
        cdf = np.cumsum(self.target.probabilities)
        values = np.interp(knots, self.target.unit_edges[1:], cdf, left=0.0, right=1.0)
        boundaries = np.empty(self.knot_count + 1, dtype=np.float64)
        boundaries[0] = 0.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = (knots[:-1] + knots[1:]) / 2.0
        boundary_cdf = np.interp(boundaries, knots, values, left=0.0, right=1.0)
        weights = np.maximum(np.diff(boundary_cdf), 1e-12)
        return weights / np.sum(weights)

    def fit(
        self,
        objective: ObjectiveName,
        adam_steps: int,
        lbfgs_steps: int,
        restarts: int,
        joint_scales: dict[str, float] | None = None,
        extra_initial_states: list[FitState] | None = None,
    ) -> FitState:
        starts = list(self.initializations[:restarts])
        if extra_initial_states:
            starts = starts[:1]
            starts.extend(
                (state.knots, state.component_weights, f"from-{state.objective}")
                for state in extra_initial_states
            )
        best: FitState | None = None
        for restart_index, (initial_knots, initial_weights, label) in enumerate(starts):
            state = self._fit_once(
                objective,
                initial_knots,
                initial_weights,
                adam_steps,
                lbfgs_steps,
                joint_scales,
                restart_index,
                label,
            )
            if best is None or state.objective_value < best.objective_value:
                best = state
            print(
                f"    {label}: objective={state.objective_value:.12g}, "
                f"adam={state.convergence['adamFinal']:.12g}, "
                f"lbfgs={state.convergence['lbfgsFinal']:.12g}",
                flush=True,
            )
        assert best is not None
        return best

    def refine(
        self,
        state: FitState,
        objective: ObjectiveName,
        adam_steps: int,
        lbfgs_steps: int,
        joint_scales: dict[str, float] | None = None,
    ) -> FitState:
        print(f"    full-resolution refinement from {state.objective}...", flush=True)
        return self._fit_once(
            objective,
            state.knots,
            state.component_weights,
            adam_steps,
            lbfgs_steps,
            joint_scales,
            0,
            f"full-from-{state.objective}",
            0.003,
        )

    def _fit_once(
        self,
        objective: ObjectiveName,
        initial_knots: np.ndarray,
        initial_weights: np.ndarray,
        adam_steps: int,
        lbfgs_steps: int,
        joint_scales: dict[str, float] | None,
        restart_index: int,
        initialization: str,
        adam_learning_rate: float = 0.025,
    ) -> FitState:
        gaps = np.diff(initial_knots)
        if np.any(gaps <= 0.0):
            raise ValueError(f"Initialization {initialization} has unordered knots.")
        gap_logits = torch.nn.Parameter(torch.as_tensor(
            np.log(gaps), dtype=self.dtype, device=self.device,
        ))
        weight_logits = torch.nn.Parameter(torch.as_tensor(
            np.log(np.maximum(initial_weights, 1e-15)), dtype=self.dtype, device=self.device,
        ))
        parameters = [gap_logits, weight_logits]
        adam = torch.optim.Adam(parameters, lr=adam_learning_rate)
        with torch.no_grad():
            initial_value = float(
                self._objective(gap_logits, weight_logits, objective, joint_scales).cpu()
            )
        best_value = initial_value
        best_parameters: tuple[torch.Tensor, torch.Tensor] | None = (
            gap_logits.detach().clone(),
            weight_logits.detach().clone(),
        )
        for step in range(adam_steps):
            adam.zero_grad(set_to_none=True)
            value = self._objective(gap_logits, weight_logits, objective, joint_scales)
            value.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            adam.step()
            numeric = float(value.detach().cpu())
            if numeric < best_value:
                best_value = numeric
                best_parameters = (
                    gap_logits.detach().clone(),
                    weight_logits.detach().clone(),
                )
            if step > 150 and step % 100 == 0:
                if not math.isfinite(numeric):
                    break
        if best_parameters is None:
            raise RuntimeError(f"{objective}/{initialization} never produced a finite objective.")
        with torch.no_grad():
            gap_logits.copy_(best_parameters[0])
            weight_logits.copy_(best_parameters[1])
        adam_final = float(
            self._objective(gap_logits, weight_logits, objective, joint_scales).detach().cpu()
        )
        lbfgs = torch.optim.LBFGS(
            parameters,
            lr=0.8,
            max_iter=lbfgs_steps,
            max_eval=lbfgs_steps * 2,
            tolerance_grad=1e-10,
            tolerance_change=1e-13,
            history_size=40,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = self._objective(gap_logits, weight_logits, objective, joint_scales)
            loss.backward()
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError as error:
            print(f"    LBFGS warning for {objective}/{initialization}: {error}", file=sys.stderr)
        final_value = float(
            self._objective(gap_logits, weight_logits, objective, joint_scales).detach().cpu()
        )
        if final_value > adam_final:
            with torch.no_grad():
                gap_logits.copy_(best_parameters[0])
                weight_logits.copy_(best_parameters[1])
            final_value = adam_final
        knots, component_weights, heights = self._decode(gap_logits, weight_logits)
        return FitState(
            objective=objective,
            knots=knots.detach().cpu().numpy(),
            component_weights=component_weights.detach().cpu().numpy(),
            knot_density_heights=heights.detach().cpu().numpy(),
            objective_value=final_value,
            convergence={
                "restart": restart_index,
                "initialization": initialization,
                "adamSteps": adam_steps,
                "lbfgsMaximumIterations": lbfgs_steps,
                "adamFinal": adam_final,
                "lbfgsFinal": final_value,
            },
        )

    def _decode(
        self,
        gap_logits: torch.Tensor,
        weight_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        minimum_gap = 1e-15
        gaps = torch.softmax(gap_logits, dim=0)
        gaps = minimum_gap + (1.0 - minimum_gap * gaps.numel()) * gaps
        knots = torch.cat((
            torch.zeros(1, dtype=self.dtype, device=self.device),
            torch.cumsum(gaps, dim=0),
        ))
        component_weights = torch.softmax(weight_logits, dim=0)
        areas = torch.empty_like(component_weights)
        areas[0] = gaps[0] / 2.0
        areas[-1] = gaps[-1] / 2.0
        areas[1:-1] = (gaps[:-1] + gaps[1:]) / 2.0
        heights = component_weights / areas
        return knots, component_weights, heights

    def _cdf(
        self,
        points: torch.Tensor,
        knots: torch.Tensor,
        heights: torch.Tensor,
    ) -> torch.Tensor:
        gaps = knots[1:] - knots[:-1]
        interval_areas = gaps * (heights[:-1] + heights[1:]) / 2.0
        prefix = torch.cat((
            torch.zeros(1, dtype=self.dtype, device=self.device),
            torch.cumsum(interval_areas, dim=0),
        ))
        interval = torch.bucketize(points.contiguous(), knots[1:-1].contiguous())
        left = knots[interval]
        width = gaps[interval]
        left_height = heights[interval]
        right_height = heights[interval + 1]
        distance = points - left
        slope = (right_height - left_height) / width
        result = prefix[interval] + left_height * distance + 0.5 * slope * distance * distance
        return torch.clamp(result, 0.0, 1.0)

    def _model_probabilities(
        self,
        knots: torch.Tensor,
        heights: torch.Tensor,
    ) -> torch.Tensor:
        cdf_edges = self._cdf(self.unit_edges, knots, heights)
        probabilities = cdf_edges[1:] - cdf_edges[:-1]
        return torch.clamp(probabilities, min=1e-300)

    def _raw_metrics(
        self,
        knots: torch.Tensor,
        heights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target = self.target_probabilities
        model = self._model_probabilities(knots, heights)
        mixture = (target + model) / 2.0
        positive_target = target > 0.0
        js = 0.5 * torch.sum(torch.where(
            positive_target,
            target * (torch.log(target) - torch.log(mixture)),
            torch.zeros_like(target),
        )) + 0.5 * torch.sum(model * (torch.log(model) - torch.log(mixture)))
        kl = torch.sum(torch.where(
            positive_target,
            target * (torch.log(target) - torch.log(model)),
            torch.zeros_like(target),
        ))
        model_cdf_centers = self._cdf(self.unit_centers, knots, heights)
        cdf_difference = model_cdf_centers - self.target_cdf_centers
        cdf_l2 = torch.sum(cdf_difference * cdf_difference * self.return_widths)
        return {
            "js": js / LOG_2,
            "kl": kl / LOG_2,
            "cdfL2": cdf_l2,
            "cdfL2Normalized": cdf_l2 / self.target.active_standard_deviation_bps,
        }

    def _objective(
        self,
        gap_logits: torch.Tensor,
        weight_logits: torch.Tensor,
        objective: ObjectiveName,
        joint_scales: dict[str, float] | None,
    ) -> torch.Tensor:
        knots, _, heights = self._decode(gap_logits, weight_logits)
        metrics = self._raw_metrics(knots, heights)
        if objective == "js":
            return metrics["js"]
        if objective == "kl":
            return metrics["kl"]
        if objective == "cdfL2":
            return metrics["cdfL2Normalized"]
        if joint_scales is None:
            raise ValueError("Joint optimization requires metric scales.")
        return (
            metrics["js"] / max(joint_scales["js"], 1e-15)
            + metrics["kl"] / max(joint_scales["kl"], 1e-15)
            + metrics["cdfL2Normalized"] / max(joint_scales["cdfL2"], 1e-15)
        ) / 3.0

    def tensors_for_state(self, state: FitState) -> tuple[torch.Tensor, torch.Tensor]:
        knots = torch.as_tensor(state.knots, dtype=self.dtype, device=self.device)
        heights = torch.as_tensor(
            state.knot_density_heights, dtype=self.dtype, device=self.device,
        )
        return knots, heights

    def metrics_for_state(self, state: FitState) -> dict[str, float]:
        knots, heights = self.tensors_for_state(state)
        raw = self._raw_metrics(knots, heights)
        model = self._model_probabilities(knots, heights)
        target = self.target_probabilities
        target_cdf = torch.cumsum(target, dim=0)
        model_cdf = torch.cumsum(model, dim=0)
        active_probability = self.target.active_observations / self.target.observations
        metrics = {
            "jsBits": float(raw["js"].detach().cpu()),
            "klBits": float(raw["kl"].detach().cpu()),
            "cdfL2Bps": float(raw["cdfL2"].detach().cpu()),
            "cdfL2Normalized": float(raw["cdfL2Normalized"].detach().cpu()),
            "totalVariation": float((0.5 * torch.sum(torch.abs(target - model))).detach().cpu()),
            "maximumCdfError": float(torch.max(torch.abs(target_cdf - model_cdf)).detach().cpu()),
        }
        metrics["fullDistributionJsBits"] = active_probability * metrics["jsBits"]
        metrics["fullDistributionKlBits"] = active_probability * metrics["klBits"]
        metrics["fullDistributionCdfL2Bps"] = active_probability**2 * metrics["cdfL2Bps"]
        return metrics

    def probabilities_for_state(self, state: FitState) -> np.ndarray:
        knots, heights = self.tensors_for_state(state)
        return self._model_probabilities(knots, heights).detach().cpu().numpy()

def build_artifact(
    analysis_path: Path,
    fits_path: Path,
    cache_path: Path,
    analysis: dict[str, Any],
    transform: StaticTransform,
    target: TargetHistogram,
    args: argparse.Namespace,
    states: dict[str, FitState],
    metrics: dict[str, dict[str, float]],
    pairwise: dict[str, Any],
    joint_scales: dict[str, float],
) -> dict[str, Any]:
    del analysis_path, fits_path, cache_path
    active_probability = target.active_observations / target.observations
    return {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": analysis["source"]["symbol"],
        "scale": "1s",
        "window": {
            "id": analysis["fullHistory"]["id"],
            "startTime": analysis["fullHistory"]["startTime"],
            "endTime": analysis["fullHistory"]["endTime"],
        },
        "source": {
            "analysis": str(args.analysis).replace("\\", "/"),
            "distributionFits": str(args.fits).replace("\\", "/"),
            "oneSecondReferences": analysis["source"]["oneSecond"]["referenceDirectory"],
        },
        "target": {
            "returnDefinition": "natural log of adjacent 1s closes, in basis points",
            "zeroTreatment": "exact zeros excluded from the continuous fit and retained as a separate point mass",
            "observations": target.observations,
            "activeObservations": target.active_observations,
            "zeroObservations": target.zero_observations,
            "zeroProbability": target.zero_observations / target.observations,
            "activeProbability": active_probability,
            "activeStandardDeviationBps": target.active_standard_deviation_bps,
            "histogram": {
                "coordinate": "latent=alpha*asinh((return-mu)/scale), then unit=sigmoid(latent)",
                "coreBins": args.core_bins,
                "latentLimit": args.latent_limit,
                "tailBins": 2,
                "optimizationStride": args.optimization_stride,
                "optimizationCoreBins": args.core_bins // args.optimization_stride,
                "finalAuditCoreBins": args.core_bins,
            },
        },
        "transform": {
            "family": "asinh-logistic",
            "trainable": False,
            "alpha": transform.alpha,
            "locationBps": transform.location_bps,
            "scaleBps": transform.scale_bps,
            "forward": "u=sigmoid(alpha*asinh((r-mu)/scale))",
            "inverse": "r=mu+scale*sinh(logit(u)/alpha)",
            "scaleSelection": "KL projection of the fitted full-history symmetric generalized-t onto the asinh-logistic family with alpha and mu fixed",
        },
        "reconstruction": {
            "family": "continuous piecewise-linear density",
            "knotCount": args.knots,
            "normalization": "positive triangular component weights sum to one; knot heights equal component weight divided by triangular basis area",
            "support": "unit interval [0,1], mapping to the complete real return line",
        },
        "objectives": {
            "js": "Jensen-Shannon divergence in bits on transformed histogram probability masses",
            "kl": "forward KL(target||fit) in bits on transformed histogram probability masses",
            "cdfL2": "integral of squared CDF error over original return bps, normalized by active-return standard deviation during optimization",
            "joint": {
                "definition": "equal mean of each metric divided by its independently optimized minimum",
                "scales": joint_scales,
            },
        },
        "fits": {
            name: {
                "objectiveValue": state.objective_value,
                "metrics": metrics[name],
                "knotsUnit": state.knots.tolist(),
                "knotsReturnBps": [
                    None,
                    *transform.inverse(state.knots[1:-1]).tolist(),
                    None,
                ],
                "componentWeights": state.component_weights.tolist(),
                "knotDensityHeights": state.knot_density_heights.tolist(),
                "convergence": state.convergence,
            }
            for name, state in states.items()
        },
        "pairwiseFitDifferences": pairwise,
        "limitations": [
            f"The optimum is numerical with {args.core_bins // args.optimization_stride} optimization bins, full-resolution refinement and audit on {args.core_bins} bins, and multiple deterministic starts; freely moving triangular knots form a non-convex continuous problem, so a proof of the global continuous optimum is not claimed.",
            "This is an in-sample whole-history representation audit, not a forecast evaluation.",
            "For chronological forecasting, the transform, knots, and baseline weights must be fitted only on the historical prefix available at the forecast origin.",
        ],
    }


def pairwise_with_transform(
    states: dict[str, FitState],
    target: TargetHistogram,
    transform: StaticTransform,
) -> dict[str, Any]:
    names = list(states)
    probabilities: dict[str, np.ndarray] = {}
    edges = torch.as_tensor(target.unit_edges, dtype=torch.float64)
    for name, state in states.items():
        probabilities[name] = model_probabilities_numpy(state, edges)
    result: dict[str, Any] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            left = probabilities[left_name]
            right = probabilities[right_name]
            mixture = (left + right) / 2.0
            js = 0.5 * np.sum(left * np.log(left / mixture)) \
                + 0.5 * np.sum(right * np.log(right / mixture))
            left_cdf = np.cumsum(left)
            right_cdf = np.cumsum(right)
            cdf_mid_difference = left_cdf - left / 2.0 - right_cdf + right / 2.0
            left_knots = states[left_name].knots[1:-1]
            right_knots = states[right_name].knots[1:-1]
            left_returns = transform.inverse(left_knots)
            right_returns = transform.inverse(right_knots)
            key = f"{left_name}Vs{right_name[0].upper()}{right_name[1:]}"
            result[key] = {
                "jsBits": float(js / LOG_2),
                "totalVariation": float(0.5 * np.sum(np.abs(left - right))),
                "maximumCdfDifference": float(np.max(np.abs(left_cdf - right_cdf))),
                "cdfL2Bps": float(np.sum(
                    cdf_mid_difference * cdf_mid_difference * target.return_widths_bps,
                )),
                "interiorKnotUnitRmsDifference": float(np.sqrt(np.mean(
                    (left_knots - right_knots) ** 2,
                ))),
                "interiorKnotReturnMedianAbsoluteDifferenceBps": float(np.median(
                    np.abs(left_returns - right_returns),
                )),
                "interiorKnotReturnMaximumAbsoluteDifferenceBps": float(np.max(
                    np.abs(left_returns - right_returns),
                )),
            }
    return result


def model_probabilities_numpy(state: FitState, edges: torch.Tensor) -> np.ndarray:
    knots = torch.as_tensor(state.knots, dtype=torch.float64)
    heights = torch.as_tensor(state.knot_density_heights, dtype=torch.float64)
    gaps = knots[1:] - knots[:-1]
    areas = gaps * (heights[:-1] + heights[1:]) / 2.0
    prefix = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(areas, dim=0)))
    interval = torch.bucketize(edges.contiguous(), knots[1:-1].contiguous())
    distance = edges - knots[interval]
    slope = (heights[interval + 1] - heights[interval]) / gaps[interval]
    cdf = prefix[interval] + heights[interval] * distance + 0.5 * slope * distance * distance
    probabilities = torch.clamp(cdf[1:] - cdf[:-1], min=1e-300)
    return probabilities.numpy()


def render_report(artifact: dict[str, Any]) -> str:
    fits = artifact["fits"]
    lines = [
        "# Static 64-knot one-second return-density fits",
        "",
        f"Generated {artifact['generatedAt']}. The fit covers nonzero BTCUSDT 1-second log returns from "
        f"{artifact['window']['startTime']} through {artifact['window']['endTime']}; the exact-zero mass "
        f"of {artifact['target']['zeroProbability']:.6%} remains separate.",
        "",
        "## Method",
        "",
        "The return transform is frozen:",
        "",
        "```text",
        f"u = sigmoid({artifact['transform']['alpha']:.9f} * asinh((r - ({artifact['transform']['locationBps']:.9f} bps)) / {artifact['transform']['scaleBps']:.9f} bps))",
        "```",
        "",
        "Each fit uses 64 movable knots on the complete unit interval and a normalized continuous piecewise-linear density. The four fits optimize JS bits, forward KL bits, original-return CDF-L2, and the equal relative joint objective respectively.",
        "",
        "## Accuracy",
        "",
        "| optimized for | JS (bits) | KL (bits) | CDF-L2 (bps) | normalized CDF-L2 | total variation | max CDF error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("js", "kl", "cdfL2", "joint"):
        metrics = fits[name]["metrics"]
        lines.append(
            f"| {name} | {metrics['jsBits']:.9f} | {metrics['klBits']:.9f} | "
            f"{metrics['cdfL2Bps']:.9g} | {metrics['cdfL2Normalized']:.9g} | "
            f"{metrics['totalVariation']:.9f} | {metrics['maximumCdfError']:.9f} |",
        )
    lines.extend([
        "",
        "The full-distribution equivalents include the common exact-zero point mass. Because that mass is identical in every target and fit, full JS and KL equal their active values times the active probability; CDF-L2 is multiplied by the squared active probability.",
        "",
        "## Pairwise fit differences",
        "",
        "| pair | JS (bits) | total variation | max CDF difference | CDF-L2 (bps) | median knot difference (bps) | max knot difference (bps) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, values in artifact["pairwiseFitDifferences"].items():
        lines.append(
            f"| {name} | {values['jsBits']:.9f} | {values['totalVariation']:.9f} | "
            f"{values['maximumCdfDifference']:.9f} | {values['cdfL2Bps']:.9g} | "
            f"{values['interiorKnotReturnMedianAbsoluteDifferenceBps']:.9g} | "
            f"{values['interiorKnotReturnMaximumAbsoluteDifferenceBps']:.9g} |",
        )
    best_by_metric = {
        metric: min(fits, key=lambda name: fits[name]["metrics"][metric])
        for metric in ("jsBits", "klBits", "cdfL2Bps")
    }
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Lowest JS: `{best_by_metric['jsBits']}`.",
        f"- Lowest KL: `{best_by_metric['klBits']}`.",
        f"- Lowest CDF-L2: `{best_by_metric['cdfL2Bps']}`.",
        "- Pairwise JS and total variation measure differences between the reconstructed probability laws; knot-coordinate differences alone can exaggerate functional differences because different knot layouts may reconstruct nearly the same density.",
        "- This is an in-sample representation audit. It does not measure whether a conditional history model forecasts the next return distribution.",
        "",
        "## Reproducibility",
        "",
        "```text",
        "node scripts/run-ml-python.mjs ml/fit_return_density_knots.py",
        "```",
        "",
        "The machine-readable knots, weights, density heights, convergence details, and all cross-metrics are stored in `data/benchmarks/one-second-return-64-knot-fits.json`.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
