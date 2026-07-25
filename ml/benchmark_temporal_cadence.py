from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fit_teacher_cuda import (
    FitConfig,
    PolicySupport,
    batched_bfgs,
    fit_batch,
    fit_diagnostics,
    fit_target,
    metric_surface,
    domain_sampled_indices,
    score_parameter_mask,
    temporal_warm_refinement,
    transition_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure temporal teacher-fit acceptance at real timestamp cadences."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("ml/training-plan.json"))
    parser.add_argument("--cadences", default="1,5,15,30,60,300,1800")
    parser.add_argument("--chain-iterations", default="20,64,128")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.input / "metadata.json").read_text())
    plan = json.loads(args.plan.read_text())
    fit_plan = plan["teacherFit"]
    execution = plan["execution"]
    count = int(metadata["count"])
    row_stride = int(metadata["rowStride"])
    action_grid = metadata["actionGrid"]
    current_grid = metadata["currentGrid"]
    packed_np = np.memmap(
        args.input / metadata["inputFile"],
        mode="r",
        dtype="<f4",
        shape=(count, row_stride),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    packed = torch.from_numpy(np.array(packed_np, copy=True)).to(device)
    actions = torch.tensor(action_grid, dtype=torch.float32, device=device)
    currents = torch.tensor(current_grid, dtype=torch.float32, device=device)
    config = FitConfig(
        action_grid=action_grid,
        current_grid=current_grid,
        friction=execution["feeBps"] / 10_000,
        transition_log_scale=1 / execution["temperature"],
        latent_lower=execution["minimumEffectiveExposure"],
        latent_upper=execution["maximumEffectiveExposure"],
        visible_lower=execution["minimumEffectiveExposure"],
        visible_upper=execution["maximumEffectiveExposure"],
        metric_visible_lower=execution["minimumUsableExposure"],
        metric_visible_upper=execution["maximumUsableExposure"],
        sample_states=fit_plan["sampleStates"],
        sample_actions=fit_plan["sampleActions"],
        projection_iterations=fit_plan["projectionIterations"],
        iterations=fit_plan["maxIterations"],
        adaptive_iterations=fit_plan["adaptiveIterations"],
        adaptive_rounds=fit_plan["adaptiveRounds"],
        restarts=fit_plan["restartCount"],
        batch_size=fit_plan["batchSize"],
        tolerance=fit_plan["tolerance"],
        max_mean_kl=fit_plan["maxMeanKlDivergence"],
        max_mean_mse=fit_plan["maxMeanSquaredError"],
        line_search_candidates=fit_plan["lineSearchCandidates"],
        temporal_refinement_rounds=fit_plan["temporalRefinementRounds"],
        temporal_iterations=fit_plan["temporalIterations"],
        temporal_followup_iterations=fit_plan["temporalFollowupIterations"],
        temporal_equivalent_loss_absolute=fit_plan["temporalEquivalentLossAbsolute"],
        temporal_equivalent_loss_relative=fit_plan["temporalEquivalentLossRelative"],
        optimizer_backend=fit_plan["optimizerBackend"],
        optimizer_host_check_interval=fit_plan["optimizerHostCheckInterval"],
        quality_fallback_iterations=fit_plan["qualityFallbackIterations"],
        input_row_stride=row_stride,
        visible_sample_fraction=fit_plan["visibleSampleFraction"],
        score_hinge_span=fit_plan["scoreHingeSpan"],
        compact_visible_initialization=fit_plan["compactVisibleInitialization"],
    )
    support = PolicySupport(
        config.latent_lower,
        config.latent_upper,
        config.visible_lower,
        config.visible_upper,
        config.friction,
        1 / config.transition_log_scale,
        config.score_hinge_span,
    )
    state_indexes = domain_sampled_indices(
        currents, config.sample_states, config.metric_visible_lower,
        config.metric_visible_upper, config.visible_sample_fraction,
    )
    action_indexes = domain_sampled_indices(
        actions, config.sample_actions, config.metric_visible_lower,
        config.metric_visible_upper, config.visible_sample_fraction,
    )
    sampled_actions = actions[action_indexes]
    sampled_currents = currents[state_indexes]

    # Compile/JIT setup is not part of the cadence comparison.
    fit_batch(
        packed[:1],
        actions,
        currents,
        sampled_actions,
        sampled_currents,
        action_indexes,
        support,
        replace(
            config,
            temporal_refinement_rounds=0,
            adaptive_rounds=0,
            quality_fallback_iterations=0,
        ),
    )
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    started = time.perf_counter()
    raw_parts: list[torch.Tensor] = []
    convergence_parts: list[torch.Tensor] = []
    diagnostic_parts: dict[str, list[torch.Tensor]] = {
        "crossEntropy": [],
        "klDivergence": [],
        "meanSquaredError": [],
    }
    direct_config = replace(config, temporal_refinement_rounds=0)
    for start in range(0, count, config.batch_size):
        end = min(count, start + config.batch_size)
        raw, diagnostic, _, converged, _ = fit_batch(
            packed[start:end],
            actions,
            currents,
            sampled_actions,
            sampled_currents,
            action_indexes,
            support,
            direct_config,
        )
        raw_parts.append(raw)
        convergence_parts.append(converged)
        for name in diagnostic_parts:
            diagnostic_parts[name].append(diagnostic[name])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    direct_seconds = time.perf_counter() - started
    independent_raw = torch.cat(raw_parts)
    independent_converged = torch.cat(convergence_parts)
    independent_diagnostic = {
        name: torch.cat(parts) for name, parts in diagnostic_parts.items()
    }

    base = packed[:, :actions.numel()]
    target = torch.softmax(
        transition_logits(
            base[:, action_indexes],
            sampled_actions,
            sampled_currents,
            config.friction,
            config.transition_log_scale,
        ),
        dim=-1,
    )
    diagnostic_base, diagnostic_actions, diagnostic_currents = metric_surface(
        base, actions, currents, config
    )
    diagnostic_target, diagnostic_entropy = fit_target(
        diagnostic_base, diagnostic_actions, diagnostic_currents, config
    )
    direct_rejected = (
        (independent_diagnostic["klDivergence"] > config.max_mean_kl)
        | (independent_diagnostic["meanSquaredError"] > config.max_mean_mse)
    )
    emit({
        "event": "temporal-cadence-direct-baseline",
        "date": metadata["date"],
        "startTime": metadata["startTime"],
        "examples": count,
        "seconds": direct_seconds,
        "fitsPerSecond": count / direct_seconds,
        "rejected": int(direct_rejected.sum()),
        "meanKlDivergence": float(independent_diagnostic["klDivergence"].mean()),
        "meanSquaredError": float(independent_diagnostic["meanSquaredError"].mean()),
    })

    for chain_iterations in sorted({
        int(value) for value in args.chain_iterations.split(",")
    }):
        if chain_iterations < 1:
            raise ValueError("chain iterations must be positive integers")
        benchmark_minute_chains(
            independent_raw,
            independent_converged,
            independent_diagnostic,
            target,
            sampled_actions,
            sampled_currents,
            diagnostic_target,
            diagnostic_entropy,
            diagnostic_actions,
            diagnostic_currents,
            support,
            config,
            device,
            chain_iterations,
        )

    cadences = sorted({int(value) for value in args.cadences.split(",")})
    for cadence in cadences:
        if cadence < 1:
            raise ValueError("cadences must be positive integer seconds")
        indexes = torch.arange(0, count, cadence, device=device)
        if indexes.numel() < 2:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        selected, selected_converged, iterations, temporal = temporal_warm_refinement(
            independent_raw[indexes],
            independent_converged[indexes],
            target[indexes],
            sampled_actions,
            sampled_currents,
            diagnostic_target[indexes],
            diagnostic_actions,
            diagnostic_currents,
            independent_diagnostic["crossEntropy"][indexes],
            support,
            config,
        )
        selected_diagnostic = fit_diagnostics(
            diagnostic_target[indexes],
            diagnostic_entropy[indexes],
            selected,
            diagnostic_actions,
            diagnostic_currents,
            support,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        selected_rejected = (
            (selected_diagnostic["klDivergence"] > config.max_mean_kl)
            | (selected_diagnostic["meanSquaredError"] > config.max_mean_mse)
        )
        emit({
            "event": "temporal-cadence-result",
            "cadenceSeconds": cadence,
            "examples": int(indexes.numel()),
            "candidateExamples": int(temporal["candidateCount"]),
            "seconds": seconds,
            "candidateFitsPerSecond": temporal["candidateCount"] / max(seconds, 1e-9),
            "temporalIterations": iterations,
            "warmSelectedFraction": temporal["selectedFraction"],
            "warmEquivalentFraction": temporal["equivalentFraction"],
            "warmQualityAcceptedFraction": temporal["qualityAcceptedFraction"],
            "warmQualityBetterFraction": temporal["qualityBetterFraction"],
            "selectedRejected": int(selected_rejected.sum()),
            "selectedConverged": int(selected_converged.sum()),
            "meanKlDivergence": float(selected_diagnostic["klDivergence"].mean()),
            "meanSquaredError": float(selected_diagnostic["meanSquaredError"].mean()),
            "meanNormalizedStepBefore": temporal["meanNormalizedStepBefore"],
            "meanNormalizedStepAfter": temporal["meanNormalizedStepAfter"],
            "medianNormalizedStepBefore": temporal["medianNormalizedStepBefore"],
            "medianNormalizedStepAfter": temporal["medianNormalizedStepAfter"],
        })


def benchmark_minute_chains(
    independent_raw: torch.Tensor,
    independent_converged: torch.Tensor,
    independent_diagnostic: dict[str, torch.Tensor],
    target: torch.Tensor,
    sampled_actions: torch.Tensor,
    sampled_currents: torch.Tensor,
    diagnostic_target: torch.Tensor,
    diagnostic_entropy: torch.Tensor,
    diagnostic_actions: torch.Tensor,
    diagnostic_currents: torch.Tensor,
    support: PolicySupport,
    config: FitConfig,
    device: torch.device,
    maximum_iterations: int,
) -> None:
    """Propagate true previous-second fits in parallel across minute segments."""
    count = independent_raw.shape[0]
    chained = independent_raw.clone()
    chained_converged = independent_converged.clone()
    equivalent_total = 0
    accepted_total = 0
    better_total = 0
    candidate_total = 0
    equivalent_by_second: list[float] = []
    accepted_by_second: list[float] = []
    iterations_total = 0
    full_mask = score_parameter_mask(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for second in range(1, 60):
        indexes = torch.arange(second, count, 60, device=device)
        if indexes.numel() == 0:
            equivalent_by_second.append(0.0)
            accepted_by_second.append(0.0)
            continue
        previous = chained[indexes - 1]
        initial = torch.cat((
            previous[:, :6],
            independent_raw[indexes, 6:],
        ), dim=-1)
        warm, _, warm_converged, iterations = batched_bfgs(
            initial[:, None, :],
            target[indexes],
            sampled_actions,
            sampled_currents,
            support,
            maximum_iterations,
            config.tolerance,
            full_mask,
            config.line_search_candidates,
            config.optimizer_backend,
            config.optimizer_host_check_interval,
        )
        iterations_total += iterations
        warm = warm[:, 0, :]
        warm_diagnostic = fit_diagnostics(
            diagnostic_target[indexes],
            diagnostic_entropy[indexes],
            warm,
            diagnostic_actions,
            diagnostic_currents,
            support,
        )
        reference = independent_diagnostic["crossEntropy"][indexes]
        tolerance = config.temporal_equivalent_loss_absolute \
            + config.temporal_equivalent_loss_relative * reference.abs()
        equivalent = warm_diagnostic["crossEntropy"] <= reference + tolerance
        accepted = (
            (warm_diagnostic["klDivergence"] <= config.max_mean_kl)
            & (warm_diagnostic["meanSquaredError"] <= config.max_mean_mse)
        )
        better = warm_diagnostic["crossEntropy"] < reference - tolerance
        chained[indexes] = warm
        chained_converged[indexes] = warm_converged[:, 0]
        rows = int(indexes.numel())
        candidate_total += rows
        equivalent_total += int(equivalent.sum())
        accepted_total += int(accepted.sum())
        better_total += int(better.sum())
        equivalent_by_second.append(float(equivalent.float().mean()))
        accepted_by_second.append(float(accepted.float().mean()))
    chained_diagnostic = fit_diagnostics(
        diagnostic_target,
        diagnostic_entropy,
        chained,
        diagnostic_actions,
        diagnostic_currents,
        support,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    rejected = (
        (chained_diagnostic["klDivergence"] > config.max_mean_kl)
        | (chained_diagnostic["meanSquaredError"] > config.max_mean_mse)
    )
    emit({
        "event": "temporal-minute-chain-result",
        "anchorCadenceSeconds": 60,
        "examples": count,
        "directAnchors": (count + 59) // 60,
        "warmOnlyCandidates": candidate_total,
        "seconds": seconds,
        "warmOnlyFitsPerSecond": candidate_total / max(seconds, 1e-9),
        "temporalIterationsPerFit": maximum_iterations,
        "temporalIterations": iterations_total,
        "warmEquivalentFraction": equivalent_total / max(1, candidate_total),
        "warmQualityAcceptedFraction": accepted_total / max(1, candidate_total),
        "warmQualityBetterFraction": better_total / max(1, candidate_total),
        "selectedRejected": int(rejected.sum()),
        "selectedConverged": int(chained_converged.sum()),
        "meanKlDivergence": float(chained_diagnostic["klDivergence"].mean()),
        "meanSquaredError": float(chained_diagnostic["meanSquaredError"].mean()),
        "equivalentBySecond": equivalent_by_second,
        "qualityAcceptedBySecond": accepted_by_second,
    })


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
