from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fit_teacher_cuda import FitConfig, fit_batch, sampled_indices
from mlp_model import PolicySupport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the queued CUDA teacher fitter.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ml-analysis/mlp-rejection-oracles"),
    )
    parser.add_argument("--examples", type=int, default=2880)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-actions", type=int)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--no-quality-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples < 1 or args.repetitions < 1:
        raise ValueError("examples and repetitions must be positive")
    metadata = json.loads((args.input / "metadata.json").read_text())
    plan = json.loads(Path("ml/training-plan.json").read_text())["teacherFit"]
    actions_list = metadata["actionGrid"]
    currents_list = metadata["currentGrid"]
    source = np.fromfile(
        args.input / metadata["probabilitiesFile"], dtype="<f4"
    ).reshape(-1, len(actions_list))
    repeats = (args.examples + source.shape[0] - 1) // source.shape[0]
    base_values = np.tile(source, (repeats, 1))[:args.examples].copy()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    base = torch.from_numpy(base_values).to(device)
    cutoff = torch.tensor([-14.0, 14.0], device=device).expand(base.shape[0], -1)
    packed = torch.cat((base, cutoff), dim=-1)
    actions = torch.tensor(actions_list, dtype=torch.float32, device=device)
    currents = torch.tensor(currents_list, dtype=torch.float32, device=device)
    execution = metadata["execution"]
    config = FitConfig(
        action_grid=actions_list,
        current_grid=currents_list,
        friction=execution["feeBps"] / 10_000,
        transition_log_scale=1 / execution["temperature"],
        latent_lower=execution["minimumEffectiveExposure"],
        latent_upper=execution["maximumEffectiveExposure"],
        visible_lower=execution["minimumEffectiveExposure"],
        visible_upper=execution["maximumEffectiveExposure"],
        metric_visible_lower=execution["minimumUsableExposure"],
        metric_visible_upper=execution["maximumUsableExposure"],
        sample_states=plan["sampleStates"],
        sample_actions=args.sample_actions or plan["sampleActions"],
        projection_iterations=plan["projectionIterations"],
        iterations=plan["maxIterations"],
        adaptive_iterations=plan["adaptiveIterations"],
        adaptive_rounds=plan["adaptiveRounds"],
        restarts=plan["restartCount"],
        batch_size=args.examples,
        tolerance=plan["tolerance"],
        max_mean_kl=plan["maxMeanKlDivergence"],
        max_mean_mse=plan["maxMeanSquaredError"],
        line_search_candidates=plan["lineSearchCandidates"],
        temporal_refinement_rounds=plan["temporalRefinementRounds"]
        if args.temporal else 0,
        temporal_iterations=plan["temporalIterations"],
        temporal_followup_iterations=plan["temporalFollowupIterations"],
        temporal_equivalent_loss_absolute=plan["temporalEquivalentLossAbsolute"],
        temporal_equivalent_loss_relative=plan["temporalEquivalentLossRelative"],
        optimizer_backend=plan["optimizerBackend"],
        optimizer_host_check_interval=plan["optimizerHostCheckInterval"],
        quality_fallback_iterations=0 if args.no_quality_fallback
        else plan["qualityFallbackIterations"],
    )
    support = PolicySupport(
        config.latent_lower,
        config.latent_upper,
        config.visible_lower,
        config.visible_upper,
        config.friction,
        1 / config.transition_log_scale,
    )
    state_indexes = sampled_indices(currents.numel(), config.sample_states, device)
    action_indexes = sampled_indices(actions.numel(), config.sample_actions, device)
    sampled_actions = actions[action_indexes]
    sampled_currents = currents[state_indexes]

    # Exclude one-time Triton JIT and allocator setup from steady-state results.
    fit_batch(
        packed[:1], actions, currents, sampled_actions, sampled_currents,
        action_indexes, support, replace(config, temporal_refinement_rounds=0),
    )
    durations: list[float] = []
    latest = None
    for repetition in range(args.repetitions):
        torch.manual_seed(1337)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        latest = fit_batch(
            packed, actions, currents, sampled_actions, sampled_currents,
            action_indexes, support, config,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        durations.append(duration)
        print(json.dumps({
            "event": "teacher-benchmark-repetition",
            "repetition": repetition + 1,
            "seconds": duration,
            "fitsPerSecond": args.examples / duration,
        }), flush=True)
    assert latest is not None
    _, diagnostics, _, converged, temporal = latest
    rejected = (
        (diagnostics["klDivergence"] > config.max_mean_kl)
        | (diagnostics["meanSquaredError"] > config.max_mean_mse)
    )
    median_seconds = float(np.median(np.asarray(durations)))
    print(json.dumps({
        "event": "teacher-benchmark-complete",
        "examples": args.examples,
        "repetitions": args.repetitions,
        "medianSeconds": median_seconds,
        "medianFitsPerSecond": args.examples / median_seconds,
        "peakAllocatedMiB": torch.cuda.max_memory_allocated(device) / 1_048_576
        if device.type == "cuda" else 0,
        "peakReservedMiB": torch.cuda.max_memory_reserved(device) / 1_048_576
        if device.type == "cuda" else 0,
        "meanKlDivergence": float(diagnostics["klDivergence"].mean()),
        "meanSquaredError": float(diagnostics["meanSquaredError"].mean()),
        "rejected": int(rejected.sum()),
        "converged": int(converged.sum()),
        "temporal": temporal,
    }), flush=True)


if __name__ == "__main__":
    main()
