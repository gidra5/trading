from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from fit_teacher_cuda import (
    METRIC_COUNT,
    FitConfig,
    domain_sampled_indices,
    fit_batch,
    run_job,
)
from mlp_model import PolicySupport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the queued CUDA teacher fitter.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/training/analysis/mlp-rejection-oracles"),
    )
    parser.add_argument("--examples", type=int, default=2880)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-actions", type=int)
    parser.add_argument(
        "--effective-action-grid",
        action="store_true",
        help="Interpret corpus columns on the plan's complete effective grid.",
    )
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--no-quality-fallback", action="store_true")
    parser.add_argument(
        "--accept-all",
        action="store_true",
        help="Disable quality-triggered adaptive and fallback work for an upper throughput bound.",
    )
    parser.add_argument(
        "--pipeline-batches",
        type=int,
        default=0,
        help="Run this many batches through the production pipelined worker path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples < 1 or args.repetitions < 1 or args.pipeline_batches < 0:
        raise ValueError("examples and repetitions must be positive")
    metadata = json.loads((args.input / "metadata.json").read_text())
    plan = json.loads(Path("ml/training-plan.json").read_text())["teacherFit"]
    execution = metadata["execution"]
    actions_list = metadata["actionGrid"]
    if args.effective_action_grid:
        actions_list = np.linspace(
            execution["minimumEffectiveExposure"],
            execution["maximumEffectiveExposure"],
            len(actions_list),
        ).tolist()
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
        max_mean_kl=1e9 if args.accept_all else plan["maxMeanKlDivergence"],
        max_mean_mse=1e9 if args.accept_all else plan["maxMeanSquaredError"],
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
        visible_sample_fraction=plan["visibleSampleFraction"],
        score_hinge_span=plan["scoreHingeSpan"],
        compact_visible_initialization=plan["compactVisibleInitialization"],
    )
    if args.pipeline_batches > 0:
        benchmark_pipeline(
            args,
            source,
            config,
            plan,
        )
        return
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


def benchmark_pipeline(
    args: argparse.Namespace,
    source: np.ndarray,
    config: FitConfig,
    plan: dict,
) -> None:
    total_examples = args.examples * args.pipeline_batches
    repeats = (total_examples + source.shape[0] - 1) // source.shape[0]
    base = np.tile(source, (repeats, 1))[:total_examples]
    alignment = int(plan["inputAlignmentFloats"])
    row_stride = (
        (base.shape[1] + 2 + alignment - 1) // alignment
    ) * alignment
    packed = np.zeros((total_examples, row_stride), dtype="<f4")
    packed[:, :base.shape[1]] = base
    packed[:, base.shape[1]:base.shape[1] + 2] = (-14.0, 14.0)
    pipeline_config = replace(
        config,
        batch_size=args.examples,
        input_row_stride=row_stride,
        input_queue_batches=int(plan["inputQueueBatches"]),
        pipelined_refinement=True,
    )
    durations: list[float] = []
    latest_metrics: np.ndarray | None = None
    with tempfile.TemporaryDirectory(prefix="mlp-teacher-pipeline-") as directory_value:
        directory = Path(directory_value)
        input_path = directory / "input.f32"
        config_path = directory / "config.json"
        parameter_path = directory / "parameters.f32"
        metric_path = directory / "metrics.f32"
        packed.tofile(input_path)
        config_path.write_text(json.dumps(asdict(pipeline_config)))
        for repetition in range(args.repetitions):
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(args.device)
                torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            run_job(argparse.Namespace(
                input=input_path,
                count=total_examples,
                config=config_path,
                parameters_output=parameter_path,
                metrics_output=metric_path,
                device=args.device,
            ))
            duration = time.perf_counter() - started
            durations.append(duration)
            latest_metrics = np.fromfile(metric_path, dtype="<f4").reshape(-1, METRIC_COUNT)
            print(json.dumps({
                "event": "teacher-pipeline-benchmark-repetition",
                "batchSize": args.examples,
                "batches": args.pipeline_batches,
                "examples": total_examples,
                "repetition": repetition + 1,
                "seconds": duration,
                "fitsPerSecond": total_examples / duration,
            }), flush=True)
    assert latest_metrics is not None
    median_seconds = float(np.median(np.asarray(durations)))
    rejected = (
        (latest_metrics[:, 1] > pipeline_config.max_mean_kl)
        | (latest_metrics[:, 2] > pipeline_config.max_mean_mse)
    )
    print(json.dumps({
        "event": "teacher-pipeline-benchmark-complete",
        "batchSize": args.examples,
        "batches": args.pipeline_batches,
        "examples": total_examples,
        "repetitions": args.repetitions,
        "medianSeconds": median_seconds,
        "medianFitsPerSecond": total_examples / median_seconds,
        "peakAllocatedMiB": torch.cuda.max_memory_allocated(args.device) / 1_048_576
        if args.device.startswith("cuda") else 0,
        "peakReservedMiB": torch.cuda.max_memory_reserved(args.device) / 1_048_576
        if args.device.startswith("cuda") else 0,
        "meanKlDivergence": float(latest_metrics[:, 1].mean()),
        "meanSquaredError": float(latest_metrics[:, 2].mean()),
        "rejected": int(rejected.sum()),
    }), flush=True)


if __name__ == "__main__":
    main()
