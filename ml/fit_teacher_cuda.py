from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mlp_model import (
    PolicySupport,
    conditional_policy_logits,
    distance_imbalance_advice,
    scaled_softplus,
)


METRIC_COUNT = 7
PARAMETER_COUNT = 8
SCORE_PARAMETER_COUNT = 6
NVTX_ENABLED = os.environ.get("TRADING_MLP_TEACHER_NVTX", "0") == "1"


def nvtx_range(name: str, value: Tensor):
    if NVTX_ENABLED and value.is_cuda:
        return torch.cuda.nvtx.range(name)
    return nullcontext()


@dataclass(frozen=True)
class FitConfig:
    action_grid: list[float]
    current_grid: list[float]
    friction: float
    transition_log_scale: float
    latent_lower: float
    latent_upper: float
    visible_lower: float
    visible_upper: float
    metric_visible_lower: float
    metric_visible_upper: float
    sample_states: int
    sample_actions: int
    projection_iterations: int
    iterations: int
    adaptive_iterations: int
    adaptive_rounds: int
    restarts: int
    batch_size: int
    tolerance: float
    max_mean_kl: float
    max_mean_mse: float
    line_search_candidates: int = 0
    temporal_refinement_rounds: int = 0
    temporal_iterations: int = 4
    temporal_followup_iterations: int = 0
    temporal_equivalent_loss_absolute: float = 2e-5
    temporal_equivalent_loss_relative: float = 2e-5
    optimizer_backend: str = "pytorch-batched"
    optimizer_host_check_interval: int = 32
    quality_fallback_iterations: int = 0
    input_row_stride: int = 0
    input_queue_batches: int = 2
    pipelined_refinement: bool = False
    distance_epsilon: float = 1e-6
    visible_sample_fraction: float = 0.0
    score_hinge_span: float = 0.0
    compact_visible_initialization: bool = False


@dataclass
class FitBatchState:
    final_raw: Tensor
    diagnostic: dict[str, Tensor]
    converged: Tensor
    total_iterations: int
    target: Tensor
    diagnostic_target: Tensor
    diagnostic_entropy: Tensor
    diagnostic_actions: Tensor
    diagnostic_currents: Tensor
    sampled_actions: Tensor
    sampled_currents: Tensor
    support: PolicySupport
    config: FitConfig


@dataclass(frozen=True)
class ProjectionTargetStatistics:
    weights: Tensor
    weight_total: Tensor
    centered_scores: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched CUDA conditional-policy teacher fitter.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--parameters-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args.device)
        return
    run_job(args)


def worker_main(device: str) -> None:
    emit({"event": "gpu-teacher-worker-ready", "device": device})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            job = json.loads(line)
            run_job(argparse.Namespace(
                input=Path(job["input"]),
                count=int(job["count"]),
                config=Path(job["config"]),
                parameters_output=Path(job["parametersOutput"]),
                metrics_output=Path(job["metricsOutput"]),
                device=device,
                diagnostic_only=bool(job.get("diagnosticOnly", False)),
            ))
        except Exception as error:
            emit({
                "event": "gpu-teacher-error",
                "message": str(error),
                "traceback": traceback.format_exc(),
            })


def run_job(args: argparse.Namespace) -> None:
    if any(value is None for value in (
        args.input,
        args.count,
        args.config,
        args.parameters_output,
        args.metrics_output,
    )):
        raise ValueError("one-shot CUDA fitting requires all input and output arguments")
    config = FitConfig(**json.loads(args.config.read_text()))
    validate(args, config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA teacher fitting requested but PyTorch cannot access the GPU")
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    action_count = len(config.action_grid)
    input_row_stride = config.input_row_stride or action_count
    inputs = open_memmap_with_retry(
        args.input,
        mode="r",
        dtype="<f4",
        shape=(args.count, input_row_stride),
    )
    parameters = np.memmap(
        temporary(args.parameters_output), mode="w+", dtype="<f4", shape=(args.count, PARAMETER_COUNT)
    )
    metrics = np.memmap(
        temporary(args.metrics_output), mode="w+", dtype="<f4", shape=(args.count, METRIC_COUNT)
    )
    actions = torch.tensor(config.action_grid, dtype=torch.float32, device=device)
    currents = torch.tensor(config.current_grid, dtype=torch.float32, device=device)
    support = PolicySupport(
        torch.tensor(config.latent_lower, dtype=torch.float32, device=device),
        torch.tensor(config.latent_upper, dtype=torch.float32, device=device),
        torch.tensor(config.visible_lower, dtype=torch.float32, device=device),
        torch.tensor(config.visible_upper, dtype=torch.float32, device=device),
        torch.tensor(config.friction, dtype=torch.float32, device=device),
        torch.tensor(
            1.0 / float(config.transition_log_scale),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            config.score_hinge_span or (config.latent_upper - config.latent_lower),
            dtype=torch.float32,
            device=device,
        ),
    )
    state_indexes = domain_sampled_indices(
        currents,
        config.sample_states,
        config.metric_visible_lower,
        config.metric_visible_upper,
        config.visible_sample_fraction,
    )
    action_indexes = domain_sampled_indices(
        actions,
        config.sample_actions,
        config.metric_visible_lower,
        config.metric_visible_upper,
        config.visible_sample_fraction,
    )
    sampled_actions = actions[action_indexes]
    sampled_currents = currents[state_indexes]
    metric_action_cells = int((
        (actions >= config.metric_visible_lower)
        & (actions <= config.metric_visible_upper)
    ).sum())
    metric_current_cells = int((
        (currents >= config.metric_visible_lower)
        & (currents <= config.metric_visible_upper)
    ).sum())
    if bool(getattr(args, "diagnostic_only", False)):
        run_direct_diagnostics(
            args,
            config,
            inputs,
            parameters,
            metrics,
            actions,
            currents,
            metric_action_cells,
            metric_current_cells,
            device,
        )
        return
    started = time.monotonic()
    totals = torch.zeros(3, dtype=torch.float64)
    rejected_total = 0
    temporal_selected_total = 0
    temporal_candidate_total = 0
    temporal_equivalent_total = 0
    temporal_quality_accepted_total = 0
    temporal_quality_better_total = 0
    temporal_examples_total = 0
    temporal_step_before = 0.0
    temporal_step_after = 0.0
    ranges = [
        (start, min(args.count, start + config.batch_size))
        for start in range(0, args.count, config.batch_size)
    ]

    def stage_host(batch_range: tuple[int, int]) -> tuple[int, int, Tensor]:
        start, end = batch_range
        dense = np.array(inputs[start:end, :action_count + 2], copy=True)
        host = torch.from_numpy(dense)
        if device.type == "cuda":
            host = host.pin_memory()
        return start, end, host

    pipeline_enabled = (
        device.type == "cuda"
        and config.pipelined_refinement
        and len(ranges) > 1
    )
    foreground_stream = torch.cuda.Stream(device=device, priority=-1) \
        if pipeline_enabled else None
    refinement_stream = torch.cuda.Stream(device=device, priority=0) \
        if pipeline_enabled else None
    if foreground_stream is not None:
        foreground_stream.wait_stream(torch.cuda.current_stream(device))
    pipeline_wait_seconds = 0.0

    def persist_batch(
        start: int,
        end: int,
        result: tuple[Tensor, dict[str, Tensor], int, Tensor, dict[str, float]],
    ) -> None:
        nonlocal rejected_total
        nonlocal temporal_selected_total, temporal_candidate_total
        nonlocal temporal_equivalent_total, temporal_quality_accepted_total
        nonlocal temporal_quality_better_total
        nonlocal temporal_examples_total, temporal_step_before, temporal_step_after
        raw, diagnostic, iterations, converged, temporal = result
        parameters[start:end] = raw.cpu().numpy().astype("<f4", copy=False)
        batch_metrics = torch.stack((
            diagnostic["crossEntropy"],
            diagnostic["klDivergence"],
            diagnostic["meanSquaredError"],
            torch.full_like(diagnostic["crossEntropy"], float(iterations)),
            torch.full_like(diagnostic["crossEntropy"], float(config.restarts)),
            converged.float(),
            diagnostic["distanceImbalance"],
        ), dim=-1)
        metrics[start:end] = batch_metrics.cpu().numpy().astype("<f4", copy=False)
        totals.add_(torch.tensor([
            float(diagnostic["klDivergence"].sum()),
            float(diagnostic["meanSquaredError"].sum()),
            end - start,
        ], dtype=torch.float64))
        rejected_total += int((
            (diagnostic["klDivergence"] > config.max_mean_kl)
            | (diagnostic["meanSquaredError"] > config.max_mean_mse)
        ).sum())
        temporal_selected_total += int(temporal["selectedCount"])
        temporal_candidate_total += int(temporal["candidateCount"])
        temporal_equivalent_total += int(temporal["equivalentCount"])
        temporal_quality_accepted_total += int(temporal["qualityAcceptedCount"])
        temporal_quality_better_total += int(temporal["qualityBetterCount"])
        temporal_examples_total += end - start
        temporal_step_before += float(temporal["meanNormalizedStepBefore"]) * (end - start)
        temporal_step_after += float(temporal["meanNormalizedStepAfter"]) * (end - start)
        elapsed = max(time.monotonic() - started, 1e-6)
        emit({
            "event": "gpu-teacher-progress",
            "examplesCompleted": end,
            "examplesTotal": args.count,
            "examplesPerSecond": round(end / elapsed, 2),
            "gpuMemoryMiB": round(torch.cuda.max_memory_allocated() / 1_048_576, 1)
            if device.type == "cuda" else 0,
            "meanKlDivergence": float(totals[0] / totals[2]),
            "meanSquaredError": float(totals[1] / totals[2]),
            "metricVisibleLower": config.metric_visible_lower,
            "metricVisibleUpper": config.metric_visible_upper,
            "metricActionCells": metric_action_cells,
            "metricCurrentCells": metric_current_cells,
            "temporalWarmSelectedFraction": temporal_selected_total
            / max(1, temporal_candidate_total),
            "temporalWarmEquivalentFraction": temporal_equivalent_total
            / max(1, temporal_candidate_total),
            "temporalWarmQualityAcceptedFraction": temporal_quality_accepted_total
            / max(1, temporal_candidate_total),
            "temporalWarmQualityBetterFraction": temporal_quality_better_total
            / max(1, temporal_candidate_total),
            "temporalMeanNormalizedStepBefore": temporal_step_before
            / max(1, temporal_examples_total),
            "temporalMeanNormalizedStepAfter": temporal_step_after
            / max(1, temporal_examples_total),
            "inputQueueBatches": config.input_queue_batches,
            "inputRowStrideFloats": input_row_stride,
            "pipelinedRefinement": pipeline_enabled,
            "pipelineWaitFraction": pipeline_wait_seconds / elapsed,
        })

    def refine_on_stream(
        state: FitBatchState,
        previous_raw: Tensor | None,
        initial_ready: torch.cuda.Event,
    ) -> tuple[
        tuple[Tensor, dict[str, Tensor], int, Tensor, dict[str, float]],
        torch.cuda.Event,
    ]:
        assert refinement_stream is not None
        with torch.cuda.device(device), torch.cuda.stream(refinement_stream):
            refinement_stream.wait_event(initial_ready)
            result = refine_fit_batch(state, previous_raw)
            finished = torch.cuda.Event()
            finished.record(refinement_stream)
        # Keep every foreground-owned tensor in ``state`` alive until its
        # cross-stream consumer has actually finished. Returning before this
        # event lets the caching allocator recycle foreground allocations while
        # the refinement stream still reads them, corrupting whichever lane
        # reuses the block next.
        finished.synchronize()
        return result, finished

    queued: deque[Future[tuple[int, int, Tensor]]] = deque()
    previous_raw: Tensor | None = None
    pending: tuple[
        int,
        int,
        Future[tuple[
            tuple[Tensor, dict[str, Tensor], int, Tensor, dict[str, float]],
            torch.cuda.Event,
        ]],
    ] | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher-input") as loader, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher-refine") as refiner:
        next_range = 0
        while next_range < min(config.input_queue_batches, len(ranges)):
            queued.append(loader.submit(stage_host, ranges[next_range]))
            next_range += 1
        while queued:
            start, end, host = queued.popleft().result()
            if next_range < len(ranges):
                queued.append(loader.submit(stage_host, ranges[next_range]))
                next_range += 1
            if not pipeline_enabled:
                base = host.to(device, non_blocking=device.type == "cuda")
                result = fit_batch(
                    base,
                    actions,
                    currents,
                    sampled_actions,
                    sampled_currents,
                    action_indexes,
                    support,
                    config,
                    previous_raw,
                )
                previous_raw = result[0][-1:].detach().clone()
                persist_batch(start, end, result)
                continue

            assert foreground_stream is not None
            with torch.cuda.stream(foreground_stream):
                base = host.to(device, non_blocking=True)
                state = fit_batch_initial(
                    base,
                    actions,
                    currents,
                    sampled_actions,
                    sampled_currents,
                    action_indexes,
                    support,
                    config,
                )
                initial_ready = torch.cuda.Event()
                initial_ready.record(foreground_stream)

            completed_batch: tuple[
                int,
                int,
                tuple[Tensor, dict[str, Tensor], int, Tensor, dict[str, float]],
            ] | None = None
            if pending is not None:
                wait_started = time.monotonic()
                pending_result, pending_done = pending[2].result()
                pending_done.synchronize()
                pipeline_wait_seconds += time.monotonic() - wait_started
                previous_raw = pending_result[0][-1:].detach()
                completed_batch = (pending[0], pending[1], pending_result)

            pending = (
                start,
                end,
                refiner.submit(
                    refine_on_stream, state, previous_raw, initial_ready
                ),
            )
            if completed_batch is not None:
                persist_batch(*completed_batch)

        if pending is not None:
            wait_started = time.monotonic()
            pending_result, pending_done = pending[2].result()
            pending_done.synchronize()
            pipeline_wait_seconds += time.monotonic() - wait_started
            persist_batch(pending[0], pending[1], pending_result)
    close_memmap(parameters)
    close_memmap(metrics)
    close_memmap(inputs)
    del parameters, metrics
    mean_kl = float(totals[0] / totals[2])
    mean_mse = float(totals[1] / totals[2])
    replace_temporary(args.parameters_output)
    replace_temporary(args.metrics_output)
    emit({
        "event": "gpu-teacher-complete",
        "examples": args.count,
        "seconds": round(time.monotonic() - started, 2),
        "meanKlDivergence": mean_kl,
        "meanSquaredError": mean_mse,
        "metricVisibleLower": config.metric_visible_lower,
        "metricVisibleUpper": config.metric_visible_upper,
        "metricActionCells": metric_action_cells,
        "metricCurrentCells": metric_current_cells,
        "rejectedExamples": rejected_total,
        "qualityAccepted": rejected_total == 0,
        "temporalWarmSelectedFraction": temporal_selected_total
        / max(1, temporal_candidate_total),
        "temporalWarmEquivalentFraction": temporal_equivalent_total
        / max(1, temporal_candidate_total),
        "temporalWarmQualityAcceptedFraction": temporal_quality_accepted_total
        / max(1, temporal_candidate_total),
        "temporalWarmQualityBetterFraction": temporal_quality_better_total
        / max(1, temporal_candidate_total),
        "temporalMeanNormalizedStepBefore": temporal_step_before
        / max(1, temporal_examples_total),
        "temporalMeanNormalizedStepAfter": temporal_step_after
        / max(1, temporal_examples_total),
        "pipelinedRefinement": pipeline_enabled,
        "pipelineWaitFraction": pipeline_wait_seconds
        / max(time.monotonic() - started, 1e-6),
    })


@torch.inference_mode()
def run_direct_diagnostics(
    args: argparse.Namespace,
    config: FitConfig,
    inputs: np.memmap,
    parameters: np.memmap,
    metrics: np.memmap,
    actions: Tensor,
    currents: Tensor,
    metric_action_cells: int,
    metric_current_cells: int,
    device: torch.device,
) -> None:
    """Persist direct-oracle weights without running the obsolete parameter fit."""
    started = time.monotonic()
    action_count = len(config.action_grid)
    input_row_stride = config.input_row_stride or action_count
    visible_actions = (
        (actions >= config.metric_visible_lower)
        & (actions <= config.metric_visible_upper)
    )
    visible_currents = currents[
        (currents >= config.metric_visible_lower)
        & (currents <= config.metric_visible_upper)
    ]
    total_entropy = 0.0
    for start in range(0, args.count, config.batch_size):
        end = min(args.count, start + config.batch_size)
        dense = torch.from_numpy(np.array(
            inputs[start:end, :action_count + 2],
            dtype=np.float32,
            copy=True,
        )).to(device, non_blocking=device.type == "cuda")
        base = dense[:, :action_count][:, visible_actions]
        visible_action_values = actions[visible_actions]
        target, entropy = fit_target(
            base,
            visible_action_values,
            visible_currents,
            config,
        )
        advice = distance_imbalance_advice(
            target,
            visible_action_values,
            visible_currents.view(1, -1).expand(end - start, -1),
            config.distance_epsilon,
        )
        cutoff_raw = dense[:, action_count:action_count + 2].cpu().numpy()
        entropy_values = entropy.cpu().numpy()
        advice_values = advice.cpu().numpy()
        parameters[start:end] = 0
        parameters[start:end, SCORE_PARAMETER_COUNT:PARAMETER_COUNT] = cutoff_raw
        metrics[start:end] = 0
        # These rows intentionally contain only direct-target diagnostics. The
        # fitted-parameter KL/MSE fields are neutral because no fitted
        # distribution was produced; columns 6:8 still preserve hard cutoffs.
        metrics[start:end, 0] = entropy_values
        metrics[start:end, 5] = 1
        metrics[start:end, 6] = advice_values
        total_entropy += float(entropy_values.sum(dtype=np.float64))
        elapsed = max(time.monotonic() - started, 1e-6)
        emit({
            "event": "gpu-teacher-progress",
            "examplesCompleted": end,
            "examplesTotal": args.count,
            "examplesPerSecond": round(end / elapsed, 2),
            "gpuMemoryMiB": round(torch.cuda.max_memory_allocated() / 1_048_576, 1)
            if device.type == "cuda" else 0,
            "meanKlDivergence": 0.0,
            "meanSquaredError": 0.0,
            "metricVisibleLower": config.metric_visible_lower,
            "metricVisibleUpper": config.metric_visible_upper,
            "metricActionCells": metric_action_cells,
            "metricCurrentCells": metric_current_cells,
            "temporalWarmSelectedFraction": 0.0,
            "temporalWarmEquivalentFraction": 0.0,
            "temporalWarmQualityAcceptedFraction": 0.0,
            "temporalWarmQualityBetterFraction": 0.0,
            "temporalMeanNormalizedStepBefore": 0.0,
            "temporalMeanNormalizedStepAfter": 0.0,
            "inputQueueBatches": 1,
            "inputRowStrideFloats": input_row_stride,
            "pipelinedRefinement": False,
            "pipelineWaitFraction": 0.0,
            "diagnosticOnly": True,
        })
    close_memmap(parameters)
    close_memmap(metrics)
    close_memmap(inputs)
    del parameters, metrics
    replace_temporary(args.parameters_output)
    replace_temporary(args.metrics_output)
    emit({
        "event": "gpu-teacher-complete",
        "examples": args.count,
        "seconds": round(time.monotonic() - started, 2),
        "meanKlDivergence": 0.0,
        "meanSquaredError": 0.0,
        "meanTargetEntropy": total_entropy / max(1, args.count),
        "metricVisibleLower": config.metric_visible_lower,
        "metricVisibleUpper": config.metric_visible_upper,
        "metricActionCells": metric_action_cells,
        "metricCurrentCells": metric_current_cells,
        "rejectedExamples": 0,
        "qualityAccepted": True,
        "temporalWarmSelectedFraction": 0.0,
        "temporalWarmEquivalentFraction": 0.0,
        "temporalWarmQualityAcceptedFraction": 0.0,
        "temporalWarmQualityBetterFraction": 0.0,
        "temporalMeanNormalizedStepBefore": 0.0,
        "temporalMeanNormalizedStepAfter": 0.0,
        "pipelinedRefinement": False,
        "pipelineWaitFraction": 0.0,
        "diagnosticOnly": True,
    })


def fit_batch(
    base: Tensor,
    actions: Tensor,
    currents: Tensor,
    sampled_actions: Tensor,
    sampled_currents: Tensor,
    action_indexes: Tensor,
    support: PolicySupport,
    config: FitConfig,
    previous_raw: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor], int, Tensor, dict[str, float]]:
    state = fit_batch_initial(
        base,
        actions,
        currents,
        sampled_actions,
        sampled_currents,
        action_indexes,
        support,
        config,
    )
    return refine_fit_batch(state, previous_raw)


def fit_batch_initial(
    packed: Tensor,
    actions: Tensor,
    currents: Tensor,
    sampled_actions: Tensor,
    sampled_currents: Tensor,
    action_indexes: Tensor,
    support: PolicySupport,
    config: FitConfig,
) -> FitBatchState:
    action_count = actions.numel()
    base = packed[:, :action_count]
    cutoff_raw = packed[:, action_count:action_count + 2]
    batch = base.shape[0]
    with nvtx_range("teacher.initial.target", base):
        target_logits = transition_logits(
            base[:, action_indexes], sampled_actions, sampled_currents,
            config.friction, config.transition_log_scale
        )
        target = torch.softmax(target_logits, dim=-1)
        if config.compact_visible_initialization:
            initial_config = compact_visible_fit_config(config)
            initial_state_indexes = domain_sampled_indices(
                currents,
                config.sample_states,
                config.metric_visible_lower,
                config.metric_visible_upper,
                1.0,
            )
            initial_action_indexes = domain_sampled_indices(
                actions,
                config.sample_actions,
                config.metric_visible_lower,
                config.metric_visible_upper,
                1.0,
            )
            initial_actions = actions[initial_action_indexes]
            initial_currents = currents[initial_state_indexes]
            initial_target_logits = transition_logits(
                base[:, initial_action_indexes],
                initial_actions,
                initial_currents,
                config.friction,
                config.transition_log_scale,
            )
            initial_target = torch.softmax(initial_target_logits, dim=-1)
            initial_support = PolicySupport(
                base.new_tensor(initial_config.latent_lower),
                base.new_tensor(initial_config.latent_upper),
                base.new_tensor(initial_config.visible_lower),
                base.new_tensor(initial_config.visible_upper),
                support.friction,
                support.temperature,
                # Use the final model's transition widths in both stages. This
                # is what makes the score-coordinate remap exact.
                support.hinge_span,
            )
            initial_cutoff_raw = remap_cutoff_support(
                cutoff_raw, config, initial_config
            )
        else:
            initial_config = config
            initial_actions = sampled_actions
            initial_currents = sampled_currents
            initial_target_logits = target_logits
            initial_target = target
            initial_support = support
            initial_cutoff_raw = cutoff_raw
    with nvtx_range("teacher.initial.variable_projection", base):
        raw = variable_projection_initial_parameters(
            initial_target_logits,
            initial_actions,
            initial_currents,
            initial_config,
            initial_cutoff_raw,
        )
    # Variable projection is a much cheaper way to explore structural basins
    # than BFGS. Score every projected start once, then carry only the best
    # candidate for each example into the expensive second-order solve.
    with nvtx_range("teacher.initial.select_projection", base):
        if raw.shape[1] > 1:
            projected_cross_entropy = cross_entropy_objective(
                raw, initial_target, initial_actions, initial_currents, initial_support
            )
            projected_winner = projected_cross_entropy.argmin(dim=1)
            raw = raw[torch.arange(batch, device=base.device), projected_winner, None, :]
    warmup_iterations = min(
        30,
        config.iterations - 1,
        max(1, config.iterations // 4),
    )
    linear_mask = torch.zeros(PARAMETER_COUNT, device=base.device)
    linear_mask[2:6] = 1
    with nvtx_range("teacher.initial.bfgs_linear", base):
        raw, _, _, warmup_used = batched_bfgs(
            raw,
            initial_target,
            initial_actions,
            initial_currents,
            initial_support,
            warmup_iterations,
            config.tolerance,
            linear_mask,
            config.line_search_candidates,
            config.optimizer_backend,
            config.optimizer_host_check_interval,
        )
    with nvtx_range("teacher.initial.bfgs_full", base):
        raw, cross_entropy, converged_restarts, refine_used = batched_bfgs(
            raw,
            initial_target,
            initial_actions,
            initial_currents,
            initial_support,
            config.iterations - warmup_iterations,
            config.tolerance,
            score_parameter_mask(base.device),
            config.line_search_candidates,
            config.optimizer_backend,
            config.optimizer_host_check_interval,
        )
    if config.compact_visible_initialization:
        raw = remap_score_support(raw, initial_config, config)
        raw[..., 6:8] = cutoff_raw[:, None, :]
        cross_entropy = cross_entropy_objective(
            raw, target, sampled_actions, sampled_currents, support
        )
        converged_restarts = torch.zeros_like(cross_entropy, dtype=torch.bool)
    selected = cross_entropy.argmin(dim=1)
    final_raw = raw[torch.arange(batch, device=base.device), selected]
    with nvtx_range("teacher.initial.diagnostics", base):
        diagnostic_base, diagnostic_actions, diagnostic_currents = metric_surface(
            base, actions, currents, config
        )
        diagnostic_target, diagnostic_entropy = fit_target(
            diagnostic_base, diagnostic_actions, diagnostic_currents, config
        )
        diagnostic = fit_diagnostics(
            diagnostic_target,
            diagnostic_entropy,
            final_raw,
            diagnostic_actions,
            diagnostic_currents,
            support,
        )
    converged = converged_restarts[
        torch.arange(batch, device=base.device), selected
    ]
    total_iterations = warmup_used + refine_used
    return FitBatchState(
        final_raw=final_raw,
        diagnostic=diagnostic,
        converged=converged,
        total_iterations=total_iterations,
        target=target,
        diagnostic_target=diagnostic_target,
        diagnostic_entropy=diagnostic_entropy,
        diagnostic_actions=diagnostic_actions,
        diagnostic_currents=diagnostic_currents,
        sampled_actions=sampled_actions,
        sampled_currents=sampled_currents,
        support=support,
        config=config,
    )


def refine_fit_batch(
    state: FitBatchState,
    previous_raw: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor], int, Tensor, dict[str, float]]:
    final_raw = state.final_raw
    diagnostic = state.diagnostic
    converged = state.converged
    total_iterations = state.total_iterations
    target = state.target
    diagnostic_target = state.diagnostic_target
    diagnostic_entropy = state.diagnostic_entropy
    diagnostic_actions = state.diagnostic_actions
    diagnostic_currents = state.diagnostic_currents
    sampled_actions = state.sampled_actions
    sampled_currents = state.sampled_currents
    support = state.support
    config = state.config
    device = final_raw.device
    for round_index in range(config.adaptive_rounds):
        with nvtx_range(f"teacher.refine.adaptive_{round_index + 1}", final_raw):
            hard = (
                (diagnostic["klDivergence"] > config.max_mean_kl)
                | (diagnostic["meanSquaredError"] > config.max_mean_mse)
            )
            hard_indexes = hard.nonzero(as_tuple=False).squeeze(-1)
            if hard_indexes.numel() == 0:
                break
            refined, _, refined_converged, iterations_used = batched_bfgs(
                final_raw[hard_indexes, None, :],
                target[hard_indexes],
                sampled_actions,
                sampled_currents,
                support,
                config.adaptive_iterations,
                config.tolerance,
                score_parameter_mask(device),
                config.line_search_candidates,
                config.optimizer_backend,
                config.optimizer_host_check_interval,
            )
            refined_raw = refined[:, 0, :]
            refined_diagnostic = fit_diagnostics(
                diagnostic_target[hard_indexes],
                diagnostic_entropy[hard_indexes],
                refined_raw,
                diagnostic_actions,
                diagnostic_currents,
                support,
            )
            improved = fit_quality_score(refined_diagnostic, config) \
                < fit_quality_score(
                    {name: value[hard_indexes] for name, value in diagnostic.items()},
                    config,
                )
            improved_indexes = hard_indexes[improved]
            if improved_indexes.numel() == 0:
                break
            final_raw = torch.index_copy(
                final_raw, 0, improved_indexes, refined_raw[improved]
            )
            for name in diagnostic:
                diagnostic[name] = torch.index_copy(
                    diagnostic[name], 0, improved_indexes,
                    refined_diagnostic[name][improved],
                )
            converged = torch.index_copy(
                converged, 0, improved_indexes, refined_converged[:, 0][improved]
            )
            total_iterations += iterations_used
    # Triton deliberately lets each fit stop independently, which removes the
    # synchronized-batch tail. Preserve the established fitter's recovery
    # behavior by handing only still-rejected examples to its autograd BFGS.
    # This is cheap on ordinary data and keeps pathological cases resumable for
    # the deeper refinement pass without lowering the quality gate.
    if config.optimizer_backend == "triton-queued" \
            and config.quality_fallback_iterations > 0:
        hard = (
            (diagnostic["klDivergence"] > config.max_mean_kl)
            | (diagnostic["meanSquaredError"] > config.max_mean_mse)
        )
        hard_indexes = hard.nonzero(as_tuple=False).squeeze(-1)
        if hard_indexes.numel() > 0:
            with nvtx_range("teacher.refine.compatibility_fallback", final_raw):
                refined, _, refined_converged, iterations_used = batched_bfgs(
                    final_raw[hard_indexes, None, :],
                    target[hard_indexes],
                    sampled_actions,
                    sampled_currents,
                    support,
                    config.quality_fallback_iterations,
                    config.tolerance,
                    score_parameter_mask(device),
                    config.line_search_candidates,
                    "pytorch-batched",
                    config.optimizer_host_check_interval,
                )
                refined_raw = refined[:, 0, :]
                refined_diagnostic = fit_diagnostics(
                    diagnostic_target[hard_indexes],
                    diagnostic_entropy[hard_indexes],
                    refined_raw,
                    diagnostic_actions,
                    diagnostic_currents,
                    support,
                )
                improved = fit_quality_score(refined_diagnostic, config) \
                    < fit_quality_score(
                        {name: value[hard_indexes] for name, value in diagnostic.items()},
                        config,
                    )
                improved_indexes = hard_indexes[improved]
                if improved_indexes.numel() > 0:
                    final_raw = torch.index_copy(
                        final_raw, 0, improved_indexes, refined_raw[improved]
                    )
                    for name in diagnostic:
                        diagnostic[name] = torch.index_copy(
                            diagnostic[name], 0, improved_indexes,
                            refined_diagnostic[name][improved],
                        )
                    converged = torch.index_copy(
                        converged, 0, improved_indexes,
                        refined_converged[:, 0][improved],
                    )
                total_iterations += iterations_used
    temporal = {
        "selectedCount": 0.0,
        "candidateCount": 0.0,
        "selectedFraction": 0.0,
        "equivalentCount": 0.0,
        "equivalentFraction": 0.0,
        "qualityAcceptedCount": 0.0,
        "qualityAcceptedFraction": 0.0,
        "qualityBetterCount": 0.0,
        "qualityBetterFraction": 0.0,
        "meanNormalizedStepBefore": mean_normalized_parameter_step(final_raw),
        "meanNormalizedStepAfter": mean_normalized_parameter_step(final_raw),
        "medianNormalizedStepBefore": median_normalized_parameter_step(final_raw),
        "medianNormalizedStepAfter": median_normalized_parameter_step(final_raw),
    }
    if config.temporal_refinement_rounds > 0 and final_raw.shape[0] > 1:
        with nvtx_range("teacher.refine.temporal", final_raw):
            final_raw, converged, temporal_iterations, temporal = temporal_warm_refinement(
                final_raw,
                converged,
                target,
                sampled_actions,
                sampled_currents,
                diagnostic_target,
                diagnostic_actions,
                diagnostic_currents,
                diagnostic["crossEntropy"],
                support,
                config,
                previous_raw,
            )
            total_iterations += temporal_iterations
            diagnostic = fit_diagnostics(
                diagnostic_target,
                diagnostic_entropy,
                final_raw,
                diagnostic_actions,
                diagnostic_currents,
                support,
            )
    # This is one scalar for the complete timestamp example, not a separate
    # weight for each current-exposure row. Both sums cover the exact
    # cutoff-applied raw oracle surface on the visible metric support.
    diagnostic["distanceImbalance"] = distance_imbalance_advice(
        diagnostic_target,
        diagnostic_actions,
        diagnostic_currents.view(1, -1).expand(diagnostic_target.shape[0], -1),
        config.distance_epsilon,
    )
    return final_raw, diagnostic, total_iterations, converged, temporal


def temporal_warm_refinement(
    independent_raw: Tensor,
    independent_converged: Tensor,
    target: Tensor,
    sampled_actions: Tensor,
    sampled_currents: Tensor,
    diagnostic_target: Tensor,
    diagnostic_actions: Tensor,
    diagnostic_currents: Tensor,
    independent_cross_entropy: Tensor,
    support: PolicySupport,
    config: FitConfig,
    initial_previous_raw: Tensor | None = None,
) -> tuple[Tensor, Tensor, int, dict[str, float]]:
    """Quality-preserving temporal warm starts in wide synchronous CUDA passes."""
    selected = independent_raw.clone()
    selected_cross_entropy = independent_cross_entropy.clone()
    converged = independent_converged.clone()
    before_mean = mean_normalized_parameter_step(selected)
    before_median = median_normalized_parameter_step(selected)
    parameter_scale = robust_parameter_scale(selected)
    iterations_used = 0
    full_mask = score_parameter_mask(selected.device)
    last_quality_equivalent = torch.zeros(0, dtype=torch.bool, device=selected.device)
    last_quality_accepted = torch.zeros(0, dtype=torch.bool, device=selected.device)
    last_quality_better = torch.zeros(0, dtype=torch.bool, device=selected.device)
    for round_index in range(config.temporal_refinement_rounds):
        previous_round = selected.clone()
        # A synchronous pass keeps all adjacent warm starts in one wide CUDA
        # batch. Repeating the pass propagates the selected trajectory forward
        # without serializing 86,400 individual optimizations for a full day.
        first_index = 0 if initial_previous_raw is not None else 1
        indexes = torch.arange(first_index, selected.shape[0], device=selected.device)
        if indexes.numel() == 0:
            break
        previous = previous_round[:-1]
        if initial_previous_raw is not None:
            previous = torch.cat((initial_previous_raw, previous), dim=0)
        # Cutoffs are exact teacher labels for the current timestamp. Warm
        # starts may carry only the six smooth score coordinates forward.
        previous = torch.cat((previous[:, :SCORE_PARAMETER_COUNT],
                              independent_raw[indexes, SCORE_PARAMETER_COUNT:]), dim=-1)
        existing = selected[indexes]
        warm, warm_loss, warm_converged, used = batched_bfgs(
            previous[:, None, :],
            target[indexes],
            sampled_actions,
            sampled_currents,
            support,
            config.temporal_iterations
            if round_index == 0 or config.temporal_followup_iterations == 0
            else config.temporal_followup_iterations,
            config.tolerance,
            full_mask,
            config.line_search_candidates,
            config.optimizer_backend,
            config.optimizer_host_check_interval,
        )
        iterations_used += used
        warm = warm[:, 0, :]
        warm_log_probability = fit_policy_log_probability(
            warm, diagnostic_actions, diagnostic_currents, support
        )
        warm_cross_entropy = -(
            diagnostic_target[indexes] * warm_log_probability
        ).sum(dim=-1).mean(dim=-1)
        target_entropy = -(
            diagnostic_target[indexes]
            * diagnostic_target[indexes].clamp_min(
                torch.finfo(diagnostic_target.dtype).tiny
            ).log()
        ).sum(dim=-1).mean(dim=-1)
        warm_kl = (warm_cross_entropy - target_entropy).clamp_min(0)
        warm_mse = (
            diagnostic_target[indexes] - warm_log_probability.exp()
        ).square().mean(dim=(-1, -2))
        existing_loss = selected_cross_entropy[indexes]
        warm_loss = warm_cross_entropy
        quality_reference = torch.minimum(
            independent_cross_entropy[indexes], existing_loss
        )
        tolerance = config.temporal_equivalent_loss_absolute \
            + config.temporal_equivalent_loss_relative * quality_reference.abs()
        quality_better = warm_loss < quality_reference - tolerance
        quality_equivalent = warm_loss <= quality_reference + tolerance
        quality_accepted = (
            (warm_kl <= config.max_mean_kl)
            & (warm_mse <= config.max_mean_mse)
        )
        last_quality_equivalent = quality_equivalent
        last_quality_accepted = quality_accepted
        last_quality_better = quality_better
        existing_distance = normalized_parameter_distance(
            existing, previous, parameter_scale
        )
        warm_distance = normalized_parameter_distance(warm, previous, parameter_scale)
        choose_warm = quality_better | (
            quality_equivalent & (warm_distance < existing_distance)
        )
        selected[indexes] = torch.where(choose_warm[:, None], warm, existing)
        selected_cross_entropy[indexes] = torch.where(
            choose_warm, warm_cross_entropy, existing_loss
        )
        converged[indexes] = torch.where(
            choose_warm, warm_converged[:, 0], converged[indexes]
        )
    selected, converged = globally_smooth_equivalent_path(
        independent_raw,
        selected,
        independent_converged,
        converged,
        parameter_scale,
        initial_previous_raw,
    )
    changed = (selected - independent_raw).abs().amax(dim=-1) > 1e-7
    first_index = 0 if initial_previous_raw is not None else 1
    selected_count = int(changed[first_index:].sum())
    candidate_count = max(0, selected.shape[0] - first_index)
    equivalent_count = int(last_quality_equivalent.sum())
    quality_accepted_count = int(last_quality_accepted.sum())
    quality_better_count = int(last_quality_better.sum())
    return selected, converged, iterations_used, {
        "selectedCount": float(selected_count),
        "candidateCount": float(candidate_count),
        "selectedFraction": selected_count / max(1, candidate_count),
        "equivalentCount": float(equivalent_count),
        "equivalentFraction": equivalent_count / max(1, candidate_count),
        "qualityAcceptedCount": float(quality_accepted_count),
        "qualityAcceptedFraction": quality_accepted_count / max(1, candidate_count),
        "qualityBetterCount": float(quality_better_count),
        "qualityBetterFraction": quality_better_count / max(1, candidate_count),
        "meanNormalizedStepBefore": before_mean,
        "meanNormalizedStepAfter": mean_normalized_parameter_step(selected),
        "medianNormalizedStepBefore": before_median,
        "medianNormalizedStepAfter": median_normalized_parameter_step(selected),
    }


def globally_smooth_equivalent_path(
    independent: Tensor,
    temporal: Tensor,
    independent_converged: Tensor,
    temporal_converged: Tensor,
    scale: Tensor,
    initial_previous_raw: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Choose the minimum-total-jump path through two quality-equivalent candidates."""
    candidates = torch.stack((independent, temporal), dim=1).detach().double().cpu().numpy()
    normalized = candidates / scale.detach().double().cpu().numpy()[None, None, :]
    count = candidates.shape[0]
    cost = np.full((count, 2), np.inf, dtype=np.float64)
    parent = np.zeros((count, 2), dtype=np.int8)
    if initial_previous_raw is None:
        cost[0] = 0
    else:
        previous = (
            initial_previous_raw.detach().double().cpu().numpy()[0]
            / scale.detach().double().cpu().numpy()
        )
        cost[0] = ((normalized[0] - previous[None, :]) ** 2).sum(axis=-1)
    for index in range(1, count):
        for choice in range(2):
            transition = ((
                normalized[index, choice][None, :] - normalized[index - 1]
            ) ** 2).sum(axis=-1)
            possibilities = cost[index - 1] + transition
            previous = int(possibilities.argmin())
            cost[index, choice] = possibilities[previous]
            parent[index, choice] = previous
    choices = np.zeros(count, dtype=np.int64)
    choices[-1] = int(cost[-1].argmin())
    for index in range(count - 1, 0, -1):
        choices[index - 1] = parent[index, choices[index]]
    choice = torch.from_numpy(choices).to(independent.device)
    rows = torch.arange(count, device=independent.device)
    raw_candidates = torch.stack((independent, temporal), dim=1)
    convergence_candidates = torch.stack(
        (independent_converged, temporal_converged), dim=1
    )
    return raw_candidates[rows, choice], convergence_candidates[rows, choice]


def robust_parameter_scale(raw: Tensor) -> Tensor:
    values = raw.detach().double()
    lower = torch.quantile(values, 0.25, dim=0)
    upper = torch.quantile(values, 0.75, dim=0)
    minimum = torch.tensor(
        [0.25, 0.25, 2.0, 0.1, 2.0, 2.0, 0.25, 0.25],
        dtype=values.dtype,
        device=values.device,
    )
    return (upper - lower).clamp_min(minimum).to(raw.dtype)


def normalized_parameter_distance(left: Tensor, right: Tensor, scale: Tensor) -> Tensor:
    return ((left - right) / scale).square().mean(dim=-1)


def mean_normalized_parameter_step(raw: Tensor) -> float:
    if raw.shape[0] < 2:
        return 0.0
    scale = robust_parameter_scale(raw)
    return float(torch.linalg.vector_norm((raw[1:] - raw[:-1]) / scale, dim=-1).mean())


def median_normalized_parameter_step(raw: Tensor) -> float:
    if raw.shape[0] < 2:
        return 0.0
    scale = robust_parameter_scale(raw)
    return float(torch.linalg.vector_norm((raw[1:] - raw[:-1]) / scale, dim=-1).median())


def batched_bfgs(
    initial_raw: Tensor,
    target: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
    maximum_iterations: int,
    tolerance: float,
    parameter_mask: Tensor,
    line_search_candidates: int,
    optimizer_backend: str = "pytorch-batched",
    optimizer_host_check_interval: int = 32,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Independent full-memory BFGS solves, vectorized across batch/restarts."""
    if optimizer_backend == "triton-queued" and initial_raw.shape[1] == 1:
        from teacher_bfgs_triton import triton_bfgs

        raw, loss, converged, diagnostics = triton_bfgs(
            initial_raw[:, 0, :],
            target,
            actions,
            currents,
            support,
            maximum_iterations,
            tolerance,
            parameter_mask,
            line_search_candidates,
            optimizer_host_check_interval,
        )
        return raw[:, None, :], loss[:, None], converged[:, None], diagnostics.iterations
    if optimizer_backend not in ("pytorch-batched", "triton-queued"):
        raise ValueError(f"unknown CUDA teacher optimizer backend: {optimizer_backend}")
    raw = initial_raw.detach().clone()
    batch, restarts, parameter_count = raw.shape
    identity = torch.eye(parameter_count, device=raw.device, dtype=raw.dtype)
    inverse_hessian = identity.expand(batch, restarts, -1, -1).clone()
    loss, gradient = cross_entropy_with_gradient(
        raw, target, actions, currents, support, parameter_mask
    )
    active = torch.isfinite(loss)
    converged = torch.zeros_like(active)
    stable_iterations = torch.zeros_like(loss, dtype=torch.int64)
    iterations_used = 0
    host_check_interval = max(1, optimizer_host_check_interval)
    for iteration in range(maximum_iterations):
        iterations_used = iteration + 1
        gradient_maximum = gradient.abs().amax(dim=-1)
        newly_converged = active & (gradient_maximum <= tolerance)
        converged |= newly_converged
        active &= ~newly_converged
        should_check_host = (
            iteration == 0
            or (iteration + 1) % host_check_interval == 0
            or iteration + 1 == maximum_iterations
        )
        if should_check_host and not bool(active.any()):
            break

        direction = -torch.einsum("brij,brj->bri", inverse_hessian, gradient)
        directional_derivative = (gradient * direction).sum(dim=-1)
        invalid_direction = active & (
            (directional_derivative >= 0) | ~torch.isfinite(directional_derivative)
        )
        direction = torch.where(invalid_direction[..., None], -gradient, direction)
        inverse_hessian = torch.where(
            invalid_direction[..., None, None], identity, inverse_hessian
        )
        directional_derivative = (gradient * direction).sum(dim=-1)
        direction_maximum = direction.abs().amax(dim=-1).clamp_min(1e-30)
        direction_scale = torch.minimum(
            torch.ones_like(direction_maximum), 4 / direction_maximum
        )
        direction *= direction_scale[..., None]
        directional_derivative *= direction_scale

        if line_search_candidates > 0:
            next_raw, next_loss, accepted = branchless_line_search(
                raw,
                loss,
                direction,
                directional_derivative,
                active,
                target,
                actions,
                currents,
                support,
                line_search_candidates,
            )
        else:
            step = torch.ones_like(loss)
            accepted = torch.zeros_like(active)
            next_raw = raw.clone()
            next_loss = loss.clone()
            for _ in range(24):
                searching = active & ~accepted
                if not bool(searching.any()):
                    break
                candidate = bound_raw(raw + step[..., None] * direction)
                candidate_loss = cross_entropy_objective(
                    candidate, target, actions, currents, support
                )
                sufficient_decrease = candidate_loss <= (
                    loss + 1e-4 * step * directional_derivative
                )
                take = searching & torch.isfinite(candidate_loss) & sufficient_decrease
                next_raw = torch.where(take[..., None], candidate, next_raw)
                next_loss = torch.where(take, candidate_loss, next_loss)
                accepted |= take
                step = torch.where(searching & ~take, step * 0.5, step)

        next_loss_with_gradient, next_gradient = cross_entropy_with_gradient(
            next_raw, target, actions, currents, support, parameter_mask
        )
        next_loss = torch.where(accepted, next_loss_with_gradient, loss)
        next_gradient = torch.where(accepted[..., None], next_gradient, gradient)
        parameter_delta = next_raw - raw
        gradient_delta = next_gradient - gradient
        curvature = (parameter_delta * gradient_delta).sum(dim=-1)
        hessian_gradient = torch.einsum(
            "brij,brj->bri", inverse_hessian, gradient_delta
        )
        gradient_hessian_gradient = (gradient_delta * hessian_gradient).sum(dim=-1)
        valid_curvature = accepted & torch.isfinite(curvature) & (curvature > 1e-12)
        safe_curvature = curvature.clamp_min(1e-12)
        outer_delta = parameter_delta[..., :, None] * parameter_delta[..., None, :]
        cross_outer = (
            hessian_gradient[..., :, None] * parameter_delta[..., None, :]
            + parameter_delta[..., :, None] * hessian_gradient[..., None, :]
        )
        updated_hessian = inverse_hessian \
            + ((safe_curvature + gradient_hessian_gradient) / safe_curvature.square())[
                ..., None, None
            ] * outer_delta \
            - cross_outer / safe_curvature[..., None, None]
        inverse_hessian = torch.where(
            valid_curvature[..., None, None], updated_hessian, identity
        )

        relative_improvement = (loss - next_loss) / loss.abs().clamp_min(1)
        stable_iterations = torch.where(
            accepted & (relative_improvement <= tolerance),
            stable_iterations + 1,
            torch.zeros_like(stable_iterations),
        )
        raw, loss, gradient = next_raw, next_loss, next_gradient
        stable_convergence = active & accepted & (stable_iterations >= 4) & (
            gradient.abs().amax(dim=-1) <= max(tolerance * 10, 1e-5)
        )
        converged |= stable_convergence
        active &= accepted & ~stable_convergence
    return raw, loss, converged, iterations_used


def branchless_line_search(
    raw: Tensor,
    loss: Tensor,
    direction: Tensor,
    directional_derivative: Tensor,
    active: Tensor,
    target: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
    candidate_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate the complete Armijo step ladder as one wide tensor operation."""
    batch, restarts, parameter_count = raw.shape
    steps = torch.pow(
        torch.as_tensor(0.5, device=raw.device, dtype=raw.dtype),
        torch.arange(candidate_count, device=raw.device, dtype=raw.dtype),
    )
    candidates = bound_raw(
        raw[..., None, :] + steps[None, None, :, None] * direction[..., None, :]
    )
    candidate_loss = cross_entropy_objective(
        candidates.reshape(batch, restarts * candidate_count, parameter_count),
        target,
        actions,
        currents,
        support,
    ).reshape(batch, restarts, candidate_count)
    sufficient_decrease = candidate_loss <= (
        loss[..., None]
        + 1e-4 * steps[None, None, :] * directional_derivative[..., None]
    )
    valid = active[..., None] & torch.isfinite(candidate_loss) & sufficient_decrease
    accepted = valid.any(dim=-1)
    choice = valid.to(torch.int8).argmax(dim=-1, keepdim=True)
    chosen_raw = torch.gather(
        candidates,
        2,
        choice[..., None].expand(-1, -1, 1, parameter_count),
    ).squeeze(2)
    chosen_loss = torch.gather(candidate_loss, 2, choice).squeeze(2)
    return (
        torch.where(accepted[..., None], chosen_raw, raw),
        torch.where(accepted, chosen_loss, loss),
        accepted,
    )


def cross_entropy_with_gradient(
    raw: Tensor,
    target: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
    parameter_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    candidate = raw.detach().requires_grad_(True)
    loss = cross_entropy_objective(candidate, target, actions, currents, support)
    gradient = torch.autograd.grad(loss.sum(), candidate)[0] * parameter_mask
    return loss.detach(), gradient.detach()


def eager_cross_entropy_objective(
    raw: Tensor,
    target: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> Tensor:
    logits = policy_logits_for_restarts(raw, actions, currents, support)
    cross_entropy = -(
        target[:, None, :, :] * torch.log_softmax(logits, dim=-1)
    ).sum(dim=-1).mean(dim=-1)
    slope_excess = (raw[..., 2:SCORE_PARAMETER_COUNT].abs() - 100).clamp_min(0)
    cross_entropy = cross_entropy + 1e-9 * slope_excess.square().sum(dim=-1)
    return cross_entropy


# BFGS evaluates this same tensor graph hundreds of times. Inductor fuses its
# elementwise policy construction and reductions, avoiding tens of thousands of
# eager CUDA launches per day while retaining dynamic hard-case batch sizes.
cross_entropy_objective = (
    torch.compile(
        eager_cross_entropy_objective,
        dynamic=True,
        fullgraph=True,
    )
    if os.environ.get("TRADING_MLP_TEACHER_COMPILE", "1") != "0"
    else eager_cross_entropy_objective
)


def bound_raw(raw: Tensor) -> Tensor:
    return torch.cat((
        raw[..., :2].clamp(-14, 14),
        raw[..., 2:SCORE_PARAMETER_COUNT].clamp(-1e4, 1e4),
        raw[..., SCORE_PARAMETER_COUNT:].clamp(-14, 14),
    ), dim=-1)


def score_parameter_mask(device: torch.device) -> Tensor:
    return torch.cat((
        torch.ones(SCORE_PARAMETER_COUNT, device=device),
        torch.zeros(PARAMETER_COUNT - SCORE_PARAMETER_COUNT, device=device),
    ))


def variable_projection_initial_parameters(
    target_logits: Tensor,
    actions: Tensor,
    currents: Tensor,
    config: FitConfig,
    cutoff_raw: Tensor,
) -> Tensor:
    """Fit [b, lambda, beta_c1, beta_c2] while optimizing only c1/c2."""
    batch = target_logits.shape[0]
    target_probability = torch.softmax(target_logits, dim=-1)
    probability_floor = target_probability.amax(dim=-1, keepdim=True) * 1e-6
    target_scores = target_probability.clamp_min(probability_floor).log()
    target_statistics = projection_target_statistics(
        target_scores, actions, currents, config
    )
    structural = structural_starts(target_scores, actions, currents, config)
    structural = structural.detach().requires_grad_(True)
    first_moment = torch.zeros_like(structural)
    second_moment = torch.zeros_like(structural)
    learning_rate = 0.08
    beta1 = 0.9
    beta2 = 0.99
    best_loss = torch.full(
        (batch, config.restarts),
        math.inf,
        device=target_logits.device,
        dtype=torch.float64,
    )
    best_structural = structural.detach().clone()
    host_check_interval = max(1, config.optimizer_host_check_interval)
    for iteration in range(config.projection_iterations):
        projected_loss, _ = project_linear_parameters(
            structural, actions, currents, config, target_statistics
        )
        loss = projected_loss.mean()
        should_check_host = (
            iteration == 0
            or (iteration + 1) % host_check_interval == 0
            or iteration + 1 == config.projection_iterations
        )
        if should_check_host and not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite CUDA variable-projection objective at iteration {iteration}"
            )
        gradient = torch.autograd.grad(loss, structural)[0]
        gradient_norm = torch.linalg.vector_norm(gradient)
        gradient.mul_(torch.clamp(5.0 / (gradient_norm + 1e-6), max=1.0))
        with torch.no_grad():
            improved = projected_loss < best_loss
            best_loss = torch.where(improved, projected_loss, best_loss)
            best_structural = torch.where(
                improved[..., None], structural.detach(), best_structural
            )
            first_moment.lerp_(gradient, 1 - beta1)
            second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            step = iteration + 1
            step_size = -learning_rate / (1 - beta1 ** step)
            denominator = second_moment.sqrt().div_(
                math.sqrt(1 - beta2 ** step)
            ).add_(1e-8)
            structural.addcdiv_(
                first_moment,
                denominator,
                value=step_size,
            )
            structural.clamp_(-14, 14)
        progress = iteration / max(1, config.projection_iterations - 1)
        learning_rate = 0.08 * (
            0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
        )

    _, coefficients = project_linear_parameters(
        best_structural, actions, currents, config, target_statistics
    )
    raw = torch.zeros(
        (batch, config.restarts, PARAMETER_COUNT),
        device=target_logits.device,
        dtype=target_logits.dtype,
    )
    raw[..., :2] = best_structural.to(raw.dtype).clamp(-14, 14)
    # Projection uses features divided by visible span, so its coefficients
    # are already in the raw slope scale expected by the shared model.
    raw[..., 2:6] = coefficients.to(raw.dtype)
    raw[..., 6:8] = cutoff_raw[:, None, :]
    return raw


def project_linear_parameters(
    structural: Tensor,
    actions: Tensor,
    currents: Tensor,
    config: FitConfig,
    target_statistics: ProjectionTargetStatistics,
) -> tuple[Tensor, Tensor]:
    visible_span = config.visible_upper - config.visible_lower
    half_visible_span = visible_span / 2
    visible_center = (config.visible_lower + config.visible_upper) / 2
    latent_span = config.latent_upper - config.latent_lower
    c1 = config.latent_lower + latent_span * torch.sigmoid(structural[..., 0])
    c2 = c1 + (config.latent_upper - c1) * torch.sigmoid(structural[..., 1])
    kappa_c = 82.0 / (config.score_hinge_span or visible_span)
    action = actions.view(1, 1, 1, -1)
    c1_feature = scaled_softplus(
        action - c1[..., None, None], torch.as_tensor(kappa_c, device=action.device)
    )
    c2_feature = scaled_softplus(
        action - c2[..., None, None], torch.as_tensor(kappa_c, device=action.device)
    )
    base_feature = (action - config.latent_lower) / visible_span
    precision_feature = -0.5 * ((action - visible_center) / half_visible_span).square()
    feature_shape = (
        structural.shape[0], structural.shape[1], currents.numel(), actions.numel()
    )
    c1_feature = c1_feature.expand(feature_shape)
    c2_feature = c2_feature.expand(feature_shape)
    base_feature = base_feature.expand(feature_shape)
    precision_feature = precision_feature.expand(feature_shape)
    features = torch.stack((
        base_feature,
        precision_feature,
        c1_feature / visible_span,
        c2_feature / visible_span,
    ), dim=-1)
    weights = target_statistics.weights
    feature_mean = (features * weights[..., None]).sum(dim=-2, keepdim=True) \
        / target_statistics.weight_total[..., None]
    centered_features = features - feature_mean
    centered_target = target_statistics.centered_scores

    # Keep the large feature grid and its reductions in CUDA-native float32.
    # Only the resulting 4x4 normal systems need float64 for the correlated
    # hinge solve; this avoids materializing three full-size double tensors.
    x = centered_features
    y = centered_target.expand(x.shape[:-1])
    weight = weights.expand(x.shape[:-1])
    normal = torch.einsum("brsa,brsai,brsaj->brij", weight, x, x).double()
    right = torch.einsum("brsa,brsai,brsa->bri", weight, x, y).double()
    ridge = 1e-7
    identity = torch.eye(4, dtype=normal.dtype, device=normal.device)
    maximum_diagonal = normal.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1)
    jitter = torch.maximum(
        torch.full_like(maximum_diagonal, 1e-14), maximum_diagonal * 1e-12
    )
    regularized = normal + identity * (ridge + jitter)[..., None, None]
    coefficients = torch.linalg.solve(regularized, right[..., None]).squeeze(-1)
    prediction = torch.einsum("brsai,bri->brsa", x, coefficients.float())
    squared_error = (weight * (prediction - y).square()).sum(dim=(-1, -2))
    loss = (
        squared_error + ridge * coefficients.square().sum(dim=-1)
    ) / (actions.numel() * currents.numel())
    return loss, coefficients


def projection_target_statistics(
    target_scores: Tensor,
    actions: Tensor,
    currents: Tensor,
    config: FitConfig,
) -> ProjectionTargetStatistics:
    visible_span = config.visible_upper - config.visible_lower
    kappa_x = 678.0 / (config.score_hinge_span or visible_span)
    beta_x = -(config.friction / (1 - config.friction) + config.friction) \
        * config.transition_log_scale
    action = actions.view(1, 1, 1, -1)
    current = currents.view(1, 1, -1, 1)
    moving_feature = scaled_softplus(
        action - current, torch.as_tensor(kappa_x, device=action.device)
    )
    residual_scores = target_scores[:, None, :, :] - beta_x * moving_feature
    weights = torch.softmax(target_scores, dim=-1)[:, None, :, :]
    weight_total = weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    target_mean = (residual_scores * weights).sum(
        dim=-1, keepdim=True
    ) / weight_total
    return ProjectionTargetStatistics(
        weights=weights,
        weight_total=weight_total,
        centered_scores=residual_scores - target_mean,
    )


def structural_starts(
    target_scores: Tensor,
    actions: Tensor,
    currents: Tensor,
    config: FitConfig,
) -> Tensor:
    batch = target_scores.shape[0]
    action_span = float(actions[-1] - actions[0])
    steps = actions[1:] - actions[:-1]
    typical_step = float(steps.median())
    moving_exclusion = max(typical_step * 2.5, action_span / 14)
    left_slope = (target_scores[..., 1:-1] - target_scores[..., :-2]) / steps[:-1]
    right_slope = (target_scores[..., 2:] - target_scores[..., 1:-1]) / steps[1:]
    center_actions = actions[1:-1]
    mask = (
        (currents[:, None] - center_actions[None, :]).abs() > moving_exclusion
    ).to(target_scores.dtype)
    energy = ((right_slope - left_slope).abs() * mask[None, :, :]).sum(dim=1) \
        / mask.sum(dim=0).clamp_min(1)[None, :]
    padded = torch.nn.functional.pad(energy[:, None, :], (2, 2), mode="replicate")
    kernel = torch.tensor(
        [1, 2, 3, 2, 1], device=target_scores.device, dtype=target_scores.dtype
    ).view(1, 1, 5)
    smoothed = torch.nn.functional.conv1d(padded, kernel).squeeze(1) / kernel.sum()
    minimum_separation = max(typical_step * 4, action_span / 8)
    first_index = smoothed.argmax(dim=-1)
    first_location = center_actions[first_index]
    separated = (
        center_actions[None, :] - first_location[:, None]
    ).abs() >= minimum_separation
    second_index = smoothed.masked_fill(~separated, -torch.inf).argmax(dim=-1)
    second_location = center_actions[second_index]
    detected = torch.sort(
        torch.stack((first_location, second_location), dim=-1), dim=-1
    ).values

    visible_thirds = torch.tensor(
        [
            config.visible_lower + action_span / 3,
            config.visible_upper - action_span / 3,
        ],
        device=target_scores.device,
        dtype=target_scores.dtype,
    ).expand(batch, -1)
    latent_span = config.latent_upper - config.latent_lower
    latent_quarters = torch.tensor(
        [config.latent_lower + latent_span / 4, config.latent_upper - latent_span / 4],
        device=target_scores.device,
        dtype=target_scores.dtype,
    ).expand(batch, -1)
    widened = torch.tensor(
        [config.visible_lower - action_span / 8, config.visible_upper + action_span / 8],
        device=target_scores.device,
        dtype=target_scores.dtype,
    ).expand(batch, -1)
    visible_fifths = torch.tensor(
        [config.visible_lower + action_span / 5, config.visible_upper - action_span / 5],
        device=target_scores.device,
        dtype=target_scores.dtype,
    ).expand(batch, -1)
    pair_choices = [
        latent_quarters, detected, visible_thirds, widened,
        visible_fifths, detected, visible_thirds,
    ]
    structural = torch.empty(
        (batch, config.restarts, 2),
        device=target_scores.device,
        dtype=target_scores.dtype,
    )
    for restart in range(config.restarts):
        pair = pair_choices[restart % len(pair_choices)]
        structural[:, restart, 0], structural[:, restart, 1] = raw_breakpoints(
            pair[:, 0], pair[:, 1], config
        )
    return structural


def raw_breakpoints(left: Tensor, right: Tensor, config: FitConfig) -> tuple[Tensor, Tensor]:
    span = config.latent_upper - config.latent_lower
    first = ((left - config.latent_lower) / span).clamp(1e-6, 1 - 1e-6)
    second = ((right - left) / (config.latent_upper - left)).clamp(1e-6, 1 - 1e-6)
    return torch.logit(first), torch.logit(second)


def compact_visible_fit_config(config: FitConfig) -> FitConfig:
    """The old useful-range fit, retaining the final model's hinge widths."""
    return replace(
        config,
        latent_lower=config.metric_visible_lower,
        latent_upper=config.metric_visible_upper,
        visible_lower=config.metric_visible_lower,
        visible_upper=config.metric_visible_upper,
        score_hinge_span=config.score_hinge_span
        or (config.latent_upper - config.latent_lower),
    )


def remap_score_support(
    raw: Tensor,
    source: FitConfig,
    destination: FitConfig,
) -> Tensor:
    """Preserve a six-parameter score while changing its support coordinates.

    The two scores may differ by an action-independent constant, which cancels
    exactly under softmax. Fixed hinge widths must be identical in both
    configurations; compact_visible_fit_config enforces that invariant.
    """
    source_span = source.latent_upper - source.latent_lower
    destination_span = destination.latent_upper - destination.latent_lower
    source_half = source_span / 2
    destination_half = destination_span / 2
    source_center = (source.latent_lower + source.latent_upper) / 2
    destination_center = (destination.latent_lower + destination.latent_upper) / 2

    mapped = raw.clone()
    first_fraction = torch.sigmoid(raw[..., 0])
    c1 = source.latent_lower + source_span * first_fraction
    second_fraction = torch.sigmoid(raw[..., 1])
    c2 = c1 + (source.latent_upper - c1) * second_fraction
    mapped[..., 0], mapped[..., 1] = raw_breakpoints(c1, c2, destination)

    precision = raw[..., 3] / (source_half * source_half)
    base_slope = raw[..., 2] / source_span \
        + precision * (source_center - destination_center)
    mapped[..., 2] = base_slope * destination_span
    mapped[..., 3] = precision * destination_half * destination_half
    mapped[..., 4:6] = raw[..., 4:6] * (destination_span / source_span)
    return bound_raw(mapped)


def remap_cutoff_support(
    raw: Tensor,
    source: FitConfig,
    destination: FitConfig,
) -> Tensor:
    """Clip physical cutoff locations into another support and re-encode them."""
    lower = source.latent_lower + (-source.latent_lower) * torch.sigmoid(raw[..., 0])
    upper = source.latent_upper * torch.sigmoid(raw[..., 1])
    lower = torch.where(raw[..., 0] <= -13.999999, source.latent_lower, lower)
    lower = torch.where(raw[..., 0] >= 13.999999, 0.0, lower)
    upper = torch.where(raw[..., 1] <= -13.999999, 0.0, upper)
    upper = torch.where(raw[..., 1] >= 13.999999, source.latent_upper, upper)
    lower = lower.clamp(destination.latent_lower, 0.0)
    upper = upper.clamp(0.0, destination.latent_upper)

    lower_fraction = (
        (lower - destination.latent_lower) / -destination.latent_lower
    ).clamp(1e-6, 1 - 1e-6)
    upper_fraction = (upper / destination.latent_upper).clamp(1e-6, 1 - 1e-6)
    mapped = torch.stack((torch.logit(lower_fraction), torch.logit(upper_fraction)), -1)
    mapped[..., 0] = torch.where(
        lower <= destination.latent_lower, -14.0, mapped[..., 0]
    )
    mapped[..., 0] = torch.where(lower >= 0.0, 14.0, mapped[..., 0])
    mapped[..., 1] = torch.where(upper <= 0.0, -14.0, mapped[..., 1])
    mapped[..., 1] = torch.where(
        upper >= destination.latent_upper, 14.0, mapped[..., 1]
    )
    return mapped


def policy_logits_for_restarts(
    raw: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> Tensor:
    batch, restarts, _ = raw.shape
    flat_raw = raw.reshape(batch * restarts, PARAMETER_COUNT)
    raw_rows = flat_raw[:, None, :].expand(-1, currents.numel(), -1)
    current_rows = currents.view(1, -1).expand(batch * restarts, -1)
    logits = conditional_policy_logits(raw_rows, actions, current_rows, support)
    return logits.reshape(batch, restarts, currents.numel(), actions.numel())


@torch.inference_mode()
def metric_surface(
    base: Tensor,
    actions: Tensor,
    currents: Tensor,
    config: FitConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select the executable action/current surface used for fit quality."""
    action_mask = (
        (actions >= config.metric_visible_lower)
        & (actions <= config.metric_visible_upper)
    )
    current_mask = (
        (currents >= config.metric_visible_lower)
        & (currents <= config.metric_visible_upper)
    )
    return base[:, action_mask], actions[action_mask], currents[current_mask]


@torch.inference_mode()
def fit_target(
    base: Tensor,
    actions: Tensor,
    currents: Tensor,
    config: FitConfig,
) -> tuple[Tensor, Tensor]:
    target_logits = transition_logits(
        base, actions, currents, config.friction, config.transition_log_scale
    )
    target_log = torch.log_softmax(target_logits, dim=-1)
    target = target_log.exp()
    entropy = -(target * target_log).sum(dim=-1).mean(dim=-1)
    return target, entropy


@torch.inference_mode()
def fit_policy_log_probability(
    raw: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> Tensor:
    predicted_logits = conditional_policy_logits(
        raw[:, None, :].expand(-1, currents.numel(), -1),
        actions,
        currents.view(1, -1).expand(raw.shape[0], -1),
        support,
    )
    return torch.log_softmax(predicted_logits, dim=-1)


@torch.inference_mode()
def fit_cross_entropy(
    target: Tensor,
    raw: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> Tensor:
    predicted_log = fit_policy_log_probability(
        raw, actions, currents, support
    )
    return -(target * predicted_log).sum(dim=-1).mean(dim=-1)


@torch.inference_mode()
def fit_diagnostics(
    target: Tensor,
    entropy: Tensor,
    raw: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> dict[str, Tensor]:
    predicted_log = fit_policy_log_probability(
        raw, actions, currents, support
    )
    cross_entropy = -(target * predicted_log).sum(dim=-1).mean(dim=-1)
    return {
        "crossEntropy": cross_entropy,
        "klDivergence": (cross_entropy - entropy).clamp_min(0),
        "meanSquaredError": (target - predicted_log.exp()).square().mean(dim=(-1, -2)),
    }


def fit_quality_score(
    diagnostic: dict[str, Tensor],
    config: FitConfig,
) -> Tensor:
    """Rank fits by their worst normalized rejection-gate violation."""
    kl_ratio = diagnostic["klDivergence"] / config.max_mean_kl
    mse_ratio = diagnostic["meanSquaredError"] / config.max_mean_mse
    # The tiny tie-breaker avoids arbitrary choices where the worst gate is
    # numerically equal while still keeping acceptance semantics dominant.
    return torch.maximum(kl_ratio, mse_ratio) + 1e-4 * (kl_ratio + mse_ratio)


def transition_logits(
    base: Tensor,
    actions: Tensor,
    currents: Tensor,
    friction: float,
    transition_log_scale: float,
) -> Tensor:
    action = actions.view(1, 1, -1)
    current = currents.view(1, -1, 1)
    difference = action - current
    buy_denominator = 1 - friction + friction * action
    sell_denominator = 1 - friction * action
    buy_factor = 1 - friction * difference / buy_denominator
    sell_factor = 1 - friction * (-difference) / sell_denominator
    factor = torch.where(
        difference > 0,
        buy_factor,
        torch.where(difference < 0, sell_factor, torch.ones_like(difference)),
    )
    # A zero oracle mass is an exact hard-survival exclusion, not merely a very
    # unlikely action. Preserve it through the current-exposure fee adjustment;
    # reviving it with clamp_min would make KL multiply a tiny target mass by the
    # fitted policy's finite-dtype -infinity and reject otherwise valid fits.
    score = torch.where(
        base > 0,
        base.clamp_min(torch.finfo(base.dtype).tiny).log(),
        torch.full_like(base, torch.finfo(base.dtype).min),
    )
    transition = transition_log_scale * factor.clamp_min(
        torch.finfo(base.dtype).tiny
    ).log()
    return torch.where(
        base[:, None, :] > 0,
        score[:, None, :] + transition,
        torch.full_like(transition, torch.finfo(base.dtype).min),
    )


def sampled_indices(length: int, requested: int, device: torch.device) -> Tensor:
    count = min(length, max(1, requested))
    return torch.linspace(0, length - 1, count, device=device).round().long().unique()


def domain_sampled_indices(
    values: Tensor,
    requested: int,
    visible_lower: float,
    visible_upper: float,
    visible_fraction: float,
) -> Tensor:
    """Sample the useful inner domain densely while retaining outer anchors."""
    if not 0 <= visible_fraction <= 1:
        raise ValueError("visible sample fraction must be between zero and one")
    if visible_fraction == 0:
        return sampled_indices(values.numel(), requested, values.device)
    count = min(values.numel(), max(1, requested))
    indexes = torch.arange(values.numel(), device=values.device)
    visible = indexes[(values >= visible_lower) & (values <= visible_upper)]
    outer = indexes[(values < visible_lower) | (values > visible_upper)]
    visible_count = min(visible.numel(), max(1, round(count * visible_fraction)))
    outer_count = min(outer.numel(), count - visible_count)
    visible_count = min(visible.numel(), count - outer_count)

    def select(source: Tensor, selected_count: int) -> Tensor:
        if selected_count <= 0:
            return source[:0]
        if selected_count >= source.numel():
            return source
        positions = torch.linspace(
            0, source.numel() - 1, selected_count, device=values.device
        ).round().long()
        return source[positions]

    selected = torch.cat((select(visible, visible_count), select(outer, outer_count)))
    return selected.sort().values


def temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def close_memmap(values: np.memmap) -> None:
    values.flush()
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


def open_memmap_with_retry(path: Path, **options) -> np.memmap:
    for attempt in range(20):
        try:
            return np.memmap(path, **options)
        except (FileNotFoundError, PermissionError):
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"unreachable mmap retry state for {path}")


def replace_temporary(path: Path) -> None:
    source = temporary(path)
    for attempt in range(10):
        try:
            source.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            # Windows virus scanners can briefly retain a handle after mmap.close().
            time.sleep(0.05 * (attempt + 1))


def validate(args: argparse.Namespace, config: FitConfig) -> None:
    if args.count < 1 or len(config.action_grid) < 5 or len(config.current_grid) < 3:
        raise ValueError("CUDA teacher fitter requires examples and complete grids")
    if min(
        config.projection_iterations,
        config.iterations,
        config.adaptive_iterations,
        config.restarts,
        config.batch_size,
    ) < 1 or config.adaptive_rounds < 0:
        raise ValueError("CUDA teacher fitting counts must be positive")
    if config.transition_log_scale <= 0 or config.friction < 0 \
            or config.distance_epsilon < 0 \
            or not 0 <= config.visible_sample_fraction <= 1 \
            or config.score_hinge_span < 0:
        raise ValueError("CUDA teacher fitting rates are invalid")
    metric_actions = [
        value for value in config.action_grid
        if config.metric_visible_lower <= value <= config.metric_visible_upper
    ]
    metric_currents = [
        value for value in config.current_grid
        if config.metric_visible_lower <= value <= config.metric_visible_upper
    ]
    if (
        not config.visible_lower <= config.metric_visible_lower
        < config.metric_visible_upper <= config.visible_upper
        or len(metric_actions) < 5
        or len(metric_currents) < 3
    ):
        raise ValueError(
            "CUDA teacher metric range must be a non-empty subset of fit support"
        )
    if config.compact_visible_initialization and not (
        config.metric_visible_lower < 0 < config.metric_visible_upper
    ):
        raise ValueError(
            "compact visible initialization requires usable support around zero"
        )
    if (
        config.line_search_candidates < 0
        or config.temporal_refinement_rounds < 0
        or config.temporal_iterations < 1
        or config.temporal_followup_iterations < 0
        or config.temporal_equivalent_loss_absolute < 0
        or config.temporal_equivalent_loss_relative < 0
        or config.optimizer_backend not in ("pytorch-batched", "triton-queued")
        or config.optimizer_host_check_interval < 1
        or config.quality_fallback_iterations < 0
        or (config.input_row_stride != 0
            and config.input_row_stride < len(config.action_grid) + 2)
        or config.input_queue_batches < 1
        or not isinstance(config.pipelined_refinement, bool)
    ):
        raise ValueError("CUDA teacher temporal-refinement settings are invalid")


def emit(value: dict) -> None:
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
