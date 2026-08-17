from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import torch

from evaluate_autoregressive_density_episodes import (
    evaluate_autoregressive_episodes,
    evaluate_sobol_expected_episodes,
)
from recover_return_density_checkpoint_selections import (
    AUTOREGRESSIVE_EVALUATION_CONTRACT,
    EPISODE_SECONDS,
    MAXIMUM_EPISODES_PER_SPLIT,
    MAXIMUM_SOBOL_EPISODES_PER_SPLIT,
    SOBOL_RANDOMIZED_REPLICATES,
    SOBOL_TRAJECTORIES,
    build_datasets,
    fresh_model,
    resolve,
    write_autoregressive_evaluation,
)
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import training_normalization
from train_normalized_glu_next_return import atomic_json


MATRIX_ID = "next-return-knot-density-scaling-v1"
RUN_IDS = tuple(
    "next-second-4-layer-knot-density-65k-v1"
    if dataset == "65k" and knots == 64
    else f"next-second-4-layer-knot-density-{dataset}-k{knots}-v1"
    for dataset in ("65k", "128k", "256k")
    for knots in (8, 16, 32, 64, 128, 256, 512, 1024)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cleaned active-return 15-minute autoregressive paths "
            "for every saved density checkpoint policy."
        )
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--wait-for-pid", type=int)
    return parser.parse_args()


def process_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong
    )
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)
    )
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    process = kernel32.OpenProcess(0x1000, False, int(pid))
    if not process:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(
            process, ctypes.byref(exit_code)
        ):
            return False
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(process)


def evaluate_run(repo: Path, run_id: str, device: torch.device) -> bool:
    run_root = repo / f"data/training/runs/{run_id}"
    result_file = run_root / "state/result.json"
    comparison_file = run_root / "state/checkpoint-selection-comparison.json"
    completion_file = run_root / "state/autoregressive-episode-evaluation.json"
    status_file = run_root / "state/autoregressive-episode-status.json"
    if completion_file.is_file():
        completion = json.loads(completion_file.read_text(encoding="utf-8"))
        if completion.get("contract") == AUTOREGRESSIVE_EVALUATION_CONTRACT:
            return True
    if not result_file.is_file() or not comparison_file.is_file():
        return False
    plan_file = run_root / "state/plan.json"
    stored = json.loads(plan_file.read_text(encoding="utf-8"))
    plan = stored.get("plan", stored)
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    history_root = resolve(repo, Path(plan["historyDir"]))
    train_dataset, validation_dataset, test_dataset = build_datasets(
        plan, history_root
    )
    normalization = training_normalization(
        train_dataset,
        batch_size=int(plan["training"]["evaluationBatchSize"]),
    )
    density = KnotDensityContract.load(
        resolve(repo, Path(plan["density"]["source"])),
        fit=str(plan["density"]["fit"]),
    )
    model = fresh_model(plan, normalization, density, device)
    policies = comparison["policies"]
    for index, (policy, value) in enumerate(policies.items()):
        existing = value.get("autoregressiveEpisodes")
        sobol_existing = value.get("sobolExpectedEpisodes")
        if existing is not None and sobol_existing is not None:
            continue
        atomic_json({
            "stage": "evaluating-active-return-episodes",
            "planId": plan["id"],
            "policy": policy,
            "completedPolicies": index,
            "totalPolicies": len(policies),
            "pid": os.getpid(),
        }, status_file)
        checkpoint = load_torch_checkpoint(
            run_root / f"checkpoints/selections/{policy}.json",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        if existing is None:
            value["autoregressiveEpisodes"] = {
                "validation": evaluate_autoregressive_episodes(
                    model,
                    validation_dataset,
                    "validation",
                    episode_seconds=EPISODE_SECONDS,
                    maximum_episodes=MAXIMUM_EPISODES_PER_SPLIT,
                    device=device,
                ),
                "test": evaluate_autoregressive_episodes(
                    model,
                    test_dataset,
                    "test",
                    episode_seconds=EPISODE_SECONDS,
                    maximum_episodes=MAXIMUM_EPISODES_PER_SPLIT,
                    device=device,
                ),
            }
        if sobol_existing is None:
            value["sobolExpectedEpisodes"] = {
                "validation": evaluate_sobol_expected_episodes(
                    model,
                    validation_dataset,
                    "validation",
                    episode_seconds=EPISODE_SECONDS,
                    maximum_episodes=MAXIMUM_SOBOL_EPISODES_PER_SPLIT,
                    trajectories=SOBOL_TRAJECTORIES,
                    randomized_replicates=SOBOL_RANDOMIZED_REPLICATES,
                    seed=int(plan["training"]["seed"]) + 100_000,
                    device=device,
                ),
                "test": evaluate_sobol_expected_episodes(
                    model,
                    test_dataset,
                    "test",
                    episode_seconds=EPISODE_SECONDS,
                    maximum_episodes=MAXIMUM_SOBOL_EPISODES_PER_SPLIT,
                    trajectories=SOBOL_TRAJECTORIES,
                    randomized_replicates=SOBOL_RANDOMIZED_REPLICATES,
                    seed=int(plan["training"]["seed"]) + 200_000,
                    device=device,
                ),
            }
        atomic_json(comparison, comparison_file)
    write_autoregressive_evaluation(comparison, completion_file)
    atomic_json({
        "stage": "complete",
        "planId": plan["id"],
        "completedPolicies": len(policies),
        "totalPolicies": len(policies),
        "pid": os.getpid(),
    }, status_file)
    return True


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.wait_for_pid is not None:
        while process_alive(int(args.wait_for_pid)):
            time.sleep(5)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA episode evaluation was requested but unavailable")
    pending = set(RUN_IDS)
    while pending:
        progressed = False
        for run_id in RUN_IDS:
            if run_id not in pending:
                continue
            if evaluate_run(repo, run_id, device):
                pending.remove(run_id)
                progressed = True
                print(json.dumps({
                    "event": "autoregressive-episode-evaluation-complete",
                    "runId": run_id,
                    "remainingRuns": len(pending),
                }, separators=(",", ":")), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if pending and not progressed:
            time.sleep(10)


if __name__ == "__main__":
    main()
