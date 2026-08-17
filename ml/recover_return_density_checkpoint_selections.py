from __future__ import annotations

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from active_return_path_dataset import ActiveReturnPathDataset
from calibrate_next_return_density import calibrate_checkpoint_selections
from evaluate_autoregressive_density_episodes import (
    evaluate_autoregressive_episodes,
    evaluate_autoregressive_leads,
    evaluate_sobol_expected_episodes,
)
from normalized_glu_return_density import NormalizedGluReturnDensity
from return_knot_density import KnotDensityContract
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_autoregressive_minute_return import (
    build_optimizers,
    training_normalization,
)
from train_next_return_knot_density import (
    SINGLE_RETURN_RUNNER_CONTRACT as RUNNER_CONTRACT,
    SELECTION_POLICIES,
    canonical_hash,
    evaluate,
    weighted_nll,
)
from train_next_return_memorization import fixed_nonzero_subset_shards
from train_normalized_glu_next_return import (
    NextReturnDataset,
    atomic_json,
    iter_device_batches,
)


RECOVERY_CONTRACT = "deterministic-density-selection-checkpoint-replay-v1"
COMPARISON_CONTRACT = "six-policy-density-checkpoint-comparison-v1"
AUTOREGRESSIVE_EVALUATION_CONTRACT = (
    "six-policy-cleaned-active-return-episode-evaluation-v3"
)
PAUSE_EXIT_CODE = 75
EPISODE_SECONDS = 15 * 60
MAXIMUM_EPISODES_PER_SPLIT = 128
MAXIMUM_SOBOL_EPISODES_PER_SPLIT = 16
SOBOL_TRAJECTORIES = 4_096
SOBOL_RANDOMIZED_REPLICATES = 16
DEFAULT_SPLIT_PLAN = (
    "ml/training-plans/direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover best train/validation MSE, correlation, and NLL "
            "checkpoint states, then evaluate every policy."
        )
    )
    parser.add_argument("--training-plan", required=True, type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--split-plan", type=Path, default=DEFAULT_SPLIT_PLAN)
    parser.add_argument(
        "--checkpoint-policies",
        nargs="+",
        choices=tuple(SELECTION_POLICIES),
        default=tuple(SELECTION_POLICIES),
    )
    parser.add_argument("--skip-episodes", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def read_epoch_events(run_root: Path) -> list[dict]:
    files = (
        run_root / "logs/training.history.jsonl",
        run_root / "logs/training.jsonl",
    )
    by_epoch: dict[int, dict] = {}
    for file in files:
        if not file.is_file():
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "minute-return-epoch":
                by_epoch[int(event["epoch"])] = event
    if not by_epoch:
        raise ValueError("density checkpoint recovery requires epoch metrics")
    return [by_epoch[index] for index in sorted(by_epoch)]


def event_policy_score(event: dict, policy: str) -> float:
    source, metric, _direction = SELECTION_POLICIES[policy]
    return float(event[source][metric])


def optimal_policy_events(events: list[dict]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for policy, (_source, _metric, direction) in SELECTION_POLICIES.items():
        key = lambda event: event_policy_score(event, policy)
        selected[policy] = min(events, key=key) if direction == "min" \
            else max(events, key=key)
    return selected


def checkpoint_value(
    checkpoint: dict,
    policy: str,
    event: dict,
    plan_hash: str,
) -> dict:
    return {
        "model": checkpoint["model"],
        "epoch": int(event["epoch"]),
        "score": event_policy_score(event, policy),
        "policy": policy,
        "selection": {
            "source": SELECTION_POLICIES[policy][0],
            "metric": SELECTION_POLICIES[policy][1],
            "direction": SELECTION_POLICIES[policy][2],
        },
        "planSha256": plan_hash,
        "runnerContract": RUNNER_CONTRACT,
        "recoveryContract": RECOVERY_CONTRACT,
    }


def build_datasets(
    plan: dict,
    history_root: Path,
) -> tuple[NextReturnDataset, NextReturnDataset, NextReturnDataset]:
    train = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["subset"]["date"])),
        int(plan["subset"]["examples"]),
    )
    heldout = int(plan["evaluation"]["examplesPerSplit"])
    validation = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["evaluation"]["validationStart"])),
        heldout,
    )
    test = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["evaluation"]["testStart"])),
        heldout,
    )
    return (
        NextReturnDataset(train, history_root, exclude_zero_targets=True),
        NextReturnDataset({
            "train": [], "validation": validation["train"], "test": []
        }, history_root, exclude_zero_targets=True),
        NextReturnDataset({
            "train": [], "validation": [], "test": test["train"]
        }, history_root, exclude_zero_targets=True),
    )


def fresh_model(
    plan: dict,
    normalization: dict,
    density: KnotDensityContract,
    device: torch.device,
) -> NormalizedGluReturnDensity:
    architecture = plan["architecture"]
    return NormalizedGluReturnDensity(
        torch.from_numpy(normalization["featureMean"]),
        torch.from_numpy(normalization["featureStd"]),
        density,
        widths=tuple(int(value) for value in architecture["widths"]),
        dropout=float(architecture["dropout"]),
        dropout_rate=float(architecture["dropoutRate"]),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture["learnableCentering"]),
    ).to(device)


def materialize_existing_states(
    run_root: Path,
    targets: dict[str, dict],
    plan_hash: str,
) -> set[str]:
    selection_root = run_root / "checkpoints/selections"
    selection_root.mkdir(parents=True, exist_ok=True)
    recovered: set[str] = set()
    sources: dict[int, dict] = {}
    for name in ("best.json", "last.json"):
        file = run_root / f"checkpoints/{name}"
        if not checkpoint_exists(file):
            continue
        checkpoint = load_torch_checkpoint(
            file, map_location="cpu", weights_only=False
        )
        if checkpoint.get("planSha256") == plan_hash:
            sources[int(checkpoint["epoch"])] = checkpoint
    for policy, event in targets.items():
        target_epoch = int(event["epoch"])
        output = selection_root / f"{policy}.json"
        if checkpoint_exists(output):
            existing = load_torch_checkpoint(
                output, map_location="cpu", weights_only=False
            )
            if existing.get("planSha256") == plan_hash \
                    and int(existing.get("epoch", -1)) == target_epoch \
                    and "prefix_conditioner" in existing.get("model", {}):
                recovered.add(policy)
                continue
        source = sources.get(target_epoch)
        if source is not None:
            save_torch_checkpoint(
                checkpoint_value(source, policy, event, plan_hash), output
            )
            recovered.add(policy)
    return recovered


def replay_missing_states(
    repo: Path,
    plan: dict,
    run_root: Path,
    targets: dict[str, dict],
    recovered: set[str],
    train_dataset: NextReturnDataset,
    normalization: dict,
    density: KnotDensityContract,
    device: torch.device,
    pause_file: Path | None,
) -> None:
    missing = {policy: event for policy, event in targets.items()
               if policy not in recovered}
    if not missing:
        return
    seed = int(plan["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = fresh_model(plan, normalization, density, device)
    optimizers = build_optimizers(model, plan["training"], device)
    plan_hash = canonical_hash(plan)
    selection_root = run_root / "checkpoints/selections"
    targets_by_epoch: dict[int, list[str]] = {}
    for policy, event in missing.items():
        targets_by_epoch.setdefault(int(event["epoch"]), []).append(policy)
    maximum_epoch = max(targets_by_epoch)
    batch_size = int(plan["training"]["batchSize"])
    for epoch in range(maximum_epoch + 1):
        if pause_file is not None and pause_file.is_file():
            atomic_json({
                "stage": "paused",
                "planId": plan["id"],
                "epoch": epoch,
                "maximumEpoch": maximum_epoch,
                "pid": os.getpid(),
            }, run_root / "state/checkpoint-selection-recovery-status.json")
            raise SystemExit(PAUSE_EXIT_CODE)
        model.train()
        for features, batch_targets, weights in iter_device_batches(
            train_dataset.iter_batches(
                "train", batch_size, shuffle=True, shuffle_rows=True,
                seed=seed + epoch, reuse_buffers=True,
            ),
            device,
        ):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss = weighted_nll(model, features, batch_targets, weights)
            loss.backward()
            clip_grad_norm_(
                model.parameters(),
                float(plan["training"]["gradientClip"]),
                foreach=device.type == "cuda",
            )
            for optimizer in optimizers:
                optimizer.step()
        for policy in targets_by_epoch.get(epoch, ()):
            save_torch_checkpoint(
                checkpoint_value(
                    {"model": model.state_dict()},
                    policy,
                    targets[policy],
                    plan_hash,
                ),
                selection_root / f"{policy}.json",
            )
        atomic_json({
            "stage": "replaying",
            "planId": plan["id"],
            "epoch": epoch,
            "maximumEpoch": maximum_epoch,
            "pid": os.getpid(),
        }, run_root / "state/checkpoint-selection-recovery-status.json")


def selected_metric(metrics: dict, policy: str) -> float:
    source, metric, _direction = SELECTION_POLICIES[policy]
    if source == "train":
        return float(metrics["train"][metric])
    if source == "validation":
        return float(metrics["validation"][metric])
    split = "train" if source == "trainDistribution" else "validation"
    return float(metrics["distribution"][split][metric])


def evaluate_policies(
    repo: Path,
    plan: dict,
    run_root: Path,
    targets: dict[str, dict],
    train_dataset: NextReturnDataset,
    validation_dataset: NextReturnDataset,
    test_dataset: NextReturnDataset,
    lead_datasets: tuple[
        ActiveReturnPathDataset,
        ActiveReturnPathDataset,
        ActiveReturnPathDataset,
    ],
    normalization: dict,
    density: KnotDensityContract,
    device: torch.device,
    evaluation_batch_size: int,
    include_episode_metrics: bool,
) -> dict:
    model = fresh_model(plan, normalization, density, device)
    target_std = float(normalization["minuteStd"])
    policies: dict[str, dict] = {}
    plan_hash = canonical_hash(plan)
    for policy, event in targets.items():
        atomic_json({
            "stage": "evaluating-checkpoints",
            "planId": plan["id"],
            "policy": policy,
            "completedPolicies": len(policies),
            "totalPolicies": len(targets),
            "pid": os.getpid(),
        }, run_root / "state/checkpoint-selection-recovery-status.json")
        file = run_root / f"checkpoints/selections/{policy}.json"
        checkpoint = load_torch_checkpoint(
            file, map_location=device, weights_only=False
        )
        if checkpoint.get("planSha256") != plan_hash \
                or int(checkpoint["epoch"]) != int(event["epoch"]):
            raise ValueError(f"invalid recovered selection checkpoint: {policy}")
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.unexpected_keys \
                or set(incompatible.missing_keys) - {"prefix_conditioner"} \
                or ("prefix_conditioner" in incompatible.missing_keys
                    and model.prefix_conditioner.numel() != 0):
            raise ValueError(
                f"incompatible recovered selection checkpoint: {policy}"
            )
        train = evaluate(
            model, train_dataset, "train", batch_size=evaluation_batch_size,
            target_std=target_std, device=device,
        )
        validation = evaluate(
            model, validation_dataset, "validation",
            batch_size=evaluation_batch_size,
            target_std=target_std, device=device,
        )
        test = evaluate(
            model, test_dataset, "test", batch_size=evaluation_batch_size,
            target_std=target_std, device=device,
        )
        metrics = {
            "train": train["expectation"],
            "validation": validation["expectation"],
            "test": test["expectation"],
            "distribution": {
                "train": train,
                "validation": validation,
                "test": test,
            },
        }
        for split_name, lead_dataset in zip(
            ("train", "validation", "test"), lead_datasets, strict=True
        ):
            metrics["distribution"][split_name]["perLeadExpectation"] = (
                evaluate_autoregressive_leads(
                    model,
                    lead_dataset,
                    split_name,
                    lead_count=3,
                    batch_size=evaluation_batch_size,
                    target_std=target_std,
                    device=device,
                )
            )
        if include_episode_metrics:
            metrics["autoregressiveEpisodes"] = {
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
            metrics["sobolExpectedEpisodes"] = {
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
        recovered_score = selected_metric(metrics, policy)
        logged_score = event_policy_score(event, policy)
        policies[policy] = {
            "label": policy.replace("-", " ").title(),
            "epoch": int(event["epoch"]),
            "selectionScore": logged_score,
            "recoveredSelectionScore": recovered_score,
            "replayAbsoluteDifference": abs(recovered_score - logged_score),
            "selection": checkpoint["selection"],
            **metrics,
            "checkpoint": str(file.relative_to(repo)),
        }
    return {
        "contract": COMPARISON_CONTRACT,
        "planId": plan["id"],
        "planSha256": plan_hash,
        "policies": policies,
    }


def write_autoregressive_evaluation(
    comparison: dict,
    output: Path,
) -> None:
    atomic_json({
        "contract": AUTOREGRESSIVE_EVALUATION_CONTRACT,
        "planId": comparison["planId"],
        "policies": {
            policy: {
                "autoregressiveEpisodes": value["autoregressiveEpisodes"],
                "sobolExpectedEpisodes": value["sobolExpectedEpisodes"],
            }
            for policy, value in comparison["policies"].items()
        },
    }, output)


def recover_plan(
    repo: Path,
    plan_file: Path,
    *,
    evaluation_batch_size: int,
    device: torch.device,
    pause_file: Path | None,
    split_plan: dict,
    split_plan_file: Path,
    policies: tuple[str, ...],
    include_episode_metrics: bool,
) -> dict:
    plan_value = json.loads(plan_file.read_text(encoding="utf-8"))
    plan = plan_value.get("plan", plan_value)
    run_root = resolve(repo, Path(plan["runDir"]))
    if not (run_root / "state/result.json").is_file():
        raise ValueError("selection recovery requires a completed density run")
    plan_hash = canonical_hash(plan)
    events = read_epoch_events(run_root)
    all_targets = optimal_policy_events(events)
    targets = {policy: all_targets[policy] for policy in policies}
    recovered = materialize_existing_states(run_root, targets, plan_hash)
    history_root = resolve(repo, Path(plan["historyDir"]))
    train_dataset, validation_dataset, test_dataset = build_datasets(
        plan, history_root
    )
    lead_datasets = tuple(
        ActiveReturnPathDataset(
            dataset.shards, history_root, return_count=3
        )
        for dataset in (train_dataset, validation_dataset, test_dataset)
    )
    normalization = training_normalization(
        train_dataset,
        batch_size=int(plan["training"]["evaluationBatchSize"]),
    )
    density = KnotDensityContract.load(
        resolve(repo, Path(plan["density"]["source"])),
        fit=str(plan["density"]["fit"]),
    )
    replay_missing_states(
        repo, plan, run_root, targets, recovered, train_dataset,
        normalization, density, device, pause_file,
    )
    comparison = evaluate_policies(
        repo, plan, run_root, targets,
        train_dataset, validation_dataset, test_dataset,
        lead_datasets,
        normalization, density, device, evaluation_batch_size,
        include_episode_metrics,
    )
    atomic_json(
        comparison, run_root / "state/checkpoint-selection-comparison.json"
    )
    if include_episode_metrics:
        write_autoregressive_evaluation(
            comparison,
            run_root / "state/autoregressive-episode-evaluation.json",
        )
    atomic_json({
        "stage": "calibrating-checkpoints",
        "planId": plan["id"],
        "completedPolicies": 0,
        "totalPolicies": len(policies),
        "pid": os.getpid(),
    }, run_root / "state/checkpoint-selection-recovery-status.json")
    calibrate_checkpoint_selections(
        repo,
        plan_file,
        split_plan,
        split_plan_file,
        policies=policies,
        calibration_days=7,
        batch_size=evaluation_batch_size,
        device=device,
    )
    atomic_json({
        "stage": "complete",
        "planId": plan["id"],
        "policies": {
            policy: {"epoch": value["epoch"]}
            for policy, value in comparison["policies"].items()
        },
        "pid": os.getpid(),
    }, run_root / "state/checkpoint-selection-recovery-status.json")
    return comparison


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint recovery was requested but unavailable")
    pause_file = None if args.pause_file is None else resolve(
        repo, args.pause_file
    )
    split_plan_file = resolve(repo, args.split_plan)
    split_plan = json.loads(split_plan_file.read_text(encoding="utf-8"))
    completed: list[str] = []
    for value in args.training_plan:
        result = recover_plan(
            repo,
            resolve(repo, value),
            evaluation_batch_size=int(args.batch_size),
            device=device,
            pause_file=pause_file,
            split_plan=split_plan,
            split_plan_file=split_plan_file,
            policies=tuple(dict.fromkeys(args.checkpoint_policies)),
            include_episode_metrics=not bool(args.skip_episodes),
        )
        completed.append(str(result["planId"]))
        print(json.dumps({
            "event": "density-checkpoint-selection-complete",
            "planId": result["planId"],
            "policies": {
                policy: {
                    "epoch": value["epoch"],
                    "validationMseSkill": value["validation"]["mseSkillVsZero"],
                    "testMseSkill": value["test"]["mseSkillVsZero"],
                }
                for policy, value in result["policies"].items()
            },
        }, separators=(",", ":")), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({
        "contract": COMPARISON_CONTRACT,
        "completedPlans": completed,
    }, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
