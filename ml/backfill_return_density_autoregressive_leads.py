from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from active_return_path_dataset import ActiveReturnPathDataset
from evaluate_autoregressive_density_episodes import (
    evaluate_autoregressive_leads,
)
from recover_return_density_checkpoint_selections import (
    build_datasets,
    fresh_model,
    resolve,
)
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import training_normalization
from train_normalized_glu_next_return import atomic_json


LEAD_COUNT = 3
CONTRACT = "single-return-autoregressive-lead-evaluation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill per-lead metrics by recursively rolling a completed "
            "single-return density checkpoint forward."
        )
    )
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--policies", nargs="+")
    return parser.parse_args()


def backfill(
    repo: Path,
    plan_file: Path,
    *,
    batch_size: int,
    device: torch.device,
    requested_policies: tuple[str, ...] | None,
) -> dict:
    stored_plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan = stored_plan.get("plan", stored_plan)
    run_root = resolve(repo, Path(plan["runDir"]))
    comparison_file = run_root / "state/checkpoint-selection-comparison.json"
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    policies = comparison["policies"]
    selected = tuple(policies) if requested_policies is None else (
        requested_policies
    )
    unknown = set(selected) - set(policies)
    if unknown:
        raise ValueError(f"unknown checkpoint policies: {sorted(unknown)}")

    history_root = resolve(repo, Path(plan["historyDir"]))
    datasets = build_datasets(plan, history_root)
    lead_datasets = tuple(
        ActiveReturnPathDataset(
            dataset.shards,
            history_root,
            return_count=LEAD_COUNT,
        )
        for dataset in datasets
    )
    normalization = training_normalization(
        datasets[0],
        batch_size=int(plan["training"]["evaluationBatchSize"]),
    )
    density = KnotDensityContract.load(
        resolve(repo, Path(plan["density"]["source"])),
        fit=str(plan["density"]["fit"]),
    )
    model = fresh_model(plan, normalization, density, device)
    target_std = float(normalization["minuteStd"])
    status_file = run_root / "state/autoregressive-lead-evaluation-status.json"
    metrics_by_epoch: dict[int, dict[str, list[dict]]] = {}

    for policy_index, policy in enumerate(selected):
        value = policies[policy]
        distribution = value.setdefault("distribution", {})
        complete = all(
            len(distribution.get(split, {}).get("perLeadExpectation", ()))
            >= LEAD_COUNT
            for split in ("train", "validation", "test")
        )
        if complete:
            continue
        atomic_json({
            "contract": CONTRACT,
            "stage": "evaluating-autoregressive-leads",
            "planId": plan["id"],
            "policy": policy,
            "completedPolicies": policy_index,
            "totalPolicies": len(selected),
            "pid": os.getpid(),
        }, status_file)
        checkpoint_path = resolve(repo, Path(value["checkpoint"]))
        checkpoint = load_torch_checkpoint(
            checkpoint_path, map_location=device, weights_only=False
        )
        checkpoint_epoch = int(checkpoint["epoch"])
        cached = metrics_by_epoch.get(checkpoint_epoch)
        if cached is not None:
            for split in ("train", "validation", "test"):
                distribution.setdefault(split, {})[
                    "perLeadExpectation"
                ] = cached[split]
            atomic_json(comparison, comparison_file)
            continue
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.unexpected_keys \
                or set(incompatible.missing_keys) - {"prefix_conditioner"} \
                or ("prefix_conditioner" in incompatible.missing_keys
                    and model.prefix_conditioner.numel() != 0):
            raise ValueError(f"incompatible selection checkpoint: {policy}")
        epoch_metrics: dict[str, list[dict]] = {}
        for split, dataset in zip(
            ("train", "validation", "test"), lead_datasets, strict=True
        ):
            epoch_metrics[split] = evaluate_autoregressive_leads(
                model,
                dataset,
                split,
                lead_count=LEAD_COUNT,
                batch_size=batch_size,
                target_std=target_std,
                device=device,
            )
            distribution.setdefault(split, {})[
                "perLeadExpectation"
            ] = epoch_metrics[split]
        metrics_by_epoch[checkpoint_epoch] = epoch_metrics
        atomic_json(comparison, comparison_file)

    atomic_json({
        "contract": CONTRACT,
        "stage": "complete",
        "planId": plan["id"],
        "completedPolicies": len(selected),
        "totalPolicies": len(selected),
        "pid": os.getpid(),
    }, status_file)
    return comparison


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA lead evaluation was requested but unavailable")
    comparison = backfill(
        repo,
        resolve(repo, args.training_plan),
        batch_size=int(args.batch_size),
        device=device,
        requested_policies=(
            None if args.policies is None
            else tuple(dict.fromkeys(args.policies))
        ),
    )
    print(json.dumps({
        "event": "autoregressive-lead-evaluation-complete",
        "planId": comparison["planId"],
        "policies": list(comparison["policies"]),
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
