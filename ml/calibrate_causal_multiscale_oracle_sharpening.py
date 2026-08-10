from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlp_model import PolicySupport, conditional_transaction_transition
from oracle_distribution_path import oracle_forward_kl_per_example
from return_oracle_decoder_screen import LearnedRadiusShrinkingDecoder
from trading_storage import load_torch_checkpoint
from train_causal_multiscale_oracle import CompactDataset, atomic_json


DEFAULT_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   8.0, 12.0, 16.0, 24.0, 32.0)
CURRENT_EXPOSURES = (-100.0, -50.0, 0.0, 50.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate direct-oracle logit sharpening on validation only."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "ml/training-plans/causal-multiscale-oracle-daily-v1.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--factors", type=str)
    return parser.parse_args()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    return {
        label: float(torch.quantile(values, value))
        for label, value in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95))
    }


def calibrate(plan: dict[str, Any], repo: Path, factors: tuple[float, ...], batch_size: int) -> dict[str, Any]:
    dataset_root = repo / plan["dataset"]["datasetDir"]
    manifest = json.loads((dataset_root / "dataset.json").read_text(encoding="utf-8"))
    run_root = repo / plan["runDir"]
    checkpoint = load_torch_checkpoint(
        run_root / "checkpoints" / "best.json",
        map_location="cuda",
        weights_only=False,
    )
    normalization = manifest["normalization"]
    model = LearnedRadiusShrinkingDecoder(
        torch.tensor(normalization["mean"]),
        torch.tensor(normalization["std"]),
        dropout=float(plan["architecture"]["dropout"]),
        dropout_rate=float(plan["architecture"]["dropoutRate"]),
        initial_radius=float(plan["architecture"]["initialRadius"]),
        minimum_radius=float(plan["architecture"]["minimumRadius"]),
        output_count=int(manifest["actionCount"]),
    ).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    dataset = CompactDataset(dataset_root, manifest)
    action_count = int(manifest["actionCount"])
    actions = torch.linspace(-100, 100, action_count, device="cuda")
    current = torch.tensor(CURRENT_EXPOSURES, device="cuda").view(1, -1)
    transition = conditional_transaction_transition(
        actions,
        current,
        PolicySupport(
            latent_lower=-100,
            latent_upper=100,
            visible_lower=-100,
            visible_upper=100,
            friction=float(plan["oracle"]["friction"]),
            temperature=float(plan["oracle"]["temperature"]),
        ),
    )[0]
    zero_action_index = action_count // 2
    zero_state_index = CURRENT_EXPOSURES.index(0.0)
    samples: dict[float, dict[str, Any]] = {
        factor: {
            "raw": [],
            "conditioned": [],
            "predictedNonzero": 0,
            "targetNonzero": 0,
            "modalAgreement": 0,
            "count": 0,
        }
        for factor in factors
    }
    probability_floor = float(plan["objective"]["probabilityFloor"])

    with torch.inference_mode():
        for features, targets in dataset.batches(
            "validation", batch_size, shuffle=False, seed=0
        ):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).float().clamp_min(0)
            targets = targets / targets.sum(dim=-1, keepdim=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(features)
            logits = logits.float()
            target_log = targets.clamp_min(torch.finfo(torch.float32).tiny).log()
            conditioned_targets = torch.softmax(
                target_log[:, None, :] + transition[None, :, :], dim=-1
            )
            target_zero_mode = conditioned_targets[:, zero_state_index].argmax(dim=-1)
            for factor in factors:
                predicted = torch.softmax(float(factor) * logits, dim=-1)
                raw_kl = oracle_forward_kl_per_example(
                    predicted, targets, probability_floor=probability_floor
                )
                conditioned_log = torch.log_softmax(
                    float(factor) * logits[:, None, :] + transition[None, :, :],
                    dim=-1,
                )
                conditioned_kl = (
                    conditioned_targets
                    * (
                        conditioned_targets.clamp_min(
                            torch.finfo(torch.float32).tiny
                        ).log()
                        - conditioned_log
                    )
                ).sum(dim=-1).mean(dim=-1)
                predicted_zero_mode = conditioned_log[:, zero_state_index].argmax(dim=-1)
                state = samples[factor]
                state["raw"].append(raw_kl.cpu())
                state["conditioned"].append(conditioned_kl.cpu())
                state["predictedNonzero"] += int(
                    (predicted_zero_mode != zero_action_index).sum()
                )
                state["targetNonzero"] += int(
                    (target_zero_mode != zero_action_index).sum()
                )
                state["modalAgreement"] += int(
                    (predicted_zero_mode == target_zero_mode).sum()
                )
                state["count"] += targets.shape[0]

    candidates = []
    for factor in factors:
        state = samples[factor]
        raw = torch.cat(state.pop("raw"))
        conditioned = torch.cat(state.pop("conditioned"))
        raw_mean = float(raw.mean())
        conditioned_mean = float(conditioned.mean())
        conditioned_q = quantiles(conditioned)
        candidates.append({
            "factor": factor,
            "rawMeanKl": raw_mean,
            "rawKlPercentiles": quantiles(raw),
            "rawSelectionObjective": raw_mean + float(torch.quantile(raw, 0.5)),
            "conditionedMeanKl": conditioned_mean,
            "conditionedKlPercentiles": conditioned_q,
            "conditionedSelectionObjective": (
                conditioned_mean + conditioned_q["p50"]
            ),
            "zeroStatePredictedNonzero": state["predictedNonzero"],
            "zeroStatePredictedNonzeroRate": state["predictedNonzero"] / state["count"],
            "zeroStateTargetNonzero": state["targetNonzero"],
            "zeroStateModalAgreement": state["modalAgreement"] / state["count"],
            "examples": state["count"],
        })
    selected = min(
        candidates,
        key=lambda value: (value["conditionedSelectionObjective"], value["factor"]),
    )
    result = {
        "contract": "validation-only-multistate-transition-conditioned-logit-sharpening-v1",
        "bestEpoch": int(checkpoint["epoch"]),
        "split": "validation",
        "sealedTestEvaluated": False,
        "currentExposures": list(CURRENT_EXPOSURES),
        "friction": float(plan["oracle"]["friction"]),
        "temperature": float(plan["oracle"]["temperature"]),
        "selectionMetric": "conditionedMeanKl+conditionedP50Kl",
        "selectedFactor": selected["factor"],
        "selected": selected,
        "candidates": candidates,
    }
    atomic_json(result, run_root / "calibration" / "logit-sharpening.json")
    return result


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    factors = (
        tuple(float(value) for value in args.factors.split(","))
        if args.factors
        else DEFAULT_FACTORS
    )
    if not factors or any(value <= 0 or not np.isfinite(value) for value in factors):
        raise ValueError("sharpening factors must be finite and positive")
    result = calibrate(plan, repo, factors, args.batch_size)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
