from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from calibrate_next_return_output import calibrate_plan

from normalized_glu_next_return import (
    CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
    NormalizedGluNextReturn,
    TRAINING_POSITION_INPUT_NORMALIZATION,
)
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import direct_calendar_shards
from train_normalized_glu_next_return import (
    NextReturnDataset,
    atomic_json,
    canonical_fingerprint,
    evaluate,
    evaluate_daily_group_cvar,
    resolve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a next-return checkpoint on a held-out calendar split."
    )
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument(
        "--skip-output-calibration",
        action="store_true",
        help=(
            "Do not fit the standard pre-validation affine output calibration "
            "after validation evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    training_plan_file = resolve(repo, args.training_plan)
    split_plan_file = resolve(repo, args.split_plan)
    training_plan_value = json.loads(
        training_plan_file.read_text(encoding="utf-8")
    )
    training_plan = training_plan_value.get("plan", training_plan_value)
    split_plan = json.loads(split_plan_file.read_text(encoding="utf-8"))
    run_root = resolve(repo, Path(training_plan["runDir"]))
    checkpoint_file = resolve(
        repo,
        args.checkpoint or Path(training_plan["runDir"]) / "checkpoints/best.json",
    )
    output_file = resolve(
        repo,
        args.output
        or Path(training_plan["runDir"]) / f"state/{args.split}-current-best.json",
    )
    history_root = resolve(repo, Path(training_plan["historyDir"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but is unavailable")

    checkpoint = load_torch_checkpoint(
        checkpoint_file, map_location="cpu", weights_only=False
    )
    if checkpoint.get("planSha256") != canonical_fingerprint(training_plan):
        raise ValueError("checkpoint does not belong to the training plan")
    state = checkpoint["model"]
    architecture = training_plan["architecture"]
    model = NormalizedGluNextReturn(
        state["feature_mean"],
        state["feature_std"],
        state["target_mean"],
        state["target_std"],
        widths=tuple(int(value) for value in architecture["widths"]),
        input_normalization=architecture.get(
            "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
        ),
        volatility_window=architecture.get("volatilityWindow"),
        dropout=0,
        dropout_rate=0,
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture.get("learnableCentering", True)),
    )
    torch.set_float32_matmul_precision("high")
    model = model.to(device)

    shards = direct_calendar_shards(
        split_plan["split"],
        history_root,
        horizon_seconds=1,
        decision_stride_seconds=1,
    )
    dataset = NextReturnDataset(
        shards,
        history_root,
        horizon_return_count=1,
        row_stride=1,
        exclude_zero_targets=training_plan.get("datasetFilter") is not None,
    )
    candidate_states = {
        "raw-best": state,
        **checkpoint.get("swaCandidates", {}),
    }
    if args.split == "test" and len(candidate_states) > 1:
        validation_file = run_root / "state/validation-current-best.json"
        if not validation_file.is_file():
            raise ValueError(
                "validation must select an SWA candidate before test"
            )
        selected = json.loads(
            validation_file.read_text(encoding="utf-8")
        ).get("selectedCandidate")
        if selected not in candidate_states:
            raise ValueError("validation-selected candidate is unavailable")
        candidate_states = {selected: candidate_states[selected]}

    causal_volatility_window = (
        int(architecture["volatilityWindow"])
        if architecture.get("inputNormalization")
        == CAUSAL_VOLATILITY_INPUT_NORMALIZATION
        else None
    )
    candidate_metrics = {}
    cvar_dro = training_plan["training"].get("cvarDro")
    for candidate_id, candidate_state in candidate_states.items():
        model.load_state_dict(candidate_state)
        if cvar_dro is not None:
            metrics, daily_cvar = evaluate_daily_group_cvar(
                model, dataset, args.split,
                batch_size=args.batch_size,
                target_std=float(model.target_std.item()),
                tail_fraction=float(cvar_dro["tailFraction"]),
                device=device,
            )
            candidate_metrics[candidate_id] = {
                **metrics, "dailyCvar": daily_cvar
            }
        else:
            candidate_metrics[candidate_id] = evaluate(
                model,
                dataset,
                args.split,
                batch_size=args.batch_size,
                target_std=float(model.target_std.item()),
                device=device,
                amp_dtype=torch.float32,
                causal_volatility_window=causal_volatility_window,
            )
    selected_candidate = min(
        candidate_metrics,
        key=lambda name: float(candidate_metrics[name]["normalizedMse"]),
    )
    metrics = candidate_metrics[selected_candidate]
    train_candidates = {
        "raw-best": checkpoint.get("train", {}),
        **checkpoint.get("swaCandidateTrain", {}),
    }
    selected_train = train_candidates.get(selected_candidate, {})
    result = {
        "trainingPlanId": training_plan["id"],
        "trainingPlanSha256": canonical_fingerprint(training_plan),
        "checkpoint": str(checkpoint_file.relative_to(repo)),
        "checkpointEpoch": int(
            checkpoint.get("rawBestEpoch", checkpoint["epoch"])
            if selected_candidate == "raw-best" else checkpoint["epoch"]
        ),
        "checkpointTrainNormalizedMse": (
            float(selected_train["normalizedMse"])
            if selected_train.get("normalizedMse") is not None else None
        ),
        "splitPlanId": split_plan["id"],
        "split": args.split,
        "examples": dataset.logical_count(args.split),
        "candidateExamples": sum(
            shard.count for shard in shards[args.split]
        ),
        "datasetFilter": training_plan.get("datasetFilter"),
        "selectedCandidate": selected_candidate,
        "candidateMetrics": candidate_metrics,
        "metrics": metrics,
    }
    atomic_json(result, output_file)
    print(json.dumps(result, separators=(",", ":")))
    if args.split == "validation" and not args.skip_output_calibration:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        calibration = calibrate_plan(
            repo,
            training_plan_file,
            split_plan,
            split_plan_file,
            calibration_days=int(args.calibration_days),
            batch_size=int(args.batch_size),
            device=device,
            checkpoint_file=checkpoint_file,
            selected_candidate=selected_candidate,
        )
        print(json.dumps({
            "event": "output-calibration-complete",
            "trainingPlanId": training_plan["id"],
            "selectedCandidate": selected_candidate,
            "scaleOnly": calibration["transforms"]["scaleOnly"],
            "validation": calibration["validation"]["scaleOnly"],
            "test": calibration["test"]["scaleOnly"],
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
