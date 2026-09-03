"""Persist raw-checkpoint denoising diagnostics, never forecasting selections."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from evaluate_stopped_structured_feature_process import model_from_state
from trading_storage import load_torch_checkpoint, require_under, training_storage_layout
from train_normalized_glu_next_return import atomic_json
from train_structured_feature_process import (
    build_dataset,
    evaluate_hindsight_curriculum_probe,
    evaluate_hindsight_reconstruction,
    validate_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--variance", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 0 <= args.variance <= 1:
        raise ValueError("variance must be in [0, 1]")
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    if plan["architecture"].get("hindsightConditioning") is None:
        raise ValueError("plan has no hindsight readout")
    run_root = require_under(
        repo / plan["runDir"], training_storage_layout(repo).runs, "run directory",
    )
    output = run_root / "state/hindsight-reconstruction.json"
    if output.exists():
        raise ValueError(f"preserving existing diagnostic: {output}")
    checkpoint = load_torch_checkpoint(
        run_root / "checkpoints/last.json", map_location="cpu", weights_only=False,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    model = model_from_state(plan, checkpoint["model"], device)
    dataset = build_dataset(plan, repo)
    started = time.monotonic()
    try:
        common = {
            "variance": args.variance,
            "batch_size": int(plan["training"]["evaluationBatchSize"]),
            "device": device,
        }
        schedule = plan["training"]["hindsightNoiseSchedule"]
        train = evaluate_hindsight_curriculum_probe(model, dataset, schedule, **common)
        # Offline measurement does not advance even the training curriculum.
        train["usedForCurriculum"] = False
        validation = evaluate_hindsight_reconstruction(
            model, dataset, schedule, split="validation", **common,
        )
        artifact = {
            "contract": "hindsight-reconstruction-diagnostic-v1",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "planId": plan["id"], "epoch": int(checkpoint["epoch"]),
            "weightSource": "raw-training-weights", "noiseVariance": args.variance,
            "usedForCurriculum": False, "usedForCheckpointSelection": False,
            "train": train, "validation": validation,
            "seconds": time.monotonic() - started,
        }
        atomic_json(artifact, output)
        print(json.dumps({
            "planId": plan["id"], "epoch": artifact["epoch"],
            "noiseVariance": args.variance,
            "train": {key: train[key] for key in ("correlation", "normalizedMse", "mseSkillVsZero")},
            "validation": {key: validation[key] for key in ("correlation", "normalizedMse", "mseSkillVsZero")},
            "artifact": str(output), "seconds": artifact["seconds"],
        }))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
