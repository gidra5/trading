from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mlp_model import ParameterExposureMlp as ExposureMlp, PolicySupport
from trading_storage import load_torch_checkpoint
from train_mlp import (
    FittedPolicyDataset,
    atomic_json,
    cached_training_parameter_scale,
    deterministic_current_states,
    emit,
    evaluate,
    export_artifact,
    loader,
    parse_loss_weights,
    parse_time_weighting,
    resolve_device,
    set_determinism,
    teacher_fit_summary,
    validate_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and evaluate an already-trained MLP study checkpoint."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-file", type=Path, required=True)
    parser.add_argument("--target-statistics-cache", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, required=True)
    parser.add_argument("--states-per-example", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), required=True)
    parser.add_argument("--selection-metric", choices=("loss", "klDivergence"), required=True)
    parser.add_argument("--loss-weights-json", required=True)
    parser.add_argument("--time-weighting-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_batch_size < 1 or args.states_per_example < 1 or args.workers < 0:
        raise ValueError("invalid study artifact export batch configuration")
    set_determinism(args.seed)
    manifest = json.loads((args.dataset / "dataset.json").read_text())
    validate_dataset_manifest(manifest, args.dataset)
    study = json.loads(args.study_file.read_text())
    state = load_torch_checkpoint(
        args.output / "checkpoints" / "best.json",
        map_location="cpu",
        weights_only=True,
    )
    train = FittedPolicyDataset(
        manifest, args.dataset, "train", target="teacherParameters",
    )
    validation = FittedPolicyDataset(
        manifest, args.dataset, "validation", target="teacherParameters",
    )
    test = FittedPolicyDataset(
        manifest, args.dataset, "test", target="teacherParameters",
    )
    device = resolve_device(args.device)
    model = ExposureMlp(state["feature_mean"], state["feature_std"], args.dropout).to(device)
    model.load_state_dict(state)
    support = PolicySupport(**manifest["policySupport"])
    actions = torch.linspace(
        support.visible_lower,
        support.visible_upper,
        int(manifest["actionCount"]),
        dtype=torch.float32,
        device=device,
    )
    current = deterministic_current_states(
        args.states_per_example, support, device, visible=True
    )
    parameter_scale = cached_training_parameter_scale(
        train, args.target_statistics_cache
    ).to(device)
    loss_weights = parse_loss_weights(args.loss_weights_json)
    time_weighting = parse_time_weighting(args.time_weighting_json)
    test_loader = loader(
        test,
        args,
        shuffle=False,
        batch_size=args.evaluation_batch_size,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        actions,
        current,
        support,
        parameter_scale,
        loss_weights,
        device,
        int(manifest["samplingIntervalMs"]),
    )
    teacher_metrics = teacher_fit_summary(manifest, args.dataset)
    export_artifact(
        model,
        args,
        manifest,
        len(train),
        len(validation),
        len(test),
        int(study["bestEpoch"]),
        study["bestValidationMetrics"],
        test_metrics,
        teacher_metrics,
        loss_weights,
        time_weighting,
        device,
        bool(study.get("finalizedEarly", False)),
    )
    curriculum = study.get("curriculumTraining")
    if isinstance(curriculum, dict):
        artifact_manifest_file = args.output / "manifest.json"
        artifact_manifest = json.loads(artifact_manifest_file.read_text())
        artifact_manifest["training"]["curriculum"] = curriculum
        atomic_json(artifact_manifest, artifact_manifest_file)
    study["testMetrics"] = test_metrics
    study["artifact"] = str(args.output / "model.onnx")
    atomic_json(study, args.study_file)
    emit({
        "event": "study-artifact-exported",
        "artifact": study["artifact"],
        "test": test_metrics,
    })


if __name__ == "__main__":
    main()
