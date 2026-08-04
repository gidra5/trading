from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from return_oracle_decoder_screen import (
    FEATURE_CONTRACT,
    INPUT_RETURN_COUNT,
    OUTPUT_ACTION_COUNT,
    RUNNER_CONTRACT,
    SELECTION_CONTRACT,
    build_decoder,
    validate_screen_plan,
)
from trading_storage import load_torch_checkpoint, require_under


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the best return-oracle decoder checkpoint to ONNX."
    )
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def export_best_artifact(plan_file: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = resolve(repo_root, plan_file)
    live_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    run_dir = require_under(
        resolve(repo_root, Path(live_plan["runDir"])),
        repo_root / "data" / "training" / "runs",
        "runDir",
    )
    plan_snapshot = json.loads(
        (run_dir / "state" / "plan.json").read_text(encoding="utf-8")
    )
    source_plan = plan_snapshot["plan"]
    validate_screen_plan(source_plan)
    dataset_file = resolve(
        repo_root, Path(source_plan["dataset"]["datasetDir"]) / "dataset.json"
    )
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    statistics = dataset["featureStandardization"]
    model = build_decoder(
        source_plan,
        torch.tensor(statistics["mean"], dtype=torch.float32),
        torch.tensor(statistics["std"], dtype=torch.float32),
    )
    checkpoint = load_torch_checkpoint(
        run_dir / "checkpoints" / "best.json",
        map_location="cpu",
        weights_only=False,
    )
    validate_export_checkpoint(
        checkpoint,
        plan_snapshot,
        sum(parameter.numel() for parameter in model.parameters()),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    artifact_root = repo_root / "data" / "models" / "return-oracle-decoder"
    artifact_dir = require_under(
        artifact_root / source_plan["id"], artifact_root, "artifactDir"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_file = artifact_dir / "model.onnx"
    temporary = artifact_dir / f"model.onnx.{os.getpid()}.tmp"
    example = torch.zeros(2, INPUT_RETURN_COUNT, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        temporary,
        input_names=["minute_returns"],
        output_names=["action_logits"],
        dynamic_axes={
            "minute_returns": {0: "batch"},
            "action_logits": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )
    exported = onnx.load(temporary, load_external_data=True)
    onnx.checker.check_model(exported, full_check=True)
    with torch.inference_mode():
        expected = model(example).numpy()
    actual = ReferenceEvaluator(exported).run(
        ["action_logits"], {"minute_returns": example.numpy()}
    )[0]
    maximum_export_error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(actual).all() or maximum_export_error > 1e-3:
        raise ValueError(
            "ONNX output does not match the best PyTorch checkpoint: "
            f"max abs error {maximum_export_error}"
        )
    os.replace(temporary, output_file)

    manifest = {
        "version": 1,
        "kind": "return-oracle-decoder",
        "id": source_plan["id"],
        "label": source_plan["label"],
        "createdAt": iso_now(),
        "modelFile": output_file.name,
        "modelSha256": sha256(output_file),
        "sourcePlanSha256": checkpoint["planSha256"],
        "architectureContract": checkpoint["architectureContract"],
        "featureContract": FEATURE_CONTRACT,
        "selectionContract": SELECTION_CONTRACT,
        "runnerContract": RUNNER_CONTRACT,
        "input": {
            "name": "minute_returns",
            "dtype": "float32",
            "shape": ["batch", INPUT_RETURN_COUNT],
            "semantics": dataset["featureSemantics"],
            "normalization": "embedded",
        },
        "output": {
            "name": "action_logits",
            "dtype": "float32",
            "shape": ["batch", OUTPUT_ACTION_COUNT],
            "actionGrid": dataset["actionGrid"],
        },
        "training": {
            "bestEpoch": checkpoint["epoch"],
            "globalStep": checkpoint["globalStep"],
            "validation": checkpoint["validation"],
            "testPolicy": "sealed-never-load",
        },
        "verification": {
            "maximumAbsoluteLogitError": maximum_export_error,
        },
    }
    atomic_json(manifest, artifact_dir / "manifest.json")
    return manifest


def validate_export_checkpoint(
    checkpoint: dict[str, Any],
    plan_snapshot: dict[str, Any],
    parameter_count: int,
) -> None:
    plan = plan_snapshot["plan"]
    if checkpoint.get("planSha256") != plan_snapshot.get("planSha256") \
            or checkpoint.get("architectureContract") \
            != plan["architecture"]["contract"] \
            or checkpoint.get("featureContract") != FEATURE_CONTRACT \
            or checkpoint.get("selectionContract") != SELECTION_CONTRACT \
            or checkpoint.get("runnerContract") != RUNNER_CONTRACT \
            or checkpoint.get("parameterCount") != parameter_count \
            or checkpoint.get("epoch") != checkpoint.get("bestEpoch") \
            or checkpoint.get("sealedTestEvaluated") is not False:
        raise ValueError("best checkpoint does not match the recorded screen run")


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: dict[str, Any], file: Path) -> None:
    temporary = file.with_suffix(file.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    exported = export_best_artifact(parse_args().plan)
    print(json.dumps({
        "artifactId": exported["id"],
        "bestEpoch": exported["training"]["bestEpoch"],
        "modelSha256": exported["modelSha256"],
        "maximumAbsoluteLogitError": (
            exported["verification"]["maximumAbsoluteLogitError"]
        ),
    }))
