from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import onnx
import torch

from joint_price_oracle import ARCHITECTURE_CONTRACT
from trading_storage import load_torch_checkpoint, require_under
from train_joint_price_oracle import DATA_CONTRACT, build_model, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the best verified joint price-oracle checkpoint to ONNX."
    )
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def export_best_artifact(plan_file: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_plan = resolve(repo_root, plan_file)
    plan = json.loads(resolved_plan.read_text(encoding="utf-8"))
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        repo_root / "data" / "training" / "runs",
        "runDir",
    )
    artifact_dir = require_under(
        resolve(repo_root, Path(plan["artifactDir"])),
        repo_root / "data" / "models" / "joint-price-oracle",
        "artifactDir",
    )
    checkpoint_file = run_dir / "checkpoints" / "best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("planId") != plan["id"] \
            or checkpoint.get("architectureContract") != ARCHITECTURE_CONTRACT \
            or checkpoint.get("dataContract") != DATA_CONTRACT \
            or checkpoint.get("modelConfig") != plan["model"]:
        raise ValueError("best checkpoint does not match the export plan")
    model = build_model(plan["model"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_file = artifact_dir / "model.onnx"
    temporary = artifact_dir / f"model.onnx.{os.getpid()}.tmp"
    context_length = int(plan["model"]["contextLength"])
    example = torch.ones(1, context_length, 1, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        temporary,
        input_names=["closes"],
        output_names=["action_logits"],
        dynamic_axes={
            "closes": {0: "batch"},
            "action_logits": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )
    exported = onnx.load(temporary, load_external_data=True)
    onnx.checker.check_model(exported, full_check=True)
    os.replace(temporary, output_file)
    target_files = sorted(resolve(
        repo_root,
        Path(plan["targetReferenceDir"]),
    ).glob("*.json"))
    if not target_files:
        raise FileNotFoundError("verified oracle target references are missing")
    target_reference = json.loads(target_files[0].read_text(encoding="utf-8"))
    contract = target_reference["metadata"]["contract"]
    model_hash = hashlib.sha256(output_file.read_bytes()).hexdigest()
    status_file = run_dir / "state" / "status.json"
    status = (
        json.loads(status_file.read_text(encoding="utf-8"))
        if status_file.is_file()
        else {}
    )
    manifest = {
        "version": 1,
        "kind": "joint-price-oracle",
        "id": plan["id"],
        "label": plan["label"],
        "createdAt": iso_now(),
        "modelFile": output_file.name,
        "modelSha256": model_hash,
        "architectureContract": ARCHITECTURE_CONTRACT,
        "dataContract": DATA_CONTRACT,
        "input": {
            "name": "closes",
            "dtype": "float32",
            "intervalMs": plan["samplingIntervalMs"],
            "contextLength": context_length,
            "variableCount": 1,
        },
        "output": {
            "name": "action_logits",
            "dtype": "float32",
            "actionCount": plan["model"]["actionCount"],
            "actionGrid": contract["usableGrid"],
        },
        "oracle": contract,
        "training": {
            "bestEpoch": checkpoint["epoch"],
            "globalStep": checkpoint["globalStep"],
            "validation": checkpoint["validation"],
            "test": status.get("test"),
            "datasetFingerprint": checkpoint["datasetFingerprint"],
        },
    }
    atomic_json(manifest, artifact_dir / "manifest.json")
    return manifest


def atomic_json(value: dict, file: Path) -> None:
    temporary = file.with_suffix(file.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    exported_manifest = export_best_artifact(parse_args().plan)
    print(json.dumps({
        "artifactId": exported_manifest["id"],
        "bestEpoch": exported_manifest["training"]["bestEpoch"],
        "modelSha256": exported_manifest["modelSha256"],
    }))
