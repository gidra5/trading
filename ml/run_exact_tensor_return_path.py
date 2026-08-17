from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


MATRIX_ID = "next-return-exact-tensor-h3-k32-v1"
RUN_ID = "next-second-exact-tensor-density-65k-h3-k32-v1"
PAUSE_EXIT_CODE = 75
POLL_SECONDS = 2
PREDECESSOR_ARTIFACT = (
    "data/training/runs/"
    "next-second-8-layer-width1024-knot-density-1024k-k256-v1/"
    "state/result.json"
)


def derived_plan(source: dict) -> dict:
    plan = json.loads(json.dumps(source))
    plan["id"] = RUN_ID
    plan["label"] = (
        "Exact joint tensor density - three single-layer width-512 GLU blocks "
        "- 65k clean starts - 3 active returns x 32 knots"
    )
    plan["datasetDir"] = f"data/training/datasets/{RUN_ID}"
    plan["runDir"] = f"data/training/runs/{RUN_ID}"
    plan["subset"]["type"] = "fixed-clean-count"
    plan["evaluation"] = {
        "type": "fixed-clean-heldout-count",
        "validationStart": "2026-06-01",
        "testStart": "2026-07-01",
        "examplesPerSplit": 65_536,
    }
    plan["datasetFilter"] = {
        "type": "active-return-path",
        "startRows": "exclude-exact-zero-immediate-target-return",
        "pathTargets": "next-3-nonzero-log-returns",
        "inputHistory": "raw-120-one-second-log-returns-including-zeros",
        "appliesTo": ["normalization", "training", "validation", "test"],
    }
    base_architecture = plan["architecture"]
    plan["architecture"] = {
        "contract": (
            "three-return-exact-32-knot-joint-tensor-three-single-layer-"
            "normalized-glu-blocks-v1"
        ),
        "blocks": 3,
        "hiddenWidth": 512,
        "conditionalTensorEntries": [32, 1_024, 32_768],
        "initialRadius": base_architecture["initialRadius"],
        "minimumRadius": base_architecture["minimumRadius"],
        "learnableCentering": base_architecture["learnableCentering"],
    }
    plan["density"] = {
        "source": "data/benchmarks/one-second-return-knot-scaling-v1.json",
        "fit": "32", "knotCount": 32, "returnCount": 3,
        "jointTensorShape": [32, 32, 32],
        "normalization": "area-aware-softmax-on-final-axis-per-prefix",
        "factorization": "exact-full-prefix-chain-rule-tensors",
    }
    plan["training"]["batchSize"] = 256
    plan["training"]["evaluationBatchSize"] = 256
    plan["training"]["selection"] = (
        "lowest-validation-mean-conditional-negative-log-likelihood"
    )
    plan["training"].pop("targetNormalizedMse", None)
    return plan


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    matrix_root = repo / f"data/training/matrices/{MATRIX_ID}"
    plans_root = matrix_root / "plans"
    plans_root.mkdir(parents=True, exist_ok=True)
    pause_file = matrix_root / "control/PAUSE"
    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"
    subprocess.run(
        (sys.executable, str(repo / "ml/build_return_density_knot_scaling.py")),
        cwd=repo, env=environment, check=True,
    )
    source = json.loads((
        repo / "ml/training-plans/"
        "next-second-4-layer-memorization-65k-static-c-v1.json"
    ).read_text(encoding="utf-8"))
    plan = derived_plan(source)
    plan_file = plans_root / f"{RUN_ID}.json"
    plan_file.write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predecessor = repo / PREDECESSOR_ARTIFACT
    while not predecessor.is_file():
        while pause_file.is_file():
            time.sleep(POLL_SECONDS)
        time.sleep(POLL_SECONDS)
    result_file = repo / plan["runDir"] / "state/result.json"
    while not result_file.is_file():
        while pause_file.is_file():
            time.sleep(POLL_SECONDS)
        completed = subprocess.run((
            sys.executable,
            str(repo / "ml/train_exact_tensor_return_path.py"),
            "--plan", str(plan_file),
            "--pause-file", str(pause_file),
        ), cwd=repo, env=environment)
        if completed.returncode == PAUSE_EXIT_CODE:
            continue
        completed.check_returncode()
        if not result_file.is_file():
            raise RuntimeError("exact tensor trainer exited without a result")
    metrics_file = (
        repo / plan["runDir"] / "state/autoregressive-episode-evaluation.json"
    )
    while not metrics_file.is_file():
        while pause_file.is_file():
            time.sleep(POLL_SECONDS)
        completed = subprocess.run((
            sys.executable,
            str(repo / "ml/evaluate_exact_tensor_checkpoint_metrics.py"),
            "--plan", str(plan_file),
            "--pause-file", str(pause_file),
        ), cwd=repo, env=environment)
        if completed.returncode == PAUSE_EXIT_CODE:
            continue
        completed.check_returncode()
        if not metrics_file.is_file():
            raise RuntimeError(
                "exact tensor evaluator exited without path metrics"
            )


if __name__ == "__main__":
    main()
