from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_next_return_memorization import (
    mean_teacher_variant,
    validate_plan,
    with_validation_curve,
)


BASE_PLAN = "next-second-4-layer-memorization-65k-static-c-v1.json"
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
MATRIX_ID = "nonzero-mean-teacher-ofat-v1"
WAIT_FOR_MATRIX = "nonzero-eiil-irm-annealing-v1"
VALIDATION_EXAMPLES = 65_536
PAUSE_EXIT_CODE = 75

DEFAULTS = {
    "halfLifeEpochs": 4.0,
    "consistencyWeight": 1.0,
    "rampUpFraction": 0.05,
    "inputPerturbationRms": 0.01,
}
HALF_LIFE_EPOCHS = (4.0, 8.0, 16.0)
CONSISTENCY_WEIGHTS = (1.0, 3.0, 10.0, 0.5)
RAMP_UP_FRACTIONS = (0.05, 0.33, 0.66, 1.0, 0.0)
INPUT_PERTURBATION_RMS = (0.01, 0.1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the IRM annealing sweep, then run a sequential, "
            "one-factor-at-a-time Mean Teacher sweep on 65,536 clean examples."
        )
    )
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def cases() -> list[dict[str, float | str]]:
    result: list[dict[str, float | str]] = [
        {"axis": "default", **DEFAULTS}
    ]
    for value in HALF_LIFE_EPOCHS[1:]:
        result.append({
            "axis": "halfLifeEpochs",
            **DEFAULTS,
            "halfLifeEpochs": value,
        })
    for value in CONSISTENCY_WEIGHTS[1:]:
        result.append({
            "axis": "consistencyWeight",
            **DEFAULTS,
            "consistencyWeight": value,
        })
    for value in RAMP_UP_FRACTIONS[1:]:
        result.append({
            "axis": "rampUpFraction",
            **DEFAULTS,
            "rampUpFraction": value,
        })
    for value in INPUT_PERTURBATION_RMS[1:]:
        result.append({
            "axis": "inputPerturbationRms",
            **DEFAULTS,
            "inputPerturbationRms": value,
        })
    return result


def suffix(case: dict[str, float | str]) -> str:
    return (
        "nonzero-mean-teacher-"
        f'hl-{token(float(case["halfLifeEpochs"]))}-'
        f'w-{token(float(case["consistencyWeight"]))}-'
        f'ramp-{token(float(case["rampUpFraction"]))}-'
        f'eps-{token(float(case["inputPerturbationRms"]))}-'
        "validation-curve-v1"
    )


def planned_variant(source: dict, case: dict[str, float | str]) -> dict:
    variant = mean_teacher_variant(
        source,
        half_life_epochs=float(case["halfLifeEpochs"]),
        consistency_weight=float(case["consistencyWeight"]),
        ramp_up_fraction=float(case["rampUpFraction"]),
        input_perturbation_rms=float(case["inputPerturbationRms"]),
        suffix=suffix(case),
    )
    return with_validation_curve(
        variant,
        split_plan=f"ml\\training-plans\\{SPLIT_PLAN}",
        examples=VALIDATION_EXAMPLES,
    )


def write_state(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    source_plan_file = plans_root / BASE_PLAN
    source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
    split_plan = plans_root / SPLIT_PLAN
    split_argument = str(split_plan.relative_to(repo))
    trainer = repo / "ml/train_next_return_memorization.py"
    evaluator = repo / "ml/evaluate_next_return_checkpoint.py"
    matrix_root = repo / "data/training/matrices" / MATRIX_ID
    pause_file = matrix_root / "control/PAUSE"
    wait_for = (
        repo / "data/training/matrices" / WAIT_FOR_MATRIX / "state/schedule.json"
    )
    sweep_cases = cases()
    manifest = {
        "matrixId": MATRIX_ID,
        "strategy": "one-factor-at-a-time",
        "basePlan": BASE_PLAN,
        "cleanTrainingExamples": int(source_plan["subset"]["examples"]),
        "validationExamples": VALIDATION_EXAMPLES,
        "defaults": DEFAULTS,
        "axes": {
            "halfLifeEpochs": HALF_LIFE_EPOCHS,
            "consistencyWeight": CONSISTENCY_WEIGHTS,
            "rampUpFraction": RAMP_UP_FRACTIONS,
            "inputPerturbationRms": INPUT_PERTURBATION_RMS,
        },
        "runs": [
            {
                **case,
                "runId": f'{source_plan["id"]}-{suffix(case)}',
            }
            for case in sweep_cases
        ],
    }
    for case in sweep_cases:
        validate_plan(planned_variant(source_plan, case))
    if args.validate_only:
        print(json.dumps(manifest, indent=2))
        return

    write_state(matrix_root / "state/queue.json", {
        **manifest,
        "status": "waiting-for-predecessor",
        "waitFor": str(wait_for.relative_to(repo)),
    })
    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"

    def wait_until_resumed() -> None:
        while pause_file.is_file():
            time.sleep(args.poll_seconds)

    def run_resumable(command: tuple[str, ...], completion_file: Path) -> None:
        if completion_file.is_file():
            return
        attempts = 0
        while not completion_file.is_file():
            wait_until_resumed()
            completed = subprocess.run(command, cwd=repo, env=environment)
            if completed.returncode == PAUSE_EXIT_CODE:
                wait_until_resumed()
                continue
            if completed.returncode == 0 and completion_file.is_file():
                return
            attempts += 1
            if attempts >= 3:
                completed.check_returncode()
                raise RuntimeError(
                    f"command completed without durable output: {completion_file}"
                )
            time.sleep(args.poll_seconds)

    while not wait_for.is_file():
        wait_until_resumed()
        time.sleep(args.poll_seconds)

    write_state(matrix_root / "state/queue.json", {
        **manifest,
        "status": "running",
        "waitFor": str(wait_for.relative_to(repo)),
    })
    validation_arguments = (
        "--validation-curve-split-plan",
        split_argument,
        "--validation-curve-examples",
        str(VALIDATION_EXAMPLES),
    )
    for case in sweep_cases:
        run_suffix = suffix(case)
        run_root = repo / f'{source_plan["runDir"]}-{run_suffix}'
        run_resumable((
            sys.executable,
            str(trainer),
            "--plan", str(source_plan_file),
            "--mean-teacher-half-life-epochs",
            str(case["halfLifeEpochs"]),
            "--mean-teacher-consistency-weight",
            str(case["consistencyWeight"]),
            "--mean-teacher-ramp-up-fraction",
            str(case["rampUpFraction"]),
            "--mean-teacher-input-perturbation-rms",
            str(case["inputPerturbationRms"]),
            "--variant-suffix", run_suffix,
            *validation_arguments,
            "--pause-file", str(pause_file),
        ), run_root / "state/result.json")
        run_resumable((
            sys.executable,
            str(evaluator),
            "--training-plan", str(run_root / "state/plan.json"),
            "--split-plan", str(split_plan),
            "--split", "validation",
            "--batch-size", "8192",
        ), run_root / "state/output-calibration-pre-validation-7d.json")

    write_state(matrix_root / "state/schedule.json", {
        **manifest,
        "status": "complete",
    })
    write_state(matrix_root / "state/queue.json", {
        **manifest,
        "status": "complete",
        "waitFor": str(wait_for.relative_to(repo)),
    })


if __name__ == "__main__":
    main()
