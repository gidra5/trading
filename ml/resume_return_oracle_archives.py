from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SECOND_MS = 1_000
MINUTE_MS = 60_000
DAY_MS = 86_400_000
INPUT_PURGE_MS = 60 * MINUTE_MS
TEST_EXAMPLES = 1_000_000
TARGET_EPOCHS = 200


@dataclass(frozen=True)
class ArchiveRun:
    id: str
    tree: str
    run_directory: str
    expected_last_epoch: int
    expected_architecture: str
    expected_objective: str


RUNS = (
    ArchiveRun(
        "return-oracle-independent-p01-w002-softln1-20260729-195620",
        "8b81271bfff42881835d40f2c6fc5576e9976323",
        "return-oracle-ce-shrinking-v1-independent-p01-w002-softln1-baseline-20260729-195620",
        91,
        "shrinking-fused-glu-learnable-shared-centering-bias-ln-full-a-v12",
        "soft-target-ce-plus-independent-one-percent-skew-reverse-kl-entropy-sharpness-soft-layernorm-centering-weight-regularizers-v12",
    ),
    ArchiveRun(
        "return-oracle-shared-c-sharpness-p1-20260729-205615",
        "8e1e1aca51e942fd15e314d8140d23533cc68b6f",
        "return-oracle-ce-shrinking-v1-shared-c-sharpness-p1-baseline-20260729-205615",
        135,
        "shrinking-fused-glu-learnable-shared-centering-bias-ln-full-a-v12",
        "soft-target-ce-plus-independent-skew-reverse-kl-p01-entropy-sharpness-p1-soft-layernorm-centering-weight-regularizers-v13",
    ),
    ArchiveRun(
        "return-oracle-independent-c-reverse-p01-sharpness-p1-20260729-231900",
        "77ab2598fe83d72ff6b4cfbce9f4797f50b82b3a",
        "return-oracle-ce-shrinking-v1-independent-c-reverse-p01-sharpness-p1-baseline-20260729-231900",
        159,
        "shrinking-fused-glu-independent-branch-centering-bias-ln-full-a-v13",
        "soft-target-ce-plus-independent-skew-reverse-kl-p01-entropy-sharpness-p1-soft-layernorm-centering-weight-regularizers-v13",
    ),
    ArchiveRun(
        "return-oracle-hard-rms-reverse-p1-sharpness-p1-20260729-235300",
        "90576fd8877fe8b30ca9c7a7edb915c18464d387",
        "return-oracle-ce-shrinking-v1-hard-rms-reverse-p1-sharpness-p1-baseline-20260729-235300",
        76,
        "shrinking-fused-glu-independent-branch-centering-bias-ln-full-a-v13",
        "soft-target-ce-plus-independent-skew-reverse-kl-p1-entropy-sharpness-p1-soft-layernorm-centering-weight-regularizers-v14",
    ),
    ArchiveRun(
        "return-oracle-earlypeak3-ce10-20260730-171419",
        "3902cb9b6df85f489786a72ea33613d4cdbde9e3",
        "return-oracle-ce-shrinking-v1-earlypeak3-ce10-baseline-20260730-171419",
        110,
        "early-peak60to1024to255-fused-glu-independent-branch-centering-sqrt-learned-radius-full-a-post-bias-v21",
        "soft-target-ce-plus-independent-skew-reverse-kl-p1-entropy-sharpness-p1-soft-layernorm-centering-weight-regularizers-v15",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume preserved return-oracle experiments exactly to epoch 200."
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only", choices=[run.id for run in RUNS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    state_root = repo_root / "data" / "training" / "archive-resume"
    state_root.mkdir(parents=True, exist_ok=True)
    selected = tuple(reversed(
        tuple(run for run in RUNS if args.only in (None, run.id))
    ))
    prepared = [prepare_run(repo_root, state_root, run) for run in selected]
    write_json(state_root / "queue.json", {
        "version": 1,
        "targetEpochs": TARGET_EPOCHS,
        "preparedAt": iso_now(),
        "runs": [item["summary"] for item in prepared],
    })
    if args.prepare_only:
        print(json.dumps({
            "event": "archive-resume-prepared",
            "runs": [item["summary"] for item in prepared],
        }, separators=(",", ":")), flush=True)
        return

    status_file = state_root / "status.json"
    queue_log = state_root / "queue.log"
    for index, item in enumerate(prepared):
        summary = item["summary"]
        if int(summary["completedEpochs"]) >= TARGET_EPOCHS:
            continue
        status = {
            "stage": "running",
            "startedAt": iso_now(),
            "index": index,
            "runs": len(prepared),
            "run": summary,
            "pid": os.getpid(),
        }
        write_json(status_file, status)
        environment = compiler_environment()
        environment["RETURN_ORACLE_REPO_ROOT"] = str(repo_root)
        compile_cache = state_root / "compile-cache"
        torchinductor_cache = compile_cache / "torchinductor"
        triton_cache = compile_cache / "triton"
        torchinductor_cache.mkdir(parents=True, exist_ok=True)
        triton_cache.mkdir(parents=True, exist_ok=True)
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache)
        environment["TRITON_CACHE_DIR"] = str(triton_cache)
        environment["PYTHONUTF8"] = "1"
        triton_tcc = (
            repo_root / ".venv-ml" / "Lib" / "site-packages"
            / "triton" / "runtime" / "tcc" / "tcc.exe"
        )
        if not triton_tcc.exists():
            raise RuntimeError(f"bundled Triton compiler is unavailable: {triton_tcc}")
        environment["CC"] = str(triton_tcc)
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(item["runtime_ml"]),
            str(repo_root / "ml"),
            environment.get("PYTHONPATH"),
        )))
        command = [
            str(repo_root / ".venv-ml" / "Scripts" / "python.exe"),
            str(item["trainer"]),
            "--plan",
            str(item["plan"]),
        ]
        with queue_log.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps({
                "event": "archive-resume-launch",
                "at": iso_now(),
                "run": summary,
                "command": command,
            }, separators=(",", ":")) + "\n")
            output.flush()
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            write_json(status_file, {
                **status,
                "stage": "failed",
                "failedAt": iso_now(),
                "returnCode": completed.returncode,
            })
            raise SystemExit(completed.returncode)
    write_json(status_file, {
        "stage": "complete",
        "completedAt": iso_now(),
        "targetEpochs": TARGET_EPOCHS,
        "runs": [item["summary"] for item in prepared],
    })


def prepare_run(repo_root: Path, state_root: Path, run: ArchiveRun) -> dict[str, Any]:
    runtime = state_root / "runtime" / run.id
    runtime_ml = runtime / "ml"
    runtime_ml.mkdir(parents=True, exist_ok=True)
    trainer_source = git_text(repo_root, run.tree, "ml/train_return_oracle_ce.py")
    model_source = git_text(repo_root, run.tree, "ml/return_oracle_ce.py")
    experiment_plan = json.loads(git_text(
        repo_root,
        run.tree,
        "ml/training-plans/return-oracle-ce-shrinking-v1.json",
    ))
    source_plan = json.loads(git_text(
        repo_root,
        run.tree,
        "ml/training-plans/mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle.json",
    ))
    if experiment_plan["training"]["architecture"] != run.expected_architecture:
        raise ValueError(f"historical architecture mismatch for {run.id}")
    if experiment_plan["training"]["objective"] != run.expected_objective:
        raise ValueError(f"historical objective mismatch for {run.id}")

    run_dir = repo_root / "data" / "training" / "runs" / run.run_directory
    checkpoint = checkpoint_reference(run_dir / "checkpoints" / "last.json")
    checkpoint_epoch = checkpoint_metadata(repo_root, run_dir, checkpoint)
    if checkpoint_epoch < run.expected_last_epoch:
        raise ValueError(
            f"checkpoint for {run.id} predates the archived state: "
            f"{checkpoint_epoch} < {run.expected_last_epoch}"
        )
    expected = historical_dataset_summary(run_dir / "logs" / "training.log")
    current_source = (
        repo_root / "data" / "training" / "datasets"
        / "mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle"
    )
    source_dataset = (
        repo_root / "data" / "training" / "datasets"
        / "archive-resume-source" / run.id
    )
    feature_dataset = (
        repo_root / "data" / "training" / "datasets"
        / "archive-resume-features" / run.id
    )
    filtered_manifest, selected_dates, counts = reconstruct_manifest(
        current_source,
        source_plan,
    )
    if counts != expected["counts"] or len(selected_dates) != expected["featureComponents"]:
        raise ValueError(
            f"reconstructed data mismatch for {run.id}: "
            f"counts={counts}, dates={len(selected_dates)}, expected={expected}"
        )
    write_json(source_dataset / "dataset.json", filtered_manifest)
    copy_source_references(current_source, source_dataset, filtered_manifest["shards"])
    copy_feature_references(repo_root, feature_dataset, selected_dates)
    copy_feature_statistics(
        repo_root,
        feature_dataset,
        expected["counts"]["train"],
        trainer_source,
    )

    trainer_file = runtime_ml / "train_return_oracle_ce.py"
    model_file = runtime_ml / "return_oracle_ce.py"
    trainer_file.write_text(patch_trainer(trainer_source), encoding="utf-8", newline="\n")
    model_file.write_text(model_source, encoding="utf-8", newline="\n")
    experiment_plan["sourceDatasetDir"] = str(source_dataset.relative_to(repo_root))
    experiment_plan["datasetDir"] = str(feature_dataset.relative_to(repo_root))
    experiment_plan["runDir"] = str(run_dir.relative_to(repo_root))
    experiment_plan["historyDir"] = str(
        Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s")
    )
    experiment_plan["training"]["epochs"] = TARGET_EPOCHS
    plan_file = runtime / "plan.json"
    write_json(plan_file, experiment_plan)
    return {
        "trainer": trainer_file,
        "runtime_ml": runtime_ml,
        "plan": plan_file,
        "summary": {
            "id": run.id,
            "tree": run.tree,
            "runDir": str(run_dir.relative_to(repo_root)).replace("\\", "/"),
            "checkpointEpoch": checkpoint_epoch,
            "completedEpochs": checkpoint_epoch + 1,
            "remainingEpochs": max(0, TARGET_EPOCHS - checkpoint_epoch - 1),
            "architecture": run.expected_architecture,
            "objective": run.expected_objective,
            "counts": counts,
            "featureComponents": len(selected_dates),
        },
    }


def compiler_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if os.name != "nt" or shutil.which("cl.exe", path=environment.get("PATH")):
        return environment

    program_files_x86 = next(
        (value for name, value in environment.items() if name.lower() == "programfiles(x86)"),
        None,
    )
    if not program_files_x86:
        raise RuntimeError("ProgramFiles(x86) is unavailable; cannot locate MSVC")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        raise RuntimeError(f"Visual Studio locator is unavailable: {vswhere}")
    installation = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not installation:
        raise RuntimeError("Visual Studio C++ Build Tools are not installed")
    vsdevcmd = Path(installation) / "Common7" / "Tools" / "VsDevCmd.bat"
    completed = subprocess.run(
        f'call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 >nul && set',
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        for existing in tuple(environment):
            if existing != name and existing.lower() == name.lower():
                del environment[existing]
        environment[name] = value
    compiler_path = next(
        (value for name, value in environment.items() if name.lower() == "path"),
        None,
    )
    if not shutil.which("cl.exe", path=compiler_path):
        raise RuntimeError(f"MSVC environment from {vsdevcmd} does not expose cl.exe")
    return environment


def reconstruct_manifest(
    source_root: Path,
    source_plan: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    manifest = json.loads((source_root / "dataset.json").read_text(encoding="utf-8"))
    train_ranges, validation_ranges = window_ranges(source_plan)
    split_anchor = parse_day(source_plan["splitAnchorDate"])
    test_end = split_anchor + DAY_MS
    test_start = test_end - int(source_plan["latestTestDays"]) * DAY_MS

    def expected_split(timestamp: int) -> str | None:
        if test_start <= timestamp < test_end:
            return "test"
        if any(start <= timestamp < end for start, end in validation_ranges):
            return "validation"
        if any(start <= timestamp < end for start, end in train_ranges):
            return "train"
        return None

    selected: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        start = int(shard["predictionTimeStart"])
        end = start + (int(shard["count"]) - 1) * SECOND_MS
        if expected_split(start) == shard["split"] == expected_split(end):
            selected.append(shard)
    selected.sort(key=lambda value: int(value["predictionTimeStart"]))
    counts, selected_dates = selected_counts(selected)
    result = dict(manifest)
    result["createdAt"] = iso_now()
    result["label"] = f"Exact archive-resume reconstruction for {source_plan['id']}"
    result["archiveResumeSourcePlan"] = {
        "id": source_plan["id"],
        "label": source_plan.get("label"),
        "windows": source_plan["windows"],
        **({"randomTrainingChunks": source_plan["randomTrainingChunks"]}
           if "randomTrainingChunks" in source_plan else {}),
    }
    result["shards"] = selected
    result["counts"] = counts
    return result, selected_dates, counts


def selected_counts(shards: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    counts = {"train": 0, "validation": 0, "test": 0}
    selected_segments: dict[str, list[tuple[dict[str, Any], int, int]]] = {
        "train": [], "validation": [], "test": [],
    }
    previous_end: int | None = None
    previous_split: str | None = None
    for shard in shards:
        start = int(shard["predictionTimeStart"])
        count = int(shard["count"])
        end = start + (count - 1) * SECOND_MS
        local_offset = 0
        if previous_end is not None:
            if start <= previous_end:
                raise ValueError("reconstructed source shards overlap")
            if shard["split"] != previous_split:
                earliest = previous_end + INPUT_PURGE_MS + SECOND_MS
                if start < earliest:
                    local_offset = math.ceil((earliest - start) / SECOND_MS)
        if local_offset < count:
            selected_segments[shard["split"]].append(
                (shard, local_offset, count - local_offset)
            )
        previous_end = end
        previous_split = shard["split"]
    remaining = TEST_EXAMPLES
    test_tail: list[tuple[dict[str, Any], int, int]] = []
    for shard, offset, count in reversed(selected_segments["test"]):
        take = min(count, remaining)
        if take:
            test_tail.append((shard, offset + count - take, take))
            remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError("reconstructed test split is too small")
    selected_segments["test"] = list(reversed(test_tail))
    dates: set[str] = set()
    for split, segments in selected_segments.items():
        counts[split] = sum(count for _, _, count in segments)
        dates.update(str(shard["date"]) for shard, _, _ in segments)
    return counts, sorted(dates)


def window_ranges(source_plan: dict[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    excluded = set(source_plan["excludedAggregateWindows"])
    train: list[tuple[int, int]] = []
    validation: list[tuple[int, int]] = []
    for window in source_plan["windows"]:
        if window["id"] in excluded:
            continue
        start = parse_day(window["start"])
        end = parse_day(window["end"]) + DAY_MS
        midpoint = start + ((end - start) // SECOND_MS // 2) * SECOND_MS
        train.append((start, midpoint))
        validation.append((midpoint, end))
    for window in source_plan.get("randomTrainingChunks", {}).get("windows", []):
        train.append((parse_day(window["start"]), parse_day(window["end"]) + DAY_MS))
    return train, validation


def patch_trainer(source: str) -> str:
    source = replace_once(
        source,
        "import zstandard\n\nfrom return_oracle_ce import (",
        "import zstandard\n\nfrom trading_storage import (\n"
        "    load_torch_checkpoint,\n"
        "    resolve_shard,\n"
        "    save_torch_checkpoint,\n"
        ")\n\nfrom return_oracle_ce import (",
    )
    source = replace_once(
        source,
        '    ".completed-minute-simple-returns-60.compact.f16.zst"\n',
        '    ".completed-minute-simple-returns-60.compact.json"\n',
    )
    source = replace_once(
        source,
        '    repo_root = Path(__file__).resolve().parent.parent\n',
        '    repo_root = Path(os.environ["RETURN_ORACLE_REPO_ROOT"]).resolve()\n',
    )
    source = replace_once(
        source,
        '        self.log_file = run_dir / "training.log"\n'
        '        self.status_file = run_dir / "status.json"\n',
        '        self.log_file = run_dir / "logs" / "training.log"\n'
        '        self.status_file = run_dir / "state" / "status.json"\n'
        '        self.log_file.parent.mkdir(parents=True, exist_ok=True)\n'
        '        self.status_file.parent.mkdir(parents=True, exist_ok=True)\n',
    )
    helper = (
        "def archive_resume_payload(file: Path) -> Path:\n"
        "    if file.suffix.lower() != \".json\":\n"
        "        return file\n"
        "    shard = resolve_shard(file)\n"
        "    return (\n"
        "        shard.storage_root / shard.reference[\"object\"][\"file\"]\n"
        "    ).resolve()\n\n\n"
    )
    source = replace_once(source, "class ComponentCache:\n", helper + "class ComponentCache:\n")
    source = replace_once(
        source,
        "    ) -> np.ndarray:\n        key = (file, dtype, shape)\n",
        "    ) -> np.ndarray:\n        file = archive_resume_payload(file)\n"
        "        key = (file, dtype, shape)\n",
    )
    source = replace_once(
        source,
        "def valid_zstd_component(file: Path, expected_bytes: int) -> bool:\n"
        "    if not file.is_file() or file.stat().st_size < 1:\n",
        "def valid_zstd_component(file: Path, expected_bytes: int) -> bool:\n"
        "    if not file.is_file():\n"
        "        return False\n"
        "    file = archive_resume_payload(file)\n"
        "    if file.stat().st_size < 1:\n",
    )
    source = replace_once(
        source,
        '    last_checkpoint = reporter.run_dir / "last.pt"\n'
        '    best_checkpoint = reporter.run_dir / "best.pt"\n',
        '    last_checkpoint = reporter.run_dir / "checkpoints" / "last.json"\n'
        '    best_checkpoint = reporter.run_dir / "checkpoints" / "best.json"\n',
    )
    source = source.replace("checkpoint = torch.load(\n", "checkpoint = load_torch_checkpoint(\n")
    source = source.replace("best = torch.load(\n", "best = load_torch_checkpoint(\n")
    atomic_start = source.index(
        "def atomic_torch_save(value: dict, file: Path) -> None:\n"
    )
    atomic_end = source.index("def atomic_json", atomic_start)
    source = (
        source[:atomic_start]
        + "def atomic_torch_save(value: dict, file: Path) -> None:\n"
        + "    save_torch_checkpoint(value, file)\n\n\n"
        + source[atomic_end:]
    )
    source = replace_once(
        source,
        '    if not stopped_for_patience:\n'
        '        raise RuntimeError(\n'
        '            "maximum epochs were exhausted before validation staleness exceeded 1024"\n'
        '        )\n',
        '    if not stopped_for_patience:\n'
        '        reporter.status(\n'
        '            "archived",\n'
        '            startedAt=started_at,\n'
        '            archivedAt=iso_now(),\n'
        '            planId=plan["id"],\n'
        '            bestValidation=best_validation,\n'
        '            bestEpoch=best_epoch,\n'
        '            staleEpochs=stale_epochs,\n'
        '            message="Archive continuation reached the requested 200-epoch ceiling.",\n'
        '        )\n'
        '        return\n',
    )
    return source


def historical_dataset_summary(log_file: Path) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    with log_file.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "dataset-complete":
                result = {
                    "counts": {
                        "train": int(event["trainExamples"]),
                        "validation": int(event["validationExamples"]),
                        "test": int(event["testExamples"]),
                    },
                    "featureComponents": int(event["featureComponents"]),
                }
    if result is None:
        raise ValueError(f"dataset summary is absent: {log_file}")
    return result


def copy_source_references(
    current_source: Path,
    destination: Path,
    shards: list[dict[str, Any]],
) -> None:
    references = {
        str(shard[field])
        for shard in shards
        for field in ("features", "minuteOracleProbabilities")
    }
    for relative in references:
        source = current_source / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_feature_references(
    repo_root: Path,
    destination: Path,
    dates: list[str],
) -> None:
    source_root = (
        repo_root / "data" / "training" / "datasets"
        / "return-oracle-ce-shrinking-v1" / "components" / "returns"
    )
    target_root = destination / "components" / "returns"
    target_root.mkdir(parents=True, exist_ok=True)
    for date in dates:
        name = f"{date}.completed-minute-simple-returns-60.compact.json"
        shutil.copy2(source_root / name, target_root / name)


def copy_feature_statistics(
    repo_root: Path,
    destination: Path,
    examples: int,
    trainer_source: str,
) -> None:
    match = re.search(r'training-feature-statistics-position-v\d+\.npz', trainer_source)
    if match is None:
        raise ValueError("historical feature-statistics filename is absent")
    source_root = (
        repo_root / "data" / "training" / "datasets"
        / "return-oracle-ce-shrinking-v1"
    )
    candidates = []
    for candidate in source_root.glob("training-feature-statistics-position-*.npz"):
        with np.load(candidate) as values:
            if int(values["count"]) == examples:
                candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one frozen feature-statistics file for {examples}, got {candidates}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], destination / match.group(0))


def checkpoint_reference(pointer: Path) -> dict[str, Any]:
    value = json.loads(pointer.read_text(encoding="utf-8"))
    if value.get("kind") != "trading-immutable-artifact":
        raise ValueError(f"invalid checkpoint reference: {pointer}")
    return value


def checkpoint_metadata(
    repo_root: Path,
    run_dir: Path,
    reference: dict[str, Any],
) -> int:
    code = (
        "import torch;"
        f"c=torch.load(r'{str(repo_root / 'data' / 'training' / 'immutable' / reference['object']['file'])}',"
        "map_location='cpu',weights_only=False,mmap=True);"
        "print(int(c['epoch']))"
    )
    completed = subprocess.run(
        [str(repo_root / ".venv-ml" / "Scripts" / "python.exe"), "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(completed.stdout.strip())


def git_text(repo_root: Path, tree: str, file: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{tree}:{file}"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    return completed.stdout.decode("utf-8")


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"historical source patch expected one match: {old[:80]!r}")
    return source.replace(old, new, 1)


def parse_day(value: str) -> int:
    parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000)


def write_json(file: Path, value: Any) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
