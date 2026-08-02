"""Evaluate preserved v18 on the exact validation target rows kept by v28.

This is an inference-only, leakage-safe comparison.  Target split boundaries
and purge offsets are reconstructed from filenames, so sealed test reference
JSON and its payload objects are never opened.  v18 retains its own 60-minute
causal input and receptive-field contract; only its scored validation row set
is restricted to the ordered 42,780-row subset used by v28.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

from audit_v18_calendar_fusion import (
    validate_preserved_checkpoint_without_test_access,
)
from evaluate_joint_price_oracle_actions import (
    raw_distribution_metrics,
    resolve_device,
)
from trading_storage import (
    load_torch_checkpoint,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    CausalOracleDataset,
    CausalSegment,
    PrefetchedBatchIterator,
    autocast_context,
    build_model,
    move_batch,
    purge_cross_split_windows,
    resolve,
    resolve_training_config,
    unpack_sequence_core_batch,
    validate_plan,
)


DEFAULT_V18_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
)
DEFAULT_V28_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-long-tcn-v28-reuse.json"
)
SECOND_MS = 1_000
MINUTE_MS = 60_000
DAY_ROWS = 1_440
V28_REPORTED_BEST_KL = 0.9672289951319113


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v18-plan", type=Path, default=DEFAULT_V18_PLAN)
    parser.add_argument("--v28-plan", type=Path, default=DEFAULT_V28_PLAN)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = audit_v18_on_v28_validation_rows(
        arguments.v18_plan,
        arguments.v28_plan,
        requested_device=arguments.device,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


def audit_v18_on_v28_validation_rows(
    v18_plan_file: Path,
    v28_plan_file: Path,
    *,
    requested_device: str = "auto",
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    storage = training_storage_layout(repo_root)
    v18_plan = load_plan(repo_root, v18_plan_file)
    v28_plan = load_plan(repo_root, v28_plan_file)
    validate_comparable_plans(v18_plan, v28_plan)

    target_root = require_under(
        resolve(repo_root, Path(v18_plan["targetReferenceDir"])),
        storage.immutable / "refs" / "oracle",
        "targetReferenceDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(v18_plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    target_files = sorted(target_root.glob("*.json"))
    v18_segments = filename_only_segments(target_files, v18_plan)
    v28_segments = filename_only_segments(target_files, v28_plan)
    v18_counts = split_counts(v18_segments)
    v28_counts = split_counts(v28_segments)

    v18_run = require_under(
        resolve(repo_root, Path(v18_plan["runDir"])),
        storage.runs,
        "v18 runDir",
    )
    v28_run = require_under(
        resolve(repo_root, Path(v28_plan["runDir"])),
        storage.runs,
        "v28 runDir",
    )
    v18_event = read_dataset_event(v18_run)
    v28_event = read_dataset_event(v28_run)
    validate_dataset_event(v18_event, v18_plan, v18_counts)
    validate_dataset_event(v28_event, v28_plan, v28_counts)

    v18_checkpoint_file = v18_run / "checkpoints" / "best.json"
    v28_checkpoint_file = v28_run / "checkpoints" / "best.json"
    v18_checkpoint = load_torch_checkpoint(
        v18_checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    v28_checkpoint = load_torch_checkpoint(
        v28_checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    v18_training = resolve_training_config(v18_plan["training"])
    validate_preserved_checkpoint_without_test_access(
        v18_checkpoint,
        v18_plan,
        v18_plan["model"],
        v18_training,
        v18_counts,
        v18_run,
    )
    validate_checkpoint_dataset_identity(
        v18_checkpoint,
        v18_event,
        v18_plan,
    )
    validate_checkpoint_dataset_identity(
        v28_checkpoint,
        v28_event,
        v28_plan,
    )
    del v28_checkpoint

    v18_keys = ordered_row_keys(v18_segments["validation"])
    v28_keys = ordered_row_keys(v28_segments["validation"])
    matched_indexes = ordered_subsequence_indexes(v18_keys, v28_keys)
    if matched_indexes.shape != (v28_counts["validation"],):
        raise RuntimeError("matched validation index count is inconsistent")

    device = resolve_device(
        requested_device,
        str(v18_plan["training"]["device"]),
    )
    dataset = CausalOracleDataset(
        history_root,
        v18_segments,
        int(v18_plan["model"]["contextLength"]),
        int(v18_plan["model"]["forecastHorizon"]),
        target_rows_per_file=DAY_ROWS,
        action_count=int(v18_plan["model"]["actionCount"]),
        close_cache_days=int(v18_plan["training"].get("closeCacheDays", 10)),
        target_cache_days=int(
            v18_plan["training"].get("targetCacheDays", 3)
        ),
        pin_memory=device.type == "cuda",
        include_future_closes=False,
    )
    opened_target_references: set[Path] = set()
    opened_history_references: set[Path] = set()
    original_target_load = dataset.target_cache.load
    original_history_load = dataset.close_cache.load_day

    def tracked_target_load(file: Path) -> torch.Tensor:
        opened_target_references.add(file.resolve())
        return original_target_load(file)

    def tracked_history_load(date_value: str) -> np.ndarray:
        opened_history_references.add(
            (history_root / f"{date_value}.json").resolve()
        )
        return original_history_load(date_value)

    dataset.target_cache.load = tracked_target_load
    dataset.close_cache.load_day = tracked_history_load

    model = build_model(v18_plan["model"]).to(device)
    model.load_state_dict(v18_checkpoint["model"])
    del v18_checkpoint
    core_rows = int(
        v18_plan["training"]["sequenceCoreTraining"]["coreRows"]
    )
    print(
        "Running v18 best checkpoint once on validation only; sealed test "
        "references and payloads remain unopened.",
        file=sys.stderr,
        flush=True,
    )
    logits, targets = collect_sequence_core_rows(
        model,
        dataset,
        "validation",
        core_rows,
        device,
        v18_training,
    )
    del model, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    expected_shape = (
        v18_counts["validation"],
        int(v18_plan["model"]["actionCount"]),
    )
    if logits.shape != expected_shape or targets.shape != expected_shape:
        raise RuntimeError("v18 validation inference row count is inconsistent")
    full_metrics = raw_distribution_metrics(logits, targets)
    matched_metrics = raw_distribution_metrics(
        logits[matched_indexes],
        targets[matched_indexes],
    )
    reported_v18_kl = float(
        v18_checkpoint_reported_kl(v18_run)
    )
    if abs(float(full_metrics["klDivergence"]) - reported_v18_kl) > 1e-6:
        raise RuntimeError("v18 full validation KL was not reproduced")

    validation_files = {
        segment.target_file.resolve()
        for segment in v18_segments["validation"]
    }
    train_files = {
        segment.target_file.resolve()
        for segment in v18_segments["train"]
    }
    test_files = {
        segment.target_file.resolve()
        for segment in v18_segments["test"]
    }
    if opened_target_references != validation_files:
        raise RuntimeError("unexpected target-reference access set")
    if opened_target_references & (train_files | test_files):
        raise RuntimeError("non-validation target-reference access detected")

    matched_kl = float(matched_metrics["klDivergence"])
    return {
        "schemaVersion": 1,
        "audit": "preserved-v18-on-exact-v28-validation-target-rows",
        "accessContract": {
            "validationTargetReferenceFilesOpened": len(
                opened_target_references
            ),
            "trainTargetReferenceFilesOpened": 0,
            "testReferenceFilesOpened": 0,
            "testPayloadsOpened": 0,
            "historyReferenceFilesOpened": len(opened_history_references),
            "modelWeightsChanged": False,
            "trainingStarted": False,
            "inferenceDevice": str(device),
        },
        "checkpoint": {
            "planId": v18_plan["id"],
            "kind": "best",
            "epoch": int(read_best_checkpoint_epoch(v18_run)),
            "file": str(v18_checkpoint_file),
            "reportedFullValidationKl": reported_v18_kl,
        },
        "rowAlignment": {
            "v18ValidationRows": v18_counts["validation"],
            "v28ValidationRows": v28_counts["validation"],
            "excludedV18PrefixRows": int(matched_indexes[0]),
            "matchedRows": int(matched_indexes.size),
            "matchedIndexesAreContiguousSuffix": bool(np.array_equal(
                matched_indexes,
                np.arange(
                    matched_indexes[0],
                    len(v18_keys),
                    dtype=np.int64,
                ),
            )),
            "firstMatchedTimestampMs": int(v28_keys[0][0]),
            "lastMatchedTimestampMs": int(v28_keys[-1][0]),
            "firstMatchedTargetDate": v28_keys[0][1],
            "firstMatchedTargetRowOffset": int(v28_keys[0][2]),
            "lastMatchedTargetDate": v28_keys[-1][1],
            "lastMatchedTargetRowOffset": int(v28_keys[-1][2]),
            "v18DatasetFingerprint": v18_event["datasetFingerprint"],
            "v28DatasetFingerprint": v28_event["datasetFingerprint"],
            "durableV28CountVerified": True,
            "targetOrderingVerifiedByExactRowKeys": True,
        },
        "raw01Kl": {
            "v18FullOriginalRows": full_metrics,
            "v18ExactV28Rows": matched_metrics,
            "v28ExactV28Rows": {
                "klDivergence": V28_REPORTED_BEST_KL,
            },
            "v28MinusMatchedV18": V28_REPORTED_BEST_KL - matched_kl,
            "v28ImprovesMatchedV18": V28_REPORTED_BEST_KL < matched_kl,
        },
    }


def load_plan(repo_root: Path, plan_file: Path) -> dict:
    resolved = resolve(repo_root, plan_file).resolve()
    plan = json.loads(resolved.read_text(encoding="utf-8"))
    validate_plan(plan)
    if plan.get("testPolicy", "sealed-never-load") != "sealed-never-load":
        raise ValueError("audit requires sealed-never-load test policy")
    return plan


def validate_comparable_plans(v18_plan: dict, v28_plan: dict) -> None:
    keys = (
        "targetReferenceDir",
        "historyDir",
        "samplingIntervalMs",
        "predictionDelayMs",
        "oracleTarget",
        "dataSplit",
    )
    if any(v18_plan.get(key) != v28_plan.get(key) for key in keys):
        raise ValueError("v18 and v28 data/target contracts differ")
    if int(v18_plan["model"]["receptiveFieldMinutes"]) != 60:
        raise ValueError("baseline plan is not the verified 60-minute v18")
    if int(v28_plan["model"]["receptiveFieldMinutes"]) != 360:
        raise ValueError("comparison plan is not the six-hour v28")


def filename_only_segments(
    target_files: list[Path],
    plan: dict,
) -> dict[str, list[CausalSegment]]:
    validation_days = int(plan["dataSplit"]["validationDays"])
    test_days = int(plan["dataSplit"]["testDays"])
    if len(target_files) <= validation_days + test_days:
        raise ValueError("target filename corpus is too short")
    train_end = len(target_files) - validation_days - test_days
    validation_end = len(target_files) - test_days
    raw = {split: [] for split in ("train", "validation", "test")}
    for index, target_file in enumerate(target_files):
        split = (
            "train" if index < train_end
            else "validation" if index < validation_end
            else "test"
        )
        day_start = int(datetime.combine(
            date.fromisoformat(target_file.stem),
            datetime.min.time(),
            timezone.utc,
        ).timestamp() * SECOND_MS)
        raw[split].append(CausalSegment(
            split=split,
            prediction_time_start=day_start + SECOND_MS - 1,
            count=DAY_ROWS,
            target_file=target_file,
            target_row_offset=0,
            step_ms=MINUTE_MS,
        ))
    return purge_cross_split_windows(
        raw,
        int(plan["model"]["contextLength"]),
        int(plan["model"]["forecastHorizon"]),
    )


def split_counts(
    segments: dict[str, list[CausalSegment]],
) -> dict[str, int]:
    return {
        split: sum(segment.count for segment in segments[split])
        for split in ("train", "validation", "test")
    }


def read_dataset_event(run_dir: Path) -> dict:
    first_line = (
        run_dir / "logs" / "training.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]
    event = json.loads(first_line)
    if event.get("event") != "dataset":
        raise ValueError("run log does not start with a dataset event")
    return event


def validate_dataset_event(event: dict, plan: dict, counts: dict) -> None:
    if event.get("counts") != counts \
            or int(event.get("contextLength", -1)) \
            != int(plan["model"]["contextLength"]) \
            or int(event.get("forecastHorizon", -1)) \
            != int(plan["model"]["forecastHorizon"]):
        raise ValueError("durable dataset event differs from filename purge")


def validate_checkpoint_dataset_identity(
    checkpoint: dict,
    dataset_event: dict,
    plan: dict,
) -> None:
    if checkpoint.get("planId") != plan["id"] \
            or checkpoint.get("datasetFingerprint") \
            != dataset_event.get("datasetFingerprint"):
        raise ValueError("checkpoint/durable dataset identity mismatch")


def ordered_row_keys(
    segments: list[CausalSegment],
) -> list[tuple[int, str, int]]:
    result: list[tuple[int, str, int]] = []
    for segment in segments:
        for local_offset in range(segment.count):
            result.append((
                segment.prediction_time_start + local_offset * segment.step_ms,
                segment.target_file.stem,
                segment.target_row_offset + local_offset,
            ))
    if any(
        following[0] <= previous[0]
        for previous, following in zip(result, result[1:])
    ):
        raise RuntimeError("validation row keys are not chronological")
    return result


def ordered_subsequence_indexes(
    source: list[tuple[int, str, int]],
    requested: list[tuple[int, str, int]],
) -> np.ndarray:
    if not source or not requested:
        raise ValueError("validation row keys cannot be empty")
    indexes: list[int] = []
    cursor = 0
    for key in requested:
        while cursor < len(source) and source[cursor] != key:
            cursor += 1
        if cursor == len(source):
            raise ValueError("v28 validation rows are not a v18 subsequence")
        indexes.append(cursor)
        cursor += 1
    return np.asarray(indexes, dtype=np.int64)


@torch.no_grad()
def collect_sequence_core_rows(
    model,
    dataset: CausalOracleDataset,
    split: str,
    core_rows: int,
    device: torch.device,
    training: dict,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    source = dataset.iter_sequence_core_batches(
        split,
        core_rows,
        receptive_field_minutes=int(model.receptive_field_minutes),
        shuffle=False,
        seed=int(training["seed"]),
        maximum_batches=None,
        pack_contiguous_runs=False,
    )
    batches = dataset.sequence_core_batch_count(split, core_rows)
    progress_every = max(1, batches // 10)
    with PrefetchedBatchIterator(
        source,
        int(training.get("prefetchBatches", 2)),
    ) as prefetched:
        for index, batch in enumerate(prefetched, start=1):
            moved = move_batch(batch, device)
            input_closes, target_policy, target_mask = (
                unpack_sequence_core_batch(moved)
            )
            if target_mask is not None:
                raise RuntimeError("v18 audit expects unpadded single cores")
            with autocast_context(device, training):
                predicted = model.forward_sequence_core(input_closes)
            if predicted.shape != target_policy.shape:
                raise RuntimeError("sequence-core inference rows are misaligned")
            logits.append(predicted[0].float().cpu().numpy())
            targets.append(target_policy[0].float().cpu().numpy())
            if index == batches or index % progress_every == 0:
                print(
                    f"MATCHED-AUDIT {split} batches {index}/{batches}",
                    file=sys.stderr,
                    flush=True,
                )
    if not logits:
        raise RuntimeError("validation produced no sequence-core rows")
    return np.concatenate(logits), np.concatenate(targets)


def read_best_checkpoint_epoch(run_dir: Path) -> int:
    checkpoint = load_torch_checkpoint(
        run_dir / "checkpoints" / "best.json",
        map_location="cpu",
        weights_only=False,
    )
    return int(checkpoint["epoch"])


def v18_checkpoint_reported_kl(run_dir: Path) -> float:
    checkpoint = load_torch_checkpoint(
        run_dir / "checkpoints" / "best.json",
        map_location="cpu",
        weights_only=False,
    )
    return float(checkpoint["validation"]["klDivergence"])


if __name__ == "__main__":
    main()
