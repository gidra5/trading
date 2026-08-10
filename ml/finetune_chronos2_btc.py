from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from chronos import Chronos2Pipeline
from forecast_model_zoo import MODEL_SPECS, snapshot_dir
from trading_storage import candle_times, read_candle_column


COLUMNS = ("open", "high", "low", "close", "volume")
STEP_MS = 60_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free LoRA fine-tuning of Chronos-2 on pre-test BTCUSDT candles."
    )
    parser.add_argument("--train-start", type=date.fromisoformat, default=date(2021, 7, 1))
    parser.add_argument("--train-end", type=date.fromisoformat, default=date(2021, 7, 31))
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2021, 8, 1))
    parser.add_argument("--validation-end", type=date.fromisoformat, default=date(2021, 8, 31))
    parser.add_argument("--test-boundary", type=date.fromisoformat, default=date(2021, 9, 8))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--prediction-length", type=int, default=15)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/forecast-foundation/chronos2-btc-pretest-lora"),
    )
    return parser.parse_args()


def days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_daily_series(root: Path, start: date, end: date) -> tuple[list[np.ndarray], list[Path], str]:
    references = [root / f"{day.isoformat()}.json" for day in days(start, end)]
    missing = [path for path in references if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} canonical shards; first is {missing[0]}")
    series: list[np.ndarray] = []
    fingerprint = hashlib.sha256()
    for reference in references:
        fingerprint.update(reference.name.encode("utf-8"))
        fingerprint.update(reference.read_bytes())
        times = candle_times(reference).astype(np.int64, copy=False)
        values = np.stack(
            [read_candle_column(reference, name).astype(np.float32, copy=False) for name in COLUMNS]
        )
        expected_start = int(datetime.combine(
            date.fromisoformat(reference.stem), datetime.min.time(), tzinfo=timezone.utc
        ).timestamp() * 1_000)
        expected_times = expected_start + np.arange(values.shape[1], dtype=np.int64) * STEP_MS
        if values.shape != (len(COLUMNS), 1_440):
            raise ValueError(f"expected a complete 5x1440 shard: {reference} has {values.shape}")
        if not np.array_equal(times, expected_times) or not np.isfinite(values).all():
            raise ValueError(f"invalid or non-contiguous canonical shard: {reference}")
        series.append(values)
    return series, references, fingerprint.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not (args.train_start <= args.train_end < args.validation_start <= args.validation_end):
        raise ValueError("training and validation ranges must be ordered and non-overlapping")
    if args.validation_end >= args.test_boundary:
        raise ValueError("validation must end strictly before the first test boundary")
    if args.steps < 1 or args.batch_size < len(COLUMNS):
        raise ValueError("steps must be positive and batch size must fit one five-variate task")

    repo_root = Path(__file__).resolve().parents[1]
    history_root = (repo_root / args.history_dir).resolve()
    output_root = (repo_root / args.output_dir).resolve()
    checkpoint_name = (
        f"steps-{args.steps}-lr-{args.learning_rate:g}-ctx-{args.context_length}-seed-{args.seed}"
    )
    run_dir = output_root / checkpoint_name
    manifest_path = run_dir / "training-manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"SKIP complete checkpoint {run_dir}", flush=True)
            return

    train_inputs, train_references, train_fingerprint = load_daily_series(
        history_root, args.train_start, args.train_end
    )
    validation_inputs, validation_references, validation_fingerprint = load_daily_series(
        history_root, args.validation_start, args.validation_end
    )
    spec = next(item for item in MODEL_SPECS if item.id == "chronos2")
    base_model_path = snapshot_dir(repo_root, spec)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "contract": "chronos2-btc-pretest-lora-v1",
        "status": "running",
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": "chronos2",
            "baseSnapshot": str(base_model_path).replace("\\", "/"),
            "sourceCommit": spec.source_commit,
            "hfRevision": spec.model_revision,
        },
        "data": {
            "market": "Binance spot BTCUSDT",
            "interval": "1m",
            "columns": list(COLUMNS),
            "representation": "raw",
            "train": {
                "start": args.train_start.isoformat(),
                "endInclusive": args.train_end.isoformat(),
                "days": len(train_inputs),
                "candles": sum(item.shape[-1] for item in train_inputs),
                "fingerprint": train_fingerprint,
                "references": [str(path.relative_to(repo_root)).replace("\\", "/") for path in train_references],
            },
            "validation": {
                "start": args.validation_start.isoformat(),
                "endInclusive": args.validation_end.isoformat(),
                "days": len(validation_inputs),
                "candles": sum(item.shape[-1] for item in validation_inputs),
                "fingerprint": validation_fingerprint,
                "references": [
                    str(path.relative_to(repo_root)).replace("\\", "/") for path in validation_references
                ],
            },
            "firstTestBoundary": args.test_boundary.isoformat(),
            "leakagePolicy": "both training and validation end strictly before the earliest inspector test window",
        },
        "training": {
            "mode": "lora",
            "contextLength": args.context_length,
            "predictionLength": args.prediction_length,
            "steps": args.steps,
            "learningRate": args.learning_rate,
            "batchSize": args.batch_size,
            "seed": args.seed,
            "device": args.device,
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        f"TRAIN Chronos-2 LoRA train={args.train_start}..{args.train_end} "
        f"validation={args.validation_start}..{args.validation_end} steps={args.steps}",
        flush=True,
    )

    pipeline = Chronos2Pipeline.from_pretrained(base_model_path, device_map=args.device)
    finetuned = pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        validation_inputs=validation_inputs,
        finetune_mode="lora",
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.steps,
        batch_size=args.batch_size,
        min_past=args.context_length,
        output_dir=run_dir / "trainer",
        finetuned_ckpt_name="adapter-checkpoint",
        logging_steps=min(25, args.steps),
        eval_steps=min(100, args.steps),
        save_steps=min(100, args.steps),
    )

    # Chronos returns a PEFT-backed pipeline for LoRA. Merge the learned low-rank
    # update into the base weights so the production adapter can load one normal,
    # self-contained Chronos-2 checkpoint without PEFT runtime state.
    merged_model = finetuned.model
    if hasattr(merged_model, "merge_and_unload"):
        merged_model = merged_model.merge_and_unload()
    merged_pipeline = Chronos2Pipeline(model=merged_model)
    merged_path = run_dir / "merged-checkpoint"
    merged_pipeline.save_pretrained(merged_path)

    manifest["status"] = "complete"
    manifest["completedAt"] = datetime.now(timezone.utc).isoformat()
    manifest["artifacts"] = {
        "adapterCheckpoint": str((run_dir / "trainer" / "adapter-checkpoint").relative_to(repo_root)).replace(
            "\\", "/"
        ),
        "mergedCheckpoint": str(merged_path.relative_to(repo_root)).replace("\\", "/"),
    }
    atomic_json(manifest_path, manifest)
    print(f"WROTE {merged_path}", flush=True)


if __name__ == "__main__":
    main()
