from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from kronos_model_zoo import selected_specs, snapshot_dir, source_root
from trading_storage import candle_times, read_candle_column


STEP_MS = 60_000
FEATURE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SeriesRange:
    timestamps: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class PredictorLosses:
    objective: torch.Tensor
    full: torch.Tensor
    forecast: torch.Tensor


@dataclass(frozen=True)
class ExclusionPlan:
    file: Path
    fingerprint: str
    ranges: tuple[tuple[int, int], ...]
    source: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a public Kronos predictor on canonical BTCUSDT 1m data."
    )
    parser.add_argument("--model", choices=("mini", "small", "base"), default="small")
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=Path,
        help="Optional locally fine-tuned tokenizer used for predictor training.",
    )
    parser.add_argument("--train-start", default="2021-07-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-07-01")
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples-per-epoch", type=int, default=20_000)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--forecast-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Additional weight for the exact 15 future-token positions. "
            "Zero reproduces the upstream all-position objective."
        ),
    )
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--exclusion-plan",
        type=Path,
        help=(
            "Optional JSON plan of time ranges that no training or validation "
            "window may touch. Use this to reserve policy/backtest episodes."
        ),
    )
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tools/Kronos-finetuned/btcusdt-1m-small-predictor"),
    )
    return parser.parse_args()


def utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000)


def load_exclusion_plan(file: Path) -> ExclusionPlan:
    encoded = file.read_bytes()
    source = json.loads(encoded.decode("utf-8"))
    if not isinstance(source, dict) \
            or source.get("version") != 1 \
            or source.get("contract") \
            != "kronos-btcusdt-1m-finetune-exclusion-plan-v1" \
            or not isinstance(source.get("ranges"), list) \
            or not source["ranges"]:
        raise ValueError("invalid Kronos fine-tune exclusion plan")
    ranges: list[tuple[int, int]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(source["ranges"]):
        if not isinstance(item, dict) \
                or not isinstance(item.get("id"), str) \
                or not isinstance(item.get("start"), str) \
                or not isinstance(item.get("end"), str):
            raise ValueError(f"invalid exclusion range at index {index}")
        if item["id"] in identifiers:
            raise ValueError(f"duplicate exclusion range id {item['id']}")
        identifiers.add(item["id"])
        start = utc_ms(item["start"])
        end = utc_ms(item["end"])
        if start >= end:
            raise ValueError(f"invalid exclusion range {item['id']}")
        if ranges and start < ranges[-1][1]:
            raise ValueError("exclusion ranges must be sorted and non-overlapping")
        ranges.append((start, end))
    return ExclusionPlan(
        file=file,
        fingerprint=hashlib.sha256(encoded).hexdigest(),
        ranges=tuple(ranges),
        source=source,
    )


def date_strings(start_ms: int, end_ms: int) -> list[str]:
    current = datetime.fromtimestamp(start_ms / 1_000, tz=timezone.utc).date()
    final = datetime.fromtimestamp((end_ms - 1) / 1_000, tz=timezone.utc).date()
    output = []
    while current <= final:
        output.append(current.isoformat())
        current += timedelta(days=1)
    return output


def load_series(root: Path, start_ms: int, end_ms: int) -> SeriesRange:
    time_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    days = date_strings(start_ms, end_ms)
    for index, day in enumerate(days, start=1):
        file = root / f"{day}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing canonical 1m history: {file}")
        timestamps = candle_times(file).astype(np.int64, copy=False)
        columns = [
            read_candle_column(file, name).astype(np.float32, copy=False)
            for name in FEATURE_COLUMNS
        ]
        values = np.column_stack(columns)
        selected = (timestamps >= start_ms) & (timestamps < end_ms)
        if selected.any():
            time_parts.append(timestamps[selected])
            value_parts.append(values[selected])
        if index % 100 == 0 or index == len(days):
            print(f"DATA loaded {index}/{len(days)} days", flush=True)
    timestamps = np.concatenate(time_parts)
    raw = np.concatenate(value_parts)
    if np.any(np.diff(timestamps) != STEP_MS):
        raise ValueError("fine-tuning range is not a contiguous 1-minute series")
    amount = raw[:, 4:5] * raw[:, :4].mean(axis=1, keepdims=True)
    values = np.concatenate((raw, amount), axis=1).astype(np.float32)
    if not np.isfinite(values).all() or np.any(values[:, :4] <= 0):
        raise ValueError("fine-tuning range contains invalid values")
    return SeriesRange(timestamps=timestamps, values=values)


def timestamp_features(timestamps: np.ndarray) -> np.ndarray:
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="ms", utc=True))
    return np.column_stack((
        index.minute,
        index.hour,
        index.weekday,
        index.day,
        index.month,
    )).astype(np.float32)


class KronosWindowDataset(Dataset):
    def __init__(
        self,
        series: SeriesRange,
        *,
        window: int,
        lookback: int,
        samples: int,
        seed: int,
        training: bool,
        excluded_ranges: tuple[tuple[int, int], ...] = (),
    ) -> None:
        self.values = series.values
        self.stamps = timestamp_features(series.timestamps)
        self.window = window
        self.lookback = lookback
        self.samples = samples
        self.seed = seed
        self.training = training
        self.epoch = 0
        self.max_start = self.values.shape[0] - window
        if self.max_start < 0 or samples < 1:
            raise ValueError("fine-tuning range is too short for requested windows")
        candidates = np.arange(self.max_start + 1, dtype=np.int64)
        allowed = np.ones(candidates.shape[0], dtype=np.bool_)
        candidate_starts = series.timestamps[candidates]
        candidate_ends = series.timestamps[candidates + window - 1] + STEP_MS
        for start, end in excluded_ranges:
            allowed &= ~((candidate_starts < end) & (candidate_ends > start))
        self.start_indexes = candidates[allowed]
        self.excluded_candidate_starts = int(candidates.size - self.start_indexes.size)
        if self.start_indexes.size == 0:
            raise ValueError("fine-tuning exclusions removed every candidate window")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples

    def sample_start_index(self, index: int) -> int:
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        available = int(self.start_indexes.size)
        if self.training:
            selected = (
                index * 9_973
                + (self.epoch + 1) * 104_729
                + self.seed * 65_537
            ) % available
        elif self.samples == 1:
            selected = available // 2
        else:
            selected = round(index * (available - 1) / (self.samples - 1))
        return int(self.start_indexes[selected])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.sample_start_index(index)
        values = self.values[start:start + self.window].copy()
        # Match Kronos inference: derive normalization statistics strictly from
        # the historical context. Using the complete window leaks the target.
        means = values[:self.lookback].mean(axis=0)
        stds = values[:self.lookback].std(axis=0)
        values = np.clip((values - means) / (stds + 1e-5), -5, 5)
        stamps = self.stamps[start:start + self.window]
        return torch.from_numpy(values), torch.from_numpy(stamps)


def predictor_losses(
    head,
    logits,
    token_0: torch.Tensor,
    token_1: torch.Tensor,
    *,
    lookback: int,
    horizon: int,
    forecast_loss_weight: float,
) -> PredictorLosses:
    full, _, _ = head.compute_loss(
        logits[0], logits[1], token_0[:, 1:], token_1[:, 1:]
    )
    start = lookback - 1
    end = start + horizon
    if start < 0 or end > logits[0].shape[1]:
        raise ValueError("forecast loss slice exceeds the predictor sequence")
    forecast, _, _ = head.compute_loss(
        logits[0][:, start:end],
        logits[1][:, start:end],
        token_0[:, 1:][:, start:end],
        token_1[:, 1:][:, start:end],
    )
    objective = (
        full + float(forecast_loss_weight) * forecast
    ) / (1.0 + float(forecast_loss_weight))
    return PredictorLosses(objective=objective, full=full, forecast=forecast)


def validate(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    *,
    lookback: int,
    horizon: int,
    forecast_loss_weight: float,
) -> tuple[float, float, float]:
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    batches = 0
    with torch.no_grad():
        for values, stamps in loader:
            values = values.to(device, non_blocking=True)
            stamps = stamps.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                token_0, token_1 = tokenizer.encode(values, half=True)
                logits = model(
                    token_0[:, :-1],
                    token_1[:, :-1],
                    stamps[:, :-1],
                )
                losses = predictor_losses(
                    model.head,
                    logits,
                    token_0,
                    token_1,
                    lookback=lookback,
                    horizon=horizon,
                    forecast_loss_weight=forecast_loss_weight,
                )
            totals += (
                float(losses.objective.item()),
                float(losses.full.item()),
                float(losses.forecast.item()),
            )
            batches += 1
    return tuple((totals / batches).tolist())


def atomic_json(value: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    for name in (
        "lookback", "horizon", "epochs", "train_samples_per_epoch",
        "validation_samples", "batch_size", "gradient_accumulation",
        "log_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive and finite")
    if not math.isfinite(args.forecast_loss_weight) \
            or args.forecast_loss_weight < 0:
        raise ValueError("--forecast-loss-weight must be non-negative and finite")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    history_root = (
        args.history_dir if args.history_dir.is_absolute()
        else repo_root / args.history_dir
    ).resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    ).resolve()
    exclusion_plan = None
    if args.exclusion_plan is not None:
        exclusion_file = (
            args.exclusion_plan
            if args.exclusion_plan.is_absolute()
            else repo_root / args.exclusion_plan
        ).resolve()
        exclusion_plan = load_exclusion_plan(exclusion_file)
    tokenizer_checkpoint = None
    if args.tokenizer_checkpoint is not None:
        tokenizer_checkpoint = (
            args.tokenizer_checkpoint
            if args.tokenizer_checkpoint.is_absolute()
            else repo_root / args.tokenizer_checkpoint
        ).resolve()
        if not (tokenizer_checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"invalid tokenizer checkpoint: {tokenizer_checkpoint}"
            )
    spec = selected_specs(args.model)[0]
    if args.lookback > spec.max_context:
        raise ValueError("lookback exceeds the selected model context")
    train_start = utc_ms(args.train_start)
    train_end = utc_ms(args.train_end)
    validation_start = utc_ms(args.validation_start)
    validation_end = utc_ms(args.validation_end)
    if not train_start < train_end <= validation_start < validation_end:
        raise ValueError("train and validation ranges must be chronological")
    train_series = load_series(history_root, train_start, train_end)
    validation_series = load_series(
        history_root, validation_start, validation_end
    )
    window = args.lookback + args.horizon + 1
    train_dataset = KronosWindowDataset(
        train_series,
        window=window,
        lookback=args.lookback,
        samples=args.train_samples_per_epoch,
        seed=args.seed,
        training=True,
        excluded_ranges=(exclusion_plan.ranges if exclusion_plan else ()),
    )
    validation_dataset = KronosWindowDataset(
        validation_series,
        window=window,
        lookback=args.lookback,
        samples=args.validation_samples,
        seed=args.seed,
        training=False,
        excluded_ranges=(exclusion_plan.ranges if exclusion_plan else ()),
    )
    print(
        f"EXCLUSIONS plan={exclusion_plan.file if exclusion_plan else None} "
        f"trainCandidateStartsRemoved={train_dataset.excluded_candidate_starts} "
        f"validationCandidateStartsRemoved="
        f"{validation_dataset.excluded_candidate_starts}",
        flush=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    source = source_root(repo_root)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from model import Kronos, KronosTokenizer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = device.type == "cuda" and not args.no_amp
    tokenizer_path = tokenizer_checkpoint or snapshot_dir(
        repo_root, spec.tokenizer_repo
    )
    tokenizer = KronosTokenizer.from_pretrained(
        str(tokenizer_path)
    ).to(device).eval()
    tokenizer.requires_grad_(False)
    model = Kronos.from_pretrained(str(snapshot_dir(
        repo_root, spec.model_repo
    ))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation
    )
    total_steps = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.03,
        div_factor=10,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    baseline_validation, baseline_full, baseline_forecast = validate(
        model,
        tokenizer,
        validation_loader,
        device,
        amp,
        lookback=args.lookback,
        horizon=args.horizon,
        forecast_loss_weight=args.forecast_loss_weight,
    )
    print(
        f"BASELINE validationLoss={baseline_validation:.6f} "
        f"fullLoss={baseline_full:.6f} forecastLoss={baseline_forecast:.6f}",
        flush=True,
    )
    checkpoint = output_dir / "best_model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    print(f"CHECKPOINT epoch=0 {checkpoint}", flush=True)
    best_validation = baseline_validation
    best_validation_full = baseline_full
    best_validation_forecast = baseline_forecast
    best_epoch = 0
    history = [{
        "epoch": 0,
        "trainLoss": None,
        "validationLoss": baseline_validation,
        "validationFullLoss": baseline_full,
        "validationForecastLoss": baseline_forecast,
    }]
    started = time.monotonic()
    global_step = 0
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (values, stamps) in enumerate(train_loader):
            values = values.to(device, non_blocking=True)
            stamps = stamps.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                token_0, token_1 = tokenizer.encode(values, half=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(
                    token_0[:, :-1],
                    token_1[:, :-1],
                    stamps[:, :-1],
                )
                losses = predictor_losses(
                    model.head,
                    logits,
                    token_0,
                    token_1,
                    lookback=args.lookback,
                    horizon=args.horizon,
                    forecast_loss_weight=args.forecast_loss_weight,
                )
                loss = losses.objective
            scaler.scale(loss / args.gradient_accumulation).backward()
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            train_total += float(loss.item())
            train_batches += 1
            if should_update and global_step % args.log_interval == 0:
                elapsed = time.monotonic() - started
                print(
                    f"TRAIN epoch={epoch + 1}/{args.epochs} "
                    f"step={global_step}/{total_steps} loss={loss.item():.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"stepsPerSecond={global_step / elapsed:.2f}",
                    flush=True,
                )
        validation_loss, validation_full, validation_forecast = validate(
            model,
            tokenizer,
            validation_loader,
            device,
            amp,
            lookback=args.lookback,
            horizon=args.horizon,
            forecast_loss_weight=args.forecast_loss_weight,
        )
        train_loss = train_total / train_batches
        history.append({
            "epoch": epoch + 1,
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "validationFullLoss": validation_full,
            "validationForecastLoss": validation_forecast,
        })
        print(
            f"EPOCH {epoch + 1} trainLoss={train_loss:.6f} "
            f"validationLoss={validation_loss:.6f} "
            f"fullLoss={validation_full:.6f} "
            f"forecastLoss={validation_forecast:.6f}",
            flush=True,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_validation_full = validation_full
            best_validation_forecast = validation_forecast
            best_epoch = epoch + 1
            model.save_pretrained(checkpoint)
            print(f"CHECKPOINT epoch={best_epoch} {checkpoint}", flush=True)
        atomic_json({
            "contract": "kronos-btcusdt-1m-predictor-finetune-v2",
            "model": args.model,
            "modelRepo": spec.model_repo,
            "modelRevision": spec.model_revision,
            "tokenizerRepo": spec.tokenizer_repo,
            "tokenizerRevision": spec.tokenizer_revision,
            "tokenizerCheckpoint": (
                str(tokenizer_checkpoint).replace("\\", "/")
                if tokenizer_checkpoint else None
            ),
            "trainRange": [args.train_start, args.train_end],
            "validationRange": [args.validation_start, args.validation_end],
            "exclusionPlan": (
                str(exclusion_plan.file).replace("\\", "/")
                if exclusion_plan else None
            ),
            "exclusionPlanFingerprint": (
                exclusion_plan.fingerprint if exclusion_plan else None
            ),
            "exclusionRanges": (
                exclusion_plan.source["ranges"] if exclusion_plan else []
            ),
            "trainCandidateStartsExcluded": (
                train_dataset.excluded_candidate_starts
            ),
            "validationCandidateStartsExcluded": (
                validation_dataset.excluded_candidate_starts
            ),
            "lookback": args.lookback,
            "horizon": args.horizon,
            "epochs": args.epochs,
            "trainSamplesPerEpoch": args.train_samples_per_epoch,
            "validationSamples": args.validation_samples,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "weightDecay": args.weight_decay,
            "forecastLossWeight": args.forecast_loss_weight,
            "gradientAccumulation": args.gradient_accumulation,
            "seed": args.seed,
            "amp": amp,
            "baselineValidationLoss": baseline_validation,
            "baselineValidationFullLoss": baseline_full,
            "baselineValidationForecastLoss": baseline_forecast,
            "bestValidationLoss": best_validation,
            "bestValidationFullLoss": best_validation_full,
            "bestValidationForecastLoss": best_validation_forecast,
            "bestEpoch": best_epoch,
            "history": history,
            "durationSeconds": time.monotonic() - started,
        }, output_dir / "manifest.json")
    print(
        f"DONE baselineValidation={baseline_validation:.6f} "
        f"bestValidation={best_validation:.6f} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
