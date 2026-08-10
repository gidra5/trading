from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finetune_kronos import (
    KronosWindowDataset,
    load_series,
    utc_ms,
)
from kronos_model_zoo import selected_specs, snapshot_dir, source_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Kronos tokenizer on canonical BTCUSDT 1m data."
    )
    parser.add_argument("--model", choices=("mini", "small", "base"), default="base")
    parser.add_argument("--train-start", default="2021-07-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-07-01")
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples-per-epoch", type=int, default=2_000)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Opt into mixed precision; FP32 is the stable default for BSQ.",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tools/Kronos-finetuned/btcusdt-1m-base-tokenizer"),
    )
    return parser.parse_args()


def validate(tokenizer, loader: DataLoader, device: torch.device, amp: bool) -> float:
    tokenizer.eval()
    squared_error = 0.0
    elements = 0
    with torch.no_grad():
        for values, _ in loader:
            values = values.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                (__, reconstructed), _, _, _ = tokenizer(values)
            squared_error += float(F.mse_loss(
                reconstructed.float(), values.float(), reduction="sum"
            ).item())
            elements += values.numel()
    if elements == 0:
        raise RuntimeError("empty tokenizer validation loader")
    return squared_error / elements


def atomic_json(value: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    for name in (
        "lookback", "epochs", "train_samples_per_epoch", "validation_samples",
        "batch_size", "gradient_accumulation", "log_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive and finite")
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
    spec = selected_specs(args.model)[0]
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
    train_dataset = KronosWindowDataset(
        train_series,
        window=args.lookback,
        lookback=args.lookback,
        samples=args.train_samples_per_epoch,
        seed=args.seed,
        training=True,
    )
    validation_dataset = KronosWindowDataset(
        validation_series,
        window=args.lookback,
        lookback=args.lookback,
        samples=args.validation_samples,
        seed=args.seed,
        training=False,
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
    from model import KronosTokenizer

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = device.type == "cuda" and args.amp
    pretrained_path = snapshot_dir(repo_root, spec.tokenizer_repo)
    tokenizer = KronosTokenizer.from_pretrained(str(pretrained_path)).to(device)
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=args.learning_rate,
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
    baseline_validation = validate(
        tokenizer, validation_loader, device, amp
    )
    print(f"BASELINE reconstructionMse={baseline_validation:.8f}", flush=True)
    best_validation = baseline_validation
    history = [{
        "epoch": 0,
        "trainObjective": None,
        "validationReconstructionMse": baseline_validation,
    }]
    started = time.monotonic()
    global_step = 0
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        tokenizer.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_batches = 0
        for batch_index, (values, _) in enumerate(train_loader):
            values = values.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                (reconstructed_pre, reconstructed), bsq_loss, _, _ = tokenizer(
                    values
                )
                reconstruction = (
                    F.mse_loss(reconstructed_pre, values)
                    + F.mse_loss(reconstructed, values)
                )
                loss = (reconstruction + bsq_loss) / 2
            scaler.scale(loss / args.gradient_accumulation).backward()
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            train_total += float(loss.item())
            train_batches += 1
            if should_update and global_step % args.log_interval == 0:
                print(
                    f"TRAIN epoch={epoch + 1}/{args.epochs} "
                    f"step={global_step}/{total_steps} loss={loss.item():.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        validation_loss = validate(
            tokenizer, validation_loader, device, amp
        )
        train_loss = train_total / train_batches
        history.append({
            "epoch": epoch + 1,
            "trainObjective": train_loss,
            "validationReconstructionMse": validation_loss,
        })
        print(
            f"EPOCH {epoch + 1} trainObjective={train_loss:.6f} "
            f"reconstructionMse={validation_loss:.8f}",
            flush=True,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            checkpoint = output_dir / "best_model"
            checkpoint.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(checkpoint)
            print(f"CHECKPOINT {checkpoint}", flush=True)
        atomic_json({
            "contract": "kronos-btcusdt-1m-tokenizer-finetune-v1",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "modelSelector": args.model,
            "tokenizerRepo": spec.tokenizer_repo,
            "tokenizerRevision": spec.tokenizer_revision,
            "trainRange": [args.train_start, args.train_end],
            "validationRange": [args.validation_start, args.validation_end],
            "lookback": args.lookback,
            "epochs": args.epochs,
            "trainSamplesPerEpoch": args.train_samples_per_epoch,
            "validationSamples": args.validation_samples,
            "batchSize": args.batch_size,
            "gradientAccumulation": args.gradient_accumulation,
            "learningRate": args.learning_rate,
            "weightDecay": args.weight_decay,
            "seed": args.seed,
            "amp": amp,
            "baselineValidationReconstructionMse": baseline_validation,
            "bestValidationReconstructionMse": best_validation,
            "history": history,
            "durationSeconds": time.monotonic() - started,
        }, output_dir / "manifest.json")
    print(
        f"DONE baselineReconstruction={baseline_validation:.8f} "
        f"bestReconstruction={best_validation:.8f} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
