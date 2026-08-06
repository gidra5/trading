from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

from differentiable_exposure_value_oracle import DifferentiableExposureValueOracle
from next_return_dataset import HISTORY_RETURN_COUNT
from trading_storage import load_torch_checkpoint, read_candle_column
from train_oracle_distribution_path import (
    build_model,
    normalization_from_json,
    oracle_config,
    validate_plan,
)


INTERVAL_MS = 60_000
ROWS_PER_DAY = 1_440
HEADER = struct.Struct("<IIQ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve causal path-model oracle distributions over a binary pipe."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--history-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


class MinuteCloseRange:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache: dict[str, np.ndarray] = {}

    def day(self, value: str) -> np.ndarray:
        cached = self.cache.get(value)
        if cached is not None:
            return cached
        file = self.root / f"{value}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing 1m candle history: {file}")
        closes = read_candle_column(file, "close")
        if closes.shape != (ROWS_PER_DAY,) \
                or not np.isfinite(closes).all() \
                or bool((closes <= 0).any()):
            raise ValueError(f"invalid 1m candle history: {file}")
        self.cache[value] = closes
        if len(self.cache) > 4:
            self.cache.pop(next(iter(self.cache)))
        return closes

    def load(self, start_ms: int, count: int) -> np.ndarray:
        if start_ms % INTERVAL_MS != 0 or count < 1:
            raise ValueError("minute close range must be aligned and non-empty")
        result = np.empty(count, dtype=np.float64)
        position = 0
        cursor = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
        while position < count:
            minute = cursor.hour * 60 + cursor.minute
            take = min(count - position, ROWS_PER_DAY - minute)
            values = self.day(cursor.date().isoformat())
            result[position:position + take] = values[minute:minute + take]
            position += take
            cursor += timedelta(minutes=take)
        return result


def histories_for_window(
    closes: MinuteCloseRange,
    start_ms: int,
    end_ms: int,
) -> np.ndarray:
    if start_ms % INTERVAL_MS != 0 or end_ms % INTERVAL_MS != 0 \
            or end_ms <= start_ms:
        raise ValueError("model window must be a non-empty aligned minute range")
    rows = (end_ms - start_ms) // INTERVAL_MS
    values = closes.load(
        start_ms - HISTORY_RETURN_COUNT * INTERVAL_MS,
        rows + HISTORY_RETURN_COUNT,
    )
    returns = np.diff(np.log(values)).astype(np.float32)
    histories = np.lib.stride_tricks.sliding_window_view(
        returns, HISTORY_RETURN_COUNT
    )
    if histories.shape != (rows, HISTORY_RETURN_COUNT):
        raise RuntimeError("model history construction is misaligned")
    return np.ascontiguousarray(histories)


def selected_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def serve(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    validate_plan(plan)
    manifest = json.loads(
        (repo_root / plan["datasetDir"] / "dataset.json").read_text(encoding="utf-8")
    )
    normalization = normalization_from_json(manifest["normalization"])
    result = json.loads(
        (repo_root / plan["runDir"] / "state" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    device = selected_device(args.device)
    model = build_model(normalization, plan["architecture"]).to(device)
    checkpoint = load_torch_checkpoint(
        repo_root / result["checkpoint"], map_location=device, weights_only=False
    )
    if checkpoint.get("planSha256") != result.get("planSha256"):
        raise ValueError("checkpoint and trained result contracts differ")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    oracle = DifferentiableExposureValueOracle(oracle_config(plan)).to(device)
    oracle.eval()
    closes = MinuteCloseRange(args.history_dir.resolve())
    action_count = int(oracle.grid.numel())
    print(json.dumps({
        "event": "ready",
        "modelId": plan["id"],
        "device": str(device),
        "actionCount": action_count,
    }, separators=(",", ":")), file=sys.stderr, flush=True)

    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        request = json.loads(line)
        start_ms = int(request["startTime"])
        end_ms = int(request["endTime"])
        started = time.perf_counter()
        histories = histories_for_window(closes, start_ms, end_ms)
        rows = histories.shape[0]
        probabilities = np.empty((rows, action_count), dtype="<f4")
        with torch.inference_mode():
            for start in range(0, rows, args.batch_size):
                end = min(rows, start + args.batch_size)
                features = torch.from_numpy(histories[start:end]).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    predicted_returns = model(features)
                with torch.autocast(device_type=device.type, enabled=False):
                    output = oracle.forward_from_log_returns(
                        predicted_returns.float()
                    )
                probabilities[start:end] = (
                    output.probabilities.detach().float().cpu().numpy()
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration_ms = round((time.perf_counter() - started) * 1000)
        sys.stdout.buffer.write(HEADER.pack(rows, action_count, duration_ms))
        sys.stdout.buffer.write(probabilities.tobytes(order="C"))
        sys.stdout.buffer.flush()
        print(json.dumps({
            "event": "result",
            "id": request.get("id"),
            "rows": rows,
            "durationMs": duration_ms,
            "rowsPerSecond": rows / max(duration_ms / 1000, 1e-9),
        }, separators=(",", ":")), file=sys.stderr, flush=True)


if __name__ == "__main__":
    serve(parse_args())
