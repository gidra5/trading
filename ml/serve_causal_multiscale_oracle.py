from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

from return_oracle_decoder_screen import LearnedRadiusShrinkingDecoder
from trading_storage import load_torch_checkpoint, read_shard_array


INTERVAL_MS = 60_000
DAY_SECONDS = 86_400
SOURCE_FEATURE_COUNT = 901
HEADER = struct.Struct("<IIQ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve direct causal multiscale oracle distributions."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--history-dir", type=Path)  # Protocol compatibility.
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def selected_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


class FeatureDays:
    def __init__(self, source_root: Path, source_manifest: dict) -> None:
        self.files = {
            value["date"]: source_root / value["features"]
            for value in source_manifest["componentLayout"]["inputComponents"]
        }
        self.cached_day: str | None = None
        self.cached_values: np.ndarray | None = None

    def load(self, day: str) -> np.ndarray:
        if day == self.cached_day and self.cached_values is not None:
            return self.cached_values
        file = self.files.get(day)
        if file is None:
            raise FileNotFoundError(
                f"causal multiscale feature day is outside the frozen corpus: {day}"
            )
        _metadata, values = read_shard_array(
            file, "<f2", (DAY_SECONDS, SOURCE_FEATURE_COUNT)
        )
        self.cached_day = day
        self.cached_values = values
        return values

    def window(self, start_ms: int, end_ms: int, feature_count: int) -> np.ndarray:
        if start_ms % INTERVAL_MS or end_ms % INTERVAL_MS or end_ms <= start_ms:
            raise ValueError("inference window must be a non-empty aligned minute range")
        count = (end_ms - start_ms) // INTERVAL_MS
        result = np.empty((count, feature_count), dtype=np.float32)
        position = 0
        cursor = start_ms
        while position < count:
            stamp = datetime.fromtimestamp(cursor / 1000, timezone.utc)
            minute = stamp.hour * 60 + stamp.minute
            take = min(count - position, 1_440 - minute)
            rows = minute * 60 + 59 + np.arange(take, dtype=np.int64) * 60
            result[position:position + take] = self.load(
                stamp.date().isoformat()
            )[rows, :feature_count]
            position += take
            cursor += take * INTERVAL_MS
        return result


def serve(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent.parent
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    dataset_root = repo / plan["dataset"]["datasetDir"]
    manifest = json.loads((dataset_root / "dataset.json").read_text(encoding="utf-8"))
    run_root = repo / plan["runDir"]
    best_file = run_root / "checkpoints" / "best.json"
    device = selected_device(args.device)
    normalization = manifest["normalization"]
    model = LearnedRadiusShrinkingDecoder(
        torch.tensor(normalization["mean"], dtype=torch.float32),
        torch.tensor(normalization["std"], dtype=torch.float32),
        dropout=float(plan["architecture"]["dropout"]),
        dropout_rate=float(plan["architecture"]["dropoutRate"]),
        initial_radius=float(plan["architecture"]["initialRadius"]),
        minimum_radius=float(plan["architecture"]["minimumRadius"]),
        output_count=int(manifest["actionCount"]),
    ).to(device)
    checkpoint = load_torch_checkpoint(best_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    source_root = repo / plan["dataset"]["sourceDatasetDir"]
    source_manifest = json.loads((source_root / "dataset.json").read_text(encoding="utf-8"))
    feature_days = FeatureDays(source_root, source_manifest)
    action_count = int(manifest["actionCount"])
    feature_count = int(manifest["featureCount"])
    calibration_file = run_root / "calibration" / "logit-sharpening.json"
    sharpening_factor = 1.0
    if calibration_file.is_file():
        calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
        sharpening_factor = float(calibration["selectedFactor"])
        if not np.isfinite(sharpening_factor) or sharpening_factor <= 0:
            raise ValueError("invalid calibrated logit sharpening factor")
    print(json.dumps({
        "event": "ready",
        "modelId": plan["id"],
        "bestEpoch": int(checkpoint["epoch"]),
        "device": str(device),
        "actionCount": action_count,
        "logitSharpeningFactor": sharpening_factor,
    }, separators=(",", ":")), file=sys.stderr, flush=True)

    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        request = json.loads(line)
        start_ms = int(request["startTime"])
        end_ms = int(request["endTime"])
        started = time.perf_counter()
        features = feature_days.window(start_ms, end_ms, feature_count)
        probabilities = np.empty((features.shape[0], action_count), dtype="<f4")
        with torch.inference_mode():
            for start in range(0, features.shape[0], args.batch_size):
                end = min(features.shape[0], start + args.batch_size)
                batch = torch.from_numpy(features[start:end]).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = sharpening_factor * model(batch)
                probabilities[start:end] = torch.softmax(
                    logits.float(), dim=-1
                ).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration_ms = round((time.perf_counter() - started) * 1000)
        sys.stdout.buffer.write(
            HEADER.pack(features.shape[0], action_count, duration_ms)
        )
        sys.stdout.buffer.write(probabilities.tobytes(order="C"))
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    serve(parse_args())
