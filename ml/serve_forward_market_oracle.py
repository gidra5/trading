from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

from forward_market_features import FORWARD_FEATURE_COUNT, build_forward_feature_day
from return_oracle_decoder_screen import LearnedRadiusShrinkingDecoder
from serve_causal_multiscale_oracle import FeatureDays, selected_device
from trading_storage import load_torch_checkpoint


INTERVAL_MS = 60_000
BASE_FEATURE_COUNT = 771
HEADER = struct.Struct("<IIQ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve causal forward-market oracle.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--history-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


class ForwardDayCache:
    def __init__(self, data_root: Path, maximum_days: int = 16) -> None:
        self.data_root = data_root
        self.maximum_days = maximum_days
        self.days: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, day: str) -> np.ndarray:
        cached = self.days.pop(day, None)
        if cached is None:
            cached = build_forward_feature_day(self.data_root, day)
        self.days[day] = cached
        while len(self.days) > self.maximum_days:
            self.days.popitem(last=False)
        return cached

    def window(self, start_ms: int, end_ms: int) -> np.ndarray:
        if start_ms % INTERVAL_MS or end_ms % INTERVAL_MS or end_ms <= start_ms:
            raise ValueError("forward inference window must be aligned and non-empty")
        count = (end_ms - start_ms) // INTERVAL_MS
        result = np.empty((count, FORWARD_FEATURE_COUNT), dtype=np.float32)
        position = 0
        cursor = start_ms
        while position < count:
            stamp = datetime.fromtimestamp(cursor / 1000, timezone.utc)
            minute = stamp.hour * 60 + stamp.minute
            take = min(count - position, 1_440 - minute)
            rows = minute + np.arange(take, dtype=np.int64)
            result[position:position + take] = self.load(
                stamp.date().isoformat()
            )[rows]
            position += take
            cursor += take * INTERVAL_MS
        return result


def serve(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent.parent
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    dataset_root = repo / plan["dataset"]["datasetDir"]
    manifest = json.loads((dataset_root / "dataset.json").read_text(encoding="utf-8"))
    if int(manifest["featureCount"]) != BASE_FEATURE_COUNT + FORWARD_FEATURE_COUNT:
        raise ValueError("forward-market serving feature count is invalid")
    run_root = repo / plan["runDir"]
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
    checkpoint = load_torch_checkpoint(
        run_root / "checkpoints/best.json", map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    source_root = repo / plan["dataset"]["sourceDatasetDir"]
    source_manifest = json.loads((source_root / "dataset.json").read_text(encoding="utf-8"))
    candle_features = FeatureDays(source_root, source_manifest)
    forward_features = ForwardDayCache(repo / "data")
    action_count = int(manifest["actionCount"])
    print(json.dumps({
        "event": "ready",
        "modelId": plan["id"],
        "bestEpoch": int(checkpoint["epoch"]),
        "device": str(device),
        "actionCount": action_count,
        "logitSharpeningFactor": 1,
    }, separators=(",", ":")), file=sys.stderr, flush=True)

    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        request = json.loads(line)
        start_ms = int(request["startTime"])
        end_ms = int(request["endTime"])
        started = time.perf_counter()
        features = np.column_stack((
            candle_features.window(start_ms, end_ms, BASE_FEATURE_COUNT),
            forward_features.window(start_ms, end_ms),
        )).astype(np.float32, copy=False)
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
                    logits = model(batch)
                probabilities[start:end] = torch.softmax(
                    logits.float(), dim=-1,
                ).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration_ms = round((time.perf_counter() - started) * 1_000)
        sys.stdout.buffer.write(HEADER.pack(features.shape[0], action_count, duration_ms))
        sys.stdout.buffer.write(probabilities.tobytes(order="C"))
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    serve(parse_args())
