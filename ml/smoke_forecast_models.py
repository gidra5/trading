from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from forecast_model_adapters import create_adapter
from forecast_model_zoo import selected_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and run every pinned forecast model.")
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context-length", type=int, default=128)
    return parser.parse_args()


def synthetic_context(context_length: int) -> np.ndarray:
    time = np.arange(context_length, dtype=np.float32)
    close = 100.0 + 0.02 * time + np.sin(time / 8.0)
    open_ = close + 0.05 * np.sin(time / 3.0)
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = 10.0 + np.cos(time / 5.0)
    return np.stack([open_, high, low, close, volume])[None, ...]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    context = synthetic_context(args.context_length)
    summaries = []
    for spec in selected_specs(args.models):
        adapter = create_adapter(
            spec.id,
            repo_root,
            context_length=args.context_length,
            device=args.device,
        )
        try:
            forecast = adapter.predict(context, prediction_length=15)
            summaries.append(
                {
                    "model": spec.id,
                    "pointShape": list(forecast.point.shape),
                    "quantileShape": list(forecast.quantiles.shape),
                    "quantileCrossingFraction": forecast.crossing_fraction,
                    "latencySeconds": forecast.latency_seconds,
                    "device": args.device,
                }
            )
        finally:
            adapter.close()
            gc.collect()
    print(json.dumps({"status": "ok", "models": summaries}, indent=2))


if __name__ == "__main__":
    main()
