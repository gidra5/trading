from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_kronos import (
    ForecastAccumulator,
    ORACLE_CONFIG,
    ProbabilisticAccumulator,
    build_origins,
    load_corpus,
    metric_batch,
    oracle_path_mixture,
    parse_windows,
    probabilistic_batch,
)
from differentiable_exposure_value_oracle import DifferentiableExposureValueOracle
from forecast_candle_transforms import REPRESENTATIONS, decode_quantile_paths, encode_contexts
from forecast_model_adapters import QUANTILE_LEVELS, create_adapter
from forecast_model_zoo import selected_specs
from kronos_probabilistic import kqsp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free sparse screen of financial foundation models.")
    parser.add_argument("--windows-json", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--contexts", default="128,256,512")
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--max-origins-per-window", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/forecast-models-sparse-screen-2026-08-07.json"),
    )
    return parser.parse_args()


def batches(items: tuple, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate(
    adapter,
    corpus,
    origins,
    lookback: int,
    representation: str,
    device: torch.device,
    batch_size: int,
) -> dict:
    oracle = DifferentiableExposureValueOracle(ORACLE_CONFIG).to(device).eval()
    grid = oracle.grid.detach().double().cpu().numpy()
    forecast_metrics = ForecastAccumulator(grid.size)
    probability_metrics = ProbabilisticAccumulator(QUANTILE_LEVELS)
    latencies = []
    for origin_batch in batches(origins, batch_size):
        contexts = np.stack([
            corpus.values[origin.target_index - lookback : origin.target_index]
            for origin in origin_batch
        ])
        actual = np.stack([
            corpus.values[origin.target_index : origin.target_index + 15]
            for origin in origin_batch
        ])
        encoded, state = encode_contexts(contexts, representation)
        forecast = adapter.predict(encoded, prediction_length=15)
        latencies.append(forecast.latency_seconds)
        raw_paths = decode_quantile_paths(forecast.quantiles, state)
        raw_quantiles = np.moveaxis(raw_paths[..., :4], 1, 2)
        repaired_quantiles = kqsp(raw_quantiles)
        repaired_paths = np.moveaxis(repaired_quantiles, 2, 1)
        anchors = contexts[:, -1, 3]
        predicted_oracle, actual_oracle = oracle_path_mixture(
            repaired_paths,
            actual,
            anchors,
            oracle,
            device,
        )
        point = repaired_quantiles[:, :, 4, :]
        metric = metric_batch(point, actual, anchors, predicted_oracle, actual_oracle, grid)
        probability = probabilistic_batch(
            repaired_paths,
            raw_quantiles,
            repaired_quantiles,
            actual,
            anchors,
            QUANTILE_LEVELS,
        )
        indexes = range(len(origin_batch))
        forecast_metrics.add(metric, indexes)
        probability_metrics.add(probability, indexes)
    return {
        "metrics": forecast_metrics.result(grid),
        "probabilistic": probability_metrics.result(),
        "inference": {
            "batches": len(latencies),
            "seconds": float(sum(latencies)),
            "meanBatchSeconds": float(np.mean(latencies)),
            "originsPerSecond": len(origins) / max(sum(latencies), 1e-9),
        },
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    windows = parse_windows(args.windows_json)
    contexts = tuple(int(value) for value in args.contexts.split(","))
    representations = tuple(value.strip() for value in args.representations.split(",") if value.strip())
    if not contexts or any(value < 32 for value in contexts):
        raise ValueError("contexts must contain positive lengths of at least 32")
    if not representations or any(value not in REPRESENTATIONS for value in representations):
        raise ValueError(f"representations must be selected from {REPRESENTATIONS}")
    corpus = load_corpus((repo_root / args.history_dir).resolve(), windows, max(contexts))
    origins = build_origins(corpus, windows, max(contexts), args.max_origins_per_window)
    device = torch.device(args.device)
    results = []
    started = time.perf_counter()
    batch_sizes = {"fincast": 8, "tirex2": 4, "chronos2": 32}
    for spec in selected_specs(args.models):
        shared_adapter = None
        if spec.id != "fincast":
            shared_adapter = create_adapter(spec.id, repo_root, context_length=max(contexts), device=args.device)
        try:
            for context_length in contexts:
                adapter = shared_adapter or create_adapter(
                    spec.id, repo_root, context_length=context_length, device=args.device
                )
                try:
                    for representation in representations:
                        print(
                            f"SCREEN {spec.id} context={context_length} representation={representation} "
                            f"origins={len(origins)}",
                            flush=True,
                        )
                        result = evaluate(
                            adapter,
                            corpus,
                            origins,
                            context_length,
                            representation,
                            device,
                            batch_sizes[spec.id],
                        )
                        item = {
                            "model": spec.id,
                            "contextLength": context_length,
                            "representation": representation,
                            **result,
                        }
                        results.append(item)
                        atomic_json(args.output, {
                            "version": 1,
                            "contract": "forecast-foundation-model-sparse-screen-v1",
                            "generatedAt": datetime.now(timezone.utc).isoformat(),
                            "status": "partial",
                            "origins": len(origins),
                            "results": results,
                        })
                        candle = result["metrics"]["candle"]
                        returns = result["metrics"]["closeReturn"]
                        print(
                            f"RESULT {spec.id}/{context_length}/{representation} "
                            f"candleSkill={candle['mseSkillVsPersistence']:.6f} "
                            f"returnCorr={returns['correlation']} "
                            f"horizonCorr={returns['horizonCorrelation']}",
                            flush=True,
                        )
                finally:
                    if shared_adapter is None:
                        adapter.close()
                        gc.collect()
        finally:
            if shared_adapter is not None:
                shared_adapter.close()
                gc.collect()
    report = {
        "version": 1,
        "contract": "forecast-foundation-model-sparse-screen-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "elapsedSeconds": time.perf_counter() - started,
        "data": {
            "market": "Binance spot BTCUSDT",
            "interval": "1m",
            "windows": len(windows),
            "excludedWindows": ["fit-full", "fit-1", "fit-2", "fit-3", "fit-4", "latest-3m"],
            "origins": len(origins),
            "selectionRole": "calibration-only sparse screen; not final evidence",
        },
        "results": results,
    }
    atomic_json(args.output, report)
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
