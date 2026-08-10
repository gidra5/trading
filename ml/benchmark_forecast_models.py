from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_kronos import (
    EXECUTION_ORACLE_CONFIG,
    ORACLE_CONFIG,
    ForecastAccumulator,
    ProbabilisticAccumulator,
    build_origins,
    causal_forecast_rows,
    load_corpus,
    metric_batch,
    oracle_path_distributions,
    oracle_path_mixture,
    parse_windows,
    probabilistic_batch,
)
from differentiable_exposure_value_oracle import DifferentiableExposureValueOracle
from forecast_candle_transforms import decode_quantile_paths, encode_contexts
from forecast_model_adapters import QUANTILE_LEVELS, create_adapter
from kronos_probabilistic import kqsp


FORECAST_CONTRACT = "foundation-forecast-causal-15x1m-v1"
REPORT_CONTRACT = "foundation-forecast-all-inspector-windows-v1"


@dataclass(frozen=True)
class Variant:
    id: str
    model_id: str
    context_length: int
    representation: str
    batch_size: int
    checkpoint: str | None = None
    selection: str = ""


VARIANTS = (
    Variant(
        id="fincast-zero-shot",
        model_id="fincast",
        context_length=128,
        representation="anchored-log",
        batch_size=16,
        selection="frozen from the pre-test sparse and 217-origin calibration screens",
    ),
    Variant(
        id="tirex2-zero-shot",
        model_id="tirex2",
        context_length=256,
        representation="anchored-log",
        batch_size=64,
        selection="frozen from the pre-test sparse and 217-origin calibration screens",
    ),
    Variant(
        id="chronos2-zero-shot",
        model_id="chronos2",
        context_length=256,
        representation="raw",
        batch_size=128,
        selection="untuned control for the selected Chronos-2 representation",
    ),
    Variant(
        id="chronos2-btc-lora",
        model_id="chronos2",
        context_length=256,
        representation="raw",
        batch_size=128,
        checkpoint=(
            "data/models/forecast-foundation/chronos2-btc-pretest-lora/"
            "steps-300-lr-0.0001-ctx-256-seed-20260807/merged-checkpoint"
        ),
        selection="frozen winner on 2,976 strictly pre-test August 2021 validation origins",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense all-window benchmark of frozen foundation forecasts.")
    parser.add_argument("--windows-json", required=True)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-origins-per-window", type=int)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/forecast-models-dense-all-windows-2026-08-07.json"),
    )
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=Path("data/benchmarks/forecast-model-forecasts-2026-08-07"),
    )
    return parser.parse_args()


def selected_variants(value: str) -> tuple[Variant, ...]:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    if not requested or requested == {"all"}:
        return VARIANTS
    available = {item.id: item for item in VARIANTS}
    unknown = requested - available.keys()
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(sorted(unknown))}")
    return tuple(item for item in VARIANTS if item.id in requested)


def batches(items: tuple, size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_signature(variant: Variant, corpus_fingerprint: str, origins: tuple) -> str:
    value = {
        "contract": REPORT_CONTRACT,
        "variant": asdict(variant),
        "corpusFingerprint": corpus_fingerprint,
        "originCount": len(origins),
        "firstOrigin": origins[0].target_start,
        "lastOrigin": origins[-1].target_start,
        "quantiles": QUANTILE_LEVELS.tolist(),
        "oracle": asdict(ORACLE_CONFIG),
        "executionOracle": asdict(EXECUTION_ORACLE_CONFIG),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def evaluate_variant(repo_root: Path, variant: Variant, corpus, origins, windows, device, forecast_dir) -> dict:
    checkpoint = (repo_root / variant.checkpoint).resolve() if variant.checkpoint else None
    adapter = create_adapter(
        variant.model_id,
        repo_root,
        context_length=variant.context_length,
        device=str(device),
        chronos_checkpoint=checkpoint,
    )
    scoring_oracle = DifferentiableExposureValueOracle(ORACLE_CONFIG).to(device).eval()
    execution_oracle = DifferentiableExposureValueOracle(EXECUTION_ORACLE_CONFIG).to(device).eval()
    scoring_grid = scoring_oracle.grid.detach().double().cpu().numpy()
    execution_grid = execution_oracle.grid.detach().double().cpu().numpy()
    global_forecast = ForecastAccumulator(scoring_grid.size)
    global_probability = ProbabilisticAccumulator(QUANTILE_LEVELS)
    window_forecast = {window.id: ForecastAccumulator(scoring_grid.size) for window in windows}
    window_probability = {window.id: ProbabilisticAccumulator(QUANTILE_LEVELS) for window in windows}
    inference_seconds = 0.0
    batch_count = 0
    fp32_fallback_batches = 0
    rows: list[dict] = []
    started = time.perf_counter()
    try:
        for start, origin_batch in batches(origins, variant.batch_size):
            contexts = np.stack([
                corpus.values[origin.target_index - variant.context_length : origin.target_index]
                for origin in origin_batch
            ])
            actual = np.stack([
                corpus.values[origin.target_index : origin.target_index + 15]
                for origin in origin_batch
            ])
            encoded, state = encode_contexts(contexts, variant.representation)
            forecast = adapter.predict(encoded, prediction_length=15)
            inference_seconds += forecast.latency_seconds
            batch_count += 1
            fp32_fallback_batches += int(bool(forecast.metadata.get("fp32Fallback", False)))
            raw_paths = decode_quantile_paths(forecast.quantiles, state)
            raw_quantiles = np.moveaxis(raw_paths[..., :4], 1, 2)
            repaired_quantiles = kqsp(raw_quantiles)
            repaired_paths = np.moveaxis(repaired_quantiles, 2, 1)
            anchors = contexts[:, -1, 3]
            predicted_oracle, actual_oracle = oracle_path_mixture(
                repaired_paths, actual, anchors, scoring_oracle, device
            )
            execution_votes, execution_utility = oracle_path_distributions(
                repaired_paths, anchors, execution_oracle, device
            )
            point = repaired_quantiles[:, :, 4, :]
            metric = metric_batch(point, actual, anchors, predicted_oracle, actual_oracle, scoring_grid)
            probability = probabilistic_batch(
                repaired_paths,
                raw_quantiles,
                repaired_quantiles,
                actual,
                anchors,
                QUANTILE_LEVELS,
            )
            all_indexes = range(len(origin_batch))
            global_forecast.add(metric, all_indexes)
            global_probability.add(probability, all_indexes)
            for window in windows:
                selected = [index for index, origin in enumerate(origin_batch) if window.id in origin.window_ids]
                window_forecast[window.id].add(metric, selected)
                window_probability[window.id].add(probability, selected)
            predictions = {
                "ensembleMean": repaired_paths.mean(axis=1),
                "ensembleMedian": np.median(repaired_paths, axis=1),
            }
            rows.extend(causal_forecast_rows(
                origin_batch,
                repaired_paths,
                predictions,
                predicted_oracle,
                execution_votes,
                execution_utility,
                anchors,
            ))
            if batch_count % 50 == 0 or start + len(origin_batch) == len(origins):
                print(
                    f"PROGRESS {variant.id} {start + len(origin_batch)}/{len(origins)} "
                    f"({100 * (start + len(origin_batch)) / len(origins):.1f}%)",
                    flush=True,
                )
    finally:
        adapter.close()
        gc.collect()

    signature = run_signature(variant, corpus.fingerprint, origins)
    forecast_path = forecast_dir / f"{variant.id}.json"
    forecast_artifact = {
        "version": 1,
        "contract": FORECAST_CONTRACT,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "runSignature": signature,
        "modelId": variant.id,
        "market": "Binance spot BTCUSDT",
        "intervalMs": 60_000,
        "lookbackCandles": variant.context_length,
        "horizonCandles": 15,
        "representation": variant.representation,
        "nativeQuantileLevels": QUANTILE_LEVELS.tolist(),
        "pathInterpretation": "nine native quantile trajectories used as equal-weight distribution support",
        "oracleGrid": scoring_grid.tolist(),
        "executionOracleGrid": execution_grid.tolist(),
        "executionOracle": asdict(EXECUTION_ORACLE_CONFIG),
        "rows": rows,
    }
    atomic_json(forecast_path, forecast_artifact)
    window_results = []
    for window in windows:
        window_results.append({
            "windowId": window.id,
            "label": window.label,
            "group": window.group,
            "startTime": window.start_time,
            "endTime": window.end_time,
            "metrics": window_forecast[window.id].result(scoring_grid),
            "probabilistic": window_probability[window.id].result(),
        })
    return {
        "variant": asdict(variant),
        "runSignature": signature,
        "forecastArtifact": str(forecast_path.relative_to(repo_root)).replace("\\", "/"),
        "metrics": global_forecast.result(scoring_grid),
        "probabilistic": global_probability.result(),
        "windows": window_results,
        "inference": {
            "batches": batch_count,
            "seconds": inference_seconds,
            "originsPerSecond": len(origins) / max(inference_seconds, 1e-9),
            "wallSeconds": time.perf_counter() - started,
            "fp32FallbackBatches": fp32_fallback_batches,
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    windows = parse_windows(args.windows_json)
    variants = selected_variants(args.variants)
    max_context = max(item.context_length for item in variants)
    corpus = load_corpus((repo_root / args.history_dir).resolve(), windows, max_context)
    origins = build_origins(corpus, windows, max_context, args.max_origins_per_window)
    if not origins:
        raise RuntimeError("benchmark has no forecast origins")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    forecast_dir = (repo_root / args.forecast_dir).resolve()
    results = []
    total_started = time.perf_counter()
    for variant in variants:
        print(
            f"BENCHMARK {variant.id} context={variant.context_length} "
            f"representation={variant.representation} origins={len(origins)}",
            flush=True,
        )
        result = evaluate_variant(repo_root, variant, corpus, origins, windows, device, forecast_dir)
        results.append(result)
        partial = {
            "version": 1,
            "contract": REPORT_CONTRACT,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": "partial",
            "data": {
                "market": "Binance spot BTCUSDT",
                "interval": "1m",
                "windows": len(windows),
                "uniqueOrigins": len(origins),
                "excludedWindows": ["fit-full", "fit-1", "fit-2", "fit-3", "fit-4", "latest-3m"],
                "corpusFingerprint": corpus.fingerprint,
            },
            "results": results,
        }
        atomic_json(args.output, partial)
    report = {
        "version": 1,
        "contract": REPORT_CONTRACT,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "elapsedSeconds": time.perf_counter() - total_started,
        "data": {
            "market": "Binance spot BTCUSDT",
            "interval": "1m",
            "horizonCandles": 15,
            "windows": len(windows),
            "uniqueOrigins": len(origins),
            "excludedWindows": ["fit-full", "fit-1", "fit-2", "fit-3", "fit-4", "latest-3m"],
            "corpusFingerprint": corpus.fingerprint,
            "references": list(corpus.references),
            "evidenceRole": "untouched all-origin test benchmark; no model selection after this point",
        },
        "results": results,
    }
    atomic_json(args.output, report)
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
