from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time

import torch

from benchmark_kronos import Window, build_origins, load_corpus
from forecast_model_adapters import Chronos2Adapter
from screen_forecast_models import atomic_json, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and BTC-tuned Chronos-2 before test time.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 8, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2021, 8, 31))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--representation", choices=("raw", "anchored-log"), default="raw")
    parser.add_argument("--max-origins", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/chronos2-btc-lora-pretest-validation-2026-08-07.json"),
    )
    return parser.parse_args()


def utc_ms(value: date) -> int:
    return int(datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1_000)


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise ValueError("validation start must not follow validation end")
    repo_root = Path(__file__).resolve().parents[1]
    checkpoints = [(repo_root / value).resolve() for value in args.checkpoint]
    manifests = []
    for checkpoint in checkpoints:
        manifest_path = checkpoint.parent / "training-manifest.json"
        if not checkpoint.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"checkpoint or its training manifest is missing: {checkpoint}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"fine-tuning checkpoint is not complete: {checkpoint}")
        manifests.append((manifest_path, manifest))
    test_boundaries = {date.fromisoformat(item[1]["data"]["firstTestBoundary"]) for item in manifests}
    if len(test_boundaries) != 1:
        raise ValueError("candidate checkpoints declare different test boundaries")
    test_boundary = next(iter(test_boundaries))
    if args.end >= test_boundary:
        raise ValueError("validation range must end strictly before the declared test boundary")

    window = Window(
        id="pretest-validation",
        label="Pre-test validation",
        group="validation",
        start_time=utc_ms(args.start),
        end_time=utc_ms(args.end + timedelta(days=1)),
    )
    corpus = load_corpus((repo_root / args.history_dir).resolve(), (window,), args.context_length)
    origins = build_origins(corpus, (window,), args.context_length, args.max_origins)
    device = torch.device(args.device)
    started = time.perf_counter()
    results = []
    variants = [("base", None), *[(checkpoint.parent.name, checkpoint) for checkpoint in checkpoints]]
    for label, candidate_path in variants:
        print(f"VALIDATE {label} origins={len(origins)} representation={args.representation}", flush=True)
        adapter = Chronos2Adapter(repo_root, device=args.device, checkpoint=candidate_path)
        try:
            result = evaluate(
                adapter,
                corpus,
                origins,
                args.context_length,
                args.representation,
                device,
                args.batch_size,
            )
        finally:
            adapter.close()
        results.append({"variant": label, **result})
        candle = result["metrics"]["candle"]
        returns = result["metrics"]["closeReturn"]
        print(
            f"RESULT {label} candleSkill={candle['mseSkillVsPersistence']:.6f} "
            f"returnCorr={returns['correlation']} horizonCorr={returns['horizonCorrelation']}",
            flush=True,
        )
    report = {
        "version": 1,
        "contract": "chronos2-btc-pretest-validation-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "elapsedSeconds": time.perf_counter() - started,
        "data": {
            "market": "Binance spot BTCUSDT",
            "interval": "1m",
            "start": args.start.isoformat(),
            "endInclusive": args.end.isoformat(),
            "origins": len(origins),
            "contextLength": args.context_length,
            "predictionLength": 15,
            "representation": args.representation,
            "corpusFingerprint": corpus.fingerprint,
            "selectionRole": "strictly pre-test model selection",
        },
        "checkpoints": [str(path.relative_to(repo_root)).replace("\\", "/") for path in checkpoints],
        "trainingManifests": [
            str(path.relative_to(repo_root)).replace("\\", "/") for path, _manifest in manifests
        ],
        "results": results,
    }
    atomic_json(args.output, report)
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
