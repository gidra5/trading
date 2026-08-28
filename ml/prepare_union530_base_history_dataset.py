from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from global_feature_registry import ROOT, safe_id


GLOBAL_DATASET = (
    ROOT / "data/training/datasets/global-btc-production-471-next-1s-30d-v1"
)
PRODUCTION_DATASET = (
    ROOT
    / "data/training/datasets/next-return-production-basis-history120-full-30d-v1"
)
WORKING_SET = ROOT / "data/runtime-cache/global-btc-expanded-working-set-v4"
MINUTE_AXIS = ROOT / "data/runtime-cache/binance-cross-asset-1m-basis-30d"
SECOND_CLOSE_AXIS = (
    ROOT / "data/runtime-cache/binance-cross-asset-spot-1s-close-30d"
)
BOOK_STREAM = (
    ROOT / "data/market/mutable/streams/spot-btcusdt/btcusdt-orderbook.jsonl"
)
OUTPUT = ROOT / "data/training/datasets/union530-base-history-30d-v1"

BTC_SECOND_CANDLES = (
    "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"
)
BTC_MINUTE_CANDLES = (
    "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"
)
ETH_MINUTE_CANDLES = (
    "data/market/immutable/refs/candles/spot-ethusdt/ethusdt/1m"
)
SOL_MINUTE_CANDLES = (
    "data/market/immutable/refs/candles/spot-solusdt/solusdt/1m"
)
ETH_SECOND_CANDLES = (
    "data/market/immutable/refs/candles/spot-ethusdt/ethusdt/1s"
)
BTC_TRADE_FLOW = (
    "data/market/immutable/refs/research/trade-flow/spot-btcusdt/btcusdt/1s"
)
BTC_FUTURES_MINUTE = (
    "data/market/immutable/refs/research/derivatives-klines/"
    "usdm-futures/btcusdt/1m"
)
SPREAD_ID = "asset/btc/binance-spot/1m/spot-book-spread-bps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the base-only source contract for the differentiable "
            "production59 + global471 feature union."
        )
    )
    parser.add_argument("--global-dataset", type=Path, default=GLOBAL_DATASET)
    parser.add_argument("--production-dataset", type=Path, default=PRODUCTION_DATASET)
    parser.add_argument("--working-set", type=Path, default=WORKING_SET)
    parser.add_argument("--minute-axis", type=Path, default=MINUTE_AXIS)
    parser.add_argument("--second-close-axis", type=Path, default=SECOND_CLOSE_AXIS)
    parser.add_argument("--book-stream", type=Path, default=BOOK_STREAM)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {
        "file": relative(path),
        "bytes": path.stat().st_size,
    }
    if include_hash:
        value["sha256"] = sha256(path)
    return value


def valid_snapshot(value: dict[str, Any]) -> bool:
    if value.get("symbol") != "BTCUSDT" or not isinstance(value.get("eventTime"), int):
        return False
    bids, asks = value.get("bids"), value.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        return False
    try:
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
    except (KeyError, TypeError, ValueError):
        return False
    return bid > 0 and ask > bid


def snapshots(path: Path) -> Iterator[tuple[int, float, float]]:
    previous_time = -1
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not valid_snapshot(value):
                continue
            event_time = int(value["eventTime"])
            if event_time < previous_time:
                raise ValueError("spot-book snapshots are not chronological")
            previous_time = event_time
            yield (
                event_time,
                float(value["bids"][0]["price"]),
                float(value["asks"][0]["price"]),
            )


def causal_top_of_book(origins: np.ndarray, stream: Path) -> np.ndarray:
    """Recover the exact primitive bid/ask pair used by the 471st channel."""
    # Prices stay float64: BTC ticks are finer than float32 at the current
    # price level and the original spread export also used float64 prices.
    output = np.empty((origins.size, 2), dtype=np.float64)
    iterator = iter(snapshots(stream))
    upcoming = next(iterator, None)
    latest: tuple[int, float, float] | None = None
    for index, origin in enumerate(origins):
        boundary = int(origin) + 1_000
        while upcoming is not None and upcoming[0] < boundary:
            latest = upcoming
            upcoming = next(iterator, None)
        if latest is None or not 0 < boundary - latest[0] <= 5_000:
            raise ValueError(
                f"origin {int(origin)} has no causal top-of-book source within 5s"
            )
        output[index] = latest[1], latest[2]
    return output


def write_array(directory: Path, name: str, values: np.ndarray, dtype: str) -> dict[str, Any]:
    final = directory / name
    partial = final.with_name(final.name + ".partial")
    np.asarray(values, dtype=dtype).tofile(partial)
    os.replace(partial, final)
    return {**file_ref(final), "dtype": dtype, "shape": list(values.shape)}


def production_channel_ids(manifest: dict[str, Any]) -> list[str]:
    history = int(manifest["featureHistorySeconds"])
    channels = int(manifest["temporalChannelCount"])
    features = manifest["features"]
    if len(features) != history * channels:
        raise ValueError("production feature catalog does not match its temporal shape")
    # The export is channel-major and each channel ends with its lag-zero value.
    result = [str(features[(index + 1) * history - 1]["id"]) for index in range(channels)]
    if len(result) != 59 or len(set(result)) != 59:
        raise ValueError("expected 59 unique lag-zero production channels")
    return result


def source_asset_map(axis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for asset in axis["assets"]:
        result[safe_id(str(asset["asset"]))] = asset
    return result


def preferred_market(asset: dict[str, Any]) -> dict[str, Any]:
    preferred = asset.get("preferredMarket")
    if not isinstance(preferred, dict):
        raise ValueError(f'{asset.get("asset")}: preferred market is missing')
    market = next(
        (
            row for row in asset["markets"]
            if row["venue"] == preferred["venue"]
        ),
        None,
    )
    if market is None:
        raise ValueError(f'{asset.get("asset")}: preferred market file is missing')
    return market


def global_feature_specs(
    feature_ids: list[str],
    working: dict[str, Any],
    minute_axis: dict[str, Any],
    second_axis: dict[str, Any],
    *,
    eth_trade_count_source: str,
) -> list[dict[str, Any]]:
    coordinate_by_id = {str(row["id"]): row for row in working["coordinates"]}
    minute_assets = source_asset_map(minute_axis)
    second_assets = {
        safe_id(str(row["asset"])): row for row in second_axis["assets"]
    }
    specs: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        if feature_id == SPREAD_ID:
            specs.append({
                "id": feature_id,
                "provider": "top-of-book",
                "baseSource": "exampleTopOfBook",
                "operation": "10000*(ask-bid)/((ask+bid)/2)",
            })
            continue
        coordinate = coordinate_by_id.get(feature_id)
        if coordinate is None:
            raise ValueError(f"working set is missing {feature_id}")
        _, subject, _, cadence, formula = feature_id.split("/", 4)
        provider = str(coordinate["source"])
        spec: dict[str, Any] = {
            "id": feature_id,
            "provider": provider,
            "subject": subject,
            "cadence": cadence,
            "formula": formula,
        }
        if provider in {"dense-minute", "long-unique", "spectral-1m"}:
            asset = minute_assets.get(subject)
            if asset is None:
                raise ValueError(f"minute base asset is missing for {feature_id}")
            spec["baseSource"] = relative(ROOT / preferred_market(asset)["file"])
        elif provider in {"technical-1s", "spectral-1s"}:
            asset = second_assets.get(subject)
            if asset is None:
                raise ValueError(f"second-close base asset is missing for {feature_id}")
            spec["baseSource"] = relative(ROOT / asset["closeFile"])
            spec["observedSource"] = relative(ROOT / asset["observedFile"])
        elif provider == "funding-grid":
            asset = minute_assets.get(subject)
            if asset is None:
                raise ValueError(f"funding base asset is missing for {feature_id}")
            spec["baseSource"] = relative(
                resolved(MINUTE_AXIS) / "assets" / _encoded_asset(asset["asset"])
                / "usdm-funding.json"
            )
        elif provider == "representative-cross-asset":
            asset = minute_assets.get(subject)
            if asset is None:
                raise ValueError(f"cross-asset base is missing for {feature_id}")
            if cadence == "1s":
                second_asset = second_assets.get(subject)
                if second_asset is None:
                    raise ValueError(f"cross-asset second base is missing for {feature_id}")
                candle_fields = formula.startswith(("range-", "log-trade-count-"))
                if candle_fields:
                    if subject != "eth":
                        raise ValueError(
                            f"raw one-second OHLCV history is not registered for {feature_id}"
                        )
                    spec["baseSources"] = [ETH_SECOND_CANDLES]
                    if formula.startswith("log-trade-count-"):
                        spec["baseSources"].append(eth_trade_count_source)
                else:
                    spec["baseSources"] = [relative(ROOT / second_asset["closeFile"])]
                    spec["observedSource"] = relative(
                        ROOT / second_asset["observedFile"]
                    )
            else:
                spec["baseSources"] = representative_sources(asset, cadence, formula)
        elif provider == "existing-recent":
            spec["baseSources"] = existing_recent_sources(subject, cadence, formula)
        else:
            raise ValueError(f"unsupported selected provider {provider!r}: {feature_id}")
        specs.append(spec)
    if len(specs) != 471:
        raise ValueError(f"expected 471 global feature specs, found {len(specs)}")
    return specs


def _encoded_asset(value: str) -> str:
    import base64

    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def representative_sources(asset: dict[str, Any], cadence: str, formula: str) -> list[str]:
    directory = resolved(MINUTE_AXIS) / "assets" / _encoded_asset(str(asset["asset"]))
    if cadence == "5m":
        return [relative(directory / "usdm-metrics.f32")]
    if formula.startswith("book-"):
        return [relative(directory / "usdm-book-depth.f32")]
    return [relative(ROOT / preferred_market(asset)["file"])]


def existing_recent_sources(subject: str, cadence: str, formula: str) -> list[str]:
    if subject == "btc" and formula.startswith("spot-flow-"):
        return [BTC_TRADE_FLOW]
    if subject == "btc" and cadence == "1s":
        return [BTC_SECOND_CANDLES]
    if subject == "btc" and formula == "completed-1h-log-volume":
        return [BTC_SECOND_CANDLES]
    if subject == "btc" and formula.startswith("futures-"):
        return [BTC_FUTURES_MINUTE, BTC_MINUTE_CANDLES]
    if subject == "btc":
        return [BTC_MINUTE_CANDLES]
    if subject == "eth":
        return [ETH_MINUTE_CANDLES]
    if subject == "sol":
        return [SOL_MINUTE_CANDLES]
    raise ValueError(f"unsupported existing-recent lineage: {subject}/{cadence}/{formula}")


def _minute_market_source(subject: str) -> str:
    axis = json.loads((MINUTE_AXIS / "manifest.json").read_text(encoding="utf-8"))
    asset = source_asset_map(axis).get(subject)
    if asset is None:
        raise ValueError(f"minute source is missing for {subject}")
    return relative(ROOT / preferred_market(asset)["file"])


def production_specs(channel_ids: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for feature_id in channel_ids:
        sources: list[str]
        if feature_id.startswith("spot-"):
            sources = [BTC_TRADE_FLOW]
        elif feature_id.startswith("futures-"):
            sources = [BTC_FUTURES_MINUTE, BTC_MINUTE_CANDLES]
        elif feature_id.startswith("eth-"):
            sources = [ETH_MINUTE_CANDLES, BTC_MINUTE_CANDLES]
        elif feature_id.startswith("calendar-"):
            sources = ["implicit-utc-origin-time"]
        else:
            sources = [BTC_SECOND_CANDLES]
        specs.append({"id": feature_id, "baseSources": sources})
    return specs


def assert_source_files(specs: list[dict[str, Any]]) -> None:
    sources: set[str] = set()
    for spec in specs:
        for key in ("baseSource",):
            value = spec.get(key)
            if isinstance(value, str):
                sources.add(value)
        values = spec.get("baseSources")
        if isinstance(values, list):
            sources.update(str(value) for value in values)
    exempt = {"implicit-utc-origin-time", "exampleTopOfBook"}
    missing = [value for value in sorted(sources - exempt) if not (ROOT / value).exists()]
    if missing:
        raise ValueError(f"base source files are missing: {missing[:10]}")


def main() -> None:
    args = parse_args()
    global_root = resolved(args.global_dataset)
    production_root = resolved(args.production_dataset)
    working_root = resolved(args.working_set)
    minute_root = resolved(args.minute_axis)
    second_root = resolved(args.second_close_axis)
    stream = resolved(args.book_stream)
    output = resolved(args.output)

    global_manifest_file = global_root / "dataset.json"
    production_manifest_file = production_root / "manifest.json"
    working_manifest_file = working_root / "manifest.json"
    minute_manifest_file = minute_root / "manifest.json"
    second_manifest_file = second_root / "manifest.json"
    global_manifest = json.loads(global_manifest_file.read_text(encoding="utf-8"))
    production_manifest = json.loads(production_manifest_file.read_text(encoding="utf-8"))
    working = json.loads(working_manifest_file.read_text(encoding="utf-8"))
    minute_axis = json.loads(minute_manifest_file.read_text(encoding="utf-8"))
    second_axis = json.loads(second_manifest_file.read_text(encoding="utf-8"))

    global_ids = [str(value) for value in global_manifest["featureIds"]]
    production_ids = production_channel_ids(production_manifest)
    if len(global_ids) != 471:
        raise ValueError("global source must contain 471 selected features")
    if len(production_ids) + len(global_ids) != 530:
        raise ValueError("base-history union must reconstruct exactly 530 channels")

    output.mkdir(parents=True, exist_ok=True)
    minute_assets = source_asset_map(minute_axis)
    eth_asset = minute_assets["eth"]
    eth_fast = minute_root / "assets" / _encoded_asset(str(eth_asset["asset"])) / "spot-fast.f32"
    fast_values = np.memmap(
        eth_fast, dtype="<f4", mode="r", shape=(int(minute_axis["window"]["rows"]), 23)
    )
    eth_trade_count = np.rint(np.expm1(np.asarray(fast_values[:, 20], dtype=np.float64)))
    if np.any(eth_trade_count < 0) or np.max(
        np.abs(np.log1p(eth_trade_count) - np.asarray(fast_values[:, 20], dtype=np.float64))
    ) > 2e-6:
        raise ValueError("ETH trade-count primitive cannot be recovered exactly")
    eth_trade_count_file = write_array(
        output,
        "eth-minute-last-second-trade-count.f32",
        eth_trade_count.astype(np.float32),
        "<f4",
    )
    eth_trade_count_source = relative(output / eth_trade_count_file["file"].split("/")[-1])

    production_graph = production_specs(production_ids)
    global_graph = global_feature_specs(
        global_ids,
        working,
        minute_axis,
        second_axis,
        eth_trade_count_source=eth_trade_count_source,
    )
    assert_source_files(production_graph + global_graph)

    rows = int(global_manifest["rows"])
    origins = np.memmap(
        global_root / global_manifest["files"]["origins"]["file"],
        dtype="<f8", mode="r", shape=(rows,),
    )
    stored_spread = np.memmap(
        global_root / global_manifest["files"]["spread"]["file"],
        dtype="<f4", mode="r", shape=(rows,),
    )
    top = causal_top_of_book(np.asarray(origins), stream)
    reconstructed_spread = (
        10_000.0 * (top[:, 1] - top[:, 0]) / ((top[:, 1] + top[:, 0]) / 2.0)
    ).astype(np.float32)
    maximum_spread_error = float(np.max(np.abs(reconstructed_spread - stored_spread)))
    if maximum_spread_error > 1e-5:
        raise ValueError(
            f"top-of-book reconstruction changed spread by {maximum_spread_error} bps"
        )
    top_file = write_array(output, "example-top-of-book.f64", top, "<f8")

    source_manifests = {
        "globalExamples": file_ref(global_manifest_file),
        "productionFeatureCatalog": file_ref(production_manifest_file),
        "workingSetLineage": file_ref(working_manifest_file),
        "minuteBaseAxis": file_ref(minute_manifest_file),
        "secondCloseBaseAxis": file_ref(second_manifest_file),
    }
    manifest = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "id": output.name,
        "contract": "union530-base-history-differentiable-feature-graph-v1",
        "purpose": (
            "Base-only causal histories for reconstructing the exact production59 + "
            "global471 model input union in differentiable code."
        ),
        "examples": {
            "rows": rows,
            "origins": {
                **global_manifest["files"]["origins"],
                "file": relative(global_root / global_manifest["files"]["origins"]["file"]),
            },
            "targets": {
                **global_manifest["files"]["targets"],
                "file": relative(global_root / global_manifest["files"]["targets"]["file"]),
            },
            "minuteSourceRows": {
                **global_manifest["files"]["sourceRows"],
                "file": relative(global_root / global_manifest["files"]["sourceRows"]["file"]),
            },
            "splits": {
                **global_manifest["files"]["splits"],
                "file": relative(global_root / global_manifest["files"]["splits"]["file"]),
            },
            "nonzero": {
                **global_manifest["files"]["nonzero"],
                "file": relative(global_root / global_manifest["files"]["nonzero"]["file"]),
            },
            "topOfBook": top_file,
            "topOfBookColumns": ["bestBidPrice", "bestAskPrice"],
            "topOfBookMaximumAgeMs": 5_000,
            "maximumSpreadParityErrorBps": maximum_spread_error,
            "ethMinuteLastSecondTradeCount": eth_trade_count_file,
        },
        "timeframe": global_manifest["timeframe"],
        "baseHistory": {
            "secondStartMs": int(second_axis["window"]["startMs"]),
            "secondEndExclusiveMs": int(second_axis["window"]["endExclusiveMs"]),
            "secondRows": int(second_axis["window"]["rows"]),
            "minuteStart": minute_axis["window"]["start"],
            "minuteEndExclusive": minute_axis["window"]["endExclusive"],
            "minuteRows": int(minute_axis["window"]["rows"]),
            "sourceManifests": source_manifests,
        },
        "features": {
            "count": 530,
            "productionChannelCount": 59,
            "globalChannelCount": 471,
            "ids": production_ids + global_ids,
            "productionGraph": production_graph,
            "globalGraph": global_graph,
        },
        "adversarialBaseParameterization": {
            "continuous": (
                "per-source-field standardized raw primitive values with "
                "domain projection after each perturbation"
            ),
            "fixed": (
                "timestamps, availability masks, categorical aggressor sides, "
                "market identity, and source publication boundaries"
            ),
            "projection": (
                "reconstruct positive OHLCV and depth; enforce high>=max(open,close), "
                "low<=min(open,close), quantities/counts>=0, and bid<ask"
            ),
            "normalization": "per-base-coordinate training-split center and scale",
        },
        "invariants": [
            "No derived model input is stored by this dataset.",
            "Every perturbable value is a primitive base coordinate or base history value.",
            "All rolling, lagged, recursive, spectral, cross-source, and spread inputs are recomputed after perturbation.",
            "Targets, timestamps, masks, categorical states, and publication boundaries are never perturbed.",
        ],
    }
    manifest_file = output / "dataset.json"
    partial = manifest_file.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, manifest_file)
    print(json.dumps({
        "dataset": relative(output),
        "rows": rows,
        "featuresReconstructed": 530,
        "productionChannels": 59,
        "globalChannels": 471,
        "spreadParityMaximumErrorBps": maximum_spread_error,
    }), flush=True)


if __name__ == "__main__":
    main()
