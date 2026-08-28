from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from global_feature_registry import ROOT
from trading_storage import read_candle_column


WORKING_SET = ROOT / "data/runtime-cache/global-btc-expanded-working-set-v4"
BASE_DATASET = ROOT / "data/runtime-cache/global-feature-basis-30d"
ROBUSTNESS = ROOT / "data/benchmarks/global-btc-v4-production-final-transfer-robustness.json"
INCUMBENT_ROBUSTNESS = ROOT / "data/benchmarks/global-btc-v3-operational-transfer-robustness.json"
HISTORY = ROOT / "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"
BOOK_STREAM = ROOT / "data/market/mutable/streams/spot-btcusdt/btcusdt-orderbook.jsonl"
OUTPUT_ROOT = ROOT / "data/training/datasets"
SPREAD_ID = "asset/btc/binance-spot/1m/spot-book-spread-bps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare compact indexed 470/471-input next-second training datasets."
    )
    parser.add_argument("--working-set", type=Path, default=WORKING_SET)
    parser.add_argument("--base-dataset", type=Path, default=BASE_DATASET)
    parser.add_argument("--robustness", type=Path, default=ROBUSTNESS)
    parser.add_argument("--incumbent-robustness", type=Path, default=INCUMBENT_ROBUSTNESS)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--book-stream", type=Path, default=BOOK_STREAM)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--suffix", default="30d-v1")
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_ids(robustness_file: Path, incumbent_file: Path) -> list[str]:
    current = json.loads(robustness_file.read_text(encoding="utf-8"))
    incumbent = json.loads(incumbent_file.read_text(encoding="utf-8"))
    present = {str(row["horizon"]) for row in current["results"]}
    rows = list(current["results"]) + [
        row for row in incumbent["results"] if str(row["horizon"]) not in present
    ]
    return sorted({
        str(feature_id)
        for row in rows
        for feature_id in row["recommendedRawInputIds"]
    })


def utc_ms(value: str) -> int:
    return int(np.datetime64(value.removesuffix("Z")).astype("datetime64[ms]").astype(np.int64))


def daily_dates(start_ms: int, end_ms: int, include_following: bool = False) -> list[str]:
    start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
    end = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date()
    count = (end - start).days + int(include_following)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def write_array(directory: Path, name: str, values: np.ndarray, dtype: str) -> dict:
    final = directory / name
    partial = final.with_name(final.name + ".partial")
    np.asarray(values, dtype=dtype).tofile(partial)
    os.replace(partial, final)
    return {"file": final.name, "dtype": dtype, "bytes": final.stat().st_size, "sha256": sha256(final)}


def valid_snapshot(value: dict) -> bool:
    if value.get("symbol") != "BTCUSDT" or not isinstance(value.get("eventTime"), int):
        return False
    bids, asks = value.get("bids"), value.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list) or len(bids) != 10 or len(asks) != 10:
        return False
    try:
        for index in range(10):
            bid_price = float(bids[index]["price"])
            bid_quantity = float(bids[index]["quantity"])
            ask_price = float(asks[index]["price"])
            ask_quantity = float(asks[index]["quantity"])
            if min(bid_price, bid_quantity, ask_price, ask_quantity) <= 0:
                return False
            if index and (
                bid_price > float(bids[index - 1]["price"])
                or ask_price < float(asks[index - 1]["price"])
            ):
                return False
        return float(bids[0]["price"]) < float(asks[0]["price"])
    except (KeyError, TypeError, ValueError):
        return False


def snapshots(path: Path) -> Iterator[tuple[int, float]]:
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
                raise ValueError("Spot-book snapshots are not chronologically ordered.")
            previous_time = event_time
            bid = float(value["bids"][0]["price"])
            ask = float(value["asks"][0]["price"])
            midpoint = (bid + ask) / 2
            yield event_time, (ask - bid) / midpoint * 10_000


def fresh_spreads(origins: np.ndarray, stream: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    keep = np.zeros(origins.size, dtype=bool)
    spread = np.empty(origins.size, dtype=np.float32)
    iterator = iter(snapshots(stream))
    upcoming = next(iterator, None)
    latest: tuple[int, float] | None = None
    consumed = 0
    for index, origin in enumerate(origins):
        boundary = int(origin) + 1_000
        while upcoming is not None and upcoming[0] < boundary:
            latest = upcoming
            consumed += 1
            upcoming = next(iterator, None)
        if latest is None:
            continue
        age = boundary - latest[0]
        if 0 < age <= 5_000:
            keep[index] = True
            spread[index] = latest[1]
    return keep, spread[keep], {"snapshotsConsumed": consumed, "maximumAgeMs": 5_000}


def count_by_split(splits: np.ndarray, nonzero: np.ndarray) -> dict:
    labels = ("train", "primary", "transfer")
    return {
        label: {
            "all": int(np.count_nonzero(splits == index)),
            "nonzero": int(np.count_nonzero((splits == index) & nonzero)),
        }
        for index, label in enumerate(labels)
    }


def main() -> None:
    args = parse_args()
    working_root = resolved(args.working_set)
    base_root = resolved(args.base_dataset)
    robustness_file = resolved(args.robustness)
    incumbent_file = resolved(args.incumbent_robustness)
    history_root = resolved(args.history)
    book_stream = resolved(args.book_stream)
    output_root = resolved(args.output_root)

    selected = production_ids(robustness_file, incumbent_file)
    if len(selected) != 471 or selected.count(SPREAD_ID) != 1:
        raise ValueError(f"Expected the audited 471-input union with one spread input, found {len(selected)}.")
    selected_470 = [feature_id for feature_id in selected if feature_id != SPREAD_ID]

    working_manifest_file = working_root / "manifest.json"
    working = json.loads(working_manifest_file.read_text(encoding="utf-8"))
    coordinate_index = {str(row["id"]): index for index, row in enumerate(working["coordinates"])}
    missing = set(selected) - set(coordinate_index)
    if missing:
        raise ValueError(f"Working set is missing {len(missing)} production coordinates.")
    selected_columns = np.asarray([coordinate_index[value] for value in selected_470], dtype=np.int64)
    matrix = np.memmap(
        working_root / working["file"], dtype=working["dtype"], mode="r",
        shape=(int(working["rows"]), int(working["columns"])),
    )
    finite_minute = np.all(np.isfinite(matrix[:, selected_columns]), axis=1)

    base = json.loads((base_root / "manifest.json").read_text(encoding="utf-8"))
    dataset = base["datasets"][0]
    if int(dataset["rows"]) != int(working["rows"]):
        raise ValueError("Base timeline and working-set row counts differ.")
    minute_times = np.memmap(
        base_root / dataset["files"]["times"], dtype="<f8", mode="r",
        shape=(int(dataset["rows"]),),
    )
    minute_splits = np.memmap(
        base_root / dataset["files"]["splits"], dtype="u1", mode="r",
        shape=(int(dataset["rows"]),),
    )
    split = base["split"]
    start_ms = utc_ms(split["start"])
    end_ms = utc_ms(split["endExclusive"])

    valid_rows = np.flatnonzero(finite_minute).astype(np.uint32)
    offsets = np.arange(60, dtype=np.int64) * 1_000
    origins = (np.asarray(minute_times[valid_rows], dtype=np.int64)[:, None] + offsets).reshape(-1)
    source_rows = np.repeat(valid_rows, 60)
    splits = np.repeat(np.asarray(minute_splits[valid_rows], dtype=np.uint8), 60)
    in_window = (origins >= start_ms) & (origins + 1_000 < end_ms + 1_000)
    origins, source_rows, splits = origins[in_window], source_rows[in_window], splits[in_window]

    dates = daily_dates(start_ms, end_ms, include_following=True)
    closes = np.concatenate([
        read_candle_column(history_root / f"{day}.json", "close") for day in dates
    ]).astype(np.float64, copy=False)
    next_returns = np.diff(np.log(closes)).astype(np.float32)
    target_indices = ((origins - start_ms) // 1_000).astype(np.int64)
    targets = next_returns[target_indices]
    valid_target = np.isfinite(targets)
    origins, source_rows, splits, targets = (
        value[valid_target] for value in (origins, source_rows, splits, targets)
    )
    nonzero = targets != 0

    common_source = {
        "strategy": "indexed-causal-minute-snapshot-with-on-demand-column-gather",
        "manifest": rel(working_manifest_file),
        "manifestSha256": sha256(working_manifest_file),
        "matrix": rel(working_root / working["file"]),
        "matrixDtype": working["dtype"],
        "matrixShape": [int(working["rows"]), int(working["columns"])],
        "selectedColumnIndices": selected_columns.tolist(),
        "carryRule": (
            "A feature snapshot whose stored origin is t is available after that completed "
            "one-second candle and is carried only across origins t through t+59s."
        ),
    }
    source_inputs = {
        "productionRobustness": {"file": rel(robustness_file), "sha256": sha256(robustness_file)},
        "incumbentRobustness": {"file": rel(incumbent_file), "sha256": sha256(incumbent_file)},
        "baseTimelineManifest": {
            "file": rel(base_root / "manifest.json"), "sha256": sha256(base_root / "manifest.json")
        },
        "btcOneSecondHistory": rel(history_root),
    }

    def emit(
        identifier: str,
        feature_ids: list[str],
        row_origins: np.ndarray,
        row_targets: np.ndarray,
        row_sources: np.ndarray,
        row_splits: np.ndarray,
        row_nonzero: np.ndarray,
        spread_values: np.ndarray | None = None,
        spread_audit: dict | None = None,
    ) -> Path:
        directory = output_root / identifier
        directory.mkdir(parents=True, exist_ok=True)
        files = {
            "origins": write_array(directory, "origins.f64", row_origins, "<f8"),
            "targets": write_array(directory, "targets.f32", row_targets, "<f4"),
            "sourceRows": write_array(directory, "source-rows.u32", row_sources, "<u4"),
            "splits": write_array(directory, "splits.u8", row_splits, "u1"),
            "nonzero": write_array(directory, "nonzero.u8", row_nonzero, "u1"),
        }
        if spread_values is not None:
            files["spread"] = write_array(directory, "spot-spread-bps.f32", spread_values, "<f4")
        counts = count_by_split(row_splits, row_nonzero)
        manifest = {
            "schemaVersion": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "id": identifier,
            "purpose": "Training-ready compact global-feature next-one-second return dataset",
            "rows": int(row_origins.size),
            "featureCount": len(feature_ids),
            "targetCount": 1,
            "featureIds": feature_ids,
            "target": {
                "id": "next-completed-second-log-return",
                "dtype": "<f4",
                "construction": "float32(diff(log(float64 completed one-second closes)))",
            },
            "sampling": "one origin per completed second",
            "timeframe": {"start": split["start"], "endExclusive": split["endExclusive"]},
            "selectionModes": {
                "all": int(row_origins.size),
                "nonzero": int(np.count_nonzero(row_nonzero)),
                "nonzeroRule": "float32-exact-zero-after-log-return-construction",
            },
            "countsBySplit": counts,
            "featureStorage": common_source,
            "liveSpread": None if spread_values is None else {
                "featureId": SPREAD_ID,
                "position": len(feature_ids) - 1,
                "availabilityRule": "latest valid snapshot strictly before boundary, maximum age 5000ms",
                "source": rel(book_stream),
                "sourceSha256": sha256(book_stream),
                "audit": spread_audit,
            },
            "sourceInputs": source_inputs,
            "files": files,
        }
        manifest_file = directory / "dataset.json"
        temporary = manifest_file.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_file)
        print(json.dumps({
            "dataset": rel(directory), "rows": manifest["rows"],
            "nonzero": manifest["selectionModes"]["nonzero"], "features": len(feature_ids),
        }), flush=True)
        return manifest_file

    suffix = str(args.suffix)
    emit(
        f"global-btc-production-470-next-1s-{suffix}", selected_470,
        origins, targets, source_rows, splits, nonzero,
    )
    spread_keep, spread_values, spread_audit = fresh_spreads(origins, book_stream)
    spread_nonzero = nonzero[spread_keep]
    emit(
        f"global-btc-production-471-next-1s-{suffix}", selected_470 + [SPREAD_ID],
        origins[spread_keep], targets[spread_keep], source_rows[spread_keep],
        splits[spread_keep], spread_nonzero, spread_values, spread_audit,
    )


if __name__ == "__main__":
    main()
