"""Point-in-time, friction-aware Binance multiscale index backtest."""

from __future__ import annotations

import argparse
import base64
import calendar
import concurrent.futures
import csv
import faulthandler
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from portfolio_basis_walk_forward import (
    capped_proportional_weights_batch,
    select_basis_batch,
)


MINUTE_MS = 60_000
DAY_MS = 86_400_000
SAMPLE_COUNT = 360
INITIAL_INDEX_LEVEL = 1_000.0
REPORT_VERSION = 2
S3_ROOT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION_ROOT = "https://data.binance.vision/data"
STABLE_ASSETS = {
    "AEUR",
    "BFUSD",
    "BUSD",
    "DAI",
    "EURI",
    "EUR",
    "FDUSD",
    "PYUSD",
    "RLUSD",
    "TUSD",
    "USD",
    "USD1",
    "USDC",
    "USDE",
    "USDP",
    "USDS",
    "USDT",
    "XUSD",
}
LEVERAGED_SUFFIXES = ("BULL", "BEAR", "UP", "DOWN")
VENUE_RANK = {"spot": 0, "usdm-futures": 1, "coinm-futures": 2}
DEFAULT_FEE_BPS = {
    "spot": 10.0,
    "usdm-futures": 5.0,
    "coinm-futures": 5.0,
}


@dataclass(frozen=True)
class Scale:
    id: str
    minutes: int
    label: str
    sleeve_weight: float = 0.2


SCALES = (
    Scale("1d", 1_440, "360d × 1d"),
    Scale("4h", 240, "60d × 4h"),
    Scale("1h", 60, "15d × 1h"),
    Scale("15m", 15, "3.75d × 15m"),
    Scale("1m", 1, "6h × 1m"),
)


@dataclass(frozen=True)
class Market:
    venue: str
    symbol: str
    asset: str

    @property
    def archive_prefix(self) -> str:
        if self.venue == "spot":
            return "spot"
        if self.venue == "usdm-futures":
            return "futures/um"
        return "futures/cm"


@dataclass
class ArchiveStats:
    requested: int = 0
    found: int = 0
    missing: int = 0
    cache_hits: int = 0
    downloaded_bytes: int = 0
    parsed_candles: int = 0


@dataclass(frozen=True)
class PreparedScale:
    scale: Scale
    root: Path
    grid_start: int
    grid_count: int
    market_count: int
    returns_file: Path
    eligible_file: Path
    liquidity_file: Path
    ohlc_files: dict[str, Path]


@dataclass(frozen=True)
class SleeveResult:
    scale: Scale
    events: int
    weights_file: Path
    stats_file: Path
    mean_basis_size: float
    mean_eligible_assets: float
    target_reached_ratio: float
    mean_exposure: float
    selected_market_counts: np.ndarray


@dataclass(frozen=True)
class FundingData:
    minute_indexes: np.ndarray
    rates: np.ndarray
    stats: ArchiveStats
    file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and backtest the five-scale Binance basis independently at "
            "every information update, with long-only friction-aware rebalancing."
        ),
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start", default="2025-07-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--download-workers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scales",
        default="1d,4h,1h,15m,1m",
        help="Comma-separated scale IDs, primarily for diagnostics.",
    )
    parser.add_argument("--maximum-weight", type=float, default=0.05)
    parser.add_argument("--min-basis-size", type=int, default=8)
    parser.add_argument("--max-basis-size", type=int, default=512)
    parser.add_argument("--target-median-r2", type=float, default=0.8)
    parser.add_argument("--target-p10-r2", type=float, default=0.5)
    parser.add_argument("--residual-equivalence-band", type=float, default=0.05)
    parser.add_argument("--spot-fee-bps", type=float, default=10.0)
    parser.add_argument("--futures-fee-bps", type=float, default=5.0)
    parser.add_argument("--baseline-slippage-bps", type=float, default=5.0)
    parser.add_argument("--conservative-slippage-bps", type=float, default=20.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--refresh-universe", action="store_true")
    parser.add_argument("--refresh-market-data", action="store_true")
    parser.add_argument("--refresh-sleeves", action="store_true")
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir).resolve()
    args.start_ms = parse_day(args.start)
    args.finish_ms = parse_day(args.end) + DAY_MS
    if args.start_ms >= args.finish_ms:
        parser.error("--start must not be after --end")
    if args.batch_size <= 0 or args.download_workers <= 0:
        parser.error("worker and batch sizes must be positive")
    if not 0 < args.maximum_weight <= 1:
        parser.error("--maximum-weight must be in (0, 1]")
    if args.baseline_slippage_bps < 0 or args.conservative_slippage_bps < 0:
        parser.error("slippage assumptions must be non-negative")
    requested = set(args.scales.split(","))
    known = {scale.id for scale in SCALES}
    if not requested or not requested <= known:
        parser.error(f"--scales must use {','.join(scale.id for scale in SCALES)}")
    args.selected_scales = tuple(scale for scale in SCALES if scale.id in requested)
    return args


def main() -> None:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = args.data_dir / "portfolio-basis" / "walk-forward-index"
    root.mkdir(parents=True, exist_ok=True)
    markets, catalog = discover_markets(root, args.refresh_universe)
    assets = sorted({market.asset for market in markets})
    asset_markets = [
        sorted(
            (index for index, market in enumerate(markets) if market.asset == asset),
            key=lambda index: market_preference(markets[index]),
        )
        for asset in assets
    ]
    print(
        f"Historical archive universe: {len(markets):,} canonical markets, "
        f"{len(assets):,} economic assets.",
        flush=True,
    )

    archive_stats = ArchiveStats()
    lock = threading.Lock()
    prepared: dict[str, PreparedScale] = {}
    sleeves: dict[str, SleeveResult] = {}
    for scale in args.selected_scales:
        prepared_scale = prepare_market_data(
            args,
            root,
            markets,
            scale,
            archive_stats,
            lock,
        )
        prepared[scale.id] = prepared_scale
        sleeve = build_scale_sleeve(
            args,
            root,
            markets,
            assets,
            asset_markets,
            prepared_scale,
        )
        sleeves[scale.id] = sleeve
    funding = prepare_funding_data(args, root, markets)
    if args.prepare_only:
        print("Preparation complete; simulation skipped by --prepare-only.", flush=True)
        return
    if {scale.id for scale in SCALES} != set(sleeves):
        raise RuntimeError("The final multiscale simulation requires all five scales.")

    report, outputs = simulate_index(
        args,
        root,
        markets,
        catalog,
        archive_stats,
        prepared,
        sleeves,
        funding,
    )
    print("", flush=True)
    print(
        f"Baseline net index: {report['performance']['baseline']['finalLevel']:.2f} "
        f"({signed_percent(report['performance']['baseline']['totalReturn'])})",
        flush=True,
    )
    print(
        f"Gross index: {report['performance']['gross']['finalLevel']:.2f} "
        f"({signed_percent(report['performance']['gross']['totalReturn'])})",
        flush=True,
    )
    print(
        f"Cumulative baseline friction: "
        f"{report['performance']['baseline']['cumulativeTransactionCost']:.2f} "
        "index points",
        flush=True,
    )
    print(f"Candles: {outputs['candles']}", flush=True)
    print(f"JSON: {outputs['json']}", flush=True)
    print(f"Markdown: {outputs['markdown']}", flush=True)


def discover_markets(root: Path, refresh: bool) -> tuple[list[Market], dict[str, Any]]:
    cache = root / "historical-universe.json"
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text())
        return [Market(**value) for value in payload["markets"]], payload["catalog"]

    definitions = (
        ("spot", "data/spot/monthly/klines/"),
        ("usdm-futures", "data/futures/um/monthly/klines/"),
        ("coinm-futures", "data/futures/cm/monthly/klines/"),
    )
    listings: dict[str, list[str]] = {}
    for venue, prefix in definitions:
        print(f"Enumerating {venue} archive markets...", flush=True)
        listings[venue] = list_s3_directories(prefix)

    markets: list[Market] = []
    for venue, symbols in listings.items():
        for symbol in symbols:
            asset = archive_economic_asset(venue, symbol)
            if asset is None or asset in STABLE_ASSETS or is_leveraged(asset):
                continue
            markets.append(Market(venue=venue, symbol=symbol, asset=asset))
    markets.sort(key=market_preference)
    unique: dict[tuple[str, str], Market] = {}
    for market in markets:
        unique[(market.venue, market.symbol)] = market
    markets = list(unique.values())

    option_catalog = fetch_option_catalog()
    full_listings_file = root / "full-product-listings.json"
    write_json_atomic(
        full_listings_file,
        {
            "generatedAt": iso_time(int(time.time() * 1_000)),
            "totalProductRows": (
                sum(len(symbols) for symbols in listings.values())
                + option_catalog["listings"]
            ),
            "products": {
                **{
                    venue: [
                        {"symbol": symbol}
                        for symbol in symbols
                    ]
                    for venue, symbols in listings.items()
                },
                "options": option_catalog["contracts"],
            },
        },
    )
    catalog = {
        "archiveDirectories": {
            venue: len(symbols) for venue, symbols in listings.items()
        },
        "allProductRows": (
            sum(len(symbols) for symbols in listings.values())
            + option_catalog["listings"]
        ),
        "fullListingsFile": str(full_listings_file),
        "filteredContinuousMarkets": {
            venue: sum(market.venue == venue for market in markets)
            for venue, _ in definitions
        },
        "currentOptionListings": option_catalog["listings"],
        "currentOptionUnderlyings": option_catalog["underlyings"],
        "optionsNote": (
            "Expiring option contracts are catalogued but not treated as durable "
            "return series. Their economic underlyings are represented through "
            "continuous Spot, USD-M, or COIN-M markets."
        ),
    }
    write_json_atomic(
        cache,
        {
            "generatedAt": iso_time(int(time.time() * 1_000)),
            "catalog": catalog,
            "markets": [asdict(market) for market in markets],
        },
    )
    return markets, catalog


def list_s3_directories(prefix: str) -> list[str]:
    marker = ""
    values: list[str] = []
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        query = {
            "delimiter": "/",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if marker:
            query["marker"] = marker
        url = f"{S3_ROOT}?{urllib.parse.urlencode(query, safe='/')}"
        payload = fetch_bytes(url)
        document = ET.fromstring(payload)
        values.extend(
            item.text.rstrip("/").split("/")[-1]
            for item in document.findall("s3:CommonPrefixes/s3:Prefix", namespace)
            if item.text
        )
        if document.findtext("s3:IsTruncated", "false", namespace) != "true":
            break
        marker = document.findtext("s3:NextMarker", "", namespace)
        if not marker:
            raise RuntimeError(f"S3 listing for {prefix} did not return NextMarker")
    return values


def fetch_option_catalog() -> dict[str, Any]:
    try:
        payload = json.loads(
            fetch_bytes("https://eapi.binance.com/eapi/v1/exchangeInfo"),
        )
        rows = payload.get("optionSymbols") or payload.get("optionContracts") or []
        contracts = [
            {
                "symbol": str(
                    row.get("symbol")
                    or row.get("contractId")
                    or row.get("id")
                    or ""
                ),
                "underlying": str(row.get("underlying", "")),
            }
            for row in rows
            if isinstance(row, dict)
        ]
        underlyings = {
            row["underlying"].split("-")[0]
            for row in contracts
        }
        return {
            "listings": len(contracts),
            "underlyings": len(underlyings - {""}),
            "contracts": contracts,
        }
    except Exception as error:
        print(f"Warning: option catalog unavailable: {error}", flush=True)
        return {"listings": 0, "underlyings": 0, "contracts": []}


def archive_economic_asset(venue: str, symbol: str) -> str | None:
    if venue in {"spot", "usdm-futures"}:
        if not symbol.endswith("USDT") or "_" in symbol:
            return None
        base = symbol[: -len("USDT")]
    else:
        suffix = "USD_PERP"
        if not symbol.endswith(suffix):
            return None
        base = symbol[: -len(suffix)]
    return re.sub(r"^(?:1000|10000|1000000)(?=[A-Z])", "", base)


def is_leveraged(asset: str) -> bool:
    return any(
        len(asset) > len(suffix) and asset.endswith(suffix)
        for suffix in LEVERAGED_SUFFIXES
    )


def market_preference(market: Market) -> tuple[int, int, str, str]:
    symbol_base = (
        market.symbol.removesuffix("USDT")
        if market.venue != "coinm-futures"
        else market.symbol.removesuffix("USD_PERP")
    )
    multiplied = int(symbol_base != market.asset)
    return VENUE_RANK[market.venue], multiplied, market.asset, market.symbol


def prepare_market_data(
    args: argparse.Namespace,
    root: Path,
    markets: list[Market],
    scale: Scale,
    archive_stats: ArchiveStats,
    stats_lock: threading.Lock,
) -> PreparedScale:
    grid_start = args.start_ms - (SAMPLE_COUNT + 1) * scale.minutes * MINUTE_MS
    interval_ms = scale.minutes * MINUTE_MS
    grid_count = (args.finish_ms - grid_start) // interval_ms
    signature = hashlib.sha256(
        json.dumps(
            {
                "version": REPORT_VERSION,
                "start": args.start,
                "end": args.end,
                "scale": scale.id,
                "markets": [asdict(market) for market in markets],
            },
            sort_keys=True,
        ).encode(),
    ).hexdigest()[:16]
    scale_root = root / "market-data" / f"{args.start}_{args.end}-{scale.id}-{signature}"
    metadata_file = scale_root / "metadata.json"
    if metadata_file.exists() and not args.refresh_market_data:
        metadata = json.loads(metadata_file.read_text())
        if metadata.get("complete") is True:
            print(f"{scale.id}: reusing prepared market matrices.", flush=True)
            return prepared_scale_from_metadata(scale, scale_root, metadata)

    if scale_root.exists():
        shutil.rmtree(scale_root)
    scale_root.mkdir(parents=True)
    returns_file = scale_root / "returns-market-major.f32"
    eligible_file = scale_root / "eligible-market-major.u8"
    liquidity_file = scale_root / "liquidity-market-major.f32"
    returns = np.memmap(
        returns_file,
        dtype=np.float32,
        mode="w+",
        shape=(len(markets), grid_count),
    )
    eligible = np.memmap(
        eligible_file,
        dtype=np.uint8,
        mode="w+",
        shape=(len(markets), grid_count),
    )
    liquidity = np.memmap(
        liquidity_file,
        dtype=np.float32,
        mode="w+",
        shape=(len(markets), grid_count),
    )
    ohlc_market_major: dict[str, np.memmap] = {}
    if scale.id == "1m":
        for field in ("open", "high", "low", "close"):
            ohlc_market_major[field] = np.memmap(
                scale_root / f"{field}-return-market-major.f32",
                dtype=np.float32,
                mode="w+",
                shape=(len(markets), grid_count),
            )

    months = calendar_months(grid_start, args.finish_ms - interval_ms)
    print(
        f"{scale.id}: preparing {len(markets):,} markets × "
        f"{grid_count:,} periods from {len(months)} monthly archive slots...",
        flush=True,
    )
    predownload_archives(
        args,
        markets,
        scale,
        months,
        archive_stats,
    )
    started = time.monotonic()
    completed = 0
    completed_lock = threading.Lock()

    def process(index: int) -> None:
        nonlocal completed
        market = markets[index]
        row = parse_market_row(
            args,
            market,
            scale,
            months,
            grid_start,
            grid_count,
            archive_stats,
            stats_lock,
        )
        returns[index] = row["returns"]
        eligible[index] = row["eligible"]
        liquidity[index] = row["liquidity"]
        for field, matrix in ohlc_market_major.items():
            matrix[index] = row[field]
        with completed_lock:
            completed += 1
            if completed % 25 == 0 or completed == len(markets):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = completed / elapsed
                eta = (len(markets) - completed) / max(rate, 1e-9)
                print(
                    f"  {scale.id}: {completed:,}/{len(markets):,} markets; "
                    f"{rate:.1f}/s; ETA {duration(eta)}",
                    flush=True,
                )

    with concurrent.futures.ThreadPoolExecutor(
        # Archive rows are market-major on disk. A single sequential parser is
        # materially faster than multiple workers doing random reads while the
        # colocated training job is memory-mapped and under swap pressure.
        # curl downloads remain independently multiplexed.
        max_workers=1,
    ) as executor:
        futures = [executor.submit(process, index) for index in range(len(markets))]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    returns.flush()
    eligible.flush()
    liquidity.flush()
    for matrix in ohlc_market_major.values():
        matrix.flush()

    ohlc_files: dict[str, Path] = {}
    if scale.id == "1m":
        print("1m: transposing OHLC return matrices for sequential simulation...", flush=True)
        for field, source in ohlc_market_major.items():
            target_file = scale_root / f"{field}-return-time-major.f32"
            target = np.memmap(
                target_file,
                dtype=np.float32,
                mode="w+",
                shape=(grid_count, len(markets)),
            )
            block = 2_048
            for offset in range(0, grid_count, block):
                end = min(grid_count, offset + block)
                target[offset:end] = source[:, offset:end].T
            target.flush()
            del target
            ohlc_files[field] = target_file
        del ohlc_market_major
        for field in ("open", "high", "low", "close"):
            (scale_root / f"{field}-return-market-major.f32").unlink()

    metadata = {
        "complete": True,
        "version": REPORT_VERSION,
        "scale": asdict(scale),
        "gridStart": grid_start,
        "gridCount": grid_count,
        "marketCount": len(markets),
        "returnsFile": str(returns_file),
        "eligibleFile": str(eligible_file),
        "liquidityFile": str(liquidity_file),
        "ohlcFiles": {key: str(value) for key, value in ohlc_files.items()},
    }
    write_json_atomic(metadata_file, metadata)
    return prepared_scale_from_metadata(scale, scale_root, metadata)


def prepared_scale_from_metadata(
    scale: Scale,
    root: Path,
    metadata: dict[str, Any],
) -> PreparedScale:
    return PreparedScale(
        scale=scale,
        root=root,
        grid_start=int(metadata["gridStart"]),
        grid_count=int(metadata["gridCount"]),
        market_count=int(metadata["marketCount"]),
        returns_file=Path(metadata["returnsFile"]),
        eligible_file=Path(metadata["eligibleFile"]),
        liquidity_file=Path(metadata["liquidityFile"]),
        ohlc_files={
            key: Path(value) for key, value in metadata.get("ohlcFiles", {}).items()
        },
    )


def parse_market_row(
    args: argparse.Namespace,
    market: Market,
    scale: Scale,
    months: list[str],
    grid_start: int,
    grid_count: int,
    archive_stats: ArchiveStats,
    stats_lock: threading.Lock,
) -> dict[str, np.ndarray]:
    close_price = np.full(grid_count, np.nan, dtype=np.float64)
    volume = np.zeros(grid_count, dtype=np.float64)
    raw_ohlc = {
        field: np.full(grid_count, np.nan, dtype=np.float64)
        for field in ("open", "high", "low")
    }
    parsed = 0
    for month in months:
        archive = load_archive(args.data_dir, market, scale.id, month)
        with stats_lock:
            archive_stats.requested += 1
            if archive is None:
                archive_stats.missing += 1
            elif archive[1]:
                archive_stats.cache_hits += 1
            else:
                archive_stats.found += 1
                archive_stats.downloaded_bytes += archive[0].stat().st_size
        if archive is None:
            continue
        rows = parse_archive_csv(archive[0])
        if rows.size == 0:
            continue
        timestamps = normalize_timestamps(rows[:, 0])
        indexes = (timestamps - grid_start) // (scale.minutes * MINUTE_MS)
        valid = (
            (indexes >= 0)
            & (indexes < grid_count)
            & (timestamps % (scale.minutes * MINUTE_MS) == 0)
        )
        if not np.any(valid):
            continue
        indexes = indexes[valid].astype(np.int64)
        selected = rows[valid]
        close_price[indexes] = selected[:, 4]
        raw_ohlc["open"][indexes] = selected[:, 1]
        raw_ohlc["high"][indexes] = selected[:, 2]
        raw_ohlc["low"][indexes] = selected[:, 3]
        if market.venue == "coinm-futures":
            volume[indexes] = selected[:, 7] * selected[:, 4]
        else:
            volume[indexes] = selected[:, 7]
        parsed += len(indexes)
    with stats_lock:
        archive_stats.parsed_candles += parsed

    returns = np.full(grid_count, np.nan, dtype=np.float32)
    valid_pair = (
        np.isfinite(close_price[1:])
        & np.isfinite(close_price[:-1])
        & (close_price[1:] > 0)
        & (close_price[:-1] > 0)
    )
    indexes = np.flatnonzero(valid_pair) + 1
    returns[indexes] = np.log(
        close_price[indexes] / close_price[indexes - 1],
    ).astype(np.float32)

    finite = np.isfinite(returns)
    count = rolling_sum(finite.astype(np.float64), SAMPLE_COUNT)
    values = np.nan_to_num(returns.astype(np.float64))
    total = rolling_sum(values, SAMPLE_COUNT)
    square_total = rolling_sum(values * values, SAMPLE_COUNT)
    centered_square = square_total - total * total / SAMPLE_COUNT
    eligible = (
        (count == SAMPLE_COUNT)
        & np.isfinite(centered_square)
        & (centered_square > (SAMPLE_COUNT - 1) * 1e-20)
    ).astype(np.uint8)
    liquidity = rolling_sum(volume, SAMPLE_COUNT).astype(np.float32)

    result: dict[str, np.ndarray] = {
        "returns": returns,
        "eligible": eligible,
        "liquidity": liquidity,
    }
    if scale.id == "1m":
        for field in ("open", "high", "low"):
            relative = np.full(grid_count, np.nan, dtype=np.float32)
            valid_field = (
                np.isfinite(raw_ohlc[field][1:])
                & np.isfinite(close_price[:-1])
                & (raw_ohlc[field][1:] > 0)
                & (close_price[:-1] > 0)
            )
            field_indexes = np.flatnonzero(valid_field) + 1
            relative[field_indexes] = (
                raw_ohlc[field][field_indexes] / close_price[field_indexes - 1] - 1
            ).astype(np.float32)
            result[field] = relative
        close_return = np.full(grid_count, np.nan, dtype=np.float32)
        close_return[indexes] = (
            close_price[indexes] / close_price[indexes - 1] - 1
        ).astype(np.float32)
        result["close"] = close_return
    return result


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    output[window - 1 :] = prefix[window:] - prefix[:-window]
    return output


def load_archive(
    data_dir: Path,
    market: Market,
    interval: str,
    month: str,
) -> tuple[Path, bool] | None:
    file, missing = archive_paths(data_dir, market, interval, month)
    cache = file.parent
    cache.mkdir(parents=True, exist_ok=True)
    if file.exists():
        return file, True
    if missing.exists():
        return None
    url = archive_url(market, interval, month)
    try:
        payload = fetch_bytes(url, attempts=5, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            missing.touch()
            return None
        raise
    temporary = file.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(file)
    return file, False


def predownload_archives(
    args: argparse.Namespace,
    markets: list[Market],
    scale: Scale,
    months: list[str],
    archive_stats: ArchiveStats,
) -> None:
    available_months = (
        load_archive_manifests(args, markets, scale)
        if len(months) > 2
        else None
    )
    jobs: list[tuple[str, Path, Path]] = []
    for market_index, market in enumerate(markets):
        for month in months:
            file, missing = archive_paths(
                args.data_dir,
                market,
                scale.id,
                month,
            )
            if file.exists() or missing.exists():
                continue
            if (
                available_months is not None
                and month not in available_months[market_index]
            ):
                missing.parent.mkdir(parents=True, exist_ok=True)
                missing.touch()
                continue
            file.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((archive_url(market, scale.id, month), file, missing))
    if not jobs:
        return
    print(
        f"{scale.id}: multiplex-downloading/probing {len(jobs):,} uncached archives...",
        flush=True,
    )
    parallel_download_jobs(
        jobs,
        label=f"{scale.id} archives",
        workers=args.download_workers,
        stats=archive_stats,
    )


def parallel_download_jobs(
    jobs: list[tuple[str, Path, Path]],
    *,
    label: str,
    workers: int,
    stats: ArchiveStats,
) -> None:
    batch_size = 250
    for offset in range(0, len(jobs), batch_size):
        batch = jobs[offset : offset + batch_size]
        temporary_by_url: dict[str, Path] = {}
        config_lines: list[str] = []
        for job_index, (url, file, _) in enumerate(batch):
            temporary = file.with_suffix(
                f".zip.{os.getpid()}.{offset + job_index}.download",
            )
            temporary_by_url[url] = temporary
            config_lines.extend(
                [
                    f'url = "{curl_config_escape(url)}"',
                    f'output = "{curl_config_escape(str(temporary))}"',
                ],
            )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"binance-{re.sub(r'[^a-zA-Z0-9]+', '-', label)}-",
            suffix=".curl",
            dir="/tmp",
            delete=False,
        ) as config:
            config.write("\n".join(config_lines))
            config_file = Path(config.name)
        try:
            process = subprocess.run(
                [
                    "curl",
                    "--parallel",
                    "--parallel-immediate",
                    "--parallel-max",
                    str(min(workers, 32)),
                    "--silent",
                    "--location",
                    "--fail",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "180",
                    "--write-out",
                    "%{http_code}\t%{url_effective}\\n",
                    "--config",
                    str(config_file),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            config_file.unlink(missing_ok=True)
        statuses: dict[str, int] = {}
        for line in process.stdout.splitlines():
            code, separator, url = line.partition("\t")
            if separator and code.isdigit():
                statuses[url] = int(code)
        failures: list[str] = []
        for url, file, missing in batch:
            temporary = temporary_by_url[url]
            status = statuses.get(url, 0)
            if status == 200 and temporary.exists() and temporary.stat().st_size > 0:
                temporary.replace(file)
                stats.found += 1
                stats.downloaded_bytes += file.stat().st_size
            elif status == 404:
                temporary.unlink(missing_ok=True)
                missing.touch()
            else:
                temporary.unlink(missing_ok=True)
                failures.append(f"{status} {url}")
        if failures:
            details = "\n".join(failures[:10])
            raise RuntimeError(
                f"{label}: {len(failures)} archive transfers failed after "
                f"retries:\n{details}\n{process.stderr[-1_000:]}",
            )
        completed = min(len(jobs), offset + len(batch))
        print(
            f"  {label}: {completed:,}/{len(jobs):,}",
            flush=True,
        )


def load_archive_manifests(
    args: argparse.Namespace,
    markets: list[Market],
    scale: Scale,
) -> list[set[str]]:
    results: list[set[str] | None] = [None] * len(markets)
    pending: list[int] = []
    for index, market in enumerate(markets):
        file, _ = archive_paths(args.data_dir, market, scale.id, "placeholder")
        manifest = file.parent / "archive-list.json"
        if manifest.exists():
            results[index] = set(json.loads(manifest.read_text())["months"])
        else:
            pending.append(index)
    if pending:
        print(
            f"{scale.id}: listing archive months for {len(pending):,} markets "
            "to avoid blind 404 probes...",
            flush=True,
        )
        completed = 0
        completed_lock = threading.Lock()

        def load(index: int) -> None:
            nonlocal completed
            market = markets[index]
            prefix = (
                f"data/{market.archive_prefix}/monthly/klines/"
                f"{market.symbol}/{scale.id}/"
            )
            url = f"{S3_ROOT}?{urllib.parse.urlencode({'prefix': prefix, 'max-keys': '1000'}, safe='/')}"
            document = ET.fromstring(fetch_bytes(url, timeout=90))
            namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            pattern = re.compile(
                rf"^{re.escape(market.symbol)}-{re.escape(scale.id)}-"
                r"(\d{4}-\d{2})\.zip$",
            )
            months: set[str] = set()
            for element in document.findall("s3:Contents/s3:Key", namespace):
                name = (element.text or "").split("/")[-1]
                match = pattern.match(name)
                if match:
                    months.add(match.group(1))
            results[index] = months
            file, _ = archive_paths(
                args.data_dir,
                market,
                scale.id,
                "placeholder",
            )
            write_json_atomic(
                file.parent / "archive-list.json",
                {
                    "generatedAt": iso_time(int(time.time() * 1_000)),
                    "months": sorted(months),
                },
            )
            with completed_lock:
                completed += 1
                if completed % 100 == 0 or completed == len(pending):
                    print(
                        f"  {scale.id} manifests: {completed:,}/{len(pending):,}",
                        flush=True,
                    )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.download_workers, 24),
        ) as executor:
            futures = [executor.submit(load, index) for index in pending]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    return [value if value is not None else set() for value in results]


def archive_paths(
    data_dir: Path,
    market: Market,
    interval: str,
    month: str,
) -> tuple[Path, Path]:
    encoded = base64.urlsafe_b64encode(market.symbol.encode()).decode().rstrip("=")
    cache = (
        data_dir
        / "portfolio-basis"
        / "archive-cache"
        / market.venue
        / encoded
        / interval
    )
    return cache / f"{month}.zip", cache / f"{month}.missing"


def archive_url(market: Market, interval: str, month: str) -> str:
    quoted_symbol = urllib.parse.quote(market.symbol, safe="")
    filename = urllib.parse.quote(
        f"{market.symbol}-{interval}-{month}.zip",
        safe="",
    )
    return (
        f"{VISION_ROOT}/{market.archive_prefix}/monthly/klines/"
        f"{quoted_symbol}/{interval}/{filename}"
    )


def curl_config_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def prepare_funding_data(
    args: argparse.Namespace,
    root: Path,
    markets: list[Market],
) -> FundingData:
    funding_root = root / "funding" / f"{args.start}_{args.end}"
    metadata_file = funding_root / "metadata.json"
    output_file = funding_root / "funding-rates.npz"
    if metadata_file.exists() and output_file.exists():
        metadata = json.loads(metadata_file.read_text())
        if metadata.get("complete") is True:
            payload = np.load(output_file)
            return FundingData(
                minute_indexes=payload["minute_indexes"],
                rates=payload["rates"],
                stats=ArchiveStats(**metadata["stats"]),
                file=output_file,
            )
    if funding_root.exists():
        shutil.rmtree(funding_root)
    funding_root.mkdir(parents=True)

    months = calendar_months(args.start_ms, args.finish_ms - MINUTE_MS)
    futures = [
        (index, market)
        for index, market in enumerate(markets)
        if market.venue != "spot"
    ]
    stats = ArchiveStats()
    jobs: list[tuple[str, Path, Path]] = []
    for _, market in futures:
        for month in months:
            file, missing = funding_archive_paths(
                args.data_dir,
                market,
                month,
            )
            if file.exists():
                stats.cache_hits += 1
                continue
            if missing.exists():
                stats.missing += 1
                continue
            file.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((funding_archive_url(market, month), file, missing))
    if jobs:
        print(
            f"Funding: multiplex-downloading/probing {len(jobs):,} archives...",
            flush=True,
        )
        parallel_download_jobs(
            jobs,
            label="funding archives",
            workers=args.download_workers,
            stats=stats,
        )

    minute_parts: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    rate_parts: list[np.ndarray] = []
    parsed_files = 0
    stats.missing = 0
    for market_index, market in futures:
        for month in months:
            stats.requested += 1
            file, missing = funding_archive_paths(
                args.data_dir,
                market,
                month,
            )
            if not file.exists():
                if not missing.exists():
                    raise RuntimeError(f"Funding archive state is missing for {file}")
                stats.missing += 1
                continue
            rows = parse_funding_archive(file)
            if rows.size == 0:
                continue
            timestamps = normalize_timestamps(rows[:, 0])
            minute_indexes = (timestamps // MINUTE_MS * MINUTE_MS - args.start_ms) // MINUTE_MS
            valid = (
                (minute_indexes >= 0)
                & (minute_indexes < (args.finish_ms - args.start_ms) // MINUTE_MS)
                & np.isfinite(rows[:, 2])
            )
            if not np.any(valid):
                continue
            selected_minutes = minute_indexes[valid].astype(np.int32)
            minute_parts.append(selected_minutes)
            market_parts.append(
                np.full(selected_minutes.shape, market_index, dtype=np.int16),
            )
            rate_parts.append(rows[valid, 2].astype(np.float32))
            stats.parsed_candles += int(valid.sum())
            parsed_files += 1
    if minute_parts:
        flat_minutes = np.concatenate(minute_parts)
        flat_markets = np.concatenate(market_parts)
        flat_rates = np.concatenate(rate_parts)
        unique_minutes = np.unique(flat_minutes)
        rates = np.zeros((len(unique_minutes), len(markets)), dtype=np.float32)
        rows = np.searchsorted(unique_minutes, flat_minutes)
        rates[rows, flat_markets] = flat_rates
    else:
        unique_minutes = np.empty(0, dtype=np.int32)
        rates = np.empty((0, len(markets)), dtype=np.float32)
    np.savez_compressed(
        output_file,
        minute_indexes=unique_minutes,
        rates=rates,
    )
    write_json_atomic(
        metadata_file,
        {
            "complete": True,
            "version": REPORT_VERSION,
            "file": str(output_file),
            "fundingEvents": len(unique_minutes),
            "nonzeroRates": int(np.count_nonzero(rates)),
            "parsedFiles": parsed_files,
            "stats": asdict(stats),
        },
    )
    print(
        f"Funding: {len(unique_minutes):,} settlement timestamps and "
        f"{np.count_nonzero(rates):,} market cashflows prepared.",
        flush=True,
    )
    return FundingData(
        minute_indexes=unique_minutes,
        rates=rates,
        stats=stats,
        file=output_file,
    )


def funding_archive_paths(
    data_dir: Path,
    market: Market,
    month: str,
) -> tuple[Path, Path]:
    encoded = base64.urlsafe_b64encode(market.symbol.encode()).decode().rstrip("=")
    root = (
        data_dir
        / "portfolio-basis"
        / "archive-cache"
        / market.venue
        / encoded
        / "fundingRate"
    )
    return root / f"{month}.zip", root / f"{month}.missing"


def funding_archive_url(market: Market, month: str) -> str:
    quoted_symbol = urllib.parse.quote(market.symbol, safe="")
    filename = urllib.parse.quote(
        f"{market.symbol}-fundingRate-{month}.zip",
        safe="",
    )
    return (
        f"{VISION_ROOT}/{market.archive_prefix}/monthly/fundingRate/"
        f"{quoted_symbol}/{filename}"
    )


def parse_funding_archive(file: Path) -> np.ndarray:
    with zipfile.ZipFile(file) as archive:
        names = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and name.endswith(".csv")
        ]
        if len(names) != 1:
            raise RuntimeError(f"{file}: expected exactly one funding CSV member")
        data = archive.read(names[0])
    values = np.loadtxt(
        io.BytesIO(data),
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
        ndmin=2,
    )
    if values.shape[1] < 3:
        raise RuntimeError(f"{file}: malformed funding-rate columns")
    return values


def parse_archive_csv(file: Path) -> np.ndarray:
    with zipfile.ZipFile(file) as archive:
        names = [
            name for name in archive.namelist() if not name.endswith("/") and name.endswith(".csv")
        ]
        if len(names) != 1:
            raise RuntimeError(f"{file}: expected exactly one CSV member")
        data = archive.read(names[0]).strip().replace(b"\r", b"")
    if not data:
        return np.empty((0, 12), dtype=np.float64)
    if not data[:1].isdigit():
        newline = data.find(b"\n")
        data = data[newline + 1 :] if newline >= 0 else b""
    if not data:
        return np.empty((0, 12), dtype=np.float64)
    first_newline = data.find(b"\n")
    first_line = data if first_newline < 0 else data[:first_newline]
    columns = first_line.count(b",") + 1
    flattened = np.fromstring(data.replace(b"\n", b","), sep=",")
    if flattened.size % columns != 0:
        raise RuntimeError(f"{file}: malformed CSV shape")
    return flattened.reshape((-1, columns))


def normalize_timestamps(values: np.ndarray) -> np.ndarray:
    timestamps = values.astype(np.int64)
    microseconds = timestamps >= 100_000_000_000_000
    timestamps[microseconds] //= 1_000
    return timestamps


def build_scale_sleeve(
    args: argparse.Namespace,
    root: Path,
    markets: list[Market],
    assets: list[str],
    asset_markets: list[list[int]],
    prepared: PreparedScale,
) -> SleeveResult:
    signature = hashlib.sha256(
        json.dumps(
            {
                "version": REPORT_VERSION,
                "marketData": str(prepared.root),
                "assets": assets,
                "parameters": selection_parameters(args),
            },
            sort_keys=True,
        ).encode(),
    ).hexdigest()[:16]
    sleeve_root = root / "sleeves" / (
        f"{args.start}_{args.end}-{prepared.scale.id}-{signature}"
    )
    metadata_file = sleeve_root / "metadata.json"
    progress_file = sleeve_root / "progress.json"
    checkpoint_counts_file = sleeve_root / "checkpoint-selected-market-counts.npy"
    if metadata_file.exists() and not args.refresh_sleeves:
        metadata = json.loads(metadata_file.read_text())
        if metadata.get("complete") is True:
            print(f"{prepared.scale.id}: reusing point-in-time sleeve.", flush=True)
            counts = np.load(sleeve_root / "selected-market-counts.npy")
            return SleeveResult(
                scale=prepared.scale,
                events=int(metadata["events"]),
                weights_file=Path(metadata["weightsFile"]),
                stats_file=Path(metadata["statsFile"]),
                mean_basis_size=float(metadata["meanBasisSize"]),
                mean_eligible_assets=float(metadata["meanEligibleAssets"]),
                target_reached_ratio=float(metadata["targetReachedRatio"]),
                mean_exposure=float(metadata["meanExposure"]),
                selected_market_counts=counts,
            )
    resume = progress_file.exists() and not args.refresh_sleeves
    if sleeve_root.exists() and not resume:
        shutil.rmtree(sleeve_root)
    sleeve_root.mkdir(parents=True, exist_ok=True)

    interval_ms = prepared.scale.minutes * MINUTE_MS
    event_times = np.arange(
        args.start_ms,
        args.finish_ms,
        interval_ms,
        dtype=np.int64,
    )
    last_indexes = ((event_times - interval_ms - prepared.grid_start) // interval_ms).astype(
        np.int64,
    )
    events = len(event_times)
    returns = np.memmap(
        prepared.returns_file,
        dtype=np.float32,
        mode="r",
        shape=(prepared.market_count, prepared.grid_count),
    )
    eligible = np.memmap(
        prepared.eligible_file,
        dtype=np.uint8,
        mode="r",
        shape=(prepared.market_count, prepared.grid_count),
    )
    liquidity = np.memmap(
        prepared.liquidity_file,
        dtype=np.float32,
        mode="r",
        shape=(prepared.market_count, prepared.grid_count),
    )
    route_file = sleeve_root / "point-in-time-route-asset-major.i16"
    routes = np.memmap(
        route_file,
        dtype=np.int16,
        mode="r+" if resume else "w+",
        shape=(len(assets), events),
    )
    if not resume:
        routes[:] = -1
        print(
            f"{prepared.scale.id}: resolving point-in-time canonical venue for "
            f"{len(assets):,} assets...",
            flush=True,
        )
        for asset_index, candidates in enumerate(asset_markets):
            route = np.full(events, -1, dtype=np.int16)
            for market_index in candidates:
                use = (route < 0) & (eligible[market_index, last_indexes] > 0)
                route[use] = market_index
            routes[asset_index] = route
        routes.flush()

    weights_file = sleeve_root / "weights-time-major.f32"
    sleeve_weights = np.memmap(
        weights_file,
        dtype=np.float32,
        mode="r+" if resume else "w+",
        shape=(events, len(markets)),
    )
    working_arrays = {
        "basis_sizes": (np.int16, sleeve_root / "basis-sizes.i16"),
        "eligible_counts": (np.int16, sleeve_root / "eligible-counts.i16"),
        "median_r2": (np.float32, sleeve_root / "median-r2.f32"),
        "p10_r2": (np.float32, sleeve_root / "p10-r2.f32"),
        "target_reached": (np.uint8, sleeve_root / "target-reached.u8"),
        "exposures": (np.float32, sleeve_root / "exposures.f32"),
    }
    working = {
        name: np.memmap(
            file,
            dtype=dtype,
            mode="r+" if resume else "w+",
            shape=(events,),
        )
        for name, (dtype, file) in working_arrays.items()
    }
    basis_sizes = working["basis_sizes"]
    eligible_counts = working["eligible_counts"]
    median_r2 = working["median_r2"]
    p10_r2 = working["p10_r2"]
    target_reached = working["target_reached"]
    exposures = working["exposures"]
    if resume:
        progress = json.loads(progress_file.read_text())
        completed_events = int(progress["completedEvents"])
        if (
            int(progress.get("events", -1)) != events
            or completed_events < 0
            or completed_events > events
            or int(progress.get("batchSize", -1)) != args.batch_size
            or (
                completed_events != events
                and completed_events % args.batch_size != 0
            )
        ):
            raise RuntimeError(
                f"{prepared.scale.id}: invalid sleeve checkpoint "
                f"{progress_file}",
            )
        selected_market_counts = np.load(checkpoint_counts_file)
        print(
            f"{prepared.scale.id}: resuming point-in-time sleeve at "
            f"{completed_events:,}/{events:,}.",
            flush=True,
        )
    else:
        completed_events = 0
        for values in working.values():
            values[:] = 0
        selected_market_counts = np.zeros(len(markets), dtype=np.int64)
    anchor_index = assets.index("BTC") if "BTC" in assets else None
    offsets = np.arange(SAMPLE_COUNT - 1, -1, -1, dtype=np.int64)
    batches = math.ceil(events / args.batch_size)
    started = time.monotonic()
    print(
        f"{prepared.scale.id}: evaluating {events:,} independent point-in-time "
        f"bases in {batches:,} GPU batches...",
        flush=True,
    )
    first_batch = completed_events // args.batch_size + 1
    for batch_number, start in enumerate(
        range(completed_events, events, args.batch_size),
        first_batch,
    ):
        end = min(events, start + args.batch_size)
        all_batch_routes = np.asarray(routes[:, start:end].T, dtype=np.int64)
        all_active = all_batch_routes >= 0
        # Consecutive timestamps normally share the same eligible universe.
        # Compact to its union before forming the batched Gram matrices; rows
        # absent from every timestamp are mathematically inert but expensive.
        # np.flatnonzero preserves global asset order, including tie-breaking.
        batch_asset_indexes = np.flatnonzero(all_active.any(axis=0))
        batch_routes = all_batch_routes[:, batch_asset_indexes]
        active = all_active[:, batch_asset_indexes]
        safe_routes = np.maximum(batch_routes, 0)
        batch_last = last_indexes[start:end]
        windows = returns[
            safe_routes[:, :, None],
            batch_last[:, None, None] - offsets[None, None, :],
        ]
        windows[~active] = 0
        if anchor_index is None:
            batch_anchor_index = None
        else:
            anchor_location = np.searchsorted(batch_asset_indexes, anchor_index)
            batch_anchor_index = (
                int(anchor_location)
                if (
                    anchor_location < len(batch_asset_indexes)
                    and batch_asset_indexes[anchor_location] == anchor_index
                )
                else None
            )
        selection = select_basis_batch(
            windows,
            active,
            anchor_index=batch_anchor_index,
            min_size=args.min_basis_size,
            max_size=args.max_basis_size,
            target_median_r_squared=args.target_median_r2,
            target_p10_r_squared=args.target_p10_r2,
            residual_equivalence_band=args.residual_equivalence_band,
            device=args.device,
        )
        width = selection.selected.shape[1]
        selected_compact_assets = np.maximum(
            selection.selected.astype(np.int64),
            0,
        )
        selected_assets = batch_asset_indexes[selected_compact_assets]
        selected_markets = np.take_along_axis(
            all_batch_routes,
            selected_assets,
            axis=1,
        )
        rank_active = (
            np.arange(width)[None, :] < selection.sizes[:, None]
        ) & (selection.sizes[:, None] >= math.ceil(1 / args.maximum_weight))
        selected_markets = np.where(rank_active, selected_markets, 0)
        all_safe_routes = np.maximum(all_batch_routes, 0)
        route_liquidity = liquidity[
            all_safe_routes,
            batch_last[:, None],
        ]
        selected_liquidity = np.take_along_axis(
            route_liquidity,
            selected_assets,
            axis=1,
        ).astype(np.float64)
        selected_liquidity = np.where(
            rank_active,
            np.maximum(selected_liquidity, 0),
            0,
        )
        selected_weights = capped_proportional_weights_batch(
            selected_liquidity,
            rank_active,
            args.maximum_weight,
        )
        target = np.zeros((end - start, len(markets)), dtype=np.float32)
        rows = np.broadcast_to(
            np.arange(end - start)[:, None],
            selected_markets.shape,
        )
        np.add.at(
            target,
            (rows[rank_active], selected_markets[rank_active]),
            selected_weights[rank_active],
        )
        sleeve_weights[start:end] = target
        basis_sizes[start:end] = selection.sizes
        eligible_counts[start:end] = selection.eligible_counts
        median_r2[start:end] = selection.median_r_squared
        p10_r2[start:end] = selection.p10_r_squared
        target_reached[start:end] = selection.target_reached
        exposures[start:end] = target.sum(axis=1)
        np.add.at(
            selected_market_counts,
            selected_markets[rank_active],
            1,
        )
        if batch_number % 20 == 0 or batch_number == batches:
            sleeve_weights.flush()
            for values in working.values():
                values.flush()
            write_npy_atomic(
                checkpoint_counts_file,
                selected_market_counts,
            )
            write_json_atomic(
                progress_file,
                {
                    "complete": False,
                    "version": REPORT_VERSION,
                    "events": events,
                    "completedEvents": end,
                    "batchSize": args.batch_size,
                },
            )
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = (end - completed_events) / elapsed
            eta = (events - end) / max(rate, 1e-9)
            print(
                f"  {prepared.scale.id}: {end:,}/{events:,}; "
                f"{rate:,.0f} timestamps/s; ETA {duration(eta)}",
                flush=True,
            )
    sleeve_weights.flush()
    stats_file = sleeve_root / "selection-stats.npz"
    np.savez_compressed(
        stats_file,
        event_times=event_times,
        basis_sizes=basis_sizes,
        eligible_counts=eligible_counts,
        median_r_squared=median_r2,
        p10_r_squared=p10_r2,
        target_reached=target_reached,
        exposures=exposures,
    )
    np.save(sleeve_root / "selected-market-counts.npy", selected_market_counts)
    metadata = {
        "complete": True,
        "version": REPORT_VERSION,
        "scale": asdict(prepared.scale),
        "events": events,
        "weightsFile": str(weights_file),
        "statsFile": str(stats_file),
        "routeFile": str(route_file),
        "meanBasisSize": float(basis_sizes.mean()),
        "meanEligibleAssets": float(eligible_counts.mean()),
        "targetReachedRatio": float(target_reached.mean()),
        "meanExposure": float(exposures.mean()),
        "parameters": selection_parameters(args),
    }
    write_json_atomic(metadata_file, metadata)
    progress_file.unlink(missing_ok=True)
    checkpoint_counts_file.unlink(missing_ok=True)
    return SleeveResult(
        scale=prepared.scale,
        events=events,
        weights_file=weights_file,
        stats_file=stats_file,
        mean_basis_size=float(basis_sizes.mean()),
        mean_eligible_assets=float(eligible_counts.mean()),
        target_reached_ratio=float(target_reached.mean()),
        mean_exposure=float(exposures.mean()),
        selected_market_counts=selected_market_counts,
    )


def selection_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sampleCount": SAMPLE_COUNT,
        "minBasisSize": args.min_basis_size,
        "maxBasisSize": args.max_basis_size,
        "targetMedianRSquared": args.target_median_r2,
        "targetP10RSquared": args.target_p10_r2,
        "residualEquivalenceBand": args.residual_equivalence_band,
        "maximumConstituentWeight": args.maximum_weight,
        "correlation": "Pearson",
        "amplitudePriority": (
            "mean-absolute-native-candle-log-return-before-vector-normalization"
        ),
        "sizeMeasure": "trailing-360-candle-quote-notional",
    }


def simulate_index(
    args: argparse.Namespace,
    root: Path,
    markets: list[Market],
    catalog: dict[str, Any],
    archive_stats: ArchiveStats,
    prepared: dict[str, PreparedScale],
    sleeves: dict[str, SleeveResult],
    funding: FundingData,
) -> tuple[dict[str, Any], dict[str, Path]]:
    minute = prepared["1m"]
    close_returns = np.memmap(
        minute.ohlc_files["close"],
        dtype=np.float32,
        mode="r",
        shape=(minute.grid_count, len(markets)),
    )
    ohlc = {
        field: np.memmap(
            minute.ohlc_files[field],
            dtype=np.float32,
            mode="r",
            shape=(minute.grid_count, len(markets)),
        )
        for field in ("open", "high", "low")
    }
    target_matrices = {
        scale.id: np.memmap(
            sleeves[scale.id].weights_file,
            dtype=np.float32,
            mode="r",
            shape=(sleeves[scale.id].events, len(markets)),
        )
        for scale in SCALES
    }
    scenario_definitions = (
        ("gross", 0.0, False),
        ("feeOnly", 0.0, True),
        ("baseline", args.baseline_slippage_bps, True),
        ("conservative", args.conservative_slippage_bps, True),
    )
    base_fee = np.array(
        [
            (
                args.spot_fee_bps
                if market.venue == "spot"
                else args.futures_fee_bps
            )
            / 10_000
            for market in markets
        ],
        dtype=np.float64,
    )
    cost_rates = {
        name: (
            base_fee + slippage_bps / 10_000
            if include_fee
            else np.zeros(len(markets), dtype=np.float64)
        )
        for name, slippage_bps, include_fee in scenario_definitions
    }
    scenario_names = [definition[0] for definition in scenario_definitions]
    levels = {name: INITIAL_INDEX_LEVEL for name in scenario_names}
    peaks = dict(levels)
    drawdowns = {name: 0.0 for name in scenario_names}
    cumulative_cost = {name: 0.0 for name in scenario_names}
    cumulative_funding = {name: 0.0 for name in scenario_names}
    cumulative_traded = {name: 0.0 for name in scenario_names}
    minute_returns = {
        name: np.zeros(
            (args.finish_ms - args.start_ms) // MINUTE_MS,
            dtype=np.float64,
        )
        for name in scenario_names
    }
    pretrade_weights = np.zeros(len(markets), dtype=np.float64)
    current_sleeves = {
        scale.id: np.zeros(len(markets), dtype=np.float64) for scale in SCALES
    }
    minute_count = len(minute_returns["gross"])
    exposures = np.zeros(minute_count, dtype=np.float32)
    active_counts = np.zeros(minute_count, dtype=np.int16)
    gross_turnovers = np.zeros(minute_count, dtype=np.float32)
    baseline_costs = np.zeros(minute_count, dtype=np.float64)
    baseline_funding = np.zeros(minute_count, dtype=np.float64)
    missing_held_weights = np.zeros(minute_count, dtype=np.float32)

    output_root = args.data_dir / "portfolio-basis" / "index-history"
    output_root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.start}_{args.end}-multiscale-point-in-time-"
        "liqcap5-long-only-friction-1m"
    )
    candles = output_root / f"{stem}.candles.csv.gz"
    temporary = candles.with_suffix(f".csv.gz.{os.getpid()}.tmp")
    print(
        f"Simulating {minute_count:,} long-only minute rebalances with friction...",
        flush=True,
    )
    started = time.monotonic()
    funding_cursor = 0
    with gzip.open(temporary, "wt", newline="", compresslevel=6) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "gross_close",
                "fee_only_close",
                "conservative_close",
                "target_exposure",
                "cash_weight",
                "active_constituents",
                "gross_traded_notional_ratio",
                "baseline_transaction_cost",
                "baseline_funding_cashflow",
                "missing_next_return_weight",
            ],
        )
        for minute_index in range(minute_count):
            timestamp = args.start_ms + minute_index * MINUTE_MS
            aggregate_target = np.zeros(len(markets), dtype=np.float64)
            for scale in SCALES:
                if minute_index % scale.minutes == 0:
                    event_index = minute_index // scale.minutes
                    current_sleeves[scale.id] = np.asarray(
                        target_matrices[scale.id][event_index],
                        dtype=np.float64,
                    )
                aggregate_target += scale.sleeve_weight * current_sleeves[scale.id]

            data_index = (timestamp - minute.grid_start) // MINUTE_MS
            prior_close = np.asarray(close_returns[data_index - 1])
            tradable = np.isfinite(prior_close)
            aggregate_target[~tradable] = 0
            target_exposure = float(aggregate_target.sum())
            if target_exposure > 1 + 1e-7 or np.any(aggregate_target < 0):
                raise RuntimeError(
                    f"Invalid long-only target at {iso_time(timestamp)}: "
                    f"{target_exposure}",
                )
            exposures[minute_index] = target_exposure
            active_counts[minute_index] = np.count_nonzero(aggregate_target > 0)

            opening_levels = dict(levels)
            gross_pretrade_weights = pretrade_weights
            net_pretrade_weights = pretrade_weights
            if (
                funding_cursor < len(funding.minute_indexes)
                and funding.minute_indexes[funding_cursor] == minute_index
            ):
                (
                    relative_after_funding,
                    net_pretrade_weights,
                ) = apply_funding_cashflow(
                    pretrade_weights,
                    funding.rates[funding_cursor],
                )
                funding_fraction = 1 - relative_after_funding
                for name in ("feeOnly", "baseline", "conservative"):
                    funding_cashflow = levels[name] * funding_fraction
                    cumulative_funding[name] += funding_cashflow
                    if name == "baseline":
                        baseline_funding[minute_index] = funding_cashflow
                    levels[name] *= relative_after_funding
                funding_cursor += 1

            relative_post_cost: dict[str, float] = {}
            traded_ratios: dict[str, float] = {}
            for name in scenario_names:
                rate = cost_rates[name]
                current_weights = (
                    gross_pretrade_weights
                    if name == "gross"
                    else net_pretrade_weights
                )
                relative = solve_relative_post_cost_value(
                    current_weights,
                    aggregate_target,
                    rate,
                )
                traded = float(
                    np.abs(
                        aggregate_target * relative - current_weights,
                    ).sum(),
                )
                previous_level = levels[name]
                cost = previous_level * (1 - relative)
                cumulative_cost[name] += cost
                cumulative_traded[name] += previous_level * traded
                relative_post_cost[name] = relative
                traded_ratios[name] = traded
            gross_turnovers[minute_index] = traded_ratios["gross"]
            baseline_costs[minute_index] = (
                levels["baseline"] * (1 - relative_post_cost["baseline"])
            )

            candle_returns = {
                field: np.nan_to_num(
                    np.asarray(matrix[data_index], dtype=np.float64),
                )
                for field, matrix in ohlc.items()
            }
            candle_returns["close"] = np.nan_to_num(
                np.asarray(close_returns[data_index], dtype=np.float64),
            )
            missing = ~np.isfinite(np.asarray(close_returns[data_index]))
            missing_held_weights[minute_index] = aggregate_target[missing].sum()
            cash_weight = max(0.0, 1 - target_exposure)
            factors = {
                field: cash_weight
                + float(np.dot(aggregate_target, 1 + values))
                for field, values in candle_returns.items()
            }
            factors["high"] = max(
                factors["high"],
                factors["open"],
                factors["close"],
            )
            factors["low"] = min(
                factors["low"],
                factors["open"],
                factors["close"],
            )
            prior_baseline = levels["baseline"]
            baseline_post = (
                prior_baseline * relative_post_cost["baseline"]
            )
            candle_open = baseline_post * factors["open"]
            candle_high = baseline_post * factors["high"]
            candle_low = baseline_post * factors["low"]
            candle_close = baseline_post * factors["close"]

            for name in scenario_names:
                previous = opening_levels[name]
                levels[name] = (
                    levels[name]
                    * relative_post_cost[name]
                    * factors["close"]
                )
                minute_returns[name][minute_index] = levels[name] / previous - 1
                peaks[name] = max(peaks[name], levels[name])
                drawdowns[name] = max(
                    drawdowns[name],
                    1 - levels[name] / peaks[name],
                )
            close_factor = max(factors["close"], 1e-15)
            pretrade_weights = (
                aggregate_target
                * (1 + candle_returns["close"])
                / close_factor
            )

            writer.writerow(
                [
                    iso_time(timestamp),
                    precise(candle_open),
                    precise(candle_high),
                    precise(candle_low),
                    precise(candle_close),
                    precise(levels["gross"]),
                    precise(levels["feeOnly"]),
                    precise(levels["conservative"]),
                    f"{target_exposure:.12f}",
                    f"{cash_weight:.12f}",
                    int(active_counts[minute_index]),
                    f"{traded_ratios['gross']:.12f}",
                    precise(baseline_costs[minute_index]),
                    precise(baseline_funding[minute_index]),
                    f"{missing_held_weights[minute_index]:.12f}",
                ],
            )
            if (minute_index + 1) % 43_200 == 0 or minute_index + 1 == minute_count:
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = (minute_index + 1) / elapsed
                eta = (minute_count - minute_index - 1) / max(rate, 1e-9)
                print(
                    f"  simulation: {minute_index + 1:,}/{minute_count:,}; "
                    f"net {levels['baseline']:.2f}; ETA {duration(eta)}",
                    flush=True,
                )
    temporary.replace(candles)

    benchmark = btc_benchmark(args, markets, close_returns, minute.grid_start, base_fee)
    performance = {
        name: performance_summary(
            levels[name],
            drawdowns[name],
            minute_returns[name],
            cumulative_cost[name],
            cumulative_funding[name],
            cumulative_traded[name],
        )
        for name in scenario_names
    }
    monthly = build_monthly_path(args, minute_returns)
    scale_reports = []
    total_events = sum(sleeve.events for sleeve in sleeves.values())
    combined_counts = sum(
        (sleeve.selected_market_counts for sleeve in sleeves.values()),
        start=np.zeros(len(markets), dtype=np.int64),
    )
    top_markets = np.argsort(combined_counts)[::-1][:50]
    for scale in SCALES:
        sleeve = sleeves[scale.id]
        scale_reports.append(
            {
                "id": scale.id,
                "label": scale.label,
                "events": sleeve.events,
                "meanEligibleAssets": sleeve.mean_eligible_assets,
                "meanBasisSize": sleeve.mean_basis_size,
                "coverageTargetReachedRatio": sleeve.target_reached_ratio,
                "meanSleeveExposure": sleeve.mean_exposure,
                "sleeveWeight": scale.sleeve_weight,
            },
        )
    report = {
        "version": REPORT_VERSION,
        "generatedAt": iso_time(int(time.time() * 1_000)),
        "methodology": {
            "name": "point-in-time multiscale orthogonal-liquidity index",
            "lookAhead": False,
            "decisionTiming": (
                "At close t, use only candles completed by t; rebuild each scale "
                "on its native close; apply the resulting target over t to t+1."
            ),
            "rebalanceInterval": "1m",
            "rebalancePolicy": (
                "Exact self-financing long-only rebalance every minute. "
                "Post-cost risky weights equal point-in-time targets."
            ),
            "exposureRange": [0, 1],
            "cashPolicy": (
                "Unavailable, not-yet-warm, infeasible, or temporarily "
                "untradable sleeve weight remains cash and is not renormalized."
            ),
            "borrowing": False,
            "shorting": False,
            "leverage": False,
            "borrowingInterestAndMaintenanceCosts": 0,
            "perpetualFunding": (
                "Actual archived USD-M/COIN-M funding rates are applied to "
                "positions held into each settlement timestamp. Positive rates "
                "are paid by the long portfolio; negative rates are received."
            ),
            "selection": selection_parameters(args),
            "scaleComparison": (
                "Each scale is selected independently from exactly 360 native "
                "returns; five independently weighted sleeves contribute 20% each."
            ),
            "pointInTimeUniverse": (
                "Historical Binance Vision archive presence plus a complete "
                "trailing window; Spot preferred, then USD-M, then COIN-M."
            ),
            "optionsTreatment": catalog["optionsNote"],
            "friction": {
                "chargedOn": "absolute dollar buys plus absolute dollar sells",
                "spotFeeBps": args.spot_fee_bps,
                "futuresFeeBps": args.futures_fee_bps,
                "baselineAdditionalExecutionBps": args.baseline_slippage_bps,
                "conservativeAdditionalExecutionBps": (
                    args.conservative_slippage_bps
                ),
                "capacityImpact": (
                    "Not estimated without AUM and historical order-book depth; "
                    "fixed execution bps are a transparent proxy."
                ),
            },
            "ohlcNote": (
                "Open and close use weighted component candle-point returns. "
                "High/low are component-extrema envelopes because extrema are "
                "not synchronous across constituents."
            ),
        },
        "window": {
            "start": args.start,
            "end": args.end,
            "minuteCandles": minute_count,
        },
        "universe": {
            "catalog": catalog,
            "continuousMarkets": len(markets),
            "economicAssets": len({market.asset for market in markets}),
            "markets": {
                venue: sum(market.venue == venue for market in markets)
                for venue in VENUE_RANK
            },
            "mostFrequentlySelectedMarkets": [
                {
                    "market": markets[index].symbol,
                    "asset": markets[index].asset,
                    "venue": markets[index].venue,
                    "selectionEvents": int(combined_counts[index]),
                    "selectionEventShare": (
                        float(combined_counts[index] / total_events)
                        if total_events
                        else 0
                    ),
                }
                for index in top_markets
                if combined_counts[index] > 0
            ],
        },
        "scales": scale_reports,
        "portfolio": {
            "meanExposure": float(exposures.mean()),
            "minimumExposure": float(exposures.min()),
            "maximumExposure": float(exposures.max()),
            "meanActiveConstituents": float(active_counts.mean()),
            "medianGrossTradedNotionalPerMinute": float(
                np.quantile(gross_turnovers, 0.5),
            ),
            "meanGrossTradedNotionalPerMinute": float(gross_turnovers.mean()),
            "summedGrossTurnover": float(gross_turnovers.sum()),
            "meanMissingNextReturnWeight": float(missing_held_weights.mean()),
            "maximumMissingNextReturnWeight": float(missing_held_weights.max()),
        },
        "performance": performance,
        "monthly": monthly,
        "benchmark": benchmark,
        "archive": {
            **asdict(archive_stats),
            "downloadedGigabytes": archive_stats.downloaded_bytes / 1_000_000_000,
            "funding": {
                **asdict(funding.stats),
                "settlementTimestamps": len(funding.minute_indexes),
                "nonzeroMarketRates": int(np.count_nonzero(funding.rates)),
                "file": str(funding.file),
            },
        },
        "artifacts": {
            "minuteCandlesCsvGzip": str(candles),
        },
    }
    json_file = output_root / f"{stem}.json"
    markdown_file = output_root / f"{stem}.md"
    write_json_atomic(json_file, report)
    write_text_atomic(markdown_file, render_report(report))
    shutil.copyfile(candles, output_root / "latest.candles.csv.gz")
    shutil.copyfile(json_file, output_root / "latest.json")
    shutil.copyfile(markdown_file, output_root / "latest.md")
    return report, {
        "candles": candles,
        "json": json_file,
        "markdown": markdown_file,
    }


def solve_relative_post_cost_value(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    cost_rates: np.ndarray,
) -> float:
    relative = 1.0
    for _ in range(8):
        relative = 1 - float(
            np.dot(
                cost_rates,
                np.abs(target_weights * relative - current_weights),
            ),
        )
    return max(0.0, relative)


def apply_funding_cashflow(
    current_weights: np.ndarray,
    funding_rates: np.ndarray,
) -> tuple[float, np.ndarray]:
    funding_fraction = float(np.dot(current_weights, funding_rates))
    relative_value = 1 - funding_fraction
    if relative_value <= 0:
        raise RuntimeError("Funding cashflow exhausted portfolio NAV.")
    return relative_value, current_weights / relative_value


def performance_summary(
    final_level: float,
    maximum_drawdown: float,
    returns: np.ndarray,
    cumulative_cost: float,
    cumulative_funding: float,
    cumulative_traded: float,
) -> dict[str, float]:
    deviation = float(np.std(returns, ddof=1))
    annualization = math.sqrt(365 * 24 * 60)
    return {
        "initialLevel": INITIAL_INDEX_LEVEL,
        "finalLevel": final_level,
        "totalReturn": final_level / INITIAL_INDEX_LEVEL - 1,
        "annualizedVolatility": deviation * annualization,
        "annualizedSharpe": (
            float(returns.mean()) / deviation * annualization
            if deviation > 0
            else 0
        ),
        "maximumDrawdown": maximum_drawdown,
        "cumulativeTransactionCost": cumulative_cost,
        "cumulativeFundingCashflow": cumulative_funding,
        "cumulativeTradedNotional": cumulative_traded,
        "meanMinuteReturn": float(returns.mean()),
        "bestMinute": float(returns.max()),
        "worstMinute": float(returns.min()),
    }


def build_monthly_path(
    args: argparse.Namespace,
    returns: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    levels = {name: INITIAL_INDEX_LEVEL for name in returns}
    rows: list[dict[str, Any]] = []
    cursor = args.start_ms
    while cursor < args.finish_ms:
        date = datetime.fromtimestamp(cursor / 1_000, UTC)
        if date.month == 12:
            next_month = datetime(date.year + 1, 1, 1, tzinfo=UTC)
        else:
            next_month = datetime(date.year, date.month + 1, 1, tzinfo=UTC)
        end = min(args.finish_ms, int(next_month.timestamp() * 1_000))
        start_index = (cursor - args.start_ms) // MINUTE_MS
        end_index = (end - args.start_ms) // MINUTE_MS
        row: dict[str, Any] = {"month": date.strftime("%Y-%m")}
        for name, values in returns.items():
            start_level = levels[name]
            factor = float(
                np.exp(
                    np.log1p(values[start_index:end_index]).sum(
                        dtype=np.float64,
                    ),
                ),
            )
            levels[name] *= factor
            row[name] = {
                "startLevel": start_level,
                "endLevel": levels[name],
                "return": factor - 1,
            }
        rows.append(row)
        cursor = end
    return rows


def btc_benchmark(
    args: argparse.Namespace,
    markets: list[Market],
    close_returns: np.memmap,
    grid_start: int,
    base_fee: np.ndarray,
) -> dict[str, Any]:
    candidate = next(
        (
            index
            for index, market in enumerate(markets)
            if market.venue == "spot" and market.symbol == "BTCUSDT"
        ),
        None,
    )
    if candidate is None:
        return {"available": False}
    fee = base_fee[candidate] + args.baseline_slippage_bps / 10_000
    level = INITIAL_INDEX_LEVEL / (1 + fee)
    peak = level
    maximum_drawdown = 0.0
    returns: list[float] = []
    count = (args.finish_ms - args.start_ms) // MINUTE_MS
    for minute_index in range(count):
        timestamp = args.start_ms + minute_index * MINUTE_MS
        data_index = (timestamp - grid_start) // MINUTE_MS
        value = float(close_returns[data_index, candidate])
        value = value if math.isfinite(value) else 0
        previous = level
        level *= 1 + value
        returns.append(level / previous - 1)
        peak = max(peak, level)
        maximum_drawdown = max(maximum_drawdown, 1 - level / peak)
    values = np.asarray(returns)
    return {
        "available": True,
        "asset": "BTC",
        "market": "BTCUSDT",
        "policy": "buy-and-hold; baseline initial purchase friction",
        **performance_summary(
            level,
            maximum_drawdown,
            values,
            INITIAL_INDEX_LEVEL - INITIAL_INDEX_LEVEL / (1 + fee),
            0.0,
            INITIAL_INDEX_LEVEL / (1 + fee),
        ),
    }


def render_report(report: dict[str, Any]) -> str:
    baseline = report["performance"]["baseline"]
    gross = report["performance"]["gross"]
    fee_only = report["performance"]["feeOnly"]
    conservative = report["performance"]["conservative"]
    lines = [
        "# Binance multiscale point-in-time index",
        "",
        f"Generated {report['generatedAt']}.",
        "",
        "## Result",
        "",
        f"- Window: {report['window']['start']} through {report['window']['end']}",
        f"- Minute candles: {report['window']['minuteCandles']:,}",
        (
            f"- Baseline net index: {baseline['initialLevel']:.2f} → "
            f"{baseline['finalLevel']:.2f} "
            f"({signed_percent(baseline['totalReturn'])})"
        ),
        (
            f"- Gross index: {gross['initialLevel']:.2f} → "
            f"{gross['finalLevel']:.2f} "
            f"({signed_percent(gross['totalReturn'])})"
        ),
        (
            f"- Fee-only index: {fee_only['finalLevel']:.2f} "
            f"({signed_percent(fee_only['totalReturn'])})"
        ),
        (
            f"- Conservative index: {conservative['finalLevel']:.2f} "
            f"({signed_percent(conservative['totalReturn'])})"
        ),
        f"- Baseline maximum drawdown: {percent(baseline['maximumDrawdown'])}",
        (
            "- Baseline cumulative transaction cost: "
            f"{baseline['cumulativeTransactionCost']:.2f} index points"
        ),
        (
            "- Baseline cumulative perpetual-funding cashflow: "
            f"{baseline['cumulativeFundingCashflow']:.2f} index points "
            "(positive is a cost, negative is income)"
        ),
        "",
        "## Point-in-time construction",
        "",
        (
            "At each close, only already-completed candles are used. The 1m "
            "sleeve is rebuilt every minute; 15m, 1h, 4h, and 1d sleeves are "
            "rebuilt on their native closes. New targets are held only over "
            "the following minute."
        ),
        "",
        (
            "Each scale independently selects a coverage-driven pivoted-QR "
            "basis from exactly 360 returns. Mean absolute return breaks only "
            "the 5% residual-equivalence band, and trailing quote notional "
            "provides capped proportional weights. Each scale contributes a "
            "20% sleeve."
        ),
        "",
        "## Friction and exposure",
        "",
        (
            f"- Spot: {report['methodology']['friction']['spotFeeBps']:.1f} bp "
            f"fee + {report['methodology']['friction']['baselineAdditionalExecutionBps']:.1f} "
            "bp baseline execution loss per dollar bought or sold"
        ),
        (
            f"- Futures: {report['methodology']['friction']['futuresFeeBps']:.1f} "
            "bp fee + the same baseline execution loss"
        ),
        (
            "- Exposure: long-only 0–1; no borrowing, shorting, leverage, "
            "interest, or maintenance cost"
        ),
        "- USD-M/COIN-M: actual archived funding cashflows at their settlement timestamps",
        f"- Mean exposure: {percent(report['portfolio']['meanExposure'])}",
        (
            "- Mean gross traded notional per minute: "
            f"{percent(report['portfolio']['meanGrossTradedNotionalPerMinute'])}"
        ),
        (
            "- Sum of per-minute gross turnover ratios: "
            f"{report['portfolio']['summedGrossTurnover']:.1f}×"
        ),
        "",
        "The fixed execution-bp model does not estimate capacity impact; AUM and historical order-book depth are required for that.",
        "",
        "## Scale diagnostics",
        "",
        "| Scale | Evaluations | Mean eligible | Mean basis | Coverage target | Mean sleeve exposure |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scale in report["scales"]:
        lines.append(
            f"| {scale['label']} | {scale['events']:,} | "
            f"{scale['meanEligibleAssets']:.1f} | {scale['meanBasisSize']:.1f} | "
            f"{percent(scale['coverageTargetReachedRatio'])} | "
            f"{percent(scale['meanSleeveExposure'])} |"
        )
    lines.extend(
        [
            "",
            "## Monthly path",
            "",
            "| Month | Baseline start | Baseline end | Baseline return | Gross return | Fee-only return | Conservative return |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for month in report["monthly"]:
        lines.append(
            f"| {month['month']} | {month['baseline']['startLevel']:.2f} | "
            f"{month['baseline']['endLevel']:.2f} | "
            f"{signed_percent(month['baseline']['return'])} | "
            f"{signed_percent(month['gross']['return'])} | "
            f"{signed_percent(month['feeOnly']['return'])} | "
            f"{signed_percent(month['conservative']['return'])} |"
        )
    lines.extend(
        [
            "",
            "## Universe",
            "",
            (
                f"- Full catalog: "
                f"{report['universe']['catalog']['allProductRows']:,} product "
                "rows across Spot, USD-M, COIN-M, and Options"
            ),
            (
                f"- Continuous markets: {report['universe']['continuousMarkets']:,} "
                f"across {report['universe']['economicAssets']:,} economic assets"
            ),
            (
                "- Markets: "
                + ", ".join(
                    f"{venue} {count:,}"
                    for venue, count in report["universe"]["markets"].items()
                )
            ),
            f"- Current option listings catalogued: {report['universe']['catalog']['currentOptionListings']:,}",
            (
                "- Full product-row list: "
                f"`{report['universe']['catalog']['fullListingsFile']}`"
            ),
            "",
            report["methodology"]["optionsTreatment"],
            "",
            "## Artifacts",
            "",
            f"- Minute candles: `{report['artifacts']['minuteCandlesCsvGzip']}`",
            "",
        ],
    )
    return "\n".join(lines)


def fetch_bytes(
    url: str,
    *,
    attempts: int = 5,
    timeout: int = 60,
) -> bytes:
    failure: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "trading-point-in-time-index/2.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise
            failure = error
        except Exception as error:
            failure = error
        if attempt < attempts:
            time.sleep(0.5 * 2 ** (attempt - 1))
    assert failure is not None
    raise failure


def calendar_months(start_ms: int, end_ms: int) -> list[str]:
    start = datetime.fromtimestamp(start_ms / 1_000, UTC)
    end = datetime.fromtimestamp(end_ms / 1_000, UTC)
    year, month = start.year, start.month
    values: list[str] = []
    while (year, month) <= (end.year, end.month):
        values.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return values


def parse_day(value: str) -> int:
    parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    if parsed.strftime("%Y-%m-%d") != value:
        raise ValueError(f"Invalid UTC day: {value}")
    return int(parsed.timestamp() * 1_000)


def write_json_atomic(file: Path, value: Any) -> None:
    write_text_atomic(file, json.dumps(value, indent=2, sort_keys=False) + "\n")


def write_npy_atomic(file: Path, value: np.ndarray) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(f"{file.suffix}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    temporary.replace(file)


def write_text_atomic(file: Path, content: str) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(f"{file.suffix}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(file)


def iso_time(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp / 1_000, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def precise(value: float) -> str:
    return f"{value:.15g}"


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def signed_percent(value: float) -> str:
    return f"{value * 100:+.1f}%"


def duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
