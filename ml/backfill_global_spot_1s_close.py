from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from global_feature_registry import ROOT


FAST_MANIFEST = ROOT / "data/runtime-cache/binance-cross-asset-1m-basis-30d/fast-manifest.json"
DEFAULT_OUTPUT = ROOT / "data/runtime-cache/binance-cross-asset-spot-1s-close-30d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill a compact verified 1s spot-close cache.")
    parser.add_argument("--manifest", type=Path, default=FAST_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def download_verified(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url + ".CHECKSUM", timeout=timeout) as response:
        checksum_text = response.read().decode("utf-8").strip()
    checksum = checksum_text.split()[0].lower()
    if len(checksum) != 64:
        raise ValueError(f"Invalid checksum for {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        archive = response.read()
    actual = hashlib.sha256(archive).hexdigest()
    if actual != checksum:
        raise ValueError(f"Checksum mismatch for {url}")
    return archive


def parse_daily_archive(archive: bytes, expected_name: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        if names != [expected_name]:
            raise ValueError(f"Expected one ZIP entry named {expected_name}, found {names}")
        raw = bundle.read(names[0])
    probe = raw.removeprefix(b"\xef\xbb\xbf")[:1]
    frame = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        skiprows=1 if probe and probe.isalpha() else 0,
        usecols=[0, 4],
        dtype=np.float64,
        engine="c",
    )
    times = frame.iloc[:, 0].to_numpy(copy=False)
    times = np.where(times > 100_000_000_000_000, np.floor(times / 1_000), times).astype(np.int64)
    return times, frame.iloc[:, 1].to_numpy(dtype=np.float32, copy=False)


def safe_asset(asset: str) -> str:
    import base64

    return base64.urlsafe_b64encode(asset.encode()).decode().rstrip("=")


def export_asset(
    asset: dict,
    output: Path,
    start_ms: int,
    end_ms: int,
    timeout: float,
) -> dict:
    subject = str(asset["asset"])
    symbol = str(asset["symbol"])
    rows = (end_ms - start_ms) // 1_000
    directory = output / "assets" / safe_asset(subject)
    close_path = directory / "close.f32"
    observed_path = directory / "observed.u1"
    sidecar_path = directory / "manifest.json"
    expected_close_bytes = rows * 4
    expected_observed_bytes = rows
    if sidecar_path.exists() and close_path.exists() and observed_path.exists():
        cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            cached.get("symbol") == symbol
            and cached.get("startMs") == start_ms
            and cached.get("endExclusiveMs") == end_ms
            and close_path.stat().st_size == expected_close_bytes
            and observed_path.stat().st_size == expected_observed_bytes
        ):
            return {**cached, "asset": subject, "cached": True}

    directory.mkdir(parents=True, exist_ok=True)
    close_partial = close_path.with_suffix(".f32.partial")
    observed_partial = observed_path.with_suffix(".u1.partial")
    close = np.memmap(close_partial, dtype="<f4", mode="w+", shape=(rows,))
    observed = np.memmap(observed_partial, dtype="u1", mode="w+", shape=(rows,))
    close[:] = np.nan
    observed[:] = 0
    source_bytes = 0
    missing_days = []
    day_ms = 86_400_000
    for day in range(start_ms, end_ms, day_ms):
        date = np.datetime64(day, "ms").astype("datetime64[D]").astype(str)
        archive_name = f"{symbol}-1s-{date}.zip"
        url = (
            "https://data.binance.vision/data/spot/daily/klines/"
            f"{urllib.parse.quote(symbol)}/1s/{urllib.parse.quote(archive_name)}"
        )
        try:
            archive = download_verified(url, timeout)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                missing_days.append(date)
                continue
            raise
        source_bytes += len(archive)
        timestamps, closes = parse_daily_archive(archive, archive_name.removesuffix(".zip") + ".csv")
        indices = ((timestamps - start_ms) // 1_000).astype(np.int64)
        valid = (indices >= 0) & (indices < rows) & ((timestamps - start_ms) % 1_000 == 0)
        close[indices[valid]] = closes[valid]
        observed[indices[valid]] = 1
    close.flush()
    observed.flush()
    observed_rows = int(np.count_nonzero(observed))
    del close
    del observed
    os.replace(close_partial, close_path)
    os.replace(observed_partial, observed_path)
    result = {
        "asset": subject,
        "symbol": symbol,
        "startMs": start_ms,
        "endExclusiveMs": end_ms,
        "rows": rows,
        "observedRows": observed_rows,
        "coverage": observed_rows / rows,
        "sourceBytes": source_bytes,
        "missingDays": missing_days,
        "closeFile": str(close_path.relative_to(ROOT)).replace("\\", "/"),
        "observedFile": str(observed_path.relative_to(ROOT)).replace("\\", "/"),
        "cached": False,
    }
    sidecar_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    manifest_path = resolved(args.manifest)
    output = resolved(args.output)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    start_ms = int(np.datetime64(source["window"]["start"].removesuffix("Z")).astype("datetime64[ms]").astype(np.int64))
    end_ms = int(np.datetime64(source["window"]["endExclusive"].removesuffix("Z")).astype("datetime64[ms]").astype(np.int64))
    assets = [row for row in source["assets"] if float(row.get("coverage", 0)) >= 0.95]
    if args.limit is not None:
        assets = assets[: args.limit]
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(export_asset, asset, output, start_ms, end_ms, args.timeout): asset
            for asset in assets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"1s close cache {completed}/{len(assets)}: {result['asset']} "
                f"coverage={result['coverage']:.4f} cached={result['cached']}",
                flush=True,
            )
    results.sort(key=lambda row: row["asset"])
    artifact = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compact checksum-verified 1s spot close cache for global technical/spectral search",
        "sourceManifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "sourceManifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "window": {"startMs": start_ms, "endExclusiveMs": end_ms, "rows": (end_ms - start_ms) // 1_000},
        "assets": results,
        "assetCount": len(results),
        "sourceBytesDownloadedThisRun": sum(row["sourceBytes"] for row in results if not row["cached"]),
        "cacheBytes": sum(int(row["rows"]) * 5 for row in results),
        "elapsedSeconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {(output / 'manifest.json').relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
