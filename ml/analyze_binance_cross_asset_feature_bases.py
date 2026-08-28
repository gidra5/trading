from __future__ import annotations

import argparse
import base64
import concurrent.futures
import heapq
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_live_component_feature_bases import (
    ALPHA,
    FEATURE_BINS,
    MAGNITUDE_QUANTILES,
    Component,
    components,
    encode_columns,
    fit_model,
    nonoverlapping_indices,
    quantize_columns,
    split_indices,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AXIS = ROOT / "data/runtime-cache/binance-cross-asset-1m-basis-30d"
DEFAULT_BASE = ROOT / "data/runtime-cache/global-feature-basis-30d"
DEFAULT_BASE_RESULTS = ROOT / "data/benchmarks/tiered-component-feature-bases.json"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/binance-cross-asset-component-feature-bases.json"
DEFAULT_REPORT = ROOT / "docs/experiments/binance-cross-asset-component-feature-bases-2026-08-20.md"
ROWS = 43_200
AXIS_START_MS = int(np.datetime64("2026-07-18T00:00:00.000").astype("datetime64[ms]").astype(np.int64))
SOURCE_COLUMNS = 10
METRICS_COLUMNS = 7
BOOK_DEPTH_COLUMNS = 25
FAST_COLUMNS = 23
FINALISTS = 16
MAX_SUBSET_SIZE = 3
RANKING_LIMIT = 160
PARSIMONY_BITS = 0.001
MIN_EFFECTIVE_TRAIN_PER_STATE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-dir", type=Path, default=DEFAULT_AXIS)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-results", type=Path, default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit-assets", type=int)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_only:
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        args.report.write_text(render_report(artifact), encoding="utf-8")
        print(f"Wrote {display(args.report)}")
        return
    axis_manifest = json.loads((args.axis_dir / "manifest.json").read_text(encoding="utf-8"))
    derivatives_manifest = read_optional_json(args.axis_dir / "derivatives-manifest.json")
    book_depth_manifest = read_optional_json(args.axis_dir / "book-depth-manifest.json")
    fast_manifest = read_optional_json(args.axis_dir / "fast-manifest.json")
    base_manifest = json.loads((args.base_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = base_manifest["datasets"][0]
    base_results = json.loads(args.base_results.read_text(encoding="utf-8"))
    rows = int(dataset["rows"])
    feature_count = int(dataset["featureCount"])
    target_count = int(dataset["targetCount"])
    base_features = np.memmap(
        args.base_dir / dataset["files"]["features"], dtype="<f4", mode="r",
        shape=(rows, feature_count),
    )
    targets = np.memmap(
        args.base_dir / dataset["files"]["targets"], dtype="<f4", mode="r",
        shape=(rows, target_count),
    )
    splits = np.asarray(np.memmap(
        args.base_dir / dataset["files"]["splits"], dtype="u1", mode="r", shape=(rows,),
    ), dtype=np.uint8)
    quantization_train = splits == 0
    times_ms = np.asarray(np.memmap(
        args.base_dir / dataset["files"]["times"], dtype="<f8", mode="r", shape=(rows,),
    ), dtype=np.float64)
    start_text = axis_manifest["window"]["start"].removesuffix("Z")
    start_ms = float(np.datetime64(start_text).astype("datetime64[ms]").astype(np.int64))
    # Origins are stamped at the final second of a completed UTC minute (for
    # example 03:59:59).  Flooring selects the 03:59 candle; rounding would
    # select 04:00 and leak the first future minute into every candidate.
    origin_rows = np.floor((times_ms - start_ms) / 60_000).astype(np.int64)
    if np.any(origin_rows < 0) or np.any(origin_rows >= ROWS):
        raise ValueError("base origins fall outside the cross-asset minute axis")
    feature_by_id = {row["id"]: index for index, row in enumerate(dataset["features"])}
    base_result_by_key = {
        (row["horizonId"], row["componentId"]): row for row in base_results["targets"]
    }
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for horizon_index, target_definition in enumerate(dataset["targets"]):
        horizon_id = target_definition["id"]
        horizon_seconds = int(round(float(target_definition["minutes"]) * 60))
        returns = np.asarray(targets[:, horizon_index], dtype=np.float64) / 10_000
        active_train = np.abs(returns[(splits == 0) & (returns != 0)])
        thresholds = np.quantile(active_train, MAGNITUDE_QUANTILES)
        for component in components():
            current = base_result_by_key[(horizon_id, component.id)]
            condition = component.condition(returns, thresholds)
            labels = component.target(returns, thresholds)
            train = condition & (splits == 0)
            primary = nonoverlapping_indices(times_ms / 1_000, condition & (splits == 1), horizon_seconds)
            transfer = nonoverlapping_indices(times_ms / 1_000, condition & (splits == 2), horizon_seconds)
            baseline_state = np.zeros(rows, dtype=np.int64)
            baseline_states = 1
            baseline_model = fit_model(
                baseline_state[train], labels[train], baseline_states, component.classes
            )
            baseline_joint, baseline_totals = baseline_model
            baseline_log_probability = np.log2(
                (baseline_joint[baseline_state, labels] + ALPHA)
                / (baseline_totals[baseline_state] + ALPHA * component.classes)
            )
            contexts[(horizon_id, component.id)] = {
                "horizonId": horizon_id,
                "horizonSeconds": horizon_seconds,
                "component": component,
                "returns": returns,
                "thresholds": thresholds,
                "labels": labels,
                "train": train,
                "primary": primary,
                "transfer": transfer,
                "primaryBlocks": split_indices(primary),
                "transferBlocks": split_indices(transfer),
                "baselineState": baseline_state,
                "baselineStates": baseline_states,
                "baselineModel": baseline_model,
                "baselineLogProbability": baseline_log_probability,
                "baseResult": current,
                "baseFeatures": [],
                "quantizationTrain": quantization_train,
                "heap": [],
                "serial": 0,
            }
    for context in contexts.values():
        previous_ids = list(
            context["baseResult"].get("selectedBasis", {}).get("features", [])
        )
        if previous_ids:
            previous_raw = np.asarray(
                base_features[:, [feature_by_id[item] for item in previous_ids]],
                dtype=np.float64,
            )
            previous_quantized, _, previous_arities = quantize_columns(
                previous_raw, quantization_train, FEATURE_BINS
            )
            previous_score = score_quantized(
                previous_quantized,
                tuple(range(previous_quantized.shape[1])),
                previous_arities,
                context,
            )
        else:
            previous_score = {
                "primaryBits": 0.0, "transferBits": 0.0,
                "primaryBlockBits": [], "transferBlockBits": [], "states": 1,
            }
        context["previousComparable"] = {"features": previous_ids, **previous_score}
    # BTC is represented by the existing 147-coordinate inventory below.  Its
    # replicated cross-market copy is excluded to avoid duplicate candidates.
    catalog: dict[str, dict[str, Any]] = {}
    examined_by_family: dict[str, int] = {}
    base_value_by_id: dict[str, np.ndarray] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    base_screen: list[tuple[str, dict[str, Any], np.ndarray]] = []
    for feature_id, feature_index in feature_by_id.items():
        definition = dataset["features"][feature_index]
        values = np.asarray(base_features[:, feature_index], dtype=np.float64)
        candidate = {
            "asset": "BTC/general baseline inventory",
            "feature": feature_id,
            "name": definition.get("name", feature_id),
            "family": definition.get("family", "existing inventory"),
            "source": "existing 147-coordinate recent broad dataset",
            "construction": definition.get("parameters", definition.get("name", feature_id)),
            "lookback": definition.get("lookback", "documented in source manifest"),
            "delay": definition.get("delay", "documented in source manifest"),
            "availabilityClass": "existing-source",
            "availabilityScore": base_availability(feature_id, definition),
            "scope": "existing BTC/general inventory",
        }
        base_screen.append((feature_id, candidate, values))
        base_value_by_id[feature_id] = values
        examined_by_family[candidate["family"]] = examined_by_family.get(candidate["family"], 0) + 1
    catalog.update(screen_group(base_screen, contexts, quantization_train, executor))
    assets = [
        row for row in axis_manifest["assets"]
        if row.get("preferredMarket") and row.get("asset") != "BTC"
    ]
    if args.limit_assets:
        assets = assets[: args.limit_assets]
    for asset_index, asset in enumerate(assets):
        source = load_asset_sources(args.axis_dir, asset)
        derived = derive_asset_features(asset, source)
        asset_screen: list[tuple[str, dict[str, Any], np.ndarray]] = []
        for candidate_id, candidate in derived.items():
            values = candidate.pop("values")[origin_rows]
            if np.count_nonzero(np.isfinite(values)) < int(0.95 * values.size):
                continue
            examined_by_family[candidate["family"]] = examined_by_family.get(candidate["family"], 0) + 1
            candidate["scope"] = "replicated cross-market inventory"
            asset_screen.append((candidate_id, candidate, values))
        catalog.update(screen_group(asset_screen, contexts, quantization_train, executor))
        if asset_index % 10 == 0 or asset_index + 1 == len(assets):
            print(f"Screened {asset_index + 1}/{len(assets)} assets ({len(catalog):,} features)", flush=True)
    finalist_ids: set[str] = set()
    for context in contexts.values():
        ranking = sorted((item[2] for item in context["heap"]), key=lambda row: row["primaryBits"], reverse=True)
        context["ranking"] = ranking
        finalists = choose_finalists(ranking)
        context["finalists"] = finalists
        finalist_ids.update(row["id"] for row in finalists)
    executor.shutdown(wait=True)
    selected_values: dict[str, np.ndarray] = {}
    selected_by_asset: dict[str, set[str]] = {}
    for candidate_id in finalist_ids:
        if catalog[candidate_id]["scope"] == "existing BTC/general inventory":
            selected_values[candidate_id] = base_value_by_id[candidate_id]
        else:
            selected_by_asset.setdefault(catalog[candidate_id]["asset"], set()).add(candidate_id)
    asset_by_name = {row["asset"]: row for row in assets}
    for asset_name, ids in selected_by_asset.items():
        derived = derive_asset_features(
            asset_by_name[asset_name], load_asset_sources(args.axis_dir, asset_by_name[asset_name])
        )
        for candidate_id in ids:
            selected_values[candidate_id] = derived[candidate_id]["values"][origin_rows]
    results = []
    for key, context in contexts.items():
        result = search_subsets(context, selected_values)
        result["candidateCount"] = len(catalog)
        results.append(result)
        print(f"Selected {key[0]} {key[1]}: {result['status']}", flush=True)
    catalog_rows = sorted(catalog.values(), key=lambda row: row["maximumPrimaryBits"], reverse=True)
    artifact = {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "objective": "Availability-aware global refit of each BTC future-return component basis over the existing and replicated cross-market inventories",
        "metric": "chronological held-out log2 likelihood gain versus the no-feature marginal predictor",
        "universe": {
            "source": axis_manifest["sourceUniverse"],
            "sourceUniverseSize": axis_manifest["sourceUniverseSize"],
            "sourceBasisSize": axis_manifest["sourceBasisSize"],
            "explicitAssets": axis_manifest["explicitAssets"],
            "requestedAssets": axis_manifest["requestedAssets"],
            "analyzedAssets": len(assets),
            "baselineAsset": "BTC",
            "axisCoverage": axis_manifest["coverage"],
            "sourceCoverage": {
                "minuteAxis": axis_manifest["coverage"],
                "futuresMetricsAtLeast95Percent": derivatives_manifest.get("completedMetrics", 0),
                "fundingWithEvents": derivatives_manifest.get("completedFunding", 0),
                "bookDepthAtLeast95Percent": book_depth_manifest.get("completedAtLeast95Percent", 0),
                "fastSpotAtLeast95Percent": fast_manifest.get("completedAtLeast95Percent", 0),
                "minuteAxisBytes": sum(
                    (ROOT / market["file"]).stat().st_size
                    for asset in axis_manifest["assets"] for market in asset["markets"]
                ),
                "fastSourceBytesDownloaded": fast_manifest.get("sourceBytes", 0),
                "bookDepthSourceBytesDownloaded": book_depth_manifest.get("sourceBytes", 0),
            },
            "assets": [
                {
                    "asset": asset["asset"],
                    "rank": asset["rank"],
                    "explicit": asset["requestedExplicitly"],
                    "preferredMarket": asset["preferredMarket"],
                    "markets": [
                        {
                            "venue": market["venue"],
                            "symbol": market["symbol"],
                            "coverage": market["coverage"],
                        }
                        for market in asset["markets"]
                    ],
                }
                for asset in axis_manifest["assets"]
            ],
        },
        "dataset": {
            "window": axis_manifest["window"],
            "split": base_manifest["split"],
            "origins": rows,
        },
        "search": {
            "candidateFeatures": len(catalog),
            "existingCandidateFeatures": len(feature_by_id),
            "crossMarketCandidateFeatures": len(catalog) - len(feature_by_id),
            "examinedByFamily": examined_by_family,
            "finalistsPerHead": FINALISTS,
            "maximumSubsetSize": MAX_SUBSET_SIZE,
            "rankingRetainedPerHead": RANKING_LIMIT,
            "selection": "all 147 existing coordinates and every source-supported cross-market candidate are scored marginally; subsets up to three are exhaustive within 16 availability/diversity-aware finalists selected without transfer data",
            "confirmation": "positive primary halves are required for selection; transfer total and halves are untouched confirmation",
            "warning": "finite transformed feature inventory and histogram estimator; not a mathematical optimum over arbitrary functions",
        },
        "featureCatalog": catalog_rows,
        "targets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {display(args.output)}")
    print(f"Wrote {display(args.report)}")


def load_asset_sources(axis_dir: Path, asset: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "spot": None, "usdm-futures": None, "preferred": None,
        "metrics": None, "funding": None, "bookDepth": None, "fast": None,
    }
    for market in asset["markets"]:
        file = ROOT / market["file"]
        values = np.memmap(file, dtype="<f4", mode="r", shape=(ROWS, SOURCE_COLUMNS))
        output[market["venue"]] = np.asarray(values, dtype=np.float64)
        if asset["preferredMarket"]["venue"] == market["venue"]:
            output["preferred"] = output[market["venue"]]
    directory = axis_dir / "assets" / encoded_asset(asset["asset"])
    metrics_file = directory / "usdm-metrics.f32"
    if metrics_file.exists() and metrics_file.stat().st_size == (ROWS // 5) * METRICS_COLUMNS * 4:
        output["metrics"] = np.asarray(np.memmap(
            metrics_file, dtype="<f4", mode="r", shape=(ROWS // 5, METRICS_COLUMNS)
        ), dtype=np.float64)
    funding_file = directory / "usdm-funding.json"
    if funding_file.exists():
        output["funding"] = json.loads(funding_file.read_text(encoding="utf-8"))
    book_depth_file = directory / "usdm-book-depth.f32"
    if book_depth_file.exists() and book_depth_file.stat().st_size == ROWS * BOOK_DEPTH_COLUMNS * 4:
        output["bookDepth"] = np.asarray(np.memmap(
            book_depth_file, dtype="<f4", mode="r", shape=(ROWS, BOOK_DEPTH_COLUMNS)
        ), dtype=np.float64)
    fast_file = directory / "spot-fast.f32"
    if fast_file.exists() and fast_file.stat().st_size == ROWS * FAST_COLUMNS * 4:
        output["fast"] = np.asarray(np.memmap(
            fast_file, dtype="<f4", mode="r", shape=(ROWS, FAST_COLUMNS)
        ), dtype=np.float64)
    return output


def derive_asset_features(asset: dict[str, Any], sources: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = sources["preferred"]
    assert isinstance(values, np.ndarray)
    name = asset["asset"]
    prefix = safe_id(name)
    preferred_venue = asset["preferredMarket"]["venue"]
    preferred_coverage = next(
        (float(row["coverage"]) for row in asset["markets"] if row["venue"] == preferred_venue),
        0.0,
    )
    close = values[:, 3]
    observed = values[:, 9] == 1
    close = np.where(observed, close, np.nan)
    returns = log_return(close, 1)
    output: dict[str, dict[str, Any]] = {}

    def add(
        feature: str,
        value: np.ndarray,
        family: str,
        lookback: str,
        construction: str,
        source: str = "preferred Binance 1m kline",
        delay: str = "latest completed minute",
        availability_score: float = 0.96,
    ) -> None:
        output[f"{prefix}__{feature}"] = {
            "values": value,
            "asset": name,
            "feature": feature,
            "name": f"{name} {feature}",
            "family": family,
            "source": source,
            "construction": construction,
            "lookback": lookback,
            "delay": delay,
            "availabilityClass": "archive-backed",
            "availabilityScore": availability_score * preferred_coverage,
        }

    for window in (1, 5, 15, 60):
        add(f"return-{window}m", log_return(close, window), "cross-asset return", f"{window}m", f"log close return over {window} completed minutes")
    add("zero-return-1m", (returns == 0).astype(np.float64), "cross-asset activity", "1m", "exact-zero previous minute return")
    for window in (15, 60):
        add(f"active-fraction-{window}m", rolling_mean((returns != 0).astype(np.float64), window), "cross-asset activity", f"{window}m", "fraction of nonzero minute returns")
    add("zero-run-age", zero_run_age(returns), "cross-asset activity", "recursive", "log1p consecutive exact-zero minute returns")
    for window in (5, 15, 30, 60, 240):
        add(f"realized-volatility-{window}m", np.sqrt(rolling_sum(returns * returns, window)), "cross-asset volatility", f"{window}m", "sqrt sum squared minute log returns")
    high, low, open_ = values[:, 1], values[:, 2], values[:, 0]
    add("range-1m", np.log(high / low) * 10_000, "cross-asset candle shape", "1m", "10000 log high/low")
    add("close-location-1m", np.divide(2 * close - high - low, high - low, out=np.zeros_like(close), where=high > low), "cross-asset candle shape", "1m", "normalized close within completed high-low range")
    quote, trades = values[:, 5], values[:, 6]
    log_quote, log_trades = np.log1p(quote), np.log1p(trades)
    add("log-quote-volume-1m", log_quote, "cross-asset activity regime", "1m", "log1p quote volume")
    add("log-trade-count-1m", log_trades, "cross-asset activity regime", "1m", "log1p trade count")
    add("log-mean-trade-notional-1m", np.log((quote + 1) / (trades + 1)), "cross-asset trade structure", "1m", "log quote volume per trade")
    add("quote-volume-surprise-60m", log_quote - ema(log_quote, 60), "cross-asset activity regime", "60m recursive", "log quote volume minus EMA(60m)")
    add("trade-count-surprise-60m", log_trades - ema(log_trades, 60), "cross-asset activity regime", "60m recursive", "log trade count minus EMA(60m)")
    add("completed-volume-60m", np.log1p(rolling_sum(quote, 60)), "cross-asset volume regime", "60m", "log1p quote volume over the latest 60 completed minutes")
    buy_base, buy_quote, base = values[:, 7], values[:, 8], values[:, 4]
    quote_imbalance = np.divide(2 * buy_quote - quote, quote, out=np.zeros_like(quote), where=quote > 0)
    base_imbalance = np.divide(2 * buy_base - base, base, out=np.zeros_like(base), where=base > 0)
    add("taker-quote-imbalance-1m", quote_imbalance, "cross-asset aggressor flow", "1m", "(2*taker-buy quote-total quote)/total quote")
    add("taker-base-imbalance-1m", base_imbalance, "cross-asset aggressor flow", "1m", "(2*taker-buy base-total base)/total base")
    for window in (5, 15, 60):
        add(f"taker-imbalance-ema-{window}m", ema(quote_imbalance, window), "cross-asset aggressor flow", f"{window}m recursive", f"EMA({window}m) taker quote imbalance")
    for period in (2, 8, 32):
        add(f"rsi-{period}m", rsi(returns, period), "cross-asset RSI", f"{period}m recursive", f"Wilder RSI({period}m)")
    emas = {period: ema(close, period) for period in (2, 8, 32)}
    for period, average in emas.items():
        distance = np.log(close / average) * 10_000
        add(f"ema-distance-{period}m", distance, "cross-asset EMA value", f"{period}m recursive", f"10000 log close/EMA({period}m)")
    add("ema-slope-8m-8m", lag_difference(np.log(emas[8]) * 10_000, 8), "cross-asset EMA slope", "8m recursive + 8m lag", "EMA(8m) log slope over 8m")
    ema2 = np.log(emas[2]) * 10_000
    slope2 = lag_difference(ema2, 1)
    add("ema-acceleration-2m-1m", lag_difference(slope2, 1), "cross-asset EMA acceleration", "2m recursive + 2m", "second difference of EMA(2m)")
    for fast, slow, signal in ((3, 10, 4), (12, 26, 9)):
        line = ema(close, fast) - ema(close, slow)
        histogram = np.divide(line - ema(line, signal), close, out=np.zeros_like(close), where=close > 0) * 10_000
        add(f"macd-histogram-{fast}-{slow}-{signal}", histogram, "cross-asset MACD", f"{slow}m recursive", f"normalized MACD({fast},{slow},{signal}) histogram")
    for window in (16, 60):
        add(f"efficiency-ratio-{window}m", efficiency_ratio(returns, window), "cross-asset path efficiency", f"{window}m", "absolute net return divided by path length")
    add("signed-variance-efficiency-16m", signed_variance_efficiency(returns, 16), "cross-asset path efficiency", "16m", "signed squared-return balance")
    for window in (16, 64):
        add(f"haar-contrast-{window}m", haar_contrast(returns, window), "cross-asset wavelet", f"{window}m", "normalized first Haar contrast")
        real, imaginary = fourier_one(returns, window)
        add(f"fourier-real-k1-{window}m", real, "cross-asset spectral phase", f"{window}m", "normalized real DFT coefficient k=1")
        add(f"fourier-imag-k1-{window}m", imaginary, "cross-asset spectral phase", f"{window}m", "normalized imaginary DFT coefficient k=1")
        add(f"fourier-energy-k1-{window}m", np.hypot(real, imaginary), "cross-asset spectral energy", f"{window}m", "normalized DFT k=1 magnitude")
    add("return-skew-60m", rolling_standardized_moment(returns, 60, 3), "cross-asset return shape", "60m", "rolling standardized third moment")
    add("return-kurtosis-60m", rolling_standardized_moment(returns, 60, 4), "cross-asset return shape", "60m", "rolling standardized fourth moment")
    spot, futures = sources.get("spot"), sources.get("usdm-futures")
    if isinstance(spot, np.ndarray) and isinstance(futures, np.ndarray):
        spot_close, futures_close = spot[:, 3], futures[:, 3]
        basis = np.log(futures_close / spot_close) * 10_000
        add("futures-basis-level", basis, "cross-asset futures basis", "1m", "10000 log futures/spot close", "paired Binance spot and USD-M 1m klines")
        add("futures-basis-change-1m", lag_difference(basis, 1), "cross-asset futures basis", "1m", "one-minute basis change", "paired Binance spot and USD-M 1m klines")
        for window in (5, 15, 60):
            add(f"futures-basis-deviation-{window}m", basis - ema(basis, window), "cross-asset futures basis", f"{window}m recursive", f"basis minus EMA({window}m)", "paired Binance spot and USD-M 1m klines")
        relative = log_return(futures_close, 1) - log_return(spot_close, 1)
        add("futures-minus-spot-return-1m", relative, "cross-asset venue lead-lag", "1m", "USD-M return minus spot return", "paired Binance spot and USD-M 1m klines")
        ratio = np.log((futures[:, 5] + 1) / (spot[:, 5] + 1))
        add("futures-spot-quote-activity-ratio-1m", ratio, "cross-asset venue activity", "1m", "log futures/spot quote-volume ratio", "paired Binance spot and USD-M 1m klines")
    metrics = sources.get("metrics")
    if isinstance(metrics, np.ndarray):
        metric_values = np.where(metrics[:, 6:7] == 1, metrics[:, :6], np.nan)
        metric_names = [
            "open-interest", "open-interest-value", "top-account-long-short",
            "top-position-long-short", "global-long-short", "taker-buy-sell",
        ]
        transformed: dict[str, np.ndarray] = {}
        for column, metric_name in enumerate(metric_names):
            series = np.log(metric_values[:, column])
            transformed[metric_name] = series
            add(
                f"{metric_name}-log-level", expand_metric_to_minutes(series),
                "cross-asset futures positioning", "latest 5m bucket",
                f"log {metric_name}", "Binance USD-M public 5m metrics archive",
                "one completed 5m publication lag", 0.93,
            )
            for buckets, label in ((1, "5m"), (3, "15m"), (12, "60m"), (48, "240m")):
                add(
                    f"{metric_name}-log-change-{label}",
                    expand_metric_to_minutes(lag_difference(series, buckets)),
                    "cross-asset futures positioning", label,
                    f"log change in {metric_name} over {label}",
                    "Binance USD-M public 5m metrics archive",
                    "one completed 5m publication lag", 0.93,
                )
            add(
                f"{metric_name}-deviation-24h",
                expand_metric_to_minutes(series - ema(series, 288)),
                "cross-asset futures positioning", "24h recursive",
                f"log {metric_name} minus EMA(288x5m)",
                "Binance USD-M public 5m metrics archive",
                "one completed 5m publication lag", 0.93,
            )
        implied_price = np.exp(transformed["open-interest-value"] - transformed["open-interest"])
        metric_price = expand_metric_to_minutes(implied_price)
        add(
            "open-interest-implied-price-basis", np.log(metric_price / close) * 10_000,
            "cross-asset futures positioning", "latest 5m bucket",
            "10000 log((open-interest value/open-interest)/preferred close)",
            "Binance USD-M public 5m metrics archive + preferred kline",
            "one completed 5m publication lag", 0.91,
        )
        add(
            "top-account-minus-global-long-short",
            expand_metric_to_minutes(transformed["top-account-long-short"] - transformed["global-long-short"]),
            "cross-asset futures positioning", "latest 5m bucket",
            "log top-account ratio minus log global ratio",
            "Binance USD-M public 5m metrics archive",
            "one completed 5m publication lag", 0.93,
        )
        add(
            "top-position-minus-account-long-short",
            expand_metric_to_minutes(transformed["top-position-long-short"] - transformed["top-account-long-short"]),
            "cross-asset futures positioning", "latest 5m bucket",
            "log top-position ratio minus log top-account ratio",
            "Binance USD-M public 5m metrics archive",
            "one completed 5m publication lag", 0.93,
        )
    funding = sources.get("funding")
    if isinstance(funding, dict) and funding.get("events"):
        funding_features = funding_to_minutes(funding["events"])
        for feature, family, construction in (
            ("funding-rate", "cross-asset funding", "latest settled funding rate"),
            ("funding-absolute-rate", "cross-asset funding", "absolute latest settled funding rate"),
            ("funding-change", "cross-asset funding", "change from preceding funding settlement"),
            ("funding-deviation-24h", "cross-asset funding", "rate minus trailing three-settlement mean"),
            ("funding-age-hours", "cross-asset funding availability", "hours since latest funding settlement"),
        ):
            add(
                feature, funding_features[feature], family, "latest event / 24h",
                construction, "Binance USD-M funding-rate REST history",
                "latest settled event", 0.95,
            )
    book_depth = sources.get("bookDepth")
    if isinstance(book_depth, np.ndarray):
        observed_book = book_depth[:, 24] == 1
        depth = np.where(observed_book[:, None], book_depth[:, :12], np.nan)
        notional = np.where(observed_book[:, None], book_depth[:, 12:24], np.nan)
        bid_indices = [4, 3, 2, 1, 0]
        ask_indices = [7, 8, 9, 10, 11]
        bands = [1, 2, 3, 4, 5]
        depth_imbalances = []
        notional_imbalances = []
        for band, bid_index, ask_index in zip(bands, bid_indices, ask_indices):
            depth_imbalance = pair_imbalance(depth[:, bid_index], depth[:, ask_index])
            notional_imbalance = pair_imbalance(notional[:, bid_index], notional[:, ask_index])
            depth_imbalances.append(depth_imbalance)
            notional_imbalances.append(notional_imbalance)
            add(
                f"book-depth-imbalance-{band}pct", depth_imbalance,
                "cross-asset futures book imbalance", "latest snapshot",
                f"base-depth bid/ask imbalance within +/-{band}%",
                "Binance USD-M public percentage-depth archive", "latest snapshot in completed minute", 0.90,
            )
            add(
                f"book-notional-imbalance-{band}pct", notional_imbalance,
                "cross-asset futures book imbalance", "latest snapshot",
                f"notional bid/ask imbalance within +/-{band}%",
                "Binance USD-M public percentage-depth archive", "latest snapshot in completed minute", 0.90,
            )
        depth_total_1 = depth[:, 4] + depth[:, 7]
        depth_total_5 = depth[:, 0] + depth[:, 11]
        notional_total_1 = notional[:, 4] + notional[:, 7]
        notional_total_5 = notional[:, 0] + notional[:, 11]
        book_raw = {
            "book-log-depth-1pct": np.log(depth_total_1),
            "book-log-depth-5pct": np.log(depth_total_5),
            "book-log-notional-1pct": np.log(notional_total_1),
            "book-log-notional-5pct": np.log(notional_total_5),
            "book-mean-depth-imbalance": np.mean(depth_imbalances, axis=0),
            "book-mean-notional-imbalance": np.mean(notional_imbalances, axis=0),
            "book-depth-imbalance-slope": depth_imbalances[-1] - depth_imbalances[0],
            "book-notional-imbalance-slope": notional_imbalances[-1] - notional_imbalances[0],
            "book-depth-concentration": np.log(depth_total_1 / depth_total_5),
            "book-notional-concentration": np.log(notional_total_1 / notional_total_5),
        }
        for feature, series in book_raw.items():
            add(
                feature, series, "cross-asset futures book shape", "latest snapshot",
                feature.replace("book-", "").replace("-", " "),
                "Binance USD-M public percentage-depth archive", "latest snapshot in completed minute", 0.90,
            )
        for feature, series in (
            ("book-delta-notional-imbalance-1pct", notional_imbalances[0]),
            ("book-delta-notional-imbalance-5pct", notional_imbalances[-1]),
            ("book-delta-mean-notional-imbalance", book_raw["book-mean-notional-imbalance"]),
            ("book-delta-log-notional-1pct", book_raw["book-log-notional-1pct"]),
            ("book-delta-log-notional-5pct", book_raw["book-log-notional-5pct"]),
        ):
            add(
                feature, lag_difference(series, 1), "cross-asset futures book change", "1m",
                f"one-minute change in {feature.removeprefix('book-delta-').replace('-', ' ')}",
                "Binance USD-M public percentage-depth archive", "latest snapshot in completed minute", 0.90,
            )
    fast = sources.get("fast")
    if isinstance(fast, np.ndarray):
        observed_fast = fast[:, 22] == 1
        fast_names = [
            "previous-return-1s", "prior-return-1s", "previous-return-zero",
            "active-count-10s", "active-count-60s", "zero-run-age-1s",
            "realized-volatility-5s", "realized-volatility-15s", "realized-volatility-60s",
            "rsi-2s", "ema-acceleration-2s-1s", "ema-slope-8s-8s",
            "range-1s", "close-location-1s", "haar-contrast-16s",
            "taker-quote-imbalance-1s", "taker-base-imbalance-1s",
            "taker-quote-imbalance-ema-2s", "taker-quote-imbalance-ema-8s",
            "log-quote-volume-1s", "log-trade-count-1s", "taker-vwap-gap-bps",
        ]
        family_by_feature = {
            "previous-return-zero": "cross-asset fast activity",
            "previous-return": "cross-asset fast return",
            "prior-return": "cross-asset fast return",
            "active-count": "cross-asset fast activity",
            "zero-run": "cross-asset fast activity",
            "realized-volatility": "cross-asset fast volatility",
            "rsi": "cross-asset fast RSI",
            "ema-acceleration": "cross-asset fast EMA acceleration",
            "ema-slope": "cross-asset fast EMA slope",
            "range": "cross-asset fast candle shape",
            "close-location": "cross-asset fast candle shape",
            "haar": "cross-asset fast wavelet",
            "taker-quote": "cross-asset fast aggressor flow",
            "taker-base": "cross-asset fast aggressor flow",
            "log-quote": "cross-asset fast activity regime",
            "log-trade": "cross-asset fast activity regime",
            "taker-vwap": "cross-asset fast aggressor flow",
        }
        lookback_by_feature = {
            "previous-return-1s": "1s", "prior-return-1s": "2s",
            "previous-return-zero": "1s", "active-count-10s": "10s",
            "active-count-60s": "60s", "zero-run-age-1s": "recursive",
            "realized-volatility-5s": "5s", "realized-volatility-15s": "15s",
            "realized-volatility-60s": "60s", "rsi-2s": "recursive",
            "ema-acceleration-2s-1s": "recursive", "ema-slope-8s-8s": "recursive",
            "range-1s": "1s", "close-location-1s": "1s", "haar-contrast-16s": "16s",
            "taker-quote-imbalance-1s": "1s", "taker-base-imbalance-1s": "1s",
            "taker-quote-imbalance-ema-2s": "recursive",
            "taker-quote-imbalance-ema-8s": "recursive",
            "log-quote-volume-1s": "1s", "log-trade-count-1s": "1s",
            "taker-vwap-gap-bps": "1s",
        }
        for column, feature in enumerate(fast_names):
            family = next((value for key, value in family_by_feature.items() if feature.startswith(key)), "cross-asset fast state")
            add(
                feature, np.where(observed_fast, fast[:, column], np.nan), family,
                lookback_by_feature[feature], feature.replace("-", " "),
                "Binance spot public 1s kline archive",
                "latest completed second at minute origin", 0.95,
            )
    return output


def score_single(values: np.ndarray, context: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, int]:
    quantized, edges, arities = quantize_columns(values[:, None], context["train"], FEATURE_BINS)
    score = score_quantized(quantized, (0,), arities, context)
    return score, edges[0], arities[0]


def quantize_candidate(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    quantized, edges, arities = quantize_columns(values[:, None], train, FEATURE_BINS)
    return quantized, edges[0], arities[0]


def aggregate_candidate(candidate: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    return {
        **candidate,
        "id": candidate_id,
        "testedHeads": 0,
        "positivePrimaryHeads": 0,
        "positiveTransferHeads": 0,
        "maximumPrimaryBits": -math.inf,
        "maximumTransferBits": -math.inf,
        "bestHead": None,
    }


def screen_group(
    entries: list[tuple[str, dict[str, Any], np.ndarray]],
    contexts: dict[tuple[str, str], dict[str, Any]],
    quantization_train: np.ndarray,
    executor: concurrent.futures.ThreadPoolExecutor,
) -> dict[str, dict[str, Any]]:
    if not entries:
        return {}
    prepared = []
    aggregates = {}
    for candidate_id, candidate, values in entries:
        quantized, edges, arity = quantize_candidate(values, quantization_train)
        prepared.append((candidate_id, candidate, quantized[:, 0], edges, arity))
        aggregates[candidate_id] = aggregate_candidate(candidate, candidate_id)

    def score_context(
        item: tuple[tuple[str, str], dict[str, Any]]
    ) -> tuple[tuple[str, str], list[tuple[str, dict[str, Any], dict[str, Any]]]]:
        key, context = item
        scored = []
        for candidate_id, candidate, state, edges, arity in prepared:
            score = score_single_quantized(state, arity, context)
            row = {
                **candidate, "id": candidate_id, **score,
                "edges": edges.tolist(), "arity": arity,
            }
            scored.append((candidate_id, score, row))
        return key, scored

    for key, scored in executor.map(score_context, contexts.items()):
        context = contexts[key]
        for candidate_id, score, row in scored:
            update_aggregate(aggregates[candidate_id], score, key)
            push_ranking(context, row)
    return aggregates


def update_aggregate(
    aggregate: dict[str, Any], score: dict[str, Any], key: tuple[str, str]
) -> None:
    aggregate["testedHeads"] += 1
    aggregate["positivePrimaryHeads"] += int(score["primaryBits"] > 0)
    aggregate["positiveTransferHeads"] += int(score["transferBits"] > 0)
    if score["primaryBits"] > aggregate["maximumPrimaryBits"]:
        aggregate["maximumPrimaryBits"] = score["primaryBits"]
        aggregate["bestHead"] = f"{key[0]}:{key[1]}"
    aggregate["maximumTransferBits"] = max(
        aggregate["maximumTransferBits"], score["transferBits"]
    )


def base_availability(feature_id: str, definition: dict[str, Any]) -> float:
    family = str(definition.get("family", "")).lower()
    if feature_id.startswith("spot-book-"):
        return 0.45
    if feature_id.startswith("gdelt-"):
        return 0.75
    if feature_id.startswith(("spot-flow-", "futures-", "open-interest-", "top-", "global-ratio-", "taker-ratio-")):
        return 0.95
    if feature_id.startswith(("eth-", "sol-", "bnb-", "doge-")):
        return 0.96
    if family in {
        "return history", "activity", "volatility", "price dynamics", "volume regime",
        "candle shape", "minute volatility", "minute candle shape",
    }:
        return 1.0
    return 0.94


def score_quantized(quantized: np.ndarray, subset: tuple[int, ...], arities: list[int], context: dict[str, Any]) -> dict[str, Any]:
    feature_state, feature_states = encode_columns(quantized, subset, arities)
    state = context["baselineState"] * feature_states + feature_state
    labels = context["labels"]
    classes = context["component"].classes
    train = context["train"]
    joint, totals = fit_model(state[train], labels[train], context["baselineStates"] * feature_states, classes)
    def ratios(indices: np.ndarray) -> np.ndarray:
        y = labels[indices]
        probability = (joint[state[indices], y] + ALPHA) / (totals[state[indices]] + ALPHA * classes)
        return np.log2(probability) - context["baselineLogProbability"][indices]

    primary_ratios = ratios(context["primary"])
    transfer_ratios = ratios(context["transfer"])
    return {
        "primaryBits": float(np.mean(primary_ratios)),
        "transferBits": float(np.mean(transfer_ratios)),
        "primaryBlockBits": [float(np.mean(ratios(block))) for block in context["primaryBlocks"]],
        "transferBlockBits": [float(np.mean(ratios(block))) for block in context["transferBlocks"]],
        "states": int(context["baselineStates"] * feature_states),
    }


def score_single_quantized(
    state: np.ndarray, states: int, context: dict[str, Any]
) -> dict[str, Any]:
    labels = context["labels"]
    classes = context["component"].classes
    train = context["train"]
    counts = np.bincount(
        state[train] * classes + labels[train], minlength=states * classes
    ).reshape(states, classes)
    totals = counts.sum(axis=1)
    denominator = totals + ALPHA * classes
    baseline_log = context["baselineLogProbability"]

    def ratios(indices: np.ndarray) -> np.ndarray:
        selected_state = state[indices]
        probability = (
            counts[selected_state, labels[indices]] + ALPHA
        ) / denominator[selected_state]
        return np.log2(probability) - baseline_log[indices]

    primary_ratios = ratios(context["primary"])
    transfer_ratios = ratios(context["transfer"])
    return {
        "primaryBits": float(np.mean(primary_ratios)),
        "transferBits": float(np.mean(transfer_ratios)),
        "primaryBlockBits": [float(np.mean(ratios(block))) for block in context["primaryBlocks"]],
        "transferBlockBits": [float(np.mean(ratios(block))) for block in context["transferBlocks"]],
        "states": int(states),
    }


def push_ranking(context: dict[str, Any], row: dict[str, Any]) -> None:
    context["serial"] += 1
    item = (float(row["primaryBits"]), context["serial"], row)
    heap = context["heap"]
    if len(heap) < RANKING_LIMIT:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def choose_finalists(ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable = [row for row in ranking if all(value > 0 for value in row["primaryBlockBits"])]
    pool = stable or ranking
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    family_count: dict[str, int] = {}
    asset_count: dict[str, int] = {}
    for row in pool:
        family = row["family"]
        if family in used:
            continue
        selected.append(row)
        used.add(family)
        family_count[family] = 1
        asset_count[row["asset"]] = asset_count.get(row["asset"], 0) + 1
        if len(selected) >= FINALISTS:
            return selected
    for row in pool:
        if row in selected or family_count.get(row["family"], 0) >= 3 or asset_count.get(row["asset"], 0) >= 3:
            continue
        selected.append(row)
        family_count[row["family"]] = family_count.get(row["family"], 0) + 1
        asset_count[row["asset"]] = asset_count.get(row["asset"], 0) + 1
        if len(selected) >= FINALISTS:
            break
    return selected


def search_subsets(context: dict[str, Any], selected_values: dict[str, np.ndarray]) -> dict[str, Any]:
    finalists = context["finalists"]
    raw = np.column_stack([selected_values[row["id"]] for row in finalists])
    quantized, edges, arities = quantize_columns(raw, context["quantizationTrain"], FEATURE_BINS)
    maximum_states = max(9, int(np.count_nonzero(context["train"])) // MIN_EFFECTIVE_TRAIN_PER_STATE)
    subsets = []
    for size in range(1, min(MAX_SUBSET_SIZE, len(finalists)) + 1):
        for subset in itertools.combinations(range(len(finalists)), size):
            states = context["baselineStates"] * math.prod(arities[index] for index in subset)
            if states > maximum_states:
                continue
            subsets.append({"indices": subset, **score_quantized(quantized, subset, arities, context)})
    stable = [row for row in subsets if all(value > 0 for value in row["primaryBlockBits"])]
    component: Component = context["component"]
    previous = context["previousComparable"]
    common = {
        "horizonId": context["horizonId"],
        "horizonSeconds": context["horizonSeconds"],
        "componentId": component.id,
        "componentLabel": component.label,
        "condition": component.condition_label,
        "previousBasis": {
            "features": previous.get("features", []),
            "primaryBits": float(previous.get("primaryBits", 0)),
            "transferBits": float(previous.get("transferBits", 0)),
        },
        "candidateCount": None,
        "finalists": [{key: row[key] for key in ("id", "asset", "feature", "family", "primaryBits", "transferBits", "primaryBlockBits", "transferBlockBits")} for row in finalists],
        "testedSubsets": len(subsets),
        "maximumAdmissibleStates": maximum_states,
    }
    if not stable:
        return {**common, "status": "no-stable-basis", "selectedBasis": empty_addition()}
    optimum = max(stable, key=lambda row: row["primaryBits"])
    near = [row for row in stable if row["primaryBits"] >= optimum["primaryBits"] - PARSIMONY_BITS]
    selected = min(near, key=lambda row: (
        -min(float(finalists[index]["availabilityScore"]) for index in row["indices"]),
        -float(np.mean([finalists[index]["availabilityScore"] for index in row["indices"]])),
        len(row["indices"]),
        -row["primaryBits"],
    ))
    selected_rows = [finalists[index] for index in selected["indices"]]
    confirmed = selected["transferBits"] > 0 and all(value > 0 for value in selected["transferBlockBits"])
    return {
        **common,
        "status": "confirmed-early" if confirmed else "primary-only",
        "primaryOptimum": describe_subset(optimum, finalists),
        "selectedBasis": {
            **describe_subset(selected, finalists),
            "featureDetails": [{key: row[key] for key in (
                "id", "asset", "feature", "family", "source", "lookback",
                "delay", "construction", "availabilityScore", "scope"
            )} for row in selected_rows],
        },
    }


def describe_subset(row: dict[str, Any], finalists: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [finalists[index] for index in row["indices"]]
    availability = [float(item["availabilityScore"]) for item in selected]
    return {
        "features": [item["id"] for item in selected],
        "assets": sorted(set(item["asset"] for item in selected)),
        "primaryBits": row["primaryBits"],
        "transferBits": row["transferBits"],
        "primaryBlockBits": row["primaryBlockBits"],
        "transferBlockBits": row["transferBlockBits"],
        "states": row["states"],
        "availabilityMinimum": min(availability, default=0),
        "availabilityMean": float(np.mean(availability)) if availability else 0,
    }


def empty_addition() -> dict[str, Any]:
    return {"features": [], "assets": [], "primaryBits": 0, "transferBits": 0, "primaryBlockBits": [], "transferBlockBits": []}


def log_return(close: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(close.size, np.nan)
    valid = np.isfinite(close[lag:]) & np.isfinite(close[:-lag]) & (close[lag:] > 0) & (close[:-lag] > 0)
    output[lag:][valid] = np.log(close[lag:][valid] / close[:-lag][valid]) * 10_000
    return output


def lag_difference(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(values.size, np.nan)
    output[lag:] = values[lag:] - values[:-lag]
    return output


def pair_imbalance(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    total = bid + ask
    return np.divide(bid - ask, total, out=np.full_like(total, np.nan), where=total > 0)


def expand_metric_to_minutes(values: np.ndarray) -> np.ndarray:
    """Make each 5m metric visible only after its documented 5m lag."""
    minute = np.arange(ROWS)
    metric_index = ((minute + 1) // 5) - 1
    output = np.full(ROWS, np.nan)
    valid = metric_index >= 0
    output[valid] = values[metric_index[valid]]
    return output


def funding_to_minutes(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    ordered = sorted(
        (row for row in events if np.isfinite(float(row["rate"]))),
        key=lambda row: int(row["time"]),
    )
    event_times = np.asarray([int(row["time"]) for row in ordered], dtype=np.int64)
    rates = np.asarray([float(row["rate"]) for row in ordered], dtype=np.float64)
    changes = np.full(rates.size, np.nan)
    if rates.size > 1:
        changes[1:] = rates[1:] - rates[:-1]
    deviations = np.full(rates.size, np.nan)
    for index in range(rates.size):
        start = max(0, index - 2)
        deviations[index] = rates[index] - float(np.mean(rates[start:index + 1]))
    origin_times = AXIS_START_MS + (np.arange(ROWS) + 1) * 60_000 - 1_000
    indices = np.searchsorted(event_times, origin_times, side="right") - 1
    valid = indices >= 0
    output = {name: np.full(ROWS, np.nan) for name in (
        "funding-rate", "funding-absolute-rate", "funding-change",
        "funding-deviation-24h", "funding-age-hours",
    )}
    output["funding-rate"][valid] = rates[indices[valid]]
    output["funding-absolute-rate"][valid] = np.abs(rates[indices[valid]])
    output["funding-change"][valid] = changes[indices[valid]]
    output["funding-deviation-24h"][valid] = deviations[indices[valid]]
    output["funding-age-hours"][valid] = (origin_times[valid] - event_times[indices[valid]]) / 3_600_000
    return output


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0)
    sums = np.concatenate(([0.0], np.cumsum(clean)))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int32))))
    output = np.full(values.size, np.nan)
    output[window - 1:] = sums[window:] - sums[:-window]
    valid = counts[window:] - counts[:-window] == window
    segment = output[window - 1:]
    segment[~valid] = np.nan
    return output


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return rolling_sum(values, window) / window


def ema(values: np.ndarray, period: int) -> np.ndarray:
    output = np.full(values.size, np.nan)
    alpha = 2 / (period + 1)
    state = math.nan
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        state = value if not np.isfinite(state) else state + alpha * (value - state)
        output[index] = state
    return output


def rsi(returns: np.ndarray, period: int) -> np.ndarray:
    gain = ema(np.maximum(np.nan_to_num(returns), 0), period)
    loss = ema(np.maximum(-np.nan_to_num(returns), 0), period)
    return np.divide(gain, gain + loss, out=np.full_like(gain, 0.5), where=gain + loss > 0)


def zero_run_age(returns: np.ndarray) -> np.ndarray:
    output = np.zeros(returns.size)
    age = 0
    for index, value in enumerate(returns):
        age = age + 1 if value == 0 else 0
        output[index] = math.log1p(age)
    return output


def efficiency_ratio(returns: np.ndarray, window: int) -> np.ndarray:
    net = rolling_sum(returns, window)
    path = rolling_sum(np.abs(returns), window)
    return np.divide(np.abs(net), path, out=np.zeros_like(net), where=path > 0)


def signed_variance_efficiency(returns: np.ndarray, window: int) -> np.ndarray:
    signed = rolling_sum(np.sign(returns) * returns * returns, window)
    total = rolling_sum(returns * returns, window)
    return np.divide(signed, total, out=np.zeros_like(signed), where=total > 0)


def haar_contrast(returns: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    recent = rolling_sum(returns, half)
    prior = np.full(returns.size, np.nan)
    prior[half:] = recent[:-half]
    scale = np.sqrt(rolling_sum(returns * returns, window))
    return np.divide(recent - prior, scale, out=np.zeros_like(scale), where=scale > 0)


def fourier_one(returns: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    clean = np.nan_to_num(returns)
    index = np.arange(window)
    cosine = np.cos(-2 * np.pi * index / window)
    sine = np.sin(-2 * np.pi * index / window)
    real = np.full(returns.size, np.nan)
    imaginary = np.full(returns.size, np.nan)
    real[window - 1:] = np.convolve(clean, cosine[::-1], mode="valid")
    imaginary[window - 1:] = np.convolve(clean, sine[::-1], mode="valid")
    scale = np.sqrt(rolling_sum(clean * clean, window)) * math.sqrt(window)
    return (
        np.divide(real, scale, out=np.zeros_like(real), where=scale > 0),
        np.divide(imaginary, scale, out=np.zeros_like(imaginary), where=scale > 0),
    )


def rolling_standardized_moment(values: np.ndarray, window: int, order: int) -> np.ndarray:
    mean = rolling_mean(values, window)
    raw2 = rolling_mean(values ** 2, window)
    variance = np.maximum(raw2 - mean ** 2, 0)
    if order == 3:
        raw3 = rolling_mean(values ** 3, window)
        moment = raw3 - 3 * mean * raw2 + 2 * mean ** 3
    elif order == 4:
        raw3 = rolling_mean(values ** 3, window)
        raw4 = rolling_mean(values ** 4, window)
        moment = raw4 - 4 * mean * raw3 + 6 * mean ** 2 * raw2 - 3 * mean ** 4
    else:
        raise ValueError(f"unsupported standardized moment order: {order}")
    denominator = variance ** (order / 2)
    return np.divide(moment, denominator, out=np.zeros_like(moment), where=denominator > 0)


def safe_id(asset: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in asset).strip("-")


def encoded_asset(asset: str) -> str:
    return base64.urlsafe_b64encode(asset.encode("utf-8")).decode("ascii").rstrip("=")


def read_optional_json(file: Path) -> dict[str, Any]:
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def render_report(artifact: dict[str, Any]) -> str:
    universe = artifact["universe"]
    contributing_assets = sorted({
        row["asset"] for row in artifact["featureCatalog"]
        if row.get("scope") == "replicated cross-market inventory"
    })
    excluded_assets = []
    for asset in universe.get("assets", []):
        if asset["asset"] == "BTC" or asset["asset"] in contributing_assets:
            continue
        preferred = asset.get("preferredMarket") or {}
        market = next((row for row in asset["markets"] if row["venue"] == preferred.get("venue")), None)
        excluded_assets.append((asset["asset"], 0 if market is None else market["coverage"]))
    coverage = universe.get("sourceCoverage", {
        "minuteAxis": universe.get("axisCoverage", {}),
        "fastSpotAtLeast95Percent": 0,
        "futuresMetricsAtLeast95Percent": 0,
        "fundingWithEvents": 0,
        "bookDepthAtLeast95Percent": 0,
    })
    lines = [
        "# Binance cross-asset BTC component feature bases",
        "",
        f"Generated `{artifact['generatedAt']}`.",
        "",
        "This experiment replicates every source-supported candle, activity, volatility, technical, spectral, flow, positioning, funding, and book-depth family across the latest study's largest 1-minute Binance basis. It then globally refits each component from the union of the existing 147-coordinate inventory and the replicated cross-market inventory; the previous basis is a benchmark, not a frozen baseline.",
        "",
        "## Scope",
        "",
        f"- Source market universe: {universe['sourceUniverseSize']} economic assets.",
        f"- Latest 1-minute independent-scale basis: {universe['sourceBasisSize']} assets.",
        f"- Explicit additions: {', '.join(universe['explicitAssets'])}.",
        f"- Requested assets: {universe['requestedAssets']} total; {universe['analyzedAssets']} non-BTC markets attempted; {len(contributing_assets)} non-BTC markets contributed at least one eligible coordinate.",
        "- BTC remains in the existing 147-coordinate inventory and is excluded only from the replicated cross-market copy.",
        f"- Wholly excluded for insufficient recent coverage: {', '.join(f'{asset} ({coverage_value:.1%})' for asset, coverage_value in excluded_assets) or 'none'}; individual low-coverage source tiers are excluded by the 95% rule below.",
        f"- Candidate coordinates: {artifact['search']['candidateFeatures']:,} ({artifact['search']['existingCandidateFeatures']:,} existing + {artifact['search']['crossMarketCandidateFeatures']:,} replicated cross-market).",
        "- Window: 2026-07-18 through 2026-08-16 UTC; chronological train, primary, and untouched transfer blocks match the recent broad-basis audit.",
        "- Alignment: each stored `:59` origin uses the candle from that same UTC minute via floor-to-minute indexing; the following minute is excluded.",
        "",
        "## Backfilled source coverage",
        "",
        "| Source tier | Markets/assets with usable coverage | Causal availability |",
        "|---|---:|---|",
        f"| 1m spot | {coverage['minuteAxis']['withSpot']} | completed minute |",
        f"| 1m USD-M futures | {coverage['minuteAxis']['withUsdmFutures']} | completed minute |",
        f"| paired spot + futures | {coverage['minuteAxis']['withBoth']} | completed minute |",
        f"| spot 1s fast state | {coverage['fastSpotAtLeast95Percent']} | latest completed second |",
        f"| USD-M OI/positioning metrics | {coverage['futuresMetricsAtLeast95Percent']} | one completed 5m publication lag |",
        f"| USD-M settled funding | {coverage['fundingWithEvents']} | latest settled event |",
        f"| USD-M 12-band percentage depth | {coverage['bookDepthAtLeast95Percent']} | latest complete snapshot in completed minute |",
        "",
        "## Decision summary",
        "",
        "The enlarged search does **not** justify replacing the current component bases wholesale. The selected refits win strongly on the primary selection week but lose on average on the untouched transfer week at every horizon, which is direct evidence of winner's-curse/multiple-testing overfit from searching 31,043 coordinates.",
        "",
        "| Horizon | Mean previous transfer bits | Mean refit transfer bits | Mean change | Refit beats previous | Positive-transfer cross-market leads that beat previous |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in ("1s", "1m", "15m", "1h"):
        selected_rows = [row for row in artifact["targets"] if row["horizonId"] == horizon]
        previous_mean = float(np.mean([row["previousBasis"]["transferBits"] for row in selected_rows]))
        refit_mean = float(np.mean([row["selectedBasis"]["transferBits"] for row in selected_rows]))
        beats = sum(row["selectedBasis"]["transferBits"] > row["previousBasis"]["transferBits"] for row in selected_rows)
        discovery = sum(
            row["status"] == "confirmed-early"
            and row["selectedBasis"]["transferBits"] > row["previousBasis"]["transferBits"]
            and any(feature.get("scope") == "replicated cross-market inventory" for feature in row["selectedBasis"].get("featureDetails", []))
            for row in selected_rows
        )
        lines.append(
            f"| {horizon} | {previous_mean:.6f} | {refit_mean:.6f} | "
            f"{refit_mean - previous_mean:+.6f} | {beats}/{len(selected_rows)} | {discovery}/{len(selected_rows)} |"
        )
    lines.extend([
        "",
        "The last column is a discovery shortlist, not a production promotion: looking at the transfer week to identify those rows consumes it for that decision, so they require a new later holdout or rolling-window confirmation.",
        "",
        "Among the four explicit additions, ETH appears repeatedly and supplies the only positive-transfer benchmark-beating leads; SOL is never selected, while the selected XRP and HYPE rows do not beat their previous bases on transfer.",
        "",
        "## Globally refitted component bases",
        "",
        "`positive vs marginal` means the refit has positive total and both half-block scores versus the no-feature marginal predictor. It does **not** mean it beats the previous basis; use the final change column for that comparison.",
        "",
        "| Horizon | BTC target component | Transfer validation | Previous basis (transfer bits) | Refitted extended basis | Primary bits | Transfer bits | Change vs previous transfer |",
        "|---|---|---|---|---|---:|---:|---:|",
    ])
    for row in artifact["targets"]:
        selected = row["selectedBasis"]
        previous = row["previousBasis"]
        validation = "positive vs marginal" if row["status"] == "confirmed-early" else "failed transfer"
        lines.append(
            f"| {row['horizonId']} | {escape(row['componentLabel'])} | {validation} | "
            f"{escape(', '.join(previous['features']) or 'none')} ({previous['transferBits']:.6f}) | "
            f"{escape(', '.join(selected['features']) or 'none')} | "
            f"{selected['primaryBits']:.6f} | {selected['transferBits']:.6f} | "
            f"{selected['transferBits'] - previous['transferBits']:+.6f} |"
        )
    lines.extend([
        "",
        "## Exact selected input contract",
        "",
        "Each row is one input to one prediction head. The delay is part of the contract; values must never be joined earlier than that boundary.",
        "",
        "| Horizon | Target component | Input | Asset/scope | Family | Lookback | Delay | Source availability score |",
        "|---|---|---|---|---|---|---|---:|",
    ])
    for row in artifact["targets"]:
        for feature in row["selectedBasis"].get("featureDetails", []):
            lines.append(
                f"| {row['horizonId']} | {escape(row['componentLabel'])} | {escape(feature['id'])} | "
                f"{escape(feature['asset'])} | {escape(feature['family'])} | {escape(feature['lookback'])} | "
                f"{escape(feature.get('delay', 'see catalog'))} | {feature.get('availabilityScore', 0):.2f} |"
            )
    lines.extend([
        "",
        "## Aggregate held-out comparison",
        "",
        "| Horizon | Heads | Positive-vs-marginal heads | Heads using cross-market input | Mean previous transfer bits | Mean refit transfer bits | Mean change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in ("1s", "1m", "15m", "1h"):
        selected_rows = [row for row in artifact["targets"] if row["horizonId"] == horizon]
        previous_mean = float(np.mean([row["previousBasis"]["transferBits"] for row in selected_rows]))
        refit_mean = float(np.mean([row["selectedBasis"]["transferBits"] for row in selected_rows]))
        cross_heads = sum(any(
            feature.get("scope") == "replicated cross-market inventory"
            for feature in row["selectedBasis"].get("featureDetails", [])
        ) for row in selected_rows)
        confirmed = sum(row["status"] == "confirmed-early" for row in selected_rows)
        lines.append(
            f"| {horizon} | {len(selected_rows)} | {confirmed} | {cross_heads} | "
            f"{previous_mean:.6f} | {refit_mean:.6f} | {refit_mean - previous_mean:+.6f} |"
        )
    lines.extend([
        "",
        "## Feature-family coverage",
        "",
        "| Examined family | Coordinates |",
        "|---|---:|",
    ])
    for family, count in sorted(artifact["search"]["examinedByFamily"].items()):
        lines.append(f"| {escape(family)} | {count:,} |")
    lines.extend([
        "",
        "## Canonical asset universe",
        "",
        "This is the latest 257-asset independent 1m basis plus the four explicit additions. A blank rank marks an explicit addition rather than a member of the source basis.",
        "",
        "| Rank | Asset | Explicit addition | Preferred market | Available products |",
        "|---:|---|---|---|---|",
    ])
    for asset in universe.get("assets", []):
        preferred = asset.get("preferredMarket") or {}
        products = ", ".join(f"{market['venue']}:{market['symbol']}" for market in asset["markets"])
        lines.append(
            f"| {asset['rank'] if asset['rank'] is not None else ''} | {escape(asset['asset'])} | "
            f"{'yes' if asset['explicit'] else 'no'} | {escape(preferred.get('venue', 'none'))} | {escape(products)} |"
        )
    lines.extend([
        "",
        "## Availability boundary",
        "",
        "The per-asset replication covers official completed Binance spot/USD-M klines, spot 1-second fast state, USD-M 5-minute positioning/open-interest metrics, settled funding, paired-venue basis/lead-lag, and percentage-depth snapshots wherever each source exists. Historical order additions/cancellations, liquidation events, exact aggregate-trade sequence/size shape, options surfaces, chain-specific flows, and non-Binance books are not uniformly available for this universe and are not fabricated; the corresponding BTC-only/live audits remain explicit separate tiers.",
        "",
        "The near-tie rule prefers reproducible candle/archive sources over local/live-only feeds and incorporates observed 30-day coverage. It does not manufacture pre-listing history: an asset-specific coordinate is unavailable before that market was listed. This is another reason the recent cross-asset winners remain discovery candidates and the older established production basis is retained.",
        "",
        "All candidate marginals are scored; joint subset enumeration is exact only inside each head's 16 finalists and size-three limit. Transfer data never chooses candidates or subsets. A `primary-only` row is discovery evidence, not a production promotion.",
        "",
        f"Machine-readable catalog and results: `{display(DEFAULT_OUTPUT)}`.",
        "",
    ])
    return "\n".join(lines)


def escape(value: str) -> str:
    return value.replace("|", "\\|")


def display(file: Path) -> str:
    try:
        return str(file.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(file.resolve())


if __name__ == "__main__":
    main()
