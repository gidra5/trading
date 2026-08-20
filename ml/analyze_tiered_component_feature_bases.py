from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_live_component_feature_bases import analyze_component, components


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/runtime-cache/global-feature-basis-30d"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/tiered-component-feature-bases.json"
DEFAULT_REPORT = ROOT / "docs/experiments/tiered-component-feature-bases-2026-08-19.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizons", default="1s,1m,15m,1h")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def availability(definition: dict[str, Any]) -> dict[str, Any]:
    feature_id = str(definition["id"])
    family = str(definition.get("family", ""))
    if feature_id.startswith("spot-book-"):
        return {
            "availabilityClass": "local-snapshot-only",
            "availabilityScore": 0.45,
            "backfill": "not reproducible from the free exchange archive",
            "expectedCoverage": "only while the local collector is running",
        }
    if feature_id.startswith("gdelt-"):
        return {
            "availabilityClass": "public-rate-limited",
            "availabilityScore": 0.75,
            "backfill": "GDELT DOC API; retrospective publication timeline",
            "expectedCoverage": "broad history, but rate-limited and not first-observed news time",
        }
    if feature_id.startswith(("spot-flow-", "futures-", "open-interest-", "top-", "global-ratio-", "taker-ratio-")):
        return {
            "availabilityClass": "free-official-archive",
            "availabilityScore": 0.95,
            "backfill": "Binance public daily archives",
            "expectedCoverage": "multi-year; source-dependent listing date",
        }
    if feature_id.startswith(("eth-", "sol-", "bnb-", "doge-")):
        return {
            "availabilityClass": "free-official-archive",
            "availabilityScore": 0.96,
            "backfill": "Binance public kline archives",
            "expectedCoverage": "from each symbol's listing date",
        }
    if family in {
        "return history", "activity", "volatility", "price dynamics", "volume regime",
        "candle shape", "minute volatility", "minute candle shape",
    }:
        return {
            "availabilityClass": "core-candle-derived",
            "availabilityScore": 1.0,
            "backfill": "derived from Binance spot candles",
            "expectedCoverage": "full retained candle history",
        }
    return {
        "availabilityClass": "free-official-archive",
        "availabilityScore": 0.94,
        "backfill": "source archive used by the recent broad exporter",
        "expectedCoverage": "source-dependent",
    }


def normalize_definition(definition: dict[str, Any]) -> dict[str, Any]:
    result = dict(definition)
    result.setdefault("source", source_name(result))
    result.setdefault("construction", result.get("parameters", result.get("name", result["id"])))
    result.update(availability(result))
    return result


def source_name(definition: dict[str, Any]) -> str:
    feature_id = str(definition["id"])
    if feature_id.startswith("spot-book-"):
        return "local Binance spot book snapshots"
    if feature_id.startswith("spot-flow-"):
        return "Binance spot aggregate-trade archive"
    if feature_id.startswith(("futures-", "open-interest-", "top-", "global-ratio-", "taker-ratio-")):
        return "Binance USD-M archive"
    if feature_id.startswith(("eth-", "sol-", "bnb-", "doge-")):
        return "Binance spot kline archive"
    if feature_id.startswith("gdelt-"):
        return "GDELT DOC 2.0"
    return "Binance BTCUSDT spot candles"


def main() -> None:
    args = parse_args()
    if args.render_only:
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        args.report.write_text(render_report(artifact), encoding="utf-8")
        print(f"Wrote {display_path(args.report)}")
        return
    root_manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    requested = set(args.horizons.split(","))
    result_rows: list[dict[str, Any]] = []
    dataset_summaries = []
    for dataset in root_manifest["datasets"]:
        definitions = [normalize_definition(item) for item in dataset["features"]]
        rows = int(dataset["rows"])
        feature_count = int(dataset["featureCount"])
        target_count = int(dataset["targetCount"])
        raw_features = np.memmap(
            args.input_dir / dataset["files"]["features"], dtype="<f4", mode="r",
            shape=(rows, feature_count),
        )
        raw_targets = np.memmap(
            args.input_dir / dataset["files"]["targets"], dtype="<f4", mode="r",
            shape=(rows, target_count),
        )
        raw_times = np.memmap(
            args.input_dir / dataset["files"]["times"], dtype="<f8", mode="r", shape=(rows,),
        )
        raw_splits = np.memmap(
            args.input_dir / dataset["files"]["splits"], dtype="u1", mode="r", shape=(rows,),
        )
        features = np.asarray(raw_features, dtype=np.float64)
        times = np.asarray(raw_times, dtype=np.float64)
        if times.size > 1 and float(np.median(np.diff(times[: min(times.size, 1_000)]))) > 1_000:
            times = times / 1_000
        splits = np.asarray(raw_splits, dtype=np.uint8)
        analysis_manifest = {
            "features": definitions,
            "targetScaleToBps": 10_000,
        }
        dataset_summaries.append({
            "id": dataset["id"],
            "rows": rows,
            "features": feature_count,
            "availabilityClasses": availability_counts(definitions),
        })
        for target_index, target in enumerate(dataset["targets"]):
            if target["id"] not in requested:
                continue
            horizon_seconds = int(round(float(target["minutes"]) * 60))
            returns = np.asarray(raw_targets[:, target_index], dtype=np.float64) / 10_000
            active_train = np.abs(returns[(splits == 0) & (returns != 0)])
            thresholds = np.quantile(active_train, [0.25, 0.5, 0.75, 0.9])
            zeros = np.zeros(rows, dtype=np.float64)
            for component in components():
                print(f"Searching {dataset['id']} {target['id']} {component.id}...", flush=True)
                result = analyze_component(
                    analysis_manifest, features, returns, zeros, zeros, times, splits,
                    horizon_seconds, thresholds, component,
                )
                result["dataset"] = dataset["id"]
                result["horizonId"] = target["id"]
                result_rows.append(result)
    comparable = [row for row in result_rows if row.get("primaryOptimum") and row["selectedBasis"].get("features")]
    changed = [row for row in comparable if row["primaryOptimum"]["features"] != row["selectedBasis"]["features"]]
    sacrifices = [row["primaryOptimum"]["primaryBits"] - row["selectedBasis"]["primaryBits"] for row in changed]
    artifact = {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "objective": "Component-specific joint bases across the recent broad feature tier, with availability-aware near-tie selection",
        "metric": "held-out log2 likelihood gain versus the unconditional component distribution",
        "split": root_manifest["split"],
        "datasets": dataset_summaries,
        "selection": {
            "predictiveToleranceBits": 0.001,
            "availabilityTieBreak": "maximize the minimum feature availability score, then the mean score, then minimize subset size, then maximize primary bits",
            "confirmation": "features and subsets are selected without the final transfer interval; confirmation requires positive total and both transfer half-blocks",
            "scope": "finite histogram search over every feature marginal and subsets of up to three among 12 family-aware finalists per component",
            "availabilityTieBreakChangedSelections": len(changed),
            "availabilityTieBreakEligibleSelections": len(comparable),
            "meanPrimaryBitsSacrificedWhenChanged": float(np.mean(sacrifices)) if sacrifices else 0.0,
            "maximumPrimaryBitsSacrificedWhenChanged": max(sacrifices, default=0.0),
        },
        "targets": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {display_path(args.output)}")
    print(f"Wrote {display_path(args.report)}")


def availability_counts(definitions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for definition in definitions:
        key = str(definition["availabilityClass"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def render_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Tiered component feature bases",
        "",
        f"Generated `{artifact['generatedAt']}`.",
        "",
        "This search evaluates every coordinate in the recent broad tier separately for each return component. "
        "The final transfer block is not used for selection. Within 0.001 primary bits, broader availability wins.",
        "",
        "## Dataset and selection",
        "",
    ]
    for dataset in artifact["datasets"]:
        lines.append(
            f"- `{dataset['id']}`: {dataset['rows']:,} origins, {dataset['features']} coordinates; "
            + ", ".join(f"{key}={value}" for key, value in dataset["availabilityClasses"].items()) + "."
        )
    lines.extend([
        "",
        "The search is exact only inside each component's 12 family-aware finalists and subsets of size at most three; "
        "all broad coordinates are nevertheless scored marginally before finalist selection.",
        "",
        f"The availability rule changed {artifact['selection']['availabilityTieBreakChangedSelections']} of "
        f"{artifact['selection']['availabilityTieBreakEligibleSelections']} eligible selections. Its mean/max held-out "
        f"primary-score sacrifice was {artifact['selection']['meanPrimaryBitsSacrificedWhenChanged']:.6f}/"
        f"{artifact['selection']['maximumPrimaryBitsSacrificedWhenChanged']:.6f} bits per target.",
        "",
        "## Selected bases",
        "",
        "| Horizon | Component | Status | Selected inputs | Primary bits | Transfer bits | Availability min/mean |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for row in artifact["targets"]:
        basis = row["selectedBasis"]
        features = ", ".join(basis.get("features", [])) or "none"
        availability_row = basis.get("availabilityTieBreak", {})
        availability_text = "n/a" if not availability_row else (
            f"{availability_row['minimumScore']:.2f}/{availability_row['meanScore']:.2f}"
        )
        lines.append(
            f"| {row['horizonId']} | {escape(row['componentLabel'])} | {row['status']} | "
            f"{escape(features)} | {basis.get('primaryBits', 0):.6f} | "
            f"{basis.get('transferBits', 0):.6f} | {availability_text} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A selected live-snapshot feature is retained only when no archive-backed subset is within the predictive tolerance. "
        "`primary-only` is discovery evidence, not a production input. `confirmed-early` still covers only this recent regime.",
        "",
        f"Machine-readable results: `{display_path(DEFAULT_OUTPUT)}`.",
        "",
    ])
    return "\n".join(lines)


def escape(value: str) -> str:
    return value.replace("|", "\\|")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
