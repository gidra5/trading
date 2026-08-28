from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from global_feature_candidate_axis import (
    DenseMinuteBatchProvider,
    FundingGridBatchProvider,
    LongUniqueMinuteBatchProvider,
    OneSecondTechnicalBatchProvider,
    PublicExternalBatchProvider,
    RepresentativeCrossAssetBatchProvider,
    SpectralMinuteBatchProvider,
    SpectralSecondBatchProvider,
)
from global_feature_registry import ROOT, normalize_second_formula, safe_id


DEFAULT_BASE = ROOT / "data/runtime-cache/global-btc-working-set"
DEFAULT_OUTPUT = ROOT / "data/runtime-cache/global-btc-expanded-working-set"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize an incumbent-containing working set expanded by streamed KKT violations."
    )
    parser.add_argument("--base-working-set", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--kkt", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dense-active", type=int, default=512)
    parser.add_argument(
        "--retain-all-base",
        action="store_true",
        help="Retain every coordinate from the base working set during iterative KKT expansion.",
    )
    parser.add_argument("--retain-per-horizon", type=int, default=128)
    parser.add_argument(
        "--horizons",
        default="1s,1m,15m,1h",
        help="Comma-separated KKT horizons whose violations may expand the set.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def dense_id(feature_id: str) -> bool:
    parts = feature_id.split("/", 4)
    if len(parts) != 5 or parts[2:4] != ["binance-preferred", "1m"]:
        return False
    formula = parts[4]
    return formula.startswith(("rsi-", "ema-distance-", "ema-slope-", "ema-acceleration-"))


def main() -> None:
    args = parse_args()
    base_dir = resolved(args.base_working_set)
    output_dir = resolved(args.output)
    kkt_paths = [resolved(path) for path in args.kkt]
    base = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "data/benchmarks/global-feature-registry.json").read_text(encoding="utf-8"))
    template_by_id = {row["canonicalId"]: row for row in registry["templates"]}
    point_in_time_by_policy = {
        row["id"]: bool(row["point_in_time"]) for row in registry["sourcePolicies"]
    }

    def registered_template(feature_id: str) -> dict | None:
        parts = feature_id.split("/", 4)
        if len(parts) != 5:
            return None
        template_id = "/".join((parts[0], parts[2], parts[3], parts[4]))
        template = template_by_id.get(template_id)
        if not template or parts[1] not in {safe_id(str(value)) for value in template.get("subjects", [])}:
            return None
        return template

    def is_point_in_time(feature_id: str) -> bool:
        template = registered_template(feature_id)
        return bool(template) and point_in_time_by_policy.get(str(template["sourcePolicy"]), False)

    base_rows = base["coordinates"]
    iterative_base = args.retain_all_base or "baseCoordinatesRetained" in base
    if iterative_base:
        base_keep = list(range(len(base_rows)))
    else:
        base_keep = [
            index
            for index, row in enumerate(base_rows)
            if row["source"] == "existing-recent"
            or (
                row["source"] == "dense-minute"
                and int(row["denseGradientRank"]) <= args.dense_active
            )
        ]
    selected_by_horizon: dict[str, dict[str, float]] = defaultdict(dict)
    selected_horizons = {
        value.strip() for value in args.horizons.split(",") if value.strip()
    }
    excluded_non_point_in_time: dict[str, dict[str, float]] = defaultdict(dict)
    excluded_unknown_registry: dict[str, dict[str, float]] = defaultdict(dict)
    kkt_artifacts = []
    for path in kkt_paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not artifact.get("complete"):
            raise ValueError(f"KKT artifact is incomplete: {path}")
        kkt_artifacts.append(artifact)
        for result in artifact["results"]:
            if str(result["horizon"]) not in selected_horizons:
                continue
            for row in result["topViolations"][: args.retain_per_horizon]:
                feature_id = str(row["id"])
                if registered_template(feature_id) is None:
                    excluded_unknown_registry[str(result["horizon"])][feature_id] = float(row["violation"])
                    continue
                if not is_point_in_time(feature_id):
                    excluded_non_point_in_time[str(result["horizon"])][feature_id] = float(row["violation"])
                    continue
                selected_by_horizon[str(result["horizon"])][feature_id] = max(
                    selected_by_horizon[str(result["horizon"])].get(feature_id, -np.inf),
                    float(row["violation"]),
                )

    base_ids = [str(base_rows[index]["id"]) for index in base_keep]
    new_ids = set().union(*(set(rows) for rows in selected_by_horizon.values())) - set(base_ids)

    representative_probe = RepresentativeCrossAssetBatchProvider()
    representative_ids = {
        representative_probe.canonical_feature_id(row)
        for row in representative_probe.catalog.values()
    }
    spectral_probe = SpectralMinuteBatchProvider(limit_assets=0)
    spectral_formulas = {str(row["id"]) for row in spectral_probe.definitions}
    spectral_second_probe = SpectralSecondBatchProvider(limit_assets=0)
    spectral_second_formulas = {str(row["id"]) for row in spectral_second_probe.definitions}
    technical_second_probe = OneSecondTechnicalBatchProvider(limit_assets=0)
    technical_second_formulas = {
        normalize_second_formula(row) for row in technical_second_probe.definitions
    }
    external_probe = PublicExternalBatchProvider(selected_ids=set())
    external_ids = {
        external_probe.canonical_feature_id(row)
        for row in external_probe.manifest["features"]
    }
    funding_formulas = set(FundingGridBatchProvider(selected_ids=set()).formulas)
    long_formulas = set(LongUniqueMinuteBatchProvider.ASSET_FORMULAS)
    long_global = {"utc-hour-sin", "utc-hour-cos"}

    ids_by_source: dict[str, set[str]] = defaultdict(set)
    for feature_id in new_ids:
        parts = feature_id.split("/", 4)
        formula = parts[4] if len(parts) == 5 else ""
        if parts[2:4] == ["binance-spot", "1s"] and formula in technical_second_formulas:
            ids_by_source["technical-1s"].add(feature_id)
        elif parts[2:4] == ["binance-spot", "1s"] and formula in spectral_second_formulas:
            ids_by_source["spectral-1s"].add(feature_id)
        elif dense_id(feature_id):
            ids_by_source["dense-minute"].add(feature_id)
        elif feature_id in external_ids:
            ids_by_source["public-external"].add(feature_id)
        elif (
            parts[2:4] == ["binance-usdm", "funding-event"]
            and formula in funding_formulas
        ):
            ids_by_source["funding-grid"].add(feature_id)
        elif formula in spectral_formulas and parts[2:4] == ["binance-preferred", "1m"]:
            ids_by_source["spectral-1m"].add(feature_id)
        elif (
            (formula in long_formulas and parts[2:4] == ["binance-preferred", "1m"])
            or (formula in long_global and parts[1:4] == ["global", "calendar", "known"])
        ):
            ids_by_source["long-unique"].add(feature_id)
        elif feature_id in representative_ids:
            ids_by_source["representative-cross-asset"].add(feature_id)
        else:
            raise ValueError(f"No materializing provider recognizes {feature_id}")

    ordered_new = sorted(new_ids)
    ordered_ids = base_ids + ordered_new
    rows = int(base["rows"])
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "working-set.raw.f32.partial"
    final = output_dir / "working-set.raw.f32"
    target = np.memmap(partial, dtype="<f4", mode="w+", shape=(rows, len(ordered_ids)))
    source = np.memmap(
        base_dir / base["file"], dtype=base["dtype"], mode="r", shape=(rows, int(base["columns"]))
    )
    target[:, : len(base_keep)] = source[:, base_keep]
    index_by_id = {feature_id: index for index, feature_id in enumerate(ordered_ids)}

    providers = []
    if ids_by_source["dense-minute"]:
        providers.append(("dense-minute", DenseMinuteBatchProvider(
            batch_size=args.batch_size, selected_ids=ids_by_source["dense-minute"], progress=True
        )))
    if ids_by_source["representative-cross-asset"]:
        providers.append(("representative-cross-asset", RepresentativeCrossAssetBatchProvider(
            batch_size=args.batch_size,
            selected_ids=ids_by_source["representative-cross-asset"],
            progress=True,
        )))
    if ids_by_source["long-unique"]:
        providers.append(("long-unique", LongUniqueMinuteBatchProvider(
            batch_size=args.batch_size, selected_ids=ids_by_source["long-unique"], progress=True
        )))
    if ids_by_source["spectral-1m"]:
        providers.append(("spectral-1m", SpectralMinuteBatchProvider(
            batch_size=args.batch_size, selected_ids=ids_by_source["spectral-1m"], progress=True
        )))
    if ids_by_source["public-external"]:
        providers.append(("public-external", PublicExternalBatchProvider(
            batch_size=args.batch_size, selected_ids=ids_by_source["public-external"]
        )))
    if ids_by_source["funding-grid"]:
        providers.append(("funding-grid", FundingGridBatchProvider(
            batch_size=args.batch_size, selected_ids=ids_by_source["funding-grid"]
        )))
    if ids_by_source["technical-1s"]:
        providers.append(("technical-1s", OneSecondTechnicalBatchProvider(
            batch_size=args.batch_size, selected_ids=ids_by_source["technical-1s"], progress=True
        )))
    if ids_by_source["spectral-1s"]:
        providers.append(("spectral-1s", SpectralSecondBatchProvider(
            batch_size=args.batch_size, selected_ids=ids_by_source["spectral-1s"], progress=True
        )))

    started = time.perf_counter()
    seen: set[str] = set()
    for provider_name, provider in providers:
        expected = ids_by_source[provider_name]
        provider_seen: set[str] = set()
        for batch in provider.raw_batches():
            for column, group in enumerate(batch.groups):
                if group.id not in expected:
                    raise RuntimeError(f"{provider_name} emitted unexpected coordinate {group.id}")
                target[:, index_by_id[group.id]] = batch.values[:, column]
                provider_seen.add(group.id)
                seen.add(group.id)
        missing = expected - provider_seen
        if missing:
            preview = ", ".join(sorted(missing)[:20])
            raise RuntimeError(
                f"{provider_name} failed to reconstruct {len(missing)} coordinates: {preview}"
            )
    target.flush()
    del target
    del source
    if seen != new_ids:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Expanded reconstruction mismatch: {len(new_ids - seen)} missing")
    partial.replace(final)

    source_by_id = {
        feature_id: source_name
        for source_name, ids in ids_by_source.items()
        for feature_id in ids
    }
    evidence = {
        horizon: {
            feature_id: violation for feature_id, violation in rows_by_id.items()
        }
        for horizon, rows_by_id in selected_by_horizon.items()
    }
    coordinate_rows = [dict(base_rows[index]) for index in base_keep] + [
        {
            "id": feature_id,
            "source": source_by_id[feature_id],
            "kktViolations": {
                horizon: rows_by_id[feature_id]
                for horizon, rows_by_id in evidence.items()
                if feature_id in rows_by_id
            },
        }
        for feature_id in ordered_new
    ]
    manifest = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Incumbent-containing active set expanded by full-provider KKT violations",
        "baseWorkingSet": str(base_dir.relative_to(ROOT)),
        "baseWorkingSetSha256": hashlib.sha256((base_dir / "manifest.json").read_bytes()).hexdigest(),
        "sourceKktArtifacts": [str(path.relative_to(ROOT)) for path in kkt_paths],
        "sourceKktSha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in kkt_paths],
        "rows": rows,
        "columns": len(ordered_ids),
        "baseCoordinatesRetained": len(base_ids),
        "retainedAllBaseCoordinates": iterative_base,
        "newCoordinates": len(ordered_new),
        "newCoordinatesBySource": dict(sorted(Counter(source_by_id.values()).items())),
        "selectedViolationCoordinatesByHorizon": {
            horizon: len(rows_by_id) for horizon, rows_by_id in sorted(selected_by_horizon.items())
        },
        "excludedNonPointInTimeCoordinatesByHorizon": {
            horizon: len(rows_by_id) for horizon, rows_by_id in sorted(excluded_non_point_in_time.items())
        },
        "excludedNonPointInTimeCoordinates": {
            horizon: rows_by_id for horizon, rows_by_id in sorted(excluded_non_point_in_time.items())
        },
        "excludedUnknownRegistryCoordinatesByHorizon": {
            horizon: len(rows_by_id) for horizon, rows_by_id in sorted(excluded_unknown_registry.items())
        },
        "excludedUnknownRegistryCoordinates": {
            horizon: rows_by_id for horizon, rows_by_id in sorted(excluded_unknown_registry.items())
        },
        "file": final.name,
        "dtype": "<f4",
        "elapsedSeconds": time.perf_counter() - started,
        "coordinates": coordinate_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {final.relative_to(ROOT)}", flush=True)
    print(json.dumps({key: manifest[key] for key in (
        "rows", "columns", "baseCoordinatesRetained", "newCoordinates",
        "newCoordinatesBySource", "selectedViolationCoordinatesByHorizon", "elapsedSeconds",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
