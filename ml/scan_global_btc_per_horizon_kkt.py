from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from global_feature_basis_search import (
    FeatureGroup,
    active_group_kkt_residuals,
    scan_quantized_batches_multitask_kkt,
)
from global_feature_candidate_axis import (
    BaseRecentBatchProvider,
    DenseMinuteBatchProvider,
    FundingGridBatchProvider,
    LongUniqueMinuteBatchProvider,
    OneSecondTechnicalBatchProvider,
    PredictionMarketBatchProvider,
    PublicExternalBatchProvider,
    QuantizedCandidateBatch,
    RepresentativeCrossAssetBatchProvider,
    SpectralMinuteBatchProvider,
    SpectralSecondBatchProvider,
)
from global_feature_registry import ROOT


DEFAULT_SEARCH = ROOT / "data/benchmarks/global-btc-per-horizon-working-set-search.json"
DEFAULT_MODEL = ROOT / "data/benchmarks/global-btc-per-horizon-working-set-model.npz"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/global-btc-per-horizon-kkt-rescan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-pass multi-horizon KKT rescan over global providers.")
    parser.add_argument("--search", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--providers",
        default="base,dense,representative,long,spectral1m,external,funding,technical1s,spectral1s,prediction",
    )
    parser.add_argument("--retain", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quantile-sample-rows", type=int, default=4_096)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--horizons",
        default=None,
        help="Optional comma-separated subset of search horizons to audit.",
    )
    parser.add_argument(
        "--regularization-mode", choices=("selected", "zero"), default="selected",
        help="Use selected-lambda KKT violations or retain top raw gradients for lambda-path discovery.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    search_path = resolved(args.search)
    model_path = resolved(args.model)
    output_path = resolved(args.output)
    search = json.loads(search_path.read_text(encoding="utf-8"))
    with np.load(model_path) as model:
        residual_all = np.asarray(model["residual"], dtype=np.float64)
        train = np.asarray(model["train"], dtype=bool)
        active_states = np.asarray(model["states"], dtype=np.uint8)[train]
        arities = np.asarray(model["arities"], dtype=np.int64)
        coordinate_ids = np.asarray(model["coordinate_ids"]).astype(str)
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    all_horizons = search["horizons"]
    if residual_all.shape != (int(np.count_nonzero(train)), len(all_horizons) * 9):
        raise ValueError("Saved per-horizon residual has an unexpected shape.")
    requested = (
        [value.strip() for value in args.horizons.split(",") if value.strip()]
        if args.horizons else [str(row["horizon"]) for row in all_horizons]
    )
    if len(requested) != len(set(requested)):
        raise ValueError("Requested horizons must be unique.")
    index_by_horizon = {
        str(row["horizon"]): index for index, row in enumerate(all_horizons)
    }
    unknown_horizons = set(requested) - set(index_by_horizon)
    if unknown_horizons:
        raise ValueError(f"Unknown horizons: {sorted(unknown_horizons)}")
    model_tasks = [index_by_horizon[name] for name in requested]
    horizons = [all_horizons[index] for index in model_tasks]
    residual = np.concatenate(
        [residual_all[:, task * 9:(task + 1) * 9] for task in model_tasks], axis=1
    )
    regularizations = tuple(
        0.0 if args.regularization_mode == "zero" else float(row["final"]["regularization"])
        for row in horizons
    )
    active_ids = tuple(
        {str(feature["id"]) for feature in row["final"]["support"]}
        for row in horizons
    )
    coordinate_index = {feature_id: index for index, feature_id in enumerate(coordinate_ids)}
    active_stationarity = []
    for task, (model_task, horizon, regularization) in enumerate(
        zip(model_tasks, horizons, regularizations)
    ):
        indices = [coordinate_index[feature_id] for feature_id in sorted(active_ids[task])]
        groups = [
            FeatureGroup(coordinate_ids[index], int(arities[index]), 1.0)
            for index in indices
        ]
        beta = [
            coefficients[model_task, index, : int(arities[index]) - 1, :]
            for index in indices
        ]
        values = active_group_kkt_residuals(
            active_states[:, indices],
            residual[:, task * 9:(task + 1) * 9],
            groups,
            beta,
            regularization,
            device=args.device,
        )
        active_stationarity.append(values)
    provider_names = [value.strip() for value in args.providers.split(",") if value.strip()]
    providers = []
    for name in provider_names:
        if name == "base":
            providers.append((name, BaseRecentBatchProvider()))
        elif name == "dense":
            providers.append((name, DenseMinuteBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
                progress=True,
            )))
        elif name == "representative":
            providers.append((name, RepresentativeCrossAssetBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
                progress=True,
            )))
        elif name == "long":
            providers.append((name, LongUniqueMinuteBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
                progress=True,
            )))
        elif name == "spectral1m":
            providers.append((name, SpectralMinuteBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
                progress=True,
            )))
        elif name == "external":
            providers.append((name, PublicExternalBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
            )))
        elif name == "funding":
            providers.append((name, FundingGridBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
            )))
        elif name == "technical1s":
            providers.append((name, OneSecondTechnicalBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
                progress=True,
            )))
        elif name == "spectral1s":
            providers.append((name, SpectralSecondBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
                progress=True,
            )))
        elif name == "prediction":
            providers.append((name, PredictionMarketBatchProvider(
                batch_size=args.batch_size,
                quantile_sample_rows=args.quantile_sample_rows,
            )))
        else:
            raise ValueError(f"Unknown provider: {name}")

    seen: set[str] = set()
    emitted_by_provider = {name: 0 for name, _ in providers}

    def batches():
        for name, provider in providers:
            for batch in provider.quantized_batches(train=train):
                keep = []
                for index, group in enumerate(batch.groups):
                    if group.id in seen:
                        continue
                    seen.add(group.id)
                    keep.append(index)
                if not keep:
                    continue
                emitted_by_provider[name] += len(keep)
                yield QuantizedCandidateBatch(
                    [batch.groups[index] for index in keep],
                    batch.states[train][:, keep],
                    [batch.edges[index] for index in keep],
                )

    started = time.perf_counter()
    scans = scan_quantized_batches_multitask_kkt(
        residual,
        batches,
        regularizations,
        active_ids,
        tuple(slice(task * 9, (task + 1) * 9) for task in range(len(horizons))),
        add_limit=args.retain,
        tolerance=args.tolerance,
        device=args.device,
    )
    elapsed = time.perf_counter() - started
    results = []
    for task, (horizon, scan, regularization) in enumerate(zip(horizons, scans, regularizations)):
        expected_scanned = len(seen - active_ids[task])
        stationarity = active_stationarity[task]
        active_maximum = float(np.max(stationarity)) if stationarity.size else 0.0
        results.append({
            "horizon": horizon["horizon"],
            "regularization": regularization,
            "activeCorrectionGroups": len(active_ids[task]),
            "coordinatesScanned": scan.scanned_groups,
            "coordinatesExpected": expected_scanned,
            "maximumViolation": scan.maximum_violation,
            "activeStationarityMaximum": active_maximum,
            "activeStationarityMedian": (
                float(np.median(stationarity)) if stationarity.size else 0.0
            ),
            "activeStationarityViolations": int(np.count_nonzero(stationarity > args.tolerance)),
            "violatingCoordinatesRetained": len(scan.violating_groups),
            "optimizerConverged": bool(horizon["final"].get("converged", False)),
            "certifiedAtTolerance": bool(
                horizon["final"].get("converged", False)
                and scan.maximum_violation <= args.tolerance
                and active_maximum <= args.tolerance
            ),
            "topViolations": [
                {"id": feature_id, "violation": violation}
                for feature_id, violation in scan.violating_groups
            ],
        })
    artifact = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "objective": "Full streamed KKT audit of the selected per-horizon additive corrections",
        "sourceSearch": str(search_path.relative_to(ROOT)).replace("\\", "/"),
        "sourceModel": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "providers": provider_names,
        "horizons": requested,
        "regularizationMode": args.regularization_mode,
        "providerCoordinatesBeforeDeduplication": {
            name: provider.coordinate_count for name, provider in providers
        },
        "providerUniqueCoordinatesEmitted": emitted_by_provider,
        "uniqueCoordinatesEmitted": len(seen),
        "quantization": {
            "bins": 4,
            "deterministicTrainingSampleRows": args.quantile_sample_rows,
            "note": "Excluded coordinates use the predeclared approximate-quantile partition; active coordinates retain their exact training quantiles.",
        },
        "tolerance": args.tolerance,
        "results": results,
        "elapsedSeconds": elapsed,
        "complete": all(row["coordinatesScanned"] == row["coordinatesExpected"] for row in results),
    }
    registry = json.loads((ROOT / "data/benchmarks/global-feature-registry.json").read_text(encoding="utf-8"))
    short_window_inventories = {
        "fast-live", "deribit-option-surface", "gdelt-news", "tardis-cross-venue"
    }
    eligible_coordinates = sum(
        int(row["coordinates"])
        for row in registry["templates"]
        if set(row["inventories"]) - short_window_inventories
    )
    point_in_time_by_policy = {
        row["id"]: bool(row["point_in_time"]) for row in registry["sourcePolicies"]
    }
    causal_eligible_coordinates = sum(
        int(row["coordinates"])
        for row in registry["templates"]
        if set(row["inventories"]) - short_window_inventories
        and point_in_time_by_policy[str(row["sourcePolicy"])]
    )
    artifact["registryCoverage"] = {
        "registryUniqueCoordinates": int(registry["summary"]["uniqueCoordinates"]),
        "robust30dEligibleCoordinates": eligible_coordinates,
        "shortWindowCoordinatesExcluded": int(registry["summary"]["uniqueCoordinates"]) - eligible_coordinates,
        "shortWindowInventoriesExcluded": sorted(short_window_inventories),
        "causalRobust30dEligibleCoordinates": causal_eligible_coordinates,
        "nonPointInTimeCoordinatesExcludedFromProductionSearch": eligible_coordinates - causal_eligible_coordinates,
        "eligibleCoordinatesEmitted": len(seen),
        "eligibleCoverage": len(seen) / eligible_coordinates,
        "complete": len(seen) == eligible_coordinates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}", flush=True)
    print(json.dumps({
        "uniqueCoordinatesEmitted": artifact["uniqueCoordinatesEmitted"],
        "providerUniqueCoordinatesEmitted": emitted_by_provider,
        "results": [
            {
                "horizon": row["horizon"],
                "maximumViolation": row["maximumViolation"],
                "retained": row["violatingCoordinatesRetained"],
                "certified": row["certifiedAtTolerance"],
            }
            for row in results
        ],
        "elapsedSeconds": elapsed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
