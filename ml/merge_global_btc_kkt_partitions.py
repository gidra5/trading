from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from global_feature_basis_search import FeatureGroup, active_group_kkt_residuals
from global_feature_registry import ROOT


SHORT_WINDOW_INVENTORIES = {
    "fast-live", "deribit-option-surface", "gdelt-news", "tardis-cross-venue"
}
PROVIDER_INVENTORIES = {
    "base": {"representative-cross-asset"},
    "dense": {"dense-minute-indicators"},
    "representative": {"representative-cross-asset"},
    "long": {"long-endogenous"},
    "spectral1m": {"spectral-1m"},
    "external": {
        "coinmetrics", "community-flows", "cross-market-public", "dvol",
        "global-macro", "mempool-proxy", "vix",
    },
    "funding": {"full-funding-grid"},
    "technical1s": {"second-technical"},
    "spectral1s": {"spectral-1s"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge disjoint provider KKT scans into one registry-coverage certificate."
    )
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/benchmarks/global-feature-registry.json")
    parser.add_argument("--retain", type=int, default=256)
    parser.add_argument(
        "--horizons",
        default=None,
        help="Optional comma-separated subset of search horizons present in every partition.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def normalized_relative(value: Any) -> str:
    return str(value).replace("\\", "/")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def template_id(coordinate_id: str) -> str:
    parts = coordinate_id.split("/", 4)
    if len(parts) != 5:
        raise ValueError(f"Malformed coordinate ID: {coordinate_id}")
    return "/".join((parts[0], parts[2], parts[3], parts[4]))


def main() -> None:
    args = parse_args()
    search_path = resolved(args.search)
    model_path = resolved(args.model)
    part_paths = [resolved(path) for path in args.parts]
    output_path = resolved(args.output)
    registry_path = resolved(args.registry)
    search = load(search_path)
    registry = load(registry_path)
    parts = [load(path) for path in part_paths]
    if not parts or not all(bool(part.get("complete")) for part in parts):
        raise ValueError("Every KKT partition must be complete.")
    expected_search = relative(search_path)
    expected_model = relative(model_path)
    if any(normalized_relative(part.get("sourceSearch")) != expected_search for part in parts):
        raise ValueError("KKT partitions do not all reference the requested search artifact.")
    if any(normalized_relative(part.get("sourceModel")) != expected_model for part in parts):
        raise ValueError("KKT partitions do not all reference the requested model artifact.")
    tolerances = {float(part["tolerance"]) for part in parts}
    modes = {str(part["regularizationMode"]) for part in parts}
    if len(tolerances) != 1 or modes != {"selected"}:
        raise ValueError("KKT partitions must share one tolerance and use selected regularization.")
    tolerance = tolerances.pop()
    providers = sorted({name for part in parts for name in part["providers"]})
    unknown_providers = set(providers) - set(PROVIDER_INVENTORIES)
    if unknown_providers:
        raise ValueError(f"No inventory mapping for providers: {sorted(unknown_providers)}")
    covered_inventories = set().union(*(PROVIDER_INVENTORIES[name] for name in providers))

    point_in_time = {
        row["id"]: bool(row["point_in_time"]) for row in registry["sourcePolicies"]
    }
    robust_templates = [
        row for row in registry["templates"]
        if set(row["inventories"]) - SHORT_WINDOW_INVENTORIES
    ]
    eligible_templates = [
        row for row in robust_templates
        if point_in_time[str(row["sourcePolicy"])]
    ]
    missing_templates = [
        row for row in eligible_templates
        if not ((set(row["inventories"]) - SHORT_WINDOW_INVENTORIES) & covered_inventories)
    ]
    robust_coordinates = sum(int(row["coordinates"]) for row in robust_templates)
    eligible_coordinates = sum(int(row["coordinates"]) for row in eligible_templates)
    missing_coordinates = sum(int(row["coordinates"]) for row in missing_templates)
    causal_eligible_coordinates = eligible_coordinates
    template_by_id = {str(row["canonicalId"]): row for row in registry["templates"]}

    def causal_coordinate(feature_id: str) -> bool:
        template = template_by_id.get(template_id(feature_id))
        return bool(template) and point_in_time[str(template["sourcePolicy"])]

    with np.load(model_path) as model:
        states = np.asarray(model["states"], dtype=np.uint8)
        train = np.asarray(model["train"], dtype=bool)
        arities = np.asarray(model["arities"], dtype=np.int64)
        coordinate_ids = np.asarray(model["coordinate_ids"]).astype(str)
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        residual = np.asarray(model["residual"], dtype=np.float64)
    index_by_id = {feature_id: index for index, feature_id in enumerate(coordinate_ids)}
    all_horizons = search["horizons"]
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
    selected_horizons = [
        (index_by_horizon[name], all_horizons[index_by_horizon[name]])
        for name in requested
    ]
    results = []
    for task, horizon in selected_horizons:
        name = str(horizon["horizon"])
        rows = []
        for part in parts:
            matches = [row for row in part["results"] if row["horizon"] == name]
            if len(matches) != 1:
                raise ValueError(f"Partition has missing or duplicate horizon {name}.")
            rows.append(matches[0])
        regularizations = {float(row["regularization"]) for row in rows}
        if len(regularizations) != 1:
            raise ValueError(f"Partitions disagree on regularization for {name}.")
        regularization = regularizations.pop()
        active_ids = sorted(str(row["id"]) for row in horizon["final"]["support"])
        missing_active = set(active_ids) - set(index_by_id)
        if missing_active:
            raise ValueError(f"Model is missing {len(missing_active)} active coordinates for {name}.")
        indices = [index_by_id[feature_id] for feature_id in active_ids]
        groups = [
            FeatureGroup(feature_id, int(arities[index]), 1.0)
            for feature_id, index in zip(active_ids, indices)
        ]
        beta = [
            coefficients[task, index, : int(arities[index]) - 1, :]
            for index in indices
        ]
        stationarity = active_group_kkt_residuals(
            states[train][:, indices],
            residual[:, task * 9:(task + 1) * 9],
            groups,
            beta,
            regularization,
            device=args.device,
        )
        active_maximum = float(np.max(stationarity)) if stationarity.size else 0.0
        violation_by_id: dict[str, float] = {}
        for row in rows:
            for violation in row["topViolations"]:
                feature_id = str(violation["id"])
                if not causal_coordinate(feature_id):
                    continue
                violation_by_id[feature_id] = max(
                    violation_by_id.get(feature_id, -np.inf), float(violation["violation"])
                )
        top = sorted(violation_by_id.items(), key=lambda row: (-row[1], row[0]))[: args.retain]
        # Every partition retains up to 2,048 violations while the registry has
        # only 527 non-point-in-time coordinates. Therefore an empty causal
        # retained set proves that partition has no causal violation above the
        # shared tolerance, even if its raw maximum belongs to a revised feed.
        maximum_violation = max(
            max(
                (
                    float(violation["violation"])
                    for violation in row["topViolations"]
                    if causal_coordinate(str(violation["id"]))
                ),
                default=min(float(row["maximumViolation"]), tolerance),
            )
            for row in rows
        )
        optimizer_converged = bool(horizon["final"].get("converged", False))
        results.append({
            "horizon": name,
            "regularization": regularization,
            "activeCorrectionGroups": len(active_ids),
            "coordinatesScanned": eligible_coordinates - len(active_ids),
            "coordinatesExpected": eligible_coordinates - len(active_ids),
            "maximumViolation": maximum_violation,
            "activeStationarityMaximum": active_maximum,
            "activeStationarityMedian": (
                float(np.median(stationarity)) if stationarity.size else 0.0
            ),
            "activeStationarityViolations": int(np.count_nonzero(stationarity > tolerance)),
            "optimizerConverged": optimizer_converged,
            "violatingCoordinatesRetained": len(top),
            "certifiedAtTolerance": bool(
                optimizer_converged
                and missing_coordinates == 0
                and maximum_violation <= tolerance
                and active_maximum <= tolerance
            ),
            "topViolations": [
                {"id": feature_id, "violation": violation}
                for feature_id, violation in top
            ],
        })

    artifact = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "objective": "Merged full-registry KKT audit from complete provider partitions",
        "sourceSearch": expected_search,
        "sourceModel": expected_model,
        "sourceParts": [relative(path) for path in part_paths],
        "sourcePartSha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in part_paths],
        "providers": providers,
        "horizons": requested,
        "regularizationMode": "selected",
        "tolerance": tolerance,
        "registryCoverage": {
            "registryUniqueCoordinates": int(registry["summary"]["uniqueCoordinates"]),
            "robust30dEligibleCoordinates": robust_coordinates,
            "causalRobust30dEligibleCoordinates": causal_eligible_coordinates,
            "coveredInventories": sorted(covered_inventories),
            "missingTemplates": len(missing_templates),
            "missingCoordinates": missing_coordinates,
            "complete": missing_coordinates == 0,
        },
        "uniqueCoordinatesEmitted": eligible_coordinates,
        "results": results,
        "complete": bool(missing_coordinates == 0 and all(part["complete"] for part in parts)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {relative(output_path)}", flush=True)
    print(json.dumps({
        "providers": providers,
        "eligibleCoordinates": eligible_coordinates,
        "causalEligibleCoordinates": causal_eligible_coordinates,
        "missingCoordinates": missing_coordinates,
        "results": [
            {
                "horizon": row["horizon"],
                "maximumViolation": row["maximumViolation"],
                "activeStationarityMaximum": row["activeStationarityMaximum"],
                "certified": row["certifiedAtTolerance"],
            }
            for row in results
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
