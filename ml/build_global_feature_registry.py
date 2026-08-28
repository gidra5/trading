from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from global_feature_registry import ROOT, SOURCE_POLICIES, build_registry, source_feature_inventory


DEFAULT_OUTPUT = ROOT / "data/benchmarks/global-feature-registry.json"
DEFAULT_REPORT = ROOT / "docs/experiments/global-feature-registry-2026-08-20.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical global feature registry.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry, assets = build_registry()
    summary = registry.summary()
    source_inventory = source_feature_inventory(assets)
    artifact: dict[str, Any] = {
        "version": 2,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Canonicalize every examined BTC predictor coordinate before the incumbent-aware "
            "global feature-basis search"
        ),
        "identity": {
            "coordinateFields": ["subject-kind", "subject", "venue/instrument", "cadence", "formula-and-parameters"],
            "templateFields": ["subject-kind", "venue/instrument", "cadence", "formula-and-parameters"],
            "subjectBinding": "A canonical template expands only over assets with the required source coverage.",
            "deduplication": "Aliases with the same canonical identity and subject are one model coordinate.",
            "levelSemantics": {
                "basicCandidate": "An atomic causally observed category or directly published source level already exposed as a candidate coordinate.",
                "derivedCandidate": "A candidate coordinate deterministically transformed from one or more source fields.",
                "sourceInventory": "All source-field definitions and their entity bindings are counted separately from candidate coordinates.",
            },
        },
        "selectionPolicy": {
            "objectiveOrder": [
                "maximum nested chronological validation information in bits per eligible target",
                "reject any path more than 0.001 bits below the incumbent on any chronological fold",
                "within the paired one-standard-error set of the best surviving path, maximum minimum source availability",
                "then minimum acquisition cost",
                "then minimum subset size",
            ],
            "incumbentPerFoldToleranceBits": 0.001,
            "qualityEquivalenceRule": "paired one-standard-error rule on chronological-fold gain over incumbent",
            "incumbentAlwaysAdmissible": True,
            "transferUse": "final confirmation only; never finalist or subset selection",
        },
        "sourcePolicies": [vars(row) for row in SOURCE_POLICIES.values()],
        "universe": {
            "assets": assets["universe"],
            "count": len(assets["universe"]),
            "availabilitySets": {key: value for key, value in assets.items() if key != "universe"},
            "availabilityCounts": {key: len(value) for key, value in assets.items() if key != "universe"},
        },
        "summary": summary,
        "sourceFeatureInventory": source_inventory,
        "templates": [
            {
                "canonicalId": row.canonical_id,
                "level": row.level,
                "family": row.family,
                "cadence": row.cadence,
                "sourcePolicy": row.source_policy,
                "subjects": sorted(row.subjects),
                "coordinates": row.coordinates,
                "inventories": sorted(row.inventories),
                "aliases": sorted(row.aliases),
                "construction": row.construction,
                "lookback": row.lookback,
                "delay": row.delay,
            }
            for row in sorted(registry.templates.values(), key=lambda item: item.canonical_id)
        ],
        "limitations": [
            "A registry coordinate is a candidate, not a recommendation to pass every coordinate to one model.",
            "Unavailable asset/source combinations are absent rather than represented by permanently missing columns.",
            "The registry deduplicates declared formula aliases; KKT expansion coalesces exact gradient-equivalent candidates so synchronized copies cannot crowd out distinct signals.",
            "Exact exhaustive enumeration of every subset is computationally impossible at this dimension; the search must state its model class and proof boundary explicitly.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {args.output.relative_to(ROOT)}")
    print(f"Wrote {args.report.relative_to(ROOT)}")
    print(json.dumps(summary, indent=2))


def render_report(artifact: dict[str, Any]) -> str:
    summary = artifact["summary"]
    inventory_counts = {
        row["id"]: int(row["rawCoordinates"])
        for row in summary["inventories"]
    }
    grouped_inventory_ids = {
        "dense-minute-indicators",
        "second-technical",
        "spectral-1m",
        "spectral-1s",
        "long-endogenous",
        "representative-cross-asset",
        "full-funding-grid",
    }
    other_external = sum(
        count
        for inventory, count in inventory_counts.items()
        if inventory not in grouped_inventory_ids
    )
    lines = [
        "# Canonical global feature registry",
        "",
        f"Generated `{artifact['generatedAt']}`.",
        "",
        "## Result",
        "",
        f"The registry covers **{artifact['universe']['count']} assets** and contains "
        f"**{summary['rawCoordinates']:,} raw ledger coordinates**. Canonical identity removes "
        f"**{summary['duplicateLedgerCoordinates']:,} declared duplicate aliases**, leaving "
        f"**{summary['uniqueCoordinates']:,} unique candidate coordinates** represented by "
        f"**{summary['canonicalTemplates']:,} compact templates**.",
        "",
        "The exact raw arithmetic is:",
        "",
        "$$",
        f"{inventory_counts['dense-minute-indicators']:,}+"
        f"{inventory_counts['second-technical']:,}+"
        f"{inventory_counts['spectral-1m']:,}+"
        f"{inventory_counts['spectral-1s']:,}+"
        f"{inventory_counts['long-endogenous']:,}+"
        f"{inventory_counts['representative-cross-asset']:,}+"
        f"{inventory_counts['full-funding-grid']:,}+"
        f"{other_external:,}={summary['rawCoordinates']:,}.",
        "$$",
        "",
        "Those terms are, respectively, dense 1m indicators, 1s technical indicators, "
        "1m spectral features, 1s spectral features, long endogenous features, the "
        "representative cross-asset catalog, the funding grid, and all other external "
        "inventories. They use different availability sets: for example, dense and 1m "
        "spectral features bind to 257 assets, while 1s technical and spectral features "
        "bind to 140. The 147 existing coordinates are already part of the 31,043-coordinate "
        "representative catalog, and the 3,471 dense variants already include their lag "
        "grid. Therefore `261 × (147 + 3,471 + ...)` is not a valid count.",
        "",
        f"After canonical deduplication, the {summary['uniqueCoordinates']:,} coordinates "
        f"split into **{summary['assetSpecificCoordinates']:,} asset-specific** and "
        f"**{summary['generalCoordinates']:,} general** coordinates. Of these, "
        f"**{summary['causalRobust30dCoordinates']:,}** are both point-in-time safe and "
        "supported by the robust 30-day history used for production search; "
        f"{summary['shortWindowCoordinates']:,} short live-only coordinates and "
        f"{summary['nonPointInTimeRobust30dCoordinates']:,} non-point-in-time/revised "
        "coordinates remain documented but excluded from that optimization.",
        "",
        "The independent scope and candidate-construction classifications intersect as follows. "
        "Templates are feature definitions; coordinates are their concrete subject bindings:",
        "",
        "| Scope | Basic candidate templates | Basic expanded coordinates | Derived candidate templates | Derived expanded coordinates |",
        "|---|---:|---:|---:|---:|",
        f"| Asset-specific | {summary['basicAssetSpecificCandidateTemplates']:,} | "
        f"{summary['basicAssetSpecificCandidateCoordinates']:,} | "
        f"{summary['derivedAssetSpecificCandidateTemplates']:,} | "
        f"{summary['derivedAssetSpecificCandidateCoordinates']:,} |",
        f"| General | {summary['basicGeneralCandidateTemplates']:,} | "
        f"{summary['basicGeneralCandidateCoordinates']:,} | "
        f"{summary['derivedGeneralCandidateTemplates']:,} | "
        f"{summary['derivedGeneralCandidateCoordinates']:,} |",
        "",
        "These are candidate counts, not the complete source-field inventory below. For example, "
        "`funding-rate` is one asset-specific base field definition with 236 asset bindings; it is "
        "therefore one source feature type and 236 expanded coordinates, not 236 feature types.",
        "",
        "A coordinate identity is `(subject kind, subject, venue/instrument, cadence, formula + parameters)`. "
        "An asset is bound only when the required source is actually available; no pre-listing or permanently missing columns are invented.",
        "",
    ]
    source_inventory = artifact["sourceFeatureInventory"]
    lines.extend([
        "## Base/source feature inventory by origin",
        "",
        "Source fields are counted before derived candidate expansion. A field definition is counted once "
        "per source schema and cadence; the asset-binding count shows how many concrete asset fields it produces.",
        "",
        "| Origin category | Field set | Level | Cadence | Field definitions | Assets | Expanded asset-field bindings |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for category in source_inventory["assetSpecific"]["categories"][:3]:
        for field_set in category["fieldSets"]:
            lines.append(
                f"| {category['id']} | {field_set['id']} | {field_set['level']} | "
                f"{field_set['cadence']} | {field_set['fieldDefinitions']:,} | "
                f"{field_set['assetsAvailable']:,} | {field_set['expandedAssetFieldBindings']:,} |"
            )
    prediction = source_inventory["predictionMarket"]
    if prediction:
        fields = prediction["fieldSchemas"]
        asset_prediction = prediction["assetSpecific"]
        general_prediction = prediction["generalEvents"]
        lines.extend([
            "",
            "### Prediction-market base hierarchy",
            "",
            f"Kalshi `series` is the persistent feature identity. Events and contracts are dynamic instances. "
            f"At `{prediction['snapshot']}` (`{prediction['observationOrigin']}` causal model origin), "
            f"{asset_prediction['recognizedAssets']:,} recognized assets plus the separate "
            f"`{asset_prediction['unclassifiedOrMultiAssetBucket']}` bucket had prediction markets.",
            "",
            f"Each raw trade has **{fields['rawTrade']['baseFieldDefinitions']} base value fields**; each completed "
            f"minute candle has **{fields['completedMinuteCandle']['baseFieldDefinitions']} base value fields**. "
            f"The stored trade-state and candle-state records expose "
            f"**{fields['tradeState']['subfeatureDefinitions']}** and "
            f"**{fields['candleState']['subfeatureDefinitions']}** subfeatures respectively. The current axis "
            f"normalizes those to **{fields['normalizedForCurrentAxis']['subfeatureDefinitions']} nullable fields per observed contract**.",
            f"Each open contract also has **{fields['contractMetadata']['fieldDefinitions']} contract-definition fields**. "
            f"That is {asset_prediction['snapshot']['potentialContractMetadataFieldSlots']:,} potential asset-contract metadata "
            f"field slots and {general_prediction['snapshot']['potentialContractMetadataFieldSlots']:,} general-event slots at the snapshot; "
            "these identity/text/timing fields are nullable and are not all numeric model inputs.",
            "",
            "| Asset/bucket | Persistent series (30d) | Events (30d) | Contracts (30d) | Open series | Open events | Open contracts | Series-field channels | Open contract-field slots | Observed 1m/1s contracts | Observed 1m/1s field-slot upper bound |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in asset_prediction["byAsset"]:
            window = row["window"]
            snapshot = row["snapshot"]
            lines.append(
                f"| {row['asset']} | {window['persistentSeries']:,} | {window['events']:,} | "
                f"{window['contracts']:,} | {snapshot['openSeries']:,} | {snapshot['openEvents']:,} | "
                f"{snapshot['openContracts']:,} | {snapshot['persistentSeriesFieldChannels']:,} | "
                f"{snapshot['openContractFieldSlots']:,} | "
                f"{snapshot['observedContracts1m']:,}/{snapshot['observedContracts1s']:,} | "
                f"{snapshot['observedContractFieldSlotsUpperBound1m']:,}/"
                f"{snapshot['observedContractFieldSlotsUpperBound1s']:,} |"
            )
        window = asset_prediction["window"]
        snapshot = asset_prediction["snapshot"]
        lines.append(
            f"| **Asset total** | **{window['persistentSeries']:,}** | **{window['events']:,}** | "
            f"**{window['contracts']:,}** | **{snapshot['openSeries']:,}** | "
            f"**{snapshot['openEvents']:,}** | **{snapshot['openContracts']:,}** | "
            f"**{snapshot['persistentSeriesFieldChannels']:,}** | **{snapshot['openContractFieldSlots']:,}** | "
            f"**{snapshot['observedContracts1m']:,}/{snapshot['observedContracts1s']:,}** | "
            f"**{snapshot['observedContractFieldSlotsUpperBound1m']:,}/"
            f"{snapshot['observedContractFieldSlotsUpperBound1s']:,}** |"
        )
        lines.extend([
            "",
            "Global-event prediction markets use the same contract field schemas but are general features: ",
            "",
            "| Scope | Persistent series (30d) | Events (30d) | Contracts (30d) | Open series | Open events | Open contracts | Series-field channels | Open contract-field slots | Observed 1m/1s contracts | Observed 1m/1s field-slot upper bound |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| General/global events | {general_prediction['window']['persistentSeries']:,} | "
            f"{general_prediction['window']['events']:,} | {general_prediction['window']['contracts']:,} | "
            f"{general_prediction['snapshot']['openSeries']:,} | {general_prediction['snapshot']['openEvents']:,} | "
            f"{general_prediction['snapshot']['openContracts']:,} | "
            f"{general_prediction['snapshot']['persistentSeriesFieldChannels']:,} | "
            f"{general_prediction['snapshot']['openContractFieldSlots']:,} | "
            f"{general_prediction['snapshot']['observedContracts1m']:,}/"
            f"{general_prediction['snapshot']['observedContracts1s']:,} | "
            f"{general_prediction['snapshot']['observedContractFieldSlotsUpperBound1m']:,}/"
            f"{general_prediction['snapshot']['observedContractFieldSlotsUpperBound1s']:,} |",
            "",
            "Open contract-field slots are the dynamic contract fieldsets available in principle. Observed field-slot upper bounds count only "
            "contracts for which the sparse importer emitted a causal update at that origin, multiplied by seven; individual fields remain nullable. "
            "They are update counts, not feature counts.",
        ])
    lines.extend([
        "",
        "## Availability",
        "",
        "| Binding set | Assets |",
        "|---|---:|",
    ])
    for name, count in artifact["universe"]["availabilityCounts"].items():
        lines.append(f"| {name} | {count:,} |")
    lines.extend([
        "",
        "## Inventory reconciliation",
        "",
        "| Inventory | Raw coordinates | Note |",
        "|---|---:|---|",
    ])
    for row in summary["inventories"]:
        lines.append(f"| {row['id']} | {row['rawCoordinates']:,} | {row['note'].replace('|', '\\|')} |")
    lines.extend([
        "",
        "## Search contract",
        "",
        "Predictive quality is lexicographically first and the incumbent basis is always admissible. "
        "A path is discarded if it loses more than 0.001 bits per eligible target to the incumbent on "
        "any chronological fold. Among the survivors, paths inside the paired one-standard-error set "
        "of the best mean fold gain are treated as statistically equivalent; only then do broader "
        "availability, lower acquisition cost, and fewer features decide the result. The final transfer "
        "interval is confirmation only.",
        "",
        "The registry does not claim that enumerating every subset of this candidate universe is tractable. "
        "The search implementation must publish the exact model class, admissible subset space, pruning bounds, "
        "and whether its optimum is proven or approximate.",
        "",
        "## Important distinction",
        "",
        "These are candidate coordinates. A production input contract is the much smaller selected subset, not the full registry.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
