from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analyze_global_feature_basis import (
    chronological_blocks,
    fit_unconditional,
    nonoverlapping_rows,
    quantize_features,
    quantize_target,
    score_subset,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/runtime-cache/global-feature-basis-30d"
MACRO = ROOT / "data/market/mutable/external/global-macro-state.json"
OUTPUT = ROOT / "data/benchmarks/macro-production-basis-additions-30d.json"
REPORT = ROOT / "docs/experiments/macro-production-basis-additions-30d-2026-08-18.md"


@dataclass(frozen=True)
class MacroCandidate:
    id: str
    name: str
    series: str
    lag: int
    absolute: bool = False


CANDIDATES = (
    MacroCandidate("euro-estr-change-5", "Euro short-term rate (€STR) change", "ECB_ESTR", 5),
    MacroCandidate("euro-2y-absolute-change-5", "Euro-area AAA 2-year yield absolute change", "ECB_YC_2Y", 5, True),
    MacroCandidate("euro-2y-change-1", "Euro-area AAA 2-year yield change", "ECB_YC_2Y", 1),
    MacroCandidate("euro-2y-absolute-change-1", "Euro-area AAA 2-year yield absolute change", "ECB_YC_2Y", 1, True),
    MacroCandidate("uk-industrial-change-3", "United Kingdom industrial production index change", "OECD_INDUSTRIAL_PRODUCTION_GBR", 3),
    MacroCandidate("china-cpi-absolute-change-1", "China CPI year-over-year absolute change", "OECD_CPI_YOY_CHN", 1, True),
    MacroCandidate("us-2y-absolute-change-21", "US 2-year Treasury yield absolute change", "DGS2", 21, True),
    MacroCandidate("us-2y-absolute-change-1", "US 2-year Treasury yield absolute change", "DGS2", 1, True),
    MacroCandidate("us-curve-change-1", "US 10-year minus 2-year yield spread change", "T10Y2Y", 1),
)


BASES: dict[str, dict[str, tuple[str, ...]]] = {
    "1m": {
        "production": ("range-1m", "realized-volatility-15m", "realized-volatility-60m"),
        "recent-broad": ("realized-volatility-60m", "futures-log-trade-count-1m"),
    },
    "15m": {
        "production": ("realized-volatility-15m", "realized-volatility-60m", "realized-volatility-240m"),
        "recent-broad": ("realized-volatility-240m",),
    },
    "1h": {
        "production": ("realized-volatility-30m", "realized-volatility-240m"),
    },
}


def main() -> None:
    manifest = json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))
    dataset = manifest["datasets"][0]
    definitions = dataset["features"]
    feature_by_id = {definition["id"]: index for index, definition in enumerate(definitions)}
    raw_features = np.memmap(
        INPUT / dataset["files"]["features"], dtype="<f4", mode="r",
        shape=(dataset["rows"], dataset["featureCount"]),
    )
    raw_targets = np.memmap(
        INPUT / dataset["files"]["targets"], dtype="<f4", mode="r",
        shape=(dataset["rows"], dataset["targetCount"]),
    )
    splits = np.memmap(INPUT / dataset["files"]["splits"], dtype="u1", mode="r", shape=(dataset["rows"],))
    times = np.memmap(INPUT / dataset["files"]["times"], dtype="<f8", mode="r", shape=(dataset["rows"],))
    macro_artifact = json.loads(MACRO.read_text(encoding="utf-8"))
    macro_values, macro_metadata = build_macro_values(np.asarray(times), macro_artifact["rows"])

    target_by_id = {target["id"]: (index, target) for index, target in enumerate(dataset["targets"])}
    results: list[dict[str, Any]] = []
    for horizon, bases in BASES.items():
        target_index, target = target_by_id[horizon]
        selected_rows = nonoverlapping_rows(np.asarray(times), np.asarray(splits), target["minutes"] * 60_000)
        selected_splits = np.asarray(splits[selected_rows], dtype=np.uint8)
        train = selected_splits == 0
        primary = selected_splits == 1
        transfer = selected_splits == 2
        target_bins, target_edges, target_classes, zero_separate = quantize_target(
            np.asarray(raw_targets[selected_rows, target_index], dtype=np.float64), train,
        )
        unconditional = fit_unconditional(target_bins[train], target_classes)
        blocks = chronological_blocks(primary, transfer)
        for basis_name, basis_ids in bases.items():
            basis_indices = [feature_by_id[item] for item in basis_ids]
            combined_values = np.column_stack((
                np.asarray(raw_features[selected_rows][:, basis_indices], dtype=np.float64),
                macro_values[selected_rows],
            ))
            combined_definitions = [definitions[index] for index in basis_indices] + macro_metadata
            quantized, edges, arities = quantize_features(combined_values, train, combined_definitions)
            basis_subset = tuple(range(len(basis_indices)))
            basis_score = score_subset(
                basis_subset, quantized, target_bins, train, primary, transfer, blocks,
                arities, target_classes, unconditional,
            )
            candidates = []
            for candidate_index, candidate in enumerate(CANDIDATES, start=len(basis_indices)):
                full_score = score_subset(
                    basis_subset + (candidate_index,), quantized, target_bins, train, primary, transfer, blocks,
                    arities, target_classes, unconditional,
                )
                conditional_blocks = [
                    full - base for full, base in zip(full_score["blockBits"], basis_score["blockBits"], strict=True)
                ]
                candidates.append({
                    "id": candidate.id,
                    "name": candidate.name,
                    "series": candidate.series,
                    "lagObservations": candidate.lag,
                    "absolute": candidate.absolute,
                    "arity": arities[candidate_index],
                    "quantileEdges": edges[candidate_index],
                    "conditionalPrimaryBits": full_score["primaryBits"] - basis_score["primaryBits"],
                    "conditionalTransferBits": full_score["transferBits"] - basis_score["transferBits"],
                    "conditionalBlockBits": conditional_blocks,
                    "survives": all(value > 0 for value in conditional_blocks),
                })
            candidates.sort(key=lambda row: (
                row["survives"], min(row["conditionalBlockBits"]), row["conditionalPrimaryBits"]
            ), reverse=True)
            results.append({
                "horizon": horizon,
                "basis": basis_name,
                "basisFeatures": list(basis_ids),
                "observations": {
                    "train": int(train.sum()), "primary": int(primary.sum()), "transfer": int(transfer.sum()),
                },
                "targetEdges": list(target_edges),
                "zeroSeparate": zero_separate,
                "basisScore": basis_score,
                "candidates": candidates,
            })

    artifact = {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "objective": "Conditional macro information after fixed production and recent broad return-distribution bases",
        "split": manifest["split"],
        "method": {
            "model": "Same smoothed categorical histogram and training-only quantile encoding as the global feature-basis search",
            "selection": "The fixed basis is never replaced; each macro coordinate is appended separately",
            "survival": "Positive conditional bits in both primary halves and both untouched transfer halves",
            "warning": "The 30-day split has only 16 train, 7 primary, and 7 transfer calendar days; intraday targets sharing a macro state are not independent macro releases",
        },
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


def build_macro_values(times: np.ndarray, rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row)
    output = np.empty((times.shape[0], len(CANDIDATES)), dtype=np.float64)
    definitions: list[dict[str, Any]] = []
    for column, candidate in enumerate(CANDIDATES):
        series = sorted(grouped[candidate.series], key=lambda row: row["availableAt"])
        available = np.asarray([row["availableAt"] for row in series], dtype=np.float64)
        values = np.asarray([row["value"] for row in series], dtype=np.float64)
        indices = np.searchsorted(available, times, side="right") - 1
        previous = indices - candidate.lag
        valid = previous >= 0
        changes = np.full(times.shape[0], np.nan, dtype=np.float64)
        changes[valid] = values[indices[valid]] - values[previous[valid]]
        if candidate.absolute:
            changes = np.abs(changes)
        output[:, column] = changes
        definitions.append({
            "id": candidate.id,
            "name": candidate.name,
            "family": "global macro",
            "parameters": f"series={candidate.series},lag={candidate.lag},absolute={candidate.absolute}",
            "lookback": f"{candidate.lag} released observation(s)",
            "delay": "latest conservatively available release",
            "kind": "continuous",
        })
    return output, definitions


def render_report(artifact: dict[str, Any]) -> str:
    focus = {
        "euro-estr-change-5", "euro-2y-absolute-change-5", "euro-2y-change-1",
        "euro-2y-absolute-change-1", "uk-industrial-change-3",
        "china-cpi-absolute-change-1", "us-2y-absolute-change-21",
        "us-2y-absolute-change-1", "us-curve-change-1",
    }
    lines = [
        "# Macro additions after the fixed feature basis — recent 30-day screen",
        "",
        f"Generated {artifact['generatedAt']}.",
        "",
        "The previously reported macro gains were conditioned only on trailing return and volatility. This test freezes the production or recent broad basis and appends one macro coordinate at a time.",
        "",
        "| horizon | fixed basis | macro candidate | survives 4/4 | conditional primary bits | conditional transfer bits | primary halves | transfer halves |",
        "|---:|---|---|:---:|---:|---:|---:|---:|",
    ]
    for result in artifact["results"]:
        for candidate in result["candidates"]:
            if candidate["survives"] or candidate["id"] in focus:
                block = candidate["conditionalBlockBits"]
                lines.append(
                    f"| {result['horizon']} | {result['basis']} | {candidate['name']} | "
                    f"{'yes' if candidate['survives'] else 'no'} | {candidate['conditionalPrimaryBits']:.6f} | "
                    f"{candidate['conditionalTransferBits']:.6f} | {block[0]:.6f}, {block[1]:.6f} | "
                    f"{block[2]:.6f}, {block[3]:.6f} |"
                )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A macro feature survives only if its appended model beats the identical fixed-basis model in all four chronological blocks. Positive standalone macro bits are not enough.",
        "",
        f"Warning: {artifact['method']['warning']} Current revised macro observations are not vintage-correct.",
        "",
        "Complete candidate results and frozen quantile edges are in `data/benchmarks/macro-production-basis-additions-30d.json`.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
