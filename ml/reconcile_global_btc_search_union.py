from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from global_feature_registry import ROOT
from search_global_btc_working_set import template_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute derived union fields in a BTC search artifact.")
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    source = args.search if args.search.is_absolute() else ROOT / args.search
    output = args.output or source
    if not output.is_absolute():
        output = ROOT / output
    artifact = json.loads(source.read_text(encoding="utf-8"))
    registry = json.loads(
        (ROOT / "data/benchmarks/global-feature-registry.json").read_text(encoding="utf-8")
    )
    recommended: set[str] = set()
    exploratory: set[str] = set()
    metadata = {}
    for horizon in artifact["horizons"]:
        final = horizon["final"]
        exploratory.update(str(value) for value in final["selectedRawInputIds"])
        selected = (
            final["selectedRawInputIds"]
            if final.get("productionEligible")
            else final["requiredIncumbentInputs"]
        )
        recommended.update(str(value) for value in selected)
        metadata.update({str(row["id"]): row for row in final.get("support", [])})
    policies = []
    for feature_id in recommended:
        row = metadata.get(feature_id)
        policies.append(row if row is not None else template_policy(registry, feature_id))
    artifact["union"] = {
        **artifact.get("union", {}),
        "rawInputCount": len(recommended),
        "rawInputIds": sorted(recommended),
        "exploratoryRawInputCountBeforeTransferConfirmation": len(exploratory),
        "exploratoryRawInputIdsBeforeTransferConfirmation": sorted(exploratory),
        "availabilityMinimum": min(
            (float(row["availability"]) for row in policies), default=1.0
        ),
        "availabilityMean": float(np.mean([
            float(row["availability"]) for row in policies
        ])) if policies else 1.0,
        "acquisitionCostMaximum": max(
            (int(row["acquisitionCost"]) for row in policies), default=0
        ),
        "assetCounts": dict(sorted(Counter(
            feature_id.split("/")[1] for feature_id in recommended
        ).items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "rawInputCount": len(recommended),
        "assetCountSum": sum(artifact["union"]["assetCounts"].values()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
