from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from global_feature_basis_search import additive_logits, softmax
from global_feature_registry import ROOT
from search_global_btc_working_set import load_targets, nonoverlapping_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily and bootstrap robustness audit on untouched BTC transfer rows.")
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    search_path = resolved(args.search)
    model_path = resolved(args.model)
    output_path = resolved(args.output)
    search = json.loads(search_path.read_text(encoding="utf-8"))
    _, _, splits, times = load_targets()
    transfer = splits == 2
    with np.load(model_path) as model:
        states = np.asarray(model["states"], dtype=np.uint8)
        arities = np.asarray(model["arities"], dtype=np.int64)
        labels = np.asarray(model["labels"], dtype=np.int64)
        offsets = np.asarray(model["offsets"], dtype=np.float64)
        intercepts = np.asarray(model["intercepts"], dtype=np.float64)
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        train = np.asarray(model["train"], dtype=bool)
    random = np.random.default_rng(20260821)
    results = []
    recommended_ids: set[str] = set()
    for task, horizon in enumerate(search["horizons"]):
        coefficient_list = [
            coefficients[task, index, : max(0, int(arity) - 1), :]
            for index, arity in enumerate(arities)
        ]
        offset = offsets[:, task * 9:(task + 1) * 9]
        model_probability = softmax(
            offset + additive_logits(states, intercepts[task], coefficient_list)
        )
        incumbent_probability = softmax(offset)
        task_labels = labels[:, task]
        counts = np.bincount(task_labels[train], minlength=9).astype(np.float64) + 0.5
        baseline = counts / counts.sum()
        rows = nonoverlapping_rows(
            transfer, times, int(horizon["horizonMinutes"])
        )
        selected = model_probability[rows, task_labels[rows]]
        incumbent = incumbent_probability[rows, task_labels[rows]]
        unconditional = baseline[task_labels[rows]]
        model_bits = np.log2(np.maximum(selected, 1e-30) / np.maximum(unconditional, 1e-30))
        incumbent_bits = np.log2(np.maximum(incumbent, 1e-30) / np.maximum(unconditional, 1e-30))
        gain = model_bits - incumbent_bits
        days = times[rows].astype(np.int64) // 86_400_000
        unique_days = np.unique(days)
        daily = []
        daily_sums = []
        daily_counts = []
        for day in unique_days:
            mask = days == day
            daily_sums.append(float(gain[mask].sum()))
            daily_counts.append(int(np.count_nonzero(mask)))
            daily.append({
                "date": str(np.datetime64(int(day), "D")),
                "rows": daily_counts[-1],
                "modelBits": float(model_bits[mask].mean()),
                "incumbentBits": float(incumbent_bits[mask].mean()),
                "gainBits": float(gain[mask].mean()),
            })
        daily_sums_array = np.asarray(daily_sums)
        daily_counts_array = np.asarray(daily_counts)
        samples = random.integers(0, unique_days.size, (args.bootstrap_samples, unique_days.size))
        bootstrap = daily_sums_array[samples].sum(axis=1) / daily_counts_array[samples].sum(axis=1)
        bootstrap_interval = np.quantile(bootstrap, (0.025, 0.975)).astype(float).tolist()
        transfer_confirmation_passed = bool(
            horizon["final"].get("transferConfirmationPassed", float(gain.mean()) >= -0.001)
        )
        search_production_eligible = bool(
            horizon["final"].get(
                "productionEligible",
                horizon["final"].get("converged", False) and transfer_confirmation_passed,
            )
        )
        # A correction strictly enlarges its incumbent basis.  Under the
        # quality-first/smallest-equivalent rule it is promoted only when the
        # untouched day-block confidence interval is wholly positive; an
        # indistinguishable correction loses to the smaller incumbent.
        robust_promotion_passed = bool(
            search_production_eligible and bootstrap_interval[0] > 0.0
        )
        if robust_promotion_passed:
            recommended_ids.update(horizon["final"]["selectedRawInputIds"])
        else:
            recommended_ids.update(horizon["final"]["requiredIncumbentInputs"])
        results.append({
            "horizon": horizon["horizon"],
            "rows": int(rows.size),
            "days": int(unique_days.size),
            "modelBits": float(model_bits.mean()),
            "incumbentBits": float(incumbent_bits.mean()),
            "gainBits": float(gain.mean()),
            "positiveDays": int(sum(row["gainBits"] > 0 for row in daily)),
            "bootstrap95GainBits": bootstrap_interval,
            "probabilityGainPositive": float(np.mean(bootstrap > 0)),
            "transferConfirmationPassed": transfer_confirmation_passed,
            "searchProductionEligible": search_production_eligible,
            "robustPromotionPassed": robust_promotion_passed,
            "recommendedRawInputIds": (
                horizon["final"]["selectedRawInputIds"]
                if robust_promotion_passed
                else horizon["final"]["requiredIncumbentInputs"]
            ),
            "daily": daily,
        })
    artifact = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceSearch": str(search_path.relative_to(ROOT)).replace("\\", "/"),
        "sourceModel": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "method": "Untouched transfer gains by UTC day; percentile bootstrap resamples whole days",
        "bootstrapSamples": args.bootstrap_samples,
        "results": results,
        "robustUnion": {
            "rawInputCount": len(recommended_ids),
            "rawInputIds": sorted(recommended_ids),
            "promotionRule": (
                "optimizer converged, transfer did not regress by more than 0.001 bits, "
                "and the 95% whole-day bootstrap interval for gain is wholly positive; "
                "otherwise retain the smaller incumbent"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}", flush=True)
    print(json.dumps([{key: row[key] for key in (
        "horizon", "gainBits", "positiveDays", "days", "bootstrap95GainBits",
        "probabilityGainPositive", "transferConfirmationPassed", "robustPromotionPassed",
    )} for row in results], indent=2), flush=True)


if __name__ == "__main__":
    main()
