from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from global_feature_registry import ROOT


DEFAULT_REGISTRY = ROOT / "data/benchmarks/global-feature-registry.json"
DEFAULT_WORKING_SET = ROOT / "data/runtime-cache/global-btc-expanded-working-set-v4/manifest.json"
DEFAULT_SEARCH = ROOT / "data/benchmarks/global-btc-v4-production-final-search.json"
DEFAULT_ROBUSTNESS = ROOT / "data/benchmarks/global-btc-v4-production-final-transfer-robustness.json"
DEFAULT_KKT = ROOT / "data/benchmarks/global-btc-v4-production-final-kkt-merged.json"
DEFAULT_INCUMBENT_SEARCH = ROOT / "data/benchmarks/global-btc-v3-operational-search.json"
DEFAULT_INCUMBENT_ROBUSTNESS = ROOT / "data/benchmarks/global-btc-v3-operational-transfer-robustness.json"
DEFAULT_OUTPUT = ROOT / "docs/experiments/global-btc-feature-basis-search-2026-08-21.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the audited global BTC feature-basis report.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--working-set", type=Path, default=DEFAULT_WORKING_SET)
    parser.add_argument("--search", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--robustness", type=Path, default=DEFAULT_ROBUSTNESS)
    parser.add_argument("--kkt", type=Path, default=DEFAULT_KKT)
    parser.add_argument("--incumbent-search", type=Path, default=DEFAULT_INCUMBENT_SEARCH)
    parser.add_argument("--incumbent-robustness", type=Path, default=DEFAULT_INCUMBENT_ROBUSTNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def template_id(coordinate_id: str) -> str:
    parts = coordinate_id.split("/", 4)
    if len(parts) != 5:
        raise ValueError(f"Malformed coordinate id: {coordinate_id}")
    return "/".join((parts[0], parts[2], parts[3], parts[4]))


def rel(path: Path) -> str:
    try:
        value = path.relative_to(ROOT)
    except ValueError:
        value = path
    return str(value).replace("\\", "/")


def selected_path_row(horizon: dict[str, Any]) -> dict[str, Any]:
    selected = float(horizon["selectedLambdaFraction"])
    return next(
        row for row in horizon["pathSummary"]
        if abs(float(row["lambdaFraction"]) - selected) <= 1e-12
    )


def main() -> None:
    args = parse_args()
    registry_path = resolved(args.registry)
    working_path = resolved(args.working_set)
    search_path = resolved(args.search)
    robustness_path = resolved(args.robustness)
    kkt_path = resolved(args.kkt)
    output_path = resolved(args.output)

    registry = load(registry_path)
    working = load(working_path)
    search = load(search_path)
    robustness = load(robustness_path)
    incumbent_search = load(resolved(args.incumbent_search))
    incumbent_robustness = load(resolved(args.incumbent_robustness))
    present_horizons = {str(row["horizon"]) for row in search["horizons"]}
    search = copy.deepcopy(search)
    search["horizons"] += [
        copy.deepcopy(row)
        for row in incumbent_search["horizons"]
        if str(row["horizon"]) not in present_horizons
    ]
    present_robustness = {str(row["horizon"]) for row in robustness["results"]}
    robustness = copy.deepcopy(robustness)
    robustness["results"] += [
        copy.deepcopy(row)
        for row in incumbent_robustness["results"]
        if str(row["horizon"]) not in present_robustness
    ]
    robust_union = sorted({
        str(feature_id)
        for row in robustness["results"]
        for feature_id in row["recommendedRawInputIds"]
    })
    robustness["robustUnion"] = {
        **robustness.get("robustUnion", {}),
        "rawInputCount": len(robust_union),
        "rawInputIds": robust_union,
    }
    kkt = load(kkt_path)
    templates = {row["canonicalId"]: row for row in registry["templates"]}
    policies = {row["id"]: row for row in registry["sourcePolicies"]}
    robustness_by_horizon = {row["horizon"]: row for row in robustness["results"]}
    empirical_by_id: dict[str, float] = {}
    for horizon in search["horizons"]:
        for row in horizon["final"]["support"]:
            empirical_by_id[row["id"]] = float(row.get("empiricalAvailability", 0.0))

    summary = registry["summary"]
    robust_ids = list(robustness["robustUnion"]["rawInputIds"])
    equivalence_rule = str(search.get("selection", {}).get("equivalenceRule", "fixed"))
    if equivalence_rule == "paired-one-se":
        equivalence_text = (
            "Among incumbent-safe paths, statistical equivalence uses the paired "
            "one-standard-error rule on chronological-fold gains; availability, acquisition "
            "cost, and support size then choose the operationally preferable path."
        )
    else:
        equivalence_text = (
            "Among paths within 0.001 bits of the best on every fold, availability, acquisition "
            "cost, and support size choose the operationally preferable path."
        )
    promoted_horizons = {
        str(row["horizon"])
        for row in robustness["results"]
        if bool(row.get("robustPromotionPassed"))
    }
    kkt_by_horizon = {str(row["horizon"]): row for row in kkt["results"]}
    certified = bool(kkt.get("complete")) and all(
        bool(kkt_by_horizon.get(name, {}).get("certifiedAtTolerance"))
        for name in promoted_horizons
    )
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Global BTC predictive feature-basis search",
        "",
        f"Generated `{now}` from the canonical registry, chronological search, transfer robustness audit, and full streamed KKT scan.",
        "",
        "## Decision",
        "",
        f"The robust production union contains **{len(robust_ids):,} raw inputs**. "
        + ("Every promoted correction is KKT-certified over the complete eligible registry; rejected horizons keep their incumbent inputs." if certified else "At least one promoted correction still has an omitted-coordinate KKT violation, so this artifact is not yet a final production certificate."),
        "",
        "A correction is retained only when its optimizer converged, it did not regress by more than 0.001 bits on transfer, and the 95% whole-day bootstrap interval for its transfer gain is wholly positive. If quality is statistically indistinguishable, the smaller incumbent wins.",
        "",
        "The final correction model serializes only the promoted 1s and 1m heads. The 15m and 1h rows below are carried from the prior four-horizon audit because their broad corrections failed transfer; their smaller incumbent inputs remain the production contract.",
        "",
        equivalence_text,
        "",
        "## Exact candidate registry",
        "",
        "| Quantity | Count |",
        "|---|---:|",
        f"| Assets in union | {registry['universe']['count']:,} |",
        f"| Raw ledger coordinates | {summary['rawCoordinates']:,} |",
        f"| Duplicate aliases/overlaps | {summary['duplicateLedgerCoordinates']:,} |",
        f"| Canonical unique coordinates | **{summary['uniqueCoordinates']:,}** |",
        f"| Canonical templates | {summary['canonicalTemplates']:,} |",
        "",
        "| Inventory | Raw coordinates |",
        "|---|---:|",
    ]
    for row in summary["inventories"]:
        lines.append(f"| {md(row['id'])} | {int(row['rawCoordinates']):,} |")

    lines += [
        "",
        "The 3,471 dense minute variants already include the declared lag grid. Templates expand only over assets with source coverage; the registry is therefore not `261 × templates`.",
        "",
        "## Search and validation design",
        "",
        f"Model class: {md(search['modelClass'])}.",
        "",
        f"Candidate working set: **{int(search['candidateSet']['total']):,}** active-set coordinates, constructed from full-registry gradient screens while always retaining the incumbent inputs. The full registry is streamed for KKT checks rather than materialized as a dense 42,901 × 990,084 matrix.",
        "",
        f"Selection protocol: {md(search['selection']['protocol'])}. The seven-day transfer segment is confirmation-only and never selects a penalty or support.",
        "",
        "| Horizon | λ / λmax | Mean fold bits | Worst fold bits | Fold support union | Final correction groups | Final stationarity | Converged |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for horizon in search["horizons"]:
        path = selected_path_row(horizon)
        final = horizon["final"]
        fold_support_union = path.get("foldSupportUnionSize")
        if fold_support_union is None:
            fold_support_union = round(float(path.get("meanSupportSize", 0.0)))
        correction_support = final.get("correctionSupportSize", len(final.get("support", [])))
        lines.append(
            f"| {horizon['horizon']} | {number(horizon['selectedLambdaFraction'], 4)} | "
            f"{number(path['meanBits'])} | {number(path['worstFoldBits'])} | "
            f"{int(fold_support_union)} | {int(correction_support)} | "
            f"{number(final.get('stationarityMaximum'), 8)} | "
            f"{'yes' if final['converged'] else 'no'} |"
        )

    lines += [
        "",
        "### Predictive-quality versus operational tie-break",
        "",
        "The best-mean path is shown before the predeclared one-standard-error equivalence rule. Within that equivalence set, selection uses bottleneck availability, maximum acquisition cost, and support size in that order; mean availability is secondary because adding an always-present input cannot improve joint model availability.",
        "",
        "| Horizon | Best-mean λ | Best mean bits | Operational λ | Operational mean bits | Mean-bit difference | Operational fold-union size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in search["horizons"]:
        pool = [
            row for row in horizon["pathSummary"]
            if row.get("allConverged") and row.get("incumbentSafe")
        ]
        best = max(pool, key=lambda row: float(row["meanBits"]))
        selected = selected_path_row(horizon)
        lines.append(
            f"| {horizon['horizon']} | {number(best['lambdaFraction'], 4)} | "
            f"{number(best['meanBits'])} | {number(selected['lambdaFraction'], 4)} | "
            f"{number(selected['meanBits'])} | "
            f"{number(float(selected['meanBits']) - float(best['meanBits']))} | "
            f"{int(selected.get('foldSupportUnionSize', 0))} |"
        )

    lines += [
        "",
        "## Transfer robustness",
        "",
        "| Horizon | Gain over incumbent, bits | Positive days | Day-bootstrap 95% interval | P(gain > 0) | Promote correction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in robustness["results"]:
        interval = row["bootstrap95GainBits"]
        lines.append(
            f"| {row['horizon']} | {number(row['gainBits'])} | {int(row['positiveDays'])}/{int(row['days'])} | "
            f"[{number(interval[0])}, {number(interval[1])}] | {number(row['probabilityGainPositive'], 4)} | "
            f"{'yes' if row['robustPromotionPassed'] else 'no; keep incumbent'} |"
        )

    lines += [
        "",
        "## Production input contract by prediction horizon",
        "",
        "Every row is a completed, causally available value at the prediction origin. Missingness/age channels named explicitly in an ID are inputs, not imputed future observations.",
    ]
    for horizon in search["horizons"]:
        name = horizon["horizon"]
        robust = robustness_by_horizon[name]
        ids = sorted(robust["recommendedRawInputIds"])
        lines += [
            "",
            f"### {name}: {len(ids):,} inputs",
            "",
            "| Coordinate | Family | Construction | Lookback | Delay | Source policy | Declared availability | Empirical availability | Acquisition cost |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ]
        for feature_id in ids:
            template = templates[template_id(feature_id)]
            policy = policies[template["sourcePolicy"]]
            lines.append(
                f"| `{md(feature_id)}` | {md(template['family'])} | {md(template.get('construction') or '—')} | "
                f"{md(template.get('lookback') or '—')} | {md(template.get('delay') or '—')} | "
                f"{md(template['sourcePolicy'])} | {number(policy['availability_score'], 3)} | "
                f"{number(empirical_by_id.get(feature_id), 3)} | {int(policy['acquisition_cost'])} |"
            )

    lines += [
        "",
        "## Full-registry KKT audit",
        "",
        f"Providers scanned: `{md(', '.join(kkt['providers']))}`. Unique emitted coordinates: **{int(kkt['uniqueCoordinatesEmitted']):,}**. Complete stream: **{'yes' if kkt.get('complete') else 'no'}**.",
        "",
        "| Horizon | Production correction | Regularization | Scanned | Maximum omitted-group violation | Maximum active-stationarity residual | Certified |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in kkt["results"]:
        lines.append(
            f"| {row['horizon']} | {'required' if row['horizon'] in promoted_horizons else 'rejected; incumbent used'} | "
            f"{number(row['regularization'], 8)} | {int(row['coordinatesScanned']):,} | "
            f"{number(row['maximumViolation'], 8)} | {number(row.get('activeStationarityMaximum'), 8)} | "
            f"{'yes' if row['certifiedAtTolerance'] else 'no'} |"
        )

    lines += [
        "",
        "## Meaning of the global certificate",
        "",
        "The proof boundary is deliberately narrow: fixed causal quantile partitions, the incumbent smoothed joint-state conditional distribution, and additive four-bin categorical corrections with a group-lasso penalty. For a selected penalty, a complete nonpositive KKT scan proves the global convex optimum in that model class across the eligible canonical registry. It does not prove an optimum over arbitrary neural interactions, alternative discretizations, revised/non-point-in-time data, or the short live-only inventories.",
        "",
        "Feature discovery used the first 23 days and chronological folds within it, so it is not fully nested feature-selection cross-validation. The final seven days remained untouched through the first confirmation. Later KKT expansions and the operational tie-break used only training residuals and stored fold paths, but this report reuses the already revealed transfer segment; its whole-day resampling is robustness evidence, not a new pristine holdout. A later calendar block is still required before deployment promotion.",
        "",
        "## Reproducibility",
        "",
        f"- Registry: `{rel(registry_path)}`",
        f"- Expanded working set: `{rel(working_path)}`",
        f"- Search: `{rel(search_path)}`",
        f"- Transfer audit: `{rel(robustness_path)}`",
        f"- KKT audit: `{rel(kkt_path)}`",
        f"- Rejected-horizon incumbent search: `{rel(resolved(args.incumbent_search))}`",
        f"- Rejected-horizon robustness: `{rel(resolved(args.incumbent_robustness))}`",
        "",
        "```powershell",
        "npm run analysis:feature-registry",
        "npm run analysis:global-basis-search:per-horizon -- --working-set data/runtime-cache/global-btc-expanded-working-set-v4 --all-coordinates --horizons 1s,1m --equivalence-rule paired-one-se",
        "npm run analysis:global-basis-search:refine-support -- --search data/benchmarks/global-btc-v4-production-search.json --model data/benchmarks/global-btc-v4-production-model.npz --working-set data/runtime-cache/global-btc-expanded-working-set-v4 --output-search data/benchmarks/global-btc-v4-production-final-search.json --output-model data/benchmarks/global-btc-v4-production-final-model.npz --horizon 1s --device cpu",
        "npm run analysis:global-basis-search:transfer-robustness -- --search data/benchmarks/global-btc-v4-production-final-search.json --model data/benchmarks/global-btc-v4-production-final-model.npz --output data/benchmarks/global-btc-v4-production-final-transfer-robustness.json",
        "# Run the three disjoint KKT provider partitions with all training rows, then:",
        "npm run analysis:global-basis-search:kkt-merge -- --search data/benchmarks/global-btc-v4-production-final-search.json --model data/benchmarks/global-btc-v4-production-final-model.npz --parts data/benchmarks/global-btc-v4-production-final-kkt-core.json data/benchmarks/global-btc-v4-production-final-kkt-spectral1m.json data/benchmarks/global-btc-v4-production-final-kkt-external-second.json --output data/benchmarks/global-btc-v4-production-final-kkt-merged.json --horizons 1s,1m",
        "npm run analysis:global-basis-search:report",
        "```",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {rel(output_path)}", flush=True)


if __name__ == "__main__":
    main()
