from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/runtime-cache/live-component-feature-bases"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/live-component-feature-bases.json"
DEFAULT_REPORT = ROOT / "docs/experiments/live-component-feature-bases-2026-08-19.md"
ALPHA = 0.5
FEATURE_BINS = 3
BASELINE_BINS = 3
FINALISTS = 12
MAX_SUBSET_SIZE = 3
PARSIMONY_BITS = 0.001
MIN_PRIMARY = 32
MIN_TRANSFER = 32
MIN_EFFECTIVE_TRAIN_PER_STATE = 4
MAGNITUDE_QUANTILES = np.asarray([0.25, 0.5, 0.75, 0.9], dtype=np.float64)


@dataclass(frozen=True)
class Component:
    id: str
    family: str
    label: str
    condition_label: str
    classes: int
    condition: Callable[[np.ndarray, np.ndarray], np.ndarray]
    target: Callable[[np.ndarray, np.ndarray], np.ndarray]
    threshold_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def components() -> list[Component]:
    result = [
        Component(
            "inactive", "activity", "P(inactive)", "all fixed-horizon returns", 2,
            lambda values, _thresholds: np.ones(values.size, dtype=bool),
            lambda values, _thresholds: (values == 0).astype(np.int16),
        ),
        Component(
            "sign_given_active", "direction", "P(positive | active)", "R != 0", 2,
            lambda values, _thresholds: values != 0,
            lambda values, _thresholds: (values > 0).astype(np.int16),
        ),
        Component(
            "joint_zero_sign_magnitude", "joint", "P(zero/sign/magnitude quartile)",
            "all fixed-horizon returns", 9,
            lambda values, _thresholds: np.ones(values.size, dtype=bool),
            joint_target,
        ),
    ]
    for index, quantile in enumerate(MAGNITUDE_QUANTILES):
        suffix = f"q{round(100 * quantile)}"
        result.append(Component(
            f"large_{suffix}_given_active", "magnitude",
            f"P(|R| >= Q{round(100 * quantile)} | active)", "R != 0", 2,
            lambda values, _thresholds: values != 0,
            lambda values, thresholds, i=index: (np.abs(values) >= thresholds[i]).astype(np.int16),
            index,
        ))
    for index in (1, 2, 3):
        quantile = MAGNITUDE_QUANTILES[index]
        result.append(Component(
            f"sign_given_large_q{round(100 * quantile)}", "direction-large",
            f"P(positive | active, |R| >= Q{round(100 * quantile)})",
            f"R != 0 and |R| >= Q{round(100 * quantile)}", 2,
            lambda values, thresholds, i=index: (values != 0) & (np.abs(values) >= thresholds[i]),
            lambda values, _thresholds: (values > 0).astype(np.int16),
            index,
        ))
    for index in (0, 1, 2):
        quantile = MAGNITUDE_QUANTILES[index]
        result.append(Component(
            f"sign_given_small_q{round(100 * quantile)}", "direction-small",
            f"P(positive | active, |R| < Q{round(100 * quantile)})",
            f"R != 0 and |R| < Q{round(100 * quantile)}", 2,
            lambda values, thresholds, i=index: (values != 0) & (np.abs(values) < thresholds[i]),
            lambda values, _thresholds: (values > 0).astype(np.int16),
            index,
        ))
    for sign, name in ((-1, "negative"), (1, "positive")):
        for index in (1, 2, 3):
            quantile = MAGNITUDE_QUANTILES[index]
            result.append(Component(
                f"large_q{round(100 * quantile)}_given_{name}", "magnitude-given-sign",
                f"P(|R| >= Q{round(100 * quantile)} | {name})", f"R is {name}", 2,
                lambda values, _thresholds, s=sign: values * s > 0,
                lambda values, thresholds, i=index: (np.abs(values) >= thresholds[i]).astype(np.int16),
                index,
            ))
    return result


def joint_target(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    output = np.zeros(values.size, dtype=np.int16)
    active = values != 0
    magnitude = np.searchsorted(thresholds[:3], np.abs(values[active]), side="left")
    output[active] = 1 + magnitude + 4 * (values[active] > 0)
    return output


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.render_only:
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(artifact), encoding="utf-8")
        print(f"Wrote {display_path(args.report)}")
        return
    rows = int(manifest["rows"])
    feature_count = int(manifest["featureCount"])
    horizon_count = int(manifest["targetCount"])
    features = np.memmap(
        args.input_dir / manifest["files"]["features"], dtype="<f4", mode="r",
        shape=(rows, feature_count),
    )
    targets = np.memmap(
        args.input_dir / manifest["files"]["targets"], dtype="<f4", mode="r",
        shape=(rows, horizon_count),
    )
    previous = np.memmap(
        args.input_dir / manifest["files"]["previous"], dtype="<f4", mode="r",
        shape=(rows, horizon_count),
    )
    volatility = np.memmap(
        args.input_dir / manifest["files"]["volatility"], dtype="<f4", mode="r",
        shape=(rows, horizon_count),
    )
    times = np.memmap(
        args.input_dir / manifest["files"]["times"], dtype="<f8", mode="r", shape=(rows,),
    )
    splits = np.memmap(
        args.input_dir / manifest["files"]["splits"], dtype="u1", mode="r", shape=(rows,),
    )
    results: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(manifest["horizonsSeconds"]):
        raw_target = np.asarray(targets[:, horizon_index], dtype=np.float64)
        train = np.asarray(splits == 0)
        active_train = np.abs(raw_target[train & (raw_target != 0)])
        thresholds = np.quantile(active_train, MAGNITUDE_QUANTILES)
        for component in components():
            print(f"Searching {horizon}s {component.id}...", flush=True)
            result = analyze_component(
                manifest,
                np.asarray(features, dtype=np.float64),
                raw_target,
                np.asarray(previous[:, horizon_index], dtype=np.float64),
                np.asarray(volatility[:, horizon_index], dtype=np.float64),
                np.asarray(times, dtype=np.float64),
                np.asarray(splits, dtype=np.uint8),
                int(horizon),
                thresholds,
                component,
            )
            results.append(result)
    statuses = {status: sum(row["status"] == status for row in results) for status in (
        "confirmed-early", "primary-only", "no-primary-addition", "insufficient-evaluation",
    )}
    artifact = {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "objective": "A separately selected nonredundant live feature basis for every future return component",
        "metric": "incremental held-out log2 likelihood versus previous same-horizon return plus trailing absolute-return state",
        "dataset": {
            "input": display_path(args.input_dir),
            "rows": rows,
            "features": feature_count,
            "coverage": manifest["coverage"],
            "split": manifest["split"],
        },
        "search": {
            "fixedBaseline": manifest["fixedBaseline"],
            "featureBins": FEATURE_BINS,
            "finalists": FINALISTS,
            "maximumSubsetSize": MAX_SUBSET_SIZE,
            "finalistRule": "best primary-period marginal member of every source family, then fill by primary marginal bits",
            "exhaustiveUniverse": f"every subset of up to {MAX_SUBSET_SIZE} inputs inside each target's {FINALISTS}-input finalist set",
            "selection": "smallest primary-stable subset within 0.001 bits/eligible target of the primary optimum",
            "sparsityConstraint": f"candidate model states may not exceed non-overlapping eligible training outcomes divided by {MIN_EFFECTIVE_TRAIN_PER_STATE}",
            "confirmation": "the final 20% transfer interval is untouched until after basis selection; confirmed-early requires positive total transfer bits and both transfer half-blocks positive",
            "warning": "Finite-universe histogram optimum only. It is not a mathematical optimum over arbitrary transforms, and one live market regime is not production promotion evidence.",
        },
        "statusCounts": statuses,
        "features": manifest["features"],
        "targets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {display_path(args.output)}")
    print(f"Wrote {display_path(args.report)}")


def analyze_component(
    manifest: dict[str, Any],
    raw_features: np.ndarray,
    raw_target: np.ndarray,
    previous: np.ndarray,
    volatility: np.ndarray,
    times: np.ndarray,
    splits: np.ndarray,
    horizon: int,
    thresholds: np.ndarray,
    component: Component,
) -> dict[str, Any]:
    condition = component.condition(raw_target, thresholds)
    labels = component.target(raw_target, thresholds)
    train = condition & (splits == 0)
    training_indices = nonoverlapping_indices(times, train, horizon)
    primary_indices = nonoverlapping_indices(times, condition & (splits == 1), horizon)
    transfer_indices = nonoverlapping_indices(times, condition & (splits == 2), horizon)
    target_scale_to_bps = float(manifest.get("targetScaleToBps", 10_000))
    threshold_bps = None if component.threshold_index is None else float(
        thresholds[component.threshold_index] * target_scale_to_bps
    )
    class_counts = np.bincount(labels[train], minlength=component.classes)
    base = {
        "horizonSeconds": horizon,
        "componentId": component.id,
        "componentFamily": component.family,
        "componentLabel": component.label,
        "condition": component.condition_label,
        "thresholdQuantile": None if component.threshold_index is None else float(
            MAGNITUDE_QUANTILES[component.threshold_index]
        ),
        "thresholdBps": threshold_bps,
        "trainingEligible": int(train.sum()),
        "trainingEffective": int(training_indices.size),
        "primaryEffective": int(primary_indices.size),
        "transferEffective": int(transfer_indices.size),
        "trainingClassCounts": class_counts.tolist(),
    }
    if (
        train.sum() < 128
        or primary_indices.size < MIN_PRIMARY
        or transfer_indices.size < MIN_TRANSFER
        or np.count_nonzero(class_counts) < 2
    ):
        return {
            **base,
            "status": "insufficient-evaluation",
            "finalists": [],
            "testedSubsets": 0,
            "selectedBasis": empty_basis(),
            "marginalRanking": [],
        }
    baseline_columns = np.column_stack((previous, volatility))
    baseline_quantized, _, baseline_arities = quantize_columns(baseline_columns, train, BASELINE_BINS)
    baseline_state, baseline_states = encode_columns(
        baseline_quantized, tuple(range(baseline_quantized.shape[1])), baseline_arities
    )
    quantized, edges, arities = quantize_columns(raw_features, train, FEATURE_BINS)
    primary_blocks = split_indices(primary_indices)
    transfer_blocks = split_indices(transfer_indices)
    baseline_model = fit_model(baseline_state[train], labels[train], baseline_states, component.classes)
    marginal: list[dict[str, Any]] = []
    for index, definition in enumerate(manifest["features"]):
        score = score_subset(
            (index,), quantized, labels, train, primary_indices, transfer_indices,
            primary_blocks, transfer_blocks, arities, component.classes,
            baseline_state, baseline_states, baseline_model,
        )
        marginal.append({"featureIndex": index, **definition, **score})
    marginal.sort(key=lambda row: row["primaryBits"], reverse=True)
    finalists = select_finalists(marginal, FINALISTS)
    finalist_indices = tuple(int(row["featureIndex"]) for row in finalists)
    subset_rows: list[dict[str, Any]] = []
    maximum_admissible_states = max(9, training_indices.size // MIN_EFFECTIVE_TRAIN_PER_STATE)
    for size in range(1, min(MAX_SUBSET_SIZE, len(finalist_indices)) + 1):
        for subset in itertools.combinations(finalist_indices, size):
            subset_states = baseline_states * math.prod(arities[index] for index in subset)
            if subset_states > maximum_admissible_states:
                continue
            score = score_subset(
                subset, quantized, labels, train, primary_indices, transfer_indices,
                primary_blocks, transfer_blocks, arities, component.classes,
                baseline_state, baseline_states, baseline_model,
            )
            subset_rows.append({"indices": list(subset), **score})
    primary_stable = [row for row in subset_rows if all(value > 0 for value in row["primaryBlockBits"])]
    if not primary_stable:
        return {
            **base,
            "status": "no-primary-addition",
            "targetEdgesBps": (thresholds * target_scale_to_bps).tolist(),
            "featureEdges": [edge.tolist() for edge in edges],
            "finalists": [summarize_marginal(row) for row in finalists],
            "testedSubsets": len(subset_rows),
            "maximumAdmissibleStates": maximum_admissible_states,
            "selectedBasis": empty_basis(),
            "marginalRanking": [summarize_marginal(row) for row in marginal],
        }
    optimum = max(primary_stable, key=lambda row: row["primaryBits"])
    near = [row for row in primary_stable if row["primaryBits"] >= optimum["primaryBits"] - PARSIMONY_BITS]
    selected = min(near, key=lambda row: availability_tie_key(row, manifest["features"]))
    status = "confirmed-early" if (
        selected["transferBits"] > 0 and all(value > 0 for value in selected["transferBlockBits"])
    ) else "primary-only"
    contributions = []
    for feature_index in selected["indices"]:
        reduced = tuple(index for index in selected["indices"] if index != feature_index)
        reduced_score = score_subset(
            reduced, quantized, labels, train, primary_indices, transfer_indices,
            primary_blocks, transfer_blocks, arities, component.classes,
            baseline_state, baseline_states, baseline_model,
        ) if reduced else zero_score()
        definition = manifest["features"][feature_index]
        contributions.append({
            **definition,
            "primaryLeaveOneOutBits": selected["primaryBits"] - reduced_score["primaryBits"],
            "transferLeaveOneOutBits": selected["transferBits"] - reduced_score["transferBits"],
        })
    return {
        **base,
        "status": status,
        "targetEdgesBps": (thresholds * target_scale_to_bps).tolist(),
        "featureEdges": [edge.tolist() for edge in edges],
        "finalists": [summarize_marginal(row) for row in finalists],
        "testedSubsets": len(subset_rows),
        "maximumAdmissibleStates": maximum_admissible_states,
        "primaryOptimum": describe_subset(optimum, manifest["features"]),
        "selectedBasis": {
            **describe_subset(selected, manifest["features"]),
            "availabilityTieBreak": describe_availability(selected, manifest["features"]),
            "conditionalContributions": contributions,
        },
        "marginalRanking": [summarize_marginal(row) for row in marginal],
    }


def describe_availability(row: dict[str, Any], definitions: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(definitions[index].get("availabilityScore", 0.5)) for index in row["indices"]]
    return {
        "minimumScore": min(scores) if scores else 1.0,
        "meanScore": float(np.mean(scores)) if scores else 1.0,
        "rule": "within 0.001 primary bits, maximize minimum then mean availability; then minimize size",
    }


def availability_tie_key(row: dict[str, Any], definitions: list[dict[str, Any]]) -> tuple[float, float, int, float]:
    availability = describe_availability(row, definitions)
    return (
        -float(availability["minimumScore"]),
        -float(availability["meanScore"]),
        len(row["indices"]),
        -float(row["primaryBits"]),
    )


def quantize_columns(
    values: np.ndarray, train: np.ndarray, bins: int
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    output = np.empty(values.shape, dtype=np.uint8)
    all_edges: list[np.ndarray] = []
    arities: list[int] = []
    for column in range(values.shape[1]):
        train_values = values[train, column]
        finite_train = train_values[np.isfinite(train_values)]
        if finite_train.size == 0:
            edges = np.asarray([], dtype=np.float64)
        else:
            edges = np.unique(np.quantile(finite_train, np.arange(1, bins) / bins))
        finite = np.isfinite(values[:, column])
        output[finite, column] = np.searchsorted(edges, values[finite, column], side="right")
        has_missing = bool(np.any(~finite))
        if has_missing:
            output[~finite, column] = len(edges) + 1
        all_edges.append(edges)
        arities.append(len(edges) + 1 + int(has_missing))
    return output, all_edges, arities


def nonoverlapping_indices(times: np.ndarray, mask: np.ndarray, horizon: int) -> np.ndarray:
    keep: list[int] = []
    last = -math.inf
    for index in np.flatnonzero(mask):
        current = float(times[index])
        if current >= last + horizon:
            keep.append(int(index))
            last = current
    return np.asarray(keep, dtype=np.int64)


def split_indices(indices: np.ndarray) -> list[np.ndarray]:
    middle = indices.size // 2
    return [indices[:middle], indices[middle:]]


def encode_columns(
    quantized: np.ndarray, subset: tuple[int, ...], arities: list[int]
) -> tuple[np.ndarray, int]:
    state = np.zeros(quantized.shape[0], dtype=np.int64)
    states = 1
    for index in subset:
        state = state * arities[index] + quantized[:, index]
        states *= arities[index]
    return state, states


def fit_model(states: np.ndarray, target: np.ndarray, state_count: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    joint = np.bincount(
        states * classes + target, minlength=state_count * classes
    ).reshape(state_count, classes).astype(np.float64)
    return joint, joint.sum(axis=1)


def score_subset(
    subset: tuple[int, ...],
    quantized: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    primary: np.ndarray,
    transfer: np.ndarray,
    primary_blocks: list[np.ndarray],
    transfer_blocks: list[np.ndarray],
    arities: list[int],
    classes: int,
    baseline_state: np.ndarray,
    baseline_states: int,
    baseline_model: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    feature_state, feature_states = encode_columns(quantized, subset, arities)
    state = baseline_state * feature_states + feature_state
    joint, totals = fit_model(state[train], target[train], baseline_states * feature_states, classes)
    baseline_joint, baseline_totals = baseline_model

    def ratios(indices: np.ndarray) -> np.ndarray:
        y = target[indices]
        probability = (joint[state[indices], y] + ALPHA) / (totals[state[indices]] + ALPHA * classes)
        base_probability = (
            baseline_joint[baseline_state[indices], y] + ALPHA
        ) / (baseline_totals[baseline_state[indices]] + ALPHA * classes)
        return np.log2(probability / base_probability)

    primary_ratios = ratios(primary)
    transfer_ratios = ratios(transfer)
    return {
        "primaryBits": float(np.mean(primary_ratios)),
        "transferBits": float(np.mean(transfer_ratios)),
        "primaryBlockBits": [float(np.mean(ratios(block))) for block in primary_blocks],
        "transferBlockBits": [float(np.mean(ratios(block))) for block in transfer_blocks],
        "states": int(baseline_states * feature_states),
    }


def select_finalists(marginal: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for family in dict.fromkeys(row["family"] for row in marginal):
        row = next(candidate for candidate in marginal if candidate["family"] == family)
        selected.append(row)
        seen.add(int(row["featureIndex"]))
    for row in marginal:
        index = int(row["featureIndex"])
        if index not in seen:
            selected.append(row)
            seen.add(index)
        if len(selected) >= count:
            break
    return selected[:count]


def summarize_marginal(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key] for key in (
            "featureIndex", "id", "family", "source", "construction", "lookback",
            "primaryBits", "transferBits", "primaryBlockBits", "transferBlockBits",
        )
    }


def describe_subset(row: dict[str, Any], definitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": [definitions[index]["id"] for index in row["indices"]],
        "size": len(row["indices"]),
        "primaryBits": row["primaryBits"],
        "transferBits": row["transferBits"],
        "primaryBlockBits": row["primaryBlockBits"],
        "transferBlockBits": row["transferBlockBits"],
        "states": row["states"],
    }


def zero_score() -> dict[str, Any]:
    return {
        "primaryBits": 0.0,
        "transferBits": 0.0,
        "primaryBlockBits": [0.0, 0.0],
        "transferBlockBits": [0.0, 0.0],
        "states": 9,
    }


def empty_basis() -> dict[str, Any]:
    return {
        "features": [], "size": 0, "primaryBits": 0.0, "transferBits": 0.0,
        "primaryBlockBits": [0.0, 0.0], "transferBlockBits": [0.0, 0.0],
        "states": 9, "conditionalContributions": [],
    }


def render_report(artifact: dict[str, Any]) -> str:
    targets = artifact["targets"]
    counts = artifact["statusCounts"]
    confirmed_frequency: dict[str, int] = {}
    for row in targets:
        if row["status"] != "confirmed-early":
            continue
        for feature in row["selectedBasis"]["features"]:
            confirmed_frequency[feature] = confirmed_frequency.get(feature, 0) + 1
    recurring = sorted(confirmed_frequency.items(), key=lambda item: (-item[1], item[0]))[:6]
    lines = [
        "# Live component-specific feature bases",
        "",
        f"Generated {artifact['generatedAt']}. This is the separate **fast-live external overlay** contract for each future return component, using {artifact['dataset']['rows']:,} common-coverage seconds and {artifact['dataset']['features']} clean fast-live candidate coordinates. It is not the total project feature inventory.",
        "",
        "## Outcome",
        "",
        f"Of {len(targets)} horizon/component heads, {counts['confirmed-early']} retain a positive basis in both halves of the untouched transfer interval, {counts['primary-only']} are selected on the primary interval but fail transfer confirmation, {counts['no-primary-addition']} select no live addition, and {counts['insufficient-evaluation']} lack enough non-overlapping eligible outcomes.",
        "",
        "The central result is that there is no single best external feature basis. Activity, direction, ordinary magnitude, and tail heads frequently select different coordinates. `confirmed-early` still means only that the result survived this approximately one-day common-coverage regime; it is not production promotion.",
        "",
        "The most recurrent coordinates across transfer-confirmed bases are " + ", ".join(
            f"`{name}` ({frequency} heads)" for name, frequency in recurring
        ) + ". Recurrence is useful for implementation sharing, but is not a substitute for each head's conditional contribution test.",
        "",
        "## Search contract",
        "",
        "- Fixed baseline for every head: previous completed return at the same horizon plus trailing absolute-return state, both training-quantized into tertiles.",
        "- Candidate universe: 40 clean cross-exchange book/depth coordinates, 12 Binance BTC liquidation coordinates, 12 Deribit perpetual-flow coordinates, and 12 Deribit option-flow coordinates. Historical contaminated Kraken coordinates are excluded. The separate 147-coordinate recent broad dataset, 309 macro transformations, long-history endogenous indicators, and slow live feeds are outside this common-coverage search.",
        "- Chronology: first 60% fits distributions, next 20% chooses finalists and subsets, final 20% confirms without influencing selection.",
        f"- Search: best family representatives plus the strongest marginal coordinates form {FINALISTS} finalists; every subset up to {MAX_SUBSET_SIZE} inputs is scored directly. The reported basis is the smallest primary-stable subset within {PARSIMONY_BITS:.3f} bits/eligible target of the primary optimum.",
        f"- Sparsity guard: a candidate's joint histogram may use at most one state per {MIN_EFFECTIVE_TRAIN_PER_STATE} non-overlapping eligible training outcomes. This automatically reduces basis size for slower or heavily conditioned heads.",
        "- Evaluation targets are made non-overlapping at their forecast horizon. Conditional-head bits are per eligible target and are not additive across differently conditioned populations.",
        "- `P(|R| >= x)` and `P(|R| < x)` are complements and have identical information at the same threshold, so only the `>=` form is searched.",
        "",
    ]
    for horizon in (1, 5, 15, 60):
        rows = [row for row in targets if row["horizonSeconds"] == horizon]
        lines.extend([
            f"## {horizon}s output contract",
            "",
            "Every row already includes the fixed two-coordinate baseline. `Primary-selected live overlay` contains the additional jointly selected coordinates; use it as a current candidate only when status is `confirmed-early`. A `primary-only` row keeps no confirmed live overlay yet.",
            "",
            "| prediction head | eligible condition | threshold | primary-selected live overlay | primary bits | transfer bits | P blocks | T blocks | status |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ])
        for row in rows:
            basis = row["selectedBasis"]
            threshold = "n/a" if row["thresholdBps"] is None else f"{row['thresholdBps']:.4f} bps"
            selected = ", ".join(f"`{name}`" for name in basis["features"]) or "none"
            lines.append(
                f"| {markdown_cell(row['componentLabel'])} | {markdown_cell(row['condition'])} | {threshold} | {selected} | "
                f"{basis['primaryBits']:.6f} | {basis['transferBits']:.6f} | "
                f"{format_blocks(basis['primaryBlockBits'])} | {format_blocks(basis['transferBlockBits'])} | {row['status']} |"
            )
        lines.append("")
    selected_ids = sorted({
        feature for row in targets if row["status"] == "confirmed-early"
        for feature in row["selectedBasis"]["features"]
    })
    definitions = {row["id"]: row for row in artifact["features"]}
    lines.extend([
        "## Confirmed-early coordinate dictionary",
        "",
        "These are the only live coordinates used by at least one transfer-confirmed component basis in this checkpoint.",
        "The status applies to the complete joint basis. An individual coordinate can still have a negative transfer leave-one-out contribution; those cases are exposed in the following section and should not be promoted independently.",
        "",
        "| coordinate | family | construction | lookback | confirmed output heads |",
        "|---|---|---|---|---|",
    ])
    for feature_id in selected_ids:
        definition = definitions[feature_id]
        heads = [
            f"{row['horizonSeconds']}s {row['componentId']}"
            for row in targets
            if row["status"] == "confirmed-early" and feature_id in row["selectedBasis"]["features"]
        ]
        lines.append(
            f"| `{feature_id}` | {definition['family']} | {definition['construction']} | "
            f"{definition['lookback']} | {', '.join(heads)} |"
        )
    if not selected_ids:
        lines.append("| none | n/a | No component basis survived transfer. | n/a | n/a |")
    lines.extend([
        "",
        "## Conditional contribution of selected coordinates",
        "",
        "Leave-one-out values measure what each coordinate contributes after the other selected coordinates. A negative transfer contribution means the primary-selected interaction did not reproduce cleanly.",
        "",
        "| output head | coordinate | primary leave-one-out bits | transfer leave-one-out bits |",
        "|---|---|---:|---:|",
    ])
    for row in targets:
        for contribution in row["selectedBasis"].get("conditionalContributions", []):
            lines.append(
                f"| {row['horizonSeconds']}s {row['componentId']} | `{contribution['id']}` | "
                f"{contribution['primaryLeaveOneOutBits']:.6f} | {contribution['transferLeaveOneOutBits']:.6f} |"
            )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Finalist and subset selection is exhaustive only inside the declared 12-coordinate finalist universe and maximum basis size three.",
        "- The final transfer block protects against direct selection leakage, but all blocks still belong to one short market regime and many target heads are tested.",
        "- The common-coverage requirement makes input comparisons fair, but uses less history than the earlier marginal screen.",
        "- At 1s, Q25/Q50/Q75 lie near one price tick and should be interpreted as a single micro-move regime. Q90 is the first clearly separated tail threshold.",
        "- This report selects only the live external overlay. The fixed endogenous baseline is not a claim that the complete long-history production core has been re-searched on this short window.",
        "",
        "Machine-readable results: `data/benchmarks/live-component-feature-bases.json`.",
        "",
    ])
    return "\n".join(lines)


def format_blocks(values: list[float]) -> str:
    return "/".join(f"{value:.4f}" for value in values)


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def display_path(file: Path) -> str:
    resolved = file.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
