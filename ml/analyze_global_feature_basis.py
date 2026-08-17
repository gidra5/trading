from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/runtime-cache/global-feature-basis"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/global-return-feature-basis.json"
DEFAULT_REPORT = ROOT / "docs/experiments/global-return-feature-basis-2026-08-17.md"
ALPHA = 0.5
FEATURE_BINS = 4
TARGET_ACTIVE_BINS = 16
FINALISTS = 10
MAX_SUBSET_SIZE = 6
PARISMONY_BITS = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizons", default="1s,1m,15m,1h")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.render_only:
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        artifact["datasetObjective"] = manifest.get("objective", "Joint chronological feature-subset search")
        artifact["resultFile"] = display_path(args.output)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        args.report.write_text(render_report(artifact), encoding="utf-8")
        print(f"Wrote {display_path(args.output)}")
        print(f"Wrote {display_path(args.report)}")
        return
    requested = set(args.horizons.split(","))
    results: list[dict[str, Any]] = []
    for dataset in manifest["datasets"]:
        features = np.memmap(
            args.input_dir / dataset["files"]["features"],
            dtype="<f4",
            mode="r",
            shape=(dataset["rows"], dataset["featureCount"]),
        )
        targets = np.memmap(
            args.input_dir / dataset["files"]["targets"],
            dtype="<f4",
            mode="r",
            shape=(dataset["rows"], dataset["targetCount"]),
        )
        splits = np.memmap(
            args.input_dir / dataset["files"]["splits"],
            dtype="u1",
            mode="r",
            shape=(dataset["rows"],),
        )
        times = np.memmap(
            args.input_dir / dataset["files"]["times"],
            dtype="<f8",
            mode="r",
            shape=(dataset["rows"],),
        )
        for target_index, target in enumerate(dataset["targets"]):
            if target["id"] not in requested:
                continue
            print(f"Searching {target['id']} across {dataset['featureCount']} candidates...", flush=True)
            selected_rows = nonoverlapping_rows(times, splits, target["minutes"] * 60_000)
            results.append(analyze_horizon(
                dataset,
                target,
                np.asarray(features[selected_rows], dtype=np.float64),
                np.asarray(targets[selected_rows, target_index], dtype=np.float64),
                np.asarray(splits[selected_rows], dtype=np.uint8),
                np.asarray(times[selected_rows], dtype=np.float64),
            ))
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        existing = {horizon["id"]: horizon for horizon in previous.get("horizons", [])}
        existing.update({horizon["id"]: horizon for horizon in results})
        order = {"1s": 0, "1m": 1, "15m": 2, "1h": 3}
        results = sorted(existing.values(), key=lambda horizon: order.get(horizon["id"], 99))
    artifact = {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "datasetObjective": manifest.get("objective", "Joint chronological feature-subset search"),
        "resultFile": display_path(args.output),
        "objective": "Exhaustive joint subset score against an unconditional return distribution, with no fixed feature baseline",
        "metric": "chronological held-out log2 likelihood gain in bits per target",
        "search": {
            "broadCandidateCount": "reported per horizon",
            "finalistRule": f"best member of every family, then fill to {FINALISTS} by primary held-out univariate bits",
            "exhaustiveUniverse": f"all subsets of the {FINALISTS} finalists up to {MAX_SUBSET_SIZE} coordinates",
            "selection": "maximum primary-period bits; the later transfer period is never used to choose finalists or subsets",
            "validationStableBasis": "highest-primary-bits subset among those positive in both chronological halves of the primary period",
            "transferConfirmation": "the selected basis is then reported on two untouched halves of the later transfer period",
            "parsimony": f"smallest subset within {PARISMONY_BITS} bits/target of the primary optimum",
            "warning": "This is globally optimal only inside the declared finalist universe and histogram model. Unrestricted mutual information is monotone and is maximized by all causal inputs.",
        },
        "split": manifest["split"],
        "horizons": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {display_path(args.output)}")
    print(f"Wrote {display_path(args.report)}")


def display_path(file: Path) -> str:
    resolved = file.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def nonoverlapping_rows(times: np.ndarray, splits: np.ndarray, horizon_ms: float) -> np.ndarray:
    keep: list[int] = []
    for split in (0, 1, 2):
        last = -math.inf
        for index in np.flatnonzero(splits == split):
            time = float(times[index])
            if time >= last + horizon_ms:
                keep.append(int(index))
                last = time
    return np.asarray(sorted(keep), dtype=np.int64)


def analyze_horizon(
    dataset: dict[str, Any],
    target: dict[str, Any],
    raw_features: np.ndarray,
    raw_target: np.ndarray,
    splits: np.ndarray,
    times: np.ndarray,
) -> dict[str, Any]:
    train = splits == 0
    primary = splits == 1
    transfer = splits == 2
    target_bins, target_edges, target_classes, zero_separate = quantize_target(raw_target, train)
    quantized, edges, arities = quantize_features(raw_features, train, dataset["features"])
    unconditional = fit_unconditional(target_bins[train], target_classes)
    blocks = chronological_blocks(primary, transfer)
    empty = score_subset((), quantized, target_bins, train, primary, transfer, blocks, arities, target_classes, unconditional)
    singles = []
    for feature_index, definition in enumerate(dataset["features"]):
        score = score_subset((feature_index,), quantized, target_bins, train, primary, transfer, blocks, arities, target_classes, unconditional)
        singles.append({
            "featureIndex": feature_index,
            **definition,
            **score,
        })
    singles.sort(key=lambda row: row["primaryBits"], reverse=True)
    finalists = select_finalists(singles, FINALISTS)
    finalist_indices = [row["featureIndex"] for row in finalists]
    subsets: list[dict[str, Any]] = []
    tested = 0
    total = sum(math.comb(len(finalist_indices), size) for size in range(1, min(MAX_SUBSET_SIZE, len(finalist_indices)) + 1))
    for size in range(1, min(MAX_SUBSET_SIZE, len(finalist_indices)) + 1):
        for subset in itertools.combinations(finalist_indices, size):
            score = score_subset(subset, quantized, target_bins, train, primary, transfer, blocks, arities, target_classes, unconditional)
            subsets.append({"indices": list(subset), **score})
            tested += 1
            if tested % 100 == 0:
                print(f"  {target['id']}: {tested}/{total} subsets", flush=True)
    primary_optimum = max(subsets, key=lambda row: row["primaryBits"])
    stable = [row for row in subsets if all(value > 0 for value in row["blockBits"][:2])]
    robust = max(stable, key=lambda row: row["primaryBits"]) if stable else primary_optimum
    threshold = primary_optimum["primaryBits"] - PARISMONY_BITS
    parsimonious = min(
        (row for row in subsets if row["primaryBits"] >= threshold),
        key=lambda row: (len(row["indices"]), -row["primaryBits"]),
    )
    selected = robust
    selected_set = tuple(selected["indices"])
    contributions = []
    for feature_index in selected_set:
        reduced = tuple(item for item in selected_set if item != feature_index)
        reduced_score = empty if not reduced else score_subset(
            reduced, quantized, target_bins, train, primary, transfer, blocks, arities, target_classes, unconditional,
        )
        contributions.append({
            "featureIndex": feature_index,
            "id": dataset["features"][feature_index]["id"],
            "conditionalPrimaryBits": selected["primaryBits"] - reduced_score["primaryBits"],
            "conditionalTransferBits": selected["transferBits"] - reduced_score["transferBits"],
        })
    selected_features = []
    contribution_by_index = {row["featureIndex"]: row for row in contributions}
    for feature_index in selected_set:
        definition = dict(dataset["features"][feature_index])
        definition["quantileEdges"] = edges[feature_index]
        definition["arity"] = arities[feature_index]
        definition.update(contribution_by_index[feature_index])
        selected_features.append(definition)
    return {
        "id": target["id"],
        "horizonMinutes": target["minutes"],
        "dataset": dataset["id"],
        "broadCandidateCount": len(dataset["features"]),
        "finalistCount": len(finalists),
        "testedSubsets": tested,
        "maximumSubsetSize": MAX_SUBSET_SIZE,
        "observations": {
            "train": int(train.sum()),
            "primary": int(primary.sum()),
            "transfer": int(transfer.sum()),
        },
        "target": {
            "classes": target_classes,
            "exactZeroSeparate": zero_separate,
            "edges": target_edges,
            "unconditionalLogLossBits": empty["primaryLogLossBits"],
        },
        "finalists": [{key: value for key, value in row.items() if key not in {"blockLogRatios"}} for row in finalists],
        "primaryOptimum": describe_subset(primary_optimum, dataset["features"]),
        "robustOptimum": describe_subset(robust, dataset["features"]),
        "parsimoniousWithinOneMilliBit": describe_subset(parsimonious, dataset["features"]),
        "selectedBasis": selected_features,
        "selectedScore": {key: value for key, value in selected.items() if key != "indices"},
        "singleFeatureRanking": singles,
    }


def quantize_target(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, list[float], int, bool]:
    train_values = values[train]
    zero_rate = float(np.mean(train_values == 0))
    zero_separate = zero_rate >= 0.001
    active = train_values[train_values != 0] if zero_separate else train_values
    edges = unique_quantiles(active, TARGET_ACTIVE_BINS)
    active_classes = len(edges) + 1
    if zero_separate:
        output = np.zeros(values.shape[0], dtype=np.int16)
        mask = values != 0
        output[mask] = 1 + np.searchsorted(edges, values[mask], side="right")
        return output, edges.tolist(), active_classes + 1, True
    return np.searchsorted(edges, values, side="right").astype(np.int16), edges.tolist(), active_classes, False


def quantize_features(
    values: np.ndarray,
    train: np.ndarray,
    definitions: list[dict[str, Any]],
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    output = np.empty(values.shape, dtype=np.uint8)
    all_edges: list[list[float]] = []
    arities: list[int] = []
    for column, definition in enumerate(definitions):
        if definition.get("kind") == "binary":
            output[:, column] = (values[:, column] > 0.5).astype(np.uint8)
            all_edges.append([0.5])
            arities.append(2)
            continue
        edges = unique_quantiles(values[train, column], FEATURE_BINS)
        output[:, column] = np.searchsorted(edges, values[:, column], side="right").astype(np.uint8)
        all_edges.append(edges.tolist())
        arities.append(len(edges) + 1)
    return output, all_edges, arities


def unique_quantiles(values: np.ndarray, bins: int) -> np.ndarray:
    raw = np.quantile(values[np.isfinite(values)], np.arange(1, bins) / bins)
    return np.unique(raw)


def chronological_blocks(primary: np.ndarray, transfer: np.ndarray) -> list[np.ndarray]:
    blocks = []
    for mask in (primary, transfer):
        indices = np.flatnonzero(mask)
        middle = len(indices) // 2
        for part in (indices[:middle], indices[middle:]):
            block = np.zeros(mask.shape[0], dtype=bool)
            block[part] = True
            blocks.append(block)
    return blocks


def fit_unconditional(target: np.ndarray, classes: int) -> np.ndarray:
    counts = np.bincount(target, minlength=classes).astype(np.float64)
    return (counts + ALPHA) / (counts.sum() + ALPHA * classes)


def encode_states(features: np.ndarray, subset: tuple[int, ...], arities: list[int]) -> tuple[np.ndarray, int]:
    state = np.zeros(features.shape[0], dtype=np.int64)
    states = 1
    for feature in subset:
        state = state * arities[feature] + features[:, feature]
        states *= arities[feature]
    return state, states


def score_subset(
    subset: tuple[int, ...],
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    primary: np.ndarray,
    transfer: np.ndarray,
    blocks: list[np.ndarray],
    arities: list[int],
    classes: int,
    unconditional: np.ndarray,
) -> dict[str, Any]:
    if subset:
        state, states = encode_states(features, subset, arities)
        joint = np.bincount(
            state[train] * classes + target[train],
            minlength=states * classes,
        ).reshape(states, classes).astype(np.float64)
        totals = joint.sum(axis=1)

        def ratios(mask: np.ndarray) -> np.ndarray:
            s = state[mask]
            y = target[mask]
            probability = (joint[s, y] + ALPHA) / (totals[s] + ALPHA * classes)
            return np.log2(probability / unconditional[y])
    else:
        def ratios(mask: np.ndarray) -> np.ndarray:
            return np.zeros(int(mask.sum()), dtype=np.float64)

    primary_ratios = ratios(primary)
    transfer_ratios = ratios(transfer)
    primary_probability_loss = -float(np.mean(np.log2(unconditional[target[primary]]))) - float(np.mean(primary_ratios))
    return {
        "size": len(subset),
        "primaryBits": float(np.mean(primary_ratios)),
        "transferBits": float(np.mean(transfer_ratios)),
        "pooledBits": float(np.mean(np.concatenate([primary_ratios, transfer_ratios]))),
        "primaryLogLossBits": primary_probability_loss,
        "blockBits": [float(np.mean(ratios(block))) for block in blocks],
        "positiveBlocks": int(sum(float(np.mean(ratios(block))) > 0 for block in blocks)),
        "positivePrimaryBlocks": int(sum(float(np.mean(ratios(block))) > 0 for block in blocks[:2])),
        "positiveTransferBlocks": int(sum(float(np.mean(ratios(block))) > 0 for block in blocks[2:])),
    }


def select_finalists(singles: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    families = []
    for row in singles:
        if row["family"] not in families:
            families.append(row["family"])
    for family in families:
        row = next(item for item in singles if item["family"] == family)
        selected.append(row)
        seen.add(row["featureIndex"])
        if len(selected) >= count:
            return selected
    for row in singles:
        if row["featureIndex"] not in seen:
            selected.append(row)
            seen.add(row["featureIndex"])
        if len(selected) >= count:
            break
    return selected


def describe_subset(row: dict[str, Any], definitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": [definitions[index]["id"] for index in row["indices"]],
        **{key: value for key, value in row.items() if key != "indices"},
    }


def render_report(artifact: dict[str, Any]) -> str:
    lines = [
        f"# {artifact['datasetObjective']} — 2026-08-17",
        "",
        "## Meaning of global",
        "",
        "Unrestricted mutual information is monotone, so every causal input is a trivial maximizer. This experiment instead searches for the jointly best finite basis under a declared model and search universe. Every subset is scored directly against the unconditional return distribution; no previous feature baseline is fixed.",
        "",
        f"The broad screen chooses {FINALISTS} finalists while retaining the best representative of every family. Every subset of up to {MAX_SUBSET_SIZE} finalists is then scored. Finalists and subsets are selected only on the primary selection period, with positivity required in both primary half-blocks. The later transfer period is untouched until the final confirmation.",
        "",
        "## Results",
        "",
        "| return horizon | broad candidates | exhaustive finalists | tested subsets | selected features | primary bits | transfer bits | positive blocks |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for horizon in artifact["horizons"]:
        selected = horizon["robustOptimum"]
        lines.append(
            f"| {horizon['id']} | {horizon['broadCandidateCount']} | {horizon['finalistCount']} | {horizon['testedSubsets']} | "
            f"{', '.join(selected['features'])} | {selected['primaryBits']:.8f} | {selected['transferBits']:.8f} | {selected['positiveBlocks']}/4 |"
        )
    for horizon in artifact["horizons"]:
        lines.extend([
            "",
            f"## {horizon['id']} return",
            "",
            f"Observations: {horizon['observations']['train']:,} train, {horizon['observations']['primary']:,} primary, {horizon['observations']['transfer']:,} transfer.",
            "",
            f"Primary maximum: `{', '.join(horizon['primaryOptimum']['features'])}` at {horizon['primaryOptimum']['primaryBits']:.8f} primary / {horizon['primaryOptimum']['transferBits']:.8f} transfer bits.",
            "",
            f"Validation-stable selection: `{', '.join(horizon['robustOptimum']['features'])}` at {horizon['robustOptimum']['primaryBits']:.8f} primary / {horizon['robustOptimum']['transferBits']:.8f} untouched transfer bits; primary/transfer half-blocks {', '.join(f'{value:.6f}' for value in horizon['robustOptimum']['blockBits'])}.",
            "",
            f"Smallest subset within {PARISMONY_BITS} bits of the primary maximum: `{', '.join(horizon['parsimoniousWithinOneMilliBit']['features'])}`.",
            "",
            "| input | family | parameters | lookback | availability delay | conditional primary bits | conditional transfer bits |",
            "|---|---|---|---|---|---:|---:|",
        ])
        for feature in horizon["selectedBasis"]:
            lines.append(
                f"| {feature['name']} (`{feature['id']}`) | {feature['family']} | {feature['parameters']} | {feature['lookback']} | {feature['delay']} | "
                f"{feature['conditionalPrimaryBits']:.8f} | {feature['conditionalTransferBits']:.8f} |"
            )
    lines.extend([
        "",
        "## Limits",
        "",
        "- `Global` means exhaustive only inside the explicitly reported finalist universe and maximum subset size, not over every mathematical transform of history.",
        "- Every candidate in this report is scored only on the exact common timestamp coverage declared by the dataset manifest; results do not imply stability outside that calendar span.",
        "- Quartile feature cells and a categorical return distribution make the search exact and inspectable, but a continuous model may exploit information inside cells.",
        "- Reported conditional contributions remove one coordinate from the final joint subset. They are not standalone feature scores and need not sum exactly because features interact.",
        "",
        f"Machine-readable results are stored in `{artifact['resultFile'].replace(chr(92), '/')}`.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
