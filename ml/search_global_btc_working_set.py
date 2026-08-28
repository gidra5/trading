from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from global_feature_basis_search import (
    FeatureGroup,
    additive_logits,
    fit_multitask_group_lasso_torch,
    multitask_probabilities,
    multitask_residual,
    scan_quantized_batches_residual,
)
from global_feature_candidate_axis import (
    BASE_DIR,
    BaseRecentBatchProvider,
    QuantizedCandidateBatch,
    quantize_batch_fast,
)
from global_feature_registry import ROOT
from screen_global_btc_dense_candidates import joint_labels


DEFAULT_WORKING_SET = ROOT / "data/runtime-cache/global-btc-working-set"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/global-btc-working-set-search.json"
DEFAULT_MODEL = ROOT / "data/benchmarks/global-btc-working-set-model.npz"
DEFAULT_FRACTIONS = "1,0.8,0.55,0.35,0.22,0.14,0.09,0.055,0.035,0.02"
HORIZON_MINUTES = (1, 1, 15, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested chronological search over the global BTC working set.")
    parser.add_argument("--working-set", type=Path, default=DEFAULT_WORKING_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dense-active", type=int, default=512)
    parser.add_argument("--lambda-fractions", default=DEFAULT_FRACTIONS)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=2e-7)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_targets() -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    dataset = manifest["datasets"][0]
    rows = int(dataset["rows"])
    targets = np.asarray(np.memmap(
        BASE_DIR / dataset["files"]["targets"],
        dtype="<f4",
        mode="r",
        shape=(rows, int(dataset["targetCount"])),
    ), dtype=np.float64)
    splits = np.asarray(np.memmap(
        BASE_DIR / dataset["files"]["splits"], dtype="u1", mode="r", shape=(rows,)
    ))
    times = np.asarray(np.memmap(
        BASE_DIR / dataset["files"]["times"], dtype="<f8", mode="r", shape=(rows,)
    ))
    return manifest, targets, splits, times


def target_labels(targets: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    labels = []
    thresholds = []
    for task in range(targets.shape[1]):
        task_labels, task_thresholds = joint_labels(targets[:, task], train)
        labels.append(task_labels)
        thresholds.append(task_thresholds)
    return np.column_stack(labels), thresholds


def chronological_folds(times: np.ndarray, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    split = manifest["split"]
    start = np.datetime64(split["start"].removesuffix("Z"), "ms").astype(np.int64)
    train_end = np.datetime64(split["trainEndExclusive"].removesuffix("Z"), "ms").astype(np.int64)
    primary_end = np.datetime64(split["primaryEndExclusive"].removesuffix("Z"), "ms").astype(np.int64)
    day = 86_400_000
    specifications = (
        ("fold-1", start + 8 * day, start + 12 * day),
        ("fold-2", start + 12 * day, train_end),
        ("fold-3-primary", train_end, primary_end),
    )
    output = []
    for name, train_end_ms, validation_end_ms in specifications:
        train = times < train_end_ms
        validation = (times >= train_end_ms) & (times < validation_end_ms)
        output.append({
            "id": name,
            "train": train,
            "validation": validation,
            "trainEndMs": int(train_end_ms),
            "validationEndMs": int(validation_end_ms),
        })
    return output


def nonoverlapping_rows(mask: np.ndarray, times: np.ndarray, minutes: int) -> np.ndarray:
    candidates = np.flatnonzero(mask)
    if candidates.size == 0 or minutes <= 1:
        return candidates
    selected = []
    next_time = -np.inf
    separation = minutes * 60_000
    for row in candidates:
        if times[row] >= next_time:
            selected.append(int(row))
            next_time = times[row] + separation
    return np.asarray(selected, dtype=np.int64)


def baseline_probabilities(labels: np.ndarray, train: np.ndarray, classes: tuple[int, ...]) -> list[np.ndarray]:
    output = []
    for task, count in enumerate(classes):
        frequencies = np.bincount(labels[train, task], minlength=count).astype(np.float64) + 1.0
        output.append(frequencies / frequencies.sum())
    return output


def validation_bits(
    labels: np.ndarray,
    probability: np.ndarray,
    baselines: list[np.ndarray],
    mask: np.ndarray,
    times: np.ndarray,
    classes: tuple[int, ...],
) -> tuple[list[float], list[int]]:
    bits = []
    counts = []
    offset = 0
    for task, count in enumerate(classes):
        rows = nonoverlapping_rows(mask, times, HORIZON_MINUTES[task])
        selected = probability[rows, offset + labels[rows, task]]
        reference = baselines[task][labels[rows, task]]
        bits.append(float(np.mean(np.log2(np.maximum(selected, 1e-30) / np.maximum(reference, 1e-30)))))
        counts.append(int(rows.size))
        offset += count
    return bits, counts


def incumbent_requirements() -> dict[str, list[str]]:
    provider = BaseRecentBatchProvider()
    canonical_by_original = {
        str(definition["id"]): provider.feature_id(definition)
        for definition in provider.dataset["features"]
    }
    artifact = json.loads(
        (ROOT / "data/benchmarks/tiered-component-feature-bases.json").read_text(encoding="utf-8")
    )
    output = {}
    for row in artifact["targets"]:
        if row["componentId"] != "joint_zero_sign_magnitude":
            continue
        output[str(row["horizonId"])] = [
            canonical_by_original[feature_id]
            for feature_id in row["selectedBasis"]["features"]
        ]
    if set(output) != {"1s", "1m", "15m", "1h"}:
        raise ValueError("The incumbent joint-return basis is incomplete.")
    return output


def incumbent_offset_logits(
    states: np.ndarray,
    arities: list[int],
    coordinate_rows: list[dict[str, Any]],
    requirements: dict[str, list[str]],
    labels: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    index_by_id = {row["id"]: index for index, row in enumerate(coordinate_rows)}
    offset = np.zeros((states.shape[0], 36), dtype=np.float64)
    for task, horizon in enumerate(("1s", "1m", "15m", "1h")):
        encoded = np.zeros(states.shape[0], dtype=np.int64)
        multiplier = 1
        for feature_id in requirements[horizon]:
            index = index_by_id[feature_id]
            encoded = encoded * int(arities[index]) + states[:, index].astype(np.int64)
            multiplier *= int(arities[index])
        counts = np.bincount(
            encoded[train] * 9 + labels[train, task], minlength=multiplier * 9
        ).reshape(multiplier, 9).astype(np.float64)
        probability = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 4.5)
        offset[:, task * 9:(task + 1) * 9] = np.log(probability[encoded])
    return offset


def lambda_maximum(
    candidate_states: np.ndarray,
    labels: np.ndarray,
    candidate_groups: list[FeatureGroup],
    offset_logits: np.ndarray,
    device: str | None,
    max_iterations: int,
    tolerance: float,
) -> float:
    incumbent = fit_multitask_group_lasso_torch(
        np.empty((candidate_states.shape[0], 0), dtype=np.uint8),
        labels,
        [],
        (9, 9, 9, 9),
        0.0,
        offset_logits=offset_logits,
        fit_intercept=False,
        max_iterations=max_iterations,
        tolerance=tolerance,
        device=device,
    )
    residual = multitask_residual(
        np.empty((candidate_states.shape[0], 0), dtype=np.uint8),
        labels,
        incumbent,
        offset_logits=offset_logits,
    )

    def batches():
        yield QuantizedCandidateBatch(candidate_groups, candidate_states, [])

    result = scan_quantized_batches_residual(
        residual, batches, 0.0, set(), add_limit=1, device=device
    )
    return result.maximum_violation


def fit_path_for_fold(
    values: np.ndarray,
    coordinate_rows: list[dict[str, Any]],
    targets: np.ndarray,
    times: np.ndarray,
    fold: dict[str, Any],
    fractions: list[float],
    requirements: dict[str, list[str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    states, edges, arities = quantize_batch_fast(values, fold["train"], bins=4, sample_rows=None)
    groups = [FeatureGroup(row["id"], arity, 1.0) for row, arity in zip(coordinate_rows, arities)]
    labels, thresholds = target_labels(targets, fold["train"])
    offsets = incumbent_offset_logits(
        states, arities, coordinate_rows, requirements, labels, fold["train"]
    )
    train_states = states[fold["train"]]
    train_labels = labels[fold["train"]]
    maximum = lambda_maximum(
        train_states,
        train_labels,
        groups,
        offsets[fold["train"]],
        args.device,
        args.max_iterations,
        args.tolerance,
    )
    baselines = baseline_probabilities(labels, fold["train"], (9, 9, 9, 9))
    rows = []
    warm = None
    for fraction in sorted(fractions, reverse=True):
        regularization = maximum * fraction
        started = time.perf_counter()
        fit = fit_multitask_group_lasso_torch(
            train_states,
            train_labels,
            groups,
            (9, 9, 9, 9),
            regularization,
            offset_logits=offsets[fold["train"]],
            fit_intercept=False,
            initial=warm,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            device=args.device,
        )
        warm = fit
        probability = multitask_probabilities(
            offsets + additive_logits(states, fit.intercept, fit.coefficients), fit.classes
        )
        bits, counts = validation_bits(
            labels, probability, baselines, fold["validation"], times, fit.classes
        )
        support = [
            group.id for group, beta in zip(groups, fit.coefficients)
            if float(np.linalg.norm(beta)) > 1e-5
        ]
        row = {
            "lambdaFraction": fraction,
            "regularization": regularization,
            "lambdaMaximum": maximum,
            "validationBitsByHorizon": bits,
            "validationRowsByHorizon": counts,
            "meanValidationBits": float(np.mean(bits)),
            "minimumValidationBits": float(np.min(bits)),
            "supportSize": len(support),
            "iterations": fit.iterations,
            "converged": fit.converged,
            "elapsedSeconds": time.perf_counter() - started,
            "thresholdsBps": thresholds,
        }
        rows.append(row)
        print(
            f"{fold['id']} lambda={fraction:.4g} support={len(support)} "
            f"bits={','.join(f'{value:.4f}' for value in bits)} "
            f"iterations={fit.iterations} converged={fit.converged}",
            flush=True,
        )
    return rows


def choose_fraction(folds: list[dict[str, Any]], fractions: list[float]) -> tuple[float, list[dict[str, Any]]]:
    summaries = []
    for fraction in fractions:
        rows = [next(row for row in fold["path"] if row["lambdaFraction"] == fraction) for fold in folds]
        task_fold = np.asarray([row["validationBitsByHorizon"] for row in rows], dtype=np.float64)
        summaries.append({
            "lambdaFraction": fraction,
            "meanBits": float(task_fold.mean()),
            "worstTaskFoldBits": float(task_fold.min()),
            "meanSupportSize": float(np.mean([row["supportSize"] for row in rows])),
            "allConverged": all(row["converged"] for row in rows),
            "taskMeanBits": task_fold.mean(axis=0).tolist(),
            "foldMeanBits": task_fold.mean(axis=1).tolist(),
            "taskFoldBits": task_fold.tolist(),
        })
    incumbent = next((row for row in summaries if row["lambdaFraction"] == 1.0), None)
    if incumbent is None:
        raise ValueError("The lambda path must include 1.0 so the incumbent is always admissible.")
    incumbent_bits = np.asarray(incumbent["taskFoldBits"], dtype=np.float64)
    for row in summaries:
        candidate_bits = np.asarray(row["taskFoldBits"], dtype=np.float64)
        row["incumbentSafe"] = bool(np.all(candidate_bits >= incumbent_bits - 0.001))
        row["minimumGainOverIncumbent"] = float(np.min(candidate_bits - incumbent_bits))
    converged = [row for row in summaries if row["allConverged"] and row["incumbentSafe"]]
    pool = converged or [incumbent]
    best = max(pool, key=lambda row: (row["worstTaskFoldBits"], row["meanBits"], -row["meanSupportSize"]))
    # Within a practically equivalent 0.001 bits/task, prefer the stronger
    # regularization (and therefore usually smaller support).
    equivalent = [
        row for row in pool
        if row["worstTaskFoldBits"] >= best["worstTaskFoldBits"] - 0.001
        and all(
            candidate >= reference - 0.001
            for candidate, reference in zip(row["taskMeanBits"], best["taskMeanBits"])
        )
    ]
    selected = max(equivalent, key=lambda row: (row["lambdaFraction"], -row["meanSupportSize"]))
    return float(selected["lambdaFraction"]), summaries


def template_policy(registry: dict[str, Any], coordinate_id: str) -> dict[str, Any]:
    parts = coordinate_id.split("/")
    template_id = "/".join((parts[0], parts[2], parts[3], parts[4]))
    template = next((row for row in registry["templates"] if row["canonicalId"] == template_id), None)
    if template is None:
        return {"sourcePolicy": "unknown", "availability": 0.0, "acquisitionCost": 99}
    policy = next(row for row in registry["sourcePolicies"] if row["id"] == template["sourcePolicy"])
    return {
        "sourcePolicy": template["sourcePolicy"],
        "availability": policy["availability_score"],
        "acquisitionCost": policy["acquisition_cost"],
        "family": template["family"],
    }


def main() -> None:
    args = parse_args()
    work_dir = resolved(args.working_set)
    output = resolved(args.output)
    model_path = resolved(args.model)
    work = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    all_rows = work["coordinates"]
    selected_indices = [index for index, row in enumerate(all_rows) if row["source"] == "existing-recent"]
    selected_indices += [
        index for index, row in enumerate(all_rows)
        if row["source"] == "dense-minute" and int(row["denseGradientRank"]) <= args.dense_active
    ]
    coordinate_rows = [all_rows[index] for index in selected_indices]
    raw = np.memmap(
        work_dir / work["file"], dtype=work["dtype"], mode="r", shape=(work["rows"], work["columns"])
    )
    values = np.asarray(raw[:, selected_indices], dtype=np.float32)
    dataset_manifest, targets, splits, times = load_targets()
    fractions = sorted({float(value) for value in args.lambda_fractions.split(",")}, reverse=True)
    folds = chronological_folds(times, dataset_manifest)
    requirements = incumbent_requirements()
    fold_results = []
    started = time.perf_counter()
    for fold in folds:
        path = fit_path_for_fold(
            values, coordinate_rows, targets, times, fold, fractions, requirements, args
        )
        fold_results.append({
            "id": fold["id"],
            "trainRows": int(np.count_nonzero(fold["train"])),
            "validationRows": int(np.count_nonzero(fold["validation"])),
            "trainEnd": datetime.fromtimestamp(fold["trainEndMs"] / 1000, timezone.utc).isoformat(),
            "validationEnd": datetime.fromtimestamp(fold["validationEndMs"] / 1000, timezone.utc).isoformat(),
            "path": path,
        })
    selected_fraction, path_summary = choose_fraction(fold_results, fractions)

    final_train = splits <= 1
    transfer = splits == 2
    states, edges, arities = quantize_batch_fast(values, final_train, bins=4, sample_rows=None)
    groups = [FeatureGroup(row["id"], arity, 1.0) for row, arity in zip(coordinate_rows, arities)]
    labels, thresholds = target_labels(targets, final_train)
    offsets = incumbent_offset_logits(
        states, arities, coordinate_rows, requirements, labels, final_train
    )
    final_maximum = lambda_maximum(
        states[final_train],
        labels[final_train],
        groups,
        offsets[final_train],
        args.device,
        args.max_iterations,
        args.tolerance,
    )
    regularization = selected_fraction * final_maximum
    final_fit = fit_multitask_group_lasso_torch(
        states[final_train],
        labels[final_train],
        groups,
        (9, 9, 9, 9),
        regularization,
        offset_logits=offsets[final_train],
        fit_intercept=False,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        device=args.device,
    )
    probability = multitask_probabilities(
        offsets + additive_logits(states, final_fit.intercept, final_fit.coefficients), final_fit.classes
    )
    baselines = baseline_probabilities(labels, final_train, final_fit.classes)
    transfer_bits, transfer_rows = validation_bits(
        labels, probability, baselines, transfer, times, final_fit.classes
    )
    support_indices = [
        index for index, beta in enumerate(final_fit.coefficients)
        if float(np.linalg.norm(beta)) > 1e-5
    ]
    registry = json.loads((ROOT / "data/benchmarks/global-feature-registry.json").read_text(encoding="utf-8"))
    support = []
    for index in support_indices:
        policy = template_policy(registry, groups[index].id)
        support.append({
            "id": groups[index].id,
            "coefficientNorm": float(np.linalg.norm(final_fit.coefficients[index])),
            **policy,
        })
    support.sort(key=lambda row: (-row["coefficientNorm"], row["id"]))
    required_input_ids = sorted(set(feature_id for ids in requirements.values() for feature_id in ids))
    selected_input_ids = sorted(set(required_input_ids) | {row["id"] for row in support})
    selected_policies = [template_policy(registry, feature_id) for feature_id in selected_input_ids]
    residual = multitask_residual(
        states[final_train], labels[final_train], final_fit, offset_logits=offsets[final_train]
    )

    artifact = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "working-set optimum; full-registry KKT rescan still required",
        "objective": "Shared-support additive 9-state BTC return distributions at 1s, 1m, 15m, and 1h",
        "modelClass": "four equal-weight categorical softmax heads over additive four-bin feature partitions with a shared group-lasso support",
        "candidateSet": {
            "existingCoordinates": sum(row["source"] == "existing-recent" for row in coordinate_rows),
            "denseCoordinates": sum(row["source"] == "dense-minute" for row in coordinate_rows),
            "total": len(coordinate_rows),
            "denseSelection": f"top {args.dense_active} from the complete lambda-max gradient scan",
        },
        "validation": {
            "protocol": "three expanding chronological folds; 15m and 1h scores use non-overlapping validation targets; transfer week untouched until final fit",
            "lambdaFractions": fractions,
            "selectedLambdaFraction": selected_fraction,
            "pathSummary": path_summary,
            "folds": fold_results,
            "equivalenceBitsPerTask": 0.001,
        },
        "final": {
            "trainRows": int(np.count_nonzero(final_train)),
            "transferRowsByHorizon": transfer_rows,
            "thresholdsBps": thresholds,
            "lambdaMaximum": final_maximum,
            "regularization": regularization,
            "iterations": final_fit.iterations,
            "converged": final_fit.converged,
            "supportSize": len(support),
            "requiredIncumbentInputCount": len(required_input_ids),
            "selectedRawInputCount": len(selected_input_ids),
            "incumbentRequirementsByHorizon": requirements,
            "selectedRawInputIds": selected_input_ids,
            "transferBitsByHorizon": transfer_bits,
            "meanTransferBits": float(np.mean(transfer_bits)),
            "minimumTransferBits": float(np.min(transfer_bits)),
            "availabilityMinimum": min((row["availability"] for row in selected_policies), default=1.0),
            "availabilityMean": float(np.mean([row["availability"] for row in selected_policies])) if selected_policies else 1.0,
            "acquisitionCostMaximum": max((row["acquisitionCost"] for row in selected_policies), default=0),
            "support": support,
            "assetCounts": Counter(row["id"].split("/")[1] for row in support),
            "familyCounts": Counter(row.get("family", "unknown") for row in support),
        },
        "certificate": {
            "scope": "The active working set only; a complete KKT scan over every registry provider is required before calling this the registry-global convex optimum.",
            "residualRows": int(residual.shape[0]),
            "residualOutputs": int(residual.shape[1]),
        },
        "elapsedSeconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    maximum_levels = max(group.arity - 1 for group in groups)
    beta = np.zeros((len(groups), maximum_levels, sum(final_fit.classes)), dtype=np.float32)
    for index, coefficient in enumerate(final_fit.coefficients):
        beta[index, : coefficient.shape[0]] = coefficient
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        states=states.astype(np.uint8),
        labels=labels.astype(np.uint8),
        train=final_train.astype(np.uint8),
        intercept=final_fit.intercept.astype(np.float32),
        coefficients=beta,
        residual=residual.astype(np.float32),
        arities=np.asarray(arities, dtype=np.uint8),
        coordinate_ids=np.asarray([group.id for group in groups]),
        thresholds_bps=np.asarray(thresholds, dtype=np.float64),
    )
    print(f"Wrote {output.relative_to(ROOT)}", flush=True)
    print(json.dumps(artifact["final"], indent=2), flush=True)


if __name__ == "__main__":
    main()
