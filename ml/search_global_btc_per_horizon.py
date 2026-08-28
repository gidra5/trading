from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from global_feature_basis_search import (
    FeatureGroup,
    MultiTaskAdditiveFit,
    additive_logits,
    fit_multitask_group_lasso_torch,
    fit_multitask_group_lasso_active_set_torch,
    multitask_probabilities,
    multitask_residual,
    scan_quantized_batches_residual,
)
from global_feature_candidate_axis import QuantizedCandidateBatch, quantize_batch_fast
from global_feature_registry import ROOT
from screen_global_btc_dense_candidates import joint_labels
from search_global_btc_working_set import (
    DEFAULT_WORKING_SET,
    chronological_folds,
    incumbent_requirements,
    load_targets,
    nonoverlapping_rows,
    resolved,
    template_policy,
)


DEFAULT_OUTPUT = ROOT / "data/benchmarks/global-btc-per-horizon-working-set-search.json"
DEFAULT_MODEL = ROOT / "data/benchmarks/global-btc-per-horizon-working-set-model.npz"
DEFAULT_FRACTIONS = "1,0.8,0.55,0.35,0.22,0.14,0.09,0.055,0.035"
HORIZONS = (("1s", 1), ("1m", 1), ("15m", 15), ("1h", 60))


def atomic_write_result_pair(
    output_path: Path,
    artifact: dict[str, Any],
    model_path: Path,
    model_arrays: dict[str, np.ndarray],
) -> None:
    """Commit a search JSON only after its complete model is durable on disk.

    Long convex refits can be interrupted externally. Writing both products
    through same-directory temporary files keeps an interrupted attempt from
    masquerading as a completed search, while the JSON rename acts as the
    completion marker for downstream robustness and KKT jobs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.time_ns()}"
    output_temporary = output_path.with_name(f".{output_path.name}.{nonce}.tmp")
    model_temporary = model_path.with_name(f".{model_path.name}.{nonce}.tmp.npz")
    try:
        output_temporary.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(model_temporary, **model_arrays)
        os.replace(model_temporary, model_path)
        os.replace(output_temporary, output_path)
    finally:
        output_temporary.unlink(missing_ok=True)
        model_temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent robust global-basis correction search by BTC horizon.")
    parser.add_argument("--working-set", type=Path, default=DEFAULT_WORKING_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dense-active", type=int, default=512)
    parser.add_argument(
        "--all-coordinates",
        action="store_true",
        help="Use every coordinate materialized in the working set. Expanded working sets are already screened.",
    )
    parser.add_argument("--lambda-fractions", default=DEFAULT_FRACTIONS)
    parser.add_argument(
        "--horizons",
        default=None,
        help="Optional comma-separated subset of horizons to fit and serialize.",
    )
    for horizon in ("1s", "1m", "15m", "1h"):
        parser.add_argument(
            f"--lambda-fractions-{horizon}",
            default=None,
            help=f"Optional comma-separated lambda grid override for {horizon}.",
        )
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--final-max-iterations", type=int, default=5_000)
    parser.add_argument("--max-active-rounds", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--path-stationarity-tolerance", type=float, default=1e-4)
    parser.add_argument("--stationarity-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--reuse-folds",
        type=Path,
        default=None,
        help="Reuse completed chronological fold paths and only reselect/refit final models.",
    )
    parser.add_argument(
        "--fixed-regularizations-from",
        type=Path,
        default=None,
        help="Keep absolute per-horizon regularizations fixed during KKT expansion.",
    )
    parser.add_argument("--warm-search", type=Path, default=None)
    parser.add_argument("--warm-model", type=Path, default=None)
    parser.add_argument(
        "--equivalence-rule",
        choices=("fixed", "paired-one-se"),
        default="fixed",
        help=(
            "Define equal validation quality by the fixed 0.001-bit per-fold band or "
            "the paired one-standard-error rule before availability/cost/size tie breaks."
        ),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def align_warm_coefficients(
    current_ids: np.ndarray,
    current_arities: np.ndarray,
    warm_ids: np.ndarray,
    warm_arities: np.ndarray,
    warm_coefficients: np.ndarray,
) -> list[np.ndarray]:
    """Map a saved solution into a KKT-expanded coordinate ledger.

    Expansion must retain every warm coordinate with the same quantization
    arity. Newly discovered groups start at zero, preserving the previous
    optimum and avoiding a cold refit after each streamed registry scan.
    """
    current = [str(value) for value in current_ids]
    warm = [str(value) for value in warm_ids]
    if len(set(current)) != len(current) or len(set(warm)) != len(warm):
        raise ValueError("Warm and current coordinate IDs must be unique.")
    current_index = {feature_id: index for index, feature_id in enumerate(current)}
    missing = [feature_id for feature_id in warm if feature_id not in current_index]
    if missing:
        raise ValueError(
            f"Expanded working set dropped {len(missing)} warm coordinates."
        )
    warm_index = {feature_id: index for index, feature_id in enumerate(warm)}
    outputs = int(warm_coefficients.shape[-1])
    aligned: list[np.ndarray] = []
    for index, feature_id in enumerate(current):
        levels = int(current_arities[index]) - 1
        source = warm_index.get(feature_id)
        if source is None:
            aligned.append(np.zeros((levels, outputs), dtype=np.float64))
            continue
        if int(warm_arities[source]) != int(current_arities[index]):
            raise ValueError(f"Warm arity differs for {feature_id}.")
        aligned.append(
            np.asarray(warm_coefficients[source, :levels, :], dtype=np.float64)
        )
    return aligned


def task_offset_logits(
    states: np.ndarray,
    arities: list[int],
    coordinate_rows: list[dict[str, Any]],
    required_ids: list[str],
    labels: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    index_by_id = {row["id"]: index for index, row in enumerate(coordinate_rows)}
    encoded = np.zeros(states.shape[0], dtype=np.int64)
    state_count = 1
    for feature_id in required_ids:
        index = index_by_id[feature_id]
        encoded = encoded * int(arities[index]) + states[:, index].astype(np.int64)
        state_count *= int(arities[index])
    counts = np.bincount(
        encoded[train] * 9 + labels[train], minlength=state_count * 9
    ).reshape(state_count, 9).astype(np.float64)
    probability = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 4.5)
    return np.log(probability[encoded])


def baseline_probability(labels: np.ndarray, train: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels[train], minlength=9).astype(np.float64) + 0.5
    return counts / counts.sum()


def score_bits(
    labels: np.ndarray,
    probability: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    horizon_minutes: int,
) -> tuple[float, int]:
    rows = nonoverlapping_rows(mask, times, horizon_minutes)
    ratio = np.log2(
        np.maximum(probability[rows, labels[rows]], 1e-30)
        / np.maximum(baseline[labels[rows]], 1e-30)
    )
    return float(np.mean(ratio)), int(rows.size)


def lambda_maximum(
    states: np.ndarray,
    labels: np.ndarray,
    groups: list[FeatureGroup],
    offset: np.ndarray,
    args: argparse.Namespace,
) -> float:
    empty_states = np.empty((states.shape[0], 0), dtype=np.uint8)
    incumbent = fit_multitask_group_lasso_torch(
        empty_states,
        labels[:, None],
        [],
        (9,),
        0.0,
        offset_logits=offset,
        fit_intercept=False,
        max_iterations=2,
        tolerance=args.tolerance,
        device=args.device,
    )
    residual = multitask_residual(
        empty_states, labels[:, None], incumbent, offset_logits=offset
    )

    def batches():
        # Chunk the lambda-max gradient scan so a wide active set does not
        # monopolize commodity-GPU memory alongside other training jobs.
        for start in range(0, len(groups), 512):
            end = min(start + 512, len(groups))
            yield QuantizedCandidateBatch(groups[start:end], states[:, start:end], [])

    result = scan_quantized_batches_residual(
        residual, batches, 0.0, set(), add_limit=1, device=args.device
    )
    return result.maximum_violation


def fit_path(
    states: np.ndarray,
    arities: list[int],
    coordinate_rows: list[dict[str, Any]],
    returns: np.ndarray,
    times: np.ndarray,
    fold: dict[str, Any],
    required_ids: list[str],
    horizon_minutes: int,
    fractions: list[float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    labels, thresholds = joint_labels(returns, fold["train"])
    offset = task_offset_logits(
        states, arities, coordinate_rows, required_ids, labels, fold["train"]
    )
    groups = [FeatureGroup(row["id"], arity, 1.0) for row, arity in zip(coordinate_rows, arities)]
    train_states = states[fold["train"]]
    train_labels = labels[fold["train"]]
    maximum = lambda_maximum(
        train_states, train_labels, groups, offset[fold["train"]], args
    )
    baseline = baseline_probability(labels, fold["train"])
    rows = []
    warm = None
    for fraction in sorted(fractions, reverse=True):
        started = time.perf_counter()
        regularization = maximum * fraction
        fit = fit_multitask_group_lasso_active_set_torch(
            train_states,
            train_labels[:, None],
            groups,
            (9,),
            regularization,
            offset_logits=offset[fold["train"]],
            fit_intercept=False,
            initial=warm,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            stationarity_tolerance=args.path_stationarity_tolerance,
            kkt_tolerance=args.path_stationarity_tolerance,
            device=args.device,
        )
        warm = fit
        probability = multitask_probabilities(
            offset + additive_logits(states, fit.intercept, fit.coefficients), (9,)
        )
        bits, validation_rows = score_bits(
            labels, probability, baseline, fold["validation"], times, horizon_minutes
        )
        support_ids = [
            group.id for group, beta in zip(groups, fit.coefficients)
            if float(np.linalg.norm(beta)) > 1e-5
        ]
        support = len(support_ids)
        rows.append({
            "lambdaFraction": fraction,
            "regularization": regularization,
            "lambdaMaximum": maximum,
            "validationBits": bits,
            "validationRows": validation_rows,
            "supportSize": support,
            "supportIds": support_ids,
            "iterations": fit.iterations,
            "converged": fit.converged,
            "stationarityMaximum": fit.stationarity_maximum,
            "thresholdsBps": thresholds,
            "elapsedSeconds": time.perf_counter() - started,
        })
        print(
            f"lambda={fraction:.4g} support={support} bits={bits:.5f} "
            f"iterations={fit.iterations} stationarity={fit.stationarity_maximum:.3g} "
            f"converged={fit.converged}",
            flush=True,
        )
    return rows


def choose_fraction(
    folds: list[dict[str, Any]],
    fractions: list[float],
    policy_by_id: dict[str, dict[str, Any]],
    equivalence_rule: str = "fixed",
) -> tuple[float, list[dict[str, Any]]]:
    summaries = []
    for fraction in fractions:
        rows = [next(row for row in fold["path"] if row["lambdaFraction"] == fraction) for fold in folds]
        fold_bits = [float(row["validationBits"]) for row in rows]
        support_ids = sorted({
            feature_id
            for row in rows
            for feature_id in row.get("supportIds", [])
        })
        policies = [policy_by_id[feature_id] for feature_id in support_ids]
        summaries.append({
            "lambdaFraction": fraction,
            "foldBits": fold_bits,
            "meanBits": float(np.mean(fold_bits)),
            "worstFoldBits": float(np.min(fold_bits)),
            "meanSupportSize": float(np.mean([row["supportSize"] for row in rows])),
            "allConverged": all(row["converged"] for row in rows),
            "foldSupportUnionSize": len(support_ids),
            "availabilityMinimum": min((row["availability"] for row in policies), default=1.0),
            "availabilityMean": float(np.mean([row["availability"] for row in policies])) if policies else 1.0,
            "acquisitionCostMaximum": max((row["acquisitionCost"] for row in policies), default=0),
            "acquisitionCostMean": float(np.mean([row["acquisitionCost"] for row in policies])) if policies else 0.0,
        })
    incumbent = next(row for row in summaries if row["lambdaFraction"] == 1.0)
    incumbent_bits = np.asarray(incumbent["foldBits"])
    for row in summaries:
        gain = np.asarray(row["foldBits"]) - incumbent_bits
        row["minimumGainOverIncumbent"] = float(gain.min())
        row["incumbentSafe"] = bool(np.all(gain >= -0.001))
    pool = [row for row in summaries if row["allConverged"] and row["incumbentSafe"]] or [incumbent]
    # Fold information levels are not comparable in absolute terms: a later,
    # more volatile block can have a higher incumbent score without being an
    # easier robustness test.  Optimize mean held-out information only after
    # enforcing the per-fold incumbent floor, then use the worst incremental
    # gain as the tie breaker.
    best = max(
        pool,
        key=lambda row: (
            row["meanBits"],
            row["minimumGainOverIncumbent"],
            row["worstFoldBits"],
        ),
    )
    if equivalence_rule == "fixed":
        equivalent = [
            row for row in pool
            if all(
                candidate >= reference - 0.001
                for candidate, reference in zip(row["foldBits"], best["foldBits"])
            )
        ]
    elif equivalence_rule == "paired-one-se":
        best_gain = np.asarray(best["foldBits"], dtype=np.float64) - incumbent_bits
        standard_error = (
            float(np.std(best_gain, ddof=1) / np.sqrt(best_gain.size))
            if best_gain.size > 1 else 0.0
        )
        quality_floor = float(best["meanBits"] - standard_error)
        for row in summaries:
            row["bestPairedGainStandardError"] = standard_error
            row["oneStandardErrorQualityFloor"] = quality_floor
            row["oneStandardErrorEligible"] = bool(
                row in pool and float(row["meanBits"]) >= quality_floor
            )
        equivalent = [row for row in pool if row["oneStandardErrorEligible"]]
    else:
        raise ValueError(f"Unknown equivalence rule: {equivalence_rule}")
    selected = max(equivalent, key=lambda row: (
        row["availabilityMinimum"],
        -row["acquisitionCostMaximum"],
        -row["foldSupportUnionSize"],
        row["availabilityMean"],
        -row["acquisitionCostMean"],
        row["lambdaFraction"],
    ))
    return float(selected["lambdaFraction"]), summaries


def main() -> None:
    args = parse_args()
    work_dir = resolved(args.working_set)
    output_path = resolved(args.output)
    model_path = resolved(args.model)
    work = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    requested_horizons = (
        [value.strip() for value in args.horizons.split(",") if value.strip()]
        if args.horizons else [name for name, _ in HORIZONS]
    )
    if len(requested_horizons) != len(set(requested_horizons)):
        raise ValueError("Requested horizons must be unique.")
    horizon_index = {name: index for index, (name, _) in enumerate(HORIZONS)}
    horizon_minutes = dict(HORIZONS)
    unknown_horizons = set(requested_horizons) - set(horizon_index)
    if unknown_horizons:
        raise ValueError(f"Unknown horizons: {sorted(unknown_horizons)}")
    selected_horizons = [
        (horizon_index[name], name, horizon_minutes[name])
        for name in requested_horizons
    ]
    all_rows = work["coordinates"]
    if args.all_coordinates:
        selected_indices = list(range(len(all_rows)))
    else:
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
    manifest, targets, splits, times = load_targets()
    requirements = incumbent_requirements()
    registry = json.loads((ROOT / "data/benchmarks/global-feature-registry.json").read_text(encoding="utf-8"))
    policy_by_id = {}
    for index, row in enumerate(coordinate_rows):
        policy = template_policy(registry, row["id"])
        empirical_availability = float(np.mean(np.isfinite(values[:, index])))
        policy["declaredAvailability"] = policy["availability"]
        policy["empiricalAvailability"] = empirical_availability
        policy["availability"] = min(policy["availability"], empirical_availability)
        policy_by_id[row["id"]] = policy
    default_fractions = sorted(
        {float(value) for value in args.lambda_fractions.split(",")}, reverse=True
    )
    fractions_by_horizon = {}
    for _, horizon, _ in selected_horizons:
        override = getattr(args, f"lambda_fractions_{horizon}")
        grid_values = default_fractions if override is None else sorted(
            {float(value) for value in override.split(",")}, reverse=True
        )
        if 1.0 not in grid_values:
            raise ValueError(f"lambda fractions for {horizon} must include 1.0.")
        fractions_by_horizon[horizon] = grid_values
    started = time.perf_counter()
    results = []
    if args.reuse_folds is not None:
        reuse_path = resolved(args.reuse_folds)
        reused = json.loads(reuse_path.read_text(encoding="utf-8"))
        reused_total = int(reused["candidateSet"]["total"])
        if reused_total != len(coordinate_rows):
            if args.fixed_regularizations_from is None or len(coordinate_rows) < reused_total:
                raise ValueError("Reused folds were computed on an incompatible candidate set.")
        reused_by_horizon = {row["horizon"]: row for row in reused["horizons"]}
        for _, horizon, horizon_minutes_value in selected_horizons:
            source = reused_by_horizon[horizon]
            folds = copy.deepcopy(source["folds"])
            fractions = fractions_by_horizon[horizon]
            observed = {
                float(row["lambdaFraction"])
                for fold in folds for row in fold["path"]
            }
            if set(fractions) != observed:
                raise ValueError(f"Reused fold grid differs for {horizon}.")
            selected_fraction, path_summary = choose_fraction(
                folds, fractions, policy_by_id, args.equivalence_rule
            )
            results.append({
                "horizon": horizon,
                "horizonMinutes": horizon_minutes_value,
                "selectedLambdaFraction": selected_fraction,
                "pathSummary": path_summary,
                "folds": folds,
            })
            print(
                f"Reused {horizon} folds; selected lambda={selected_fraction:.4g}",
                flush=True,
            )
    else:
        fold_definitions = chronological_folds(times, manifest)
        # Quantization is causal to each fold and shared across horizon-specific
        # searches, avoiding four redundant copies of the preprocessing.
        fold_states = []
        for fold in fold_definitions:
            states, _, arities = quantize_batch_fast(
                values, fold["train"], bins=4, sample_rows=None
            )
            fold_states.append((states, arities))
        for task, horizon, horizon_minutes_value in selected_horizons:
            fractions = fractions_by_horizon[horizon]
            print(f"Starting {horizon}", flush=True)
            folds = []
            for fold, (states, arities) in zip(fold_definitions, fold_states):
                print(f"{horizon} {fold['id']}", flush=True)
                path = fit_path(
                    states,
                    arities,
                    coordinate_rows,
                    targets[:, task],
                    times,
                    fold,
                    requirements[horizon],
                    horizon_minutes_value,
                    fractions,
                    args,
                )
                folds.append({
                    "id": fold["id"],
                    "trainRows": int(np.count_nonzero(fold["train"])),
                    "validationRows": int(np.count_nonzero(fold["validation"])),
                    "path": path,
                })
            selected_fraction, path_summary = choose_fraction(
                folds, fractions, policy_by_id, args.equivalence_rule
            )
            results.append({
                "horizon": horizon,
                "horizonMinutes": horizon_minutes_value,
                "selectedLambdaFraction": selected_fraction,
                "pathSummary": path_summary,
                "folds": folds,
            })

    final_train = splits <= 1
    transfer = splits == 2
    final_states, _, final_arities = quantize_batch_fast(values, final_train, bins=4, sample_rows=None)
    groups = [FeatureGroup(row["id"], arity, 1.0) for row, arity in zip(coordinate_rows, final_arities)]
    warm_by_horizon = None
    if (args.warm_search is None) != (args.warm_model is None):
        raise ValueError("--warm-search and --warm-model must be provided together.")
    if args.warm_search is not None:
        warm_search_path = resolved(args.warm_search)
        warm_model_path = resolved(args.warm_model)
        warm_search = json.loads(warm_search_path.read_text(encoding="utf-8"))
        warm_horizon_index = {
            str(row["horizon"]): index
            for index, row in enumerate(warm_search["horizons"])
        }
        missing_warm_horizons = set(requested_horizons) - set(warm_horizon_index)
        if missing_warm_horizons:
            raise ValueError(f"Warm model is missing horizons: {sorted(missing_warm_horizons)}")
        with np.load(warm_model_path) as warm_model:
            warm_ids = np.asarray(warm_model["coordinate_ids"]).astype(str)
            warm_arities = np.asarray(warm_model["arities"], dtype=np.int64)
            warm_intercepts = np.asarray(warm_model["intercepts"], dtype=np.float64)
            warm_coefficients = np.asarray(warm_model["coefficients"], dtype=np.float64)
        current_ids = np.asarray([group.id for group in groups])
        warm_by_horizon = {}
        for horizon in requested_horizons:
            warm_task = warm_horizon_index[horizon]
            warm_by_horizon[horizon] = MultiTaskAdditiveFit(
                intercept=np.asarray(warm_intercepts[warm_task], dtype=np.float64),
                coefficients=align_warm_coefficients(
                    current_ids,
                    np.asarray(final_arities, dtype=np.int64),
                    warm_ids,
                    warm_arities,
                    warm_coefficients[warm_task],
                ),
                classes=(9,),
                objective=np.inf,
                iterations=0,
                converged=False,
            )
    residual_parts = []
    labels_parts = []
    offset_parts = []
    intercept_parts = []
    coefficient_parts = []
    thresholds_parts = []
    union_ids: set[str] = set()
    fixed_by_horizon = None
    if args.fixed_regularizations_from is not None:
        fixed_path = resolved(args.fixed_regularizations_from)
        fixed_artifact = json.loads(fixed_path.read_text(encoding="utf-8"))
        fixed_by_horizon = {
            row["horizon"]: row for row in fixed_artifact["horizons"]
        }
        current_ids = {row["id"] for row in coordinate_rows}
        fixed_ids = {
            feature_id
            for row in fixed_artifact["horizons"]
            if row["horizon"] in requested_horizons
            for feature_id in (
                list(row["final"]["requiredIncumbentInputs"])
                + [support["id"] for support in row["final"]["support"]]
            )
        }
        missing_fixed = fixed_ids - current_ids
        if missing_fixed:
            raise ValueError(
                f"Expanded working set is missing {len(missing_fixed)} fixed-model coordinates."
            )
    for serialized_task, result in enumerate(results):
        horizon = result["horizon"]
        task = horizon_index[horizon]
        labels, thresholds = joint_labels(targets[:, task], final_train)
        offset = task_offset_logits(
            final_states, final_arities, coordinate_rows, requirements[horizon], labels, final_train
        )
        if fixed_by_horizon is None:
            maximum = lambda_maximum(
                final_states[final_train], labels[final_train], groups, offset[final_train], args
            )
            regularization = maximum * result["selectedLambdaFraction"]
        else:
            fixed = fixed_by_horizon[horizon]["final"]
            maximum = float(fixed["lambdaMaximum"])
            regularization = float(fixed["regularization"])
        def report_active_round(progress: dict[str, object]) -> None:
            print(
                f"Final {horizon} active round={progress['round']} "
                f"groups={progress['activeGroups']} additions={progress['additions']} "
                f"inner_iterations={progress['innerIterations']} "
                f"total_iterations={progress['totalIterations']} "
                f"stationarity={float(progress['activeStationarityMaximum']):.3g} "
                f"omitted_violation={float(progress['omittedMaximumViolation']):.3g} "
                f"converged={progress['innerConverged']}",
                flush=True,
            )

        fit = fit_multitask_group_lasso_active_set_torch(
            final_states[final_train],
            labels[final_train][:, None],
            groups,
            (9,),
            regularization,
            offset_logits=offset[final_train],
            fit_intercept=False,
            initial=(warm_by_horizon or {}).get(horizon),
            max_iterations=args.final_max_iterations,
            tolerance=args.tolerance,
            stationarity_tolerance=args.stationarity_tolerance,
            kkt_tolerance=args.stationarity_tolerance,
            prune_converged_zeros=True,
            max_active_rounds=args.max_active_rounds,
            device=args.device,
            progress_callback=report_active_round,
        )
        probability = multitask_probabilities(
            offset + additive_logits(final_states, fit.intercept, fit.coefficients), (9,)
        )
        baseline = baseline_probability(labels, final_train)
        transfer_bits, transfer_rows = score_bits(
            labels, probability, baseline, transfer, times, result["horizonMinutes"]
        )
        incumbent_probability = multitask_probabilities(offset, (9,))
        incumbent_transfer_bits, _ = score_bits(
            labels, incumbent_probability, baseline, transfer, times, result["horizonMinutes"]
        )
        transfer_confirmation_passed = transfer_bits >= incumbent_transfer_bits - 0.001
        production_eligible = bool(fit.converged and transfer_confirmation_passed)
        support = []
        for group, beta in zip(groups, fit.coefficients):
            if float(np.linalg.norm(beta)) <= 1e-5:
                continue
            support.append({
                "id": group.id,
                "coefficientNorm": float(np.linalg.norm(beta)),
                **policy_by_id[group.id],
            })
        support.sort(key=lambda row: (-row["coefficientNorm"], row["id"]))
        selected_ids = set(requirements[horizon]) | {row["id"] for row in support}
        union_ids.update(selected_ids)
        result["final"] = {
            "lambdaMaximum": maximum,
            "regularization": regularization,
            "converged": fit.converged,
            "iterations": fit.iterations,
            "stationarityMaximum": fit.stationarity_maximum,
            "thresholdsBps": thresholds,
            "requiredIncumbentInputs": requirements[horizon],
            "correctionSupportSize": len(support),
            "selectedRawInputCount": len(selected_ids),
            "selectedRawInputIds": sorted(selected_ids),
            "transferRows": transfer_rows,
            "transferBits": transfer_bits,
            "incumbentTransferBits": incumbent_transfer_bits,
            "transferGainOverIncumbent": transfer_bits - incumbent_transfer_bits,
            "transferConfirmationPassed": transfer_confirmation_passed,
            "productionEligible": production_eligible,
            "support": support,
        }
        residual_parts.append(multitask_residual(
            final_states[final_train], labels[final_train, None], fit, offset_logits=offset[final_train]
        ))
        labels_parts.append(labels)
        offset_parts.append(offset)
        intercept_parts.append(fit.intercept)
        maximum_levels = max(group.arity - 1 for group in groups)
        beta = np.zeros((len(groups), maximum_levels, 9), dtype=np.float32)
        for index, coefficient in enumerate(fit.coefficients):
            beta[index, : coefficient.shape[0]] = coefficient
        coefficient_parts.append(beta)
        thresholds_parts.append(thresholds)
        print(
            f"Final {horizon}: support={len(support)} transfer={transfer_bits:.5f} "
            f"incumbent={incumbent_transfer_bits:.5f}",
            flush=True,
        )

    recommended_ids: set[str] = set()
    for result in results:
        if result["final"]["productionEligible"]:
            recommended_ids.update(result["final"]["selectedRawInputIds"])
        else:
            recommended_ids.update(result["final"]["requiredIncumbentInputs"])
    union_policies = [policy_by_id[feature_id] for feature_id in recommended_ids]
    candidate_sources = Counter(str(row.get("source", "unknown")) for row in coordinate_rows)
    artifact = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "per-horizon working-set optimum; full-registry KKT rescans still required",
        "objective": "Independent convex conditional-density correction search for BTC "
        + ", ".join(requested_horizons)
        + " returns",
        "modelClass": "incumbent smoothed joint-state log probabilities plus an additive four-bin group-lasso log-odds correction",
        "candidateSet": {
            "total": len(coordinate_rows),
            "bySource": dict(sorted(candidate_sources.items())),
        },
        "selection": {
            "protocol": "three expanding chronological folds; each horizon independently must remain within 0.001 bits of its incumbent on every fold",
            "equivalenceRule": args.equivalence_rule,
            "lambdaFractionsByHorizon": fractions_by_horizon,
            "transferUse": "confirmation only; never used to select lambda or support",
            "reusedFoldArtifact": (
                str(resolved(args.reuse_folds).relative_to(ROOT)).replace("\\", "/")
                if args.reuse_folds is not None else None
            ),
            "fixedRegularizationArtifact": (
                str(resolved(args.fixed_regularizations_from).relative_to(ROOT)).replace("\\", "/")
                if args.fixed_regularizations_from is not None else None
            ),
            "serializedHorizons": requested_horizons,
            "warmSearchArtifact": (
                str(resolved(args.warm_search).relative_to(ROOT)).replace("\\", "/")
                if args.warm_search is not None else None
            ),
            "warmModelArtifact": (
                str(resolved(args.warm_model).relative_to(ROOT)).replace("\\", "/")
                if args.warm_model is not None else None
            ),
        },
        "horizons": results,
        "union": {
            "rawInputCount": len(recommended_ids),
            "rawInputIds": sorted(recommended_ids),
            "exploratoryRawInputCountBeforeTransferConfirmation": len(union_ids),
            "exploratoryRawInputIdsBeforeTransferConfirmation": sorted(union_ids),
            "availabilityMinimum": min((row["availability"] for row in union_policies), default=1.0),
            "availabilityMean": float(np.mean([row["availability"] for row in union_policies])) if union_policies else 1.0,
            "acquisitionCostMaximum": max((row["acquisitionCost"] for row in union_policies), default=0),
            "assetCounts": Counter(feature_id.split("/")[1] for feature_id in recommended_ids),
        },
        "certificate": {
            "scope": "Each selected correction is optimal only within the active working set at its chosen lambda. Full provider KKT scans remain required.",
        },
        "elapsedSeconds": time.perf_counter() - started,
    }
    atomic_write_result_pair(
        output_path,
        artifact,
        model_path,
        {
            "states": final_states.astype(np.uint8),
            "arities": np.asarray(final_arities, dtype=np.uint8),
            "coordinate_ids": np.asarray([group.id for group in groups]),
            "labels": np.column_stack(labels_parts).astype(np.uint8),
            "offsets": np.column_stack(offset_parts).astype(np.float32),
            "intercepts": np.vstack(intercept_parts).astype(np.float32),
            "coefficients": np.stack(coefficient_parts).astype(np.float32),
            "residual": np.column_stack(residual_parts).astype(np.float32),
            "train": final_train.astype(np.uint8),
            "thresholds_bps": np.asarray(thresholds_parts, dtype=np.float64),
        },
    )
    print(f"Wrote {output_path.relative_to(ROOT)}", flush=True)
    print(json.dumps({
        "unionRawInputs": artifact["union"]["rawInputCount"],
        "horizons": [
            {
                "horizon": row["horizon"],
                "lambdaFraction": row["selectedLambdaFraction"],
                "support": row["final"]["correctionSupportSize"],
                "transferBits": row["final"]["transferBits"],
                "transferGainOverIncumbent": row["final"]["transferGainOverIncumbent"],
            }
            for row in results
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
