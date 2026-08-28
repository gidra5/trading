from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from global_feature_basis_search import (
    FeatureGroup,
    multitask_probabilities,
    scan_quantized_batches_residual,
)
from global_feature_registry import ROOT
from search_global_btc_per_horizon import baseline_probability, score_bits
from search_global_btc_working_set import (
    incumbent_requirements,
    load_targets,
    template_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine a saved sparse BTC head with a flattened design and exact KKT checks."
    )
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--working-set", type=Path, required=True)
    parser.add_argument("--output-search", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--zero-threshold", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def active_design(
    states,
    active: list[int],
    arities: np.ndarray,
    maximum_levels: int,
    *,
    torch,
    device: str,
):
    if not active:
        return torch.empty((states.shape[0], 0), dtype=torch.float64, device=device)
    selected = torch.as_tensor(states[:, active], dtype=torch.uint8, device=device)
    columns = []
    for level in range(1, maximum_levels + 1):
        eligible = torch.as_tensor(
            [int(arities[index]) > level for index in active],
            dtype=torch.bool,
            device=device,
        )
        values = (selected == level).to(torch.float64)
        values[:, ~eligible] = 0.0
        columns.append(values)
    # Level-major -> group-major flattened order.
    return torch.stack(columns, dim=2).reshape(states.shape[0], -1)


def main() -> None:
    args = parse_args()
    search_path = resolved(args.search)
    model_path = resolved(args.model)
    working_dir = resolved(args.working_set)
    output_search = resolved(args.output_search)
    output_model = resolved(args.output_model)
    search = json.loads(search_path.read_text(encoding="utf-8"))
    horizon_index = {
        str(row["horizon"]): index for index, row in enumerate(search["horizons"])
    }
    if args.horizon not in horizon_index:
        raise ValueError(f"Search does not contain horizon {args.horizon}.")
    task = horizon_index[args.horizon]
    horizon = search["horizons"][task]
    regularization = float(horizon["final"]["regularization"])
    work = json.loads((working_dir / "manifest.json").read_text(encoding="utf-8"))
    coordinate_rows = work["coordinates"]
    with np.load(model_path) as source:
        model = {key: np.asarray(source[key]) for key in source.files}
    coordinate_ids = model["coordinate_ids"].astype(str)
    if coordinate_ids.tolist() != [str(row["id"]) for row in coordinate_rows]:
        raise ValueError("Working-set coordinates differ from the saved model.")
    arities = model["arities"].astype(np.int64)
    states = model["states"].astype(np.uint8)
    train = model["train"].astype(bool)
    labels = model["labels"].astype(np.int64)[:, task]
    offsets = model["offsets"].astype(np.float64)[:, task * 9:(task + 1) * 9]
    coefficients = model["coefficients"].astype(np.float64)
    maximum_levels = coefficients.shape[2]
    index_by_id = {feature_id: index for index, feature_id in enumerate(coordinate_ids)}
    active = sorted(
        index_by_id[str(row["id"])]
        for row in horizon["final"]["support"]
        if str(row["id"]) in index_by_id
        and float(
            np.linalg.norm(
                coefficients[
                    task,
                    index_by_id[str(row["id"])],
                    : int(arities[index_by_id[str(row["id"])]]) - 1,
                    :,
                ]
            )
        ) > args.zero_threshold
    )
    beta_by_index = {
        index: coefficients[task, index, : int(arities[index]) - 1, :].copy()
        for index in active
    }

    import torch
    import torch.nn.functional as functional

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_states = states[train]
    train_labels = torch.as_tensor(labels[train], dtype=torch.int64, device=device)
    train_offsets = torch.as_tensor(offsets[train], dtype=torch.float64, device=device)
    groups = [FeatureGroup(feature_id, int(arity), 1.0) for feature_id, arity in zip(coordinate_ids, arities)]
    started = time.perf_counter()
    total_iterations = 0
    active_maximum = np.inf
    omitted_maximum = np.inf
    final_probability = None
    final_residual = None

    for round_index in range(1, args.max_rounds + 1):
        active = sorted(set(active))
        design = active_design(
            train_states, active, arities, maximum_levels, torch=torch, device=device
        )
        initial = np.zeros((len(active), maximum_levels, 9), dtype=np.float64)
        valid = np.zeros_like(initial)
        for local, index in enumerate(active):
            levels = int(arities[index]) - 1
            initial[local, :levels] = beta_by_index.get(
                index, np.zeros((levels, 9), dtype=np.float64)
            )
            valid[local, :levels] = 1.0
        valid_tensor = torch.as_tensor(valid, dtype=torch.float64, device=device)
        beta = torch.nn.Parameter(
            torch.as_tensor(initial, dtype=torch.float64, device=device)
        )
        optimizer = torch.optim.LBFGS(
            [beta],
            lr=1.0,
            max_iter=args.max_iterations,
            tolerance_grad=max(1e-10, args.tolerance * 0.02),
            tolerance_change=1e-13,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            candidate = beta * valid_tensor
            logits = train_offsets + design @ candidate.reshape(-1, 9)
            loss = functional.cross_entropy(logits, train_labels, reduction="mean")
            loss = loss + regularization * torch.linalg.vector_norm(
                candidate, dim=(1, 2)
            ).sum()
            loss.backward()
            evaluations += 1
            if evaluations % 50 == 0:
                print(
                    f"{args.horizon} refinement round {round_index}: "
                    f"evaluations={evaluations} objective={float(loss.detach().cpu()):.9f}",
                    flush=True,
                )
            return loss

        optimizer.step(closure)
        state = optimizer.state.get(beta, {})
        total_iterations += int(state.get("n_iter", evaluations))
        with torch.no_grad():
            fitted = (beta * valid_tensor).detach()
            norms = torch.linalg.vector_norm(fitted, dim=(1, 2))
            keep_local = [
                local
                for local, norm in enumerate(norms.cpu().numpy())
                if float(norm) > args.zero_threshold
            ]
            active = [active[local] for local in keep_local]
            beta_by_index = {
                index: fitted[local, : int(arities[index]) - 1].cpu().numpy()
                for local, index in zip(keep_local, active)
            }

        design = active_design(
            train_states, active, arities, maximum_levels, torch=torch, device=device
        )
        packed = np.zeros((len(active), maximum_levels, 9), dtype=np.float64)
        for local, index in enumerate(active):
            packed[local, : int(arities[index]) - 1] = beta_by_index[index]
        packed_tensor = torch.as_tensor(packed, dtype=torch.float64, device=device)
        with torch.no_grad():
            logits = train_offsets + design @ packed_tensor.reshape(-1, 9)
            probability = torch.softmax(logits, dim=1)
            residual = probability.clone()
            residual[torch.arange(residual.shape[0], device=device), train_labels] -= 1.0
            gradient = (design.T @ residual / residual.shape[0]).reshape(
                len(active), maximum_levels, 9
            )
            norms = torch.linalg.vector_norm(packed_tensor, dim=(1, 2))
            kkt = gradient + regularization * packed_tensor / torch.clamp(
                norms[:, None, None], min=1e-30
            )
            active_residuals = torch.linalg.vector_norm(kkt, dim=(1, 2))
            active_maximum = float(
                active_residuals.max().cpu() if active_residuals.numel() else 0.0
            )
            if active_residuals.numel():
                worst_local = int(torch.argmax(active_residuals).cpu())
                worst_id = coordinate_ids[active[worst_local]]
                worst_norm = float(norms[worst_local].cpu())
            else:
                worst_id = None
                worst_norm = 0.0
            residual_numpy = residual.cpu().numpy()

        active_ids = {coordinate_ids[index] for index in active}

        def batches():
            yield SimpleNamespace(groups=groups, states=train_states)

        scan = scan_quantized_batches_residual(
            residual_numpy,
            batches,
            regularization,
            active_ids,
            add_limit=256,
            tolerance=args.tolerance,
            device=device,
        )
        omitted_maximum = float(scan.maximum_violation)
        additions = [index_by_id[feature_id] for feature_id, _ in scan.violating_groups]
        print(
            f"{args.horizon} refinement round {round_index}: active={len(active)} "
            f"activeKkt={active_maximum:.9g} omittedKkt={omitted_maximum:.9g} "
            f"additions={len(additions)} worst={worst_id} worstNorm={worst_norm:.9g}",
            flush=True,
        )
        if active_maximum <= args.tolerance and omitted_maximum <= args.tolerance:
            final_residual = residual_numpy
            break
        for index in additions:
            if index not in beta_by_index:
                beta_by_index[index] = np.zeros((int(arities[index]) - 1, 9), dtype=np.float64)
        active.extend(additions)
    else:
        raise RuntimeError(
            f"Refinement did not converge: active={active_maximum}, omitted={omitted_maximum}"
        )

    full_design = active_design(
        states, active, arities, maximum_levels, torch=torch, device=device
    )
    packed = np.zeros((len(active), maximum_levels, 9), dtype=np.float64)
    for local, index in enumerate(active):
        packed[local, : int(arities[index]) - 1] = beta_by_index[index]
    with torch.no_grad():
        full_logits = torch.as_tensor(offsets, dtype=torch.float64, device=device)
        full_logits = full_logits + full_design @ torch.as_tensor(
            packed.reshape(-1, 9), dtype=torch.float64, device=device
        )
        final_probability = torch.softmax(full_logits, dim=1).cpu().numpy()

    coefficients[task].fill(0.0)
    for index in active:
        levels = int(arities[index]) - 1
        coefficients[task, index, :levels] = beta_by_index[index]
    model["coefficients"] = coefficients.astype(np.float32)
    residual = model["residual"].astype(np.float32)
    residual[:, task * 9:(task + 1) * 9] = final_residual.astype(np.float32)
    model["residual"] = residual

    registry = json.loads(
        (ROOT / "data/benchmarks/global-feature-registry.json").read_text(encoding="utf-8")
    )
    old_support = {str(row["id"]): row for row in horizon["final"]["support"]}
    raw = np.memmap(
        working_dir / work["file"],
        dtype=work["dtype"],
        mode="r",
        shape=(work["rows"], work["columns"]),
    )
    support = []
    for index in active:
        feature_id = coordinate_ids[index]
        metadata = dict(old_support.get(feature_id, {}))
        if not metadata:
            metadata = template_policy(registry, feature_id)
            empirical = float(np.mean(np.isfinite(np.asarray(raw[:, index]))))
            metadata["declaredAvailability"] = metadata["availability"]
            metadata["empiricalAvailability"] = empirical
            metadata["availability"] = min(float(metadata["availability"]), empirical)
        metadata["id"] = feature_id
        metadata["coefficientNorm"] = float(np.linalg.norm(beta_by_index[index]))
        support.append(metadata)
    support.sort(key=lambda row: (-float(row["coefficientNorm"]), str(row["id"])))

    _, targets, splits, times = load_targets()
    transfer = splits == 2
    final_train = splits <= 1
    baseline = baseline_probability(labels, final_train)
    transfer_bits, transfer_rows = score_bits(
        labels,
        final_probability,
        baseline,
        transfer,
        times,
        int(horizon["horizonMinutes"]),
    )
    incumbent_probability = multitask_probabilities(offsets, (9,))
    incumbent_bits, _ = score_bits(
        labels,
        incumbent_probability,
        baseline,
        transfer,
        times,
        int(horizon["horizonMinutes"]),
    )
    requirements = incumbent_requirements()[args.horizon]
    selected_ids = sorted(set(requirements) | {str(row["id"]) for row in support})
    horizon["final"].update({
        "converged": True,
        "iterations": total_iterations,
        "stationarityMaximum": max(active_maximum, max(0.0, omitted_maximum)),
        "correctionSupportSize": len(support),
        "selectedRawInputCount": len(selected_ids),
        "selectedRawInputIds": selected_ids,
        "transferRows": transfer_rows,
        "transferBits": transfer_bits,
        "incumbentTransferBits": incumbent_bits,
        "transferGainOverIncumbent": transfer_bits - incumbent_bits,
        "transferConfirmationPassed": transfer_bits >= incumbent_bits - 0.001,
        "productionEligible": transfer_bits >= incumbent_bits - 0.001,
        "support": support,
        "refinement": {
            "method": "flattened double-precision active-support L-BFGS plus full working-set zero-group KKT scans",
            "sourceSearch": str(search_path.relative_to(ROOT)).replace("\\", "/"),
            "sourceModel": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "elapsedSeconds": time.perf_counter() - started,
        },
    })
    recommended: set[str] = set()
    exploratory: set[str] = set()
    for row in search["horizons"]:
        exploratory.update(row["final"]["selectedRawInputIds"])
        if row["final"].get("productionEligible"):
            recommended.update(row["final"]["selectedRawInputIds"])
        else:
            recommended.update(row["final"]["requiredIncumbentInputs"])
    search["union"]["rawInputCount"] = len(recommended)
    search["union"]["rawInputIds"] = sorted(recommended)
    search["union"]["exploratoryRawInputCountBeforeTransferConfirmation"] = len(exploratory)
    search["union"]["exploratoryRawInputIdsBeforeTransferConfirmation"] = sorted(exploratory)
    search["certificate"] = {
        "scope": "Every working-set group satisfies direct KKT at the recorded tolerance; full-registry provider rescans remain required.",
        "workingSetTolerance": args.tolerance,
        "refinedHorizon": args.horizon,
    }
    output_search.parent.mkdir(parents=True, exist_ok=True)
    output_search.write_text(json.dumps(search, indent=2) + "\n", encoding="utf-8")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_model, **model)
    print(
        json.dumps({
            "horizon": args.horizon,
            "support": len(support),
            "stationarityMaximum": horizon["final"]["stationarityMaximum"],
            "transferBits": transfer_bits,
            "gainBits": transfer_bits - incumbent_bits,
            "outputSearch": str(output_search.relative_to(ROOT)).replace("\\", "/"),
            "outputModel": str(output_model.relative_to(ROOT)).replace("\\", "/"),
        }, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
