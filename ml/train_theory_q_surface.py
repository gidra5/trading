from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional
from torch.nn.utils import clip_grad_norm_

from compressed_path_return_density import path_log_density_terms
from distribution_q_policy import (
    drifted_exposure,
    polyak_update,
    rebalance_log_reward_bps,
)
from theory_q_surface import (
    ARCHITECTURE_CONTRACT,
    RecurrentQPrimeSurface,
    expected_action_values,
    q_prime_bellman_target,
    theory_curriculum,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_distribution_q_policy import (
    build_source_dataset,
    build_transition_model,
    pad_indexes,
    source_checkpoint_hash,
    source_rows,
)
from train_next_return_knot_density import PAUSE_EXIT_CODE, canonical_hash
from train_normalized_glu_next_return import Reporter, atomic_json


RUNNER_CONTRACT = "recurrent-return-action-q-prime-surface-trainer-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train theory-doc q' on recurrent return/action surfaces."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--replace-smoke", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the latest and stored-best checkpoints without training.",
    )
    return parser.parse_args()


def build_initialized_transition(
    plan: dict, device: torch.device
):
    repo = Path(plan["repoRoot"])
    base_plan_file = (repo / plan["transition"]["basePlanFile"]).resolve()
    base_plan = json.loads(base_plan_file.read_text(encoding="utf-8"))
    plan["_sourcePlan"] = base_plan
    base_checkpoint = (repo / plan["transition"]["baseCheckpoint"]).resolve()
    model = build_transition_model(plan, base_checkpoint, device)
    source_file = (repo / plan["transition"]["checkpoint"]).resolve()
    saved = load_torch_checkpoint(source_file, map_location=device, weights_only=False)
    if "transitionModel" not in saved:
        raise ValueError("q' source checkpoint lacks the fine-tuned transition model")
    model.load_state_dict(saved["transitionModel"])
    model.train()
    return model, source_checkpoint_hash(source_file)


def weighted_surface_mse(
    prediction: Tensor,
    target: Tensor,
    log_masses: Tensor,
    valid: Tensor,
) -> Tensor:
    if prediction.shape != target.shape \
            or prediction.shape[:2] != log_masses.shape:
        raise ValueError("q' loss tensors have incompatible shapes")
    weights = log_masses.exp()[:, :, None] * valid[:, None, None]
    return (
        (prediction - target).square() * weights
    ).sum() / (valid.sum() * prediction.shape[2]).clamp_min(1.0)


@torch.no_grad()
def q_surface_metrics(
    transition,
    online: RecurrentQPrimeSurface,
    target: RecurrentQPrimeSurface,
    dataset,
    split: str,
    actions: Tensor,
    *,
    batch_size: int,
    epoch: int,
    device: torch.device,
    limit: int | None = None,
) -> dict:
    transition.eval()
    online.eval()
    target.eval()
    schedule = theory_curriculum(epoch)
    count = dataset.logical_count(split)
    if limit is not None:
        count = min(count, int(limit))
    loss_sum = 0.0
    examples = 0
    q_rows: list[np.ndarray] = []
    return_rows: list[np.ndarray] = []
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        indexes, valid = pad_indexes(np.arange(start, stop), batch_size)
        features, targets = source_rows(dataset, split, indexes)
        emissions = transition.recurrent_rollout(
            features.to(device, non_blocking=True), 2
        )
        prediction = online(emissions[0].conditioned_market)
        next_surface = target(emissions[1].conditioned_market)
        backup = q_prime_bellman_target(
            emissions[0],
            next_surface,
            emissions[1].component_means,
            actions,
            discount=schedule.discount,
            temperature=schedule.temperature,
            effective_steps=schedule.effective_steps,
            friction_bps=schedule.friction_bps,
        ).values
        valid_tensor = torch.from_numpy(valid).to(device)
        loss = weighted_surface_mse(
            prediction, backup, emissions[0].log_masses, valid_tensor
        )
        valid_count = stop - start
        loss_sum += float(loss) * valid_count
        examples += valid_count
        q_rows.append(
            expected_action_values(emissions[0], prediction)[:valid_count]
            .cpu().numpy()
        )
        return_rows.append(targets[:valid_count, 0].numpy())

    q = np.concatenate(q_rows, axis=0)
    returns = np.concatenate(return_rows, axis=0).astype(np.float64)
    action_values = actions.cpu().numpy().astype(np.float64)
    exposure = 0.0
    total = 0.0
    gross = 0.0
    turnover = 0.0
    trades = 0
    peak = 0.0
    maximum_drawdown = 0.0
    perfect = 0.0
    for index, log_return in enumerate(returns):
        costs = np.log1p(np.maximum(
            -0.999999,
            -schedule.friction_bps / 10_000.0
            * np.abs(action_values - exposure),
        ))
        selected = int(np.argmax(q[index] + costs))
        target_exposure = float(action_values[selected])
        rebalance_bps = float(costs[selected] * 10_000.0)
        holding_bps = float(np.log(max(
            1e-12, 1 + target_exposure * np.expm1(log_return)
        )) * 10_000.0)
        change = abs(target_exposure - exposure)
        turnover += change
        trades += int(change > 1e-8)
        gross += holding_bps
        total += rebalance_bps + holding_bps
        peak = max(peak, total)
        maximum_drawdown = max(maximum_drawdown, peak - total)
        perfect += abs(log_return) * 10_000.0
        multiplier = np.exp(log_return)
        exposure = target_exposure * multiplier / max(
            1e-12, 1 + target_exposure * (multiplier - 1)
        )
    return {
        "qPrimeMse": loss_sum / max(examples, 1),
        "netLogReturnBps": total,
        "grossHoldingLogReturnBps": gross,
        "perfectMarginLogReturnBps": perfect,
        "perfectCapture": total / perfect if perfect > 0 else 0.0,
        "turnover": turnover,
        "trades": trades,
        "maximumDrawdownBps": maximum_drawdown,
        "finalExposure": exposure,
        "transitions": count,
        "qAcrossStateStd": float(np.mean(np.std(q, axis=0))),
        "qAcrossActionRange": float(np.mean(np.ptp(q, axis=1))),
    }


def checkpoint_payload(
    transition,
    model: RecurrentQPrimeSurface,
    target: RecurrentQPrimeSurface,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    plan_hash: str,
    source_hash: str,
    best: dict,
) -> dict:
    return {
        "transitionModel": transition.state_dict(),
        "model": model.state_dict(),
        "targetModel": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "globalStep": global_step,
        "planSha256": plan_hash,
        "sourceCheckpointSha256": source_hash,
        "runnerContract": RUNNER_CONTRACT,
        "best": best,
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan["architecture"].get("contract") != ARCHITECTURE_CONTRACT:
        raise ValueError("q' surface architecture contract changed")
    plan_hash = canonical_hash(plan)
    plan_snapshot = json.loads(json.dumps(plan))
    plan["repoRoot"] = str(repo)
    run_root = (repo / plan["runDir"]).resolve()
    reporter = Reporter(run_root)
    pause_file = None if args.pause_file is None else (
        args.pause_file if args.pause_file.is_absolute()
        else repo / args.pause_file
    )
    if args.replace_smoke:
        status_file = run_root / "state/status.json"
        status = json.loads(status_file.read_text(encoding="utf-8")) \
            if status_file.is_file() else {}
        if status.get("stage") != "smoke-complete":
            raise ValueError("--replace-smoke requires a completed smoke run")
        for file in (
            run_root / "logs/training.jsonl",
            run_root / "checkpoints/last.json",
            run_root / "checkpoints/best-validation-net.json",
            run_root / "state/plan.json",
            status_file,
        ):
            file.unlink(missing_ok=True)
    try:
        device = torch.device(plan["training"].get("device", "cuda"))
        seed = int(plan["training"]["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.set_float32_matmul_precision("high")
        transition, source_hash = build_initialized_transition(plan, device)
        dataset = build_source_dataset(plan)
        action_count = int(plan["architecture"]["actionCount"])
        model = RecurrentQPrimeSurface(
            transition.market_width,
            transition.output_width,
            action_count,
            dropout=float(plan["architecture"]["dropout"]),
        ).to(device)
        target = model.target_copy().to(device)
        optimizer = torch.optim.AdamW([
            {
                "params": model.parameters(),
                "lr": float(plan["training"]["learningRate"]),
            },
            {
                "params": transition.parameters(),
                "lr": float(plan["training"]["transitionLearningRate"]),
            },
        ], weight_decay=float(plan["training"]["weightDecay"]))
        actions = torch.linspace(
            float(plan["actionSpace"]["minimumExposure"]),
            float(plan["actionSpace"]["maximumExposure"]),
            action_count,
            device=device,
        )
        batch_size = int(args.batch_size or plan["training"]["batchSize"])
        smoke_limit = None if args.smoke_batches is None else max(
            2, int(args.smoke_batches) * batch_size
        )
        atomic_json(
            {"planSha256": plan_hash, "plan": plan_snapshot},
            run_root / "state/plan.json",
        )
        total_epochs = 1 if args.smoke_batches is not None \
            else int(plan["training"]["epochs"])
        density_weight = float(plan["training"]["transitionDensityLossWeight"])
        tau = float(plan["training"]["targetUpdateTau"])
        evaluation_interval = int(plan["training"]["evaluationIntervalEpochs"])
        best = {"validationNetLogReturnBps": -math.inf, "epoch": -1}
        global_step = 0
        start_epoch = 0
        last_file = run_root / "checkpoints/last.json"
        if checkpoint_exists(last_file) and args.smoke_batches is None:
            saved = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            resume_compatible_hashes = {
                str(value)
                for value in plan["training"].get(
                    "resumeCompatiblePlanSha256", []
                )
            }
            checkpoint_plan_compatible = (
                saved["planSha256"] == plan_hash
                or saved["planSha256"] in resume_compatible_hashes
            )
            if not checkpoint_plan_compatible \
                    or saved["sourceCheckpointSha256"] != source_hash \
                    or saved["runnerContract"] != RUNNER_CONTRACT:
                raise ValueError("q' checkpoint source, plan, or runner changed")
            transition.load_state_dict(saved["transitionModel"])
            model.load_state_dict(saved["model"])
            target.load_state_dict(saved["targetModel"])
            optimizer.load_state_dict(saved["optimizer"])
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            best = saved["best"]
        if args.evaluate_only:
            if start_epoch == 0:
                raise ValueError("--evaluate-only requires a durable checkpoint")
            candidates = []
            candidate_files = [
                ("latest", last_file),
                (
                    "stored-best-validation-net",
                    run_root / "checkpoints/best-validation-net.json",
                ),
            ]
            seen_epochs: set[int] = set()
            for name, checkpoint_file in candidate_files:
                if not checkpoint_exists(checkpoint_file):
                    continue
                candidate = load_torch_checkpoint(
                    checkpoint_file, map_location=device, weights_only=False
                )
                candidate_epoch = int(candidate["epoch"])
                if candidate_epoch in seen_epochs:
                    continue
                seen_epochs.add(candidate_epoch)
                transition.load_state_dict(candidate["transitionModel"])
                model.load_state_dict(candidate["model"])
                target.load_state_dict(candidate["targetModel"])
                validation = q_surface_metrics(
                    transition,
                    model,
                    target,
                    dataset,
                    "validation",
                    actions,
                    batch_size=batch_size,
                    epoch=total_epochs - 1,
                    device=device,
                    limit=smoke_limit,
                )
                candidates.append({
                    "name": name,
                    "checkpoint": str(checkpoint_file.relative_to(repo)),
                    "epoch": candidate_epoch,
                    "globalStep": int(candidate["globalStep"]),
                    "validation": validation,
                })
            selected = max(
                candidates,
                key=lambda value: float(
                    value["validation"]["netLogReturnBps"]
                ),
            )
            selected_checkpoint = load_torch_checkpoint(
                repo / selected["checkpoint"],
                map_location=device,
                weights_only=False,
            )
            transition.load_state_dict(selected_checkpoint["transitionModel"])
            model.load_state_dict(selected_checkpoint["model"])
            target.load_state_dict(selected_checkpoint["targetModel"])
            test_metrics = q_surface_metrics(
                transition,
                model,
                target,
                dataset,
                "test",
                actions,
                batch_size=batch_size,
                epoch=total_epochs - 1,
                device=device,
                limit=smoke_limit,
            )
            final_schedule = theory_curriculum(total_epochs - 1)
            result = {
                "planId": plan["id"],
                "evaluation": "validation-select-then-sealed-test",
                "targetSchedule": {
                    "temperature": final_schedule.temperature,
                    "frictionBps": final_schedule.friction_bps,
                    "horizonSeconds": final_schedule.horizon_seconds,
                    "stepSeconds": final_schedule.step_seconds,
                    "effectiveSteps": final_schedule.effective_steps,
                    "discount": final_schedule.discount,
                },
                "candidates": candidates,
                "selected": {
                    "name": selected["name"],
                    "checkpoint": selected["checkpoint"],
                    "epoch": selected["epoch"],
                    "globalStep": selected["globalStep"],
                    "validation": selected["validation"],
                    "test": test_metrics,
                },
            }
            atomic_json(result, run_root / "state/final-evaluation.json")
            reporter.emit({
                "event": "theory-q-prime-final-evaluation",
                **result,
            })
            reporter.status(
                "complete",
                planId=plan["id"],
                completedEpoch=selected["epoch"],
                finalEvaluation=result,
            )
            return
        reporter.emit({
            "event": "theory-q-prime-training-start",
            "planId": plan["id"],
            "startEpoch": start_epoch,
            "epochs": total_epochs,
            "sourceCheckpointSha256": source_hash,
            "parameters": sum(p.numel() for p in model.parameters())
                + sum(p.numel() for p in transition.parameters()),
            "objective": "theory-q-prime-nlse-squared-bellman",
            "dropout": float(plan["architecture"]["dropout"]),
        })
        started = time.monotonic()
        train_count = dataset.logical_count("train")
        for epoch in range(start_epoch, total_epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status("paused", planId=plan["id"], epoch=epoch)
                raise SystemExit(PAUSE_EXIT_CODE)
            schedule = theory_curriculum(epoch)
            transition.train()
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(train_count)
            q_sum = 0.0
            density_sum = 0.0
            example_count = 0
            for batch_index, start in enumerate(range(0, order.size, batch_size)):
                indexes, valid = pad_indexes(
                    order[start:start + batch_size], batch_size
                )
                features, path_targets = source_rows(dataset, "train", indexes)
                features = features.to(device, non_blocking=True)
                path_targets = path_targets.to(device, non_blocking=True)
                valid_tensor = torch.from_numpy(valid).to(device)
                emissions = transition.recurrent_rollout(
                    features, transition.return_count + 1
                )
                phase = (batch_index + epoch) % transition.return_count
                prediction = model(emissions[phase].conditioned_market)
                with torch.no_grad():
                    target.eval()
                    next_surface = target(
                        emissions[phase + 1].conditioned_market.detach()
                    )
                    backup = q_prime_bellman_target(
                        emissions[phase],
                        next_surface,
                        emissions[phase + 1].component_means.detach(),
                        actions,
                        discount=schedule.discount,
                        temperature=schedule.temperature,
                        effective_steps=schedule.effective_steps,
                        friction_bps=schedule.friction_bps,
                    ).values
                q_loss = weighted_surface_mse(
                    prediction,
                    backup,
                    emissions[phase].log_masses.detach(),
                    valid_tensor,
                )
                density_output = transition.output_from_emissions(
                    emissions[:transition.return_count]
                )
                density_terms = path_log_density_terms(
                    density_output, path_targets, transition
                )
                density_loss = -(
                    density_terms * valid_tensor[:, None]
                ).sum() / (valid_tensor.sum() * transition.return_count)
                loss = q_loss + density_weight * density_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(
                    list(model.parameters()) + list(transition.parameters()),
                    float(plan["training"]["gradientClip"]),
                )
                optimizer.step()
                polyak_update(target, model, tau)
                valid_count = int(valid.sum())
                q_sum += float(q_loss.detach()) * valid_count
                density_sum += float(density_loss.detach()) * valid_count
                example_count += valid_count
                global_step += 1
                if (batch_index + 1) % 25 == 0:
                    reporter.status("training", planId=plan["id"], latest={
                        "epoch": epoch,
                        "epochs": total_epochs,
                        "batch": batch_index + 1,
                        "batches": math.ceil(order.size / batch_size),
                        "globalStep": global_step,
                        "qPrimeMse": q_sum / example_count,
                        "transitionNegativeLogLikelihood": (
                            density_sum / example_count
                        ),
                        "temperature": schedule.temperature,
                        "effectiveTemperature": (
                            schedule.temperature
                            * math.sqrt(schedule.effective_steps)
                        ),
                        "frictionBps": schedule.friction_bps,
                        "horizonSeconds": schedule.horizon_seconds,
                        "stepSeconds": schedule.step_seconds,
                        "discount": schedule.discount,
                        "latentPhase": phase,
                        "seconds": time.monotonic() - started,
                    })
                if args.smoke_batches is not None \
                        and batch_index + 1 >= args.smoke_batches:
                    break
            should_evaluate = args.smoke_batches is not None \
                or epoch % evaluation_interval == 0 or epoch == total_epochs - 1
            event = {
                "event": "theory-q-prime-optimization-epoch",
                "epoch": epoch,
                "epochs": total_epochs,
                "globalStep": global_step,
                "trainQPrimeMse": q_sum / example_count,
                "trainTransitionNegativeLogLikelihood": density_sum / example_count,
                "temperature": schedule.temperature,
                "effectiveTemperature": (
                    schedule.temperature * math.sqrt(schedule.effective_steps)
                ),
                "frictionBps": schedule.friction_bps,
                "horizonSeconds": schedule.horizon_seconds,
                "stepSeconds": schedule.step_seconds,
                "effectiveSteps": schedule.effective_steps,
                "discount": schedule.discount,
                "seconds": time.monotonic() - started,
            }
            if should_evaluate:
                validation = q_surface_metrics(
                    transition,
                    model,
                    target,
                    dataset,
                    "validation",
                    actions,
                    batch_size=batch_size,
                    epoch=511,
                    device=device,
                    limit=smoke_limit,
                )
                event["validationAtTargetSchedule"] = validation
                score = float(validation["netLogReturnBps"])
                if score > float(best["validationNetLogReturnBps"]):
                    best = {"validationNetLogReturnBps": score, "epoch": epoch}
                    save_torch_checkpoint(
                        checkpoint_payload(
                            transition, model, target, optimizer,
                            epoch=epoch, global_step=global_step,
                            plan_hash=plan_hash, source_hash=source_hash,
                            best=best,
                        ),
                        run_root / "checkpoints/best-validation-net.json",
                    )
            save_torch_checkpoint(
                checkpoint_payload(
                    transition, model, target, optimizer,
                    epoch=epoch, global_step=global_step,
                    plan_hash=plan_hash, source_hash=source_hash, best=best,
                ),
                last_file,
            )
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
        if args.smoke_batches is not None:
            reporter.status("smoke-complete", planId=plan["id"], latest=event)
            return
        selected = load_torch_checkpoint(
            run_root / "checkpoints/best-validation-net.json",
            map_location=device,
            weights_only=False,
        )
        transition.load_state_dict(selected["transitionModel"])
        model.load_state_dict(selected["model"])
        target.load_state_dict(selected["targetModel"])
        test = q_surface_metrics(
            transition,
            model,
            target,
            dataset,
            "test",
            actions,
            batch_size=batch_size,
            epoch=511,
            device=device,
        )
        result = {"best": best, "testAtSelectedSchedule": test, "globalStep": global_step}
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "theory-q-prime-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except SystemExit:
        raise
    except Exception as error:
        reporter.status("failed", planId=plan.get("id", "unknown"), error=repr(error))
        raise


if __name__ == "__main__":
    main()
