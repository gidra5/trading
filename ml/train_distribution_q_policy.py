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

from causal_return_normalization import append_return_statistics
from distribution_q_policy import (
    ARCHITECTURE_CONTRACT,
    DistributionQNetwork,
    drifted_exposure,
    greedy_policy,
    holding_log_reward_bps,
    polyak_update,
    predicted_distribution_double_q_target,
    rebalance_log_reward_bps,
    recurrent_q_feature_width,
    recurrent_q_features,
)
from compressed_path_return_density import path_log_density_terms
from low_rank_path_matrix_density import CyclicDenseCompressedPathMatrixDensity
from return_knot_density import KnotDensityContract
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_feature_compressed_path_density import (
    ImmediateFeatureActivePathDataset,
    pad_weighted_batch,
)
from train_next_return_knot_density import PAUSE_EXIT_CODE, canonical_hash
from train_normalized_glu_next_return import Reporter, atomic_json


RUNNER_CONTRACT = "joint-path-distribution-model-based-fitted-q-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly fine-tune a path distribution and its all-action Q policy."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--replace-smoke", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--compile-transition", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def source_checkpoint_hash(file: Path) -> str:
    reference = json.loads(file.read_text(encoding="utf-8"))
    return str(reference["object"]["contentHash"])


def build_transition_model(
    plan: dict, checkpoint_file: Path, device: torch.device
) -> CyclicDenseCompressedPathMatrixDensity:
    source_plan = plan["_sourcePlan"]
    architecture = source_plan["architecture"]
    if architecture["contract"] != "cyclic-dense-compressed-path-matrix-density-v1":
        raise ValueError("Q policy currently requires the shared cyclic density model")
    saved = load_torch_checkpoint(
        checkpoint_file, map_location="cpu", weights_only=False
    )
    state = saved.get("emaModel", saved["model"])
    density = KnotDensityContract.load(
        Path(plan["repoRoot"]) / source_plan["density"]["source"],
        fit=str(int(architecture["outputKnots"])),
    )
    model = CyclicDenseCompressedPathMatrixDensity(
        state["feature_mean"],
        state["feature_std"],
        density,
        market_width=int(architecture["marketWidth"]),
        path_embedding_width=int(architecture["pathEmbeddingWidth"]),
        path_count=int(architecture["pathCount"]),
        return_count=int(architecture["returnCount"]),
        stage_block_count=int(architecture["stageBlockCount"]),
        path_compression_width=int(architecture["pathCompressionWidth"]),
        joint_compression_width=int(architecture["jointCompressionWidth"]),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture["learnableCentering"]),
        target_normalization_variance_floor=(
            None if source_plan.get("targetNormalization") is None else float(
                source_plan["targetNormalization"]["varianceFloor"]
            )
        ),
        target_normalization_center=(
            source_plan.get("targetNormalization") is None
            or source_plan["targetNormalization"].get("type")
            == "causal-trailing-log-return-zscore"
        ),
    )
    model.load_state_dict(state)
    model.train()
    return model.to(device)


def build_source_dataset(plan: dict) -> ImmediateFeatureActivePathDataset:
    source = plan["_sourcePlan"]
    normalization = source.get("targetNormalization")
    return ImmediateFeatureActivePathDataset(
        Path(plan["repoRoot"]) / source["datasetDir"],
        Path(plan["repoRoot"]) / source["historyDir"],
        int(source["architecture"]["returnCount"]),
        int(source["architecture"]["inputFeatureLags"]),
        int(source["subset"]["examples"]),
        normalization_window_seconds=(
            None if normalization is None else int(normalization["windowSeconds"])
        ),
        normalization_variance_floor=(
            1e-16 if normalization is None else float(normalization["varianceFloor"])
        ),
        normalization_statistic=(
            "log-price"
            if normalization is not None and normalization.get("type")
            == "causal-trailing-log-price-zscore-difference"
            else "log-return"
        ),
    )


def linear_curriculum(epoch: int, schedule: dict) -> float:
    if schedule.get("type") != "linear":
        raise ValueError("only linear curricula are supported")
    start = float(schedule["start"])
    end = float(schedule["end"])
    epochs = int(schedule["epochs"])
    if epochs < 1 or not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("invalid curriculum")
    if epochs == 1:
        return end
    fraction = min(max(int(epoch), 0), epochs - 1) / (epochs - 1)
    return start + fraction * (end - start)


def source_rows(
    dataset: ImmediateFeatureActivePathDataset,
    split: str,
    indexes: np.ndarray,
) -> tuple[Tensor, Tensor]:
    timeline, origins, targets, return_means, return_variances = (
        dataset.splits[split]
    )
    selected = np.asarray(indexes, dtype=np.int64)
    selected_origins = np.asarray(origins[selected], dtype=np.int64)
    offsets = np.arange(
        dataset.feature_history - 1, -1, -1, dtype=np.int64
    )
    feature_rows = np.asarray(
        timeline[selected_origins[:, None] - offsets[None, :]],
        dtype=np.float32,
    ).reshape(selected.size, dataset.base_feature_count)
    if return_means is not None and return_variances is not None:
        feature_rows = append_return_statistics(
            feature_rows, return_means[selected], return_variances[selected]
        )
    target_rows = np.asarray(targets[selected], dtype=np.float32)
    return (
        torch.from_numpy(feature_rows.copy()),
        torch.from_numpy(target_rows.copy()),
    )


def pad_indexes(indexes: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(indexes, dtype=np.int64)
    if values.size < 1 or values.size > batch_size:
        raise ValueError("invalid index batch")
    valid = np.ones(values.size, dtype=np.float32)
    if values.size == batch_size:
        return values, valid
    padding = np.resize(values, batch_size - values.size)
    return (
        np.concatenate((values, padding)),
        np.concatenate((valid, np.zeros(padding.size, dtype=np.float32))),
    )


@torch.no_grad()
def recurrent_state_statistics(
    model: CyclicDenseCompressedPathMatrixDensity,
    dataset: ImmediateFeatureActivePathDataset,
    *,
    batch_size: int,
    limit: int,
    device: torch.device,
    reporter: Reporter,
) -> tuple[Tensor, Tensor]:
    width = recurrent_q_feature_width(
        model.market_width,
        model.path_count,
        model.path_embedding_width,
        model.output_width,
    )
    total = torch.zeros(width, dtype=torch.float64)
    square = torch.zeros(width, dtype=torch.float64)
    count = 0
    input_count = 0
    model.eval()
    for features, _targets, weights in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0, limit=limit
    ):
        valid = int(weights.sum().item())
        if valid < batch_size:
            features, _targets, weights = pad_weighted_batch(
                features, _targets, weights, batch_size
            )
        emissions = model.recurrent_rollout(
            features.to(device, non_blocking=True), model.return_count
        )
        encoded = torch.cat(tuple(
            recurrent_q_features(emission)[:valid] for emission in emissions
        ), dim=0).double().cpu()
        total += encoded.sum(dim=0)
        square += encoded.square().sum(dim=0)
        count += int(encoded.shape[0])
        input_count += valid
        if input_count % (batch_size * 25) == 0:
            reporter.status(
                "estimating-recurrent-state-statistics",
                examples=input_count,
                total=limit,
            )
    if input_count != limit or count != limit * model.return_count:
        raise RuntimeError("recurrent-state statistics sample is incomplete")
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-12)
    model.train()
    return mean.float(), variance.sqrt().float()


@torch.no_grad()
def q_metrics(
    transition: CyclicDenseCompressedPathMatrixDensity,
    online: DistributionQNetwork,
    target: DistributionQNetwork,
    dataset: ImmediateFeatureActivePathDataset,
    split: str,
    actions: Tensor,
    *,
    batch_size: int,
    td_discount: float,
    td_friction_bps: float,
    policy_friction_bps: float,
    device: torch.device,
    limit: int | None = None,
) -> dict:
    transition.eval()
    online.eval()
    target.eval()
    loss_sum = 0.0
    cells = 0
    q_rows: list[np.ndarray] = []
    return_rows: list[np.ndarray] = []
    transition_count = dataset.logical_count(split)
    if limit is not None:
        transition_count = min(transition_count, int(limit))
    for start in range(0, transition_count, batch_size):
        stop = min(transition_count, start + batch_size)
        indexes, valid = pad_indexes(np.arange(start, stop), batch_size)
        current, current_targets = source_rows(dataset, split, indexes)
        emissions = transition.recurrent_rollout(
            current.to(device, non_blocking=True), 2
        )
        current_encoded = recurrent_q_features(emissions[0])
        following_encoded = recurrent_q_features(emissions[1])
        prediction = online(current_encoded)
        backup = predicted_distribution_double_q_target(
            online(following_encoded),
            target(following_encoded),
            emissions[0].log_masses,
            emissions[0].component_means,
            actions,
            discount=td_discount, friction_bps=td_friction_bps,
        ).values
        valid_tensor = torch.from_numpy(valid).to(device)
        cell_loss = functional.smooth_l1_loss(
            prediction, backup, reduction="none"
        ) * valid_tensor[:, None]
        loss_sum += float(cell_loss.sum())
        cells += int(valid_tensor.sum().item()) * prediction.shape[1]
        valid_count = stop - start
        q_rows.append(prediction[:valid_count].cpu().numpy())
        return_rows.append(current_targets[:valid_count, 0].numpy())
    q = np.concatenate(q_rows, axis=0)
    returns = np.concatenate(return_rows, axis=0).astype(np.float64)
    exposure = 0.0
    total = 0.0
    gross = 0.0
    turnover = 0.0
    trades = 0
    peak = 0.0
    max_drawdown = 0.0
    perfect = 0.0
    action_values = actions.cpu().numpy().astype(np.float64)
    for index, log_return in enumerate(returns):
        costs = np.log1p(np.maximum(
            -0.999999,
            -policy_friction_bps / 10_000.0 * np.abs(action_values - exposure),
        )) * 10_000.0
        selected = int(np.argmax(q[index] + costs))
        target_exposure = float(action_values[selected])
        rebalance = float(costs[selected])
        holding = float(np.log(max(
            1e-12, 1 + target_exposure * np.expm1(log_return)
        )) * 10_000.0)
        change = abs(target_exposure - exposure)
        turnover += change
        trades += int(change > 1e-8)
        gross += holding
        total += rebalance + holding
        peak = max(peak, total)
        max_drawdown = max(max_drawdown, peak - total)
        perfect += abs(log_return) * 10_000.0
        multiplier = np.exp(log_return)
        exposure = target_exposure * multiplier / max(
            1e-12, 1 + target_exposure * (multiplier - 1)
        )
    return {
        "tdHuber": loss_sum / max(cells, 1),
        "netLogReturnBps": total,
        "grossHoldingLogReturnBps": gross,
        "perfectMarginLogReturnBps": perfect,
        "perfectCapture": total / perfect if perfect > 0 else 0.0,
        "turnover": turnover,
        "trades": trades,
        "maximumDrawdownBps": max_drawdown,
        "finalExposure": exposure,
        "transitions": transition_count,
        "qAcrossStateStd": float(np.mean(np.std(q, axis=0))),
        "qAcrossActionRange": float(np.mean(np.ptp(q, axis=1))),
    }


def checkpoint_payload(
    transition: CyclicDenseCompressedPathMatrixDensity,
    model: DistributionQNetwork,
    target: DistributionQNetwork,
    optimizer: torch.optim.Optimizer,
    *, epoch: int, global_step: int, plan_hash: str, source_hash: str,
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
        raise ValueError("Q-policy architecture contract changed")
    plan_hash = canonical_hash(plan)
    plan_snapshot = json.loads(json.dumps(plan))
    plan["repoRoot"] = str(repo)
    run_root = (repo / plan["runDir"]).resolve()
    reporter = Reporter(run_root)
    pause_file = None if args.pause_file is None else (
        args.pause_file if args.pause_file.is_absolute() else repo / args.pause_file
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
        source_plan_file = (repo / plan["transition"]["planFile"]).resolve()
        source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
        plan["_sourcePlan"] = source_plan
        checkpoint_file = (repo / plan["transition"]["checkpoint"]).resolve()
        checkpoint_hash = source_checkpoint_hash(checkpoint_file)
        device = torch.device(plan["training"].get("device", "cuda"))
        torch.manual_seed(int(plan["training"]["seed"]))
        np.random.seed(int(plan["training"]["seed"]))
        random.seed(int(plan["training"]["seed"]))
        torch.set_float32_matmul_precision("high")
        dataset = build_source_dataset(plan)
        transition_model = build_transition_model(plan, checkpoint_file, device)
        if args.compile_transition:
            raise ValueError(
                "compilation is disabled for the explicit recurrent-state API"
            )
        batch_size = int(args.batch_size or plan["training"]["batchSize"])
        smoke_limit = None if args.smoke_batches is None else max(
            2, int(args.smoke_batches) * batch_size + 1
        )
        statistics_examples = min(
            dataset.logical_count("train"),
            int(plan["transition"].get("statisticsExamples", 65_536)),
        )
        if smoke_limit is not None:
            statistics_examples = min(statistics_examples, batch_size)
        statistics_file = run_root / "state/recurrent-q-feature-statistics.json"
        stored_statistics = (
            json.loads(statistics_file.read_text(encoding="utf-8"))
            if statistics_file.is_file() else {}
        )
        feature_width = recurrent_q_feature_width(
            transition_model.market_width,
            transition_model.path_count,
            transition_model.path_embedding_width,
            transition_model.output_width,
        )
        if stored_statistics.get("sourceCheckpointSha256") == checkpoint_hash \
                and stored_statistics.get("examples") == statistics_examples \
                and stored_statistics.get("featureWidth") == feature_width:
            input_mean = torch.tensor(stored_statistics["featureMean"])
            input_std = torch.tensor(stored_statistics["featureStd"])
        else:
            reporter.status(
                "estimating-recurrent-state-statistics",
                examples=0, total=statistics_examples,
            )
            input_mean, input_std = recurrent_state_statistics(
                transition_model, dataset,
                batch_size=batch_size, limit=statistics_examples,
                device=device, reporter=reporter,
            )
            atomic_json({
                "version": 1,
                "sourceCheckpointSha256": checkpoint_hash,
                "examples": statistics_examples,
                "featureWidth": feature_width,
                "featureMean": input_mean.tolist(),
                "featureStd": input_std.tolist(),
            }, statistics_file)
        model = DistributionQNetwork(
            feature_width,
            int(plan["architecture"]["actionCount"]),
            state_width=int(plan["architecture"]["stateWidth"]),
            input_mean=input_mean,
            input_std=input_std,
        ).to(device)
        target = model.target_copy().to(device)
        optimizer = torch.optim.AdamW([
            {
                "params": model.parameters(),
                "lr": float(plan["training"]["learningRate"]),
            },
            {
                "params": transition_model.parameters(),
                "lr": float(plan["training"]["transitionLearningRate"]),
            },
        ], weight_decay=float(plan["training"]["weightDecay"]))
        actions = torch.linspace(
            float(plan["actionSpace"]["minimumExposure"]),
            float(plan["actionSpace"]["maximumExposure"]),
            int(plan["architecture"]["actionCount"]),
            device=device,
        )
        atomic_json(
            {"planSha256": plan_hash, "plan": plan_snapshot},
            run_root / "state/plan.json",
        )
        epochs = 1 if args.smoke_batches is not None \
            else int(plan["training"]["epochs"])
        discount_schedule = plan["rl"]["discountSchedule"]
        friction_schedule = plan["rl"]["frictionScheduleBps"]
        target_discount = float(discount_schedule["end"])
        target_friction = float(friction_schedule["end"])
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
            if saved["planSha256"] != plan_hash \
                    or saved["sourceCheckpointSha256"] != checkpoint_hash \
                    or saved["runnerContract"] != RUNNER_CONTRACT:
                raise ValueError("Q checkpoint source, plan, or runner changed")
            transition_model.load_state_dict(saved["transitionModel"])
            model.load_state_dict(saved["model"])
            target.load_state_dict(saved["targetModel"])
            optimizer.load_state_dict(saved["optimizer"])
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            best = saved["best"]
        reporter.emit({
            "event": "distribution-q-training-start",
            "planId": plan["id"],
            "startEpoch": start_epoch,
            "epochs": epochs,
            "sourceCheckpointSha256": checkpoint_hash,
            "parameters": sum(value.numel() for value in model.parameters())
                + sum(value.numel() for value in transition_model.parameters()),
            "trainableTransition": True,
            "actionCount": actions.numel(),
            "objective": (
                "joint-density-and-model-based-distribution-integrated-"
                "double-fitted-q"
            ),
            "bellmanTransition": "predicted-recurrent-market-and-path-state",
            "discountSchedule": discount_schedule,
            "frictionScheduleBps": friction_schedule,
        })
        started = time.monotonic()
        train_count = dataset.logical_count("train")
        for epoch in range(start_epoch, epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status("paused", planId=plan["id"], epoch=epoch)
                raise SystemExit(PAUSE_EXIT_CODE)
            discount = linear_curriculum(epoch, discount_schedule)
            friction_bps = linear_curriculum(epoch, friction_schedule)
            transition_model.train()
            model.train()
            order = np.random.default_rng(
                int(plan["training"]["seed"]) + epoch
            ).permutation(train_count)
            td_sum = 0.0
            density_sum = 0.0
            example_count = 0
            for batch_index, start in enumerate(range(0, order.size, batch_size)):
                indexes, valid = pad_indexes(
                    order[start:start + batch_size], batch_size
                )
                current, current_targets = source_rows(dataset, "train", indexes)
                current = current.to(device, non_blocking=True)
                current_targets = current_targets.to(device, non_blocking=True)
                valid_tensor = torch.from_numpy(valid).to(device)
                emissions = transition_model.recurrent_rollout(
                    current, transition_model.return_count + 1
                )
                current_output = transition_model.output_from_emissions(
                    emissions[:transition_model.return_count]
                )
                phase = (
                    batch_index + epoch
                ) % transition_model.return_count
                current_encoded = recurrent_q_features(emissions[phase])
                prediction = model(current_encoded)
                with torch.no_grad():
                    following_encoded = recurrent_q_features(
                        emissions[phase + 1]
                    ).detach()
                    backup = predicted_distribution_double_q_target(
                        model(following_encoded),
                        target(following_encoded),
                        emissions[phase].log_masses.detach(),
                        emissions[phase].component_means.detach(),
                        actions,
                        discount=discount, friction_bps=friction_bps,
                    ).values
                td_cells = functional.smooth_l1_loss(
                    prediction, backup, reduction="none"
                )
                td_loss = (
                    td_cells * valid_tensor[:, None]
                ).sum() / (valid_tensor.sum() * prediction.shape[1])
                density_terms = path_log_density_terms(
                    current_output, current_targets, transition_model
                )
                density_loss = -(
                    density_terms * valid_tensor[:, None]
                ).sum() / (valid_tensor.sum() * transition_model.return_count)
                loss = td_loss + density_weight * density_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(
                    list(model.parameters()) + list(transition_model.parameters()),
                    float(plan["training"]["gradientClip"]),
                )
                optimizer.step()
                polyak_update(target, model, tau)
                valid_count = int(valid.sum())
                td_sum += float(td_loss.detach()) * valid_count
                density_sum += float(density_loss.detach()) * valid_count
                example_count += valid_count
                global_step += 1
                if (batch_index + 1) % 25 == 0:
                    reporter.status("training", planId=plan["id"], latest={
                        "epoch": epoch,
                        "epochs": epochs,
                        "batch": batch_index + 1,
                        "batches": math.ceil(order.size / batch_size),
                        "globalStep": global_step,
                        "tdHuber": td_sum / example_count,
                        "transitionNegativeLogLikelihood": (
                            density_sum / example_count
                        ),
                        "discount": discount,
                        "frictionBps": friction_bps,
                        "latentPhase": phase,
                        "seconds": time.monotonic() - started,
                    })
                if args.smoke_batches is not None \
                        and batch_index + 1 >= args.smoke_batches:
                    break
            should_evaluate = args.smoke_batches is not None \
                or epoch % evaluation_interval == 0 or epoch == epochs - 1
            event = {
                "event": "distribution-q-optimization-epoch",
                "epoch": epoch,
                "epochs": epochs,
                "globalStep": global_step,
                "trainTdHuber": td_sum / example_count,
                "trainTransitionNegativeLogLikelihood": density_sum / example_count,
                "discount": discount,
                "frictionBps": friction_bps,
                "transitionSource": "predicted-recurrent-market-and-path-state",
                "seconds": time.monotonic() - started,
            }
            if should_evaluate:
                validation = q_metrics(
                    transition_model, model, target,
                    dataset, "validation", actions, batch_size=batch_size,
                    td_discount=target_discount,
                    td_friction_bps=target_friction,
                    policy_friction_bps=target_friction,
                    device=device, limit=smoke_limit,
                )
                event["validationAtTargetSchedule"] = validation
                score = float(validation["netLogReturnBps"])
                if score > float(best["validationNetLogReturnBps"]):
                    best = {"validationNetLogReturnBps": score, "epoch": epoch}
                    save_torch_checkpoint(
                        checkpoint_payload(
                            transition_model, model, target, optimizer,
                            epoch=epoch, global_step=global_step,
                            plan_hash=plan_hash, source_hash=checkpoint_hash,
                            best=best,
                        ),
                        run_root / "checkpoints/best-validation-net.json",
                    )
            save_torch_checkpoint(
                checkpoint_payload(
                    transition_model, model, target, optimizer,
                    epoch=epoch, global_step=global_step,
                    plan_hash=plan_hash, source_hash=checkpoint_hash, best=best,
                ), last_file,
            )
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
        if args.smoke_batches is not None:
            reporter.status("smoke-complete", planId=plan["id"], latest=event)
            return
        best_file = run_root / "checkpoints/best-validation-net.json"
        selected = load_torch_checkpoint(
            best_file, map_location=device, weights_only=False
        )
        transition_model.load_state_dict(selected["transitionModel"])
        model.load_state_dict(selected["model"])
        target.load_state_dict(selected["targetModel"])
        test = q_metrics(
            transition_model, model, target,
            dataset, "test", actions, batch_size=batch_size,
            td_discount=target_discount,
            td_friction_bps=target_friction,
            policy_friction_bps=target_friction,
            device=device,
        )
        result = {
            "best": best,
            "testAtTargetSchedule": test,
            "globalStep": global_step,
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "distribution-q-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except SystemExit:
        raise
    except Exception as error:
        reporter.status("failed", planId=plan.get("id", "unknown"), error=repr(error))
        raise


if __name__ == "__main__":
    main()
