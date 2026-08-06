from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from differentiable_exposure_value_oracle import (
    DifferentiableExposureValueOracle,
    DifferentiableExposureValueOracleConfig,
)
from multiscale_candle_resolution import (
    MultiscaleCandleResolutionDataset,
    trim_shards_for_resolution_history,
)
from multiscale_next_return import (
    TrainingPositionGlu,
    optimizer_parameter_groups,
    parameter_count,
)
from next_return_dataset import (
    HISTORY_RETURN_COUNT,
    ExampleShard,
    select_example_shards,
)
from oracle_distribution_path import (
    MEAN_PLUS_P50_OBJECTIVE_CONTRACT,
    OBJECTIVE_CONTRACT,
    OracleDistributionMetricAccumulator,
    oracle_forward_kl_loss,
    oracle_mean_plus_p50_kl_loss,
)
from train_normalized_glu_next_return import (
    Reporter,
    atomic_json,
    canonical_fingerprint,
    iso_now,
    iter_device_batches,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)


RUNNER_CONTRACT = "minute-return-path-through-differentiable-oracle-kl-v1"
ARCHITECTURE_CONTRACT = "two-layer-normalized-glu-fifteen-minute-path-v1"
RESOLUTION = "1m"
HORIZON_CANDLES = 15
RESOLUTION_SECONDS = 60


@dataclass(frozen=True)
class PathNormalization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def validate(self) -> None:
        if self.feature_mean.shape != (1, HISTORY_RETURN_COUNT) \
                or self.feature_std.shape != self.feature_mean.shape \
                or self.target_mean.shape != (HORIZON_CANDLES,) \
                or self.target_std.shape != self.target_mean.shape:
            raise ValueError("oracle path normalization shapes are invalid")
        for values in self.__dict__.values():
            if not np.isfinite(values).all():
                raise ValueError("oracle path normalization must be finite")
        if bool((self.feature_std <= 0).any()) \
                or bool((self.target_std <= 0).any()):
            raise ValueError("oracle path normalization scales must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a two-layer GLU path forecast through the differentiable "
            "exposure-value oracle using distribution KL."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def validate_plan(plan: dict) -> None:
    required = (
        "id", "sourceDatasetDir", "historyDir", "datasetDir", "runDir",
        "testExamples", "testTailOffsetExamples", "comparisonMaxWindow",
        "objective", "oracle", "architecture", "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("oracle distribution path plan is incomplete")
    objective = plan["objective"]
    if objective.get("contract") not in (
        OBJECTIVE_CONTRACT, MEAN_PLUS_P50_OBJECTIVE_CONTRACT
    ) \
            or not math.isfinite(float(objective.get("probabilityFloor", 0))) \
            or float(objective.get("probabilityFloor", 0)) <= 0:
        raise ValueError("oracle distribution objective is invalid")
    if objective.get("contract") == MEAN_PLUS_P50_OBJECTIVE_CONTRACT:
        for name in ("meanKlWeight", "p50KlWeight"):
            if not math.isfinite(float(objective.get(name, -1))) \
                    or float(objective.get(name, -1)) < 0:
                raise ValueError(f"oracle distribution {name} is invalid")
        if float(objective["meanKlWeight"]) \
                + float(objective["p50KlWeight"]) <= 0:
            raise ValueError("oracle distribution loss weights are empty")
    architecture = plan["architecture"]
    if architecture.get("contract") != ARCHITECTURE_CONTRACT \
            or architecture.get("widths") != [512, 512] \
            or architecture.get("inputNormalization") != "training-position":
        raise ValueError("oracle path architecture must be the two-layer GLU")
    oracle = plan["oracle"]
    if int(oracle.get("holdingPeriodCandles", 0)) != 1 \
            or int(oracle.get("decisionDelayCandles", 0)) != 1 \
            or int(oracle.get("valueHorizonCandles", 0)) != HORIZON_CANDLES:
        raise ValueError("oracle must use the complete fifteen-minute path")
    training = plan["training"]
    for name in (
        "epochs", "batchSize", "evaluationBatchSize",
        "normalizationBatchSize", "earlyStoppingPatience", "seed",
    ):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"oracle path training {name} must be positive")


def oracle_config(plan: dict) -> DifferentiableExposureValueOracleConfig:
    value = plan["oracle"]
    return DifferentiableExposureValueOracleConfig(
        holding_period_steps=int(value["holdingPeriodCandles"]),
        decision_delay_steps=int(value["decisionDelayCandles"]),
        value_horizon_steps=int(value["valueHorizonCandles"]),
        friction=float(value["friction"]),
        grid_size=int(value["gridSize"]),
        temperature=float(value["temperature"]),
        min_exposure=float(value["minExposure"]),
        max_exposure=float(value["maxExposure"]),
        max_effective_exposure=float(value["maxEffectiveExposure"]),
        quote_borrow_rate=float(value["quoteBorrowRatePerCandle"]),
        asset_borrow_rate=float(value["assetBorrowRatePerCandle"]),
    )


def uses_p50_objective(plan: dict) -> bool:
    return plan["objective"]["contract"] == MEAN_PLUS_P50_OBJECTIVE_CONTRACT


def add_selection_objective(metrics: dict, plan: dict) -> dict:
    if uses_p50_objective(plan):
        percentiles = metrics.get("klPercentiles")
        if not isinstance(percentiles, dict):
            raise ValueError("P50 objective requires validation KL percentiles")
        metrics["selectionObjective"] = (
            float(plan["objective"]["meanKlWeight"])
            * float(metrics["klDivergence"])
            + float(plan["objective"]["p50KlWeight"])
            * float(percentiles["p50"])
        )
    else:
        metrics["selectionObjective"] = float(metrics["klDivergence"])
    return metrics


def corpus_fingerprint(shards: dict[str, list[ExampleShard]]) -> str:
    payload = {
        "contract": (
            "past-120-aligned-minute-returns-to-next-15-minute-returns-"
            "oracle-source-splits-common-3M-warmup-v1"
        ),
        "resolution": RESOLUTION,
        "horizonCandles": HORIZON_CANDLES,
        "shards": [
            {
                "split": split,
                "start": shard.decision_time_start,
                "count": shard.count,
                "date": shard.date,
                "rowOffset": shard.row_offset,
            }
            for split in ("train", "validation", "test")
            for shard in shards[split]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def training_normalization(
    dataset: MultiscaleCandleResolutionDataset,
    batch_size: int,
) -> PathNormalization:
    feature_sum = np.zeros((1, HISTORY_RETURN_COUNT), np.float64)
    feature_square = np.zeros_like(feature_sum)
    target_sum = np.zeros(HORIZON_CANDLES, np.float64)
    target_square = np.zeros_like(target_sum)
    total = 0.0
    for feature_tensor, target_tensor, weight_tensor in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0
    ):
        features = feature_tensor.numpy().astype(np.float64, copy=False)
        targets = target_tensor.numpy().astype(np.float64, copy=False)
        weights = weight_tensor.numpy().astype(np.float64, copy=False)
        feature_sum += np.einsum("i,icj->cj", weights, features)
        feature_square += np.einsum(
            "i,icj->cj", weights, np.square(features)
        )
        target_sum += np.einsum("i,ij->j", weights, targets)
        target_square += np.einsum(
            "i,ij->j", weights, np.square(targets)
        )
        total += float(weights.sum(dtype=np.float64))
    if int(round(total)) != dataset.logical_count("train"):
        raise RuntimeError("oracle path normalization missed training examples")

    def mean_std(total_sum: np.ndarray, square_sum: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:
        mean = total_sum / total
        variance = square_sum / total - np.square(mean)
        return (
            mean.astype(np.float32),
            np.sqrt(np.maximum(variance, 1e-24)).astype(np.float32),
        )

    feature_mean, feature_std = mean_std(feature_sum, feature_square)
    target_mean, target_std = mean_std(target_sum, target_square)
    result = PathNormalization(
        feature_mean, feature_std, target_mean, target_std
    )
    result.validate()
    return result


def normalization_json(value: PathNormalization) -> dict:
    return {
        name: getattr(value, name).tolist()
        for name in value.__dataclass_fields__
    }


def normalization_from_json(value: dict) -> PathNormalization:
    result = PathNormalization(**{
        name: np.asarray(value[name], dtype=np.float32)
        for name in PathNormalization.__dataclass_fields__
    })
    result.validate()
    return result


def build_model(normalization: PathNormalization, architecture: dict) \
        -> TrainingPositionGlu:
    return TrainingPositionGlu(
        torch.from_numpy(normalization.feature_mean.reshape(-1)),
        torch.from_numpy(normalization.feature_std.reshape(-1)),
        torch.from_numpy(normalization.target_mean),
        torch.from_numpy(normalization.target_std),
        widths=tuple(int(value) for value in architecture["widths"]),
        dropout=float(architecture["dropout"]),
        dropout_rate=float(architecture["dropoutRate"]),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
    )


def build_optimizers(
    model: TrainingPositionGlu,
    training: dict,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, ...]:
    muon_parameters, adamw_parameters = optimizer_parameter_groups(model)  # type: ignore[arg-type]
    config = training["optimizer"]
    muon = config["muon"]
    adamw = config["adamw"]
    return (
        torch.optim.Muon(
            muon_parameters,
            lr=float(training["learningRate"]),
            weight_decay=float(muon["weightDecay"]),
            momentum=float(muon["momentum"]),
            nesterov=bool(muon["nesterov"]),
            ns_steps=int(muon["newtonSchulzSteps"]),
            eps=float(muon["epsilon"]),
            adjust_lr_fn=str(muon["adjustLearningRate"]),
        ),
        torch.optim.AdamW(
            adamw_parameters,
            lr=float(training["learningRate"]),
            betas=tuple(float(value) for value in adamw["betas"]),
            eps=float(adamw["epsilon"]),
            weight_decay=float(adamw["weightDecay"]),
            fused=device.type == "cuda",
        ),
    )


def oracle_probabilities(
    oracle: DifferentiableExposureValueOracle,
    returns: Tensor,
) -> Tensor:
    # Keep the Bellman recurrence in FP32 even while the GLU uses BF16 AMP.
    with torch.autocast(
        device_type=returns.device.type,
        enabled=False,
    ):
        return oracle.forward_from_log_returns(returns.float()).probabilities


@torch.no_grad()
def evaluate(
    model: TrainingPositionGlu,
    oracle: DifferentiableExposureValueOracle,
    dataset: MultiscaleCandleResolutionDataset,
    split: str,
    *,
    batch_size: int,
    probability_floor: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    constant_prediction: Tensor | None = None,
    track_kl_percentiles: bool = False,
    plan: dict | None = None,
) -> dict:
    model.eval()
    metrics = OracleDistributionMetricAccumulator(
        oracle.grid, probability_floor, track_kl_percentiles
    )
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ), device,
    ):
        if constant_prediction is None:
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = model(features.flatten(start_dim=1))
        else:
            prediction = constant_prediction.expand(targets.shape[0], -1)
        predicted_probabilities = oracle_probabilities(oracle, prediction)
        target_probabilities = oracle_probabilities(oracle, targets)
        metrics.add(
            prediction, targets, predicted_probabilities,
            target_probabilities, weights,
        )
    result = metrics.result()
    return add_selection_objective(result, plan) if plan is not None else result


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    plan_sha = canonical_fingerprint(plan)
    layout = training_storage_layout(repo_root)
    source_root = require_under(
        (repo_root / plan["sourceDatasetDir"]).resolve(),
        layout.datasets, "sourceDatasetDir",
    )
    history_root = require_under(
        (repo_root / plan["historyDir"]).resolve(),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    dataset_root = require_under(
        (repo_root / plan["datasetDir"]).resolve(),
        layout.datasets, "datasetDir",
    )
    run_dir = require_under(
        (repo_root / plan["runDir"]).resolve(), layout.runs, "runDir"
    )
    source_manifest = json.loads(
        (source_root / "dataset.json").read_text(encoding="utf-8")
    )
    actual_span_ms = (
        HISTORY_RETURN_COUNT + HORIZON_CANDLES
    ) * RESOLUTION_SECONDS * 1_000
    shards = select_example_shards(
        source_manifest,
        test_count=int(plan["testExamples"]),
        test_tail_offset=int(plan["testTailOffsetExamples"]),
        horizon_return_count=HORIZON_CANDLES * RESOLUTION_SECONDS,
        cross_split_purge_ms=actual_span_ms,
    )
    first_day = min(file.stem for file in history_root.glob("*.json"))
    shards = trim_shards_for_resolution_history(
        shards,
        first_history_day=first_day,
        resolution=RESOLUTION,
        max_window=str(plan["comparisonMaxWindow"]),
    )
    dataset = MultiscaleCandleResolutionDataset(
        shards, history_root,
        resolution=RESOLUTION,
        max_window=RESOLUTION,
        horizon_candle_count=HORIZON_CANDLES,
    )
    counts = {
        split: dataset.logical_count(split)
        for split in ("train", "validation", "test")
    }
    fingerprint = corpus_fingerprint(shards)
    selection = {
        "event": "oracle-distribution-path-dataset-selected",
        "planId": plan["id"],
        "counts": counts,
        "resolution": RESOLUTION,
        "historyCandles": HISTORY_RETURN_COUNT,
        "horizonCandles": HORIZON_CANDLES,
        "corpusFingerprint": fingerprint,
        "testPolicy": "sealed-validation-only-no-evaluation",
    }
    if args.validate_only:
        print(json.dumps({"at": iso_now(), **selection}, separators=(",", ":")))
        return

    reporter = Reporter(run_dir)
    reporter.emit(selection)
    reporter.status("normalizing", planId=plan["id"], latest=selection)
    normalization_file = dataset_root / "dataset.json"
    normalization = None
    if normalization_file.is_file():
        manifest = json.loads(normalization_file.read_text(encoding="utf-8"))
        if manifest.get("corpusFingerprint") == fingerprint:
            normalization = normalization_from_json(manifest["normalization"])
    training = plan["training"]
    if normalization is None:
        normalization = training_normalization(
            dataset, int(training["normalizationBatchSize"])
        )
        atomic_json({
            "version": 1,
            "createdAt": iso_now(),
            "contract": (
                "past-120-aligned-minute-returns-to-next-15-minute-returns-v1"
            ),
            "counts": counts,
            "corpusFingerprint": fingerprint,
            "normalization": normalization_json(normalization),
        }, normalization_file)

    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    model = build_model(normalization, plan["architecture"]).to(device)
    oracle = DifferentiableExposureValueOracle(oracle_config(plan)).to(device)
    parameters = parameter_count(model)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    probability_floor = float(plan["objective"]["probabilityFloor"])

    if args.evaluate_test:
        result_file = run_dir / "state" / "result.json"
        if not result_file.is_file():
            raise FileNotFoundError("complete result is required for test evaluation")
        result = json.loads(result_file.read_text(encoding="utf-8"))
        selected = load_torch_checkpoint(
            run_dir / "checkpoints" / "best.json",
            map_location=device, weights_only=False,
        )
        if selected.get("planSha256") != plan_sha \
                or selected.get("corpusFingerprint") != fingerprint \
                or selected.get("runnerContract") != RUNNER_CONTRACT:
            raise ValueError("oracle distribution checkpoint contract changed")
        model.load_state_dict(selected["model"])
        result["bestValidation"] = evaluate(
            model, oracle, dataset, "validation",
            batch_size=int(training["evaluationBatchSize"]),
            probability_floor=probability_floor,
            device=device, amp_dtype=amp_dtype,
            track_kl_percentiles=True,
            plan=plan,
        )
        result["validationZeroOracle"] = evaluate(
            model, oracle, dataset, "validation",
            batch_size=int(training["evaluationBatchSize"]),
            probability_floor=probability_floor,
            device=device, amp_dtype=amp_dtype,
            constant_prediction=torch.zeros(
                HORIZON_CANDLES, device=device, dtype=torch.float32
            ),
            track_kl_percentiles=True,
            plan=plan,
        )
        result["test"] = evaluate(
            model, oracle, dataset, "test",
            batch_size=int(training["evaluationBatchSize"]),
            probability_floor=probability_floor,
            device=device, amp_dtype=amp_dtype,
            track_kl_percentiles=True,
            plan=plan,
        )
        result["testZeroOracle"] = evaluate(
            model, oracle, dataset, "test",
            batch_size=int(training["evaluationBatchSize"]),
            probability_floor=probability_floor,
            device=device, amp_dtype=amp_dtype,
            constant_prediction=torch.zeros(
                HORIZON_CANDLES, device=device, dtype=torch.float32
            ),
            track_kl_percentiles=True,
            plan=plan,
        )
        result["testEvaluated"] = True
        atomic_json(result, result_file)
        event = {"event": "oracle-distribution-path-test-evaluated", **result}
        reporter.emit(event)
        reporter.status("complete", planId=plan["id"], latest=event)
        return

    validation_zero = evaluate(
        model, oracle, dataset, "validation",
        batch_size=int(training["evaluationBatchSize"]),
        probability_floor=probability_floor,
        device=device, amp_dtype=amp_dtype,
        constant_prediction=torch.zeros(
            HORIZON_CANDLES, device=device, dtype=torch.float32
        ),
        track_kl_percentiles=uses_p50_objective(plan),
        plan=plan,
    )
    optimizers = build_optimizers(model, training, device)
    schedule = training["learningRateSchedule"]
    schedulers = tuple(torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(schedule["factor"]),
        patience=int(schedule["patience"]),
        threshold=float(schedule["threshold"]),
        threshold_mode="abs",
        min_lr=float(schedule["minimumLearningRate"]),
    ) for optimizer in optimizers)
    start_epoch = 0
    global_step = 0
    best_score = math.inf
    best_epoch = -1
    stale_epochs = 0
    last_checkpoint = run_dir / "checkpoints" / "last.json"
    best_checkpoint = run_dir / "checkpoints" / "best.json"
    if checkpoint_exists(last_checkpoint):
        checkpoint = load_torch_checkpoint(
            last_checkpoint, map_location=device, weights_only=False
        )
        if checkpoint.get("planSha256") != plan_sha \
                or checkpoint.get("corpusFingerprint") != fingerprint \
                or checkpoint.get("runnerContract") != RUNNER_CONTRACT:
            raise ValueError("oracle distribution checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        for optimizer, state in zip(
            optimizers, checkpoint["optimizers"], strict=True
        ):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(
            schedulers, checkpoint["schedulers"], strict=True
        ):
            scheduler.load_state_dict(state)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["globalStep"])
        best_score = float(checkpoint["bestValidationScore"])
        best_epoch = int(checkpoint["bestEpoch"])
        stale_epochs = int(checkpoint["staleEpochs"])

    reporter.status(
        "training", planId=plan["id"], startEpoch=start_epoch,
        parameters=parameters, validationZeroOracle=validation_zero,
        testPolicy="sealed-validation-only-no-evaluation",
    )
    paused = False
    batch_size = int(training["batchSize"])
    evaluation_batch_size = int(training["evaluationBatchSize"])
    for epoch in range(start_epoch, int(training["epochs"])):
        started = time.monotonic()
        model.train()
        train_metrics = OracleDistributionMetricAccumulator(
            oracle.grid, probability_floor, uses_p50_objective(plan)
        )
        for features, targets, weights in iter_device_batches(
            dataset.iter_batches(
                "train", batch_size, shuffle=True,
                seed=seed + epoch, reuse_buffers=True,
            ), device,
        ):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = model(features.flatten(start_dim=1))
            predicted_probabilities = oracle_probabilities(oracle, prediction)
            with torch.no_grad():
                target_probabilities = oracle_probabilities(oracle, targets)
            if uses_p50_objective(plan):
                loss = oracle_mean_plus_p50_kl_loss(
                    predicted_probabilities,
                    target_probabilities,
                    weights,
                    probability_floor=probability_floor,
                    mean_weight=float(plan["objective"]["meanKlWeight"]),
                    p50_weight=float(plan["objective"]["p50KlWeight"]),
                )
            else:
                loss = oracle_forward_kl_loss(
                    predicted_probabilities,
                    target_probabilities,
                    weights,
                    probability_floor=probability_floor,
                )
            loss.backward()
            clip_grad_norm_(
                model.parameters(), float(training["gradientClip"]),
                foreach=device.type == "cuda",
            )
            for optimizer in optimizers:
                optimizer.step()
            train_metrics.add(
                prediction, targets, predicted_probabilities,
                target_probabilities, weights,
            )
            global_step += 1

        validation = evaluate(
            model, oracle, dataset, "validation",
            batch_size=evaluation_batch_size,
            probability_floor=probability_floor,
            device=device, amp_dtype=amp_dtype,
            track_kl_percentiles=uses_p50_objective(plan),
            plan=plan,
        )
        score = float(validation["selectionObjective"])
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        lr_before = float(optimizers[0].param_groups[0]["lr"])
        for scheduler in schedulers:
            scheduler.step(score)
        learning_rate = float(optimizers[0].param_groups[0]["lr"])
        checkpoint = {
            "model": model.state_dict(),
            "optimizers": [value.state_dict() for value in optimizers],
            "schedulers": [value.state_dict() for value in schedulers],
            "epoch": epoch,
            "globalStep": global_step,
            "bestValidationScore": best_score,
            "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs,
            "validation": validation,
            "validationZeroOracle": validation_zero,
            "parameterCount": parameters,
            "planSha256": plan_sha,
            "corpusFingerprint": fingerprint,
            "architectureContract": ARCHITECTURE_CONTRACT,
            "runnerContract": RUNNER_CONTRACT,
            "objectiveContract": plan["objective"]["contract"],
            "testEvaluated": False,
        }
        save_torch_checkpoint(checkpoint, last_checkpoint)
        if improved:
            save_torch_checkpoint(checkpoint, best_checkpoint)
        event = {
            "event": "oracle-distribution-path-epoch",
            "epoch": epoch,
            "epochs": int(training["epochs"]),
            "seconds": time.monotonic() - started,
            "globalStep": global_step,
            "train": train_metrics.result(),
            "validation": validation,
            "validationZeroOracle": validation_zero,
            "bestValidationScore": best_score,
            "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs,
            "improved": improved,
            "learningRate": learning_rate,
            "learningRateReduced": learning_rate < lr_before,
            "testEvaluated": False,
        }
        reporter.emit(event)
        reporter.status("training", planId=plan["id"], latest=event)
        if args.stop_after_epoch is not None \
                and epoch + 1 >= args.stop_after_epoch:
            paused = True
            break
        if stale_epochs >= int(training["earlyStoppingPatience"]):
            break

    if paused:
        reporter.status("paused", planId=plan["id"], latest={
            "event": "oracle-distribution-path-paused",
            "completedEpochs": epoch + 1,
            "bestEpoch": best_epoch,
            "testEvaluated": False,
        })
        return
    best = load_torch_checkpoint(
        best_checkpoint, map_location=device, weights_only=False
    )
    result = {
        "completedAt": iso_now(),
        "planId": plan["id"],
        "planSha256": plan_sha,
        "corpusFingerprint": fingerprint,
        "architectureContract": ARCHITECTURE_CONTRACT,
        "runnerContract": RUNNER_CONTRACT,
        "objectiveContract": plan["objective"]["contract"],
        "parameterCount": parameters,
        "resolution": RESOLUTION,
        "historyCandles": HISTORY_RETURN_COUNT,
        "horizonCandles": HORIZON_CANDLES,
        "oracle": plan["oracle"],
        "bestEpoch": int(best["bestEpoch"]),
        "bestValidationScore": float(best["bestValidationScore"]),
        "bestValidation": best["validation"],
        "validationZeroOracle": validation_zero,
        "checkpoint": str(best_checkpoint.relative_to(repo_root)),
        "testEvaluated": False,
    }
    atomic_json(result, run_dir / "state" / "result.json")
    reporter.emit({"event": "oracle-distribution-path-complete", **result})
    reporter.status("complete", planId=plan["id"], latest=result)


if __name__ == "__main__":
    main()
