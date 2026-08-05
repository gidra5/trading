from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from multiscale_next_return import (
    ARCHITECTURE_CONTRACT,
    MAX_WINDOW_LABELS,
    MODEL_VARIANTS,
    JointInputGlu,
    MultiscaleNormalization,
    SeparateComponentGlu,
    component_labels,
    parameter_count,
    window_path_slug,
)
from multiscale_next_return_dataset import (
    MultiscaleNextReturnDataset,
    trim_shards_for_moving_average_history,
)
from next_return_dataset import (
    HISTORY_RETURN_COUNT,
    ExampleShard,
    count_examples,
    example_span_ms,
    select_example_shards,
    validate_horizon_return_count,
)
from next_return_sequence import (
    SUMMARY_NAMES,
    SequenceMetricAccumulator,
    SequenceNormalization,
    numpy_path_summaries,
)
from train_normalized_glu_next_return import (
    Reporter,
    atomic_json,
    build_optimizers,
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


RUNNER_CONTRACT = "telescoping-multiscale-return-glu-matrix-v2"
SELECTION_CONTRACT = "summed-raw-return-validation-objective-v1"
OBJECTIVE_CONTRACT = (
    "normalized-per-candle-mse-plus-cumulative-return-mse-only-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one telescoping moving-average GLU experiment."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--max-window", required=True, choices=MAX_WINDOW_LABELS)
    parser.add_argument("--variant", required=True, choices=MODEL_VARIANTS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def validate_plan(plan: dict) -> None:
    required = (
        "id", "sourceDatasetDir", "historyDir", "datasetDirTemplate",
        "runDirTemplate", "horizonSeconds", "testExamples", "architecture",
        "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("multiscale plan is missing required fields")
    if validate_horizon_return_count(int(plan["horizonSeconds"])) < 2:
        raise ValueError("multiscale experiment requires a path horizon")
    architecture = plan["architecture"]
    if architecture.get("contract") != ARCHITECTURE_CONTRACT \
            or architecture.get("inputNormalization") != "training-position" \
            or not isinstance(architecture.get("widths"), list) \
            or not architecture["widths"] \
            or any(int(width) < 2 for width in architecture["widths"]):
        raise ValueError("multiscale architecture is invalid")
    training = plan["training"]
    for name in (
        "epochs", "jointBatchSize", "separateBatchSizePerComponent",
        "evaluationBatchSize", "normalizationBatchSize",
        "earlyStoppingPatience", "seed",
    ):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"multiscale training {name} must be positive")
    if training.get("device") not in {"cpu", "cuda"} \
            or training.get("mixedPrecision") != "bfloat16" \
            or float(training.get("learningRate", 0)) <= 0 \
            or float(training.get("gradientClip", 0)) <= 0:
        raise ValueError("multiscale runtime settings are invalid")
    schedule = training.get("learningRateSchedule", {})
    if schedule.get("type") != "reduce-on-validation-plateau" \
            or not 0 < float(schedule.get("factor", 0)) < 1 \
            or int(schedule.get("patience", -1)) < 0 \
            or float(schedule.get("minimumLearningRate", 0)) <= 0:
        raise ValueError("multiscale learning-rate schedule is invalid")


def derived_value(template: str, max_window: str, variant: str) -> str:
    return template.format(
        maxWindow=window_path_slug(max_window), variant=variant
    )


def trim_fingerprint(
    shards: dict[str, list[ExampleShard]],
    *,
    max_window: str,
    horizon: int,
) -> str:
    value = {
        "contract": "causal-telescoping-return-moving-average-components-v2",
        "maxWindow": max_window,
        "horizon": horizon,
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
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mean_std(
    values_sum: np.ndarray,
    values_square_sum: np.ndarray,
    total: float,
    *,
    minimum_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = values_sum / total
    variance = values_square_sum / total - np.square(mean)
    return (
        mean.astype(np.float32),
        np.sqrt(np.maximum(variance, minimum_variance)).astype(np.float32),
    )


def training_normalization(
    dataset: MultiscaleNextReturnDataset,
    *,
    batch_size: int,
) -> MultiscaleNormalization:
    components = dataset.component_count
    horizon = dataset.horizon_return_count
    feature_sum = np.zeros((components, HISTORY_RETURN_COUNT), np.float64)
    feature_square_sum = np.zeros_like(feature_sum)
    component_target_sum = np.zeros((components, horizon), np.float64)
    component_target_square_sum = np.zeros_like(component_target_sum)
    component_cumulative_sum = np.zeros(components, np.float64)
    component_cumulative_square_sum = np.zeros(components, np.float64)
    raw_target_sum = np.zeros(horizon, np.float64)
    raw_target_square_sum = np.zeros(horizon, np.float64)
    raw_summary_sum = np.zeros(len(SUMMARY_NAMES), np.float64)
    raw_summary_square_sum = np.zeros(len(SUMMARY_NAMES), np.float64)
    total = 0.0
    for feature_tensor, target_tensor, weight_tensor in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0
    ):
        features = feature_tensor.numpy().astype(np.float64, copy=False)
        targets = target_tensor.numpy().astype(np.float64, copy=False)
        weights = weight_tensor.numpy().astype(np.float64, copy=False)
        raw_targets = targets.sum(axis=1)
        raw_summaries = numpy_path_summaries(raw_targets)
        component_cumulative = np.expm1(targets.sum(axis=2))
        feature_sum += np.einsum("i,icj->cj", weights, features)
        feature_square_sum += np.einsum(
            "i,icj->cj", weights, np.square(features)
        )
        component_target_sum += np.einsum("i,icj->cj", weights, targets)
        component_target_square_sum += np.einsum(
            "i,icj->cj", weights, np.square(targets)
        )
        component_cumulative_sum += np.einsum(
            "i,ic->c", weights, component_cumulative
        )
        component_cumulative_square_sum += np.einsum(
            "i,ic->c", weights, np.square(component_cumulative)
        )
        raw_target_sum += np.einsum("i,ij->j", weights, raw_targets)
        raw_target_square_sum += np.einsum(
            "i,ij->j", weights, np.square(raw_targets)
        )
        raw_summary_sum += np.einsum("i,ij->j", weights, raw_summaries)
        raw_summary_square_sum += np.einsum(
            "i,ij->j", weights, np.square(raw_summaries)
        )
        total += float(weights.sum(dtype=np.float64))
    if int(round(total)) != dataset.logical_count("train"):
        raise RuntimeError("multiscale normalization missed training examples")
    feature_mean, feature_std = mean_std(
        feature_sum, feature_square_sum, total, minimum_variance=1e-24
    )
    component_target_mean, component_target_std = mean_std(
        component_target_sum,
        component_target_square_sum,
        total,
        minimum_variance=1e-24,
    )
    component_cumulative_mean, component_cumulative_std = mean_std(
        component_cumulative_sum,
        component_cumulative_square_sum,
        total,
        minimum_variance=1e-30,
    )
    raw_target_mean, raw_target_std = mean_std(
        raw_target_sum, raw_target_square_sum, total, minimum_variance=1e-24
    )
    raw_summary_mean, raw_summary_std = mean_std(
        raw_summary_sum,
        raw_summary_square_sum,
        total,
        minimum_variance=1e-30,
    )
    result = MultiscaleNormalization(
        feature_mean,
        feature_std,
        component_target_mean,
        component_target_std,
        component_cumulative_mean,
        component_cumulative_std,
        raw_target_mean,
        raw_target_std,
        raw_summary_mean,
        raw_summary_std,
    )
    result.validate()
    return result


def normalization_json(value: MultiscaleNormalization) -> dict:
    return {
        name: getattr(value, name).tolist()
        for name in value.__dataclass_fields__
    }


def normalization_from_json(value: dict) -> MultiscaleNormalization:
    result = MultiscaleNormalization(**{
        name: np.asarray(value[name], dtype=np.float32)
        for name in MultiscaleNormalization.__dataclass_fields__
    })
    result.validate()
    return result


def raw_sequence_normalization(
    value: MultiscaleNormalization,
) -> SequenceNormalization:
    result = SequenceNormalization(
        feature_mean=np.zeros(HISTORY_RETURN_COUNT, np.float32),
        feature_std=np.ones(HISTORY_RETURN_COUNT, np.float32),
        target_mean=value.raw_target_mean,
        target_std=value.raw_target_std,
        summary_mean=value.raw_summary_mean,
        summary_std=value.raw_summary_std,
    )
    result.validate()
    return result


def build_model(
    variant: str,
    normalization: MultiscaleNormalization,
    architecture: dict,
) -> SeparateComponentGlu | JointInputGlu:
    options = {
        "widths": tuple(int(value) for value in architecture["widths"]),
        "dropout": float(architecture["dropout"]),
        "dropout_rate": float(architecture["dropoutRate"]),
        "initial_radius": float(architecture["initialRadius"]),
        "minimum_radius": float(architecture["minimumRadius"]),
    }
    if variant == "separate-components":
        return SeparateComponentGlu(normalization, **options)
    if variant == "joint-input":
        return JointInputGlu(normalization, **options)
    raise ValueError(f"unsupported multiscale variant: {variant}")


def component_objective_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    target_std: Tensor,
    cumulative_std: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("component predictions are misaligned")
    candle = ((prediction.float() - target.float()) / target_std).square().mean(
        dim=(1, 2)
    )
    prediction_cumulative = torch.expm1(prediction.float().sum(dim=2))
    target_cumulative = torch.expm1(target.float().sum(dim=2))
    cumulative = (
        (prediction_cumulative - target_cumulative) / cumulative_std
    ).square().mean(dim=1)
    per_example = candle + cumulative
    return (per_example * weights.float()).sum() / weights.sum()


def raw_objective_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    target_std: Tensor,
    cumulative_std: Tensor,
) -> Tensor:
    candle = ((prediction.float() - target.float()) / target_std).square().mean(
        dim=1
    )
    cumulative = (
        (
            torch.expm1(prediction.float().sum(dim=1))
            - torch.expm1(target.float().sum(dim=1))
        ) / cumulative_std
    ).square()
    return ((candle + cumulative) * weights.float()).sum() / weights.sum()


@torch.no_grad()
def evaluate(
    model: SeparateComponentGlu | JointInputGlu,
    dataset: MultiscaleNextReturnDataset,
    split: str,
    *,
    batch_size: int,
    normalization: MultiscaleNormalization,
    device: torch.device,
    amp_dtype: torch.dtype,
    track_per_lead: bool = True,
) -> dict:
    model.eval()
    metrics = SequenceMetricAccumulator(
        raw_sequence_normalization(normalization),
        candle_weight=1,
        summary_weight=1,
        summary_metric_weights=(0, 0, 0, 0, 1),
        device=device,
        track_per_lead=track_per_lead,
    )
    for features, component_targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        raw_targets = component_targets.sum(dim=1)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            prediction = model(features)
        metrics.add(prediction, raw_targets, weights)
    return metrics.result(include_per_lead=track_per_lead)


def main() -> None:
    args = parse_args()
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    horizon = int(plan["horizonSeconds"])
    combo_id = f"{plan['id']}-{args.max_window}-{args.variant}"
    combo_plan = {
        **plan,
        "comboId": combo_id,
        "maxWindow": args.max_window,
        "variant": args.variant,
    }
    plan_fingerprint = canonical_fingerprint(combo_plan)
    layout = training_storage_layout(repo_root)
    source_root = require_under(
        (repo_root / plan["sourceDatasetDir"]).resolve(),
        layout.datasets,
        "sourceDatasetDir",
    )
    history_root = require_under(
        (repo_root / plan["historyDir"]).resolve(),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    dataset_root = require_under(
        (repo_root / derived_value(
            plan["datasetDirTemplate"], args.max_window, args.variant
        )).resolve(),
        layout.datasets,
        "datasetDirTemplate",
    )
    run_dir = require_under(
        (repo_root / derived_value(
            plan["runDirTemplate"], args.max_window, args.variant
        )).resolve(),
        layout.runs,
        "runDirTemplate",
    )
    source_manifest = json.loads(
        (source_root / "dataset.json").read_text(encoding="utf-8")
    )
    shards = select_example_shards(
        source_manifest,
        test_count=int(plan["testExamples"]),
        test_tail_offset=int(plan.get("testTailOffsetExamples", 0)),
        horizon_return_count=horizon,
        cross_split_purge_ms=example_span_ms(horizon),
    )
    first_history_day = min(file.stem for file in history_root.glob("*.json"))
    shards = trim_shards_for_moving_average_history(
        shards,
        first_history_day=first_history_day,
        max_window=args.max_window,
    )
    counts = count_examples(shards)
    fingerprint = trim_fingerprint(
        shards, max_window=args.max_window, horizon=horizon
    )
    components = len(component_labels(args.max_window))
    dummy_normalization = MultiscaleNormalization(
        np.zeros((components, HISTORY_RETURN_COUNT), np.float32),
        np.ones((components, HISTORY_RETURN_COUNT), np.float32),
        np.zeros((components, horizon), np.float32),
        np.ones((components, horizon), np.float32),
        np.zeros(components, np.float32),
        np.ones(components, np.float32),
        np.zeros(horizon, np.float32),
        np.ones(horizon, np.float32),
        np.zeros(len(SUMMARY_NAMES), np.float32),
        np.ones(len(SUMMARY_NAMES), np.float32),
    )
    dummy_model = build_model(
        args.variant, dummy_normalization, plan["architecture"]
    )
    model_parameters = parameter_count(dummy_model)
    del dummy_model
    selection = {
        "event": "multiscale-dataset-selected",
        "planId": combo_id,
        "maxWindow": args.max_window,
        "variant": args.variant,
        "componentLabels": component_labels(args.max_window),
        "counts": counts,
        "corpusFingerprint": fingerprint,
        "parameters": model_parameters,
        "testPolicy": "sealed-validation-only-no-evaluation",
    }
    if args.validate_only:
        print(json.dumps({"at": iso_now(), **selection}, separators=(",", ":")))
        return

    reporter = Reporter(run_dir)
    reporter.emit(selection)
    reporter.status("normalizing", planId=combo_id, latest=selection)
    dataset = MultiscaleNextReturnDataset(
        shards,
        history_root,
        max_window=args.max_window,
        horizon_return_count=horizon,
    )
    normalization_file = dataset_root / "dataset.json"
    normalization = None
    if normalization_file.is_file():
        manifest = json.loads(normalization_file.read_text(encoding="utf-8"))
        if manifest.get("corpusFingerprint") == fingerprint \
                and manifest.get("maxWindow") == args.max_window \
                and int(manifest.get("horizonSeconds", 0)) == horizon:
            normalization = normalization_from_json(manifest["normalization"])
    if normalization is None:
        normalization = training_normalization(
            dataset,
            batch_size=int(plan["training"]["normalizationBatchSize"]),
        )
        manifest = {
            "version": 1,
            "createdAt": iso_now(),
            "contract": "telescoping-return-moving-average-components-v2",
            "maxWindow": args.max_window,
            "componentLabels": component_labels(args.max_window),
            "horizonSeconds": horizon,
            "corpusFingerprint": fingerprint,
            "counts": counts,
            "normalization": normalization_json(normalization),
        }
        atomic_json(manifest, normalization_file)

    training = plan["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    model = build_model(
        args.variant, normalization, plan["architecture"]
    ).to(device)
    optimizers = build_optimizers(model, training, device)  # type: ignore[arg-type]
    schedule = training["learningRateSchedule"]
    schedulers = tuple(
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(schedule["factor"]),
            patience=int(schedule["patience"]),
            threshold=float(schedule["threshold"]),
            threshold_mode="abs",
            min_lr=float(schedule["minimumLearningRate"]),
        )
        for optimizer in optimizers
    )
    start_epoch = 0
    global_step = 0
    best_validation = math.inf
    best_epoch = -1
    stale_epochs = 0
    last_checkpoint = run_dir / "checkpoints" / "last.json"
    best_checkpoint = run_dir / "checkpoints" / "best.json"
    if checkpoint_exists(last_checkpoint):
        checkpoint = load_torch_checkpoint(
            last_checkpoint, map_location=device, weights_only=False
        )
        for name, expected in (
            ("planSha256", plan_fingerprint),
            ("corpusFingerprint", fingerprint),
            ("runnerContract", RUNNER_CONTRACT),
        ):
            if checkpoint.get(name) != expected:
                raise ValueError(f"multiscale checkpoint {name} changed")
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
        best_validation = float(checkpoint["bestValidationScore"])
        best_epoch = int(checkpoint["bestEpoch"])
        stale_epochs = int(checkpoint["staleEpochs"])
    batch_size = int(
        training["jointBatchSize"]
        if args.variant == "joint-input"
        else max(
            1,
            int(training["separateBatchSizePerComponent"]) // components,
        )
    )
    evaluation_batch_size = min(
        int(training["evaluationBatchSize"]), batch_size
    )
    maximum_epochs = int(training["epochs"])
    patience = int(training["earlyStoppingPatience"])
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    component_target_std = torch.from_numpy(
        normalization.component_target_std
    ).to(device)
    component_cumulative_std = torch.from_numpy(
        normalization.component_cumulative_std
    ).to(device)
    raw_target_std = torch.from_numpy(normalization.raw_target_std).to(device)
    raw_cumulative_std = torch.as_tensor(
        normalization.raw_summary_std[SUMMARY_NAMES.index("cumulativeReturn")],
        device=device,
    )
    reporter.status(
        "training",
        planId=combo_id,
        startEpoch=start_epoch,
        batchSize=batch_size,
        parameters=model_parameters,
        bestEpoch=best_epoch,
        testPolicy="sealed-validation-only-no-evaluation",
    )
    paused = False
    for epoch in range(start_epoch, maximum_epochs):
        started = time.monotonic()
        model.train()
        train_metrics = SequenceMetricAccumulator(
            raw_sequence_normalization(normalization),
            candle_weight=1,
            summary_weight=1,
            summary_metric_weights=(0, 0, 0, 0, 1),
            device=device,
            track_per_lead=False,
        )
        for features, component_targets, weights in iter_device_batches(
            dataset.iter_batches(
                "train",
                batch_size,
                shuffle=True,
                seed=seed + epoch,
                reuse_buffers=True,
            ),
            device,
        ):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            raw_targets = component_targets.sum(dim=1)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                if args.variant == "separate-components":
                    component_prediction = model.forward_components(features)  # type: ignore[union-attr]
                    prediction = component_prediction.sum(dim=1)
                    loss = component_objective_loss(
                        component_prediction,
                        component_targets,
                        weights,
                        target_std=component_target_std,
                        cumulative_std=component_cumulative_std,
                    )
                else:
                    prediction = model(features)
                    loss = raw_objective_loss(
                        prediction,
                        raw_targets,
                        weights,
                        target_std=raw_target_std,
                        cumulative_std=raw_cumulative_std,
                    )
            loss.backward()
            clip_grad_norm_(
                model.parameters(),
                float(training["gradientClip"]),
                foreach=device.type == "cuda",
            )
            for optimizer in optimizers:
                optimizer.step()
            train_metrics.add(prediction, raw_targets, weights)
            global_step += 1
        validation = evaluate(
            model,
            dataset,
            "validation",
            batch_size=evaluation_batch_size,
            normalization=normalization,
            device=device,
            amp_dtype=amp_dtype,
        )
        validation_score = float(validation["objective"])
        improved = validation_score < best_validation
        if improved:
            best_validation = validation_score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        learning_rate_before = float(optimizers[0].param_groups[0]["lr"])
        for scheduler in schedulers:
            scheduler.step(validation_score)
        learning_rate = float(optimizers[0].param_groups[0]["lr"])
        checkpoint = {
            "model": model.state_dict(),
            "optimizers": [value.state_dict() for value in optimizers],
            "schedulers": [value.state_dict() for value in schedulers],
            "epoch": epoch,
            "globalStep": global_step,
            "bestValidationScore": best_validation,
            "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs,
            "validation": validation,
            "parameterCount": model_parameters,
            "planSha256": plan_fingerprint,
            "corpusFingerprint": fingerprint,
            "architectureContract": ARCHITECTURE_CONTRACT,
            "runnerContract": RUNNER_CONTRACT,
            "selectionContract": SELECTION_CONTRACT,
            "objectiveContract": OBJECTIVE_CONTRACT,
            "testEvaluated": False,
        }
        save_torch_checkpoint(checkpoint, last_checkpoint)
        if improved:
            save_torch_checkpoint(checkpoint, best_checkpoint)
        event = {
            "event": "multiscale-epoch",
            "epoch": epoch,
            "epochs": maximum_epochs,
            "seconds": time.monotonic() - started,
            "globalStep": global_step,
            "train": train_metrics.result(include_per_lead=False),
            "validation": validation,
            "bestValidationScore": best_validation,
            "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs,
            "improved": improved,
            "learningRate": learning_rate,
            "learningRateReduced": learning_rate < learning_rate_before,
            "testEvaluated": False,
        }
        reporter.emit(event)
        reporter.status("training", planId=combo_id, latest=event)
        if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            paused = True
            break
        if stale_epochs >= patience:
            break
    if paused:
        reporter.status(
            "paused",
            planId=combo_id,
            latest={
                "event": "multiscale-paused",
                "completedEpochs": epoch + 1,
                "bestEpoch": best_epoch,
                "testEvaluated": False,
            },
        )
        return
    best = load_torch_checkpoint(
        best_checkpoint, map_location=device, weights_only=False
    )
    result = {
        "completedAt": iso_now(),
        "planId": combo_id,
        "planSha256": plan_fingerprint,
        "corpusFingerprint": fingerprint,
        "architectureContract": ARCHITECTURE_CONTRACT,
        "runnerContract": RUNNER_CONTRACT,
        "objectiveContract": OBJECTIVE_CONTRACT,
        "maxWindow": args.max_window,
        "variant": args.variant,
        "componentLabels": component_labels(args.max_window),
        "parameterCount": model_parameters,
        "batchSize": batch_size,
        "horizonSeconds": horizon,
        "bestEpoch": int(best["bestEpoch"]),
        "bestValidationScore": float(best["bestValidationScore"]),
        "bestValidation": best["validation"],
        "checkpoint": str(best_checkpoint.relative_to(repo_root)),
        "testEvaluated": False,
    }
    atomic_json(result, run_dir / "state" / "result.json")
    reporter.emit({"event": "multiscale-complete", **result})
    reporter.status("complete", planId=combo_id, latest=result)


if __name__ == "__main__":
    main()
