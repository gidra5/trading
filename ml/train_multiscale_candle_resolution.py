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

from multiscale_candle_resolution import (
    DailyCandleResolutionDataset,
    HORIZON_CANDLE_COUNT,
    RESOLUTION_SECONDS,
    MultiscaleCandleResolutionDataset,
    allowed_max_windows,
    resolution_component_labels,
    trim_shards_for_resolution_history,
)
from multiscale_next_return import (
    ARCHITECTURE_CONTRACT,
    JointInputGlu,
    MultiscaleNormalization,
    parameter_count,
    window_path_slug,
)
from next_return_dataset import (
    HISTORY_RETURN_COUNT,
    ExampleShard,
    select_example_shards,
)
from next_return_sequence import (
    SUMMARY_NAMES,
    SequenceMetricAccumulator,
    SequenceNormalization,
    numpy_path_summaries,
    sequence_objective_loss,
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


RUNNER_CONTRACT = "aligned-candle-resolution-joint-multiscale-glu-v1"
OBJECTIVE_CONTRACT = (
    "aligned-five-candle-normalized-mse-plus-cumulative-return-mse-v1"
)


@dataclass(frozen=True)
class ResolutionNormalization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    summary_mean: np.ndarray
    summary_std: np.ndarray

    def validate(self) -> None:
        components = self.feature_mean.shape[0]
        if self.feature_mean.shape != (components, HISTORY_RETURN_COUNT) \
                or self.feature_std.shape != self.feature_mean.shape \
                or self.target_mean.shape != (HORIZON_CANDLE_COUNT,) \
                or self.target_std.shape != self.target_mean.shape \
                or self.summary_mean.shape != (len(SUMMARY_NAMES),) \
                or self.summary_std.shape != self.summary_mean.shape:
            raise ValueError("resolution normalization shapes are invalid")
        for values in self.__dict__.values():
            if not np.isfinite(values).all():
                raise ValueError("resolution normalization must be finite")
        if bool((self.feature_std <= 0).any()) \
                or bool((self.target_std <= 0).any()) \
                or bool((self.summary_std <= 0).any()):
            raise ValueError("resolution normalization scales must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train joint MA components at an aligned candle resolution."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--resolution", required=True, choices=RESOLUTION_SECONDS)
    parser.add_argument("--max-window", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate an already-complete selected run on its sealed test split.",
    )
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def validate_plan(plan: dict) -> None:
    required = (
        "id", "sourceDatasetDir", "historyDir", "datasetDirTemplate",
        "runDirTemplate", "comparisonMaxWindow",
        "architecture", "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("resolution plan is missing required fields")
    architecture = plan["architecture"]
    if architecture.get("contract") != ARCHITECTURE_CONTRACT \
            or architecture.get("inputNormalization") != "training-position" \
            or not isinstance(architecture.get("widths"), list) \
            or not architecture["widths"]:
        raise ValueError("resolution architecture is invalid")
    if any(value != "1d" for value in plan.get("resolutions", ())) \
            and int(plan.get("testExamples", 0)) < 1:
        raise ValueError("sub-daily resolution plan needs testExamples")
    training = plan["training"]
    for name in ("epochs", "earlyStoppingPatience", "seed"):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"resolution training {name} must be positive")
    for resolution in plan.get("resolutions", RESOLUTION_SECONDS):
        settings = training.get("byResolution", {}).get(resolution, {})
        if any(int(settings.get(name, 0)) < 1 for name in (
            "batchSize", "evaluationBatchSize", "normalizationBatchSize"
        )):
            raise ValueError(f"missing runtime settings for {resolution}")


def derived_path(template: str, resolution: str, max_window: str) -> str:
    return template.format(
        resolution=resolution,
        maxWindow=window_path_slug(max_window),
    )


def corpus_fingerprint(
    shards: dict[str, list[ExampleShard]],
    *,
    resolution: str,
    max_window: str,
) -> str:
    payload = {
        "contract": "aligned-resolution-telescoping-candle-components-v1",
        "resolution": resolution,
        "maxWindow": max_window,
        "horizonCandles": HORIZON_CANDLE_COUNT,
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


def mean_std(
    values_sum: np.ndarray,
    square_sum: np.ndarray,
    total: float,
    minimum_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = values_sum / total
    variance = square_sum / total - np.square(mean)
    return (
        mean.astype(np.float32),
        np.sqrt(np.maximum(variance, minimum_variance)).astype(np.float32),
    )


def training_normalization(
    dataset: MultiscaleCandleResolutionDataset,
    batch_size: int,
) -> ResolutionNormalization:
    components = dataset.component_count
    feature_sum = np.zeros((components, HISTORY_RETURN_COUNT), np.float64)
    feature_square = np.zeros_like(feature_sum)
    target_sum = np.zeros(HORIZON_CANDLE_COUNT, np.float64)
    target_square = np.zeros_like(target_sum)
    summary_sum = np.zeros(len(SUMMARY_NAMES), np.float64)
    summary_square = np.zeros_like(summary_sum)
    total = 0.0
    for feature_tensor, target_tensor, weight_tensor in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0
    ):
        features = feature_tensor.numpy().astype(np.float64, copy=False)
        targets = target_tensor.numpy().astype(np.float64, copy=False)
        weights = weight_tensor.numpy().astype(np.float64, copy=False)
        summaries = numpy_path_summaries(targets)
        feature_sum += np.einsum("i,icj->cj", weights, features)
        feature_square += np.einsum(
            "i,icj->cj", weights, np.square(features)
        )
        target_sum += np.einsum("i,ij->j", weights, targets)
        target_square += np.einsum(
            "i,ij->j", weights, np.square(targets)
        )
        summary_sum += np.einsum("i,ij->j", weights, summaries)
        summary_square += np.einsum(
            "i,ij->j", weights, np.square(summaries)
        )
        total += float(weights.sum(dtype=np.float64))
    if int(round(total)) != dataset.logical_count("train"):
        raise RuntimeError("resolution normalization missed examples")
    feature_mean, feature_std = mean_std(
        feature_sum, feature_square, total, 1e-24
    )
    target_mean, target_std = mean_std(
        target_sum, target_square, total, 1e-24
    )
    summary_mean, summary_std = mean_std(
        summary_sum, summary_square, total, 1e-30
    )
    result = ResolutionNormalization(
        feature_mean, feature_std, target_mean, target_std,
        summary_mean, summary_std,
    )
    result.validate()
    return result


def normalization_json(value: ResolutionNormalization) -> dict:
    return {name: getattr(value, name).tolist() for name in value.__dataclass_fields__}


def normalization_from_json(value: dict) -> ResolutionNormalization:
    result = ResolutionNormalization(**{
        name: np.asarray(value[name], dtype=np.float32)
        for name in ResolutionNormalization.__dataclass_fields__
    })
    result.validate()
    return result


def model_normalization(value: ResolutionNormalization) -> MultiscaleNormalization:
    components = value.feature_mean.shape[0]
    return MultiscaleNormalization(
        feature_mean=value.feature_mean,
        feature_std=value.feature_std,
        component_target_mean=np.zeros(
            (components, HORIZON_CANDLE_COUNT), np.float32
        ),
        component_target_std=np.ones(
            (components, HORIZON_CANDLE_COUNT), np.float32
        ),
        component_cumulative_mean=np.zeros(components, np.float32),
        component_cumulative_std=np.ones(components, np.float32),
        raw_target_mean=value.target_mean,
        raw_target_std=value.target_std,
        raw_summary_mean=value.summary_mean,
        raw_summary_std=value.summary_std,
    )


def sequence_normalization(value: ResolutionNormalization) -> SequenceNormalization:
    return SequenceNormalization(
        feature_mean=np.zeros(HISTORY_RETURN_COUNT, np.float32),
        feature_std=np.ones(HISTORY_RETURN_COUNT, np.float32),
        target_mean=value.target_mean,
        target_std=value.target_std,
        summary_mean=value.summary_mean,
        summary_std=value.summary_std,
    )


def build_model(
    normalization: ResolutionNormalization,
    architecture: dict,
) -> JointInputGlu:
    return JointInputGlu(
        model_normalization(normalization),
        widths=tuple(int(value) for value in architecture["widths"]),
        dropout=float(architecture["dropout"]),
        dropout_rate=float(architecture["dropoutRate"]),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
    )


@torch.no_grad()
def evaluate(
    model: JointInputGlu,
    dataset: MultiscaleCandleResolutionDataset,
    split: str,
    *,
    batch_size: int,
    normalization: ResolutionNormalization,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict:
    model.eval()
    metrics = SequenceMetricAccumulator(
        sequence_normalization(normalization),
        candle_weight=1,
        summary_weight=1,
        summary_metric_weights=(0, 0, 0, 0, 1),
        device=device,
    )
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ), device,
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            prediction = model(features)
        metrics.add(prediction, targets, weights)
    return metrics.result()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    if args.max_window not in allowed_max_windows(args.resolution):
        raise ValueError("maximum window is invalid for resolution")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    combo_id = f"{plan['id']}-{args.resolution}-{args.max_window}"
    combo_plan = {
        **plan, "comboId": combo_id,
        "resolution": args.resolution, "maxWindow": args.max_window,
    }
    plan_sha = canonical_fingerprint(combo_plan)
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
        (repo_root / derived_path(
            plan["datasetDirTemplate"], args.resolution, args.max_window
        )).resolve(), layout.datasets, "datasetDirTemplate",
    )
    run_dir = require_under(
        (repo_root / derived_path(
            plan["runDirTemplate"], args.resolution, args.max_window
        )).resolve(), layout.runs, "runDirTemplate",
    )
    source_manifest = json.loads(
        (source_root / "dataset.json").read_text(encoding="utf-8")
    )
    first_day = min(file.stem for file in history_root.glob("*.json"))
    last_day = max(file.stem for file in history_root.glob("*.json"))
    comparison_max_window = str(plan["comparisonMaxWindow"])
    if comparison_max_window not in allowed_max_windows(args.resolution):
        raise ValueError("comparison maximum window is invalid for resolution")
    if args.resolution == "1d":
        daily_history_root = require_under(
            (repo_root / plan["dailyHistoryDir"]).resolve(),
            repo_root / "data" / "market" / "immutable" / "refs" / "candles",
            "dailyHistoryDir",
        )
        dataset = DailyCandleResolutionDataset(
            daily_history_root,
            first_day=first_day,
            last_day=last_day,
            max_window=args.max_window,
            comparison_max_window=comparison_max_window,
        )
        shards = dataset.shards
    else:
        resolution_seconds = RESOLUTION_SECONDS[args.resolution]
        example_span_ms = (
            HISTORY_RETURN_COUNT + HORIZON_CANDLE_COUNT
        ) * resolution_seconds * 1_000
        shards = select_example_shards(
            source_manifest,
            test_count=int(plan["testExamples"]),
            test_tail_offset=int(plan.get("testTailOffsetExamples", 0)),
            horizon_return_count=HORIZON_CANDLE_COUNT * resolution_seconds,
            cross_split_purge_ms=example_span_ms,
        )
        shards = trim_shards_for_resolution_history(
            shards,
            first_history_day=first_day,
            resolution=args.resolution,
            max_window=comparison_max_window,
        )
        dataset = MultiscaleCandleResolutionDataset(
            shards, history_root,
            resolution=args.resolution, max_window=args.max_window,
        )
    counts = {
        split: dataset.logical_count(split)
        for split in ("train", "validation", "test")
    }
    fingerprint = corpus_fingerprint(
        shards, resolution=args.resolution, max_window=args.max_window
    )
    components = dataset.component_count
    dummy = ResolutionNormalization(
        np.zeros((components, HISTORY_RETURN_COUNT), np.float32),
        np.ones((components, HISTORY_RETURN_COUNT), np.float32),
        np.zeros(HORIZON_CANDLE_COUNT, np.float32),
        np.ones(HORIZON_CANDLE_COUNT, np.float32),
        np.zeros(len(SUMMARY_NAMES), np.float32),
        np.ones(len(SUMMARY_NAMES), np.float32),
    )
    parameters = parameter_count(build_model(dummy, plan["architecture"]))
    selection = {
        "event": "resolution-dataset-selected",
        "planId": combo_id,
        "resolution": args.resolution,
        "maxWindow": args.max_window,
        "comparisonMaxWindow": comparison_max_window,
        "componentLabels": dataset.component_labels,
        "counts": counts,
        "corpusFingerprint": fingerprint,
        "parameters": parameters,
        "testPolicy": "sealed-validation-only-no-evaluation",
    }
    if args.validate_only:
        print(json.dumps({"at": iso_now(), **selection}, separators=(",", ":")))
        return
    reporter = Reporter(run_dir)
    reporter.emit(selection)
    reporter.status("normalizing", planId=combo_id, latest=selection)
    normalization_file = dataset_root / "dataset.json"
    normalization = None
    if normalization_file.is_file():
        manifest = json.loads(normalization_file.read_text(encoding="utf-8"))
        if manifest.get("corpusFingerprint") == fingerprint:
            normalization = normalization_from_json(manifest["normalization"])
    settings = plan["training"]["byResolution"][args.resolution]
    if normalization is None:
        normalization = training_normalization(
            dataset, int(settings["normalizationBatchSize"])
        )
        atomic_json({
            "version": 1,
            "createdAt": iso_now(),
            "contract": "aligned-resolution-telescoping-candle-components-v1",
            "resolution": args.resolution,
            "maxWindow": args.max_window,
            "componentLabels": dataset.component_labels,
            "counts": counts,
            "corpusFingerprint": fingerprint,
            "normalization": normalization_json(normalization),
        }, normalization_file)
    training = plan["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    seed = int(training["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    model = build_model(normalization, plan["architecture"]).to(device)
    if args.evaluate_test:
        result_file = run_dir / "state" / "result.json"
        if not result_file.is_file():
            raise FileNotFoundError("complete result is required for test evaluation")
        result = json.loads(result_file.read_text(encoding="utf-8"))
        selected = load_torch_checkpoint(
            run_dir / "checkpoints" / "best.json",
            map_location=device,
            weights_only=False,
        )
        if selected.get("planSha256") != plan_sha \
                or selected.get("corpusFingerprint") != fingerprint \
                or selected.get("runnerContract") != RUNNER_CONTRACT:
            raise ValueError("selected checkpoint contract changed")
        model.load_state_dict(selected["model"])
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        test_metrics = evaluate(
            model,
            dataset,
            "test",
            batch_size=int(settings["evaluationBatchSize"]),
            normalization=normalization,
            device=device,
            amp_dtype=amp_dtype,
        )
        result["testEvaluated"] = True
        result["test"] = test_metrics
        atomic_json(result, result_file)
        event = {"event": "resolution-test-evaluated", **result}
        reporter.emit(event)
        reporter.status("complete", planId=combo_id, latest=event)
        return
    optimizers = build_optimizers(model, training, device)  # type: ignore[arg-type]
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
    start_epoch = 0; global_step = 0; best_score = math.inf
    best_epoch = -1; stale_epochs = 0
    last_checkpoint = run_dir / "checkpoints" / "last.json"
    best_checkpoint = run_dir / "checkpoints" / "best.json"
    if checkpoint_exists(last_checkpoint):
        checkpoint = load_torch_checkpoint(
            last_checkpoint, map_location=device, weights_only=False
        )
        if checkpoint.get("planSha256") != plan_sha \
                or checkpoint.get("corpusFingerprint") != fingerprint \
                or checkpoint.get("runnerContract") != RUNNER_CONTRACT:
            raise ValueError("resolution checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        for optimizer, state in zip(
            optimizers, checkpoint["optimizers"], strict=True
        ): optimizer.load_state_dict(state)
        for scheduler, state in zip(
            schedulers, checkpoint["schedulers"], strict=True
        ): scheduler.load_state_dict(state)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["globalStep"])
        best_score = float(checkpoint["bestValidationScore"])
        best_epoch = int(checkpoint["bestEpoch"])
        stale_epochs = int(checkpoint["staleEpochs"])
    batch_size = int(settings["batchSize"])
    evaluation_batch_size = int(settings["evaluationBatchSize"])
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    target_std = torch.from_numpy(normalization.target_std).to(device)
    summary_std = torch.from_numpy(normalization.summary_std).to(device)
    summary_weights = torch.tensor((0, 0, 0, 0, 1), device=device)
    reporter.status(
        "training", planId=combo_id, startEpoch=start_epoch,
        batchSize=batch_size, parameters=parameters,
        bestEpoch=best_epoch, testPolicy="sealed-validation-only-no-evaluation",
    )
    paused = False
    for epoch in range(start_epoch, int(training["epochs"])):
        started = time.monotonic(); model.train()
        train_metrics = SequenceMetricAccumulator(
            sequence_normalization(normalization),
            candle_weight=1, summary_weight=1,
            summary_metric_weights=(0, 0, 0, 0, 1),
            device=device, track_per_lead=False,
        )
        for features, targets, weights in iter_device_batches(
            dataset.iter_batches(
                "train", batch_size, shuffle=True,
                seed=seed + epoch, reuse_buffers=True,
            ), device,
        ):
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = model(features)
                loss = sequence_objective_loss(
                    prediction, targets, weights,
                    target_std=target_std,
                    summary_std=summary_std,
                    summary_metric_weights=summary_weights,
                    candle_weight=1, summary_weight=1,
                )
            loss.backward()
            clip_grad_norm_(
                model.parameters(), float(training["gradientClip"]),
                foreach=device.type == "cuda",
            )
            for optimizer in optimizers: optimizer.step()
            train_metrics.add(prediction, targets, weights); global_step += 1
        validation = evaluate(
            model, dataset, "validation",
            batch_size=evaluation_batch_size,
            normalization=normalization, device=device, amp_dtype=amp_dtype,
        )
        score = float(validation["objective"])
        improved = score < best_score
        if improved:
            best_score = score; best_epoch = epoch; stale_epochs = 0
        else: stale_epochs += 1
        lr_before = float(optimizers[0].param_groups[0]["lr"])
        for scheduler in schedulers: scheduler.step(score)
        learning_rate = float(optimizers[0].param_groups[0]["lr"])
        checkpoint = {
            "model": model.state_dict(),
            "optimizers": [value.state_dict() for value in optimizers],
            "schedulers": [value.state_dict() for value in schedulers],
            "epoch": epoch, "globalStep": global_step,
            "bestValidationScore": best_score, "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs, "validation": validation,
            "parameterCount": parameters, "planSha256": plan_sha,
            "corpusFingerprint": fingerprint,
            "architectureContract": ARCHITECTURE_CONTRACT,
            "runnerContract": RUNNER_CONTRACT,
            "objectiveContract": OBJECTIVE_CONTRACT,
            "testEvaluated": False,
        }
        save_torch_checkpoint(checkpoint, last_checkpoint)
        if improved: save_torch_checkpoint(checkpoint, best_checkpoint)
        event = {
            "event": "resolution-epoch", "epoch": epoch,
            "epochs": int(training["epochs"]),
            "seconds": time.monotonic() - started,
            "globalStep": global_step,
            "train": train_metrics.result(include_per_lead=False),
            "validation": validation,
            "bestValidationScore": best_score, "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs, "improved": improved,
            "learningRate": learning_rate,
            "learningRateReduced": learning_rate < lr_before,
            "testEvaluated": False,
        }
        reporter.emit(event); reporter.status("training", planId=combo_id, latest=event)
        if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            paused = True; break
        if stale_epochs >= int(training["earlyStoppingPatience"]): break
    if paused:
        reporter.status("paused", planId=combo_id, latest={
            "event": "resolution-paused", "completedEpochs": epoch + 1,
            "bestEpoch": best_epoch, "testEvaluated": False,
        }); return
    best = load_torch_checkpoint(best_checkpoint, map_location=device, weights_only=False)
    result = {
        "completedAt": iso_now(), "planId": combo_id,
        "planSha256": plan_sha, "corpusFingerprint": fingerprint,
        "architectureContract": ARCHITECTURE_CONTRACT,
        "runnerContract": RUNNER_CONTRACT,
        "objectiveContract": OBJECTIVE_CONTRACT,
        "resolution": args.resolution, "maxWindow": args.max_window,
        "componentLabels": dataset.component_labels,
        "parameterCount": parameters,
        "horizonCandles": HORIZON_CANDLE_COUNT,
        "bestEpoch": int(best["bestEpoch"]),
        "bestValidationScore": float(best["bestValidationScore"]),
        "bestValidation": best["validation"],
        "checkpoint": str(best_checkpoint.relative_to(repo_root)),
        "testEvaluated": False,
    }
    atomic_json(result, run_dir / "state" / "result.json")
    reporter.emit({"event": "resolution-complete", **result})
    reporter.status("complete", planId=combo_id, latest=result)


if __name__ == "__main__":
    main()
