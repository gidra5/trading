from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from autoregressive_minute_return import (
    ARCHITECTURE_CONTRACT,
    AutoregressiveMinuteReturn,
)
from next_return_dataset import (
    ExampleShard,
    HISTORY_RETURN_COUNT,
    count_examples,
    select_example_shards,
)
from normalized_glu_next_return import (
    NormalizedGluNextReturn,
    optimizer_parameter_groups,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)
from train_normalized_glu_next_return import (
    MetricAccumulator,
    NextReturnDataset,
    Reporter,
    atomic_json,
    canonical_fingerprint,
    corpus_fingerprint,
    iter_device_batches,
    resolve,
    validate_source_manifest,
)


RUNNER_CONTRACT = "minute-aligned-full-bptt-two-second-chunks-to-one-minute-v1"
LEGACY_ARCHITECTURE_CONTRACT = (
    "autoregressive-pretrained-two-second-glu-to-single-minute-return-v1"
)
DIRECT_ARCHITECTURE_CONTRACT = "direct-glu-to-single-return-v3"
LEGACY_DIRECT_ARCHITECTURE_CONTRACTS = frozenset({
    "direct-one-layer-glu-to-single-return-v2",
    "direct-one-layer-glu-to-single-minute-return-v1",
})
DIRECT_ARCHITECTURE_CONTRACTS = frozenset({
    DIRECT_ARCHITECTURE_CONTRACT,
    *LEGACY_DIRECT_ARCHITECTURE_CONTRACTS,
})
MinuteReturnModel = AutoregressiveMinuteReturn | NormalizedGluNextReturn


def aligned_minute_shards(
    shards: dict[str, list[ExampleShard]],
) -> dict[str, list[ExampleShard]]:
    result: dict[str, list[ExampleShard]] = {name: [] for name in shards}
    for split, values in shards.items():
        for shard in values:
            remainder = shard.decision_time_start % 60_000
            offset = 0 if remainder == 0 else (60_000 - remainder) // 1_000
            if offset < shard.count:
                result[split].append(shard.shifted(int(offset)) if offset else shard)
    return result


def minute_target(target_path: Tensor) -> Tensor:
    if target_path.ndim != 2 or target_path.shape[1] != 60:
        raise ValueError("minute target requires sixty one-second returns")
    return target_path.float().sum(dim=1)


def scalar_target(target: Tensor, horizon_seconds: int) -> Tensor:
    if int(horizon_seconds) == 1:
        if target.ndim != 1:
            raise ValueError("next-second target must be one-dimensional")
        return target.float()
    if int(horizon_seconds) == 60:
        return minute_target(target)
    raise ValueError("single-return trainer supports only 1s and 60s horizons")


def training_normalization(
    dataset: NextReturnDataset,
    *,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    feature_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    feature_square_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    chunk_sum = np.zeros(2, dtype=np.float64)
    chunk_square_sum = np.zeros(2, dtype=np.float64)
    minute_sum = 0.0
    minute_square_sum = 0.0
    total = 0.0
    for features, targets, weights in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0, reuse_buffers=True
    ):
        input_values = features.numpy().astype(np.float64, copy=False)
        raw_targets = targets.numpy().astype(np.float64, copy=False)
        target_values = raw_targets[:, None] if raw_targets.ndim == 1 else raw_targets
        chunks = target_values[:, :2] if target_values.shape[1] >= 2 \
            else np.repeat(target_values, 2, axis=1)
        minutes = target_values.sum(axis=1)
        sample_weights = weights.numpy().astype(np.float64, copy=False)
        feature_sum += np.einsum("i,ij->j", sample_weights, input_values)
        feature_square_sum += np.einsum(
            "i,ij->j", sample_weights, np.square(input_values)
        )
        chunk_sum += np.einsum("i,ij->j", sample_weights, chunks)
        chunk_square_sum += np.einsum(
            "i,ij->j", sample_weights, np.square(chunks)
        )
        minute_sum += float(np.dot(sample_weights, minutes))
        minute_square_sum += float(np.dot(sample_weights, np.square(minutes)))
        total += float(sample_weights.sum())
    if int(round(total)) != dataset.logical_count("train"):
        raise RuntimeError("minute normalization did not cover the training corpus")

    def mean_std(
        values_sum: np.ndarray, square_sum: np.ndarray, minimum: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = values_sum / total
        variance = np.maximum(square_sum / total - np.square(mean), minimum)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    feature_mean, feature_std = mean_std(
        feature_sum, feature_square_sum, 1e-20
    )
    chunk_mean, chunk_std = mean_std(chunk_sum, chunk_square_sum, 1e-20)
    minute_mean = minute_sum / total
    minute_variance = max(
        minute_square_sum / total - minute_mean * minute_mean, 1e-14
    )
    return {
        "featureMean": feature_mean,
        "featureStd": feature_std,
        "chunkMean": chunk_mean,
        "chunkStd": chunk_std,
        "minuteMean": float(minute_mean),
        "minuteStd": math.sqrt(minute_variance),
    }


def direct_calendar_shards(
    split: dict,
    history_root: Path,
    *,
    horizon_seconds: int = 60,
    decision_stride_seconds: int = 60,
) -> dict[str, list[ExampleShard]]:
    if split.get("type") != "direct-calendar-months-v1":
        raise ValueError("direct candle split contract is invalid")
    if horizon_seconds not in {1, 60} or decision_stride_seconds not in {1, 60}:
        raise ValueError("calendar horizon and decision stride are unsupported")
    result: dict[str, list[ExampleShard]] = {
        "train": [], "validation": [], "test": [],
    }
    for name in result:
        start = date.fromisoformat(str(split[f"{name}Start"]))
        end = date.fromisoformat(str(split[f"{name}End"]))
        if end < start:
            raise ValueError(f"{name} calendar range is reversed")
        current = start
        while current <= end:
            previous = current - timedelta(days=1)
            following = current + timedelta(days=1)
            for required in (previous, current, following):
                if not (history_root / f"{required.isoformat()}.json").is_file():
                    raise FileNotFoundError(
                        f"calendar split requires candle day {required.isoformat()}"
                    )
            # Leave two minutes between the final training/validation target
            # and the next split's first 120-second input history.
            count = (
                86_400 - (HISTORY_RETURN_COUNT + horizon_seconds)
                + decision_stride_seconds
                if name != "test" and current == end else 86_400
            )
            timestamp = int(datetime(
                current.year, current.month, current.day, tzinfo=timezone.utc
            ).timestamp() * 1_000)
            result[name].append(ExampleShard(
                split=name,
                decision_time_start=timestamp,
                count=count,
                date=current.isoformat(),
                row_offset=0,
            ))
            current += timedelta(days=1)
    if not (
        result["train"][-1].decision_time_end
        < result["validation"][0].decision_time_start
        < result["test"][0].decision_time_start
    ):
        raise ValueError("calendar splits must be chronological")
    return result


def build_optimizers(
    model: MinuteReturnModel,
    training: dict,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, ...]:
    core = model.core if isinstance(model, AutoregressiveMinuteReturn) else model
    muon_parameters, adamw_parameters = optimizer_parameter_groups(core)
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


@torch.no_grad()
def evaluate(
    model: MinuteReturnModel,
    dataset: NextReturnDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    horizon_seconds: int = 60,
) -> dict:
    model.eval()
    metrics = MetricAccumulator(target_std, device)
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            prediction = model(features)
        metrics.add(prediction, scalar_target(targets, horizon_seconds), weights)
    return metrics.result()


def validate_plan(plan: dict) -> None:
    required = (
        "id", "datasetDir", "runDir", "historyDir", "architecture", "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("autoregressive minute plan is missing required fields")
    direct_split = plan.get("split")
    if direct_split is None:
        for name in ("sourceDatasetDir", "initialCheckpoint", "testExamples"):
            if name not in plan or plan[name] in (None, ""):
                raise ValueError(f"source-split plan requires {name}")
    else:
        for name in (
            "trainStart", "trainEnd", "validationStart", "validationEnd",
            "testStart", "testEnd",
        ):
            if name not in direct_split:
                raise ValueError(f"direct calendar split requires {name}")
    architecture = plan["architecture"]
    architecture_contract = architecture.get("contract")
    horizon_seconds = int(architecture.get("horizonSeconds", 0))
    widths = architecture.get("widths")
    if architecture_contract not in {
        ARCHITECTURE_CONTRACT,
        LEGACY_ARCHITECTURE_CONTRACT,
        *DIRECT_ARCHITECTURE_CONTRACTS,
    } or horizon_seconds not in {1, 60} \
            or not isinstance(widths, list) \
            or not widths \
            or any(
                not isinstance(width, int)
                or isinstance(width, bool)
                or width < 2
                for width in widths
            ) \
            or (
                architecture_contract not in DIRECT_ARCHITECTURE_CONTRACTS
                and (
                    horizon_seconds != 60
                    or int(architecture.get("chunkSeconds", 0)) != 2
                    or widths != [512]
                )
            ):
        raise ValueError("autoregressive minute architecture is invalid")
    training = plan["training"]
    for name in (
        "epochs", "batchSize", "evaluationBatchSize",
        "earlyStoppingPatience", "seed",
    ):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"training {name} must be positive")
    if training.get("device") not in {"cpu", "cuda"} \
            or training.get("mixedPrecision") != "bfloat16" \
            or float(training.get("learningRate", 0)) <= 0 \
            or float(training.get("gradientClip", 0)) <= 0:
        raise ValueError("autoregressive runtime settings are invalid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a full-BPTT autoregressive predictor for one 1m return."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    repo = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    architecture = plan["architecture"]
    horizon_seconds = int(architecture["horizonSeconds"])
    decision_stride_seconds = int(plan.get(
        "decisionStrideSeconds", 60 if horizon_seconds == 60 else 1
    ))
    plan_hash = canonical_fingerprint(plan)
    layout = training_storage_layout(repo)
    source_root = require_under(
        resolve(repo, Path(plan["sourceDatasetDir"])), layout.datasets,
        "sourceDatasetDir",
    ) if plan.get("sourceDatasetDir") else None
    dataset_root = require_under(
        resolve(repo, Path(plan["datasetDir"])), layout.datasets, "datasetDir"
    )
    run_root = require_under(
        resolve(repo, Path(plan["runDir"])), layout.runs, "runDir"
    )
    history_root = require_under(
        resolve(repo, Path(plan["historyDir"])),
        repo / "data/market/immutable/refs/candles", "historyDir",
    )
    initial_checkpoint_file = require_under(
        resolve(repo, Path(plan["initialCheckpoint"])), layout.runs,
        "initialCheckpoint",
    ) if plan.get("initialCheckpoint") else None
    reporter = Reporter(run_root)
    reporter.status("selecting-examples", planId=plan["id"])
    try:
        if plan.get("split") is not None:
            shards = direct_calendar_shards(
                plan["split"], history_root,
                horizon_seconds=horizon_seconds,
                decision_stride_seconds=decision_stride_seconds,
            )
            split_source = "direct immutable one-second candle calendar"
        else:
            if source_root is None:
                raise RuntimeError("source dataset root is missing")
            source_manifest = json.loads(
                (source_root / "dataset.json").read_text(encoding="utf-8")
            )
            validate_source_manifest(source_manifest)
            shards = aligned_minute_shards(select_example_shards(
                source_manifest,
                test_count=int(plan["testExamples"]),
                test_tail_offset=int(plan.get("testTailOffsetExamples", 0)),
                horizon_return_count=60,
            ))
            split_source = "existing immutable oracle corpus split"
        source_counts = count_examples(shards)
        fingerprint = corpus_fingerprint(
            shards, horizon_return_count=horizon_seconds
        )
        dataset = NextReturnDataset(
            shards, history_root, horizon_return_count=horizon_seconds,
            row_stride=decision_stride_seconds,
        )
        counts = {split: dataset.logical_count(split) for split in shards}
        selected = {
            "event": "minute-return-dataset-selected",
            "planId": plan["id"],
            "counts": counts,
            "sourceContiguousCounts": source_counts,
            "horizonSeconds": horizon_seconds,
            "chunkSeconds": int(plan["architecture"].get("chunkSeconds", 0)),
            "decisionStrideSeconds": decision_stride_seconds,
            "splitSource": split_source,
            "corpusFingerprint": fingerprint,
            "testPolicy": (
                "sealed-until-validation-selection"
                if bool(plan.get("evaluateTest", False))
                else "sealed-validation-only-no-evaluation"
            ),
        }
        reporter.emit(selected)
        if args.validate_only:
            reporter.status("paused", latest=selected)
            return

        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file():
            if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("run directory belongs to a different plan")
        else:
            atomic_json(snapshot, snapshot_file)

        reporter.status("computing-training-statistics", planId=plan["id"])
        normalization = training_normalization(
            dataset, batch_size=int(plan["training"]["evaluationBatchSize"])
        )
        minute_mean = float(normalization["minuteMean"])
        minute_std = float(normalization["minuteStd"])
        initial_state = None
        if initial_checkpoint_file is not None:
            initial = load_torch_checkpoint(
                initial_checkpoint_file, map_location="cpu", weights_only=False
            )
            if int(initial.get("horizonSeconds", 0)) != 2:
                raise ValueError("initial checkpoint is not a two-second predictor")
            initial_state = initial["model"]
        common_model_options = {
            "widths": tuple(int(width) for width in architecture["widths"]),
            "dropout": float(architecture["dropout"]),
            "dropout_rate": float(architecture["dropoutRate"]),
            "initial_radius": float(architecture["initialRadius"]),
            "minimum_radius": float(architecture["minimumRadius"]),
        }
        if architecture["contract"] in DIRECT_ARCHITECTURE_CONTRACTS:
            if initial_state is not None:
                raise ValueError("direct minute predictor must start from scratch")
            model = NormalizedGluNextReturn(
                torch.from_numpy(normalization["featureMean"]),
                torch.from_numpy(normalization["featureStd"]),
                torch.tensor(minute_mean),
                torch.tensor(minute_std),
                **common_model_options,
            )
        else:
            model = AutoregressiveMinuteReturn(
                initial_state["feature_mean"] if initial_state is not None
                else torch.from_numpy(normalization["featureMean"]),
                initial_state["feature_std"] if initial_state is not None
                else torch.from_numpy(normalization["featureStd"]),
                initial_state["target_mean"].reshape(2) if initial_state is not None
                else torch.from_numpy(normalization["chunkMean"]),
                initial_state["target_std"].reshape(2) if initial_state is not None
                else torch.from_numpy(normalization["chunkStd"]),
                horizon_seconds=60,
                chunk_seconds=2,
                **common_model_options,
            )
            if initial_state is not None:
                model.core.load_state_dict(initial_state)
        parameter_count = sum(value.numel() for value in model.parameters())
        atomic_json({
            "version": 1,
            "planId": plan["id"],
            "input": "120 completed one-second log returns",
            "target": (
                "the immediately following one-second log return"
                if horizon_seconds == 1
                else "one log return over the immediately following minute"
            ),
            "internalRollout": (
                "none; one direct model call"
                if architecture["contract"] in DIRECT_ARCHITECTURE_CONTRACTS
                else "30 differentiable two-second chunks"
            ),
            "counts": counts,
            "decisionStrideSeconds": decision_stride_seconds,
            "targetNormalization": {"mean": minute_mean, "std": minute_std},
            "modelNormalizationSource": (
                str(initial_checkpoint_file.relative_to(repo))
                if initial_checkpoint_file is not None else "training split only"
            ),
            "split": plan.get("split"),
            "corpusFingerprint": fingerprint,
        }, dataset_root / "dataset.json")

        training = plan["training"]
        device = torch.device(training["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but is unavailable")
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        model = model.to(device)
        optimizers = build_optimizers(model, training, device)
        schedule = training["learningRateSchedule"]
        schedulers = tuple(torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(schedule["factor"]),
            patience=int(schedule["patience"]),
            threshold=float(schedule["threshold"]), threshold_mode="abs",
            min_lr=float(schedule["minimumLearningRate"]),
        ) for optimizer in optimizers)
        last_file = run_root / "checkpoints/last.json"
        best_file = run_root / "checkpoints/best.json"
        start_epoch = 0
        global_step = 0
        best_validation = math.inf
        best_epoch = -1
        stale_epochs = 0
        if checkpoint_exists(last_file):
            checkpoint = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            if checkpoint.get("planSha256") != plan_hash \
                    or checkpoint.get("corpusFingerprint") != fingerprint \
                    or checkpoint.get("runnerContract") != RUNNER_CONTRACT:
                raise ValueError("autoregressive checkpoint contract changed")
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

        batch_size = int(training["batchSize"])
        evaluation_batch_size = int(training["evaluationBatchSize"])
        maximum_epochs = int(training["epochs"])
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        reporter.status(
            "training", planId=plan["id"], startEpoch=start_epoch,
            parameters=parameter_count, bestEpoch=best_epoch,
        )
        paused = False
        for epoch in range(start_epoch, maximum_epochs):
            started = time.monotonic()
            model.train()
            train_metrics = MetricAccumulator(minute_std, device)
            for features, target_paths, weights in iter_device_batches(
                dataset.iter_batches(
                    "train", batch_size, shuffle=True, seed=seed + epoch,
                    shuffle_rows=True, reuse_buffers=True,
                ),
                device,
            ):
                targets = scalar_target(target_paths, horizon_seconds)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(features)
                    per_example = ((prediction - targets) / minute_std).square()
                    loss = (per_example * weights).sum() / weights.sum()
                loss.backward()
                clip_grad_norm_(
                    model.parameters(), float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    optimizer.step()
                train_metrics.add(prediction, targets, weights)
                global_step += 1

            validation = evaluate(
                model, dataset, "validation", batch_size=evaluation_batch_size,
                target_std=minute_std, device=device, amp_dtype=amp_dtype,
                horizon_seconds=horizon_seconds,
            )
            score = float(validation["normalizedMse"])
            if not math.isfinite(score):
                raise FloatingPointError("validation MSE is non-finite")
            improved = score < best_validation
            if improved:
                best_validation = score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            learning_rate_before = float(optimizers[0].param_groups[0]["lr"])
            for scheduler in schedulers:
                scheduler.step(score)
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
                "parameterCount": parameter_count,
                "planSha256": plan_hash,
                "corpusFingerprint": fingerprint,
                "architectureContract": str(architecture["contract"]),
                "runnerContract": RUNNER_CONTRACT,
                "initialCheckpoint": (
                    str(initial_checkpoint_file.relative_to(repo))
                    if initial_checkpoint_file is not None else None
                ),
                "testEvaluated": False,
            }
            save_torch_checkpoint(checkpoint, last_file)
            if improved:
                save_torch_checkpoint(checkpoint, best_file)
            event = {
                "event": "minute-return-epoch",
                "epoch": epoch,
                "epochs": maximum_epochs,
                "seconds": time.monotonic() - started,
                "globalStep": global_step,
                "train": train_metrics.result(),
                "validation": validation,
                "bestValidationScore": best_validation,
                "bestEpoch": best_epoch,
                "staleEpochs": stale_epochs,
                "improved": improved,
                "learningRate": learning_rate,
                "learningRateReduced": learning_rate < learning_rate_before,
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
            if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
                paused = True
                break
            if stale_epochs >= int(training["earlyStoppingPatience"]):
                break

        if paused:
            reporter.status("paused", planId=plan["id"], bestEpoch=best_epoch)
            return
        best = load_torch_checkpoint(best_file, map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        test_metrics = evaluate(
            model, dataset, "test", batch_size=evaluation_batch_size,
            target_std=minute_std, device=device, amp_dtype=amp_dtype,
            horizon_seconds=horizon_seconds,
        ) if bool(plan.get("evaluateTest", False)) else None
        result = {
            "completedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "planId": plan["id"],
            "planSha256": plan_hash,
            "corpusFingerprint": fingerprint,
            "architectureContract": str(architecture["contract"]),
            "parameterCount": parameter_count,
            "prediction": (
                "single next-second log return"
                if horizon_seconds == 1 else "single next-minute log return"
            ),
            "internalRolloutChunks": (
                0 if architecture["contract"] in DIRECT_ARCHITECTURE_CONTRACTS
                else 30
            ),
            "bestEpoch": best_epoch,
            "bestValidationScore": best_validation,
            "bestValidation": best["validation"],
            "test": test_metrics,
            "checkpoint": str(best_file.relative_to(repo)),
            "testEvaluated": test_metrics is not None,
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "minute-return-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except KeyboardInterrupt:
        reporter.status("paused", planId=plan["id"], message="Interrupted")
        raise
    except Exception as error:
        reporter.status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
