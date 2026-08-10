from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from next_return_dataset import DAY_SECONDS, ExampleShard
from normalized_glu_next_return import (
    CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
    NormalizedGluNextReturn,
    TRAINING_POSITION_INPUT_NORMALIZATION,
    depth_width_parameter_assignments,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)
from train_autoregressive_minute_return import (
    build_optimizers,
    direct_calendar_shards,
    training_normalization,
)
from train_normalized_glu_next_return import (
    NextReturnDataset,
    Reporter,
    atomic_json,
    canonical_fingerprint,
    corpus_fingerprint,
    evaluate,
    iter_device_batches,
    resolve,
)
from swa import (
    EpochSwaSweep,
    swa_config,
    validate_swa_config,
)


RUNNER_CONTRACT = "next-second-fixed-subset-memorization-no-regularization-v1"
DROPOUT_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-dropout-regularization-v1"
)
CAUSAL_VOLATILITY_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-causal-volatility-v1"
)
SWA_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-swa-sweep-v1"
)
L2_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-explicit-l2-v1"
)
OPTIMIZER_WEIGHT_DECAY_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-optimizer-weight-decay-v1"
)
PAUSE_EXIT_CODE = 75


def runner_contract(plan: dict) -> str:
    if plan["training"].get("optimizerWeightDecay") is not None:
        return OPTIMIZER_WEIGHT_DECAY_RUNNER_CONTRACT
    if plan["training"].get("l2Regularization") is not None:
        return L2_RUNNER_CONTRACT
    if plan["training"].get("swa") is not None:
        return SWA_RUNNER_CONTRACT
    if plan["architecture"].get(
        "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
    ) == CAUSAL_VOLATILITY_INPUT_NORMALIZATION:
        return CAUSAL_VOLATILITY_RUNNER_CONTRACT
    return (
        RUNNER_CONTRACT
        if float(plan["architecture"].get("dropout", 0)) == 0
        else DROPOUT_RUNNER_CONTRACT
    )


def swa_variant(source: dict, *, suffix: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("SWA variant suffix is invalid")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - SWA sweep'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["swa"] = swa_config()
    return plan


def l2_variant(source: dict, *, rate: float, suffix: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("L2 variant suffix is invalid")
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("L2 rate must be finite and in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - explicit L2 rate {rate:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["l2Regularization"] = {
        "type": "explicit-loss-term",
        "coefficient": rate,
        "parameters": "all-trainable-matrices",
        "reduction": "half-sum-squared",
    }
    return plan


def optimizer_weight_decay_variant(
    source: dict, *, rate: float, suffix: str
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("optimizer weight-decay variant suffix is invalid")
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("optimizer weight-decay rate must be in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - optimizer weight decay {rate:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    optimizer = plan["training"]["optimizer"]
    optimizer["muon"]["weightDecay"] = rate
    optimizer["adamw"]["weightDecay"] = rate
    plan["training"]["optimizerWeightDecay"] = {
        "type": "optimizer-native",
        "coefficient": rate,
        "parameterGroups": ["muon", "adamw"],
    }
    return plan


def dropout_variant(
    source: dict,
    *,
    probability: float,
    application_rate: float,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("dropout variant suffix is invalid")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - dropout '
        f'p={probability:g}, application={application_rate:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["architecture"]["dropout"] = probability
    plan["architecture"]["dropoutRate"] = application_rate
    return plan


def causal_volatility_variant(
    source: dict,
    *,
    window: int,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("causal volatility variant suffix is invalid")
    if isinstance(window, bool) or not 1 <= window <= DAY_SECONDS:
        raise ValueError("causal volatility window must be in [1, 86,400]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - causal volatility window {window}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["architecture"]["inputNormalization"] = (
        CAUSAL_VOLATILITY_INPUT_NORMALIZATION
    )
    plan["architecture"]["volatilityWindow"] = window
    return plan


def fixed_subset_shards(
    history_root: Path,
    subset_date: date,
    examples: int,
) -> dict[str, list[ExampleShard]]:
    if examples < 1:
        raise ValueError("memorization subset examples must be positive")
    final_date = subset_date + timedelta(days=(examples - 1) // DAY_SECONDS)
    required = subset_date - timedelta(days=1)
    while required <= final_date + timedelta(days=1):
        if not (history_root / f"{required.isoformat()}.json").is_file():
            raise FileNotFoundError(
                f"memorization subset requires candle day {required.isoformat()}"
            )
        required += timedelta(days=1)
    train: list[ExampleShard] = []
    remaining = examples
    current = subset_date
    while remaining:
        count = min(remaining, DAY_SECONDS)
        timestamp = int(datetime(
            current.year,
            current.month,
            current.day,
            tzinfo=timezone.utc,
        ).timestamp() * 1_000)
        train.append(ExampleShard(
            split="train",
            decision_time_start=timestamp,
            count=count,
            date=current.isoformat(),
            row_offset=0,
        ))
        remaining -= count
        current += timedelta(days=1)
    return {
        "train": train,
        "validation": [],
        "test": [],
    }


def validate_plan(plan: dict) -> None:
    for name in (
        "id", "datasetDir", "runDir", "historyDir", "subset",
        "architecture", "training",
    ):
        if name not in plan or plan[name] in (None, ""):
            raise ValueError(f"memorization plan requires {name}")
    subset = plan["subset"]
    subset_type = str(subset.get("type", "fixed-contiguous"))
    if subset_type == "fixed-contiguous":
        date.fromisoformat(str(subset["date"]))
        if int(subset["examples"]) < 1:
            raise ValueError("memorization subset size is invalid")
    elif subset_type == "calendar-training-split":
        split = subset.get("split", {})
        for name in (
            "trainStart", "trainEnd", "validationStart", "validationEnd",
            "testStart", "testEnd",
        ):
            date.fromisoformat(str(split[name]))
    else:
        raise ValueError("memorization subset type is invalid")
    architecture = plan["architecture"]
    input_normalization = architecture.get(
        "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
    )
    volatility_window = architecture.get("volatilityWindow")
    widths = architecture.get("widths")
    if not isinstance(widths, list) \
            or not widths \
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 2
                for value in widths
            ) \
            or not 0 <= float(architecture.get("dropout", -1)) < 1 \
            or not 0 <= float(architecture.get("dropoutRate", -1)) <= 1 \
            or (
                float(architecture.get("dropout", 0)) > 0
                and float(architecture.get("dropoutRate", 0)) == 0
            ) \
            or not isinstance(architecture.get("learnableCentering", True), bool):
        raise ValueError(
            "memorization diagnostic architecture is invalid"
        )
    if input_normalization == CAUSAL_VOLATILITY_INPUT_NORMALIZATION:
        if not isinstance(volatility_window, int) \
                or isinstance(volatility_window, bool) \
                or not 1 <= volatility_window <= DAY_SECONDS:
            raise ValueError(
                "causal volatility window must be in [1, 86,400]"
            )
    elif input_normalization != TRAINING_POSITION_INPUT_NORMALIZATION \
            or volatility_window is not None:
        raise ValueError("memorization input normalization is invalid")
    training = plan["training"]
    if int(training.get("epochs", 0)) < 1 \
            or int(training.get("batchSize", 0)) < 1 \
            or int(training.get("evaluationBatchSize", 0)) < 1 \
            or float(training.get("learningRate", 0)) <= 0 \
            or not 0 < float(training.get("targetNormalizedMse", 0)) < 1 \
            or training.get("mixedPrecision") != "float32" \
            or training.get("device") not in {"cpu", "cuda"}:
        raise ValueError("memorization training settings are invalid")
    if "learningRateSchedule" in training or "earlyStoppingPatience" in training:
        raise ValueError("memorization diagnostic must not use validation controls")
    if training.get("swa") is not None:
        validate_swa_config(training["swa"])
    l2 = training.get("l2Regularization")
    if l2 is not None and (
        l2.get("type") != "explicit-loss-term"
        or l2.get("parameters") != "all-trainable-matrices"
        or l2.get("reduction") != "half-sum-squared"
        or not math.isfinite(float(l2.get("coefficient", 0)))
        or not 0 < float(l2["coefficient"]) <= 1
    ):
        raise ValueError("explicit L2 regularization contract is invalid")
    optimizer_decay = training.get("optimizerWeightDecay")
    if optimizer_decay is not None:
        coefficient = float(optimizer_decay.get("coefficient", 0))
        optimizer = training["optimizer"]
        if optimizer_decay.get("type") != "optimizer-native" \
                or optimizer_decay.get("parameterGroups") != ["muon", "adamw"] \
                or not math.isfinite(coefficient) \
                or not 0 < coefficient <= 1 \
                or float(optimizer["muon"]["weightDecay"]) != coefficient \
                or float(optimizer["adamw"]["weightDecay"]) != coefficient:
            raise ValueError("optimizer weight-decay contract is invalid")
    if l2 is not None and optimizer_decay is not None:
        raise ValueError("L2 loss and optimizer weight decay are separate variants")
    partitioning = training.get("parameterPartitioning")
    if partitioning is not None and partitioning != {
        "type": "depth-width-quadrants-v1",
        "rotation": "epoch-round-robin",
    }:
        raise ValueError("memorization parameter partitioning is invalid")
    if partitioning is not None and (
        len(architecture["widths"]) % 2 != 0
        or any(int(width) % 2 != 0 for width in architecture["widths"])
    ):
        raise ValueError("depth-width partitioning requires even depth and width")


@torch.no_grad()
def mask_inactive_parameter_updates(
    assignments: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    active_partition: int,
    optimizers: tuple[torch.optim.Optimizer, ...],
) -> None:
    """Mask gradients and optimizer state outside the active quadrant."""
    if active_partition not in range(4):
        raise ValueError("active parameter partition must be in [0, 3]")
    for parameter, assignment in assignments:
        inactive = assignment != active_partition
        if parameter.grad is not None:
            parameter.grad.masked_fill_(inactive, 0)
        for optimizer in optimizers:
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(inactive, 0)


@torch.no_grad()
def snapshot_inactive_parameter_values(
    assignments: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    active_partition: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            parameter,
            assignment != active_partition,
            parameter.detach()[assignment != active_partition].clone(),
        )
        for parameter, assignment in assignments
    )


@torch.no_grad()
def assert_inactive_parameter_values_unchanged(
    frozen: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> None:
    if any(
        not torch.equal(parameter.detach()[inactive], values)
        for parameter, inactive, values in frozen
    ):
        raise RuntimeError("inactive parameter quadrant changed during training")


def matrix_l2_penalty(model: NormalizedGluNextReturn) -> torch.Tensor:
    matrices = tuple(
        parameter.float()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim == 2
    )
    if not matrices:
        raise RuntimeError("explicit L2 objective found no trainable matrices")
    return 0.5 * sum(
        (parameter.square().sum() for parameter in matrices),
        start=torch.zeros((), device=matrices[0].device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether a normalized GLU can memorize a fixed subset."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--dropout-rate", type=float)
    parser.add_argument("--causal-volatility-window", type=int)
    parser.add_argument("--swa-sweep", action="store_true")
    parser.add_argument("--l2-rate", type=float)
    parser.add_argument("--optimizer-weight-decay", type=float)
    parser.add_argument("--variant-suffix")
    parser.add_argument(
        "--pause-file",
        type=Path,
        help=(
            "At completed-epoch boundaries, exit with a resumable status while "
            "this file exists."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    dropout_values = (args.dropout, args.dropout_rate)
    if any(value is not None for value in dropout_values):
        if any(value is None for value in dropout_values) \
                or args.variant_suffix is None \
                or args.causal_volatility_window is not None \
                or args.swa_sweep \
                or args.l2_rate is not None \
                or args.optimizer_weight_decay is not None:
            raise ValueError(
                "dropout, dropout-rate, and variant-suffix must be supplied together"
            )
        plan = dropout_variant(
            plan,
            probability=float(args.dropout),
            application_rate=float(args.dropout_rate),
            suffix=str(args.variant_suffix),
        )
    elif args.causal_volatility_window is not None:
        if args.variant_suffix is None \
                or args.swa_sweep \
                or args.l2_rate is not None \
                or args.optimizer_weight_decay is not None:
            raise ValueError(
                "causal-volatility-window and variant-suffix must be supplied together"
            )
        plan = causal_volatility_variant(
            plan,
            window=int(args.causal_volatility_window),
            suffix=str(args.variant_suffix),
        )
    elif args.swa_sweep:
        if args.variant_suffix is None \
                or args.l2_rate is not None \
                or args.optimizer_weight_decay is not None:
            raise ValueError(
                "swa-sweep and variant-suffix must be supplied together"
            )
        plan = swa_variant(
            plan, suffix=str(args.variant_suffix)
        )
    elif args.l2_rate is not None:
        if args.variant_suffix is None \
                or args.optimizer_weight_decay is not None:
            raise ValueError(
                "l2-rate and variant-suffix must be supplied together"
            )
        plan = l2_variant(
            plan,
            rate=float(args.l2_rate),
            suffix=str(args.variant_suffix),
        )
    elif args.optimizer_weight_decay is not None:
        if args.variant_suffix is None:
            raise ValueError(
                "optimizer-weight-decay and variant-suffix must be supplied together"
            )
        plan = optimizer_weight_decay_variant(
            plan,
            rate=float(args.optimizer_weight_decay),
            suffix=str(args.variant_suffix),
        )
    elif args.variant_suffix is not None:
        raise ValueError("variant-suffix requires a configured variant")
    validate_plan(plan)
    active_runner_contract = runner_contract(plan)
    plan_hash = canonical_fingerprint(plan)
    layout = training_storage_layout(repo)
    dataset_root = require_under(
        resolve(repo, Path(plan["datasetDir"])), layout.datasets, "datasetDir"
    )
    run_root = require_under(
        resolve(repo, Path(plan["runDir"])), layout.runs, "runDir"
    )
    history_root = require_under(
        resolve(repo, Path(plan["historyDir"])),
        repo / "data/market/immutable/refs/candles",
        "historyDir",
    )
    reporter = Reporter(run_root)
    try:
        subset = plan["subset"]
        subset_type = str(subset.get("type", "fixed-contiguous"))
        if subset_type == "fixed-contiguous":
            shards = fixed_subset_shards(
                history_root,
                date.fromisoformat(str(subset["date"])),
                int(subset["examples"]),
            )
            split_source = "fixed contiguous memorization subset"
        else:
            calendar = direct_calendar_shards(
                subset["split"], history_root,
                horizon_seconds=1, decision_stride_seconds=1,
            )
            shards = {
                "train": calendar["train"],
                "validation": [],
                "test": [],
            }
            split_source = "complete current calendar training split"
        fingerprint = corpus_fingerprint(shards, horizon_return_count=1)
        dataset = NextReturnDataset(
            shards, history_root, horizon_return_count=1, row_stride=1
        )
        selection = {
            "event": "minute-return-dataset-selected",
            "planId": plan["id"],
            "counts": {"train": dataset.logical_count("train")},
            "horizonSeconds": 1,
            "decisionStrideSeconds": 1,
            "splitSource": split_source,
            "corpusFingerprint": fingerprint,
            "testPolicy": "none; training interpolation diagnostic only",
        }
        reporter.emit(selection)
        if args.validate_only:
            reporter.status("paused", latest=selection)
            return

        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file():
            if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("run directory belongs to a different plan")
        else:
            atomic_json(snapshot, snapshot_file)

        training = plan["training"]
        reporter.status("computing-training-statistics", planId=plan["id"])
        normalization = training_normalization(
            dataset, batch_size=int(training["evaluationBatchSize"])
        )
        target_mean = float(normalization["minuteMean"])
        target_std = float(normalization["minuteStd"])
        architecture = plan["architecture"]
        causal_volatility_window = (
            int(architecture["volatilityWindow"])
            if architecture.get("inputNormalization")
            == CAUSAL_VOLATILITY_INPUT_NORMALIZATION
            else None
        )
        model = NormalizedGluNextReturn(
            torch.from_numpy(normalization["featureMean"]),
            torch.from_numpy(normalization["featureStd"]),
            torch.tensor(target_mean),
            torch.tensor(target_std),
            widths=tuple(int(value) for value in architecture["widths"]),
            input_normalization=architecture.get(
                "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
            ),
            volatility_window=architecture.get("volatilityWindow"),
            dropout=float(architecture["dropout"]),
            dropout_rate=float(architecture["dropoutRate"]),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
            learnable_centering=bool(
                architecture.get("learnableCentering", True)
            ),
        )
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_parameter_count = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        atomic_json({
            "version": 1,
            "planId": plan["id"],
            "input": "120 completed one-second log returns",
            "inputNormalization": (
                {
                    "type": "causal-volatility-rms",
                    "window": int(architecture["volatilityWindow"]),
                    "targetScale": "same causal input-only volatility",
                }
                if architecture.get("inputNormalization")
                == CAUSAL_VOLATILITY_INPUT_NORMALIZATION
                else "training-set mean and standard deviation per lag position"
            ),
            "target": "the immediately following one-second log return",
            "counts": {"train": dataset.logical_count("train")},
            "selection": "lowest deterministic evaluation-mode training MSE",
            "regularization": (
                training["l2Regularization"]
                if training.get("l2Regularization") is not None
                else (
                    training["optimizerWeightDecay"]
                    if training.get("optimizerWeightDecay") is not None
                    else (
                        "none: dropout and weight decay are zero"
                        if float(architecture.get("dropout", 0)) == 0
                        else {
                            "type": "intermittent-activation-dropout",
                            "probability": float(architecture["dropout"]),
                            "applicationRate": float(architecture["dropoutRate"]),
                            "weightDecay": 0,
                        }
                    )
                )
            ),
            "schedule": "fixed learning rate; no validation or early stopping",
            "swa": training.get("swa"),
            "parameterTraining": (
                training.get("parameterPartitioning")
                or "all trainable parameters updated every batch"
            ),
            "centeringMatrix": (
                "learned independently for value and gate branches"
                if architecture.get("learnableCentering", True)
                else "static canonical projector I - 11^T/d"
            ),
            "corpusFingerprint": fingerprint,
        }, dataset_root / "dataset.json")

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
        maximum_epochs = int(training["epochs"])
        swa = (
            EpochSwaSweep(model, maximum_epochs=maximum_epochs)
            if training.get("swa") is not None else None
        )
        partitioning = training.get("parameterPartitioning")
        partition_assignments = (
            depth_width_parameter_assignments(model)
            if partitioning is not None else ()
        )
        partition_parameter_counts = (
            [
                sum(
                    int((assignment == index).sum().item())
                    for _parameter, assignment in partition_assignments
                )
                for index in range(4)
            ]
            if partition_assignments else []
        )
        last_file = run_root / "checkpoints/last.json"
        best_file = run_root / "checkpoints/best.json"
        start_epoch = 0
        global_step = 0
        best_train_score = math.inf
        best_epoch = -1
        if checkpoint_exists(last_file):
            checkpoint = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            if checkpoint.get("planSha256") != plan_hash \
                    or checkpoint.get("corpusFingerprint") != fingerprint \
                    or checkpoint.get("runnerContract") != active_runner_contract:
                raise ValueError("memorization checkpoint contract changed")
            model.load_state_dict(checkpoint["model"])
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["globalStep"])
            best_train_score = float(checkpoint["bestTrainScore"])
            best_epoch = int(checkpoint["bestEpoch"])
            if swa is not None:
                swa.load_state_dict(
                    checkpoint["swaState"]
                )

        batch_size = int(training["batchSize"])
        eval_batch_size = int(training["evaluationBatchSize"])
        target_score = float(training["targetNormalizedMse"])
        l2_rate = float(
            training.get("l2Regularization", {}).get("coefficient", 0)
        )
        reporter.status(
            "training", planId=plan["id"], startEpoch=start_epoch,
            parameters=parameter_count,
            trainableParameters=trainable_parameter_count,
            parameterPartitionCounts=partition_parameter_counts,
            bestEpoch=best_epoch,
        )
        converged = False
        for epoch in range(start_epoch, maximum_epochs):
            started = time.monotonic()
            active_partition = epoch % 4 if partition_assignments else None
            frozen_epoch = (
                snapshot_inactive_parameter_values(
                    partition_assignments, int(active_partition)
                )
                if active_partition is not None else ()
            )
            model.train()
            for batch in iter_device_batches(
                dataset.iter_batches(
                    "train", batch_size, shuffle=True, seed=seed + epoch,
                    shuffle_rows=True, reuse_buffers=True,
                    causal_volatility_window=causal_volatility_window,
                ),
                device,
            ):
                features, targets, weights = batch[:3]
                volatility = batch[3] if len(batch) == 4 else None
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    features, causal_volatility_rms=volatility
                )
                per_example = (
                    (prediction - targets.float()) / target_std
                ).square()
                loss = (per_example * weights).sum() / weights.sum()
                if l2_rate > 0:
                    loss = loss + l2_rate * matrix_l2_penalty(model)
                loss.backward()
                if active_partition is not None:
                    mask_inactive_parameter_updates(
                        partition_assignments,
                        int(active_partition),
                        optimizers,
                    )
                clip_grad_norm_(
                    model.parameters(), float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    optimizer.step()
                global_step += 1

            assert_inactive_parameter_values_unchanged(frozen_epoch)
            if swa is not None:
                swa.update(model, epoch=epoch)

            deterministic_train = evaluate(
                model, dataset, "train", batch_size=eval_batch_size,
                target_std=target_std, device=device, amp_dtype=torch.float32,
                causal_volatility_window=causal_volatility_window,
            )
            score = float(deterministic_train["normalizedMse"])
            l2_penalty = float(
                (l2_rate * matrix_l2_penalty(model)).detach().item()
            ) if l2_rate > 0 else 0.0
            if not math.isfinite(score):
                raise FloatingPointError("training MSE is non-finite")
            improved = score < best_train_score
            if improved:
                best_train_score = score
                best_epoch = epoch
            checkpoint = {
                "model": model.state_dict(),
                "optimizers": [value.state_dict() for value in optimizers],
                "epoch": epoch,
                "globalStep": global_step,
                "bestTrainScore": best_train_score,
                "bestEpoch": best_epoch,
                "train": deterministic_train,
                "l2Penalty": l2_penalty,
                "objective": score + l2_penalty,
                "parameterCount": parameter_count,
                "trainableParameterCount": trainable_parameter_count,
                "parameterPartitionCounts": partition_parameter_counts,
                "planSha256": plan_hash,
                "corpusFingerprint": fingerprint,
                "runnerContract": active_runner_contract,
                "swaState": (
                    swa.state_dict() if swa is not None else None
                ),
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
                "train": deterministic_train,
                "l2Penalty": l2_penalty,
                "objective": score + l2_penalty,
                "validation": {},
                "bestTrainScore": best_train_score,
                "bestEpoch": best_epoch,
                "improved": improved,
                "learningRate": float(optimizers[0].param_groups[0]["lr"]),
                "activeParameterPartition": active_partition,
                "diagnostic": "deterministic eval-mode train metrics",
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
            if args.pause_file is not None \
                    and resolve(repo, args.pause_file).is_file():
                reporter.status(
                    "paused",
                    planId=plan["id"],
                    latest=event,
                    pausedAt=datetime.now(timezone.utc).isoformat(),
                    message=(
                        "Matrix pause requested; resume from the last completed "
                        "epoch."
                    ),
                    resumableCheckpoint=str(last_file.relative_to(repo)),
                )
                raise SystemExit(PAUSE_EXIT_CODE)
            if score <= target_score:
                converged = True
                break

        best = load_torch_checkpoint(
            best_file, map_location=device, weights_only=False
        )
        model.load_state_dict(best["model"])
        final_train = evaluate(
            model, dataset, "train", batch_size=eval_batch_size,
            target_std=target_std, device=device, amp_dtype=torch.float32,
            causal_volatility_window=causal_volatility_window,
        )
        selected_checkpoint_file = best_file
        swa_candidates = None
        if swa is not None:
            candidate_states = swa.candidate_states()
            candidate_train = {}
            for candidate_id, candidate_state in candidate_states.items():
                model.load_state_dict(candidate_state)
                candidate_train[candidate_id] = evaluate(
                    model, dataset, "train", batch_size=eval_batch_size,
                    target_std=target_std, device=device,
                    amp_dtype=torch.float32,
                    causal_volatility_window=causal_volatility_window,
                )
            selected_checkpoint_file = (
                run_root / "checkpoints/swa-sweep.json"
            )
            save_torch_checkpoint({
                "model": best["model"],
                "swaCandidates": candidate_states,
                "train": final_train,
                "swaCandidateTrain": candidate_train,
                "epoch": swa.completed_epochs - 1,
                "rawBestEpoch": best_epoch,
                "planSha256": plan_hash,
                "corpusFingerprint": fingerprint,
                "runnerContract": active_runner_contract,
            }, selected_checkpoint_file)
            swa_candidates = {
                "raw-best": final_train,
                **candidate_train,
            }
        result = {
            "planId": plan["id"],
            "planSha256": plan_hash,
            "corpusFingerprint": fingerprint,
            "runnerContract": active_runner_contract,
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_parameter_count,
            "parameterPartitionCounts": partition_parameter_counts,
            "examples": dataset.logical_count("train"),
            "bestEpoch": best_epoch,
            "bestTrainScore": best_train_score,
            "train": final_train,
            "l2Rate": l2_rate,
            "optimizerWeightDecay": training.get("optimizerWeightDecay"),
            "l2Penalty": (
                float((l2_rate * matrix_l2_penalty(model)).detach().item())
                if l2_rate > 0 else 0.0
            ),
            "targetNormalizedMse": target_score,
            "converged": converged,
            "checkpoint": str(selected_checkpoint_file.relative_to(repo)),
            "swaCandidates": swa_candidates,
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
