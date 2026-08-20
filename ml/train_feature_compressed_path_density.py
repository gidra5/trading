from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Iterator

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from active_return_path_dataset import CandleCloseCache
from compressed_path_return_density import (
    ARCHITECTURE_CONTRACT,
    CompressedPathReturnDensity,
    path_log_density_terms,
)
from recurrent_market_path_density import (
    ARCHITECTURE_CONTRACT as RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
    RecurrentMarketPathDensity,
)
from normalized_glu_next_return import optimizer_parameter_groups
from return_knot_density import KnotDensityContract
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_autoregressive_minute_return import build_optimizers
from train_next_return_knot_density import PAUSE_EXIT_CODE, canonical_hash
from train_next_return_memorization import (
    mean_teacher_ema_decay,
    sam_perturb_parameters,
    sam_restore_parameters,
)
from train_normalized_glu_next_return import (
    MetricAccumulator,
    Reporter,
    atomic_json,
)


RUNNER_CONTRACT = "feature-immediate-compressed-active-return-path-density-v3"
RECURRENT_MARKET_RUNNER_CONTRACT = (
    "feature-lag3-recurrent-market-compressed-path-density-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the compressed multi-step return-density model."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--replace-smoke", action="store_true")
    return parser.parse_args()


class ImmediateFeatureActivePathDataset:
    """Recent channel rows with the next H chronological active returns."""

    def __init__(
        self,
        root: Path,
        history_root: Path,
        return_count: int,
        feature_history: int = 1,
        train_examples: int | None = None,
    ) -> None:
        self.root = root
        self.history_root = history_root
        self.return_count = int(return_count)
        self.manifest = json.loads((root / "manifest.json").read_text(
            encoding="utf-8"
        ))
        if self.manifest.get("storageLayout") != "temporal-channel-timeline-v1":
            raise ValueError("compressed path run requires a compact timeline dataset")
        self.channel_count = int(self.manifest["temporalChannelCount"])
        self.feature_history = int(feature_history)
        if self.feature_history <= 0:
            raise ValueError("feature history must be positive")
        self.feature_count = self.channel_count * self.feature_history
        self.splits: dict[str, tuple[np.memmap, np.memmap, np.ndarray]] = {}
        for split in ("train", "validation", "test"):
            stored_count = int(self.manifest["examplesBySplit"][split])
            count = (
                min(stored_count, int(train_examples))
                if split == "train" and train_examples is not None
                else stored_count
            )
            timeline_rows = int(self.manifest["timelineRowsBySplit"][split])
            timeline = np.memmap(
                root / f"{split}.timeline-features.f32", dtype="<f4", mode="r",
                shape=(timeline_rows, self.channel_count),
            )
            origins = np.memmap(
                root / f"{split}.origins.i32", dtype="<i4", mode="r",
                shape=(stored_count,),
            )[:count]
            targets = np.asarray(np.memmap(
                root / f"{split}.targets.f32", dtype="<f4", mode="r",
                shape=(stored_count,),
            )[:count], dtype=np.float32)
            times = np.asarray(np.memmap(
                root / f"{split}.times.f64", dtype="<f8", mode="r",
                shape=(stored_count,),
            )[:count], dtype=np.float64)
            if np.any(targets == 0) or np.any(np.diff(times) <= 0):
                raise ValueError(f"{split} is not a chronological clean corpus")
            if int(origins.min()) < self.feature_history - 1:
                raise ValueError(f"{split} lacks the requested feature history")
            tail = self._active_tail(int(times[-1]), self.return_count - 1)
            extended = np.concatenate((targets, tail))
            paths = np.lib.stride_tricks.sliding_window_view(
                extended, self.return_count
            )[:count]
            self.splits[split] = (timeline, origins, paths)

    def _active_tail(self, final_target_time_ms: int, count: int) -> np.ndarray:
        cache = CandleCloseCache(self.history_root, rows_per_day=86_400)
        values: list[np.float32] = []
        timestamp = int(final_target_time_ms) + 1_000
        while len(values) < count:
            point = datetime.fromtimestamp(timestamp / 1_000, timezone.utc)
            day = point.date()
            second = point.hour * 3600 + point.minute * 60 + point.second
            current = cache.load(day.isoformat())
            if second == 0:
                previous = cache.load((day - timedelta(days=1)).isoformat())[-1]
            else:
                previous = current[second - 1]
            value = np.float32(np.log(current[second] / previous))
            if value != 0:
                values.append(value)
            timestamp += 1_000
        return np.asarray(values, dtype=np.float32)

    def logical_count(self, split: str) -> int:
        return int(self.splits[split][1].size)

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        timeline, origins, targets = self.splits[split]
        count = origins.size
        if limit is not None:
            count = min(count, int(limit))
        order = np.arange(origins.size, dtype=np.int64)
        if limit is not None and count < origins.size:
            order = np.linspace(0, origins.size - 1, count, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, count, batch_size):
            selected = order[start:start + batch_size]
            selected_origins = np.asarray(origins[selected], dtype=np.int64)
            offsets = np.arange(
                self.feature_history - 1, -1, -1, dtype=np.int64
            )
            feature_rows = np.asarray(
                timeline[selected_origins[:, None] - offsets[None, :]],
                dtype=np.float32,
            ).reshape(selected.size, self.feature_count)
            target_rows = np.asarray(targets[selected], dtype=np.float32)
            yield (
                torch.from_numpy(feature_rows.copy()),
                torch.from_numpy(target_rows.copy()),
                torch.ones(selected.size, dtype=torch.float32),
            )


class CalibrationPathDataset:
    """Immediate features and clean paths immediately before validation."""

    def __init__(
        self, root: Path, return_count: int, feature_history: int = 1
    ) -> None:
        self.root = root
        self.return_count = int(return_count)
        self.manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        count = int(self.manifest["examples"])
        feature_count = int(self.manifest["featureCount"])
        matrix = np.memmap(
            root / "calibration.features.f32", dtype="<f4", mode="r",
            shape=(count, feature_count),
        )
        channels_value = self.manifest.get("temporalChannelCount")
        history = int(self.manifest.get("featureHistorySeconds", 1))
        requested_history = int(feature_history)
        if requested_history <= 0:
            raise ValueError("feature history must be positive")
        if channels_value is None:
            if requested_history != 1:
                raise ValueError("flat calibration data has no temporal history")
            self.features = matrix
            self.feature_count = feature_count
        else:
            channels = int(channels_value)
            if feature_count != channels * history:
                raise ValueError("calibration temporal dimensions are inconsistent")
            if requested_history > history:
                raise ValueError("calibration history is shorter than requested")
            if self.manifest.get("temporalLayout") == "time-major":
                self.features = np.ascontiguousarray(
                    matrix.reshape(count, history, channels)[:, -requested_history:, :]
                    .reshape(count, channels * requested_history)
                )
            else:
                self.features = np.ascontiguousarray(
                    matrix.reshape(count, channels, history)[:, :, -requested_history:]
                    .transpose(0, 2, 1)
                    .reshape(count, channels * requested_history)
                )
            self.feature_count = channels * requested_history
        targets = np.asarray(np.memmap(
            root / "calibration.targets.f32", dtype="<f4", mode="r",
            shape=(count,),
        ), dtype=np.float32)
        self.times = np.asarray(np.memmap(
            root / "calibration.times.f64", dtype="<f8", mode="r",
            shape=(count,),
        ), dtype=np.float64)
        if np.any(targets == 0) or np.any(np.diff(self.times) <= 0):
            raise ValueError("calibration examples are not clean and chronological")
        if count < self.return_count:
            raise ValueError("calibration split is shorter than one path")
        self.targets = np.lib.stride_tricks.sliding_window_view(
            targets, self.return_count
        )

    def logical_count(self, split: str) -> int:
        if split != "calibration":
            raise KeyError(split)
        return int(self.targets.shape[0])

    def iter_batches(
        self, split: str, batch_size: int, *, shuffle: bool, seed: int,
        limit: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        del shuffle, seed
        if split != "calibration":
            raise KeyError(split)
        count = self.logical_count(split)
        if limit is not None:
            count = min(count, int(limit))
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            yield (
                torch.from_numpy(np.asarray(
                    self.features[start:stop], dtype=np.float32
                ).copy()),
                torch.from_numpy(np.asarray(
                    self.targets[start:stop], dtype=np.float32
                ).copy()),
                torch.ones(stop - start, dtype=torch.float32),
            )


@torch.no_grad()
def collect_expectation_arrays(
    model: CompressedPathReturnDensity, dataset, split: str, *,
    batch_size: int, device: torch.device, limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    for features, target, _weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit
    ):
        predictions.append(model(features.to(
            device, non_blocking=True
        )).expectations.cpu().numpy())
        targets.append(target.numpy())
    return (
        np.concatenate(predictions).astype(np.float64, copy=False),
        np.concatenate(targets).astype(np.float64, copy=False),
    )


def rolling_online_per_step_affine(
    history_prediction: np.ndarray,
    history_target: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    window: int,
    ridge: float,
    input_scales: np.ndarray | None = None,
    output_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Causal affine fits using only targets resolved before each origin."""
    if history_prediction.shape != history_target.shape \
            or prediction.shape != target.shape:
        raise ValueError("online calibration prediction/target shapes differ")
    steps = prediction.shape[1]
    input_scales = np.maximum(
        history_prediction.std(axis=0), 1e-12
    ) if input_scales is None else input_scales
    output_scales = np.maximum(
        history_target.std(axis=0), 1e-12
    ) if output_scales is None else output_scales
    output = np.empty_like(prediction, dtype=np.float64)
    history_events = history_prediction.shape[0]
    current_events = prediction.shape[0]
    ends = history_events + np.arange(current_events)
    starts = np.maximum(0, ends - int(window))
    penalty = np.diag((0.0, float(ridge)))
    for lead in range(steps):
        gram_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for values, targets in (
            (history_prediction, history_target[:, 0]),
            (prediction, target[:, 0]),
        ):
            events = values.shape[0]
            gram = np.zeros((events, 2, 2), dtype=np.float64)
            right = np.zeros((events, 2), dtype=np.float64)
            origins = events - lead
            if origins > 0:
                normalized = values[:origins, lead] / input_scales[lead]
                design = np.stack((np.ones_like(normalized), normalized), axis=1)
                resolved = targets[lead:lead + origins] / output_scales[lead]
                gram[lead:lead + origins] = np.einsum(
                    "ni,nj->nij", design, design
                )
                right[lead:lead + origins] = design * resolved[:, None]
            gram_parts.append(gram)
            right_parts.append(right)
        gram = np.concatenate(gram_parts)
        right = np.concatenate(right_parts)
        cumulative_gram = np.concatenate((
            np.zeros((1, 2, 2), dtype=np.float64), np.cumsum(gram, axis=0)
        ))
        cumulative_right = np.concatenate((
            np.zeros((1, 2), dtype=np.float64), np.cumsum(right, axis=0)
        ))
        rolling_gram = cumulative_gram[ends] - cumulative_gram[starts]
        rolling_right = cumulative_right[ends] - cumulative_right[starts]
        coefficients = np.linalg.solve(
            rolling_gram + penalty[None, :, :], rolling_right[..., None]
        )[..., 0]
        normalized = prediction[:, lead] / input_scales[lead]
        output[:, lead] = output_scales[lead] * (
            coefficients[:, 0] + coefficients[:, 1] * normalized
        )
    return output


def calibration_scales(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.maximum(prediction.std(axis=0), 1e-12),
        np.maximum(target.std(axis=0), 1e-12),
    )


def expectation_metrics_from_arrays(
    prediction: np.ndarray, target: np.ndarray, *, target_std: float,
    cumulative_std: float, device: torch.device,
) -> dict:
    prediction_tensor = torch.from_numpy(prediction).to(device=device)
    target_tensor = torch.from_numpy(target).to(device=device)
    weights = torch.ones(prediction.shape[0], dtype=torch.float64, device=device)
    pooled = MetricAccumulator(target_std, device)
    pooled.add(prediction_tensor.flatten(), target_tensor.flatten(),
               weights[:, None].expand_as(target_tensor).flatten())
    per_lead = []
    for lead in range(prediction.shape[1]):
        metric = MetricAccumulator(target_std, device)
        metric.add(prediction_tensor[:, lead], target_tensor[:, lead], weights)
        per_lead.append(metric.result())
    cumulative = MetricAccumulator(cumulative_std, device)
    cumulative.add(prediction_tensor.sum(dim=1), target_tensor.sum(dim=1), weights)
    return {
        "expectation": pooled.result(),
        "perLeadExpectation": per_lead,
        "cumulativeExpectation": cumulative.result(),
    }


def training_statistics(
    dataset: ImmediateFeatureActivePathDataset, batch_size: int
) -> dict[str, np.ndarray | float]:
    feature_sum = np.zeros(dataset.feature_count, dtype=np.float64)
    feature_square = np.zeros_like(feature_sum)
    target_sum = 0.0
    target_square = 0.0
    cumulative_sum = 0.0
    cumulative_square = 0.0
    examples = 0
    returns = 0
    for features, targets, _weights in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0
    ):
        x = features.numpy().astype(np.float64, copy=False)
        y = targets.numpy().astype(np.float64, copy=False)
        feature_sum += x.sum(axis=0)
        feature_square += np.square(x).sum(axis=0)
        target_sum += float(y.sum())
        target_square += float(np.square(y).sum())
        cumulative = y.sum(axis=1)
        cumulative_sum += float(cumulative.sum())
        cumulative_square += float(np.square(cumulative).sum())
        examples += x.shape[0]
        returns += y.size
    feature_mean = feature_sum / examples
    feature_variance = np.maximum(
        feature_square / examples - np.square(feature_mean), 1e-20
    )
    target_mean = target_sum / returns
    target_std = math.sqrt(max(
        target_square / returns - target_mean * target_mean, 1e-20
    ))
    cumulative_mean = cumulative_sum / examples
    cumulative_std = math.sqrt(max(
        cumulative_square / examples - cumulative_mean * cumulative_mean, 1e-20
    ))
    return {
        "featureMean": feature_mean.astype(np.float32),
        "featureStd": np.sqrt(feature_variance).astype(np.float32),
        "targetStd": target_std,
        "cumulativeStd": cumulative_std,
    }


class PathMetrics:
    def __init__(self, target_std: float, cumulative_std: float, steps: int,
                 device: torch.device) -> None:
        self.pooled = MetricAccumulator(target_std, device)
        self.per_step = tuple(MetricAccumulator(target_std, device) for _ in range(steps))
        self.cumulative = MetricAccumulator(cumulative_std, device)
        self.nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.count = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, prediction: torch.Tensor, target: torch.Tensor,
            weights: torch.Tensor, log_density: torch.Tensor) -> None:
        expanded = weights[:, None].expand_as(target)
        self.pooled.add(prediction.flatten(), target.flatten(), expanded.flatten())
        for index, metric in enumerate(self.per_step):
            metric.add(prediction[:, index], target[:, index], weights)
        self.cumulative.add(prediction.sum(dim=1), target.sum(dim=1), weights)
        self.nll_sum += (expanded.double() * -log_density.double()).sum()
        self.count += expanded.double().sum()

    def result(self) -> dict:
        return {
            "negativeLogLikelihood": float(self.nll_sum / self.count),
            "expectation": self.pooled.result(),
            "perLeadExpectation": [value.result() for value in self.per_step],
            "cumulativeExpectation": self.cumulative.result(),
        }


@torch.no_grad()
def evaluate(
    model: CompressedPathReturnDensity,
    dataset: ImmediateFeatureActivePathDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    cumulative_std: float,
    device: torch.device,
    limit: int | None = None,
    collect_arrays: bool = False,
) -> dict:
    model.eval()
    result = PathMetrics(
        target_std, cumulative_std, model.return_count, device
    )
    predictions: list[np.ndarray] = []
    targets_collected: list[np.ndarray] = []
    for features, targets, weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit
    ):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        output = model(features)
        result.add(
            output.expectations, targets, weights,
            path_log_density_terms(output, targets, model),
        )
        if collect_arrays:
            predictions.append(output.expectations.cpu().numpy())
            targets_collected.append(targets.cpu().numpy())
    values = result.result()
    if collect_arrays:
        values["arrays"] = {
            "prediction": np.concatenate(predictions).astype(
                np.float64, copy=False
            ),
            "target": np.concatenate(targets_collected).astype(
                np.float64, copy=False
            ),
        }
    return values


@torch.no_grad()
def update_weight_ema(
    ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float
) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value.detach(), 1.0 - decay)
        else:
            ema[name].copy_(value)


@contextmanager
def use_state(model: torch.nn.Module, state: dict[str, torch.Tensor]):
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(original)


def checkpoint_payload(
    model: torch.nn.Module,
    ema: dict[str, torch.Tensor],
    optimizers: tuple[torch.optim.Optimizer, ...],
    *, epoch: int, global_step: int, plan_hash: str,
    runner_contract: str, best: dict,
) -> dict:
    return {
        "model": model.state_dict(), "emaModel": ema,
        "optimizers": [value.state_dict() for value in optimizers],
        "epoch": epoch, "globalStep": global_step, "best": best,
        "planSha256": plan_hash, "runnerContract": runner_contract,
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_hash = canonical_hash(plan)
    run_root = (repo / plan["runDir"]).resolve()
    if args.replace_smoke:
        status_file = run_root / "state/status.json"
        status = json.loads(status_file.read_text(encoding="utf-8")) \
            if status_file.is_file() else {}
        if status.get("stage") != "smoke-complete":
            raise ValueError("--replace-smoke may only replace a completed smoke run")
        for file in (
            run_root / "logs/training.jsonl",
            run_root / "checkpoints/last.json",
            run_root / "checkpoints/selections/validation-nll.json",
            run_root / "checkpoints/selections/validation-mse.json",
            run_root / "checkpoints/selections/validation-correlation.json",
            run_root / "state/plan.json",
            status_file,
        ):
            file.unlink(missing_ok=True)
    reporter = Reporter(run_root)
    pause_file = None if args.pause_file is None else (
        args.pause_file if args.pause_file.is_absolute() else repo / args.pause_file
    )
    try:
        training = plan["training"]
        architecture = plan["architecture"]
        architecture_contract = architecture.get("contract")
        if architecture_contract not in {
            ARCHITECTURE_CONTRACT,
            RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
        }:
            raise ValueError("compressed path architecture contract changed")
        recurrent_market = (
            architecture_contract == RECURRENT_MARKET_ARCHITECTURE_CONTRACT
        )
        runner_contract = (
            RECURRENT_MARKET_RUNNER_CONTRACT if recurrent_market
            else RUNNER_CONTRACT
        )
        if training.get("adversarialInput") is not None:
            raise ValueError("this run explicitly forbids adversarial inputs")
        sam = training.get("sam")
        sam_rho = 0.0 if sam is None else float(sam["rho"])
        if sam_rho < 0 or not math.isfinite(sam_rho):
            raise ValueError("SAM rho must be finite and non-negative")
        ema_half_life = float(training["weightEma"]["halfLifeEpochs"])
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device(training["device"])
        feature_history = int(architecture.get("inputFeatureLags", 1))
        dataset = ImmediateFeatureActivePathDataset(
            (repo / plan["datasetDir"]).resolve(),
            (repo / plan["historyDir"]).resolve(),
            int(architecture["returnCount"]),
            feature_history,
            int(plan["subset"]["examples"]),
        )
        calibration_spec = plan.get("validationCalibration")
        if calibration_spec is None \
                or calibration_spec.get("type") != "online-affine-log-per-step":
            raise ValueError(
                "compressed path training requires online per-step calibration"
            )
        calibration_window = int(calibration_spec["windowActiveReturns"])
        calibration_ridge = float(calibration_spec.get("ridge", 0.0))
        calibration_dataset = CalibrationPathDataset(
            (repo / calibration_spec["datasetDir"]).resolve(),
            int(architecture["returnCount"]),
            feature_history,
        )
        if calibration_dataset.feature_count != dataset.feature_count:
            raise ValueError("calibration and training feature counts differ")
        if calibration_dataset.logical_count("calibration") < calibration_window:
            raise ValueError("calibration corpus is shorter than its online window")
        counts = {name: dataset.logical_count(name) for name in dataset.splits}
        if counts["train"] != int(plan["subset"]["examples"]):
            raise RuntimeError(f"training clean count changed: {counts}")
        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file() and json.loads(snapshot_file.read_text(
            encoding="utf-8"
        )) != snapshot:
            raise ValueError("run directory belongs to a different plan")
        atomic_json(snapshot, snapshot_file)
        reporter.emit({
            "event": "minute-return-dataset-selected", "planId": plan["id"],
            "counts": counts, "featureCount": dataset.feature_count,
            "returnCount": int(architecture["returnCount"]),
        })
        reporter.status("computing-training-statistics", planId=plan["id"])
        stats = training_statistics(dataset, int(training["evaluationBatchSize"]))
        widths = tuple(int(value) for value in architecture["stateWidths"])
        density_file = (repo / plan["density"]["source"]).resolve()
        if recurrent_market:
            density = KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            model = RecurrentMarketPathDensity(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                density,
                market_width=int(architecture["marketWidth"]),
                state_widths=widths,
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        else:
            densities = tuple(
                KnotDensityContract.load(density_file, fit=str(width))
                for width in widths
            )
            model = CompressedPathReturnDensity(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                densities,
                market_width=int(architecture["marketWidth"]),
                state_widths=widths,
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_count = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        optimizer_parameter_groups(model)
        optimizers = build_optimizers(model, training, device)
        epochs = int(training["epochs"])
        batch_size = int(training["batchSize"])
        evaluation_batch_size = int(training["evaluationBatchSize"])
        steps_per_epoch = math.ceil(counts["train"] / batch_size)
        ema_decay = mean_teacher_ema_decay(ema_half_life, steps_per_epoch)
        ema = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        best = {
            "validation-nll": {"score": math.inf, "epoch": -1},
            "validation-mse": {"score": math.inf, "epoch": -1},
            "validation-correlation": {"score": -math.inf, "epoch": -1},
        }
        last_file = run_root / "checkpoints/last.json"
        start_epoch = 0
        global_step = 0
        if checkpoint_exists(last_file):
            saved = load_torch_checkpoint(last_file, map_location=device,
                                          weights_only=False)
            if saved.get("planSha256") != plan_hash \
                    or saved.get("runnerContract") != runner_contract:
                raise ValueError("compressed path checkpoint contract changed")
            model.load_state_dict(saved["model"])
            ema = saved["emaModel"]
            for optimizer, state in zip(
                optimizers, saved["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            best = saved["best"]
        reporter.emit({
            "event": "training-start", "planId": plan["id"],
            "startEpoch": start_epoch, "epochs": epochs,
            "parameters": parameter_count, "trainableParameters": trainable_count,
            "objective": "mean-15-step-marginal-negative-log-likelihood",
            "samRho": sam_rho, "weightEmaHalfLifeEpochs": ema_half_life,
            "weightEmaDecayPerStep": ema_decay,
            "adversarialInput": None,
            "validationCalibration": calibration_spec,
        })
        reporter.status(
            "training", planId=plan["id"], epochs=epochs,
            parameters=parameter_count, examples=counts["train"],
        )
        started = time.monotonic()
        smoke_limit = None if args.smoke_batches is None else (
            int(args.smoke_batches) * batch_size
        )
        for epoch in range(start_epoch, epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status("paused", planId=plan["id"], epoch=epoch)
                raise SystemExit(PAUSE_EXIT_CODE)
            model.train()
            loss_sum = 0.0
            example_count = 0
            batch_counter = 0
            online_expectation = MetricAccumulator(float(stats["targetStd"]), device)
            for features, targets, weights in dataset.iter_batches(
                "train", batch_size, shuffle=True, seed=seed + epoch,
                limit=smoke_limit,
            ):
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                output = model(features)
                terms = path_log_density_terms(output, targets, model)
                loss = -(terms * weights[:, None]).sum() / (
                    weights.sum() * model.return_count
                )
                loss.backward()
                if sam_rho > 0:
                    perturbations = sam_perturb_parameters(model, sam_rho)
                    try:
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        perturbed = model(features)
                        perturbed_loss = -(
                            path_log_density_terms(perturbed, targets, model)
                            * weights[:, None]
                        ).sum() / (weights.sum() * model.return_count)
                        perturbed_loss.backward()
                    finally:
                        sam_restore_parameters(perturbations)
                clip_grad_norm_(model.parameters(), float(training["gradientClip"]),
                                foreach=device.type == "cuda")
                for optimizer in optimizers:
                    optimizer.step()
                update_weight_ema(ema, model, ema_decay)
                loss_sum += float(loss.detach()) * int(targets.shape[0])
                example_count += int(targets.shape[0])
                online_expectation.add(
                    output.expectations.detach().flatten(),
                    targets.detach().flatten(),
                    weights[:, None].expand_as(targets).flatten(),
                )
                global_step += 1
                batch_counter += 1
                if batch_counter % 25 == 0:
                    reporter.status("training", planId=plan["id"], latest={
                        "epoch": epoch, "epochs": epochs,
                        "batch": batch_counter, "batches": steps_per_epoch,
                        "globalStep": global_step,
                        "onlineNegativeLogLikelihood": loss_sum / example_count,
                        "onlineTrain": online_expectation.result(),
                        "seconds": time.monotonic() - started,
                    })
            with use_state(model, ema):
                train_metrics = evaluate(
                    model, dataset, "train", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                    limit=int(training["epochTrainEvaluationExamples"]),
                )
                validation_metrics = evaluate(
                    model, dataset, "validation", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                    collect_arrays=True,
                    limit=smoke_limit,
                )
                calibration_prediction, calibration_target = (
                    collect_expectation_arrays(
                        model, calibration_dataset, "calibration",
                        batch_size=evaluation_batch_size, device=device,
                        limit=smoke_limit,
                    )
                )
            effective_calibration_window = min(
                calibration_window, calibration_prediction.shape[0]
            )
            calibration_prediction = calibration_prediction[
                -effective_calibration_window:
            ]
            calibration_target = calibration_target[-effective_calibration_window:]
            input_scales, output_scales = calibration_scales(
                calibration_prediction, calibration_target
            )
            validation_arrays = validation_metrics.pop("arrays")
            calibrated_validation_prediction = rolling_online_per_step_affine(
                calibration_prediction,
                calibration_target,
                validation_arrays["prediction"],
                validation_arrays["target"],
                window=effective_calibration_window,
                ridge=calibration_ridge,
                input_scales=input_scales,
                output_scales=output_scales,
            )
            calibrated_validation = expectation_metrics_from_arrays(
                calibrated_validation_prediction,
                validation_arrays["target"],
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]),
                device=device,
            )
            candidates = {
                "validation-nll": float(validation_metrics["negativeLogLikelihood"]),
                "validation-mse": float(
                    calibrated_validation["expectation"]["normalizedMse"]
                ),
                "validation-correlation": float(
                    calibrated_validation["expectation"]["correlation"]
                ),
            }
            for policy, score in candidates.items():
                improved = score > best[policy]["score"] \
                    if policy.endswith("correlation") \
                    else score < best[policy]["score"]
                if improved:
                    best[policy] = {"score": score, "epoch": epoch}
                    save_torch_checkpoint({
                        "model": ema, "epoch": epoch, "score": score,
                        "policy": policy, "planSha256": plan_hash,
                        "runnerContract": runner_contract,
                    }, run_root / f"checkpoints/selections/{policy}.json")
            save_torch_checkpoint(checkpoint_payload(
                model, ema, optimizers, epoch=epoch, global_step=global_step,
                plan_hash=plan_hash, runner_contract=runner_contract, best=best,
            ), last_file)
            event = {
                "event": "minute-return-epoch", "epoch": epoch,
                "epochs": epochs, "globalStep": global_step,
                "seconds": time.monotonic() - started,
                "train": train_metrics["expectation"],
                "validation": calibrated_validation["expectation"],
                "rawValidation": validation_metrics["expectation"],
                "trainDistribution": train_metrics,
                "validationDistribution": validation_metrics,
                "calibratedValidation": calibrated_validation,
                "validationCalibration": calibration_spec,
                "onlineNegativeLogLikelihood": loss_sum / example_count,
                "bestTrainScore": best["validation-nll"]["score"],
                "bestValidationNll": best["validation-nll"]["score"],
                "bestEpoch": best["validation-nll"]["epoch"],
                "parameterCount": parameter_count,
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
            if args.smoke_batches is not None:
                reporter.status("smoke-complete", planId=plan["id"], latest=event)
                return

        policies: dict[str, dict] = {}
        for policy in best:
            file = run_root / f"checkpoints/selections/{policy}.json"
            saved = load_torch_checkpoint(file, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            values = {
                "train": evaluate(
                    model, dataset, "train", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                ),
                "validation": evaluate(
                    model, dataset, "validation", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                    collect_arrays=True,
                ),
                "test": evaluate(
                    model, dataset, "test", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                    collect_arrays=True,
                ),
            }
            calibration_prediction, calibration_target = collect_expectation_arrays(
                model, calibration_dataset, "calibration",
                batch_size=evaluation_batch_size, device=device,
            )
            calibration_prediction = calibration_prediction[-calibration_window:]
            calibration_target = calibration_target[-calibration_window:]
            input_scales, output_scales = calibration_scales(
                calibration_prediction, calibration_target
            )
            validation_arrays = values["validation"].pop("arrays")
            test_arrays = values["test"].pop("arrays")
            calibrated_validation_prediction = rolling_online_per_step_affine(
                calibration_prediction, calibration_target,
                validation_arrays["prediction"], validation_arrays["target"],
                window=calibration_window, ridge=calibration_ridge,
                input_scales=input_scales, output_scales=output_scales,
            )
            calibrated_test_prediction = rolling_online_per_step_affine(
                validation_arrays["prediction"], validation_arrays["target"],
                test_arrays["prediction"], test_arrays["target"],
                window=calibration_window, ridge=calibration_ridge,
                input_scales=input_scales, output_scales=output_scales,
            )
            calibrated_validation = expectation_metrics_from_arrays(
                calibrated_validation_prediction, validation_arrays["target"],
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]), device=device,
            )
            calibrated_test = expectation_metrics_from_arrays(
                calibrated_test_prediction, test_arrays["target"],
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]), device=device,
            )
            policies[policy] = {
                "epoch": int(saved["epoch"]),
                "selectionScore": float(saved["score"]),
                "train": values["train"]["expectation"],
                "validation": calibrated_validation["expectation"],
                "test": calibrated_test["expectation"],
                "rawValidation": values["validation"]["expectation"],
                "rawTest": values["test"]["expectation"],
                "calibratedValidation": calibrated_validation,
                "calibratedTest": calibrated_test,
                "distribution": values,
                "validationCalibration": calibration_spec,
                "checkpoint": str(file.relative_to(repo)),
            }
        atomic_json({
            "contract": "compressed-path-checkpoint-selection-comparison-v1",
            "policies": policies,
        }, run_root / "state/checkpoint-selection-comparison.json")
        selected = policies["validation-nll"]
        result = {
            "version": 1, "planId": plan["id"], "planSha256": plan_hash,
            "runnerContract": runner_contract, "examples": counts["train"],
            "featureCount": dataset.feature_count,
            "parameterCount": parameter_count,
            "bestEpoch": selected["epoch"],
            "bestValidationNll": selected["selectionScore"],
            "train": selected["train"], "validation": selected["validation"],
            "test": selected["test"], "distribution": selected["distribution"],
            "checkpoint": selected["checkpoint"],
            "robustTraining": {
                "samRho": sam_rho,
                "weightEmaHalfLifeEpochs": ema_half_life,
                "weightEmaDecayPerOptimizerStep": ema_decay,
                "adversarialInput": None,
            },
            "validationCalibration": calibration_spec,
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "minute-return-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        reporter.status("paused", planId=plan.get("id", "unknown"))
        raise
    except BaseException as error:
        reporter.status("failed", planId=plan.get("id", "unknown"),
                        error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
