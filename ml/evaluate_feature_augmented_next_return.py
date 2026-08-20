from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_autoregressive_density_episodes import autoregressive_episode_metrics
from normalized_glu_return_density import NormalizedGluReturnDensity
from normalized_glu_next_return import NormalizedGluNextReturn
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint, read_candle_column
from train_feature_augmented_next_return import FeatureMatrixDataset
from train_normalized_glu_next_return import MetricAccumulator, atomic_json


CONTRACT = "feature-augmented-deterministic-active-return-evaluation-v1"
EPISODE_SECONDS = 900
MAXIMUM_EPISODES = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill multi-step and cleaned 15m metrics for a feature GLU."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


FeatureModel = NormalizedGluNextReturn | NormalizedGluReturnDensity


def load_model(
    repo: Path, plan: dict, checkpoint: dict, device: torch.device
) -> FeatureModel:
    state = checkpoint["model"]
    architecture = plan["architecture"]
    common = {
        "widths": tuple(int(value) for value in architecture["widths"]),
        "dropout": 0,
        "dropout_rate": 0,
        "initial_radius": float(architecture["initialRadius"]),
        "minimum_radius": float(architecture["minimumRadius"]),
        "learnable_centering": bool(architecture["learnableCentering"]),
    }
    if "density" in plan:
        density_plan = plan["density"]
        density = KnotDensityContract.load(
            (repo / density_plan["source"]).resolve(),
            fit=str(density_plan["fit"]),
        )
        model = NormalizedGluReturnDensity(
            state["feature_mean"], state["feature_std"], density,
            return_count=1, **common,
        )
    else:
        model = NormalizedGluNextReturn(
            state["feature_mean"], state["feature_std"],
            state["target_mean"], state["target_std"], **common,
        )
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def expected_return(model: FeatureModel, features: torch.Tensor) -> torch.Tensor:
    if isinstance(model, NormalizedGluReturnDensity):
        expectation, _mode = model.point_predictions_from_log_masses(
            model.log_component_masses(features), include_mode=False
        )
        return expectation
    return model(features)


class FeatureUpdater:
    def __init__(self, manifest: dict) -> None:
        self.index = {
            value["id"]: index for index, value in enumerate(manifest["features"])
        }
        self.history_seconds = int(manifest.get("featureHistorySeconds", 1))
        self.temporal_channel_count = int(
            manifest.get("temporalChannelCount") or 0
        )
        if self.history_seconds > 1 and (
            self.temporal_channel_count * self.history_seconds
            != len(manifest["features"])
        ):
            raise ValueError("temporal feature manifest dimensions are inconsistent")

    def _set(
        self, features: torch.Tensor, identifier: str, values: torch.Tensor
    ) -> None:
        features[:, self.index[identifier]] = values

    @staticmethod
    def _moments(
        history: torch.Tensor, window: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = history[:, -min(window, history.shape[1]):]
        mean_absolute = values.abs().mean(dim=1)
        rms = values.square().mean(dim=1).sqrt()
        active = (values != 0).float().mean(dim=1)
        return mean_absolute, rms, active

    @staticmethod
    def _indicator_values(history: torch.Tensor) -> tuple[torch.Tensor, ...]:
        log_price = torch.cumsum(history[:, -120:], dim=1)
        log_price = log_price - log_price[:, -1:]
        price = torch.exp(log_price)
        changes = price[:, 1:] - price[:, :-1]
        gain = torch.zeros(price.shape[0], device=price.device)
        loss = torch.zeros_like(gain)
        ema2 = price[:, 0].clone()
        ema8 = price[:, 0].clone()
        ema2_history = [ema2]
        ema8_history = [ema8]
        for position in range(1, price.shape[1]):
            change = changes[:, position - 1]
            gain += 0.5 * (change.clamp_min(0) - gain)
            loss += 0.5 * ((-change).clamp_min(0) - loss)
            ema2 += (2 / 3) * (price[:, position] - ema2)
            ema8 += (2 / 9) * (price[:, position] - ema8)
            ema2_history.append(ema2)
            ema8_history.append(ema8)
        rsi = torch.where(
            loss > 0,
            100 - 100 / (1 + gain / loss.clamp_min(1e-20)),
            torch.where(gain > 0, 100.0, 50.0),
        )
        ema2_values = torch.stack(ema2_history, dim=1)
        ema8_values = torch.stack(ema8_history, dim=1)
        slope2 = 10_000 * torch.log(ema2_values[:, -1] / ema2_values[:, -2])
        prior_slope2 = 10_000 * torch.log(ema2_values[:, -2] / ema2_values[:, -3])
        acceleration2 = slope2 - prior_slope2
        slope8 = 10_000 * torch.log(ema8_values[:, -1] / ema8_values[:, -9]) / 8
        return (rsi - 50) / 50, acceleration2, slope8

    def update(
        self,
        features: torch.Tensor,
        history: torch.Tensor,
        prediction: torch.Tensor,
        active: torch.Tensor,
        origin_ms: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.history_seconds > 1:
            temporal = features.reshape(
                features.shape[0], self.temporal_channel_count,
                self.history_seconds,
            )
            result = torch.cat(
                (temporal[:, :, 1:], temporal[:, :, -1:]), dim=2
            ).reshape_as(features)
        else:
            result = features.clone()
        next_history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        history = torch.where(active[:, None], next_history, history)
        raw = history[:, -120:]
        result[:, :120] = torch.where(active[:, None], raw, result[:, :120])
        _mean5, rms5, _active5 = self._moments(history, 5)
        _mean15, rms15, _active15 = self._moments(history, 15)
        _mean60, rms60, _active60 = self._moments(history, 60)
        rms60_safe = rms60.clamp_min(1e-8)
        self._set(result, "return-latest-vol-normalized", raw[:, -1] / rms60_safe)
        self._set(result, "return-lag-1-vol-normalized", raw[:, -2] / rms60_safe)
        self._set(result, "return-latest-is-zero", (raw[:, -1] == 0).float())
        self._set(result, "return-lag-1-is-zero", (raw[:, -2] == 0).float())
        self._set(result, "active-fraction-10s", (raw[:, -10:] != 0).float().mean(dim=1))
        self._set(result, "active-fraction-60s", (raw[:, -60:] != 0).float().mean(dim=1))
        zero_age = torch.zeros(raw.shape[0], device=raw.device)
        still_zero = torch.ones(raw.shape[0], dtype=torch.bool, device=raw.device)
        for lag in range(min(120, raw.shape[1])):
            still_zero &= raw[:, -1 - lag] == 0
            zero_age += still_zero.float()
        self._set(result, "zero-run-age-log", torch.log1p(zero_age))

        available_windows = [
            value for value in (5, 15, 60, 300, 900, 1_800, 3_600, 14_400)
            if value <= history.shape[1]
        ]
        log_rms = {
            window: 0.5 * torch.log(
                1e-16 + history[:, -window:].square().mean(dim=1)
            )
            for window in available_windows
        }
        if 3_600 in log_rms:
            anchor = log_rms[3_600]
            self._set(result, "log-rms-anchor-1h", anchor)
            for window in available_windows:
                if window != 3_600:
                    self._set(
                        result, f"log-rms-{window}s-minus-1h",
                        log_rms[window] - anchor,
                    )
        else:
            anchor = result[:, self.index["log-rms-anchor-1h"]]
            for window in (5, 15, 60):
                self._set(
                    result, f"log-rms-{window}s-minus-1h",
                    log_rms[window] - anchor,
                )
        for window in (5, 15, 60, 300, 900, 3_600):
            if window <= history.shape[1]:
                mean_absolute = history[:, -window:].abs().mean(dim=1)
                self._set(
                    result, f"log-mean-absolute-return-{window}s",
                    torch.log(1e-12 + mean_absolute),
                )

        rsi, acceleration, slope = self._indicator_values(raw)
        self._set(result, "rsi-2-mapped", rsi)
        self._set(
            result, "ema-acceleration-2s-1s-vol-normalized",
            acceleration / (rms5 * 10_000).clamp_min(1e-4),
        )
        self._set(
            result, "ema-slope-8s-8s-vol-normalized",
            slope / (rms15 * 10_000).clamp_min(1e-4),
        )
        last16 = raw[:, -16:]
        square16 = last16.square().sum(dim=1)
        sum16 = last16.sum(dim=1)
        absolute16 = last16.abs().sum(dim=1)
        self._set(
            result, "haar-adjacent-contrast-16s",
            (last16[:, -2] - last16[:, -1])
            / (math.sqrt(2) * square16.sqrt()).clamp_min(1e-12),
        )
        self._set(
            result, "efficiency-absolute-16s",
            torch.where(absolute16 > 0, sum16 / absolute16, 0),
        )
        self._set(
            result, "efficiency-rms-16s",
            torch.where(square16 > 0, sum16 / square16.sqrt(), 0),
        )

        for identifier in (
            "completed-1h-volume-age-log-seconds",
            "futures-age-log-seconds",
            "eth-volatility-age-log-seconds",
        ):
            age = torch.expm1(result[:, self.index[identifier]]).clamp_min(0)
            self._set(result, identifier, torch.log1p(age + active.float()))
        timestamp = origin_ms.double() + (step + 1) * 1_000
        seconds = torch.remainder(timestamp / 1_000, 60)
        minutes = torch.remainder(timestamp / 60_000, 60)
        hours = torch.remainder(timestamp / 3_600_000, 24)
        days = torch.remainder(torch.floor(timestamp / 86_400_000) + 4, 7)
        for name, value, period in (
            ("second", seconds, 60),
            ("minute", minutes, 60),
            ("hour", hours, 24),
            ("day-of-week", days, 7),
        ):
            angle = 2 * math.pi * value / period
            self._set(result, f"calendar-{name}-sin", torch.sin(angle).float())
            self._set(result, f"calendar-{name}-cos", torch.cos(angle).float())
        return torch.where(active[:, None], result, features), history


@torch.no_grad()
def rollout(
    model: FeatureModel,
    updater: FeatureUpdater,
    features: torch.Tensor,
    history: torch.Tensor,
    lengths: torch.Tensor,
    origin_ms: torch.Tensor,
) -> torch.Tensor:
    predictions = torch.zeros(
        (features.shape[0], int(lengths.max())), device=features.device
    )
    for step in range(predictions.shape[1]):
        active = step < lengths
        prediction = expected_return(model, features)
        predictions[:, step] = torch.where(active, prediction, 0)
        features, history = updater.update(
            features, history, prediction, active, origin_ms, step
        )
    return predictions


def metric_result(
    prediction: torch.Tensor, target: torch.Tensor, target_std: float
) -> dict:
    accumulator = MetricAccumulator(target_std, prediction.device)
    accumulator.add(
        prediction.reshape(-1), target.reshape(-1),
        torch.ones(target.numel(), device=target.device),
    )
    return accumulator.result()


@torch.no_grad()
def evaluate_three_steps(
    model: FeatureModel,
    updater: FeatureUpdater,
    dataset: FeatureMatrixDataset,
    split: str,
    device: torch.device,
    target_std: float,
    batch_size: int = 4_096,
) -> dict:
    values = dataset.splits[split]
    count = values.count - 2
    accumulators = [
        MetricAccumulator(target_std, device)
        for _ in range(3)
    ]
    pooled = MetricAccumulator(target_std, device)
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        feature = torch.from_numpy(
            np.asarray(values.features[start:stop], dtype=np.float32).copy()
        ).to(device)
        targets = torch.from_numpy(np.stack(tuple(
            np.asarray(values.targets[start + lead:stop + lead], dtype=np.float32)
            for lead in range(3)
        ), axis=1)).to(device)
        history = feature[:, :120].clone()
        lengths = torch.full((feature.shape[0],), 3, device=device, dtype=torch.long)
        times = torch.from_numpy(
            np.asarray(values.times[start:stop], dtype=np.float64).copy()
        ).to(device)
        prediction = rollout(model, updater, feature, history, lengths, times)
        weights = torch.ones(feature.shape[0], device=device)
        for lead, accumulator in enumerate(accumulators):
            accumulator.add(prediction[:, lead], targets[:, lead], weights)
        pooled.add(prediction.reshape(-1), targets.reshape(-1), torch.ones(
            targets.numel(), device=device
        ))
    return {
        "contract": "deterministic-autoregressive-three-active-returns-v1",
        "expectation": pooled.result(),
        "perLeadExpectation": [value.result() for value in accumulators],
    }


class CloseCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.values: dict[str, np.ndarray] = {}

    def load(self, day: str) -> np.ndarray:
        if day not in self.values:
            self.values[day] = read_candle_column(self.root / f"{day}.json", "close")
        return self.values[day]

    def history(self, boundary_ms: int, count: int) -> np.ndarray:
        final = datetime.fromtimestamp((boundary_ms - 1) / 1_000, tz=timezone.utc)
        day = final.date()
        row = final.hour * 3_600 + final.minute * 60 + final.second
        previous = self.load((day - timedelta(days=1)).isoformat())
        current = self.load(day.isoformat())
        combined = np.concatenate((previous, current)).astype(np.float64, copy=False)
        end = 86_400 + row
        closes = combined[end - count:end + 1]
        if closes.size != count + 1:
            raise RuntimeError("candle history does not cover episode origin")
        return np.diff(np.log(closes)).astype(np.float32)


@torch.no_grad()
def evaluate_episodes(
    model: FeatureModel,
    updater: FeatureUpdater,
    dataset: FeatureMatrixDataset,
    split: str,
    history_root: Path,
    device: torch.device,
) -> dict:
    values = dataset.splits[split]
    target_times = np.asarray(values.times, dtype=np.int64)
    buckets = target_times // (EPISODE_SECONDS * 1_000)
    starts = np.flatnonzero(np.r_[True, buckets[1:] != buckets[:-1]])
    stops = np.r_[starts[1:], values.count]
    selected = [
        (int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)
        if stop - start >= 2
    ]
    if len(selected) > MAXIMUM_EPISODES:
        indexes = np.linspace(0, len(selected) - 1, MAXIMUM_EPISODES, dtype=np.int64)
        selected = [selected[int(index)] for index in indexes]
    lengths_np = np.asarray([stop - start for start, stop in selected], dtype=np.int64)
    maximum = int(lengths_np.max())
    features_np = np.stack([
        np.asarray(values.features[start], dtype=np.float32) for start, _stop in selected
    ])
    targets_np = np.zeros((len(selected), maximum), dtype=np.float32)
    for index, (start, stop) in enumerate(selected):
        targets_np[index, :stop - start] = values.targets[start:stop]
    cache = CloseCache(history_root)
    histories_np = np.stack([
        cache.history(int(values.times[start]), 14_400) for start, _stop in selected
    ])
    np.testing.assert_allclose(
        histories_np[:, -120:], features_np[:, :120], rtol=1e-5, atol=1e-10
    )
    features = torch.from_numpy(features_np).to(device)
    targets = torch.from_numpy(targets_np).to(device)
    histories = torch.from_numpy(histories_np).to(device)
    lengths = torch.from_numpy(lengths_np).to(device)
    times = torch.from_numpy(np.asarray([
        values.times[start] for start, _stop in selected
    ], dtype=np.float64)).to(device)
    prediction = rollout(model, updater, features, histories, lengths, times)
    return autoregressive_episode_metrics(
        prediction.cpu(), targets.cpu(), source_episode_seconds=EPISODE_SECONDS
    )


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = (repo / args.plan).resolve() if not args.plan.is_absolute() else args.plan
    plan = json.loads(plan_file.read_text("utf-8"))
    dataset = FeatureMatrixDataset((repo / plan["datasetDir"]).resolve())
    target_std = float(np.asarray(
        dataset.splits["train"].targets, dtype=np.float64
    ).std())
    updater = FeatureUpdater(dataset.manifest)
    run_root = (repo / plan["runDir"]).resolve()
    comparison_file = run_root / "state/checkpoint-selection-comparison.json"
    comparison = json.loads(comparison_file.read_text("utf-8"))
    status_file = run_root / "state/feature-episode-evaluation-status.json"
    device = torch.device(args.device)
    history_root = repo / "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"
    for policy_index, (policy, value) in enumerate(comparison["policies"].items()):
        atomic_json({
            "stage": "evaluating-feature-augmented-paths",
            "policy": policy,
            "completedPolicies": policy_index,
            "totalPolicies": len(comparison["policies"]),
        }, status_file)
        checkpoint = load_torch_checkpoint(
            repo / value["checkpoint"], map_location=device, weights_only=False
        )
        model = load_model(repo, plan, checkpoint, device)
        multi_step = {
            split: evaluate_three_steps(
                model, updater, dataset, split, device, target_std
            )
            for split in ("train", "validation", "test")
        }
        if isinstance(model, NormalizedGluReturnDensity):
            for split, metrics in multi_step.items():
                distribution = value["distribution"][split]
                distribution["perLeadExpectation"] = metrics[
                    "perLeadExpectation"
                ]
                distribution["autoregressivePooledExpectation"] = metrics[
                    "expectation"
                ]
        else:
            value["distribution"] = multi_step
        value["autoregressiveEpisodes"] = {
            split: evaluate_episodes(
                model, updater, dataset, split, history_root, device
            )
            for split in ("validation", "test")
        }
        atomic_json(comparison, comparison_file)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json({
        "contract": CONTRACT,
        "stage": "complete",
        "completedPolicies": len(comparison["policies"]),
        "totalPolicies": len(comparison["policies"]),
        "note": (
            "Multi-step expectation and cleaned 15-minute episode metrics are "
            "complete. Density-path sampling diagnostics are a separate pass."
            if "density" in plan else
            "Sobol estimator variance and realized-path density are unavailable "
            "because this model has a deterministic scalar output."
        ),
    }, status_file)


if __name__ == "__main__":
    main()
