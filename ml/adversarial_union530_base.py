from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from differentiable_union530_features import (
    Global471Base,
    production59_features,
    reconstruct_global470,
    spread_from_top_of_book,
)
from union530_base_dataset import Union530BaseHistoryDataset


@dataclass(frozen=True)
class AdversarialUnionFeatureView:
    production59: torch.Tensor
    global470: torch.Tensor
    spread: torch.Tensor
    minute_source_rows: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.production59.shape[0])

    def rows(self, selected: np.ndarray | torch.Tensor) -> torch.Tensor:
        index = torch.as_tensor(
            selected, dtype=torch.long, device=self.production59.device
        )
        return torch.cat((
            self.production59[index],
            self.global470[self.minute_source_rows[index]],
            self.spread[index, None],
        ), dim=1)


class BaseHistoryAdversary:
    """One-step global attack in standardized primitive-history space.

    A single perturbation is shared by every overlapping example in an epoch.
    Thus two features that depend on the same historical candle always see the
    same attacked candle, and every derived channel is rerun after projection.
    """

    def __init__(
        self,
        dataset: Union530BaseHistoryDataset,
        *,
        device: torch.device,
        epsilon_rms: float,
        attack_examples: int,
        seed: int,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("base-history adversarial generation requires CUDA")
        if not math.isfinite(epsilon_rms) or epsilon_rms <= 0:
            raise ValueError("base-history epsilon RMS must be positive")
        self.dataset = dataset
        self.device = device
        self.epsilon_rms = float(epsilon_rms)
        self.attack_examples = int(attack_examples)
        self.seed = int(seed)
        self.clean = dataset.global_base(device, torch.float32)
        self.tensor_scales = {
            source: self._column_scale(value)
            for source, value in self.clean.tensors.items()
        }
        self.trade_scales = {
            source: {
                name: self._column_scale(value)
                for name, value in flow.items()
            }
            for source, flow in self.clean.trade_flows.items()
        }
        self.funding_scales = {
            source: self._column_scale(rates)
            for source, (_times, rates) in self.clean.funding_events.items()
        }

    @staticmethod
    def _column_scale(value: torch.Tensor) -> torch.Tensor:
        values = value.float()
        dimension = None if values.ndim == 1 else 0
        finite = torch.isfinite(values)
        safe = torch.where(finite, values, torch.zeros_like(values))
        count = finite.sum(dim=dimension).clamp_min(1)
        mean = safe.sum(dim=dimension) / count
        centered = torch.where(
            finite,
            values - mean if dimension is None else values - mean[None, :],
            torch.zeros_like(values),
        )
        scale = torch.sqrt(centered.square().sum(dim=dimension) / count)
        return torch.where(
            torch.isfinite(scale) & (scale > 1e-12), scale, torch.ones_like(scale)
        )

    @staticmethod
    def _tensor_mask(source: str, value: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(value)
        if value.ndim == 2:
            if source.endswith("usdm-metrics.f32") and value.shape[1] == 7:
                mask = mask & (value[:, 6:7] == 1)
                mask[:, 6] = False
            elif source.endswith("usdm-book-depth.f32") and value.shape[1] == 25:
                mask = mask & (value[:, 24:25] == 1)
                mask[:, 24] = False
            elif value.shape[1] == 10:
                mask = mask & (value[:, 9:10] == 1)
                mask[:, 9] = False
        return mask

    @staticmethod
    def _trade_mask(name: str, value: torch.Tensor) -> torch.Tensor:
        if name in {"firstAggressorSide", "lastAggressorSide"}:
            return torch.zeros_like(value, dtype=torch.bool)
        return torch.isfinite(value)

    def _leaf_base(self) -> tuple[Global471Base, list[tuple[str, str, torch.Tensor]]]:
        variables: list[tuple[str, str, torch.Tensor]] = []
        tensors: dict[str, torch.Tensor] = {}
        for source, value in self.clean.tensors.items():
            leaf = value.detach().requires_grad_(True)
            tensors[source] = leaf
            variables.append(("tensor", source, leaf))
        trade_flows: dict[str, dict[str, torch.Tensor]] = {}
        for source, flow in self.clean.trade_flows.items():
            trade_flows[source] = {}
            for name, value in flow.items():
                leaf = value.detach().requires_grad_(True)
                trade_flows[source][name] = leaf
                variables.append((f"trade:{source}", name, leaf))
        funding: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for source, (times, rates) in self.clean.funding_events.items():
            leaf = rates.detach().requires_grad_(True)
            funding[source] = (times, leaf)
            variables.append(("funding", source, leaf))
        return Global471Base(
            tensors=tensors,
            observed=self.clean.observed,
            trade_flows=trade_flows,
            funding_events=funding,
            minute_origin_rows=self.clean.minute_origin_rows,
            minute_origin_times_ms=self.clean.minute_origin_times_ms,
            minute_rows=self.clean.minute_rows,
            second_rows=self.clean.second_rows,
        ), variables

    @staticmethod
    def _project_tensor(
        source: str,
        attacked: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        if attacked.ndim == 1:
            if source.endswith("usdm-funding.json"):
                return attacked
            return torch.clamp_min(attacked, 1e-12)
        if source.endswith("usdm-metrics.f32"):
            result = attacked.clone()
            result[:, :6] = torch.clamp_min(result[:, :6], 1e-12)
            result[:, 6] = clean[:, 6]
            return result
        if source.endswith("usdm-book-depth.f32"):
            result = attacked.clone()
            result[:, :24] = torch.clamp_min(result[:, :24], 1e-12)
            result[:, 24] = clean[:, 24]
            return result
        if attacked.shape[1] in {5, 9, 10}:
            result = attacked.clone()
            open_ = torch.clamp_min(result[:, 0], 1e-12)
            close = torch.clamp_min(result[:, 3], 1e-12)
            result[:, 0] = open_
            result[:, 3] = close
            result[:, 1] = torch.maximum(
                torch.clamp_min(result[:, 1], 1e-12), torch.maximum(open_, close)
            )
            result[:, 2] = torch.minimum(
                torch.clamp_min(result[:, 2], 1e-12), torch.minimum(open_, close)
            )
            stop = 9 if attacked.shape[1] >= 9 else 5
            result[:, 4:stop] = torch.clamp_min(result[:, 4:stop], 0)
            if attacked.shape[1] == 10:
                result[:, 9] = clean[:, 9]
            return result
        return attacked

    def _attacked_base(
        self,
        leaf: Global471Base,
        variables: list[tuple[str, str, torch.Tensor]],
        gradients: tuple[torch.Tensor | None, ...],
    ) -> tuple[Global471Base, dict[str, float]]:
        prepared: list[tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        norm_square = torch.zeros((), dtype=torch.float64, device=self.device)
        active_count = 0
        for (kind, key, value), gradient in zip(variables, gradients, strict=True):
            if gradient is None:
                continue
            if kind == "tensor":
                scale = self.tensor_scales[key]
                mask = self._tensor_mask(key, value)
            elif kind.startswith("trade:"):
                source = kind.split(":", 1)[1]
                scale = self.trade_scales[source][key]
                mask = self._trade_mask(key, value)
            else:
                scale = self.funding_scales[key]
                mask = torch.isfinite(value)
            normalized_gradient = gradient.float() * scale
            active = mask & torch.isfinite(normalized_gradient) & (normalized_gradient != 0)
            selected = torch.where(active, normalized_gradient, torch.zeros_like(normalized_gradient))
            norm_square += selected.double().square().sum()
            active_count += int(active.sum().item())
            prepared.append((kind, key, value, selected, scale))
        norm = torch.sqrt(norm_square).float()
        if active_count < 1 or not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise FloatingPointError("base-history attack produced no finite gradient")
        radius = self.epsilon_rms * math.sqrt(active_count)
        multiplier = radius / norm

        tensors = dict(self.clean.tensors)
        trade = {
            source: dict(values) for source, values in self.clean.trade_flows.items()
        }
        funding = dict(self.clean.funding_events)
        maximum_normalized = 0.0
        for kind, key, _value, gradient, scale in prepared:
            normalized_delta = gradient * multiplier
            maximum_normalized = max(
                maximum_normalized, float(normalized_delta.abs().max().item())
            )
            if kind == "tensor":
                clean = self.clean.tensors[key]
                tensors[key] = self._project_tensor(
                    key, clean + normalized_delta * scale, clean
                ).detach()
            elif kind.startswith("trade:"):
                source = kind.split(":", 1)[1]
                clean = self.clean.trade_flows[source][key]
                attacked = clean + normalized_delta * scale
                trade[source][key] = torch.clamp_min(attacked, 0).detach()
            else:
                times, clean = self.clean.funding_events[key]
                funding[key] = (times, (clean + normalized_delta * scale).detach())
        attacked = Global471Base(
            tensors=tensors,
            observed=self.clean.observed,
            trade_flows=trade,
            funding_events=funding,
            minute_origin_rows=self.clean.minute_origin_rows,
            minute_origin_times_ms=self.clean.minute_origin_times_ms,
            minute_rows=self.clean.minute_rows,
            second_rows=self.clean.second_rows,
        )
        return attacked, {
            "activeBaseCoordinates": active_count,
            "normalizedDeltaRms": self.epsilon_rms,
            "normalizedDeltaMaximum": maximum_normalized,
        }

    def generate(
        self,
        model: torch.nn.Module,
        train_physical_rows: np.ndarray,
        train_targets: np.ndarray,
        *,
        epoch: int,
        objective: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[AdversarialUnionFeatureView, dict[str, float]]:
        count = int(train_physical_rows.size)
        attack_count = min(self.attack_examples, count)
        rng = np.random.default_rng(self.seed + int(epoch))
        logical = np.sort(rng.choice(count, size=attack_count, replace=False))
        physical = np.asarray(train_physical_rows[logical], dtype=np.int64)
        leaf, variables = self._leaf_base()
        production = production59_features(
            self.dataset.production_base_from_global(leaf),
            torch.as_tensor(
                self.dataset.second_indices(physical),
                dtype=torch.long,
                device=self.device,
            ),
        )
        global470 = reconstruct_global470(leaf, self.dataset.global_specs)
        minute_rows = torch.as_tensor(
            np.asarray(self.dataset.minute_source_rows[physical], dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        top = torch.as_tensor(
            np.asarray(self.dataset.top_of_book[physical], dtype=np.float32).copy(),
            dtype=torch.float32,
            device=self.device,
        ).requires_grad_(True)
        features = torch.cat((
            production,
            global470[minute_rows],
            spread_from_top_of_book(top)[:, None],
        ), dim=1)
        targets = torch.as_tensor(
            train_targets[logical, None].copy(),
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.ones(attack_count, dtype=torch.float32, device=self.device)
        loss = objective(features, targets, weights)
        attack_objective = float(loss.detach().item())
        feature_gradient, = torch.autograd.grad(
            loss, (features,), retain_graph=True, allow_unused=False
        )
        feature_gradient_finite = int(torch.isfinite(feature_gradient).sum().item())
        feature_gradient_count = int(feature_gradient.numel())
        all_variables = [value for _kind, _key, value in variables] + [top]
        all_gradients = torch.autograd.grad(
            loss, all_variables, allow_unused=True, only_inputs=True
        )
        try:
            attacked, metrics = self._attacked_base(
                leaf, variables, all_gradients[:-1]
            )
        except FloatingPointError as error:
            raise FloatingPointError(
                f"{error}; objective={attack_objective}; finite feature gradients="
                f"{feature_gradient_finite}/{feature_gradient_count}"
            ) from error
        metrics["finiteFeatureGradientCoordinates"] = feature_gradient_finite
        metrics["featureGradientCoordinates"] = feature_gradient_count
        top_gradient = all_gradients[-1]
        if top_gradient is None or not bool(torch.isfinite(top_gradient).all()):
            raise FloatingPointError("top-of-book attack gradient is invalid")
        top_scale = torch.std(top.detach(), dim=0, unbiased=False).clamp_min(1e-12)
        normalized_top_gradient = top_gradient * top_scale
        top_norm = torch.linalg.vector_norm(normalized_top_gradient.float())
        top_radius = self.epsilon_rms * math.sqrt(top.numel())
        top_delta = normalized_top_gradient * (top_radius / top_norm.clamp_min(1e-12))
        attacked_top = top.detach() + top_delta * top_scale
        attacked_top = torch.sort(torch.clamp_min(attacked_top, 1e-12), dim=1).values

        del production, global470, features, loss, leaf
        torch.cuda.empty_cache()
        all_physical = np.asarray(train_physical_rows, dtype=np.int64)
        with torch.no_grad():
            attacked_production = production59_features(
                self.dataset.production_base_from_global(attacked),
                torch.as_tensor(
                    self.dataset.second_indices(all_physical),
                    dtype=torch.long,
                    device=self.device,
                ),
            ).float()
            attacked_global = reconstruct_global470(
                attacked, self.dataset.global_specs
            ).float()
            top_all = torch.as_tensor(
                np.asarray(
                    self.dataset.top_of_book[all_physical], dtype=np.float32
                ).copy(),
                dtype=torch.float32,
                device=self.device,
            )
            spread = spread_from_top_of_book(top_all)
            spread[torch.as_tensor(logical, dtype=torch.long, device=self.device)] = (
                spread_from_top_of_book(attacked_top)
            )
            source_rows = torch.as_tensor(
                np.asarray(
                    self.dataset.minute_source_rows[all_physical], dtype=np.int64
                ).copy(),
                dtype=torch.long,
                device=self.device,
            )
        metrics["attackExamples"] = attack_count
        metrics["attackObjective"] = attack_objective
        return AdversarialUnionFeatureView(
            production59=attacked_production,
            global470=attacked_global,
            spread=spread,
            minute_source_rows=source_rows,
        ), metrics
