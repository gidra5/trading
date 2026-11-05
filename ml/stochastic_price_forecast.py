"""Causal all-tick price law for the finite-scenario CRRA policy.

The zero-return atom is explicit. Continuous nonzero returns use the existing
normalized-GLU/triangular-knot density. Hypothetical rollouts update return lags
and freeze the other observed features; this is a forecast-model assumption,
not a claim that future exogenous features are deterministic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from normalized_glu_next_return import NormalizedGluNextReturn
from return_knot_density import (
    ReturnTransform, component_log_masses, component_return_means,
    interpolated_log_density_unit, inverse_unit_to_returns,
    transform_returns_to_unit, triangular_basis_areas,
)

CONTRACT = "causal-production178-zero-atom-knot-density-v1"
HISTORY = 120


@dataclass(frozen=True)
class TimelineSplit:
    timeline: np.ndarray
    origins: np.ndarray
    axis_start_ms: int

    def features(self, origins: np.ndarray | None = None) -> np.ndarray:
        selected = self.origins if origins is None else np.asarray(origins)
        lag_rows = selected[:, None] + np.arange(1 - HISTORY, 1)[None, :]
        return np.concatenate((self.timeline[lag_rows, 0], self.timeline[selected, 1:]), axis=1)

    def targets(self, origins: np.ndarray | None = None) -> np.ndarray:
        selected = self.origins if origins is None else np.asarray(origins)
        return np.asarray(self.timeline[selected + 1, 0])


def load_timeline_split(root: Path, split: str) -> TimelineSplit:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["storageLayout"] != "temporal-channel-timeline-v1" or manifest["temporalChannelCount"] != 59:
        raise ValueError("forecast requires the causal production59 timeline")
    files = manifest["files"][split]
    rows = manifest["timelineRowsBySplit"][split]
    timeline = np.memmap(root / files["timelineFeatures"], dtype="<f4", mode="r", shape=(rows, 59))
    old_origins = np.fromfile(root / files["origins"], dtype="<i4")
    times = np.fromfile(root / files["times"], dtype="<f8")
    offsets = times - old_origins * 1000
    if not offsets.size or np.any(offsets != offsets[0]):
        raise ValueError("timeline timestamp mapping is not contiguous")
    axis_start_ms = int(offsets[0])
    split_start_ms = int(np.datetime64(manifest["splits"][split]["start"].replace("Z", ""), "ms").astype("int64"))
    start = max(HISTORY - 1, (split_start_ms - axis_start_ms + 999) // 1000)
    origins = np.arange(start, rows - 1, dtype=np.int64)
    if not origins.size or not np.isfinite(timeline).all():
        raise ValueError("timeline is empty or non-finite")
    return TimelineSplit(timeline, origins, axis_start_ms)


def fit_training_basis(targets: np.ndarray, knot_count: int = 256) -> dict:
    """Fit every data-dependent coordinate using training targets only."""
    values = np.asarray(targets, dtype=np.float64)
    active = values[values != 0]
    if active.size < knot_count or not np.isfinite(values).all():
        raise ValueError("insufficient finite active training returns")
    transform = ReturnTransform(alpha=4.0, location_bps=0.0,
                                scale_bps=max(float(np.std(active)) * 1e4 * 2, 1e-3))
    unit, _ = transform_returns_to_unit(torch.from_numpy(active), transform)
    # A mixture of regular and empirical coordinates resolves the tick peak
    # while retaining full-support tails. De-duplication is intentional.
    quantiles = np.quantile(unit.numpy(), np.linspace(0, 1, knot_count // 2))
    knots = np.unique(np.concatenate(([0.0, 1.0], np.linspace(0, 1, knot_count // 2), quantiles))).astype(np.float32)
    knots = np.unique(knots).astype(np.float64)
    areas = triangular_basis_areas(torch.from_numpy(knots)).numpy()
    # Histogram initialization is trained further by proper mixed-law NLL.
    boundaries = (knots[1:] + knots[:-1]) / 2
    assignments = np.searchsorted(boundaries, unit.numpy())
    masses = np.bincount(assignments, minlength=len(knots)).astype(np.float64) + 0.5
    masses /= masses.sum()
    return {"transform": asdict(transform), "knots": knots.tolist(),
            "priorMasses": masses.tolist(), "priorZeroProbability": float(np.mean(values == 0)),
            "basisAreas": areas.tolist(), "fitSplit": "train", "observations": len(values)}


class StochasticPriceForecast(NormalizedGluNextReturn):
    def __init__(self, feature_mean: Tensor, feature_std: Tensor, basis: dict,
                 widths: tuple[int, ...] = (512, 512, 512, 512)) -> None:
        self.basis = basis
        self.transform = ReturnTransform(**basis["transform"])
        count = len(basis["knots"])
        super().__init__(feature_mean, feature_std, torch.zeros(count + 1), torch.ones(count + 1),
                         widths=widths, learnable_centering=False, dropout=0, dropout_rate=0,
                         initial_radius=0.0031622776601683794, minimum_radius=1e-4)
        self.register_buffer("knots", torch.tensor(basis["knots"], dtype=torch.float32))
        self.register_buffer("areas", triangular_basis_areas(self.knots))
        means = component_return_means(np.asarray(basis["knots"]), self.transform)
        self.register_buffer("component_means", torch.tensor(means, dtype=torch.float32))
        p0 = min(max(basis["priorZeroProbability"], 1e-6), 1 - 1e-6)
        with torch.no_grad():
            self.output.bias[0] = np.log(p0 / (1 - p0))
            self.output.bias[1:] = torch.log(torch.tensor(basis["priorMasses"]) / self.areas)

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_standardized(features)

    def log_prob_from_logits(self, logits: Tensor, targets: Tensor) -> Tensor:
        unit, jacobian = transform_returns_to_unit(targets, self.transform)
        log_masses = component_log_masses(logits[:, 1:], self.areas)
        active_log_density = interpolated_log_density_unit(log_masses, unit, self.knots, self.areas) + jacobian
        return torch.where(targets == 0, F.logsigmoid(logits[:, 0]),
                           F.logsigmoid(-logits[:, 0]) + active_log_density)

    def expectation_from_logits(self, logits: Tensor) -> Tensor:
        masses = component_log_masses(logits[:, 1:], self.areas).exp()
        return torch.sigmoid(-logits[:, 0]) * (masses @ self.component_means)

    @torch.no_grad()
    def probabilities(self, features: Tensor) -> tuple[np.ndarray, np.ndarray]:
        logits = self(features)
        return (torch.sigmoid(logits[:, 0]).cpu().numpy().astype(np.float64),
                component_log_masses(logits[:, 1:], self.areas).exp().cpu().numpy().astype(np.float64))

    def quantile_units(self, masses: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        knots = np.asarray(self.basis["knots"])
        areas = np.asarray(self.basis["basisAreas"])
        masses = masses / masses.sum(axis=1, keepdims=True)
        heights = masses / areas[None, :]
        interval_mass = (heights[:, :-1] + heights[:, 1:]) * np.diff(knots)[None, :] / 2
        cdf = np.concatenate((np.zeros((len(masses), 1)), np.cumsum(interval_mass, axis=1)), axis=1)
        cdf /= cdf[:, -1:]
        result = np.empty((len(masses), len(quantiles)))
        for branch, quantile in enumerate(quantiles):
            interval = np.minimum(np.sum(cdf[:, 1:] < quantile, axis=1), len(knots) - 2)
            row = np.arange(len(masses))
            local = quantile - cdf[row, interval]
            left_height = heights[row, interval]
            slope = (heights[row, interval + 1] - left_height) / (knots[interval + 1] - knots[interval])
            radical = np.sqrt(np.maximum(0, left_height ** 2 + 2 * slope * local))
            distance = 2 * local / np.maximum(left_height + radical, np.finfo(float).tiny)
            unit = np.clip(knots[interval] + distance, knots[interval], knots[interval + 1])
            result[:, branch] = unit
        return result

    def quantile_returns(self, masses: np.ndarray, branches: int) -> np.ndarray:
        if branches < 1:
            raise ValueError("at least one active branch is required")
        return inverse_unit_to_returns(self.quantile_units(masses, (np.arange(branches) + 0.5) / branches), self.transform)

    def quantile_bin_means(self, masses: np.ndarray, branches: int) -> np.ndarray:
        """Equal-mass bin means retain tail contributions and the first moment.

        Integration uses 64-point Gauss-Legendre per knot interval, matching
        the existing density mean routine. This is numerical quadrature,
        not exact preservation of nonlinear terminal utility or tail risk.
        """
        if branches < 1:
            raise ValueError("at least one active branch is required")
        masses = masses / masses.sum(axis=1, keepdims=True)
        knots = np.asarray(self.basis["knots"])
        heights = masses / np.asarray(self.basis["basisAreas"])[None, :]
        nodes, weights = np.polynomial.legendre.leggauss(64)
        widths = np.diff(knots)
        unit = (knots[:-1, None] + knots[1:, None]) / 2 + widths[:, None] * nodes[None, :] / 2
        returns = inverse_unit_to_returns(unit, self.transform)
        fractions = (nodes + 1) / 2
        left_integral = np.sum(returns * (1 - fractions)[None, :] * weights[None, :], axis=1) * widths / 2
        right_integral = np.sum(returns * fractions[None, :] * weights[None, :], axis=1) * widths / 2
        interval_moment = heights[:, :-1] * left_integral + heights[:, 1:] * right_integral
        cumulative = np.concatenate((np.zeros((len(masses), 1)), np.cumsum(interval_moment, axis=1)), axis=1)
        cuts = self.quantile_units(masses, np.arange(1, branches) / branches)
        moments = np.zeros((len(masses), branches + 1))
        moments[:, -1] = cumulative[:, -1]
        row = np.arange(len(masses))
        for boundary in range(branches - 1):
            upper = cuts[:, boundary]
            interval = np.minimum(np.searchsorted(knots, upper, side="right") - 1, len(knots) - 2)
            left = knots[interval]
            half = (upper - left) / 2
            partial_unit = (upper[:, None] + left[:, None]) / 2 + half[:, None] * nodes[None, :]
            partial_return = inverse_unit_to_returns(partial_unit, self.transform)
            frac = (partial_unit - left[:, None]) / widths[interval, None]
            density = heights[row, interval, None] * (1 - frac) + heights[row, interval + 1, None] * frac
            partial = np.sum(partial_return * density * weights[None, :], axis=1) * half
            moments[:, boundary + 1] = cumulative[row, interval] + partial
        return np.diff(moments, axis=1) * branches

    @torch.no_grad()
    def scenarios(self, features: Tensor, branches: int = 3) -> tuple[np.ndarray, np.ndarray]:
        p0, masses = self.probabilities(features)
        returns = np.concatenate((np.zeros((len(p0), 1)), self.quantile_bin_means(masses, branches)), axis=1)
        probabilities = np.concatenate((p0[:, None], np.repeat(((1 - p0) / branches)[:, None], branches, axis=1)), axis=1)
        return returns, probabilities


def append_hypothetical_return(features: Tensor, returns: Tensor) -> Tensor:
    """Update all 120 price lags; keep 58 observed non-price features frozen."""
    if features.shape[-1] != 178 or returns.shape != features.shape[:-1]:
        raise ValueError("hypothetical feature shape mismatch")
    return torch.cat((features[..., 1:HISTORY], returns[..., None], features[..., HISTORY:]), dim=-1)


@torch.no_grad()
def two_tick_scenarios(model: StochasticPriceForecast, features: Tensor, branches: int = 3
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared historical-export and online-worker scenario construction."""
    first_returns, first_probabilities = model.scenarios(features, branches)
    branch_count = branches + 1
    repeated = features.repeat_interleave(branch_count, dim=0)
    hypothetical = torch.from_numpy(first_returns.reshape(-1).astype(np.float32)).to(features.device)
    child_features = append_hypothetical_return(repeated, hypothetical)
    second_returns, second_probabilities = model.scenarios(child_features, branches)
    return (first_returns, first_probabilities,
            second_returns.reshape(len(features), branch_count, branch_count),
            second_probabilities.reshape(len(features), branch_count, branch_count))


def price_tree(observed_price: float, first_returns: np.ndarray, first_probabilities: np.ndarray,
               second_returns: np.ndarray, second_probabilities: np.ndarray) -> dict:
    """Convert one causal two-tick return tree into the public price-tree contract."""
    if not np.isfinite(observed_price) or observed_price <= 0:
        raise ValueError("observedPrice must be finite and positive")
    children = []
    for first in range(len(first_returns)):
        price = observed_price * float(np.exp(first_returns[first]))
        grandchildren = [{"probability": float(second_probabilities[first, second]),
                          "price": price * float(np.exp(second_returns[first, second]))}
                         for second in range(len(first_returns))]
        children.append({"probability": float(first_probabilities[first]), "price": price,
                         "children": grandchildren})
    if any(not np.isfinite(child["price"]) or child["price"] <= 0
           for child in children + [leaf for branch in children for leaf in branch["children"]]):
        raise ValueError("forecast price is outside the finite positive domain")
    return {"children": children}


def load_forecast(file: Path, device: torch.device) -> tuple[StochasticPriceForecast, dict]:
    payload = torch.load(file, map_location="cpu", weights_only=False)
    if payload["contract"] != CONTRACT:
        raise ValueError("unsupported stochastic forecast checkpoint")
    model = StochasticPriceForecast(payload["featureMean"], payload["featureStd"], payload["basis"], tuple(payload["widths"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload
