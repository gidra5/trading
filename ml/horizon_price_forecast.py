"""Causal direct-horizon price laws with finite exponential moments.

The GLU and normalized triangular density are shared with the one-second
forecast. Gaussian-CDF log-return coordinates replace polynomial log tails:
gamma=0 requires an integrable gross price return, not just an integrable log
return. A bounded piecewise-linear density on Gaussian probability coordinates
has all exponential moments. No action-frequency constraint is imposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
import torch
from torch import Tensor
from torch.nn import functional as F

from return_knot_density import component_log_masses, interpolated_log_density_unit, triangular_basis_areas
from stochastic_price_forecast import StochasticPriceForecast, TimelineSplit, load_timeline_split

CONTRACT = "causal-production178-horizon-zero-atom-gaussian-knot-density-v1"


@dataclass(frozen=True)
class HorizonSplit:
    source: TimelineSplit
    horizon_seconds: int
    origins: np.ndarray
    horizon_targets: np.ndarray

    def features(self, origins: np.ndarray | None = None) -> np.ndarray:
        return self.source.features(self.origins if origins is None else origins)

    def targets(self) -> np.ndarray:
        return self.horizon_targets


def horizon_split(source: TimelineSplit, horizon_seconds: int) -> HorizonSplit:
    if isinstance(horizon_seconds, bool) or not isinstance(horizon_seconds, int) or horizon_seconds < 1:
        raise ValueError("horizon must be a positive integer number of seconds")
    origins = source.origins[source.origins + horizon_seconds < len(source.timeline)]
    if not len(origins):
        raise ValueError("horizon leaves no complete within-split targets")
    # Float64 accumulation avoids horizon-dependent float32 summation error.
    # Labels use only future price-return channel, never future input features.
    accumulated = np.concatenate(([0.0], np.cumsum(source.timeline[:, 0], dtype=np.float64)))
    targets = (accumulated[origins + horizon_seconds + 1] - accumulated[origins + 1]).astype(np.float32)
    return HorizonSplit(source, horizon_seconds, origins, targets)


def load_horizon_split(root: Path, split: str, horizon_seconds: int) -> HorizonSplit:
    from export_stochastic_price_forecast import load_close_prices
    derived = horizon_split(load_timeline_split(root, split), horizon_seconds)
    source = derived.source
    rows = np.arange(derived.origins[0], derived.origins[-1] + horizon_seconds + 1)
    times = source.axis_start_ms + rows * 1000
    prices = load_close_prices(times, Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
    indices = derived.origins - rows[0]
    targets = np.log(prices[indices + horizon_seconds] / prices[indices]).astype(np.float32)
    np.testing.assert_allclose(targets, derived.horizon_targets, atol=3e-9, rtol=2e-5,
                               err_msg="direct endpoint targets disagree with causal return timeline")
    # Exact equality of endpoint prices is an exact zero-return atom. Adding
    # float32 one-step log returns can leave roundtrip rounding noise instead.
    return HorizonSplit(source, horizon_seconds, derived.origins, targets)


def fit_horizon_basis(targets: np.ndarray, knot_count: int = 256) -> dict:
    values = np.asarray(targets, dtype=np.float64)
    active = values[values != 0]
    if len(active) < knot_count or not np.isfinite(values).all():
        raise ValueError("insufficient finite active training targets")
    sigma = max(float(np.std(active)) * 2, 1e-8)
    unit = ndtr(active / sigma)
    knots = np.unique(np.concatenate((np.linspace(0, 1, knot_count // 2),
                                     np.quantile(unit, np.linspace(0, 1, knot_count // 2))))).astype(np.float32)
    knots = np.unique(knots).astype(np.float64)
    areas = triangular_basis_areas(torch.from_numpy(knots)).numpy()
    assignment = np.searchsorted((knots[1:] + knots[:-1]) / 2, unit)
    masses = np.bincount(assignment, minlength=len(knots)).astype(np.float64) + 0.5
    masses /= masses.sum()
    return {"coordinate": "gaussian-cdf", "logReturnSigma": sigma,
            # The shared base class initializes unused asinh coordinates; all
            # likelihood, quantile and expectation methods are replaced below.
            "transform": {"alpha": 4.0, "location_bps": 0.0, "scale_bps": sigma * 1e4},
            "knots": knots.tolist(), "basisAreas": areas.tolist(), "priorMasses": masses.tolist(),
            "priorZeroProbability": float(np.mean(values == 0)), "fitSplit": "train",
            "observations": len(values), "maximumAbsoluteTrainingReturn": float(np.max(np.abs(values)))}


class HorizonPriceForecast(StochasticPriceForecast):
    def __init__(self, feature_mean: Tensor, feature_std: Tensor, basis: dict,
                 widths: tuple[int, ...] = (512, 512, 512, 512)) -> None:
        if basis.get("coordinate") != "gaussian-cdf":
            raise ValueError("direct-horizon forecast requires Gaussian-CDF coordinates")
        super().__init__(feature_mean, feature_std, basis, widths)
        self.sigma = float(basis["logReturnSigma"])
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Gaussian coordinate scale must be positive and finite")
        self.component_means.copy_(torch.from_numpy(self._component_moments("log")).float())
        self.register_buffer("component_gross_means", torch.from_numpy(self._component_moments("gross")).float())
        self.register_buffer("component_excess_means", torch.from_numpy(self._component_moments("excess")).float())

    def log_prob_from_logits(self, logits: Tensor, targets: Tensor) -> Tensor:
        z = targets / self.sigma
        unit = torch.special.ndtr(z).clamp(0, 1)
        jacobian = -z.square() / 2 - np.log(self.sigma) - np.log(2 * np.pi) / 2
        masses = component_log_masses(logits[:, 1:], self.areas)
        active = interpolated_log_density_unit(masses, unit, self.knots, self.areas) + jacobian
        return torch.where(targets == 0, F.logsigmoid(logits[:, 0]), F.logsigmoid(-logits[:, 0]) + active)

    def _component_moments(self, kind: str, quadrature_order: int = 96) -> np.ndarray:
        knots = np.asarray(self.basis["knots"])
        areas = np.asarray(self.basis["basisAreas"])
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        widths = np.diff(knots)
        unit = (knots[:-1, None] + knots[1:, None]) / 2 + widths[:, None] * nodes[None, :] / 2
        log_returns = self.sigma * ndtri(unit)
        values = log_returns if kind == "log" else (np.expm1(log_returns) if kind == "excess" else np.exp(log_returns))
        fractions = (nodes + 1) / 2
        left = np.sum(values * (1 - fractions)[None, :] * weights[None, :], axis=1) * widths / 2
        right = np.sum(values * fractions[None, :] * weights[None, :], axis=1) * widths / 2
        result = np.zeros(len(knots))
        result[:-1] += left
        result[1:] += right
        return result / areas

    def gross_expectation_from_logits(self, logits: Tensor) -> Tensor:
        masses = component_log_masses(logits[:, 1:], self.areas).exp()
        p0 = torch.sigmoid(logits[:, 0])
        return p0 + (1 - p0) * (masses @ self.component_gross_means)

    def excess_expectation_from_logits(self, logits: Tensor) -> Tensor:
        masses = component_log_masses(logits[:, 1:], self.areas).exp()
        return torch.sigmoid(-logits[:, 0]) * (masses @ self.component_excess_means)

    def active_gross_mean(self, masses: np.ndarray) -> np.ndarray:
        # Float64 avoids erasing economically small net-return differences.
        normalized = masses / masses.sum(axis=1, keepdims=True)
        return normalized @ self._component_moments("gross")

    def quantile_returns(self, masses: np.ndarray, branches: int) -> np.ndarray:
        if branches < 1:
            raise ValueError("at least one active branch is required")
        return self.sigma * ndtri(self.quantile_units(masses, (np.arange(branches) + .5) / branches))

    def quantile_bin_means(self, masses: np.ndarray, branches: int) -> np.ndarray:
        raise NotImplementedError("direct-horizon laws use endpoint_scenarios and gross_quantile_bin_means")

    def gross_quantile_bin_means(self, masses: np.ndarray, branches: int) -> np.ndarray:
        """Finite gross-return representatives for an endpoint gamma=0 menu.

        96-point quadrature is used per knot interval. This approximates a
        well-defined continuous first price moment, unlike polynomial log tails.
        Compression preserves this quadrature moment, not nonlinear utilities.
        """
        if branches < 1:
            raise ValueError("at least one active branch is required")
        masses = masses / masses.sum(axis=1, keepdims=True)
        knots = np.asarray(self.basis["knots"])
        heights = masses / np.asarray(self.basis["basisAreas"])[None, :]
        nodes, weights = np.polynomial.legendre.leggauss(96)
        widths = np.diff(knots)
        unit = (knots[:-1, None] + knots[1:, None]) / 2 + widths[:, None] * nodes[None, :] / 2
        gross = np.exp(self.sigma * ndtri(unit))
        fractions = (nodes + 1) / 2
        left = np.sum(gross * (1 - fractions)[None, :] * weights[None, :], axis=1) * widths / 2
        right = np.sum(gross * fractions[None, :] * weights[None, :], axis=1) * widths / 2
        moments = heights[:, :-1] * left + heights[:, 1:] * right
        cumulative = np.concatenate((np.zeros((len(masses), 1)), np.cumsum(moments, axis=1)), axis=1)
        cuts = self.quantile_units(masses, np.arange(1, branches) / branches)
        result = np.zeros((len(masses), branches + 1))
        result[:, -1] = cumulative[:, -1]
        row = np.arange(len(masses))
        for boundary in range(branches - 1):
            upper = cuts[:, boundary]
            interval = np.minimum(np.searchsorted(knots, upper, side="right") - 1, len(knots) - 2)
            left_bound = knots[interval]
            half = (upper - left_bound) / 2
            partial_unit = (upper[:, None] + left_bound[:, None]) / 2 + half[:, None] * nodes[None, :]
            partial_gross = np.exp(self.sigma * ndtri(partial_unit))
            fraction = (partial_unit - left_bound[:, None]) / widths[interval, None]
            density = heights[row, interval, None] * (1 - fraction) + heights[row, interval + 1, None] * fraction
            partial = np.sum(partial_gross * density * weights[None, :], axis=1) * half
            result[:, boundary + 1] = cumulative[row, interval] + partial
        return np.diff(result, axis=1) * branches

    @torch.no_grad()
    def endpoint_scenarios(self, features: Tensor, branches: int = 7) -> tuple[np.ndarray, np.ndarray]:
        p0, masses = self.probabilities(features)
        gross = np.concatenate((np.ones((len(p0), 1)), self.gross_quantile_bin_means(masses, branches)), axis=1)
        probabilities = np.concatenate((p0[:, None], np.repeat(((1 - p0) / branches)[:, None], branches, axis=1)), axis=1)
        return gross, probabilities


def load_horizon_forecast(file: Path, device: torch.device) -> tuple[HorizonPriceForecast, dict]:
    payload = torch.load(file, map_location="cpu", weights_only=False)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported direct-horizon forecast contract")
    model = HorizonPriceForecast(payload["featureMean"], payload["featureStd"], payload["basis"], tuple(payload["widths"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload
