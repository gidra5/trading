"""Decompose frozen joint densities into direction probabilities and mean sizes.

This diagnostic changes neither predictions nor model selection. It separates
the amount of movement a joint law already predicts from its directional edge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
import torch

from horizon_price_forecast import load_horizon_forecast, load_horizon_split


def side_integrals(model, positive: bool) -> tuple[np.ndarray, np.ndarray]:
    knots = np.asarray(model.basis["knots"], dtype=np.float64)
    areas = np.asarray(model.basis["basisAreas"], dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(192)
    lower = np.maximum(knots[:-1], .5) if positive else knots[:-1]
    upper = knots[1:] if positive else np.minimum(knots[1:], .5)
    half = np.maximum(upper - lower, 0) / 2
    # Empty intervals are evaluated at the finite zero-return coordinate.
    middle = np.where(half > 0, (lower + upper) / 2, .5)
    unit = middle[:, None] + half[:, None] * nodes
    fraction = (unit - knots[:-1, None]) / np.diff(knots)[:, None]
    magnitude = np.abs(np.expm1(model.sigma * ndtri(unit)))
    result = []
    for value in (np.ones_like(unit), magnitude):
        integral = np.zeros(len(knots))
        integral[:-1] += np.sum(value * (1 - fraction) * weights, axis=1) * half
        integral[1:] += np.sum(value * fraction * weights, axis=1) * half
        result.append(integral / areas)
    return tuple(result)


def summary(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


@torch.no_grad()
def audit(horizon: int) -> dict:
    runs = Path("data/training/runs")
    family = "stochastic-crra-direct-horizons-5m15m-v1" if horizon == 900 else "stochastic-crra-horizon-search-v1"
    selection = json.loads((runs / family / f"h{horizon}" / "model-selection.json").read_text())
    checkpoint = Path(selection["selectedCheckpoint"])
    model, payload = load_horizon_forecast(checkpoint, torch.device("cpu"))
    dataset = Path(payload["plan"]["dataset"])
    training = load_horizon_split(dataset, "train", horizon).targets().astype(np.float64)
    training_return = np.expm1(training)
    train_abs_mean = float(np.abs(training_return).mean())
    train_up_probability = float(np.mean(training[training != 0] > 0))
    up_mass, up_moment = side_integrals(model, True)
    down_mass, down_moment = side_integrals(model, False)
    np.testing.assert_allclose(up_mass + down_mass, 1, atol=2e-12, rtol=0)
    original_moment = model._component_moments("excess", 192)
    np.testing.assert_allclose(up_moment - down_moment, original_moment, atol=1e-8, rtol=0)
    splits = {}
    for split_name in ("validation", "test"):
        split = load_horizon_split(dataset, split_name, horizon)
        count = None if split_name == "validation" else 151200 // horizon
        origins = split.origins[::horizon][:count]
        realized = np.expm1(split.targets()[::horizon][:count].astype(np.float64))
        zero, masses = model.probabilities(torch.from_numpy(split.features(origins)))
        masses = masses / masses.sum(axis=1, keepdims=True)
        active_up = masses @ up_mass
        active_down = masses @ down_mass
        p_up = (1 - zero) * active_up
        p_down = (1 - zero) * active_down
        positive_part = (1 - zero) * (masses @ up_moment)
        negative_part = (1 - zero) * (masses @ down_moment)
        mean_size = positive_part + negative_part
        signed_mean = positive_part - negative_part
        up_size = positive_part / p_up
        down_size = negative_part / p_down
        reconstructed = p_up * up_size - p_down * down_size
        original = (1 - zero) * (masses @ original_moment)
        np.testing.assert_allclose(reconstructed, original, atol=1e-8, rtol=0)
        active = realized != 0
        up = (realized[active] > 0).astype(np.float64)
        probability = np.clip(active_up[active], 1e-12, 1 - 1e-12)
        sign_loss = -(up * np.log(probability) + (1 - up) * np.log1p(-probability)).mean()
        magnitude_mse = np.mean(np.square(mean_size - np.abs(realized)))
        baseline_mse = np.mean(np.square(train_abs_mean - np.abs(realized)))
        rows = [{"timestamp": int(split.source.axis_start_ms + origin * 1000),
                 "zeroProbability": float(zero[index]), "activeUpProbability": float(active_up[index]),
                 "positiveSizeBps": float(up_size[index] * 1e4),
                 "negativeSizeBps": float(down_size[index] * 1e4),
                 "expectedAbsoluteReturnBps": float(mean_size[index] * 1e4),
                 "expectedSignedReturnBps": float(signed_mean[index] * 1e4)}
                for index, origin in enumerate(origins)]
        splits[split_name] = {"contexts": len(origins), "activeContexts": int(active.sum()),
            "predictedActiveUpProbability": summary(active_up),
            "predictedAverageSizeBps": summary(mean_size * 1e4),
            "predictedSignedMeanBps": summary(signed_mean * 1e4),
            "realizedAverageSizeBps": float(np.abs(realized).mean() * 1e4),
            "sizeMseSkillVsTrainConstant": float(1 - magnitude_mse / baseline_mse),
            "activeSignLogLoss": float(sign_loss),
            "activeSignAccuracy": float(np.mean((probability >= .5) == up)),
            "activeSignBrier": float(np.square(probability - up).mean()),
            "activeSignBrierTrainConstant": float(np.square(train_up_probability - up).mean()),
            "meanReconstructionMaxError": float(np.abs(reconstructed - original).max()),
            "expectedSizeAbove10BpsCount": int((mean_size > .001).sum()),
            "expectedSignedMeanAboveFullFeeCount": int((signed_mean > 1 / .9995**2 - 1).sum()),
            "rows": rows}
    return {"horizonSeconds": horizon, "checkpoint": str(checkpoint),
            "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "trainAbsoluteReturnMeanBps": train_abs_mean * 1e4,
            "trainActiveUpProbability": train_up_probability, "splits": splits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("data/backtests/oracle-sign-size/existing-joint-decomposition.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for horizon in (900, 1800, 3600, 7200):
        row = audit(horizon)
        results.append(row)
        print(json.dumps({"horizon": horizon, "test": {key: value for key, value in row["splits"]["test"].items()
                                                       if key != "rows"}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"contract": "frozen-joint-sign-size-decomposition-audit-v1",
        "changesPredictionsOrSelection": False,
        "integration": "192-point Gaussian-CDF quadrature split at zero; floating numerical audit",
        "evaluation": "nonoverlapping validation contexts and common42h test origins",
        "rows": results}, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
