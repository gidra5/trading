"""Independent physical-return integration to audit forecast means and units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
import torch

from horizon_price_forecast import load_horizon_forecast, load_horizon_split
from train_stochastic_price_forecast import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path,
                        default=Path("data/training/runs/stochastic-crra-direct-horizons-5m15m-v1"))
    parser.add_argument("--horizon", type=int, default=900)
    parser.add_argument("--output", type=Path,
                        default=Path("data/backtests/oracle-horizon-search/price-mean-audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    selection = json.loads((args.runs_root / f"h{args.horizon}/model-selection.json").read_text())
    model, payload = load_horizon_forecast(Path(selection["selectedCheckpoint"]), torch.device("cpu"))
    split = load_horizon_split(Path(payload["plan"]["dataset"]), "validation", args.horizon)
    origins = split.origins[::args.horizon]
    features = torch.from_numpy(split.features(origins))
    zero, masses = model.probabilities(features)
    masses = masses / masses.sum(axis=1, keepdims=True)
    knot_means = (1 - zero) * (masses @ model._component_moments("excess"))
    selected = int(np.argmax(knot_means))
    knots = np.asarray(model.basis["knots"])
    heights = masses[selected] / np.asarray(model.basis["basisAreas"])
    sigma = model.sigma
    bound = 12.
    # Integrate in physical Gaussian log-return coordinates, independently of
    # the exporter's Gauss-Legendre integration over Gaussian-CDF intervals.
    breaks = ndtri(knots[1:-1])
    breaks = breaks[(breaks > -bound) & (breaks < bound)]

    def density(z):
        return np.interp(ndtr(z), knots, heights) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)

    total, mass_error = quad(density, -bound, bound, points=breaks, epsabs=1e-11, epsrel=1e-10, limit=1200)
    signed, signed_error = quad(lambda z: np.expm1(sigma * z) * density(z), -bound, bound,
                               points=breaks, epsabs=1e-12, epsrel=1e-10, limit=1200)
    absolute, _ = quad(lambda z: abs(np.expm1(sigma * z)) * density(z), -bound, bound,
                       points=np.unique(np.append(breaks, 0.)), epsabs=1e-12, epsrel=1e-10, limit=1200)
    # Since f_U <= max(heights), Gaussian exponential tilting bounds the omitted
    # absolute first moment by the integral of (exp(sigma*z)+1)*phi(z).
    tail_bound = float(max(heights) * (np.exp(sigma**2 / 2) *
        (ndtr(-bound + sigma) + ndtr(-bound - sigma)) + 2 * ndtr(-bound)))
    factor = 1 - zero[selected]
    direct = factor * signed
    exported_gross, exported_probabilities = model.endpoint_scenarios(features[selected:selected + 1], 7)
    exported_mean = float((exported_probabilities * (exported_gross - 1)).sum())
    if abs(total - 1) > 1e-7 or abs(direct - knot_means[selected]) > 1e-7:
        raise ValueError("Independent integration disagrees with the forecast's normalization or mean")
    result = {"horizonSeconds": args.horizon, "checkpoint": selection["selectedCheckpoint"],
        "split": "validation", "selectedContext": "largest predicted mean on nonoverlapping validation origins",
        "timestamp": int(split.source.axis_start_ms + origins[selected] * 1000),
        "zeroAtomProbability": float(zero[selected]), "integratedActiveMass": total,
        "knotMeanBps": float(knot_means[selected] * 1e4), "independentIntegralMeanBps": direct * 1e4,
        "exportedScenarioMeanBpsBeforePriceRounding": exported_mean * 1e4,
        "independentExpectedAbsoluteReturnBps": factor * absolute * 1e4,
        "sameSignedMeanPercent": direct * 100,
        "absoluteMeanDifferenceBps": float(abs(direct - knot_means[selected]) * 1e4),
        "quadratureEstimatedErrorBps": factor * signed_error * 1e4,
        "massQuadratureEstimatedError": mass_error,
        "analyticTruncatedTailUpperBoundBps": factor * tail_bound * 1e4,
        "precisionScope": "SciPy error estimates and floating-point bounds, not a Lean-certified integration proof",
        "conversionControl": {"grossRatio": 1.003, "simpleReturnPercent": .3, "simpleReturnBps": 30},
        "changesModelSelection": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
