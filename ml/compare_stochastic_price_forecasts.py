"""Compare conditional GLU with a train-fitted unconditional mixed-law baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stochastic_price_forecast import StochasticPriceForecast, load_forecast, load_timeline_split
from return_knot_density import transform_returns_to_unit
from train_stochastic_price_forecast import evaluate, write_json


def fit_marginal_masses(targets: np.ndarray, model: StochasticPriceForecast, iterations: int = 100) -> np.ndarray:
    """EM fits triangular mixture masses on nonzero training observations."""
    active = torch.from_numpy(np.asarray(targets[targets != 0]).copy())
    unit, _ = transform_returns_to_unit(active, model.transform)
    values = unit.numpy().astype(np.float64)
    knots = np.asarray(model.basis["knots"])
    areas = np.asarray(model.basis["basisAreas"])
    interval = np.searchsorted(knots[1:-1], values)
    fraction = (values - knots[interval]) / (knots[interval + 1] - knots[interval])
    left_density = (1 - fraction) / areas[interval]
    right_density = fraction / areas[interval + 1]
    masses = np.asarray(model.basis["priorMasses"])
    for _ in range(iterations):
        left = masses[interval] * left_density
        right = masses[interval + 1] * right_density
        responsibility = left / np.maximum(left + right, np.finfo(float).tiny)
        updated = np.bincount(interval, weights=responsibility, minlength=len(masses))
        updated += np.bincount(interval + 1, weights=1 - responsibility, minlength=len(masses))
        updated += 0.5
        masses = updated / updated.sum()
    return masses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/training/runs/stochastic-crra-production178-zeroatom-k256-v1/best.pt"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    neural, payload = load_forecast(args.checkpoint, device)
    dataset = Path(payload["plan"]["dataset"])
    train = load_timeline_split(dataset, "train")
    masses = fit_marginal_masses(train.targets(), neural)
    basis = {**payload["basis"], "priorMasses": masses.tolist()}
    marginal = StochasticPriceForecast(payload["featureMean"], payload["featureStd"], basis, tuple(payload["widths"])).to(device).eval()
    validation = load_timeline_split(dataset, "validation")
    x = torch.from_numpy(validation.features()).to(device)
    y = torch.from_numpy(validation.targets().copy()).to(device)
    scores = {"conditionalGlu": evaluate(neural, x, y, 4096),
              "unconditionalMarginal": evaluate(marginal, x, y, 4096)}
    winner = min(scores, key=lambda name: scores[name]["negativeLogLikelihood"])
    output = args.checkpoint.parent
    marginal_payload = {**payload, "basis": basis, "model": marginal.state_dict(),
                        "selectedEpoch": 0, "validation": scores["unconditionalMarginal"],
                        "baseline": "100 train-only EM iterations with half-count smoothing"}
    torch.save(marginal_payload, output / "marginal.pt")
    selected_checkpoint = args.checkpoint if winner == "conditionalGlu" else output / "marginal.pt"
    # This choice is persisted before loading/evaluating test targets.
    selection = {"selection": "minimum validation mixed-law negative log likelihood",
                 "selectedModel": winner, "selectedCheckpoint": str(selected_checkpoint),
                 "validation": scores, "testUsedForSelection": False}
    write_json(output / "model-selection.json", selection)
    test = load_timeline_split(dataset, "test")
    tx = torch.from_numpy(test.features()).to(device)
    ty = torch.from_numpy(test.targets().copy()).to(device)
    result = {**selection, "test": {"conditionalGlu": evaluate(neural, tx, ty, 4096),
                                    "unconditionalMarginal": evaluate(marginal, tx, ty, 4096)}}
    write_json(output / "model-comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
