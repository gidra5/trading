"""Validation-selected head temperature, with a frozen test comparison.

Temperature scales Gaussian-coordinate knot HEIGHT logits, retaining triangular
areas in normalization and holding the zero-atom gate fixed. It is neither a power of
the continuous price density nor an assumed mean-preserving transformation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from horizon_price_forecast import load_horizon_forecast, load_horizon_split
from train_stochastic_price_forecast import write_json


@torch.no_grad()
def evaluate_temperatures(model, split, temperatures, device, batch_size=4096):
    features = split.features()
    targets = split.targets()
    moments = torch.as_tensor(model._component_moments("excess"), device=device, dtype=torch.float64)
    accumulators = {temperature: {"nll": 0., "mse": 0., "means": []} for temperature in temperatures}
    for start in range(0, len(targets), batch_size):
        x = torch.from_numpy(features[start:start + batch_size]).to(device)
        y = torch.from_numpy(targets[start:start + batch_size].copy()).to(device)
        logits = model(x)
        for temperature in temperatures:
            scaled = torch.cat((logits[:, :1], logits[:, 1:] / temperature), dim=1)
            nll = -model.log_prob_from_logits(scaled, y).double().sum()
            # Double precision protects sub-basis-point price means.
            masses = torch.softmax(scaled[:, 1:].double() + model.areas.double().log(), dim=-1)
            mean = torch.sigmoid(-scaled[:, 0].double()) * (masses @ moments)
            accumulator = accumulators[temperature]
            accumulator["nll"] += float(nll)
            accumulator["mse"] += float((mean - y.double().expm1()).square().sum())
            accumulator["means"].append(mean.cpu().numpy())
    result = {}
    actual = np.expm1(targets.astype(np.float64))[::split.horizon_seconds]
    fee = .0005
    multiplier = (1 - fee) ** 2
    for temperature, accumulator in accumulators.items():
        means = np.concatenate(accumulator["means"])[::split.horizon_seconds]
        predicted_net = multiplier * (1 + means) - 1
        actual_net = multiplier * (1 + actual) - 1
        entries = predicted_net > 0
        if not np.isfinite(means).all() or not np.isfinite(accumulator["nll"]):
            raise ValueError("Temperature produced nonfinite forecast metrics")
        result[str(temperature)] = {
            "temperature": temperature, "examples": len(targets),
            "negativeLogLikelihood": accumulator["nll"] / len(targets),
            "priceReturnMse": accumulator["mse"] / len(targets),
            "decisionContexts": len(means),
            "predictedMeanQuantilesBps": dict(zip(["min", "p50", "p95", "p99", "max"],
                (np.quantile(means, [0, .5, .95, .99, 1]) * 1e4).tolist())),
            "fullCloseBreakEvenBps": (1 / multiplier - 1) * 1e4,
            "paidLongDecisions": int(entries.sum()), "paidCashDecisions": int((~entries).sum()),
            "paidSwitchingGatePassed": bool(entries.any() and (predicted_net < 0).any()),
            "realizedNetWhenLongBps": float(actual_net[entries].mean() * 1e4) if entries.any() else None,
            "realizedMeanPolicyGainBps": float(np.where(entries, actual_net, 0).mean() * 1e4),
            "predictedMeanPolicyGainBps": float(np.where(entries, predicted_net, 0).mean() * 1e4),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path,
                        default=Path("data/training/runs/stochastic-crra-direct-horizons-gaussian-v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/training/runs/stochastic-crra-temperature-v1"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 30, 120])
    args = parser.parse_args()
    temperatures = [.05, .1, .25, .5, .75, 1., 1.5, 2., 4.]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "plan.json", {
        "temperatures": temperatures, "selection": "minimum validation mixed-law negative log likelihood",
        "testUsedForSelection": False, "checkpointsChanged": False,
        "temperatureDefinition": "active knot height logits divided by T; basis areas retained; zero-atom probability unchanged",
        "scope": "fresh cash long/full-close threshold diagnostics, not an account trajectory backtest",
        "calibrationData": "existing validation split, already used for checkpoint selection",
        "chronology": "grid fixed before this temperature evaluation; test was used in earlier uncalibrated experiments",
    })
    reports = []
    for horizon in args.horizons:
        selection = json.loads((args.runs_root / f"h{horizon}/model-selection.json").read_text())
        model, payload = load_horizon_forecast(Path(selection["selectedCheckpoint"]), device)
        dataset = Path(payload["plan"]["dataset"])
        validation = evaluate_temperatures(model, load_horizon_split(dataset, "validation", horizon),
                                            temperatures, device)
        # T=1 is an identity control against the completed selected checkpoint.
        if abs(validation["1.0"]["negativeLogLikelihood"] - payload["validation"]["negativeLogLikelihood"]) > 2e-5:
            raise ValueError("Temperature-one likelihood differs from original checkpoint validation")
        selected_temperature = min(temperatures, key=lambda t: validation[str(t)]["negativeLogLikelihood"])
        report = {"horizonSeconds": horizon, "checkpoint": selection["selectedCheckpoint"],
                  "selectedTemperature": selected_temperature, "validation": validation,
                  "selectionUsesTest": False}
        # Freeze the selected T before loading any test input or target.
        write_json(args.output / f"h{horizon}-selection.json", report)
        print(json.dumps({"horizonSeconds": horizon, "selectedTemperature": selected_temperature,
                          "validation": validation[str(selected_temperature)]}), flush=True)
        test_temperatures = sorted(set([1., selected_temperature]))
        report["test"] = evaluate_temperatures(model, load_horizon_split(dataset, "test", horizon),
                                               test_temperatures, device)
        write_json(args.output / f"h{horizon}-result.json", report)
        reports.append(report)
        print(json.dumps({"horizonSeconds": horizon, "test": report["test"]}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(args.output / "report.json", {"horizons": reports, "checkpointsChanged": False})


if __name__ == "__main__":
    main()
