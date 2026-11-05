"""Train direct-horizon NLL baselines and select without test returns or PnL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
import torch

from horizon_price_forecast import CONTRACT, HorizonPriceForecast, fit_horizon_basis, load_horizon_split
from train_autoregressive_minute_return import build_optimizers
from train_stochastic_price_forecast import evaluate, write_json


def marginal_basis(basis: dict, targets: np.ndarray, iterations: int = 100) -> dict:
    """Train-only EM fit of an unconditional density; no feature information."""
    unit = ndtr(np.asarray(targets[targets != 0], dtype=np.float64) / basis["logReturnSigma"])
    knots = np.asarray(basis["knots"])
    areas = np.asarray(basis["basisAreas"])
    interval = np.searchsorted(knots[1:-1], unit)
    fraction = (unit - knots[interval]) / (knots[interval + 1] - knots[interval])
    left_density = (1 - fraction) / areas[interval]
    right_density = fraction / areas[interval + 1]
    masses = np.asarray(basis["priorMasses"])
    for _ in range(iterations):
        left, right = masses[interval] * left_density, masses[interval + 1] * right_density
        responsibility = left / np.maximum(left + right, np.finfo(float).tiny)
        updated = np.bincount(interval, weights=responsibility, minlength=len(masses))
        updated += np.bincount(interval + 1, weights=1 - responsibility, minlength=len(masses))
        updated += .5
        masses = updated / updated.sum()
    return {**basis, "priorMasses": masses.tolist()}


@torch.no_grad()
def evaluate_price(model: HorizonPriceForecast, x: torch.Tensor, y: torch.Tensor,
                   batch_size: int, normalization: float) -> dict:
    result = evaluate(model, x, y, batch_size)
    total_mse, total_zero = 0.0, 0.0
    for start in range(0, len(y), batch_size):
        target = y[start:start + batch_size].expm1()
        prediction = model.excess_expectation_from_logits(model(x[start:start + batch_size]))
        total_mse += float((prediction - target).square().double().sum())
        total_zero += float(target.square().double().sum())
    result["excessPriceMse"] = total_mse / len(y)
    result["excessPriceMseSkillVsZero"] = 1 - total_mse / total_zero
    result["normalizedExcessPriceMse"] = total_mse / len(y) / normalization
    result["selectionScore"] = result["negativeLogLikelihood"] + result["normalizedExcessPriceMse"]
    return result


@torch.no_grad()
def economic_diagnostics(model: HorizonPriceForecast, x: torch.Tensor, y: torch.Tensor,
                         batch_size: int, decision_stride: int = 1) -> dict:
    forecasts = []
    component_excess = model._component_moments("excess")
    for start in range(0, len(y), batch_size):
        zero, masses = model.probabilities(x[start:start + batch_size])
        masses /= masses.sum(axis=1, keepdims=True)
        forecasts.append((1 - zero) * (masses @ component_excess))
    excess = np.concatenate(forecasts)[::decision_stride]
    gross = 1 + excess
    realized = np.exp(y.detach().cpu().numpy().astype(np.float64))[::decision_stride]
    edge = excess * 1e4
    finite = np.isfinite(gross).all()
    if not finite:
        raise ValueError("nonfinite predicted expected gross return")
    result = {"contexts": len(gross), "decisionStrideSeconds": decision_stride, "finitePredictedMeans": bool(finite),
              "predictedGrossReturnMean": float(gross.mean()),
              "realizedGrossReturnMean": float(realized.mean()),
              "forecastReturnMeanBps": float(edge.mean()),
              "forecastReturnStdBps": float(edge.std()),
              "forecastReturnQuantilesBps": dict(zip(("min", "p01", "p50", "p95", "p99", "max"),
                                                     np.quantile(edge, [0, .01, .5, .95, .99, 1]).tolist())),
              "longCashThresholds": {},
              "evaluation": "stride-selected horizon labels; diagnostics are not compounded backtest PnL",
              "grossMoment": "96-point Gaussian-CDF quadrature per knot interval; finite true exponential moment"}
    for fee in (0.0, .0001, .0002, .0005):
        # Oracle buy and sell fees each multiply acquired/sold units by 1-fee.
        predicted_gain = (1 - fee) ** 2 * gross - 1
        choose_long = predicted_gain > 0
        actual_gain = (1 - fee) ** 2 * realized - 1
        result["longCashThresholds"][str(fee)] = {
            "roundTripBreakEvenGrossReturn": float(1 / (1 - fee) ** 2),
            "chooseLongCount": int(choose_long.sum()), "chooseCashCount": int((~choose_long).sum()),
            "strictCashCount": int((predicted_gain < 0).sum()), "tieCount": int((predicted_gain == 0).sum()),
            "chooseLongFraction": float(choose_long.mean()),
            "nonconstantPredictedLongCashDecision": bool(choose_long.any() and (predicted_gain < 0).any()),
            "predictedMeanPolicyGainBps": float(np.where(choose_long, predicted_gain, 0).mean() * 1e4),
            "realizedMeanPolicyGainBps": float(np.where(choose_long, actual_gain, 0).mean() * 1e4),
            "realizedGainWhenLongBps": float(actual_gain[choose_long].mean() * 1e4) if choose_long.any() else None,
        }
    return result


def train(args: argparse.Namespace, horizon: int, mean_weight: float) -> dict:
    output = args.output / f"h{horizon}" / f"lambda{mean_weight:g}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise ValueError(f"refusing to overwrite {output / 'best.pt'}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    splits = {name: load_horizon_split(args.dataset, name, horizon) for name in ("train", "validation")}
    arrays = {name: split.features() for name, split in splits.items()}
    targets = {name: split.targets() for name, split in splits.items()}
    normalization = float(np.mean(np.expm1(targets["train"].astype(np.float64)) ** 2))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("training excess-price second moment must be positive and finite")
    feature_mean = torch.from_numpy(arrays["train"].mean(axis=0, dtype=np.float64).astype(np.float32))
    feature_std = torch.from_numpy(np.maximum(arrays["train"].std(axis=0, dtype=np.float64), 1e-8).astype(np.float32))
    basis = fit_horizon_basis(targets["train"])
    model = HorizonPriceForecast(feature_mean, feature_std, basis).to(device)
    x = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
    y = {name: torch.from_numpy(value.copy()).to(device) for name, value in targets.items()}
    del arrays
    plan = {"contract": CONTRACT, "dataset": str(args.dataset), "horizonSeconds": horizon,
            "target": "log(actual endpoint price / observed price), verified against sum of next H returns; exact endpoint zero atom",
            "input": "120 causal one-second log-return lags and 58 latest observed features; no future inputs",
            "splitBoundary": "discard every origin whose target crosses its chronological split end",
            "seed": args.seed, "maxEpochs": args.epochs, "earlyStoppingPatience": args.patience,
            "batchSize": args.batch_size, "widths": [512] * 4,
            "trainingLoss": "mixed-law NLL + lambda * excess-price mean squared error / training second moment",
            "meanLossWeight": mean_weight, "excessPriceSecondMomentTraining": normalization,
            "selection": "validation mixed-law NLL + normalized excess-price MSE, coefficient 1 for every candidate",
            "decisionFrequencyConstraint": None, "testUsedForSelection": False,
            "coordinates": "Gaussian CDF of log return; finite exponential moments",
            "counts": {name: len(value) for name, value in targets.items()},
            "intervals": {name: {"firstOriginMs": int(split.source.axis_start_ms + 1000 * split.origins[0]),
                                  "lastOriginMs": int(split.source.axis_start_ms + 1000 * split.origins[-1]),
                                  "lastTargetMs": int(split.source.axis_start_ms + 1000 * (split.origins[-1] + horizon))}
                          for name, split in splits.items()},
            "datasetManifestSha256": hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()}
    write_json(output / "plan.json", plan)
    write_json(output / "basis.json", basis)
    configuration = json.loads(Path("ml/training-plans/next-return-production-basis-density-4l-65k-k256-v1.json").read_text())["training"]
    optimizers = build_optimizers(model, configuration, device)
    started = time.monotonic()
    best, best_epoch = float("inf"), -1
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            indices = torch.randperm(len(y["train"]), device=device)
            for start in range(0, len(indices), args.batch_size):
                selected = indices[start:start + args.batch_size]
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(x["train"][selected])
                selected_targets = y["train"][selected]
                nll = -model.log_prob_from_logits(logits, selected_targets).mean()
                mean_loss = (model.excess_expectation_from_logits(logits) - selected_targets.expm1()).square().mean() / normalization
                loss = nll + mean_weight * mean_loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite training NLL")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                for optimizer in optimizers:
                    optimizer.step()
        validation = evaluate_price(model, x["validation"], y["validation"], args.batch_size, normalization)
        score = validation["selectionScore"]
        if score < best:
            best, best_epoch = score, epoch
            payload = {"contract": CONTRACT, "model": model.state_dict(), "featureMean": feature_mean,
                       "featureStd": feature_std, "basis": basis, "widths": [512] * 4,
                       "selectedEpoch": epoch, "validation": validation, "plan": plan}
            torch.save(payload, output / "best.pt.tmp")
            os.replace(output / "best.pt.tmp", output / "best.pt")
        row = {"horizonSeconds": horizon, "meanLossWeight": mean_weight, "epoch": epoch, "bestEpoch": best_epoch,
               "seconds": time.monotonic() - started, "validation": validation}
        with (output / "training.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    payload = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    marginal = HorizonPriceForecast(feature_mean, feature_std, marginal_basis(basis, targets["train"])).to(device).eval()
    scores = {"conditionalGlu": payload["validation"],
              "unconditionalMarginal": evaluate_price(marginal, x["validation"], y["validation"], args.batch_size, normalization)}
    winner = min(scores, key=lambda name: scores[name]["selectionScore"])
    torch.save({**payload, "basis": marginal.basis, "model": marginal.state_dict(), "selectedEpoch": 0,
                "validation": scores["unconditionalMarginal"], "baseline": "train-only 100-iteration EM"}, output / "marginal.pt")
    selected_checkpoint = output / ("best.pt" if winner == "conditionalGlu" else "marginal.pt")
    selection = {"selectedModel": winner, "selectedCheckpoint": str(selected_checkpoint),
                 "selection": "validation NLL plus normalized excess-price MSE, coefficient 1",
                 "validation": scores, "testUsedForSelection": False}
    write_json(output / "model-selection.json", selection)
    validation_economics = {name: economic_diagnostics(candidate, x["validation"], y["validation"], args.batch_size)
                           for name, candidate in (("conditionalGlu", model), ("unconditionalMarginal", marginal))}
    nonoverlapping_validation = {name: economic_diagnostics(candidate, x["validation"], y["validation"], args.batch_size, horizon)
                                for name, candidate in (("conditionalGlu", model), ("unconditionalMarginal", marginal))}
    write_json(output / "validation-economic-gate.json",
               {"split": "validation", "decisionStrideSeconds": horizon,
                "testUsed": False, "models": nonoverlapping_validation})
    # The selected checkpoint and entire selection plan are persisted before test loading.
    test = load_horizon_split(args.dataset, "test", horizon)
    tx, ty = torch.from_numpy(test.features()).to(device), torch.from_numpy(test.targets().copy()).to(device)
    test_results = {name: {"distribution": evaluate_price(candidate, tx, ty, args.batch_size, normalization),
                           "economics": economic_diagnostics(candidate, tx, ty, args.batch_size)}
                    for name, candidate in (("conditionalGlu", model), ("unconditionalMarginal", marginal))}
    result = {"contract": CONTRACT, "plan": plan, "selectedEpoch": best_epoch, "completedEpoch": epoch,
              "seconds": time.monotonic() - started, "selection": selection,
              "validationEconomics": validation_economics, "nonoverlappingValidationEconomics": nonoverlapping_validation,
              "test": test_results,
              "checkpointSha256": hashlib.sha256(selected_checkpoint.read_bytes()).hexdigest()}
    write_json(output / "result.json", result)
    print(json.dumps({"completedHorizonSeconds": horizon, "selectedModel": winner,
                      "meanLossWeight": mean_weight, "selectedEpoch": best_epoch,
                      "result": str(output / "result.json")}), flush=True)
    return {"meanLossWeight": mean_weight, "selectedModel": winner,
            "selectedCheckpoint": str(selected_checkpoint), "validation": scores[winner],
            "result": str(output / "result.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/training/runs/stochastic-crra-direct-horizons-gaussian-v2"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 30, 120])
    parser.add_argument("--mean-loss-weights", type=float, nargs="+", default=[0, 1])
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(not np.isfinite(weight) or weight < 0 for weight in args.mean_loss_weights):
        raise ValueError("proper auxiliary loss weights must be finite and nonnegative")
    if len(set(args.mean_loss_weights)) != len(args.mean_loss_weights) or len(set(args.horizons)) != len(args.horizons):
        raise ValueError("horizons and auxiliary weights must be unique")
    if args.epochs < 0 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("invalid training duration or batch size")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    for horizon in args.horizons:
        candidates = [train(args, horizon, weight) for weight in args.mean_loss_weights]
        selected = min(candidates, key=lambda candidate: candidate["validation"]["selectionScore"])
        write_json(args.output / f"h{horizon}" / "model-selection.json",
                   {"horizonSeconds": horizon, "selection": "minimum validation NLL plus train-normalized excess-price MSE",
                    "selectedCheckpoint": selected["selectedCheckpoint"], "selectedMeanLossWeight": selected["meanLossWeight"],
                    "selectedModel": selected["selectedModel"], "candidates": candidates, "testUsedForSelection": False})


if __name__ == "__main__":
    main()
