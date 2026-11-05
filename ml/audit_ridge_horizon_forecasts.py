"""Bounded CPU ridge/lognormal control for conditional price-mean underfitting.

The chronological validation set selects regularization and horizon before any
test data is loaded. No model is required to emit trades. The actual law is a
fixed train-zero atom plus an active conditional Gaussian log return.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from horizon_price_forecast import load_horizon_split

HORIZONS = [1800, 3600, 7200, 14400, 28800, 43200, 86400]
LAMBDAS = [.0001, .001, .01, .1, 1., 10., 100.]


def write_json(file: Path, value: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Candidate:
    name: str
    coefficient: np.ndarray
    intercept: float
    residual_variance: float
    zero_probability: float
    ridge_lambda: float | None


def moments(source, origins: np.ndarray, targets: np.ndarray, chunk_size: int) -> tuple:
    width, horizon_count = 178, targets.shape[1]
    count = 0
    mean = np.zeros(width)
    squares = np.zeros(width)
    for start in range(0, len(origins), chunk_size):
        x = source.features(origins[start:start + chunk_size]).astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_n = len(x)
        delta = batch_mean - mean
        squares += np.square(x - batch_mean).sum(axis=0) + np.square(delta) * count * batch_n / (count + batch_n)
        mean += delta * batch_n / (count + batch_n)
        count += batch_n
    std = np.sqrt(squares / count)
    std[std < 1e-12] = 1
    gram = np.zeros((width, width))
    sum_x = np.zeros(width)
    xy = np.zeros((width, horizon_count))
    zero_gram = np.zeros((horizon_count, width, width))
    zero_x = np.zeros((horizon_count, width))
    zero_n = np.zeros(horizon_count, dtype=np.int64)
    for start in range(0, len(origins), chunk_size):
        x = (source.features(origins[start:start + chunk_size]).astype(np.float64) - mean) / std
        y = targets[start:start + chunk_size]
        gram += x.T @ x
        sum_x += x.sum(axis=0)
        xy += x.T @ y
        for index in range(horizon_count):
            active_zero = x[y[:, index] == 0]
            zero_n[index] += len(active_zero)
            if len(active_zero):
                zero_gram[index] += active_zero.T @ active_zero
                zero_x[index] += active_zero.sum(axis=0)
    active_n = count - zero_n
    y_sum, y_squares = targets.sum(axis=0), np.square(targets).sum(axis=0)
    candidates = {}
    training = {}
    for index, horizon in enumerate(HORIZONS):
        n = int(active_n[index])
        if n <= width:
            raise ValueError("insufficient nonzero targets for ridge control")
        x_mean = (sum_x - zero_x[index]) / n
        y_mean = y_sum[index] / n
        covariance = (gram - zero_gram[index]) / n - np.outer(x_mean, x_mean)
        covariance = (covariance + covariance.T) / 2
        covariance_xy = xy[:, index] / n - x_mean * y_mean
        variance = max(float(y_squares[index] / n - y_mean**2), 1e-16)
        p0 = (int(zero_n[index]) + .5) / (count + 1)
        zero = np.zeros(width)
        rows = [Candidate("zeroPriceMean", zero, -variance / 2, variance, p0, None),
                Candidate("trainDrift", zero, float(y_mean), variance, p0, None)]
        for strength in LAMBDAS:
            beta = np.linalg.solve(covariance + strength * np.eye(width), covariance_xy)
            residual = max(float(variance - 2 * beta @ covariance_xy + beta @ covariance @ beta), 1e-16)
            rows.append(Candidate(f"ridge{strength:g}", beta, float(y_mean - x_mean @ beta), residual, p0, strength))
        candidates[horizon] = rows
        training[horizon] = {"commonOrigins": count, "activeOrigins": n, "zeroOrigins": int(zero_n[index]),
                             "rawZeroFrequency": float(zero_n[index] / count), "smoothedZeroProbability": p0,
                             "activeLogReturnMean": float(y_mean), "activeLogReturnVariance": variance}
    return mean, std, candidates, training


def evaluate(split, candidates: list[Candidate], feature_mean: np.ndarray, feature_std: np.ndarray,
             batch_size: int) -> dict:
    targets = split.targets().astype(np.float64)
    excess = np.expm1(targets)
    coefficient = np.stack([candidate.coefficient for candidate in candidates], axis=1)
    intercept = np.array([candidate.intercept for candidate in candidates])
    variance = np.array([candidate.residual_variance for candidate in candidates])
    p0 = np.array([candidate.zero_probability for candidate in candidates])
    forecasts = np.empty((len(targets), len(candidates)))
    nll_sum = np.zeros(len(candidates))
    for start in range(0, len(targets), batch_size):
        end = min(start + batch_size, len(targets))
        x = (split.features(split.origins[start:end]).astype(np.float64) - feature_mean) / feature_std
        mu = x @ coefficient + intercept
        forecasts[start:end] = (1 - p0) * np.expm1(mu + variance / 2)
        target = targets[start:end, None]
        active_loss = -np.log1p(-p0) + .5 * np.log(2 * np.pi * variance) + np.square(target - mu) / (2 * variance)
        nll_sum += np.where(target == 0, -np.log(p0), active_loss).sum(axis=0)
    if not np.isfinite(forecasts).all() or not np.isfinite(nll_sum).all():
        raise ValueError("ridge control produced a nonfinite expectation or likelihood")
    zero_mse = float(np.square(excess).mean())
    stride = split.horizon_seconds
    threshold = 1 / .9995**2 - 1
    realized_nonoverlap = excess[::stride]
    result = {}
    for index, candidate in enumerate(candidates):
        predicted = forecasts[:, index]
        mse = float(np.square(predicted - excess).mean())
        nonoverlap = predicted[::stride]
        long = nonoverlap > threshold
        net_gain = .9995**2 * (1 + realized_nonoverlap) - 1
        constant = float(predicted.std()) <= 1e-15
        result[candidate.name] = {
            "examples": len(targets), "ridgeLambda": candidate.ridge_lambda,
            "mixedNegativeLogLikelihood": float(nll_sum[index] / len(targets)),
            "excessPriceMse": mse, "mseSkillVsZero": 1 - mse / zero_mse,
            "predictedMeanBps": float(predicted.mean() * 1e4),
            "realizedMeanBps": float(excess.mean() * 1e4),
            "predictedMeanReturnCorrelation": None if constant else float(np.corrcoef(predicted, excess)[0, 1]),
            "nonoverlapping": {
                "contexts": len(nonoverlap), "strideSeconds": stride,
                "predictedMinBps": float(nonoverlap.min() * 1e4), "predictedMaxBps": float(nonoverlap.max() * 1e4),
                "chooseLongCount": int(long.sum()), "strictCashCount": int((nonoverlap < threshold).sum()),
                "strictDecisionSwitching": bool(long.any() and (nonoverlap < threshold).any()),
                "realizedMeanNetGainWhenPredictedLongBps": float(net_gain[long].mean() * 1e4) if long.any() else None,
                "cashStartLongCashMeanNetGainBps": float(np.where(long, net_gain, 0).mean() * 1e4),
                "scope": "independent cash-start/full-close decisions, not an account replay; valid full-close/survival assumptions required",
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
        default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--output", type=Path,
        default=Path("data/training/runs/stochastic-crra-ridge-horizon-search-v1"))
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("positive chunk size required")
    torch.set_num_threads(2)
    plan = {"contract": "ridge-active-logreturn-fixed-zero-lognormal-horizon-control-v1",
            "dataset": str(args.dataset), "horizonsSeconds": HORIZONS, "ridgeStrengths": LAMBDAS,
            "features": "same 178 causal production features; train-only shared means and scales",
            "trainingOrigins": "intersection of complete training origins for every listed horizon",
            "fit": "active (nonzero) target covariance per horizon, centered intercept, normalized covariance ridge",
            "residualVariance": "active training population residual variance, lower bound 1e-16",
            "zeroAtom": "fixed train-only Jeffreys estimate (zeroCount+0.5)/(commonCount+1)",
            "expectedExcessPrice": "(1-p0)*(exp(mu + residualVariance/2)-1)",
            "candidateSelection": "minimum validation expected-simple-return MSE among seven ridge lambdas, zero-price-mean and train-drift controls",
            "bestRidgeRecordedSeparately": True,
            "horizonSelection": "first ascending horizon whose validation-selected candidate has a strictly positive full-fee mean gap on a nonoverlapping validation context",
            "noCrossingFallback": "no horizon is selected and no test data is evaluated",
            "testProtocol": "load test only after all validation candidate/horizon choices are saved; evaluate selected horizon candidate and its fixed controls",
            "economicThreshold": "1/(1-.0005)^2 - 1; full-close no-liquidation long/cash diagnostic",
            "testUsedForSelection": False, "tradeFrequencyConstraint": None, "cpuThreads": 2,
            "chunkSize": args.chunk_size,
            "limitations": "long horizons have few nonoverlapping validation contexts; conditional lognormal may fit tails worse than the GLU density"}
    write_json(args.output / "plan.json", plan)
    started = time.monotonic()
    train = {horizon: load_horizon_split(args.dataset, "train", horizon) for horizon in HORIZONS}
    source = train[HORIZONS[-1]].source
    common_origins = train[HORIZONS[-1]].origins
    count = len(common_origins)
    targets = np.stack([train[horizon].targets()[:count] for horizon in HORIZONS], axis=1).astype(np.float64)
    for split in train.values():
        np.testing.assert_array_equal(split.origins[:count], common_origins)
    feature_mean, feature_std, candidates, training = moments(source, common_origins, targets, args.chunk_size)
    rows = []
    selected_horizon = None
    selected_name = None
    for horizon in HORIZONS:
        validation = load_horizon_split(args.dataset, "validation", horizon)
        scores = evaluate(validation, candidates[horizon], feature_mean, feature_std, args.chunk_size)
        selected = min(scores, key=lambda name: scores[name]["excessPriceMse"])
        ridge_names = [candidate.name for candidate in candidates[horizon] if candidate.ridge_lambda is not None]
        best_ridge = min(ridge_names, key=lambda name: scores[name]["excessPriceMse"])
        economic = scores[selected]["nonoverlapping"]
        if selected_horizon is None and economic["chooseLongCount"] > 0:
            selected_horizon, selected_name = horizon, selected
        row = {"horizonSeconds": horizon, "training": training[horizon], "selectedCandidate": selected,
               "bestRidgeCandidate": best_ridge, "validation": scores,
               "limitedIndependentContexts": economic["contexts"] < 30,
               "firstValidationTimestamp": int(validation.source.axis_start_ms + validation.origins[0] * 1000),
               "lastValidationEndpointTimestamp": int(validation.source.axis_start_ms + (validation.origins[-1] + horizon) * 1000)}
        rows.append(row)
        np.savez(args.output / f"h{horizon}.npz", featureMean=feature_mean, featureStd=feature_std,
                 coefficients=np.stack([candidate.coefficient for candidate in candidates[horizon]]),
                 intercepts=np.array([candidate.intercept for candidate in candidates[horizon]]),
                 residualVariances=np.array([candidate.residual_variance for candidate in candidates[horizon]]),
                 zeroProbability=training[horizon]["smoothedZeroProbability"],
                 candidateNames=np.array([candidate.name for candidate in candidates[horizon]]))
        write_json(args.output / f"h{horizon}-validation.json", row)
        print(json.dumps({"horizon": horizon, "selected": selected, "bestRidge": best_ridge,
                          "mseSkill": scores[selected]["mseSkillVsZero"], "economics": economic}), flush=True)
    selection = {"plan": plan, "selectedHorizonSeconds": selected_horizon, "selectedCandidate": selected_name,
                 "testLoaded": False, "horizons": rows, "secondsBeforeTest": time.monotonic() - started}
    write_json(args.output / "validation-selection.json", selection)
    test_result = None
    if selected_horizon is not None:
        test = load_horizon_split(args.dataset, "test", selected_horizon)
        evaluated = [candidate for candidate in candidates[selected_horizon]
                     if candidate.name in {selected_name, "zeroPriceMean", "trainDrift"}]
        test_scores = evaluate(test, evaluated, feature_mean, feature_std, args.chunk_size)
        test_result = {"horizonSeconds": selected_horizon, "selectedCandidate": selected_name,
                       "selectionSavedBeforeTest": True, "testUsedForSelection": False, "scores": test_scores,
                       "firstTimestamp": int(test.source.axis_start_ms + test.origins[0] * 1000),
                       "lastEndpointTimestamp": int(test.source.axis_start_ms + (test.origins[-1] + selected_horizon) * 1000)}
        write_json(args.output / "selected-test.json", test_result)
        print(json.dumps({"selectedTest": test_result}), flush=True)
    result = {"selection": selection, "test": test_result, "seconds": time.monotonic() - started,
              "datasetManifestSha256": hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()}
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
