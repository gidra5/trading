"""Test whether the frozen production59 predictor transfers to cost-sized events.

The saved structured model is not refit.  Its causal predicted returns and
primitive outputs become inputs to a small regularized logistic sign head.
Candidate event clocks and feature views are selected on a chronological tail
of the validation export; the test export is evaluated only after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


BARRIERS_BPS = (24, 48)
TIMEOUTS_SECONDS = (60, 300, 900)
FEATURE_SETS = ("score1", "scores2", "score-history", "primitives2")
PENALTY = 0.1
ONE_WAY_COST_BPS = 12.0
WARMUP = 15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    out = np.empty(len(values), dtype=np.float64)
    for i in range(len(values)):
        start = max(0, i + 1 - width)
        out[i] = (total[i + 1] - total[start]) / (i + 1 - start)
    return out


def feature_matrix(data: np.lib.npyio.NpzFile, feature_set: str) -> np.ndarray:
    scores = data["predictionLogReturns"].astype(np.float64)
    primitives = data["predictedPrimitives"].astype(np.float64)
    if feature_set == "score1":
        return scores[:, :1]
    if feature_set == "scores2":
        return scores[:, :2]
    if feature_set == "score-history":
        score = scores[:, 0]
        mean5 = rolling_mean(score, 5)
        mean15 = rolling_mean(score, 15)
        mean_square15 = rolling_mean(score * score, 15)
        std15 = np.sqrt(np.maximum(0.0, mean_square15 - mean15 * mean15))
        return np.column_stack((score, mean5, mean15, std15))
    if feature_set == "primitives2":
        return primitives.reshape(len(primitives), -1)
    raise ValueError(f"unknown feature set {feature_set}")


def event_labels(log_returns: np.ndarray, barrier_bps: int, timeout: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = len(log_returns) - timeout
    terminal = np.empty(rows, dtype=np.float64)
    duration = np.empty(rows, dtype=np.int32)
    hit = np.zeros(rows, dtype=bool)
    threshold = barrier_bps / 10_000.0
    for origin in range(rows):
        path = np.cumsum(log_returns[origin:origin + timeout], dtype=np.float64)
        arithmetic = np.expm1(path)
        crossings = np.flatnonzero(np.abs(arithmetic) >= threshold)
        end = int(crossings[0]) if len(crossings) else timeout - 1
        terminal[origin] = arithmetic[end]
        duration[origin] = end + 1
        hit[origin] = len(crossings) > 0
    return terminal, duration, hit


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(features: np.ndarray, returns: np.ndarray, penalty: float = PENALTY) -> dict:
    active = returns != 0
    raw = features[active]
    y = (returns[active] > 0).astype(np.float64)
    means = raw.mean(axis=0)
    scales = np.maximum(1e-8, raw.std(axis=0))
    x = np.clip((raw - means) / scales, -5.0, 5.0)
    x = np.column_stack((np.ones(len(x)), x))
    prior = (y.sum() + 0.5) / (len(y) + 1.0)
    beta = np.zeros(x.shape[1], dtype=np.float64)
    beta[0] = math.log(prior / (1.0 - prior))
    reg = np.diag(np.r_[0.0, np.full(x.shape[1] - 1, penalty)])
    for iteration in range(40):
        p = sigmoid(x @ beta)
        gradient = x.T @ (p - y) / len(y) + reg @ beta
        weights = p * (1.0 - p)
        hessian = (x.T * weights) @ x / len(y) + reg
        hessian[0, 0] += 1e-10
        if np.max(np.abs(gradient)) < 1e-8:
            break
        step = np.linalg.solve(hessian, gradient)
        beta -= step
    positive = returns > 0
    negative = returns < 0
    return {
        "means": means,
        "scales": scales,
        "beta": beta,
        "iterations": iteration + 1,
        "activeProbability": float(active.mean()),
        "positiveMean": float(returns[positive].mean()),
        "negativeMean": float(returns[negative].mean()),
        "constantPositiveProbability": float(prior),
    }


def predict(model: dict, features: np.ndarray) -> np.ndarray:
    x = np.clip((features - model["means"]) / model["scales"], -5.0, 5.0)
    return sigmoid(model["beta"][0] + x @ model["beta"][1:])


def metrics(model: dict, features: np.ndarray, returns: np.ndarray, duration: np.ndarray,
            hit: np.ndarray, block_seconds: int) -> dict:
    active = returns != 0
    y = (returns[active] > 0).astype(np.float64)
    p = np.clip(predict(model, features)[active], 1e-6, 1.0 - 1e-6)
    p0 = np.clip(model["constantPositiveProbability"], 1e-6, 1.0 - 1e-6)
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    baseline_losses = -(y * math.log(p0) + (1.0 - y) * math.log(1.0 - p0))
    correct = (p >= 0.5) == (y > 0.5)
    weights = np.abs(returns[active])
    all_p = predict(model, features)
    forecast = model["activeProbability"] * (
        all_p * model["positiveMean"] + (1.0 - all_p) * model["negativeMean"]
    )
    baseline_return = model["activeProbability"] * (
        p0 * model["positiveMean"] + (1.0 - p0) * model["negativeMean"]
    )
    zero_mse = float(np.mean((returns - baseline_return) ** 2))
    mse = float(np.mean((returns - forecast) ** 2))
    action = np.where(np.abs(forecast) * 10_000 > 2 * ONE_WAY_COST_BPS, np.sign(forecast), 0.0)
    gross = action * returns * 10_000
    net = gross - np.abs(action) * 2 * ONE_WAY_COST_BPS
    active_indices = np.flatnonzero(active)
    block_gain = []
    for start in range(0, len(returns), block_seconds):
        selected = (active_indices >= start) & (active_indices < start + block_seconds)
        if selected.any():
            block_gain.append(float(np.mean(baseline_losses[selected] - losses[selected]) / math.log(2)))
    return {
        "rows": int(len(returns)),
        "activeRows": int(active.sum()),
        "barrierHitRate": float(hit.mean()),
        "meanDurationSeconds": float(duration.mean()),
        "signLogLoss": float(losses.mean()),
        "constantSignLogLoss": float(baseline_losses.mean()),
        "signInformationBitsPerActiveRow": float(np.mean(baseline_losses - losses) / math.log(2)),
        "directionAccuracy": float(correct.mean()),
        "magnitudeWeightedDirectionAccuracy": float(np.dot(correct, weights) / weights.sum()),
        "returnMseSkillVsTrainingMean": float(1.0 - mse / zero_mse) if zero_mse else 0.0,
        "maximumAbsoluteForecastMeanBps": float(np.max(np.abs(forecast)) * 10_000),
        "actionableRowsAt24BpsRoundTrip": int(np.count_nonzero(action)),
        "overlappingActionScreenNetMeanBps": float(net.mean()),
        "blockSeconds": block_seconds,
        "blocks": len(block_gain),
        "positiveInformationBlocks": int(np.count_nonzero(np.asarray(block_gain) > 0)),
        "blockInformationBits": block_gain,
    }


def serializable_model(model: dict) -> dict:
    return {
        "means": model["means"].tolist(),
        "scales": model["scales"].tolist(),
        "coefficients": model["beta"][1:].tolist(),
        "intercept": float(model["beta"][0]),
        "iterations": model["iterations"],
        "activeProbability": model["activeProbability"],
        "positiveMeanBps": model["positiveMean"] * 10_000,
        "negativeMeanBps": model["negativeMean"] * 10_000,
        "constantPositiveProbability": model["constantPositiveProbability"],
        "penalty": PENALTY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    validation_path = args.source / "validation.npz"
    test_path = args.source / "test.npz"
    validation = np.load(validation_path)
    test = np.load(test_path)
    validation_returns = validation["actualLogReturns"][:, 0].astype(np.float64)
    test_returns = test["actualLogReturns"][:, 0].astype(np.float64)
    split_at = len(validation_returns) * 2 // 3
    candidates = []
    labels = {}
    for barrier in BARRIERS_BPS:
        for timeout in TIMEOUTS_SECONDS:
            labels[("validation", barrier, timeout)] = event_labels(validation_returns, barrier, timeout)
            labels[("test", barrier, timeout)] = event_labels(test_returns, barrier, timeout)
            terminal, duration, hit = labels[("validation", barrier, timeout)]
            fit_end = split_at - timeout
            select_start = split_at
            for feature_set in FEATURE_SETS:
                features = feature_matrix(validation, feature_set)[:len(terminal)]
                model = fit_logistic(features[WARMUP:fit_end], terminal[WARMUP:fit_end])
                result = metrics(model, features[select_start:], terminal[select_start:],
                                 duration[select_start:], hit[select_start:], timeout)
                candidates.append({
                    "barrierBps": barrier,
                    "timeoutSeconds": timeout,
                    "featureSet": feature_set,
                    "fitRows": fit_end - WARMUP,
                    "selection": result,
                })
    eligible = [row for row in candidates if row["selection"]["signInformationBitsPerActiveRow"] > 0
                and row["selection"]["blocks"] >= 4
                and row["selection"]["positiveInformationBlocks"] / row["selection"]["blocks"] >= 0.6]
    selected = max(eligible, key=lambda row: (row["selection"]["returnMseSkillVsTrainingMean"],
                                               row["selection"]["signInformationBitsPerActiveRow"]), default=None)
    final = None
    if selected is not None:
        barrier = selected["barrierBps"]
        timeout = selected["timeoutSeconds"]
        feature_set = selected["featureSet"]
        validation_terminal, _, _ = labels[("validation", barrier, timeout)]
        test_terminal, test_duration, test_hit = labels[("test", barrier, timeout)]
        validation_features = feature_matrix(validation, feature_set)[:len(validation_terminal)]
        test_features = feature_matrix(test, feature_set)[:len(test_terminal)]
        model = fit_logistic(validation_features[WARMUP:], validation_terminal[WARMUP:])
        final = {
            "specification": {"barrierBps": barrier, "timeoutSeconds": timeout, "featureSet": feature_set},
            "model": serializable_model(model),
            "test": metrics(model, test_features[WARMUP:], test_terminal[WARMUP:],
                            test_duration[WARMUP:], test_hit[WARMUP:], timeout),
        }
    summary = {
        "contract": "structured-sign-cost-sized-event-transfer-v1",
        "source": str(args.source),
        "sourceFiles": {
            "validation": {"path": str(validation_path), "sha256": sha256(validation_path)},
            "test": {"path": str(test_path), "sha256": sha256(test_path)},
        },
        "method": {
            "barriersBps": BARRIERS_BPS,
            "timeoutsSeconds": TIMEOUTS_SECONDS,
            "featureSets": FEATURE_SETS,
            "penalty": PENALTY,
            "selection": "Fit on the first two thirds of validation without crossing the split; require positive sign information in at least 60% of timeout-sized chronological blocks, then maximize return MSE skill on the final third. Refit only the selected specification on all validation rows before one test evaluation.",
            "event": "First completed-close crossing of a symmetric arithmetic-return barrier, otherwise the timeout close.",
            "costScreen": "Diagnostic independent round trips pay 12 bp on entry and 12 bp on exit. Rows and labels overlap and do not form an executable strategy.",
        },
        "candidates": candidates,
        "eligibleCandidates": len(eligible),
        "selected": final,
        "testConsultedForSelection": False,
        "elapsedSeconds": time.perf_counter() - started,
        "limitations": [
            "The original checkpoint was selected on the same validation date, so the inner chronological split does not create a pristine model-selection population.",
            "The exports cover only bounded prefixes of November 4 and 5, and overlapping event origins are dependent.",
            "This diagnoses feature transfer. It does not integrate the score into the full joint event law or account policy.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Structured sign transfer to cost-sized events",
        "",
        "Candidates were selected without test outcomes. Information is relative to the fit-period constant sign probability.",
        "",
        "| Barrier | Timeout | Features | Validation-tail sign bits/row | Positive blocks | Return MSE skill | Max forecast mean magnitude |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(candidates, key=lambda item: item["selection"]["signInformationBitsPerActiveRow"], reverse=True):
        metric = row["selection"]
        lines.append(f"| {row['barrierBps']} bp | {row['timeoutSeconds']} s | {row['featureSet']} | "
                     f"{metric['signInformationBitsPerActiveRow']:.6f} | {metric['positiveInformationBlocks']}/{metric['blocks']} | "
                     f"{metric['returnMseSkillVsTrainingMean'] * 100:.3f}% | {metric['maximumAbsoluteForecastMeanBps']:.3f} bp |")
    lines.extend(["", f"Eligible candidates: **{len(eligible)}**."])
    if final is None:
        lines.append("No candidate passed the predeclared validation stability gate; test metrics were not opened.")
    else:
        metric = final["test"]
        spec = final["specification"]
        lines.extend(["", f"Selected `{spec['featureSet']}` at {spec['barrierBps']} bp / {spec['timeoutSeconds']} s.", "",
                      f"Test sign information: **{metric['signInformationBitsPerActiveRow']:.6f} bits/active row**; "
                      f"return MSE skill: **{metric['returnMseSkillVsTrainingMean'] * 100:.3f}%**; "
                      f"maximum absolute forecast mean: **{metric['maximumAbsoluteForecastMeanBps']:.3f} bp**; "
                      f"24 bp round-trip actionable rows: **{metric['actionableRowsAt24BpsRoundTrip']}**."])
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"eligibleCandidates": len(eligible), "selected": final,
                      "elapsedSeconds": summary["elapsedSeconds"]}))


if __name__ == "__main__":
    main()
