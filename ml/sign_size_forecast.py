"""Separate sign and sign-conditional SIMPLE-return magnitude forecasts.

The executable three-point law preserves the supplied sign-conditional first
moments. It deliberately does not model within-sign spread or path crossings.
"""
from __future__ import annotations

import numpy as np

CONTRACT = "causal-production178-sign-conditional-simple-size-three-point-v1"
DOWNSIDE_MAXIMUM = 1 - 1e-12


def three_point_law(p_zero: float, p_up: np.ndarray, up_size: np.ndarray,
                    down_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    p_up, up_size, down_size = np.broadcast_arrays(np.asarray(p_up, dtype=np.float64),
        np.asarray(up_size, dtype=np.float64), np.asarray(down_size, dtype=np.float64))
    if p_up.ndim != 1 or not 0 <= p_zero <= 1:
        raise ValueError("one-dimensional forecasts and a valid zero-atom probability required")
    if any(not np.isfinite(value).all() for value in (p_up, up_size, down_size)):
        raise ValueError("nonfinite sign or magnitude forecast")
    if np.any((p_up < 0) | (p_up > 1)) or np.any(up_size < 0) or np.any(down_size < 0):
        raise ValueError("invalid probability or negative magnitude")
    clipped = down_size > DOWNSIDE_MAXIMUM
    down = np.minimum(down_size, DOWNSIDE_MAXIMUM)
    probability = np.stack(((1-p_zero)*(1-p_up), np.full(len(p_up), p_zero), (1-p_zero)*p_up), axis=1)
    returns = np.stack((-down, np.zeros(len(p_up)), up_size), axis=1)
    return returns, probability, {"downsideClippedCount": int(clipped.sum()),
                                 "maximumRawDownsideSize": float(down_size.max()) if len(down_size) else None}


def moment_values(returns: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if returns.shape != probabilities.shape or returns.ndim != 2 or returns.shape[1] != 3:
        raise ValueError("three-point law arrays must have equal shape")
    return (np.sum(returns * probabilities, axis=1),
            np.sum(np.abs(returns) * probabilities, axis=1))


def predict_components(artifact: dict, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if artifact["contract"] != CONTRACT:
        raise ValueError("unsupported sign-size model")
    constants, selected, models = artifact["constants"], artifact["selectedHeads"], artifact["models"]
    p_up = (models["sign"].predict_proba(features)[:, 1] if selected["sign"] == "boosted"
            else np.full(len(features), constants["pUpGivenActive"]))
    up = (models["up"].predict(features) * constants["upMeanBps"] if selected["up"] == "boosted"
          else np.full(len(features), constants["upMeanBps"])) / 1e4
    down = (models["down"].predict(features) * constants["downMeanBps"] if selected["down"] == "boosted"
            else np.full(len(features), constants["downMeanBps"])) / 1e4
    return p_up, up, down


def predict_laws(artifact: dict, features: np.ndarray) -> dict:
    p_up, up, down = predict_components(artifact, features)
    constant = artifact["constants"]
    p0 = constant["pZero"]
    pooled_prediction = p_up * up + (1-p_up) * down
    pooled_clipped = np.minimum(pooled_prediction, DOWNSIDE_MAXIMUM)
    pooled_constant = min(constant["pooledActiveMeanBps"] / 1e4, DOWNSIDE_MAXIMUM)
    laws = {
        "full": three_point_law(p0, p_up, up, down),
        "signConstantConditionalSizes": three_point_law(p0, p_up,
            np.full(len(p_up), constant["upMeanBps"] / 1e4),
            np.full(len(p_up), constant["downMeanBps"] / 1e4)),
        "signCommonConstantSize": three_point_law(p0, p_up,
            np.full(len(p_up), pooled_constant), np.full(len(p_up), pooled_constant)),
        "magnitudeOnlyFairSign": three_point_law(p0, np.full(len(p_up), .5), pooled_clipped, pooled_clipped),
    }
    laws["magnitudeOnlyFairSign"][2]["symmetricPooledClippedCount"] = int((pooled_prediction > DOWNSIDE_MAXIMUM).sum())
    laws["signCommonConstantSize"][2]["symmetricConstantClippedCount"] = int(constant["pooledActiveMeanBps"] / 1e4 > DOWNSIDE_MAXIMUM) * len(p_up)
    return laws


def sign_metrics(target_up: np.ndarray, p_up: np.ndarray) -> dict:
    if not len(target_up) or np.any((p_up <= 0) | (p_up >= 1)):
        raise ValueError("sign scoring requires active examples and interior probabilities")
    return {"examples": len(target_up),
            "logLoss": float(-np.where(target_up, np.log(p_up), np.log1p(-p_up)).mean()),
            "brier": float(np.square(p_up-target_up).mean()),
            "accuracy": float(((p_up >= .5) == target_up).mean()),
            "predictedUpProbability": float(p_up.mean()), "observedUpFrequency": float(target_up.mean())}


def magnitude_metrics(actual: np.ndarray, forecast: np.ndarray, constant: float) -> dict:
    baseline_mse = float(np.square(actual-constant).mean())
    mse = float(np.square(actual-forecast).mean())
    return {"examples": len(actual), "mseBpsSquared": mse,
            "constantMseBpsSquared": baseline_mse,
            "mseSkillVsTrainConstant": 1-mse/baseline_mse if baseline_mse else None,
            "meanPredictedBps": float(forecast.mean()), "meanRealizedBps": float(actual.mean())}


def evaluate(artifact: dict, features: np.ndarray, target_bps: np.ndarray,
             nonoverlap_features: np.ndarray, nonoverlap_target_bps: np.ndarray) -> dict:
    constants = artifact["constants"]
    p_up, up, down = predict_components(artifact, features)
    active, positive, negative = target_bps != 0, target_bps > 0, target_bps < 0
    result = {
        "examples": len(target_bps), "signGivenActive": sign_metrics(positive[active], p_up[active]),
        "signTrainConstant": sign_metrics(positive[active], np.full(int(active.sum()), constants["pUpGivenActive"])),
        "positiveSize": magnitude_metrics(target_bps[positive], up[positive]*1e4, constants["upMeanBps"]),
        "negativeSize": magnitude_metrics(-target_bps[negative],
            np.minimum(down[negative], DOWNSIDE_MAXIMUM)*1e4, constants["downMeanBps"]),
        "observedZeroFrequency": float((~active).mean()), "fixedZeroProbability": constants["pZero"],
        "ablations": {},
    }
    zero_mse = float(np.square(target_bps).mean())
    abs_constant = (1-constants["pZero"])*constants["pooledActiveMeanBps"]
    abs_baseline_mse = float(np.square(np.abs(target_bps)-abs_constant).mean())
    laws = predict_laws(artifact, features)
    nonoverlap_laws = predict_laws(artifact, nonoverlap_features)
    for name, (returns, weights, clipping) in laws.items():
        mean, size = moment_values(returns, weights)
        means_bps, sizes_bps = mean*1e4, size*1e4
        mse = float(np.square(means_bps-target_bps).mean())
        size_mse = float(np.square(sizes_bps-np.abs(target_bps)).mean())
        nreturns, nweights, nclipping = nonoverlap_laws[name]
        nmean, nsize = moment_values(nreturns, nweights)
        threshold = 1/.9995**2 - 1
        choose_long = nmean > threshold
        actual_net = .9995**2*(1+nonoverlap_target_bps/1e4)-1
        result["ablations"][name] = {
            "expectedMeanReturnBps": float(means_bps.mean()), "expectedAverageAbsoluteSizeBps": float(sizes_bps.mean()),
            "realizedAverageAbsoluteSizeBps": float(np.abs(target_bps).mean()),
            "signedPriceMseBpsSquared": mse, "signedPriceMseSkillVsZero": 1-mse/zero_mse if zero_mse else None,
            "averageSizeMseBpsSquared": size_mse,
            "averageSizeMseSkillVsTrainConstant": 1-size_mse/abs_baseline_mse if abs_baseline_mse else None,
            "clipping": clipping,
            "nonoverlapping": {"contexts": len(nmean), "predictedMinBps": float(nmean.min()*1e4),
                "predictedMaxBps": float(nmean.max()*1e4), "expectedAverageSizeBps": float(nsize.mean()*1e4),
                "chooseLongCount": int(choose_long.sum()), "strictCashCount": int((nmean < threshold).sum()),
                "strictDecisionSwitching": bool(choose_long.any() and (nmean < threshold).any()),
                "realizedMeanNetWhenPredictedLongBps": float(actual_net[choose_long].mean()*1e4) if choose_long.any() else None,
                "independentCashStartPolicyMeanNetBps": float(np.where(choose_long, actual_net, 0).mean()*1e4),
                "clipping": nclipping,
                "scope": "full-close cash-start threshold diagnostic, not compounded account backtest"},
        }
    return result
