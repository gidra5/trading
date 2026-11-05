"""Prespecified nonlinear sign/conditional-size experiment, CPU only."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import torch

from horizon_price_forecast import load_horizon_split
from sign_size_forecast import CONTRACT, DOWNSIDE_MAXIMUM, evaluate, magnitude_metrics, sign_metrics

HORIZONS = [900, 1800, 3600, 7200]
STRIDE = 8
PARAMETERS = {"learning_rate": .05, "max_iter": 150, "max_leaf_nodes": 15, "min_samples_leaf": 128,
              "l2_regularization": 1., "early_stopping": True, "validation_fraction": None,
              "n_iter_no_change": 12, "tol": 1e-7, "random_state": 1337, "categorical_features": None}


def write_json(file: Path, value: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def fit_head(kind: str, x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray):
    factory = HistGradientBoostingClassifier if kind == "sign" else HistGradientBoostingRegressor
    options = {**PARAMETERS, "scoring": "neg_log_loss" if kind == "sign" else "neg_mean_squared_error"}
    if kind != "sign":
        options["loss"] = "poisson"
    model = factory(**options).fit(x, y, X_val=xv, y_val=yv)
    # sklearn stops after patience; explicitly restore the validation-best
    # number of iterations by deterministic train-only refitting.
    best = int(np.argmax(model.validation_score_))
    completed = int(model.n_iter_)
    if best == 0:
        return None, {"completedIterations": completed, "bestIteration": 0, "bestIsInitialConstant": True}
    refit = factory(**{**options, "max_iter": best, "early_stopping": False}).fit(x, y)
    return refit, {"completedIterations": completed, "bestIteration": best, "trainOnlyRefit": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
        default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/training/runs/stochastic-crra-sign-size-v1"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = {"contract": CONTRACT, "dataset": str(args.dataset), "horizonsSeconds": HORIZONS,
        "trainingStrideSeconds": STRIDE, "validationSelectionStrideSeconds": STRIDE, "cpuThreads": 4,
        "features": "same 178 causal inputs: 120 observed return lags and 58 latest observed features",
        "splitBoundary": "same direct-horizon filter excludes every origin whose endpoint is outside its own chronological split",
        "parameters": PARAMETERS,
        "earlyStopping": "explicit chronological X_val/y_val; no random validation split; restore best iteration by deterministic train-only refit",
        "headSelection": "compare each fitted head to its train-only constant using validation log loss for sign and conditional MSE for magnitude; constant wins ties",
        "targetUnits": "sign of nonzero SIMPLE return; positive magnitude in bps scaled by its own sign-conditional training mean",
        "zeroAtom": "fixed train-only Jeffreys count estimate (zero+.5)/(N+1), using sampled training origins",
        "conditionalSignConstant": "fixed train-only Jeffreys up-given-active estimate",
        "magnitudeModel": "separate up/down Poisson-link HistGradientBoosting regressors; no sign-size independence assumption",
        "downsideDomain": {"maximumSimpleLoss": DOWNSIDE_MAXIMUM, "clipping": "upper clip only; record every affected prediction"},
        "law": "weights ((1-p0)(1-p_up),p0,(1-p0)p_up) on simple returns (-m_minus,0,+m_plus)",
        "mean": "(1-p0)*(p_up*m_plus-(1-p_up)*m_minus)",
        "averageAbsoluteSize": "(1-p0)*(p_up*m_plus+(1-p_up)*m_minus)",
        "ablations": ["full", "signConstantConditionalSizes", "signCommonConstantSize", "magnitudeOnlyFairSign"],
        "fairSignControl": "same pooled predicted magnitude on both sides, p_up=.5, equal clipping if required; exact zero signed mean",
        "testProtocol": "freeze every head choice for all four horizons before loading any test data; report all horizons, no test-based winner",
        "testEvaluationStrideSeconds": STRIDE, "replayObservedSeconds": 151200,
        "replayCounts": {"900": 168, "1800": 84, "3600": 42, "7200": 21},
        "testUsedForSelection": False, "tradeFrequencyConstraint": None,
        "distributionLimitation": "three-point law approximates within-sign spread and nonlinear capped-close values; not recovery of a continuous law or price path"}
    write_json(args.output / "plan.json", plan)
    started = time.monotonic()
    selections = []
    artifacts = {}
    for horizon in HORIZONS:
        directory = args.output / f"h{horizon}"
        if (directory / "model.joblib").exists():
            raise ValueError(f"refusing to overwrite fitted model {directory / 'model.joblib'}")
        train = load_horizon_split(args.dataset, "train", horizon)
        validation = load_horizon_split(args.dataset, "validation", horizon)
        tx, vx = train.features(train.origins[::STRIDE]), validation.features(validation.origins[::STRIDE])
        ty, vy = np.expm1(train.targets()[::STRIDE].astype(np.float64))*1e4, np.expm1(validation.targets()[::STRIDE].astype(np.float64))*1e4
        ta, va = ty != 0, vy != 0
        tp, tn, vp, vn = ty > 0, ty < 0, vy > 0, vy < 0
        constants = {"pZero": float(((~ta).sum()+.5)/(len(ty)+1)),
                     "pUpGivenActive": float((tp.sum()+.5)/(ta.sum()+1)),
                     "upMeanBps": float(ty[tp].mean()), "downMeanBps": float(-ty[tn].mean()),
                     "pooledActiveMeanBps": float(np.abs(ty[ta]).mean())}
        models, choices, heads = {}, {}, {}
        for head, mask, vmask, target, vtarget in (
            ("sign", ta, va, tp, vp),
            ("up", tp, vp, ty/constants["upMeanBps"], vy/constants["upMeanBps"]),
            ("down", tn, vn, -ty/constants["downMeanBps"], -vy/constants["downMeanBps"]),
        ):
            print(json.dumps({"stage": "fitting", "horizon": horizon, "head": head,
                              "trainingExamples": int(mask.sum()), "validationExamples": int(vmask.sum())}), flush=True)
            model, iteration = fit_head(head, tx[mask], target[mask], vx[vmask], vtarget[vmask])
            models[head] = model
            if head == "sign":
                constant_metric = sign_metrics(vp[va], np.full(int(va.sum()), constants["pUpGivenActive"]))
                model_metric = sign_metrics(vp[va], model.predict_proba(vx[va])[:, 1]) if model else None
                wins = model_metric is not None and model_metric["logLoss"] < constant_metric["logLoss"]
            else:
                scale = constants[f"{head}MeanBps"]
                actual_bps = vtarget[vmask]*scale
                constant_metric = magnitude_metrics(actual_bps, np.full(len(actual_bps), scale), scale)
                prediction = model.predict(vx[vmask])*scale if model else None
                raw_clipped = int((prediction > DOWNSIDE_MAXIMUM*1e4).sum()) if model and head == "down" else 0
                if prediction is not None and head == "down":
                    prediction = np.minimum(prediction, DOWNSIDE_MAXIMUM*1e4)
                model_metric = magnitude_metrics(actual_bps, prediction, scale) if model else None
                if model_metric is not None:
                    model_metric["downsideClippedCount"] = raw_clipped
                wins = model_metric is not None and model_metric["mseBpsSquared"] < constant_metric["mseBpsSquared"]
            choices[head] = "boosted" if wins else "constant"
            heads[head] = {"selected": choices[head], "iterations": iteration,
                           "constantValidation": constant_metric, "boostedValidation": model_metric}
            print(json.dumps({"stage": "selected", "horizon": horizon, "head": head,
                              "choice": choices[head], "iterations": iteration}), flush=True)
        artifact = {"contract": CONTRACT, "horizonSeconds": horizon, "constants": constants,
                    "selectedHeads": choices, "models": models, "trainingStrideSeconds": STRIDE}
        artifacts[horizon] = artifact
        directory.mkdir(parents=True, exist_ok=True)
        model_file = directory / "model.joblib"
        joblib.dump(artifact, model_file)
        nonoverlap_x = validation.features(validation.origins[::horizon])
        nonoverlap_y = np.expm1(validation.targets()[::horizon].astype(np.float64))*1e4
        evaluation = evaluate(artifact, vx, vy, nonoverlap_x, nonoverlap_y)
        selected = {"contract": CONTRACT, "horizonSeconds": horizon, "selectedHeads": choices,
                    "headValidation": heads, "constants": constants, "validation": evaluation,
                    "modelFile": str(model_file), "modelSha256": sha(model_file),
                    "trainingExamples": len(ty), "validationSelectionExamples": len(vy),
                    "testUsedForSelection": False, "testLoaded": False,
                    "firstTrainingOriginMs": int(train.source.axis_start_ms+train.origins[0]*1000),
                    "lastTrainingEndpointMs": int(train.source.axis_start_ms+(train.origins[::STRIDE][-1]+horizon)*1000),
                    "firstValidationOriginMs": int(validation.source.axis_start_ms+validation.origins[0]*1000),
                    "lastValidationEndpointMs": int(validation.source.axis_start_ms+(validation.origins[::STRIDE][-1]+horizon)*1000)}
        write_json(directory / "selection.json", selected)
        selections.append(selected)
        print(json.dumps({"stage": "validationComplete", "horizon": horizon, "selectedHeads": choices,
                          "economics": evaluation["ablations"]["full"]["nonoverlapping"]}), flush=True)
    frozen = {"contract": CONTRACT, "plan": plan, "horizons": selections,
              "allHeadSelectionsFrozen": True, "testLoaded": False, "testUsedForSelection": False}
    write_json(args.output / "selection.json", frozen)
    tests = []
    for horizon in HORIZONS:
        test = load_horizon_split(args.dataset, "test", horizon)
        count = 151200//horizon
        origins = test.origins[::horizon][:count]
        if len(origins) != count:
            raise ValueError("incomplete common 42-hour test interval")
        x = test.features(test.origins[::STRIDE])
        y = np.expm1(test.targets()[::STRIDE].astype(np.float64))*1e4
        nx = test.features(origins)
        ny = np.expm1(test.targets()[::horizon][:count].astype(np.float64))*1e4
        evaluation = evaluate(artifacts[horizon], x, y, nx, ny)
        result = {"horizonSeconds": horizon, "selectedHeads": artifacts[horizon]["selectedHeads"],
                  "evaluationStrideSeconds": STRIDE, "headSelectionsFrozenBeforeTest": True,
                  "testUsedForSelection": False, "evaluation": evaluation,
                  "nonoverlapFirstTimestamp": int(test.source.axis_start_ms+origins[0]*1000),
                  "nonoverlapTerminalTimestamp": int(test.source.axis_start_ms+(origins[-1]+horizon)*1000)}
        write_json(args.output / f"h{horizon}" / "test.json", result)
        tests.append(result)
        print(json.dumps({"stage": "testComplete", "horizon": horizon,
                          "full": evaluation["ablations"]["full"],
                          "signConstant": evaluation["ablations"]["signConstantConditionalSizes"]}), flush=True)
    write_json(args.output / "result.json", {"selection": frozen, "test": tests,
        "seconds": time.monotonic()-started, "datasetManifestSha256": sha(args.dataset / "manifest.json")})


if __name__ == "__main__":
    main()
