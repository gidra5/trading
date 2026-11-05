"""Matched-origin marginal audit of frozen Gaussian joint laws and specialists.

No training, calibration, model selection, or policy selection occurs here.
These available joint densities are not the deleted historical 7% point model.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, ndtri
import torch

from horizon_price_forecast import load_horizon_forecast
from sign_size_forecast import DOWNSIDE_MAXIMUM, magnitude_metrics, sign_metrics
from stochastic_price_forecast import load_timeline_split
from train_neural_sign_size_forecast import HORIZONS, examples, load_selected, predict, select_origins, sha, write_json

CONTRACT = "frozen-joint-versus-specialists-matched-marginal-audit-v1"
FAMILIES = {1: "stochastic-crra-direct-horizons-gaussian-v2",
            900: "stochastic-crra-direct-horizons-5m15m-v1",
            1800: "stochastic-crra-horizon-search-v1", 7200: "stochastic-crra-horizon-search-v1"}


def side_integrals(model, positive: bool, order: int) -> tuple[np.ndarray, np.ndarray]:
    if model.basis.get("coordinate") != "gaussian-cdf":
        raise ValueError("simple-return moment audit requires the finite-moment Gaussian-CDF law")
    knots = np.asarray(model.basis["knots"], dtype=np.float64)
    areas = np.asarray(model.basis["basisAreas"], dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    lower = np.maximum(knots[:-1], .5) if positive else knots[:-1]
    upper = knots[1:] if positive else np.minimum(knots[1:], .5)
    half = np.maximum(upper-lower, 0)/2
    middle = np.where(half > 0, (upper+lower)/2, .5)
    unit = middle[:, None]+half[:, None]*nodes
    fraction = (unit-knots[:-1, None])/np.diff(knots)[:, None]
    magnitude = np.abs(np.expm1(model.sigma*ndtri(unit)))
    results = []
    for value in (np.ones_like(unit), magnitude):
        integral = np.zeros(len(knots))
        integral[:-1] += np.sum(value*(1-fraction)*weights, axis=1)*half
        integral[1:] += np.sum(value*fraction*weights, axis=1)*half
        results.append(integral/areas)
    return tuple(results)


def decompose(masses: np.ndarray, up_mass: np.ndarray, down_mass: np.ndarray,
              up_moment: np.ndarray, down_moment: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not np.isfinite(masses).all() or np.any(masses < 0) or np.any(masses.sum(axis=1) <= 0):
        raise ValueError("invalid joint mixture weights")
    normalized = masses/masses.sum(axis=1, keepdims=True)
    pu, pd = normalized@up_mass, normalized@down_mass
    positive, negative = normalized@up_moment, normalized@down_moment
    np.testing.assert_allclose(pu+pd, 1, atol=3e-12, rtol=0)
    if np.any(pu < -1e-14) or np.any(pd < -1e-14):
        raise ValueError("negative sign mass")
    pu, pd = np.maximum(pu, 0), np.maximum(pd, 0)
    if np.any((pu == 0) & (positive != 0)) or np.any((pd == 0) & (negative != 0)):
        raise ValueError("nonzero first moment on a zero-mass side")
    # Conditional size on an impossible side is unidentified; zero is a declared
    # placeholder that contributes exactly zero to the reconstructed moments.
    up = np.divide(positive, pu, out=np.zeros_like(pu), where=pu > 0)
    down = np.divide(negative, pd, out=np.zeros_like(pd), where=pd > 0)
    reconstructed = pu*up-pd*down
    return pu, up, down, {"undefinedUpSizeCount": int((pu == 0).sum()), "undefinedDownSizeCount": int((pd == 0).sum()),
        "minimumActiveUpProbability": float(pu.min()), "maximumActiveUpProbability": float(pu.max()),
        "activeMeanReconstructionMaximumError": float(np.max(np.abs(reconstructed-(positive-negative)))),
        "downsideOutsidePositivePriceSupportCount": int((down >= 1).sum())}


def score_components(y: np.ndarray, zero: np.ndarray | float, p_up: np.ndarray,
                     up: np.ndarray, down: np.ndarray, c: dict) -> dict:
    zero, p_up, up, down = np.broadcast_arrays(np.asarray(zero, dtype=np.float64), p_up, up, down)
    if any(not np.isfinite(v).all() for v in (y,zero,p_up,up,down)) or np.any((zero < 0) | (zero > 1)) \
            or np.any((p_up < -1e-12) | (p_up > 1+1e-12)) or np.any(up < 0) or np.any(down < 0):
        raise ValueError("invalid marginal components")
    active, positive, negative = y != 0, y > 0, y < 0
    if not positive.any() or not negative.any():
        raise ValueError("both realized polarities required for matched conditional scores")
    clipped_down = np.minimum(down, DOWNSIDE_MAXIMUM)
    p_up = np.clip(p_up, 0, 1)
    conditional_mean = p_up*up-(1-p_up)*clipped_down
    mean = (1-zero)*conditional_mean
    absolute = (1-zero)*(p_up*up+(1-p_up)*clipped_down)
    p_score = np.clip(p_up, 1e-12, 1-1e-12)
    zero_score = np.clip(zero, 1e-12, 1-1e-12)
    signed_baseline = float(np.square(y).mean())
    active_baseline = float(np.square(y[active]).mean())
    abs_constant = (1-c["pZero"])*c["pooledActiveMeanSimple"]
    abs_baseline = float(np.square(np.abs(y)-abs_constant).mean())
    signed_error = np.square(mean-y)
    signed_mse = float(signed_error.mean())
    abs_mse = float(np.square(absolute-np.abs(y)).mean())
    conditional_active_mse = float(np.square(conditional_mean[active]-y[active]).mean())
    deployed_active_mse = float(signed_error[active].mean())
    correct = (p_up[active] >= .5) == positive[active]
    return {"examples": len(y), "activeExamples": int(active.sum()), "zeroExamples": int((~active).sum()),
        "signGivenActive": sign_metrics(positive[active], p_score[active]),
        "signTrainConstant": sign_metrics(positive[active], np.full(int(active.sum()), c["pUpGivenActive"])),
        "signMagnitudeWeightedAccuracy": float(np.dot(np.abs(y[active]), correct)/np.abs(y[active]).sum()),
        "positiveSize": magnitude_metrics(y[positive]*1e4, up[positive]*1e4, c["upMeanSimple"]*1e4),
        "negativeSize": magnitude_metrics(-y[negative]*1e4, clipped_down[negative]*1e4, c["downMeanSimple"]*1e4),
        "allTickSignedMeanMseBpsSquared": signed_mse*1e8, "allTickSignedMeanMseSkillVsZero": 1-signed_mse/signed_baseline,
        "allTickAbsoluteSizeMseBpsSquared": abs_mse*1e8, "allTickAbsoluteSizeMseSkillVsTrainConstant": 1-abs_mse/abs_baseline,
        "activeOnly": {"scope": "score only realized nonzero endpoints; conditional-active forecast is a diagnostic with an unavailable future-activity gate",
            "conditionalActiveMeanMseBpsSquared": conditional_active_mse*1e8,
            "conditionalActiveMeanMseSkillVsZero": 1-conditional_active_mse/active_baseline,
            "deployedUnconditionalMeanMseBpsSquared": deployed_active_mse*1e8,
            "deployedUnconditionalMeanMseSkillVsZero": 1-deployed_active_mse/active_baseline},
        "zeroOnlyMeanSquaredPredictionBpsSquared": float(np.square(mean[~active]).mean()*1e8) if (~active).any() else None,
        "zeroTickContributionToTotalMseBpsSquared": float(np.sum(np.square(mean[~active]))/len(y)*1e8),
        "predictedZeroProbabilityMean": float(zero.mean()), "observedZeroFrequency": float((~active).mean()),
        "zeroBrier": float(np.square(zero-(~active)).mean()),
        "zeroLogLoss": float(-np.where(~active, np.log(zero_score), np.log1p(-zero_score)).mean()),
        "predictedAbsoluteMeanBps": float(absolute.mean()*1e4), "predictedSignedMeanBps": float(mean.mean()*1e4),
        "probabilityNumericalScoringClipCount": int(((p_up < 1e-12) | (p_up > 1-1e-12)).sum()),
        "downsideClipCount": int((down > DOWNSIDE_MAXIMUM).sum())}


@torch.no_grad()
def density_components(model, features: np.ndarray, batch: int, device: torch.device) -> tuple[np.ndarray,np.ndarray]:
    zero, masses = [], []
    for start in range(0, len(features), batch):
        p0, density = model.probabilities(torch.from_numpy(features[start:start+batch]).to(device))
        zero.append(p0)
        masses.append(density)
    return np.concatenate(zero), np.concatenate(masses)


def chosen_densities(h: int) -> dict:
    directory = Path("data/training/runs")/FAMILIES[h]/f"h{h}"
    selection_path = directory/"model-selection.json"
    selected = json.loads(selection_path.read_text())
    if selected["testUsedForSelection"]:
        raise ValueError("joint selection reports test use")
    checkpoint = Path(selected["selectedCheckpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    constant_output = not bool(torch.count_nonzero(payload["model"]["output.weight"]))
    candidates = [{"role": "previouslySelectedDensity", "checkpoint": str(checkpoint), "sha256": sha(checkpoint),
                   "selectedModel": selected["selectedModel"], "initializedEpochZero": payload.get("selectedEpoch") == 0,
                   "constantOutputLayer": constant_output}]
    available_trained = []
    if constant_output:
        for file in sorted(directory.glob("lambda*/best.pt")):
            saved = torch.load(file, map_location="cpu", weights_only=False)
            available_trained.append({"checkpoint":str(file), "selectedEpoch":saved.get("selectedEpoch"),
                "trainedConditionalAvailable": saved.get("selectedEpoch",0) > 0 and bool(torch.count_nonzero(saved["model"]["output.weight"]))})
    if selected["selectedModel"] != "conditionalGlu":
        # The already-selected loss weight determines which trained candidate to
        # inspect. This is a diagnostic, not a new winner chosen by this audit.
        candidate = checkpoint.parent/"best.pt"
        candidates.append({"role": "conditionalCandidateAtPreviouslySelectedLossWeight", "checkpoint": str(candidate),
                           "sha256": sha(candidate), "selectedModel": "conditionalGlu"})
    return {"horizonSeconds": h, "selectionFile": str(selection_path), "selectionSha256": sha(selection_path),
            "selection": selected, "candidates": candidates, "savedConditionalCandidateInventory": available_trained,
            "missingTrainedConditionalReason": "only initialized epoch0 best checkpoints are retained; later trained states were not saved"
                if constant_output and available_trained and not any(x["trainedConditionalAvailable"] for x in available_trained) else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialists", type=Path, default=Path("data/training/runs/stochastic-crra-neural-specialists-v1/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-neural-specialists/joint-marginal-audit.json"))
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2560)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    frozen = json.loads(args.specialists.read_text())
    if not frozen["allHeadSelectionsFrozen"] or frozen["testUsedForSelection"]:
        raise ValueError("specialist selection must already be frozen without test")
    plan = frozen["plan"]
    dataset = Path(plan["dataset"])
    if sha(dataset/"manifest.json") != plan["datasetManifestSha256"]:
        raise ValueError("current dataset manifest differs from frozen specialist training data")
    density_plans = [chosen_densities(h) for h in HORIZONS]
    output = {"contract": CONTRACT, "createdAt": datetime.now(timezone.utc).isoformat(),
        "specialistSelection": str(args.specialists), "specialistSelectionSha256": sha(args.specialists),
        "datasetManifestSha256": plan["datasetManifestSha256"],
        "densityPlans": density_plans, "changesTrainingOrSelection": False,
        "comparisonProvenance": "available frozen Gaussian-CDF joint density GLUs, NOT the deleted historical direct59 point/MSE model with 7% score",
        "inputs": "joint laws retain their causal178 input; specialists retain their two production59 rows; identical forecast origins/outcomes, not identical architecture or training objective",
        "sampling": "exact specialist common-origin stride4 validation/test with every endpoint through7200s complete; reused test benchmark; no independent-context claim",
        "evaluationBatchSize": args.batch_size, "matmulPrecision": "high; original specialist inference batch reproduced",
        "target": "SIMPLE price return from actual endpoint prices; all-tick and realized-active diagnostics separately labeled",
        "baselines": "both models use same specialist training constants and identical zero-return/absolute-size baselines",
        "quadrature": "piecewise triangular density integrated in Gaussian-CDF coordinate, split at zero;384point primary with192point convergence check; numerical not symbolic integration proof",
        "crossedComponents": "joint and specialist sign/conditional sizes crossed at identical specialist fixed p0; separate native joint p0 and crosszero ablation",
        "selectedModelChanges": None, "results": []}
    write_json(args.output.with_suffix(".plan.json"), output)
    started = time.monotonic()
    data = {}
    for name in ("validation","test"):
        source = load_timeline_split(dataset, name)
        origins = select_origins(source, plan["strideSeconds"], plan["validationLimit"] if name == "validation" else None)
        data[name] = (source, examples(source, origins))
    for j, density_plan in enumerate(density_plans):
        h, entry = density_plan["horizonSeconds"], frozen["horizons"][j]
        if entry["horizonSeconds"] != h:
            raise ValueError("horizon order mismatch")
        specialist_predictions = {}
        for name, (_, ex) in data.items():
            sign = load_selected(entry["sign"], device)
            pu = expit(predict(sign, ex.features, args.batch_size, device)[:, j])
            del sign
            size = load_selected(entry["size"], device)
            normalized = predict(size, ex.features, args.batch_size, device)[:, j]
            del size
            c = entry["constants"]
            specialist_predictions[name] = (pu, normalized[:,0]*c["upMeanSimple"], normalized[:,1]*c["downMeanSimple"])
        for candidate in density_plan["candidates"]:
            model, payload = load_horizon_forecast(Path(candidate["checkpoint"]), device)
            if Path(payload["plan"]["dataset"]).resolve() != dataset.resolve() or payload["plan"]["horizonSeconds"] != h:
                raise ValueError("joint checkpoint dataset/horizon mismatch")
            integral = {}
            for order in (192,384):
                um, us = side_integrals(model, True, order)
                dm, ds = side_integrals(model, False, order)
                np.testing.assert_allclose(um+dm, 1, atol=3e-12, rtol=0)
                integral[order] = (um,dm,us,ds)
            original = model._component_moments("excess", 384)
            coefficient_error = float(np.max(np.abs(integral[384][2]-integral[384][3]-original)))
            row = {"horizonSeconds": h, **candidate, "densitySelectedEpoch": payload.get("selectedEpoch"),
                "densityTrainingPlan": payload["plan"], "specialistSign": entry["sign"], "specialistSize": entry["size"],
                "commonConstants": c, "coefficientSignedMomentReconstructionMaxError": coefficient_error, "splits": {}}
            for name, (source, ex) in data.items():
                zero, masses = density_components(model, source.features(ex.origins), args.batch_size, device)
                jp, ju, jd, diagnostics = decompose(masses, *integral[384])
                lp, lu, ld, _ = decompose(masses, *integral[192])
                sp, su, sd = specialist_predictions[name]
                normalized_masses = masses/masses.sum(axis=1,keepdims=True)
                direct = (1-zero)*(normalized_masses@original)
                reconstructed = (1-zero)*(jp*ju-(1-jp)*jd)
                low = (1-zero)*(lp*lu-(1-lp)*ld)
                diagnostics.update({"unconditionalSignedMeanReconstructionMaxError": float(np.max(np.abs(direct-reconstructed))),
                    "quadrature192vs384MaxSignedMeanBps": float(np.max(np.abs(reconstructed-low))*1e4),
                    "quadrature192vs384MaxUpSizeBps": float(np.max(np.abs(ju-lu))*1e4),
                    "quadrature192vs384MaxDownSizeBps": float(np.max(np.abs(jd-ld))*1e4)})
                if diagnostics["unconditionalSignedMeanReconstructionMaxError"] > 1e-8:
                    raise ValueError("joint moment reconstruction failed")
                mixtures = {
                    "specialists": (c["pZero"],sp,su,sd), "jointNativeZero": (zero,jp,ju,jd),
                    "jointFixedZero": (c["pZero"],jp,ju,jd),
                    "jointSignSpecialistSizesFixedZero": (c["pZero"],jp,su,sd),
                    "specialistSignJointSizesFixedZero": (c["pZero"],sp,ju,jd),
                    "specialistsWithJointZero": (zero,sp,su,sd),
                    "allTrainConstants": (c["pZero"],np.full(len(sp),c["pUpGivenActive"]),
                        np.full(len(sp),c["upMeanSimple"]),np.full(len(sp),c["downMeanSimple"]))}
                scored = {key: score_components(ex.targets[:,j], *components, c) for key,components in mixtures.items()}
                row["splits"][name] = {"firstTimestamp": int(ex.axis_start_ms+ex.origins[0]*1000),
                    "lastOriginTimestamp": int(ex.axis_start_ms+ex.origins[-1]*1000),
                    "examples": len(ex.origins), "diagnostics": diagnostics, "mixtures": scored}
                print(json.dumps({"horizon":h,"role":candidate["role"],"split":name,
                    "selectedEpoch":payload.get("selectedEpoch"),"metrics": {key: {"signAccuracy":m["signGivenActive"]["accuracy"],
                    "signLogLoss":m["signGivenActive"]["logLoss"],"signedMeanSkill":m["allTickSignedMeanMseSkillVsZero"],
                    "absoluteSizeSkill":m["allTickAbsoluteSizeMseSkillVsTrainConstant"],
                    "activeConditionalMeanSkill":m["activeOnly"]["conditionalActiveMeanMseSkillVsZero"]}
                    for key,m in scored.items()}}),flush=True)
            output["results"].append(row)
            output["seconds"] = time.monotonic()-started
            write_json(args.output, output)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
