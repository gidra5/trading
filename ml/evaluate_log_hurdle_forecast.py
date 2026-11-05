"""Matched evaluation of freshly trained log-return hurdle specialists.

No fitting or selection occurs here. Old canonical issuance times and observed
prices define the comparison. Old, already frozen calibration settings are
reused on each new checkpoint, with a fresh causal state for every split.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
import torch

from audit_hurdle_log_returns import log_means as retained_log_means
from audit_joint_log_returns import score_means
from evaluate_online_hurdle_calibration import CachedStream, immutable
from hurdle_sign_size_forecast import hurdle_nll_elements
from log_hurdle_sign_size_forecast import (CONTRACT as MODEL_CONTRACT,
    MINIMUM_MEAN, MAXIMUM_UP_MEAN_RATIO, log_hurdle_components, log_hurdle_gross_components, log_hurdle_log_probabilities,
    log_hurdle_nll_elements, log_hurdle_simple_components, log_hurdle_simple_mean)
from online_log_hurdle_calibration import CalibrationConfig, OnlineLogHurdleCalibrator
from stochastic_price_forecast import load_timeline_split
from train_neural_sign_size_forecast import HORIZONS, now, select_origins, sha, write_json

CONTRACT = "matched-log-hurdle-evaluation-v1"
STAGES = ("pretrained", "finetuned")


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def simulate_log_stream(stream: CachedStream, config: CalibrationConfig, *, trace: bool = False) -> dict:
    """Use the established issue/maturity schedule, passing LOG labels to updates.

`CachedStream.target` is the exact endpoint SIMPLE return. `log1p` deliberately
matches the archived log-return audits' targets and their sample hashes.
Accuracy indices never enter the event loop and do not affect its state.
"""
    if config.horizon_seconds != stream.horizon_seconds:
        raise ValueError("calibration and stream horizon differ")
    if np.any(np.diff(stream.timestamps) != config.cadence_seconds*1000):
        raise ValueError("canonical issuance cadence changed")
    calibrator = OnlineLogHurdleCalibrator(config)
    count, pending = len(stream.origins), 0
    parameters = np.empty_like(stream.raw)
    snapshots = {"coefficients": np.empty((count, 4)), "discountedInformation": np.empty((count, 4)),
        "componentUpdates": np.empty((count, 4), dtype=np.int64),
        "boundaryUpdates": np.empty((count, 4), dtype=np.int64),
        "latestMaturityUsed": np.empty(count, dtype=np.int64),
        "maturedObservations": np.empty(count, dtype=np.int64)} if trace else {}
    maturities = stream.timestamps+stream.horizon_seconds*1000
    for index, stamp in enumerate(stream.timestamps):
        while pending < index and maturities[pending] <= stamp:
            log_target = float(np.log1p(stream.target(pending)))
            calibrator.update(int(stream.timestamps[pending]), stream.raw[pending], log_target)
            pending += 1
        parameters[index] = calibrator.apply(int(stamp), stream.raw[index])
        if trace:
            snapshot = calibrator.snapshot(int(stamp))
            expected_last = None if pending == 0 else int(maturities[pending-1])
            if snapshot["maturedObservations"] != pending or snapshot["latestMaturityUsed"] != expected_last:
                raise ValueError("calibrator and canonical maturity stream disagree")
            for key in snapshots:
                snapshots[key][index] = -1 if snapshot[key] is None else snapshot[key]
    terminal = stream.axis_start_ms+(stream.first_price_origin+len(stream.prices)-1)*1000
    while pending < count and maturities[pending] <= terminal:
        calibrator.update(int(stream.timestamps[pending]), stream.raw[pending],
                          float(np.log1p(stream.target(pending))))
        pending += 1
    if pending != count or not np.isfinite(parameters).all():
        raise ValueError("nonfinite forecast or unprocessed within-split endpoint")
    np.testing.assert_array_equal(parameters[maturities[0] > stream.timestamps],
                                  stream.raw[maturities[0] > stream.timestamps])
    np.testing.assert_array_equal(parameters[:, 4:], stream.raw[:, 4:])
    return {"parameters": parameters, "snapshots": snapshots,
        "canonicalIssues": count, "maturedUpdates": pending,
        "terminalTimestamp": int(terminal), "finalState": calibrator.snapshot(int(terminal))}


def class_metrics(log_probabilities: np.ndarray, y: np.ndarray) -> dict:
    classes = (y > 0).astype(np.int64)+(y >= 0).astype(np.int64)  # DOWN,ZERO,UP
    predicted = np.argmax(log_probabilities, axis=1)
    p = np.exp(log_probabilities)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (classes, predicted), 1)
    active = y != 0
    active_logp = log_probabilities[active][:, [0, 2]]
    active_logp -= np.logaddexp(active_logp[:, 0], active_logp[:, 1])[:, None]
    return {"classOrder": ["DOWN", "ZERO", "UP"], "examples": len(y),
        "activeExamples": int(active.sum()), "zeroExamples": int((~active).sum()),
        "threeClassLogLoss": float(-log_probabilities[np.arange(len(y)), classes].mean()),
        "threeClassAccuracy": float((predicted == classes).mean()), "threeClassConfusion": confusion.tolist(),
        "zeroBrier": float(np.square(p[:, 1]-(y == 0)).mean()),
        "predictedZeroProbability": float(p[:, 1].mean()), "actualZeroFrequency": float((y == 0).mean()),
        "conditionalActiveSignLogLoss": float(-active_logp[np.arange(int(active.sum())), (y[active] > 0).astype(np.int64)].mean()),
        "conditionalActiveSignAccuracy": float(((active_logp[:, 1] >= active_logp[:, 0]) == (y[active] > 0)).mean())}


def size_metrics(y: np.ndarray, up: np.ndarray, down: np.ndarray, constants: tuple[float, float] | None = None) -> dict:
    result = {}
    for j, (name, mask, predicted) in enumerate((("UP", y > 0, up), ("DOWN", y < 0, down))):
        actual = np.abs(y[mask])
        mse = float(np.square(predicted[mask]-actual).mean())
        entry = {"examples": len(actual), "mseBpsSquared": mse*1e8,
            "meanPredictedBps": float(predicted[mask].mean()*1e4), "meanRealizedBps": float(actual.mean()*1e4)}
        if constants is not None:
            baseline = float(np.square(constants[j]-actual).mean())
            entry.update({"trainingConstantMeanBps": constants[j]*1e4,
                          "mseSkillVsTrainingConstant": 1-mse/baseline})
        result[name] = entry
    return result


def score_log_law(packed: np.ndarray, simple_targets: np.ndarray, constants: dict | None = None) -> dict:
    """Score actual law expectations, never log of its simple-return mean."""
    if packed.shape != (len(simple_targets), 6) or packed.dtype != np.float64:
        raise ValueError("expected float64 six-parameter log law")
    if not np.isfinite(packed).all() or not np.all(simple_targets > -1):
        raise ValueError("invalid physical forecast or target")
    log_targets = np.log1p(simple_targets)
    np.testing.assert_array_equal(log_targets == 0, simple_targets == 0)
    tensor = torch.from_numpy(packed)
    p, up, down = (part.numpy() for part in log_hurdle_components(tensor))
    simple_p, simple_up, simple_down = (part.numpy() for part in log_hurdle_simple_components(tensor))
    np.testing.assert_array_equal(p, simple_p)
    pup = expit(packed[:, 1])
    log_conditional = pup*up-(1-pup)*down
    simple_conditional = pup*simple_up-(1-pup)*simple_down
    active_probability = expit(-packed[:, 0])
    log_deployed = active_probability*log_conditional
    simple_deployed = active_probability*simple_conditional
    log_nll = float(log_hurdle_nll_elements(tensor, torch.from_numpy(log_targets)).mean())
    return {"law": MODEL_CONTRACT, "logReturns": score_means(log_targets, log_conditional, log_deployed),
        "simpleReturns": score_means(simple_targets, simple_conditional, simple_deployed),
        "conditionalLogSizes": size_metrics(log_targets, up, down,
            None if constants is None else (constants["upMeanLog"], constants["downMeanLog"])),
        "conditionalSimpleSizes": size_metrics(simple_targets, simple_up, simple_down),
        "classification": class_metrics(log_hurdle_log_probabilities(tensor).numpy(), simple_targets),
        "logReturnNegativeLogLikelihood": log_nll,
        "simpleReturnNegativeLogLikelihood": log_nll+float(log_targets.mean()),
        "likelihoodMeasure": "zero atom plus Lebesgue density in the named return coordinate; nonzero Jacobian included",
        "finiteSimpleMean": bool(np.isfinite(simple_deployed).all()),
        "maximumUpLogMeanOverShape": float(np.max(packed[:, 2]/packed[:, 4])),
        "shapes": packed[0, 4:].tolist()}


def score_retained_simple_law(packed: np.ndarray, simple_targets: np.ndarray, constants: dict | None = None) -> dict:
    """Reintegrate the unchanged Gamma/Beta law in log coordinates."""
    log_targets = np.log1p(simple_targets)
    p0, pup = expit(packed[:, 0]), expit(packed[:, 1])
    up_low, down_low = retained_log_means(packed, 64)
    up, down = retained_log_means(packed, 128)
    conditional = pup*up+(1-pup)*down
    difference = float(np.max(np.abs(conditional-(pup*up_low+(1-pup)*down_low))))
    if difference > 1e-12:
        raise ValueError("retained expected-log quadrature did not converge")
    simple_conditional = pup*packed[:, 2]-(1-pup)*packed[:, 3]
    simple_nll = float(hurdle_nll_elements(torch.from_numpy(packed), torch.from_numpy(simple_targets)).mean())
    # Stable log class probabilities, independent of the new log-law helper.
    log_p0, log_active = -np.logaddexp(0, -packed[:, 0]), -np.logaddexp(0, packed[:, 0])
    log_pup, log_pdown = -np.logaddexp(0, -packed[:, 1]), -np.logaddexp(0, packed[:, 1])
    logp = np.stack((log_active+log_pdown, log_p0, log_active+log_pup), axis=1)
    return {"law": "retained-simple-Gamma-Beta-hurdle", "logReturns": score_means(log_targets, conditional, (1-p0)*conditional),
        "simpleReturns": score_means(simple_targets, simple_conditional, (1-p0)*simple_conditional),
        "conditionalLogSizes": size_metrics(log_targets, up, -down,
            None if constants is None else (constants["upMeanLog"], constants["downMeanLog"])),
        "conditionalSimpleSizes": size_metrics(simple_targets, packed[:, 2], packed[:, 3]),
        "classification": class_metrics(logp, simple_targets),
        "logReturnNegativeLogLikelihood": simple_nll-float(log_targets.mean()),
        "simpleReturnNegativeLogLikelihood": simple_nll,
        "maximum64vs128ConditionalLogMeanErrorBps": difference*1e4,
        "logExpectation": "UP generalized Gauss-Laguerre128; DOWN exact digamma identity with stable large-beta difference"}


@torch.inference_mode()
def infer_shared(model: torch.nn.Module, timeline: np.ndarray, origins: np.ndarray,
                 accuracy_count: int, device: torch.device, batch: int) -> np.ndarray:
    if tuple(model.horizons) != HORIZONS:
        raise ValueError("shared model horizon order changed")
    pieces = []
    for lo, hi in ((0, accuracy_count), (accuracy_count, len(origins))):
        for start in range(lo, hi, batch):
            rows = origins[start:min(start+batch, hi)]
            features = np.asarray(timeline[rows[:, None]+np.array([-1, 0])], dtype=np.float32)
            pieces.append(model(torch.from_numpy(features).to(device)).cpu().numpy())
    result = np.concatenate(pieces)
    if result.shape != (len(origins), len(HORIZONS), 6) or result.dtype != np.float64 or not np.isfinite(result).all():
        raise ValueError("invalid shared log-hurdle prediction array")
    return result


def save_stream(root: Path, split: str, stage: str, stream: CachedStream, simulated: dict | None = None) -> dict:
    file = root/"cache"/f"{split}-{stage}-h{stream.horizon_seconds}.npz"
    file.parent.mkdir(parents=True, exist_ok=True)
    parameters = stream.raw if simulated is None else simulated["parameters"]
    np.savez_compressed(file, parameters=parameters, origins=stream.origins, timestamps=stream.timestamps,
                        accuracyIndices=stream.accuracy_indices)
    record = {"parametersFile": str(file), "parametersSha256": sha(file)}
    if simulated is not None:
        trace = file.with_suffix(".trace.npz")
        np.savez_compressed(trace, origins=stream.origins, timestamps=stream.timestamps, **simulated["snapshots"])
        record.update({"traceFile": str(trace), "traceSha256": sha(trace),
            **{key: simulated[key] for key in ("canonicalIssues", "maturedUpdates", "terminalTimestamp", "finalState")}})
    return record


def endpoint_log_law(packed: np.ndarray) -> dict:
    """Three conditional price means, preserving E[simple return] in float64.

The DOWN representative is computed directly as a positive gross moment. A
roundoff-prone `1 - lossMagnitude` must not reconstruct that endpoint price.
Magnitude spread is discarded; this is not a general stochastic-law reduction.
"""
    tensor = torch.from_numpy(packed)
    p, up, down = (part.numpy() for part in log_hurdle_gross_components(tensor))
    probability_error = float(np.max(np.abs(p.sum(axis=1)-1)))
    if probability_error > 1e-12 or np.any(p < 0):
        raise ValueError("invalid log-law probabilities")
    probability = p/p.sum(axis=1, keepdims=True)
    gross = np.stack((down, np.ones(len(p)), up), axis=1)
    if not np.isfinite(gross).all() or np.any(gross <= 0):
        raise ValueError("export requires finite strictly positive endpoint prices")
    _, simple_up, simple_down = (part.numpy() for part in log_hurdle_simple_components(tensor))
    mean = probability[:, 2]*simple_up-probability[:, 0]*simple_down
    size = probability[:, 2]*simple_up+probability[:, 0]*simple_down
    discrepancy = float(np.max(np.abs(mean-log_hurdle_simple_mean(tensor).numpy())))
    if discrepancy > 1e-10:
        raise ValueError("scenario mean differs from the trained law")
    np.testing.assert_allclose(np.sum(probability*gross, axis=1), 1+mean, rtol=2e-14, atol=2e-15)
    return {"probabilities": probability, "gross": gross, "mean": mean, "size": size,
        "probabilitySumErrorBeforeNormalization": probability_error, "meanAbsoluteError": discrepancy}


def raw_up_bound_diagnostics(packed: np.ndarray) -> dict:
    """Invert the model's tanh map only where the inverse is identifiable."""
    ratio = packed[:, 2]/(packed[:, 4]*MAXIMUM_UP_MEAN_RATIO)
    floor, saturation = packed[:, 2] <= MINIMUM_MEAN, ratio >= 1
    interior = ~(floor | saturation)
    z = ratio[interior]
    reduction = 1-z/np.arctanh(z)
    return {"scope": "raw neural UP means only; inverse tanh is undefined at saturation and ambiguous at the mean floor",
        "examples": len(packed), "meanFloorCount": int(floor.sum()), "saturationCount": int(saturation.sum()),
        "upMeanOverCapQuantilesP50P95P99Max": np.quantile(ratio, [.5, .95, .99, 1]).tolist(),
        "relativeMeanReductionQuantilesP50P95P99Max": None if not len(z) else np.quantile(reduction, [.5, .95, .99, 1]).tolist()}


def export_stream(root: Path, stream: CachedStream, parameters: np.ndarray, variant: str,
                  selection_file: Path, selected: dict, stage: str, saved: dict,
                  replay: dict, config: dict | None, simulated: dict | None) -> dict:
    h = stream.horizon_seconds
    count = replay["decisionCountsByHorizon"][str(h)]
    origins = stream.origins[0]+np.arange(count, dtype=np.int64)*h
    indices = np.searchsorted(stream.origins, origins)
    np.testing.assert_array_equal(stream.origins[indices], origins)
    stamps = stream.timestamps[indices]
    if int(stamps[0]) != replay["firstTimestamp"] or int(stamps[-1])+h*1000 != replay["terminalTimestamp"] \
            or count*h != replay["comparisonObservedSeconds"]:
        raise ValueError("replay times differ from the registered common tape")
    law = endpoint_log_law(parameters[indices])
    file = root/f"{variant}-h{h}.jsonl"
    with file.open("w", encoding="utf-8", newline="\n") as sink:
        for i, origin in enumerate(origins):
            offset = int(origin)-stream.first_price_origin
            current = float(stream.prices[offset])
            row = {"horizonSeconds": h, "timestamp": int(stamps[i]), "observedPrice": current,
                "realizedNextTimestamp": int(stamps[i]+h*1000),
                "realizedNextPrice": float(stream.prices[offset+h]),
                "realizedPrices": stream.prices[offset+1:offset+h+1].tolist(),
                "tree": {"children": [{"probability": float(law["probabilities"][i, k]),
                    "price": current*float(law["gross"][i, k])} for k in range(3)]}}
            sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
    replay_states = []
    if simulated is not None:
        trace = simulated["snapshots"]
        for i, index in enumerate(indices):
            latest = int(trace["latestMaturityUsed"][index])
            if latest > stamps[i]:
                raise ValueError("replay state uses a future endpoint")
            replay_states.append({"timestamp": int(stamps[i]),
                "latestMaturityUsed": None if latest < 0 else latest,
                "maturedObservations": int(trace["maturedObservations"][index]),
                "coefficients": trace["coefficients"][index].tolist()})
    mean, size = law["mean"], law["size"]
    record = selected[stage]
    manifest = {"contract": CONTRACT, "variant": variant, "combined": record,
        "checkpoint": record["checkpoint"], "checkpointSha256": record["sha256"],
        "selectionFile": str(selection_file), "selectionSha256": sha(selection_file),
        "headSelectionsFrozenBeforeTest": True, "calibrationSettingsFrozenBeforeTest": True,
        "testUsedForSelection": False, "refit": False, "reselection": False,
        "onlineLabelUpdates": simulated is not None, "calibrationColdStart": simulated is not None,
        "frozenCalibration": config, "savedStream": saved, "replayStateSamples": replay_states,
        "replayCanonicalIndices": indices.tolist(), "calibrationShapeUpdates": False,
        "forecastFile": str(file), "forecastSha256": sha(file), "split": "test", "rows": count,
        "horizonSeconds": h, "decisionStrideSeconds": h, "firstTimestamp": int(stamps[0]),
        "lastRealizedTimestamp": int(stamps[-1]+h*1000), "observedSeconds": count*h,
        "probabilityOrder": ["down", "zero", "up"],
        "predictedZeroProbabilityMean": float(law["probabilities"][:, 1].mean()),
        "predictedSignedMeanBps": float(mean.mean()*1e4), "predictedAverageAbsoluteSizeBps": float(size.mean()*1e4),
        "predictedMinMeanBps": float(mean.min()*1e4), "predictedMaxMeanBps": float(mean.max()*1e4),
        "fullFeeThresholdLongCount": int((mean > 1/.9995**2-1).sum()),
        "maximumProbabilitySumErrorBeforeNormalization": law["probabilitySumErrorBeforeNormalization"],
        "maximumExportVsTorchMeanErrorBps": law["meanAbsoluteError"]*1e4,
        "expectation": "analytic conditional E[exp(X)] under signed Gamma log law, retaining native zero atom",
        "sourceMarket": "BTCUSDT spot one-second closes with synthetic Oracle account parameters",
        "limitation": "Three conditional-price means discard magnitude spread. Restricted long/cash endpoint controller only; no general CRRA, leverage, limit-order or path-law optimality claim."}
    write_json(file.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("data/training/runs/stochastic-crra-log-hurdle-v1/selection.json"))
    parser.add_argument("--online-root", type=Path, default=Path("data/training/runs/stochastic-crra-online-calibration-v1"))
    parser.add_argument("--output-root", type=Path, default=Path("data/backtests/oracle-log-hurdle"))
    parser.add_argument("--joint-audit", type=Path, default=Path("data/backtests/oracle-online-calibration/joint-log-return-audit.json"))
    parser.add_argument("--shared-simple-result", type=Path, default=Path("data/training/runs/stochastic-crra-shared-hurdle-v1/result.json"))
    parser.add_argument("--batch-size", type=int, default=2560)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-exports", action="store_true")
    args = parser.parse_args()
    from train_log_hurdle_sign_size_forecast import CONTRACT as TRAINING_CONTRACT, load_combined
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    tracked: dict[str, str] = {}

    def checked(file: Path, expected: str | None = None) -> Path:
        digest = sha(file)
        if expected is not None and digest != expected:
            raise ValueError(f"immutable artifact hash mismatch: {file}")
        tracked[str(file)] = digest
        return file

    def read(file: Path, expected: str | None = None) -> dict:
        return json.loads(checked(file, expected).read_text(encoding="utf-8"))

    def arrays(file: Path, expected: str) -> dict[str, np.ndarray]:
        with np.load(checked(file, expected), allow_pickle=False) as data:
            return {key: immutable(data[key]) for key in data.files}

    implementation_files = (Path(__file__), Path("ml/log_hurdle_sign_size_forecast.py"),
        Path("ml/online_log_hurdle_calibration.py"), Path("ml/train_log_hurdle_sign_size_forecast.py"),
        Path("ml/audit_hurdle_log_returns.py"), Path("ml/audit_joint_log_returns.py"))
    for file in implementation_files:
        checked(file)
    selected = read(args.selection)
    if selected["contract"] != TRAINING_CONTRACT or not selected["allHeadSelectionsFrozen"] \
            or selected["testLoaded"] or selected["testUsedForSelection"]:
        raise ValueError("all new neural selections must be frozen without test selection")
    if tuple(row["horizonSeconds"] for row in selected["horizons"]) != HORIZONS \
            or tuple(row["horizonSeconds"] for row in selected["constants"]) != HORIZONS:
        raise ValueError("new checkpoint/constant horizon order changed")
    training_result = read(args.selection.parent/"training-result.json")
    if training_result["selectionSha256"] != tracked[str(args.selection)] \
            or training_result["testLoaded"] or training_result["testUsedForSelection"]:
        raise ValueError("completed training does not match the frozen checkpoint selection")
    training_plan = read(args.selection.parent/"plan.json", selected["planSha256"])
    if training_plan != selected["plan"]:
        raise ValueError("embedded training plan changed")
    dataset = Path(training_plan["dataset"])
    checked(dataset/"manifest.json", training_plan["datasetManifestSha256"])
    old_selection_file = args.online_root/"selection.json"
    old = read(old_selection_file, selected["onlineSelectionSha256"])
    if old["testLoaded"] or not old["allCalibrationSelectionsFrozen"]:
        raise ValueError("old calibration settings were not frozen before test")
    if read(args.online_root/"plan.json", old["planSha256"]) != old["plan"]:
        raise ValueError("old calibration selection differs from its immutable plan")
    if Path(selected["onlineSelectionFile"]).resolve() != old_selection_file.resolve():
        raise ValueError("unexpected frozen calibration selection file")
    checked(Path(selected["sourceSelectionFile"]), selected["sourceSelectionSha256"])
    if selected["sourceSelectionSha256"] != old["sourceSelectionSha256"]:
        raise ValueError("baseline neural provenance changed")
    if old["plan"]["datasetManifestSha256"] != training_plan["datasetManifestSha256"] or old["testUsedForSelection"]:
        raise ValueError("comparison data or old selection provenance changed")
    old_result = read(args.online_root/"result.json")
    if old_result["selectionSha256"] != tracked[str(old_selection_file)]:
        raise ValueError("old results do not match frozen calibration")
    shared_result = read(args.shared_simple_result)
    shared_selection = read(Path(shared_result["selectionFile"]), shared_result["selectionSha256"])
    if shared_result["calibrationSelectionSha256"] != tracked[str(old_selection_file)] \
            or shared_selection["plan"]["datasetManifestSha256"] != training_plan["datasetManifestSha256"] \
            or shared_result["testUsedForSelection"] or shared_result["calibrationReselection"] \
            or shared_selection["testLoaded"] or not shared_selection["allHeadSelectionsFrozen"]:
        raise ValueError("retained shared-simple baseline provenance differs")
    if read(Path(shared_result["selectionFile"]).parent/"plan.json", shared_selection["planSha256"]) != shared_selection["plan"]:
        raise ValueError("retained shared-simple training plan changed")
    checked(Path(shared_selection["finetuned"]["checkpoint"]), shared_selection["finetuned"]["sha256"])
    for stage in STAGES:
        record = selected[stage]
        checkpoint = checked(Path(record["checkpoint"]), record["sha256"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["planSha256"] != selected["planSha256"] or payload["contract"] != TRAINING_CONTRACT \
                or payload["task"] != "combined" or payload["targetCoordinate"] != "log-return" \
                or payload["horizonsSeconds"] != list(HORIZONS) or payload["epoch"] != record["epoch"]:
            raise ValueError("new checkpoint and training plan differ")
        del payload
    configs = {str(row["horizonSeconds"]): {"sourcePreferredMode": row["validationPreferredMode"],
        **row["modes"][row["validationPreferredMode"]]} for row in old["horizons"]}
    if tuple(row["horizonSeconds"] for row in old["horizons"]) != HORIZONS:
        raise ValueError("comparison horizon order changed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root/"evaluation.json").exists():
        raise ValueError("refusing to overwrite completed frozen evaluation")
    old_replay = read(Path(old["plan"]["forecastOutput"])/"replay-plan.json")
    replay = {**old_replay, "contract": "oracle-log-hurdle-comparison-v1",
        "runsRoot": str(args.selection.parent), "reportScript": "scripts/report-oracle-log-hurdle.mjs",
        "variants": [{"name": stage+"-"+mode, "feeScales": [1, 0]} for stage in STAGES for mode in ("raw", "calibrated")],
        "selection": "Fresh shared log-hurdle pretraining/fine-tuning selected by validation; old online preferred modes and hyperparameters retained with no new search",
        "onlineAdaptation": "Identity per split/checkpoint/horizon; canonical4s issued forecasts update only at maturity using observed LOG labels; fixed old settings",
        "scope": "Three analytic conditional price means from signed Gamma LOG law; same restricted Oracle long/cash endpoint menu, gamma0, unit funding, no orders, all1s marks, fees/caps/grid"}
    if not args.skip_exports:
        replay_file = args.output_root/"replay-plan.json"
        if replay_file.exists():
            if read(replay_file) != replay:
                raise ValueError("prepared replay plan changed")
        else:
            write_json(replay_file, replay)
            checked(replay_file)
    evaluation_plan = {"contract": CONTRACT, "createdAt": now(), "selectionFile": str(args.selection),
        "selectionSha256": tracked[str(args.selection)], "onlineSelectionFile": str(old_selection_file),
        "onlineSelectionSha256": tracked[str(old_selection_file)], "calibrationConfigurations": configs,
        "implementationSha256": {str(file): tracked[str(file)] for file in implementation_files},
        "newHyperparameterSelection": False, "testUsedForSelection": False,
        "stages": ["pretrainedRaw", "pretrainedCalibrated", "finetunedRaw", "finetunedCalibrated"],
        "calibration": "old preferred hyperparameters and modes unchanged; log likelihood updates; independent cold state per checkpoint/split/horizon",
        "population": "old canonical four-second issue streams, old maximum7200-complete common accuracy origins",
        "logTarget": "float64 log1p of exact endpoint simple return, matching retained log-return audits",
        "prediction": "integrate LOG and SIMPLE returns under each law; never log1p of a predicted simple mean",
        "overlap": "min(1,4/H) is an update-weight heuristic, not independent observations",
        "device": str(device), "batchSize": args.batch_size, "matmulPrecision": "high"}
    plan_file = args.output_root/"evaluation-plan.json"
    if plan_file.exists():
        prior = read(plan_file)
        evaluation_plan["createdAt"] = prior["createdAt"]
        if prior != evaluation_plan:
            raise ValueError("prepared evaluation plan changed")
    else:
        write_json(plan_file, evaluation_plan)
    checked(plan_file)
    started = time.monotonic()
    output = {**evaluation_plan, "evaluationPlanFile": str(plan_file), "evaluationPlanSha256": sha(plan_file),
              "results": [], "sourceTrainingPlanSha256": selected["planSha256"]}
    exports = []
    for split in ("validation", "test"):
        old_split = read(args.online_root/f"{split}.json")
        if old_split["horizons"] != old_result[split]:
            raise ValueError("old split results differ from the completed evaluation")
        timeline = load_timeline_split(dataset, split)
        common = select_origins(timeline, 4, None)
        expected_count = 34560 if split == "validation" else 36134
        if len(common) != expected_count:
            raise ValueError("common evaluation population changed")
        contexts = []
        for j, h in enumerate(HORIZONS):
            manifest = old_split["cacheManifests"][j]
            old_checkpoint = old["horizons"][j]["checkpoint"]
            if manifest["horizonSeconds"] != h or manifest["sourceSelectionSha256"] != old["sourceSelectionSha256"] \
                    or manifest["planSha256"] != old["planSha256"] \
                    or manifest["checkpointSha256"] != old_checkpoint["sha256"] \
                    or Path(manifest["checkpoint"]).resolve() != Path(old_checkpoint["checkpoint"]).resolve():
                raise ValueError("old cache horizon order changed")
            checked(Path(old_checkpoint["checkpoint"]), old_checkpoint["sha256"])
            cache = arrays(Path(manifest["cacheFile"]), manifest["cacheSha256"])
            market = arrays(Path(manifest["marketFile"]), manifest["marketSha256"])
            np.testing.assert_array_equal(cache["origins"][cache["accuracyIndices"]], common)
            np.testing.assert_array_equal(market["commonOrigins"], common)
            np.testing.assert_array_equal(cache["timestamps"], timeline.axis_start_ms+cache["origins"]*1000)
            offsets = common-int(market["firstPriceOrigin"])
            simple = market["prices"][offsets+h]/market["prices"][offsets]-1
            log_target = np.log1p(simple)
            if split == "test" and h == 1 and int((simple != 0).sum()) != 15505:
                raise ValueError("matched H1 active evaluation population changed")
            sample_hashes = {"origins": array_sha(common), "timestamps": array_sha(cache["timestamps"][cache["accuracyIndices"]]),
                "simpleTargets": array_sha(simple), "logTargets": array_sha(log_target)}
            row = {"split": split, "horizonSeconds": h, "sampleHashes": sample_hashes,
                "examples": len(simple), "activeExamples": int((simple != 0).sum()),
                "constants": selected["constants"][j], "models": {}, "oldCanonicalCacheManifest": manifest}
            mode = configs[str(h)]["sourcePreferredMode"]
            old_calibration = old_split["horizons"][j]["models"][mode]
            saved = old_calibration["savedStream"]
            old_calibrated = arrays(Path(saved["parametersFile"]), saved["parametersSha256"])
            np.testing.assert_array_equal(old_calibrated["origins"], cache["origins"])
            np.testing.assert_array_equal(old_calibrated["accuracyIndices"], cache["accuracyIndices"])
            for name, packed, reference in (("retainedSimpleRaw", cache["raw"], old_split["horizons"][j]["models"]["none"]),
                    ("retainedSimpleCalibrated", old_calibrated["parameters"], old_calibration)):
                baseline = score_retained_simple_law(packed[cache["accuracyIndices"]], simple, selected["constants"][j])
                np.testing.assert_allclose(baseline["simpleReturns"]["allTickDeployedMseSkillVsZero"],
                    reference["allTickSignedMeanMseSkillVsZero"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(baseline["simpleReturnNegativeLogLikelihood"],
                    reference["fullHurdleReturnNegativeLogLikelihood"], rtol=0, atol=1e-11)
                row["models"][name] = baseline
            for source_name, name in (("shared-raw", "retainedSharedSimpleRaw"), ("shared-online", "retainedSharedSimpleCalibrated")):
                reference = shared_result[split][j]["models"][source_name]
                saved_shared = reference["savedStream"]
                shared_arrays = arrays(Path(saved_shared["parametersFile"]), saved_shared["parametersSha256"])
                np.testing.assert_array_equal(shared_arrays["origins"], cache["origins"])
                np.testing.assert_array_equal(shared_arrays["accuracyIndices"], cache["accuracyIndices"])
                baseline = score_retained_simple_law(shared_arrays["parameters"][cache["accuracyIndices"]], simple, selected["constants"][j])
                np.testing.assert_allclose(baseline["simpleReturns"]["allTickDeployedMseSkillVsZero"],
                    reference["allTickSignedMeanMseSkillVsZero"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(baseline["simpleReturnNegativeLogLikelihood"],
                    reference["fullHurdleReturnNegativeLogLikelihood"], rtol=0, atol=1e-11)
                row["models"][name] = baseline
            if split == "test" and h == 1:
                joint = read(args.joint_audit)
                if joint["sampleHashes"] != sample_hashes or joint["datasetManifestSha256"] != training_plan["datasetManifestSha256"]:
                    raise ValueError("retained joint audit uses different observations")
                row["models"]["retainedJointDensity"] = {"logReturns": joint["logReturns"],
                    "simpleReturns": joint["simpleReturnRegression"], "auditFile": str(args.joint_audit),
                    "auditSha256": tracked[str(args.joint_audit)], "checkpointSha256": joint["checkpointSha256"]}
            output["results"].append(row)
            contexts.append((cache, market, simple, row))
        union = contexts[0][0]["origins"]
        for stage in STAGES:
            model = load_combined(selected[stage], device)
            model.eval()
            packed = infer_shared(model, timeline.timeline, union, len(common), device, args.batch_size)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for j, h in enumerate(HORIZONS):
                cache, market, simple, row = contexts[j]
                indices = np.searchsorted(union, cache["origins"])
                np.testing.assert_array_equal(union[indices], cache["origins"])
                raw = immutable(packed[indices, j])
                stream = CachedStream(cache["origins"], cache["timestamps"], raw, market["prices"],
                    int(market["firstPriceOrigin"]), cache["accuracyIndices"], int(market["axisStartMs"]), h)
                raw_metrics = score_log_law(raw[stream.accuracy_indices], simple, selected["constants"][j])
                raw_metrics["upSmoothBound"] = raw_up_bound_diagnostics(raw[stream.accuracy_indices])
                if split == "validation":
                    expected_nll = selected[stage]["validationScoresByHorizon"][j]
                    np.testing.assert_allclose(raw_metrics["logReturnNegativeLogLikelihood"], expected_nll, rtol=0, atol=1e-8)
                    raw_metrics["selectedValidationNllAbsoluteDifference"] = abs(raw_metrics["logReturnNegativeLogLikelihood"]-expected_nll)
                raw_metrics["savedStream"] = save_stream(args.output_root, split, stage+"-raw", stream)
                config_record = configs[str(h)]
                simulated = simulate_log_stream(stream, CalibrationConfig(**config_record["config"]), trace=True)
                calibrated_metrics = score_log_law(simulated["parameters"][stream.accuracy_indices], simple, selected["constants"][j])
                calibrated_metrics["savedStream"] = save_stream(args.output_root, split, stage+"-calibrated", stream, simulated)
                calibrated_metrics["frozenCalibration"] = config_record
                row["models"][stage+"Raw"] = raw_metrics
                row["models"][stage+"Calibrated"] = calibrated_metrics
                if split == "test" and h in replay["horizonsSeconds"] and not args.skip_exports:
                    for mode, parameters, scored, simulation in (("raw", raw, raw_metrics, None),
                            ("calibrated", simulated["parameters"], calibrated_metrics, simulated)):
                        manifest = export_stream(args.output_root, stream, parameters, stage+"-"+mode,
                            args.selection, selected, stage, scored["savedStream"], replay,
                            None if simulation is None else config_record, simulation)
                        exports.append(manifest)
                        checked(Path(manifest["forecastFile"]), manifest["forecastSha256"])
                        checked(Path(manifest["forecastFile"]).with_suffix(".manifest.json"))
                for result in (raw_metrics, calibrated_metrics):
                    saved = result["savedStream"]
                    checked(Path(saved["parametersFile"]), saved["parametersSha256"])
                    if "traceFile" in saved:
                        checked(Path(saved["traceFile"]), saved["traceSha256"])
                print(json.dumps({"event": "logHurdleScored", "split": split, "stage": stage, "horizonSeconds": h,
                    "rawLogSkill": raw_metrics["logReturns"]["allTickDeployedMseSkillVsZero"],
                    "calibratedLogSkill": calibrated_metrics["logReturns"]["allTickDeployedMseSkillVsZero"]}), flush=True)
    if not args.skip_exports:
        exports_file = args.output_root/"exports.json"
        write_json(exports_file, {"contract": CONTRACT, "manifests": exports})
        checked(exports_file)
        output["exportsFile"] = str(exports_file)
        output["exportsSha256"] = sha(exports_file)
    for file, digest in tracked.items():
        if sha(Path(file)) != digest:
            raise ValueError(f"artifact changed during evaluation: {file}")
    output["artifactHashes"] = tracked
    output["seconds"] = time.monotonic()-started
    output_file = args.output_root/"evaluation.json"
    write_json(output_file, output)
    print(json.dumps({"event": "logHurdleEvaluationComplete", "output": str(output_file), "sha256": sha(output_file),
                      "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
