"""Independent NumPy audit of frozen log-law forecasts and calibration traces.

No neural inference, fitting, model helpers, or calibrator implementation is
imported. The audit checks every saved canonical query and recomputes analytic
log/simple expectations and MSE on the exact archived evaluation population.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

CONTRACT = "independent-log-hurdle-calibration-audit-v1"
HORIZONS = (1,900,1800,7200)
STAGES = ("pretrained","finetuned")
MINIMUM_MEAN,MAXIMUM_UP_MEAN_RATIO = 1e-12,.49
BOUNDS = np.array([2.,2.,math.log(2),math.log(2)])


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(file: Path) -> dict:
    return json.loads(file.read_text(encoding="utf-8"))


def checked(file: str | Path, expected: str) -> Path:
    path = Path(file)
    if file_sha(path) != expected:
        raise ValueError(f"immutable artifact hash mismatch: {path}")
    return path


def load_arrays(file: str | Path, expected: str) -> dict[str,np.ndarray]:
    with np.load(checked(file,expected),allow_pickle=False) as saved:
        return {name:saved[name] for name in saved.files}


def logistic(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.,-value))


def expected_parameters(raw: np.ndarray,theta: np.ndarray) -> np.ndarray:
    """Apply the declared coordinates independently with vectorized NumPy."""
    result = raw.copy()
    result[:,:2] = raw[:,:2]+theta[:,:2]
    active = theta[:,2] != 0
    up,alpha = raw[active,2],raw[active,4]
    logit = np.log(up)-np.log(alpha-up)+theta[active,2]
    result[active,2] = np.minimum(alpha*MAXIMUM_UP_MEAN_RATIO,
        np.maximum(MINIMUM_MEAN,alpha*logistic(logit)))
    active = theta[:,3] != 0
    result[active,3] = np.maximum(MINIMUM_MEAN,raw[active,3]*np.exp(theta[active,3]))
    return result


def audit_trace(raw: np.ndarray,calibrated: np.ndarray,timestamps: np.ndarray,
                trace: dict,config: dict,saved: dict) -> dict:
    h,mode = config["horizon_seconds"],config["mode"]
    if config["cadence_seconds"] != 4:
        raise ValueError("unexpected canonical calibration cadence")
    n = len(timestamps)
    np.testing.assert_array_equal(np.diff(timestamps),np.full(n-1,4000))
    if raw.shape != (n,6) or calibrated.shape != raw.shape or not np.isfinite(raw).all() \
            or not np.isfinite(calibrated).all():
        raise ValueError("invalid raw/calibrated canonical shape or finite values")
    for parameters in (raw,calibrated):
        if np.any(parameters[:,2:4] < MINIMUM_MEAN) or np.any(parameters[:,4:] <= 0) \
                or np.any(parameters[:,4:] > 100) or np.any(parameters[:,2] > MAXIMUM_UP_MEAN_RATIO*parameters[:,4]):
            raise ValueError("saved log-law parameters violate declared magnitude bounds")
    maturity = timestamps+h*1000
    counts = np.searchsorted(maturity,timestamps,side="right")
    last = np.where(counts==0,-1,maturity[np.maximum(counts-1,0)])
    np.testing.assert_array_equal(trace["timestamps"],timestamps)
    np.testing.assert_array_equal(trace["maturedObservations"],counts)
    np.testing.assert_array_equal(trace["latestMaturityUsed"],last)
    theta = trace["coefficients"]
    if theta.shape != (n,4) or not np.isfinite(theta).all() or np.any(np.abs(theta)>BOUNDS+1e-15):
        raise ValueError("invalid saved calibration coefficients")
    enabled = {"none":0,"zero":1,"probabilities":2,"full":4}[mode]
    np.testing.assert_array_equal(theta[:,enabled:],np.zeros_like(theta[:,enabled:]))
    quiet = counts==0
    np.testing.assert_array_equal(theta[quiet],np.zeros((int(quiet.sum()),4)))
    np.testing.assert_array_equal(calibrated[quiet],raw[quiet])
    np.testing.assert_array_equal(calibrated[:,4:],raw[:,4:])
    expected = expected_parameters(raw,theta)
    maximum_error = float(np.max(np.abs(expected-calibrated)))
    np.testing.assert_allclose(calibrated,expected,rtol=3e-14,atol=2e-15)
    information = trace["discountedInformation"]
    if information.shape != (n,4) or not np.isfinite(information).all() or np.any(information < 0):
        raise ValueError("invalid discounted Fisher information")
    np.testing.assert_array_equal(information[:,enabled:],np.zeros_like(information[:,enabled:]))
    for name in ("componentUpdates","boundaryUpdates"):
        values = trace[name]
        if values.shape != (n,4) or np.any(values<0) or np.any(np.diff(values,axis=0)<0) \
                or np.any(values>counts[:,None]):
            raise ValueError(f"invalid cumulative calibration {name}")
        np.testing.assert_array_equal(values[:,enabled:],np.zeros_like(values[:,enabled:]))
    if saved["canonicalIssues"] != n or saved["maturedUpdates"] != n \
            or saved["finalState"]["maturedObservations"] != n \
            or saved["finalState"]["latestMaturityUsed"] != int(maturity[-1]) \
            or saved["terminalTimestamp"] < maturity[-1]:
        raise ValueError("canonical terminal maturity flush is incomplete")
    return {"canonicalQueries":n,"coldIdentityQueries":int(quiet.sum()),
        "allMaturityCountsExact":True,"allLatestMaturityTimesExact":True,
        "shapesUnchanged":True,"maximumIndependentParameterError":maximum_error,
        "coefficientMaximumAbsoluteValues":np.max(np.abs(theta),axis=0).tolist(),
        "lastQueryTimestamp":int(timestamps[-1]),"lastIncorporatedMaturityAtQuery":int(last[-1]),
        "finalMaturedOriginTimestamp":int(timestamps[-1]),"finalMaturityTimestamp":int(maturity[-1])}


def independent_scores(parameters: np.ndarray,simple: np.ndarray) -> dict:
    """E[X] and E[exp(X)-1] are integrated separately under signed Gamma."""
    active_probability = logistic(-parameters[:,0])
    up_probability = logistic(parameters[:,1])
    up,down,alpha_up,alpha_down = (parameters[:,j] for j in range(2,6))
    conditional_log = up_probability*up-(1-up_probability)*down
    conditional_gross_up_log = -alpha_up*np.log1p(-up/alpha_up)
    conditional_gross_down_log = -alpha_down*np.log1p(down/alpha_down)
    simple_up = np.expm1(conditional_gross_up_log)
    simple_down = -np.expm1(conditional_gross_down_log)
    conditional_simple = up_probability*simple_up-(1-up_probability)*simple_down
    result = {}
    for coordinate,target,conditional in (("logReturns",np.log1p(simple),conditional_log),
                                           ("simpleReturns",simple,conditional_simple)):
        deployed = active_probability*conditional
        active = target!=0
        if not np.isfinite(conditional).all() or not np.isfinite(deployed).all() or not active.any():
            raise ValueError("nonfinite mean or no active outcomes in an audit population")
        active_base = float(np.square(target[active]).mean())
        baseline = float(np.square(target).mean())
        conditional_mse = float(np.square(conditional[active]-target[active]).mean())
        active_mse = float(np.square(deployed[active]-target[active]).mean())
        all_mse = float(np.square(deployed-target).mean())
        result[coordinate] = {"examples":len(target),"activeExamples":int(active.sum()),
            "activeConditionalMseBpsSquared":conditional_mse*1e8,
            "activeConditionalMseSkillVsZero":1-conditional_mse/active_base,
            "activeDeployedMseBpsSquared":active_mse*1e8,
            "activeDeployedMseSkillVsZero":1-active_mse/active_base,
            "allTickDeployedMseBpsSquared":all_mse*1e8,
            "allTickDeployedMseSkillVsZero":1-all_mse/baseline,
            "activeZeroBaselineMseBpsSquared":active_base*1e8,
            "allTickZeroBaselineMseBpsSquared":baseline*1e8}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation",type=Path,default=Path("data/backtests/oracle-log-hurdle/evaluation.json"))
    parser.add_argument("--output",type=Path,default=Path("data/backtests/oracle-log-hurdle/calibration-diagnostics.json"))
    args = parser.parse_args()
    started = time.monotonic()
    evaluation = read_json(args.evaluation)
    evaluation_sha = file_sha(args.evaluation)
    if evaluation["contract"] != "matched-log-hurdle-evaluation-v1" or evaluation["testUsedForSelection"] \
            or evaluation["newHyperparameterSelection"]:
        raise ValueError("unexpected evaluation or test-based parameter selection")
    for file,digest in evaluation["artifactHashes"].items():
        checked(file,digest)
    selection = read_json(checked(evaluation["selectionFile"],evaluation["selectionSha256"]))
    prior = read_json(checked(evaluation["onlineSelectionFile"],evaluation["onlineSelectionSha256"]))
    if not selection["allHeadSelectionsFrozen"] or selection["testLoaded"] or selection["testUsedForSelection"] \
            or not prior["allCalibrationSelectionsFrozen"] or prior["testLoaded"] or prior["testUsedForSelection"] \
            or selection["onlineSelectionSha256"] != evaluation["onlineSelectionSha256"]:
        raise ValueError("neural/calibration settings were not independently frozen")
    expected_configs = {str(row["horizonSeconds"]):{"sourcePreferredMode":row["validationPreferredMode"],
        **row["modes"][row["validationPreferredMode"]]} for row in prior["horizons"]}
    if evaluation["calibrationConfigurations"] != expected_configs:
        raise ValueError("log evaluation changed the prior calibration settings")
    expected_keys = {(split,h) for split in ("validation","test") for h in HORIZONS}
    if {(row["split"],row["horizonSeconds"]) for row in evaluation["results"]} != expected_keys \
            or len(evaluation["results"]) != len(expected_keys):
        raise ValueError("incomplete or duplicate evaluation split/horizon coverage")
    reports = []
    fine_epochs = {h: (next(r for r in selection["horizons"] if r["horizonSeconds"] == h)["finetuned"]["epoch"]
        if selection["contract"] == "per-horizon-log-hurdle-training-v1" else selection["finetuned"]["epoch"]) for h in HORIZONS}
    query_total,max_parameter_error,max_metric_error = 0,0.,0.
    for row in evaluation["results"]:
        split,h = row["split"],row["horizonSeconds"]
        manifest = row["oldCanonicalCacheManifest"]
        source = load_arrays(manifest["cacheFile"],manifest["cacheSha256"])
        market = load_arrays(manifest["marketFile"],manifest["marketSha256"])
        origins,timestamps,accuracy = (source[key] for key in ("origins","timestamps","accuracyIndices"))
        np.testing.assert_array_equal(timestamps,int(market["axisStartMs"])+origins*1000)
        np.testing.assert_array_equal(origins[accuracy],market["commonOrigins"])
        offsets = origins[accuracy]-int(market["firstPriceOrigin"])
        simple = market["prices"][offsets+h]/market["prices"][offsets]-1
        hashes = {"origins":array_sha(origins[accuracy]),"timestamps":array_sha(timestamps[accuracy]),
            "simpleTargets":array_sha(simple),"logTargets":array_sha(np.log1p(simple))}
        if hashes != row["sampleHashes"] or len(simple) != row["examples"] \
                or int((simple!=0).sum()) != row["activeExamples"]:
            raise ValueError("reported target/origin population differs from archived source")
        report = {"split":split,"horizonSeconds":h,"sampleHashes":hashes,"sourcePopulationExact":True,"stages":{}}
        pretrained_parameters = None
        for stage in STAGES:
            raw_scored,calibrated_scored = row["models"][stage+"Raw"],row["models"][stage+"Calibrated"]
            raw_saved,calibrated_saved = raw_scored["savedStream"],calibrated_scored["savedStream"]
            raw_arrays = load_arrays(raw_saved["parametersFile"],raw_saved["parametersSha256"])
            calibrated_arrays = load_arrays(calibrated_saved["parametersFile"],calibrated_saved["parametersSha256"])
            if stage == "pretrained":
                pretrained_parameters = raw_arrays["parameters"],calibrated_arrays["parameters"]
            elif fine_epochs[h] == 0:
                np.testing.assert_array_equal(raw_arrays["parameters"],pretrained_parameters[0])
                np.testing.assert_array_equal(calibrated_arrays["parameters"],pretrained_parameters[1])
                report["selectedEpochZeroRawAndCalibratedIdentity"] = True
            trace = load_arrays(calibrated_saved["traceFile"],calibrated_saved["traceSha256"])
            for values in (raw_arrays,calibrated_arrays):
                for key,expected in (("origins",origins),("timestamps",timestamps),("accuracyIndices",accuracy)):
                    np.testing.assert_array_equal(values[key],expected)
            np.testing.assert_array_equal(trace["origins"],origins)
            config_record = expected_configs[str(h)]
            if calibrated_scored["frozenCalibration"] != config_record:
                raise ValueError("saved stage calibration differs from the frozen prior settings")
            trace_report = audit_trace(raw_arrays["parameters"],calibrated_arrays["parameters"],timestamps,
                trace,config_record["config"],calibrated_saved)
            query_total += trace_report["canonicalQueries"]
            max_parameter_error = max(max_parameter_error,trace_report["maximumIndependentParameterError"])
            stage_report = {"frozenConfig":config_record,"trace":trace_report,"metrics":{}}
            for mode,values,scored in (("raw",raw_arrays,raw_scored),("calibrated",calibrated_arrays,calibrated_scored)):
                metrics = independent_scores(values["parameters"][accuracy],simple)
                for coordinate,entries in metrics.items():
                    for name,value in entries.items():
                        reference = scored[coordinate][name]
                        np.testing.assert_allclose(value,reference,rtol=3e-12,atol=2e-11,
                            err_msg=f"{split}/{h}/{stage}/{mode}/{coordinate}/{name}")
                        max_metric_error = max(max_metric_error,abs(value-reference))
                stage_report["metrics"][mode] = metrics
            report["stages"][stage] = stage_report
        reports.append(report)
    checked(args.evaluation,evaluation_sha)
    result = {"contract":CONTRACT,"createdAt":datetime.now(timezone.utc).isoformat(),
        "evaluationFile":str(args.evaluation),"evaluationSha256":evaluation_sha,
        "auditScriptFile":str(Path(__file__)),"auditScriptSha256":file_sha(Path(__file__)),
        "selectionSha256":evaluation["selectionSha256"],"calibrationSelectionSha256":evaluation["onlineSelectionSha256"],
        "allChecksPassed":True,"neuralInference":False,"newFitting":False,
        "selectedEpochZeroIdentityChecked":any(epoch == 0 for epoch in fine_epochs.values()),
        "selectedEpochsByHorizon":fine_epochs,
        "canonicalCalibrationQueriesChecked":query_total,"calibratedStreamsChecked":len(reports)*len(STAGES),
        "rawStreamsChecked":len(reports)*len(STAGES),"splitHorizonRowsChecked":len(reports),
        "maximumIndependentParameterError":max_parameter_error,"maximumIndependentMetricError":max_metric_error,
        "checks":"All frozen hashes and per-horizon settings; full query maturity count/latest-time arrays; cold identity; untouched shapes; bounded coefficients and cumulative counts; independent NumPy parameter transform; exact archived origin/target hashes; independent analytic Elog and Esimple MSE on both splits",
        "scope":"Checks saved event metadata, transformed parameters, and metrics. Existing production stream tests establish future-suffix and query-schedule invariance. Does not infer neural predictions or reconstruct every optimizer statistic from labels.",
        "results":reports,"seconds":time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("allChecksPassed","canonicalCalibrationQueriesChecked",
        "calibratedStreamsChecked","maximumIndependentParameterError","maximumIndependentMetricError","seconds")}))


if __name__ == "__main__":
    main()
