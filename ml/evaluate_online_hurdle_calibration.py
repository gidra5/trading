"""Causal online calibration of frozen hurdle forecasts; no neural retraining.

One canonical four-second issuance/maturity stream serves every metric and
replay query. Validation fixes the settings within each mode before test loads.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from export_hurdle_sign_size_forecast import frozen_selection, load_combined
from export_stochastic_price_forecast import load_close_prices
from export_ternary_sign_size_forecast import endpoint_law
from hurdle_sign_size_forecast import hurdle_components, hurdle_mean
from online_hurdle_calibration import CONTRACT, CalibrationConfig, OnlineHurdleCalibrator
from stochastic_price_forecast import load_timeline_split
from train_hurdle_sign_size_forecast import metrics
from train_neural_sign_size_forecast import HORIZONS, PRICE_ROOT, now, select_origins, sha, write_json

MODES = ("none", "zero", "probabilities", "full")


@dataclass(frozen=True)
class CachedStream:
    origins: np.ndarray
    timestamps: np.ndarray
    raw: np.ndarray
    prices: np.ndarray
    first_price_origin: int
    accuracy_indices: np.ndarray
    axis_start_ms: int
    horizon_seconds: int

    def target(self, index: int) -> float:
        offset = int(self.origins[index])-self.first_price_origin
        return float(self.prices[offset+self.horizon_seconds]/self.prices[offset]-1)

    def accuracy_targets(self) -> np.ndarray:
        offsets = self.origins[self.accuracy_indices]-self.first_price_origin
        return self.prices[offsets+self.horizon_seconds]/self.prices[offsets]-1


def candidates(horizon: int) -> list[dict]:
    result = [{"candidateId": "none", "config": asdict(CalibrationConfig(horizon_seconds=horizon, mode="none"))}]
    for mode in MODES[1:]:
        for half_life in (3600, 14400, 57600):
            for ridge in (8, 32):
                result.append({"candidateId": f"{mode}-hl{half_life}-ridge{ridge}",
                    "config": asdict(CalibrationConfig(horizon_seconds=horizon, mode=mode,
                        half_life_seconds=half_life, ridge_precision=ridge, cadence_seconds=4))})
    return result


def immutable(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def simulate_stream(stream: CachedStream, config: CalibrationConfig, *, trace: bool = False) -> dict:
    """Process only already-issued forecasts at their exact observed maturity.

The metric/export query set never enters this function. Every canonical issue
receives one immutable raw forecast and exactly one future maturity update.
"""
    if config.horizon_seconds != stream.horizon_seconds:
        raise ValueError("calibration and cached horizon disagree")
    if np.any(np.diff(stream.timestamps) != config.cadence_seconds*1000):
        raise ValueError("canonical issuance cadence changed")
    calibrator = OnlineHurdleCalibrator(config)
    count, pending = len(stream.origins), 0
    output = np.empty_like(stream.raw)
    snapshots = {"coefficients":np.empty((count,4)),"discountedInformation":np.empty((count,4)),
        "componentUpdates":np.empty((count,4),dtype=np.int64),"boundaryUpdates":np.empty((count,4),dtype=np.int64),
        "latestMaturityUsed":np.empty(count,dtype=np.int64),"maturedObservations":np.empty(count,dtype=np.int64)} if trace else {}
    maturity = stream.timestamps+stream.horizon_seconds*1000
    for index,stamp in enumerate(stream.timestamps):
        # Updates run at their actual maturity, including between four-second
        # queries for H=1. apply() is read-only and may decay to the query time.
        while pending < index and maturity[pending] <= stamp:
            calibrator.update(int(stream.timestamps[pending]), stream.raw[pending], stream.target(pending))
            pending += 1
        output[index] = calibrator.apply(int(stamp), stream.raw[index])
        if trace:
            snapshot = calibrator.snapshot(int(stamp))
            if snapshot["maturedObservations"] != pending or snapshot["latestMaturityUsed"] != (None if pending == 0 else int(maturity[pending-1])):
                raise ValueError("canonical maturity and calibrator state disagree")
            for key in snapshots:
                snapshots[key][index] = -1 if snapshot[key] is None else snapshot[key]
    # The split continues to its last observed price after its last complete
    # forecast origin. Flush remaining maturities without issuing new queries.
    terminal = stream.axis_start_ms+(stream.first_price_origin+len(stream.prices)-1)*1000
    while pending < count and maturity[pending] <= terminal:
        calibrator.update(int(stream.timestamps[pending]), stream.raw[pending], stream.target(pending))
        pending += 1
    if pending != count or not np.isfinite(output).all():
        raise ValueError("unmatured within-split forecast or nonfinite calibration")
    return {"parameters": output, "snapshots": snapshots,
        "finalState": calibrator.snapshot(int(terminal)), "canonicalIssues": count,
        "maturedUpdates": pending, "terminalTimestamp": int(terminal)}


@torch.inference_mode()
def infer_raw(model, timeline: np.ndarray, origins: np.ndarray, accuracy_count: int,
              device: torch.device, batch: int) -> np.ndarray:
    pieces = []
    # Preserve the previous common-origin evaluation's batch boundaries for
    # its prefix; extended calibration/replay origins are inferred only once.
    for lo,hi in ((0, accuracy_count), (accuracy_count, len(origins))):
        for start in range(lo, hi, batch):
            rows = origins[start:min(start+batch, hi)]
            features = np.asarray(timeline[rows[:, None]+np.array([-1,0])], dtype=np.float32)
            pieces.append(model(torch.from_numpy(features).to(device))[:,0].cpu().numpy())
    result = np.concatenate(pieces)
    if result.shape != (len(origins),6) or result.dtype != np.float64 or not np.isfinite(result).all():
        raise ValueError("invalid cached frozen forecast")
    return result


def cache_split(split: str, source: dict, source_file: Path, output: Path,
                plan_sha: str, device: torch.device, batch: int) -> tuple[dict[int,CachedStream], list[dict]]:
    dataset = Path(source["plan"]["dataset"])
    timeline = load_timeline_split(dataset, split)
    common = select_origins(timeline, 4, None)
    expected = 34560 if split == "validation" else 36134
    if len(common) != expected:
        raise ValueError("source common-origin evaluation population changed")
    first = int(timeline.origins[0])
    price_times = timeline.axis_start_ms+np.arange(first,len(timeline.timeline),dtype=np.int64)*1000
    prices = load_close_prices(price_times, PRICE_ROOT)
    if not np.all(np.isfinite(prices) & (prices > 0)):
        raise ValueError("observed tape must contain positive finite prices")
    prices = immutable(prices)
    market_file = output/"cache"/f"{split}-observed-market.npz"
    market_file.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(market_file, prices=prices, commonOrigins=common,
                        firstPriceOrigin=first, axisStartMs=timeline.axis_start_ms)
    streams, manifests = {}, []
    for entry in source["horizons"]:
        h = entry["horizonSeconds"]
        origins = timeline.origins[timeline.origins+h < len(timeline.timeline)][::4]
        accuracy_indices = np.searchsorted(origins,common)
        np.testing.assert_array_equal(origins[accuracy_indices],common)
        np.testing.assert_array_equal(accuracy_indices,np.arange(len(common)))
        record = entry["finetuned"]
        model = load_combined(record,h,source["planSha256"],device)
        raw = infer_raw(model,timeline.timeline,origins,len(common),device,batch)
        del model
        stamps = timeline.axis_start_ms+origins*1000
        file = output/"cache"/f"{split}-h{h}.npz"
        np.savez_compressed(file,origins=origins,timestamps=stamps,raw=raw,accuracyIndices=accuracy_indices)
        manifest = {"contract":CONTRACT,"split":split,"horizonSeconds":h,"cacheFile":str(file),"cacheSha256":sha(file),
            "marketFile":str(market_file),"marketSha256":sha(market_file),"sourceSelectionFile":str(source_file),
            "sourceSelectionSha256":sha(source_file),"checkpoint":record["checkpoint"],"checkpointSha256":record["sha256"],
            "planSha256":plan_sha,"canonicalOrigins":len(origins),"accuracyOrigins":len(common),
            "firstTimestamp":int(stamps[0]),"lastCanonicalTimestamp":int(stamps[-1]),"lastObservedTimestamp":int(price_times[-1]),
            "causalFeatures":"raw production59 states at origin-1 and origin; every raw forecast inferred once",
            "maturity":"actual endpoint return is revealed only at origin+horizon; stored tape is replay/evaluation data",
            "cadenceSeconds":4,"accuracyMaximumHorizonSeconds":max(HORIZONS)}
        write_json(file.with_suffix(".manifest.json"),manifest)
        streams[h] = CachedStream(immutable(origins),immutable(stamps),immutable(raw),prices,first,
            immutable(accuracy_indices),timeline.axis_start_ms,h)
        manifests.append(manifest)
        print(json.dumps({"event":"cached","split":split,"horizonSeconds":h,"canonicalOrigins":len(origins),"accuracyOrigins":len(common)}),flush=True)
    return streams, manifests


def save_simulation(result: dict, stream: CachedStream, split: str, mode: str, root: Path) -> dict:
    file = root/"calibrated"/f"{split}-{mode}-h{stream.horizon_seconds}.npz"
    file.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(file,parameters=result["parameters"],timestamps=stream.timestamps,
        origins=stream.origins,accuracyIndices=stream.accuracy_indices)
    trace_file = file.with_suffix(".trace.npz")
    np.savez_compressed(trace_file,timestamps=stream.timestamps,origins=stream.origins,**result["snapshots"])
    return {"parametersFile":str(file),"parametersSha256":sha(file),"stateTraceFile":str(trace_file),
        "stateTraceSha256":sha(trace_file),"canonicalIssues":result["canonicalIssues"],
        "maturedUpdates":result["maturedUpdates"],"finalState":result["finalState"],
        "terminalTimestamp":result["terminalTimestamp"],"traceFormat":"NPZ; canonical issue timestamps, coefficients, information, update counts; -1 denotes no incorporated maturity"}


def evaluate_mode(stream: CachedStream, config: dict, constants: dict, *, trace: bool = False) -> tuple[dict,dict]:
    simulated = simulate_stream(stream,CalibrationConfig(**config),trace=trace)
    scored = metrics(simulated["parameters"][stream.accuracy_indices],stream.accuracy_targets(),constants)
    scored["calibration"] = {"config":config,"canonicalIssues":simulated["canonicalIssues"],
        "maturedUpdates":simulated["maturedUpdates"],"coldStart":True}
    return scored,simulated


def export_mode(stream: CachedStream, parameters: np.ndarray, mode: str, config_record: dict,
                selection_file: Path, selected: dict, source_entry: dict, cache_manifest: dict,
                simulation_record: dict, trace_arrays: dict, replay: dict, output: Path) -> dict:
    h = stream.horizon_seconds
    count = replay["decisionCountsByHorizon"][str(h)]
    origins = stream.origins[0]+np.arange(count,dtype=np.int64)*h
    indices = np.searchsorted(stream.origins,origins)
    np.testing.assert_array_equal(stream.origins[indices],origins)
    stamps = stream.timestamps[indices]
    if int(stamps[0]) != replay["firstTimestamp"] or int(stamps[-1])+h*1000 != replay["terminalTimestamp"] \
            or count*h != replay["comparisonObservedSeconds"]:
        raise ValueError("online replay interval differs from registered common period")
    packed = torch.from_numpy(parameters[indices,None,:])
    p,up,down = (v[:,0].numpy() for v in hurdle_components(packed))
    returns,probabilities,mean,size,normalization = endpoint_law(p,up,down)
    discrepancy = float(np.max(np.abs(mean-hurdle_mean(packed)[:,0].numpy())))
    if discrepancy > 1e-10:
        raise ValueError("calibrated endpoint mean does not match the declared hurdle law")
    file = output/f"{mode}-h{h}.jsonl"
    with file.open("w",encoding="utf-8",newline="\n") as sink:
        for i,origin in enumerate(origins):
            offset = int(origin)-stream.first_price_origin
            current = float(stream.prices[offset])
            row = {"horizonSeconds":h,"timestamp":int(stamps[i]),"observedPrice":current,
                "realizedNextTimestamp":int(stamps[i]+h*1000),"realizedNextPrice":float(stream.prices[offset+h]),
                "realizedPrices":stream.prices[offset+1:offset+h+1].tolist(),
                "tree":{"children":[{"probability":float(probabilities[i,k]),"price":current*float(1+returns[i,k])} for k in range(3)]}}
            sink.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    record = source_entry["finetuned"]
    replay_states = [{"timestamp":int(stamps[i]),
        "latestMaturityUsed":None if trace_arrays["latestMaturityUsed"][index] < 0 else int(trace_arrays["latestMaturityUsed"][index]),
        "maturedObservations":int(trace_arrays["maturedObservations"][index]),
        "coefficients":trace_arrays["coefficients"][index].tolist()} for i,index in enumerate(indices)]
    manifest = {"contract":CONTRACT,"variant":mode,"combined":record,"checkpoint":record["checkpoint"],
        "checkpointSha256":record["sha256"],"sourceSelectionFile":selected["sourceSelectionFile"],
        "sourceSelectionSha256":selected["sourceSelectionSha256"],"selectionFile":str(selection_file),
        "selectionSha256":sha(selection_file),"headSelectionsFrozenBeforeTest":True,
        "calibrationSettingsFrozenBeforeTest":True,"testUsedForSelection":False,"refit":False,"reselection":False,
        "neuralRefit":False,"onlineLabelUpdates":mode!="none","calibrationColdStart":True,
        "onlineUpdates":"only previously issued canonical raw forecasts whose endpoint has been observed; cold split reset",
        "selectedCalibration":config_record,"canonicalCache":cache_manifest,"calibratedStream":simulation_record,
        "selectedConfig":config_record["config"],"calibrationTraceFile":simulation_record["stateTraceFile"],
        "calibrationTraceSha256":simulation_record["stateTraceSha256"],"replayStateSamples":replay_states,
        "replayCanonicalIndices":indices.tolist(),"calibrationShapeUpdates":False,
        "forecastFile":str(file),"forecastSha256":sha(file),"split":"test","rows":count,"horizonSeconds":h,
        "decisionStrideSeconds":h,"firstTimestamp":int(stamps[0]),"lastRealizedTimestamp":int(stamps[-1]+h*1000),
        "observedSeconds":count*h,"probabilityOrder":["down","zero","up"],
        "predictedZeroProbabilityMean":float(probabilities[:,1].mean()),
        "predictedSignedMeanBps":float(mean.mean()*1e4),"predictedAverageAbsoluteSizeBps":float(size.mean()*1e4),
        "predictedMinMeanBps":float(mean.min()*1e4),"predictedMaxMeanBps":float(mean.max()*1e4),
        "fullFeeThresholdLongCount":int((mean>1/.9995**2-1).sum()),
        "maximumProbabilitySumErrorBeforeNormalization":normalization,"maximumExportVsTorchMeanErrorBps":discrepancy*1e4,
        "sourceMarket":"BTCUSDT spot one-second closes with synthetic Oracle account parameters",
        "limitation":"Online Fisher updates approximate regularized likelihood; overlap weighting is heuristic. Three conditional-mean supports discard magnitude spread; only restricted long/cash endpoint policy consumes these exports."}
    write_json(file.with_suffix(".manifest.json"),manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-selection",type=Path,default=Path("data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json"))
    parser.add_argument("--output",type=Path,default=Path("data/training/runs/stochastic-crra-online-calibration-v1"))
    parser.add_argument("--forecasts",type=Path,default=Path("data/backtests/oracle-online-calibration"))
    parser.add_argument("--batch-size",type=int,default=2560)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--run-prepared",action="store_true")
    parser.add_argument("--validation-only",action="store_true")
    parser.add_argument("--evaluate-test",action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("positive inference batch size required")
    source = frozen_selection(args.source_selection)
    dataset = Path(source["plan"]["dataset"])
    if sha(dataset/"manifest.json") != source["plan"]["datasetManifestSha256"]:
        raise ValueError("source dataset manifest changed")
    plan = {"contract":CONTRACT,"createdAt":now(),"sourceSelectionFile":str(args.source_selection),
        "sourceSelectionSha256":sha(args.source_selection),"dataset":str(dataset),
        "datasetManifestSha256":source["plan"]["datasetManifestSha256"],"horizonsSeconds":list(HORIZONS),
        "sourceStage":"latest frozen hurdle NLL finetuned selections only","neuralRetraining":False,
        "canonicalCadenceSeconds":4,"canonicalOrigins":"each horizon's complete within-split origins at cadence4 from split.origins[0], including origins after common accuracy cutoff",
        "features":"two observed raw production59 states at t-1,t; cache each frozen neural forecast once",
        "accuracyOrigins":"same common max7200-complete origins at stride4 as previous specialist evaluations",
        "maturity":"at query time t first apply pending updates for issued origins with origin+H<=t at their exact maturity; then query without updating state",
        "onlineAlgorithm":"discounted Fisher-quadratic approximation: independent zero/sign bias and UP-logmean/DOWN-logoddsmean scales; frozen shapes",
        "updateWeight":"min(1,4/H), overlap heuristic rather than an independence guarantee",
        "coldStart":"identity separately for validation and test; no carried calibration state",
        "candidateGrid":{str(h):candidates(h) for h in HORIZONS},
        "selection":"minimum full validation NLL within each of none/zero/probabilities/full, deterministic grid-order tie break; all four modes reported",
        "testProtocol":"all four horizons' mode settings frozen before test timeline, forecasts or outcomes load; no test-based model/mode choice",
        "inputBatchSize":args.batch_size,"precision":"float32 neural networks/high matmul precision; float64 frozen likelihood and calibration",
        "forecastOutput":str(args.forecasts),"replay":"four fixed modes at15m/30m/2h; common42h; same saved canonical forecast stream; no extra updates from replay queries",
        "testUsedForSelection":False,"benchmark":"previously inspected chronological test split; not a fresh blind holdout"}
    path = args.output/"plan.json"
    if args.run_prepared or args.evaluate_test:
        previous = json.loads(path.read_text())
        plan["createdAt"] = previous["createdAt"]
        if plan != previous or ((args.output/"cache").exists() and not args.evaluate_test):
            raise ValueError("prepared plan changed or already evaluated")
        if args.evaluate_test and (args.output/"cache"/"test-observed-market.npz").exists():
            raise ValueError("test cache already exists; refusing repeated evaluation")
    else:
        if path.exists():
            raise ValueError("refusing to overwrite immutable calibration plan")
        write_json(path,plan)
    if args.prepare_only:
        print(json.dumps({"event":"prepared","plan":str(path),"sha256":sha(path)}),flush=True)
        return
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    if args.evaluate_test:
        selection_file = args.output/"selection.json"
        selected = json.loads(selection_file.read_text())
        if not selected["allCalibrationSelectionsFrozen"] or selected["testLoaded"] or selected["testUsedForSelection"] \
                or selected["planSha256"] != sha(path) or selected["sourceSelectionSha256"] != sha(args.source_selection) \
                or selected["validationCandidatesSha256"] != sha(Path(selected["validationCandidatesFile"])):
            raise ValueError("calibration choices not frozen before test")
        validation_results = json.loads((args.output/"validation.json").read_text())["horizons"]
        evaluate_test(args,source,selected,validation_results,device,started)
        return
    streams,validation_caches = cache_split("validation",source,args.source_selection,args.output,sha(path),device,args.batch_size)
    candidate_results,entries,validation_results = [],[],[]
    for source_entry in source["horizons"]:
        h,constants = source_entry["horizonSeconds"],source_entry["constants"]
        rows = []
        for candidate in plan["candidateGrid"][str(h)]:
            scored,_ = evaluate_mode(streams[h],candidate["config"],constants)
            row = {**candidate,"validation":scored,"validationNll":scored["fullHurdleReturnNegativeLogLikelihood"]}
            rows.append(row)
        selected_modes,mode_metrics = {},{}
        for mode in MODES:
            chosen = min((row for row in rows if row["config"]["mode"]==mode),key=lambda row:row["validationNll"])
            selected_modes[mode] = {key:chosen[key] for key in ("candidateId","config","validationNll")}
            scored,simulated = evaluate_mode(streams[h],chosen["config"],constants,trace=True)
            if scored["fullHurdleReturnNegativeLogLikelihood"] != chosen["validationNll"]:
                raise ValueError("calibration query/trace changed validation selection score")
            scored["savedStream"] = save_simulation(simulated,streams[h],"validation",mode,args.output)
            mode_metrics[mode] = scored
        preferred = min(MODES,key=lambda mode:selected_modes[mode]["validationNll"])
        entries.append({"horizonSeconds":h,"constants":constants,"checkpoint":source_entry["finetuned"],
            "modes":selected_modes,"validationPreferredMode":preferred})
        candidate_results.append({"horizonSeconds":h,"candidates":rows})
        validation_results.append({"horizonSeconds":h,"models":mode_metrics})
        print(json.dumps({"event":"validationSelected","horizonSeconds":h,"modes":selected_modes}),flush=True)
    write_json(args.output/"validation-candidates.json",{"contract":CONTRACT,"horizons":candidate_results,"testLoaded":False})
    write_json(args.output/"validation.json",{"contract":CONTRACT,"horizons":validation_results,"cacheManifests":validation_caches})
    selected = {"contract":CONTRACT,"createdAt":now(),"plan":plan,"planSha256":sha(path),
        "sourceSelectionFile":str(args.source_selection),"sourceSelectionSha256":sha(args.source_selection),
        "horizons":entries,"allCalibrationSelectionsFrozen":True,"allHeadSelectionsFrozen":True,
        "testLoaded":False,"testUsedForSelection":False,
        "validationCandidatesFile":str(args.output/"validation-candidates.json"),
        "validationCandidatesSha256":sha(args.output/"validation-candidates.json")}
    selection_file = args.output/"selection.json"
    write_json(selection_file,selected)
    del streams
    if args.validation_only:
        print(json.dumps({"event":"validationFrozen","selection":str(selection_file),"sha256":sha(selection_file),
            "seconds":time.monotonic()-started}),flush=True)
        return
    evaluate_test(args,source,selected,validation_results,device,started)


def evaluate_test(args,source:dict,selected:dict,validation_results:list,device:torch.device,started:float) -> None:
    selection_file = args.output/"selection.json"
    selection_sha = sha(selection_file)
    entries = selected["horizons"]
    streams,test_caches = cache_split("test",source,args.source_selection,args.output,selected["planSha256"],device,args.batch_size)
    torch.cuda.empty_cache()
    replay = json.loads((args.forecasts/"replay-plan.json").read_text())
    exports,test_results = [],[]
    for index,entry in enumerate(entries):
        h = entry["horizonSeconds"]
        modes = {}
        for mode in MODES:
            chosen = entry["modes"][mode]
            scored,simulated = evaluate_mode(streams[h],chosen["config"],entry["constants"],trace=True)
            saved_stream = save_simulation(simulated,streams[h],"test",mode,args.output)
            scored["savedStream"] = saved_stream
            modes[mode] = scored
            if h in replay["horizonsSeconds"]:
                exports.append(export_mode(streams[h],simulated["parameters"],mode,chosen,selection_file,selected,
                    source["horizons"][index],test_caches[index],saved_stream,simulated["snapshots"],replay,args.forecasts))
        test_results.append({"horizonSeconds":h,"models":modes})
        print(json.dumps({"event":"testComplete","horizonSeconds":h,"nll":{mode:modes[mode]["fullHurdleReturnNegativeLogLikelihood"] for mode in MODES}}),flush=True)
    if sha(args.source_selection) != selected["sourceSelectionSha256"] or sha(selection_file) != selection_sha:
        raise ValueError("source selection changed during calibration")
    frozen_selection(args.source_selection)
    write_json(args.output/"test.json",{"contract":CONTRACT,"horizons":test_results,"cacheManifests":test_caches,"testUsedForSelection":False})
    write_json(args.forecasts/"exports.json",{"contract":CONTRACT,"manifests":exports})
    result = {"contract":CONTRACT,"selectionFile":str(selection_file),"selectionSha256":sha(selection_file),
        "validation":validation_results,"test":test_results,"testUsedForSelection":False,"seconds":time.monotonic()-started}
    write_json(args.output/"result.json",result)
    print(json.dumps({"event":"allComplete","result":str(args.output/"result.json"),"seconds":result["seconds"]}),flush=True)


if __name__ == "__main__":
    main()
