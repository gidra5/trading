"""Compare one shared hurdle triplet with the frozen per-horizon baseline.

The shared network emits all four horizons in one forward pass. Calibration
settings are inherited unchanged from the preceding validation-only experiment;
each split starts from identity and incorporates only matured labels.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_online_hurdle_calibration import (CachedStream, immutable,
    save_simulation, simulate_stream)
from export_hurdle_sign_size_forecast import frozen_selection as frozen_baseline
from export_ternary_sign_size_forecast import endpoint_law
from hurdle_sign_size_forecast import CONTRACT as MODEL_CONTRACT, hurdle_components, hurdle_mean
from online_hurdle_calibration import CONTRACT as CALIBRATION_CONTRACT, CalibrationConfig
from stochastic_price_forecast import load_timeline_split
from train_hurdle_sign_size_forecast import load_combined, metrics
from train_neural_sign_size_forecast import HORIZONS, now, select_origins, sha, write_json

CONTRACT = "shared-horizon-hurdle-evaluation-v1"
SHARED_CONTRACT = "shared-horizon-hurdle-finetune-v1"
VARIANTS = ("separate-raw", "separate-online", "shared-raw", "shared-online")


def read_json(file: Path) -> dict:
    return json.loads(file.read_text(encoding="utf-8"))


def checked_file(file: str | Path, expected: str) -> Path:
    path = Path(file)
    if sha(path) != expected:
        raise ValueError(f"immutable artifact changed: {path}")
    return path


def frozen_shared(file: Path) -> dict:
    selected = read_json(file)
    if selected["contract"] != SHARED_CONTRACT or selected["allHeadSelectionsFrozen"] is not True \
            or selected["testLoaded"] is not False or selected["testUsedForSelection"] is not False:
        raise ValueError("shared checkpoint selection must be frozen before test")
    plan_file = checked_file(file.parent/"plan.json", selected["planSha256"])
    if read_json(plan_file) != selected["plan"] or [row["horizonSeconds"] for row in selected["horizons"]] != list(HORIZONS):
        raise ValueError("shared plan or horizon alignment differs")
    for stage in ("pretrained", "finetuned"):
        record = selected[stage]
        checkpoint = torch.load(checked_file(record["checkpoint"], record["sha256"]), map_location="cpu", weights_only=True)
        if checkpoint["contract"] != MODEL_CONTRACT or checkpoint["task"] != "combined" \
                or checkpoint["horizonsSeconds"] != list(HORIZONS) or checkpoint["epoch"] != record["epoch"] \
                or checkpoint["planSha256"] != selected["planSha256"]:
            raise ValueError("shared checkpoint provenance differs")
    return selected


def frozen_calibration(file: Path, source_file: Path) -> dict:
    selected = read_json(file)
    if selected["contract"] != CALIBRATION_CONTRACT or selected["allCalibrationSelectionsFrozen"] is not True \
            or selected["allHeadSelectionsFrozen"] is not True or selected["testLoaded"] is not False \
            or selected["testUsedForSelection"] is not False or selected["sourceSelectionSha256"] != sha(source_file):
        raise ValueError("calibration selections were not frozen on the source models")
    plan_file = checked_file(file.parent/"plan.json", selected["planSha256"])
    checked_file(selected["validationCandidatesFile"], selected["validationCandidatesSha256"])
    if read_json(plan_file) != selected["plan"]:
        raise ValueError("calibration plan changed")
    return selected


def baseline_stream(split: str, h: int, calibration_file: Path, calibration: dict,
                    source: dict) -> tuple[CachedStream, dict]:
    root = calibration_file.parent
    manifest = read_json(root/"cache"/f"{split}-h{h}.manifest.json")
    entry = next(row for row in source["horizons"] if row["horizonSeconds"] == h)
    if manifest["checkpointSha256"] != entry["finetuned"]["sha256"] \
            or manifest["sourceSelectionSha256"] != calibration["sourceSelectionSha256"] \
            or manifest["planSha256"] != calibration["planSha256"]:
        raise ValueError("baseline cache does not belong to the selected model")
    with np.load(checked_file(manifest["cacheFile"], manifest["cacheSha256"]), allow_pickle=False) as cached:
        arrays = {key: immutable(cached[key]) for key in ("origins", "timestamps", "raw", "accuracyIndices")}
    with np.load(checked_file(manifest["marketFile"], manifest["marketSha256"]), allow_pickle=False) as market:
        prices = immutable(market["prices"])
        first, axis = int(market["firstPriceOrigin"]), int(market["axisStartMs"])
        np.testing.assert_array_equal(arrays["origins"][arrays["accuracyIndices"]], market["commonOrigins"])
    if arrays["raw"].shape != (len(arrays["origins"]),6) or arrays["raw"].dtype != np.float64 \
            or not np.isfinite(arrays["raw"]).all() or not np.all(np.isfinite(prices) & (prices > 0)):
        raise ValueError("invalid baseline cache values")
    np.testing.assert_array_equal(arrays["timestamps"], axis+arrays["origins"]*1000)
    return CachedStream(arrays["origins"], arrays["timestamps"], arrays["raw"], prices,
        first, arrays["accuracyIndices"], axis, h), manifest


def load_saved_simulation(split: str, mode: str, h: int, calibration_file: Path,
                          stream: CachedStream, config: dict) -> tuple[np.ndarray, dict, dict]:
    report = read_json(calibration_file.parent/f"{split}.json")
    entry = next(row for row in report["horizons"] if row["horizonSeconds"] == h)
    if entry["models"][mode]["calibration"]["config"] != config:
        raise ValueError("reused baseline calibration differs from frozen settings")
    record = entry["models"][mode]["savedStream"]
    with np.load(checked_file(record["parametersFile"], record["parametersSha256"]), allow_pickle=False) as saved:
        parameters = immutable(saved["parameters"])
        for key, expected in (("origins",stream.origins),("timestamps",stream.timestamps),("accuracyIndices",stream.accuracy_indices)):
            np.testing.assert_array_equal(saved[key],expected)
    with np.load(checked_file(record["stateTraceFile"], record["stateTraceSha256"]), allow_pickle=False) as saved:
        trace = {key:immutable(saved[key]) for key in saved.files if key not in ("origins","timestamps")}
        np.testing.assert_array_equal(saved["origins"],stream.origins)
        np.testing.assert_array_equal(saved["timestamps"],stream.timestamps)
    if parameters.shape != stream.raw.shape or not np.isfinite(parameters).all():
        raise ValueError("invalid calibrated baseline")
    if mode == "none":
        np.testing.assert_array_equal(parameters,stream.raw)
    validate_trace(stream, trace)
    return parameters, trace, record


def validate_trace(stream: CachedStream, trace: dict) -> None:
    maturity = stream.timestamps+stream.horizon_seconds*1000
    counts = np.searchsorted(maturity,stream.timestamps,side="right")
    last = np.where(counts == 0,-1,maturity[np.maximum(counts-1,0)])
    np.testing.assert_array_equal(trace["maturedObservations"],counts)
    np.testing.assert_array_equal(trace["latestMaturityUsed"],last)
    if trace["coefficients"].shape != (len(stream.origins),4) or not np.isfinite(trace["coefficients"]).all():
        raise ValueError("invalid calibration coefficients")


@torch.inference_mode()
def infer_shared(model: torch.nn.Module, timeline: np.ndarray, origins: np.ndarray,
                 accuracy_count: int, device: torch.device, batch: int) -> np.ndarray:
    """Every batch passes once through one zero, one sign, and one size expert."""
    pieces = []
    for lo,hi in ((0,accuracy_count),(accuracy_count,len(origins))):
        for start in range(lo,hi,batch):
            rows = origins[start:min(start+batch,hi)]
            features = np.asarray(timeline[rows[:,None]+np.array([-1,0])],dtype=np.float32)
            pieces.append(model(torch.from_numpy(features).to(device)).cpu().numpy())
    result = np.concatenate(pieces)
    if result.shape != (len(origins),len(HORIZONS),6) or result.dtype != np.float64 or not np.isfinite(result).all():
        raise ValueError("shared model must emit all four physical hurdle laws per input")
    return result


def cache_shared(split: str, shared: dict, selection_file: Path, baseline: dict[int,CachedStream],
                 output: Path, device: torch.device, batch: int) -> tuple[dict[int,CachedStream],dict,np.ndarray]:
    timeline = load_timeline_split(Path(shared["plan"]["dataset"]),split)
    common = select_origins(timeline,4,None)
    union = timeline.origins[timeline.origins+min(HORIZONS)<len(timeline.timeline)][::4]
    stamps = timeline.axis_start_ms+union*1000
    for h,stream in baseline.items():
        expected = timeline.origins[timeline.origins+h<len(timeline.timeline)][::4]
        np.testing.assert_array_equal(stream.origins,expected)
        np.testing.assert_array_equal(stream.origins[stream.accuracy_indices],common)
        np.testing.assert_array_equal(union[:len(expected)],expected)
        if timeline.axis_start_ms != stream.axis_start_ms:
            raise ValueError("shared and separate timelines differ")
    cache = output/"cache"
    cache.mkdir(parents=True,exist_ok=True)
    manifest = {"contract":CONTRACT,"split":split,"horizonsSeconds":list(HORIZONS),
        "selectionFile":str(selection_file),"selectionSha256":sha(selection_file),
        "canonicalUnionOrigins":len(union),"accuracyOrigins":len(common),
        "firstTimestamp":int(stamps[0]),"lastTimestamp":int(stamps[-1]),
        "inference":"one forward pass per batch outputs [origin,4 horizons,6 hurdle parameters]; no per-horizon model copies",
        "features":"raw production59 states at origin-1 and origin only",
        "horizonValidPrefixCounts":{str(h):len(stream.origins) for h,stream in baseline.items()},
        "fullHorizonMatrixTail":"longer horizons can be inferred near split end; only complete per-horizon prefixes are scored or updated"}
    models = {}
    for stage in ("pretrained","finetuned"):
        origins = common if stage == "pretrained" else union
        model = load_combined(shared[stage],device)
        raw = infer_shared(model,timeline.timeline,origins,len(common),device,batch)
        del model
        file = cache/f"{split}-shared-{stage}.npz"
        np.savez_compressed(file,origins=origins,timestamps=timeline.axis_start_ms+origins*1000,
            raw=raw,commonOrigins=common,horizons=np.asarray(HORIZONS,dtype=np.int64))
        models[stage] = raw
        manifest[stage] = {"cacheFile":str(file),"cacheSha256":sha(file),
            "checkpoint":shared[stage]["checkpoint"],"checkpointSha256":shared[stage]["sha256"],"shape":list(raw.shape)}
        print(json.dumps({"event":"sharedCached","split":split,"stage":stage,"shape":list(raw.shape)}),flush=True)
    write_json(cache/f"{split}-shared.manifest.json",manifest)
    matrix = immutable(models["finetuned"])
    streams = {}
    for j,h in enumerate(HORIZONS):
        stream = baseline[h]
        streams[h] = CachedStream(stream.origins,stream.timestamps,matrix[:len(stream.origins),j,:],stream.prices,
            stream.first_price_origin,stream.accuracy_indices,stream.axis_start_ms,h)
    return streams,manifest,models["pretrained"]


def export_variant(stream: CachedStream, parameters: np.ndarray, trace: dict, variant: str,
                   config: dict, checkpoint: dict, source_cache: dict, simulation: dict,
                   selection_file: Path, shared: dict, calibration_file: Path, calibration: dict,
                   replay: dict, output: Path) -> dict:
    h = stream.horizon_seconds
    count = replay["decisionCountsByHorizon"][str(h)]
    origins = stream.origins[0]+np.arange(count,dtype=np.int64)*h
    indices = np.searchsorted(stream.origins,origins)
    np.testing.assert_array_equal(stream.origins[indices],origins)
    stamps = stream.timestamps[indices]
    if int(stamps[0]) != replay["firstTimestamp"] or int(stamps[-1])+h*1000 != replay["terminalTimestamp"] \
            or count*h != replay["comparisonObservedSeconds"]:
        raise ValueError("shared replay period differs from the registered comparison")
    packed = torch.from_numpy(np.array(parameters[indices,None,:],copy=True))
    p,up,down = (value[:,0].numpy() for value in hurdle_components(packed))
    returns,probabilities,mean,size,normalization = endpoint_law(p,up,down)
    error = float(np.max(np.abs(mean-hurdle_mean(packed)[:,0].numpy())))
    if error > 1e-10:
        raise ValueError("export changed the model conditional mean")
    file = output/f"{variant}-h{h}.jsonl"
    with file.open("w",encoding="utf-8",newline="\n") as sink:
        for i,origin in enumerate(origins):
            offset = int(origin)-stream.first_price_origin
            current = float(stream.prices[offset])
            row = {"horizonSeconds":h,"timestamp":int(stamps[i]),"observedPrice":current,
                "realizedNextTimestamp":int(stamps[i]+h*1000),"realizedNextPrice":float(stream.prices[offset+h]),
                "realizedPrices":stream.prices[offset+1:offset+h+1].tolist(),
                "tree":{"children":[{"probability":float(probabilities[i,k]),"price":current*float(1+returns[i,k])} for k in range(3)]}}
            sink.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    states = [{"timestamp":int(stamps[i]),
        "latestMaturityUsed":None if trace["latestMaturityUsed"][index]<0 else int(trace["latestMaturityUsed"][index]),
        "maturedObservations":int(trace["maturedObservations"][index]),
        "coefficients":trace["coefficients"][index].tolist()} for i,index in enumerate(indices)]
    manifest = {"contract":CONTRACT,"variant":variant,"combined":checkpoint,
        "checkpoint":checkpoint["checkpoint"],"checkpointSha256":checkpoint["sha256"],
        "selectionFile":str(selection_file),"selectionSha256":sha(selection_file),
        "sourceSelectionFile":shared["sourceSelectionFile"],"sourceSelectionSha256":shared["sourceSelectionSha256"],
        "calibrationSelectionFile":str(calibration_file),"calibrationSelectionSha256":sha(calibration_file),
        "headSelectionsFrozenBeforeTest":True,"calibrationSettingsFrozenBeforeTest":True,
        "testUsedForSelection":False,"neuralRefit":False,"refit":False,"reselection":False,
        "onlineLabelUpdates":config["mode"]!="none","calibrationColdStart":True,"calibrationShapeUpdates":False,
        "selectedConfig":config,"sourceCalibrationPreferredMode":next(row["validationPreferredMode"] for row in calibration["horizons"] if row["horizonSeconds"]==h),
        "calibrationPolicy":"identical previously validation-selected settings applied to both architectures; no reselection",
        "canonicalCache":source_cache,"calibratedStream":simulation,
        "calibrationTraceFile":simulation["stateTraceFile"],"calibrationTraceSha256":simulation["stateTraceSha256"],
        "replayStateSamples":states,"replayCanonicalIndices":indices.tolist(),
        "sharedAcrossHorizons":variant.startswith("shared-"),"baselineArtifactsReused":variant.startswith("separate-"),
        "forecastFile":str(file),"forecastSha256":sha(file),"split":"test","rows":count,"horizonSeconds":h,
        "decisionStrideSeconds":h,"firstTimestamp":int(stamps[0]),"lastRealizedTimestamp":int(stamps[-1]+h*1000),
        "observedSeconds":count*h,"probabilityOrder":["down","zero","up"],
        "predictedZeroProbabilityMean":float(probabilities[:,1].mean()),"predictedSignedMeanBps":float(mean.mean()*1e4),
        "predictedAverageAbsoluteSizeBps":float(size.mean()*1e4),"predictedMinMeanBps":float(mean.min()*1e4),"predictedMaxMeanBps":float(mean.max()*1e4),
        "fullFeeThresholdLongCount":int((mean>1/.9995**2-1).sum()),
        "maximumProbabilitySumErrorBeforeNormalization":normalization,"maximumExportVsTorchMeanErrorBps":error*1e4,
        "sourceMarket":"BTCUSDT spot one-second closes with synthetic Oracle account parameters",
        "limitation":"Three conditional-mean supports discard magnitude spread; restricted long/cash endpoint policy only. Reused historical test benchmark."}
    write_json(file.with_suffix(".manifest.json"),manifest)
    return manifest


def evaluate_split(split: str, args, shared: dict, source: dict, calibration: dict,
                   device: torch.device, replay: dict) -> tuple[dict,list[dict]]:
    baseline,baseline_cache = {},{}
    for h in HORIZONS:
        baseline[h],baseline_cache[h] = baseline_stream(split,h,args.calibration_selection,calibration,source)
    streams,shared_cache,pretrained = cache_shared(split,shared,args.selection,baseline,args.output,device,args.batch_size)
    rows,exports = [],[]
    for j,h in enumerate(HORIZONS):
        source_entry = next(row for row in source["horizons"] if row["horizonSeconds"]==h)
        shared_entry = next(row for row in shared["horizons"] if row["horizonSeconds"]==h)
        calibration_entry = next(row for row in calibration["horizons"] if row["horizonSeconds"]==h)
        if shared_entry["constants"] != source_entry["constants"]:
            raise ValueError("forecast comparisons must share training constants")
        chosen = calibration_entry["modes"][calibration_entry["validationPreferredMode"]]
        raw_config = asdict(CalibrationConfig(horizon_seconds=h,mode="none"))
        configurations = {variant:chosen["config"] if variant.endswith("online") else raw_config for variant in VARIANTS}
        row = {"horizonSeconds":h,"calibrationConfig":chosen["config"],"models":{},
            "sharedPretrainedDiagnostic":metrics(pretrained[:,j,:],streams[h].accuracy_targets(),shared_entry["constants"])}
        for variant in VARIANTS:
            config = configurations[variant]
            if variant.startswith("separate-"):
                stream = baseline[h]
                parameters,trace,record = load_saved_simulation(split,config["mode"],h,args.calibration_selection,stream,config)
                cache,checkpoint = baseline_cache[h],source_entry["finetuned"]
            else:
                stream = streams[h]
                simulated = simulate_stream(stream,CalibrationConfig(**config),trace=True)
                parameters,trace = simulated["parameters"],simulated["snapshots"]
                validate_trace(stream,trace)
                record = save_simulation(simulated,stream,split,variant,args.output)
                cache,checkpoint = shared_cache,shared["finetuned"]
            scored = metrics(parameters[stream.accuracy_indices],stream.accuracy_targets(),shared_entry["constants"])
            scored.update({"savedStream":record,"calibrationConfig":config,"baselineArtifactsReused":variant.startswith("separate-")})
            row["models"][variant] = scored
            if split == "test" and h in replay["horizonsSeconds"]:
                exports.append(export_variant(stream,parameters,trace,variant,config,checkpoint,cache,record,
                    args.selection,shared,args.calibration_selection,calibration,replay,args.forecasts))
        rows.append(row)
        print(json.dumps({"event":"sharedComparison","split":split,"horizonSeconds":h,
            "nll":{variant:row["models"][variant]["fullHurdleReturnNegativeLogLikelihood"] for variant in VARIANTS},
            "mseSkill":{variant:row["models"][variant]["allTickSignedMeanMseSkillVsZero"] for variant in VARIANTS}}),flush=True)
    result = {"contract":CONTRACT,"split":split,"horizons":rows,"sharedCache":shared_cache,
        "baselineCaches":list(baseline_cache.values()),"testUsedForSelection":False}
    write_json(args.output/f"{split}.json",result)
    return result,exports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection",type=Path,default=Path("data/training/runs/stochastic-crra-shared-hurdle-v1/selection.json"))
    parser.add_argument("--calibration-selection",type=Path,default=Path("data/training/runs/stochastic-crra-online-calibration-v1/selection.json"))
    parser.add_argument("--output",type=Path,default=Path("data/training/runs/stochastic-crra-shared-hurdle-v1"))
    parser.add_argument("--forecasts",type=Path,default=Path("data/backtests/oracle-shared-hurdle"))
    parser.add_argument("--batch-size",type=int,default=2560)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--run-prepared",action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("positive inference batch required")
    shared = frozen_shared(args.selection)
    training_file = args.selection.parent/"training-result.json"
    training = read_json(training_file)
    if training["selectionSha256"] != sha(args.selection) or training["testLoaded"] or training["testUsedForSelection"] \
            or "train" not in training or "validation" not in training:
        raise ValueError("training and validation diagnostics must finish before evaluation")
    source_file = checked_file(shared["sourceSelectionFile"],shared["sourceSelectionSha256"])
    source = frozen_baseline(source_file)
    calibration = frozen_calibration(args.calibration_selection,source_file)
    if shared["onlineSelectionSha256"] != sha(args.calibration_selection):
        raise ValueError("online settings differ from the preregistered shared training comparison")
    dataset = Path(shared["plan"]["dataset"])
    checked_file(dataset/"manifest.json",shared["plan"]["datasetManifestSha256"])
    if shared["plan"]["datasetManifestSha256"] != source["plan"]["datasetManifestSha256"]:
        raise ValueError("shared and separate datasets differ")
    replay_file = args.forecasts/"replay-plan.json"
    replay = read_json(replay_file)
    if Path(replay["runsRoot"]).resolve() != args.output.resolve() \
            or [row["name"] for row in replay["variants"]] != list(VARIANTS):
        raise ValueError("replay is not the registered shared-versus-separate comparison")
    baseline_artifacts = []
    for split in ("validation","test"):
        for file in (args.calibration_selection.parent/f"{split}.json",
                     *(args.calibration_selection.parent/"cache"/f"{split}-h{h}.manifest.json" for h in HORIZONS)):
            baseline_artifacts.append({"file":str(file),"sha256":sha(file)})
    plan = {"contract":CONTRACT,"createdAt":now(),"selectionFile":str(args.selection),"selectionSha256":sha(args.selection),
        "trainingResultFile":str(training_file),"trainingResultSha256":sha(training_file),
        "trainingPlanSha256":shared["planSha256"],"sourceSelectionFile":str(source_file),"sourceSelectionSha256":sha(source_file),
        "calibrationSelectionFile":str(args.calibration_selection),"calibrationSelectionSha256":sha(args.calibration_selection),
        "replayPlanFile":str(replay_file),"replayPlanSha256":sha(replay_file),"variants":list(VARIANTS),"horizonsSeconds":list(HORIZONS),
        "sharedCheckpoints":{stage:shared[stage] for stage in ("pretrained","finetuned")},
        "frozenCalibration":{str(row["horizonSeconds"]):row["modes"][row["validationPreferredMode"]] for row in calibration["horizons"]},
        "baselineArtifactMetadata":baseline_artifacts,
        "sharedInference":"single all-horizon [N,4,6] matrix per checkpoint/split; fine union includes every H1-complete stride4 origin; longer horizons slice complete prefixes",
        "baseline":"hash-checked existing canonical neural predictions and causal calibration streams; no reinference",
        "accuracyPopulation":"same common max7200-complete stride4 origins as preceding experiments",
        "calibration":"previous validation-preferred mode and settings reused identically for both architectures without reselection; independent cold state per split/horizon",
        "maturity":"only previously issued raw forecasts with origin+H<=query time; exact maturity; no query-triggered extra update",
        "selection":"shared global checkpoint already frozen; test never used for selection; pretrained raw diagnostics are descriptive",
        "batchSize":args.batch_size,"testUsedForSelection":False,"benchmark":"previously inspected chronological test split, not a fresh blind holdout"}
    plan_file = args.output/"evaluation-plan.json"
    if args.run_prepared:
        previous = read_json(plan_file)
        plan["createdAt"] = previous["createdAt"]
        if plan != previous:
            raise ValueError("prepared shared evaluation plan changed")
    else:
        if plan_file.exists():
            raise ValueError("refusing to overwrite immutable evaluation plan")
        write_json(plan_file,plan)
    if args.prepare_only:
        print(json.dumps({"event":"sharedEvaluationPrepared","file":str(plan_file),"sha256":sha(plan_file)}),flush=True)
        return
    if (args.output/"cache").exists() or (args.output/"result.json").exists():
        raise ValueError("refusing to rerun or overwrite a shared evaluation")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    validation,_ = evaluate_split("validation",args,shared,source,calibration,device,replay)
    test,exports = evaluate_split("test",args,shared,source,calibration,device,replay)
    checked_file(args.selection,plan["selectionSha256"])
    checked_file(args.calibration_selection,plan["calibrationSelectionSha256"])
    for artifact in plan["baselineArtifactMetadata"]:
        checked_file(artifact["file"],artifact["sha256"])
    frozen_shared(args.selection)
    write_json(args.forecasts/"exports.json",{"contract":CONTRACT,"manifests":exports})
    result = {"contract":CONTRACT,"selectionFile":str(args.selection),"selectionSha256":sha(args.selection),
        "evaluationPlanFile":str(plan_file),"evaluationPlanSha256":sha(plan_file),"calibrationSelectionFile":str(args.calibration_selection),
        "calibrationSelectionSha256":sha(args.calibration_selection),"validation":validation["horizons"],"test":test["horizons"],
        "testUsedForSelection":False,"calibrationReselection":False,"seconds":time.monotonic()-started}
    write_json(args.output/"result.json",result)
    print(json.dumps({"event":"sharedEvaluationComplete","result":str(args.output/"result.json"),"seconds":result["seconds"]}),flush=True)


if __name__ == "__main__":
    main()
