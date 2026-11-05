"""Matched retained-vs-retrained evaluation with independent horizon models."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="4"
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from evaluate_log_hurdle_forecast import (CONTRACT,array_sha,score_log_law,save_stream,
    simulate_log_stream,export_stream,raw_up_bound_diagnostics)
from evaluate_online_hurdle_calibration import CachedStream,immutable
from online_log_hurdle_calibration import CalibrationConfig
from stochastic_price_forecast import load_timeline_split
from train_neural_sign_size_forecast import HORIZONS,select_origins,sha,write_json,now
from train_per_horizon_log_hurdle import CONTRACT as TRAIN_CONTRACT,load_combined,verified


@torch.inference_mode()
def infer_horizon(model,timeline,origins,common_count,batch,device,h):
    if model.horizons!=(h,): raise ValueError("wrong horizon model")
    pieces=[]
    # Preserve the original validation batch boundaries for numerical parity.
    for lo,hi in ((0,common_count),(common_count,len(origins))):
        for start in range(lo,hi,batch):
            rows=origins[start:min(start+batch,hi)]
            x=np.asarray(timeline[rows[:,None]+np.array([-1,0])],dtype=np.float32)
            pieces.append(model(torch.from_numpy(x).to(device)).cpu().numpy()[:,0])
    result=np.concatenate(pieces)
    if result.shape!=(len(origins),6) or result.dtype!=np.float64 or not np.isfinite(result).all():
        raise ValueError("invalid per-horizon predictions")
    return immutable(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--selection",type=Path,default=Path("data/training/runs/stochastic-crra-per-horizon-log-v1/selection.json"))
    parser.add_argument("--output",type=Path,default=Path("data/backtests/oracle-per-horizon-log"))
    parser.add_argument("--baseline",type=Path,default=Path("data/backtests/oracle-log-hurdle/evaluation.json"))
    parser.add_argument("--batch-size",type=int,default=2560)
    args=parser.parse_args()
    if (args.output/"evaluation.json").exists(): raise ValueError("refusing to overwrite completed evaluation")
    torch.set_num_threads(4);torch.set_float32_matmul_precision("high")
    device=torch.device("cuda")
    tracked={}
    def checked(file,expected=None):
        file=Path(file);digest=sha(file)
        if expected is not None and digest!=expected: raise ValueError(f"immutable artifact changed: {file}")
        tracked[str(file)]=digest
        return file
    def read(file,expected=None): return json.loads(checked(file,expected).read_text())
    def arrays(file,digest):
        with np.load(checked(file,digest),allow_pickle=False) as data:
            return {k:immutable(data[k]) for k in data.files}
    selected=read(args.selection)
    if selected["contract"]!=TRAIN_CONTRACT or not selected["allHeadSelectionsFrozen"] \
            or selected["testLoaded"] or selected["testUsedForSelection"] or selected["expertBackbones"]!=12:
        raise ValueError("independent horizon choices not frozen")
    training=read(args.selection.parent/"training-result.json")
    if training["selectionSha256"]!=sha(args.selection) or training["testLoaded"]: raise ValueError("training incomplete")
    plan=read(args.selection.parent/"plan.json",selected["planSha256"])
    if plan!=selected["plan"]: raise ValueError("embedded training plan differs")
    dataset=Path(plan["dataset"])
    checked(dataset/"manifest.json",plan["datasetManifestSha256"])
    prior=read(selected["onlineSelectionFile"],selected["onlineSelectionSha256"])
    checked(selected["sourceSelectionFile"],selected["sourceSelectionSha256"])
    configs={str(r["horizonSeconds"]):{"sourcePreferredMode":r["validationPreferredMode"],
        **r["modes"][r["validationPreferredMode"]]} for r in prior["horizons"]}
    baseline=read(args.baseline)
    if baseline["testUsedForSelection"] or baseline["sourceTrainingPlanSha256"] is None \
            or baseline["onlineSelectionSha256"]!=selected["onlineSelectionSha256"]:
        raise ValueError("baseline calibration mismatch")
    for file,digest in baseline["artifactHashes"].items(): checked(file,digest)
    for entry in selected["horizons"]:
        for stage in ("pretrained","finetuned"):
            record=entry[stage]
            saved=verified(record)
            checked(record["checkpoint"],record["sha256"])
            if saved["contract"]!=TRAIN_CONTRACT or saved["horizonsSeconds"]!=[entry["horizonSeconds"]] \
                    or saved["planSha256"]!=selected["planSha256"] or saved["targetCoordinate"]!="log-return":
                raise ValueError("checkpoint identity mismatch")
    old_replay=read(args.baseline.parent/"replay-plan.json")
    replay={**old_replay,"contract":"oracle-per-horizon-log-comparison-v1","runsRoot":str(args.selection.parent),
        "selection":"independent per-horizon zero/sign initializers, fresh log-size pretraining selections and separate log NLL fine-tuning",
        "reportScript":"scripts/report-oracle-log-hurdle.mjs"}
    args.output.mkdir(parents=True,exist_ok=True)
    replay_file=args.output/"replay-plan.json"
    if replay_file.exists():
        if read(replay_file)!=replay: raise ValueError("replay plan changed")
    else: write_json(replay_file,replay)
    checked(replay_file)
    for file in (Path(__file__),Path('ml/train_per_horizon_log_hurdle.py'),Path('ml/log_hurdle_sign_size_forecast.py'),
            Path('ml/online_log_hurdle_calibration.py'),Path('ml/evaluate_log_hurdle_forecast.py'),Path('scripts/report-oracle-log-hurdle.mjs')):
        checked(file)
    evaluation_plan={"contract":CONTRACT,"createdAt":now(),"selectionFile":str(args.selection),"selectionSha256":sha(args.selection),
        "onlineSelectionFile":selected["onlineSelectionFile"],"onlineSelectionSha256":selected["onlineSelectionSha256"],
        "calibrationConfigurations":configs,"newHyperparameterSelection":False,"testUsedForSelection":False,
        "architecture":"four separately selected and trained singleton-horizon triplets",
        "baselineFile":str(args.baseline),"baselineSha256":sha(args.baseline),"batchSize":args.batch_size,
        "sourceTrainingPlanSha256":selected["planSha256"],"replayPlanSha256":sha(replay_file),
        "population":"exact retained common accuracy and canonical calibration origins; both return coordinates",
        "sourceHashesBeforeEvaluation":dict(tracked)}
    evaluation_file=args.output/"evaluation-plan.json"
    if evaluation_file.exists():
        previous=read(evaluation_file);evaluation_plan["createdAt"]=previous["createdAt"]
        if previous!=evaluation_plan: raise ValueError("evaluation plan changed")
    else: write_json(evaluation_file,evaluation_plan)
    checked(evaluation_file)
    result={**evaluation_plan,"evaluationPlanFile":str(evaluation_file),"evaluationPlanSha256":sha(evaluation_file),"results":[]}
    started=time.monotonic();exports=[]
    for split in ("validation","test"):
        timeline=load_timeline_split(dataset,split)
        common=select_origins(timeline,4,None)
        if len(common)!=(34560 if split=="validation" else 36134): raise ValueError("population count changed")
        for j,h in enumerate(HORIZONS):
            entry=selected["horizons"][j]
            if entry["horizonSeconds"]!=h: raise ValueError("wrong checkpoint horizon order")
            old=next(r for r in baseline["results"] if r["split"]==split and r["horizonSeconds"]==h)
            manifest=old["oldCanonicalCacheManifest"]
            cache=arrays(manifest["cacheFile"],manifest["cacheSha256"])
            market=arrays(manifest["marketFile"],manifest["marketSha256"])
            np.testing.assert_array_equal(cache["origins"][cache["accuracyIndices"]],common)
            offsets=common-int(market["firstPriceOrigin"])
            simple=market["prices"][offsets+h]/market["prices"][offsets]-1
            hashes={"origins":array_sha(common),"timestamps":array_sha(cache["timestamps"][cache["accuracyIndices"]]),
                "simpleTargets":array_sha(simple),"logTargets":array_sha(np.log1p(simple))}
            if hashes!=old["sampleHashes"]: raise ValueError("baseline target mismatch")
            row={"split":split,"horizonSeconds":h,"examples":len(common),"activeExamples":int((simple!=0).sum()),
                "sampleHashes":hashes,"oldCanonicalCacheManifest":manifest,"constants":entry["constants"],
                "models":{k:v for k,v in old["models"].items() if k.startswith("retained")}}
            row["models"].update({"sharedLogRaw":old["models"]["finetunedRaw"],"sharedLogCalibrated":old["models"]["finetunedCalibrated"]})
            for stage in ("pretrained","finetuned"):
                model=load_combined(entry[stage],device)
                raw=infer_horizon(model,timeline.timeline,cache["origins"],len(common),args.batch_size,device,h)
                del model;torch.cuda.empty_cache()
                stream=CachedStream(cache["origins"],cache["timestamps"],raw,market["prices"],int(market["firstPriceOrigin"]),
                    cache["accuracyIndices"],int(market["axisStartMs"]),h)
                raw_metrics=score_log_law(raw[stream.accuracy_indices],simple,entry["constants"])
                raw_metrics["upSmoothBound"]=raw_up_bound_diagnostics(raw[stream.accuracy_indices])
                if split=="validation":
                    np.testing.assert_allclose(raw_metrics["logReturnNegativeLogLikelihood"],entry[stage]["validationScore"],rtol=0,atol=1e-8)
                raw_metrics["savedStream"]=save_stream(args.output,split,stage+"-raw",stream)
                config=configs[str(h)]
                sim=simulate_log_stream(stream,CalibrationConfig(**config["config"]),trace=True)
                cal_metrics=score_log_law(sim["parameters"][stream.accuracy_indices],simple,entry["constants"])
                cal_metrics["savedStream"]=save_stream(args.output,split,stage+"-calibrated",stream,sim)
                cal_metrics["frozenCalibration"]=config
                row["models"][stage+"Raw"],row["models"][stage+"Calibrated"]=raw_metrics,cal_metrics
                for metrics in (raw_metrics,cal_metrics):
                    saved=metrics["savedStream"]
                    checked(saved["parametersFile"],saved["parametersSha256"])
                    if "traceFile" in saved: checked(saved["traceFile"],saved["traceSha256"])
                if split=="test" and h in replay["horizonsSeconds"]:
                    stage_view={**selected,stage:entry[stage]}
                    for mode,parameters,metrics,simulation in (("raw",raw,raw_metrics,None),("calibrated",sim["parameters"],cal_metrics,sim)):
                        manifest_out=export_stream(args.output,stream,parameters,stage+"-"+mode,args.selection,stage_view,stage,
                            metrics["savedStream"],replay,None if simulation is None else config,simulation)
                        exports.append(manifest_out)
                        checked(manifest_out["forecastFile"],manifest_out["forecastSha256"])
                        checked(Path(manifest_out["forecastFile"]).with_suffix(".manifest.json"))
                print(json.dumps({"event":"scored","split":split,"horizonSeconds":h,"stage":stage,
                    "rawLogSkill":raw_metrics["logReturns"]["allTickDeployedMseSkillVsZero"],
                    "calibratedLogSkill":cal_metrics["logReturns"]["allTickDeployedMseSkillVsZero"]}),flush=True)
            result["results"].append(row)
            write_json(args.output/f"{split}-progress.json",{"results":[r for r in result["results"] if r["split"]==split]})
    write_json(args.output/"exports.json",{"contract":CONTRACT,"manifests":exports})
    checked(args.output/"exports.json")
    for file,digest in tracked.items():
        if sha(Path(file))!=digest: raise ValueError(f"artifact changed during evaluation: {file}")
    result.update({"artifactHashes":tracked,"seconds":time.monotonic()-started})
    write_json(args.output/"evaluation.json",result)
    print(json.dumps({"event":"allComplete","seconds":result["seconds"],"evaluationSha256":sha(args.output/"evaluation.json")}),flush=True)


if __name__=="__main__": main()
