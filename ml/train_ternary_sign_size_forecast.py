"""Train ternary sign, then fine-tune each sign/size pair for signed mean MSE.

Old size checkpoints are frozen initializers. All new selections precede test
loading; post-fine-tuning component scores are diagnostics, not calibration claims.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from neural_sign_size_forecast import StructuredSpecialist, architecture_config
from sign_size_forecast import magnitude_metrics, sign_metrics
from stochastic_price_forecast import load_timeline_split
from ternary_sign_size_forecast import CONTRACT, CombinedSignSize, TernarySignSpecialist
from train_neural_sign_size_forecast import (HORIZONS, Examples, constants_from_training, examples,
    make_optimizers, now, select_origins, sha, write_json)


def target_classes(y: torch.Tensor | np.ndarray):
    """Class order DOWN=0, ZERO=1, UP=2, preserving exact zero endpoints."""
    if isinstance(y, torch.Tensor):
        return (y > 0).long() + (y >= 0).long()
    return (y > 0).astype(np.int64) + (y >= 0).astype(np.int64)


def loss_elements(output: torch.Tensor, y: torch.Tensor, task: str,
                  second_moments: torch.Tensor) -> torch.Tensor:
    if task == "ternary":
        return F.cross_entropy(output.flatten(0,1), target_classes(y).flatten(), reduction="none").reshape_as(y)
    return (output-y).square()/second_moments


def scores(output: np.ndarray, y: np.ndarray, task: str, second_moments: np.ndarray) -> np.ndarray:
    if task == "ternary":
        maximum = output.max(axis=-1, keepdims=True)
        logsum = maximum[...,0] + np.log(np.exp(output-maximum).sum(axis=-1))
        actual = np.take_along_axis(output,target_classes(y)[...,None],axis=-1)[...,0]
        return (logsum-actual).mean(axis=0)
    return np.square(output-y).mean(axis=0)/second_moments


@torch.no_grad()
def predict_output(model: torch.nn.Module, x: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    result = np.concatenate([model(x[start:start+batch]).cpu().numpy().astype(np.float64)
                             for start in range(0,len(x),batch)])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite validation model output")
    return result


def checkpoint_payload(model: torch.nn.Module, task: str, horizons: tuple[int,...], epoch: int,
                       mean: np.ndarray, std: np.ndarray, scales: np.ndarray,
                       plan_sha: str, initialization: dict) -> dict:
    return {"contract": CONTRACT, "task": task, "horizonsSeconds": list(horizons),
        "modelState": {k:v.detach().cpu() for k,v in model.state_dict().items()},
        "inputMean": torch.from_numpy(mean), "inputStd": torch.from_numpy(std),
        "sizeScales": torch.from_numpy(scales), "epoch": epoch, "planSha256": plan_sha,
        "initialization": initialization}


def fit(model: torch.nn.Module, task: str, horizons: tuple[int,...], tx: torch.Tensor, ty: torch.Tensor,
        vx: torch.Tensor, vy: np.ndarray, second: np.ndarray, mean: np.ndarray, std: np.ndarray,
        scales: np.ndarray, initialization: dict, args, *, maximum: int, minimum: int, patience: int,
        learning_rate: float, half_life: float, prefix: str) -> list[dict]:
    device = tx.device
    optimizers = make_optimizers(model)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    ema = deepcopy(model).eval()
    steps = math.ceil(len(tx)/args.batch_size)
    decay = .5**(1/(half_life*steps))
    denominator = torch.tensor(second,dtype=torch.float64,device=device)
    best = [{"score": math.inf, "epoch": -1} for _ in horizons]
    last_improvement = 0
    started = time.monotonic()
    plan_sha = sha(args.output/"plan.json")
    forward = model if args.no_compile else torch.compile(model,dynamic=True,options={"triton.cudagraphs":False})
    log = args.output/"training.jsonl"

    def emit(value: dict):
        value = {"stage":prefix,"horizonsSeconds":list(horizons),**value}
        line = json.dumps(value,allow_nan=False)
        print(line,flush=True)
        with log.open("a",encoding="utf-8") as file:
            file.write(line+"\n")

    def evaluate_epoch(epoch: int, train_loss: float | None) -> bool:
        prediction = predict_output(ema,vx,args.microbatch_size)
        validation = scores(prediction,vy,task,second)
        improved = False
        for j,h in enumerate(horizons):
            if validation[j] < best[j]["score"]:
                improved = True
                file = args.output/f"{prefix}-h{h}.pt"
                saved = checkpoint_payload(ema,task,horizons,epoch,mean,std,scales,plan_sha,initialization)
                saved.update({"selectedHorizonSeconds":h,"validationScore":float(validation[j])})
                torch.save(saved,file)
                best[j] = {"checkpoint":str(file),"sha256":sha(file),"epoch":epoch,"score":float(validation[j])}
        emit({"event":"epoch","epoch":epoch,"trainingLoss":train_loss,"validationScores":validation.tolist(),
              "bestEpochs":[b["epoch"] for b in best],"seconds":time.monotonic()-started,
              "maximumGpuMiB":torch.cuda.max_memory_allocated()/2**20})
        return improved

    emit({"event":"start","trainingExamples":len(tx),"validationExamples":len(vx),
          "parameters":sum(p.numel() for p in model.parameters()),"effectiveBatchSize":args.batch_size,
          "microbatchSize":args.microbatch_size,"learningRate":learning_rate,"emaDecayPerStep":decay})
    evaluate_epoch(0,None)
    generator = torch.Generator(device=device).manual_seed(1337)
    completed = 0
    for epoch in range(1,maximum+1):
        model.train()
        order = torch.randperm(len(tx),generator=generator,device=device)
        total_loss = 0.
        for start in range(0,len(tx),args.batch_size):
            indices = order[start:start+args.batch_size]
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros((),device=device)
            for offset in range(0,len(indices),args.microbatch_size):
                chosen = indices[offset:offset+args.microbatch_size]
                value = loss_elements(forward(tx[chosen]),ty[chosen],task,denominator)
                loss = value.sum()/(len(indices)*ty.shape[1])
                loss.backward()
                accumulated += loss.detach()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
            if not torch.isfinite(gradient) or not torch.isfinite(accumulated):
                raise ValueError("nonfinite optimizer step")
            for optimizer in optimizers:
                optimizer.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters(),strict=True):
                    ep.lerp_(mp,1-decay)
                for eb,mb in zip(ema.buffers(),model.buffers(),strict=True):
                    eb.copy_(mb)
            total_loss += float(accumulated)
        completed = epoch
        if evaluate_epoch(epoch,total_loss/steps):
            last_improvement = epoch
        if epoch >= minimum and epoch-last_improvement >= patience:
            break
    emit({"event":"complete","completedEpochs":completed,"seconds":time.monotonic()-started})
    results = []
    for b in best:
        results.append({"checkpoint":b["checkpoint"],"sha256":b["sha256"],"epoch":b["epoch"],
            "validationScore":b["score"],"validationMetric":"threeClassLogLoss" if task=="ternary" else "allTickSignedReturnMseNormalizedByTrainSecondMoment",
            "completedEpochs":completed,"checkpointSource":("train-frequency-initialized-sign-prior" if task=="ternary" else "pre-finetune-combined-pair")
                if b["epoch"]==0 else "trained-EMA","constantFallback":False})
    del forward,ema,optimizers
    torch.cuda.empty_cache()
    return results


def singleton_state(state: dict, horizon_index: int, outputs_per_horizon: int) -> dict:
    lo = horizon_index*outputs_per_horizon
    return {**state,"readout.weight":state["readout.weight"][lo:lo+outputs_per_horizon],
            "readout.bias":state["readout.bias"][lo:lo+outputs_per_horizon]}


def load_verified(record: dict) -> dict:
    file = Path(record["checkpoint"])
    if sha(file) != record["sha256"]:
        raise ValueError("checkpoint provenance SHA mismatch")
    return torch.load(file,map_location="cpu",weights_only=True)


def make_pair(sign_record: dict, size_record: dict, j: int, mean: np.ndarray, std: np.ndarray,
              scales: np.ndarray) -> CombinedSignSize:
    sign_saved,size_saved = load_verified(sign_record),load_verified(size_record)
    h = HORIZONS[j]
    if sign_saved["horizonsSeconds"] != list(HORIZONS) or size_saved["horizonsSeconds"] != list(HORIZONS):
        raise ValueError("pretrained horizon alignment differs")
    for saved in (sign_saved,size_saved):
        np.testing.assert_array_equal(saved["inputMean"].numpy(),mean)
        np.testing.assert_array_equal(saved["inputStd"].numpy(),std)
    sign = TernarySignSpecialist(torch.from_numpy(mean),torch.from_numpy(std),horizons=(h,))
    size = StructuredSpecialist("size",torch.from_numpy(mean),torch.from_numpy(std),horizons=(h,))
    sign.load_state_dict(singleton_state(sign_saved["modelState"],j,3))
    size.load_state_dict(singleton_state(size_saved["modelState"],j,2))
    np.testing.assert_array_equal(size_saved["sizeScales"].numpy()[j:j+1],scales)
    return CombinedSignSize(sign,size,torch.from_numpy(scales))


def load_combined(record: dict, device: torch.device) -> CombinedSignSize:
    saved = load_verified(record)
    if saved["contract"] != CONTRACT or saved["task"] != "combined":
        raise ValueError("unsupported combined checkpoint")
    sign = TernarySignSpecialist(saved["inputMean"],saved["inputStd"],horizons=tuple(saved["horizonsSeconds"]))
    size = StructuredSpecialist("size",saved["inputMean"],saved["inputStd"],horizons=tuple(saved["horizonsSeconds"]))
    model = CombinedSignSize(sign,size,saved["sizeScales"])
    model.load_state_dict(saved["modelState"])
    return model.to(device).eval()


def metrics(y: np.ndarray, probability: np.ndarray, up: np.ndarray, down: np.ndarray,
            model_mean: np.ndarray, c: dict) -> dict:
    if not all(np.isfinite(a).all() for a in (y,probability,up,down,model_mean)):
        raise ValueError("nonfinite combined prediction")
    active,positive,negative = y!=0,y>0,y<0
    rows = np.arange(len(y))
    probabilities = probability/probability.sum(axis=1,keepdims=True)
    p0,pup,pdown = probabilities[:,1],probabilities[:,2],probabilities[:,0]
    pactive = pup+pdown
    conditional_up = np.divide(pup,pactive,out=np.full_like(pup,.5),where=pactive>0)
    scorep = np.clip(conditional_up,1e-12,1-1e-12)
    mean = pup*up-pdown*down
    absolute = pup*up+pdown*down
    signed_mse = float(np.square(mean-y).mean())
    size_mse = float(np.square(absolute-np.abs(y)).mean())
    signed_baseline = float(np.square(y).mean())
    size_baseline = float(np.square(np.abs(y)-(1-c["pZero"])*c["pooledActiveMeanSimple"]).mean())
    return {"examples":len(y),"activeExamples":int(active.sum()),
        "threeClassLogLoss":float(-np.log(np.clip(probabilities[rows,target_classes(y)],1e-12,1)).mean()),
        "threeClassBrier":float(np.square(probabilities-np.eye(3)[target_classes(y)]).sum(axis=1).mean()),
        "zeroBrier":float(np.square(p0-(~active)).mean()),"predictedZeroProbability":float(p0.mean()),
        "realizedZeroFrequency":float((~active).mean()),
        "signGivenActive":sign_metrics(positive[active],scorep[active]),
        "magnitudeWeightedSignAccuracy":float(np.dot(np.abs(y[active]),(conditional_up[active]>=.5)==positive[active])/np.abs(y[active]).sum()),
        "positiveSize":magnitude_metrics(y[positive]*1e4,up[positive]*1e4,c["upMeanSimple"]*1e4),
        "negativeSize":magnitude_metrics(-y[negative]*1e4,down[negative]*1e4,c["downMeanSimple"]*1e4),
        "allTickSignedMeanMseBpsSquared":signed_mse*1e8,"allTickSignedMeanMseSkillVsZero":1-signed_mse/signed_baseline,
        "allTickAbsoluteSizeMseBpsSquared":size_mse*1e8,"allTickAbsoluteSizeMseSkillVsTrainConstant":1-size_mse/size_baseline,
        "predictedMeanReturnBps":float(mean.mean()*1e4),"predictedAverageAbsoluteSizeBps":float(absolute.mean()*1e4),
        "maximumPredictedMeanBps":float(mean.max()*1e4),"minimumPredictedMeanBps":float(mean.min()*1e4),
        "feeThresholdCrossings":int((mean>1/.9995**2-1).sum()),
        "probabilityNormalizationMaximumError":float(np.max(np.abs(probability.sum(axis=1)-1))),
        "normalizedLawVsForwardMeanMaximumError":float(np.max(np.abs(mean-model_mean))),
        "undefinedActiveConditionalSignCount":int((pactive==0).sum()),"downsideAtPhysicalBoundaryCount":int((down>=1-np.finfo(np.float32).eps).sum()),
        "componentInterpretation":"post-finetune probabilities and conditional sizes are representations; pure mean MSE does not identify their true probabilistic values"}


@torch.no_grad()
def evaluate(entries: list[dict], data: Examples, batch: int, device: torch.device) -> list[dict]:
    x = torch.from_numpy(data.features).to(device)
    result = []
    for j,entry in enumerate(entries):
        row = {"horizonSeconds":entry["horizonSeconds"],"models":{}}
        for name,record in (("separatePretrained",entry["pretrained"]["combined"]),("jointlyFinetuned",entry["finetuned"])):
            model = load_combined(record,device)
            parts,means = [],[]
            for start in range(0,len(x),batch):
                values = model.components(x[start:start+batch])
                parts.append(tuple(v[:,0].cpu().numpy().astype(np.float64) for v in values))
                means.append(model(x[start:start+batch])[:,0].cpu().numpy().astype(np.float64))
            probability,up,down = (np.concatenate([part[k] for part in parts]) for k in range(3))
            row["models"][name] = metrics(data.targets[:,j],probability,up,down,np.concatenate(means),entry["constants"])
            del model
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-selection",type=Path,default=Path("data/training/runs/stochastic-crra-neural-specialists-v1/selection.json"))
    parser.add_argument("--output",type=Path,default=Path("data/training/runs/stochastic-crra-ternary-finetune-v1"))
    parser.add_argument("--sign-epochs",type=int,default=96)
    parser.add_argument("--sign-min-epochs",type=int,default=24)
    parser.add_argument("--sign-patience",type=int,default=20)
    parser.add_argument("--fine-epochs",type=int,default=64)
    parser.add_argument("--fine-min-epochs",type=int,default=12)
    parser.add_argument("--fine-patience",type=int,default=12)
    parser.add_argument("--batch-size",type=int,default=10240)
    parser.add_argument("--microbatch-size",type=int,default=2560)
    parser.add_argument("--debug-train-limit",type=int)
    parser.add_argument("--debug-validation-limit",type=int)
    parser.add_argument("--no-compile",action="store_true")
    parser.add_argument("--skip-test",action="store_true")
    args = parser.parse_args()
    if (args.output/"plan.json").exists():
        raise ValueError("refusing to overwrite existing experiment plan")
    if any(v<=0 for v in (args.sign_epochs,args.sign_min_epochs,args.sign_patience,args.fine_epochs,args.fine_min_epochs,args.fine_patience,args.batch_size,args.microbatch_size)):
        raise ValueError("positive training limits required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1337)
    if not args.no_compile:
        cache = Path("data/training/cache/torchinductor").resolve()
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
        os.environ["TRITON_CACHE_DIR"] = str(cache/"triton")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    source = json.loads(args.source_selection.read_text())
    source_plan = source["plan"]
    source_meta = json.loads((args.source_selection.parent/"metadata.json").read_text())
    if not source["allHeadSelectionsFrozen"] or source["testUsedForSelection"]:
        raise ValueError("source size selections must be frozen without test")
    dataset = Path(source_plan["dataset"])
    if sha(dataset/"manifest.json") != source_plan["datasetManifestSha256"]:
        raise ValueError("dataset manifest changed since original specialists")
    for entry in source["horizons"]:
        load_verified(entry["size"])
    plan = {"contract":CONTRACT,"createdAt":now(),"dataset":str(dataset),"datasetManifestSha256":sha(dataset/"manifest.json"),
        "horizonsSeconds":list(HORIZONS),"sourceSelection":str(args.source_selection),"sourceSelectionSha256":sha(args.source_selection),
        "features":source_plan["features"],"strideSeconds":source_plan["strideSeconds"],"trainingLimit":source_plan["trainingLimit"],
        "initialization":"fresh ternary base17 sign with zero readout weights and log half-count-smoothed TRAIN class priors; reuse independently trained size checkpoint per horizon",
        "classOrder":["DOWN","ZERO","UP"],"sizeOrder":["UP","DOWN"],
        "signPretraining":{"maximumEpochs":args.sign_epochs,"minimumEpochs":args.sign_min_epochs,"patience":args.sign_patience,
            "loss":"unweighted three-class cross entropy over all contexts and horizons","selection":"per-horizon validation three-class logloss",
            "learningRate":1e-4,"emaHalfLifeEpochs":4.,"architecture":architecture_config("sign")},
        "fineTuning":{"maximumEpochs":args.fine_epochs,"minimumEpochs":args.fine_min_epochs,"patience":args.fine_patience,
            "loss":"ONLY squared error of pUP*mUP - pDOWN*mDOWN versus actual SIMPLE return, normalized by train E[R^2]",
            "initialization":"independent single-horizon pairs from that horizon's selected sign and size backbones/readout rows; preserve exact epoch0 combined checkpoint",
            "allWeightsUnfrozen":True,"auxiliaryLoss":None,"selection":"per-horizon validation signed-return MSE including epoch0",
            "learningRate":1e-5,"emaHalfLifeEpochs":2.},
        "optimizer":{"type":"Muon/AdamW","muonMomentum":.95,"muonNesterov":True,"muonNewtonSchulzSteps":3,
            "muonAdjustLearningRate":"match_rms_adamw","muonEpsilon":1e-7,"adamBetas":[.9,.999],"adamEpsilon":1e-8,
            "weightDecay":0.,"gradientClip":100.,"seed":1337},
        "effectiveBatchSize":args.batch_size,"microbatchSize":args.microbatch_size,"gradientAccumulation":"exact full effective-batch mean",
        "compile":not args.no_compile,"compiledCudaGraphs":False,"precision":"float32 networks/high matmul precision; float64 physical size scales, SIMPLE-return targets, composed mean and fine-tuning loss",
        "physicalDownsideCap":"min(1-1e-12,1-finfo(dtype).eps), identical in combined training and exported components",
        "testProtocol":"ALL ternary sign and per-horizon fine selections frozen before any test load; reused benchmark, no test-based choice",
        "componentSemantics":"proper CE and conditional MSE pretraining; pure signed mean MSE fine-tuning does not identify true probabilities or conditional sizes",
        "constantFallback":False,"testUsedForSelection":False,"skipTest":args.skip_test,
        "debugLimits":{"train":args.debug_train_limit,"validation":args.debug_validation_limit},
        "replay":{"horizonsSeconds":[900,1800,7200],"commonSeconds":151200,"variants":["separatePretrained","jointlyFinetuned"]}}
    write_json(args.output/"plan.json",plan)
    started = time.monotonic()
    train_source = load_timeline_split(dataset,"train")
    validation_source = load_timeline_split(dataset,"validation")
    train = examples(train_source,select_origins(train_source,source_plan["strideSeconds"],source_plan["trainingLimit"]))
    validation = examples(validation_source,select_origins(validation_source,source_plan["strideSeconds"],source_plan["validationLimit"]))
    if len(train.targets)!=source_meta["trainingExamples"] or len(validation.targets)!=source_meta["validationExamples"]:
        raise ValueError("training/validation origin counts changed")
    constants = constants_from_training(train.targets)
    for fresh,old in zip(constants,source["horizons"],strict=True):
        if fresh != old["constants"]:
            raise ValueError("training conditional size scales or constants changed")
    mean,std = np.asarray(source_meta["inputMean"],dtype=np.float32),np.asarray(source_meta["inputStd"],dtype=np.float32)
    np.testing.assert_array_equal(train.features.reshape(-1,59).mean(axis=0,dtype=np.float64).astype(np.float32),mean)
    np.testing.assert_array_equal(np.maximum(train.features.reshape(-1,59).std(axis=0,dtype=np.float64),1e-10).astype(np.float32),std)
    counts = np.stack([(target_classes(train.targets)==k).sum(axis=0) for k in range(3)],axis=-1)
    priors = (counts+.5)/(len(train.targets)+1.5)
    second = np.square(train.targets).mean(axis=0)
    scales = np.array([[c["upMeanSimple"],c["downMeanSimple"]] for c in constants])
    full_counts = (len(train.targets),len(validation.targets))
    def debug_subset(data: Examples, limit: int | None) -> Examples:
        if limit is None or limit>=len(data.targets):
            return data
        ix = np.linspace(0,len(data.targets)-1,limit,dtype=np.int64)
        return Examples(data.features[ix],data.targets[ix],data.origins[ix],data.axis_start_ms)
    train,validation = debug_subset(train,args.debug_train_limit),debug_subset(validation,args.debug_validation_limit)
    metadata = {"contract":CONTRACT,"planSha256":sha(args.output/"plan.json"),"inputMean":mean.tolist(),"inputStd":std.tolist(),
        "constants":constants,"classCounts":counts.tolist(),"classPriors":priors.tolist(),"trainReturnSecondMoments":second.tolist(),
        "trainingExamples":len(train.targets),"validationExamples":len(validation.targets),"originalExamples":list(full_counts),
        "testLoaded":False,"testUsedForSelection":False}
    write_json(args.output/"metadata.json",metadata)
    tx,ty = torch.from_numpy(train.features).to(device),torch.from_numpy(train.targets).to(device)
    vx = torch.from_numpy(validation.features).to(device)
    sign = TernarySignSpecialist(torch.from_numpy(mean),torch.from_numpy(std),horizons=HORIZONS).to(device)
    with torch.no_grad():
        sign.readout.weight.zero_()
        sign.readout.bias.copy_(torch.from_numpy(np.log(priors).reshape(-1)).to(device).float())
    initialization = {"kind":"train-class-prior","classPriors":priors.tolist(),"zeroReadoutWeight":True}
    sign_records = fit(sign,"ternary",HORIZONS,tx,ty,vx,validation.targets,second,mean,std,scales,initialization,args,
        maximum=args.sign_epochs,minimum=args.sign_min_epochs,patience=args.sign_patience,
        learning_rate=1e-4,half_life=4.,prefix="ternary-sign")
    del sign
    write_json(args.output/"sign-selection.json",{"contract":CONTRACT,"signChoicesFrozen":True,"horizons":sign_records,"testLoaded":False})
    entries = []
    for j,h in enumerate(HORIZONS):
        torch.manual_seed(1337)
        size_record = source["horizons"][j]["size"]
        pair = make_pair(sign_records[j],size_record,j,mean,std,scales[j:j+1]).to(device)
        initialization = {"kind":"independently-selected-pretrained-singleton-pair","sign":sign_records[j],"size":size_record,
            "sourceHorizonIndex":j,"allWeightsUnfrozen":True}
        baseline = args.output/f"pretrained-combined-h{h}.pt"
        saved = checkpoint_payload(pair,"combined",(h,),0,mean,std,scales[j:j+1],sha(args.output/"plan.json"),initialization)
        base_score = float(scores(predict_output(pair,vx,args.microbatch_size),validation.targets[:,j:j+1],"combined",second[j:j+1])[0])
        saved.update({"selectedHorizonSeconds":h,"validationScore":base_score})
        torch.save(saved,baseline)
        base_record = {"checkpoint":str(baseline),"sha256":sha(baseline),"epoch":0,"validationScore":base_score,
            "validationMetric":"allTickSignedReturnMseNormalizedByTrainSecondMoment","checkpointSource":"separate-pretrained-combined-pair"}
        fine = fit(pair,"combined",(h,),tx,ty[:,j:j+1],vx,validation.targets[:,j:j+1],second[j:j+1],mean,std,scales[j:j+1],initialization,args,
            maximum=args.fine_epochs,minimum=args.fine_min_epochs,patience=args.fine_patience,
            learning_rate=1e-5,half_life=2.,prefix="finetuned")[0]
        entries.append({"horizonSeconds":h,"constants":constants[j],"pretrained":{"sign":sign_records[j],"size":size_record,"combined":base_record},"finetuned":fine})
        del pair
        torch.cuda.empty_cache()
    frozen = {"contract":CONTRACT,"createdAt":now(),"plan":plan,"planSha256":sha(args.output/"plan.json"),
        "horizons":entries,"allHeadSelectionsFrozen":True,"testLoaded":False,"testUsedForSelection":False}
    write_json(args.output/"selection.json",frozen)
    write_json(args.output/"checkpoint-selection.json",frozen)
    del tx,ty,vx
    torch.cuda.empty_cache()
    validation_results = evaluate(entries,validation,args.microbatch_size,device)
    write_json(args.output/"validation.json",{"contract":CONTRACT,"horizons":validation_results})
    result = {"contract":CONTRACT,"selectionFile":str(args.output/"selection.json"),"selectionSha256":sha(args.output/"selection.json"),
        "validation":validation_results,"testUsedForSelection":False,"beforeTestSeconds":time.monotonic()-started}
    if not args.skip_test:
        test_source = load_timeline_split(dataset,"test")
        test = examples(test_source,select_origins(test_source,source_plan["strideSeconds"],None))
        result["test"] = evaluate(entries,test,args.microbatch_size,device)
        result["testExamples"] = len(test.targets)
        result["testEvaluationStrideSeconds"] = source_plan["strideSeconds"]
        write_json(args.output/"test.json",{"contract":CONTRACT,"horizons":result["test"],"testUsedForSelection":False})
    result["seconds"] = time.monotonic()-started
    write_json(args.output/"result.json",result)
    print(json.dumps({"event":"allComplete","result":str(args.output/"result.json"),"seconds":result["seconds"]}),flush=True)


if __name__ == "__main__":
    main()
