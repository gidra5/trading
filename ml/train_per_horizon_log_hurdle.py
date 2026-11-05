"""Corrected log-target experiment using the retained per-horizon architecture.

Reuse the unchanged-label zero/sign initializers, retrain log magnitudes with
independent horizon selections, and fine-tune four separately owned triplets.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from log_hurdle_sign_size_forecast import LogHurdleSignSize, log_hurdle_mean
from neural_sign_size_forecast import StructuredSpecialist
from stochastic_price_forecast import load_timeline_split
from train_log_hurdle_sign_size_forecast import (log_examples,log_constants,task_scores,
    batch_denominators,microbatch_loss)
from train_neural_sign_size_forecast import HORIZONS,make_optimizers,select_origins,sha,write_json,now
from train_ternary_sign_size_forecast import predict_output,singleton_state

CONTRACT = "per-horizon-log-hurdle-training-v1"


def payload(model,task,epoch,mean,std,scales,plan_sha,initialization):
    return {"contract":CONTRACT,"task":task,"horizonsSeconds":list(model.horizons),
        "targetCoordinate":"log-return","epoch":epoch,"inputMean":torch.from_numpy(mean),
        "inputStd":torch.from_numpy(std),"sizeScales":torch.from_numpy(scales),
        "modelState":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "planSha256":plan_sha,"initialization":initialization}


def verified(record):
    if sha(Path(record["checkpoint"])) != record["sha256"]:
        raise ValueError("frozen checkpoint changed")
    return torch.load(record["checkpoint"],map_location="cpu",weights_only=True)


def load_combined(record,device):
    saved=verified(record)
    if saved["contract"] != CONTRACT or saved["task"] != "combined" or saved["targetCoordinate"] != "log-return" \
            or len(saved["horizonsSeconds"]) != 1:
        raise ValueError("per-horizon log-law checkpoint required")
    h=tuple(saved["horizonsSeconds"])
    experts=[StructuredSpecialist(task,saved["inputMean"],saved["inputStd"],h) for task in ("sign","sign","size")]
    model=LogHurdleSignSize(*experts,saved["sizeScales"])
    model.load_state_dict(saved["modelState"])
    return model.to(device).eval()


def horizon_improvements(scores,best):
    """Each horizon selects against its own objective, never a horizon average."""
    scores=np.asarray(scores,dtype=np.float64)
    if scores.shape != (len(best),) or not np.isfinite(scores).all():
        raise ValueError("finite per-horizon validation scores required")
    return scores < np.asarray([b["score"] for b in best])


def fit(model,task,tx,ty,vx,vy,mean,std,scales,initialization,args):
    combined=task=="combined"
    maximum,minimum,patience=(64,12,12) if combined else (96,24,20)
    lr,half_life=(1e-5,2.) if combined else (1e-4,4.)
    horizons=model.horizons
    optimizers=make_optimizers(model)
    for optimizer in optimizers:
        for group in optimizer.param_groups: group["lr"]=lr
    ema=deepcopy(model).eval()
    steps=math.ceil(len(tx)/args.batch_size)
    decay=.5**(1/(half_life*steps))
    best=[{"score":math.inf,"epoch":-1} for _ in horizons]
    last_improvement,started=0,time.monotonic()
    plan_sha=sha(args.output/"plan.json")
    tensor_scales=torch.from_numpy(scales).to(tx.device)
    restored=[]
    if combined:
        for expert in (model.zero,model.sign,model.size):
            restored.append((expert,expert.forward))
            expert.forward=torch.compile(expert.forward,dynamic=True,options={"triton.cudagraphs":False})
        forward=model
    else: forward=torch.compile(model,dynamic=True,options={"triton.cudagraphs":False})

    def emit(value):
        line=json.dumps({"task":task,"horizonsSeconds":list(horizons),**value},allow_nan=False)
        print(line,flush=True)
        with (args.output/"training.jsonl").open("a",encoding="utf-8") as file: file.write(line+"\n")

    def evaluate_epoch(epoch,train_loss):
        output=predict_output(ema,vx,args.microbatch_size)
        values=task_scores(output,vy,task,scales)
        improved=horizon_improvements(values,best)
        for j,h in enumerate(horizons):
            if not improved[j]: continue
            file=args.output/f"{task}-h{h}.pt"
            saved=payload(ema,task,epoch,mean,std,scales,plan_sha,initialization)
            saved.update({"selectedHorizonSeconds":h,"validationScore":float(values[j]),
                "validationScoresByHorizon":values.tolist()})
            torch.save(saved,file)
            best[j]={"checkpoint":str(file),"sha256":sha(file),"epoch":epoch,"score":float(values[j]),
                "validationScoresByHorizon":values.tolist()}
        diagnostics={}
        if combined:
            mu=log_hurdle_mean(torch.from_numpy(output)).numpy()
            diagnostics={"validationLogMseSkill":(1-np.square(mu-vy).mean(0)/np.square(vy).mean(0)).tolist()}
        emit({"event":"epoch","epoch":epoch,"trainingLoss":train_loss,"validationScores":values.tolist(),
            "bestEpochs":[v["epoch"] for v in best],"seconds":time.monotonic()-started,**diagnostics})
        return bool(improved.any())

    emit({"event":"start","trainingExamples":len(tx),"validationExamples":len(vx),
        "parameters":sum(p.numel() for p in model.parameters()),"learningRate":lr,"emaDecayPerStep":decay})
    evaluate_epoch(0,None)
    generator=torch.Generator(device=tx.device).manual_seed(1337)
    completed=0
    for epoch in range(1,maximum+1):
        model.train()
        order=torch.randperm(len(tx),generator=generator,device=tx.device)
        weighted=0.
        for start in range(0,len(tx),args.batch_size):
            indices=order[start:start+args.batch_size]
            denominator=batch_denominators(ty[indices],task)
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            accumulated=torch.zeros((),dtype=torch.float64,device=tx.device)
            for offset in range(0,len(indices),args.microbatch_size):
                chosen=indices[offset:offset+args.microbatch_size]
                value=microbatch_loss(forward(tx[chosen]),ty[chosen],task,tensor_scales,denominator)
                value.backward()
                accumulated+=value.detach()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
            if not torch.isfinite(norm) or not torch.isfinite(accumulated): raise ValueError("nonfinite training")
            for optimizer in optimizers: optimizer.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters(),strict=True): ep.lerp_(mp,1-decay)
                for eb,mb in zip(ema.buffers(),model.buffers(),strict=True): eb.copy_(mb)
            weighted+=float(accumulated)*len(indices)
        completed=epoch
        if evaluate_epoch(epoch,weighted/len(tx)): last_improvement=epoch
        if epoch>=minimum and epoch-last_improvement>=patience: break
    emit({"event":"complete","completedEpochs":completed,"seconds":time.monotonic()-started})
    for expert,original in restored: expert.forward=original
    del forward,optimizers,ema
    torch.cuda.empty_cache()
    return [{"checkpoint":v["checkpoint"],"sha256":v["sha256"],"epoch":v["epoch"],
        "validationScore":v["score"],"validationScoresByHorizon":v["validationScoresByHorizon"],
        "horizonSeconds":h,"horizonsSeconds":list(horizons),"completedEpochs":completed,
        "validationMetric":"fullLogHurdleNLL" if combined else "balancedConditionalNormalizedLogSizeMSE",
        "constantFallback":False} for h,v in zip(horizons,best,strict=True)]


def make_triplet(source_entry,size_record,mean,std,scales,j):
    h=HORIZONS[j]
    records={"zero":source_entry["pretrained"]["zero"],"sign":source_entry["pretrained"]["sign"],"size":size_record}
    experts=[]
    for name,task,outputs in (("zero","sign",1),("sign","sign",1),("size","size",2)):
        saved=verified(records[name])
        if saved["horizonsSeconds"] != list(HORIZONS): raise ValueError("initializer horizon order changed")
        np.testing.assert_array_equal(saved["inputMean"].numpy(),mean)
        np.testing.assert_array_equal(saved["inputStd"].numpy(),std)
        if name=="size":
            if saved["contract"] != CONTRACT or saved["targetCoordinate"] != "log-return": raise ValueError("wrong size targets")
            np.testing.assert_array_equal(saved["sizeScales"].numpy(),scales)
        expert=StructuredSpecialist(task,torch.from_numpy(mean),torch.from_numpy(std),(h,))
        expert.load_state_dict(singleton_state(saved["modelState"],j,outputs))
        experts.append(expert)
    return LogHurdleSignSize(*experts,torch.from_numpy(scales[j:j+1])),records


@torch.no_grad()
def initialize_shapes(model,x,y,batch):
    model.eval()
    total=np.zeros(2); counts=np.zeros(2,dtype=np.int64)
    for start in range(0,len(x),batch):
        means=(model.size(x[start:start+batch])*model.scales).cpu().numpy()[:,0]
        values=y[start:start+batch,0]
        for k,mask in enumerate((values>0,values<0)):
            total[k]+=np.square(np.abs(values[mask])/np.maximum(means[mask,k],1e-12)-1).sum()
            counts[k]+=mask.sum()
    if (counts==0).any() or not np.isfinite(total).all(): raise ValueError("invalid shape residuals")
    relative=total/counts
    shapes=np.clip(np.divide(1.,relative,out=np.full_like(relative,50.),where=relative>0),.01,50.)[None]
    model.set_shapes_(torch.from_numpy(shapes).to(x.device))
    return {"shapes":shapes.tolist(),"relativeResidualSecondMoments":relative.tolist(),
        "examplesByDirection":counts.tolist(),"testUsed":False,"method":"inverse train squared relative log-size residual"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("data/training/runs/stochastic-crra-per-horizon-log-v1"))
    parser.add_argument("--source-selection",type=Path,default=Path("data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json"))
    parser.add_argument("--online-selection",type=Path,default=Path("data/training/runs/stochastic-crra-online-calibration-v1/selection.json"))
    parser.add_argument("--batch-size",type=int,default=10240)
    parser.add_argument("--microbatch-size",type=int,default=2560)
    args=parser.parse_args()
    if (args.output/"plan.json").exists(): raise ValueError("refusing to overwrite experiment")
    source=json.loads(args.source_selection.read_text())
    metadata=json.loads((args.source_selection.parent/"metadata.json").read_text())
    online=json.loads(args.online_selection.read_text())
    if source["testLoaded"] or source["testUsedForSelection"] or not source["allHeadSelectionsFrozen"] \
            or online["sourceSelectionSha256"]!=sha(args.source_selection): raise ValueError("source selection provenance")
    for row in source["horizons"]:
        for task in ("zero","sign"): verified(row["pretrained"][task])
    dataset=Path(source["plan"]["dataset"])
    if sha(dataset/"manifest.json")!=source["plan"]["datasetManifestSha256"]: raise ValueError("dataset changed")
    plan={"contract":CONTRACT,"createdAt":now(),"dataset":str(dataset),
        "datasetManifestSha256":sha(dataset/"manifest.json"),"horizonsSeconds":list(HORIZONS),
        "sourceSelectionFile":str(args.source_selection),"sourceSelectionSha256":sha(args.source_selection),
        "onlineSelectionFile":str(args.online_selection),"onlineSelectionSha256":sha(args.online_selection),
        "strideSeconds":4,"trainingLimit":256000,"features":source["plan"]["features"],
        "target":"X=log1p(exact endpoint simple return), float64, exact zero preserved",
        "initialization":"reuse each retained horizon's original zero/sign pretrained checkpoint; unchanged labels/objectives. Retrain size from seed1337 on log targets with independent horizon selection",
        "sizePretraining":{"maximumEpochs":96,"minimumEpochs":24,"patience":20,"learningRate":1e-4,"emaHalfLifeEpochs":4.,
            "loss":"conditional normalized LOG-size MSE, equal directions/horizons","selection":"each horizon validation minimum; patience since ANY horizon improved"},
        "fineTuning":{"maximumEpochs":64,"minimumEpochs":12,"patience":12,"learningRate":1e-5,"emaHalfLifeEpochs":2.,
            "loss":"full signed-Gamma LOG-return NLL; all rows","selection":"independent validation minimum per horizon including epoch0","allWeightsUnfrozen":True},
        "architecture":"four independently owned singleton-horizon zero/sign/size triplets; same retained architecture and per-horizon checkpoint procedure",
        "optimizer":source["plan"]["optimizer"],"effectiveBatchSize":args.batch_size,"microbatchSize":args.microbatch_size,
        "magnitudeLaw":"exact zero and signed-Gamma log magnitudes; UPscale<=.49 via alpha*.49*tanh(rawMean/(alpha*.49)); same log family as preceding run",
        "calibration":"prior validation-selected per-horizon configs unchanged, log likelihood updates only after label maturity",
        "testProtocol":"trainer never loads test; freeze all four independent checkpoint choices before evaluator",
        "testUsedForSelection":False}
    write_json(args.output/"plan.json",plan)
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    cache=Path("data/training/cache/torchinductor").resolve()
    os.environ["TORCHINDUCTOR_CACHE_DIR"],os.environ["TRITON_CACHE_DIR"]=str(cache),str(cache/"triton")
    device,started=torch.device("cuda"),time.monotonic()
    ts,vs=load_timeline_split(dataset,"train"),load_timeline_split(dataset,"validation")
    train=log_examples(ts,select_origins(ts,4,256000)); validation=log_examples(vs,select_origins(vs,4,None))
    if len(train.targets)!=metadata["trainingExamples"] or len(validation.targets)!=metadata["validationExamples"]: raise ValueError("sample mismatch")
    if train.axis_start_ms+(train.origins[-1]+7200)*1000>=validation.axis_start_ms+validation.origins[0]*1000: raise ValueError("split overlap")
    flat=train.features.reshape(-1,59)
    mean=flat.mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(flat.std(0,dtype=np.float64),1e-10).astype(np.float32)
    np.testing.assert_array_equal(mean,np.asarray(metadata["inputMean"],dtype=np.float32))
    np.testing.assert_array_equal(std,np.asarray(metadata["inputStd"],dtype=np.float32))
    constants=log_constants(train.targets)
    scales=np.array([[c["upMeanLog"],c["downMeanLog"]] for c in constants])
    write_json(args.output/"metadata.json",{"trainingExamples":len(train.targets),"validationExamples":len(validation.targets),
        "constants":constants,"inputMean":mean.tolist(),"inputStd":std.tolist(),"testLoaded":False})
    tx,ty=torch.from_numpy(train.features).to(device),torch.from_numpy(train.targets).to(device)
    vx=torch.from_numpy(validation.features).to(device)
    torch.manual_seed(1337)
    size=StructuredSpecialist("size",torch.from_numpy(mean),torch.from_numpy(std),HORIZONS).to(device)
    size_records=fit(size,"size",tx,ty,vx,validation.targets,mean,std,scales,{"freshSeed":1337},args)
    write_json(args.output/"size-selection.json",{"horizons":size_records,"testLoaded":False})
    del size; torch.cuda.empty_cache()
    entries=[]
    for j,h in enumerate(HORIZONS):
        model,records=make_triplet(source["horizons"][j],size_records[j],mean,std,scales,j)
        model=model.to(device)
        shape_init=initialize_shapes(model,tx,train.targets[:,j:j+1],args.microbatch_size)
        initialization={"experts":records,"shapeInitialization":shape_init,"sourceHorizonIndex":j,"allWeightsUnfrozen":True}
        values=task_scores(predict_output(model,vx,args.microbatch_size),validation.targets[:,j:j+1],"combined",scales[j:j+1])
        file=args.output/f"pretrained-h{h}.pt"
        saved=payload(model,"combined",0,mean,std,scales[j:j+1],sha(args.output/"plan.json"),initialization)
        saved.update({"validationScore":float(values[0]),"validationScoresByHorizon":values.tolist()})
        torch.save(saved,file)
        base={"checkpoint":str(file),"sha256":sha(file),"epoch":0,"horizonSeconds":h,"horizonsSeconds":[h],
            "validationScore":float(values[0]),"validationScoresByHorizon":values.tolist()}
        fine=fit(model,"combined",tx,ty[:,j:j+1],vx,validation.targets[:,j:j+1],mean,std,scales[j:j+1],initialization,args)[0]
        entries.append({"horizonSeconds":h,"constants":constants[j],"initializers":records,"pretrained":base,"finetuned":fine,
            "parameters":sum(p.numel() for p in model.parameters())})
        del model; torch.cuda.empty_cache()
        write_json(args.output/"progress.json",{"completedHorizons":entries,"testLoaded":False})
    selection={"contract":CONTRACT,"createdAt":now(),"plan":plan,"planSha256":sha(args.output/"plan.json"),
        "sourceSelectionFile":str(args.source_selection),"sourceSelectionSha256":sha(args.source_selection),
        "onlineSelectionFile":str(args.online_selection),"onlineSelectionSha256":sha(args.online_selection),
        "horizons":entries,"constants":constants,"expertBackbones":12,"parameters":sum(r["parameters"] for r in entries),
        "allHeadSelectionsFrozen":True,"testLoaded":False,"testUsedForSelection":False}
    write_json(args.output/"selection.json",selection)
    result={"selectionSha256":sha(args.output/"selection.json"),"seconds":time.monotonic()-started,
        "testLoaded":False,"testUsedForSelection":False,"selectedEpochs":[r["finetuned"]["epoch"] for r in entries]}
    write_json(args.output/"training-result.json",result)
    print(json.dumps({"event":"allComplete",**result}),flush=True)


if __name__=="__main__": main()
