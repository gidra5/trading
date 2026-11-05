"""Fresh shared-horizon zero/sign/log-size experts, then proper log-law NLL.

Training and all selections use only train/validation. Historical simple-law
checkpoints stay immutable comparison artifacts; they are never initializers.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

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
from stochastic_price_forecast import load_timeline_split
from train_neural_sign_size_forecast import (HORIZONS, Examples, examples,
    make_optimizers, now, score_predictions, select_origins, sha, write_json)
from train_ternary_sign_size_forecast import predict_output

CONTRACT = "shared-log-hurdle-training-v1"


def log_examples(source, origins: np.ndarray) -> Examples:
    data = examples(source, origins)
    targets = np.log1p(data.targets)
    if not np.isfinite(targets).all() or not np.array_equal(targets == 0, data.targets == 0):
        raise ValueError("finite log targets preserving exact zero outcomes required")
    return Examples(data.features, targets, data.origins, data.axis_start_ms)


def log_constants(y: np.ndarray) -> list[dict]:
    result = []
    for j, h in enumerate(HORIZONS):
        value = y[:, j]
        active, up, down = value != 0, value > 0, value < 0
        if not up.any() or not down.any():
            raise ValueError("both training polarities required")
        result.append({"horizonSeconds": h,
            "pZero": float(((~active).sum()+.5)/(len(value)+1)),
            "pUpGivenActive": float((up.sum()+.5)/(active.sum()+1)),
            "upMeanLog": float(value[up].mean()), "downMeanLog": float(-value[down].mean()),
            "pooledActiveMeanLog": float(np.abs(value[active]).mean()),
            "activeTrainingExamples": int(active.sum()), "upTrainingExamples": int(up.sum()),
            "downTrainingExamples": int(down.sum())})
    return result


def task_scores(output: np.ndarray, y: np.ndarray, task: str, scales: np.ndarray) -> np.ndarray:
    if task in ("sign", "size"):
        return score_predictions(output, y, task, scales)
    if task == "zero":
        return (np.logaddexp(0, output)-output*(y == 0)).mean(0)
    if task != "combined":
        raise ValueError("unknown log training task")
    from log_hurdle_sign_size_forecast import log_hurdle_nll_elements
    return log_hurdle_nll_elements(torch.from_numpy(output), torch.from_numpy(y)).mean(0).numpy()


def batch_denominators(y: torch.Tensor, task: str) -> torch.Tensor:
    if task == "sign":
        return (y != 0).sum(0).clamp_min(1)
    if task == "size":
        return torch.stack((y > 0, y < 0), -1).sum(0).clamp_min(1)
    return torch.full((y.shape[1],), len(y), dtype=y.dtype, device=y.device)


def microbatch_loss(output: torch.Tensor, y: torch.Tensor, task: str,
                    scales: torch.Tensor, denominators: torch.Tensor) -> torch.Tensor:
    if task == "sign":
        values = F.binary_cross_entropy_with_logits(output, (y > 0).to(output.dtype), reduction="none")*(y != 0)
    elif task == "size":
        values = (output-y.abs().unsqueeze(-1)/scales).square()*torch.stack((y > 0,y < 0),-1)
    elif task == "zero":
        values = F.binary_cross_entropy_with_logits(output, (y == 0).to(output.dtype), reduction="none")
    elif task == "combined":
        from log_hurdle_sign_size_forecast import log_hurdle_nll_elements
        values = log_hurdle_nll_elements(output, y)
    else:
        raise ValueError("unknown log training task")
    return (values.sum(0)/denominators).mean()


def payload(model, task, epoch, mean, std, scales, plan_sha, initialization) -> dict:
    return {"contract": CONTRACT, "task": task, "horizonsSeconds": list(HORIZONS),
        "targetCoordinate": "log-return", "epoch": epoch,
        "modelState": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "inputMean": torch.from_numpy(mean), "inputStd": torch.from_numpy(std),
        "sizeScales": torch.from_numpy(scales), "planSha256": plan_sha,
        "initialization": initialization}


def verified_record(record: dict) -> dict:
    path = Path(record["checkpoint"])
    if sha(path) != record["sha256"]:
        raise ValueError("log checkpoint hash changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["contract"] != CONTRACT or saved["targetCoordinate"] != "log-return" \
            or saved["horizonsSeconds"] != list(HORIZONS):
        raise ValueError("checkpoint is not the shared log-target model")
    return saved


def load_combined(record: dict, device: torch.device):
    from log_hurdle_sign_size_forecast import LogHurdleSignSize
    saved = verified_record(record)
    if saved["task"] != "combined":
        raise ValueError("combined log checkpoint required")
    experts = [StructuredSpecialist(task, saved["inputMean"], saved["inputStd"], HORIZONS)
               for task in ("sign", "sign", "size")]
    model = LogHurdleSignSize(*experts, saved["sizeScales"])
    model.load_state_dict(saved["modelState"])
    return model.to(device).eval()


def fit(model, task, tx, ty, vx, vy, mean, std, scales, initialization, args) -> dict:
    combined = task == "combined"
    maximum, minimum, patience = ((args.fine_epochs, args.fine_min_epochs, args.fine_patience)
        if combined else (args.epochs, args.min_epochs, args.patience))
    lr, half_life = (1e-5, 2.) if combined else (1e-4, 4.)
    optimizers = make_optimizers(model)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = lr
    ema = deepcopy(model).eval()
    steps = math.ceil(len(tx)/args.batch_size)
    decay = .5**(1/(half_life*steps))
    best, last_improvement = {"score": math.inf, "epoch": -1}, 0
    started = time.monotonic()
    plan_sha = sha(args.output/"plan.json")
    tensor_scales = torch.from_numpy(scales).to(tx.device)
    restored = []
    if args.no_compile:
        forward = model
    elif combined:
        for expert in (model.zero, model.sign, model.size):
            restored.append((expert, expert.forward))
            expert.forward = torch.compile(expert.forward, dynamic=True, options={"triton.cudagraphs": False})
        forward = model
    else:
        forward = torch.compile(model, dynamic=True, options={"triton.cudagraphs": False})

    def emit(row):
        line = json.dumps({"task": task, **row}, allow_nan=False)
        print(line, flush=True)
        with (args.output/"training.jsonl").open("a", encoding="utf-8") as file:
            file.write(line+"\n")

    def evaluate_epoch(epoch, train_loss):
        output = predict_output(ema, vx, args.microbatch_size)
        scores = task_scores(output, vy, task, scales)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite validation score")
        aggregate = float(scores.mean())
        improved = aggregate < best["score"]
        if improved:
            path = args.output/f"{task}-shared.pt"
            saved = payload(ema, task, epoch, mean, std, scales, plan_sha, initialization)
            saved.update({"validationScore": aggregate, "validationScoresByHorizon": scores.tolist()})
            torch.save(saved, path)
            best.update({"checkpoint": str(path), "sha256": sha(path), "score": aggregate,
                "epoch": epoch, "validationScoresByHorizon": scores.tolist()})
        diagnostics = {}
        if combined:
            from log_hurdle_sign_size_forecast import log_hurdle_mean
            mu = log_hurdle_mean(torch.from_numpy(output)).numpy()
            diagnostics = {"validationLogMeanMseSkill": (1-np.square(mu-vy).mean(0)/np.square(vy).mean(0)).tolist(),
                "shapes": output[0, :, 4:6].tolist()}
        emit({"event": "epoch", "epoch": epoch, "trainingLoss": train_loss,
            "validationScores": scores.tolist(), "meanValidationScore": aggregate,
            "bestEpoch": best["epoch"], "seconds": time.monotonic()-started,
            "maximumGpuMiB": torch.cuda.max_memory_allocated()/2**20, **diagnostics})
        return improved

    emit({"event": "start", "trainingExamples": len(tx), "validationExamples": len(vx),
        "parameters": sum(p.numel() for p in model.parameters()), "learningRate": lr,
        "emaDecayPerStep": decay, "effectiveBatchSize": args.batch_size, "microbatchSize": args.microbatch_size})
    evaluate_epoch(0, None)
    generator = torch.Generator(device=tx.device).manual_seed(1337)
    completed = 0
    for epoch in range(1, maximum+1):
        model.train()
        order = torch.randperm(len(tx), generator=generator, device=tx.device)
        weighted_loss = 0.
        for start in range(0, len(tx), args.batch_size):
            indices = order[start:start+args.batch_size]
            denominator = batch_denominators(ty[indices], task)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros((), dtype=torch.float64, device=tx.device)
            for offset in range(0, len(indices), args.microbatch_size):
                chosen = indices[offset:offset+args.microbatch_size]
                value = microbatch_loss(forward(tx[chosen]), ty[chosen], task, tensor_scales, denominator)
                value.backward()
                accumulated += value.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            if not torch.isfinite(norm) or not torch.isfinite(accumulated):
                raise ValueError("nonfinite optimizer gradient or loss")
            for optimizer in optimizers:
                optimizer.step()
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters(), strict=True):
                    ep.lerp_(mp, 1-decay)
                for eb, mb in zip(ema.buffers(), model.buffers(), strict=True):
                    eb.copy_(mb)
            weighted_loss += float(accumulated)*len(indices)
        completed = epoch
        if evaluate_epoch(epoch, weighted_loss/len(tx)):
            last_improvement = epoch
        if epoch >= minimum and epoch-last_improvement >= patience:
            break
    emit({"event": "complete", "completedEpochs": completed, "seconds": time.monotonic()-started})
    for expert, original in restored:
        expert.forward = original
    del forward, optimizers, ema
    torch.cuda.empty_cache()
    return {"checkpoint": best["checkpoint"], "sha256": best["sha256"], "epoch": best["epoch"],
        "validationScore": best["score"], "validationScoresByHorizon": best["validationScoresByHorizon"],
        "completedEpochs": completed, "horizonsSeconds": list(HORIZONS),
        "validationMetric": {"zero": "meanAllRowZeroBCE", "sign": "meanActiveSignBCE",
            "size": "meanBalancedConditionalNormalizedLogMagnitudeMSE", "combined": "meanFullLogHurdleNLL"}[task],
        "checkpointSource": "starting-neural-model" if best["epoch"] == 0 else "trained-neural-EMA",
        "constantFallback": False}


@torch.no_grad()
def initialize_shapes(model, x, y, batch):
    totals, counts = np.zeros((len(HORIZONS), 2)), np.zeros((len(HORIZONS), 2), dtype=np.int64)
    # Use uncapped pretrained means to initialize shapes, then the model applies
    # its declared finite-exponential-moment map using these fixed initial shapes.
    for start in range(0, len(x), batch):
        means = (model.size(x[start:start+batch])*model.scales).cpu().numpy()
        values = y[start:start+batch]
        for k, mask in enumerate((values > 0, values < 0)):
            residual = np.abs(values)/np.maximum(means[..., k], 1e-12)-1
            totals[:, k] += np.where(mask, residual**2, 0).sum(0)
            counts[:, k] += mask.sum(0)
    if (counts == 0).any() or not np.isfinite(totals).all():
        raise ValueError("invalid training residual shape initialization")
    relative = totals/counts
    shapes = np.clip(np.divide(1., relative, out=np.full_like(relative, 50.), where=relative > 0), .01, 50.)
    model.set_shapes_(torch.from_numpy(shapes).to(x.device))
    return {"method": "inverse training mean squared relative log-magnitude residual, both Gamma sides",
        "relativeResidualSecondMoments": relative.tolist(), "shapes": shapes.tolist(),
        "examplesByDirection": counts.tolist(), "clipBounds": [.01, 50.], "testUsed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/training/runs/stochastic-crra-log-hurdle-v1"))
    parser.add_argument("--source-selection", type=Path, default=Path("data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json"))
    parser.add_argument("--online-selection", type=Path, default=Path("data/training/runs/stochastic-crra-online-calibration-v1/selection.json"))
    for name, default in (("epochs",96), ("min-epochs",24), ("patience",20), ("fine-epochs",64),
            ("fine-min-epochs",12), ("fine-patience",12), ("batch-size",10240),
            ("microbatch-size",2560), ("train-limit",256000), ("stride",4)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    if any(getattr(args, name) <= 0 for name in ("epochs","min_epochs","patience","fine_epochs",
            "fine_min_epochs","fine_patience","batch_size","microbatch_size","train_limit","stride")):
        raise ValueError("positive training limits required")
    plan = {"contract": CONTRACT, "createdAt": now(), "dataset": str(args.dataset),
        "datasetManifestSha256": sha(args.dataset/"manifest.json"), "horizonsSeconds": list(HORIZONS),
        "sourceSelectionFile": str(args.source_selection), "sourceSelectionSha256": sha(args.source_selection),
        "onlineSelectionFile": str(args.online_selection), "onlineSelectionSha256": sha(args.online_selection),
        "target": "float64 log1p(exact endpoint SIMPLE return); X=log(P[t+H]/P[t]); exact zeros preserved",
        "features": "two consecutive production59 observed rows t-1,t; train-only channel mean/std",
        "strideSeconds": args.stride, "trainingLimit": args.train_limit,
        "architecture": {"experts": 3, "horizonsPerExpert": 4, "crossExpertWeightSharing": False,
            "withinExpertHorizonBackboneSharing": True, "zero": architecture_config("sign"),
            "sign": architecture_config("sign"), "size": architecture_config("size")},
        "initialization": "all three experts freshly seeded1337; no previous weights",
        "pretraining": {"maximumEpochs": args.epochs, "minimumEpochs": args.min_epochs, "patience": args.patience,
            "zeroLoss": "all-row BCE", "signLoss": "active-only BCE; equal horizons",
            "sizeLoss": "conditional LOG-magnitude normalized MSE; equal polarities and horizons",
            "learningRate": 1e-4, "emaHalfLifeEpochs": 4., "selection": "one global validation-minimum checkpoint per expert, including epoch0"},
        "fineTuning": {"maximumEpochs": args.fine_epochs, "minimumEpochs": args.fine_min_epochs, "patience": args.fine_patience,
            "loss": "proper zero/sign/signed-Gamma log-return NLL; equal rows and horizons; no auxiliary MSE",
            "learningRate": 1e-5, "emaHalfLifeEpochs": 2., "allWeightsUnfrozen": True,
            "selection": "one global validation-minimum EMA checkpoint, including pretrained epoch0"},
        "magnitudeLaw": "signed Gamma log magnitudes and exact zero atom; UP mean=alpha*.49*tanh(rawMean/(alpha*.49)); scale<=.49 gives finite first and second price moments; DOWN positive unbounded log magnitude",
        "optimizer": "Muon/AdamW as preceding specialists, zero weight decay, grad norm cap100,seed1337",
        "effectiveBatchSize": args.batch_size, "microbatchSize": args.microbatch_size,
        "compile": not args.no_compile, "precision": "float32 neural high matmul; float64 log targets/scales/likelihood",
        "testProtocol": "trainer never loads test; evaluator runs after all checkpoint choices frozen",
        "onlineCalibration": "same frozen per-horizon modes/half-lives/ridges on log-law likelihood; causal maturity; no selection",
        "benchmark": "previously inspected test split; report both log and simple expected returns, active and all rows",
        "testUsedForSelection": False}
    plan_file = args.output/"plan.json"
    if args.run_prepared:
        previous = json.loads(plan_file.read_text())
        plan["createdAt"] = previous["createdAt"]
        if plan != previous or (args.output/"metadata.json").exists():
            raise ValueError("prepared plan differs or already ran")
    else:
        if plan_file.exists():
            raise ValueError("refusing to overwrite existing plan")
        write_json(plan_file, plan)
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "plan": str(plan_file), "sha256": sha(plan_file)}), flush=True)
        return
    from log_hurdle_sign_size_forecast import LogHurdleSignSize
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if not args.no_compile:
        cache = Path("data/training/cache/torchinductor").resolve()
        os.environ["TORCHINDUCTOR_CACHE_DIR"], os.environ["TRITON_CACHE_DIR"] = str(cache), str(cache/"triton")
    device, started = torch.device("cuda"), time.monotonic()
    ts, vs = load_timeline_split(args.dataset,"train"), load_timeline_split(args.dataset,"validation")
    train = log_examples(ts, select_origins(ts,args.stride,args.train_limit))
    validation = log_examples(vs, select_origins(vs,args.stride,None))
    if train.axis_start_ms+(train.origins[-1]+max(HORIZONS))*1000 >= validation.axis_start_ms+validation.origins[0]*1000:
        raise ValueError("train endpoint overlaps validation")
    flat = train.features.reshape(-1,59)
    mean = flat.mean(0,dtype=np.float64).astype(np.float32)
    std = np.maximum(flat.std(0,dtype=np.float64),1e-10).astype(np.float32)
    constants = log_constants(train.targets)
    scales = np.array([[c["upMeanLog"],c["downMeanLog"]] for c in constants])
    write_json(args.output/"metadata.json", {"contract": CONTRACT, "planSha256": sha(plan_file),
        "inputMean":mean.tolist(),"inputStd":std.tolist(),"constants":constants,
        "trainingExamples":len(train.targets),"validationExamples":len(validation.targets),
        "trainingFirstTimestamp":int(train.axis_start_ms+train.origins[0]*1000),
        "trainingLastEndpointTimestamp":int(train.axis_start_ms+(train.origins[-1]+max(HORIZONS))*1000),
        "validationFirstTimestamp":int(validation.axis_start_ms+validation.origins[0]*1000),
        "validationLastEndpointTimestamp":int(validation.axis_start_ms+(validation.origins[-1]+max(HORIZONS))*1000),
        "testLoaded":False,"testUsedForSelection":False})
    tx, ty = torch.from_numpy(train.features).to(device), torch.from_numpy(train.targets).to(device)
    vx = torch.from_numpy(validation.features).to(device)
    records = {}
    for task in ("zero","sign","size"):
        torch.manual_seed(1337)
        model = StructuredSpecialist("sign" if task == "zero" else task, torch.from_numpy(mean),torch.from_numpy(std),HORIZONS).to(device)
        records[task] = fit(model,task,tx,ty,vx,validation.targets,mean,std,scales,{"freshSeed":1337},args)
        del model
        torch.cuda.empty_cache()
        write_json(args.output/"pretraining-selection.json", {"experts":records,"testLoaded":False,"testUsedForSelection":False})
    experts = []
    for task in ("zero","sign","size"):
        saved = verified_record(records[task])
        expert = StructuredSpecialist("sign" if task == "zero" else task,torch.from_numpy(mean),torch.from_numpy(std),HORIZONS)
        expert.load_state_dict(saved["modelState"])
        experts.append(expert)
    model = LogHurdleSignSize(*experts,torch.from_numpy(scales)).to(device)
    shape_init = initialize_shapes(model,tx,train.targets,args.microbatch_size)
    initialization = {"experts":records,"shapeInitialization":shape_init,"allWeightsUnfrozen":True}
    base_scores = task_scores(predict_output(model,vx,args.microbatch_size),validation.targets,"combined",scales)
    baseline = args.output/"pretrained-shared.pt"
    saved = payload(model,"combined",0,mean,std,scales,sha(plan_file),initialization)
    saved.update({"validationScore":float(base_scores.mean()),"validationScoresByHorizon":base_scores.tolist()})
    torch.save(saved,baseline)
    pretrained = {"checkpoint":str(baseline),"sha256":sha(baseline),"epoch":0,
        "validationScore":float(base_scores.mean()),"validationScoresByHorizon":base_scores.tolist()}
    fine = fit(model,"combined",tx,ty,vx,validation.targets,mean,std,scales,initialization,args)
    selection = {"contract":CONTRACT,"createdAt":now(),"plan":plan,"planSha256":sha(plan_file),
        "constants":constants,"horizons":[{"horizonSeconds":h,"constants":constants[j]} for j,h in enumerate(HORIZONS)],
        "pretraining":records,"pretrained":pretrained,"finetuned":fine,
        "sourceSelectionFile":str(args.source_selection),"sourceSelectionSha256":plan["sourceSelectionSha256"],
        "onlineSelectionFile":str(args.online_selection),"onlineSelectionSha256":plan["onlineSelectionSha256"],
        "parameters":sum(p.numel() for p in model.parameters()),"expertBackbones":3,
        "allHeadSelectionsFrozen":True,"testLoaded":False,"testUsedForSelection":False}
    write_json(args.output/"selection.json",selection)
    result = {"contract":CONTRACT,"selectionFile":str(args.output/"selection.json"),
        "selectionSha256":sha(args.output/"selection.json"),"pretraining":records,"finetuned":fine,
        "seconds":time.monotonic()-started,"testLoaded":False,"testUsedForSelection":False}
    write_json(args.output/"training-result.json",result)
    print(json.dumps({"event":"allComplete",**result}),flush=True)


if __name__ == "__main__":
    main()
