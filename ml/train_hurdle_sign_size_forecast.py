"""Fine-tune one zero/sign/size triplet jointly across every forecast horizon.

All checkpoint choices use validation only. The magnitude density is a declared
continuous approximation of nonzero quantized returns, with an exact zero atom.
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
from scipy.special import expit
import torch
from torch.nn import functional as F

from hurdle_sign_size_forecast import (CONTRACT, MINIMUM_MEAN, HurdleSignSize,
    hurdle_components, hurdle_log_probabilities, hurdle_loss_parts, hurdle_mean,
    hurdle_nll_elements)
from neural_sign_size_forecast import StructuredSpecialist, architecture_config
from sign_size_forecast import magnitude_metrics, sign_metrics
from stochastic_price_forecast import load_timeline_split
from train_neural_sign_size_forecast import (HORIZONS, Examples, constants_from_training,
    examples, make_optimizers, now, score_predictions, select_origins, sha, write_json)
from train_ternary_sign_size_forecast import load_verified, predict_output

SHARED_CONTRACT = "shared-horizon-hurdle-finetune-v1"


def aggregate_validation_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty horizon scores required")
    return float(values.mean())


def loss_elements(output: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    if task == "zero":
        return F.binary_cross_entropy_with_logits(output, (y == 0).to(output.dtype), reduction="none")
    if task != "combined":
        raise ValueError("unknown training task")
    return hurdle_nll_elements(output, y)


def scores(output: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    return loss_elements(torch.from_numpy(output), torch.from_numpy(y), task).mean(dim=0).numpy()


def epoch_diagnostics(output: np.ndarray, y: np.ndarray, task: str, second: np.ndarray) -> dict:
    if task == "zero":
        p0 = expit(output)
        return {"zeroBrier": np.square(p0-(y == 0)).mean(axis=0).tolist(),
                "predictedZeroProbability": p0.mean(axis=0).tolist()}
    packed, targets = torch.from_numpy(output), torch.from_numpy(y)
    parts = hurdle_loss_parts(packed, targets)
    logp = hurdle_log_probabilities(packed).numpy()
    classes = (y > 0).astype(np.int64)+(y >= 0).astype(np.int64)
    actual = np.take_along_axis(logp, classes[..., None], axis=-1)[..., 0]
    mu = hurdle_mean(packed).numpy()
    return {"allTickSignedMeanMseNormalizedByTrainSecondMoment": (np.square(mu-y).mean(axis=0)/second).tolist(),
            "threeClassLogLoss": (-actual.mean(axis=0)).tolist(),
            "zeroBrier": np.square(np.exp(logp[..., 1])-(y == 0)).mean(axis=0).tolist(),
            "nllParts": {k:v.mean(dim=0).tolist() for k,v in parts.items()},
            "shapes": output[0, :, 4:6].tolist()}


def checkpoint_payload(model: torch.nn.Module, task: str, horizons: tuple[int, ...], epoch: int,
                       mean: np.ndarray, std: np.ndarray, scales: np.ndarray,
                       plan_sha: str, initialization: dict) -> dict:
    return {"contract": CONTRACT, "task": task, "horizonsSeconds": list(horizons),
        "modelState": {k:v.detach().cpu() for k,v in model.state_dict().items()},
        "inputMean": torch.from_numpy(mean), "inputStd": torch.from_numpy(std),
        "sizeScales": torch.from_numpy(scales), "epoch": epoch,
        "planSha256": plan_sha, "initialization": initialization}


def fit(model: torch.nn.Module, task: str, horizons: tuple[int, ...], tx: torch.Tensor, ty: torch.Tensor,
        vx: torch.Tensor, vy: np.ndarray, second: np.ndarray, mean: np.ndarray, std: np.ndarray,
        scales: np.ndarray, initialization: dict, args, *, maximum: int, minimum: int, patience: int,
        learning_rate: float, half_life: float, prefix: str) -> dict:
    optimizers = make_optimizers(model)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    ema = deepcopy(model).eval()
    steps = math.ceil(len(tx)/args.batch_size)
    decay = .5**(1/(half_life*steps))
    best = {"score": math.inf, "epoch": -1}
    last_improvement = 0
    started = time.monotonic()
    plan_sha = sha(args.output/"plan.json")
    restored_forwards = []
    if args.no_compile:
        forward = model
    elif task == "combined":
        # Compile the expensive experts independently. The small float64 hurdle
        # packing stays eager: compiling it as a whole on Windows can request a
        # CPU C++ kernel even though all learned parameters reside on CUDA.
        for expert in (model.zero, model.sign, model.size):
            restored_forwards.append((expert, expert.forward))
            expert.forward = torch.compile(expert.forward, dynamic=True, options={"triton.cudagraphs": False})
        forward = model
    else:
        forward = torch.compile(model, dynamic=True, options={"triton.cudagraphs": False})

    def emit(value: dict):
        line = json.dumps({"stage": prefix, "horizonsSeconds": list(horizons), **value}, allow_nan=False)
        print(line, flush=True)
        with (args.output/"training.jsonl").open("a", encoding="utf-8") as file:
            file.write(line+"\n")

    def evaluate_epoch(epoch: int, training_loss: float | None) -> bool:
        prediction = predict_output(ema, vx, args.microbatch_size)
        validation = scores(prediction, vy, task)
        if not np.isfinite(validation).all():
            raise ValueError("nonfinite validation likelihood")
        score = aggregate_validation_score(validation)
        improved = score < best["score"]
        if improved:
            path = args.output/f"{prefix}-shared.pt"
            saved = checkpoint_payload(ema, task, horizons, epoch, mean, std, scales, plan_sha, initialization)
            saved.update({"validationScore": score, "validationScoresByHorizon": validation.tolist()})
            torch.save(saved, path)
            best.update({"checkpoint": str(path), "sha256": sha(path), "epoch": epoch, "score": score,
                         "validationScoresByHorizon": validation.tolist()})
        emit({"event": "epoch", "epoch": epoch, "trainingLoss": training_loss,
              "validationScores": validation.tolist(), "meanValidationScore": score, "bestEpoch": best["epoch"],
              "validationDiagnostics": epoch_diagnostics(prediction, vy, task, second),
              "seconds": time.monotonic()-started, "maximumGpuMiB": torch.cuda.max_memory_allocated()/2**20})
        return improved

    emit({"event": "start", "trainingExamples": len(tx), "validationExamples": len(vx),
          "parameters": sum(p.numel() for p in model.parameters()), "effectiveBatchSize": args.batch_size,
          "microbatchSize": args.microbatch_size, "learningRate": learning_rate, "emaDecayPerStep": decay})
    evaluate_epoch(0, None)
    generator = torch.Generator(device=tx.device).manual_seed(1337)
    completed = 0
    for epoch in range(1, maximum+1):
        model.train()
        order = torch.randperm(len(tx), generator=generator, device=tx.device)
        weighted_loss = 0.
        for start in range(0, len(tx), args.batch_size):
            indices = order[start:start+args.batch_size]
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros((), dtype=torch.float64, device=tx.device)
            for offset in range(0, len(indices), args.microbatch_size):
                chosen = indices[offset:offset+args.microbatch_size]
                value = loss_elements(forward(tx[chosen]), ty[chosen], task)
                loss = value.sum()/(len(indices)*ty.shape[1])
                loss.backward()
                accumulated += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            if not torch.isfinite(norm) or not torch.isfinite(accumulated):
                raise ValueError("nonfinite likelihood or optimizer gradient")
            for optimizer in optimizers:
                optimizer.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(), model.parameters(), strict=True):
                    ep.lerp_(mp, 1-decay)
                for eb,mb in zip(ema.buffers(), model.buffers(), strict=True):
                    eb.copy_(mb)
            weighted_loss += float(accumulated)*len(indices)
        completed = epoch
        if evaluate_epoch(epoch, weighted_loss/len(tx)):
            last_improvement = epoch
        if epoch >= minimum and epoch-last_improvement >= patience:
            break
    emit({"event": "complete", "completedEpochs": completed, "seconds": time.monotonic()-started})
    record = {"checkpoint": best["checkpoint"], "sha256": best["sha256"], "epoch": best["epoch"],
        "validationScore": best["score"], "validationScoresByHorizon": best["validationScoresByHorizon"],
        "validationMetric": "meanHorizonFullHurdleNLL", "horizonsSeconds": list(horizons),
        "completedEpochs": completed, "checkpointSource": "shared-pretrained-triplet" if best["epoch"] == 0 else "trained-EMA",
        "constantFallback": False}
    for expert, original_forward in restored_forwards:
        expert.forward = original_forward
    del forward, ema, optimizers
    torch.cuda.empty_cache()
    return record


def make_shared_triplet(records: dict, mean: np.ndarray, std: np.ndarray,
                        scales: np.ndarray) -> HurdleSignSize:
    saved = [load_verified(records[name]) for name in ("zero", "sign", "size")]
    experts = []
    for state,task in zip(saved, ("sign", "sign", "size"), strict=True):
        if state["horizonsSeconds"] != list(HORIZONS):
            raise ValueError("pretrained horizon alignment differs")
        np.testing.assert_array_equal(state["inputMean"].numpy(), mean)
        np.testing.assert_array_equal(state["inputStd"].numpy(), std)
        expert = StructuredSpecialist(task, torch.from_numpy(mean), torch.from_numpy(std), horizons=HORIZONS)
        expert.load_state_dict(state["modelState"])
        experts.append(expert)
    np.testing.assert_array_equal(saved[2]["sizeScales"].numpy(), scales)
    return HurdleSignSize(*experts, torch.from_numpy(scales))


@torch.no_grad()
def choose_initializer(candidates: list[dict], x: torch.Tensor, y: np.ndarray, task: str,
                       mean: np.ndarray, std: np.ndarray, scales: np.ndarray,
                       batch: int, device: torch.device) -> tuple[dict, list[dict]]:
    """Select one intact retained multi-horizon expert using validation only."""
    evaluated, seen = [], set()
    for record in candidates:
        if record["sha256"] in seen:
            continue
        seen.add(record["sha256"])
        saved = load_verified(record)
        if saved["horizonsSeconds"] != list(HORIZONS) or saved["task"] != task:
            raise ValueError("initializer task or horizon order differs")
        np.testing.assert_array_equal(saved["inputMean"].numpy(), mean)
        np.testing.assert_array_equal(saved["inputStd"].numpy(), std)
        model = StructuredSpecialist("sign" if task == "zero" else task,
            torch.from_numpy(mean), torch.from_numpy(std), horizons=HORIZONS).to(device)
        model.load_state_dict(saved["modelState"])
        output = predict_output(model, x, batch)
        values = scores(output, y, "zero") if task == "zero" else score_predictions(output, y, task, scales)
        evaluated.append({"record":record,"validationScoresByHorizon":values.tolist(),
                          "meanValidationScore":aggregate_validation_score(values)})
        del model
    if not evaluated:
        raise ValueError("no retained initializer candidates")
    selected = min(evaluated, key=lambda row: row["meanValidationScore"])
    return selected["record"], evaluated


def load_combined(record: dict, device: torch.device) -> HurdleSignSize:
    saved = load_verified(record)
    if saved["contract"] != CONTRACT or saved["task"] != "combined":
        raise ValueError("unsupported hurdle checkpoint")
    experts = [StructuredSpecialist(task, saved["inputMean"], saved["inputStd"], horizons=tuple(saved["horizonsSeconds"]))
               for task in ("sign", "sign", "size")]
    model = HurdleSignSize(*experts, saved["sizeScales"])
    model.load_state_dict(saved["modelState"])
    return model.to(device).eval()


@torch.no_grad()
def initialize_shapes(model: HurdleSignSize, x: torch.Tensor, y: np.ndarray, batch: int) -> dict:
    """Train-only relative-residual moment initializer; Beta analogue is heuristic."""
    model.eval()
    if y.shape != (len(x), len(model.horizons)):
        raise ValueError("shape initializer targets must cover all model horizons")
    totals = np.zeros((len(model.horizons), 2))
    counts = np.zeros_like(totals, dtype=np.int64)
    for start in range(0, len(x), batch):
        output = model(x[start:start+batch]).cpu().numpy()
        targets = y[start:start+batch]
        for k,positive in enumerate((True, False)):
            mask = targets > 0 if positive else targets < 0
            ratio = np.abs(targets)/output[..., 2+k]
            totals[:, k] += np.where(mask, np.square(ratio-1), 0).sum(axis=0)
            counts[:, k] += mask.sum(axis=0)
    if np.any(counts == 0) or not np.isfinite(totals).all():
        raise ValueError("finite relative residuals and both training signs required")
    relative = totals/counts
    raw = np.divide(1., relative, out=np.full_like(relative, 50.), where=relative > 0)
    shapes = np.clip(raw, .01, 50.)
    model.set_shapes_(torch.from_numpy(shapes).to(x.device))
    return {"method": "inverse train mean squared relative residual; DOWN Beta initialization is a heuristic",
            "examplesByDirection": counts.tolist(), "relativeResidualSecondMoments": relative.tolist(),
            "unclippedShapes": raw.tolist(), "shapes": shapes.tolist(), "clipBounds": [.01, 50.],
            "clippedCount": int((raw != shapes).sum()), "testUsed": False}


def metrics(output: np.ndarray, y: np.ndarray, c: dict) -> dict:
    if not np.isfinite(output).all() or not np.all(np.isfinite(y) & (y > -1)):
        raise ValueError("finite likelihood parameters and returns greater than -1 required")
    packed, target = torch.from_numpy(output[:, None, :]), torch.from_numpy(y[:, None])
    probability, up_t, down_t = hurdle_components(packed)
    p, up, down = probability[:, 0].numpy(), up_t[:, 0].numpy(), down_t[:, 0].numpy()
    mu = hurdle_mean(packed)[:, 0].numpy()
    absolute = p[:, 2]*up+p[:, 0]*down
    active, positive, negative = y != 0, y > 0, y < 0
    classes = positive.astype(np.int64)+(y >= 0).astype(np.int64)
    logp = hurdle_log_probabilities(packed)[:, 0].numpy()
    parts = {k:float(v.mean()) for k,v in hurdle_loss_parts(packed, target).items()}
    mse, mean_second = float(np.square(mu-y).mean()), float(np.square(y).mean())
    size_mse = float(np.square(absolute-np.abs(y)).mean())
    size_baseline = float(np.square(np.abs(y)-(1-c["pZero"])*c["pooledActiveMeanSimple"]).mean())
    p_sign = expit(output[:, 1])
    conditional_mu = p_sign*up-(1-p_sign)*down
    active_second = float(np.square(y[active]).mean())
    sign_score = sign_metrics(positive[active], np.clip(p_sign[active], 1e-12, 1-1e-12))
    sign_score["logLoss"] = float((np.logaddexp(0, output[active, 1])-output[active, 1]*positive[active]).mean())
    return {"examples": len(y), "activeExamples": int(active.sum()),
        "fullHurdleReturnNegativeLogLikelihood": sum(parts.values()), "nllParts": parts,
        "threeClassLogLoss": float(-logp[np.arange(len(y)), classes].mean()),
        "threeClassBrier": float(np.square(p-np.eye(3)[classes]).sum(axis=1).mean()),
        "zeroBrier": float(np.square(p[:, 1]-(~active)).mean()),
        "predictedZeroProbability": float(p[:, 1].mean()), "realizedZeroFrequency": float((~active).mean()),
        "signGivenActive": sign_score,
        "magnitudeWeightedSignAccuracy": float(np.dot(np.abs(y[active]), (output[active, 1] >= 0) == positive[active])/np.abs(y[active]).sum()),
        "positiveSize": magnitude_metrics(y[positive]*1e4, up[positive]*1e4, c["upMeanSimple"]*1e4),
        "negativeSize": magnitude_metrics(-y[negative]*1e4, down[negative]*1e4, c["downMeanSimple"]*1e4),
        "allTickSignedMeanMseBpsSquared": mse*1e8, "allTickSignedMeanMseSkillVsZero": 1-mse/mean_second,
        "allTickAbsoluteSizeMseBpsSquared": size_mse*1e8, "allTickAbsoluteSizeMseSkillVsTrainConstant": 1-size_mse/size_baseline,
        "activeOnlyConditionalSignedMeanMseSkillVsZero": 1-float(np.square(conditional_mu[active]-y[active]).mean())/active_second,
        "activeOnlyDeployedSignedMeanMseSkillVsZero": 1-float(np.square(mu[active]-y[active]).mean())/active_second,
        "predictedMeanReturnBps": float(mu.mean()*1e4), "predictedAverageAbsoluteSizeBps": float(absolute.mean()*1e4),
        "maximumPredictedMeanBps": float(mu.max()*1e4), "minimumPredictedMeanBps": float(mu.min()*1e4),
        "feeThresholdCrossings": int((mu > 1/.9995**2-1).sum()),
        "probabilityNormalizationMaximumError": float(np.max(np.abs(p.sum(axis=1)-1))),
        "meanIdentityMaximumError": float(np.max(np.abs(mu-(p[:, 2]*up-p[:, 0]*down)))),
        "minimumMeanFloorHits": {"up": int((up <= MINIMUM_MEAN).sum()), "down": int((down <= MINIMUM_MEAN).sum())},
        "downsideAtPhysicalBoundaryCount": int((down >= 1-1e-12).sum()),
        "actualClassProbabilityBelow1eMinus12Count": int((logp[np.arange(len(y)), classes] < math.log(1e-12)).sum()),
        "shapes": output[0, 4:6].tolist(),
        "componentInterpretation": "proper mixed likelihood under a declared Gamma/Beta family; misspecification, especially Beta, need not minimize mean squared error"}


@torch.no_grad()
def evaluate(records: dict, data: Examples, constants: list[dict], batch: int,
             device: torch.device) -> list[dict]:
    """One forward per full triplet and batch, scoring all horizon outputs."""
    x = torch.from_numpy(data.features).to(device)
    result = [{"horizonSeconds":h,"models":{}} for h in HORIZONS]
    for name,record in records.items():
        model = load_combined(record,device)
        if model.horizons != HORIZONS:
            raise ValueError("shared evaluation requires every horizon in one checkpoint")
        output = predict_output(model,x,batch)
        for j,row in enumerate(result):
            row["models"][name] = metrics(output[:,j],data.targets[:,j],constants[j])
        del model
    return result


def make_plan(args, source: dict, candidate_records: dict) -> dict:
    previous = source["plan"]
    return {"contract":SHARED_CONTRACT,"createdAt":now(),"dataset":previous["dataset"],
        "datasetManifestSha256":previous["datasetManifestSha256"],"horizonsSeconds":list(HORIZONS),
        "sourceSelectionFile":str(args.source_selection),"sourceSelectionSha256":sha(args.source_selection),
        "onlineSelectionFile":str(args.online_selection),"onlineSelectionSha256":sha(args.online_selection),
        "features":previous["features"],"strideSeconds":previous["strideSeconds"],"trainingLimit":previous["trainingLimit"],
        "initialization":{"candidateRecords":candidate_records,"selection":"one intact retained checkpoint per expert, minimum equal-horizon mean of original task validation scores; candidate order breaks ties",
            "zeroObjective":"all-row zero BCE per horizon","signObjective":"active-only sign BCE per horizon",
            "sizeObjective":"conditional normalized size MSE, equal directions within each horizon",
            "scope":"selection among retained multi-horizon checkpoints, not unretained training epochs",
            "preserve":"all readout rows and full shared backbone; no per-horizon checkpoint splicing"},
        "architecture":{"experts":3,"horizonsPerExpert":4,"crossExpertWeightSharing":False,
            "withinExpertHorizonBackboneSharing":True,"zero":architecture_config("sign"),
            "sign":architecture_config("sign"),"size":architecture_config("size")},
        "fineTuning":{"maximumEpochs":args.fine_epochs,"minimumEpochs":args.fine_min_epochs,"patience":args.fine_patience,
            "loss":"full hurdle NLL averaged equally over every row and horizon; no auxiliary MSE",
            "selection":"ONE shared EMA checkpoint minimizing equal-horizon mean full validation NLL, including pretrained epoch0",
            "learningRate":1e-5,"emaHalfLifeEpochs":2.,"allWeightsUnfrozen":True,
            "diagnosticsEveryEpoch":["fullNLLByHorizon","NLLparts","signedMeanMSE","threeClassCE","zeroBrier"]},
        "magnitudeLaw":previous["magnitudeLaw"],"optimizer":previous["optimizer"],
        "effectiveBatchSize":args.batch_size,"microbatchSize":args.microbatch_size,
        "gradientAccumulation":"exact full effective-batch and horizon mean","compile":not args.no_compile,
        "compiledExecution":"compile each expert separately; float64 physical packing/NLL eager; CUDA graphs disabled",
        "precision":"float32 neural experts/high matmul precision; float64 physical means/scales/shapes/targets/NLL",
        "testProtocol":"trainer never loads test; one shared initializer and fine checkpoint freeze before evaluator loads test",
        "onlineCalibration":"reuse prior validation-preferred config independently per horizon for BOTH architectures; no calibration reselection",
        "fixedCheckpointEvaluation":["train","validation"],"testUsedForSelection":False,
        "debugLimits":{"train":args.debug_train_limit,"validation":args.debug_validation_limit}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-selection",type=Path,default=Path("data/training/runs/stochastic-crra-hurdle-nll-v1/selection.json"))
    parser.add_argument("--online-selection",type=Path,default=Path("data/training/runs/stochastic-crra-online-calibration-v1/selection.json"))
    parser.add_argument("--output",type=Path,default=Path("data/training/runs/stochastic-crra-shared-hurdle-v1"))
    for name,default in (("fine-epochs",64),("fine-min-epochs",12),("fine-patience",12),
                         ("batch-size",10240),("microbatch-size",2560)):
        parser.add_argument("--"+name,type=int,default=default)
    parser.add_argument("--debug-train-limit",type=int)
    parser.add_argument("--debug-validation-limit",type=int)
    parser.add_argument("--no-compile",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--run-prepared",action="store_true")
    args = parser.parse_args()
    if any(v <= 0 for v in (args.fine_epochs,args.fine_min_epochs,args.fine_patience,args.batch_size,args.microbatch_size)):
        raise ValueError("positive training limits required")
    source = json.loads(args.source_selection.read_text())
    source_meta = json.loads((args.source_selection.parent/"metadata.json").read_text())
    online = json.loads(args.online_selection.read_text())
    if not source["allHeadSelectionsFrozen"] or source["testUsedForSelection"] or source["testLoaded"]:
        raise ValueError("source choices must be frozen without test")
    if not online["allCalibrationSelectionsFrozen"] or online["testLoaded"] or online["testUsedForSelection"] \
            or online["sourceSelectionSha256"] != sha(args.source_selection):
        raise ValueError("online baseline choices must be frozen and match source models")
    dataset,previous = Path(source["plan"]["dataset"]),source["plan"]
    if sha(dataset/"manifest.json") != previous["datasetManifestSha256"]:
        raise ValueError("source dataset changed")
    candidates = {task:[entry["pretrained"][task] for entry in source["horizons"]] for task in ("zero","sign","size")}
    for records in candidates.values():
        for record in records:
            load_verified(record)
    plan = make_plan(args,source,candidates)
    plan_file = args.output/"plan.json"
    if args.run_prepared:
        prepared = json.loads(plan_file.read_text())
        plan["createdAt"] = prepared["createdAt"]
        if plan != prepared or (args.output/"metadata.json").exists() or (args.output/"training.jsonl").exists():
            raise ValueError("prepared plan differs or already ran; refusing overwrite")
    else:
        if plan_file.exists():
            raise ValueError("refusing to overwrite immutable experiment plan")
        write_json(plan_file,plan)
    if args.prepare_only:
        print(json.dumps({"event":"prepared","plan":str(plan_file),"sha256":sha(plan_file)}),flush=True)
        return
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1337)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if not args.no_compile:
        cache = Path("data/training/cache/torchinductor").resolve()
        os.environ["TORCHINDUCTOR_CACHE_DIR"],os.environ["TRITON_CACHE_DIR"] = str(cache),str(cache/"triton")
    device,started = torch.device("cuda"),time.monotonic()
    train_source,validation_source = load_timeline_split(dataset,"train"),load_timeline_split(dataset,"validation")
    train = examples(train_source,select_origins(train_source,previous["strideSeconds"],previous["trainingLimit"]))
    validation = examples(validation_source,select_origins(validation_source,previous["strideSeconds"],None))
    if len(train.targets) != source_meta["trainingExamples"] or len(validation.targets) != source_meta["validationExamples"]:
        raise ValueError("source common-origin population changed")
    constants = constants_from_training(train.targets)
    for current,entry in zip(constants,source["horizons"],strict=True):
        if current != entry["constants"]:
            raise ValueError("source training constants changed")
    mean,std = np.asarray(source_meta["inputMean"],dtype=np.float32),np.asarray(source_meta["inputStd"],dtype=np.float32)
    np.testing.assert_array_equal(train.features.reshape(-1,59).mean(axis=0,dtype=np.float64).astype(np.float32),mean)
    np.testing.assert_array_equal(np.maximum(train.features.reshape(-1,59).std(axis=0,dtype=np.float64),1e-10).astype(np.float32),std)
    scales = np.array([[c["upMeanSimple"],c["downMeanSimple"]] for c in constants])
    second = np.square(train.targets).mean(axis=0)
    full_counts = (len(train.targets),len(validation.targets))
    def subset(data: Examples,limit: int | None):
        if limit is None or limit >= len(data.targets):
            return data
        if limit <= 0:
            raise ValueError("positive debug limit required")
        indices = np.linspace(0,len(data.targets)-1,limit,dtype=np.int64)
        return Examples(data.features[indices],data.targets[indices],data.origins[indices],data.axis_start_ms)
    train,validation = subset(train,args.debug_train_limit),subset(validation,args.debug_validation_limit)
    for data in (train,validation):
        if not np.all(np.isfinite(data.targets) & (data.targets > -1)):
            raise ValueError("likelihood requires finite simple returns greater than -1")
    write_json(args.output/"metadata.json",{"contract":SHARED_CONTRACT,"planSha256":sha(plan_file),
        "inputMean":mean.tolist(),"inputStd":std.tolist(),"constants":constants,
        "trainingExamples":len(train.targets),"validationExamples":len(validation.targets),"originalExamples":list(full_counts),
        "trainReturnSecondMoments":second.tolist(),"testLoaded":False,"testUsedForSelection":False})
    tx,ty = torch.from_numpy(train.features).to(device),torch.from_numpy(train.targets).to(device)
    vx = torch.from_numpy(validation.features).to(device)
    records,initializers = {},{}
    for task in ("zero","sign","size"):
        chosen,compared = choose_initializer(candidates[task],vx,validation.targets,task,mean,std,scales,args.microbatch_size,device)
        records[task] = chosen
        initializers[task] = {"selected":chosen,"candidates":compared}
        print(json.dumps({"event":"initializerSelected","task":task,"epoch":chosen["epoch"],"checkpoint":chosen["checkpoint"],
            "meanValidationScores":[row["meanValidationScore"] for row in compared]}),flush=True)
    initializer_file = args.output/"initializer-selection.json"
    write_json(initializer_file,{"contract":SHARED_CONTRACT,"planSha256":sha(plan_file),"experts":initializers,
        "allInitializerChoicesFrozen":True,"testLoaded":False,"testUsedForSelection":False})
    model = make_shared_triplet(records,mean,std,scales).to(device)
    shape_init = initialize_shapes(model,tx,train.targets,args.microbatch_size)
    init = {"kind":"one-intact-pretrained-multihorizon-checkpoint-per-expert","experts":records,
        "initializerSelectionFile":str(initializer_file),"initializerSelectionSha256":sha(initializer_file),
        "shapeInitialization":shape_init,"allWeightsUnfrozen":True,"horizonsSeconds":list(HORIZONS)}
    baseline = args.output/"pretrained-shared.pt"
    base_values = scores(predict_output(model,vx,args.microbatch_size),validation.targets,"combined")
    base_score = aggregate_validation_score(base_values)
    saved = checkpoint_payload(model,"combined",HORIZONS,0,mean,std,scales,sha(plan_file),init)
    saved.update({"validationScore":base_score,"validationScoresByHorizon":base_values.tolist()})
    torch.save(saved,baseline)
    base_record = {"checkpoint":str(baseline),"sha256":sha(baseline),"epoch":0,"validationScore":base_score,
        "validationScoresByHorizon":base_values.tolist(),"validationMetric":"meanHorizonFullHurdleNLL",
        "horizonsSeconds":list(HORIZONS),"checkpointSource":"shared-pretrained-triplet"}
    fine = fit(model,"combined",HORIZONS,tx,ty,vx,validation.targets,second,mean,std,scales,init,args,
        maximum=args.fine_epochs,minimum=args.fine_min_epochs,patience=args.fine_patience,
        learning_rate=1e-5,half_life=2.,prefix="finetuned")
    parameter_count = sum(p.numel() for p in model.parameters())
    del model,tx,ty,vx
    torch.cuda.empty_cache()
    selection = {"contract":SHARED_CONTRACT,"createdAt":now(),"plan":plan,"planSha256":sha(plan_file),
        "sourceSelectionFile":str(args.source_selection),"sourceSelectionSha256":sha(args.source_selection),
        "onlineSelectionFile":str(args.online_selection),"onlineSelectionSha256":sha(args.online_selection),
        "initializerSelectionFile":str(initializer_file),"initializerSelectionSha256":sha(initializer_file),
        "pretrained":base_record,"finetuned":fine,"parameters":parameter_count,"expertBackbones":3,
        "horizons":[{"horizonSeconds":h,"constants":constants[j]} for j,h in enumerate(HORIZONS)],
        "allHeadSelectionsFrozen":True,"testLoaded":False,"testUsedForSelection":False}
    selection_file = args.output/"selection.json"
    write_json(selection_file,selection)
    result = {"contract":SHARED_CONTRACT,"selectionFile":str(selection_file),"selectionSha256":sha(selection_file),
        "testLoaded":False,"testUsedForSelection":False}
    for split,data in (("train",train),("validation",validation)):
        result[split] = evaluate({"sharedPretrained":base_record,"sharedFinetuned":fine},data,constants,args.microbatch_size,device)
        write_json(args.output/f"training-{split}.json",{"contract":SHARED_CONTRACT,"horizons":result[split],"selectionFrozen":True})
    result["seconds"] = time.monotonic()-started
    write_json(args.output/"training-result.json",result)
    for file,digest in ((args.source_selection,plan["sourceSelectionSha256"]),(args.online_selection,plan["onlineSelectionSha256"])):
        if sha(file) != digest:
            raise ValueError("source selection changed during training")
    print(json.dumps({"event":"allComplete","result":str(args.output/"training-result.json"),"seconds":result["seconds"],
        "selectedEpoch":fine["epoch"],"completedEpochs":fine["completedEpochs"],"parameters":parameter_count}),flush=True)


if __name__ == "__main__":
    main()
