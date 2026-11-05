"""Fresh historical structured backbones specialized to sign and conditional size.

All checkpoint choices precede test loading. Constants are controls, never model
substitutions. The output law uses SIMPLE-return conditional first moments.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
import torch
from torch.nn import functional as F

from export_stochastic_price_forecast import load_close_prices
from neural_sign_size_forecast import CONTRACT, StructuredSpecialist, architecture_config
from normalized_glu_next_return import optimizer_parameter_groups
from sign_size_forecast import DOWNSIDE_MAXIMUM, magnitude_metrics, sign_metrics, three_point_law, moment_values
from stochastic_price_forecast import TimelineSplit, load_timeline_split

HORIZONS = (1, 900, 1800, 7200)
PRICE_ROOT = Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s")


def write_json(file: Path, value: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Examples:
    features: np.ndarray
    targets: np.ndarray
    origins: np.ndarray
    axis_start_ms: int


def select_origins(source: TimelineSplit, stride: int, limit: int | None) -> np.ndarray:
    origins = source.origins[source.origins + max(HORIZONS) < len(source.timeline)][::stride]
    if limit is not None and len(origins) > limit:
        origins = origins[np.linspace(0, len(origins)-1, limit, dtype=np.int64)]
    if not len(origins):
        raise ValueError("no complete within-split multi-horizon examples")
    return origins


def examples(source: TimelineSplit, origins: np.ndarray) -> Examples:
    if origins.ndim != 1 or np.any(origins < 1) or np.any(origins + max(HORIZONS) >= len(source.timeline)):
        raise ValueError("incomplete causal input or horizon endpoint")
    first, final = int(origins[0]), int(origins[-1] + max(HORIZONS))
    rows = np.arange(first, final+1)
    prices = load_close_prices(source.axis_start_ms + rows*1000, PRICE_ROOT)
    if not np.all(np.isfinite(prices) & (prices > 0)):
        raise ValueError("positive finite endpoint prices required")
    indices = origins-first
    targets = np.stack([prices[indices+h]/prices[indices]-1 for h in HORIZONS], axis=1)
    x = np.asarray(source.timeline[origins[:, None] + np.array([-1, 0])], dtype=np.float32)
    return Examples(x, targets, origins, source.axis_start_ms)


def constants_from_training(y: np.ndarray) -> list[dict]:
    constants = []
    for j, h in enumerate(HORIZONS):
        value = y[:, j]
        active, up, down = value != 0, value > 0, value < 0
        if not up.any() or not down.any():
            raise ValueError("both active polarities required in training")
        constants.append({"horizonSeconds": h, "pZero": float(((~active).sum()+.5)/(len(value)+1)),
            "pUpGivenActive": float((up.sum()+.5)/(active.sum()+1)),
            "upMeanSimple": float(value[up].mean()), "downMeanSimple": float(-value[down].mean()),
            "pooledActiveMeanSimple": float(np.abs(value[active]).mean()),
            "activeTrainingExamples": int(active.sum()), "upTrainingExamples": int(up.sum()),
            "downTrainingExamples": int(down.sum())})
    return constants


def loss_terms(prediction: torch.Tensor, targets: torch.Tensor, task: str,
               scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-context loss/mask; sums permit exact effective-batch accumulation."""
    if task == "sign":
        mask = targets != 0
        losses = F.binary_cross_entropy_with_logits(prediction, (targets > 0).float(), reduction="none")
    else:
        mask = torch.stack((targets > 0, targets < 0), dim=-1)
        normalized = targets.abs().unsqueeze(-1)/scales
        losses = (prediction-normalized).square()
    return losses * mask, mask


def score_predictions(prediction: np.ndarray, y: np.ndarray, task: str,
                      scales: np.ndarray) -> np.ndarray:
    if task == "sign":
        mask = y != 0
        losses = np.logaddexp(0, prediction) - prediction*(y > 0)
        return (losses*mask).sum(axis=0)/mask.sum(axis=0)
    mask = np.stack((y > 0, y < 0), axis=-1)
    losses = (prediction - np.abs(y)[..., None]/scales)**2
    return ((losses*mask).sum(axis=0)/mask.sum(axis=0)).mean(axis=-1)


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray | torch.Tensor, batch: int,
            device: torch.device) -> np.ndarray:
    model.eval()
    pieces = []
    for start in range(0, len(x), batch):
        part = x[start:start+batch]
        tensor = part if isinstance(part, torch.Tensor) else torch.from_numpy(part).to(device)
        pieces.append(model(tensor).float().cpu().numpy().astype(np.float64))
    result = np.concatenate(pieces)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite neural forecast")
    return result


def make_optimizers(model: StructuredSpecialist) -> tuple[torch.optim.Optimizer, ...]:
    muon, adam = optimizer_parameter_groups(model)
    return (torch.optim.Muon(muon, lr=1e-4, weight_decay=0., momentum=.95, nesterov=True,
                ns_steps=3, eps=1e-7, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(adam, lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0., fused=True))


def train_task(task: str, train: Examples, validation: Examples, mean: np.ndarray,
               std: np.ndarray, constants: list[dict], plan: dict, args, device: torch.device,
               log_file: Path) -> list[dict]:
    torch.manual_seed(1337)
    model = StructuredSpecialist(task, torch.from_numpy(mean), torch.from_numpy(std), horizons=HORIZONS).to(device)
    ema = deepcopy(model).eval()
    optimizers = make_optimizers(model)
    scales_array = np.array([[c["upMeanSimple"], c["downMeanSimple"]] for c in constants])
    scales = torch.tensor(scales_array, dtype=torch.float32, device=device)
    x, y = torch.from_numpy(train.features).to(device), torch.from_numpy(train.targets).float().to(device)
    vx = torch.from_numpy(validation.features).to(device)
    steps = math.ceil(len(x)/args.batch_size)
    decay = .5**(1/(4*steps))
    best = [{"score": math.inf, "epoch": -1} for _ in HORIZONS]
    last_improvement = 0
    started = time.monotonic()
    forward = model
    if not args.no_compile:
        cache = Path("data/training/cache/torchinductor").resolve()
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
        os.environ["TRITON_CACHE_DIR"] = str(cache/"triton")
        forward = torch.compile(model, dynamic=True, options={"triton.cudagraphs": False})

    def emit(row: dict) -> None:
        row = {"task": task, **row}
        print(json.dumps(row, allow_nan=False), flush=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, allow_nan=False)+"\n")

    def evaluate_epoch(epoch: int, train_loss: float | None) -> bool:
        prediction = predict(ema, vx, args.microbatch_size, device)
        scores = score_predictions(prediction, validation.targets, task, scales_array)
        improved = False
        for j, h in enumerate(HORIZONS):
            if scores[j] < best[j]["score"]:
                improved = True
                path = args.output/f"{task}-h{h}.pt"
                payload = {"contract": CONTRACT, "task": task, "horizonsSeconds": list(HORIZONS),
                    "modelState": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                    "inputMean": torch.from_numpy(mean), "inputStd": torch.from_numpy(std),
                    "sizeScales": torch.from_numpy(scales_array), "constants": constants,
                    "epoch": epoch, "selectedHorizonSeconds": h, "validationScore": float(scores[j]),
                    "planSha256": sha(args.output/"plan.json"), "architecture": architecture_config(task)}
                torch.save(payload, path)
                best[j] = {"score": float(scores[j]), "epoch": epoch, "checkpoint": str(path),
                           "sha256": sha(path)}
        emit({"stage": "epoch", "epoch": epoch, "trainingLoss": train_loss,
              "validationScores": scores.tolist(), "bestEpochs": [b["epoch"] for b in best],
              "seconds": time.monotonic()-started, "maximumGpuMiB": torch.cuda.max_memory_allocated()/2**20})
        return improved

    emit({"stage": "start", "trainingExamples": len(train.targets), "validationExamples": len(validation.targets),
          "parameters": sum(v.numel() for v in model.parameters()), "emaDecayPerOptimizerStep": decay,
          "effectiveBatchSize": args.batch_size, "microbatchSize": args.microbatch_size})
    evaluate_epoch(0, None)
    generator = torch.Generator(device=device).manual_seed(1337)
    completed = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        order = torch.randperm(len(x), generator=generator, device=device)
        accumulated = 0.
        for start in range(0, len(x), args.batch_size):
            indices = order[start:start+args.batch_size]
            by = y[indices]
            if task == "sign":
                denominators = (by != 0).sum(dim=0).clamp_min(1)
            else:
                denominators = torch.stack((by > 0, by < 0), dim=-1).sum(dim=0).clamp_min(1)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for offset in range(0, len(indices), args.microbatch_size):
                chosen = indices[offset:offset+args.microbatch_size]
                prediction = forward(x[chosen])
                values, _ = loss_terms(prediction, y[chosen], task, scales)
                loss = (values.sum(dim=0)/denominators).mean()
                loss.backward()
                batch_loss += loss.detach()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            if not torch.isfinite(gradient) or not torch.isfinite(batch_loss):
                raise ValueError("nonfinite training loss or gradient")
            for optimizer in optimizers:
                optimizer.step()
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters(), strict=True):
                    ep.lerp_(mp, 1-decay)
                for eb, mb in zip(ema.buffers(), model.buffers(), strict=True):
                    eb.copy_(mb)
            accumulated += float(batch_loss)
        completed = epoch
        if evaluate_epoch(epoch, accumulated/steps):
            last_improvement = epoch
        if epoch >= args.min_epochs and epoch-last_improvement >= args.patience:
            break
    emit({"stage": "complete", "completedEpochs": completed, "seconds": time.monotonic()-started})
    result = []
    for item in best:
        result.append({"checkpoint": item["checkpoint"], "sha256": item["sha256"], "epoch": item["epoch"],
            "validationScore": item["score"], "validationMetric": "activeSignLogLoss" if task == "sign" else "balancedConditionalNormalizedSizeMse",
            "checkpointSource": "initialized-neural-model" if item["epoch"] == 0 else "trained-neural-EMA",
            "completedEpochs": completed, "constantFallback": False})
    del x, y, vx, model, ema, forward, optimizers
    torch.cuda.empty_cache()
    return result


def load_selected(entry: dict, device: torch.device) -> StructuredSpecialist:
    path = Path(entry["checkpoint"])
    if sha(path) != entry["sha256"]:
        raise ValueError("frozen checkpoint checksum mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = StructuredSpecialist(saved["task"], saved["inputMean"], saved["inputStd"], horizons=tuple(saved["horizonsSeconds"]))
    model.load_state_dict(saved["modelState"])
    return model.to(device).eval()


def combined_metrics(y: np.ndarray, p: np.ndarray, up: np.ndarray, down: np.ndarray, c: dict) -> dict:
    active, positive, negative = y != 0, y > 0, y < 0
    p = np.clip(p, 1e-12, 1-1e-12)
    returns, weights, clipping = three_point_law(c["pZero"], p, up, down)
    mean, size = moment_values(returns, weights)
    signed_mse = float(np.square(mean-y).mean())
    size_mse = float(np.square(size-np.abs(y)).mean())
    signed_baseline = float(np.square(y).mean())
    size_baseline = float(np.square(np.abs(y)-(1-c["pZero"])*c["pooledActiveMeanSimple"]).mean())
    correct = (p[active] >= .5) == positive[active]
    threshold = 1/.9995**2 - 1
    choose_long = mean > threshold
    return {"examples": len(y), "signGivenActive": sign_metrics(positive[active], p[active]),
        "signTrainConstant": sign_metrics(positive[active], np.full(int(active.sum()), c["pUpGivenActive"])),
        "signMagnitudeWeightedAccuracy": float(np.sum(np.abs(y[active])*correct)/np.abs(y[active]).sum()),
        "positiveSize": magnitude_metrics(y[positive]*1e4, up[positive]*1e4, c["upMeanSimple"]*1e4),
        "negativeSize": magnitude_metrics(-y[negative]*1e4, np.minimum(down[negative], DOWNSIDE_MAXIMUM)*1e4, c["downMeanSimple"]*1e4),
        "observedZeroFrequency": float((~active).mean()), "fixedZeroProbability": c["pZero"],
        "signedMeanMseBpsSquared": signed_mse*1e8, "signedMeanMseSkillVsZero": 1-signed_mse/signed_baseline,
        "absoluteSizeMseBpsSquared": size_mse*1e8, "absoluteSizeMseSkillVsTrainConstant": 1-size_mse/size_baseline,
        "meanPredictedSignedBps": float(mean.mean()*1e4), "meanPredictedAbsoluteBps": float(size.mean()*1e4),
        "meanRealizedAbsoluteBps": float(np.abs(y).mean()*1e4), "maximumPredictedMeanBps": float(mean.max()*1e4),
        "minimumPredictedMeanBps": float(mean.min()*1e4), "feeThresholdCrossings": int(choose_long.sum()),
        "realizedMeanNetWhenPredictedLongBps": float((.9995**2*(1+y[choose_long])-1).mean()*1e4) if choose_long.any() else None,
        "clipping": clipping, "economicsScope": "independent full-close cash-start threshold diagnostic, not account replay"}


def evaluate_combination(entries: list[dict], data: Examples, batch: int, device: torch.device) -> list[dict]:
    results = []
    for j, entry in enumerate(entries):
        sign = load_selected(entry["sign"], device)
        p = expit(predict(sign, data.features, batch, device)[:, j])
        del sign
        size = load_selected(entry["size"], device)
        normalized = predict(size, data.features, batch, device)[:, j]
        del size
        c = entry["constants"]
        up, down = normalized[:, 0]*c["upMeanSimple"], normalized[:, 1]*c["downMeanSimple"]
        value = data.targets[:, j]
        controls = {
            "full": (p, up, down),
            "signConstantSize": (p, np.full(len(p), c["upMeanSimple"]), np.full(len(p), c["downMeanSimple"])),
            "constantSignSize": (np.full(len(p), c["pUpGivenActive"]), up, down),
            "allTrainConstants": (np.full(len(p), c["pUpGivenActive"]), np.full(len(p), c["upMeanSimple"]), np.full(len(p), c["downMeanSimple"])),
        }
        results.append({"horizonSeconds": HORIZONS[j], "ablations": {name: combined_metrics(value, *arrays, c) for name, arrays in controls.items()}})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/training/runs/stochastic-crra-neural-specialists-v1"))
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10240)
    parser.add_argument("--microbatch-size", type=int, default=2560)
    parser.add_argument("--train-limit", type=int, default=256000)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if any(v <= 0 for v in (args.epochs, args.min_epochs, args.patience, args.batch_size, args.microbatch_size, args.train_limit, args.stride)):
        raise ValueError("positive training limits required")
    if (args.output/"plan.json").exists():
        raise ValueError("refusing to overwrite an existing experiment plan")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this bounded historical-backbone experiment")
    plan = {"contract": CONTRACT, "createdAt": now(), "dataset": str(args.dataset), "horizonsSeconds": list(HORIZONS),
        "datasetManifestSha256": sha(args.dataset/"manifest.json"),
        "features": "two consecutive raw production59 rows t-1,t; train-only per-channel mean/std over both observed rows",
        "trainingOrigins": "all origins with all horizons complete within split; every stride-th; if over limit use uniformly spaced index selection",
        "strideSeconds": args.stride, "trainingLimit": args.train_limit, "validationLimit": args.validation_limit,
        "architectures": {task: architecture_config(task) for task in ("sign", "size")},
        "target": "exact actual endpoint-price SIMPLE return P[t+H]/P[t]-1; exact equal prices give zero",
        "signLoss": "active-only BCE averaged equally over horizons; no magnitude weighting, active sign probability",
        "sizeLoss": "up/down masked squared error of positive SIMPLE size normalized by train conditional mean; equal horizon and polarity weights",
        "physicalFeatureAuxiliaryLoss": None, "training": {"epochs": args.epochs, "minimumEpochs": args.min_epochs,
            "patienceEpochsWithoutAnyHorizonImprovement": args.patience, "effectiveBatchSize": args.batch_size,
            "microbatchSize": args.microbatch_size, "gradientAccumulation": "exact loss denominators from full effective batch",
            "learningRate": .0001, "optimizer": "hybrid Muon/AdamW matching source plans", "muonMomentum": .95,
            "muonNesterov": True, "muonNewtonSchulzSteps": 3, "muonEpsilon": 1e-7, "muonAdjustLearningRate": "match_rms_adamw",
            "adamBetas": [.9,.999], "adamEpsilon": 1e-8, "weightDecay": 0., "dropout": 0., "seed": 1337,
            "gradientClip": 100., "emaHalfLifeEpochs": 4., "precision": "float32 high matmul precision",
            "compile": not args.no_compile, "compiledCudaGraphs": False},
        "selection": "independently per task/horizon validation logloss or balanced normalized conditional MSE; include initialized neural epoch0; no constant fallback",
        "zeroAtom": "fixed train Jeffreys frequency (zero+.5)/(N+1)",
        "constantControls": "train-only active sign and conditional sizes, never substituted for selected learned heads",
        "law": "three-point SIMPLE return {-mDown,0,+mUp}; weights {(1-p0)(1-pUp),p0,(1-p0)pUp}; within-sign spread approximated",
        "downsideClipping": {"maximum": DOWNSIDE_MAXIMUM, "scope": "law and reported negative magnitude metrics, recorded counts"},
        "testProtocol": "freeze all eight task/horizon checkpoint selections before loading any test split; report all horizons without test selection",
        "testUsedForSelection": False, "skipTest": args.skip_test,
        "policyReplay": {"commonSeconds": 151200, "horizonsSeconds": [900,1800,7200], "counts": [168,84,21],
            "full": ["paid", "zero-fee"], "signConstantSize": ["paid"], "constantSignSize": ["paid"],
            "nativeOneSecond": "forecast metrics only"}}
    write_json(args.output/"plan.json", plan)
    started = time.monotonic()
    train_source = load_timeline_split(args.dataset, "train")
    validation_source = load_timeline_split(args.dataset, "validation")
    train = examples(train_source, select_origins(train_source, args.stride, args.train_limit))
    validation = examples(validation_source, select_origins(validation_source, args.stride, args.validation_limit))
    flat = train.features.reshape(-1, 59)
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(flat.std(axis=0, dtype=np.float64), 1e-10).astype(np.float32)
    constants = constants_from_training(train.targets)
    metadata = {"contract": CONTRACT, "planSha256": sha(args.output/"plan.json"), "createdAt": now(),
        "inputMean": mean.tolist(), "inputStd": std.tolist(), "constants": constants,
        "trainingExamples": len(train.targets), "validationExamples": len(validation.targets),
        "trainingFirstTimestamp": int(train.axis_start_ms+train.origins[0]*1000),
        "trainingLastEndpointTimestamp": int(train.axis_start_ms+(train.origins[-1]+max(HORIZONS))*1000),
        "validationFirstTimestamp": int(validation.axis_start_ms+validation.origins[0]*1000),
        "validationLastEndpointTimestamp": int(validation.axis_start_ms+(validation.origins[-1]+max(HORIZONS))*1000),
        "testLoaded": False, "testUsedForSelection": False}
    if metadata["trainingLastEndpointTimestamp"] >= metadata["validationFirstTimestamp"]:
        raise ValueError("training endpoints overlap validation")
    write_json(args.output/"metadata.json", metadata)
    selected = {}
    for task in ("sign", "size"):
        selected[task] = train_task(task, train, validation, mean, std, constants, plan, args, device, args.output/"training.jsonl")
    entries = [{"horizonSeconds": h, "constants": constants[j], "sign": selected["sign"][j], "size": selected["size"][j]}
               for j, h in enumerate(HORIZONS)]
    frozen = {"contract": CONTRACT, "createdAt": now(), "plan": plan, "planSha256": sha(args.output/"plan.json"),
        "horizons": entries, "allHeadSelectionsFrozen": True, "testLoaded": False, "testUsedForSelection": False}
    write_json(args.output/"checkpoint-selection.json", frozen)
    write_json(args.output/"selection.json", frozen)
    validation_results = evaluate_combination(entries, validation, args.microbatch_size, device)
    write_json(args.output/"validation.json", {"contract": CONTRACT, "horizons": validation_results})
    result = {"contract": CONTRACT, "selectionFile": str(args.output/"selection.json"),
        "selectionSha256": sha(args.output/"selection.json"), "validation": validation_results,
        "testUsedForSelection": False, "trainingAndValidationSeconds": time.monotonic()-started}
    if not args.skip_test:
        test_source = load_timeline_split(args.dataset, "test")
        test = examples(test_source, select_origins(test_source, args.stride, None))
        if metadata["validationLastEndpointTimestamp"] >= test.axis_start_ms+test.origins[0]*1000:
            raise ValueError("validation endpoints overlap test")
        result["test"] = evaluate_combination(entries, test, args.microbatch_size, device)
        result["testExamples"] = len(test.targets)
        result["testEvaluationStrideSeconds"] = args.stride
        result["testFirstTimestamp"] = int(test.axis_start_ms+test.origins[0]*1000)
        result["testLastEndpointTimestamp"] = int(test.axis_start_ms+(test.origins[-1]+max(HORIZONS))*1000)
        write_json(args.output/"test.json", {"contract": CONTRACT, "horizons": result["test"],
            "evaluationStrideSeconds": args.stride, "testUsedForSelection": False})
    result["seconds"] = time.monotonic()-started
    write_json(args.output/"result.json", result)
    print(json.dumps({"stage": "allComplete", "result": str(args.output/"result.json"), "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
