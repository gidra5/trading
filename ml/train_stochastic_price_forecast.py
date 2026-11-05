"""Train/select a causal mixed price distribution without inspecting test PnL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from stochastic_price_forecast import CONTRACT, StochasticPriceForecast, fit_training_basis, load_timeline_split
from train_autoregressive_minute_return import build_optimizers


def write_json(file: Path, value: dict) -> None:
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, file)


@torch.no_grad()
def evaluate(model: StochasticPriceForecast, features: torch.Tensor, targets: torch.Tensor, batch: int) -> dict:
    model.eval()
    totals = np.zeros(6, dtype=np.float64)
    for start in range(0, len(targets), batch):
        values = targets[start:start + batch]
        logits = model(features[start:start + batch])
        mean = model.expectation_from_logits(logits)
        p0 = torch.sigmoid(logits[:, 0])
        metrics = torch.stack((-model.log_prob_from_logits(logits, values).sum(),
                               (mean - values).square().sum(), values.square().sum(),
                               (p0 - (values == 0).float()).square().sum(),
                               p0.sum(), (values == 0).float().sum()))
        totals += metrics.double().cpu().numpy()
    return {"examples": len(targets), "negativeLogLikelihood": float(totals[0] / len(targets)),
            "mse": float(totals[1] / len(targets)), "mseSkillVsZero": float(1 - totals[1] / totals[2]),
            "zeroBrier": float(totals[3] / len(targets)), "predictedZeroProbability": float(totals[4] / len(targets)),
            "observedZeroProbability": float(totals[5] / len(targets))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/training/runs/stochastic-crra-production178-zeroatom-k256-v1"))
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "best.pt").exists():
        raise ValueError("refusing to overwrite an existing trained forecast; use a new output directory")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    splits = {name: load_timeline_split(args.dataset, name) for name in ("train", "validation")}
    arrays = {name: split.features() for name, split in splits.items()}
    targets = {name: split.targets() for name, split in splits.items()}
    feature_mean = torch.from_numpy(arrays["train"].mean(axis=0, dtype=np.float64).astype(np.float32))
    feature_std = torch.from_numpy(np.maximum(arrays["train"].std(axis=0, dtype=np.float64), 1e-8).astype(np.float32))
    basis = fit_training_basis(targets["train"])
    model = StochasticPriceForecast(feature_mean, feature_std, basis).to(device)
    x = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
    y = {name: torch.from_numpy(value.copy()).to(device) for name, value in targets.items()}
    del arrays
    plan = {"contract": CONTRACT, "dataset": str(args.dataset), "seed": args.seed,
            "maxEpochs": args.epochs, "earlyStoppingPatience": args.patience,
            "batchSize": args.batch_size, "widths": [512] * 4, "selection": "minimum-all-tick-validation-mixed-law-NLL",
            "includesZeroReturns": True, "basisFitSplit": "train", "testUsedForSelection": False,
            "nonPriceRolloutFeatures": "58 observed features frozen; 120 price-return lags updated causally",
            "counts": {k: len(v) for k, v in targets.items()},
            "intervals": {k: {"firstOriginMs": int(v.axis_start_ms + 1000 * v.origins[0]),
                               "lastOriginMs": int(v.axis_start_ms + 1000 * v.origins[-1])} for k, v in splits.items()},
            "datasetManifestSha256": hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()}
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "basis.json", basis)
    optimizer_config = json.loads(Path("ml/training-plans/next-return-production-basis-density-4l-65k-k256-v1.json").read_text())["training"]
    optimizers = build_optimizers(model, optimizer_config, device)
    start_time = time.monotonic()
    best = float("inf")
    best_epoch = -1
    results = []
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            indices = torch.randperm(len(y["train"]), device=device)
            for start in range(0, len(indices), args.batch_size):
                selected = indices[start:start + args.batch_size]
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss = -model.log_prob_from_logits(model(x["train"][selected]), y["train"][selected]).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite training NLL")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                for optimizer in optimizers:
                    optimizer.step()
        validation = evaluate(model, x["validation"], y["validation"], args.batch_size)
        score = validation["negativeLogLikelihood"]
        row = {"epoch": epoch, "seconds": time.monotonic() - start_time, "validation": validation}
        results.append(row)
        if score < best:
            best, best_epoch = score, epoch
            payload = {"contract": CONTRACT, "model": model.state_dict(), "featureMean": feature_mean,
                       "featureStd": feature_std, "basis": basis, "widths": [512] * 4,
                       "selectedEpoch": epoch, "validation": validation, "plan": plan}
            temporary = args.output / "best.pt.tmp"
            torch.save(payload, temporary)
            os.replace(temporary, args.output / "best.pt")
        with (args.output / "training.jsonl").open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({**row, "bestEpoch": best_epoch}), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    payload = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    # Test is loaded only after the complete checkpoint selection has finished.
    test = load_timeline_split(args.dataset, "test")
    test_metrics = evaluate(model, torch.from_numpy(test.features()).to(device),
                            torch.from_numpy(test.targets().copy()).to(device), args.batch_size)
    result = {"contract": CONTRACT, "plan": plan, "selectedEpoch": best_epoch,
              "completedEpoch": results[-1]["epoch"], "seconds": time.monotonic() - start_time,
              "validation": payload["validation"], "test": test_metrics,
              "checkpoint": str(args.output / "best.pt"),
              "checkpointSha256": hashlib.sha256((args.output / "best.pt").read_bytes()).hexdigest(),
              "basisKnotCount": len(basis["knots"]), "trainingValidationOnly": True}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
