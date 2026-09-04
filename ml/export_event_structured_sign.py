"""Export a frozen historical production59 predictor at every retained second.

The training population excludes flat targets. This export restores them before
calibration so a direction score is never treated as a calibrated return law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_stopped_structured_feature_process import model_from_state
from structured_union530_base import StructuredProduction59BaseDataset
from trading_storage import load_torch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    plan = json.loads(args.plan.read_text())
    run = Path(plan["runDir"])
    assert json.loads((run / "state/status.json").read_text())["stage"] == "complete"
    pointer = run / "checkpoints/selections/validation-mse.json"
    checkpoint = load_torch_checkpoint(pointer, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = model_from_state(plan, checkpoint["model"], device)
    dataset = StructuredProduction59BaseDataset(
        Path(plan["datasetDir"]), Path(plan["baseHistoryDir"]),
        train_examples=plan["subset"]["examples"],
    )
    summary = {"contract": "event-structured-sign-dense-export-v1",
               "plan": str(args.plan), "checkpoint": str(pointer),
               "checkpointSha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
               "selectedEpoch": int(checkpoint["epoch"]), "splits": {}}
    started = time.perf_counter()
    exported = {}
    with torch.inference_mode():
        for split in ("validation", "test"):
            # Retain the scored interval of the original active-target sample,
            # including every intervening flat second. No new dates are added.
            original = dataset.origins[split]
            dataset.origins[split] = np.arange(original[0], original[-1] + 1)
            dataset.counts[split] = len(dataset.origins[split])
            predictions, targets, primitive = [], [], []
            for batch in dataset.iter_batches(split, 2048, shuffle=False, seed=0, pad=False):
                inputs = batch.inputs.to(device)
                context = {k: v.to(device) for k, v in batch.context.items() if k != "baseTargets"}
                raw = model.raw_outputs(model(inputs))
                derived = dataset.derive(raw, inputs, context)
                predictions.append(derived[:, :, 0].cpu().numpy())
                targets.append(batch.targets[:, :, 0].numpy())
                primitive.append(raw.cpu().numpy())
            prediction, target, raw = map(np.concatenate, (predictions, targets, primitive))
            times = dataset.base.second_start_ms + (
                dataset.second_offsets[split] + dataset.origins[split] + 1
            ) * 1000
            assert np.isfinite(prediction).all() and np.isfinite(raw).all()
            assert np.all(np.diff(times) == 1000)
            np.savez(args.output / f"{split}.npz", times=times,
                     predictionLogReturns=prediction, actualLogReturns=target,
                     predictedPrimitives=raw)
            active = target[:, 0] != 0
            correct = np.sign(prediction[:, 0]) == np.sign(target[:, 0])
            weights = np.abs(target[:, 0])
            summary["splits"][split] = {
                "rows": len(times), "start": int(times[0]), "end": int(times[-1]),
                "activeRows": int(active.sum()), "activeDirectionAccuracy": float(correct[active].mean()),
                "magnitudeWeightedDirectionAccuracy": float(np.dot(correct, weights) / weights.sum()),
                "maximumAbsolutePredictedReturnBps": float(np.abs(np.expm1(prediction[:, 0])).max() * 10000),
                "absolutePredictedReturnBpsQuantiles": np.quantile(np.abs(np.expm1(prediction[:, 0])) * 10000, [.5, .9, .99, 1]).tolist(),
            }
            exported[split] = prediction[:, 0], np.expm1(target[:, 0]).astype(np.float64)
    # Sixteen fixed quantile buckets use the earlier calibration population.
    # Their joint down/flat/up frequencies and return sizes stay together.
    edges = np.unique(np.quantile(exported["validation"][0], np.arange(1, 16) / 16))
    bins = {}
    for split, (score, returns) in exported.items():
        groups = np.searchsorted(edges, score)
        bins[split] = []
        for index in range(len(edges) + 1):
            values = returns[groups == index]
            bins[split].append({"index": index, "rows": len(values),
                "meanReturnBps": float(values.mean() * 10000) if len(values) else None,
                "probabilitiesDownFlatUp": [float((np.sign(values) == sign).mean()) for sign in (-1, 0, 1)] if len(values) else None})
    summary["calibration"] = {"method": "16 score-quantile buckets; edges and law fit on validation only",
        "edgesLogReturn": edges.tolist(), "buckets": bins,
        "maximumAbsoluteBucketMeanBps": max(abs(row["meanReturnBps"]) for row in bins["validation"]),
        "limitation": "Empirical point estimates, not uncertainty bounds; no policy or horizon claim."}
    summary["elapsedSeconds"] = time.perf_counter() - started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "source.py").write_text(Path(__file__).read_text())
    dataset.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration"}))


if __name__ == "__main__":
    main()
