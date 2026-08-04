from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from linear_next_second_return import (
    MODEL_CONTRACT,
    StandardizedRegressionStatistics,
    predict_raw,
    raw_coefficients,
    solve_ridge,
    sufficient_normalized_mse,
)
from next_return_dataset import (
    EXAMPLE_SPAN_MS,
    FEATURE_CONTRACT,
    HISTORY_RETURN_COUNT,
    SPLIT_CONTRACT,
    count_examples,
    select_example_shards,
)
from trading_storage import require_under, training_storage_layout
from train_normalized_glu_next_return import (
    MetricAccumulator,
    NextReturnDataset,
    Reporter,
    atomic_json,
    canonical_fingerprint,
    corpus_fingerprint,
    iso_now,
    move_batch,
    training_normalization,
    validate_source_manifest,
)


RUNNER_CONTRACT = "streamed-standardized-sufficient-statistics-fp64-ridge-v1"
SELECTION_CONTRACT = "validation-normalized-mse-ridge-lambda-before-sealed-test-v1"
PARAMETER_COUNT = HISTORY_RETURN_COUNT + 1


def resolve(repo_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def validate_plan(plan: dict) -> None:
    required = (
        "id",
        "sourceDatasetDir",
        "datasetDir",
        "runDir",
        "historyDir",
        "testExamples",
        "expectedCorpusFingerprint",
        "fit",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("linear next-second-return plan is missing required fields")
    fit = plan["fit"]
    if fit.get("contract") != MODEL_CONTRACT:
        raise ValueError("linear model contract changed")
    if fit.get("device") not in {"cpu", "cuda"}:
        raise ValueError("linear fit device must be cpu or cuda")
    for name in ("statisticsBatchSize", "evaluationBatchSize"):
        if int(fit.get(name, 0)) < 1:
            raise ValueError(f"linear fit {name} must be positive")
    lambdas = fit.get("ridgeLambdas")
    if not isinstance(lambdas, list) or not lambdas:
        raise ValueError("linear fit requires at least one ridge lambda")
    values = [float(value) for value in lambdas]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("ridge lambdas must be finite and non-negative")
    if len(set(values)) != len(values):
        raise ValueError("ridge lambdas must be unique")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a linear baseline from 120 completed one-second returns "
            "to the next completed one-second return."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def collect_statistics(
    dataset: NextReturnDataset,
    split: str,
    *,
    batch_size: int,
    normalization,
    device: torch.device,
) -> StandardizedRegressionStatistics:
    statistics = StandardizedRegressionStatistics(device)
    for cpu_batch in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0
    ):
        features, targets, weights = move_batch(cpu_batch, device)
        statistics.add(features, targets, weights, normalization)
    return statistics.cpu()


@torch.no_grad()
def evaluate_linear(
    dataset: NextReturnDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    intercept: float,
    slopes: np.ndarray,
    device: torch.device,
) -> dict[str, float | int | None]:
    metrics = MetricAccumulator(target_std, device)
    slope_tensor = torch.as_tensor(slopes, dtype=torch.float32, device=device)
    for cpu_batch in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0
    ):
        features, targets, weights = move_batch(cpu_batch, device)
        metrics.add(predict_raw(features, intercept, slope_tensor), targets, weights)
    return metrics.result()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo_root, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    plan_fingerprint = canonical_fingerprint(plan)
    layout = training_storage_layout(repo_root)
    source_root = require_under(
        resolve(repo_root, Path(plan["sourceDatasetDir"])),
        layout.datasets,
        "sourceDatasetDir",
    )
    dataset_root = require_under(
        resolve(repo_root, Path(plan["datasetDir"])),
        layout.datasets,
        "datasetDir",
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        layout.runs,
        "runDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    reporter = Reporter(run_dir)
    reporter.status("selecting-examples", planId=plan["id"])
    try:
        source_manifest = json.loads(
            (source_root / "dataset.json").read_text(encoding="utf-8")
        )
        validate_source_manifest(source_manifest)
        shards = select_example_shards(
            source_manifest, test_count=int(plan["testExamples"])
        )
        counts = count_examples(shards)
        fingerprint = corpus_fingerprint(shards)
        if fingerprint != str(plan["expectedCorpusFingerprint"]):
            raise ValueError(
                "selected corpus differs from the comparison corpus: "
                f"{fingerprint}"
            )
        selection_event = {
            "event": "linear-dataset-selected",
            "planId": plan["id"],
            "corpusFingerprint": fingerprint,
            "counts": counts,
            "crossSplitPurgeMs": EXAMPLE_SPAN_MS,
            "modelContract": MODEL_CONTRACT,
            "parameters": PARAMETER_COUNT,
            "testPolicy": "sealed-until-ridge-lambda-selected",
        }
        reporter.emit(selection_event)
        if args.validate_only:
            reporter.status("paused", latest=selection_event)
            return

        snapshot = {"planSha256": plan_fingerprint, "plan": plan}
        snapshot_file = run_dir / "state" / "plan.json"
        if snapshot_file.is_file():
            if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("run directory belongs to a different plan")
        else:
            atomic_json(snapshot, snapshot_file)

        fit = plan["fit"]
        device = torch.device(str(fit["device"]))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA fitting was requested but is unavailable")
        # The normal equations use full IEEE FP32 matrix products and accumulate
        # their small 121x121 reductions in FP64.
        torch.set_float32_matmul_precision("highest")
        statistics_batch_size = int(fit["statisticsBatchSize"])
        evaluation_batch_size = int(fit["evaluationBatchSize"])
        dataset = NextReturnDataset(shards, history_root)

        reporter.status("computing-training-normalization", planId=plan["id"])
        normalization = training_normalization(
            dataset, batch_size=statistics_batch_size
        )
        dataset_manifest = {
            "version": 1,
            "createdAt": iso_now(),
            "planId": plan["id"],
            "sourceDataset": str(
                (source_root / "dataset.json").relative_to(repo_root)
            ),
            "featureContract": FEATURE_CONTRACT,
            "splitContract": SPLIT_CONTRACT,
            "corpusFingerprint": fingerprint,
            "input": "120 adjacent completed close-to-close one-second log returns",
            "target": "the immediately following completed one-second log return",
            "normalization": {
                "source": "training split only",
                "featureMean": normalization.feature_mean.tolist(),
                "featureStd": normalization.feature_std.tolist(),
                "targetMean": normalization.target_mean,
                "targetStd": normalization.target_std,
            },
            "crossSplitPurgeMs": EXAMPLE_SPAN_MS,
            "counts": counts,
            "storage": "streamed strided daily return windows; no row compaction",
        }
        atomic_json(dataset_manifest, dataset_root / "dataset.json")

        reporter.status("collecting-train-sufficient-statistics", planId=plan["id"])
        train_statistics = collect_statistics(
            dataset,
            "train",
            batch_size=statistics_batch_size,
            normalization=normalization,
            device=device,
        )
        reporter.status(
            "collecting-validation-sufficient-statistics", planId=plan["id"]
        )
        validation_statistics = collect_statistics(
            dataset,
            "validation",
            batch_size=statistics_batch_size,
            normalization=normalization,
            device=device,
        )

        candidates = []
        best_coefficients: np.ndarray | None = None
        best_lambda: float | None = None
        best_validation_score = math.inf
        for ridge_lambda in (float(value) for value in fit["ridgeLambdas"]):
            coefficients = solve_ridge(train_statistics, ridge_lambda)
            train_score = sufficient_normalized_mse(train_statistics, coefficients)
            validation_score = sufficient_normalized_mse(
                validation_statistics, coefficients
            )
            candidate = {
                "ridgeLambda": ridge_lambda,
                "trainNormalizedMse": train_score,
                "validationNormalizedMse": validation_score,
            }
            candidates.append(candidate)
            reporter.emit({"event": "linear-candidate", **candidate})
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                best_lambda = ridge_lambda
                best_coefficients = coefficients
        if best_coefficients is None or best_lambda is None:
            raise RuntimeError("ridge selection produced no model")

        intercept, slopes = raw_coefficients(best_coefficients, normalization)
        selection = {
            "ridgeLambda": best_lambda,
            "validationNormalizedMse": best_validation_score,
            "selectionContract": SELECTION_CONTRACT,
            "testEvaluated": False,
        }
        reporter.emit({"event": "linear-selected", **selection})
        reporter.status("evaluating-selected-model", latest=selection)

        validation_metrics = evaluate_linear(
            dataset,
            "validation",
            batch_size=evaluation_batch_size,
            target_std=normalization.target_std,
            intercept=intercept,
            slopes=slopes,
            device=device,
        )
        # The test tail is opened exactly once, after lambda selection is frozen.
        test_metrics = evaluate_linear(
            dataset,
            "test",
            batch_size=evaluation_batch_size,
            target_std=normalization.target_std,
            intercept=intercept,
            slopes=slopes,
            device=device,
        )
        model_artifact = {
            "version": 1,
            "createdAt": iso_now(),
            "modelContract": MODEL_CONTRACT,
            "featureContract": FEATURE_CONTRACT,
            "parameterCount": PARAMETER_COUNT,
            "corpusFingerprint": fingerprint,
            "ridgeLambda": best_lambda,
            "standardizedCoefficients": best_coefficients.tolist(),
            "rawIntercept": intercept,
            "rawLagCoefficientsOldestToNewest": slopes.tolist(),
            "normalization": dataset_manifest["normalization"],
        }
        model_file = run_dir / "model.json"
        atomic_json(model_artifact, model_file)
        result = {
            "completedAt": iso_now(),
            "planId": plan["id"],
            "planSha256": plan_fingerprint,
            "corpusFingerprint": fingerprint,
            "modelContract": MODEL_CONTRACT,
            "runnerContract": RUNNER_CONTRACT,
            "selectionContract": SELECTION_CONTRACT,
            "parameterCount": PARAMETER_COUNT,
            "ridgeLambda": best_lambda,
            "candidates": candidates,
            "validation": validation_metrics,
            "test": test_metrics,
            "model": str(model_file.relative_to(repo_root)),
            "testEvaluated": True,
        }
        atomic_json(result, run_dir / "state" / "result.json")
        reporter.emit({"event": "linear-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except KeyboardInterrupt:
        reporter.status("paused", planId=plan["id"], message="Interrupted")
        raise
    except Exception as error:
        reporter.status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
