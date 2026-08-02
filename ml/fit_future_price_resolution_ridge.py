from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

# Closed-form accumulation and solves are intentionally single-threaded so a
# rerun with the pinned plan/runtime has a stable reduction order.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import torch

from future_price_resolution_ridge import (
    FeatureScaler,
    RIDGE_FEATURE_CONTRACT,
    RIDGE_FEATURE_NAMES,
    RIDGE_FIT_CONTRACT,
    WeightedRegressionStatistics,
    audited_ridge_features,
    predict_standardized_ridge,
    raw_feature_coefficients,
    raw_history_coefficients,
    solve_standardized_ridge,
    standardize_statistics,
    sufficient_mse,
    validate_ridge_spec,
)
from future_price_resolution_screen import ResolutionNormalization, ResolutionSpec
from train_future_price_feature_screen import prepare_base_components
from train_future_price_predictor import (
    HOUR_MS,
    MINUTE_MS,
    SECOND_MS,
    JsonReporter,
    PairShard,
    atomic_json,
    compact_pair_rows,
    corpus_fingerprint,
    count_examples,
    validate_predictor_split_disjointness,
    validate_source_manifest,
)
from train_future_price_resolution_screen import (
    RESOLUTION_CORPUS_CONTRACT,
    ResolutionDataset,
    ResolutionMetricAccumulator,
    _previous_date,
    compute_training_normalization,
    compute_validation_baselines,
    normalization_fingerprint,
    resolution_data_fingerprint,
    select_resolution_segments,
)
from trading_storage import (
    load_torch_checkpoint,
    require_under,
    training_storage_layout,
    write_shard_payload,
)


RIDGE_ARTIFACT_SCHEMA = "future-price-resolution-ridge-model-v3"
RIDGE_STATISTICS_SCHEMA = "future-price-resolution-ridge-statistics-v2"


def _slice_shard(shard: PairShard, start: int, count: int) -> PairShard:
    if start < 0 or count < 1 or start + count > shard.count:
        raise ValueError("internal ridge shard slice is invalid")
    return PairShard(
        split=shard.split,
        prediction_time_start=shard.prediction_time_start + start * SECOND_MS,
        count=count,
        future_date=shard.future_date,
        history_date=shard.history_date,
        future_row_offset=shard.future_row_offset + start,
        history_row_offset=shard.history_row_offset + start,
    )


def split_internal_training(
    train_segments: Iterable[PairShard],
    *,
    fit_fraction: float,
    example_span_ms: int,
) -> dict[str, list[PairShard]]:
    """Chronologically split training with a full example-span embargo."""
    if not 0.5 <= fit_fraction <= 0.95:
        raise ValueError("ridge fit fraction must be within [0.5, 0.95]")
    ordered = sorted(train_segments, key=lambda value: value.prediction_time_start)
    total = sum(shard.count for shard in ordered)
    fit_target = int(math.floor(total * fit_fraction))
    if fit_target < 1 or fit_target >= total:
        raise ValueError("ridge internal split cannot be empty")
    fit: list[PairShard] = []
    candidates: list[PairShard] = []
    seen = 0
    for shard in ordered:
        remaining = fit_target - seen
        if remaining <= 0:
            candidates.append(shard)
            continue
        if remaining >= shard.count:
            fit.append(shard)
            seen += shard.count
            continue
        fit.append(_slice_shard(shard, 0, remaining))
        seen += remaining
        if remaining < shard.count:
            candidates.append(_slice_shard(
                shard,
                remaining,
                shard.count - remaining,
            ))
    if seen != fit_target or not fit or not candidates:
        raise RuntimeError("ridge internal fit allocation changed")
    earliest_calibration = (
        max(shard.prediction_time_end for shard in fit)
        + example_span_ms
        + SECOND_MS
    )
    calibration: list[PairShard] = []
    for shard in candidates:
        local_offset = max(0, math.ceil(
            (earliest_calibration - shard.prediction_time_start) / SECOND_MS
        ))
        if local_offset < shard.count:
            calibration.append(
                shard.shifted(local_offset) if local_offset else shard
            )
    if not calibration:
        raise ValueError("ridge internal embargo removed all calibration rows")
    internal = {"train": fit, "validation": calibration}
    validate_predictor_split_disjointness(
        internal,
        example_span_ms=example_span_ms,
    )
    return {"fit": fit, "calibration": calibration}


def _compact_count(segments: Iterable[PairShard]) -> int:
    return sum(
        compact_pair_rows(
            shard.history_row_offset,
            shard.future_row_offset,
            shard.count,
        )[2].shape[0]
        for shard in segments
    )


def _statistics_for(
    dataset: ResolutionDataset,
    split: str,
    *,
    batch_size: int,
) -> WeightedRegressionStatistics:
    statistics = WeightedRegressionStatistics()
    for history, target, weights in dataset.iter_batches(
        split,
        batch_size,
        shuffle=False,
        seed=0,
    ):
        statistics.add(
            audited_ridge_features(history.numpy()),
            target.numpy()[:, 0],
            weights.numpy(),
        )
    expected = dataset.logical_count(split)
    if int(round(statistics.weight)) != expected:
        raise RuntimeError(
            f"ridge {split} statistics saw {statistics.weight} != {expected}"
        )
    return statistics


def _statistics_content_fingerprint(
    statistics: dict[str, WeightedRegressionStatistics],
) -> str:
    digest = hashlib.sha256()
    digest.update(RIDGE_STATISTICS_SCHEMA.encode("utf-8"))
    for name in sorted(statistics):
        digest.update(name.encode("utf-8"))
        for key, value in sorted(statistics[name].to_arrays(name).items()):
            array = np.asarray(value, dtype="<f8")
            digest.update(key.encode("utf-8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _load_or_compute_statistics(
    cache_file: Path,
    *,
    fingerprint: str,
    full_dataset: ResolutionDataset,
    internal_dataset: ResolutionDataset,
    batch_size: int,
    reporter: JsonReporter,
) -> dict[str, WeightedRegressionStatistics]:
    expected = {
        "full": full_dataset.logical_count("train"),
        "fit": internal_dataset.logical_count("fit"),
        "calibration": internal_dataset.logical_count("calibration"),
    }
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cached:
            if str(cached["schema"]) != RIDGE_STATISTICS_SCHEMA \
                    or str(cached["fingerprint"]) != fingerprint:
                raise ValueError("ridge sufficient-statistics cache is stale")
            values = {name: cached[name] for name in cached.files}
        result = {
            name: WeightedRegressionStatistics.from_arrays(values, name)
            for name in expected
        }
        if any(int(round(result[name].weight)) != count for name, count in expected.items()):
            raise ValueError("ridge sufficient-statistics cache count changed")
        content_fingerprint = _statistics_content_fingerprint(result)
        if str(values.get("contentFingerprint", "")) != content_fingerprint:
            raise ValueError("ridge sufficient-statistics cache content changed")
        reporter.emit({
            "event": "ridge-statistics-cache",
            "hit": True,
            "fingerprint": fingerprint,
            "contentFingerprint": content_fingerprint,
            "counts": expected,
            "file": str(cache_file),
        })
        return result
    result = {
        "full": _statistics_for(full_dataset, "train", batch_size=batch_size),
        "fit": _statistics_for(internal_dataset, "fit", batch_size=batch_size),
        "calibration": _statistics_for(
            internal_dataset,
            "calibration",
            batch_size=batch_size,
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(RIDGE_STATISTICS_SCHEMA),
        "fingerprint": np.asarray(fingerprint),
    }
    for name, statistics in result.items():
        arrays.update(statistics.to_arrays(name))
    content_fingerprint = _statistics_content_fingerprint(result)
    arrays["contentFingerprint"] = np.asarray(content_fingerprint)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **arrays)
    os.replace(temporary, cache_file)
    reporter.emit({
        "event": "ridge-statistics-cache",
        "hit": False,
        "fingerprint": fingerprint,
        "contentFingerprint": content_fingerprint,
        "counts": expected,
        "file": str(cache_file),
    })
    return result


def _evaluate_coefficients(
    dataset: ResolutionDataset,
    normalization: ResolutionNormalization,
    *,
    scaler: FeatureScaler,
    coefficients: np.ndarray,
    batch_size: int,
    huber_delta: float,
) -> dict:
    accumulator = ResolutionMetricAccumulator(
        normalization,
        huber_delta=huber_delta,
        target_steps=1,
        device=torch.device("cpu"),
    )
    for history, target, weights in dataset.iter_batches(
        "validation",
        batch_size,
        shuffle=False,
        seed=0,
    ):
        prediction = predict_standardized_ridge(
            history.numpy(),
            scaler,
            coefficients,
        )
        accumulator.add(
            torch.from_numpy(prediction[:, None]),
            target,
            weights,
        )
    result = accumulator.result()
    result["loss"] = result["normalizedMse"]
    return result


def _comparison(ridge: dict, patch: dict) -> dict:
    metrics = (
        "normalizedMse",
        "rawMse",
        "rawMae",
        "directionAccuracy",
        "endpointCorrelation",
        "endpointPredictionScaleRatio",
    )
    return {
        name: {
            "ridge": float(ridge[name]),
            "patchTcn": float(patch[name]),
            "ridgeMinusPatchTcn": float(ridge[name]) - float(patch[name]),
            "ridgeOverPatchTcn": (
                float(ridge[name]) / float(patch[name])
                if float(patch[name]) != 0
                else None
            ),
        }
        for name in metrics
    }


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_fingerprint(files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def validate_plan(plan: dict) -> ResolutionSpec:
    required = (
        "id",
        "corpusId",
        "baseComponentCorpusId",
        "sourceDatasetDir",
        "decoderDatasetDir",
        "datasetDir",
        "runDir",
        "historyDir",
        "resolution",
        "featureContract",
        "internalValidation",
        "ridgeLambdas",
        "batchSize",
        "huberDelta",
        "artifact",
        "referencePatchTcn",
        "expectedDatasetFingerprint",
        "ridgeLimit",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("ridge plan is missing required fields")
    spec = ResolutionSpec.from_config(plan["resolution"])
    validate_ridge_spec(spec)
    if plan["featureContract"] != RIDGE_FEATURE_CONTRACT:
        raise ValueError("ridge feature contract changed")
    internal = plan["internalValidation"]
    if internal.get("method") != "chronological-tail-with-example-span-embargo" \
            or not 0.5 <= float(internal.get("fitFraction", 0)) <= 0.95:
        raise ValueError("ridge internal validation contract is invalid")
    lambdas = tuple(float(value) for value in plan["ridgeLambdas"])
    if not lambdas \
            or any(not math.isfinite(value) or value < 0 for value in lambdas) \
            or tuple(sorted(set(lambdas))) != lambdas:
        raise ValueError("ridge lambda grid must be sorted and unique")
    if plan["ridgeLimit"] != "training-mean":
        raise ValueError("ridge plan must explicitly include its mean limit")
    if int(plan["batchSize"]) < 1 or float(plan["huberDelta"]) <= 0:
        raise ValueError("ridge fit dimensions are invalid")
    artifact = plan["artifact"]
    if set(artifact) != {"namespace", "key"}:
        raise ValueError("ridge artifact destination is invalid")
    reference = plan["referencePatchTcn"]
    if int(reference.get("epoch", -1)) < 1 \
            or len(str(reference.get("contentHash", ""))) != 64 \
            or not str(reference.get("checkpoint", "")):
        raise ValueError("ridge Patch-TCN reference is invalid")
    return spec


def resolve(repo_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an immutable closed-form 6h/5m -> next-5m ridge model."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo_root, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    spec = validate_plan(plan)
    layout = training_storage_layout(repo_root)
    source_root = require_under(
        resolve(repo_root, Path(plan["sourceDatasetDir"])),
        layout.datasets,
        "sourceDatasetDir",
    )
    decoder_root = require_under(
        resolve(repo_root, Path(plan["decoderDatasetDir"])),
        layout.datasets,
        "decoderDatasetDir",
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
    reporter = JsonReporter(run_dir)
    try:
        source_manifest = json.loads(
            (source_root / "dataset.json").read_text(encoding="utf-8")
        )
        validate_source_manifest(source_manifest)
        decoder_manifest = json.loads(
            (decoder_root / "dataset.json").read_text(encoding="utf-8")
        )
        if int(decoder_manifest.get("crossSplitPurgeMs", 0)) != HOUR_MS:
            raise ValueError("reference decoder corpus contract changed")
        segments = select_resolution_segments(source_manifest, spec)
        example_span_ms = spec.example_span_minutes * MINUTE_MS
        internal = split_internal_training(
            segments["train"],
            fit_fraction=float(plan["internalValidation"]["fitFraction"]),
            example_span_ms=example_span_ms,
        )
        timestamp_fingerprint = corpus_fingerprint(
            segments,
            contract=f"{RESOLUTION_CORPUS_CONTRACT}:{spec.contract}",
        )
        data_fingerprint = resolution_data_fingerprint(
            timestamp_fingerprint,
            spec,
        )
        if data_fingerprint != plan["expectedDatasetFingerprint"]:
            raise ValueError("pinned ridge dataset fingerprint changed")
        internal_segments = {
            "train": internal["fit"],
            "validation": internal["calibration"],
        }
        internal_fingerprint = corpus_fingerprint(
            internal_segments,
            contract=f"{RIDGE_FIT_CONTRACT}:{spec.contract}",
        )
        statistics_fingerprint = hashlib.sha256(_canonical_bytes({
            "schema": RIDGE_STATISTICS_SCHEMA,
            "datasetFingerprint": data_fingerprint,
            "internalFingerprint": internal_fingerprint,
            "featureContract": RIDGE_FEATURE_CONTRACT,
            "batchSize": int(plan["batchSize"]),
        })).hexdigest()
        validation_event = {
            "event": "ridge-validation-complete",
            "planId": plan["id"],
            "datasetFingerprint": data_fingerprint,
            "internalSplitFingerprint": internal_fingerprint,
            "counts": count_examples(segments),
            "internalCounts": {
                "fit": sum(value.count for value in internal["fit"]),
                "calibration": sum(
                    value.count for value in internal["calibration"]
                ),
            },
            "internalCompactCounts": {
                name: _compact_count(values)
                for name, values in internal.items()
            },
            "crossSplitPurgeMs": example_span_ms,
            "internalPurgeMs": example_span_ms,
            "featureContract": RIDGE_FEATURE_CONTRACT,
            "fitContract": RIDGE_FIT_CONTRACT,
            "heldoutTest": "not selected or read",
            "gpuUsed": False,
        }
        reporter.emit(validation_event)
        if args.validate_only:
            reporter.status(
                "paused",
                planId=plan["id"],
                latest=validation_event,
                message="Ridge plan validated without reading candle payloads.",
            )
            return

        component_dates = {
            component_date
            for values in segments.values()
            for shard in values
            for component_date in (
                shard.history_date,
                _previous_date(shard.history_date),
                shard.future_date,
                _previous_date(shard.future_date),
            )
        }
        component_root = (
            layout.immutable
            / "refs"
            / "features"
            / "future-price-predictor-minute-base-v1"
            / plan["baseComponentCorpusId"]
        )
        component_files = prepare_base_components(
            component_dates,
            component_root=component_root,
            history_root=history_root,
            immutable_root=layout.immutable,
            corpus_id=plan["baseComponentCorpusId"],
            reporter=reporter,
        )
        full_dataset = ResolutionDataset(segments, component_files, spec)
        internal_dataset = ResolutionDataset({
            "fit": internal["fit"],
            "calibration": internal["calibration"],
        }, component_files, spec)
        normalization = compute_training_normalization(
            full_dataset,
            dataset_root / "training-resolution-statistics-v1.npz",
            fingerprint=data_fingerprint,
            batch_size=int(plan["batchSize"]),
            reporter=reporter,
        )
        baselines = compute_validation_baselines(
            full_dataset,
            normalization,
            dataset_root / "validation-resolution-baselines-v2.json",
            data_fingerprint=data_fingerprint,
            batch_size=int(plan["batchSize"]),
            objective="mse",
            huber_delta=float(plan["huberDelta"]),
            reporter=reporter,
        )
        statistics = _load_or_compute_statistics(
            run_dir / "state" / "ridge-sufficient-statistics-v2.npz",
            fingerprint=statistics_fingerprint,
            full_dataset=full_dataset,
            internal_dataset=internal_dataset,
            batch_size=int(plan["batchSize"]),
            reporter=reporter,
        )
        fit_scaler = FeatureScaler.from_statistics(statistics["fit"])
        fit_standardized = standardize_statistics(statistics["fit"], fit_scaler)
        calibration_standardized = standardize_statistics(
            statistics["calibration"],
            fit_scaler,
        )
        target_std = float(normalization.target_std[0])
        calibration_grid = []
        selected: tuple[float, int, str, float | None, np.ndarray] | None = None
        for candidate_index, ridge_lambda in enumerate(
            float(value) for value in plan["ridgeLambdas"]
        ):
            candidate = solve_standardized_ridge(
                fit_standardized,
                ridge_lambda,
            )
            raw_mse = sufficient_mse(calibration_standardized, candidate)
            normalized_mse = raw_mse / (target_std * target_std)
            calibration_grid.append({
                "model": "finite-ridge",
                "lambda": ridge_lambda,
                "rawMse": raw_mse,
                "normalizedMse": normalized_mse,
            })
            score = (
                normalized_mse,
                candidate_index,
                "finite-ridge",
                ridge_lambda,
                candidate,
            )
            if selected is None or score[:2] < selected[:2]:
                selected = score
        mean_limit = np.zeros_like(fit_standardized.xty)
        mean_limit[0] = fit_standardized.y_sum / fit_standardized.weight
        mean_raw_mse = sufficient_mse(calibration_standardized, mean_limit)
        mean_normalized_mse = mean_raw_mse / (target_std * target_std)
        calibration_grid.append({
            "model": "training-mean-ridge-limit",
            "lambda": "infinity",
            "rawMse": mean_raw_mse,
            "normalizedMse": mean_normalized_mse,
        })
        mean_score = (
            mean_normalized_mse,
            len(calibration_grid) - 1,
            "training-mean-ridge-limit",
            None,
            mean_limit,
        )
        if selected is None or mean_score[:2] < selected[:2]:
            selected = mean_score
        assert selected is not None
        selected_model = selected[2]
        selected_lambda = selected[3]
        selected_lambda_value: float | str = (
            selected_lambda if selected_lambda is not None else "infinity"
        )
        final_scaler = FeatureScaler.from_statistics(statistics["full"])
        full_standardized = standardize_statistics(statistics["full"], final_scaler)
        ols_coefficients = solve_standardized_ridge(full_standardized, 0.0)
        if selected_model == "training-mean-ridge-limit":
            ridge_coefficients = np.zeros_like(full_standardized.xty)
            ridge_coefficients[0] = (
                full_standardized.y_sum / full_standardized.weight
            )
        else:
            assert selected_lambda is not None
            ridge_coefficients = solve_standardized_ridge(
                full_standardized,
                selected_lambda,
            )
        ols_metrics = _evaluate_coefficients(
            full_dataset,
            normalization,
            scaler=final_scaler,
            coefficients=ols_coefficients,
            batch_size=int(plan["batchSize"]),
            huber_delta=float(plan["huberDelta"]),
        )
        ridge_metrics = _evaluate_coefficients(
            full_dataset,
            normalization,
            scaler=final_scaler,
            coefficients=ridge_coefficients,
            batch_size=int(plan["batchSize"]),
            huber_delta=float(plan["huberDelta"]),
        )
        reference_config = plan["referencePatchTcn"]
        reference_file = require_under(
            resolve(repo_root, Path(reference_config["checkpoint"])),
            layout.runs,
            "referencePatchTcn.checkpoint",
        )
        reference_pointer = json.loads(reference_file.read_text(encoding="utf-8"))
        reference_hash = str(reference_pointer.get("object", {}).get("contentHash", ""))
        if reference_hash != reference_config["contentHash"]:
            raise ValueError("pinned Patch-TCN checkpoint object changed")
        patch_checkpoint = load_torch_checkpoint(
            reference_file,
            map_location="cpu",
            weights_only=False,
        )
        if int(patch_checkpoint.get("epoch", -1)) != int(reference_config["epoch"]) \
                or patch_checkpoint.get("datasetFingerprint") != data_fingerprint \
                or patch_checkpoint.get("resolutionContract") != spec.contract:
            raise ValueError("pinned Patch-TCN comparison contract changed")
        patch_metrics = patch_checkpoint["validation"]
        if abs(
            float(patch_metrics["normalizedMse"])
            - float(reference_config["normalizedMse"])
        ) > 1e-15:
            raise ValueError("pinned Patch-TCN metric changed")
        snapshot_file = run_dir / "checkpoints" / "patch-tcn-epoch11-reference.json"
        if snapshot_file.is_file():
            existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
            if existing.get("object", {}).get("contentHash") != reference_hash:
                raise ValueError("immutable Patch-TCN comparison snapshot changed")
        else:
            atomic_json(reference_pointer, snapshot_file)

        ridge_raw_intercept, ridge_raw_slopes = raw_feature_coefficients(
            final_scaler,
            ridge_coefficients,
        )
        ols_raw_intercept, ols_raw_slopes = raw_feature_coefficients(
            final_scaler,
            ols_coefficients,
        )
        plan_fingerprint = hashlib.sha256(_canonical_bytes(plan)).hexdigest()
        implementation_files = tuple(
            Path(__file__).with_name(name)
            for name in (
                "fit_future_price_resolution_ridge.py",
                "future_price_resolution_ridge.py",
                "future_price_resolution_screen.py",
                "train_future_price_resolution_screen.py",
                "train_future_price_feature_screen.py",
                "train_future_price_predictor.py",
                "trading_storage.py",
            )
        )
        implementation_fingerprint = _file_fingerprint(implementation_files)
        implementation_file_hashes = {
            file.relative_to(repo_root).as_posix(): hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in implementation_files
        }
        statistics_content_fingerprint = _statistics_content_fingerprint(
            statistics
        )
        artifact = {
            "schemaVersion": 1,
            "schema": RIDGE_ARTIFACT_SCHEMA,
            "planId": plan["id"],
            "planFingerprint": plan_fingerprint,
            "implementationFingerprint": implementation_fingerprint,
            "implementationFiles": implementation_file_hashes,
            "dataset": {
                "corpusId": plan["corpusId"],
                "datasetFingerprint": data_fingerprint,
                "timestampCorpusFingerprint": timestamp_fingerprint,
                "resolutionContract": spec.contract,
                "crossSplitPurgeMs": example_span_ms,
                "trainLogicalExamples": full_dataset.logical_count("train"),
                "trainCompactExamples": full_dataset.compact_count("train"),
                "validationLogicalExamples": full_dataset.logical_count(
                    "validation"
                ),
                "validationCompactExamples": full_dataset.compact_count(
                    "validation"
                ),
                "normalizationFingerprint": normalization_fingerprint(
                    normalization
                ),
            },
            "accessContract": {
                "testExamplesSelected": 0,
                "testPayloadsRead": 0,
                "gpuUsed": False,
            },
            "features": {
                "contract": RIDGE_FEATURE_CONTRACT,
                "names": list(RIDGE_FEATURE_NAMES),
                "historyMinutes": spec.history_minutes,
                "candleMinutes": spec.candle_minutes,
                "target": "immediate next completed 5m close log return",
            },
            "fit": {
                "contract": RIDGE_FIT_CONTRACT,
                "weighting": "original one-second multiplicity",
                "arithmetic": "numpy-float64-closed-form-sufficient-statistics",
                "ridgeObjective": (
                    "weightedMeanSquaredError + lambda * squared standardized "
                    "non-intercept coefficients"
                ),
                "internalSplitFingerprint": internal_fingerprint,
                "internalFitFractionRequested": float(
                    plan["internalValidation"]["fitFraction"]
                ),
                "internalFitLogicalExamples": internal_dataset.logical_count("fit"),
                "internalFitCompactExamples": internal_dataset.compact_count("fit"),
                "internalCalibrationLogicalExamples": (
                    internal_dataset.logical_count("calibration")
                ),
                "internalCalibrationCompactExamples": (
                    internal_dataset.compact_count("calibration")
                ),
                "internalPurgeMs": example_span_ms,
                "statisticsFingerprint": statistics_fingerprint,
                "statisticsContentFingerprint": (
                    statistics_content_fingerprint
                ),
                "lambdaCalibration": calibration_grid,
                "selectedModel": selected_model,
                "selectedLambda": selected_lambda_value,
                "refit": (
                    "selected finite ridge or explicit mean limit refit on the "
                    "complete training split"
                ),
            },
            "scalers": {
                "source": "complete training split only after lambda selection",
                "featureMean": final_scaler.mean.tolist(),
                "featureStd": final_scaler.std.tolist(),
                "targetMean": float(normalization.target_mean[0]),
                "targetStd": target_std,
            },
            "models": {
                "ridge": {
                    "selectionStatus": "selected-by-train-internal-calibration",
                    "eligibleForDownstreamSelection": True,
                    "standardizedFeatureIntercept": float(ridge_coefficients[0]),
                    "standardizedFeatureCoefficients": (
                        ridge_coefficients[1:].tolist()
                    ),
                    "rawFeatureIntercept": ridge_raw_intercept,
                    "rawFeatureCoefficients": ridge_raw_slopes.tolist(),
                    "rawHistoryCoefficients": raw_history_coefficients(
                        ridge_raw_slopes
                    ).tolist(),
                    "validation": ridge_metrics,
                },
                "ols": {
                    "selectionStatus": "diagnostic-unselected",
                    "eligibleForDownstreamSelection": False,
                    "selectionWarning": (
                        "External validation is comparison-only and must not be "
                        "used to override the train-internal ridge selection."
                    ),
                    "standardizedFeatureIntercept": float(ols_coefficients[0]),
                    "standardizedFeatureCoefficients": (
                        ols_coefficients[1:].tolist()
                    ),
                    "rawFeatureIntercept": ols_raw_intercept,
                    "rawFeatureCoefficients": ols_raw_slopes.tolist(),
                    "rawHistoryCoefficients": raw_history_coefficients(
                        ols_raw_slopes
                    ).tolist(),
                    "validation": ols_metrics,
                },
            },
            "validationBaselines": baselines,
            "referencePatchTcn": {
                "planId": reference_config["planId"],
                "epoch": int(reference_config["epoch"]),
                "checkpointContentHash": reference_hash,
                "validation": patch_metrics,
            },
            "ridgeVsPatchTcn": _comparison(ridge_metrics, patch_metrics),
            "runtime": {
                "numpyVersion": str(np.__version__),
                "torchVersion": str(torch.__version__),
                "blasThreads": 1,
            },
        }
        payload = _canonical_bytes(artifact)
        artifact_reference = write_shard_payload(
            layout.immutable,
            plan["artifact"]["namespace"],
            plan["artifact"]["key"],
            payload,
            sequence={"start": 0, "step": 1, "count": 1, "unit": "index"},
            layout={
                "encoding": "canonical-json-utf8-v1",
                "schema": RIDGE_ARTIFACT_SCHEMA,
            },
            metadata={
                "planId": plan["id"],
                "datasetFingerprint": data_fingerprint,
                "selectedModel": selected_model,
                "selectedLambda": selected_lambda_value,
                "validationNormalizedMse": ridge_metrics["normalizedMse"],
            },
        )
        artifact_pointer = json.loads(
            artifact_reference.read_text(encoding="utf-8")
        )
        result = {
            "version": 1,
            "planId": plan["id"],
            "artifactReference": str(artifact_reference.relative_to(repo_root)),
            "artifactContentHash": artifact_pointer["object"]["contentHash"],
            "datasetFingerprint": data_fingerprint,
            "selectedModel": selected_model,
            "selectedLambda": selected_lambda_value,
            "ridgeValidation": ridge_metrics,
            "olsValidation": ols_metrics,
            "validationBaselines": baselines,
            "referencePatchTcn": artifact["referencePatchTcn"],
            "ridgeVsPatchTcn": artifact["ridgeVsPatchTcn"],
            "accessContract": artifact["accessContract"],
        }
        atomic_json(result, run_dir / "result.json")
        reporter.emit({"event": "ridge-fit-complete", **result})
        reporter.status(
            "complete",
            planId=plan["id"],
            selectedModel=selected_model,
            selectedLambda=selected_lambda_value,
            validationNormalizedMse=ridge_metrics["normalizedMse"],
            artifactContentHash=artifact_pointer["object"]["contentHash"],
            message="Immutable closed-form ridge artifact fitted and validated.",
        )
    except Exception as error:
        reporter.status("failed", planId=plan.get("id"), error=str(error))
        raise


if __name__ == "__main__":
    main()
