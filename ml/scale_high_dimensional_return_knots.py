"""Scale JS-only consecutive-return point clouds from 4D through 15D."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from multidimensional_return_knots import (
    AsinhMatrixTransform,
    CovarianceTransform,
    sample_point_cloud,
)
from search_multidimensional_return_knots import (
    DEFAULT_PAIR_ANALYSIS,
    SearchOptions,
    evenly_spaced_rows,
    fast_weighted_lloyd,
    fit_tail_transform,
    read_json,
)
from trading_storage import read_candle_column


DEFAULT_SAMPLE_CACHE = Path(
    "data/runtime-cache/consecutive-return-sequences-4d-15d-v1.npz",
)
DEFAULT_CALIBRATION_ARTIFACT = Path(
    "data/benchmarks/multidimensional-return-knot-joint-js-only.json",
)
DEFAULT_BASE_SAMPLE_CACHE = Path(
    "data/runtime-cache/multidimensional-return-knot-samples-v1.npz",
)
DEFAULT_OUTPUT = Path(
    "data/benchmarks/high-dimensional-return-knot-scaling-4d-15d.json",
)
DEFAULT_MODELS_OUTPUT = Path(
    "data/benchmarks/high-dimensional-return-knot-scaling-4d-15d-models.npz",
)
DEFAULT_REPORT = Path(
    "docs/experiments/high-dimensional-return-knot-scaling-4d-15d-2026-08-18.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit scalable sliced-JS point clouds for 4D through 15D returns.",
    )
    parser.add_argument("--analysis", type=Path, default=DEFAULT_PAIR_ANALYSIS)
    parser.add_argument("--sample-cache", type=Path, default=DEFAULT_SAMPLE_CACHE)
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=DEFAULT_CALIBRATION_ARTIFACT,
    )
    parser.add_argument("--base-sample-cache", type=Path, default=DEFAULT_BASE_SAMPLE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-output", type=Path, default=DEFAULT_MODELS_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(range(4, 16)))
    parser.add_argument("--maximum-samples", type=int, default=320_000)
    parser.add_argument("--metric-draws", type=int, default=65_536)
    parser.add_argument("--projection-count", type=int, default=96)
    parser.add_argument("--projection-bins", type=int, default=256)
    parser.add_argument("--fast-cloud-sample", type=int, default=65_536)
    parser.add_argument("--maximum-knots", type=int, default=32_768)
    parser.add_argument("--high-fidelity-restarts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dimensions = sorted(set(args.dimensions))
    if not dimensions or dimensions[0] < 4 or dimensions[-1] > 15:
        raise ValueError("dimensions must lie between 4 and 15")
    if args.maximum_samples < 1_000 or args.metric_draws < 1_000:
        raise ValueError("sample and metric budgets are too small")
    if args.projection_count < dimensions[-1] or args.projection_bins < 16:
        raise ValueError("projection budget is too small")
    repo = Path(__file__).resolve().parents[1]
    analysis_path = resolve(repo, args.analysis)
    sample_cache_path = resolve(repo, args.sample_cache)
    calibration_path = resolve(repo, args.calibration_artifact)
    base_sample_cache_path = resolve(repo, args.base_sample_cache)
    output_path = resolve(repo, args.output)
    models_output_path = resolve(repo, args.models_output)
    report_path = resolve(repo, args.report)
    analysis = read_json(analysis_path)
    maximum_samples = min(args.maximum_samples, 50_000) if args.quick else args.maximum_samples
    metric_draws = min(args.metric_draws, 16_384) if args.quick else args.metric_draws
    projection_count = min(args.projection_count, 32) if args.quick else args.projection_count
    fast_cloud_sample = min(args.fast_cloud_sample, 24_000) \
        if args.quick else args.fast_cloud_sample
    maximum_knots = min(args.maximum_knots, 4_096) if args.quick else args.maximum_knots
    samples, sample_metadata = load_or_build_sequence_samples(
        repo,
        analysis,
        sample_cache_path,
        maximum_dimension=max(dimensions),
        maximum_samples=maximum_samples,
        rebuild=args.rebuild_cache,
    )
    calibration = calibrate_sliced_js_threshold(
        calibration_path,
        base_sample_cache_path,
        metric_draws,
        projection_count,
        args.projection_bins,
    )
    print(
        f"Calibrated mean sliced-JS threshold: "
        f"{calibration['thresholdMeanSlicedJsBits']:.8g} bits",
        flush=True,
    )
    if args.resume and output_path.exists():
        artifact = read_json(output_path)
        results = artifact.get("dimensions", {})
    else:
        results: dict[str, Any] = {}
    artifact = {
        "version": 2,
        "generatedAt": utc_now(),
        "complete": False,
        "source": {
            "analysis": relative(repo, analysis_path),
            "sampleCache": relative(repo, sample_cache_path),
            "calibrationArtifact": relative(repo, calibration_path),
            "models": relative(repo, models_output_path),
        },
        "methodology": {
            "component": "All-active consecutive-return vectors; exact-zero masks remain discrete.",
            "split": "The first four UTC years fit the representation and determine the knot boundary; the final UTC year is a temporal-drift diagnostic, matching the 2D/3D protocol.",
            "mapping": "Empirical covariance whitening followed by a fitted diagonal invertible sigmoid-asinh transform.",
            "cloud": "Adaptive k-means product-triangular point cloud with one low-mass full-support background component.",
            "criterion": "Balanced mean one-dimensional JS, with equal weight for coordinate axes, adjacent sum/difference directions, and random Cramer-Wold projections.",
            "calibration": "The balanced sliced-JS threshold is the larger score of the exact one-knot 2D and 3D JS boundaries under the identical projection protocol.",
            "search": "Fast sampled cKDTree Lloyd screening followed by deterministic full-data Lloyd refinement, Ward-style nested pruning, and one-knot bracketing; multiple initial full-data fits are retained by lowest sliced JS.",
            "limitation": "Sliced JS is a scalable joint-distribution lower-dimensional projection criterion, not the exponentially sized full-grid JS used in 2D and 3D.",
        },
        "settings": {
            "maximumSamplesPerDimension": maximum_samples,
            "metricDraws": metric_draws,
            "projectionCount": projection_count,
            "projectionBins": args.projection_bins,
            "fastCloudSample": fast_cloud_sample,
            "maximumKnots": maximum_knots,
            "highFidelityRestarts": args.high_fidelity_restarts,
        },
        "sample": sample_metadata,
        "calibration": calibration,
        "dimensions": results,
    }
    for dimension in dimensions:
        key = str(dimension)
        if key in results:
            print(f"\nSkipping completed {dimension}D result.", flush=True)
            continue
        train = samples[f"train{key}"].astype(np.float64)
        validation = samples[f"validation{key}"].astype(np.float64)
        print(
            f"\nFitting {dimension}D: {train.shape[0]:,} train / "
            f"{validation.shape[0]:,} validation all-active samples "
            f"({sample_metadata['allActiveFractions'][key]:.4%} of windows)...",
            flush=True,
        )
        results[key] = fit_dimension(
            dimension,
            train,
            validation,
            calibration["thresholdMeanSlicedJsBits"],
            metric_draws,
            projection_count,
            args.projection_bins,
            fast_cloud_sample,
            maximum_knots,
            args.high_fidelity_restarts,
            args.device,
            args.quick,
        )
        artifact["generatedAt"] = utc_now()
        artifact["dimensions"] = results
        externalize_model_arrays(artifact, models_output_path)
        atomic_write_json(output_path, artifact)
        print(
            f"  checkpointed {dimension}D: "
            + (
                f"{results[key]['final']['knotCount']:,} knots"
                if results[key]["resolved"]
                else f"no pass through {results[key]['maximumTestedKnotCount']:,} knots"
            ),
            flush=True,
        )
    artifact["complete"] = True
    artifact["generatedAt"] = utc_now()
    externalize_model_arrays(artifact, models_output_path)
    atomic_write_json(output_path, artifact)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(artifact), encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


def load_or_build_sequence_samples(
    repo: Path,
    analysis: dict[str, Any],
    cache_path: Path,
    maximum_dimension: int,
    maximum_samples: int,
    rebuild: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    full = analysis["scales"][0]["fullHistory"]
    start = parse_iso(full["startTime"])
    end = parse_iso(full["endTime"])
    train_end = start.replace(year=start.year + 4)
    metadata_key = np.asarray([
        start.timestamp(),
        end.timestamp(),
        train_end.timestamp(),
        maximum_dimension,
        maximum_samples,
        3,
    ], dtype=np.float64)
    if cache_path.exists() and not rebuild:
        with np.load(cache_path) as cached:
            if np.array_equal(cached["metadataKey"], metadata_key):
                arrays = {
                    f"{split}{dimension}": cached[f"{split}{dimension}"]
                    for split in ("train", "validation")
                    for dimension in range(4, maximum_dimension + 1)
                }
                metadata = json.loads(str(cached["metadataJson"][0]))
                return arrays, metadata
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    selected = sorted(
        file for file in source.glob("????-??-??.json")
        if start <= datetime.fromisoformat(file.stem).replace(tzinfo=timezone.utc) < end
    )
    if not selected:
        raise RuntimeError("no 1s candle references cover the analysis window")
    prior_candidates = sorted(file for file in source.glob("????-??-??.json") if file < selected[0])
    if not prior_candidates:
        raise RuntimeError("a pre-window close is required")
    previous_close = float(read_candle_column(prior_candidates[-1], "close")[-1])
    buffers: dict[str, list[np.ndarray]] = {
        f"{split}{dimension}": []
        for split in ("train", "validation")
        for dimension in range(4, maximum_dimension + 1)
    }
    mask_counts = {
        f"{split}{dimension}": np.zeros(1 << dimension, dtype=np.int64)
        for split in ("train", "validation")
        for dimension in range(4, maximum_dimension + 1)
    }
    total_windows = {"train": 0, "validation": 0}
    tails = np.empty(0, dtype=np.float64)
    assumed_nonzero_probability = 0.70
    strides = {
        f"{split}{dimension}": max(1, int(round(
            max((split_end - split_start).total_seconds(), 1.0)
            * assumed_nonzero_probability ** dimension
            / (maximum_samples * 1.2)
        )))
        for split, split_start, split_end in (
            ("train", start, train_end),
            ("validation", train_end, end),
        )
        for dimension in range(4, maximum_dimension + 1)
    }
    for day_index, reference in enumerate(selected):
        if day_index % 50 == 0:
            print(f"Reading 4D-15D samples {day_index}/{len(selected)}...", flush=True)
        closes = read_candle_column(reference, "close").astype(np.float64, copy=False)
        prior = np.empty(closes.size + 1, dtype=np.float64)
        prior[0] = previous_close
        prior[1:] = closes
        returns = np.log(prior[1:] / prior[:-1]) * 10_000.0
        previous_close = float(closes[-1])
        day = datetime.fromisoformat(reference.stem).replace(tzinfo=timezone.utc)
        split = "train" if day < train_end else "validation"
        combined = np.concatenate((tails, returns))
        if combined.size >= maximum_dimension:
            windows = np.lib.stride_tricks.sliding_window_view(combined, maximum_dimension)
            masks = np.zeros(windows.shape[0], dtype=np.int32)
            for axis in range(maximum_dimension):
                masks |= (windows[:, axis] != 0.0).astype(np.int32) << axis
            total_windows[split] += int(windows.shape[0])
            for dimension in range(4, maximum_dimension + 1):
                key = f"{split}{dimension}"
                prefix_mask = (1 << dimension) - 1
                dimension_masks = masks & prefix_mask
                mask_counts[key] += np.bincount(
                    dimension_masks,
                    minlength=1 << dimension,
                )
                eligible = np.flatnonzero(dimension_masks == prefix_mask)
                stride = strides[key]
                offset = ((day_index + 1) * (41 + 2 * dimension)) % stride
                chosen = eligible[offset::stride]
                if chosen.size:
                    buffers[key].append(
                        windows[chosen, :dimension].astype(np.float32),
                    )
        tails = combined[-(maximum_dimension - 1):].copy()
    arrays: dict[str, np.ndarray] = {}
    for key, parts in buffers.items():
        dimension = int("".join(character for character in key if character.isdigit()))
        values = np.concatenate(parts, axis=0) \
            if parts else np.empty((0, dimension), dtype=np.float32)
        if values.shape[0] > maximum_samples:
            values = evenly_spaced_rows(values, maximum_samples)
        arrays[key] = values
    combined_counts = {
        dimension: int(mask_counts[f"train{dimension}"][-1])
        + int(mask_counts[f"validation{dimension}"][-1])
        for dimension in range(4, maximum_dimension + 1)
    }
    combined_windows = total_windows["train"] + total_windows["validation"]
    metadata = {
        "startTime": start.isoformat().replace("+00:00", "Z"),
        "trainEndTime": train_end.isoformat().replace("+00:00", "Z"),
        "endTime": end.isoformat().replace("+00:00", "Z"),
        "totalWindows": total_windows,
        "maximumDimension": maximum_dimension,
        "targetSamplesPerDimension": maximum_samples,
        "samplingStrides": strides,
        "sampleCounts": {key: int(value.shape[0]) for key, value in arrays.items()},
        "zeroMaskCounts": {key: value.tolist() for key, value in mask_counts.items()},
        "maskBitOrder": "least-significant bit is the earliest return; 1 means nonzero",
        "allActiveCounts": {
            str(key): value for key, value in combined_counts.items()
        },
        "allActiveFractions": {
            str(key): value / max(combined_windows, 1)
            for key, value in combined_counts.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        metadataKey=metadata_key,
        metadataJson=np.asarray([json.dumps(metadata)]),
        **arrays,
    )
    return arrays, metadata


def calibrate_sliced_js_threshold(
    artifact_path: Path,
    sample_cache_path: Path,
    metric_draws: int,
    projection_count: int,
    projection_bins: int,
) -> dict[str, Any]:
    artifact = read_json(artifact_path)
    scores: dict[str, Any] = {}
    with np.load(sample_cache_path) as samples:
        for dimension in (2, 3):
            result = artifact["dimensions"][str(dimension)]
            final = result["final"]
            train = samples[f"train{dimension}"].astype(np.float64)
            validation = samples[f"validation{dimension}"].astype(np.float64)
            covariance = covariance_from_payload(final["covariance"])
            transform = AsinhMatrixTransform(
                np.asarray(final["postAsinhMatrix"], dtype=np.float64),
            )
            true_unit = transform.forward(covariance.forward(
                evenly_spaced_rows(train, metric_draws),
            ))
            fit = final["fit"]
            model_unit = sample_point_cloud(
                np.asarray(fit["centersUnit"], dtype=np.float64),
                np.asarray(fit["bandwidthsUnit"], dtype=np.float64),
                np.asarray(fit["componentWeights"], dtype=np.float64),
                metric_draws,
                seed=73_001 + dimension,
            )
            directions = projection_directions(dimension, projection_count, 11_003)
            metric = sliced_js_metrics(
                true_unit,
                model_unit,
                directions,
                projection_bins,
            )
            validation_unit = transform.forward(covariance.forward(
                evenly_spaced_rows(validation, metric_draws),
            ))
            validation_metric = sliced_js_metrics(
                validation_unit,
                model_unit[:validation_unit.shape[0]],
                directions,
                projection_bins,
            )
            scores[str(dimension)] = {
                "knotCount": int(final["knotCount"]),
                "exactGridJsBitsPerDimension": float(
                    fit["density"]["jensenShannonBitsPerDimension"],
                ),
                **metric,
                "validation": validation_metric,
            }
    threshold = max(float(value["meanSlicedJsBits"]) for value in scores.values())
    return {
        "dimensions": scores,
        "thresholdMeanSlicedJsBits": threshold,
        "selection": "maximum calibrated 2D/3D boundary score",
    }


def fit_dimension(
    dimension: int,
    train: np.ndarray,
    validation: np.ndarray,
    threshold: float,
    metric_draws: int,
    projection_count: int,
    projection_bins: int,
    fast_cloud_sample: int,
    maximum_knots: int,
    high_fidelity_restarts: int,
    device: str,
    quick: bool,
) -> dict[str, Any]:
    if train.shape[0] < 1_000 or validation.shape[0] < 1_000:
        raise RuntimeError(
            f"only {train.shape[0]} train / {validation.shape[0]} validation "
            f"all-active {dimension}D samples are available",
        )
    covariance = CovarianceTransform.fit(train)
    whitened = covariance.forward(train)
    whitened_validation = covariance.forward(validation)
    transform_options = SearchOptions(
        device=device,
        transform_steps=40 if quick else 140,
        transform_sample=min(train.shape[0], 50_000 if quick else 160_000),
        cloud_sample=fast_cloud_sample,
        conditional_draws=0,
        train_stride=1,
        validation_stride=1,
        rebuild_cache=False,
        quick=quick,
    )
    scalar = fit_tail_transform(
        whitened,
        whitened_validation,
        "scalar",
        transform_options,
        None,
    )
    diagonal = fit_tail_transform(
        whitened,
        whitened_validation,
        "vector",
        transform_options,
        scalar,
    )
    transform = AsinhMatrixTransform(np.asarray(diagonal["matrix"], dtype=np.float64))
    unit = transform.forward(whitened)
    validation_unit = transform.forward(whitened_validation)
    directions = projection_directions(dimension, projection_count, 11_003)
    true_metric_sample = evenly_spaced_rows(unit, metric_draws)
    fit_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def fast_fit(count: int, restart: int = 0) -> dict[str, Any]:
        key = (count, restart)
        if key not in fit_cache:
            fit_cache[key] = fit_point_cloud(
                unit,
                count,
                directions,
                true_metric_sample,
                projection_bins,
                metric_draws,
                cloud_sample=fast_cloud_sample,
                high_fidelity=False,
                seed=101_003 + dimension * 10_000 + restart,
                metric_seed=701_003 + dimension * 1_000 + restart * 31,
            )
        return fit_cache[key]

    summaries: list[dict[str, Any]] = []
    count = min(256, maximum_knots, train.shape[0] - 1)
    current = fast_fit(count)
    summaries.append(fit_summary(current, "fast screening"))
    if passes(current, threshold):
        while count > 64:
            next_count = max(64, count // 2)
            candidate = fast_fit(next_count)
            summaries.append(fit_summary(candidate, "fast lower screening"))
            if not passes(candidate, threshold):
                break
            current = candidate
            count = next_count
            if count == 64:
                break
    else:
        while count < min(maximum_knots, train.shape[0] - 1):
            count = min(count * 2, maximum_knots, train.shape[0] - 1)
            current = fast_fit(count)
            summaries.append(fit_summary(current, "fast upper screening"))
            if passes(current, threshold):
                break
    fast_selected = int(current["knotCount"])

    def initial_high_fit(count_value: int) -> dict[str, Any]:
        candidates = [
            fit_point_cloud(
                unit,
                count_value,
                directions,
                true_metric_sample,
                projection_bins,
                metric_draws,
                cloud_sample=train.shape[0],
                high_fidelity=True,
                seed=201_003 + dimension * 10_000 + restart,
                metric_seed=801_003 + dimension * 1_000,
            )
            for restart in range(high_fidelity_restarts)
        ]
        for candidate in candidates:
            summaries.append(fit_summary(candidate, "high-fidelity candidate"))
        return min(candidates, key=lambda candidate: candidate["meanSlicedJsBits"])

    candidate = initial_high_fit(fast_selected)
    high_pass: dict[str, Any] | None = candidate if passes(candidate, threshold) else None
    high_count = fast_selected
    while high_pass is None and high_count < min(maximum_knots, train.shape[0] - 1):
        high_count = min(high_count * 2, maximum_knots, train.shape[0] - 1)
        candidate = initial_high_fit(high_count)
        if passes(candidate, threshold):
            high_pass = candidate
    def validation_score(model: dict[str, Any]) -> dict[str, Any]:
        validation_draws = min(metric_draws, validation_unit.shape[0])
        validation_model_sample = sample_point_cloud(
            np.asarray(model["centersUnit"], dtype=np.float64),
            np.asarray(model["bandwidthsUnit"], dtype=np.float64),
            np.asarray(model["componentWeights"], dtype=np.float64),
            validation_draws,
            seed=191_003 + dimension,
        )
        return sliced_js_metrics(
            evenly_spaced_rows(validation_unit, validation_draws),
            validation_model_sample,
            directions,
            projection_bins,
        )

    if high_pass is None:
        return {
            "dimension": dimension,
            "resolved": False,
            "observations": int(train.shape[0]),
            "validationObservations": int(validation.shape[0]),
            "covariance": covariance_payload(covariance),
            "postAsinhTransform": diagonal,
            "calibratedThresholdMeanSlicedJsBits": threshold,
            "fastSelectedKnotCount": fast_selected,
            "candidateSummaries": summaries,
            "maximumTestedKnotCount": int(candidate["knotCount"]),
            "lowerBoundKnotCount": int(candidate["knotCount"]) + 1,
            "lowerFailure": fit_summary(candidate, "maximum tested failure"),
            "final": {
                "knotCount": int(candidate["knotCount"]),
                "passes": False,
                "fit": candidate,
                "validation": validation_score(candidate),
            },
        }

    def nested_fit(parent: dict[str, Any], target_count: int) -> dict[str, Any]:
        initial_centers = prune_adaptive_centers(parent, target_count)
        tested = fit_point_cloud(
            unit,
            target_count,
            directions,
            true_metric_sample,
            projection_bins,
            metric_draws,
            cloud_sample=train.shape[0],
            high_fidelity=True,
            seed=301_003 + dimension * 10_000,
            metric_seed=801_003 + dimension * 1_000,
            initial_adaptive_centers=initial_centers,
        )
        summaries.append(fit_summary(tested, "nested high-fidelity pruning"))
        return tested

    low_fail: dict[str, Any] | None = None
    while int(high_pass["knotCount"]) > 64:
        high_count = int(high_pass["knotCount"])
        prune_step = min(128, max(16, high_count // 8))
        probe = max(64, high_count - prune_step)
        tested = nested_fit(high_pass, probe)
        if passes(tested, threshold):
            high_pass = tested
        else:
            low_fail = tested
            break
    if low_fail is None:
        low_count = int(high_pass["knotCount"]) - 1
        if low_count < 2:
            raise RuntimeError("the minimum supported point cloud still passes")
        low_fail = nested_fit(high_pass, low_count)
        if passes(low_fail, threshold):
            raise RuntimeError("failed to establish a lower nested rejection")
    low = int(low_fail["knotCount"])
    high = int(high_pass["knotCount"])
    while high - low > 1:
        mid = (low + high) // 2
        tested = nested_fit(high_pass, mid)
        if passes(tested, threshold):
            high = mid
            high_pass = tested
        else:
            low = mid
            low_fail = tested
        print(
            f"  {dimension}D high-fidelity bracket: {low:,} fail / {high:,} pass",
            flush=True,
        )
    return {
        "dimension": dimension,
        "resolved": True,
        "observations": int(train.shape[0]),
        "validationObservations": int(validation.shape[0]),
        "covariance": covariance_payload(covariance),
        "postAsinhTransform": diagonal,
        "calibratedThresholdMeanSlicedJsBits": threshold,
        "fastSelectedKnotCount": fast_selected,
        "candidateSummaries": summaries,
        "lowerFailure": fit_summary(low_fail, "exact lower failure"),
        "final": {
            "knotCount": int(high_pass["knotCount"]),
            "passes": True,
            "fit": high_pass,
            "validation": validation_score(high_pass),
        },
    }


def fit_point_cloud(
    unit_values: np.ndarray,
    count: int,
    directions: np.ndarray,
    true_metric_sample: np.ndarray,
    projection_bins: int,
    metric_draws: int,
    cloud_sample: int,
    high_fidelity: bool,
    seed: int,
    metric_seed: int | None = None,
    initial_adaptive_centers: np.ndarray | None = None,
) -> dict[str, Any]:
    dimension = unit_values.shape[1]
    adaptive_count = count - 1
    selected = evenly_spaced_rows(unit_values, cloud_sample)
    if adaptive_count >= selected.shape[0]:
        raise ValueError("knot count must be smaller than cloud sample")
    adaptive_centers, labels = fast_weighted_lloyd(
        selected,
        np.ones(selected.shape[0], dtype=np.float64),
        adaptive_count,
        initial_adaptive_centers,
        seed,
    )
    cluster_counts = np.bincount(labels, minlength=adaptive_count).astype(np.float64)
    residual = selected - adaptive_centers[labels]
    squared = np.zeros((adaptive_count, dimension), dtype=np.float64)
    for axis in range(dimension):
        np.add.at(squared[:, axis], labels, residual[:, axis] ** 2)
    standard_deviation = np.sqrt(squared / np.maximum(cluster_counts[:, None], 1.0))
    base_widths = np.maximum(
        math.sqrt(6.0) * standard_deviation,
        0.015 * adaptive_count ** (-1.0 / dimension),
    )
    background_weight = min(0.002, 1.0 / count)
    adaptive_weights = cluster_counts / np.sum(cluster_counts) * (1.0 - background_weight)
    weights = np.concatenate((adaptive_weights, [background_weight]))
    centers = np.vstack((adaptive_centers, np.full((1, dimension), 0.5)))
    best: tuple[float, float, dict[str, Any], np.ndarray] | None = None
    scale_metrics = []
    for scale_index, scale in enumerate((0.75, 0.95, 1.2, 1.5, 1.9)):
        widths = np.vstack((
            np.clip(base_widths * scale, 1e-5, 2.0),
            np.full((1, dimension), 0.5),
        ))
        model_sample = sample_point_cloud(
            centers,
            widths,
            weights,
            metric_draws,
            (seed + 1_000 if metric_seed is None else metric_seed) + scale_index,
        )
        metric = sliced_js_metrics(
            true_metric_sample,
            model_sample,
            directions,
            projection_bins,
        )
        scale_metrics.append({"scale": scale, **metric})
        candidate = (float(metric["meanSlicedJsBits"]), scale, metric, widths)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("bandwidth fit failed")
    _, selected_scale, metric, widths = best
    return {
        "knotCount": count,
        "fitFidelity": "high" if high_fidelity else "fast",
        "cloudSample": int(selected.shape[0]),
        "metricDraws": metric_draws,
        "projectionCount": int(directions.shape[0]),
        "projectionBins": projection_bins,
        "bandwidthScale": selected_scale,
        "bandwidthCandidates": scale_metrics,
        "centersUnit": centers.tolist(),
        "bandwidthsUnit": widths.tolist(),
        "componentWeights": weights.tolist(),
        **metric,
    }


def prune_adaptive_centers(fit: dict[str, Any], target_count: int) -> np.ndarray:
    """Merge the least costly nearest pairs until a nested target size is reached."""
    centers = np.asarray(fit["centersUnit"], dtype=np.float64)[:-1].copy()
    weights = np.asarray(fit["componentWeights"], dtype=np.float64)[:-1].copy()
    target_adaptive_count = target_count - 1
    if target_adaptive_count < 1 or target_adaptive_count >= centers.shape[0]:
        raise ValueError("nested target must remove at least one adaptive center")
    while centers.shape[0] > target_adaptive_count:
        tree = cKDTree(centers)
        distances, neighbors = tree.query(centers, k=2, workers=-1)
        nearest = neighbors[:, 1]
        pair_cost = (
            weights * weights[nearest]
            / np.maximum(weights + weights[nearest], np.finfo(np.float64).tiny)
            * distances[:, 1] ** 2
        )
        maximum_merges = min(
            centers.shape[0] - target_adaptive_count,
            centers.shape[0] // 2,
        )
        used = np.zeros(centers.shape[0], dtype=bool)
        pairs: list[tuple[int, int]] = []
        for first_raw in np.argsort(pair_cost):
            first = int(first_raw)
            second = int(nearest[first])
            if first == second or used[first] or used[second]:
                continue
            used[first] = True
            used[second] = True
            pairs.append((first, second))
            if len(pairs) == maximum_merges:
                break
        if not pairs:
            raise RuntimeError("nearest-pair pruning made no progress")
        merged_centers = []
        merged_weights = []
        for first, second in pairs:
            merged_weight = weights[first] + weights[second]
            merged_centers.append((
                weights[first] * centers[first] + weights[second] * centers[second]
            ) / max(merged_weight, np.finfo(np.float64).tiny))
            merged_weights.append(merged_weight)
        centers = np.vstack((centers[~used], np.asarray(merged_centers)))
        weights = np.concatenate((weights[~used], np.asarray(merged_weights)))
    return np.clip(centers, 1e-8, 1.0 - 1e-8)


def projection_directions(dimension: int, count: int, seed: int) -> np.ndarray:
    directions: list[np.ndarray] = []
    for axis in range(dimension):
        direction = np.zeros(dimension, dtype=np.float64)
        direction[axis] = 1.0
        directions.append(direction)
    for axis in range(dimension - 1):
        for sign in (1.0, -1.0):
            direction = np.zeros(dimension, dtype=np.float64)
            direction[axis] = 1.0
            direction[axis + 1] = sign
            directions.append(direction / math.sqrt(2.0))
    rng = np.random.default_rng(seed + dimension)
    while len(directions) < count:
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        directions.append(direction)
    return np.asarray(directions[:count], dtype=np.float64)


def sliced_js_metrics(
    true_unit: np.ndarray,
    model_unit: np.ndarray,
    directions: np.ndarray,
    bins: int,
) -> dict[str, Any]:
    true_centered = true_unit - 0.5
    model_centered = model_unit - 0.5
    true_projected = true_centered @ directions.T
    model_projected = model_centered @ directions.T
    values = np.empty(directions.shape[0], dtype=np.float64)
    for index, direction in enumerate(directions):
        bound = 0.5 * float(np.sum(np.abs(direction)))
        target = np.histogram(
            true_projected[:, index],
            bins=bins,
            range=(-bound, bound),
        )[0].astype(np.float64)
        model = np.histogram(
            model_projected[:, index],
            bins=bins,
            range=(-bound, bound),
        )[0].astype(np.float64)
        target /= np.sum(target)
        model /= np.sum(model)
        midpoint = (target + model) / 2.0
        target_positive = target > 0.0
        model_positive = model > 0.0
        values[index] = 0.5 * (
            np.sum(target[target_positive] * np.log2(
                target[target_positive] / midpoint[target_positive],
            ))
            + np.sum(model[model_positive] * np.log2(
                model[model_positive] / midpoint[model_positive],
            ))
        )
    dimension = true_unit.shape[1]
    adjacent_end = min(directions.shape[0], dimension + 2 * (dimension - 1))
    axis_mean = float(np.mean(values[:dimension]))
    adjacent_mean = float(np.mean(values[dimension:adjacent_end])) \
        if adjacent_end > dimension else axis_mean
    random_mean = float(np.mean(values[adjacent_end:])) \
        if directions.shape[0] > adjacent_end else adjacent_mean
    balanced_mean = float(np.mean((axis_mean, adjacent_mean, random_mean)))
    return {
        "meanSlicedJsBits": balanced_mean,
        "projectionMeanSlicedJsBits": float(np.mean(values)),
        "medianSlicedJsBits": float(np.median(values)),
        "p90SlicedJsBits": float(np.quantile(values, 0.9)),
        "maximumSlicedJsBits": float(np.max(values)),
        "axisMeanSlicedJsBits": axis_mean,
        "adjacentMeanSlicedJsBits": adjacent_mean,
        "randomMeanSlicedJsBits": random_mean,
    }


def passes(fit: dict[str, Any], threshold: float) -> bool:
    return float(fit["meanSlicedJsBits"]) <= threshold


def fit_summary(fit: dict[str, Any], stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "knotCount": int(fit["knotCount"]),
        "fitFidelity": fit["fitFidelity"],
        "meanSlicedJsBits": float(fit["meanSlicedJsBits"]),
        "p90SlicedJsBits": float(fit["p90SlicedJsBits"]),
        "maximumSlicedJsBits": float(fit["maximumSlicedJsBits"]),
        "bandwidthScale": float(fit["bandwidthScale"]),
    }


def covariance_payload(covariance: CovarianceTransform) -> dict[str, Any]:
    return {
        "centerBps": covariance.center.tolist(),
        "covarianceBpsSquared": covariance.covariance.tolist(),
        "whitening": covariance.whitening.tolist(),
        "coloring": covariance.coloring.tolist(),
    }


def covariance_from_payload(payload: dict[str, Any]) -> CovarianceTransform:
    whitening = np.asarray(payload["whitening"], dtype=np.float64)
    coloring = np.asarray(payload["coloring"], dtype=np.float64)
    return CovarianceTransform(
        center=np.asarray(payload["centerBps"], dtype=np.float64),
        whitening=whitening,
        coloring=coloring,
        covariance=coloring @ coloring.T,
    )


def render_report(artifact: dict[str, Any]) -> str:
    calibration = artifact["calibration"]
    lines = [
        "# High-dimensional consecutive-return knot scaling (4D-15D)",
        "",
        f"Generated {artifact['generatedAt']}.",
        "",
        "The dense full-grid JS used through 3D grows exponentially. This experiment instead "
        "uses deterministic sliced JS across coordinate, adjacent sum/difference, and random "
        "Cramer-Wold projections. The three group means receive equal weight, and the threshold "
        "is calibrated to the final exact 2D/3D boundaries.",
        "",
        f"Calibrated balanced mean sliced-JS threshold: "
        f"**{calibration['thresholdMeanSlicedJsBits']:.8g} bits**.",
        "",
        "| dimensions | train / validation observations | all-active mass | knots | train balanced JS | validation balanced JS | p90 / max train JS |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sample = artifact["sample"]
    for key in sorted(artifact["dimensions"], key=int):
        result = artifact["dimensions"][key]
        fit = result["final"]["fit"]
        validation = result["final"]["validation"]
        resolved = bool(result.get("resolved", result["final"]["passes"]))
        knot_label = f"{fit['knotCount']:,}" if resolved else f">{fit['knotCount']:,}"
        lines.append(
            f"| {key} | {result['observations']:,} / "
            f"{result['validationObservations']:,} | "
            f"{sample['allActiveFractions'][key]:.6%} | {knot_label} | "
            f"{fit['meanSlicedJsBits']:.8g} | "
            f"{validation['meanSlicedJsBits']:.8g} | "
            f"{fit['p90SlicedJsBits']:.8g} / {fit['maximumSlicedJsBits']:.8g} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- These counts are one-knot training boundaries under the recorded deterministic multistart protocol; validation is diagnostic.",
        "- A leading `>` means the fit still failed at the maximum tested count, so only a lower bound is established.",
        "- Sliced JS tests the joint law through projections but is not numerically identical to full-grid JS.",
        "- Only the all-active continuous component is fitted. Exact-zero mask probabilities are retained separately.",
        "- Conditional mean/median errors are not acceptance criteria in this JS-only experiment.",
        "",
        "## Reproduction",
        "",
        "```text",
        "node scripts/run-ml-python.mjs ml/scale_high_dimensional_return_knots.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def externalize_model_arrays(artifact: dict[str, Any], path: Path) -> None:
    """Move bulky final cloud arrays from JSON into a compressed numeric artifact."""
    arrays: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as existing:
            arrays.update({key: existing[key] for key in existing.files})
    changed = False
    for dimension, result in artifact.get("dimensions", {}).items():
        result.setdefault("resolved", bool(result["final"].get("passes", False)))
        fit = result["final"]["fit"]
        if "centersUnit" not in fit:
            continue
        keys = {
            "centersUnit": f"d{dimension}CentersUnit",
            "bandwidthsUnit": f"d{dimension}BandwidthsUnit",
            "componentWeights": f"d{dimension}ComponentWeights",
        }
        shapes: dict[str, list[int]] = {}
        for json_key, array_key in keys.items():
            value = np.asarray(fit.pop(json_key), dtype=np.float64)
            arrays[array_key] = value
            shapes[json_key] = list(value.shape)
        fit["modelArrays"] = {
            "keys": keys,
            "shapes": shapes,
            "dtype": "float64",
        }
        changed = True
    if not changed and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def relative(repo: Path, path: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
