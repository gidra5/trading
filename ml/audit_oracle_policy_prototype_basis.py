"""Train/validation-only audit of a soft oracle-policy prototype basis.

The prototypes are forward-KL (Bregman) centroids fitted exclusively from the
purged training targets.  Hard assignments are used only to fit and diagnose
the dictionary.  A production model using this basis would predict continuous
simplex weights and optimize the original soft-target KL through ``weights @
prototypes``; no hard cluster label is proposed as supervision.

Held-out test reference files and payloads are deliberately never opened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from audit_causal_oracle_predictability import (
    ACTION_COUNT,
    DAY_ROWS,
    SOURCE_TEMPERATURE,
    TEST_DAYS,
    VALIDATION_DAYS,
    mean_entropy,
    mean_kl,
    normalized_mean,
    split_and_purge,
)
from trading_storage import read_shard_array
from trading_storage import write_shard_payload


DEFAULT_KS = (8, 16, 32, 64)
PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class FitResult:
    prototypes: np.ndarray
    train_weights: np.ndarray
    sample_objective: tuple[float, ...]
    full_polish_objective: tuple[float, ...]
    fit_sample_rows: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ks",
        default=",".join(str(value) for value in DEFAULT_KS),
        help="Comma-separated prototype counts.",
    )
    parser.add_argument("--fit-sample-rows", type=int, default=65_536)
    parser.add_argument("--lloyd-iterations", type=int, default=10)
    parser.add_argument("--full-polish-passes", type=int, default=1)
    parser.add_argument("--projection-iterations", type=int, default=32)
    parser.add_argument("--projection-line-search-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=71391)
    parser.add_argument(
        "--artifact-reference",
        type=Path,
        help=(
            "Optional immutable reference path under data/training/immutable/"
            "refs. Exactly one K must be requested."
        ),
    )
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help=(
            "Fit and write the train-only prototype artifact without opening "
            "validation targets. Requires --artifact-reference."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    ks = tuple(int(value) for value in arguments.ks.split(","))
    if not ks or any(value <= 1 for value in ks):
        raise ValueError("prototype counts must all be greater than one")
    if arguments.fit_sample_rows <= max(ks):
        raise ValueError("fit sample must contain more rows than prototypes")
    if arguments.lloyd_iterations < 1 \
            or arguments.full_polish_passes < 0 \
            or arguments.projection_iterations < 1 \
            or arguments.projection_line_search_steps < 1:
        raise ValueError("iteration counts are invalid")
    if arguments.artifact_reference is not None and len(ks) != 1:
        raise ValueError("artifact generation requires exactly one K")
    if arguments.artifact_only and arguments.artifact_reference is None:
        raise ValueError("--artifact-only requires --artifact-reference")

    repo_root = Path(__file__).resolve().parents[1]
    target_root = (
        repo_root
        / "data/training/immutable/refs/oracle/1s"
        / "hindsight-bot-71391c44b323e044e6ab"
    )
    target_files = sorted(target_root.glob("*.json"))
    if len(target_files) <= VALIDATION_DAYS + TEST_DAYS:
        raise ValueError("causal oracle corpus is too short")
    segments = split_and_purge(target_files)
    requested_splits = "train only" if arguments.artifact_only \
        else "train/validation"
    print(
        f"Loading purged {requested_splits} oracle targets; sealed test "
        "references and payloads remain unopened.",
        file=sys.stderr,
        flush=True,
    )
    train_targets = load_targets("train", segments["train"])
    if arguments.artifact_only:
        k = ks[0]
        fitted = fit_kl_prototypes(
            train_targets,
            k,
            sample_rows=arguments.fit_sample_rows,
            lloyd_iterations=arguments.lloyd_iterations,
            full_polish_passes=arguments.full_polish_passes,
            seed=arguments.seed + k,
        )
        artifact = write_prototype_artifact(
            repo_root,
            arguments.artifact_reference,
            fitted,
            k=k,
            seed=arguments.seed + k,
            lloyd_iterations=arguments.lloyd_iterations,
            full_polish_passes=arguments.full_polish_passes,
            train_targets=train_targets,
            train_segments=segments["train"],
        )
        print(json.dumps({
            "schemaVersion": 1,
            "accessContract": {
                "trainTargetPayloadsOpened": len({
                    segment.target_file for segment in segments["train"]
                }),
                "validationTargetPayloadsOpened": 0,
                "testReferencesOpened": 0,
                "testPayloadsOpened": 0,
                "gpuUsed": False,
            },
            "artifact": artifact,
        }, indent=2, allow_nan=False))
        return
    validation_targets = load_targets("validation", segments["validation"])
    train_prior = normalized_mean(train_targets)
    prior_prediction = np.broadcast_to(train_prior, validation_targets.shape)
    prior_kl = mean_kl(validation_targets, prior_prediction)
    target_entropy = mean_entropy(validation_targets)

    results = []
    for k in ks:
        print(f"Fitting K={k} forward-KL prototypes.", file=sys.stderr, flush=True)
        fitted = fit_kl_prototypes(
            train_targets,
            k,
            sample_rows=arguments.fit_sample_rows,
            lloyd_iterations=arguments.lloyd_iterations,
            full_polish_passes=arguments.full_polish_passes,
            seed=arguments.seed + k,
        )
        artifact = None
        if arguments.artifact_reference is not None:
            artifact = write_prototype_artifact(
                repo_root,
                arguments.artifact_reference,
                fitted,
                k=k,
                seed=arguments.seed + k,
                lloyd_iterations=arguments.lloyd_iterations,
                full_polish_passes=arguments.full_polish_passes,
                train_targets=train_targets,
                train_segments=segments["train"],
            )
        nearest_kl, nearest_labels = nearest_prototype_kl(
            validation_targets,
            fitted.prototypes,
        )
        basis_prior = fitted.train_weights @ fitted.prototypes
        basis_prior /= basis_prior.sum()
        basis_prior_kl = mean_kl(
            validation_targets,
            np.broadcast_to(basis_prior, validation_targets.shape),
        )
        print(
            f"K={k}: nearest KL={nearest_kl:.9f}; projecting soft mixtures.",
            file=sys.stderr,
            flush=True,
        )
        projection = frank_wolfe_kl_projection(
            validation_targets,
            fitted.prototypes,
            initial_labels=nearest_labels,
            iterations=arguments.projection_iterations,
            line_search_steps=arguments.projection_line_search_steps,
        )
        projected_kl = mean_kl(validation_targets, projection["prediction"])
        projected_lower_bound = max(
            0.0,
            projected_kl - float(projection["meanGap"]),
        )
        results.append({
            "prototypeCount": k,
            **({"artifact": artifact} if artifact is not None else {}),
            "fit": {
                "geometry": "forward-kl-bregman",
                "centroid": "arithmetic-mean-soft-policy",
                "hardAssignmentsUsedForSupervision": False,
                "fitSampleRows": fitted.fit_sample_rows,
                "sampleLloydMeanKl": list(fitted.sample_objective),
                "fullTrainPolishMeanKl": list(fitted.full_polish_objective),
                "nonemptyPrototypeCount": int(np.count_nonzero(
                    fitted.train_weights > 0
                )),
                "minTrainMixtureWeight": float(fitted.train_weights.min()),
                "maxTrainMixtureWeight": float(fitted.train_weights.max()),
                "effectiveTrainMixturePrototypeCount": float(np.exp(
                    -np.sum(
                        fitted.train_weights
                        * np.log(np.clip(fitted.train_weights, 1e-300, None))
                    )
                )),
            },
            "validationRepresentation": {
                "targetEntropyNats": target_entropy,
                "nearestPrototypeKl": nearest_kl,
                "nearestPrototypeKlFractionOfPrior": nearest_kl / prior_kl,
                "convexMixtureProjectedKl": projected_kl,
                "convexMixtureOptimalKlLowerBoundFromMeanGap": (
                    projected_lower_bound
                ),
                "convexMixtureOptimalKlCertifiedInterval": [
                    projected_lower_bound,
                    projected_kl,
                ],
                "convexMixtureProjectedKlFractionOfPrior": (
                    projected_kl / prior_kl
                ),
                "projectionIterations": arguments.projection_iterations,
                "projectionFinalMeanFrankWolfeGap": projection["meanGap"],
                "projectionFinalP95FrankWolfeGap": projection["p95Gap"],
                "projectionObjectiveTrace": projection["objectiveTrace"],
            },
            "prototypePriorMixture": {
                "validationKl": basis_prior_kl,
                "directTrainPriorValidationKl": prior_kl,
                "absoluteKlDifferenceFromDirectTrainPrior": (
                    basis_prior_kl - prior_kl
                ),
                "maxAbsoluteProbabilityDifferenceFromDirectTrainPrior": float(
                    np.max(np.abs(basis_prior - train_prior))
                ),
            },
        })
        print(
            f"K={k}: projected KL={projected_kl:.9f}, "
            f"mean FW gap={projection['meanGap']:.6g}.",
            file=sys.stderr,
            flush=True,
        )

    report = {
        "schemaVersion": 1,
        "accessContract": {
            "trainTargetPayloadsOpened": len({
                segment.target_file for segment in segments["train"]
            }),
            "validationTargetPayloadsOpened": len({
                segment.target_file for segment in segments["validation"]
            }),
            "testReferencesOpened": 0,
            "testPayloadsOpened": 0,
            "gpuUsed": False,
        },
        "corpus": {
            "trainRowsAfterPurge": int(train_targets.shape[0]),
            "validationRowsAfterPurge": int(validation_targets.shape[0]),
            "heldoutTestFileCount": TEST_DAYS,
            "actionCount": ACTION_COUNT,
            "oracleTemperature": SOURCE_TEMPERATURE,
        },
        "validationBaselines": {
            "targetEntropyNats": target_entropy,
            "targetCrossEntropyFromTrainPriorNats": target_entropy + prior_kl,
            "directTrainPriorKl": prior_kl,
        },
        "interpretationContract": {
            "nearestPrototype": (
                "conservative hard-reconstruction diagnostic only"
            ),
            "convexMixtureProjection": (
                "oracle-information representation ceiling for p=weights@basis"
            ),
            "causalHeadTraining": (
                "predict continuous weights and minimize original soft-target "
                "KL; never supervise hard prototype assignments"
            ),
            "causalPredictability": (
                "projection removes only representation error and does not "
                "show that history can predict the oracle-specific weights"
            ),
        },
        "prototypeCounts": results,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


def load_targets(split: str, segments: Iterable[object]) -> np.ndarray:
    parts: list[np.ndarray] = []
    segments = list(segments)
    for index, segment in enumerate(segments, start=1):
        _shard, raw = read_shard_array(
            segment.target_file,
            "<f4",
            (DAY_ROWS, ACTION_COUNT),
        )
        start = segment.target_row_offset
        end = start + segment.count
        part = np.asarray(raw[start:end], dtype=np.float32)
        if part.ndim != 2 or part.shape[1] != ACTION_COUNT:
            raise RuntimeError(f"{split} oracle target shape is invalid")
        parts.append(part)
        if index % 50 == 0 or index == len(segments):
            print(
                f"{split}: {index}/{len(segments)} target days loaded",
                file=sys.stderr,
                flush=True,
            )
    targets = np.concatenate(parts, axis=0)
    if not np.isfinite(targets).all() \
            or bool((targets < 0).any()) \
            or not np.allclose(
                targets.sum(axis=1), 1, atol=2e-4, rtol=2e-4
            ):
        raise ValueError(f"{split} oracle targets are invalid")
    targets /= targets.sum(axis=1, keepdims=True)
    return targets


def train_source_fingerprint(segments: Iterable[object]) -> str:
    """Fingerprint only the train references and purged row slices."""

    digest = hashlib.sha256()
    for segment in segments:
        reference = json.loads(segment.target_file.read_text(encoding="utf-8"))
        content_hash = reference.get("object", {}).get("contentHash")
        if not isinstance(content_hash, str) or len(content_hash) != 64:
            raise ValueError("training target reference lacks a content hash")
        digest.update(
            (
                f"{segment.target_file.stem}:{content_hash}:"
                f"{segment.target_row_offset}:{segment.count}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def write_prototype_artifact(
    repo_root: Path,
    reference_file: Path,
    fitted: FitResult,
    *,
    k: int,
    seed: int,
    lloyd_iterations: int,
    full_polish_passes: int,
    train_targets: np.ndarray,
    train_segments: Iterable[object],
) -> dict[str, object]:
    """Install one train-only, content-addressed fixed prototype matrix."""

    storage_root = (repo_root / "data/training/immutable").resolve()
    reference_root = storage_root / "refs"
    requested = (
        reference_file
        if reference_file.is_absolute()
        else repo_root / reference_file
    ).resolve()
    try:
        relative = requested.relative_to(reference_root)
    except ValueError as error:
        raise ValueError(
            f"artifact reference must be under {reference_root}"
        ) from error
    if relative.suffix != ".json" or len(relative.parts) < 2:
        raise ValueError(
            "artifact reference must be a namespaced .json file"
        )
    without_suffix = relative.with_suffix("")
    namespace = "/".join(without_suffix.parts[:-1])
    key = without_suffix.parts[-1]
    prototypes = np.ascontiguousarray(fitted.prototypes, dtype="<f4")
    if prototypes.shape != (k, ACTION_COUNT):
        raise ValueError("prototype artifact matrix shape is invalid")
    payload = prototypes.tobytes(order="C")
    segments = list(train_segments)
    metadata = {
        "artifactContract": "fixed-soft-oracle-policy-prototype-basis-v1",
        "fitSplit": "train",
        "validationUsedForFit": False,
        "testReferencesOpened": 0,
        "testPayloadsOpened": 0,
        "geometry": "forward-kl-bregman",
        "centroid": "arithmetic-mean-soft-policy",
        "modelUse": "continuous-simplex-weights-times-fixed-prototypes",
        "hardAssignmentsUsedForSupervision": False,
        "sourceNamespace": (
            "oracle/1s/hindsight-bot-71391c44b323e044e6ab"
        ),
        "sourceTemperature": SOURCE_TEMPERATURE,
        "sourceTrainFingerprintSha256": train_source_fingerprint(segments),
        "trainRowsAfterPurge": int(train_targets.shape[0]),
        "trainFileCount": len(segments),
        "trainDateStart": segments[0].target_file.stem,
        "trainDateEnd": segments[-1].target_file.stem,
        "fitAlgorithm": "deterministic-kl-plus-plus-lloyd-full-polish-v1",
        "seed": seed,
        "fitSampleRows": fitted.fit_sample_rows,
        "sampleLloydIterationsRequested": lloyd_iterations,
        "sampleLloydMeanKl": list(fitted.sample_objective),
        "fullTrainPolishPasses": full_polish_passes,
        "fullTrainPolishMeanKl": list(fitted.full_polish_objective),
        "trainPrototypeMixtureWeights": fitted.train_weights.tolist(),
        "matrixSha256": hashlib.sha256(payload).hexdigest(),
    }
    installed = write_shard_payload(
        storage_root,
        namespace,
        key,
        payload,
        sequence={
            "start": 0,
            "step": 1,
            "count": k,
            "unit": "index",
        },
        layout={
            "encoding": "raw-row-major",
            "dtype": "float32-le",
            "rows": k,
            "columns": ACTION_COUNT,
            "rowMeaning": "soft-oracle-action-probability-prototype",
        },
        metadata=metadata,
        compression_level=9,
    )
    reference = json.loads(installed.read_text(encoding="utf-8"))
    content_hash = reference["object"]["contentHash"]
    if content_hash != metadata["matrixSha256"]:
        raise RuntimeError("prototype artifact content hash changed on install")
    return {
        "referenceFile": str(installed),
        "contentHash": content_hash,
        "sourceTrainFingerprintSha256": metadata[
            "sourceTrainFingerprintSha256"
        ],
        "matrixShape": [k, ACTION_COUNT],
        "matrixDtype": "float32-le",
    }


def normalized_prototypes(values: np.ndarray) -> np.ndarray:
    result = np.maximum(
        np.asarray(values, dtype=np.float64),
        PROBABILITY_FLOOR,
    )
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def deterministic_sample(values: np.ndarray, count: int) -> np.ndarray:
    if count >= values.shape[0]:
        return values.copy()
    indexes = (
        (np.arange(count, dtype=np.int64) * values.shape[0]) // count
    )
    return values[indexes].copy()


def row_entropy(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(values > 0, values * np.log(values), 0)
    return -terms.sum(axis=1, dtype=np.float64)


def assign_prototypes(
    values: np.ndarray,
    prototypes: np.ndarray,
    *,
    chunk_rows: int = 16_384,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty(values.shape[0], dtype=np.int32)
    cross_entropy = np.empty(values.shape[0], dtype=np.float64)
    log_prototypes = np.log(
        np.maximum(prototypes, PROBABILITY_FLOOR)
    ).astype(np.float32)
    for start in range(0, values.shape[0], chunk_rows):
        end = min(start + chunk_rows, values.shape[0])
        scores = -(values[start:end] @ log_prototypes.T)
        current = scores.argmin(axis=1)
        labels[start:end] = current
        cross_entropy[start:end] = scores[
            np.arange(end - start), current
        ]
    return labels, cross_entropy


def centroid_update(
    values: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(labels, kind="stable")
    ordered_labels = labels[order]
    starts = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(ordered_labels[1:] != ordered_labels[:-1]) + 1,
    ))
    populated = ordered_labels[starts]
    sums = np.zeros((k, values.shape[1]), dtype=np.float64)
    sums[populated] = np.add.reduceat(
        values[order].astype(np.float64, copy=False),
        starts,
        axis=0,
    )
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    if bool((counts == 0).any()):
        raise RuntimeError("KL prototype fit produced an empty cluster")
    prototypes = sums / counts[:, None]
    return normalized_prototypes(prototypes), counts


def initialize_kl_plus_plus(
    values: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    prototypes = np.empty((k, values.shape[1]), dtype=np.float32)
    prior = normalized_prototypes(values.mean(axis=0, keepdims=True))[0]
    prior_ce = -(values @ np.log(prior).astype(np.float32))
    first = int(np.argmin(prior_ce))
    prototypes[0] = values[first]
    entropy = row_entropy(values)
    closest = np.full(values.shape[0], np.inf, dtype=np.float64)
    for index in range(k):
        if index > 0:
            total = closest.sum(dtype=np.float64)
            if not math.isfinite(total) or total <= 0:
                raise RuntimeError("KL++ initialization lost all distance")
            selected = int(rng.choice(values.shape[0], p=closest / total))
            prototypes[index] = values[selected]
        current = normalized_prototypes(prototypes[index:index + 1])[0]
        current_ce = -(values @ np.log(current).astype(np.float32))
        current_kl = np.maximum(current_ce - entropy, 0)
        closest = np.minimum(closest, current_kl)
    return normalized_prototypes(prototypes)


def fit_kl_prototypes(
    train_targets: np.ndarray,
    k: int,
    *,
    sample_rows: int,
    lloyd_iterations: int,
    full_polish_passes: int,
    seed: int,
) -> FitResult:
    sample = deterministic_sample(train_targets, sample_rows)
    entropy = row_entropy(sample)
    prototypes = initialize_kl_plus_plus(
        sample,
        k,
        np.random.default_rng(seed),
    )
    sample_trace: list[float] = []
    previous_labels: np.ndarray | None = None
    for _iteration in range(lloyd_iterations):
        labels, cross_entropy = assign_prototypes(sample, prototypes)
        objective = float(np.mean(cross_entropy - entropy))
        sample_trace.append(max(objective, 0.0))
        prototypes, _counts = centroid_update(sample, labels, k)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        previous_labels = labels

    full_trace: list[float] = []
    full_entropy = row_entropy(train_targets)
    counts: np.ndarray | None = None
    for _pass in range(full_polish_passes):
        labels, cross_entropy = assign_prototypes(train_targets, prototypes)
        full_trace.append(max(float(np.mean(cross_entropy - full_entropy)), 0.0))
        prototypes, counts = centroid_update(train_targets, labels, k)
    if counts is None:
        labels, cross_entropy = assign_prototypes(train_targets, prototypes)
        full_trace.append(max(float(np.mean(cross_entropy - full_entropy)), 0.0))
        prototypes, counts = centroid_update(train_targets, labels, k)
    weights = counts.astype(np.float64) / counts.sum()
    return FitResult(
        prototypes=prototypes,
        train_weights=weights,
        sample_objective=tuple(sample_trace),
        full_polish_objective=tuple(full_trace),
        fit_sample_rows=sample.shape[0],
    )


def nearest_prototype_kl(
    targets: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[float, np.ndarray]:
    labels, cross_entropy = assign_prototypes(targets, prototypes)
    value = float(np.mean(cross_entropy - row_entropy(targets)))
    return max(value, 0.0), labels


def frank_wolfe_kl_projection(
    targets: np.ndarray,
    prototypes: np.ndarray,
    *,
    initial_labels: np.ndarray,
    iterations: int,
    line_search_steps: int,
) -> dict[str, object]:
    """Approximately I-project every target onto the prototype convex hull.

    The final Frank-Wolfe gap is a convex optimality certificate.  The achieved
    KL is an upper bound on irreducible representation error; subtracting the
    mean gap gives a (possibly loose) lower bound on the optimum mean KL.
    """

    target = np.asarray(targets, dtype=np.float32)
    basis = normalized_prototypes(prototypes)
    prediction = basis[initial_labels].copy()
    entropy = row_entropy(target)
    objective_trace: list[dict[str, float | int]] = []
    gap = np.full(target.shape[0], np.inf, dtype=np.float32)
    for iteration in range(iterations):
        ratio = target / np.maximum(prediction, PROBABILITY_FLOOR)
        scores = ratio @ basis.T
        selected = scores.argmax(axis=1)
        gap = np.maximum(scores[np.arange(target.shape[0]), selected] - 1, 0)
        candidate = basis[selected]
        delta = candidate - prediction

        low = np.zeros(target.shape[0], dtype=np.float32)
        high = np.ones(target.shape[0], dtype=np.float32)
        # Convex one-dimensional exact line search by derivative bisection.
        for _step in range(line_search_steps):
            middle = (low + high) * 0.5
            mixed = prediction + middle[:, None] * delta
            derivative = -np.sum(
                target * delta / np.maximum(mixed, PROBABILITY_FLOOR),
                axis=1,
            )
            negative = derivative < 0
            low[negative] = middle[negative]
            high[~negative] = middle[~negative]
        alpha = (low + high) * 0.5
        alpha[gap <= 1e-7] = 0
        prediction += alpha[:, None] * delta
        prediction = np.maximum(prediction, PROBABILITY_FLOOR)
        prediction /= prediction.sum(axis=1, keepdims=True)

        if iteration in {0, 1, 3, 7, 15, iterations - 1}:
            current_kl = float(np.mean(
                -np.sum(
                    target * np.log(np.maximum(prediction, PROBABILITY_FLOOR)),
                    axis=1,
                    dtype=np.float64,
                ) - entropy
            ))
            objective_trace.append({
                "iteration": iteration + 1,
                "meanKl": max(current_kl, 0.0),
                "meanPreUpdateFrankWolfeGap": float(np.mean(gap)),
            })

    ratio = target / np.maximum(prediction, PROBABILITY_FLOOR)
    final_scores = ratio @ basis.T
    final_gap = np.maximum(final_scores.max(axis=1) - 1, 0)
    return {
        "prediction": prediction,
        "meanGap": float(np.mean(final_gap)),
        "p95Gap": float(np.quantile(final_gap, 0.95)),
        "objectiveTrace": objective_trace,
    }


if __name__ == "__main__":
    main()
