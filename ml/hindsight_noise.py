"""One-pass hindsight curricula and residual-readout configuration.

The model predicts clean standardized features, not epsilon. At variance one
the side input is exactly independent Gaussian noise (no residual target).
Forecast evaluation never calls the corruption function with held-out targets.
The curriculum gate uses training targets only; a separately labeled held-out
reconstruction diagnostic never controls that gate or checkpoint selection.
The separate frozen-teacher embedding variant linearly replaces target
embeddings with forecast component centers, using the same scalar gate.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


CONDITIONING_TYPE = "layer2-register-residual-hindsight-v1"
TEACHER_REPRESENTATION = "frozen-teacher-component-embedding"
SCHEDULE_TYPE = "cosine-noise-variance-v1"
CORRELATION_SCHEDULE_TYPE = "correlation-gated-noise-variance-v1"
TARGET_FREE_SCHEDULE_TYPE = "fixed-target-free-v1"


def initial_hindsight_curriculum(schedule: dict | None) -> dict | None:
    if schedule is None or schedule.get("type") != CORRELATION_SCHEDULE_TYPE:
        return None
    return {"noiseStep": 0, "lastCompletedEpoch": -1, "lastGateCorrelation": None}


def hindsight_variance_for_epoch(
    schedule: dict, epoch: int, curriculum: dict | None,
) -> float:
    if schedule["type"] == TARGET_FREE_SCHEDULE_TYPE:
        return 1.0
    if schedule["type"] == SCHEDULE_TYPE:
        return noise_variance_at_epoch(epoch, int(schedule["endEpoch"]))
    if schedule["type"] != CORRELATION_SCHEDULE_TYPE or curriculum is None:
        raise ValueError("missing correlation-gated hindsight curriculum")
    if curriculum["lastCompletedEpoch"] != epoch - 1:
        raise ValueError("hindsight curriculum epoch is inconsistent with checkpoint")
    step = curriculum["noiseStep"]
    if not isinstance(step, int) or step < 0:
        raise ValueError("invalid hindsight noise step")
    # Integer steps avoid repeated floating-point addition and cap at pure noise.
    return min(1.0, round(
        float(schedule["startVariance"]) + step * float(schedule["varianceIncrement"]),
        12,
    ))


def advance_hindsight_curriculum(
    schedule: dict, curriculum: dict, *, epoch: int, correlation: float | None,
) -> dict:
    variance = hindsight_variance_for_epoch(schedule, epoch, curriculum)
    finite_correlation = (
        float(correlation) if correlation is not None and math.isfinite(correlation)
        else None
    )
    passed = finite_correlation is not None \
        and finite_correlation >= float(schedule["requiredCorrelation"])
    return {
        "noiseStep": curriculum["noiseStep"] + int(passed and variance < 1.0),
        "lastCompletedEpoch": epoch,
        "lastGateCorrelation": finite_correlation,
    }


def noise_variance_at_epoch(epoch: int, end_epoch: int) -> float:
    if epoch < 0 or end_epoch <= 0:
        raise ValueError("hindsight epoch must be nonnegative and endpoint positive")
    if epoch >= end_epoch:
        return 1.0
    return math.sin(math.pi * 0.5 * epoch / end_epoch) ** 2


def corrupt_standardized_hindsight(
    clean: Tensor, noise: Tensor, variance: float,
) -> Tensor:
    if clean.shape != noise.shape or not 0.0 <= variance <= 1.0:
        raise ValueError("invalid hindsight shape or noise variance")
    if variance == 1.0:
        # Deliberately do not read clean: even 0*NaN would contaminate noise.
        return noise
    if variance == 0.0:
        return clean.detach()
    return math.sqrt(1.0 - variance) * clean.detach() + math.sqrt(variance) * noise


def evaluation_noise_rng(config: dict, split: str) -> np.random.Generator:
    # Separate fixed streams; restarting evaluation never advances training RNG.
    offset = {"train": 0, "validation": 1, "test": 2}[split]
    return np.random.default_rng(int(config["evaluationSeed"]) + offset)


def gaussian_hindsight(
    rng: np.random.Generator, batch: int, steps: int, features: int,
    device: torch.device, *, samples: int = 1,
) -> tuple[Tensor, Tensor]:
    # NumPy's flat stream is independent of evaluation batch partitioning.
    shape = (batch, steps, features) if samples == 1 else (batch, steps, samples, features)
    values = rng.standard_normal(shape, dtype=np.float32)
    return (
        torch.from_numpy(values).to(device),
        torch.ones((batch, 1), dtype=torch.float32, device=device),
    )


def expand_hindsight_samples(clean: Tensor, samples: int) -> Tensor:
    """Copies of one target; callers generate independent noise for each copy."""
    return clean if samples == 1 else clean.unsqueeze(2).expand(-1, -1, samples, -1)


def validate_hindsight_readout(config: dict) -> None:
    if not isinstance(config, dict) or config.get("type") != CONDITIONING_TYPE:
        raise ValueError("invalid hindsight conditioning")
    if config.get("injectionLayer") != 2 \
            or config.get("registerLayer") != 12 \
            or config.get("registerInitialization") != "learned-zero-vector" \
            or config.get("carryRegisters") is not False:
        raise ValueError("hindsight requires reset learned registers and layer 2")
    if type(config.get("layer13Enabled")) is not bool:
        raise ValueError("layer13Enabled must be boolean")
    if type(config.get("sampleCount")) is not int or config["sampleCount"] < 1 \
            or config.get("sampleRepresentation") not in (
                "full-output-feature-vector", TEACHER_REPRESENTATION,
            ):
        raise ValueError("invalid hindsight sample representation")
    if config["sampleRepresentation"] == TEACHER_REPRESENTATION:
        if config["sampleCount"] <= 1 or type(config.get("sampleInputWidth")) is not int \
                or config["sampleInputWidth"] <= 0:
            raise ValueError("teacher embeddings require compressed samples and an input width")
    if config["sampleCount"] > 1:
        if config.get("sampleEmbeddingWidth") != 256 \
                or config.get("sampleCompression") != "concatenated-affine-v1" \
                or config.get("sampleObjective") != "mean-per-sample-mse" \
                or config.get("samplePrediction") != "arithmetic-mean":
            raise ValueError("multiple samples require affine-256 compression and per-sample supervision")
    elif "sampleEmbeddingWidth" in config:
        raise ValueError("single-sample baseline must remain uncompressed")
    for key in ("registerWidth", "residualHiddenWidth"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError("hindsight register and hidden widths must be positive integers")
    passes = config.get("residualPasses", 1)
    if type(passes) is not int or passes <= 0:
        raise ValueError("hindsight residual passes must be a positive integer")
    if passes > 1 and config.get("residualParameterSharing") != "across-forecast-steps-only":
        raise ValueError("hindsight residual passes must be independent, shared only across forecast steps")
    if config["layer13Enabled"]:
        count = config.get("layer13Count", 1)
        if type(count) is not int or count < 1:
            raise ValueError("layer 13 count must be a positive integer")
        if passes != 1 or config.get("residualParameterSharing") != "across-forecast-steps-only":
            raise ValueError("layer 13 replaces the second layer-2 pass, with independent weights")
        width = config.get("layer13HiddenWidth")
        if type(width) is not int or width <= 0:
            raise ValueError("layer 13 hidden width must be a positive integer")
    elif config.get("layer13Count", 0) != 0:
        raise ValueError("disabled layer 13 cannot have additional blocks")
    if not 0 < float(config.get("residualProjectionInitScale", math.nan)) <= 0.01:
        raise ValueError("hindsight requires a small initial residual projection")


def validate_hindsight_plan(plan: dict) -> None:
    architecture = plan["architecture"]
    config = architecture.get("hindsightConditioning")
    schedule = plan["training"].get("hindsightNoiseSchedule")
    if config is None and schedule is None:
        return
    validate_hindsight_readout(config)
    teacher = plan.get("hindsightTeacher")
    if config["sampleRepresentation"] == TEACHER_REPRESENTATION:
        if not isinstance(teacher, dict) \
                or teacher.get("type") != "frozen-base-support-component-centers-v1" \
                or teacher.get("embeddingWidth") != config["sampleInputWidth"] \
                or teacher.get("blend") != "linear-embedding-replacement" \
                or teacher.get("gaussianJitter") != 0 \
                or int(teacher.get("inferenceBatchSize", 0)) <= 0 \
                or len(teacher.get("checkpointSha256", "")) != 64 \
                or len(teacher.get("planSha256", "")) != 64:
            raise ValueError("invalid frozen hindsight teacher")
    elif teacher is not None:
        raise ValueError("teacher requires the teacher embedding representation")
    if not isinstance(schedule, dict) or schedule.get("type") not in (
        SCHEDULE_TYPE, CORRELATION_SCHEDULE_TYPE,
    ):
        raise ValueError("invalid hindsight noise schedule")
    if schedule["type"] == SCHEDULE_TYPE:
        if int(schedule.get("endEpoch", 0)) <= 0 \
                or int(schedule["endEpoch"]) >= int(plan["training"]["epochs"]):
            raise ValueError("hindsight curriculum needs a pure-noise training phase")
    else:
        threshold = float(schedule.get("requiredCorrelation", math.nan))
        increment = float(schedule.get("varianceIncrement", math.nan))
        if not 0.0 < threshold < 1.0 or not 0.0 < increment <= 1.0 \
                or schedule.get("probeSplit") != "train" \
                or schedule.get("probeMetric") != "nextReturn.correlation" \
                or schedule.get("probeWeightSource") != "raw-training-weights" \
                or int(schedule.get("probeExamples", 0)) < 2 \
                or not isinstance(schedule.get("probeSeed"), int) \
                or schedule["probeSeed"] < 0:
            raise ValueError("invalid hindsight correlation gate")
    if schedule.get("startEpoch") != 0 or schedule.get("startVariance") != 0.0 \
            or schedule.get("endVariance") != 1.0:
        raise ValueError("hindsight schedule must run from variance zero to one")
    if config.get("evaluationVariance") != 1.0 \
            or not isinstance(config.get("evaluationSeed"), int) \
            or config["evaluationSeed"] < 0:
        raise ValueError("hindsight evaluation must use the seeded target-free endpoint")
    if plan["training"]["loss"]["type"] != "mean-standardized-next-feature-mse-v1" \
            or plan.get("datasetContract") is not None \
            or architecture.get("returnDensity") is not None \
            or architecture.get("featureEmbeddingDensity") is not None \
            or architecture.get("recurrentMemory") is not None \
            or architecture.get("linearFactorization") is not None \
            or architecture.get("layer8FunctionApproximator") is not None:
        raise ValueError("hindsight experiment requires direct feature MSE")
