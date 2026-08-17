from __future__ import annotations

import argparse
import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from eiil import (
    infer_binary_environment_ids,
    irm_v1_regression_objective,
    regression_scale_gradients,
)

from next_return_dataset import DAY_SECONDS, ExampleShard
from normalized_glu_next_return import (
    CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
    NormalizedGluNextReturn,
    TRAINING_POSITION_INPUT_NORMALIZATION,
    depth_width_parameter_assignments,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)
from train_autoregressive_minute_return import (
    build_optimizers,
    direct_calendar_shards,
    training_normalization,
)
from train_normalized_glu_next_return import (
    NextReturnDataset,
    MetricAccumulator,
    Reporter,
    atomic_json,
    canonical_fingerprint,
    corpus_fingerprint,
    evaluate,
    evaluate_daily_group_cvar,
    iter_device_batches,
    resolve,
    uniform_group_cvar,
)
from swa import (
    EpochSwaSweep,
    swa_config,
    validate_swa_config,
)


RUNNER_CONTRACT = "next-second-fixed-subset-memorization-no-regularization-v1"
DROPOUT_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-dropout-regularization-v1"
)
CAUSAL_VOLATILITY_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-causal-volatility-v1"
)
SWA_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-swa-sweep-v1"
)
L2_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-explicit-l2-v1"
)
OPTIMIZER_WEIGHT_DECAY_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-optimizer-weight-decay-v1"
)
SAM_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-sharpness-aware-minimization-v1"
)
ADVERSARIAL_INPUT_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-adversarial-input-v1"
)
ADVERSARIAL_RETURN_VECTOR_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-adversarial-return-vector-v1"
)
SAM_ADVERSARIAL_RETURN_VECTOR_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-sam-adversarial-return-vector-v1"
)
ADVERSARIAL_LOG_PRICE_PATH_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-adversarial-log-price-path-v1"
)
ADVERSARIAL_OBSERVED_LOG_PRICE_PATH_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-adversarial-observed-log-price-path-v1"
)
ROBUST_REGRESSION_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-robust-regression-v1"
)
NONZERO_TARGET_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-nonzero-targets-v1"
)
CVAR_DRO_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-nonzero-daily-cvar-dro-v1"
)
EIIL_IRM_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-nonzero-eiil-irmv1-v1"
)
MEAN_TEACHER_RUNNER_CONTRACT = (
    "next-second-fixed-subset-memorization-nonzero-mean-teacher-v1"
)
PAUSE_EXIT_CODE = 75


def runner_contract(plan: dict) -> str:
    if plan["training"].get("meanTeacher") is not None:
        return MEAN_TEACHER_RUNNER_CONTRACT
    if plan["training"].get("environmentInference") is not None:
        return EIIL_IRM_RUNNER_CONTRACT
    if plan["training"].get("cvarDro") is not None:
        return CVAR_DRO_RUNNER_CONTRACT
    if plan.get("datasetFilter") is not None:
        return NONZERO_TARGET_RUNNER_CONTRACT
    if plan["training"].get("robustRegression") is not None:
        return ROBUST_REGRESSION_RUNNER_CONTRACT
    if plan["training"].get("sam") is not None \
            and plan["training"].get("adversarialReturnVector") is not None:
        return SAM_ADVERSARIAL_RETURN_VECTOR_RUNNER_CONTRACT
    if plan["training"].get("adversarialReturnVector") is not None:
        return ADVERSARIAL_RETURN_VECTOR_RUNNER_CONTRACT
    adversarial_log_price_path = plan["training"].get(
        "adversarialLogPricePath"
    )
    if adversarial_log_price_path is not None:
        return (
            ADVERSARIAL_OBSERVED_LOG_PRICE_PATH_RUNNER_CONTRACT
            if adversarial_log_price_path.get("space")
            == "reconstructed-observed-log-price-path-fixed-future-endpoint"
            else ADVERSARIAL_LOG_PRICE_PATH_RUNNER_CONTRACT
        )
    if plan["training"].get("adversarialInput") is not None:
        return ADVERSARIAL_INPUT_RUNNER_CONTRACT
    if plan["training"].get("sam") is not None:
        return SAM_RUNNER_CONTRACT
    if plan["training"].get("optimizerWeightDecay") is not None:
        return OPTIMIZER_WEIGHT_DECAY_RUNNER_CONTRACT
    if plan["training"].get("l2Regularization") is not None:
        return L2_RUNNER_CONTRACT
    if plan["training"].get("swa") is not None:
        return SWA_RUNNER_CONTRACT
    if plan["architecture"].get(
        "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
    ) == CAUSAL_VOLATILITY_INPUT_NORMALIZATION:
        return CAUSAL_VOLATILITY_RUNNER_CONTRACT
    return (
        RUNNER_CONTRACT
        if float(plan["architecture"].get("dropout", 0)) == 0
        else DROPOUT_RUNNER_CONTRACT
    )


def robust_regression_variant(
    source: dict,
    *,
    loss_type: str,
    parameter: float,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("robust-regression variant suffix is invalid")
    if loss_type not in {"huber", "student-t"}:
        raise ValueError("robust loss must be huber or student-t")
    if not math.isfinite(parameter) or parameter <= 0:
        raise ValueError("robust-loss parameter must be finite and positive")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    parameter_name = "delta" if loss_type == "huber" else "degreesOfFreedom"
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - robust {loss_type} '
        f'{parameter_name} {parameter:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["robustRegression"] = {
        "type": loss_type,
        parameter_name: parameter,
        "residualSpace": "training-target-standard-deviations",
        "smallResidualScale": "mse-equivalent",
    }
    return plan


def nonzero_target_variant(source: dict, *, suffix: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("nonzero-target variant suffix is invalid")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - exact-zero targets excluded'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["datasetFilter"] = {
        "type": "exclude-exact-zero-target-return",
        "appliesTo": ["normalization", "training", "validation", "calibration", "test"],
        "comparison": "float32-exact-zero-after-log-return-construction",
    }
    return plan


def cvar_dro_variant(
    source: dict, *, tail_fraction: float, suffix: str
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("CVaR-DRO variant suffix is invalid")
    if not math.isfinite(tail_fraction) or not 0 < tail_fraction <= 1:
        raise ValueError("CVaR-DRO tail fraction must be in (0, 1]")
    plan = nonzero_target_variant(source, suffix=suffix)
    plan["label"] = (
        f'{source.get("label", source["id"])} - nonzero targets, daily '
        f'CVaR-DRO worst {100 * tail_fraction:g}%'
    )
    plan["training"]["cvarDro"] = {
        "type": "conditional-value-at-risk",
        "loss": "normalized-mse",
        "group": "utc-calendar-day",
        "groupWeighting": "uniform",
        "tailFraction": tail_fraction,
        "batchConstruction": "globally-shuffled-across-groups",
    }
    return plan


def eiil_irm_variant(
    source: dict,
    *,
    reference_plan: str,
    penalty_weight: float,
    penalty_anneal_steps: int,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("EIIL variant suffix is invalid")
    if not reference_plan or Path(reference_plan).is_absolute():
        raise ValueError("EIIL reference plan must be a repo-relative path")
    if not math.isfinite(penalty_weight) or penalty_weight <= 0:
        raise ValueError("IRM penalty weight must be finite and positive")
    if isinstance(penalty_anneal_steps, bool) or penalty_anneal_steps < 0:
        raise ValueError("IRM penalty anneal steps must be non-negative")
    plan = nonzero_target_variant(source, suffix=suffix)
    plan["label"] = (
        f'{source.get("label", source["id"])} - nonzero targets, '
        "EIIL-inferred environments + IRMv1 "
        f'(penalty {penalty_weight:g}, anneal '
        f'{penalty_anneal_steps:,} optimizer steps)'
    )
    plan["training"]["environmentInference"] = {
        "type": "eiil",
        "environmentCount": 2,
        "referencePlan": reference_plan,
        "referenceCheckpoint": "best",
        "objective": "maximize-irmv1-output-scale-gradient",
        "assignment": "analytic-hard-optimum-scalar-regression",
    }
    plan["training"]["invariantRiskMinimization"] = {
        "type": "irmv1",
        "loss": "normalized-mse",
        "environmentWeighting": "uniform",
        "penalty": "squared-output-scale-gradient-at-one",
        "penaltyWeight": float(penalty_weight),
        "penaltyAnnealSteps": int(penalty_anneal_steps),
        "largePenaltyObjectiveRescaling": True,
        "batchConstruction": "globally-shuffled-across-environments",
    }
    return plan


def mean_teacher_variant(
    source: dict,
    *,
    half_life_epochs: float,
    consistency_weight: float,
    ramp_up_fraction: float,
    input_perturbation_rms: float,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("Mean Teacher variant suffix is invalid")
    if not math.isfinite(half_life_epochs) or half_life_epochs <= 0:
        raise ValueError("Mean Teacher half-life must be positive")
    if not math.isfinite(consistency_weight) or consistency_weight <= 0:
        raise ValueError("Mean Teacher consistency weight must be positive")
    if not math.isfinite(ramp_up_fraction) \
            or not 0 <= ramp_up_fraction <= 1:
        raise ValueError("Mean Teacher ramp-up fraction must be in [0, 1]")
    if not math.isfinite(input_perturbation_rms) \
            or not 0 < input_perturbation_rms <= 1:
        raise ValueError("Mean Teacher input perturbation RMS must be in (0, 1]")
    plan = nonzero_target_variant(source, suffix=suffix)
    plan["label"] = (
        f'{source.get("label", source["id"])} - nonzero targets, '
        f'Mean Teacher EMA half-life {half_life_epochs:g} epochs, '
        f'consistency weight {consistency_weight:g}, linear ramp '
        f'{ramp_up_fraction:g} training duration, Gaussian input RMS '
        f'{input_perturbation_rms:g}'
    )
    plan["training"]["meanTeacher"] = {
        "contract": "ema-self-distillation-regression-v1",
        "teacherUpdate": "ema-after-student-optimizer-step",
        "halfLifeEpochs": float(half_life_epochs),
        "consistencyLoss": "normalized-mse",
        "consistencyWeight": float(consistency_weight),
        "rampUp": "linear-by-optimizer-step",
        "rampUpFraction": float(ramp_up_fraction),
        "supervisedInput": "clean",
        "teacherInput": "clean",
        "studentConsistencyInput": {
            "type": "gaussian",
            "space": "training-position-normalized-input",
            "norm": "exact-per-example-rms",
            "epsilonRms": float(input_perturbation_rms),
        },
        "inferenceModel": "ema-teacher",
    }
    return plan


def with_validation_curve(
    plan: dict,
    *,
    split_plan: str,
    examples: int,
) -> dict:
    if not split_plan or Path(split_plan).is_absolute():
        raise ValueError("validation-curve split plan must be repo-relative")
    if isinstance(examples, bool) or examples < 1:
        raise ValueError("validation-curve example count must be positive")
    result = json.loads(json.dumps(plan))
    result["label"] = (
        f'{result.get("label", result["id"])} - per-epoch validation '
        f'curve ({examples:,} clean held-out examples)'
    )
    result["validationCurve"] = {
        "contract": "fixed-clean-heldout-subset-every-epoch-v1",
        "splitPlan": split_plan,
        "split": "validation",
        "examples": int(examples),
        "frequency": "every-epoch",
        "checkpointSelection": "training-objective-only",
        "fullValidationAfterTraining": True,
    }
    return result


def swa_variant(source: dict, *, suffix: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("SWA variant suffix is invalid")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - SWA sweep'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["swa"] = swa_config()
    return plan


def l2_variant(source: dict, *, rate: float, suffix: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("L2 variant suffix is invalid")
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("L2 rate must be finite and in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - explicit L2 rate {rate:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["l2Regularization"] = {
        "type": "explicit-loss-term",
        "coefficient": rate,
        "parameters": "all-trainable-matrices",
        "reduction": "half-sum-squared",
    }
    return plan


def optimizer_weight_decay_variant(
    source: dict, *, rate: float, suffix: str
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("optimizer weight-decay variant suffix is invalid")
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("optimizer weight-decay rate must be in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - optimizer weight decay {rate:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    optimizer = plan["training"]["optimizer"]
    optimizer["muon"]["weightDecay"] = rate
    optimizer["adamw"]["weightDecay"] = rate
    plan["training"]["optimizerWeightDecay"] = {
        "type": "optimizer-native",
        "coefficient": rate,
        "parameterGroups": ["muon", "adamw"],
    }
    return plan


def sam_variant(source: dict, *, rho: float, suffix: str) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("SAM variant suffix is invalid")
    if not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("SAM rho must be finite and in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - SAM rho {rho:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["sam"] = {
        "type": "sharpness-aware-minimization",
        "rho": rho,
        "adaptive": False,
        "gradientNorm": "global-l2",
        "perturbationUnit": "optimizer-update",
    }
    return plan


def adversarial_input_variant(
    source: dict,
    *,
    epsilon_rms: float,
    steps: int,
    adversarial_weight: float,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("adversarial-input variant suffix is invalid")
    if not math.isfinite(epsilon_rms) or not 0 < epsilon_rms <= 1:
        raise ValueError("adversarial-input epsilon RMS must be in (0, 1]")
    if isinstance(steps, bool) or not 1 <= steps <= 16:
        raise ValueError("adversarial-input steps must be in [1, 16]")
    if not math.isfinite(adversarial_weight) \
            or not 0 < adversarial_weight <= 1:
        raise ValueError("adversarial-input weight must be in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - adversarial input '
        f'epsilon RMS {epsilon_rms:g}, steps {steps}, '
        f'weight {adversarial_weight:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["adversarialInput"] = {
        "type": "projected-gradient-ascent",
        "space": "training-position-normalized-input",
        "norm": "rms-l2",
        "epsilonRms": epsilon_rms,
        "steps": steps,
        "stepSizeRms": epsilon_rms / steps,
        "randomStart": False,
        "adversarialWeight": adversarial_weight,
        "target": "unchanged-next-return",
    }
    return plan


def adversarial_return_vector_variant(
    source: dict,
    *,
    epsilon_rms: float,
    steps: int,
    adversarial_weight: float,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("adversarial return-vector variant suffix is invalid")
    if not math.isfinite(epsilon_rms) or not 0 < epsilon_rms <= 1:
        raise ValueError("adversarial return-vector epsilon RMS must be in (0, 1]")
    if isinstance(steps, bool) or not 1 <= steps <= 16:
        raise ValueError("adversarial return-vector steps must be in [1, 16]")
    if not math.isfinite(adversarial_weight) \
            or not 0 < adversarial_weight <= 1:
        raise ValueError("adversarial return-vector weight must be in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - adversarial input and output '
        f'log returns epsilon RMS {epsilon_rms:g}, steps {steps}, '
        f'weight {adversarial_weight:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["adversarialReturnVector"] = {
        "type": "projected-gradient-ascent",
        "space": "independent-normalized-input-and-output-log-returns",
        "norm": "rms-l2",
        "epsilonRms": epsilon_rms,
        "steps": steps,
        "stepSizeRms": epsilon_rms / steps,
        "randomStart": False,
        "adversarialWeight": adversarial_weight,
        "inputScale": "training-standard-deviation-per-lag-position",
        "targetScale": "training-target-return-standard-deviation",
        "returnCount": 121,
    }
    return plan


def sam_adversarial_return_vector_variant(
    source: dict,
    *,
    rho: float,
    epsilon_rms: float,
    steps: int,
    adversarial_weight: float,
    epochs: int,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("SAM + adversarial return-vector suffix is invalid")
    if not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("SAM rho must be finite and in (0, 1]")
    if not math.isfinite(epsilon_rms) or not 0 < epsilon_rms <= 1:
        raise ValueError("adversarial return-vector epsilon RMS must be in (0, 1]")
    if isinstance(steps, bool) or not 1 <= steps <= 16:
        raise ValueError("adversarial return-vector steps must be in [1, 16]")
    if not math.isfinite(adversarial_weight) \
            or not 0 < adversarial_weight <= 1:
        raise ValueError("adversarial return-vector weight must be in (0, 1]")
    if isinstance(epochs, bool) or not 1 <= epochs <= 100_000:
        raise ValueError("combined training epochs must be in [1, 100,000]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - SAM rho {rho:g} + '
        f'adversarial input/output log returns epsilon RMS {epsilon_rms:g}, '
        f'steps {steps}, weight {adversarial_weight:g}, epochs {epochs}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["epochs"] = epochs
    plan["training"]["sam"] = {
        "type": "sharpness-aware-minimization",
        "rho": rho,
        "adaptive": False,
        "gradientNorm": "global-l2",
        "perturbationUnit": "optimizer-update",
        "baseObjective": "clean-plus-adversarial-return-vector",
    }
    plan["training"]["adversarialReturnVector"] = {
        "type": "projected-gradient-ascent",
        "space": "independent-normalized-input-and-output-log-returns",
        "norm": "rms-l2",
        "epsilonRms": epsilon_rms,
        "steps": steps,
        "stepSizeRms": epsilon_rms / steps,
        "randomStart": False,
        "adversarialWeight": adversarial_weight,
        "inputScale": "training-standard-deviation-per-lag-position",
        "targetScale": "training-target-return-standard-deviation",
        "returnCount": 121,
        "regenerateAtSamPerturbedWeights": True,
    }
    return plan


def adversarial_log_price_path_variant(
    source: dict,
    *,
    epsilon_rms: float,
    steps: int,
    adversarial_weight: float,
    perturb_future_endpoint: bool = True,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("adversarial log-price-path variant suffix is invalid")
    if not math.isfinite(epsilon_rms) or not 0 < epsilon_rms <= 1:
        raise ValueError(
            "adversarial log-price-path epsilon RMS must be in (0, 1]"
        )
    if isinstance(steps, bool) or not 1 <= steps <= 16:
        raise ValueError("adversarial log-price-path steps must be in [1, 16]")
    if not math.isfinite(adversarial_weight) \
            or not 0 < adversarial_weight <= 1:
        raise ValueError("adversarial log-price-path weight must be in (0, 1]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    path_label = (
        "adversarial log-price path"
        if perturb_future_endpoint
        else "adversarial observed log-price path with fixed future endpoint"
    )
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - {path_label} '
        f'epsilon RMS {epsilon_rms:g}, steps {steps}, '
        f'weight {adversarial_weight:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["training"]["adversarialLogPricePath"] = {
        "type": "projected-gradient-ascent",
        "space": (
            "reconstructed-log-price-path-including-future-endpoint"
            if perturb_future_endpoint
            else "reconstructed-observed-log-price-path-fixed-future-endpoint"
        ),
        "norm": "rms-l2",
        "scale": "training-target-return-standard-deviation",
        "epsilonRms": epsilon_rms,
        "steps": steps,
        "stepSizeRms": epsilon_rms / steps,
        "randomStart": False,
        "adversarialWeight": adversarial_weight,
        "input": "recomputed-adjacent-log-price-differences",
        "target": (
            "recomputed-final-log-price-difference"
            if perturb_future_endpoint
            else "recomputed-final-log-price-difference-with-fixed-future-endpoint"
        ),
        "pathPoints": 122 if perturb_future_endpoint else 121,
        "futureEndpoint": (
            "perturbed" if perturb_future_endpoint else "fixed-observed"
        ),
    }
    return plan


def dropout_variant(
    source: dict,
    *,
    probability: float,
    application_rate: float,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("dropout variant suffix is invalid")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - dropout '
        f'p={probability:g}, application={application_rate:g}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["architecture"]["dropout"] = probability
    plan["architecture"]["dropoutRate"] = application_rate
    return plan


def causal_volatility_variant(
    source: dict,
    *,
    window: int,
    suffix: str,
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", suffix):
        raise ValueError("causal volatility variant suffix is invalid")
    if isinstance(window, bool) or not 1 <= window <= DAY_SECONDS:
        raise ValueError("causal volatility window must be in [1, 86,400]")
    plan = json.loads(json.dumps(source))
    plan["id"] = f'{plan["id"]}-{suffix}'
    plan["label"] = (
        f'{plan.get("label", plan["id"])} - causal volatility window {window}'
    )
    plan["datasetDir"] = f'{plan["datasetDir"]}-{suffix}'
    plan["runDir"] = f'{plan["runDir"]}-{suffix}'
    plan["architecture"]["inputNormalization"] = (
        CAUSAL_VOLATILITY_INPUT_NORMALIZATION
    )
    plan["architecture"]["volatilityWindow"] = window
    return plan


def fixed_subset_shards(
    history_root: Path,
    subset_date: date,
    examples: int,
) -> dict[str, list[ExampleShard]]:
    if examples < 1:
        raise ValueError("memorization subset examples must be positive")
    final_date = subset_date + timedelta(days=(examples - 1) // DAY_SECONDS)
    required = subset_date - timedelta(days=1)
    while required <= final_date + timedelta(days=1):
        if not (history_root / f"{required.isoformat()}.json").is_file():
            raise FileNotFoundError(
                f"memorization subset requires candle day {required.isoformat()}"
            )
        required += timedelta(days=1)
    train: list[ExampleShard] = []
    remaining = examples
    current = subset_date
    while remaining:
        count = min(remaining, DAY_SECONDS)
        timestamp = int(datetime(
            current.year,
            current.month,
            current.day,
            tzinfo=timezone.utc,
        ).timestamp() * 1_000)
        train.append(ExampleShard(
            split="train",
            decision_time_start=timestamp,
            count=count,
            date=current.isoformat(),
            row_offset=0,
        ))
        remaining -= count
        current += timedelta(days=1)
    return {
        "train": train,
        "validation": [],
        "test": [],
    }


def fixed_nonzero_subset_shards(
    history_root: Path,
    subset_date: date,
    examples: int,
) -> dict[str, list[ExampleShard]]:
    """Select exactly N nonzero targets from a contiguous candidate window."""
    if examples < 1:
        raise ValueError("nonzero memorization subset examples must be positive")
    probe = NextReturnDataset(
        {"train": [], "validation": [], "test": []},
        history_root,
        horizon_return_count=1,
    )
    train: list[ExampleShard] = []
    remaining = examples
    current = subset_date
    while remaining > 0:
        for required in (
            current - timedelta(days=1), current, current + timedelta(days=1)
        ):
            if not (history_root / f"{required.isoformat()}.json").is_file():
                raise FileNotFoundError(
                    "nonzero memorization subset requires candle day "
                    f"{required.isoformat()}"
                )
        _history, target = probe._component(current.isoformat())
        nonzero_rows = np.flatnonzero(target != 0)
        if nonzero_rows.size == 0:
            current += timedelta(days=1)
            continue
        if remaining <= nonzero_rows.size:
            candidate_count = int(nonzero_rows[remaining - 1]) + 1
            retained = remaining
        else:
            candidate_count = DAY_SECONDS
            retained = int(nonzero_rows.size)
        timestamp = int(datetime(
            current.year, current.month, current.day, tzinfo=timezone.utc
        ).timestamp() * 1_000)
        train.append(ExampleShard(
            split="train",
            decision_time_start=timestamp,
            count=candidate_count,
            date=current.isoformat(),
            row_offset=0,
        ))
        remaining -= retained
        current += timedelta(days=1)
    return {"train": train, "validation": [], "test": []}


def validate_plan(plan: dict) -> None:
    for name in (
        "id", "datasetDir", "runDir", "historyDir", "subset",
        "architecture", "training",
    ):
        if name not in plan or plan[name] in (None, ""):
            raise ValueError(f"memorization plan requires {name}")
    dataset_filter = plan.get("datasetFilter")
    if dataset_filter is not None and (
        dataset_filter.get("type") != "exclude-exact-zero-target-return"
        or dataset_filter.get("appliesTo") != [
            "normalization", "training", "validation", "calibration", "test"
        ]
        or dataset_filter.get("comparison")
        != "float32-exact-zero-after-log-return-construction"
    ):
        raise ValueError("nonzero-target dataset-filter contract is invalid")
    validation_curve = plan.get("validationCurve")
    if validation_curve is not None and (
        validation_curve.get("contract")
        != "fixed-clean-heldout-subset-every-epoch-v1"
        or not str(validation_curve.get("splitPlan", ""))
        or Path(str(validation_curve.get("splitPlan", ""))).is_absolute()
        or validation_curve.get("split") != "validation"
        or not isinstance(validation_curve.get("examples"), int)
        or isinstance(validation_curve.get("examples"), bool)
        or int(validation_curve["examples"]) < 1
        or validation_curve.get("frequency") != "every-epoch"
        or validation_curve.get("checkpointSelection")
        != "training-objective-only"
        or validation_curve.get("fullValidationAfterTraining") is not True
        or dataset_filter is None
    ):
        raise ValueError("validation-curve contract is invalid")
    subset = plan["subset"]
    subset_type = str(subset.get("type", "fixed-contiguous"))
    if subset_type == "fixed-contiguous":
        date.fromisoformat(str(subset["date"]))
        if int(subset["examples"]) < 1:
            raise ValueError("memorization subset size is invalid")
    elif subset_type == "calendar-training-split":
        split = subset.get("split", {})
        for name in (
            "trainStart", "trainEnd", "validationStart", "validationEnd",
            "testStart", "testEnd",
        ):
            date.fromisoformat(str(split[name]))
    else:
        raise ValueError("memorization subset type is invalid")
    architecture = plan["architecture"]
    input_normalization = architecture.get(
        "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
    )
    volatility_window = architecture.get("volatilityWindow")
    widths = architecture.get("widths")
    if not isinstance(widths, list) \
            or not widths \
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 2
                for value in widths
            ) \
            or not 0 <= float(architecture.get("dropout", -1)) < 1 \
            or not 0 <= float(architecture.get("dropoutRate", -1)) <= 1 \
            or (
                float(architecture.get("dropout", 0)) > 0
                and float(architecture.get("dropoutRate", 0)) == 0
            ) \
            or not isinstance(architecture.get("learnableCentering", True), bool):
        raise ValueError(
            "memorization diagnostic architecture is invalid"
        )
    if input_normalization == CAUSAL_VOLATILITY_INPUT_NORMALIZATION:
        if not isinstance(volatility_window, int) \
                or isinstance(volatility_window, bool) \
                or not 1 <= volatility_window <= DAY_SECONDS:
            raise ValueError(
                "causal volatility window must be in [1, 86,400]"
            )
    elif input_normalization != TRAINING_POSITION_INPUT_NORMALIZATION \
            or volatility_window is not None:
        raise ValueError("memorization input normalization is invalid")
    training = plan["training"]
    if int(training.get("epochs", 0)) < 1 \
            or int(training.get("batchSize", 0)) < 1 \
            or int(training.get("evaluationBatchSize", 0)) < 1 \
            or float(training.get("learningRate", 0)) <= 0 \
            or not 0 < float(training.get("targetNormalizedMse", 0)) < 1 \
            or training.get("mixedPrecision") != "float32" \
            or training.get("device") not in {"cpu", "cuda"}:
        raise ValueError("memorization training settings are invalid")
    if "learningRateSchedule" in training or "earlyStoppingPatience" in training:
        raise ValueError("memorization diagnostic must not use validation controls")
    robust_regression = training.get("robustRegression")
    if robust_regression is not None:
        loss_type = robust_regression.get("type")
        parameter_name = "delta" if loss_type == "huber" else "degreesOfFreedom"
        parameter = float(robust_regression.get(parameter_name, 0))
        if loss_type not in {"huber", "student-t"} \
                or robust_regression.get("residualSpace") \
                != "training-target-standard-deviations" \
                or robust_regression.get("smallResidualScale") \
                != "mse-equivalent" \
                or not math.isfinite(parameter) \
                or parameter <= 0:
            raise ValueError("robust-regression contract is invalid")
    cvar_dro = training.get("cvarDro")
    if cvar_dro is not None and (
        cvar_dro.get("type") != "conditional-value-at-risk"
        or cvar_dro.get("loss") != "normalized-mse"
        or cvar_dro.get("group") != "utc-calendar-day"
        or cvar_dro.get("groupWeighting") != "uniform"
        or cvar_dro.get("batchConstruction")
        != "globally-shuffled-across-groups"
        or not math.isfinite(float(cvar_dro.get("tailFraction", 0)))
        or not 0 < float(cvar_dro["tailFraction"]) <= 1
        or dataset_filter is None
    ):
        raise ValueError("CVaR-DRO contract is invalid")
    environment_inference = training.get("environmentInference")
    invariant_risk = training.get("invariantRiskMinimization")
    if (environment_inference is None) != (invariant_risk is None):
        raise ValueError("EIIL inference and IRMv1 training must be configured together")
    if environment_inference is not None:
        reference_plan = str(environment_inference.get("referencePlan", ""))
        penalty_weight = float(invariant_risk.get("penaltyWeight", 0))
        penalty_anneal_steps = invariant_risk.get("penaltyAnnealSteps")
        if environment_inference.get("type") != "eiil" \
                or environment_inference.get("environmentCount") != 2 \
                or not reference_plan \
                or Path(reference_plan).is_absolute() \
                or environment_inference.get("referenceCheckpoint") != "best" \
                or environment_inference.get("objective") \
                != "maximize-irmv1-output-scale-gradient" \
                or environment_inference.get("assignment") \
                != "analytic-hard-optimum-scalar-regression" \
                or invariant_risk.get("type") != "irmv1" \
                or invariant_risk.get("loss") != "normalized-mse" \
                or invariant_risk.get("environmentWeighting") != "uniform" \
                or invariant_risk.get("penalty") \
                != "squared-output-scale-gradient-at-one" \
                or not math.isfinite(penalty_weight) \
                or penalty_weight <= 0 \
                or not isinstance(penalty_anneal_steps, int) \
                or isinstance(penalty_anneal_steps, bool) \
                or penalty_anneal_steps < 0 \
                or invariant_risk.get("largePenaltyObjectiveRescaling") is not True \
                or invariant_risk.get("batchConstruction") \
                != "globally-shuffled-across-environments" \
                or dataset_filter is None:
            raise ValueError("EIIL + IRMv1 contract is invalid")
    mean_teacher = training.get("meanTeacher")
    if mean_teacher is not None:
        student_input = mean_teacher.get("studentConsistencyInput", {})
        half_life_epochs = float(mean_teacher.get("halfLifeEpochs", 0))
        consistency_weight = float(mean_teacher.get("consistencyWeight", 0))
        ramp_up_fraction = float(mean_teacher.get("rampUpFraction", -1))
        epsilon_rms = float(student_input.get("epsilonRms", 0))
        if mean_teacher.get("contract") \
                != "ema-self-distillation-regression-v1" \
                or mean_teacher.get("teacherUpdate") \
                != "ema-after-student-optimizer-step" \
                or not math.isfinite(half_life_epochs) \
                or half_life_epochs <= 0 \
                or mean_teacher.get("consistencyLoss") != "normalized-mse" \
                or not math.isfinite(consistency_weight) \
                or consistency_weight <= 0 \
                or mean_teacher.get("rampUp") \
                != "linear-by-optimizer-step" \
                or not math.isfinite(ramp_up_fraction) \
                or not 0 <= ramp_up_fraction <= 1 \
                or mean_teacher.get("supervisedInput") != "clean" \
                or mean_teacher.get("teacherInput") != "clean" \
                or student_input.get("type") != "gaussian" \
                or student_input.get("space") \
                != "training-position-normalized-input" \
                or student_input.get("norm") != "exact-per-example-rms" \
                or not math.isfinite(epsilon_rms) \
                or not 0 < epsilon_rms <= 1 \
                or mean_teacher.get("inferenceModel") != "ema-teacher" \
                or dataset_filter is None:
            raise ValueError("Mean Teacher contract is invalid")
        if input_normalization != TRAINING_POSITION_INPUT_NORMALIZATION:
            raise ValueError(
                "Mean Teacher currently requires training-position normalization"
            )
    if training.get("swa") is not None:
        validate_swa_config(training["swa"])
    l2 = training.get("l2Regularization")
    if l2 is not None and (
        l2.get("type") != "explicit-loss-term"
        or l2.get("parameters") != "all-trainable-matrices"
        or l2.get("reduction") != "half-sum-squared"
        or not math.isfinite(float(l2.get("coefficient", 0)))
        or not 0 < float(l2["coefficient"]) <= 1
    ):
        raise ValueError("explicit L2 regularization contract is invalid")
    optimizer_decay = training.get("optimizerWeightDecay")
    if optimizer_decay is not None:
        coefficient = float(optimizer_decay.get("coefficient", 0))
        optimizer = training["optimizer"]
        if optimizer_decay.get("type") != "optimizer-native" \
                or optimizer_decay.get("parameterGroups") != ["muon", "adamw"] \
                or not math.isfinite(coefficient) \
                or not 0 < coefficient <= 1 \
                or float(optimizer["muon"]["weightDecay"]) != coefficient \
                or float(optimizer["adamw"]["weightDecay"]) != coefficient:
            raise ValueError("optimizer weight-decay contract is invalid")
    if l2 is not None and optimizer_decay is not None:
        raise ValueError("L2 loss and optimizer weight decay are separate variants")
    regularizers = (
        training.get("swa"),
        l2,
        optimizer_decay,
        training.get("sam"),
        training.get("adversarialInput"),
        training.get("adversarialReturnVector"),
        training.get("adversarialLogPricePath"),
        cvar_dro,
        environment_inference,
        mean_teacher,
    )
    if robust_regression is not None and any(
        value is not None for value in regularizers
    ):
        raise ValueError(
            "the initial robust-regression screen must isolate the loss"
        )
    if cvar_dro is not None and any(value is not None for value in (
        training.get("swa"), l2, optimizer_decay, training.get("sam"),
        training.get("adversarialInput"),
        training.get("adversarialReturnVector"),
        training.get("adversarialLogPricePath"), robust_regression,
        mean_teacher,
    )):
        raise ValueError("the initial CVaR-DRO matrix must isolate group risk")
    if environment_inference is not None and any(value is not None for value in (
        training.get("swa"), l2, optimizer_decay, training.get("sam"),
        training.get("adversarialInput"),
        training.get("adversarialReturnVector"),
        training.get("adversarialLogPricePath"), robust_regression, cvar_dro,
        mean_teacher,
    )):
        raise ValueError("the initial EIIL matrix must isolate invariant training")
    if mean_teacher is not None and any(value is not None for value in (
        training.get("swa"), l2, optimizer_decay, training.get("sam"),
        training.get("adversarialInput"),
        training.get("adversarialReturnVector"),
        training.get("adversarialLogPricePath"), robust_regression, cvar_dro,
        environment_inference,
    )):
        raise ValueError("the initial Mean Teacher sweep must isolate self-distillation")
    sam = training.get("sam")
    if sam is not None and (
        sam.get("type") != "sharpness-aware-minimization"
        or sam.get("adaptive") is not False
        or sam.get("gradientNorm") != "global-l2"
        or sam.get("perturbationUnit") != "optimizer-update"
        or not math.isfinite(float(sam.get("rho", 0)))
        or not 0 < float(sam["rho"]) <= 1
    ):
        raise ValueError("SAM contract is invalid")
    adversarial_input = training.get("adversarialInput")
    if adversarial_input is not None:
        epsilon_rms = float(adversarial_input.get("epsilonRms", 0))
        steps = adversarial_input.get("steps")
        step_size_rms = float(adversarial_input.get("stepSizeRms", 0))
        adversarial_weight = float(
            adversarial_input.get("adversarialWeight", 0)
        )
        if adversarial_input.get("type") != "projected-gradient-ascent" \
                or adversarial_input.get("space") \
                != "training-position-normalized-input" \
                or adversarial_input.get("norm") != "rms-l2" \
                or adversarial_input.get("randomStart") is not False \
                or adversarial_input.get("target") \
                != "unchanged-next-return" \
                or not math.isfinite(epsilon_rms) \
                or not 0 < epsilon_rms <= 1 \
                or not isinstance(steps, int) \
                or isinstance(steps, bool) \
                or not 1 <= steps <= 16 \
                or not math.isfinite(step_size_rms) \
                or step_size_rms <= 0 \
                or not math.isfinite(adversarial_weight) \
                or not 0 < adversarial_weight <= 1:
            raise ValueError("adversarial-input training contract is invalid")
        if input_normalization != TRAINING_POSITION_INPUT_NORMALIZATION:
            raise ValueError(
                "adversarial-input training currently requires training-position "
                "normalization"
            )
    adversarial_return_vector = training.get("adversarialReturnVector")
    if adversarial_return_vector is not None:
        epsilon_rms = float(adversarial_return_vector.get("epsilonRms", 0))
        steps = adversarial_return_vector.get("steps")
        step_size_rms = float(
            adversarial_return_vector.get("stepSizeRms", 0)
        )
        adversarial_weight = float(
            adversarial_return_vector.get("adversarialWeight", 0)
        )
        if adversarial_return_vector.get("type") \
                != "projected-gradient-ascent" \
                or adversarial_return_vector.get("space") \
                != "independent-normalized-input-and-output-log-returns" \
                or adversarial_return_vector.get("norm") != "rms-l2" \
                or adversarial_return_vector.get("randomStart") is not False \
                or adversarial_return_vector.get("inputScale") \
                != "training-standard-deviation-per-lag-position" \
                or adversarial_return_vector.get("targetScale") \
                != "training-target-return-standard-deviation" \
                or adversarial_return_vector.get("returnCount") != 121 \
                or not math.isfinite(epsilon_rms) \
                or not 0 < epsilon_rms <= 1 \
                or not isinstance(steps, int) \
                or isinstance(steps, bool) \
                or not 1 <= steps <= 16 \
                or not math.isfinite(step_size_rms) \
                or step_size_rms <= 0 \
                or not math.isfinite(adversarial_weight) \
                or not 0 < adversarial_weight <= 1:
            raise ValueError(
                "adversarial input/output return-vector contract is invalid"
            )
        if input_normalization != TRAINING_POSITION_INPUT_NORMALIZATION:
            raise ValueError(
                "adversarial return-vector training currently requires "
                "training-position normalization"
            )
    combined_sam_return_vector = (
        sam is not None and adversarial_return_vector is not None
    )
    if combined_sam_return_vector and (
        sam.get("baseObjective") != "clean-plus-adversarial-return-vector"
        or adversarial_return_vector.get("regenerateAtSamPerturbedWeights")
        is not True
    ):
        raise ValueError(
            "combined SAM + adversarial return-vector contract is invalid"
        )
    adversarial_log_price_path = training.get("adversarialLogPricePath")
    if adversarial_log_price_path is not None:
        epsilon_rms = float(adversarial_log_price_path.get("epsilonRms", 0))
        steps = adversarial_log_price_path.get("steps")
        step_size_rms = float(
            adversarial_log_price_path.get("stepSizeRms", 0)
        )
        adversarial_weight = float(
            adversarial_log_price_path.get("adversarialWeight", 0)
        )
        path_space = adversarial_log_price_path.get("space")
        perturbs_future_endpoint = (
            path_space
            == "reconstructed-log-price-path-including-future-endpoint"
        )
        fixes_future_endpoint = (
            path_space
            == "reconstructed-observed-log-price-path-fixed-future-endpoint"
        )
        path_contract_valid = (
            (
                perturbs_future_endpoint
                and adversarial_log_price_path.get("target")
                == "recomputed-final-log-price-difference"
                and adversarial_log_price_path.get("pathPoints") == 122
                and adversarial_log_price_path.get(
                    "futureEndpoint", "perturbed"
                ) == "perturbed"
            )
            or (
                fixes_future_endpoint
                and adversarial_log_price_path.get("target")
                == (
                    "recomputed-final-log-price-difference-with-fixed-"
                    "future-endpoint"
                )
                and adversarial_log_price_path.get("pathPoints") == 121
                and adversarial_log_price_path.get("futureEndpoint")
                == "fixed-observed"
            )
        )
        if adversarial_log_price_path.get("type") \
                != "projected-gradient-ascent" \
                or not path_contract_valid \
                or adversarial_log_price_path.get("norm") != "rms-l2" \
                or adversarial_log_price_path.get("scale") \
                != "training-target-return-standard-deviation" \
                or adversarial_log_price_path.get("randomStart") is not False \
                or adversarial_log_price_path.get("input") \
                != "recomputed-adjacent-log-price-differences" \
                or not math.isfinite(epsilon_rms) \
                or not 0 < epsilon_rms <= 1 \
                or not isinstance(steps, int) \
                or isinstance(steps, bool) \
                or not 1 <= steps <= 16 \
                or not math.isfinite(step_size_rms) \
                or step_size_rms <= 0 \
                or not math.isfinite(adversarial_weight) \
                or not 0 < adversarial_weight <= 1:
            raise ValueError(
                "adversarial log-price-path training contract is invalid"
            )
        if input_normalization != TRAINING_POSITION_INPUT_NORMALIZATION:
            raise ValueError(
                "adversarial log-price-path training currently requires "
                "training-position normalization"
            )
    configured_regularizers = sum(value is not None for value in (
        l2, optimizer_decay, sam, adversarial_input,
        adversarial_return_vector, adversarial_log_price_path, mean_teacher,
    ))
    if configured_regularizers > 1 and not (
        configured_regularizers == 2
        and combined_sam_return_vector
        and l2 is None
        and optimizer_decay is None
        and adversarial_input is None
        and adversarial_log_price_path is None
    ):
        raise ValueError(
            "L2, optimizer weight decay, SAM, adversarial input training, "
            "adversarial return-vector training, and adversarial log-price-path "
            "training, and Mean Teacher are separate variants"
        )
    partitioning = training.get("parameterPartitioning")
    if partitioning is not None and partitioning != {
        "type": "depth-width-quadrants-v1",
        "rotation": "epoch-round-robin",
    }:
        raise ValueError("memorization parameter partitioning is invalid")
    if partitioning is not None and (
        len(architecture["widths"]) % 2 != 0
        or any(int(width) % 2 != 0 for width in architecture["widths"])
    ):
        raise ValueError("depth-width partitioning requires even depth and width")


@torch.no_grad()
def mask_inactive_parameter_updates(
    assignments: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    active_partition: int,
    optimizers: tuple[torch.optim.Optimizer, ...],
) -> None:
    """Mask gradients and optimizer state outside the active quadrant."""
    if active_partition not in range(4):
        raise ValueError("active parameter partition must be in [0, 3]")
    for parameter, assignment in assignments:
        inactive = assignment != active_partition
        if parameter.grad is not None:
            parameter.grad.masked_fill_(inactive, 0)
        for optimizer in optimizers:
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(inactive, 0)


@torch.no_grad()
def snapshot_inactive_parameter_values(
    assignments: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    active_partition: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            parameter,
            assignment != active_partition,
            parameter.detach()[assignment != active_partition].clone(),
        )
        for parameter, assignment in assignments
    )


@torch.no_grad()
def assert_inactive_parameter_values_unchanged(
    frozen: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> None:
    if any(
        not torch.equal(parameter.detach()[inactive], values)
        for parameter, inactive, values in frozen
    ):
        raise RuntimeError("inactive parameter quadrant changed during training")


def matrix_l2_penalty(model: NormalizedGluNextReturn) -> torch.Tensor:
    matrices = tuple(
        parameter.float()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim == 2
    )
    if not matrices:
        raise RuntimeError("explicit L2 objective found no trainable matrices")
    return 0.5 * sum(
        (parameter.square().sum() for parameter in matrices),
        start=torch.zeros((), device=matrices[0].device),
    )


def weighted_normalized_mse(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    target_std: float,
) -> torch.Tensor:
    per_example = ((prediction - targets.float()) / target_std).square()
    return (per_example * weights).sum() / weights.sum()


def weighted_robust_normalized_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    target_std: float,
    robust_regression: dict,
) -> torch.Tensor:
    """Robust loss with the same local quadratic scale as normalized MSE."""
    residual = (prediction - targets.float()) / target_std
    loss_type = robust_regression["type"]
    if loss_type == "huber":
        delta = float(robust_regression["delta"])
        absolute = residual.abs()
        # Twice the conventional Huber loss: near zero this is residual^2,
        # matching the scale of the existing normalized-MSE objective.
        per_example = torch.where(
            absolute <= delta,
            residual.square(),
            2.0 * delta * absolute - delta * delta,
        )
    elif loss_type == "student-t":
        degrees_of_freedom = float(robust_regression["degreesOfFreedom"])
        # This scaled Student-t NLL is also residual^2 + O(residual^4) near zero.
        per_example = degrees_of_freedom * torch.log1p(
            residual.square() / degrees_of_freedom
        )
    else:
        raise ValueError(f"unsupported robust-regression loss: {loss_type}")
    return (per_example * weights).sum() / weights.sum()


def weighted_regression_objective(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    target_std: float,
    robust_regression: dict | None,
) -> torch.Tensor:
    if robust_regression is None:
        return weighted_normalized_mse(
            prediction, targets, weights, target_std
        )
    return weighted_robust_normalized_loss(
        prediction, targets, weights, target_std, robust_regression
    )


def mean_teacher_ema_decay(
    half_life_epochs: float,
    steps_per_epoch: int,
) -> float:
    if not math.isfinite(half_life_epochs) or half_life_epochs <= 0:
        raise ValueError("EMA half-life must be positive")
    if isinstance(steps_per_epoch, bool) or steps_per_epoch < 1:
        raise ValueError("steps per epoch must be positive")
    return math.exp(
        math.log(0.5) / (float(half_life_epochs) * int(steps_per_epoch))
    )


def mean_teacher_ramp_multiplier(
    global_step: int,
    total_steps: int,
    ramp_up_fraction: float,
) -> float:
    if isinstance(global_step, bool) or global_step < 0:
        raise ValueError("global step must be non-negative")
    if isinstance(total_steps, bool) or total_steps < 1:
        raise ValueError("total steps must be positive")
    if not math.isfinite(ramp_up_fraction) \
            or not 0 <= ramp_up_fraction <= 1:
        raise ValueError("ramp-up fraction must be in [0, 1]")
    if ramp_up_fraction == 0:
        return 1.0
    ramp_steps = max(1.0, float(total_steps) * ramp_up_fraction)
    return min(1.0, float(global_step) / ramp_steps)


def gaussian_normalized_input_perturbation(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    epsilon_rms: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add deterministic Gaussian noise with exact per-example normalized RMS."""
    feature_std = getattr(model, "feature_std", None)
    if not torch.is_tensor(feature_std) \
            or feature_std.ndim != 1 \
            or feature_std.shape[0] != features.shape[1]:
        raise ValueError("model must expose one training standard deviation per input")
    if not math.isfinite(epsilon_rms) or not 0 < epsilon_rms <= 1:
        raise ValueError("input perturbation RMS must be in (0, 1]")
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    normalized_noise = torch.randn(
        features.shape,
        dtype=torch.float32,
        device=features.device,
        generator=generator,
    )
    rms = normalized_noise.square().mean(dim=1, keepdim=True).sqrt()
    normalized_noise = (
        normalized_noise / rms.clamp_min(1e-12) * float(epsilon_rms)
    )
    attacked = (
        features.detach().float()
        + normalized_noise * feature_std.detach().float()
    )
    return attacked, normalized_noise


@torch.no_grad()
def update_ema_teacher(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    decay: float,
) -> None:
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if teacher_state.keys() != student_state.keys():
        raise ValueError("teacher and student state contracts differ")
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if teacher_value.is_floating_point():
            teacher_value.mul_(decay).add_(
                student_value.to(teacher_value.dtype), alpha=1.0 - decay
            )
        else:
            teacher_value.copy_(student_value)


def project_normalized_rms_l2(
    delta: torch.Tensor,
    epsilon_rms: float,
) -> torch.Tensor:
    if delta.ndim != 2 or delta.shape[1] < 1:
        raise ValueError("adversarial input delta must be [example, feature]")
    radius = float(epsilon_rms) * math.sqrt(delta.shape[1])
    norms = torch.linalg.vector_norm(delta.float(), dim=1, keepdim=True)
    scale = (radius / norms.clamp_min(1e-12)).clamp(max=1.0)
    return delta * scale.to(delta.dtype)


def adversarial_input_examples(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_std: float,
    epsilon_rms: float,
    steps: int,
    step_size_rms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maximize batch MSE inside an L2 ball in normalized input space."""
    feature_std = getattr(model, "feature_std", None)
    if not torch.is_tensor(feature_std) \
            or feature_std.ndim != 1 \
            or feature_std.shape[0] != features.shape[1]:
        raise ValueError("model must expose one training standard deviation per input")
    base = features.detach().float()
    delta = torch.zeros_like(base)
    step_radius = float(step_size_rms) * math.sqrt(base.shape[1])
    for _step in range(int(steps)):
        delta.requires_grad_(True)
        attacked = base + delta * feature_std.detach().float()
        prediction = model(attacked)
        objective = weighted_normalized_mse(
            prediction, targets, weights, target_std
        )
        gradient, = torch.autograd.grad(objective, delta, only_inputs=True)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("adversarial input gradient is non-finite")
        gradient_norm = torch.linalg.vector_norm(
            gradient.float(), dim=1, keepdim=True
        )
        direction = gradient / gradient_norm.clamp_min(1e-12).to(gradient.dtype)
        delta = project_normalized_rms_l2(
            (delta + step_radius * direction).detach(), epsilon_rms
        )
    return (
        base + delta * feature_std.detach().float(),
        delta.detach(),
    )


def adversarial_return_vector_examples(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_std: float,
    epsilon_rms: float,
    steps: int,
    step_size_rms: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perturb the 120 inputs and one target as independent log returns."""
    feature_std = getattr(model, "feature_std", None)
    if not torch.is_tensor(feature_std) \
            or feature_std.ndim != 1 \
            or feature_std.shape[0] != features.shape[1]:
        raise ValueError("model must expose one training standard deviation per input")
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("target standard deviation must be positive")
    base_features = features.detach().float()
    base_targets = targets.detach().float()
    normalized_delta = torch.zeros(
        (features.shape[0], features.shape[1] + 1),
        dtype=base_features.dtype,
        device=base_features.device,
    )
    step_radius = float(step_size_rms) * math.sqrt(
        normalized_delta.shape[1]
    )
    input_scale = feature_std.detach().float()
    for _step in range(int(steps)):
        normalized_delta.requires_grad_(True)
        attacked_features = (
            base_features + normalized_delta[:, :-1] * input_scale
        )
        attacked_targets = (
            base_targets + normalized_delta[:, -1] * float(target_std)
        )
        prediction = model(attacked_features)
        objective = weighted_normalized_mse(
            prediction, attacked_targets, weights, target_std
        )
        gradient, = torch.autograd.grad(
            objective, normalized_delta, only_inputs=True
        )
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                "adversarial return-vector gradient is non-finite"
            )
        gradient_norm = torch.linalg.vector_norm(
            gradient.float(), dim=1, keepdim=True
        )
        direction = gradient / gradient_norm.clamp_min(1e-12).to(
            gradient.dtype
        )
        normalized_delta = project_normalized_rms_l2(
            (normalized_delta + step_radius * direction).detach(), epsilon_rms
        )
    return (
        (
            base_features + normalized_delta[:, :-1] * input_scale
        ).detach(),
        (
            base_targets + normalized_delta[:, -1] * float(target_std)
        ).detach(),
        normalized_delta.detach(),
    )


def reconstruct_relative_log_price_path(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Build 122 relative log-price points from 120 inputs and one target."""
    if features.ndim != 2 or features.shape[1] != 120:
        raise ValueError("log-price path attack requires 120 input returns")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("log-price path attack requires one target per example")
    origin = torch.zeros(
        (features.shape[0], 1), dtype=features.dtype, device=features.device
    )
    observed = torch.cat((origin, torch.cumsum(features, dim=1)), dim=1)
    future = observed[:, -1:] + targets[:, None]
    return torch.cat((observed, future), dim=1)


def adversarial_log_price_path_examples(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_std: float,
    epsilon_rms: float,
    steps: int,
    step_size_rms: float,
    perturb_future_endpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Maximize MSE by perturbing a reconstructed relative log-price path.

    Epsilon is RMS log-price displacement in units of the training target-return
    standard deviation. When ``perturb_future_endpoint`` is false, only the 121
    observed boundaries move and the recorded future endpoint remains fixed.
    Inputs and the target are reconstructed by adjacent differencing.
    """
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("target standard deviation must be positive")
    base_path = reconstruct_relative_log_price_path(
        features.detach().float(), targets.detach().float()
    )
    perturbable_path = (
        base_path if perturb_future_endpoint else base_path[:, :-1]
    )
    fixed_future = (
        None if perturb_future_endpoint else base_path[:, -1:].detach()
    )
    normalized_delta = torch.zeros_like(perturbable_path)
    step_radius = float(step_size_rms) * math.sqrt(perturbable_path.shape[1])
    for _step in range(int(steps)):
        normalized_delta.requires_grad_(True)
        attacked_perturbable_path = (
            perturbable_path + normalized_delta * float(target_std)
        )
        attacked_path = (
            attacked_perturbable_path
            if fixed_future is None
            else torch.cat((attacked_perturbable_path, fixed_future), dim=1)
        )
        attacked_returns = torch.diff(attacked_path, dim=1)
        attacked_features = attacked_returns[:, :-1]
        attacked_targets = attacked_returns[:, -1]
        prediction = model(attacked_features)
        objective = weighted_normalized_mse(
            prediction, attacked_targets, weights, target_std
        )
        gradient, = torch.autograd.grad(
            objective, normalized_delta, only_inputs=True
        )
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                "adversarial log-price-path gradient is non-finite"
            )
        gradient_norm = torch.linalg.vector_norm(
            gradient.float(), dim=1, keepdim=True
        )
        direction = gradient / gradient_norm.clamp_min(1e-12).to(
            gradient.dtype
        )
        normalized_delta = project_normalized_rms_l2(
            (normalized_delta + step_radius * direction).detach(), epsilon_rms
        )
    attacked_perturbable_path = (
        perturbable_path + normalized_delta * float(target_std)
    )
    attacked_path = (
        attacked_perturbable_path
        if fixed_future is None
        else torch.cat((attacked_perturbable_path, fixed_future), dim=1)
    )
    attacked_returns = torch.diff(attacked_path, dim=1)
    return (
        attacked_returns[:, :-1].detach(),
        attacked_returns[:, -1].detach(),
        normalized_delta.detach(),
    )


@torch.no_grad()
def sam_perturb_parameters(
    model: torch.nn.Module,
    rho: float,
) -> tuple[tuple[torch.nn.Parameter, torch.Tensor], ...]:
    parameters = tuple(
        parameter for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if not parameters:
        raise RuntimeError("SAM found no gradients to perturb")
    gradient_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
    ]))
    if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) <= 0:
        raise FloatingPointError("SAM gradient norm is non-finite or zero")
    scale = float(rho) / float(gradient_norm)
    snapshots = tuple(
        (parameter, parameter.detach().clone()) for parameter in parameters
    )
    for parameter in parameters:
        parameter.add_(parameter.grad.detach(), alpha=scale)
    return snapshots


@torch.no_grad()
def sam_restore_parameters(
    snapshots: tuple[tuple[torch.nn.Parameter, torch.Tensor], ...],
) -> None:
    for parameter, original in snapshots:
        parameter.copy_(original)


@torch.no_grad()
def infer_eiil_environments(
    model: NormalizedGluNextReturn,
    dataset: NextReturnDataset,
    *,
    reference_snapshot_file: Path,
    corpus_fingerprint_value: str,
    batch_size: int,
    target_std: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    snapshot = json.loads(reference_snapshot_file.read_text(encoding="utf-8"))
    reference_plan = snapshot.get("plan", snapshot)
    reference_checkpoint_file = (
        reference_snapshot_file.parents[1] / "checkpoints/best.json"
    )
    reference_checkpoint = load_torch_checkpoint(
        reference_checkpoint_file, map_location=device, weights_only=False
    )
    if reference_checkpoint.get("planSha256") != snapshot.get("planSha256") \
            or reference_checkpoint.get("corpusFingerprint") \
            != corpus_fingerprint_value:
        raise ValueError("EIIL reference checkpoint does not match the clean corpus")
    reference_model = copy.deepcopy(model)
    reference_model.load_state_dict(reference_checkpoint["model"])
    reference_model.eval()
    gradients: list[np.ndarray] = []
    for features, targets, _weights in iter_device_batches(
        dataset.iter_batches(
            "train", batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        predictions = reference_model(features)
        gradients.append(
            regression_scale_gradients(
                predictions, targets, target_std=target_std
            ).detach().cpu().numpy().astype(np.float64, copy=False)
        )
    scale_gradients = np.concatenate(gradients)
    environment_ids = infer_binary_environment_ids(scale_gradients)
    counts = np.bincount(environment_ids, minlength=2)
    environment_fingerprint = hashlib.sha256(
        environment_ids.tobytes()
    ).hexdigest()
    return environment_ids, {
        "version": 1,
        "type": "eiil",
        "environmentCount": 2,
        "referencePlanId": reference_plan["id"],
        "referencePlanSha256": snapshot.get("planSha256"),
        "referenceCheckpoint": str(reference_checkpoint_file),
        "referenceEpoch": int(reference_checkpoint["epoch"]),
        "objective": "maximize-irmv1-output-scale-gradient",
        "assignment": "analytic-hard-optimum-scalar-regression",
        "environmentCounts": [int(value) for value in counts],
        "environmentFractions": [
            float(value / environment_ids.size) for value in counts
        ],
        "scaleGradientMean": float(scale_gradients.mean()),
        "scaleGradientStd": float(scale_gradients.std()),
        "environmentFingerprint": environment_fingerprint,
    }


@torch.no_grad()
def evaluate_eiil_environments(
    model: NormalizedGluNextReturn,
    dataset: NextReturnDataset,
    environment_ids: np.ndarray,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
) -> tuple[dict[str, float | int | None], dict[str, object]]:
    model.eval()
    environment_count = int(np.max(environment_ids)) + 1
    metrics = MetricAccumulator(target_std, device)
    squared = torch.zeros(environment_count, dtype=torch.float64, device=device)
    gradient_sum = torch.zeros_like(squared)
    weights_by_environment = torch.zeros_like(squared)
    for features, targets, weights, groups in iter_device_batches(
        dataset.iter_assigned_group_batches(
            "train", batch_size, environment_ids,
            shuffle=False, seed=0, reuse_buffers=True,
        ),
        device,
    ):
        predictions = model(features)
        metrics.add(predictions, targets, weights)
        normalized_squared = (
            (predictions.double() - targets.double()) / float(target_std)
        ).square() * weights.double()
        scale_gradient = regression_scale_gradients(
            predictions, targets, target_std=target_std
        ).double() * weights.double()
        squared.scatter_add_(0, groups.long(), normalized_squared)
        gradient_sum.scatter_add_(0, groups.long(), scale_gradient)
        weights_by_environment.scatter_add_(0, groups.long(), weights.double())
    if bool((weights_by_environment <= 0).any()):
        raise RuntimeError("EIIL evaluation contains an empty environment")
    environment_losses = squared / weights_by_environment
    environment_gradients = gradient_sum / weights_by_environment
    penalty = environment_gradients.square().mean()
    return metrics.result(), {
        "environmentCount": environment_count,
        "environmentCounts": [int(value) for value in weights_by_environment],
        "meanEnvironmentNormalizedMse": float(environment_losses.mean()),
        "bestEnvironmentNormalizedMse": float(environment_losses.min()),
        "worstEnvironmentNormalizedMse": float(environment_losses.max()),
        "environmentNormalizedMse": [
            float(value) for value in environment_losses
        ],
        "environmentScaleGradients": [
            float(value) for value in environment_gradients
        ],
        "irmPenalty": float(penalty),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether a normalized GLU can memorize a fixed subset."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--dropout-rate", type=float)
    parser.add_argument("--causal-volatility-window", type=int)
    parser.add_argument("--swa-sweep", action="store_true")
    parser.add_argument("--l2-rate", type=float)
    parser.add_argument("--optimizer-weight-decay", type=float)
    parser.add_argument("--sam-rho", type=float)
    parser.add_argument("--exclude-zero-targets", action="store_true")
    parser.add_argument("--cvar-tail-fraction", type=float)
    parser.add_argument("--eiil-reference-plan")
    parser.add_argument("--irm-penalty-weight", type=float, default=10_000.0)
    parser.add_argument("--irm-penalty-anneal-steps", type=int, default=100)
    parser.add_argument("--validation-curve-split-plan")
    parser.add_argument("--validation-curve-examples", type=int, default=65_536)
    parser.add_argument("--mean-teacher-half-life-epochs", type=float)
    parser.add_argument("--mean-teacher-consistency-weight", type=float)
    parser.add_argument("--mean-teacher-ramp-up-fraction", type=float)
    parser.add_argument("--mean-teacher-input-perturbation-rms", type=float)
    parser.add_argument("--robust-loss", choices=("huber", "student-t"))
    parser.add_argument(
        "--robust-parameter",
        type=float,
        help="Huber delta or Student-t degrees of freedom in target-std units.",
    )
    parser.add_argument("--adversarial-input-epsilon-rms", type=float)
    parser.add_argument("--adversarial-input-steps", type=int, default=1)
    parser.add_argument("--adversarial-input-weight", type=float, default=0.5)
    parser.add_argument("--adversarial-return-vector-epsilon-rms", type=float)
    parser.add_argument("--adversarial-return-vector-steps", type=int, default=1)
    parser.add_argument("--adversarial-return-vector-weight", type=float, default=0.5)
    parser.add_argument("--adversarial-log-price-epsilon-rms", type=float)
    parser.add_argument("--adversarial-log-price-steps", type=int, default=1)
    parser.add_argument("--adversarial-log-price-weight", type=float, default=0.5)
    parser.add_argument(
        "--combined-epochs",
        type=int,
        help="Epoch budget for an explicitly combined regularization variant.",
    )
    parser.add_argument(
        "--adversarial-log-price-future-endpoint",
        choices=("perturbed", "fixed"),
        default="perturbed",
    )
    parser.add_argument("--variant-suffix")
    parser.add_argument(
        "--pause-file",
        type=Path,
        help=(
            "At completed-epoch boundaries, exit with a resumable status while "
            "this file exists."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    adversarial_input_requested = args.adversarial_input_epsilon_rms is not None
    adversarial_return_vector_requested = (
        args.adversarial_return_vector_epsilon_rms is not None
    )
    adversarial_log_price_requested = (
        args.adversarial_log_price_epsilon_rms is not None
    )
    if not adversarial_log_price_requested \
            and args.adversarial_log_price_future_endpoint != "perturbed":
        raise ValueError(
            "adversarial log-price future endpoint requires a log-price epsilon"
        )
    if sum((
        adversarial_input_requested,
        adversarial_return_vector_requested,
        adversarial_log_price_requested,
    )) > 1:
        raise ValueError(
            "input-only, input/output-return, and log-price-path attacks are "
            "separate variants"
        )
    adversarial_requested = (
        adversarial_input_requested
        or adversarial_return_vector_requested
        or adversarial_log_price_requested
    )
    mean_teacher_values = (
        args.mean_teacher_half_life_epochs,
        args.mean_teacher_consistency_weight,
        args.mean_teacher_ramp_up_fraction,
        args.mean_teacher_input_perturbation_rms,
    )
    mean_teacher_requested = any(
        value is not None for value in mean_teacher_values
    )
    if mean_teacher_requested:
        if any(value is None for value in mean_teacher_values) \
                or args.variant_suffix is None:
            raise ValueError(
                "Mean Teacher half-life, consistency weight, ramp-up fraction, "
                "input perturbation RMS, and variant-suffix are required together"
            )
        if args.exclude_zero_targets or any((
            args.cvar_tail_fraction is not None,
            args.eiil_reference_plan is not None,
            args.robust_loss is not None,
            args.dropout is not None,
            args.dropout_rate is not None,
            args.causal_volatility_window is not None,
            args.swa_sweep,
            args.l2_rate is not None,
            args.optimizer_weight_decay is not None,
            args.sam_rho is not None,
            adversarial_requested,
        )):
            raise ValueError(
                "the initial Mean Teacher sweep must isolate self-distillation"
            )
    combined_sam_return_requested = (
        args.sam_rho is not None
        and adversarial_return_vector_requested
        and not adversarial_input_requested
        and not adversarial_log_price_requested
    )
    if args.combined_epochs is not None and not combined_sam_return_requested:
        raise ValueError(
            "combined-epochs requires SAM plus adversarial return-vector training"
        )
    robust_values = (args.robust_loss, args.robust_parameter)
    robust_requested = any(value is not None for value in robust_values)
    if robust_requested:
        if any(value is None for value in robust_values) \
                or args.variant_suffix is None:
            raise ValueError(
                "robust-loss, robust-parameter, and variant-suffix must be "
                "supplied together"
            )
        if any((
            args.dropout is not None,
            args.dropout_rate is not None,
            args.causal_volatility_window is not None,
            args.swa_sweep,
            args.l2_rate is not None,
            args.optimizer_weight_decay is not None,
            args.sam_rho is not None,
            adversarial_requested,
            mean_teacher_requested,
        )):
            raise ValueError(
                "the initial robust-regression screen must isolate the loss"
            )
    cvar_requested = args.cvar_tail_fraction is not None
    eiil_requested = args.eiil_reference_plan is not None
    if not eiil_requested and (
        args.irm_penalty_weight != 10_000.0
        or args.irm_penalty_anneal_steps != 100
    ):
        raise ValueError("IRM settings require an EIIL reference plan")
    if eiil_requested:
        if args.variant_suffix is None:
            raise ValueError("EIIL requires a variant-suffix")
        if args.exclude_zero_targets or cvar_requested or robust_requested or any((
            args.dropout is not None,
            args.dropout_rate is not None,
            args.causal_volatility_window is not None,
            args.swa_sweep,
            args.l2_rate is not None,
            args.optimizer_weight_decay is not None,
            args.sam_rho is not None,
            adversarial_requested,
            mean_teacher_requested,
        )):
            raise ValueError("the initial EIIL matrix must isolate invariant training")
    if args.exclude_zero_targets and cvar_requested:
        raise ValueError(
            "CVaR-DRO already enables the required nonzero-target filter"
        )
    if cvar_requested:
        if args.variant_suffix is None:
            raise ValueError("cvar-tail-fraction requires a variant-suffix")
        if robust_requested or any((
            args.dropout is not None,
            args.dropout_rate is not None,
            args.causal_volatility_window is not None,
            args.swa_sweep,
            args.l2_rate is not None,
            args.optimizer_weight_decay is not None,
            args.sam_rho is not None,
            adversarial_requested,
            mean_teacher_requested,
        )):
            raise ValueError("the initial CVaR-DRO matrix must isolate group risk")
    if args.exclude_zero_targets:
        if args.variant_suffix is None:
            raise ValueError(
                "exclude-zero-targets requires a variant-suffix"
            )
        if robust_requested or any((
            args.dropout is not None,
            args.dropout_rate is not None,
            args.causal_volatility_window is not None,
            args.swa_sweep,
            args.l2_rate is not None,
            args.optimizer_weight_decay is not None,
            args.sam_rho is not None,
            adversarial_requested,
            mean_teacher_requested,
        )):
            raise ValueError(
                "the initial nonzero-target scaling matrix must isolate filtering"
            )
    dropout_values = (args.dropout, args.dropout_rate)
    if eiil_requested:
        plan = eiil_irm_variant(
            plan,
            reference_plan=str(args.eiil_reference_plan),
            penalty_weight=float(args.irm_penalty_weight),
            penalty_anneal_steps=int(args.irm_penalty_anneal_steps),
            suffix=str(args.variant_suffix),
        )
    elif cvar_requested:
        plan = cvar_dro_variant(
            plan,
            tail_fraction=float(args.cvar_tail_fraction),
            suffix=str(args.variant_suffix),
        )
    elif mean_teacher_requested:
        plan = mean_teacher_variant(
            plan,
            half_life_epochs=float(args.mean_teacher_half_life_epochs),
            consistency_weight=float(args.mean_teacher_consistency_weight),
            ramp_up_fraction=float(args.mean_teacher_ramp_up_fraction),
            input_perturbation_rms=float(
                args.mean_teacher_input_perturbation_rms
            ),
            suffix=str(args.variant_suffix),
        )
    elif args.exclude_zero_targets:
        plan = nonzero_target_variant(
            plan, suffix=str(args.variant_suffix)
        )
    elif robust_requested:
        plan = robust_regression_variant(
            plan,
            loss_type=str(args.robust_loss),
            parameter=float(args.robust_parameter),
            suffix=str(args.variant_suffix),
        )
    elif any(value is not None for value in dropout_values):
        if any(value is None for value in dropout_values) \
                or args.variant_suffix is None \
                or args.causal_volatility_window is not None \
                or args.swa_sweep \
                or args.l2_rate is not None \
                or args.optimizer_weight_decay is not None \
                or args.sam_rho is not None \
                or adversarial_requested:
            raise ValueError(
                "dropout, dropout-rate, and variant-suffix must be supplied together"
            )
        plan = dropout_variant(
            plan,
            probability=float(args.dropout),
            application_rate=float(args.dropout_rate),
            suffix=str(args.variant_suffix),
        )
    elif args.causal_volatility_window is not None:
        if args.variant_suffix is None \
                or args.swa_sweep \
                or args.l2_rate is not None \
                or args.optimizer_weight_decay is not None \
                or args.sam_rho is not None \
                or adversarial_requested:
            raise ValueError(
                "causal-volatility-window and variant-suffix must be supplied together"
            )
        plan = causal_volatility_variant(
            plan,
            window=int(args.causal_volatility_window),
            suffix=str(args.variant_suffix),
        )
    elif args.swa_sweep:
        if args.variant_suffix is None \
                or args.l2_rate is not None \
                or args.optimizer_weight_decay is not None \
                or args.sam_rho is not None \
                or adversarial_requested:
            raise ValueError(
                "swa-sweep and variant-suffix must be supplied together"
            )
        plan = swa_variant(
            plan, suffix=str(args.variant_suffix)
        )
    elif args.l2_rate is not None:
        if args.variant_suffix is None \
                or args.optimizer_weight_decay is not None \
                or args.sam_rho is not None \
                or adversarial_requested:
            raise ValueError(
                "l2-rate and variant-suffix must be supplied together"
            )
        plan = l2_variant(
            plan,
            rate=float(args.l2_rate),
            suffix=str(args.variant_suffix),
        )
    elif args.optimizer_weight_decay is not None:
        if args.variant_suffix is None \
                or args.sam_rho is not None \
                or adversarial_requested:
            raise ValueError(
                "optimizer-weight-decay and variant-suffix must be supplied together"
            )
        plan = optimizer_weight_decay_variant(
            plan,
            rate=float(args.optimizer_weight_decay),
            suffix=str(args.variant_suffix),
        )
    elif combined_sam_return_requested:
        if args.variant_suffix is None or args.combined_epochs is None:
            raise ValueError(
                "combined SAM + adversarial return-vector training requires "
                "variant-suffix and combined-epochs"
            )
        plan = sam_adversarial_return_vector_variant(
            plan,
            rho=float(args.sam_rho),
            epsilon_rms=float(args.adversarial_return_vector_epsilon_rms),
            steps=int(args.adversarial_return_vector_steps),
            adversarial_weight=float(args.adversarial_return_vector_weight),
            epochs=int(args.combined_epochs),
            suffix=str(args.variant_suffix),
        )
    elif args.sam_rho is not None:
        if args.variant_suffix is None or adversarial_requested:
            raise ValueError("sam-rho and variant-suffix must be supplied together")
        plan = sam_variant(
            plan, rho=float(args.sam_rho), suffix=str(args.variant_suffix)
        )
    elif adversarial_input_requested:
        if args.variant_suffix is None:
            raise ValueError(
                "adversarial-input epsilon and variant-suffix must be supplied "
                "together"
            )
        plan = adversarial_input_variant(
            plan,
            epsilon_rms=float(args.adversarial_input_epsilon_rms),
            steps=int(args.adversarial_input_steps),
            adversarial_weight=float(args.adversarial_input_weight),
            suffix=str(args.variant_suffix),
        )
    elif adversarial_return_vector_requested:
        if args.variant_suffix is None:
            raise ValueError(
                "adversarial return-vector epsilon and variant-suffix must be "
                "supplied together"
            )
        plan = adversarial_return_vector_variant(
            plan,
            epsilon_rms=float(args.adversarial_return_vector_epsilon_rms),
            steps=int(args.adversarial_return_vector_steps),
            adversarial_weight=float(args.adversarial_return_vector_weight),
            suffix=str(args.variant_suffix),
        )
    elif adversarial_log_price_requested:
        if args.variant_suffix is None:
            raise ValueError(
                "adversarial log-price epsilon and variant-suffix must be "
                "supplied together"
            )
        plan = adversarial_log_price_path_variant(
            plan,
            epsilon_rms=float(args.adversarial_log_price_epsilon_rms),
            steps=int(args.adversarial_log_price_steps),
            adversarial_weight=float(args.adversarial_log_price_weight),
            perturb_future_endpoint=(
                args.adversarial_log_price_future_endpoint == "perturbed"
            ),
            suffix=str(args.variant_suffix),
        )
    elif args.variant_suffix is not None:
        raise ValueError("variant-suffix requires a configured variant")
    if args.validation_curve_split_plan is not None:
        plan = with_validation_curve(
            plan,
            split_plan=str(args.validation_curve_split_plan),
            examples=int(args.validation_curve_examples),
        )
    elif args.validation_curve_examples != 65_536:
        raise ValueError(
            "validation-curve-examples requires a validation-curve split plan"
        )
    validate_plan(plan)
    active_runner_contract = runner_contract(plan)
    plan_hash = canonical_fingerprint(plan)
    layout = training_storage_layout(repo)
    dataset_root = require_under(
        resolve(repo, Path(plan["datasetDir"])), layout.datasets, "datasetDir"
    )
    run_root = require_under(
        resolve(repo, Path(plan["runDir"])), layout.runs, "runDir"
    )
    history_root = require_under(
        resolve(repo, Path(plan["historyDir"])),
        repo / "data/market/immutable/refs/candles",
        "historyDir",
    )
    reporter = Reporter(run_root)
    try:
        subset = plan["subset"]
        dataset_filter = plan.get("datasetFilter")
        subset_type = str(subset.get("type", "fixed-contiguous"))
        if subset_type == "fixed-contiguous":
            subset_builder = (
                fixed_nonzero_subset_shards
                if dataset_filter is not None else fixed_subset_shards
            )
            shards = subset_builder(
                history_root,
                date.fromisoformat(str(subset["date"])),
                int(subset["examples"]),
            )
            split_source = (
                "fixed-count nonzero targets from a contiguous candidate window"
                if dataset_filter is not None
                else "fixed contiguous memorization subset"
            )
        else:
            calendar = direct_calendar_shards(
                subset["split"], history_root,
                horizon_seconds=1, decision_stride_seconds=1,
            )
            shards = {
                "train": calendar["train"],
                "validation": [],
                "test": [],
            }
            split_source = "complete current calendar training split"
        exclude_zero_targets = dataset_filter is not None
        base_fingerprint = corpus_fingerprint(shards, horizon_return_count=1)
        fingerprint = (
            canonical_fingerprint({
                "baseCorpusFingerprint": base_fingerprint,
                "datasetFilter": dataset_filter,
            })
            if exclude_zero_targets else base_fingerprint
        )
        dataset = NextReturnDataset(
            shards,
            history_root,
            horizon_return_count=1,
            row_stride=1,
            exclude_zero_targets=exclude_zero_targets,
        )
        validation_curve = plan.get("validationCurve")
        validation_dataset: NextReturnDataset | None = None
        validation_curve_fingerprint: str | None = None
        validation_curve_split_plan_id: str | None = None
        if validation_curve is not None:
            validation_curve_plan_file = require_under(
                resolve(repo, Path(validation_curve["splitPlan"])),
                repo / "ml/training-plans",
                "validationCurve.splitPlan",
            )
            validation_curve_split_plan = json.loads(
                validation_curve_plan_file.read_text(encoding="utf-8")
            )
            validation_curve_split_plan_id = str(
                validation_curve_split_plan["id"]
            )
            validation_start = date.fromisoformat(
                str(validation_curve_split_plan["split"]["validationStart"])
            )
            validation_source = fixed_nonzero_subset_shards(
                history_root,
                validation_start,
                int(validation_curve["examples"]),
            )
            validation_shards = {
                "train": [],
                "validation": validation_source["train"],
                "test": [],
            }
            validation_curve_fingerprint = canonical_fingerprint({
                "baseCorpusFingerprint": corpus_fingerprint(
                    validation_shards, horizon_return_count=1
                ),
                "datasetFilter": dataset_filter,
                "validationCurve": validation_curve,
            })
            validation_dataset = NextReturnDataset(
                validation_shards,
                history_root,
                horizon_return_count=1,
                row_stride=1,
                exclude_zero_targets=True,
            )
        selection = {
            "event": "minute-return-dataset-selected",
            "planId": plan["id"],
            "counts": {
                "train": dataset.logical_count("train"),
                **(
                    {
                        "validationCurve": validation_dataset.logical_count(
                            "validation"
                        )
                    }
                    if validation_dataset is not None else {}
                ),
            },
            "candidateCounts": {
                "train": sum(shard.count for shard in shards["train"])
            },
            "datasetFilter": dataset_filter,
            "horizonSeconds": 1,
            "decisionStrideSeconds": 1,
            "splitSource": split_source,
            "corpusFingerprint": fingerprint,
            "testPolicy": "none; training interpolation diagnostic only",
            "validationCurve": (
                {
                    **validation_curve,
                    "splitPlanId": validation_curve_split_plan_id,
                    "counts": {
                        "validation": validation_dataset.logical_count(
                            "validation"
                        ),
                    },
                    "corpusFingerprint": validation_curve_fingerprint,
                }
                if validation_curve is not None
                and validation_dataset is not None else None
            ),
        }
        reporter.emit(selection)
        if args.validate_only:
            reporter.status("paused", latest=selection)
            return

        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file():
            if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("run directory belongs to a different plan")
        else:
            atomic_json(snapshot, snapshot_file)

        training = plan["training"]
        reporter.status("computing-training-statistics", planId=plan["id"])
        normalization = training_normalization(
            dataset, batch_size=int(training["evaluationBatchSize"])
        )
        target_mean = float(normalization["minuteMean"])
        target_std = float(normalization["minuteStd"])
        architecture = plan["architecture"]
        causal_volatility_window = (
            int(architecture["volatilityWindow"])
            if architecture.get("inputNormalization")
            == CAUSAL_VOLATILITY_INPUT_NORMALIZATION
            else None
        )
        model = NormalizedGluNextReturn(
            torch.from_numpy(normalization["featureMean"]),
            torch.from_numpy(normalization["featureStd"]),
            torch.tensor(target_mean),
            torch.tensor(target_std),
            widths=tuple(int(value) for value in architecture["widths"]),
            input_normalization=architecture.get(
                "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
            ),
            volatility_window=architecture.get("volatilityWindow"),
            dropout=float(architecture["dropout"]),
            dropout_rate=float(architecture["dropoutRate"]),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
            learnable_centering=bool(
                architecture.get("learnableCentering", True)
            ),
        )
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_parameter_count = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        atomic_json({
            "version": 1,
            "planId": plan["id"],
            "input": "120 completed one-second log returns",
            "inputNormalization": (
                {
                    "type": "causal-volatility-rms",
                    "window": int(architecture["volatilityWindow"]),
                    "targetScale": "same causal input-only volatility",
                }
                if architecture.get("inputNormalization")
                == CAUSAL_VOLATILITY_INPUT_NORMALIZATION
                else "training-set mean and standard deviation per lag position"
            ),
            "target": "the immediately following one-second log return",
            "counts": {"train": dataset.logical_count("train")},
            "selection": (
                "lowest deterministic evaluation-mode IRMv1 objective"
                if training.get("environmentInference") is not None
                else "lowest deterministic evaluation-mode training MSE"
            ),
            "regularization": (
                training["meanTeacher"]
                if training.get("meanTeacher") is not None
                else (
                    training["adversarialReturnVector"]
                    if training.get("adversarialReturnVector") is not None
                    else (
                        training["adversarialLogPricePath"]
                        if training.get("adversarialLogPricePath") is not None
                        else (
                            training["adversarialInput"]
                            if training.get("adversarialInput") is not None
                            else (
                                training["sam"]
                                if training.get("sam") is not None
                                else {
                                    **(
                                        training["l2Regularization"]
                                        if training.get("l2Regularization")
                                        is not None
                                        else (
                                            training["optimizerWeightDecay"]
                                            if training.get(
                                                "optimizerWeightDecay"
                                            ) is not None
                                            else (
                                                {
                                                    "type": "none",
                                                    "dropout": 0,
                                                    "weightDecay": 0,
                                                }
                                                if float(architecture.get(
                                                    "dropout", 0
                                                )) == 0
                                                else {
                                                    "type": (
                                                        "intermittent-"
                                                        "activation-dropout"
                                                    ),
                                                    "probability": float(
                                                        architecture["dropout"]
                                                    ),
                                                    "applicationRate": float(
                                                        architecture[
                                                            "dropoutRate"
                                                        ]
                                                    ),
                                                    "weightDecay": 0,
                                                }
                                            )
                                        )
                                    )
                                }
                            )
                        )
                    )
                )
            ),
            "schedule": "fixed learning rate; no validation or early stopping",
            "swa": training.get("swa"),
            "parameterTraining": (
                training.get("parameterPartitioning")
                or "all trainable parameters updated every batch"
            ),
            "centeringMatrix": (
                "learned independently for value and gate branches"
                if architecture.get("learnableCentering", True)
                else "static canonical projector I - 11^T/d"
            ),
            "corpusFingerprint": fingerprint,
            "datasetFilter": dataset_filter,
            "environmentInference": training.get("environmentInference"),
            "invariantRiskMinimization": training.get(
                "invariantRiskMinimization"
            ),
            "meanTeacher": training.get("meanTeacher"),
            "validationCurve": validation_curve,
            "validationCurveCorpusFingerprint": validation_curve_fingerprint,
        }, dataset_root / "dataset.json")

        device = torch.device(training["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but is unavailable")
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        model = model.to(device)
        batch_size = int(training["batchSize"])
        eval_batch_size = int(training["evaluationBatchSize"])
        mean_teacher_config = training.get("meanTeacher")
        mean_teacher_model = None
        if mean_teacher_config is not None:
            mean_teacher_model = copy.deepcopy(model).to(device)
            mean_teacher_model.eval()
            for parameter in mean_teacher_model.parameters():
                parameter.requires_grad_(False)
        environment_inference = training.get("environmentInference")
        invariant_risk = training.get("invariantRiskMinimization")
        environment_ids: np.ndarray | None = None
        environment_summary: dict[str, object] | None = None
        environment_fingerprint: str | None = None
        if environment_inference is not None:
            reference_snapshot_file = require_under(
                resolve(repo, Path(environment_inference["referencePlan"])),
                layout.runs,
                "EIIL referencePlan",
            )
            reporter.status(
                "inferring-environments", planId=plan["id"],
                referencePlan=str(reference_snapshot_file.relative_to(repo)),
            )
            environment_ids, environment_summary = infer_eiil_environments(
                model,
                dataset,
                reference_snapshot_file=reference_snapshot_file,
                corpus_fingerprint_value=fingerprint,
                batch_size=int(training["evaluationBatchSize"]),
                target_std=target_std,
                device=device,
            )
            environment_fingerprint = str(
                environment_summary["environmentFingerprint"]
            )
            atomic_json(
                environment_summary,
                run_root / "state/environment-inference.json",
            )
        optimizers = build_optimizers(model, training, device)
        maximum_epochs = int(training["epochs"])
        swa = (
            EpochSwaSweep(model, maximum_epochs=maximum_epochs)
            if training.get("swa") is not None else None
        )
        partitioning = training.get("parameterPartitioning")
        partition_assignments = (
            depth_width_parameter_assignments(model)
            if partitioning is not None else ()
        )
        partition_parameter_counts = (
            [
                sum(
                    int((assignment == index).sum().item())
                    for _parameter, assignment in partition_assignments
                )
                for index in range(4)
            ]
            if partition_assignments else []
        )
        last_file = run_root / "checkpoints/last.json"
        best_file = run_root / "checkpoints/best.json"
        start_epoch = 0
        global_step = 0
        best_train_score = math.inf
        best_epoch = -1
        if checkpoint_exists(last_file):
            checkpoint = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            if checkpoint.get("planSha256") != plan_hash \
                    or checkpoint.get("corpusFingerprint") != fingerprint \
                    or checkpoint.get("runnerContract") != active_runner_contract \
                    or checkpoint.get("environmentFingerprint") \
                    != environment_fingerprint \
                    or checkpoint.get("validationCurveCorpusFingerprint") \
                    != validation_curve_fingerprint:
                raise ValueError("memorization checkpoint contract changed")
            if mean_teacher_model is not None:
                if checkpoint.get("studentModel") is None:
                    raise ValueError("Mean Teacher checkpoint has no student state")
                model.load_state_dict(checkpoint["studentModel"])
                mean_teacher_model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint["model"])
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["globalStep"])
            best_train_score = float(checkpoint["bestTrainScore"])
            best_epoch = int(checkpoint["bestEpoch"])
            if swa is not None:
                swa.load_state_dict(
                    checkpoint["swaState"]
                )

        target_score = float(training["targetNormalizedMse"])
        steps_per_epoch = math.ceil(
            dataset.logical_count("train") / batch_size
        )
        total_optimizer_steps = maximum_epochs * steps_per_epoch
        mean_teacher_decay = (
            mean_teacher_ema_decay(
                float(mean_teacher_config["halfLifeEpochs"]),
                steps_per_epoch,
            )
            if mean_teacher_config is not None else 0.0
        )
        mean_teacher_consistency_weight = (
            float(mean_teacher_config["consistencyWeight"])
            if mean_teacher_config is not None else 0.0
        )
        mean_teacher_ramp_up_fraction = (
            float(mean_teacher_config["rampUpFraction"])
            if mean_teacher_config is not None else 0.0
        )
        mean_teacher_input_rms = (
            float(
                mean_teacher_config["studentConsistencyInput"]["epsilonRms"]
            )
            if mean_teacher_config is not None else 0.0
        )
        l2_rate = float(
            training.get("l2Regularization", {}).get("coefficient", 0)
        )
        robust_regression = training.get("robustRegression")
        cvar_dro = training.get("cvarDro")
        cvar_tail_fraction = (
            float(cvar_dro["tailFraction"])
            if cvar_dro is not None else 0.0
        )
        irm_penalty_weight = (
            float(invariant_risk["penaltyWeight"])
            if invariant_risk is not None else 0.0
        )
        irm_penalty_anneal_steps = (
            int(invariant_risk["penaltyAnnealSteps"])
            if invariant_risk is not None else 0
        )
        sam_rho = float(training.get("sam", {}).get("rho", 0))
        adversarial_input = training.get("adversarialInput")
        adversarial_epsilon_rms = float(
            adversarial_input.get("epsilonRms", 0)
        ) if adversarial_input is not None else 0.0
        adversarial_steps = int(
            adversarial_input.get("steps", 0)
        ) if adversarial_input is not None else 0
        adversarial_step_size_rms = float(
            adversarial_input.get("stepSizeRms", 0)
        ) if adversarial_input is not None else 0.0
        adversarial_weight = float(
            adversarial_input.get("adversarialWeight", 0)
        ) if adversarial_input is not None else 0.0
        adversarial_return_vector = training.get("adversarialReturnVector")
        if adversarial_return_vector is not None:
            adversarial_epsilon_rms = float(
                adversarial_return_vector["epsilonRms"]
            )
            adversarial_steps = int(adversarial_return_vector["steps"])
            adversarial_step_size_rms = float(
                adversarial_return_vector["stepSizeRms"]
            )
            adversarial_weight = float(
                adversarial_return_vector["adversarialWeight"]
            )
        adversarial_log_price_path = training.get("adversarialLogPricePath")
        if adversarial_log_price_path is not None:
            adversarial_epsilon_rms = float(
                adversarial_log_price_path["epsilonRms"]
            )
            adversarial_steps = int(adversarial_log_price_path["steps"])
            adversarial_step_size_rms = float(
                adversarial_log_price_path["stepSizeRms"]
            )
            adversarial_weight = float(
                adversarial_log_price_path["adversarialWeight"]
            )
        reporter.status(
            "training", planId=plan["id"], startEpoch=start_epoch,
            parameters=parameter_count,
            trainableParameters=trainable_parameter_count,
            parameterPartitionCounts=partition_parameter_counts,
            bestEpoch=best_epoch,
        )
        converged = False
        for epoch in range(start_epoch, maximum_epochs):
            started = time.monotonic()
            active_partition = epoch % 4 if partition_assignments else None
            frozen_epoch = (
                snapshot_inactive_parameter_values(
                    partition_assignments, int(active_partition)
                )
                if active_partition is not None else ()
            )
            model.train()
            online_clean_numerator = 0.0
            online_adversarial_numerator = 0.0
            online_weight_denominator = 0.0
            online_training_objective_numerator = 0.0
            online_training_objective_denominator = 0.0
            online_path_input_delta_rms_numerator = 0.0
            online_path_target_delta_rms_numerator = 0.0
            online_return_input_delta_rms_numerator = 0.0
            online_return_target_delta_rms_numerator = 0.0
            online_irm_risk_numerator = 0.0
            online_irm_penalty_numerator = 0.0
            online_mean_teacher_consistency_numerator = 0.0
            online_mean_teacher_weight_denominator = 0.0
            online_mean_teacher_objective_numerator = 0.0
            active_mean_teacher_weight = 0.0
            training_batches = (
                dataset.iter_assigned_group_batches(
                    "train", batch_size, environment_ids,
                    shuffle=True, seed=seed + epoch, reuse_buffers=True,
                )
                if environment_ids is not None
                else dataset.iter_daily_group_batches(
                    "train", batch_size, shuffle=True, seed=seed + epoch,
                    reuse_buffers=True,
                )
                if cvar_dro is not None
                else dataset.iter_batches(
                    "train", batch_size, shuffle=True, seed=seed + epoch,
                    shuffle_rows=True, reuse_buffers=True,
                    causal_volatility_window=causal_volatility_window,
                )
            )
            for batch in iter_device_batches(training_batches, device):
                features, targets, weights = batch[:3]
                group_ids = (
                    batch[3]
                    if cvar_dro is not None or environment_ids is not None
                    else None
                )
                volatility = (
                    batch[3]
                    if cvar_dro is None
                    and environment_ids is None
                    and len(batch) == 4 else None
                )
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                prediction = model(features, causal_volatility_rms=volatility)
                if invariant_risk is not None:
                    active_penalty_weight = (
                        irm_penalty_weight
                        if global_step >= irm_penalty_anneal_steps else 1.0
                    )
                    clean_loss, irm_risk, irm_penalty = (
                        irm_v1_regression_objective(
                            prediction,
                            targets,
                            weights,
                            group_ids,
                            target_std=target_std,
                            penalty_weight=active_penalty_weight,
                        )
                    )
                    batch_weight = float(weights.sum().detach().item())
                    online_irm_risk_numerator += (
                        float(irm_risk.detach().item()) * batch_weight
                    )
                    online_irm_penalty_numerator += (
                        float(irm_penalty.detach().item()) * batch_weight
                    )
                elif cvar_dro is not None:
                    per_example = (
                        (prediction - targets.float()) / target_std
                    ).square()
                    unique_groups, inverse = torch.unique(
                        group_ids.long(), sorted=True, return_inverse=True
                    )
                    group_numerator = torch.zeros(
                        unique_groups.numel(), device=device,
                        dtype=per_example.dtype,
                    )
                    group_denominator = torch.zeros_like(group_numerator)
                    group_numerator.scatter_add_(
                        0, inverse, per_example * weights
                    )
                    group_denominator.scatter_add_(0, inverse, weights)
                    clean_loss = uniform_group_cvar(
                        group_numerator / group_denominator,
                        cvar_tail_fraction,
                    )
                else:
                    clean_loss = weighted_regression_objective(
                        prediction,
                        targets,
                        weights,
                        target_std,
                        robust_regression,
                    )
                batch_weight = float(weights.sum().detach().item())
                online_training_objective_numerator += (
                    float(clean_loss.detach().item()) * batch_weight
                )
                online_training_objective_denominator += batch_weight
                adversarial_loss = None
                if mean_teacher_model is not None:
                    perturbed_features, _normalized_teacher_noise = (
                        gaussian_normalized_input_perturbation(
                            model,
                            features,
                            epsilon_rms=mean_teacher_input_rms,
                            seed=seed * 1_000_003 + global_step,
                        )
                    )
                    with torch.no_grad():
                        teacher_prediction = mean_teacher_model(features)
                    student_consistency_prediction = model(perturbed_features)
                    consistency_loss = weighted_normalized_mse(
                        student_consistency_prediction,
                        teacher_prediction,
                        weights,
                        target_std,
                    )
                    active_mean_teacher_weight = (
                        mean_teacher_consistency_weight
                        * mean_teacher_ramp_multiplier(
                            global_step,
                            total_optimizer_steps,
                            mean_teacher_ramp_up_fraction,
                        )
                    )
                    loss = (
                        clean_loss
                        + active_mean_teacher_weight * consistency_loss
                    )
                    online_mean_teacher_consistency_numerator += (
                        float(consistency_loss.detach().item()) * batch_weight
                    )
                    online_mean_teacher_objective_numerator += (
                        float(loss.detach().item()) * batch_weight
                    )
                    online_mean_teacher_weight_denominator += batch_weight
                elif adversarial_input is not None:
                    attacked_features, _normalized_delta = (
                        adversarial_input_examples(
                            model,
                            features,
                            targets,
                            weights,
                            target_std=target_std,
                            epsilon_rms=adversarial_epsilon_rms,
                            steps=adversarial_steps,
                            step_size_rms=adversarial_step_size_rms,
                        )
                    )
                    adversarial_prediction = model(attacked_features)
                    adversarial_loss = weighted_normalized_mse(
                        adversarial_prediction, targets, weights, target_std
                    )
                    loss = (
                        (1.0 - adversarial_weight) * clean_loss
                        + adversarial_weight * adversarial_loss
                    )
                    batch_weight = float(weights.sum().detach().item())
                    online_clean_numerator += (
                        float(clean_loss.detach().item()) * batch_weight
                    )
                    online_adversarial_numerator += (
                        float(adversarial_loss.detach().item()) * batch_weight
                    )
                    online_weight_denominator += batch_weight
                elif adversarial_return_vector is not None:
                    (
                        attacked_features,
                        attacked_targets,
                        _normalized_return_delta,
                    ) = adversarial_return_vector_examples(
                        model,
                        features,
                        targets,
                        weights,
                        target_std=target_std,
                        epsilon_rms=adversarial_epsilon_rms,
                        steps=adversarial_steps,
                        step_size_rms=adversarial_step_size_rms,
                    )
                    adversarial_prediction = model(attacked_features)
                    adversarial_loss = weighted_normalized_mse(
                        adversarial_prediction,
                        attacked_targets,
                        weights,
                        target_std,
                    )
                    loss = (
                        (1.0 - adversarial_weight) * clean_loss
                        + adversarial_weight * adversarial_loss
                    )
                    batch_weight = float(weights.sum().detach().item())
                    online_clean_numerator += (
                        float(clean_loss.detach().item()) * batch_weight
                    )
                    online_adversarial_numerator += (
                        float(adversarial_loss.detach().item()) * batch_weight
                    )
                    input_delta_rms = _normalized_return_delta[
                        :, :-1
                    ].square().mean(dim=1).sqrt()
                    target_delta_rms = _normalized_return_delta[:, -1].abs()
                    online_return_input_delta_rms_numerator += float(
                        (input_delta_rms * weights).sum().detach().item()
                    )
                    online_return_target_delta_rms_numerator += float(
                        (target_delta_rms * weights).sum().detach().item()
                    )
                    online_weight_denominator += batch_weight
                elif adversarial_log_price_path is not None:
                    (
                        attacked_features,
                        attacked_targets,
                        _normalized_path_delta,
                    ) = adversarial_log_price_path_examples(
                        model,
                        features,
                        targets,
                        weights,
                        target_std=target_std,
                        epsilon_rms=adversarial_epsilon_rms,
                        steps=adversarial_steps,
                        step_size_rms=adversarial_step_size_rms,
                        perturb_future_endpoint=(
                            adversarial_log_price_path.get(
                                "futureEndpoint", "perturbed"
                            ) == "perturbed"
                        ),
                    )
                    adversarial_prediction = model(attacked_features)
                    adversarial_loss = weighted_normalized_mse(
                        adversarial_prediction,
                        attacked_targets,
                        weights,
                        target_std,
                    )
                    loss = (
                        (1.0 - adversarial_weight) * clean_loss
                        + adversarial_weight * adversarial_loss
                    )
                    batch_weight = float(weights.sum().detach().item())
                    online_clean_numerator += (
                        float(clean_loss.detach().item()) * batch_weight
                    )
                    online_adversarial_numerator += (
                        float(adversarial_loss.detach().item()) * batch_weight
                    )
                    input_delta_rms = (
                        (attacked_features - features.float()) / target_std
                    ).square().mean(dim=1).sqrt()
                    target_delta_rms = (
                        (attacked_targets - targets.float()) / target_std
                    ).abs()
                    online_path_input_delta_rms_numerator += float(
                        (input_delta_rms * weights).sum().detach().item()
                    )
                    online_path_target_delta_rms_numerator += float(
                        (target_delta_rms * weights).sum().detach().item()
                    )
                    online_weight_denominator += batch_weight
                else:
                    loss = clean_loss
                if l2_rate > 0:
                    loss = loss + l2_rate * matrix_l2_penalty(model)
                loss.backward()
                if sam_rho > 0:
                    perturbations = sam_perturb_parameters(model, sam_rho)
                    try:
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        perturbed_prediction = model(
                            features, causal_volatility_rms=volatility
                        )
                        perturbed_clean_loss = weighted_normalized_mse(
                            perturbed_prediction,
                            targets,
                            weights,
                            target_std,
                        )
                        if adversarial_return_vector is not None:
                            (
                                perturbed_attacked_features,
                                perturbed_attacked_targets,
                                _perturbed_return_delta,
                            ) = adversarial_return_vector_examples(
                                model,
                                features,
                                targets,
                                weights,
                                target_std=target_std,
                                epsilon_rms=adversarial_epsilon_rms,
                                steps=adversarial_steps,
                                step_size_rms=adversarial_step_size_rms,
                            )
                            perturbed_adversarial_prediction = model(
                                perturbed_attacked_features
                            )
                            perturbed_adversarial_loss = weighted_normalized_mse(
                                perturbed_adversarial_prediction,
                                perturbed_attacked_targets,
                                weights,
                                target_std,
                            )
                            loss = (
                                (1.0 - adversarial_weight)
                                * perturbed_clean_loss
                                + adversarial_weight
                                * perturbed_adversarial_loss
                            )
                        else:
                            loss = perturbed_clean_loss
                        loss.backward()
                    finally:
                        sam_restore_parameters(perturbations)
                if active_partition is not None:
                    mask_inactive_parameter_updates(
                        partition_assignments,
                        int(active_partition),
                        optimizers,
                    )
                clip_grad_norm_(
                    model.parameters(), float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    optimizer.step()
                if mean_teacher_model is not None:
                    update_ema_teacher(
                        mean_teacher_model,
                        model,
                        mean_teacher_decay,
                    )
                global_step += 1

            assert_inactive_parameter_values_unchanged(frozen_epoch)
            if swa is not None:
                swa.update(model, epoch=epoch)

            evaluation_model = (
                mean_teacher_model
                if mean_teacher_model is not None else model
            )
            if environment_ids is not None:
                deterministic_train, deterministic_eiil = (
                    evaluate_eiil_environments(
                        evaluation_model, dataset, environment_ids,
                        batch_size=eval_batch_size,
                        target_std=target_std,
                        device=device,
                    )
                )
                deterministic_cvar = None
                selection_penalty_weight = (
                    irm_penalty_weight
                    if global_step >= irm_penalty_anneal_steps else 1.0
                )
                score = (
                    float(deterministic_eiil["meanEnvironmentNormalizedMse"])
                    + selection_penalty_weight
                    * float(deterministic_eiil["irmPenalty"])
                ) / max(1.0, selection_penalty_weight)
                convergence_score = float(
                    deterministic_train["normalizedMse"]
                )
            elif cvar_dro is not None:
                deterministic_train, deterministic_cvar = (
                    evaluate_daily_group_cvar(
                        evaluation_model, dataset, "train",
                        batch_size=eval_batch_size,
                        target_std=target_std,
                        tail_fraction=cvar_tail_fraction,
                        device=device,
                    )
                )
                deterministic_eiil = None
                score = float(deterministic_cvar["normalizedMse"])
                convergence_score = score
            else:
                deterministic_train = evaluate(
                    evaluation_model, dataset, "train",
                    batch_size=eval_batch_size,
                    target_std=target_std, device=device,
                    amp_dtype=torch.float32,
                    causal_volatility_window=causal_volatility_window,
                )
                deterministic_cvar = None
                deterministic_eiil = None
                score = float(deterministic_train["normalizedMse"])
                convergence_score = score
            deterministic_validation = (
                evaluate(
                    evaluation_model,
                    validation_dataset,
                    "validation",
                    batch_size=eval_batch_size,
                    target_std=target_std,
                    device=device,
                    amp_dtype=torch.float32,
                    causal_volatility_window=causal_volatility_window,
                )
                if validation_dataset is not None else {}
            )
            l2_penalty = float(
                (l2_rate * matrix_l2_penalty(model)).detach().item()
            ) if l2_rate > 0 else 0.0
            if not math.isfinite(score):
                raise FloatingPointError("training MSE is non-finite")
            improved = score < best_train_score
            if improved:
                best_train_score = score
                best_epoch = epoch
            checkpoint = {
                "model": evaluation_model.state_dict(),
                "studentModel": (
                    model.state_dict()
                    if mean_teacher_model is not None else None
                ),
                "optimizers": [value.state_dict() for value in optimizers],
                "epoch": epoch,
                "globalStep": global_step,
                "bestTrainScore": best_train_score,
                "bestEpoch": best_epoch,
                "train": deterministic_train,
                "trainDailyCvar": deterministic_cvar,
                "trainEiilEnvironments": deterministic_eiil,
                "validation": deterministic_validation,
                "l2Penalty": l2_penalty,
                "objective": score + l2_penalty,
                "parameterCount": parameter_count,
                "trainableParameterCount": trainable_parameter_count,
                "parameterPartitionCounts": partition_parameter_counts,
                "planSha256": plan_hash,
                "corpusFingerprint": fingerprint,
                "runnerContract": active_runner_contract,
                "environmentFingerprint": environment_fingerprint,
                "validationCurveCorpusFingerprint": (
                    validation_curve_fingerprint
                ),
                "swaState": (
                    swa.state_dict() if swa is not None else None
                ),
                "meanTeacher": mean_teacher_config,
            }
            save_torch_checkpoint(checkpoint, last_file)
            if improved:
                save_torch_checkpoint(checkpoint, best_file)
            event = {
                "event": "minute-return-epoch",
                "epoch": epoch,
                "epochs": maximum_epochs,
                "seconds": time.monotonic() - started,
                "globalStep": global_step,
                "train": deterministic_train,
                "trainDailyCvar": deterministic_cvar,
                "trainEiilEnvironments": deterministic_eiil,
                "l2Penalty": l2_penalty,
                "objective": score + l2_penalty,
                "validation": deterministic_validation,
                "bestTrainScore": best_train_score,
                "bestEpoch": best_epoch,
                "improved": improved,
                "learningRate": float(optimizers[0].param_groups[0]["lr"]),
                "activeParameterPartition": active_partition,
                "diagnostic": "deterministic eval-mode train metrics",
                "onlineTrainingObjective": (
                    online_training_objective_numerator
                    / online_training_objective_denominator
                ),
                "onlineIrm": (
                    {
                        "risk": (
                            online_irm_risk_numerator
                            / online_training_objective_denominator
                        ),
                        "penalty": (
                            online_irm_penalty_numerator
                            / online_training_objective_denominator
                        ),
                        "penaltyWeight": (
                            irm_penalty_weight
                            if global_step > irm_penalty_anneal_steps else 1.0
                        ),
                    }
                    if invariant_risk is not None else None
                ),
                "robustRegression": robust_regression,
                "onlineMeanTeacher": (
                    {
                        "consistencyNormalizedMse": (
                            online_mean_teacher_consistency_numerator
                            / online_mean_teacher_weight_denominator
                        ),
                        "combinedObjective": (
                            online_mean_teacher_objective_numerator
                            / online_mean_teacher_weight_denominator
                        ),
                        "activeConsistencyWeight": active_mean_teacher_weight,
                        "emaDecayPerOptimizerStep": mean_teacher_decay,
                    }
                    if mean_teacher_model is not None else None
                ),
                **(
                    {
                        "onlineAdversarialObjectives": {
                            "cleanNormalizedMse": (
                                online_clean_numerator
                                / online_weight_denominator
                            ),
                            "adversarialNormalizedMse": (
                                online_adversarial_numerator
                                / online_weight_denominator
                            ),
                            "combinedNormalizedMse": (
                                (1.0 - adversarial_weight)
                                * online_clean_numerator
                                / online_weight_denominator
                                + adversarial_weight
                                * online_adversarial_numerator
                                / online_weight_denominator
                            ),
                        }
                    }
                    if (
                        adversarial_input is not None
                        or adversarial_return_vector is not None
                        or adversarial_log_price_path is not None
                    ) else {}
                ),
                **(
                    {
                        "onlineAdversarialReturnPerturbation": {
                            "inputReturnDeltaRms": (
                                online_return_input_delta_rms_numerator
                                / online_weight_denominator
                            ),
                            "targetReturnDeltaRms": (
                                online_return_target_delta_rms_numerator
                                / online_weight_denominator
                            ),
                        }
                    }
                    if adversarial_return_vector is not None else {}
                ),
                **(
                    {
                        "onlineAdversarialPathPerturbation": {
                            "inputReturnDeltaRmsInTargetStd": (
                                online_path_input_delta_rms_numerator
                                / online_weight_denominator
                            ),
                            "targetReturnDeltaRmsInTargetStd": (
                                online_path_target_delta_rms_numerator
                                / online_weight_denominator
                            ),
                        }
                    }
                    if adversarial_log_price_path is not None else {}
                ),
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
            if args.pause_file is not None \
                    and resolve(repo, args.pause_file).is_file():
                reporter.status(
                    "paused",
                    planId=plan["id"],
                    latest=event,
                    pausedAt=datetime.now(timezone.utc).isoformat(),
                    message=(
                        "Matrix pause requested; resume from the last completed "
                        "epoch."
                    ),
                    resumableCheckpoint=str(last_file.relative_to(repo)),
                )
                raise SystemExit(PAUSE_EXIT_CODE)
            if convergence_score <= target_score:
                converged = True
                break

        best = load_torch_checkpoint(
            best_file, map_location=device, weights_only=False
        )
        model.load_state_dict(best["model"])
        if environment_ids is not None:
            final_train, final_train_eiil = evaluate_eiil_environments(
                model, dataset, environment_ids,
                batch_size=eval_batch_size,
                target_std=target_std,
                device=device,
            )
            final_train_cvar = None
        elif cvar_dro is not None:
            final_train, final_train_cvar = evaluate_daily_group_cvar(
                model, dataset, "train", batch_size=eval_batch_size,
                target_std=target_std,
                tail_fraction=cvar_tail_fraction,
                device=device,
            )
            final_train_eiil = None
        else:
            final_train = evaluate(
                model, dataset, "train", batch_size=eval_batch_size,
                target_std=target_std, device=device, amp_dtype=torch.float32,
                causal_volatility_window=causal_volatility_window,
            )
            final_train_cvar = None
            final_train_eiil = None
        selected_checkpoint_file = best_file
        swa_candidates = None
        if swa is not None:
            candidate_states = swa.candidate_states()
            candidate_train = {}
            for candidate_id, candidate_state in candidate_states.items():
                model.load_state_dict(candidate_state)
                candidate_train[candidate_id] = evaluate(
                    model, dataset, "train", batch_size=eval_batch_size,
                    target_std=target_std, device=device,
                    amp_dtype=torch.float32,
                    causal_volatility_window=causal_volatility_window,
                )
            selected_checkpoint_file = (
                run_root / "checkpoints/swa-sweep.json"
            )
            save_torch_checkpoint({
                "model": best["model"],
                "swaCandidates": candidate_states,
                "train": final_train,
                "swaCandidateTrain": candidate_train,
                "epoch": swa.completed_epochs - 1,
                "rawBestEpoch": best_epoch,
                "planSha256": plan_hash,
                "corpusFingerprint": fingerprint,
                "runnerContract": active_runner_contract,
            }, selected_checkpoint_file)
            swa_candidates = {
                "raw-best": final_train,
                **candidate_train,
            }
        result = {
            "planId": plan["id"],
            "planSha256": plan_hash,
            "corpusFingerprint": fingerprint,
            "runnerContract": active_runner_contract,
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_parameter_count,
            "parameterPartitionCounts": partition_parameter_counts,
            "examples": dataset.logical_count("train"),
            "bestEpoch": best_epoch,
            "bestTrainScore": best_train_score,
            "train": final_train,
            "trainDailyCvar": final_train_cvar,
            "trainEiilEnvironments": final_train_eiil,
            "validation": best.get("validation", {}),
            "validationCurve": (
                {
                    **validation_curve,
                    "splitPlanId": validation_curve_split_plan_id,
                    "corpusFingerprint": validation_curve_fingerprint,
                }
                if validation_curve is not None else None
            ),
            "l2Rate": l2_rate,
            "optimizerWeightDecay": training.get("optimizerWeightDecay"),
            "sam": training.get("sam"),
            "cvarDro": cvar_dro,
            "environmentInference": environment_summary,
            "invariantRiskMinimization": invariant_risk,
            "meanTeacher": mean_teacher_config,
            "robustRegression": robust_regression,
            "adversarialInput": adversarial_input,
            "adversarialReturnVector": adversarial_return_vector,
            "adversarialLogPricePath": adversarial_log_price_path,
            "datasetFilter": dataset_filter,
            "l2Penalty": (
                float((l2_rate * matrix_l2_penalty(model)).detach().item())
                if l2_rate > 0 else 0.0
            ),
            "targetNormalizedMse": target_score,
            "converged": converged,
            "checkpoint": str(selected_checkpoint_file.relative_to(repo)),
            "swaCandidates": swa_candidates,
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "minute-return-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except KeyboardInterrupt:
        reporter.status("paused", planId=plan["id"], message="Interrupted")
        raise
    except Exception as error:
        reporter.status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
