from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from low_rank_path_matrix_density import (
    CYCLIC_DENSE_COMPRESSED_ARCHITECTURE_CONTRACT,
    JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT,
    CyclicDenseCompressedPathMatrixDensity,
    JointPrefixContractedCyclicPathMatrixDensity,
)
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint
from train_feature_compressed_path_density import (
    CYCLIC_DENSE_COMPRESSED_PATH_MATRIX_RUNNER_CONTRACT,
    DISCRETE_CRPS_DISTRIBUTION_LOSS,
    UNION_IMMEDIATE_CYCLIC_DENSE_PATH_MATRIX_RUNNER_CONTRACT,
    UNION_JOINT_PREFIX_CYCLIC_PATH_MATRIX_RUNNER_CONTRACT,
    CalibrationPathDataset,
    ImmediateFeatureActivePathDataset,
    UnionImmediateCalibrationDataset,
    UnionImmediateFeatureActivePathDataset,
    calibration_scales,
    collect_expectation_arrays,
    distribution_loss_type,
    evaluate,
    expectation_metrics_from_arrays,
    rolling_online_per_step_affine,
    training_statistics,
)
from union530_base_dataset import DifferentiableUnion530Dataset
from train_next_return_knot_density import canonical_hash
from train_normalized_glu_next_return import Reporter, atomic_json


CONTRACT = "stopped-cyclic-path-density-checkpoint-evaluation-v1"


def publish_terminal_result(
    repo: Path,
    run_root: Path,
    plan: dict,
    evaluation: dict,
    *,
    runner_contract: str,
    parameter_count: int,
    examples: int,
) -> dict:
    """Publish stopped-run metrics through the same contract as natural completion."""
    policies = evaluation["policies"]
    crps_training = (
        distribution_loss_type(plan["training"])
        == DISCRETE_CRPS_DISTRIBUTION_LOSS
    )
    selected_policy, selected = min(
        policies.items(),
        key=lambda item: float(
            item[1]["distribution"]["validation"][
                "normalizedCrps" if crps_training else "negativeLogLikelihood"
            ]
        ),
    )
    comparison = {
        "contract": "compressed-path-checkpoint-selection-comparison-v1",
        "policies": {
            "validation-nll": policies["best-validation-nll"],
            "validation-mse": policies["best-validation-mse"],
            "validation-correlation": policies["best-validation-correlation"],
            **({
                "validation-crps": policies["best-validation-crps"],
            } if crps_training else {}),
            "last": policies["last"],
        },
        "selectedPolicy": selected_policy,
    }
    atomic_json(
        comparison, run_root / "state/checkpoint-selection-comparison.json"
    )
    training = plan["training"]
    terminal = {
        "version": 1,
        "planId": plan["id"],
        "planSha256": evaluation["planSha256"],
        "runnerContract": runner_contract,
        "stoppedEarly": True,
        "examples": int(examples),
        "featureCount": int(evaluation["featureCount"]),
        "parameterCount": int(parameter_count),
        "bestEpoch": int(selected["epoch"]),
        "selectionPolicy": (
            "validation-crps" if crps_training else "validation-nll"
        ),
        "checkpointPolicy": selected_policy,
        "bestValidationScore": float(
            selected["distribution"]["validation"][
                "normalizedCrps" if crps_training else "negativeLogLikelihood"
            ]
        ),
        "train": selected["train"],
        "validation": selected["validation"],
        "test": selected["test"],
        "distribution": selected["distribution"],
        "robustTraining": {
            "samRho": 0.0 if training.get("sam") is None else float(
                training["sam"]["rho"]
            ),
            "inputDropoutProbability": float(
                training.get("inputDropoutProbability", 0.0)
            ),
            "embeddingDropoutProbability": float(
                training.get("embeddingDropoutProbability", 0.0)
            ),
            "embeddingDropoutSchedule": training.get("embeddingDropoutSchedule"),
            "expectedReturnCorrelationLossWeight": float(
                training.get("expectedReturnCorrelationLossWeight", 0.0)
            ),
            "expectedReturnCorrelationLossWeightSchedule": training.get(
                "expectedReturnCorrelationLossWeightSchedule"
            ),
            "expectedReturnMseLossWeight": float(
                training.get("expectedReturnMseLossWeight", 0.0)
            ),
            "expectedReturnMseLossWeightSchedule": training.get(
                "expectedReturnMseLossWeightSchedule"
            ),
            "weightEmaHalfLifeEpochs": float(
                training["weightEma"]["halfLifeEpochs"]
            ),
            "adversarialInput": training.get("adversarialInput"),
            "adversarialOutput": training.get("adversarialOutput"),
            "distributionLoss": training.get("distributionLoss"),
        },
        "validationCalibration": plan["validationCalibration"],
        "evaluationArtifact": str(
            (
                repo / "data/benchmarks"
                / f"{plan['id']}-stopped-eval.json"
            ).relative_to(repo)
        ),
    }
    atomic_json(terminal, run_root / "state/result.json")
    reporter = Reporter(run_root)
    reporter.emit({"event": "stopped-run-checkpoints-evaluated", **terminal})
    reporter.status("complete", planId=plan["id"], latest=terminal)
    return terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected and final EMA cyclic path checkpoints."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def build_model(
    plan: dict,
    state: dict[str, torch.Tensor],
    density: KnotDensityContract,
    device: torch.device,
) -> CyclicDenseCompressedPathMatrixDensity:
    architecture = plan["architecture"]
    architecture_contract = architecture.get("contract")
    if architecture_contract not in {
        CYCLIC_DENSE_COMPRESSED_ARCHITECTURE_CONTRACT,
        JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT,
    }:
        raise ValueError("this evaluator requires the cyclic dense path model")
    target_normalization = plan.get("targetNormalization")
    joint_prefix = (
        architecture_contract
        == JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT
    )
    model_type = (
        JointPrefixContractedCyclicPathMatrixDensity
        if joint_prefix else CyclicDenseCompressedPathMatrixDensity
    )
    model = model_type(
        state["feature_mean"],
        state["feature_std"],
        density,
        market_width=int(architecture["marketWidth"]),
        path_embedding_width=int(architecture["pathEmbeddingWidth"]),
        path_count=int(architecture["pathCount"]),
        return_count=int(architecture["returnCount"]),
        stage_block_count=int(architecture["stageBlockCount"]),
        path_compression_width=int(architecture["pathCompressionWidth"]),
        joint_compression_width=int(architecture["jointCompressionWidth"]),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture["learnableCentering"]),
        input_dropout_probability=float(
            plan["training"].get("inputDropoutProbability", 0.0)
        ),
        embedding_dropout_probability=float(
            plan["training"].get("embeddingDropoutProbability", 0.0)
        ),
        target_normalization_variance_floor=(
            None if target_normalization is None else float(
                target_normalization.get("varianceFloor", 1e-16)
            )
        ),
        target_normalization_center=(
            target_normalization is None
            or target_normalization.get("type")
            == "causal-trailing-log-return-zscore"
        ),
        **({
            "recurrent_activation_checkpointing": bool(
                architecture.get("recurrentActivationCheckpointing", False)
            ),
        } if joint_prefix else {}),
    )
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def evaluate_state(
    model: CyclicDenseCompressedPathMatrixDensity,
    dataset,
    calibration_dataset,
    *,
    batch_size: int,
    target_std: float,
    cumulative_std: float,
    calibration_window: int,
    calibration_ridge: float,
    device: torch.device,
) -> dict:
    values = {
        "train": evaluate(
            model, dataset, "train", batch_size=batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
        ),
        "validation": evaluate(
            model, dataset, "validation", batch_size=batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
            collect_arrays=True,
        ),
        "test": evaluate(
            model, dataset, "test", batch_size=batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
            collect_arrays=True,
        ),
    }
    calibration_prediction, calibration_target = collect_expectation_arrays(
        model, calibration_dataset, "calibration",
        batch_size=batch_size, device=device,
    )
    calibration_prediction = calibration_prediction[-calibration_window:]
    calibration_target = calibration_target[-calibration_window:]
    input_scales, output_scales = calibration_scales(
        calibration_prediction, calibration_target
    )
    validation_arrays = values["validation"].pop("arrays")
    test_arrays = values["test"].pop("arrays")
    calibrated_validation_prediction = rolling_online_per_step_affine(
        calibration_prediction,
        calibration_target,
        validation_arrays["prediction"],
        validation_arrays["target"],
        window=calibration_window,
        ridge=calibration_ridge,
        input_scales=input_scales,
        output_scales=output_scales,
    )
    calibrated_test_prediction = rolling_online_per_step_affine(
        validation_arrays["prediction"],
        validation_arrays["target"],
        test_arrays["prediction"],
        test_arrays["target"],
        window=calibration_window,
        ridge=calibration_ridge,
        input_scales=input_scales,
        output_scales=output_scales,
    )
    calibrated_validation = expectation_metrics_from_arrays(
        calibrated_validation_prediction,
        validation_arrays["target"],
        target_std=target_std,
        cumulative_std=cumulative_std,
        device=device,
    )
    calibrated_test = expectation_metrics_from_arrays(
        calibrated_test_prediction,
        test_arrays["target"],
        target_std=target_std,
        cumulative_std=cumulative_std,
        device=device,
    )
    return {
        "evaluationHorizon": {"steps": 1, "seconds": 1, "lead": 0},
        "train": values["train"]["perLeadExpectation"][0],
        "validation": calibrated_validation["perLeadExpectation"][0],
        "test": calibrated_test["perLeadExpectation"][0],
        "rawValidation": values["validation"]["perLeadExpectation"][0],
        "rawTest": values["test"]["perLeadExpectation"][0],
        "calibratedValidation": calibrated_validation,
        "calibratedTest": calibrated_test,
        "distribution": values,
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text("utf-8"))
    plan_hash = canonical_hash(plan)
    run_root = (repo / plan["runDir"]).resolve()
    status_file = run_root / "state/status.json"
    status_before = json.loads(status_file.read_text("utf-8")) \
        if status_file.is_file() else {}
    was_naturally_complete = (
        status_before.get("stage") == "complete"
        and (run_root / "state/result.json").is_file()
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    architecture = plan["architecture"]
    return_count = int(architecture["returnCount"])
    feature_history = int(architecture.get("inputFeatureLags", 1))
    target_normalization = plan.get("targetNormalization")
    if target_normalization is None:
        normalization_window_seconds = None
        normalization_variance_floor = 1e-16
        normalization_statistic = "log-return"
    else:
        normalization_type = target_normalization.get("type")
        if normalization_type not in {
            "causal-trailing-log-return-zscore",
            "causal-trailing-log-price-zscore-difference",
        }:
            raise ValueError("unsupported target normalization")
        normalization_window_seconds = int(
            target_normalization["windowSeconds"]
        )
        normalization_variance_floor = float(
            target_normalization.get("varianceFloor", 1e-16)
        )
        normalization_statistic = (
            "log-price"
            if normalization_type
            == "causal-trailing-log-price-zscore-difference"
            else "log-return"
        )
    base_history_union = plan.get("baseHistoryDatasetDir") is not None
    union_immediate = (
        plan.get("unionHistoryDatasetDir") is not None or base_history_union
    )
    if union_immediate:
        if base_history_union:
            dataset = DifferentiableUnion530Dataset(
                (repo / plan["baseHistoryDatasetDir"]).resolve(),
                examples_by_split={
                    name: int(value)
                    for name, value in plan["examplesBySplit"].items()
                },
                return_count=return_count,
            )
        else:
            dataset = UnionImmediateFeatureActivePathDataset(
                (repo / plan["datasetDir"]).resolve(),
                (repo / plan["unionHistoryDatasetDir"]).resolve(),
                return_count=return_count,
                examples_by_split={
                    name: int(value)
                    for name, value in plan["examplesBySplit"].items()
                },
            )
        calibration_dataset = UnionImmediateCalibrationDataset(
            dataset, int(plan["validationCalibration"]["examples"])
        )
        expected_runner = (
            UNION_JOINT_PREFIX_CYCLIC_PATH_MATRIX_RUNNER_CONTRACT
            if architecture.get("contract")
            == JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT
            else UNION_IMMEDIATE_CYCLIC_DENSE_PATH_MATRIX_RUNNER_CONTRACT
        )
    else:
        history_root = (repo / plan["historyDir"]).resolve()
        dataset = ImmediateFeatureActivePathDataset(
            (repo / plan["datasetDir"]).resolve(),
            history_root,
            return_count,
            feature_history,
            int(plan["subset"]["examples"]),
            normalization_window_seconds,
            normalization_variance_floor,
            normalization_statistic,
        )
        calibration_dataset = CalibrationPathDataset(
            (repo / plan["validationCalibration"]["datasetDir"]).resolve(),
            return_count,
            feature_history,
            history_root=history_root,
            normalization_window_seconds=normalization_window_seconds,
            normalization_variance_floor=normalization_variance_floor,
            normalization_statistic=normalization_statistic,
        )
        expected_runner = (
            UNION_JOINT_PREFIX_CYCLIC_PATH_MATRIX_RUNNER_CONTRACT
            if architecture.get("contract")
            == JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT
            else CYCLIC_DENSE_COMPRESSED_PATH_MATRIX_RUNNER_CONTRACT
        )
    batch_size = int(
        args.batch_size or plan["training"]["evaluationBatchSize"]
    )
    stats = training_statistics(dataset, batch_size)
    density = KnotDensityContract.load(
        (repo / plan["density"]["source"]).resolve(),
        fit=str(int(architecture["outputKnots"])),
    )
    calibration_spec = plan["validationCalibration"]
    calibration_window = int(calibration_spec["windowActiveReturns"])
    calibration_ridge = float(calibration_spec.get("ridge", 0.0))
    checkpoints = {
        "best-validation-nll": (
            run_root / "checkpoints/selections/validation-nll.json", "model"
        ),
        "best-validation-mse": (
            run_root / "checkpoints/selections/validation-mse.json", "model"
        ),
        "best-validation-correlation": (
            run_root / "checkpoints/selections/validation-correlation.json", "model"
        ),
        "last": (run_root / "checkpoints/last.json", "emaModel"),
    }
    if distribution_loss_type(plan["training"]) \
            == DISCRETE_CRPS_DISTRIBUTION_LOSS:
        checkpoints["best-validation-crps"] = (
            run_root / "checkpoints/selections/validation-crps.json", "model"
        )
    policies: dict[str, dict] = {}
    parameter_count: int | None = None
    for label, (checkpoint_file, state_key) in checkpoints.items():
        saved = load_torch_checkpoint(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        if saved.get("planSha256") != plan_hash:
            raise ValueError(f"{label} checkpoint belongs to another plan")
        if saved.get("runnerContract") != expected_runner:
            raise ValueError(f"{label} checkpoint runner contract changed")
        model = build_model(plan, saved[state_key], density, device)
        if parameter_count is None:
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
        policies[label] = {
            "epoch": int(saved["epoch"]),
            "globalStep": saved.get("globalStep"),
            "selectionScore": saved.get("score"),
            **evaluate_state(
                model,
                dataset,
                calibration_dataset,
                batch_size=batch_size,
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]),
                calibration_window=calibration_window,
                calibration_ridge=calibration_ridge,
                device=device,
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "contract": CONTRACT,
        "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "planId": plan["id"],
        "planSha256": plan_hash,
        "featureCount": dataset.feature_count,
        "returnCount": return_count,
        "validationCalibration": calibration_spec,
        "policies": policies,
    }
    artifact = repo / "data/benchmarks" / f"{plan['id']}-stopped-eval.json"
    atomic_json(result, artifact)
    atomic_json(result, run_root / "state/stopped-evaluation.json")
    assert parameter_count is not None
    if not was_naturally_complete:
        publish_terminal_result(
            repo,
            run_root,
            plan,
            result,
            runner_contract=expected_runner,
            parameter_count=parameter_count,
            examples=dataset.logical_count("train"),
        )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
