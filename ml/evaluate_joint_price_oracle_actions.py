"""Evaluate execution-aligned joint price/oracle actions without training.

Validation is the default and safe split.  Reading the sealed chronological
test split requires the explicit ``--allow-test`` flag.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from joint_price_oracle import parameter_count
from joint_price_oracle_actions import (
    actionable_policy_metrics_numpy,
    exact_state_actionable_policy_metrics_numpy,
    resolve_execution_policy_config,
)
from trading_storage import (
    load_torch_checkpoint,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    DATA_CONTRACT,
    CausalOracleDataset,
    CausalSegment,
    LEGACY_UNFINGERPRINTED_PLAN_IDS,
    PrefetchedBatchIterator,
    architecture_contract_for_model_config,
    autocast_context,
    build_model,
    configuration_fingerprint,
    load_causal_segments,
    load_exact_state_trace_providers,
    move_batch,
    resolve,
    resolve_training_config,
    unpack_batch,
    unpack_policy_only_batch,
    validate_plan,
)


ACTION_TEMPERATURE_CALIBRATION_METHOD = (
    "validation-action-temperature-scaling-v1"
)
ACTION_TEMPERATURE_OBJECTIVE = "actions.signedTransitionF1"
EXACT_STATE_ACTION_TEMPERATURE_OBJECTIVE = (
    "exactStateActions.signedTransitionF1"
)
ACTION_TEMPERATURE_TIE_BREAKERS = (
    "actions.signedTransitionPrecision:maximize",
    "actions.signedTransitionRecall:maximize",
    "actions.exactTransitionF1:maximize",
    "actions.pathDirectionalAgreement:maximize",
    "actions.pathMeanAbsoluteError:minimize",
    "actions.turnoverRatioDistanceFromOne:minimize",
    "logitTemperatureLogDistanceFromIdentity:minimize",
    "logitTemperature:minimize",
)
EXACT_STATE_ACTION_TEMPERATURE_TIE_BREAKERS = (
    "exactStateActions.signedTransitionPrecision:maximize",
    "exactStateActions.signedTransitionRecall:maximize",
    "exactStateActions.exactTransitionF1:maximize",
    "exactStateActions.executableTargetDirectionalAgreement:maximize",
    "exactStateActions.executableTargetMeanAbsoluteError:minimize",
    "exactStateActions.turnoverRelativeError:minimize",
    "logitTemperatureLogDistanceFromIdentity:minimize",
    "logitTemperature:minimize",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chronological, transition-conditioned joint "
            "price/oracle actions from a durable checkpoint."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly permit reading and evaluating the sealed test split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path. Relative paths resolve from the repo root.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the plan's evaluation batch size.",
    )
    parser.add_argument(
        "--logit-temperatures",
        default="1",
        help=(
            "Comma-separated positive validation-only temperatures. Inference "
            "runs once and every value is scored on the same ordered logits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = evaluate_plan_checkpoint(
        arguments.plan,
        checkpoint_kind=arguments.checkpoint,
        split=arguments.split,
        allow_test=arguments.allow_test,
        output=arguments.output,
        requested_device=arguments.device,
        batch_size=arguments.batch_size,
        logit_temperatures=parse_positive_temperatures(
            arguments.logit_temperatures
        ),
    )
    print(json.dumps(report, indent=2, allow_nan=False))


def evaluate_plan_checkpoint(
    plan_file: Path,
    *,
    checkpoint_kind: str = "best",
    split: str = "validation",
    allow_test: bool = False,
    output: Path | None = None,
    requested_device: str = "auto",
    batch_size: int | None = None,
    logit_temperatures: tuple[float, ...] = (1.0,),
) -> dict:
    """Load one plan/checkpoint and evaluate a complete chronological split."""
    validate_split_access(split, allow_test)
    if checkpoint_kind not in {"best", "last"}:
        raise ValueError("checkpoint must be best or last")
    temperatures = validated_temperatures(logit_temperatures)
    repo_root = Path(__file__).resolve().parents[1]
    storage = training_storage_layout(repo_root)
    resolved_plan_file = resolve(repo_root, plan_file).resolve()
    plan = json.loads(resolved_plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    model_config = plan["model"]
    training = plan["training"]
    resolved_training = resolve_training_config(training)
    training_fingerprint = configuration_fingerprint(resolved_training)
    execution_policy = resolved_action_execution_policy(resolved_training)
    resolved_action_objective = resolved_training.get("actionObjective")
    rollout_score_version = (
        int(resolved_action_objective.get("rolloutScoreVersion", 1))
        if isinstance(resolved_action_objective, dict) else 1
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        storage.runs,
        "runDir",
    )
    target_root = require_under(
        resolve(repo_root, Path(plan["targetReferenceDir"])),
        storage.immutable / "refs" / "oracle",
        "targetReferenceDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    (
        segments,
        target_manifest,
        excluded_dates,
        dataset_fingerprint,
    ) = load_causal_segments(
        target_root,
        plan,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
    )
    (
        exact_state_providers,
        teacher_state_provenance,
    ) = load_exact_state_trace_providers(
        repo_root,
        target_root,
        segments,
        resolved_action_objective,
        splits=(split,),
        allow_test=allow_test,
    )
    device = resolve_device(requested_device, str(training["device"]))
    evaluation_batch_size = (
        int(training["evaluationBatchSize"])
        if batch_size is None else int(batch_size)
    )
    if evaluation_batch_size < 1:
        raise ValueError("batch size must be positive")
    pin_memory = device.type == "cuda"
    policy_only = bool(training.get("policyOnly", False))
    dataset = CausalOracleDataset(
        history_root,
        segments,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
        target_rows_per_file=1_440,
        action_count=int(model_config["actionCount"]),
        close_cache_days=int(training.get("closeCacheDays", 10)),
        target_cache_days=int(training.get("targetCacheDays", 3)),
        pin_memory=pin_memory,
        include_future_closes=not policy_only,
    )
    model = build_model(model_config).to(device)
    model_parameters = parameter_count(model)
    checkpoint_file = run_dir / "checkpoints" / f"{checkpoint_kind}.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    validate_checkpoint(
        checkpoint,
        plan,
        model_config,
        dataset_fingerprint,
        model_parameters,
        training_fingerprint,
        teacher_state_provenance=(
            teacher_state_provenance.get(split)
            if split in exact_state_providers else None
        ),
        split=split,
    )
    model.load_state_dict(checkpoint["model"])
    logits, targets = collect_policy_rows(
        model,
        dataset,
        split,
        evaluation_batch_size,
        device,
        training,
    )
    expected_count = dataset.count(split)
    if logits.shape != (expected_count, int(model_config["actionCount"])) \
            or targets.shape != logits.shape:
        raise RuntimeError(
            "chronological inference rows do not match the selected split"
        )
    reset_mask = chronological_reset_mask(segments[split])
    if reset_mask.shape != (expected_count,):
        raise RuntimeError("rollout reset mask does not match inference rows")
    contract = target_manifest["contract"]
    grid = np.asarray(contract["usableGrid"], dtype=np.float64)
    friction = float(contract["options"]["friction"])
    temperature = float(contract["options"]["temperature"])
    temperature_sweep = []
    exact_current_exposures = None
    exact_provider = exact_state_providers.get(split)
    if exact_provider is not None:
        exact_current_exposures = np.concatenate([
            exact_provider(split, segment)
            for segment in segments[split]
        ])
        if exact_current_exposures.shape != (expected_count,):
            raise RuntimeError(
                "exact-state trace rows do not match chronological inference"
            )
    for logit_temperature in temperatures:
        scaled_logits = logits / logit_temperature
        surrogate_actions = actionable_policy_metrics_numpy(
            scaled_logits,
            targets,
            grid,
            reset_mask=reset_mask,
            friction=friction,
            temperature=temperature,
            execution_policy=execution_policy,
        )
        sweep_item = {
            "logitTemperature": logit_temperature,
            "raw": raw_distribution_metrics(scaled_logits, targets),
            # ``actions`` remains as a compatibility alias for v3 reports.
            "actions": surrogate_actions,
            "surrogateRolloutActions": surrogate_actions,
        }
        if exact_current_exposures is not None:
            sweep_item["exactStateActions"] = (
                exact_state_actionable_policy_metrics_numpy(
                scaled_logits,
                targets,
                grid,
                exact_current_exposures,
                friction=friction,
                temperature=temperature,
                execution_policy=execution_policy,
                )
            )
        temperature_sweep.append(sweep_item)
    primary = next(
        (
            item for item in temperature_sweep
            if item["logitTemperature"] == 1.0
        ),
        temperature_sweep[0],
    )
    calibration_selection = select_action_temperature(temperature_sweep)
    best_transition = temperature_sweep[
        calibration_selection["selectedIndex"]
    ]
    split_segments = segments[split]
    report = {
        "version": 3,
        "evaluatedAt": iso_now(),
        "plan": {
            "id": plan["id"],
            "file": str(resolved_plan_file),
        },
        "checkpoint": {
            "kind": checkpoint_kind,
            "file": str(checkpoint_file),
            "epoch": int(checkpoint["epoch"]),
            "globalStep": int(checkpoint["globalStep"]),
            "validation": checkpoint.get("validation"),
        },
        "split": {
            "name": split,
            "examples": expected_count,
            "segmentCount": len(split_segments),
            "rolloutResetCount": int(reset_mask.sum()),
            "firstPredictionTime": int(
                split_segments[0].prediction_time_start
            ),
            "lastPredictionTime": int(
                split_segments[-1].prediction_time_end
            ),
            "excludedTargetDates": excluded_dates,
        },
        "inference": {
            "device": str(device),
            "batchSize": evaluation_batch_size,
            "parameterCount": model_parameters,
        },
        "oracle": {
            "friction": friction,
            "temperature": temperature,
            "actionCount": grid.size,
        },
        "actionEvaluation": {
            "executionPolicy": execution_policy,
            "rolloutScoreVersion": rollout_score_version,
            "rolloutResetPolicy": "reset-only-at-true-timeline-gaps",
            "teacherStateProvenance": teacher_state_provenance.get(split),
        },
        "primaryLogitTemperature": primary["logitTemperature"],
        "raw": primary["raw"],
        "actions": primary["actions"],
        "surrogateRolloutActions": primary["surrogateRolloutActions"],
        "exactStateActions": primary.get("exactStateActions"),
        "temperatureSweep": temperature_sweep,
        "actionTemperatureCalibration": calibration_selection,
        "bestSignedTransitionF1Temperature": best_transition[
            "logitTemperature"
        ],
        "datasetFingerprint": dataset_fingerprint,
    }
    report = finite_json_value(report)
    if output is not None:
        output_file = resolve(repo_root, output).resolve()
        atomic_json(report, output_file)
    return report


def select_action_temperature(temperature_sweep: list[dict]) -> dict:
    """Select an executable-policy temperature with stable tie-breaks.

    Raw KL is intentionally not part of this ranking.  Temperature changes
    the balance between model logits and execution friction, so calibration
    must be selected on a chronological rollout metric.  The last two keys
    make the result independent of candidate input order when action metrics
    tie exactly.
    """
    if not temperature_sweep:
        raise ValueError("temperature sweep cannot be empty")
    exact_flags = [
        isinstance(item.get("exactStateActions"), dict)
        for item in temperature_sweep
    ]
    if any(exact_flags) and not all(exact_flags):
        raise ValueError(
            "temperature sweep cannot mix exact-state and surrogate-only rows"
        )
    use_exact_state = all(exact_flags)
    objective = (
        EXACT_STATE_ACTION_TEMPERATURE_OBJECTIVE
        if use_exact_state else ACTION_TEMPERATURE_OBJECTIVE
    )
    tie_breakers = (
        EXACT_STATE_ACTION_TEMPERATURE_TIE_BREAKERS
        if use_exact_state else ACTION_TEMPERATURE_TIE_BREAKERS
    )
    scored: list[tuple[tuple[float, ...], int]] = []
    seen: set[float] = set()
    for index, item in enumerate(temperature_sweep):
        temperature = _finite_number(
            item.get("logitTemperature"),
            "logit temperature",
        )
        if temperature <= 0 or temperature in seen:
            raise ValueError(
                "temperature sweep values must be positive and unique"
            )
        seen.add(temperature)
        actions = item.get(
            "exactStateActions" if use_exact_state else "actions"
        )
        if not isinstance(actions, dict):
            raise ValueError("temperature sweep item has no action metrics")
        if use_exact_state:
            direction_agreement = _finite_number(
                actions.get("executableTargetDirectionalAgreement"),
                "executable target directional agreement",
            )
            path_error = _finite_number(
                actions.get("executableTargetMeanAbsoluteError"),
                "executable target mean absolute error",
            )
            turnover_distance = _finite_number(
                actions.get("turnoverRelativeError"),
                "turnover relative error",
            )
        else:
            direction_agreement = _finite_number(
                actions.get("pathDirectionalAgreement"),
                "path directional agreement",
            )
            path_error = _finite_number(
                actions.get("pathMeanAbsoluteError"),
                "path mean absolute error",
            )
            turnover_ratio = _finite_or_infinite_number(
                actions.get("turnoverRatio"),
                "turnover ratio",
            )
            turnover_distance = (
                abs(turnover_ratio - 1)
                if math.isfinite(turnover_ratio) else math.inf
            )
        key = (
            _finite_number(
                actions.get("signedTransitionF1"),
                "signed transition F1",
            ),
            _finite_number(
                actions.get("signedTransitionPrecision"),
                "signed transition precision",
            ),
            _finite_number(
                actions.get("signedTransitionRecall"),
                "signed transition recall",
            ),
            _finite_number(
                actions.get("exactTransitionF1"),
                "exact transition F1",
            ),
            _finite_number(
                direction_agreement,
                "directional agreement",
            ),
            -_finite_number(
                path_error,
                "target/path mean absolute error",
            ),
            -turnover_distance,
            -abs(math.log(temperature)),
            -temperature,
        )
        scored.append((key, index))
    _key, selected_index = max(scored, key=lambda value: value[0])
    selected = temperature_sweep[selected_index]
    identity_index = next((
        index for index, item in enumerate(temperature_sweep)
        if float(item["logitTemperature"]) == 1.0
    ), None)
    selected_metrics = {
        "raw": selected.get("raw"),
        "actions": selected["actions"],
    }
    for name in ("surrogateRolloutActions", "exactStateActions"):
        if name in selected:
            selected_metrics[name] = selected[name]
    identity_metrics = None
    if identity_index is not None:
        identity_item = temperature_sweep[identity_index]
        identity_metrics = {
            "raw": identity_item.get("raw"),
            "actions": identity_item["actions"],
        }
        for name in ("surrogateRolloutActions", "exactStateActions"):
            if name in identity_item:
                identity_metrics[name] = identity_item[name]
    return {
        "method": ACTION_TEMPERATURE_CALIBRATION_METHOD,
        "objective": {
            "metric": objective,
            "direction": "maximize",
        },
        "tieBreakers": list(tie_breakers),
        "candidateCount": len(temperature_sweep),
        "selectedIndex": selected_index,
        "selectedLogitTemperature": selected["logitTemperature"],
        "selectedMetrics": selected_metrics,
        "identityIndex": identity_index,
        "identityMetrics": identity_metrics,
    }


def _finite_number(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_or_infinite_number(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if math.isnan(result):
        raise ValueError(f"{name} cannot be NaN")
    return result


def parse_positive_temperatures(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("logit temperatures must be comma-separated numbers") \
            from error
    return validated_temperatures(values)


def resolved_action_execution_policy(training: dict) -> dict | None:
    """Resolve the plan's opt-in bot execution semantics for every sweep row."""
    action_objective = training.get("actionObjective")
    if not isinstance(action_objective, dict) \
            or "executionPolicy" not in action_objective:
        return None
    return resolve_execution_policy_config(action_objective["executionPolicy"])


def validated_temperatures(values: Iterable[float]) -> tuple[float, ...]:
    result: list[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("logit temperatures must be finite and positive")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one logit temperature is required")
    return tuple(result)


@torch.no_grad()
def collect_policy_rows(
    model,
    dataset: CausalOracleDataset,
    split: str,
    batch_size: int,
    device: torch.device,
    training: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ordered batches and concatenate base logits and target rows."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    model.eval()
    policy_only = bool(training.get("policyOnly", False))
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batch_source = dataset.iter_batches(
        split,
        batch_size,
        shuffle=False,
        seed=int(training["seed"]),
        maximum_batches=None,
    )
    batches = dataset.batch_count(split, batch_size)
    progress_every = max(1, batches // 10)
    with PrefetchedBatchIterator(
        batch_source,
        int(training.get("prefetchBatches", 2)),
    ) as prefetched_batches:
        for index, batch in enumerate(prefetched_batches, start=1):
            moved_batch = move_batch(batch, device)
            if policy_only:
                input_closes, target_policy = unpack_policy_only_batch(
                    moved_batch
                )
            else:
                input_closes, _future_closes, target_policy, _teacher = (
                    unpack_batch(moved_batch)
                )
            with autocast_context(device, training):
                if policy_only:
                    predicted = model.forward_policy_logits(input_closes)
                else:
                    predicted = model(input_closes)
            logits.append(predicted.float().cpu().numpy())
            targets.append(target_policy.float().cpu().numpy())
            if index == batches or index % progress_every == 0:
                print(
                    f"ACTION-EVAL {split} batches {index}/{batches}",
                    file=sys.stderr,
                    flush=True,
                )
    if not logits:
        raise RuntimeError(f"{split} produced no evaluation batches")
    return np.concatenate(logits), np.concatenate(targets)


def chronological_reset_mask(
    segments: Iterable[CausalSegment],
) -> np.ndarray:
    """Reset only at true timeline gaps, not ordinary daily shard boundaries."""
    values = list(segments)
    if not values:
        raise ValueError("cannot construct resets for an empty split")
    count = sum(segment.count for segment in values)
    result = np.zeros(count, dtype=np.bool_)
    cursor = 0
    previous: CausalSegment | None = None
    for segment in values:
        if segment.count < 1:
            raise ValueError("causal segments must be non-empty")
        if previous is not None \
                and segment.prediction_time_start <= previous.prediction_time_end:
            raise ValueError("causal segments must be ordered and non-overlapping")
        continuous = (
            previous is not None
            and segment.step_ms == previous.step_ms
            and segment.prediction_time_start
            == previous.prediction_time_end + previous.step_ms
        )
        if not continuous:
            result[cursor] = True
        cursor += segment.count
        previous = segment
    if cursor != count:
        raise RuntimeError("causal segment reset accounting failed")
    return result


def raw_distribution_metrics(
    predicted_base_logits: np.ndarray,
    target_base_probabilities: np.ndarray,
) -> dict[str, float]:
    """Return raw forward KL and target/prediction entropy diagnostics."""
    logits = np.asarray(predicted_base_logits, dtype=np.float64)
    target = np.asarray(target_base_probabilities, dtype=np.float64)
    if logits.ndim != 2 or target.shape != logits.shape \
            or not np.isfinite(logits).all() \
            or not np.isfinite(target).all() \
            or bool((target < 0).any()):
        raise ValueError("raw metrics require finite matching action rows")
    totals = target.sum(axis=-1, keepdims=True)
    if bool((totals <= 0).any()):
        raise ValueError("raw target rows must have positive probability mass")
    target = target / totals
    maximum = logits.max(axis=-1, keepdims=True)
    log_normalizer = maximum + np.log(
        np.exp(logits - maximum).sum(axis=-1, keepdims=True)
    )
    predicted_log = logits - log_normalizer
    predicted = np.exp(predicted_log)
    with np.errstate(divide="ignore", invalid="ignore"):
        target_log = np.where(target > 0, np.log(target), 0.0)
    target_entropy_per_row = -(target * target_log).sum(axis=-1)
    predicted_entropy_per_row = -(predicted * predicted_log).sum(axis=-1)
    cross_entropy_per_row = -(target * predicted_log).sum(axis=-1)
    kl_per_row = cross_entropy_per_row - target_entropy_per_row
    return {
        "crossEntropy": float(cross_entropy_per_row.mean()),
        "klDivergence": float(kl_per_row.mean()),
        "targetEntropy": float(target_entropy_per_row.mean()),
        "predictedEntropy": float(predicted_entropy_per_row.mean()),
    }


def validate_checkpoint(
    checkpoint: dict,
    plan: dict,
    model_config: dict,
    dataset_fingerprint: str,
    model_parameters: int,
    training_fingerprint: str,
    teacher_state_provenance: dict[str, object] | None = None,
    split: str | None = None,
) -> None:
    checkpoint_training_fingerprint = checkpoint.get(
        "trainingConfigFingerprint"
    )
    legacy_fingerprint_allowed = (
        checkpoint_training_fingerprint is None
        and plan["id"] in LEGACY_UNFINGERPRINTED_PLAN_IDS
    )
    if checkpoint.get("planId") != plan["id"] \
            or checkpoint.get("architectureContract") \
            != architecture_contract_for_model_config(model_config) \
            or checkpoint.get("dataContract") != DATA_CONTRACT \
            or checkpoint.get("datasetFingerprint") != dataset_fingerprint \
            or checkpoint.get("modelConfig") != model_config \
            or checkpoint.get("parameterCount") != model_parameters \
            or (
                not legacy_fingerprint_allowed
                and checkpoint_training_fingerprint != training_fingerprint
            ) \
            or (
                teacher_state_provenance is not None
                and (
                    not isinstance(
                        checkpoint.get("teacherStateProvenance"),
                        dict,
                    )
                    or checkpoint["teacherStateProvenance"].get(split)
                    != teacher_state_provenance
                )
            ):
        raise ValueError(
            "checkpoint model, data, or training configuration is incompatible"
        )
    if checkpoint.get("interrupted") is True:
        raise ValueError("partial-epoch interrupted checkpoints cannot be evaluated")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("checkpoint has no model state")


def validate_split_access(split: str, allow_test: bool) -> None:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if split == "test" and not allow_test:
        raise PermissionError(
            "test evaluation requires the explicit --allow-test flag"
        )


def resolve_device(requested: str, planned: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    selected = planned if requested == "auto" else requested
    if selected == "cuda" and not torch.cuda.is_available():
        if requested == "auto":
            selected = "cpu"
        else:
            raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    if selected not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported plan evaluation device: {selected}")
    if selected == "cuda":
        torch.set_float32_matmul_precision("high")
    return torch.device(selected)


def finite_json_value(value):
    """Replace non-finite scalar metrics with null for strict JSON reports."""
    if isinstance(value, dict):
        return {key: finite_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [finite_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(value: dict, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
