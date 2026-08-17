from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path

import torch

from next_return_dataset import EXAMPLE_SPAN_MS, ExampleShard
from normalized_glu_next_return import (
    CAUSAL_VOLATILITY_INPUT_NORMALIZATION,
    NormalizedGluNextReturn,
    TRAINING_POSITION_INPUT_NORMALIZATION,
)
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import direct_calendar_shards
from train_normalized_glu_next_return import (
    MetricAccumulator,
    NextReturnDataset,
    atomic_json,
    canonical_fingerprint,
    iter_device_batches,
    resolve,
)


CALIBRATION_CONTRACT = (
    "closed-form-affine-on-unseen-pre-validation-calendar-tail-v1"
)


@dataclass(frozen=True)
class AffineTransform:
    scale: float
    intercept: float

    def as_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "intercept": self.intercept}


class AffineStatistics:
    def __init__(self, device: torch.device) -> None:
        self.values = torch.zeros(6, dtype=torch.float64, device=device)

    def add(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        prediction = prediction.detach().to(dtype=torch.float64)
        target = target.detach().to(dtype=torch.float64)
        weights = weights.detach().to(dtype=torch.float64)
        self.values += torch.stack((
            weights.sum(),
            (weights * prediction).sum(),
            (weights * target).sum(),
            (weights * prediction.square()).sum(),
            (weights * target.square()).sum(),
            (weights * prediction * target).sum(),
        ))

    def fit(self) -> dict[str, AffineTransform]:
        weight, p_sum, y_sum, p_square, _y_square, product = (
            float(value) for value in self.values
        )
        if weight <= 0:
            raise RuntimeError("cannot fit calibration on an empty split")
        if p_square <= 0:
            raise RuntimeError("cannot calibrate a constant-zero prediction")
        p_mean = p_sum / weight
        y_mean = y_sum / weight
        p_variance = p_square / weight - p_mean * p_mean
        covariance = product / weight - p_mean * y_mean
        if p_variance <= 0:
            raise RuntimeError("cannot calibrate a constant prediction")
        scale_only = product / p_square
        affine_scale = covariance / p_variance
        return {
            "identity": AffineTransform(1.0, 0.0),
            "scaleOnly": AffineTransform(scale_only, 0.0),
            "affine": AffineTransform(
                affine_scale,
                y_mean - affine_scale * p_mean,
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a frozen affine output calibration on an unseen calendar tail "
            "and evaluate it on validation and test."
        )
    )
    parser.add_argument(
        "--training-plan", required=True, type=Path, nargs="+"
    )
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def unseen_calibration_shards(
    training_plan: dict,
    calendar: dict[str, list[ExampleShard]],
    *,
    calibration_days: int,
) -> tuple[list[ExampleShard], dict[str, object]]:
    if calibration_days < 1 or calibration_days >= len(calendar["train"]):
        raise ValueError("calibration days must select a proper training-calendar tail")
    subset = training_plan.get("subset", {})
    if subset.get("type", "fixed-contiguous") != "fixed-contiguous":
        raise ValueError(
            "pre-validation calibration currently requires a fixed-contiguous "
            "diagnostic subset so unseen provenance can be proven"
        )
    subset_start = datetime.combine(
        date.fromisoformat(str(subset["date"])),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    subset_examples = int(subset["examples"])
    subset_end_ms = int(subset_start.timestamp() * 1_000) \
        + (subset_examples - 1) * 1_000
    selected = calendar["train"][-calibration_days:]
    calibration_start_ms = selected[0].decision_time_start
    if subset_end_ms + EXAMPLE_SPAN_MS >= calibration_start_ms:
        raise ValueError(
            "calibration interval overlaps the model's training examples or "
            "their 120-second context"
        )
    validation_start_ms = calendar["validation"][0].decision_time_start
    if selected[-1].decision_time_end >= validation_start_ms:
        raise ValueError("calibration interval overlaps validation")
    return selected, {
        "trainingSubsetStart": str(subset["date"]),
        "trainingSubsetEnd": datetime.fromtimestamp(
            subset_end_ms / 1_000, tz=timezone.utc
        ).isoformat(),
        "calibrationStart": selected[0].date,
        "calibrationEnd": selected[-1].date,
        "calibrationDays": calibration_days,
        "calibrationExamples": sum(value.count for value in selected),
        "validationStart": calendar["validation"][0].date,
        "validationEnd": calendar["validation"][-1].date,
        "testStart": calendar["test"][0].date,
        "testEnd": calendar["test"][-1].date,
        "historyTargetEmbargoSeconds": EXAMPLE_SPAN_MS // 1_000,
        "testPolicy": "untouched until transforms were frozen on calibration",
    }


def load_model(
    training_plan: dict,
    checkpoint: dict,
    device: torch.device,
) -> NormalizedGluNextReturn:
    state = checkpoint["model"]
    architecture = training_plan["architecture"]
    model = NormalizedGluNextReturn(
        state["feature_mean"],
        state["feature_std"],
        state["target_mean"],
        state["target_std"],
        widths=tuple(int(value) for value in architecture["widths"]),
        input_normalization=architecture.get(
            "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
        ),
        volatility_window=architecture.get("volatilityWindow"),
        dropout=0,
        dropout_rate=0,
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture.get("learnableCentering", True)),
    )
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


@torch.no_grad()
def fit_transforms(
    model: NormalizedGluNextReturn,
    dataset: NextReturnDataset,
    *,
    batch_size: int,
    device: torch.device,
    causal_volatility_window: int | None,
) -> tuple[dict[str, AffineTransform], dict[str, float | int | None]]:
    statistics = AffineStatistics(device)
    raw_metrics = MetricAccumulator(float(model.target_std.item()), device)
    for batch in iter_device_batches(
        dataset.iter_batches(
            "calibration", batch_size, shuffle=False, seed=0,
            reuse_buffers=True,
            causal_volatility_window=causal_volatility_window,
        ),
        device,
    ):
        features, targets, weights = batch[:3]
        volatility = batch[3] if len(batch) == 4 else None
        prediction = model(features, causal_volatility_rms=volatility)
        statistics.add(prediction, targets, weights)
        raw_metrics.add(prediction, targets, weights)
    return statistics.fit(), raw_metrics.result()


@torch.no_grad()
def evaluate_transforms(
    model: NormalizedGluNextReturn,
    dataset: NextReturnDataset,
    split: str,
    transforms: dict[str, AffineTransform],
    *,
    batch_size: int,
    device: torch.device,
    causal_volatility_window: int | None,
) -> dict[str, dict[str, float | int | None]]:
    metrics = {
        name: MetricAccumulator(float(model.target_std.item()), device)
        for name in transforms
    }
    for batch in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True,
            causal_volatility_window=causal_volatility_window,
        ),
        device,
    ):
        features, targets, weights = batch[:3]
        volatility = batch[3] if len(batch) == 4 else None
        prediction = model(features, causal_volatility_rms=volatility)
        for name, transform in transforms.items():
            metrics[name].add(
                prediction * transform.scale + transform.intercept,
                targets,
                weights,
            )
    return {name: value.result() for name, value in metrics.items()}


def calibrate_plan(
    repo: Path,
    plan_file: Path,
    split_plan: dict,
    split_plan_file: Path,
    *,
    calibration_days: int,
    batch_size: int,
    device: torch.device,
    checkpoint_file: Path | None = None,
    selected_candidate: str | None = None,
) -> dict[str, object]:
    plan_value = json.loads(plan_file.read_text(encoding="utf-8"))
    plan = plan_value.get("plan", plan_value)
    run_root = resolve(repo, Path(plan["runDir"]))
    checkpoint_file = checkpoint_file or run_root / "checkpoints/best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file, map_location="cpu", weights_only=False
    )
    plan_hash = canonical_fingerprint(plan)
    if checkpoint.get("planSha256") != plan_hash:
        raise ValueError("checkpoint does not belong to the training plan")
    candidate_states = {
        "raw-best": checkpoint["model"],
        **checkpoint.get("swaCandidates", {}),
    }
    if selected_candidate is None and len(candidate_states) > 1:
        validation_file = run_root / "state/validation-current-best.json"
        if not validation_file.is_file():
            raise ValueError(
                "validation must select an SWA candidate before calibration"
            )
        selected_candidate = json.loads(
            validation_file.read_text(encoding="utf-8")
        ).get("selectedCandidate")
    selected_candidate = selected_candidate or "raw-best"
    if selected_candidate not in candidate_states:
        raise ValueError("calibration-selected checkpoint candidate is unavailable")
    selected_checkpoint = {
        **checkpoint,
        "model": candidate_states[selected_candidate],
    }
    history_root = resolve(repo, Path(plan["historyDir"]))
    calendar = direct_calendar_shards(
        split_plan["split"], history_root,
        horizon_seconds=1, decision_stride_seconds=1,
    )
    calibration, provenance = unseen_calibration_shards(
        plan, calendar, calibration_days=calibration_days
    )
    dataset = NextReturnDataset({
        "calibration": calibration,
        "validation": calendar["validation"],
        "test": calendar["test"],
    }, history_root, horizon_return_count=1, row_stride=1,
        exclude_zero_targets=plan.get("datasetFilter") is not None)
    model = load_model(plan, selected_checkpoint, device)
    architecture = plan["architecture"]
    causal_volatility_window = (
        int(architecture["volatilityWindow"])
        if architecture.get("inputNormalization")
        == CAUSAL_VOLATILITY_INPUT_NORMALIZATION
        else None
    )
    transforms, raw_calibration = fit_transforms(
        model, dataset, batch_size=batch_size, device=device,
        causal_volatility_window=causal_volatility_window,
    )
    validation = evaluate_transforms(
        model, dataset, "validation", transforms,
        batch_size=batch_size, device=device,
        causal_volatility_window=causal_volatility_window,
    )
    test = evaluate_transforms(
        model, dataset, "test", transforms,
        batch_size=batch_size, device=device,
        causal_volatility_window=causal_volatility_window,
    )
    result: dict[str, object] = {
        "contract": CALIBRATION_CONTRACT,
        "trainingPlanId": plan["id"],
        "trainingPlanSha256": plan_hash,
        "checkpoint": str(checkpoint_file.relative_to(repo)),
        "checkpointEpoch": int(
            checkpoint.get("rawBestEpoch", checkpoint["epoch"])
            if selected_candidate == "raw-best" else checkpoint["epoch"]
        ),
        "selectedCandidate": selected_candidate,
        "splitPlanId": split_plan["id"],
        "splitPlanSha256": canonical_fingerprint(split_plan),
        "splitPlan": str(split_plan_file.relative_to(repo)),
        "provenance": provenance,
        "datasetFilter": plan.get("datasetFilter"),
        "fitObjective": "weighted ordinary least squares",
        "transforms": {
            name: transform.as_dict() for name, transform in transforms.items()
        },
        "calibrationRaw": raw_calibration,
        "validation": validation,
        "test": test,
    }
    output_file = run_root / "state/output-calibration-pre-validation-7d.json"
    atomic_json(result, output_file)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    split_plan_file = resolve(repo, args.split_plan)
    split_plan = json.loads(split_plan_file.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration was requested but is unavailable")
    torch.set_float32_matmul_precision("high")
    results = []
    for value in args.training_plan:
        plan_file = resolve(repo, value)
        result = calibrate_plan(
            repo, plan_file, split_plan, split_plan_file,
            calibration_days=int(args.calibration_days),
            batch_size=int(args.batch_size), device=device,
        )
        results.append(result)
        print(json.dumps(result, separators=(",", ":")), flush=True)
    print(json.dumps({
        "contract": CALIBRATION_CONTRACT,
        "completedPlans": [value["trainingPlanId"] for value in results],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
