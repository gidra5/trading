from __future__ import annotations

from dataclasses import dataclass
import json
import math

import torch
from torch import Tensor
import torch.nn.functional as functional


FORECAST_MINUTES = 60
MULTISCALE_COMPONENTS = (
    "hour",
    "quarterHourContrast",
    "fiveMinuteContrast",
    "minuteResidual",
)
OBJECTIVE_COMPONENTS = (
    "return",
    "path",
    *MULTISCALE_COMPONENTS,
)


def multiscale_decompose(returns: Tensor) -> dict[str, Tensor]:
    """Lossless 60→15→5→1 minute additive decomposition."""
    if returns.shape[-1] != FORECAST_MINUTES:
        raise ValueError("multiscale decomposition expects 60 returns")
    five_minute_sum = returns.reshape(*returns.shape[:-1], 12, 5).sum(dim=-1)
    quarter_hour_sum = five_minute_sum.reshape(
        *returns.shape[:-1],
        4,
        3,
    ).sum(dim=-1)
    hour_sum = quarter_hour_sum.sum(dim=-1, keepdim=True)
    quarter_contrast = quarter_hour_sum - hour_sum / 4
    five_contrast = five_minute_sum - quarter_hour_sum.repeat_interleave(3, dim=-1) / 3
    minute_residual = returns - five_minute_sum.repeat_interleave(5, dim=-1) / 5
    return {
        "hour": hour_sum,
        "quarterHourContrast": quarter_contrast,
        "fiveMinuteContrast": five_contrast,
        "minuteResidual": minute_residual,
    }


def multiscale_reconstruct(components: dict[str, Tensor]) -> Tensor:
    missing = tuple(name for name in MULTISCALE_COMPONENTS if name not in components)
    if missing:
        raise ValueError(f"multiscale reconstruction is missing: {missing}")
    hour = components["hour"]
    quarter = components["quarterHourContrast"]
    five = components["fiveMinuteContrast"]
    residual = components["minuteResidual"]
    if hour.shape[-1] != 1 \
            or quarter.shape[-1] != 4 \
            or five.shape[-1] != 12 \
            or residual.shape[-1] != FORECAST_MINUTES \
            or not (
                hour.shape[:-1]
                == quarter.shape[:-1]
                == five.shape[:-1]
                == residual.shape[:-1]
            ):
        raise ValueError("multiscale component shapes are inconsistent")
    quarter_sum = hour / 4 + quarter
    five_sum = quarter_sum.repeat_interleave(3, dim=-1) / 3 + five
    return five_sum.repeat_interleave(5, dim=-1) / 5 + residual


def target_structure(target_returns: Tensor) -> dict[str, Tensor]:
    if target_returns.ndim != 2 or target_returns.shape[-1] != FORECAST_MINUTES:
        raise ValueError("forecast structure expects [example, 60] targets")
    return {
        "path": target_returns.cumsum(dim=-1),
        **multiscale_decompose(target_returns),
    }


@dataclass(frozen=True)
class ForecastObjectiveStatistics:
    path_std: Tensor
    hour_std: Tensor
    quarter_hour_contrast_std: Tensor
    five_minute_contrast_std: Tensor
    minute_residual_std: Tensor

    def validate(self) -> None:
        expected = {
            "path_std": (60,),
            "hour_std": (1,),
            "quarter_hour_contrast_std": (4,),
            "five_minute_contrast_std": (12,),
            "minute_residual_std": (60,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape \
                    or not bool(torch.isfinite(value).all()) \
                    or bool((value <= 0).any()):
                raise ValueError(f"objective statistic {name} must have shape {shape}")

    def scales(self) -> dict[str, Tensor]:
        return {
            "path": self.path_std,
            "hour": self.hour_std,
            "quarterHourContrast": self.quarter_hour_contrast_std,
            "fiveMinuteContrast": self.five_minute_contrast_std,
            "minuteResidual": self.minute_residual_std,
        }

    def to(self, device: torch.device) -> ForecastObjectiveStatistics:
        return ForecastObjectiveStatistics(*(
            value.to(device=device, dtype=torch.float32)
            for value in (
                self.path_std,
                self.hour_std,
                self.quarter_hour_contrast_std,
                self.five_minute_contrast_std,
                self.minute_residual_std,
            )
        ))

    def as_json(self) -> dict[str, list[float]]:
        return {
            "pathStd": self.path_std.tolist(),
            "hourStd": self.hour_std.tolist(),
            "quarterHourContrastStd": self.quarter_hour_contrast_std.tolist(),
            "fiveMinuteContrastStd": self.five_minute_contrast_std.tolist(),
            "minuteResidualStd": self.minute_residual_std.tolist(),
        }


def validate_forecast_objective(config: dict) -> None:
    objective_type = config.get("type")
    if objective_type == "curriculum":
        allowed = {"type", "stages", "validationObjective"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown curriculum settings: {sorted(unknown)}")
        stages = config.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("forecast curriculum must contain stages")
        previous = 0
        for stage in stages:
            if set(stage) != {"throughEpoch", "objective"}:
                raise ValueError("forecast curriculum stage contract is invalid")
            through = int(stage["throughEpoch"])
            if through <= previous:
                raise ValueError("forecast curriculum epochs must increase")
            previous = through
            validate_forecast_objective(stage["objective"])
            if stage["objective"].get("type") == "curriculum":
                raise ValueError("nested forecast curricula are unsupported")
        validate_forecast_objective(config["validationObjective"])
        if config["validationObjective"].get("type") == "curriculum":
            raise ValueError("validation objective cannot be a curriculum")
        return
    if objective_type not in {"cumulative_path", "multiscale_reconstruction"}:
        raise ValueError(f"unsupported forecast objective: {objective_type}")
    allowed = {"type", "loss", "huberDelta", "weights"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown forecast objective settings: {sorted(unknown)}")
    if config.get("loss") not in {"huber", "mse"} \
            or float(config.get("huberDelta", 0)) <= 0:
        raise ValueError("forecast objective loss settings are invalid")
    weights = config.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("forecast objective requires explicit component weights")
    unknown_weights = set(weights) - set(OBJECTIVE_COMPONENTS)
    if unknown_weights:
        raise ValueError(f"unknown forecast objective weights: {sorted(unknown_weights)}")
    if any(float(weight) < 0 for weight in weights.values()) \
            or sum(float(weight) for weight in weights.values()) <= 0:
        raise ValueError("forecast objective weights must be non-negative and non-zero")
    if objective_type == "cumulative_path" \
            and float(weights.get("path", 0)) <= 0:
        raise ValueError("cumulative-path objective requires a path weight")
    if objective_type == "multiscale_reconstruction" \
            and not any(float(weights.get(name, 0)) > 0 for name in MULTISCALE_COMPONENTS):
        raise ValueError("multiscale objective must weight a reconstructable component")


def objective_for_epoch(
    config: dict,
    epoch: int,
    *,
    validation: bool,
) -> tuple[dict, str]:
    validate_forecast_objective(config)
    if config["type"] != "curriculum":
        return config, config["type"]
    if validation:
        return config["validationObjective"], "validationObjective"
    for index, stage in enumerate(config["stages"], start=1):
        if epoch <= int(stage["throughEpoch"]):
            return stage["objective"], f"stage{index}"
    return config["stages"][-1]["objective"], f"stage{len(config['stages'])}"


def _weighted_mean(
    elements: Tensor,
    example_weights: Tensor,
    weight_mass: Tensor,
) -> Tensor:
    if elements.ndim < 2 or elements.shape[0] != example_weights.shape[0]:
        raise ValueError("objective elements and example weights are misaligned")
    weights = example_weights.to(dtype=elements.dtype).reshape(
        elements.shape[0],
        *((1,) * (elements.ndim - 1)),
    )
    per_example_elements = elements[0].numel()
    denominator = weight_mass * per_example_elements
    return (elements * weights).sum() / denominator


def _element_loss(error: Tensor, loss: str, huber_delta: float) -> Tensor:
    if loss == "mse":
        return error.square()
    return functional.huber_loss(
        error,
        torch.zeros_like(error),
        reduction="none",
        delta=huber_delta,
    )


def normalized_component_errors(
    prediction: Tensor,
    target: Tensor,
    target_std: Tensor,
    statistics: ForecastObjectiveStatistics,
) -> dict[str, Tensor]:
    if prediction.shape != target.shape \
            or prediction.ndim != 2 \
            or prediction.shape[-1] != FORECAST_MINUTES:
        raise ValueError("forecast objective expects matching [example, 60] tensors")
    if target_std.shape != (FORECAST_MINUTES,):
        raise ValueError("return target scales must have shape [60]")
    error = prediction - target
    multiscale = multiscale_decompose(error)
    scales = statistics.scales()
    return {
        "return": error / target_std,
        "path": error.cumsum(dim=-1) / scales["path"],
        **{
            name: multiscale[name] / scales[name]
            for name in MULTISCALE_COMPONENTS
        },
    }


def forecast_objective_loss(
    prediction: Tensor,
    target: Tensor,
    example_weights: Tensor,
    target_std: Tensor,
    statistics: ForecastObjectiveStatistics,
    config: dict,
    *,
    epoch: int,
    validation: bool,
) -> tuple[Tensor, dict[str, Tensor], str]:
    active, stage = objective_for_epoch(config, epoch, validation=validation)
    errors = normalized_component_errors(
        prediction,
        target,
        target_std,
        statistics,
    )
    huber_delta = float(active["huberDelta"])
    weights = {
        name: float(weight)
        for name, weight in active["weights"].items()
        if float(weight) > 0
    }
    weight_mass = example_weights.to(dtype=prediction.dtype).sum()
    metrics: dict[str, Tensor] = {}
    for name, error in errors.items():
        mse_error = (
            error
            if active["loss"] == "mse" and name in weights
            else error.detach()
        )
        huber_error = (
            error
            if active["loss"] == "huber" and name in weights
            else error.detach()
        )
        metrics[f"normalized{name[0].upper() + name[1:]}Mse"] = _weighted_mean(
            mse_error.square(),
            example_weights,
            weight_mass,
        )
        metrics[f"normalized{name[0].upper() + name[1:]}Huber"] = _weighted_mean(
            _element_loss(huber_error, "huber", huber_delta),
            example_weights,
            weight_mass,
        )
    suffix = "Huber" if active["loss"] == "huber" else "Mse"
    total_weight = sum(weights.values())
    loss = sum(
        metrics[f"normalized{name[0].upper() + name[1:]}{suffix}"] * weight
        for name, weight in weights.items()
    ) / total_weight
    metrics["forecastObjectiveLoss"] = loss
    return loss, metrics, stage


def forecast_objective_value_from_metrics(
    metrics: dict[str, float],
    config: dict,
) -> float:
    """Compose a fixed validation objective from aggregate component metrics."""
    active, _ = objective_for_epoch(config, 1, validation=True)
    suffix = "Huber" if active["loss"] == "huber" else "Mse"
    weights = {
        name: float(weight)
        for name, weight in active["weights"].items()
        if float(weight) > 0
    }
    required = {
        f"normalized{name[0].upper() + name[1:]}{suffix}"
        for name in weights
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(
            f"forecast objective metrics are missing: {sorted(missing)}"
        )
    value = sum(
        float(metrics[
            f"normalized{name[0].upper() + name[1:]}{suffix}"
        ]) * weight
        for name, weight in weights.items()
    ) / sum(weights.values())
    if not math.isfinite(value):
        raise FloatingPointError("composed forecast objective is non-finite")
    return value


def objective_contract(config: dict) -> str:
    validate_forecast_objective(config)
    return "variance-normalized-future-structure-v1:" + json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )
