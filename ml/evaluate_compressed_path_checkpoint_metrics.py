from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from calibrate_next_return_output import AffineStatistics, AffineTransform
from compressed_path_return_density import (
    CompressedPathOutput,
    CompressedPathReturnDensity,
    path_log_density_terms,
)
from recurrent_market_path_density import (
    ARCHITECTURE_CONTRACT as RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
    RESIDUAL_ARCHITECTURE_CONTRACT as RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
    RecurrentMarketPathDensity,
    ResidualRecurrentMarketPathDensity,
)
from low_rank_path_matrix_density import (
    ARCHITECTURE_CONTRACT as LOW_RANK_PATH_MATRIX_ARCHITECTURE_CONTRACT,
    DynamicLowRankPathMatrixDensity,
)
from return_knot_density import KnotDensityContract
from return_knot_density import inverse_unit_to_returns, triangular_basis_areas
from trading_storage import load_torch_checkpoint
from train_feature_compressed_path_density import (
    ImmediateFeatureActivePathDataset,
    PathMetrics,
    LOW_RANK_PATH_MATRIX_RUNNER_CONTRACT,
    RECURRENT_MARKET_RUNNER_CONTRACT,
    RUNNER_CONTRACT,
    evaluate,
    training_statistics,
)
from train_next_return_knot_density import canonical_hash
from train_normalized_glu_next_return import (
    MetricAccumulator,
    Reporter,
    atomic_json,
)


CONTRACT = "compressed-path-checkpoint-evaluation-and-calibration-v1"
CALIBRATION_CONTRACT = "compressed-path-clean-pre-validation-calibration-v1"
POLICIES = ("validation-nll", "validation-mse", "validation-correlation")
TEMPERATURE_GRID = (0.16, 0.25, 0.4, 0.63, 0.8, 1.0, 1.25, 1.6, 2.5, 4.0, 6.3)
DEFAULT_CALIBRATION_DIR = (
    "data/training/datasets/"
    "next-return-production-basis-calibration-tail-16k-v1"
)
ROLLING_ACTIVE_RETURNS = 16_000
POLYNOMIAL_RIDGE = 1e-6
MATRIX_AFFINE_RIDGE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate preserved compressed-path checkpoints."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--checkpoint-policies", nargs="+", choices=POLICIES, default=POLICIES
    )
    parser.add_argument(
        "--calibration-windows", nargs="+", type=int,
        default=(ROLLING_ACTIVE_RETURNS,),
        help="Trailing clean calibration examples used by each calibration fit.",
    )
    parser.add_argument(
        "--matrix-affine-only", action="store_true",
        help="Compute only joint matrix-affine point variants for fast backfills.",
    )
    return parser.parse_args()


def resolve(repo: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def model_state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Identify selection checkpoints that carry identical model weights."""
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class CalibrationPathDataset:
    """Immediate feature rows and chronological active-return calibration paths."""

    def __init__(
        self, root: Path, return_count: int, feature_history: int = 1
    ) -> None:
        self.root = root
        self.return_count = int(return_count)
        self.manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        count = int(self.manifest["examples"])
        feature_count = int(self.manifest["featureCount"])
        matrix = np.memmap(
            root / "calibration.features.f32", dtype="<f4", mode="r",
            shape=(count, feature_count),
        )
        channels_value = self.manifest.get("temporalChannelCount")
        history_value = self.manifest.get("featureHistorySeconds", 1)
        requested_history = int(feature_history)
        if channels_value is None:
            if requested_history != 1:
                raise ValueError("flat calibration data has no temporal history")
            self.feature_count = feature_count
            self.features = matrix
        else:
            channels = int(channels_value)
            history = int(history_value)
            if feature_count != channels * history:
                raise ValueError("calibration temporal dimensions are inconsistent")
            if requested_history > history:
                raise ValueError("calibration history is shorter than requested")
            if self.manifest.get("temporalLayout") == "time-major":
                self.features = np.ascontiguousarray(
                    matrix.reshape(count, history, channels)[:, -requested_history:, :]
                    .reshape(count, channels * requested_history)
                )
            else:
                self.features = np.ascontiguousarray(
                    matrix.reshape(count, channels, history)[:, :, -requested_history:]
                    .transpose(0, 2, 1)
                    .reshape(count, channels * requested_history)
                )
            self.feature_count = channels * requested_history
        targets = np.asarray(np.memmap(
            root / "calibration.targets.f32", dtype="<f4", mode="r",
            shape=(count,),
        ), dtype=np.float32)
        times = np.asarray(np.memmap(
            root / "calibration.times.f64", dtype="<f8", mode="r",
            shape=(count,),
        ), dtype=np.float64)
        if np.any(targets == 0) or np.any(np.diff(times) <= 0):
            raise ValueError("calibration examples are not clean and chronological")
        if count < self.return_count:
            raise ValueError("calibration split is shorter than one path")
        self.targets = np.lib.stride_tricks.sliding_window_view(
            targets, self.return_count
        )
        self.times = times[: self.targets.shape[0]]

    def logical_count(self, split: str) -> int:
        if split != "calibration":
            raise KeyError(split)
        return int(self.targets.shape[0])

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        del shuffle, seed
        if split != "calibration":
            raise KeyError(split)
        count = self.logical_count(split)
        if limit is not None:
            count = min(count, int(limit))
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            yield (
                torch.from_numpy(np.asarray(
                    self.features[start:stop], dtype=np.float32
                ).copy()),
                torch.from_numpy(np.asarray(
                    self.targets[start:stop], dtype=np.float32
                ).copy()),
                torch.ones(stop - start, dtype=torch.float32),
            )


def load_model(
    plan: dict, densities: tuple[KnotDensityContract, ...], checkpoint: dict,
    device: torch.device,
) -> CompressedPathReturnDensity | RecurrentMarketPathDensity \
        | DynamicLowRankPathMatrixDensity:
    if checkpoint.get("planSha256") != canonical_hash(plan):
        raise ValueError("checkpoint belongs to a different training plan")
    architecture_contract = plan["architecture"].get("contract")
    recurrent_market = architecture_contract in {
        RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
        RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
    }
    residual_recurrent_market = (
        architecture_contract
        == RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT
    )
    low_rank_path_matrix = (
        architecture_contract == LOW_RANK_PATH_MATRIX_ARCHITECTURE_CONTRACT
    )
    expected_runner = (
        LOW_RANK_PATH_MATRIX_RUNNER_CONTRACT
        if low_rank_path_matrix else (
            RECURRENT_MARKET_RUNNER_CONTRACT if recurrent_market else RUNNER_CONTRACT
        )
    )
    if checkpoint.get("runnerContract") != expected_runner:
        raise ValueError("checkpoint runner contract changed")
    state = checkpoint["model"]
    architecture = plan["architecture"]
    if low_rank_path_matrix:
        model = DynamicLowRankPathMatrixDensity(
            state["feature_mean"], state["feature_std"], densities[0],
            market_width=int(architecture["marketWidth"]),
            path_embedding_width=int(architecture["pathEmbeddingWidth"]),
            path_count=int(architecture["pathCount"]),
            return_count=int(architecture["returnCount"]),
            matrix_rank=int(architecture["matrixRank"]),
            hidden_width_cap=int(architecture["hiddenWidthCap"]),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
            learnable_centering=bool(architecture["learnableCentering"]),
        )
    else:
        model = (
            (
            ResidualRecurrentMarketPathDensity
            if residual_recurrent_market else RecurrentMarketPathDensity
            )(
                state["feature_mean"], state["feature_std"], densities[0],
                market_width=int(architecture["marketWidth"]),
                state_widths=tuple(
                    int(value) for value in architecture["stateWidths"]
                ),
                transition_rank=int(architecture.get("transitionRank", 1)),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            )
            if recurrent_market else
            CompressedPathReturnDensity(
                state["feature_mean"], state["feature_std"], densities,
                market_width=int(architecture["marketWidth"]),
                state_widths=tuple(
                    int(value) for value in architecture["stateWidths"]
                ),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            )
        )
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def iter_device_batches(dataset, split: str, batch_size: int, device: torch.device):
    for features, targets, weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0
    ):
        yield (
            features.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
            weights.to(device, non_blocking=True),
        )


@torch.no_grad()
def fit_point_transforms(
    model: CompressedPathReturnDensity,
    dataset: CalibrationPathDataset,
    *, batch_size: int, device: torch.device,
) -> tuple[dict[str, AffineTransform], dict]:
    statistics = AffineStatistics(device)
    raw = MetricAccumulator(1.0, device)
    for features, targets, weights in iter_device_batches(
        dataset, "calibration", batch_size, device
    ):
        predictions = model(features).expectations
        expanded = weights[:, None].expand_as(targets)
        statistics.add(
            predictions.flatten(), targets.flatten(), expanded.flatten()
        )
        raw.add(predictions.flatten(), targets.flatten(), expanded.flatten())
    return statistics.fit(), raw.result()


def temperature_scaled_output(
    output: CompressedPathOutput,
    model: CompressedPathReturnDensity,
    temperature: float,
) -> CompressedPathOutput:
    scaled = tuple(
        torch.log_softmax(log_mass / temperature, dim=1)
        for log_mass in output.log_masses
    )
    expectations = torch.stack([
        (
            log_mass.exp() @ model.means(step)
            if output.component_means is None else
            (log_mass.exp() * output.component_means[step]).sum(dim=1)
        )
        for step, log_mass in enumerate(scaled)
    ], dim=1)
    return CompressedPathOutput(
        scaled,
        expectations,
        knots_unit=output.knots_unit,
        areas_unit=output.areas_unit,
        component_means=output.component_means,
        arithmetic_component_means=output.arithmetic_component_means,
    )


def component_arithmetic_return_means(
    density: KnotDensityContract, *, quadrature_order: int = 64
) -> np.ndarray:
    """Integrate exp(r)-1 against each normalized triangular basis."""
    knots = np.asarray(density.knots_unit, dtype=np.float64)
    areas = triangular_basis_areas(torch.from_numpy(knots)).numpy()
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    result = np.zeros(knots.size, dtype=np.float64)
    for interval in range(knots.size - 1):
        left = float(knots[interval])
        right = float(knots[interval + 1])
        midpoint = (left + right) / 2
        half_width = (right - left) / 2
        unit = midpoint + half_width * nodes
        arithmetic = np.expm1(inverse_unit_to_returns(unit, density.transform))
        fraction = (unit - left) / (right - left)
        scaled_weights = weights * half_width
        result[interval] += float(np.sum(
            scaled_weights * arithmetic * (1 - fraction)
        ))
        result[interval + 1] += float(np.sum(
            scaled_weights * arithmetic * fraction
        ))
    return result / areas


@torch.no_grad()
def collect_point_arrays(
    model: CompressedPathReturnDensity,
    dataset,
    split: str,
    arithmetic_component_means: tuple[torch.Tensor, ...],
    *, batch_size: int, device: torch.device,
) -> dict[str, np.ndarray]:
    log_predictions: list[np.ndarray] = []
    arithmetic_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for features, target, _weights in iter_device_batches(
        dataset, split, batch_size, device
    ):
        output = model(features)
        arithmetic = torch.stack([
            (
                log_mass.exp() @ arithmetic_component_means[step]
                if output.arithmetic_component_means is None else
                (
                    log_mass.exp()
                    * output.arithmetic_component_means[step]
                ).sum(dim=1)
            )
            for step, log_mass in enumerate(output.log_masses)
        ], dim=1)
        log_predictions.append(output.expectations.cpu().numpy())
        arithmetic_predictions.append(arithmetic.cpu().numpy())
        targets.append(target.cpu().numpy())
    log_targets = np.concatenate(targets).astype(np.float64, copy=False)
    return {
        "logPrediction": np.concatenate(log_predictions).astype(
            np.float64, copy=False
        ),
        "arithmeticPrediction": np.concatenate(arithmetic_predictions).astype(
            np.float64, copy=False
        ),
        "logTarget": log_targets,
        "arithmeticTarget": np.expm1(log_targets),
    }


def polynomial_design(values: np.ndarray, degree: int, scale: float) -> np.ndarray:
    normalized = values / scale
    return np.stack(
        [np.ones_like(normalized)]
        + [np.power(normalized, power) for power in range(1, degree + 1)],
        axis=-1,
    )


def fit_polynomial(
    prediction: np.ndarray,
    target: np.ndarray,
    *, degree: int, input_scale: float, output_scale: float, ridge: float,
) -> np.ndarray:
    design = polynomial_design(prediction.reshape(-1), degree, input_scale)
    normalized_target = target.reshape(-1) / output_scale
    gram = design.T @ design / design.shape[0]
    right = design.T @ normalized_target / design.shape[0]
    penalty = np.eye(degree + 1, dtype=np.float64) * ridge
    penalty[0, 0] = 0
    return np.linalg.solve(gram + penalty, right)


def apply_polynomial(
    prediction: np.ndarray,
    coefficients: np.ndarray,
    *, input_scale: float, output_scale: float,
) -> np.ndarray:
    design = polynomial_design(
        prediction, coefficients.size - 1, input_scale
    )
    return output_scale * np.einsum("...d,d->...", design, coefficients)


def metric_result(
    prediction_log: np.ndarray, target_log: np.ndarray, target_std: float
) -> dict:
    metric = MetricAccumulator(target_std, torch.device("cpu"))
    prediction = torch.from_numpy(np.asarray(
        prediction_log, dtype=np.float64
    ).reshape(-1))
    target = torch.from_numpy(np.asarray(
        target_log, dtype=np.float64
    ).reshape(-1))
    metric.add(prediction, target, torch.ones_like(target))
    return metric.result()


def per_lead_metric_results(
    prediction_log: np.ndarray, target_log: np.ndarray, target_std: float
) -> list[dict]:
    if prediction_log.ndim != 2 or target_log.shape != prediction_log.shape:
        raise ValueError("per-lead metrics require matching [examples, leads] arrays")
    return [
        metric_result(prediction_log[:, lead], target_log[:, lead], target_std)
        for lead in range(prediction_log.shape[1])
    ]


def event_sufficient_statistics(
    prediction: np.ndarray,
    target: np.ndarray,
    *, degree: int, input_scale: float, output_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate all already-issued lead forecasts by realized target event."""
    events, leads = prediction.shape
    size = degree + 1
    gram = np.zeros((events, size, size), dtype=np.float64)
    right = np.zeros((events, size), dtype=np.float64)
    for lead in range(leads):
        origins = events - lead
        if origins <= 0:
            break
        design = polynomial_design(
            prediction[:origins, lead], degree, input_scale
        )
        normalized_target = target[lead:lead + origins] / output_scale
        gram[lead:lead + origins] += np.einsum(
            "ni,nj->nij", design, design
        )
        right[lead:lead + origins] += design * normalized_target[:, None]
    return gram, right


def rolling_polynomial_predictions(
    history_prediction: np.ndarray,
    history_target: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *, degree: int, input_scale: float, output_scale: float,
    ridge: float, window: int,
) -> np.ndarray:
    history_gram, history_right = event_sufficient_statistics(
        history_prediction, history_target, degree=degree,
        input_scale=input_scale, output_scale=output_scale,
    )
    current_gram, current_right = event_sufficient_statistics(
        prediction, target, degree=degree,
        input_scale=input_scale, output_scale=output_scale,
    )
    gram = np.concatenate((history_gram, current_gram), axis=0)
    right = np.concatenate((history_right, current_right), axis=0)
    cumulative_gram = np.concatenate((
        np.zeros((1, degree + 1, degree + 1), dtype=np.float64),
        np.cumsum(gram, axis=0),
    ))
    cumulative_right = np.concatenate((
        np.zeros((1, degree + 1), dtype=np.float64),
        np.cumsum(right, axis=0),
    ))
    history_count = history_prediction.shape[0]
    ends = history_count + np.arange(prediction.shape[0])
    starts = np.maximum(0, ends - window)
    rolling_gram = cumulative_gram[ends] - cumulative_gram[starts]
    rolling_right = cumulative_right[ends] - cumulative_right[starts]
    penalty = np.eye(degree + 1, dtype=np.float64) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(
        rolling_gram + penalty[None, :, :], rolling_right[..., None]
    )[..., 0]
    design = polynomial_design(prediction, degree, input_scale)
    return output_scale * np.einsum("nhd,nd->nh", design, coefficients)


def fit_per_step_polynomials(
    prediction: np.ndarray,
    target: np.ndarray,
    *, degree: int, ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_scales = np.maximum(prediction.std(axis=0), 1e-12)
    output_scales = np.maximum(target.std(axis=0), 1e-12)
    coefficients = np.stack([
        fit_polynomial(
            prediction[:, lead], target[:, lead], degree=degree,
            input_scale=float(input_scales[lead]),
            output_scale=float(output_scales[lead]), ridge=ridge,
        )
        for lead in range(prediction.shape[1])
    ])
    return coefficients, input_scales, output_scales


def apply_per_step_polynomials(
    prediction: np.ndarray,
    coefficients: np.ndarray,
    input_scales: np.ndarray,
    output_scales: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(prediction, dtype=np.float64)
    for lead in range(prediction.shape[1]):
        output[:, lead] = apply_polynomial(
            prediction[:, lead], coefficients[lead],
            input_scale=float(input_scales[lead]),
            output_scale=float(output_scales[lead]),
        )
    return output


def fit_affine_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    *, ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a jointly calibrated output = prediction @ A.T + bias map."""
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("matrix affine calibration requires matching matrices")
    input_scales = np.maximum(prediction.std(axis=0), 1e-12)
    output_scales = np.maximum(target.std(axis=0), 1e-12)
    design = np.concatenate((
        np.ones((prediction.shape[0], 1), dtype=np.float64),
        prediction / input_scales,
    ), axis=1)
    normalized_target = target / output_scales
    gram = design.T @ design / design.shape[0]
    right = design.T @ normalized_target / design.shape[0]
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(gram + penalty, right)
    return coefficients, input_scales, output_scales


def apply_affine_matrix(
    prediction: np.ndarray,
    coefficients: np.ndarray,
    input_scales: np.ndarray,
    output_scales: np.ndarray,
) -> np.ndarray:
    design = np.concatenate((
        np.ones((prediction.shape[0], 1), dtype=np.float64),
        prediction / input_scales,
    ), axis=1)
    return (design @ coefficients) * output_scales


def _cumulative_matrix_affine_statistics(
    prediction: np.ndarray,
    target: np.ndarray,
    input_scales: np.ndarray,
    output_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate((
        np.ones((prediction.shape[0], 1), dtype=np.float64),
        prediction / input_scales,
    ), axis=1)
    normalized_target = target / output_scales
    dimension = design.shape[1]
    outputs = target.shape[1]
    cumulative_gram = np.empty(
        (prediction.shape[0] + 1, dimension, dimension), dtype=np.float64
    )
    cumulative_right = np.empty(
        (prediction.shape[0] + 1, dimension, outputs), dtype=np.float64
    )
    cumulative_gram[0] = 0
    cumulative_right[0] = 0
    chunk_size = 4096
    for start in range(0, prediction.shape[0], chunk_size):
        end = min(start + chunk_size, prediction.shape[0])
        gram_rows = np.einsum(
            "ni,nj->nij", design[start:end], design[start:end]
        )
        right_rows = np.einsum(
            "ni,nj->nij", design[start:end], normalized_target[start:end]
        )
        cumulative_gram[start + 1:end + 1] = (
            np.cumsum(gram_rows, axis=0) + cumulative_gram[start]
        )
        cumulative_right[start + 1:end + 1] = (
            np.cumsum(right_rows, axis=0) + cumulative_right[start]
        )
    return cumulative_gram, cumulative_right


def rolling_affine_matrix_predictions(
    history_prediction: np.ndarray,
    history_target: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *, input_scales: np.ndarray, output_scales: np.ndarray,
    ridge: float, window: int,
) -> np.ndarray:
    """Causally refit a joint affine map from fully resolved forecast paths."""
    if history_prediction.shape != history_target.shape:
        raise ValueError("matrix calibration history shapes differ")
    if prediction.shape != target.shape:
        raise ValueError("matrix calibration evaluation shapes differ")
    combined_prediction = np.concatenate((history_prediction, prediction))
    combined_target = np.concatenate((history_target, target))
    cumulative_gram, cumulative_right = _cumulative_matrix_affine_statistics(
        combined_prediction, combined_target, input_scales, output_scales
    )
    history_count = history_prediction.shape[0]
    horizon = prediction.shape[1]
    resolved_current_paths = np.maximum(
        0, np.arange(prediction.shape[0], dtype=np.int64) - horizon + 1
    )
    ends = history_count + resolved_current_paths
    starts = np.maximum(0, ends - window)
    design = np.concatenate((
        np.ones((prediction.shape[0], 1), dtype=np.float64),
        prediction / input_scales,
    ), axis=1)
    output = np.empty_like(prediction, dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0
    solve_chunk = 2048
    for start in range(0, prediction.shape[0], solve_chunk):
        end = min(start + solve_chunk, prediction.shape[0])
        gram = cumulative_gram[ends[start:end]] - cumulative_gram[starts[start:end]]
        right = cumulative_right[ends[start:end]] - cumulative_right[starts[start:end]]
        counts = np.maximum(ends[start:end] - starts[start:end], 1)
        gram /= counts[:, None, None]
        right /= counts[:, None, None]
        coefficients = np.linalg.solve(
            gram + penalty[None, :, :], right
        )
        output[start:end] = output_scales * np.einsum(
            "nd,ndo->no", design[start:end], coefficients
        )
    return output


def rolling_per_step_polynomial_predictions(
    history_prediction: np.ndarray,
    history_target: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *, degree: int, input_scales: np.ndarray,
    output_scales: np.ndarray, ridge: float, window: int,
) -> np.ndarray:
    output = np.empty_like(prediction, dtype=np.float64)
    size = degree + 1
    history_events = history_prediction.shape[0]
    current_events = prediction.shape[0]
    ends = history_events + np.arange(current_events)
    starts = np.maximum(0, ends - window)
    penalty = np.eye(size, dtype=np.float64) * ridge
    penalty[0, 0] = 0
    for lead in range(prediction.shape[1]):
        gram_parts = []
        right_parts = []
        for values, targets in (
            (history_prediction, history_target), (prediction, target)
        ):
            events = values.shape[0]
            gram = np.zeros((events, size, size), dtype=np.float64)
            right = np.zeros((events, size), dtype=np.float64)
            origins = events - lead
            if origins > 0:
                design = polynomial_design(
                    values[:origins, lead], degree, float(input_scales[lead])
                )
                normalized_target = targets[lead:lead + origins] \
                    / float(output_scales[lead])
                gram[lead:lead + origins] = np.einsum(
                    "ni,nj->nij", design, design
                )
                right[lead:lead + origins] = (
                    design * normalized_target[:, None]
                )
            gram_parts.append(gram)
            right_parts.append(right)
        gram = np.concatenate(gram_parts)
        right = np.concatenate(right_parts)
        cumulative_gram = np.concatenate((
            np.zeros((1, size, size), dtype=np.float64),
            np.cumsum(gram, axis=0),
        ))
        cumulative_right = np.concatenate((
            np.zeros((1, size), dtype=np.float64),
            np.cumsum(right, axis=0),
        ))
        rolling_gram = cumulative_gram[ends] - cumulative_gram[starts]
        rolling_right = cumulative_right[ends] - cumulative_right[starts]
        coefficients = np.linalg.solve(
            rolling_gram + penalty[None, :, :], rolling_right[..., None]
        )[..., 0]
        design = polynomial_design(
            prediction[:, lead], degree, float(input_scales[lead])
        )
        output[:, lead] = float(output_scales[lead]) * np.einsum(
            "nd,nd->n", design, coefficients
        )
    return output


def expectation_calibration_variants_from_arrays(
    arrays: dict[str, dict[str, np.ndarray]],
    *, target_std: float, calibration_window: int,
    matrix_affine_only: bool = False,
) -> dict[str, dict]:
    windowed_arrays = dict(arrays)
    windowed_arrays["calibration"] = {
        key: value[-calibration_window:]
        for key, value in arrays["calibration"].items()
    }
    specifications = () if matrix_affine_only else (
        ("affine-log", "Affine · log returns", "log", 1),
        ("affine-arithmetic", "Affine · arithmetic returns", "arithmetic", 1),
        ("cubic-log", "Full cubic · log returns", "log", 3),
        ("cubic-arithmetic", "Full cubic · arithmetic returns", "arithmetic", 3),
    )
    result: dict[str, dict] = {}
    for key, label, domain, degree in specifications:
        prediction_key = f"{domain}Prediction"
        target_key = f"{domain}Target"
        calibration_prediction = windowed_arrays["calibration"][prediction_key]
        calibration_target = windowed_arrays["calibration"][target_key]
        input_scale = max(float(calibration_prediction.std()), 1e-12)
        output_scale = max(float(calibration_target.std()), 1e-12)
        ridge = 0.0 if degree == 1 else POLYNOMIAL_RIDGE
        coefficients = fit_polynomial(
            calibration_prediction, calibration_target, degree=degree,
            input_scale=input_scale, output_scale=output_scale, ridge=ridge,
        )

        def static_metrics(split: str) -> tuple[dict, list[dict]]:
            calibrated = apply_polynomial(
                windowed_arrays[split][prediction_key], coefficients,
                input_scale=input_scale, output_scale=output_scale,
            )
            calibrated_log = calibrated if domain == "log" else np.log1p(
                np.maximum(calibrated, -1 + 1e-12)
            )
            target_log = windowed_arrays[split]["logTarget"]
            return (
                metric_result(calibrated_log, target_log, target_std),
                per_lead_metric_results(calibrated_log, target_log, target_std),
            )

        static = {
            split: static_metrics(split)
            for split in ("calibration", "validation", "test")
        }

        base = {
            "label": label,
            "mode": "static-pre-validation",
            "domain": domain,
            "degree": degree,
            "ridge": ridge,
            "calibrationWindow": calibration_window,
            "inputScale": input_scale,
            "outputScale": output_scale,
            "coefficients": [float(value) for value in coefficients],
            "calibration": static["calibration"][0],
            "validation": static["validation"][0],
            "test": static["test"][0],
            "calibrationPerLead": static["calibration"][1],
            "validationPerLead": static["validation"][1],
            "testPerLead": static["test"][1],
        }
        result[key] = base

        def online_metrics(split: str, history: str) -> tuple[dict, list[dict]]:
            calibrated = rolling_polynomial_predictions(
                arrays[history][prediction_key], arrays[history][target_key][:, 0],
                arrays[split][prediction_key], arrays[split][target_key][:, 0],
                degree=degree, input_scale=input_scale,
                output_scale=output_scale, ridge=ridge,
                window=calibration_window,
            )
            calibrated_log = calibrated if domain == "log" else np.log1p(
                np.maximum(calibrated, -1 + 1e-12)
            )
            target_log = arrays[split]["logTarget"]
            return (
                metric_result(calibrated_log, target_log, target_std),
                per_lead_metric_results(calibrated_log, target_log, target_std),
            )

        online_validation = online_metrics("validation", "calibration")
        online_test = online_metrics("test", "validation")

        result[f"online-{key}"] = {
            "label": f"Online {label.lower()}",
            "mode": "rolling-resolved-active-returns",
            "domain": domain,
            "degree": degree,
            "ridge": ridge,
            "calibrationWindow": calibration_window,
            "inputScale": input_scale,
            "outputScale": output_scale,
            "trailingActiveReturns": calibration_window,
            "initialHistory": "clean pre-validation calibration slice",
            "validation": online_validation[0],
            "test": online_test[0],
            "validationPerLead": online_validation[1],
            "testPerLead": online_test[1],
        }

        per_step_coefficients, per_step_input_scales, per_step_output_scales = (
            fit_per_step_polynomials(
                calibration_prediction, calibration_target,
                degree=degree, ridge=ridge,
            )
        )

        def per_step_static_metrics(split: str) -> tuple[dict, list[dict]]:
            calibrated = apply_per_step_polynomials(
                windowed_arrays[split][prediction_key], per_step_coefficients,
                per_step_input_scales, per_step_output_scales,
            )
            calibrated_log = calibrated if domain == "log" else np.log1p(
                np.maximum(calibrated, -1 + 1e-12)
            )
            target_log = windowed_arrays[split]["logTarget"]
            return (
                metric_result(calibrated_log, target_log, target_std),
                per_lead_metric_results(calibrated_log, target_log, target_std),
            )

        per_step_static = {
            split: per_step_static_metrics(split)
            for split in ("calibration", "validation", "test")
        }

        result[f"{key}-per-step"] = {
            "label": f"{label} · per step",
            "mode": "static-pre-validation-per-step",
            "domain": domain,
            "degree": degree,
            "ridge": ridge,
            "calibrationWindow": calibration_window,
            "perStep": True,
            "inputScales": [float(value) for value in per_step_input_scales],
            "outputScales": [float(value) for value in per_step_output_scales],
            "coefficientsByStep": [
                [float(value) for value in row]
                for row in per_step_coefficients
            ],
            "calibration": per_step_static["calibration"][0],
            "validation": per_step_static["validation"][0],
            "test": per_step_static["test"][0],
            "calibrationPerLead": per_step_static["calibration"][1],
            "validationPerLead": per_step_static["validation"][1],
            "testPerLead": per_step_static["test"][1],
        }

        def per_step_online_metrics(
            split: str, history: str
        ) -> tuple[dict, list[dict]]:
            calibrated = rolling_per_step_polynomial_predictions(
                arrays[history][prediction_key], arrays[history][target_key][:, 0],
                arrays[split][prediction_key], arrays[split][target_key][:, 0],
                degree=degree, input_scales=per_step_input_scales,
                output_scales=per_step_output_scales, ridge=ridge,
                window=calibration_window,
            )
            calibrated_log = calibrated if domain == "log" else np.log1p(
                np.maximum(calibrated, -1 + 1e-12)
            )
            target_log = arrays[split]["logTarget"]
            return (
                metric_result(calibrated_log, target_log, target_std),
                per_lead_metric_results(calibrated_log, target_log, target_std),
            )

        per_step_online_validation = per_step_online_metrics(
            "validation", "calibration"
        )
        per_step_online_test = per_step_online_metrics("test", "validation")

        result[f"online-{key}-per-step"] = {
            "label": f"Online {label.lower()} · per step",
            "mode": "rolling-resolved-active-returns-per-step",
            "domain": domain,
            "degree": degree,
            "ridge": ridge,
            "calibrationWindow": calibration_window,
            "perStep": True,
            "trailingActiveReturns": calibration_window,
            "initialHistory": "immediate clean pre-validation calibration tail",
            "validation": per_step_online_validation[0],
            "test": per_step_online_test[0],
            "validationPerLead": per_step_online_validation[1],
            "testPerLead": per_step_online_test[1],
        }

    for domain in ("log", "arithmetic"):
        prediction_key = f"{domain}Prediction"
        target_key = f"{domain}Target"
        calibration_prediction = windowed_arrays["calibration"][prediction_key]
        calibration_target = windowed_arrays["calibration"][target_key]
        coefficients, input_scales, output_scales = fit_affine_matrix(
            calibration_prediction, calibration_target,
            ridge=MATRIX_AFFINE_RIDGE,
        )

        def matrix_metrics(split: str) -> tuple[dict, list[dict]]:
            calibrated = apply_affine_matrix(
                windowed_arrays[split][prediction_key], coefficients,
                input_scales, output_scales,
            )
            calibrated_log = calibrated if domain == "log" else np.log1p(
                np.maximum(calibrated, -1 + 1e-12)
            )
            target_log = windowed_arrays[split]["logTarget"]
            return (
                metric_result(calibrated_log, target_log, target_std),
                per_lead_metric_results(calibrated_log, target_log, target_std),
            )

        matrix_static = {
            split: matrix_metrics(split)
            for split in ("calibration", "validation", "test")
        }
        key = f"matrix-affine-{domain}"
        label = f"Matrix affine · {domain} returns"
        leads = calibration_prediction.shape[1]
        result[key] = {
            "label": label,
            "mode": "static-pre-validation-joint-matrix",
            "domain": domain,
            "degree": 1,
            "ridge": MATRIX_AFFINE_RIDGE,
            "calibrationWindow": calibration_window,
            "jointMatrix": True,
            "matrixRows": leads,
            "matrixColumns": leads,
            "coefficientCount": int(coefficients.size),
            "calibration": matrix_static["calibration"][0],
            "validation": matrix_static["validation"][0],
            "test": matrix_static["test"][0],
            "calibrationPerLead": matrix_static["calibration"][1],
            "validationPerLead": matrix_static["validation"][1],
            "testPerLead": matrix_static["test"][1],
        }

        def online_matrix_metrics(split: str, history: str) -> tuple[dict, list[dict]]:
            calibrated = rolling_affine_matrix_predictions(
                arrays[history][prediction_key], arrays[history][target_key],
                arrays[split][prediction_key], arrays[split][target_key],
                input_scales=input_scales, output_scales=output_scales,
                ridge=MATRIX_AFFINE_RIDGE, window=calibration_window,
            )
            calibrated_log = calibrated if domain == "log" else np.log1p(
                np.maximum(calibrated, -1 + 1e-12)
            )
            target_log = arrays[split]["logTarget"]
            return (
                metric_result(calibrated_log, target_log, target_std),
                per_lead_metric_results(calibrated_log, target_log, target_std),
            )

        online_matrix_validation = online_matrix_metrics(
            "validation", "calibration"
        )
        online_matrix_test = online_matrix_metrics("test", "validation")
        result[f"online-{key}"] = {
            "label": f"Online {label.lower()}",
            "mode": "rolling-fully-resolved-paths-joint-matrix",
            "domain": domain,
            "degree": 1,
            "ridge": MATRIX_AFFINE_RIDGE,
            "calibrationWindow": calibration_window,
            "trailingActiveReturns": calibration_window,
            "jointMatrix": True,
            "matrixRows": leads,
            "matrixColumns": leads,
            "coefficientCount": int(coefficients.size),
            "initialHistory": "clean pre-validation calibration paths",
            "validation": online_matrix_validation[0],
            "test": online_matrix_test[0],
            "validationPerLead": online_matrix_validation[1],
            "testPerLead": online_matrix_test[1],
        }
    return result


def expectation_calibration_window_variants(
    model: CompressedPathReturnDensity,
    calibration_dataset: CalibrationPathDataset,
    evaluation_dataset: ImmediateFeatureActivePathDataset,
    densities: tuple[KnotDensityContract, ...],
    *, calibration_windows: tuple[int, ...], batch_size: int,
    target_std: float, device: torch.device,
    matrix_affine_only: bool = False,
) -> dict[str, dict[str, dict]]:
    arithmetic_means = tuple(
        torch.from_numpy(component_arithmetic_return_means(density))
        .float().to(device)
        for density in densities
    )
    arrays = {
        "calibration": collect_point_arrays(
            model, calibration_dataset, "calibration", arithmetic_means,
            batch_size=batch_size, device=device,
        ),
        "validation": collect_point_arrays(
            model, evaluation_dataset, "validation", arithmetic_means,
            batch_size=batch_size, device=device,
        ),
        "test": collect_point_arrays(
            model, evaluation_dataset, "test", arithmetic_means,
            batch_size=batch_size, device=device,
        ),
    }
    return {
        str(window): expectation_calibration_variants_from_arrays(
            arrays, target_std=target_std, calibration_window=window,
            matrix_affine_only=matrix_affine_only,
        )
        for window in calibration_windows
    }


@torch.no_grad()
def fit_temperature(
    model: CompressedPathReturnDensity,
    dataset: CalibrationPathDataset,
    *, batch_size: int, device: torch.device,
) -> tuple[float, dict]:
    totals = torch.zeros(len(TEMPERATURE_GRID), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    seen = 0
    for features, targets, weights in iter_device_batches(
        dataset, "calibration", batch_size, device
    ):
        output = model(features)
        expanded = weights[:, None].expand_as(targets)
        for index, temperature in enumerate(TEMPERATURE_GRID):
            scaled = temperature_scaled_output(output, model, temperature)
            terms = path_log_density_terms(scaled, targets, model)
            totals[index] += (expanded.double() * -terms.double()).sum()
        count += expanded.double().sum()
        seen += int(features.shape[0])
    values = totals / count
    best = int(values.argmin())
    raw_index = TEMPERATURE_GRID.index(1.0)
    return float(TEMPERATURE_GRID[best]), {
        "optimizer": "held-out-log-grid-search",
        "candidates": list(TEMPERATURE_GRID),
        "negativeLogLikelihoods": [float(value) for value in values],
        "examples": seen,
        "temperature": float(TEMPERATURE_GRID[best]),
        "rawNegativeLogLikelihood": float(values[raw_index]),
        "calibratedNegativeLogLikelihood": float(values[best]),
        "nllImprovementVsRaw": float(values[raw_index] - values[best]),
    }


@torch.no_grad()
def evaluate_calibrated_split(
    model: CompressedPathReturnDensity,
    dataset,
    split: str,
    transforms: dict[str, AffineTransform],
    temperature: float,
    *, batch_size: int, target_std: float, cumulative_std: float,
    device: torch.device,
) -> tuple[dict[str, dict], dict]:
    points = {
        name: PathMetrics(target_std, cumulative_std, model.return_count, device)
        for name in transforms
    }
    density = PathMetrics(
        target_std, cumulative_std, model.return_count, device
    )
    for features, targets, weights in iter_device_batches(
        dataset, split, batch_size, device
    ):
        output = model(features)
        raw_terms = path_log_density_terms(output, targets, model)
        for name, transform in transforms.items():
            prediction = output.expectations * transform.scale + transform.intercept
            points[name].add(prediction, targets, weights, raw_terms)
        scaled = temperature_scaled_output(output, model, temperature)
        density.add(
            scaled.expectations, targets, weights,
            path_log_density_terms(scaled, targets, model),
        )
    return (
        {name: metric.result()["expectation"] for name, metric in points.items()},
        density.result(),
    )


def calibration_record(
    model: CompressedPathReturnDensity,
    calibration_dataset: CalibrationPathDataset,
    evaluation_dataset: ImmediateFeatureActivePathDataset,
    densities: tuple[KnotDensityContract, ...],
    provenance: dict,
    *, policy: str, epoch: int, calibration_windows: tuple[int, ...],
    batch_size: int,
    target_std: float, cumulative_std: float, device: torch.device,
    matrix_affine_only: bool = False,
) -> dict:
    transforms, calibration_raw = fit_point_transforms(
        model, calibration_dataset, batch_size=batch_size, device=device
    )
    temperature, temperature_fit = fit_temperature(
        model, calibration_dataset, batch_size=batch_size, device=device
    )
    calibration_points, calibration_density = evaluate_calibrated_split(
        model, calibration_dataset, "calibration", transforms, temperature,
        batch_size=batch_size, target_std=target_std,
        cumulative_std=cumulative_std, device=device,
    )
    validation_points, validation_density = evaluate_calibrated_split(
        model, evaluation_dataset, "validation", transforms, temperature,
        batch_size=batch_size, target_std=target_std,
        cumulative_std=cumulative_std, device=device,
    )
    test_points, test_density = evaluate_calibrated_split(
        model, evaluation_dataset, "test", transforms, temperature,
        batch_size=batch_size, target_std=target_std,
        cumulative_std=cumulative_std, device=device,
    )
    point_window_variants = expectation_calibration_window_variants(
        model, calibration_dataset, evaluation_dataset, densities,
        calibration_windows=calibration_windows, batch_size=batch_size,
        target_std=target_std, device=device,
        matrix_affine_only=matrix_affine_only,
    )
    point_variants = point_window_variants[str(calibration_windows[0])]
    return {
        "contract": CALIBRATION_CONTRACT,
        "checkpointEpoch": epoch,
        "checkpointPolicy": policy,
        "provenance": provenance,
        "fitObjective": {
            "expectation": "shared weighted least squares across all 15 leads",
            "distribution": "mean marginal return-density NLL across all 15 leads",
        },
        "transforms": {
            name: transform.as_dict() for name, transform in transforms.items()
        },
        "calibrationRaw": calibration_raw,
        "calibration": calibration_points,
        "validation": validation_points,
        "test": test_points,
        "pointVariants": point_variants,
        "pointWindowVariants": point_window_variants,
        "densityTemperature": {
            "temperature": temperature,
            "fit": temperature_fit,
            "calibration": calibration_density,
            "validation": validation_density,
            "test": test_density,
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    calibration_windows = tuple(dict.fromkeys(args.calibration_windows))
    if not calibration_windows or any(window < 1 for window in calibration_windows):
        raise ValueError("calibration windows must be positive")
    repo = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo, args.plan)
    plan = json.loads(plan_file.read_text("utf-8"))
    plan_hash = canonical_hash(plan)
    run_root = resolve(repo, Path(plan["runDir"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but is unavailable")
    architecture = plan["architecture"]
    recurrent_market = architecture.get("contract") in {
        RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
        RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
        LOW_RANK_PATH_MATRIX_ARCHITECTURE_CONTRACT,
    }
    feature_history = int(architecture.get("inputFeatureLags", 1))
    dataset_root = resolve(repo, Path(plan["datasetDir"]))
    evaluation_dataset = ImmediateFeatureActivePathDataset(
        dataset_root,
        resolve(repo, Path(plan["historyDir"])),
        int(architecture["returnCount"]),
        feature_history,
        int(plan["subset"]["examples"]),
    )
    stats = training_statistics(
        evaluation_dataset, int(plan["training"]["evaluationBatchSize"])
    )
    density_file = resolve(repo, Path(plan["density"]["source"]))
    densities = (
        tuple(
            KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            for _ in range(int(architecture["returnCount"]))
        )
        if recurrent_market else
        tuple(
            KnotDensityContract.load(density_file, fit=str(width))
            for width in architecture["stateWidths"]
        )
    )
    calibration_root = resolve(repo, args.calibration_dir)
    calibration_dataset = CalibrationPathDataset(
        calibration_root, int(architecture["returnCount"]), feature_history
    )
    available_calibration = calibration_dataset.logical_count("calibration")
    if any(window > available_calibration for window in calibration_windows):
        raise ValueError(
            f"calibration window exceeds {available_calibration} available examples"
        )
    if calibration_dataset.feature_count != evaluation_dataset.feature_count:
        raise ValueError("calibration and model immediate feature widths differ")
    train_times = np.memmap(dataset_root / "train.times.f64", dtype="<f8", mode="r")
    validation_times = np.memmap(
        dataset_root / "validation.times.f64", dtype="<f8", mode="r"
    )
    train_end = float(train_times[-1])
    calibration_start = float(calibration_dataset.times[0])
    validation_start = float(validation_times[0])
    if not train_end < calibration_start < validation_start:
        raise ValueError("calibration split is not strictly pre-validation")
    provenance = {
        "datasetDir": str(calibration_root.relative_to(repo)),
        "examples": calibration_dataset.logical_count("calibration"),
        "sourceExamples": int(calibration_dataset.manifest["examples"]),
        "discardedTerminalStarts": int(architecture["returnCount"]) - 1,
        "interval": calibration_dataset.manifest["interval"],
        "testPolicy": "untouched until calibration parameters were frozen",
    }
    reporter = Reporter(run_root)
    comparison_file = run_root / "state/checkpoint-selection-comparison.json"
    calibration_file = run_root / "state/checkpoint-selection-calibrations.json"
    comparison = json.loads(comparison_file.read_text("utf-8")) \
        if comparison_file.exists() else {
            "contract": "compressed-path-checkpoint-selection-comparison-v1",
            "policies": {},
        }
    calibrations = json.loads(calibration_file.read_text("utf-8")) \
        if calibration_file.exists() else {
            "contract": CONTRACT,
            "trainingPlanId": plan["id"],
            "trainingPlanSha256": plan_hash,
            "policies": {},
        }
    requested = tuple(args.checkpoint_policies)
    evaluated_models: dict[str, str] = {}
    for index, policy in enumerate(requested):
        checkpoint_file = run_root / f"checkpoints/selections/{policy}.json"
        checkpoint = load_torch_checkpoint(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        epoch = int(checkpoint["epoch"])
        fingerprint = model_state_fingerprint(checkpoint["model"])
        reused_policy = evaluated_models.get(fingerprint)
        if reused_policy is not None:
            reused_comparison = copy.deepcopy(
                comparison["policies"][reused_policy]
            )
            reused_comparison.update({
                "epoch": epoch,
                "selectionScore": float(checkpoint["score"]),
                "checkpoint": str(checkpoint_file.relative_to(repo)),
                "evaluationReusedFrom": reused_policy,
            })
            comparison["policies"][policy] = reused_comparison
            atomic_json(comparison, comparison_file)
            reused_calibration = copy.deepcopy(
                calibrations["policies"][reused_policy]
            )
            reused_calibration.update({
                "checkpointEpoch": epoch,
                "checkpointPolicy": policy,
                "evaluationReusedFrom": reused_policy,
            })
            calibrations["policies"][policy] = reused_calibration
            atomic_json(calibrations, calibration_file)
            reporter.status(
                "reused-identical-checkpoint-evaluation",
                planId=plan["id"], policy=policy, epoch=epoch,
                reusedFrom=reused_policy, completedPolicies=index + 1,
                totalPolicies=len(requested),
            )
            continue
        reporter.status(
            "evaluating-checkpoints", planId=plan["id"], policy=policy,
            epoch=epoch, completedPolicies=index, totalPolicies=len(requested),
        )
        model = load_model(plan, densities, checkpoint, device)
        values = {
            split: evaluate(
                model, evaluation_dataset, split, batch_size=args.batch_size,
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]), device=device,
            )
            for split in ("train", "validation", "test")
        }
        comparison["policies"][policy] = {
            "epoch": epoch,
            "selectionScore": float(checkpoint["score"]),
            "train": values["train"]["expectation"],
            "validation": values["validation"]["expectation"],
            "test": values["test"]["expectation"],
            "distribution": values,
            "checkpoint": str(checkpoint_file.relative_to(repo)),
        }
        atomic_json(
            comparison, comparison_file
        )
        reporter.status(
            "calibrating-checkpoints", planId=plan["id"], policy=policy,
            epoch=epoch, completedPolicies=index, totalPolicies=len(requested),
        )
        previous_calibration = calibrations["policies"].get(policy, {})
        updated_calibration = calibration_record(
            model, calibration_dataset, evaluation_dataset, densities, provenance,
            policy=policy, epoch=epoch,
            calibration_windows=calibration_windows, batch_size=args.batch_size,
            target_std=float(stats["targetStd"]),
            cumulative_std=float(stats["cumulativeStd"]), device=device,
            matrix_affine_only=args.matrix_affine_only,
        )
        previous_windows = previous_calibration.get("pointWindowVariants", {})
        updated_windows = updated_calibration["pointWindowVariants"]
        updated_calibration["pointWindowVariants"] = {
            window: {
                **previous_windows.get(window, {}),
                **updated_windows.get(window, {}),
            }
            for window in set(previous_windows) | set(updated_windows)
        }
        updated_calibration["pointVariants"] = {
            **previous_calibration.get("pointVariants", {}),
            **updated_calibration.get("pointVariants", {}),
        }
        calibrations["policies"][policy] = updated_calibration
        atomic_json(
            calibrations, calibration_file
        )
        evaluated_models[fingerprint] = policy
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = comparison["policies"]["validation-nll"]
    result = {
        "version": 1,
        "planId": plan["id"],
        "planSha256": plan_hash,
        "runnerContract": (
            RECURRENT_MARKET_RUNNER_CONTRACT if recurrent_market
            else RUNNER_CONTRACT
        ),
        "stoppedEarly": True,
        "evaluatedSelectionPolicies": list(requested),
        "examples": evaluation_dataset.logical_count("train"),
        "featureCount": evaluation_dataset.feature_count,
        "bestEpoch": selected["epoch"],
        "bestValidationNll": selected["selectionScore"],
        "train": selected["train"],
        "validation": selected["validation"],
        "test": selected["test"],
        "distribution": selected["distribution"],
        "checkpoint": selected["checkpoint"],
    }
    atomic_json(result, run_root / "state/result.json")
    reporter.emit({"event": "stopped-run-checkpoints-evaluated", **result})
    reporter.status("complete", planId=plan["id"], latest=result)


if __name__ == "__main__":
    main()
