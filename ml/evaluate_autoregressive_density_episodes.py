from __future__ import annotations

from collections.abc import Callable
import numpy as np
import torch

from exact_tensor_return_density import (
    ExactTensorReturnDensity,
    exact_tensor_path_log_density,
)
from next_return_dataset import HISTORY_RETURN_COUNT, ROWS_PER_DAY
from return_knot_density import (
    interpolated_log_density_unit,
    transform_returns_to_unit,
)
from train_normalized_glu_next_return import (
    MetricAccumulator,
    NextReturnDataset,
    iter_device_batches,
)


EPISODE_CONTRACT = "autoregressive-cleaned-active-return-episodes-v3"
SOBOL_EXPECTED_EPISODE_CONTRACT = (
    "randomized-sobol-expected-cleaned-active-return-episodes-v1"
)
EXACT_TENSOR_EPISODE_CONTRACT = (
    "exact-tensor-blockwise-cleaned-active-return-episodes-v1"
)
EXACT_TENSOR_SOBOL_EPISODE_CONTRACT = (
    "exact-tensor-blockwise-randomized-sobol-expected-episodes-v1"
)
RETURN_TO_BPS = 10_000.0


def _correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float | None:
    x = prediction.double().reshape(-1)
    y = target.double().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        return None
    return float(torch.dot(x, y) / denominator)


def _fisher_episode_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    cumulative: bool,
) -> tuple[float | None, float | None, int]:
    correlations: list[float] = []
    for index in range(prediction.shape[0]):
        selected_prediction = prediction[index][mask[index]]
        selected_target = target[index][mask[index]]
        if selected_prediction.numel() < 2:
            continue
        if cumulative:
            selected_prediction = torch.cumsum(
                selected_prediction.double(), dim=0
            )
            selected_target = torch.cumsum(selected_target.double(), dim=0)
        correlation = _correlation(selected_prediction, selected_target)
        if correlation is not None:
            correlations.append(correlation)
    if not correlations:
        return None, None, 0
    values = torch.tensor(correlations, dtype=torch.float64)
    values = values.clamp(-0.999999, 0.999999)
    fisher = torch.tanh(torch.atanh(values).mean())
    return float(fisher), float(values.mean()), len(correlations)


def _mse_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | None]:
    error = prediction.double() - target.double()
    mse = float(torch.square(error).mean())
    zero_mse = float(torch.square(target.double()).mean())
    skill = 1 - mse / zero_mse if zero_mse > 0 else None
    return {
        "mse": mse,
        "zeroBaselineMse": zero_mse,
        "mseSkillVsZero": skill,
    }


def _episode_average_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | None]:
    mse: list[float] = []
    zero_mse: list[float] = []
    direction: list[float] = []
    for index in range(prediction.shape[0]):
        selected_prediction = prediction[index][mask[index]]
        selected_target = target[index][mask[index]]
        values = _mse_metrics(selected_prediction, selected_target)
        mse.append(float(values["mse"]))
        zero_mse.append(float(values["zeroBaselineMse"]))
        direction.append(float(
            (torch.sign(selected_prediction) == torch.sign(selected_target))
            .double().mean()
        ))
    mean_mse = float(np.mean(mse))
    mean_zero_mse = float(np.mean(zero_mse))
    return {
        "mse": mean_mse,
        "zeroBaselineMse": mean_zero_mse,
        "mseSkillVsZero": (
            1 - mean_mse / mean_zero_mse if mean_zero_mse > 0 else None
        ),
        "directionAccuracy": float(np.mean(direction)),
    }


def autoregressive_episode_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    source_episode_seconds: int | None = None,
) -> dict[str, object]:
    if prediction.ndim != 2 or prediction.shape != target.shape \
            or prediction.shape[0] < 2 or prediction.shape[1] < 2:
        raise ValueError("episode metrics require matching [episode, step] tensors")
    if not bool(torch.isfinite(prediction).all()) \
            or not bool(torch.isfinite(target).all()):
        raise ValueError("episode forecasts and targets must be finite")
    active = target != 0
    if not bool(active.any()):
        raise ValueError("episode metrics contain no nonzero actual returns")
    clean_prediction = prediction[active]
    clean_target = target[active]
    pooled = _mse_metrics(clean_prediction, clean_target)
    pooled["correlation"] = _correlation(clean_prediction, clean_target)
    pooled["directionAccuracy"] = float(
        (torch.sign(clean_prediction) == torch.sign(clean_target))
        .double().mean()
    )

    episode_correlation, episode_correlation_mean, valid_return = (
        _fisher_episode_correlation(
            prediction, target, active, cumulative=False
        )
    )
    episode_average = _episode_average_metrics(prediction, target, active)
    episode_average["correlation"] = episode_correlation
    path_correlation, path_correlation_mean, valid_path = (
        _fisher_episode_correlation(
            prediction, target, active, cumulative=True
        )
    )
    predicted_endpoint = (prediction.double() * active).sum(dim=1)
    actual_endpoint = (target.double() * active).sum(dim=1)
    endpoint = _mse_metrics(predicted_endpoint, actual_endpoint)
    endpoint["correlation"] = _correlation(
        predicted_endpoint, actual_endpoint
    )
    endpoint["directionAccuracy"] = float(
        (torch.sign(predicted_endpoint) == torch.sign(actual_endpoint))
        .double().mean()
    )
    active_counts = active.sum(dim=1)
    result: dict[str, object] = {
        "contract": EPISODE_CONTRACT,
        "episodes": int(prediction.shape[0]),
        "activeCandles": int(active.sum()),
        "activeCandlesPerEpisode": {
            "minimum": int(active_counts.min()),
            "mean": float(active_counts.double().mean()),
            "maximum": int(active_counts.max()),
        },
        "datasetFilter": (
            "remove-exact-zero-realized-returns-before-rollout"
        ),
        "pooledCandles": pooled,
        "episodeAverage": episode_average,
        "episodeReturnCorrelation": episode_correlation,
        "episodeReturnCorrelationArithmeticMean": episode_correlation_mean,
        "validReturnCorrelationEpisodes": valid_return,
        "episodeCumulativePathCorrelation": path_correlation,
        "episodeCumulativePathCorrelationArithmeticMean": (
            path_correlation_mean
        ),
        "validCumulativePathCorrelationEpisodes": valid_path,
        "episodeEndpoint": endpoint,
    }
    if source_episode_seconds is not None:
        result["sourceEpisodeSeconds"] = int(source_episode_seconds)
    return result


def collect_contiguous_episodes(
    dataset: NextReturnDataset,
    split: str,
    *,
    episode_seconds: int,
    maximum_episodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if episode_seconds < 2 or maximum_episodes < 2:
        raise ValueError("episode length and count must both be at least two")
    starts: list[tuple[str, int]] = []
    for shard in dataset.shards[split]:
        first = int(shard.row_offset)
        final = min(
            ROWS_PER_DAY,
            int(shard.row_offset) + int(shard.count),
        )
        for row in range(first, final - episode_seconds + 1, episode_seconds):
            starts.append((shard.date, row))
    if len(starts) < 2:
        raise ValueError("held-out split contains fewer than two full episodes")
    if len(starts) > maximum_episodes:
        indices = np.linspace(
            0, len(starts) - 1, num=maximum_episodes, dtype=np.int64
        )
        starts = [starts[int(index)] for index in indices]
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for day, row in starts:
        component = dataset._component(day)
        history, target = component[0], component[-1]
        realized = target[row:row + episode_seconds]
        cleaned = realized[realized != 0]
        if cleaned.size == 0:
            continue
        histories.append(history[row])
        targets.append(cleaned)
    if len(histories) < 2:
        raise ValueError("held-out split contains fewer than two active episodes")
    feature = np.stack(histories).astype(np.float32, copy=False)
    lengths = np.asarray([value.size for value in targets], dtype=np.int64)
    actual = np.zeros((len(targets), int(lengths.max())), dtype=np.float32)
    for index, value in enumerate(targets):
        actual[index, :value.size] = value
    if feature.shape != (len(histories), HISTORY_RETURN_COUNT) \
            or actual.shape[0] != len(histories):
        raise RuntimeError("invalid contiguous episode construction")
    return (
        torch.from_numpy(feature),
        torch.from_numpy(actual),
        torch.from_numpy(lengths),
    )


def _randomized_sobol_uniforms(
    steps: int,
    *,
    replicates: int,
    trajectories_per_replicate: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    blocks = [
        torch.quasirandom.SobolEngine(
            dimension=2 * steps,
            scramble=True,
            seed=seed + replicate,
        ).draw(trajectories_per_replicate)
        for replicate in range(replicates)
    ]
    return torch.cat(blocks, dim=0).to(device)


def _sample_triangular_components(
    component: torch.Tensor,
    uniform: torch.Tensor,
    knots: torch.Tensor,
) -> torch.Tensor:
    component = component.long()
    uniform = uniform.float().clamp(0, 1)
    final = int(knots.numel() - 1)
    result = torch.empty_like(uniform)
    first = component == 0
    last = component == final
    interior = ~(first | last)
    if bool(first.any()):
        gap = knots[1] - knots[0]
        result[first] = knots[0] + gap * (
            1 - torch.sqrt(1 - uniform[first])
        )
    if bool(last.any()):
        gap = knots[-1] - knots[-2]
        result[last] = knots[-2] + gap * torch.sqrt(uniform[last])
    if bool(interior.any()):
        index = component[interior]
        values = uniform[interior]
        left_gap = knots[index] - knots[index - 1]
        right_gap = knots[index + 1] - knots[index]
        left_probability = left_gap / (left_gap + right_gap)
        choose_left = values < left_probability
        position = torch.empty_like(values)
        if bool(choose_left.any()):
            conditional = (
                values[choose_left] / left_probability[choose_left]
            ).clamp(0, 1)
            selected = index[choose_left]
            position[choose_left] = (
                knots[selected - 1]
                + left_gap[choose_left] * torch.sqrt(conditional)
            )
        choose_right = ~choose_left
        if bool(choose_right.any()):
            conditional = (
                (values[choose_right] - left_probability[choose_right])
                / (1 - left_probability[choose_right])
            ).clamp(0, 1)
            selected = index[choose_right]
            position[choose_right] = (
                knots[selected]
                + right_gap[choose_right]
                * (1 - torch.sqrt(1 - conditional))
            )
        result[interior] = position
    return result


def _sample_returns_from_component_masses(
    log_masses: torch.Tensor,
    component_uniform: torch.Tensor,
    basis_uniform: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    cdf = torch.cumsum(log_masses.exp(), dim=1)
    component = torch.searchsorted(
        cdf.contiguous(), component_uniform[:, None].contiguous()
    ).squeeze(1).clamp_max(cdf.shape[1] - 1)
    unit = _sample_triangular_components(
        component,
        basis_uniform,
        model.density_knots_unit,
    )
    epsilon = torch.finfo(unit.dtype).eps
    unit = unit.clamp(epsilon, 1 - epsilon)
    transform = model.density_transform
    logit = torch.log(unit) - torch.log1p(-unit)
    return (
        float(transform.location_bps)
        + float(transform.scale_bps)
        * torch.sinh(logit / float(transform.alpha))
    ) / RETURN_TO_BPS


def _sample_exact_tensor_returns(
    model_output: object,
    uniforms: torch.Tensor,
    model: ExactTensorReturnDensity,
) -> torch.Tensor:
    """Draw one continuous three-return path from the exact joint tensor."""
    if uniforms.ndim != 2 or uniforms.shape[1] != model.return_count + 1:
        raise ValueError("exact tensor sampling requires four uniforms per path")
    joint = model_output.joint_component_masses[-1]
    flattened = joint.reshape(joint.shape[0], -1)
    cdf = torch.cumsum(flattened, dim=1)
    component = torch.searchsorted(
        cdf.contiguous(), uniforms[:, :1].contiguous()
    ).squeeze(1).clamp_max(flattened.shape[1] - 1)
    knot_count = model.knot_count
    indexes = (
        component // (knot_count * knot_count),
        (component // knot_count) % knot_count,
        component % knot_count,
    )
    units = torch.stack(tuple(
        _sample_triangular_components(
            index, uniforms[:, lead + 1], model.density_knots_unit
        )
        for lead, index in enumerate(indexes)
    ), dim=1)
    epsilon = torch.finfo(units.dtype).eps
    units = units.clamp(epsilon, 1 - epsilon)
    transform = model.density_transform
    logits = torch.log(units) - torch.log1p(-units)
    return (
        float(transform.location_bps)
        + float(transform.scale_bps)
        * torch.sinh(logits / float(transform.alpha))
    ) / RETURN_TO_BPS


def _return_log_density(
    log_masses: torch.Tensor,
    returns: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    unit, log_jacobian = transform_returns_to_unit(
        returns, model.density_transform
    )
    log_density_unit = interpolated_log_density_unit(
        log_masses,
        unit,
        model.density_knots_unit,
        model.density_basis_areas,
    )
    return log_density_unit + log_jacobian


def _estimator_summary(values: list[torch.Tensor]) -> dict[str, float]:
    flattened = torch.cat([value.double().reshape(-1) for value in values])
    standard_error = torch.sqrt(flattened.clamp_min(0))
    return {
        "meanVariance": float(flattened.mean()),
        "meanStandardError": float(standard_error.mean()),
        "p95StandardError": float(torch.quantile(standard_error, 0.95)),
        "maximumStandardError": float(standard_error.max()),
    }


def _path_likelihood_summary(
    negative_log_density_per_candle: list[float],
    density_percentiles: list[float],
    two_sided_typicalities: list[float],
    log_density_z_scores: list[float],
    density_ratios: list[float],
) -> dict[str, object]:
    nll = np.asarray(negative_log_density_per_candle, dtype=np.float64)
    percentiles = np.asarray(density_percentiles, dtype=np.float64)
    typicalities = np.asarray(two_sided_typicalities, dtype=np.float64)
    z_scores = np.asarray(log_density_z_scores, dtype=np.float64)
    ratios = np.asarray(density_ratios, dtype=np.float64)
    return {
        "measure": "continuous-return-space-joint-log-density",
        "exactPathProbabilityMass": 0.0,
        "exactMassExplanation": (
            "an exact path has zero mass under a continuous density"
        ),
        "episodes": int(nll.size),
        "realizedNegativeLogDensityPerCandle": {
            "mean": float(nll.mean()),
            "median": float(np.median(nll)),
            "p95": float(np.quantile(nll, 0.95)),
        },
        "realizedBitsPerCandle": {
            "mean": float(nll.mean() / np.log(2)),
            "median": float(np.median(nll) / np.log(2)),
        },
        "sampledPathLogDensityPercentile": {
            "mean": float(percentiles.mean()),
            "median": float(np.median(percentiles)),
            "minimum": float(percentiles.min()),
            "fractionBelow1Percent": float((percentiles < 0.01).mean()),
            "fractionBelow5Percent": float((percentiles < 0.05).mean()),
        },
        "twoSidedTypicality": {
            "mean": float(typicalities.mean()),
            "median": float(np.median(typicalities)),
            "minimum": float(typicalities.min()),
        },
        "logDensityZScoreVsSampledPaths": {
            "mean": float(z_scores.mean()),
            "median": float(np.median(z_scores)),
        },
        "perCandleDensityRatioVsSampleMedian": {
            "mean": float(ratios.mean()),
            "median": float(np.median(ratios)),
        },
    }


@torch.no_grad()
def evaluate_sobol_expected_episodes(
    model: torch.nn.Module,
    dataset: NextReturnDataset,
    split: str,
    *,
    episode_seconds: int,
    maximum_episodes: int,
    trajectories: int,
    randomized_replicates: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    if trajectories < 2 or randomized_replicates < 2 \
            or trajectories % randomized_replicates != 0:
        raise ValueError(
            "trajectories must divide into at least two Sobol replicates"
        )
    per_replicate = trajectories // randomized_replicates
    feature, actual, lengths = collect_contiguous_episodes(
        dataset,
        split,
        episode_seconds=episode_seconds,
        maximum_episodes=maximum_episodes,
    )
    actual = actual.to(device)
    expected = torch.zeros_like(actual)
    return_trajectory_variances: list[torch.Tensor] = []
    return_estimator_variances: list[torch.Tensor] = []
    cumulative_estimator_variances: list[torch.Tensor] = []
    endpoint_trajectory_variances: list[torch.Tensor] = []
    endpoint_estimator_variances: list[torch.Tensor] = []
    realized_nll_per_candle: list[float] = []
    realized_density_percentiles: list[float] = []
    realized_two_sided_typicalities: list[float] = []
    realized_log_density_z_scores: list[float] = []
    realized_density_ratios: list[float] = []
    model.eval()
    for episode in range(feature.shape[0]):
        steps = int(lengths[episode])
        history = feature[episode].to(device)[None, :].repeat(
            trajectories, 1
        )
        uniforms = _randomized_sobol_uniforms(
            steps,
            replicates=randomized_replicates,
            trajectories_per_replicate=per_replicate,
            seed=seed + episode * randomized_replicates,
            device=device,
        )
        paths = torch.empty(
            (trajectories, steps), dtype=history.dtype, device=device
        )
        sampled_path_log_density = torch.zeros(
            trajectories, dtype=torch.float64, device=device
        )
        for step in range(steps):
            log_masses = model.log_component_masses(history)
            sampled = _sample_returns_from_component_masses(
                log_masses,
                uniforms[:, 2 * step],
                uniforms[:, 2 * step + 1],
                model,
            )
            paths[:, step] = sampled
            sampled_path_log_density += _return_log_density(
                log_masses, sampled, model
            ).double()
            history = torch.cat((history[:, 1:], sampled[:, None]), dim=1)
        realized_history = feature[episode].to(device)[None, :]
        realized_path_log_density = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        for step in range(steps):
            realized_return = actual[episode, step:step + 1]
            realized_log_masses = model.log_component_masses(realized_history)
            realized_path_log_density += _return_log_density(
                realized_log_masses, realized_return, model
            ).double().squeeze(0)
            realized_history = torch.cat(
                (realized_history[:, 1:], realized_return[:, None]), dim=1
            )
        realized_score = float(realized_path_log_density)
        sampled_mean = float(sampled_path_log_density.mean())
        sampled_std = float(sampled_path_log_density.std(unbiased=True))
        sampled_median = float(sampled_path_log_density.median())
        lower_count = int(
            (sampled_path_log_density <= realized_path_log_density).sum()
        )
        percentile = (lower_count + 0.5) / (trajectories + 1.0)
        realized_nll_per_candle.append(-realized_score / steps)
        realized_density_percentiles.append(percentile)
        realized_two_sided_typicalities.append(
            min(1.0, 2 * min(percentile, 1 - percentile))
        )
        realized_log_density_z_scores.append(
            (realized_score - sampled_mean) / sampled_std
            if sampled_std > 0 else 0.0
        )
        realized_density_ratios.append(float(np.exp(np.clip(
            (realized_score - sampled_median) / steps, -50, 50
        ))))
        expected[episode, :steps] = paths.mean(dim=0)
        replicate_means = paths.reshape(
            randomized_replicates, per_replicate, steps
        ).mean(dim=1)
        return_trajectory_variances.append(paths.var(dim=0, unbiased=True))
        return_estimator_variances.append(
            replicate_means.var(dim=0, unbiased=True)
            / randomized_replicates
        )
        cumulative_replicate_means = torch.cumsum(replicate_means, dim=1)
        cumulative_estimator_variances.append(
            cumulative_replicate_means.var(dim=0, unbiased=True)
            / randomized_replicates
        )
        endpoints = paths.sum(dim=1)
        endpoint_trajectory_variances.append(
            endpoints.var(unbiased=True).reshape(1)
        )
        endpoint_estimator_variances.append(
            replicate_means.sum(dim=1).var(unbiased=True).reshape(1)
            / randomized_replicates
        )
    metrics = autoregressive_episode_metrics(
        expected,
        actual,
        source_episode_seconds=episode_seconds,
    )
    metrics["contract"] = SOBOL_EXPECTED_EPISODE_CONTRACT
    metrics["estimator"] = {
        "method": "randomized-scrambled-sobol",
        "trajectories": trajectories,
        "randomizedReplicates": randomized_replicates,
        "trajectoriesPerReplicate": per_replicate,
        "returnTrajectoryVarianceMean": float(torch.cat(
            return_trajectory_variances
        ).double().mean()),
        "returnMean": _estimator_summary(return_estimator_variances),
        "cumulativeLogPricePathMean": _estimator_summary(
            cumulative_estimator_variances
        ),
        "endpointTrajectoryVarianceMean": float(torch.cat(
            endpoint_trajectory_variances
        ).double().mean()),
        "endpointMean": _estimator_summary(endpoint_estimator_variances),
    }
    metrics["pathLikelihood"] = _path_likelihood_summary(
        realized_nll_per_candle,
        realized_density_percentiles,
        realized_two_sided_typicalities,
        realized_log_density_z_scores,
        realized_density_ratios,
    )
    return metrics


@torch.no_grad()
def evaluate_autoregressive_episodes(
    model: torch.nn.Module,
    dataset: NextReturnDataset,
    split: str,
    *,
    episode_seconds: int,
    maximum_episodes: int,
    device: torch.device,
) -> dict[str, object]:
    feature, actual, lengths = collect_contiguous_episodes(
        dataset,
        split,
        episode_seconds=episode_seconds,
        maximum_episodes=maximum_episodes,
    )
    history = feature.to(device)
    actual = actual.to(device)
    lengths = lengths.to(device)
    prediction = torch.zeros_like(actual)
    model.eval()
    for step in range(actual.shape[1]):
        active = lengths > step
        active_history = history[active]
        log_masses = model.log_component_masses(active_history)
        forecast = (
            log_masses.exp() @ model.density_component_return_means
        )
        prediction[active, step] = forecast
        history[active] = torch.cat(
            (active_history[:, 1:], forecast[:, None]), dim=1
        )
    return autoregressive_episode_metrics(
        prediction,
        actual,
        source_episode_seconds=episode_seconds,
    )


@torch.no_grad()
def evaluate_autoregressive_leads(
    model: torch.nn.Module,
    dataset: object,
    split: str,
    *,
    lead_count: int,
    batch_size: int,
    target_std: float,
    device: torch.device,
) -> list[dict[str, float | int | None]]:
    """Evaluate fixed lead positions from recursive expected-return rollout."""
    if lead_count < 1:
        raise ValueError("autoregressive lead count must be positive")
    if getattr(dataset, "return_count", None) != lead_count:
        raise ValueError("autoregressive lead dataset has the wrong path width")
    metrics = tuple(
        MetricAccumulator(target_std, device) for _ in range(lead_count)
    )
    model.eval()
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        history = features
        for lead, accumulator in enumerate(metrics):
            log_masses = model.log_component_masses(history)
            forecast = (
                log_masses.exp() @ model.density_component_return_means
            )
            accumulator.add(forecast, targets[:, lead], weights)
            history = torch.cat((history[:, 1:], forecast[:, None]), dim=1)
    return [value.result() for value in metrics]


@torch.no_grad()
def evaluate_exact_tensor_autoregressive_episodes(
    model: ExactTensorReturnDensity,
    dataset: object,
    split: str,
    *,
    episode_seconds: int,
    maximum_episodes: int,
    device: torch.device,
) -> dict[str, object]:
    """Roll out three expected active returns per exact-tensor forward pass."""
    feature, actual, lengths = collect_contiguous_episodes(
        dataset,
        split,
        episode_seconds=episode_seconds,
        maximum_episodes=maximum_episodes,
    )
    history = feature.to(device)
    actual = actual.to(device)
    lengths = lengths.to(device)
    prediction = torch.zeros_like(actual)
    model.eval()
    for block_start in range(0, actual.shape[1], model.return_count):
        block_active = lengths > block_start
        if not bool(block_active.any()):
            break
        active_history = history[block_active]
        forecast = model(active_history).expectations
        active_lengths = lengths[block_active]
        for lead in range(model.return_count):
            step = block_start + lead
            if step >= actual.shape[1]:
                break
            lead_active = active_lengths > step
            prediction[block_active, step] = torch.where(
                lead_active,
                forecast[:, lead],
                prediction[block_active, step],
            )
            if bool(lead_active.any()):
                selected_history = active_history[lead_active]
                active_history[lead_active] = torch.cat((
                    selected_history[:, 1:],
                    forecast[lead_active, lead:lead + 1],
                ), dim=1)
        history[block_active] = active_history
    metrics = autoregressive_episode_metrics(
        prediction,
        actual,
        source_episode_seconds=episode_seconds,
    )
    metrics["contract"] = EXACT_TENSOR_EPISODE_CONTRACT
    metrics["rolloutBlockReturns"] = model.return_count
    return metrics


@torch.no_grad()
def evaluate_exact_tensor_sobol_expected_episodes(
    model: ExactTensorReturnDensity,
    dataset: object,
    split: str,
    *,
    episode_seconds: int,
    maximum_episodes: int,
    trajectories: int,
    randomized_replicates: int,
    seed: int,
    device: torch.device,
    trajectory_batch_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Estimate expected paths by sampling exact three-return joint blocks."""
    if trajectories < 2 or randomized_replicates < 2 \
            or trajectories % randomized_replicates != 0:
        raise ValueError(
            "trajectories must divide into at least two Sobol replicates"
        )
    if trajectory_batch_size < 1:
        raise ValueError("trajectory batch size must be positive")
    per_replicate = trajectories // randomized_replicates
    feature, actual, lengths = collect_contiguous_episodes(
        dataset,
        split,
        episode_seconds=episode_seconds,
        maximum_episodes=maximum_episodes,
    )
    actual = actual.to(device)
    expected = torch.zeros_like(actual)
    return_trajectory_variances: list[torch.Tensor] = []
    return_estimator_variances: list[torch.Tensor] = []
    cumulative_estimator_variances: list[torch.Tensor] = []
    endpoint_trajectory_variances: list[torch.Tensor] = []
    endpoint_estimator_variances: list[torch.Tensor] = []
    realized_nll_per_candle: list[float] = []
    realized_density_percentiles: list[float] = []
    realized_two_sided_typicalities: list[float] = []
    realized_log_density_z_scores: list[float] = []
    realized_density_ratios: list[float] = []
    model.eval()
    for episode in range(feature.shape[0]):
        steps = int(lengths[episode])
        blocks = (steps + model.return_count - 1) // model.return_count
        history = feature[episode].to(device)[None, :].repeat(
            trajectories, 1
        )
        # The shared helper allocates two dimensions per requested "step";
        # passing 2*blocks gives one component and three basis uniforms/block.
        uniforms = _randomized_sobol_uniforms(
            2 * blocks,
            replicates=randomized_replicates,
            trajectories_per_replicate=per_replicate,
            seed=seed + episode * randomized_replicates,
            device=device,
        )
        paths = torch.empty(
            (trajectories, steps), dtype=history.dtype, device=device
        )
        sampled_path_log_density = torch.zeros(
            trajectories, dtype=torch.float64, device=device
        )
        for block in range(blocks):
            start = block * model.return_count
            take = min(model.return_count, steps - start)
            sampled_block = torch.empty(
                (trajectories, model.return_count),
                dtype=history.dtype,
                device=device,
            )
            for batch_start in range(0, trajectories, trajectory_batch_size):
                batch_end = min(
                    trajectories, batch_start + trajectory_batch_size
                )
                selection = slice(batch_start, batch_end)
                output = model(history[selection])
                sampled = _sample_exact_tensor_returns(
                    output,
                    uniforms[
                        selection, block * 4:(block + 1) * 4
                    ],
                    model,
                )
                sampled_block[selection] = sampled
                sampled_path_log_density[selection] += (
                    exact_tensor_path_log_density(
                        output, sampled, model
                    )[:, :take].double().sum(dim=1)
                )
            paths[:, start:start + take] = sampled_block[:, :take]
            history = torch.cat((
                history[:, take:], sampled_block[:, :take]
            ), dim=1)
        realized_history = feature[episode].to(device)[None, :]
        realized_path_log_density = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        for block in range(blocks):
            start = block * model.return_count
            take = min(model.return_count, steps - start)
            realized = torch.zeros(
                (1, model.return_count),
                dtype=realized_history.dtype,
                device=device,
            )
            realized[:, :take] = actual[episode, start:start + take]
            output = model(realized_history)
            realized_path_log_density += exact_tensor_path_log_density(
                output, realized, model
            )[:, :take].double().sum()
            realized_history = torch.cat((
                realized_history[:, take:], realized[:, :take]
            ), dim=1)
        realized_score = float(realized_path_log_density)
        sampled_mean = float(sampled_path_log_density.mean())
        sampled_std = float(sampled_path_log_density.std(unbiased=True))
        sampled_median = float(sampled_path_log_density.median())
        lower_count = int(
            (sampled_path_log_density <= realized_path_log_density).sum()
        )
        percentile = (lower_count + 0.5) / (trajectories + 1.0)
        realized_nll_per_candle.append(-realized_score / steps)
        realized_density_percentiles.append(percentile)
        realized_two_sided_typicalities.append(
            min(1.0, 2 * min(percentile, 1 - percentile))
        )
        realized_log_density_z_scores.append(
            (realized_score - sampled_mean) / sampled_std
            if sampled_std > 0 else 0.0
        )
        realized_density_ratios.append(float(np.exp(np.clip(
            (realized_score - sampled_median) / steps, -50, 50
        ))))
        expected[episode, :steps] = paths.mean(dim=0)
        replicate_means = paths.reshape(
            randomized_replicates, per_replicate, steps
        ).mean(dim=1)
        return_trajectory_variances.append(paths.var(dim=0, unbiased=True))
        return_estimator_variances.append(
            replicate_means.var(dim=0, unbiased=True)
            / randomized_replicates
        )
        cumulative_replicate_means = torch.cumsum(replicate_means, dim=1)
        cumulative_estimator_variances.append(
            cumulative_replicate_means.var(dim=0, unbiased=True)
            / randomized_replicates
        )
        endpoints = paths.sum(dim=1)
        endpoint_trajectory_variances.append(
            endpoints.var(unbiased=True).reshape(1)
        )
        endpoint_estimator_variances.append(
            replicate_means.sum(dim=1).var(unbiased=True).reshape(1)
            / randomized_replicates
        )
        if progress is not None:
            progress(episode + 1, feature.shape[0])
    metrics = autoregressive_episode_metrics(
        expected,
        actual,
        source_episode_seconds=episode_seconds,
    )
    metrics["contract"] = EXACT_TENSOR_SOBOL_EPISODE_CONTRACT
    metrics["rolloutBlockReturns"] = model.return_count
    metrics["estimator"] = {
        "method": "randomized-scrambled-sobol-exact-joint-blocks",
        "trajectories": trajectories,
        "randomizedReplicates": randomized_replicates,
        "trajectoriesPerReplicate": per_replicate,
        "returnTrajectoryVarianceMean": float(torch.cat(
            return_trajectory_variances
        ).double().mean()),
        "returnMean": _estimator_summary(return_estimator_variances),
        "cumulativeLogPricePathMean": _estimator_summary(
            cumulative_estimator_variances
        ),
        "endpointTrajectoryVarianceMean": float(torch.cat(
            endpoint_trajectory_variances
        ).double().mean()),
        "endpointMean": _estimator_summary(endpoint_estimator_variances),
    }
    metrics["pathLikelihood"] = _path_likelihood_summary(
        realized_nll_per_candle,
        realized_density_percentiles,
        realized_two_sided_typicalities,
        realized_log_density_z_scores,
        realized_density_ratios,
    )
    return metrics
