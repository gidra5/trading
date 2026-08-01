from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fit_teacher_cuda import (
    FitConfig,
    fit_batch,
    fit_target,
    domain_sampled_indices,
)
from mlp_model import PolicySupport, conditional_policy_logits, scaled_softplus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cutoff-aware teacher-fit rejection shapes.")
    parser.add_argument("--input", type=Path, default=Path("data/training/analysis/mlp-rejection-oracles"))
    parser.add_argument("--output", type=Path, default=Path("data/training/analysis/mlp-rejection-analysis.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.input / "metadata.json").read_text())
    actions_list = metadata["actionGrid"]
    currents_list = metadata["currentGrid"]
    count = len(metadata["cases"])
    base_np = np.fromfile(args.input / metadata["probabilitiesFile"], dtype="<f4")
    base_np = base_np.reshape(count, len(actions_list))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(20260721)
    base = torch.from_numpy(base_np.copy()).to(device)
    cutoff = torch.tensor([-14.0, 14.0], device=device).expand(base.shape[0], -1)
    packed = torch.cat((base, cutoff), dim=-1)
    actions = torch.tensor(actions_list, dtype=torch.float32, device=device)
    currents = torch.tensor(currents_list, dtype=torch.float32, device=device)
    execution = metadata["execution"]
    fit_plan = json.loads(Path("ml/training-plan.json").read_text())["teacherFit"]
    config = FitConfig(
        action_grid=actions_list,
        current_grid=currents_list,
        friction=execution["feeBps"] / 10_000,
        transition_log_scale=1 / execution["temperature"],
        latent_lower=execution["minimumEffectiveExposure"],
        latent_upper=execution["maximumEffectiveExposure"],
        visible_lower=execution["minimumEffectiveExposure"],
        visible_upper=execution["maximumEffectiveExposure"],
        metric_visible_lower=execution["minimumUsableExposure"],
        metric_visible_upper=execution["maximumUsableExposure"],
        sample_states=fit_plan["sampleStates"],
        sample_actions=fit_plan["sampleActions"],
        projection_iterations=fit_plan["projectionIterations"],
        iterations=fit_plan["maxIterations"],
        adaptive_iterations=fit_plan["adaptiveIterations"],
        adaptive_rounds=fit_plan["adaptiveRounds"],
        restarts=fit_plan["restartCount"],
        batch_size=count,
        tolerance=fit_plan["tolerance"],
        max_mean_kl=fit_plan["maxMeanKlDivergence"],
        max_mean_mse=fit_plan["maxMeanSquaredError"],
        line_search_candidates=fit_plan["lineSearchCandidates"],
        temporal_refinement_rounds=0,
        temporal_iterations=fit_plan["temporalIterations"],
        temporal_followup_iterations=fit_plan["temporalFollowupIterations"],
        temporal_equivalent_loss_absolute=fit_plan["temporalEquivalentLossAbsolute"],
        temporal_equivalent_loss_relative=fit_plan["temporalEquivalentLossRelative"],
        optimizer_backend=fit_plan["optimizerBackend"],
        optimizer_host_check_interval=fit_plan["optimizerHostCheckInterval"],
        quality_fallback_iterations=fit_plan["qualityFallbackIterations"],
        input_queue_batches=fit_plan["inputQueueBatches"],
        visible_sample_fraction=fit_plan["visibleSampleFraction"],
        score_hinge_span=fit_plan["scoreHingeSpan"],
        compact_visible_initialization=fit_plan["compactVisibleInitialization"],
    )
    support = PolicySupport(
        config.latent_lower,
        config.latent_upper,
        config.visible_lower,
        config.visible_upper,
        config.friction,
        1 / config.transition_log_scale,
        config.score_hinge_span,
    )
    state_indexes = domain_sampled_indices(
        currents, config.sample_states, config.metric_visible_lower,
        config.metric_visible_upper, config.visible_sample_fraction,
    )
    action_indexes = domain_sampled_indices(
        actions, config.sample_actions, config.metric_visible_lower,
        config.metric_visible_upper, config.visible_sample_fraction,
    )
    sampled_actions = actions[action_indexes]
    sampled_currents = currents[state_indexes]
    print(json.dumps({"event": "analysis-fit-baseline", "examples": count}), flush=True)
    baseline_raw, _, _, _, _ = fit_batch(
        packed,
        actions,
        currents,
        sampled_actions,
        sampled_currents,
        action_indexes,
        support,
        config,
    )
    full_target, _ = fit_target(base, actions, currents, config)
    sampled_target = full_target[:, state_indexes][:, :, action_indexes]
    sampled_target = sampled_target / sampled_target.sum(dim=-1, keepdim=True)
    metric_action_mask = (
        (actions >= config.metric_visible_lower)
        & (actions <= config.metric_visible_upper)
    )
    metric_current_mask = (
        (currents >= config.metric_visible_lower)
        & (currents <= config.metric_visible_upper)
    )
    metric_actions = actions[metric_action_mask]
    metric_currents = currents[metric_current_mask]
    metric_target, metric_entropy = fit_target(
        base[:, metric_action_mask], metric_actions, metric_currents, config
    )
    baseline = evaluate(
        "six_parameter", baseline_raw, None,
        metric_target, metric_entropy, metric_actions, metric_currents, support,
    )

    variants = [
        ("learn_beta_x", 1, [None]),
        ("learn_kappa_x", 1, [None]),
        ("learn_kappa_c", 2, [None]),
        ("cubic", 1, [None]),
        ("split_cubic_tails", 2, [None]),
        ("fee_quadratic_sides", 2, [None]),
        ("action_current_interaction", 1, [None]),
        ("third_hinge", 2, [-100.0, -50.0, 0.0, 50.0, 100.0]),
    ]
    results: dict[str, dict[str, Any]] = {"six_parameter": baseline}
    candidates: dict[str, tuple[Tensor, Tensor | None]] = {
        "six_parameter": (baseline_raw, None),
    }
    for name, extra_count, starts in variants:
        print(json.dumps({"event": "analysis-fit-variant", "variant": name}), flush=True)
        raw, extra = optimize_variant(
            name,
            baseline_raw,
            sampled_target,
            sampled_actions,
            sampled_currents,
            support,
            extra_count,
            starts,
            args.steps,
        )
        result = evaluate(
            name, raw, extra,
            metric_target, metric_entropy, metric_actions, metric_currents, support,
        )
        results[name] = result
        candidates[name] = (raw, extra)
    baseline_kl = np.asarray(baseline["perExample"]["klDivergence"])
    for name, result in results.items():
        values = np.asarray(result["perExample"]["klDivergence"])
        mse_values = np.asarray(result["perExample"]["meanSquaredError"])
        result["summary"]["meanKlReductionVsBaseline"] = float(np.mean(baseline_kl - values))
        result["summary"]["relativeMeanKlReductionVsBaseline"] = float(
            (baseline_kl.mean() - values.mean()) / baseline_kl.mean()
        )
        result["summary"]["improvedExampleFraction"] = float(np.mean(values < baseline_kl - 1e-7))
        ordinary = baseline_kl <= 0.03
        result["summary"]["ordinaryCaseCount"] = int(ordinary.sum())
        result["summary"]["ordinaryMeanKlDivergence"] = float(values[ordinary].mean())
        result["summary"]["ordinaryMeanSquaredError"] = float(mse_values[ordinary].mean())
        result["summary"]["ordinaryRelativeKlReductionVsBaseline"] = float(
            (baseline_kl[ordinary].mean() - values[ordinary].mean()) / baseline_kl[ordinary].mean()
        )

    residual = residual_diagnostics(
        full_target,
        candidates["six_parameter"][0],
        actions,
        currents,
        support,
    )
    base_shapes = base_shape_diagnostics(base, actions)
    dominant = min(
        (name for name in results if name != "six_parameter"),
        key=lambda name: results[name]["summary"]["meanKlDivergence"],
    )
    representatives = select_representatives(
        metadata["cases"],
        baseline_kl,
        np.asarray(results[dominant]["perExample"]["klDivergence"]),
        residual["actionOnlyFraction"],
    )
    curves = representative_curves(
        representatives,
        metadata["cases"],
        full_target,
        candidates,
        dominant,
        actions,
        currents,
        support,
    )
    dominant_extra = candidates[dominant][1]
    dominant_parameter_summary: dict[str, Any] | None = None
    if dominant == "third_hinge" and dominant_extra is not None:
        location = config.latent_lower + (config.latent_upper - config.latent_lower) \
            * torch.sigmoid(dominant_extra[:, 0])
        coefficient = dominant_extra[:, 1]
        dominant_parameter_summary = {
            "medianLocation": float(location.median()),
            "p10Location": float(torch.quantile(location, 0.1)),
            "p90Location": float(torch.quantile(location, 0.9)),
            "visibleLocationFraction": float((
                (location >= config.metric_visible_lower)
                & (location <= config.metric_visible_upper)
            ).float().mean()),
            "medianAbsoluteCoefficient": float(coefficient.abs().median()),
        }
    case_rows = []
    for index, source in enumerate(metadata["cases"]):
        row = dict(source)
        row.update({
            "analysisIndex": index,
            "baselineKlDivergence": float(baseline_kl[index]),
            "baselineMeanSquaredError": baseline["perExample"]["meanSquaredError"][index],
            "dominantVariantKlDivergence": results[dominant]["perExample"]["klDivergence"][index],
            "actionOnlyResidualFraction": residual["actionOnlyFraction"][index],
            "baseModeCount": base_shapes["modeCount"][index],
            "baseBoundaryMass": base_shapes["boundaryMass"][index],
        })
        case_rows.append(row)
    output = {
        "version": 1,
        "source": str(args.input),
        "sample": metadata["selection"],
        "queueSummary": metadata["queueSummary"],
        "thresholds": {
            "maxMeanKlDivergence": config.max_mean_kl,
            "maxMeanSquaredError": config.max_mean_mse,
        },
        "metricSupport": {
            "visibleLower": config.metric_visible_lower,
            "visibleUpper": config.metric_visible_upper,
            "actionCells": metric_actions.numel(),
            "currentCells": metric_currents.numel(),
        },
        "dominantVariant": dominant,
        "dominantVariantParameters": dominant_parameter_summary,
        "variants": {name: result["summary"] for name, result in results.items()},
        "residualSummary": residual["summary"],
        "baseShapeSummary": base_shapes["summary"],
        "cases": case_rows,
        "representatives": curves,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "event": "analysis-complete",
        "output": str(args.output),
        "dominantVariant": dominant,
        "baseline": results["six_parameter"]["summary"],
        "best": results[dominant]["summary"],
    }), flush=True)


def optimize_variant(
    name: str,
    baseline_raw: Tensor,
    target: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
    extra_count: int,
    starts: list[float | None],
    steps: int,
) -> tuple[Tensor, Tensor]:
    batch = baseline_raw.shape[0]
    restart_count = len(starts)
    raw = baseline_raw[:, None, :].expand(-1, restart_count, -1).clone().detach().requires_grad_(True)
    extra = torch.zeros((batch, restart_count, extra_count), device=raw.device).requires_grad_(True)
    if name == "third_hinge":
        with torch.no_grad():
            for restart, location in enumerate(starts):
                fraction = (float(location) - support.latent_lower) / (
                    support.latent_upper - support.latent_lower
                )
                extra[:, restart, 0] = math.log(fraction / (1 - fraction))
                extra[:, restart, 1] = 0.01
    optimizer = torch.optim.Adam([
        {"params": [raw], "lr": 0.012},
        {"params": [extra], "lr": 0.06},
    ], betas=(0.9, 0.995))
    best_loss = torch.full((batch, restart_count), math.inf, device=raw.device)
    best_raw = raw.detach().clone()
    best_extra = extra.detach().clone()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = extended_logits(name, raw, extra, actions, currents, support)
        loss_per = -(target[:, None] * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean(dim=-1)
        loss = loss_per.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw, extra], 100.0)
        optimizer.step()
        with torch.no_grad():
            raw[..., :2].clamp_(-14, 14)
            raw[..., 2:6].clamp_(-1e4, 1e4)
            extra.clamp_(-1e4, 1e4)
            improved = loss_per < best_loss
            best_loss = torch.where(improved, loss_per, best_loss)
            best_raw = torch.where(improved[..., None], raw.detach(), best_raw)
            best_extra = torch.where(improved[..., None], extra.detach(), best_extra)
        progress = step / max(1, steps - 1)
        optimizer.param_groups[0]["lr"] = 0.012 * (0.08 + 0.92 * 0.5 * (1 + math.cos(math.pi * progress)))
        optimizer.param_groups[1]["lr"] = 0.06 * (0.08 + 0.92 * 0.5 * (1 + math.cos(math.pi * progress)))
    selected = best_loss.argmin(dim=1)
    rows = torch.arange(batch, device=raw.device)
    return best_raw[rows, selected], best_extra[rows, selected]


def extended_logits(
    name: str,
    raw: Tensor,
    extra: Tensor | None,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> Tensor:
    batch, restarts, _ = raw.shape
    flat_raw = raw.reshape(batch * restarts, 6)
    current_rows = currents.view(1, -1).expand(batch * restarts, -1)
    rows = flat_raw[:, None, :].expand(-1, currents.numel(), -1)
    logits = conditional_policy_logits(rows, actions, current_rows, support).reshape(
        batch, restarts, currents.numel(), actions.numel()
    )
    if name == "six_parameter" or extra is None:
        return logits
    visible_span = support.visible_upper - support.visible_lower
    half_span = visible_span / 2
    center = (support.visible_lower + support.visible_upper) / 2
    action = actions.view(1, 1, 1, -1)
    current = currents.view(1, 1, -1, 1)
    z = (action - center) / half_span
    current_z = (current - center) / half_span
    if name == "cubic":
        return logits + extra[..., 0, None, None] * z.pow(3)
    if name == "split_cubic_tails":
        return logits + extra[..., 0, None, None] * (-z).clamp_min(0).pow(3) \
            + extra[..., 1, None, None] * z.clamp_min(0).pow(3)
    if name == "fee_quadratic_sides":
        relative = (action - current) / visible_span
        return logits + extra[..., 0, None, None] * (-relative).clamp_min(0).square() \
            + extra[..., 1, None, None] * relative.clamp_min(0).square()
    if name == "action_current_interaction":
        return logits + extra[..., 0, None, None] * z * current_z
    moving = scaled_softplus(
        action - current,
        torch.as_tensor(678.0 / visible_span, device=raw.device),
    )
    if name == "learn_beta_x":
        return logits + extra[..., 0, None, None] / visible_span * moving
    if name == "learn_kappa_x":
        fixed_beta_x = -(
            support.friction / (1 - support.friction) + support.friction
        ) / support.temperature
        learned = (678.0 / visible_span) * torch.exp(extra[..., 0, None, None].clamp(-3, 3))
        learned_feature = scaled_softplus(action - current, learned)
        return logits + fixed_beta_x * (learned_feature - moving)
    if name == "learn_kappa_c":
        c1, c2, beta_c1, beta_c2 = decoded_hinges(raw, support)
        fixed = torch.as_tensor(82.0 / visible_span, device=raw.device)
        kappa1 = fixed * torch.exp(extra[..., 0, None, None].clamp(-3, 3))
        kappa2 = fixed * torch.exp(extra[..., 1, None, None].clamp(-3, 3))
        old1 = scaled_softplus(action - c1[..., None, None], fixed)
        old2 = scaled_softplus(action - c2[..., None, None], fixed)
        new1 = scaled_softplus(action - c1[..., None, None], kappa1)
        new2 = scaled_softplus(action - c2[..., None, None], kappa2)
        return logits + beta_c1[..., None, None] / visible_span * (new1 - old1) \
            + beta_c2[..., None, None] / visible_span * (new2 - old2)
    if name == "third_hinge":
        fraction = torch.sigmoid(extra[..., 0])
        c3 = support.latent_lower + (support.latent_upper - support.latent_lower) * fraction
        beta3 = extra[..., 1] / visible_span
        feature = scaled_softplus(
            action - c3[..., None, None],
            torch.as_tensor(82.0 / visible_span, device=raw.device),
        )
        return logits + beta3[..., None, None] * feature
    raise ValueError(name)


def decoded_hinges(raw: Tensor, support: PolicySupport) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    first = torch.sigmoid(raw[..., 0])
    c1 = support.latent_lower + (support.latent_upper - support.latent_lower) * first
    second = torch.sigmoid(raw[..., 1])
    c2 = c1 + (support.latent_upper - c1) * second
    return c1, c2, raw[..., 4], raw[..., 5]


@torch.inference_mode()
def evaluate(
    name: str,
    raw: Tensor,
    extra: Tensor | None,
    target: Tensor,
    entropy: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> dict[str, Any]:
    logits = extended_logits(name, raw[:, None], None if extra is None else extra[:, None], actions, currents, support)[:, 0]
    log_probability = torch.log_softmax(logits, dim=-1)
    probability = log_probability.exp()
    cross_entropy = -(target * log_probability).sum(dim=-1).mean(dim=-1)
    kl = (cross_entropy - entropy).clamp_min(0)
    mse = (target - probability).square().mean(dim=(-1, -2))
    return {
        "summary": summarize_metrics(kl, mse),
        "perExample": {
            "crossEntropy": cross_entropy.cpu().tolist(),
            "klDivergence": kl.cpu().tolist(),
            "meanSquaredError": mse.cpu().tolist(),
        },
    }


def summarize_metrics(kl: Tensor, mse: Tensor) -> dict[str, float]:
    return {
        "meanKlDivergence": float(kl.mean()),
        "medianKlDivergence": float(kl.median()),
        "p90KlDivergence": float(torch.quantile(kl, 0.9)),
        "meanSquaredError": float(mse.mean()),
        "medianSquaredError": float(mse.median()),
        "p90SquaredError": float(torch.quantile(mse, 0.9)),
        "acceptedFraction": float(((kl <= 0.003) & (mse <= 3e-6)).float().mean()),
    }


@torch.inference_mode()
def residual_diagnostics(
    target: Tensor,
    raw: Tensor,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> dict[str, Any]:
    logits = extended_logits("six_parameter", raw[:, None], None, actions, currents, support)[:, 0]
    predicted_log = torch.log_softmax(logits, dim=-1)
    target_log = target.clamp_min(1e-30).log()
    residual = target_log - predicted_log
    residual -= (target * residual).sum(dim=-1, keepdim=True)
    total = (target * residual.square()).sum(dim=(-1, -2)).clamp_min(1e-20)
    action_component = (target * residual).sum(dim=1) / target.sum(dim=1).clamp_min(1e-20)
    reconstructed = action_component[:, None, :]
    reconstructed = reconstructed - (target * reconstructed).sum(dim=-1, keepdim=True)
    action_energy = (target * reconstructed.square()).sum(dim=(-1, -2))
    fraction = (action_energy / total).clamp(0, 1)
    values = fraction.cpu().tolist()
    return {
        "actionOnlyFraction": values,
        "summary": {
            "meanActionOnlyResidualFraction": float(fraction.mean()),
            "medianActionOnlyResidualFraction": float(fraction.median()),
            "p10ActionOnlyResidualFraction": float(torch.quantile(fraction, 0.1)),
            "p90ActionOnlyResidualFraction": float(torch.quantile(fraction, 0.9)),
        },
    }


@torch.inference_mode()
def base_shape_diagnostics(base: Tensor, actions: Tensor) -> dict[str, Any]:
    peak = base.amax(dim=-1, keepdim=True)
    local = (base[:, 1:-1] > base[:, :-2]) & (base[:, 1:-1] >= base[:, 2:]) \
        & (base[:, 1:-1] >= peak * 0.05)
    modes = local.sum(dim=-1)
    modes += (base[:, 0] >= base[:, 1]) & (base[:, 0] >= peak[:, 0] * 0.05)
    modes += (base[:, -1] > base[:, -2]) & (base[:, -1] >= peak[:, 0] * 0.05)
    edge = max(1, int(round(actions.numel() * 0.1)))
    boundary = base[:, :edge].sum(dim=-1) + base[:, -edge:].sum(dim=-1)
    return {
        "modeCount": modes.cpu().tolist(),
        "boundaryMass": boundary.cpu().tolist(),
        "summary": {
            "oneModeFraction": float((modes == 1).float().mean()),
            "twoOrMoreModesFraction": float((modes >= 2).float().mean()),
            "meanModeCount": float(modes.float().mean()),
            "meanBoundaryMass": float(boundary.mean()),
            "p90BoundaryMass": float(torch.quantile(boundary, 0.9)),
        },
    }


def select_representatives(
    cases: list[dict[str, Any]],
    baseline_kl: np.ndarray,
    candidate_kl: np.ndarray,
    action_fraction: list[float],
) -> list[int]:
    improvement = baseline_kl - candidate_kl
    order = np.argsort(baseline_kl)
    choices = [
        int(order[len(order) // 2]),
        int(order[round(len(order) * 0.9)]),
        int(order[-1]),
        int(np.argmax(improvement)),
        int(np.argmin(np.asarray(action_fraction))),
        int(np.argmax(np.asarray(action_fraction))),
    ]
    unique: list[int] = []
    for index in choices:
        if index not in unique:
            unique.append(index)
    for index in order[::-1]:
        if len(unique) >= 6:
            break
        if int(index) not in unique:
            unique.append(int(index))
    return unique[:6]


@torch.inference_mode()
def representative_curves(
    indexes: list[int],
    cases: list[dict[str, Any]],
    target: Tensor,
    candidates: dict[str, tuple[Tensor, Tensor | None]],
    dominant: str,
    actions: Tensor,
    currents: Tensor,
    support: PolicySupport,
) -> list[dict[str, Any]]:
    desired_currents = [-100, -50, 0, 50, 100]
    current_indexes = [int((currents - value).abs().argmin()) for value in desired_currents]
    output = []
    for index in indexes:
        baseline_raw, _ = candidates["six_parameter"]
        candidate_raw, candidate_extra = candidates[dominant]
        baseline_logits = extended_logits(
            "six_parameter", baseline_raw[index:index + 1, None], None, actions, currents, support
        )[0, 0]
        candidate_logits = extended_logits(
            dominant,
            candidate_raw[index:index + 1, None],
            None if candidate_extra is None else candidate_extra[index:index + 1, None],
            actions,
            currents,
            support,
        )[0, 0]
        baseline_probability = torch.softmax(baseline_logits, dim=-1)
        candidate_probability = torch.softmax(candidate_logits, dim=-1)
        source = cases[index]
        output.append({
            "analysisIndex": index,
            "date": source["date"],
            "time": source["time"],
            "price": source["price"],
            "return1h": source["return1h"],
            "queueKlDivergence": source["klDivergence"],
            "currents": [float(currents[item]) for item in current_indexes],
            "actionGrid": [round(float(value), 5) for value in actions.cpu()],
            "oracle": [rounded(target[index, item]) for item in current_indexes],
            "baseline": [rounded(baseline_probability[item]) for item in current_indexes],
            "candidate": [rounded(candidate_probability[item]) for item in current_indexes],
        })
    return output


def rounded(values: Tensor) -> list[float]:
    return [round(float(value), 8) for value in values.cpu()]


if __name__ == "__main__":
    main()
