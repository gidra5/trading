"""Fixed-width observable-state quantization for the learned NSF market kernel.

Each parent still has a conditional row of sample probabilities. We never pool
those rows into an unconditional price law. Representatives are sampled pairs
of observations, not latent/noise identifiers. Account states are NOT merged.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from export_nsf_planning_trees import next_observations


@dataclass(frozen=True)
class MarketLayer:
    features: np.ndarray  # [nodes, two observations, 59 features]
    prices: np.ndarray    # [nodes, two prices]
    times: np.ndarray     # [nodes, two timestamps]
    mass: torch.Tensor   # unconditional mass used only for representative fitting


@dataclass(frozen=True)
class ScenarioGraph:
    layers: tuple[MarketLayer, ...]
    prices: tuple[torch.Tensor, ...]
    weights: tuple[torch.Tensor, ...]
    destinations: tuple[torch.Tensor, ...]
    diagnostics: tuple[dict, ...]
    refinements: tuple[NestedPartition | None, ...]
    sampled_prices: tuple[torch.Tensor, ...]  # unprojected conditional execution marks


@dataclass(frozen=True)
class NestedPartition:
    selected: torch.Tensor
    assignment: torch.Tensor
    coarse_selected: torch.Tensor
    coarse_assignment: torch.Tensor
    fine_to_coarse: torch.Tensor


@torch.inference_mode()
def nested_representatives(coordinates, mass, width, base_width=64):
    """Split coarse cells, retaining their representatives as explicit fallbacks.

    Each split adds a sampled representative and only moves points that become
    strictly closer inside the chosen cell. Thus squared metric error cannot
    increase for any sample. This is a guarantee on THIS fixed candidate set,
    not on a subsequently regenerated rollout or Bellman value.
    """
    if base_width < 1:
        raise ValueError('Positive base width required')
    coarse, coarse_assignment = representatives(coordinates, mass, min(width, base_width))
    selected = list(coarse.unbind())
    assignment = coarse_assignment.clone()
    ancestry = list(torch.arange(len(coarse), device=coordinates.device).unbind())
    error = (coordinates-coordinates[coarse][assignment]).square().sum(1)
    while len(selected) < min(width, len(coordinates)):
        distortion = torch.zeros(len(selected), device=mass.device, dtype=mass.dtype)
        distortion.index_add_(0, assignment, mass*error)
        cell = distortion.argmax()
        members = assignment == cell
        score = torch.where(members, mass*error, -1.)
        candidate = score.argmax()
        if float(score[candidate]) <= 0:
            break
        new_error = (coordinates-coordinates[candidate]).square().sum(1)
        move = members & (new_error < error)
        assignment = torch.where(move, len(selected), assignment)
        error = torch.where(move, new_error, error)
        selected.append(candidate)
        ancestry.append(ancestry[int(cell)])
    return NestedPartition(torch.stack(selected), assignment, coarse, coarse_assignment, torch.stack(ancestry))


@torch.inference_mode()
def representatives(coordinates, mass, width, iterations=5):
    """Weighted Lloyd centers, snapped to sample representatives (medoids).

    All samples are assigned; low-mass clusters are not dropped. Exact duplicate
    coordinates share a representative. Empty/duplicate representatives shrink
    the output below the cap. Deterministic farthest-point initialization avoids
    changing random sampling when we change only the width.
    """
    if coordinates.ndim != 2 or len(coordinates) != len(mass) or width < 1 or iterations < 1:
        raise ValueError('Invalid clustering input')
    if not bool(torch.isfinite(coordinates).all() & torch.isfinite(mass).all() & (mass > 0).all()):
        raise ValueError('Clustering needs finite coordinates and positive mass')
    if len(coordinates) <= width:
        identity = torch.arange(len(coordinates), device=coordinates.device)
        unique, inverse = torch.unique(coordinates, dim=0, return_inverse=True)
        if len(unique) == len(coordinates):
            return identity, identity
        # Even without a capacity constraint, identical observations cannot
        # acquire distinct future laws through an unobservable sample index.
        first = torch.full((len(unique),), len(coordinates), dtype=torch.long, device=coordinates.device)
        first.scatter_reduce_(0, inverse, identity, reduce='amin')
        selected = first.sort().values
        remap = torch.empty_like(first)
        remap[inverse[selected]] = torch.arange(len(selected), device=coordinates.device)
        return selected, remap[inverse]
    mean = (coordinates*mass[:, None]).sum(0)/mass.sum()
    selected = [(coordinates-mean).square().sum(1).argmin()]
    nearest = (coordinates-coordinates[selected[0]]).square().sum(1)
    for _ in range(1, min(width, len(coordinates))):
        idx = (nearest*mass).argmax()
        selected.append(idx)
        nearest = torch.minimum(nearest, (coordinates-coordinates[idx]).square().sum(1))
    centers = coordinates[torch.stack(selected)].clone()
    for _ in range(iterations):
        distance = torch.cdist(coordinates, centers).square()
        assignment = distance.argmin(1)
        counts = torch.zeros(len(centers), device=mass.device, dtype=mass.dtype).index_add_(0, assignment, mass)
        sums = torch.zeros_like(centers).index_add_(0, assignment, mass[:, None]*coordinates)
        centers = torch.where(counts[:, None] > 0, sums/counts[:, None].clamp_min(1e-30), centers)
    # Choose the closest sample in each center's own cell. This preserves the
    # price grid and actual joint feature coordinates of a generated sample.
    distance = torch.cdist(coordinates, centers).square()
    assignment = distance.argmin(1)
    own_cell = assignment[:, None] == torch.arange(len(centers), device=mass.device)[None, :]
    snapped = torch.where(own_cell, distance, torch.inf).argmin(0)
    nonempty = own_cell.any(0)
    selected = torch.unique(snapped[nonempty], sorted=True)
    assignment = torch.cdist(coordinates, coordinates[selected]).argmin(1)
    # Duplicate coordinates can make a snapped representative receive no mass.
    used, assignment = torch.unique(assignment, sorted=True, return_inverse=True)
    return selected[used], assignment


@torch.inference_mode()
def build_graph(model, features, prices, times, offsets, width=64, samples=64, seed=20260916,
                price_resolution_bps=1., feature_weight=1., refinement='independent', base_width=64,on_layer=None):
    if width < 1 or samples < 2 or samples % 2 or not offsets:
        raise ValueError('Positive width and even sample count required')
    if not np.isfinite(price_resolution_bps) or price_resolution_bps <= 0 or not np.isfinite(feature_weight) or feature_weight < 0:
        raise ValueError('Invalid compression metric')
    if refinement not in ('independent', 'nested') or base_width < 1:
        raise ValueError('Invalid refinement method')
    if any(h <= 0 or h > 86400 for h in np.diff([0]+list(offsets))):
        raise ValueError('Invalid time grid')
    device = next(model.parameters()).device
    layers = [MarketLayer(np.asarray(features)[None], np.asarray(prices)[None], np.asarray(times)[None],
                          torch.ones(1, dtype=torch.float64, device=device))]
    edge_prices, weights, destinations, diagnostics, refinements, sampled_prices = [], [], [], [], [], []
    origin_price = float(prices[-1])
    origin_time = float(times[-1])
    for depth, offset in enumerate(offsets):
        previous = layers[-1]
        count = len(previous.prices)
        horizon = np.full(count, origin_time+offset)-previous.times[:, -1]
        future_features, future_prices = next_observations(model, previous.features, previous.prices,
            previous.times, horizon, samples, seed+depth)
        f = np.stack((np.repeat(previous.features[:, -1], samples, axis=0), future_features.reshape(-1, 59)), axis=1)
        p = np.stack((np.repeat(previous.prices[:, -1], samples), future_prices.reshape(-1)), axis=1)
        t = np.stack((np.repeat(previous.times[:, -1], samples), np.full(count*samples, origin_time+offset)), axis=1)
        mass = previous.mass.repeat_interleave(samples)/samples
        standardized = (torch.as_tensor(f, device=device)-model.feature_mean)/model.feature_std
        log_prices = torch.as_tensor(np.log(p/origin_price)/(price_resolution_bps/10000), device=device)
        # Fixed training normalization, not per-layer variance scaling: otherwise
        # a nearly constant coordinate could dominate by numerical noise.
        # Balance the feature block by its RMS rather than giving 118 features
        # 118 times the weight of one price coordinate. One price unit is one
        # basis point by default, NOT the NSF decoder's 100-bps training scale.
        feature_coordinates = standardized.double().reshape(len(f), -1)
        feature_coordinates = feature_coordinates*(feature_weight/np.sqrt(feature_coordinates.shape[1]))
        coordinates = torch.cat((feature_coordinates, log_prices), dim=1)
        refinement_diagnostics = {}
        if refinement == 'nested':
            partition = nested_representatives(coordinates, mass, width, base_width)
            selected, assignment = partition.selected, partition.assignment
            coarse_error = (coordinates-coordinates[partition.coarse_selected][partition.coarse_assignment]).square().sum(1)
            fine_error = (coordinates-coordinates[selected][assignment]).square().sum(1)
            exact_ancestry = torch.equal(partition.fine_to_coarse[assignment], partition.coarse_assignment)
            maximum_increase = float((fine_error-coarse_error).max())
            if not exact_ancestry or maximum_increase > 1e-10:
                raise AssertionError('Nested refinement invariant violated')
            refinement_diagnostics = dict(coarseNodes=len(partition.coarse_selected),
                coarseMetricMSE=float((mass*coarse_error).sum()), fineMetricMSE=float((mass*fine_error).sum()),
                maximumPointwiseMetricIncrease=maximum_increase, exactCoarseRecovery=exact_ancestry)
        else:
            partition = None
            selected, assignment = representatives(coordinates, mass, width)
        refinements.append(partition)
        chosen = selected.cpu().numpy()
        destination = assignment.reshape(count, samples)
        new_mass = torch.zeros(len(selected), device=device, dtype=torch.float64).index_add_(0, assignment, mass)
        representative_price = torch.as_tensor(p[chosen, -1], device=device, dtype=torch.float64)
        projected = representative_price[destination]
        raw = torch.as_tensor(future_prices, device=device, dtype=torch.float64)
        log_error = torch.log(projected/raw)
        parent_price = torch.as_tensor(previous.prices[:, -1, None], device=device, dtype=torch.float64)
        mean_return_shift = ((projected-raw)/parent_price).mean(1)
        rms_coordinates = ((coordinates-coordinates[selected][assignment]).square().sum(1)*mass).sum().sqrt()
        diagnostics.append(dict(depth=depth+1, offsetSeconds=offset, parentNodes=count, generatedSamples=count*samples,
            retainedNodes=len(selected), weightedLogPriceRMSEBps=float((log_error.square().mean(1)*previous.mass).sum().sqrt()*10000),
            worstParentMeanReturnShiftBps=float(mean_return_shift.abs().max()*10000),
            weightedSignChangeProbability=float((((projected-parent_price).sign() != (raw-parent_price).sign()).double().mean(1)*previous.mass).sum()),
            standardizedCoordinateRMSE=float(rms_coordinates), mass=float(new_mass.sum()), **refinement_diagnostics))
        if on_layer is not None:on_layer(diagnostics[-1])
        layers.append(MarketLayer(f[chosen], p[chosen], t[chosen], new_mass))
        edge_prices.append(projected)
        sampled_prices.append(raw)
        weights.append(torch.full((count, samples), 1/samples, device=device, dtype=torch.float64))
        destinations.append(destination)
    return ScenarioGraph(tuple(layers), tuple(edge_prices), tuple(weights), tuple(destinations), tuple(diagnostics), tuple(refinements), tuple(sampled_prices))
