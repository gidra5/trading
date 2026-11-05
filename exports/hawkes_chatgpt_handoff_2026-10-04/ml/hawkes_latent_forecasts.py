"""Causal hidden-state forecasts: exact additive means and nonlinear rollouts.

Origins snapshot the filtering law before reading the current observation.
Neither forecast recursion takes future observations as an argument. Numerical
runaway invalidates a horizon; failed paths are never dropped from its mean.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.special import gammaln

from hawkes_latent import LatentModel, filter_model, physical_arrays
from hawkes_research import HORIZONS, count_metrics, load_corpus, save_json


def linear_operators(model, horizons=HORIZONS):
    if any(value != 0 for row in model.beta for value in row) or model.diffusions:
        raise ValueError('Exact linear means require no nonlinear anchors or diffusion')
    k = len(model.modes)
    decay = np.exp(-1 / np.array([m.tau for m in model.modes]))
    coefficient = np.zeros((3, k))
    for j, mode in enumerate(model.modes):
        coefficient[mode.target, j] = mode.mass
    transition = np.zeros((k + 3, k + 3))
    for j, mode in enumerate(model.modes):
        transition[j, :k] = (1 - decay[j]) * coefficient[mode.source]
        transition[j, j] += decay[j]
        transition[j, k] = (1 - decay[j]) * model.background[mode.source]
    transition[k, k] = 1.
    transition[k + 1:, :k] = coefficient[:2]
    transition[k + 1:, k] = model.background[:2]
    transition[k + 1:, k + 1:] = np.eye(2)
    return np.stack([np.linalg.matrix_power(transition, int(h))[k + 1:, :k + 1] for h in horizons])


def exact_means(model, snapshots, horizons=HORIZONS):
    if any(value != 0 for row in model.beta for value in row):
        raise ValueError('Nonlinear amplitude modulation requires future simulation')
    fast, slopes, _, _, pending, previous_z = snapshots
    expected_states = np.mean(fast + slopes * pending[:, :, None], axis=1)
    if model.diffusions:
        return exact_diffusion_means(np.array(model.background), np.array([m.target for m in model.modes], dtype=np.int64),
                                     np.array([m.source for m in model.modes], dtype=np.int64), np.array([m.mass for m in model.modes]),
                                     np.exp(-1 / np.array([m.tau for m in model.modes])),
                                     np.exp(-1 / np.array([z.tau for z in model.diffusions])),
                                     np.array([z.loading for z in model.diffusions]), expected_states, previous_z,
                                     np.asarray(horizons, dtype=np.int64))
    state = np.c_[expected_states, np.ones(len(expected_states))]
    return np.einsum('hsk,ok->ohs', linear_operators(model, horizons), state)


@njit(cache=True, parallel=True)
def exact_diffusion_means(background, target, source, mass, decay, z_decay, z_loading, expected_states, previous_z, horizons):
    """Integrate future Gaussian immigration exactly, then propagate linear means.

The conditioning law is the empirical filtering posterior at the origin. Future
OU innovations are integrated analytically, including their lognormal moment
correction. Correlation between immigration and earlier events does not prevent
closure of the additive Hawkes first-moment recursion.
"""
    output = np.zeros((len(expected_states), len(horizons), 2))
    particles, q_count = previous_z.shape[1], len(z_decay)
    for origin in prange(len(expected_states)):
        state = expected_states[origin].copy()
        multipliers = z_decay.copy()
        cumulative = np.zeros(2)
        h = 0
        for t in range(1, horizons[-1] + 1):
            rate = background.copy()
            for side in range(3):
                correction = 0.
                for q in range(q_count):
                    correction += (z_loading[q, side] * multipliers[q]) ** 2
                moment = 0.
                for p in range(particles):
                    exponent = -.5 * correction
                    for q in range(q_count):
                        exponent += z_loading[q, side] * multipliers[q] * previous_z[origin, p, q]
                    moment += np.exp(exponent)
                rate[side] *= moment / particles
            for j in range(len(target)):
                rate[target[j]] += mass[j] * state[j]
            cumulative += rate[:2]
            for j in range(len(target)):
                state[j] = decay[j] * state[j] + (1 - decay[j]) * rate[source[j]]
            if t == horizons[h]:
                output[origin, h] = cumulative
                h += 1
                if h == len(horizons):
                    break
            multipliers *= z_decay
    return output


@njit(cache=True, parallel=True)
def generative_rollout(background, target, source, mass, decay, anchor_source, slow_decay, beta_gate, initial_mean, saturating,
                       z_decay, z_loading, snapshots, horizons, paths, seed, origins):
    fast, slopes, slow, slow_slopes, pending, z_prior = snapshots
    n, k, a, q_count = len(fast), len(target), len(anchor_source), len(z_decay)
    mean = np.zeros((n, len(horizons), 2))
    mc_error = np.zeros_like(mean)
    failure = np.zeros((n, len(horizons)))
    samples = np.zeros((n, paths, len(horizons), 2))
    z_sd = np.sqrt(1 - z_decay ** 2)
    z_variance = np.sum(z_loading ** 2, axis=0)
    for origin in prange(n):
        np.random.seed(seed + int(origins[origin]))
        states = np.zeros((paths, k))
        slow_states = np.zeros((paths, a))
        z_states = np.zeros((paths, q_count))
        for p in range(paths):
            ancestor = np.random.randint(0, fast.shape[1])
            previous_l = np.random.poisson(pending[origin, ancestor])
            states[p] = fast[origin, ancestor] + slopes[origin, ancestor] * previous_l
            slow_states[p] = slow[origin, ancestor] + slow_slopes[origin, ancestor] * previous_l
            z_states[p] = z_prior[origin, ancestor]
        intensity_total = np.zeros((paths, 2))
        count_total = np.zeros((paths, 2))
        failed = np.zeros(paths, dtype=np.bool_)
        h = 0
        for step in range(1, int(horizons[-1]) + 1):
            for p in range(paths):
                if failed[p]:
                    continue
                for q in range(q_count):
                    z_states[p, q] = z_decay[q] * z_states[p, q] + z_sd[q] * np.random.normal()
                rate = background.copy()
                for side in range(3):
                    exponent = -.5 * z_variance[side]
                    for q in range(q_count):
                        exponent += z_loading[q, side] * z_states[p, q]
                    rate[side] *= np.exp(exponent)
                for j in range(k):
                    rate[target[j]] += mass[j] * states[p, j]
                if not np.isfinite(rate).all() or np.max(rate) > 1e12:
                    failed[p] = True
                    continue
                event = np.zeros(3)
                for side in range(3):
                    event[side] = np.random.poisson(rate[side])
                for side in range(2):
                    intensity_total[p, side] += rate[side]
                    count_total[p, side] += event[side]
                normalized = np.zeros(a)
                for q in range(a):
                    value = slow_states[p, q] / max(initial_mean[anchor_source[q]], 1e-8)
                    normalized[q] = np.tanh(value) if saturating else value
                for j in range(k):
                    factor = 1.
                    for q in range(a):
                        factor += beta_gate[j, q] * normalized[q]
                    states[p, j] = decay[j] * states[p, j] + (1 - decay[j]) * factor * event[source[j]]
                for q in range(a):
                    slow_states[p, q] = slow_decay[q] * slow_states[p, q] + (1 - slow_decay[q]) * event[anchor_source[q]]
            if step == horizons[h]:
                failure[origin, h] = np.mean(failed)
                for side in range(2):
                    if np.any(failed):
                        mean[origin, h, side], mc_error[origin, h, side] = np.nan, np.nan
                        samples[origin, :, h, side] = np.nan
                    else:
                        value = np.mean(intensity_total[:, side])
                        mean[origin, h, side] = value
                        mc_error[origin, h, side] = np.sqrt(np.mean((intensity_total[:, side] - value) ** 2) / paths)
                        samples[origin, :, h, side] = count_total[:, side]
                h += 1
                if h == len(horizons):
                    break
    return mean, mc_error, failure, samples


def rollout(model, snapshots, observed_mean, paths=512, horizons=HORIZONS, seed=20261003, origins=None):
    if paths <= 1:
        raise ValueError('At least two Monte Carlo paths are required')
    if origins is None:
        origins = np.arange(len(snapshots[0]), dtype=np.int64)
    z_decay = np.exp(-1 / np.array([z.tau for z in model.diffusions]))
    z_loading = np.array([z.loading for z in model.diffusions]).reshape(-1, 3)
    return generative_rollout(*physical_arrays(model, observed_mean), z_decay, z_loading, snapshots,
                              np.asarray(horizons, dtype=np.int64), paths, seed, np.asarray(origins, dtype=np.int64))


def evaluate_forecasts(model, corpus, output, label='ML', filter_particles=256, paths=512, stride=900, split='validation', filter_seed=71391):
    output.mkdir(parents=True, exist_ok=True)
    from hawkes_research import evaluation_region, forecast_origins, safe_corr
    _, end, mask = evaluation_region(corpus, split)
    save_json(output / 'snapshot-model.json', {'model': model.serialize(), 'corpus': corpus.identity,
              'split': split, 'test_evaluated': split == 'test'})
    origins = forecast_origins(corpus, split, stride, max(HORIZONS))
    result = filter_model(model, corpus.y[:end], corpus.valid[:end], corpus.initial_mean, filter_particles, filter_seed, origins)
    if not np.isfinite(result[1][mask]).all():
        raise ValueError('Numerical filtering failure: no valid forecast comparison')
    one_step = count_metrics(corpus.y[:end][mask], result[0][mask])
    one_step['predictive_marginal_nll'] = float(np.mean(result[1][mask] + gammaln(corpus.y[:end][mask] + 1).sum(axis=1)))
    np.savez_compressed(output / f'{split}_predictive_loss.npz', seconds=np.flatnonzero(mask),
                        marginal_nll=result[1][mask] + gammaln(corpus.y[:end][mask] + 1).sum(axis=1))
    one_step['poisson_mean_log_score_note'] = 'The other score is Poisson at the predictive mean; marginal NLL integrates latent states'
    save_json(output / f'{split}_one_step_metrics.json', one_step)
    times = np.flatnonzero(mask)
    previous_counts = corpus.y[times - 1]
    inferred_previous = result[4][mask]
    save_json(output / f'{split}_latent_diagnostics.json', {
              'model': label, 'split': split,
              'filtered_current_latent_intensity_mean': float(np.mean(result[3][mask])),
              'previous_latent_count_posterior_mean': float(np.mean(inferred_previous)),
              'previous_latent_active_probability_mean': float(np.mean(result[5][mask])),
              'fraction_bins_previous_latent_active_probability_gt_0_01': float(np.mean(result[5][mask] > .01)),
              'previous_latent_count_correlation_previous_buy': safe_corr(inferred_previous, previous_counts[:, 0]),
              'previous_latent_count_correlation_previous_sell': safe_corr(inferred_previous, previous_counts[:, 1]),
              'previous_latent_count_correlation_current_buy': safe_corr(inferred_previous, corpus.y[times, 0]),
              'previous_latent_count_correlation_current_sell': safe_corr(inferred_previous, corpus.y[times, 1]),
              'interpretation': 'Filtered hidden intensities and preceding hidden counts use observations through the current bin; one-step B/S forecasts precede those observations',
              'physical_latent_identifiability_established': False})
    np.savez_compressed(output / f'{split}_latent_trace.npz', seconds=times,
                        filtered_current_latent_intensity=result[3][mask],
                        previous_latent_count_posterior_mean=inferred_previous,
                        previous_latent_active_probability=result[5][mask])
    snapshots = result[6]
    np.savez_compressed(output / f'{split}_origin_latent_states.npz', origins=origins,
                        previous_diffusion_posterior_mean=np.mean(snapshots[-1], axis=1))
    prediction = np.zeros((len(origins), len(HORIZONS), 2))
    errors = np.zeros_like(prediction)
    failure = np.zeros((len(origins), len(HORIZONS)))
    samples = np.zeros((len(origins), paths, len(HORIZONS), 2))
    for start in range(0, len(origins), 16):
        stop = min(start + 16, len(origins))
        prediction[start:stop], errors[start:stop], failure[start:stop], samples[start:stop] = rollout(
            model, tuple(x[start:stop] for x in snapshots), corpus.initial_mean, paths, origins=origins[start:stop])
        print(f'{label} forecasts {stop}/{len(origins)} origins', flush=True)
    additive = not any(value != 0 for row in model.beta for value in row)
    if additive:
        exact = exact_means(model, snapshots)
        # Preserve failures even where an algebraic finite mean exists. This
        # avoids concealing simulated numerical/stability problems.
        prediction[failure == 0] = exact[failure == 0]
    cumulative = np.vstack((np.zeros(2), np.cumsum(corpus.y[:end], axis=0)))
    actual = np.stack([cumulative[origins + h] - cumulative[origins] for h in HORIZONS], axis=1)
    total_interval = np.quantile(samples.sum(axis=-1), [.05, .5, .95], axis=1).transpose(1, 2, 0)
    rows = []
    for h, horizon in enumerate(HORIZONS):
        available = np.isfinite(total_interval[:, h]).all(axis=1)
        rows.append({'model': label, 'split': split, 'horizon_seconds': horizon, 'origins': len(origins),
                     **count_metrics(actual[:, h], prediction[:, h]),
                     'mc_standard_error_mean': float(np.mean(errors[:, h])) if np.isfinite(errors[:, h]).all() else None,
                     'failed_origin_fraction': float(np.mean(failure[:, h] > 0)),
                     'divergent_path_fraction': float(np.mean(failure[:, h])),
                     'interval90_coverage_available': float(np.mean((actual[available, h].sum(axis=1) >= total_interval[available, h, 0]) &
                                                                 (actual[available, h].sum(axis=1) <= total_interval[available, h, 2]))) if available.any() else None})
    with (output / f'{split}_horizon_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / f'{split}_forecasts.npz', origins=origins, prediction=prediction, actual=actual,
                        horizons=np.array(HORIZONS), mc_standard_error=errors, failure_fraction=failure, total_count_quantile90=total_interval)
    save_json(output / 'forecast-protocol.json', {'filter_particles': filter_particles, 'filter_seed': filter_seed, 'future_paths': paths, 'stride_seconds': stride,
              'model': model.serialize(), 'corpus': corpus.identity,
              'mean': 'Exact additive Hawkes first moment with analytically integrated OU immigration' if additive else 'Integrated future intensities from nonlinear generative paths',
              'mc_standard_error_note': 'Simulation diagnostic; additive-model point forecasts use exact conditional moments',
              'interval': '5/50/95% simulated total count quantiles; fixed parameters, excluding parameter uncertainty',
              'horizon_log_score': 'Poisson mean score, not exact marginal multi-step likelihood', 'split': split, 'test_evaluated': split == 'test'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='ML')
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--particles', type=int, default=256)
    parser.add_argument('--paths', type=int, default=512)
    parser.add_argument('--seed', type=int, default=71391)
    args = parser.parse_args()
    model = LatentModel.deserialize(json.loads(args.model.read_text())['model'])
    evaluate_forecasts(model, load_corpus(args.cache, 3600), args.output, args.label, args.particles, args.paths, filter_seed=args.seed)


if __name__ == '__main__':
    main()
