"""Causal observed-state filtering and Monte Carlo open-loop M2 forecasts."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from numba import njit, prange
from hawkes_research import HORIZONS, count_metrics, evaluation_region, forecast_origins, save_json
from hawkes_modulation import gate_values


@njit(cache=True)
def observed_states(theta, target, source, anchor_source, y, valid, initial_mean, origins, saturating=False):
    k, a = len(target), len(anchor_source)
    d = np.exp(-1 / np.exp(theta[2 + k:2 + 2 * k]))
    slow_start = 2 + 2 * k + k * a
    ds = np.exp(-1 / np.exp(theta[slow_start:]))
    beta = theta[2 + 2 * k:slow_start].reshape(k, a)
    gates = gate_values(theta[2 + k:2 + 2 * k], theta[slow_start:])
    state = np.zeros(k)
    slow = np.zeros(a)
    for j in range(k):
        state[j] = initial_mean[source[j]]
    for p in range(a):
        slow[p] = initial_mean[anchor_source[p]]
    prediction = np.zeros_like(y)
    fast_origins = np.zeros((len(origins), k))
    slow_origins = np.zeros((len(origins), a))
    pointer = 0
    for t in range(len(y)):
        if not valid[t]:
            for j in range(k):
                state[j] = initial_mean[source[j]]
            for p in range(a):
                slow[p] = initial_mean[anchor_source[p]]
            continue
        rate = theta[:2].copy()
        for j in range(k):
            rate[target[j]] += theta[2 + j] * state[j]
        prediction[t] = rate
        if pointer < len(origins) and t == origins[pointer]:
            fast_origins[pointer] = state
            slow_origins[pointer] = slow
            pointer += 1
        bounded = np.zeros(a)
        for p in range(a):
            raw = slow[p] / max(initial_mean[anchor_source[p]], 1e-8)
            bounded[p] = np.tanh(raw) if saturating else raw
        for j in range(k):
            factor = 1.
            for p in range(a):
                factor += beta[j, p] * gates[j, p] * bounded[p]
            state[j] = d[j] * state[j] + (1 - d[j]) * y[t, source[j]] * factor
        for p in range(a):
            slow[p] = ds[p] * slow[p] + (1 - ds[p]) * y[t, anchor_source[p]]
    return prediction, fast_origins, slow_origins


@njit(cache=True, parallel=True)
def rollout(theta, target, source, anchor_source, initial_mean, fast_origins, slow_origins, horizons, particles, seed, origin_ids, saturating=False):
    k, a = len(target), len(anchor_source)
    d = np.exp(-1 / np.exp(theta[2 + k:2 + 2 * k]))
    slow_start = 2 + 2 * k + k * a
    ds = np.exp(-1 / np.exp(theta[slow_start:]))
    beta = theta[2 + 2 * k:slow_start].reshape(k, a)
    gates = gate_values(theta[2 + k:2 + 2 * k], theta[slow_start:])
    output = np.zeros((len(fast_origins), len(horizons), 2))
    standard_error = np.zeros_like(output)
    divergent_fraction = np.zeros((len(fast_origins), len(horizons)))
    for origin in prange(len(fast_origins)):
        np.random.seed(seed + origin_ids[origin])
        states = np.zeros((particles, k))
        slow_states = np.zeros((particles, a))
        for p in range(particles):
            states[p] = fast_origins[origin]
            slow_states[p] = slow_origins[origin]
        totals = np.zeros((particles, 2))
        divergent = np.zeros(particles, dtype=np.bool_)
        h = 0
        for t in range(1, horizons[-1] + 1):
            for p in range(particles):
                if divergent[p]:
                    continue
                rate = theta[:2].copy()
                for j in range(k):
                    rate[target[j]] += theta[2 + j] * states[p, j]
                if not np.isfinite(rate).all() or np.max(rate) > 1e12:
                    divergent[p] = True
                    continue
                # Integrate future conditional intensities to estimate E[count].
                totals[p] += rate
                event = np.zeros(2)
                for side in range(2):
                    event[side] = np.random.poisson(rate[side])
                bounded = np.zeros(a)
                for q in range(a):
                    raw = slow_states[p, q] / max(initial_mean[anchor_source[q]], 1e-8)
                    bounded[q] = np.tanh(raw) if saturating else raw
                for j in range(k):
                    factor = 1.
                    for q in range(a):
                        factor += beta[j, q] * gates[j, q] * bounded[q]
                    states[p, j] = d[j] * states[p, j] + (1 - d[j]) * event[source[j]] * factor
                for q in range(a):
                    slow_states[p, q] = ds[q] * slow_states[p, q] + (1 - ds[q]) * event[anchor_source[q]]
            if t == horizons[h]:
                divergent_fraction[origin, h] = divergent.sum() / particles
                for side in range(2):
                    if divergent.any():
                        # Do not discard failed paths and report a biased finite mean.
                        output[origin, h, side] = np.nan
                        standard_error[origin, h, side] = np.nan
                        continue
                    mean = np.mean(totals[:, side])
                    variance = np.mean((totals[:, side] - mean) ** 2)
                    output[origin, h, side] = mean
                    standard_error[origin, h, side] = np.sqrt(variance / particles)
                h += 1
                if h == len(horizons):
                    break
    return output, standard_error, divergent_fraction


def evaluate_forecasts(model, corpus, output: Path, particles=512, stride=900, split='validation'):
    output.mkdir(parents=True, exist_ok=True)
    _, end, mask = evaluation_region(corpus, split)
    origins = forecast_origins(corpus, split, stride, max(HORIZONS))
    theta, target, source, anchor_source = model.arrays()
    one_step, states, slow_states = observed_states(theta, target, source, anchor_source, corpus.y[:end], corpus.valid[:end], corpus.initial_mean, origins, model.link == 'bounded')
    one_step_metrics = count_metrics(corpus.y[:end][mask], one_step[mask])
    scored_y, scored_rate = corpus.y[:end][mask], one_step[mask]
    loss = (scored_rate - scored_y * np.log(np.maximum(scored_rate, 1e-12)) + gammaln(scored_y + 1)).sum(axis=1)
    np.savez_compressed(output / f'{split}_predictive_loss.npz', seconds=np.flatnonzero(mask), marginal_nll=loss)
    predictions = np.zeros((len(origins), len(HORIZONS), 2))
    errors = np.zeros_like(predictions)
    divergence = np.zeros((len(origins), len(HORIZONS)))
    for start in range(0, len(origins), 16):
        stop = min(start + 16, len(origins))
        predictions[start:stop], errors[start:stop], divergence[start:stop] = rollout(theta, target, source, anchor_source, corpus.initial_mean, states[start:stop], slow_states[start:stop], np.array(HORIZONS), particles, 20261003, origins[start:stop], model.link == 'bounded')
        print(f'M2 forecasts {stop}/{len(origins)} origins', flush=True)
    cumulative = np.vstack((np.zeros(2), np.cumsum(corpus.y[:end], axis=0)))
    rows, targets = [], []
    for j, horizon in enumerate(HORIZONS):
        actual = cumulative[origins + horizon] - cumulative[origins]
        targets.append(actual)
        rows.append({'model': 'M2', 'split': split, 'horizon_seconds': horizon, 'origins': len(origins),
                     **count_metrics(actual, predictions[:, j]),
                     'mc_standard_error_buy_mean': float(errors[:, j, 0].mean()) if np.isfinite(errors[:, j, 0]).all() else None,
                     'mc_standard_error_sell_mean': float(errors[:, j, 1].mean()) if np.isfinite(errors[:, j, 1]).all() else None,
                     'failed_origin_fraction': float((divergence[:, j] > 0).mean()),
                     'divergent_particle_fraction': float(divergence[:, j].mean())})
    with (output / f'{split}_horizon_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / f'{split}_forecasts.npz', origins=origins, prediction=predictions, mc_standard_error=errors,
                        actual=np.stack(targets, axis=1), horizons=np.array(HORIZONS), divergent_particle_fraction=divergence)
    save_json(output / f'{split}_one_step_metrics.json', one_step_metrics)
    save_json(output / 'forecast-protocol.json', {'stride_seconds': stride, 'particles': particles,
              'seed': 20261003, 'mean': 'Generative Poisson future paths; integrated conditional intensities',
              'horizon_log_score': 'Poisson mean score, not exact marginal multi-step likelihood',
              'model': model.serialize(), 'corpus': corpus.identity, 'split': split, 'test_evaluated': split == 'test'})
    save_json(output / 'rollout-diagnostics.json', {'numerical_runaway_threshold_count_per_second': 1e12,
              'failed_origins': int((divergence[:, -1] > 0).sum()), 'origins': len(origins),
              'failed_paths_policy': 'Mark affected forecast means unavailable; never condition on surviving paths'})
