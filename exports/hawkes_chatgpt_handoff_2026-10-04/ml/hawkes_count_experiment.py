"""Reproducible *binned* Hawkes ablation, with a generative latent count channel.

This is a discrete-time approximation, not continuous-time missing-event inference.
Latent counts are sampled from their own B/S/L-driven Poisson transition and
integrated by a bootstrap particle filter. All predictions precede the target bin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize


def write_json(path: Path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare(source: Path, metadata: Path, cache: Path):
    manifest = json.loads(metadata.read_text())
    from datetime import datetime
    first = int(datetime.fromisoformat(manifest['firstTimestampUtc'].replace('Z', '+00:00')).timestamp())
    last = int(datetime.fromisoformat(manifest['lastTimestampUtc'].replace('Z', '+00:00')).timestamp())
    identity = {'sha256': manifest['sha256'], 'rows': manifest['rows'], 'gap_threshold_seconds': 60}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved['identity'])) == identity:
                return saved['counts'], saved['valid'], json.loads(str(saved['audit']))
        raise ValueError('Cache source identity differs; use a new cache path')
    counts = np.zeros((last - first + 1, 2), dtype=np.int64)
    valid = np.ones(len(counts), dtype=np.bool_)
    valid[[0, -1]] = False  # partial edge seconds
    gaps = []
    digest = hashlib.sha256()
    previous_ms = None
    rows = 0
    with source.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            timestamp = row['timestampMs']
            second = timestamp // 1000 - first
            if previous_ms is not None and timestamp < previous_ms:
                raise ValueError('Timestamp regression')
            if previous_ms is not None and timestamp - previous_ms > 60_000:
                start = previous_ms // 1000 - first
                valid[start:second + 1] = False
                gaps.append({'start_second': start, 'end_second': second,
                             'duration_seconds': (timestamp - previous_ms) / 1000})
            side = {'buy': 0, 'sell': 1}[row['aggressorSide']]
            counts[second, side] += 1
            previous_ms = timestamp
            rows += 1
            if rows % 2_000_000 == 0:
                print(f'Parsed {rows:,} / {manifest["rows"]:,}', flush=True)
    if rows != manifest['rows'] or digest.hexdigest() != manifest['sha256']:
        raise ValueError('Export does not match manifest')
    audit = {'rows': rows, 'seconds': len(counts), 'first_second_unix': first,
             'long_gaps': gaps, 'excluded_seconds': int((~valid).sum()),
             'complete_exchange_history': manifest['completeExchangeHistory'],
             'missing_aggregate_ids': manifest['missingAggregateTradeIds'],
             'coverage_note': '60s gap mask is a heuristic; shorter recording losses remain unknown.'}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.partial.npz')
    np.savez_compressed(temporary, counts=counts, valid=valid,
                        identity=json.dumps(identity), audit=json.dumps(audit))
    temporary.replace(cache)
    return counts, valid, audit


@njit(cache=True)
def features(y, valid, taus, dt):
    n, k = len(y), len(taus)
    pairs = [(fast, slow) for fast in range(k) for slow in range(fast + 1, k)]
    linear = np.zeros((n, 1 + 2 * k))
    modulated = np.zeros((n, 4 * len(pairs)))
    state = np.zeros((k, 2))
    products = np.zeros((len(pairs), 2, 2))
    decay = np.exp(-dt / taus)
    age = np.zeros(n, dtype=np.int64)
    elapsed = 0
    for t in range(n):
        linear[t, 0] = 1.
        if not valid[t]:
            state[:] = 0.
            products[:] = 0.
            elapsed = 0
            continue
        age[t] = elapsed
        linear[t, 1:] = state.ravel()
        modulated[t] = products.ravel()
        # Slow state is taken before the event bin; the product enters next bin.
        for p in range(len(pairs)):
            fast, slow = pairs[p]
            for a in range(2):
                for b in range(2):
                    products[p, a, b] = decay[fast] * products[p, a, b] + (1 - decay[fast]) * y[t, a] * state[slow, b]
        for j in range(k):
            state[j] = decay[j] * state[j] + (1 - decay[j]) * y[t]
        elapsed += 1
    return linear, modulated, age


@njit(cache=True)
def objective(w, x, y):
    loss = 0.
    grad = np.zeros_like(w)
    for t in range(len(y)):
        for side in range(2):
            rate = max(np.dot(x[t], w[:, side]), 1e-10)
            loss += rate - y[t, side] * np.log(rate)
            grad[:, side] += x[t] * (1 - y[t, side] / rate)
    return loss / len(y), grad / len(y)


def fit(x, y, train):
    scale = np.maximum(x[train].mean(axis=0), 1e-8)
    xx = np.ascontiguousarray(x[train] / scale)
    yy = np.ascontiguousarray(y[train])
    w = np.full((x.shape[1], 2), 0.01)
    w[0] = yy.mean(axis=0) * .5
    def fun(flat):
        loss, gradient = objective(flat.reshape(w.shape), xx, yy)
        return loss, gradient.ravel()
    result = minimize(fun, w.ravel(), jac=True, method='L-BFGS-B',
                      bounds=[(1e-8, None)] * 2 + [(0., None)] * (w.size - 2),
                      options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-5})
    weights = result.x.reshape(w.shape) / scale[:, None]
    return weights, {'success': bool(result.success), 'message': str(result.message),
                     'iterations': int(result.nit), 'train_loss': float(result.fun)}


@njit(cache=True)
def particle_filter(y, valid, baseline, source_state, params, tau, dt, particles, seed):
    """Weights use y[t] only AFTER recording its predictive distribution.

    State at t contains events before t. Resample posterior state, then draw L[t]
    from its generative intensity (conditional on that prior state). B/S[t] and
    L[t] are conditionally independent within a bin in this discretization.
    """
    np.random.seed(seed)
    z = np.zeros(particles)
    next_z = np.zeros(particles)
    weights = np.full(particles, 1 / particles)
    logw = np.zeros(particles)
    prediction = np.zeros_like(y)
    losses = np.zeros(len(y))
    ess = np.zeros(len(y))
    decay = np.exp(-dt / tau)
    for t in range(len(y)):
        if not valid[t]:
            z[:] = 0.
            weights[:] = 1 / particles
            continue
        for p in range(particles):
            ll = np.log(max(weights[p], 1e-300))
            for side in range(2):
                rate = max(params[6 + side] * baseline[t, side, 0] + params[8 + side] * baseline[t, side, 1] + params[4 + side] * z[p], 1e-10)
                prediction[t, side] += weights[p] * rate
                ll += y[t, side] * np.log(rate) - rate
            logw[p] = ll
        peak = np.max(logw)
        posterior = np.exp(logw - peak)
        total = posterior.sum()
        losses[t] = -(peak + np.log(total))
        posterior /= total
        ess[t] = 1 / np.sum(posterior * posterior)
        # Systematic resampling; propagate latent events after observation.
        u = np.random.random() / particles
        index = 0
        cumulative = posterior[0]
        for p in range(particles):
            target = u + p / particles
            while cumulative < target and index < particles - 1:
                index += 1
                cumulative += posterior[index]
            latent_rate = params[0] + params[1] * source_state[t, 0] + params[2] * source_state[t, 1] + params[3] * z[index]
            latent_count = np.random.poisson(latent_rate)
            next_z[p] = decay * z[index] + (1 - decay) * latent_count
        z[:] = next_z
        weights[:] = 1 / particles
    return prediction, losses, ess


def metrics(y, prediction, mask, losses=None):
    if losses is None:
        rate = np.maximum(prediction, 1e-10)
        losses = (rate - y * np.log(rate)).sum(axis=1)
    return {'bins': int(mask.sum()), 'poisson_loss_without_factorial': float(losses[mask].mean()),
            'total_count_mae': float(np.abs(y[mask].sum(axis=1) - prediction[mask].sum(axis=1)).mean()),
            'side_mae': np.abs(y[mask] - prediction[mask]).mean(axis=0).tolist()}


def branching_core(weights, params, k):
    """Integrated linear B/S/L kernel; M2 nonlinear terms are outside this core."""
    matrix = np.zeros((3, 3))
    matrix[:2, :2] = weights[1:1 + 2 * k].reshape(k, 2, 2).sum(axis=0).T * params[8:10, None]
    matrix[:2, 2] = params[4:6]
    matrix[2] = params[1:4]
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/exports/BTCUSDT-aggTrades-live-2026-08-17_2026-08-30.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--latent', action='store_true')
    parser.add_argument('--particles', type=int, default=32)
    parser.add_argument('--sweeps', type=int, default=3)
    parser.add_argument('--audit-particles', type=int, default=0,
                        help='Reevaluate saved latent models with this particle count; no refitting')
    args = parser.parse_args()
    if args.particles < 2 or args.sweeps < 1:
        parser.error('--particles must be >=2 and --sweeps must be >=1')
    args.output.mkdir(parents=True, exist_ok=True)
    counts, valid, audit = prepare(args.source, args.source.with_suffix('.metadata.json'), args.output / 'counts-1s.npz')
    write_json(args.output / 'data-audit.json', audit)
    print(json.dumps(audit), flush=True)
    if args.prepare_only:
        return
    dt = 5
    n = len(counts) // dt
    y = counts[:n * dt].reshape(n, dt, 2).sum(axis=1).astype(np.float64)
    valid = valid[:n * dt].reshape(n, dt).all(axis=1)
    taus = np.array([10., 40., 160., 640., 2560.])
    linear, modulation, age = features(y, valid, taus, dt)
    # Exclude initialization transients after each known long recording gap.
    usable = valid & (age >= int(3 * taus[-1] / dt))
    index = np.arange(n)
    masks = [usable & (index < int(.6 * n)), usable & (index >= int(.6 * n)) & (index < int(.8 * n)), usable & (index >= int(.8 * n))]
    if any(not mask.any() for mask in masks):
        raise ValueError('At least one chronological split has no usable observations')
    report_path = args.output / 'report.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    if args.audit_particles:
        if args.audit_particles < 2:
            parser.error('--audit-particles must be >=2')
        for name in ['linear', 'M2']:
            entry = previous['models'][name + '+L']
            with np.load(args.output / f'{name}.npz', allow_pickle=False) as saved:
                weights, prediction = saved['weights'], saved['prediction']
            baseline = np.stack((np.broadcast_to(weights[0], prediction.shape), prediction - weights[0]), axis=2)
            tau, params = entry['tau_seconds'], np.array(entry['params'])
            state_x = features(y, valid, np.array([tau]), dt)[0][:, 1:3]
            runs = []
            for seed in [12345, 23456, 34567]:
                pred, losses, ess = particle_filter(y, valid, baseline, state_x, params, tau, dt, args.audit_particles, seed)
                runs.append({'seed': seed, 'metrics': {label: metrics(y, pred, mask, losses) for label, mask in zip(['train', 'validation', 'test'], masks)}, 'test_ess_mean': float(ess[masks[2]].mean()), 'test_ess_below_10pct_fraction': float((ess[masks[2]] < .1 * args.audit_particles).mean())})
            entry['particle_audit'] = {'particles': args.audit_particles, 'runs': runs}
            write_json(report_path, previous)
            print(name + '+L', 'particle audit', json.dumps(entry['particle_audit']), flush=True)
        return
    fit_config = {'particles': args.particles, 'sweeps': args.sweeps, 'linear_core_radius_limit': .995}
    if previous and previous['data'] != audit:
        raise ValueError('Output directory belongs to different source data')
    report = {'approximation': '5s discrete Poisson count process; not exact continuous Hawkes',
              'taus_seconds': taus.tolist(), 'split': 'chronological elapsed-time 60/20/20',
              'gap_mask': 'Exclude >60s gaps and 3*max(tau) warmup after each reset',
              'data': audit, 'fit_config': fit_config, 'models': {}}
    for name, x in [('linear', linear), ('M2', np.column_stack((linear, modulation)))]:
        fitted_path = args.output / f'{name}.npz'
        if fitted_path.exists() and name in previous.get('models', {}):
            with np.load(fitted_path, allow_pickle=False) as saved:
                weights, prediction = saved['weights'], saved['prediction']
            fit_info = previous['models'][name]['fit']
            if weights.shape != (x.shape[1], 2) or prediction.shape != y.shape:
                raise ValueError('Saved observed fit has incompatible dimensions')
        else:
            weights, fit_info = fit(x, y, masks[0])
            prediction = x @ weights
        entry = {'fit': fit_info, 'features': x.shape[1],
                 'metrics': {label: metrics(y, prediction, mask) for label, mask in zip(['train', 'validation', 'test'], masks)}}
        report['models'][name] = entry
        np.savez_compressed(args.output / f'{name}.npz', weights=weights, prediction=prediction)
        write_json(report_path, report)
        print(name, json.dumps(entry), flush=True)
        if not args.latent:
            continue
        if previous.get('fit_config') == fit_config and name + '+L' in previous.get('models', {}):
            report['models'][name + '+L'] = previous['models'][name + '+L']
            write_json(report_path, report)
            print('Resumed completed', name + '+L', flush=True)
            continue
        # Restricted joint fit: each observed baseline/kernel block is rescaled,
        # while all B->L, S->L, L->L, L->B and L->S coefficients are fitted.
        baseline = np.stack((np.broadcast_to(weights[0], prediction.shape), prediction - weights[0]), axis=2)
        end = int(.6 * n)
        best_validation = np.inf
        selected = None
        for tau in [10., 40., 160.]:
            state_x = features(y, valid, np.array([tau]), dt)[0][:, 1:3]
            params = np.array([.1, .01, .01, .2, .3, .3, .9, .9, .9, .9])
            progress_path = args.output / f'{name}-latent-{int(tau)}s-progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
            if progress and progress['fit_config'] != fit_config:
                raise ValueError('Particle fit settings changed; use a new output directory')
            train_args = (y[:end], valid[:end], baseline[:end], state_x[:end])
            def loss(p):
                radius = float(np.max(np.abs(np.linalg.eigvals(branching_core(weights, p, len(taus))))))
                if radius >= .995:
                    return np.inf
                _, losses, _ = particle_filter(*train_args, p, tau, dt, args.particles, 71391)
                return float(losses[masks[0][:end]].mean())
            best = loss(params)
            next_step = 0
            if progress:
                params = np.array(progress['params'])
                best = progress['train_loss']
                next_step = progress['next_step']
            for sweep in range(args.sweeps):
                for j in range(len(params)):
                    if sweep * len(params) + j < next_step:
                        continue
                    candidates = [params[j] * .5, max(params[j] * 2, .01)]
                    if j < 6:
                        candidates.append(0.)
                    for value in candidates:
                        if j == 3 and value >= .95:
                            continue
                        trial = params.copy()
                        trial[j] = value
                        candidate_loss = loss(trial)
                        if candidate_loss < best:
                            params, best = trial, candidate_loss
                    checkpoint = {'tau': tau, 'fit_config': fit_config, 'next_step': sweep * len(params) + j + 1, 'params': params.tolist(), 'train_loss': best}
                    write_json(progress_path, checkpoint)
                print(name, 'latent tau', tau, 'sweep', sweep, 'train', best, flush=True)
            pred, losses, ess = particle_filter(y, valid, baseline, state_x, params, tau, dt, args.particles * 4, 12345)
            validation = float(losses[masks[1]].mean())
            if validation < best_validation:
                best_validation = validation
                selected = (tau, params.copy())
        tau, params = selected
        core = branching_core(weights, params, len(taus))
        state_x = features(y, valid, np.array([tau]), dt)[0][:, 1:3]
        runs = []
        for seed in [12345, 23456, 34567]:
            pred, losses, ess = particle_filter(y, valid, baseline, state_x, params, tau, dt, args.particles * 4, seed)
            runs.append({'seed': seed, 'metrics': {label: metrics(y, pred, mask, losses) for label, mask in zip(['train', 'validation', 'test'], masks)}, 'test_ess_mean': float(ess[masks[2]].mean()), 'test_ess_min': float(ess[masks[2]].min())})
        report['models'][name + '+L'] = {'tau_seconds': tau, 'params': params.tolist(), 'linear_core_spectral_radius': float(np.max(np.abs(np.linalg.eigvals(core)))), 'stability_note': 'Radius <1 establishes stability for the linear count model only; M2 nonlinear feedback is not globally constrained.', 'param_names': ['muL', 'B_to_L', 'S_to_L', 'L_to_L', 'L_to_B', 'L_to_S', 'B_baseline_scale', 'S_baseline_scale', 'B_kernel_scale', 'S_kernel_scale'], 'fit': f'{args.sweeps} multiplicative coordinate sweeps; restricted observed blocks; convergence not established', 'particles_fit': args.particles, 'particles_eval': args.particles * 4, 'runs': runs}
        write_json(report_path, report)
        print(name + '+L', json.dumps(report['models'][name + '+L']), flush=True)


if __name__ == '__main__':
    main()
