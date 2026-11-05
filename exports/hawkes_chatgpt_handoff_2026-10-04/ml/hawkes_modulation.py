"""Jointly fitted cross-scale event-amplitude modulation (M2).

For fast mode k, an event jump is multiplied by
1 + sum_a beta[k,a]*gate(tau_a/tau_k)*(slow_a/train_mean[source_a]).
The smooth gate favors anchors at least four times slower than the fast mode.
All masses, fast timescales, modulation amplitudes and anchor timescales are
jointly optimized. A dominating linear branching matrix provides a global
stability condition for the optional bounded-tanh variant. The handoff's linear
amplitude modulation is the default; its nonlinear stability is not implied by
the additive core's spectral radius and must be tested by generative rollouts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize

from hawkes_research import Model, Mode, counts, from_theta, load_corpus, radius, save_json


@dataclass(frozen=True)
class Anchor:
    source: int
    tau: float


@dataclass(frozen=True)
class Modulated:
    base: Model
    anchors: tuple[Anchor, ...]
    beta: tuple[tuple[float, ...], ...]
    link: str = 'linear'

    def arrays(self):
        theta, target, source = self.base.arrays()
        theta = np.r_[theta, np.array(self.beta).ravel(), [np.log(a.tau) for a in self.anchors]]
        return theta, target, source, np.array([a.source for a in self.anchors], dtype=np.int64)

    def serialize(self):
        return {'base': self.base.serialize(), 'anchors': [dict(source=a.source, tau=a.tau) for a in self.anchors], 'beta': [list(row) for row in self.beta], 'link': self.link}

    @classmethod
    def deserialize(cls, value):
        return cls(Model.deserialize(value['base']), tuple(Anchor(**a) for a in value['anchors']), tuple(tuple(row) for row in value['beta']), value['link'])


def unpack(theta, target, source, anchor_source, link='linear'):
    k, a = len(target), len(anchor_source)
    # Keep mode order aligned with the modulation rows while sorting for reporting.
    order = sorted(range(k), key=lambda j: (target[j], source[j], theta[2 + k + j]))
    modes = tuple(Mode(int(target[j]), int(source[j]), float(theta[2 + j]), float(np.exp(theta[2 + k + j]))) for j in order)
    beta = theta[2 + 2 * k:2 + 2 * k + k * a].reshape(k, a)
    anchors = tuple(Anchor(int(anchor_source[j]), float(np.exp(theta[2 + 2 * k + k * a + j]))) for j in range(a))
    return Modulated(Model(tuple(float(x) for x in theta[:2]), modes), anchors, tuple(tuple(float(v) for v in beta[j]) for j in order), link)


@njit(cache=True)
def gate_values(fast_log_tau, slow_log_tau):
    k, a = len(fast_log_tau), len(slow_log_tau)
    gates = np.zeros((k, a))
    for j in range(k):
        for p in range(a):
            exponent = 4 * (slow_log_tau[p] - fast_log_tau[j] - np.log(4.))
            gates[j, p] = 1 / (1 + np.exp(-exponent))
    return gates


def upper_branching(theta, target, source, anchor_source):
    k, a = len(target), len(anchor_source)
    gates = gate_values(theta[2 + k:2 + 2 * k], theta[2 + 2 * k + k * a:])
    beta = theta[2 + 2 * k:2 + 2 * k + k * a].reshape(k, a)
    matrix = np.zeros((2, 2))
    for j in range(k):
        matrix[target[j], source[j]] += theta[2 + j] * (1 + np.dot(beta[j], gates[j]))
    return matrix


@njit(cache=True)
def modulated_objective(theta, target, source, anchor_source, y, valid, score, initial_mean, saturating=False):
    k, a = len(target), len(anchor_source)
    mass = theta[2:2 + k]
    fast_log = theta[2 + k:2 + 2 * k]
    beta_start = 2 + 2 * k
    slow_start = beta_start + k * a
    beta = theta[beta_start:slow_start].reshape(k, a)
    fast_tau = np.exp(fast_log)
    slow_tau = np.exp(theta[slow_start:])
    d = np.exp(-1 / fast_tau)
    ds = np.exp(-1 / slow_tau)
    gates = gate_values(fast_log, theta[slow_start:])
    gate_derivative = 4 * gates * (1 - gates)
    state = np.zeros(k)
    fast_derivative = np.zeros(k)
    beta_derivative = np.zeros((k, a))
    anchor_derivative = np.zeros((k, a))
    slow_state = np.zeros(a)
    slow_derivative = np.zeros(a)
    for j in range(k):
        state[j] = initial_mean[source[j]]
    for p in range(a):
        slow_state[p] = initial_mean[anchor_source[p]]
    gradient = np.zeros(len(theta))
    total, nscore = 0., 0
    for t in range(len(y)):
        if not valid[t]:
            for j in range(k):
                state[j] = initial_mean[source[j]]
            for p in range(a):
                slow_state[p] = initial_mean[anchor_source[p]]
            fast_derivative[:] = 0.
            beta_derivative[:] = 0.
            anchor_derivative[:] = 0.
            slow_derivative[:] = 0.
            continue
        rate = theta[:2].copy()
        for j in range(k):
            rate[target[j]] += mass[j] * state[j]
        if score[t]:
            nscore += 1
            residual = np.zeros(2)
            for side in range(2):
                total += rate[side] - y[t, side] * np.log(rate[side])
                residual[side] = 1 - y[t, side] / rate[side]
                gradient[side] += residual[side]
            for j in range(k):
                r = residual[target[j]]
                gradient[2 + j] += r * state[j]
                gradient[2 + k + j] += r * mass[j] * fast_derivative[j]
                for p in range(a):
                    gradient[beta_start + j * a + p] += r * mass[j] * beta_derivative[j, p]
                    gradient[slow_start + p] += r * mass[j] * anchor_derivative[j, p]
        bounded = np.zeros(a)
        bounded_derivative = np.zeros(a)
        for p in range(a):
            denominator = max(initial_mean[anchor_source[p]], 1e-8)
            if saturating:
                bounded[p] = np.tanh(slow_state[p] / denominator)
                bounded_derivative[p] = (1 - bounded[p] ** 2) * slow_derivative[p] / denominator
            else:
                bounded[p] = slow_state[p] / denominator
                bounded_derivative[p] = slow_derivative[p] / denominator
        for j in range(k):
            factor, fast_factor_derivative = 1., 0.
            event = y[t, source[j]]
            for p in range(a):
                factor += beta[j, p] * gates[j, p] * bounded[p]
                fast_factor_derivative -= beta[j, p] * gate_derivative[j, p] * bounded[p]
                beta_derivative[j, p] = d[j] * beta_derivative[j, p] + (1 - d[j]) * event * gates[j, p] * bounded[p]
                anchor_factor_derivative = beta[j, p] * (gate_derivative[j, p] * bounded[p] + gates[j, p] * bounded_derivative[p])
                anchor_derivative[j, p] = d[j] * anchor_derivative[j, p] + (1 - d[j]) * event * anchor_factor_derivative
            driver = event * factor
            fast_derivative[j] = d[j] * fast_derivative[j] + d[j] / fast_tau[j] * (state[j] - driver) + (1 - d[j]) * event * fast_factor_derivative
            state[j] = d[j] * state[j] + (1 - d[j]) * driver
        for p in range(a):
            event = y[t, anchor_source[p]]
            slow_derivative[p] = ds[p] * slow_derivative[p] + ds[p] / slow_tau[p] * (slow_state[p] - event)
            slow_state[p] = ds[p] * slow_state[p] + (1 - ds[p]) * event
    return total / nscore, gradient / nscore


def fit_joint(seed, corpus, maxiter=700, rho_max=.95):
    theta, target, source, anchor_source = seed.arrays()
    k, a = len(target), len(anchor_source)
    def constraint_matrix(x):
        if seed.link == 'bounded':
            return upper_branching(x, target, source, anchor_source)
        matrix = np.zeros((2, 2))
        for j in range(k):
            matrix[target[j], source[j]] += x[2 + j]
        return matrix
    current_radius = radius(constraint_matrix(theta))
    if current_radius >= rho_max:
        theta[2:2 + k] *= .9 * rho_max / current_radius
    n = corpus.train_end
    def objective(x):
        return modulated_objective(x, target, source, anchor_source, corpus.y[:n], corpus.valid[:n], corpus.score[:n], corpus.initial_mean, seed.link == 'bounded')
    bounds = [(1e-7, 100.)] * 2 + [(0., 2.)] * k + [(np.log(.1), np.log(120000.))] * k
    bounds += [(0., 3.)] * (k * a) + [(np.log(1.), np.log(480000.))] * a
    start = time.monotonic()
    result = minimize(objective, theta, method='SLSQP', jac=True, bounds=bounds,
                      constraints=[{'type': 'ineq', 'fun': lambda x: rho_max - radius(constraint_matrix(x))}],
                      options={'maxiter': maxiter, 'ftol': 1e-9})
    validation_score = corpus.score.copy()
    validation_score[:n] = False
    validation, _ = modulated_objective(result.x, target, source, anchor_source, corpus.y, corpus.valid, validation_score, corpus.initial_mean, seed.link == 'bounded')
    rho = radius(constraint_matrix(result.x))
    info = {'success': bool(result.success and rho <= rho_max + 1e-6), 'message': str(result.message),
            'iterations': int(result.nit), 'train_nll_without_factorial': float(result.fun),
            'validation_nll_without_factorial': float(validation), 'constraint_spectral_radius': rho,
            'stability_type': 'dominating linear kernel' if seed.link == 'bounded' else 'additive core only; nonlinear stability unresolved',
            'seconds': time.monotonic() - start, 'n_parameters': len(theta),
            'joint_parameters': 'backgrounds, masses, fast scales, all beta, all slow-anchor scales'}
    return unpack(result.x, target, source, anchor_source, seed.link), info


def search(corpus, base, output, max_modes=6, max_anchors=3, maxiter=700, link='linear'):
    output.mkdir(parents=True, exist_ok=True)
    config = {'version': 1, 'family': 'M2', 'max_modes_per_edge': max_modes, 'max_anchors_per_source': max_anchors,
              'maxiter': maxiter, 'rho_max': .95, 'link': link, 'corpus': corpus.identity, 'base_seed': base.serialize()}
    protocol = output / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise ValueError('M2 settings changed; use a new output directory')
    save_json(protocol, config)
    rows = []
    nv = corpus.identity['validation_scored_seconds']
    def score(info):
        return info['validation_nll_without_factorial'] + info['n_parameters'] / nv
    def fit(seed, label):
        key = hashlib.sha256(json.dumps({'seed': seed.serialize(), 'config': config}, sort_keys=True).encode()).hexdigest()[:20]
        path = output / 'candidates' / f'{key}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            model, info = Modulated.deserialize(saved['model']), saved['fit']
        else:
            model, info = fit_joint(seed, corpus, maxiter=maxiter)
            save_json(path, {'model': model.serialize(), 'fit': info})
        row = {'candidate': key, 'operation': label, 'edge_counts': str(counts(model.base)),
               'anchor_counts': str(tuple(sum(a.source == s for a in model.anchors) for s in range(2))),
               **info, 'selection_score': score(info)}
        rows.append(row)
        with (output / 'model_selection.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        return model, info
    initial = [fit(Modulated(base, (), tuple(() for _ in base.modes), link), 'initialize_nested_M0')]
    for tau in [30., 300., 3600.]:
        initial.append(fit(Modulated(base, (Anchor(0, tau), Anchor(1, tau)), tuple((.1, .1) for _ in base.modes), link), f'initialize_{tau}'))
    feasible = [pair for pair in initial if pair[1]['success']]
    if not feasible:
        raise RuntimeError('No M2 initialization converged')
    current, info = min(feasible, key=lambda pair: score(pair[1]))
    history = []
    stop = 'iteration_limit'
    for iteration in range(32):
        history.append({'iteration': iteration, 'model': current.serialize(), 'fit': info})
        save_json(output / 'selection-history.json', history)
        save_json(output / 'selected-M2.json', {'model': current.serialize(), 'fit': info, 'selection_complete': False})
        additions = []
        edge_counts = counts(current.base)
        for i in range(2):
            for j in range(2):
                if edge_counts[2 * i + j] >= max_modes:
                    continue
                for tau in [1., 100., 10000.]:
                    base_seed = Model(current.base.background, current.base.modes + (Mode(i, j, .02, tau),))
                    candidate = Modulated(base_seed, current.anchors, current.beta + (tuple(.1 for _ in current.anchors),), link)
                    result = fit(candidate, f'add_edge_{i}_{j}_{tau}')
                    if result[1]['success']:
                        additions.append(result)
        for source in range(2):
            if sum(a.source == source for a in current.anchors) >= max_anchors:
                continue
            for tau in [30., 300., 3600.]:
                candidate = Modulated(current.base, current.anchors + (Anchor(source, tau),), tuple(row + (.1,) for row in current.beta), link)
                result = fit(candidate, f'add_anchor_{source}_{tau}')
                if result[1]['success']:
                    additions.append(result)
        changed = False
        if additions:
            best = min(additions, key=lambda pair: score(pair[1]))
            if score(best[1]) < score(info) - 1e-5:
                current, info = best
                changed = True
        while current.base.modes or current.anchors:
            deletions = []
            for j in range(len(current.base.modes)):
                reduced_base = Model(current.base.background, current.base.modes[:j] + current.base.modes[j + 1:])
                seed = Modulated(reduced_base, current.anchors, current.beta[:j] + current.beta[j + 1:], link)
                result = fit(seed, f'delete_edge_mode_{j}')
                if result[1]['success']:
                    deletions.append(result)
            for j in range(len(current.anchors)):
                seed = Modulated(current.base, current.anchors[:j] + current.anchors[j + 1:], tuple(row[:j] + row[j + 1:] for row in current.beta), link)
                result = fit(seed, f'delete_anchor_{j}')
                if result[1]['success']:
                    deletions.append(result)
            if not deletions:
                break
            best = min(deletions, key=lambda pair: score(pair[1]))
            if score(best[1]) >= score(info) - 1e-5:
                break
            current, info = best
            changed = True
        if not changed:
            stop = 'no_forward_or_backward_validation_improvement'
            break
    save_json(output / 'selected-M2.json', {'model': current.serialize(), 'fit': info,
              'selection_complete': stop != 'iteration_limit', 'stop_reason': stop, 'history_steps': len(history)})
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--base', type=Path, default=Path('data/benchmarks/hawkes-joint-1s-2026-10-03/M0/selected-M0.json'))
    parser.add_argument('--output', type=Path, default=Path('data/benchmarks/hawkes-joint-1s-2026-10-03/M2'))
    parser.add_argument('--warmup', type=int, default=3600)
    parser.add_argument('--max-modes', type=int, default=6)
    parser.add_argument('--max-anchors', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=700)
    parser.add_argument('--forecast-particles', type=int, default=512)
    parser.add_argument('--forecast-stride', type=int, default=900)
    parser.add_argument('--link', choices=['linear', 'bounded'], default='linear')
    args = parser.parse_args()
    selected = json.loads(args.base.read_text())
    if not selected['selection_complete']:
        raise ValueError('M0 architecture search is still running')
    corpus = load_corpus(args.cache, args.warmup)
    result = search(corpus, Model.deserialize(selected['model']), args.output, args.max_modes, args.max_anchors, args.maxiter, args.link)
    from hawkes_observed_rollouts import evaluate_forecasts
    evaluate_forecasts(result, corpus, args.output, args.forecast_particles, args.forecast_stride)
    print('SELECTED', json.dumps(result.serialize()), flush=True)


if __name__ == '__main__':
    main()
