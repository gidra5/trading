"""Jointly fitted one-second Hawkes research pipeline.

Each directed edge owns an independent exponential mixture. Masses are discrete
integrated offspring counts: x[t+1]=d*x[t]+(1-d)*Y[t], lambda=mu+sum(g*x).
The parameter vector contains every background, mass and continuous log-timescale.
Architecture selection never reads the test observations or computes test scores.
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
from scipy.special import gammaln


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


@dataclass(frozen=True)
class Mode:
    target: int
    source: int
    mass: float
    tau: float


@dataclass(frozen=True)
class Model:
    background: tuple[float, float]
    modes: tuple[Mode, ...]

    def arrays(self):
        target = np.array([m.target for m in self.modes], dtype=np.int64)
        source = np.array([m.source for m in self.modes], dtype=np.int64)
        theta = np.array([*self.background, *[m.mass for m in self.modes], *[np.log(m.tau) for m in self.modes]])
        return theta, target, source

    def serialize(self):
        return {'background': list(self.background), 'modes': [dict(target=m.target, source=m.source, mass=m.mass, tau=m.tau) for m in self.modes]}

    @classmethod
    def deserialize(cls, value):
        return cls(tuple(value['background']), tuple(Mode(**m) for m in value['modes']))


def from_theta(theta, target, source):
    k = len(target)
    modes = tuple(Mode(int(target[j]), int(source[j]), float(theta[2 + j]), float(np.exp(theta[2 + k + j]))) for j in range(k))
    return Model(tuple(float(x) for x in theta[:2]), tuple(sorted(modes, key=lambda m: (m.target, m.source, m.tau))))


def branching(model: Model):
    matrix = np.zeros((2, 2))
    for mode in model.modes:
        matrix[mode.target, mode.source] += mode.mass
    return matrix


def radius(matrix):
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


HORIZONS = (1, 5, 15, 60, 300, 900, 3600)


def forecast_operators(model, horizons=HORIZONS):
    """Exact conditional mean of future counts under the linear count model.

    The augmented transition carries traces, a constant and accumulated counts.
    Future jumps are their expected counts; observed future data never enter it.
    """
    k = len(model.modes)
    decay = np.exp(-1 / np.array([m.tau for m in model.modes]))
    coefficient = np.zeros((2, k))
    for j, mode in enumerate(model.modes):
        coefficient[mode.target, j] = mode.mass
    transition = np.zeros((k + 3, k + 3))
    for j, mode in enumerate(model.modes):
        transition[j, :k] = (1 - decay[j]) * coefficient[mode.source]
        transition[j, j] += decay[j]
        transition[j, k] = (1 - decay[j]) * model.background[mode.source]
    transition[k, k] = 1.
    transition[k + 1:, :k] = coefficient
    transition[k + 1:, k] = model.background
    transition[k + 1:, k + 1:] = np.eye(2)
    return np.stack([np.linalg.matrix_power(transition, horizon)[k + 1:, :k + 1] for horizon in horizons])


@njit(cache=True)
def open_loop_linear(theta, target, source, y, valid, initial_mean, origins, operators):
    k = len(target)
    state = np.ones(k + 1)
    for j in range(k):
        state[j] = initial_mean[source[j]]
    decay = np.exp(-1 / np.exp(theta[2 + k:]))
    prediction = np.zeros((len(origins), len(operators), 2))
    pointer = 0
    for t in range(len(y)):
        if not valid[t]:
            for j in range(k):
                state[j] = initial_mean[source[j]]
            continue
        if pointer < len(origins) and t == origins[pointer]:
            for h in range(len(operators)):
                for side in range(2):
                    prediction[pointer, h, side] = np.dot(operators[h, side], state)
            pointer += 1
        for j in range(k):
            state[j] = decay[j] * state[j] + (1 - decay[j]) * y[t, source[j]]
    return prediction


def safe_corr(left, right):
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def count_metrics(y, prediction):
    if not np.isfinite(prediction).all():
        return dict.fromkeys(['total_mae', 'total_rmse', 'total_r2', 'total_correlation',
                              'buy_mae', 'sell_mae', 'imbalance_correlation', 'poisson_mean_log_score'])
    total, predicted_total = y.sum(axis=-1), prediction.sum(axis=-1)
    mse = float(np.mean((total - predicted_total) ** 2))
    variance = float(np.var(total))
    return {'total_mae': float(np.mean(np.abs(total - predicted_total))),
            'total_rmse': float(np.sqrt(mse)), 'total_r2': 1 - mse / variance if variance else None,
            'total_correlation': safe_corr(total, predicted_total),
            'buy_mae': float(np.abs(y[:, 0] - prediction[:, 0]).mean()),
            'sell_mae': float(np.abs(y[:, 1] - prediction[:, 1]).mean()),
            'imbalance_correlation': safe_corr(y[:, 0] - y[:, 1], prediction[:, 0] - prediction[:, 1]),
            'poisson_mean_log_score': float(np.mean((prediction - y * np.log(np.maximum(prediction, 1e-12)) + gammaln(y + 1)).sum(axis=1)))}


def evaluation_region(corpus, split):
    if split not in ('validation', 'test'):
        raise ValueError('Unknown evaluation split')
    start = corpus.train_end if split == 'validation' else corpus.validation_end
    end = corpus.validation_end if split == 'validation' else len(corpus.y)
    if start >= end:
        raise ValueError(f'{split} observations are unavailable')
    mask = corpus.score[:end].copy()
    mask[:start] = False
    mask.setflags(write=False)
    return start, end, mask


def forecast_origins(corpus, split='validation', stride=900, maximum=3600):
    if stride < 1 or maximum < 1:
        raise ValueError('Positive horizon and stride required')
    start, end, _ = evaluation_region(corpus, split)
    origins = np.arange(start, end - maximum + 1, stride)
    excluded = np.r_[0, np.cumsum(~corpus.score[:end])]
    origins = origins[excluded[origins + maximum] == excluded[origins]]
    if not len(origins):
        raise ValueError('No fully scored forecast origins')
    origins.setflags(write=False)
    return origins


def evaluate_forecasts(model, corpus, output, stride=900, split='validation'):
    output.mkdir(parents=True, exist_ok=True)
    _, end, mask = evaluation_region(corpus, split)
    origins = forecast_origins(corpus, split, stride, max(HORIZONS))
    theta, target, source = model.arrays()
    one_step = linear_predictions(theta, target, source, corpus.y[:end], corpus.valid[:end], corpus.initial_mean)
    save_json(output / f'{split}_one_step_metrics.json', count_metrics(corpus.y[:end][mask], one_step[mask]))
    scored_y, scored_rate = corpus.y[:end][mask], one_step[mask]
    loss = (scored_rate - scored_y * np.log(np.maximum(scored_rate, 1e-12)) + gammaln(scored_y + 1)).sum(axis=1)
    np.savez_compressed(output / f'{split}_predictive_loss.npz', seconds=np.flatnonzero(mask), marginal_nll=loss)
    prediction = open_loop_linear(theta, target, source, corpus.y[:end], corpus.valid[:end], corpus.initial_mean, origins, forecast_operators(model))
    cumulative = np.vstack((np.zeros(2), np.cumsum(corpus.y[:end], axis=0)))
    rows = []
    targets = []
    for j, horizon in enumerate(HORIZONS):
        actual = cumulative[origins + horizon] - cumulative[origins]
        targets.append(actual)
        rows.append({'model': 'M0', 'split': split, 'horizon_seconds': horizon,
                     'origins': len(origins), **count_metrics(actual, prediction[:, j])})
    with (output / f'{split}_horizon_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / f'{split}_forecasts.npz', origins=origins, prediction=prediction,
                        actual=np.stack(targets, axis=1), horizons=np.array(HORIZONS))
    save_json(output / 'forecast-protocol.json', {'stride_seconds': stride,
              'origins': 'Common origins whose full 3600s future is scored/covered',
              'mean': 'Exact linear conditional mean via augmented matrix powers',
              'horizon_log_score': 'Poisson mean score; not the exact multi-step Hawkes marginal count likelihood',
              'model': model.serialize(), 'corpus': corpus.identity, 'split': split,
              'test_evaluated': split == 'test'})


@njit(cache=True)
def linear_objective(theta, target, source, y, valid, score, initial_mean):
    k = len(target)
    mass = theta[2:2 + k]
    tau = np.exp(theta[2 + k:])
    decay = np.exp(-1 / tau)
    decay_derivative = decay / tau  # derivative wrt log(tau)
    state = np.zeros(k)
    derivative = np.zeros(k)
    for j in range(k):
        state[j] = initial_mean[source[j]]
    gradient = np.zeros(len(theta))
    total = 0.
    nscore = 0
    for t in range(len(y)):
        if not valid[t]:
            for j in range(k):
                state[j] = initial_mean[source[j]]
                derivative[j] = 0.
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
                gradient[2 + j] += residual[target[j]] * state[j]
                gradient[2 + k + j] += residual[target[j]] * mass[j] * derivative[j]
        for j in range(k):
            event = y[t, source[j]]
            derivative[j] = decay[j] * derivative[j] + decay_derivative[j] * (state[j] - event)
            state[j] = decay[j] * state[j] + (1 - decay[j]) * event
    return total / nscore, gradient / nscore


@njit(cache=True)
def linear_predictions(theta, target, source, y, valid, initial_mean):
    k = len(target)
    decay = np.exp(-1 / np.exp(theta[2 + k:]))
    state = np.zeros(k)
    for j in range(k):
        state[j] = initial_mean[source[j]]
    prediction = np.zeros_like(y)
    for t in range(len(y)):
        if not valid[t]:
            for j in range(k):
                state[j] = initial_mean[source[j]]
            continue
        prediction[t] = theta[:2]
        for j in range(k):
            prediction[t, target[j]] += theta[2 + j] * state[j]
            state[j] = decay[j] * state[j] + (1 - decay[j]) * y[t, source[j]]
    return prediction


@njit(cache=True)
def coverage_age(valid):
    age = np.zeros(len(valid), dtype=np.int64)
    elapsed = 0
    for t in range(len(valid)):
        if not valid[t]:
            elapsed = 0
        else:
            age[t] = elapsed
            elapsed += 1
    return age


@dataclass(frozen=True)
class Corpus:
    y: np.ndarray
    valid: np.ndarray
    score: np.ndarray
    train_end: int
    validation_end: int
    initial_mean: np.ndarray
    identity: dict


def load_corpus(cache: Path, warmup: int):
    return _load_corpus(cache, warmup, False)


def _load_corpus(cache: Path, warmup: int, include_test: bool):
    with np.load(cache, allow_pickle=False) as data:
        y = data['counts'].astype(np.float64)
        valid = data['valid']
        identity = json.loads(str(data['identity']))
        audit = json.loads(str(data['audit']))
    n = len(y)
    train_end, validation_end = int(.6 * n), int(.8 * n)
    score = valid & (coverage_age(valid) >= warmup)
    initial_mean = y[:train_end][score[:train_end]].mean(axis=0)
    # Prevent selection routines from accessing test observations at all.
    if not include_test:
        y = y[:validation_end]
        valid = valid[:validation_end]
        score = score[:validation_end]
    for array in (y, valid, score, initial_mean):
        array.setflags(write=False)
    return Corpus(y, valid, score, train_end, validation_end, initial_mean,
                  {'cache_identity': identity, 'source_audit': audit, 'warmup_seconds': warmup, 'full_seconds': n,
                   'train_end': train_end, 'validation_end': validation_end, 'initial_mean': initial_mean.tolist(),
                   'train_scored_seconds': int(score[:train_end].sum()), 'validation_scored_seconds': int(score[train_end:validation_end].sum())})


def fit_joint(model, corpus, rho_max=.95, maxiter=500):
    theta, target, source = model.arrays()
    k = len(target)
    def matrix(x):
        result = np.zeros((2, 2))
        for j in range(k):
            result[target[j], source[j]] += x[2 + j]
        return result
    # A feasible initialization avoids asking SLSQP to repair unstable kernels.
    current_radius = radius(matrix(theta))
    if current_radius >= rho_max:
        theta[2:2 + k] *= .9 * rho_max / current_radius
    n = corpus.train_end
    def objective(x):
        return linear_objective(x, target, source, corpus.y[:n], corpus.valid[:n], corpus.score[:n], corpus.initial_mean)
    start = time.monotonic()
    result = minimize(objective, theta, method='SLSQP', jac=True,
                      bounds=[(1e-7, 100.)] * 2 + [(0., 2.)] * k + [(np.log(.1), np.log(120000.))] * k,
                      constraints=[{'type': 'ineq', 'fun': lambda x: rho_max - radius(matrix(x))}],
                      options={'maxiter': maxiter, 'ftol': 1e-9})
    fitted = from_theta(result.x, target, source)
    validation_score = corpus.score.copy()
    validation_score[:corpus.train_end] = False
    validation, _ = linear_objective(result.x, target, source, corpus.y, corpus.valid, validation_score, corpus.initial_mean)
    actual_radius = radius(branching(fitted))
    diagnostics = {'success': bool(result.success), 'message': str(result.message), 'iterations': int(result.nit),
                   'train_nll_without_factorial': float(result.fun), 'validation_nll_without_factorial': float(validation),
                   'spectral_radius': actual_radius, 'seconds': time.monotonic() - start,
                   'n_parameters': len(theta), 'joint_parameters': 'all backgrounds, masses and log-timescales'}
    if actual_radius > rho_max + 1e-6 or not np.isfinite(validation):
        diagnostics['success'] = False
        diagnostics['message'] += '; infeasible or nonfinite final parameters'
    return fitted, diagnostics


def counts(model):
    return tuple(sum(m.target == target and m.source == source for m in model.modes) for target in range(2) for source in range(2))


def architecture_search(corpus, output, max_modes=6, rho_max=.95, maxiter=500):
    output.mkdir(parents=True, exist_ok=True)
    config = {'version': 1, 'family': 'M0', 'max_modes_per_edge': max_modes,
              'rho_max': rho_max, 'maxiter': maxiter, 'corpus': corpus.identity,
              'selection': 'validation NLL + n_parameters/n_validation (complexity penalty)'}
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Research settings changed; use a new output directory')
    save_json(path, config)
    records = []
    nvalidation = corpus.identity['validation_scored_seconds']
    def penalized(diagnostics):
        return diagnostics['validation_nll_without_factorial'] + diagnostics['n_parameters'] / nvalidation
    def candidate(seed, label):
        fingerprint = hashlib.sha256(json.dumps({'seed': seed.serialize(), 'config': config}, sort_keys=True).encode()).hexdigest()[:20]
        cache_path = output / 'candidates' / f'{fingerprint}.json'
        if cache_path.exists():
            saved = json.loads(cache_path.read_text())
            fitted, info = Model.deserialize(saved['model']), saved['fit']
        else:
            fitted, info = fit_joint(seed, corpus, rho_max, maxiter)
            save_json(cache_path, {'model': fitted.serialize(), 'fit': info})
        row = {'candidate': fingerprint, 'operation': label, 'counts_BB_BS_SB_SS': str(counts(fitted)), **info,
               'selection_score': penalized(info)}
        records.append(row)
        with (output / 'model_selection.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps(row), flush=True)
        return fitted, info

    mean = corpus.initial_mean
    starts = []
    for tau in [2., 100., 10000.]:
        seed = Model(tuple(mean * .3), tuple(Mode(i, j, .3 if i == j else .1, tau if i == j else 10 * tau) for i in range(2) for j in range(2)))
        starts.append(candidate(seed, 'initialize'))
    feasible = [(m, d) for m, d in starts if d['success']]
    if not feasible:
        raise RuntimeError('No initial fit converged; see candidate diagnostics')
    current, current_info = min(feasible, key=lambda pair: penalized(pair[1]))
    history = []
    stop_reason = 'iteration_limit'
    for iteration in range(32):
        history.append({'iteration': iteration, 'model': current.serialize(), 'fit': current_info})
        save_json(output / 'selection-history.json', history)
        save_json(output / 'selected-M0.json', {'model': current.serialize(), 'fit': current_info, 'selection_complete': False})
        signature = counts(current)
        additions = []
        for target in range(2):
            for source in range(2):
                if signature[2 * target + source] >= max_modes:
                    continue
                for tau in [1., 100., 10000.]:
                    seed = Model(current.background, current.modes + (Mode(target, source, .03, tau),))
                    model, info = candidate(seed, f'add_{target}_{source}_seed_{tau}')
                    if info['success']:
                        additions.append((model, info))
        changed = False
        if additions:
            best = min(additions, key=lambda pair: penalized(pair[1]))
            if penalized(best[1]) < penalized(current_info) - 1e-5:
                current, current_info = best
                changed = True
        # Every deletion candidate refits every remaining amplitude and scale.
        while len(current.modes):
            deletions = []
            for j in range(len(current.modes)):
                seed = Model(current.background, current.modes[:j] + current.modes[j + 1:])
                model, info = candidate(seed, f'delete_{j}')
                if info['success']:
                    deletions.append((model, info))
            if not deletions:
                break
            best = min(deletions, key=lambda pair: penalized(pair[1]))
            if penalized(best[1]) >= penalized(current_info) - 1e-5:
                break
            current, current_info = best
            changed = True
        if not changed:
            stop_reason = 'no_forward_or_backward_validation_improvement'
            break
    save_json(output / 'selected-M0.json', {'model': current.serialize(), 'fit': current_info,
                                          'selection_complete': stop_reason != 'iteration_limit', 'stop_reason': stop_reason,
                                          'history_steps': len(history),
                                          'warning': 'Local multi-start search, not proof of a global optimum; test untouched by this run.'})
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--output', type=Path, default=Path('data/benchmarks/hawkes-joint-1s-2026-10-03/M0'))
    parser.add_argument('--warmup', type=int, default=3600)
    parser.add_argument('--max-modes', type=int, default=6)
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--forecast-stride', type=int, default=900)
    args = parser.parse_args()
    corpus = load_corpus(args.cache, args.warmup)
    selected = architecture_search(corpus, args.output, args.max_modes, maxiter=args.maxiter)
    evaluate_forecasts(selected, corpus, args.output, args.forecast_stride)
    print('SELECTED', json.dumps(selected.serialize()), flush=True)


if __name__ == '__main__':
    main()
