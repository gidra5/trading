"""Generative three-channel Hawkes count inference and joint likelihood fitting.

L is a Poisson event stream with B/S/L-driven intensity. The filter integrates
the preceding integer L count against the current B/S likelihood, then samples
its posterior. This adapted proposal avoids missing rare impulses simply
because a bootstrap prior draw never contained them. All traces are causal.

Counts remain discrete; no continuous residual proxy is used. Kernel parameters
are independent per directed edge, and every physical parameter is available
to the joint optimizer. This module uses the same one-second normalized kernels
as the observed-model pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import gammaln

from hawkes_research import Model, Mode, load_corpus, radius, save_json
from hawkes_modulation import Anchor, gate_values

LOG_FACTORIAL = gammaln(np.arange(1024, dtype=np.float64) + 1)


@dataclass(frozen=True)
class Diffusion:
    tau: float
    loading: tuple[float, float, float]


@dataclass(frozen=True)
class LatentModel:
    background: tuple[float, float, float]
    modes: tuple[Mode, ...]
    anchors: tuple[Anchor, ...] = ()
    beta: tuple[tuple[float, ...], ...] = ()
    diffusions: tuple[Diffusion, ...] = ()
    link: str = 'linear'

    def serialize(self):
        return {'background': list(self.background),
                'modes': [dict(target=m.target, source=m.source, mass=m.mass, tau=m.tau) for m in self.modes],
                'anchors': [dict(source=a.source, tau=a.tau) for a in self.anchors],
                'beta': [list(row) for row in self.beta],
                'diffusions': [dict(tau=z.tau, loading=list(z.loading)) for z in self.diffusions],
                'link': self.link}

    @classmethod
    def deserialize(cls, value):
        return cls(tuple(value['background']), tuple(Mode(**m) for m in value['modes']),
                   tuple(Anchor(**a) for a in value['anchors']), tuple(tuple(row) for row in value['beta']),
                   tuple(Diffusion(z['tau'], tuple(z['loading'])) for z in value['diffusions']), value['link'])

    def layout(self):
        return (np.array([m.target for m in self.modes], dtype=np.int64),
                np.array([m.source for m in self.modes], dtype=np.int64),
                np.array([a.source for a in self.anchors], dtype=np.int64), len(self.diffusions))

    def raw(self):
        beta = self.beta if self.beta else tuple(tuple(0. for _ in self.anchors) for _ in self.modes)
        return np.r_[np.log(np.maximum([*self.background, *[m.mass for m in self.modes], *[m.tau for m in self.modes]], 1e-12)),
                     [x for row in beta for x in row], np.log([a.tau for a in self.anchors]),
                     np.log([z.tau for z in self.diffusions]), [x for z in self.diffusions for x in z.loading]]


def raw_bounds(layout):
    target, source, anchor_source, q = layout
    k, a = len(target), len(anchor_source)
    log_limits = [(1e-8, 100.), (1e-8, 100.), (1e-8, 10000.)]
    # One hidden macro event can generate many observed trades. A small common
    # amplitude ceiling would rule out precisely the long-scale latent channel
    # being tested. Stability constrains feedback products, not individual edges.
    log_limits += [(1e-12, 1e6 if source[j] == 2 or target[j] == 2 else 50.) for j in range(k)]
    log_limits += [(.1, 120000.)] * k
    bounds = [(np.log(lo), np.log(hi)) for lo, hi in log_limits]
    bounds += [(0., 3.)] * (k * a)
    bounds += [(np.log(1.), np.log(480000.))] * a + [(np.log(.1), np.log(480000.))] * q
    bounds += [(-3., 3.)] * (3 * q)
    return np.array(bounds).T


def decode(raw, layout, link='linear', rho_max=.95):
    target, source, anchor_source, q = layout
    k, a = len(target), len(anchor_source)
    lower, upper = raw_bounds(layout)
    clipped = np.clip(raw, lower, upper)
    values = np.exp(clipped)
    beta_start = 3 + 2 * k
    values[beta_start:beta_start + k * a] = clipped[beta_start:beta_start + k * a]
    if q:
        values[-3 * q:] = clipped[-3 * q:]
    mass = values[3:3 + k].copy()
    tau = values[3 + k:3 + 2 * k]
    beta_start = 3 + 2 * k
    beta = values[beta_start:beta_start + k * a].reshape(k, a)
    slow_tau = values[beta_start + k * a:beta_start + k * a + a]
    gates = gate_values(np.log(tau), np.log(slow_tau))
    matrix = np.zeros((3, 3))
    for j in range(k):
        factor = 1 + np.dot(beta[j], gates[j]) if link == 'bounded' else 1.
        matrix[target[j], source[j]] += mass[j] * factor
    rho = radius(matrix)
    if rho > rho_max:
        mass *= rho_max / rho
    modes = tuple(Mode(int(target[j]), int(source[j]), float(mass[j]), float(tau[j])) for j in range(k))
    anchors = tuple(Anchor(int(anchor_source[j]), float(slow_tau[j])) for j in range(a))
    z_start = beta_start + k * a + a
    z_tau = values[z_start:z_start + q]
    z_loading = values[z_start + q:].reshape(q, 3)
    diffusion = tuple(Diffusion(float(z_tau[j]), tuple(float(x) for x in z_loading[j])) for j in range(q))
    return LatentModel(tuple(float(x) for x in values[:3]), modes, anchors,
                       tuple(tuple(float(x) for x in row) for row in beta), diffusion, link)


def core_branching(model):
    matrix = np.zeros((3, 3))
    gates = gate_values(np.log([m.tau for m in model.modes]), np.log([a.tau for a in model.anchors]))
    for j, mode in enumerate(model.modes):
        factor = 1.
        if model.link == 'bounded' and model.anchors:
            factor += np.dot(model.beta[j], gates[j])
        matrix[mode.target, mode.source] += mode.mass * factor
    return matrix


@njit(cache=True)
def count_logweight(n, prior, base, slope, observed):
    factorial = LOG_FACTORIAL[n] if n < len(LOG_FACTORIAL) else math.lgamma(n + 1.)
    value = n * np.log(prior) - prior - factorial
    for side in range(2):
        rate = base[side] + slope[side] * n
        value += observed[side] * np.log(rate) - rate
    return value


@njit(cache=True)
def count_logratio(n, prior, base, slope, observed):
    value = np.log(prior) - np.log(n + 1.) - slope.sum()
    for side in range(2):
        value += observed[side] * np.log1p(slope[side] / (base[side] + slope[side] * n))
    return value


@njit(cache=True)
def integrated_count(prior, base, slope, observed):
    """Log integral and posterior support/moments, with geometric tail bounds.

    The posterior mass in n is log-concave. Its mode is found by log-ratio
    bisection, so rare posterior impulses outside prior credible intervals are
    included. Relative discarded tail mass is bounded below 1e-12 per side.
    Negative support tags select exact degenerate/Poisson special cases.
    """
    if prior <= 0:
        ll = 0.
        for side in range(2):
            ll += observed[side] * np.log(base[side]) - base[side]
        return ll, -1, -1, 0., 0.
    if slope.sum() == 0:
        ll = 0.
        for side in range(2):
            ll += observed[side] * np.log(base[side]) - base[side]
        return ll, -2, -2, prior, 1 - np.exp(-prior)
    if observed.sum() == 0:
        posterior_mean = prior * np.exp(-slope.sum())
        ll = -base.sum() + prior * (np.exp(-slope.sum()) - 1)
        return ll, -2, -2, posterior_mean, 1 - np.exp(-posterior_mean)
    low, high = 0, max(1, int(prior))
    while count_logratio(high, prior, base, slope, observed) > 0:
        high *= 2
        if high > 10_000_000:
            return -np.inf, -3, -3, 0., 0.
    while low < high:
        mid = (low + high) // 2
        if count_logratio(mid, prior, base, slope, observed) > 0:
            low = mid + 1
        else:
            high = mid
    mode = low
    peak = count_logweight(mode, prior, base, slope, observed)
    lo, hi = mode, mode
    while lo > 0:
        ratio = count_logratio(lo - 1, prior, base, slope, observed)
        if ratio > 0:
            log_bound = count_logweight(lo - 1, prior, base, slope, observed) - peak - np.log(-np.expm1(-ratio))
            if log_bound < np.log(1e-12):
                break
        lo -= 1
    while True:
        ratio = count_logratio(hi + 1, prior, base, slope, observed)
        if ratio < 0:
            log_bound = count_logweight(hi + 1, prior, base, slope, observed) - peak - np.log(-np.expm1(ratio))
            if log_bound < np.log(1e-12):
                break
        hi += 1
        if hi - lo > 100000:
            return -np.inf, -3, -3, 0., 0.
    total, first, nonzero = 0., 0., 0.
    for n in range(lo, hi + 1):
        weight = np.exp(count_logweight(n, prior, base, slope, observed) - peak)
        total += weight
        first += n * weight
        if n > 0:
            nonzero += weight
    return peak + np.log(total), lo, hi, first / total, nonzero / total


@njit(cache=True)
def poisson_from_uniform(mean, uniform):
    """Inverse CDF for ordinary latent rates; one draw couples perturbed fits.

    The large-rate path retains NumPy's exact Poisson generator. The ordinary
    path only truncates a residual tail below floating-point CDF accuracy.
    """
    if mean >= 600:
        return np.random.poisson(mean)
    probability = np.exp(-mean)
    cumulative = probability
    count = 0
    maximum = max(128, int(mean + 15 * np.sqrt(mean) + 100))
    while cumulative < uniform and count < maximum:
        count += 1
        probability *= mean / count
        cumulative += probability
    return count


@njit(cache=True)
def posterior_count(prior, base, slope, observed, ll, lo, hi, posterior_mean):
    if lo == -1:
        return 0
    if lo == -2:
        return poisson_from_uniform(posterior_mean, np.random.random())
    u, cumulative = np.random.random(), 0.
    for n in range(lo, hi + 1):
        cumulative += np.exp(count_logweight(n, prior, base, slope, observed) - ll)
        if cumulative >= u:
            return n
    return hi  # bounded discarded posterior tail, not a clipped prior count


def physical_arrays(model, observed_mean):
    target = np.array([m.target for m in model.modes], dtype=np.int64)
    source = np.array([m.source for m in model.modes], dtype=np.int64)
    mass = np.array([m.mass for m in model.modes])
    decay = np.exp(-1 / np.array([m.tau for m in model.modes]))
    anchor_source = np.array([a.source for a in model.anchors], dtype=np.int64)
    slow_decay = np.exp(-1 / np.array([a.tau for a in model.anchors]))
    beta = np.array(model.beta).reshape(len(target), len(anchor_source))
    gates = gate_values(np.log([m.tau for m in model.modes]), np.log([a.tau for a in model.anchors]))
    matrix = core_branching(model)
    latent_mean = (model.background[2] + np.dot(matrix[2, :2], observed_mean)) / max(1 - matrix[2, 2], .01)
    initial_mean = np.r_[observed_mean, latent_mean]
    return (np.array(model.background), target, source, mass, decay, anchor_source,
            slow_decay, beta * gates, initial_mean, model.link == 'bounded')


@njit(cache=True)
def ou_proposal(previous, decay, sd, loading, variance, background, kernel_mean, observed):
    """Defensive prior/guided Gaussian mixture with exact correction.

    Guidance uses an approximate Fisher curvature. It does not approximate the
    target likelihood: log prior/proposal is included in each particle weight.
    Standardized innovations avoid ill-conditioned long-scale OU variances.
    """
    q_count = len(decay)
    prior_mean = decay * previous
    center = np.zeros(q_count)
    precision = np.ones(q_count)
    # Reuse the Newton workspace across iterations; the proposal law and
    # random draw order are unchanged.
    gradient = np.empty(q_count)
    for iteration in range(9):
        for q in range(q_count):
            gradient[q] = -center[q]
        precision[:] = 1.
        for side in range(2):
            exponent = -.5 * variance[side]
            for q in range(q_count):
                exponent += loading[q, side] * (prior_mean[q] + sd[q] * center[q])
            bg = background[side] * np.exp(exponent)
            rate = max(kernel_mean[side] + bg, 1e-300)
            for q in range(q_count):
                coefficient = loading[q, side] * sd[q]
                gradient[q] += coefficient * bg * (observed[side] / rate - 1)
                precision[q] += coefficient ** 2 * bg ** 2 / rate
        if iteration == 8:
            break  # curvature at the actual proposal center
        maximum_step = 0.
        for q in range(q_count):
            delta = max(-2., min(2., gradient[q] / precision[q]))
            if q_count > 1:
                delta *= .5
            center[q] += delta
            maximum_step = max(maximum_step, abs(delta))
        if maximum_step < 1e-4:
            break
    proposed = np.zeros(q_count)
    prior_component = np.random.random() < .25
    log_prior, log_guided = 0., 0.
    for q in range(q_count):
        proposal_sd = 1 / np.sqrt(precision[q])
        innovation = np.random.normal() if prior_component else center[q] + proposal_sd * np.random.normal()
        proposed[q] = prior_mean[q] + sd[q] * innovation
        log_prior -= .5 * innovation ** 2
        log_guided -= .5 * ((innovation - center[q]) / proposal_sd) ** 2 + np.log(proposal_sd)
    # Gaussian constants are identical and cancel. Prior mass bounds the
    # importance ratio by four even when the approximate guided mode is poor.
    proposal_log = np.logaddexp(np.log(.25) + log_prior, np.log(.75) + log_guided)
    return proposed, log_prior - proposal_log


@njit(cache=True, nogil=True)
def adapted_filter(y, valid, background, target, source, mass, decay, anchor_source, slow_decay, beta_gate, initial_mean, saturating, particles, seed, z_decay=np.empty(0), z_loading=np.empty((0, 3)), origins=np.empty(0, dtype=np.int64), initial_fast=np.empty(0), initial_slow=np.empty(0)):
    """Fully adapted L proposal and guided OU proposal with correct weights.

    states hold the next traces with the preceding L jump still deferred. Its
    prior rate and jump slope are known from the previous posterior state. At t,
    integrate that count against Y[t], record the *prior* predictive mean, then
    infer/sample it. Sampling never influences the already-recorded forecast.
    """
    np.random.seed(seed)
    n, k, a = len(y), len(target), len(anchor_source)
    states = np.zeros((particles, k))
    slow = np.zeros((particles, a))
    slopes = np.zeros((particles, k))
    slow_slopes = np.zeros((particles, a))
    pending = np.zeros(particles)
    prediction = np.zeros((n, 2))
    losses = np.zeros(n)
    ess = np.zeros(n)
    latent_rate = np.zeros(n)
    previous_count = np.zeros(n)
    active = np.zeros(n)
    base = np.zeros((particles, 2))
    slope_obs = np.zeros((particles, 2))
    ll = np.zeros(particles)
    count_normalizer = np.zeros(particles)
    lo, hi = np.zeros(particles, dtype=np.int64), np.zeros(particles, dtype=np.int64)
    means, probabilities = np.zeros(particles), np.zeros(particles)
    next_states, next_slow = np.zeros_like(states), np.zeros_like(slow)
    next_slopes, next_slow_slopes = np.zeros_like(slopes), np.zeros_like(slow_slopes)
    next_pending = np.zeros_like(pending)
    weights = np.zeros(particles)
    bounded = np.zeros(a)
    z_count = len(z_decay)
    z_state = np.zeros((particles, z_count))
    z_proposal = np.zeros_like(z_state)
    next_z = np.zeros_like(z_state)
    z_sd = np.sqrt(1 - z_decay ** 2)
    z_variance = np.sum(z_loading ** 2, axis=0)
    current_background = np.zeros((particles, 3))
    snapshot_fast = np.zeros((len(origins), particles, k))
    snapshot_slopes = np.zeros_like(snapshot_fast)
    snapshot_slow = np.zeros((len(origins), particles, a))
    snapshot_slow_slopes = np.zeros_like(snapshot_slow)
    snapshot_pending = np.zeros((len(origins), particles))
    snapshot_z = np.zeros((len(origins), particles, z_count))
    snapshots = (snapshot_fast, snapshot_slopes, snapshot_slow, snapshot_slow_slopes, snapshot_pending, snapshot_z)
    origin_index = 0
    reset = True
    for t in range(n):
        if not valid[t]:
            reset = True
            continue
        if reset:
            for p in range(particles):
                for j in range(k):
                    states[p, j] = initial_fast[j] if t == 0 and len(initial_fast) else initial_mean[source[j]]
                    slopes[p, j] = 0.
                for q in range(a):
                    slow[p, q] = initial_slow[q] if t == 0 and len(initial_slow) else initial_mean[anchor_source[q]]
                    slow_slopes[p, q] = 0.
                pending[p] = 0.
                for q in range(z_count):
                    z_state[p, q] = np.random.normal()
            reset = False
        if origin_index < len(origins) and origins[origin_index] == t:
            snapshot_fast[origin_index] = states
            snapshot_slopes[origin_index] = slopes
            snapshot_slow[origin_index] = slow
            snapshot_slow_slopes[origin_index] = slow_slopes
            snapshot_pending[origin_index] = pending
            snapshot_z[origin_index] = z_state
            origin_index += 1
        for p in range(particles):
            base[p] = 0.
            slope_obs[p] = 0.
            for j in range(k):
                if target[j] < 2:
                    base[p, target[j]] += mass[j] * states[p, j]
                    slope_obs[p, target[j]] += mass[j] * slopes[p, j]
            for side in range(2):
                # Prior forecast is independent of observation-guided draws.
                exponent = 0.
                for q in range(z_count):
                    exponent += z_loading[q, side] * z_decay[q] * z_state[p, q] - .5 * (z_loading[q, side] * z_decay[q]) ** 2
                prediction[t, side] += (background[side] * np.exp(exponent) + base[p, side] + slope_obs[p, side] * pending[p]) / particles
            correction = 0.
            if z_count:
                z_proposal[p], correction = ou_proposal(z_state[p], z_decay, z_sd, z_loading, z_variance, background,
                                                       base[p] + slope_obs[p] * pending[p], y[t])
            current_background[p] = background
            for side in range(3):
                exponent = -.5 * z_variance[side]
                for q in range(z_count):
                    exponent += z_loading[q, side] * z_proposal[p, q]
                current_background[p, side] *= np.exp(exponent)
            base[p] += current_background[p, :2]
            if not np.isfinite(pending[p]) or pending[p] > 1e10 or not np.isfinite(base[p]).all():
                ll[p], lo[p], hi[p], means[p], probabilities[p] = -np.inf, -3, -3, 0., 0.
            else:
                ll[p], lo[p], hi[p], means[p], probabilities[p] = integrated_count(pending[p], base[p], slope_obs[p], y[t])
                count_normalizer[p] = ll[p]
                ll[p] += correction
        peak = np.max(ll)
        # A numerical support failure is not zero model probability. Reject the
        # evaluation rather than silently deleting that particle from the law.
        if not np.isfinite(peak) or np.any(lo == -3):
            losses[t:] = np.inf
            prediction[t:] = np.nan
            return prediction, losses, ess, latent_rate, previous_count, active, snapshots
        for p in range(particles):
            weights[p] = np.exp(ll[p] - peak)
        normalization = weights.sum()
        losses[t] = -(peak + np.log(normalization / particles))
        weights /= normalization
        ess[t] = 1 / np.sum(weights ** 2)
        previous_count[t] = np.dot(weights, means)
        active[t] = np.dot(weights, probabilities)
        u, ancestor, cumulative = np.random.random() / particles, 0, weights[0]
        for p in range(particles):
            while cumulative < u + p / particles and ancestor < particles - 1:
                ancestor += 1
                cumulative += weights[ancestor]
            # Count posterior normalizer excludes the independent OU importance
            # correction already included in the ancestor weight.
            count = posterior_count(pending[ancestor], base[ancestor], slope_obs[ancestor], y[t], count_normalizer[ancestor], lo[ancestor], hi[ancestor], means[ancestor])
            rate = current_background[ancestor, 2]
            for q in range(z_count):
                next_z[p, q] = z_proposal[ancestor, q]
            for j in range(k):
                if target[j] == 2:
                    rate += mass[j] * (states[ancestor, j] + slopes[ancestor, j] * count)
            next_pending[p] = rate
            latent_rate[t] += rate / particles
            for q in range(a):
                raw = (slow[ancestor, q] + slow_slopes[ancestor, q] * count) / max(initial_mean[anchor_source[q]], 1e-8)
                bounded[q] = np.tanh(raw) if saturating else raw
            for j in range(k):
                factor = 1.
                for q in range(a):
                    factor += beta_gate[j, q] * bounded[q]
                full_state = states[ancestor, j] + slopes[ancestor, j] * count
                next_states[p, j] = decay[j] * full_state
                next_slopes[p, j] = 0.
                if source[j] == 2:
                    next_slopes[p, j] = (1 - decay[j]) * factor
                else:
                    next_states[p, j] += (1 - decay[j]) * factor * y[t, source[j]]
            for q in range(a):
                full_slow = slow[ancestor, q] + slow_slopes[ancestor, q] * count
                next_slow[p, q] = slow_decay[q] * full_slow
                next_slow_slopes[p, q] = 0.
                if anchor_source[q] == 2:
                    next_slow_slopes[p, q] = 1 - slow_decay[q]
                else:
                    next_slow[p, q] += (1 - slow_decay[q]) * y[t, anchor_source[q]]
        states[:] = next_states
        slow[:] = next_slow
        slopes[:] = next_slopes
        slow_slopes[:] = next_slow_slopes
        pending[:] = next_pending
        z_state[:] = next_z
    return prediction, losses, ess, latent_rate, previous_count, active, snapshots


def filter_model(model, y, valid, observed_mean, particles, seed, origins=None, initial_fast=None, initial_slow=None):
    origin_array = np.empty(0, dtype=np.int64) if origins is None else np.asarray(origins, dtype=np.int64)
    if len(origin_array) and (np.any(np.diff(origin_array) <= 0) or origin_array[0] < 0 or origin_array[-1] >= len(y) or not np.all(valid[origin_array])):
        raise ValueError('Forecast origins must be unique, sorted and covered')
    z_decay, z_loading = np.empty(0), np.empty((0, 3))
    if model.diffusions:
        z_decay = np.exp(-1 / np.array([z.tau for z in model.diffusions]))
        z_loading = np.array([z.loading for z in model.diffusions])
    first_fast = np.empty(0) if initial_fast is None else np.asarray(initial_fast, dtype=np.float64)
    first_slow = np.empty(0) if initial_slow is None else np.asarray(initial_slow, dtype=np.float64)
    if len(first_fast) not in (0, len(model.modes)) or len(first_slow) not in (0, len(model.anchors)):
        raise ValueError('Initial state layout differs from the model')
    result = adapted_filter(y, valid, *physical_arrays(model, observed_mean), particles, seed, z_decay, z_loading, origin_array, first_fast, first_slow)
    return result[:6] if origins is None else result


def seed_latent(base, tau=10., strength=.1, mu_latent=.1, out_gain=None, in_gain=None):
    outgoing = strength if out_gain is None else out_gain
    incoming = strength if in_gain is None else in_gain
    new = tuple(Mode(i, j, .2 if i == j else outgoing if j == 2 else incoming, tau)
                for i in range(3) for j in range(3) if i == 2 or j == 2)
    modes = base.modes + new
    return LatentModel((*base.background, mu_latent), modes, beta=tuple(() for _ in modes))


def fit_joint(seed, corpus, output: Path, particles=32, iterations=120, learning_rate=.025, optimizer_seed=20261003):
    """All-parameter simultaneous-perturbation stochastic likelihood optimization.

    Every step perturbs/updates the full raw parameter vector, not fixed observed
    kernels. Common random numbers couple the two likelihood evaluations.
    Checkpoints include optimizer moments and RNG state for exact continuation.
    Stopping/candidate selection uses validation only; iteration exhaustion is
    not described as demonstrated numerical convergence.
    """
    output.mkdir(parents=True, exist_ok=True)
    layout, raw = seed.layout(), seed.raw()
    path = output / 'joint-progress.json'
    config = {'version': 1, 'particles': particles, 'iterations': iterations, 'learning_rate': learning_rate,
              'optimizer_seed': optimizer_seed, 'seed_model': seed.serialize(), 'corpus': corpus.identity}
    rng = np.random.default_rng(optimizer_seed)
    first, second = np.zeros_like(raw), np.zeros_like(raw)
    start_iteration = 0
    history = []
    if path.exists():
        progress = json.loads(path.read_text())
        if progress['config'] != config:
            raise ValueError('Joint fitting settings changed; use a new candidate directory')
        raw = np.array(progress['raw'])
        first, second = np.array(progress['first_moment']), np.array(progress['second_moment'])
        start_iteration = progress['next_iteration']
        history = progress['history']
        rng.bit_generator.state = progress['rng_state']
    n = corpus.train_end
    def train_loss(theta, filter_seed):
        model = decode(theta, layout, seed.link)
        result = filter_model(model, corpus.y[:n], corpus.valid[:n], corpus.initial_mean, particles, filter_seed)
        return float(result[1][corpus.score[:n]].mean())
    best_raw = raw.copy()
    best_validation = np.inf
    selected_path = output / 'best-validation.json'
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        best_raw, best_validation = np.array(selected['raw']), selected['validation_nll_without_factorial']
    begin = time.monotonic()
    for step in range(start_iteration, iterations):
        direction = rng.choice(np.array([-1., 1.]), len(raw))
        c = .08 / (1 + step / 20) ** .101
        random_seed = int(rng.integers(1, 2**30))
        plus = train_loss(raw + c * direction, random_seed)
        minus = train_loss(raw - c * direction, random_seed)
        if np.isfinite(plus) and np.isfinite(minus):
            gradient = (plus - minus) * direction / (2 * c)
            first = .9 * first + .1 * gradient
            second = .999 * second + .001 * gradient ** 2
            correction = first / (1 - .9 ** (step + 1)) / (np.sqrt(second / (1 - .999 ** (step + 1))) + 1e-8)
            raw -= learning_rate / (1 + step / 50) ** .602 * correction
            raw = np.clip(raw, *raw_bounds(layout))
        row = {'iteration': step, 'plus_train_loss': plus, 'minus_train_loss': minus,
               'seconds_elapsed': time.monotonic() - begin}
        if step % 10 == 0 or step + 1 == iterations:
            model = decode(raw, layout, seed.link)
            result = filter_model(model, corpus.y, corpus.valid, corpus.initial_mean, particles * 2, 71391)
            mask = corpus.score.copy()
            mask[:n] = False
            validation = float(result[1][mask].mean())
            row['validation_nll_without_factorial'] = validation
            row['train_nll_without_factorial'] = float(result[1][:n][corpus.score[:n]].mean())
            row['ess_validation_mean_fraction'] = float(result[2][mask].mean() / (2 * particles))
            if validation < best_validation:
                best_raw, best_validation = raw.copy(), validation
                save_json(selected_path, {'raw': best_raw.tolist(), 'model': model.serialize(), 'validation_nll_without_factorial': validation,
                                         'train_nll_without_factorial': row['train_nll_without_factorial'], 'iteration': step})
        history.append(row)
        save_json(path, {'config': config, 'raw': raw.tolist(), 'first_moment': first.tolist(), 'second_moment': second.tolist(),
                         'next_iteration': step + 1, 'rng_state': rng.bit_generator.state, 'history': history})
        print(json.dumps(row), flush=True)
    model = decode(best_raw, layout, seed.link)
    save_json(output / 'fit-result.json', {'model': model.serialize(), 'raw': best_raw.tolist(),
              'validation_nll_without_factorial': best_validation, 'joint_parameters': len(raw),
              'optimizer': 'all-parameter SPSA with Adam moments and validation stopping',
              'convergence_established': False, 'iterations': iterations,
              'spectral_radius': radius(core_branching(model)), 'particles_fit': particles})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--base', type=Path, default=Path('data/benchmarks/hawkes-joint-1s-2026-10-03/M0/selected-M0.json'))
    parser.add_argument('--output', type=Path, default=Path('data/benchmarks/hawkes-joint-1s-2026-10-03/ML/initial-10s'))
    parser.add_argument('--particles', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=120)
    parser.add_argument('--tau', type=float, default=10.)
    parser.add_argument('--mu-latent', type=float, default=.1)
    parser.add_argument('--out-gain', type=float, default=.1)
    parser.add_argument('--in-gain', type=float, default=.1)
    args = parser.parse_args()
    corpus = load_corpus(args.cache, 3600)
    base = Model.deserialize(json.loads(args.base.read_text())['model'])
    fit_joint(seed_latent(base, args.tau, mu_latent=args.mu_latent, out_gain=args.out_gain, in_gain=args.in_gain),
              corpus, args.output, args.particles, args.iterations)


if __name__ == '__main__':
    main()
