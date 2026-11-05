"""Joint stochastic likelihood fitting with exact observed-history traces.

Training windows sample a fixed partition uniformly, with score-count weighting.
All parameters are perturbed and updated together. Known observed-history traces
are reconstructed from their full prefix at each perturbed parameter vector.
Latent warmup expands with continuous scales and feedback; models whose latent
anchors or modulation prevent this reconstruction use the full particle prefix.
Full training/validation likelihoods select checkpoints. Test counts are absent.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from numba import njit
from hawkes_latent import LatentModel, core_branching, decode, filter_model, physical_arrays, raw_bounds
from hawkes_research import load_corpus, radius, save_json


@njit(cache=True, nogil=True)
def observed_initializations(y, valid, source, decay, anchor_source, slow_decay, beta_gate, initial_mean, saturating, starts):
    """Exact causal traces for observed inputs with observed-only anchors."""
    state = np.array([initial_mean[s] for s in source])
    slow = np.array([initial_mean[s] for s in anchor_source])
    fast_out = np.zeros((len(starts), len(source)))
    slow_out = np.zeros((len(starts), len(anchor_source)))
    index, reset = 0, True
    for t in range(int(starts[-1]) + 1):
        if valid[t] and reset:
            for j in range(len(source)):
                state[j] = initial_mean[source[j]]
            for a in range(len(anchor_source)):
                slow[a] = initial_mean[anchor_source[a]]
            reset = False
        if starts[index] == t:
            fast_out[index], slow_out[index] = state, slow
            index += 1
            if index == len(starts):
                break
        if not valid[t]:
            reset = True
            continue
        for j in range(len(source)):
            if source[j] == 2:
                continue  # unknown inputs use stationary initialization + burn
            factor = 1.
            for a in range(len(anchor_source)):
                raw = slow[a] / max(initial_mean[anchor_source[a]], 1e-8)
                factor += beta_gate[j, a] * (np.tanh(raw) if saturating else raw)
            state[j] = decay[j] * state[j] + (1 - decay[j]) * factor * y[t, source[j]]
        for a in range(len(anchor_source)):
            slow[a] = slow_decay[a] * slow[a] + (1 - slow_decay[a]) * y[t, anchor_source[a]]
    return fast_out, slow_out


@njit(cache=True, nogil=True)
def observed_slow_upper(y, valid, anchor_source, slow_decay, initial_mean):
    slow = np.array([initial_mean[s] for s in anchor_source])
    upper = np.ones(len(anchor_source))
    reset = True
    for t in range(len(y)):
        if not valid[t]:
            reset = True
            continue
        if reset:
            for a in range(len(anchor_source)):
                slow[a] = initial_mean[anchor_source[a]]
            reset = False
        for a in range(len(anchor_source)):
            upper[a] = max(upper[a], slow[a] / max(initial_mean[anchor_source[a]], 1e-8))
            slow[a] = slow_decay[a] * slow[a] + (1 - slow_decay[a]) * y[t, anchor_source[a]]
    return upper


def conditioning_context(model, observed_mean, observed_max, minimum=3600, anchor_upper=None):
    """Conservative decay context or full-prefix fallback, never a scale cap.

    Forty decay constants make unconditional initial-state influence tiny for
    the conditional linear hidden subsystem. Posterior forgetting is additionally
    checked against full filters; this is an optimizer approximation, not a
    replacement likelihood used for final selection/evaluation.
    """
    if any(a.source == 2 for a in model.anchors):
        return None
    arrays = physical_arrays(model, observed_mean)
    beta_gate = arrays[7]
    gain = 0.
    for j, mode in enumerate(model.modes):
        if mode.source == 2 and mode.target == 2:
            factor = 1.
            for a, anchor in enumerate(model.anchors):
                upper = 1. if model.link == 'bounded' else float(anchor_upper[a]) if anchor_upper is not None else max(1., observed_max[anchor.source] / observed_mean[anchor.source])
                factor += beta_gate[j, a] * upper
            gain += mode.mass * factor
    if gain >= .99:
        return None
    hidden_tau = max([m.tau for m in model.modes if m.source == 2] + [0.])
    diffusion_tau = max([z.tau for z in model.diffusions] + [0.])
    return max(minimum, int(np.ceil(40 * hidden_tau / max(1 - gain, .01))), int(np.ceil(40 * diffusion_tau)))


def partitions(score, width):
    blocks = [(s, min(s + width, len(score))) for s in range(0, len(score), width)]
    return np.array([(s, e) for s, e in blocks if np.any(score[s:e])], dtype=np.int64)


def coordinate_scales(layout):
    target, _, anchor_source, q = layout
    k, a = len(target), len(anchor_source)
    scale = np.ones(3 + 2 * k + k * a + a + 4 * q)
    # Direct beta is not log-parameterized: a .08 perturbation was much larger
    # than fitted active coefficients and dominated unrelated scale gradients.
    scale[3 + 2 * k:3 + 2 * k + k * a] = .03
    if q:
        scale[-3 * q:] = .2
    return scale


def model_context(model, corpus, minimum_context):
    n = corpus.train_end
    arrays = physical_arrays(model, corpus.initial_mean)
    observed_max = np.max(corpus.y[:n], axis=0)
    upper = None
    if model.anchors and not any(a.source == 2 for a in model.anchors):
        upper = observed_slow_upper(corpus.y[:n], corpus.valid[:n], arrays[5], arrays[6], arrays[8])
    return conditioning_context(model, corpus.initial_mean, observed_max, minimum_context, upper)


def window_losses(model, corpus, selected_blocks, particles, seeds, minimum_context=3600, context_override='auto'):
    n = corpus.train_end
    arrays = physical_arrays(model, corpus.initial_mean)
    context = model_context(model, corpus, minimum_context) if context_override == 'auto' else context_override
    blocks = np.asarray(selected_blocks, dtype=np.int64)
    end = int(np.max(blocks[:, 1]))
    if context is not None and np.sum(blocks[:, 1] - np.maximum(0, blocks[:, 0] - context)) >= end:
        context = None  # one exact prefix is cheaper than overlapping warmups
    if context is None:
        result = filter_model(model, corpus.y[:end], corpus.valid[:end], corpus.initial_mean, particles, int(seeds[0]))
        values = np.array([np.sum(result[1][s:e][corpus.score[s:e]]) for s, e in blocks])
        return values, {'full_prefix': True, 'particle_seconds': end}
    begins = np.maximum(0, blocks[:, 0] - context)
    starts, reverse = np.unique(begins, return_inverse=True)
    fast, slow = observed_initializations(corpus.y[:n], corpus.valid[:n], arrays[2], arrays[4], arrays[5], arrays[6], arrays[7], arrays[8], arrays[9], starts)
    values = np.zeros(len(blocks))
    particle_seconds = 0
    for j, ((s, e), begin, seed) in enumerate(zip(blocks, begins, seeds)):
        result = filter_model(model, corpus.y[begin:e], corpus.valid[begin:e], corpus.initial_mean, particles, int(seed),
                              initial_fast=fast[reverse[j]], initial_slow=slow[reverse[j]])
        values[j] = np.sum(result[1][s - begin:e - begin][corpus.score[s:e]])
        particle_seconds += int(e - begin)
    return values, {'full_prefix': False, 'context_seconds': context, 'particle_seconds': particle_seconds}


def fit_configuration(seed, corpus, particles=16, iterations=1500, learning_rate=.025, optimizer_seed=20261003,
                      batch_windows=4, window_seconds=1024, checkpoint_every=50, checkpoint_particles=32, minimum_context=3600):
    return {'version': 'joint-minibatch-v1', 'seed_model': seed.serialize(), 'corpus': corpus.identity,
            'particles': particles, 'iterations': iterations, 'learning_rate': learning_rate, 'optimizer_seed': optimizer_seed,
            'batch_windows': batch_windows, 'window_seconds': window_seconds, 'checkpoint_every': checkpoint_every,
            'checkpoint_particles': checkpoint_particles, 'minimum_context': minimum_context,
            'coordinate_scales': coordinate_scales(seed.layout()).tolist()}


def fit_joint(seed, corpus, output, particles=16, iterations=1500, learning_rate=.025, optimizer_seed=20261003,
              batch_windows=4, window_seconds=1024, checkpoint_every=50, checkpoint_particles=32, minimum_context=3600):
    output.mkdir(parents=True, exist_ok=True)
    layout, raw = seed.layout(), seed.raw()
    scales = coordinate_scales(layout)
    blocks = partitions(corpus.score[:corpus.train_end], window_seconds)
    total_scored = int(np.sum(corpus.score[:corpus.train_end]))
    if not len(blocks) or particles < 2 or batch_windows < 1:
        raise ValueError('Need scored training blocks and at least two particles')
    config = fit_configuration(seed, corpus, particles, iterations, learning_rate, optimizer_seed,
                               batch_windows, window_seconds, checkpoint_every, checkpoint_particles, minimum_context)
    progress_path = output / 'joint-progress.json'
    rng = np.random.default_rng(optimizer_seed)
    first, second, history = np.zeros_like(raw), np.zeros_like(raw), []
    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['config'] != config:
            raise ValueError('Joint minibatch configuration changed; use a new output directory')
        raw = np.array(progress['raw'])
        first, second = np.array(progress['first_moment']), np.array(progress['second_moment'])
        rng.bit_generator.state = progress['rng_state']
        history, start = progress['history'], progress['next_iteration']
    best_path = output / 'best-validation.json'
    best_raw, best_validation = raw.copy(), np.inf
    if best_path.exists():
        best = json.loads(best_path.read_text())
        best_raw, best_validation = np.array(best['raw']), best['validation_nll_without_factorial']
    validation_mask = corpus.score.copy()
    validation_mask[:corpus.train_end] = False
    begin = time.monotonic()
    def checkpoint(iteration, theta):
        nonlocal best_raw, best_validation
        model = decode(theta, layout, seed.link)
        result = filter_model(model, corpus.y, corpus.valid, corpus.initial_mean, checkpoint_particles, 71391)
        training = float(result[1][:corpus.train_end][corpus.score[:corpus.train_end]].mean())
        validation = float(result[1][validation_mask].mean())
        row = {'iteration': iteration, 'train_nll_without_factorial': training, 'validation_nll_without_factorial': validation,
               'full_data_checkpoint': True, 'checkpoint_particles': checkpoint_particles,
               'ess_validation_mean_fraction': float(np.mean(result[2][validation_mask]) / checkpoint_particles),
               'seconds_elapsed': time.monotonic() - begin}
        if validation < best_validation:
            best_raw, best_validation = theta.copy(), validation
            save_json(best_path, {'model': model.serialize(), 'raw': theta.tolist(), **row})
        return row
    if start == 0:
        row = checkpoint(-1, raw)
        history.append(row)
        print(json.dumps(row), flush=True)
    for step in range(start, iterations):
        direction = rng.choice(np.array([-1., 1.]), len(raw))
        c = .04 / (1 + step / 100) ** .101
        chosen = blocks[rng.integers(len(blocks), size=batch_windows)]
        random_seeds = rng.integers(1, 2**30, size=batch_windows)
        plus_model = decode(raw + c * scales * direction, layout, seed.link)
        minus_model = decode(raw - c * scales * direction, layout, seed.link)
        left_context, right_context = model_context(plus_model, corpus, minimum_context), model_context(minus_model, corpus, minimum_context)
        shared_context = None if left_context is None or right_context is None else max(left_context, right_context)
        if shared_context is not None and np.sum(chosen[:, 1] - np.maximum(0, chosen[:, 0] - shared_context)) >= np.max(chosen[:, 1]):
            shared_context = None
        plus_values, plus_context = window_losses(plus_model, corpus, chosen, particles, random_seeds, minimum_context, shared_context)
        minus_values, minus_context = window_losses(minus_model, corpus, chosen, particles, random_seeds, minimum_context, shared_context)
        plus = float(np.sum(plus_values) * len(blocks) / (batch_windows * total_scored))
        minus = float(np.sum(minus_values) * len(blocks) / (batch_windows * total_scored))
        if np.isfinite(plus) and np.isfinite(minus):
            gradient = (plus - minus) * direction / (2 * c * scales)
            first = .9 * first + .1 * gradient
            second = .999 * second + .001 * gradient ** 2
            correction = first / (1 - .9 ** (step + 1)) / (np.sqrt(second / (1 - .999 ** (step + 1))) + 1e-8)
            raw -= learning_rate / (1 + step / 500) ** .602 * scales * correction
            raw = np.clip(raw, *raw_bounds(layout))
        row = {'iteration': step, 'plus_stochastic_train_loss': plus, 'minus_stochastic_train_loss': minus,
               'plus_context': plus_context, 'minus_context': minus_context, 'seconds_elapsed': time.monotonic() - begin}
        if (step + 1) % checkpoint_every == 0 or step + 1 == iterations:
            row.update(checkpoint(step, raw))
        history.append(row)
        save_json(progress_path, {'config': config, 'raw': raw.tolist(), 'first_moment': first.tolist(), 'second_moment': second.tolist(),
                  'rng_state': rng.bit_generator.state, 'history': history, 'next_iteration': step + 1})
        print(json.dumps(row), flush=True)
    model = decode(best_raw, layout, seed.link)
    save_json(output / 'fit-result.json', {'model': model.serialize(), 'raw': best_raw.tolist(), 'joint_parameters': len(raw),
              'validation_nll_without_factorial': best_validation, 'particles_fit': particles,
              'optimizer': 'All-parameter preconditioned SPSA on weighted chronological windows; full-data checkpoint selection',
              'iterations': iterations, 'convergence_established': False, 'spectral_radius': radius(core_branching(model)),
              'approximation': 'Adaptive latent-context optimizer only; full conditional likelihood used for checkpoint comparison'})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--particles', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--learning-rate', type=float, default=.025)
    parser.add_argument('--batch-windows', type=int, default=4)
    parser.add_argument('--window-seconds', type=int, default=1024)
    parser.add_argument('--checkpoint-every', type=int, default=50)
    parser.add_argument('--checkpoint-particles', type=int, default=32)
    args = parser.parse_args()
    model = LatentModel.deserialize(json.loads(args.model.read_text())['model'])
    fit_joint(model, load_corpus(args.cache, 3600), args.output, args.particles, args.iterations, args.learning_rate,
              batch_windows=args.batch_windows, window_seconds=args.window_seconds,
              checkpoint_every=args.checkpoint_every, checkpoint_particles=args.checkpoint_particles)


if __name__ == '__main__':
    main()
