"""Joint finite-difference refinement of all latent-model parameters.

All gradient coordinates use the same sampled training blocks, random numbers
and conservative hidden context. The complete parameter vector is then updated
together. This is an independent optimizer control for weak directions obscured
by the random-direction SPSA estimator, not an automatic convergence certificate.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from hawkes_latent import LatentModel, core_branching, decode, filter_model, raw_bounds
from hawkes_latent_minibatch import coordinate_scales, model_context, partitions, window_losses
from hawkes_latent_sensitivity import synthetic_corpus
from hawkes_research import load_corpus, radius, save_json


def probes(raw, widths, bounds):
    lower, upper = bounds
    for j, width in enumerate(widths):
        left, right = raw.copy(), raw.copy()
        left[j] = max(lower[j], raw[j] - width)
        right[j] = min(upper[j], raw[j] + width)
        yield j, left, right


def difference_gradient(raw, widths, bounds, objective):
    """Bound-aware coordinate differences; the caller supplies a shared law."""
    gradient = np.zeros_like(raw)
    for j, left, right in probes(raw, widths, bounds):
        gradient[j] = (objective(right) - objective(left)) / (right[j] - left[j])
    return gradient


def window_gradient(raw, layout, link, corpus, chosen, random_seeds, particles,
                    widths, total_blocks, total_scored, minimum_context=3600):
    bounds = raw_bounds(layout)
    models = [(j, left, right, decode(left, layout, link), decode(right, layout, link))
              for j, left, right in probes(raw, widths, bounds)]
    contexts = [model_context(model, corpus, minimum_context)
                for _, _, _, left, right in models for model in (left, right)]
    shared_context = None if any(c is None for c in contexts) else max(contexts)
    if shared_context is not None and np.sum(chosen[:, 1] - np.maximum(0, chosen[:, 0] - shared_context)) >= np.max(chosen[:, 1]):
        shared_context = None
    weight = total_blocks / (len(chosen) * total_scored)
    gradient = np.zeros_like(raw)
    differences = []
    for j, left, right, left_model, right_model in models:
        l, context_info = window_losses(left_model, corpus, chosen, particles, random_seeds,
                                        minimum_context, shared_context)
        r, _ = window_losses(right_model, corpus, chosen, particles, random_seeds,
                             minimum_context, shared_context)
        difference = float((np.sum(r) - np.sum(l)) * weight)
        gradient[j] = difference / (right[j] - left[j])
        differences.append(difference)
    return gradient, {'shared_context': shared_context, **context_info,
                      'coordinate_loss_differences': differences}


def fit_joint(seed, corpus, output, particles=128, iterations=100, learning_rate=.02,
              difference_width=.1, batch_windows=8, window_seconds=1024,
              checkpoint_every=10, checkpoint_particles=256, optimizer_seed=20261004,
              minimum_context=3600):
    output.mkdir(parents=True, exist_ok=True)
    layout, raw = seed.layout(), seed.raw()
    bounds = raw_bounds(layout)
    raw = np.clip(raw, *bounds)
    scales = coordinate_scales(layout)
    if particles < 2 or iterations < 1 or difference_width <= 0 or np.any(scales <= 0):
        raise ValueError('Need positive refinement settings and at least two particles')
    blocks = partitions(corpus.score[:corpus.train_end], window_seconds)
    total_scored = int(np.sum(corpus.score[:corpus.train_end]))
    if not len(blocks) or batch_windows < 1 or checkpoint_every < 1:
        raise ValueError('Need scored training blocks and positive checkpoint/window counts')
    config = {'version': 'joint-coordinate-difference-v1', 'seed_model': seed.serialize(), 'corpus': corpus.identity,
              'particles': particles, 'iterations': iterations, 'learning_rate': learning_rate,
              'difference_width': difference_width, 'batch_windows': batch_windows, 'window_seconds': window_seconds,
              'checkpoint_every': checkpoint_every, 'checkpoint_particles': checkpoint_particles,
              'optimizer_seed': optimizer_seed, 'minimum_context': minimum_context, 'coordinate_scales': scales.tolist()}
    progress_path = output / 'joint-progress.json'
    rng = np.random.default_rng(optimizer_seed)
    first, second, history, start = np.zeros_like(raw), np.zeros_like(raw), [], 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['config'] != config:
            raise ValueError('Refinement protocol changed; use a new output directory')
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

    def checkpoint(step):
        nonlocal best_raw, best_validation
        model = decode(raw, layout, seed.link)
        result = filter_model(model, corpus.y, corpus.valid, corpus.initial_mean, checkpoint_particles, 71391)
        train_loss = float(result[1][:corpus.train_end][corpus.score[:corpus.train_end]].mean())
        validation_loss = float(result[1][validation_mask].mean())
        row = {'iteration': step, 'train_nll_without_factorial': train_loss,
               'validation_nll_without_factorial': validation_loss, 'full_data_checkpoint': True,
               'checkpoint_particles': checkpoint_particles, 'seconds_elapsed': time.monotonic() - begin}
        if validation_loss < best_validation:
            best_raw, best_validation = raw.copy(), validation_loss
            save_json(best_path, {'model': model.serialize(), 'raw': raw.tolist(), **row})
        return row

    if start == 0:
        row = checkpoint(-1)
        history.append(row)
        print(json.dumps(row), flush=True)
    for step in range(start, iterations):
        chosen = blocks[rng.integers(len(blocks), size=batch_windows)]
        random_seeds = rng.integers(1, 2**30, size=batch_windows)
        widths = difference_width * scales / (1 + step / 100) ** .1
        gradient, info = window_gradient(raw, layout, seed.link, corpus, chosen, random_seeds,
                                         particles, widths, len(blocks), total_scored, minimum_context)
        if np.all(np.isfinite(gradient)):
            first = .9 * first + .1 * gradient
            second = .999 * second + .001 * gradient ** 2
            correction = first / (1 - .9 ** (step + 1)) / (np.sqrt(second / (1 - .999 ** (step + 1))) + 1e-8)
            raw = np.clip(raw - learning_rate / (1 + step / 100) ** .602 * scales * correction, *bounds)
        row = {'iteration': step, 'gradient': gradient.tolist(), 'finite_gradient': bool(np.isfinite(gradient).all()),
               'seconds_elapsed': time.monotonic() - begin, **info}
        if (step + 1) % checkpoint_every == 0 or step + 1 == iterations:
            row.update(checkpoint(step))
        history.append(row)
        save_json(progress_path, {'config': config, 'raw': raw.tolist(), 'first_moment': first.tolist(),
                  'second_moment': second.tolist(), 'rng_state': rng.bit_generator.state,
                  'history': history, 'next_iteration': step + 1})
        print(json.dumps(row), flush=True)
    model = decode(best_raw, layout, seed.link)
    save_json(output / 'fit-result.json', {'model': model.serialize(), 'raw': best_raw.tolist(),
              'joint_parameters': len(raw), 'validation_nll_without_factorial': best_validation,
              'particles_fit': particles, 'iterations': iterations, 'convergence_established': False,
              'optimizer': 'All-coordinate finite-difference gradient with simultaneous preconditioned Adam updates',
              'spectral_radius': radius(core_branching(model)),
              'approximation': 'Conservative shared latent context during optimization; full-data checkpoints select the fit'})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--particles', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--difference-width', type=float, default=.1)
    parser.add_argument('--batch-windows', type=int, default=8)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--checkpoint-particles', type=int, default=256)
    args = parser.parse_args()
    model = LatentModel.deserialize(json.loads(args.model.read_text())['model'])
    corpus = synthetic_corpus(args.synthetic) if args.synthetic else load_corpus(args.cache, 3600)
    fit_joint(model, corpus, args.output, args.particles, args.iterations, args.learning_rate,
              args.difference_width, args.batch_windows, checkpoint_every=args.checkpoint_every,
              checkpoint_particles=args.checkpoint_particles)


if __name__ == '__main__':
    main()
