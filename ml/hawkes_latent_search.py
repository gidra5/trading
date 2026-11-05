"""Independent mixture-order proposals and joint refits for ML/M2L/M2LZ.

Forward additions and backward deletions always refit every retained parameter.
Architecture plateau is distinct from numerical optimization convergence; the
serialized selection remains provisional when convergence is unverified.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from hawkes_latent import Diffusion, LatentModel, seed_latent
from hawkes_latent_minibatch import fit_configuration, fit_joint
from hawkes_fit_cache import shared_joint_fit
from hawkes_modulation import Anchor, Modulated
from hawkes_research import Mode, load_corpus, save_json


def lift_modulated(observed, tau=10., mu_latent=.1, out_gain=.1, in_gain=.1):
    latent = seed_latent(observed.base, tau, mu_latent=mu_latent, out_gain=out_gain, in_gain=in_gain)
    new_rows = tuple(tuple(.01 for _ in observed.anchors) for _ in range(len(latent.modes) - len(observed.base.modes)))
    return LatentModel(latent.background, latent.modes, observed.anchors, observed.beta + new_rows, link=observed.link)


def modulate_latent(model, anchors):
    if model.anchors:
        raise ValueError('Nested ML initialization must not already have anchors')
    return LatentModel(model.background, model.modes, tuple(anchors), tuple(tuple(0. for _ in anchors) for _ in model.modes), model.diffusions, model.link)


def seed_diffusion(model, tau, loading, observed_mean, core_fraction=1.):
    """A starting point only: every resulting parameter remains free."""
    from hawkes_latent import core_branching, physical_arrays
    fractions = np.broadcast_to(np.asarray(core_fraction, dtype=float), (3,))
    if np.any((fractions <= 0) | (fractions > 1)):
        raise ValueError('Initial core fractions must be in (0,1]')
    mean = physical_arrays(model, observed_mean)[8]
    background = np.array(model.background) + (1 - fractions) * (core_branching(model) @ mean)
    modes = tuple(Mode(m.target, m.source, float(fractions[m.target] * m.mass), m.tau) for m in model.modes)
    return LatentModel(tuple(float(x) for x in background), modes, model.anchors, model.beta,
                       model.diffusions + (Diffusion(tau, tuple(loading)),), model.link)


def mirror_observed(model):
    """Exchange buy/sell labels to construct an unconstrained alternate start."""
    permutation = (1, 0, 2)
    return LatentModel(tuple(model.background[i] for i in permutation),
                       tuple(Mode(permutation[m.target], permutation[m.source], m.mass, m.tau) for m in model.modes),
                       tuple(Anchor(permutation[a.source], a.tau) for a in model.anchors), model.beta,
                       tuple(Diffusion(z.tau, tuple(z.loading[i] for i in permutation)) for z in model.diffusions), model.link)


def edge_counts(model):
    return tuple(sum(m.target == target and m.source == source for m in model.modes) for target in range(3) for source in range(3))


def replace(model, *, modes=None, anchors=None, beta=None, diffusions=None):
    return LatentModel(model.background, model.modes if modes is None else modes,
                       model.anchors if anchors is None else anchors, model.beta if beta is None else beta,
                       model.diffusions if diffusions is None else diffusions, model.link)


def neighbors(model, label, max_modes=6, max_anchors=3, max_diffusions=4, additions=True, observed_mean=None):
    if label not in ('ML', 'M2L', 'M2LZ'):
        raise ValueError('Unknown latent hierarchy model')
    if label == 'ML' and (model.anchors or model.diffusions):
        raise ValueError('ML excludes modulation and diffusion')
    if label == 'M2L' and model.diffusions:
        raise ValueError('M2L excludes diffusion')
    beta = model.beta or tuple(tuple(0. for _ in model.anchors) for _ in model.modes)
    conditional_mean = None
    if observed_mean is not None:
        from hawkes_latent import physical_arrays
        conditional_mean = physical_arrays(model, observed_mean)[8]
    if additions:
        for target in range(3):
            for source in range(3):
                existing = [m for m in model.modes if (m.target, m.source) == (target, source)]
                if len(existing) >= max_modes:
                    continue
                # Scale the new component relative to its own directed mixture;
                # no common hidden amplitude ceiling constrains macro-events.
                mass = max(.05 * sum(m.mass for m in existing) if existing else .01, 1e-8)
                amplitudes = [('relative', mass)]
                if conditional_mean is not None:
                    # Rare hidden events can require a large offspring mass.
                    # A weak existing edge must not prevent another scale on
                    # that edge from starting at a materially different size.
                    calibrated = min(1e6 if source == 2 or target == 2 else 50.,
                                     max(1e-8, .05 * conditional_mean[target] / max(conditional_mean[source], 1e-8)))
                    if max(calibrated, mass) / min(calibrated, mass) > 4:
                        amplitudes.append(('rate_calibrated', calibrated))
                for initialization, amplitude in amplitudes:
                    for tau in (1., 100., 10000.):
                        mode = Mode(target, source, amplitude, tau)
                        yield f'add_mode_{target}_{source}_{tau}_{initialization}', replace(model, modes=model.modes + (mode,), beta=beta + (tuple(.01 for _ in model.anchors),))
        if label != 'ML':
            for source in range(3):
                if sum(a.source == source for a in model.anchors) >= max_anchors:
                    continue
                for tau in (30., 300., 3600.):
                    yield f'add_anchor_{source}_{tau}', replace(model, anchors=model.anchors + (Anchor(source, tau),), beta=tuple(row + (.01,) for row in beta))
        if label == 'M2LZ' and len(model.diffusions) < max_diffusions:
            for tau in (3., 100., 3600.):
                for name, loading in [('activity', (.1, .1, .1)), ('imbalance', (.1, -.1, 0.))]:
                    yield f'add_diffusion_{name}_{tau}', replace(model, diffusions=model.diffusions + (Diffusion(tau, loading),))
    else:
        for j in range(len(model.modes)):
            yield f'delete_mode_{j}', replace(model, modes=model.modes[:j] + model.modes[j + 1:], beta=beta[:j] + beta[j + 1:])
        for j in range(len(model.anchors)):
            yield f'delete_anchor_{j}', replace(model, anchors=model.anchors[:j] + model.anchors[j + 1:], beta=tuple(row[:j] + row[j + 1:] for row in beta))
        for j in range(len(model.diffusions)):
            yield f'delete_diffusion_{j}', replace(model, diffusions=model.diffusions[:j] + model.diffusions[j + 1:])


def search(initial_models, corpus, output, label='ML', particles=16, iterations=150, max_modes=6, max_anchors=3, max_diffusions=4, rounds=32, shared_fits=None, fit_many=None):
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'label': label, 'particles': particles, 'iterations_per_joint_refit': iterations, 'corpus': corpus.identity,
                'optimizer': 'Preconditioned all-parameter stochastic windows with full-data initial/final likelihood checkpoints',
                'max_modes_per_directed_edge': max_modes, 'max_anchors_per_source': max_anchors, 'max_diffusions': max_diffusions,
                'rounds': rounds,
                'amplitude_initializations': 'Relative mixture mass plus 5% of target training rate when starting scales differ by more than fourfold',
                'incumbent_controls': 'Unchanged architecture gets the same joint-refit budget on every addition/deletion comparison',
                'architecture_score': 'Validation marginal NLL without factorial + parameter count / validation scored seconds',
                'joint_parameters': 'All backgrounds, all directed masses and scales, all beta and anchor scales, all OU scales and loadings',
                'initial_models': [m.serialize() for m in initial_models], 'test_evaluated': False}
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Search protocol changed; use a new output directory')
    save_json(path, protocol)
    rows, history = [], []
    cache = {}
    def candidates(proposals):
        proposals = list(proposals)
        if fit_many is not None:
            pending = {}
            for operation, seed in proposals:
                key = hashlib.sha256(json.dumps(seed.serialize(), sort_keys=True).encode()).hexdigest()[:20]
                directory = output / 'candidates' / key
                if key not in cache and not (directory / 'fit-result.json').exists():
                    pending[key] = (seed, directory)
            fit_many(list(pending.values()))
        return [candidate(seed, operation) for operation, seed in proposals]
    def candidate(seed, operation):
        key = hashlib.sha256(json.dumps(seed.serialize(), sort_keys=True).encode()).hexdigest()[:20]
        if key not in cache:
            directory = output / 'candidates' / key
            result = directory / 'fit-result.json'
            if not result.exists():
                if fit_many is not None:
                    raise RuntimeError(f'Batch fitter did not produce {result}')
                def produce():
                    fit_joint(seed, corpus, directory, particles, iterations, checkpoint_every=iterations, checkpoint_particles=max(32, 2 * particles))
                if shared_fits is None:
                    produce()
                else:
                    config = fit_configuration(seed, corpus, particles, iterations, checkpoint_every=iterations, checkpoint_particles=max(32, 2 * particles))
                    shared_joint_fit(directory, config, shared_fits, produce)
            fitted = json.loads(result.read_text())
            model = LatentModel.deserialize(fitted['model'])
            row = {'candidate': key, 'operation': operation, 'edge_counts': str(edge_counts(model)),
                   'anchor_counts': str(tuple(sum(a.source == source for a in model.anchors) for source in range(3))),
                   'diffusion_count': len(model.diffusions), 'joint_parameters': fitted['joint_parameters'],
                   'validation_nll_without_factorial': fitted['validation_nll_without_factorial'],
                   'core_spectral_radius': fitted['spectral_radius'], 'optimizer_convergence_established': fitted['convergence_established']}
            row['selection_score'] = row['validation_nll_without_factorial'] + row['joint_parameters'] / corpus.identity['validation_scored_seconds']
            cache[key] = model, row
            rows.append(row)
            with (output / 'model_selection.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(row), flush=True)
        return cache[key]
    current, info = min(candidates([(f'initial_{j}', m) for j, m in enumerate(initial_models)]), key=lambda pair: pair[1]['selection_score'])
    reason = 'iteration_limit'
    for round_index in range(rounds):
        history.append({'round': round_index, 'candidate': info, 'model': current.serialize()})
        save_json(output / 'selection-history.json', history)
        changed = False
        additions = candidates([('joint_refit_control_additions', current), *neighbors(current, label, max_modes, max_anchors, max_diffusions, observed_mean=corpus.initial_mean)])
        reference = additions[0]
        if additions:
            best = min(additions, key=lambda pair: pair[1]['selection_score'])
            if best[1]['selection_score'] < info['selection_score'] - 1e-5:
                current, info = best
                changed = True
        while current.modes or current.anchors or current.diffusions:
            compared = candidates([('joint_refit_control_deletions', current), *neighbors(current, label, additions=False)])
            reference, deletions = compared[0], compared[1:]
            best = min(deletions, key=lambda pair: pair[1]['selection_score'])
            if best[1]['selection_score'] >= reference[1]['selection_score'] - 1e-5:
                if reference[1]['selection_score'] < info['selection_score'] - 1e-5:
                    current, info = reference
                    changed = True
                break
            if best[1]['selection_score'] >= info['selection_score'] - 1e-5:
                break
            current, info = best
            changed = True
        if not changed:
            reason = 'no_forward_or_backward_validation_improvement'
            break
    ceilings = any(x >= max_modes for x in edge_counts(current)) or any(sum(a.source == s for a in current.anchors) >= max_anchors for s in range(3)) or len(current.diffusions) >= max_diffusions
    optimizer_verified = all(row['optimizer_convergence_established'] for row in rows)
    save_json(output / f'selected-{label}.json', {'model': current.serialize(), 'fit': info, 'stop_reason': reason,
              'architecture_plateau': reason != 'iteration_limit', 'count_ceiling_reached': ceilings,
              'selection_complete': reason != 'iteration_limit' and not ceilings and optimizer_verified,
              'remaining': 'Particle accuracy, optimizer convergence, multistart/identifiability and frozen test evaluation must be established'})
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', choices=['ML', 'M2L', 'M2LZ'], default='ML')
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--particles', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--max-modes', type=int, default=6)
    parser.add_argument('--max-anchors', type=int, default=3)
    parser.add_argument('--max-diffusions', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=32)
    parser.add_argument('--shared-fits', type=Path)
    args = parser.parse_args()
    initial = [LatentModel.deserialize(json.loads(path.read_text())['model']) for path in args.initial]
    search(initial, load_corpus(args.cache, 3600), args.output, args.label, args.particles, args.iterations,
           args.max_modes, args.max_anchors, args.max_diffusions, args.rounds,
           args.shared_fits or args.output.parent.parent / 'shared-latent-fits')


if __name__ == '__main__':
    main()
