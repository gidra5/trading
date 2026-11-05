"""Known hidden-channel control: causal state recovery and all-parameter fits.

Observed training counts alone enter the fit. The generating hidden stream is
saved for subsequent diagnostics, never supplied to inference/optimization.
Failure to recover is reported, not converted into a passed control.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from numba import njit
from hawkes_latent import LatentModel, core_branching, decode, filter_model, physical_arrays
from hawkes_latent_minibatch import fit_joint
from hawkes_research import Corpus, Mode, save_json, safe_corr


@njit(cache=True)
def simulate(background, target, source, mass, decay, mean, length, seed):
    np.random.seed(seed)
    states = np.array([mean[s] for s in source])
    counts = np.zeros((length, 3))
    for t in range(length):
        rate = background.copy()
        for j in range(len(target)):
            rate[target[j]] += mass[j] * states[j]
        for side in range(3):
            counts[t, side] = np.random.poisson(rate[side])
        for j in range(len(target)):
            states[j] = decay[j] * states[j] + (1 - decay[j]) * counts[t, source[j]]
    return counts


def run(output, length=30000, iterations=300, particles=64, initial_models=(), learning_rate=.015):
    output.mkdir(parents=True, exist_ok=True)
    truth = LatentModel((.4, .6, .035), (Mode(0, 0, .15, 2.), Mode(1, 1, .2, 3.),
                         Mode(0, 2, 15., 1.5), Mode(1, 2, 10., 2.5),
                         Mode(2, 0, .001, 4.), Mode(2, 1, .002, 6.), Mode(2, 2, .3, 3.)),
                         beta=tuple(() for _ in range(7)))
    seed = 20261003
    mean = np.linalg.solve(np.eye(3) - core_branching(truth), truth.background)
    arrays = physical_arrays(truth, mean[:2])
    generated = simulate(*arrays[:5], mean, length, seed)
    train_end, validation_end = int(.6 * length), int(.8 * length)
    y = generated[:validation_end, :2].copy()
    valid = np.ones(len(y), dtype=bool)
    score = valid.copy()
    score[:1000] = False
    observed_mean = y[:train_end][score[:train_end]].mean(axis=0)
    for value in (y, valid, score, observed_mean):
        value.setflags(write=False)
    identity = {'synthetic_seed': seed, 'length': length, 'true_model': truth.serialize(),
                'train_scored_seconds': int(score[:train_end].sum()), 'validation_scored_seconds': int(score[train_end:].sum())}
    corpus = Corpus(y, valid, score, train_end, validation_end, observed_mean, identity)
    protocol = {'identity': identity, 'joint_parameters': len(truth.raw()),
              'hidden_counts_used_for_fitting': False, 'architecture': 'Known seven-edge single-mode architecture; order search is a separate control',
              'seeds': 'Refinement of previous joint fits' if initial_models else 'Two perturbed physical starting points',
              'initial_models': [m.serialize() for m in initial_models], 'all_parameters_free': True,
              'optimizer': 'All-parameter stochastic windows with full-data checkpoints',
              'iterations': iterations, 'particles': particles, 'learning_rate': learning_rate, 'batch_windows': 8}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Synthetic control protocol changed; use a new output directory')
    save_json(protocol_path, protocol)
    np.savez_compressed(output / 'generated.npz', observed=generated[:, :2], hidden=generated[:, 2])
    results = []
    for i in range(2):
        rng = np.random.default_rng(seed + i)
        initial = decode(truth.raw() + rng.normal(0, .5, len(truth.raw())), truth.layout())
        if initial_models:
            initial = initial_models[i]
            if initial.anchors or initial.diffusions or any(not np.array_equal(a, b) for a, b in zip(initial.layout()[:3], truth.layout()[:3])):
                raise ValueError('Refinement must use the same known synthetic architecture')
        fitted = fit_joint(initial, corpus, output / f'start-{i}', particles, iterations,
                           learning_rate=learning_rate, optimizer_seed=seed + i,
                           checkpoint_every=100, checkpoint_particles=max(64, particles), batch_windows=8)
        inferred = filter_model(fitted, corpus.y, corpus.valid, observed_mean, particles * 4, 71391)
        previous_hidden = generated[:validation_end - 1, 2]
        mask = corpus.score[1:].copy()
        mask[:train_end - 1] = False
        estimate = inferred[4][1:][mask]
        true = previous_hidden[mask]
        raw_difference = fitted.raw() - truth.raw()
        comparison = []
        for reference, candidate in zip(truth.modes, fitted.modes):
            comparison.append({'target': reference.target, 'source': reference.source,
                               'true_mass': reference.mass, 'mass': candidate.mass, 'true_tau': reference.tau, 'tau': candidate.tau,
                               'mass_ratio': candidate.mass / reference.mass, 'tau_ratio': candidate.tau / reference.tau})
        parameter_pass = bool(np.max(np.abs(raw_difference)) < np.log(2))
        correlation = safe_corr(true, estimate)
        row = {'start': i, 'true_model': truth.serialize(), 'fitted_model': fitted.serialize(),
               'parameter_comparison': comparison, 'background_ratios': (np.array(fitted.background) / truth.background).tolist(),
               'hidden_count_correlation': correlation, 'latent_count_mae': float(np.mean(np.abs(estimate - true))),
               'mean_hidden_count': float(np.mean(true)), 'mean_inferred_previous_count': float(np.mean(estimate)),
               'all_parameters_within_factor_two': parameter_pass,
               'recovery_passed': bool(parameter_pass and correlation is not None and correlation > .5),
               'optimizer_convergence_established': False}
        results.append(row)
        save_json(output / 'recovery.json', {'results': results, 'control_complete': len(results) == 2,
                  'interpretation': 'A successful recovery threshold is evidence for this known architecture only; stochastic optimization convergence and order recovery remain separate'})
        print('RECOVERY', json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/benchmarks/hawkes-joint-1s-2026-10-03/synthetic-ML'))
    parser.add_argument('--length', type=int, default=30000)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--particles', type=int, default=64)
    parser.add_argument('--initial', type=Path, nargs=2)
    parser.add_argument('--learning-rate', type=float, default=.015)
    args = parser.parse_args()
    initial = tuple(LatentModel.deserialize(json.loads(path.read_text())['model']) for path in args.initial) if args.initial else ()
    run(args.output, args.length, args.iterations, args.particles, initial, args.learning_rate)


if __name__ == '__main__':
    main()
