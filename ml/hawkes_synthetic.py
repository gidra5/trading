"""Independent, seeded recovery control for joint Hawkes architecture selection."""
from pathlib import Path
import json
import numpy as np
from numba import njit
from hawkes_research import Corpus, Model, Mode, architecture_search, branching, counts, save_json


@njit(cache=True)
def simulate(theta, target, source, length, initial_mean, seed):
    np.random.seed(seed)
    k = len(target)
    d = np.exp(-1 / np.exp(theta[2 + k:]))
    state = np.zeros(k)
    for j in range(k):
        state[j] = initial_mean[source[j]]
    y = np.zeros((length, 2))
    for t in range(length):
        rate = theta[:2].copy()
        for j in range(k):
            rate[target[j]] += theta[2 + j] * state[j]
        for side in range(2):
            y[t, side] = np.random.poisson(rate[side])
        for j in range(k):
            state[j] = d[j] * state[j] + (1 - d[j]) * y[t, source[j]]
    return y


def main():
    output = Path('data/benchmarks/hawkes-joint-1s-2026-10-03/synthetic-M0')
    true = Model((.5, .5), (Mode(0, 0, .35, .8), Mode(0, 0, .5, 80.),
                           Mode(1, 1, .4, 2.), Mode(1, 1, .4, 50.)))
    stationary_mean = np.linalg.solve(np.eye(2) - branching(true), np.array(true.background))
    seed, length = 20261003, 75000
    generated = simulate(*true.arrays(), length, stationary_mean, seed)
    ntrain, nvalidation = int(.6 * length), int(.8 * length)
    y = generated[:nvalidation]
    valid = np.ones(len(y), dtype=bool)
    score = valid.copy()
    score[:1000] = False
    mean = y[:ntrain][score[:ntrain]].mean(axis=0)
    for array in (y, valid, score, mean):
        array.setflags(write=False)
    identity = {'synthetic_seed': seed, 'true_model': true.serialize(), 'full_seconds': length,
                'train_scored_seconds': int(score[:ntrain].sum()), 'validation_scored_seconds': nvalidation - ntrain}
    corpus = Corpus(y, valid, score, ntrain, nvalidation, mean, identity)
    fitted = architecture_search(corpus, output, max_modes=2, maxiter=500)
    diag = []
    recovered = True
    for side in range(2):
        modes = sorted([m for m in fitted.modes if m.target == side and m.source == side and m.mass > .05], key=lambda m: m.tau)
        expected = [m for m in true.modes if m.target == side]
        result = {'side': side, 'truth': [m.tau for m in expected], 'estimated': [m.tau for m in modes],
                  'two_distinct_modes': len(modes) == 2 and modes[1].tau / modes[0].tau > 5}
        if len(modes) == 2:
            result['timescales_within_factor_two'] = all(.5 < estimate.tau / reference.tau < 2 for estimate, reference in zip(modes, expected))
        else:
            result['timescales_within_factor_two'] = False
        recovered &= result['two_distinct_modes'] and result['timescales_within_factor_two']
        diag.append(result)
    cross_mass = sum(m.mass for m in fitted.modes if m.target != m.source)
    result = {'seed': seed, 'true': true.serialize(), 'fitted': fitted.serialize(), 'edge_counts': counts(fitted),
              'diagnostics': diag, 'cross_mass': cross_mass, 'recovery_passed': bool(recovered and cross_mass < .1)}
    save_json(output / 'recovery.json', result)
    print('RECOVERY', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
