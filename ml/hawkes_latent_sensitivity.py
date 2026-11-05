"""Fixed-parameter continuous-scale likelihood slices on a synthetic control.

All other parameters are held fixed: this is a sensitivity slice, not a profile
likelihood or joint refit. Paired block uncertainty helps distinguish a weak
scale from an obvious remaining optimization gap. Hidden counts are not read.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from hawkes_latent import LatentModel, filter_model
from hawkes_research import Corpus, Mode, save_json


def synthetic_corpus(directory):
    identity = json.loads((directory / 'protocol.json').read_text())['identity']
    with np.load(directory / 'generated.npz', allow_pickle=False) as stored:
        observed = stored['observed']
    n = len(observed)
    train_end, validation_end = int(.6 * n), int(.8 * n)
    y = observed[:validation_end]
    valid = np.ones(len(y), dtype=bool)
    score = valid.copy()
    score[:1000] = False
    mean = y[:train_end][score[:train_end]].mean(axis=0)
    for array in (y, valid, score, mean):
        array.setflags(write=False)
    return Corpus(y, valid, score, train_end, validation_end, mean, identity)


def paired_interval(difference, mask, width=1000):
    blocks = [(float(np.sum(difference[s:s + width][mask[s:s + width]])), int(np.sum(mask[s:s + width])))
              for s in range(0, len(difference), width) if np.any(mask[s:s + width])]
    sums, counts = np.array(blocks).T
    rng = np.random.default_rng(20261003)
    indices = rng.integers(len(blocks), size=(4000, len(blocks)))
    means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def plot_summary(output):
    """Plot stored likelihood slices without rerunning inference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((output / 'summary.json').read_text())
    if not summary['slice_complete']:
        raise ValueError('Wait for the complete frozen slice before plotting')
    protocol = summary['protocol']
    fitted = protocol['model']['modes'][protocol['mode_index']]['tau']
    truth = protocol['corpus']['true_model']['modes'][protocol['mode_index']]['tau']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout='constrained')
    for ax, split in zip(axes, ('training', 'validation')):
        for seed in protocol['seeds']:
            rows = [r for r in summary['rows'] if r['seed'] == seed]
            x = np.array([r['tau_seconds'] for r in rows])
            y = np.array([r[f'{split}_delta_from_fitted_scale'] for r in rows])
            interval = np.array([r[f'{split}_delta_block_interval95'] for r in rows])
            ax.errorbar(x, y * 1e5, yerr=np.maximum(0, np.array([y - interval[:, 0], interval[:, 1] - y])) * 1e5,
                        fmt='o-', capsize=3, markersize=4, label=f'Particle seed {seed}')
        ax.axhline(0, color='black', linewidth=.8)
        ax.axvline(fitted, color='gray', linestyle=':', label='Fitted scale')
        ax.axvline(truth, color='black', linestyle='--', label='Generating scale')
        ax.set_xscale('log')
        ax.set_xlabel('Sell-to-hidden exponential scale (seconds)')
        ax.set_ylabel('NLL change × 100000; negative is better')
        ax.set_title(split.capitalize())
        ax.grid(True, alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Weak hidden-scale slice: all other parameters held fixed', fontsize=12)
    fig.supxlabel('95% paired 1000-second block intervals, conditional on this fit; no joint refit or parameter confidence region', fontsize=8)
    path = output / 'likelihood_scale_slice.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def run(model, corpus, output, mode_index, scales, particles=256, seeds=(71391, 91337)):
    mode = model.modes[mode_index]
    scales = sorted(set([float(x) for x in scales] + [mode.tau]))
    protocol = {'model': model.serialize(), 'corpus': corpus.identity, 'mode_index': mode_index, 'scales': scales,
                'particles': particles, 'seeds': list(seeds), 'hidden_counts_read': False,
                'interpretation': 'Fixed-other-parameter likelihood slice; no joint reoptimization or parameter confidence region',
                'paired_uncertainty': '4000 bootstrap replicates of 1000-second blocks; conditional on fitted parameters', 'test_evaluated': False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Sensitivity protocol changed; use a new output directory')
    save_json(path, protocol)
    training_mask = corpus.score[:corpus.train_end]
    validation_mask = corpus.score[corpus.train_end:]
    rows = []
    for seed in seeds:
        base = filter_model(model, corpus.y, corpus.valid, corpus.initial_mean, particles, seed)[1]
        for tau in scales:
            result_path = output / f'tau-{tau:.12g}-seed-{seed}.json'
            if result_path.exists():
                row = json.loads(result_path.read_text())
            else:
                modes = tuple(Mode(m.target, m.source, m.mass, tau if j == mode_index else m.tau) for j, m in enumerate(model.modes))
                candidate = LatentModel(model.background, modes, model.anchors, model.beta, model.diffusions, model.link)
                loss = base if tau == mode.tau else filter_model(candidate, corpus.y, corpus.valid, corpus.initial_mean, particles, seed)[1]
                difference = loss - base
                row = {'tau_seconds': tau, 'seed': seed,
                       'training_nll_without_factorial': float(np.mean(loss[:corpus.train_end][training_mask])),
                       'validation_nll_without_factorial': float(np.mean(loss[corpus.train_end:][validation_mask])),
                       'training_delta_from_fitted_scale': float(np.mean(difference[:corpus.train_end][training_mask])),
                       'training_delta_block_interval95': paired_interval(difference[:corpus.train_end], training_mask),
                       'validation_delta_from_fitted_scale': float(np.mean(difference[corpus.train_end:][validation_mask])),
                       'validation_delta_block_interval95': paired_interval(difference[corpus.train_end:], validation_mask)}
                save_json(result_path, row)
            rows.append(row)
            save_json(output / 'summary.json', {'rows': rows, 'slice_complete': len(rows) == len(scales) * len(seeds), 'protocol': protocol})
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode-index', type=int, required=True)
    parser.add_argument('--scales', type=float, nargs='+', required=True)
    parser.add_argument('--particles', type=int, default=256)
    args = parser.parse_args()
    model = LatentModel.deserialize(json.loads(args.model.read_text())['model'])
    run(model, synthetic_corpus(args.synthetic), args.output, args.mode_index, args.scales, args.particles)


if __name__ == '__main__':
    main()
