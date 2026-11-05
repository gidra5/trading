"""Compare frozen forecasts on identical origins, retaining numerical failures.

Paired uncertainty resamples contiguous origin blocks without crossing coverage
gaps. Results are conditional on fitted parameters and stored forecast means;
they exclude parameter, selection, filtering and future-path Monte Carlo error.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from hawkes_research import count_metrics, save_json


def origin_blocks(origins, stride=900, block_seconds=21600):
    if stride < 1 or block_seconds < stride or not len(origins) or np.any(np.diff(origins) <= 0):
        raise ValueError('Need ordered origins and positive block settings')
    runs = np.split(np.arange(len(origins)), np.flatnonzero(np.diff(origins) != stride) + 1)
    width = max(1, block_seconds // stride)
    return tuple(run[s:s + width] for run in runs for s in range(0, len(run), width))


def bootstrap_draws(origins, replicates=5000, stride=900, block_seconds=21600):
    blocks = origin_blocks(origins, stride, block_seconds)
    rng = np.random.default_rng(20261003)
    return tuple(np.concatenate([blocks[j] for j in indices])
                 for indices in rng.integers(len(blocks), size=(replicates, len(blocks))))


def paired_metrics(actual, reference, candidate, draws):
    if actual.shape != reference.shape or actual.shape != candidate.shape:
        raise ValueError('Paired forecast shapes differ')
    if not np.isfinite(actual).all():
        raise ValueError('Nonfinite observed forecast targets')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {'comparison_available': False, 'mae_delta': None, 'mae_delta_interval95': None,
                'r2_delta': None, 'r2_delta_interval95': None,
                'failure_policy': 'All-origin comparison unavailable; no survivor conditioning'}
    y, p, q = actual.sum(axis=1), reference.sum(axis=1), candidate.sum(axis=1)
    absolute = np.abs(q - y) - np.abs(p - y)
    squared = (p - y) ** 2 - (q - y) ** 2
    denominator = float(np.sum((y - y.mean()) ** 2))
    mae_samples = [float(absolute[draw].mean()) for draw in draws]
    r2_samples = []
    for draw in draws:
        variance_sum = float(np.sum((y[draw] - y[draw].mean()) ** 2))
        if variance_sum > 0:
            r2_samples.append(float(np.sum(squared[draw]) / variance_sum))
    return {'comparison_available': True, 'mae_delta': float(absolute.mean()),
            'mae_delta_interval95': np.quantile(mae_samples, [.025, .975]).tolist(),
            'r2_delta': float(np.sum(squared) / denominator) if denominator else None,
            'r2_delta_interval95': np.quantile(r2_samples, [.025, .975]).tolist() if r2_samples else None}


def compare_predictive_losses(directories, labels, output, split='test', replicates=5000, block_seconds=21600):
    """Paired marginal-NLL intervals using block sums, without huge resample arrays."""
    if len(directories) != len(labels) or len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError('Need at least two uniquely labelled loss traces')
    losses, provenance, seconds = [], [], None
    for directory, label in zip(directories, labels):
        path = directory / f'{split}_predictive_loss.npz'
        with np.load(path, allow_pickle=False) as stored:
            if seconds is None:
                seconds = stored['seconds']
            elif not np.array_equal(seconds, stored['seconds']):
                raise ValueError('Predictive loss traces have different scored times')
            loss = stored['marginal_nll']
        if loss.shape != seconds.shape or not np.isfinite(loss).all():
            raise ValueError('All scored predictive losses must be finite and aligned; no survivor conditioning')
        metric = json.loads((directory / f'{split}_one_step_metrics.json').read_text())
        reported = metric['predictive_marginal_nll'] if 'predictive_marginal_nll' in metric else metric['poisson_mean_log_score']
        if not np.isclose(loss.mean(), reported, rtol=1e-10, atol=1e-10):
            raise ValueError('Predictive loss trace disagrees with the reported marginal NLL')
        losses.append(loss)
        provenance.append({'model': label, 'path': str(path.resolve()),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    blocks = origin_blocks(seconds, stride=1, block_seconds=block_seconds)
    counts = np.array([len(block) for block in blocks])
    rng = np.random.default_rng(20261003)
    draws = rng.integers(len(blocks), size=(replicates, len(blocks)))
    denominators = counts[draws].sum(axis=1)
    comparisons = []
    for j in range(len(labels)):
        for k in range(j + 1, len(labels)):
            difference = losses[k] - losses[j]
            sums = np.array([difference[block].sum() for block in blocks])
            estimates = sums[draws].sum(axis=1) / denominators
            comparisons.append({'reference': labels[j], 'candidate': labels[k], 'scored_seconds': len(seconds),
                                'marginal_nll_delta': float(difference.mean()),
                                'paired_block_interval95': np.quantile(estimates, [.025, .975]).tolist()})
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'paired_one_step_differences.json', {
        'protocol': {'split': split, 'replicates': replicates, 'block_seconds': block_seconds,
                     'blocks': 'Fixed contiguous scored-second blocks; stop at coverage gaps; score-count weighting for unequal block sizes',
                     'uncertainty': 'Pointwise conditional sampling intervals; no multiplicity adjustment; parameter, selection and filtering uncertainty excluded',
                     'series': provenance}, 'comparisons': comparisons})
    return comparisons


def run(directories, labels, output, split='validation', replicates=5000, stride=900, block_seconds=21600):
    if len(directories) != len(labels) or len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError('Need at least two uniquely labelled forecast series')
    output.mkdir(parents=True, exist_ok=True)
    series, provenance, rows, one_step = [], [], [], []
    origins, actual, horizons = None, None, None
    for directory, label in zip(directories, labels):
        path = directory / f'{split}_forecasts.npz'
        with np.load(path, allow_pickle=False) as stored:
            if origins is None:
                origins, actual, horizons = stored['origins'], stored['actual'], stored['horizons']
            elif not (np.array_equal(origins, stored['origins']) and np.array_equal(actual, stored['actual']) and np.array_equal(horizons, stored['horizons'])):
                raise ValueError('Frozen forecast targets or origins differ')
            prediction = stored['prediction']
            quantiles = stored['total_count_quantile90'] if 'total_count_quantile90' in stored else None
        series.append(prediction)
        for j, horizon in enumerate(horizons):
            metrics = count_metrics(actual[:, j], prediction[:, j])
            coverage = None
            if quantiles is not None and np.isfinite(quantiles[:, j]).all():
                total = actual[:, j].sum(axis=1)
                coverage = float(np.mean((total >= quantiles[:, j, 0]) & (total <= quantiles[:, j, 2])))
            rows.append({'model': label, 'split': split, 'horizon_seconds': int(horizon), 'origins': len(origins),
                         **metrics, 'forecast_means_available_all_origins': bool(np.isfinite(prediction[:, j]).all()),
                         'interval90_coverage_all_origins': coverage})
        metrics_path = directory / f'{split}_one_step_metrics.json'
        metric = json.loads(metrics_path.read_text())
        one_step.append({'model': label, 'split': split,
                         'predictive_marginal_nll': metric['predictive_marginal_nll'] if 'predictive_marginal_nll' in metric else metric['poisson_mean_log_score'],
                         'total_mae': metric['total_mae'], 'total_r2': metric['total_r2']})
        provenance.append({'model': label, 'forecast': str(path.resolve()),
                           'forecast_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'one_step_metrics_sha256': hashlib.sha256(metrics_path.read_bytes()).hexdigest()})
    draws = bootstrap_draws(origins, replicates, stride, block_seconds)
    comparisons = []
    for j, reference in enumerate(series):
        for k in range(j + 1, len(series)):
            for h, horizon in enumerate(horizons):
                comparisons.append({'reference': labels[j], 'candidate': labels[k], 'horizon_seconds': int(horizon),
                                    **paired_metrics(actual[:, h], reference[:, h], series[k][:, h], draws)})
    protocol = {'split': split, 'replicates': replicates, 'stride_seconds': stride, 'block_seconds': block_seconds,
                'blocks': 'Fixed contiguous origin blocks; stop at coverage gaps; resample the original block count, weighting metrics by each draw actual origin count',
                'series': provenance, 'models_refitted': False,
                'horizon_log_score': 'Poisson score at forecast mean; not exact marginal multistep count likelihood',
                'uncertainty': 'Conditional paired sampling intervals; parameter, selection, filtering and Monte Carlo uncertainty excluded'}
    save_json(output / 'paired_horizon_differences.json', {'protocol': protocol, 'comparisons': comparisons})
    for name, data in [('horizon_metrics.csv', rows), ('one_step_metrics.csv', one_step)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), layout='constrained')
    for label in labels:
        selected = [r for r in rows if r['model'] == label]
        for ax, field in zip(axes, ('total_mae', 'total_r2', 'interval90_coverage_all_origins')):
            ax.plot(horizons, [r[field] if r[field] is not None else np.nan for r in selected], 'o-', label=label, markersize=3)
    for ax, title in zip(axes, ('Total-count MAE', 'Total-count R²', '90% interval coverage')):
        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_xlabel('Open-loop horizon (seconds)')
        ax.grid(True, alpha=.2)
    axes[0].set_yscale('log')
    axes[1].axhline(0., color='black', linewidth=.7)
    axes[1].legend(fontsize=8)
    axes[2].axhline(.9, color='black', linestyle='--', linewidth=.7)
    axes[2].set_ylim(0., 1.)
    fig.suptitle(f'Frozen {split} forecasts; model selection and numerical review remain separate', fontsize=11)
    fig.savefig(output / 'horizon_comparison.png', dpi=170)
    plt.close(fig)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directories', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    comparisons = run(args.directories, args.labels, args.output, args.split)
    for row in comparisons:
        if row['horizon_seconds'] == 3600:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
