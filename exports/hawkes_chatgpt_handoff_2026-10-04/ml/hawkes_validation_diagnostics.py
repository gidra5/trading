"""Join frozen provisional validation forecasts and diagnose point-forecast bias.

This report never fits or selects parameters and never reads test observations.
It retains failed forecasts and labels unfinished latent fits as provisional.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from hawkes_research import Model, branching, load_corpus, save_json
from hawkes_latent import LatentModel, core_branching
from hawkes_modulation import Modulated


def table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, cache):
    output = root / 'provisional-comparison'
    output.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(cache, 3600)
    origins_reference, actual_reference, horizons_reference = None, None, None
    one_step_rows, horizon_rows, diagnostic_rows, provenance = [], [], [], []
    for label in ('M0', 'M2', 'ML', 'M2L', 'M2LZ'):
        directory = root / label if label in ('M0', 'M2') else root / label / 'provisional-horizons'
        model_path = root / label / f'selected-{label}.json' if label in ('M0', 'M2') else directory / 'snapshot-model.json'
        model_value = json.loads(model_path.read_text())['model']
        model = Model.deserialize(model_value) if label == 'M0' else (Modulated.deserialize(model_value) if label == 'M2' else LatentModel.deserialize(model_value))
        metrics = json.loads((directory / 'validation_one_step_metrics.json').read_text())
        one_step_rows.append({'model': label, 'split': 'validation', 'provisional': True,
                              'predictive_marginal_nll': metrics.get('predictive_marginal_nll', metrics['poisson_mean_log_score']),
                              'poisson_mean_log_score': metrics['poisson_mean_log_score'],
                              'total_mae': metrics['total_mae'], 'total_r2': metrics['total_r2'],
                              'buy_mae': metrics['buy_mae'], 'sell_mae': metrics['sell_mae']})
        forecast_path = directory / 'validation_forecasts.npz'
        with np.load(forecast_path, allow_pickle=False) as stored:
            origins, actual, predicted, horizons = stored['origins'], stored['actual'], stored['prediction'], stored['horizons']
            if origins_reference is None:
                origins_reference, actual_reference, horizons_reference = origins, actual, horizons
            elif not (np.array_equal(origins, origins_reference) and np.array_equal(actual, actual_reference) and np.array_equal(horizons, horizons_reference)):
                raise ValueError(f'{label}: forecast origins/targets differ')
            if np.any(origins < corpus.train_end) or np.any(origins + horizons.max() > corpus.validation_end):
                raise ValueError('Forecast targets must be confined to validation')
            valid_future = np.r_[0, np.cumsum(~corpus.score)]
            if np.any(valid_future[origins + horizons.max()] != valid_future[origins]):
                raise ValueError('Unscored forecast target')
            for j, horizon in enumerate(horizons):
                finite = bool(np.isfinite(predicted[:, j]).all())
                diagnostic_rows.append({'model': label, 'horizon_seconds': int(horizon), 'origins': len(origins),
                                        'all_forecast_means_finite': finite,
                                        'actual_total_mean': float(actual[:, j].sum(axis=1).mean()),
                                        'predicted_total_mean': float(predicted[:, j].sum(axis=1).mean()) if finite else None,
                                        'signed_total_bias': float((predicted[:, j] - actual[:, j]).sum(axis=1).mean()) if finite else None,
                                        'buy_bias': float((predicted[:, j, 0] - actual[:, j, 0]).mean()) if finite else None,
                                        'sell_bias': float((predicted[:, j, 1] - actual[:, j, 1]).mean()) if finite else None,
                                        'relative_total_bias': float((predicted[:, j] - actual[:, j]).sum(axis=1).mean() / actual[:, j].sum(axis=1).mean()) if finite else None})
        with (directory / 'validation_horizon_metrics.csv').open(newline='') as stream:
            horizon_rows.extend(dict(row, provisional=True) for row in csv.DictReader(stream))
        if label == 'M0':
            stationary = np.linalg.solve(np.eye(2) - branching(model), model.background)
        elif label == 'ML':
            stationary = np.linalg.solve(np.eye(3) - core_branching(model), model.background)
        else:
            stationary = None
        provenance.append({'model': label, 'model_path': str(model_path), 'forecast_path': str(forecast_path),
                           'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
                           'forecast_sha256': hashlib.sha256(forecast_path.read_bytes()).hexdigest(),
                           'linear_stationary_rate': stationary.tolist() if stationary is not None else None})
    table(output / 'one_step_metrics.csv', one_step_rows)
    # Normalize the heterogeneous Monte Carlo diagnostics before joining.
    keys = list(dict.fromkeys(key for row in horizon_rows for key in row))
    table(output / 'horizon_metrics.csv', [{key: row.get(key) for key in keys} for row in horizon_rows])
    table(output / 'forecast_bias.csv', diagnostic_rows)
    train_mean = corpus.y[:corpus.train_end][corpus.score[:corpus.train_end]].mean(axis=0)
    validation_mean = corpus.y[corpus.train_end:][corpus.score[corpus.train_end:]].mean(axis=0)
    save_json(output / 'provenance.json', {'provisional': True, 'final_selection_complete': False, 'test_evaluated': False,
              'corpus': corpus.identity, 'train_mean': train_mean.tolist(), 'validation_mean': validation_mean.tolist(),
              'sources': provenance, 'failed_forecasts_policy': 'Retain failures; do not compute means conditional on successful origins',
              'interpretation': 'Latent order, optimization and particle verification remain incomplete; these fits are not final hierarchy winners'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    for label in ('M0', 'M2', 'ML', 'M2L', 'M2LZ'):
        rows = [r for r in horizon_rows if r['model'] == label]
        for ax, key in zip(axes[:2], ('total_mae', 'total_r2')):
            values = [float(r[key]) if r[key] else np.nan for r in rows]
            ax.plot([int(r['horizon_seconds']) for r in rows], values, 'o-', label=label, markersize=3)
        biases = [r for r in diagnostic_rows if r['model'] == label]
        axes[2].plot([r['horizon_seconds'] for r in biases], [r['relative_total_bias'] if r['relative_total_bias'] is not None else np.nan for r in biases], 'o-', label=label, markersize=3)
    for ax, title in zip(axes, ('Total-count MAE', 'Total-count R²', 'Relative total-count bias')):
        ax.set_xscale('log')
        ax.set_xlabel('Open-loop horizon (seconds)')
        ax.set_title(title)
        ax.grid(True, alpha=.2)
    axes[0].set_yscale('log')
    axes[1].legend(fontsize=8)
    axes[2].axhline(0., color='black', linewidth=.7)
    fig.suptitle('Provisional validation: latent order and convergence checks pending', fontsize=11)
    fig.savefig(output / 'forecast_comparison.png', dpi=170)
    plt.close(fig)
    for row in one_step_rows:
        print(json.dumps(row))
    for row in diagnostic_rows:
        if row['horizon_seconds'] == 3600:
            print(json.dumps(row))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/benchmarks/hawkes-joint-1s-2026-10-03'))
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    args = parser.parse_args()
    summarize(args.root, args.cache)


if __name__ == '__main__':
    main()
