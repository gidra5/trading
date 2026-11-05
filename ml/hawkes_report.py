"""Build the human-readable report from completed, frozen five-model results.

The scientific decision is supplied separately after reviewing validation,
optimization and numerical evidence. Ranking by one-step NLL alone is avoided.
No observations are opened and no models are fitted by this command.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from hawkes_research import HORIZONS, save_json

LABELS = ('M0', 'M2', 'ML', 'M2L', 'M2LZ')


def number(value):
    if value is None or value == '':
        return 'N/A'
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return f'{numeric:.8g}' if math.isfinite(numeric) else 'N/A'


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(number(cell) for cell in row) + ' |' for row in rows])


def latent_plot(results, output):
    """Plot inference diagnostics only, preserving gaps between scored minutes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
    fields = ('filtered_current_latent_intensity', 'previous_latent_count_posterior_mean',
              'previous_latent_active_probability')
    reference = None
    for label in LABELS[2:]:
        with np.load(results / label / 'test_latent_trace.npz', allow_pickle=False) as saved:
            seconds = saved['seconds']
            if reference is None:
                reference = seconds
            elif not np.array_equal(seconds, reference):
                raise ValueError('Latent traces have different scored times')
            minute = (seconds - seconds[0]) // 60
            counts = np.bincount(minute)
            for ax, field in zip(axes, fields):
                means = np.full(len(counts), np.nan)
                populated = counts > 0
                means[populated] = np.bincount(minute, weights=saved[field])[populated] / counts[populated]
                ax.plot((np.arange(len(counts)) + .5) / 60, means, label=label, linewidth=.8)
    for ax, title in zip(axes, ('Filtered current L intensity (counts/s)',
                               'Posterior preceding L count', 'Posterior preceding L active probability')):
        ax.set_ylabel(title)
        ax.grid(True, alpha=.2)
    axes[0].legend()
    axes[2].set_xlabel('Hours after first scored test bin')
    fig.suptitle('Inferred latent states: one-minute means; empty minutes remain gaps')
    fig.savefig(output, dpi=170)
    plt.close(fig)


def report(results, decision_path, output):
    completion = json.loads((results / 'evaluation-complete.json').read_text())
    if not (completion.get('all_five_models_evaluated') is True and completion.get('test_evaluated') is True
            and completion.get('parameters_changed_after_freeze') is False
            and completion.get('forecast_origins_identical') is True):
        raise ValueError('A completed frozen five-model evaluation is required')
    decision = json.loads(decision_path.read_text())
    if (not decision.get('rationale') or not decision.get('evidence_review')
            or not decision.get('keep_models') or not set(decision['keep_models']) <= set(LABELS)):
        raise ValueError('Supply the reviewed model decision and its scientific rationale')
    inputs = [results / 'evaluation-complete.json', decision_path]

    def read_csv(name):
        path = results / name
        inputs.append(path)
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    one_step = read_csv('final_test_metrics.csv')
    horizons = read_csv('horizon_metrics.csv')
    stability = read_csv('stability.csv')
    latent = read_csv('latent_diagnostics.csv')
    parameters = read_csv('fitted_scales.csv')
    selection = read_csv('model_selection.csv')
    for name, rows in [('one-step', one_step), ('stability', stability)]:
        if len(rows) != len(LABELS) or {row['model'] for row in rows} != set(LABELS):
            raise ValueError(f'Incomplete or duplicated {name} model coverage')
    if {row['model'] for row in latent} != set(LABELS[2:]):
        raise ValueError('All three latent diagnostic rows are required')
    for label in LABELS:
        selected = [row for row in horizons if row['model'] == label]
        if len(selected) != len(HORIZONS) or {int(row['horizon_seconds']) for row in selected} != set(HORIZONS):
            raise ValueError(f'{label}: incomplete or duplicated forecast horizons')
        if any(row.get('split') != 'test' for row in selected):
            raise ValueError('Report horizon metrics must use the frozen test split')
        if not any(row['model'] == label for row in parameters) or not any(row.get('family') == label for row in selection):
            raise ValueError(f'{label}: parameter or architecture evidence missing')
    protocol = completion['protocol']
    loss_path = results / 'comparison' / 'paired_one_step_differences.json'
    inputs.append(loss_path)
    loss_comparison = json.loads(loss_path.read_text())
    if loss_comparison['protocol']['split'] != 'test':
        raise ValueError('Paired likelihood comparisons must use the test split')
    lines = ['# Hawkes five-model experiment', '',
             'Retain: ' + ', '.join(decision['keep_models']) + '.', '', decision['rationale'], '',
             '## Evidence and limits', '', decision['evidence_review'], '',
             'All parameters were frozen before this evaluation. The fitted values below are the best found '
             'under the documented searches; they do not prove a global optimum. '
             'A latent channel is an inferred statistical component, without an established physical interpretation.', '',
             'Historical test inspection: ' + str(protocol.get('historical_test_inspection') or 'No disclosure recorded; review required.'), '',
             f"Test filtering uses {protocol['filter_particles']} particles and simulated forecasts use "
             f"{protocol['future_paths']} future paths, at {protocol['stride_seconds']} second origin spacing.", '',
             '## One-step test accuracy', '',
             'NLL includes factorial constants and integrates latent states for the latent models. '
             'MAE and R² use the predictive mean for total buy-plus-sell counts.', '',
             markdown_table(['Model', 'Marginal NLL', 'Total MAE', 'Total R²', 'Buy MAE', 'Sell MAE'],
                            [[row['model'], row['predictive_marginal_nll'], row['total_mae'], row['total_r2'],
                              row['buy_mae'], row['sell_mae']] for row in one_step]), '',
             '## Paired one-step likelihood differences', '',
             'Candidate minus reference marginal NLL; negative values favor the candidate. '
             'Pointwise 95% intervals resample contiguous scored-second blocks and stop at coverage gaps. '
             'They exclude parameter, selection and filtering uncertainty and have no multiplicity adjustment.', '',
             markdown_table(['Reference', 'Candidate', 'Scored seconds', 'NLL change', '95% interval'],
                            [[row['reference'], row['candidate'], row['scored_seconds'], row['marginal_nll_delta'],
                              ', '.join(number(value) for value in row['paired_block_interval95'])]
                             for row in loss_comparison['comparisons']]), '',
             '## Open-loop test forecasts', '',
             'Forecasts do not use future observations. A failed origin makes aggregate point accuracy '
             'unavailable; results do not condition on surviving paths. Horizon log scores are Poisson scores '
             'at the forecast mean, not exact multistep marginal likelihoods.', '']
    for label in LABELS:
        rows = sorted((row for row in horizons if row['model'] == label), key=lambda row: int(row['horizon_seconds']))
        cells = []
        for row in rows:
            failed = float(row.get('failed_origin_fraction') or 0)
            cells.append([row['horizon_seconds'], row['origins'],
                          None if failed else row['total_mae'], None if failed else row['total_r2'],
                          None if failed else row['total_correlation'],
                          row.get('mc_standard_error_mean'), failed,
                          row.get('interval90_coverage_available'),
                          None if failed else row.get('poisson_mean_log_score')])
        lines += [f'### {label}', '', markdown_table(
            ['Horizon (s)', 'Origins', 'Total MAE', 'Total R²', 'Correlation', 'Path MC SE', 'Failed origins',
             '90% coverage, available origins', 'Poisson mean log score'], cells), '']
    paired_path = results / 'comparison' / 'paired_horizon_differences.json'
    inputs.append(paired_path)
    paired = json.loads(paired_path.read_text())
    if paired['protocol']['split'] != 'test':
        raise ValueError('Paired comparisons must use the test split')
    lines += ['## Paired one-hour differences', '',
              'Differences are candidate minus reference. Fixed contiguous blocks stop at coverage gaps. '
              'These 95% sampling intervals are conditional on stored fits and forecast means; they exclude '
              'parameter, model-selection, filtering and future-path Monte Carlo uncertainty.', '',
              markdown_table(['Reference', 'Candidate', 'MAE change', 'MAE 95% interval', 'R² change', 'R² 95% interval'],
                             [[row['reference'], row['candidate'], row['mae_delta'],
                               ', '.join(number(v) for v in row['mae_delta_interval95']) if row['mae_delta_interval95'] else 'N/A',
                               row['r2_delta'],
                               ', '.join(number(v) for v in row['r2_delta_interval95']) if row['r2_delta_interval95'] else 'N/A']
                              for row in paired['comparisons'] if row['horizon_seconds'] == 3600]), '',
              '![Test horizon comparison](comparison/horizon_comparison.png)', '',
              '## Stability and inferred hidden states', '',
              markdown_table(['Model', 'Branching radius', 'Global bound', 'Hour failed origins'],
                             [[row['model'], row['spectral_radius'], row['global_branching_bound'],
                               row['one_hour_failed_origin_fraction']] for row in stability]), '',
              'A linear-core radius below one is not a global stability guarantee for unbounded positive '
              'amplitude modulation. OU immigration is exogenous. Its loadings have arbitrary common signs, '
              'and equal-timescale components may be rotated without changing their covariance.', '',
              markdown_table(['Model', 'Mean filtered L rate', 'Mean previous L count', 'Mean active probability',
                              'Fraction active probability > 0.01', 'Physical identity established'],
                             [[row['model'], row['filtered_current_latent_intensity_mean'],
                               row['previous_latent_count_posterior_mean'], row['previous_latent_active_probability_mean'],
                               row['fraction_bins_previous_latent_active_probability_gt_0_01'],
                               row['physical_latent_identifiability_established']] for row in latent]), '',
              'Filtered hidden estimates incorporate observations through the current bin. Correlations with '
              'current observed counts are descriptive inference diagnostics, not causal forecasting scores. '
              'Subsecond decay estimates belong to a one-second count model; they do not establish event-time resolution.', '',
              '![Inferred latent states](latent_states.png)', '',
              '## All fitted parameters', '',
              'B means buy, S sell, and L the hidden event channel. Each source→target edge has its own '
              'exponential count and continuous decay scales. Integrated mass is offspring per source event '
              'for the linear core. Background rates are counts per second; decay scales are seconds. '
              'Modulation coefficients and OU loadings are dimensionless.', '']
    edges = [(source, target) for target in 'BSL' for source in 'BSL']
    lines += [markdown_table(['Model'] + [f'{source}→{target} count' for source, target in edges], [
        [label] + [sum(row['kind'] == 'exponential' and row.get('source') == source and row.get('target') == target
                       for row in parameters if row['model'] == label)
                   if label in LABELS[2:] or 'L' not in (source, target) else 'N/A'
                   for source, target in edges] for label in LABELS]), '']
    for label in LABELS:
        rows = [row for row in parameters if row['model'] == label]
        lines += [f'### {label}', '']
        for kind in ('background', 'exponential', 'anchor', 'modulation', 'diffusion_loading'):
            matches = [row for row in rows if row['kind'] == kind]
            if not matches:
                continue
            keys = [key for key in matches[0] if key not in ('model', 'kind') and any(row.get(key, '') != '' for row in matches)]
            lines += [kind.replace('_', ' ').capitalize(), '',
                      markdown_table(keys, [[row.get(key) for key in keys] for row in matches]), '']
        model_path = results / label / 'frozen-model.json'
        inputs += [model_path, results / label / 'selection-audit.json', results / label / 'architecture-selection.csv']
        lines += [f'Full precision model: [{label}/frozen-model.json]({label}/frozen-model.json). '
                  f'Selection review: [{label}/selection-audit.json]({label}/selection-audit.json). '
                  f'Architecture history: [{label}/architecture-selection.csv]({label}/architecture-selection.csv).', '']
    lines += ['Complete architecture history: [model_selection.csv](model_selection.csv). '
              'Numerical and initialization reviews are recorded with the frozen selection evidence.', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    for label in LABELS[2:]:
        inputs.append(results / label / 'test_latent_trace.npz')
    latent_plot(results, results / 'latent_states.png')
    output.write_text('\n'.join(lines), encoding='utf-8')
    save_json(output.with_suffix('.provenance.json'), {'inputs': [
        {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs],
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fits_performed': False, 'observations_opened': False})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--decision', type=Path, required=True)
    args = parser.parse_args()
    print(report(args.results, args.decision, args.results / 'report.md'))


if __name__ == '__main__':
    main()
