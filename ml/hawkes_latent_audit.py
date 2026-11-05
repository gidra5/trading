"""Fixed-parameter particle convergence and latent identifiability diagnostics.

This audit never refits parameters or opens the test period. A numerical error
smaller than the model-comparison margin must be demonstrated separately from
optimization convergence. Particle ESS alone does not establish either.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from hawkes_latent import LatentModel, core_branching, filter_model, physical_arrays, raw_bounds
from hawkes_research import count_metrics, load_corpus, radius, save_json


def parameter_diagnostics(model, observed_mean):
    raw = model.raw()
    lower, upper = raw_bounds(model.layout())
    position = (raw - lower) / (upper - lower)
    pairs = []
    for j, mode in enumerate(model.modes):
        for other in model.modes[j + 1:]:
            if (mode.source, mode.target) == (other.source, other.target):
                ratio = max(mode.tau, other.tau) / min(mode.tau, other.tau)
                if ratio < 1.25:
                    pairs.append({'source': mode.source, 'target': mode.target, 'tau1': mode.tau, 'tau2': other.tau, 'ratio': ratio})
    initial = physical_arrays(model, observed_mean)[8]
    diffusion_pairs = []
    for q, diffusion in enumerate(model.diffusions):
        for p, other in enumerate(model.diffusions[q + 1:], q + 1):
            ratio = max(diffusion.tau, other.tau) / min(diffusion.tau, other.tau)
            if ratio < 1.25:
                diffusion_pairs.append({'first': q, 'second': p, 'tau1': diffusion.tau, 'tau2': other.tau, 'ratio': ratio})
    return {'core_branching_matrix': core_branching(model).tolist(), 'core_spectral_radius': radius(core_branching(model)),
            'stability_claim': 'Linear core only; linear amplitude modulation requires separate nonlinear stability checks' if model.anchors and model.link == 'linear' else 'Normalized integrated branching bound',
            'parameters_near_lower_bound': np.flatnonzero(position < .001).tolist(),
            'parameters_near_upper_bound': np.flatnonzero(position > .999).tolist(),
            'near_duplicate_kernel_scales': pairs,
            'near_duplicate_diffusion_scales': diffusion_pairs,
            'diffusion_loadings': [list(z.loading) for z in model.diffusions],
            'diffusion_identifiability_note': 'Components are exchangeable and each loading vector has an arbitrary overall sign; equal-timescale Gaussian components can be rotated without changing their covariance',
            'subsecond_kernel_scales': [m.tau for m in model.modes if m.tau < 1],
            'subsecond_interpretation': 'Discrete one-second decay parameters; event-time resolution is unavailable',
            'additive_core_conditional_latent_mean': float(initial[2]),
            'mean_latent_to_observed_core_contribution': [sum(m.mass * initial[2] for m in model.modes if m.source == 2 and m.target == side) for side in range(2)],
            'latent_identifiability_established': False,
            'remaining_identifiability_checks': 'Multistart, profile sensitivity, recovery controls and model-order refinement'}


def audit(model, corpus, output, particle_counts=(32, 128, 512), seeds=(71391, 91337)):
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'model': model.serialize(), 'corpus': corpus.identity, 'particle_counts': list(particle_counts),
                'seeds': list(seeds), 'purpose': 'Fixed-parameter numerical convergence; not architecture or optimizer convergence', 'test_evaluated': False}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Audit candidate/configuration changed; use another output directory')
    save_json(protocol_path, protocol)
    mask = corpus.score.copy()
    mask[:corpus.train_end] = False
    factorial = gammaln(corpus.y[mask] + 1).sum(axis=1)
    rows = []
    for particles in particle_counts:
        for seed in seeds:
            path = output / f'particles-{particles}-seed-{seed}.json'
            if path.exists():
                row = json.loads(path.read_text())
            else:
                begin = time.monotonic()
                result = filter_model(model, corpus.y, corpus.valid, corpus.initial_mean, particles, seed)
                finite = bool(np.isfinite(result[1][mask]).all())
                row = {'particles': particles, 'seed': seed, 'seconds': time.monotonic() - begin, 'finite': finite,
                       'validation_nll_without_factorial': float(np.mean(result[1][mask])) if finite else None,
                       'validation_predictive_marginal_nll': float(np.mean(result[1][mask] + factorial)) if finite else None,
                       'ess_mean_fraction': float(np.mean(result[2][mask] / particles)),
                       'ess_below_10pct_fraction': float(np.mean(result[2][mask] < .1 * particles)),
                       'latent_prior_rate_mean': float(np.mean(result[3][mask])),
                       'previous_latent_count_posterior_mean': float(np.mean(result[4][mask])),
                       'previous_latent_active_probability_mean': float(np.mean(result[5][mask])),
                       **count_metrics(corpus.y[mask], result[0][mask])}
                save_json(path, row)
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (output / 'particle_metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    convergence = []
    for p in particle_counts:
        loss = [r['validation_nll_without_factorial'] for r in rows if r['particles'] == p and r['finite']]
        convergence.append({'particles': p, 'replicates': len(loss), 'mean': float(np.mean(loss)) if loss else None,
                            'seed_standard_deviation': float(np.std(loss, ddof=1)) if len(loss) > 1 else None})
    save_json(output / 'summary.json', {'particle_convergence': convergence,
              'convergence_established': False, 'interpretation': 'Compare successive particle counts and seed uncertainty with the observed model margin; no fixed threshold automatically proves convergence',
              'parameters': parameter_diagnostics(model, corpus.initial_mean)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particles', type=int, nargs='+', default=[32, 128, 512])
    parser.add_argument('--seeds', type=int, nargs='+', default=[71391, 91337])
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    args = parser.parse_args()
    model = LatentModel.deserialize(json.loads(args.model.read_text())['model'])
    audit(model, load_corpus(args.cache, 3600), args.output, args.particles, args.seeds)


if __name__ == '__main__':
    main()
