"""Rerank a frozen shortlist of joint fits with matched particle evaluations.

The shortlist is frozen from a search CSV at first invocation. Subsequent search
progress does not alter this audit cohort. This is validation-period numerical
and paired sampling evidence, not optimization convergence or a final test.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from hawkes_latent import LatentModel, filter_model
from hawkes_research import load_corpus, save_json


def paired_interval(difference, mask, width=3600, replicates=4000):
    indices = np.flatnonzero(mask)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    blocks = [run[s:s + width] for run in runs for s in range(0, len(run), width)]
    if not blocks or not np.isfinite(difference[mask]).all():
        return None
    sums = np.array([np.sum(difference[block]) for block in blocks])
    counts = np.array([len(block) for block in blocks])
    rng = np.random.default_rng(20261003)
    samples = rng.integers(len(blocks), size=(replicates, len(blocks)))
    means = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def shortlist(search, top):
    if top < 1:
        raise ValueError('Shortlist size must be positive')
    source = search / 'model_selection.csv'
    with source.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: float(r['selection_score']))
    if not rows:
        raise ValueError('Search has no completed candidate fits')
    chosen = rows[:top]
    # Preserve the seed and unchanged-architecture controls even if an addition
    # seems better at the coarse checkpoint particle count.
    for operation in ('initial_0', 'joint_refit_control_additions'):
        control = next((r for r in rows if r['operation'] == operation), None)
        if control is not None and control not in chosen:
            chosen.append(control)
    entries = []
    for row in chosen:
        path = search / 'candidates' / row['candidate'] / 'fit-result.json'
        data = path.read_bytes()
        model = LatentModel.deserialize(json.loads(data)['model'])
        entries.append({'candidate': row['candidate'], 'operation': row['operation'],
                        'source': str(path.resolve()), 'source_sha256': hashlib.sha256(data).hexdigest(),
                        'coarse_selection_score': float(row['selection_score']),
                        'joint_parameters': len(model.raw()), 'model': model.serialize()})
    return entries, source.read_bytes()


def run(search, corpus, output, top=5, particle_counts=(128, 512), seeds=(71391, 91337), block_seconds=3600):
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    settings = {'search': str(search.resolve()), 'top': top, 'corpus': corpus.identity,
                'particle_counts': list(particle_counts), 'seeds': list(seeds), 'block_seconds': block_seconds}
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if any(protocol[key] != value for key, value in settings.items()):
            raise ValueError('Reranking settings changed; use a new output directory')
    else:
        entries, source = shortlist(search, top)
        protocol = {**settings, 'cohort': entries, 'search_csv_sha256': hashlib.sha256(source).hexdigest(),
                    'test_evaluated': False, 'fits_refined': False,
                    'purpose': 'Matched-particle reranking of a frozen completed-fit cohort',
                    'uncertainty': 'Paired contiguous scored-block bootstrap conditional on stored fits; excludes parameter and selection uncertainty'}
        (output / 'source-model-selection.csv').write_bytes(source)
        save_json(protocol_path, protocol)
    mask = corpus.score.copy()
    mask[:corpus.train_end] = False
    scored = int(np.sum(mask))
    reference = protocol['cohort'][0]['candidate']
    rows, comparisons = [], []
    for particles in particle_counts:
        for seed in seeds:
            reference_loss = None
            for entry in protocol['cohort']:
                candidate = entry['candidate']
                path = output / f'{candidate}-particles-{particles}-seed-{seed}.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as stored:
                        loss, seconds = stored['loss'], float(stored['seconds'])
                    if len(loss) != len(corpus.y):
                        raise ValueError('Cached likelihood length differs from the frozen corpus')
                else:
                    model = LatentModel.deserialize(entry['model'])
                    begin = time.monotonic()
                    loss = filter_model(model, corpus.y, corpus.valid, corpus.initial_mean, particles, seed)[1]
                    seconds = time.monotonic() - begin
                    np.savez_compressed(path, loss=loss, seconds=seconds)
                finite = bool(np.isfinite(loss[mask]).all())
                nll = float(loss[mask].mean()) if finite else None
                row = {'candidate': candidate, 'operation': entry['operation'], 'particles': particles,
                       'seed': seed, 'finite': finite, 'seconds': seconds,
                       'validation_nll_without_factorial': nll,
                       'joint_parameters': entry['joint_parameters'],
                       'selection_score': nll + entry['joint_parameters'] / scored if finite else None}
                rows.append(row)
                if candidate == reference:
                    reference_loss = loss
                difference = loss - reference_loss
                comparisons.append({'candidate': candidate, 'reference': reference, 'particles': particles, 'seed': seed,
                                    'validation_nll_delta': float(difference[mask].mean()) if finite and np.isfinite(reference_loss[mask]).all() else None,
                                    'paired_block_interval95': paired_interval(difference, mask, block_seconds)})
                save_json(output / 'replicates.json', {'rows': rows, 'comparisons': comparisons,
                          'cohort_complete': len(rows) == len(protocol['cohort']) * len(particle_counts) * len(seeds)})
                print(json.dumps(row), flush=True)
    ranking = []
    for particles in particle_counts:
        for entry in protocol['cohort']:
            matches = [r for r in rows if r['particles'] == particles and r['candidate'] == entry['candidate']]
            finite = all(r['finite'] for r in matches)
            losses = [r['validation_nll_without_factorial'] for r in matches]
            ranking.append({'candidate': entry['candidate'], 'operation': entry['operation'], 'particles': particles,
                            'replicates': len(matches), 'all_replicates_finite': finite,
                            'mean_validation_nll_without_factorial': float(np.mean(losses)) if finite else None,
                            'seed_standard_deviation': float(np.std(losses, ddof=1)) if finite and len(losses) > 1 else None,
                            'mean_selection_score': float(np.mean(losses)) + entry['joint_parameters'] / scored if finite else None,
                            'joint_parameters': entry['joint_parameters']})
    with (output / 'ranking.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    save_json(output / 'summary.json', {'ranking': ranking, 'comparisons': comparisons, 'cohort_complete': True,
              'architecture_selection_complete': False, 'optimization_convergence_established': False,
              'interpretation': 'Review changes across particle counts and seeds against the architecture margins; this shortlist does not cover unfinished search candidates'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--particles', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--seeds', type=int, nargs='+', default=[71391, 91337])
    parser.add_argument('--block-seconds', type=int, default=3600)
    args = parser.parse_args()
    run(args.search, load_corpus(args.cache, 3600), args.output, args.top, args.particles, args.seeds, args.block_seconds)


if __name__ == '__main__':
    main()
