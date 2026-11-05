"""Frozen five-model evaluation; no fitting, architecture choice or test tuning.

This entry point verifies parameter, selection-evidence and data fingerprints
before opening the test observations. A freeze manifest is created only after
the independent searches and their numerical/optimization audits are reviewed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from hawkes_research import HORIZONS, Model, _load_corpus, branching, evaluate_forecasts as linear_forecasts, radius, save_json
from hawkes_modulation import Modulated, gate_values, upper_branching
from hawkes_observed_rollouts import evaluate_forecasts as modulated_forecasts
from hawkes_latent import LatentModel, core_branching, physical_arrays
from hawkes_latent_audit import parameter_diagnostics
from hawkes_latent_forecasts import evaluate_forecasts as latent_forecasts
from hawkes_forecast_comparison import compare_predictive_losses, run as compare_forecasts

LABELS = ('M0', 'M2', 'ML', 'M2L', 'M2LZ')
REQUIREMENTS = ('joint_parameter_refits', 'independent_mixture_counts', 'continuous_scales',
                'optimization_stopping_review', 'particle_accuracy_review', 'validation_comparison')
SOURCE_BYTES = {name: Path(__file__).with_name(name).read_bytes() for name in
                ('hawkes_final_test.py', 'hawkes_research.py', 'hawkes_modulation.py',
                 'hawkes_observed_rollouts.py', 'hawkes_latent.py', 'hawkes_latent_forecasts.py', 'hawkes_latent_audit.py',
                 'hawkes_forecast_comparison.py')}


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_frozen_selection(path, cache):
    manifest = json.loads(path.read_text())
    if manifest.get('ready_for_test') is not True or manifest.get('final_test_evaluated_in_this_pipeline_before_freeze') is not False:
        raise ValueError('All selections must be frozen before final test evaluation')
    if set(manifest.get('models', {})) != set(LABELS):
        raise ValueError('Freeze must contain all five hierarchy models')
    models = {}
    for label in LABELS:
        entry = manifest['models'][label]
        verified = {}
        for key in ('model', 'selection_audit', 'selection_csv'):
            file = Path(entry[key])
            file = file if file.is_absolute() else path.parent / file
            if sha256(file) != entry[f'{key}_sha256']:
                raise ValueError(f'{label}: frozen {key} changed')
            verified[key] = file
        evidence = json.loads(verified['selection_audit'].read_text())
        if evidence.get('ready_for_test') is not True or any(evidence.get('requirement_checks', {}).get(key) is not True for key in REQUIREMENTS):
            raise ValueError(f'{label}: unfinished selection/numerical review')
        if (evidence.get('model_label') != label or evidence.get('model_sha256') != entry['model_sha256']
                or evidence.get('selection_csv_sha256') != entry['selection_csv_sha256']):
            raise ValueError(f'{label}: selection audit does not match the frozen model and architecture evidence')
        value = json.loads(verified['model'].read_text())['model']
        model = Model.deserialize(value) if label == 'M0' else (Modulated.deserialize(value) if label == 'M2' else LatentModel.deserialize(value))
        if label == 'ML' and (model.anchors or model.diffusions):
            raise ValueError('ML must exclude modulation and diffusion')
        if label == 'M2L' and model.diffusions:
            raise ValueError('M2L must exclude diffusion')
        models[label] = (model, verified, evidence)
    if sha256(cache) != manifest['cache_sha256']:
        raise ValueError('Frozen count corpus changed')
    return manifest, models


def table(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def verify_evaluation_plan(manifest, freeze_path, particles, paths, stride):
    if not manifest.get('evaluation_plan') or not manifest.get('evaluation_plan_sha256'):
        raise ValueError('Freeze must bind the evaluation plan before opening test data')
    path = Path(manifest['evaluation_plan'])
    path = path if path.is_absolute() else freeze_path.parent / path
    if sha256(path) != manifest['evaluation_plan_sha256']:
        raise ValueError('Frozen evaluation plan changed')
    plan = json.loads(path.read_text())
    expected = {'models': list(LABELS), 'evaluation_split': 'test',
                'filter_particles': particles, 'filter_seed': 71391,
                'future_paths': paths, 'future_seed': 20261003, 'stride_seconds': stride,
                'horizons_seconds': list(HORIZONS), 'warmup_seconds': manifest['warmup_seconds'],
                'cache_sha256': manifest['cache_sha256'], 'test_evaluated_in_this_pipeline': False}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Evaluation settings differ from the frozen plan')
    return path


def fitted_scales(models, observed_mean):
    rows = []
    for label, (model, _, _) in models.items():
        base = model.base if label == 'M2' else model
        for target, background in enumerate(base.background):
            rows.append({'model': label, 'kind': 'background', 'target': 'BSL'[target], 'value': background})
        for j, mode in enumerate(base.modes):
            rows.append({'model': label, 'kind': 'exponential', 'source': 'BSL'[mode.source], 'target': 'BSL'[mode.target],
                         'mode_index': j, 'integrated_mass': mode.mass, 'tau_seconds': mode.tau})
        anchors = getattr(model, 'anchors', ())
        reference_mean = observed_mean if label in ('M0', 'M2') else physical_arrays(model, observed_mean)[8]
        gates = gate_values(np.log([m.tau for m in base.modes]), np.log([a.tau for a in anchors]))
        for p, anchor in enumerate(anchors):
            rows.append({'model': label, 'kind': 'anchor', 'source': 'BSL'[anchor.source], 'anchor_index': p, 'tau_seconds': anchor.tau})
        for j, row in enumerate(getattr(model, 'beta', ())):
            for p, coefficient in enumerate(row):
                normalized = float(coefficient * gates[j, p])
                reference = float(reference_mean[anchors[p].source])
                rows.append({'model': label, 'kind': 'modulation', 'mode_index': j, 'anchor_index': p, 'value': coefficient,
                             'gate': float(gates[j, p]), 'coefficient_on_normalized_slow': normalized,
                             'normalization_reference_mean': reference, 'link': model.link,
                             'coefficient_on_raw_slow': normalized / max(reference, 1e-8) if model.link == 'linear' else None})
        for q, diffusion in enumerate(getattr(model, 'diffusions', ())):
            for target, coefficient in enumerate(diffusion.loading):
                rows.append({'model': label, 'kind': 'diffusion_loading', 'diffusion_index': q,
                             'target': 'BSL'[target], 'tau_seconds': diffusion.tau, 'value': coefficient})
    return rows


def reusable_forecasts(directory, model, corpus_identity, label, particles, paths, stride):
    """Only resume complete forecasts produced by these exact frozen settings."""
    protocol_path = directory / 'forecast-protocol.json'
    if not protocol_path.exists():
        return False
    saved = json.loads(protocol_path.read_text())
    expected = {'model': model.serialize(), 'corpus': corpus_identity, 'split': 'test',
                'test_evaluated': True, 'stride_seconds': stride}
    if label == 'M2':
        expected.update(particles=paths, seed=20261003)
    elif label in ('ML', 'M2L', 'M2LZ'):
        expected.update(filter_particles=particles, filter_seed=71391, future_paths=paths)
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{label}: saved forecasts do not match frozen evaluation settings')
    required = ['test_forecasts.npz', 'test_one_step_metrics.json', 'test_horizon_metrics.csv', 'test_predictive_loss.npz']
    if label in ('ML', 'M2L', 'M2LZ'):
        required += ['test_latent_diagnostics.json', 'test_latent_trace.npz',
                     'test_origin_latent_states.npz']
    return all((directory / name).is_file() for name in required)


def reuse_identical_latent_forecasts(source, destination, model, corpus_identity, label, particles, paths, stride):
    """Reuse a completed exact-model test evaluation, changing family labels only."""
    if label not in LABELS[2:]:
        raise ValueError('Only identical latent-family evaluations may be reused')
    if not reusable_forecasts(source, model, corpus_identity, label, particles, paths, stride):
        raise ValueError('Source latent evaluation is incomplete')
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ('test_forecasts.npz', 'test_predictive_loss.npz', 'test_latent_trace.npz',
                 'test_origin_latent_states.npz', 'forecast-protocol.json'):
        shutil.copyfile(source / name, destination / name)
        artifacts[name] = sha256(source / name)
    for name in ('test_one_step_metrics.json', 'test_latent_diagnostics.json'):
        value = json.loads((source / name).read_text())
        if 'model' in value:
            value['model'] = label
        save_json(destination / name, value)
        artifacts[name] = sha256(source / name)
    with (source / 'test_horizon_metrics.csv').open(newline='') as stream:
        rows = [dict(row, model=label) for row in csv.DictReader(stream)]
    table(destination / 'test_horizon_metrics.csv', rows)
    artifacts['test_horizon_metrics.csv'] = sha256(source / 'test_horizon_metrics.csv')
    save_json(destination / 'identical-model-reuse.json', {
        'source': str(source.resolve()), 'model': model.serialize(),
        'numeric_artifacts_unchanged': True, 'source_artifacts_sha256': artifacts,
        'interpretation': 'Identical frozen parameters and evaluation settings; family labels alone differ'})


def run(freeze_path, cache, output, particles=512, paths=2048, stride=900):
    manifest, models = read_frozen_selection(freeze_path, cache)
    evaluation_plan = verify_evaluation_plan(manifest, freeze_path, particles, paths, stride)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'freeze_sha256': sha256(freeze_path), 'cache_sha256': manifest['cache_sha256'],
                'evaluation_plan_sha256': manifest['evaluation_plan_sha256'],
                'filter_particles': particles, 'future_paths': paths, 'stride_seconds': stride,
                'split': 'test', 'fitting_performed': False,
                'historical_test_inspection': manifest.get('historical_test_inspection'),
                'source_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in SOURCE_BYTES.items()}}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Final evaluation settings changed; use a new output directory')
    save_json(protocol_path, protocol)
    (output / 'frozen-selection-manifest.json').write_bytes(freeze_path.read_bytes())
    (output / 'frozen-evaluation-plan.json').write_bytes(evaluation_plan.read_bytes())
    source_directory = output / 'source'
    source_directory.mkdir(exist_ok=True)
    for name, value in SOURCE_BYTES.items():
        (source_directory / name).write_bytes(value)
    corpus = _load_corpus(cache, int(manifest['warmup_seconds']), True)
    one_step, horizons, selection, stability, latent = [], [], [], [], []
    origins_reference, actual_reference = None, None
    completed_latent = {}
    for label in LABELS:
        model, source, evidence = models[label]
        directory = output / label
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / 'frozen-model.json', {'model': model.serialize(), 'source_sha256': sha256(source['model'])})
        (directory / 'selection-audit.json').write_bytes(source['selection_audit'].read_bytes())
        (directory / 'architecture-selection.csv').write_bytes(source['selection_csv'].read_bytes())
        # A completed forecast file is reusable only under this frozen protocol.
        complete = directory / 'test_forecasts.npz'
        if not reusable_forecasts(directory, model, corpus.identity, label, particles, paths, stride):
            if label == 'M0':
                linear_forecasts(model, corpus, directory, stride, split='test')
            elif label == 'M2':
                modulated_forecasts(model, corpus, directory, paths, stride, split='test')
            else:
                identical = completed_latent.get(json.dumps(model.serialize(), sort_keys=True))
                if identical is None:
                    latent_forecasts(model, corpus, directory, label, particles, paths, stride, split='test')
                else:
                    reuse_identical_latent_forecasts(identical, directory, model, corpus.identity,
                                                    label, particles, paths, stride)
        if label in LABELS[2:]:
            completed_latent[json.dumps(model.serialize(), sort_keys=True)] = directory
        with np.load(complete, allow_pickle=False) as stored:
            if origins_reference is None:
                origins_reference, actual_reference = stored['origins'], stored['actual']
            elif not (np.array_equal(origins_reference, stored['origins']) and np.array_equal(actual_reference, stored['actual'])):
                raise ValueError(f'{label}: final forecast targets/origins differ')
        metric = json.loads((directory / 'test_one_step_metrics.json').read_text())
        one_step.append({'model': label, 'split': 'test', 'predictive_marginal_nll': metric.get('predictive_marginal_nll', metric['poisson_mean_log_score']), **metric})
        with (directory / 'test_horizon_metrics.csv').open(newline='') as stream:
            model_horizons = list(csv.DictReader(stream))
        horizons.extend(model_horizons)
        with source['selection_csv'].open(newline='') as stream:
            selection.extend(dict(row, family=label) for row in csv.DictReader(stream))
        if label == 'M0':
            matrix, global_bound = branching(model), True
        elif label == 'M2':
            matrix = upper_branching(*model.arrays()) if model.link == 'bounded' else branching(model.base)
            global_bound = model.link == 'bounded' or not any(value > 0 for row in model.beta for value in row)
        else:
            matrix = core_branching(model)
            global_bound = model.link == 'bounded' or not any(value > 0 for row in model.beta for value in row)
        stability.append({'model': label, 'integrated_branching_matrix': json.dumps(matrix.tolist()),
                          'spectral_radius': radius(matrix), 'global_branching_bound': global_bound,
                          'one_hour_failed_origin_fraction': model_horizons[-1].get('failed_origin_fraction', '0'),
                          'interpretation': 'Unbounded positive amplitude modulation has no global stability guarantee' if not global_bound else 'Normalized integrated branching bound; OU is exogenous stationary immigration'})
        if label in ('ML', 'M2L', 'M2LZ'):
            row = json.loads((directory / 'test_latent_diagnostics.json').read_text())
            diagnostic = parameter_diagnostics(model, corpus.initial_mean)
            latent.append(dict(row, integrated_masses=json.dumps(core_branching(model).tolist()),
                               near_duplicate_scales=json.dumps(diagnostic['near_duplicate_kernel_scales']),
                               subsecond_scales=json.dumps(diagnostic['subsecond_kernel_scales']),
                               validation_identifiability_review=json.dumps(evidence.get('identifiability_review'))))
        table(output / 'final_test_metrics.csv', one_step)
        table(output / 'horizon_metrics.csv', horizons)
        table(output / 'model_selection.csv', selection)
        table(output / 'stability.csv', stability)
        if latent:
            table(output / 'latent_diagnostics.csv', latent)
    table(output / 'fitted_scales.csv', fitted_scales(models, corpus.initial_mean))
    compare_forecasts([output / label for label in LABELS], LABELS, output / 'comparison',
                     split='test', stride=stride)
    compare_predictive_losses([output / label for label in LABELS], LABELS, output / 'comparison', split='test')
    save_json(output / 'evaluation-complete.json', {'all_five_models_evaluated': True, 'test_evaluated': True,
              'parameters_changed_after_freeze': False, 'forecast_origins_identical': True, 'protocol': protocol,
              'interpretation': 'Evaluation completion does not establish improvement; compare all metrics and recorded failures/non-identifiability'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('data/benchmarks/hawkes-resume-2026-10-03/counts-1s.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particles', type=int, default=512)
    parser.add_argument('--paths', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=900)
    args = parser.parse_args()
    run(args.freeze, args.cache, args.output, args.particles, args.paths, args.stride)


if __name__ == '__main__':
    main()
