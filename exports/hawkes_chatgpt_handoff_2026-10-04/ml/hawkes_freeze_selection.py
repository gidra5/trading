"""Publish a verified five-model freeze without opening test observations."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from hawkes_final_test import LABELS, read_frozen_selection, sha256, verify_evaluation_plan


def freeze(request_path: Path, output: Path):
    request = json.loads(request_path.read_text())
    if set(request.get('models', {})) != set(LABELS):
        raise ValueError('Freeze request must provide all five models')

    def resolve(value):
        path = Path(value)
        return (path if path.is_absolute() else request_path.parent / path).resolve()

    cache, plan_path = resolve(request['cache']), resolve(request['evaluation_plan'])
    plan = json.loads(plan_path.read_text())
    manifest = {
        'ready_for_test': True, 'final_test_evaluated_in_this_pipeline_before_freeze': False,
        'cache': str(cache), 'cache_sha256': sha256(cache),
        'warmup_seconds': plan['warmup_seconds'],
        'evaluation_plan': str(plan_path), 'evaluation_plan_sha256': sha256(plan_path),
        'historical_test_inspection': plan['historical_test_inspection'],
        'freeze_builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'models': {},
    }
    for label in LABELS:
        entry = {}
        for key in ('model', 'selection_audit', 'selection_csv'):
            path = resolve(request['models'][label][key])
            entry[key], entry[f'{key}_sha256'] = str(path), sha256(path)
        manifest['models'][label] = entry

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.pending',
                                     prefix='.hawkes-freeze-', dir=output.parent, delete=False) as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    try:
        # Audit labels, model/csv hashes, hierarchy restrictions and all review
        # requirements are checked by the same validator as final evaluation.
        verified, _ = read_frozen_selection(temporary, cache)
        verify_evaluation_plan(verified, temporary, plan['filter_particles'],
                               plan['future_paths'], plan['stride_seconds'])
        if output.exists():
            if json.loads(output.read_text()) != manifest:
                raise ValueError('Existing freeze differs; preserve it and use a new output path')
        else:
            # Atomic publication that cannot overwrite another completed freeze.
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True,
                        help='JSON: cache, evaluation_plan, and model/audit/csv paths for each family')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.request, args.output)
    print(json.dumps({'freeze': str(args.output.resolve()), 'models': list(manifest['models']),
                      'test_observations_opened': False}))


if __name__ == '__main__':
    main()
