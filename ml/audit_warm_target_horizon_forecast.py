"""Check parent identity, frozen H1 outputs and observed-state inference."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
from pathlib import Path
import json
import numpy as np
import torch
from audit_forecast_agreement import sha
from stochastic_price_forecast import load_timeline_split
from warm_target_horizon_forecast import load_warm_forecast, load_warm_observed_forecast
from train_target_horizon_forecast import write


def main():
    root = Path('data/backtests/oracle-warm-target-horizon')
    if (root/'audit.json').exists():
        raise ValueError('completed audit exists')
    plan = json.loads((root/'plan.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    evaluation = json.loads((root/'evaluation.json').read_text())
    for file, expected in plan['sourceHashes'].items():
        if sha(file) != expected:
            raise ValueError('training source changed')
    if sha(root/'selection.json') != evaluation['selectionSha256']:
        raise ValueError('selection changed')
    for name, record in selection['models'].items():
        if sha(root/f'{name}.pt') != record['checkpointSha256']:
            raise ValueError('selected checkpoint changed')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    result = {'sourceAndSelectionHashesVerified': True, 'parents': {}}
    for parent in plan['parents']:
        native_root = Path('data/backtests/oracle-expanded-mixture')/parent
        if sha(native_root/'head.pt') != selection['parents'][parent]['headSha256']:
            raise ValueError('parent checkpoint changed')
        initial, _ = load_warm_forecast(root/f'{parent}-initial.pt', device)
        original = torch.load(native_root/'head.pt', map_location=device, weights_only=False)
        for key, value in original['model'].items():
            if not torch.equal(value, initial.head.state_dict()[key]):
                raise ValueError('initial head is not an exact copy of original')
        frozen, _ = load_warm_forecast(root/f'{parent}-horizonWeightsOnly.pt', device)
        with np.load(root/f'{parent}-initial-test-h1.npz') as before, np.load(root/f'{parent}-horizonWeightsOnly-test-h1.npz') as after:
            for field in ['mean', 'logLikelihood', 'zeroProbability', 'conditionalPositiveProbability']:
                np.testing.assert_array_equal(before[field], after[field])
        adapter = load_warm_observed_forecast(root/f'{parent}-horizonWeightsOnly.pt', device)
        native_plan = json.loads((native_root/'plan.json').read_text())
        dataset = Path(json.loads(Path(native_plan['nativePlan']).read_text())['datasetDir'])
        source = load_timeline_split(dataset, 'validation')
        with np.load(native_root/'validation-native.npz') as cache:
            origins = cache['origins'][:2560]
            expected = torch.from_numpy(cache['features'][:2560]).to(device)
        observed = torch.from_numpy(np.asarray(source.timeline[origins[:, None]+[-1, 0]], dtype=np.float32)).to(device)
        with torch.inference_mode():
            features = adapter.native(observed).flatten(1)
            torch.testing.assert_close(features, expected, atol=1e-6, rtol=1e-5)
            horizons = torch.tensor(np.resize([1, 300, 7200, 86400], len(origins)), device=device)
            actual, reference = adapter(observed, horizons), frozen(expected, horizons)
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            weight_effect = float((frozen(expected, 7200)-initial(expected, 7200)).abs().max())
        result['parents'][parent] = {'initialStateExactlyEqualsParent': True, 'h1ForecastSummariesUnchangedAfterTraining': True,
            'maximumCausalNativeFeatureDifference': float((features-expected).abs().max()),
            'maximumObservedAdapterLogitDifference': float((actual-reference).abs().max()),
            'maximumLearnedH2hLogitChange': weight_effect}
    result['auditSourceSha256'] = sha(__file__)
    write(root, 'audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
