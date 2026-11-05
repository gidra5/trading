"""Audit saved inference and rescore cancellation-sensitive MMD in float64."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
from pathlib import Path
import json
import numpy as np
import torch
from audit_forecast_agreement import sha
from neural_stochastic_flow import load_nsf
from target_horizon_forecast import HorizonRange
from train_neural_stochastic_flow import market_data
from train_target_horizon_forecast import write
from evaluate_neural_stochastic_flow import consistency


def main():
    root = Path('data/backtests/oracle-neural-stochastic-flow')
    if (root/'audit.json').exists():
        raise ValueError('completed audit exists')
    plan = json.loads((root/'plan.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    result = json.loads((root/'evaluation.json').read_text())
    for file, expected in plan['sourceHashes'].items():
        if sha(file) != expected:
            raise ValueError('training source changed')
    selection_hash = sha(root/'selection.json')
    if selection_hash != result['selectionSha256']:
        raise ValueError('selection changed')
    write(root, 'evaluation-before-double-mmd.json', result)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    data = market_data(Path(plan['nativeDirectory']), Path(plan['dataset']), 'test', HorizonRange())
    audit = {'trainingHashesVerified': True, 'selectionSha256Unchanged': selection_hash, 'models': {}}
    for name, record in selection['models'].items():
        if sha(root/f'{name}.pt') != record['checkpointSha256']:
            raise ValueError('selected checkpoint changed')
        model, _ = load_nsf(root/f'{name}.pt', device)
        x = torch.from_numpy(data['state'][:128]).to(device)
        f = torch.from_numpy(data['features'][:128]).to(device)
        d = torch.linspace(.0001, 4., len(x), device=device); start = torch.zeros_like(d)
        with torch.inference_mode():
            if not torch.equal(model.rsample(x, f, start, start), x):
                raise ValueError('trained zero-duration identity failed')
            y = model.rsample(x, f, d, start)
            if not bool(torch.isfinite(model.log_prob(y, x, f, d, start)).all()):
                raise ValueError('nonfinite trained density')
            cond = model.condition(x, f, d, start)
            output, jac = model.couplings.transform(x, cond, d)
            reconstructed, invjac = model.couplings.transform(output, cond, d, True)
            torch.testing.assert_close(x, reconstructed, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(jac, -invjac, atol=2e-5, rtol=2e-5)
        result['consistency'][name] = {str(h): consistency(model, data, device, h) for h in [300, 7200, 21600]}
        for value in result['consistency'][name].values():
            if any(row['jointMmdSquared'] < -1e-12 for row in value.values()):
                raise ValueError('negative float64 biased MMD')
        audit['models'][name] = {'zeroDurationIdentity': True, 'finiteSampleLogDensities': True,
            'maximumTrainedInverseError': float((x-reconstructed).abs().max()), 'float64MmdNonnegative': True}
    result['originalEvaluationSourceSha256'] = result['sourceSha256']
    result['sourceSha256'] = sha('ml/evaluate_neural_stochastic_flow.py')
    result['numericalAudit'] = 'MMD and Wasserstein recomputed in float64; price scores and selected models unchanged.'
    audit['auditSourceSha256'] = sha(__file__)
    write(root, 'evaluation.json', result)
    write(root, 'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
