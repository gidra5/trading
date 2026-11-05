"""Replay the first training epoch and audit causal inference and provenance.

The diagnostic trained checkpoint is not substituted for the validation choice.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
from pathlib import Path
import json
import numpy as np
import torch
from audit_forecast_agreement import sha
from target_horizon_forecast import HorizonRange, load_target_horizon_forecast, load_observed_target_horizon_forecast
from train_target_horizon_forecast import prepare, targets, objective, write
from stochastic_price_forecast import load_timeline_split


def main():
    root = Path('data/backtests/oracle-target-horizon')
    if (root/'audit.json').exists():
        raise ValueError('audit already completed')
    plan = json.loads((root/'plan.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    selection_hash = sha(root/'selection.json')
    for file, expected in plan['sourceHashes'].items():
        if sha(file) != expected:
            raise ValueError(f'training source changed: {file}')
    for name, item in selection['models'].items():
        if sha(root/f'{name}.pt') != item['checkpointSha256']:
            raise ValueError('selected checkpoint changed')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    hrange = HorizonRange(**plan['horizonRange'])
    native_root, dataset = Path(plan['nativeDirectory']), Path(plan['dataset'])
    train, val = (prepare(native_root, dataset, s, hrange) for s in ('train', 'validation'))
    fit_h = hrange.sample(np.random.default_rng(20260912), len(train['features']))
    normalization = float(np.mean(targets(train, fit_h)**2/fit_h))
    with np.load(root/'validation-selection-pairs.npz') as pairs:
        vi, vh, vy = pairs['ids'], pairs['horizons'], pairs['y']
    np.testing.assert_array_equal(vy, targets(val, vh, vi))
    rng = np.random.default_rng(20260913)
    ids = np.tile(np.arange(len(train['features'])), 2)
    rng.shuffle(ids)
    h = hrange.sample(rng, len(ids))
    y = targets(train, h, ids)
    x = torch.from_numpy(train['features']).to(device)
    hh, yy = torch.from_numpy(h).to(device), torch.from_numpy(y).to(device)
    vx, vhh, vyy = torch.from_numpy(val['features'][vi]).to(device), torch.from_numpy(vh).to(device), torch.from_numpy(vy).to(device)
    model, payload = load_target_horizon_forecast(root/'featuresAndHorizon.pt', device)
    if payload['epoch'] != 0:
        raise ValueError('this audit expects the initial-prior selection from this run')
    @torch.inference_mode()
    def score(features, horizons, ys):
        return sum(float(objective(model, features[i:i+4096], horizons[i:i+4096], ys[i:i+4096], normalization))*len(ys[i:i+4096])
            for i in range(0, len(ys), 4096))/len(ys)
    sx = x[torch.from_numpy(ids).to(device)]
    before = {'train': score(sx, hh, yy), 'validation': score(vx, vhh, vyy)}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    for i in range(0, len(ids), 4096):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, sx[i:i+4096], hh[i:i+4096], yy[i:i+4096], normalization)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 100); optimizer.step()
    model.eval()
    after = {'train': score(sx, hh, yy), 'validation': score(vx, vhh, vyy)}
    history = json.loads((root/'featuresAndHorizon-history.json').read_text())
    np.testing.assert_allclose(after['validation'], history[1]['validationLoss'], atol=1e-7, rtol=0)
    torch.save({**payload, 'model': model.state_dict(), 'epoch': 1, 'validationLoss': after['validation'],
        'diagnosticOnly': True, 'selection': 'first epoch replay for audit; not selected for deployment'}, root/'diagnostic-trained-epoch1.pt')
    adapter = load_observed_target_horizon_forecast(root/'featuresAndHorizon.pt', device)
    source = load_timeline_split(dataset, 'validation')
    origins = val['origins'][:2560]
    observed = np.asarray(source.timeline[origins[:, None]+np.array([-1, 0])], dtype=np.float32)
    with torch.inference_mode():
        actual_features = adapter.native(torch.from_numpy(observed).to(device)).flatten(1)
        expected_features = torch.from_numpy(val['features'][:len(origins)]).to(device)
        torch.testing.assert_close(actual_features, expected_features, atol=1e-6, rtol=1e-5)
        hs = torch.tensor(np.resize([1, 900, 7200, 86400], len(origins)), device=device)
        # Audit the trained network so this cannot pass merely because epoch 0 ignores inputs.
        actual_logits = model(actual_features, hs)
        expected_logits = model(expected_features, hs)
        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-5, rtol=2e-5)
        feature_error = float((actual_features-expected_features).abs().max())
        logit_error = float((actual_logits-expected_logits).abs().max())
    if sha(root/'selection.json') != selection_hash:
        raise ValueError('audit changed frozen selection')
    result = {'sourceHashesVerified': True, 'checkpointHashesVerified': True,
        'selectionSha256Unchanged': selection_hash, 'beforeFirstEpoch': before, 'afterFirstEpoch': after,
        'firstEpochValidationReproduced': True, 'maximumCausalNativeFeatureDifference': feature_error,
        'maximumTrainedLogitDifference': logit_error,
        'diagnosticCheckpointSha256': sha(root/'diagnostic-trained-epoch1.pt'), 'auditSourceSha256': sha(__file__)}
    write(root, 'audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
