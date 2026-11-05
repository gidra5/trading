"""Evaluate compressed-graph root choices under the original small sampled law."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from nsf_scenario_graph import build_graph
from gpu_oracle_bellman import Bellman, Menu


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('data/backtests/oracle-nsf-scenario-graph-price-aware'))
    root = ap.parse_args().source
    rows = json.loads((root/'comparisons.json').read_text(encoding='utf8'))
    plan = json.loads((root/'plan.json').read_text(encoding='utf8'))
    samples = plan['comparisonSamples']
    reference_width = samples**2
    menu_step = plan['comparisonExposureStep']
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == plan['checkpointSha256']
    source = json.loads((checkpoint.parent/'plan.json').read_text(encoding='utf8'))
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    model, _ = load_latent_nsf(checkpoint, torch.device('cuda'))
    data = latent_data(Path(source['nativeDirectory']), Path(source['dataset']), 'test', HorizonRange(21600., 1))
    first = int(data['origins'][0]); start = float(data['timestamps'][0])/1000
    output = []
    for reference in [x for x in rows if x['width'] == reference_width]:
        idx = reference['origin']; offsets = reference['offsets']
        exposure = reference['initialExposure']
        prices = data['pricesWithPast'][idx:idx+2]
        graph = build_graph(model, np.asarray(data['timeline'][first+idx-1:first+idx+1], dtype=np.float32),
            prices, np.array([start+idx-1, start+idx]), offsets, width=reference_width, samples=samples, seed=20260916+idx)
        solver = Bellman(graph.prices, graph.weights, [Menu(step=menu_step)]*2, destinations=graph.destinations, chunk=131072)
        state = torch.tensor([[10000*(1-exposure), 10000*exposure/prices[-1], prices[-1], prices[-1], 0., 0., 0., 0., 0.]], dtype=torch.float64, device='cuda')
        candidates, valid, _ = solver.oracle.actions(state, solver.descriptors[0])
        for row in [x for x in rows if x['origin'] == idx and x['offsets'] == offsets and x['initialExposure'] == exposure]:
            descriptor = torch.tensor([float('nan') if x is None else x for x in row['descriptor']], dtype=torch.float64, device='cuda')
            action = torch.isclose(solver.descriptors[0], descriptor, equal_nan=True).all(1).nonzero()[0, 0]
            assert valid[0, action].item()
            moved = solver.oracle.advance(candidates[:, action, None, :].expand(-1, samples, -1), graph.prices[0])
            value, _ = solver._value(moved.reshape(-1, 9), graph.destinations[0].reshape(-1), 1)
            expected = (value*graph.weights[0].reshape(-1)).sum().item()
            regret = reference['value']-expected
            assert regret >= -1e-7
            output.append(dict(origin=idx, offsets=offsets, initialExposure=exposure, width=row['width'],
                referenceOptimalValue=reference['value'], referenceValueOfCompressedRootChoice=expected,
                firstActionRegretDollars=regret, feasibleUnderReferenceLaw=bool(np.isfinite(expected))))
    (root/'action-audit.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf8')
    (root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for row in output:
        if row['firstActionRegretDollars'] > 1e-7:
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
