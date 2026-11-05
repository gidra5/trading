"""Measure fixed-width NSF graph construction and small-tree decision distortion."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from nsf_scenario_graph import build_graph
from gpu_oracle_bellman import Bellman, Menu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=64)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--comparison-samples', type=int, default=64)
    ap.add_argument('--construction-only', action='store_true')
    ap.add_argument('--comparisons-only', action='store_true')
    ap.add_argument('--exposures', default='0')
    ap.add_argument('--refinement', choices=['independent', 'nested'], default='independent')
    ap.add_argument('--output', type=Path, default=Path('data/backtests/oracle-nsf-scenario-graph'))
    args = ap.parse_args()
    exposures = [float(x) for x in args.exposures.split(',')]
    if not exposures or any(not np.isfinite(x) or abs(x) > 2 for x in exposures) or (args.construction_only and args.comparisons_only):
        raise ValueError('Invalid benchmark selection')
    if args.depth < 1 or args.width < 1 or any(x < 2 or x % 2 for x in [args.samples, args.comparison_samples]):
        raise ValueError('Positive depth/width and even sample counts required')
    root = args.output
    if (root/'plan.json').exists():
        raise ValueError('Experiment exists')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source_plan = json.loads((checkpoint.parent/'plan.json').read_text(encoding='utf8'))
    selection = json.loads((checkpoint.parent/'selection.json').read_text(encoding='utf8'))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == selection['models']['latentSampledSkipFlow']['checkpointSha256']
    model, _ = load_latent_nsf(checkpoint, device)
    data = latent_data(Path(source_plan['nativeDirectory']), Path(source_plan['dataset']), 'test', HorizonRange(21600., 1))
    first = int(data['origins'][0]); start = float(data['timestamps'][0])/1000
    source_files = ['ml/nsf_scenario_graph.py', 'ml/benchmark_nsf_scenario_graph.py', 'ml/gpu_oracle_bellman.py',
                    'ml/export_nsf_planning_trees.py', 'ml/latent_neural_stochastic_flow.py']
    (root/'source').mkdir()
    for source in source_files:
        (root/'source'/Path(source).name).write_bytes(Path(source).read_bytes())
    comparison_widths = list(dict.fromkeys([args.comparison_samples**2, args.width, 64]))
    (root/'plan.json').write_text(json.dumps(dict(checkpointSha256=digest, depth=args.depth, width=args.width, samplesPerParent=args.samples,
        comparisonSamples=args.comparison_samples, comparisonWidths=comparison_widths, comparisonExposureStep=.1,
        comparisonInitialExposures=exposures,
        refinement=args.refinement, baseWidth=64,
        sourceHashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
        compression='weighted two-observation medoids; prices scaled by 1 bp; features RMS balanced; conditional rows preserved',
        accountStatesCompressed=False, epsilonCertificate=None), indent=2), encoding='utf8')
    def inputs(idx):
        return (np.asarray(data['timeline'][first+idx-1:first+idx+1], dtype=np.float32),
                data['pricesWithPast'][idx:idx+2], np.array([start+idx-1, start+idx]))
    if not args.comparisons_only:
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        graph = build_graph(model, *inputs(0), list(range(1, args.depth+1)), width=args.width, samples=args.samples, refinement=args.refinement)
        torch.cuda.synchronize()
        construction = dict(seconds=time.perf_counter()-tick, storedMarketNodes=sum(len(x.prices) for x in graph.layers),
            maximumLayerWidth=max(len(x.prices) for x in graph.layers),
            storedEdges=sum(x.numel() for x in graph.weights), peakAllocatedMiB=torch.cuda.max_memory_allocated()/2**20,
            diagnostics=graph.diagnostics)
        (root/'construction.json').write_text(json.dumps(construction, indent=2), encoding='utf8')
        print(json.dumps({k:v for k,v in construction.items() if k != 'diagnostics'}), flush=True)
        del graph
    if args.construction_only:
        return
    comparisons = []
    # Independent origins, plus a longer gap that gives nontrivial root decisions.
    # samples**2 is uncompressed for two transitions with this sample count.
    for idx in [0, 300, 900]:
        for offsets in [[1, 2], [900, 1800]]:
            for width in comparison_widths:
                # The uncompressed reference needs no split history.
                method = 'independent' if width >= args.comparison_samples**2 else args.refinement
                graph = build_graph(model, *inputs(idx), offsets, width=width, samples=args.comparison_samples, seed=20260916+idx, refinement=method)
                solver = Bellman(graph.prices, graph.weights, [Menu(step=.1)]*2, destinations=graph.destinations, chunk=131072)
                price = float(inputs(idx)[1][-1])
                for exposure in exposures:
                    state = torch.tensor([[10000*(1-exposure), 10000*exposure/price, price, price, 0, 0, 0, 0, 0]], device=device, dtype=torch.float64)
                    result = solver.solve(state)
                    descriptor = [None if np.isnan(x) else x for x in result['descriptor'][0].tolist()]
                    record = dict(origin=idx, offsets=offsets, initialExposure=exposure, width=width, value=result['value'].item(),
                        marketQuote=result['amount'].item(), descriptor=descriptor, terminalEvaluations=result['work'],
                        diagnostics=graph.diagnostics)
                    comparisons.append(record)
                    print(json.dumps({k:v for k,v in record.items() if k != 'diagnostics'}), flush=True)
    (root/'comparisons.json').write_text(json.dumps(comparisons, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
