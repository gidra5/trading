"""CPU-only paired local-refinement audit on the learned NSF and saved tail."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from numba import set_num_threads
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_dense_value_map import DenseValueMap
from oracle_book_mc import move_arrive
from oracle_local_market_refresh import local_fields, root_decision, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offsets', type=int, nargs='+', default=[30, 300, 600])
    p.add_argument('--replicates', type=int, default=2)
    p.add_argument('--branches', type=int, default=8)
    args = p.parse_args()
    if args.replicates < 1:
        raise ValueError('Positive replicate count required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    set_num_threads(2)
    started = time.perf_counter()
    m = DenseValueMap.load(args.map)
    source = json.loads((args.map.parent/'plan.json').read_text())
    settings = source['arguments']
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source['checkpointSha256']:
        raise ValueError('Checkpoint differs from archived field')
    model, _ = load_latent_nsf(checkpoint, 'cpu')
    training = json.loads((checkpoint.parent/'plan.json').read_text())
    data = latent_data(Path(training['nativeDirectory']), Path(training['dataset']), 'test', HorizonRange(21600., 1))
    text = (args.replay/'updates.jsonl').read_text()
    lines = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    log = {row['update']: row for row in (json.loads(line) for line in lines)}
    if any(offset not in log or offset-1 not in log for offset in args.offsets):
        raise ValueError('Every audited observation and its previous decision must already be logged')
    rows, diagnostics = [], []
    for offset in args.offsets:
        f, prices, times = [x.numpy()[0] for x in timeline_inputs(data, np.array([settings['origin']+offset]), 'cpu')]
        if prices[-1] != log[offset]['price']:
            raise ValueError('Realized log and source observation differ')
        arrived, consumed, filled, valid = move_arrive(np.array(log[offset-1]['state']), float(prices[-1]), m.rules)
        if not valid:
            raise ValueError('Actual account did not survive the audited arrival')
        states = {'actual': (arrived, consumed, filled),
            'cash': (np.r_[10000., 0., prices[-1], prices[-1], np.zeros(3*m.grid.max_orders)], 0., False)}
        for replicate in range(args.replicates):
            batch_start = time.perf_counter()
            fields, factors, diagnostic = local_fields(m, model, f, prices, times, offset,
                settings['seed'], settings['candidates'],
                settings['seed']+settings['origin']+offset+100000*replicate, args.branches)
            diagnostics.append(dict(offset=offset, replicate=replicate, **diagnostic))
            for name, (state, used, had_fill) in states.items():
                choices = {key: root_decision(m, state, used, had_fill, field, factors)
                           for key, field in fields.items()}
                fresh_value, fresh_action, _ = choices['fresh_history']
                for key, (value, action, _) in choices.items():
                    crossed = score(m, action, fields['fresh_history'], factors)
                    rows.append(dict(offset=offset, replicate=replicate, initialState=name, method=key,
                        value=float(value) if np.isfinite(value) else None, state=action.tolist(),
                        sameActionAsFresh=bool(np.array_equal(action, fresh_action)),
                        bestValueShiftVsRetainedLog=float(value-choices['retained_resampled'][0])
                            if np.isfinite(value) and np.isfinite(choices['retained_resampled'][0]) else None,
                        gapUnderFreshBps=float(-np.expm1(crossed-fresh_value)*10000)
                            if np.isfinite(crossed) and np.isfinite(fresh_value) else None))
                if name == 'actual':
                    logged = np.array(log[offset]['state'])
                    logged_value = score(m, logged, fields['fresh_history'], factors)
                    rows.append(dict(offset=offset, replicate=replicate, initialState=name, method='logged_action',
                        state=logged.tolist(), sameActionAsFresh=bool(np.array_equal(logged, fresh_action)),
                        gapUnderFreshBps=float(-np.expm1(logged_value-fresh_value)*10000)
                            if np.isfinite(logged_value) and np.isfinite(fresh_value) else None))
            result = dict(rows=rows, diagnostics=diagnostics, elapsedSeconds=time.perf_counter()-started,
                sourceMap=str(args.map), sourceReplay=str(args.replay), branches=args.branches,
                sourceHashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('audit_oracle_local_market_refresh.py', 'oracle_local_market_refresh.py')},
                scope='Two local Bellman layers, fixed archived distant tail and finite future menus. '
                    'Retained and fresh child histories share random draws. Catalog/retained-book root search excludes interior samples; '
                    'the logged action is cross-scored separately and may beat that catalog. '
                    'Value gaps are local sensitivity under the refreshed approximation, not regret against the true optimum.')
            (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            print(json.dumps(dict(offset=offset, replicate=replicate, seconds=time.perf_counter()-batch_start,
                elapsedSeconds=result['elapsedSeconds'], **diagnostic)), flush=True)


if __name__ == '__main__':
    main()
