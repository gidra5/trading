"""Re-evaluate frozen forecasts with continuous actions and their full price law.

Does not train, select checkpoints, fit calibration, or infer within-horizon
paths. All realized marks are used only AFTER an action has been selected.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from continuous_oracle_endpoint import (Rules, capacities, terminal_close,
    continuous_decision, execute, log_hurdle_law)


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def write(file, value):
    Path(file).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def action_encoding(trade, q, a, p, rules):
    if trade == 0:
        return {'marketQuote': '0'}
    capacity = capacities(q, a, p, rules)[0 if trade < 0 else 1]
    kind = 'capacity' if abs(trade) == capacity else 'minimum' if abs(trade) == rules.minimum else 'quote'
    return {'side': 'buy' if trade < 0 else 'sell', 'size': kind, 'amount': repr(abs(trade))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/backtests/oracle-joint-calibration'))
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-continuous-full-law'))
    parser.add_argument('--tolerance', type=float, default=1e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'comparison.json').exists():
        raise ValueError('completed comparison already exists')
    source_file = args.source/'evaluation.json'
    evaluation = json.loads(source_file.read_text())
    plan = json.loads((args.source/'replay-plan.json').read_text())
    hashes = {str(source_file): sha(source_file), str(args.source/'replay-plan.json'): sha(args.source/'replay-plan.json')}
    source_paths = [Path(__file__), Path('ml/continuous_oracle_endpoint.py')]
    hashes.update({str(file): sha(file) for file in source_paths})
    new_plan = {'contract': 'oracle-continuous-full-log-law-endpoint-flat-v2',
        'terminalRequirement': 'zero-asset-and-no-orders', 'source': str(args.source), 'horizons': [900, 1800, 7200],
        'variants': ['output-only-calibrated', 'joint-calibrated'], 'feeScales': [1, 0],
        'riskAversion': 0, 'fundingFactor': 1, 'entryBand': [0, 1], 'maintenanceBand': [0, 1],
        'orders': 0, 'rules': asdict(Rules()), 'absoluteValueTolerance': args.tolerance,
        'tailProbabilityPerConditionalSide': 1e-11, 'testUsedForSelection': False,
        'forecastProjection': 'max(1, nearest integer(P/dp))*dp; exact central tick-cell masses; analytic tail bounds',
        'scope': 'continuous signed market quote trades; zero is no trade; endpoint gamma0 MPC, not full-path optimality',
        'sourceHashes': hashes}
    write(args.output/'plan.json', new_plan)
    started = time.monotonic()
    reports = []
    for horizon in new_plan['horizons']:
        row = next(r for r in evaluation['results'] if r['split'] == 'test' and r['horizonSeconds'] == horizon)
        for variant in new_plan['variants']:
            saved = row['models'][variant]['savedStream']
            packed_file = Path(saved['parametersFile'])
            if sha(packed_file) != saved['parametersSha256']:
                raise ValueError('saved full distribution changed')
            hashes[str(packed_file)] = sha(packed_file)
            with np.load(packed_file) as z:
                parameters, timestamps = z['parameters'], z['timestamps']
            forecast_file = args.source/f'{variant}-h{horizon}.jsonl'
            manifest = json.loads(forecast_file.with_suffix('.manifest.json').read_text())
            if sha(forecast_file) != manifest['forecastSha256']:
                raise ValueError('observed replay tape changed')
            hashes[str(forecast_file)] = sha(forecast_file)
            rows = [json.loads(line) for line in forecast_file.read_text().splitlines() if line.strip()]
            if len(rows) != plan['decisionCountsByHorizon'][str(horizon)]:
                raise ValueError('decision population changed')
            simulations = {fee: {'q': 10000., 'a': 0., 'peak': 10000., 'drawdown': 0.,
                'markets': 0, 'exposure': [], 'decisions': [], 'sumSeconds': 0.} for fee in (1, 0)}
            previous = None
            for index, observed in enumerate(rows):
                stamp, price = observed['timestamp'], observed['observedPrice']
                if (observed['horizonSeconds'] != horizon or len(observed['realizedPrices']) != horizon or
                        observed['realizedNextTimestamp'] != stamp+horizon*1000 or
                        observed['realizedPrices'][-1] != observed['realizedNextPrice'] or
                        (previous is not None and previous != (stamp, price))):
                    raise ValueError('noncontiguous or incomplete realized tape')
                issue = int(np.searchsorted(timestamps, stamp))
                if issue >= len(timestamps) or timestamps[issue] != stamp:
                    raise ValueError('forecast must have been issued exactly at the decision timestamp')
                law_started = time.monotonic()
                law = log_hurdle_law(parameters[issue], price)
                law_seconds = time.monotonic()-law_started
                for fee_scale, sim in simulations.items():
                    rules = Rules(fee=.0005*fee_scale)
                    q, a = sim['q'], sim['a']
                    decision_started = time.monotonic()
                    decision = continuous_decision(q, a, price, law, rules, tolerance=args.tolerance)
                    sim['sumSeconds'] += time.monotonic()-decision_started+law_seconds
                    action = action_encoding(decision['marketQuote'], q, a, price, rules)
                    sim['q'], sim['a'] = decision['quote'], decision['asset']
                    if min(sim['q'], sim['a']) < -1e-9:
                        raise ValueError('continuous decision left unleveraged admissible set')
                    sim['q'], sim['a'] = max(0., sim['q']), max(0., sim['a'])
                    sim['markets'] += decision['marketQuote'] != 0
                    e = sim['a']*price/(sim['q']+sim['a']*price)
                    sim['exposure'].append(e)
                    # Only this block may inspect the future realized price path.
                    marks = np.asarray(observed['realizedPrices'])
                    if np.any(marks <= 0) or np.max(np.abs(marks/.01-np.rint(marks/.01))) > 1e-6:
                        raise ValueError('invalid actual price grid')
                    wealth = sim['q']+sim['a']*marks
                    peaks = np.maximum.accumulate(np.r_[sim['peak'], wealth])[1:]
                    sim['peak'] = float(peaks[-1])
                    sim['drawdown'] = max(sim['drawdown'], float(np.max(1-wealth/peaks)))
                    sim['decisions'].append({'timestamp': stamp, 'price': price, 'action': action,
                        'before': {'quote': q, 'asset': a}, 'after': {'quote': sim['q'], 'asset': sim['a']},
                        'forecastIndex': issue, 'parameters': parameters[issue].tolist(),
                        'centralPriceTicks': len(law.prices), 'tailMass': sum(t.mass for t in law.tails),
                        'expectedPrice': law.mean, 'expectedPriceErrorBound': law.mean_error,
                        'exposure': e, 'optimization': decision})
                previous = observed['realizedNextTimestamp'], observed['realizedNextPrice']
                if index % 20 == 0 or index == len(rows)-1:
                    print(json.dumps({'horizon': horizon, 'variant': variant, 'decisions': index+1,
                        'count': len(rows), 'seconds': time.monotonic()-started}), flush=True)
            for fee_scale, sim in simulations.items():
                rules = Rules(fee=.0005*fee_scale)
                final_price = rows[-1]['realizedNextPrice']
                q, a, flow = terminal_close(sim['q'], sim['a'], final_price, rules)
                terminal = q
                sim['markets'] += flow != 0
                sim['drawdown'] = max(sim['drawdown'], 1-terminal/sim['peak'])
                directory = args.output/f'{variant}-h{horizon}-fees{fee_scale}'
                directory.mkdir(exist_ok=True)
                decision_file = directory/'decisions.json'
                write(decision_file, {'forecastFile': str(forecast_file), 'forecastSha256': sha(forecast_file),
                    'parametersFile': str(packed_file), 'parametersSha256': sha(packed_file),
                    'rules': asdict(rules), 'decisions': sim['decisions']})
                old = json.loads((args.source/directory.name/'report.json').read_text())
                report = {'contract': new_plan['contract'], 'variant': variant, 'horizonSeconds': horizon,
                    'feeScale': fee_scale, 'gamma': 0, 'initialWealth': 10000.,
                    'complete': True, 'terminalRequirement': 'zero-asset-and-no-orders', 'terminalWealth': terminal, 'returnPercent': (terminal/10000-1)*100,
                    'maximumDrawdownPercent': 100*sim['drawdown'], 'markets': int(sim['markets']),
                    'decisionCount': len(rows), 'firstTimestamp': rows[0]['timestamp'],
                    'terminalTimestamp': rows[-1]['realizedNextTimestamp'],
                    'terminalResidualAsset': a, 'terminalQuote': q, 'averageExposure': float(np.mean(sim['exposure'])),
                    'fractionalExposureDecisions': sum(1e-8 < e < 1-1e-8 for e in sim['exposure']),
                    'maximumValueGap': max(d['optimization']['optimalityGap'] for d in sim['decisions']),
                    'maximumEvaluations': max(d['optimization']['evaluations'] for d in sim['decisions']),
                    'oldThreeMeanMenuReturnPercent': old['returnPercent'], 'oldMarkets': old['markets'],
                    'seconds': sim['sumSeconds'], 'decisionsFile': str(decision_file), 'decisionsSha256': sha(decision_file)}
                write(directory/'report.json', report)
                reports.append(report)
                print(json.dumps({'completed': directory.name, 'returnPercent': report['returnPercent'],
                    'gap': report['maximumValueGap']}), flush=True)
    for file, digest in hashes.items():
        if sha(file) != digest:
            raise ValueError('source changed during replay: '+file)
    write(args.output/'comparison.json', {'plan': new_plan, 'reports': reports,
        'sourceHashes': hashes, 'seconds': time.monotonic()-started,
        'executionVerified': False, 'formalContinuousOptimalityCertificate': False})


if __name__ == '__main__':
    main()
