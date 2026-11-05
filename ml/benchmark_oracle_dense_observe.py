"""Measure independent proposal batching on an archived real value field."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numba import set_num_threads
from oracle_account_grid import CellErrors
from oracle_dense_metadata import prepared
from oracle_dense_observe_batch import observe_batched
from oracle_dense_value_map import DenseValueMap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    args = p.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.threads):
        raise ValueError('Positive repeat/thread counts required')
    m = DenseValueMap.load(args.map)
    with np.load(args.map/'front.npz') as f:
        front = {key: f[key] for key in f.files}
    f, price_pair = m.cloud.features[1, 0], m.cloud.price_pairs[1, 0]
    call_args = (front, float(price_pair[-1]), f, price_pair, np.ones(59))
    reference = None
    timings = []
    for threads in [0, *args.threads]:
        set_num_threads(max(1, threads))
        durations = []
        for repeat in range(args.repeats+1):
            m.errors = CellErrors.empty(len(m.a['coordinates']))
            view = prepared(m)
            start = time.perf_counter()
            result = observe_batched(view, *call_args) if threads else DenseValueMap.observe(view, *call_args)
            duration = time.perf_counter()-start
            if reference is None:
                reference = (result, m.errors)
            else:
                if result != reference[0]:
                    raise AssertionError('Observation summary changed')
                for key in ('count', 'mean', 'm2'):
                    np.testing.assert_array_equal(getattr(m.errors, key), getattr(reference[1], key))
            if repeat:
                durations.append(duration)
            else:
                warmup = duration
        timings.append(dict(method='batched' if threads else 'original', threads=threads or 1,
            warmupSeconds=warmup, seconds=durations, medianSeconds=float(np.median(durations)),
            exactSummaryAndStatistics=True))
    baseline = timings[0]['medianSeconds']
    for row in timings:
        row['speedup'] = baseline/row['medianSeconds']
    report = dict(map=str(args.map), finiteProposals=int(np.isfinite(front['scores']).sum()),
        timings=timings, sourceHashes={name:hashlib.sha256(Path('ml', name).read_bytes()).hexdigest()
            for name in ('oracle_dense_observe_batch.py', 'oracle_dense_metadata.py', 'oracle_book_mc.py')},
        scope='CPU proposal-error update only. Same archived field, original accumulation order; no GPU sweep or new forecast.')
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
