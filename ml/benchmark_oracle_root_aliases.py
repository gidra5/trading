"""Paired root-query equivalence and timing under one fixed finite context.

The identity-key run executes every descriptor, reproducing the pre-alias
workload. Both modes use identical fields, controls, proposals and outcomes.
This measures root evaluation only, not a complete replay speedup.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from numba import set_num_threads
from oracle_dense_value_map import DenseValueMap
from oracle_incremental_table import IncrementalTable
from oracle_book_mc import move_arrive
from oracle_root_aliases import exact_root_aliases
from audit_oracle_incremental_checkpoint import digest


def identity_keys(states):
    indices=np.arange(len(states))
    return indices,indices


def run(directory,reference,output,repeats):
    if repeats<2: raise ValueError('At least two timed pairs required')
    directory=Path(directory);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);set_num_threads(2)
    meta=json.loads((directory/'countdown-anchor'/'metadata.json').read_text())
    start=meta['update'];settings=json.loads((directory/'plan.json').read_text())['arguments']
    rows=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
    source=DenseValueMap.load(reference)
    if source.cloud.origin_seconds!=rows[start-1]['originSeconds']:
        raise ValueError('Reference timestamp differs from the decision')
    table=IncrementalTable(source,settings['seed'],settings['candidates'])
    table.update=start
    arrived,used,filled,valid=move_arrive(np.asarray(rows[start-2]['state']),
        source.cloud.prices[0,0],source.rules)
    if not valid: raise ValueError('Logged arrival is infeasible')
    # Warm both paths and then alternate their order to limit timing bias.
    records=[];expected=None;expected_controls=None
    for iteration in range(repeats+1):
        pair={}
        for name in (('allDescriptors','aliases') if iteration%2==0 else ('aliases','allDescriptors')):
            table.root_controls=None
            function=identity_keys if name=='allDescriptors' else exact_root_aliases
            with patch('oracle_incremental_table.exact_root_aliases',function):
                torch.cuda.synchronize();began=time.perf_counter()
                front,report=table.choose(arrived,used,filled,draws=table.B)
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
            if expected is None:
                expected=front;expected_controls=table.root_controls.copy()
            else:
                for key in ('value','state','choice','scores','states'):
                    np.testing.assert_array_equal(front[key],expected[key],err_msg=key)
                np.testing.assert_array_equal(table.root_controls,expected_controls)
            pair[name]=dict(seconds=elapsed,**{key:report[key] for key in
                ('proposals','uniqueProposals','evaluatedOutcomes','logicalOutcomes','partialRows')})
        if iteration:records.append(pair)
    medians={name:float(np.median([r[name]['seconds'] for r in records]))
        for name in ('allDescriptors','aliases')}
    result=dict(update=start,horizon=table.H,repeats=repeats,bitExact=True,
        medianSeconds=medians,rootSpeedRatio=medians['allDescriptors']/medians['aliases'],
        pairs=records,sourceHashes={name:digest(Path(__file__).with_name(name)) for name in
            ('oracle_incremental_table.py','oracle_root_aliases.py','oracle_selected_outcomes.py')},
        scope='Paired root-only timing and exact front/control equality in one saved full-size context; not overall runtime or optimality.')
    (output/'results.json').write_text(json.dumps(result,indent=2))
    (output/'benchmark-source.py').write_bytes(Path(__file__).read_bytes())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    print(json.dumps(run(args.directory,args.reference,args.output,args.repeats)),flush=True)
