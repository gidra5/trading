"""Replay a completed experiment from its full-size resident-table checkpoint.

Requires unchanged implementation/model identities and independently checks the
execution ledger before loading CUDA state. Reference sweeps are unnecessary:
they never updated the original sampler. This grades restart equivalence only.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from audit_oracle_incremental_checkpoint import audit, digest


def compare_step(actual, expected, tolerance=1.e-12):
    for key in ('update','originSeconds','filled','marketTrade'):
        if actual[key]!=expected[key]: raise AssertionError('Restored '+key+' differs')
    if not np.array_equal(actual['state'],expected['state']):
        raise AssertionError('Restored account differs')
    if ('estimatedLogValue' in actual)!=('estimatedLogValue' in expected):
        raise AssertionError('Restored value phase differs')
    error=0.
    if 'estimatedLogValue' in actual:
        error=abs(actual['estimatedLogValue']-expected['estimatedLogValue'])
        if not np.isfinite(error) or error>tolerance:
            raise AssertionError('Restored log value differs')
        for key in ('horizon',):
            if actual[key]!=expected[key]: raise AssertionError('Restored '+key+' differs')
        # Runtime is allowed to differ; actual row/edge allocation must not.
        for key in ('updates','updatedLayers','updatedRows','evaluatedOutcomes','cachedRows'):
            if actual['table'][key]!=expected['table'][key]:
                raise AssertionError('Restored sampling '+key+' differs')
        for key in ('proposals','evaluatedOutcomes','partialRows'):
            if actual['root'][key]!=expected['root'][key]:
                raise AssertionError('Restored root sampling '+key+' differs')
    return error


def run(directory, output, checkpoint, price_source=None):
    import torch
    from numba import get_num_threads,set_num_threads
    from latent_neural_stochastic_flow import load_latent_nsf
    from train_latent_neural_stochastic_flow import latent_data
    from target_horizon_forecast import HorizonRange
    from oracle_continuation_value import timeline_inputs
    from oracle_dense_controller import DenseControllerConfig
    from oracle_incremental_controller import IncrementalController
    from oracle_incremental_table import TableBudget
    from oracle_incremental_checkpoint import load_table

    directory=Path(directory).resolve(); output=Path(output).resolve()
    completed=json.loads((directory/'results.json').read_text())
    if not completed['completed']: raise ValueError('Require a completed uninterrupted replay')
    plan=json.loads((directory/'plan.json').read_text())
    for source,checksum in plan['sourceHashes'].items():
        if not source.startswith('ml/') or '/' in source[3:]: raise ValueError('Unexpected source path')
        if digest(Path(__file__).with_name(source[3:]))!=checksum:
            raise ValueError('Current replay implementation differs: '+source)
    if digest(checkpoint)!=plan['checkpointSha256']: raise ValueError('Model checkpoint differs')
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    started=time.perf_counter()
    integrity=audit(directory,price_source)
    root=directory/'checkpoints'
    meta=json.loads((root/integrity['snapshot']/'metadata.json').read_text())
    context=meta['context']  # Independently matched to the run plan by audit().
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    set_num_threads(min(4,get_num_threads()))
    restore_started=time.perf_counter(); table,front=load_table(root,context)
    restore_seconds=time.perf_counter()-restore_started
    if table.update!=integrity['checkpointUpdate']: raise ValueError('Checkpoint changed during restore')
    model,_=load_latent_nsf(checkpoint,'cuda')
    training=json.loads((checkpoint.parent/'plan.json').read_text())
    data=latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    controller=IncrementalController(table,model,front['state'],DenseControllerConfig(**context['config']),
        budget=TableBudget(**context['budget']),root_draws=context['rootDraws'],front=front,
        episode_origin=context['episodeOrigin'],maximum_horizon=context['maximumHorizon'])
    rows=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
    first=controller.update+1; maximum=0.; replay_started=time.perf_counter()
    with (output/'decisions.jsonl').open('w') as stream:
        for expected in rows[first-1:]:
            update=expected['update']
            f,p,t=[x.numpy()[0] for x in timeline_inputs(data,
                np.array([context['config']['forecast_origin_index']+update]),'cpu')]
            actual=controller.advance(f,p,t)
            error=compare_step(actual,expected); maximum=max(maximum,error)
            record=dict(update=update,stateMatched=True,samplingMatched=True,logValueError=error,
                stepSeconds=actual['stepSeconds'])
            stream.write(json.dumps(record)+'\n'); stream.flush()
            if update==first or update%30==0: print(json.dumps(record),flush=True)
    if controller.phase!='complete' or controller.terminal_wealth!=completed['terminalWealth']:
        raise AssertionError('Restored terminal result differs')
    result=dict(checkpointUpdate=first-1,checkedUpdates=len(rows)-first+1,
        residentCapacity=table.capacity,residentValueBytes=table.values.numel()*table.values.element_size(),
        maximumLogValueError=maximum,restoreSeconds=restore_seconds,
        replaySeconds=time.perf_counter()-replay_started,elapsedSeconds=time.perf_counter()-started,
        terminalWealth=controller.terminal_wealth,terminalMatched=True,integrity=integrity,
        scope='Full-size checkpoint recovery and subsequent account, sample allocation, value and terminal equivalence. Not policy optimality.')
    (output/'results.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,
        default=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt'))
    parser.add_argument('--price-source',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.directory,args.output,args.checkpoint,args.price_source)),flush=True)
