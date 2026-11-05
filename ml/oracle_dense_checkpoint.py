"""Validate and resume a cleanly stopped dense-policy replay."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


def initial_archive(directory,plan=None):
    directory=Path(directory)
    if plan is None:plan=json.loads((directory/'plan.json').read_text())
    if 'initialArchive' not in plan:return directory/'initial'
    path=Path(plan['initialArchive'])
    return path if path.is_absolute() else directory/path


@dataclass(frozen=True)
class ReplayCheckpoint:
    directory:Path
    current:object
    front:dict
    state:np.ndarray
    rows:tuple
    comparisons:tuple
    update:int
    elapsed_seconds:float
    initial_seconds:float
    market_trades:int
    fill_ticks:int
    initial:Path


def load_checkpoint(directory,expected,initial_origin,load_map=None):
    """Require a complete saved boundary; never recover from a partial write.

    Earlier logs and the original initial archive remain immutable. Loading a
    value map also verifies its current Bellman operator signature.
    """
    directory=Path(directory).resolve();plan=json.loads((directory/'plan.json').read_text())
    result=json.loads((directory/'results.json').read_text())
    if (result.get('completed') is not False or result.get('failure') is not None
        or result.get('terminalWealth') is not None or result.get('terminalReturn') is not None):
        raise ValueError('Resume requires an incomplete replay without execution failure')
    for key in ('grid','rules','fundingMultiplier','gamma','checkpointSha256'):
        if key not in plan or json.dumps(plan[key],sort_keys=True)!=json.dumps(expected[key],sort_keys=True):
            raise ValueError(f'Resume changes {key}')
    for key in ('seed','origin','horizon','branches','candidates','seconds','refresh_every','repair_tolerance','repair_interval'):
        if plan['arguments'].get(key)!=expected['arguments'].get(key):raise ValueError(f'Resume changes {key}')
    for key in ('ml/oracle_dense_rollout.py','ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py'):
        if key not in plan['sourceHashes'] or plan['sourceHashes'][key]!=expected['sourceHashes'][key]:
            raise ValueError(f'Resume changes forecast generator: {key}')
    update=result['observedUpdates'];end=expected['arguments']['seconds']
    if not isinstance(update,int) or not 0<=update<end:raise ValueError('Invalid saved replay update')
    text=(directory/'updates.jsonl').read_text()
    if text and not text.endswith('\n'):raise ValueError('Saved log has an unflushed line')
    rows=tuple(json.loads(line) for line in text.splitlines() if line.strip())
    if len(rows)!=update or any(row['update']!=i+1 or 'state' not in row for i,row in enumerate(rows)):
        raise ValueError('Saved log does not cover the checkpoint boundary')
    if load_map is None:
        from oracle_dense_value_map import DenseValueMap
        load_map=DenseValueMap.load
    current=load_map(directory/'last-map')
    from oracle_dense_repair import contract
    remaining=min(expected['arguments']['horizon'],end-update)
    if current.shape[0]!=remaining or current.cloud.origin_seconds!=initial_origin+update:
        raise ValueError('Saved value map has a different time boundary')
    if current.bellman_contract!=contract(expected['arguments']['seed'],expected['arguments']['candidates']):
        raise ValueError('Saved value map has a different action schedule')
    residual=current.bellman_residuals
    if (residual is None or residual.shape!=(remaining,) or np.any(~np.isfinite(residual)|(residual<0))
        or not np.all(current.samples==expected['arguments']['branches'])):
        raise ValueError('Saved map lacks complete conditional rows or residual accounting')
    with np.load(directory/'last-map'/'front.npz') as data:front={key:data[key] for key in data.files}
    state=np.asarray(result['state'],dtype=float)
    if (state.shape!=(4+3*expected['grid']['max_orders'],) or not np.isfinite(state).all()
        or not np.array_equal(state,front['state']) or not np.isfinite(front['value'])
        or (rows and not np.array_equal(state,rows[-1]['state']))
        or state[2]!=current.cloud.prices[0,0]):
        raise ValueError('Saved account and chosen action disagree')
    cells=current.shape[2]
    if front['scores'].shape!=(cells,) or front['states'].shape!=(cells,len(state)):
        raise ValueError('Saved front proposals differ from the account grid')
    from audit_oracle_dense_ledger import audit
    ledger=audit(directory)
    if (ledger['checkedUpdates']!=update or ledger['marketTrades']!=result['marketTrades']
        or ledger['fillTicks']!=result['fillTicks'] or not np.array_equal(ledger['lastState'],state)):
        raise ValueError('Saved aggregate execution ledger differs')
    elapsed=float(result['elapsedSeconds']);initial_seconds=float(result['initialSeconds'])
    if not np.isfinite(elapsed) or not 0<=initial_seconds<=elapsed:raise ValueError('Invalid saved runtime')
    return ReplayCheckpoint(directory,current,front,state,rows,tuple(result['comparisons']),update,
        elapsed,initial_seconds,result['marketTrades'],result['fillTicks'],initial_archive(directory,plan).resolve())
