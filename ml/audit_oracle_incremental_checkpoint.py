"""CPU-only audit of a replay's immutable resident-table checkpoint.

Checks the published bytes, source/model context, ring dimensions and accepted
account boundary. The separate execution auditor checks the logged trades.
This does not evaluate forecasts, estimate Bellman errors or upload to a GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_oracle_dense_ledger import audit as audit_ledger,require


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): result.update(chunk)
    return result.hexdigest()


def array_header(path):
    array = np.load(path,mmap_mode='r',allow_pickle=False)
    try: return array.shape,array.dtype
    finally: array._mmap.close()


def audit(directory,price_source=None):
    directory = Path(directory).resolve(); root = directory/'checkpoints'
    plan = json.loads((directory/'plan.json').read_text())
    pointer = json.loads((root/'current.json').read_text()); name = pointer['snapshot']
    require(isinstance(name,str) and len(name)==32 and all(c in '0123456789abcdef' for c in name),
        'Invalid snapshot identifier')
    snapshot = (root/name).resolve(); require(snapshot.parent==root,'Snapshot escaped checkpoint root')
    require(digest(snapshot/'manifest.json')==pointer['manifestSha256'],'Manifest checksum differs')
    manifest = json.loads((snapshot/'manifest.json').read_text())
    require({p.name for p in snapshot.iterdir()}==set(manifest)|{'manifest.json'},'Snapshot file inventory differs')
    size = 0
    for filename,checksum in manifest.items():
        path = (snapshot/filename).resolve()
        require(path.parent==snapshot,'Checkpoint file escaped snapshot')
        require(digest(path)==checksum,'Checkpoint file checksum differs: '+filename)
        size += path.stat().st_size
    meta = json.loads((snapshot/'metadata.json').read_text()); context = meta['context']
    require(meta['version']==1 and meta['kind']=='incremental-table','Incorrect checkpoint type')
    require(context['modelIdentity']==plan['checkpointSha256'],'Model identity differs')
    require(context['sourceHashes']==plan['sourceHashes'],'Replay implementation context differs')
    for source,checksum in plan['sourceHashes'].items():
        require(source.startswith('ml/') and '/' not in source[3:],'Unexpected source path')
        require(digest(directory/'source'/source[3:])==checksum,'Source snapshot differs: '+source)
    for source,checksum in meta['sources'].items():
        require(plan['sourceHashes'].get('ml/'+source)==checksum,'Table implementation differs: '+source)
    args = plan['arguments']; capacity,W,C,M = meta['shape']; update = meta['update']; H = meta['horizon']
    require(meta['grid']==plan['grid'],'Account grid differs')
    r = plan['rules']
    expected_rules = [r[key] for key in ('maker','taker','reserve_leverage','lower','upper',
        'maintenance_lower','maintenance_upper','minimum','maximum','tick')]+[plan['fundingMultiplier']]
    require(meta['rules']==expected_rules,'Execution parameters differ')
    require(capacity==meta['capacity']==args['horizon']==context['maximumHorizon'],'Ring capacity differs')
    require(meta['head']==update%capacity and H==min(capacity,args['seconds']-update),'Incorrect ring time boundary')
    require(meta['budget']==context['budget'],'Sampling budget differs')
    require(meta['seed']==args['seed'] and meta['candidates']==args['candidates'],'Action schedule differs')
    cfg = context['config']; origin = context['episodeOrigin']
    require(cfg['terminal_seconds']==origin+args['seconds'],'Terminal timestamp differs')
    require(cfg['seed']==args['seed'] and cfg['candidates']==args['candidates']
        and cfg['forecast_origin_index']==args['origin'],'Controller schedule differs')
    for filename,dtype in (('values.npy',np.float64),('visits.npy',np.uint32),('variance.npy',np.float32)):
        shape,actual_type = array_header(snapshot/filename)
        require(shape==tuple(meta['shape']) and actual_type==dtype,'Dense array contract differs: '+filename)
    with np.load(snapshot/'market.npz') as market:
        require(float(market['origin_seconds'])==origin+update,'Market snapshot timestamp differs')
        require(market['prices'].shape==(H+1,M) and market['destinations'].shape==(H,M,args['branches']),
            'Market snapshot dimensions differ')
        price = float(market['prices'][0,0])
    with np.load(snapshot/'front.npz') as front:
        state = front['state'].copy(); choice = int(front['choice']); scores = front['scores']
        require(np.isfinite(front['value']) and float(front['value'])==scores[choice],
            'Front value differs from selected proposal')
        require(np.array_equal(state,front['states'][choice]),'Front account differs from selected proposal')
        trades,fills = int(front['marketTrades']),int(front['fillTicks'])
    text = (directory/'updates.jsonl').read_text()
    lines = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    rows = [json.loads(line) for line in lines if line.strip()]
    require(1<=update<=len(rows),'Checkpoint is outside the accepted execution log')
    row = rows[update-1]
    require(row['update']==update and row['originSeconds']==origin+update,'Checkpoint log timestamp differs')
    require(np.array_equal(state,row['state']) and state[2]==price,'Checkpoint account differs from accepted log')
    with np.load(directory/plan['initialArchive']/'front.npz') as initial:
        initial_trades = int(initial['state'][0]!=10000.)
    require(trades==initial_trades+sum(bool(row['marketTrade']) for row in rows[:update]),'Trade count differs')
    require(fills==sum(bool(row['filled']) for row in rows[:update]),'Fill count differs')
    with np.load(snapshot/'errors.npz') as errors:
        require(all(errors[key].shape==(W*C,) for key in ('count','mean','m2')),'Error statistic shape differs')
        require(np.all((errors['count']>=0)&(errors['count']<=update)),
            'Correlated proposals were counted as independent observations')
    ledger = audit_ledger(directory,price_source)
    return dict(snapshot=name,checkpointUpdate=update,checkedBytes=size,checkedFiles=len(manifest),
        checkedLogUpdates=ledger['checkedUpdates'],checkpointState=state.tolist(),terminalTimestamp=cfg['terminal_seconds'],
        ledger=ledger,scope='Published snapshot integrity, implementation/model identity, ring/account boundary and independent execution ledger. No value-accuracy certificate.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory',type=Path)
    parser.add_argument('--price-source',type=Path); args = parser.parse_args()
    result = audit(args.directory,args.price_source)
    (args.directory/'checkpoint-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
