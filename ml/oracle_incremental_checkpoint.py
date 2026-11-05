"""Atomic snapshots of the resident table and its actual sampling state.

The ring, visit statistics, LRU order and control vectors are preserved. This
format deliberately cannot be loaded as a certified DenseValueMap archive.
The caller supplies the episode/model context and the accepted root decision.
"""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import numpy as np
import torch
from oracle_account_grid import AccountGrid,CellErrors
from oracle_dense_value_map import MarketCloud
from oracle_incremental_table import IncrementalTable, TableBudget


SOURCES = ('oracle_incremental_checkpoint.py','oracle_incremental_table.py',
    'oracle_root_aliases.py',
    'oracle_selected_outcomes.py','oracle_guided_sampling.py','oracle_root_candidates.py',
    'oracle_local_market_refresh.py','oracle_dense_metadata.py','oracle_book_mc.py',
    'oracle_book_gpu.py','oracle_account_grid.py','oracle_dense_value_map.py','oracle_incremental_observation.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def signatures():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCES}


def write_json(path,value):
    with path.open('w',encoding='utf8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False))
        stream.flush(); os.fsync(stream.fileno())


def validate_front(table,front):
    state = np.asarray(front['state']); scores = np.asarray(front['scores'])
    E = len(table.a['exposures']); slots = table.C*2+(1<<table.grid.max_orders)*(E+1)+1
    if (state.shape!=(4+3*table.grid.max_orders,) or not np.isfinite(state).all()
            or state[2]!=table.cloud.prices[0,0] or scores.shape!=(slots,)
            or np.asarray(front['states']).shape!=(slots,len(state))):
        raise ValueError('Checkpoint front differs from the current account grid')
    choice = int(front['choice'])
    if (not 0<=choice<slots or not np.isfinite(front['value'])
            or float(front['value'])!=scores[choice]
            or not np.array_equal(state,front['states'][choice])):
        raise ValueError('Checkpoint front does not identify its executed proposal')
    return slots


def save_table(table,front,context,root):
    if table.failed: raise ValueError('Cannot checkpoint a failed incremental table')
    if table.source_identity!=signatures(): raise ValueError('Implementation changed since table startup')
    validate_front(table,front)
    # Validate external context before writing any arrays; use plain JSON only.
    json.dumps(context,allow_nan=False)
    root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    previous = json.loads((root/'current.json').read_text()) if (root/'current.json').exists() else None
    identifier = uuid.uuid4().hex; snapshot = root/identifier; snapshot.mkdir()
    metadata = dict(version=1,kind='incremental-table',grid=asdict(table.grid),rules=table.rules.tolist(),
        capacity=table.capacity,horizon=table.H,head=table.head,update=table.update,
        shape=list(table.values.shape),seed=table.seed,candidates=table.candidates,budget=asdict(table.budget),
        sources=table.source_identity,context=context,rootControls=table.root_controls is not None)
    table.cloud.save(snapshot)
    np.save(snapshot/'values.npy',table.values.cpu().numpy())
    np.save(snapshot/'visits.npy',table.visits)
    np.save(snapshot/'variance.npy',table.sampling_variance)
    np.savez(snapshot/'errors.npz',count=table.errors.count,mean=table.errors.mean,m2=table.errors.m2)
    keys = np.asarray(list(table.controls),dtype=np.int64).reshape(-1,2)
    controls = np.asarray(list(table.controls.values()),dtype=np.float64).reshape(-1,table.B)
    np.savez(snapshot/'controls.npz',keys=keys,values=controls)
    if table.root_controls is not None: np.save(snapshot/'root-controls.npy',table.root_controls)
    np.savez(snapshot/'front.npz',**front)
    write_json(snapshot/'metadata.json',metadata)
    files = sorted(path for path in snapshot.iterdir() if path.is_file())
    for path in files:
        with path.open('rb+') as stream: os.fsync(stream.fileno())
    write_json(snapshot/'manifest.json',{path.name:digest(path) for path in files})
    pending = root/(identifier+'.tmp')
    write_json(pending,dict(snapshot=identifier,manifestSha256=digest(snapshot/'manifest.json')))
    os.replace(pending,root/'current.json')
    keep = {identifier}
    if previous: keep.add(previous['snapshot'])
    removed = []
    for old in root.iterdir():
        if (old.name in keep or len(old.name)!=32 or any(c not in '0123456789abcdef' for c in old.name)
                or not old.is_dir() or not (old/'manifest.json').is_file()): continue
        target = old.resolve()
        if target.parent!=root or any(target not in p.resolve().parents for p in target.rglob('*')): continue
        try:
            shutil.rmtree(target); removed.append(old.name)
        except OSError:
            pass  # Publishing succeeded; a busy old snapshot can be pruned next time.
    return dict(snapshot=identifier,update=table.update,bytes=sum(path.stat().st_size for path in files),
        removedSnapshots=removed)


def load_table(root,expected_context):
    root = Path(root).resolve(); pointer = json.loads((root/'current.json').read_text())
    identifier = pointer['snapshot']
    if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Invalid snapshot identifier')
    snapshot = (root/identifier).resolve()
    if snapshot.parent!=root or digest(snapshot/'manifest.json')!=pointer['manifestSha256']:
        raise ValueError('Invalid snapshot path or manifest checksum')
    manifest = json.loads((snapshot/'manifest.json').read_text())
    if {p.name for p in snapshot.iterdir()}!=set(manifest)|{'manifest.json'}:
        raise ValueError('Unexpected or missing checkpoint files')
    for name,checksum in manifest.items():
        path = (snapshot/name).resolve()
        if path.parent!=snapshot or digest(path)!=checksum: raise ValueError('Checkpoint checksum differs: '+name)
    meta = json.loads((snapshot/'metadata.json').read_text())
    if meta['version']!=1 or meta['kind']!='incremental-table' or meta['sources']!=signatures():
        raise ValueError('Checkpoint implementation differs')
    canonical = lambda x:json.dumps(x,sort_keys=True,allow_nan=False)
    if canonical(meta['context'])!=canonical(expected_context): raise ValueError('Checkpoint episode/model context differs')
    table = IncrementalTable.__new__(IncrementalTable)
    table.source_identity = meta['sources']
    grid = dict(meta['grid'])
    for name in ('wealth_points','exposure_points'): grid[name] = tuple(grid[name])
    table.grid = AccountGrid(**grid); table.a = table.grid.arrays()
    table.rules = np.asarray(meta['rules'],dtype=np.float64)
    if table.rules.shape!=(11,) or not np.isfinite(table.rules).all(): raise ValueError('Invalid execution rules')
    table.cloud = MarketCloud.load(snapshot/'market.npz'); table.B = table.cloud.destinations.shape[-1]
    table.capacity,table.W,table.C,table.M = meta['shape']
    table.H,table.head,table.update = meta['horizon'],meta['head'],meta['update']
    if (not 1<=table.H<=table.capacity or not 0<=table.head<table.capacity or table.update<0
            or table.capacity!=meta['capacity'] or table.H!=len(table.cloud.counts)-1
            or (table.W,table.C,table.M)!=(len(table.a['wealth']),len(table.a['coordinates']),table.cloud.prices.shape[1])):
        raise ValueError('Invalid ring dimensions or time boundary')
    table.seed,table.candidates = meta['seed'],meta['candidates']; table.budget = TableBudget(**meta['budget'])
    if table.budget.draws>table.B: raise ValueError('Invalid conditional sample budget')
    # Own this temporary upload buffer; no Windows mmap handle survives restore.
    values = np.load(snapshot/'values.npy')
    table.visits = np.load(snapshot/'visits.npy'); table.sampling_variance = np.load(snapshot/'variance.npy')
    with np.load(snapshot/'errors.npz') as archive:
        table.errors = CellErrors(archive['count'],archive['mean'],archive['m2'])
    if (any(x.shape!=(table.W*table.C,) for x in (table.errors.count,table.errors.mean,table.errors.m2))
            or table.errors.count.dtype!=np.int64 or np.any(table.errors.count<0)
            or np.any(~np.isfinite(table.errors.mean)) or np.any(~np.isfinite(table.errors.m2)|(table.errors.m2<0))):
        raise ValueError('Invalid observed residual statistics')
    shape = tuple(meta['shape'])
    if (values.shape!=shape or values.dtype!=np.float64 or table.visits.shape!=shape or table.visits.dtype!=np.uint32
            or table.sampling_variance.shape!=shape or table.sampling_variance.dtype!=np.float32):
        raise ValueError('Saved dense arrays differ from ring dimensions')
    with np.load(snapshot/'controls.npz') as archive:
        keys,controls = archive['keys'],archive['values']
    if (keys.shape!=(len(controls),2) or controls.shape!=(len(keys),table.B)
            or len(keys)>table.budget.cached_rows or keys.dtype!=np.int64
            or len({tuple(key) for key in keys})!=len(keys)):
        raise ValueError('Invalid cached control rows')
    if len(keys) and (np.any(keys[:,0]<table.cloud.origin_seconds)
            or np.any(keys[:,0]>=table.cloud.origin_seconds+table.H)
            or np.any((keys[:,1]<0)|(keys[:,1]>=table.W*table.C*table.M))):
        raise ValueError('Cached controls refer to expired or unknown cells')
    table.controls = OrderedDict((tuple(map(int,key)),value.copy()) for key,value in zip(keys,controls))
    table.root_controls = np.load(snapshot/'root-controls.npy') if meta['rootControls'] else None
    with np.load(snapshot/'front.npz') as archive: front = {key:archive[key] for key in archive.files}
    slots = validate_front(table,front)
    if table.root_controls is not None and table.root_controls.shape!=(slots+1,table.B):
        raise ValueError('Invalid root control dimensions')
    # Upload only after checksums, identity, grid and all sampler state validate.
    table.values = torch.tensor(np.asarray(values),device='cuda',dtype=torch.float64)
    table.failed = False
    return table,front
