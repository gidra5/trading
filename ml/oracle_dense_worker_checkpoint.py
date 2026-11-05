"""Checksummed, atomic-boundary checkpoints for the persistent paper worker.

Each snapshot is immutable. A small pointer is replaced only after the entire
snapshot has been written and flushed. Failed saves leave the previous pointer
usable. Checkpointing happens between requests, never midway through a tick.
"""
from dataclasses import asdict
import hashlib
import json
import os
import shutil
from pathlib import Path
import uuid
import numpy as np
from oracle_dense_controller import DenseController
from oracle_dense_value_map import DenseValueMap
from oracle_dense_repair import contract


SOURCES = ('oracle_book_mc.py', 'oracle_book_gpu.py', 'oracle_account_grid.py',
    'oracle_dense_value_map.py', 'oracle_dense_repair.py', 'oracle_dense_rollout.py',
    'oracle_dense_controller.py', 'oracle_dense_metadata.py', 'oracle_dense_observe_batch.py',
    'serve_oracle_dense.py', 'oracle_dense_worker_checkpoint.py',
    'nsf_scenario_graph.py', 'export_nsf_planning_trees.py',
    'oracle_incremental_controller.py','oracle_incremental_table.py','oracle_incremental_checkpoint.py',
    'oracle_root_aliases.py',
    'oracle_selected_outcomes.py','oracle_guided_sampling.py','oracle_root_candidates.py',
    'oracle_local_market_refresh.py','oracle_incremental_observation.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    return {'ml/'+name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def flushed_json(path, value):
    with path.open('w', encoding='utf8') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False))
        f.flush()
        os.fsync(f.fileno())


def incremental_context(worker):
    return dict(modelIdentity=worker.model_identity,settings=asdict(worker.settings),sources=worker.source_identity)


def save_worker(worker, root):
    c = worker.controller
    if worker.failed or c is None or c.phase != 'active':
        raise ValueError('Checkpoint requires a successful active decision boundary')
    if not worker.model_identity:
        raise ValueError('Checkpoint requires a verified model identity')
    if sources() != worker.source_identity:
        raise ValueError('Implementation changed since worker startup')
    if worker.settings.planner=='incremental':
        from oracle_incremental_checkpoint import save_table
        front = dict(c.front,episodeOrigin=c.episode_origin,maximumHorizon=c.maximum_horizon,
            terminalSeconds=c.config.terminal_seconds)
        saved = save_table(c.current,front,incremental_context(worker),root)
        return dict(saved,timestamp=float(c.current.cloud.origin_seconds))
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous_pointer = json.loads((root/'current.json').read_text()) if (root/'current.json').exists() else None
    identifier = uuid.uuid4().hex
    snapshot = root/identifier
    snapshot.mkdir()
    plan = dict(grid=asdict(c.current.grid), rules=asdict(worker.settings.rules),
        fundingMultiplier=worker.settings.funding, sourceHashes=worker.source_identity)
    flushed_json(snapshot/'plan.json', plan)
    c.current.export(snapshot/'map')
    np.savez(snapshot/'front.npz', **c.front)
    runtime = dict(version=1, modelIdentity=worker.model_identity, settings=asdict(worker.settings),
        config=asdict(c.config), episodeOrigin=c.episode_origin, maximumHorizon=c.maximum_horizon,
        update=c.update, timestamp=float(c.current.cloud.origin_seconds), state=c.state.tolist())
    flushed_json(snapshot/'runtime.json', runtime)
    files = sorted(p for p in snapshot.rglob('*') if p.is_file())
    for path in files:
        with path.open('rb+') as f:
            os.fsync(f.fileno())
    manifest = {p.relative_to(snapshot).as_posix(): digest(p) for p in files}
    flushed_json(snapshot/'manifest.json', manifest)
    pointer = dict(snapshot=identifier, manifestSha256=digest(snapshot/'manifest.json'))
    pending = root/(identifier+'.pointer.tmp')
    flushed_json(pending, pointer)
    os.replace(pending, root/'current.json')
    keep = {identifier}
    if previous_pointer:
        keep.add(previous_pointer['snapshot'])
    if c.current.directory is not None:
        # A restored map may still have open Windows mmap handles.
        loaded = Path(c.current.directory).resolve().parent
        if loaded.parent == root:
            keep.add(loaded.name)
    removed, retained = [], []
    for old in root.iterdir():
        name = old.name
        if (name in keep or len(name) != 32 or any(x not in '0123456789abcdef' for x in name)
                or not old.is_dir() or not (old/'manifest.json').is_file()):
            continue
        target = old.resolve()
        # Delete only snapshots within this managed root; never traverse links outside it.
        if target.parent != root or any(target not in p.resolve().parents for p in target.rglob('*')):
            retained.append(name)
            continue
        try:
            shutil.rmtree(target)
            removed.append(name)
        except OSError:
            retained.append(name)  # A committed checkpoint remains valid even if cleanup is deferred.
    return dict(snapshot=identifier, update=c.update, timestamp=runtime['timestamp'],
                files=len(manifest), bytes=sum(p.stat().st_size for p in files),
                removedSnapshots=removed, deferredCleanup=retained)


def restore_worker(worker, root):
    if worker.controller is not None or worker.failed:
        raise ValueError('Restore requires a fresh worker')
    if worker.settings.planner=='incremental':
        from oracle_incremental_checkpoint import load_table
        from oracle_incremental_controller import IncrementalController
        if not worker.model_identity or sources()!=worker.source_identity:
            raise ValueError('Checkpoint requires an unchanged implementation and verified model identity')
        table,front = load_table(root,incremental_context(worker))
        origin = float(front.pop('episodeOrigin')); horizon = int(front.pop('maximumHorizon'))
        terminal = float(front.pop('terminalSeconds'))
        c = IncrementalController(table,worker.model,front['state'],worker.settings.controller_config(terminal),
            front=front,episode_origin=origin,maximum_horizon=horizon,predict=worker.predict,
            budget=worker.settings.table_budget,root_draws=worker.settings.root_draws)
        worker.controller = c
        return dict(update=c.update,timestamp=float(table.cloud.origin_seconds),planner='incremental')
    root = Path(root).resolve()
    pointer = json.loads((root/'current.json').read_text())
    identifier = pointer['snapshot']
    if not isinstance(identifier, str) or len(identifier) != 32 or any(x not in '0123456789abcdef' for x in identifier):
        raise ValueError('Invalid checkpoint snapshot identifier')
    snapshot = root/identifier
    if digest(snapshot/'manifest.json') != pointer['manifestSha256']:
        raise ValueError('Checkpoint manifest checksum differs')
    manifest = json.loads((snapshot/'manifest.json').read_text())
    for name, expected in manifest.items():
        path = (snapshot/name).resolve()
        if snapshot not in path.parents or digest(path) != expected:
            raise ValueError('Checkpoint file checksum differs: '+name)
    actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
    if actual != set(manifest) | {'manifest.json'}:
        raise ValueError('Checkpoint contains unexpected or missing files')
    runtime = json.loads((snapshot/'runtime.json').read_text())
    plan = json.loads((snapshot/'plan.json').read_text())
    if runtime['version'] != 1 or not worker.model_identity or runtime['modelIdentity'] != worker.model_identity:
        raise ValueError('Checkpoint model or format differs')
    if (canonical(runtime['settings']) != canonical(asdict(worker.settings))
            or plan['sourceHashes'] != worker.source_identity or sources() != worker.source_identity):
        raise ValueError('Checkpoint configuration or implementation differs')
    config = worker.settings.controller_config(runtime['config']['terminal_seconds'])
    if canonical(runtime['config']) != canonical(asdict(config)):
        raise ValueError('Checkpoint controller configuration differs')
    current = DenseValueMap.load(snapshot/'map')
    if (current.bellman_contract != contract(config.seed, config.candidates)
            or current.bellman_residuals is None or np.any(~np.isfinite(current.bellman_residuals))
            or np.any(current.bellman_residuals < 0)
            or current.bellman_residuals.shape != (current.shape[0],)
            or not np.all(current.samples == worker.settings.branches)):
        raise ValueError('Checkpoint lacks a complete checked value census')
    with np.load(snapshot/'front.npz') as values:
        front = {name: values[name] for name in values.files}
    state = np.asarray(runtime['state'], dtype=float)
    c = DenseController(current, worker.model, state, config, front=front,
        episode_origin=runtime['episodeOrigin'], maximum_horizon=runtime['maximumHorizon'], predict=worker.predict)
    if c.update != runtime['update'] or current.cloud.origin_seconds != runtime['timestamp']:
        raise ValueError('Checkpoint update and observation boundary disagree')
    if (front['scores'].shape != (len(current.a['coordinates']),)
            or front['states'].shape != (len(current.a['coordinates']), len(state))):
        raise ValueError('Checkpoint front proposal shape differs')
    worker.controller = c
    return dict(snapshot=identifier, update=c.update, timestamp=runtime['timestamp'])
