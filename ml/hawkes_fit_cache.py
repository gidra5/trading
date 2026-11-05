"""Share numerically identical completed joint fits across nested-model searches.

The full optimizer configuration, initial model and corpus identify a fit. An OS
file lock allows one producer at a time and releases automatically on process
exit; an abandoned lock file is not evidence that a producer is still running.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from hawkes_research import save_json

FILES = ('joint-progress.json', 'best-validation.json', 'fit-result.json')


def configuration_key(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@contextmanager
def exclusive_fit(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        announced = False
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
            except OSError as error:
                if error.errno not in (11, 13, 35, 36):
                    raise
            if not announced:
                print(f'Waiting for identical shared joint fit {path.parent.name}', flush=True)
                announced = True
            time.sleep(.2)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verified_complete(directory, config):
    if not all((directory / name).exists() for name in FILES):
        return False
    progress = json.loads((directory / 'joint-progress.json').read_text())
    if progress['config'] != config or progress['next_iteration'] != config['iterations']:
        raise ValueError('Completed joint fit does not match the requested optimization protocol')
    return True


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def shared_joint_fit(directory, config, cache, producer=None):
    key = configuration_key(config)
    shared = cache / key
    with exclusive_fit(shared / '.lock'):
        config_path = shared / 'config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Shared fit fingerprint collision')
        else:
            save_json(config_path, config)
        if verified_complete(shared, config):
            if not verified_complete(directory, config):
                for name in FILES:
                    atomic_copy(shared / name, directory / name)
                save_json(directory / 'shared-fit-provenance.json', {
                    'configuration_sha256': key, 'source': str(shared.resolve()),
                    'fit_sha256': hashlib.sha256((shared / 'fit-result.json').read_bytes()).hexdigest(),
                    'parameters_refitted_with_identical_configuration': True})
            return
        if not verified_complete(directory, config):
            if producer is None:
                raise ValueError('Only completed joint fits can warm the cache')
            producer()
        if not verified_complete(directory, config):
            raise ValueError('Producer did not finish the requested joint fit')
        for name in FILES:
            atomic_copy(directory / name, shared / name)
        save_json(shared / 'source.json', {'directory': str(directory.resolve()),
                  'fit_sha256': hashlib.sha256((directory / 'fit-result.json').read_bytes()).hexdigest()})


def warm_completed(searches, cache):
    published = 0
    for search in searches:
        for result in sorted((search / 'candidates').glob('*/fit-result.json')):
            progress_path = result.parent / 'joint-progress.json'
            if not progress_path.exists():
                continue
            progress = json.loads(progress_path.read_text())
            config = progress['config']
            if progress['next_iteration'] != config['iterations']:
                continue
            shared_joint_fit(result.parent, config, cache)
            published += 1
    return published
