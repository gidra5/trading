"""Coordinate independent hierarchy searches through one batched CUDA worker."""
from __future__ import annotations
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from queue import Queue, Empty
from threading import Lock, Thread
import time
from hawkes_fit_cache import configuration_key, verified_complete, atomic_copy, FILES
from hawkes_latent import LatentModel
from hawkes_latent_minibatch import fit_configuration
from hawkes_latent_gpu_fit import fit_joint_batch, backend_identity, atomic_json
from hawkes_latent_search import search
from hawkes_research import load_corpus


class FitCoordinator:
    def __init__(self, corpus, shared_fits, particles=16, iterations=150, batch_fits=64, cpu_workers=6):
        if batch_fits < 1 or cpu_workers < 1 or particles < 2 or iterations < 1:
            raise ValueError('Invalid GPU coordinator settings')
        self.corpus, self.shared_fits = corpus, shared_fits
        self.particles, self.iterations = particles, iterations
        self.batch_fits, self.cpu_workers = batch_fits, cpu_workers
        self.queue, self.lock, self.futures = Queue(), Lock(), {}
        self.failed = None
        self.worker = Thread(target=self.run, name='CUDA-fit-batches', daemon=True)
        self.worker.start()

    def submit_many(self, jobs):
        waiting = []
        with self.lock:
            if self.failed is not None:
                raise RuntimeError('CUDA batch worker failed') from self.failed
            for seed, output in jobs:
                config = fit_configuration(seed, self.corpus, self.particles, self.iterations,
                    checkpoint_every=self.iterations, checkpoint_particles=max(32, 2 * self.particles))
                key = configuration_key(config)
                shared = self.shared_fits / key
                if verified_complete(shared, config):
                    for name in FILES:
                        atomic_copy(shared / name, output / name)
                    continue
                if key not in self.futures:
                    future = Future()
                    self.futures[key] = future
                    self.queue.put((key, seed, output, future))
                waiting.append((self.futures[key], output))
        for future, destination in waiting:
            source = future.result()
            if source != destination:
                for name in (*FILES, 'accelerator-resume.json'):
                    atomic_copy(source / name, destination / name)
                atomic_json(destination / 'gpu-shared-fit-provenance.json', {
                    'source': str(source.resolve()), 'backend': backend_identity(),
                    'identical_seed_and_optimizer_configuration': True})

    def run(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                batch = [item]
                deadline = time.monotonic() + 2
                while len(batch) < self.batch_fits:
                    try:
                        item = self.queue.get(timeout=max(.001, deadline - time.monotonic()))
                    except Empty:
                        break
                    if item is None:
                        raise RuntimeError('Coordinator closed with unfinished fits')
                    batch.append(item)
                    if time.monotonic() >= deadline:
                        break
                print(json.dumps({'gpu_fit_batch': len(batch), 'queued_fits': self.queue.qsize(),
                                  'directories': [str(t[2]) for t in batch]}), flush=True)
                fit_joint_batch([(t[1], t[2]) for t in batch], self.corpus, self.particles,
                                self.iterations, self.cpu_workers)
                for _, _, output, future in batch:
                    future.set_result(output)
        except BaseException as error:
            with self.lock:
                self.failed = error
                for future in self.futures.values():
                    if not future.done():
                        future.set_exception(error)

    def close(self):
        self.queue.put(None)
        self.worker.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--batch-fits', type=int, default=64)
    p.add_argument('--cpu-workers', type=int, default=6)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    corpus = load_corpus(Path(plan['cache']), 3600)
    coordinator = FitCoordinator(corpus, Path(plan['shared_fits']), plan['particles'], plan['iterations'], args.batch_fits, args.cpu_workers)
    try:
        with ThreadPoolExecutor(max_workers=len(plan['searches'])) as pool:
            futures = []
            for job in plan['searches']:
                seeds = [LatentModel.deserialize(json.loads(Path(path).read_text())['model']) for path in job['initial']]
                futures.append(pool.submit(search, seeds, corpus, Path(job['output']), job['label'],
                    plan['particles'], plan['iterations'], plan['max_modes'], plan['max_anchors'],
                    plan['max_diffusions'], plan['rounds'], None, coordinator.submit_many))
            for f in futures:
                f.result()
    finally:
        coordinator.close()


if __name__ == '__main__':
    main()
