"""Batch independent joint fits on CUDA; retain CPU checkpoint likelihoods.

The optimizer configuration and completed CPU fits are preserved. A resumed
trajectory retains its raw vector, Adam moments, iteration and optimizer RNG.
CUDA particle draws change at the recorded transition; no bitwise equivalence
to the CPU trajectory is claimed. Every retained coordinate remains free.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from hawkes_latent import decode, filter_model, physical_arrays, core_branching, raw_bounds
from hawkes_latent_dispatch import evaluate_batches, CPU_THRESHOLD, CPU_WORKERS
from hawkes_latent_minibatch import (fit_configuration, coordinate_scales, partitions,
                                    model_context, observed_initializations)
from hawkes_research import save_json, radius


def backend_identity():
    names = ('hawkes_latent_gpu_fit.py', 'hawkes_latent_gpu.py',
             'hawkes_latent_cuda.cu', 'hawkes_cuda_runtime.py', 'hawkes_latent_minibatch.py', 'hawkes_latent.py', 'hawkes_latent_dispatch.py')
    return {'backend': 'Hybrid float64 adapted filter: CUDA Philox and CPU NumPy matched pairs',
            'dispatch': {'cpu_threshold': CPU_THRESHOLD, 'cpu_workers': CPU_WORKERS, 'plus_minus_generator_pairing': True},
            'checkpoint_backend': 'CPU reference, seed 71391',
            'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
            'particle_rng_transition_changes_optimizer_trajectory': True}


def atomic_json(path, value):
    save_json(path, value)


class FitState:
    def __init__(self, seed, corpus, output, particles, iterations, **settings):
        self.seed, self.corpus, self.output = seed, corpus, output
        output.mkdir(parents=True, exist_ok=True)
        self.config = fit_configuration(seed, corpus, particles, iterations, **settings)
        self.layout, self.raw = seed.layout(), seed.raw()
        self.scales = coordinate_scales(self.layout)
        self.first, self.second = np.zeros_like(self.raw), np.zeros_like(self.raw)
        self.rng = np.random.default_rng(self.config['optimizer_seed'])
        self.history, self.step = [], 0
        progress_path = output / 'joint-progress.json'
        if progress_path.exists():
            p = json.loads(progress_path.read_text())
            if p['config'] != self.config:
                raise ValueError(f'Joint fit configuration mismatch: {output}')
            self.raw = np.array(p['raw'])
            self.first, self.second = np.array(p['first_moment']), np.array(p['second_moment'])
            self.rng.bit_generator.state = p['rng_state']
            self.history, self.step = p['history'], p['next_iteration']
        if not 0 <= self.step <= iterations:
            raise ValueError('Invalid resume iteration')
        self.best_raw, self.best_validation = self.raw.copy(), np.inf
        best_path = output / 'best-validation.json'
        if best_path.exists():
            best = json.loads(best_path.read_text())
            self.best_raw, self.best_validation = np.array(best['raw']), best['validation_nll_without_factorial']
            # Initial CPU scoring can finish asynchronously before its history
            # row is committed. Preserve that authoritative checkpoint on resume.
            if not any(h.get('full_data_checkpoint') and h['iteration'] == best['iteration'] for h in self.history):
                row = {k: v for k, v in best.items() if k not in ('model', 'raw')}
                existing = next((h for h in self.history if h['iteration'] == best['iteration']), None)
                if existing is None:
                    self.history.insert(0, row)
                else:
                    existing.update(row)
        self.blocks = partitions(corpus.score[:corpus.train_end], self.config['window_seconds'])
        self.total_scored = int(corpus.score[:corpus.train_end].sum())
        if not len(self.blocks) or not self.total_scored:
            raise ValueError('No scored training blocks')
        self.started = time.monotonic()
        self.backend = backend_identity()
        transition_path = output / 'accelerator-resume.json'
        if transition_path.exists():
            previous = json.loads(transition_path.read_text())
            if previous['backend'] != self.backend:
                manifest_path = output / 'backend-migration.json'
                if not manifest_path.exists():
                    raise ValueError('Backend changed without a checkpoint-bound migration manifest')
                manifest = json.loads(manifest_path.read_text())
                if (manifest['target_backend'] != self.backend
                    or manifest['progress_sha256'] != hashlib.sha256(progress_path.read_bytes()).hexdigest()
                    or manifest['previous_transition_sha256'] != hashlib.sha256(transition_path.read_bytes()).hexdigest()
                    or manifest['next_iteration'] != self.step):
                    raise ValueError('Backend migration does not bind to this exact checkpoint')
                self.transition = {**previous, 'backend': self.backend,
                    'numerical_backend_transitions': previous.get('numerical_backend_transitions', []) + [{
                        'at_iteration': self.step, 'from_backend': previous['backend'],
                        'source_progress_sha256': manifest['progress_sha256'],
                        'previous_transition_sha256': manifest['previous_transition_sha256']}]}
                atomic_json(transition_path, self.transition)
            else:
                self.transition = previous
        else:
            self.transition = {'backend': self.backend, 'first_accelerator_iteration': self.step, 'first_cuda_iteration': None,
                               'optimizer_state_preserved': True,
                               'source_progress_sha256': hashlib.sha256(progress_path.read_bytes()).hexdigest() if progress_path.exists() else None}
            atomic_json(transition_path, self.transition)

    def prepare(self):
        step, config = self.step, self.config
        self.direction = self.rng.choice(np.array([-1., 1.]), len(self.raw))
        self.width = .04 / (1 + step / 100) ** .101
        chosen = self.blocks[self.rng.integers(len(self.blocks), size=config['batch_windows'])]
        seeds = self.rng.integers(1, 2**30, size=config['batch_windows'])
        models = [decode(self.raw + sign * self.width * self.scales * self.direction, self.layout, self.seed.link) for sign in (1, -1)]
        contexts = [model_context(m, self.corpus, config['minimum_context']) for m in models]
        context = None if any(c is None for c in contexts) else max(contexts)
        if context is not None and np.sum(chosen[:, 1] - np.maximum(0, chosen[:, 0] - context)) >= np.max(chosen[:, 1]):
            context = None
        self.context_info = {'full_prefix': context is None}
        trajectories = []
        for leg, model in enumerate(models):
            if context is None:
                end = int(chosen[:, 1].max())
                trajectories.append((self, leg, model, 0, end, int(seeds[0]), None, None, chosen))
                self.context_info['particle_seconds'] = end
            else:
                begins = np.maximum(0, chosen[:, 0] - context)
                starts, reverse = np.unique(begins, return_inverse=True)
                arrays = physical_arrays(model, self.corpus.initial_mean)
                fast, slow = observed_initializations(self.corpus.y[:self.corpus.train_end], self.corpus.valid[:self.corpus.train_end],
                    arrays[2], arrays[4], arrays[5], arrays[6], arrays[7], arrays[8], arrays[9], starts)
                for j, ((s, e), begin, seed) in enumerate(zip(chosen, begins, seeds)):
                    trajectories.append((self, leg, model, int(begin), int(e), int(seed), fast[reverse[j]], slow[reverse[j]], np.array([[s, e]])))
                self.context_info.update(context_seconds=context, particle_seconds=int(np.sum(chosen[:, 1] - begins)))
        return trajectories

    def update(self, losses, backends=None):
        if backends is not None and backends['cuda_philox'] and self.transition['first_cuda_iteration'] is None:
            self.transition['first_cuda_iteration'] = self.step
            atomic_json(self.output / 'accelerator-resume.json', self.transition)
        plus, minus = np.asarray(losses) * len(self.blocks) / (self.config['batch_windows'] * self.total_scored)
        if np.isfinite(plus) and np.isfinite(minus):
            gradient = (plus - minus) * self.direction / (2 * self.width * self.scales)
            self.first = .9 * self.first + .1 * gradient
            self.second = .999 * self.second + .001 * gradient ** 2
            correction = self.first / (1 - .9 ** (self.step + 1)) / (np.sqrt(self.second / (1 - .999 ** (self.step + 1))) + 1e-8)
            self.raw -= self.config['learning_rate'] / (1 + self.step / 500) ** .602 * self.scales * correction
            self.raw = np.clip(self.raw, *raw_bounds(self.layout))
        row = {'iteration': self.step, 'plus_stochastic_train_loss': float(plus), 'minus_stochastic_train_loss': float(minus),
               'plus_context': self.context_info, 'minus_context': self.context_info,
               'seconds_elapsed': time.monotonic() - self.started, 'particle_backend': self.backend['backend'],
               'stochastic_particle_backends': backends}
        self.step += 1
        self.history.append(row)
        self.save()

    def save(self):
        atomic_json(self.output / 'joint-progress.json', {'config': self.config, 'raw': self.raw.tolist(),
            'first_moment': self.first.tolist(), 'second_moment': self.second.tolist(),
            'rng_state': self.rng.bit_generator.state, 'history': self.history,
            'next_iteration': self.step, 'accelerator_transition': self.transition})

    def checkpoint(self, iteration, theta):
        c = self.corpus
        model = decode(theta, self.layout, self.seed.link)
        result = filter_model(model, c.y, c.valid, c.initial_mean, self.config['checkpoint_particles'], 71391)
        mask = c.score[c.train_end:]
        validation = float(result[1][c.train_end:][mask].mean())
        row = {'iteration': iteration, 'train_nll_without_factorial': float(result[1][:c.train_end][c.score[:c.train_end]].mean()),
               'validation_nll_without_factorial': validation, 'full_data_checkpoint': True,
               'checkpoint_particles': self.config['checkpoint_particles'],
               'ess_validation_mean_fraction': float(result[2][c.train_end:][mask].mean() / self.config['checkpoint_particles']),
               'seconds_elapsed': time.monotonic() - self.started}
        if validation < self.best_validation:
            self.best_raw, self.best_validation = theta.copy(), validation
            atomic_json(self.output / 'best-validation.json', {'model': model.serialize(), 'raw': theta.tolist(), **row})
        return row

    def finish(self):
        model = decode(self.best_raw, self.layout, self.seed.link)
        atomic_json(self.output / 'fit-result.json', {'model': model.serialize(), 'raw': self.best_raw.tolist(),
            'joint_parameters': len(self.raw), 'validation_nll_without_factorial': self.best_validation,
            'particles_fit': self.config['particles'], 'optimizer': 'Batched all-parameter preconditioned SPSA; CPU full-data checkpoint selection',
            'iterations': self.config['iterations'], 'convergence_established': False,
            'spectral_radius': radius(core_branching(model)), 'accelerator_transition': self.transition,
            'approximation': 'Adaptive latent-context optimizer only; CPU full conditional likelihood used for checkpoint comparison'})


def evaluate_trajectories(trajectories, corpus, particles, max_trajectories=128, backend_counts=None):
    """Group by trajectory length so short windows are not padded to a prefix."""
    ordered = sorted(trajectories, key=lambda t: t[4] - t[3])
    batches = []
    current = []
    for t in ordered:
        if current and (len(current) >= max_trajectories or (t[4] - t[3]) > 2 * (current[0][4] - current[0][3])):
            batches.append(current)
            current = []
        current.append(t)
    if current:
        batches.append(current)
    losses, counts = evaluate_batches(batches, corpus, particles)
    if backend_counts is not None:
        backend_counts.update(counts)
    return losses, [len(b) for b in batches]


def fit_joint_batch(jobs, corpus, particles=16, iterations=150, cpu_workers=6, **settings):
    settings.setdefault('checkpoint_every', iterations)
    settings.setdefault('checkpoint_particles', max(32, 2 * particles))
    states = [FitState(seed, corpus, output, particles, iterations, **settings) for seed, output in jobs if not (output / 'fit-result.json').exists()]
    if not states:
        return
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool, ThreadPoolExecutor(max_workers=2) as preparation_pool:
        # Independent initial CPU checkpoints can overlap GPU training.
        initial = {id(s): pool.submit(s.checkpoint, -1, s.seed.raw()) for s in states
                   if not any(h.get('full_data_checkpoint') for h in s.history) and not (s.output / 'best-validation.json').exists()}
        for s in states:
            if s.step >= iterations and id(s) in initial:
                s.history.append(initial.pop(id(s)).result())
        cycle = 0
        while active := [s for s in states if s.step < iterations]:
            begin = time.monotonic()
            prepared = list(preparation_pool.map(lambda s: s.prepare(), active))
            backend_counts = {}
            losses, batches = evaluate_trajectories([t for group in prepared for t in group], corpus, particles, backend_counts=backend_counts)
            checkpoints = []
            for s in active:
                s.update(losses[id(s)], backend_counts[id(s)])
                if s.step % s.config['checkpoint_every'] == 0 or s.step == iterations:
                    if id(s) in initial:
                        s.history.insert(0, initial.pop(id(s)).result())
                    checkpoints.append((s, pool.submit(s.checkpoint, s.step - 1, s.raw.copy())))
            for s, future in checkpoints:
                s.history[-1].update(future.result())
                s.save()
            cycle += 1
            print(json.dumps({'gpu_cycle': cycle, 'active_fits': len(active), 'trajectory_batches': batches,
                              'iteration_min': min(s.step for s in active), 'iteration_max': max(s.step for s in active),
                              'cpu_trajectories': sum(c['cpu_numpy'] for c in backend_counts.values()),
                              'cuda_trajectories': sum(c['cuda_philox'] for c in backend_counts.values()),
                              'cycle_seconds': time.monotonic() - begin}), flush=True)
        for s in states:
            if id(s) in initial:
                s.history.insert(0, initial[id(s)].result())
            # A crash after the last update but before its CPU checkpoint must
            # still score that updated vector when resumed.
            if not any(h.get('full_data_checkpoint') and h['iteration'] == iterations - 1 for h in s.history):
                s.history.append(s.checkpoint(iterations - 1, s.raw))
            s.save()
            s.finish()
