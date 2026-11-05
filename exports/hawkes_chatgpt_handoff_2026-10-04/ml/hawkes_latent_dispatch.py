"""Float64 trajectory dispatch, preserving paired optimizer random streams."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from hawkes_latent import filter_model, physical_arrays
from hawkes_latent_gpu import filter_batch

CPU_THRESHOLD = 8
CPU_WORKERS = 6


def pair_key(t):
    return id(t[0]), t[3], t[4], t[5]


def cpu_loss(t, corpus, particles):
    result = filter_model(t[2], corpus.y[t[3]:t[4]], corpus.valid[t[3]:t[4]],
        corpus.initial_mean, particles, t[5], initial_fast=t[6], initial_slow=t[7])
    return sum(float(result[1][s-t[3]:e-t[3]][corpus.score[s:e]].sum()) for s, e in t[8])


def cuda_losses(batch, corpus, particles):
    fast, slow = [], []
    weights = np.zeros((len(batch), max(t[4] - t[3] for t in batch)), dtype=np.int8)
    for i, t in enumerate(batch):
        arrays = physical_arrays(t[2], corpus.initial_mean)
        fast.append(arrays[8][arrays[2]] if t[6] is None else t[6])
        slow.append(arrays[8][arrays[5]] if t[7] is None else t[7])
        for s, e in t[8]:
            weights[i, s-t[3]:e-t[3]] += corpus.score[s:e]
    return filter_batch([t[2] for t in batch], corpus.y[:corpus.train_end], corpus.valid[:corpus.train_end],
        corpus.initial_mean, particles, [t[5] for t in batch], initial_fast=fast, initial_slow=slow,
        segments=np.array([[t[3], t[4]] for t in batch]), loss_weights=weights)


def evaluate_batches(batches, corpus, particles, cpu_threshold=CPU_THRESHOLD, cpu_workers=CPU_WORKERS):
    if cpu_threshold < 0 or cpu_workers < 1:
        raise ValueError('Invalid trajectory dispatch settings')
    # A plus/minus pair must use the same generator even when a batch boundary
    # splits its two entries. Repeated windows retain their original seed.
    cpu_keys = set()
    while True:
        added = {pair_key(t) for group in batches
                 if 0 < sum(pair_key(t) not in cpu_keys for t in group) <= cpu_threshold
                 for t in group if pair_key(t) not in cpu_keys}
        if not added:
            break
        cpu_keys.update(added)
    trajectories = [t for group in batches for t in group]
    cpu = [t for t in trajectories if pair_key(t) in cpu_keys]
    gpu = [[t for t in group if pair_key(t) not in cpu_keys] for group in batches]
    losses = {id(t[0]): np.zeros(2) for t in trajectories}
    counts = {id(t[0]): {'cpu_numpy': 0, 'cuda_philox': 0} for t in trajectories}
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        futures = [pool.submit(cpu_loss, t, corpus, particles) for t in cpu]
        for group in gpu:
            if group:
                for t, value in zip(group, cuda_losses(group, corpus, particles)):
                    losses[id(t[0])][t[1]] += value
                    counts[id(t[0])]['cuda_philox'] += 1
        for t, future in zip(cpu, futures):
            losses[id(t[0])][t[1]] += future.result()
            counts[id(t[0])]['cpu_numpy'] += 1
    return losses, counts
