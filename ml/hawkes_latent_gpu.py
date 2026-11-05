"""Batched float64 CUDA filtering with the adapted discrete hidden-event law.

Independent models run in parallel, with one CUDA block per model. Time remains
sequential. Short kernel chunks retain private state between launches. Addressed
Philox draws make chunk boundaries and batching independent of random results.
"""
from __future__ import annotations
from functools import cache
from pathlib import Path
import numpy as np
import torch
from hawkes_cuda_runtime import CudaKernel
from hawkes_latent import LOG_FACTORIAL, physical_arrays


@cache
def cuda_filter_kernel():
    return CudaKernel(Path(__file__).with_name('hawkes_latent_cuda.cu').read_text(), 'filter_chunk')


def filter_batch(models, y, valid, observed_mean, particles, seeds, *,
                 initial_fast=None, initial_slow=None, chunk_seconds=512, segments=None, loss_weights=None):
    """Return six read-only CPU array views with a leading model batch axis.

    Forecast snapshots remain in the independent CPU implementation. CUDA
    uses float64 and a different random generator; it does not promise bitwise
    equality to CPU Monte Carlo. Degenerate exact cases and independent
    integration/recovery checks establish numerical agreement.
    With loss_weights, return only the weighted NLL sum per model. The compact
    optimizer output avoids allocating a full time series for every proposal.
    """
    if not models or particles < 2 or particles > 512 or chunk_seconds < 1:
        raise ValueError('Need a nonempty batch, 2–512 particles and positive chunks')
    if any(len(m.diffusions) > 4 for m in models):
        raise ValueError('CUDA filtering supports at most four OU components')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable')
    y, valid = np.asarray(y, dtype=np.float64), np.asarray(valid, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 2 or len(valid) != len(y) or len(seeds) != len(models):
        raise ValueError('Counts, mask and model seed batch have incompatible shapes')
    b = len(models)
    segments = np.tile((0, len(y)), (b, 1)).astype(np.int32) if segments is None else np.asarray(segments, dtype=np.int32)
    if segments.shape != (b, 2) or np.any(segments[:, 0] < 0) or np.any(segments[:, 1] > len(y)) or np.any(segments[:, 1] < segments[:, 0]):
        raise ValueError('Filter segments must be ordered covered input ranges per model')
    n = int(np.max(segments[:, 1] - segments[:, 0]))
    if loss_weights is not None:
        loss_weights = np.asarray(loss_weights, dtype=np.int8)
        if loss_weights.shape != (b, n) or np.any(loss_weights < 0):
            raise ValueError('Nonnegative integer loss weights must match local segment padding')
    k = max(1, max(len(m.modes) for m in models))
    a = max(1, max(len(m.anchors) for m in models))
    q = max(1, max(len(m.diffusions) for m in models))
    bg, initial = np.zeros((b, 3)), np.zeros((b, 3))
    target, source = np.full((b, k), 3, np.int32), np.zeros((b, k), np.int32)
    mass, decay = np.zeros((b, k)), np.zeros((b, k))
    anchor, adecay = np.zeros((b, a), np.int32), np.zeros((b, a))
    beta, zdecay, loading = np.zeros((b, k, a)), np.zeros((b, q)), np.zeros((b, q, 3))
    first_fast, first_slow, saturation = np.zeros((b, k)), np.zeros((b, a)), np.zeros(b, np.int32)
    counts = np.zeros((b, 3), np.int32)
    for i, model in enumerate(models):
        arrays = physical_arrays(model, observed_mean)
        mi, ai, qi = len(model.modes), len(model.anchors), len(model.diffusions)
        counts[i] = mi, ai, qi
        bg[i], initial[i], saturation[i] = arrays[0], arrays[8], arrays[9]
        target[i, :mi], source[i, :mi], mass[i, :mi], decay[i, :mi] = arrays[1:5]
        anchor[i, :ai], adecay[i, :ai], beta[i, :mi, :ai] = arrays[5:8]
        first_fast[i, :mi] = arrays[8][arrays[2]] if initial_fast is None else initial_fast[i]
        first_slow[i, :ai] = arrays[8][arrays[5]] if initial_slow is None else initial_slow[i]
        if qi:
            zdecay[i, :qi] = np.exp(-1. / np.array([z.tau for z in model.diffusions]))
            loading[i, :qi] = [z.loading for z in model.diffusions]
    def tensor(value):
        return torch.tensor(value, device='cuda').contiguous()
    inputs = [tensor(v) for v in (y, valid, bg, target, source, mass, decay,
                                  anchor, adecay, beta, initial, saturation,
                                  zdecay, loading, np.sum(loading ** 2, axis=1),
                                  np.asarray(seeds, dtype=np.int64), counts, segments)]
    shapes = ((b, particles, k), (b, particles, k), (b, particles, a),
              (b, particles, a), (b, particles), (b, particles, q))
    states = [torch.zeros(shape, dtype=torch.float64, device='cuda') for shape in shapes]
    following = [torch.empty_like(state) for state in states]
    output = torch.zeros((b,) if loss_weights is not None else (b, n, 7), dtype=torch.float64, device='cuda')
    failed = torch.full((b,), n, dtype=torch.int64, device='cuda')
    temporary = torch.empty((b, particles, 10), dtype=torch.float64, device='cuda')
    tensors = [*inputs, *states, *following, tensor(first_fast), tensor(first_slow),
               tensor(LOG_FACTORIAL), temporary, output, failed, tensor(np.zeros(1, np.int8) if loss_weights is None else loss_weights)]
    kernel = cuda_filter_kernel()
    for start in range(0, n, chunk_seconds):
        kernel(tensors, [n, particles, k, a, q, start, min(n, start + chunk_seconds), int(loss_weights is not None)],
               b, max(32, 1 << (particles - 1).bit_length()))
    values, failure_steps = output.cpu().numpy(), failed.cpu().numpy()
    if loss_weights is not None:
        for i, step in enumerate(failure_steps):
            if step < n and np.any(loss_weights[i, step:]):
                values[i] = np.inf
        values.setflags(write=False)
        return values
    for i, step in enumerate(failure_steps):
        if step < n:
            values[i, step:, :] = 0.
            values[i, step:, :2] = np.nan
            values[i, step:, 2] = np.inf
    values.setflags(write=False)
    return (values[:, :, :2], *(values[:, :, i] for i in range(2, 7)))
