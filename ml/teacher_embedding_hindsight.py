"""Frozen, causal component-center embeddings for one-pass hindsight refinement.

The teacher never receives targets when constructing endpoint forecasts. Only
the explicitly target-assisted training/reconstruction branch encodes targets.
There is no embedding-to-feature decoder between teacher and student.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from trading_storage import load_torch_checkpoint


def blend_embeddings(clean: Tensor | None, prior: Tensor, fraction: float) -> Tensor:
    if not 0 <= fraction <= 1:
        raise ValueError("teacher replacement fraction must lie in [0,1]")
    if fraction == 1:
        return prior.detach()  # Do not access even a NaN target at the endpoint.
    if clean is None or clean.shape != prior.shape:
        raise ValueError("target-assisted embedding shape mismatch")
    if fraction == 0:
        return clean.detach()
    return torch.lerp(clean.detach(), prior.detach(), fraction)


def select_components(features: Tensor, log_weights: Tensor, uniforms: Tensor) -> Tensor:
    """Inverse CDF sampling with supplied batch-partition-invariant uniforms."""
    if features.shape[:-1] != log_weights.shape or uniforms.shape[:2] != log_weights.shape[:2]:
        raise ValueError("component sampling shapes do not agree")
    cdf = log_weights.exp().cumsum(-1)
    cdf = cdf / cdf[..., -1:]
    indices = torch.searchsorted(cdf.contiguous(), uniforms.contiguous(), right=True)
    indices = indices.clamp_max(features.shape[-2] - 1)
    return torch.gather(features, 2, indices[..., None].expand(*indices.shape, features.shape[-1]))


@dataclass
class TeacherEmbeddingBatch:
    inputs: Tensor
    targets: Tensor
    weights: Tensor
    teacher_context: dict[str, Tensor]

    def __iter__(self):
        yield self.inputs
        yield self.targets
        yield self.weights


class TeacherEmbeddingDataset:
    """Same direct-feature population, plus origin-aligned causal teacher context."""

    def __init__(self, source, config: dict, repo: Path):
        from structured_union530_base import StructuredProduction59BaseDataset

        self.source = source
        self.config = dict(config)
        self.repo = repo
        self._teacher = None
        self._device = None
        self.teacher_plan = json.loads((repo / config["plan"]).read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(self.teacher_plan, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        if digest != config["planSha256"]:
            raise ValueError("frozen teacher plan hash changed")
        a = self.teacher_plan["architecture"]
        if a["inputSteps"] != source.input_steps or a["outputSteps"] != source.output_steps \
                or a["featureWidth"] != config["embeddingWidth"]:
            raise ValueError("teacher/student horizon or embedding width mismatch")
        if a["featureEmbeddingDensity"]["type"] != "causal-base-support-gaussian-mixture-density-v2":
            raise ValueError("teacher has no causal component support")
        self.teacher_dataset = StructuredProduction59BaseDataset(
            (repo / self.teacher_plan["datasetDir"]).resolve(),
            (repo / self.teacher_plan["baseHistoryDir"]).resolve(),
            train_examples=source.counts["train"], input_steps=source.input_steps,
            output_steps=source.output_steps,
        )
        if self.teacher_dataset.root != source.root:
            raise ValueError("teacher and student must use the same feature timeline")
        for split in source.counts:
            if not np.array_equal(source.origins[split], self.teacher_dataset.origins[split]):
                raise ValueError(f"teacher/student {split} origins differ")

    def __getattr__(self, name):
        return getattr(self.source, name)

    def statistics(self, batch_size):
        return self.source.statistics(batch_size)

    def iter_batches(self, split, batch_size, *, shuffle, seed, limit=None, pad=True):
        count = self.counts[split] if limit is None else min(self.counts[split], limit)
        indices = np.arange(count, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, count, batch_size):
            selected = indices[start:start + batch_size]
            x, y = self.source._examples(split, selected)
            seconds = self.teacher_dataset.second_offsets[split] + self.origins[split][selected]
            # Construct only causal history, not _examples()' future baseTargets.
            context = self.teacher_dataset._context_seconds(seconds)
            if "baseTargets" in context:
                raise AssertionError("future targets entered teacher context")
            yield TeacherEmbeddingBatch(torch.from_numpy(x), torch.from_numpy(y),
                torch.ones(len(selected), dtype=torch.float32), context)

    def teacher(self, device):
        from evaluate_stopped_structured_feature_process import model_from_state

        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._teacher is None:
            pointer = self.repo / self.config["checkpoint"]
            metadata = json.loads(pointer.read_text(encoding="utf-8"))
            if metadata["object"]["contentHash"] != self.config["checkpointSha256"]:
                raise ValueError("frozen teacher checkpoint hash changed")
            checkpoint = load_torch_checkpoint(pointer, map_location="cpu", weights_only=False)
            # Teacher construction cannot perturb the student's initialization/RNG.
            with torch.random.fork_rng(devices=[]):
                teacher = model_from_state(self.teacher_plan, checkpoint["model"], device)
            teacher.requires_grad_(False).eval()
            self._teacher, self._device = teacher, device
        if device != self._device:
            raise ValueError("frozen teacher device changed during this dataset lifetime")
        return self._teacher

    @torch.no_grad()
    def hindsight(self, inputs: Tensor, context: dict[str, Tensor], *,
                  targets: Tensor | None, fraction: float, samples: int,
                  rng: np.random.Generator) -> Tensor:
        from train_structured_feature_process import derive_base_support_feature_states

        if "baseTargets" in context:
            raise ValueError("teacher endpoint context contains future targets")
        if not 0 <= fraction <= 1:
            raise ValueError("teacher replacement fraction must lie in [0,1]")
        teacher = self.teacher(inputs.device)
        batch = inputs.shape[0]
        steps = self.output_steps
        width = self.config["embeddingWidth"]
        result = torch.empty((batch, steps, samples, width), device=inputs.device)
        # Generate one flat stream before microbatching: batch size does not
        # change sample identities. The endpoint is NOT the Gaussian GMM jitter.
        uniforms = rng.random((batch, steps, samples), dtype=np.float32)
        micro = self.config["inferenceBatchSize"]

        def encode(raw):
            shape = raw.shape[:-1]
            normalized = (raw.reshape(-1, self.feature_count).float() - teacher.input_mean) / teacher.input_std
            embedded = teacher.layer1(normalized)
            embedded = F.normalize(embedded, dim=-1,
                eps=teacher.embedding_density_attention.normalization_epsilon) * math.sqrt(width)
            return embedded.reshape(*shape, width)

        for start in range(0, batch, micro):
            stop = min(start + micro, batch)
            clean = None
            if fraction < 1:
                if targets is None:
                    raise ValueError("target-assisted curriculum requires targets")
                clean = encode(targets[start:stop]).unsqueeze(2).expand(-1, -1, samples, -1)
            if fraction == 0:
                result[start:stop].copy_(clean)
                continue
            chunk_inputs = inputs[start:stop]
            raw, log_weights = teacher.feature_embedding_base_distribution(chunk_inputs)
            causal_context = {k: v[start:stop].to(inputs.device) for k, v in context.items()}
            # Derive complete component paths BEFORE selecting per-step atoms,
            # preserving exactly the component centers used by teacher NLL.
            features = derive_base_support_feature_states(
                self.teacher_dataset, raw, chunk_inputs, causal_context)
            selected = select_components(features, log_weights,
                torch.from_numpy(uniforms[start:stop]).to(inputs.device))
            prior = encode(selected)
            result[start:stop].copy_(blend_embeddings(clean, prior, fraction))
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite frozen teacher embeddings")
        return result

    def close(self):
        self._teacher = None
        self.teacher_dataset.close()
        self.source.close()
