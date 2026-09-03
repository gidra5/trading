"""Read-only teacher inference checks, persisted separately from student metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from train_normalized_glu_next_return import atomic_json
from train_structured_feature_process import build_dataset, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads((repo / args.plan).read_text(encoding="utf-8"))
    validate_plan(plan)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    dataset = build_dataset(plan, repo)
    try:
        print(json.dumps({"alignedCounts": dataset.counts}), flush=True)
        batch = next(dataset.iter_batches("validation", args.batch_size,
                     shuffle=False, seed=0, pad=False))
        x, y, _ = (v.to(device) for v in batch)
        samples = plan["architecture"]["hindsightConditioning"]["sampleCount"]
        common = dict(fraction=1, samples=samples)
        prior = dataset.hindsight(x, batch.teacher_context, targets=None,
                                  rng=np.random.default_rng(711), **common)
        poisoned = dataset.hindsight(x, batch.teacher_context,
            targets=torch.full_like(y, float("nan")), rng=np.random.default_rng(711), **common)
        torch.testing.assert_close(prior, poisoned, rtol=0, atol=0)
        clean = dataset.hindsight(x, batch.teacher_context, targets=y, fraction=0,
                                  samples=samples, rng=np.random.default_rng(711))
        teacher = dataset.teacher(device)
        width = teacher.feature_width
        with torch.no_grad():
            encoded = teacher.layer1(((y - teacher.input_mean) / teacher.input_std).reshape(-1, 59))
            encoded = F.normalize(encoded, dim=-1,
                eps=teacher.embedding_density_attention.normalization_epsilon) * width**.5
            expected = encoded.reshape(len(x), dataset.output_steps, width).unsqueeze(2).expand_as(clean)
        torch.testing.assert_close(clean, expected, rtol=1e-4, atol=1e-4)
        assert not prior.requires_grad and not clean.requires_grad
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        artifact = {
            "contract": "teacher-embedding-hindsight-input-audit-v1",
            "planId": plan["id"], "counts": dataset.counts,
            "teacherCheckpointSha256": plan["hindsightTeacher"]["checkpointSha256"],
            "teacherParameters": sum(p.numel() for p in teacher.parameters()),
            "embeddingShape": list(prior.shape), "endpointIndependentOfTargets": True,
            "cleanMatchesFrozenEncoder": True, "teacherFrozen": True,
            "sampleSource": "discrete-component-centers-no-gaussian-jitter",
            "seconds": time.monotonic() - started,
        }
        atomic_json(artifact, repo / plan["runDir"] / "state/teacher-input-audit.json")
        print(json.dumps(artifact), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
