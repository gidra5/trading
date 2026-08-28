from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from evaluate_stopped_cyclic_path_density import build_model
from normalized_glu_next_return import optimizer_parameter_groups
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import build_optimizers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one exact joint-prefix training step in a fresh CUDA "
            "process so compiled graph pools cannot leak between candidates."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune-no-cudagraphs"),
        default="default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    torch.set_float32_matmul_precision("high")
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    run_root = (repo / plan["runDir"]).resolve()
    checkpoint = load_torch_checkpoint(
        run_root / "checkpoints/last.json",
        map_location="cpu",
        weights_only=False,
    )
    density = KnotDensityContract.load(
        (repo / plan["density"]["source"]).resolve(),
        fit=str(int(plan["architecture"]["outputKnots"])),
    )
    device = torch.device("cuda")
    model = build_model(plan, checkpoint["model"], density, device)
    model.train()
    optimizer_parameter_groups(model)
    optimizers = build_optimizers(model, plan["training"], device)
    for optimizer, state in zip(
        optimizers, checkpoint["optimizers"], strict=True
    ):
        optimizer.load_state_dict(state)
    execution_model = model
    if args.compile_mode != "none":
        compile_cache = (
            repo / "data/training/cache/torchinductor"
        ).resolve()
        compile_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
        os.environ["TRITON_CACHE_DIR"] = str(compile_cache / "triton")
        model.compile_shared_recurrent_step(
            mode=args.compile_mode,
            dynamic=False,
        )
    generator = torch.Generator(device=device).manual_seed(101)
    features = model.feature_mean[None, :] + torch.randn(
        args.batch_size,
        model.feature_mean.numel(),
        generator=generator,
        device=device,
    ) * model.feature_std[None, :]
    targets = torch.randn(
        args.batch_size,
        model.return_count,
        generator=generator,
        device=device,
    ) * 5e-5

    def step() -> float:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        output = execution_model(features, targets)
        terms = output.joint_log_density_terms
        if terms is None or terms.shape != targets.shape:
            raise RuntimeError("joint-prefix density terms were not produced")
        loss = -terms.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(plan["training"]["gradientClip"]),
            foreach=True,
        )
        for optimizer in optimizers:
            optimizer.step()
        return float(loss.detach())

    try:
        warmup_started = time.monotonic()
        warmup_loss = step()
        torch.cuda.synchronize()
        warmup_seconds = time.monotonic() - warmup_started
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        measured_loss = step()
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result = {
            "status": "ok",
            "batchSize": args.batch_size,
            "compileMode": args.compile_mode,
            "cudaGraphsEnabled": False,
            "warmupLoss": warmup_loss,
            "measuredLoss": measured_loss,
            "warmupSeconds": warmup_seconds,
            "secondsPerStep": seconds,
            "examplesPerSecond": args.batch_size / seconds,
            "allocatedGiB": torch.cuda.memory_allocated(device) / 2**30,
            "reservedGiB": torch.cuda.memory_reserved(device) / 2**30,
            "peakAllocatedGiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "peakReservedGiB": torch.cuda.max_memory_reserved(device) / 2**30,
            "driverUsedGiB": (total_bytes - free_bytes) / 2**30,
            "totalGiB": total_bytes / 2**30,
        }
    except torch.OutOfMemoryError as error:
        result = {
            "status": "oom",
            "batchSize": args.batch_size,
            "compileMode": args.compile_mode,
            "cudaGraphsEnabled": False,
            "error": str(error).splitlines()[0],
        }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
