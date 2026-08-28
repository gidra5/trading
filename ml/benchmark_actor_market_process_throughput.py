from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch

from actor_market_process_density import ActorMarketProcessDensity
from compressed_path_return_density import path_log_density_terms
from normalized_glu_next_return import optimizer_parameter_groups
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import build_optimizers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full actor-market forward/backward/optimizer steps."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune-no-cudagraphs"),
        default="none",
    )
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"),
        default="high",
    )
    return parser.parse_args()


def build_model(
    plan: dict, state: dict[str, torch.Tensor], repo: Path, device: torch.device
) -> ActorMarketProcessDensity:
    architecture = plan["architecture"]
    density = KnotDensityContract.load(
        (repo / plan["density"]["source"]).resolve(),
        fit=str(int(architecture["outputKnots"])),
    )
    model = ActorMarketProcessDensity(
        state["feature_mean"],
        state["feature_std"],
        density,
        embedding_width=int(architecture["embeddingWidth"]),
        actor_width=int(architecture["actorWidth"]),
        actor_decision_width=int(architecture["actorDecisionWidth"]),
        market_width=int(architecture["marketWidth"]),
        actor_count=int(architecture["actorCount"]),
        market_count=int(architecture["marketCount"]),
        action_count=int(architecture["actionCount"]),
        action_basis_width=int(architecture["actionBasisWidth"]),
        reward_width=int(architecture["rewardWidth"]),
        return_count=int(architecture["returnCount"]),
        certainty_maximum=float(architecture["certaintyMaximum"]),
        quadrature_order=int(architecture.get("quadratureOrder", 16)),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture["learnableCentering"]),
    ).to(device)
    model.load_state_dict(state)
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.steps < 1 or args.warmup_steps < 1:
        raise ValueError("benchmark step counts must be positive")
    torch.set_float32_matmul_precision(args.matmul_precision)
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    run_root = (repo / plan["runDir"]).resolve()
    compile_cache = (run_root / "state/torchinductor-cache").resolve()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
    os.environ["TRITON_CACHE_DIR"] = str(compile_cache / "triton")
    checkpoint = load_torch_checkpoint(
        run_root / "checkpoints/last.json", map_location="cpu", weights_only=False
    )
    device = torch.device("cuda")
    results = []

    for batch_size in args.batch_sizes:
        model = None
        optimizers = None
        compiled = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_model(plan, checkpoint["model"], repo, device)
            optimizer_parameter_groups(model)
            optimizers = build_optimizers(model, plan["training"], device)
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            if args.compile_mode == "none":
                compiled = model
            else:
                compile_arguments = {"fullgraph": False, "dynamic": False}
                if args.compile_mode == "default":
                    compile_arguments["options"] = {
                        "triton.cudagraphs": False
                    }
                else:
                    compile_arguments["mode"] = args.compile_mode
                compiled = torch.compile(model, **compile_arguments)
            features = (
                model.feature_mean[None, :]
                + torch.randn(
                    batch_size, model.feature_mean.numel(), device=device
                ) * model.feature_std[None, :]
            )
            targets = torch.randn(
                batch_size, model.return_count, device=device
            ) * 5e-5
            weights = torch.ones(batch_size, device=device)

            def step() -> None:
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                output = compiled(features)
                terms = path_log_density_terms(output, targets, model)
                loss = -(terms * weights[:, None]).sum() / (
                    weights.sum() * model.return_count
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(plan["training"]["gradientClip"]),
                    foreach=True,
                )
                for optimizer in optimizers:
                    optimizer.step()

            compile_started = time.monotonic()
            for _ in range(args.warmup_steps):
                step()
            torch.cuda.synchronize()
            warmup_seconds = time.monotonic() - compile_started
            torch.cuda.reset_peak_memory_stats()
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            for _ in range(args.steps):
                step()
            ended.record()
            ended.synchronize()
            seconds = started.elapsed_time(ended) / 1_000
            result = {
                "batchSize": batch_size,
                "status": "ok",
                "compileMode": args.compile_mode,
                "warmupSeconds": warmup_seconds,
                "secondsPerStep": seconds / args.steps,
                "examplesPerSecond": batch_size * args.steps / seconds,
                "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 2**30,
                "peakReservedGiB": torch.cuda.max_memory_reserved() / 2**30,
            }
        except torch.OutOfMemoryError as error:
            result = {
                "batchSize": batch_size,
                "status": "oom",
                "compileMode": args.compile_mode,
                "error": str(error).splitlines()[0],
            }
        results.append(result)
        print(json.dumps(result), flush=True)
        del compiled, optimizers, model
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
