from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from causal_return_normalization import (
    trailing_log_price_statistics,
    trailing_log_return_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a triangular prior for causally volatility-normalized returns."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=256_000)
    parser.add_argument("--window-seconds", type=int, default=7_200)
    parser.add_argument("--variance-floor", type=float, default=1e-16)
    parser.add_argument(
        "--statistic", choices=("log-return", "log-price"),
        default="log-return",
    )
    parser.add_argument("--knots", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--em-iterations", type=int, default=100)
    return parser.parse_args()


def triangular_areas(knots: np.ndarray) -> np.ndarray:
    gaps = np.diff(knots)
    areas = np.empty_like(knots)
    areas[0] = gaps[0] / 2
    areas[-1] = gaps[-1] / 2
    areas[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return areas


def fit_component_masses(
    unit: np.ndarray, knots: np.ndarray, iterations: int
) -> np.ndarray:
    areas = triangular_areas(knots)
    interval = np.searchsorted(knots[1:-1], unit, side="right")
    left = knots[interval]
    right = knots[interval + 1]
    fraction = np.clip((unit - left) / (right - left), 0, 1)
    left_basis = (1 - fraction) / areas[interval]
    right_basis = fraction / areas[interval + 1]
    masses = np.full(knots.size, 1 / knots.size, dtype=np.float64)
    for _ in range(iterations):
        left_value = masses[interval] * left_basis
        right_value = masses[interval + 1] * right_basis
        total = np.maximum(left_value + right_value, np.finfo(np.float64).tiny)
        updated = np.bincount(
            interval, weights=left_value / total, minlength=knots.size
        ) + np.bincount(
            interval + 1, weights=right_value / total, minlength=knots.size
        )
        masses = np.maximum(updated / unit.size, 1e-10)
        masses /= masses.sum()
    return masses


def main() -> None:
    args = parse_args()
    if args.examples < args.knots or args.knots < 8:
        raise ValueError("the fit requires at least eight knots and enough examples")
    if args.alpha <= 1 or args.em_iterations < 1:
        raise ValueError("alpha and EM iteration count are invalid")
    repo = Path(__file__).resolve().parents[1]
    dataset = args.dataset_dir if args.dataset_dir.is_absolute() \
        else repo / args.dataset_dir
    history = args.history_dir if args.history_dir.is_absolute() \
        else repo / args.history_dir
    output = args.output if args.output.is_absolute() else repo / args.output
    manifest = json.loads((dataset / "manifest.json").read_text("utf-8"))
    available = int(manifest["examplesBySplit"]["train"])
    count = min(int(args.examples), available)
    targets = np.asarray(np.memmap(
        dataset / "train.targets.f32", dtype="<f4", mode="r",
        shape=(available,),
    )[:count], dtype=np.float64)
    times = np.asarray(np.memmap(
        dataset / "train.times.f64", dtype="<f8", mode="r",
        shape=(available,),
    )[:count], dtype=np.float64)
    statistics_function = (
        trailing_log_price_statistics
        if args.statistic == "log-price"
        else trailing_log_return_statistics
    )
    means, variances = statistics_function(
        history,
        times,
        window_seconds=int(args.window_seconds),
        variance_floor=float(args.variance_floor),
    )
    scale = np.sqrt(variances.astype(np.float64))
    normalized = (
        targets / scale
        if args.statistic == "log-price"
        else (targets - means.astype(np.float64)) / scale
    )
    latent = float(args.alpha) * np.arcsinh(normalized)
    unit = 1 / (1 + np.exp(-latent))
    epsilon = np.finfo(np.float32).eps
    unit = np.clip(unit, epsilon, 1 - epsilon)
    probabilities = np.linspace(0, 1, int(args.knots))
    knots = np.empty(int(args.knots), dtype=np.float64)
    knots[0] = 0
    knots[-1] = 1
    knots[1:-1] = np.quantile(unit, probabilities[1:-1])
    if np.any(np.diff(knots) <= 0):
        raise ValueError("empirical normalized-return knots are not distinct")
    masses = fit_component_masses(unit, knots, int(args.em_iterations))
    areas = triangular_areas(knots)

    value = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(dataset.relative_to(repo)),
        "transform": {
            "family": "asinh-logistic",
            "trainable": False,
            "alpha": float(args.alpha),
            "locationBps": 0.0,
            "scaleBps": 10_000.0,
            "forward": "u=sigmoid(alpha*asinh(z))",
            "inverse": "z=sinh(logit(u)/alpha)",
        },
        "target": {
            "returnDefinition": (
                "z=next active 1s log return / sqrt(causal trailing "
                "log-price variance)"
                if args.statistic == "log-price" else
                "z=(next active 1s log return - causal trailing return mean) / "
                "sqrt(causal trailing return variance)"
            ),
            "windowSeconds": int(args.window_seconds),
            "windowPopulation": (
                "all completed one-second log-price levels"
                if args.statistic == "log-price" else
                "all completed 1s returns including exact zeros"
            ),
            "windowStatistic": args.statistic,
            "causal": True,
            "varianceFloor": float(args.variance_floor),
            "observations": int(count),
            "normalizedMean": float(normalized.mean()),
            "normalizedVariance": float(normalized.var()),
        },
        "fits": {
            str(int(args.knots)): {
                "knotsUnit": knots.tolist(),
                "componentWeights": masses.tolist(),
                "knotDensityHeights": (masses / areas).tolist(),
                "derivation": (
                    "empirical unit-coordinate quantiles followed by triangular-"
                    "mixture maximum-likelihood EM"
                ),
            }
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "examples": count,
        "normalizedMean": value["target"]["normalizedMean"],
        "normalizedVariance": value["target"]["normalizedVariance"],
        "minimumMass": float(masses.min()),
        "maximumMass": float(masses.max()),
    }))


if __name__ == "__main__":
    main()
