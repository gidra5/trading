from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path("data/benchmarks/one-second-return-64-knot-fits.json")
DEFAULT_OUTPUT = Path("data/benchmarks/one-second-return-knot-scaling-v1.json")
KNOT_COUNTS = (8, 16, 32, 64, 128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a controlled knot-count family from the global KL fit."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def basis_areas(knots: np.ndarray) -> np.ndarray:
    gaps = np.diff(knots)
    areas = np.empty(knots.size, dtype=np.float64)
    areas[0] = gaps[0] / 2
    areas[-1] = gaps[-1] / 2
    areas[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return areas


def interval_masses(knots: np.ndarray, heights: np.ndarray) -> np.ndarray:
    return np.diff(knots) * (heights[:-1] + heights[1:]) / 2


def inverse_piecewise_linear_cdf(
    probabilities: np.ndarray,
    knots: np.ndarray,
    heights: np.ndarray,
) -> np.ndarray:
    masses = interval_masses(knots, heights)
    cumulative = np.concatenate(([0.0], np.cumsum(masses)))
    cumulative[-1] = 1.0
    result = np.empty_like(probabilities, dtype=np.float64)
    result[probabilities <= 0] = 0
    result[probabilities >= 1] = 1
    active = (probabilities > 0) & (probabilities < 1)
    selected = probabilities[active]
    interval = np.searchsorted(cumulative[1:-1], selected, side="right")
    local_mass = selected - cumulative[interval]
    left = knots[interval]
    width = knots[interval + 1] - left
    left_height = heights[interval]
    slope = (heights[interval + 1] - left_height) / width
    discriminant = np.maximum(
        left_height * left_height + 2 * slope * local_mass, 0
    )
    denominator = left_height + np.sqrt(discriminant)
    distance = np.where(
        np.abs(slope) < 1e-14,
        local_mass / left_height,
        2 * local_mass / denominator,
    )
    result[active] = left + np.clip(distance, 0, width)
    return result


def refined_knots(
    count: int,
    source_knots: np.ndarray,
    source_heights: np.ndarray,
) -> np.ndarray:
    if count < source_knots.size:
        return inverse_piecewise_linear_cdf(
            np.linspace(0, 1, count), source_knots, source_heights
        )
    if count == source_knots.size:
        return source_knots.copy()
    masses = interval_masses(source_knots, source_heights)
    extra = count - source_knots.size
    desired = extra * masses / masses.sum()
    allocations = np.floor(desired).astype(np.int64)
    remainder = extra - int(allocations.sum())
    if remainder:
        order = np.argsort(-(desired - allocations), kind="stable")
        allocations[order[:remainder]] += 1
    cumulative = np.concatenate(([0.0], np.cumsum(masses)))
    values: list[float] = [0.0]
    for index, allocation in enumerate(allocations):
        segments = int(allocation) + 1
        if segments > 1:
            local_probabilities = cumulative[index] + (
                masses[index] * np.arange(1, segments) / segments
            )
            values.extend(inverse_piecewise_linear_cdf(
                local_probabilities, source_knots, source_heights
            ).tolist())
        values.append(float(source_knots[index + 1]))
    result = np.asarray(values, dtype=np.float64)
    if result.size != count or bool((np.diff(result) <= 0).any()):
        raise RuntimeError("controlled knot refinement produced invalid knots")
    return result


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_file = (repo / args.source).resolve()
    output_file = (repo / args.output).resolve()
    source = json.loads(source_file.read_text(encoding="utf-8"))
    kl = source["fits"]["kl"]
    source_knots = np.asarray(kl["knotsUnit"], dtype=np.float64)
    source_knots[0] = 0
    source_knots[-1] = 1
    source_heights = np.asarray(kl["knotDensityHeights"], dtype=np.float64)
    fits: dict[str, dict] = {}
    for count in KNOT_COUNTS:
        knots = refined_knots(count, source_knots, source_heights)
        heights = np.interp(knots, source_knots, source_heights)
        masses = basis_areas(knots) * heights
        masses /= masses.sum()
        heights = masses / basis_areas(knots)
        fits[str(count)] = {
            "knotsUnit": knots.tolist(),
            "componentWeights": masses.tolist(),
            "knotDensityHeights": heights.tolist(),
            "derivation": (
                "global-CDF quantiles with source-density interpolation"
                if count < source_knots.size else (
                    "unchanged source KL fit"
                    if count == source_knots.size else
                    "mass-proportional exact subdivision of source KL intervals"
                )
            ),
        }
    result = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "source": str(args.source).replace("\\", "/"),
        "sourceFit": "kl",
        "transform": source["transform"],
        "target": source["target"],
        "reconstruction": {
            "family": "continuous piecewise-linear density",
            "counts": list(KNOT_COUNTS),
            "control": (
                "all counts approximate the same global KL density; counts at "
                "or above 64 preserve every original breakpoint"
            ),
        },
        "fits": fits,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
