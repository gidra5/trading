from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from return_knot_density import RETURN_TO_BPS
from trading_storage import read_candle_column


DEFAULT_HISTORY = Path(
    "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"
)
DEFAULT_DISTRIBUTIONS = Path("data/benchmarks/log-return-distributions.json")
DEFAULT_FITS = Path("data/benchmarks/log-return-distribution-fits.json")
DEFAULT_OUTPUT = Path("data/benchmarks/one-minute-return-knots-k32-v1.json")
KNOT_COUNT = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit fixed global 32-knot density coordinates to 1m returns."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--distributions", type=Path, default=DEFAULT_DISTRIBUTIONS
    )
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def basis_areas(knots: np.ndarray) -> np.ndarray:
    gaps = np.diff(knots)
    areas = np.empty_like(knots)
    areas[0] = gaps[0] / 2
    areas[-1] = gaps[-1] / 2
    areas[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return areas


def fit_component_masses(
    unit: np.ndarray,
    knots: np.ndarray,
    *,
    iterations: int = 64,
) -> np.ndarray:
    """Maximum-likelihood masses for normalized triangular knot bases."""
    areas = basis_areas(knots)
    interval = np.searchsorted(knots[1:-1], unit, side="right")
    fraction = (unit - knots[interval]) / (
        knots[interval + 1] - knots[interval]
    )
    left_basis = (1 - fraction) / areas[interval]
    right_basis = fraction / areas[interval + 1]
    masses = np.full(knots.size, 1 / knots.size, dtype=np.float64)
    for _ in range(iterations):
        left_weight = masses[interval] * left_basis
        right_weight = masses[interval + 1] * right_basis
        denominator = np.maximum(left_weight + right_weight, 1e-300)
        counts = np.bincount(
            interval,
            weights=left_weight / denominator,
            minlength=knots.size,
        )
        counts += np.bincount(
            interval + 1,
            weights=right_weight / denominator,
            minlength=knots.size,
        )
        masses = np.maximum(counts, 1e-12)
        masses /= masses.sum()
    return masses


def load_active_returns(history: Path) -> tuple[np.ndarray, int]:
    files = sorted(history.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no one-minute candle references in {history}")
    chunks: list[np.ndarray] = []
    previous_close: float | None = None
    previous_day: date | None = None
    observations = 0
    for file in files:
        current_day = date.fromisoformat(file.stem)
        closes = read_candle_column(file, "close").astype(np.float64, copy=False)
        if closes.shape != (1_440,) or bool((closes <= 0).any()):
            raise ValueError(f"invalid one-minute close shard: {file}")
        if previous_close is None \
                or previous_day is None \
                or current_day != previous_day + timedelta(days=1):
            returns = np.diff(np.log(closes))
        else:
            returns = np.diff(np.log(np.concatenate((
                np.asarray([previous_close]), closes
            ))))
        observations += int(returns.size)
        active = returns[returns != 0]
        if active.size:
            chunks.append(active)
        previous_close = float(closes[-1])
        previous_day = current_day
    return np.concatenate(chunks), observations


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    history = (repo / args.history).resolve()
    distributions_file = (repo / args.distributions).resolve()
    fits_file = (repo / args.fits).resolve()
    output_file = (repo / args.output).resolve()
    distributions = json.loads(distributions_file.read_text(encoding="utf-8"))
    fits = json.loads(fits_file.read_text(encoding="utf-8"))
    distribution = next(
        value for value in distributions["fullHistory"]["scales"]
        if value["id"] == "1m"
    )
    fitted = next(value for value in fits["scales"] if value["id"] == "1m")
    parameters = fitted["parameters"]
    sigma_bps = float(fitted["sigmaBps"])
    alpha = float(fitted["asymptoticSurvivalExponent"])
    location_bps = sigma_bps * float(parameters["locationSigma"])
    scale_bps = sigma_bps * float(parameters["scaleSigma"])
    returns, observations = load_active_returns(history)
    bps = returns * RETURN_TO_BPS
    unit = 1 / (1 + np.exp(
        -alpha * np.arcsinh((bps - location_bps) / scale_bps)
    ))
    knots = np.quantile(unit, np.linspace(0, 1, KNOT_COUNT))
    knots[0] = 0
    knots[-1] = 1
    if bool((np.diff(knots) <= 0).any()):
        raise RuntimeError("one-minute transformed quantiles are not unique")
    masses = fit_component_masses(unit, knots)
    areas = basis_areas(knots)
    result = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "source": str(args.distributions).replace("\\", "/"),
        "sourceFit": "global-1m-generalized-t-plus-empirical-knot-ml",
        "transform": {
            "family": "asinh-logistic",
            "trainable": False,
            "alpha": alpha,
            "locationBps": location_bps,
            "scaleBps": scale_bps,
            "forward": "u=sigmoid(alpha*asinh((r-mu)/scale))",
            "inverse": "r=mu+scale*sinh(logit(u)/alpha)",
            "scaleSelection": (
                "full-history 1m symmetric generalized-t scale and tail exponent"
            ),
        },
        "target": {
            "returnDefinition": "natural log of adjacent 1m closes, in basis points",
            "zeroTreatment": "exact zeros excluded from the continuous fit",
            "observations": observations,
            "activeObservations": int(returns.size),
            "zeroObservations": observations - int(returns.size),
            "zeroProbability": 1 - returns.size / observations,
            "activeProbability": returns.size / observations,
            "activeStandardDeviationBps": float(np.std(bps)),
        },
        "reconstruction": {
            "family": "continuous piecewise-linear density",
            "counts": [KNOT_COUNT],
            "control": (
                "global empirical transformed quantiles with fixed-basis "
                "maximum-likelihood component masses"
            ),
        },
        "fits": {
            str(KNOT_COUNT): {
                "knotsUnit": knots.tolist(),
                "componentWeights": masses.tolist(),
                "knotDensityHeights": (masses / areas).tolist(),
                "derivation": (
                    "global 1m transformed quantiles and triangular-mixture EM"
                ),
            }
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
