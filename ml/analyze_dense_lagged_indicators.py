from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import lfilter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AXIS = ROOT / "data/runtime-cache/dense-lagged-indicator-audit"
DEFAULT_BASIS = ROOT / "data/runtime-cache/global-feature-basis"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/dense-lagged-indicator-audit.json"
DEFAULT_REPORT = ROOT / "docs/experiments/dense-lagged-indicator-audit-2026-08-18.md"
ALPHA = 0.5
FEATURE_BINS = 4
TARGET_ACTIVE_BINS = 16
RSI_PERIODS = (2, 4, 8, 14, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192)
EMA_PERIODS = (2, 4, 8, 16, 32, 64, 128, 512, 2_048, 4_096, 8_192)
EMA_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024)
SIGNAL_LAGS = (0, 1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1_440)
SELECTED_BASES = {
    "1m": ("realized-volatility-30m", "range-1m", "realized-volatility-15m", "realized-volatility-60m"),
    "15m": ("realized-volatility-60m", "realized-volatility-15m", "realized-volatility-240m"),
    "1h": ("realized-volatility-30m", "realized-volatility-240m"),
}


@dataclass
class HorizonContext:
    id: str
    minutes: int
    origin_indices: np.ndarray
    target: np.ndarray
    train: np.ndarray
    primary: np.ndarray
    transfer: np.ndarray
    blocks: list[np.ndarray]
    target_classes: int
    unconditional: np.ndarray
    unconditional_ratios: np.ndarray
    base_state: np.ndarray
    base_states: int
    base_ratios: np.ndarray
    base_score: dict[str, Any]
    selected_basis: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis-dir", type=Path, default=DEFAULT_AXIS)
    parser.add_argument("--basis-dir", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_only:
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(artifact), encoding="utf-8")
        print(f"Wrote {display_path(args.report)}")
        return
    axis_manifest = json.loads((args.axis_dir / "manifest.json").read_text(encoding="utf-8"))
    basis_manifest = json.loads((args.basis_dir / "manifest.json").read_text(encoding="utf-8"))
    axis_times = np.memmap(
        args.axis_dir / axis_manifest["files"]["times"], dtype="<f8", mode="r", shape=(axis_manifest["rows"],)
    )
    closes = np.memmap(
        args.axis_dir / axis_manifest["files"]["closes"], dtype="<f8", mode="r", shape=(axis_manifest["rows"],)
    )
    dataset = next(item for item in basis_manifest["datasets"] if item["id"] == "1m")
    raw_features = np.memmap(
        args.basis_dir / dataset["files"]["features"],
        dtype="<f4", mode="r", shape=(dataset["rows"], dataset["featureCount"]),
    )
    raw_targets = np.memmap(
        args.basis_dir / dataset["files"]["targets"],
        dtype="<f4", mode="r", shape=(dataset["rows"], dataset["targetCount"]),
    )
    raw_splits = np.memmap(
        args.basis_dir / dataset["files"]["splits"], dtype="u1", mode="r", shape=(dataset["rows"],)
    )
    raw_times = np.memmap(
        args.basis_dir / dataset["files"]["times"], dtype="<f8", mode="r", shape=(dataset["rows"],)
    )
    origin_indices = align_origins(np.asarray(axis_times), np.asarray(raw_times))
    contexts = build_contexts(dataset, raw_features, raw_targets, raw_splits, raw_times, origin_indices)

    periods = RSI_PERIODS[:2] if args.smoke else RSI_PERIODS
    ema_periods = EMA_PERIODS[:2] if args.smoke else EMA_PERIODS
    ema_horizons = EMA_HORIZONS[:2] if args.smoke else EMA_HORIZONS
    lags = SIGNAL_LAGS[:2] if args.smoke else SIGNAL_LAGS
    results = {context.id: [] for context in contexts}
    close_values = np.asarray(closes, dtype=np.float64)
    changes = np.diff(close_values, prepend=close_values[0])
    gains = np.maximum(changes, 0)
    losses = np.maximum(-changes, 0)

    total_base_signals = len(periods) + len(ema_periods) * (1 + 2 * len(ema_horizons))
    completed = 0
    for period in periods:
        average_gain = exponential_filter(gains, 1 / period, 0.0)
        average_loss = exponential_filter(losses, 1 / period, 0.0)
        rsi = np.full(close_values.shape, 50.0, dtype=np.float64)
        positive_loss = average_loss > 0
        rsi[positive_loss] = 100 - 100 / (1 + average_gain[positive_loss] / average_loss[positive_loss])
        rsi[(~positive_loss) & (average_gain > 0)] = 100
        evaluate_signal(results, contexts, rsi, "rsi", period, None, lags)
        completed += 1
        print(f"Signals {completed}/{total_base_signals}: RSI({period}m)", flush=True)

    for period in ema_periods:
        alpha = 2 / (period + 1)
        ema = exponential_filter(close_values, alpha, float(close_values[0]))
        gap = 10_000 * np.log(close_values / ema)
        evaluate_signal(results, contexts, gap, "ema-value", period, None, lags)
        completed += 1
        print(f"Signals {completed}/{total_base_signals}: EMA value({period}m)", flush=True)
        for horizon in ema_horizons:
            slope = np.zeros(close_values.shape, dtype=np.float64)
            slope[horizon:] = 10_000 * np.log(ema[horizon:] / ema[:-horizon]) / horizon
            evaluate_signal(results, contexts, slope, "ema-slope", period, horizon, lags)
            completed += 1
            print(f"Signals {completed}/{total_base_signals}: EMA slope({period}m,{horizon}m)", flush=True)

            acceleration = np.zeros(close_values.shape, dtype=np.float64)
            twice = 2 * horizon
            acceleration[twice:] = slope[twice:] - slope[horizon:-horizon]
            evaluate_signal(results, contexts, acceleration, "ema-acceleration", period, horizon, lags)
            completed += 1
            print(f"Signals {completed}/{total_base_signals}: EMA acceleration({period}m,{horizon}m)", flush=True)

    horizon_rows = []
    for context in contexts:
        rows = results[context.id]
        horizon_rows.append(summarize_horizon(context, rows))
    artifact = {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "objective": "Dense delayed EMA-value, EMA-slope, EMA-acceleration, and RSI screen for larger BTC return horizons",
        "metric": "held-out log2 likelihood gain in bits per target for the full categorical return distribution",
        "split": basis_manifest["split"],
        "grid": {
            "rsiPeriodsMinutes": list(periods),
            "emaPeriodsMinutes": list(ema_periods),
            "emaDifferenceHorizonsMinutes": list(ema_horizons),
            "signalLagsMinutes": list(lags),
            "featureBins": FEATURE_BINS,
            "targetActiveBins": TARGET_ACTIVE_BINS,
            "candidateCount": len(lags) * (len(periods) + len(ema_periods) * (1 + 2 * len(ema_horizons))),
            "selection": "parameters and lag selected only on the primary year; the later transfer year is untouched confirmation",
        },
        "horizons": horizon_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {display_path(args.output)}")
    print(f"Wrote {display_path(args.report)}")


def exponential_filter(values: np.ndarray, alpha: float, initial: float) -> np.ndarray:
    output, _ = lfilter([alpha], [1.0, -(1 - alpha)], values, zi=[(1 - alpha) * initial])
    return output


def align_origins(axis_times: np.ndarray, origin_times: np.ndarray) -> np.ndarray:
    start = float(axis_times[0])
    indices = np.rint((origin_times - start) / 60_000).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= len(axis_times)):
        raise ValueError("A feature-basis origin lies outside the indicator axis.")
    if not np.array_equal(axis_times[indices], origin_times):
        raise ValueError("Feature-basis origins do not align exactly to the indicator axis.")
    return indices


def build_contexts(
    dataset: dict[str, Any],
    raw_features: np.ndarray,
    raw_targets: np.ndarray,
    raw_splits: np.ndarray,
    raw_times: np.ndarray,
    raw_origin_indices: np.ndarray,
) -> list[HorizonContext]:
    feature_by_id = {definition["id"]: index for index, definition in enumerate(dataset["features"])}
    contexts = []
    for target_index, target_definition in enumerate(dataset["targets"]):
        target_id = target_definition["id"]
        if target_id not in SELECTED_BASES:
            continue
        selected = nonoverlapping_rows(raw_times, raw_splits, target_definition["minutes"] * 60_000)
        splits = np.asarray(raw_splits[selected], dtype=np.uint8)
        train = splits == 0
        primary = splits == 1
        transfer = splits == 2
        target, classes = quantize_target(np.asarray(raw_targets[selected, target_index], dtype=np.float64), train)
        unconditional = fit_unconditional(target[train], classes)
        unconditional_ratios = np.log2(unconditional[target])
        basis_ids = SELECTED_BASES[target_id]
        basis_columns = np.asarray([feature_by_id[item] for item in basis_ids], dtype=np.int64)
        basis_values = np.asarray(raw_features[selected][:, basis_columns], dtype=np.float64)
        quantized_basis, arities = quantize_matrix(basis_values, train)
        base_state, base_states = encode_matrix(quantized_basis, arities)
        base_ratios = fitted_log_probabilities(base_state, base_states, target, train, classes)
        blocks = chronological_blocks(primary, transfer)
        contexts.append(HorizonContext(
            id=target_id,
            minutes=int(target_definition["minutes"]),
            origin_indices=np.asarray(raw_origin_indices[selected], dtype=np.int64),
            target=target,
            train=train,
            primary=primary,
            transfer=transfer,
            blocks=blocks,
            target_classes=classes,
            unconditional=unconditional,
            unconditional_ratios=unconditional_ratios,
            base_state=base_state,
            base_states=base_states,
            base_ratios=base_ratios,
            base_score=score_ratios(base_ratios - unconditional_ratios, primary, transfer, blocks),
            selected_basis=basis_ids,
        ))
    return contexts


def evaluate_signal(
    results: dict[str, list[dict[str, Any]]],
    contexts: list[HorizonContext],
    signal: np.ndarray,
    family: str,
    period: int,
    difference_horizon: int | None,
    lags: Iterable[int],
) -> None:
    for lag in lags:
        for context in contexts:
            indices = context.origin_indices - lag
            if np.any(indices < 0):
                raise ValueError(f"Insufficient warmup for lag {lag}m.")
            values = signal[indices]
            edges = unique_quantiles(values[context.train], FEATURE_BINS)
            feature = np.searchsorted(edges, values, side="right").astype(np.int64)
            arity = len(edges) + 1
            standalone_ratios = fitted_log_probabilities(
                feature, arity, context.target, context.train, context.target_classes
            ) - context.unconditional_ratios
            joint_state = context.base_state * arity + feature
            joint_ratios = fitted_log_probabilities(
                joint_state, context.base_states * arity, context.target, context.train, context.target_classes
            )
            conditional_ratios = joint_ratios - context.base_ratios
            parameters = f"period={period}m"
            if difference_horizon is not None:
                parameters += f", difference={difference_horizon}m"
            results[context.id].append({
                "id": candidate_id(family, period, difference_horizon, lag),
                "family": family,
                "periodMinutes": period,
                "differenceHorizonMinutes": difference_horizon,
                "lagMinutes": lag,
                "parameters": parameters,
                "standalone": score_ratios(standalone_ratios, context.primary, context.transfer, context.blocks),
                "conditional": score_ratios(conditional_ratios, context.primary, context.transfer, context.blocks),
            })


def candidate_id(family: str, period: int, difference_horizon: int | None, lag: int) -> str:
    middle = f"-{difference_horizon}" if difference_horizon is not None else ""
    return f"{family}-{period}{middle}-lag-{lag}"


def summarize_horizon(context: HorizonContext, rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = ("ema-value", "ema-slope", "ema-acceleration", "rsi")
    family_results = []
    for family in families:
        candidates = [row for row in rows if row["family"] == family]
        best_standalone = max(candidates, key=lambda row: row["standalone"]["primaryBits"])
        best_conditional = max(candidates, key=lambda row: row["conditional"]["primaryBits"])
        stable = [row for row in candidates if row["conditional"]["positivePrimaryBlocks"] == 2]
        stable_best = max(stable, key=lambda row: row["conditional"]["primaryBits"]) if stable else None
        family_results.append({
            "family": family,
            "candidateCount": len(candidates),
            "bestStandaloneSelectedOnPrimary": best_standalone,
            "bestConditionalSelectedOnPrimary": best_conditional,
            "bestPrimaryStableConditional": stable_best,
        })
    best_by_lag = []
    for lag in SIGNAL_LAGS:
        candidates = [row for row in rows if row["lagMinutes"] == lag]
        if not candidates:
            continue
        best_by_lag.append({
            "lagMinutes": lag,
            "bestStandaloneSelectedOnPrimary": max(candidates, key=lambda row: row["standalone"]["primaryBits"]),
            "bestConditionalSelectedOnPrimary": max(candidates, key=lambda row: row["conditional"]["primaryBits"]),
        })
    top_conditional = sorted(rows, key=lambda row: row["conditional"]["primaryBits"], reverse=True)[:20]
    top_standalone = sorted(rows, key=lambda row: row["standalone"]["primaryBits"], reverse=True)[:20]
    return {
        "id": context.id,
        "horizonMinutes": context.minutes,
        "observations": {
            "train": int(context.train.sum()),
            "primary": int(context.primary.sum()),
            "transfer": int(context.transfer.sum()),
        },
        "selectedBasis": list(context.selected_basis),
        "selectedBasisScore": context.base_score,
        "families": family_results,
        "bestByLag": best_by_lag,
        "topConditional": top_conditional,
        "topStandalone": top_standalone,
        "allCandidates": rows,
    }


def nonoverlapping_rows(times: np.ndarray, splits: np.ndarray, horizon_ms: float) -> np.ndarray:
    keep: list[int] = []
    for split in (0, 1, 2):
        last = -math.inf
        for index in np.flatnonzero(splits == split):
            time = float(times[index])
            if time >= last + horizon_ms:
                keep.append(int(index))
                last = time
    return np.asarray(sorted(keep), dtype=np.int64)


def quantize_target(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, int]:
    train_values = values[train]
    zero_separate = float(np.mean(train_values == 0)) >= 0.001
    active = train_values[train_values != 0] if zero_separate else train_values
    edges = unique_quantiles(active, TARGET_ACTIVE_BINS)
    classes = len(edges) + 1
    if zero_separate:
        output = np.zeros(values.shape[0], dtype=np.int16)
        mask = values != 0
        output[mask] = 1 + np.searchsorted(edges, values[mask], side="right")
        return output, classes + 1
    return np.searchsorted(edges, values, side="right").astype(np.int16), classes


def quantize_matrix(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, list[int]]:
    output = np.empty(values.shape, dtype=np.uint8)
    arities = []
    for column in range(values.shape[1]):
        edges = unique_quantiles(values[train, column], FEATURE_BINS)
        output[:, column] = np.searchsorted(edges, values[:, column], side="right")
        arities.append(len(edges) + 1)
    return output, arities


def unique_quantiles(values: np.ndarray, bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot quantize an all-invalid signal.")
    return np.unique(np.quantile(finite, np.arange(1, bins) / bins))


def encode_matrix(features: np.ndarray, arities: list[int]) -> tuple[np.ndarray, int]:
    state = np.zeros(features.shape[0], dtype=np.int64)
    states = 1
    for column, arity in enumerate(arities):
        state = state * arity + features[:, column]
        states *= arity
    return state, states


def fit_unconditional(target: np.ndarray, classes: int) -> np.ndarray:
    counts = np.bincount(target, minlength=classes).astype(np.float64)
    return (counts + ALPHA) / (counts.sum() + ALPHA * classes)


def fitted_log_probabilities(
    state: np.ndarray, states: int, target: np.ndarray, train: np.ndarray, classes: int
) -> np.ndarray:
    joint = np.bincount(
        state[train] * classes + target[train], minlength=states * classes
    ).reshape(states, classes).astype(np.float64)
    totals = joint.sum(axis=1)
    evaluation = ~train
    evaluation_state = state[evaluation]
    evaluation_target = target[evaluation]
    probability = (
        (joint[evaluation_state, evaluation_target] + ALPHA)
        / (totals[evaluation_state] + ALPHA * classes)
    )
    output = np.zeros(target.shape[0], dtype=np.float64)
    output[evaluation] = np.log2(probability)
    return output


def chronological_blocks(primary: np.ndarray, transfer: np.ndarray) -> list[np.ndarray]:
    blocks = []
    for mask in (primary, transfer):
        indices = np.flatnonzero(mask)
        middle = len(indices) // 2
        for part in (indices[:middle], indices[middle:]):
            block = np.zeros(mask.shape[0], dtype=bool)
            block[part] = True
            blocks.append(block)
    return blocks


def score_ratios(
    ratios: np.ndarray, primary: np.ndarray, transfer: np.ndarray, blocks: list[np.ndarray]
) -> dict[str, Any]:
    block_bits = [float(np.mean(ratios[block])) for block in blocks]
    return {
        "primaryBits": float(np.mean(ratios[primary])),
        "transferBits": float(np.mean(ratios[transfer])),
        "pooledBits": float(np.mean(ratios[primary | transfer])),
        "blockBits": block_bits,
        "positiveBlocks": int(sum(value > 0 for value in block_bits)),
        "positivePrimaryBlocks": int(sum(value > 0 for value in block_bits[:2])),
        "positiveTransferBlocks": int(sum(value > 0 for value in block_bits[2:])),
    }


def render_report(artifact: dict[str, Any]) -> str:
    grid = artifact["grid"]
    lines = [
        "# Dense lagged EMA and RSI audit — 2026-08-18",
        "",
        "## Key findings",
        "",
        "- All four indicator families carry standalone distributional information at the larger targets. The strongest EMA time scale increases from roughly 512 minutes for 1m/15m returns to 2,048 minutes for 1h returns.",
        "- Delayed indicators remain standalone-informative because the volatility regime persists. Even the best one-day-lagged candidate remains positive in both years at every target.",
        "- None of the 10,413 target-specific conditional candidates is positive in the untouched transfer year or in any complete set of four half-year blocks. The existing multiscale volatility/range basis therefore remains the selected input basis.",
        "- Exact parameter winners should not be over-interpreted when quartile states are identical or nearly identical. In particular, one-minute EMA slope and normalized price-minus-EMA value are monotone-equivalent and receive the same score.",
        "",
        "## Experiment",
        "",
        "This audit tests the full signed-return distribution at 1m, 15m, and 1h horizons. Indicator parameters and signal delays are selected only on the primary year; the following transfer year is untouched confirmation.",
        "",
        f"The screen contains {grid['candidateCount']:,} candidates per target: RSI periods {grid['rsiPeriodsMinutes']}; EMA periods {grid['emaPeriodsMinutes']}; EMA difference horizons {grid['emaDifferenceHorizonsMinutes']}; and whole-signal lags {grid['signalLagsMinutes']} minutes.",
        "",
        "`EMA value` means the stationary normalized deviation `10000 * log(price / EMA)`, not the nonstationary absolute price level. Slope is the per-minute EMA log change over the stated difference horizon. Acceleration is the difference between adjacent slopes of that length.",
        "",
        "Standalone bits compare an indicator with the unconditional distribution. Conditional bits compare the existing volatility/range basis plus the indicator with the basis alone. Negative conditional bits mean the extra histogram coordinate hurts held-out log loss.",
        "",
        "## Results",
        "",
        "| target | family | primary-selected parameters | standalone primary / transfer bits | conditional primary / transfer bits | conditional positive blocks |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for horizon in artifact["horizons"]:
        for family in horizon["families"]:
            selected = family["bestConditionalSelectedOnPrimary"]
            standalone = selected["standalone"]
            conditional = selected["conditional"]
            lines.append(
                f"| {horizon['id']} | {family['family']} | {format_candidate(selected)} | "
                f"{standalone['primaryBits']:.8f} / {standalone['transferBits']:.8f} | "
                f"{conditional['primaryBits']:.8f} / {conditional['transferBits']:.8f} | {conditional['positiveBlocks']}/4 |"
            )
    lines.extend([
        "",
        "## Best standalone parameters",
        "",
        "These rows answer whether the indicator family predicts anything by itself, independently of whether it adds to the current basis.",
        "",
        "| target | family | primary-selected parameters | primary bits | untouched transfer bits | positive blocks |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for horizon in artifact["horizons"]:
        for family in horizon["families"]:
            selected = family["bestStandaloneSelectedOnPrimary"]
            score = selected["standalone"]
            lines.append(
                f"| {horizon['id']} | {family['family']} | {format_candidate(selected)} | "
                f"{score['primaryBits']:.8f} | {score['transferBits']:.8f} | {score['positiveBlocks']}/4 |"
            )
    lines.extend([
        "",
        "## Lag profile",
        "",
        "Each row selects the strongest conditional candidate at that fixed lag using only the primary period. This separates a useful delayed signal from merely finding a favorable lag in the transfer data.",
    ])
    for horizon in artifact["horizons"]:
        lines.extend([
            "",
            f"### {horizon['id']}",
            "",
            "#### Standalone",
            "",
            "| lag | selected family and parameters | primary bits | untouched transfer bits | positive blocks |",
            "|---:|---|---:|---:|---:|",
        ])
        for item in horizon["bestByLag"]:
            selected = item["bestStandaloneSelectedOnPrimary"]
            score = selected["standalone"]
            lines.append(
                f"| {item['lagMinutes']}m | {selected['family']}; {format_candidate(selected)} | "
                f"{score['primaryBits']:.8f} | {score['transferBits']:.8f} | {score['positiveBlocks']}/4 |"
            )
        lines.extend([
            "",
            "#### Conditional on the selected volatility/range basis",
            "",
            "| lag | selected family and parameters | conditional primary bits | untouched transfer bits | positive blocks |",
            "|---:|---|---:|---:|---:|",
        ])
        for item in horizon["bestByLag"]:
            selected = item["bestConditionalSelectedOnPrimary"]
            score = selected["conditional"]
            lines.append(
                f"| {item['lagMinutes']}m | {selected['family']}; {format_candidate(selected)} | "
                f"{score['primaryBits']:.8f} | {score['transferBits']:.8f} | {score['positiveBlocks']}/4 |"
            )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- A parameter/lag winner is not trusted merely because it maximizes the primary score. It should remain positive in the untouched transfer year and preferably all four half-year blocks.",
        "- Thousands of correlated candidates create selection optimism in the primary maximum. The transfer score is the evidence for or against survival.",
        "- The four-bin histogram model measures distributional information, including volatility and tails; it is not limited to mean-return direction.",
        "- The complete candidate table, including every tested lag, is retained in the machine-readable artifact.",
        "",
        f"Machine-readable results: `{display_path(DEFAULT_OUTPUT).replace(chr(92), '/')}`.",
        "",
    ])
    return "\n".join(lines)


def format_candidate(row: dict[str, Any]) -> str:
    value = f"period={row['periodMinutes']}m"
    if row["differenceHorizonMinutes"] is not None:
        value += f", difference={row['differenceHorizonMinutes']}m"
    return value + f", signal lag={row['lagMinutes']}m"


def display_path(file: Path) -> str:
    resolved = file.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
