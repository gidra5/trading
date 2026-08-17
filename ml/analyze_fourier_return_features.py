from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_global_feature_basis import (
    ALPHA,
    FEATURE_BINS,
    chronological_blocks,
    fit_unconditional,
    nonoverlapping_rows,
    quantize_features,
    quantize_target,
    score_subset,
    unique_quantiles,
)
from trading_storage import read_shard_payload, resolve_shard


ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT = ROOT / "data/runtime-cache/global-feature-basis"
BASE_RESULT = ROOT / "data/benchmarks/global-return-feature-basis.json"
SPECTRAL_CACHE = ROOT / "data/runtime-cache/fourier-return-features"
OUTPUT = ROOT / "data/benchmarks/fourier-return-feature-information.json"
REPORT = ROOT / "docs/experiments/fourier-return-feature-information-2026-08-17.md"
ONE_SECOND_REFS = ROOT / "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"
ONE_MINUTE_REFS = ROOT / "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m"
WINDOWS = (16, 64, 256)
MAX_ADDED_FEATURES = 2
FINALISTS = 10
PARSIMONY_BITS = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input", type=Path, default=BASE_INPUT)
    parser.add_argument("--base-result", type=Path, default=BASE_RESULT)
    parser.add_argument("--spectral-cache", type=Path, default=SPECTRAL_CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--horizons", default="1s,1m,15m,1h")
    return parser.parse_args()


def spectral_definitions(sample_unit: str) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for window in WINDOWS:
        lookback = f"{window}{sample_unit}"
        common = {
            "windowSamples": window,
            "lookback": lookback,
            "delay": "through latest completed candle",
            "transform": "Hann-windowed rFFT of signed log returns",
        }
        definitions.extend([
            definition(f"fft-log-energy-{lookback}", "Fourier log energy", "Fourier energy", "log1p(sqrt(sum(r^2)))", common),
            definition(f"fft-low-power-share-{lookback}", "Low-frequency power share", "Fourier shape", "power at periods >=8 samples / non-DC power", common),
            definition(f"fft-high-power-share-{lookback}", "High-frequency power share", "Fourier shape", "power at periods <=4 samples / non-DC power", common),
            definition(f"fft-spectral-entropy-{lookback}", "Normalized spectral entropy", "Fourier shape", "entropy of normalized non-DC power / log(number of bins)", common),
            definition(f"fft-spectral-centroid-{lookback}", "Normalized spectral centroid", "Fourier shape", "power-weighted frequency / Nyquist", common),
            definition(f"fft-dominant-frequency-{lookback}", "Dominant normalized frequency", "Fourier shape", "argmax(non-DC power) / Nyquist", common),
            definition(f"fft-k1-real-{lookback}", "Normalized k=1 real coefficient", "Fourier complex coefficients", "Re(FFT[1]) / sqrt(non-DC power)", common),
            definition(f"fft-k1-imag-{lookback}", "Normalized k=1 imaginary coefficient", "Fourier complex coefficients", "Im(FFT[1]) / sqrt(non-DC power)", common),
            definition(f"fft-k2-real-{lookback}", "Normalized k=2 real coefficient", "Fourier complex coefficients", "Re(FFT[2]) / sqrt(non-DC power)", common),
            definition(f"fft-k2-imag-{lookback}", "Normalized k=2 imaginary coefficient", "Fourier complex coefficients", "Im(FFT[2]) / sqrt(non-DC power)", common),
            definition(f"fft-k4-real-{lookback}", "Normalized k=4 real coefficient", "Fourier complex coefficients", "Re(FFT[4]) / sqrt(non-DC power)", common),
            definition(f"fft-k4-imag-{lookback}", "Normalized k=4 imaginary coefficient", "Fourier complex coefficients", "Im(FFT[4]) / sqrt(non-DC power)", common),
        ])
        for order in (0.25, 0.5, 0.75):
            order_id = str(order).replace(".", "p")
            definitions.extend([
                definition(
                    f"frft-{order_id}-k1-real-{lookback}",
                    f"Fractional-DFT order {order} k=1 real coefficient",
                    "fractional Fourier complex coefficients",
                    f"Re(F^{order}[1]) / ||window||2; unitary DFT fractional power",
                    common,
                ),
                definition(
                    f"frft-{order_id}-k1-imag-{lookback}",
                    f"Fractional-DFT order {order} k=1 imaginary coefficient",
                    "fractional Fourier complex coefficients",
                    f"Im(F^{order}[1]) / ||window||2; unitary DFT fractional power",
                    common,
                ),
                definition(
                    f"frft-{order_id}-entropy-{lookback}",
                    f"Fractional-DFT order {order} energy entropy",
                    "fractional Fourier shape",
                    f"entropy of |F^{order}|^2 / log(window samples)",
                    common,
                ),
            ])
        definitions.extend([
            definition(f"haar-fine-energy-share-{lookback}", "Haar fine-scale energy share", "wavelet energy shape", "DWT detail levels 1-2 / total orthonormal energy", common),
            definition(f"haar-mid-energy-share-{lookback}", "Haar mid-scale energy share", "wavelet energy shape", "DWT detail levels 3-4 / total orthonormal energy", common),
            definition(f"haar-coarse-energy-share-{lookback}", "Haar coarse-scale energy share", "wavelet energy shape", "remaining DWT detail and approximation energy / total", common),
            definition(f"haar-latest-detail-l1-{lookback}", "Latest normalized Haar detail, level 1", "wavelet local coefficients", "latest DWT detail coefficient / ||window||2", common),
            definition(f"haar-latest-detail-l2-{lookback}", "Latest normalized Haar detail, level 2", "wavelet local coefficients", "latest DWT detail coefficient / ||window||2", common),
            definition(f"haar-latest-detail-l3-{lookback}", "Latest normalized Haar detail, level 3", "wavelet local coefficients", "latest DWT detail coefficient / ||window||2", common),
            definition(f"morlet-fast-real-{lookback}", "Causal Morlet fast-scale real coefficient", "complex wavelet coefficients", "Re(causal Morlet, scale=window/8) / ||window||2", common),
            definition(f"morlet-fast-imag-{lookback}", "Causal Morlet fast-scale imaginary coefficient", "complex wavelet coefficients", "Im(causal Morlet, scale=window/8) / ||window||2", common),
            definition(f"morlet-slow-real-{lookback}", "Causal Morlet slow-scale real coefficient", "complex wavelet coefficients", "Re(causal Morlet, scale=window/2) / ||window||2", common),
            definition(f"morlet-slow-imag-{lookback}", "Causal Morlet slow-scale imaginary coefficient", "complex wavelet coefficients", "Im(causal Morlet, scale=window/2) / ||window||2", common),
            definition(f"path-efficiency-{lookback}", "Kaufman path efficiency ratio", "path efficiency", "abs(sum(r)) / sum(abs(r))", common),
            definition(f"signed-path-efficiency-{lookback}", "Signed path efficiency ratio", "signed path efficiency", "sum(r) / sum(abs(r))", common),
            definition(f"signed-variance-efficiency-{lookback}", "Signed variance-normalized return efficiency", "signed variance efficiency", "sum(r) / sqrt(sum(r^2))", common),
        ])
    return definitions


def definition(feature_id: str, name: str, family: str, parameters: str, common: dict[str, Any]) -> dict[str, Any]:
    return {"id": feature_id, "name": name, "family": family, "parameters": parameters, **common, "kind": "continuous"}


def time_domain_control_definitions() -> list[dict[str, Any]]:
    common = {"windowSamples": 256, "lookback": "256s", "delay": "through latest completed second", "transform": "direct time-domain control reconstructed exactly from cached Haar/energy coordinates"}
    return [
        definition("return-lag-2s-control", "Return two completed seconds back", "time-domain controls", "r[t-1] at forecast origin t", common),
        definition("return-difference-1s-control", "Raw adjacent-return difference", "time-domain controls", "r[t]-r[t-1]", common),
        definition("return-block-contrast-4s-control", "Raw adjacent two-return block contrast", "time-domain controls", "(r[t-3]+r[t-2])-(r[t-1]+r[t])", common),
    ]


def fractional_dft(values: np.ndarray, order: float) -> np.ndarray:
    """Fractional power of the orthonormal DFT, with F^0=I and F^1=DFT."""
    values = np.asarray(values, dtype=np.complex128)
    powers = [values]
    for _ in range(3):
        powers.append(np.fft.fft(powers[-1], axis=1, norm="ortho"))
    coefficients = []
    for power in range(4):
        coefficient = sum(
            np.exp(-0.5j * np.pi * order * eigen) * np.exp(0.5j * np.pi * power * eigen)
            for eigen in range(4)
        ) / 4
        coefficients.append(coefficient)
    return sum(coefficient * transformed for coefficient, transformed in zip(coefficients, powers))


def haar_features(windows: np.ndarray, norm: np.ndarray) -> np.ndarray:
    approximation = np.asarray(windows, dtype=np.float64)
    detail_energies = []
    latest_details = []
    while approximation.shape[1] >= 2:
        left = approximation[:, 0::2]
        right = approximation[:, 1::2]
        detail = (left - right) / math.sqrt(2)
        approximation = (left + right) / math.sqrt(2)
        detail_energies.append(np.square(detail).sum(axis=1))
        latest_details.append(detail[:, -1] / norm)
    total = np.square(windows).sum(axis=1)
    safe_total = np.maximum(total, np.finfo(np.float64).tiny)
    fine = sum(detail_energies[:2]) / safe_total
    mid = sum(detail_energies[2:4]) / safe_total
    used = sum(detail_energies[:4])
    coarse = (total - used) / safe_total
    return np.column_stack([fine, mid, coarse, *latest_details[:3]])


def morlet_coefficient(windows: np.ndarray, scale: float, norm: np.ndarray) -> np.ndarray:
    size = windows.shape[1]
    lag = np.arange(size - 1, -1, -1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(lag / scale)) * np.exp(-6j * lag / scale)
    kernel /= np.sqrt(np.square(np.abs(kernel)).sum())
    return (windows @ kernel) / norm


def spectral_features(windows: np.ndarray) -> np.ndarray:
    """Return causal Fourier, fractional-Fourier, and wavelet summaries."""
    if windows.ndim != 2 or windows.shape[1] < 4:
        raise ValueError("spectral windows must have shape (rows, samples>=4)")
    rows, size = windows.shape
    weights = np.hanning(size)
    transformed = np.fft.rfft(windows * weights[None, :], axis=1)
    power = np.square(transformed.real) + np.square(transformed.imag)
    positive = power[:, 1:]
    total = positive.sum(axis=1)
    safe_total = np.maximum(total, np.finfo(np.float64).tiny)
    proportions = positive / safe_total[:, None]
    empty = total <= np.finfo(np.float64).tiny
    if np.any(empty):
        proportions[empty] = 0

    bins = positive.shape[1]
    low_end = max(1, size // 8)
    high_start = max(low_end, size // 4)
    low_share = proportions[:, :low_end].sum(axis=1)
    high_share = proportions[:, high_start:].sum(axis=1)
    entropy_terms = np.where(proportions > 0, proportions * np.log(np.maximum(proportions, np.finfo(np.float64).tiny)), 0)
    entropy = -entropy_terms.sum(axis=1) / math.log(bins)
    frequencies = np.arange(1, bins + 1, dtype=np.float64) / bins
    centroid = proportions @ frequencies
    dominant = (np.argmax(positive, axis=1) + 1) / bins
    scale = np.sqrt(safe_total)
    energy = np.log1p(np.sqrt(np.square(windows).sum(axis=1)))

    base = np.column_stack([
        energy,
        low_share,
        high_share,
        entropy,
        centroid,
        dominant,
        transformed[:, 1].real / scale,
        transformed[:, 1].imag / scale,
        transformed[:, 2].real / scale,
        transformed[:, 2].imag / scale,
        transformed[:, 4].real / scale,
        transformed[:, 4].imag / scale,
    ])
    windowed = windows * weights[None, :]
    window_norm = np.sqrt(np.maximum(np.square(windowed).sum(axis=1), np.finfo(np.float64).tiny))
    fractional_columns = []
    for order in (0.25, 0.5, 0.75):
        fractional = fractional_dft(windowed, order)
        fractional_power = np.square(fractional.real) + np.square(fractional.imag)
        fractional_total = np.maximum(fractional_power.sum(axis=1), np.finfo(np.float64).tiny)
        fractional_probability = fractional_power / fractional_total[:, None]
        fractional_entropy = -np.sum(
            np.where(
                fractional_probability > 0,
                fractional_probability * np.log(np.maximum(fractional_probability, np.finfo(np.float64).tiny)),
                0,
            ),
            axis=1,
        ) / math.log(size)
        fractional_columns.extend([
            fractional[:, 1].real / window_norm,
            fractional[:, 1].imag / window_norm,
            fractional_entropy,
        ])
    haar = haar_features(windows, np.sqrt(np.maximum(np.square(windows).sum(axis=1), np.finfo(np.float64).tiny)))
    morlet_fast = morlet_coefficient(windows, size / 8, window_norm)
    morlet_slow = morlet_coefficient(windows, size / 2, window_norm)
    net_return = windows.sum(axis=1)
    absolute_path = np.abs(windows).sum(axis=1)
    raw_norm = np.sqrt(np.square(windows).sum(axis=1))
    path_efficiency = np.divide(np.abs(net_return), absolute_path, out=np.zeros(rows), where=absolute_path > 0)
    signed_path_efficiency = np.divide(net_return, absolute_path, out=np.zeros(rows), where=absolute_path > 0)
    signed_variance_efficiency = np.divide(net_return, raw_norm, out=np.zeros(rows), where=raw_norm > 0)
    result = np.column_stack([
        base,
        *fractional_columns,
        haar,
        morlet_fast.real,
        morlet_fast.imag,
        morlet_slow.real,
        morlet_slow.imag,
        path_efficiency,
        signed_path_efficiency,
        signed_variance_efficiency,
    ])
    result[empty] = 0
    if result.shape != (rows, 34) or not np.isfinite(result).all():
        raise AssertionError("invalid time-frequency feature matrix")
    return result


def decode_close_fast(reference_file: Path) -> tuple[Any, np.ndarray]:
    shard, payload = read_shard_payload(reference_file, verify=False)
    layout = shard.reference["layout"]
    column = next(value for value in layout["columns"] if value["name"] == "close")
    start = int(column["offset"])
    encoded = np.frombuffer(payload, dtype=np.uint8, count=int(column["bytes"]), offset=start)
    if column["encoding"] == "float64-le":
        return shard, encoded.view("<f8")
    if column["encoding"] != "scaled-delta-zigzag-varint":
        raise ValueError(f"unsupported close encoding in {reference_file}")
    ends = np.flatnonzero(encoded < 128)
    if ends.size != shard.axis.count:
        raise ValueError(f"close count mismatch in {reference_file}")
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lengths = ends - starts + 1
    packed = np.zeros(ends.size, dtype=np.uint64)
    for offset in range(int(lengths.max())):
        valid = lengths > offset
        bytes_at_offset = encoded[starts[valid] + offset].astype(np.uint64) & np.uint64(0x7f)
        packed[valid] |= bytes_at_offset << np.uint64(7 * offset)
    deltas = np.where((packed & 1) == 0, packed >> 1, -((packed + 1) >> 1).astype(np.int64)).astype(np.int64)
    values = np.cumsum(deltas, dtype=np.int64).astype(np.float64) / int(column["scale"])
    return shard, values


def extract_spectral_cache(base_input: Path, cache: Path, rebuild: bool) -> dict[str, Any]:
    manifest = json.loads((base_input / "manifest.json").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    metadata_file = cache / "manifest.json"
    if metadata_file.exists() and not rebuild:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("sourceGeneratedAt") == manifest.get("generatedAt"):
            expected = [cache / dataset["file"] for dataset in metadata["datasets"]]
            if all(file.exists() for file in expected):
                return metadata

    outputs = []
    for dataset in manifest["datasets"]:
        dataset_id = dataset["id"]
        if dataset_id not in {"1s", "1m"}:
            continue
        sample_unit = "s" if dataset_id == "1s" else "m"
        step = 1_000 if dataset_id == "1s" else 60_000
        refs = ONE_SECOND_REFS if dataset_id == "1s" else ONE_MINUTE_REFS
        times = np.memmap(base_input / dataset["files"]["times"], dtype="<f8", mode="r", shape=(dataset["rows"],))
        output_file = cache / f"{dataset_id}.features.f32"
        output = np.memmap(output_file, dtype="<f4", mode="w+", shape=(dataset["rows"], len(spectral_definitions(sample_unit))))
        written = np.zeros(dataset["rows"], dtype=bool)
        previous_close = math.nan
        previous_end = None
        tail = np.empty(0, dtype=np.float64)
        earliest = int(times[0]) - max(WINDOWS) * step
        latest = int(times[-1]) + step
        reference_files = sorted(refs.glob("????-??-??.json"))
        for file_index, reference_file in enumerate(reference_files):
            shard = resolve_shard(reference_file)
            shard_end = shard.axis.start + shard.axis.step * shard.axis.count
            if shard_end < earliest:
                continue
            if shard.axis.start >= latest:
                break
            shard, closes = decode_close_fast(reference_file)
            contiguous = previous_end is None or shard.axis.start == previous_end
            returns = np.zeros(closes.size, dtype=np.float64)
            if contiguous and math.isfinite(previous_close):
                returns[0] = math.log(closes[0] / previous_close) * 10_000
            if closes.size > 1:
                returns[1:] = np.log(closes[1:] / closes[:-1]) * 10_000
            if not contiguous:
                tail = np.empty(0, dtype=np.float64)
            combined = np.concatenate([tail, returns])
            lo = int(np.searchsorted(times, shard.axis.start, side="left"))
            hi = int(np.searchsorted(times, shard_end, side="left"))
            if hi > lo:
                origin_times = np.asarray(times[lo:hi], dtype=np.int64)
                local = (origin_times - shard.axis.start) // shard.axis.step
                if not np.all(shard.axis.start + local * shard.axis.step == origin_times):
                    raise ValueError(f"sample times do not align to {reference_file}")
                end = tail.size + local
                columns = []
                for window in WINDOWS:
                    if np.any(end < window - 1):
                        raise ValueError(f"insufficient causal history at {reference_file}")
                    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
                    history = combined[end[:, None] - offsets[None, :]]
                    columns.append(spectral_features(history))
                output[lo:hi] = np.concatenate(columns, axis=1).astype(np.float32)
                written[lo:hi] = True
            tail = combined[-(max(WINDOWS) - 1):]
            previous_close = float(closes[-1])
            previous_end = shard_end
            if (file_index + 1) % 100 == 0:
                print(f"{dataset_id} Fourier extraction: {file_index + 1}/{len(reference_files)} shards", flush=True)
        output.flush()
        if not written.all():
            missing = int((~written).sum())
            raise ValueError(f"{dataset_id} Fourier cache is missing {missing} rows")
        outputs.append({
            "id": dataset_id,
            "rows": dataset["rows"],
            "featureCount": len(spectral_definitions(sample_unit)),
            "file": output_file.name,
            "sampleUnit": sample_unit,
            "features": spectral_definitions(sample_unit),
        })
    metadata = {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "sourceGeneratedAt": manifest.get("generatedAt"),
        "windows": list(WINDOWS),
        "datasets": outputs,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def quantize_columns(values: np.ndarray, train: np.ndarray, definitions: list[dict[str, Any]]) -> tuple[np.ndarray, list[list[float]], list[int]]:
    output = np.empty(values.shape, dtype=np.uint8)
    edges: list[list[float]] = []
    arities: list[int] = []
    for column, definition in enumerate(definitions):
        raw = np.asarray(values[:, column], dtype=np.float64)
        cut = unique_quantiles(raw[train], FEATURE_BINS)
        output[:, column] = np.searchsorted(cut, raw, side="right").astype(np.uint8)
        edges.append(cut.tolist())
        arities.append(len(cut) + 1)
    return output, edges, arities


def target_variants(values: np.ndarray, train: np.ndarray) -> dict[str, tuple[np.ndarray, int, list[float]]]:
    full, full_edges, full_classes, _ = quantize_target(values, train)
    magnitude, magnitude_edges, magnitude_classes, _ = quantize_target(np.abs(values), train)
    zero_separate = float(np.mean(values[train] == 0)) >= 0.001
    if zero_separate:
        sign = np.where(values < 0, 0, np.where(values > 0, 2, 1)).astype(np.int16)
        sign_classes = 3
    else:
        sign = (values >= 0).astype(np.int16)
        sign_classes = 2
    return {
        "full distribution": (full, full_classes, full_edges),
        "magnitude": (magnitude, magnitude_classes, magnitude_edges),
        "sign": (sign, sign_classes, []),
    }


def subtract_score(score: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "conditionalPrimaryBits": score["primaryBits"] - baseline["primaryBits"],
        "conditionalTransferBits": score["transferBits"] - baseline["transferBits"],
        "conditionalBlockBits": [left - right for left, right in zip(score["blockBits"], baseline["blockBits"])],
        "combinedPrimaryBits": score["primaryBits"],
        "combinedTransferBits": score["transferBits"],
    }


def analyze_objective(
    target: np.ndarray,
    classes: int,
    quantized: np.ndarray,
    train: np.ndarray,
    primary: np.ndarray,
    transfer: np.ndarray,
    blocks: list[np.ndarray],
    arities: list[int],
    baseline_size: int,
    definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    unconditional = fit_unconditional(target[train], classes)
    baseline_indices = tuple(range(baseline_size))
    baseline = score_subset(baseline_indices, quantized, target, train, primary, transfer, blocks, arities, classes, unconditional)
    singles = []
    for feature_offset, feature in enumerate(definitions):
        feature_index = baseline_size + feature_offset
        standalone = score_subset((feature_index,), quantized, target, train, primary, transfer, blocks, arities, classes, unconditional)
        combined = score_subset(baseline_indices + (feature_index,), quantized, target, train, primary, transfer, blocks, arities, classes, unconditional)
        singles.append({
            **feature,
            "featureOffset": feature_offset,
            "standalonePrimaryBits": standalone["primaryBits"],
            "standaloneTransferBits": standalone["transferBits"],
            **subtract_score(combined, baseline),
        })
    singles.sort(key=lambda row: row["conditionalPrimaryBits"], reverse=True)
    stable = [row for row in singles if all(value > 0 for value in row["conditionalBlockBits"][:2])]
    finalists = []
    represented_families: set[str] = set()
    for row in stable:
        if row["family"] not in represented_families:
            finalists.append(row)
            represented_families.add(row["family"])
        if len(finalists) >= FINALISTS:
            break
    if len(finalists) < FINALISTS:
        selected_ids = {row["id"] for row in finalists}
        finalists.extend(row for row in stable if row["id"] not in selected_ids) 
        finalists = finalists[:FINALISTS]
    candidates = []
    for size in range(1, min(MAX_ADDED_FEATURES, len(finalists)) + 1):
        for subset in itertools.combinations(finalists, size):
            offsets = tuple(int(row["featureOffset"]) for row in subset)
            indices = baseline_indices + tuple(baseline_size + offset for offset in offsets)
            score = score_subset(indices, quantized, target, train, primary, transfer, blocks, arities, classes, unconditional)
            delta = subtract_score(score, baseline)
            if all(value > 0 for value in delta["conditionalBlockBits"][:2]):
                candidates.append({"offsets": offsets, **delta})
    if candidates:
        optimum = max(candidates, key=lambda row: row["conditionalPrimaryBits"])
        threshold = optimum["conditionalPrimaryBits"] - PARSIMONY_BITS
        selected = min(
            (row for row in candidates if row["conditionalPrimaryBits"] >= threshold),
            key=lambda row: (len(row["offsets"]), -row["conditionalPrimaryBits"]),
        )
        selected_result = {
            "features": [definitions[offset]["id"] for offset in selected["offsets"]],
            **{key: value for key, value in selected.items() if key != "offsets"},
        }
    else:
        selected_result = {
            "features": [],
            "conditionalPrimaryBits": 0.0,
            "conditionalTransferBits": 0.0,
            "conditionalBlockBits": [0.0, 0.0, 0.0, 0.0],
            "combinedPrimaryBits": baseline["primaryBits"],
            "combinedTransferBits": baseline["transferBits"],
        }
    return {
        "baseline": baseline,
        "selected": selected_result,
        "singleFeatureRanking": singles,
        "finalists": [row["id"] for row in finalists],
    }


def analyze_post_lag2_ablation(
    target_variants_by_name: dict[str, tuple[np.ndarray, int, list[float]]],
    quantized: np.ndarray,
    train: np.ndarray,
    primary: np.ndarray,
    transfer: np.ndarray,
    blocks: list[np.ndarray],
    arities: list[int],
    baseline_size: int,
    definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    feature_offset = {feature["id"]: index for index, feature in enumerate(definitions)}
    lag2 = baseline_size + feature_offset["return-lag-2s-control"]
    candidate_ids = [
        "haar-latest-detail-l1-16s",
        "haar-latest-detail-l2-256s",
        "signed-path-efficiency-16s",
        "signed-variance-efficiency-16s",
    ]
    candidate_indices = [baseline_size + feature_offset[feature_id] for feature_id in candidate_ids]
    original_indices = tuple(range(baseline_size))
    fixed_indices = original_indices + (lag2,)
    results = {}
    for objective_name, (target, classes, _) in target_variants_by_name.items():
        unconditional = fit_unconditional(target[train], classes)
        original = score_subset(original_indices, quantized, target, train, primary, transfer, blocks, arities, classes, unconditional)
        fixed = score_subset(fixed_indices, quantized, target, train, primary, transfer, blocks, arities, classes, unconditional)
        candidates = []
        all_candidates = []
        for size in (1, 2):
            for subset in itertools.combinations(range(len(candidate_indices)), size):
                indices = fixed_indices + tuple(candidate_indices[index] for index in subset)
                score = score_subset(indices, quantized, target, train, primary, transfer, blocks, arities, classes, unconditional)
                delta = subtract_score(score, fixed)
                row = {
                    "features": [candidate_ids[index] for index in subset],
                    **delta,
                }
                all_candidates.append(row)
                if all(value > 0 for value in delta["conditionalBlockBits"][:2]):
                    candidates.append(row)
        if candidates:
            optimum = max(candidates, key=lambda row: row["conditionalPrimaryBits"])
            threshold = optimum["conditionalPrimaryBits"] - PARSIMONY_BITS
            selected = min(
                (row for row in candidates if row["conditionalPrimaryBits"] >= threshold),
                key=lambda row: (len(row["features"]), -row["conditionalPrimaryBits"]),
            )
        else:
            selected = {
                "features": [],
                "conditionalPrimaryBits": 0.0,
                "conditionalTransferBits": 0.0,
                "conditionalBlockBits": [0.0, 0.0, 0.0, 0.0],
                "combinedPrimaryBits": fixed["primaryBits"],
                "combinedTransferBits": fixed["transferBits"],
            }
        results[objective_name] = {
            "lag2GainBeyondOriginalBaseline": subtract_score(fixed, original),
            "selectedWaveletsBeyondLag2": selected,
            "candidateRankingBeyondLag2": sorted(
                all_candidates,
                key=lambda row: row["conditionalPrimaryBits"],
                reverse=True,
            ),
        }
    return results


def analyze(args: argparse.Namespace, spectral_manifest: dict[str, Any]) -> dict[str, Any]:
    base_manifest = json.loads((args.base_input / "manifest.json").read_text(encoding="utf-8"))
    base_result = json.loads(args.base_result.read_text(encoding="utf-8"))
    spectral_by_id = {dataset["id"]: dataset for dataset in spectral_manifest["datasets"]}
    selected_by_horizon = {horizon["id"]: horizon for horizon in base_result["horizons"]}
    requested = set(args.horizons.split(","))
    results = []
    for dataset in base_manifest["datasets"]:
        if dataset["id"] not in spectral_by_id:
            continue
        spectral_meta = spectral_by_id[dataset["id"]]
        base_features = np.memmap(args.base_input / dataset["files"]["features"], dtype="<f4", mode="r", shape=(dataset["rows"], dataset["featureCount"]))
        targets = np.memmap(args.base_input / dataset["files"]["targets"], dtype="<f4", mode="r", shape=(dataset["rows"], dataset["targetCount"]))
        splits_all = np.memmap(args.base_input / dataset["files"]["splits"], dtype="u1", mode="r", shape=(dataset["rows"],))
        times_all = np.memmap(args.base_input / dataset["files"]["times"], dtype="<f8", mode="r", shape=(dataset["rows"],))
        spectral = np.memmap(args.spectral_cache / spectral_meta["file"], dtype="<f4", mode="r", shape=(dataset["rows"], spectral_meta["featureCount"]))
        feature_index = {feature["id"]: index for index, feature in enumerate(dataset["features"])}
        for target_index, target_meta in enumerate(dataset["targets"]):
            if target_meta["id"] not in requested:
                continue
            horizon = selected_by_horizon[target_meta["id"]]
            selected_rows = nonoverlapping_rows(times_all, splits_all, target_meta["minutes"] * 60_000)
            splits = np.asarray(splits_all[selected_rows], dtype=np.uint8)
            train = splits == 0
            primary = splits == 1
            transfer = splits == 2
            blocks = chronological_blocks(primary, transfer)
            baseline_ids = [feature["id"] for feature in horizon["selectedBasis"]]
            baseline_indices = [feature_index[feature_id] for feature_id in baseline_ids]
            baseline_definitions = [dataset["features"][index] for index in baseline_indices]
            raw_baseline = np.asarray(base_features[selected_rows][:, baseline_indices], dtype=np.float64)
            raw_spectral = spectral[selected_rows]
            candidate_definitions = list(spectral_meta["features"])
            q_base, base_edges, base_arities = quantize_features(raw_baseline, train, baseline_definitions)
            q_spectral, spectral_edges, spectral_arities = quantize_columns(raw_spectral, train, spectral_meta["features"])
            if dataset["id"] == "1s":
                spectral_index = {feature["id"]: index for index, feature in enumerate(spectral_meta["features"])}
                norm_256s = np.expm1(np.asarray(raw_spectral[:, spectral_index["fft-log-energy-256s"]], dtype=np.float64))
                haar_l1_256s = np.asarray(raw_spectral[:, spectral_index["haar-latest-detail-l1-256s"]], dtype=np.float64)
                haar_l2_256s = np.asarray(raw_spectral[:, spectral_index["haar-latest-detail-l2-256s"]], dtype=np.float64)
                current_return = np.asarray(raw_baseline[:, baseline_ids.index("previous-return-1s")], dtype=np.float64)
                lagged_return = current_return + math.sqrt(2) * norm_256s * haar_l1_256s
                controls = np.column_stack([
                    lagged_return,
                    current_return - lagged_return,
                    2 * norm_256s * haar_l2_256s,
                ])
                control_definitions = time_domain_control_definitions()
                q_controls, control_edges, control_arities = quantize_columns(controls, train, control_definitions)
                q_spectral = np.concatenate([q_spectral, q_controls], axis=1)
                spectral_edges.extend(control_edges)
                spectral_arities.extend(control_arities)
                candidate_definitions.extend(control_definitions)
            quantized = np.concatenate([q_base, q_spectral], axis=1)
            arities = base_arities + spectral_arities
            raw_target = np.asarray(targets[selected_rows, target_index], dtype=np.float64)
            objective_targets = target_variants(raw_target, train)
            objectives = {}
            for objective_name, (target_bins, classes, edges) in objective_targets.items():
                print(f"Scoring time-frequency features for {target_meta['id']} {objective_name}...", flush=True)
                objective = analyze_objective(
                    target_bins, classes, quantized, train, primary, transfer, blocks,
                    arities, len(baseline_ids), candidate_definitions,
                )
                objective["classes"] = classes
                objective["targetEdges"] = edges
                objectives[objective_name] = objective
            full_ranking = objectives["full distribution"]["singleFeatureRanking"]
            result = {
                "id": target_meta["id"],
                "horizonMinutes": target_meta["minutes"],
                "observations": {"train": int(train.sum()), "primary": int(primary.sum()), "transfer": int(transfer.sum())},
                "baselineFeatures": baseline_ids,
                "baselineQuantileEdges": base_edges,
                "spectralFeatures": [
                    {**feature, "quantileEdges": spectral_edges[index], "arity": spectral_arities[index]}
                    for index, feature in enumerate(candidate_definitions)
                ],
                "objectives": objectives,
                "topStandalone": sorted(full_ranking, key=lambda row: row["standalonePrimaryBits"], reverse=True)[:10],
            }
            if dataset["id"] == "1s":
                result["postLag2Ablation"] = analyze_post_lag2_ablation(
                    objective_targets, quantized, train, primary, transfer, blocks,
                    arities, len(baseline_ids), candidate_definitions,
                )
            results.append(result)
    return {
        "version": 1,
        "generatedAt": np.datetime_as_string(np.datetime64("now"), unit="s") + "Z",
        "objective": "Causal Fourier, fractional-Fourier, and wavelet information beyond the fixed long-history global return feature basis",
        "metric": "chronological held-out log2 likelihood gain in bits per target",
        "method": {
            "windows": list(WINDOWS),
            "transform": "Hann-windowed FFT and unitary fractional-DFT powers plus Haar and causal complex-Morlet transforms of trailing signed returns",
            "featureBins": FEATURE_BINS,
            "smoothingAlpha": ALPHA,
            "selection": f"up to {FINALISTS} primary-stable time-frequency finalists with family coverage; exhaustive additions of one or two; smallest subset within {PARSIMONY_BITS} bits of the primary maximum",
            "transferPolicy": "transfer period is untouched until after selection",
            "warning": "Fourier log energy is a volatility proxy by Parseval's theorem; normalized complex, fractional-domain, and wavelet coordinates test non-volatility time-frequency information.",
        },
        "source": {
            "baseManifest": str((args.base_input / "manifest.json").relative_to(ROOT)).replace("\\", "/"),
            "baseResult": str(args.base_result.relative_to(ROOT)).replace("\\", "/"),
            "spectralCache": str(args.spectral_cache.relative_to(ROOT)).replace("\\", "/"),
        },
        "horizons": results,
    }


def render_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Fourier, fractional-Fourier, and wavelet features for BTC return distributions — 2026-08-17",
        "",
        "## Result",
        "",
        "The audit applies causal FFT, unitary fractional-DFT powers at orders 0.25/0.5/0.75, orthonormal Haar decompositions, and complex causal Morlet filters to trailing signed returns. Complex coordinates are supplied as separate normalized real and imaginary values. It tests 16, 64, and 256-sample windows at each native input cadence. Selection uses only the primary period; the later transfer year is untouched until confirmation.",
        "",
        "| target | fixed baseline | selected time-frequency additions | primary gain (bits) | transfer gain (bits) | transfer conclusion |",
        "|---:|---|---|---:|---:|---|",
    ]
    for horizon in artifact["horizons"]:
        selected = horizon["objectives"]["full distribution"]["selected"]
        conclusion = "confirmed" if selected["conditionalTransferBits"] > 0 else "not confirmed"
        lines.append(
            f"| {horizon['id']} | {', '.join(horizon['baselineFeatures'])} | {', '.join(selected['features']) or 'none'} | "
            f"{selected['conditionalPrimaryBits']:.8f} | {selected['conditionalTransferBits']:.8f} | {conclusion} |"
        )
    lines.extend([
        "",
        "Fourier energy is not intrinsically new: by Parseval's theorem it is a re-expression of return energy and therefore closely overlaps realized volatility. A useful result requires normalized complex coefficients, fractional-domain structure, or wavelet localization to remain positive after conditioning on the fixed basis and on the untouched transfer period.",
        "",
        "## Efficiency-ratio audit",
        "",
        "| target | objective | best efficiency coordinate | standalone primary/transfer | conditional primary/transfer | conditional blocks |",
        "|---:|---|---|---:|---:|---|",
    ])
    for horizon in artifact["horizons"]:
        for name in ("full distribution", "magnitude", "sign"):
            efficiency = [
                row for row in horizon["objectives"][name]["singleFeatureRanking"]
                if "efficiency" in row["id"]
            ]
            best = max(efficiency, key=lambda row: row["conditionalPrimaryBits"])
            lines.append(
                f"| {horizon['id']} | {name} | {best['id']} | {best['standalonePrimaryBits']:.6f} / {best['standaloneTransferBits']:.6f} | "
                f"{best['conditionalPrimaryBits']:.6f} / {best['conditionalTransferBits']:.6f} | {', '.join(f'{value:.5f}' for value in best['conditionalBlockBits'])} |"
            )
    for horizon in artifact["horizons"]:
        objectives = horizon["objectives"]
        lines.extend([
            "",
            f"## {horizon['id']} target",
            "",
            f"Observations: {horizon['observations']['train']:,} train, {horizon['observations']['primary']:,} primary, {horizon['observations']['transfer']:,} transfer.",
            "",
            "| objective | selected time-frequency additions | primary gain | transfer gain | primary/transfer half-block gains |",
            "|---|---|---:|---:|---|",
        ])
        for name in ("full distribution", "magnitude", "sign"):
            selected = objectives[name]["selected"]
            lines.append(
                f"| {name} | {', '.join(selected['features']) or 'none'} | {selected['conditionalPrimaryBits']:.8f} | "
                f"{selected['conditionalTransferBits']:.8f} | {', '.join(f'{value:.6f}' for value in selected['conditionalBlockBits'])} |"
            )
        if "postLag2Ablation" in horizon:
            lines.extend([
                "",
                "After fixing the second-lag return in the baseline:",
                "",
                "| objective | lag-2 primary/transfer gain | remaining selected path features | additional primary/transfer gain |",
                "|---|---:|---|---:|",
            ])
            for name in ("full distribution", "magnitude", "sign"):
                row = horizon["postLag2Ablation"][name]
                lag = row["lag2GainBeyondOriginalBaseline"]
                wavelet = row["selectedWaveletsBeyondLag2"]
                lines.append(
                    f"| {name} | {lag['conditionalPrimaryBits']:.8f} / {lag['conditionalTransferBits']:.8f} | "
                    f"{', '.join(wavelet['features']) or 'none'} | {wavelet['conditionalPrimaryBits']:.8f} / {wavelet['conditionalTransferBits']:.8f} |"
                )
            sign_candidates = horizon["postLag2Ablation"]["sign"].get("candidateRankingBeyondLag2", [])
            single_sign_candidates = [row for row in sign_candidates if len(row["features"]) == 1]
            if single_sign_candidates:
                lines.extend([
                    "",
                    "Individual 1s sign coordinates after fixing lag 2:",
                    "",
                    "| coordinate | additional primary/transfer bits | blocks |",
                    "|---|---:|---|",
                ])
                for candidate in single_sign_candidates:
                    lines.append(
                        f"| {candidate['features'][0]} | {candidate['conditionalPrimaryBits']:.8f} / "
                        f"{candidate['conditionalTransferBits']:.8f} | "
                        f"{', '.join(f'{value:.6f}' for value in candidate['conditionalBlockBits'])} |"
                    )
            lines.extend([
                "",
                "For the 1s sign head, the surviving coefficient after fixing lag 2 is",
                "",
                "$$",
                "d_{1,16}(t)=\\frac{r_{t-1}-r_t}{\\sqrt{2}\\,\\sqrt{\\sum_{j=0}^{15}r_{t-j}^2}}.",
                "$$",
                "",
                "The tested 16-s efficiency coordinates were Kaufman's path ratio, its signed form, and a variance-normalized signed form:",
                "",
                "$$",
                "ER_{16}(t)=\\frac{|\\sum_{j=0}^{15}r_{t-j}|}{\\sum_{j=0}^{15}|r_{t-j}|},\\qquad SER_{16}(t)=\\frac{\\sum_{j=0}^{15}r_{t-j}}{\\sum_{j=0}^{15}|r_{t-j}|},\\qquad E_{16}(t)=\\frac{\\sum_{j=0}^{15}r_{t-j}}{\\sqrt{\\sum_{j=0}^{15}r_{t-j}^2}}.",
                "$$",
                "",
                "Thus the complete-distribution gain is primarily ordinary second-lag information. The efficiency ratios retain a small, stable 1s sign gain after lag 2, but the normalized adjacent-return Haar contrast is stronger and is the parsimonious selection. No efficiency coordinate survives for the full-distribution or magnitude heads, or at 1m through 1h.",
            ])
        lines.extend([
            "",
            "Top full-distribution time-frequency coordinates, each added separately to the fixed baseline:",
            "",
            "| feature | family | lookback | standalone primary/transfer | conditional primary/transfer | conditional blocks |",
            "|---|---|---:|---:|---:|---|",
        ])
        for row in objectives["full distribution"]["singleFeatureRanking"][:10]:
            lines.append(
                f"| {row['id']} | {row['family']} | {row['lookback']} | {row['standalonePrimaryBits']:.6f} / {row['standaloneTransferBits']:.6f} | "
                f"{row['conditionalPrimaryBits']:.6f} / {row['conditionalTransferBits']:.6f} | {', '.join(f'{value:.5f}' for value in row['conditionalBlockBits'])} |"
            )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- The FFT windows end at the latest completed candle, so no future sample enters a feature.",
        "- Quartile feature cells detect robust distribution changes but can miss smooth information that a neural sequence model could exploit.",
        "- This audit tests fixed-window transform coordinates, not a learned spectral/wavelet layer or arbitrary transform-parameter search.",
        "- Selection is conditional on the previously fixed global basis. Negative transfer gain means the apparent primary-period addition should not be promoted.",
        "",
        "Machine-readable results are stored in `data/benchmarks/fourier-return-feature-information.json`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    spectral_manifest = extract_spectral_cache(args.base_input, args.spectral_cache, args.rebuild_cache)
    if args.extract_only:
        print(f"Wrote {args.spectral_cache / 'manifest.json'}")
        return
    artifact = analyze(args, spectral_manifest)
    requested = set(args.horizons.split(","))
    if args.output.exists() and requested != {"1s", "1m", "15m", "1h"}:
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        merged = {horizon["id"]: horizon for horizon in previous.get("horizons", [])}
        merged.update({horizon["id"]: horizon for horizon in artifact["horizons"]})
        order = {"1s": 0, "1m": 1, "15m": 2, "1h": 3}
        artifact["horizons"] = sorted(merged.values(), key=lambda horizon: order[horizon["id"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(artifact), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
