from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import lfilter

from global_feature_basis_search import FeatureGroup
from global_feature_registry import (
    ROOT,
    coordinate_id,
    infer_subject,
    normalize_representative_formula,
    normalize_second_formula,
    normalize_cross_public,
    public_subject,
    representative_cadence,
    representative_venue,
    safe_id,
)


AXIS_DIR = ROOT / "data/runtime-cache/binance-cross-asset-1m-basis-30d"
BASE_DIR = ROOT / "data/runtime-cache/global-feature-basis-30d"
SOURCE_COLUMNS = 10


@dataclass(frozen=True)
class RawCandidateBatch:
    groups: list[FeatureGroup]
    values: np.ndarray


@dataclass(frozen=True)
class QuantizedCandidateBatch:
    groups: list[FeatureGroup]
    states: np.ndarray
    edges: list[np.ndarray]


def quantize_batch_fast(
    values: np.ndarray,
    train: np.ndarray,
    bins: int = 4,
    sample_rows: int | None = 4_096,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Vectorized equal-mass quantization for a wide streamed batch.

    Repeated edges intentionally remain as empty levels. This preserves the
    partition represented by unique-edge quantization while allowing one
    vectorized pass over hundreds of candidates.
    """
    values = np.asarray(values)
    train_values = values[np.asarray(train, dtype=bool)]
    if sample_rows is not None and train_values.shape[0] > sample_rows:
        sample = np.linspace(0, train_values.shape[0] - 1, sample_rows, dtype=np.int64)
        train_values = train_values[sample]
    finite_counts = np.sum(np.isfinite(train_values), axis=0)
    safe = train_values.copy()
    safe[:, finite_counts == 0] = 0.0
    with np.errstate(all="ignore"):
        edge_matrix = np.nanquantile(safe, np.arange(1, bins) / bins, axis=0)
    states = np.zeros(values.shape, dtype=np.uint8)
    finite = np.isfinite(values)
    for edge in edge_matrix:
        states += (values >= edge).astype(np.uint8)
    has_missing = np.any(~finite, axis=0)
    states[~finite] = bins
    edges = [edge_matrix[:, index].astype(np.float64) for index in range(values.shape[1])]
    arities = [bins + int(missing) for missing in has_missing]
    return states, edges, arities


def exponential_filter(values: np.ndarray, alpha: float, initial: float) -> np.ndarray:
    output, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], values, zi=[(1.0 - alpha) * initial])
    return output


def causal_fill(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    valid = observed & np.isfinite(result) & (result > 0)
    indices = np.maximum.accumulate(np.where(valid, np.arange(result.size), -1))
    first = int(np.flatnonzero(valid)[0]) if np.any(valid) else 0
    indices[indices < 0] = first
    return result[indices]


def dense_grid() -> dict[str, list[int]]:
    artifact = json.loads((ROOT / "data/benchmarks/dense-lagged-indicator-audit.json").read_text(encoding="utf-8"))
    grid = artifact["grid"]
    return {
        "rsi": [int(value) for value in grid["rsiPeriodsMinutes"]],
        "ema": [int(value) for value in grid["emaPeriodsMinutes"]],
        "horizons": [int(value) for value in grid["emaDifferenceHorizonsMinutes"]],
        "lags": [int(value) for value in grid["signalLagsMinutes"]],
    }


def origin_rows() -> tuple[np.ndarray, np.ndarray]:
    base = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    dataset = base["datasets"][0]
    times = np.asarray(np.memmap(
        BASE_DIR / dataset["files"]["times"], dtype="<f8", mode="r", shape=(int(dataset["rows"]),)
    ))
    splits = np.asarray(np.memmap(
        BASE_DIR / dataset["files"]["splits"], dtype="u1", mode="r", shape=(int(dataset["rows"]),)
    ))
    axis = json.loads((AXIS_DIR / "manifest.json").read_text(encoding="utf-8"))
    start = float(np.datetime64(axis["window"]["start"].removesuffix("Z")).astype("datetime64[ms]").astype(np.int64))
    rows = np.floor((times - start) / 60_000).astype(np.int64)
    return rows, splits


def preferred_assets() -> list[dict]:
    manifest = json.loads((AXIS_DIR / "manifest.json").read_text(encoding="utf-8"))
    output = []
    for asset in manifest["assets"]:
        preferred = asset.get("preferredMarket")
        if not preferred:
            continue
        market = next((row for row in asset["markets"] if row["venue"] == preferred["venue"]), None)
        if market is not None and float(market["coverage"]) >= 0.99:
            output.append({**asset, "selectedMarket": market})
    return output


def load_close(asset: dict) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((AXIS_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = int(manifest["window"]["rows"])
    market = asset["selectedMarket"]
    values = np.memmap(ROOT / market["file"], dtype="<f4", mode="r", shape=(rows, SOURCE_COLUMNS))
    observed = np.asarray(values[:, 9] == 1)
    return causal_fill(np.asarray(values[:, 3]), observed), observed


def base_signals(close: np.ndarray, grid: dict[str, list[int]]) -> Iterator[tuple[str, str, np.ndarray]]:
    changes = np.diff(close, prepend=close[0])
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    for period in grid["rsi"]:
        average_gain = exponential_filter(gains, 1.0 / period, 0.0)
        average_loss = exponential_filter(losses, 1.0 / period, 0.0)
        values = np.full(close.shape, 50.0, dtype=np.float64)
        positive_loss = average_loss > 0
        values[positive_loss] = 100.0 - 100.0 / (1.0 + average_gain[positive_loss] / average_loss[positive_loss])
        values[(~positive_loss) & (average_gain > 0)] = 100.0
        yield "RSI", f"rsi-{period}m", values
    for period in grid["ema"]:
        average = exponential_filter(close, 2.0 / (period + 1.0), float(close[0]))
        yield "EMA value", f"ema-distance-{period}m", 10_000.0 * np.log(close / average)
        for horizon in grid["horizons"]:
            slope = np.zeros(close.shape, dtype=np.float64)
            slope[horizon:] = 10_000.0 * np.log(average[horizon:] / average[:-horizon]) / horizon
            yield "EMA slope", f"ema-slope-{period}m-{horizon}m", slope
            acceleration = np.zeros(close.shape, dtype=np.float64)
            twice = 2 * horizon
            acceleration[twice:] = slope[twice:] - slope[horizon:-horizon]
            yield "EMA acceleration", f"ema-acceleration-{period}m-{horizon}m", acceleration


def lagged_at_origins(
    values: np.ndarray,
    observed: np.ndarray,
    rows: np.ndarray,
    lag: int,
) -> np.ndarray:
    indices = rows - lag
    result = np.full(rows.shape, np.nan, dtype=np.float32)
    valid = indices >= 0
    result[valid] = values[indices[valid]].astype(np.float32)
    result[valid & ~observed[indices.clip(min=0)]] = np.nan
    return result


class DenseMinuteBatchProvider:
    def __init__(
        self,
        *,
        batch_size: int = 256,
        limit_assets: int | None = None,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
        progress: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        assets = preferred_assets()
        if self.selected_ids is not None:
            subjects = {feature_id.split("/", 4)[1] for feature_id in self.selected_ids}
            assets = [row for row in assets if safe_id(str(row["asset"])) in subjects]
        self.assets = assets[:limit_assets]
        self.quantile_sample_rows = quantile_sample_rows
        self.progress = progress
        self.rows, self.splits = origin_rows()
        self.grid = dense_grid()

    @property
    def coordinate_count(self) -> int:
        base = len(self.grid["rsi"]) + len(self.grid["ema"]) * (1 + 2 * len(self.grid["horizons"]))
        return len(self.assets) * base * len(self.grid["lags"])

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        groups: list[FeatureGroup] = []
        columns: list[np.ndarray] = []
        started = time.perf_counter()
        for asset_index, asset in enumerate(self.assets):
            subject = str(asset["asset"])
            close, observed = load_close(asset)
            for family, formula, signal in base_signals(close, self.grid):
                for lag in self.grid["lags"]:
                    lagged_formula = formula if lag == 0 else f"{formula}-lag-{lag}m"
                    feature_id = coordinate_id(
                        subject, "asset", "binance-preferred", "1m", lagged_formula
                    )
                    if self.selected_ids is not None and feature_id not in self.selected_ids:
                        continue
                    groups.append(FeatureGroup(
                        feature_id,
                        0,
                        1.0,
                    ))
                    columns.append(lagged_at_origins(signal, observed, self.rows, lag))
                    if len(columns) >= self.batch_size:
                        yield RawCandidateBatch(groups, np.column_stack(columns))
                        groups, columns = [], []
            if self.progress and ((asset_index + 1) % 5 == 0 or asset_index + 1 == len(self.assets)):
                elapsed = time.perf_counter() - started
                print(
                    f"Dense minute generation {asset_index + 1}/{len(self.assets)} assets "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
        if columns:
            yield RawCandidateBatch(groups, np.column_stack(columns))

    def quantized_batches(self, train: np.ndarray | None = None, bins: int = 4) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class BaseRecentBatchProvider:
    """The existing 147-coordinate BTC/general dataset in canonical identity."""

    def __init__(self, *, selected_ids: set[str] | None = None) -> None:
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.manifest = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.dataset = self.manifest["datasets"][0]
        catalog_artifact = json.loads(
            (ROOT / "data/benchmarks/binance-cross-asset-component-feature-bases.json").read_text(
                encoding="utf-8"
            )
        )
        self.catalog = {
            str(row["id"]): row
            for row in catalog_artifact["featureCatalog"]
            if row.get("scope") == "existing BTC/general inventory"
        }
        rows = int(self.dataset["rows"])
        self.values = np.memmap(
            BASE_DIR / self.dataset["files"]["features"],
            dtype="<f4",
            mode="r",
            shape=(rows, int(self.dataset["featureCount"])),
        )
        self.splits = np.asarray(np.memmap(
            BASE_DIR / self.dataset["files"]["splits"], dtype="u1", mode="r", shape=(rows,)
        ))

    @property
    def coordinate_count(self) -> int:
        if self.selected_ids is None:
            return int(self.dataset["featureCount"])
        return sum(
            self.feature_id(definition) in self.selected_ids
            for definition in self.dataset["features"]
        )

    def feature_id(self, definition: dict) -> str:
        feature = str(definition["id"])
        row = self.catalog[feature]
        subject = infer_subject(row)
        return coordinate_id(
            subject,
            "asset" if subject != "general" else "general",
            representative_venue(row, subject),
            representative_cadence(feature, str(row.get("lookback", ""))),
            normalize_representative_formula(feature, subject, str(row.get("scope", ""))),
        )

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        groups = []
        columns = []
        for index, definition in enumerate(self.dataset["features"]):
            feature_id = self.feature_id(definition)
            if self.selected_ids is not None and feature_id not in self.selected_ids:
                continue
            groups.append(FeatureGroup(feature_id, 0, 1.0))
            columns.append(np.asarray(self.values[:, index], dtype=np.float32))
        if columns:
            yield RawCandidateBatch(groups, np.column_stack(columns))

    def quantized_batches(
        self,
        train: np.ndarray | None = None,
        bins: int = 4,
        sample_rows: int | None = None,
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, sample_rows=sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class RepresentativeCrossAssetBatchProvider:
    """Stream the source-supported replicated 31k cross-market inventory."""

    def __init__(
        self,
        *,
        batch_size: int = 256,
        limit_assets: int | None = None,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
        progress: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.quantile_sample_rows = quantile_sample_rows
        self.progress = progress
        self.rows, self.splits = origin_rows()
        self.axis_manifest = json.loads((AXIS_DIR / "manifest.json").read_text(encoding="utf-8"))
        assets = [
            row for row in self.axis_manifest["assets"]
            if row.get("preferredMarket") and row.get("asset") != "BTC"
        ]
        if self.selected_ids is not None:
            subjects = {feature_id.split("/", 4)[1] for feature_id in self.selected_ids}
            assets = [row for row in assets if safe_id(str(row["asset"])) in subjects]
        self.assets = assets[:limit_assets]
        artifact = json.loads(
            (ROOT / "data/benchmarks/binance-cross-asset-component-feature-bases.json").read_text(
                encoding="utf-8"
            )
        )
        allowed_assets = {str(row["asset"]) for row in self.assets}
        self.catalog = {
            str(row["id"]): row
            for row in artifact["featureCatalog"]
            if row.get("scope") == "replicated cross-market inventory"
            and str(row["asset"]) in allowed_assets
        }

    @staticmethod
    def canonical_feature_id(row: dict) -> str:
        subject = infer_subject(row)
        feature = str(row["feature"])
        return coordinate_id(
            subject,
            "asset",
            representative_venue(row, subject),
            representative_cadence(feature, str(row.get("lookback", ""))),
            normalize_representative_formula(feature, subject, str(row.get("scope", ""))),
        )

    @property
    def coordinate_count(self) -> int:
        if self.selected_ids is None:
            return len(self.catalog)
        return sum(
            self.canonical_feature_id(row) in self.selected_ids
            for row in self.catalog.values()
        )

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        from analyze_binance_cross_asset_feature_bases import derive_asset_features, load_asset_sources

        groups: list[FeatureGroup] = []
        columns: list[np.ndarray] = []
        started = time.perf_counter()
        for asset_index, asset in enumerate(self.assets):
            derived = derive_asset_features(asset, load_asset_sources(AXIS_DIR, asset))
            for original_id, candidate in derived.items():
                row = self.catalog.get(original_id)
                if row is None:
                    continue
                feature_id = self.canonical_feature_id(row)
                if self.selected_ids is not None and feature_id not in self.selected_ids:
                    continue
                groups.append(FeatureGroup(feature_id, 0, 1.0))
                columns.append(np.asarray(candidate["values"][self.rows], dtype=np.float32))
                if len(columns) >= self.batch_size:
                    yield RawCandidateBatch(groups, np.column_stack(columns))
                    groups, columns = [], []
            if self.progress and ((asset_index + 1) % 10 == 0 or asset_index + 1 == len(self.assets)):
                print(
                    f"Representative generation {asset_index + 1}/{len(self.assets)} assets "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
        if columns:
            yield RawCandidateBatch(groups, np.column_stack(columns))

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


def completed_bucket_log_volume(volume: np.ndarray, window: int) -> np.ndarray:
    output = np.full(volume.shape, np.nan, dtype=np.float64)
    for end in range(window - 1, volume.size, window):
        value = np.log1p(np.sum(volume[end - window + 1:end + 1]))
        next_end = min(end + window, volume.size)
        output[end:next_end] = value
    return output


class LongUniqueMinuteBatchProvider:
    """Unique long-endogenous coordinates not already covered by dense/representative."""

    ASSET_FORMULAS = (
        "return-2m",
        "realized-volatility-2m",
        "log-volume-1m",
        "relative-log-volume-32m",
        "completed-5m-log-volume",
        "completed-15m-log-volume",
        "completed-60m-log-volume",
    )

    def __init__(
        self,
        *,
        batch_size: int = 256,
        limit_assets: int | None = None,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
        progress: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.quantile_sample_rows = quantile_sample_rows
        self.progress = progress
        assets = preferred_assets()
        if self.selected_ids is not None:
            subjects = {feature_id.split("/", 4)[1] for feature_id in self.selected_ids}
            assets = [row for row in assets if safe_id(str(row["asset"])) in subjects]
        self.assets = assets[:limit_assets]
        self.rows, self.splits = origin_rows()
        self.axis_manifest = json.loads((AXIS_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.axis_rows = int(self.axis_manifest["window"]["rows"])

    @property
    def coordinate_count(self) -> int:
        ids = [
            coordinate_id(asset["asset"], "asset", "binance-preferred", "1m", formula)
            for asset in self.assets for formula in self.ASSET_FORMULAS
        ] + [
            coordinate_id("GLOBAL", "general", "calendar", "known", formula)
            for formula in ("utc-hour-sin", "utc-hour-cos")
        ]
        return sum(self.selected_ids is None or feature_id in self.selected_ids for feature_id in ids)

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        groups: list[FeatureGroup] = []
        columns: list[np.ndarray] = []

        def add(feature_id: str, values: np.ndarray):
            nonlocal groups, columns
            if self.selected_ids is not None and feature_id not in self.selected_ids:
                return None
            groups.append(FeatureGroup(feature_id, 0, 1.0))
            columns.append(np.asarray(values[self.rows], dtype=np.float32))
            if len(columns) >= self.batch_size:
                batch = RawCandidateBatch(groups, np.column_stack(columns))
                groups, columns = [], []
                return batch
            return None

        started = time.perf_counter()
        for asset_index, asset in enumerate(self.assets):
            market = asset["selectedMarket"]
            raw = np.memmap(
                ROOT / market["file"],
                dtype="<f4",
                mode="r",
                shape=(self.axis_rows, SOURCE_COLUMNS),
            )
            observed = np.asarray(raw[:, 9] == 1)
            close = causal_fill(np.asarray(raw[:, 3]), observed)
            volume = np.where(observed, np.asarray(raw[:, 4], dtype=np.float64), 0.0)
            returns = np.diff(np.log(close), prepend=np.log(close[0])) * 10_000
            return_2m = np.full(close.shape, np.nan, dtype=np.float64)
            return_2m[2:] = np.log(close[2:] / close[:-2]) * 10_000
            volatility_2m = np.full(close.shape, np.nan, dtype=np.float64)
            volatility_2m[1:] = np.sqrt(returns[1:] ** 2 + returns[:-1] ** 2)
            log_volume = np.log1p(volume)
            relative_volume = log_volume - exponential_filter(log_volume, 2.0 / 33.0, float(log_volume[0]))
            values = {
                "return-2m": return_2m,
                "realized-volatility-2m": volatility_2m,
                "log-volume-1m": log_volume,
                "relative-log-volume-32m": relative_volume,
                "completed-5m-log-volume": completed_bucket_log_volume(volume, 5),
                "completed-15m-log-volume": completed_bucket_log_volume(volume, 15),
                "completed-60m-log-volume": completed_bucket_log_volume(volume, 60),
            }
            for formula in self.ASSET_FORMULAS:
                batch = add(
                    coordinate_id(asset["asset"], "asset", "binance-preferred", "1m", formula),
                    values[formula],
                )
                if batch is not None:
                    yield batch
            if self.progress and ((asset_index + 1) % 25 == 0 or asset_index + 1 == len(self.assets)):
                print(
                    f"Long unique generation {asset_index + 1}/{len(self.assets)} assets "
                    f"({time.perf_counter() - started:.1f}s)", flush=True,
                )
        # Calendar values are global and emitted exactly once.
        base = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))["datasets"][0]
        times = np.asarray(np.memmap(
            BASE_DIR / base["files"]["times"], dtype="<f8", mode="r", shape=(int(base["rows"]),)
        ))
        phase = 2 * np.pi * np.mod(times, 86_400_000) / 86_400_000
        for formula, values in (("utc-hour-sin", np.sin(phase)), ("utc-hour-cos", np.cos(phase))):
            feature_id = coordinate_id("GLOBAL", "general", "calendar", "known", formula)
            if self.selected_ids is None or feature_id in self.selected_ids:
                groups.append(FeatureGroup(feature_id, 0, 1.0))
                columns.append(values.astype(np.float32))
        if columns:
            yield RawCandidateBatch(groups, np.column_stack(columns))

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class SpectralMinuteBatchProvider:
    """All 102 causal 1-minute spectral/wavelet coordinates per market."""

    def __init__(
        self,
        *,
        batch_size: int = 256,
        limit_assets: int | None = None,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
        transform_batch_rows: int = 1_024,
        progress: bool = False,
    ) -> None:
        from analyze_fourier_return_features import spectral_definitions

        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        assets = preferred_assets()
        if self.selected_ids is not None:
            subjects = {feature_id.split("/", 4)[1] for feature_id in self.selected_ids}
            assets = [row for row in assets if safe_id(str(row["asset"])) in subjects]
        self.assets = assets[:limit_assets]
        self.quantile_sample_rows = quantile_sample_rows
        self.transform_batch_rows = transform_batch_rows
        self.progress = progress
        self.rows, self.splits = origin_rows()
        self.definitions = spectral_definitions("m")

    @property
    def coordinate_count(self) -> int:
        ids = [
            coordinate_id(asset["asset"], "asset", "binance-preferred", "1m", row["id"])
            for asset in self.assets for row in self.definitions
        ]
        return sum(self.selected_ids is None or feature_id in self.selected_ids for feature_id in ids)

    def asset_values(self, asset: dict, selected_indices: list[int] | None = None) -> np.ndarray:
        from analyze_fourier_return_features import spectral_features

        close, _ = load_close(asset)
        returns = np.diff(np.log(close), prepend=np.log(close[0])) * 10_000
        if selected_indices is None:
            selected_indices = list(range(len(self.definitions)))
        selected_windows = {
            int(self.definitions[index]["windowSamples"]) for index in selected_indices
        }
        output = np.full((self.rows.size, len(self.definitions)), np.nan, dtype=np.float32)
        for window in sorted(selected_windows):
            chunks = []
            offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
            for start in range(0, self.rows.size, self.transform_batch_rows):
                selected_rows = self.rows[start:start + self.transform_batch_rows]
                indices = selected_rows[:, None] - offsets[None, :]
                valid = indices[:, 0] >= 0
                values = np.zeros((selected_rows.size, 34), dtype=np.float64)
                if np.any(valid):
                    values[valid] = spectral_features(returns[indices[valid]])
                chunks.append(values.astype(np.float32))
            definition_indices = [
                index for index, row in enumerate(self.definitions)
                if int(row["windowSamples"]) == window
            ]
            output[:, definition_indices] = np.vstack(chunks)
        return output

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        groups: list[FeatureGroup] = []
        columns: list[np.ndarray] = []
        started = time.perf_counter()
        for asset_index, asset in enumerate(self.assets):
            selected_indices = [
                index for index, definition in enumerate(self.definitions)
                if self.selected_ids is None or coordinate_id(
                    asset["asset"], "asset", "binance-preferred", "1m", definition["id"]
                ) in self.selected_ids
            ]
            if not selected_indices:
                continue
            values = self.asset_values(asset, selected_indices)
            for index in selected_indices:
                definition = self.definitions[index]
                feature_id = coordinate_id(
                    asset["asset"], "asset", "binance-preferred", "1m", definition["id"]
                )
                groups.append(FeatureGroup(feature_id, 0, 1.0))
                columns.append(values[:, index])
                if len(columns) >= self.batch_size:
                    yield RawCandidateBatch(groups, np.column_stack(columns))
                    groups, columns = [], []
            if self.progress and ((asset_index + 1) % 5 == 0 or asset_index + 1 == len(self.assets)):
                print(
                    f"Spectral minute generation {asset_index + 1}/{len(self.assets)} assets "
                    f"({time.perf_counter() - started:.1f}s)", flush=True,
                )
        if columns:
            yield RawCandidateBatch(groups, np.column_stack(columns))

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class PublicExternalBatchProvider:
    """The 675 aligned public macro, derivative, flow, and network candidates."""

    CACHE = ROOT / "data/runtime-cache/public-external-recent-30d"

    def __init__(
        self,
        *,
        batch_size: int = 256,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
    ) -> None:
        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.quantile_sample_rows = quantile_sample_rows
        self.manifest = json.loads((self.CACHE / "manifest.json").read_text(encoding="utf-8"))
        self.rows = int(self.manifest["rows"])
        self.values = np.memmap(
            self.CACHE / self.manifest["file"],
            dtype=self.manifest["dtype"],
            mode="r",
            shape=(self.rows, int(self.manifest["columns"])),
        )
        _, self.splits = origin_rows()

    @staticmethod
    def canonical_feature_id(definition: dict) -> str:
        feature_id = str(definition["id"])
        source = str(definition["source"])
        if source == "binance-funding":
            return coordinate_id("BTC", "asset", "binance-usdm", "funding-event", feature_id)
        subject, subject_kind, venue = public_subject(feature_id, source)
        return coordinate_id(
            subject,
            subject_kind,
            venue,
            "1m" if source == "cross-market" else "slow",
            normalize_cross_public(feature_id, subject),
        )

    @property
    def coordinate_count(self) -> int:
        return sum(
            self.selected_ids is None or self.canonical_feature_id(row) in self.selected_ids
            for row in self.manifest["features"]
        )

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        for start in range(0, len(self.manifest["features"]), self.batch_size):
            definitions = self.manifest["features"][start:start + self.batch_size]
            keep = []
            groups = []
            for offset, definition in enumerate(definitions):
                feature_id = self.canonical_feature_id(definition)
                if self.selected_ids is not None and feature_id not in self.selected_ids:
                    continue
                keep.append(start + offset)
                groups.append(FeatureGroup(feature_id, 0, 1.0))
            if groups:
                yield RawCandidateBatch(groups, np.asarray(self.values[:, keep], dtype=np.float32))

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class FundingGridBatchProvider:
    """All declared settled-funding transforms for every eligible USD-M asset."""

    def __init__(
        self,
        *,
        batch_size: int = 256,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
    ) -> None:
        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.quantile_sample_rows = quantile_sample_rows
        self.manifest = json.loads((AXIS_DIR / "derivatives-manifest.json").read_text(encoding="utf-8"))
        assets = [row for row in self.manifest["assets"] if row.get("funding")]
        if self.selected_ids is not None:
            subjects = {feature_id.split("/", 4)[1] for feature_id in self.selected_ids}
            assets = [row for row in assets if safe_id(str(row["asset"])) in subjects]
        self.assets = assets
        base = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))["datasets"][0]
        self.times = np.asarray(np.memmap(
            BASE_DIR / base["files"]["times"], dtype="<f8", mode="r", shape=(int(base["rows"]),)
        ))
        _, self.splits = origin_rows()
        self.formulas = ("funding-level", "funding-absolute-level", "funding-age") + tuple(
            f"funding-{kind}-{settlements}"
            for settlements in (1, 3, 9, 21, 90)
            for kind in ("change", "mean", "absolute-mean")
        )

    @staticmethod
    def required_events(formula: str) -> int:
        tail = formula.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 1

    @property
    def coordinate_count(self) -> int:
        return sum(
            int(asset["funding"]["events"]) >= self.required_events(formula)
            and (
                self.selected_ids is None
                or coordinate_id(asset["asset"], "asset", "binance-usdm", "funding-event", formula)
                in self.selected_ids
            )
            for asset in self.assets for formula in self.formulas
        )

    def asset_values(self, asset: dict) -> dict[str, np.ndarray]:
        artifact = json.loads((ROOT / asset["funding"]["file"]).read_text(encoding="utf-8"))
        events = artifact["events"]
        event_times = np.asarray([float(row["time"]) + 60_000 for row in events])
        rates = np.asarray([float(row["rate"]) for row in events], dtype=np.float64)
        latest = np.searchsorted(event_times, self.times, side="right") - 1
        valid = latest >= 0
        output = {
            "funding-level": np.full(self.times.shape, np.nan, dtype=np.float64),
            "funding-absolute-level": np.full(self.times.shape, np.nan, dtype=np.float64),
            "funding-age": np.full(self.times.shape, np.nan, dtype=np.float64),
        }
        output["funding-level"][valid] = rates[latest[valid]]
        output["funding-absolute-level"][valid] = np.abs(rates[latest[valid]])
        output["funding-age"][valid] = (self.times[valid] - event_times[latest[valid]]) / 3_600_000
        prefix = np.concatenate(([0.0], np.cumsum(rates)))
        absolute_prefix = np.concatenate(([0.0], np.cumsum(np.abs(rates))))
        for settlements in (1, 3, 9, 21, 90):
            change = np.full(self.times.shape, np.nan, dtype=np.float64)
            enough_change = latest >= settlements
            change[enough_change] = rates[latest[enough_change]] - rates[latest[enough_change] - settlements]
            mean = np.full(self.times.shape, np.nan, dtype=np.float64)
            absolute_mean = np.full(self.times.shape, np.nan, dtype=np.float64)
            enough_mean = latest >= settlements - 1
            end = latest[enough_mean] + 1
            start = end - settlements
            mean[enough_mean] = (prefix[end] - prefix[start]) / settlements
            absolute_mean[enough_mean] = (absolute_prefix[end] - absolute_prefix[start]) / settlements
            output[f"funding-change-{settlements}"] = change
            output[f"funding-mean-{settlements}"] = mean
            output[f"funding-absolute-mean-{settlements}"] = absolute_mean
        return output

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        groups: list[FeatureGroup] = []
        columns: list[np.ndarray] = []
        for asset in self.assets:
            values = self.asset_values(asset)
            events = int(asset["funding"]["events"])
            for formula in self.formulas:
                if events < self.required_events(formula):
                    continue
                feature_id = coordinate_id(
                    asset["asset"], "asset", "binance-usdm", "funding-event", formula
                )
                if self.selected_ids is not None and feature_id not in self.selected_ids:
                    continue
                groups.append(FeatureGroup(feature_id, 0, 1.0))
                columns.append(values[formula].astype(np.float32))
                if len(columns) >= self.batch_size:
                    yield RawCandidateBatch(groups, np.column_stack(columns))
                    groups, columns = [], []
        if columns:
            yield RawCandidateBatch(groups, np.column_stack(columns))

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class OneSecondTechnicalBatchProvider:
    """All 166 archived 1s RSI/EMA/MACD coordinates per eligible spot asset."""

    CACHE = ROOT / "data/runtime-cache/binance-cross-asset-spot-1s-close-30d"

    def __init__(
        self,
        *,
        batch_size: int = 256,
        limit_assets: int | None = None,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
        progress: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.quantile_sample_rows = quantile_sample_rows
        self.progress = progress
        self.rows, self.splits = origin_rows()
        self.sample_seconds = self.rows * 60 + 59
        manifest_path = self.CACHE / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing 1s close cache {manifest_path}; run ml/backfill_global_spot_1s_close.py"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = [row for row in self.manifest["assets"] if float(row["coverage"]) >= 0.95]
        if self.selected_ids is not None:
            subjects = {feature_id.split("/", 4)[1] for feature_id in self.selected_ids}
            assets = [row for row in assets if safe_id(str(row["asset"])) in subjects]
        self.assets = assets[:limit_assets]
        audit = json.loads(
            (ROOT / "data/benchmarks/technical-indicator-predictiveness.json").read_text(encoding="utf-8")
        )
        self.definitions = audit["signals"]

    def feature_id(self, asset: str, definition: dict) -> str:
        return coordinate_id(
            asset, "asset", "binance-spot", "1s", normalize_second_formula(definition)
        )

    @property
    def coordinate_count(self) -> int:
        return sum(
            self.selected_ids is None or self.feature_id(str(asset["asset"]), definition) in self.selected_ids
            for asset in self.assets for definition in self.definitions
        )

    def load_asset(self, asset: dict) -> tuple[np.ndarray, np.ndarray]:
        rows = int(asset["rows"])
        close = np.asarray(np.memmap(
            ROOT / asset["closeFile"], dtype="<f4", mode="r", shape=(rows,)
        ), dtype=np.float64)
        observed = np.asarray(np.memmap(
            ROOT / asset["observedFile"], dtype="u1", mode="r", shape=(rows,)
        ) == 1)
        return causal_fill(close, observed), observed

    def asset_values(self, asset: dict, definitions: list[dict]) -> np.ndarray:
        close, observed = self.load_asset(asset)
        sampled_observed = observed[self.sample_seconds]
        output = np.full((self.rows.size, len(definitions)), np.nan, dtype=np.float32)
        changes = np.diff(close, prepend=close[0])
        gains = np.maximum(changes, 0.0)
        losses = np.maximum(-changes, 0.0)
        by_period: dict[int, list[tuple[int, dict]]] = {}
        macd: list[tuple[int, dict]] = []
        for index, definition in enumerate(definitions):
            if str(definition["family"]) == "MACD":
                macd.append((index, definition))
            else:
                by_period.setdefault(int(definition["period"]), []).append((index, definition))
        for period, rows in by_period.items():
            if any(str(row["family"]) == "RSI" for _, row in rows):
                average_gain = exponential_filter(gains, 1.0 / period, 0.0)
                average_loss = exponential_filter(losses, 1.0 / period, 0.0)
                values = np.full(close.shape, 50.0, dtype=np.float64)
                positive_loss = average_loss > 0
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    values[positive_loss] = 100.0 - 100.0 / (
                        1.0 + average_gain[positive_loss] / average_loss[positive_loss]
                    )
                values[(~positive_loss) & (average_gain > 0)] = 100.0
                for index, row in rows:
                    if str(row["family"]) == "RSI":
                        output[:, index] = values[self.sample_seconds]
            ema_rows = [(index, row) for index, row in rows if str(row["family"]) != "RSI"]
            if not ema_rows:
                continue
            ema = exponential_filter(close, 2.0 / (period + 1.0), float(close[0]))
            for index, row in ema_rows:
                family = str(row["family"])
                if family == "EMA":
                    values = 10_000.0 * np.log(close[self.sample_seconds] / ema[self.sample_seconds])
                else:
                    horizon = int(row["horizon"])
                    current = self.sample_seconds
                    lagged = current - horizon
                    slope = 10_000.0 * np.log(ema[current] / ema[lagged]) / horizon
                    if family == "EMA slope":
                        values = slope
                    else:
                        lagged_twice = current - 2 * horizon
                        previous = 10_000.0 * np.log(ema[lagged] / ema[lagged_twice]) / horizon
                        values = slope - previous
                output[:, index] = values
        for index, definition in macd:
            fast = int(definition["fast"])
            slow = int(definition["slow"])
            signal_period = int(definition["signal"])
            fast_ema = exponential_filter(close, 2.0 / (fast + 1.0), float(close[0]))
            slow_ema = exponential_filter(close, 2.0 / (slow + 1.0), float(close[0]))
            line = fast_ema - slow_ema
            if str(definition["kind"]) == "macdLine":
                values = line[self.sample_seconds]
            else:
                signal = exponential_filter(line, 2.0 / (signal_period + 1.0), 0.0)
                values = line[self.sample_seconds] - signal[self.sample_seconds]
            output[:, index] = 10_000.0 * values / close[self.sample_seconds]
        output[~sampled_observed] = np.nan
        return output

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        started = time.perf_counter()
        for asset_index, asset in enumerate(self.assets):
            subject = str(asset["asset"])
            definitions = [
                row for row in self.definitions
                if self.selected_ids is None or self.feature_id(subject, row) in self.selected_ids
            ]
            if definitions:
                values = self.asset_values(asset, definitions)
                for start in range(0, len(definitions), self.batch_size):
                    rows = definitions[start:start + self.batch_size]
                    groups = [FeatureGroup(self.feature_id(subject, row), 0, 1.0) for row in rows]
                    yield RawCandidateBatch(groups, values[:, start:start + self.batch_size])
            if self.progress and ((asset_index + 1) % 5 == 0 or asset_index + 1 == len(self.assets)):
                print(
                    f"Second technical generation {asset_index + 1}/{len(self.assets)} assets "
                    f"({time.perf_counter() - started:.1f}s)", flush=True,
                )

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class SpectralSecondBatchProvider:
    """All 102 causal 1s spectral/wavelet coordinates at aligned minute origins."""

    def __init__(
        self,
        *,
        batch_size: int = 256,
        limit_assets: int | None = None,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
        transform_batch_rows: int = 1_024,
        progress: bool = False,
    ) -> None:
        from analyze_fourier_return_features import spectral_definitions

        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.quantile_sample_rows = quantile_sample_rows
        self.transform_batch_rows = transform_batch_rows
        self.progress = progress
        self.rows, self.splits = origin_rows()
        self.sample_seconds = self.rows * 60 + 59
        manifest_path = OneSecondTechnicalBatchProvider.CACHE / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing 1s close cache {manifest_path}; run ml/backfill_global_spot_1s_close.py"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = [row for row in self.manifest["assets"] if float(row["coverage"]) >= 0.95]
        if self.selected_ids is not None:
            subjects = {feature_id.split("/", 4)[1] for feature_id in self.selected_ids}
            assets = [row for row in assets if safe_id(str(row["asset"])) in subjects]
        self.assets = assets[:limit_assets]
        self.definitions = spectral_definitions("s")

    def feature_id(self, asset: str, definition: dict) -> str:
        return coordinate_id(asset, "asset", "binance-spot", "1s", str(definition["id"]))

    @property
    def coordinate_count(self) -> int:
        return sum(
            self.selected_ids is None or self.feature_id(str(asset["asset"]), row) in self.selected_ids
            for asset in self.assets for row in self.definitions
        )

    def load_returns(self, asset: dict) -> tuple[np.ndarray, np.ndarray]:
        rows = int(asset["rows"])
        close = np.asarray(np.memmap(
            ROOT / asset["closeFile"], dtype="<f4", mode="r", shape=(rows,)
        ), dtype=np.float64)
        observed = np.asarray(np.memmap(
            ROOT / asset["observedFile"], dtype="u1", mode="r", shape=(rows,)
        ) == 1)
        close = causal_fill(close, observed)
        return np.diff(np.log(close), prepend=np.log(close[0])) * 10_000, observed

    def asset_values(self, asset: dict, selected_indices: list[int] | None = None) -> np.ndarray:
        from analyze_fourier_return_features import spectral_features

        returns, observed = self.load_returns(asset)
        if selected_indices is None:
            selected_indices = list(range(len(self.definitions)))
        selected_windows = {
            int(self.definitions[index]["windowSamples"]) for index in selected_indices
        }
        output = np.full((self.sample_seconds.size, len(self.definitions)), np.nan, dtype=np.float32)
        for window in sorted(selected_windows):
            chunks = []
            offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
            for start in range(0, self.sample_seconds.size, self.transform_batch_rows):
                selected = self.sample_seconds[start:start + self.transform_batch_rows]
                indices = selected[:, None] - offsets[None, :]
                values = spectral_features(returns[indices]).astype(np.float32)
                chunks.append(values)
            definition_indices = [
                index for index, row in enumerate(self.definitions)
                if int(row["windowSamples"]) == window
            ]
            output[:, definition_indices] = np.vstack(chunks)
        output[~observed[self.sample_seconds]] = np.nan
        return output

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        started = time.perf_counter()
        for asset_index, asset in enumerate(self.assets):
            subject = str(asset["asset"])
            selected_indices = [
                index for index, row in enumerate(self.definitions)
                if self.selected_ids is None or self.feature_id(subject, row) in self.selected_ids
            ]
            if selected_indices:
                values = self.asset_values(asset, selected_indices)
                for start in range(0, len(selected_indices), self.batch_size):
                    indices = selected_indices[start:start + self.batch_size]
                    groups = [
                        FeatureGroup(self.feature_id(subject, self.definitions[index]), 0, 1.0)
                        for index in indices
                    ]
                    yield RawCandidateBatch(groups, values[:, indices])
            if self.progress and ((asset_index + 1) % 5 == 0 or asset_index + 1 == len(self.assets)):
                print(
                    f"Spectral 1s generation {asset_index + 1}/{len(self.assets)} assets "
                    f"({time.perf_counter() - started:.1f}s)", flush=True,
                )

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)


class PredictionMarketBatchProvider:
    """Fixed-width causal summaries of dynamic Kalshi asset and global-event sets."""

    AXIS = ROOT / "data/runtime-cache/prediction-market-candidate-axis-30d"

    def __init__(
        self,
        *,
        batch_size: int = 256,
        selected_ids: set[str] | None = None,
        quantile_sample_rows: int | None = 4_096,
    ) -> None:
        self.batch_size = batch_size
        self.selected_ids = None if selected_ids is None else set(selected_ids)
        self.quantile_sample_rows = quantile_sample_rows
        self.manifest = json.loads((self.AXIS / "manifest.json").read_text(encoding="utf-8"))
        self.rows, self.splits = origin_rows()
        if int(self.manifest["rows"]) != self.rows.size:
            raise ValueError("Prediction-market axis does not align with global feature origins")
        self.coordinates = [
            row for row in self.manifest["coordinates"]
            if self.selected_ids is None or str(row["id"]) in self.selected_ids
        ]
        all_indices = {str(row["id"]): index for index, row in enumerate(self.manifest["coordinates"])}
        self.indices = [all_indices[str(row["id"])] for row in self.coordinates]

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinates)

    def raw_batches(self) -> Iterator[RawCandidateBatch]:
        raw = np.memmap(
            self.AXIS / self.manifest["file"], dtype=self.manifest["dtype"], mode="r",
            shape=(int(self.manifest["rows"]), int(self.manifest["columns"])),
        )
        for start in range(0, len(self.coordinates), self.batch_size):
            rows = self.coordinates[start:start + self.batch_size]
            indices = self.indices[start:start + self.batch_size]
            yield RawCandidateBatch(
                [FeatureGroup(str(row["id"]), 0, 1.0) for row in rows],
                np.asarray(raw[:, indices], dtype=np.float32),
            )

    def quantized_batches(
        self, train: np.ndarray | None = None, bins: int = 4
    ) -> Iterator[QuantizedCandidateBatch]:
        train_mask = self.splits == 0 if train is None else np.asarray(train, dtype=bool)
        for batch in self.raw_batches():
            states, edges, arities = quantize_batch_fast(
                batch.values, train_mask, bins, self.quantile_sample_rows
            )
            groups = [
                FeatureGroup(group.id, arity, group.penalty_weight)
                for group, arity in zip(batch.groups, arities)
            ]
            yield QuantizedCandidateBatch(groups, states, edges)
