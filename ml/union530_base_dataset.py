from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from differentiable_union530_features import (
    Global471Base,
    Production59Base,
    production59_features,
    reconstruct_global470,
    spread_from_top_of_book,
)
from global_feature_registry import ROOT
from trading_storage import (
    read_candle_column,
    read_derivatives_kline_columns,
    read_trade_flow_columns,
)


BASE_CONTRACT = "union530-base-history-differentiable-feature-graph-v1"
MINUTE_COLUMNS = 10
METRICS_COLUMNS = 7
BOOK_COLUMNS = 25
GLOBAL_FEATURE_BASIS = ROOT / "data/runtime-cache/global-feature-basis-30d"
FUTURES_COLUMNS = (
    "open", "high", "low", "close", "baseVolume", "quoteVolume",
    "tradeCount", "takerBuyBaseVolume", "takerBuyQuoteVolume",
)
TRADE_FLOW_COLUMNS = (
    "aggressiveBuyBaseVolume",
    "aggressiveSellBaseVolume",
    "aggressiveBuyQuoteVolume",
    "aggressiveSellQuoteVolume",
    "aggressiveBuyAggregateQuantitySquared",
    "aggressiveSellAggregateQuantitySquared",
    "aggressiveBuyMaxAggregateQuantity",
    "aggressiveSellMaxAggregateQuantity",
    "aggressiveBuyBaseVolumeTimeMoment",
    "aggressiveSellBaseVolumeTimeMoment",
    "aggressiveBuyAggregateTradeCount",
    "aggressiveSellAggregateTradeCount",
    "aggressiveBuyTradeCount",
    "aggressiveSellTradeCount",
    "aggressorSideFlipCount",
    "firstAggressorSide",
    "lastAggressorSide",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _dates(start_ms: int, end_ms: int) -> list[str]:
    start = datetime.fromtimestamp(start_ms / 1_000, tz=timezone.utc).date()
    end = datetime.fromtimestamp(end_ms / 1_000, tz=timezone.utc).date()
    return [
        (start + timedelta(days=index)).isoformat()
        for index in range((end - start).days)
    ]


def _candle_history(
    directory: Path,
    dates: Iterable[str],
    columns: tuple[str, ...],
) -> np.ndarray:
    days = []
    for day in dates:
        reference = directory / f"{day}.json"
        days.append(np.column_stack([
            read_candle_column(reference, column) for column in columns
        ]))
    return np.concatenate(days).astype(np.float64, copy=False)


def _futures_history(directory: Path, dates: Iterable[str]) -> np.ndarray:
    days = []
    for day in dates:
        values, valid = read_derivatives_kline_columns(directory / f"{day}.json")
        if not bool(valid.all()):
            raise ValueError(f"futures minute base is incomplete on {day}")
        days.append(np.column_stack([values[name] for name in FUTURES_COLUMNS]))
    return np.concatenate(days).astype(np.float64, copy=False)


def _trade_history(directory: Path, dates: Iterable[str]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {name: [] for name in TRADE_FLOW_COLUMNS}
    for day in dates:
        values = read_trade_flow_columns(directory / f"{day}.json", TRADE_FLOW_COLUMNS)
        for name in TRADE_FLOW_COLUMNS:
            output[name].append(np.asarray(values[name]))
    return {
        name: np.concatenate(parts).astype(np.float64, copy=False)
        for name, parts in output.items()
    }


@dataclass(frozen=True)
class ExampleSplit:
    physical_rows: np.ndarray
    targets: np.ndarray
    origins: np.ndarray

    @property
    def count(self) -> int:
        return int(self.physical_rows.size)


@dataclass(frozen=True)
class DerivedFeatureSplit:
    features: object
    targets: np.ndarray
    times: np.ndarray
    physical_rows: np.ndarray

    @property
    def count(self) -> int:
        return int(self.physical_rows.size)


class DerivedUnionFeatureView:
    """Assemble current 530-channel rows without storing repeated snapshots."""

    def __init__(
        self,
        physical_rows: np.ndarray,
        production: np.ndarray,
        global470: np.ndarray,
        minute_source_rows: np.ndarray,
        top_of_book: np.ndarray,
    ) -> None:
        self.physical_rows = np.asarray(physical_rows, dtype=np.int64)
        self.production = np.asarray(production, dtype=np.float32)
        self.global470 = np.asarray(global470, dtype=np.float32)
        self.minute_source_rows = minute_source_rows
        self.top_of_book = top_of_book
        self.shape = (self.physical_rows.size, 530)

    def __getitem__(self, selected):
        scalar = isinstance(selected, (int, np.integer))
        if isinstance(selected, slice):
            logical = np.arange(*selected.indices(self.shape[0]), dtype=np.int64)
        else:
            logical = np.asarray(selected, dtype=np.int64)
            if scalar:
                logical = logical.reshape(1)
        physical = self.physical_rows[logical]
        output = np.empty((logical.size, 530), dtype=np.float32)
        output[:, :59] = self.production[logical]
        output[:, 59:529] = self.global470[
            np.asarray(self.minute_source_rows[physical], dtype=np.int64)
        ]
        top = np.asarray(self.top_of_book[physical], dtype=np.float64)
        output[:, 529] = (
            10_000.0 * (top[:, 1] - top[:, 0])
            / ((top[:, 1] + top[:, 0]) / 2.0)
        ).astype(np.float32)
        return output[0] if scalar else output


class Production59BaseHistoryDataset:
    """The five primitive source histories needed by the production59 graph.

    A union530 source manifest can supply the same histories without opening
    its unrelated global-feature matrices or example population.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.manifest = json.loads((self.directory / "dataset.json").read_text(encoding="utf-8"))
        contract = self.manifest.get("contract")
        if contract not in ("production59-base-history-v1", BASE_CONTRACT):
            raise ValueError("unexpected production59 source-history contract")
        history = self.manifest["baseHistory"]
        self.second_start_ms = int(history["secondStartMs"])
        self.second_end_ms = int(history["secondEndExclusiveMs"])
        self.second_rows = int(history["secondRows"])
        self.minute_rows = int(history["minuteRows"])
        duration = self.second_end_ms - self.second_start_ms
        if duration <= 0 or self.second_start_ms % 86_400_000 or self.second_end_ms % 86_400_000 \
                or self.second_rows * 1_000 != duration or self.minute_rows * 60_000 != duration:
            raise ValueError("source history must contain complete consecutive UTC days")
        self._numpy_cache: dict[str, Any] = {}
        if contract == "production59-base-history-v1":
            self.sources = self.manifest["sources"]
            if set(self.sources) != {"btcSecond", "tradeFlow", "futuresMinute", "btcMinute", "ethMinute"}:
                raise ValueError("production59 source catalog changed")
            references = self.manifest["sourceReferences"]
            expected = {f"{source}/{day}.json" for source in self.sources.values() for day in self._days()}
            if len(references) != len(expected) or {row["file"] for row in references} != expected:
                raise ValueError("production59 source references do not cover the history")
            for row in references:
                raw = _resolve(row["file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError(f"production59 source reference changed: {row['file']}")
                reference = json.loads(raw)
                axis = reference["sequence"]
                day = datetime.strptime(Path(row["file"]).stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                step = 1_000 if row["file"].endswith(f"/1s/{day.date()}.json") else 60_000
                if axis["start"] != int(day.timestamp() * 1_000) or axis["step"] != step \
                        or axis["count"] * step != 86_400_000:
                    raise ValueError(f"production59 source axis is misaligned: {row['file']}")
        else:
            self._sources_from_union()

    def _sources_from_union(self) -> None:
        production = self.manifest["features"]["productionGraph"]
        btc_candle_source = next(
            value for spec in production for value in spec["baseSources"]
            if "candles/spot-btcusdt" in value and value.endswith("/1s")
        )
        flow_source = next(
            value for spec in production for value in spec["baseSources"]
            if "trade-flow/spot-btcusdt" in value
        )
        futures_source = next(
            value for spec in production for value in spec["baseSources"]
            if "derivatives-klines" in value
        )
        btc_minute_source = next(
            value for spec in production for value in spec["baseSources"]
            if value.endswith("/1m") and "spot-btcusdt" in value
            and spec["id"].startswith("futures-")
        )
        eth_minute_source = next(
            value for spec in production for value in spec["baseSources"]
            if value.endswith("/1m") and "spot-ethusdt" in value
            and spec["id"].startswith("eth-")
            and value != btc_minute_source
        )
        self.sources = dict(btcSecond=btc_candle_source, tradeFlow=flow_source,
                            futuresMinute=futures_source, btcMinute=btc_minute_source,
                            ethMinute=eth_minute_source)

    def _days(self) -> list[str]:
        return _dates(self.second_start_ms, self.second_end_ms)

    def candle_history(self, source: str, cadence: str) -> np.ndarray:
        key = f"candle:{source}:{cadence}"
        cached = self._numpy_cache.get(key)
        if cached is None:
            cached = _candle_history(
                _resolve(source), self._days(),
                ("open", "high", "low", "close", "volume"),
            )
            expected = self.second_rows if cadence == "1s" else self.minute_rows
            if cached.shape != (expected, 5):
                raise ValueError(f"{source} base shape changed: {cached.shape}")
            self._numpy_cache[key] = cached
        return cached

    def futures_history(self, source: str) -> np.ndarray:
        key = f"futures:{source}"
        cached = self._numpy_cache.get(key)
        if cached is None:
            cached = _futures_history(_resolve(source), self._days())
            if cached.shape != (self.minute_rows, 9):
                raise ValueError(f"futures base shape changed: {cached.shape}")
            self._numpy_cache[key] = cached
        return cached

    def trade_history(self, source: str) -> dict[str, np.ndarray]:
        key = f"trade:{source}"
        cached = self._numpy_cache.get(key)
        if cached is None:
            cached = _trade_history(_resolve(source), self._days())
            if any(values.shape != (self.second_rows,) for values in cached.values()):
                raise ValueError("trade-flow base shape changed")
            self._numpy_cache[key] = cached
        return cached

    @staticmethod
    def tensor(values: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values).copy(), device=device, dtype=dtype)

    def production_base(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Production59Base:
        btc_candle_source = self.sources["btcSecond"]
        flow_source = self.sources["tradeFlow"]
        futures_source = self.sources["futuresMinute"]
        btc_minute_source = self.sources["btcMinute"]
        eth_minute_source = self.sources["ethMinute"]
        candles = self.candle_history(btc_candle_source, "1s")
        flow = self.trade_history(flow_source)
        btc_minute = self.candle_history(btc_minute_source, "1m")
        eth_minute = self.candle_history(eth_minute_source, "1m")
        futures = self.futures_history(futures_source)
        return Production59Base(
            btc_second_candles=self.tensor(candles, device, dtype),
            btc_trade_flow={
                name: self.tensor(values, device, dtype) for name, values in flow.items()
            },
            btc_minute_candles=self.tensor(btc_minute, device, dtype),
            eth_minute_candles=self.tensor(eth_minute, device, dtype),
            btc_futures_minute=self.tensor(futures, device, dtype),
            start_ms=self.second_start_ms,
        )


class Union530BaseHistoryDataset(Production59BaseHistoryDataset):
    """Base-only histories plus aligned example indices for the 530 graph."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        if self.manifest.get("contract") != BASE_CONTRACT:
            raise ValueError("unexpected union530 base-history contract")
        examples = self.manifest["examples"]
        self.rows = int(examples["rows"])
        self.origins = self._memmap(examples["origins"], "<f8", (self.rows,))
        self.targets = self._memmap(examples["targets"], "<f4", (self.rows,))
        self.minute_source_rows = self._memmap(
            examples["minuteSourceRows"], "<u4", (self.rows,)
        )
        self.split_codes = self._memmap(examples["splits"], "u1", (self.rows,))
        self.nonzero = self._memmap(examples["nonzero"], "u1", (self.rows,))
        self.top_of_book = self._memmap(
            examples["topOfBook"], "<f8", (self.rows, 2)
        )
        self.feature_ids = tuple(self.manifest["features"]["ids"])
        self.production_specs = tuple(self.manifest["features"]["productionGraph"])
        self.global_specs = tuple(self.manifest["features"]["globalGraph"])
        if len(self.feature_ids) != 530 or len(self.production_specs) != 59 \
                or len(self.global_specs) != 471:
            raise ValueError("union530 base-history feature catalog changed")

    def _memmap(self, spec: Mapping[str, Any], dtype: str, shape: tuple[int, ...]) -> np.memmap:
        path = _resolve(str(spec["file"]))
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def split(self, name: str, limit: int | None = None) -> ExampleSplit:
        code = {"train": 0, "validation": 1, "test": 2}[name]
        physical = np.flatnonzero(
            (np.asarray(self.split_codes) == code)
            & (np.asarray(self.nonzero) != 0)
        )
        if limit is not None:
            if limit < 1 or limit > physical.size:
                raise ValueError(f"{name} requested {limit:,} of {physical.size:,} examples")
            physical = physical[:limit]
        return ExampleSplit(
            physical_rows=physical,
            targets=np.asarray(self.targets[physical], dtype=np.float32),
            origins=np.asarray(self.origins[physical], dtype=np.float64),
        )

    def second_indices(self, physical_rows: np.ndarray) -> np.ndarray:
        indices = (
            (np.asarray(self.origins[physical_rows], dtype=np.int64) - self.second_start_ms)
            // 1_000
        )
        if np.any(indices < 0) or np.any(indices >= self.second_rows):
            raise ValueError("example origins escape the second base axis")
        return indices

    def raw_matrix(self, source: str, columns: int, rows: int) -> np.memmap:
        key = f"matrix:{source}:{columns}:{rows}"
        cached = self._numpy_cache.get(key)
        if cached is None:
            cached = np.memmap(
                _resolve(source), dtype="<f4", mode="r", shape=(rows, columns)
            )
            self._numpy_cache[key] = cached
        return cached

    def second_close(self, source: str) -> np.memmap:
        key = f"close:{source}"
        cached = self._numpy_cache.get(key)
        if cached is None:
            cached = np.memmap(
                _resolve(source), dtype="<f4", mode="r", shape=(self.second_rows,)
            )
            self._numpy_cache[key] = cached
        return cached

    def observed_timeline(self, source: str) -> np.memmap:
        key = f"observed:{source}"
        cached = self._numpy_cache.get(key)
        if cached is None:
            cached = np.memmap(
                _resolve(source), dtype="u1", mode="r", shape=(self.second_rows,)
            )
            self._numpy_cache[key] = cached
        return cached

    def funding_events(self, source: str) -> tuple[np.ndarray, np.ndarray]:
        key = f"funding:{source}"
        cached = self._numpy_cache.get(key)
        if cached is None:
            artifact = json.loads(_resolve(source).read_text(encoding="utf-8"))
            cached = (
                np.asarray([row["time"] for row in artifact["events"]], dtype=np.int64),
                np.asarray([row["rate"] for row in artifact["events"]], dtype=np.float32),
            )
            self._numpy_cache[key] = cached
        return cached

    def production_base_from_global(self, base: Global471Base) -> Production59Base:
        production = self.production_specs
        btc_second = next(
            value for spec in production for value in spec["baseSources"]
            if "candles/spot-btcusdt" in value and value.endswith("/1s")
        )
        flow = next(
            value for spec in production for value in spec["baseSources"]
            if "/trade-flow/" in value
        )
        futures = next(
            value for spec in production for value in spec["baseSources"]
            if "/derivatives-klines/" in value
        )
        btc_minute = next(
            value for spec in production for value in spec["baseSources"]
            if "candles/spot-btcusdt" in value and value.endswith("/1m")
        )
        eth_minute = next(
            value for spec in production for value in spec["baseSources"]
            if "candles/spot-ethusdt" in value and value.endswith("/1m")
        )
        return Production59Base(
            btc_second_candles=base.tensors[btc_second],
            btc_trade_flow=base.trade_flows[flow],
            btc_minute_candles=base.tensors[btc_minute],
            eth_minute_candles=base.tensors[eth_minute],
            btc_futures_minute=base.tensors[futures],
            start_ms=self.second_start_ms,
        )

    def global_base(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> Global471Base:
        """Load only primitive sources referenced by the selected graph."""
        source_paths: set[str] = set()
        observed_paths: set[str] = set()
        for spec in self.global_specs:
            if isinstance(spec.get("baseSource"), str):
                source_paths.add(str(spec["baseSource"]))
            source_paths.update(str(value) for value in spec.get("baseSources", ()))
            if isinstance(spec.get("observedSource"), str):
                observed_paths.add(str(spec["observedSource"]))
        source_paths -= {"exampleTopOfBook", "implicit-utc-origin-time"}

        tensors: dict[str, torch.Tensor] = {}
        trade_flows: dict[str, Mapping[str, torch.Tensor]] = {}
        funding: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for source in sorted(source_paths):
            if "/trade-flow/" in source:
                trade_flows[source] = {
                    name: self.tensor(values, device, dtype)
                    for name, values in self.trade_history(source).items()
                }
            elif "/derivatives-klines/" in source:
                tensors[source] = self.tensor(
                    self.futures_history(source), device, dtype
                )
            elif "/refs/candles/" in source:
                cadence = "1s" if source.endswith("/1s") else "1m"
                tensors[source] = self.tensor(
                    self.candle_history(source, cadence), device, dtype
                )
            elif source.endswith("usdm-funding.json"):
                times, rates = self.funding_events(source)
                funding[source] = (
                    torch.as_tensor(times.copy(), device=device, dtype=torch.int64),
                    self.tensor(rates, device, dtype),
                )
            elif source.endswith("/close.f32"):
                tensors[source] = self.tensor(self.second_close(source), device, dtype)
            elif source.endswith("eth-minute-last-second-trade-count.f32"):
                tensors[source] = self.tensor(
                    np.memmap(
                        _resolve(source), dtype="<f4", mode="r",
                        shape=(self.minute_rows,),
                    ),
                    device,
                    dtype,
                )
            elif source.endswith("usdm-metrics.f32"):
                tensors[source] = self.tensor(
                    self.raw_matrix(
                        source, METRICS_COLUMNS, self.minute_rows // 5
                    ),
                    device,
                    dtype,
                )
            elif source.endswith("usdm-book-depth.f32"):
                tensors[source] = self.tensor(
                    self.raw_matrix(source, BOOK_COLUMNS, self.minute_rows),
                    device,
                    dtype,
                )
            elif source.endswith(".f32"):
                tensors[source] = self.tensor(
                    self.raw_matrix(source, MINUTE_COLUMNS, self.minute_rows),
                    device,
                    dtype,
                )
            else:
                raise ValueError(f"unknown global primitive source type: {source}")

        observed = {
            source: torch.as_tensor(
                np.asarray(self.observed_timeline(source), dtype=np.uint8).copy(),
                device=device,
                dtype=torch.uint8,
            )
            for source in sorted(observed_paths)
        }
        basis_manifest = json.loads(
            (GLOBAL_FEATURE_BASIS / "manifest.json").read_text(encoding="utf-8")
        )
        basis = basis_manifest["datasets"][0]
        count = int(basis["rows"])
        times = np.memmap(
            GLOBAL_FEATURE_BASIS / basis["files"]["times"],
            dtype="<f8", mode="r", shape=(count,),
        )
        minute_rows = np.floor(
            (np.asarray(times, dtype=np.float64) - self.second_start_ms) / 60_000
        ).astype(np.int64)
        if np.any(minute_rows < 0) or np.any(minute_rows >= self.minute_rows):
            raise ValueError("global working origins escape the primitive minute axis")
        return Global471Base(
            tensors=tensors,
            observed=observed,
            trade_flows=trade_flows,
            funding_events=funding,
            minute_origin_rows=torch.as_tensor(
                minute_rows.copy(), device=device, dtype=torch.long
            ),
            minute_origin_times_ms=torch.as_tensor(
                np.asarray(times, dtype=np.int64).copy(),
                device=device,
                dtype=torch.int64,
            ),
            minute_rows=self.minute_rows,
            second_rows=self.second_rows,
        )


class DifferentiableUnion530Dataset:
    """Runtime-derived clean union backed only by primitive source histories."""

    def __init__(
        self,
        directory: Path,
        *,
        examples_by_split: Mapping[str, int],
        return_count: int = 1,
    ) -> None:
        self.base = Union530BaseHistoryDataset(directory)
        self.feature_count = 530
        self.return_count = int(return_count)
        if self.return_count < 1:
            raise ValueError("return count must be positive")
        physical_by_split = {
            name: self.base.split(name).physical_rows
            for name in ("train", "validation", "test")
        }
        all_physical = np.concatenate(tuple(physical_by_split.values()))

        production_base = self.base.production_base(
            torch.device("cpu"), torch.float64
        )
        production = production59_features(
            production_base,
            torch.as_tensor(
                self.base.second_indices(all_physical), dtype=torch.long
            ),
        ).detach().float().numpy()
        del production_base
        gc.collect()

        global_base = self.base.global_base(torch.device("cpu"), torch.float64)
        global470 = reconstruct_global470(
            global_base, self.base.global_specs
        ).detach().float().numpy()
        del global_base
        self.base._numpy_cache.clear()
        gc.collect()

        self.all_splits: dict[str, DerivedFeatureSplit] = {}
        self.splits: dict[str, DerivedFeatureSplit] = {}
        offset = 0
        for name in ("train", "validation", "test"):
            physical = physical_by_split[name]
            count = int(physical.size)
            split_production = production[offset:offset + count]
            offset += count
            view = DerivedUnionFeatureView(
                physical,
                split_production,
                global470,
                self.base.minute_source_rows,
                self.base.top_of_book,
            )
            full = DerivedFeatureSplit(
                features=view,
                targets=np.asarray(self.base.targets[physical], dtype=np.float32),
                times=np.asarray(self.base.origins[physical], dtype=np.float64),
                physical_rows=physical,
            )
            self.all_splits[name] = full
            available_paths = count - self.return_count + 1
            limit = int(examples_by_split[name])
            if limit < 1 or limit > available_paths:
                raise ValueError(
                    f"base union {name} requested {limit:,} of "
                    f"{available_paths:,} complete paths"
                )
            limited_physical = physical[:limit]
            target_paths = np.lib.stride_tricks.sliding_window_view(
                full.targets, self.return_count
            )[:limit]
            self.splits[name] = DerivedFeatureSplit(
                features=DerivedUnionFeatureView(
                    limited_physical,
                    split_production[:limit],
                    global470,
                    self.base.minute_source_rows,
                    self.base.top_of_book,
                ),
                targets=target_paths,
                times=full.times[:limit],
                physical_rows=limited_physical,
            )
        self._global470 = global470

    def logical_count(self, split: str) -> int:
        return self.splits[split].count

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
    ):
        values = self.splits[split]
        available = values.count
        count = available if limit is None else min(available, int(limit))
        order = np.arange(available, dtype=np.int64)
        if count < available:
            order = np.linspace(0, available - 1, count, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, count, batch_size):
            selected = order[start:start + batch_size]
            if shuffle:
                selected = np.sort(selected)
            yield (
                torch.from_numpy(np.asarray(values.features[selected]).copy()),
                torch.from_numpy(values.targets[selected].copy()),
                torch.ones(selected.size, dtype=torch.float32),
            )

    def iter_adversarial_batches(
        self,
        adversarial_view,
        batch_size: int,
        *,
        seed: int,
        limit: int | None = None,
    ):
        values = self.splits["train"]
        available = values.count
        if adversarial_view.count != available:
            raise ValueError("adversarial view does not match the training split")
        count = available if limit is None else min(available, int(limit))
        order = np.arange(available, dtype=np.int64)
        if count < available:
            order = np.linspace(0, available - 1, count, dtype=np.int64)
        np.random.default_rng(seed).shuffle(order)
        for start in range(0, count, batch_size):
            selected = np.sort(order[start:start + batch_size])
            yield (
                torch.from_numpy(np.asarray(values.features[selected]).copy()),
                adversarial_view.rows(selected),
                torch.from_numpy(values.targets[selected].copy()),
                torch.ones(selected.size, dtype=torch.float32),
            )
