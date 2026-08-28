from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from global_feature_candidate_axis import BASE_DIR
from global_feature_registry import ROOT, coordinate_id, prediction_market_asset, safe_id
from search_global_btc_working_set import load_targets


DEFAULT_INPUT = ROOT / "data/market/mutable/prediction-markets/kalshi/relevant-2026-07-18_2026-08-17"
DEFAULT_OUTPUT = ROOT / "data/runtime-cache/prediction-market-candidate-axis-30d"


SEMANTIC_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("monetary-policy", re.compile(r"\b(fed|fomc|interest rate|rate cut|rate hike|central bank|ecb|boj|boe)\b", re.I)),
    ("inflation", re.compile(r"\b(cpi|pce|ppi|inflation|consumer price)\b", re.I)),
    ("labor", re.compile(r"\b(payroll|jobs?|unemployment|jobless|wage|jolts|layoff)\b", re.I)),
    ("growth", re.compile(r"\b(gdp|recession|retail sales|industrial production|pmi)\b", re.I)),
    ("trade", re.compile(r"\b(tariff|trade deficit|sanction|export|import)\b", re.I)),
    ("geopolitics", re.compile(r"\b(war|invasion|ceasefire|nato|iran|israel|ukraine|china|taiwan|russia)\b", re.I)),
    ("elections", re.compile(r"\b(election|president|congress|senate|house|prime minister|vote)\b", re.I)),
    ("regulation", re.compile(r"\b(regulation|law|bill|sec|cftc|reserve|legal|ban)\b", re.I)),
    ("technology-ai", re.compile(r"\b(ai|artificial intelligence|openai|chip|semiconductor|nvidia|cyber|hack)\b", re.I)),
    ("energy-commodities", re.compile(r"\b(oil|gas|gold|silver|copper|energy|opec|wheat|corn)\b", re.I)),
    ("systemic-risk", re.compile(r"\b(default|bank|debt|shutdown|pandemic|earthquake|hurricane|suez|shipping)\b", re.I)),
)


def semantic_families(series_ticker: str, series_title: str | None) -> list[str]:
    text = f"{series_ticker} {series_title or ''}"
    return [name for name, pattern in SEMANTIC_FAMILIES if pattern.search(text)]


def market_buckets(market: dict[str, Any]) -> list[tuple[str, str, str]]:
    scope = str(market.get("scope", "global"))
    category = safe_id(str(market.get("category", "unclassified")))
    series = str(market.get("series", ""))
    title = market.get("seriesTitle")
    if scope == "asset":
        asset = prediction_market_asset(market).lower()
        buckets = [
            ("asset", asset, "all"),
            ("general", "global-crypto", f"asset-{asset}"),
            ("general", "global-crypto", f"category-{category}"),
            ("general", "global", "all-prediction-markets"),
        ]
        buckets.extend(
            ("general", "global", f"family-{family}")
            for family in semantic_families(series, title)
        )
        return buckets
    buckets = [
        ("general", "global", "all-global-events"),
        ("general", "global", f"category-{category}"),
        ("general", "global", "all-prediction-markets"),
    ]
    buckets.extend(
        ("general", "global", f"family-{family}")
        for family in semantic_families(series, title)
    )
    return buckets


@dataclass
class Accumulator:
    count: np.ndarray
    weight: np.ndarray
    probability: np.ndarray
    probability_square: np.ndarray
    probability_minimum: np.ndarray
    probability_maximum: np.ndarray
    weighted_probability: np.ndarray
    spread: np.ndarray
    spread_count: np.ndarray
    change: np.ndarray
    change_count: np.ndarray
    positive_shock: np.ndarray
    negative_shock: np.ndarray
    rising: np.ndarray
    volume: np.ndarray
    open_interest: np.ndarray
    time_to_close: np.ndarray
    age: np.ndarray
    age_count: np.ndarray

    @classmethod
    def create(cls, rows: int) -> "Accumulator":
        zeros = lambda: np.zeros(rows, dtype=np.float64)
        return cls(
            count=zeros(), weight=zeros(), probability=zeros(), probability_square=zeros(),
            probability_minimum=np.full(rows, np.inf), probability_maximum=np.full(rows, -np.inf),
            weighted_probability=zeros(), spread=zeros(), spread_count=zeros(), change=zeros(),
            change_count=zeros(), positive_shock=np.full(rows, -np.inf),
            negative_shock=np.full(rows, np.inf), rising=zeros(), volume=zeros(),
            open_interest=zeros(), time_to_close=zeros(), age=zeros(), age_count=zeros(),
        )

    def add(
        self, row: int, *, probability: float | None, spread: float | None,
        change: float | None, volume: float, open_interest: float | None,
        time_to_close_ms: float | None, age_ms: float | None,
    ) -> None:
        if probability is None or not math.isfinite(probability):
            return
        logit = math.log(min(0.995, max(0.005, probability)) / (1 - min(0.995, max(0.005, probability))))
        point_weight = 1.0 + math.log1p(max(0.0, volume))
        if open_interest is not None and math.isfinite(open_interest):
            point_weight += 0.1 * math.log1p(max(0.0, open_interest))
        self.count[row] += 1
        self.weight[row] += point_weight
        self.probability[row] += logit
        self.probability_square[row] += logit * logit
        self.probability_minimum[row] = min(self.probability_minimum[row], logit)
        self.probability_maximum[row] = max(self.probability_maximum[row], logit)
        self.weighted_probability[row] += point_weight * logit
        self.volume[row] += max(0.0, volume)
        if open_interest is not None and math.isfinite(open_interest):
            self.open_interest[row] += max(0.0, open_interest)
        if spread is not None and math.isfinite(spread):
            self.spread[row] += spread
            self.spread_count[row] += 1
        if change is not None and math.isfinite(change):
            self.change[row] += change
            self.change_count[row] += 1
            self.rising[row] += float(change > 0)
            self.positive_shock[row] = max(self.positive_shock[row], change)
            self.negative_shock[row] = min(self.negative_shock[row], change)
        if time_to_close_ms is not None and math.isfinite(time_to_close_ms):
            self.time_to_close[row] += math.log1p(max(0.0, time_to_close_ms) / 60_000)
        if age_ms is not None and math.isfinite(age_ms):
            self.age[row] += math.log1p(max(0.0, age_ms) / 1_000)
            self.age_count[row] += 1

    def add_batch(self, records: list[tuple[float, ...]]) -> None:
        if not records:
            return
        values = np.asarray(records, dtype=np.float64)
        rows = values[:, 0].astype(np.intp)
        probability, spread, change = values[:, 1], values[:, 2], values[:, 3]
        volume, open_interest = values[:, 4], values[:, 5]
        time_to_close_ms, age_ms = values[:, 6], values[:, 7]
        valid = np.isfinite(probability)
        if not np.any(valid):
            return
        rows = rows[valid]
        probability, spread, change = probability[valid], spread[valid], change[valid]
        volume, open_interest = volume[valid], open_interest[valid]
        time_to_close_ms, age_ms = time_to_close_ms[valid], age_ms[valid]
        logit = np.log(np.clip(probability, 0.005, 0.995) / (1 - np.clip(probability, 0.005, 0.995)))
        point_weight = 1.0 + np.log1p(np.maximum(0.0, volume))
        valid_open_interest = np.isfinite(open_interest)
        point_weight[valid_open_interest] += 0.1 * np.log1p(np.maximum(0.0, open_interest[valid_open_interest]))
        np.add.at(self.count, rows, 1)
        np.add.at(self.weight, rows, point_weight)
        np.add.at(self.probability, rows, logit)
        np.add.at(self.probability_square, rows, logit * logit)
        np.minimum.at(self.probability_minimum, rows, logit)
        np.maximum.at(self.probability_maximum, rows, logit)
        np.add.at(self.weighted_probability, rows, point_weight * logit)
        np.add.at(self.volume, rows, np.maximum(0.0, volume))
        if np.any(valid_open_interest):
            np.add.at(self.open_interest, rows[valid_open_interest], np.maximum(0.0, open_interest[valid_open_interest]))
        valid_spread = np.isfinite(spread)
        if np.any(valid_spread):
            np.add.at(self.spread, rows[valid_spread], spread[valid_spread])
            np.add.at(self.spread_count, rows[valid_spread], 1)
        valid_change = np.isfinite(change)
        if np.any(valid_change):
            changed_rows = rows[valid_change]
            changed = change[valid_change]
            np.add.at(self.change, changed_rows, changed)
            np.add.at(self.change_count, changed_rows, 1)
            np.add.at(self.rising, changed_rows, changed > 0)
            np.maximum.at(self.positive_shock, changed_rows, changed)
            np.minimum.at(self.negative_shock, changed_rows, changed)
        valid_time_to_close = np.isfinite(time_to_close_ms)
        if np.any(valid_time_to_close):
            np.add.at(
                self.time_to_close,
                rows[valid_time_to_close],
                np.log1p(np.maximum(0.0, time_to_close_ms[valid_time_to_close]) / 60_000),
            )
        valid_age = np.isfinite(age_ms)
        if np.any(valid_age):
            np.add.at(self.age, rows[valid_age], np.log1p(np.maximum(0.0, age_ms[valid_age]) / 1_000))
            np.add.at(self.age_count, rows[valid_age], 1)


@dataclass
class CarryAccumulator:
    """Range-difference accumulator for the latest known per-contract state.

    Probability, spread, and open interest are state variables and remain
    causally known until a newer observation or contract close. Interval
    volume and probability change deliberately stay in ``Accumulator`` and
    are never carried through quiet minutes.
    """

    count: np.ndarray
    weight: np.ndarray
    probability: np.ndarray
    probability_square: np.ndarray
    weighted_probability: np.ndarray
    spread: np.ndarray
    spread_count: np.ndarray
    open_interest: np.ndarray

    @classmethod
    def create(cls, rows: int) -> "CarryAccumulator":
        zeros = lambda: np.zeros(rows + 1, dtype=np.float64)
        return cls(*(zeros() for _ in range(8)))

    @staticmethod
    def _range_add(target: np.ndarray, start: int, end: int, value: float) -> None:
        if start >= end or not math.isfinite(value):
            return
        target[start] += value
        target[end] -= value

    def add_interval(
        self,
        start: int,
        end: int,
        *,
        probability: float | None,
        spread: float | None,
        open_interest: float | None,
    ) -> None:
        if probability is None or not math.isfinite(probability):
            return
        clipped = min(0.995, max(0.005, probability))
        logit = math.log(clipped / (1 - clipped))
        point_weight = 1.0
        if open_interest is not None and math.isfinite(open_interest):
            point_weight += 0.1 * math.log1p(max(0.0, open_interest))
            self._range_add(self.open_interest, start, end, max(0.0, open_interest))
        self._range_add(self.count, start, end, 1.0)
        self._range_add(self.weight, start, end, point_weight)
        self._range_add(self.probability, start, end, logit)
        self._range_add(self.probability_square, start, end, logit * logit)
        self._range_add(self.weighted_probability, start, end, point_weight * logit)
        if spread is not None and math.isfinite(spread):
            self._range_add(self.spread, start, end, spread)
            self._range_add(self.spread_count, start, end, 1.0)

    def features(self) -> dict[str, np.ndarray]:
        count = np.cumsum(self.count[:-1])
        weight = np.cumsum(self.weight[:-1])
        probability = np.cumsum(self.probability[:-1])
        probability_square = np.cumsum(self.probability_square[:-1])
        mean = finite_ratio(probability, count)
        second = finite_ratio(probability_square, count)
        return {
            "stateful-log-active-market-count": np.log1p(count),
            "stateful-logit-probability-mean": mean,
            "stateful-logit-probability-dispersion": np.sqrt(np.maximum(0.0, second - mean * mean)),
            "stateful-liquidity-weighted-logit-probability": finite_ratio(
                np.cumsum(self.weighted_probability[:-1]), weight
            ),
            "stateful-quote-spread-mean": finite_ratio(
                np.cumsum(self.spread[:-1]), np.cumsum(self.spread_count[:-1])
            ),
            "stateful-log-open-interest": np.log1p(np.cumsum(self.open_interest[:-1])),
        }


def finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    output[valid] = numerator[valid] / denominator[valid]
    return output


def lag_difference(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values[lag:]) & np.isfinite(values[:-lag])
    output[lag:][valid] = values[lag:][valid] - values[:-lag][valid]
    return output


def accumulator_features(accumulator: Accumulator) -> dict[str, np.ndarray]:
    mean = finite_ratio(accumulator.probability, accumulator.count)
    second = finite_ratio(accumulator.probability_square, accumulator.count)
    dispersion = np.sqrt(np.maximum(0.0, second - mean * mean))
    weighted = finite_ratio(accumulator.weighted_probability, accumulator.weight)
    change_mean = finite_ratio(accumulator.change, accumulator.change_count)
    output = {
        "log-active-market-count": np.log1p(accumulator.count),
        "logit-probability-mean": mean,
        "logit-probability-dispersion": dispersion,
        "logit-probability-minimum": np.where(accumulator.count > 0, accumulator.probability_minimum, np.nan),
        "logit-probability-maximum": np.where(accumulator.count > 0, accumulator.probability_maximum, np.nan),
        "liquidity-weighted-logit-probability": weighted,
        "quote-spread-mean": finite_ratio(accumulator.spread, accumulator.spread_count),
        "probability-change-mean": change_mean,
        "positive-probability-shock": np.where(accumulator.change_count > 0, accumulator.positive_shock, np.nan),
        "negative-probability-shock": np.where(accumulator.change_count > 0, accumulator.negative_shock, np.nan),
        "rising-market-fraction": finite_ratio(accumulator.rising, accumulator.change_count),
        "log-contract-volume": np.log1p(accumulator.volume),
        "log-open-interest": np.log1p(accumulator.open_interest),
        "log-time-to-close-mean": finite_ratio(accumulator.time_to_close, accumulator.count),
        "log-last-trade-age-mean": finite_ratio(accumulator.age, accumulator.age_count),
    }
    for lag in (1, 5, 15, 60):
        output[f"logit-probability-change-{lag}m"] = lag_difference(weighted, lag)
    return output


def row_values(raw: dict[str, Any], cadence: str) -> dict[str, float | None]:
    if cadence == "1s" or "lastProbability" in raw:
        return {
            "probability": raw.get("lastProbability"),
            "spread": None,
            "change": raw.get("intervalProbabilityChange"),
            "volume": float(raw.get("intervalContractVolume") or 0),
            "open_interest": None,
            "age_ms": raw.get("lastTradeAgeMs"),
        }
    probability = raw.get("quoteMidProbability")
    if probability is None:
        probability = raw.get("tradeCloseProbability")
    if probability is None:
        probability = raw.get("tradeMeanProbability")
    close = raw.get("tradeCloseProbability")
    previous = raw.get("previousTradeProbability")
    return {
        "probability": probability,
        "spread": raw.get("quotedSpread"),
        "change": close - previous if close is not None and previous is not None else None,
        "volume": float(raw.get("contractVolume") or 0),
        "open_interest": raw.get("openInterest"),
        "age_ms": 0.0,
    }


def find_artifact(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*.{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *.{suffix} in {directory}, found {len(matches)}")
    return matches[0]


def causal_origin_index(origins: np.ndarray, available_at: int) -> int | None:
    """Return the first model origin that can causally observe a source row."""
    index = int(np.searchsorted(origins, available_at, side="left"))
    return index if index < origins.size else None


def flush_pending(
    pending: dict[tuple[str, tuple[str, str, str]], list[tuple[float, ...]]],
    accumulators: dict[tuple[str, tuple[str, str, str]], Accumulator],
    rows: int,
) -> None:
    for key, records in pending.items():
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = Accumulator.create(rows)
            accumulators[key] = accumulator
        accumulator.add_batch(records)
    pending.clear()


def add_contract_carry_intervals(
    raw_rows: list[tuple[int, dict[str, Any]]],
    *,
    market: dict[str, Any],
    cadence: str,
    origins: np.ndarray,
    buckets: Iterable[tuple[str, str, str]],
    accumulators: dict[tuple[str, tuple[str, str, str]], CarryAccumulator],
) -> None:
    """Carry each latest causal state until replacement or contract close."""
    if not raw_rows:
        return
    by_origin = {origin: raw for origin, raw in raw_rows}
    ordered = sorted(by_origin.items())
    close_ms = int(datetime.fromisoformat(str(market["closeTime"]).replace("Z", "+00:00")).timestamp() * 1_000)
    close_origin = int(np.searchsorted(origins, close_ms, side="left"))
    for index, (start, raw) in enumerate(ordered):
        end = ordered[index + 1][0] if index + 1 < len(ordered) else close_origin
        end = min(end, close_origin, origins.size)
        if start >= end:
            continue
        values = row_values(raw, cadence)
        for bucket in buckets:
            key = (cadence, bucket)
            accumulator = accumulators.get(key)
            if accumulator is None:
                accumulator = CarryAccumulator.create(origins.size)
                accumulators[key] = accumulator
            accumulator.add_interval(
                start,
                end,
                probability=values["probability"],
                spread=values["spread"],
                open_interest=values["open_interest"],
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize fixed-width causal Kalshi candidate summaries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    metadata_path = find_artifact(input_dir, "markets.json")
    source_manifest_path = find_artifact(input_dir, "manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["markets"]
    _, _, _, origins = load_targets()
    market_by_id = {int(row["id"]): row for row in metadata}
    buckets_by_market = {market_id: market_buckets(row) for market_id, row in market_by_id.items()}
    accumulators: dict[tuple[str, tuple[str, str, str]], Accumulator] = {}
    carry_accumulators: dict[tuple[str, tuple[str, str, str]], CarryAccumulator] = {}
    pending: dict[tuple[str, tuple[str, str, str]], list[tuple[float, ...]]] = {}
    pending_records = 0
    source_rows = {}
    for cadence in ("1s", "1m"):
        source = find_artifact(input_dir, f"{cadence}.ndjson.gz")
        rows = 0
        carry_market_id: int | None = None
        carry_rows: list[tuple[int, dict[str, Any]]] = []
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            for line in stream:
                raw = json.loads(line)
                origin = causal_origin_index(origins, int(raw["availableAt"]))
                if origin is None:
                    continue
                market_id = int(raw["marketId"])
                if carry_market_id is not None and market_id != carry_market_id:
                    if market_id < carry_market_id:
                        raise ValueError("Prediction-market source rows must be ordered by market id")
                    add_contract_carry_intervals(
                        carry_rows,
                        market=market_by_id[carry_market_id],
                        cadence=cadence,
                        origins=origins,
                        buckets=buckets_by_market.get(carry_market_id, ()),
                        accumulators=carry_accumulators,
                    )
                    carry_rows = []
                carry_market_id = market_id
                carry_rows.append((origin, raw))
                values = row_values(raw, cadence)
                record = (
                    float(origin),
                    float(values["probability"]) if values["probability"] is not None else math.nan,
                    float(values["spread"]) if values["spread"] is not None else math.nan,
                    float(values["change"]) if values["change"] is not None else math.nan,
                    float(values["volume"] or 0),
                    float(values["open_interest"]) if values["open_interest"] is not None else math.nan,
                    float(raw["timeToCloseMs"]) if raw.get("timeToCloseMs") is not None else math.nan,
                    float(values["age_ms"]) if values["age_ms"] is not None else math.nan,
                )
                for bucket in buckets_by_market.get(market_id, ()):
                    key = (cadence, bucket)
                    pending.setdefault(key, []).append(record)
                    pending_records += 1
                if pending_records >= 250_000:
                    flush_pending(pending, accumulators, origins.size)
                    pending_records = 0
                rows += 1
        if carry_market_id is not None:
            add_contract_carry_intervals(
                carry_rows,
                market=market_by_id[carry_market_id],
                cadence=cadence,
                origins=origins,
                buckets=buckets_by_market.get(carry_market_id, ()),
                accumulators=carry_accumulators,
            )
        flush_pending(pending, accumulators, origins.size)
        pending_records = 0
        source_rows[cadence] = rows

    coordinates: list[dict[str, Any]] = []
    columns: list[np.ndarray] = []
    for cadence, (subject_kind, subject, bucket) in sorted(set(accumulators) | set(carry_accumulators)):
        key = (cadence, (subject_kind, subject, bucket))
        feature_values = {}
        if key in accumulators:
            feature_values.update(accumulator_features(accumulators[key]))
        if key in carry_accumulators:
            feature_values.update(carry_accumulators[key].features())
        for metric, values in sorted(feature_values.items()):
            formula = metric if bucket == "all" else f"{bucket}-{metric}"
            feature_id = coordinate_id(subject, subject_kind, "kalshi", cadence, formula)
            availability = float(np.mean(np.isfinite(values)))
            if availability <= 0:
                continue
            coordinates.append({
                "id": feature_id,
                "source": "prediction-market",
                "family": "prediction-market-state",
                "cadence": cadence,
                "bucket": bucket,
                "metric": metric,
                "empiricalAvailability": availability,
            })
            columns.append(values.astype(np.float32))
    if not columns:
        raise RuntimeError("Prediction-market source produced no aligned candidate coordinates")
    matrix = np.column_stack(columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "prediction-market-axis.f32.partial"
    final = output_dir / "prediction-market-axis.f32"
    mapped = np.memmap(partial, dtype="<f4", mode="w+", shape=matrix.shape)
    mapped[:] = matrix
    mapped.flush()
    del mapped
    partial.replace(final)
    manifest = {
        "version": 2,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceManifest": str(source_manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "sourceRequest": source_manifest["request"],
        "causalSemantics": source_manifest["semantics"],
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "file": final.name,
        "dtype": "<f4",
        "sourceRows": source_rows,
        "markets": len(metadata),
        "bucketCount": len(set(accumulators) | set(carry_accumulators)),
        "stateSemantics": {
            "carried": "Latest causal probability, quote spread, and open interest persist until replacement or contract close.",
            "intervalOnly": "Contract volume and probability-change fields remain attached only to their source interval.",
            "statefulMetrics": "Metrics prefixed stateful summarize carried contract state and are distinct from sparse update metrics.",
        },
        "coordinates": coordinates,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("rows", "columns", "sourceRows", "markets", "bucketCount")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
