from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourcePolicy:
    id: str
    availability_score: float
    acquisition_cost: int
    history_class: str
    point_in_time: bool


@dataclass
class TemplateBinding:
    canonical_id: str
    level: str
    family: str
    cadence: str
    source_policy: str
    subjects: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    inventories: set[str] = field(default_factory=set)
    construction: str = ""
    lookback: str = ""
    delay: str = ""

    @property
    def coordinates(self) -> int:
        return len(self.subjects)


SOURCE_POLICIES = {
    "candle-derived": SourcePolicy("candle-derived", 1.0, 0, "archive", True),
    "binance-archive": SourcePolicy("binance-archive", 0.96, 1, "archive", True),
    "binance-futures-archive": SourcePolicy("binance-futures-archive", 0.95, 1, "archive", True),
    "official-market-archive": SourcePolicy("official-market-archive", 0.92, 2, "archive", True),
    "official-public-slow": SourcePolicy("official-public-slow", 0.90, 2, "public-api", False),
    "community-revised": SourcePolicy("community-revised", 0.65, 3, "revised-history", False),
    "sampled-history": SourcePolicy("sampled-history", 0.60, 4, "sparse-sample", True),
    "live-only": SourcePolicy("live-only", 0.45, 5, "collector", True),
}


SHORT_WINDOW_INVENTORIES = {
    "fast-live",
    "deribit-option-surface",
    "gdelt-news",
    "tardis-cross-venue",
}


OPTION_SURFACE_FIELDS = (
    "atm-iv-1d", "atm-iv-7d", "atm-iv-30d", "atm-iv-term-7d-minus-1d",
    "atm-iv-term-30d-minus-7d", "put-call-25d-skew-1d", "put-call-25d-skew-7d",
    "put-call-25d-skew-30d", "call-put-oi-imbalance", "one-day-call-put-oi-imbalance",
    "log-distance-to-major-strike", "hours-to-next-expiry", "atm-iv-change-1m",
    "atm-iv-change-15m", "atm-iv-change-60m", "skew-change-1m", "skew-change-15m",
    "skew-change-60m", "implied-minus-realized-volatility", "option-surface-age",
)


PREDICTION_MARKET_ASSET_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BTC", ("btc", "bitcoin")), ("ETH", ("eth", "ethereum")),
    ("SOL", ("sol", "solana")), ("XRP", ("xrp", "ripple")),
    ("BNB", ("bnb",)), ("DOGE", ("doge", "dogecoin")),
    ("ADA", ("ada", "cardano")), ("AVAX", ("avax", "avalanche")),
    ("BCH", ("bch", "bitcoin cash")), ("DOT", ("dot", "polkadot")),
    ("LINK", ("link", "chainlink")), ("LTC", ("ltc", "litecoin")),
    ("HYPE", ("hype", "hyperliquid")), ("NEAR", ("near",)),
    ("ZEC", ("zec", "zcash")), ("XMR", ("xmr", "monero")),
    ("SHIB", ("shib", "shiba")), ("SUI", ("sui",)),
    ("TRON", ("tron", "trx")), ("ARB", ("arbitrum", "arb")),
)


PREDICTION_NORMALIZED_STATE_FIELDS = (
    "probability", "quoted-spread", "probability-change", "contract-volume",
    "open-interest", "last-trade-age", "time-to-close",
)


def load_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    if any(ord(character) > 127 for character in value):
        return "unicode-" + value.lower().encode("utf-8").hex()
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if result:
        return result
    raise ValueError(f"Identifier component has no encodable content: {value!r}")


def canonical_id(subject_kind: str, venue: str, cadence: str, formula: str) -> str:
    return "/".join(map(safe_id, (subject_kind, venue, cadence, formula)))


def coordinate_id(subject: str, subject_kind: str, venue: str, cadence: str, formula: str) -> str:
    return "/".join(map(safe_id, (subject_kind, subject, venue, cadence, formula)))


def feature_level(formula: str) -> str:
    """Classify an already-exposed model candidate at the acquisition boundary.

    This is deliberately separate from ``source_feature_inventory``: a raw
    OHLC field that is retained only in storage is a base source field even if
    it is not present here as a candidate coordinate. Basic candidates are
    atomic causal values already exposed to the search; arithmetic transforms,
    including log/absolute levels, cross-market basis, changes, rolling
    summaries, ages, and indicators, are derived candidates.
    """
    normalized = safe_id(formula)
    if normalized == "spot-flow-last-side-1s":
        return "basic"
    if (
        normalized.endswith("-level")
        and not normalized.endswith(("-absolute-level", "-log-level"))
        and normalized != "futures-basis-level"
    ):
        return "basic"
    return "derived"


class FeatureRegistry:
    def __init__(self) -> None:
        self.templates: dict[str, TemplateBinding] = {}
        self.raw_by_inventory: dict[str, int] = {}
        self.inventory_notes: dict[str, str] = {}

    def add(
        self,
        *,
        inventory: str,
        subject: str,
        subject_kind: str,
        venue: str,
        cadence: str,
        formula: str,
        alias: str,
        family: str,
        source_policy: str,
        construction: str = "",
        lookback: str = "",
        delay: str = "",
    ) -> None:
        if source_policy not in SOURCE_POLICIES:
            raise ValueError(f"Unknown source policy: {source_policy}")
        key = canonical_id(subject_kind, venue, cadence, formula)
        level = feature_level(formula)
        binding = self.templates.get(key)
        if binding is None:
            binding = TemplateBinding(
                canonical_id=key,
                level=level,
                family=family,
                cadence=cadence,
                source_policy=source_policy,
                construction=construction,
                lookback=lookback,
                delay=delay,
            )
            self.templates[key] = binding
        else:
            if binding.level != level:
                raise ValueError(f"Feature level differs for canonical template {key}.")
            current = SOURCE_POLICIES[binding.source_policy]
            candidate = SOURCE_POLICIES[source_policy]
            if (
                (candidate.point_in_time and not current.point_in_time)
                or (
                    candidate.point_in_time == current.point_in_time
                    and (
                        candidate.availability_score > current.availability_score
                        or (
                            candidate.availability_score == current.availability_score
                            and candidate.acquisition_cost < current.acquisition_cost
                        )
                    )
                )
            ):
                binding.source_policy = source_policy
        binding.subjects.add(subject)
        binding.aliases.add(f"{inventory}:{alias}")
        binding.inventories.add(inventory)
        self.raw_by_inventory[inventory] = self.raw_by_inventory.get(inventory, 0) + 1

    @property
    def unique_coordinates(self) -> int:
        return sum(row.coordinates for row in self.templates.values())

    @property
    def raw_coordinates(self) -> int:
        return sum(self.raw_by_inventory.values())

    def summary(self) -> dict[str, Any]:
        # A template overlaps inventories when the same canonical construction
        # was declared by more than one source inventory.  Alias and subject
        # cardinalities are unrelated (one inventory can expose several aliases
        # for the same subject), so comparing those counts misclassifies overlap.
        overlaps = [row for row in self.templates.values() if len(row.inventories) > 1]
        asset_specific = sum(
            row.coordinates
            for row in self.templates.values()
            if row.canonical_id.startswith("asset/")
        )
        taxonomy = {
            (scope, level): sum(
                row.coordinates
                for row in self.templates.values()
                if row.canonical_id.startswith(scope + "/") and row.level == level
            )
            for scope in ("asset", "general")
            for level in ("basic", "derived")
        }
        template_taxonomy = {
            (scope, level): sum(
                1
                for row in self.templates.values()
                if row.canonical_id.startswith(scope + "/") and row.level == level
            )
            for scope in ("asset", "general")
            for level in ("basic", "derived")
        }
        robust_templates = [
            row
            for row in self.templates.values()
            if row.inventories - SHORT_WINDOW_INVENTORIES
        ]
        robust_coordinates = sum(row.coordinates for row in robust_templates)
        causal_robust_coordinates = sum(
            row.coordinates
            for row in robust_templates
            if SOURCE_POLICIES[row.source_policy].point_in_time
        )
        return {
            "rawCoordinates": self.raw_coordinates,
            "uniqueCoordinates": self.unique_coordinates,
            "duplicateLedgerCoordinates": self.raw_coordinates - self.unique_coordinates,
            "assetSpecificCoordinates": asset_specific,
            "generalCoordinates": self.unique_coordinates - asset_specific,
            "basicAssetSpecificCandidateTemplates": template_taxonomy[("asset", "basic")],
            "derivedAssetSpecificCandidateTemplates": template_taxonomy[("asset", "derived")],
            "basicGeneralCandidateTemplates": template_taxonomy[("general", "basic")],
            "derivedGeneralCandidateTemplates": template_taxonomy[("general", "derived")],
            "basicAssetSpecificCandidateCoordinates": taxonomy[("asset", "basic")],
            "derivedAssetSpecificCandidateCoordinates": taxonomy[("asset", "derived")],
            "basicGeneralCandidateCoordinates": taxonomy[("general", "basic")],
            "derivedGeneralCandidateCoordinates": taxonomy[("general", "derived")],
            "robust30dCoordinates": robust_coordinates,
            "shortWindowCoordinates": self.unique_coordinates - robust_coordinates,
            "causalRobust30dCoordinates": causal_robust_coordinates,
            "nonPointInTimeRobust30dCoordinates": robust_coordinates - causal_robust_coordinates,
            "canonicalTemplates": len(self.templates),
            "overlappingTemplates": len(overlaps),
            "inventories": [
                {
                    "id": inventory,
                    "rawCoordinates": count,
                    "note": self.inventory_notes.get(inventory, ""),
                }
                for inventory, count in sorted(self.raw_by_inventory.items())
            ],
        }


def asset_sets() -> dict[str, list[str]]:
    minute = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/manifest.json")
    fast = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/fast-manifest.json")
    derivatives = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/derivatives-manifest.json")
    book = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/book-depth-manifest.json")
    preferred = []
    spot_1m = []
    usdm_1m = []
    paired = []
    for row in minute["assets"]:
        coverage = max((float(market["coverage"]) for market in row["markets"]), default=0.0)
        if coverage >= 0.99:
            preferred.append(row["asset"])
        if any(
            market["venue"] == "spot" and float(market["coverage"]) >= 0.99
            for market in row["markets"]
        ):
            spot_1m.append(row["asset"])
        if any(
            market["venue"] == "usdm-futures" and float(market["coverage"]) >= 0.99
            for market in row["markets"]
        ):
            usdm_1m.append(row["asset"])
        venues = {market["venue"] for market in row["markets"] if float(market["coverage"]) >= 0.95}
        if {"spot", "usdm-futures"} <= venues:
            paired.append(row["asset"])
    return {
        "universe": [row["asset"] for row in minute["assets"]],
        "preferred1m99": preferred,
        "spot1m99": spot_1m,
        "usdm1m99": usdm_1m,
        "spot1s95": [row["asset"] for row in fast["assets"] if float(row["coverage"]) >= 0.95],
        "usdmMetrics95": [
            row["asset"] for row in derivatives["assets"]
            if row.get("metrics") and float(row["metrics"]["coverage"]) >= 0.95
        ],
        "usdmFunding": [row["asset"] for row in derivatives["assets"] if row.get("funding")],
        "usdmBook95": [row["asset"] for row in book["assets"] if float(row["coverage"]) >= 0.95],
        "spotAndUsdm95": paired,
    }


def prediction_market_asset(market: dict[str, Any]) -> str:
    """Return the stable asset subject used by the prediction-market axis."""
    text = f"{market.get('series', '')} {market.get('seriesTitle', '')}".lower()
    padded = f" {re.sub(r'[^a-z0-9]+', ' ', text)} "
    for asset, aliases in PREDICTION_MARKET_ASSET_ALIASES:
        if any(f" {alias} " in padded for alias in aliases):
            return asset
    return "CRYPTO-OTHER"


def _prediction_observation_counts(snapshot_ms: int) -> tuple[int, dict[str, dict[str, int]], dict[str, int]]:
    """Read the materialized dynamic-set count fields at the nearest causal origin."""
    try:
        import numpy as np

        base = ROOT / "data/runtime-cache/global-feature-basis-30d"
        base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        dataset = base_manifest["datasets"][0]
        rows = int(dataset["rows"])
        times = np.memmap(
            base / dataset["files"]["times"], dtype="<f8", mode="r", shape=(rows,)
        )
        causal_rows = np.flatnonzero(times <= snapshot_ms)
        row_index = int(causal_rows[-1])
        origin_ms = int(times[row_index])

        axis_dir = ROOT / "data/runtime-cache/prediction-market-candidate-axis-30d"
        manifest = json.loads((axis_dir / "manifest.json").read_text(encoding="utf-8"))
        values = np.memmap(
            axis_dir / manifest["file"],
            dtype="<f4",
            mode="r",
            shape=(int(manifest["rows"]), int(manifest["columns"])),
        )
        asset_counts: dict[str, dict[str, int]] = defaultdict(dict)
        general_counts: dict[str, int] = {}
        for index, coordinate in enumerate(manifest["coordinates"]):
            coordinate_id_value = str(coordinate["id"])
            if not coordinate_id_value.endswith("log-active-market-count"):
                continue
            parts = coordinate_id_value.split("/")
            count = int(round(float(np.expm1(values[row_index, index]))))
            if parts[0] == "asset" and parts[4] == "log-active-market-count":
                asset_counts[parts[1].upper()][parts[3]] = count
            elif (
                coordinate_id_value.endswith("/all-global-events-log-active-market-count")
                and parts[0] == "general"
            ):
                general_counts[parts[3]] = count
        return origin_ms, dict(asset_counts), general_counts
    except (FileNotFoundError, KeyError, ValueError):
        return snapshot_ms, {}, {}


def prediction_market_source_inventory() -> dict[str, Any] | None:
    directory = ROOT / "data/market/mutable/prediction-markets/kalshi/relevant-2026-07-18_2026-08-17"
    metadata_paths = sorted(directory.glob("*.markets.json"))
    if len(metadata_paths) != 1:
        return None
    markets = json.loads(metadata_paths[0].read_text(encoding="utf-8"))["markets"]
    snapshot = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    snapshot_ms = int(snapshot.timestamp() * 1_000)

    def instant(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def is_open(market: dict[str, Any]) -> bool:
        return instant(str(market["openTime"])) <= snapshot < instant(str(market["closeTime"]))

    asset_markets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_markets = []
    for market in markets:
        if market["scope"] == "asset":
            asset_markets[prediction_market_asset(market)].append(market)
        else:
            global_markets.append(market)

    observation_origin_ms, observed_assets, observed_general = _prediction_observation_counts(snapshot_ms)
    asset_rows = []
    for asset, rows in sorted(asset_markets.items()):
        opened = [row for row in rows if is_open(row)]
        observations = observed_assets.get(asset, {})
        asset_rows.append({
            "asset": asset,
            "recognizedAsset": asset != "CRYPTO-OTHER",
            "window": {
                "persistentSeries": len({row["series"] for row in rows}),
                "events": len({row["eventTicker"] for row in rows}),
                "contracts": len(rows),
            },
            "snapshot": {
                "openSeries": len({row["series"] for row in opened}),
                "openEvents": len({row["eventTicker"] for row in opened}),
                "openContracts": len(opened),
                "persistentSeriesFieldChannels": (
                    len({row["series"] for row in opened}) * len(PREDICTION_NORMALIZED_STATE_FIELDS)
                ),
                "openContractFieldSlots": len(opened) * len(PREDICTION_NORMALIZED_STATE_FIELDS),
                "potentialContractMetadataFieldSlots": len(opened) * 25,
                "observedContracts1m": int(observations.get("1m", 0)),
                "observedContracts1s": int(observations.get("1s", 0)),
                "observedContractFieldSlotsUpperBound1m": (
                    int(observations.get("1m", 0)) * len(PREDICTION_NORMALIZED_STATE_FIELDS)
                ),
                "observedContractFieldSlotsUpperBound1s": (
                    int(observations.get("1s", 0)) * len(PREDICTION_NORMALIZED_STATE_FIELDS)
                ),
            },
        })

    opened_global = [row for row in global_markets if is_open(row)]
    asset_snapshot = {
        key: sum(int(row["snapshot"][key]) for row in asset_rows)
        for key in (
            "openSeries", "openEvents", "openContracts", "persistentSeriesFieldChannels",
            "openContractFieldSlots", "potentialContractMetadataFieldSlots",
            "observedContracts1m",
            "observedContracts1s", "observedContractFieldSlotsUpperBound1m",
            "observedContractFieldSlotsUpperBound1s",
        )
    }
    return {
        "provider": "kalshi",
        "window": {
            "start": "2026-07-18T00:00:00Z",
            "endExclusive": "2026-08-17T00:00:00Z",
        },
        "snapshot": snapshot.isoformat().replace("+00:00", "Z"),
        "observationOrigin": datetime.fromtimestamp(
            observation_origin_ms / 1_000, timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "identity": {
            "persistent": "series",
            "event": "one dated/settling occurrence of a series; not persistent across resolutions",
            "contract": "one outcome or strike inside an event; not persistent across resolutions",
            "continuousFeatureUnit": "subject asset plus provider series id",
            "dynamicInstanceUnit": "event id plus contract id at the causal origin",
        },
        "fieldSchemas": {
            "contractMetadata": {
                "fieldDefinitions": 25,
                "fields": [
                    "id", "provider", "series", "seriesTitle", "category", "frequency",
                    "scope", "fast", "ticker", "eventTicker", "title", "subtitle",
                    "yesSubtitle", "noSubtitle", "createdTime", "metadataUpdatedTime",
                    "mutableMetadataAvailableAt", "openTime", "closeTime",
                    "expectedExpirationTime", "expirationTime", "strikeType", "floorStrike",
                    "capStrike", "functionalStrike",
                ],
                "use": "identity, contract interpretation, strike normalization, and causal availability",
            },
            "rawTrade": {
                "baseFieldDefinitions": 5,
                "fields": ["timestamp", "yes-price", "contract-count", "taker-outcome-side", "block-trade-flag"],
            },
            "completedMinuteCandle": {
                "baseFieldDefinitions": 10,
                "fields": [
                    "yes-bid-close", "yes-ask-close", "trade-open-probability",
                    "trade-low-probability", "trade-high-probability", "trade-close-probability",
                    "trade-mean-probability", "previous-trade-probability", "contract-volume",
                    "open-interest",
                ],
            },
            "tradeState": {
                "subfeatureDefinitions": 10,
                "fields": [
                    "last-probability", "last-trade-age", "interval-trade-count",
                    "interval-contract-volume", "interval-vwap-probability",
                    "interval-min-probability", "interval-max-probability",
                    "interval-probability-change", "interval-yes-taker-fraction", "time-to-close",
                ],
            },
            "candleState": {
                "subfeatureDefinitions": 13,
                "fields": [
                    "yes-bid-close", "yes-ask-close", "quote-mid-probability", "quoted-spread",
                    "trade-open-probability", "trade-low-probability", "trade-high-probability",
                    "trade-close-probability", "trade-mean-probability",
                    "previous-trade-probability", "contract-volume", "open-interest", "time-to-close",
                ],
            },
            "normalizedForCurrentAxis": {
                "subfeatureDefinitions": len(PREDICTION_NORMALIZED_STATE_FIELDS),
                "fields": list(PREDICTION_NORMALIZED_STATE_FIELDS),
                "nullable": True,
            },
        },
        "assetSpecific": {
            "recognizedAssets": sum(row["recognizedAsset"] for row in asset_rows),
            "subjectBuckets": len(asset_rows),
            "unclassifiedOrMultiAssetBucket": "CRYPTO-OTHER",
            "window": {
                "persistentSeries": sum(int(row["window"]["persistentSeries"]) for row in asset_rows),
                "events": sum(int(row["window"]["events"]) for row in asset_rows),
                "contracts": sum(int(row["window"]["contracts"]) for row in asset_rows),
            },
            "snapshot": asset_snapshot,
            "byAsset": asset_rows,
        },
        "generalEvents": {
            "window": {
                "persistentSeries": len({row["series"] for row in global_markets}),
                "events": len({row["eventTicker"] for row in global_markets}),
                "contracts": len(global_markets),
            },
            "snapshot": {
                "openSeries": len({row["series"] for row in opened_global}),
                "openEvents": len({row["eventTicker"] for row in opened_global}),
                "openContracts": len(opened_global),
                "persistentSeriesFieldChannels": (
                    len({row["series"] for row in opened_global}) * len(PREDICTION_NORMALIZED_STATE_FIELDS)
                ),
                "openContractFieldSlots": len(opened_global) * len(PREDICTION_NORMALIZED_STATE_FIELDS),
                "potentialContractMetadataFieldSlots": len(opened_global) * 25,
                "observedContracts1m": int(observed_general.get("1m", 0)),
                "observedContracts1s": int(observed_general.get("1s", 0)),
                "observedContractFieldSlotsUpperBound1m": (
                    int(observed_general.get("1m", 0)) * len(PREDICTION_NORMALIZED_STATE_FIELDS)
                ),
                "observedContractFieldSlotsUpperBound1s": (
                    int(observed_general.get("1s", 0)) * len(PREDICTION_NORMALIZED_STATE_FIELDS)
                ),
            },
        },
    }


def source_feature_inventory(assets: dict[str, list[str]]) -> dict[str, Any]:
    minute = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/manifest.json")
    derivatives = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/derivatives-manifest.json")
    book = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/book-depth-manifest.json")
    kline_fields = [field for field in minute["layout"]["columns"] if field != "observed"]
    metric_fields = [field for field in derivatives["metrics"]["columns"] if field != "observed"]
    book_fields = [field for field in book["columns"] if field != "observed"]
    prediction = prediction_market_source_inventory()

    def field_set(
        identifier: str, fields: list[str], asset_count: int, *, cadence: str, level: str
    ) -> dict[str, Any]:
        return {
            "id": identifier,
            "cadence": cadence,
            "level": level,
            "fieldDefinitions": len(fields),
            "fields": fields,
            "assetsAvailable": asset_count,
            "expandedAssetFieldBindings": len(fields) * asset_count,
        }

    spot_sets = [
        field_set("completed-spot-kline-1s", kline_fields, len(assets["spot1s95"]), cadence="1s", level="base"),
        field_set("completed-spot-kline-1m", kline_fields, len(assets["spot1m99"]), cadence="1m", level="base"),
    ]
    futures_sets = [
        field_set("completed-usdm-kline-1m", kline_fields, len(assets["usdm1m99"]), cadence="1m", level="base"),
        field_set("published-usdm-positioning", metric_fields, len(assets["usdmMetrics95"]), cadence="5m", level="base"),
        field_set("settled-usdm-funding", ["funding-rate"], len(assets["usdmFunding"]), cadence="funding-event", level="base"),
        field_set("usdm-book-depth-bands", book_fields, len(assets["usdmBook95"]), cadence="5m", level="derived"),
    ]
    option_sets = [
        field_set("deribit-option-surface", list(OPTION_SURFACE_FIELDS), 1, cadence="live-snapshot", level="derived")
    ]
    return {
        "countingSemantics": {
            "fieldDefinition": "one named field in one source schema/cadence, independent of how many assets expose it",
            "assetBinding": "one field definition instantiated for one asset",
            "predictionPersistentFeature": "one asset/general subject plus Kalshi series id",
            "predictionDynamicFeature": "one active event/contract field value at a causal origin",
            "candidateCoordinate": "a raw or derived model column; reported separately from this source ledger",
        },
        "assetSpecific": {
            "categories": [
                {"id": "spot-stats", "fieldSets": spot_sets},
                {"id": "futures-stats", "fieldSets": futures_sets},
                {"id": "options-stats", "fieldSets": option_sets, "history": "live-only; no robust 30-day option archive"},
                {"id": "prediction-market-stats", "hierarchy": prediction["assetSpecific"] if prediction else None},
            ],
        },
        "general": {
            "categories": [
                {"id": "global-event-prediction-market-stats", "hierarchy": prediction["generalEvents"] if prediction else None},
            ],
        },
        "predictionMarket": prediction,
    }


def infer_subject(row: dict[str, Any]) -> str:
    asset = str(row["asset"])
    if asset != "BTC/general baseline inventory":
        return asset
    feature = str(row["feature"])
    for prefix in ("eth", "sol", "bnb", "doge"):
        if feature.startswith(prefix + "-"):
            return prefix.upper()
    return "BTC"


def normalize_representative_formula(feature: str, subject: str, scope: str = "") -> str:
    value = feature.lower()
    prefix = subject.lower() + "-"
    if value.startswith(prefix) and subject in {"ETH", "SOL", "BNB", "DOGE"}:
        value = value[len(prefix):]
    funding_aliases = {
        "funding-rate": "funding-level",
        "funding-absolute-rate": "funding-absolute-level",
        "funding-change": "funding-change-1",
        "funding-age-hours": "funding-age",
    }
    value = funding_aliases.get(value, value)
    if value.startswith("efficiency-ratio-"):
        value = value.replace("efficiency-ratio-", "path-efficiency-", 1)
    if scope == "replicated cross-market inventory":
        # The compact fast exporter predates the exhaustive technical bank.
        # Its RSI(2) uses EMA alpha 2/3 (not Wilder alpha 1/2), and its EMA8
        # change is not divided by the eight-second horizon.  Those are useful
        # coordinates, but they are not aliases of the later exact formulas.
        value = {
            "rsi-2s": "rsi-2s-ema-alpha-2-over-3",
            "ema-slope-8s-8s": "ema-log-change-8s-ema-over-8s",
        }.get(value, value)
    return value


def representative_cadence(feature: str, lookback: str) -> str:
    if re.search(r"(?:^|-)\d+s(?:-|$)", feature) or "second" in lookback:
        return "1s"
    if "funding" in feature:
        return "funding-event"
    if any(token in feature for token in ("open-interest", "long-short", "top-account", "top-position")):
        return "5m"
    return "1m"


def representative_venue(row: dict[str, Any], subject: str) -> str:
    feature = str(row["feature"])
    source = str(row["source"])
    if feature.startswith("spot-flow-") or feature.startswith("spot-book-"):
        return "binance-spot"
    if representative_cadence(feature, str(row.get("lookback", ""))) == "1s":
        return "binance-spot"
    if "funding" in feature or "USD-M" in source or feature.startswith(("futures-", "open-interest-", "top-")):
        return "binance-usdm"
    if subject in {"ETH", "SOL", "BNB", "DOGE"}:
        return "binance-preferred"
    return "binance-preferred"


def add_representative(registry: FeatureRegistry) -> None:
    artifact = load_json("data/benchmarks/binance-cross-asset-component-feature-bases.json")
    for row in artifact["featureCatalog"]:
        subject = infer_subject(row)
        feature = str(row["feature"])
        source_policy = "binance-archive"
        if "book" in feature and row["scope"] == "existing BTC/general inventory":
            source_policy = "live-only"
        elif any(token in feature for token in ("futures", "open-interest", "funding", "long-short", "top-")):
            source_policy = "binance-futures-archive"
        registry.add(
            inventory="representative-cross-asset",
            subject=subject,
            subject_kind="asset",
            venue=representative_venue(row, subject),
            cadence=representative_cadence(feature, str(row.get("lookback", ""))),
            formula=normalize_representative_formula(feature, subject, str(row["scope"])),
            alias=str(row["id"]),
            family=str(row["family"]),
            source_policy=source_policy,
            construction=str(row.get("construction", "")),
            lookback=str(row.get("lookback", "")),
            delay=str(row.get("delay", "")),
        )
    registry.inventory_notes["representative-cross-asset"] = (
        "The 31,043-coordinate study catalog: 147 existing BTC/general entries plus "
        "source-supported replicated entries."
    )


def add_dense_minute(registry: FeatureRegistry, subjects: Iterable[str]) -> None:
    audit = load_json("data/benchmarks/dense-lagged-indicator-audit.json")
    grid = audit["grid"]
    formulas: list[tuple[str, str]] = []
    for period in grid["rsiPeriodsMinutes"]:
        formulas.append(("RSI", f"rsi-{period}m"))
    for period in grid["emaPeriodsMinutes"]:
        formulas.append(("EMA value", f"ema-distance-{period}m"))
        for horizon in grid["emaDifferenceHorizonsMinutes"]:
            formulas.append(("EMA slope", f"ema-slope-{period}m-{horizon}m"))
            formulas.append(("EMA acceleration", f"ema-acceleration-{period}m-{horizon}m"))
    base = formulas
    for lag in grid["signalLagsMinutes"]:
        for family, formula in base:
            lagged = formula if lag == 0 else f"{formula}-lag-{lag}m"
            for subject in subjects:
                registry.add(
                    inventory="dense-minute-indicators",
                    subject=subject,
                    subject_kind="asset",
                    venue="binance-preferred",
                    cadence="1m",
                    formula=lagged,
                    alias=lagged,
                    family=family,
                    source_policy="candle-derived",
                    lookback="recursive",
                    delay="through origin",
                )
    registry.inventory_notes["dense-minute-indicators"] = (
        "All 3,471 EMA/RSI period, difference-horizon, and whole-signal-lag variants; "
        "the 13 requested lags are already included."
    )


def normalize_second_formula(row: dict[str, Any]) -> str:
    feature_id = str(row["id"])
    family = str(row["family"])
    if family == "RSI":
        return f"rsi-{row['period']}s"
    if family == "EMA":
        return f"ema-distance-{row['period']}s"
    if family == "EMA slope":
        return f"ema-slope-{row['period']}s-{row['horizon']}s"
    if family == "EMA acceleration":
        return f"ema-acceleration-{row['period']}s-{row['horizon']}s"
    return feature_id.replace("macd-hist-", "macd-histogram-") + "-1s-cadence"


def add_second_technical(registry: FeatureRegistry, subjects: Iterable[str]) -> None:
    audit = load_json("data/benchmarks/technical-indicator-predictiveness.json")
    for row in audit["signals"]:
        formula = normalize_second_formula(row)
        for subject in subjects:
            registry.add(
                inventory="second-technical",
                subject=subject,
                subject_kind="asset",
                venue="binance-spot",
                cadence="1s",
                formula=formula,
                alias=str(row["id"]),
                family=str(row["family"]),
                source_policy="candle-derived",
                lookback="recursive",
                delay="through origin",
            )
    registry.inventory_notes["second-technical"] = (
        "All 166 one-second RSI, EMA value/slope/acceleration, and MACD coordinates."
    )


def add_spectral(registry: FeatureRegistry, subjects_by_cadence: dict[str, Iterable[str]]) -> None:
    manifest = load_json("data/runtime-cache/fourier-return-features/manifest.json")
    for dataset in manifest["datasets"]:
        cadence = str(dataset["id"])
        for row in dataset["features"]:
            for subject in subjects_by_cadence[cadence]:
                registry.add(
                    inventory=f"spectral-{cadence}",
                    subject=subject,
                    subject_kind="asset",
                    venue="binance-spot" if cadence == "1s" else "binance-preferred",
                    cadence=cadence,
                    formula=str(row["id"]),
                    alias=str(row["id"]),
                    family=str(row["family"]),
                    source_policy="candle-derived",
                    construction=str(row.get("transform", "")),
                    lookback=str(row.get("lookback", "")),
                    delay=str(row.get("delay", "")),
                )
        registry.inventory_notes[f"spectral-{cadence}"] = (
            "All 102 Fourier, fractional-Fourier, Haar, Morlet, and path-efficiency coordinates."
        )


def normalize_long_formula(row: dict[str, Any]) -> str:
    feature_id = str(row["id"])
    if re.fullmatch(r"rsi-\d+", feature_id):
        return feature_id + "m"
    match = re.fullmatch(r"ema-(slope|acceleration)-(\d+)-(\d+)", feature_id)
    if match:
        return f"ema-{match.group(1)}-{match.group(2)}m-{match.group(3)}m"
    return feature_id


def add_long_endogenous(registry: FeatureRegistry, subjects: Iterable[str]) -> None:
    manifest = load_json("data/runtime-cache/global-feature-basis/manifest.json")
    minute = next(row for row in manifest["datasets"] if row["id"] == "1m")
    for row in minute["features"]:
        feature_id = str(row["id"])
        is_general = feature_id.startswith("utc-")
        targets = ["GLOBAL"] if is_general else subjects
        for subject in targets:
            registry.add(
                inventory="long-endogenous",
                subject=subject,
                subject_kind="general" if is_general else "asset",
                venue="calendar" if is_general else "binance-preferred",
                cadence="known" if is_general else "1m",
                formula=normalize_long_formula(row),
                alias=feature_id,
                family=str(row["family"]),
                source_policy="candle-derived",
                construction=str(row.get("parameters", "")),
                lookback=str(row.get("lookback", "")),
                delay=str(row.get("delay", "")),
            )
    registry.inventory_notes["long-endogenous"] = (
        "The 34-coordinate minute ledger, with 32 asset-derived templates expanded by asset "
        "and the two UTC calendar coordinates stored once."
    )


def selected_public_group(artifact: dict[str, Any], prefix: str) -> dict[str, Any]:
    preferred = [f"{prefix}-1m", f"{prefix}-5m", f"{prefix}-15m"]
    for group_id in preferred:
        match = next((row for row in artifact["groups"] if row["id"] == group_id), None)
        if match is not None:
            return match
    raise KeyError(prefix)


def public_subject(feature_id: str, source: str) -> tuple[str, str, str]:
    match = re.match(r"cross-(eth|sol|bnb|doge)-", feature_id)
    if match:
        return match.group(1).upper(), "asset", "binance-preferred"
    if source in {"global-macro", "cboe-vix"}:
        return "GLOBAL", "general", source
    if source in {"deribit-dvol", "coinmetrics", "mempool-proxy"}:
        return "BTC", "asset", source
    if source == "community-daily":
        return "GLOBAL-CRYPTO", "general", source
    return "BTC", "asset", source


def normalize_cross_public(feature_id: str, subject: str) -> str:
    prefix = f"cross-{subject.lower()}-"
    value = feature_id[len(prefix):] if feature_id.startswith(prefix) else feature_id
    value = value.replace("volatility-", "realized-volatility-")
    return value


def add_public_external(registry: FeatureRegistry) -> None:
    artifact = load_json("data/benchmarks/public-external-feature-information.json")
    inventories = {
        "cross-market": "cross-market-public",
        "global-macro": "global-macro",
        "deribit-dvol": "dvol",
        "cboe-vix": "vix",
        "coinmetrics": "coinmetrics",
        "community-daily": "community-flows",
        "mempool-proxy": "mempool-proxy",
    }
    for prefix, inventory in inventories.items():
        group = selected_public_group(artifact, prefix)
        for row in group["ranked"]:
            feature_id = str(row["id"])
            source = str(row.get("source", prefix))
            subject, subject_kind, venue = public_subject(feature_id, source)
            policy = "official-public-slow"
            if prefix == "cross-market":
                policy = "binance-archive"
            elif prefix in {"deribit-dvol", "cboe-vix"}:
                policy = "official-market-archive"
            elif source in {"coinmetrics", "community-daily", "mempool-proxy"}:
                policy = "community-revised"
            registry.add(
                inventory=inventory,
                subject=subject,
                subject_kind=subject_kind,
                venue=venue,
                cadence="1m" if prefix == "cross-market" else "slow",
                formula=normalize_cross_public(feature_id, subject),
                alias=feature_id,
                family=str(row["family"]),
                source_policy=policy,
                lookback=str(row.get("lookback", "")),
                delay="publication/source delay",
            )


def funding_required_events(feature_id: str) -> int:
    match = re.search(r"-(1|3|9|21|90)$", feature_id)
    return int(match.group(1)) if match else 1


def add_full_funding(registry: FeatureRegistry) -> None:
    artifact = load_json("data/benchmarks/public-external-feature-information.json")
    group = selected_public_group(artifact, "binance-funding")
    derivatives = load_json("data/runtime-cache/binance-cross-asset-1m-basis-30d/derivatives-manifest.json")
    events = {row["asset"]: int(row["funding"]["events"]) for row in derivatives["assets"] if row.get("funding")}
    for feature in group["ranked"]:
        feature_id = str(feature["id"])
        required = funding_required_events(feature_id)
        for subject, observed in events.items():
            if observed < required:
                continue
            registry.add(
                inventory="full-funding-grid",
                subject=subject,
                subject_kind="asset",
                venue="binance-usdm",
                cadence="funding-event",
                formula=feature_id,
                alias=feature_id,
                family=str(feature["family"]),
                source_policy="binance-futures-archive",
                lookback=str(feature.get("lookback", "")),
                delay="one minute after settlement",
            )
    registry.inventory_notes["full-funding-grid"] = (
        "The complete 18-template settled-funding grid, availability-filtered by each market's event count."
    )


def add_tardis(registry: FeatureRegistry) -> None:
    artifact = load_json("data/benchmarks/tardis-monthly-sample-information.json")
    group = artifact["groups"][0]
    for row in group["ranked"]:
        registry.add(
            inventory="tardis-cross-venue",
            subject="BTC",
            subject_kind="asset",
            venue="tardis-multivenue",
            cadence="1s",
            formula=str(row["id"]),
            alias=str(row["id"]),
            family=str(row["family"]),
            source_policy="sampled-history",
            lookback=str(row.get("lookback", "")),
            delay="through completed sample second",
        )


def add_prediction_markets(registry: FeatureRegistry) -> None:
    manifest_path = ROOT / "data/runtime-cache/prediction-market-candidate-axis-30d/manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["coordinates"]:
        parts = str(row["id"]).split("/", 4)
        if len(parts) != 5:
            raise ValueError(f"Invalid prediction-market coordinate: {row['id']}")
        subject_kind, subject, venue, cadence, formula = parts
        registry.add(
            inventory="prediction-markets",
            subject=subject,
            subject_kind=subject_kind,
            venue=venue,
            cadence=cadence,
            formula=formula,
            alias=str(row["id"]),
            family=str(row.get("family", "prediction-market-state")),
            source_policy="official-market-archive",
            construction=f"causal dynamic-set summary ({row.get('bucket', '')})",
            lookback=str(row.get("metric", "")),
            delay="strictly before origin for 1s trades; completed candle at origin for 1m",
        )
    registry.inventory_notes["prediction-markets"] = (
        "Causal Kalshi summaries over every discovered relevant traded asset and global-event market."
    )


def add_live_fast(registry: FeatureRegistry) -> None:
    manifest = load_json("data/runtime-cache/live-component-feature-bases/manifest.json")
    for row in manifest["features"]:
        registry.add(
            inventory="fast-live",
            subject="BTC",
            subject_kind="asset",
            venue=str(row["source"]),
            cadence="1s",
            formula=str(row["id"]),
            alias=str(row["id"]),
            family=str(row["family"]),
            source_policy="live-only",
            construction=str(row.get("construction", "")),
            lookback=str(row.get("lookback", "")),
            delay="through completed source bucket",
        )


def add_declared_live_ledgers(registry: FeatureRegistry) -> None:
    for feature_id in OPTION_SURFACE_FIELDS:
        registry.add(
            inventory="deribit-option-surface",
            subject="BTC",
            subject_kind="asset",
            venue="deribit-options",
            cadence="live-snapshot",
            formula=feature_id,
            alias=feature_id,
            family="option surface",
            source_policy="live-only",
            delay="through latest completed snapshot",
        )
    gdelt_ids = (
        "crypto-terms-per-million", "story-count", "source-count", "mean-tone",
        "mean-positive", "mean-negative", "mean-polarity", "news-source-age",
    )
    for feature_id in gdelt_ids:
        registry.add(
            inventory="gdelt-news",
            subject="GLOBAL-CRYPTO",
            subject_kind="general",
            venue="gdelt",
            cadence="15m",
            formula=feature_id,
            alias=feature_id,
            family="news",
            source_policy="official-public-slow",
            delay="through latest completed 15m bucket",
        )


def build_registry() -> tuple[FeatureRegistry, dict[str, list[str]]]:
    assets = asset_sets()
    registry = FeatureRegistry()
    add_representative(registry)
    add_dense_minute(registry, assets["preferred1m99"])
    add_second_technical(registry, assets["spot1s95"])
    add_spectral(registry, {"1m": assets["preferred1m99"], "1s": assets["spot1s95"]})
    add_long_endogenous(registry, assets["preferred1m99"])
    add_public_external(registry)
    add_full_funding(registry)
    add_tardis(registry)
    add_prediction_markets(registry)
    add_live_fast(registry)
    add_declared_live_ledgers(registry)
    return registry, assets
