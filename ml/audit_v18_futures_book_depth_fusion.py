"""Leakage-safe USD-M book-depth residual audit beyond frozen v18.

The official irregular book-depth snapshots are decoded without repair.  The
caller derives a causal usable/corrupt mask against the latest *fully closed*
USD-M one-minute candle: source second 59 still sees the preceding candle and
source second 60 may see minute zero.  Implied-VWAP geometry errors, gross
completed-close mismatches, and causally detectable stuck matrices are masked,
never repaired or backfilled as observations.

Every table configuration is selected on training only.  Frozen v18 is run
once on development validation; the first half selects one pre-fitted regime
and log-ratio scalar, 60 rows are embargoed, and only that fixed winner feeds
the raw T=.01 KL >=.002 development screen.  The sealed test split is never
opened.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

from audit_causal_oracle_ohlcv_predictability import (
    ACTION_COUNT,
    DAY_ROWS,
    FEATURE_NAMES as OHLCV_FEATURE_NAMES,
    causal_ohlcv_features,
    completed_candle_windows,
    completed_close_windows,
    mean_kl_indexed,
    mean_kl_stacked_indexed,
    normalized_mean,
    smoothed_table,
    split_and_purge,
    sufficient_table,
    utc_day_start_ms,
)
from audit_v18_calendar_fusion import (
    convex_probability_fusion,
    select_scalar,
    softmax,
    validate_preserved_checkpoint_without_test_access,
)
from audit_v18_futures_basis_fusion import (
    FUTURES_KLINE_ENCODING,
    FUTURES_KLINE_REFERENCE_DIR,
    FUTURES_KLINE_SCHEMA,
    FUTURES_KLINE_SOURCE_ROOT,
    FuturesKlineDayCache,
    required_feature_days,
    validate_reference_coverage,
)
from audit_v18_futures_metrics_fusion import (
    CALIBRATION_FRACTION,
    DEVELOPMENT_SCREEN_GATE,
    EMBARGO_ROWS,
    EXPECTED_CHECKPOINT_EPOCH,
    MinuteCandleDayCache,
    PairedRegime,
    embargoed_split_metrics,
    mean_kl_dense,
    primary_development_screen,
    select_first_half_complementarity,
    validate_segment_clock_alignment,
)
from audit_v18_ohlcv_fusion import (
    AuditedMarketCloseCache,
    AuditedOracleTargetCache,
    fusion_metrics,
    log_ratio_feature_fusion,
    validation_timestamps,
)
from audit_v18_on_v28_validation_rows import collect_sequence_core_rows
from audit_v18_trade_flow_fusion import (
    CORPUS_SPLIT_CONTRACT,
    validate_corpus_split_contract,
    validate_runtime_split_assignment,
)
from evaluate_joint_price_oracle_actions import resolve_device
from oracle_futures_basis_features import FUTURES_KLINE_COLUMNS
from oracle_futures_book_depth_features import (
    BOOK_DEPTH_BANDS,
    BOOK_DEPTH_DAY_COLUMNS,
    BOOK_DEPTH_VALUE_COLUMNS,
    FUTURES_BOOK_DEPTH_FEATURE_NAMES,
    causal_futures_book_depth_features,
)
from trading_storage import (
    load_torch_checkpoint,
    read_derivatives_book_depth_columns,
    read_shard_array,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    CausalOracleDataset,
    CausalSegment,
    build_model,
    resolve,
    resolve_training_config,
    validate_plan,
)


DEFAULT_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
)
BOOK_DEPTH_REFERENCE_DIR = Path(
    "data/market/immutable/refs/derivatives-book-depth/"
    "usdm-futures/btcusdt"
)
BOOK_DEPTH_ENCODING = "derivatives-book-depth-columnar-v1"
BOOK_DEPTH_SCHEMA = "binance-usdm-futures-book-depth-v1"
BOOK_DEPTH_SOURCE_ROOT = (
    "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT"
)
EXPECTED_SOURCE_DAYS = 420
EXPECTED_AVAILABLE_REFERENCE_DAYS = 368
EXPECTED_UNAVAILABLE_SOURCE_DAYS = 52
SEALED_TEST_START = "2026-06-24"
SEALED_TEST_END = "2026-07-23"
MINUTE_MS = 60_000
GROSS_CLOSE_MISMATCH_RATIO = 1.15
GROSS_CLOSE_MISMATCH_LOG = math.log(GROSS_CLOSE_MISMATCH_RATIO)
STUCK_MATRIX_SECONDS = 300
PRIOR_STRENGTHS = (64.0, 256.0, 1_024.0)

_UNAVAILABLE_RANGES = (
    ("2021-09-07", "2021-09-14"),
    ("2021-10-18", "2021-10-21"),
    ("2021-12-13", "2021-12-20"),
    ("2022-05-13", "2022-05-20"),
    ("2022-06-06", "2022-06-21"),
    ("2022-07-27", "2022-08-03"),
)


def _date_range(first: str, last: str) -> tuple[str, ...]:
    current = date.fromisoformat(first)
    end = date.fromisoformat(last)
    values: list[str] = []
    while current <= end:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


OFFICIAL_UNAVAILABLE_DAYS = tuple(
    value
    for first, last in _UNAVAILABLE_RANGES
    for value in _date_range(first, last)
)
OFFICIAL_UNAVAILABLE_SET = frozenset(OFFICIAL_UNAVAILABLE_DAYS)
OFFICIAL_UNAVAILABLE_SHA256 = hashlib.sha256(
    ("\n".join(OFFICIAL_UNAVAILABLE_DAYS) + "\n").encode("utf-8")
).hexdigest()
if len(OFFICIAL_UNAVAILABLE_SET) != EXPECTED_UNAVAILABLE_SOURCE_DAYS:
    raise RuntimeError("book-depth official-unavailable contract changed")


FEATURE_NAMES = OHLCV_FEATURE_NAMES + FUTURES_BOOK_DEPTH_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
CATEGORICAL_FEATURE_EDGES = {
    name: np.asarray((0.5,), dtype=np.float64)
    for name in (
        "bookDepthCurrentSecondObserved",
        "bookDepthCurrentSecondCorrupt",
        "bookDepthBand0p2ValueAvailable",
    )
}


# Each signed book feature has its binned absolute-value counterpart in the
# paired control.  Source/schema/age/activity/OHLCV controls are deliberately
# present in every regime; separate quantile bins can still refine magnitude,
# so these are complementarity screens rather than exact sign attribution.
REGIMES = (
    PairedRegime(
        "near-imbalance",
        (
            "return15m", "rmsReturn15m", "meanLogRange15m",
            "absBaseBidAskImbalance1pct",
            "absNotionalBidAskImbalance1pct",
            "bookDepthUsableValueAge24h",
            "bookDepthUsableSnapshotUpdatesPerMinute15m",
            "bookDepthStaleSnapshotFraction15m",
            "bookDepthBand0p2ValueAvailable",
        ),
        (
            "baseBidAskImbalance1pct",
            "notionalBidAskImbalance1pct",
        ),
        (4, 4, 3, 3, 3, 3, 3, 3, 2),
        (3, 3),
    ),
    PairedRegime(
        "curve-rings-convexity",
        (
            "return60m", "rmsReturn60m", "meanLogRange60m",
            "absBaseMarginalRing1pctTo2pctBidAskImbalance",
            "absNotionalConvexityDifference",
            "baseNearConcentrationMean", "notionalConvexityMean",
            "bookDepthUsableValueAge24h",
            "bookDepthRawSnapshotUpdatesPerMinute15m",
            "bookDepthStaleSnapshotFraction15m",
            "bookDepthBand0p2ValueAvailable",
        ),
        (
            "baseMarginalRing1pctTo2pctBidAskImbalance",
            "notionalConvexityDifference",
        ),
        (5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2),
        (3, 3),
    ),
    PairedRegime(
        "depth-change",
        (
            "return15m", "return60m", "rmsReturn60m",
            "absBaseTotal1pctLogChange15m",
            "absNotionalTotal5pctLogChange15m",
            "bookDepthUsableValueAge24h",
            "bookDepthUsableSnapshotUpdatesPerMinute15m",
            "bookDepthStaleSnapshotFraction15m",
            "bookDepthBand0p2ValueAvailable",
        ),
        (
            "baseTotal1pctLogChange15m",
            "notionalTotal5pctLogChange15m",
        ),
        (4, 5, 4, 3, 3, 3, 3, 3, 2),
        (3, 3),
    ),
    PairedRegime(
        "update-staleness",
        (
            "return60m", "rmsReturn60m", "logVolume15mVs60m",
            "bookDepthRawSnapshotAge24h",
            "bookDepthRawSnapshotUpdatesPerMinute15m",
            "bookDepthBand0p2ValueAvailable",
            "bookDepthCurrentSecondObserved",
            "bookDepthCurrentSecondCorrupt",
        ),
        (
            "bookDepthUsableSnapshotUpdatesPerMinute15m",
            "bookDepthStaleSnapshotFraction15m",
        ),
        (5, 4, 3, 3, 3, 2, 2, 2),
        (3, 3),
    ),
)


@dataclass(frozen=True)
class SnapshotQuality:
    snapshot_count: int
    usable_count: int
    corrupt_count: int
    missing_completed_candle_count: int
    geometry_corrupt_count: int
    gross_close_mismatch_count: int
    stuck_matrix_count: int


@dataclass(frozen=True)
class TrainSelectedRegime:
    regime: PairedRegime
    control_strength: float
    joint_strength: float
    joint_weight: float


@dataclass(frozen=True)
class FittedRegimeTables:
    selection: TrainSelectedRegime
    control_edges: tuple[np.ndarray, ...]
    control_table: np.ndarray
    control_counts: np.ndarray
    joint_edges: tuple[np.ndarray, ...]
    joint_table: np.ndarray
    joint_counts: np.ndarray


@dataclass
class AccessLog:
    target_files: set[Path]
    candle_files: set[Path]
    futures_kline_files: set[Path]
    book_depth_files: set[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON artifact path under data/runtime/logs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = audit_v18_futures_book_depth_fusion(
        arguments.plan,
        requested_device=arguments.device,
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(payload, end="")
        return
    repo_root = Path(__file__).resolve().parents[1]
    output = arguments.output
    if not output.is_absolute():
        output = repo_root / output
    output = output.resolve()
    runtime_logs = (repo_root / "data/runtime/logs").resolve()
    if output.parent != runtime_logs:
        raise ValueError("audit output must be directly under data/runtime/logs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale audit temporary file: {temporary}")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(f"Wrote audit artifact: {output}", file=sys.stderr, flush=True)


def validate_book_depth_namespace(
    root: Path,
    required_days: set[str],
    *,
    sealed_test_start: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate exactly 368 refs plus the fixed 52 non-payload source days.

    This preflight reads directory entries only.  It must complete before any
    source reference JSON, target payload, or model checkpoint is opened.
    """
    if len(required_days) != EXPECTED_SOURCE_DAYS:
        raise ValueError(
            f"book-depth scope has {len(required_days)} days, "
            f"expected {EXPECTED_SOURCE_DAYS}"
        )
    if any(value >= sealed_test_start for value in required_days):
        raise ValueError("book-depth scope crosses sealed-test boundary")
    unavailable = tuple(sorted(required_days & OFFICIAL_UNAVAILABLE_SET))
    if unavailable != tuple(sorted(OFFICIAL_UNAVAILABLE_DAYS)):
        raise ValueError(
            "book-depth official-unavailable intersection differs from contract"
        )
    available = tuple(sorted(required_days - OFFICIAL_UNAVAILABLE_SET))
    if len(available) != EXPECTED_AVAILABLE_REFERENCE_DAYS:
        raise ValueError(
            f"book-depth available scope has {len(available)} days, "
            f"expected {EXPECTED_AVAILABLE_REFERENCE_DAYS}"
        )
    try:
        entries = tuple(root.iterdir())
    except FileNotFoundError as error:
        raise FileNotFoundError("book-depth reference namespace is missing") from error
    invalid_entries = sorted(
        entry.name
        for entry in entries
        if not entry.is_file()
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}\.json", entry.name) is None
    )
    if invalid_entries:
        raise ValueError(
            "book-depth namespace contains invalid entries: "
            + ",".join(invalid_entries[:3])
        )
    actual = {entry.stem for entry in entries}
    sealed = sorted(
        value
        for value in actual
        if SEALED_TEST_START <= value <= SEALED_TEST_END
    )
    if sealed:
        raise ValueError("book-depth namespace contains sealed-test references")
    missing = sorted(set(available) - actual)
    unexpected = sorted(actual - set(available))
    if missing or unexpected:
        raise ValueError(
            "book-depth namespace differs from exact 368-reference contract: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return available, unavailable


def validate_book_depth_reference_coverage(
    root: Path,
    segments: dict[str, list[CausalSegment]],
    *,
    sealed_test_start: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    required = required_feature_days(segments)
    available, unavailable = validate_book_depth_namespace(
        root,
        required,
        sealed_test_start=sealed_test_start,
    )
    return tuple(sorted(required)), available, unavailable


def empty_book_depth_day() -> dict[str, np.ndarray]:
    band_count = len(BOOK_DEPTH_BANDS)
    return {
        "timestampOffsetSeconds": np.empty(0, dtype=np.int64),
        "schemaBandCount": np.empty(0, dtype=np.uint8),
        **{
            name: np.empty((0, band_count), dtype=np.float64)
            for name in BOOK_DEPTH_VALUE_COLUMNS
        },
        "bandAvailable": np.empty((0, band_count), dtype=bool),
    }


def completed_futures_close_for_snapshots(
    timestamp_offsets: np.ndarray,
    previous_close: np.ndarray,
    previous_validity: np.ndarray,
    current_close: np.ndarray,
    current_validity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candle closes available strictly before each source second."""
    offsets = np.asarray(timestamp_offsets)
    previous_close = np.asarray(previous_close, dtype=np.float64)
    current_close = np.asarray(current_close, dtype=np.float64)
    previous_validity = np.asarray(previous_validity, dtype=bool)
    current_validity = np.asarray(current_validity, dtype=bool)
    if offsets.ndim != 1 or offsets.dtype.kind not in ("i", "u") \
            or bool((offsets >= 86_400).any()) \
            or (offsets.dtype.kind == "i" and bool((offsets < 0).any())):
        raise ValueError("invalid book-depth timestamp offsets")
    if any(value.shape != (DAY_ROWS,) for value in (
        previous_close,
        previous_validity,
        current_close,
        current_validity,
    )):
        raise ValueError("completed-candle arrays must each span one UTC day")
    merged_close = np.concatenate((previous_close, current_close))
    merged_valid = np.concatenate((previous_validity, current_validity))
    # Source second [0,59] precedes current minute zero's close.  At source
    # second 60, minute zero closed one millisecond earlier and is eligible.
    indexes = DAY_ROWS - 1 + offsets.astype(np.int64, copy=False) // 60
    closes = merged_close[indexes]
    valid = merged_valid[indexes] & np.isfinite(closes) & (closes > 0)
    return closes, valid, indexes


def classify_book_depth_snapshots(
    raw_day: dict[str, np.ndarray],
    previous_kline_values: dict[str, np.ndarray],
    previous_kline_validity: np.ndarray,
    current_kline_values: dict[str, np.ndarray],
    current_kline_validity: np.ndarray,
) -> tuple[dict[str, np.ndarray], SnapshotQuality]:
    """Add causal usable/corrupt masks without changing any source value."""
    expected = {
        "timestampOffsetSeconds",
        "schemaBandCount",
        *BOOK_DEPTH_VALUE_COLUMNS,
        "bandAvailable",
    }
    if set(raw_day) != expected:
        raise ValueError("raw book-depth columns differ from storage contract")
    offsets = np.asarray(raw_day["timestampOffsetSeconds"])
    schema = np.asarray(raw_day["schemaBandCount"])
    available = np.asarray(raw_day["bandAvailable"], dtype=bool)
    count = offsets.size
    if offsets.ndim != 1 or offsets.dtype.kind not in ("i", "u") \
            or schema.shape != (count,) \
            or available.shape != (count, len(BOOK_DEPTH_BANDS)):
        raise ValueError("invalid raw book-depth snapshot axis")
    matrices = {
        name: np.asarray(raw_day[name], dtype=np.float64)
        for name in BOOK_DEPTH_VALUE_COLUMNS
    }
    if any(values.shape != (count, len(BOOK_DEPTH_BANDS))
           for values in matrices.values()):
        raise ValueError("invalid raw book-depth matrix shape")
    expected_available = np.ones_like(available)
    expected_available[schema == 10, 0] = False
    if bool(((schema != 10) & (schema != 12)).any()) \
            or not np.array_equal(available, expected_available):
        raise ValueError("raw book-depth 10/12-band schema differs")
    if count == 0:
        classified = {
            **{name: np.asarray(raw_day[name]).copy() for name in raw_day},
            "snapshotUsable": np.empty(0, dtype=bool),
            "snapshotCorrupt": np.empty(0, dtype=bool),
        }
        return classified, SnapshotQuality(0, 0, 0, 0, 0, 0, 0)

    closes, candle_valid, _indexes = completed_futures_close_for_snapshots(
        offsets,
        previous_kline_values["close"],
        previous_kline_validity,
        current_kline_values["close"],
        current_kline_validity,
    )
    bid_vwap = np.divide(
        matrices["bidNotional"],
        matrices["bidDepth"],
        out=np.zeros_like(matrices["bidNotional"]),
        where=matrices["bidDepth"] > 0,
    )
    ask_vwap = np.divide(
        matrices["askNotional"],
        matrices["askDepth"],
        out=np.zeros_like(matrices["askNotional"]),
        where=matrices["askDepth"] > 0,
    )
    geometry_corrupt = np.zeros(count, dtype=bool)
    for row in range(count):
        active = available[row]
        bid = bid_vwap[row, active]
        ask = ask_vwap[row, active]
        geometry_corrupt[row] = (
            bid.size == 0
            or not np.isfinite(bid).all()
            or not np.isfinite(ask).all()
            or bool((bid <= 0).any())
            or bool((ask <= 0).any())
            or bool((bid >= ask).any())
            or bool((np.diff(bid) > 1e-12).any())
            or bool((np.diff(ask) < -1e-12).any())
        )

    # The 1% band is present in both source schemas.  A 15% ratio threshold is
    # intentionally coarse: it rejects the observed roughly-20% source issue
    # without treating normal basis or minute movement as corruption.
    common_band = 1
    gross_close_mismatch = ~candle_valid
    comparable = candle_valid & ~geometry_corrupt
    if comparable.any():
        bid_ratio = np.abs(np.log(
            bid_vwap[comparable, common_band] / closes[comparable]
        ))
        ask_ratio = np.abs(np.log(
            ask_vwap[comparable, common_band] / closes[comparable]
        ))
        gross_close_mismatch[comparable] = (
            np.maximum(bid_ratio, ask_ratio) > GROSS_CLOSE_MISMATCH_LOG
        )

    stuck = np.zeros(count, dtype=bool)
    run_start = 0
    for row in range(1, count):
        unchanged = np.array_equal(available[row], available[row - 1]) \
            and all(np.array_equal(values[row], values[row - 1])
                    for values in matrices.values())
        if not unchanged:
            run_start = row
            continue
        if int(offsets[row]) - int(offsets[run_start]) >= STUCK_MATRIX_SECONDS:
            stuck[row] = True

    corrupt = geometry_corrupt | gross_close_mismatch | stuck
    usable = ~corrupt
    classified = {
        "timestampOffsetSeconds": offsets.copy(),
        "schemaBandCount": schema.copy(),
        **{name: values.copy() for name, values in matrices.items()},
        "bandAvailable": available.copy(),
        "snapshotUsable": usable,
        "snapshotCorrupt": corrupt,
    }
    if set(classified) != set(BOOK_DEPTH_DAY_COLUMNS):
        raise RuntimeError("classified book-depth contract differs from feature API")
    quality = SnapshotQuality(
        snapshot_count=count,
        usable_count=int(np.count_nonzero(usable)),
        corrupt_count=int(np.count_nonzero(corrupt)),
        missing_completed_candle_count=int(np.count_nonzero(~candle_valid)),
        geometry_corrupt_count=int(np.count_nonzero(geometry_corrupt)),
        gross_close_mismatch_count=int(np.count_nonzero(gross_close_mismatch)),
        stuck_matrix_count=int(np.count_nonzero(stuck)),
    )
    return classified, quality


def read_book_depth_day(
    root: Path,
    day_value: str,
    opened: set[Path],
    *,
    target_contract: str,
    oracle_scope: str,
    sealed_test_start: str,
    sealed_test_end: str,
    oracle_reference_count: int,
    oracle_reference_filename_sha256: str,
    metadata_callback=None,
) -> dict[str, np.ndarray]:
    if day_value >= sealed_test_start:
        raise ValueError(f"refusing sealed-test book-depth date {day_value}")
    reference = (root / f"{day_value}.json").resolve()
    if day_value in OFFICIAL_UNAVAILABLE_SET:
        if reference.exists():
            raise ValueError(
                f"official-unavailable book-depth day has a payload: {day_value}"
            )
        return empty_book_depth_day()
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    opened.add(reference)
    sequence = manifest.get("sequence", {})
    layout = manifest.get("layout", {})
    metadata = manifest.get("metadata", {})
    count = int(sequence.get("count", -1))
    archive = f"BTCUSDT-bookDepth-{day_value}.zip"
    csv_entry = f"BTCUSDT-bookDepth-{day_value}.csv"
    expected_url = f"{BOOK_DEPTH_SOURCE_ROOT}/{archive}"
    integer_names = (
        "sourceArchiveBytes",
        "sourceChecksumResponseBytes",
        "sourceCsvBytes",
        "sourceCsvHeaderRows",
        "sourceCsvRows",
        "sourceBandRows",
        "sourceSnapshotCount",
        "tenBandSnapshotCount",
        "twelveBandSnapshotCount",
        "firstTimestampOffsetSeconds",
        "lastTimestampOffsetSeconds",
        "minimumSnapshotGapSeconds",
        "maximumSnapshotGapSeconds",
    )
    try:
        counters = {name: int(metadata[name]) for name in integer_names}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid book-depth quality counters: {reference}"
        ) from error
    hash_names = (
        "sourceArchiveSha256",
        "sourceChecksumResponseSha256",
        "sourceCsvSha256",
    )
    if manifest.get("namespace") \
            != "derivatives-book-depth/usdm-futures/btcusdt" \
            or manifest.get("key") != day_value \
            or sequence != {
                "start": 0, "step": 1, "count": count, "unit": "index",
            } \
            or count < 1 \
            or layout.get("encoding") != BOOK_DEPTH_ENCODING \
            or layout.get("utcDayStartMs") != utc_day_start_ms(day_value) \
            or metadata.get("featureSchema") != BOOK_DEPTH_SCHEMA \
            or metadata.get("source") != "data.binance.vision" \
            or metadata.get("sourceDataset") \
            != "futures/um/daily/bookDepth" \
            or metadata.get("sourceArchiveUrl") != expected_url \
            or metadata.get("sourceArchiveChecksumUrl") \
            != f"{expected_url}.CHECKSUM" \
            or metadata.get("sourceArchiveChecksumAlgorithm") != "sha256" \
            or metadata.get("sourceArchiveChecksumFilename") != archive \
            or any(re.fullmatch(r"[a-f0-9]{64}", str(metadata.get(name, "")))
                   is None for name in hash_names) \
            or metadata.get("sourceCsvEntry") != csv_entry \
            or metadata.get("market") != "usdm-futures" \
            or metadata.get("symbol") != "BTCUSDT" \
            or metadata.get("timestampResolutionMs") != 1_000 \
            or metadata.get("irregularSnapshotAxis") is not True \
            or metadata.get("valuesAreCumulative") is not True \
            or metadata.get("bandPercentages") != list(BOOK_DEPTH_BANDS) \
            or metadata.get("schemaBandCounts") != [10, 12] \
            or metadata.get("sourceAvailability") \
            != "official-archive-present" \
            or metadata.get("officialUnavailableDateCount") \
            != EXPECTED_UNAVAILABLE_SOURCE_DAYS \
            or metadata.get("officialUnavailableDatesSha256") \
            != OFFICIAL_UNAVAILABLE_SHA256 \
            or metadata.get("oracleAllowlistDays") != EXPECTED_SOURCE_DAYS \
            or metadata.get("oracleAvailableIntersectionDays") \
            != EXPECTED_AVAILABLE_REFERENCE_DAYS \
            or metadata.get("oracleTargetContract") != target_contract \
            or metadata.get("oracleReferenceCount") != oracle_reference_count \
            or metadata.get("oracleReferenceFilenameSha256") \
            != oracle_reference_filename_sha256 \
            or metadata.get("oracleScope") != oracle_scope \
            or metadata.get("sealedTestStart") != sealed_test_start \
            or metadata.get("sealedTestEnd") != sealed_test_end:
        raise ValueError(f"invalid book-depth source contract: {reference}")
    if counters["sourceArchiveBytes"] < 1 \
            or counters["sourceChecksumResponseBytes"] < 1 \
            or counters["sourceCsvBytes"] < 1 \
            or counters["sourceCsvHeaderRows"] != 1 \
            or counters["sourceCsvRows"] != counters["sourceBandRows"] \
            or counters["sourceSnapshotCount"] != count \
            or counters["tenBandSnapshotCount"] \
            + counters["twelveBandSnapshotCount"] != count \
            or counters["tenBandSnapshotCount"] * 10 \
            + counters["twelveBandSnapshotCount"] * 12 \
            != counters["sourceBandRows"] \
            or counters["firstTimestampOffsetSeconds"] < 0 \
            or counters["lastTimestampOffsetSeconds"] \
            < counters["firstTimestampOffsetSeconds"] \
            or counters["lastTimestampOffsetSeconds"] >= 86_400 \
            or counters["minimumSnapshotGapSeconds"] < 0 \
            or counters["maximumSnapshotGapSeconds"] \
            < counters["minimumSnapshotGapSeconds"] \
            or any(metadata.get(name) != 0 for name in (
                "timestampAdjustedRows", "timestampFlooredRows",
                "timestampShiftedRows", "filledSnapshotCount",
                "repairedBandRows",
            )):
        raise ValueError(f"invalid book-depth quality counters: {reference}")
    values = read_derivatives_book_depth_columns(reference)
    offsets = values["timestampOffsetSeconds"].astype(np.int64, copy=False)
    schema = values["schemaBandCount"]
    gaps = np.diff(offsets)
    if offsets.shape != (count,) \
            or int(np.count_nonzero(schema == 10)) \
            != counters["tenBandSnapshotCount"] \
            or int(np.count_nonzero(schema == 12)) \
            != counters["twelveBandSnapshotCount"] \
            or int(offsets[0]) != counters["firstTimestampOffsetSeconds"] \
            or int(offsets[-1]) != counters["lastTimestampOffsetSeconds"] \
            or (0 if not gaps.size else int(gaps.min())) \
            != counters["minimumSnapshotGapSeconds"] \
            or (0 if not gaps.size else int(gaps.max())) \
            != counters["maximumSnapshotGapSeconds"]:
        raise ValueError(f"book-depth payload counters disagree: {reference}")
    if metadata_callback is not None:
        metadata_callback(day_value, metadata)
    return values
