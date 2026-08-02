"""Strictly causal features from irregular USD-M book-depth snapshots.

The public API accepts two day dictionaries (previous day, current day).  Each
dictionary has exactly these NumPy arrays:

``timestampOffsetSeconds``
    Strictly increasing signed/unsigned integer seconds in ``[0, 86399]``.
``bidDepth``, ``askDepth``, ``bidNotional``, ``askNotional``
    Float-like matrices with shape ``(snapshot_count, 6)``.  The fixed band
    axis is ``(0.2, 1, 2, 3, 4, 5)`` percent from the book boundary.  Values
    are cumulative quantities; they are not interpreted as prices.
``bandAvailable``
    Boolean matrix with the same shape.  A usable snapshot must contain the
    five 1--5 percent bands; the optional 0.2-percent band distinguishes the
    12-band schema from the 10-band schema.
``snapshotUsable``
    Boolean vector supplied by the caller after structural, market-alignment,
    and corruption checks.
``snapshotCorrupt``
    Boolean vector retaining the explicit corruption observation.  Corrupt
    rows are never used, even if ``snapshotUsable`` was set incorrectly.

Target row ``k`` occurs at ``day + k * 60s + 999ms``.  Only a snapshot whose
integer timestamp is at most ``k * 60`` is eligible.  Missing values are
forward-filled from prior usable observations, including across the day
boundary, and are never backward-filled.
"""

from __future__ import annotations

import numpy as np


DAY_SECONDS = 86_400
DAY_ROWS = 1_440
BOOK_DEPTH_BANDS = (0.2, 1.0, 2.0, 3.0, 4.0, 5.0)
BOOK_DEPTH_BAND_LABELS = ("0p2pct", "1pct", "2pct", "3pct", "4pct", "5pct")
BOOK_DEPTH_VALUE_COLUMNS = (
    "bidDepth",
    "askDepth",
    "bidNotional",
    "askNotional",
)
BOOK_DEPTH_DAY_COLUMNS = (
    "timestampOffsetSeconds",
    *BOOK_DEPTH_VALUE_COLUMNS,
    "bandAvailable",
    "snapshotUsable",
    "snapshotCorrupt",
)
CHANGE_HORIZONS = (1, 5, 15, 60)
RATE_HORIZONS = (1, 5, 15, 60)
HORIZON_LABELS = {1: "1m", 5: "5m", 15: "15m", 60: "1h"}


def _absolute_name(name: str) -> str:
    return f"abs{name[0].upper()}{name[1:]}"


PER_BAND_SIGNED_FEATURE_NAMES = tuple(
    f"{quantity}BidAskImbalance{band}"
    for quantity in ("base", "notional")
    for band in BOOK_DEPTH_BAND_LABELS
)
MARGINAL_RING_SIGNED_FEATURE_NAMES = tuple(
    f"{quantity}MarginalRing{inner}To{outer}BidAskImbalance"
    for quantity in ("base", "notional")
    for inner, outer in zip(
        BOOK_DEPTH_BAND_LABELS[:-1], BOOK_DEPTH_BAND_LABELS[1:], strict=True,
    )
)
SHAPE_SIGNED_FEATURE_NAMES = tuple(
    f"{quantity}{shape}"
    for quantity in ("base", "notional")
    for shape in (
        "NearConcentrationImbalance",
        "FarConcentrationImbalance",
        "ConvexityDifference",
    )
)
SHAPE_LEVEL_FEATURE_NAMES = tuple(
    f"{quantity}{shape}"
    for quantity in ("base", "notional")
    for shape in (
        "NearConcentrationMean",
        "FarConcentrationMean",
        "ConvexityMean",
    )
)
CHANGE_SIGNED_FEATURE_NAMES = tuple(
    f"{quantity}Total{band}LogChange{HORIZON_LABELS[horizon]}"
    for quantity in ("base", "notional")
    for band in BOOK_DEPTH_BAND_LABELS
    for horizon in CHANGE_HORIZONS
)
SIGNED_FUTURES_BOOK_DEPTH_FEATURE_NAMES = (
    PER_BAND_SIGNED_FEATURE_NAMES
    + MARGINAL_RING_SIGNED_FEATURE_NAMES
    + SHAPE_SIGNED_FEATURE_NAMES
    + CHANGE_SIGNED_FEATURE_NAMES
)
RATE_FEATURE_NAMES = tuple(
    name
    for horizon in RATE_HORIZONS
    for name in (
        f"bookDepthRawSnapshotUpdatesPerMinute{HORIZON_LABELS[horizon]}",
        f"bookDepthUsableSnapshotUpdatesPerMinute{HORIZON_LABELS[horizon]}",
        f"bookDepthStaleSnapshotFraction{HORIZON_LABELS[horizon]}",
    )
)
STATE_FEATURE_NAMES = (
    "bookDepthUsableValueAge24h",
    "bookDepthRawSnapshotAge24h",
    "bookDepthCurrentSecondObserved",
    "bookDepthCurrentSecondUsable",
    "bookDepthCurrentSecondCorrupt",
    "bookDepthCurrentSecondSchema10",
    "bookDepthCurrentSecondSchema12",
    "bookDepthBand0p2ValueAvailable",
    "bookDepthBand0p2ObservationAge24h",
)
FUTURES_BOOK_DEPTH_FEATURE_NAMES = tuple(
    name
    for signed in SIGNED_FUTURES_BOOK_DEPTH_FEATURE_NAMES
    for name in (signed, _absolute_name(signed))
) + SHAPE_LEVEL_FEATURE_NAMES + RATE_FEATURE_NAMES + STATE_FEATURE_NAMES


def causal_futures_book_depth_features(
    previous_day: dict[str, np.ndarray],
    current_day: dict[str, np.ndarray],
) -> np.ndarray:
    """Return 1,440 finite ``float32`` causal target-minute feature rows."""
    previous = _validate_day(previous_day, "previous")
    current = _validate_day(current_day, "current")
    merged = _merge_days(previous, current)

    raw_timestamps = merged["timestampOffsetSeconds"]
    raw_usable = merged["snapshotUsable"]
    raw_corrupt = merged["snapshotCorrupt"]
    usable = raw_usable & ~raw_corrupt
    usable_timestamps = raw_timestamps[usable]

    query_seconds = np.arange(-DAY_SECONDS, DAY_SECONDS, 60, dtype=np.int64)
    current_queries = query_seconds[DAY_ROWS:]
    usable_indexes, usable_ever = _latest_indexes(
        usable_timestamps, query_seconds,
    )

    sampled: dict[str, np.ndarray] = {}
    band_ever = np.repeat(usable_ever[:, None], len(BOOK_DEPTH_BANDS), axis=1)
    sampled_band_available = np.zeros_like(band_ever)
    for name in BOOK_DEPTH_VALUE_COLUMNS:
        values = merged[name][usable]
        output = np.zeros((query_seconds.size, len(BOOK_DEPTH_BANDS)))
        if usable_timestamps.size:
            output[usable_ever] = values[usable_indexes[usable_ever]]
        sampled[name] = output
    if usable_timestamps.size:
        sampled_band_available[usable_ever] = merged["bandAvailable"][usable][
            usable_indexes[usable_ever]
        ]
    latest_usable_has_band0 = sampled_band_available[:, 0].copy()

    # The optional 0.2% stream is nullable independently of the five common
    # bands.  Preserve its last genuinely observed value across 10-band rows.
    usable_band0 = usable & merged["bandAvailable"][:, 0]
    band0_timestamps = raw_timestamps[usable_band0]
    band0_indexes, band0_ever = _latest_indexes(band0_timestamps, query_seconds)
    for name in BOOK_DEPTH_VALUE_COLUMNS:
        band0_values = merged[name][usable_band0, 0]
        sampled[name][:, 0] = 0.0
        if band0_timestamps.size:
            sampled[name][band0_ever, 0] = band0_values[
                band0_indexes[band0_ever]
            ]
    band_ever[:, 0] = band0_ever
    sampled_band_available[:, 0] = band0_ever

    features: dict[str, np.ndarray] = {}

    def add_signed(name: str, values: np.ndarray) -> None:
        finite = np.where(np.isfinite(values), values, 0.0)
        features[name] = finite
        features[_absolute_name(name)] = np.abs(finite)

    for quantity, bid_name, ask_name in (
        ("base", "bidDepth", "askDepth"),
        ("notional", "bidNotional", "askNotional"),
    ):
        bid = sampled[bid_name]
        ask = sampled[ask_name]
        for band_index, band in enumerate(BOOK_DEPTH_BAND_LABELS):
            available = band_ever[:, band_index]
            add_signed(
                f"{quantity}BidAskImbalance{band}",
                _imbalance(bid[:, band_index], ask[:, band_index], available),
            )

        # Ring zero is valid only when both cumulative boundaries came from
        # the same latest usable snapshot.  This avoids subtracting a stale
        # carried 0.2% observation from a newer 1% observation.
        for outer_index in range(1, len(BOOK_DEPTH_BANDS)):
            inner_index = outer_index - 1
            available = band_ever[:, inner_index] & band_ever[:, outer_index]
            if inner_index == 0:
                available &= latest_usable_has_band0
            bid_ring = bid[:, outer_index] - bid[:, inner_index]
            ask_ring = ask[:, outer_index] - ask[:, inner_index]
            add_signed(
                f"{quantity}MarginalRing"
                f"{BOOK_DEPTH_BAND_LABELS[inner_index]}To"
                f"{BOOK_DEPTH_BAND_LABELS[outer_index]}BidAskImbalance",
                _imbalance(bid_ring, ask_ring, available),
            )

        common_available = usable_ever
        bid_near = _safe_ratio(bid[:, 1], bid[:, 5], common_available)
        ask_near = _safe_ratio(ask[:, 1], ask[:, 5], common_available)
        bid_far = _safe_ratio(
            bid[:, 5] - bid[:, 2], bid[:, 5], common_available,
        )
        ask_far = _safe_ratio(
            ask[:, 5] - ask[:, 2], ask[:, 5], common_available,
        )
        bid_convexity = np.where(
            common_available, np.log(np.maximum(bid_near, 1e-30) / 0.2), 0.0,
        )
        ask_convexity = np.where(
            common_available, np.log(np.maximum(ask_near, 1e-30) / 0.2), 0.0,
        )
        features[f"{quantity}NearConcentrationMean"] = (
            0.5 * (bid_near + ask_near)
        )
        features[f"{quantity}FarConcentrationMean"] = (
            0.5 * (bid_far + ask_far)
        )
        features[f"{quantity}ConvexityMean"] = (
            0.5 * (bid_convexity + ask_convexity)
        )
        add_signed(
            f"{quantity}NearConcentrationImbalance",
            _imbalance(bid_near, ask_near, common_available),
        )
        add_signed(
            f"{quantity}FarConcentrationImbalance",
            _imbalance(bid_far, ask_far, common_available),
        )
        add_signed(
            f"{quantity}ConvexityDifference",
            np.where(
                common_available, bid_convexity - ask_convexity, 0.0,
            ),
        )

        log_total = _log_pair_sum(bid, ask, band_ever)
        for band_index, band in enumerate(BOOK_DEPTH_BAND_LABELS):
            for horizon in CHANGE_HORIZONS:
                values = np.zeros(query_seconds.size)
                current_indexes = np.arange(horizon, query_seconds.size)
                previous_indexes = current_indexes - horizon
                available = (
                    band_ever[current_indexes, band_index]
                    & band_ever[previous_indexes, band_index]
                )
                selected = current_indexes[available]
                prior = previous_indexes[available]
                values[selected] = (
                    log_total[selected, band_index]
                    - log_total[prior, band_index]
                )
                add_signed(
                    f"{quantity}Total{band}LogChange"
                    f"{HORIZON_LABELS[horizon]}",
                    values,
                )

    for horizon in RATE_HORIZONS:
        raw_counts = _window_counts(raw_timestamps, current_queries, horizon * 60)
        usable_counts = _window_counts(
            usable_timestamps, current_queries, horizon * 60,
        )
        features[
            f"bookDepthRawSnapshotUpdatesPerMinute{HORIZON_LABELS[horizon]}"
        ] = raw_counts / horizon
        features[
            f"bookDepthUsableSnapshotUpdatesPerMinute{HORIZON_LABELS[horizon]}"
        ] = usable_counts / horizon
        features[
            f"bookDepthStaleSnapshotFraction{HORIZON_LABELS[horizon]}"
        ] = np.divide(
            raw_counts - usable_counts,
            raw_counts,
            out=np.zeros_like(raw_counts),
            where=raw_counts > 0,
        )

    raw_latest, raw_ever = _latest_indexes(raw_timestamps, current_queries)
    usable_latest, current_usable_ever = _latest_indexes(
        usable_timestamps, current_queries,
    )
    raw_age = _ages(raw_timestamps, raw_latest, raw_ever, current_queries)
    usable_age = _ages(
        usable_timestamps, usable_latest, current_usable_ever, current_queries,
    )
    band0_latest, current_band0_ever = _latest_indexes(
        band0_timestamps, current_queries,
    )
    band0_age = _ages(
        band0_timestamps, band0_latest, current_band0_ever, current_queries,
    )
    exact_indexes, exact_observed = _exact_indexes(
        raw_timestamps, current_queries,
    )
    exact_usable = np.zeros(DAY_ROWS, dtype=bool)
    exact_corrupt = np.zeros(DAY_ROWS, dtype=bool)
    exact_schema10 = np.zeros(DAY_ROWS, dtype=bool)
    exact_schema12 = np.zeros(DAY_ROWS, dtype=bool)
    if exact_observed.any():
        source = exact_indexes[exact_observed]
        available = merged["bandAvailable"][source]
        exact_usable[exact_observed] = usable[source]
        exact_corrupt[exact_observed] = raw_corrupt[source]
        exact_schema10[exact_observed] = (
            ~available[:, 0] & available[:, 1:].all(axis=1)
        )
        exact_schema12[exact_observed] = available.all(axis=1)

    features.update({
        "bookDepthUsableValueAge24h": usable_age / DAY_SECONDS,
        "bookDepthRawSnapshotAge24h": raw_age / DAY_SECONDS,
        "bookDepthCurrentSecondObserved": exact_observed.astype(np.float64),
        "bookDepthCurrentSecondUsable": exact_usable.astype(np.float64),
        "bookDepthCurrentSecondCorrupt": exact_corrupt.astype(np.float64),
        "bookDepthCurrentSecondSchema10": exact_schema10.astype(np.float64),
        "bookDepthCurrentSecondSchema12": exact_schema12.astype(np.float64),
        "bookDepthBand0p2ValueAvailable": current_band0_ever.astype(np.float64),
        "bookDepthBand0p2ObservationAge24h": band0_age / DAY_SECONDS,
    })

    result = np.column_stack([
        np.asarray(features[name], dtype=np.float64)[DAY_ROWS:]
        if np.asarray(features[name]).shape == (DAY_ROWS * 2,)
        else np.asarray(features[name], dtype=np.float64)
        for name in FUTURES_BOOK_DEPTH_FEATURE_NAMES
    ]).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, len(FUTURES_BOOK_DEPTH_FEATURE_NAMES)) \
            or len(FUTURES_BOOK_DEPTH_FEATURE_NAMES) \
            != len(set(FUTURES_BOOK_DEPTH_FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError("invalid causal futures book-depth feature matrix")
    return result


def _validate_day(
    day: dict[str, np.ndarray],
    label: str,
) -> dict[str, np.ndarray]:
    if set(day) != set(BOOK_DEPTH_DAY_COLUMNS):
        raise ValueError(f"{label} book-depth columns differ from the contract")
    offsets = np.asarray(day["timestampOffsetSeconds"])
    if offsets.ndim != 1 or offsets.dtype.kind not in ("i", "u"):
        raise ValueError(f"{label} book-depth timestamps must be integer seconds")
    if offsets.size and (
        bool((offsets < 0).any()) if offsets.dtype.kind == "i" else False
    ):
        raise ValueError(f"{label} book-depth timestamp is outside the day")
    if offsets.size and bool((offsets > DAY_SECONDS - 1).any()):
        raise ValueError(f"{label} book-depth timestamp is outside the day")
    offsets = offsets.astype(np.int64, copy=False)
    if offsets.size > 1 and bool((np.diff(offsets) <= 0).any()):
        raise ValueError(f"{label} book-depth timestamps are not unique/increasing")

    count = offsets.size
    output: dict[str, np.ndarray] = {"timestampOffsetSeconds": offsets}
    for name in BOOK_DEPTH_VALUE_COLUMNS:
        raw = np.asarray(day[name])
        if raw.shape != (count, len(BOOK_DEPTH_BANDS)) \
                or raw.dtype.kind not in ("f", "i", "u"):
            raise ValueError(f"invalid {label} book-depth matrix: {name}")
        output[name] = raw.astype(np.float64, copy=False)
    for name, shape in (
        ("bandAvailable", (count, len(BOOK_DEPTH_BANDS))),
        ("snapshotUsable", (count,)),
        ("snapshotCorrupt", (count,)),
    ):
        raw = np.asarray(day[name])
        if raw.shape != shape or raw.dtype.kind != "b":
            raise ValueError(f"invalid {label} book-depth boolean: {name}")
        output[name] = raw.astype(bool, copy=False)

    effective = output["snapshotUsable"] & ~output["snapshotCorrupt"]
    available = output["bandAvailable"][effective]
    if effective.any() and not available[:, 1:].all():
        raise ValueError(f"usable {label} snapshot is not a 10/12-band schema")
    for name in BOOK_DEPTH_VALUE_COLUMNS:
        values = output[name][effective]
        if not values.size:
            continue
        active = available
        if not np.isfinite(values[active]).all() \
                or bool((values[active] <= 0).any()):
            raise ValueError(f"usable {label} snapshot has invalid {name}")
        for row, row_available in zip(values, available, strict=True):
            start = 0 if row_available[0] else 1
            if bool((np.diff(row[start:]) < 0).any()):
                raise ValueError(
                    f"usable {label} snapshot has non-cumulative {name}"
                )
    return output


def _merge_days(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    merged = {
        name: np.concatenate((previous[name], current[name]), axis=0)
        for name in BOOK_DEPTH_DAY_COLUMNS
    }
    merged["timestampOffsetSeconds"] = np.concatenate((
        previous["timestampOffsetSeconds"] - DAY_SECONDS,
        current["timestampOffsetSeconds"],
    ))
    return merged


def _latest_indexes(
    timestamps: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.searchsorted(timestamps, queries, side="right") - 1
    return indexes, indexes >= 0


def _exact_indexes(
    timestamps: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.searchsorted(timestamps, queries, side="left")
    observed = indexes < timestamps.size
    observed[observed] &= timestamps[indexes[observed]] == queries[observed]
    return indexes, observed


def _ages(
    timestamps: np.ndarray,
    indexes: np.ndarray,
    ever: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    ages = np.full(queries.size, DAY_SECONDS, dtype=np.float64)
    ages[ever] = queries[ever] - timestamps[indexes[ever]]
    return np.minimum(ages, DAY_SECONDS)


def _window_counts(
    timestamps: np.ndarray,
    queries: np.ndarray,
    width_seconds: int,
) -> np.ndarray:
    right = np.searchsorted(timestamps, queries, side="right")
    left = np.searchsorted(timestamps, queries - width_seconds, side="right")
    return (right - left).astype(np.float64)


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=available & (denominator > 0),
    )


def _imbalance(
    bid: np.ndarray,
    ask: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    scale = np.maximum(np.abs(bid), np.abs(ask))
    scaled_bid = np.divide(
        bid, scale, out=np.zeros_like(bid, dtype=np.float64), where=scale > 0,
    )
    scaled_ask = np.divide(
        ask, scale, out=np.zeros_like(ask, dtype=np.float64), where=scale > 0,
    )
    denominator = scaled_bid + scaled_ask
    return np.divide(
        scaled_bid - scaled_ask,
        denominator,
        out=np.zeros_like(bid, dtype=np.float64),
        where=available & (denominator > 0),
    )


def _log_pair_sum(
    left: np.ndarray,
    right: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    maximum = np.maximum(left, right)
    minimum = np.minimum(left, right)
    result = np.zeros_like(maximum, dtype=np.float64)
    result[available] = (
        np.log(maximum[available])
        + np.log1p(minimum[available] / maximum[available])
    )
    return result
