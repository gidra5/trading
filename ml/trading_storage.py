"""Language-neutral storage adapters for market and training artifacts.

TypeScript owns most data generation, while training runs in Python. Keeping
this adapter deliberately small makes the on-disk contract—not a process or
language runtime—the shared storage API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import zstandard


DERIVATIVES_KLINE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "baseVolume",
    "quoteVolume",
    "tradeCount",
    "takerBuyBaseVolume",
    "takerBuyQuoteVolume",
)
DERIVATIVES_BOOK_DEPTH_BANDS = (0.2, 1.0, 2.0, 3.0, 4.0, 5.0)
DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS = (
    "bidDepth",
    "askDepth",
    "bidNotional",
    "askNotional",
)


@dataclass(frozen=True)
class SequentialAxis:
    start: int
    step: int
    count: int
    unit: str


@dataclass(frozen=True)
class SequentialShard:
    reference_file: Path
    storage_root: Path
    reference: dict[str, Any]
    axis: SequentialAxis


@dataclass(frozen=True)
class TrainingStorageLayout:
    root: Path
    immutable: Path
    datasets: Path
    runs: Path
    cache: Path
    analysis: Path


def training_storage_layout(repo_root: Path) -> TrainingStorageLayout:
    root = repo_root.resolve() / "data" / "training"
    return TrainingStorageLayout(
        root=root,
        immutable=root / "immutable",
        datasets=root / "datasets",
        runs=root / "runs",
        cache=root / "cache",
        analysis=root / "analysis",
    )


def require_under(candidate: Path, root: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be under {root.resolve()}: {resolved}") from error
    if not relative.parts:
        raise ValueError(f"{label} must be a child of {root.resolve()}")
    return resolved


def resolve_shard(reference_file: Path) -> SequentialShard:
    """Validate and resolve one direct immutable shard reference."""
    current = reference_file.resolve()
    value = json.loads(current.read_text(encoding="utf-8"))
    _validate_reference(value)
    root = _storage_root(current)
    sequence = value["sequence"]
    return SequentialShard(
        reference_file=current,
        storage_root=root,
        reference=value,
        axis=SequentialAxis(
            start=int(sequence["start"]),
            step=int(sequence["step"]),
            count=int(sequence["count"]),
            unit=str(sequence["unit"]),
        ),
    )


def read_shard_payload(
    reference_file: Path,
    *,
    verify: bool = True,
) -> tuple[SequentialShard, bytes]:
    """Read, decompress, size-check, and optionally hash-check one shard."""
    shard = resolve_shard(reference_file)
    stored = shard.reference["object"]
    object_file = (shard.storage_root / stored["file"]).resolve()
    try:
        object_file.relative_to(shard.storage_root)
    except ValueError as error:
        raise ValueError(
            f"storage object escapes its root: {stored['file']}"
        ) from error
    expected_bytes = int(stored["uncompressedBytes"])
    payload = zstandard.ZstdDecompressor().decompress(
        object_file.read_bytes(),
        max_output_size=expected_bytes,
    )
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{object_file} decoded to {len(payload)} bytes; "
            f"expected {expected_bytes}"
        )
    if verify:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != stored["contentHash"]:
            raise ValueError(f"{object_file} failed its SHA-256 content check")
    return shard, payload


def read_shard_array(
    reference_file: Path,
    dtype: str,
    shape: tuple[int, ...],
    *,
    verify: bool = True,
) -> tuple[SequentialShard, np.ndarray]:
    """Expose a generic row-major shard as a zero-copy NumPy array."""
    shard, payload = read_shard_payload(reference_file, verify=verify)
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{reference_file} has {len(payload)} decoded bytes; "
            f"array shape {shape} requires {expected_bytes}"
        )
    if shape and shape[0] != shard.axis.count:
        raise ValueError(
            f"{reference_file} sequence contains {shard.axis.count} rows; "
            f"requested shape contains {shape[0]}"
        )
    return shard, np.frombuffer(payload, dtype=dtype).reshape(shape)


def write_shard_payload(
    storage_root: Path,
    namespace: str,
    key: str,
    payload: bytes,
    *,
    sequence: dict[str, int | str],
    layout: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    compression_level: int = 9,
) -> Path:
    """Install an immutable sequential payload and its direct reference."""
    storage_root = storage_root.resolve()
    namespace = _identifier(namespace)
    key = _identifier(key)
    content_hash = hashlib.sha256(payload).hexdigest()
    relative_object = Path(
        "objects", "sha256", content_hash[:2], f"{content_hash}.zst"
    )
    object_file = storage_root / relative_object
    object_file.parent.mkdir(parents=True, exist_ok=True)
    if not object_file.exists():
        compressed = zstandard.ZstdCompressor(
            level=compression_level,
            threads=0,
        ).compress(payload)
        temporary = object_file.with_name(
            f"{object_file.name}.{os.getpid()}-{random.randrange(1 << 30)}.tmp"
        )
        try:
            temporary.write_bytes(compressed)
            try:
                os.link(temporary, object_file)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    reference_file = storage_root / "refs" / Path(*namespace.split("/")) \
        / Path(*key.split("/"))
    reference_file = reference_file.with_suffix(reference_file.suffix + ".json")
    reference = {
        "version": 1,
        "kind": "trading-sequential-shard",
        "namespace": namespace,
        "key": key,
        "createdAt": _iso_now(),
        "object": {
            "algorithm": "sha256",
            "contentHash": content_hash,
            "file": relative_object.as_posix(),
            "compression": "zstd",
            "compressionLevel": compression_level,
            "uncompressedBytes": len(payload),
            "compressedBytes": object_file.stat().st_size,
        },
        "sequence": sequence,
        "layout": layout,
        **({"metadata": metadata} if metadata else {}),
    }
    if reference_file.exists():
        existing = json.loads(reference_file.read_text(encoding="utf-8"))
        _validate_reference(existing)
        if existing["object"]["contentHash"] != content_hash:
            raise ValueError(
                f"immutable storage reference changed: {namespace}/{key}"
            )
    else:
        _atomic_json(reference, reference_file)
    return reference_file


def implicit_times(axis: SequentialAxis, dtype: str = "<i8") -> np.ndarray:
    """Materialize timestamps only in memory when an algorithm needs them."""
    return axis.start + np.arange(axis.count, dtype=dtype) * axis.step


def candle_times(reference_file: Path) -> np.ndarray:
    """Materialize a candle axis and apply its sparse discontinuities."""
    shard, _ = read_shard_payload(reference_file)
    values = implicit_times(shard.axis)
    for jump in shard.reference["layout"].get("timeJumps", []):
        index = int(jump["index"])
        delta = int(jump["deltaMs"])
        if index < 1 or index >= shard.axis.count:
            raise ValueError(f"invalid candle time jump: {reference_file}")
        values[index:] += delta
    return values


def read_candle_column(reference_file: Path, name: str) -> np.ndarray:
    """Decode one numeric column from the canonical candle codec."""
    shard, payload = read_shard_payload(reference_file)
    layout = shard.reference["layout"]
    if layout.get("encoding") != "candle-columnar-delta-v1":
        raise ValueError(f"unsupported candle encoding: {reference_file}")
    column = next(
        (value for value in layout.get("columns", []) if value.get("name") == name),
        None,
    )
    if column is None:
        raise ValueError(f"candle column {name!r} is missing: {reference_file}")
    start = int(column["offset"])
    end = start + int(column["bytes"])
    encoded = payload[start:end]
    if column["encoding"] == "float64-le":
        values = np.frombuffer(encoded, dtype="<f8")
        if values.shape != (shard.axis.count,):
            raise ValueError(f"candle column is truncated: {reference_file}")
        return values
    if column["encoding"] != "scaled-delta-zigzag-varint":
        raise ValueError(f"unsupported candle column encoding: {reference_file}")
    scale = int(column["scale"])
    values = np.empty(shard.axis.count, dtype=np.float64)
    offset = 0
    previous = 0
    for index in range(shard.axis.count):
        encoded_value = 0
        shift = 0
        while True:
            if offset >= len(encoded) or shift > 70:
                raise ValueError(f"truncated candle varint: {reference_file}")
            byte = encoded[offset]
            offset += 1
            encoded_value |= (byte & 0x7f) << shift
            if byte & 0x80 == 0:
                break
            shift += 7
        delta = (
            encoded_value // 2
            if encoded_value % 2 == 0
            else -(encoded_value + 1) // 2
        )
        previous += delta
        values[index] = previous / scale
    if offset != len(encoded):
        raise ValueError(f"candle column has trailing bytes: {reference_file}")
    return values


def read_trade_flow_columns(
    reference_file: Path,
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Decode selected columns from one canonical trade-flow shard."""
    shard, payload = read_shard_payload(reference_file)
    layout = shard.reference["layout"]
    if layout.get("encoding") != "trade-flow-columnar-v1":
        raise ValueError(f"unsupported trade-flow encoding: {reference_file}")
    columns = layout.get("columns")
    if not isinstance(columns, list) or len(columns) == 0:
        raise ValueError(f"trade-flow columns are missing: {reference_file}")
    by_name: dict[str, dict[str, Any]] = {}
    occupied: list[tuple[int, int]] = []
    for column in columns:
        if not isinstance(column, dict) or not isinstance(column.get("name"), str):
            raise ValueError(f"invalid trade-flow column: {reference_file}")
        column_name = str(column["name"])
        if column_name in by_name:
            raise ValueError(f"duplicate trade-flow column: {column_name}")
        start = int(column.get("offset", -1))
        size = int(column.get("bytes", -1))
        end = start + size
        if start < 0 or size < 0 or end > len(payload):
            raise ValueError(f"invalid trade-flow column bounds: {column_name}")
        occupied.append((start, end))
        by_name[column_name] = column
    if any(right_start < left_end for (_, left_end), (right_start, _) in zip(
        sorted(occupied), sorted(occupied)[1:], strict=False,
    )):
        raise ValueError(f"overlapping trade-flow columns: {reference_file}")

    result: dict[str, np.ndarray] = {}
    for name in names:
        column = by_name.get(name)
        if column is None:
            raise ValueError(
                f"trade-flow column {name!r} is missing: {reference_file}"
            )
        encoding = column.get("encoding")
        dtype = {
            "float64-le": "<f8",
            "uint32-le": "<u4",
            "int8": "i1",
        }.get(encoding)
        if dtype is None:
            raise ValueError(
                f"unsupported trade-flow column encoding {encoding!r}"
            )
        start = int(column["offset"])
        end = start + int(column["bytes"])
        values = np.frombuffer(payload[start:end], dtype=dtype)
        if values.shape != (shard.axis.count,):
            raise ValueError(f"trade-flow column is truncated: {name}")
        result[name] = values
    return result


def read_derivatives_metrics_columns(
    reference_file: Path,
    names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Decode nullable float columns and their per-field validity masks."""
    shard, payload = read_shard_payload(reference_file)
    layout = shard.reference["layout"]
    if layout.get("encoding") != "derivatives-metrics-columnar-v1":
        raise ValueError(
            f"unsupported derivatives-metrics encoding: {reference_file}"
        )
    columns = layout.get("columns")
    if not isinstance(columns, list) or len(columns) < 2:
        raise ValueError(
            f"derivatives-metrics columns are missing: {reference_file}"
        )
    by_name = {
        str(column.get("name")): column
        for column in columns
        if isinstance(column, dict) and isinstance(column.get("name"), str)
    }
    if len(by_name) != len(columns):
        raise ValueError(
            f"invalid or duplicate derivatives-metrics columns: {reference_file}"
        )
    mask_column = by_name.get("validMask")
    if mask_column is None or mask_column.get("encoding") != "uint8":
        raise ValueError(
            f"derivatives-metrics validity mask is missing: {reference_file}"
        )
    mask_start = int(mask_column.get("offset", -1))
    mask_end = mask_start + int(mask_column.get("bytes", -1))
    if mask_start < 0 or mask_end > len(payload):
        raise ValueError(
            f"invalid derivatives-metrics validity mask: {reference_file}"
        )
    masks = np.frombuffer(payload[mask_start:mask_end], dtype="u1")
    if masks.shape != (shard.axis.count,):
        raise ValueError(
            f"truncated derivatives-metrics validity mask: {reference_file}"
        )
    canonical_names = [
        str(column["name"]) for column in columns if column.get("name") != "validMask"
    ]
    values: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    for name in names:
        column = by_name.get(name)
        if column is None or column.get("encoding") != "float64-le":
            raise ValueError(
                f"derivatives-metrics column {name!r} is missing: {reference_file}"
            )
        try:
            bit = canonical_names.index(name)
        except ValueError as error:
            raise ValueError(
                f"derivatives-metrics column order is invalid: {name}"
            ) from error
        start = int(column.get("offset", -1))
        end = start + int(column.get("bytes", -1))
        if start < 0 or end > len(payload):
            raise ValueError(
                f"invalid derivatives-metrics column bounds: {name}"
            )
        decoded = np.frombuffer(payload[start:end], dtype="<f8")
        if decoded.shape != (shard.axis.count,):
            raise ValueError(f"derivatives-metrics column is truncated: {name}")
        valid = (masks & (1 << bit)) != 0
        if not np.isfinite(decoded).all() \
                or bool((decoded[valid] <= 0).any()) \
                or bool((decoded[~valid] != 0).any()):
            raise ValueError(f"invalid nullable derivatives-metrics values: {name}")
        values[name] = decoded
        validity[name] = valid
    return values, validity


def read_derivatives_kline_columns(
    reference_file: Path,
    names: tuple[str, ...] = DERIVATIVES_KLINE_COLUMNS,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Decode complete USD-M 1m kline rows and their shared row validity."""
    shard, payload = read_shard_payload(reference_file)
    layout = shard.reference["layout"]
    if layout.get("encoding") != "derivatives-klines-columnar-v1":
        raise ValueError(
            f"unsupported derivatives-klines encoding: {reference_file}"
        )
    if shard.axis.unit != "unix-ms" \
            or shard.axis.step != 60_000 \
            or shard.axis.start % 60_000 != 0 \
            or layout.get("closeTimeOffsetMs") != 59_999 \
            or layout.get("closed") is not True:
        raise ValueError(f"invalid derivatives-klines time layout: {reference_file}")
    columns = layout.get("columns")
    if not isinstance(columns, list) \
            or len(columns) != len(DERIVATIVES_KLINE_COLUMNS) + 1:
        raise ValueError(f"derivatives-klines columns are missing: {reference_file}")
    if len(names) != len(set(names)) \
            or any(name not in DERIVATIVES_KLINE_COLUMNS for name in names):
        raise ValueError("invalid or duplicate requested derivatives-klines columns")

    expected_offset = 0
    decoded: dict[str, np.ndarray] = {}
    for index, name in enumerate(DERIVATIVES_KLINE_COLUMNS):
        column = columns[index]
        encoding = "uint64-le" if name == "tradeCount" else "float64-le"
        size = shard.axis.count * 8
        if not isinstance(column, dict) \
                or column.get("name") != name \
                or column.get("encoding") != encoding \
                or column.get("offset") != expected_offset \
                or column.get("bytes") != size:
            raise ValueError(f"invalid derivatives-klines column layout: {name}")
        dtype = "<u8" if name == "tradeCount" else "<f8"
        values = np.frombuffer(
            payload[expected_offset:expected_offset + size],
            dtype=dtype,
        )
        if values.shape != (shard.axis.count,):
            raise ValueError(f"derivatives-klines column is truncated: {name}")
        decoded[name] = values
        expected_offset += size

    mask_column = columns[-1]
    if not isinstance(mask_column, dict) \
            or mask_column.get("name") != "validMask" \
            or mask_column.get("encoding") != "uint8" \
            or mask_column.get("offset") != expected_offset \
            or mask_column.get("bytes") != shard.axis.count \
            or len(payload) != expected_offset + shard.axis.count:
        raise ValueError(f"invalid derivatives-klines validity layout: {reference_file}")
    masks = np.frombuffer(payload[expected_offset:], dtype="u1")
    if masks.shape != (shard.axis.count,) \
            or bool(((masks != 0) & (masks != 1)).any()):
        raise ValueError(f"invalid derivatives-klines validity mask: {reference_file}")
    valid = masks == 1

    for name, values in decoded.items():
        if bool((values[~valid] != 0).any()):
            raise ValueError(
                f"missing derivatives-klines rows contain data in {name}"
            )
    float_names = tuple(
        name for name in DERIVATIVES_KLINE_COLUMNS if name != "tradeCount"
    )
    if any(not np.isfinite(decoded[name]).all() for name in float_names):
        raise ValueError(f"non-finite derivatives-klines values: {reference_file}")
    if valid.any():
        open_values = decoded["open"][valid]
        high_values = decoded["high"][valid]
        low_values = decoded["low"][valid]
        close_values = decoded["close"][valid]
        volume_names = (
            "baseVolume",
            "quoteVolume",
            "takerBuyBaseVolume",
            "takerBuyQuoteVolume",
        )
        if bool((open_values <= 0).any()) \
                or bool((high_values <= 0).any()) \
                or bool((low_values <= 0).any()) \
                or bool((close_values <= 0).any()) \
                or any(bool((decoded[name][valid] < 0).any()) for name in volume_names) \
                or bool((high_values < np.maximum(open_values, close_values)).any()) \
                or bool((low_values > np.minimum(open_values, close_values)).any()) \
                or bool((low_values > high_values).any()) \
                or bool((decoded["takerBuyBaseVolume"][valid]
                         > decoded["baseVolume"][valid]).any()) \
                or bool((decoded["takerBuyQuoteVolume"][valid]
                         > decoded["quoteVolume"][valid]).any()) \
                or bool((decoded["tradeCount"][valid]
                         > np.uint64(9_007_199_254_740_991)).any()):
            raise ValueError(f"invalid derivatives-klines row values: {reference_file}")
        trade_counts = decoded["tradeCount"][valid]
        no_trade = trade_counts == 0
        live = ~no_trade
        if bool((open_values[no_trade] != high_values[no_trade]).any()) \
                or bool((open_values[no_trade] != low_values[no_trade]).any()) \
                or bool((open_values[no_trade] != close_values[no_trade]).any()) \
                or any(
                    bool((decoded[name][valid][no_trade] != 0).any())
                    for name in volume_names
                ) \
                or bool((decoded["baseVolume"][valid][live] == 0).any()) \
                or bool((decoded["quoteVolume"][valid][live] == 0).any()):
            raise ValueError(
                f"invalid derivatives-klines no-trade semantics: {reference_file}"
            )
    return {name: decoded[name] for name in names}, valid


def read_derivatives_book_depth_columns(
    reference_file: Path,
) -> dict[str, np.ndarray]:
    """Decode one irregular USD-M book-depth day into fixed-band matrices."""
    shard, payload = read_shard_payload(reference_file)
    layout = shard.reference["layout"]
    if layout.get("encoding") != "derivatives-book-depth-columnar-v1" \
            or shard.axis.start != 0 \
            or shard.axis.step != 1 \
            or shard.axis.unit != "index" \
            or shard.axis.count < 1 \
            or layout.get("timestampResolutionMs") != 1_000 \
            or layout.get("timestampStorage") != "utc-day-second-offset" \
            or layout.get("bandPercentages") \
            != list(DERIVATIVES_BOOK_DEPTH_BANDS) \
            or layout.get("percentageSemantics") \
            != "negative-bid-positive-ask" \
            or layout.get("valuesAreCumulative") is not True \
            or layout.get("matrixOrder") != "snapshot-major-band-minor":
        raise ValueError(
            f"invalid derivatives-book-depth layout: {reference_file}"
        )
    utc_day_start = layout.get("utcDayStartMs")
    if not isinstance(utc_day_start, int) \
            or isinstance(utc_day_start, bool) \
            or utc_day_start % 86_400_000 != 0:
        raise ValueError(
            f"invalid derivatives-book-depth UTC day: {reference_file}"
        )

    count = shard.axis.count
    band_count = len(DERIVATIVES_BOOK_DEPTH_BANDS)
    expected = (
        ("timestampOffsetSeconds", "uint32-le", (count,), count * 4),
        ("schemaBandCount", "uint8", (count,), count),
        *((
            name,
            "float64-le",
            (count, band_count),
            count * band_count * 8,
        ) for name in DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS),
        ("bandAvailable", "uint8", (count, band_count), count * band_count),
    )
    columns = layout.get("columns")
    if not isinstance(columns, list) or len(columns) != len(expected):
        raise ValueError(
            f"invalid derivatives-book-depth columns: {reference_file}"
        )
    offset = 0
    decoded: dict[str, np.ndarray] = {}
    for column, (name, encoding, shape, size) in zip(
        columns, expected, strict=True,
    ):
        if not isinstance(column, dict) \
                or column.get("name") != name \
                or column.get("encoding") != encoding \
                or column.get("offset") != offset \
                or column.get("bytes") != size \
                or column.get("shape") != list(shape):
            raise ValueError(
                f"invalid derivatives-book-depth column layout: {name}"
            )
        dtype = {
            "uint32-le": "<u4",
            "uint8": "u1",
            "float64-le": "<f8",
        }[encoding]
        values = np.frombuffer(payload[offset:offset + size], dtype=dtype)
        if values.size != math.prod(shape):
            raise ValueError(
                f"truncated derivatives-book-depth column: {name}"
            )
        decoded[name] = values.reshape(shape)
        offset += size
    if offset != len(payload):
        raise ValueError(
            f"derivatives-book-depth payload has trailing bytes: {reference_file}"
        )

    timestamps = decoded["timestampOffsetSeconds"]
    schema = decoded["schemaBandCount"]
    availability_bytes = decoded["bandAvailable"]
    timestamp_differences = np.diff(timestamps.astype(np.int64, copy=False))
    if bool((timestamps >= 86_400).any()) \
            or bool((timestamp_differences <= 0).any()) \
            or bool(((schema != 10) & (schema != 12)).any()) \
            or bool(((availability_bytes != 0) & (availability_bytes != 1)).any()):
        raise ValueError(
            f"invalid derivatives-book-depth snapshot axis: {reference_file}"
        )
    availability = availability_bytes == 1
    expected_availability = np.ones((count, band_count), dtype=bool)
    expected_availability[schema == 10, 0] = False
    if not np.array_equal(availability, expected_availability):
        raise ValueError(
            f"invalid derivatives-book-depth availability: {reference_file}"
        )

    for name in DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS:
        values = decoded[name]
        if not np.isfinite(values).all() \
                or bool((values[availability] <= 0).any()) \
                or bool((values[~availability] != 0).any()):
            raise ValueError(
                f"invalid derivatives-book-depth values: {name}"
            )
        for row in range(count):
            available_values = values[row, availability[row]]
            if available_values.size > 1 \
                    and bool((np.diff(available_values) < 0).any()):
                raise ValueError(
                    f"non-monotone derivatives-book-depth values: {name}"
                )

    return {
        "timestampOffsetSeconds": timestamps,
        "schemaBandCount": schema,
        **{
            name: decoded[name]
            for name in DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS
        },
        "bandAvailable": availability,
    }


def is_storage_reference(file: Path) -> bool:
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) \
        and value.get("version") == 1 \
        and value.get("kind") == "trading-sequential-shard"


def save_torch_checkpoint(
    value: Any,
    pointer_file: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store checkpoint bytes once and atomically advance a mutable pointer."""
    import torch

    pointer_file = pointer_file.resolve()
    store_root = _training_store_root(pointer_file)
    checkpoint_directory = next(
        (parent for parent in pointer_file.parents if parent.name == "checkpoints"),
        None,
    )
    if checkpoint_directory is None:
        raise ValueError(f"checkpoint pointer lacks a checkpoints directory: {pointer_file}")
    temporary_dir = checkpoint_directory.parent / "tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary = temporary_dir / (
        f"checkpoint-{os.getpid()}-{random.randrange(1 << 30)}.tmp"
    )
    try:
        torch.save(value, temporary)
        content_hash = _file_hash(temporary)
        relative_object = Path(
            "objects", "sha256", content_hash[:2], f"{content_hash}.bin"
        )
        object_file = store_root / relative_object
        object_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(temporary, object_file)
        except FileExistsError:
            pass
        except OSError:
            if not object_file.exists():
                os.replace(temporary, object_file)
        size = temporary.stat().st_size if temporary.exists() else object_file.stat().st_size
        reference = {
            "version": 1,
            "kind": "trading-immutable-artifact",
            "namespace": "training/checkpoints",
            "key": _checkpoint_key(pointer_file),
            "createdAt": _iso_now(),
            "object": {
                "algorithm": "sha256",
                "contentHash": content_hash,
                "file": relative_object.as_posix(),
                "compression": "none",
                "uncompressedBytes": size,
                "compressedBytes": size,
            },
            "mediaType": "application/x-pytorch-checkpoint",
            **({"metadata": metadata} if metadata else {}),
        }
        _atomic_json(reference, pointer_file)
        _maybe_prune_checkpoint_orphans(store_root)
        return reference
    finally:
        # Windows indexers and virus scanners can briefly retain a handle to a
        # newly written PyTorch zip archive after the immutable object and its
        # pointer have already been committed. Cleanup must not turn that
        # successful checkpoint save into a failed training run.
        for attempt in range(20):
            try:
                temporary.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt < 19:
                    time.sleep(min(0.05 * (attempt + 1), 0.5))


def load_torch_checkpoint(
    pointer_file: Path,
    *,
    map_location: Any = None,
    weights_only: bool = False,
) -> Any:
    """Load a PyTorch checkpoint through its mutable canonical pointer."""
    import torch

    reference, object_file = resolve_artifact(pointer_file)
    return torch.load(
        object_file,
        map_location=map_location,
        weights_only=weights_only,
    )


def resolve_artifact(pointer_file: Path) -> tuple[dict[str, Any], Path]:
    pointer_file = pointer_file.resolve()
    reference = json.loads(pointer_file.read_text(encoding="utf-8"))
    _validate_artifact_reference(reference)
    store_root = _training_store_root(pointer_file)
    object_file = (store_root / reference["object"]["file"]).resolve()
    try:
        object_file.relative_to(store_root)
    except ValueError as error:
        raise ValueError("training artifact escapes the immutable store") from error
    if object_file.stat().st_size != int(reference["object"]["compressedBytes"]):
        raise ValueError(f"training artifact size mismatch: {object_file}")
    return reference, object_file


def checkpoint_exists(pointer_file: Path) -> bool:
    try:
        resolve_artifact(pointer_file)
        return True
    except FileNotFoundError:
        return False


def _storage_root(reference_file: Path) -> Path:
    for parent in reference_file.parents:
        if parent.name.lower() == "refs":
            return parent.parent.resolve()
        if parent.name.lower() == "datasets" \
                and parent.parent.name.lower() == "training":
            return (parent.parent / "immutable").resolve()
    raise ValueError(
        f"canonical storage reference is not inside a refs directory: "
        f"{reference_file}"
    )


def _validate_reference(value: Any) -> None:
    if not isinstance(value, dict) \
            or value.get("version") != 1 \
            or value.get("kind") != "trading-sequential-shard":
        raise ValueError("invalid sequential shard reference")
    stored = value.get("object")
    sequence = value.get("sequence")
    layout = value.get("layout")
    if not isinstance(stored, dict) \
            or stored.get("algorithm") != "sha256" \
            or stored.get("compression") != "zstd" \
            or not isinstance(stored.get("contentHash"), str) \
            or len(stored["contentHash"]) != 64 \
            or not isinstance(stored.get("file"), str) \
            or int(stored.get("uncompressedBytes", -1)) < 0:
        raise ValueError("invalid sequential shard object")
    content_hash = stored["contentHash"]
    expected_file = (
        f"objects/sha256/{content_hash[:2]}/{content_hash}.zst"
    )
    if stored["file"].replace("\\", "/") != expected_file:
        raise ValueError(
            "sequential shard object path does not match its content hash"
        )
    if not isinstance(sequence, dict) \
            or not isinstance(sequence.get("start"), int) \
            or not isinstance(sequence.get("step"), int) \
            or sequence["step"] < 1 \
            or not isinstance(sequence.get("count"), int) \
            or sequence["count"] < 0 \
            or sequence.get("unit") not in ("unix-ms", "index"):
        raise ValueError("invalid sequential shard axis")
    if not isinstance(layout, dict) \
            or not isinstance(layout.get("encoding"), str):
        raise ValueError("invalid sequential shard layout")


def _validate_artifact_reference(value: Any) -> None:
    if not isinstance(value, dict) \
            or value.get("version") != 1 \
            or value.get("kind") != "trading-immutable-artifact" \
            or value.get("mediaType") != "application/x-pytorch-checkpoint":
        raise ValueError("invalid training artifact reference")
    stored = value.get("object")
    if not isinstance(stored, dict) \
            or stored.get("algorithm") != "sha256" \
            or stored.get("compression") != "none" \
            or not isinstance(stored.get("contentHash"), str) \
            or len(stored["contentHash"]) != 64:
        raise ValueError("invalid training artifact object")
    content_hash = stored["contentHash"]
    expected = f"objects/sha256/{content_hash[:2]}/{content_hash}.bin"
    if str(stored.get("file", "")).replace("\\", "/") != expected:
        raise ValueError("training artifact path does not match its content hash")


def _training_store_root(pointer_file: Path) -> Path:
    for parent in pointer_file.parents:
        if parent.name.lower() == "runs" \
                and parent.parent.name.lower() == "training":
            return (parent.parent / "immutable").resolve()
    raise ValueError(
        f"checkpoint pointer is not under data/training/runs: {pointer_file}"
    )


def _checkpoint_key(pointer_file: Path) -> str:
    for parent in pointer_file.parents:
        if parent.name.lower() == "runs":
            return pointer_file.relative_to(parent).as_posix()
    raise ValueError(f"checkpoint pointer is outside training runs: {pointer_file}")


def _maybe_prune_checkpoint_orphans(store_root: Path) -> None:
    """Throttle full reference scans while still bounding unattended growth."""
    training_root = store_root.parent
    interval_minutes = float(os.environ.get(
        "TRADING_STORAGE_GC_INTERVAL_MINUTES",
        "15",
    ))
    if not math.isfinite(interval_minutes) or interval_minutes <= 0:
        raise ValueError("TRADING_STORAGE_GC_INTERVAL_MINUTES must be positive")
    marker = training_root / "cache" / ".checkpoint-gc"
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        if time.time() - marker.stat().st_mtime < interval_minutes * 60:
            return
    except FileNotFoundError:
        pass
    marker.touch()
    _prune_checkpoint_orphans(store_root)


def _prune_checkpoint_orphans(store_root: Path) -> None:
    """Bound checkpoint growth while preserving a commit-race grace period.

    Checkpoint GC is deliberately checkpoint-only: ``.bin`` objects can only
    be retained by canonical artifact pointers below
    ``data/training/runs/**/checkpoints``. Sequential-shard references retain
    ``.zst`` objects and must never be traversed by this hot-path maintenance.
    The grace period still protects an object committed concurrently before
    its pointer becomes visible to this scan.
    """
    training_root = store_root.parent
    referenced: set[Path] = set()
    invalid = False
    runs_root = training_root / "runs"
    if runs_root.exists():
        for file in runs_root.rglob("checkpoints/**/*.json"):
            try:
                value = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if value.get("kind") != "trading-immutable-artifact":
                continue
            try:
                _validate_artifact_reference(value)
            except ValueError:
                invalid = True
                continue
            relative = value.get("object", {}).get("file")
            if not isinstance(relative, str):
                invalid = True
                continue
            object_file = (store_root / relative).resolve()
            try:
                object_file.relative_to(store_root)
            except ValueError:
                invalid = True
                continue
            referenced.add(object_file)
    if invalid:
        return
    grace_hours = float(os.environ.get(
        "TRADING_STORAGE_ORPHAN_GRACE_HOURS",
        "1",
    ))
    if not math.isfinite(grace_hours) or grace_hours <= 0:
        raise ValueError(
            "TRADING_STORAGE_ORPHAN_GRACE_HOURS must be positive"
        )
    cutoff = time.time() - grace_hours * 60 * 60
    checkpoint_object_root = store_root / "objects" / "sha256"
    if not checkpoint_object_root.exists():
        return
    # The candidate set is explicitly limited to checkpoint payloads. Never
    # broaden this to sequential-shard ``.zst`` objects.
    for object_file in checkpoint_object_root.glob("*/*.bin"):
        try:
            if object_file.resolve() not in referenced \
                    and object_file.stat().st_mtime <= cutoff:
                object_file.unlink(missing_ok=True)
        except FileNotFoundError:
            continue


def _file_hash(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if not parts or any(
        not part or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._=-"
            )
            for character in part
        )
        for part in parts
    ):
        raise ValueError(f"unsafe storage identifier: {value}")
    return "/".join(parts)


def _atomic_json(value: dict[str, Any], file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(
        f"{file.name}.{os.getpid()}-{random.randrange(1 << 30)}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
