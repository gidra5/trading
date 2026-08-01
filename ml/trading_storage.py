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
        temporary.unlink(missing_ok=True)


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
    """Bound checkpoint growth while preserving a commit-race grace period."""
    training_root = store_root.parent
    referenced: set[Path] = set()
    invalid = False
    for root in (store_root / "refs", training_root / "runs"):
        if not root.exists():
            continue
        for file in root.rglob("*.json"):
            try:
                value = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if value.get("kind") not in (
                "trading-sequential-shard",
                "trading-immutable-artifact",
            ):
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
    object_root = store_root / "objects" / "sha256"
    if not object_root.exists():
        return
    for object_file in object_root.glob("*/*.bin"):
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
