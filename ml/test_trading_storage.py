import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np
import zstandard

from trading_storage import (
    _prune_checkpoint_orphans,
    checkpoint_exists,
    implicit_times,
    load_torch_checkpoint,
    read_derivatives_metrics_columns,
    read_trade_flow_columns,
    read_shard_array,
    save_torch_checkpoint,
)
from train_mlp import load_compressed_component


class TradingStorageTests(unittest.TestCase):
    @staticmethod
    def checkpoint_reference(content_hash: str) -> dict:
        return {
            "version": 1,
            "kind": "trading-immutable-artifact",
            "namespace": "training/checkpoints",
            "key": "nested/checkpoints/best.json",
            "createdAt": "2026-01-01T00:00:00.000Z",
            "object": {
                "algorithm": "sha256",
                "contentHash": content_hash,
                "file": (
                    f"objects/sha256/{content_hash[:2]}/"
                    f"{content_hash}.bin"
                ),
                "compression": "none",
                "uncompressedBytes": 1,
                "compressedBytes": 1,
            },
            "mediaType": "application/x-pytorch-checkpoint",
        }

    def test_reads_direct_reference_and_reconstructs_implicit_axis(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "storage"
            reference_file = root / "refs" / "features" / "day.json"
            object_root = root / "objects" / "sha256"
            values = np.arange(24, dtype="<f2").reshape(6, 4)
            payload = values.tobytes()
            content_hash = hashlib.sha256(payload).hexdigest()
            object_file = (
                object_root / content_hash[:2] / f"{content_hash}.zst"
            )
            object_file.parent.mkdir(parents=True)
            compressed = zstandard.ZstdCompressor(level=9).compress(payload)
            object_file.write_bytes(compressed)
            reference_file.parent.mkdir(parents=True)
            reference_file.write_text(json.dumps({
                "version": 1,
                "kind": "trading-sequential-shard",
                "namespace": "features",
                "key": "day",
                "createdAt": "2026-01-01T00:00:00.000Z",
                "object": {
                    "algorithm": "sha256",
                    "contentHash": content_hash,
                    "file": (
                        f"objects/sha256/{content_hash[:2]}/"
                        f"{content_hash}.zst"
                    ),
                    "compression": "zstd",
                    "compressionLevel": 9,
                    "uncompressedBytes": len(payload),
                    "compressedBytes": len(compressed),
                },
                "sequence": {
                    "start": 1_000,
                    "step": 1_000,
                    "count": 6,
                    "unit": "unix-ms",
                },
                "layout": {
                    "encoding": "row-major",
                    "dtype": "float16-le",
                    "columns": 4,
                },
            }), encoding="utf-8")
            shard, actual = read_shard_array(reference_file, "<f2", (6, 4))

            np.testing.assert_array_equal(actual, values)
            np.testing.assert_array_equal(
                load_compressed_component(reference_file, "<f2", 4, 6),
                values,
            )
            np.testing.assert_array_equal(
                implicit_times(shard.axis),
                np.arange(1_000, 7_000, 1_000, dtype="<i8"),
            )

    def test_checkpoint_pointer_targets_immutable_content(self) -> None:
        with TemporaryDirectory() as temporary:
            pointer = (
                Path(temporary) / "data" / "training" / "runs" / "test-run"
                / "checkpoints" / "last.json"
            )
            save_torch_checkpoint({"epoch": 7}, pointer)
            self.assertTrue(checkpoint_exists(pointer))
            self.assertEqual(
                load_torch_checkpoint(pointer, map_location="cpu")["epoch"],
                7,
            )
            reference = json.loads(pointer.read_text(encoding="utf-8"))
            object_file = (
                Path(temporary) / "data" / "training" / "immutable"
                / reference["object"]["file"]
            )
            self.assertTrue(object_file.is_file())
            self.assertNotEqual(pointer.stat().st_ino, object_file.stat().st_ino)

    def test_reads_selected_trade_flow_columns_once(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "storage"
            reference_file = root / "refs" / "trade-flow" / "day.json"
            floats = np.asarray([1.25, 2.5], dtype="<f8")
            counts = np.asarray([3, 4], dtype="<u4")
            sides = np.asarray([-1, 1], dtype="i1")
            payload = floats.tobytes() + counts.tobytes() + sides.tobytes()
            content_hash = hashlib.sha256(payload).hexdigest()
            object_file = (
                root / "objects" / "sha256" / content_hash[:2]
                / f"{content_hash}.zst"
            )
            object_file.parent.mkdir(parents=True)
            compressed = zstandard.ZstdCompressor(level=9).compress(payload)
            object_file.write_bytes(compressed)
            reference_file.parent.mkdir(parents=True)
            reference_file.write_text(json.dumps({
                "version": 1,
                "kind": "trading-sequential-shard",
                "namespace": "trade-flow",
                "key": "day",
                "createdAt": "2026-01-01T00:00:00.000Z",
                "object": {
                    "algorithm": "sha256",
                    "contentHash": content_hash,
                    "file": (
                        f"objects/sha256/{content_hash[:2]}/"
                        f"{content_hash}.zst"
                    ),
                    "compression": "zstd",
                    "compressionLevel": 9,
                    "uncompressedBytes": len(payload),
                    "compressedBytes": len(compressed),
                },
                "sequence": {
                    "start": 1_000,
                    "step": 1_000,
                    "count": 2,
                    "unit": "unix-ms",
                },
                "layout": {
                    "encoding": "trade-flow-columnar-v1",
                    "columns": [
                        {
                            "name": "volume",
                            "encoding": "float64-le",
                            "offset": 0,
                            "bytes": floats.nbytes,
                        },
                        {
                            "name": "count",
                            "encoding": "uint32-le",
                            "offset": floats.nbytes,
                            "bytes": counts.nbytes,
                        },
                        {
                            "name": "side",
                            "encoding": "int8",
                            "offset": floats.nbytes + counts.nbytes,
                            "bytes": sides.nbytes,
                        },
                    ],
                },
            }), encoding="utf-8")
            actual = read_trade_flow_columns(
                reference_file, ("side", "volume", "count"),
            )
            np.testing.assert_array_equal(actual["volume"], floats)
            np.testing.assert_array_equal(actual["count"], counts)
            np.testing.assert_array_equal(actual["side"], sides)

    def test_reads_nullable_derivatives_metrics_columns(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "storage"
            reference_file = root / "refs" / "derivatives-metrics" / "day.json"
            first = np.asarray([1.25, 0], dtype="<f8")
            second = np.asarray([2.5, 3.5], dtype="<f8")
            masks = np.asarray([0b11, 0b10], dtype="u1")
            payload = first.tobytes() + second.tobytes() + masks.tobytes()
            content_hash = hashlib.sha256(payload).hexdigest()
            object_file = (
                root / "objects" / "sha256" / content_hash[:2]
                / f"{content_hash}.zst"
            )
            object_file.parent.mkdir(parents=True)
            compressed = zstandard.ZstdCompressor(level=9).compress(payload)
            object_file.write_bytes(compressed)
            reference_file.parent.mkdir(parents=True)
            reference_file.write_text(json.dumps({
                "version": 1,
                "kind": "trading-sequential-shard",
                "namespace": "derivatives-metrics",
                "key": "day",
                "createdAt": "2026-01-01T00:00:00.000Z",
                "object": {
                    "algorithm": "sha256",
                    "contentHash": content_hash,
                    "file": (
                        f"objects/sha256/{content_hash[:2]}/"
                        f"{content_hash}.zst"
                    ),
                    "compression": "zstd",
                    "compressionLevel": 9,
                    "uncompressedBytes": len(payload),
                    "compressedBytes": len(compressed),
                },
                "sequence": {
                    "start": 1_000,
                    "step": 300_000,
                    "count": 2,
                    "unit": "unix-ms",
                },
                "layout": {
                    "encoding": "derivatives-metrics-columnar-v1",
                    "columns": [
                        {
                            "name": "first",
                            "encoding": "float64-le",
                            "offset": 0,
                            "bytes": first.nbytes,
                        },
                        {
                            "name": "second",
                            "encoding": "float64-le",
                            "offset": first.nbytes,
                            "bytes": second.nbytes,
                        },
                        {
                            "name": "validMask",
                            "encoding": "uint8",
                            "offset": first.nbytes + second.nbytes,
                            "bytes": masks.nbytes,
                        },
                    ],
                },
            }), encoding="utf-8")
            values, validity = read_derivatives_metrics_columns(
                reference_file, ("first", "second"),
            )
            np.testing.assert_array_equal(values["first"], first)
            np.testing.assert_array_equal(values["second"], second)
            np.testing.assert_array_equal(validity["first"], [True, False])
            np.testing.assert_array_equal(validity["second"], [True, True])

    def test_checkpoint_gc_scans_only_nested_checkpoint_pointers(self) -> None:
        with TemporaryDirectory() as temporary:
            training_root = Path(temporary) / "data" / "training"
            store_root = training_root / "immutable"
            object_root = store_root / "objects" / "sha256"
            protected_hash = "11" * 32
            orphan_hash = "22" * 32
            fresh_hash = "33" * 32

            def object_file(content_hash: str) -> Path:
                file = (
                    object_root
                    / content_hash[:2]
                    / f"{content_hash}.bin"
                )
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"x")
                return file

            protected = object_file(protected_hash)
            orphan = object_file(orphan_hash)
            fresh = object_file(fresh_hash)
            old = time.time() - 2 * 60 * 60
            os.utime(protected, (old, old))
            os.utime(orphan, (old, old))

            nested_pointer = (
                training_root
                / "runs"
                / "disposable-smoke"
                / "nested-run"
                / "checkpoints"
                / "selections"
                / "best.json"
            )
            nested_pointer.parent.mkdir(parents=True, exist_ok=True)
            nested_pointer.write_text(
                json.dumps(self.checkpoint_reference(protected_hash)),
                encoding="utf-8",
            )
            sequential_reference = (
                store_root / "refs" / "oracle" / "day.json"
            )
            sequential_reference.parent.mkdir(parents=True, exist_ok=True)
            sequential_reference.write_text(
                "this file must not be opened by checkpoint GC",
                encoding="utf-8",
            )

            original_read_text = Path.read_text
            opened: list[Path] = []

            def monitored_read_text(file: Path, *args, **kwargs):
                resolved = file.resolve()
                try:
                    resolved.relative_to((store_root / "refs").resolve())
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "checkpoint GC opened a sequential-shard reference"
                    )
                opened.append(resolved)
                return original_read_text(file, *args, **kwargs)

            with patch.object(Path, "read_text", monitored_read_text), \
                    patch.dict(
                        os.environ,
                        {"TRADING_STORAGE_ORPHAN_GRACE_HOURS": "1"},
                    ):
                _prune_checkpoint_orphans(store_root)

            self.assertEqual(opened, [nested_pointer.resolve()])
            self.assertTrue(protected.is_file())
            self.assertFalse(orphan.exists())
            self.assertTrue(fresh.is_file())

    def test_checkpoint_gc_aborts_on_invalid_artifact_pointer(self) -> None:
        with TemporaryDirectory() as temporary:
            training_root = Path(temporary) / "data" / "training"
            store_root = training_root / "immutable"
            orphan_hash = "44" * 32
            orphan = (
                store_root
                / "objects"
                / "sha256"
                / orphan_hash[:2]
                / f"{orphan_hash}.bin"
            )
            orphan.parent.mkdir(parents=True, exist_ok=True)
            orphan.write_bytes(b"x")
            old = time.time() - 2 * 60 * 60
            os.utime(orphan, (old, old))

            invalid_pointer = (
                training_root
                / "runs"
                / "nested"
                / "checkpoints"
                / "last.json"
            )
            invalid_pointer.parent.mkdir(parents=True, exist_ok=True)
            invalid_reference = self.checkpoint_reference("55" * 32)
            invalid_reference["object"]["file"] = "../../escape.bin"
            invalid_pointer.write_text(
                json.dumps(invalid_reference),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"TRADING_STORAGE_ORPHAN_GRACE_HOURS": "1"},
            ):
                _prune_checkpoint_orphans(store_root)
            self.assertTrue(orphan.is_file())


if __name__ == "__main__":
    unittest.main()
