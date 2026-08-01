import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import zstandard

from trading_storage import (
    checkpoint_exists,
    implicit_times,
    load_torch_checkpoint,
    read_shard_array,
    save_torch_checkpoint,
)
from train_mlp import load_compressed_component


class TradingStorageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
