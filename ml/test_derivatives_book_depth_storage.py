import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from trading_storage import (
    DERIVATIVES_BOOK_DEPTH_BANDS,
    DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS,
    read_derivatives_book_depth_columns,
    write_shard_payload,
)


class DerivativesBookDepthStorageTests(unittest.TestCase):
    @staticmethod
    def arrays() -> dict[str, np.ndarray]:
        result = {
            "timestampOffsetSeconds": np.asarray([4, 34], dtype="<u4"),
            "schemaBandCount": np.asarray([10, 12], dtype="u1"),
            "bandAvailable": np.asarray([
                [0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ], dtype="u1"),
        }
        for offset, name in enumerate(DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS):
            result[name] = np.asarray([
                [0, 10 + offset, 20 + offset, 30 + offset,
                 40 + offset, 50 + offset],
                [5 + offset, 15 + offset, 25 + offset, 35 + offset,
                 45 + offset, 55 + offset],
            ], dtype="<f8")
        return result

    @classmethod
    def write_reference(
        cls,
        root: Path,
        arrays: dict[str, np.ndarray] | None = None,
    ) -> Path:
        values = cls.arrays() if arrays is None else arrays
        columns: list[dict[str, object]] = []
        chunks: list[bytes] = []
        offset = 0
        specifications = (
            ("timestampOffsetSeconds", "uint32-le", (2,)),
            ("schemaBandCount", "uint8", (2,)),
            *((name, "float64-le", (2, 6))
              for name in DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS),
            ("bandAvailable", "uint8", (2, 6)),
        )
        for name, encoding, shape in specifications:
            encoded = np.asarray(values[name]).tobytes()
            chunks.append(encoded)
            columns.append({
                "name": name,
                "encoding": encoding,
                "offset": offset,
                "bytes": len(encoded),
                "shape": list(shape),
            })
            offset += len(encoded)
        return write_shard_payload(
            root,
            "derivatives-book-depth/usdm-futures/btcusdt",
            "2026-01-01",
            b"".join(chunks),
            sequence={"start": 0, "step": 1, "count": 2, "unit": "index"},
            layout={
                "encoding": "derivatives-book-depth-columnar-v1",
                "columns": columns,
                "utcDayStartMs": 1_767_225_600_000,
                "timestampResolutionMs": 1_000,
                "timestampStorage": "utc-day-second-offset",
                "bandPercentages": list(DERIVATIVES_BOOK_DEPTH_BANDS),
                "percentageSemantics": "negative-bid-positive-ask",
                "valuesAreCumulative": True,
                "matrixOrder": "snapshot-major-band-minor",
            },
        )

    def test_reads_fixed_matrices_and_exact_ten_twelve_band_masks(self) -> None:
        with TemporaryDirectory() as temporary:
            reference = self.write_reference(Path(temporary))
            result = read_derivatives_book_depth_columns(reference)
            self.assertEqual(tuple(result), (
                "timestampOffsetSeconds",
                "schemaBandCount",
                "bidDepth",
                "askDepth",
                "bidNotional",
                "askNotional",
                "bandAvailable",
            ))
            np.testing.assert_array_equal(
                result["timestampOffsetSeconds"], [4, 34],
            )
            np.testing.assert_array_equal(result["schemaBandCount"], [10, 12])
            np.testing.assert_array_equal(result["bandAvailable"], [
                [False, True, True, True, True, True],
                [True, True, True, True, True, True],
            ])
            self.assertEqual(result["bidDepth"].shape, (2, 6))

    def test_rejects_axis_availability_and_cumulative_corruption(self) -> None:
        cases: list[tuple[str, dict[str, np.ndarray], str]] = []
        decreasing = self.arrays()
        decreasing["timestampOffsetSeconds"] = np.asarray([34, 4], dtype="<u4")
        cases.append(("time", decreasing, "snapshot axis"))
        mask = self.arrays()
        mask["bandAvailable"] = mask["bandAvailable"].copy()
        mask["bandAvailable"][0, 0] = 1
        cases.append(("mask", mask, "availability"))
        monotonic = self.arrays()
        monotonic["askDepth"] = monotonic["askDepth"].copy()
        monotonic["askDepth"][1, 4] = 1
        cases.append(("monotonic", monotonic, "non-monotone"))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, arrays, message in cases:
                with self.subTest(name=name):
                    reference = self.write_reference(root / name, arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        read_derivatives_book_depth_columns(reference)

    def test_requires_exact_storage_layout(self) -> None:
        with TemporaryDirectory() as temporary:
            reference = self.write_reference(Path(temporary))
            manifest = json.loads(reference.read_text(encoding="utf-8"))
            manifest["layout"]["timestampResolutionMs"] = 999
            reference.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layout"):
                read_derivatives_book_depth_columns(reference)


if __name__ == "__main__":
    unittest.main()
