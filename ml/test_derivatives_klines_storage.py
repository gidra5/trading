import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from trading_storage import (
    DERIVATIVES_KLINE_COLUMNS,
    read_derivatives_kline_columns,
    write_shard_payload,
)


class DerivativesKlinesStorageTests(unittest.TestCase):
    @staticmethod
    def write_reference(
        root: Path,
        key: str,
        columns: dict[str, np.ndarray],
        validity: np.ndarray,
    ) -> Path:
        chunks: list[bytes] = []
        layout_columns: list[dict[str, object]] = []
        offset = 0
        count = len(validity)
        for name in DERIVATIVES_KLINE_COLUMNS:
            encoding = "uint64-le" if name == "tradeCount" else "float64-le"
            dtype = "<u8" if name == "tradeCount" else "<f8"
            encoded = np.asarray(columns[name], dtype=dtype).tobytes()
            chunks.append(encoded)
            layout_columns.append({
                "name": name,
                "encoding": encoding,
                "offset": offset,
                "bytes": len(encoded),
            })
            offset += len(encoded)
        mask = np.asarray(validity, dtype="u1").tobytes()
        chunks.append(mask)
        layout_columns.append({
            "name": "validMask",
            "encoding": "uint8",
            "offset": offset,
            "bytes": len(mask),
        })
        return write_shard_payload(
            root,
            "derivatives-klines/usdm-futures/btcusdt/1m",
            key,
            b"".join(chunks),
            sequence={
                "start": 1_767_225_600_000,
                "step": 60_000,
                "count": count,
                "unit": "unix-ms",
            },
            layout={
                "encoding": "derivatives-klines-columnar-v1",
                "columns": layout_columns,
                "closeTimeOffsetMs": 59_999,
                "closed": True,
            },
        )

    @staticmethod
    def columns() -> dict[str, np.ndarray]:
        return {
            "open": np.array([100.0, 0.0]),
            "high": np.array([102.0, 0.0]),
            "low": np.array([99.0, 0.0]),
            "close": np.array([101.0, 0.0]),
            "baseVolume": np.array([10.0, 0.0]),
            "quoteVolume": np.array([1_005.0, 0.0]),
            "tradeCount": np.array([17, 0], dtype="<u8"),
            "takerBuyBaseVolume": np.array([4.0, 0.0]),
            "takerBuyQuoteVolume": np.array([402.0, 0.0]),
        }

    def test_reads_selected_columns_with_one_shared_full_row_mask(self) -> None:
        with TemporaryDirectory() as temporary:
            reference = self.write_reference(
                Path(temporary),
                "2026-01-01",
                self.columns(),
                np.array([1, 0], dtype="u1"),
            )
            values, validity = read_derivatives_kline_columns(
                reference,
                ("close", "tradeCount", "takerBuyBaseVolume"),
            )
            self.assertEqual(tuple(values), (
                "close",
                "tradeCount",
                "takerBuyBaseVolume",
            ))
            np.testing.assert_array_equal(values["close"], [101.0, 0.0])
            np.testing.assert_array_equal(values["tradeCount"], [17, 0])
            np.testing.assert_array_equal(validity, [True, False])

    def test_rejects_cross_column_and_missing_row_invariant_violations(self) -> None:
        cases = []
        invalid_taker = self.columns()
        invalid_taker["takerBuyBaseVolume"] = np.array([11.0, 0.0])
        cases.append(("taker", invalid_taker, np.array([1, 0], dtype="u1")))
        hidden_data = self.columns()
        cases.append(("missing", hidden_data, np.array([0, 0], dtype="u1")))
        invalid_high = self.columns()
        invalid_high["high"] = np.array([100.5, 0.0])
        cases.append(("ohlc", invalid_high, np.array([1, 0], dtype="u1")))
        no_trade_with_flow = self.columns()
        no_trade_with_flow["tradeCount"] = np.array([0, 0], dtype="<u8")
        cases.append((
            "no-trade", no_trade_with_flow, np.array([1, 0], dtype="u1")
        ))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, columns, validity in cases:
                with self.subTest(key=key):
                    reference = self.write_reference(root, key, columns, validity)
                    with self.assertRaisesRegex(ValueError, "derivatives-klines"):
                        read_derivatives_kline_columns(reference)

    def test_requires_close_availability_metadata(self) -> None:
        with TemporaryDirectory() as temporary:
            reference = self.write_reference(
                Path(temporary),
                "bad-layout",
                self.columns(),
                np.array([1, 0], dtype="u1"),
            )
            value = json.loads(reference.read_text(encoding="utf-8"))
            value["layout"]["closeTimeOffsetMs"] = 60_000
            reference.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "time layout"):
                read_derivatives_kline_columns(reference)


if __name__ == "__main__":
    unittest.main()
