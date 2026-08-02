from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from audit_v18_futures_basis_fusion import (
    DAY_ROWS,
    FEATURE_INDEX,
    FUTURES_BASIS_FEATURE_NAMES,
    FUTURES_KLINE_COLUMNS,
    REGIMES,
    SECOND_ROWS,
    SPOT_CROSS_FEATURE_NAMES,
    completed_cross_market_flow_features,
    completed_spot_quote_features,
    encode_cells,
    fit_edges,
    read_futures_kline_day,
    validate_reference_coverage,
)


class V18FuturesBasisFusionTests(unittest.TestCase):
    def test_fixed_regimes_use_available_nonredundant_features(self) -> None:
        self.assertEqual(
            [regime.name for regime in REGIMES],
            [
                "basis-level-reversion",
                "basis-change",
                "futures-taker-pressure",
                "futures-spot-imbalance-divergence-agreement",
                "relative-activity",
            ],
        )
        for regime in REGIMES:
            self.assertEqual(len(regime.control_features), len(regime.control_bins))
            self.assertEqual(
                len(regime.directional_features),
                len(regime.directional_bins),
            )
            self.assertTrue(set(regime.control_features).isdisjoint(
                regime.directional_features
            ))
            for name in regime.joint_features:
                self.assertIn(name, FEATURE_INDEX)
        self.assertFalse(any(
            "FuturesMinusSpot" in name and "Return" in name
            for name in FEATURE_INDEX
        ))
        self.assertIn("basisLogChange1h", FUTURES_BASIS_FEATURE_NAMES)

    def test_spot_quote_imbalance_uses_completed_previous_minute(self) -> None:
        def empty() -> dict[str, np.ndarray]:
            return {
                "aggressiveBuyQuoteVolume": np.zeros(
                    SECOND_ROWS, dtype=np.float64,
                ),
                "aggressiveSellQuoteVolume": np.zeros(
                    SECOND_ROWS, dtype=np.float64,
                ),
            }

        previous = empty()
        current = empty()
        previous["aggressiveBuyQuoteVolume"][-60:] = 1
        # This is target row zero's current second and must not enter row zero.
        current["aggressiveSellQuoteVolume"][0] = 1_000
        imbalance, _activity, quote = completed_spot_quote_features(
            previous,
            current,
        )
        self.assertEqual(imbalance[0], 1)
        self.assertEqual(quote[0], 60)
        # It becomes eligible only at minute one, after minute zero closes.
        self.assertEqual(imbalance[1], -1)
        self.assertEqual(quote[1], 1_000)
        current["aggressiveSellQuoteVolume"][0] = 2_000
        changed, _activity, changed_quote = completed_spot_quote_features(
            previous,
            current,
        )
        self.assertEqual(changed[0], imbalance[0])
        self.assertEqual(changed_quote[0], quote[0])
        self.assertEqual(changed[1], -1)
        self.assertNotEqual(changed_quote[1], quote[1])

    def test_cross_features_distinguish_live_no_trade_and_missing_rows(self) -> None:
        previous_flow = {
            "aggressiveBuyQuoteVolume": np.zeros(SECOND_ROWS),
            "aggressiveSellQuoteVolume": np.zeros(SECOND_ROWS),
        }
        current_flow = {
            "aggressiveBuyQuoteVolume": np.zeros(SECOND_ROWS),
            "aggressiveSellQuoteVolume": np.zeros(SECOND_ROWS),
        }
        previous_flow["aggressiveBuyQuoteVolume"][-60:] = 1
        current_flow["aggressiveBuyQuoteVolume"][:120] = 1
        previous_futures = {"quoteVolume": np.zeros(DAY_ROWS)}
        current_futures = {"quoteVolume": np.zeros(DAY_ROWS)}
        previous_validity = np.zeros(DAY_ROWS, dtype=bool)
        current_validity = np.zeros(DAY_ROWS, dtype=bool)
        previous_validity[-1] = True
        previous_futures["quoteVolume"][-1] = 120
        current_validity[0] = True  # Official no-trade row, quote volume zero.
        basis = np.zeros((DAY_ROWS, len(FUTURES_BASIS_FEATURE_NAMES)))
        basis_index = {
            name: index
            for index, name in enumerate(FUTURES_BASIS_FEATURE_NAMES)
        }
        basis[0, basis_index["futuresRowCurrentObserved"]] = 1
        basis[1, basis_index["futuresRowCurrentObserved"]] = 1
        basis[1, basis_index["futuresNoTradeCurrent"]] = 1
        result = completed_cross_market_flow_features(
            previous_flow,
            current_flow,
            previous_futures,
            previous_validity,
            current_futures,
            current_validity,
            basis,
        )
        state = SPOT_CROSS_FEATURE_NAMES.index("futuresSourceState")
        np.testing.assert_array_equal(result[:3, state], (2, 1, 0))
        exact_ratio = SPOT_CROSS_FEATURE_NAMES.index(
            "futuresSpotLogQuoteVolumeRatio1m"
        )
        self.assertGreater(result[0, exact_ratio], 0)
        self.assertLess(result[1, exact_ratio], 0)
        self.assertEqual(result[2, exact_ratio], 0)

    def test_source_state_bins_do_not_collapse_rare_missing_or_no_trade(self) -> None:
        features = np.zeros((100, len(FEATURE_INDEX)), dtype=np.float64)
        state_index = FEATURE_INDEX["futuresSourceState"]
        features[:, state_index] = 2
        features[0, state_index] = 0
        features[1, state_index] = 1
        edges = fit_edges(features, ("futuresSourceState",), (3,))
        np.testing.assert_array_equal(edges[0], (0.5, 1.5))
        ids, count = encode_cells(
            features[:3],
            ("futuresSourceState",),
            edges,
        )
        np.testing.assert_array_equal(ids, (0, 1, 2))
        self.assertEqual(count, 3)

    def test_preflight_requires_exact_complete_non_test_namespace(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments = {
                "train": [SimpleNamespace(
                    target_file=Path("2026-01-02.json")
                )],
                "validation": [SimpleNamespace(
                    target_file=Path("2026-01-03.json")
                )],
                "test": [],
            }
            for day_value in ("2026-01-01", "2026-01-02", "2026-01-03"):
                (root / f"{day_value}.json").write_text(
                    "preflight must not parse this payload",
                    encoding="utf-8",
                )
            actual = validate_reference_coverage(
                root,
                segments,
                sealed_test_start="2026-01-04",
                label="test source",
                expected_count=3,
            )
            self.assertEqual(actual, (
                "2026-01-01", "2026-01-02", "2026-01-03",
            ))
            (root / "2025-12-31.json").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the fixed"):
                validate_reference_coverage(
                    root,
                    segments,
                    sealed_test_start="2026-01-04",
                    label="test source",
                    expected_count=3,
                )
            (root / "2025-12-31.json").unlink()
            segments["validation"] = [SimpleNamespace(
                target_file=Path("2026-01-04.json")
            )]
            with self.assertRaisesRegex(ValueError, "sealed-test boundary"):
                validate_reference_coverage(
                    root,
                    segments,
                    sealed_test_start="2026-01-04",
                    label="test source",
                    expected_count=4,
                )

    def test_futures_reader_enforces_close_scope_and_payload_counts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "2024-01-01.json"
            archive = "BTCUSDT-1m-2024-01-01.zip"
            metadata = {
                "featureSchema": "binance-usdm-futures-klines-v1",
                "source": "data.binance.vision",
                "sourceDataset": "futures/um/daily/klines",
                "sourceArchiveUrl": (
                    "https://data.binance.vision/data/futures/um/daily/klines/"
                    f"BTCUSDT/1m/{archive}"
                ),
                "sourceArchiveChecksumUrl": (
                    "https://data.binance.vision/data/futures/um/daily/klines/"
                    f"BTCUSDT/1m/{archive}.CHECKSUM"
                ),
                "sourceArchiveChecksumAlgorithm": "sha256",
                "sourceArchiveChecksumFilename": archive,
                "sourceArchiveSha256": "a" * 64,
                "sourceArchiveBytes": 1,
                "sourceCsvEntry": "BTCUSDT-1m-2024-01-01.csv",
                "sourceCsvBytes": 1,
                "sourceCsvRows": DAY_ROWS,
                "sourceCsvHeaderRows": 1,
                "sourceTimestampUnit": "millisecond",
                "observedGridRows": DAY_ROWS,
                "liveGridRows": DAY_ROWS,
                "noTradeGridRows": 0,
                "missingGridRows": 0,
                "outsideUtcDayRows": 0,
                "offGridRows": 0,
                "timestampAdjustedRows": 0,
                "filledGridRows": 0,
                "market": "usdm-futures",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "denseUtcDayAxis": True,
                "rowValidity": "official-source-row-present",
                "liveObservationRule": "validMask && tradeCount > 0",
                "closeAvailability": "openTime+59999ms",
                "closeTimeOffsetMs": 59_999,
                "oracleTargetContract": "target-contract",
                "oracleScope": "train-or-validation-target",
                "sealedTestStart": "2026-06-24",
                "sealedTestEnd": "2026-07-23",
            }
            manifest = {
                "sequence": {
                    "start": 1_704_067_200_000,
                    "step": 60_000,
                    "count": DAY_ROWS,
                    "unit": "unix-ms",
                },
                "layout": {
                    "encoding": "derivatives-klines-columnar-v1",
                    "closeTimeOffsetMs": 59_999,
                    "closed": True,
                },
                "metadata": metadata,
            }
            reference.write_text(json.dumps(manifest), encoding="utf-8")
            values = {
                name: np.ones(DAY_ROWS, dtype=np.float64)
                for name in FUTURES_KLINE_COLUMNS
            }
            values["tradeCount"] = np.ones(DAY_ROWS, dtype=np.uint64)
            validity = np.ones(DAY_ROWS, dtype=bool)
            with patch(
                "audit_v18_futures_basis_fusion."
                "read_derivatives_kline_columns",
                return_value=(values, validity),
            ):
                loaded = read_futures_kline_day(
                    root,
                    "2024-01-01",
                    set(),
                    target_contract="target-contract",
                    oracle_scope="train-or-validation-target",
                    sealed_test_start="2026-06-24",
                    sealed_test_end="2026-07-23",
                )
                self.assertIs(loaded[0], values)
                self.assertIs(loaded[1], validity)
                metadata["closeTimeOffsetMs"] = 60_000
                reference.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source contract"):
                    read_futures_kline_day(
                        root,
                        "2024-01-01",
                        set(),
                        target_contract="target-contract",
                        oracle_scope="train-or-validation-target",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )
                metadata["closeTimeOffsetMs"] = 59_999
                metadata["observedGridRows"] = DAY_ROWS - 1
                reference.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "quality counters"):
                    read_futures_kline_day(
                        root,
                        "2024-01-01",
                        set(),
                        target_contract="target-contract",
                        oracle_scope="train-or-validation-target",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )
                with self.assertRaisesRegex(ValueError, "sealed-test"):
                    read_futures_kline_day(
                        root,
                        "2026-06-24",
                        set(),
                        target_contract="target-contract",
                        oracle_scope="predecessor-context",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )


if __name__ == "__main__":
    unittest.main()
