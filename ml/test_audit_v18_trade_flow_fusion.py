import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from audit_v18_trade_flow_fusion import (
    EMBARGO_ROWS,
    FEATURE_INDEX,
    REGIMES,
    CORPUS_SPLIT_CONTRACT,
    SECOND_ROWS,
    TRADE_FLOW_COLUMNS,
    embargoed_split_metrics,
    mean_kl_dense,
    read_trade_flow_day,
    reconcile_trade_flow_with_candles,
    rolling_tick_rule_proxy,
    validate_corpus_split_contract,
    validate_runtime_split_assignment,
)


class V18TradeFlowFusionTests(unittest.TestCase):
    def test_every_regime_is_a_matched_control_plus_direction_superset(self) -> None:
        for regime in REGIMES:
            self.assertTrue(set(regime.control_features).isdisjoint(
                regime.directional_features
            ))
            self.assertEqual(
                len(regime.control_features), len(regime.control_bins)
            )
            self.assertEqual(
                len(regime.directional_features), len(regime.directional_bins)
            )
            for name in regime.joint_features:
                self.assertIn(name, FEATURE_INDEX)
            exact_magnitude = {
                "quoteImbalance1s": "absQuoteImbalance1s",
                "quoteImbalance5s": "absQuoteImbalance5s",
                "quoteImbalance60s": "absQuoteImbalance60s",
                "quoteImbalance5m": "absQuoteImbalance5m",
                "tradeCountImbalance5s": "absTradeCountImbalance5s",
                "tradeCountImbalance60s": "absTradeCountImbalance60s",
                "tradeCountImbalance5m": "absTradeCountImbalance5m",
                "quantitySquaredSkew60s": "absQuantitySquaredSkew60s",
                "maxAggregateSkew60s": "absMaxAggregateSkew60s",
                "signedVwapGap60s": "absVwapGap60s",
                "signedArrivalCentroidGap60s": "absArrivalCentroidGap60s",
                "lastAggressorSide1s": "absLastAggressorSide1s",
            }
            for directional in regime.directional_features:
                self.assertIn(
                    exact_magnitude[directional], regime.control_features,
                )

    def test_tick_rule_proxy_ends_at_current_closed_second(self) -> None:
        previous = np.ones((86_400, 5), dtype=np.float64)
        current = np.ones((86_400, 5), dtype=np.float64)
        previous[:, 3] = 100
        current[:, 3] = 100
        previous[:, 4] = 0
        current[:, 4] = 0
        previous[-1, 3] = 99
        current[0, 3] = 101
        current[0, 4] = 2
        current[1, 3] = 50
        current[1, 4] = 1_000
        proxy = rolling_tick_rule_proxy(previous, current)
        self.assertEqual(proxy[0], 1)
        self.assertLess(proxy[1], 0)

    def test_trade_flow_reconciliation_matches_volume_and_side_vwap(self) -> None:
        candles = np.ones((86_400, 5), dtype=np.float64)
        candles[:, 1] = 102
        candles[:, 2] = 98
        candles[:, 4] = 0
        flow = {
            "aggressiveBuyBaseVolume": np.zeros(86_400),
            "aggressiveSellBaseVolume": np.zeros(86_400),
            "aggressiveBuyQuoteVolume": np.zeros(86_400),
            "aggressiveSellQuoteVolume": np.zeros(86_400),
        }
        flow["aggressiveBuyBaseVolume"][0] = 2
        flow["aggressiveBuyQuoteVolume"][0] = 202
        flow["aggressiveSellBaseVolume"][0] = 1
        flow["aggressiveSellQuoteVolume"][0] = 100
        candles[0, 4] = 3
        reconcile_trade_flow_with_candles("2026-01-01", candles, flow)
        candles[0, 4] = 4
        with self.assertRaisesRegex(ValueError, "volume mismatch"):
            reconcile_trade_flow_with_candles("2026-01-01", candles, flow)

    def test_forecast_horizon_embargo_is_sixty_minute_rows(self) -> None:
        self.assertEqual(EMBARGO_ROWS, 60)

    def test_embargoed_metrics_keep_holdout_purged_but_full_metric_full(self) -> None:
        targets = np.full((8, 3), 0.1, dtype=np.float64)
        targets[:, 0] = 0.8
        probabilities = targets.copy()
        probabilities[3:5] = (0.1, 0.1, 0.8)
        metrics = embargoed_split_metrics(
            targets, probabilities, split_at=3, holdout_start=5,
        )
        self.assertEqual(metrics["firstHalfKl"], 0)
        self.assertEqual(metrics["secondHalfKl"], 0)
        self.assertAlmostEqual(
            metrics["fullValidationKl"],
            mean_kl_dense(targets, probabilities),
        )
        self.assertGreater(metrics["fullValidationKl"], 0)

    def test_corpus_contract_rejects_appended_filename_before_payload_read(self) -> None:
        with TemporaryDirectory() as temporary:
            repo = Path(temporary)
            contract_file = repo / CORPUS_SPLIT_CONTRACT
            target_root = repo / "refs" / "contract"
            contract_file.parent.mkdir(parents=True)
            target_root.mkdir(parents=True)
            names = ["2026-01-01.json", "2026-01-02.json", "2026-01-03.json"]
            files = []
            for name in names:
                file = target_root / name
                file.write_text("must not be parsed", encoding="utf-8")
                files.append(file)
            contract_file.write_text(json.dumps({
                "schemaVersion": 1,
                "targetContract": "contract",
                "referenceCount": 3,
                "referenceFilenameSha256": hashlib.sha256(
                    ("\n".join(names) + "\n").encode("utf-8")
                ).hexdigest(),
                "filenameHashEncoding": "utf8-lf-with-trailing-lf",
                "train": {"count": 1, "first": "2026-01-01", "last": "2026-01-01"},
                "validation": {"count": 1, "first": "2026-01-02", "last": "2026-01-02"},
                "test": {
                    "count": 1,
                    "first": "2026-01-03",
                    "last": "2026-01-03",
                    "policy": "sealed-never-load",
                },
            }), encoding="utf-8")
            appended = target_root / "2026-01-04.json"
            appended.write_text("appended", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "filenames differ"):
                validate_corpus_split_contract(
                    repo, target_root, files + [appended],
                )

    def test_trade_flow_reader_reconciles_source_row_and_id_counters(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "2024-01-01.json"
            metadata = {
                "featureSchema": "binance-spot-agg-trade-flow-v1",
                "oracleTargetContract": "target-contract",
                "sealedTestStart": "2026-06-24",
                "sealedTestEnd": "2026-07-23",
                "completeUtcDay": True,
                "aggregateIdGapCount": "0",
                "source": "data.binance.vision",
                "sourceDataset": "spot/daily/aggTrades",
                "sourceArchiveUrl": (
                    "https://data.binance.vision/data/spot/daily/aggTrades/"
                    "BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip"
                ),
                "sourceArchiveSha256": "a" * 64,
                "sourceArchiveBytes": 1,
                "sourceCsvEntry": "BTCUSDT-aggTrades-2024-01-01.csv",
                "sourceCsvBytes": 1,
                "market": "spot",
                "symbol": "BTCUSDT",
                "interval": "1s",
                "sourceTimestampUnit": "millisecond",
                "sourceCsvRows": 2,
                "invalidSentinelRows": 0,
                "firstAggregateTradeId": "10",
                "lastAggregateTradeId": "11",
                "firstTradeId": "20",
                "lastTradeId": "22",
            }
            manifest = {
                "sequence": {
                    "start": 1_704_067_200_000,
                    "step": 1_000,
                    "count": SECOND_ROWS,
                    "unit": "unix-ms",
                },
                "layout": {"encoding": "trade-flow-columnar-v1"},
                "metadata": metadata,
            }
            reference.write_text(json.dumps(manifest), encoding="utf-8")
            values = {
                name: np.zeros(SECOND_ROWS, dtype=np.float64)
                for name in TRADE_FLOW_COLUMNS
            }
            values["aggressiveBuyAggregateTradeCount"][0] = 2
            values["aggressiveBuyTradeCount"][0] = 3
            with patch(
                "audit_v18_trade_flow_fusion.read_trade_flow_columns",
                return_value=values,
            ):
                loaded = read_trade_flow_day(
                    root,
                    "2024-01-01",
                    set(),
                    target_contract="target-contract",
                    sealed_test_start="2026-06-24",
                    sealed_test_end="2026-07-23",
                )
                self.assertIs(loaded, values)
                metadata["sourceCsvRows"] = 3
                reference.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "counters disagree"):
                    read_trade_flow_day(
                        root,
                        "2024-01-01",
                        set(),
                        target_contract="target-contract",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )

    def test_runtime_split_assignment_cannot_drift_from_contract(self) -> None:
        files = [Path(f"2026-01-0{index}.json") for index in range(1, 4)]
        contract = {
            "train": {"count": 1},
            "validation": {"count": 1},
            "test": {"count": 1},
        }
        valid = {
            split: [SimpleNamespace(split=split, target_file=file)]
            for split, file in zip(("train", "validation", "test"), files)
        }
        validate_runtime_split_assignment(files, valid, contract)
        drifted = {key: list(value) for key, value in valid.items()}
        drifted["train"] = [
            SimpleNamespace(split="train", target_file=files[1]),
        ]
        drifted["validation"] = [
            SimpleNamespace(split="validation", target_file=files[0]),
        ]
        with self.assertRaisesRegex(ValueError, "assignment differs"):
            validate_runtime_split_assignment(files, drifted, contract)


if __name__ == "__main__":
    unittest.main()
