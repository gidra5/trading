from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from audit_v18_futures_metrics_fusion import (
    DIRECTIONAL_MAGNITUDE_CONTROLS,
    EMBARGO_ROWS,
    FEATURE_INDEX,
    FirstHalfComplementaritySelection,
    METRIC_COLUMNS,
    METRIC_ROWS,
    REGIMES,
    embargoed_split_metrics,
    mean_kl_dense,
    primary_development_screen,
    read_futures_metrics_day,
    select_first_half_complementarity,
    validate_segment_clock_alignment,
    validate_futures_reference_coverage,
)


class V18FuturesMetricsFusionTests(unittest.TestCase):
    def test_every_signed_feature_has_paired_absolute_control(self) -> None:
        for regime in REGIMES:
            self.assertTrue(set(regime.control_features).isdisjoint(
                regime.directional_features
            ))
            self.assertEqual(
                len(regime.control_features),
                len(regime.control_bins),
            )
            self.assertEqual(
                len(regime.directional_features),
                len(regime.directional_bins),
            )
            for name in regime.joint_features:
                self.assertIn(name, FEATURE_INDEX)
            for directional in regime.directional_features:
                self.assertIn(
                    DIRECTIONAL_MAGNITUDE_CONTROLS[directional],
                    regime.control_features,
                )

    def test_first_half_can_select_complement_over_better_table(self) -> None:
        targets = np.tile((0.9, 0.1), (12, 1)).astype(np.float64)
        base = np.tile((0.8, 0.2), (12, 1)).astype(np.float64)
        better_standalone = np.tile((0.85, 0.15), (12, 1))
        complementary_control = np.tile((0.4, 0.6), (12, 1))
        worse_standalone = np.tile((0.6, 0.4), (12, 1))
        self.assertLess(
            mean_kl_dense(targets, better_standalone),
            mean_kl_dense(targets, worse_standalone),
        )
        selected, candidates = select_first_half_complementarity(
            targets,
            base,
            (
                (
                    "better-standalone",
                    better_standalone,
                    better_standalone,
                ),
                (
                    "better-complement",
                    complementary_control,
                    worse_standalone,
                ),
            ),
        )
        self.assertEqual(selected.regime_name, "better-complement")
        self.assertEqual(
            {candidate.regime_name for candidate in candidates},
            {"better-standalone", "better-complement"},
        )
        self.assertLess(selected.raw_kl, 1e-12)

    def test_primary_gate_never_reselects_from_holdout_diagnostics(self) -> None:
        selected = FirstHalfComplementaritySelection(
            "first-half-winner",
            0.5,
            0.1,
        )
        metrics = {
            "first-half-winner": {
                "secondHalfKlReductionFromV18": 0.001,
            },
            "holdout-winner": {
                "secondHalfKlReductionFromV18": 0.5,
            },
        }
        result = primary_development_screen(selected, metrics)
        self.assertEqual(result["selectedRegime"], "first-half-winner")
        self.assertFalse(result["passesWithinAudit0.002Screen"])

    def test_forecast_horizon_embargo_is_sixty_rows(self) -> None:
        self.assertEqual(EMBARGO_ROWS, 60)

    def test_segment_clock_is_bound_to_target_row_offset(self) -> None:
        day = Path("2026-01-02.json")
        aligned = SimpleNamespace(
            split="train",
            target_file=day,
            target_row_offset=17,
            count=100,
            step_ms=60_000,
            prediction_time_start=(
                1_767_312_000_000 + 999 + 17 * 60_000
            ),
        )
        segments = {"train": [aligned], "validation": [], "test": []}
        validate_segment_clock_alignment(segments)
        for name, value in (
            ("prediction_time_start", aligned.prediction_time_start + 1),
            ("target_row_offset", 18),
            ("step_ms", 1_000),
            ("count", 1_500),
        ):
            malformed = SimpleNamespace(**vars(aligned))
            setattr(malformed, name, value)
            with self.assertRaisesRegex(ValueError, "misaligned"):
                validate_segment_clock_alignment({
                    "train": [malformed], "validation": [], "test": [],
                })

    def test_embargoed_metrics_exclude_embargo_only_from_holdout(self) -> None:
        targets = np.full((8, 3), 0.1, dtype=np.float64)
        targets[:, 0] = 0.8
        probabilities = targets.copy()
        probabilities[3:5] = (0.1, 0.1, 0.8)
        metrics = embargoed_split_metrics(
            targets,
            probabilities,
            split_at=3,
            holdout_start=5,
        )
        self.assertEqual(metrics["firstHalfKl"], 0)
        self.assertEqual(metrics["secondHalfKl"], 0)
        self.assertAlmostEqual(
            metrics["fullValidationKl"],
            mean_kl_dense(targets, probabilities),
        )
        self.assertGreater(metrics["fullValidationKl"], 0)

    def test_coverage_preflight_requires_complete_non_test_scope(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [SimpleNamespace(target_file=Path("2026-01-02.json"))]
            validation = [
                SimpleNamespace(target_file=Path("2026-01-03.json"))
            ]
            segments = {"train": train, "validation": validation, "test": []}
            for day_value in ("2026-01-01", "2026-01-02", "2026-01-03"):
                (root / f"{day_value}.json").write_text(
                    "payload must not be parsed by coverage preflight",
                    encoding="utf-8",
                )
            actual = validate_futures_reference_coverage(
                root,
                segments,
                sealed_test_start="2026-01-04",
                expected_count=3,
            )
            self.assertEqual(
                actual,
                ("2026-01-01", "2026-01-02", "2026-01-03"),
            )
            (root / "2026-01-02.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "ingestion incomplete"):
                validate_futures_reference_coverage(
                    root,
                    segments,
                    sealed_test_start="2026-01-04",
                    expected_count=3,
                )
            (root / "2026-01-02.json").write_text("restored", encoding="utf-8")
            (root / "2025-12-31.json").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the fixed"):
                validate_futures_reference_coverage(
                    root,
                    segments,
                    sealed_test_start="2026-01-04",
                    expected_count=3,
                )
            (root / "2025-12-31.json").unlink()
            segments["validation"] = [
                SimpleNamespace(target_file=Path("2026-01-04.json"))
            ]
            with self.assertRaisesRegex(ValueError, "sealed-test boundary"):
                validate_futures_reference_coverage(
                    root,
                    segments,
                    sealed_test_start="2026-01-04",
                    expected_count=4,
                )

    def test_reader_enforces_lag_contract_and_payload_counters(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "2024-01-01.json"
            missing_counts = {name: 0 for name in METRIC_COLUMNS}
            metadata = {
                "featureSchema": "binance-usdm-futures-metrics-v1",
                "source": "data.binance.vision",
                "sourceDataset": "futures/um/daily/metrics",
                "sourceArchiveUrl": (
                    "https://data.binance.vision/data/futures/um/daily/metrics/"
                    "BTCUSDT/BTCUSDT-metrics-2024-01-01.zip"
                ),
                "sourceArchiveSha256": "a" * 64,
                "sourceArchiveBytes": 1,
                "sourceCsvEntry": "BTCUSDT-metrics-2024-01-01.csv",
                "sourceCsvBytes": 1,
                "sourceCsvRows": METRIC_ROWS,
                "observedGridRows": METRIC_ROWS,
                "missingGridRows": 0,
                "outsideUtcDayRows": 0,
                "offGridRows": 0,
                "timestampAdjustedRows": 0,
                "missingValueCounts": missing_counts,
                "market": "usdm-futures",
                "symbol": "BTCUSDT",
                "interval": "5m",
                "denseUtcDayAxis": True,
                "availabilityLagMs": 300_000,
                "oracleTargetContract": "target-contract",
                "oracleScope": "train-or-validation-target",
                "sealedTestStart": "2026-06-24",
                "sealedTestEnd": "2026-07-23",
            }
            manifest = {
                "sequence": {
                    "start": 1_704_067_200_000,
                    "step": 300_000,
                    "count": METRIC_ROWS,
                    "unit": "unix-ms",
                },
                "layout": {"encoding": "derivatives-metrics-columnar-v1"},
                "metadata": metadata,
            }
            reference.write_text(json.dumps(manifest), encoding="utf-8")
            values = {
                name: np.ones(METRIC_ROWS, dtype=np.float64)
                for name in METRIC_COLUMNS
            }
            validity = {
                name: np.ones(METRIC_ROWS, dtype=bool)
                for name in METRIC_COLUMNS
            }
            with patch(
                "audit_v18_futures_metrics_fusion."
                "read_derivatives_metrics_columns",
                return_value=(values, validity),
            ):
                loaded = read_futures_metrics_day(
                    root,
                    "2024-01-01",
                    set(),
                    target_contract="target-contract",
                    sealed_test_start="2026-06-24",
                    sealed_test_end="2026-07-23",
                )
                self.assertIs(loaded[0], values)
                self.assertIs(loaded[1], validity)
                metadata["availabilityLagMs"] = 0
                reference.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source contract"):
                    read_futures_metrics_day(
                        root,
                        "2024-01-01",
                        set(),
                        target_contract="target-contract",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )
                metadata["availabilityLagMs"] = 300_000
                metadata["timestampAdjustedRows"] = 1
                reference.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "quality counters"):
                    read_futures_metrics_day(
                        root,
                        "2024-01-01",
                        set(),
                        target_contract="target-contract",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )
                metadata["timestampAdjustedRows"] = 0
                metadata["offGridRows"] = 1
                reference.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "quality counters"):
                    read_futures_metrics_day(
                        root,
                        "2024-01-01",
                        set(),
                        target_contract="target-contract",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )
                with self.assertRaisesRegex(ValueError, "sealed-test"):
                    read_futures_metrics_day(
                        root,
                        "2026-06-24",
                        set(),
                        target_contract="target-contract",
                        sealed_test_start="2026-06-24",
                        sealed_test_end="2026-07-23",
                    )


if __name__ == "__main__":
    unittest.main()
