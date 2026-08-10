from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from benchmark_kronos import (
    EXECUTION_ORACLE_CONFIG,
    Corpus,
    ForecastAccumulator,
    HORIZON,
    MetricBatch,
    Origin,
    ProbabilisticAccumulator,
    ProbabilisticBatch,
    atomic_json,
    benchmark_signature,
    oracle_path_distributions,
    seed_inference_batch,
)
from kronos_model_zoo import selected_specs
from differentiable_exposure_value_oracle import (
    DifferentiableExposureValueOracle,
    DifferentiableExposureValueOracleConfig,
)


class KronosBenchmarkStateTest(unittest.TestCase):
    def test_run_signature_depends_on_requested_batch_layout(self) -> None:
        corpus = Corpus(
            times=np.array([0], dtype=np.int64),
            values=np.zeros((1, 5), dtype=np.float64),
            references=("test",),
            fingerprint="corpus",
        )
        origins = (
            Origin(target_start=60_000, target_index=0, window_ids=("a",)),
        )
        common = {
            "lookback": 512,
            "sample_count": 20,
            "temperature": 0.8,
            "top_p": 0.9,
            "point_estimator": "ensembleMean",
            "seed": 1_337,
            "max_origins_per_window": 4,
            "model_checkpoint": None,
            "tokenizer_checkpoint": None,
            "model_label": None,
            "forecast_output": None,
            "ensemble_predictor_checkpoint": None,
        }
        first = benchmark_signature(
            windows=(),
            corpus=corpus,
            origins=origins,
            args=argparse.Namespace(**common, batch_size=1),
            specs=selected_specs("base"),
        )
        second = benchmark_signature(
            windows=(),
            corpus=corpus,
            origins=origins,
            args=argparse.Namespace(**common, batch_size=2),
            specs=selected_specs("base"),
        )

        self.assertNotEqual(first, second)

    def test_atomic_json_retries_a_transient_windows_replace_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            original_replace = Path.replace
            attempts = 0

            def transient_replace(path: Path, target: Path) -> Path:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("simulated destination read lock")
                return original_replace(path, target)

            with mock.patch.object(Path, "replace", transient_replace), \
                    mock.patch("benchmark_kronos.time.sleep") as sleep:
                atomic_json({"complete": True}, output)

            self.assertEqual(attempts, 2)
            sleep.assert_called_once_with(0.05)
            self.assertEqual(output.read_text(encoding="utf-8"), (
                '{\n  "complete": true\n}\n'
            ))

    def test_execution_oracle_matches_forecast_decision_cadence(self) -> None:
        self.assertEqual(EXECUTION_ORACLE_CONFIG.holding_period_steps, HORIZON)
        self.assertEqual(EXECUTION_ORACLE_CONFIG.decision_delay_steps, HORIZON)
        self.assertEqual(EXECUTION_ORACLE_CONFIG.value_horizon_steps, HORIZON)

    def test_forecast_accumulator_round_trip(self) -> None:
        batch_size = 2
        action_count = 3
        scalar_names = (
            "candleLogMse", "candlePersistenceMse", "rawPriceMse",
            "validOhlcFraction", "closeReturnMse", "closeReturnZeroMse",
            "directionAccuracy", "cumulativeMse", "zeroCumulativeMse",
            "horizonDirectionAccuracy", "closePathMse",
            "closePathPersistenceMse", "priceSeriesIc",
            "priceSeriesRankIc", "oracleKl", "oracleProbabilityMse",
            "oracleTotalVariation", "oracleModeAgreement",
            "oracleExpectedExposureMae", "oraclePredictedEntropy",
            "oracleActualEntropy",
        )
        scalars = {
            name: np.array([index + 1.0, index + 2.0])
            for index, name in enumerate(scalar_names)
        }
        candle_actual = np.arange(
            batch_size * HORIZON * 4, dtype=np.float64
        ).reshape(batch_size, HORIZON, 4)
        candle_predicted = candle_actual * 1.01 + 0.1
        close_actual = np.arange(
            batch_size * HORIZON, dtype=np.float64
        ).reshape(batch_size, HORIZON)
        close_predicted = close_actual * 0.9 + 0.2
        horizon_actual = np.array([0.1, -0.2])
        horizon_predicted = np.array([0.05, -0.1])
        oracle_actual = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
        oracle_predicted = np.array([[0.3, 0.3, 0.4], [0.5, 0.3, 0.2]])
        batch = MetricBatch(
            scalars=scalars,
            candle_actual=candle_actual,
            candle_predicted=candle_predicted,
            close_return_actual=close_actual,
            close_return_predicted=close_predicted,
            close_path_actual=close_actual,
            close_path_predicted=close_predicted,
            horizon_return_actual=horizon_actual,
            horizon_return_predicted=horizon_predicted,
            oracle_actual=oracle_actual,
            oracle_predicted=oracle_predicted,
        )
        original = ForecastAccumulator(action_count)
        original.add(batch, range(batch_size))

        restored = ForecastAccumulator.from_state(
            original.state(), action_count
        )

        self.assertEqual(original.state(), restored.state())

    def test_probabilistic_accumulator_round_trip(self) -> None:
        levels = np.array([0.1, 0.5, 0.9])
        scalars = {
            name: np.array([0.25, 0.75])
            for name in (
                "samplePathCrps", "samplePathValidOhlcFraction",
                "originsWithAnyInvalidSamplePath",
                "rawQuantileValidOhlcFraction",
                "repairedQuantileValidOhlcFraction", "rawPinball",
                "repairedPinball", "central80Coverage", "central80Width",
            )
        }
        coverage = np.zeros((2, HORIZON, levels.size, 4), dtype=bool)
        coverage[1] = True
        batch = ProbabilisticBatch(
            scalars=scalars,
            raw_coverage=coverage,
            repaired_coverage=~coverage,
        )
        original = ProbabilisticAccumulator(levels)
        original.add(batch, range(2))

        restored = ProbabilisticAccumulator.from_state(
            original.state(), levels
        )

        self.assertEqual(original.state(), restored.state())

    def test_batch_seed_depends_on_seed_and_origin_times(self) -> None:
        origins = (
            Origin(target_start=60_000, target_index=1, window_ids=("a",)),
            Origin(target_start=120_000, target_index=2, window_ids=("a",)),
        )

        first = seed_inference_batch(1_337, origins)
        second = seed_inference_batch(1_337, origins)
        changed = seed_inference_batch(1_337, origins[:1])

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_expected_utility_aggregation_differs_from_oracle_vote(self) -> None:
        oracle = DifferentiableExposureValueOracle(
            DifferentiableExposureValueOracleConfig(
                holding_period_steps=HORIZON,
                decision_delay_steps=HORIZON,
                value_horizon_steps=HORIZON,
                friction=0,
                grid_size=3,
                temperature=0.01,
                min_exposure=-1,
                max_exposure=1,
            )
        )
        paths = np.ones((1, 3, HORIZON, 5), dtype=np.float64)
        paths[..., 0:4] = 100
        paths[0, 0, :, 0:4] = np.linspace(100, 103, HORIZON)[:, None]
        paths[0, 1, :, 0:4] = np.linspace(100, 103, HORIZON)[:, None]
        paths[0, 2, :, 0:4] = np.linspace(100, 94, HORIZON)[:, None]

        vote, expected_utility = oracle_path_distributions(
            paths,
            np.array([100.0]),
            oracle,
            torch.device("cpu"),
        )

        self.assertEqual(vote.shape, (1, 3))
        self.assertEqual(expected_utility.shape, (1, 3))
        self.assertAlmostEqual(float(vote.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(expected_utility.sum()), 1.0, places=6)
        self.assertFalse(np.allclose(vote, expected_utility))

    def test_execution_oracle_uses_close_paths_not_invalid_ohlc_geometry(self) -> None:
        oracle = DifferentiableExposureValueOracle(
            DifferentiableExposureValueOracleConfig(
                holding_period_steps=HORIZON,
                decision_delay_steps=HORIZON,
                value_horizon_steps=HORIZON,
                friction=0.00175,
                grid_size=5,
                temperature=0.01,
                min_exposure=-2,
                max_exposure=2,
            )
        )
        paths = np.ones((1, 2, HORIZON, 5), dtype=np.float64) * 100
        paths[0, 0, :, 3] = np.linspace(100, 104, HORIZON)
        paths[0, 1, :, 3] = np.linspace(100, 96, HORIZON)
        invalid = paths.copy()
        invalid[..., 0] = 130
        invalid[..., 1] = 70
        invalid[..., 2] = 120

        valid_vote, valid_utility = oracle_path_distributions(
            paths, np.array([100.0]), oracle, torch.device("cpu")
        )
        invalid_vote, invalid_utility = oracle_path_distributions(
            invalid, np.array([100.0]), oracle, torch.device("cpu")
        )

        np.testing.assert_allclose(valid_vote, invalid_vote, rtol=0, atol=0)
        np.testing.assert_allclose(valid_utility, invalid_utility, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
