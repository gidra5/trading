from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from audit_aggregate_return_predictability import regression_features
from fit_future_price_resolution_ridge import (
    RIDGE_ARTIFACT_SCHEMA,
    _canonical_bytes,
    _file_fingerprint,
    split_internal_training,
)
from future_price_resolution_ridge import (
    FeatureScaler,
    RIDGE_FEATURE_NAMES,
    WeightedRegressionStatistics,
    audited_ridge_features,
    predict_standardized_ridge,
    raw_feature_coefficients,
    raw_history_coefficients,
    solve_standardized_ridge,
    standardize_statistics,
)
from future_price_resolution_screen import aggregate_minute_log_returns
from train_future_price_predictor import PairShard, SECOND_MS
from trading_storage import read_shard_payload


class FuturePriceResolutionRidgeTests(unittest.TestCase):
    def test_features_exactly_match_the_aggregate_audit(self) -> None:
        minute = np.arange(2 * 360, dtype=np.float64).reshape(2, 360) / 1e8
        five_minute = aggregate_minute_log_returns(minute, 5)
        expected = regression_features(minute, 5)[:, 1:]
        actual = audited_ridge_features(five_minute)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-9)

    def test_internal_lambda_split_has_full_example_span_embargo(self) -> None:
        shard = PairShard(
            split="train",
            prediction_time_start=1_000_000_000,
            count=100_000,
            future_date="2026-01-02",
            history_date="2026-01-01",
            future_row_offset=0,
            history_row_offset=0,
        )
        span_ms = 365 * 60_000
        split = split_internal_training(
            [shard],
            fit_fraction=0.6,
            example_span_ms=span_ms,
        )
        self.assertEqual(sum(value.count for value in split["fit"]), 60_000)
        self.assertEqual(
            split["calibration"][0].prediction_time_start
            - split["fit"][-1].prediction_time_end,
            span_ms + SECOND_MS,
        )
        self.assertEqual(
            split["calibration"][0].history_row_offset,
            60_000 + span_ms // SECOND_MS,
        )

    def test_closed_form_matches_direct_weighted_ridge(self) -> None:
        generator = np.random.default_rng(7)
        features = generator.normal(size=(1_000, len(RIDGE_FEATURE_NAMES)))
        target = generator.normal(size=1_000)
        weights = generator.integers(1, 60, size=1_000).astype(np.float64)
        statistics = WeightedRegressionStatistics()
        statistics.add(features, target, weights)
        scaler = FeatureScaler.from_statistics(statistics)
        standardized = standardize_statistics(statistics, scaler)
        ridge_lambda = 1e-3
        actual = solve_standardized_ridge(standardized, ridge_lambda)
        design = np.column_stack((
            np.ones(features.shape[0]),
            (features - scaler.mean) / scaler.std,
        ))
        matrix = design.T @ (weights[:, None] * design)
        penalty = np.eye(matrix.shape[0]) * weights.sum() * ridge_lambda
        penalty[0, 0] = 0
        expected = np.linalg.solve(
            matrix + penalty,
            design.T @ (weights * target),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_raw_history_coefficients_reproduce_audited_features(self) -> None:
        generator = np.random.default_rng(11)
        history = generator.normal(scale=0.001, size=(32, 72))
        mean = generator.normal(scale=0.001, size=len(RIDGE_FEATURE_NAMES))
        std = generator.uniform(0.001, 0.01, size=len(RIDGE_FEATURE_NAMES))
        coefficients = generator.normal(size=len(RIDGE_FEATURE_NAMES) + 1)
        scaler = FeatureScaler(mean, std)
        expected = predict_standardized_ridge(history, scaler, coefficients)
        intercept, slopes = raw_feature_coefficients(scaler, coefficients)
        actual = intercept + history @ raw_history_coefficients(slopes)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_generated_v3_artifact_hash_and_provenance_are_stable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        reference_file = (
            repo_root
            / "data/training/immutable/refs/models/future-price-resolution"
            / "future-price-resolution-6h-5m-next-5m-ridge-v3/model.json"
        )
        if not reference_file.is_file():
            self.skipTest("locally fitted immutable ridge artifact is unavailable")
        reference, payload = read_shard_payload(reference_file)
        artifact = json.loads(payload)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "d7f56a48138f3878b13b8adb5195cdd8d87e3346cb4cbb8df725e12105a03f26",
        )
        self.assertEqual(
            reference.reference["object"]["contentHash"],
            hashlib.sha256(payload).hexdigest(),
        )
        self.assertEqual(artifact["schema"], RIDGE_ARTIFACT_SCHEMA)
        self.assertEqual(
            artifact["dataset"]["datasetFingerprint"],
            "0756d93ce95262718b0064506d5c4faf36f67d73936e021e91a1f4c35d6df3f0",
        )
        self.assertEqual(artifact["accessContract"], {
            "testExamplesSelected": 0,
            "testPayloadsRead": 0,
            "gpuUsed": False,
        })
        plan_file = (
            repo_root
            / "ml/training-plans"
            / "future-price-resolution-6h-5m-next-5m-ridge-v3.json"
        )
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        self.assertEqual(
            artifact["planFingerprint"],
            hashlib.sha256(_canonical_bytes(plan)).hexdigest(),
        )
        self.assertEqual(
            artifact["implementationFingerprint"],
            _file_fingerprint(tuple(
                repo_root / "ml" / name
                for name in (
                    "fit_future_price_resolution_ridge.py",
                    "future_price_resolution_ridge.py",
                    "future_price_resolution_screen.py",
                    "train_future_price_resolution_screen.py",
                    "train_future_price_feature_screen.py",
                    "train_future_price_predictor.py",
                    "trading_storage.py",
                )
            )),
        )
        self.assertEqual(
            artifact["fit"]["selectedModel"],
            "training-mean-ridge-limit",
        )
        self.assertEqual(artifact["fit"]["selectedLambda"], "infinity")
        self.assertTrue(
            artifact["models"]["ridge"]["eligibleForDownstreamSelection"]
        )
        self.assertFalse(
            artifact["models"]["ols"]["eligibleForDownstreamSelection"]
        )
        self.assertEqual(
            artifact["referencePatchTcn"]["epoch"],
            11,
        )
        for name in ("ridge", "ols"):
            model = artifact["models"][name]
            self.assertEqual(
                len(model["standardizedFeatureCoefficients"]),
                len(RIDGE_FEATURE_NAMES),
            )
            self.assertEqual(len(model["rawHistoryCoefficients"]), 72)


if __name__ == "__main__":
    unittest.main()
