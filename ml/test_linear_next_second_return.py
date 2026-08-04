from __future__ import annotations

import numpy as np
import torch
import unittest

from linear_next_second_return import (
    StandardizedRegressionStatistics,
    predict_raw,
    raw_coefficients,
    solve_ridge,
    sufficient_normalized_mse,
)
from next_return_dataset import HISTORY_RETURN_COUNT
from train_normalized_glu_next_return import Normalization


class LinearNextSecondReturnTest(unittest.TestCase):
    def test_ols_recovers_a_known_standardized_linear_map(self) -> None:
        generator = np.random.default_rng(81)
        count = 2_000
        normalization = Normalization(
            feature_mean=generator.normal(
                0, 0.1, HISTORY_RETURN_COUNT
            ).astype(np.float32),
            feature_std=generator.uniform(
                0.5, 2, HISTORY_RETURN_COUNT
            ).astype(np.float32),
            target_mean=0.03,
            target_std=0.4,
        )
        standardized_features = generator.normal(
            0, 1, (count, HISTORY_RETURN_COUNT)
        ).astype(np.float32)
        expected = generator.normal(0, 0.02, HISTORY_RETURN_COUNT + 1)
        normalized_target = expected[0] + standardized_features @ expected[1:]
        features = (
            normalization.feature_mean
            + standardized_features * normalization.feature_std
        ).astype(np.float32)
        targets = (
            normalization.target_mean
            + normalization.target_std * normalized_target
        ).astype(np.float32)
        statistics = StandardizedRegressionStatistics(torch.device("cpu"))
        statistics.add(
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.ones(count),
            normalization,
        )
        actual = solve_ridge(statistics, 0.0)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertLess(sufficient_normalized_mse(statistics, actual), 1e-7)

    def test_raw_coefficients_preserve_standardized_predictions(self) -> None:
        generator = np.random.default_rng(21)
        normalization = Normalization(
            feature_mean=generator.normal(
                size=HISTORY_RETURN_COUNT
            ).astype(np.float32),
            feature_std=generator.uniform(
                0.2, 3, HISTORY_RETURN_COUNT
            ).astype(np.float32),
            target_mean=-0.07,
            target_std=0.6,
        )
        coefficients = generator.normal(size=HISTORY_RETURN_COUNT + 1)
        features = generator.normal(
            size=(13, HISTORY_RETURN_COUNT)
        ).astype(np.float32)
        intercept, slopes = raw_coefficients(coefficients, normalization)
        actual = predict_raw(
            torch.from_numpy(features),
            intercept,
            torch.from_numpy(slopes.astype(np.float32)),
        ).numpy()
        standardized = (
            features - normalization.feature_mean
        ) / normalization.feature_std
        expected = normalization.target_mean + normalization.target_std * (
            coefficients[0] + standardized @ coefficients[1:]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_ridge_penalizes_slopes_but_not_the_intercept(self) -> None:
        normalization = Normalization(
            feature_mean=np.zeros(HISTORY_RETURN_COUNT, dtype=np.float32),
            feature_std=np.ones(HISTORY_RETURN_COUNT, dtype=np.float32),
            target_mean=0.0,
            target_std=1.0,
        )
        features = torch.zeros((100, HISTORY_RETURN_COUNT))
        features[:, 0] = torch.linspace(-1, 1, 100)
        targets = 2.0 + 3.0 * features[:, 0]
        statistics = StandardizedRegressionStatistics(torch.device("cpu"))
        statistics.add(features, targets, torch.ones(100), normalization)
        unregularized = solve_ridge(statistics, 0.0)
        regularized = solve_ridge(statistics, 10.0)
        self.assertLess(abs(regularized[1]), abs(unregularized[1]))
        self.assertLess(abs(regularized[0] - 2.0), 1e-6)


if __name__ == "__main__":
    unittest.main()
