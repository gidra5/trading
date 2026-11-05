"""Independent least-squares check of streaming active-only ridge moments."""
import unittest

from audit_ridge_horizon_forecasts import HORIZONS, moments
import numpy as np


class ArraySource:
    def __init__(self, features):
        self.values = features

    def features(self, origins):
        return self.values[origins]


class RidgeMomentTest(unittest.TestCase):
    def test_zero_rows_are_excluded_per_horizon_and_intercepts_are_centered(self):
        rng = np.random.default_rng(120)
        features = rng.normal(size=(997, 178)).astype(np.float32)
        features[:, 0] += 4
        features[:, 1] = features[:, 1] * 3 + 2
        features[:, -1] = 7  # Constant input must not produce division by zero.
        targets = np.empty((len(features), len(HORIZONS)))
        for index in range(len(HORIZONS)):
            targets[:, index] = .003 + .01 * features[:, index] + rng.normal(0, .002, len(features))
            targets[index::7 + index, index] = 0  # Each horizon has a different active sample.
        mean, std, candidates, training = moments(ArraySource(features), np.arange(len(features)), targets, 127)
        normalized = (features.astype(np.float64) - mean) / std
        for index, horizon in enumerate(HORIZONS):
            active = targets[:, index] != 0
            x, y = normalized[active], targets[active, index]
            centered_x, centered_y = x - x.mean(axis=0), y - y.mean()
            strength = .1
            # Independently solve augmented least squares, without reusing a
            # Gram matrix or the zero-row subtraction employed by the audit.
            augmented_x = np.vstack([centered_x / np.sqrt(len(y)), np.sqrt(strength) * np.eye(178)])
            augmented_y = np.concatenate([centered_y / np.sqrt(len(y)), np.zeros(178)])
            expected, *_ = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
            candidate = next(c for c in candidates[horizon] if c.ridge_lambda == strength)
            np.testing.assert_allclose(candidate.coefficient, expected, atol=1e-12, rtol=1e-9)
            expected_intercept = y.mean() - x.mean(axis=0) @ expected
            self.assertAlmostEqual(candidate.intercept, expected_intercept, places=12)
            residual = np.square(y - x @ expected - expected_intercept).mean()
            self.assertAlmostEqual(candidate.residual_variance, residual, places=14)
            self.assertEqual(training[horizon]["activeOrigins"], int(active.sum()))
            self.assertEqual(candidate.zero_probability, ((~active).sum() + .5) / (len(active) + 1))


if __name__ == "__main__":
    unittest.main()
