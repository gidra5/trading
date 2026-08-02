from __future__ import annotations

import unittest

import numpy as np

from audit_causal_oracle_predictability import mean_kl
from audit_v18_calendar_fusion import (
    convex_probability_fusion,
    log_ratio_calendar_fusion,
    select_scalar,
)


class FusionScalarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.asarray([
            [0.75, 0.20, 0.05],
            [0.10, 0.30, 0.60],
            [0.55, 0.25, 0.20],
            [0.20, 0.50, 0.30],
        ], dtype=np.float64)
        self.calendar = np.asarray([
            [0.30, 0.50, 0.20],
            [0.50, 0.20, 0.30],
            [0.15, 0.15, 0.70],
            [0.60, 0.10, 0.30],
        ], dtype=np.float64)

    def test_probability_search_recovers_constructed_weight(self) -> None:
        expected = 0.37
        targets = convex_probability_fusion(
            self.base,
            self.calendar,
            expected,
        )
        weight, score = select_scalar(
            lambda value: mean_kl(
                targets,
                convex_probability_fusion(
                    self.base,
                    self.calendar,
                    value,
                ),
            ),
            0,
            1,
        )
        self.assertAlmostEqual(weight, expected, places=5)
        self.assertLess(score, 1e-10)

    def test_log_ratio_search_recovers_constructed_weight(self) -> None:
        prior = np.asarray([0.4, 0.35, 0.25], dtype=np.float64)
        expected = 0.61
        targets = log_ratio_calendar_fusion(
            self.base,
            self.calendar,
            prior,
            expected,
        )
        weight, score = select_scalar(
            lambda value: mean_kl(
                targets,
                log_ratio_calendar_fusion(
                    self.base,
                    self.calendar,
                    prior,
                    value,
                ),
            ),
            0,
            2,
        )
        self.assertAlmostEqual(weight, expected, places=5)
        self.assertLess(score, 1e-10)

    def test_zero_weight_preserves_v18_probabilities(self) -> None:
        prior = np.asarray([0.4, 0.35, 0.25], dtype=np.float64)
        np.testing.assert_allclose(
            convex_probability_fusion(self.base, self.calendar, 0),
            self.base,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            log_ratio_calendar_fusion(
                self.base,
                self.calendar,
                prior,
                0,
            ),
            self.base,
            atol=1e-15,
        )

    def test_search_retains_exact_zero_endpoint(self) -> None:
        weight, score = select_scalar(
            lambda value: mean_kl(
                self.base,
                convex_probability_fusion(
                    self.base,
                    self.calendar,
                    value,
                ),
            ),
            0,
            1,
        )
        self.assertEqual(weight, 0)
        self.assertLess(score, 1e-12)


if __name__ == "__main__":
    unittest.main()
