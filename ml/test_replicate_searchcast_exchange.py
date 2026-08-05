from __future__ import annotations

import unittest

import numpy as np

from replicate_searchcast_exchange import persistence_metrics, standardize_like_official


class ReplicateSearchcastExchangeTest(unittest.TestCase):
    def test_standardization_uses_only_first_seventy_percent(self) -> None:
        values = np.column_stack(
            [np.arange(10, dtype=np.float64), 2.0 * np.arange(10, dtype=np.float64)]
        )
        standardized = standardize_like_official(values)
        np.testing.assert_allclose(standardized[:7].mean(axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(standardized[:7].std(axis=0), 1.0, atol=1e-7)

    def test_persistence_repeats_last_context_value(self) -> None:
        values = np.arange(20, dtype=np.float32)[:, None]
        metrics = persistence_metrics(values, horizon=2, context=3)
        # 20 rows -> 14/2/4 split. Test starts at 16 and has 3 valid two-step windows.
        # Every target is [last+1, last+2], so each window contributes errors 1 and 2.
        self.assertEqual(metrics["windows_per_series"], 3)
        self.assertAlmostEqual(float(metrics["mse"]), 2.5)
        self.assertAlmostEqual(float(metrics["mae"]), 1.5)


if __name__ == "__main__":
    unittest.main()
