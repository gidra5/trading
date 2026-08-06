from __future__ import annotations

import unittest

import numpy as np

from serve_oracle_distribution_path import histories_for_window, INTERVAL_MS


class _Range:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.request: tuple[int, int] | None = None

    def load(self, start_ms: int, count: int) -> np.ndarray:
        self.request = (start_ms, count)
        return self.values[:count]


class OracleDistributionPathServerTests(unittest.TestCase):
    def test_histories_end_at_each_current_candle(self) -> None:
        closes = np.exp(np.arange(125, dtype=np.float64) * 0.01)
        source = _Range(closes)
        start = 1_000 * INTERVAL_MS
        histories = histories_for_window(
            source, start, start + 5 * INTERVAL_MS
        )
        self.assertEqual(histories.shape, (5, 120))
        np.testing.assert_allclose(histories, 0.01, rtol=0, atol=2e-7)
        self.assertEqual(source.request, (start - 120 * INTERVAL_MS, 125))


if __name__ == "__main__":
    unittest.main()
