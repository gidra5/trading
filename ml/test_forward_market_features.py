from __future__ import annotations

import unittest

import numpy as np

from forward_market_features import (
    DAY_ROWS,
    _shift_minute_days_to_close,
    _shift_second_days_to_minute_close,
)
from oracle_futures_basis_features import FUTURES_KLINE_COLUMNS
from oracle_trade_flow_features import SECOND_ROWS, TRADE_FLOW_COLUMNS


class ForwardMarketFeatureAlignmentTest(unittest.TestCase):
    def test_second_shift_aligns_rows_to_completed_minute_closes(self) -> None:
        previous = {
            name: np.arange(SECOND_ROWS, dtype=np.float64)
            for name in TRADE_FLOW_COLUMNS
        }
        current = {
            name: SECOND_ROWS + np.arange(SECOND_ROWS, dtype=np.float64)
            for name in TRADE_FLOW_COLUMNS
        }
        shifted_previous, shifted_current = _shift_second_days_to_minute_close(
            previous, current,
        )
        values = shifted_current[TRADE_FLOW_COLUMNS[0]]
        self.assertEqual(shifted_previous[TRADE_FLOW_COLUMNS[0]][-1], SECOND_ROWS + 58)
        self.assertEqual(values[0], SECOND_ROWS + 59)
        self.assertEqual(values[1_439 * 60], 2 * SECOND_ROWS - 1)
        self.assertTrue(np.all(values[1_439 * 60 + 1:] == 0))

    def test_minute_shift_exposes_current_completed_futures_candle(self) -> None:
        previous = {
            name: np.arange(DAY_ROWS, dtype=np.float64)
            for name in FUTURES_KLINE_COLUMNS
        }
        current = {
            name: DAY_ROWS + np.arange(DAY_ROWS, dtype=np.float64)
            for name in FUTURES_KLINE_COLUMNS
        }
        valid = np.ones(DAY_ROWS, dtype=bool)
        (shifted_previous, shifted_current), validity = _shift_minute_days_to_close(
            previous, current, valid, valid,
        )
        values = shifted_previous[FUTURES_KLINE_COLUMNS[0]]
        self.assertEqual(values[-1], DAY_ROWS)
        self.assertEqual(shifted_current[FUTURES_KLINE_COLUMNS[0]][-2], 2 * DAY_ROWS - 1)
        self.assertEqual(shifted_current[FUTURES_KLINE_COLUMNS[0]][-1], 0)
        self.assertTrue(validity[0].all())
        self.assertFalse(validity[1][-1])


if __name__ == "__main__":
    unittest.main()
