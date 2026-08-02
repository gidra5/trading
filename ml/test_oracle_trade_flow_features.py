import unittest

import numpy as np

from oracle_trade_flow_features import (
    DAY_ROWS,
    FLOW_FEATURE_NAMES,
    SECOND_ROWS,
    TRADE_FLOW_COLUMNS,
    causal_trade_flow_features,
    rolling_metrics,
)


def empty_day() -> dict[str, np.ndarray]:
    counts = {
        "aggressiveBuyAggregateTradeCount",
        "aggressiveSellAggregateTradeCount",
        "aggressiveBuyTradeCount",
        "aggressiveSellTradeCount",
        "aggressorSideFlipCount",
        "firstAggressorSide",
        "lastAggressorSide",
    }
    return {
        name: np.zeros(
            SECOND_ROWS,
            dtype=np.int64 if name in counts else np.float64,
        )
        for name in TRADE_FLOW_COLUMNS
    }


def add_trade(
    day: dict[str, np.ndarray],
    index: int,
    *,
    buy: bool,
    quantity: float,
    price: float,
    offset_seconds: float,
) -> None:
    prefix = "aggressiveBuy" if buy else "aggressiveSell"
    side = 1 if buy else -1
    day[f"{prefix}BaseVolume"][index] += quantity
    day[f"{prefix}QuoteVolume"][index] += quantity * price
    day[f"{prefix}AggregateQuantitySquared"][index] += quantity * quantity
    day[f"{prefix}MaxAggregateQuantity"][index] = max(
        day[f"{prefix}MaxAggregateQuantity"][index], quantity,
    )
    day[f"{prefix}BaseVolumeTimeMoment"][index] += quantity * offset_seconds
    day[f"{prefix}AggregateTradeCount"][index] += 1
    day[f"{prefix}TradeCount"][index] += 1
    if day["firstAggressorSide"][index] == 0:
        day["firstAggressorSide"][index] = side
    elif day["lastAggressorSide"][index] != side:
        day["aggressorSideFlipCount"][index] += 1
    day["lastAggressorSide"][index] = side


class OracleTradeFlowFeatureTests(unittest.TestCase):
    def test_rolling_metrics_count_cross_second_side_flips(self) -> None:
        previous = empty_day()
        current = empty_day()
        add_trade(current, 0, buy=True, quantity=2, price=100, offset_seconds=.1)
        add_trade(current, 58, buy=True, quantity=2, price=100, offset_seconds=.1)
        add_trade(current, 59, buy=False, quantity=1, price=101, offset_seconds=.2)
        add_trade(current, 60, buy=True, quantity=3, price=102, offset_seconds=.8)
        values = rolling_metrics(previous, current, 5)
        self.assertAlmostEqual(values["quoteImbalance"][0], 1)
        self.assertEqual(values["lastAggressorSide"][0], 1)
        self.assertEqual(values["aggressorFlipRate"][0], 0)
        self.assertAlmostEqual(
            values["aggressorFlipRate"][1], 1,
        )
        self.assertGreater(values["signedArrivalCentroidGap"][1], 0)

    def test_features_exclude_unfinished_current_minute_except_closed_second(self) -> None:
        previous = empty_day()
        current = empty_day()
        add_trade(
            previous, SECOND_ROWS - 1,
            buy=False, quantity=1, price=100, offset_seconds=.5,
        )
        add_trade(
            current, 0,
            buy=True, quantity=3, price=101, offset_seconds=.25,
        )
        add_trade(
            current, 1,
            buy=False, quantity=100, price=99, offset_seconds=.25,
        )
        features = causal_trade_flow_features(previous, current)
        index = {name: position for position, name in enumerate(FLOW_FEATURE_NAMES)}
        self.assertEqual(features.shape, (DAY_ROWS, len(FLOW_FEATURE_NAMES)))
        self.assertEqual(features[0, index["quoteImbalance1s"]], 1)
        self.assertAlmostEqual(
            float(features[0, index["quoteImbalance5s"]]),
            (3 * 101 - 100) / (3 * 101 + 100),
            places=7,
        )
        self.assertEqual(features[1, index["quoteImbalance1s"]], 0)
        self.assertLess(features[1, index["quoteImbalance60s"]], 0)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
