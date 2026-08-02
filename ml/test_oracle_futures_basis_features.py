import unittest

import numpy as np

from oracle_futures_basis_features import (
    DAY_ROWS,
    FUTURES_BASIS_FEATURE_NAMES,
    FUTURES_KLINE_COLUMNS,
    SIGNED_FUTURES_BASIS_FEATURE_NAMES,
    causal_futures_basis_features,
)


class FuturesBasisFeatureTests(unittest.TestCase):
    @staticmethod
    def day(
        futures_price: float = 100.0,
        spot_price: float = 100.0,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        values = {
            "open": np.full(DAY_ROWS, futures_price),
            "high": np.full(DAY_ROWS, futures_price * 1.001),
            "low": np.full(DAY_ROWS, futures_price * 0.999),
            "close": np.full(DAY_ROWS, futures_price),
            "baseVolume": np.full(DAY_ROWS, 10.0),
            "quoteVolume": np.full(DAY_ROWS, futures_price * 10.0),
            "tradeCount": np.full(DAY_ROWS, 100, dtype=np.uint64),
            "takerBuyBaseVolume": np.full(DAY_ROWS, 6.0),
            "takerBuyQuoteVolume": np.full(DAY_ROWS, futures_price * 6.0),
        }
        validity = np.ones(DAY_ROWS, dtype=bool)
        spot = np.column_stack((
            np.full(DAY_ROWS, spot_price),
            np.full(DAY_ROWS, spot_price * 1.001),
            np.full(DAY_ROWS, spot_price * 0.999),
            np.full(DAY_ROWS, spot_price),
            np.full(DAY_ROWS, 10.0),
        ))
        return values, validity, spot

    @staticmethod
    def set_live_price(values: dict[str, np.ndarray], index: int, price: float) -> None:
        values["open"][index] = price
        values["high"][index] = price * 1.001
        values["low"][index] = price * 0.999
        values["close"][index] = price
        values["quoteVolume"][index] = price * values["baseVolume"][index]
        values["takerBuyQuoteVolume"][index] = (
            price * values["takerBuyBaseVolume"][index]
        )

    def features(
        self,
        previous: tuple[dict[str, np.ndarray], np.ndarray, np.ndarray],
        current: tuple[dict[str, np.ndarray], np.ndarray, np.ndarray],
    ) -> np.ndarray:
        return causal_futures_basis_features(
            previous[0], previous[1], current[0], current[1],
            previous[2], current[2],
        )

    def test_one_minute_lag_and_future_perturbation(self) -> None:
        previous = self.day()
        current = self.day()
        self.set_live_price(current[0], 0, 200.0)
        self.set_live_price(current[0], 1, 400.0)
        features = self.features(previous, current)
        level = FUTURES_BASIS_FEATURE_NAMES.index("basisLogLevel")
        self.assertAlmostEqual(features[0, level], 0.0, places=6)
        self.assertAlmostEqual(features[1, level], np.log(2), places=6)
        self.assertAlmostEqual(features[2, level], np.log(4), places=6)

        perturbed = self.day()
        for index in range(100, DAY_ROWS):
            self.set_live_price(perturbed[0], index, 250.0)
        baseline = self.features(previous, self.day())
        changed = self.features(previous, perturbed)
        np.testing.assert_array_equal(baseline[:101], changed[:101])
        self.assertTrue(np.any(baseline[101] != changed[101]))

        still_open = self.day()
        self.set_live_price(still_open[0], DAY_ROWS - 1, 10_000.0)
        np.testing.assert_array_equal(
            baseline, self.features(previous, still_open),
        )

    def test_no_trade_price_and_basis_are_stale_not_observed(self) -> None:
        previous = self.day()
        current = self.day()
        for name in (
            "baseVolume", "quoteVolume", "takerBuyBaseVolume",
            "takerBuyQuoteVolume",
        ):
            current[0][name][0] = 0
        current[0]["tradeCount"][0] = 0
        for name in ("open", "high", "low", "close"):
            current[0][name][0] = 999.0
        current[2][0, :4] = (50.0, 50.05, 49.95, 50.0)
        self.set_live_price(current[0], 1, 110.0)

        features = self.features(previous, current)
        level = FUTURES_BASIS_FEATURE_NAMES.index("basisLogLevel")
        observed = FUTURES_BASIS_FEATURE_NAMES.index("basisCurrentObserved")
        age = FUTURES_BASIS_FEATURE_NAMES.index("basisObservationAge24h")
        row_observed = FUTURES_BASIS_FEATURE_NAMES.index(
            "futuresRowCurrentObserved"
        )
        no_trade = FUTURES_BASIS_FEATURE_NAMES.index("futuresNoTradeCurrent")
        return1m = FUTURES_BASIS_FEATURE_NAMES.index("futuresLogReturn1m")
        self.assertEqual(features[1, level], 0)
        self.assertEqual(features[1, observed], 0)
        self.assertAlmostEqual(features[1, age], 1 / DAY_ROWS)
        self.assertEqual(features[1, row_observed], 1)
        self.assertEqual(features[1, no_trade], 1)
        self.assertEqual(features[1, return1m], 0)
        self.assertAlmostEqual(features[2, level], np.log(1.1), places=6)

    def test_missing_row_carries_price_without_becoming_no_trade(self) -> None:
        previous = self.day()
        current = self.day()
        current[1][0] = False
        for name in FUTURES_KLINE_COLUMNS:
            current[0][name][0] = 0
        features = self.features(previous, current)
        level = FUTURES_BASIS_FEATURE_NAMES.index("basisLogLevel")
        observed = FUTURES_BASIS_FEATURE_NAMES.index(
            "futuresPriceCurrentObserved"
        )
        age = FUTURES_BASIS_FEATURE_NAMES.index(
            "futuresPriceObservationAge24h"
        )
        row_observed = FUTURES_BASIS_FEATURE_NAMES.index(
            "futuresRowCurrentObserved"
        )
        no_trade = FUTURES_BASIS_FEATURE_NAMES.index("futuresNoTradeCurrent")
        quote_surprise = FUTURES_BASIS_FEATURE_NAMES.index(
            "futuresLogQuoteRate1mVs1h"
        )
        trade_surprise = FUTURES_BASIS_FEATURE_NAMES.index(
            "futuresLogTradeRate1mVs1h"
        )
        self.assertEqual(features[1, level], 0)
        self.assertEqual(features[1, observed], 0)
        self.assertAlmostEqual(features[1, age], 1 / DAY_ROWS)
        self.assertEqual(features[1, row_observed], 0)
        self.assertEqual(features[1, no_trade], 0)
        self.assertEqual(features[1, quote_surprise], 0)
        self.assertEqual(features[1, trade_surprise], 0)

    def test_feature_names_shape_finiteness_and_absolute_pairs(self) -> None:
        features = self.features(self.day(), self.day())
        self.assertEqual(
            features.shape, (DAY_ROWS, len(FUTURES_BASIS_FEATURE_NAMES)),
        )
        self.assertEqual(
            len(FUTURES_BASIS_FEATURE_NAMES),
            len(set(FUTURES_BASIS_FEATURE_NAMES)),
        )
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(any(
            "returnSpread" in name for name in FUTURES_BASIS_FEATURE_NAMES
        ))
        for name in SIGNED_FUTURES_BASIS_FEATURE_NAMES:
            absolute = f"abs{name[0].upper()}{name[1:]}"
            self.assertIn(absolute, FUTURES_BASIS_FEATURE_NAMES)
            np.testing.assert_array_equal(
                features[:, FUTURES_BASIS_FEATURE_NAMES.index(absolute)],
                np.abs(features[:, FUTURES_BASIS_FEATURE_NAMES.index(name)]),
            )


if __name__ == "__main__":
    unittest.main()
