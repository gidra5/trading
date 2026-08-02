import unittest

import numpy as np

from oracle_futures_metrics_features import (
    FUTURES_FEATURE_NAMES,
    METRIC_COLUMNS,
    METRIC_ROWS,
    causal_futures_metrics_features,
)


class FuturesMetricsFeatureTests(unittest.TestCase):
    @staticmethod
    def day() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        values = {
            name: np.full(METRIC_ROWS, 2 + index, dtype=np.float64)
            for index, name in enumerate(METRIC_COLUMNS)
        }
        validity = {
            name: np.ones(METRIC_ROWS, dtype=bool)
            for name in METRIC_COLUMNS
        }
        return values, validity

    def test_one_bin_lag_never_uses_same_timestamp_snapshot(self) -> None:
        previous, previous_validity = self.day()
        current, current_validity = self.day()
        previous["sumOpenInterest"][-2:] = (100, 110)
        current["sumOpenInterest"][0:2] = (220, 8_800)
        features = causal_futures_metrics_features(
            previous, previous_validity, current, current_validity,
        )
        index = FUTURES_FEATURE_NAMES.index("openInterestLogChange5m")
        self.assertAlmostEqual(features[0, index], np.log(110 / 100), places=6)
        self.assertAlmostEqual(features[4, index], np.log(110 / 100), places=6)
        self.assertAlmostEqual(features[5, index], np.log(220 / 110), places=6)
        self.assertAlmostEqual(features[9, index], np.log(220 / 110), places=6)
        self.assertAlmostEqual(features[10, index], np.log(8_800 / 220), places=6)

    def test_missing_snapshot_is_carried_forward_and_exposed_by_mask(self) -> None:
        previous, previous_validity = self.day()
        current, current_validity = self.day()
        previous["sumOpenInterest"][-1] = 100
        current["sumOpenInterest"][0] = 0
        current_validity["sumOpenInterest"][0] = False
        features = causal_futures_metrics_features(
            previous, previous_validity, current, current_validity,
        )
        change = FUTURES_FEATURE_NAMES.index("openInterestLogChange5m")
        observed = FUTURES_FEATURE_NAMES.index("sumOpenInterestCurrentObserved")
        age = FUTURES_FEATURE_NAMES.index("sumOpenInterestObservationAge24h")
        self.assertEqual(features[5, change], 0)
        self.assertEqual(features[5, observed], 0)
        self.assertAlmostEqual(features[5, age], 1 / 288)

    def test_all_missing_optional_stream_stays_finite_and_marked_missing(self) -> None:
        previous, previous_validity = self.day()
        current, current_validity = self.day()
        name = "topTraderAccountLongShortRatio"
        previous[name][:] = 0
        current[name][:] = 0
        previous_validity[name][:] = False
        current_validity[name][:] = False
        features = causal_futures_metrics_features(
            previous, previous_validity, current, current_validity,
        )
        self.assertEqual(features.shape, (1_440, len(FUTURES_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        level = FUTURES_FEATURE_NAMES.index(f"{name}LogLevel")
        observed = FUTURES_FEATURE_NAMES.index(f"{name}CurrentObserved")
        age = FUTURES_FEATURE_NAMES.index(f"{name}ObservationAge24h")
        np.testing.assert_array_equal(features[:, level], 0)
        np.testing.assert_array_equal(features[:, observed], 0)
        np.testing.assert_array_equal(features[:, age], 1)

    def test_initial_missing_oi_pair_keeps_derived_price_nullable(self) -> None:
        previous, previous_validity = self.day()
        current, current_validity = self.day()
        for name in ("sumOpenInterest", "sumOpenInterestValue"):
            previous[name][0] = 0
            previous_validity[name][0] = False
        features = causal_futures_metrics_features(
            previous, previous_validity, current, current_validity,
        )
        self.assertTrue(np.isfinite(features).all())
        implied = FUTURES_FEATURE_NAMES.index(
            "impliedMarkPriceLogChange5m"
        )
        self.assertTrue(np.isfinite(features[:, implied]).all())

    def test_future_metric_perturbation_cannot_move_earlier_minutes(self) -> None:
        previous, previous_validity = self.day()
        current, current_validity = self.day()
        baseline = causal_futures_metrics_features(
            previous, previous_validity, current, current_validity,
        )
        changed = {name: values.copy() for name, values in current.items()}
        for name in METRIC_COLUMNS:
            changed[name][100:] *= 10
        perturbed = causal_futures_metrics_features(
            previous, previous_validity, changed, current_validity,
        )
        np.testing.assert_array_equal(baseline[:505], perturbed[:505])
        self.assertFalse(np.array_equal(baseline[505], perturbed[505]))


if __name__ == "__main__":
    unittest.main()
