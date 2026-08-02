import unittest

import numpy as np

from oracle_futures_book_depth_features import (
    BOOK_DEPTH_BAND_LABELS,
    BOOK_DEPTH_BANDS,
    BOOK_DEPTH_VALUE_COLUMNS,
    DAY_ROWS,
    FUTURES_BOOK_DEPTH_FEATURE_NAMES,
    SIGNED_FUTURES_BOOK_DEPTH_FEATURE_NAMES,
    causal_futures_book_depth_features,
)


class FuturesBookDepthFeatureTests(unittest.TestCase):
    @staticmethod
    def empty_day() -> dict[str, np.ndarray]:
        return {
            "timestampOffsetSeconds": np.empty(0, dtype=np.int64),
            "bidDepth": np.empty((0, 6), dtype=np.float64),
            "askDepth": np.empty((0, 6), dtype=np.float64),
            "bidNotional": np.empty((0, 6), dtype=np.float64),
            "askNotional": np.empty((0, 6), dtype=np.float64),
            "bandAvailable": np.empty((0, 6), dtype=bool),
            "snapshotUsable": np.empty(0, dtype=bool),
            "snapshotCorrupt": np.empty(0, dtype=bool),
        }

    @classmethod
    def day(
        cls,
        offsets: tuple[int, ...],
        *,
        bid_scale: float | tuple[float, ...] = 1.0,
        ask_scale: float | tuple[float, ...] = 1.0,
        schema12: bool | tuple[bool, ...] = True,
        usable: bool | tuple[bool, ...] = True,
        corrupt: bool | tuple[bool, ...] = False,
    ) -> dict[str, np.ndarray]:
        count = len(offsets)

        def vector(value: object, dtype: object) -> np.ndarray:
            if isinstance(value, tuple):
                return np.asarray(value, dtype=dtype)
            return np.full(count, value, dtype=dtype)

        bid_scales = vector(bid_scale, np.float64)
        ask_scales = vector(ask_scale, np.float64)
        schemas = vector(schema12, bool)
        base_curve = np.asarray((2, 10, 19, 30, 43, 58), dtype=np.float64)
        bid = bid_scales[:, None] * base_curve[None, :]
        ask = ask_scales[:, None] * base_curve[None, :]
        available = np.ones((count, len(BOOK_DEPTH_BANDS)), dtype=bool)
        available[~schemas, 0] = False
        bid[~schemas, 0] = 0
        ask[~schemas, 0] = 0
        return {
            "timestampOffsetSeconds": np.asarray(offsets, dtype=np.int64),
            "bidDepth": bid,
            "askDepth": ask,
            "bidNotional": bid * 100,
            "askNotional": ask * 100,
            "bandAvailable": available,
            "snapshotUsable": vector(usable, bool),
            "snapshotCorrupt": vector(corrupt, bool),
        }

    @staticmethod
    def column(name: str) -> int:
        return FUTURES_BOOK_DEPTH_FEATURE_NAMES.index(name)

    def test_same_second_is_eligible_but_next_second_is_not(self) -> None:
        current = self.day(
            (0, 1, 60),
            bid_scale=(2.0, 100.0, 4.0),
            ask_scale=(1.0, 1.0, 1.0),
        )
        features = causal_futures_book_depth_features(
            self.empty_day(), current,
        )
        imbalance = self.column("baseBidAskImbalance1pct")
        self.assertAlmostEqual(features[0, imbalance], 1 / 3, places=6)
        self.assertAlmostEqual(features[1, imbalance], 3 / 5, places=6)
        observed = self.column("bookDepthCurrentSecondObserved")
        usable = self.column("bookDepthCurrentSecondUsable")
        self.assertEqual(features[0, observed], 1)
        self.assertEqual(features[0, usable], 1)
        self.assertEqual(features[1, observed], 1)

    def test_future_snapshot_perturbation_cannot_move_earlier_rows(self) -> None:
        previous = self.day((86_399,))
        baseline_day = self.day((0, 61, 120), bid_scale=(1.0, 2.0, 3.0))
        changed_day = self.day((0, 61, 120), bid_scale=(1.0, 200.0, 300.0))
        baseline = causal_futures_book_depth_features(previous, baseline_day)
        changed = causal_futures_book_depth_features(previous, changed_day)
        np.testing.assert_array_equal(baseline[:2], changed[:2])
        self.assertFalse(np.array_equal(baseline[2], changed[2]))

    def test_corrupt_snapshot_is_skipped_while_raw_and_usable_ages_diverge(self) -> None:
        previous = self.day((86_300,), bid_scale=2.0, ask_scale=1.0)
        current = self.day(
            (0, 60),
            bid_scale=(50.0, 4.0),
            ask_scale=(1.0, 1.0),
            usable=(False, True),
            corrupt=(True, False),
        )
        # Corrupt content is deliberately non-cumulative and non-positive; it
        # must not be validated as a usable value or leak into a feature.
        current["bidDepth"][0] = (9, -2, 3, 2, 1, 0)
        current["bidNotional"][0] = np.nan
        features = causal_futures_book_depth_features(previous, current)
        imbalance = self.column("baseBidAskImbalance1pct")
        self.assertAlmostEqual(features[0, imbalance], 1 / 3, places=6)
        self.assertAlmostEqual(features[1, imbalance], 3 / 5, places=6)
        raw_age = self.column("bookDepthRawSnapshotAge24h")
        usable_age = self.column("bookDepthUsableValueAge24h")
        corrupt = self.column("bookDepthCurrentSecondCorrupt")
        usable = self.column("bookDepthCurrentSecondUsable")
        self.assertEqual(features[0, raw_age], 0)
        self.assertAlmostEqual(features[0, usable_age], 100 / 86_400)
        self.assertEqual(features[0, corrupt], 1)
        self.assertEqual(features[0, usable], 0)
        self.assertEqual(features[1, usable_age], 0)

    def test_previous_day_value_carries_without_backward_fill(self) -> None:
        previous = self.day(
            (86_399,), bid_scale=3.0, ask_scale=1.0, schema12=False,
        )
        features = causal_futures_book_depth_features(
            previous, self.empty_day(),
        )
        imbalance = self.column("baseBidAskImbalance1pct")
        raw_age = self.column("bookDepthRawSnapshotAge24h")
        usable_age = self.column("bookDepthUsableValueAge24h")
        self.assertAlmostEqual(features[0, imbalance], 0.5, places=6)
        self.assertAlmostEqual(features[0, raw_age], 1 / 86_400)
        self.assertAlmostEqual(features[0, usable_age], 1 / 86_400)

        only_future = causal_futures_book_depth_features(
            self.empty_day(), self.day((60,)),
        )
        np.testing.assert_array_equal(
            only_future[0, :],
            causal_futures_book_depth_features(
                self.empty_day(), self.empty_day(),
            )[0, :],
        )

    def test_10_and_12_band_schema_flags_and_optional_band_age(self) -> None:
        current = self.day((0, 60), schema12=(False, True))
        features = causal_futures_book_depth_features(
            self.empty_day(), current,
        )
        schema10 = self.column("bookDepthCurrentSecondSchema10")
        schema12 = self.column("bookDepthCurrentSecondSchema12")
        available = self.column("bookDepthBand0p2ValueAvailable")
        age = self.column("bookDepthBand0p2ObservationAge24h")
        band0 = self.column("baseBidAskImbalance0p2pct")
        self.assertEqual(features[0, schema10], 1)
        self.assertEqual(features[0, schema12], 0)
        self.assertEqual(features[0, available], 0)
        self.assertEqual(features[0, age], 1)
        self.assertEqual(features[0, band0], 0)
        self.assertEqual(features[1, schema10], 0)
        self.assertEqual(features[1, schema12], 1)
        self.assertEqual(features[1, available], 1)
        self.assertEqual(features[1, age], 0)

    def test_usable_cumulative_values_are_strictly_validated(self) -> None:
        non_cumulative = self.day((0,))
        non_cumulative["askDepth"][0, 3] = 1
        with self.assertRaisesRegex(ValueError, "non-cumulative askDepth"):
            causal_futures_book_depth_features(
                self.empty_day(), non_cumulative,
            )

        non_positive = self.day((0,))
        non_positive["bidNotional"][0, 2] = 0
        with self.assertRaisesRegex(ValueError, "invalid bidNotional"):
            causal_futures_book_depth_features(
                self.empty_day(), non_positive,
            )

        float_timestamp = self.day((0,))
        float_timestamp["timestampOffsetSeconds"] = np.asarray((0.0,))
        with self.assertRaisesRegex(ValueError, "integer seconds"):
            causal_futures_book_depth_features(
                self.empty_day(), float_timestamp,
            )

    def test_names_shape_finiteness_abs_pairs_and_all_families(self) -> None:
        features = causal_futures_book_depth_features(
            self.day((86_399,)),
            self.day((0, 60), bid_scale=(1.0, 2.0)),
        )
        self.assertEqual(
            features.shape, (DAY_ROWS, len(FUTURES_BOOK_DEPTH_FEATURE_NAMES)),
        )
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(
            len(FUTURES_BOOK_DEPTH_FEATURE_NAMES),
            len(set(FUTURES_BOOK_DEPTH_FEATURE_NAMES)),
        )
        self.assertTrue(np.isfinite(features).all())
        for name in SIGNED_FUTURES_BOOK_DEPTH_FEATURE_NAMES:
            absolute = f"abs{name[0].upper()}{name[1:]}"
            self.assertIn(absolute, FUTURES_BOOK_DEPTH_FEATURE_NAMES)
            np.testing.assert_array_equal(
                features[:, self.column(absolute)],
                np.abs(features[:, self.column(name)]),
            )
        for quantity in ("base", "notional"):
            for band in BOOK_DEPTH_BAND_LABELS:
                self.assertIn(
                    f"{quantity}BidAskImbalance{band}",
                    FUTURES_BOOK_DEPTH_FEATURE_NAMES,
                )
                self.assertIn(
                    f"{quantity}Total{band}LogChange1h",
                    FUTURES_BOOK_DEPTH_FEATURE_NAMES,
                )
        for name in BOOK_DEPTH_VALUE_COLUMNS:
            self.assertIn(name, self.day((0,)))


if __name__ == "__main__":
    unittest.main()
