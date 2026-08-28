from __future__ import annotations

import unittest

import numpy as np

from global_feature_registry import prediction_market_asset
from materialize_prediction_market_candidate_axis import (
    Accumulator,
    CarryAccumulator,
    add_contract_carry_intervals,
    accumulator_features,
    causal_origin_index,
    market_buckets,
    semantic_families,
)


class PredictionMarketCandidateAxisTest(unittest.TestCase):
    def test_asset_and_global_semantics_are_deterministic(self) -> None:
        self.assertEqual(prediction_market_asset({"series": "KXBTC15M", "seriesTitle": "Bitcoin price up"}), "BTC")
        self.assertEqual(prediction_market_asset({"series": "KXXRPD", "seriesTitle": "Ripple price"}), "XRP")
        self.assertIn("monetary-policy", semantic_families("KXFEDDECISION", "Fed decision"))
        self.assertIn("inflation", semantic_families("KXCPIYOY", "US inflation"))
        buckets = market_buckets({
            "scope": "global", "category": "Economics", "series": "KXFEDDECISION",
            "seriesTitle": "Fed decision",
        })
        self.assertIn(("general", "global", "family-monetary-policy"), buckets)

    def test_aggregation_uses_only_values_available_on_each_origin(self) -> None:
        accumulator = Accumulator.create(3)
        accumulator.add(
            1, probability=0.75, spread=0.04, change=0.05, volume=10,
            open_interest=100, time_to_close_ms=60_000, age_ms=200,
        )
        features = accumulator_features(accumulator)
        self.assertTrue(np.isnan(features["logit-probability-mean"][0]))
        self.assertAlmostEqual(features["logit-probability-mean"][1], np.log(3.0))
        self.assertEqual(features["log-active-market-count"][0], 0.0)
        self.assertTrue(np.isnan(features["logit-probability-change-1m"][1]))

    def test_source_rows_align_to_the_first_causally_available_origin(self) -> None:
        origins = np.asarray([59_999, 119_999, 179_999], dtype=np.int64)
        self.assertEqual(causal_origin_index(origins, 59_999), 0)
        self.assertEqual(causal_origin_index(origins, 60_000), 1)
        self.assertEqual(causal_origin_index(origins, 119_000), 1)
        self.assertIsNone(causal_origin_index(origins, 180_000))

    def test_batched_accumulation_matches_scalar_accumulation(self) -> None:
        scalar = Accumulator.create(3)
        scalar.add(
            1, probability=0.75, spread=0.04, change=0.05, volume=10,
            open_interest=100, time_to_close_ms=60_000, age_ms=200,
        )
        scalar.add(
            1, probability=0.25, spread=None, change=-0.02, volume=4,
            open_interest=None, time_to_close_ms=120_000, age_ms=None,
        )
        batched = Accumulator.create(3)
        batched.add_batch([
            (1, 0.75, 0.04, 0.05, 10, 100, 60_000, 200),
            (1, 0.25, np.nan, -0.02, 4, np.nan, 120_000, np.nan),
        ])
        for name in scalar.__dataclass_fields__:
            np.testing.assert_allclose(getattr(batched, name), getattr(scalar, name), equal_nan=True)

    def test_latest_contract_state_carries_until_replacement_or_close(self) -> None:
        origins = np.asarray([59_999, 119_999, 179_999, 239_999], dtype=np.int64)
        buckets = [("general", "global", "all-global-events")]
        accumulators: dict[tuple[str, tuple[str, str, str]], CarryAccumulator] = {}
        add_contract_carry_intervals(
            [
                (1, {"lastProbability": 0.25, "intervalContractVolume": 3}),
                (2, {"lastProbability": 0.75, "intervalContractVolume": 2}),
            ],
            market={"closeTime": "1970-01-01T00:04:00Z"},
            cadence="1m",
            origins=origins,
            buckets=buckets,
            accumulators=accumulators,
        )
        features = accumulators[("1m", buckets[0])].features()
        np.testing.assert_array_equal(
            np.rint(np.expm1(features["stateful-log-active-market-count"])),
            [0, 1, 1, 1],
        )
        self.assertAlmostEqual(features["stateful-logit-probability-mean"][1], -np.log(3.0))
        self.assertAlmostEqual(features["stateful-logit-probability-mean"][2], np.log(3.0))
        self.assertAlmostEqual(features["stateful-logit-probability-mean"][3], np.log(3.0))


if __name__ == "__main__":
    unittest.main()
