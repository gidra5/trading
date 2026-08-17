import unittest

import numpy as np

from analyze_one_second_dependence_model import (
    AcfAccumulator,
    aggregate_aligned_minutes,
    fit_states,
    fit_transition,
    jensen_shannon_bits,
    permuted_block_indexes,
    variance_ratio_from_acf,
)


class OneSecondDependenceModelTest(unittest.TestCase):
    def test_acf_detects_alternating_returns(self) -> None:
        values = np.tile(np.array([-1.0, 1.0]), 501)
        accumulator = AcfAccumulator.create()
        accumulator.add_returns(values)
        acf = accumulator.finish()["return"]
        self.assertAlmostEqual(acf[1], -1.0, places=3)
        self.assertAlmostEqual(acf[2], 1.0, places=3)

    def test_state_transition_is_stochastic(self) -> None:
        counts = np.tile(np.arange(1, 61, dtype=np.uint8), 20)
        variance = counts.astype(np.float64) ** 2
        states = fit_states(counts, variance)
        transition = fit_transition(states["ids"], states["count"])
        np.testing.assert_allclose(np.sum(transition, axis=1), 1.0)
        self.assertEqual(np.min(states["ids"]), 0)
        self.assertEqual(np.max(states["ids"]), states["count"] - 1)

    def test_variance_ratio_and_js_identity(self) -> None:
        acf = [1.0, 0.1, 0.0]
        self.assertAlmostEqual(variance_ratio_from_acf(acf, 2), 1.1)
        probability = np.array([0.25, 0.75])
        self.assertAlmostEqual(jensen_shannon_bits(probability, probability), 0.0)

    def test_stationary_blocks_are_contiguous_inside_each_block(self) -> None:
        indexes = permuted_block_indexes(
            11,
            block_length=4,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(np.diff(indexes[:4]), np.ones(3))
        np.testing.assert_array_equal(np.diff(indexes[4:8]), np.ones(3))
        np.testing.assert_array_equal(np.diff(indexes[8:]), np.ones(2))
        np.testing.assert_array_equal(np.sort(indexes), np.arange(11))

    def test_minute_aggregation_advances_to_aligned_boundary(self) -> None:
        # Element zero is the return for UTC minute index 1. With factor 3,
        # aggregation starts at minute index 3 and uses indexes 3, 4, 5.
        values = np.arange(1.0, 9.0)
        np.testing.assert_array_equal(
            aggregate_aligned_minutes(values, 3),
            np.array([3.0 + 4.0 + 5.0, 6.0 + 7.0 + 8.0]),
        )


if __name__ == "__main__":
    unittest.main()
