from __future__ import annotations

import unittest

import numpy as np

from export_neural_sign_size_forecast import combined_laws
from sign_size_forecast import moment_values


class SpecialistCombinationTest(unittest.TestCase):
    def test_conditional_size_dependence_and_controls(self):
        constants = {"pZero": .1, "pUpGivenActive": .5,
                     "upMeanSimple": .02, "downMeanSimple": .03}
        p_up = np.array([.8, .2])
        up = np.array([.01, .08])
        down = np.array([.08, .01])
        laws = combined_laws(constants, p_up, up, down)
        returns, probabilities, _ = laws["specialists"]
        mean, size = moment_values(returns, probabilities)
        np.testing.assert_allclose(mean, .9 * (p_up * up - (1-p_up) * down))
        np.testing.assert_allclose(size, .9 * (p_up * up + (1-p_up) * down))
        # High upward probability can still have negative mean, and vice versa.
        self.assertLess(mean[0], 0)
        self.assertGreater(mean[1], 0)
        fixed_sizes, sign_probs, _ = laws["sign-constant-size"]
        np.testing.assert_array_equal(sign_probs, probabilities)
        np.testing.assert_allclose(fixed_sizes, [[-.03, 0, .02]] * 2)
        same_sizes, fixed_probs, _ = laws["constant-sign-size"]
        np.testing.assert_array_equal(same_sizes, returns)
        np.testing.assert_allclose(fixed_probs, [[.45, .1, .45]] * 2)

    def test_100_percent_downside_is_bounded_and_recorded(self):
        constants = {"pZero": 0, "pUpGivenActive": .5,
                     "upMeanSimple": .02, "downMeanSimple": .03}
        laws = combined_laws(constants, np.array([.5]), np.array([.1]), np.array([1.2]))
        returns, probabilities, clipping = laws["specialists"]
        self.assertEqual(clipping["downsideClippedCount"], 1)
        self.assertTrue(np.all(1 + returns > 0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
