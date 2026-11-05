import unittest
import numpy as np

from horizon_price_forecast import horizon_split
from sign_size_forecast import DOWNSIDE_MAXIMUM, moment_values, predict_laws, three_point_law
from stochastic_price_forecast import TimelineSplit


class SignSizeContractTest(unittest.TestCase):
    def test_sign_conditional_sizes_preserve_mean_without_independence(self):
        returns, probabilities, _ = three_point_law(.2, np.array([.6]), np.array([.03]), np.array([.01]))
        mean, size = moment_values(returns, probabilities)
        np.testing.assert_allclose(mean, [.0112], atol=1e-15)
        np.testing.assert_allclose(size, [.0176], atol=1e-15)
        self.assertNotAlmostEqual(float(mean[0]), (2*.6-1)*float(size[0]))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_equal_average_size_can_have_opposite_expected_returns(self):
        mean, size = moment_values(*three_point_law(0., np.array([.5, .5]),
            np.array([.004, .002]), np.array([.002, .004]))[:2])
        np.testing.assert_allclose(size, [.003, .003])
        np.testing.assert_allclose(mean, [.001, -.001])

    def test_fair_sign_magnitude_control_remains_zero_after_symmetric_clipping(self):
        artifact = {"contract": "causal-production178-sign-conditional-simple-size-three-point-v1",
                    "constants": {"pZero": .1, "pUpGivenActive": .9, "upMeanBps": 20000,
                                  "downMeanBps": 2000, "pooledActiveMeanBps": 18200},
                    "selectedHeads": {"sign": "constant", "up": "constant", "down": "constant"}, "models": {}}
        law = predict_laws(artifact, np.zeros((2, 178)))["magnitudeOnlyFairSign"]
        mean, size = moment_values(*law[:2])
        np.testing.assert_array_equal(mean, [0., 0.])
        np.testing.assert_allclose(size, .9*DOWNSIDE_MAXIMUM)
        self.assertEqual(law[2]["symmetricPooledClippedCount"], 2)
        self.assertTrue(np.all(1+law[0] > 0))

    def test_horizon_targets_stay_inside_split_and_features_ignore_future_rows(self):
        timeline = np.zeros((150, 59), dtype=np.float32)
        timeline[:, 0] = np.arange(150, dtype=np.float32)*1e-6
        origins = np.arange(119, 149)
        source = TimelineSplit(timeline, origins, 1000)
        split = horizon_split(source, 10)
        self.assertEqual(int(split.origins[-1]), 139)
        self.assertTrue(np.all(split.origins+10 < len(timeline)))
        original = source.features(np.array([125]))
        changed = timeline.copy()
        changed[126:] = 1000
        np.testing.assert_array_equal(original, TimelineSplit(changed, origins, 1000).features(np.array([125])))
        expected = float(np.sum(timeline[126:136, 0], dtype=np.float64))
        self.assertAlmostEqual(float(split.targets()[125-119]), expected, places=9)


if __name__ == "__main__":
    unittest.main()
