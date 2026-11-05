import unittest

import numpy as np

from export_ternary_sign_size_forecast import endpoint_law


class TernaryExportTest(unittest.TestCase):
    def test_direction_order_and_learned_zero_probability(self):
        returns, probability, mean, size, error = endpoint_law(
            np.array([[.2, .5, .3], [.1, .8, .1]], dtype=np.float32),
            np.array([.01, .04]), np.array([.02, .04]))
        np.testing.assert_allclose(returns, [[-.02, 0, .01], [-.04, 0, .04]])
        np.testing.assert_allclose(mean, [-.001, 0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(size, [.007, .008], atol=1e-9, rtol=0)
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-15, rtol=0)
        self.assertLessEqual(error, 1e-7)

    def test_rejects_unphysical_support_and_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            endpoint_law(np.array([[.5, 0, .5]]), np.array([.1]), np.array([1.]))
        with self.assertRaises(ValueError):
            endpoint_law(np.array([[.5, .5, .5]]), np.array([.1]), np.array([.1]))


if __name__ == "__main__":
    unittest.main()
