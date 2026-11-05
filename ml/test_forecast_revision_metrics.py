import unittest
import numpy as np
from forecast_revision_metrics import sample_metrics, fixed_target_horizons


class RevisionTests(unittest.TestCase):
    def test_translation_has_exact_wasserstein_and_zero_delta_variance(self):
        x = np.linspace(-.02,.02,10001)[None]
        metrics = sample_metrics(x, x+.001)
        self.assertAlmostEqual(metrics['w1Bps'][0], 10.)
        self.assertLess(metrics['log_priceDeltaVarianceOverFresh'][0], 1e-28)

    def test_same_empirical_law_with_different_pairing(self):
        x = np.array([[-.01,0,0,.01]])
        metrics = sample_metrics(x, x[:,::-1])
        self.assertEqual(metrics['w1Bps'][0], 0.)
        self.assertEqual(metrics['empiricalKsPercentagePoints'][0], 0.)
        self.assertGreater(metrics['log_priceDeltaVarianceOverFresh'][0], 0.)

    def test_fixed_target_shortens_horizon(self):
        target = np.array([10000., 10000.])
        np.testing.assert_array_equal(fixed_target_horizons([6400.,6401.],target),[3600.,3599.])
        with self.assertRaises(ValueError): fixed_target_horizons([10000.], [10000.])

    def test_ks_disjoint_laws(self):
        m = sample_metrics(np.array([[0.,0.,0.,0.]]),np.ones((1,4)))
        self.assertEqual(m['empiricalKsPercentagePoints'][0],100.)


if __name__ == '__main__': unittest.main()
