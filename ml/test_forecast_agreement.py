import unittest

import numpy as np

from audit_forecast_agreement import basis_cdf, gamma_cdf, gamma_loglik, law_distances, mix_likelihood


class AgreementTests(unittest.TestCase):
    def test_triangular_cdf_has_exact_boundaries_and_known_interior(self):
        result = basis_cdf(np.array([0., .5, 1.]), np.array([.25, .5, .25]),
                          np.array([0., .25, .5, .75, 1.]))
        np.testing.assert_allclose(result[:, 0], 0)
        np.testing.assert_allclose(result[:, -1], 1)
        np.testing.assert_allclose(result[1], [0, .125, .5, .875, 1])
        self.assertTrue((np.diff(result, axis=1) >= 0).all())

    def test_tv_detects_zero_atom_difference_and_is_symmetric(self):
        cdf = np.array([[0, .2, .8, 1.]])
        tv, js = law_distances(np.array([.2]), cdf, np.array([.8]), cdf)
        other_tv, other_js = law_distances(np.array([.8]), cdf, np.array([.2]), cdf)
        self.assertAlmostEqual(tv[0], .6)
        np.testing.assert_allclose(tv, other_tv)
        np.testing.assert_allclose(js, other_js)
        same_tv, same_js = law_distances(np.array([.2]), cdf, np.array([.2]), cdf)
        np.testing.assert_allclose([same_tv, same_js], 0)

    def test_refinement_reveals_disagreement_hidden_by_coarse_bins(self):
        j = np.array([[0, .1, .5, .9, 1.]])
        s = np.array([[0, .4, .5, .6, 1.]])
        fine_tv, fine_js = law_distances(np.array([0.]), j, np.array([0.]), s)
        coarse_tv, coarse_js = law_distances(np.array([0.]), j[:, [0, 2, 4]], np.array([0.]), s[:, [0, 2, 4]])
        self.assertGreater(fine_tv[0], coarse_tv[0])
        self.assertGreater(fine_js[0], coarse_js[0])

    def test_gamma_cdf_and_likelihood_match_signed_exponential_law(self):
        packed = np.array([[0, 0, 1, 1, 1, 1.]])
        cdf = gamma_cdf(packed, np.array([-np.inf, -1., 0., 1., np.inf]))
        np.testing.assert_allclose(cdf, [[0, .5/np.e, .5, 1-.5/np.e, 1]])
        labels = np.array([-1., 0, 1.])
        likelihood = gamma_loglik(np.repeat(packed, 3, axis=0), labels)
        np.testing.assert_allclose(np.exp(likelihood), [.25/np.e, .5, .25/np.e])

    def test_mixture_uses_density_average_and_retains_endpoints(self):
        j = np.log(np.array([.1, .7, .2]))
        s = np.log(np.array([.6, .2, .2]))
        np.testing.assert_allclose(mix_likelihood(0, j, s), s)
        np.testing.assert_allclose(mix_likelihood(1, j, s), j)
        np.testing.assert_allclose(np.exp(mix_likelihood(.3, j, s)), .3*np.exp(j)+.7*np.exp(s))


if __name__ == '__main__':
    unittest.main()
