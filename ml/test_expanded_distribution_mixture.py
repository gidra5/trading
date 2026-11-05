import unittest
import numpy as np
from scipy.special import logsumexp
from expanded_distribution_mixture import fit_weights, pool_loglik, simple_law


class ExpandedMixtureTests(unittest.TestCase):
    def test_known_mse_optimum(self):
        x = np.array([[-2., 1.], [1., -1.], [3., 4.]])*1e-4
        y = x@np.array([.3, .7])
        w = fit_weights(x, np.zeros_like(x), y, 'mse')
        np.testing.assert_allclose(w, [.3, .7], atol=1e-7)

    def test_nll_recovers_categorical_law(self):
        laws = np.array([[.9, .2], [.1, .8]])
        ll = np.log(laws[np.repeat([0, 1], [55, 45])])
        w = fit_weights(np.zeros_like(ll), ll, np.ones(100), 'nll')
        np.testing.assert_allclose(w, [.5, .5], atol=1e-6)
        self.assertAlmostEqual(np.exp(pool_loglik(np.log(laws), w)).sum(), 1.)

    def test_identical_laws_and_boundary(self):
        ll = np.array([[-1000., -1000.], [-.2, -.2]])
        np.testing.assert_allclose(pool_loglik(ll, [.1, .9]), ll[:, 0])
        np.testing.assert_allclose(pool_loglik(ll, [1., 0.]), ll[:, 0])
        with self.assertRaises(ValueError):
            pool_loglik(ll, [-.1, 1.1])

    def test_simple_to_log_atom_and_jacobian(self):
        from scipy.stats import gamma, beta
        from scipy.special import expit
        p = np.tile([.1, -.2, .04, .03, 2., 3.], (3, 1))
        y = np.array([0., .01, -.01])
        _, ll = simple_law(p, y)
        z, up = expit(p[0, :2])
        expected = [np.log(z), np.log((1-z)*up)+gamma.logpdf(np.expm1(y[1]), 2, scale=.02)+y[1],
            np.log((1-z)*(1-up))+beta.logpdf(-np.expm1(y[2]), 3, 3*(1-.03)/.03)+y[2]]
        np.testing.assert_allclose(ll, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
