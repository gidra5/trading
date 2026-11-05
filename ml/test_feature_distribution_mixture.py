import unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import approx_fprime
from feature_distribution_mixture import (LAYOUTS, classes, predict, loglik, fit_nll,
    mse_objective, full_prediction)


class FeatureMixtureTests(unittest.TestCase):
    def setUp(self):
        self.c = np.array([[[.2, .8, .04, .02], [.6, .3, .01, .06]]])

    def law_ll(self, y):
        c = np.repeat(self.c, len(y), axis=0)
        cat = classes(c)
        side = np.sign(y).astype(int)+1
        size = np.where(y[:, None] > 0, c[..., 2], c[..., 3])
        density = -np.log(size)-abs(y[:, None])/size
        return c, np.log(cat[np.arange(len(y)), :, side])+np.where(y[:, None] != 0, density, 0)

    def test_hybrid_uses_separate_sign_and_size(self):
        p = predict(self.c, np.array([[1., 0.], [0., 1.]]), 'ternaryMagnitude')
        np.testing.assert_allclose(p['classProbability'], [[.16, .2, .64]])
        self.assertAlmostEqual(p['mean'][0], .64*.01-.16*.06)
        self.assertAlmostEqual(p['absoluteMean'][0], .64*.01+.16*.06)

    def test_full_density_normalizes_with_zero_atom(self):
        for layout in LAYOUTS:
            w = np.array([[.25+.1*i, .75-.1*i] for i in range(len(LAYOUTS[layout]))])
            def density(x):
                c, ll = self.law_ll(np.array([x]))
                return np.exp(loglik(c, ll, np.array([x]), w, layout))[0]
            mass = quad(density, -np.inf, 0)[0]+quad(density, 0, np.inf)[0]+density(0)
            self.assertAlmostEqual(mass, 1., places=10)
            mean = quad(lambda x: x*density(x), -np.inf, 0)[0]+quad(lambda x: x*density(x), 0, np.inf)[0]
            self.assertAlmostEqual(mean, predict(self.c, w, layout)['mean'][0], places=10)

    def test_singleton_recovers_original_law(self):
        y = np.array([-.03, 0., .012])
        c, ll = self.law_ll(y)
        for layout in LAYOUTS:
            for i in range(2):
                w = np.tile(np.eye(2)[i], (len(LAYOUTS[layout]), 1))
                np.testing.assert_allclose(loglik(c, ll, y, w, layout), ll[:, i], atol=1e-13)
                np.testing.assert_allclose(predict(c, w, layout)['mean'],
                    full_prediction(c, np.eye(2)[i])['mean'], atol=1e-15)

    def test_gradient_through_all_factors(self):
        rng = np.random.default_rng(92)
        c = np.repeat(self.c, 9, axis=0)*rng.uniform(.8, 1.1, (9, 2, 4))
        y = rng.normal(0, .04, 9)
        for layout in LAYOUTS:
            w = rng.uniform(.1, 1, (len(LAYOUTS[layout]), 2))
            w /= w.sum(1, keepdims=True)
            f = lambda v: mse_objective(v, c, y, layout)[0]
            numerical = approx_fprime(w.ravel(), f, 1e-7)
            analytic = mse_objective(w.ravel(), c, y, layout)[1]
            np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=1e-5)

    def test_factor_nll_fit_improves_every_singleton(self):
        y = np.r_[np.zeros(30), np.linspace(.0001, .05, 55), -np.linspace(.0001, .09, 15)]
        c, ll = self.law_ll(y)
        for layout in LAYOUTS:
            w = fit_nll(c, ll, y, layout)
            fitted = -loglik(c, ll, y, w, layout).mean()
            self.assertLessEqual(fitted, min(-ll.mean(0))+1e-9)


if __name__ == '__main__':
    unittest.main()
