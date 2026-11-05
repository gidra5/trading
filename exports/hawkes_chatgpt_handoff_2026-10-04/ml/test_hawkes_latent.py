import unittest
import numpy as np
from scipy.special import logsumexp, gammaln
from hawkes_research import Model, Mode, linear_objective, linear_predictions
from hawkes_latent import Diffusion, LatentModel, core_branching, decode, filter_model, integrated_count, poisson_from_uniform, seed_latent


class AdaptedLatentTests(unittest.TestCase):
    def test_single_uniform_poisson_coupling_matches_exact_quantiles(self):
        from scipy.stats import poisson
        for mean in (.01, 2., 80., 499.):
            for u in (1e-6, .01, .1, .5, .9, .999, 1 - 1e-8):
                self.assertEqual(poisson_from_uniform(mean, u), int(poisson.ppf(u, mean)))

    def test_latent_count_integral_matches_exhaustive_sum_even_for_rare_events(self):
        cases = [(.01, [5., 2.], [10., 4.], [300., 150.]),
                 (.2, [1., 2.], [.5, 2.], [0., 0.]),
                 (3., [1., 2.], [4., 1.], [500., 300.]),
                 (.2, [1., 2.], [0., 0.], [5., 8.])]
        for prior, base, slope, observed in cases:
            base, slope, observed = map(np.array, [base, slope, observed])
            n = np.arange(2000)
            rates = base[None] + n[:, None] * slope[None]
            weights = n * np.log(prior) - prior - gammaln(n + 1) + (observed * np.log(rates) - rates).sum(axis=1)
            exact = logsumexp(weights)
            ll, lo, hi, mean, active = integrated_count(prior, base, slope, observed)
            self.assertAlmostEqual(ll, exact, places=9)
            posterior = np.exp(weights - exact)
            self.assertAlmostEqual(mean, np.dot(n, posterior), places=9)
            self.assertAlmostEqual(active, posterior[1:].sum(), places=9)
            if lo >= 0:
                self.assertLess(posterior[:lo].sum() + posterior[hi + 1:].sum(), 3e-12)

    def test_disconnected_latent_stream_recovers_observed_model_exactly(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(0, 1, .1, 10.), Mode(1, 0, .1, 5.), Mode(1, 1, .4, 40.)))
        latent = LatentModel((*observed.background, .2), observed.modes, beta=tuple(() for _ in observed.modes))
        y = np.random.default_rng(123).poisson(3, (100, 2)).astype(float)
        valid = np.ones(100, dtype=bool)
        valid[40:45] = False
        mean = np.array([3., 3.])
        prediction, losses, *_ = filter_model(latent, y, valid, mean, 16, 42)
        theta, target, source = observed.arrays()
        direct = linear_predictions(theta, target, source, y, valid, mean)
        np.testing.assert_allclose(prediction, direct, atol=1e-12)
        direct_loss, _ = linear_objective(theta, target, source, y, valid, valid, mean)
        self.assertAlmostEqual(losses[valid].mean(), direct_loss, places=12)

    def test_causal_predictions_and_latent_feedback(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        latent = seed_latent(observed, tau=5, strength=.1)
        y = np.random.default_rng(123).poisson(3, (100, 2)).astype(float)
        valid = np.ones(100, dtype=bool)
        mean = np.array([3., 3.])
        original = filter_model(latent, y, valid, mean, 32, 42)
        changed = y.copy()
        changed[50:] = 500
        altered = filter_model(latent, changed, valid, mean, 32, 42)
        np.testing.assert_array_equal(original[0][:51], altered[0][:51])
        self.assertTrue(np.isfinite(original[1]).all())
        self.assertTrue(np.all(original[3] > 0))
        self.assertTrue(np.all((original[5] >= 0) & (original[5] <= 1 + 1e-12)))

    def test_joint_parameter_vector_contains_each_independent_scale(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        latent = seed_latent(observed, tau=5, strength=.1)
        restored = decode(latent.raw(), latent.layout())
        self.assertEqual(len(latent.raw()), 3 + 2 * len(latent.modes))
        self.assertLessEqual(np.max(np.abs(np.linalg.eigvals(core_branching(restored)))), .95000001)
        np.testing.assert_allclose(restored.raw(), latent.raw(), atol=1e-12)

    def test_continuous_prior_likelihood_matches_gaussian_quadrature(self):
        # Essentially independent OU draws at this decay; Poisson-lognormal
        # likelihood has an independent quadrature reference, including the
        # unequal/signed side loadings rather than a residual proxy.
        from numpy.polynomial.hermite import hermgauss
        loading = np.array([.6, -.4, 0.])
        background = np.array([2., 3., .2])
        model = LatentModel(tuple(background), (), diffusions=(Diffusion(.1, tuple(loading)),))
        y = np.array([[4., 1.]])
        p = 100000
        result = filter_model(model, y, np.ones(1, dtype=bool), np.array([2., 3.]), p, 98123)
        z, w = hermgauss(100)
        rates = background[None, :2] * np.exp(np.sqrt(2) * z[:, None] * loading[None, :2] - .5 * loading[None, :2] ** 2)
        exact = logsumexp(np.log(w / np.sqrt(np.pi)) + (y * np.log(rates) - rates).sum(axis=1))
        self.assertAlmostEqual(result[1][0], -exact, delta=.01)
        np.testing.assert_allclose(result[0][0], background[:2], atol=1e-5)

    def test_diffusion_prediction_is_causal_and_zero_loading_is_nested(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        base = seed_latent(observed, tau=5, strength=.1)
        model = LatentModel(base.background, base.modes, beta=base.beta, diffusions=(Diffusion(50., (.5, -.3, .2)),))
        y = np.random.default_rng(123).poisson(3, (100, 2)).astype(float)
        valid, mean = np.ones(100, dtype=bool), np.array([3., 3.])
        original = filter_model(model, y, valid, mean, 64, 42)
        changed = y.copy()
        changed[50:] = 500
        altered = filter_model(model, changed, valid, mean, 64, 42)
        np.testing.assert_array_equal(original[0][:51], altered[0][:51])
        self.assertTrue(np.isfinite(original[1]).all())
        # With no L-connected edges and zero continuous loading, the observed
        # prediction and likelihood equal the analytic two-channel model.
        nested = LatentModel((*observed.background, .2), observed.modes, beta=tuple(() for _ in observed.modes), diffusions=(Diffusion(50., (0., 0., 0.)),))
        nested_result = filter_model(nested, y, valid, mean, 32, 42)
        theta, target, source = observed.arrays()
        np.testing.assert_allclose(nested_result[0], linear_predictions(theta, target, source, y, valid, mean), atol=1e-12)

    def test_multiple_diffusions_match_independent_gaussian_quadrature(self):
        # Two and four Gaussian components induce the same kind of two-side
        # covariance law. Compare the weighted filter likelihood to independent
        # quadrature, exercising all coordinates of the reused proposal workspace.
        from numpy.polynomial.hermite import hermgauss
        nodes, weight = hermgauss(80)
        for loadings in (((.6, 0., 0.), (0., -.4, 0.)),
                         ((.5, 0., .1), (.25, 0., 0.), (0., -.3, 0.), (0., .3, -.1))):
            background = np.array([2., 3., .2])
            observed = np.array([[4., 1.]])
            variance = np.sum(np.array(loadings) ** 2, axis=0)
            exact = 0.
            for side in range(2):
                rates = background[side] * np.exp(np.sqrt(2 * variance[side]) * nodes - .5 * variance[side])
                exact += logsumexp(np.log(weight / np.sqrt(np.pi)) + observed[0, side] * np.log(rates) - rates)
            model = LatentModel(tuple(background), (),
                                diffusions=tuple(Diffusion(.001, loading) for loading in loadings))
            particles = 32768
            result = filter_model(model, observed, np.ones(1, dtype=bool), background[:2], particles, 98123)
            log_score_se = np.sqrt(max(0., 1 / result[2][0] - 1 / particles))
            self.assertAlmostEqual(result[1][0], -exact, delta=5 * log_score_se + 1e-4)
            np.testing.assert_allclose(result[0][0], background[:2], atol=1e-12)

    def test_guided_continuous_proposal_resolves_rare_burst_with_correct_weight(self):
        from scipy.integrate import quad
        from scipy.optimize import minimize_scalar
        background, loading, observed = .2, 1.5, 200.
        def log_integrand(z):
            rate = background * np.exp(loading * z - .5 * loading ** 2)
            return observed * np.log(rate) - rate - .5 * z ** 2 - .5 * np.log(2 * np.pi)
        optimum = minimize_scalar(lambda z: -log_integrand(z), bounds=(-10, 10), method='bounded')
        peak = log_integrand(optimum.x)
        integral = quad(lambda z: np.exp(log_integrand(z) - peak), -10, 10, epsabs=1e-11, points=[optimum.x])[0]
        reference = peak + np.log(integral)
        model = LatentModel((background, 1., .2), (), diffusions=(Diffusion(.1, (loading, 0., 0.)),))
        result = filter_model(model, np.array([[observed, 0.]]), np.ones(1, dtype=bool), np.array([1., 1.]), 4096, 98123)
        # Sell count zero contributes -1 independently. Predictive mean remains
        # the prior .2, even though the posterior is guided to a rare large z.
        log_score_se = np.sqrt(1 / result[2][0] - 1 / 4096)
        self.assertLess(log_score_se, .02)
        self.assertAlmostEqual(result[1][0], -reference + 1, delta=5 * log_score_se + 1e-5)
        self.assertAlmostEqual(result[0][0, 0], background, delta=1e-5)

    def test_mixed_event_diffusion_likelihood_and_latent_posterior_match_enumeration(self):
        from numpy.polynomial.hermite import hermgauss
        # Independent Gaussian states make the second-bin likelihood an
        # independently computable Gaussian quadrature / Poisson-count sum.
        # A negligible OU decay removes history correlations for this control.
        mode = Mode(0, 2, 2., 1.)
        model = LatentModel((.5, .8, .4), (mode,), beta=((),), diffusions=(Diffusion(.001, (.4, -.2, 0.)),))
        y = np.array([[3., 3.], [12., 3.]])
        result = filter_model(model, y, np.ones(2, dtype=bool), np.array([1., 1.]), 65536, 1234)
        z, weight = hermgauss(100)
        counts = np.arange(40)
        d = np.exp(-1.)
        rate = np.array([.5, .8])[None, :] * np.exp(np.sqrt(2) * z[:, None] * np.array([.4, -.2]) - .5 * np.array([.4, -.2]) ** 2)
        full_rate = np.tile(rate[:, None, :], (1, len(counts), 1))
        full_rate[:, :, 0] += 2 * d * .4 + 2 * (1 - d) * counts[None, :]
        log_weight = np.log(weight[:, None] / np.sqrt(np.pi)) + counts[None, :] * np.log(.4) - .4 - gammaln(counts[None, :] + 1)
        log_weight += (y[1] * np.log(full_rate) - full_rate).sum(axis=2)
        reference = logsumexp(log_weight)
        posterior_mean = (np.exp(log_weight - reference) * counts[None, :]).sum()
        log_score_se = np.sqrt(1 / result[2][1] - 1 / 65536)
        self.assertLess(log_score_se, .02)
        self.assertAlmostEqual(result[1][1], -reference, delta=5 * log_score_se + 1e-5)
        self.assertAlmostEqual(result[4][1], posterior_mean, delta=.02)


if __name__ == '__main__':
    unittest.main()
