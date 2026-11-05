import unittest
import numpy as np
from hawkes_research import Model, Mode, linear_objective, radius
from hawkes_modulation import Anchor, Modulated, modulated_objective, unpack, upper_branching
from hawkes_observed_rollouts import observed_states, rollout


class ModulatedHawkesTests(unittest.TestCase):
    def setUp(self):
        self.model = Modulated(Model((.4, .7), (Mode(0, 0, .2, 2.), Mode(1, 1, .3, 5.))),
                               (Anchor(0, 10.), Anchor(1, 30.)), ((.2, .1), (.1, .3)))
        self.theta, self.target, self.source, self.anchor_source = self.model.arrays()
        self.y = np.random.default_rng(123).poisson(2, (100, 2)).astype(float)
        self.valid = np.ones(100, dtype=bool)
        self.valid[40:45] = False
        self.mean = np.array([2., 2.])

    def test_all_joint_derivatives_including_gate_and_slow_scales(self):
        arguments = (self.target, self.source, self.anchor_source, self.y, self.valid, self.valid, self.mean)
        _, gradient = modulated_objective(self.theta, *arguments)
        for j in range(len(self.theta)):
            step = np.zeros_like(self.theta)
            step[j] = 1e-5
            difference = (modulated_objective(self.theta + step, *arguments)[0] - modulated_objective(self.theta - step, *arguments)[0]) / 2e-5
            self.assertAlmostEqual(gradient[j], difference, places=6, msg=f'parameter {j}')

    def test_zero_modulation_recovers_nested_linear_likelihood(self):
        theta = self.theta.copy()
        theta[6:10] = 0.
        loss, _ = modulated_objective(theta, self.target, self.source, self.anchor_source, self.y, self.valid, self.valid, self.mean)
        original, target, source = self.model.base.arrays()
        linear, _ = linear_objective(original, target, source, self.y, self.valid, self.valid, self.mean)
        self.assertAlmostEqual(loss, linear, places=12)

    def test_upper_branching_bounds_every_event_amplitude(self):
        matrix = upper_branching(self.theta, self.target, self.source, self.anchor_source)
        self.assertGreater(matrix[0, 0], .2)
        self.assertGreater(matrix[1, 1], .3)
        self.assertLess(radius(matrix), 1.)
        self.assertEqual(unpack(self.theta, self.target, self.source, self.anchor_source).serialize()['beta'], self.model.serialize()['beta'])

    def test_filter_likelihood_agrees_with_fit_and_is_causal(self):
        arguments = (self.theta, self.target, self.source, self.anchor_source)
        prediction, fast, slow = observed_states(*arguments, self.y, self.valid, self.mean, np.array([50]))
        loss, _ = modulated_objective(*arguments, self.y, self.valid, self.valid, self.mean)
        computed = (prediction[self.valid] - self.y[self.valid] * np.log(prediction[self.valid])).sum(axis=1).mean()
        self.assertAlmostEqual(loss, computed, places=12)
        changed = self.y.copy()
        changed[50:] = 1000
        altered, altered_fast, altered_slow = observed_states(*arguments, changed, self.valid, self.mean, np.array([50]))
        np.testing.assert_array_equal(prediction[:51], altered[:51])
        np.testing.assert_array_equal(fast, altered_fast)
        np.testing.assert_array_equal(slow, altered_slow)

    def test_open_loop_one_second_mean_is_exact_and_deterministic(self):
        arguments = (self.theta, self.target, self.source, self.anchor_source)
        prediction, fast, slow = observed_states(*arguments, self.y, self.valid, self.mean, np.array([50]))
        mean, stderr, divergence = rollout(*arguments, self.mean, fast, slow, np.array([1, 5]), 64, 123, np.array([50]))
        np.testing.assert_allclose(mean[0, 0], prediction[50], atol=1e-12)
        np.testing.assert_allclose(stderr[0, 0], 0., atol=1e-12)
        np.testing.assert_array_equal(divergence[:, 0], 0.)

    def test_nonlinear_runaway_is_reported_without_discarding_failed_paths(self):
        model = Modulated(Model((1., 1.), (Mode(0, 0, .4, 1.),)), (Anchor(0, 5.),), ((3.,),))
        arguments = model.arrays()
        prediction, _, divergence = rollout(*arguments, np.array([2., 2.]), np.array([[100.]]), np.array([[100.]]), np.array([1, 60]), 32, 123, np.array([1]))
        self.assertGreater(divergence[0, -1], 0.)
        self.assertTrue(np.isnan(prediction[0, -1]).all())


if __name__ == '__main__':
    unittest.main()
