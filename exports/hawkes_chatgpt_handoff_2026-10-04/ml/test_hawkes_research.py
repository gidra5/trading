import unittest
import numpy as np
from hawkes_research import Model, Mode, branching, forecast_operators, from_theta, linear_objective, linear_predictions, open_loop_linear, radius


class JointHawkesTests(unittest.TestCase):
    def setUp(self):
        self.model = Model((.4, .7), (Mode(0, 0, .3, 2.), Mode(0, 1, .1, 11.), Mode(1, 0, .2, 5.), Mode(1, 1, .4, 50.)))
        self.theta, self.target, self.source = self.model.arrays()
        self.y = np.random.default_rng(123).poisson(2, (100, 2)).astype(float)
        self.valid = np.ones(100, dtype=bool)
        self.valid[40:45] = False
        self.score = self.valid.copy()
        self.mean = np.array([2., 2.])

    def test_all_joint_parameter_gradients(self):
        _, gradient = linear_objective(self.theta, self.target, self.source, self.y, self.valid, self.score, self.mean)
        for j in range(len(self.theta)):
            step = np.zeros_like(self.theta)
            step[j] = 1e-5
            plus = linear_objective(self.theta + step, self.target, self.source, self.y, self.valid, self.score, self.mean)[0]
            minus = linear_objective(self.theta - step, self.target, self.source, self.y, self.valid, self.score, self.mean)[0]
            self.assertAlmostEqual(gradient[j], (plus - minus) / 2e-5, places=6)

    def test_recursion_agrees_with_direct_convolution(self):
        valid = np.ones(100, dtype=bool)
        prediction = linear_predictions(self.theta, self.target, self.source, self.y, valid, self.mean)
        expected = np.broadcast_to(self.model.background, self.y.shape).copy()
        for mode in self.model.modes:
            d = np.exp(-1 / mode.tau)
            for t in range(len(self.y)):
                trace = self.mean[mode.source] * d ** t
                for s in range(t):
                    trace += (1 - d) * d ** (t - s - 1) * self.y[s, mode.source]
                expected[t, mode.target] += mode.mass * trace
        np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)

    def test_future_observations_cannot_change_predictions(self):
        altered = self.y.copy()
        altered[50:] = 1000
        before = linear_predictions(self.theta, self.target, self.source, self.y, self.valid, self.mean)
        after = linear_predictions(self.theta, self.target, self.source, altered, self.valid, self.mean)
        np.testing.assert_array_equal(before[:51], after[:51])

    def test_integrated_discrete_kernel_and_cross_channel_stability(self):
        matrix = branching(self.model)
        np.testing.assert_allclose(matrix, [[.3, .1], [.2, .4]])
        for mode in self.model.modes:
            d = np.exp(-1 / mode.tau)
            integral = mode.mass * (1 - d) * np.sum(d ** np.arange(10000))
            self.assertAlmostEqual(integral, mode.mass, places=10)
        self.assertLess(radius(matrix), .95)
        self.assertGreater(radius(np.array([[.8, .4], [.4, .8]])), 1)

    def test_round_trip_preserves_independent_edge_scales(self):
        restored = from_theta(self.theta, self.target, self.source)
        np.testing.assert_allclose(restored.arrays()[0], self.theta)
        self.assertEqual(Model.deserialize(self.model.serialize()), self.model)

    def test_open_loop_operators_match_expected_future_recursion(self):
        horizons = (1, 5, 60, 3600)
        operators = forecast_operators(self.model, horizons)
        state = np.array([1., 2., 3., 4.])
        cumulative = np.zeros(2)
        expected = []
        for t in range(1, max(horizons) + 1):
            rate = np.array(self.model.background)
            for j, mode in enumerate(self.model.modes):
                rate[mode.target] += mode.mass * state[j]
            cumulative += rate
            for j, mode in enumerate(self.model.modes):
                d = np.exp(-1 / mode.tau)
                state[j] = d * state[j] + (1 - d) * rate[mode.source]
            if t in horizons:
                expected.append(cumulative.copy())
        actual = operators @ np.array([1., 2., 3., 4., 1.])
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_open_loop_predictions_do_not_read_future_observations(self):
        altered = self.y.copy()
        altered[50:] = 1000
        horizons = (1, 5, 60, 3600)
        arguments = (self.theta, self.target, self.source)
        operators = forecast_operators(self.model, horizons)
        before = open_loop_linear(*arguments, self.y, self.valid, self.mean, np.array([50]), operators)
        after = open_loop_linear(*arguments, altered, self.valid, self.mean, np.array([50]), operators)
        np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
