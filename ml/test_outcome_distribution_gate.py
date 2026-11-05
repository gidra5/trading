import json
import unittest

import numpy as np
from scipy.special import expit

from outcome_distribution_gate import (CONTRACT, RETURN_EDGES, GateData, normalized_prediction,
    objective, outcome_regions, predict_gate)
from train_outcome_distribution_gate import specialist_regions


def fixture(seed=9):
    rng = np.random.default_rng(seed)
    n, bins = 12, len(RETURN_EDGES)
    j, s = rng.dirichlet(np.ones(bins), n), rng.dirichlet(np.ones(bins), n)
    regions = np.arange(n) % bins
    points = np.r_[0, np.linspace(-.001, .001, bins-1)]
    return GateData(j, s, j*points, s*points, np.log(j[np.arange(n), regions]),
        np.log(s[np.arange(n), regions]), regions, rng.normal(size=(n, 2)))


class OutcomeGateTests(unittest.TestCase):
    def test_constant_weight_recovers_the_ordinary_full_mixture(self):
        d = fixture()
        result = normalized_prediction(np.full(d.jmass.shape, .7), d)
        w = expit(.7)
        np.testing.assert_allclose(result['normalizer'], 1, atol=1e-15)
        np.testing.assert_allclose(result['regionMass'], w*d.jmass+(1-w)*d.smass)
        np.testing.assert_allclose(result['mean'], (w*d.jmoment+(1-w)*d.smoment).sum(axis=1))
        np.testing.assert_allclose(np.exp(result['logLikelihood']), w*np.exp(d.jll)+(1-w)*np.exp(d.sll))

    def test_outcome_weights_require_normalization_including_zero(self):
        d = fixture()
        logits = np.where(d.jmass > d.smass, 4., -4.)
        result = normalized_prediction(logits, d)
        self.assertTrue(np.all(result['normalizer'] > 1))
        np.testing.assert_allclose(result['regionMass'].sum(axis=1), 1, atol=1e-15)
        expected_zero = (expit(logits[:, 0])*d.jmass[:, 0]+expit(-logits[:, 0])*d.smass[:, 0])/result['normalizer']
        np.testing.assert_allclose(result['regionMass'][:, 0], expected_zero)

    def test_complete_normalized_loss_gradient_matches_finite_differences(self):
        d = fixture()
        design = np.column_stack((np.ones(len(d.region)), d.features))
        prior = np.zeros((3, len(RETURN_EDGES)))
        beta = np.random.default_rng(42).normal(scale=.4, size=prior.size)
        _, gradient = objective(beta, design, d, prior, .02)
        epsilon = 1e-5
        numerical = []
        for k in range(len(beta)):
            direction = np.zeros_like(beta); direction[k] = epsilon
            numerical.append((objective(beta+direction, design, d, prior, .02)[0]
                -objective(beta-direction, design, d, prior, .02)[0])/(2*epsilon))
        np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=1e-6)

    def test_identical_sources_are_unchanged_by_any_outcome_gate(self):
        d = fixture()
        same = GateData(d.jmass, d.jmass, d.jmoment, d.jmoment, d.jll, d.jll, d.region, d.features)
        result = normalized_prediction(np.random.default_rng(6).normal(size=d.jmass.shape), same)
        np.testing.assert_allclose(result['normalizer'], 1)
        np.testing.assert_allclose(result['regionMass'], d.jmass)
        np.testing.assert_allclose(result['logLikelihood'], d.jll)

    def test_prediction_does_not_read_future_label_and_gate_serializes(self):
        d = fixture()
        gate = {'contract': CONTRACT, 'contextEnabled': True, 'center': [0., 0.], 'scale': [1., 1.],
                'coefficients': np.random.default_rng(7).normal(size=(3, len(RETURN_EDGES))).tolist()}
        result = predict_gate(gate, d)
        changed = GateData(d.jmass, d.smass, d.jmoment, d.smoment, d.jll+100, d.sll-100,
            np.zeros_like(d.region), d.features)
        other = predict_gate(json.loads(json.dumps(gate)), changed)
        for key in ('mean', 'weights', 'normalizer', 'regionMass'):
            np.testing.assert_array_equal(result[key], other[key])

    def test_gamma_region_integrals_preserve_atom_and_analytic_signed_mean(self):
        packed = np.array([[.2, -.3, .0001, .0002, .7, 1.3], [-1, .4, .01, .02, 2, 3]])
        mass, moment = specialist_regions(packed)
        np.testing.assert_allclose(mass.sum(axis=1), 1, atol=1e-15)
        np.testing.assert_array_equal(mass[:, 0], expit(packed[:, 0]))
        expected = (1-expit(packed[:, 0]))*(expit(packed[:, 1])*packed[:, 2]-expit(-packed[:, 1])*packed[:, 3])
        np.testing.assert_allclose(moment.sum(axis=1), expected, atol=1e-18)
        self.assertTrue(np.all(moment[:, 0] == 0))

    def test_zero_atom_is_not_a_continuous_bin(self):
        mapped = outcome_regions(np.array([-100, -1e-20, 0, 1e-20, 100.]))
        self.assertEqual(mapped[2], 0)
        self.assertTrue(np.all(mapped[[0, 1, 3, 4]] != 0))
        self.assertNotEqual(mapped[1], mapped[3])


if __name__ == '__main__':
    unittest.main()
