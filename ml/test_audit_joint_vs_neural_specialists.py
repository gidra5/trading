import unittest
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from audit_joint_vs_neural_specialists import decompose, score_components, side_integrals


class MarginalAuditTests(unittest.TestCase):
    def test_split_gaussian_integrals_recover_analytic_sign_and_size(self):
        knots = np.array([0., .2, .5, .8, 1.])
        widths = np.diff(knots)
        areas = np.zeros(5)
        areas[:-1] += widths/2
        areas[1:] += widths/2
        sigma = .01
        model = SimpleNamespace(sigma=sigma, basis={"coordinate":"gaussian-cdf", "knots":knots, "basisAreas":areas})
        um, us = side_integrals(model, True, 384)
        dm, ds = side_integrals(model, False, 384)
        np.testing.assert_allclose(um+dm, 1, atol=1e-13)
        pu, up, down, diagnostic = decompose(areas[None,:], um,dm,us,ds)
        np.testing.assert_allclose(pu, .5, atol=1e-13)
        np.testing.assert_allclose(.5*up, np.exp(sigma*sigma/2)*ndtr(sigma)-.5, atol=3e-9)
        np.testing.assert_allclose(.5*down, .5-np.exp(sigma*sigma/2)*ndtr(-sigma), atol=3e-9)
        self.assertEqual(diagnostic["undefinedUpSizeCount"], 0)

    def test_impossible_side_has_zero_placeholder_and_no_moment(self):
        p, up, down, diagnostic = decompose(np.array([[1.]]), np.array([0.]),np.array([1.]),np.array([0.]),np.array([.2]))
        np.testing.assert_array_equal(p, 0)
        np.testing.assert_array_equal(up, 0)
        np.testing.assert_allclose(down, .2)
        self.assertEqual(diagnostic["undefinedUpSizeCount"], 1)
        with self.assertRaisesRegex(ValueError,"zero-mass"):
            decompose(np.array([[1.]]),np.array([0.]),np.array([1.]),np.array([.1]),np.array([.2]))

    def test_active_conditional_and_unconditional_scores_have_distinct_gate(self):
        y = np.array([.1,-.1,0.,0.])
        p = np.array([1.,0.,1.,0.])
        size = np.full(4,.1)
        constants = {"pZero":.5,"pUpGivenActive":.5,"upMeanSimple":.1,"downMeanSimple":.1,"pooledActiveMeanSimple":.1}
        scored = score_components(y,.5,p,size,size,constants)
        self.assertAlmostEqual(scored["activeOnly"]["conditionalActiveMeanMseSkillVsZero"],1.)
        self.assertAlmostEqual(scored["activeOnly"]["deployedUnconditionalMeanMseSkillVsZero"],.75)
        self.assertAlmostEqual(scored["allTickSignedMeanMseSkillVsZero"],.5)
        self.assertAlmostEqual(scored["zeroTickContributionToTotalMseBpsSquared"],125000.)


if __name__ == "__main__":
    unittest.main()
