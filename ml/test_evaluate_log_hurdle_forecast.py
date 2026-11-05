"""Independent checks of coordinate scoring and physical endpoint export."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from evaluate_log_hurdle_forecast import endpoint_log_law, score_log_law, class_metrics


class LogHurdleEvaluationTests(unittest.TestCase):
    def test_actual_log_and_simple_expectations_are_distinct(self):
        # p(DOWN,ZERO,UP)=(1/4,1/2,1/4), two exponential magnitudes.
        raw = np.tile([0., 0., .1, .2, 1., 1.], (3, 1)).astype(np.float64)
        y = np.array([-.1, 0., .2])
        scores = score_log_law(raw, y)
        log_mean = .25*.1-.25*.2
        simple_mean = .25*(1/.9-1)-.25*(1-1/1.2)
        self.assertNotAlmostEqual(math.log1p(simple_mean), log_mean, places=5)
        self.assertAlmostEqual(scores["logReturns"]["allTickDeployedMseBpsSquared"],
            float(np.square(np.log1p(y)-log_mean).mean()*1e8), places=6)
        self.assertAlmostEqual(scores["simpleReturns"]["allTickDeployedMseBpsSquared"],
            float(np.square(y-simple_mean).mean()*1e8), places=6)
        self.assertEqual(scores["logReturns"]["activeExamples"], 2)
        self.assertAlmostEqual(scores["logReturns"]["activeConditionalMseBpsSquared"],
            float(np.square(np.log1p(y[y != 0])-2*log_mean).mean()*1e8), places=6)

    def test_density_coordinate_jacobian_and_exact_zero_atom(self):
        raw = np.tile([0., 0., .1, .2, 1., 1.], (3, 1)).astype(np.float64)
        y = np.array([-.1, 0., .2])
        x = np.log1p(y)
        # Direct density of Y=exp(X)-1, including dy/dx=exp(x).
        direct_simple_nll = np.array([
            -math.log(.25*5*math.exp(5*x[0])/(1+y[0])),
            -math.log(.5),
            -math.log(.25*10*math.exp(-10*x[2])/(1+y[2]))])
        scored = score_log_law(raw, y)
        self.assertAlmostEqual(scored["simpleReturnNegativeLogLikelihood"], float(direct_simple_nll.mean()), places=13)
        self.assertAlmostEqual(scored["logReturnNegativeLogLikelihood"], float((direct_simple_nll-x).mean()), places=13)

    def test_positive_down_price_uses_gross_moment_directly(self):
        raw = np.array([[0., 0., .1, 1e10, 1., 2.]], dtype=np.float64)
        law = endpoint_log_law(raw)
        expected_down = (1+1e10/2)**-2
        self.assertGreater(law["gross"][0, 0], 0)
        self.assertEqual(1-(1-expected_down), 0.)  # subtraction would erase it
        self.assertAlmostEqual(law["gross"][0, 0]/expected_down, 1., places=13)
        np.testing.assert_allclose(law["probabilities"], [[.25, .5, .25]], rtol=0, atol=1e-15)
        np.testing.assert_allclose((law["gross"]*law["probabilities"]).sum(1), 1+law["mean"], atol=1e-15, rtol=0)

    def test_class_order_and_conditional_sign_do_not_confuse_zeros(self):
        probability = np.array([[.8, .1, .1], [.1, .8, .1], [.2, .1, .7]])
        y = np.array([-.1, 0., .2])
        output = class_metrics(np.log(probability), y)
        self.assertEqual(output["threeClassConfusion"], np.eye(3, dtype=int).tolist())
        self.assertEqual(output["threeClassAccuracy"], 1.)
        self.assertEqual(output["conditionalActiveSignAccuracy"], 1.)
        self.assertEqual(output["activeExamples"], 2)
        self.assertAlmostEqual(output["conditionalActiveSignLogLoss"], -.5*(math.log(.8/.9)+math.log(.7/.9)), places=14)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
