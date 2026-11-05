from __future__ import annotations

import unittest

import numpy as np

from audit_ternary_finetune import class_metrics, diagnostics, stable_class_logloss


class TernaryDiagnosticTest(unittest.TestCase):
    def test_confusion_order_and_exact_zero_recall(self):
        y = np.array([-.01, 0., .01, -.002, -0., .003])
        probability = np.eye(3)[[0, 1, 2, 1, 2, 0]]
        result = class_metrics(y, probability)
        self.assertEqual(result["confusionRowsActualColumnsPredicted"], [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertEqual(result["classRecall"], {"DOWN": .5, "ZERO": .5, "UP": .5})
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["actualCounts"], [2, 2, 2])
        self.assertEqual(result["actualClassProbabilityBelowScoreFloorCount"], 3)
        self.assertEqual(stable_class_logloss(np.array([.01]), np.array([[1000., 0., 0.]])), 1000.)

    def test_active_conditional_mean_has_separate_gate_and_zero_error(self):
        y = np.array([.01, -.01, 0., 0.])
        p = np.array([[0., .5, .5], [.5, .5, 0.], [0., .5, .5], [.5, .5, 0.]])
        constants = {"pZero": .5, "pUpGivenActive": .5, "upMeanSimple": .01,
                     "downMeanSimple": .01, "pooledActiveMeanSimple": .01}
        result = diagnostics(y, p, np.full(4, .01), np.full(4, .01), constants)
        self.assertAlmostEqual(result["allTickSignedMeanMseSkillVsZero"], .5)
        self.assertAlmostEqual(result["activeOnly"]["conditionalActiveMeanMseSkillVsZero"], 1.)
        self.assertAlmostEqual(result["activeOnly"]["deployedUnconditionalMeanMseSkillVsZero"], .75)
        self.assertAlmostEqual(result["zeroTickContributionToTotalMseBpsSquared"], 1250.)
        p[0] = [0., 1., 0.]
        undefined = diagnostics(y, p, np.full(4, .01), np.full(4, .01), constants)
        self.assertEqual(undefined["activeOnly"]["undefinedConditionalForecastOnRealizedActiveCount"], 1)
        self.assertIsNone(undefined["activeOnly"]["conditionalActiveMeanMseSkillVsZero"])


if __name__ == "__main__":
    unittest.main()
