"""Independent likelihood derivatives and causal online-calibration contracts."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import unittest

import mpmath as mp
import numpy as np
import torch

from evaluate_online_hurdle_calibration import CachedStream, simulate_stream
from audit_online_hurdle_calibration import audit_trace
from hurdle_sign_size_forecast import hurdle_loss_parts
from online_hurdle_calibration import (
    COEFFICIENT_BOUNDS, MAXIMUM_DOWN_MEAN, MINIMUM_MEAN, MODES,
    CalibrationConfig, OnlineHurdleCalibrator, beta_log_odds_fisher,
    likelihood_gradient_fisher, transform_raw,
)


class OnlineHurdleCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def raw(self):
        return np.array([.3, -.4, .002, .003, .7, 2.3], dtype=np.float64)

    def test_cold_start_and_disabled_calibration_are_exact_identity_without_mutation(self):
        raw = self.raw()
        expected = raw.copy()
        raw.setflags(write=False)
        for mode in MODES:
            config = CalibrationConfig(900, mode)
            calibrator = OnlineHurdleCalibrator(config)
            result = calibrator.apply(0, raw)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(raw, expected)
            self.assertFalse(np.shares_memory(result, raw))
            with self.assertRaises(FrozenInstanceError): config.mode = "full"
        disabled = OnlineHurdleCalibrator(CalibrationConfig(1, "none"))
        for i, y in enumerate((0., .002, -.003)):
            disabled.update(i*4000, raw, y)
        np.testing.assert_array_equal(disabled.apply(10000, raw), expected)
        np.testing.assert_array_equal(disabled.coefficients_at(10000), np.zeros(4))

    def test_analytic_gradient_and_fisher_match_torch_likelihood_autodiff(self):
        base = self.raw()
        theta = np.array([.2, -.1, .15, -.2])
        for mode in MODES:
            for y in (0., .0013, -.004):
                with self.subTest(mode=mode, y=y):
                    c = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
                    raw = torch.from_numpy(base)
                    packed = torch.stack((raw[0]+c[0], raw[1]+c[1], raw[2]*c[2].exp(),
                        (torch.logit(raw[3])+c[3]).sigmoid(), raw[4], raw[5])).reshape(1, 1, 6)
                    parts = hurdle_loss_parts(packed, torch.tensor([[y]], dtype=torch.float64))
                    keys = () if mode == "none" else ("zero",) if mode == "zero" else \
                        ("zero", "sign") if mode == "probabilities" else ("zero", "sign", "up", "down")
                    objective = c.sum()*0+sum(parts[k].sum() for k in keys)
                    gradient = torch.autograd.grad(objective, c, create_graph=True)[0]
                    observed = torch.stack([torch.autograd.grad(gradient[i], c, retain_graph=True)[0][i]
                                            for i in range(4)]) if mode != "none" else torch.zeros(4, dtype=torch.float64)
                    expected_fisher = observed+gradient*torch.tensor([0., 0., 1., 1.])
                    actual_g, actual_f = likelihood_gradient_fisher(base, theta, y, mode)
                    np.testing.assert_allclose(actual_g, gradient.detach().numpy(), rtol=2e-9, atol=2e-10)
                    np.testing.assert_allclose(actual_f, expected_fisher.detach().numpy(), rtol=2e-9, atol=2e-10)

    def test_beta_fisher_matches_high_precision_at_branch_thresholds_and_physical_bounds(self):
        with mp.workdps(80):
            for alpha in (.001, .1, 1., 2., 2.001, 3.5, 99.9, 100.):
                for beta in (1e-14, .1, 1000., 1000.01, 10000., 10000.01, 100000., 100000.01, 1e14):
                    with self.subTest(alpha=alpha, beta=beta):
                        a, b = mp.mpf(alpha), mp.mpf(beta)
                        score = float(b*(mp.digamma(b+a)-mp.digamma(b)))
                        fisher = float(b*b*(mp.polygamma(1, b)-mp.polygamma(1, b+a)))
                        actual_s, actual_f = beta_log_odds_fisher(alpha, beta)
                        self.assertAlmostEqual(actual_s, score, delta=3e-9)
                        self.assertAlmostEqual(actual_f, fisher, delta=3e-9)
                        self.assertGreater(actual_f, 0)

    def test_head_masks_and_exact_physical_boundary_derivative_convention(self):
        raw, theta = self.raw(), np.zeros(4)
        for mode, y, enabled in (("none", .01, ()), ("zero", -.01, (0,)),
                                 ("probabilities", .01, (0, 1)), ("full", 0., (0,)),
                                 ("full", .01, (0, 1, 2)), ("full", -.01, (0, 1, 3))):
            gradient, fisher = likelihood_gradient_fisher(raw, theta, y, mode)
            np.testing.assert_array_equal(np.flatnonzero(fisher), enabled)
            self.assertTrue(all(gradient[i] == 0 for i in range(4) if i not in enabled))
        for boundary in (MINIMUM_MEAN, MAXIMUM_DOWN_MEAN):
            edge = raw.copy()
            edge[3] = boundary
            gradient, fisher = likelihood_gradient_fisher(edge, theta, -.002, "full")
            self.assertTrue(np.isfinite(gradient).all())
            self.assertGreater(fisher[3], 0., "exact boundary uses the interior derivative")
            outward = theta.copy()
            outward[3] = -.2 if boundary == MINIMUM_MEAN else .2
            gradient, fisher = likelihood_gradient_fisher(edge, outward, -.002, "full")
            self.assertEqual(gradient[3], 0.)
            self.assertEqual(fisher[3], 0.)
            self.assertEqual(transform_raw(edge, outward)[3], boundary)
        edge = raw.copy()
        edge[2] = MINIMUM_MEAN
        outward = theta.copy()
        outward[2] = -.2
        gradient, fisher = likelihood_gradient_fisher(edge, outward, .002, "full")
        self.assertEqual(gradient[2], 0.)
        self.assertEqual(fisher[2], 0.)
        self.assertEqual(transform_raw(edge, outward)[2], MINIMUM_MEAN)

    def test_discounted_fisher_equation_uses_actual_maturity_time_and_overlap_weight(self):
        config = CalibrationConfig(8, "zero", half_life_seconds=8., ridge_precision=8., cadence_seconds=4)
        calibrator = OnlineHurdleCalibrator(config)
        raw = self.raw()
        raw[0] = 0.
        self.assertEqual(config.observation_weight, .5)
        calibrator.update(0, raw, 0.)
        a, b = .5*.25, .5*.5
        self.assertAlmostEqual(calibrator.coefficients_at(8000)[0], b/(a+8.))
        self.assertAlmostEqual(calibrator.coefficients_at(16000)[0], .5*b/(.5*a+8.))
        theta = .5*b/(.5*a+8.)
        probability = 1/(1+math.exp(-theta))
        next_a = .5*a+.5*probability*(1-probability)
        next_b = .5*b+.5*(probability*(1-probability)*theta-probability)
        calibrator.update(8000, raw, .01)
        self.assertAlmostEqual(calibrator.information[0], next_a)
        self.assertAlmostEqual(calibrator.linear[0], next_b)
        self.assertAlmostEqual(calibrator.coefficients_at(16000)[0], next_b/(next_a+8.))
        self.assertAlmostEqual(calibrator.coefficients_at(24000)[0], .5*next_b/(.5*next_a+8.))

    def test_query_frequency_and_chunking_do_not_change_event_anchored_state(self):
        config = CalibrationConfig(8, "full", half_life_seconds=16., ridge_precision=8.)
        sparse, dense = OnlineHurdleCalibrator(config), OnlineHurdleCalibrator(config)
        labels = (0., .001, -.004, .003, 0., -.002)
        raw = self.raw()
        for i, label in enumerate(labels):
            origin = i*4000
            sparse.update(origin, raw, label)
        for chunk in ((0, 1), (1, 4), (4, 6)):
            for i in range(*chunk):
                origin = i*4000
                dense.update(origin, raw, labels[i])
                state_a, state_b = dense.information.copy(), dense.linear.copy()
                for stamp in (origin+8000, origin+9000, origin+11999):
                    dense.apply(stamp, raw)
                    dense.snapshot(stamp)
                np.testing.assert_array_equal(dense.information, state_a)
                np.testing.assert_array_equal(dense.linear, state_b)
        self.assertEqual(sparse.snapshot(40000), dense.snapshot(40000))
        np.testing.assert_array_equal(sparse.apply(40000, raw), dense.apply(40000, raw))

    def test_duplicate_origins_past_queries_and_invalid_updates_are_rejected(self):
        calibrator = OnlineHurdleCalibrator(CalibrationConfig(1, "full"))
        raw = self.raw()
        calibrator.update(4000, raw, .003)
        before = calibrator.snapshot(5000)
        for origin in (4000, 3000):
            with self.assertRaises(ValueError): calibrator.update(origin, raw, .002)
        for stamp in (4999, 0):
            with self.assertRaises(ValueError): calibrator.apply(stamp, raw)
            with self.assertRaises(ValueError): calibrator.snapshot(stamp)
        for label in (-1., -1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError): calibrator.update(8000, raw, label)
        self.assertEqual(calibrator.snapshot(5000), before)
        self.assertTrue(np.all(np.abs(calibrator.coefficients_at(5000)) <= COEFFICIENT_BOUNDS))

    def stream(self, horizon=1):
        origins = np.arange(0, 24, 4, dtype=np.int64)
        raw = np.tile(self.raw(), (len(origins), 1))
        prices = 100+np.arange(int(origins[-1])+horizon+1)*.05
        prices[1] = 100.2
        return CachedStream(origins, 100000+origins*1000, raw, prices, 0,
                            np.arange(len(origins), dtype=np.int64), 100000, horizon)

    def test_real_pipeline_cold_start_exact_maturity_and_one_update_per_origin(self):
        for h in (1, 4, 8):
            with self.subTest(horizon=h):
                stream = self.stream(h)
                config = CalibrationConfig(h, "full", half_life_seconds=8., ridge_precision=8.)
                actual = simulate_stream(stream, config)
                manual = OnlineHurdleCalibrator(config)
                pending = 0
                expected = []
                for i, stamp in enumerate(stream.timestamps):
                    for old in range(pending, i):
                        if stream.timestamps[old]+h*1000 > stamp:
                            break
                        manual.update(int(stream.timestamps[old]), stream.raw[old], stream.target(old))
                        pending += 1
                    expected.append(manual.apply(int(stamp), stream.raw[i]))
                np.testing.assert_array_equal(actual["parameters"], expected)
                np.testing.assert_array_equal(actual["parameters"][0], stream.raw[0])
                self.assertEqual(actual["maturedUpdates"], len(stream.origins))
                self.assertEqual(actual["finalState"]["maturedObservations"], len(stream.origins))
                self.assertEqual(actual["finalState"]["latestMaturityUsed"], int(stream.timestamps[-1]+h*1000))
                if h == 8:
                    np.testing.assert_array_equal(actual["parameters"][1], stream.raw[1])
                else:
                    self.assertFalse(np.array_equal(actual["parameters"][1], stream.raw[1]))

    def test_real_pipeline_future_suffix_and_query_subset_do_not_change_past_outputs(self):
        stream = self.stream(8)
        config = CalibrationConfig(8, "full", half_life_seconds=8., ridge_precision=8.)
        raw_before, prices_before = stream.raw.copy(), stream.prices.copy()
        initial = simulate_stream(stream, config)
        query_subset = replace(stream, accuracy_indices=np.array([1, 4]))
        subset = simulate_stream(query_subset, config)
        np.testing.assert_array_equal(initial["parameters"], subset["parameters"])
        self.assertEqual(initial["finalState"], subset["finalState"])
        altered_prices, altered_raw = stream.prices.copy(), stream.raw.copy()
        altered_prices[13:] *= 1.3
        altered_raw[stream.origins > 12, :2] += 1.
        altered = simulate_stream(replace(stream, prices=altered_prices, raw=altered_raw), config)
        prefix = stream.origins <= 12
        np.testing.assert_array_equal(initial["parameters"][prefix], altered["parameters"][prefix])
        self.assertFalse(np.array_equal(initial["parameters"][~prefix], altered["parameters"][~prefix]))
        np.testing.assert_array_equal(stream.raw, raw_before)
        np.testing.assert_array_equal(stream.prices, prices_before)
        restarted = simulate_stream(stream, config)
        np.testing.assert_array_equal(initial["parameters"], restarted["parameters"])

    def test_independent_artifact_audit_rejects_future_maturity_and_mutated_shapes(self):
        stream = self.stream(8)
        actual = simulate_stream(stream, CalibrationConfig(8, "full"), trace=True)
        checked = audit_trace(stream.raw, actual["parameters"], stream.timestamps, actual["snapshots"], 8, "full")
        self.assertEqual(checked["coldStartIdentityQueries"], 2)
        future = {k:v.copy() for k,v in actual["snapshots"].items()}
        future["latestMaturityUsed"][-1] = stream.timestamps[-1]+1
        with self.assertRaises(AssertionError):
            audit_trace(stream.raw, actual["parameters"], stream.timestamps, future, 8, "full")
        altered = actual["parameters"].copy()
        altered[:, 4] *= 1.1
        with self.assertRaises(AssertionError):
            audit_trace(stream.raw, altered, stream.timestamps, actual["snapshots"], 8, "full")


if __name__ == "__main__":
    unittest.main()
