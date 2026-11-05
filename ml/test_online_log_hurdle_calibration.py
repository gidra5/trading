"""Independent signed-Gamma likelihood derivatives and causal calibration checks."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma
import torch
from torch.nn import functional as F

from evaluate_online_hurdle_calibration import CachedStream
from evaluate_log_hurdle_forecast import simulate_log_stream
from online_log_hurdle_calibration import (COEFFICIENT_BOUNDS, MAXIMUM_UP_MEAN_RATIO,
    MINIMUM_MEAN, MODES, CalibrationConfig, OnlineLogHurdleCalibrator,
    likelihood_gradient_fisher, transform_raw, validate_raw)


class OnlineLogHurdleCalibrationTest(unittest.TestCase):
    def raw(self):
        return np.array([.3,-.4,.2,.3,.7,2.3],dtype=np.float64)

    def test_cold_start_disabled_modes_and_frozen_input_identity(self):
        raw = self.raw()
        expected = raw.copy()
        raw.setflags(write=False)
        for mode in MODES:
            config = CalibrationConfig(900,mode)
            calibrated = OnlineLogHurdleCalibrator(config).apply(0,raw)
            np.testing.assert_array_equal(calibrated,expected)
            self.assertFalse(np.shares_memory(raw,calibrated))
            with self.assertRaises(FrozenInstanceError): config.mode = "none"
        disabled = OnlineLogHurdleCalibrator(CalibrationConfig(1,"none"))
        for i,y in enumerate((0.,.2,-3.)):
            disabled.update(i*4000,raw,y)
        np.testing.assert_array_equal(disabled.apply(10000,raw),expected)
        np.testing.assert_array_equal(disabled.coefficients_at(10000),np.zeros(4))
        np.testing.assert_array_equal(raw,expected)
        shifted = transform_raw(raw,np.array([.2,-.1,0.,0.]))
        np.testing.assert_array_equal(shifted[2:],raw[2:])

    def test_gradient_matches_independent_torch_gamma_likelihood(self):
        base = self.raw()
        theta = np.array([.2,-.1,.15,-.2])
        for mode in MODES:
            for realized in (0.,.13,-.4,-2.7):
                with self.subTest(mode=mode,realized=realized):
                    c = torch.tensor(theta,dtype=torch.float64,requires_grad=True)
                    raw = torch.from_numpy(base)
                    p_up_mean = raw[4]*torch.sigmoid(torch.logit(raw[2]/raw[4])+c[2])
                    p_down_mean = raw[3]*torch.exp(c[3])
                    objective = c.sum()*0
                    if mode != "none":
                        objective = objective+F.binary_cross_entropy_with_logits(raw[0]+c[0],c.new_tensor(float(realized==0)))
                    if mode in ("probabilities","full") and realized != 0:
                        objective = objective+F.binary_cross_entropy_with_logits(raw[1]+c[1],c.new_tensor(float(realized>0)))
                    if mode == "full" and realized != 0:
                        alpha,mean = (raw[4],p_up_mean) if realized > 0 else (raw[5],p_down_mean)
                        objective = objective-torch.distributions.Gamma(alpha,alpha/mean).log_prob(c.new_tensor(abs(realized)))
                    gradient = torch.autograd.grad(objective,c)[0]
                    actual,_ = likelihood_gradient_fisher(base,theta,realized,mode)
                    np.testing.assert_allclose(actual,gradient.detach().numpy(),rtol=3e-13,atol=3e-13)

    def test_expected_fisher_matches_integrated_score_variance(self):
        raw,theta = self.raw(),np.array([.2,-.1,.15,-.2])
        adjusted = transform_raw(raw,theta)
        for component,sign in ((2,1),(3,-1)):
            alpha,mean = adjusted[component+2],adjusted[component]
            _,information = likelihood_gradient_fisher(raw,theta,sign*mean,"full")
            # Integrate against a unit-scale Gamma, changing the realized log
            # magnitude to preserve the model's actual scale. This checks the
            # expected score variance rather than an observed Hessian identity.
            def integrand(unit):
                score,_ = likelihood_gradient_fisher(raw,theta,sign*unit*mean/alpha,"full")
                return score[component]**2*gamma.pdf(unit,a=alpha)
            expected,error = quad(integrand,0,np.inf,epsabs=1e-10,epsrel=1e-10)
            self.assertLess(error,1e-8)
            self.assertAlmostEqual(information[component],expected,delta=2e-9)

    def test_head_masks_and_gamma_fisher_chain_factor(self):
        raw,theta = self.raw(),np.zeros(4)
        for mode,y,enabled in (("none",.1,()),("zero",-.1,(0,)),("probabilities",.1,(0,1)),
                               ("full",0.,(0,)),("full",.1,(0,1,2)),("full",-3.,(0,1,3))):
            gradient,information = likelihood_gradient_fisher(raw,theta,y,mode)
            np.testing.assert_array_equal(np.flatnonzero(information),enabled)
            self.assertTrue(all(gradient[i]==0 for i in range(4) if i not in enabled))
        _,up = likelihood_gradient_fisher(raw,theta,.1,"full")
        _,down = likelihood_gradient_fisher(raw,theta,-3.,"full")
        self.assertAlmostEqual(up[2],raw[4]*(1-raw[2]/raw[4])**2)
        self.assertAlmostEqual(down[3],raw[5])
        self.assertLess(up[2],raw[4],"UP constrained-mean Fisher differs from ordinary log-mean Fisher")

    def test_exact_mean_boundaries_use_interior_derivative_and_outward_clamps(self):
        raw,theta = self.raw(),np.zeros(4)
        for boundary in (MINIMUM_MEAN,MAXIMUM_UP_MEAN_RATIO*raw[4]):
            edge = raw.copy()
            edge[2] = boundary
            validate_raw(edge)
            gradient,information = likelihood_gradient_fisher(edge,theta,.002,"full")
            self.assertTrue(np.isfinite(gradient).all())
            self.assertGreater(information[2],0)
            outward = theta.copy()
            outward[2] = -.2 if boundary == MINIMUM_MEAN else .2
            gradient,information = likelihood_gradient_fisher(edge,outward,.002,"full")
            self.assertEqual(gradient[2],0)
            self.assertEqual(information[2],0)
            self.assertEqual(transform_raw(edge,outward)[2],boundary)
        edge = raw.copy()
        edge[3] = MINIMUM_MEAN
        _,information = likelihood_gradient_fisher(edge,theta,-.002,"full")
        self.assertGreater(information[3],0)
        outward = theta.copy()
        outward[3] = -.2
        gradient,information = likelihood_gradient_fisher(edge,outward,-.002,"full")
        self.assertEqual(gradient[3],0)
        self.assertEqual(information[3],0)
        self.assertEqual(transform_raw(edge,outward)[3],MINIMUM_MEAN)

    def test_up_first_and_second_exponential_moments_stay_finite(self):
        for alpha in (.001,.01,1.,100.):
            for fraction in (.0001,.1,MAXIMUM_UP_MEAN_RATIO):
                raw = self.raw()
                raw[4],raw[2] = alpha,alpha*fraction
                for coefficient in (-math.log(2),0.,math.log(2)):
                    theta = np.array([0.,0.,coefficient,math.log(2)])
                    adjusted = transform_raw(raw,theta)
                    validate_raw(adjusted)
                    for order in (1,2):
                        value = math.exp(-alpha*math.log1p(-order*adjusted[2]/alpha))
                        self.assertTrue(math.isfinite(value))
                        self.assertGreaterEqual(value,1.)
        raw = self.raw()
        raw[2] = np.nextafter(MAXIMUM_UP_MEAN_RATIO*raw[4],math.inf)
        with self.assertRaises(ValueError): validate_raw(raw)

    def test_log_returns_below_minus_one_and_unbounded_down_means_are_valid(self):
        raw = self.raw()
        raw[3] = 4.
        calibrated = OnlineLogHurdleCalibrator(CalibrationConfig(1,"full"))
        for i,y in enumerate((-1.,-2.7,-20.,5.)):
            calibrated.update(i*4000,raw,y)
        result = calibrated.apply(14000,raw)
        self.assertGreater(result[3],1.)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(calibrated.snapshot(14000)["maturedObservations"],4)

    def test_updates_use_actual_maturity_wall_decay_and_overlap_weight(self):
        config = CalibrationConfig(8,"zero",half_life_seconds=8.,ridge_precision=8.,cadence_seconds=4)
        calibrated = OnlineLogHurdleCalibrator(config)
        raw = self.raw()
        raw[0] = 0.
        calibrated.update(0,raw,0.)
        information,linear = .5*.25,.5*.5
        self.assertEqual(config.observation_weight,.5)
        self.assertAlmostEqual(calibrated.coefficients_at(8000)[0],linear/(information+8))
        self.assertAlmostEqual(calibrated.coefficients_at(16000)[0],.5*linear/(.5*information+8))
        theta = .5*linear/(.5*information+8)
        probability = 1/(1+math.exp(-theta))
        calibrated.update(8000,raw,-2.)
        expected_information = .5*information+.5*probability*(1-probability)
        expected_linear = .5*linear+.5*(probability*(1-probability)*theta-probability)
        self.assertAlmostEqual(calibrated.information[0],expected_information)
        self.assertAlmostEqual(calibrated.linear[0],expected_linear)
        self.assertEqual(calibrated.snapshot(16000)["latestMaturityUsed"],16000)

    def test_queries_do_not_change_state_or_subsequent_forecasts(self):
        config = CalibrationConfig(8,"full",half_life_seconds=16.,ridge_precision=8.)
        sparse,dense = OnlineLogHurdleCalibrator(config),OnlineLogHurdleCalibrator(config)
        raw = self.raw()
        for i,y in enumerate((0.,.1,-3.,.4,0.,-.2)):
            origin = i*4000
            sparse.update(origin,raw,y)
            dense.update(origin,raw,y)
            before = dense.snapshot(origin+8000)
            for stamp in (origin+8000,origin+9000,origin+11999):
                dense.apply(stamp,raw)
                dense.snapshot(stamp)
            self.assertEqual(dense.snapshot(origin+8000),before)
        self.assertEqual(sparse.snapshot(40000),dense.snapshot(40000))
        np.testing.assert_array_equal(sparse.apply(40000,raw),dense.apply(40000,raw))

    def test_invalid_and_duplicate_updates_are_atomic_and_past_queries_rejected(self):
        calibrated = OnlineLogHurdleCalibrator(CalibrationConfig(1,"full"))
        raw = self.raw()
        calibrated.update(4000,raw,-2.)
        before = calibrated.snapshot(5000)
        for origin in (4000,3000):
            with self.assertRaises(ValueError): calibrated.update(origin,raw,.2)
        for stamp in (4999,0):
            with self.assertRaises(ValueError): calibrated.apply(stamp,raw)
            with self.assertRaises(ValueError): calibrated.snapshot(stamp)
        for y in (float("nan"),float("inf"),float("-inf")):
            with self.assertRaises(ValueError): calibrated.update(8000,raw,y)
        invalid = raw.copy()
        invalid[2] = raw[4]
        with self.assertRaises(ValueError): calibrated.update(8000,invalid,.2)
        self.assertEqual(calibrated.snapshot(5000),before)
        self.assertTrue(np.all(np.abs(calibrated.coefficients_at(5000))<=COEFFICIENT_BOUNDS))

    def stream(self,horizon=1):
        origins = np.arange(0,24,4,dtype=np.int64)
        raw = np.tile(self.raw(),(len(origins),1))
        prices = 100+np.arange(int(origins[-1])+horizon+1)*.05
        prices[horizon] = 5.  # A valid endpoint log return below -1.
        return CachedStream(origins,100000+origins*1000,raw,prices,0,
            np.arange(len(origins),dtype=np.int64),100000,horizon)

    def test_real_pipeline_uses_log_labels_and_exact_maturity_once_per_origin(self):
        for h in (1,4,8):
            with self.subTest(horizon=h):
                stream = self.stream(h)
                config = CalibrationConfig(h,"full",half_life_seconds=8.,ridge_precision=8.)
                actual = simulate_log_stream(stream,config,trace=True)
                manual = OnlineLogHurdleCalibrator(config)
                pending,expected = 0,[]
                for index,stamp in enumerate(stream.timestamps):
                    while pending < index and stream.timestamps[pending]+h*1000 <= stamp:
                        label = float(np.log1p(stream.target(pending)))
                        manual.update(int(stream.timestamps[pending]),stream.raw[pending],label)
                        pending += 1
                    expected.append(manual.apply(int(stamp),stream.raw[index]))
                    self.assertEqual(actual["snapshots"]["maturedObservations"][index],pending)
                    expected_maturity = -1 if pending==0 else int(stream.timestamps[pending-1]+h*1000)
                    self.assertEqual(actual["snapshots"]["latestMaturityUsed"][index],expected_maturity)
                np.testing.assert_array_equal(actual["parameters"],expected)
                np.testing.assert_array_equal(actual["parameters"][0],stream.raw[0])
                np.testing.assert_array_equal(actual["parameters"][:,4:],stream.raw[:,4:])
                self.assertLess(np.log1p(stream.target(0)),-1.)
                self.assertEqual(actual["maturedUpdates"],len(stream.origins))
                self.assertEqual(actual["finalState"]["latestMaturityUsed"],int(stream.timestamps[-1]+h*1000))

    def test_real_pipeline_future_suffix_and_query_subset_cannot_change_prior_outputs(self):
        stream = self.stream(8)
        config = CalibrationConfig(8,"full",half_life_seconds=8.,ridge_precision=8.)
        raw_before,prices_before = stream.raw.copy(),stream.prices.copy()
        initial = simulate_log_stream(stream,config)
        subset = simulate_log_stream(replace(stream,accuracy_indices=np.array([1,4])),config)
        np.testing.assert_array_equal(initial["parameters"],subset["parameters"])
        self.assertEqual(initial["finalState"],subset["finalState"])
        changed_prices,changed_raw = stream.prices.copy(),stream.raw.copy()
        changed_prices[13:] *= .7
        changed_raw[stream.origins>12,:2] += 1.
        changed = simulate_log_stream(replace(stream,prices=changed_prices,raw=changed_raw),config)
        prefix = stream.origins<=12
        np.testing.assert_array_equal(initial["parameters"][prefix],changed["parameters"][prefix])
        self.assertFalse(np.array_equal(initial["parameters"][~prefix],changed["parameters"][~prefix]))
        np.testing.assert_array_equal(stream.raw,raw_before)
        np.testing.assert_array_equal(stream.prices,prices_before)


if __name__ == "__main__":
    unittest.main()
