import unittest
import numpy as np
from oracle_certified_value import ValuePlanes
from oracle_mc_error import log_envelope_errors, action_loss_bound_percent


class MonteCarloErrorTests(unittest.TestCase):
    def test_market_only_scale_has_zero_centered_error(self):
        c = np.array([[1000, 1010], [2000, 1999]], dtype=np.int64)
        d = np.array([[1010, 1000], [1999, 2000]], dtype=np.int64)
        old = ValuePlanes(c, d)
        new = ValuePlanes(c*np.array([[2], [3]]), d*np.array([[2], [3]]))
        absolute, centered = log_envelope_errors(old, new)
        np.testing.assert_allclose(absolute, np.log([2, 3]), atol=1e-14)
        np.testing.assert_allclose(centered, 0, atol=1e-14)

    def test_breakpoint_check_covers_a_dense_independent_grid(self):
        rng = np.random.default_rng(418)
        for _ in range(12):
            c0 = rng.integers(99000, 101000, (5, 9)); d0 = c0+rng.integers(-300, 301, c0.shape)
            c1 = rng.integers(99000, 101000, (5, 11)); d1 = c1+rng.integers(-300, 301, c1.shape)
            absolute, centered = log_envelope_errors(ValuePlanes(c0, d0), ValuePlanes(c1, d1))
            x = np.linspace(-3, 3, 10001)
            old = (c0[:, :, None]+(d0-c0)[:, :, None]*x).min(1)
            new = (c1[:, :, None]+(d1-c1)[:, :, None]*x).min(1)
            delta = np.log(new/old)
            shift = np.log(c1.min(1)/c0.min(1))[:, None]
            self.assertTrue(np.all(np.abs(delta).max(1) <= absolute+1e-13))
            self.assertTrue(np.all(np.abs(delta-shift).max(1) <= centered+1e-13))

    def test_ce_bound_covers_worst_possible_reversal(self):
        error = .0005
        actual_gap = 2*error-1e-12
        self.assertLess(100*(-np.expm1(-actual_gap)), action_loss_bound_percent(error))

    def test_nonpositive_envelope_fails_with_infinite_error(self):
        old = ValuePlanes(np.array([[100]]), np.array([[100]]))
        new = ValuePlanes(np.array([[100]]), np.array([[200]]))
        absolute, centered = log_envelope_errors(old, new)
        self.assertTrue(np.isinf(absolute[0]) and np.isinf(centered[0]))


if __name__ == '__main__':
    unittest.main()
