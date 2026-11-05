import itertools
import unittest
import numpy as np
from oracle_guided_sampling import SamplePlan, proposals, sample_plan, corrected_estimate


class GuidedSamplingTests(unittest.TestCase):
    def test_exhaustive_weighted_draws_recover_target_mean(self):
        old = np.array([[-7., 2., 9.]])
        current = np.array([[-9., 3., 14.]])
        q = proposals(old)[0]
        expectation = 0.
        for selected in itertools.product(range(3), repeat=2):
            indices = np.array([[*selected, 0]])
            weights = np.array([[*(1/(3*q[list(selected)])), 1.]])
            plan = SamplePlan(indices, np.array([2]), weights, q[None])
            value, _, _ = corrected_estimate(old, current[:,indices[0]], plan)
            expectation += q[selected[0]]*q[selected[1]]*value[0]
        self.assertAlmostEqual(expectation,current.mean(),places=13)

    def test_both_tails_receive_priority_with_positive_coverage(self):
        q = proposals(np.array([[-10., 0., 0., 10.]]),.2)[0]
        self.assertEqual(q[0],q[-1])
        self.assertGreater(q[0],q[1])
        self.assertTrue(np.all(q >= .2/4))
        self.assertAlmostEqual(q.sum(),1.)

    def test_unknown_support_and_nonfinite_controls_force_census(self):
        old = np.array([[1., 2., 3.], [1., -np.inf, 3.], [1., 2., 3.]])
        plan = sample_plan(old,[False,True,True],2,31)
        np.testing.assert_array_equal(plan.counts,[3,3,2])
        current = np.array([[2., 4., 6.], [3., 5., 7.], [2., 3., 4.]])
        draws = np.take_along_axis(current,plan.indices,1)
        values,variance,cache = corrected_estimate(old,draws,plan)
        np.testing.assert_array_equal(values[:2],[4.,5.])
        np.testing.assert_array_equal(variance[:2],[0.,0.])
        np.testing.assert_array_equal(cache[:2],current[:2])
        self.assertTrue(np.isneginf(old[1,1]))

    def test_uniform_common_shift_is_exact_and_cache_does_not_alias(self):
        old = np.array([[-100., -1., 20., 1000.]])
        plan = sample_plan(old,[True],2,7,exploration=1.)
        current = old+.25
        values,variance,cache = corrected_estimate(old,np.take_along_axis(current,plan.indices,1),plan)
        self.assertEqual(values[0],current.mean())
        self.assertEqual(variance[0],0.)
        self.assertFalse(np.shares_memory(old,cache))


if __name__ == '__main__': unittest.main()
