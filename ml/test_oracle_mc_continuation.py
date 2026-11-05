import unittest
import numpy as np
from oracle_incremental_value import PlaneGraph, PlaneCache
from oracle_mc_continuation import MonteCarloPlan, sampled_destinations, estimate_extension
from test_oracle_incremental_value import solve
from numba import set_num_threads


class MonteCarloContinuationTests(unittest.TestCase):
    def setUp(self):
        set_num_threads(2)
        p = (np.array([10000]),)+(np.array([9990, 10000, 10010]),)*4
        ds = (np.array([[0, 1, 1, 2]], np.int32),)+(np.array(
            [[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 2]], np.int32),)*3
        self.graph = PlaneGraph(p, ds, points=5)
        self.cache = solve(self.graph)
        self.extended = PlaneGraph(p+(np.array([9985, 10000, 10015]),), ds+(ds[-1],), points=5)

    def test_census_recovers_full_update_at_every_depth(self):
        reference = solve(self.extended)
        observed = {}
        _, report = estimate_extension(self.cache, self.extended, MonteCarloPlan(4, 314),
            on_layer=lambda h, value, _: observed.update({h: value}))
        self.assertTrue(report['census'])
        for h, got in observed.items():
            np.testing.assert_array_equal(got.c, reference.values[h].c)
            np.testing.assert_array_equal(got.d, reference.values[h].d)

    def test_samples_are_nested_and_census_keeps_exact_original_order(self):
        ds = np.tile(np.arange(64), (5, 1))
        small = sampled_destinations(ds, 4, 15, 3)
        large = sampled_destinations(ds, 16, 15, 3)
        np.testing.assert_array_equal(small, large[:, :4])
        self.assertIs(sampled_destinations(ds, 64, 15, 3), ds)
        self.assertTrue(all(len(set(row)) == 16 for row in large))

    def test_paired_correction_cancels_identical_child_inputs_exactly(self):
        # If an appended exact tail happens to equal the old terminal, every
        # paired correction is zero regardless of sampled branch count.
        from oracle_mc_continuation import correction, sampled_backup
        ds = sampled_destinations(self.graph.destinations[1], 2, 19, 1)
        value = sampled_backup(self.graph, 1, self.cache.values[2], ds)
        got = correction(self.cache.values[1], value, value)
        np.testing.assert_array_equal(got.c, self.cache.values[1].c)
        np.testing.assert_array_equal(got.d, self.cache.values[1].d)

    def test_deterministic_and_does_not_mutate_the_cache(self):
        before = tuple((v.c.copy(), v.d.copy()) for v in self.cache.values)
        a, _ = estimate_extension(self.cache, self.extended, MonteCarloPlan(2, 51), paired=True)
        b, _ = estimate_extension(self.cache, self.extended, MonteCarloPlan(2, 51), paired=True)
        np.testing.assert_array_equal(a.c, b.c); np.testing.assert_array_equal(a.d, b.d)
        for value, (c, d) in zip(self.cache.values, before):
            np.testing.assert_array_equal(value.c, c); np.testing.assert_array_equal(value.d, d)

    def test_changed_law_is_not_silently_accepted_as_same_baseline(self):
        changed = PlaneGraph((self.graph.prices[0].copy(),)+self.extended.prices[1:],
            self.extended.destinations, points=5)
        with self.assertRaisesRegex(ValueError, 'unchanged finite prefix'):
            estimate_extension(self.cache, changed, MonteCarloPlan(2, 0))

    def test_face_switch_does_not_change_the_scalar_cash_control_variate(self):
        from oracle_mc_continuation import correction
        from oracle_certified_value import ValuePlanes
        old = ValuePlanes(np.array([[1000, 1100]]), np.array([[1000, 1100]]))
        old_sample = ValuePlanes(np.array([[800, 1200]]), np.array([[800, 1200]]))
        new_sample = ValuePlanes(np.array([[1100, 1000]]), np.array([[1100, 1000]]))
        result = correction(old, new_sample, old_sample)
        self.assertEqual(result.c.min(), 1000+1000-800)

    def test_default_equals_the_full_solution_of_its_empirical_law(self):
        seed = 53; samples = 2; boundary = len(self.graph.destinations)
        destinations = tuple(sampled_destinations(ds, samples, seed, h) if h < boundary else ds
            for h, ds in enumerate(self.extended.destinations))
        empirical = PlaneGraph(self.extended.prices, destinations, points=5)
        want = solve(empirical)
        observed = {}
        _, report = estimate_extension(self.cache, self.extended, MonteCarloPlan(samples, seed),
            on_layer=lambda h, value, _: observed.update({h: value}))
        self.assertFalse(report['paired'])
        for h, value in observed.items():
            np.testing.assert_array_equal(value.c, want.values[h].c)
            np.testing.assert_array_equal(value.d, want.values[h].d)


if __name__ == '__main__':
    unittest.main()
