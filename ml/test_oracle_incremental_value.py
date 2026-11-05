import unittest
import numpy as np
from numba import set_num_threads
from oracle_incremental_value import PlaneGraph, PlaneCache, update_planes


def solve(graph):
    values = [None]*len(graph.prices)
    values[-1] = graph.terminal()
    for h in range(len(graph.destinations)-1, -1, -1):
        values[h] = graph.backup(h, values[h+1], np.arange(len(graph.prices[h])))
    return PlaneCache(graph, tuple(values))


class IncrementalValueTests(unittest.TestCase):
    def setUp(self):
        set_num_threads(2)
        prices = (np.array([10000]),) + tuple(np.array([9990-h, 10010+h]) for h in range(5))
        ds = (np.array([[0, 1]], dtype=np.int32),) + tuple(
            np.array([[0, 0], [1, 1]], dtype=np.int32) for _ in range(4))
        self.graph = PlaneGraph(prices, ds, points=5)
        self.cache = solve(self.graph)

    def check_refresh(self, graph):
        before = tuple((v.c.copy(), v.d.copy()) for v in self.cache.values)
        got, stats = update_planes(self.cache, graph)
        want = solve(graph)
        for actual, expected in zip(got.values, want.values):
            np.testing.assert_array_equal(actual.c, expected.c)
            np.testing.assert_array_equal(actual.d, expected.d)
        for original, (c, d) in zip(self.cache.values, before):
            np.testing.assert_array_equal(original.c, c)
            np.testing.assert_array_equal(original.d, d)
        return got, stats

    def test_unchanged_graph_reuses_every_value(self):
        got, stats = self.check_refresh(self.graph)
        self.assertEqual(stats['recomputedRows'], 0)
        for a, b in zip(got.values, self.cache.values):
            self.assertIs(a, b)

    def test_changed_price_invalidates_incoming_and_outgoing_backups(self):
        prices = list(self.graph.prices)
        prices[3] = np.array([9982, 10012])
        got, stats = self.check_refresh(PlaneGraph(tuple(prices), self.graph.destinations, points=5))
        self.assertLess(stats['recomputedRows'], stats['totalRows'])
        self.assertIs(got.values[4], self.cache.values[4])

    def test_changed_transition_invalidates_ancestors_only(self):
        ds = list(self.graph.destinations)
        ds[3] = np.array([[0, 1], [1, 1]], dtype=np.int32)
        _, stats = self.check_refresh(PlaneGraph(self.graph.prices, tuple(ds), points=5))
        self.assertLess(stats['recomputedRows'], stats['totalRows'])

    def test_new_endpoint_propagates_without_regenerating_prefix(self):
        graph = PlaneGraph(self.graph.prices+(np.array([9980, 10020]),),
            self.graph.destinations+(np.array([[0, 0], [1, 1]], dtype=np.int32),), points=5)
        self.check_refresh(graph)

    def test_shorter_endpoint_is_not_confused_with_cached_continuation(self):
        self.check_refresh(PlaneGraph(self.graph.prices[:-1], self.graph.destinations[:-1], points=5))

    def test_changed_reachability_bound_invalidates_all_backups(self):
        _, stats = self.check_refresh(PlaneGraph(self.graph.prices, self.graph.destinations,
            points=5, gross_numerator=2100))
        self.assertEqual(stats['recomputedRows'], stats['totalRows'])


if __name__ == '__main__':
    unittest.main()
