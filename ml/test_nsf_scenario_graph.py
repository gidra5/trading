import unittest
import numpy as np
import torch
from gpu_oracle_bellman import Bellman, Menu, Rules
from latent_neural_stochastic_flow import LatentNSF
from nsf_scenario_graph import build_graph, representatives
from benchmark_gpu_oracle import synthetic_tree


class ScenarioGraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_quantization_keeps_every_sample_and_bounds_width(self):
        x = torch.randn(40, 7, dtype=torch.float64)
        mass = torch.arange(1, 41, dtype=torch.float64); mass /= mass.sum()
        selected, mapping = representatives(x, mass, 4)
        self.assertLessEqual(len(selected), 4)
        self.assertEqual(len(mapping), 40)
        output = torch.zeros(len(selected), dtype=mass.dtype).index_add_(0, mapping, mass)
        self.assertTrue(bool((output > 0).all()))
        self.assertAlmostEqual(output.sum().item(), 1., places=12)

    def test_no_compression_is_identity_and_duplicates_are_safe(self):
        x = torch.randn(10, 3, dtype=torch.float64)
        selected, mapping = representatives(x, torch.ones(10, dtype=torch.float64)/10, 10)
        torch.testing.assert_close(selected, torch.arange(10))
        torch.testing.assert_close(mapping, torch.arange(10))
        selected, mapping = representatives(torch.zeros_like(x), torch.ones(10, dtype=torch.float64)/10, 4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(mapping.sum().item(), 0)
        selected, mapping = representatives(torch.zeros_like(x), torch.ones(10, dtype=torch.float64)/10, 20)
        self.assertEqual(len(selected), 1)
        self.assertEqual(mapping.sum().item(), 0)

    def test_graph_solver_matches_unrolled_graph(self):
        prices = [torch.tensor([[99., 101.]], dtype=torch.float64),
                  torch.tensor([[98., 102.], [102., 98.]], dtype=torch.float64),
                  torch.tensor([[97., 103.], [103., 97.]], dtype=torch.float64)]
        weights = [torch.full_like(p, .5) for p in prices]
        dest = [torch.tensor([[0, 1]]), torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 1], [1, 0]])]
        menus = [Menu(step=1.)]*3
        root = torch.tensor([[10000., 0., 100., 100., 0., 0., 0., 0., 0.]], dtype=torch.float64)
        graph = Bellman(prices, weights, menus, destinations=dest).solve(root)
        frontier = torch.zeros(1, dtype=torch.long)
        tree_prices, tree_weights = [], []
        for p, w, d in zip(prices, weights, dest):
            tree_prices.append(p[frontier]); tree_weights.append(w[frontier])
            frontier = d[frontier].reshape(-1)
        tree = Bellman(tree_prices, tree_weights, menus).solve(root)
        torch.testing.assert_close(graph['value'], tree['value'], atol=1e-9, rtol=0)
        torch.testing.assert_close(graph['state'], tree['state'], atol=1e-9, rtol=0)

    def test_conditional_rows_are_not_replaced_by_marginal_mass(self):
        root = torch.tensor([[10000., 0., 100., 100., 0., 0., 0., 0., 0.]], dtype=torch.float64)
        prices = [torch.tensor([[100., 100.]], dtype=torch.float64), torch.tensor([[90., 90.], [110., 110.]], dtype=torch.float64)]
        weights = [torch.full_like(p, .5) for p in prices]
        informed = Bellman(prices, weights, [Menu(step=1.)]*2, rules=Rules(maximum=50000), destinations=[torch.tensor([[0, 1]]), torch.tensor([[0, 0], [1, 1]])]).solve(root)
        pooled = Bellman([prices[0], torch.tensor([[90., 110.]], dtype=torch.float64)],
                         [weights[0], weights[1][:1]], [Menu(step=1.)]*2, rules=Rules(maximum=50000),
                         destinations=[torch.tensor([[0, 0]]), torch.tensor([[0, 1]])]).solve(root)
        self.assertGreater(informed['value'].item(), pooled['value'].item()+900)

    def test_nsf_graph_is_bounded_at_every_depth_and_preserves_row_mass(self):
        model = LatentNSF(torch.zeros(59), torch.ones(59), dimensions=4, width=8).eval()
        features = np.zeros((2, 59), dtype=np.float32)
        prices = np.array([100., 100.01]); times = np.array([1700000000., 1700000001.])
        graph = build_graph(model, features, prices, times, list(range(1, 9)), width=3, samples=4)
        self.assertEqual(len(graph.layers), 9)
        self.assertTrue(all(len(layer.prices) <= 3 for layer in graph.layers))
        for layer, weight in zip(graph.layers[1:], graph.weights):
            self.assertAlmostEqual(layer.mass.sum().item(), 1., places=12)
            torch.testing.assert_close(weight.sum(1), torch.ones(len(weight), dtype=torch.float64))
            np.testing.assert_allclose(layer.prices/.01, np.round(layer.prices/.01), atol=1e-8)

    def test_no_compression_matches_existing_nsf_tree(self):
        model = LatentNSF(torch.zeros(59), torch.ones(59), dimensions=4, width=8).eval()
        features = np.zeros((2, 59), dtype=np.float32)
        prices = np.array([100., 100.01]); times = np.array([1700000000., 1700000001.])
        graph = build_graph(model, features, prices, times, [1, 2, 3], width=8, samples=2, seed=23)
        _, (p, w) = synthetic_tree(model, features, prices, times, [1, 2, 3], [2, 2, 2], 23, 'cpu')
        for actual, expected in zip(graph.prices, p):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(graph.weights, w):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
