import unittest
import numpy as np
import torch
from nsf_scenario_graph import nested_representatives, build_graph
from latent_neural_stochastic_flow import LatentNSF


class NestedGraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(53)

    def test_nested_partitions_preserve_fallback_and_reduce_each_sample_error(self):
        x = torch.randn(100, 7, dtype=torch.float64)
        mass = torch.rand(100, dtype=torch.float64); mass /= mass.sum()
        p = nested_representatives(x, mass, 20, base_width=4)
        torch.testing.assert_close(p.selected[:len(p.coarse_selected)], p.coarse_selected)
        torch.testing.assert_close(p.fine_to_coarse[p.assignment], p.coarse_assignment)
        old = (x-x[p.coarse_selected][p.coarse_assignment]).square().sum(1)
        new = (x-x[p.selected][p.assignment]).square().sum(1)
        self.assertTrue(bool((new <= old+1e-12).all()))
        self.assertLess((mass*new).sum().item(), (mass*old).sum().item())
        self.assertEqual(len(p.selected), 20)

    def test_more_nodes_extend_the_same_refinement(self):
        x = torch.randn(100, 7, dtype=torch.float64)
        mass = torch.ones(100, dtype=torch.float64)/100
        a = nested_representatives(x, mass, 10, base_width=4)
        b = nested_representatives(x, mass, 20, base_width=4)
        torch.testing.assert_close(a.selected, b.selected[:10])
        ea = (x-x[a.selected][a.assignment]).square().sum(1)
        eb = (x-x[b.selected][b.assignment]).square().sum(1)
        self.assertTrue(bool((eb <= ea+1e-12).all()))

    def test_duplicates_terminate_without_empty_or_hidden_nodes(self):
        x = torch.zeros(50, 3, dtype=torch.float64)
        p = nested_representatives(x, torch.ones(50, dtype=torch.float64)/50, 20, base_width=4)
        self.assertEqual(len(p.selected), 1)
        self.assertEqual(p.assignment.sum().item(), 0)

    def test_metric_guarantee_does_not_imply_price_coordinate_guarantee(self):
        x = torch.tensor([[0., 0.], [10., 1.], [10., 4.]], dtype=torch.float64)
        mass = torch.tensor([1000., 1., 2.], dtype=torch.float64)/1003
        p = nested_representatives(x, mass, 2, base_width=1)
        old = x[p.coarse_selected][p.coarse_assignment]
        new = x[p.selected][p.assignment]
        self.assertTrue(bool(((x-new).square().sum(1) <= (x-old).square().sum(1)).all()))
        self.assertGreater(abs(float(x[1, 1]-new[1, 1])), abs(float(x[1, 1]-old[1, 1])))

    def test_graph_exposes_recoverable_coarse_projection(self):
        model = LatentNSF(torch.zeros(59), torch.ones(59), dimensions=4, width=8).eval()
        graph = build_graph(model, np.zeros((2, 59), np.float32), np.array([100., 100.01]),
            np.array([1700000000., 1700000001.]), [1, 2, 3], width=8, samples=4,
            refinement='nested', base_width=2)
        for d, partition in enumerate(graph.refinements):
            self.assertIsNotNone(partition)
            torch.testing.assert_close(partition.fine_to_coarse[graph.destinations[d]],
                                       partition.coarse_assignment.reshape(graph.destinations[d].shape))
            self.assertTrue(graph.diagnostics[d]['exactCoarseRecovery'])
            self.assertLessEqual(graph.diagnostics[d]['maximumPointwiseMetricIncrease'], 1e-10)


if __name__ == '__main__':
    unittest.main()
