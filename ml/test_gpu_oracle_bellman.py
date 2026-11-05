import unittest
import torch
from gpu_oracle_bellman import Bellman, TensorOracle, Rules, Menu, equity


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


class GPUOracleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.state = tensor([[10000, 0, 100, 100, 0, 0, 0, 0, 0]])

    def test_action_precedes_random_outcome(self):
        solver = Bellman([tensor([[90, 110]])], [tensor([[.5, .5]])], [Menu(step=.01)], rules=Rules(maker=0, taker=0))
        result = solver.solve(self.state)
        self.assertAlmostEqual(result['value'].item(), 10000., places=8)

    def test_feedback_uses_only_arrived_information(self):
        # Two distinguishable feature observations at the same intermediate
        # price predict opposite next returns. Root action is still shared.
        solver = Bellman([tensor([[100, 100]]), tensor([[90, 90], [110, 110]])],
                         [tensor([[.5, .5]]), tensor([[.5, .5], [.5, .5]])], [Menu(step=1.)]*2, rules=Rules(maximum=50000))
        self.assertGreater(solver.solve(self.state)['value'].item(), 10900.)

    def test_chunk_size_does_not_change_search(self):
        prices = [tensor([[99, 101]]), tensor([[100, 98], [100, 102]])]
        weights = [tensor([[.5, .5]]), tensor([[.5, .5], [.5, .5]])]
        a = Bellman(prices, weights, [Menu(step=.5)]*2, chunk=64).solve(self.state)
        b = Bellman(prices, weights, [Menu(step=.5)]*2, chunk=8192).solve(self.state)
        torch.testing.assert_close(a['value'], b['value'], atol=1e-10, rtol=0)
        torch.testing.assert_close(a['state'], b['state'], atol=1e-10, rtol=0)

    def test_liquidation_precedes_limit_fill_and_terminal_rejects_residual(self):
        oracle = TensorOracle()
        leveraged = tensor([[-10000, 200, 100, 100, 1, 90, 5, 0, 0]])
        moved = oracle.advance(leveraged, tensor([60]))
        self.assertEqual(equity(moved).item(), 0.)
        self.assertEqual(oracle.arrival(moved)[2].item(), False)
        holding = tensor([[-10000, 200, 100, 100, 0, 0, 0, 0, 0]])
        self.assertEqual(oracle.terminal(holding).item(), -float('inf'))

    def test_terminal_rejects_dust_and_arrival_traffic_and_solves_early_close(self):
        oracle = TensorOracle(Rules(maximum=100))
        for asset in [.01, -.01, 2, -2]:
            s = tensor([[1000, asset, 100, 100, 0, 0, 0, 0, 0]])
            self.assertEqual(oracle.terminal(s).item(), -float('inf'))
        s = tensor([[1000, 0, 95, 100, 1, 95, 95, 0, 1]])
        self.assertEqual(oracle.terminal(s).item(), -float('inf'))
        tiny = tensor([[1000, 5e-324, .01, .01, 0, 0, 0, 0, 0]])
        self.assertEqual(oracle.terminal(tiny).item(), -float('inf'))
        solver = Bellman([tensor([[100]])], [tensor([[1.]])], [Menu(step=.01)], rules=Rules(maximum=100))
        s = tensor([[1000, 2, 100, 100, 0, 0, 0, 0, 0]])
        solved = solver.solve(s)
        self.assertGreater(solved['amount'].item(), 0)
        self.assertTrue(torch.isfinite(oracle.terminal(solved['state'])).all())

    def test_next_tick_activation_and_committed_fill(self):
        oracle = TensorOracle()
        state = tensor([[10000, 0, 99, 100, 1, 99.5, 1000, 1, 1]])
        self.assertFalse(oracle.arrival(state)[2].item())
        state[:, 7] = 0
        arrived, consumed, filled, valid = oracle.arrival(state)
        self.assertTrue(filled.item() and valid.item())
        self.assertEqual(consumed.item(), 1000.)
        self.assertEqual(arrived[0, 0].item(), 9000.)

    def test_work_budget_rejects_incomplete_answer(self):
        solver = Bellman([tensor([[99, 101]])], [tensor([[.5, .5]])], [Menu(step=.01)], max_work=1)
        with self.assertRaisesRegex(RuntimeError, 'budget exceeded'):
            solver.solve(self.state)

    def test_binary_roundoff_cannot_change_tick_boundary_fill(self):
        oracle = TensorOracle()
        for p in [64892.01, 64891.99, 100.03]:
            below = round((p-.01)*100)/100
            state = tensor([[10000, 0, below, p, 1, p-.01, 1000, 0, 1]])
            self.assertTrue(oracle.arrival(state)[2].item())

    def test_exact_leverage_boundary_is_not_lost_to_float32_scalar(self):
        oracle = TensorOracle()
        state = tensor([[-9989.436928821595, .3078963033353798, 65138.17, 64888.32, 0, 0, 0, 0, 2]])
        desc = oracle.descriptors(Menu(step=1.), 'cpu')
        _, valid, _ = oracle.actions(state, desc)
        idx = ((desc[:, 0] == 2) & (desc[:, 1] == 0) & (desc[:, 2] == 0)).nonzero()[0, 0]
        self.assertTrue(valid[0, idx].item())

    def test_exact_close_does_not_create_dust_or_erase_capped_inventory(self):
        oracle=TensorOracle();descriptors=oracle.descriptors(Menu(step=.01),'cpu')
        idx=((descriptors[:,0]==0)&(descriptors[:,1]==0)&(descriptors[:,2]==0)).nonzero()[0,0]
        s=tensor([[10000,.017398743217,64891.99,64891.99,0,0,0,0,0],
            [10000,.4,64891.99,64891.99,0,0,0,0,0],
            [10000,.000001,64891.99,64891.99,0,0,0,0,0]])
        out,valid,_=oracle.actions(s,descriptors)
        self.assertTrue(valid[0,idx]);self.assertEqual(out[0,idx,1].item(),0)
        self.assertGreater(out[1,idx,1].item(),0)
        self.assertFalse(valid[2,idx])
        idle=torch.isnan(descriptors[:,0])&(descriptors[:,1]==0)&(descriptors[:,2]==0)
        self.assertEqual(out[2,idle,1].item(),s[2,1].item())


if __name__ == '__main__':
    unittest.main()
