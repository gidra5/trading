"""End-to-end worker lifecycle for the resident sampler (requires CUDA)."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from oracle_incremental_table import TableBudget
from serve_oracle_dense import DenseWorker,WorkerSettings,process_line
from test_serve_oracle_dense import flat_graph,observation,initialize
from test_oracle_dense_controller import Model,flat_prediction


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class IncrementalWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from numba import set_num_threads
        set_num_threads(2); torch.set_num_threads(2)

    def worker(self,path):
        settings = WorkerSettings(AccountGrid(max_orders=2,price_bins=1,size_bins=2,
            exposure_points=(-1.,0.,1.),wealth_points=(50.,100.,200.)),Rules(),
            horizon=6,width=2,branches=4,candidates=6,seed=23,backend='cuda',refresh_every=0,
            planner='incremental',table_budget=TableBudget(layers=2,rows=5,draws=2),root_draws=4)
        return DenseWorker(Model().cuda(),settings,graph_builder=flat_graph,predict=flat_prediction,
            state_directory=path,model_identity='a'*64)

    def start(self,w):
        request = initialize(); request['terminalSeconds'] = 14.
        return w.process(request)

    def tick(self,w,step):
        return w.process(dict(id=str(step),kind='tick',observation=observation(step)))

    def test_wrap_checkpoint_restart_and_terminal_close_match_uninterrupted(self):
        with tempfile.TemporaryDirectory() as path:
            w = self.worker(path); self.start(w)
            pointer = w.controller.current.values.data_ptr()
            for step in range(1,8): self.tick(w,step)
            w.process(dict(id='save',kind='checkpoint'))
            restored = self.worker(path); restored.process(dict(id='load',kind='restore'))
            for step in range(8,15):
                left = self.tick(w,step); right = self.tick(restored,step)
                self.assertEqual(left['account'],right['account'])
                self.assertEqual(left['terminalWealth'],right['terminalWealth'])
                self.assertEqual(w.controller.current.values.data_ptr(),pointer)
                np.testing.assert_array_equal(w.controller.current.cpu_values(),restored.controller.current.cpu_values())
                np.testing.assert_array_equal(w.controller.current.visits,restored.controller.current.visits)
            self.assertEqual(left['phase'],'complete')
            self.assertEqual(left['terminalWealth'],100.)
            self.assertEqual(left['account']['asset'],0.)
            self.assertEqual(left['account']['orders'],[])

    def test_invalid_input_is_rejected_and_failed_update_cannot_be_reapplied(self):
        with tempfile.TemporaryDirectory() as path:
            w = self.worker(path); self.start(w)
            with self.assertRaises(ValueError): self.tick(w,2)
            self.assertEqual(w.controller.update,0)
            self.assertEqual(w.controller.current.head,0)
            with patch.object(w.controller.current,'refresh_layer',side_effect=RuntimeError('kernel failed')):
                with self.assertRaisesRegex(RuntimeError,'kernel failed'): self.tick(w,1)
            self.assertTrue(w.failed)
            self.assertEqual(w.controller.phase,'failed')
            with self.assertRaises(RuntimeError): self.tick(w,1)
            with self.assertRaises(ValueError): w.process(dict(id='save',kind='checkpoint'))


if __name__=='__main__': unittest.main()
