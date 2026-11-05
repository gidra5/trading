import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collections import OrderedDict
import numpy as np
import torch
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid,CellErrors
from oracle_book_mc import rule_vector
from oracle_incremental_table import IncrementalTable,TableBudget
from oracle_incremental_checkpoint import save_table,load_table,signatures
from test_oracle_book_gpu import market


class IncrementalCheckpointTests(unittest.TestCase):
    def table(self):
        t = IncrementalTable.__new__(IncrementalTable)
        t.source_identity = signatures()
        t.grid = AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        t.a = t.grid.arrays(); t.rules = rule_vector(Rules())
        t.cloud = market([[100.,100.]]*5); t.B = 4
        t.capacity,t.H,t.W,t.C,t.M = 6,4,2,3,2
        t.head,t.update,t.seed,t.candidates = 5,17,7,6
        t.budget = TableBudget(draws=2); t.failed = False
        t.errors = CellErrors.empty(t.W*t.C); t.errors.observe(2,.3); t.errors.observe(2,-.1)
        shape = (t.capacity,t.W,t.C,t.M)
        t.values = torch.arange(np.prod(shape),dtype=torch.float64).reshape(shape)
        t.visits = np.arange(np.prod(shape),dtype=np.uint32).reshape(shape)
        t.sampling_variance = np.ones(shape,dtype=np.float32)*.2
        t.controls = OrderedDict([((2,5),np.array([1.,2.,3.,4.])),((1,0),np.array([4.,3.,2.,1.]))])
        slots = t.C*2+4+1; t.root_controls = np.arange((slots+1)*t.B,dtype=float).reshape(slots+1,t.B)
        state = np.array([100.,0.,100.,100.]); states = np.tile(state,(slots,1))
        scores = np.zeros(slots); scores[0] = 1.
        front = dict(state=state,states=states,scores=scores,value=1.,choice=0)
        return t,front

    def load_cpu(self,path,context):
        # Serialization validation is CPU-only; GPU continuation equivalence is
        # exercised separately by the table tests.
        tensor = torch.tensor
        with patch('oracle_incremental_checkpoint.torch.tensor',side_effect=lambda x,**kw:tensor(x,dtype=kw['dtype'])):
            return load_table(path,context)

    def test_ring_statistics_and_lru_round_trip(self):
        t,front = self.table(); context = dict(model='sha256',terminal=1000)
        with tempfile.TemporaryDirectory() as path:
            save_table(t,front,context,path)
            restored,actual = self.load_cpu(path,context)
            self.assertEqual((restored.head,restored.H,restored.update),(5,4,17))
            self.assertEqual(list(restored.controls),list(t.controls))
            for key in t.controls: np.testing.assert_array_equal(restored.controls[key],t.controls[key])
            for key in front: np.testing.assert_array_equal(actual[key],front[key])
            for name in ('values','visits','sampling_variance','root_controls'):
                np.testing.assert_array_equal(getattr(restored,name),getattr(t,name))
            for name in ('count','mean','m2'):
                np.testing.assert_array_equal(getattr(restored.errors,name),getattr(t.errors,name))
            for depth in range(t.H): np.testing.assert_array_equal(restored.field(depth),t.field(depth))

    def test_failed_atomic_publish_preserves_previous_boundary(self):
        t,front = self.table()
        with tempfile.TemporaryDirectory() as path:
            save_table(t,front,{},path); pointer = (Path(path)/'current.json').read_bytes()
            t.update += 1
            with patch('oracle_incremental_checkpoint.os.replace',side_effect=OSError('disk failure')):
                with self.assertRaisesRegex(OSError,'disk failure'): save_table(t,front,{},path)
            self.assertEqual((Path(path)/'current.json').read_bytes(),pointer)
            restored,_ = self.load_cpu(path,{})
            self.assertEqual(restored.update,17)

    def test_context_and_corruption_are_rejected_before_gpu_allocation(self):
        t,front = self.table()
        with tempfile.TemporaryDirectory() as path:
            save_table(t,front,dict(model='a'),path)
            with patch('oracle_incremental_checkpoint.torch.tensor') as upload:
                with self.assertRaisesRegex(ValueError,'context'): load_table(path,dict(model='b'))
                upload.assert_not_called()
                pointer = json.loads((Path(path)/'current.json').read_text())
                cache = Path(path)/pointer['snapshot']/'controls.npz'
                with cache.open('ab') as stream: stream.write(b'corruption')
                with self.assertRaisesRegex(ValueError,'checksum'): load_table(path,dict(model='a'))
                upload.assert_not_called()

    def test_rotation_retains_current_and_previous_snapshot(self):
        t,front = self.table()
        with tempfile.TemporaryDirectory() as path:
            first = save_table(t,front,{},path)
            second = save_table(t,front,{},path)
            third = save_table(t,front,{},path)
            self.assertEqual(third['removedSnapshots'],[first['snapshot']])
            self.assertEqual({p.name for p in Path(path).iterdir() if p.is_dir()},
                {second['snapshot'],third['snapshot']})


if __name__=='__main__': unittest.main()
