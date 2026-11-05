import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from oracle_dense_value_map import DenseValueMap
from oracle_dense_repair import repair_bound,copy_residuals
from test_oracle_book_gpu import market


class ResidualAccountingTests(unittest.TestCase):
    def test_old_seams_are_charged_and_repaired_tail_seams_removed(self):
        old=np.array([0.,0.,.2,.3,0.,.7,0.,0.])
        self.assertAlmostEqual(repair_bound(old,4,.1,.01),.65)
        np.testing.assert_allclose(copy_residuals(old,4,.1,.01),[0,.2,.3,.11,0,0,0,0])


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.grid=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        rules=Rules(maker=0,taker=0,maximum=100000);self.rules=rule_vector(rules)
        kernel=Path(__file__).with_name('oracle_book_mc.py')
        (self.root/'plan.json').write_text(json.dumps(dict(grid=asdict(self.grid),rules=asdict(rules),
            fundingMultiplier=1.,sourceHashes={'ml/oracle_book_mc.py':hashlib.sha256(kernel.read_bytes()).hexdigest()})))

    def tearDown(self):
        import gc
        gc.collect();self.temp.cleanup()

    def shifted(self,old,tail=100.):
        raw=old.sampled_prices.copy();raw[1:-1]=old.sampled_prices[2:];raw[0]=100.;raw[-1]=tail
        return replace(old,origin_seconds=old.origin_seconds+1,sampled_prices=raw)

    def test_flat_tail_reuses_prefix_with_exact_values_and_persists(self):
        c=market([[100.,100.]]*11)
        old=DenseValueMap(self.grid,c,self.rules,self.root/'old');old.sweep(7,3,4)
        current=DenseValueMap(self.grid,self.shifted(c),self.rules,self.root/'new')
        stats=current.sweep(7,3,4,previous=old,estimator='repair',repair_interval=1)
        reference=DenseValueMap(self.grid,current.cloud,self.rules,self.root/'reference');reference.sweep(7,3,4)
        self.assertGreater(stats['reusedLayers'],0)
        np.testing.assert_array_equal(current.values,reference.values)
        loaded=DenseValueMap.load(current.directory)
        np.testing.assert_array_equal(loaded.bellman_residuals,current.bellman_residuals)
        self.assertEqual(loaded.bellman_contract,current.bellman_contract)
        del old,current,reference,loaded

    def test_in_memory_repair_exports_and_reloads(self):
        old=DenseValueMap(self.grid,market([[100.,100.]]*9),self.rules,None);old.sweep(7,3,4)
        new=DenseValueMap(self.grid,self.shifted(old.cloud),self.rules,None)
        stats=new.sweep(7,3,4,previous=old,estimator='repair',repair_interval=1)
        self.assertGreater(stats['reusedLayers'],0)
        self.assertIsNone(new.directory)
        new.export(self.root/'exported')
        loaded=DenseValueMap.load(self.root/'exported')
        np.testing.assert_array_equal(loaded.values,new.values)
        np.testing.assert_array_equal(loaded.scheduled_accounts,new.scheduled_accounts)
        del old,new,loaded

    def test_fixed_terminal_date_shrinks_to_one_step_with_fresh_root(self):
        old=DenseValueMap(self.grid,market([[100.,100.]]*7),self.rules,None);old.sweep(7,3,4)
        for remaining in range(5,0,-1):
            c=old.cloud
            current_cloud=replace(c,prices=c.prices[1:],counts=c.counts[1:],masses=c.masses[1:],
                features=c.features[1:],price_pairs=c.price_pairs[1:],destinations=c.destinations[1:],
                sampled_prices=c.sampled_prices[1:],origin_seconds=c.origin_seconds+1)
            current=DenseValueMap(self.grid,current_cloud,self.rules,None)
            stats=current.sweep(7,3,4,previous=old,estimator='repair')
            reference=DenseValueMap(self.grid,current_cloud,self.rules,None);reference.sweep(7,3,4)
            self.assertEqual(stats['repairedLayers'],1)
            self.assertEqual(stats['rootErrorAllowance'],0.)
            np.testing.assert_array_equal(current.values,reference.values)
            del reference,old
            old=current
        del old,current

    def test_repeated_nonzero_repairs_stay_within_accounted_root_budget(self):
        old=DenseValueMap(self.grid,market([[100.,100.]]*11),self.rules,self.root/'old');old.sweep(7,3,4)
        reused=0;saw_seam=False
        for i in range(4):
            current=DenseValueMap(self.grid,self.shifted(old.cloud,100.1),self.rules,self.root/f'new-{i}')
            stats=current.sweep(7,3,4,previous=old,estimator='repair',repair_tolerance=.0025,repair_interval=1)
            reference=DenseValueMap(self.grid,current.cloud,self.rules,self.root/f'reference-{i}');reference.sweep(7,3,4)
            np.testing.assert_array_equal(np.isfinite(current.values),np.isfinite(reference.values))
            mask=np.isfinite(reference.values[0]);error=np.max(abs(current.values[0][mask]-reference.values[0][mask]))
            self.assertLessEqual(error,stats['rootErrorAllowance']+1.e-13)
            self.assertLessEqual(stats['rootErrorAllowance'],.0025)
            if stats['repairBoundary'] is not None:
                self.assertLessEqual(stats['repairBoundary'],current.shape[0]-3)
            reused+=stats['reusedLayers'];saw_seam|=bool(np.any(current.bellman_residuals>1.e-6))
            del reference,old
            old=current
        self.assertGreater(reused,0);self.assertTrue(saw_seam)
        del current,old

    def test_changed_kernel_or_schedule_cannot_reuse_prefix(self):
        c=market([[100.,100.]]*7)
        old=DenseValueMap(self.grid,c,self.rules,self.root/'old');old.sweep(7,3,4)
        raw=self.shifted(c).sampled_prices.copy();raw[1,0,0]=99.
        changed=replace(self.shifted(c),sampled_prices=raw)
        current=DenseValueMap(self.grid,changed,self.rules,self.root/'changed')
        with self.assertRaisesRegex(ValueError,'conditional rows'):
            current.sweep(7,3,4,previous=old,estimator='repair')
        another=DenseValueMap(self.grid,self.shifted(c),self.rules,self.root/'schedule')
        with self.assertRaisesRegex(ValueError,'schedules'):
            another.sweep(8,3,4,previous=old,estimator='repair')
        del old,current,another

    def test_subsampled_parent_is_not_a_zero_error_reference(self):
        c=market([[100.,100.]]*7)
        old=DenseValueMap(self.grid,c,self.rules,self.root/'old');old.sweep(7,3,2)
        current=DenseValueMap(self.grid,self.shifted(c),self.rules,self.root/'new')
        with self.assertRaisesRegex(ValueError,'residual accounting'):
            current.sweep(7,3,4,previous=old,estimator='repair')
        del old,current

    def test_new_root_can_add_previously_invalid_account_cells(self):
        grid=AccountGrid(max_orders=2,price_bins=1,size_bins=2,
            exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        c=market([[.01,.01]]*7)
        old=DenseValueMap(grid,c,self.rules,None);old.sweep(7,16,4)
        price=c.prices.copy();price[0]=100.
        pairs=c.price_pairs.copy();pairs[0]=100.
        raw=c.sampled_prices.copy();raw[0]=100.
        new_cloud=replace(c,prices=price,price_pairs=pairs,sampled_prices=raw,origin_seconds=1.)
        current=DenseValueMap(grid,new_cloud,self.rules,None)
        current.sweep(7,16,4,previous=old,estimator='repair',repair_interval=1)
        reference=DenseValueMap(grid,new_cloud,self.rules,None);reference.sweep(7,16,4)
        self.assertGreater(len(current.scheduled_accounts),len(old.scheduled_accounts))
        np.testing.assert_array_equal(current.values,reference.values)
        del old,current,reference

    def test_shrinking_horizon_retains_nonzero_seam_budget(self):
        base=DenseValueMap(self.grid,market([[100.,100.]]*11),self.rules,None);base.sweep(7,3,4)
        old=DenseValueMap(self.grid,self.shifted(base.cloud,100.1),self.rules,None)
        old.sweep(7,3,4,previous=base,estimator='repair',repair_interval=1,repair_tolerance=.0025)
        c=old.cloud
        small=replace(c,prices=c.prices[1:],counts=c.counts[1:],masses=c.masses[1:],features=c.features[1:],
            price_pairs=c.price_pairs[1:],destinations=c.destinations[1:],sampled_prices=c.sampled_prices[1:],origin_seconds=2.)
        current=DenseValueMap(self.grid,small,self.rules,None)
        stats=current.sweep(7,3,4,previous=old,estimator='repair',repair_tolerance=.0025)
        reference=DenseValueMap(self.grid,small,self.rules,None);reference.sweep(7,3,4)
        self.assertGreater(stats['rootErrorAllowance'],0.)
        mask=np.isfinite(reference.values[0]);error=np.max(abs(current.values[0][mask]-reference.values[0][mask]))
        self.assertLessEqual(error,stats['rootErrorAllowance'])
        del base,old,current,reference


if __name__=='__main__':unittest.main()
