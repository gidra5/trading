"""Gradient, frozen-state and causal replay tests for joint calibration."""
import unittest
from copy import deepcopy
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from online_joint_hurdle_calibration import (JointCalibrationConfig,JointHurdleAdapter,
    OnlineJointHurdleCalibrator,matured_window,replay_joint)
from online_log_hurdle_calibration import transform_raw


class ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.25))
    def forward(self,x):
        z=x.mean((1,2)).double()*self.weight
        one=torch.ones_like(z)
        return torch.stack((z,-z,.01*torch.exp(z),.02*torch.exp(-z),one,2*one),dim=-1)[:,None]


class JointCalibrationTest(unittest.TestCase):
    def adapter(self,enabled=True):
        return JointHurdleAdapter(ToyPredictor(),torch.zeros(59),torch.ones(59),enabled)

    def test_identity_and_output_transform_matches_existing_law(self):
        a=self.adapter();x=torch.arange(354,dtype=torch.float32).reshape(3,2,59)/354
        torch.testing.assert_close(a(x),a.base(x),rtol=0,atol=0)
        theta=np.array([.2,-.1,.15,-.2])
        with torch.no_grad(): a.output_offset.copy_(torch.from_numpy(theta))
        raw=a.base(x).detach().numpy()[:,0]
        expected=np.stack([transform_raw(r,theta) for r in raw])
        np.testing.assert_allclose(a(x).detach().numpy()[:,0],expected,rtol=2e-15,atol=1e-15)

    def test_one_loss_updates_both_ends_but_never_frozen_predictor(self):
        a=self.adapter();c=OnlineJointHurdleCalibrator(a,JointCalibrationConfig(1))
        before={k:v.clone() for k,v in a.base.state_dict().items()}
        e=c.update(10000,torch.ones(3,2,59),torch.tensor([[.05],[-.03],[0.]],dtype=torch.float64),
            np.array([0,4000,8000]),torch.ones(3,dtype=torch.float64))
        self.assertGreater(e['inputGradientNorm'],0);self.assertGreater(e['outputGradientNorm'],0)
        self.assertGreater(float(a.input_log_gain.detach().abs().sum()),0)
        self.assertGreater(float(a.input_shift.detach().abs().sum()),0)
        self.assertGreater(float(a.output_offset.detach().abs().sum()),0)
        for k,v in a.base.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in a.base.parameters()))

    def test_rejects_early_labels_and_backward_clock(self):
        c=OnlineJointHurdleCalibrator(self.adapter(),JointCalibrationConfig(5))
        x=torch.ones(1,2,59);y=torch.ones(1,1,dtype=torch.float64)*.01;w=torch.ones(1,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'matured'): c.update(4999,x,y,np.array([0]),w)
        c.update(5000,x,y,np.array([0]),w)
        with self.assertRaisesRegex(ValueError,'clock'): c.update(5000,x,y,np.array([0]),w)
        with self.assertRaisesRegex(ValueError,'precedes'): c.predict(4999,x)

    def test_window_respects_maturity_and_retains_effective_mass(self):
        stamps=np.arange(1000,dtype=np.int64)*4000
        conf=JointCalibrationConfig(900,batch_size=17)
        indices,weights,matured=matured_window(stamps,2000000,500,conf)
        self.assertTrue(np.all(stamps[indices]+900000<=2000000))
        self.assertEqual(matured,276);self.assertLessEqual(len(indices),17)
        full=np.exp(-np.log(2)*(2000000-stamps[:matured]-900000)/3600000)*(4/900)
        self.assertAlmostEqual(weights.sum(),full.sum(),places=14)

    def test_future_changes_cannot_change_prior_forecasts_and_queries_are_read_only(self):
        stamps=np.arange(80,dtype=np.int64)*4000
        x=torch.linspace(-1,1,80*118).reshape(80,2,59)
        y=np.sin(np.arange(80))*.01
        config=JointCalibrationConfig(9,update_seconds=20,batch_size=8)
        def replay(features,targets):
            a=self.adapter();c=OnlineJointHurdleCalibrator(a,config)
            raw=a.base(features).detach()
            seen=[]
            def labels(indices): seen.extend(indices.tolist());return targets[indices]
            r=replay_joint(c,features,stamps,labels,raw)
            before=[p.detach().clone() for p in a.adapter_parameters()]
            c.predict(int(stamps[-1]),features[:2]);c.predict(int(stamps[-1])+1000,features[:2])
            for old,new in zip(before,a.adapter_parameters()): torch.testing.assert_close(old,new,rtol=0,atol=0)
            return r
        original=replay(x,y);changed=x.clone();changed[40:]+=10
        y2=y.copy();y2[40:]=.5
        later=replay(changed,y2)
        np.testing.assert_array_equal(original['parameters'][:40],later['parameters'][:40])
        self.assertTrue(np.all(original['snapshots']['latestMaturityUsed']<=stamps))

    def test_output_only_ablation_keeps_input_identity_and_uses_cache(self):
        a=self.adapter(False);c=OnlineJointHurdleCalibrator(a,JointCalibrationConfig(1,input_enabled=False))
        x=torch.ones(3,2,59);raw=a.base(x).detach()
        c.update(10000,x,torch.zeros(3,1,dtype=torch.float64),np.array([0,4000,8000]),
            torch.ones(3,dtype=torch.float64),raw=raw)
        self.assertEqual(float(a.input_shift.detach().abs().sum()+a.input_log_gain.detach().abs().sum()),0)
        torch.testing.assert_close(c.predict(10000,x),c.predict(10000,x,raw=raw),rtol=0,atol=0)

    def test_restart_restores_both_adapters_optimizer_and_causal_clock(self):
        config=JointCalibrationConfig(1)
        first=OnlineJointHurdleCalibrator(self.adapter(),config)
        x=torch.ones(3,2,59);y=torch.tensor([[.02],[0.],[-.03]],dtype=torch.float64)
        stamps=np.array([0,4000,8000]);w=torch.ones(3,dtype=torch.float64)
        first.update(10000,x,y,stamps,w)
        restored=OnlineJointHurdleCalibrator(self.adapter(),config)
        restored.load_state_dict(deepcopy(first.state_dict()))
        for model in (first,restored):model.update(20000,x*2,y,stamps+4000,w)
        torch.testing.assert_close(first.predict(20000,x),restored.predict(20000,x),rtol=0,atol=0)
        self.assertEqual(restored.last_update_ms,first.last_update_ms)
        self.assertEqual(restored.updates,first.updates)


if __name__=='__main__': unittest.main()
