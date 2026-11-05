"""Coordinate and effective-batch checks for fresh log-target retraining."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from train_log_hurdle_sign_size_forecast import (Examples, HORIZONS, batch_denominators,
    log_constants, log_examples, microbatch_loss, task_scores)


class LogTrainingTest(unittest.TestCase):
    def test_log_targets_keep_zero_sign_but_change_magnitudes(self):
        simple = np.tile(np.array([0., .5, -.5, -.99]), (3,1))
        data = Examples(np.zeros((3,2,59),dtype=np.float32),simple,np.arange(3)+1,0)
        with patch("train_log_hurdle_sign_size_forecast.examples",return_value=data):
            actual = log_examples(None,data.origins)
        np.testing.assert_allclose(actual.targets,np.log1p(simple),rtol=0,atol=0)
        self.assertIs(actual.features,data.features)
        self.assertIs(actual.origins,data.origins)
        np.testing.assert_array_equal(np.sign(actual.targets),np.sign(simple))
        self.assertLess(actual.targets[0,3],-1.)
        self.assertNotEqual(abs(actual.targets[0,1]),abs(actual.targets[0,2]))

    def test_constants_use_only_given_log_targets(self):
        y = np.tile(np.array([0.,.2,-.3,.1,-.6])[:,None],(1,len(HORIZONS)))
        for c in log_constants(y):
            self.assertAlmostEqual(c['upMeanLog'],.15)
            self.assertAlmostEqual(c['downMeanLog'],.45)
            self.assertEqual(c['activeTrainingExamples'],4)
            self.assertNotIn('upMeanSimple',c)

    def test_uneven_microbatches_equal_independent_full_objectives(self):
        y = torch.tensor([[0.,.2],[.3,-.4],[-.1,0.],[0.,-.2],[.4,.1]],dtype=torch.float64)
        scales = torch.tensor([[.2,.3],[.4,.5]],dtype=torch.float64)
        for task in ('zero','sign','size','combined'):
            shape = (*y.shape,2) if task=='size' else ((*y.shape,6) if task=='combined' else y.shape)
            base = torch.linspace(.1,.7,int(np.prod(shape)),dtype=torch.float64).reshape(shape)
            if task=='combined':
                base[...,2:]=torch.tensor([.2,.4,1.4,1.1],dtype=torch.float64)
            full = base.clone().requires_grad_()
            pieces = base.clone().requires_grad_()
            if task=='zero':
                expected=F.binary_cross_entropy_with_logits(full,(y==0).double())
            elif task=='sign':
                expected=torch.stack([F.binary_cross_entropy_with_logits(full[:,j][y[:,j]!=0],(y[:,j][y[:,j]!=0]>0).double()) for j in range(2)]).mean()
            elif task=='size':
                expected=torch.stack([(full[:,j,k][mask]-y[:,j][mask].abs()/scales[j,k]).square().mean()
                    for j in range(2) for k,mask in enumerate((y[:,j]>0,y[:,j]<0))]).mean()
            else:
                zero,sign,up,down,au,ad=full.unbind(-1)
                expected=F.binary_cross_entropy_with_logits(zero,(y==0).double(),reduction='sum')
                expected+=F.binary_cross_entropy_with_logits(sign[y!=0],(y[y!=0]>0).double(),reduction='sum')
                expected-=torch.distributions.Gamma(au[y>0],au[y>0]/up[y>0]).log_prob(y[y>0]).sum()
                expected-=torch.distributions.Gamma(ad[y<0],ad[y<0]/down[y<0]).log_prob(-y[y<0]).sum()
                expected/=y.numel()
            expected.backward()
            den=batch_denominators(y,task)
            loss=0.
            for start,stop in ((0,1),(1,3),(3,5)):
                value=microbatch_loss(pieces[start:stop],y[start:stop],task,scales,den)
                value.backward()
                loss+=float(value.detach())
            self.assertAlmostEqual(loss,float(expected.detach()),places=12)
            torch.testing.assert_close(pieces.grad,full.grad,rtol=1e-12,atol=1e-12)
            self.assertAlmostEqual(float(task_scores(base.numpy(),y.numpy(),task,scales.numpy()).mean()),loss,places=12)


if __name__=='__main__':
    unittest.main()
