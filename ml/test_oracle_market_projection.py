import unittest
import numpy as np
from audit_oracle_market_projection import input_coordinates,nearest,law_errors


class MarketProjectionTests(unittest.TestCase):
    def test_relative_coordinates_ignore_common_price_level_scaling(self):
        f=np.zeros((2,2,59));p=np.array([[99.,100.],[101.,100.]])
        expected=input_coordinates(f,p,np.zeros(59),np.ones(59))
        actual=input_coordinates(f,p*np.array([[3.],[.5]]),np.zeros(59),np.ones(59))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)

    def test_relative_matching_can_recover_identical_return_at_another_level(self):
        f=np.zeros((1,2,59));g=np.zeros((2,2,59));mean=np.zeros(59);std=np.ones(59)
        p=np.array([[99.,100.]]);q=np.array([[198.,200.],[100.,100.]])
        self.assertEqual(nearest(input_coordinates(f,p,mean,std),input_coordinates(g,q,mean,std))[0],0)
        self.assertEqual(nearest(input_coordinates(f,p,mean,std,100.),input_coordinates(g,q,mean,std,100.))[0],1)

    def test_empirical_distribution_distance_is_permutation_invariant(self):
        a=np.array([[90.,100.,110.]]);b=np.array([[110.,90.,100.]])
        errors=law_errors(a,b,np.array([100.]))
        self.assertEqual(errors['empiricalW2Bps'],0.)
        self.assertGreater(errors['coupledSampleRMSEBps'],0.)


if __name__=='__main__':unittest.main()
