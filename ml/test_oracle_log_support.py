import unittest
import numpy as np
from gpu_oracle_bellman import Rules
from oracle_book_mc import rule_vector, move_arrive
from oracle_log_support import liquidation_witness


class LogSupportTests(unittest.TestCase):
    def test_short_upper_tail_and_maintenance_boundary(self):
        r = rule_vector(Rules(maker=0., taker=0.))
        s = np.array([200., -1., 100., 100.])
        result = liquidation_witness(s, r)
        self.assertEqual(result['maintenanceThreshold'], 150.)
        self.assertTrue(move_arrive(s, 150., r)[3])
        self.assertFalse(move_arrive(s, result['witnessPrice'], r)[3])

    def test_leveraged_long_lower_tail(self):
        r = rule_vector(Rules(maker=0., taker=0.))
        s = np.array([-100., 2., 100., 100.])
        result = liquidation_witness(s, r)
        self.assertEqual(result['maintenanceThreshold'], 75.)
        self.assertTrue(move_arrive(s, 75., r)[3])
        self.assertLess(result['witnessPrice'], 75.)

    def test_cash_and_funded_long_have_no_inventory_witness(self):
        r = rule_vector(Rules())
        self.assertIsNone(liquidation_witness(np.array([100., 0., 100., 100.]), r))
        self.assertIsNone(liquidation_witness(np.array([0., 1., 100., 100.]), r))

    def test_funding_and_fees_match_execution(self):
        s = np.array([200., -1., 100., 100., 1., 99., 50.])
        base = liquidation_witness(s, rule_vector(Rules()))
        funded = liquidation_witness(s, rule_vector(Rules(), .99))
        self.assertGreater(funded['maintenanceThreshold'], base['maintenanceThreshold'])
        self.assertLess(base['maintenanceThreshold'], 150.)


if __name__ == '__main__':
    unittest.main()
