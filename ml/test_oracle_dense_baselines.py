import unittest
from report_oracle_dense_baselines import hold_baseline
from oracle_book_mc import rule_vector
from gpu_oracle_bellman import Rules


class BaselineTests(unittest.TestCase):
    def test_half_exposure_direction_and_cash_without_fees(self):
        rules=rule_vector(Rules(taker=0.,maker=0.))
        for exposure,wealth in ((0.,10000.),(.5,10100.),(-.5,9900.)):
            row=hold_baseline([100.,99.,102.],rules,exposure)
            self.assertTrue(row['completed']);self.assertAlmostEqual(row['terminalWealth'],wealth)

    def test_received_amount_fees_and_actual_target_exposure(self):
        row=hold_baseline([100.,102.],rule_vector(Rules()),.5)
        fee=Rules().taker;b=1-fee;entry=5000/(b+.5*fee)
        expected=10000-entry+entry*b*b*1.02
        self.assertAlmostEqual(row['terminalWealth'],expected)
        self.assertAlmostEqual(row['entryFeeQuote'],entry*fee)
        self.assertAlmostEqual(row['closingFeeQuote'],entry*b*1.02*fee)

    def test_funding_changes_held_inventory_on_every_transition(self):
        rules=rule_vector(Rules(taker=0.,maker=0.),.99)
        long=hold_baseline([100.,100.,100.],rules,.5)
        short=hold_baseline([100.,100.,100.],rules,-.5)
        self.assertAlmostEqual(long['terminalWealth'],5000+5000*.99**2)
        self.assertAlmostEqual(short['terminalWealth'],15000-5000*.99**2)

    def test_liquidation_and_uncloseable_position_are_not_valid_returns(self):
        rules=rule_vector(Rules(taker=0.,maker=0.,maximum=10000.))
        for exposure,prices in ((-.5,[100.,400.]),(.5,[100.,300.])):
            row=hold_baseline(prices,rules,exposure)
            self.assertFalse(row['completed']);self.assertIsNone(row['terminalReturn'])


if __name__=='__main__':unittest.main()
