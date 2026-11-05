import copy
import unittest
from audit_oracle_incremental_restore import compare_step


class RestoreAuditTests(unittest.TestCase):
    def row(self):
        return dict(update=4,originSeconds=104,filled=False,marketTrade=True,state=[98.,1.,2.,2.],
            estimatedLogValue=4.6,horizon=12,stepSeconds=1.,
            table=dict(updates=[dict(depth=1,rows=2,outcomes=8,partialRows=0)],
                updatedLayers=1,updatedRows=2,evaluatedOutcomes=8,cachedRows=2,seconds=.9),
            root=dict(proposals=6,evaluatedOutcomes=24,partialRows=0,seconds=.1))

    def test_runtime_can_change_but_account_value_and_actual_sampling_cannot(self):
        expected=self.row(); actual=copy.deepcopy(expected)
        actual['stepSeconds']=2.; actual['table']['seconds']=1.8
        self.assertEqual(compare_step(actual,expected),0.)
        for key in ('account','value','sampling','root'):
            bad=copy.deepcopy(actual)
            if key=='account': bad['state'][0]+=.01
            elif key=='value': bad['estimatedLogValue']+=1.e-8
            elif key=='sampling': bad['table']['updates'][0]['partialRows']=1
            else: bad['root']['evaluatedOutcomes']=20
            with self.assertRaises(AssertionError): compare_step(bad,expected)

    def test_terminal_account_still_checked_without_a_value_estimate(self):
        row=dict(update=5,originSeconds=105,filled=False,marketTrade=True,state=[100.,0.,2.,2.])
        self.assertEqual(compare_step(row,row),0.)
        wrong=dict(row,state=[99.,0.,2.,2.])
        with self.assertRaises(AssertionError): compare_step(wrong,row)


if __name__=='__main__': unittest.main()
