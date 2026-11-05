import unittest
import numpy as np
import torch
from unittest.mock import patch
from oracle_incremental_table import IncrementalTable
from oracle_root_aliases import exact_root_aliases
from oracle_root_candidates import root_candidates
from oracle_local_market_refresh import candidate_schedule
from oracle_selected_outcomes import SelectedOutcomes
from test_oracle_incremental_table import IncrementalTableTests


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RootAliasKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        IncrementalTableTests.setUpClass()

    def test_shared_evaluation_preserves_every_outcome_and_support_flag(self):
        m=IncrementalTableTests().initial()
        arrived=np.array([1000.,0.,100.,100.,1.,99.,100.,-1.,101.,100.])
        _,states,_=root_candidates(m,arrived,seed=7)
        states=np.concatenate((states,states[:5]))
        selected,inverse=exact_root_aliases(states)
        self.assertLess(len(selected),len(states))
        B=m.cloud.destinations.shape[-1]
        self.assertEqual(B,4)
        # Outcomes exercise both order directions, including crossed limits.
        prices=np.tile([98.5,99.5,100.5,101.5],(len(states),1))
        children=np.tile([0,1,0,1],(len(states),1)).astype(np.int64)
        field=np.asarray(m.values[1]).copy()
        field+=np.arange(field.size).reshape(field.shape)*1.e-5
        factors=m.cloud.factors(1); candidates=candidate_schedule(m,0,7,6)
        for remaining in (1,2,6):
            whole=SelectedOutcomes(m,states,prices,children,field,factors,candidates,remaining)
            unique=SelectedOutcomes(m,states[selected],prices[selected],children[selected],
                field,factors,candidates,remaining)
            expected=whole.evaluate(np.tile(np.arange(B),(len(states),1)),np.full(len(states),B))
            actual=unique.evaluate(np.tile(np.arange(B),(len(selected),1)),np.full(len(selected),B))[inverse]
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(unique.supported[inverse],whole.supported)

    def test_root_scatter_preserves_controls_ties_and_partial_sampling(self):
        m=IncrementalTableTests().initial()
        table=IncrementalTable(m,7,6)
        state=np.array([150.,-.5,100.,100.,1.,99.99,10.,-1.,100.01,10.])
        # Full census initializes descriptor-specific controls before testing
        # both a warm full census and partial conditional sampling.
        for draws in (table.B,table.B,2):
            before=None if table.root_controls is None else table.root_controls.copy()
            with patch('oracle_incremental_table.exact_root_aliases',
                    side_effect=lambda states:(np.arange(len(states)),np.arange(len(states)))):
                expected,old_report=table.choose(state,draws=draws,reference_field=table.field(1))
            controls=table.root_controls.copy();table.root_controls=before
            actual,report=table.choose(state,draws=draws,reference_field=table.field(1))
            for key in ('value','state','choice','scores','states'):
                np.testing.assert_array_equal(actual[key],expected[key])
            np.testing.assert_array_equal(table.root_controls,controls)
            self.assertEqual(report['reference'],old_report['reference'])
            self.assertEqual(report['logicalOutcomes'],old_report['logicalOutcomes'])
            if draws==table.B:
                self.assertLess(report['evaluatedOutcomes'],old_report['evaluatedOutcomes'])
            else:
                self.assertEqual(report['evaluatedOutcomes'],old_report['evaluatedOutcomes'])


if __name__=='__main__': unittest.main()
