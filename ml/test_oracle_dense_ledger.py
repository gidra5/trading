import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from audit_oracle_dense_ledger import arrival,arrival_components,market_fee,check_action,audit


RULES=dict(maker=.0002,taker=.0005,reserve_leverage=2.,lower=-2.,upper=2.,
    maintenance_lower=-3.,maintenance_upper=3.,minimum=5.,maximum=10000.,tick=.01)


class LedgerTests(unittest.TestCase):
    def test_attribution_reconciles_signed_funding_and_gapped_limit_fills(self):
        for quantity in (-2.,2.):
            for funding in (.99,1.01):
                for price in (97.,100.,103.):
                    s=np.array([1000.,quantity,100.,100.,1.,99.,50.,-1.,101.,50.])
                    post,_,_=arrival(s,price,RULES,funding)
                    parts=arrival_components(s,price,RULES,funding)
                    pnl=parts['heldPricePnl']+parts['signedFundingPnl']+parts['limitExecutionImprovement']-parts['makerFeesMarkedAtTick']
                    self.assertAlmostEqual(post[0]+post[1]*price-(s[0]+s[1]*100),pnl,places=10)
                    if price==100.:self.assertEqual(parts['makerFeesMarkedAtTick'],0.)
        self.assertAlmostEqual(market_fee(-100.,RULES),.05)
        self.assertAlmostEqual(market_fee(99.95,RULES),.05)

    def test_funding_and_crossed_orders_use_received_amount_fees(self):
        s=np.array([1000.,2.,100.,100.,1.,99.,50.,1.,98.,50.])
        x,capacity,fills=arrival(s,97.,RULES,.99)
        self.assertEqual(fills,2);self.assertEqual(capacity,100.)
        self.assertEqual(x[0],900.)
        self.assertAlmostEqual(x[1],1.98+.9998*50/99+.9998*50/98)
        self.assertTrue(np.all(x[4:]==0))
        np.testing.assert_array_equal(s,np.array([1000.,2.,100.,100.,1.,99.,50.,1.,98.,50.]))

    def test_upward_move_fills_only_sell_and_flat_move_fills_neither(self):
        s=np.array([1000.,0.,100.,100.,1.,99.,50.,-1.,101.,50.])
        x,capacity,fills=arrival(s,101.,RULES,1.)
        self.assertEqual(fills,1);self.assertAlmostEqual(capacity,49.99)
        self.assertAlmostEqual(x[0],1049.99);self.assertAlmostEqual(x[1],-50/101)
        np.testing.assert_array_equal(x[4:7],s[4:7]);self.assertTrue(np.all(x[7:]==0))
        x,capacity,fills=arrival(s,100.,RULES,1.)
        self.assertEqual((capacity,fills),(0.,0));np.testing.assert_array_equal(x,s)

    def test_fee_omission_and_aggregate_capacity_violation_are_rejected(self):
        s=np.array([1000.,0.,100.,100.]);post=np.array([900.,.9995,100.,100.])
        self.assertEqual(check_action(s,post,0.,0,RULES)['quoteFlow'],-100.)
        wrong=post.copy();wrong[1]=1.
        with self.assertRaisesRegex(ValueError,'Asset delta'):check_action(s,wrong,0.,0,RULES)
        with self.assertRaisesRegex(ValueError,'remaining tick capacity'):
            check_action(s,post,50.,1,{**RULES,'maximum':120.})

    def test_passivity_and_shared_reservation_are_checked(self):
        s=np.array([1000.,0.,100.,100.,0.,0.,0.,0.,0.,0.]);post=s.copy()
        post[4:]=[1.,101.,50.,0.,0.,0.]
        with self.assertRaisesRegex(ValueError,'strictly passive'):check_action(s,post,0.,0,RULES)
        post[4:]=[1.,99.,1200.,-1.,101.,1200.]
        with self.assertRaisesRegex(ValueError,'unavailable equity'):check_action(s,post,0.,0,RULES)

    def test_complete_logged_round_trip_verifies_terminal_cash(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'initial').mkdir()
            (p/'plan.json').write_text(json.dumps(dict(rules=RULES,fundingMultiplier=1.,grid=dict(max_orders=0),
                arguments=dict(seconds=2,horizon=2))))
            np.savez(p/'initial'/'front.npz',state=np.array([9000.,9.995,100.,100.]))
            w=9000+9.995*101;final=9000+.9995*9.995*102
            rows=[dict(update=1,horizon=1,price=101.,state=[9000.,9.995,101.,100.],tradeQuote=0.,
                markedEquity=w,exposure=9.995*101/w,orders=0,filled=False),
                dict(update=2,markedEquity=9000+9.995*102,terminalWealth=final)]
            (p/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=dict(completed=True,state=[final,0.,102.,102.],terminalWealth=final,marketTrades=2,fillTicks=0)
            (p/'results.json').write_text(json.dumps(result))
            checked=audit(p)
            self.assertEqual(checked['checkedUpdates'],2);self.assertTrue(checked['terminalChecked'])
            self.assertEqual(checked['marketTrades'],2)
            parts=checked['equityAttribution']
            self.assertAlmostEqual(parts['heldPricePnl'],19.99)
            self.assertAlmostEqual(parts['takerFees'],.5+9.995*102*.0005)
            self.assertAlmostEqual(parts['netPnl'],final-10000.)
            self.assertAlmostEqual(parts['reconciliationError'],0.)
            result['state'][1]=1.
            (p/'results.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'executed close'):audit(p)


if __name__=='__main__':unittest.main()
