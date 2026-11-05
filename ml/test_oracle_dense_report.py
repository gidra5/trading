import json
from pathlib import Path
import tempfile
import unittest
from report_oracle_dense_backtest import summarize


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'plan.json').write_text(json.dumps(dict(arguments=dict(seconds=5400,horizon=3600))))

    def tearDown(self):self.temp.cleanup()

    def test_incomplete_run_has_no_terminal_return_and_ignores_unflushed_line(self):
        (self.root/'updates.jsonl').write_text(json.dumps(dict(update=1,markedEquity=9900.))+'\n'+ '{"update":2')
        (self.root/'progress.json').write_text(json.dumps(dict(update=1,elapsedSeconds=2.)))
        r=summarize(self.root)
        self.assertIsNone(r['terminalReturn']);self.assertEqual(r['observedSeconds'],1)
        self.assertAlmostEqual(r['maximumObservedDrawdown'],.01)

    def test_terminal_wealth_must_match_flat_cash(self):
        result=dict(observedUpdates=5400,completed=True,markedEquity=10100.,terminalWealth=10100.,
            terminalReturn=.01,state=[10100.,1.,100.,100.],elapsedSeconds=2000.)
        (self.root/'results.json').write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,'flat'):summarize(self.root)
        result['state'][1]=0.
        (self.root/'results.json').write_text(json.dumps(result))
        r=summarize(self.root)
        self.assertTrue(r['terminalFlat']);self.assertEqual(r['terminalReturn'],.01)

    def test_interruption_and_live_reference_checks_are_preserved(self):
        (self.root/'interruption.json').write_text(json.dumps(dict(reason='Operator correction')))
        checks=[dict(update=1,maximumRootError=0.)]
        (self.root/'comparisons.json').write_text(json.dumps(checks))
        r=summarize(self.root)
        self.assertEqual(r['status'],'interrupted')
        self.assertEqual(r['referenceChecks'],checks)
        self.assertIsNone(r['terminalReturn'])


if __name__=='__main__':unittest.main()
