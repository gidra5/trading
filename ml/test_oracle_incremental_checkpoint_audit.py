import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gpu_oracle_bellman import Rules
from oracle_incremental_checkpoint import save_table
from audit_oracle_incremental_checkpoint import audit
import test_oracle_incremental_checkpoint as fixtures


class CheckpointAuditTests(unittest.TestCase):
    def fixture(self,directory):
        directory = Path(directory); table,front = fixtures.IncrementalCheckpointTests().table()
        front['state'][0] = 10000.; front['states'][:,0] = 10000.
        front.update(marketTrades=0,fillTicks=0)
        source = {'ml/'+name:digest for name,digest in table.source_identity.items()}
        plan = dict(arguments=dict(horizon=6,seconds=21,seed=7,candidates=6,origin=0,branches=4),
            grid=asdict(table.grid),rules=asdict(Rules()),fundingMultiplier=1.,checkpointSha256='a'*64,
            sourceHashes=source,initialArchive='initial')
        cfg = dict(terminal_seconds=4.,seed=7,candidates=6,forecast_origin_index=0)
        context = dict(modelIdentity='a'*64,sourceHashes=source,maximumHorizon=6,episodeOrigin=-17.,
            budget=asdict(table.budget),config=cfg)
        (directory/'plan.json').write_text(json.dumps(plan)); (directory/'source').mkdir()
        for name in table.source_identity:
            (directory/'source'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        (directory/'initial').mkdir(); np.savez(directory/'initial'/'front.npz',state=front['state'])
        rows = [dict(update=j,originSeconds=-17.+j,price=100.,horizon=min(6,21-j),markedEquity=10000.,
            exposure=0.,orders=0,filled=False,marketTrade=False,tradeQuote=0.,state=front['state'].tolist())
            for j in range(1,18)]
        (directory/'updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        save_table(table,front,context,directory/'checkpoints')

    def test_independent_snapshot_and_execution_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory); result = audit(directory)
            self.assertEqual(result['checkpointUpdate'],17)
            self.assertEqual(result['checkedLogUpdates'],17)
            self.assertEqual(result['ledger']['marketTrades'],0)
            self.assertFalse(result['ledger']['terminalChecked'])
            self.assertGreater(result['checkedBytes'],0)

    def test_snapshot_must_match_accepted_execution_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            path = Path(directory)/'updates.jsonl'; rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]['state'][0] += 1.
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'account differs'): audit(directory)


if __name__=='__main__': unittest.main()
