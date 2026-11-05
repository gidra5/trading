"""Record explicit full-support liquidation witnesses for logged post-states."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gpu_oracle_bellman import Rules
from oracle_book_mc import rule_vector
from oracle_log_support import liquidation_witness


def main():
    p = argparse.ArgumentParser()
    p.add_argument('replay', type=Path)
    args = p.parse_args()
    plan = json.loads((args.replay/'plan.json').read_text())
    rules = rule_vector(Rules(**plan['rules']), plan['fundingMultiplier'])
    text = (args.replay/'updates.jsonl').read_text()
    lines = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    rows = [json.loads(line) for line in lines if line.strip()]
    witnesses = []
    for row in rows:
        if 'state' not in row or 'horizon' not in row:
            continue
        result = liquidation_witness(np.array(row['state']), rules)
        if result is not None:
            witnesses.append(dict(update=row['update'], **result))
    result = dict(checkedDecisions=sum('horizon' in r for r in rows),
        decisionsWithInventoryRuinWitness=len(witnesses), firstWitness=witnesses[0] if witnesses else None,
        sourceHashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('oracle_log_support.py', 'oracle_book_mc.py', 'export_nsf_planning_trees.py',
                         'latent_neural_stochastic_flow.py')},
        scope='Explicit Oracle pre-fill liquidation prices under the ideal continuous NSF emission support. '
            'The emission has positive Gaussian scale and only a lower price clamp. Every upper-tail witness interval '
            'has positive probability, however small. No probability magnitude is estimated. '
            'This is distinct from the finite sampled planning law, and absence of an inventory witness does not '
            'certify pending-order or terminal-close feasibility.')
    (args.replay/'log-support-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
