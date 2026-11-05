"""Write the joint online calibration comparison from verified artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('data/backtests/oracle-joint-calibration'))
    p.add_argument('--output',type=Path,default=Path('docs/experiments/oracle-crra-joint-input-output-calibration-2026-09-11.md'))
    args=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    e=read(args.root/'evaluation.json');audit=read(args.root/'calibration-diagnostics.json')
    comparison=read(args.root/'comparison.json');lean=read(args.root/'lean-verification.json')
    if not audit['allChecksPassed'] or not comparison['artifactChecksPassed'] or any(not r['passed'] for r in lean['reports']):
        raise ValueError('verification is incomplete')
    rows=[r for r in e['results'] if r['split']=='test'];h1=next(r for r in rows if r['horizonSeconds']==1)
    names=('fisherOutputOnly','output-only-calibrated','joint-calibrated')
    labels={1:'1 second',900:'15 minutes',1800:'30 minutes',7200:'2 hours'}
    key='allTickDeployedMseSkillVsZero'
    def skill(row,variant,metric=key):return 100*row['models'][variant]['logReturns'][metric]
    wins=sum(skill(r,'joint-calibrated')>skill(r,'output-only-calibrated') for r in rows)
    previous_wins=sum(skill(r,'joint-calibrated')>skill(r,'fisherOutputOnly') for r in rows)
    lines=['# Joint online input and output calibration','',
        'Implemented and replayed joint online calibration on the corrected, independently trained per-horizon log-return specialists. All predictor weights, distribution shapes and training normalization remain frozen. A single current likelihood loss backpropagates through the predictor to both adapters.','',
        f'The joint variant improves test all-tick log-return MSE skill over the same-optimizer output-only ablation at {wins}/4 horizons, and over the retained Fisher output calibration at {previous_wins}/4 horizons. At one second, joint calibration scores {skill(h1,"joint-calibrated"):.6f}%, versus {skill(h1,"output-only-calibrated"):.6f}% for the optimizer-matched output-only ablation and {skill(h1,"fisherOutputOnly"):.6f}% for the retained output calibrator.','',
        '## Comparable forecast results','',
        'Log-return MSE skill is `100*(1-MSE/mean(target^2))`, measured on the same 36,134 test origins. Negative skill loses to predicting zero. The retained column is the previous causal Fisher output calibration on the same frozen log checkpoints.','',
        '| Horizon | Retained output calibration | Output-only, matched Adam | Joint input + output |',
        '| --- | ---: | ---: | ---: |']
    for row in rows:lines.append('| '+labels[row['horizonSeconds']]+' | '+' | '.join(f'{skill(row,n):.6f}%' for n in names)+' |')
    lines+=['','For the same 15,505 nonzero one-second outcomes:','',
        '| Metric | Retained output calibration | Output-only, matched Adam | Joint input + output |',
        '| --- | ---: | ---: | ---: |']
    for label,k in (('Normal forecast, nonzero outcomes','activeDeployedMseSkillVsZero'),
                    ('Conditional-on-nonzero forecast','activeConditionalMseSkillVsZero')):
        lines.append('| '+label+' | '+' | '.join(f'{skill(h1,n,k):.6f}%' for n in names)+' |')
    lines+=['','The conditional-on-nonzero score removes the forecast zero-atom weight. It is a diagnostic, because future activity is unknown at issuance.','',
        'Test log-return negative log likelihood (lower is better; the update objective):','',
        '| Horizon | Retained output calibration | Output-only, matched Adam | Joint input + output |',
        '| --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| '+labels[row['horizonSeconds']]+' | '+' | '.join(f'{row["models"][n]["logReturnNegativeLogLikelihood"]:.6f}' for n in names)+' |')
    lines+=['','Validation log-return MSE skill, shown separately from test:','',
        '| Horizon | Retained output calibration | Output-only, matched Adam | Joint input + output |',
        '| --- | ---: | ---: | ---: |']
    for row in (r for r in e['results'] if r['split']=='validation'):
        lines.append('| '+labels[row['horizonSeconds']]+' | '+' | '.join(f'{skill(row,n):.6f}%' for n in names)+' |')
    lines+=['','## Joint training and causal timing','',
        'Each horizon owns 122 adaptive parameters: 59 log gains and 59 shifts on the normalized inputs, plus four output offsets. The input transform is shared over the two observed ticks and the three experts. In standardized coordinates it is `z -> exp(g)*z+b`. The output adapter shifts the zero/sign logits and the two magnitude means. UP uses mean/shape log odds to preserve finite first/second positive price moments; shapes stay fixed.','',
        'One Adam optimizer updates both adapters from full log-hurdle negative log likelihood, with an identity-centered quadratic prior. Input learning rate is 0.001, output learning rate 0.01; respective prior precisions are 32 and 8. Gains are bounded to [0.5,2], shifts to one training standard deviation, probability offsets to ±2 and magnitude offsets to ±log(2). These settings were fixed before validation/test evaluation; no test-driven selection or tuning occurred.','',
        'Updates happen on a fixed 60-second clock. At time t, only forecasts issued earlier whose endpoint t_origin+H has already been observed are eligible. The loss uses the last 3,600 seconds of mature labels, exponentially weighted with a 3,600-second half-life. At most 128 deterministic midpoint representatives approximate that window, preserving each bin’s total weight. The previous overlap weight min(1,4/H) is retained as a heuristic for effective information.','',
        'A single loss backward pass reaches both adapters through the frozen predictor. This is joint end-to-end online optimization, with no claim that one gradient step finds the exact window optimum. Earlier features are replayed under the current adapters when their labels mature; historical issued forecasts are never rewritten. Prediction queries do not train, and changing the trading-query frequency cannot create extra updates.','',
        'All variants start at identity independently for each split/horizon, using the same four-second issuance stream and frozen base checkpoints. The output-only ablation uses exactly the same clock, objective, window, prior and output optimizer, with input parameters disabled. Archived Fisher calibration remains a separate reference. Saved online state includes both adapters, Adam state and the last update timestamp for deterministic restart with the same base checkpoint.','',
        '## Backtests','',
        'Gamma remains zero. The same 42-hour BTCUSDT spot tape is replayed through the restricted synthetic Oracle long/cash endpoint account. Paid runs charge received-amount taker fees of 0.05% per side.','',
        '| Horizon | Output-only, fees | Joint, fees | Joint trades | Output-only, zero fees | Joint, zero fees |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in comparison['horizons']:
        lookup=lambda v,fee:next(p for p in next(vv for vv in row['variants'] if vv['name']==v)['replays'] if p['feeScale']==str(fee))
        old,new=lookup('output-only-calibrated',1),lookup('joint-calibrated',1)
        old0,new0=lookup('output-only-calibrated',0),lookup('joint-calibrated',0)
        lines.append(f'| {labels[row["horizonSeconds"]]} | {old["returnPercent"]:.6f}% | {new["returnPercent"]:.6f}% | {new["markets"]} | {old0["returnPercent"]:.6f}% | {new0["returnPercent"]:.6f}% |')
    lines+=['','These historical backtests reduced the forecast law to three conditional price means and used target exposures 0,0.5,1 plus a zero trade. Zero trade holds the account; zero target exposure attempts to close it. This was a matched comparison under a restricted controller, not a comparison using continuous optimal actions. The [continuous-action, full-distribution replay](oracle-crra-continuous-full-law-2026-09-12.md) supersedes this controller comparison. Both experiments use unit funding, no pending orders, quote-size caps, price quantization, one-second maintenance marks and valid final closing with residuals.','',
        '## Verification','',
        'The compiler specialization limit is 64 and exceeding it raises an error. The default limit caused an execution fallback during an incomplete validation stream; that partial stream was restarted before test data loaded. Completed validation streams were retained, and the original source, plan and progress are preserved under `implementation-history`, with the change recorded in `runtime-correction.json`. Calibration settings did not change.','',
        'All 23 relevant tests passed, including seven new checks for identity/physical output transforms, a common gradient updating both adapters, frozen predictor weights, maturity rejection, future-prefix invariance, query invariance, the output-only ablation and deterministic restart.','',
        f'The independent audit checked {audit["canonicalQueriesChecked"]:,} canonical forecasts and {audit["optimizerStepsChecked"]:,} optimizer events, reconstructing maturity, representative indices and decayed weights. It replayed all {audit["adaptedNeuralPredictionsChecked"]:,} adapted joint forecasts from saved input/output traces with the recorded compiled batch schedule. Maximum replayed logit error was {audit["maximumReplayedLogitError"]:.9g}; maximum relative magnitude error was {audit["maximumReplayedRelativeMagnitudeError"]:.9g}. Cold forecasts match the archived base predictions exactly.','',
        f'Log/simple expected-return scores were independently reconstructed with maximum skill error {audit["maximumIndependentSkillError"]:.9g}. All 12 replay artifact checks passed on the identical tape. Lean verified {comparison["certificateRecords"]:,} supplied decision certificates and 108 execution fixtures. This checks the supplied finite scenario/menu decisions, not forecast quality or unrestricted optimality.','',
        'This is a first fixed-configuration calibration trial on an already inspected historical test period. It does not establish that input calibration will improve every model, horizon or future period. Existing retained forecast references are preserved; no live deployment is changed by the experiment.','',
        f'- Frozen protocol: `{args.root/"plan.json"}`.',
        f'- Forecast results and provenance: `{args.root/"evaluation.json"}`.',
        f'- Causal/forward audit: `{args.root/"calibration-diagnostics.json"}`.',
        f'- Replays and Lean: `{args.root/"comparison.json"}`, `{args.root/"lean-verification.json"}`.',
        f'- Per-split/horizon traces and restart state: `{args.root/"cache"}`.','',
        '```powershell',
        '.venv-ml/Scripts/python.exe ml/evaluate_joint_hurdle_calibration.py',
        '.venv-ml/Scripts/python.exe ml/audit_joint_hurdle_calibration.py',
        'node scripts/run-oracle-neural-specialists.mjs --output data/backtests/oracle-joint-calibration',
        '.venv-ml/Scripts/python.exe ml/report_joint_hurdle_calibration.py','```','']
    args.output.write_text('\n'.join(lines),encoding='utf-8');print(args.output)


if __name__=='__main__':main()
