"""Summarize archived forecast-revision experiments without rerunning models."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast_revision_metrics import summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--direct',type=Path,required=True)
    ap.add_argument('--rollouts',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=False)
    direct=json.loads((args.direct/'report.json').read_text())
    plan=json.loads((args.direct/'plan.json').read_text())
    h=plan['horizons'];result=dict(direct={},rollout={},refinement={},provenance=dict(direct=str(args.direct),rollouts=[str(p) for p in args.rollouts]))
    for horizon in h:
        key=f'main-h{horizon}'
        a=np.load(args.direct/f'{key}.npz');b=np.load(args.direct/f'refined-h{horizon}.npz')
        result['direct'][str(horizon)]=direct['results'][key]
        indices=b['caseIndices']
        result['refinement'][str(horizon)]=dict(w1AbsoluteDifferenceBps=summary(abs(a['w1Bps'][indices]-b['w1Bps'])),
            independentSameLawW1Bps=direct['results'][f'noise-h{horizon}']['routine']['w1Bps'])
        joined={}
        for root in args.rollouts:
            rp=json.loads((root/'plan.json').read_text())
            if rp['checkpointSha256']!=plan['checkpointSha256']:raise ValueError('Checkpoint mismatch')
            rows=np.load(root/f'h{horizon}.npz')
            for group,sl in [('routine',slice(0,rp['routine'])),('stress',slice(rp['routine'],None))]:
                for k in rows.files:
                    if rows[k].ndim==1:joined.setdefault((group,k),[]).append(rows[k][sl])
        result['rollout'][str(horizon)]={g:{k:summary(np.concatenate(v)) for (group,k),v in joined.items() if group==g} for g in ['routine','stress']}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['| Target initially ahead | Median shift (bps) | 95th percentile (bps) | Median forecast SD (bps) |',
        '| --- | ---: | ---: | ---: |']
    for horizon in h:
        r=result['direct'][str(horizon)]['routine']
        lines.append(f"| {horizon}s | {r['w1Bps']['median']:.4f} | {r['w1Bps']['p95']:.4f} | {r['oldStdBps']['median']:.3f} |")
    lines+=['','| Rollout horizon | Price-mean correction / fresh variance | Positive-part payoff | Threshold event |',
        '| --- | ---: | ---: | ---: |']
    for horizon in h:
        r=result['rollout'][str(horizon)]['routine']
        cells=[str(horizon)+'s']
        for k in ['priceDeltaVarianceOverFresh','upside_10bpsDeltaVarianceOverFresh','above_10bpsDeltaVarianceOverFresh']:
            cells.append(f"{r[k]['median']:.6g}" if r[k]['finite'] else 'not resolved')
        lines.append('| '+' | '.join(cells)+' |')
    (args.output/'tables.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    routine=[result['direct'][str(x)]['routine'] for x in h]
    for name,label,color,style in [('median','Median revision','#187e81','-'),('p95','95th percentile revision','#df8835','-')]:
        ax[0].plot(h,[r['w1Bps'][name] for r in routine],style+'o',color=color,label=label)
    ax[0].plot(h,[r['oldStdBps']['median'] for r in routine],'--',color='#66728e',label='Forecast standard deviation')
    ax[0].plot(h,[result['refinement'][str(x)]['independentSameLawW1Bps']['median'] for x in h],':',color='#955e9e',label='Independent-sampling noise')
    ax[0].set(xscale='log',yscale='log',xlabel='Seconds from earlier origin to fixed target',ylabel='Log-price basis points',title='Direct horizon queries: 2,400 adjacent pairs')
    ax[0].legend(fontsize=9);ax[0].grid(alpha=.16)
    for k,label,color in [('priceDeltaVarianceOverFresh','Terminal price','#187e81'),
        ('upside_10bpsDeltaVarianceOverFresh','Upside beyond 10 bps','#df8835'),
        ('above_10bpsDeltaVarianceOverFresh','Above +10 bps event','#955e9e')]:
        vals=[result['rollout'][str(x)]['routine'][k].get('median',np.nan) for x in h]
        vals=np.asarray(vals)
        ax[1].plot(h,np.where(vals>0,vals,np.nan),'-o',color=color,label=label)
    ax[1].axhline(1,color='#66728e',ls='--',lw=1,label='Fresh estimator variance')
    ax[1].set(xscale='log',yscale='log',xlabel='Seconds from earlier origin to fixed target',ylabel='Paired correction variance / fresh variance',title='One-second rollouts: shared future shocks')
    ax[1].legend(fontsize=9);ax[1].grid(alpha=.16)
    fig.savefig(args.output/'revisions.png',dpi=180)
    fig.savefig(args.output/'revisions.svg')
    plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
