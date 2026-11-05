"""Render the entire time x account-cell field; market averaging is display only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oracle_dense_value_map import finite_average


def render(directory):
    directory=Path(directory);plan=json.loads((directory/'plan.json').read_text())
    results=json.loads((directory/'results.json').read_text())
    source=directory/'update-0';v=np.load(source/'values.npy',mmap_mode='r')
    grid=np.load(source/'account-grid.npz');wi=int(abs(grid['wealth']-10000).argmin())
    counts=grid['order_counts'];H=v.shape[0]
    mean=finite_average(v[:,wi],2).T*10000
    with np.load(source/'market.npz') as market:node_counts=market['counts'][:-1]
    valid=(np.sum(np.isfinite(v[:,wi]),axis=2)/node_counts[:,None]).T
    finite=mean[np.isfinite(mean)]
    bound=max(1.,float(np.percentile(abs(finite),98))) if len(finite) else 1.
    plt.rcParams.update({'font.size':10,'figure.facecolor':'#f5f6fa','axes.facecolor':'#dddfe5'})
    fig,axes=plt.subplots(3,2,figsize=(15,10),gridspec_kw={'width_ratios':[3,1]},layout='constrained')
    for k in range(3):
        rows=np.flatnonzero(counts==k)
        if not len(rows):
            for ax in axes[k]:ax.set_axis_off()
            continue
        a=axes[k,0]
        im=a.imshow(np.ma.masked_invalid(mean[rows]),aspect='auto',origin='lower',
            extent=[0,H/60,rows[0]-.5,rows[-1]+.5],cmap='RdBu',vmin=-bound,vmax=bound,interpolation='nearest')
        a.set_ylabel(f'{k} active orders / account cell');a.set_xlabel('Seconds ahead / 60')
        coverage=axes[k,1].imshow(valid[rows],aspect='auto',origin='lower',extent=[0,H/60,0,len(rows)],
            cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
        axes[k,1].set_xlabel('Minutes ahead');axes[k,1].set_yticks([])
    axes[0,0].set_title('Conditional log-wealth gain, averaged across finite market samples (bps)')
    axes[0,1].set_title('Fraction of market samples with finite lookup')
    fig.colorbar(im,ax=axes[:,0],label='Log gain (bps); grey = no finite representative')
    fig.colorbar(coverage,ax=axes[:,1],label='Finite sample fraction')
    fig.suptitle(f'Dense account value map • {H:,} one-second layers • {v.shape[2]} cells • equity ${grid["wealth"][wi]:,.0f}\n'
        'Every account cell allocated; exogenous market states sampled. Display averages are not decision probabilities.',fontsize=13)
    fig.savefig(directory/'heatmap.png',dpi=160);plt.close(fig)
    rows=results['rows']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].bar([r['update']-.15 for r in rows],[r['marketSeconds'] for r in rows],width=.3,label='Market cloud (GPU)')
    backends='/'.join(sorted({r.get('backend','cpu').upper() for r in rows}))
    axes[0].bar([r['update']+.15 for r in rows],[r['seconds'] for r in rows],width=.3,label=f'Bellman sweep ({backends})')
    axes[0].set(xlabel='Observed update',ylabel='Seconds',title='Full hour forecast and value refresh');axes[0].legend()
    for row in rows:
        samples=np.load(directory/f'update-{row["update"]}'/'samples.npy')
        axes[1].hist(samples.ravel(),bins=np.arange(.5,plan['arguments']['branches']+1.5),histtype='step',
            linewidth=2,label=f'Update {row["update"]}')
    axes[1].set(xlabel='Conditional outcomes evaluated per cell',ylabel='Time × account cells',title='Compute allocation; probability weights remain uniform')
    axes[1].legend();fig.savefig(directory/'runtime-and-samples.png',dpi=160);plt.close(fig)
    comparisons=[]
    for row in rows:
        update=row['update'];ref=directory/f'reference-{update}'
        if not row.get('reference') or not (ref/'samples.npy').exists():continue
        av=np.load(directory/f'update-{update}'/'values.npy',mmap_mode='r')[:,wi]
        rv=np.load(ref/'values.npy',mmap_mode='r')[:,wi]
        paired=np.isfinite(av)&np.isfinite(rv)
        delta=np.full(av.shape,np.nan);np.subtract(av,rv,out=delta,where=paired)
        display=finite_average(delta,2).T*10000
        maximum=float(np.nanmax(abs(delta)))*10000
        comparisons.append(dict(update=update,wealth=float(grid['wealth'][wi]),
            pairedSamples=int(paired.sum()),finiteCoverageDisagreements=int(np.count_nonzero(np.isfinite(av)!=np.isfinite(rv))),
            maximumPairedFieldDifferenceBps=maximum))
        fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
        limit=max(float(np.nanpercentile(abs(display),99)),.001)
        im=ax.imshow(np.ma.masked_invalid(display),origin='lower',aspect='auto',extent=[0,H/60,0,len(counts)],
            cmap='RdBu',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set(xlabel='Minutes ahead',ylabel='Account cell',title=f'Update {update}: reduced-sample minus full-sample field (bps)\n'
            'Mean over finite paired market points; same finite forecast cloud and account grid')
        fig.colorbar(im,ax=ax,label='Log-value difference (bps)');fig.savefig(directory/f'mc-difference-{update}.png',dpi=160);plt.close(fig)
    (directory/'field-comparisons.json').write_text(json.dumps(comparisons,indent=2))
    return directory/'heatmap.png'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();print(render(args.directory))
