"""Isolated exact post-state scoring benchmark; no planner mutations.

--compile-only builds CUDA kernels for sm_86 without launching GPU work.
Default mode checks numerical conformance before reporting GPU timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import triton
import triton.language as tl
from oracle_book_gpu import _post
from oracle_gpu_action_compaction import compact_max
from oracle_book_mc import representative,move_arrive,execute_target,post_value
from oracle_dense_value_map import DenseValueMap


@triton.jit
def score_post(s,node,args,constants:tl.constexpr):
    return _post(s,node,args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],
        constants[0],constants[1],constants[2],constants[3],constants[4],constants[5],
        constants[6],constants[7],constants[8],constants[9],constants[10],constants[11])


@triton.jit
def score_actions(states,feasible,nodes,depths,values,wealth,exposures,offsets,book_lookup,size_edges,rules,
                  factors,output,AC:tl.constexpr,D:tl.constexpr,W:tl.constexpr,E:tl.constexpr,
                  C:tl.constexpr,M:tl.constexpr,WBITS:tl.constexpr,EBITS:tl.constexpr,
                  PB:tl.constexpr,SB:tl.constexpr,PS:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
                  LAST:tl.constexpr,COMPACT:tl.constexpr):
    task=tl.program_id(0);lane=tl.arange(0,AC)
    node=tl.load(nodes+task);depth=tl.load(depths+task);s=();r=()
    for j in tl.static_range(D):s+=(tl.load(states+(task*AC+lane)*D+j),)
    for j in tl.static_range(11):r+=(tl.load(rules+j),)
    mask=tl.load(feasible+task*AC+lane)
    args=(values+depth*W*C*M,wealth,exposures,offsets,book_lookup,size_edges,r,factors+task*M*B)
    static:tl.constexpr=(W,E,C,M,WBITS,EBITS,PB,SB,PS,K,B,LAST)
    if COMPACT:value=compact_max(s,mask,node,args,static,AC,score_post)
    else:value=tl.max(tl.where(mask,score_post(s,node,args,static),-float('inf')),0)
    tl.store(output+task,value)


def make_cases(m,settings,count,seed):
    a=m.a;g=m.grid;c=m.cloud;r=m.rules;E=len(a['exposures']);C=len(a['coordinates'])
    NC=min(C,max(E,settings['candidates']));AC=triton.next_power_of_2(1+NC+(1<<g.max_orders)*(E+1))
    rng=np.random.default_rng(seed);states=[];masks=[];nodes=[];depths=[];factors=[];attempts=0
    while len(states)<count:
        attempts+=1
        if attempts>100*count:raise RuntimeError('Too few valid arrivals')
        h=int(rng.integers(len(c.sampled_prices)-1));node=int(rng.integers(c.counts[h]))
        wi=int(rng.integers(len(a['wealth'])));book,ei=divmod(int(rng.integers(C)),E)
        state,okay=representative(a['wealth'][wi],a['exposures'][ei],a['books'][book],
            a['book_counts'][book],c.prices[h,node],r,g.price_scale)
        if not okay:continue
        draw=int(rng.integers(c.destinations.shape[-1]));child=c.destinations[h,node,draw]
        arrived,used,filled,okay=move_arrive(state,c.sampled_prices[h,node,draw],r)
        if not okay:continue
        schedule=np.random.default_rng(settings['seed']+int(c.origin_seconds)+h)
        candidates=np.r_[np.arange(E),schedule.choice(np.arange(E,C),NC-E,replace=False)]
        row=np.zeros((AC,3+3*g.max_orders));row[:,0]=10000.;row[:,2]=arrived[2]
        valid=np.zeros(AC,dtype=bool)
        for lane,cell in enumerate(candidates,1):
            book,ei=divmod(int(cell),E)
            post,okay=execute_target(arrived,used,filled,a['exposures'][ei],a['books'][book],
                a['book_counts'][book],r,g.price_scale)
            row[lane]=np.r_[post[:3],post[4:]];valid[lane]=okay
        occupied=sum(1<<j for j in range(g.max_orders) if arrived[4+3*j]!=0)
        for mask in range(1<<g.max_orders):
            for ei in range(-1,E):
                lane=NC+1+mask*(E+1)+ei+1
                post,okay=execute_target(arrived,used,filled,np.nan if ei<0 else a['exposures'][ei],
                    a['books'][0],0,r,g.price_scale,mask)
                row[lane]=np.r_[post[:3],post[4:]];valid[lane]=okay and not(mask&~occupied)
        factor=np.zeros((c.prices.shape[1],c.destinations.shape[-1]));actual=c.factors(h+1)
        factor[:len(actual)]=actual
        states.append(row);masks.append(valid);nodes.append(child);depths.append(h+1);factors.append(factor)
    return tuple(np.asarray(x) for x in (states,masks,nodes,depths,factors))


def main():
    p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=2000)
    p.add_argument('--compile-only',action='store_true');args=p.parse_args()
    if args.samples<1:raise ValueError('Positive sample count required')
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();m=DenseValueMap.load(args.map)
    settings=json.loads((args.map.parent/'plan.json').read_text())['arguments'];a=m.a;g=m.grid;c=m.cloud
    states,feasible,nodes,depths,factors=make_cases(m,settings,args.samples,20260955)
    _,AC,D=states.shape;_,W,C,M=m.shape;E=len(a['exposures']);B=c.destinations.shape[-1]
    constants=dict(AC=AC,D=D,W=W,E=E,C=C,M=M,WBITS=W.bit_length(),EBITS=E.bit_length(),
        PB=g.price_bins,SB=g.size_bins,PS=g.price_scale,K=g.max_orders,B=B)
    rows=[]
    if args.compile_only:
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        signature={name:'*fp64' for name in ('states','values','wealth','exposures','size_edges','rules','factors','output')}
        signature.update({name:'*i64' for name in ('nodes','depths','offsets','book_lookup')});signature['feasible']='*i1'
        for last in (False,True):
            for compact in (False,True):
                t=time.perf_counter()
                kernel=triton.compile(ASTSource(score_actions,signature,constexprs={**constants,'LAST':last,'COMPACT':compact}),
                    target=GPUTarget('cuda',86,32),options=dict(num_warps=4,enable_fp_fusion=False))
                rows.append(dict(lastDecision=last,compact=compact,compileSeconds=time.perf_counter()-t,
                    ptxBytes=len(kernel.asm['ptx'])))
                print(json.dumps(rows[-1]),flush=True)
    else:
        tensor=lambda x:torch.tensor(np.ascontiguousarray(x),device='cuda')
        output=torch.empty(args.samples,device='cuda',dtype=torch.float64)
        inputs=[tensor(x) for x in (states,feasible,nodes,depths,m.values,a['wealth'],a['exposures'],
            a['code_offsets'],a['book_lookup'],a['size_edges'],m.rules,factors)]
        for last in (False,True):
            outputs={};timings={}
            for compact in (False,True):
                def launch():score_actions[(args.samples,)](*inputs,output,**constants,LAST=last,COMPACT=compact,
                    num_warps=4,enable_fp_fusion=False)
                launch();torch.cuda.synchronize();outputs[compact]=output.cpu().numpy().copy()
                timings[compact]=[]
                for _ in range(5):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for repeat in range(10):launch()
                    end.record();end.synchronize();timings[compact].append(start.elapsed_time(end)/10)
            x,y=outputs[False],outputs[True];finite=np.isfinite(x)
            np.testing.assert_array_equal(finite,np.isfinite(y));np.testing.assert_allclose(x[finite],y[finite],atol=2e-11,rtol=0)
            cpu_error=0.
            for task in range(min(20,args.samples)):
                reference=-np.inf;h=depths[task];field=np.full_like(m.values[h],-np.inf) if last else m.values[h]
                for lane in np.flatnonzero(feasible[task]):
                    s=states[task,lane];state=np.r_[s[:3],s[2],s[3:]]
                    value=post_value(state,nodes[task],field,a['wealth'],a['exposures'],a['code_offsets'],a['book_lookup'],
                        a['size_edges'],g.price_bins,g.price_scale,m.rules,factors[task])
                    reference=max(reference,value)
                if np.isfinite(reference)!=finite[task]:raise AssertionError('CPU feasibility disagreement')
                if finite[task]:cpu_error=max(cpu_error,abs(reference-y[task]))
            if cpu_error>2e-11:raise AssertionError('CPU score mismatch')
            rows.append(dict(lastDecision=last,fullMilliseconds=timings[False],compactMilliseconds=timings[True],
                medianSpeedup=float(np.median(timings[False])/np.median(timings[True])),
                maximumScoreError=float(np.max(abs(x[finite]-y[finite]))) if finite.any() else 0.,
                cpuReferenceMaximumError=cpu_error,finiteScores=int(finite.sum())))
            print(json.dumps(rows[-1]),flush=True)
    result=dict(compileOnly=args.compile_only,samples=args.samples,rows=rows,sourceMap=str(args.map),
        operator=json.loads((args.map/'operator.json').read_text()),elapsedSeconds=time.perf_counter()-started,
        sourceHashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            ('benchmark_oracle_action_compaction.py','oracle_gpu_action_compaction.py')},
        scope='Isolated continuation scoring on precomputed candidate states. Includes GPU compaction cost, excludes account action construction and full-sweep launch costs. Live planner unchanged.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
