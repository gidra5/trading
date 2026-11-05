import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { finiteActionMenu, observableTree, solveCrra, type PriceTree, type ForecastTree, type ActionMenu, type SolverConfig } from '../packages/bot-algo/src/oracle-crra-policy.js';
import { type OracleRules, type OracleState } from '../packages/bot-algo/src/oracle-execution.js';
import { shadowTreeUpper, martingaleShadowUpper, continuousShadowUpper, refinedShadowUpper } from '../packages/bot-algo/src/oracle-shadow-tail.js';
import { ZERO, ONE, rat, add, sub, div, lt, gt, toNumber, serializeRat, isZero, type Rational } from '../packages/bot-algo/src/oracle-rational.js';

async function main() {
  const source=process.argv[2],output=process.argv[3];
  if (!source||!output) throw new Error('Usage: evaluate-oracle-shadow-depth.ts <matched-forecast-directory> <new-output-directory>');
  await mkdir(output); // Immutable completed experiments are never overwritten.
  const input=await readFile(`${source}/forecasts.json`);
  const sourcePlan=JSON.parse(await readFile(`${source}/plan.json`,'utf8'));
  const oldReport=JSON.parse(await readFile(`${source}/report.json`,'utf8'));
  const hash=(data:Uint8Array)=>createHash('sha256').update(data).digest('hex');
  const sources=['scripts/evaluate-oracle-shadow-depth.ts','packages/bot-algo/src/oracle-shadow-tail.ts',
    'packages/bot-algo/src/oracle-execution.ts','packages/bot-algo/src/oracle-crra-policy.ts','packages/bot-algo/src/oracle-rational.ts'];
  const offsets:readonly number[]=sourcePlan.offsets;
  const plan={source,forecastSha256:hash(input),checkpointSha256:sourcePlan.checkpointSha256,offsets,
    initialCash:10000,tolerance:0.001,gamma:0,fixedFirstAction:'chosen by the one-hour finite policy, then held fixed across all cutoffs',
    searchScope:'every checkpoint of the unchanged matched five-stage law; no inserted checkpoints or per-second coverage',
    sourceHashes:Object.fromEntries(await Promise.all(sources.map(async p=>[p,hash(await readFile(p))])))};
  await writeFile(`${output}/plan.json`,JSON.stringify(plan,null,2));
  const rows:readonly {readonly origin:number;readonly price:number;readonly tree:ForecastTree}[]=JSON.parse(input.toString());
  const rules:OracleRules={makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),entryLower:rat(-2),entryUpper:rat(2),
    maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:rat(5),quoteMaximum:rat(10000),priceStep:rat('0.01'),maxOrders:1};
  const raw=finiteActionMenu({exposureTargets:[-2,-1,0,1,2].map(rat),limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1});
  const menu:ActionMenu=(s,t,r)=>raw(s,t,r).filter(x=>isZero(x.action.marketQuote)||!x.action.place.length);
  const config:SolverConfig={riskAversion:0,rules,menu,funding:()=>ONE};
  const prune=(tree:PriceTree,d:number):PriceTree=>({children:d?tree.children.map(b=>({...b,...prune(b,d-1)})):[]});
  const number=(v:Rational)=>({exact:serializeRat(v),decimal:toNumber(v)});
  const positive=(v:Rational)=>lt(v,ZERO)?ZERO:v;
  const eps=(upper:Rational,lower:Rational)=>div(sub(upper,lower),lower);
  const results=[];
  for(const row of rows) {
    const state:OracleState={quote:rat(10000),asset:ZERO,price:rat(row.price),previousPrice:rat(row.price),time:0,orders:[]};
    const tree=observableTree(row.tree,rules.priceStep);
    const selected=solveCrra(state,prune(tree,3),config);
    const fixed:SolverConfig={...config,menu:(s,t,r)=>s.state.time===0?[{id:selected.id,action:selected.action}]:menu(s,t,r)};
    const full=solveCrra(state,tree,fixed); assert(full.value!==null);
    const fullValue=add(full.value,ONE);
    const prior=oldReport.results.find((x:any)=>x.origin===row.origin);
    assert.equal(serializeRat(fullValue),prior.sameActionFiniteFullValue.exact,'Frozen law/menu changed');
    const pathUpper=shadowTreeUpper(state,tree,config,selected.action).upper;
    const martingale=martingaleShadowUpper(state,tree,config,selected.action);
    const direct=continuousShadowUpper(state,tree,config,selected.action);
    const refinement=[];
    let tightened=direct;
    for(let d=0;d<=4;d++) {
      const start=performance.now();
      const upper=d===0?direct:refinedShadowUpper(state,tree,config,d,selected.action);
      assert(upper!==null&&!lt(upper,fullValue)&&!gt(upper,tightened));
      tightened=upper;
      refinement.push({accountDepth:d,upper:number(upper),relativeNumericalGap:number(eps(upper,fullValue)),seconds:(performance.now()-start)/1000});
      console.log(JSON.stringify({origin:row.origin,refinement:d,upper:toNumber(upper),numericalGapPercent:100*toNumber(eps(upper,fullValue)),seconds:refinement.at(-1)!.seconds}));
      // This only bounds numerical slack of the finite-model evaluator.
      // It is NOT the requested omitted-horizon contribution.
      if(d>=3&&lt(eps(upper,fullValue),rat('0.0001')))break;
    }
    const cutoffs=[];
    for(let d=1;d<=offsets.length;d++) {
      const terminal=d===offsets.length;
      const prefix=terminal?full:solveCrra(state,prune(tree,d),fixed); assert(prefix.value!==null);
      const q=add(prefix.value,ONE);
      const prefixUpper=continuousShadowUpper(state,prune(tree,d),config,selected.action);
      const finiteLower=terminal?ZERO:eps(fullValue,q);
      const finiteUpper=terminal?ZERO:eps(tightened,q);
      const continuousLower=terminal?ZERO:positive(eps(fullValue,prefixUpper));
      const continuousUpper=terminal?ZERO:eps(direct,q);
      const result={horizonSeconds:offsets[d-1],modeledTransitions:d,prefixFiniteValue:number(q),prefixContinuousUpper:number(prefixUpper),
        finiteRelativeLower:number(finiteLower),finiteRelativeUpper:number(finiteUpper),
        continuousRelativeLower:number(continuousLower),continuousRelativeUpper:number(continuousUpper),
        finitePass:lt(finiteUpper,rat('0.001')),continuousPass:lt(continuousUpper,rat('0.001')),
        continuousRejected:!lt(continuousLower,rat('0.001')),fullHorizonIdentityUsed:terminal};
      cutoffs.push(result);
      console.log(JSON.stringify({origin:row.origin,H:result.horizonSeconds,finiteEpsPercent:100*toNumber(finiteUpper),
        continuousLowerPercent:100*toNumber(continuousLower),continuousUpperPercent:100*toNumber(continuousUpper),pass:result.continuousPass}));
    }
    const firstFinite=cutoffs.find(c=>c.finitePass),firstContinuous=cutoffs.find(c=>c.continuousPass);
    const result={origin:row.origin,fixedAction:selected.id,fullFiniteValue:number(fullValue),
      previousContinuousUpper:prior.shadowUpper,directionalPathUpper:number(pathUpper),
      martingaleUpper:martingale?number(martingale.upper):null,
      martingaleEdgeCheck:martingale?number(martingale.maximumScaledEdgeMove):null,
      newContinuousUpper:number(direct),refinedFiniteUpper:number(tightened),refinement,cutoffs,
      firstCertifiedFiniteHorizonSeconds:firstFinite?.horizonSeconds??null,
      firstCertifiedContinuousHorizonSeconds:firstContinuous?.horizonSeconds??null,
      everyEarlierCheckpointRejected:cutoffs.filter(c=>c.horizonSeconds! < (firstContinuous?.horizonSeconds??Infinity)).every(c=>c.continuousRejected)};
    results.push(result);
    await writeFile(`${output}/progress.json`,JSON.stringify(results,null,2));
  }
  await writeFile(`${output}/report.json`,JSON.stringify({plan,results},null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
