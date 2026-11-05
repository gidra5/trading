import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { finiteActionMenu, observableTree, type PriceTree, type ForecastTree, type ActionMenu } from '../packages/bot-algo/src/oracle-crra-policy.js';
import { type OracleRules, type OracleState } from '../packages/bot-algo/src/oracle-execution.js';
import { solveLogCrra, logShadowUpper, refinedLogShadowUpper, logInterval, belowCeTolerance,
  type LogInterval, type LogSolverConfig } from '../packages/bot-algo/src/oracle-log-policy.js';
import { ZERO, ONE, rat, sub, gt, lt, eq, toNumber, serializeRat, isZero, type Rational } from '../packages/bot-algo/src/oracle-rational.js';

async function main() {
  const source=process.argv[2],output=process.argv[3];
  if(!source||!output)throw new Error('Usage: evaluate-oracle-log-tail.ts <frozen-shadow-forecast-directory> <new-output-directory>');
  await mkdir(output);
  const input=await readFile(`${source}/forecasts.json`),prior=JSON.parse(await readFile(`${source}/plan.json`,'utf8'));
  const previous=JSON.parse(await readFile('data/backtests/oracle-shadow-depth-20260913-v2/report.json','utf8'));
  const hash=(data:Uint8Array)=>createHash('sha256').update(data).digest('hex');
  assert.equal(hash(input),previous.plan.forecastSha256,'The gamma comparison must use the identical frozen law');
  const offsets:readonly number[]=prior.offsets;
  const sources=['scripts/evaluate-oracle-log-tail.ts','packages/bot-algo/src/oracle-log-policy.ts',
    'packages/bot-algo/src/oracle-shadow-tail.ts','packages/bot-algo/src/oracle-execution.ts',
    'packages/bot-algo/src/oracle-crra-policy.ts','packages/bot-algo/src/oracle-rational.ts',
    'docs/theory/lean/TradingTheory/OracleLogTailBound.lean'];
  const plan={source,forecastSha256:hash(input),checkpointSha256:prior.checkpointSha256,offsets,gamma:1,wealthUnit:10000,
    objective:'max E log(terminal wealth / 10000), valid flat terminal close',
    epsilon:'exp(V_48h - V_H) - 1: relative certainty-equivalent continuation contribution',
    tolerance:0.001,numericMethod:'64-bit outward fixed-point log intervals; exact rational probability backups; exp is display-only',
    fixedFirstAction:'selected by gamma-one one-hour finite policy, then held fixed across cutoffs',
    scope:'unchanged five-stage 32-leaf sampled law; no omitted per-second decisions or unsampled NSF tails',
    sourceHashes:Object.fromEntries(await Promise.all(sources.map(async p=>[p,hash(await readFile(p))])))};
  await writeFile(`${output}/plan.json`,JSON.stringify(plan,null,2));
  const rows:readonly {origin:number;price:number;tree:ForecastTree}[]=JSON.parse(input.toString());
  const rules:OracleRules={makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),entryLower:rat(-2),entryUpper:rat(2),
    maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:rat(5),quoteMaximum:rat(10000),priceStep:rat('0.01'),maxOrders:1};
  const raw=finiteActionMenu({exposureTargets:[-2,-1,0,1,2].map(rat),limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1});
  const menu:ActionMenu=(s,t,r)=>raw(s,t,r).filter(x=>isZero(x.action.marketQuote)||!x.action.place.length);
  const config:LogSolverConfig={riskAversion:1,wealthUnit:rat(10000),rules,menu,funding:()=>ONE};
  const prune=(tree:PriceTree,d:number):PriceTree=>({children:d?tree.children.map(b=>({...b,...prune(b,d-1)})):[]});
  const number=(x:Rational)=>({exact:serializeRat(x),decimal:toNumber(x)});
  const interval=(v:LogInterval)=>({lower:number(v.lower),upper:number(v.upper),width:number(sub(v.upper,v.lower))});
  const positive=(x:Rational)=>lt(x,ZERO)?ZERO:x;
  const threshold=logInterval(rat('1.001'));
  const results=[];
  for(const row of rows) {
    const state:OracleState={quote:rat(10000),asset:ZERO,price:rat(row.price),previousPrice:rat(row.price),time:0,orders:[]};
    const tree=observableTree(row.tree,rules.priceStep);
    const selected=solveLogCrra(state,prune(tree,3),config);
    const fixed:LogSolverConfig={...config,menu:(s,t,r)=>s.state.time===0?[{id:selected.id,action:selected.action}]:menu(s,t,r)};
    const full=solveLogCrra(state,tree,fixed);assert(full.value!==null);
    const direct=logShadowUpper(state,tree,config,selected.action);assert(direct!==null&&!lt(direct,full.value.lower));
    const refined=refinedLogShadowUpper(state,tree,config,4,selected.action);
    assert(refined!==null&&!lt(refined,full.value.lower)&&!gt(refined,direct));
    const old=previous.results.find((r:any)=>r.origin===row.origin);assert(old);
    const cutoffs=[];
    for(let d=1;d<=offsets.length;d++) {
      const terminal=d===offsets.length;
      const prefix=terminal?full:solveLogCrra(state,prune(tree,d),fixed);assert(prefix.value!==null);
      const prefixUpper=logShadowUpper(state,prune(tree,d),config,selected.action);assert(prefixUpper!==null);
      const deltaLower=terminal?ZERO:positive(sub(full.value.lower,prefixUpper));
      const deltaUpper=terminal?ZERO:sub(direct,prefix.value.lower);
      const finiteLower=terminal?ZERO:sub(full.value.lower,prefix.value.upper);
      const finiteUpper=terminal?ZERO:sub(full.value.upper,prefix.value.lower);
      const refinedDelta=terminal?ZERO:sub(refined,prefix.value.lower);
      const result={horizonSeconds:offsets[d-1],modeledTransitions:d,prefixFiniteLogValue:interval(prefix.value),
        prefixContinuousLogUpper:number(prefixUpper),prefixPolicyExpectedWealth:number(prefix.expectedWealth),
        prefixFiniteCeApprox:10000*Math.exp(toNumber(prefix.value.lower)),
        continuousDeltaLower:number(deltaLower),continuousDeltaUpper:number(deltaUpper),
        continuousCeEpsilonLowerApprox:Math.expm1(toNumber(deltaLower)),continuousCeEpsilonUpperApprox:Math.expm1(toNumber(deltaUpper)),
        finiteDeltaLower:number(finiteLower),finiteDeltaUpper:number(finiteUpper),
        finiteCeEpsilonLowerApprox:Math.expm1(toNumber(finiteLower)),finiteCeEpsilonUpperApprox:Math.expm1(toNumber(finiteUpper)),
        refinedFiniteCeEpsilonUpperApprox:Math.expm1(toNumber(refinedDelta)),
        continuousPass:belowCeTolerance(deltaUpper,rat('0.001')),finitePass:belowCeTolerance(refinedDelta,rat('0.001')),
        continuousRejected:!lt(deltaLower,threshold.upper),fullHorizonIdentityUsed:terminal};
      cutoffs.push(result);
      console.log(JSON.stringify({origin:row.origin,H:result.horizonSeconds,lowerPercent:result.continuousCeEpsilonLowerApprox*100,
        upperPercent:result.continuousCeEpsilonUpperApprox*100,finitePercent:result.finiteCeEpsilonLowerApprox*100,pass:result.continuousPass}));
    }
    const result={origin:row.origin,fixedAction:selected.id,gammaZeroFixedAction:old.fixedAction,
      gammaZeroFullFiniteWealth:old.fullFiniteValue.decimal,
      fullFiniteLogValue:interval(full.value),fullLogPolicyExpectedWealth:number(full.expectedWealth),
      fullFiniteCeApprox:10000*Math.exp(toNumber(full.value.lower)),continuousFullLogUpper:number(direct),
      continuousFullCeUpperApprox:10000*Math.exp(toNumber(direct)),refinedFiniteFullLogUpper:number(refined),cutoffs,
      firstCertifiedContinuousHorizonSeconds:cutoffs.find(c=>c.continuousPass)?.horizonSeconds??null,
      firstCertifiedFiniteHorizonSeconds:cutoffs.find(c=>c.finitePass)?.horizonSeconds??null,
      allEarlierCheckpointsRejected:cutoffs.slice(0,-1).every(c=>c.continuousRejected)};
    results.push(result);
    await writeFile(`${output}/progress.json`,JSON.stringify(results,null,2));
  }
  await writeFile(`${output}/report.json`,JSON.stringify({plan,results},null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
