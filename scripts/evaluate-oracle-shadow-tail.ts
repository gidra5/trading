import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { finiteActionMenu, observableTree, solveCrra, type PriceTree, type ForecastTree, type ActionMenu, type SolverConfig } from '../packages/bot-algo/src/oracle-crra-policy.js';
import { type OracleRules, type OracleState } from '../packages/bot-algo/src/oracle-execution.js';
import { quotePrefixUpper, relativeTailEnvelope } from '../packages/bot-algo/src/oracle-tail-bound.js';
import { shadowTreeUpper } from '../packages/bot-algo/src/oracle-shadow-tail.js';
import { ZERO, ONE, rat, add, sub, div, lt, toNumber, serializeRat, isZero, type Rational } from '../packages/bot-algo/src/oracle-rational.js';

async function main() {
  const root = process.argv[2];
  if (!root) throw new Error('Supply new forecast directory');
  try { await readFile(`${root}/report.json`); throw new Error('Completed result exists'); }
  catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e; }
  const plan = JSON.parse(await readFile(`${root}/plan.json`,'utf8'));
  const sha = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');
  for (const [path,hash] of Object.entries(plan.sourceHashes)) assert.equal(sha(await readFile(path)),hash,path);
  const original = await readFile(`${plan.source}/forecasts.json`);
  assert.equal(sha(original),plan.sourceForecastSha256);
  const previous = JSON.parse(original.toString());
  const rows: readonly { readonly origin: number; readonly price: number; readonly tree: ForecastTree }[] = JSON.parse(await readFile(`${root}/forecasts.json`,'utf8'));
  const rules: OracleRules = { makerFee: rat('0.0002'), takerFee: rat('0.0005'), reserveLeverage: rat(2),
    entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
    quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat('0.01'), maxOrders: 1 };
  const raw = finiteActionMenu({ exposureTargets: [-2,-1,0,1,2].map(rat),limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1 });
  const menu: ActionMenu = (s,t,r) => raw(s,t,r).filter(x=>isZero(x.action.marketQuote)||!x.action.place.length);
  const config: SolverConfig = { riskAversion:0,rules,menu,funding:()=>ONE };
  const prune = (tree: PriceTree, depth: number): PriceTree => ({ children: depth ? tree.children.map(b=>({...b,...prune(b,depth-1)})) : [] });
  const number = (r: Rational) => ({ exact:serializeRat(r),decimal:toNumber(r) });
  const results = [];
  for (const row of rows) {
    const state: OracleState = { quote:rat(10000),asset:ZERO,price:rat(row.price),previousPrice:rat(row.price),time:0,orders:[] };
    const tree = observableTree(row.tree,rules.priceStep), prefix = prune(tree,3);
    assert.deepEqual(prefix,observableTree(previous.find((r: any)=>r.origin===row.origin&&r.horizonSeconds===3600).tree,rules.priceStep));
    const selected = solveCrra(state,prefix,config);
    assert(selected.value !== null);
    const q = add(selected.value,ONE);
    const start = performance.now();
    const upper = shadowTreeUpper(state,tree,config,selected.action);
    const prefixDual = shadowTreeUpper(state,prefix,config,selected.action);
    const upperSeconds = (performance.now()-start)/1000;
    const prefixUpper = quotePrefixUpper(state,prefix,config,selected.action)!;
    const oldBound = relativeTailEnvelope(q,prefixUpper,2,rules.quoteMaximum);
    const fixedConfig: SolverConfig = { ...config,menu:(s,t,r)=>s.state.time===0?[{id:selected.id,action:selected.action}]:menu(s,t,r) };
    console.log(JSON.stringify({origin:row.origin,stage:'upper',Q1h:toNumber(q),upper:toNumber(upper.upper),seconds:upperSeconds}));
    const solveStart = performance.now();
    const full = solveCrra(state,tree,fixedConfig);
    assert(full.value !== null);
    const fullWealth = add(full.value,ONE);
    assert(!lt(fullWealth,q) && !lt(upper.upper,fullWealth));
    const continuousLower = div(sub(fullWealth,prefixDual.upper),prefixDual.upper);
    const result = { origin:row.origin,prefixValue:number(q),sameActionFiniteFullValue:number(fullWealth),
      continuousPrefixUpper:number(prefixDual.upper),
      continuousRelativeLower:number(lt(continuousLower,ZERO)?ZERO:continuousLower),
      shadowUpper:number(upper.upper),cashUpper:number(upper.cashUpper),
      oldMatchedRelativeBound:number(oldBound),newRelativeBound:number(div(sub(upper.upper,q),q)),
      feasibleRelativeContribution:number(div(sub(fullWealth,q),q)),
      belowPointOnePercent:lt(div(sub(upper.upper,q),q),rat('0.001')),
      selectedFirstAction:selected.id,paths:upper.paths.length,upperSeconds,fullSolveSeconds:(performance.now()-solveStart)/1000,
      upperScope:'every continuous feasible policy under the matched five-stage finite law, fixed first action',
      lowerScope:'exact optimal fixed-first-action wealth in the current finite action menu under that same law' };
    results.push(result);
    console.log(JSON.stringify({origin:row.origin,Q1h:toNumber(q),Q48h:toNumber(fullWealth),
      continuousPrefixUpper:toNumber(prefixDual.upper),upper:toNumber(upper.upper),
      continuousRelativeLower:result.continuousRelativeLower.decimal,
      relativeUpper:result.newRelativeBound.decimal,upperSeconds,fullSolveSeconds:result.fullSolveSeconds}));
    await writeFile(`${root}/progress.json`,JSON.stringify(results,null,2));
  }
  await writeFile(`${root}/report.json`,JSON.stringify({plan,results,scope:plan.scope},null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
