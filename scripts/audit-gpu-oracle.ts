import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { observableTree, solveCrra, finiteActionMenu, type ActionMenu } from '../packages/bot-algo/src/oracle-crra-policy.js';
import { equity, requestForTarget, executeAction, type OracleRules, type OracleState, type OracleAction } from '../packages/bot-algo/src/oracle-execution.js';
import { rat, ONE, ZERO, toNumber, serializeRat, abs, min, mul, sub, div, isZero } from '../packages/bot-algo/src/oracle-rational.js';

async function main() {
 const outputIndex=process.argv.indexOf('--output');
 const directory=outputIndex<0?'data/backtests/oracle-gpu-scaling':process.argv[outputIndex+1]!;
 await mkdir(directory,{recursive:true});
 const rules:OracleRules={makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),entryLower:rat(-2),entryUpper:rat(2),
  maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:rat(5),quoteMaximum:rat(10000),priceStep:rat('0.01'),maxOrders:1};
 const makeMenu=(step:number):ActionMenu=>{
  const raw=finiteActionMenu({exposureTargets:Array.from({length:Math.round(4/step)+1},(_,i)=>rat((-2+i*step).toFixed(12))),
   limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1});
  return (a,t,r)=>raw(a,t,r).filter(x=>isZero(x.action.marketQuote)||!x.action.place.length);
 };
 if(process.argv.includes('--check')){
  const cases=JSON.parse(await readFile(`${directory}/reference.json`,'utf8'));
  const gpu=JSON.parse(await readFile(`${directory}/gpu-audit.json`,'utf8'));
  let largestRegret=0;
  for(const result of gpu.cases){
   const c=cases.find((x:any)=>x.id===result.id);const s=c.state;
   const state:OracleState={quote:rat(s[0]),asset:rat(s[1]),price:rat(s[2]),previousPrice:rat(s[3]),time:s[8],orders:s[4]?[{id:'pending',side:s[4]>0?'buy':'sell',price:rat(s[5]),notional:rat(s[6]),placedAt:s[7]}]:[]};
   const tree=observableTree(c.tree,rules.priceStep);const base=makeMenu(c.step);
   const selected:ActionMenu=(a,t,r)=>{
    if(a.state.time!==state.time)return base(a,t,r);
    const [target,cancel,side,offset,fraction]=result.descriptor;
    const request=target===null?ZERO:requestForTarget(a.state,rat(target),r);
    const amount=mul(rat(request.n<0n?-1:1),min(abs(request),sub(r.quoteMaximum,a.consumedQuote)));
    const action:OracleAction={marketQuote:amount,cancelIds:cancel?a.state.orders.map(x=>x.id):[],
     place:side?[{id:'gpu',side:side>0?'buy':'sell',price:sub(a.state.price,mul(rat(side*offset),r.priceStep)),
      notional:min(mul(equity(a.state),rat(fraction)),side>0?r.quoteMaximum:div(r.quoteMaximum,sub(ONE,r.makerFee))),placedAt:a.state.time}]:[]};
    assert(executeAction(a,action,r).ok,`GPU action infeasible: ${c.id}`);
    return [{id:'selected-gpu',action}];
   };
   const value=solveCrra(state,tree,{riskAversion:0,rules,menu:selected,funding:()=>rat(c.funding)},false).value!;
   const regret=c.expected-(toNumber(value)+1);largestRegret=Math.max(largestRegret,regret);
   assert(Math.abs(regret)<1e-7,`GPU selection regret ${c.id}: ${regret}`);
  }
  const report={terminalRequirement:"zero-asset-and-no-orders",cases:gpu.cases.length,allSelectedActionsExactlyFeasible:true,maximumExactSelectionRegretDollars:largestRegret};
  await writeFile(`${directory}/exact-selection-check.json`,JSON.stringify(report,null,2));console.log(report);return;
 }
 const rows:any[]=[];
 function tree(prices:number[][],depth=0,node=0):any{
  if(depth===prices.length)return {children:[]};
  const branches=depth===0?prices[0].length:2;
  return {children:Array.from({length:branches},(_,b)=>({probability:`1/${branches}`,price:prices[depth][node*branches+b],
   observationKey:`observed-feature-${b}`,...tree(prices,depth+1,node*branches+b)}))};
 }
 const started=performance.now();
 for(let i=0;i<36;i++){
  const side=i%3===0?0:i%3===1?1:-1;
  const price=100; const inventory=[0,40,-40,100,-100,190][i%6]!;
  const quote=10000-inventory*price;
  const previous=[100,100.03,99.97][i%3]!;
  const s=[quote,inventory,price,previous,side,side>0?100.01:99.99,[5,2500,10000][i%3]!,0,1];
  const state:OracleState={quote:rat(quote),asset:rat(inventory),price:rat(price),previousPrice:rat(previous),time:1,
   orders:side?[{id:'pending',side:side>0?'buy':'sell',price:rat(s[5]!),notional:rat(s[6]!),placedAt:0}]:[]};
  const scenario=i%4===0?[[80,130]]:i%4===1?[[99.98,100.02],[100,99,101,100]]:[[100,100.01]];
  const t=tree(scenario);const step=i<30?1:.01;const funding=[1,.999,1.001][i%3]!;
  try {
   const tick=performance.now();const d=solveCrra(state,observableTree(t,rules.priceStep),{riskAversion:0,rules,menu:makeMenu(step),funding:()=>rat(funding)},false);
   rows.push({id:i,state:s,tree:t,step,funding,expected:toNumber(d.value!)+1,marketQuote:serializeRat(d.action.marketQuote),seconds:(performance.now()-tick)/1000});
  } catch(e) { if(!String(e).includes('No feasible'))throw e; }
 }
 await writeFile(`${directory}/reference.json`,JSON.stringify(rows));console.log({cases:rows.length,seconds:(performance.now()-started)/1000});
}
main().catch(e=>{console.error(e);process.exitCode=1;});
