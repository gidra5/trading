import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { observableTree, solveCrra, finiteActionMenu, type ActionMenu } from '../packages/bot-algo/src/oracle-crra-policy.js';
import { advance, type OracleRules, type OracleState } from '../packages/bot-algo/src/oracle-execution.js';
import { rat, ONE, ZERO, toNumber, isZero, serializeRat } from '../packages/bot-algo/src/oracle-rational.js';

async function main(){
 const root='data/backtests/oracle-gpu-scaling';
 const benchmark=JSON.parse(await readFile(`${root}/benchmark.json`,'utf8'));
 const marks=JSON.parse(await readFile('data/backtests/oracle-nsf-planning/marks.json','utf8'));
 const rules:OracleRules={makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),entryLower:rat(-2),entryUpper:rat(2),
  maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:rat(5),quoteMaximum:rat(10000),priceStep:rat('0.01'),maxOrders:1};
 const state:OracleState={quote:rat(10000),asset:ZERO,price:rat(marks.prices[0]),previousPrice:rat(marks.prices[0]),time:0,orders:[]};
 const reports=[];
 for(const name of ['64-branches-one-second','fixed-terminal-fine-root']){
  const row=benchmark.results.find((r:any)=>r.name===name);assert(row);
  const tree=observableTree(JSON.parse(await readFile(`${root}/${name}-tree.json`,'utf8')),rules.priceStep);
  const menus=row.exposureSteps.map((step:number)=>finiteActionMenu({
   exposureTargets:Array.from({length:Math.round(4/step)+1},(_,i)=>rat((-2+i*step).toFixed(12))),
   limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1}));
  const menu:ActionMenu=(a,t,r)=>menus[a.state.time](a,t,r).filter((x:any)=>isZero(x.action.marketQuote)||!x.action.place.length);
  const started=performance.now();
  const result=solveCrra(state,tree,{riskAversion:0,rules,menu,funding:()=>ONE},false);
  const seconds=(performance.now()-started)/1000;
  const error=Math.abs(toNumber(result.value!)+1-row.expectedWealth);
  if(error>=1e-7){
   const debug:any[]=[];
   function serializeTree(t:any):any{return {children:t.children.map((b:any)=>({probability:serializeRat(b.probability),price:toNumber(b.price),observationKey:b.observationKey,...serializeTree(b)}))};}
   function inspect(s:OracleState,t:any,d:any){
    const o=s.orders[0];
    debug.push({state:[toNumber(s.quote),toNumber(s.asset),toNumber(s.price),toNumber(s.previousPrice),o?(o.side==='buy'?1:-1):0,o?toNumber(o.price):0,o?toNumber(o.notional):0,o?.placedAt??0,s.time],
     tree:serializeTree(t),steps:row.exposureSteps.slice(s.time),value:toNumber(d.value)+1,chosenQuote:toNumber(d.action.marketQuote),chosenPlace:d.action.place.map((x:any)=>({side:x.side,price:toNumber(x.price),notional:toNumber(x.notional)}))});
    for(const b of t.children)if(b.children.length){const moved=advance(d.state,b.price,ONE,rules);const child=solveCrra(moved,b,{riskAversion:0,rules,menu,funding:()=>ONE},false);inspect(moved,b,child);}
   }
   inspect(state,tree,result);await writeFile(`${root}/debug-reference.json`,JSON.stringify(debug));
   console.log({cpuValue:toNumber(result.value!)+1,cpuQuote:toNumber(result.action.marketQuote),gpuValue:row.expectedWealth,gpuQuote:row.marketQuote});
  }
  assert(error<1e-7,`GPU benchmark value mismatch: ${name}: ${error}`);
  const report={name,exactExpectedWealth:toNumber(result.value!)+1,gpuValueError:error,cpuSeconds:seconds,
   gpuSeconds:row.bellmanSeconds,speedup:seconds/row.bellmanSeconds};reports.push(report);console.log(report);
 }
 await writeFile(`${root}/benchmark-exact-check.json`,JSON.stringify(reports,null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
