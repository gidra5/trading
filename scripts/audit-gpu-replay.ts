import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { advance, equity, exposure, effectiveAsset, requestForTarget, prepareArrival, executeAction, terminalCloseChecked, type OracleRules, type OracleState, type OracleAction } from '../packages/bot-algo/src/oracle-execution.js';
import { rat, ONE, ZERO, toNumber, abs, min, mul, sub, div, cmp } from '../packages/bot-algo/src/oracle-rational.js';

async function main(){
 const root=process.argv[2]??'data/backtests/oracle-gpu-scaling/second-replay';
 const rows=(await readFile(`${root}/decisions.jsonl`,'utf8')).trim().split('\n').map(x=>JSON.parse(x));
 const marks=JSON.parse(await readFile(`${root}/marks.json`,'utf8'));
 const plan=JSON.parse(await readFile(`${root}/plan.json`,'utf8'));
 const rules:OracleRules={makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),entryLower:rat(-2),entryUpper:rat(2),
  maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:rat(5),quoteMaximum:rat(10000),priceStep:rat('0.01'),maxOrders:1};
 const price=(x:number)=>rat(x.toFixed(2));
 let state:OracleState={quote:rat(10000),asset:ZERO,price:price(marks.prices[0]),previousPrice:price(marks.prices[0]),time:0,orders:[]};
 let maxError=0,fees=0,fills=0,markets=0,placements=0,cancellations=0,liquidations=0;
 for(const row of rows){
  assert.equal(row.index,state.time);
  const prepared=prepareArrival(state,rules);assert(prepared.ok,`arrival ${row.index}`);
  const a=prepared.value;
  for(const order of a.fills){fees+=toNumber(mul(order.notional,rules.makerFee));fills++;}
  const [target,cancel,side,offset,fraction]=row.descriptor??[null,true,0,0,0];
  const request=target===null?ZERO:requestForTarget(a.state,rat(target),rules);
  const available=sub(rules.quoteMaximum,a.consumedQuote);
  let amount=mul(rat(request.n<0n?-1:1),min(abs(request),available));
  if(row.descriptor===null){
   const effective=effectiveAsset(a.state,rules),absolute=abs(effective);
   const direction=rat(effective.n<0n?-1:effective.n>0n?1:0);
   const twiceMinimum=mul(rat(2),rules.quoteMinimum);
   const dust=absolute.n!==0n&&cmp(absolute,rules.quoteMinimum)<0;
   amount=dust?mul(mul(rat(-1),direction),twiceMinimum):mul(direction,min(absolute,available));
   const residual=sub(absolute,abs(amount));
   if(!dust&&residual.n>0n&&cmp(residual,twiceMinimum)<0){amount=mul(direction,sub(absolute,twiceMinimum));}
  }
  const action:OracleAction={marketQuote:amount,cancelIds:cancel?a.state.orders.map(x=>x.id):[],place:side?[{
   id:`t${row.index}`,side:side>0?'buy':'sell',price:sub(a.state.price,mul(rat(side*offset),rules.priceStep)),
   notional:min(mul(equity(a.state),rat(fraction)),side>0?rules.quoteMaximum:div(rules.quoteMaximum,sub(ONE,rules.makerFee))),placedAt:state.time}]:[]};
  const executed=executeAction(a,action,rules);assert(executed.ok,`exact action rejected at ${row.index}: ${!executed.ok&&executed.error.message}`);
  state=executed.value;
  for(const [exact,float] of [[toNumber(state.quote),row.after[0]],[toNumber(state.asset),row.after[1]],[toNumber(amount),row.marketQuote]]){
   maxError=Math.max(maxError,Math.abs(exact-float));assert(Math.abs(exact-float)<1e-7,`numerical drift ${row.index}`);
  }
  markets+=Number(amount.n!==0n);placements+=action.place.length;cancellations+=action.cancelIds.length;
  fees+=toNumber(mul(abs(amount),amount.n<0n?rules.takerFee:div(rules.takerFee,sub(ONE,rules.takerFee))));
  const positive=toNumber(equity(state))>0;
  state=advance(state,price(marks.prices[state.time+1]),ONE,rules);
  liquidations+=Number(positive&&state.quote.n===0n&&state.asset.n===0n);
  assert(Math.abs(toNumber(equity(state))-row.nextMarkedWealth)<1e-7);
 }
 const complete=state.time===plan.archiveSeconds;
 let terminalWealth:number|null=null;
 if(complete){const terminal=terminalCloseChecked(state,rules);assert(terminal.ok);terminalWealth=toNumber(terminal.value.wealth);state=terminal.value.state;}
 const report={complete,terminalRequirement:"zero-asset-and-no-orders",decisions:rows.length,allActualActionsExactlyFeasible:true,maximumNumericalStateOrTradeError:maxError,
  markedWealthBeforeNextArrival:toNumber(equity(state)),terminalWealth,markedReturnPercent:100*(toNumber(equity(state))/10000-1),
  feesPaid:fees,markets,placements,cancellations,fills,liquidations,residualQuote:toNumber(state.quote),residualAsset:toNumber(state.asset),exposure:toNumber(exposure(state))};
 await writeFile(`${root}/exact-replay.json`,JSON.stringify(report,null,2));console.log(report);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
