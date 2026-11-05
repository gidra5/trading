import { exhaustiveOracleMenu } from '../packages/bot-algo/src/oracle-approximation.js';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { observableTree, solveCrra, finiteActionMenu, type ActionMenu } from '../packages/bot-algo/src/oracle-crra-policy.js';
import { advance, equity, exposure, prepareArrival, executeAction, terminalCloseChecked,
  type OracleRules, type OracleState, type OracleAction } from '../packages/bot-algo/src/oracle-execution.js';
import { rat, ONE, ZERO, toNumber, serializeRat, eq, isZero, abs, div, mul, sub, type Rational } from '../packages/bot-algo/src/oracle-rational.js';
async function main(){
const args=process.argv.slice(2);
const opt=(k:string,d:string)=>{const i=args.indexOf(k);return i<0?d:args[i+1]!;};
const root=opt('--source','data/backtests/oracle-nsf-planning');
const feeScale=Number(opt('--fees','1'));
const refinement=Number(opt('--action-level','1'));
const enumerationLevel=Number(opt('--enumeration-level','0'));
const limitEnabled=opt('--limits','1')==='1';
const maxDecisions=Number(opt('--max-decisions','1000000'));
const out=opt('--output',`${root}/fees${feeScale}-actions${refinement}-limits${Number(limitEnabled)}`);
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const forecastText=await readFile(`${root}/forecasts.jsonl`,'utf8');
const marksText=await readFile(`${root}/marks.json`,'utf8');
const plan=JSON.parse(await readFile(`${root}/forecast-plan.json`,'utf8'));
assert.equal(hash(forecastText),plan.forecastSha256);assert.equal(hash(marksText),plan.marksSha256);
const rows=forecastText.trim().split('\n').map(x=>JSON.parse(x));
const marks=JSON.parse(marksText);
assert.equal(marks.startTimestamp+(marks.prices.length-1)*1000,plan.terminalTimestamp);
const rules:OracleRules={makerFee:rat(.0002*feeScale),takerFee:rat(.0005*feeScale),reserveLeverage:rat(2),
 entryLower:rat(-2),entryUpper:rat(2),maintenanceLower:rat(-3),maintenanceUpper:rat(3),
 quoteMinimum:rat(5),quoteMaximum:rat(10000),priceStep:rat('0.01'),maxOrders:limitEnabled?1:0};
const grid=Array.from({length:2**refinement+1},(_,i)=>rat(-2+4*i/2**refinement));
const rawMenu=finiteActionMenu({exposureTargets:grid,limitNotionalFractions:limitEnabled?[rat('0.25')]:[],
 limitOffsetsInTicks:limitEnabled?[1]:[],maximumNewOrders:limitEnabled?1:0});
// The coarse menu chooses a market adjustment OR placement. The full Cartesian
// menu remains available for refinement via --joint-actions 1.
const baseMenu:ActionMenu=opt('--joint-actions','0')==='1'?rawMenu:(a,t,r)=>rawMenu(a,t,r).filter(x=>isZero(x.action.marketQuote)||!x.action.place.length);
const menu:ActionMenu=enumerationLevel>0?(a,t,r)=>[...baseMenu(a,t,r),...exhaustiveOracleMenu(enumerationLevel)(a,t,r).map(x=>({...x,id:'enumerated:'+x.id}))]:baseMenu;
let state:OracleState={quote:rat(10000),asset:ZERO,price:rat(marks.prices[0]),previousPrice:rat(marks.prices[0]),orders:[],time:0};
let peak=10000,drawdown=0,fees=0,marketCount=0,fillCount=0,placements=0,cancellations=0,liquidations=0,exposureSum=0;
const decisions:any[]=[];
const started=performance.now();
const empty:OracleAction={marketQuote:ZERO,cancelIds:[],place:[]};
const marketFee=(amount:Rational)=>toNumber(mul(abs(amount),amount.n<0n?rules.takerFee:div(rules.takerFee,sub(ONE,rules.takerFee))));
function arrivalFees(s:OracleState){const a=prepareArrival(s,rules);if(!a.ok)throw new Error(a.error.message);
 for(const o of a.value.fills){fees+=toNumber(mul(o.notional,rules.makerFee));fillCount++;}return a.value;}
const sources=['scripts/backtest-nsf-oracle.ts','packages/bot-algo/src/oracle-execution.ts','packages/bot-algo/src/oracle-crra-policy.ts','packages/bot-algo/src/oracle-approximation.ts'];
await mkdir(out,{recursive:true});
try{await readFile(`${out}/report.json`);throw new Error('completed backtest exists');}catch(e:any){if(e.code!=='ENOENT')throw e;}
await writeFile(`${out}/plan.json`,JSON.stringify({source:root,feeScale,refinement,enumerationLevel,limitEnabled,gamma:0,terminalRequirement:"zero-asset-and-no-orders",
 initialWealth:10000,terminalTimestamp:plan.terminalTimestamp,sourceHashes:Object.fromEntries(await Promise.all(sources.map(async f=>[f,hash(await readFile(f,'utf8'))]))),
 execution:'exact rational Oracle execution at every actual second; funding factor 1; valid single terminal close',
 approximation:'finite sampled observation tree, coarse future action/mark times, finite market and order menu; no continuous global gap certified'},null,2));
for(let r=0;r<Math.min(rows.length,maxDecisions);r++){
 const row=rows[r];assert.equal(row.index,state.time);assert.equal(row.timestamp,marks.startTimestamp+state.time*1000);
 assert.equal(row.terminalTimestamp,plan.terminalTimestamp);assert(eq(state.price,rat(row.price)));
 const tree=observableTree(row.tree,rules.priceStep);
 const before=state; const arrived=arrivalFees(state);
 const tickStarted=performance.now();
 const decision=solveCrra(state,tree,{riskAversion:0,rules,menu,funding:()=>ONE},false);
 fees+=marketFee(decision.action.marketQuote);marketCount+=Number(!isZero(decision.action.marketQuote));
 placements+=decision.action.place.length;cancellations+=decision.action.cancelIds.filter(id=>arrived.state.orders.some(o=>o.id===id)).length;
 state=decision.state;
 const seconds=(performance.now()-tickStarted)/1000;
 decisions.push({index:row.index,timestamp:row.timestamp,price:row.price,expectedTerminalWealth:toNumber(decision.value!)+1,
 marketQuote:serializeRat(decision.action.marketQuote),cancelIds:decision.action.cancelIds,
 place:decision.action.place.map(o=>({...o,price:serializeRat(o.price),notional:serializeRat(o.notional)})),
 before:{quote:serializeRat(before.quote),asset:serializeRat(before.asset)},after:{quote:serializeRat(state.quote),asset:serializeRat(state.asset)},
 exposure:toNumber(exposure(state)),feasibleCandidates:decision.feasibleCandidates,solveSeconds:seconds});
 const next=r+1<rows.length?rows[r+1].index:marks.prices.length-1;
 while(state.time<next){
  const hadEquity=toNumber(equity(state))>0;
  state=advance(state,rat(marks.prices[state.time+1]),ONE,rules);
  if(hadEquity&&isZero(state.quote)&&isZero(state.asset))liquidations++;
  if(state.time<next){const a=arrivalFees(state);const e=executeAction(a,empty,rules);if(!e.ok)throw new Error(`Invalid committed fill at ${state.time}: ${e.error.message}`);state=e.value;}
  const w=toNumber(equity(state));peak=Math.max(peak,w);drawdown=Math.max(drawdown,1-w/peak);exposureSum+=Math.abs(toNumber(exposure(state)));
 }
 console.log(JSON.stringify({decision:r+1,count:rows.length,wealth:toNumber(equity(state)),markets:marketCount,fills:fillCount,solveSeconds:seconds,elapsedSeconds:(performance.now()-started)/1000}));
 await writeFile(`${out}/decisions.json`,JSON.stringify(decisions));
}
const reachedHorizon=state.time===marks.prices.length-1;
let terminalFailure:string|null=null;
if(reachedHorizon){const terminal=terminalCloseChecked(state,rules);
 if(!terminal.ok)terminalFailure=terminal.error.code+": "+terminal.error.message;
 else {arrivalFees(state);fees+=marketFee(terminal.value.marketQuote);marketCount+=Number(!isZero(terminal.value.marketQuote));state=terminal.value.state;}}
const complete=reachedHorizon && terminalFailure===null;
const wealth=toNumber(equity(state));drawdown=Math.max(drawdown,1-wealth/peak);
const report={complete,reachedHorizon,terminalFailure,terminalRequirement:"zero-asset-and-no-orders",terminalTimestamp:marks.startTimestamp+state.time*1000,terminalWealth:complete?wealth:null,returnPercent:complete?100*(wealth/10000-1):null,markedWealth:wealth,
 maximumDrawdownPercent:100*drawdown,feesPaidAtExecutionPrices:fees,marketOrders:marketCount,limitPlacements:placements,limitFills:fillCount,cancellations,liquidations,
 meanAbsoluteExposure:exposureSum/state.time,decisions:decisions.length,seconds:(performance.now()-started)/1000,
 residualAsset:toNumber(state.asset),residualQuote:toNumber(state.quote),buyHoldGrossReturnPercent:100*(marks.prices[state.time]/marks.prices[0]-1)};
await writeFile(`${out}/report.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report));
if(terminalFailure)process.exitCode=1;

}
main().catch(error=>{console.error(error);process.exitCode=1;});
