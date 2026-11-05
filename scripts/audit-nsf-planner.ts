import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {observableTree,solveCrra,finiteActionMenu,type ActionMenu} from '../packages/bot-algo/src/oracle-crra-policy.js';
import {exhaustiveOracleMenu} from '../packages/bot-algo/src/oracle-approximation.js';
import {type OracleRules,type OracleState,executeQuote,advance,terminalCloseChecked,equity} from '../packages/bot-algo/src/oracle-execution.js';
import {rat,ONE,ZERO,toNumber,isZero} from '../packages/bot-algo/src/oracle-rational.js';
async function main(){
 const root='data/backtests/oracle-nsf-planning';
 const rules:OracleRules={makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),entryLower:rat(-2),entryUpper:rat(2),maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:rat(5),quoteMaximum:rat(10000),priceStep:rat('0.01'),maxOrders:1};
 const certs:any[]=[],reports:any[]=[];
 for(const source of [root,root+'-branches4']){
  const row=JSON.parse((await readFile(source+'/forecasts.jsonl','utf8')).split('\n')[0]!);
  const tree=observableTree(row.tree,rules.priceStep);
  const state:OracleState={quote:rat(10000),asset:ZERO,price:rat(row.price),previousPrice:rat(row.price),orders:[],time:0};
  let previous=-Infinity;
  for(const level of [1,2]){
   const raw=finiteActionMenu({exposureTargets:Array.from({length:2**level+1},(_,i)=>rat(-2+4*i/2**level)),limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1});
   const menu:ActionMenu=(a,t,r)=>raw(a,t,r).filter(x=>isZero(x.action.marketQuote)||!x.action.place.length);
   const result=solveCrra(state,tree,{riskAversion:0,rules,menu,funding:()=>ONE},true);
   const value=toNumber(result.value!)+1;assert(value>=previous);previous=value;
   certs.push(result.certificate);reports.push({source,actionLevel:level,initialExpectedTerminalWealth:value,feasibleRootActions:result.feasibleCandidates});
  }
 }
 const marks=JSON.parse(await readFile(root+'/marks.json','utf8'));
 const first=rat(marks.prices[0]),last=rat(marks.prices.at(-1));
 let hold:OracleState={quote:rat(10000),asset:ZERO,price:first,previousPrice:first,orders:[],time:0};
 hold=executeQuote(hold,rat(-10000),rules.takerFee);
 hold=advance(hold,last,ONE,rules);
 const terminal=terminalCloseChecked(hold,rules);
 await writeFile(root+'/certificates.jsonl',certs.map(x=>JSON.stringify(x)).join('\n')+'\n');
 await writeFile(root+'/search-audit.json',JSON.stringify({rootRefinement:reports,
  terminalRequirement:"zero-asset-and-no-orders",buyHoldFeasible:terminal.ok,
  buyHoldNetTerminalWealth:terminal.ok?toNumber(terminal.value.wealth):null,buyHoldNetReturnPercent:terminal.ok?100*(toNumber(terminal.value.wealth)/10000-1):null,
  scope:'finite supplied trees/menus; certificate checker verifies arithmetic and maximization, not forecast accuracy or continuous coverage'},null,2));
 console.log(JSON.stringify(reports));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
