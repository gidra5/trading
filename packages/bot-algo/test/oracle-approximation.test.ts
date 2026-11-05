import assert from 'node:assert/strict';
import test from 'node:test';
import { exhaustiveOracleMenu } from '../src/oracle-approximation.js';
import { prepareArrival, type OracleState, type OracleRules } from '../src/oracle-execution.js';
import { observableTree, solveCrra } from '../src/oracle-crra-policy.js';
import { ONE,ZERO,rat,serializeRat,toNumber } from '../src/oracle-rational.js';
const rules:OracleRules={makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),entryLower:rat(-2),entryUpper:rat(2),maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:rat(1),quoteMaximum:rat(200),priceStep:rat('0.01'),maxOrders:1};
const state:OracleState={quote:rat(100),asset:ZERO,price:rat(100),previousPrice:rat(100),orders:[],time:0};
test('rational refinement includes earlier levels, non-dyadic sizes, and joint actions',()=>{
 const a=prepareArrival(state,rules);assert(a.ok);const tree={children:[]};
 const one=exhaustiveOracleMenu(1)(a.value,tree,rules),three=exhaustiveOracleMenu(3)(a.value,tree,rules);
 const ids=new Set(three.map(x=>x.id));assert.equal(ids.size,three.length);
 assert(one.every(x=>ids.has(x.id)));
 assert(three.some(x=>serializeRat(x.action.marketQuote)==='100/3'));
 assert(three.some(x=>x.action.marketQuote.n!==0n&&x.action.place.length===1));
});
test('exact finite optimization is monotone under rational menu refinement',()=>{
 const tree=observableTree({children:[{probability:'1/2',price:102},{probability:'1/2',price:101}]},rules.priceStep);
 const first=solveCrra(state,tree,{riskAversion:0,rules,menu:exhaustiveOracleMenu(1),funding:()=>ONE});
 const second=solveCrra(state,tree,{riskAversion:0,rules,menu:exhaustiveOracleMenu(2),funding:()=>ONE});
 assert(toNumber(second.value!)>=toNumber(first.value!));
});
