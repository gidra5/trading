import assert from 'node:assert/strict';
import test from 'node:test';
import { finiteActionMenu, observableTree, solveCrra, NoFeasiblePolicy } from '../src/oracle-crra-policy.js';
import { type OracleState, type OracleRules } from '../src/oracle-execution.js';
import { logInterval, expectedLog, solveLogCrra, logShadowUpper, refinedLogShadowUpper,
  belowCeTolerance, type LogSolverConfig } from '../src/oracle-log-policy.js';
import { ZERO, ONE, rat, add, sub, div, eq, gte, lt, toNumber } from '../src/oracle-rational.js';

const rules:OracleRules={makerFee:ZERO,takerFee:ZERO,reserveLeverage:rat(2),entryLower:ZERO,entryUpper:ONE,
  maintenanceLower:rat(-3),maintenanceUpper:rat(3),quoteMinimum:ZERO,quoteMaximum:rat(1000),priceStep:rat('0.01'),maxOrders:1};
const state:OracleState={quote:rat(100),asset:ZERO,price:rat(100),previousPrice:rat(100),time:0,orders:[]};
const hold={marketQuote:ZERO,cancelIds:[],place:[]};
const config:LogSolverConfig={riskAversion:1,wealthUnit:rat(100),rules,funding:()=>ONE,
  menu:finiteActionMenu({exposureTargets:[0,.5,1].map(rat),limitNotionalFractions:[],limitOffsetsInTicks:[],maximumNewOrders:0})};

test('log intervals enclose high-precision reference constants and extreme rational values',()=>{
  const references=[
    ['2','0.6931471805599453094172321214581765680755001343602552'],
    ['1/2','-0.6931471805599453094172321214581765680755001343602552'],
    ['3','1.0986122886681096913952452369225257046474905578227494'],
    ['10','2.3025850929940456840179914546843642076011014886287729'],
    ['0.001','-6.9077552789821370520539743640530926228033044658863189'],
  ];
  for(const [x,truth] of references) {
    const value=logInterval(rat(x!)),reference=rat(truth!);
    assert(gte(reference,value.lower)&&gte(value.upper,reference));
    assert(toNumber(sub(value.upper,value.lower))<1e-14);
  }
  assert(eq(logInterval(ONE).lower,ZERO)&&eq(logInterval(ONE).upper,ZERO));
  const large=logInterval(rat('1e1000')),small=logInterval(rat('1e-1000'));
  assert(gte(add(large.upper,small.upper),ZERO)&&gte(ZERO,add(large.lower,small.lower)));
  assert.throws(()=>logInterval(ZERO),/positive/);
  assert.throws(()=>logInterval(rat(-1)),/positive/);
});

test('log Bellman selects the half-exposure Kelly action, while gamma zero selects full exposure',()=>{
  const tree=observableTree({children:[{probability:'1/2',price:200},{probability:'1/2',price:50}]},rules.priceStep);
  const log=solveLogCrra(state,tree,config);
  const neutral=solveCrra(state,tree,{...config,riskAversion:0});
  assert(eq(log.action.marketQuote,rat(-50)));
  assert(eq(neutral.action.marketQuote,rat(-100)));
  assert(eq(log.expectedWealth,rat('112.5')));
  assert(log.value!==null);
  const exact=.5*Math.log(1.5)+.5*Math.log(.75);
  assert(Math.abs(toNumber(log.value.lower)-exact)<1e-14);
  assert(toNumber(sub(log.value.upper,log.value.lower))<1e-14);
  assert(gte(logShadowUpper(state,tree,config)!,log.value.lower));
});

test('log policy and continuation differences are independent of the wealth unit',()=>{
  const tree=observableTree({children:[{probability:'1/2',price:200},{probability:'1/2',price:50}]},rules.priceStep);
  const a=solveLogCrra(state,tree,config),b=solveLogCrra(state,tree,{...config,wealthUnit:ONE});
  assert(eq(a.action.marketQuote,b.action.marketQuote));
  assert(a.value!==null&&b.value!==null);
  assert(Math.abs(toNumber(sub(b.value.lower,a.value.lower))-Math.log(100))<1e-13);
});

test('positive probability of ruin has bottom utility; zero probability does not contaminate the expectation',()=>{
  assert.equal(expectedLog([{probability:ONE,value:null}]),null);
  const zero=logInterval(ONE);
  assert(eq(expectedLog([{probability:ONE,value:zero},{probability:ZERO,value:null}])!.lower,ZERO));
  assert.throws(()=>expectedLog([{probability:rat(2),value:zero}]),/normalized/);
  const short={...state,quote:rat(200),asset:rat(-1)};
  const tree=observableTree({children:[{probability:ONE,price:1000}]},rules.priceStep);
  const allRuin=solveLogCrra(short,tree,{...config,menu:()=>[{id:'hold',action:hold}]});
  assert.equal(allRuin.value,null);
  assert(eq(allRuin.expectedWealth,ZERO));
});

test('log shadow and nested refinement cover fee/funding/limit continuations through a valid flat close',()=>{
  const feeRules={...rules,makerFee:rat('0.0002'),takerFee:rat('0.0005'),entryLower:rat(-2),entryUpper:rat(2),quoteMinimum:ONE};
  const active:LogSolverConfig={...config,rules:feeRules,funding:(t)=>rat(t%2?'1.001':'0.999'),
    menu:finiteActionMenu({exposureTargets:[-1,0,1].map(rat),limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1})};
  const tree=observableTree({children:[
    {probability:'1/2',price:99,children:[{probability:1,price:103}]},
    {probability:'1/2',price:101,children:[{probability:1,price:97}]},
  ]},rules.priceStep);
  const exact=solveLogCrra(state,tree,active);assert(exact.value!==null);
  let upper=logShadowUpper(state,tree,active);assert(upper!==null);
  for(let d=0;d<=2;d++) {
    const refined=refinedLogShadowUpper(state,tree,active,d);
    assert(refined!==null&&gte(refined,exact.value.lower)&&gte(upper,refined));
    upper=refined;
  }
  assert(eq(upper,exact.value.upper));
  const dust={...state,asset:rat('0.001')};
  assert.equal(refinedLogShadowUpper(dust,{children:[]},active,0),null);
  assert.throws(()=>solveLogCrra(dust,tree,{...active,menu:()=>[{id:'hold',action:hold}]}),NoFeasiblePolicy);
});

test('strict certainty-equivalent tolerance uses interval comparisons, not floating exponentials',()=>{
  const eps=rat('0.001'),threshold=logInterval(add(ONE,eps));
  assert(belowCeTolerance(ZERO,eps));
  assert(!belowCeTolerance(threshold.upper,eps));
  assert(!belowCeTolerance(threshold.lower,eps));
  assert(belowCeTolerance(div(threshold.lower,rat(2)),eps));
  assert(lt(logInterval(rat('1.0005')).upper,threshold.lower));
});
