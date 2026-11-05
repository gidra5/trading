import assert from 'node:assert/strict';
import test from 'node:test';
import { observableTree, finiteActionMenu, solveCrra, NoFeasiblePolicy, type SolverConfig } from '../src/oracle-crra-policy.js';
import { type OracleAction, type OracleRules, type OracleState, prepareArrival } from '../src/oracle-execution.js';
import { shadowPathUpper, shadowTreeUpper, martingaleShadowUpper, continuousShadowUpper, refinedShadowUpper } from '../src/oracle-shadow-tail.js';
import { ZERO, ONE, rat, add, eq, gte, gt, lt } from '../src/oracle-rational.js';

const rules: OracleRules = { makerFee:rat('0.0002'),takerFee:rat('0.0005'),reserveLeverage:rat(2),
  entryLower:rat(-2),entryUpper:rat(2),maintenanceLower:rat(-3),maintenanceUpper:rat(3),
  quoteMinimum:ONE,quoteMaximum:rat(100),priceStep:rat('0.01'),maxOrders:1 };
const state: OracleState = { quote:rat(100),asset:ZERO,price:rat(100),previousPrice:rat(100),time:0,orders:[] };
const hold: OracleAction = { marketQuote:ZERO,cancelIds:[],place:[] };
const config: SolverConfig = { riskAversion:0,rules,funding:()=>ONE,
  menu:finiteActionMenu({exposureTargets:[-1,0,1].map(rat),limitNotionalFractions:[rat('0.25')],limitOffsetsInTicks:[1],maximumNewOrders:1}) };

test('flat prices certify cash exactly, with no phantom gain from repeated short sales', () => {
  const frames = Array.from({length:48},()=>({price:rat(100),previousPrice:rat(100),cumulativeFunding:ONE}));
  const result = shadowPathUpper(state,frames,rules);
  assert(eq(result.upper,rat(100)));
  assert(gt(result.referencePrice,ZERO));
});

test('a narrow fee-covered price band also certifies no trading gain', () => {
  const frames = [100.01,100,100.01,100].map((p,i)=>({price:rat(p),previousPrice:rat(i%2?100.01:100),cumulativeFunding:ONE}));
  assert(eq(shadowPathUpper(state,frames,rules).upper,rat(100)));
  const free = shadowPathUpper(state,frames,{...rules,makerFee:ZERO,takerFee:ZERO});
  assert(gt(free.upper,rat(100)));
});

test('shadow upper covers exact Oracle continuations, root fills, both directions and funding', () => {
  const tree = observableTree({children:[
    {probability:'1/2',price:99,children:[{probability:1,price:105,children:[{probability:1,price:100}]}]},
    {probability:'1/2',price:101,children:[{probability:1,price:95,children:[{probability:1,price:100}]}]},
  ]},rules.priceStep);
  const pending: OracleState = {...state,time:1,previousPrice:rat(101),orders:[
    {id:'prior-buy',side:'buy',price:rat('100.5'),notional:rat(20),placedAt:0},
  ]};
  let checked=0,withLimits=0;
  for (const initial of [state,pending]) for (const funding of [()=>ONE,(t:number)=>rat(t%2?'1.001':'0.999')]) {
    const arrived=prepareArrival(initial,rules); assert(arrived.ok);
    for (const first of config.menu(arrived.value,tree,rules)) {
      const fixed: SolverConfig={...config,funding,menu:(s,t,r)=>s.state.time===initial.time?[first]:config.menu(s,t,r)};
      let decision;
      try { decision=solveCrra(initial,tree,fixed); }
      catch(error) { if(error instanceof NoFeasiblePolicy)continue; throw error; }
      assert(decision.value!==null);
      const upper=shadowTreeUpper(initial,tree,{...config,funding},first.action);
      assert(gte(upper.upper,add(decision.value,ONE)));
      assert(gte(upper.cashUpper,upper.upper));
      checked++;
      if(first.action.place.length)withLimits++;
    }
  }
  assert(checked>10&&withLimits>0);
});

test('zero-account liquidation is covered even when shadow liabilities would be negative', () => {
  const short={...state,quote:rat(200),asset:rat(-1)};
  const tree=observableTree({children:[{probability:1,price:1000}]},rules.priceStep);
  const decision=solveCrra(short,tree,{...config,menu:()=>[{id:'hold',action:hold}]});
  assert(eq(add(decision.value!,ONE),ZERO));
  assert(gte(shadowTreeUpper(short,tree,config,hold).upper,ZERO));
});

test('funding and probability domains are checked before reporting a bound', () => {
  assert.throws(()=>shadowPathUpper(state,[{price:ONE,previousPrice:ONE,cumulativeFunding:ZERO}],rules),/Positive/);
  const tree={children:[{probability:rat(2),price:rat(100),children:[]}]};
  assert.throws(()=>shadowTreeUpper(state,tree,config,hold),/normalized/);
  const canonical=observableTree({children:[{probability:1,price:100}]},rules.priceStep);
  assert.throws(()=>shadowTreeUpper(state,canonical,{...config,funding:()=>ZERO},hold),/Positive/);
  assert.throws(()=>shadowTreeUpper(state,canonical,{...config,riskAversion:2},hold),/gamma=0/);
  const nonflat=shadowPathUpper(state,[{price:rat(110),previousPrice:rat(100),cumulativeFunding:ONE}],rules);
  assert(lt(nonflat.upper,rat(200)));
});

test('conditional shadow removes clairvoyant gains on a symmetric final move', () => {
  const tree=observableTree({children:[{probability:'1/2',price:99},{probability:'1/2',price:101}]},rules.priceStep);
  const bound=martingaleShadowUpper(state,tree,config);
  assert(bound!==null);
  assert(eq(bound.upper,state.quote));
  assert(gt(shadowTreeUpper(state,tree,config).upper,bound.upper));
  assert(eq(continuousShadowUpper(state,tree,config),state.quote));
  const solved=solveCrra(state,tree,config);
  assert(eq(add(solved.value!,ONE),bound.upper));
});

test('unfixed root, pending fills, and nonunit funding are covered by both bounds', () => {
  const tree=observableTree({children:[
    {probability:'1/3',price:99,children:[{probability:1,price:103}]},
    {probability:'2/3',price:101,children:[{probability:1,price:97}]},
  ]},rules.priceStep);
  const pending:OracleState={...state,time:1,previousPrice:rat(101),orders:[
    {id:'buy',side:'buy',price:rat('100.5'),notional:rat(20),placedAt:0},
  ]};
  for(const initial of [state,pending,{...state,quote:rat(200),asset:rat(-1)}]) {
    const funded={...config,funding:(t:number)=>rat(t%2?'1.001':'0.999')};
    const exact=add(solveCrra(initial,tree,funded).value!,ONE);
    const path=shadowTreeUpper(initial,tree,funded).upper;
    const martingale=martingaleShadowUpper(initial,tree,funded);
    assert(martingale!==null);
    assert(gte(martingale.upper,exact));
    assert(gte(path,exact));
    assert(gte(continuousShadowUpper(initial,tree,funded),exact));
  }
});

test('uncovered liquidation edges and gross leverage fall back to the pathwise bound', () => {
  const short={...state,quote:rat(200),asset:rat(-1)};
  const tree=observableTree({children:[{probability:1,price:1000}]},rules.priceStep);
  assert.equal(martingaleShadowUpper(short,tree,config,hold),null);
  assert(eq(continuousShadowUpper(short,tree,config,hold),shadowTreeUpper(short,tree,config,hold).upper));
  const small=observableTree({children:[{probability:1,price:100}]},rules.priceStep);
  assert.equal(martingaleShadowUpper({...state,quote:rat(-300),asset:rat(4)},small,config),null);
  assert.throws(()=>martingaleShadowUpper(state,{children:[{probability:rat(2),price:rat(100),children:[]}]},config),/normalized/);
});

test('Bellman refinement is nested and becomes the exact finite-menu value at full depth', () => {
  const tree=observableTree({children:[
    {probability:'1/2',price:95,children:[{probability:1,price:100}]},
    {probability:'1/2',price:105,children:[{probability:1,price:100}]},
  ]},rules.priceStep);
  for(const first of [undefined,hold]) {
    const fixed=first?{...config,menu:(s:Parameters<SolverConfig['menu']>[0],t:Parameters<SolverConfig['menu']>[1],r:OracleRules)=>
      s.state.time===0?[{id:'hold',action:first}]:config.menu(s,t,r)}:config;
    const exact=add(solveCrra(state,tree,fixed).value!,ONE);
    let prior=continuousShadowUpper(state,tree,config,first);
    for(let depth=0;depth<=2;depth++) {
      const bound=refinedShadowUpper(state,tree,config,depth,first);
      assert(bound!==null&&gte(bound,exact)&&gte(prior,bound));
      prior=bound;
    }
    assert(eq(prior,exact));
  }
});

test('refinement preserves infeasible terminal dust instead of giving it a cash value', () => {
  const dust={...state,asset:rat('0.001')};
  assert.equal(refinedShadowUpper(dust,{children:[]},config,0),null);
  const tree=observableTree({children:[{probability:1,price:100}]},rules.priceStep);
  const fixed={...config,menu:()=>[{id:'hold',action:hold}]};
  assert.equal(refinedShadowUpper(dust,tree,fixed,1),null);
  assert.throws(()=>solveCrra(dust,tree,fixed),NoFeasiblePolicy);
});
