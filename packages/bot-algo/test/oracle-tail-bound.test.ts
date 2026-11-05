import assert from 'node:assert/strict';
import test from 'node:test';
import { finiteActionMenu, observableTree, solveCrra, NoFeasiblePolicy, type SolverConfig } from '../src/oracle-crra-policy.js';
import { prepareArrival, terminalCloseChecked, type OracleAction, type OracleRules, type OracleState } from '../src/oracle-execution.js';
import { quotePrefixUpper, relativeTailEnvelope, necessaryPerTickDepth } from '../src/oracle-tail-bound.js';
import { ONE, ZERO, rat, add, mul, lt, eq, gte } from '../src/oracle-rational.js';

const rules: OracleRules = { makerFee: rat('0.0002'), takerFee: rat('0.0005'), reserveLeverage: rat(2),
  entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
  quoteMinimum: ONE, quoteMaximum: rat(100), priceStep: rat('0.01'), maxOrders: 1 };
const initial: OracleState = { quote: rat(100), asset: ZERO, price: rat(100), previousPrice: rat(100), time: 0, orders: [] };
const hold: OracleAction = { marketQuote: ZERO, cancelIds: [], place: [] };
const config: SolverConfig = { riskAversion: 0, rules, funding: () => ONE,
  menu: finiteActionMenu({ exposureTargets: [-1,0,1].map(rat), limitNotionalFractions: [rat('0.25')],
    limitOffsetsInTicks: [1], maximumNewOrders: 1 }) };

test('the prefix upper score retains the chosen first action', () => {
  const tree = observableTree({ children: [{ probability: 1, price: 100 }] }, rules.priceStep);
  const c: SolverConfig = { ...config, menu: () => [
    { id: 'wait', action: hold }, { id: 'short', action: { ...hold, marketQuote: rat(100) } },
  ] };
  assert(eq(quotePrefixUpper(initial,tree,c,hold)!,rat(200)));
  assert(eq(quotePrefixUpper(initial,tree,c)!,rat(300)));
  const closed = solveCrra(initial,tree,{ ...c, menu: () => [{ id: 'wait', action: hold }] });
  assert(eq(add(closed.value!,ONE),initial.quote)); // Gamma-zero wealth, not utility.
});

test('a frontier upper bound permits inventory that cannot close at the cutoff', () => {
  const residual = { ...initial, quote: rat(1000), asset: rat(2) };
  assert.equal(terminalCloseChecked(residual,rules).ok,false);
  assert(eq(quotePrefixUpper(residual,{ children: [] },config)!,rat(1100)));
  const negativeQuote = { ...initial, quote: rat(-10), asset: ONE };
  assert(eq(quotePrefixUpper(negativeQuote,{ children: [] },config)!,rat(100)));
});

test('the prefix envelope dominates exact full continuations with fees, funding and limit fills', () => {
  const full = observableTree({ children: [
    { probability: '1/2', price: 99, children: [{ probability: 1, price: 110 }] },
    { probability: '1/2', price: 101, children: [{ probability: 1, price: 90 }] },
  ] },rules.priceStep);
  const prefix = { children: full.children.map(branch => ({ ...branch, children: [] })) };
  const funded = { ...config, funding: () => rat('0.9999') };
  const arrival = prepareArrival(initial,rules);
  assert(arrival.ok);
  let checked = 0, checkedLimits = 0;
  for (const named of config.menu(arrival.value,prefix,rules)) {
    const fixed: SolverConfig = { ...funded, menu: (s,t,r) => s.state.time === 0 ? [named] : config.menu(s,t,r) };
    let long;
    try { long = solveCrra(initial,full,fixed); }
    catch (error) { if (error instanceof NoFeasiblePolicy) continue; throw error; }
    const upper = quotePrefixUpper(initial,prefix,funded,named.action);
    assert(upper !== null && long.value !== null);
    assert(gte(add(upper,rules.quoteMaximum),add(long.value,ONE)));
    checked++;
    if (named.action.place.length) checkedLimits++;
  }
  assert(checked > 2 && checkedLimits > 0);
});

test('relative units and the strict per-second 0.1% threshold are exact', () => {
  assert(eq(relativeTailEnvelope(rat(10000),rat(40000),169200,rat(10000)),rat(169203)));
  assert.throws(() => relativeTailEnvelope(ZERO,ONE,1,ONE));
  assert.throws(() => relativeTailEnvelope(ONE,ZERO,1,ONE));
  assert.throws(() => relativeTailEnvelope(ONE,ONE,0.5,ONE));
  const total = 172800, cap = rat(10000), cash = rat(10000), tolerance = rat('0.001');
  const h = necessaryPerTickDepth(total,cash,cap,tolerance);
  assert.equal(h,172628);
  const testDepth = (k: number) => lt(mul(rat(total-k),cap),mul(tolerance,add(cash,mul(rat(k+1),cap))));
  assert.equal(testDepth(h-1),false);
  assert.equal(testDepth(h),true);
  // Equality does not pass a strict epsilon threshold.
  assert.equal(necessaryPerTickDepth(8,ONE,ONE,ONE),4);
});

test('the quote envelope rejects unsupported risk and unnormalized laws', () => {
  assert.throws(() => quotePrefixUpper(initial,{ children: [] },{ ...config, riskAversion: 2 }), /gamma=0/);
  assert.throws(() => quotePrefixUpper(initial,{ children: [{ price: rat(100), probability: rat(2), children: [] }] },config), /normalized/);
});
