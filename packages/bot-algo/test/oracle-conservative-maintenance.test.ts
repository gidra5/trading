import assert from "node:assert/strict";
import test from "node:test";
import { assessBorrowing, assessFunding, assessMaintenance, conservativeTerminalWealth } from "../src/oracle-conservative-maintenance.js";
import { advance, effectiveAsset, effectiveEquity, equity, executeAction, prepareArrival, closeWithinCapacityChecked, terminalCloseChecked,
  type OracleAction, type OracleRules, type OracleState, type Result } from "../src/oracle-execution.js";
import { ONE, ZERO, add, eq, floorToStep, gt, isZero, lte, mul, rat, sub, type Rational } from "../src/oracle-rational.js";

const same = (actual: Rational, expected: string): void => assert.ok(eq(actual, rat(expected)), expected);
const unwrap = <T>(result: Result<T>): T => {
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`);
  return result.value;
};
const state = (quote: string, asset: string): OracleState => ({
  quote: rat(quote), asset: rat(asset), price: rat(100), previousPrice: rat(100), orders: [], time: 0,
});
const rules: OracleRules = {
  makerFee: rat("0.0002"), takerFee: rat("0.0005"), reserveLeverage: rat(2),
  entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
  quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 1,
};
const action = (marketQuote: Rational): OracleAction => ({ marketQuote, cancelIds: [], place: [] });

test("funding credits are removed for either sign, while actual funding losses are retained", () => {
  const rate = rat("0.0001");
  for (const [asset, factor, change, debit] of [
    ["1", "1.01", "1", "1.01"], ["-1", "0.99", "1", "1.01"],
    ["1", "0.99", "-1", "0.01"], ["-1", "1.01", "-1", "0.01"],
  ] as const) {
    const account = Object.freeze(state("1000", asset));
    const assessment = assessFunding(account, rat(factor), rate);
    same(assessment.actualChange, change);
    same(assessment.debit, debit);
    assert.ok(assessment.assessedChange.n < 0n);
    same(account.asset, asset);
  }
});

test("borrowing is a separate alternative, including paid longs and both types of debt", () => {
  const rate = rat("0.0001");
  const qr = rat("0.01"), ar = rat("0.02");
  same(assessBorrowing(state("1000", "1"), qr, ar, rate).actualChange, "0");
  same(assessBorrowing(state("-50", "1"), qr, ar, rate).actualChange, "-0.5");
  same(assessBorrowing(state("1000", "-1"), qr, ar, rate).actualChange, "-2");
  same(assessBorrowing(state("1000", "1"), qr, ar, rate).assessedChange, "-0.01");
  same(assessBorrowing(state("1000", "0"), qr, ar, rate).debit, "0");
});

test("assessment cannot silently include a price move or apply a negative penalty", () => {
  const account = state("1000", "1");
  assert.throws(() => assessMaintenance(account, { ...account, price: rat(101) }, rat("0.01")), /before price marking/);
  assert.throws(() => assessFunding(account, ONE, rat("-0.01")), /nonnegative/);
  assert.throws(() => assessBorrowing(account, rat("-0.01"), ZERO, ZERO), /nonnegative/);
  same(conservativeTerminalWealth(rat(10), rat(12)), "0");
  same(conservativeTerminalWealth(rat(10), rat(2)), "8");
});

test("valid residual cleanup gives a stable conservative wealth bound with exact Oracle execution", () => {
  const initial = state("10000", "0.001");
  const initialTerminal = unwrap(closeWithinCapacityChecked(initial, rules));
  assert.ok(isZero(initialTerminal.marketQuote));
  assert.ok(!isZero(initialTerminal.state.asset));

  const bought = unwrap(executeAction(unwrap(prepareArrival(initial, rules)), action(rat(-5)), rules));
  const cost = assessFunding(bought, ONE, rat("0.00001")).debit;
  same(cost, "0.000050975");
  const next = advance(bought, rat("100.01"), ONE, rules);
  const sellQuote = effectiveAsset(next, rules);
  assert.ok(lte(rules.quoteMinimum, sellQuote));
  assert.ok(lte(sellQuote, rules.quoteMaximum));
  const flat = unwrap(executeAction(unwrap(prepareArrival(next, rules)), action(sellQuote), rules));
  assert.ok(isZero(flat.asset));
  same(equity(flat), "10000.095460745125");
  same(assessFunding(flat, ONE, rat("0.00001")).debit, "0");
  const lower = conservativeTerminalWealth(equity(flat), cost);
  same(lower, "10000.095409770125");
  same(sub(rat("10000.10001"), lower), "0.004600229875");

  // No residual or pending order remains. Future marks and positive funding
  // leave this real cash account and its accumulated assessment unchanged.
  let cash = flat;
  for (const price of ["100", "100.01", "0.01", "1000000"]) {
    cash = advance(cash, rat(price), rat("0.9999"), rules);
    const actualTerminal = unwrap(closeWithinCapacityChecked(cash, rules));
    assert.ok(isZero(actualTerminal.state.asset));
    assert.ok(eq(conservativeTerminalWealth(actualTerminal.wealth, cost), lower));
    assert.ok(lte(lower, actualTerminal.wealth));
  }
  assert.ok(gt(lower, rat(10000)));
  assert.ok(eq(add(lower, cost), equity(flat)));
});

test("fixed-charge nonconvergence witness executes with a large account and capped terminal close", () => {
  const rate = rat("0.00001"), initialEquity = rat(4010000), closeCost = rat("10000/1999");
  let base = rat(4000), debt = ZERO;
  let held: OracleState = { ...state("10000", "1000"), price: base, previousPrice: base };
  // This validates the concrete execution witness. The universal optimal-value
  // upper bound and infinite-horizon nonconvergence are separately proved in Lean.
  for (let t = 0; t < 4096; t++) {
    const error = sub(held.price, base);
    const terminal = unwrap(closeWithinCapacityChecked(held, rules));
    same(terminal.marketQuote, "10000");
    assert.ok(gt(terminal.state.asset, ZERO));
    const value = conservativeTerminalWealth(terminal.wealth, debt);
    assert.ok(eq(value, sub(add(initialEquity, mul(rat(1000), error)), closeCost)));
    assert.ok(gt(value, sub(initialEquity, rat(16))));
    assert.ok(lte(value, add(initialEquity, rat(20))));
    if (t % 2 === 1) assert.ok(gt(value, add(initialEquity, rat(4))));
    else assert.ok(lte(value, initialEquity));

    const rebalanced = unwrap(executeAction(unwrap(prepareArrival(held, rules)), action(ZERO), rules));
    const assessment = assessFunding(rebalanced, ONE, rate);
    assert.ok(gt(assessment.debit, ZERO));
    assert.ok(gt(ZERO, assessment.assessedChange));
    debt = add(debt, assessment.debit);
    base = add(base, mul(rate, held.price));
    const price = add(floorToStep(base, rules.priceStep), rat((t + 1) % 2 === 1 ? "0.02" : "0"));
    assert.ok(gt(price, held.price));
    held = advance(rebalanced, price, ONE, rules);
    same(held.quote, "10000");
    same(held.asset, "1000");
  }
});

test("bounded-price terminal witnesses keep the same excess value after arbitrarily delayed cash waiting", () => {
  const initial = rat(1000000), factor = rat("1.0002"), rate = rat("0.00001");
  let reference: Rational | undefined;
  for (const terminalTick of [21, 23, 101, 1021]) {
    let s = state("1000000", "0"), debt = ZERO, fills = 0;
    for (let t = 0; t < terminalTick; t++) {
      // Wait in cash, then run the same last 21 ticks. Buy at 100 after
      // placing strictly below 100.04; at most one order is ever active.
      const arrival = unwrap(prepareArrival(s, rules));
      fills += arrival.fills.length;
      const place = t >= terminalTick - 20 && t % 2 === 1 ? [{
        id: String(t), side: "buy" as const, price: rat(100), notional: rat(10000), placedAt: t,
      }] : [];
      const r = unwrap(executeAction(arrival, { marketQuote: ZERO, cancelIds: [], place }, rules));
      const assessment = assessFunding(r, factor, rate);
      if (!isZero(r.asset)) assert.ok(gt(ZERO, assessment.assessedChange));
      debt = add(debt, assessment.debit);
      s = advance(r, rat((t + 1) % 2 === 1 ? "100.04" : "100"), factor, rules);
      assert.ok(lte(s.price, rat("100.04")));
      assert.ok(gt(equity(s), ZERO));
    }
    assert.equal(fills, 10);
    assert.equal(terminalCloseChecked(s, rules).ok, false,
      "The historical marked-residual witness is excluded by the flat-terminal constraint");
    const terminal = unwrap(closeWithinCapacityChecked(s, rules));
    same(terminal.marketQuote, "10000");
    assert.ok(gt(terminal.state.asset, ZERO));
    const value = conservativeTerminalWealth(terminal.wealth, debt);
    assert.ok(gt(value, add(initial, rat("5.02"))));
    assert.ok(lte(value, add(initial, rat("5.03"))));
    if (reference) assert.ok(eq(value, reference));
    reference = value;
    // Charging the fee for all outstanding inventory removes this witness's
    // gain. This is a valuation comparison, not an executable oversized close.
    assert.ok(gt(initial, conservativeTerminalWealth(effectiveEquity(s, rules), debt)));
  }
});
