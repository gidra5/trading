import assert from "node:assert/strict";
import test from "node:test";
import { ONE, ZERO, add, div, eq, mul, neg, rat, serializeRat, sub } from "../src/oracle-rational.js";
import {
  accountAllowed, advance, effectiveEquity, equity, executeAction, executeQuote, exposure,
  prepareArrival, requestForTarget, reserved, survives, terminalClose, closeWithinCapacityChecked,
  type OracleAction, type OracleOrder, type OracleRules, type OracleState, type Result,
} from "../src/oracle-execution.js";

const rules = (changes: Partial<OracleRules> = {}): OracleRules => ({
  makerFee: rat("0.0002"), takerFee: rat("0.0005"), reserveLeverage: rat(5),
  entryLower: rat(-5), entryUpper: rat(5), maintenanceLower: rat(-10), maintenanceUpper: rat(10),
  quoteMinimum: ZERO, quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 8,
  ...changes,
});
const state = (changes: Partial<OracleState> = {}): OracleState => ({
  quote: rat(100), asset: ZERO, price: rat(100), previousPrice: rat(100), orders: [], time: 1,
  ...changes,
});
const order = (changes: Partial<OracleOrder> = {}): OracleOrder => ({
  id: "buy", side: "buy", price: rat(95), notional: rat(50), placedAt: 0, ...changes,
});
const action = (changes: Partial<OracleAction> = {}): OracleAction => ({
  marketQuote: ZERO, cancelIds: [], place: [], ...changes,
});
function ok<T>(result: Result<T>): T {
  if (!result.ok) assert.fail(JSON.stringify(result.error));
  return result.value;
}
const exact = (actual: ReturnType<typeof rat>, expected: string | number) =>
  assert.equal(serializeRat(actual), serializeRat(rat(expected)));

test("received-amount fees and signed quote execution match both Oracle branches", () => {
  const config = rules();
  const buy = executeQuote(state(), rat(-100), config.takerFee);
  exact(buy.quote, 0);
  exact(buy.asset, "0.9995");
  exact(equity(buy), "99.95");
  const sold = executeQuote(buy, rat("99.900025"), config.takerFee);
  exact(sold.asset, 0);
  exact(sold.quote, "99.900025");
  assert.equal(executeQuote(buy, ZERO, config.takerFee), buy);
});

test("fee-aware targeting reaches requested long and short exposure exactly", () => {
  const config = rules();
  for (const target of [rat(-5), rat(-1), ZERO, rat("0.5"), ONE, rat(5)]) {
    const source = state();
    const request = requestForTarget(source, target, config);
    const next = ok(executeAction(ok(prepareArrival(source, config)), action({ marketQuote: request }), config));
    assert.ok(eq(exposure(next), target));
    assert.ok(accountAllowed(next, config));
  }
});

test("funding multiplies signed held assets without changing quote or resting order notional", () => {
  const config = rules();
  const book = [order()];
  const short = state({ quote: rat(200), asset: rat(-1), orders: book });
  const next = advance(short, rat(100), rat("0.9"), config);
  exact(next.quote, 200);
  exact(next.asset, "-0.9");
  exact(equity(next), 110);
  assert.equal(next.orders, book);
  exact(next.orders[0].notional, 50);
  assert.equal(next.time, short.time + 1);
});

test("coasting retains maintenance-safe exposure drift beyond entry bounds", () => {
  const config = rules({ entryUpper: rat(2), maintenanceUpper: rat(4) });
  const drifted = state({ quote: rat(-100), asset: rat(2), price: rat(90), previousPrice: rat(100) });
  assert.ok(eq(exposure(drifted), rat("2.25")));
  assert.equal(survives(drifted, config), true);
  assert.equal(executeAction(ok(prepareArrival(drifted, config)), action(), config).ok, true);
  const adjusted = executeAction(ok(prepareArrival(drifted, config)), action({ marketQuote: rat(-1) }), config);
  assert.equal(adjusted.ok, false);
  if (!adjusted.ok) assert.equal(adjusted.error.code, "entry-exposure");
});

test("downward arrivals fill only mature crossed buys, in stored book order at their limit prices", () => {
  const config = rules();
  const first = order({ id: "first", price: rat(95), notional: rat(95) });
  const second = order({ id: "second", price: rat(98), notional: rat(98) });
  const sell = order({ id: "sell", side: "sell", price: rat(95) });
  const fresh = order({ id: "fresh", placedAt: 1 });
  const source = state({ quote: rat(1000), price: rat(90), previousPrice: rat(100), orders: [first, second, sell, fresh] });
  const prepared = ok(prepareArrival(source, config));
  assert.deepEqual(prepared.fills, [first, second]);
  assert.deepEqual(prepared.state.orders, [sell, fresh]);
  exact(prepared.state.quote, 807);
  exact(prepared.state.asset, "1.9996");
  exact(prepared.state.price, 90);
  exact(prepared.consumedQuote, 193);
  assert.equal(source.orders.length, 4);
});

test("flat moves never fill limits; upward endpoint touches fill only sells", () => {
  const config = rules();
  const sell = order({ id: "sell", side: "sell", price: rat(100), notional: rat(10) });
  const buy = order({ price: rat(100) });
  assert.equal(ok(prepareArrival(state({ orders: [buy, sell] }), config)).fills.length, 0);
  const up = ok(prepareArrival(state({ previousPrice: rat(90), orders: [buy, sell] }), config));
  assert.deepEqual(up.fills, [sell]);
  exact(up.consumedQuote, "9.998");
});

test("cancellation cannot undo a fill already triggered at arrival", () => {
  const config = rules();
  const source = state({ quote: rat(1000), price: rat(90), orders: [order()] });
  const prepared = ok(prepareArrival(source, config));
  const next = ok(executeAction(prepared, action({ cancelIds: ["buy"] }), config));
  exact(next.quote, 950);
  assert.ok(eq(next.asset, prepared.state.asset));
  assert.deepEqual(next.orders, []);
});

test("new passive orders activate only on the next arrival and retain fixed notional", () => {
  const config = rules();
  const fresh = order({ placedAt: 1 });
  const placed = ok(executeAction(ok(prepareArrival(state(), config)), action({ place: [fresh] }), config));
  assert.equal(ok(prepareArrival(placed, config)).fills.length, 0);
  const next = advance(placed, rat(95), ONE, config);
  assert.deepEqual(ok(prepareArrival(next, config)).fills, [fresh]);
});

test("shared quote cap counts full arrival fills and market trade without netting", () => {
  const config = rules({ quoteMaximum: rat(100) });
  const source = state({ quote: rat(1000), price: rat(90), orders: [order({ notional: rat(60) })] });
  const prepared = ok(prepareArrival(source, config));
  const rejected = executeAction(prepared, action({ marketQuote: rat(50) }), config);
  assert.equal(rejected.ok, false);
  if (!rejected.ok) assert.equal(rejected.error.code, "aggregate-quote-cap");
  assert.equal(executeAction(prepared, action({ marketQuote: rat(40) }), config).ok, true);
  const overbooked = prepareArrival({ ...source, orders: [order({ id: "a", notional: rat(60) }), order({ id: "b", notional: rat(60) })] }, config);
  assert.equal(overbooked.ok, false);
});

test("each committed fill must obey minimum, including net sell quote size", () => {
  const config = rules({ quoteMinimum: rat(10) });
  const sell = order({ side: "sell", price: rat(110), notional: rat(10) });
  const result = prepareArrival(state({ price: rat(110), orders: [sell] }), config);
  assert.equal(result.ok, false); // Received 9.998 quote is below minimum.
  if (!result.ok) assert.equal(result.error.code, "fill-size");
  assert.equal(executeAction(ok(prepareArrival(state(), config)), action(), config).ok, true);
});

test("placements enforce strict passivity, price grid, current timestamp, unique IDs and slots", () => {
  const config = rules({ maxOrders: 1 });
  const prepared = ok(prepareArrival(state(), config));
  for (const bad of [
    order({ price: rat(100), placedAt: 1 }),
    order({ price: rat("95.001"), placedAt: 1 }),
    order({ placedAt: 0 }),
  ]) assert.equal(executeAction(prepared, action({ place: [bad] }), config).ok, false);
  assert.equal(executeAction(prepared, action({ place: [order({ placedAt: 1 }), order({ id: "b", placedAt: 1 })] }), config).ok, false);
  const withBook = ok(prepareArrival(state({ orders: [order()] }), config));
  assert.equal(executeAction(withBook, action({ place: [order({ placedAt: 1 })] }), rules()).ok, false);
  assert.equal(executeAction(withBook, action({ cancelIds: ["buy"], place: [order({ placedAt: 1 })] }), config).ok, true);
});

test("reservations use notional/leverage and cannot reuse held-position collateral", () => {
  const config = rules();
  const existing = order({ notional: rat(300) });
  const prepared = ok(prepareArrival(state({ orders: [existing] }), config));
  exact(reserved(prepared.state, config), 60);
  const result = executeAction(prepared, action({ marketQuote: rat(-50), place: [order({ id: "extra", placedAt: 1, notional: rat(150) })] }), config);
  assert.equal(result.ok, false);
  if (!result.ok) assert.equal(result.error.code, "shared-collateral");
  assert.equal(executeAction(prepared, action({ marketQuote: rat(-50), cancelIds: ["buy"] }), config).ok, true);
});

test("effective maintenance bounds are inclusive; breaching them zeroes the entire account and book", () => {
  const config = rules({ maintenanceUpper: rat(2) });
  const atBoundary = state({ quote: rat("-49.975"), asset: ONE, orders: [order()] });
  exact(effectiveEquity(atBoundary, config), "49.975");
  assert.equal(survives(atBoundary, config), true);
  assert.ok(eq(advance(atBoundary, rat(100), ONE, config).asset, ONE));
  const beyond = advance({ ...atBoundary, quote: rat("-49.976") }, rat(100), ONE, config);
  exact(beyond.quote, 0);
  exact(beyond.asset, 0);
  assert.deepEqual(beyond.orders, []);
});

test("liquidation precedes next-arrival fills even when a resting order was crossed", () => {
  const config = rules();
  const source = state({ quote: rat(-100), asset: rat(2), orders: [order({ price: rat(50) })] });
  const moved = advance(source, rat(40), ONE, config);
  exact(equity(moved), 0);
  assert.deepEqual(ok(prepareArrival(moved, config)).fills, []);
  const inert = ok(executeAction(ok(prepareArrival(moved, config)), action(), config));
  exact(equity(advance(inert, rat(1000), ONE, config)), 0);
});

test("physical partial close never rounds dust upward or silently closes beyond the cap", () => {
  const config = rules({ makerFee: rat("0.05"), takerFee: rat("0.1"), quoteMinimum: rat(5), quoteMaximum: rat(10) });
  const dust = ok(closeWithinCapacityChecked(state({ asset: rat("0.02"), orders: [order()] }), config));
  exact(dust.marketQuote, 0);
  exact(dust.state.asset, "0.02");
  exact(dust.wealth, 102);
  assert.deepEqual(dust.state.orders, []);
  const capped = ok(closeWithinCapacityChecked(state({ quote: ZERO, asset: ONE }), config));
  exact(capped.marketQuote, 10);
  exact(capped.state.asset, "8/9");
  exact(capped.wealth, "890/9");
  const short = ok(closeWithinCapacityChecked(state({ quote: rat(200), asset: rat(-1) }), config));
  exact(short.marketQuote, -10);
  exact(short.state.asset, "-0.91");
  exact(short.wealth, 99);
});

test("terminal arrival fills precede cancellation and share capacity with the final close", () => {
  const config = rules({ quoteMaximum: rat(100) });
  const incoming = state({ quote: rat(1000), price: rat(95), previousPrice: rat(100), orders: [order({ notional: rat(95) })] });
  const finished = ok(closeWithinCapacityChecked(incoming, config));
  exact(finished.marketQuote, 5); // The 95 quote arrival already consumed the rest.
  exact(finished.state.quote, 910);
  assert.ok(eq(finished.state.asset, sub(rat("0.9998"), div(rat(5), mul(rat("0.9995"), rat(95))))));
  assert.deepEqual(finished.state.orders, []);
  assert.throws(() => terminalClose({ ...incoming, orders: [order({ id: "a", notional: rat(60) }), order({ id: "b", notional: rat(60) })] }, config), /aggregate-quote-cap/);
});

test("complete terminal liquidation reproduces the Lean binary-price CRRA counterexample wealth", () => {
  const config = rules();
  for (const price of [90, 99, 110]) for (const size of [ZERO, rat("0.5"), ONE]) {
    const entered = executeQuote(state({ quote: ONE }), neg(size), config.takerFee);
    const finished = terminalClose(advance(entered, rat(price), ONE, config), config);
    const expected = add(sub(ONE, size), div(mul(mul(mul(size, sub(ONE, config.takerFee)), sub(ONE, config.takerFee)), rat(price)), rat(100)));
    assert.ok(eq(finished.wealth, expected));
    exact(finished.state.asset, 0);
  }
});

test("completed policies reject dust, capped residuals and terminal fills without closing capacity", () => {
  const config = rules({ quoteMinimum: rat(5), quoteMaximum: rat(100) });
  for (const asset of [rat("0.01"), rat("-0.01"), rat(2), rat(-2)]) {
    assert.throws(() => terminalClose(state({ quote: rat(1000), asset }), config), /terminal-exposure/);
  }
  const incoming = state({ quote: rat(1000), price: rat(95), previousPrice: rat(100),
    orders: [order({ notional: rat(95) })] });
  assert.throws(() => terminalClose(incoming, config), /terminal-exposure/);
  for (const asset of [ZERO, rat("0.5"), rat("-0.5")]) {
    const closed = terminalClose(state({ quote: rat(1000), asset, orders: [order()] }), config);
    exact(closed.state.asset, 0);
    assert.deepEqual(closed.state.orders, []);
    assert(eq(closed.wealth, closed.state.quote));
  }
});

test("invalid observed grid prices and malformed static parameters are rejected explicitly", () => {
  assert.equal(prepareArrival(state({ price: rat("100.001") }), rules()).ok, false);
  assert.equal(prepareArrival(state(), rules({ makerFee: rat("0.0006") })).ok, false);
  assert.equal(prepareArrival(state(), rules({ makerFee: rat("0.0005") })).ok, false);
  assert.equal(prepareArrival(state(), rules({ makerFee: ZERO, takerFee: ZERO })).ok, true);
  assert.equal(prepareArrival(state(), rules({ quoteMinimum: rat(20), quoteMaximum: rat(10) })).ok, false);
  assert.throws(() => advance(state(), rat("100.001"), ONE, rules()));
  assert.throws(() => advance(state(), rat(100), ZERO, rules()));
});

test("the original-Lean conformance trace fixes every account/book phase and aggregate cap rejection", () => {
  // These literal outputs are proved by ExactOracleConformance.lean, independently
  // of the JSONL fixture generator and this TypeScript execution implementation.
  const config = rules({ makerFee: ZERO, takerFee: rat("1/1000"), quoteMinimum: ONE,
    quoteMaximum: rat(200), maintenanceLower: rat(-2), maintenanceUpper: rat(2), maxOrders: 4 });
  const initial = state({ quote: rat(1000), time: 0 });
  const buy = order({ id: "1", notional: rat(95), placedAt: 0 });
  const sell = order({ id: "2", side: "sell", price: rat(110), notional: rat(110), placedAt: 0 });
  const first = ok(executeAction(ok(prepareArrival(initial, config)), action({ marketQuote: rat(-100), place: [buy, sell] }), config));
  assert.deepEqual(first, { ...initial, quote: rat(900), asset: rat("999/1000"), orders: [buy, sell] });
  const moved = advance(first, rat(90), ONE, config);
  assert.deepEqual(moved, { ...first, price: rat(90), previousPrice: rat(100), time: 1 });
  const arrived = ok(prepareArrival(moved, config));
  assert.deepEqual(arrived, { state: { ...moved, quote: rat(805), asset: rat("1999/1000"), orders: [sell] },
    consumedQuote: rat(95), fills: [buy] });
  const placed = order({ id: "3", price: rat(85), notional: rat(85), placedAt: 1 });
  const secondAction = action({ marketQuote: rat("8991/100"), cancelIds: ["2"], place: [placed] });
  const second = ok(executeAction(arrived, secondAction, config));
  assert.deepEqual(second, { ...moved, quote: rat("89491/100"), asset: rat("999/1000"), orders: [placed] });
  const final = advance(second, rat(100), rat("9/10"), config);
  assert.deepEqual(final, { ...second, asset: rat("8991/10000"), price: rat(100), previousPrice: rat(90), time: 2 });
  const rejected = executeAction(arrived, secondAction, { ...config, quoteMaximum: rat(100) });
  assert.equal(rejected.ok, false);
  if (!rejected.ok) assert.equal(rejected.error.code, "aggregate-quote-cap");
});
