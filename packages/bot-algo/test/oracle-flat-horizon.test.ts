import assert from "node:assert/strict";
import test from "node:test";
import { advance, effectiveAsset, equity, executeAction, prepareArrival, terminalClose,
  type OracleRules, type OracleState, type Result } from "../src/oracle-execution.js";
import { assessFunding, conservativeTerminalWealth } from "../src/oracle-conservative-maintenance.js";
import { ONE, ZERO, add, div, eq, gt, isZero, neg, rat } from "../src/oracle-rational.js";

const rules: OracleRules = { makerFee: rat("0.0002"), takerFee: rat("0.0005"),
  reserveLeverage: ONE, entryLower: ZERO, entryUpper: ONE, maintenanceLower: ZERO, maintenanceUpper: ONE,
  quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 0 };
const must = <T>(r: Result<T>): T => { if (!r.ok) throw new Error(r.error.message); return r.value; };

test("distant conditional opportunity changes the first action despite bounded prices and flat endings", () => {
  for (const delay of [0, 1, 10, 1000, 10000]) {
    const values = [false, true].map(up => {
      let s: OracleState = { quote: rat(1000), asset: ZERO, price: rat(100), previousPrice: rat(100), time: 0, orders: [] };
      let debt = ZERO;
      const tick = (dq: typeof ZERO, price: number) => {
        const r = must(executeAction(must(prepareArrival(s, rules)), { marketQuote: dq, cancelIds: [], place: [] }, rules));
        const assessed = assessFunding(r, ONE, rat("0.00001"));
        if (!isZero(r.asset)) assert(gt(ZERO, assessed.assessedChange));
        else assert(isZero(assessed.debit));
        debt = add(debt, assessed.debit);
        s = advance(r, rat(price), ONE, rules);
        assert(gt(equity(s), ZERO));
      };
      const price = up ? 110 : 90;
      tick(rat(-1000), price);
      tick(effectiveAsset(s, rules), price); // close at the first observed branch
      assert(isZero(s.asset));
      for (let i = 0; i < delay; i++) tick(ZERO, price);
      tick(ZERO, up ? 100 : 90);
      tick(up ? neg(s.quote) : ZERO, price);
      const terminal = terminalClose(s, rules);
      assert(isZero(terminal.state.asset));
      assert.deepEqual(terminal.state.orders, []);
      assert.equal(terminal.state.time, delay + 4);
      return conservativeTerminalWealth(terminal.wealth, debt);
    });
    const expected = div(add(values[0]!, values[1]!), rat(2));
    assert(eq(expected, rat("2106651065886827/2000000000000")));
    // Lean's all-continuation growth bound for an initial cash action is 1050.
    assert(gt(expected, rat(1053)));
  }
});
