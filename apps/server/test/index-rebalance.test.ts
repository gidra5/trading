import assert from "node:assert/strict";
import test from "node:test";
import { rebalanceLongOnly } from "../src/index-rebalance.js";

test("initial long-only deployment pays buy costs and stays self-financing", () => {
  const result = rebalanceLongOnly(1_000, [
    { currentValue: 0, targetWeight: 0.6, costRate: 0.01 },
    { currentValue: 0, targetWeight: 0.4, costRate: 0.02 },
  ]);

  assert.ok(Math.abs(result.postTradeValue - 986.1932938856) < 1e-8);
  assert.ok(Math.abs(result.positions[0] / result.postTradeValue - 0.6) < 1e-12);
  assert.ok(Math.abs(result.positions[1] / result.postTradeValue - 0.4) < 1e-12);
  assert.equal(result.cashValue, 0);
  assert.ok(
    Math.abs(
      result.preTradeValue -
        result.transactionCost -
        result.postTradeValue,
    ) < 1e-9,
  );
});

test("switching assets charges both the sale and the purchase", () => {
  const result = rebalanceLongOnly(0, [
    { currentValue: 1_000, targetWeight: 0, costRate: 0.01 },
    { currentValue: 0, targetWeight: 1, costRate: 0.01 },
  ]);

  assert.ok(Math.abs(result.postTradeValue - 980.198019802) < 1e-8);
  assert.ok(Math.abs(result.grossTradedNotional - 1_980.198019802) < 1e-8);
  assert.ok(Math.abs(result.transactionCost - 19.801980198) < 1e-8);
  assert.equal(result.positions[0], 0);
  assert.ok(Math.abs(result.positions[1] - result.postTradeValue) < 1e-12);
});

test("partial exposure leaves the remainder in cash", () => {
  const result = rebalanceLongOnly(1_000, [
    { currentValue: 0, targetWeight: 0.25, costRate: 0 },
    { currentValue: 0, targetWeight: 0.5, costRate: 0 },
  ]);

  assert.equal(result.postTradeValue, 1_000);
  assert.deepEqual(result.positions, [250, 500]);
  assert.equal(result.cashValue, 250);
});

test("rejects leverage and short targets", () => {
  assert.throws(
    () =>
      rebalanceLongOnly(1_000, [
        { currentValue: 0, targetWeight: 1.01, costRate: 0 },
      ]),
    /must not exceed one/,
  );
  assert.throws(
    () =>
      rebalanceLongOnly(1_000, [
        { currentValue: 0, targetWeight: -0.01, costRate: 0 },
      ]),
    /Target weight/,
  );
});
