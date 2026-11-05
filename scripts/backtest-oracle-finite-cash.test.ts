import assert from "node:assert/strict";
import test from "node:test";
import { cashEnvelope } from "./backtest-oracle-finite-cash.js";
import { observableTree, solveCrra, finiteActionMenu, defaultMenu } from "../packages/bot-algo/src/oracle-crra-policy.js";
import type { OracleRules } from "../packages/bot-algo/src/oracle-execution.js";
import { ONE, ZERO, rat, eq } from "../packages/bot-algo/src/oracle-rational.js";

test("a fee-spread certificate agrees with exhaustive Bellman search including passive orders", () => {
  const rules: OracleRules = { makerFee: rat("0.0002"), takerFee: rat("0.0005"), reserveLeverage: rat(2),
    entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
    quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 1 };
  for (let k = 1; k <= 8; k++) {
    const p = 1000*k;
    const tree = observableTree({ children: [
      { probability: "1/3", price: p-.01, children: [{ probability: "2/3", price: p+.02 }, { probability: "1/3", price: p-.02 }] },
      { probability: "2/3", price: p+.01, children: [{ probability: "1/3", price: p-.01 }, { probability: "2/3", price: p+.03 }] },
    ] }, rules.priceStep);
    const bound = cashEnvelope(rat(p), tree); assert(bound);
    assert(4999n*BigInt(bound.upperTick) <= 5000n*BigInt(bound.lowerTick));
    const result = solveCrra({ quote: rat(10000), asset: ZERO, price: rat(p), previousPrice: rat(p), orders: [], time: 0 }, tree,
      { riskAversion: 0, rules, menu: finiteActionMenu(defaultMenu), funding: () => ONE });
    assert(eq(result.action.marketQuote, ZERO)); assert.equal(result.action.place.length, 0);
    assert(result.value && eq(result.value, rat(9999)));
  }
});

test("a forecast outside the fee spread is sent to exhaustive search", () => {
  const tree = observableTree({ children: [{ probability: 1, price: 101 }] }, rat("0.01"));
  assert.equal(cashEnvelope(rat(100), tree), undefined);
});
