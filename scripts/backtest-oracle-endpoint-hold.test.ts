import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { advance, terminalClose, type OracleRules, type OracleState } from "../packages/bot-algo/src/oracle-crra-bot.js";
import { ONE, ZERO, rat, eq } from "../packages/bot-algo/src/oracle-rational.js";

test("nonnegative hold has the same terminal account after extreme positive intermediate marks", () => {
  const rules: OracleRules = { makerFee: rat("0.0002"), takerFee: rat("0.0005"), reserveLeverage: ONE,
    entryLower: ZERO, entryUpper: ONE, maintenanceLower: ZERO, maintenanceUpper: ONE,
    quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 0 };
  for (const quote of [ZERO, rat(5000)]) {
    const initial: OracleState = { quote, asset: ONE, price: rat(100), previousPrice: rat(100), orders: [], time: 0 };
    let actual = initial;
    for (const price of ["0.01", "1000000", "120"]) actual = advance(actual, rat(price), ONE, rules);
    const direct = advance(initial, rat(120), ONE, rules);
    assert(eq(actual.quote, direct.quote)); assert(eq(actual.asset, direct.asset));
    assert(eq(terminalClose(actual, rules).wealth, terminalClose(direct, rules).wealth));
  }
});

test("endpoint replay uses all intermediate marks, costs, and nonoverlapping horizon decisions", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "oracle-endpoint-test-"));
  const input = path.join(directory, "forecast.jsonl");
  const rows = [{ horizonSeconds: 3, timestamp: 1000, observedPrice: 100,
    realizedNextTimestamp: 4000, realizedNextPrice: 120, realizedPrices: [80, 90, 120],
    tree: { children: [{ probability: 1, price: 120 }] } }];
  await writeFile(input, rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
  const result = spawnSync(process.execPath, ["--conditions=development", "--import", "tsx",
    "scripts/backtest-oracle-endpoint-hold.ts", "--forecasts", input, "--output", directory],
    { encoding: "utf8", windowsHide: true });
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(await readFile(path.join(directory, "report.json"), "utf8"));
  assert.equal(report.observedSeconds, 3); assert.equal(report.decisionCount, 1);
  assert.equal(report.longDecisions, 1);
  // The 100% target would exceed the $10k terminal cap after the known rise.
  // The declared finite menu can instead enter at 50% and close completely.
  assert(report.returnPercent > 9 && report.returnPercent < 10);
  assert(report.maximumDrawdownPercent > 10 && report.maximumDrawdownPercent < 11);
  assert.equal(report.terminalResidualAsset, 0);
  assert.equal(report.terminalRequirement, "zero-asset-and-no-orders");
  assert.equal(report.buyHoldFeasible, false);
  assert.equal(report.buyHoldReturnPercent, null);
});

test("endpoint replay rejects overlapping or missing intervals before making the next decision", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "oracle-endpoint-continuity-test-"));
  const input = path.join(directory, "forecast.jsonl");
  const first = { horizonSeconds: 3, timestamp: 1000, observedPrice: 100,
    realizedNextTimestamp: 4000, realizedNextPrice: 100, realizedPrices: [100, 100, 100],
    tree: { children: [{ probability: 1, price: 100 }] } };
  for (const timestamp of [3000, 5000]) {
    // Each row is internally valid and prices match. Only the decision boundary
    // is wrong: the second row must start at 4000, not overlap or skip a second.
    const second = { ...first, timestamp, realizedNextTimestamp: timestamp + 3000 };
    await writeFile(input, [first, second].map((row) => JSON.stringify(row)).join("\n") + "\n");
    const result = spawnSync(process.execPath, ["--conditions=development", "--import", "tsx",
      "scripts/backtest-oracle-endpoint-hold.ts", "--forecasts", input, "--output", directory],
      { encoding: "utf8", windowsHide: true });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /contiguous nonoverlapping intervals/);
  }
});
