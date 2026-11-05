import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import path from "node:path";
import { advance, equity, executeAction, priceAllowed, terminalClose, type OracleRules, type OracleState } from "../packages/bot-algo/src/oracle-execution.js";
import { ONE, ZERO, abs, min, mul, neg, rat, sub, toNumber } from "../packages/bot-algo/src/oracle-rational.js";

async function sha(file: string) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

/** Exact Oracle checks at each floating-point issue state. Resetting to the
 * recorded issue state avoids unbounded rational-denominator growth across
 * tens of thousands of trades; this is local execution refinement, not an
 * exact-rational compounded backtest or a certificate for the optimizer. */
async function main() {
  const root = path.resolve(process.argv[2] ?? "data/backtests/oracle-refitted-h1-backtest");
  const comparison = JSON.parse(await readFile(path.join(root, "comparison.json"), "utf8"));
  const tape = JSON.parse(await readFile(path.join(root, "tape.json"), "utf8"));
  for (const [file, hash] of Object.entries(comparison.sourceHashes)) assert.equal(await sha(file), hash);
  const prices = (tape.prices as number[]).map(value => rat(value));
  let decisions = 0, marks = 0, maximumQuoteError = 0, maximumAssetError = 0;
  const check = (actual: number, expected: number, asset = false) => {
    const error = Math.abs(actual - expected);
    if (asset) maximumAssetError = Math.max(maximumAssetError, error);
    else maximumQuoteError = Math.max(maximumQuoteError, error);
    assert(error <= (asset ? 2e-12 : 2e-7), `${actual} != ${expected}; error ${error}`);
  };
  for (const report of comparison.reports) {
    assert.equal(await sha(report.decisionsFile), report.decisionsSha256);
    const rules: OracleRules = { makerFee: ZERO, takerFee: rat(report.rules.fee), reserveLeverage: ONE,
      entryLower: ZERO, entryUpper: ONE, maintenanceLower: ZERO, maintenanceUpper: ONE,
      quoteMinimum: rat(report.rules.minimum), quoteMaximum: rat(report.rules.maximum),
      priceStep: rat(report.rules.price_step), maxOrders: 0 };
    if (report === comparison.reports[0]) for (const p of prices) assert(priceAllowed(p, rules));
    let state: OracleState = { quote: rat(10000), asset: ZERO, price: prices[0]!, previousPrice: prices[0]!, orders: [], time: 0 };
    let index = 0, trades = 0, turnover = 0, fees = 0, peak = 10000, drawdown = 0;
    const mark = () => {
      assert(state.quote.n >= 0n && state.asset.n >= 0n);
      const wealth = toNumber(equity(state));
      assert(wealth > 0);
      peak = Math.max(peak, wealth); drawdown = Math.max(drawdown, 1 - wealth / peak);
    };
    const lines = createInterface({ input: createReadStream(report.decisionsFile), crlfDelay: Infinity });
    for await (const line of lines) {
      if (!line.trim()) continue;
      const row = JSON.parse(line), offset = tape.offsets[index];
      assert.equal(row.index, index);
      assert.equal(toNumber(state.price), tape.prices[offset]);
      check(toNumber(state.quote), row.before[0]); check(toNumber(state.asset), row.before[1], true);
      state = { ...state, quote: rat(row.before[0]), asset: rat(row.before[1]) };
      const action = row.action;
      let trade = ZERO;
      if (action.marketQuote !== "0") {
        const capacity = min(rules.quoteMaximum, action.side === "buy" ? state.quote
          : mul(mul(sub(ONE, rules.takerFee), state.asset), state.price));
        const amount = action.size === "capacity" ? capacity : action.size === "minimum" ? rules.quoteMinimum : rat(action.amount);
        trade = action.side === "buy" ? neg(amount) : amount;
      }
      const beforeEquity = toNumber(equity(state));
      const executed = executeAction({ state, consumedQuote: ZERO, fills: [] }, { marketQuote: trade, cancelIds: [], place: [] }, rules);
      assert(executed.ok, executed.ok ? "" : JSON.stringify(executed.error));
      state = executed.value;
      if (trade.n) trades++;
      turnover += toNumber(abs(trade)); fees += beforeEquity - toNumber(equity(state));
      check(toNumber(trade), row.trade);
      check(toNumber(state.quote), row.after[0]); check(toNumber(state.asset), row.after[1], true);
      assert(row.gap >= 0 && row.gap <= comparison.plan.absoluteValueTolerance * 1.00001);
      check(row.globalUpper - row.valueLower, row.gap);
      mark();
      state = advance(state, prices[offset + 1]!, ONE, rules); mark(); marks++;
      const preclose = toNumber(equity(state));
      const close = terminalClose(state, rules);
      if (close.marketQuote.n) trades++;
      turnover += toNumber(close.marketQuote); fees += preclose - toNumber(close.wealth);
      check(toNumber(close.marketQuote), row.closeTrade);
      state = close.state;
      check(toNumber(state.quote), row.closed[0]); check(toNumber(state.asset), row.closed[1], true);
      mark();
      const next = index + 1 < tape.offsets.length ? tape.offsets[index + 1] : offset + 1;
      for (let i = offset + 2; i <= next; i++) { state = advance(state, prices[i]!, ONE, rules); mark(); marks++; }
      index++; decisions++;
    }
    assert.equal(index, report.decisionCount); assert.equal(trades, report.markets);
    check(toNumber(equity(state)), report.terminalWealth);
    check(toNumber(state.quote), report.terminalQuote); check(toNumber(state.asset), report.terminalResidualAsset, true);
    check(drawdown * 100, report.maximumDrawdownPercent);
    assert(Math.abs(turnover - report.quoteTurnover) < 1e-3);
    assert(Math.abs(fees - report.feesQuoteEquivalent) < 1e-5);
    console.log(`Verified ${report.variant}, fee scale ${report.feeScale}, ${index} decisions`);
  }
  const result = { allChecksPassed: true, runs: comparison.reports.length, decisions, marks, maximumQuoteError, maximumAssetError,
    issueStatesRoundedToRecordedFloatingPoint: true, continuousOptimizerFormallyCertified: false,
    scope: "Exact rational Oracle action feasibility, entry/close fees, caps, dust, funding and maintenance at each recorded floating-point issue state; contiguous account linkage and all prices checked",
    verifierSha256: await sha("scripts/verify-refitted-oracle-backtest.ts") };
  await writeFile(path.join(root, "execution-verification.json"), JSON.stringify(result, null, 2) + "\n");
  await writeFile(path.join(root, "comparison.json"), JSON.stringify({ ...comparison, executionVerified: true }, null, 2) + "\n");
  console.log(JSON.stringify(result));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
