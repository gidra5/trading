import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { advance, equity, executeAction, terminalClose, type OracleRules, type OracleState } from "../packages/bot-algo/src/oracle-execution.js";
import { ONE, ZERO, eq, min, mul, neg, rat, sub, toNumber } from "../packages/bot-algo/src/oracle-rational.js";

/** Independently execute every chosen continuous action in the exact Oracle
 * engine. This verifies execution refinement, not the numerical integration or
 * continuous optimizer's value bounds. */
async function main() {
  const root = path.resolve(process.argv[2] ?? "data/backtests/oracle-continuous-full-law");
  const read = async (file: string) => JSON.parse(await readFile(file, "utf8"));
  const sha = async (file: string) => createHash("sha256").update(await readFile(file)).digest("hex");
  const comparison = await read(path.join(root, "comparison.json"));
  for (const [file, hash] of Object.entries(comparison.sourceHashes)) assert.equal(await sha(file), hash);
  let decisions = 0, marks = 0, maximumStateError = 0;
  const check = (actual: number, expected: number) => {
    const error = Math.abs(actual - expected);
    maximumStateError = Math.max(maximumStateError, error);
    assert(error <= 1e-7 * Math.max(1, Math.abs(expected)), `${actual} differs from ${expected}`);
  };
  for (const report of comparison.reports) {
    assert.equal(await sha(report.decisionsFile), report.decisionsSha256);
    const run = await read(report.decisionsFile);
    assert.equal(await sha(run.forecastFile), run.forecastSha256);
    assert.equal(await sha(run.parametersFile), run.parametersSha256);
    const rows = (await readFile(run.forecastFile, "utf8")).trim().split(/\r?\n/).map(line => JSON.parse(line));
    assert.equal(rows.length, run.decisions.length);
    const rules: OracleRules = { makerFee: ZERO, takerFee: rat(run.rules.fee), reserveLeverage: ONE,
      entryLower: ZERO, entryUpper: ONE, maintenanceLower: ZERO, maintenanceUpper: ONE,
      quoteMinimum: rat(run.rules.minimum), quoteMaximum: rat(run.rules.maximum),
      priceStep: rat(run.rules.price_step), maxOrders: 0 };
    let state: OracleState = { quote: rat(10000), asset: ZERO, price: rat(rows[0].observedPrice),
      previousPrice: rat(rows[0].observedPrice), orders: [], time: 0 };
    let marketTrades = 0, peak = 10000, drawdown = 0;
    for (let index = 0; index < rows.length; index++) {
      const row = rows[index], decision = run.decisions[index];
      assert.equal(row.timestamp, decision.timestamp);
      assert(eq(state.price, rat(decision.price)));
      check(toNumber(state.quote), decision.before.quote);
      check(toNumber(state.asset), decision.before.asset);
      const encoded = decision.action;
      let trade = ZERO;
      if (encoded.marketQuote !== "0") {
        const capacity = min(rules.quoteMaximum, encoded.side === "buy" ? state.quote
          : mul(mul(sub(ONE, rules.takerFee), state.asset), state.price));
        const amount = encoded.size === "capacity" ? capacity : encoded.size === "minimum"
          ? rules.quoteMinimum : rat(encoded.amount);
        trade = encoded.side === "buy" ? neg(amount) : amount;
      }
      const action = { marketQuote: trade, cancelIds: [], place: [] } as const;
      const executed = executeAction({ state, consumedQuote: ZERO, fills: [] }, action, rules);
      assert(executed.ok, executed.ok ? "" : JSON.stringify(executed.error));
      state = executed.value;
      if (trade.n) marketTrades++;
      check(toNumber(trade), decision.optimization.marketQuote);
      check(toNumber(state.quote), decision.after.quote);
      check(toNumber(state.asset), decision.after.asset);
      assert(state.quote.n >= 0n && state.asset.n >= 0n);
      assert(decision.optimization.optimalityGap <= comparison.plan.absoluteValueTolerance * 1.00001);
      for (const price of row.realizedPrices) {
        state = advance(state, rat(price), ONE, rules);
        assert(state.quote.n >= 0n && state.asset.n >= 0n);
        const wealth = toNumber(equity(state));
        assert(wealth > 0, "A nonnegative account must survive every positive mark");
        peak = Math.max(peak, wealth);
        drawdown = Math.max(drawdown, 1 - wealth / peak);
        marks++;
      }
      decisions++;
    }
    const closed = terminalClose(state, rules);
    if (closed.marketQuote.n) marketTrades++;
    const wealth = toNumber(closed.wealth);
    drawdown = Math.max(drawdown, 1 - wealth / peak);
    check(wealth, report.terminalWealth);
    check(toNumber(closed.state.asset), report.terminalResidualAsset);
    check(toNumber(closed.state.quote), report.terminalQuote);
    check(drawdown * 100, report.maximumDrawdownPercent);
    assert.equal(marketTrades, report.markets);
    console.log(`Verified ${report.variant} h${report.horizonSeconds} fees${report.feeScale}`);
  }
  const result = { allChecksPassed: true, runs: comparison.reports.length, decisions, marks,
    maximumStateError, continuousValueBoundFormallyVerified: false,
    scope: "All selected actions, fees, minima/caps, price grids, intermediate funding/maintenance, final close and residuals replayed with exact rational Oracle execution",
    verifierSha256: await sha("scripts/verify-continuous-oracle.ts") };
  await writeFile(path.join(root, "execution-verification.json"), JSON.stringify(result, null, 2) + "\n");
  await writeFile(path.join(root, "comparison.json"), JSON.stringify({ ...comparison, executionVerified: true }, null, 2) + "\n");
  console.log(JSON.stringify(result));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
