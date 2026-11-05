import { createReadStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import path from "node:path";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { advance, equity, executeAction, exposure, finiteActionMenu, observableTree, requestForTarget,
  solveCrra, terminalClose, terminalCloseChecked, type ForecastTree, type OracleRules, type OracleState } from "../packages/bot-algo/src/oracle-crra-bot.js";
import { ONE, ZERO, add, div, eq, lt, mul, nearestStep, rat, serializeRat, sub, toNumber } from "../packages/bot-algo/src/oracle-rational.js";

/** Endpoint forecasts are sufficient only for this restricted no-order,
 * nonnegative-cash/nonnegative-asset hold between decision times. Every actual
 * intermediate mark is still executed and checked. No leveraged/path optimum
 * is inferred from the endpoint distribution. */
interface Row {
  readonly horizonSeconds: number;
  readonly timestamp: number;
  readonly observedPrice: number;
  readonly realizedNextTimestamp: number;
  readonly realizedNextPrice: number;
  readonly realizedPrices: readonly number[];
  readonly tree: ForecastTree;
}
async function main() {
  const args = process.argv.slice(2);
  const option = (key: string, fallback: string) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
  const input = path.resolve(option("--forecasts", "data/backtests/oracle-crra-horizon-h30.jsonl"));
  const output = path.resolve(option("--output", "data/backtests/stochastic-crra/endpoint-h30"));
  const requested = Number(option("--count", "0"));
  const certificateCount = Number(option("--certificate-count", "3"));
  const feeScale = rat(option("--fee-scale", "1"));
  if (!Number.isSafeInteger(requested) || requested < 0 ||
      !Number.isSafeInteger(certificateCount) || certificateCount < 0 || lt(feeScale, ZERO))
    throw new Error("Invalid count, certificate count or fee scale");
  const rules: OracleRules = { makerFee: mul(rat("0.0002"), feeScale), takerFee: mul(rat("0.0005"), feeScale),
    reserveLeverage: ONE, entryLower: ZERO, entryUpper: ONE, maintenanceLower: ZERO, maintenanceUpper: ONE,
    quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 0 };
  const config = { riskAversion: 0, rules, funding: () => ONE, menu: finiteActionMenu({
    exposureTargets: [ZERO, rat("1/2"), ONE], limitNotionalFractions: [], limitOffsetsInTicks: [], maximumNewOrders: 0,
  }) } as const;
  await mkdir(output, { recursive: true });
  const reader = createInterface({ input: createReadStream(input), crlfDelay: Infinity });
  const hash = createHash("sha256");
  let state: OracleState | undefined, buyHold: OracleState | undefined;
  let count = 0, markets = 0, longDecisions = 0, cashDecisions = 0, directionChanges = 0;
  let firstTimestamp = 0, expectedTimestamp = 0, horizon = 0, previousDirection: number | undefined;
  let initialPrice = 0, lastPrice = 0, peak = 10000, drawdown = 0, sumAbsExposure = 0;
  let predictedGrossBpsSum = 0, maxPredictedGrossBps = -Infinity;
  let aboveFullCloseThreshold = 0;
  const certificates: string[] = [];
  const trajectory: unknown[] = [];
  const started = performance.now();
  for await (const line of reader) {
    if (!line.trim()) continue;
    hash.update(line + "\n");
    const row: Row = JSON.parse(line);
    if (!Number.isSafeInteger(row.horizonSeconds) || row.horizonSeconds < 1 ||
        row.realizedPrices.length !== row.horizonSeconds ||
        row.realizedNextTimestamp !== row.timestamp + row.horizonSeconds * 1000 ||
        row.realizedPrices.at(-1) !== row.realizedNextPrice)
      throw new Error("Endpoint row must contain every actual intermediate second and its final mark");
    if (count && (row.horizonSeconds !== horizon || row.timestamp !== expectedTimestamp))
      throw new Error("Endpoint decisions must have a fixed horizon and contiguous nonoverlapping intervals");
    const currentPrice = rat(row.observedPrice);
    if (!state) {
      state = { quote: rat(10000), asset: ZERO, price: currentPrice, previousPrice: currentPrice, orders: [], time: 0 };
      firstTimestamp = row.timestamp; initialPrice = row.observedPrice; horizon = row.horizonSeconds;
      const entered = executeAction({ state, consumedQuote: ZERO, fills: [] }, {
        marketQuote: requestForTarget(state, ONE, rules), cancelIds: [], place: [],
      }, rules);
      if (!entered.ok) throw new Error(entered.error.message);
      buyHold = entered.value;
    }
    if (!eq(state.price, currentPrice) || state.orders.length || lt(state.quote, ZERO) || lt(state.asset, ZERO))
      throw new Error("Restricted endpoint state requires the current observed price, q>=0, a>=0 and no orders");
    const tree = observableTree(row.tree, rules.priceStep);
    if (tree.children.some((branch) => branch.children.length)) throw new Error("Endpoint input must have exactly one forecast level");
    const expectedGross = tree.children.reduce((total, branch) => add(total,
      mul(branch.probability, div(branch.price, state!.price))), ZERO);
    const predictedBps = toNumber(sub(expectedGross, ONE)) * 1e4;
    predictedGrossBpsSum += predictedBps;
    maxPredictedGrossBps = Math.max(maxPredictedGrossBps, predictedBps);
    const breakEven = div(ONE, mul(sub(ONE, rules.takerFee), sub(ONE, rules.takerFee)));
    if (lt(breakEven, expectedGross)) aboveFullCloseThreshold++;
    const decision = solveCrra(state, tree, config, count < certificateCount);
    state = decision.state;
    if (decision.action.marketQuote.n) markets++;
    const direction = state.asset.n ? 1 : 0;
    if (direction) longDecisions++; else cashDecisions++;
    if (previousDirection !== undefined && direction !== previousDirection) directionChanges++;
    previousDirection = direction;
    sumAbsExposure += toNumber(exposure(state));
    if (decision.certificate) certificates.push(JSON.stringify(decision.certificate));
    // With q,a>=0, unit funding and maintenance [0,1], every positive price
    // survives. Intermediate marks cannot change the endpoint account q,a.
    for (const observed of row.realizedPrices) {
      const price = rat(observed);
      if (!eq(price, nearestStep(price, rules.priceStep))) throw new Error("Actual price is outside declared grid");
      state = advance(state, price, ONE, rules);
      buyHold = advance(buyHold!, price, ONE, rules);
      if (lt(state.quote, ZERO) || lt(state.asset, ZERO) || !state.quote.n && !state.asset.n)
        throw new Error("Endpoint equivalence violated: restricted positive account failed survival");
      const wealth = toNumber(equity(state));
      peak = Math.max(peak, wealth);
      drawdown = Math.max(drawdown, (peak - wealth) / peak);
    }
    expectedTimestamp = row.realizedNextTimestamp; lastPrice = row.realizedNextPrice; count++;
    if (count < 6 || count % Math.max(1, Math.floor(300 / horizon)) === 0)
      trajectory.push({ timestamp: expectedTimestamp, equity: toNumber(equity(state)), exposure: toNumber(exposure(state)) });
    if (requested && count >= requested) break;
  }
  if (!state || !count || requested && count !== requested) throw new Error("Incomplete requested endpoint replay");
  const closed = terminalClose(state, rules);
  if (closed.marketQuote.n) markets++;
  const wealth = toNumber(closed.wealth);
  drawdown = Math.max(drawdown, (peak - wealth) / peak);
  const benchmarkClose = terminalCloseChecked(buyHold!, rules);
  const benchmark = benchmarkClose.ok ? toNumber(benchmarkClose.value.wealth) : null;
  const report = { contract: "oracle-long-cash-endpoint-mpc-flat-v2", complete: true, terminalRequirement: "zero-asset-and-no-orders",
    buyHoldFeasible: benchmarkClose.ok, buyHoldFailure: benchmarkClose.ok ? null : benchmarkClose.error, input, forecastSha256: hash.digest("hex"),
    firstTimestamp, terminalTimestamp: expectedTimestamp, horizonSeconds: horizon, decisionCount: count,
    observedSeconds: count * horizon, gamma: 0, feeScale: serializeRat(feeScale), initialWealth: 10000,
    terminalWealth: wealth, returnPercent: (wealth / 10000 - 1) * 100,
    maximumDrawdownPercent: drawdown * 100, markets, longDecisions, cashDecisions, directionChanges,
    averageExposure: sumAbsExposure / count, terminalResidualAsset: toNumber(closed.state.asset),
    buyHoldWealth: benchmark, buyHoldReturnPercent: benchmark === null ? null : (benchmark / 10000 - 1) * 100,
    underlyingReturnPercent: (lastPrice / initialPrice - 1) * 100,
    averagePredictedGrossBps: predictedGrossBpsSum / count, maxPredictedGrossBps,
    predictedAboveFullCloseBreakEven: aboveFullCloseThreshold,
    fullCloseBreakEvenBps: toNumber(sub(div(ONE, mul(sub(ONE, rules.takerFee), sub(ONE, rules.takerFee))), ONE)) * 1e4,
    seconds: (performance.now() - started) / 1000, decisionCertificates: certificates.length,
    assumptions: ["long/cash only, q>=0 and asset>=0, entry grid 0/0.5/1 plus hold, no pending orders, k=1",
      "actual marks and maintenance executed every second; decisions only once per forecast horizon",
      "all positive intermediate prices survive for this restricted account; endpoint law suffices for the hold",
      "every hypothetical and final close must reach exactly zero holdings within the shared 10000 cap",
      "rolling finite-menu endpoint optimization, not a full-path leveraged or limit-order optimum"] };
  await writeFile(path.join(output, "report.json"), JSON.stringify(report, null, 2) + "\n");
  await writeFile(path.join(output, "decisions.jsonl"), certificates.join("\n") + "\n");
  await writeFile(path.join(output, "trajectory.json"), JSON.stringify(trajectory) + "\n");
  process.stdout.write(JSON.stringify(report, null, 2) + "\n");
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
