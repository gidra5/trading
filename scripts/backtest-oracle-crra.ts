import { createReadStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";
import { OracleCrraBot, advance, defaultMenu, equity, executeAction, finiteActionMenu, exposure, prepareArrival,
  requestForTarget, terminalClose, terminalCloseChecked, terminalUtility, defaultRiskAversion,
  type ForecastTree, type OracleRules, type OracleState } from "../packages/bot-algo/src/oracle-crra-bot.js";
import { ONE, ZERO, abs, add, eq, isZero, lt, mul, rat, serializeRat, toNumber } from "../packages/bot-algo/src/oracle-rational.js";

interface ForecastRow {
  readonly timestamp: number;
  readonly observedPrice: number;
  readonly realizedNextTimestamp: number;
  readonly realizedNextPrice: number;
  readonly tree: ForecastTree;
}
async function main(): Promise<void> {
const args = process.argv.slice(2);
function option(name: string, fallback: string): string { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; }
const input = path.resolve(option("--forecasts", "data/backtests/stochastic-crra/test-first-hour-3bins.jsonl"));
const gamma = Number(option("--gamma", String(defaultRiskAversion)));
if (gamma !== 0 && gamma !== 2) throw new Error("--gamma must be 0 or 2");
const output = path.resolve(option("--output", `data/backtests/stochastic-crra/first-hour-gamma${gamma}`));
const maximumRows = Number(option("--count", "3600"));
const certificates = Number(option("--certificates", "3"));
const horizon = Number(option("--horizon", "2"));
const feeScale = rat(option("--fee-scale", "1"));
if (lt(feeScale, ZERO)) throw new Error("Fee scale must be nonnegative");
if (!Number.isSafeInteger(maximumRows) || maximumRows < 1 || ![1, 2].includes(horizon)) throw new Error("Invalid backtest count/horizon");

// Declared research contract, not dynamically inferred exchange trading rules.
const rules: OracleRules = {
  makerFee: mul(rat("0.0002"), feeScale), takerFee: mul(rat("0.0005"), feeScale), reserveLeverage: rat(2),
  entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
  quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 1,
};
const menuConfig = { ...defaultMenu,
  exposureTargets: option("--targets", defaultMenu.exposureTargets.map(serializeRat).join(",")).split(",").map(rat) };
const config = { riskAversion: gamma, rules, menu: finiteActionMenu(menuConfig), funding: () => ONE } as const;
await mkdir(output, { recursive: true });
const reader = createInterface({ input: createReadStream(input), crlfDelay: Infinity });
const evaluatedForecastHash = createHash("sha256");
let bot: OracleCrraBot | undefined;
let buyHold: OracleState | undefined;
let firstTimestamp = 0, lastTimestamp = 0, lastNextTimestamp = 0, count = 0;
let markets = 0, fills = 0, placements = 0, cancellations = 0, liquidations = 0;
let longDecisions = 0, cashDecisions = 0, shortDecisions = 0, directionChanges = 0;
let previousDirection: number | undefined;
let totalAbsoluteExposure = 0, maximumAbsoluteExposure = 0;
let peak = 10000, maximumDrawdown = 0, tradedQuote = ZERO, firstPrice = ZERO, lastPrice = ZERO;
const certificateLines: string[] = [];
const trajectory: Record<string, unknown>[] = [];
const started = performance.now();
for await (const line of reader) {
  if (!line.trim()) continue;
  evaluatedForecastHash.update(line + "\n");
  const row: ForecastRow = JSON.parse(line);
  if (count && row.timestamp !== lastNextTimestamp) throw new Error("Forecast rows must be contiguous observed ticks");
  if (row.realizedNextTimestamp !== row.timestamp + 1000) throw new Error("Expected one-second observation step");
  const observedPrice = rat(row.observedPrice);
  if (!bot) {
    const initial: OracleState = { quote: rat(10000), asset: ZERO, price: observedPrice,
      previousPrice: observedPrice, orders: [], time: 0 };
    bot = new OracleCrraBot(initial, config);
    const entered = executeAction({ state: initial, consumedQuote: ZERO, fills: [] }, {
      marketQuote: requestForTarget(initial, ONE, rules), cancelIds: [], place: [],
    }, rules);
    if (!entered.ok) throw new Error(`Buy-hold baseline entry invalid: ${entered.error.message}`);
    buyHold = entered.value;
    firstTimestamp = row.timestamp; firstPrice = observedPrice;
  }
  if (!eq(bot.state.price, observedPrice)) throw new Error("Observed price does not equal previous realized price");
  const arrival = prepareArrival(bot.state, rules);
  if (!arrival.ok) throw new Error(`Realized arrival inadmissible: ${arrival.error.message}`);
  fills += arrival.value.fills.length;
  tradedQuote = add(tradedQuote, arrival.value.consumedQuote);
  const tree = horizon === 1 ? { children: row.tree.children.map(({ children: _, ...branch }) => branch) } : row.tree;
  const decision = bot.decide(observedPrice, tree, count < certificates);
  const direction = bot.state.asset.n > 0n ? 1 : bot.state.asset.n < 0n ? -1 : 0;
  if (direction > 0) longDecisions++; else if (direction < 0) shortDecisions++; else cashDecisions++;
  if (previousDirection !== undefined && previousDirection !== direction) directionChanges++;
  previousDirection = direction;
  const absoluteExposure = Math.abs(toNumber(exposure(bot.state)));
  totalAbsoluteExposure += absoluteExposure;
  maximumAbsoluteExposure = Math.max(maximumAbsoluteExposure, absoluteExposure);
  if (!isZero(decision.action.marketQuote)) markets++;
  placements += decision.action.place.length;
  cancellations += decision.action.cancelIds.length;
  tradedQuote = add(tradedQuote, abs(decision.action.marketQuote));
  if (decision.certificate) certificateLines.push(JSON.stringify(decision.certificate));
  const before = equity(bot.state);
  bot.observe(rat(row.realizedNextPrice));
  buyHold = advance(buyHold!, rat(row.realizedNextPrice), ONE, rules);
  if (!isZero(before) && isZero(equity(bot.state))) liquidations++;
  const wealth = toNumber(equity(bot.state));
  peak = Math.max(peak, wealth);
  maximumDrawdown = Math.max(maximumDrawdown, (peak - wealth) / peak);
  lastTimestamp = row.timestamp; lastNextTimestamp = row.realizedNextTimestamp; lastPrice = rat(row.realizedNextPrice);
  count++;
  if (count <= 5 || count % 60 === 0) trajectory.push({ timestamp: lastNextTimestamp, equity: wealth,
    exposure: toNumber(exposure(bot.state)), pendingOrders: bot.state.orders.length });
  if (count % 60 === 0) process.stdout.write(JSON.stringify({ rows: count, seconds: (performance.now() - started) / 1000, equity: wealth }) + "\n");
  if (count >= maximumRows) break;
}
if (!bot || count !== maximumRows) throw new Error(`Expected ${maximumRows} forecast rows, received ${count}`);
const preClose = bot.state;
const terminalArrival = prepareArrival(preClose, rules);
if (!terminalArrival.ok) throw new Error(terminalArrival.error.message);
fills += terminalArrival.value.fills.length;
tradedQuote = add(tradedQuote, terminalArrival.value.consumedQuote);
const terminal = bot.close();
if (!isZero(terminal.marketQuote)) markets++;
tradedQuote = add(tradedQuote, abs(terminal.marketQuote));
const finalWealth = toNumber(terminal.wealth);
maximumDrawdown = Math.max(maximumDrawdown, (peak - finalWealth) / peak);
const benchmark = terminalCloseChecked(buyHold!, rules);
const buyHoldWealth = benchmark.ok ? toNumber(benchmark.value.wealth) : null;
const terminalScore = terminalUtility(terminal.wealth, gamma);
const report = {
  contract: "oracle-finite-tree-crra-flat-v3", complete: true, terminalRequirement: "zero-asset-and-no-orders",
  buyHoldFeasible: benchmark.ok, buyHoldFailure: benchmark.ok ? null : benchmark.error, forecastFile: input, rows: count, firstTimestamp, lastTimestamp,
  evaluatedForecastPrefixSha256: evaluatedForecastHash.digest("hex"),
  terminalTimestamp: lastNextTimestamp, horizon, gamma,
  feeScale: serializeRat(feeScale),
  initialWealth: 10000, terminalWealth: finalWealth, returnPercent: 100 * (finalWealth / 10000 - 1),
  maximumDrawdownPercent: 100 * maximumDrawdown, markets, limitFills: fills, limitPlacements: placements,
  longDecisions, cashDecisions, shortDecisions, directionChanges,
  averageAbsoluteExposure: totalAbsoluteExposure / count, maximumAbsoluteExposure,
  cancellations, liquidations, tradedQuote: toNumber(tradedQuote), terminalCloseQuote: toNumber(terminal.marketQuote),
  terminalResidualAsset: toNumber(terminal.state.asset), pendingBeforeClose: preClose.orders.length,
  cashBaselineWealth: 10000, buyHoldBaselineWealth: buyHoldWealth,
  buyHoldBaselineReturnPercent: buyHoldWealth === null ? null : 100 * (buyHoldWealth / 10000 - 1),
  underlyingBuyHoldReturnPercent: 100 * (toNumber(lastPrice) / toNumber(firstPrice) - 1),
  terminalUtility: terminalScore === null ? "-infinity" : toNumber(terminalScore),
  seconds: (performance.now() - started) / 1000,
  decisionCertificates: certificateLines.length,
  rules: Object.fromEntries(Object.entries(rules).map(([key, value]) => [key, typeof value === "number" ? value : serializeRat(value)])),
  menu: { exposureTargets: menuConfig.exposureTargets.map(serializeRat),
    limitNotionalFractions: menuConfig.limitNotionalFractions.map(serializeRat),
    limitOffsetsInTicks: menuConfig.limitOffsetsInTicks, maximumNewOrders: menuConfig.maximumNewOrders,
    cancellationSubsets: "all", marketAmounts: "exact fee-aware targets capped by remaining decision budget" },
  assumptions: ["BTCUSDT spot one-second close tape with synthetic Oracle contract parameters",
    "unit signed-asset funding factor; this is not actual Binance USD-M funding replay",
    "full fills at crossed passive limit; liquidation before next arrival fills",
    "terminal arrivals consume shared capacity; remaining orders canceled; exact full close required on every supported branch",
    "finite conditional-bin-mean price tree rounded to 0.01; original continuous-law utility is not preserved",
    "rolling horizon optimum within declared finite menu; not the full-period or true-market optimum"],
};
await writeFile(path.join(output, "report.json"), JSON.stringify(report, null, 2) + "\n");
await writeFile(path.join(output, "trajectory.json"), JSON.stringify(trajectory) + "\n");
await writeFile(path.join(output, "decisions.jsonl"), certificateLines.join("\n") + (certificateLines.length ? "\n" : ""));
process.stdout.write(JSON.stringify(report, null, 2) + "\n");
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
