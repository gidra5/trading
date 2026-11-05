/** Full-interval replay of the existing finite-tree/menu policy, accelerated
 * only when OracleShadowBound proves cash dominates every possible trade.
 * Independent workers are valid ONLY if every selected action is a no-op.
 * A non-cash action aborts this specialization rather than resetting inventory.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { spawn } from "node:child_process";
import path from "node:path";
import { finished } from "node:stream/promises";
import { performance } from "node:perf_hooks";
import { observableTree, solveCrra, finiteActionMenu, defaultMenu, type PriceTree, type OracleRules,
  type OracleState } from "../packages/bot-algo/src/oracle-crra-bot.js";
import { ONE, ZERO, rat, sub, add, min, max, mul, div, lte, gt, eq } from "../packages/bot-algo/src/oracle-rational.js";

const input = "data/backtests/stochastic-crra/test-full-3bins.jsonl";
const terminalTimestamp = 1786464534000;
const initialWealth = 10000;
const rules: OracleRules = { makerFee: rat("0.0002"), takerFee: rat("0.0005"), reserveLeverage: rat(2),
  entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
  quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 1 };
const config = { riskAversion: 0 as const, rules, menu: finiteActionMenu(defaultMenu), funding: () => ONE };
const args = process.argv.slice(2);
const option = (key: string, fallback: string) => { const i = args.indexOf(key); return i < 0 ? fallback : args[i+1]!; };
const root = path.resolve(option("--output", "data/backtests/oracle-finite-cash-replay"));
const workers = Number(option("--workers", "8"));

export function cashEnvelope(price: ReturnType<typeof rat>, tree: PriceTree) {
  // All possible menu limits are exactly one tick away from an observed node.
  // Including leaf prices +/- one tick is conservative. Terminal market closes
  // execute at leaf marks. No orders are present at the root.
  let lo = price, hi = price;
  const visit = (node: PriceTree) => { for (const child of node.children) {
    lo = min(lo, child.price); hi = max(hi, child.price); visit(child);
  } };
  visit(tree); lo = sub(lo, rules.priceStep); hi = add(hi, rules.priceStep);
  if (!gt(lo, ZERO) || !lte(mul(sub(ONE, rules.makerFee), hi), lo)) return undefined;
  const lower = div(lo, rules.priceStep), upper = div(hi, rules.priceStep);
  assert.equal(lower.d, 1n); assert.equal(upper.d, 1n);
  return { lowerTick: lower.n.toString(), upperTick: upper.n.toString() };
}

async function worker(id: number) {
  const directory = path.join(root, `worker-${id}`); await mkdir(directory, { recursive: true });
  const decisions = createWriteStream(path.join(directory, "decisions.jsonl"));
  const shadows = createWriteStream(path.join(directory, "shadow-certificates.jsonl"));
  const certificates = createWriteStream(path.join(directory, "bellman-certificates.jsonl"));
  const hash = createHash("sha256");
  let index = 0, processed = 0, bounded = 0, exhaustive = 0, spotChecks = 0, certified = 0;
  let lastNextTimestamp: number | undefined, lastNextPrice: number | undefined;
  let previousPrice: number | undefined, firstTimestamp: number | undefined;
  const started = performance.now();
  for await (const line of createInterface({ input: createReadStream(input), crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    if (row.realizedNextTimestamp > terminalTimestamp) break;
    if (lastNextTimestamp !== undefined) {
      assert.equal(row.timestamp, lastNextTimestamp); assert.equal(row.observedPrice, lastNextPrice);
    }
    assert.equal(row.realizedNextTimestamp, row.timestamp+1000);
    assert(row.observedPrice > 0 && row.realizedNextPrice > 0);
    hash.update(line+"\n"); firstTimestamp ??= row.timestamp;
    if (index % workers === id) {
      const price = rat(row.observedPrice);
      const state: OracleState = { quote: rat(initialWealth), asset: ZERO, price,
        previousPrice: rat(previousPrice ?? row.observedPrice), orders: [], time: index };
      const forecast = row.realizedNextTimestamp === terminalTimestamp
        ? { children: row.tree.children.map(({ children: _, ...child }: any) => child) } : row.tree;
      const tree = observableTree(forecast, rules.priceStep);
      const bound = cashEnvelope(price, tree);
      const sample = processed === 0 || processed % 4096 === 0 || (!bound && exhaustive < 2);
      if (bound) {
        shadows.write(JSON.stringify({ index, ...bound })+"\n"); bounded++;
      } else exhaustive++;
      if (!bound || sample) {
        const decision = solveCrra(state, tree, config, sample);
        if (!eq(decision.action.marketQuote, ZERO) || decision.action.place.length || decision.action.cancelIds.length)
          throw new Error(`Non-cash policy at index ${index}: this replay requires sequential continuation from that state`);
        assert(eq(decision.state.quote, state.quote) && eq(decision.state.asset, ZERO) && !decision.state.orders.length);
        assert(decision.value && eq(decision.value, rat(initialWealth-1)));
        if (bound) spotChecks++;
        if (sample) { certificates.write(JSON.stringify(decision.certificate)+"\n"); certified++; }
      }
      decisions.write(JSON.stringify({ index, timestamp: row.timestamp, marketQuote: "0", orders: 0,
        method: bound ? "oracle-shadow-bound" : "exhaustive-bellman", leanBellmanSample: sample })+"\n");
      processed++;
      if (processed % 2000 === 0) console.log(JSON.stringify({ worker: id, processed, bounded, exhaustive,
        seconds: (performance.now()-started)/1000 }));
    }
    // These observed prices are used only for the next issue's account and
    // continuity checks. They never enter the current hypothetical price tree.
    previousPrice = row.observedPrice; lastNextPrice = row.realizedNextPrice;
    lastNextTimestamp = row.realizedNextTimestamp; index++;
  }
  for (const stream of [decisions, shadows, certificates]) stream.end();
  await Promise.all([decisions, shadows, certificates].map(stream => finished(stream)));
  assert.equal(lastNextTimestamp, terminalTimestamp);
  assert.equal(processed, Math.floor((index+workers-1-id)/workers));
  const report = { id, workers, rows: index, processed, bounded, exhaustive, spotChecks, certified,
    firstTimestamp, terminalTimestamp, evaluatedForecastPrefixSha256: hash.digest("hex"),
    terminalWealth: initialWealth, allDecisionsCash: true, seconds: (performance.now()-started)/1000 };
  await writeFile(path.join(directory, "report.json"), JSON.stringify(report, null, 2)+"\n");
  console.log(JSON.stringify(report));
}

async function main() {
  assert(Number.isSafeInteger(workers) && workers > 0 && workers <= 16);
  if (args.includes("--worker")) return worker(Number(option("--worker", "0")));
  await mkdir(root, { recursive: true });
  const sources = ["scripts/backtest-oracle-finite-cash.ts", "packages/bot-algo/src/oracle-crra-policy.ts",
    "packages/bot-algo/src/oracle-execution.ts", "packages/bot-algo/src/oracle-rational.ts",
    "docs/theory/lean/TradingTheory/OracleShadowBound.lean"];
  const hashes = Object.fromEntries(await Promise.all(sources.map(async file => [file,
    createHash("sha256").update(await readFile(file)).digest("hex")])));
  await writeFile(path.join(root, "plan.json"), JSON.stringify({ input, workers, gamma: 0, initialWealth,
    terminalTimestamp, terminalRequirement: "zero-asset-and-no-orders", maximumPlanningTicks: 2, actualClose: "only at final replay tick",
    menu: defaultMenu, rules, sourceHashes: hashes,
    scope: "Exact supplied two-tick tree/menu optimization; carried account replay; not full-interval Bellman optimality",
    independentWorkersJustification: "Every selected action must preserve cash and empty book; otherwise abort",
    forecast: "Existing causal two-tick archived model; no fresh-mixture rollout is available" },
    (_, value) => typeof value === "bigint" ? value.toString() : value, 2)+"\n");
  const started = performance.now();
  await Promise.all(Array.from({ length: workers }, (_, id) => new Promise<void>((resolve, reject) => {
    const child = spawn(process.execPath, ["--conditions=development", "--import", "tsx", import.meta.filename,
      "--worker", String(id), "--workers", String(workers), "--output", root], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    const log = createWriteStream(path.join(root, `worker-${id}.log`));
    child.stdout.pipe(log); child.stderr.pipe(log);
    child.once("error", reject); child.once("close", code => code === 0 ? resolve() : reject(new Error(`Worker ${id} failed (${code}); see its log`)));
  })));
  const reports = await Promise.all(Array.from({ length: workers }, async (_, id) =>
    JSON.parse(await readFile(path.join(root, `worker-${id}`, "report.json"), "utf8"))));
  for (const r of reports) { assert(r.allDecisionsCash); assert.equal(r.rows, reports[0].rows);
    assert.equal(r.evaluatedForecastPrefixSha256, reports[0].evaluatedForecastPrefixSha256); }
  assert.equal(reports.reduce((sum, r) => sum+r.processed, 0), reports[0].rows);
  for (const [file, hash] of Object.entries(hashes)) assert.equal(createHash("sha256").update(await readFile(file)).digest("hex"), hash);
  const report = { complete: true, terminalRequirement: "zero-asset-and-no-orders", rows: reports[0].rows, firstTimestamp: reports[0].firstTimestamp, terminalTimestamp,
    terminalWealth: initialWealth, returnPercent: 0, maximumDrawdownPercent: 0,
    markets: 0, limitPlacements: 0, limitFills: 0, cancellations: 0, liquidations: 0,
    terminalResidualAsset: 0, evaluatedForecastPrefixSha256: reports[0].evaluatedForecastPrefixSha256,
    shadowBoundDecisions: reports.reduce((s, r) => s+r.bounded, 0),
    exhaustiveDecisions: reports.reduce((s, r) => s+r.exhaustive, 0),
    shadowExhaustiveCrossChecks: reports.reduce((s, r) => s+r.spotChecks, 0),
    bellmanCertificates: reports.reduce((s, r) => s+r.certified, 0),
    seconds: (performance.now()-started)/1000, workers: reports, leanVerified: false };
  await writeFile(path.join(root, "report.json"), JSON.stringify(report, null, 2)+"\n");
  console.log(JSON.stringify(report));
}
if (process.argv[1] && path.resolve(process.argv[1]) === import.meta.filename)
  main().catch(error => { console.error(error); process.exitCode = 1; });
