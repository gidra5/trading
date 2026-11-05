import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { finished } from "node:stream/promises";
import path from "node:path";
import { observableTree, type PriceTree } from "../packages/bot-algo/src/oracle-crra-policy.js";
import { priceAllowed, executeAction, requestForTarget, advance, terminalClose, terminalCloseChecked, equity,
  type OracleRules, type OracleState } from "../packages/bot-algo/src/oracle-execution.js";
import { rat, sub, add, lte, ONE, ZERO, toNumber, eq } from "../packages/bot-algo/src/oracle-rational.js";

async function hash(file: string) { const h = createHash("sha256"); for await (const b of createReadStream(file)) h.update(b); return h.digest("hex"); }
async function main() {
  const root = path.resolve(process.argv[2] ?? "data/backtests/oracle-finite-cash-replay");
  const report = JSON.parse(await readFile(path.join(root, "report.json"), "utf8"));
  const plan = JSON.parse(await readFile(path.join(root, "plan.json"), "utf8"));
  for (const [file, digest] of Object.entries(plan.sourceHashes)) assert.equal(await hash(file), digest);
  const seen = new Uint8Array(report.rows), proofs = new Map<number, { lowerTick: string; upperTick: string }>();
  const mergedShadows = createWriteStream(path.join(root, "shadow-certificates.jsonl"));
  const mergedBellman = createWriteStream(path.join(root, "bellman-certificates.jsonl"));
  const artifactHashes: Record<string, string> = {};
  let bellman = 0, shadow = 0, exhaustive = 0;
  for (const worker of report.workers) {
    const dir = path.join(root, `worker-${worker.id}`);
    for (const name of ["decisions.jsonl", "shadow-certificates.jsonl", "bellman-certificates.jsonl"])
      artifactHashes[path.join(dir, name)] = await hash(path.join(dir, name));
    for await (const line of createInterface({ input: createReadStream(path.join(dir, "decisions.jsonl")), crlfDelay: Infinity })) {
      const row = JSON.parse(line); assert(row.index >= 0 && row.index < report.rows);
      assert.equal(row.index % plan.workers, worker.id); assert.equal(seen[row.index], 0);
      assert.equal(row.marketQuote, "0"); assert.equal(row.orders, 0);
      assert.equal(row.timestamp, report.firstTimestamp+row.index*1000);
      seen[row.index] = row.method === "oracle-shadow-bound" ? 1 : 2;
      if (seen[row.index] === 2) { assert.equal(row.method, "exhaustive-bellman"); exhaustive++; }
    }
    for await (const line of createInterface({ input: createReadStream(path.join(dir, "shadow-certificates.jsonl")), crlfDelay: Infinity })) {
      const p = JSON.parse(line); assert.equal(seen[p.index], 1); assert(!proofs.has(p.index));
      assert(0n < BigInt(p.lowerTick) && BigInt(p.lowerTick) <= BigInt(p.upperTick));
      assert(4999n*BigInt(p.upperTick) <= 5000n*BigInt(p.lowerTick));
      proofs.set(p.index, p); mergedShadows.write(line+"\n"); shadow++;
    }
    for await (const line of createInterface({ input: createReadStream(path.join(dir, "bellman-certificates.jsonl")), crlfDelay: Infinity })) {
      const c = JSON.parse(line); assert.equal(c.riskAversion, 0);
      assert.deepEqual(JSON.parse(c.chosenId), ["0", [], []]);
      mergedBellman.write(line+"\n"); bellman++;
    }
  }
  mergedShadows.end(); mergedBellman.end(); await Promise.all([finished(mergedShadows), finished(mergedBellman)]);
  assert(seen.every(n => n !== 0)); assert.equal(shadow, report.shadowBoundDecisions);
  assert.equal(exhaustive, report.exhaustiveDecisions); assert.equal(bellman, report.bellmanCertificates);
  const rules: OracleRules = { makerFee: rat("0.0002"), takerFee: rat("0.0005"), reserveLeverage: rat(2),
    entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
    quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat("0.01"), maxOrders: 1 };
  let state: OracleState | undefined, buyHold: OracleState | undefined, index = 0;
  let peak = 10000, dd = 0; const prefix = createHash("sha256");
  for await (const line of createInterface({ input: createReadStream(plan.input), crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    const row = JSON.parse(line); if (row.realizedNextTimestamp > plan.terminalTimestamp) break;
    prefix.update(line+"\n");
    const p = rat(row.observedPrice), next = rat(row.realizedNextPrice);
    assert(priceAllowed(p, rules) && priceAllowed(next, rules));
    assert.equal(row.timestamp, report.firstTimestamp+1000*index);
    assert.equal(row.realizedNextTimestamp, row.timestamp+1000);
    if (!state) {
      state = { quote: rat(10000), asset: ZERO, price: p, previousPrice: p, orders: [], time: 0 };
      const entered = executeAction({ state, consumedQuote: ZERO, fills: [] },
        { marketQuote: requestForTarget(state, ONE, rules), cancelIds: [], place: [] }, rules);
      assert(entered.ok); buyHold = entered.value;
    }
    assert(eq(state.price, p));
    const proof = proofs.get(index);
    if (proof) {
      const lo = rat(`${proof.lowerTick}/100`), hi = rat(`${proof.upperTick}/100`);
      const forecast = row.realizedNextTimestamp === plan.terminalTimestamp
        ? { children: row.tree.children.map(({ children: _, ...child }: any) => child) } : row.tree;
      const tree = observableTree(forecast, rules.priceStep);
      const contained = (price: ReturnType<typeof rat>) => {
        assert(lte(lo, sub(price, rules.priceStep)) && lte(add(price, rules.priceStep), hi));
      };
      contained(p);
      const visit = (node: PriceTree) => { for (const c of node.children) { contained(c.price); visit(c); } }; visit(tree);
    } else assert.equal(seen[index], 2);
    const executed = executeAction({ state, consumedQuote: ZERO, fills: [] }, { marketQuote: ZERO, cancelIds: [], place: [] }, rules);
    assert(executed.ok); state = advance(executed.value, next, ONE, rules);
    assert(eq(equity(state), rat(10000)) && eq(state.asset, ZERO) && state.orders.length === 0);
    buyHold = advance(buyHold!, next, ONE, rules);
    const w = toNumber(equity(buyHold)); peak = Math.max(peak, w); dd = Math.max(dd, 1-w/peak);
    index++;
  }
  assert.equal(index, report.rows); assert.equal(prefix.digest("hex"), report.evaluatedForecastPrefixSha256);
  assert(eq(terminalClose(state!, rules).wealth, rat(report.terminalWealth)));
  const benchmark = terminalCloseChecked(buyHold!, rules); const benchmarkWealth = benchmark.ok ? toNumber(benchmark.value.wealth) : null;
  if (benchmarkWealth !== null) dd = Math.max(dd, 1-benchmarkWealth/peak);
  const result = { checkedRows: index, completeUniqueCoverage: true, exactCashExecutionPassed: true,
    forecastEnvelopesMatched: shadow, exhaustiveBellmanRows: exhaustive, bellmanCertificates: bellman,
    buyAndHoldFeasible: benchmark.ok, buyAndHoldFailure: benchmark.ok ? null : benchmark.error,
    buyAndHoldReturnPercent: benchmarkWealth === null ? null : 100*(benchmarkWealth/10000-1),
    buyAndHoldDrawdownPercent: benchmarkWealth === null ? null : 100*dd,
    artifactHashes, verifierSha256: await hash(import.meta.filename),
    leanStillRequired: ["shadow-certificates.jsonl", "bellman-certificates.jsonl"] };
  await writeFile(path.join(root, "execution-verification.json"), JSON.stringify(result, null, 2)+"\n");
  console.log(JSON.stringify({ checkedRows: index, shadow, exhaustive, bellman, benchmarkWealth }));
}
main().catch(e => { console.error(e); process.exitCode = 1; });
