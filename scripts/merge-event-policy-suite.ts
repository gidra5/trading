/** Verify and summarize complete H1/H2 coverage from immutable audit parts. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("one") || !arg("two") || !arg("output")) throw new Error("Specify --one, comma-separated --two and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const one = directory(arg("one")), two = arg("two").split(",").map(directory), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const catalog = new KamaInspector(path.join(root, "data")).catalog().windows.filter(w => w.id !== "latest" && !w.id.startsWith("fit-"));
const oneConfig = read(one, "config.json"), oneRows = read(one, "summary.json").results;
assert.equal(oneConfig.contract, "event-fixed-law-suite-audit-v1"); assert.equal(oneConfig.depth, 1);
assert.deepEqual(oneConfig.catalog, catalog);
const twoRows = two.flatMap(source => {
  const c = read(source, "config.json"); assert.equal(c.contract, oneConfig.contract); assert.equal(c.depth, 2);
  assert.equal(c.source, oneConfig.source); assert.deepEqual(c.catalog, catalog);
  return read(source, "summary.json").results.map((r: any) => ({ ...r, auditSource: source, tolerance: c.tolerance }));
});
for (const rows of [oneRows, twoRows]) {
  assert.deepEqual(rows.map((r: any) => r.window.id).sort(), catalog.map(w => w.id).sort());
  assert.ok(rows.every((r: any) => r.fullWindow && r.end === r.window.endTime));
}
const behavior: any[] = [];
const results = catalog.map(window => {
  const a = oneRows.find((r: any) => r.window.id === window.id), b = twoRows.find((r: any) => r.window.id === window.id);
  assert.deepEqual(a.window, window); assert.deepEqual(b.window, window); assert.equal(a.modelHash, b.modelHash);
  assert.equal(a.trainStart, b.trainStart); assert.equal(a.trainEnd, b.trainEnd);
  const modelFile = `${window.id}-model.json`, saved = read(oneConfig.source, modelFile);
  assert.equal(createHash("sha256").update(fs.readFileSync(path.join(oneConfig.source, modelFile))).digest("hex"), b.modelHash);
  const policy = restoreEventPolicy(saved.policy), same = (x: number, y: number) => Math.abs(x - y) < policy.costs.quantityStep * 1e-4;
  const trace = read(b.auditSource, `${window.id}-trades.json`);
  const oneTrace = read(one, `${window.id}-trades.json`);
  assert.deepEqual(trace.map((r: any) => [r.time, r.endTime, r.leaf]), oneTrace.map((r: any) => [r.time, r.endTime, r.leaf]));
  assert.equal(trace.length, b.metrics.decisions); assert.equal(trace.length, b.bounds.count);
  let certified = 0, maximumGapBps = 0;
  for (const row of trace) {
    const o = row.order, validGap = Number.isFinite(o.lowerValue) && Number.isFinite(o.upperValue) && Number.isFinite(o.gap)
      && o.upperValue >= o.lowerValue && Math.abs(o.upperValue - o.lowerValue - o.gap) < 1e-12;
    assert.ok(!o.converged || validGap && o.gap <= b.tolerance);
    if (o.feasible && o.converged && validGap) certified++;
    maximumGapBps = Math.max(maximumGapBps, Number.isFinite(o.gap) ? o.gap * 10000 : Infinity);
  }
  assert.equal(certified, b.bounds.certified);
  const h1 = trace.map((r: any) => decideEventOneStep(policy.model.kernels[r.leaf],
    { equity: r.equityBefore, price: r.order.price, exposure: r.exposureBefore }, policy.costs, "marked"));
  const diagnostics = trace.map((r: any, i: number) => {
    const next = trace[i + 1], alternative = r.order.initial.candidates.find((v: any) => same(v.quantity, h1[i].quantity));
    const continuationMatches = Boolean(next && same(h1[i + 1].quantity, next.order.quantity));
    const bothOrdersFilled = Boolean(next && same(r.orderQuantity, r.order.quantity) && same(next.orderQuantity, next.order.quantity));
    // The window's final event can be truncated by the research boundary.
    const aligned = continuationMatches && bothOrdersFilled && next.endTime < window.endTime;
    const realizedTwoBps = aligned ? Math.log(next.equityAfter / r.equityBefore) * 10000 : null;
    return { window: window.id, time: new Date(r.time).toISOString(), leaf: r.leaf, exposure: r.exposureBefore,
      h1Quantity: h1[i].quantity, h2Quantity: r.order.quantity, filledQuantity: r.orderQuantity,
      exposureDifference: r.order.exposure - h1[i].exposure, changed: !same(h1[i].quantity, r.order.quantity),
      h2ValueBps: r.order.value * 10000, alternativeH2ValueBps: alternative ? alternative.value * 10000 : null,
      gapBps: r.order.gap * 10000, continuationMatches, bothOrdersFilled, aligned, realizedTwoBps,
      optimismBps: aligned ? r.order.value * 10000 - realizedTwoBps! : null };
  });
  behavior.push(...diagnostics);
  return { window, modelHash: b.modelHash, trainEnd: b.trainEnd, h1Source: one, h2Source: b.auditSource,
    h1: a.metrics, h2: b.metrics, returnDifferencePct: b.metrics.returnPct - a.metrics.returnPct,
    decisions: trace.length, certified, maximumGapBps, h2Seconds: b.elapsedSec };
});
const summarize = (key: "h1" | "h2") => ({ positive: results.filter(r => r[key].returnPct > 1e-9).length,
  negative: results.filter(r => r[key].returnPct < -1e-9).length, cash: results.filter(r => r[key].trades === 0).length,
  meanReturnPct: results.reduce((s, r) => s + r[key].returnPct, 0) / results.length,
  meanLogGrowth: results.reduce((s, r) => s + r[key].logGrowth, 0) / results.length,
  worstReturnPct: Math.min(...results.map(r => r[key].returnPct)), maximumDrawdownPct: Math.max(...results.map(r => r[key].maxDrawdownPct)),
  orders: results.reduce((s, r) => s + r[key].trades, 0), fees: results.reduce((s, r) => s + r[key].fees, 0),
  borrow: results.reduce((s, r) => s + r[key].borrow, 0), canceledOrders: results.reduce((s, r) => s + r[key].canceledOrders, 0),
  liquidations: results.reduce((s, r) => s + r[key].liquidations, 0) });
const aligned = behavior.filter(r => r.aligned), changed = behavior.filter(r => r.changed);
const summary = { catalogWindows: catalog.length, windows: results.length, decisions: results.reduce((s, r) => s + r.decisions, 0),
  certified: results.reduce((s, r) => s + r.certified, 0), allDecisionsCertified: results.every(r => r.certified === r.decisions),
  maximumGapBps: Math.max(...results.map(r => r.maximumGapBps)), h1: summarize("h1"), h2: summarize("h2"),
  improvedWindows: results.filter(r => r.returnDifferencePct > 1e-9).length,
  worsenedWindows: results.filter(r => r.returnDifferencePct < -1e-9).length,
  h2Seconds: results.reduce((s, r) => s + r.h2Seconds, 0),
  behavior: { sameAccountChangedActions: changed.length, continuationMatches: behavior.filter(r => r.continuationMatches).length,
    aligned: aligned.length, meanAlignedPredictedBps: aligned.reduce((s, r) => s + r.h2ValueBps, 0) / aligned.length,
    meanAlignedRealizedBps: aligned.reduce((s, r) => s + r.realizedTwoBps, 0) / aligned.length,
    mostOptimistic: [...aligned].sort((a, b) => b.optimismBps - a.optimismBps).slice(0, 8),
    largestActionDifferences: [...changed].sort((a, b) => Math.abs(b.exposureDifference) - Math.abs(a.exposureDifference)).slice(0, 8) }, results };
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-fixed-law-suite-merge-v1", one, two,
  method: "Require exactly one complete result per current non-fit inspector window for each horizon, equal saved model hashes and training boundaries, and valid numerical H2 gaps in every stored decision. Summaries average independent account resets across overlapping research windows; they are not a compounded portfolio or independent holdout estimate. Recompute H1 on H2-visited accounts; aligned two-event residuals require agreement at the successor, both orders filled, and no terminal-boundary event. These overlapping agreement-selected pairs are diagnostics. H2 certificates apply to finite-horizon model actions, not stationary or execution-consistent optimality." }, null, 2));
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("summary.json", summary); save("behavior.json", behavior);
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify({ "scripts/merge-event-policy-suite.ts": fs.readFileSync(__filename, "utf8") }));
console.log(JSON.stringify({ ...summary, results: undefined }));
