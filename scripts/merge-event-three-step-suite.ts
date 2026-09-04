/** Verify complete H3 replays and their optional saved-state bound refinements. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!["reference", "three", "output"].every(k => arg(k))) throw new Error("Specify --reference, comma-separated --three and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const reference = directory(arg("reference")), sources = arg("three").split(",").map(directory);
const refinements = arg("bounds") ? arg("bounds").split(",").map(directory) : [], output = directory(arg("output"));
const partial = process.argv.includes("--partial");
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const referenceConfig = read(reference, "config.json"), previous = read(reference, "summary.json");
assert.equal(referenceConfig.contract, "event-fixed-law-suite-merge-v1");
const catalog = new KamaInspector(path.join(root, "data")).catalog().windows.filter(w => w.id !== "latest" && !w.id.startsWith("fit-"));
assert.deepEqual(previous.results.map((r: any) => r.window.id).sort(), catalog.map(w => w.id).sort());
const forecastSource = read(referenceConfig.one, "config.json").source;
const refined = refinements.map(source => {
  const config = read(source, "config.json"); assert.equal(config.contract, "event-three-step-bound-refinement-v1");
  assert.ok(sources.includes(config.source));
  return { source, config, results: read(source, "summary.json").results };
});
const rows = sources.flatMap(source => {
  const config = read(source, "config.json"), summary = read(source, "summary.json");
  assert.equal(config.contract, "event-fixed-law-suite-audit-v1"); assert.equal(config.depth, 3);
  assert.equal(config.source, forecastSource); assert.deepEqual(config.catalog, catalog);
  assert.equal(config.limitEvents, undefined, "Prefix profiles cannot count as full coverage");
  if (summary.results.length !== config.windows.length) {
    const stopped = read(source, "stopped.json");
    assert.equal(stopped.status, "stopped", "Unfinished running batches cannot be merged");
    assert.deepEqual(stopped.completedWindows, summary.results.map((r: any) => r.window.id));
    assert.ok(summary.results.every((r: any) => config.windows.some((w: any) => w.window.id === r.window.id)));
  }
  return summary.results.map((r: any) => ({ ...r, source, tolerance: config.tolerance }));
});
assert.equal(new Set(rows.map((r: any) => r.window.id)).size, rows.length, "Exactly one replay per covered window");
assert.ok(rows.every((r: any) => catalog.some(w => w.id === r.window.id)));
if (!partial) assert.deepEqual(rows.map((r: any) => r.window.id).sort(), catalog.map(w => w.id).sort());
const behavior: unknown[] = [];
const results = catalog.filter(w => rows.some((r: any) => r.window.id === w.id)).map(window => {
  const row = rows.find((r: any) => r.window.id === window.id), old = previous.results.find((r: any) => r.window.id === window.id);
  assert.deepEqual(row.window, window); assert.ok(row.fullWindow && row.end === window.endTime);
  assert.equal(row.modelHash, old.modelHash); assert.equal(row.trainEnd, old.trainEnd);
  const modelBytes = fs.readFileSync(path.join(forecastSource, `${window.id}-model.json`));
  assert.equal(createHash("sha256").update(modelBytes).digest("hex"), row.modelHash);
  const trace = read(row.source, `${window.id}-trades.json`), oldTrace = read(old.h2Source, `${window.id}-trades.json`);
  assert.equal(trace.length, row.metrics.decisions); assert.equal(trace.length, row.bounds.count);
  assert.deepEqual(trace.map((r: any) => [r.time, r.endTime, r.leaf]), oldTrace.map((r: any) => [r.time, r.endTime, r.leaf]));
  const matching = refined.filter(r => r.config.source === row.source && r.results.some((v: any) => v.window.id === window.id));
  assert.ok(matching.length <= 1, "Use one final refinement per replay/window");
  const refinement = matching[0], bounds = refinement ? read(refinement.source, `${window.id}-bounds.json`) : undefined;
  if (bounds) assert.equal(bounds.length, trace.length);
  let certified = 0, feasible = 0, maximumGapBps = 0, maximumContinuationGapBps = 0;
  const failures = [];
  for (let i = 0; i < trace.length; i++) {
    const r = trace[i], o = r.order; assert.equal(r.optimizer, "three-event-bounds");
    assert.equal(o.terminal, "marked"); assert.equal(o.tolerance, row.tolerance);
    assert.equal(o.value, o.lowerValue);
    const lower = Number(o.lowerValue), originalUpper = Number(o.upperValue);
    let upper = originalUpper;
    if (bounds) {
      const b = bounds[i]; assert.equal(b.time, r.time); assert.equal(b.leaf, r.leaf); assert.equal(b.quantity, o.quantity);
      assert.equal(Number(b.lowerValue), lower); assert.equal(Number(b.oldUpper), originalUpper);
      upper = Number(b.upperValue); assert.ok(upper <= originalUpper + 2e-10);
      assert.ok(upper >= lower && Math.abs(upper - lower - Number(b.gap)) < 1e-12);
    }
    const valid = Number.isFinite(lower) && Number.isFinite(upper) && upper >= lower;
    const gap = valid ? upper - lower : Infinity;
    const certificate = o.feasible && valid && gap <= row.tolerance;
    assert.ok(!o.converged || Number.isFinite(originalUpper - lower) && originalUpper - lower <= row.tolerance);
    if (bounds) assert.equal(bounds[i].certified, certificate);
    feasible += Number(o.feasible && Number.isFinite(lower)); certified += Number(certificate);
    maximumGapBps = Math.max(maximumGapBps, gap * 10000);
    const chosen = o.evaluated.filter((a: any) => a.quantity === o.quantity)
      .sort((a: any, b: any) => Number(b.lowerValue) - Number(a.lowerValue))[0];
    const continuationGapBps = o.cashLowerPolicy ? 0 : chosen ? Number(chosen.gap) * 10000 : Infinity;
    maximumContinuationGapBps = Math.max(maximumContinuationGapBps, continuationGapBps);
    if (!certificate) failures.push({ decision: i + 1, time: r.time, quantity: o.quantity, gapBps: gap * 10000,
      continuationGapBps, rootEvaluations: o.rootEvaluations });
  }
  const h3 = row.metrics, h2 = old.h2, h1 = old.h1;
  const wealthDifference = h3.finalEquity - h2.finalEquity;
  const grossDifference = h3.longPnl + h3.shortPnl - h2.longPnl - h2.shortPnl;
  const feeDifference = h3.fees - h2.fees, borrowDifference = h3.borrow - h2.borrow;
  assert.ok(Math.abs(wealthDifference - grossDifference + feeDifference + borrowDifference) < 1e-6);
  const matched = trace.map((r: any, i: number) => {
    const p = oldTrace[i]; assert.equal(p.expectedReturnBps, r.expectedReturnBps);
    assert.equal(p.realizedReturnBps, r.realizedReturnBps);
    return { decision: i + 1, time: new Date(r.time).toISOString(), endTime: r.endTime, leaf: r.leaf,
      forecastBps: r.expectedReturnBps, realizedBps: r.realizedReturnBps,
      h2Quantity: p.order.quantity, h3Quantity: r.order.quantity, h2Filled: p.orderQuantity, h3Filled: r.orderQuantity,
      h2Position: p.previousQuantity + p.orderQuantity, h3Position: r.previousQuantity + r.orderQuantity,
      h2EndExposure: p.exposureAfter, h3EndExposure: r.exposureAfter,
      h2EquityChange: p.equityAfter - p.equityBefore, h3EquityChange: r.equityAfter - r.equityBefore,
      difference: (r.equityAfter - r.equityBefore) - (p.equityAfter - p.equityBefore) };
  });
  const materialChanges = matched.filter((r: any) => (r.h2Filled || r.h3Filled)
    && Math.abs(r.h3EndExposure - r.h2EndExposure) > .1);
  const leafResiduals = [...new Set(matched.map((r: any) => r.leaf))].flatMap(leaf => {
    const group = matched.filter((r: any) => r.leaf === leaf && r.endTime < window.endTime);
    return group.length ? [{ leaf, count: group.length, forecastMeanBps: group[0].forecastBps,
      realizedMeanBps: group.reduce((s: number, r: any) => s + r.realizedBps, 0) / group.length,
      signMatches: group.filter((r: any) => Math.sign(r.forecastBps) === Math.sign(r.realizedBps)).length }] : [];
  });
  behavior.push({ window: window.id, wealthDifference, grossDifference, feeDifference, borrowDifference,
    terminalSettlementDifference: wealthDifference - matched.reduce((s: number, r: any) => s + r.difference, 0),
    leafResiduals, materialChangeCount: materialChanges.length, materialChanges,
    largestGains: [...matched].sort((a, b) => b.difference - a.difference).slice(0, 4),
    largestLosses: [...matched].sort((a, b) => a.difference - b.difference).slice(0, 4),
    canceledH2Decisions: matched.filter((r: any) => r.h2Quantity && !r.h2Filled).map((r: any) => r.decision),
    canceledH3Decisions: matched.filter((r: any) => r.h3Quantity && !r.h3Filled).map((r: any) => r.decision) });
  return { window, modelHash: row.modelHash, trainEnd: row.trainEnd, h1Source: old.h1Source, h2Source: old.h2Source,
    h3Source: row.source, refinementSource: refinement?.source, h1, h2, h3,
    decisions: trace.length, feasible, certified, maximumGapBps, maximumContinuationGapBps, failures,
    returnDifferencePct: h3.returnPct - h2.returnPct, wealthDifference, grossDifference, feeDifference, borrowDifference,
    h3Seconds: row.elapsedSec };
});
const summarize = (key: "h1" | "h2" | "h3") => ({
  positive: results.filter(r => r[key].returnPct > 1e-9).length, negative: results.filter(r => r[key].returnPct < -1e-9).length,
  cash: results.filter(r => r[key].trades === 0).length,
  meanReturnPct: results.reduce((s, r) => s + r[key].returnPct, 0) / results.length,
  worstReturnPct: Math.min(...results.map(r => r[key].returnPct)), maximumDrawdownPct: Math.max(...results.map(r => r[key].maxDrawdownPct)),
  orders: results.reduce((s, r) => s + r[key].trades, 0), fees: results.reduce((s, r) => s + r[key].fees, 0),
  canceledOrders: results.reduce((s, r) => s + r[key].canceledOrders, 0), liquidations: results.reduce((s, r) => s + r[key].liquidations, 0) });
const missingWindows = catalog.filter(w => !results.some(r => r.window.id === w.id)).map(w => w.id);
const summary = { catalogWindows: catalog.length, windows: results.length, missingWindows,
  completeCoverage: missingWindows.length === 0, decisions: results.reduce((s, r) => s + r.decisions, 0),
  certified: results.reduce((s, r) => s + r.certified, 0), allCoveredDecisionsCertified: results.every(r => r.certified === r.decisions),
  maximumGapBps: Math.max(...results.map(r => r.maximumGapBps)), h1: summarize("h1"), h2: summarize("h2"), h3: summarize("h3"),
  improvedWindows: results.filter(r => r.returnDifferencePct > 1e-9).length,
  worsenedWindows: results.filter(r => r.returnDifferencePct < -1e-9).length, results };
fs.mkdirSync(output, { recursive: true });
const save = (f: string, x: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(x,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-three-step-suite-merge-v1", reference, sources, refinements, partial,
  method: "Require completed batches or explicitly recorded stopped batches with an exact list of completed windows. Merge only unique full windows from the current non-fit inspector catalog, with unchanged forecast hashes, matched H2/H3 event clocks and internally consistent global gaps. An optional refinement must preserve each saved root quantity and feasible lower value. Partial mode explicitly lists missing windows; summary returns compare independent account resets over the same covered subset, not a compounded portfolio or fresh holdout. Certificates apply to finite H3 decision-price values, not stationary or next-open execution optimality." });
save("sources.json", { "scripts/merge-event-three-step-suite.ts": fs.readFileSync(__filename, "utf8") });
save("behavior.json", { method: "Match H2/H3 event times on each verified frozen law. Differences describe full account trajectories, not isolated counterfactual action effects. Material changes require an executed order and more than 0.1 difference in end-of-event exposure. Largest gains/losses rank net event equity differences; terminal settlement is reconciled separately. Leaf forecast residuals exclude the event ending at the window boundary. No retraining or return-based policy selection.", results: behavior });
save("summary.json", summary); console.log(JSON.stringify({ ...summary, results: undefined }));
