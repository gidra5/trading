/** Replay a long-horizon finite-grid policy against exact and grid H3 controls. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { buildEventPolicy, restoreEventPolicy, serializeEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventCalibrationRanges, loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const argument = (name: string, fallback = "") => {
  const at = process.argv.indexOf(`--${name}`); return at < 0 ? fallback : process.argv[at + 1];
};
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.join(root, "data/benchmarks", name);
const source = directory(argument("source", "event-policy-all-mean120-1x-cap-v10"));
const exactSource = directory(argument("exact-source", "event-policy-three-event-full-suite-v390"));
const output = directory(argument("output"));
const depth = Number(argument("depth", "64")), actionSteps = Number(argument("action-steps", "10"));
const requested = argument("windows", "all").split(",");
const calibrationSelect = process.argv.includes("--calibration-select"), riskPenalty = Number(argument("risk-penalty", "0.1"));
if (!argument("output")) throw new Error("Specify a new --output directory");
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
if (!Number.isInteger(depth) || depth < 4 || depth > 512 || !Number.isInteger(actionSteps) || actionSteps < 1 || actionSteps > 100)
  throw new Error("Invalid long-horizon replay policy");
if (!Number.isFinite(riskPenalty) || riskPenalty < 0) throw new Error("Invalid calibration risk penalty");
const read = (d: string, file: string) => JSON.parse(fs.readFileSync(path.join(d, file), "utf8"));
const sourceConfig = read(source, "config.json"), sourceSummary = read(source, "summary.json");
const exactConfig = read(exactSource, "config.json"), exactSummary = read(exactSource, "summary.json");
assert.equal(sourceConfig.contract, "causal-event-tree-bellman-v1");
assert.equal(exactConfig.contract, "event-three-step-suite-merge-v1");
assert.equal(exactSummary.completeCoverage, true); assert.equal(exactSummary.catalogWindows, exactSummary.windows);
const sourceRows = Array.isArray(sourceSummary) ? sourceSummary : sourceSummary.results;
assert.deepEqual(sourceRows.map((r: any) => r.window.id).sort(), exactSummary.results.map((r: any) => r.window.id).sort());
assert.ok(requested[0] === "all" || requested.every(id => exactSummary.results.some((r: any) => r.window.id === id)));
const selected = exactSummary.results.filter((r: any) => requested[0] === "all" || requested.includes(r.window.id));

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-grid-long-horizon-replay-v1", source, exactSource, depth, actionSteps,
  calibrationSelect, riskPenalty,
  windows: selected.map((r: any) => r.window.id), terminal: "marked",
  method: "Freeze every previously fitted event law. Rebuild the finite equity/price/exposure interpolation policy with a marked Bellman boundary, then replay its receding H3 and declared long horizon through the same next-open simulator and terminal settlement as the certified exact H3 control. Returns are reused-window diagnostics. The long policy is optimal only in the finite interpolated action/state grid, not certified on the full exchange lot lattice." });
save("sources.json", Object.fromEntries([
  "scripts/replay-event-grid-horizon-suite.ts", "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-distribution.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));

const results: any[] = [], started = performance.now();
for (const exact of selected) {
  const begin = performance.now(), filename = `${exact.window.id}-model.json`;
  const bytes = fs.readFileSync(path.join(source, filename));
  assert.equal(createHash("sha256").update(bytes).digest("hex"), exact.modelHash);
  const saved = JSON.parse(bytes.toString()), base = restoreEventPolicy(saved.policy);
  const policy = buildEventPolicy(base.model, base.costs, { depths: depth, referenceEquity: base.equities[2],
    referencePrice: base.prices[1], actionSteps, terminal: "marked" });
  assert.ok(Number.isFinite(saved.policyCalibrationStart) && saved.policyCalibrationStart < exact.window.startTime);
  const candles = loadEventCandles(saved.policyCalibrationStart - 2 * 86_400_000, exact.window.endTime);
  const excluded = saved.selectionExcludedWindows ?? sourceConfig.excludedWindows;
  assert.ok(Array.isArray(excluded));
  const ranges = eventCalibrationRanges(saved.policyCalibrationStart, exact.window.startTime, excluded);
  const calibrate = (candidateDepth: number) => {
    const runs = ranges.map(r => replayEventPolicy(candles, policy, r.startTime, r.endTime, candidateDepth));
    const logGrowth = runs.reduce((s, r) => s + r.logGrowth, 0);
    const maximumDrawdown = Math.max(0, ...runs.map(r => r.maxDrawdownPct)) / 100;
    return { depth: candidateDepth, ranges: ranges.length, logGrowth, maximumDrawdown,
      score: logGrowth - riskPenalty * maximumDrawdown };
  };
  const calibration = calibrationSelect ? [calibrate(3), calibrate(depth)].sort((a, b) => b.score - a.score)
    : [{ depth, ranges: 0, logGrowth: 0, maximumDrawdown: 0, score: 0 }];
  const chosen = calibration[0], cash = calibrationSelect && (!ranges.length || chosen.score <= 0);
  // Commit the causal selection artifact before invoking either scored replay.
  save(`${exact.window.id}-selection.json`, { modelHash: exact.modelHash, policy: serializeEventPolicy(policy),
    policyCalibrationStart: saved.policyCalibrationStart, calibrationEnd: exact.window.startTime,
    excluded, calibration, chosenDepth: chosen.depth, cash });
  const h3Replay = replayEventPolicy(candles, policy, exact.window.startTime, exact.window.endTime, 3, { trace: true });
  const longReplay = replayEventPolicy(candles, policy, exact.window.startTime, exact.window.endTime, depth, { trace: true });
  const selectedReplay = cash ? replayEventPolicy(candles, policy, exact.window.startTime, exact.window.endTime, 3, { cash: true, trace: true })
    : chosen.depth === 3 ? h3Replay : longReplay;
  const exactTrace = read(exact.h3Source, `${exact.window.id}-trades.json`);
  assert.deepEqual(h3Replay.trace.map((r: any) => [r.time, r.endTime, r.leaf]),
    exactTrace.map((r: any) => [r.time, r.endTime, r.leaf]));
  assert.deepEqual(longReplay.trace.map((r: any) => [r.time, r.endTime, r.leaf]),
    exactTrace.map((r: any) => [r.time, r.endTime, r.leaf]));
  const strip = ({ trace, ...metrics }: any) => metrics;
  const h3Grid = strip(h3Replay), long = strip(longReplay), selectedMetrics = strip(selectedReplay);
  const paired = longReplay.trace.map((row: any, i: number) => {
    const old = h3Replay.trace[i];
    return { decision: i + 1, time: row.time, leaf: row.leaf, h3Exposure: old.order.exposure,
      longExposure: row.order.exposure, exposureDifference: row.order.exposure - old.order.exposure,
      h3Executed: old.orderQuantity, longExecuted: row.orderQuantity };
  });
  const result = { window: exact.window, modelHash: exact.modelHash, exactH3: exact.h3, gridH3: h3Grid, long,
    selected: selectedMetrics, calibration, chosenDepth: chosen.depth, cash,
    h3GridDifferencePct: h3Grid.returnPct - exact.h3.returnPct,
    longDifferenceFromExactH3Pct: long.returnPct - exact.h3.returnPct,
    longDifferenceFromGridH3Pct: long.returnPct - h3Grid.returnPct,
    actionChanges: paired.filter((r: any) => Math.abs(r.exposureDifference) > 1e-8).length,
    maximumExposureChange: Math.max(0, ...paired.map((r: any) => Math.abs(r.exposureDifference))),
    finalConvergence: policy.tables.at(-1)!.convergence,
    elapsedSec: (performance.now() - begin) / 1000 };
  save(`${exact.window.id}-grid-h3-trades.json`, h3Replay.trace);
  save(`${exact.window.id}-long-trades.json`, longReplay.trace);
  save(`${exact.window.id}-selected-trades.json`, selectedReplay.trace);
  save(`${exact.window.id}-comparison.json`, { ...result, paired });
  results.push(result); save("progress.json", { results, elapsedSec: (performance.now() - started) / 1000 });
  console.log(JSON.stringify({ window: exact.window.id, exactH3: exact.h3.returnPct, gridH3: h3Grid.returnPct,
    long: long.returnPct, selected: selectedMetrics.returnPct, chosenDepth: chosen.depth, cash,
    changeFromExact: result.longDifferenceFromExactH3Pct,
    actionChanges: result.actionChanges, maximumExposureChange: result.maximumExposureChange,
    elapsedSec: result.elapsedSec }));
}
type Key = "exactH3" | "gridH3" | "long" | "selected";
const summarize = (key: Key) => ({ positive: results.filter(r => r[key].returnPct > 1e-9).length,
  negative: results.filter(r => r[key].returnPct < -1e-9).length, cash: results.filter(r => r[key].trades === 0).length,
  meanReturnPct: results.reduce((s, r) => s + r[key].returnPct, 0) / results.length,
  worstReturnPct: Math.min(...results.map(r => r[key].returnPct)),
  maximumDrawdownPct: Math.max(...results.map(r => r[key].maxDrawdownPct)),
  orders: results.reduce((s, r) => s + r[key].trades, 0), fees: results.reduce((s, r) => s + r[key].fees, 0),
  canceledOrders: results.reduce((s, r) => s + r[key].canceledOrders, 0), liquidations: results.reduce((s, r) => s + r[key].liquidations, 0) });
const summary = { windows: results.length, depth, actionSteps, exactH3: summarize("exactH3"), gridH3: summarize("gridH3"),
  long: summarize("long"), selected: summarize("selected"),
  selectedH3: results.filter(r => !r.cash && r.chosenDepth === 3).length,
  selectedLong: results.filter(r => !r.cash && r.chosenDepth === depth).length, selectedCash: results.filter(r => r.cash).length,
  improvedVsExactH3: results.filter(r => r.longDifferenceFromExactH3Pct > 1e-9).length,
  worsenedVsExactH3: results.filter(r => r.longDifferenceFromExactH3Pct < -1e-9).length,
  selectedImprovedVsExactH3: results.filter(r => r.selected.returnPct - r.exactH3.returnPct > 1e-9).length,
  selectedWorsenedVsExactH3: results.filter(r => r.selected.returnPct - r.exactH3.returnPct < -1e-9).length,
  improvedVsGridH3: results.filter(r => r.longDifferenceFromGridH3Pct > 1e-9).length,
  worsenedVsGridH3: results.filter(r => r.longDifferenceFromGridH3Pct < -1e-9).length,
  actionChanges: results.reduce((s, r) => s + r.actionChanges, 0),
  maximumExposureChange: Math.max(...results.map(r => r.maximumExposureChange)),
  elapsedSec: (performance.now() - started) / 1000, results };
save("summary.json", summary); console.log(JSON.stringify({ ...summary, results: undefined }));
