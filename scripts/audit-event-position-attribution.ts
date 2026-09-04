/** Compare optional lifecycle attribution against an unchanged saved H1 suite. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("reference") || !arg("output")) throw new Error("Specify --reference and new --output");
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const reference = directory(arg("reference")), output = directory(arg("output"));
assert.ok(!fs.existsSync(output), "Choose a new output directory");
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const config = read(reference, "config.json"), previous = read(reference, "summary.json");
assert.equal(config.contract, "event-fixed-law-suite-audit-v1");
assert.equal(config.depth, 1); assert.equal(config.limitEvents, undefined);
assert.deepEqual(previous.results.map((r: any) => r.window.id).sort(), config.catalog.map((w: any) => w.id).sort());
assert.ok(config.catalog.every((w: any) => w.id !== "latest" && !w.id.startsWith("fit-")));
fs.mkdirSync(output, { recursive: true });
const serial = (v: unknown) => JSON.stringify(v, (_, x) => typeof x === "number" && !Number.isFinite(x) ? String(x) : x);
const normalized = (v: unknown) => JSON.parse(serial(v));
const save = (file: string, v: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(normalized(v), null, 2));
const stripTrace = (rows: Record<string, unknown>[]) => rows.map(({ positionExitDomains, positionChanges, positionQuantity, ...r }) => r);
const stripResult = (r: ReturnType<typeof replayEventPolicy>) => {
  const { positionSummary, positions, terminalPositionChanges, trace, ...metrics } = r;
  return { ...metrics, trace: stripTrace(trace) };
};
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
save("config.json", { contract: "event-position-attribution-audit-v1", reference, forecastSource: config.source,
  timingPairs: 3, depth: 1, windows: config.catalog,
  method: "Preserve each frozen forecast and exact H1 account optimizer. Compare all saved metrics and trace fields with attribution disabled, then require identical decisions, values, fills and account paths with attribution enabled. One untimed warmup per mode precedes three alternating timing pairs; data loading and verification are excluded. Timings are a local diagnostic under concurrent machine load, not proof of speed improvement. Lifecycle accounting is not an independent position-based decision policy." });
save("sources.json", Object.fromEntries(["scripts/audit-event-position-attribution.ts", "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-positions.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-second-features.ts",
  "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const results: any[] = [];
for (const old of previous.results) {
  const { window } = old, file = `${window.id}-model.json`, bytes = fs.readFileSync(path.join(config.source, file));
  const hash = createHash("sha256").update(bytes).digest("hex"); assert.equal(hash, old.modelHash);
  const policy = restoreEventPolicy(JSON.parse(bytes.toString()).policy);
  const candles = loadEventCandles(window.startTime - 2 * 86400000, window.endTime);
  const run = (enabled: boolean) => replayEventPolicy(candles, policy, window.startTime, window.endTime, 1,
    { trace: true, positionAttribution: enabled, oneStepTerminal: "marked" });
  const baseline = run(false), attributed = run(true);
  for (const key of Object.keys(old.metrics)) assert.deepEqual(normalized(baseline[key as keyof typeof baseline]), old.metrics[key], `${window.id}: ${key}`);
  assert.deepEqual(normalized(baseline.trace), read(reference, `${window.id}-trades.json`));
  assert.deepEqual(normalized(stripResult(attributed)), normalized(stripResult(baseline)));
  assert.ok(attributed.positionSummary);
  const timing = { disabledMs: [] as number[], enabledMs: [] as number[] };
  for (let pair = 0; pair < 3; pair++) {
    for (const enabled of pair % 2 ? [true, false] : [false, true]) {
      const start = performance.now(), current = run(enabled), ms = performance.now() - start;
      assert.deepEqual(normalized(stripResult(current)), normalized(stripResult(baseline)));
      timing[enabled ? "enabledMs" : "disabledMs"].push(ms);
    }
  }
  const changes = attributed.trace.flatMap((r: any) => r.positionChanges.flatMap((v: any) => v.changes));
  const result = { window: window.id, modelHash: hash, decisions: baseline.decisions, trades: baseline.trades,
    reversals: baseline.reversals, canceledOrders: baseline.canceledOrders, returnPct: baseline.returnPct,
    identicalSavedMetricsAndTrace: true, identicalAttributionMetricsAndTrace: true,
    positionSummary: attributed.positionSummary, partialReductions: changes.filter((v: any) => v.operation === "reduce").length,
    equityReconciliationError: 10000 + attributed.positionSummary.equityChange - baseline.finalEquity,
    timing: { ...timing, medianDisabledMs: median(timing.disabledMs), medianEnabledMs: median(timing.enabledMs) } };
  results.push(result); save(`${window.id}-positions.json`, attributed.positions);
  save("summary.json", { windows: results.length, decisions: results.reduce((s, r) => s + r.decisions, 0),
    trades: results.reduce((s, r) => s + r.trades, 0), canceledOrders: results.reduce((s, r) => s + r.canceledOrders, 0),
    identicalSavedMetricsAndTrace: true, identicalAttributionMetricsAndTrace: true,
    maximumEquityReconciliationError: Math.max(...results.map(r => Math.abs(r.equityReconciliationError))),
    timing: { sumMedianDisabledMs: results.reduce((s, r) => s + r.timing.medianDisabledMs, 0),
      sumMedianEnabledMs: results.reduce((s, r) => s + r.timing.medianEnabledMs, 0) }, results });
  console.log(JSON.stringify({ window: result.window, decisions: result.decisions, equityError: result.equityReconciliationError,
    disabledMs: result.timing.medianDisabledMs, enabledMs: result.timing.medianEnabledMs }));
}
