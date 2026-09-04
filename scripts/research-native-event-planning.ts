/** Profile or replay deeper planning under a saved, unchanged native event law. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { prepareEventTwoStep } from "../packages/bot-algo/src/event-two-step.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
import { eventAvailabilitySourceStart, validateEventMarketClosures, type EventMarketClosure } from "../packages/bot-algo/src/event-market-availability.js";

const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const config = read("config.json"), bytes = fs.readFileSync(path.join(source, "model.json"));
assert.equal(config.contract, "native-second-event-screen-v1");
const phase = arg("phase", "calibration"), mode = arg("mode", "probe"), budget = Number(arg("budget", "16"));
const count = Number(arg("count", "1")), offset = Number(arg("offset", "0")), tolerance = 1e-7;
assert.ok(["false", "true"].includes(arg("global-upper", "false")));
const globalUpper = arg("global-upper", "false") === "true";
const fullWindow = process.argv.includes("--full-window");
assert.ok(["calibration", "test"].includes(phase) && ["probe", "replay"].includes(mode));
assert.ok(!fullWindow || phase === "test", "Full-window planning applies to the saved test window");
assert.ok(Number.isInteger(budget) && budget >= 2 && Number.isInteger(count) && count > 0 && Number.isInteger(offset) && offset >= 0);
const policy = restoreEventPolicy(JSON.parse(bytes.toString()));
assert.equal(policy.model.clock.candleIntervalMs, 1000);
assert.deepEqual(policy.costs, config.costs); assert.deepEqual(policy.model.clock, config.clock);
const hash = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
const availabilityFile = arg("availability") ? path.resolve(root, arg("availability")) : undefined;
let marketClosures: EventMarketClosure[] | undefined, marketAvailability;
if (availabilityFile) {
  const bytes = fs.readFileSync(availabilityFile), manifest = JSON.parse(bytes.toString());
  assert.equal(manifest.contract, "native-event-market-availability-v1");
  assert.ok(Array.isArray(manifest.closures) && manifest.closures.length && typeof manifest.assumptions === "string" && manifest.assumptions.length);
  assert.ok(Array.isArray(manifest.sourceReferences) && manifest.sourceReferences.length);
  validateEventMarketClosures(manifest.closures);
  for (const ref of manifest.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
  marketClosures = manifest.closures;
  marketAvailability = { ...manifest, file: availabilityFile, sha256: hash(bytes) };
}
for (const ref of config.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
const start = phase === "calibration" ? config.calibrationStart : config.start;
const end = phase === "calibration" ? config.calibrationEnd : fullWindow ? config.window.endTime : config.end;
assert.ok(end > start && (!fullWindow || start === config.window.startTime));
const replayStart = start - (config.warmupCandles + 1) * 1000, replayReferences = [];
for (let day = Math.floor(eventAvailabilitySourceStart(replayStart, marketClosures ?? []) / 86400000) * 86400000; day < end; day += 86400000) {
  const file = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s", `${new Date(day).toISOString().slice(0, 10)}.json`);
  replayReferences.push({ file, sha256: hash(fs.readFileSync(file)) });
}
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-event-planning-screen-v1", source, modelHash: hash(bytes), phase, mode, start, end,
  budget, tolerance, count, offset, globalUpper, fullWindow: phase === "test" && end === config.window.endTime,
  replaySourceReferences: replayReferences,
  ...(marketAvailability ? { marketAvailability } : {}),
  method: "Same frozen full joint kernel, native completed seconds, fees, borrowing, leverage and order limits. Exact H1 control. Probe first occurrence of every H1 leaf and first held-inventory state; offset/count bound the profiling work. Replay uses receding H2 with exact H1 continuation, retaining upper/lower gaps at budget stops. Orders commit at close and attempt next open; actual terminal settlement is shared. Full-window mode extends through the saved inspector window end with separately hashed replay sources and no forecast refitting. This is finite-horizon conditional optimization, not stationary or next-open execution optimality, forecast selection, or full-inspector coverage." });
save("sources.json", Object.fromEntries(["scripts/research-native-event-planning.ts", "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-two-step.ts", "packages/bot-algo/src/event-one-step-prepared.ts",
  "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-log-policy.ts",
  "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-market-availability.ts",
  "packages/bot-algo/src/event-distribution.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const candles = loadEventCandles(replayStart, end, 1000, marketClosures);
const begin = performance.now();
const { trace, positions, ...h1 } = replayEventPolicy(candles, policy, start, end, 1, { trace: true, oneStepTerminal: "marked", marketClosures });
save("h1-trades.json", trace); save("h1-positions.json", positions);
const results: unknown[] = [];
const checkpoint = () => save("summary.json", { h1, results, elapsedSeconds: (performance.now() - begin) / 1000 });
checkpoint();
if (mode === "probe") {
  const seen = new Set<number>(), probes: typeof trace = [];
  for (const row of trace) if (!seen.has(Number(row.leaf))) { seen.add(Number(row.leaf)); probes.push(row); }
  const inventory = trace.find(row => Number(row.exposureBefore) !== 0);
  if (inventory && !probes.includes(inventory)) probes.push(inventory);
  assert.ok(offset + count <= probes.length, `Only ${probes.length} probes available`);
  const solve = prepareEventTwoStep(policy.model, policy.costs, "marked", { globalUpper });
  for (const probe of probes.slice(offset, offset + count)) {
    const order = probe.order as { price: number; quantity: number; value: number };
    const account = { equity: Number(probe.equityBefore), price: order.price, exposure: Number(probe.exposureBefore) };
    const started = performance.now(), h2 = solve(Number(probe.leaf), account, { maxEvaluations: budget, tolerance });
    const result = { time: probe.time, leaf: probe.leaf, account, rootAtoms: policy.model.kernels[Number(probe.leaf)].length,
      h1Quantity: order.quantity, h1Value: order.value, h2, elapsedSeconds: (performance.now() - started) / 1000 };
    results.push(result); checkpoint(); console.log(JSON.stringify(result));
  }
} else {
  const started = performance.now();
  const { trace: h2Trace, positions: h2Positions, ...metrics } = replayEventPolicy(candles, policy, start, end, 2,
    { trace: true, marketClosures, twoStep: { terminal: "marked", maxEvaluations: budget, tolerance,
      ...(globalUpper ? { globalUpper: true as const } : {}) },
      onDecision: row => { save("progress.json", { time: row.time, end, order: row.order }); } });
  assert.deepEqual(h2Trace.map(r => [r.time, r.leaf]), trace.map(r => [r.time, r.leaf]));
  save("h2-trades.json", h2Trace); save("h2-positions.json", h2Positions);
  const orders = h2Trace.map(r => r.order as { gap: number; converged: boolean; feasible: boolean });
  const result = { metrics, decisions: orders.length, converged: orders.filter(o => o.converged).length,
    allFeasible: orders.every(o => o.feasible), maxGapBps: Math.max(...orders.map(o => o.gap * 10000)),
    elapsedSeconds: (performance.now() - started) / 1000 };
  results.push(result); checkpoint(); console.log(JSON.stringify(result));
}
