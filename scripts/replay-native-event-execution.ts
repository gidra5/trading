/** Receding execution-aware H1 against an unchanged empirical forecasting law. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventAvailabilitySourceStart, validateEventMarketClosures } from "../packages/bot-algo/src/event-market-availability.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output")), phase = arg("phase", "calibration");
assert.ok(arg("source") && arg("output") && !fs.existsSync(output) && ["calibration", "test"].includes(phase));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const executionConfig = read(path.join(source, "config.json")), law = read(path.join(source, "law.json"));
const compiled = read(path.join(source, "summary.json"));
assert.equal(executionConfig.contract, "native-event-execution-law-v1"); assert.equal(hash(path.join(source, "law.json")), compiled.lawHash);
const baseSource = executionConfig.source, config = read(path.join(baseSource, "config.json")), modelFile = path.join(baseSource, "model.json");
assert.equal(hash(modelFile), law.modelHash); const policy = restoreEventPolicy(read(modelFile));
for (const ref of executionConfig.sourceReferences) assert.equal(hash(ref.file), ref.sha256);
let marketAvailability, marketClosures;
if (arg("availability")) {
  const file = path.resolve(root, arg("availability")), manifest = read(file);
  assert.equal(manifest.contract, "native-event-market-availability-v1"); validateEventMarketClosures(manifest.closures);
  for (const ref of manifest.sourceReferences) assert.equal(hash(ref.file), ref.sha256);
  marketClosures = manifest.closures; marketAvailability = { ...manifest, file, sha256: hash(file) };
}
const start = phase === "calibration" ? config.calibrationStart : config.window.startTime;
const end = phase === "calibration" ? config.calibrationEnd : config.window.endTime;
const sourceStart = start - (config.warmupCandles + 1) * 1000, references = [];
for (let day = Math.floor(eventAvailabilitySourceStart(sourceStart, marketClosures ?? []) / 86400000) * 86400000; day < end; day += 86400000) {
  const file = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s", `${new Date(day).toISOString().slice(0, 10)}.json`);
  references.push({ file, sha256: hash(file) });
}
fs.mkdirSync(output, { recursive: true });
const save = (name: string, data: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(data,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-event-execution-one-step-replay-v1", source, baseSource, modelHash: law.modelHash,
  lawHash: compiled.lawHash, window: config.window, phase, start, end, fullWindow: phase === "test", costs: policy.costs,
  terminal: "marked", replaySourceReferences: references, ...(marketAvailability ? { marketAvailability } : {}),
  method: "Globally searched one-event base-quantity requests under a frozen empirical execution-path mixture. The original joint return/duration/extrema/successor law is unchanged. Both policies use the same next-open replay and actual terminal settlement; the control optimizes its original decision-price model. Above-entry-cap no-trade semantics follow the replay in the execution optimizer. Trace order equity/exposure describe the pre-order account and turnover/cost are decision-price diagnostics, not a fictitious fill; confirmed fills and position changes remain separate. This is receding H1, not deeper/stationary optimization or a new independent holdout." });
save("sources.json", Object.fromEntries(["scripts/replay-native-event-execution.ts", "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-market-availability.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const candles = loadEventCandles(sourceStart, end, 1000, marketClosures), started = performance.now();
const { trace: controlTrace, positions: controlPositions, ...control } = replayEventPolicy(candles, policy, start, end, 1,
  { trace: true, oneStepTerminal: "marked", marketClosures });
save("control-trades.json", controlTrace); save("control-positions.json", controlPositions);
if (phase === "calibration") assert.deepEqual(JSON.parse(JSON.stringify(controlTrace)), read(path.join(baseSource, "calibration-trades.json")));
const executionOneStep = law.kernels.map((kernel: any[]) => kernel.map(atom => ({ probability: atom.probability, next: atom.next, path: law.paths[atom.path] })));
save("progress.json", { phase: "execution", start, end });
const begin = performance.now();
const { trace, positions, ...execution } = replayEventPolicy(candles, policy, start, end, 1,
  { trace: true, oneStepTerminal: "marked", executionOneStep, marketClosures,
    onDecision: row => save("progress.json", { time: row.time, end, order: row.order }) });
const executionSeconds = (performance.now() - begin) / 1000;
const common = Math.min(trace.length, controlTrace.length);
assert.deepEqual(trace.slice(0, common).map(r => [r.time, r.leaf]), controlTrace.slice(0, common).map(r => [r.time, r.leaf]));
if (!execution.liquidations && !control.liquidations) assert.equal(trace.length, controlTrace.length);
save("execution-trades.json", trace); save("execution-positions.json", positions);
const orders = trace.map(r => r.order as { complete: boolean; feasible: boolean; search: { evaluatedOrders: number } });
assert.ok(orders.every(o => o.complete));
save("summary.json", { control, execution, decisions: orders.length, completeSearches: orders.filter(o => o.complete).length,
  finiteValueDecisions: orders.filter(o => o.feasible).length, evaluatedOrders: orders.reduce((s, o) => s + o.search.evaluatedOrders, 0),
  executionSeconds, elapsedSeconds: (performance.now() - started) / 1000 });
console.log(JSON.stringify({ window: config.window.id, phase, decisions: orders.length, finiteValueDecisions: orders.filter(o => o.feasible).length,
  controlReturnPct: control.returnPct, executionReturnPct: execution.returnPct, controlOrders: control.trades, executionOrders: execution.trades,
  controlCancellations: control.canceledOrders, executionCancellations: execution.canceledOrders, executionSeconds }));
