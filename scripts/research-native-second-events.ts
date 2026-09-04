/** Small chronological native-second fit and exact-H1 replay before scaling. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
import { NATIVE_SECOND_EVENT_FEATURES, NATIVE_SECOND_CONTEXT_FEATURES, NATIVE_SECOND_DAY_CONTEXT_FEATURES } from "../packages/bot-algo/src/event-second-features.js";
import { distributionMetrics, eventLeaf, eventFeatureWarmup, trainEventDistribution, type EventClock, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { buildEventPolicy, DEFAULT_EVENT_COSTS, serializeEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventMarginalCrps } from "../packages/bot-algo/src/event-crps.js";
import { loadNativeEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
import { eventFitPeriods, eventSourceDays, mergeEventSourceRanges } from "./event-fit-periods.js";

const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), output = path.resolve(root, "data/benchmarks", arg("output"));
assert.ok(arg("output") && !fs.existsSync(output), "Specify a new --output directory");
const catalog = new KamaInspector(path.join(root, "data")).catalog().windows;
const window = catalog.find(w => w.id === arg("window"));
assert.ok(window && window.id !== "latest" && !window.id.startsWith("fit-"), "Specify a non-fit --window");
const mode = arg("clock", "run"), fitDays = Number(arg("fit-days", "2")), testHours = Number(arg("test-hours", "1"));
const stride = Number(arg("stride", "30")), maxCandles = mode === "next-second" ? 1 : Number(arg("max-candles", "60"));
assert.ok(["run", "next-second", "barrier"].includes(mode) && Number.isInteger(fitDays) && fitDays >= 1 && fitDays <= 7);
assert.ok(["fast", "context", "day-context"].includes(arg("features", "fast")));
const featureNames = arg("features", "fast") === "day-context" ? NATIVE_SECOND_DAY_CONTEXT_FEATURES
  : arg("features", "fast") === "context" ? NATIVE_SECOND_CONTEXT_FEATURES : NATIVE_SECOND_EVENT_FEATURES;
const estimationMode = arg("estimation", "shared"), separateEstimation = estimationMode !== "shared";
assert.ok(["shared", "later-fit-chain", "later-fit-stride"].includes(estimationMode) && (!separateEstimation || fitDays >= 2));
assert.ok(Number.isFinite(testHours) && testHours > 0 && testHours <= 24 && Number.isInteger(testHours * 3600));
assert.ok(Number.isInteger(stride) && stride >= 1 && stride <= 60);
const DAY = 86400000, start = window.startTime, end = Math.min(window.endTime, start + testHours * 3600000);
const excluded = catalog.filter(w => w.id !== "latest");
const clock: EventClock = { candleIntervalMs: 1000, thresholdBps: Number(arg("threshold-bps", "1")),
  maxCandles, ...(mode === "run" ? { runClock: true } : {}) };
if (arg("duration-bins-seconds")) {
  const bins = arg("duration-bins-seconds").split(",").map(Number);
  assert.ok(bins.length === 2 && bins.every(v => Number.isFinite(v) && v > 0) && bins[0] < bins[1]);
  clock.durationBinsMinutes = [bins[0] / 60, bins[1] / 60];
}
const warmup = eventFeatureWarmup(clock, featureNames), periods = eventFitPeriods(start, fitDays, (warmup + 1) * 1000, excluded);
const { fitStart, fitEnd, calibrationEnd, sourceStart } = periods;
const sourceRanges = mergeEventSourceRanges([{ start: sourceStart, end: calibrationEnd },
  { start: start - (warmup + 1) * 1000, end }]);
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
const refs = [];
for (const day of eventSourceDays(sourceRanges)) {
  const file = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s", `${new Date(day).toISOString().slice(0, 10)}.json`);
  refs.push({ file, sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex") });
}
save("config.json", { contract: "native-second-event-screen-v1", window, start, end, fullWindow: end === window.endTime,
  fitStart, fitEnd, calibrationStart: fitEnd, calibrationEnd, fitPeriods: periods, excluded, stride, clock,
  featureNames, warmupCandles: warmup, sourceRanges, sourceReferences: refs, costs: DEFAULT_EVENT_COSTS,
  estimationMode,
  estimation: separateEstimation ? `Partition on earlier fit days; estimate on the final fit day using ${estimationMode === "later-fit-chain" ? "a complete non-overlapping event chain" : "the same fixed-stride origins"}. The following calibration day remains diagnostic only.` : "Partition and estimate from the same fixed-stride fit population.",
  method: "Native completed 1s OHLCV, declared bounded feature support, fixed-stride training origins, purged complete input/target support against all inspector windows including fit windows. Choose the latest whole-day fit/calibration block whose full feature support avoids every catalog exclusion, recording any backward date shifts without consulting returns. A depth-2 distribution tree is frozen before a separate diagnostic calibration day and the test prefix; neither segment selects or recalibrates it. Replay observes every causal run boundary (including its revealing candle) or each observed price barrier/timeout or every next second. Timeouts are decision boundaries, not claims that a run ended. Move and borrowing durations remain physical minutes; class-duration bins use seconds. Orders are fixed at the completed close and attempted at the next open. Exact marked-terminal H1, with actual fee-paying terminal replay settlement. This bounded screen is not full-window coverage, the production59 sign model, or a minute-versus-second comparison." });
save("sources.json", Object.fromEntries(["scripts/research-native-second-events.ts", "scripts/research-event-policy.ts", "scripts/event-fit-periods.ts",
  "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-second-features.ts", "packages/bot-algo/src/event-log-policy.ts",
  "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-positions.ts", "packages/bot-algo/src/event-crps.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const timing: Record<string, number> = {}, timed = <T>(label: string, fn: () => T): T => {
  const begin = performance.now(), value = fn(); timing[label] = (performance.now() - begin) / 1000; return value;
};
const candles = timed("loadSeconds", () => loadNativeEventCandles(sourceRanges));
const training = timed("trainingSampleSeconds", () => makeSamples(candles, clock, fitStart,
  separateEstimation ? fitEnd - DAY : fitEnd, excluded, stride, "stride", featureNames));
const estimation = separateEstimation ? timed("estimationSampleSeconds", () =>
  makeSamples(candles, clock, fitEnd - DAY, fitEnd, excluded,
    estimationMode === "later-fit-chain" ? 1 : stride, estimationMode === "later-fit-chain" ? "chain" : "stride", featureNames)) : undefined;
if (estimation) {
  assert.ok(estimation.length && training.every(row => row.end < estimation[0].start));
  if (estimationMode === "later-fit-chain") assert.ok(estimation.every((row, i) => !i || row.start >= estimation[i - 1].end));
}
const calibration = timed("calibrationSampleSeconds", () => makeSamples(candles, clock, fitEnd, calibrationEnd, excluded, stride, "stride", featureNames));
// The known test window is intentionally scored; it remains excluded from fitting.
const test = timed("testSampleSeconds", () => makeSamples(candles, clock, start, end, [], 1, "chain", featureNames));
assert.ok(training.length >= 256 && calibration.length > 0 && test.length > 0);
const distribution = timed("trainSeconds", () => trainEventDistribution(training, clock,
  { maxDepth: 2, minLeaf: 128, prior: 32, criterion: "distribution", featureNames, estimationSamples: estimation }));
const policy = buildEventPolicy(distribution, DEFAULT_EVENT_COSTS,
  { depths: 0, referenceEquity: 10000, referencePrice: candles[test[0].start].close });
save("model.json", serializeEventPolicy(policy));
const quantiles = (values: number[]) => {
  const sorted = values.slice().sort((a, b) => a - b);
  return Object.fromEntries([.5, .9, .99, 1].map(q => [q, sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]]));
};
const describe = (samples: MoveSample[]) => {
  const signs = [0, 0, 0], confusion = Array.from({ length: 3 }, () => [0, 0, 0]);
  const masses = distribution.kernels.map(k => [-1, 0, 1].map(sign => k.reduce((s, a) => s + (Math.sign(a.return) === sign ? a.probability : 0), 0)));
  let correct = 0, nonzero = 0, correctNonzero = 0, timedOut = 0;
  const durationClassCounts = [0, 0, 0];
  for (const s of samples) {
    const leaf = eventLeaf(distribution, s.features), actual = Math.sign(s.return) + 1, m = masses[leaf];
    durationClassCounts[s.label % 3]++;
    const prediction = m.indexOf(Math.max(...m)); signs[actual]++; confusion[actual][prediction]++; correct += Number(actual === prediction);
    if (actual !== 1) { nonzero++; correctNonzero += Number((m[2] > m[0] ? 2 : 0) === actual); }
    if (mode === "run" && Math.sign(candles[s.end].close - candles[s.end - 1].close)
      === Math.sign(candles[s.start].close - candles[s.start - 1].close)) timedOut++;
    if (mode !== "run" && s.end - s.start === maxCandles && Math.abs(s.return) * 10000 < clock.thresholdBps) timedOut++;
  }
  return { ...distributionMetrics(distribution, samples), signs, confusion, accuracy: correct / samples.length,
    ...eventMarginalCrps(distribution, samples), durationClassCounts,
    nonzero, nonzeroDirectionAccuracy: correctNonzero / nonzero, timedOut,
    durationSeconds: quantiles(samples.map(s => s.duration * 60)), absoluteReturnBps: quantiles(samples.map(s => Math.abs(s.return) * 10000)) };
};
const forecast = { training: describe(training), ...(estimation ? { estimation: describe(estimation) } : {}), calibration: describe(calibration), test: describe(test),
  leaves: distribution.kernels.map((k, i) => ({ leaf: i, count: distribution.counts[i], atoms: k.length,
    meanReturnBps: k.reduce((s, a) => s + a.probability * a.return * 10000, 0),
    meanDurationSeconds: k.reduce((s, a) => s + a.probability * a.duration * 60, 0) })) };
save("forecast.json", forecast);
console.log(JSON.stringify({ phase: "forecast", rows: candles.length, training: training.length, test: test.length, timing }));
const { trace, positions, ...metrics } = timed("replaySeconds", () => replayEventPolicy(candles, policy, start, end, 1,
  { trace: true, oneStepTerminal: "marked", onDecision: row => {
    if (Number(row.time) % 300000 < 1000) save("progress.json", { time: row.time, end, equity: row.equityAfter });
  } }));
save("trades.json", trace); save("positions.json", positions);
const summary = { window, start, end, fullWindow: end === window.endTime, candles: candles.length, training: training.length,
  estimation: estimation?.length,
  calibration: calibration.length, test: test.length, metrics, timing,
  allDecisionsFeasible: trace.every((r: any) => r.order.feasible && Number.isFinite(r.order.value)),
  processMemoryMiB: Object.fromEntries(Object.entries(process.memoryUsage()).map(([key, value]) => [key, value / 1048576])) };
save("summary.json", summary); console.log(JSON.stringify(summary));
