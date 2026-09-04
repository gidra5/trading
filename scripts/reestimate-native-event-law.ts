/** Add earlier estimation data to a frozen native tree; never relearn its states. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { distributionMetrics, eventLeaf, reestimateEventTree, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventMarginalCrps } from "../packages/bot-algo/src/event-crps.js";
import { eventCashHorizon } from "../packages/bot-algo/src/event-cash-horizon.js";
import { eventAverageUniqueness } from "../packages/bot-algo/src/event-sampling.js";
import { restoreEventPolicy, serializeEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { loadNativeEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
import { eventSourceDays, mergeEventSourceRanges } from "./event-fit-periods.js";

const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const extraDays = Number(arg("extra-days", "3")), sampling = arg("sampling", "stride"), priorStrength = Number(arg("prior", "32"));
const weighting = arg("weighting", "unit"), priorMode = arg("prior-mode", "fixed");
assert.ok(Number.isInteger(extraDays) && extraDays >= 0 && extraDays <= 14 && ["stride", "chain"].includes(sampling));
assert.ok(Number.isFinite(priorStrength) && priorStrength >= 0);
assert.ok(["unit", "uniqueness"].includes(weighting) && ["fixed", "matched-ratio"].includes(priorMode));
const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const config = read("config.json"), oldSummary = read("summary.json"), modelBytes = fs.readFileSync(path.join(source, "model.json"));
assert.equal(config.contract, "native-second-event-screen-v1");
assert.ok(config.estimation.startsWith("Partition on earlier fit days") && oldSummary.estimation);
const baseline = restoreEventPolicy(JSON.parse(modelBytes.toString()));
assert.equal(baseline.model.clock.candleIntervalMs, 1000); assert.equal(baseline.tables.length, 0);
const DAY = 86400000, partitionStart = config.fitStart, partitionEnd = config.fitEnd - DAY;
const earlierStart = partitionStart - extraDays * DAY, finalStart = config.fitEnd - DAY;
const sourceStart = (extraDays ? earlierStart : finalStart) - (config.warmupCandles + 1) * 1000;
const sourceRanges = mergeEventSourceRanges([
  ...(extraDays ? [{ start: sourceStart, end: partitionStart }] : []),
  { start: finalStart - (config.warmupCandles + 1) * 1000, end: config.calibrationEnd },
  { start: config.start - (config.warmupCandles + 1) * 1000, end: config.end },
]);
const hash = (data: Buffer | string) => createHash("sha256").update(data).digest("hex");
for (const ref of config.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
const references = [];
for (const day of eventSourceDays(sourceRanges)) {
  const file = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s", `${new Date(day).toISOString().slice(0, 10)}.json`);
  references.push({ file, sha256: hash(fs.readFileSync(file)) });
}
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("sources.json", Object.fromEntries(["scripts/reestimate-native-event-law.ts", "scripts/research-event-policy.ts", "scripts/event-fit-periods.ts",
  "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-second-features.ts", "packages/bot-algo/src/event-crps.ts", "packages/bot-algo/src/event-cash-horizon.ts",
  "packages/bot-algo/src/event-sampling.ts", "packages/bot-algo/src/event-log-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), candles = loadNativeEventCandles(sourceRanges);
const sample = (start: number, end: number, mode: "stride" | "chain", excluded = config.excluded) =>
  makeSamples(candles, baseline.model.clock, start, end, excluded, mode === "chain" ? 1 : config.stride, mode, baseline.model.featureNames);
const later = sample(finalStart, config.fitEnd, sampling as "stride" | "chain");
const earlier = extraDays ? sample(earlierStart, partitionStart, sampling as "stride" | "chain") : [];
const estimation = [...earlier, ...later];
assert.ok(estimation.length && earlier.every(s => candles[s.end].openTime + 1000 < partitionStart)
  && later.every(s => candles[s.start].openTime + 1000 >= partitionEnd && candles[s.end].openTime + 1000 < config.calibrationStart));
const weights = weighting === "uniqueness" ? eventAverageUniqueness(estimation) : undefined;
const estimationMass = weights?.reduce((sum, w) => sum + w, 0) ?? estimation.length;
const prior = priorMode === "fixed" ? priorStrength : priorStrength * estimationMass / estimation.length;
save("config.json", { ...config, sourceRanges, sourceReferences: references, frozenPartitionSource: source, frozenPartitionModelHash: hash(modelBytes),
  partitionStart, partitionEnd, earlierEstimationStart: extraDays ? earlierStart : null, earlierEstimationEnd: extraDays ? partitionStart : null,
  finalEstimationStart: finalStart, extraDays, sampling, weighting, priorMode, priorStrength, prior, estimationMass,
  estimationMode: `frozen-partition-${sampling}`,
  estimation: "Frozen parent partition. Estimate only from the final fit day plus declared earlier periods before partition training. Calibration and test remain diagnostic.",
  method: "Same frozen tree nodes, features, event clock and account costs. Additional earlier label intervals exclude partition-training dates; all inputs and labels are purged against every inspector window including fit windows. All estimation data precedes calibration. Historical feature support can overlap between training populations, so disjoint labels do not imply statistical independence. Optional average-uniqueness weights use only completed estimation labels, preserving full joint atoms and weighting the prior as well. Matched-ratio mode scales prior strength by weight mass / raw sample count. This weight mass is not an independent sample count. CRPS compares identical physical targets. This is a bounded forecast-estimation ablation, not strategy promotion or a new independent holdout." });
if (weights) save("weights.json", estimation.map((s, i) => ({ start: candles[s.start].openTime + 1000,
  end: candles[s.end].openTime + 1000, weight: weights[i] })));
const model = reestimateEventTree(baseline.model, estimation, prior, weights), policy = { ...baseline, model };
assert.deepEqual(model.nodes, baseline.model.nodes);
if (!extraDays && sampling === "stride" && weighting === "unit" && prior === 32)
  assert.deepEqual(model, baseline.model, "Zero-extra-day control must reproduce the parent law");
save("model.json", serializeEventPolicy(policy));
const calibration = sample(config.calibrationStart, config.calibrationEnd, "stride"), test = sample(config.start, config.end, "chain", []);
assert.equal(calibration.length, oldSummary.calibration); assert.equal(test.length, oldSummary.test);
const nonOverlapping = (group: MoveSample[]) => {
  let end = -Infinity, count = 0;
  for (const s of group.slice().sort((a, b) => a.end - b.end || a.start - b.start)) if (s.start >= end) { count++; end = s.end; }
  return count;
};
const support = (rows: MoveSample[]) => model.kernels.map((_, leaf) => {
  const group = rows.filter(s => eventLeaf(model, s.features) === leaf);
  return { leaf, count: group.length, maxNonOverlappingLabels: nonOverlapping(group),
    observedMeanBps: group.length ? group.reduce((sum, s) => sum + s.return * 10000, 0) / group.length : null };
});
const describe = (rows: MoveSample[]) => ({ ...distributionMetrics(model, rows), ...eventMarginalCrps(model, rows),
  baseline: { ...distributionMetrics(baseline.model, rows), ...eventMarginalCrps(baseline.model, rows) },
  physicalTargetsHash: hash(JSON.stringify(rows.map(s => [candles[s.start].openTime, candles[s.end].openTime, s.return, s.duration, s.low, s.high]))),
  support: support(rows) });
const means = model.kernels.map(k => k.reduce((sum, a) => sum + a.probability * a.return, 0));
const forecast = { estimation: describe(estimation), calibration: describe(calibration), test: describe(test),
  earlier: support(earlier), later: support(later),
  leaves: model.kernels.map((k, leaf) => ({ leaf, count: model.counts[leaf], meanReturnBps: means[leaf] * 10000,
    twoEventMeanBps: k.reduce((sum, a) => sum + a.probability * ((1 + a.return) * (1 + means[a.next]) - 1) * 10000, 0) })) };
save("forecast.json", forecast);
const bounds = eventCashHorizon(model, policy.costs, 100); save("cash-bound.json", bounds);
const replay = (phase: "calibration" | "test", start: number, end: number) => {
  const { trace, positions, ...metrics } = replayEventPolicy(candles, policy, start, end, 1, { trace: true, oneStepTerminal: "marked" });
  save(`${phase}-trades.json`, trace); save(`${phase}-positions.json`, positions); return metrics;
};
const calibrationMetrics = replay("calibration", config.calibrationStart, config.calibrationEnd), metrics = replay("test", config.start, config.end);
save("summary.json", { window: config.window, start: config.start, end: config.end, fullWindow: false, training: oldSummary.training,
  estimation: estimation.length, estimationMass, earlierEstimation: earlier.length, laterEstimation: later.length, calibration: calibration.length, test: test.length,
  metrics, calibrationMetrics, cashBoundDepth: bounds.verifiedDepth, elapsedSeconds: (performance.now() - started) / 1000 });
console.log(JSON.stringify({ extraDays, sampling, earlier: earlier.length, later: later.length, cashBoundDepth: bounds.verifiedDepth,
  calibrationCrps: forecast.calibration.returnCrpsBps, testCrps: forecast.test.returnCrpsBps,
  calibrationReturnPct: calibrationMetrics.returnPct, testReturnPct: metrics.returnPct, leaves: forecast.leaves,
  elapsedSeconds: (performance.now() - started) / 1000 }));
