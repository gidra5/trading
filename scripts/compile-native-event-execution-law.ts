/** Enrich an unchanged empirical event mixture with its source paths' account transitions. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { reestimateEventTreeWithSources } from "../packages/bot-algo/src/event-distribution.js";
import { eventAverageUniqueness } from "../packages/bot-algo/src/event-sampling.js";
import { summarizeEventExecutionPath, evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
import { eventSourceDays, mergeEventSourceRanges } from "./event-fit-periods.js";
import { loadNativeEventCandles, makeSamples } from "./research-event-policy.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const config = read(path.join(source, "config.json")), policy = read(path.join(source, "model.json"));
assert.equal(config.contract, "native-second-event-screen-v1"); assert.equal(config.clock.candleIntervalMs, 1000);
assert.ok(["unit", "uniqueness"].includes(config.weighting) && ["stride", "chain"].includes(config.sampling));
for (const ref of config.sourceReferences) assert.equal(hash(ref.file), ref.sha256);
const history = (config.warmupCandles + 1) * 1000;
const periods = [...(config.extraDays ? [{ start: config.earlierEstimationStart, end: config.earlierEstimationEnd }] : []),
  { start: config.finalEstimationStart, end: config.fitEnd }];
assert.ok(periods.every(p => p.start < p.end && p.end <= config.calibrationStart));
const ranges = mergeEventSourceRanges(periods.map(p => ({ start: p.start - history, end: p.end })));
const references = eventSourceDays(ranges).map(day => {
  const file = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s", `${new Date(day).toISOString().slice(0, 10)}.json`);
  return { file, sha256: hash(file) };
});
fs.mkdirSync(output, { recursive: true });
const save = (file: string, data: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(data,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-event-execution-law-v1", source, modelHash: hash(path.join(source, "model.json")),
  periods, ranges, sourceReferences: references, costs: policy.costs, clock: policy.model.clock,
  method: "Reconstruct the saved fitting population, weights and exact source index of each observed/prior atom. Preserve all old return/duration/extrema/successor probabilities bit for bit, then attach cost-specific controlled path summaries from those same estimation rows. No calibration/test labels are used for fitting. Calibration account probes compare a predeclared small set of root requests under this enriched law; they are not global H1/H2 certificates or realized-profit tests." });
save("sources.json", Object.fromEntries(["scripts/compile-native-event-execution-law.ts", "packages/bot-algo/src/event-distribution.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-sampling.ts", "scripts/research-event-policy.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), candles = loadNativeEventCandles(ranges);
const samples = periods.flatMap(p => makeSamples(candles, policy.model.clock, p.start, p.end, config.excluded,
  config.sampling === "chain" ? 1 : config.stride, config.sampling, policy.model.featureNames));
const weights = config.weighting === "uniqueness" ? eventAverageUniqueness(samples) : undefined;
const estimated = reestimateEventTreeWithSources(policy.model, samples, config.prior, weights);
assert.deepEqual(estimated.model, policy.model, "Enrichment must preserve the entire saved fitted distribution");
if (weights) assert.deepEqual(samples.map((s, i) => ({ start: candles[s.start].openTime + 1000,
  end: candles[s.end].openTime + 1000, weight: weights[i] })), read(path.join(source, "weights.json")));
const paths = samples.map(s => summarizeEventExecutionPath(candles, s.start, s.end, policy.costs));
const kernels = estimated.sources.map((rows, leaf) => rows.map((index, j) => ({ probability: policy.model.kernels[leaf][j].probability,
  next: policy.model.kernels[leaf][j].next, path: index })));
const projected = kernels.map(kernel => kernel.map(a => ({ probability: a.probability, next: a.next,
  return: paths[a.path].closeRatio - 1, duration: paths[a.path].seconds / 60,
  low: Math.min(1, paths[a.path].lowRatio) - 1, high: Math.max(1, paths[a.path].highRatio) - 1 })));
assert.deepEqual(projected, policy.model.kernels);
save("law.json", { version: 1, modelHash: hash(path.join(source, "model.json")), costs: policy.costs,
  paths, kernels, origins: samples.map(s => ({ time: candles[s.start].openTime + 1000, endTime: candles[s.end].openTime + 1000 })) });
const compilationSeconds = (performance.now() - started) / 1000;
const calibrationFile = path.join(source, "calibration-trades.json"), calibration = read(calibrationFile);
const seen = new Set<number>(), probes: any[] = [];
for (const row of calibration) if (!seen.has(row.leaf)) { seen.add(row.leaf); probes.push(row); }
const held = calibration.find((row: any) => row.exposureBefore !== 0); if (held && !probes.includes(held)) probes.push(held);
const step = policy.costs.quantityStep, results = [];
for (const row of probes) {
  const account = { equity: row.equityBefore, price: row.order.price, exposure: row.exposureBefore };
  const oldLots = Math.round(row.order.quantity / step);
  const candidates = [...new Set([0, oldLots, Math.trunc(oldLots / 2), ...[-5, -2, -1, 1, 2, 5].map(d => oldLots + d)])];
  const score = (lots: number) => {
    let value = 0, rejectedMass = 0, ruinMass = 0;
    for (const atom of kernels[row.leaf]) {
      const result = evaluateEventExecutionPath(paths[atom.path], account, lots * step);
      value += atom.probability * result.logGrowth;
      if (result.canceled) rejectedMass += atom.probability;
      if (result.liquidated) ruinMass += atom.probability;
    }
    return { quantity: lots * step, value, rejectedMass, ruinMass };
  };
  const evaluated = candidates.map(score), original = evaluated.find(r => Math.abs(r.quantity - row.order.quantity) < 1e-12)!;
  const best = evaluated.reduce((best, candidate) => candidate.value > best.value ? candidate : best, original);
  results.push({ time: row.time, leaf: row.leaf, account, oldDecisionValue: row.order.value, original, best,
    candidateImprovementBps: Number.isFinite(original.value) ? (best.value - original.value) * 10000 : null, evaluated });
}
for (const ref of references) assert.equal(hash(ref.file), ref.sha256);
save("summary.json", { window: config.window, samples: samples.length, paths: paths.length,
  atoms: kernels.reduce((s, k) => s + k.length, 0), exactSavedModel: true, exactWeights: true, exactOldJointProjection: true,
  calibrationTraceHash: hash(calibrationFile), compilationSeconds, elapsedSeconds: (performance.now() - started) / 1000,
  lawHash: hash(path.join(output, "law.json")), probes: results });
console.log(JSON.stringify({ window: config.window.id, samples: samples.length, compilationSeconds,
  probes: results.map(r => ({ time: r.time, leaf: r.leaf, candidateImprovementBps: r.candidateImprovementBps,
    originalQuantity: r.original.quantity, bestQuantity: r.best.quantity, originalRejectedMass: r.original.rejectedMass })) }));
