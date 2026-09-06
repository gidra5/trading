/** Enrich an unchanged empirical event mixture with its source paths' account transitions. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { EVENT_FEATURES, eventFeatures, eventLeaf, reestimateEventTreeWithSources } from "../packages/bot-algo/src/event-distribution.js";
import { eventAverageUniqueness } from "../packages/bot-algo/src/event-sampling.js";
import { summarizeEventExecutionPath, evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
import { eventProbabilityFromReturnWeight, eventSignMass, predictEventSign } from "../packages/bot-algo/src/event-sign.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSourceDays, mergeEventSourceRanges } from "./event-fit-periods.js";
import { loadEventCandles, loadNativeEventCandles, makeSamples } from "./research-event-policy.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
const signSource = arg("sign-source") ? directory(arg("sign-source")) : undefined;
const directionEnsembleSource = arg("direction-ensemble-source") ? directory(arg("direction-ensemble-source")) : undefined;
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
assert.ok(!signSource || !directionEnsembleSource, "Use either a sign source or a direction ensemble");
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const config = read(path.join(source, "config.json")), policy = read(path.join(source, "model.json"));
assert.ok(["native-second-event-screen-v1", "native-second-event-screen-v2"].includes(config.contract));
assert.equal(config.clock.candleIntervalMs, 1000);
const weighting = config.weighting ?? "unit";
const sampling = config.sampling ?? (config.estimationMode === "later-fit-chain" ? "chain" : "stride");
const prior = config.prior ?? 32;
assert.ok(["unit", "uniqueness"].includes(weighting) && ["stride", "chain"].includes(sampling)
  && Number.isFinite(prior) && prior >= 0);
const directionEnsembleConfig = directionEnsembleSource ? read(path.join(directionEnsembleSource, "config.json")) : undefined;
const directionEnsemble = directionEnsembleSource ? read(path.join(directionEnsembleSource, "summary.json")) : undefined;
const selectedDirection = directionEnsemble?.selected?.specification;
const ensembleFastSource = directionEnsembleConfig?.fastSource as string | undefined;
const ensembleSlowSource = directionEnsembleConfig?.slowSource as string | undefined;
const signDirectory = signSource ?? ensembleFastSource;
const signConfig = signDirectory ? read(path.join(signDirectory, "config.json")) : undefined;
const signHead = signDirectory ? read(path.join(signDirectory, "head.json")) : undefined;
const signModel = signConfig ? restoreEventPolicy(read(path.join(signConfig.source, "model.json"))).model : undefined;
const signFeatureNames = signConfig?.featureNames as readonly string[] | undefined;
const slowConfig = ensembleSlowSource ? read(path.join(ensembleSlowSource, "config.json")) : undefined;
const slowModel = ensembleSlowSource ? read(path.join(ensembleSlowSource, "slow-model.json")) : undefined;
if (signConfig) {
  assert.equal(signConfig.contract, "native-event-sign-screen-v2"); assert.equal(signConfig.objective, "return-weighted");
  assert.equal(signConfig.penalty, .01); assert.deepEqual(signConfig.blends, [0, .5, 1]);
}
if (directionEnsembleConfig) {
  assert.equal(directionEnsembleConfig.contract, "native-event-direction-ensemble-screen-v1");
  assert.equal(path.resolve(directionEnsembleConfig.source), source);
  assert.equal(path.resolve(ensembleFastSource!), signDirectory);
  assert.equal(slowConfig.contract, "native-event-slow-direction-screen-v1");
  assert.equal(path.resolve(slowConfig.source), source);
  for (const reference of slowConfig.slowSourceReferences ?? []) assert.equal(hash(reference.file), reference.sha256);
  assert.ok(Number.isFinite(selectedDirection?.slowCoefficient) && selectedDirection.slowCoefficient >= 0
    && Number.isFinite(selectedDirection?.fastCoefficient) && selectedDirection.fastCoefficient >= 0);
}
const includeTradeFlow = usesNativeSecondTradeFlowFeatures(policy.model.featureNames)
  || Boolean(signModel && usesNativeSecondTradeFlowFeatures(signModel.featureNames))
  || Boolean(signFeatureNames && usesNativeSecondTradeFlowFeatures(signFeatureNames));
for (const ref of config.sourceReferences) assert.equal(hash(ref.file), ref.sha256);
const history = (config.warmupCandles + 1) * 1000;
const periods = Number.isFinite(config.finalEstimationStart)
  ? [...(config.extraDays ? [{ start: config.earlierEstimationStart, end: config.earlierEstimationEnd }] : []),
    { start: config.finalEstimationStart, end: config.fitEnd }]
  : [{ start: config.estimationMode && config.estimationMode !== "shared"
    ? (config.fitPeriods?.fitEnd ?? config.fitEnd) - 86_400_000
    : config.fitPeriods?.fitStart ?? config.fitStart,
  end: config.fitPeriods?.fitEnd ?? config.fitEnd }];
assert.ok(periods.every(p => p.start < p.end && p.end <= config.calibrationStart));
const ranges = mergeEventSourceRanges(periods.map(p => ({ start: p.start - history, end: p.end })));
const references = eventSourceDays(ranges).flatMap(day => {
  const file = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s", `${new Date(day).toISOString().slice(0, 10)}.json`);
  const candle = { file, sha256: hash(file) };
  if (!includeTradeFlow) return [candle];
  const flowFile = path.join(root, "data/market/immutable/refs/trade-flow/spot-btcusdt/btcusdt/1s", `${new Date(day).toISOString().slice(0, 10)}.json`);
  return [candle, { file: flowFile, sha256: hash(flowFile) }];
});
fs.mkdirSync(output, { recursive: true });
const save = (file: string, data: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(data,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-event-execution-law-v1", source, modelHash: hash(path.join(source, "model.json")),
  periods, ranges, sourceReferences: references, costs: policy.costs, clock: policy.model.clock,
  ...(signSource ? { signSource, signConfigHash: hash(path.join(signSource, "config.json")),
    signHeadHash: hash(path.join(signSource, "head.json")), signBlend: .5 } : {}),
  ...(directionEnsembleSource ? { directionEnsembleSource,
    directionEnsembleConfigHash: hash(path.join(directionEnsembleSource, "config.json")),
    directionEnsembleSummaryHash: hash(path.join(directionEnsembleSource, "summary.json")),
    directionFastConfigHash: hash(path.join(ensembleFastSource!, "config.json")),
    directionFastHeadHash: hash(path.join(ensembleFastSource!, "head.json")),
    directionSlowConfigHash: hash(path.join(ensembleSlowSource!, "config.json")),
    directionSlowModelHash: hash(path.join(ensembleSlowSource!, "slow-model.json")),
    selectedDirection } : {}),
  method: "Reconstruct the saved fitting population, weights and exact source index of each observed/prior atom. Preserve all old return/duration/extrema/successor probabilities bit for bit, then attach cost-specific controlled path summaries from those same estimation rows. No calibration/test labels are used for fitting. Calibration account probes compare a predeclared small set of root requests under this enriched law; they are not global H1/H2 certificates or realized-profit tests." });
save("sources.json", Object.fromEntries(["scripts/compile-native-event-execution-law.ts", "packages/bot-algo/src/event-distribution.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-sampling.ts", "scripts/research-event-policy.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), candles = loadNativeEventCandles(ranges,
  includeTradeFlow);
const samples = periods.flatMap(p => makeSamples(candles, policy.model.clock, p.start, p.end, config.excluded,
  sampling === "chain" ? 1 : config.stride, sampling, policy.model.featureNames));
const weights = weighting === "uniqueness" ? eventAverageUniqueness(samples) : undefined;
const estimated = reestimateEventTreeWithSources(policy.model, samples, prior, weights);
assert.deepEqual(estimated.model, policy.model, "Enrichment must preserve the entire saved fitted distribution");
if (weights) assert.deepEqual(samples.map((s, i) => ({ start: candles[s.start].openTime + 1000,
  end: candles[s.end].openTime + 1000, weight: weights[i] })), read(path.join(source, "weights.json")));
const signStats = (kernel: readonly { probability: number; return: number }[]) => {
  const mass = eventSignMass(kernel), positiveTotal = kernel.reduce((sum, atom) =>
    sum + (atom.return > 0 ? atom.probability * atom.return : 0), 0);
  const negativeTotal = kernel.reduce((sum, atom) =>
    sum + (atom.return < 0 ? atom.probability * atom.return : 0), 0);
  return { ...mass, positiveMean: positiveTotal / (mass.positive || 1), negativeMean: negativeTotal / (mass.negative || 1) };
};
const bounded = (probability: number) => Math.max(1e-6, Math.min(1 - 1e-6, probability));
const logit = (probability: number) => Math.log(bounded(probability) / (1 - bounded(probability)));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
const MINUTE = 60_000;
const minuteCandles = slowConfig ? loadEventCandles(slowConfig.slowTrainStart - 1441 * MINUTE,
  Math.max(...periods.map(period => period.end))) : undefined;
const minuteByClose = minuteCandles ? new Map(minuteCandles.map((row, index) => [row.openTime + MINUTE, index])) : undefined;
const slowLaws = slowModel?.kernels.map((kernel: any[]) => eventSignMass(kernel));
const signForecastCache = new Map<number, any>();
const sameFeatures = (left: readonly string[], right: readonly string[]) =>
  left.length === right.length && left.every((name, feature) => name === right[feature]);
const signForecast = (index: number, savedFeatures?: readonly number[]) => {
  if (!signModel) return undefined;
  const saved = signForecastCache.get(index);
  if (saved) return saved;
  const magnitudeFeatures = savedFeatures && sameFeatures(signModel.featureNames, policy.model.featureNames)
    ? savedFeatures : eventFeatures(candles, index, signModel.featureNames, signModel.clock);
  const features = savedFeatures && sameFeatures(signFeatureNames ?? signModel.featureNames, policy.model.featureNames)
    ? savedFeatures : eventFeatures(candles, index, signFeatureNames ?? signModel.featureNames, signModel.clock);
  const leaf = eventLeaf(signModel, magnitudeFeatures), stats = signStats(signModel.kernels[leaf]);
  const rawReturnWeight = predictEventSign(signHead, features);
  const adjustedProbability = eventProbabilityFromReturnWeight(rawReturnWeight, stats.positiveMean, stats.negativeMean);
  if (!slowModel) {
    const forecast = { leaf, rawReturnWeight, adjustedProbability, baseProbability: stats.probability,
      directionProbability: .5 * stats.probability + .5 * adjustedProbability };
    signForecastCache.set(index, forecast); return forecast;
  }
  const decisionTime = candles[index].openTime + 1000;
  const minute = minuteByClose!.get(Math.floor(decisionTime / MINUTE) * MINUTE);
  assert.notEqual(minute, undefined, "Slow direction history is unavailable at an event decision");
  const slowLeaf = eventLeaf(slowModel, eventFeatures(minuteCandles!, minute!, EVENT_FEATURES, slowModel.clock));
  const slowProbability = slowLaws![slowLeaf].probability, fastProbability = adjustedProbability;
  const directionProbability = bounded(sigmoid(selectedDirection.slowCoefficient * logit(slowProbability)
    + selectedDirection.fastCoefficient * logit(fastProbability)));
  const forecast = { leaf, slowLeaf, slowProbability, fastProbability, rawReturnWeight, adjustedProbability,
    baseProbability: stats.probability, directionProbability };
  signForecastCache.set(index, forecast); return forecast;
};
const paths = samples.map(s => ({ ...summarizeEventExecutionPath(candles, s.start, s.end, policy.costs),
  ...(signModel ? { sign: signForecast(s.start, s.features), nextSign: signForecast(s.end, s.nextFeatures) } : {}) }));
const kernels = estimated.sources.map((rows, leaf) => rows.map((index, j) => ({ probability: policy.model.kernels[leaf][j].probability,
  next: policy.model.kernels[leaf][j].next, path: index })));
const projected = kernels.map(kernel => kernel.map(a => ({ probability: a.probability, next: a.next,
  return: paths[a.path].closeRatio - 1, duration: paths[a.path].seconds / 60,
  low: Math.min(1, paths[a.path].lowRatio) - 1, high: Math.max(1, paths[a.path].highRatio) - 1 })));
assert.deepEqual(projected, policy.model.kernels);
save("law.json", { version: 1, modelHash: hash(path.join(source, "model.json")), costs: policy.costs,
  paths, kernels, origins: samples.map(s => ({ time: candles[s.start].openTime + 1000, endTime: candles[s.end].openTime + 1000 })) });
const compilationSeconds = (performance.now() - started) / 1000;
const calibrationFile = path.join(source, "calibration-trades.json"), calibration = fs.existsSync(calibrationFile) ? read(calibrationFile) : [];
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
  calibrationTraceHash: fs.existsSync(calibrationFile) ? hash(calibrationFile) : null, compilationSeconds,
  elapsedSeconds: (performance.now() - started) / 1000,
  lawHash: hash(path.join(output, "law.json")), probes: results });
console.log(JSON.stringify({ window: config.window.id, samples: samples.length, compilationSeconds,
  probes: results.map(r => ({ time: r.time, leaf: r.leaf, candidateImprovementBps: r.candidateImprovementBps,
    originalQuantity: r.original.quantity, bestQuantity: r.best.quantity, originalRejectedMass: r.original.rejectedMass })) }));
