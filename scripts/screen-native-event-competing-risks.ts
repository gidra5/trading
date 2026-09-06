/** Calibration-only factorization of a native event into barrier arrival and side. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { eventFeatureWarmup, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { predictEventSign, trainEventSign, type EventSignHead } from "../packages/bot-algo/src/event-sign.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES } from "../packages/bot-algo/src/event-second-features.js";
import { eventSourceDays } from "./event-fit-periods.js";
import { loadNativeEventCandles, makeSamples } from "./research-event-policy.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json")), modelFile = path.join(source, "model.json");
assert.equal(config.contract, "native-second-event-screen-v2");
assert.equal(config.scoredTestLoaded, false, "Requires a calibration-only source");
const { model } = restoreEventPolicy(read(modelFile));
assert.equal(model.clock.candleIntervalMs, 1000);
assert.ok(!model.clock.runClock && !model.clock.reversalClock && model.clock.thresholdBps > 0);
for (const ref of config.sourceReferences) assert.equal(hash(ref.file), ref.sha256);

const DAY = 86_400_000, partitionEnd = config.partitionEnd ?? config.fitEnd - DAY;
const partitionStart = config.partitionStart ?? config.fitStart;
assert.ok(partitionStart < partitionEnd && config.calibrationEnd - config.calibrationStart >= 2 * DAY);
const calibrationSplit = config.calibrationStart + Math.floor((config.calibrationEnd - config.calibrationStart) / (2 * DAY)) * DAY;
assert.ok(config.calibrationStart < calibrationSplit && calibrationSplit < config.calibrationEnd);
const penalties = [0.01, 0.1, 1], hitBlends = [0.25, 0.5, 0.75, 1], sideBlends = [0, 0.25, 0.5, 0.75, 1];
const featureMode = arg("features") || "model";
assert.ok(["model", "volatility-context"].includes(featureMode));
const featureNames = featureMode === "volatility-context" ? NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES : model.featureNames;
const warmup = Math.max(config.warmupCandles, eventFeatureWarmup(model.clock, featureNames));
const sourceRanges = [{ start: partitionStart - (warmup + 1) * 1000, end: config.calibrationEnd }];
const includeTradeFlow = usesNativeSecondTradeFlowFeatures(model.featureNames) || usesNativeSecondTradeFlowFeatures(featureNames);
const sourceReferences = eventSourceDays(sourceRanges).flatMap(dayStart => {
  const date = new Date(dayStart).toISOString().slice(0, 10);
  const files = [path.join(root, `data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s/${date}.json`)];
  if (includeTradeFlow) files.push(path.join(root, `data/market/immutable/refs/trade-flow/spot-btcusdt/btcusdt/1s/${date}.json`));
  return files.map(file => ({ file, sha256: hash(file) }));
});
const candles = loadNativeEventCandles(sourceRanges, includeTradeFlow);
const sample = (start: number, end: number) => {
  const magnitude = makeSamples(candles, model.clock, start, end, config.excluded, config.stride, "stride", model.featureNames);
  if (featureNames === model.featureNames) return magnitude;
  const features = makeSamples(candles, model.clock, start, end, config.excluded, config.stride, "stride", featureNames);
  assert.equal(features.length, magnitude.length);
  return magnitude.map((row, index) => {
    const other = features[index]!;
    assert.deepEqual([other.start, other.end, other.return, other.duration], [row.start, row.end, row.return, row.duration]);
    return { ...row, features: other.features, nextFeatures: other.nextFeatures, magnitudeFeatures: row.features };
  });
};
const training = sample(partitionStart, partitionEnd), calibration = sample(config.calibrationStart, config.calibrationEnd);
const selection = calibration.filter(row => candles[row.start].openTime + 1000 < calibrationSplit);
const holdout = calibration.filter(row => candles[row.start].openTime + 1000 >= calibrationSplit);
assert.equal(selection.length + holdout.length, calibration.length);

type Group = "down" | "timeout" | "up";
const group = (value: number): Group => value * 10_000 <= -model.clock.thresholdBps ? "down"
  : value * 10_000 >= model.clock.thresholdBps ? "up" : "timeout";
const syntheticHit = (row: MoveSample) => ({ features: row.features, return: group(row.return) === "timeout" ? -1 : 1 });
const syntheticSide = (row: MoveSample) => ({ features: row.features,
  return: group(row.return) === "timeout" ? 0 : row.return });
const hitHeads = new Map<number, EventSignHead>(), sideHeads = new Map<number, EventSignHead>();
for (const penalty of penalties) {
  hitHeads.set(penalty, trainEventSign(training.map(syntheticHit), penalty));
  sideHeads.set(penalty, trainEventSign(training.map(syntheticSide), penalty));
}

const groups = ["down", "timeout", "up"] as const;
const laws = model.kernels.map(kernel => {
  const mass = { down: 0, timeout: 0, up: 0 }, weighted = { down: 0, timeout: 0, up: 0 };
  for (const atom of kernel) { const g = group(atom.return); mass[g] += atom.probability; weighted[g] += atom.probability * atom.return; }
  const mean = Object.fromEntries(groups.map(g => [g, mass[g] ? weighted[g] / mass[g] : 0])) as Record<Group, number>;
  return { mass, mean };
});
assert.ok(laws.every(law => groups.every(g => law.mass[g] > 0)), "Every leaf must retain all competing-risk supports");

interface Candidate { hitPenalty: number; sidePenalty: number; hitBlend: number; sideBlend: number; }
interface Score {
  samples: number; groupNll: number; groupBrier: number; returnMse: number; zeroReturnMse: number;
  returnMseSkill: number; directionAccuracy: number; magnitudeWeightedDirectionAccuracy: number;
  predictedMeanBps: number; maxAbsPredictedMeanBps: number; aboveOneWayCost: number; aboveRoundTripCost: number;
  actualGroups: Record<Group, number>; predictedGroups: Record<Group, number>;
}
const probabilities = (row: MoveSample, candidate?: Candidate) => {
  const law = laws[eventLeaf(model, (row as MoveSample & { magnitudeFeatures?: number[] }).magnitudeFeatures ?? row.features)], base = law.mass;
  if (!candidate) return { law, probabilities: { ...base } };
  const baseHit = base.down + base.up, baseUpGivenHit = base.up / baseHit;
  const headHit = predictEventSign(hitHeads.get(candidate.hitPenalty)!, row.features);
  const headUpGivenHit = predictEventSign(sideHeads.get(candidate.sidePenalty)!, row.features);
  const pHit = (1 - candidate.hitBlend) * baseHit + candidate.hitBlend * headHit;
  const pUpGivenHit = (1 - candidate.sideBlend) * baseUpGivenHit + candidate.sideBlend * headUpGivenHit;
  const factor = { down: pHit * (1 - pUpGivenHit), timeout: 1 - pHit, up: pHit * pUpGivenHit };
  return { law, probabilities: factor, pHit, pUpGivenHit };
};
const score = (rows: MoveSample[], candidate?: Candidate): Score => {
  let nll = 0, brier = 0, mse = 0, zeroMse = 0, correct = 0, weightedCorrect = 0, magnitude = 0, mean = 0;
  let maxAbsPredictedMeanBps = 0, aboveOneWayCost = 0, aboveRoundTripCost = 0;
  const actualGroups = { down: 0, timeout: 0, up: 0 }, predictedGroups = { down: 0, timeout: 0, up: 0 };
  const oneWayCost = config.costs.feeBps + config.costs.slippageBps;
  for (const row of rows) {
    const actual = group(row.return), forecast = probabilities(row, candidate), p = forecast.probabilities;
    const expected = groups.reduce((sum, g) => sum + p[g] * forecast.law.mean[g], 0);
    nll -= Math.log(Math.max(1e-12, p[actual]));
    brier += groups.reduce((sum, g) => sum + (p[g] - Number(g === actual)) ** 2, 0);
    mse += (row.return - expected) ** 2; zeroMse += row.return ** 2; mean += expected;
    const isCorrect = Math.sign(expected) === Math.sign(row.return);
    correct += Number(isCorrect); weightedCorrect += Number(isCorrect) * Math.abs(row.return); magnitude += Math.abs(row.return);
    const abs = Math.abs(expected) * 10_000; maxAbsPredictedMeanBps = Math.max(maxAbsPredictedMeanBps, abs);
    aboveOneWayCost += Number(abs > oneWayCost); aboveRoundTripCost += Number(abs > 2 * oneWayCost);
    actualGroups[actual]++; for (const g of groups) predictedGroups[g] += p[g];
  }
  for (const g of groups) predictedGroups[g] /= rows.length;
  return { samples: rows.length, groupNll: nll / rows.length, groupBrier: brier / rows.length,
    returnMse: mse / rows.length, zeroReturnMse: zeroMse / rows.length, returnMseSkill: 1 - mse / zeroMse,
    directionAccuracy: correct / rows.length, magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude,
    predictedMeanBps: mean / rows.length * 10_000, maxAbsPredictedMeanBps, aboveOneWayCost, aboveRoundTripCost,
    actualGroups, predictedGroups };
};
const byUtcDay = (rows: MoveSample[]) => Map.groupBy(rows, row => Math.floor((candles[row.start].openTime + 1000) / DAY));
const daily = (rows: MoveSample[], candidate?: Candidate) => [...byUtcDay(rows).entries()].map(([day, values]) => ({ day, ...score(values, candidate) }));
const baseline = { training: score(training), selection: score(selection), holdout: score(holdout),
  selectionDays: daily(selection), holdoutDays: daily(holdout) };
const candidates = penalties.flatMap(hitPenalty => penalties.flatMap(sidePenalty => hitBlends.flatMap(hitBlend => sideBlends.map(sideBlend => {
  const specification = { hitPenalty, sidePenalty, hitBlend, sideBlend };
  const selectionDays = daily(selection, specification), holdoutDays = daily(holdout, specification);
  return { specification, training: score(training, specification), selection: score(selection, specification),
    holdout: score(holdout, specification), selectionDays, holdoutDays,
    selectionPositiveMseDays: selectionDays.filter((value, day) => value.returnMse < baseline.selectionDays[day]!.returnMse).length,
    selectionPositiveNllDays: selectionDays.filter((value, day) => value.groupNll < baseline.selectionDays[day]!.groupNll).length,
    holdoutPositiveMseDays: holdoutDays.filter((value, day) => value.returnMse < baseline.holdoutDays[day]!.returnMse).length,
    holdoutPositiveNllDays: holdoutDays.filter((value, day) => value.groupNll < baseline.holdoutDays[day]!.groupNll).length };
}))));
const eligible = candidates.filter(row => row.selection.groupNll < baseline.selection.groupNll
  && row.selection.returnMse < baseline.selection.returnMse
  && row.selectionPositiveMseDays / row.selectionDays.length >= .6
  && row.selectionPositiveNllDays / row.selectionDays.length >= .6);
const selected = [...eligible].sort((a, b) => a.selection.returnMse - b.selection.returnMse
  || a.selection.groupNll - b.selection.groupNll)[0];
const promote = Boolean(selected && selected.holdout.returnMse < baseline.holdout.returnMse
  && selected.holdout.groupNll < baseline.holdout.groupNll
  && selected.holdoutPositiveMseDays / selected.holdoutDays.length >= .6
  && selected.holdoutPositiveNllDays / selected.holdoutDays.length >= .6
  && selected.holdout.aboveRoundTripCost > 0);

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-competing-risks-screen-v1", source, modelHash: hash(modelFile),
  featureMode, featureNames, magnitudeFeatureNames: model.featureNames, thresholdBps: model.clock.thresholdBps,
  penalties, hitBlends, sideBlends,
  partitionStart, partitionEnd, calibrationStart: config.calibrationStart, calibrationSplit, calibrationEnd: config.calibrationEnd,
  scoredTestLoaded: false, sourceRanges, sourceReferences,
  method: "Factor the frozen native event into P(barrier before timeout) and P(up barrier | barrier). Fit separate penalized logistic heads on the pre-estimation partition interval, and blend each head with its corresponding frozen-leaf probability using independently selected weights. Preserve the conditional path law inside down-hit, timeout and up-hit groups. Select on the first half of calibration only when both pooled group NLL and return MSE improve and each improves on at least 60% of UTC-day blocks; report the untouched second half without reselection. Promotion requires the same holdout gates plus at least one forecast above round-trip cost. No test window or policy replay is loaded.",
  caveat: "This changes only group masses. Conditional return, extrema, duration and successor laws remain the frozen empirical leaf laws. Overlapping fixed-stride labels reduce effective sample size; an apparent holdout gain still needs uncertainty and economic gates before integration." });
save("heads.json", { hit: Object.fromEntries(hitHeads), side: Object.fromEntries(sideHeads) });
save("sources.json", Object.fromEntries(["scripts/screen-native-event-competing-risks.ts", "packages/bot-algo/src/event-sign.ts",
  "packages/bot-algo/src/event-distribution.ts", "scripts/research-event-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
save("summary.json", { training: training.length, selection: selection.length, holdout: holdout.length,
  baseline, eligibleCandidates: eligible.length, selected, promote, candidates });
console.log(JSON.stringify({ training: training.length, selection: selection.length, holdout: holdout.length,
  baseline: { selection: baseline.selection, holdout: baseline.holdout }, eligibleCandidates: eligible.length, selected, promote }, null, 2));
