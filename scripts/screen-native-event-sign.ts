/** Isolate a native event-direction head before paying for policy integration. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventFeatureWarmup, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventProbabilityFromReturnWeight, eventSignMass, predictEventSign, trainEventSign } from "../packages/bot-algo/src/event-sign.js";
import { NATIVE_SECOND_SELECTED_SIGN_FEATURES, usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSourceDays } from "./event-fit-periods.js";
import { loadNativeEventCandles, makeSamples } from "./research-event-policy.js";
const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const directory = (name: string) => path.resolve(__dirname, "../data/benchmarks", name), source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
const day = 86_400_000;
const partitionEnd = config.partitionEnd ?? config.fitEnd - day;
const trainDays = Number(arg("train-days") || "0");
assert.ok(Number.isSafeInteger(trainDays) && trainDays >= 0 && trainDays <= 90);
const partitionStart = trainDays ? partitionEnd - trainDays * day : config.partitionStart ?? config.fitStart;
assert.ok(partitionStart < partitionEnd && config.calibrationStart < config.calibrationEnd);
const bytes = fs.readFileSync(path.join(source, "model.json")), { model } = restoreEventPolicy(JSON.parse(bytes.toString()));
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
for (const ref of config.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
const penalty = Number(arg("penalty", "0.1")); assert.ok(penalty > 0 && Number.isFinite(penalty));
const objective = arg("objective", "ordinary"); assert.ok(["ordinary", "return-weighted"].includes(objective));
const signFeatures = arg("sign-features", "model");
assert.ok(["model", "selected-flow-context"].includes(signFeatures));
const featureNames = signFeatures === "selected-flow-context" ? NATIVE_SECOND_SELECTED_SIGN_FEATURES : model.featureNames;
const calibrationOnly = process.argv.includes("--calibration-only");
if (!calibrationOnly) assert.ok(config.start < config.end);
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("sources.json", Object.fromEntries(["scripts/screen-native-event-sign.ts", "packages/bot-algo/src/event-sign.ts",
  "scripts/event-fit-periods.ts",
  "packages/bot-algo/src/event-second-features.ts", "packages/bot-algo/src/event-distribution.ts", "scripts/research-event-policy.ts"].map(file => [file, fs.readFileSync(path.resolve(__dirname, "..", file), "utf8")])));
const start = performance.now();
const warmup = Math.max(config.warmupCandles, eventFeatureWarmup(model.clock, featureNames));
const sourceRanges = [{ start: partitionStart - (warmup + 1) * 1000, end: config.calibrationEnd }];
if (!calibrationOnly) sourceRanges.push({ start: config.start - (warmup + 1) * 1000, end: config.end });
const includeTradeFlow = usesNativeSecondTradeFlowFeatures(model.featureNames)
  || usesNativeSecondTradeFlowFeatures(featureNames);
const sourceReferences = eventSourceDays(sourceRanges).flatMap(dayStart => {
  const date = new Date(dayStart).toISOString().slice(0, 10);
  const candle = path.resolve(__dirname, `../data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s/${date}.json`);
  const files = [candle];
  if (includeTradeFlow) files.push(path.resolve(__dirname,
    `../data/market/immutable/refs/trade-flow/spot-btcusdt/btcusdt/1s/${date}.json`));
  return files.map(file => ({ file, sha256: hash(fs.readFileSync(file)) }));
});
save("config.json", { contract: "native-event-sign-screen-v2", source, modelHash: hash(bytes), penalty, objective, featureNames,
  magnitudeFeatureNames: model.featureNames, signFeatureMode: signFeatures,
  trainDays: trainDays || (partitionEnd - partitionStart) / day,
  trainStart: partitionStart, trainEnd: partitionEnd, calibrationStart: config.calibrationStart, calibrationEnd: config.calibrationEnd,
  ...(calibrationOnly ? {} : { testStart: config.start, testEnd: config.end }), blends: [0, .5, 1], scoredTestLoaded: !calibrationOnly,
  sourceRanges, sourceReferences,
  method: "Fit the existing penalized logistic active-sign head on native cost-sized event targets and the separately declared causal sign features. The saved joint magnitude/duration/extrema/successor tree and its feature basis stay frozen. Fit only on the declared contiguous interval ending before the magnitude-law estimation day. Train-only scaling. The optional return-weighted objective uses absolute target returns and converts its weighted score back to ordinary sign probability using the frozen leaf-conditional magnitudes. Compare fixed blends 0, 0.5, 1 on identical diagnostic populations; preserve zero mass and conditional positive/negative magnitudes. Calibration-only mode never loads the scored window. No policy or PnL simulation." });
const candles = loadNativeEventCandles(sourceRanges, includeTradeFlow);
type SignSample = MoveSample & { signFeatures: number[]; nextSignFeatures: number[] };
const sample = (start: number, end: number, sampling: "chain" | "stride", excluded = config.excluded): SignSample[] => {
  const step = sampling === "chain" ? 1 : config.stride;
  const magnitude = makeSamples(candles, model.clock, start, end, excluded, step, sampling, model.featureNames);
  if (featureNames === model.featureNames) return magnitude.map(row => ({ ...row,
    signFeatures: row.features, nextSignFeatures: row.nextFeatures }));
  const sign = makeSamples(candles, model.clock, start, end, excluded, step, sampling, featureNames);
  assert.equal(sign.length, magnitude.length, "Sign and magnitude samples must align");
  return magnitude.map((row, index) => {
    const other = sign[index]!;
    assert.deepEqual([other.start, other.end, other.return, other.duration, other.low, other.high],
      [row.start, row.end, row.return, row.duration, row.low, row.high], "Sign and magnitude event targets must be identical");
    return { ...row, signFeatures: other.features, nextSignFeatures: other.nextFeatures };
  });
};
const training = sample(partitionStart, partitionEnd, "stride");
const calibration = sample(config.calibrationStart, config.calibrationEnd, "stride");
const test = calibrationOnly ? undefined : sample(config.start, config.end, "chain", []);
const signTraining = training.map(row => ({ features: row.signFeatures, return: row.return }));
const head = trainEventSign(signTraining, penalty, objective === "return-weighted" ? training.map(s => Math.abs(s.return)) : undefined); save("head.json", head);
const laws = model.kernels.map(kernel => {
  const mass = eventSignMass(kernel), positive = kernel.reduce((s, a) => s + (a.return > 0 ? a.probability * a.return : 0), 0);
  const negative = kernel.reduce((s, a) => s + (a.return < 0 ? a.probability * a.return : 0), 0);
  return { ...mass, positiveMean: positive / (mass.positive || 1), negativeMean: negative / (mass.negative || 1) };
});
const score = (rows: SignSample[], blend: number) => {
  let ce = 0, correct = 0, active = 0, weightedCorrect = 0, magnitude = 0, mse = 0, zeroMse = 0, mean = 0;
  let maxAbsPredictedMeanBps = 0, aboveOneWayCost = 0, aboveRoundTripCost = 0;
  const oneWayCostBps = config.costs.feeBps + config.costs.slippageBps;
  for (const row of rows) {
    const law = laws[eventLeaf(model, row.features)], raw = predictEventSign(head, row.signFeatures);
    const predicted = objective === "return-weighted" && law.positive && law.negative
      ? eventProbabilityFromReturnWeight(raw, law.positiveMean, law.negativeMean) : raw;
    const p = law.positive && law.negative ? (1 - blend) * law.probability + blend * predicted : law.probability;
    const expected = (law.positive + law.negative) * (p * law.positiveMean + (1 - p) * law.negativeMean);
    const absPredictedMeanBps = Math.abs(expected) * 10_000;
    maxAbsPredictedMeanBps = Math.max(maxAbsPredictedMeanBps, absPredictedMeanBps);
    aboveOneWayCost += Number(absPredictedMeanBps > oneWayCostBps);
    aboveRoundTripCost += Number(absPredictedMeanBps > 2 * oneWayCostBps);
    mse += (row.return - expected) ** 2; zeroMse += row.return ** 2; mean += expected;
    if (row.return) {
      const y = Number(row.return > 0), isCorrect = (p > .5 ? 1 : 0) === y;
      ce -= y * Math.log(Math.max(1e-12, p)) + (1 - y) * Math.log(Math.max(1e-12, 1 - p));
      correct += Number(isCorrect); weightedCorrect += Number(isCorrect) * Math.abs(row.return);
      active++; magnitude += Math.abs(row.return);
    }
  }
  return { samples: rows.length, active, signLogLoss: ce / active, directionAccuracy: correct / active,
    magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude, mseSkill: 1 - mse / zeroMse, predictedMeanBps: mean / rows.length * 10000,
    maxAbsPredictedMeanBps, aboveOneWayCost, aboveRoundTripCost };
};
const predictionRows = (rows: SignSample[]) => rows.map(row => {
  const leaf = eventLeaf(model, row.features), law = laws[leaf], raw = predictEventSign(head, row.signFeatures);
  const predicted = objective === "return-weighted" && law.positive && law.negative
    ? eventProbabilityFromReturnWeight(raw, law.positiveMean, law.negativeMean) : raw;
  const blends = [0, .5, 1].map(blend => {
    const probability = law.positive && law.negative ? (1 - blend) * law.probability + blend * predicted : law.probability;
    return { blend, probability, expectedReturnBps: (law.positive + law.negative)
      * (probability * law.positiveMean + (1 - probability) * law.negativeMean) * 10_000 };
  });
  return { start: row.start, end: row.end, leaf, realizedReturnBps: row.return * 10_000,
    baseProbability: law.probability, activeMass: law.positive + law.negative,
    positiveMeanBps: law.positiveMean * 10_000, negativeMeanBps: law.negativeMean * 10_000,
    rawHeadProbability: raw, adjustedHeadProbability: predicted, blends };
});
const results = [0, .5, 1].map(blend => ({ blend, training: score(training, blend), calibration: score(calibration, blend),
  ...(test ? { test: score(test, blend) } : {}) }));
save("calibration-predictions.json", predictionRows(calibration));
if (test) save("test-predictions.json", predictionRows(test).map((row, index) => ({ ...row,
  time: candles[test[index].start].openTime + 1000, endTime: candles[test[index].end].openTime + 1000 })));
save("summary.json", { training: training.length, calibration: calibration.length, ...(test ? { test: test.length } : {}), results,
  constantHalfProbabilitySignLogLoss: Math.log(2),
  elapsedSeconds: (performance.now() - start) / 1000 });
console.log(JSON.stringify({ results, elapsedSeconds: (performance.now() - start) / 1000 }));
