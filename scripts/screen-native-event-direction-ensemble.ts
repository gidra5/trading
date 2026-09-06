/** Calibration-only log-odds ensemble of native fast and slower minute-event direction experts. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { EVENT_FEATURES, eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventProbabilityFromReturnWeight, eventSignMass, predictEventSign } from "../packages/bot-algo/src/event-sign.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { loadEventCandles, loadNativeEventCandles, makeSamples } from "./research-event-policy.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), dir = (name: string) => path.join(root, "data/benchmarks", name);
const source = dir(arg("source")), slowSource = dir(arg("slow-source")), fastSource = dir(arg("fast-source")), output = dir(arg("output"));
assert.ok(arg("source") && arg("slow-source") && arg("fast-source") && arg("output") && !fs.existsSync(output));
const read = (directory: string, file: string) => JSON.parse(fs.readFileSync(path.join(directory, file), "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(source, "config.json"), modelFile = path.join(source, "model.json"), { model } = restoreEventPolicy(read(source, "model.json"));
const slowConfig = read(slowSource, "config.json"), slowModel = read(slowSource, "slow-model.json");
const fastConfig = read(fastSource, "config.json"), fastHead = read(fastSource, "head.json");
assert.equal(config.contract, "native-second-event-screen-v2"); assert.equal(config.scoredTestLoaded, false);
assert.equal(slowConfig.contract, "native-event-slow-direction-screen-v1"); assert.equal(path.resolve(slowConfig.source), source);
assert.equal(fastConfig.contract, "native-event-sign-screen-v2"); assert.equal(path.resolve(fastConfig.source), source);
assert.equal(fastConfig.modelHash, hash(modelFile)); assert.equal(fastConfig.objective, "return-weighted");
assert.deepEqual(fastConfig.featureNames, model.featureNames);

const DAY = 86_400_000, MINUTE = 60_000;
const minuteCandles = loadEventCandles(slowConfig.slowTrainStart - 1441 * MINUTE, config.calibrationEnd);
const minuteByClose = new Map(minuteCandles.map((row, index) => [row.openTime + MINUTE, index]));
const slowLaws = slowModel.kernels.map((kernel: any[]) => eventSignMass(kernel));
const includeFlow = usesNativeSecondTradeFlowFeatures(model.featureNames);
const nativeCandles = loadNativeEventCandles([{ start: config.fitStart - (config.warmupCandles + 1) * 1000,
  end: config.calibrationEnd }], includeFlow);
const nativeLaws = model.kernels.map((kernel: any[]) => {
  const mass = eventSignMass(kernel), positiveMean = kernel.reduce((sum, atom) => sum
    + (atom.return > 0 ? atom.probability * atom.return : 0), 0) / mass.positive;
  const negativeMean = kernel.reduce((sum, atom) => sum
    + (atom.return < 0 ? atom.probability * atom.return : 0), 0) / mass.negative;
  return { ...mass, positiveMean, negativeMean };
});
assert.ok(nativeLaws.every(law => law.positive > 0 && law.negative > 0));
type Row = MoveSample & { leaf: number; slowProbability: number; fastProbability: number };
const samples: Row[] = makeSamples(nativeCandles, model.clock, config.calibrationStart, config.calibrationEnd,
  config.excluded, config.stride, "stride", model.featureNames).map(row => {
  const leaf = eventLeaf(model, row.features), law = nativeLaws[leaf];
  const rawFast = predictEventSign(fastHead, row.features);
  const fastProbability = eventProbabilityFromReturnWeight(rawFast, law.positiveMean, law.negativeMean);
  const decisionTime = nativeCandles[row.start].openTime + 1000;
  const minute = minuteByClose.get(Math.floor(decisionTime / MINUTE) * MINUTE);
  assert.notEqual(minute, undefined);
  const slowLeaf = eventLeaf(slowModel, eventFeatures(minuteCandles, minute!, EVENT_FEATURES, slowModel.clock));
  return { ...row, leaf, slowProbability: slowLaws[slowLeaf].probability, fastProbability };
});
const split = config.calibrationStart + Math.floor((config.calibrationEnd - config.calibrationStart) / (2 * DAY)) * DAY;
const selection = samples.filter(row => nativeCandles[row.start].openTime + 1000 < split);
const holdout = samples.filter(row => nativeCandles[row.start].openTime + 1000 >= split);
const coefficients = [0, 0.5, 1, 1.5, 2];
const bounded = (p: number) => Math.max(1e-6, Math.min(1 - 1e-6, p));
const logit = (p: number) => Math.log(bounded(p) / (1 - bounded(p)));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, z))));
const probability = (row: Row, slowCoefficient: number, fastCoefficient: number) => {
  if (!slowCoefficient && !fastCoefficient) return nativeLaws[row.leaf].probability;
  return bounded(sigmoid(slowCoefficient * logit(row.slowProbability) + fastCoefficient * logit(row.fastProbability)));
};
const score = (rows: Row[], slowCoefficient: number, fastCoefficient: number) => {
  let nll = 0, mse = 0, zeroMse = 0, correct = 0, weightedCorrect = 0, magnitude = 0, maximum = 0, one = 0, round = 0;
  const cost = config.costs.feeBps + config.costs.slippageBps;
  for (const row of rows) {
    const law = nativeLaws[row.leaf], p = probability(row, slowCoefficient, fastCoefficient);
    const mean = (law.positive + law.negative) * (p * law.positiveMean + (1 - p) * law.negativeMean);
    const y = Number(row.return > 0); nll -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
    mse += (row.return - mean) ** 2; zeroMse += row.return ** 2;
    const hit = Math.sign(mean) === Math.sign(row.return); correct += Number(hit);
    weightedCorrect += Number(hit) * Math.abs(row.return); magnitude += Math.abs(row.return);
    const abs = Math.abs(mean) * 10_000; maximum = Math.max(maximum, abs); one += Number(abs > cost); round += Number(abs > 2 * cost);
  }
  return { samples: rows.length, signNll: nll / rows.length, returnMse: mse / rows.length,
    returnMseSkill: 1 - mse / zeroMse, directionAccuracy: correct / rows.length,
    magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude, maximumAbsoluteMeanBps: maximum,
    aboveOneWayCost: one, aboveRoundTripCost: round };
};
const byDay = (rows: Row[]) => Map.groupBy(rows, row => Math.floor((nativeCandles[row.start].openTime + 1000) / DAY));
const daily = (rows: Row[], slow: number, fast: number) => [...byDay(rows).values()].map(day => score(day, slow, fast));
const baseline = { selection: score(selection, 0, 0), holdout: score(holdout, 0, 0),
  selectionDays: daily(selection, 0, 0), holdoutDays: daily(holdout, 0, 0) };
const candidates = coefficients.flatMap(slowCoefficient => coefficients.map(fastCoefficient => ({ slowCoefficient, fastCoefficient })))
  .filter(row => row.slowCoefficient || row.fastCoefficient).map(specification => {
    const selectionDays = daily(selection, specification.slowCoefficient, specification.fastCoefficient);
    const holdoutDays = daily(holdout, specification.slowCoefficient, specification.fastCoefficient);
    return { specification, selection: score(selection, specification.slowCoefficient, specification.fastCoefficient),
      holdout: score(holdout, specification.slowCoefficient, specification.fastCoefficient), selectionDays, holdoutDays,
      selectionPositiveMseDays: selectionDays.filter((v, i) => v.returnMse < baseline.selectionDays[i]!.returnMse).length,
      selectionPositiveNllDays: selectionDays.filter((v, i) => v.signNll < baseline.selectionDays[i]!.signNll).length,
      holdoutPositiveMseDays: holdoutDays.filter((v, i) => v.returnMse < baseline.holdoutDays[i]!.returnMse).length,
      holdoutPositiveNllDays: holdoutDays.filter((v, i) => v.signNll < baseline.holdoutDays[i]!.signNll).length };
  });
const eligible = candidates.filter(row => row.selection.returnMse < baseline.selection.returnMse
  && row.selection.signNll < baseline.selection.signNll
  && row.selectionPositiveMseDays / row.selectionDays.length >= .6
  && row.selectionPositiveNllDays / row.selectionDays.length >= .6);
const selected = [...eligible].sort((a, b) => a.selection.returnMse - b.selection.returnMse)[0];
const promote = Boolean(selected && selected.holdout.returnMse < baseline.holdout.returnMse
  && selected.holdout.signNll < baseline.holdout.signNll
  && selected.holdoutPositiveMseDays / selected.holdoutDays.length >= .6
  && selected.holdoutPositiveNllDays / selected.holdoutDays.length >= .6
  && selected.holdout.aboveRoundTripCost > 0);

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-direction-ensemble-screen-v1", source, slowSource, fastSource,
  modelHash: hash(modelFile), coefficients, calibrationStart: config.calibrationStart, calibrationSplit: split,
  calibrationEnd: config.calibrationEnd, scoredTestLoaded: false,
  method: "Combine the strictly earlier minute-event P(up) and return-weighted native fast-head P(up) in symmetric log-odds space. Coefficients are nonnegative and contain no directional intercept. Reweight only the frozen native positive/negative masses, preserving conditional magnitude, duration, extrema, paths and successors. Select on the first seven days using pooled and 60%-of-day sign-NLL/MSE gates, then evaluate the later seven days once. No inspector window or policy replay is loaded." });
save("summary.json", { selection: selection.length, holdout: holdout.length, baseline, candidates,
  eligibleCandidates: eligible.length, selected, promote });
save("sources.json", Object.fromEntries(["scripts/screen-native-event-direction-ensemble.ts",
  "packages/bot-algo/src/event-sign.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
console.log(JSON.stringify({ selection: selection.length, holdout: holdout.length,
  baseline: { selection: baseline.selection, holdout: baseline.holdout }, eligibleCandidates: eligible.length, selected, promote }, null, 2));
