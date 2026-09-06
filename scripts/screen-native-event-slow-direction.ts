/** Calibration-only transfer of a strictly earlier minute event law into native event direction. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { EVENT_FEATURES, eventFeatures, eventLeaf, trainEventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSignMass } from "../packages/bot-algo/src/event-sign.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { eventSourceDays } from "./event-fit-periods.js";
import { loadEventCandles, loadNativeEventCandles, makeSamples } from "./research-event-policy.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.join(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const config = read(path.join(source, "config.json")), modelFile = path.join(source, "model.json");
assert.equal(config.contract, "native-second-event-screen-v2");
assert.equal(config.scoredTestLoaded, false, "Requires a calibration-only native law");
const { model } = restoreEventPolicy(read(modelFile));
for (const ref of config.sourceReferences) assert.equal(hash(ref.file), ref.sha256);

const DAY = 86_400_000, MINUTE = 60_000, partitionEnd = config.partitionEnd ?? config.fitEnd - DAY;
const slowTrainDays = 120, slowTrainStart = partitionEnd - slowTrainDays * DAY;
const slowClock = { thresholdBps: 120, maxCandles: 1440 } as const;
const slowSourceStart = slowTrainStart - 1441 * MINUTE;
const slowCandles = loadEventCandles(slowSourceStart, config.calibrationEnd);
const slowTraining = makeSamples(slowCandles, slowClock, slowTrainStart, partitionEnd, config.excluded, 30, "stride", EVENT_FEATURES);
assert.ok(slowTraining.length >= 200);
const slowModel = trainEventDistribution(slowTraining, slowClock,
  { maxDepth: 2, minLeaf: 100, prior: 100, criterion: "mean", featureNames: EVENT_FEATURES });
const slowLaws = slowModel.kernels.map(kernel => {
  const sign = eventSignMass(kernel);
  return { directionProbability: sign.probability,
    meanBps: kernel.reduce((sum, atom) => sum + atom.probability * atom.return, 0) * 10_000 };
});
const minuteByClose = new Map(slowCandles.map((row, index) => [row.openTime + MINUTE, index]));

const includeTradeFlow = usesNativeSecondTradeFlowFeatures(model.featureNames);
const nativeRanges = [{ start: config.fitStart - (config.warmupCandles + 1) * 1000, end: config.calibrationEnd }];
const nativeCandles = loadNativeEventCandles(nativeRanges, includeTradeFlow);
type Row = MoveSample & { slowLeaf: number; slowDirectionProbability: number; slowMeanBps: number };
const samples = makeSamples(nativeCandles, model.clock, config.calibrationStart, config.calibrationEnd,
  config.excluded, config.stride, "stride", model.featureNames).map((row): Row => {
  const decisionTime = nativeCandles[row.start].openTime + 1000;
  const closeTime = Math.floor(decisionTime / MINUTE) * MINUTE, minute = minuteByClose.get(closeTime);
  assert.notEqual(minute, undefined, "Missing last completed minute");
  const slowLeaf = eventLeaf(slowModel, eventFeatures(slowCandles, minute!, EVENT_FEATURES, slowClock));
  return { ...row, slowLeaf, slowDirectionProbability: slowLaws[slowLeaf].directionProbability,
    slowMeanBps: slowLaws[slowLeaf].meanBps };
});
const split = config.calibrationStart + Math.floor((config.calibrationEnd - config.calibrationStart) / (2 * DAY)) * DAY;
const selection = samples.filter(row => nativeCandles[row.start].openTime + 1000 < split);
const holdout = samples.filter(row => nativeCandles[row.start].openTime + 1000 >= split);
assert.equal(selection.length + holdout.length, samples.length);

type Group = "down" | "timeout" | "up";
const groups = ["down", "timeout", "up"] as const;
const group = (value: number): Group => value * 10_000 <= -model.clock.thresholdBps ? "down"
  : value * 10_000 >= model.clock.thresholdBps ? "up" : "timeout";
const nativeLaws = model.kernels.map(kernel => {
  const mass = { down: 0, timeout: 0, up: 0 }, weighted = { down: 0, timeout: 0, up: 0 };
  for (const atom of kernel) { const g = group(atom.return); mass[g] += atom.probability; weighted[g] += atom.probability * atom.return; }
  return { mass, mean: Object.fromEntries(groups.map(g => [g, weighted[g] / mass[g]])) as Record<Group, number> };
});
assert.ok(nativeLaws.every(law => groups.every(g => law.mass[g] > 0)));
const weights = [0, 0.25, 0.5, 0.75, 1], logOddsScales = [0.5, 1, 2, 4, 8];
const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
const scaleProbability = (probability: number, scale: number) => sigmoid(scale * Math.log(probability / (1 - probability)));
const probabilities = (row: Row, weight: number, logOddsScale: number) => {
  const law = nativeLaws[eventLeaf(model, row.features)], hit = law.mass.down + law.mass.up;
  const baseUp = law.mass.up / hit, slowUp = scaleProbability(row.slowDirectionProbability, logOddsScale);
  const up = (1 - weight) * baseUp + weight * slowUp;
  return { law, p: { down: hit * (1 - up), timeout: law.mass.timeout, up: hit * up } };
};
const score = (rows: Row[], weight: number, logOddsScale: number) => {
  let nll = 0, mse = 0, zeroMse = 0, correct = 0, weightedCorrect = 0, magnitude = 0, maximum = 0, one = 0, round = 0;
  const cost = config.costs.feeBps + config.costs.slippageBps;
  for (const row of rows) {
    const forecast = probabilities(row, weight, logOddsScale), actual = group(row.return);
    const mean = groups.reduce((sum, g) => sum + forecast.p[g] * forecast.law.mean[g], 0);
    nll -= Math.log(Math.max(1e-12, forecast.p[actual])); mse += (row.return - mean) ** 2; zeroMse += row.return ** 2;
    const hit = Math.sign(mean) === Math.sign(row.return); correct += Number(hit);
    weightedCorrect += Number(hit) * Math.abs(row.return); magnitude += Math.abs(row.return);
    const abs = Math.abs(mean) * 10_000; maximum = Math.max(maximum, abs); one += Number(abs > cost); round += Number(abs > 2 * cost);
  }
  return { samples: rows.length, groupNll: nll / rows.length, returnMse: mse / rows.length,
    returnMseSkill: 1 - mse / zeroMse, directionAccuracy: correct / rows.length,
    magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude, maximumAbsoluteMeanBps: maximum,
    aboveOneWayCost: one, aboveRoundTripCost: round };
};
const byDay = (rows: Row[]) => Map.groupBy(rows, row => Math.floor((nativeCandles[row.start].openTime + 1000) / DAY));
const daily = (rows: Row[], weight: number, scale: number) => [...byDay(rows).values()].map(day => score(day, weight, scale));
const baseline = { selection: score(selection, 0, 1), holdout: score(holdout, 0, 1),
  selectionDays: daily(selection, 0, 1), holdoutDays: daily(holdout, 0, 1) };
const candidates = weights.slice(1).flatMap(weight => logOddsScales.map(logOddsScale => {
  const selectionDays = daily(selection, weight, logOddsScale), holdoutDays = daily(holdout, weight, logOddsScale);
  return { weight, logOddsScale, selection: score(selection, weight, logOddsScale),
    holdout: score(holdout, weight, logOddsScale), selectionDays, holdoutDays,
    selectionPositiveMseDays: selectionDays.filter((v, i) => v.returnMse < baseline.selectionDays[i]!.returnMse).length,
    selectionPositiveNllDays: selectionDays.filter((v, i) => v.groupNll < baseline.selectionDays[i]!.groupNll).length,
    holdoutPositiveMseDays: holdoutDays.filter((v, i) => v.returnMse < baseline.holdoutDays[i]!.returnMse).length,
    holdoutPositiveNllDays: holdoutDays.filter((v, i) => v.groupNll < baseline.holdoutDays[i]!.groupNll).length };
}));
const eligible = candidates.filter(row => row.selection.returnMse < baseline.selection.returnMse
  && row.selection.groupNll < baseline.selection.groupNll
  && row.selectionPositiveMseDays / row.selectionDays.length >= .6
  && row.selectionPositiveNllDays / row.selectionDays.length >= .6);
const selected = [...eligible].sort((a, b) => a.selection.returnMse - b.selection.returnMse)[0];
const promote = Boolean(selected && selected.holdout.returnMse < baseline.holdout.returnMse
  && selected.holdout.groupNll < baseline.holdout.groupNll
  && selected.holdoutPositiveMseDays / selected.holdoutDays.length >= .6
  && selected.holdoutPositiveNllDays / selected.holdoutDays.length >= .6
  && selected.holdout.aboveRoundTripCost > 0);

const slowSourceReferences = eventSourceDays([{ start: slowSourceStart, end: config.calibrationEnd }]).flatMap(day => {
  const file = path.join(root, `data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m/${new Date(day).toISOString().slice(0, 10)}.json`);
  return fs.existsSync(file) ? [{ file, sha256: hash(file) }] : [];
});
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-slow-direction-screen-v1", source, nativeModelHash: hash(modelFile),
  slowClock, slowTrainDays, slowTrainStart, slowTrainEnd: partitionEnd, slowSpecification: {
    stride: 30, maxDepth: 2, minLeaf: 100, prior: 100, criterion: "mean", featureNames: EVENT_FEATURES },
  calibrationStart: config.calibrationStart, calibrationSplit: split, calibrationEnd: config.calibrationEnd,
  weights, logOddsScales, scoredTestLoaded: false, slowSourceReferences,
  method: "Train a fixed minute-resolution 120 bp/one-day event tree entirely before native calibration, using the older v10 hyperparameters without consulting this screen's outcomes. At each native origin use only the last fully completed minute. Apply a symmetry-preserving scale to its log-odds around P=0.5, then blend that P(up) into the frozen native P(up | 48 bp barrier), preserving native barrier-arrival probability and every conditional path. Select scale and blend on the first seven calibration days with pooled and 60%-of-day NLL/MSE gates; evaluate the later seven days once. No inspector window or policy replay is loaded.",
  caveat: "The slow and native event targets differ. A selected blend is a transfer prior, not a probability identity, and requires the untouched and economic gates before integration." });
save("slow-model.json", slowModel);
save("summary.json", { slowTraining: slowTraining.length, slowLeaves: slowModel.kernels.length,
  selection: selection.length, holdout: holdout.length, baseline, candidates, eligibleCandidates: eligible.length, selected, promote });
save("sources.json", Object.fromEntries(["scripts/screen-native-event-slow-direction.ts", "packages/bot-algo/src/event-distribution.ts",
  "scripts/research-event-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
console.log(JSON.stringify({ slowTraining: slowTraining.length, slowLeaves: slowModel.kernels.length,
  selection: selection.length, holdout: holdout.length, baseline: { selection: baseline.selection, holdout: baseline.holdout },
  eligibleCandidates: eligible.length, selected, promote }, null, 2));
