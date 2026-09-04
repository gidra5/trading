/** Isolate a native event-direction head before paying for policy integration. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventProbabilityFromReturnWeight, eventSignMass, predictEventSign, trainEventSign } from "../packages/bot-algo/src/event-sign.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { loadNativeEventCandles, makeSamples } from "./research-event-policy.js";
const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const directory = (name: string) => path.resolve(__dirname, "../data/benchmarks", name), source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
assert.ok(config.frozenPartitionSource && config.partitionStart < config.partitionEnd);
const bytes = fs.readFileSync(path.join(source, "model.json")), { model } = restoreEventPolicy(JSON.parse(bytes.toString()));
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
for (const ref of config.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
const penalty = Number(arg("penalty", "0.1")); assert.ok(penalty > 0 && Number.isFinite(penalty));
const objective = arg("objective", "ordinary"); assert.ok(["ordinary", "return-weighted"].includes(objective));
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-sign-screen-v1", source, modelHash: hash(bytes), penalty, objective, featureNames: model.featureNames,
  trainStart: config.partitionStart, trainEnd: config.partitionEnd, calibrationStart: config.calibrationStart, calibrationEnd: config.calibrationEnd,
  testStart: config.start, testEnd: config.end, blends: [0, .5, 1],
  method: "Reuse the existing penalized logistic active-sign head on native cost-sized event targets and declared causal features. Fit only on the original three partition-training days; the joint magnitude/duration/extrema/successor law stays frozen and its estimation labels are disjoint. Train-only scaling. The optional return-weighted objective uses absolute target returns and converts its weighted score back to ordinary sign probability using the frozen conditional magnitudes. Compare fixed blends 0, 0.5, 1 on identical diagnostic populations; preserve zero mass and conditional positive/negative magnitudes. No policy or PnL simulation. These research dates have already been inspected." });
save("sources.json", Object.fromEntries(["scripts/screen-native-event-sign.ts", "packages/bot-algo/src/event-sign.ts",
  "scripts/event-fit-periods.ts",
  "packages/bot-algo/src/event-distribution.ts", "scripts/research-event-policy.ts"].map(file => [file, fs.readFileSync(path.resolve(__dirname, "..", file), "utf8")])));
const start = performance.now();
const candles = loadNativeEventCandles([
  { start: config.partitionStart - (config.warmupCandles + 1) * 1000, end: config.calibrationEnd },
  { start: config.start - (config.warmupCandles + 1) * 1000, end: config.end },
]);
const sample = (start: number, end: number, sampling: "chain" | "stride", excluded = config.excluded) =>
  makeSamples(candles, model.clock, start, end, excluded, sampling === "chain" ? 1 : config.stride, sampling, model.featureNames);
const training = sample(config.partitionStart, config.partitionEnd, "stride");
const calibration = sample(config.calibrationStart, config.calibrationEnd, "stride"), test = sample(config.start, config.end, "chain", []);
const head = trainEventSign(training, penalty, objective === "return-weighted" ? training.map(s => Math.abs(s.return)) : undefined); save("head.json", head);
const laws = model.kernels.map(kernel => {
  const mass = eventSignMass(kernel), positive = kernel.reduce((s, a) => s + (a.return > 0 ? a.probability * a.return : 0), 0);
  const negative = kernel.reduce((s, a) => s + (a.return < 0 ? a.probability * a.return : 0), 0);
  return { ...mass, positiveMean: positive / (mass.positive || 1), negativeMean: negative / (mass.negative || 1) };
});
const score = (rows: MoveSample[], blend: number) => {
  let ce = 0, correct = 0, active = 0, weightedCorrect = 0, magnitude = 0, mse = 0, zeroMse = 0, mean = 0;
  for (const row of rows) {
    const law = laws[eventLeaf(model, row.features)], raw = predictEventSign(head, row.features);
    const predicted = objective === "return-weighted" && law.positive && law.negative
      ? eventProbabilityFromReturnWeight(raw, law.positiveMean, law.negativeMean) : raw;
    const p = law.positive && law.negative ? (1 - blend) * law.probability + blend * predicted : law.probability;
    const expected = (law.positive + law.negative) * (p * law.positiveMean + (1 - p) * law.negativeMean);
    mse += (row.return - expected) ** 2; zeroMse += row.return ** 2; mean += expected;
    if (row.return) {
      const y = Number(row.return > 0), isCorrect = (p > .5 ? 1 : 0) === y;
      ce -= y * Math.log(Math.max(1e-12, p)) + (1 - y) * Math.log(Math.max(1e-12, 1 - p));
      correct += Number(isCorrect); weightedCorrect += Number(isCorrect) * Math.abs(row.return);
      active++; magnitude += Math.abs(row.return);
    }
  }
  return { samples: rows.length, active, signLogLoss: ce / active, directionAccuracy: correct / active,
    magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude, mseSkill: 1 - mse / zeroMse, predictedMeanBps: mean / rows.length * 10000 };
};
const results = [0, .5, 1].map(blend => ({ blend, training: score(training, blend), calibration: score(calibration, blend), test: score(test, blend) }));
save("summary.json", { training: training.length, calibration: calibration.length, test: test.length, results,
  constantHalfProbabilitySignLogLoss: Math.log(2),
  elapsedSeconds: (performance.now() - start) / 1000 });
console.log(JSON.stringify({ results, elapsedSeconds: (performance.now() - start) / 1000 }));
