/** Estimate epistemic uncertainty in the frozen base/sign probability blend. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

type Blend = { blend: number; probability: number; expectedReturnBps: number };
type Row = { start: number; realizedReturnBps: number; blends: Blend[] };

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, "..");
const source = path.resolve(root, "data/benchmarks", arg("source"));
const output = path.resolve(root, "data/benchmarks", arg("output"));
assert.ok(arg("source") && arg("output") && fs.existsSync(source) && !fs.existsSync(output));
const read = <T>(name: string): T => JSON.parse(fs.readFileSync(path.join(source, name), "utf8"));
const calibration = read<Row[]>("calibration-predictions.json");
const test = read<Row[]>("test-predictions.json");
assert.ok(calibration.length >= 100 && test.length > 0);
for (const row of [...calibration, ...test]) assert.deepEqual(row.blends.map(value => value.blend), [0, .5, 1]);

const prediction = (row: Row, weight: number) => row.blends[0].expectedReturnBps
  + weight * (row.blends[2].expectedReturnBps - row.blends[0].expectedReturnBps);
const fitWeight = (rows: readonly Row[]) => {
  let numerator = 0, denominator = 0;
  for (const row of rows) {
    const base = row.blends[0].expectedReturnBps;
    const delta = row.blends[2].expectedReturnBps - base;
    numerator += delta * (row.realizedReturnBps - base);
    denominator += delta * delta;
  }
  return Math.max(0, Math.min(1, numerator / Math.max(denominator, 1e-12)));
};
const metrics = (rows: readonly Row[], weight: number) => {
  let error = 0, zero = 0, correct = 0, weightedCorrect = 0, magnitude = 0;
  for (const row of rows) {
    const predicted = prediction(row, weight), actual = row.realizedReturnBps;
    error += (actual - predicted) ** 2; zero += actual ** 2;
    const hit = Math.sign(predicted) === Math.sign(actual);
    correct += Number(hit); weightedCorrect += Number(hit) * Math.abs(actual); magnitude += Math.abs(actual);
  }
  return { rows: rows.length, mseSkill: 1 - error / zero, directionAccuracy: correct / rows.length,
    magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude };
};
const uncertainty = (rows: readonly Row[], low: number, high: number) => {
  let containsZero = 0, robustMagnitude = 0, aboveOneWayCost = 0;
  for (const row of rows) {
    const endpoints = [prediction(row, low), prediction(row, high)];
    const lo = Math.min(...endpoints), hi = Math.max(...endpoints);
    containsZero += Number(lo <= 0 && hi >= 0);
    const robust = lo > 0 ? lo : hi < 0 ? -hi : 0;
    robustMagnitude += robust; aboveOneWayCost += Number(robust > 12);
  }
  return { rows: rows.length, containsZero, containsZeroFraction: containsZero / rows.length,
    meanRobustAbsoluteBps: robustMagnitude / rows.length, aboveOneWayCost };
};

const blocks = new Map<number, Row[]>();
for (const row of calibration) {
  const key = Math.floor(row.start / 3600);
  const block = blocks.get(key); if (block) block.push(row); else blocks.set(key, [row]);
}
const blockRows = [...blocks.values()];
assert.ok(blockRows.length >= 12 && blockRows.every(rows => rows.length > 0));
let seed = 0x5eeda11;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
const bootstrapWeights = Array.from({ length: 2000 }, () => {
  const sampled: Row[] = [];
  for (let i = 0; i < blockRows.length; i++) sampled.push(...blockRows[Math.floor(random() * blockRows.length)]);
  return fitWeight(sampled);
}).sort((a, b) => a - b);
const quantile = (p: number) => bootstrapWeights[Math.round(p * (bootstrapWeights.length - 1))];
const interval = { low: quantile(.05), high: quantile(.95), confidence: .9 };
const central = fitWeight(calibration), boundary = Math.floor(calibration.length / 2);
const result = {
  contract: "native-event-sign-blend-uncertainty-v1", source,
  fitContract: "Fit the convex base/head expected-return blend on calibration only. Estimate a 90% epistemic interval by resampling contiguous one-hour calibration blocks with replacement using a fixed seed. Freeze the central weight and interval before reporting test metrics.",
  calibrationRows: calibration.length, testRows: test.length, blocks: blockRows.length,
  centralWeight: central, chronologicalWeights: {
    firstHalf: fitWeight(calibration.slice(0, boundary)), secondHalf: fitWeight(calibration.slice(boundary)),
  },
  bootstrap: { samples: bootstrapWeights.length, seed: "0x5eeda11", interval,
    median: quantile(.5), minimum: bootstrapWeights[0], maximum: bootstrapWeights.at(-1) },
  calibration: { central: metrics(calibration, central), low: metrics(calibration, interval.low),
    high: metrics(calibration, interval.high), uncertainty: uncertainty(calibration, interval.low, interval.high) },
  test: { central: metrics(test, central), low: metrics(test, interval.low), high: metrics(test, interval.high),
    uncertainty: uncertainty(test, interval.low, interval.high) },
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(new URL(import.meta.url), "utf8"));
console.log(JSON.stringify(result));
