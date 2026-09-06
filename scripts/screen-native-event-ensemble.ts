/** Calibrate sign/magnitude combinations without touching the final test fit. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { predictEventSign, trainEventSign, type EventSignHead } from "../packages/bot-algo/src/event-sign.js";

type Blend = { blend: number; probability: number; expectedReturnBps: number };
type Row = {
  realizedReturnBps: number; leaf: number; activeMass: number;
  positiveMeanBps: number; negativeMeanBps: number; blends: Blend[];
};
type Fitted = { name: string; parameters: unknown; predict: (row: Row) => number };
const arg = (key: string) => { const index = process.argv.indexOf(`--${key}`); return index < 0 ? "" : process.argv[index + 1]; };
const root = path.resolve(__dirname, ".."), source = path.resolve(root, "data/benchmarks", arg("source"));
const output = path.resolve(root, "data/benchmarks", arg("output"));
assert.ok(arg("source") && arg("output") && fs.existsSync(source) && !fs.existsSync(output));
const read = <T>(file: string): T => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const calibration = read<Row[]>("calibration-predictions.json"), test = read<Row[]>("test-predictions.json");
assert.ok(calibration.length >= 100 && test.length > 0);
for (const row of [...calibration, ...test]) {
  assert.equal(row.blends.length, 3); assert.deepEqual(row.blends.map(value => value.blend), [0, .5, 1]);
  assert.ok(row.positiveMeanBps > 0 && row.negativeMeanBps < 0 && row.activeMass > 0 && row.activeMass <= 1 + 1e-10);
}
const boundedProbability = (p: number) => Math.max(1e-6, Math.min(1 - 1e-6, p));
const signProbability = (row: Row) => row.blends[1].probability;
const conditionalMean = (row: Row, probability: number, positiveScale = 1, negativeScale = 1) => row.activeMass
  * (probability * positiveScale * row.positiveMeanBps + (1 - probability) * negativeScale * row.negativeMeanBps);
const metrics = (rows: Row[], predict: (row: Row) => number) => {
  let error = 0, zero = 0, correct = 0, weightedCorrect = 0, magnitude = 0, predictedMean = 0, maxAbsolute = 0;
  for (const row of rows) {
    const prediction = predict(row), actual = row.realizedReturnBps;
    assert.ok(Number.isFinite(prediction)); error += (actual - prediction) ** 2; zero += actual ** 2;
    const isCorrect = Math.sign(prediction) === Math.sign(actual); correct += Number(isCorrect);
    weightedCorrect += Number(isCorrect) * Math.abs(actual); magnitude += Math.abs(actual);
    predictedMean += prediction; maxAbsolute = Math.max(maxAbsolute, Math.abs(prediction));
  }
  return { rows: rows.length, mseSkill: 1 - error / zero, directionAccuracy: correct / rows.length,
    magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude, predictedMeanBps: predictedMean / rows.length,
    maxAbsolutePredictionBps: maxAbsolute };
};
const fitScale = (rows: Row[], base: (row: Row) => number) => {
  let xy = 0, xx = 0;
  for (const row of rows) { const x = base(row); xy += x * row.realizedReturnBps; xx += x * x; }
  return Math.max(0, Math.min(10, xy / Math.max(xx, 1e-12)));
};
const fitConvex = (rows: Row[]): Fitted => {
  let numerator = 0, denominator = 0;
  for (const row of rows) {
    const base = row.blends[0].expectedReturnBps, delta = row.blends[2].expectedReturnBps - base;
    numerator += delta * (row.realizedReturnBps - base); denominator += delta * delta;
  }
  const weight = Math.max(0, Math.min(1, numerator / Math.max(denominator, 1e-12)));
  return { name: "mse-convex-base-head", parameters: { headWeight: weight },
    predict: row => row.blends[0].expectedReturnBps + weight
      * (row.blends[2].expectedReturnBps - row.blends[0].expectedReturnBps) };
};
const fitSignedMagnitude = (rows: Row[], soft: boolean, expectedAbsolute: boolean): Fitted => {
  const base = (row: Row) => {
    const p = signProbability(row), sign = soft ? 2 * p - 1 : (p >= .5 ? 1 : -1);
    const magnitude = expectedAbsolute
      ? row.activeMass * (row.blends[2].probability * row.positiveMeanBps
        - (1 - row.blends[2].probability) * row.negativeMeanBps)
      : Math.abs(row.blends[2].expectedReturnBps);
    return sign * magnitude;
  };
  const scale = fitScale(rows, base);
  return { name: `${soft ? "soft" : "hard"}-direction-${expectedAbsolute ? "expected-absolute" : "head-absolute"}`,
    parameters: { scale }, predict: row => scale * base(row) };
};
const fitPlatt = (rows: Row[], penalty: number): EventSignHead => trainEventSign(rows.map(row => ({
  features: [Math.log(boundedProbability(signProbability(row)) / (1 - boundedProbability(signProbability(row))))],
  return: row.realizedReturnBps,
})), penalty);
const plattProbability = (row: Row, head: EventSignHead) => {
  const p = boundedProbability(signProbability(row));
  return predictEventSign(head, [Math.log(p / (1 - p))]);
};
const fitMagnitudeScales = (rows: Row[], head: EventSignHead, penalty: number) => {
  let aa = penalty, ab = 0, bb = penalty, ay = penalty, by = penalty;
  for (const row of rows) {
    const p = plattProbability(row, head), a = row.activeMass * p * row.positiveMeanBps;
    const b = row.activeMass * (1 - p) * row.negativeMeanBps, y = row.realizedReturnBps;
    aa += a * a / rows.length; ab += a * b / rows.length; bb += b * b / rows.length;
    ay += a * y / rows.length; by += b * y / rows.length;
  }
  const determinant = aa * bb - ab * ab;
  const clamp = (value: number) => Math.max(0, Math.min(4, value));
  return { positive: clamp((ay * bb - by * ab) / determinant), negative: clamp((by * aa - ay * ab) / determinant) };
};
const fitPlattLaw = (rows: Row[], signPenalty: number, magnitudePenalty?: number): Fitted => {
  const head = fitPlatt(rows, signPenalty);
  const scales = magnitudePenalty === undefined ? { positive: 1, negative: 1 }
    : fitMagnitudeScales(rows, head, magnitudePenalty);
  return { name: magnitudePenalty === undefined ? `platt-sign-p${signPenalty}`
    : `platt-sign-magnitude-p${signPenalty}-r${magnitudePenalty}`,
  parameters: { head, scales }, predict: row => conditionalMean(row, plattProbability(row, head), scales.positive, scales.negative) };
};
const fitCandidates = (rows: Row[]): Fitted[] => [
  { name: "fixed-half-probability-blend", parameters: {}, predict: row => row.blends[1].expectedReturnBps },
  fitConvex(rows), fitSignedMagnitude(rows, false, false), fitSignedMagnitude(rows, true, false),
  fitSignedMagnitude(rows, false, true), fitSignedMagnitude(rows, true, true),
  ...[.01, .1, 1].flatMap(penalty => [fitPlattLaw(rows, penalty), fitPlattLaw(rows, penalty, 100)]),
];
const boundary = Math.floor(calibration.length / 2), fitRows = calibration.slice(0, boundary), selectionRows = calibration.slice(boundary);
const selection = fitCandidates(fitRows).map(candidate => ({ name: candidate.name, parameters: candidate.parameters,
  fit: metrics(fitRows, candidate.predict), selection: metrics(selectionRows, candidate.predict) }));
selection.sort((a, b) => b.selection.mseSkill - a.selection.mseSkill);
const selectedName = selection[0].name, refitted = fitCandidates(calibration).find(candidate => candidate.name === selectedName)!;
const baselines = [0, .5, 1].map((blend, index) => ({ blend,
  calibration: metrics(calibration, row => row.blends[index].expectedReturnBps),
  test: metrics(test, row => row.blends[index].expectedReturnBps) }));
const result = { contract: "native-event-sign-magnitude-ensemble-screen-v1", source,
  fitContract: "Chronological first half of calibration fits each candidate; second half selects by return MSE; selected family is refit on all calibration before one final test evaluation.",
  calibrationRows: calibration.length, fitRows: fitRows.length, selectionRows: selectionRows.length, testRows: test.length,
  selection, selected: { name: refitted.name, parameters: refitted.parameters,
    calibration: metrics(calibration, refitted.predict), test: metrics(test, refitted.predict) }, baselines };
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(new URL(import.meta.url), "utf8"));
console.log(JSON.stringify({ selected: result.selected, selection: selection.slice(0, 5), baselines }));
