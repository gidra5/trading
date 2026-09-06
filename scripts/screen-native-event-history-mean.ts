/** Calibration-only ridge screen for a causal completed-event sequence state. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { EventCompletedHistory, EVENT_COMPLETED_HISTORY_INPUTS } from "../packages/bot-algo/src/event-completed-history.js";
import { eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventConformalResidualRadius, eventIntervalConfidence } from "../packages/bot-algo/src/event-uncertainty.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { loadNativeEventCandles, makeSamples } from "./research-event-policy.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), benchmark = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = benchmark(arg("source")), output = benchmark(arg("output"));
assert.ok(arg("source") && arg("output") && fs.existsSync(source) && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const hash = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
const config = read("config.json"), modelBytes = fs.readFileSync(path.join(source, "model.json"));
const policy = restoreEventPolicy(JSON.parse(modelBytes.toString())), model = policy.model;
assert.equal(config.contract, "native-second-event-screen-v2");
assert.equal(config.scoredTestLoaded, false); assert.equal(model.clock.candleIntervalMs, 1000);
assert.ok(config.fitStart < config.fitEnd && config.fitEnd === config.calibrationStart
  && config.calibrationStart < config.calibrationEnd);
for (const reference of config.sourceReferences)
  assert.equal(hash(fs.readFileSync(reference.file)), reference.sha256);

const candles = loadNativeEventCandles(config.sourceRanges, usesNativeSecondTradeFlowFeatures(model.featureNames));
const chain = (start: number, end: number) => makeSamples(candles, model.clock, start, end,
  config.excluded, 1, "chain", model.featureNames);
const training = chain(config.fitStart, config.fitEnd), calibration = chain(config.calibrationStart, config.calibrationEnd);
assert.ok(training.length >= 100 && calibration.length >= 100);
assert.ok(training.every(row => row.end < calibration[0].start));

type SequenceRow = MoveSample & { at: number; baseMean: number; historyFeatures: number[] };
const sequence = (rows: readonly MoveSample[]): SequenceRow[] => {
  const out: SequenceRow[] = [];
  let previousEnd = -1, history: EventCompletedHistory | undefined;
  for (const row of rows) {
    const at = candles[row.start].openTime + 1_000, availableAt = candles[row.end].openTime + 1_000;
    if (row.start !== previousEnd) history = new EventCompletedHistory(at);
    const historyFeatures = history!.features(at);
    const baseMean = model.kernels[eventLeaf(model, row.features)]
      .reduce((sum, atom) => sum + atom.probability * atom.return, 0);
    out.push({ ...row, at, baseMean, historyFeatures });
    history!.observe({ originTime: at, availableAt, return: row.return, duration: row.duration }, availableAt);
    previousEnd = row.end;
  }
  return out;
};
const trainRows = sequence(training), calibrationRows = sequence(calibration);
const splitAt = Math.floor(calibrationRows.length / 2), selectionRows = calibrationRows.slice(0, splitAt);
const validationRows = calibrationRows.slice(splitAt);
assert.ok(selectionRows.length >= 40 && validationRows.length >= 40
  && selectionRows.every(row => row.end <= validationRows[0].start));

const compactNames = ["ema2-acceleration-1s-bps", "log1p-rv-60s-bps", "run-return-capped-63s-bps",
  "return-300s-bps", "return-900s-bps", "return-3600s-bps", "return-14400s-bps",
  "spot-aggregate-count-imbalance-1s", "spot-flow-quote-imbalance-ema-2", "spot-flow-maximum-skew-1s"];
const compactIndices = compactNames.map(name => {
  const index = model.featureNames.indexOf(name); assert.ok(index >= 0, `Missing sequence feature ${name}`); return index;
});
const featureSets = {
  "base-only": (row: SequenceRow) => [row.baseMean],
  "base-history": (row: SequenceRow) => [row.baseMean, ...row.historyFeatures],
  "base-context-history": (row: SequenceRow) => [row.baseMean,
    ...compactIndices.map(index => row.features[index]), ...row.historyFeatures],
};
type FeatureSet = keyof typeof featureSets;

interface RidgeModel { featureSet: FeatureSet; penalty: number; means: number[]; scales: number[];
  intercept: number; coefficients: number[]; }
const fitRidge = (rows: readonly SequenceRow[], featureSet: FeatureSet, penalty: number): RidgeModel => {
  const vectors = rows.map(featureSets[featureSet]), width = vectors[0].length;
  const means = Array.from({ length: width }, (_, feature) =>
    vectors.reduce((sum, values) => sum + values[feature], 0) / vectors.length);
  const scales = means.map((mean, feature) => Math.max(1e-8, Math.sqrt(vectors.reduce(
    (sum, values) => sum + (values[feature] - mean) ** 2, 0) / vectors.length)));
  const intercept = rows.reduce((sum, row) => sum + row.return, 0) / rows.length;
  const matrix = Array.from({ length: width }, () => new Array<number>(width + 1).fill(0));
  for (let row = 0; row < rows.length; row++) {
    const values = vectors[row].map((value, feature) => Math.max(-5, Math.min(5,
      (value - means[feature]) / scales[feature])));
    for (let i = 0; i < width; i++) {
      matrix[i][width] += values[i] * (rows[row].return - intercept);
      for (let j = 0; j < width; j++) matrix[i][j] += values[i] * values[j];
    }
  }
  for (let i = 0; i < width; i++) matrix[i][i] += penalty * rows.length;
  for (let column = 0; column < width; column++) {
    let pivot = column;
    for (let row = column + 1; row < width; row++)
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    const divisor = matrix[column][column]; assert.ok(Math.abs(divisor) > 1e-14);
    for (let j = column; j <= width; j++) matrix[column][j] /= divisor;
    for (let row = 0; row < width; row++) if (row !== column) {
      const factor = matrix[row][column];
      for (let j = column; j <= width; j++) matrix[row][j] -= factor * matrix[column][j];
    }
  }
  return { featureSet, penalty, means, scales, intercept, coefficients: matrix.map(row => row[width]) };
};
const predict = (ridge: RidgeModel, row: SequenceRow) => ridge.intercept
  + ridge.coefficients.reduce((sum, coefficient, feature) => sum + coefficient * Math.max(-5, Math.min(5,
    (featureSets[ridge.featureSet](row)[feature] - ridge.means[feature]) / ridge.scales[feature])), 0);
const metrics = (rows: readonly SequenceRow[], forecast: (row: SequenceRow) => number) => {
  let error = 0, zero = 0, base = 0, correct = 0, weightedCorrect = 0, magnitude = 0;
  let maximum = 0;
  const blocks = new Map<number, number>();
  for (const row of rows) {
    const value = forecast(row), actual = row.return, hit = Math.sign(value) === Math.sign(actual);
    assert.ok(Number.isFinite(value));
    error += (actual - value) ** 2; zero += actual ** 2; base += (actual - row.baseMean) ** 2;
    correct += Number(hit); weightedCorrect += Number(hit) * Math.abs(actual); magnitude += Math.abs(actual);
    maximum = Math.max(maximum, Math.abs(value) * 10_000);
    const day = Math.floor(row.at / 86_400_000);
    blocks.set(day, (blocks.get(day) ?? 0) + (actual - row.baseMean) ** 2 - (actual - value) ** 2);
  }
  return { rows: rows.length, mseSkillVsZero: 1 - error / zero, mseImprovementVsBase: 1 - error / base,
    directionAccuracy: correct / rows.length, magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude,
    maximumAbsoluteMeanBps: maximum, days: blocks.size,
    positiveDays: [...blocks.values()].filter(value => value > 0).length };
};
const specifications = (Object.keys(featureSets) as FeatureSet[]).flatMap(featureSet =>
  [.1, 1, 10, 100].map(penalty => ({ featureSet, penalty })));
const candidates = specifications.map(specification => {
  const ridge = fitRidge(trainRows, specification.featureSet, specification.penalty);
  return { specification, ridge, training: metrics(trainRows, row => predict(ridge, row)),
    selection: metrics(selectionRows, row => predict(ridge, row)) };
});
const eligible = candidates.filter(candidate => candidate.selection.mseImprovementVsBase > 0
  && candidate.selection.mseSkillVsZero > 0
  && candidate.selection.positiveDays / candidate.selection.days >= .6)
  .sort((left, right) => right.selection.mseImprovementVsBase - left.selection.mseImprovementVsBase);
const selected = eligible[0];
const validation = selected ? metrics(validationRows, row => predict(selected.ridge, row)) : null;
const residualRadiusBps = selected ? eventConformalResidualRadius(validationRows.map(row => ({
  predicted: predict(selected.ridge, row) * 10_000, realized: row.return * 10_000 })), .75) : null;
const uncertainty = selected && Number.isFinite(residualRadiusBps) ? validationRows.map(row => {
  const meanBps = predict(selected.ridge, row) * 10_000;
  const lowBps = meanBps - residualRadiusBps!, highBps = meanBps + residualRadiusBps!;
  return { at: row.at, meanBps, lowBps, highBps, confidence: eventIntervalConfidence(meanBps, lowBps, highBps),
    robustMarginBps: Math.max(0, Math.abs(meanBps) - residualRadiusBps!) };
}) : [];
const gate = Boolean(selected && validation && validation.mseImprovementVsBase > 0 && validation.mseSkillVsZero > 0
  && validation.positiveDays / validation.days >= .6 && uncertainty.some(row => row.robustMarginBps > 24));

interface SignModel { featureSet: FeatureSet; penalty: number; objective: "ordinary" | "return-weighted";
  means: number[]; scales: number[]; intercept: number; coefficients: number[]; }
const leafSigns = model.kernels.map(kernel => {
  const positive = kernel.filter(atom => atom.return > 0), negative = kernel.filter(atom => atom.return < 0);
  const positiveMass = positive.reduce((sum, atom) => sum + atom.probability, 0);
  const negativeMass = negative.reduce((sum, atom) => sum + atom.probability, 0), active = positiveMass + negativeMass;
  return { active, probability: positiveMass / active,
    positiveMean: positive.reduce((sum, atom) => sum + atom.probability * atom.return, 0) / positiveMass,
    negativeMean: negative.reduce((sum, atom) => sum + atom.probability * atom.return, 0) / negativeMass };
});
const rowSigns = (row: SequenceRow) => leafSigns[eventLeaf(model, row.features)];
const signVector = (featureSet: FeatureSet, row: SequenceRow) => {
  const values = featureSets[featureSet](row), p = Math.max(1e-6, Math.min(1 - 1e-6, rowSigns(row).probability));
  values[0] = Math.log(p / (1 - p)); return values;
};
const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
const solve = (matrix: number[][]) => {
  const width = matrix.length;
  for (let column = 0; column < width; column++) {
    let pivot = column;
    for (let row = column + 1; row < width; row++)
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    const divisor = matrix[column][column]; assert.ok(Math.abs(divisor) > 1e-14);
    for (let j = column; j <= width; j++) matrix[column][j] /= divisor;
    for (let row = 0; row < width; row++) if (row !== column) {
      const factor = matrix[row][column];
      for (let j = column; j <= width; j++) matrix[row][j] -= factor * matrix[column][j];
    }
  }
  return matrix.map(row => row[width]);
};
const fitSign = (rows: readonly SequenceRow[], featureSet: FeatureSet, penalty: number,
  objective: SignModel["objective"]): SignModel => {
  const vectors = rows.map(row => signVector(featureSet, row)), width = vectors[0].length;
  const means = Array.from({ length: width }, (_, feature) => vectors.reduce((sum, values) =>
    sum + values[feature], 0) / vectors.length);
  const scales = means.map((mean, feature) => Math.max(1e-8, Math.sqrt(vectors.reduce((sum, values) =>
    sum + (values[feature] - mean) ** 2, 0) / vectors.length)));
  const normalized = vectors.map(values => values.map((value, feature) => Math.max(-5, Math.min(5,
    (value - means[feature]) / scales[feature]))));
  const weights = rows.map(row => objective === "return-weighted" ? Math.abs(row.return) : 1);
  const weightMean = weights.reduce((sum, value) => sum + value, 0) / weights.length;
  weights.forEach((value, index) => weights[index] = value / weightMean);
  const positive = rows.reduce((sum, row, index) => sum + weights[index] * Number(row.return > 0), .5);
  const total = weights.reduce((sum, value) => sum + value, 1), prior = positive / total;
  const coefficients = new Array<number>(width).fill(0); let intercept = Math.log(prior / (1 - prior));
  for (let iteration = 0; iteration < 40; iteration++) {
    const dimension = width + 1, system = Array.from({ length: dimension }, () => new Array<number>(dimension + 1).fill(0));
    let largestGradient = 0;
    for (let row = 0; row < rows.length; row++) {
      const x = [1, ...normalized[row]], y = Number(rows[row].return > 0);
      const p = sigmoid(intercept + coefficients.reduce((sum, coefficient, feature) => sum + coefficient * normalized[row][feature], 0));
      const gradient = weights[row] * (p - y), curvature = weights[row] * p * (1 - p);
      for (let i = 0; i < dimension; i++) {
        system[i][dimension] += x[i] * gradient;
        for (let j = 0; j < dimension; j++) system[i][j] += curvature * x[i] * x[j];
      }
    }
    for (let feature = 0; feature < width; feature++) {
      system[feature + 1][feature + 1] += penalty * rows.length;
      system[feature + 1][width + 1] += penalty * rows.length * coefficients[feature];
    }
    system[0][0] += 1e-10;
    for (const row of system) largestGradient = Math.max(largestGradient, Math.abs(row[width + 1]));
    if (largestGradient / rows.length < 1e-9) break;
    const step = solve(system); intercept -= step[0];
    coefficients.forEach((value, feature) => coefficients[feature] = value - step[feature + 1]);
  }
  return { featureSet, penalty, objective, means, scales, intercept, coefficients };
};
const signProbability = (head: SignModel, row: SequenceRow) => {
  const values = signVector(head.featureSet, row);
  return sigmoid(head.intercept + head.coefficients.reduce((sum, coefficient, feature) => sum
    + coefficient * Math.max(-5, Math.min(5, (values[feature] - head.means[feature]) / head.scales[feature])), 0));
};
const combinedMean = (head: SignModel, blend: number, row: SequenceRow) => {
  const stats = rowSigns(row), probability = (1 - blend) * stats.probability + blend * signProbability(head, row);
  return stats.active * (probability * stats.positiveMean + (1 - probability) * stats.negativeMean);
};
const probabilityMetrics = (rows: readonly SequenceRow[], head: SignModel, blend: number) => {
  let loss = 0, baseLoss = 0, correct = 0, weightedCorrect = 0, magnitude = 0;
  for (const row of rows) {
    const stats = rowSigns(row), p = Math.max(1e-6, Math.min(1 - 1e-6,
      (1 - blend) * stats.probability + blend * signProbability(head, row)));
    const p0 = Math.max(1e-6, Math.min(1 - 1e-6, stats.probability)), y = Number(row.return > 0);
    loss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
    baseLoss -= y * Math.log(p0) + (1 - y) * Math.log(1 - p0);
    const hit = (p >= .5) === Boolean(y); correct += Number(hit);
    weightedCorrect += Number(hit) * Math.abs(row.return); magnitude += Math.abs(row.return);
  }
  return { signLogLoss: loss / rows.length, baseSignLogLoss: baseLoss / rows.length,
    signInformationBits: (baseLoss - loss) / rows.length / Math.log(2), directionAccuracy: correct / rows.length,
    magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude };
};
const signSpecifications = (Object.keys(featureSets) as FeatureSet[]).flatMap(featureSet =>
  (["ordinary", "return-weighted"] as const).flatMap(objective => [.1, 1, 10, 100].map(penalty => ({ featureSet, objective, penalty }))));
const signHeads = signSpecifications.map(specification => ({ specification,
  head: fitSign(trainRows, specification.featureSet, specification.penalty, specification.objective) }));
const signCandidates = signHeads.flatMap(({ specification, head }) => [.5, 1].map(blend => ({ specification: { ...specification, blend }, head,
  training: { ...metrics(trainRows, row => combinedMean(head, blend, row)), ...probabilityMetrics(trainRows, head, blend) },
  selection: { ...metrics(selectionRows, row => combinedMean(head, blend, row)), ...probabilityMetrics(selectionRows, head, blend) } })));
const eligibleSign = signCandidates.filter(candidate => candidate.selection.mseImprovementVsBase > 0
  && candidate.selection.mseSkillVsZero > 0 && candidate.selection.signInformationBits > 0
  && candidate.selection.positiveDays / candidate.selection.days >= .6)
  .sort((left, right) => right.selection.mseImprovementVsBase - left.selection.mseImprovementVsBase);
const selectedSign = eligibleSign[0];
const signValidation = selectedSign ? { ...metrics(validationRows, row => combinedMean(selectedSign.head,
  selectedSign.specification.blend, row)), ...probabilityMetrics(validationRows, selectedSign.head,
  selectedSign.specification.blend) } : null;
const signRadiusBps = selectedSign ? eventConformalResidualRadius(validationRows.map(row => ({
  predicted: combinedMean(selectedSign.head, selectedSign.specification.blend, row) * 10_000,
  realized: row.return * 10_000 })), .75) : null;
const signUncertainty = selectedSign && Number.isFinite(signRadiusBps) ? validationRows.map(row => {
  const meanBps = combinedMean(selectedSign.head, selectedSign.specification.blend, row) * 10_000;
  const lowBps = meanBps - signRadiusBps!, highBps = meanBps + signRadiusBps!;
  return { at: row.at, meanBps, lowBps, highBps, confidence: eventIntervalConfidence(meanBps, lowBps, highBps),
    robustMarginBps: Math.max(0, Math.abs(meanBps) - signRadiusBps!) };
}) : [];
const signGate = Boolean(selectedSign && signValidation && signValidation.mseImprovementVsBase > 0
  && signValidation.mseSkillVsZero > 0 && signValidation.signInformationBits > 0
  && signValidation.positiveDays / signValidation.days >= .6 && signUncertainty.some(row => row.robustMarginBps > 24));

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-history-mean-screen-v1", source, sourceModelHash: hash(modelBytes),
  fitStart: config.fitStart, fitEnd: config.fitEnd, calibrationStart: config.calibrationStart,
  calibrationEnd: config.calibrationEnd, featureSets: Object.fromEntries(Object.entries(featureSets).map(([name]) =>
    [name, name === "base-only" ? ["base-mean"] : name === "base-history"
      ? ["base-mean", ...EVENT_COMPLETED_HISTORY_INPUTS]
      : ["base-mean", ...compactNames, ...EVENT_COMPLETED_HISTORY_INPUTS]])), specifications,
  selection: "Fit every fixed ridge mean and logistic sign specification on the complete non-overlapping fit-event chain. Select each family independently on the first chronological half of the seven-day pre-test calibration chain by positive skill versus zero and improvement versus the frozen base, requiring at least 60% positive UTC-day blocks; sign candidates must also add sign information. Evaluate only the selected candidate on the untouched second half. Promotion additionally requires the same validation gates and at least one 75% split-conformal robust mean exceeding the 24 bp round-trip cost. No inspector-window candle is loaded.",
  method: "Sequence features contain only prior completed event returns and durations, reset across gaps and bounded to sixteen events/one day. Current-event outcomes never enter their own features. The frozen base mean and a small documented static context are optional inputs. The separate sign family combines its probability with the frozen leaf-specific positive and negative magnitudes and active mass." });
save("summary.json", { trainingEvents: trainRows.length, calibrationEvents: calibrationRows.length,
  selectionEvents: selectionRows.length, validationEvents: validationRows.length,
  base: { training: metrics(trainRows, row => row.baseMean), selection: metrics(selectionRows, row => row.baseMean),
    validation: metrics(validationRows, row => row.baseMean) },
  candidates: candidates.map(({ ridge: _ridge, ...candidate }) => candidate), eligibleCandidates: eligible.length,
  selected: selected ? { specification: selected.specification, ridge: selected.ridge,
    training: selected.training, selection: selected.selection, validation,
    conformalCoverage: .75, residualRadiusBps, uncertainty: {
      directional: uncertainty.filter(row => row.confidence > 0).length,
      aboveOneWayCost: uncertainty.filter(row => row.robustMarginBps > 12).length,
      aboveRoundTripCost: uncertainty.filter(row => row.robustMarginBps > 24).length,
      maximumRobustMarginBps: Math.max(0, ...uncertainty.map(row => row.robustMarginBps)) },
    promote: gate } : null,
  signMagnitude: { candidates: signCandidates.map(({ head: _head, ...candidate }) => candidate),
    eligibleCandidates: eligibleSign.length, selected: selectedSign ? { specification: selectedSign.specification,
      head: selectedSign.head, training: selectedSign.training, selection: selectedSign.selection,
      validation: signValidation, conformalCoverage: .75, residualRadiusBps: signRadiusBps,
      uncertainty: { directional: signUncertainty.filter(row => row.confidence > 0).length,
        aboveOneWayCost: signUncertainty.filter(row => row.robustMarginBps > 12).length,
        aboveRoundTripCost: signUncertainty.filter(row => row.robustMarginBps > 24).length,
        maximumRobustMarginBps: Math.max(0, ...signUncertainty.map(row => row.robustMarginBps)) },
      promote: signGate } : null } });
save("validation-uncertainty.json", uncertainty);
save("sign-validation-uncertainty.json", signUncertainty);
save("sources.json", Object.fromEntries(["scripts/screen-native-event-history-mean.ts",
  "packages/bot-algo/src/event-completed-history.ts", "packages/bot-algo/src/event-uncertainty.ts",
  "scripts/research-event-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
console.log(JSON.stringify({ trainingEvents: trainRows.length, calibrationEvents: calibrationRows.length,
  eligibleCandidates: eligible.length, selected: selected ? { specification: selected.specification,
    selection: selected.selection, validation, residualRadiusBps,
    uncertainty: { directional: uncertainty.filter(row => row.confidence > 0).length,
      aboveRoundTripCost: uncertainty.filter(row => row.robustMarginBps > 24).length,
      maximumRobustMarginBps: Math.max(0, ...uncertainty.map(row => row.robustMarginBps)) }, promote: gate } : null,
  signMagnitude: { eligibleCandidates: eligibleSign.length, selected: selectedSign ? { specification: selectedSign.specification,
    selection: selectedSign.selection, validation: signValidation, residualRadiusBps: signRadiusBps,
    uncertainty: { directional: signUncertainty.filter(row => row.confidence > 0).length,
      aboveRoundTripCost: signUncertainty.filter(row => row.robustMarginBps > 24).length,
      maximumRobustMarginBps: Math.max(0, ...signUncertainty.map(row => row.robustMarginBps)) }, promote: signGate } : null } }));
