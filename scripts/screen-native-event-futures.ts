/** Calibration-only incremental futures sign screen for a native event law. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventConformalResidualRadius, eventIntervalConfidence } from "../packages/bot-algo/src/event-uncertainty.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { EVENT_NATIVE_FUTURES_ACTIVITY_INPUTS, EVENT_NATIVE_FUTURES_CENTERED_INPUTS,
  EVENT_NATIVE_FUTURES_FLOW_INPUTS, eventNativeSecondFuturesFeatures, loadEventFuturesRows,
  type NativeSecondEventFuturesFeatures } from "./event-futures-basis.js";
import { loadNativeEventCandles, makeSamples } from "./research-event-policy.js";

const arg = (key: string) => { const index = process.argv.indexOf(`--${key}`); return index < 0 ? "" : process.argv[index + 1]; };
const root = path.resolve(__dirname, ".."), benchmark = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = benchmark(arg("source")), output = benchmark(arg("output"));
assert.ok(arg("source") && arg("output") && fs.existsSync(source) && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const hash = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
const config = read("config.json"), modelBytes = fs.readFileSync(path.join(source, "model.json"));
const policy = restoreEventPolicy(JSON.parse(modelBytes.toString())), model = policy.model;
assert.equal(config.contract, "native-second-event-screen-v2");
assert.equal(config.scoredTestLoaded, false); assert.equal(model.clock.candleIntervalMs, 1_000);
for (const reference of config.sourceReferences) assert.equal(hash(fs.readFileSync(reference.file)), reference.sha256);

const candles = loadNativeEventCandles(config.sourceRanges, usesNativeSecondTradeFlowFeatures(model.featureNames));
const external = loadEventFuturesRows(config.sourceRanges[0].start, config.calibrationEnd);
assert.deepEqual(external.missing, []);
const cache = new Map<number, NativeSecondEventFuturesFeatures | null>();
const futuresAt = (index: number) => {
  if (!cache.has(index)) cache.set(index, eventNativeSecondFuturesFeatures(candles, index, time => external.rows.get(time)));
  return cache.get(index)!;
};
const chain = (start: number, end: number) => makeSamples(candles, model.clock, start, end,
  config.excluded, 1, "chain", model.featureNames);
const training = chain(config.fitStart, config.fitEnd), calibration = chain(config.calibrationStart, config.calibrationEnd);
assert.ok(training.length >= 40 && calibration.length >= 80 && training.every(row => row.end < calibration[0].start));

type Row = MoveSample & { at: number; baseMean: number; baseProbability: number; external: NativeSecondEventFuturesFeatures };
const leafSigns = model.kernels.map(kernel => {
  const positive = kernel.filter(atom => atom.return > 0), negative = kernel.filter(atom => atom.return < 0);
  const positiveMass = positive.reduce((sum, atom) => sum + atom.probability, 0);
  const negativeMass = negative.reduce((sum, atom) => sum + atom.probability, 0), active = positiveMass + negativeMass;
  assert.ok(active > 0 && positiveMass > 0 && negativeMass > 0);
  return { active, probability: positiveMass / active,
    positiveMean: positive.reduce((sum, atom) => sum + atom.probability * atom.return, 0) / positiveMass,
    negativeMean: negative.reduce((sum, atom) => sum + atom.probability * atom.return, 0) / negativeMass };
});
const prepare = (rows: readonly MoveSample[]): Row[] => rows.map(row => {
  const external = futuresAt(row.start); assert.ok(external, `Missing causal futures state at ${candles[row.start].openTime}`);
  const stats = leafSigns[eventLeaf(model, row.features)];
  return { ...row, at: candles[row.start].openTime + 1_000, external,
    baseMean: stats.active * (stats.probability * stats.positiveMean + (1 - stats.probability) * stats.negativeMean),
    baseProbability: stats.probability };
});
const trainRows = prepare(training), calibrationRows = prepare(calibration);
const splitAt = Math.floor(calibrationRows.length / 2), selectionRows = calibrationRows.slice(0, splitAt);
const validationRows = calibrationRows.slice(splitAt);
assert.ok(selectionRows.length >= 40 && validationRows.length >= 40
  && selectionRows.every(row => row.end <= validationRows[0].start));

const featureSets = {
  "base-only": (row: Row) => [row.baseProbability],
  activity: (row: Row) => [row.baseProbability, row.external.flow[0], row.external.rangeBps],
  centered: (row: Row) => [row.baseProbability, ...row.external.price.slice(1), ...row.external.deviations],
  "activity-centered": (row: Row) => [row.baseProbability, row.external.flow[0], row.external.rangeBps,
    ...row.external.price.slice(1), ...row.external.deviations],
  all: (row: Row) => [row.baseProbability, row.external.flow[0], row.external.rangeBps,
    ...row.external.price.slice(1), ...row.external.deviations, ...row.external.flow.slice(1)],
};
type FeatureSet = keyof typeof featureSets;
interface SignModel { featureSet: FeatureSet; penalty: number; objective: "ordinary" | "return-weighted";
  means: number[]; scales: number[]; intercept: number; coefficients: number[]; }
const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
const vector = (featureSet: FeatureSet, row: Row) => {
  const values = featureSets[featureSet](row);
  values[0] = Math.log(Math.max(1e-6, Math.min(1 - 1e-6, values[0]))
    / (1 - Math.max(1e-6, Math.min(1 - 1e-6, values[0]))));
  return values;
};
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
const fit = (rows: readonly Row[], featureSet: FeatureSet, penalty: number, objective: SignModel["objective"]): SignModel => {
  const vectors = rows.map(row => vector(featureSet, row)), width = vectors[0].length;
  const means = Array.from({ length: width }, (_, feature) => vectors.reduce((sum, values) => sum + values[feature], 0) / vectors.length);
  const scales = means.map((mean, feature) => Math.max(1e-8, Math.sqrt(vectors.reduce((sum, values) =>
    sum + (values[feature] - mean) ** 2, 0) / vectors.length)));
  const normalized = vectors.map(values => values.map((value, feature) => Math.max(-5, Math.min(5,
    (value - means[feature]) / scales[feature]))));
  const weights = rows.map(row => objective === "return-weighted" ? Math.abs(row.return) : 1);
  const weightMean = weights.reduce((sum, value) => sum + value, 0) / weights.length;
  weights.forEach((value, index) => weights[index] = value / weightMean);
  const positive = rows.reduce((sum, row, index) => sum + weights[index] * Number(row.return > 0), .5);
  const total = weights.reduce((sum, value) => sum + value, 1), coefficients = new Array<number>(width).fill(0);
  let intercept = Math.log(positive / (total - positive));
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
const probability = (head: SignModel, row: Row) => {
  const values = vector(head.featureSet, row);
  return sigmoid(head.intercept + head.coefficients.reduce((sum, coefficient, feature) => sum
    + coefficient * Math.max(-5, Math.min(5, (values[feature] - head.means[feature]) / head.scales[feature])), 0));
};
const blendedProbability = (head: SignModel, blend: number, row: Row) => (1 - blend) * row.baseProbability + blend * probability(head, row);
const mean = (head: SignModel, blend: number, row: Row) => {
  const stats = leafSigns[eventLeaf(model, row.features)], p = blendedProbability(head, blend, row);
  return stats.active * (p * stats.positiveMean + (1 - p) * stats.negativeMean);
};
const compare = (rows: readonly Row[], head: SignModel, blend: number, matched?: { head: SignModel; blend: number }) => {
  let error = 0, zero = 0, baseError = 0, matchedError = 0, loss = 0, baseLoss = 0, matchedLoss = 0;
  let correct = 0, weightedCorrect = 0, magnitude = 0, maximum = 0;
  const days = new Map<number, number>();
  for (const row of rows) {
    const value = mean(head, blend, row), p = Math.max(1e-6, Math.min(1 - 1e-6, blendedProbability(head, blend, row)));
    const matchedValue = matched ? mean(matched.head, matched.blend, row) : row.baseMean;
    const matchedP = matched ? Math.max(1e-6, Math.min(1 - 1e-6, blendedProbability(matched.head, matched.blend, row))) : row.baseProbability;
    const y = Number(row.return > 0), hit = (p >= .5) === Boolean(y);
    error += (row.return - value) ** 2; zero += row.return ** 2; baseError += (row.return - row.baseMean) ** 2;
    matchedError += (row.return - matchedValue) ** 2;
    loss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
    baseLoss -= y * Math.log(row.baseProbability) + (1 - y) * Math.log(1 - row.baseProbability);
    matchedLoss -= y * Math.log(matchedP) + (1 - y) * Math.log(1 - matchedP);
    correct += Number(hit); weightedCorrect += Number(hit) * Math.abs(row.return); magnitude += Math.abs(row.return);
    maximum = Math.max(maximum, Math.abs(value) * 1e4);
    const day = Math.floor(row.at / 86_400_000);
    days.set(day, (days.get(day) ?? 0) + (row.return - matchedValue) ** 2 - (row.return - value) ** 2);
  }
  return { rows: rows.length, mseSkillVsZero: 1 - error / zero, mseImprovementVsBaseLaw: 1 - error / baseError,
    mseImprovementVsMatchedBaseOnly: 1 - error / matchedError, signInformationBitsVsBaseLaw: (baseLoss - loss) / rows.length / Math.log(2),
    signInformationBitsVsMatchedBaseOnly: (matchedLoss - loss) / rows.length / Math.log(2),
    directionAccuracy: correct / rows.length, magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude,
    maximumAbsoluteMeanBps: maximum, days: days.size, positiveDaysVsMatchedBaseOnly: [...days.values()].filter(value => value > 0).length };
};

const specifications = (["activity", "centered", "activity-centered", "all"] as FeatureSet[]).flatMap(featureSet =>
  (["ordinary", "return-weighted"] as const).flatMap(objective => [.1, 1, 10, 100].flatMap(penalty => [.5, 1].map(blend =>
    ({ featureSet, objective, penalty, blend })) )));
const baselineHeads = new Map<string, SignModel>();
for (const objective of ["ordinary", "return-weighted"] as const) for (const penalty of [.1, 1, 10, 100])
  baselineHeads.set(`${objective}:${penalty}`, fit(trainRows, "base-only", penalty, objective));
const candidates = specifications.map(specification => {
  const head = fit(trainRows, specification.featureSet, specification.penalty, specification.objective);
  const baseline = baselineHeads.get(`${specification.objective}:${specification.penalty}`)!;
  const matched = { head: baseline, blend: specification.blend };
  return { specification, head, baseline, training: compare(trainRows, head, specification.blend, matched),
    selection: compare(selectionRows, head, specification.blend, matched) };
});
const eligible = candidates.filter(candidate => candidate.selection.mseSkillVsZero > 0
  && candidate.selection.mseImprovementVsBaseLaw > 0 && candidate.selection.mseImprovementVsMatchedBaseOnly > 0
  && candidate.selection.signInformationBitsVsBaseLaw > 0 && candidate.selection.signInformationBitsVsMatchedBaseOnly > 0
  && candidate.selection.positiveDaysVsMatchedBaseOnly / candidate.selection.days >= .6)
  .sort((left, right) => right.selection.mseImprovementVsMatchedBaseOnly - left.selection.mseImprovementVsMatchedBaseOnly);
const selected = eligible[0];
const validation = selected ? compare(validationRows, selected.head, selected.specification.blend,
  { head: selected.baseline, blend: selected.specification.blend }) : null;
const radius = selected ? eventConformalResidualRadius(validationRows.map(row => ({
  predicted: mean(selected.head, selected.specification.blend, row) * 1e4, realized: row.return * 1e4 })), .75) : null;
const intervals = selected && Number.isFinite(radius) ? validationRows.map(row => {
  const meanBps = mean(selected.head, selected.specification.blend, row) * 1e4;
  const lowBps = meanBps - radius!, highBps = meanBps + radius!;
  return { at: row.at, meanBps, lowBps, highBps,
    confidence: eventIntervalConfidence(meanBps, lowBps, highBps), robustMarginBps: Math.max(0, Math.abs(meanBps) - radius!) };
}) : [];
const promote = Boolean(selected && validation && validation.mseSkillVsZero > 0
  && validation.mseImprovementVsBaseLaw > 0 && validation.mseImprovementVsMatchedBaseOnly > 0
  && validation.signInformationBitsVsBaseLaw > 0 && validation.signInformationBitsVsMatchedBaseOnly > 0
  && validation.positiveDaysVsMatchedBaseOnly / validation.days >= .6
  && intervals.some(row => row.robustMarginBps > 24));

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
const names = { "base-only": ["base-sign-logit"], activity: ["base-sign-logit", ...EVENT_NATIVE_FUTURES_ACTIVITY_INPUTS],
  centered: ["base-sign-logit", ...EVENT_NATIVE_FUTURES_CENTERED_INPUTS],
  "activity-centered": ["base-sign-logit", ...EVENT_NATIVE_FUTURES_ACTIVITY_INPUTS, ...EVENT_NATIVE_FUTURES_CENTERED_INPUTS],
  all: ["base-sign-logit", ...EVENT_NATIVE_FUTURES_ACTIVITY_INPUTS, ...EVENT_NATIVE_FUTURES_CENTERED_INPUTS,
    ...EVENT_NATIVE_FUTURES_FLOW_INPUTS] };
save("config.json", { contract: "native-event-futures-incremental-sign-screen-v1", source,
  sourceModelHash: hash(modelBytes), fitStart: config.fitStart, fitEnd: config.fitEnd,
  calibrationStart: config.calibrationStart, calibrationEnd: config.calibrationEnd,
  featureSets: names, specifications,
  selection: "Fit fixed external and matched base-only heads on the non-overlapping fit chain. Select on the first chronological calibration half only when the external head improves MSE and sign information over both the frozen law and its matched base-only head, with at least 60% positive UTC days. Evaluate that one choice on the untouched second half. Promotion also requires all validation gates and a 75% residual interval with more than 24 bp robust mean. No inspector-window candle is loaded.",
  availability: "Each input uses the latest fully completed futures minute and matching spot minute close. The unfinished minute and future rows are excluded.",
  futuresReferences: external.references, futuresFingerprint: external.fingerprint, futuresMissingDays: external.missing });
save("summary.json", { trainingEvents: trainRows.length, calibrationEvents: calibrationRows.length,
  selectionEvents: selectionRows.length, validationEvents: validationRows.length,
  candidates: candidates.map(({ head: _head, baseline: _baseline, ...candidate }) => candidate),
  eligibleCandidates: eligible.length, selected: selected ? { specification: selected.specification,
    head: selected.head, matchedBaseOnly: selected.baseline, training: selected.training, selection: selected.selection,
    validation, conformalCoverage: .75, residualRadiusBps: radius,
    uncertainty: { directional: intervals.filter(row => row.confidence > 0).length,
      aboveRoundTripCost: intervals.filter(row => row.robustMarginBps > 24).length,
      maximumRobustMarginBps: Math.max(0, ...intervals.map(row => row.robustMarginBps)) }, promote } : null });
save("validation-uncertainty.json", intervals);
save("sources.json", Object.fromEntries(["scripts/screen-native-event-futures.ts", "scripts/event-futures-basis.ts",
  "packages/bot-algo/src/event-uncertainty.ts", "packages/bot-algo/src/event-distribution.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
console.log(JSON.stringify({ trainingEvents: trainRows.length, calibrationEvents: calibrationRows.length,
  eligibleCandidates: eligible.length, selected: selected ? { specification: selected.specification,
    selection: selected.selection, validation, residualRadiusBps: radius,
    uncertainty: { directional: intervals.filter(row => row.confidence > 0).length,
      aboveRoundTripCost: intervals.filter(row => row.robustMarginBps > 24).length,
      maximumRobustMarginBps: Math.max(0, ...intervals.map(row => row.robustMarginBps)) }, promote } : null }));
