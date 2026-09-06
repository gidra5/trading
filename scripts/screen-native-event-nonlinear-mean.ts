/** Calibration-only nonlinear mean head for cost-sized native events. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { eventBoostScore, eventFeatureWarmup, eventLeaf, trainEventBoost,
  type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { NATIVE_SECOND_SELECTED_SIGN_FEATURES,
  usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { loadNativeEventCandles, makeSamples } from "./research-event-policy.js";

const arg = (key: string) => { const index = process.argv.indexOf(`--${key}`); return index < 0 ? "" : process.argv[index + 1]; };
const root = path.resolve(__dirname, "..");
const benchmark = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = benchmark(arg("source")), output = benchmark(arg("output"));
assert.ok(arg("source") && arg("output") && fs.existsSync(source) && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const hash = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
const config = read("config.json"), modelBytes = fs.readFileSync(path.join(source, "model.json"));
const { model, costs } = restoreEventPolicy(JSON.parse(modelBytes.toString()));
assert.equal(model.clock.candleIntervalMs, 1000);
assert.ok(config.fitStart < config.fitEnd && config.calibrationStart < config.calibrationEnd);
for (const reference of config.sourceReferences) assert.equal(hash(fs.readFileSync(reference.file)), reference.sha256);

const DAY = 86_400_000, featureNames = NATIVE_SECOND_SELECTED_SIGN_FEATURES;
const scoreTest = process.argv.includes("--score-test");
if (scoreTest) assert.ok(config.start < config.end);
const partitionEnd = config.fitEnd - DAY;
const warmup = Math.max(config.warmupCandles, eventFeatureWarmup(model.clock, featureNames));
const ranges = [{ start: config.fitStart - (warmup + 1) * 1000, end: scoreTest ? config.end : config.calibrationEnd }];
const candles = loadNativeEventCandles(ranges, usesNativeSecondTradeFlowFeatures(model.featureNames)
  || usesNativeSecondTradeFlowFeatures(featureNames));
type Row = MoveSample & { headFeatures: number[]; nextHeadFeatures: number[] };
const samples = (start: number, end: number, sampling: "stride" | "chain" = "stride",
  excluded = config.excluded): Row[] => {
  const step = sampling === "stride" ? config.stride : 1;
  const base = makeSamples(candles, model.clock, start, end, excluded, step, sampling, model.featureNames);
  const head = makeSamples(candles, model.clock, start, end, excluded, step, sampling, featureNames);
  assert.equal(base.length, head.length);
  return base.map((row, index) => {
    const other = head[index]!;
    assert.deepEqual([other.start, other.end, other.return, other.duration, other.low, other.high],
      [row.start, row.end, row.return, row.duration, row.low, row.high]);
    return { ...row, headFeatures: other.features, nextHeadFeatures: other.nextFeatures };
  });
};
const partition = samples(config.fitStart, partitionEnd);
const tuning = samples(partitionEnd, config.fitEnd);
const calibration = samples(config.calibrationStart, config.calibrationEnd);
const test = scoreTest ? samples(config.start, config.end, "chain", []) : undefined;
assert.ok(partition.length >= 256 && tuning.length >= 256 && calibration.length >= 256);
assert.ok(partition.every(row => row.end < tuning[0]!.start)
  && tuning.every(row => row.end < calibration[0]!.start));

const baseMean = (row: Row) => model.kernels[eventLeaf(model, row.features)].reduce(
  (sum: number, atom: { probability: number; return: number }) => sum + atom.probability * atom.return, 0);
const fitBlend = (rows: readonly Row[], predictHead: (row: Row) => number) => {
  let numerator = 0, denominator = 0;
  for (const row of rows) {
    const base = baseMean(row), delta = predictHead(row) - base;
    numerator += delta * (row.return - base); denominator += delta * delta;
  }
  return Math.max(0, Math.min(1, numerator / Math.max(denominator, 1e-18)));
};
const metrics = (rows: readonly Row[], predict: (row: Row) => number) => {
  let error = 0, zero = 0, baseError = 0, correct = 0, weightedCorrect = 0, magnitude = 0;
  let mean = 0, maximum = 0, oneWay = 0, roundTrip = 0;
  const oneWayCost = costs.feeBps + costs.slippageBps;
  for (const row of rows) {
    const value = predict(row), actual = row.return, base = baseMean(row), hit = Math.sign(value) === Math.sign(actual);
    assert.ok(Number.isFinite(value));
    error += (actual - value) ** 2; zero += actual ** 2; baseError += (actual - base) ** 2;
    correct += Number(hit); weightedCorrect += Number(hit) * Math.abs(actual); magnitude += Math.abs(actual);
    mean += value; maximum = Math.max(maximum, Math.abs(value) * 10_000);
    oneWay += Number(Math.abs(value) * 10_000 > oneWayCost);
    roundTrip += Number(Math.abs(value) * 10_000 > 2 * oneWayCost);
  }
  return { rows: rows.length, mseSkillVsZero: 1 - error / zero, mseImprovementVsBase: 1 - error / baseError,
    directionAccuracy: correct / rows.length, magnitudeWeightedDirectionAccuracy: weightedCorrect / magnitude,
    predictedMeanBps: mean / rows.length * 10_000, maximumAbsoluteMeanBps: maximum,
    aboveOneWayCost: oneWay, aboveRoundTripCost: roundTrip };
};
const blockStability = (rows: readonly Row[], predict: (row: Row) => number) => {
  const blocks = new Map<number, { improvement: number; rows: number }>();
  for (const row of rows) {
    const key = Math.floor(candles[row.start]!.openTime / 3_600_000);
    const base = baseMean(row), value = predict(row);
    const saved = blocks.get(key) ?? { improvement: 0, rows: 0 };
    saved.improvement += (row.return - base) ** 2 - (row.return - value) ** 2; saved.rows++;
    blocks.set(key, saved);
  }
  const values = [...blocks.values()];
  return { blocks: values.length, positiveBlocks: values.filter(row => row.improvement > 0).length,
    meanImprovement: values.reduce((sum, row) => sum + row.improvement / row.rows, 0) / values.length };
};

type Specification = { iterations: number; rate: number; minLeaf: number };
const specifications: Specification[] = [8, 16, 32].flatMap(iterations => [.03, .1].flatMap(rate =>
  [128, 256].map(minLeaf => ({ iterations, rate, minLeaf }))));
const calibrationBoundary = Math.floor(calibration.length / 2);
const blendRows = calibration.slice(0, calibrationBoundary), selectionRows = calibration.slice(calibrationBoundary);
const started = performance.now();
const candidates = specifications.map(specification => {
  const training = partition.map(row => ({ ...row, features: row.headFeatures, nextFeatures: row.nextHeadFeatures }));
  const fitted = trainEventBoost(training, model.clock, { ...specification, cells: 8, prior: 32, featureNames });
  assert.ok(fitted.boost);
  const raw = (row: Row) => eventBoostScore(fitted.boost!, row.headFeatures);
  const blend = fitBlend(blendRows, raw);
  const predict = (row: Row) => (1 - blend) * baseMean(row) + blend * raw(row);
  return { specification, blend, head: fitted.boost, tuning: metrics(tuning, predict),
    blendFit: metrics(blendRows, predict), selection: metrics(selectionRows, predict),
    stability: blockStability(selectionRows, predict), calibration: metrics(calibration, predict), raw };
});
const eligible = candidates.filter(candidate => candidate.selection.mseImprovementVsBase > 0
  && candidate.stability.positiveBlocks / candidate.stability.blocks >= .6);
eligible.sort((left, right) => right.selection.mseImprovementVsBase - left.selection.mseImprovementVsBase);
const selected = eligible[0];
const refitBlend = selected ? fitBlend(calibration, selected.raw) : undefined;
const refitPredict = selected ? (row: Row) => (1 - refitBlend!) * baseMean(row) + refitBlend! * selected.raw(row) : undefined;

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-nonlinear-mean-screen-v1", source,
  sourceModelHash: hash(modelBytes), featureNames, partitionStart: config.fitStart, partitionEnd,
  tuningStart: partitionEnd, tuningEnd: config.fitEnd,
  calibrationStart: config.calibrationStart, calibrationEnd: config.calibrationEnd,
  ...(scoreTest ? { testStart: config.start, testEnd: config.end } : {}),
  blendFitRows: blendRows.length, selectionRows: selectionRows.length,
  scoredTestLoaded: scoreTest, specifications,
  selection: "Fit shallow boosted-return heads on the partition days. Fit each frozen-base/head blend on the first chronological half of calibration, then select by MSE improvement on the second half while requiring improvement in at least 60% of its one-hour blocks. Refit only the selected blend weight on all calibration rows for a subsequent frozen test.",
  method: `The head predicts the arithmetic return of the same 48 bp barrier/one-hour-timeout event used by the policy. The saved magnitude/duration/extrema/successor law and state map remain unchanged.${scoreTest
    ? " Selection and refitting finish before the inspector prefix is scored once."
    : " No inspector-window candle or label is loaded."}` });
save("summary.json", { partition: partition.length, tuning: tuning.length, calibration: calibration.length,
  ...(test ? { test: test.length } : {}),
  base: { tuning: metrics(tuning, baseMean), blendFit: metrics(blendRows, baseMean),
    selection: metrics(selectionRows, baseMean), calibration: metrics(calibration, baseMean) },
  candidates: candidates.map(({ head: _head, raw: _raw, ...candidate }) => candidate), eligibleCandidates: eligible.length,
  selected: selected ? { specification: selected.specification, blend: selected.blend,
    tuning: selected.tuning, blendFit: selected.blendFit, selection: selected.selection,
    stability: selected.stability, calibration: selected.calibration,
    refitBlend, refitCalibration: metrics(calibration, refitPredict!),
    ...(test ? { test: metrics(test, refitPredict!) } : {}) } : null,
  elapsedSeconds: (performance.now() - started) / 1000 });
if (selected) save("head.json", selected.head);
save("sources.json", Object.fromEntries(["scripts/screen-native-event-nonlinear-mean.ts",
  "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-second-features.ts",
  "scripts/research-event-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
console.log(JSON.stringify({ eligibleCandidates: eligible.length,
  selected: selected ? { specification: selected.specification, blend: selected.blend,
    blendFit: selected.blendFit, selection: selected.selection, stability: selected.stability,
    refitBlend, refitCalibration: metrics(calibration, refitPredict!),
    ...(test ? { test: metrics(test, refitPredict!) } : {}) } : null,
  base: { blendFit: metrics(blendRows, baseMean), selection: metrics(selectionRows, baseMean),
    calibration: metrics(calibration, baseMean) },
  elapsedSeconds: (performance.now() - started) / 1000 }));
