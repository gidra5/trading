/** Compare small honest conditional-mean learners without opening the scored test window. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  distributionMetrics,
  eventFeatureWarmup,
  trainEventDistribution,
  trainEventProjection,
  type EventDistribution,
  type MoveSample,
} from "../packages/bot-algo/src/event-distribution.js";
import { eventMarginalCrps } from "../packages/bot-algo/src/event-crps.js";
import { eventCashHorizon } from "../packages/bot-algo/src/event-cash-horizon.js";
import {
  buildEventPolicy,
  restoreEventPolicy,
  serializeEventPolicy,
} from "../packages/bot-algo/src/event-log-policy.js";
import { loadNativeEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";

const argument = (key: string, fallback = "") => {
  const index = process.argv.indexOf(`--${key}`);
  return index < 0 ? fallback : process.argv[index + 1];
};
const root = path.resolve(__dirname, "..");
const benchmark = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = benchmark(argument("source"));
const output = benchmark(argument("output"));
assert.ok(argument("source") && argument("output") && fs.existsSync(source) && !fs.existsSync(output),
  "Specify an existing --source and a new --output");

const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
const parentConfig = read("config.json");
const parentSummary = read("summary.json");
const parentBytes = fs.readFileSync(path.join(source, "model.json"));
const parent = restoreEventPolicy(JSON.parse(parentBytes.toString()));
assert.equal(parentConfig.contract, "native-second-event-screen-v1");
assert.equal(parent.model.clock.candleIntervalMs, 1000);
assert.equal(parentConfig.estimationMode, "later-fit-stride");
assert.ok(parentSummary.estimation > 0 && parent.model.featureNames.length > 0);
assert.equal(eventFeatureWarmup(parent.model.clock, parent.model.featureNames), parentConfig.warmupCandles);

const hash = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
for (const reference of parentConfig.sourceReferences) {
  assert.equal(hash(fs.readFileSync(reference.file)), reference.sha256, `Changed source ${reference.file}`);
}

const DAY = 86_400_000;
const sourceRanges = [{ start: parentConfig.fitPeriods.sourceStart, end: parentConfig.calibrationEnd }];
const candles = loadNativeEventCandles(sourceRanges, usesNativeSecondTradeFlowFeatures(parent.model.featureNames));
const sample = (start: number, end: number, mode: "stride" | "chain", excluded = parentConfig.excluded): MoveSample[] =>
  makeSamples(candles, parent.model.clock, start, end, excluded,
    mode === "stride" ? parentConfig.stride : 1, mode, parent.model.featureNames);
const partition = sample(parentConfig.fitStart, parentConfig.fitEnd - DAY, "stride");
const estimation = sample(parentConfig.fitEnd - DAY, parentConfig.fitEnd, "stride");
const calibration = sample(parentConfig.calibrationStart, parentConfig.calibrationEnd, "stride");
const calibrationChain = sample(parentConfig.calibrationStart, parentConfig.calibrationEnd, "chain", []);
assert.equal(partition.length, parentSummary.training);
assert.equal(estimation.length, parentSummary.estimation);
assert.equal(calibration.length, parentSummary.calibration);
assert.ok(calibrationChain.length > 0 && partition.every(row => row.end < estimation[0].start));

type Specification =
  | { learner: "tree"; criterion: "distribution" | "mean"; depth: number }
  | { learner: "projection"; penalty: number; cells: number };
const specifications: Specification[] = [
  { learner: "tree", criterion: "distribution", depth: 2 },
  { learner: "tree", criterion: "mean", depth: 1 },
  { learner: "tree", criterion: "mean", depth: 2 },
  { learner: "tree", criterion: "mean", depth: 3 },
  ...[0.01, 0.1, 1, 10].map(penalty => ({ learner: "projection" as const, penalty, cells: 8 })),
];

const fit = (specification: Specification): EventDistribution => {
  if (specification.learner === "tree") return trainEventDistribution(partition, parent.model.clock, {
    maxDepth: specification.depth,
    minLeaf: 128,
    prior: 32,
    criterion: specification.criterion,
    estimationSamples: estimation,
    featureNames: parent.model.featureNames,
  });
  const rows = [...partition, ...estimation];
  const model = trainEventProjection(rows, parent.model.clock, {
    penalty: specification.penalty,
    cells: specification.cells,
    minLeaf: 128,
    prior: 32,
    honestyFraction: estimation.length / rows.length,
    featureNames: parent.model.featureNames,
  });
  assert.equal(model.trainingSamples, estimation.length, "Projection split must preserve the explicit estimation day");
  return model;
};

const summarize = (model: EventDistribution) => {
  const policy = buildEventPolicy(model, parent.costs, {
    depths: 0,
    referenceEquity: 10_000,
    referencePrice: candles[calibrationChain[0].start].close,
  });
  const { trace, positions: _positions, ...replay } = replayEventPolicy(candles, policy,
    parentConfig.calibrationStart, parentConfig.calibrationEnd, 1, { trace: true, oneStepTerminal: "marked" });
  return {
    ...distributionMetrics(model, calibration),
    ...eventMarginalCrps(model, calibration),
    states: model.kernels.length,
    leafCounts: model.counts,
    maximumAbsoluteMeanBps: Math.max(...model.kernels.map(kernel => Math.abs(kernel.reduce(
      (sum, atom) => sum + atom.probability * atom.return, 0)) * 10_000)),
    cashBoundDepth: eventCashHorizon(model, parent.costs, 8).verifiedDepth,
    replay: {
      returnPct: replay.returnPct,
      maxDrawdownPct: replay.maxDrawdownPct,
      trades: replay.trades,
      decisions: replay.decisions,
      nonzeroRequests: trace.filter((row: any) => row.order.request !== 0).length,
    },
  };
};

const started = performance.now();
const fitted = specifications.map(specification => {
  const model = fit(specification);
  return { specification, model, metrics: summarize(model) };
});
assert.deepEqual(fitted[0].model, parent.model, "Distribution-tree control must reproduce the source model");

const unconditional = trainEventDistribution(partition, parent.model.clock, {
  maxDepth: 0,
  minLeaf: 128,
  prior: 32,
  criterion: "distribution",
  estimationSamples: estimation,
  featureNames: parent.model.featureNames,
});
const unconditionalMetrics = summarize(unconditional);
const eligible = fitted.filter(candidate => candidate.metrics.mseSkill > 0
  && candidate.metrics.nll < unconditionalMetrics.nll
  && candidate.metrics.returnCrpsBps < unconditionalMetrics.returnCrpsBps);
eligible.sort((a, b) => b.metrics.mseSkill - a.metrics.mseSkill);
const selected = eligible[0];

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
const publicCandidate = ({ specification, metrics }: typeof fitted[number]) => ({ specification, metrics });
save("summary.json", {
  contract: "native-event-honest-mean-learner-screen-v1",
  source,
  sourceModelHash: hash(parentBytes),
  scoredTestLoaded: false,
  split: {
    partition: partition.length,
    estimation: estimation.length,
    calibration: calibration.length,
    calibrationChain: calibrationChain.length,
    partitionStart: parentConfig.fitStart,
    partitionEnd: parentConfig.fitEnd - DAY,
    estimationStart: parentConfig.fitEnd - DAY,
    estimationEnd: parentConfig.fitEnd,
    calibrationStart: parentConfig.calibrationStart,
    calibrationEnd: parentConfig.calibrationEnd,
  },
  selection: "Require positive calibration return-MSE skill and lower calibration class NLL and return CRPS than the honest unconditional law; then maximize calibration return-MSE skill.",
  unconditional: unconditionalMetrics,
  candidates: fitted.map(publicCandidate),
  eligibleCandidates: eligible.length,
  selected: selected ? publicCandidate(selected) : null,
  elapsedSeconds: (performance.now() - started) / 1000,
  scope: "All candidates are fit before the diagnostic calibration day. No candle or label from the scored inspector window is loaded. Calibration replay is a model-selection diagnostic, not an out-of-sample strategy result.",
});
if (selected) save("selected-model.json", serializeEventPolicy({ ...parent, model: selected.model }));
save("sources.json", Object.fromEntries([
  "scripts/screen-native-event-mean-learners.ts",
  "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-distribution.ts",
  "packages/bot-algo/src/event-crps.ts",
  "packages/bot-algo/src/event-cash-horizon.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
console.log(JSON.stringify({
  eligibleCandidates: eligible.length,
  selected: selected ? publicCandidate(selected) : null,
  unconditional: unconditionalMetrics,
  elapsedSeconds: (performance.now() - started) / 1000,
}));
