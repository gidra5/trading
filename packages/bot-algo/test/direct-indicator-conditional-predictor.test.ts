import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentPolicyMatrix,
  createExposureValueDistillationAccumulator,
  decodeDirectIndicatorConditionalParameters,
  DEFAULT_DIRECT_INDICATOR_PARAMETERS,
  exposureConditionalProbabilityStatistics,
  predictDirectIndicatorConditionalDistribution,
  finalizeExposureValueDistillation,
  observeExposureConditionalProbabilityDistillation,
  observeExposureFourSegmentProbabilityDistillation,
  prepareDirectIndicatorConditionalPredictor,
  prepareExposureValueOracle,
} from "../src/index.js";

const actionGrid = Float64Array.from({ length: 31 }, (_, index) => -100 + index * 200 / 30);
const latentGrid = Float64Array.from({ length: 31 }, (_, index) => -250 + index * 500 / 30);
const execution = {
  friction: 0.00175,
  minExposure: -100,
  maxExposure: 100,
  maxEffectiveExposure: 250,
  quoteBorrowRate: 0,
  assetBorrowRate: 0,
};
const options = {
  intervalMs: 60_000,
  holdingPeriodSteps: 1,
  valueHorizonSteps: 60,
  temperature: 0.01,
  execution,
};
const state = {
  drift: 0.00012,
  variance: 8e-7,
  longRunVariance: 1.1e-6,
};

test("direct indicator decoder applies the analytic coefficients and yields valid parameters", () => {
  const decoded = decodeDirectIndicatorConditionalParameters(
    actionGrid,
    latentGrid,
    state,
    DEFAULT_DIRECT_INDICATOR_PARAMETERS,
    options,
  );
  const parameters = decoded.parameters;
  const metadata = decoded.metadata;
  const [s0, s1, s2] = metadata.backgroundSecantSlopes;
  assert.ok(parameters.latentLower < parameters.c1);
  assert.ok(parameters.c1 < parameters.c2);
  assert.ok(parameters.c2 < parameters.latentUpper);
  assert.ok(parameters.kappaC1 > 0);
  assert.ok(parameters.kappaX > 0);
  assert.ok(parameters.kappaC2 > 0);
  assert.ok(Math.abs(parameters.kappaC1 * metadata.transitionWidths10To90[0] - 4.394) < 1e-10);
  assert.ok(Math.abs(parameters.kappaX * metadata.transitionWidths10To90[1] - 4.394) < 1e-10);
  assert.ok(Math.abs(parameters.kappaC2 * metadata.transitionWidths10To90[2] - 4.394) < 1e-10);
  assert.equal(parameters.cutoffLower, parameters.latentLower);
  assert.equal(parameters.cutoffUpper, parameters.latentUpper);
  assert.equal(decoded.backgroundValues.length, latentGrid.length);
  assert.ok(Number.isFinite(parameters.baseSlope));
  assert.ok(Number.isFinite(parameters.quadraticPrecision));
  assert.ok(Number.isFinite(parameters.betaC1));
  assert.ok(Number.isFinite(parameters.betaC2));
  assert.ok(Math.abs(parameters.betaX
    + (metadata.effectiveBuyCostSlope + metadata.effectiveSellCostSlope)
      / options.temperature) < 1e-12);
  assert.equal(Object.keys(metadata.parameterVector6).length, 6);
});

test("direct indicator conditional rows normalize throughout latent current support", () => {
  const prediction = predictDirectIndicatorConditionalDistribution(
    actionGrid,
    latentGrid,
    0,
    state,
    DEFAULT_DIRECT_INDICATOR_PARAMETERS,
    options,
  );
  for (const current of latentGrid) {
    const probabilities = conditionalFourSegmentExposureProbabilities(
      actionGrid,
      current,
      prediction.conditionalParameters,
    );
    const total = probabilities.reduce((sum, probability) => sum + probability, 0);
    assert.ok(Math.abs(total - 1) < 1e-12, { current, total });
    assert.ok(probabilities.every((probability) => Number.isFinite(probability) && probability >= 0));
  }
});

test("prepared direct predictions match the standalone forecast exactly", () => {
  const prepared = prepareDirectIndicatorConditionalPredictor(
    actionGrid,
    latentGrid,
    DEFAULT_DIRECT_INDICATOR_PARAMETERS,
    options,
  );
  const expected = predictDirectIndicatorConditionalDistribution(
    actionGrid,
    latentGrid,
    -17,
    state,
    DEFAULT_DIRECT_INDICATOR_PARAMETERS,
    options,
  );
  const actual = prepared.predict(-17, state);
  assert.ok(Math.abs(actual.meanExposure - expected.meanExposure) < 1e-12);
  assert.ok(actual.probabilities.every((value, index) =>
    Math.abs(value - expected.probabilities[index]!) < 1e-13));
  assert.ok(actual.backgroundValues.every((value, index) =>
    Math.abs(value - expected.backgroundValues[index]!) < 1e-13));
});

test("direct runtime source has no regret fitter or two-dimensional regret path", () => {
  const source = readFileSync(
    new URL("../src/direct-indicator-conditional-predictor.ts", import.meta.url),
    "utf8",
  );
  assert.equal(source.includes("fitConditionalFourSegmentRegret"), false);
  assert.equal(source.includes("fitConditionalFourSegmentPolicy"), false);
  assert.equal(source.includes("predictHandcraftedIndicatorRegret"), false);
});

test("direct conditional policy uses the shared distillation metrics without a second fee pass", () => {
  const prices = Float64Array.from({ length: 62 }, (_, index) => 100 * Math.exp(index * 0.0001));
  const oracle = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 60,
    friction: execution.friction,
    gridSize: actionGrid.length,
    minExposure: execution.minExposure,
    maxExposure: execution.maxExposure,
    maxEffectiveExposure: execution.maxEffectiveExposure,
    temperature: options.temperature,
    includeProbabilities: true,
  });
  const prediction = predictDirectIndicatorConditionalDistribution(
    oracle.grid,
    oracle.currentGrid,
    0,
    state,
    DEFAULT_DIRECT_INDICATOR_PARAMETERS,
    options,
  );
  const policy = conditionalFourSegmentPolicyMatrix(
    oracle.grid,
    oracle.currentGrid,
    prediction.conditionalParameters,
  );
  const accumulator = createExposureValueDistillationAccumulator({}, oracle.grid.length);
  observeExposureConditionalProbabilityDistillation(accumulator, oracle, 0, policy);
  const metrics = finalizeExposureValueDistillation(accumulator);
  const directAccumulator = createExposureValueDistillationAccumulator({}, oracle.grid.length);
  const observation = observeExposureFourSegmentProbabilityDistillation(
    directAccumulator,
    oracle,
    0,
    prediction.conditionalParameters,
    true,
  );
  const directMetrics = finalizeExposureValueDistillation(directAccumulator);
  const expectedStatistics = exposureConditionalProbabilityStatistics(
    policy,
    oracle.grid,
    oracle.currentGrid,
    oracle.execution.friction,
  );
  assert.equal(metrics.sampleCount, 1);
  assert.ok(Number.isFinite(metrics.crossEntropy) && metrics.crossEntropy > 0, metrics);
  assert.ok(metrics.score > 0 && metrics.score <= 1, metrics);
  assert.ok(Math.abs(directMetrics.crossEntropy - metrics.crossEntropy) < 1e-12);
  for (const key of ["mean", "secondMoment", "meanLogRebalance", "entropy"] as const) {
    assert.ok(Math.abs(observation!.statistics[key] - expectedStatistics[key]) < 1e-10, key);
  }
});
