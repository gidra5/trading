import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  predictHandcraftedIndicatorRegret,
  prepareHandcraftedIndicatorStateAt,
  prepareHandcraftedIndicatorStates,
} from "../src/handcrafted-indicator-predictor.js";
import {
  createExposureValueDistillationAccumulator,
  exposureProbabilityTransitionCrossEntropy,
  finalizeExposureValueDistillation,
  observeExposureProbabilityDistillation,
  prepareExposureValueOracle,
} from "../src/exposure-value-distillation.js";

const execution = {
  friction: 0.001,
  minExposure: -2,
  maxExposure: 2,
  maxEffectiveExposure: 3,
  quoteBorrowRate: 0,
  assetBorrowRate: 0,
};
const grid = Float64Array.of(-2, -1, 0, 1, 2);

test("handcrafted indicator states are causal and finite", () => {
  const prices = [100, 101, 102, 101, 103];
  const states = prepareHandcraftedIndicatorStates(
    prices,
    60_000,
    DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  );
  const prefix = prepareHandcraftedIndicatorStates(
    prices.slice(0, 4),
    60_000,
    DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  );
  assert.deepEqual(states.slice(0, 4), prefix);
  assert.ok(states.every((state) => Object.values(state).every(Number.isFinite)));
  assert.deepEqual(
    prepareHandcraftedIndicatorStateAt(
      prices,
      prices.length - 1,
      60_000,
      DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
    ),
    states.at(-1),
  );
});

test("flat forecast with friction prefers cash and produces normalized regret", () => {
  const prediction = predictHandcraftedIndicatorRegret(
    grid,
    { drift: 0, variance: 0.0001, longRunVariance: 0.0001 },
    DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
    { intervalMs: 60_000, holdingPeriodSteps: 1, valueHorizonSteps: 10, temperature: 0.01, execution },
  );
  assert.equal(prediction.optimalExposure, 0);
  assert.equal(Math.min(...prediction.regrets), 0);
  assert.ok(prediction.regrets.every((regret) => regret >= 0));
  assert.ok(Math.abs(prediction.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("strong causal drift selects the matching exposure direction", () => {
  const options = {
    intervalMs: 60_000,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 5,
    temperature: 0.01,
    execution,
  };
  const parameters = { ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS, driftScale: 1 };
  const up = predictHandcraftedIndicatorRegret(
    grid,
    { drift: 0.01, variance: 0.00001, longRunVariance: 0.00001 },
    parameters,
    options,
  );
  const down = predictHandcraftedIndicatorRegret(
    grid,
    { drift: -0.01, variance: 0.00001, longRunVariance: 0.00001 },
    parameters,
    options,
  );
  assert.ok(up.optimalExposure > 0, up);
  assert.ok(down.optimalExposure < 0, down);
});

test("forecast probabilities feed the transition-aware oracle loss", () => {
  const oracle = prepareExposureValueOracle([100, 101, 102], {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 2,
    friction: execution.friction,
    gridSize: grid.length,
    minExposure: grid[0],
    maxExposure: grid.at(-1),
    maxEffectiveExposure: 3,
    temperature: 0.01,
    includeProbabilities: true,
  });
  const prediction = predictHandcraftedIndicatorRegret(
    oracle.grid,
    { drift: 0.001, variance: 0.0001, longRunVariance: 0.0001 },
    DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
    {
      intervalMs: 60_000,
      holdingPeriodSteps: oracle.holdingPeriodSteps,
      valueHorizonSteps: oracle.valueHorizonSteps,
      temperature: oracle.temperature,
      execution: oracle.execution,
    },
  );
  const accumulator = createExposureValueDistillationAccumulator({}, oracle.grid.length);
  observeExposureProbabilityDistillation(
    accumulator,
    oracle,
    0,
    prediction.probabilities,
    1 / oracle.temperature,
  );
  const metrics = finalizeExposureValueDistillation(accumulator);
  assert.equal(metrics.sampleCount, 1);
  assert.ok(Number.isFinite(metrics.crossEntropy) && metrics.crossEntropy >= 0, metrics);
  assert.ok(Math.abs(metrics.crossEntropy - exposureProbabilityTransitionCrossEntropy(
    oracle,
    0,
    prediction.probabilities,
    1 / oracle.temperature,
  )) < 1e-12);
});
