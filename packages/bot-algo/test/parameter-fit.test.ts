import assert from "node:assert/strict";
import test from "node:test";
import {
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentParametersFromRaw,
} from "../src/conditional-exposure-distribution.js";
import {
  conditionalFourSegmentScoreMatrix,
  fitConditionalFourSegmentRegret,
  fitConditionalFourSegmentScores,
} from "../src/parameter-fit.js";

const options = {
  latentLower: -250,
  latentUpper: 250,
  visibleLower: -100,
  visibleUpper: 100,
  friction: 0.00175,
  temperature: 0.01,
};
const parameters = conditionalFourSegmentParametersFromRaw(
  Float64Array.of(-0.8, -0.15, 1.7, -0.45, 4.1, -2.8, -14, 14),
  options,
);

test("score evaluator is the unnormalized kernel used by the policy", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const current = 23;
  const scores = conditionalFourSegmentScoreMatrix(actions, Float64Array.of(current), parameters);
  const maximum = scores.reduce((best, value) => Math.max(best, value), -Infinity);
  const expected = Float64Array.from(scores, (score) => Math.exp(score - maximum));
  const total = expected.reduce((sum, value) => sum + value, 0);
  const actual = conditionalFourSegmentExposureProbabilities(actions, current, parameters);
  assert.ok(actual.every((value, index) => Math.abs(value - expected[index]! / total) < 1e-14));
});

test("score fit eliminates arbitrary per-state offsets", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const states = Float64Array.of(-190, -95, 0, 85, 180);
  const offsets = Float64Array.of(3.2, -1.7, 0.4, 2.1, -4.8);
  const scores = conditionalFourSegmentScoreMatrix(actions, states, parameters, offsets);
  const fit = fitConditionalFourSegmentScores(actions, scores, states, {
    ...options,
    initialC1: parameters.c1,
    initialC2: parameters.c2,
    restartCount: 2,
    maxIterations: 100,
  });
  assert.ok(fit.weightedMeanSquaredError < 5e-3, fit);
  assert.ok(fit.weightedRSquared > 0.999, fit);
  assert.equal(fit.fittedSupport, false);
});

test("regret fit preserves the temperature-scaled conditional surface", () => {
  const actions = Float64Array.from({ length: 51 }, (_, index) => -100 + index * 4);
  const states = Float64Array.from({ length: 11 }, (_, index) => -180 + index * 36);
  const scores = conditionalFourSegmentScoreMatrix(actions, states, parameters);
  const regrets = new Float64Array(scores.length);
  for (let row = 0; row < states.length; row += 1) {
    const offset = row * actions.length;
    const maximum = scores.subarray(offset, offset + actions.length)
      .reduce((best, value) => Math.max(best, value), -Infinity);
    for (let action = 0; action < actions.length; action += 1) {
      regrets[offset + action] = (maximum - scores[offset + action]!) * options.temperature;
    }
  }
  const fit = fitConditionalFourSegmentRegret(
    actions,
    regrets,
    states,
    options.temperature,
    { ...options, maxIterations: 100 },
  );
  assert.ok(fit.weightedRSquared > 0.999, fit);
});
