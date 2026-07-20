import assert from "node:assert/strict";
import test from "node:test";
import {
  conditionalFourSegmentExposureProbabilities,
  type ConditionalFourSegmentParameters,
} from "../src/conditional-exposure-distribution.js";
import {
  conditionalFourSegmentScoreMatrix,
  fitConditionalFourSegmentRegret,
  fitConditionalFourSegmentScores,
} from "../src/parameter-fit.js";

const parameters: ConditionalFourSegmentParameters = {
  latentLower: -250,
  latentUpper: 250,
  visibleLower: -100,
  visibleUpper: 100,
  c1: -42,
  c2: 47,
  leftSupportWidth: 95,
  rightSupportWidth: 80,
  leftSupportSharpness: 0.8,
  rightSupportSharpness: 1.35,
  baseSlope: [0.012, -0.004],
  betaC1: [0.027, 0.006],
  betaX: [-0.052, 0.009],
  betaC2: [0.021, -0.007],
  kappaC1: 0.18,
  kappaX: 0.24,
  kappaC2: 0.2,
};

test("score evaluator is the unnormalized kernel used by the conditional policy", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const current = 23;
  const scores = conditionalFourSegmentScoreMatrix(actions, Float64Array.of(current), parameters);
  const maximum = scores.reduce((best, value) => Math.max(best, value), -Infinity);
  const expected = Float64Array.from(scores, (score) => Math.exp(score - maximum));
  const total = expected.reduce((sum, value) => sum + value, 0);
  const actual = conditionalFourSegmentExposureProbabilities(actions, current, parameters);
  for (let index = 0; index < actions.length; index += 1) {
    assert.ok(Math.abs(actual[index]! - expected[index]! / total) < 1e-14);
  }
});

test("score fit eliminates arbitrary slice offsets and solves all eight linear coefficients", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const states = Float64Array.of(-190, -95, 0, 85, 180);
  const offsets = Float64Array.of(3.2, -1.7, 0.4, 2.1, -4.8);
  const scores = conditionalFourSegmentScoreMatrix(actions, states, parameters, offsets);
  const fit = fitConditionalFourSegmentScores(actions, scores, states, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    initialC1: parameters.c1,
    initialC2: parameters.c2,
    initialKappaC1: parameters.kappaC1,
    initialKappaX: parameters.kappaX,
    initialKappaC2: parameters.kappaC2,
    initialLeftSupportWidth: parameters.leftSupportWidth,
    initialRightSupportWidth: parameters.rightSupportWidth,
    leftSupportSharpness: parameters.leftSupportSharpness,
    rightSupportSharpness: parameters.rightSupportSharpness,
    ridge: 0,
    restartCount: 1,
    maxIterations: 0,
  });

  assert.ok(fit.weightedMeanSquaredError < 1e-15, fit);
  assert.ok(fit.maximumAbsoluteError < 2e-7, fit);
  for (const [actual, expected] of [
    [fit.parameters.baseSlope, parameters.baseSlope],
    [fit.parameters.betaC1, parameters.betaC1],
    [fit.parameters.betaX, parameters.betaX],
    [fit.parameters.betaC2, parameters.betaC2],
  ] as const) {
    assert.ok(Math.abs(actual[0] - expected[0]) < 2e-7, { actual, expected });
    assert.ok(Math.abs(actual[1] - expected[1]) < 2e-7, { actual, expected });
  }
  for (let index = 0; index < offsets.length; index += 1) {
    assert.ok(Math.abs(fit.sliceOffsets[index]! - offsets[index]!) < 2e-5, fit);
  }
  assert.equal(fit.fittedSupport, false);
});

test("regret fit converts temperature-scaled regret and refines interior structure", () => {
  const actions = Float64Array.from({ length: 51 }, (_, index) => -100 + index * 4);
  const states = Float64Array.from({ length: 11 }, (_, index) => -180 + index * 36);
  const scores = conditionalFourSegmentScoreMatrix(actions, states, parameters);
  const temperature = 0.07;
  const regrets = new Float64Array(scores.length);
  for (let row = 0; row < states.length; row += 1) {
    let maximum = Number.NEGATIVE_INFINITY;
    for (let action = 0; action < actions.length; action += 1) {
      maximum = Math.max(maximum, scores[row * actions.length + action]!);
    }
    for (let action = 0; action < actions.length; action += 1) {
      regrets[row * actions.length + action]
        = (maximum - scores[row * actions.length + action]!) * temperature;
    }
  }
  const fit = fitConditionalFourSegmentRegret(actions, regrets, states, temperature, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    initialC1: -28,
    initialC2: 33,
    initialKappaC1: 0.13,
    initialKappaX: 0.16,
    initialKappaC2: 0.14,
    initialLeftSupportWidth: parameters.leftSupportWidth,
    initialRightSupportWidth: parameters.rightSupportWidth,
    leftSupportSharpness: parameters.leftSupportSharpness,
    rightSupportSharpness: parameters.rightSupportSharpness,
    ridge: 1e-10,
    restartCount: 3,
    maxIterations: 100,
  });

  assert.ok(fit.weightedMeanSquaredError < 2e-6, JSON.stringify(fit));
  assert.ok(Math.abs(fit.parameters.c1 - parameters.c1) < 12, JSON.stringify(fit));
  assert.ok(Math.abs(fit.parameters.c2 - parameters.c2) < 12, JSON.stringify(fit));
  assert.ok(fit.weightedRSquared > 0.999, JSON.stringify(fit));
});

test("support taper fitting is explicit and uses latent boundary-layer scores", () => {
  const boundaryParameters: ConditionalFourSegmentParameters = {
    ...parameters,
    visibleLower: -240,
    visibleUpper: 240,
    baseSlope: [0, 0],
    betaC1: [0, 0],
    betaX: [0, 0],
    betaC2: [0, 0],
    leftSupportWidth: 62,
    rightSupportWidth: 104,
    leftSupportSharpness: 0.65,
    rightSupportSharpness: 1.7,
  };
  const actions = Float64Array.from({ length: 49 }, (_, index) => -240 + index * 10);
  const states = Float64Array.of(-150, 0, 150);
  const scores = conditionalFourSegmentScoreMatrix(actions, states, boundaryParameters);
  const fit = fitConditionalFourSegmentScores(actions, scores, states, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -240,
    visibleUpper: 240,
    initialC1: boundaryParameters.c1,
    initialC2: boundaryParameters.c2,
    initialKappaC1: boundaryParameters.kappaC1,
    initialKappaX: boundaryParameters.kappaX,
    initialKappaC2: boundaryParameters.kappaC2,
    initialLeftSupportWidth: 85,
    initialRightSupportWidth: 80,
    leftSupportSharpness: 1,
    rightSupportSharpness: 1,
    fitSupport: true,
    ridge: 1e-8,
    restartCount: 1,
    maxIterations: 120,
  });

  assert.equal(fit.fittedSupport, true);
  assert.ok(fit.weightedMeanSquaredError < 2e-5, fit);
  assert.ok(Math.abs(fit.parameters.leftSupportWidth - 85) > 2, fit);
  assert.ok(Math.abs(fit.parameters.rightSupportWidth - 80) > 2, fit);
  assert.ok(fit.parameters.leftSupportWidth + fit.parameters.rightSupportWidth < 500, fit);
});

test("zero-weight cells are ignored but every slice still requires positive mass", () => {
  const actions = Float64Array.of(-2, -1, 0, 1, 2);
  const states = Float64Array.of(-1, 0, 1);
  const scores = new Float64Array(actions.length * states.length);
  const weights = Float64Array.from(scores, () => 1);
  scores[2] = 1e6;
  weights[2] = 0;
  const fit = fitConditionalFourSegmentScores(actions, scores, states, {
    latentLower: -3,
    latentUpper: 3,
    initialC1: -1,
    initialC2: 1,
    initialLeftSupportWidth: 0.5,
    initialRightSupportWidth: 0.5,
    weights,
    maxIterations: 0,
    restartCount: 1,
  });
  assert.ok(fit.maximumAbsoluteError >= 1e6);
  assert.ok(fit.weightedMeanSquaredError < 1e-15, fit);
});
