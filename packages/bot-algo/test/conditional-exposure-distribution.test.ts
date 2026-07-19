import assert from "node:assert/strict";
import test from "node:test";
import {
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentParametersAt,
  conditionalFourSegmentPolicyMatrix,
  fitConditionalFourSegmentPolicy,
  type ConditionalFourSegmentParameters,
} from "../src/conditional-exposure-distribution.js";

const parameters: ConditionalFourSegmentParameters = {
  latentLower: -250,
  latentUpper: 250,
  visibleLower: -100,
  visibleUpper: 100,
  c1: -35,
  c2: 40,
  leftSupportWidth: 75,
  rightSupportWidth: 75,
  leftSupportSharpness: 0.7,
  rightSupportSharpness: 1.8,
  baseSlope: [0.012, 0.004],
  betaC1: [0.025, -0.008],
  betaX: [-0.04, 0.005],
  betaC2: [-0.018, 0.006],
  kappaC1: 0.15,
  kappaX: 0.22,
  kappaC2: 0.2,
};

test("conditional four-segment rows use strict visible truncation and normalization", () => {
  const actions = Float64Array.from({ length: 61 }, (_, index) => -150 + index * 5);
  const row = conditionalFourSegmentExposureProbabilities(actions, 170, parameters);

  assert.ok(Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  for (let index = 0; index < actions.length; index += 1) {
    assert.ok(Number.isFinite(row[index]!));
    if (actions[index]! < -100 || actions[index]! > 100) assert.equal(row[index], 0);
  }
});

test("conditional four-segment parameters remain finite as the moving breakpoint changes order", () => {
  for (const current of [-200, -35, 0, 40, 200]) {
    const slice = conditionalFourSegmentParametersAt(current, parameters);
    assert.equal(slice.segmentSlopes.length, 4);
    assert.ok(Object.values(slice).flat().every(Number.isFinite), { current, slice });
    assert.ok(slice.betaX < 0);
    assert.ok(slice.kappaC1 > 0 && slice.kappaX > 0 && slice.kappaC2 > 0);
  }
});

test("left and right compact-envelope sharpnesses independently affect boundary rows", () => {
  const actions = Float64Array.from({ length: 49 }, (_, index) => -240 + index * 10);
  const visibleSupportParameters = { ...parameters, visibleLower: -240, visibleUpper: 240 };
  const baselineLeft = conditionalFourSegmentExposureProbabilities(actions, 0, visibleSupportParameters);
  const sharperLeft = conditionalFourSegmentExposureProbabilities(actions, 0, {
    ...visibleSupportParameters,
    leftSupportSharpness: 3,
  });
  const baselineRight = conditionalFourSegmentExposureProbabilities(actions, 0, visibleSupportParameters);
  const sharperRight = conditionalFourSegmentExposureProbabilities(actions, 0, {
    ...visibleSupportParameters,
    rightSupportSharpness: 3,
  });

  assert.ok(baselineLeft.some((value, index) => Math.abs(value - sharperLeft[index]!) > 1e-6));
  assert.ok(baselineRight.some((value, index) => Math.abs(value - sharperRight[index]!) > 1e-6));
});

test("compact-envelope taper widths are independently fitted", () => {
  const actions = Float64Array.from({ length: 49 }, (_, index) => -240 + index * 10);
  const currents = Float64Array.of(-120, 0, 120);
  const targetParameters: ConditionalFourSegmentParameters = {
    ...parameters,
    visibleLower: -240,
    visibleUpper: 240,
    leftSupportWidth: 42,
    rightSupportWidth: 86,
    leftSupportSharpness: 0.8,
    rightSupportSharpness: 1.4,
    baseSlope: [0, 0],
    betaC1: [0, 0],
    betaX: [0, 0],
    betaC2: [0, 0],
  };
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, targetParameters);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -240,
    visibleUpper: 240,
    initialLeftSupportWidth: 100,
    initialRightSupportWidth: 100,
    maxIterations: 140,
    sampleStates: currents.length,
    sampleActions: actions.length,
  });

  assert.ok(Math.abs(fit.parameters.leftSupportWidth - 100) > 1, fit);
  assert.ok(Math.abs(fit.parameters.rightSupportWidth - 100) > 1, fit);
  assert.ok(fit.parameters.leftSupportWidth > 0, fit);
  assert.ok(fit.parameters.rightSupportWidth > 0, fit);
  assert.ok(
    fit.parameters.leftSupportWidth + fit.parameters.rightSupportWidth < 500,
    fit,
  );
  assert.ok(fit.meanSquaredError < 1e-7, fit);
});

test("conditional four-segment fitter learns a complete conditional surface", () => {
  const actions = Float64Array.from({ length: 31 }, (_, index) => -100 + index * 200 / 30);
  const currents = Float64Array.from({ length: 17 }, (_, index) => -240 + index * 480 / 16);
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, parameters);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    initialC1: -35,
    initialC2: 40,
    maxIterations: 70,
    sampleStates: currents.length,
    sampleActions: actions.length,
  });
  const fitted = conditionalFourSegmentPolicyMatrix(actions, currents, fit.parameters);
  const uniformMse = target.reduce((sum, probability) =>
    sum + (probability - 1 / actions.length) ** 2 / target.length, 0);
  const fittedMse = fitted.reduce((sum, probability, index) =>
    sum + (probability - target[index]!) ** 2 / target.length, 0);

  assert.ok(fit.iterations > 0 && fit.iterations <= 70, fit);
  assert.ok(Number.isFinite(fit.crossEntropy) && Number.isFinite(fit.klDivergence), fit);
  assert.ok(fit.parameters.leftSupportSharpness > 0, fit);
  assert.ok(fit.parameters.rightSupportSharpness > 0, fit);
  assert.ok(fittedMse < uniformMse * 0.25, { fit, fittedMse, uniformMse });
  assert.ok(Math.abs(fittedMse - fit.meanSquaredError) < 1e-14, { fit, fittedMse });
});

test("simple visible log-gradient fit keeps both unused breakpoints outside the visible window", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const currents = Float64Array.from({ length: 9 }, (_, index) => -240 + index * 60);
  const slope = 0.012;
  const row = Float64Array.from(actions, (action) => Math.exp(slope * action));
  const total = row.reduce((sum, value) => sum + value, 0);
  for (let index = 0; index < row.length; index += 1) row[index] /= total;
  const target = Float64Array.from(
    { length: currents.length * actions.length },
    (_, index) => row[index % actions.length]!,
  );
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    maxIterations: 50,
  });

  assert.ok(fit.parameters.c1 < -100, fit);
  assert.ok(fit.parameters.c2 > 100, fit);
  assert.ok(Math.abs(fit.parameters.baseSlope[0] - slope) < 1e-8, fit);
  assert.ok(Math.abs(fit.parameters.betaC1[0]) < 1e-8, fit);
  assert.ok(Math.abs(fit.parameters.betaX[0]) < 1e-8, fit);
  assert.ok(Math.abs(fit.parameters.betaC2[0]) < 1e-8, fit);
  assert.ok(fit.meanSquaredError < 1e-16, fit);
});

test("breakpoints and a positive moving slope change are freely learned", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const currents = Float64Array.from({ length: 13 }, (_, index) => -90 + index * 15);
  const targetParameters: ConditionalFourSegmentParameters = {
    ...parameters,
    c1: -58,
    c2: 47,
    baseSlope: [0.025, 0],
    betaC1: [-0.065, 0],
    betaX: [0.024, 0],
    betaC2: [0.075, 0],
    kappaC1: 0.45,
    kappaX: 0.35,
    kappaC2: 0.5,
  };
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, targetParameters);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    initialC1: -20,
    initialC2: 20,
    maxIterations: 160,
    restartCount: 3,
    sampleStates: currents.length,
    sampleActions: actions.length,
  });

  assert.ok(
    Math.abs(fit.parameters.c1 - targetParameters.c1) < 25,
    JSON.stringify(fit),
  );
  assert.ok(Math.abs(fit.parameters.c2 - targetParameters.c2) < 25, fit);
  assert.ok(fit.parameters.betaX[0] > 0, fit);
  assert.ok(fit.meanSquaredError < 2e-6, fit);
});

test("aggregate log-slope seeds recover a visible second breakpoint without kappa collapse", () => {
  const actions = Float64Array.from({ length: 51 }, (_, index) => -100 + index * 4);
  const currents = Float64Array.from({ length: 25 }, (_, index) => -240 + index * 20);
  const targetParameters: ConditionalFourSegmentParameters = {
    ...parameters,
    c1: -94,
    c2: 81,
    baseSlope: [-0.025, 0],
    betaC1: [0.04, 0],
    betaX: [-0.06, 0.008],
    betaC2: [-0.055, 0],
    kappaC1: 0.8,
    kappaX: 0.55,
    kappaC2: 0.9,
  };
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, targetParameters);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    sampleStates: currents.length,
    sampleActions: actions.length,
  });

  assert.ok(
    Math.abs(fit.parameters.c1 - targetParameters.c1) < 15,
    JSON.stringify(fit),
  );
  assert.ok(
    Math.abs(fit.parameters.c2 - targetParameters.c2) < 15,
    JSON.stringify(fit),
  );
  assert.ok(fit.parameters.kappaC2 >= 4.394 / 500, fit);
  assert.ok(fit.parameters.betaC2[0] < 0, fit);
  assert.ok(fit.meanSquaredError < 2e-6, JSON.stringify(fit));
});

test("moving slope changes are not clipped at the former beta limit", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const currents = Float64Array.from({ length: 13 }, (_, index) => -90 + index * 15);
  const targetParameters: ConditionalFourSegmentParameters = {
    ...parameters,
    c1: -150,
    c2: 150,
    baseSlope: [0.18, 0],
    betaC1: [0, 0],
    betaX: [-0.36, 0],
    betaC2: [0, 0],
    kappaX: 0.6,
  };
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, targetParameters);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    maxIterations: 160,
    sampleStates: currents.length,
    sampleActions: actions.length,
  });

  assert.ok(fit.parameters.betaX[0] < -0.3, JSON.stringify(fit));
  assert.ok(fit.meanSquaredError < 1e-8, JSON.stringify(fit));
});
