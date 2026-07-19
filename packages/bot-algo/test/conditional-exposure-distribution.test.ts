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
  supportSharpness: 1,
  betaC1: [0.025, -0.008],
  betaC2: [-0.018, 0.006],
  tilt: [0.003, 0.012],
  strengthRaw: [-3.2, 0.25],
  widthRaw: [3.2, -0.15],
  kappaC1Raw: -1.8,
  kappaC2Raw: -1.5,
  minimumStrength: 1e-5,
  minimumWidth: 0.5,
  minimumKappa: 1e-4,
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

test("conditional four-segment fitter learns a complete conditional surface", () => {
  const actions = Float64Array.from({ length: 31 }, (_, index) => -100 + index * 200 / 30);
  const currents = Float64Array.from({ length: 17 }, (_, index) => -240 + index * 480 / 16);
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, parameters);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    latentLower: -250,
    latentUpper: 250,
    visibleLower: -100,
    visibleUpper: 100,
    c1: -35,
    c2: 40,
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
  assert.ok(fittedMse < uniformMse * 0.25, { fit, fittedMse, uniformMse });
  assert.ok(Math.abs(fittedMse - fit.meanSquaredError) < 1e-14, { fit, fittedMse });
});
