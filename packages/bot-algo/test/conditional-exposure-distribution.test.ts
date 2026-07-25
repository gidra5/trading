import assert from "node:assert/strict";
import test from "node:test";
import {
  conditionalFourSegmentExposureProbabilities,
  conditionalFourSegmentLogSlope,
  conditionalFourSegmentParametersAt,
  conditionalFourSegmentParametersFromRaw,
  conditionalFourSegmentPolicyMatrix,
  CONDITIONAL_FOUR_SEGMENT_PARAMETER_COUNT,
  fitConditionalFourSegmentPolicy,
  type ConditionalFourSegmentParameters,
} from "../src/conditional-exposure-distribution.js";

const friction = 0.00175;
const temperature = 0.01;
const support = {
  latentLower: -250,
  latentUpper: 250,
  visibleLower: -100,
  visibleUpper: 100,
  friction,
  temperature,
};
const raw = Float64Array.of(-0.7, -0.3, 2.4, 4.0, 5.2, -3.7, -14, 14);
const parameters = conditionalFourSegmentParametersFromRaw(raw, support);

test("eight raw coordinates decode effective-range geometry and a quadratic score", () => {
  assert.equal(CONDITIONAL_FOUR_SEGMENT_PARAMETER_COUNT, 8);
  assert.throws(
    () => conditionalFourSegmentParametersFromRaw(raw.slice(0, 6), support),
    /must contain 8 values/,
  );
  assert.equal(parameters.kappaC1, 82 / 500);
  assert.equal(parameters.kappaX, 678 / 500);
  assert.equal(parameters.kappaC2, 82 / 500);
  assert.ok(Math.abs(parameters.betaX
    + (friction / (1 - friction) + friction) / temperature) < 1e-14);
  assert.equal(parameters.quadraticPrecision, raw[3]! / 62_500);
  assert.equal(parameters.cutoffLower, -250);
  assert.equal(parameters.cutoffUpper, 250);
});

test("explicit usable-span calibration survives wider effective support", () => {
  const compactSharpness = conditionalFourSegmentParametersFromRaw(raw, {
    ...support,
    hingeSpan: 200,
  });
  assert.equal(compactSharpness.kappaC1, 82 / 200);
  assert.equal(compactSharpness.kappaX, 678 / 200);
  assert.equal(compactSharpness.kappaC2, 82 / 200);
});

test("conditional rows use strict visible truncation and normalization", () => {
  const actions = Float64Array.from({ length: 61 }, (_, index) => -150 + index * 5);
  const row = conditionalFourSegmentExposureProbabilities(actions, 170, parameters);
  assert.ok(Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  for (let index = 0; index < actions.length; index += 1) {
    assert.ok(Number.isFinite(row[index]!));
    if (actions[index]! < -100 || actions[index]! > 100) assert.equal(row[index], 0);
  }
});

test("learned survival cutoffs are exact hard probability boundaries", () => {
  const cutoff = conditionalFourSegmentParametersFromRaw(
    Float64Array.of(-0.7, -0.3, 2.4, 0.65, 5.2, -3.7, 0, 0),
    { ...support, visibleLower: -200, visibleUpper: 200 },
  );
  const actions = Float64Array.from({ length: 41 }, (_, index) => -200 + index * 10);
  const row = conditionalFourSegmentExposureProbabilities(actions, 0, cutoff);
  for (let index = 0; index < actions.length; index += 1) {
    if (actions[index]! < -125 || actions[index]! > 125) assert.equal(row[index], 0);
  }
  assert.ok(row.some((value) => value > 0));
});

test("quadratic precision contributes the exact linear action-slope change", () => {
  const withoutQuadratic: ConditionalFourSegmentParameters = {
    ...parameters,
    quadraticPrecision: 0,
  };
  for (const action of [-80, -15, 55]) {
    const delta = conditionalFourSegmentLogSlope(action, 12, parameters)
      - conditionalFourSegmentLogSlope(action, 12, withoutQuadratic);
    assert.ok(Math.abs(delta + parameters.quadraticPrecision * action) < 1e-12);
  }
});

test("ordered segment diagnostics exclude smooth quadratic curvature", () => {
  for (const current of [-200, parameters.c1, 0, parameters.c2, 200]) {
    const slice = conditionalFourSegmentParametersAt(current, parameters);
    assert.equal(slice.segmentSlopeOffsets.length, 4);
    assert.ok(Object.values(slice).flat().every(Number.isFinite), { current, slice });
    assert.ok(slice.betaX < 0);
  }
});

test("eight-parameter fitter recovers a complete quadratic conditional surface", () => {
  const actions = Float64Array.from({ length: 51 }, (_, index) => -100 + index * 4);
  const currents = Float64Array.from({ length: 21 }, (_, index) => -240 + index * 24);
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, parameters);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    ...support,
    initialC1: parameters.c1,
    initialC2: parameters.c2,
    maxIterations: 120,
    sampleStates: currents.length,
    sampleActions: actions.length,
  });
  assert.equal(fit.rawParameters.length, 8);
  assert.ok(fit.meanSquaredError < 3e-6, JSON.stringify(fit));
  assert.ok(Math.abs(fit.parameters.quadraticPrecision) > 1e-5);
  assert.equal(fit.parameters.betaX, parameters.betaX);
  const visibleRows = Array.from(currents).filter((current) =>
    current >= support.visibleLower && current <= support.visibleUpper);
  let visibleCrossEntropy = 0;
  let visibleMeanSquaredError = 0;
  for (const current of visibleRows) {
    const row = conditionalFourSegmentExposureProbabilities(actions, current, fit.parameters);
    const sourceRow = Array.from(currents).indexOf(current);
    for (let action = 0; action < actions.length; action += 1) {
      const expected = target[sourceRow * actions.length + action]!;
      if (expected > 0) {
        visibleCrossEntropy -= expected * Math.log(Math.max(1e-300, row[action]!))
          / visibleRows.length;
      }
      visibleMeanSquaredError += (row[action]! - expected) ** 2
        / (visibleRows.length * actions.length);
    }
  }
  assert.ok(Math.abs(fit.crossEntropy - visibleCrossEntropy) < 1e-12);
  assert.ok(Math.abs(fit.meanSquaredError - visibleMeanSquaredError) < 1e-12);
});

test("the quadratic coordinate materially improves a curved target", () => {
  const actions = Float64Array.from({ length: 51 }, (_, index) => -100 + index * 4);
  const currents = Float64Array.from({ length: 15 }, (_, index) => -210 + index * 30);
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, parameters);
  const withoutQuadratic = conditionalFourSegmentPolicyMatrix(actions, currents, {
    ...parameters,
    quadraticPrecision: 0,
  });
  const baselineMse = target.reduce((sum, value, index) =>
    sum + (value - withoutQuadratic[index]!) ** 2 / target.length, 0);
  const fit = fitConditionalFourSegmentPolicy(actions, target, currents, {
    ...support,
    maxIterations: 100,
    sampleStates: currents.length,
    sampleActions: actions.length,
  });
  assert.ok(fit.meanSquaredError < baselineMse * 0.25, { fit, baselineMse });
});

test("cross-entropy refinement cannot worsen the projected initializer", () => {
  const actions = Float64Array.from({ length: 41 }, (_, index) => -100 + index * 5);
  const currents = Float64Array.from({ length: 13 }, (_, index) => -180 + index * 30);
  const target = conditionalFourSegmentPolicyMatrix(actions, currents, parameters);
  const projected = fitConditionalFourSegmentPolicy(actions, target, currents, {
    ...support,
    restartCount: 2,
    refineProjectedFit: false,
  });
  const refined = fitConditionalFourSegmentPolicy(actions, target, currents, {
    ...support,
    restartCount: 2,
    maxIterations: 80,
  });
  assert.equal(projected.refined, false);
  assert.equal(refined.refined, true);
  assert.ok(refined.crossEntropy <= projected.crossEntropy + 1e-10, { projected, refined });
});
