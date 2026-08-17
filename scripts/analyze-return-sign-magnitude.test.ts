import assert from "node:assert/strict";
import test from "node:test";
import {
  equalMassCuts,
  independenceMetrics,
} from "./analyze-return-sign-magnitude.ts";

test("independent sign rows have zero measured dependence", () => {
  const metrics = independenceMetrics([10, 20, 30], [10, 20, 30]);
  assert.ok(Math.abs(metrics.mutualInformationBits) < 1e-15);
  assert.ok(Math.abs(metrics.jsFromIndependentProductBits) < 1e-15);
  assert.ok(Math.abs(metrics.totalVariationFromIndependentProduct) < 1e-15);
  assert.equal(metrics.magnitudeOnlyAccuracyGain, 0);
});

test("magnitude can perfectly identify sign", () => {
  const metrics = independenceMetrics([20, 0], [0, 20]);
  assert.equal(metrics.mutualInformationBits, 1);
  assert.equal(metrics.totalVariationFromIndependentProduct, 0.5);
  assert.equal(metrics.baselineMajoritySignAccuracy, 0.5);
  assert.equal(metrics.optimalMagnitudeOnlySignAccuracy, 1);
  assert.equal(metrics.magnitudeOnlyAccuracyGain, 0.5);
});

test("equal-mass cuts are ordered and conserve all cells", () => {
  const counts = Float64Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const cuts = equalMassCuts(counts, 4);
  assert.equal(cuts[0], 0);
  assert.equal(cuts.at(-1), counts.length);
  for (let index = 1; index < cuts.length; index += 1) {
    assert.ok(cuts[index]! > cuts[index - 1]!);
  }
});
