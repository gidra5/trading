import assert from "node:assert/strict";
import test from "node:test";
import {
  conditionalMetrics,
  rollingEvaluation,
} from "./analyze-return-conditional-sign-magnitude.ts";

test("conditionally factorized tables have zero conditional information", () => {
  // h0: sign rows proportional across magnitude; h1 has different marginals but also factorizes.
  const counts = Float64Array.from([
    10, 20, 30,
    20, 40, 60,
    30, 10, 20,
    60, 20, 40,
  ]);
  const metrics = conditionalMetrics(counts, 2, 3);
  assert.ok(Math.abs(metrics.conditionalMutualInformationBits) < 1e-14);
  assert.ok(Math.abs(metrics.totalVariationFromConditionalIndependentProduct) < 1e-14);
});

test("conditional magnitude can perfectly identify sign", () => {
  const counts = Float64Array.from([
    20, 0,
    0, 20,
  ]);
  const metrics = conditionalMetrics(counts, 1, 2);
  assert.equal(metrics.conditionalMutualInformationBits, 1);
  assert.equal(metrics.inSampleHistoryOnlySignAccuracy, 0.5);
  assert.equal(metrics.inSampleHistoryAndMagnitudeSignAccuracy, 1);
});

test("rolling holdout rewards stable conditional coupling", () => {
  const annual = Array.from({ length: 3 }, () => Float64Array.from([
    100, 10,
    10, 100,
  ]));
  const result = rollingEvaluation(annual, 1, 2);
  assert.ok(result.signLogLossGainBitsPerReturn > 0.5);
  assert.ok(result.magnitudeLogLossGainBitsPerReturn > 0.5);
  assert.ok(Math.abs(
    result.signLogLossGainBitsPerReturn - result.magnitudeLogLossGainBitsPerReturn
  ) < 1e-12);
  assert.ok(result.signAccuracyGain > 0.4);
});

test("rolling sign and magnitude gains obey the Bayes identity with empty cells", () => {
  const train = new Float64Array([
    1_000, 0, 0, 500,
    900, 0, 0, 600,
  ]);
  const test = new Float64Array([
    1, 200, 300, 2,
    2, 250, 350, 1,
  ]);
  const result = rollingEvaluation([train, test], 1, 4);
  assert.ok(Math.abs(
    result.signLogLossGainBitsPerReturn - result.magnitudeLogLossGainBitsPerReturn
  ) < 1e-12);
});
