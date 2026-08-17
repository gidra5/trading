import assert from "node:assert/strict";
import test from "node:test";
import {
  aggregateTargetCounts,
  correlation,
  emaSlopeFromLevels,
  equalMassEdges,
  rollingDiscreteEvaluation,
} from "./analyze-technical-indicator-predictiveness.ts";

test("equal-mass edges are ordered", () => {
  const edges = equalMassEdges(Float64Array.from([9, 1, 8, 2, 7, 3, 6, 4, 5, 0]), 5);
  assert.deepEqual(edges, [2, 4, 6, 8]);
});

test("target aggregation conserves included joint counts", () => {
  const mapping = new Int16Array(33);
  mapping.fill(-1);
  mapping[1] = 0;
  mapping[17] = 1;
  const counts = new Float64Array(33);
  counts[0] = 100;
  counts[1] = 30;
  counts[17] = 20;
  const aggregated = aggregateTargetCounts(counts, mapping, 2);
  assert.deepEqual(Array.from(aggregated), [30, 20]);
});

test("rolling nested score rewards a stable indicator beyond history", () => {
  const annual = Array.from({ length: 3 }, () => {
    const counts = new Float64Array(33 * 16 * 33);
    for (let history = 0; history < 33; history += 1) {
      // Feature 0 predicts negative active class; feature 1 predicts positive.
      counts[(history * 16) * 33 + 1] = 100;
      counts[(history * 16 + 1) * 33 + 17] = 100;
    }
    return counts;
  });
  const mapping = new Int16Array(33);
  mapping.fill(-1);
  for (let jointClass = 1; jointClass < 33; jointClass += 1) {
    mapping[jointClass] = Math.floor((jointClass - 1) / 16);
  }
  const result = rollingDiscreteEvaluation(annual, {
    id: "activeSign",
    label: "active sign",
    classes: 2,
    mapping,
    binary: true,
  });
  assert.ok(result.standaloneGainBits > 0.9);
  assert.ok(result.incrementalGainBits > 0.9);
  assert.ok((result.incrementalAccuracyGain ?? 0) > 0.49);
});

test("correlation handles centered linear predictions", () => {
  assert.equal(correlation(3, 6, 12, 14, 56, 28), 1);
});

test("multi-step EMA slope is constant and acceleration vanishes on exponential growth", () => {
  const rate = 0.0002;
  const horizon = 16;
  const lag2 = 100;
  const lag1 = lag2 * Math.exp(rate * horizon);
  const current = lag1 * Math.exp(rate * horizon);
  const currentSlope = emaSlopeFromLevels(current, lag1, horizon);
  const previousSlope = emaSlopeFromLevels(lag1, lag2, horizon);
  assert.ok(Math.abs(currentSlope - 10_000 * rate) < 1e-12);
  assert.ok(Math.abs(currentSlope - previousSlope) < 1e-12);
});
