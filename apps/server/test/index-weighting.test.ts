import assert from "node:assert/strict";
import { test } from "node:test";
import { cappedProportionalWeights } from "../src/index-weighting.js";

test("capped proportional weights redistribute excess by remaining size", () => {
  const weights = cappedProportionalWeights([100, 10, 1], 0.5);

  assert.ok(Math.abs(weights[0] - 0.5) < 1e-12);
  assert.ok(Math.abs(weights[1] - 5 / 11) < 1e-12);
  assert.ok(Math.abs(weights[2] - 0.5 / 11) < 1e-12);
  assert.ok(Math.abs(weights.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("zero-size constituents receive equal weights when no size data exists", () => {
  assert.deepEqual(cappedProportionalWeights([0, 0, 0], 0.5), [
    1 / 3,
    1 / 3,
    1 / 3,
  ]);
});

test("weighting rejects invalid sizes and infeasible caps", () => {
  assert.throws(
    () => cappedProportionalWeights([1, 1, 1], 0.2),
    /infeasible/,
  );
  assert.throws(
    () => cappedProportionalWeights([1, -1], 0.5),
    /non-negative/,
  );
});
