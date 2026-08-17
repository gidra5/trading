import assert from "node:assert/strict";
import test from "node:test";
import {
  coarsenFeatureCounts,
  evaluateCandidateYear,
  quartileEdges,
} from "./analyze-indicator-information-basis.ts";

test("quartile edges select the 4/8/12 sixteenth cuts", () => {
  assert.deepEqual(quartileEdges(Array.from({ length: 15 }, (_, index) => index + 1)), [4, 8, 12]);
});

test("feature coarsening conserves all target counts", () => {
  const source = new Float64Array(33 * 16 * 33);
  source[(7 * 16 + 15) * 33 + 32] = 123;
  const result = coarsenFeatureCounts(source);
  assert.equal(result[(7 * 4 + 3) * 33 + 32], 123);
  assert.equal(result.reduce((sum, value) => sum + value, 0), 123);
});

test("conditional-information evaluation rewards a complementary candidate", () => {
  // One selected quartile plus one candidate quartile. The selected feature is
  // uninformative, while candidate state 0/1 perfectly separates target 0/1.
  const length = 33 * 16 * 33;
  const train = new Float64Array(length);
  const test = new Float64Array(length);
  for (let history = 0; history < 33; history += 1) {
    for (let selected = 0; selected < 4; selected += 1) {
      const state0 = history * 16 + selected * 4;
      const state1 = state0 + 1;
      train[state0 * 33] = 100;
      train[state1 * 33 + 1] = 100;
      test[state0 * 33] = 50;
      test[state1 * 33 + 1] = 50;
    }
  }
  const result = evaluateCandidateYear(train, test, 1, 1);
  assert.ok(result.marginalGainBits > 0.8);
  assert.ok(result.cumulativeGainBits > 0.8);
});
