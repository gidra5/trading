import assert from "node:assert/strict";
import test from "node:test";
import {
  addBlockCounts,
  addLagCounts,
  reversalMetrics,
  reverseCodeIndices,
} from "./analyze-return-reversibility.ts";

test("reverse block codes form an involution", () => {
  const reverse = reverseCodeIndices(5, 4);
  for (let index = 0; index < reverse.length; index += 1) {
    assert.equal(reverse[reverse[index]!]!, index);
  }
});

test("symmetric counts are exactly reversible", () => {
  const counts = Float64Array.from([
    10, 3, 4,
    3, 20, 5,
    4, 5, 30,
  ]);
  const metrics = reversalMetrics(counts, reverseCodeIndices(3, 2));
  assert.equal(metrics.jsDivergenceBits, 0);
  assert.equal(metrics.totalVariation, 0);
  assert.equal(metrics.jeffreysSmoothedEntropyProductionBits, 0);
});

test("a directed three-state cycle is maximally oriented", () => {
  const counts = Float64Array.from([
    0, 100, 0,
    0, 0, 100,
    100, 0, 0,
  ]);
  const metrics = reversalMetrics(counts, reverseCodeIndices(3, 2));
  assert.equal(metrics.jsDivergenceBits, 1);
  assert.equal(metrics.totalVariation, 1);
  assert.equal(metrics.optimalArrowClassifierAccuracy, 1);
});

test("streamed lag and block counts include boundaries exactly once", () => {
  const lagCounts = { "1": new Float64Array(9) };
  let lagCarry = addLagCounts(lagCounts, Uint8Array.from([0, 1, 2]), new Uint8Array(), 3);
  lagCarry = addLagCounts(lagCounts, Uint8Array.from([1, 0]), lagCarry, 3);
  assert.deepEqual(Array.from(lagCounts["1"]), [0, 1, 0, 1, 0, 1, 0, 1, 0]);
  assert.ok(lagCarry.length > 0);

  const blockCounts = { "3": new Float64Array(27) };
  let blockCarry = addBlockCounts(
    blockCounts,
    Uint8Array.from([0, 1, 2]),
    new Uint8Array(),
    3,
  );
  blockCarry = addBlockCounts(blockCounts, Uint8Array.from([1, 0]), blockCarry, 3);
  const nonzero = Array.from(blockCounts["3"].entries())
    .filter((entry) => entry[1] > 0)
    .map((entry) => entry[0]);
  assert.deepEqual(nonzero, [5, 16, 21]);
  assert.ok(blockCarry.length > 0);
});
