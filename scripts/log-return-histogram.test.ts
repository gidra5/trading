import assert from "node:assert/strict";
import test from "node:test";
import {
  createLogReturnHistogramSpec,
  LogReturnHistogramCounter,
} from "./lib/log-return-histogram.js";

test("histogram has a zero-centered bin and preserves probability mass", () => {
  const spec = createLogReturnHistogramSpec(10, 0.1, 2);
  assert.equal(spec.binCount, 41);
  assert.ok(Math.abs(spec.binWidthBps - 1) < 1e-12);
  const counter = new LogReturnHistogramCounter(spec);
  for (const bps of [-30, -20.5, -20.49, -0.1, 0, 0.49, 20.49, 20.5, 30]) {
    counter.addBps(bps);
  }
  const result = counter.finish();
  const zeroBin = result.bins.find((bin) => bin.lowerBps <= 0 && bin.upperBps > 0);
  assert.ok(zeroBin);
  assert.ok(Math.abs(zeroBin.centerBps) < 1e-12);
  assert.equal(zeroBin.probability, 3 / 9);
  assert.equal(result.underflowProbability, 1 / 9);
  assert.equal(result.overflowProbability, 2 / 9);
  const total = result.bins.reduce((sum, bin) => sum + bin.probability, 0)
    + result.underflowProbability
    + result.overflowProbability;
  assert.ok(Math.abs(total - 1) < 1e-12);
});
