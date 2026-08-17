import assert from "node:assert/strict";
import test from "node:test";
import {
  aggregateLogReturns,
  returnsInWindow,
  summarizeReturnDistribution,
} from "./lib/log-return-distribution.js";

const MINUTE_MS = 60_000;

test("aggregates aligned closes and does not bridge incomplete buckets", () => {
  const candles = [
    { openTime: 0, close: 100 },
    { openTime: MINUTE_MS, close: 101 },
    { openTime: 2 * MINUTE_MS, close: 102 },
    { openTime: 3 * MINUTE_MS, close: 104 },
    { openTime: 4 * MINUTE_MS, close: 103 },
    { openTime: 5 * MINUTE_MS, close: 106 },
    { openTime: 8 * MINUTE_MS, close: 107 },
    { openTime: 9 * MINUTE_MS, close: 108 },
    { openTime: 10 * MINUTE_MS, close: 109 },
    { openTime: 11 * MINUTE_MS, close: 110 },
  ];

  const returns = aggregateLogReturns(candles, 2 * MINUTE_MS);
  assert.equal(returns.length, 3);
  assert.deepEqual(
    returns.map((item) => item.endTime),
    [4, 6, 12].map((minute) => minute * MINUTE_MS),
  );
  assert.ok(Math.abs(returns[0]!.value - Math.log(104 / 101)) < 1e-15);
  assert.ok(Math.abs(returns[1]!.value - Math.log(106 / 104)) < 1e-15);
  assert.ok(Math.abs(returns[2]!.value - Math.log(110 / 108)) < 1e-15);
});

test("selects returns ending inside the open-left, closed-right window", () => {
  const returns = [1, 2, 3, 4].map((minute) => ({
    endTime: minute * MINUTE_MS,
    value: minute / 1_000,
  }));
  assert.deepEqual(
    returnsInWindow(returns, MINUTE_MS, 3 * MINUTE_MS),
    [0.002, 0.003],
  );
});

test("normal-like symmetric data has stable location, shape, and dependence metrics", () => {
  const values = [-2, -1, 0, 1, 2].flatMap((value) => Array(100).fill(value / 10_000));
  const summary = summarizeReturnDistribution(values);
  assert.equal(summary.observations, 500);
  assert.ok(Math.abs(summary.meanBps) < 1e-12);
  assert.ok(Math.abs(summary.medianBps) < 1e-12);
  assert.ok(Math.abs(summary.skewness ?? 1) < 1e-12);
  assert.equal(summary.positiveFraction, 0.4);
  assert.equal(summary.zeroFraction, 0.2);
  assert.ok((summary.standardDeviationBps ?? 0) > 1);
  assert.equal(summary.absoluteTail.length, 15);
  assert.ok(summary.absoluteTail.every((item, index, array) =>
    index === 0 || item.probability <= array[index - 1]!.probability));
});
