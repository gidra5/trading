import assert from "node:assert/strict";
import test from "node:test";
import {
  fillOracleClosePrices,
  type OracleReturnNoiseConfig,
} from "./hindsight-oracle-return-noise.js";

const candles = [100, 101, 99, 102, 100.5, 103].map((close, index) => ({
  close,
  closeTime: (index + 1) * 60_000 - 1,
}));

test("clean oracle close prices remain exact", () => {
  const prices = new Float64Array(candles.length);
  fillOracleClosePrices(prices, candles, [], undefined);
  assert.deepEqual(Array.from(prices), candles.map((candle) => candle.close));
});

test("oracle return noise is deterministic, seeded, and does not mutate candles", () => {
  const snapshot = structuredClone(candles);
  const noise: OracleReturnNoiseConfig = {
    correlation: 0.5,
    seed: 17,
    rollingHorizonSteps: 2,
  };
  const first = new Float64Array(candles.length);
  const repeated = new Float64Array(candles.length);
  const different = new Float64Array(candles.length);

  fillOracleClosePrices(first, candles, [], noise);
  fillOracleClosePrices(repeated, candles, [], noise);
  fillOracleClosePrices(different, candles, [], { ...noise, seed: 18 });

  assert.deepEqual(first, repeated);
  assert.notDeepEqual(first, different);
  assert.deepEqual(candles, snapshot);
});

test("noise realizes the requested rolling-horizon return correlation", () => {
  const source = Array.from({ length: 1_001 }, (_, index) => ({
    close: 100 * Math.exp(0.0001 * index + 0.003 * Math.sin(index * 1.731)),
    closeTime: (index + 1) * 60_000 - 1,
  }));
  const real = logReturns(Float64Array.from(source, (candle) => candle.close));
  for (const horizon of [15, 60]) {
    for (const targetCorrelation of [0, 0.5, 0.8, 0.95]) {
      const prices = new Float64Array(source.length);
      fillOracleClosePrices(prices, source, [], {
        correlation: targetCorrelation,
        seed: 7,
        rollingHorizonSteps: horizon,
      });
      const perturbed = logReturns(prices);
      const realized = correlation(rollingSums(real, horizon), rollingSums(perturbed, horizon));
      assert.ok(
        Math.abs(realized - targetCorrelation) < 1e-10,
        `horizon=${horizon} target=${targetCorrelation} realized=${realized}`,
      );
      assert.ok(Math.abs(sum(real) - sum(perturbed)) < 1e-12);
      assert.equal(
        Array.from(perturbed).some((value, index) =>
          Math.abs(Math.abs(value) - Math.abs(real[index]!)) > 1e-6),
        true,
      );
    }
  }
});

function logReturns(prices: Float64Array): Float64Array {
  return Float64Array.from(
    { length: prices.length - 1 },
    (_, index) => Math.log(prices[index + 1]! / prices[index]!),
  );
}

function correlation(left: Float64Array, right: Float64Array): number {
  const leftMean = mean(left);
  const rightMean = mean(right);
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftDelta = left[index]! - leftMean;
    const rightDelta = right[index]! - rightMean;
    covariance += leftDelta * rightDelta;
    leftVariance += leftDelta ** 2;
    rightVariance += rightDelta ** 2;
  }
  return covariance / Math.sqrt(leftVariance * rightVariance);
}

function rollingSums(values: Float64Array, horizon: number): Float64Array {
  return Float64Array.from(
    { length: values.length - horizon + 1 },
    (_, start) => values.slice(start, start + horizon)
      .reduce((total, value) => total + value, 0),
  );
}

function sum(values: Float64Array): number {
  return values.reduce((total, value) => total + value, 0);
}

function mean(values: Float64Array): number {
  return sum(values) / values.length;
}
