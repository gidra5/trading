import assert from "node:assert/strict";
import test from "node:test";
import {
  createExposureReturnAccumulator,
  createExposureValueDistillationAccumulator,
  exposureValueOracleProbabilities,
  finalizeExposureReturn,
  finalizeExposureValueDistillation,
  observeExposureReturn,
  observeExposureValueDistillation,
  prepareExposureValueOracle,
  rebalanceEquityFactor,
  shareExposureValueOracle,
  strategyExposureProbabilities,
} from "../src/exposure-value-distillation.js";

test("exposure value oracle prefers the sign of the next price move", () => {
  const up = prepareExposureValueOracle([100, 110], {
    scoreStartIndex: 0,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });
  const down = prepareExposureValueOracle([100, 90], {
    scoreStartIndex: 0,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });

  assert.ok(up.means[0]! > 0.5);
  assert.ok(down.means[0]! < -0.5);
  assert.equal(up.optimalExposures[0], 1);
  assert.equal(down.optimalExposures[0], -1);
  assert.ok(up.opportunities[0]! > 0);
  assert.ok(Math.abs(up.means[1]!) < 1e-7);
});

test("flat future prices produce a uniform exposure preference", () => {
  const oracle = prepareExposureValueOracle([100, 100, 100], {
    scoreStartIndex: 0,
    friction: 0,
    gridSize: 11,
    temperature: 0.01,
  });

  assert.ok(Math.abs(oracle.means[0]!) < 1e-7);
  assert.ok(Math.abs(oracle.secondMoments[0]! - 0.4) < 1e-6);
  assert.ok(Math.abs(oracle.entropies[0]! - Math.log(11)) < 1e-6);
  assert.ok(oracle.opportunities[0]! < 1e-12);
});

test("distillation loss rewards a strategy distribution centered on oracle preference", () => {
  const oracle = prepareExposureValueOracle([100, 110], {
    scoreStartIndex: 0,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });
  const aligned = createExposureValueDistillationAccumulator();
  const opposed = createExposureValueDistillationAccumulator();
  observeExposureValueDistillation(aligned, oracle, 0, 1, 0.15);
  observeExposureValueDistillation(opposed, oracle, 0, -1, 0.15);
  const alignedMetrics = finalizeExposureValueDistillation(aligned);
  const opposedMetrics = finalizeExposureValueDistillation(opposed);

  assert.ok(alignedMetrics.crossEntropy < opposedMetrics.crossEntropy);
  assert.ok(alignedMetrics.klDivergence < opposedMetrics.klDivergence);
  assert.ok(alignedMetrics.score > opposedMetrics.score);
});

test("rebalance factor follows the exact buy and sell fee equations", () => {
  const fee = 0.01;
  assert.ok(Math.abs(
    rebalanceEquityFactor(0, 0.5, fee)
      - (1 - fee * 0.5 / (1 - fee + fee * 0.5)),
  ) < 1e-15);
  assert.ok(Math.abs(
    rebalanceEquityFactor(0.5, -0.25, fee)
      - (1 - fee * 0.75 / (1 + fee * 0.25)),
  ) < 1e-15);
});

test("value oracle arrays can be shared across candidate workers", () => {
  const shared = shareExposureValueOracle(prepareExposureValueOracle([100, 101], {
    scoreStartIndex: 0,
    friction: 0.001,
    gridSize: 5,
    temperature: 0.01,
  }));

  assert.ok(shared.means.buffer instanceof SharedArrayBuffer);
  assert.ok(shared.grid.buffer instanceof SharedArrayBuffer);
});

test("retained oracle and strategy distributions are normalized", () => {
  const oracle = prepareExposureValueOracle([100, 102], {
    scoreStartIndex: 0,
    friction: 0,
    gridSize: 9,
    temperature: 0.01,
    includeProbabilities: true,
  });
  const oracleProbabilities = exposureValueOracleProbabilities(oracle, 0);
  const strategyProbabilities = strategyExposureProbabilities(oracle.grid, 0.4, 0.15);
  assert.ok(Math.abs(oracleProbabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
  assert.ok(Math.abs(strategyProbabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("return measurement starts flat and marks the requested exposure", () => {
  const returns = createExposureReturnAccumulator();
  observeExposureReturn(returns, 1, 100, 110, {
    friction: 0,
    minExposure: -1,
    maxExposure: 1,
    quoteLendRate: 0,
    quoteBorrowRate: 0,
    assetBorrowRate: 0,
  });
  const metrics = finalizeExposureReturn(returns);
  assert.ok(Math.abs(metrics.equity - 1.1) < 1e-12);
  assert.ok(Math.abs(metrics.totalReturn - 0.1) < 1e-12);
  assert.equal(metrics.rebalanceCount, 1);
  assert.equal(metrics.turnover, 1);
});
