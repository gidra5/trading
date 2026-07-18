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
  quadraticExponentialLogNormalizer,
  rebalanceEquityFactor,
  shareExposureValueOracle,
  strategyExposureProbabilities,
  strategyExposureQuadraticCoefficient,
  strategyExposureTemperatures,
  strategyExposureVolatilities,
  truncatedExponentialLogNormalizer,
} from "../src/exposure-value-distillation.js";
import {
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
} from "../src/vw-kama-cuda.js";

test("CUDA exposure-value oracle matches the CPU Bellman recurrence", async (context) => {
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    context.skip(status.reason);
    return;
  }
  const prices = Array.from({ length: 2_000 }, (_, index) =>
    100 * Math.exp(index * 0.00001 + Math.sin(index / 17) * 0.002 + Math.sin(index / 83) * 0.004));
  for (const gridSize of [21, 65]) {
    const options = {
      scoreStartIndex: 300,
      horizonSteps: 17,
      friction: 0.00175,
      gridSize,
      minExposure: -1,
      maxExposure: 1,
      temperature: 0.001,
      opportunityEpsilon: 0.000001,
      quoteLendRate: 0.000001,
      quoteBorrowRate: 0.000002,
      assetBorrowRate: 0.000003,
      includeProbabilities: true,
    };
    const cpu = prepareExposureValueOracle(prices, options);
    const { oracle: gpu } = await prepareExposureValueOracleCuda(prices, options);

    for (const key of [
      "means",
      "secondMoments",
      "entropies",
      "weights",
      "opportunities",
      "probabilities",
    ] as const) {
      const expected = cpu[key]!;
      const actual = gpu[key]!;
      let maximumError = 0;
      for (let index = 0; index < expected.length; index += 1) {
        maximumError = Math.max(maximumError, Math.abs(expected[index]! - actual[index]!));
      }
      assert.ok(maximumError < 2e-5, `${key} drifted by ${maximumError} at grid ${gridSize}`);
    }
    assert.deepEqual(gpu.modalExposures, cpu.modalExposures);
    assert.deepEqual(gpu.optimalExposures, cpu.optimalExposures);
  }

  const leveragedOptions = {
    scoreStartIndex: 300,
    horizonSteps: 5,
    friction: 0.00175,
    gridSize: 21,
    minExposure: -100,
    maxExposure: 100,
    maxEffectiveExposure: 250,
    temperature: 0.01,
  };
  const leveragedCpu = prepareExposureValueOracle(prices, leveragedOptions);
  const { oracle: leveragedGpu } = await prepareExposureValueOracleCuda(prices, leveragedOptions);
  let maximumMeanError = 0;
  for (let index = 300; index < prices.length; index += 1) {
    maximumMeanError = Math.max(
      maximumMeanError,
      Math.abs(leveragedCpu.means[index]! - leveragedGpu.means[index]!),
    );
  }
  assert.ok(maximumMeanError < 2e-3, `leveraged mean drifted by ${maximumMeanError}`);
  assert.deepEqual(leveragedGpu.optimalExposures, leveragedCpu.optimalExposures);
});

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

test("exposure value oracle retains selected H when the segment tail truncates the hold", () => {
  const oracle = prepareExposureValueOracle([100, 101, 102], {
    scoreStartIndex: 0,
    horizonSteps: 10,
    friction: 0,
    gridSize: 3,
    temperature: 0.01,
  });

  assert.equal(oracle.horizonSteps, 10);
});

test("exposure value holds the input target for H before perfect continuation", () => {
  const prices = [100, 120, 60, 60];
  const oneStep = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    horizonSteps: 1,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });
  const twoSteps = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    horizonSteps: 2,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });

  assert.equal(oneStep.modalExposures[0], 1);
  assert.equal(twoSteps.modalExposures[0], -1);
  assert.equal(twoSteps.optimalExposures[0], twoSteps.optimalExposures[1]);
});

test("H-step oracle values include drift-correcting rebalances", () => {
  const prices = [100, 98, 95];
  const friction = 0.00175;
  const temperature = 0.01;
  const oracle = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    horizonSteps: 2,
    friction,
    gridSize: 5,
    temperature,
    includeProbabilities: true,
  });
  const values = Array.from(oracle.grid, (exposure) => {
    const firstFactor = 1 - exposure + exposure * prices[1]! / prices[0]!;
    const driftedExposure = exposure * prices[1]! / prices[0]! / firstFactor;
    const rebalance = rebalanceEquityFactor(driftedExposure, exposure, friction);
    const secondFactor = 1 - exposure + exposure * prices[2]! / prices[1]!;
    return Math.log(firstFactor * rebalance * secondFactor);
  });
  const maximum = Math.max(...values);
  const weights = values.map((value) => Math.exp((value - maximum) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  const actual = exposureValueOracleProbabilities(oracle, 0);
  for (let index = 0; index < weights.length; index += 1) {
    assert.ok(Math.abs(actual[index]! - weights[index]! / total) < 1e-6);
  }
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

test("distillation loss rewards a signed-rate exponential biased toward the oracle", () => {
  const oracle = prepareExposureValueOracle([100, 110], {
    scoreStartIndex: 0,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });
  const aligned = createExposureValueDistillationAccumulator();
  const opposed = createExposureValueDistillationAccumulator();
  observeExposureValueDistillation(aligned, oracle, 0, 100, 3_600_000, 0.01);
  observeExposureValueDistillation(opposed, oracle, 0, -100, 3_600_000, 0.01);
  const alignedMetrics = finalizeExposureValueDistillation(aligned);
  const opposedMetrics = finalizeExposureValueDistillation(opposed);

  assert.ok(alignedMetrics.crossEntropy < opposedMetrics.crossEntropy);
  assert.ok(alignedMetrics.klDivergence < opposedMetrics.klDivergence);
  assert.ok(alignedMetrics.score > opposedMetrics.score);
});

test("truncated exponential partition matches explicit log-sum-exp", () => {
  for (const gridSize of [3, 21, 101, 1_024]) {
    for (const slope of [0, 1e-12, -1e-12, 0.7, -4.2, 1_000, -1_000]) {
      const minimum = -1.5;
      const maximum = 2;
      const values = Array.from({ length: gridSize }, (_, index) =>
        slope * (minimum + index / (gridSize - 1) * (maximum - minimum)));
      const peak = Math.max(...values);
      const expected = peak + Math.log(values.reduce(
        (sum, value) => sum + Math.exp(value - peak),
        0,
      ));
      const actual = truncatedExponentialLogNormalizer(gridSize, minimum, maximum, slope);
      assert.ok(Math.abs(actual - expected) < 1e-10, {
        gridSize,
        slope,
        expected,
        actual,
      });
    }
  }
});

test("quadratic exponential partition and probabilities match explicit log-sum-exp", () => {
  const minimum = -3;
  const maximum = 5;
  for (const gridSize of [3, 21, 101]) {
    for (const linear of [-4, 0, 2.5]) {
      for (const quadratic of [-2, -0.01, 0, 0.03, 1.5]) {
        const values = Array.from({ length: gridSize }, (_, index) => {
          const exposure = minimum + index / (gridSize - 1) * (maximum - minimum);
          return linear * exposure + quadratic * exposure * exposure;
        });
        const peak = Math.max(...values);
        const expected = peak + Math.log(values.reduce(
          (sum, value) => sum + Math.exp(value - peak),
          0,
        ));
        const actual = quadraticExponentialLogNormalizer(
          gridSize,
          minimum,
          maximum,
          linear,
          quadratic,
        );
        assert.ok(Math.abs(actual - expected) < 1e-10, {
          gridSize,
          linear,
          quadratic,
          expected,
          actual,
        });
      }
    }
  }

  const grid = new Float64Array([-2, -1, 0, 1, 2]);
  const concave = strategyExposureProbabilities(grid, 0, 3_600_000, 1, -1);
  const convex = strategyExposureProbabilities(grid, 0, 3_600_000, 1, 1);
  assert.ok(concave[2]! > concave[0]!);
  assert.ok(convex[0]! > convex[2]!);
  assert.ok(Math.abs(convex[0]! - convex.at(-1)!) < 1e-12);
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
  const strategyProbabilities = strategyExposureProbabilities(oracle.grid, 100, 3_600_000, 0.01);
  assert.ok(Math.abs(oracleProbabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
  assert.ok(Math.abs(strategyProbabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("signed rate selects the side and lower temperature increases concentration", () => {
  const grid = new Float64Array([-1, -0.5, 0, 0.5, 1]);
  const positive = strategyExposureProbabilities(grid, 100, 3_600_000, 0.01);
  const negative = strategyExposureProbabilities(grid, -100, 3_600_000, 0.01);
  const colder = strategyExposureProbabilities(grid, 100, 3_600_000, 0.005);
  for (let index = 0; index < grid.length; index += 1) {
    assert.ok(Math.abs(positive[index]! - negative[grid.length - 1 - index]!) < 1e-12);
  }
  assert.ok(positive.at(-1)! > positive[0]!);
  assert.ok(negative[0]! > negative.at(-1)!);
  assert.ok(colder.at(-1)! > positive.at(-1)!);
});

test("strategy calibration uses trailing-H log-return volatility", () => {
  const hourly = strategyExposureTemperatures([100, 101], {
    intervalMs: 3_600_000,
    horizonSteps: 1,
    temperature: 0.01,
  });
  const quarterHourly = strategyExposureTemperatures([100, 101], {
    intervalMs: 900_000,
    horizonSteps: 4,
    temperature: 0.01,
  });
  assert.ok(Math.abs(hourly[0]! - 0.01) < 1e-8);
  assert.ok(Math.abs(quarterHourly[0]! - 0.02) < 1e-8);

  const exact = strategyExposureTemperatures([100, 110, 99], {
    intervalMs: 60_000,
    horizonSteps: 2,
    temperature: 0.01,
    scaleByVolatility: true,
  });
  const firstReturn = Math.log(1.1);
  const secondReturn = Math.log(0.9);
  const expectedVolatility = Math.abs(firstReturn - secondReturn) / 2;
  assert.ok(Math.abs(exact[2]! - 0.01 * Math.sqrt(2) * expectedVolatility) < 1e-8);

  const volatilities = strategyExposureVolatilities([100, 110, 99], 2);
  assert.ok(Math.abs(volatilities[2]! - expectedVolatility) < 1e-8);
  const quadraticScale = 250;
  assert.ok(Math.abs(
    strategyExposureQuadraticCoefficient(quadraticScale, volatilities[2]!)
      + quadraticScale * volatilities[2]! * volatilities[2]!,
  ) < 1e-12);
  assert.equal(strategyExposureQuadraticCoefficient(quadraticScale, volatilities[0]!), 0);
  assert.throws(() => strategyExposureQuadraticCoefficient(-1, expectedVolatility));

  const prices = (amplitude: number): number[] => Array.from({ length: 20 }, (_, index) =>
    100 * Math.exp((index % 2 === 0 ? -1 : 1) * amplitude));
  const low = strategyExposureTemperatures(prices(0.001), {
    intervalMs: 60_000,
    horizonSteps: 10,
    temperature: 0.01,
    scaleByVolatility: true,
  });
  const high = strategyExposureTemperatures(prices(0.01), {
    intervalMs: 60_000,
    horizonSteps: 10,
    temperature: 0.01,
    scaleByVolatility: true,
  });
  assert.ok(high.at(-1)! > low.at(-1)! * 9);
});

test("return measurement starts flat and marks the requested exposure", () => {
  const returns = createExposureReturnAccumulator();
  observeExposureReturn(returns, 1, 100, 110, {
    friction: 0,
    minExposure: -1,
    maxExposure: 1,
    maxEffectiveExposure: 250,
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

test("tradable exposure may drift beyond its target bound up to the effective limit", () => {
  const execution = {
    friction: 0.00175,
    minExposure: -100,
    maxExposure: 100,
    maxEffectiveExposure: 250,
    quoteLendRate: 0,
    quoteBorrowRate: 0,
    assetBorrowRate: 0,
  };
  const returns = createExposureReturnAccumulator();
  observeExposureReturn(returns, 100, 100, 99.9, execution);
  assert.equal(returns.liquidationCount, 0);
  assert.ok(returns.exposure > 100 && returns.exposure < 250);

  observeExposureReturn(returns, 100, 100, 99.5, execution);
  assert.equal(returns.liquidationCount, 1);
  assert.equal(returns.exposure, 0);
});
