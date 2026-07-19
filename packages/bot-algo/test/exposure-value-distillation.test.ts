import assert from "node:assert/strict";
import test from "node:test";
import {
  binnedConditionalExposureProbabilities,
  conditionalExposureProbabilities,
  conditionalQuadraticExposureProbabilities,
  createExposureReturnAccumulator,
  createExposureValueDistillationAccumulator,
  exposureValueOracleProbabilities,
  finalizeExposureReturn,
  finalizeExposureValueDistillation,
  fitConditionalQuadraticPolicy,
  observeExposureReturn,
  observeExposureValueDistillation,
  prepareExposureValueOracle,
  quadraticExponentialLogNormalizer,
  rebalanceEquityFactor,
  shareExposureValueOracle,
  strategyExposureProbabilities,
  strategyExposureQuadraticCoefficient,
  strategyExposureTemperatures,
  strategyExposureTransitionCrossEntropy,
  strategyExposureVolatilities,
  truncateExposureValueOracle,
  truncatedExponentialLogNormalizer,
} from "../src/exposure-value-distillation.js";
import {
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
} from "../src/vw-kama-cuda.js";

test("conditional exposure probabilities are the normalized transition policy used by the loss", () => {
  const grid = Float64Array.of(-1, 0, 1);
  const base = Float64Array.of(0.2, 0.3, 0.5);
  const current = -0.4;
  const friction = 0.01;
  const transitionLogScale = 7;
  const actual = conditionalExposureProbabilities(
    base,
    grid,
    current,
    friction,
    transitionLogScale,
  );
  const unnormalized = Array.from(grid, (target, index) =>
    base[index]! * rebalanceEquityFactor(current, target, friction) ** transitionLogScale);
  const total = unnormalized.reduce((sum, value) => sum + value, 0);

  assert.ok(Math.abs(Array.from(actual).reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs(actual[index]! - unnormalized[index]! / total) < 1e-12);
  }
});

test("conditional quadratic policy fit recovers shared predictor coefficients", () => {
  const grid = Float64Array.from({ length: 41 }, (_, index) => -2 + index / 10);
  const currentExposures = Float64Array.of(-1.7, -0.3, 0.8, 1.9);
  const linearCoefficient = 0.7;
  const quadraticCoefficient = -0.4;
  const friction = 0.012;
  const transitionLogScale = 5;
  const targets = new Float64Array(grid.length * currentExposures.length);
  for (let row = 0; row < currentExposures.length; row += 1) {
    conditionalQuadraticExposureProbabilities(
      grid,
      currentExposures[row]!,
      linearCoefficient,
      quadraticCoefficient,
      friction,
      transitionLogScale,
      targets.subarray(row * grid.length, (row + 1) * grid.length),
    );
  }
  const fit = fitConditionalQuadraticPolicy(grid, targets, currentExposures, {
    friction,
    transitionLogScale,
  });

  assert.equal(fit.converged, true);
  assert.ok(Math.abs(fit.linearCoefficient - linearCoefficient) < 1e-7, fit);
  assert.ok(Math.abs(fit.quadraticCoefficient - quadraticCoefficient) < 1e-7, fit);
  assert.ok(fit.klDivergence < 1e-10, fit);
});

test("probability-MSE quadratic fit minimizes pointwise distribution error", () => {
  const grid = Float64Array.from({ length: 101 }, (_, index) => -1 + index / 50);
  const targets = Float64Array.from(grid, (exposure) =>
    0.7 * Math.exp(-18 * (exposure - 0.55) ** 2)
      + 0.3 * Math.exp(1.2 * exposure));
  const mseFit = fitConditionalQuadraticPolicy(
    grid,
    targets,
    Float64Array.of(0),
    { friction: 0, transitionLogScale: 0, objective: "probability-mse" },
  );
  const klFit = fitConditionalQuadraticPolicy(
    grid,
    targets,
    Float64Array.of(0),
    { friction: 0, transitionLogScale: 0, objective: "forward-kl" },
  );
  const probabilityMse = (linearCoefficient: number, quadraticCoefficient: number) => {
    const predicted = conditionalQuadraticExposureProbabilities(
      grid,
      0,
      linearCoefficient,
      quadraticCoefficient,
      0,
      0,
    );
    const targetTotal = targets.reduce((sum, value) => sum + value, 0);
    return predicted.reduce((sum, probability, index) => {
      const residual = probability - targets[index]! / targetTotal;
      return sum + residual * residual / grid.length;
    }, 0);
  };

  assert.equal(mseFit.objective, "probability-mse");
  assert.equal(mseFit.converged, true, mseFit);
  assert.ok(Math.abs(probabilityMse(
    mseFit.linearCoefficient,
    mseFit.quadraticCoefficient,
  ) - mseFit.meanSquaredError) < 1e-14, mseFit);
  assert.ok(mseFit.meanSquaredError < probabilityMse(
    klFit.linearCoefficient,
    klFit.quadraticCoefficient,
  ), { mseFit, klFit });
  for (const [linearOffset, quadraticOffset] of [
    [-0.02, 0],
    [0.02, 0],
    [0, -0.02],
  ]) {
    assert.ok(mseFit.meanSquaredError <= probabilityMse(
      mseFit.linearCoefficient + linearOffset,
      mseFit.quadraticCoefficient + quadraticOffset,
    ), { mseFit, linearOffset, quadraticOffset });
  }
});

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
      holdingPeriodSteps: 17,
      valueHorizonSteps: prices.length - 1 - 300,
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
      "policyMeans",
      "policySecondMoments",
      "policyMeanLogRebalances",
      "policyEntropies",
      "averageRegrets",
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
    assert.deepEqual(gpu.path.exposures, cpu.path.exposures);
  }

  const oneStepOptions = {
    scoreStartIndex: 300,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 2_000,
    friction: 0.00175,
    gridSize: 21,
    minExposure: -1,
    maxExposure: 1,
    temperature: 0.001,
    initialExposure: 0,
  };
  const oneStepCpu = prepareExposureValueOracle(prices, oneStepOptions);
  const { oracle: oneStepGpu } = await prepareExposureValueOracleCuda(prices, oneStepOptions);
  assert.deepEqual(oneStepGpu.path.exposures, oneStepCpu.path.exposures);
  assert.ok(Math.abs(oneStepGpu.path.logReturn - oneStepCpu.path.logReturn) < 1e-10);
  assert.equal(oneStepGpu.path.terminalExposure, 0);
  assert.ok(oneStepGpu.path.totalReturn >= 0);

  const boundedOptions = {
    ...oneStepOptions,
    holdingPeriodSteps: 17,
    initialExposure: 0.3,
    terminalIndex: 1_700,
  };
  const boundedCpu = prepareExposureValueOracle(prices, boundedOptions);
  const { oracle: boundedGpu } = await prepareExposureValueOracleCuda(prices, boundedOptions);
  assert.deepEqual(boundedGpu.path.exposures, boundedCpu.path.exposures);
  assert.ok(Math.abs(boundedGpu.path.logReturn - boundedCpu.path.logReturn) < 1e-10);
  assert.equal(boundedGpu.path.exposures[boundedOptions.terminalIndex], 0);
  assert.equal(boundedGpu.path.equities[boundedOptions.terminalIndex + 1], 0);

  const monotonicOptions = {
    scoreStartIndex: 0,
    terminalIndex: 3,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 3,
    friction: 0.001,
    gridSize: 21,
    minExposure: -1,
    maxExposure: 1,
    temperature: 0.001,
  };
  const monotonicCpu = prepareExposureValueOracle([100, 110, 121, 133.1], monotonicOptions);
  const { oracle: monotonicGpu } = await prepareExposureValueOracleCuda(
    [100, 110, 121, 133.1],
    monotonicOptions,
  );
  for (const oracle of [monotonicCpu, monotonicGpu]) {
    assert.ok(oracle.path.totalReturn > 0);
    assert.ok(oracle.path.maxDrawdown <= monotonicOptions.friction * 2);
    assert.equal(oracle.path.exposures[monotonicOptions.terminalIndex], 0);
  }

  const driftOptions = {
    scoreStartIndex: 0,
    terminalIndex: 2,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 2,
    friction: 0.01,
    gridSize: 5,
    minExposure: -0.5,
    maxExposure: 0.5,
    maxEffectiveExposure: 2,
    temperature: 0.005,
  };
  const driftCpu = prepareExposureValueOracle([100, 110, 111], driftOptions);
  const { oracle: driftGpu } = await prepareExposureValueOracleCuda(
    [100, 110, 111],
    driftOptions,
  );
  assert.deepEqual(driftGpu.path.exposures, driftCpu.path.exposures);
  assert.equal(driftGpu.path.rebalanceCount, 2);
  assert.ok(Math.abs(driftGpu.path.logReturn - driftCpu.path.logReturn) < 1e-10);

  const leveragedOptions = {
    scoreStartIndex: 300,
    holdingPeriodSteps: 5,
    valueHorizonSteps: 2_000,
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
  assert.deepEqual(leveragedGpu.path.exposures, leveragedCpu.path.exposures);
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
  assert.equal(up.path.exposures[0], 1);
  assert.equal(down.path.exposures[0], -1);
  assert.ok(up.opportunities[0]! > 0);
  assert.ok(Math.abs(up.means[1]!) < 1e-7);
});

test("full-window oracle reports Q0 at the input exposure and liquidates at T", () => {
  const friction = 0.001;
  const oracle = prepareExposureValueOracle([100, 110, 121, 133.1], {
    scoreStartIndex: 0,
    terminalIndex: 3,
    initialExposure: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 3,
    friction,
    gridSize: 21,
    minExposure: -1,
    maxExposure: 1,
    temperature: 0.005,
  });

  assert.equal(oracle.path.startIndex, 0);
  assert.equal(oracle.path.terminalIndex, 3);
  assert.equal(oracle.path.initialExposure, 0);
  assert.equal(oracle.path.terminalExposure, 0);
  assert.equal(oracle.path.exposures[3], 0);
  assert.ok(oracle.path.totalReturn > 0);
  assert.ok(Math.abs(oracle.path.totalReturn - Math.expm1(oracle.path.logReturn)) < 1e-12);
  assert.ok(Math.abs(oracle.path.equities[3]! - Math.exp(oracle.path.logReturn)) < 1e-12);
  assert.ok(oracle.path.maxDrawdown <= friction * 2);
});

test("full-window horizon is measured from score start rather than the warmup prefix", () => {
  const options = {
    scoreStartIndex: 2,
    terminalIndex: 5,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 3,
    friction: 0.001,
    gridSize: 21,
    minExposure: -1,
    maxExposure: 1,
    temperature: 0.005,
  };
  const withWarmup = prepareExposureValueOracle([80, 90, 100, 110, 105, 120], options);
  const scoredOnly = prepareExposureValueOracle([100, 110, 105, 120], {
    ...options,
    scoreStartIndex: 0,
    terminalIndex: 3,
  });

  assert.deepEqual(
    Array.from(withWarmup.means.subarray(2)),
    Array.from(scoredOnly.means),
  );
  assert.deepEqual(
    Array.from(withWarmup.path.exposures.subarray(2)),
    Array.from(scoredOnly.path.exposures),
  );
  assert.ok(Math.abs(withWarmup.path.logReturn - scoredOnly.path.logReturn) < 1e-12);
});

test("full-window oracle stays in cash on a flat market and includes terminal closeout", () => {
  const flat = prepareExposureValueOracle([100, 100, 100, 100], {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 3,
    friction: 0.00175,
    gridSize: 21,
    temperature: 0.005,
  });
  assert.equal(flat.path.totalReturn, 0);
  assert.deepEqual(flat.path.exposures, new Float32Array(4));

  const initiallyLong = prepareExposureValueOracle([100, 100], {
    scoreStartIndex: 0,
    terminalIndex: 1,
    initialExposure: 1,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 1,
    friction: 0.00175,
    gridSize: 21,
    temperature: 0.005,
  });
  assert.equal(initiallyLong.path.exposures[1], 0);
  assert.ok(initiallyLong.path.totalReturn < 0);
  assert.ok(Math.abs(initiallyLong.path.totalReturn) <= 0.00175 * 1.01);
});

test("value grid must include cash as an exact state", () => {
  assert.throws(() => prepareExposureValueOracle([100, 101], {
    scoreStartIndex: 0,
    friction: 0,
    gridSize: 20,
    minExposure: -1,
    maxExposure: 1,
    temperature: 0.005,
  }), /exact zero/);
});

test("exposure value oracle retains selected H when the segment tail truncates the hold", () => {
  const oracle = prepareExposureValueOracle([100, 101, 102], {
    scoreStartIndex: 0,
    holdingPeriodSteps: 10,
    valueHorizonSteps: 10,
    friction: 0,
    gridSize: 3,
    temperature: 0.01,
  });

  assert.equal(oracle.holdingPeriodSteps, 10);
  assert.equal(oracle.valueHorizonSteps, 10);
});

test("exposure value holds the post-action portfolio for H before perfect continuation", () => {
  const prices = [100, 120, 60, 60];
  const oneStep = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 3,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });
  const twoSteps = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: 2,
    valueHorizonSteps: 3,
    friction: 0,
    gridSize: 21,
    temperature: 0.005,
  });

  assert.equal(oneStep.modalExposures[0], 1);
  assert.equal(twoSteps.modalExposures[0], -1);
  assert.equal(twoSteps.path.exposures[0], -1);
  assert.ok(Math.abs(twoSteps.path.exposures[1]! + 1.5) < 1e-7);
});

test("value horizon T caps final equity independently from holding period H", () => {
  const prices = [100, 90, 200, 50];
  const shortHorizon = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 1,
    friction: 0.1,
    gridSize: 21,
    temperature: 0.01,
  });
  const longHorizon = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 3,
    friction: 0.1,
    gridSize: 21,
    temperature: 0.01,
  });

  assert.equal(shortHorizon.modalExposures[0], -1);
  assert.equal(longHorizon.modalExposures[0], 0);
  assert.equal(shortHorizon.holdingPeriodSteps, longHorizon.holdingPeriodSteps);
  assert.notEqual(shortHorizon.valueHorizonSteps, longHorizon.valueHorizonSteps);
});

test("a scored oracle prefix can keep values from post-window candles", () => {
  const options = {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 3,
    friction: 0.1,
    gridSize: 21,
    temperature: 0.01,
  };
  const extended = truncateExposureValueOracle(
    prepareExposureValueOracle([100, 90, 200, 50], options),
    2,
  );
  const truncated = prepareExposureValueOracle([100, 90], {
    ...options,
    valueHorizonSteps: 1,
  });

  assert.equal(extended.means.length, 2);
  assert.equal(extended.modalExposures[0], 0);
  assert.equal(truncated.modalExposures[0], -1);
});

test("H-step oracle values let the portfolio drift without intermediate rebalancing", () => {
  const prices = [100, 98, 95];
  const friction = 0.00175;
  const temperature = 0.01;
  const oracle = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: 2,
    valueHorizonSteps: 2,
    friction,
    gridSize: 5,
    temperature,
    includeProbabilities: true,
  });
  const values = Array.from(oracle.grid, (exposure) => {
    return Math.log(1 - exposure + exposure * prices[2]! / prices[0]!);
  });
  const maximum = Math.max(...values);
  const weights = values.map((value) => Math.exp((value - maximum) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  const actual = exposureValueOracleProbabilities(oracle, 0);
  for (let index = 0; index < weights.length; index += 1) {
    assert.ok(Math.abs(actual[index]! - weights[index]! / total) < 1e-6);
  }
});

test("the full-window oracle can keep an off-grid drifted exposure without trading", () => {
  const oracle = prepareExposureValueOracle([100, 110, 111], {
    scoreStartIndex: 0,
    terminalIndex: 2,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 2,
    friction: 0.01,
    gridSize: 5,
    minExposure: -0.5,
    maxExposure: 0.5,
    maxEffectiveExposure: 2,
    temperature: 0.005,
  });
  const drifted = 0.5 * 1.1 / (0.5 + 0.5 * 1.1);

  assert.equal(oracle.path.exposures[0], 0.5);
  assert.ok(Math.abs(oracle.path.exposures[1]! - drifted) < 1e-7);
  assert.equal(oracle.path.rebalanceCount, 2);
  assert.ok(oracle.path.totalReturn > 0);
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

test("transition-aware oracle statistics learn every current-target exposure pair", () => {
  const temperature = 0.02;
  const friction = 0.01;
  const oracle = prepareExposureValueOracle([100, 110], {
    scoreStartIndex: 0,
    friction,
    gridSize: 3,
    temperature,
    includeProbabilities: true,
    opportunityEpsilon: 1e-6,
  });
  const postAction = exposureValueOracleProbabilities(oracle, 0);
  let mean = 0;
  let secondMoment = 0;
  let meanLogRebalance = 0;
  let entropy = 0;
  let averageRegret = 0;
  for (const current of oracle.grid) {
    const logits = Array.from(oracle.grid, (target, index) =>
      Math.log(postAction[index]!)
        + Math.log(rebalanceEquityFactor(current, target, friction)) / temperature);
    const maximum = Math.max(...logits);
    const probabilities = logits.map((logit) => Math.exp(logit - maximum));
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    const conditionalValues = Array.from(oracle.grid, (target, index) =>
      temperature * Math.log(postAction[index]!)
        + Math.log(rebalanceEquityFactor(current, target, friction)));
    averageRegret += Math.max(...conditionalValues)
      - conditionalValues.reduce((sum, value) => sum + value, 0) / conditionalValues.length;
    for (let index = 0; index < oracle.grid.length; index += 1) {
      const probability = probabilities[index]! / total;
      const target = oracle.grid[index]!;
      const logRebalance = Math.log(rebalanceEquityFactor(current, target, friction));
      mean += probability * target / oracle.grid.length;
      secondMoment += probability * target * target / oracle.grid.length;
      meanLogRebalance += probability * logRebalance / oracle.grid.length;
      if (probability > 0) entropy -= probability * Math.log(probability) / oracle.grid.length;
    }
  }
  averageRegret /= oracle.grid.length;
  assert.ok(Math.abs(oracle.policyMeans[0]! - mean) < 1e-6);
  assert.ok(Math.abs(oracle.policySecondMoments[0]! - secondMoment) < 1e-6);
  assert.ok(Math.abs(oracle.policyMeanLogRebalances[0]! - meanLogRebalance) < 1e-6);
  assert.ok(Math.abs(oracle.policyEntropies[0]! - entropy) < 1e-6);
  assert.ok(Math.abs(oracle.averageRegrets[0]! - averageRegret) < 1e-6);
  assert.ok(Math.abs(oracle.weights[0]! - (averageRegret + 1e-6)) < 1e-6);
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

test("distillation loss mix exposes entropy and both mutual-information estimators", () => {
  const prices = [100, 110, 90, 115, 85, 120];
  const oracle = prepareExposureValueOracle(prices, {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 1,
    friction: 0,
    gridSize: 21,
    temperature: 0.002,
    includeProbabilities: true,
  });
  const config = {
    entropyGapLambda: 0.3,
    stateMutualInformationLambda: 0.2,
    oracleMutualInformationLambda: 0.4,
    oracleMutualInformationMode: "precise" as const,
    mutualInformationBins: 7,
  };
  const accumulator = createExposureValueDistillationAccumulator(config, oracle.grid.length);
  for (let index = 0; index < prices.length - 1; index += 1) {
    const rate = prices[index + 1]! > prices[index]! ? 100 : -100;
    observeExposureValueDistillation(accumulator, oracle, index, rate, 3_600_000, 0.01);
  }
  const metrics = finalizeExposureValueDistillation(accumulator);

  assert.ok(metrics.entropyGap >= 0);
  assert.ok(metrics.stateMutualInformation > 0 && metrics.stateMutualInformation <= 1);
  assert.ok(metrics.oracleMutualInformation > 0 && metrics.oracleMutualInformation <= 1);
  assert.equal(metrics.oracleMutualInformationMode, "precise");
  assert.ok(Math.abs(metrics.mixedLoss - (
    metrics.crossEntropy
    + config.entropyGapLambda * metrics.entropyGap
    - config.stateMutualInformationLambda * metrics.stateMutualInformation
    - config.oracleMutualInformationLambda * metrics.oracleMutualInformation
  )) < 1e-12);

  const disabled = createExposureValueDistillationAccumulator({}, oracle.grid.length);
  observeExposureValueDistillation(disabled, oracle, 0, 100, 3_600_000, 0.01);
  const disabledMetrics = finalizeExposureValueDistillation(disabled);
  assert.equal(disabledMetrics.mixedLoss, disabledMetrics.crossEntropy);
  assert.equal(disabledMetrics.entropyGap, 0);
  assert.equal(disabledMetrics.stateMutualInformation, 0);
  assert.equal(disabledMetrics.oracleMutualInformation, 0);
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

test("strategy normal mixture is normalized and its transition loss matches a full conditional map", () => {
  const oracle = prepareExposureValueOracle([100, 102, 99], {
    scoreStartIndex: 0,
    friction: 0.01,
    gridSize: 9,
    temperature: 0.03,
    includeProbabilities: true,
  });
  const rate = 40;
  const holdingMs = 3_600_000;
  const temperature = 0.05;
  const quadratic = -0.2;
  const center = 0.35;
  const mixture = 0.4;
  const sigma = 0.3;
  const strategyBase = strategyExposureProbabilities(
    oracle.grid,
    rate,
    holdingMs,
    temperature,
    quadratic,
    center,
    mixture,
    sigma,
  );
  assert.ok(Math.abs(strategyBase.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);

  const oracleBase = exposureValueOracleProbabilities(oracle, 0);
  let expected = 0;
  for (const current of oracle.grid) {
    const oracleConditional = conditionalExposureProbabilities(
      oracleBase,
      oracle.grid,
      current,
      oracle.execution.friction,
      1 / oracle.temperature,
    );
    const strategyConditional = conditionalExposureProbabilities(
      strategyBase,
      oracle.grid,
      current,
      oracle.execution.friction,
      1 / temperature,
    );
    for (let index = 0; index < oracle.grid.length; index += 1) {
      expected -= oracleConditional[index]!
        * Math.log(Math.max(Number.MIN_VALUE, strategyConditional[index]!))
        / oracle.grid.length;
    }
  }
  const actual = strategyExposureTransitionCrossEntropy(
    oracle,
    0,
    rate,
    holdingMs,
    temperature,
    quadratic,
    1,
    center,
    mixture,
    sigma,
  );
  assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} != ${expected}`);
});

test("conditional policy bins match a grid-squared reference", () => {
  const grid = new Float64Array([-1, -0.5, 0, 0.5, 1]);
  const base = strategyExposureProbabilities(grid, 35, 3_600_000, 0.03, -0.2);
  const friction = 0.01;
  const transitionScale = 20;
  const bins = 3;
  const actual = binnedConditionalExposureProbabilities(
    base,
    grid,
    friction,
    transitionScale,
    bins,
  );
  const expected = new Float64Array(bins);
  for (const current of grid) {
    const weights = Array.from(grid, (target, index) =>
      base[index]! * rebalanceEquityFactor(current, target, friction) ** transitionScale);
    const total = weights.reduce((sum, value) => sum + value, 0);
    for (let index = 0; index < grid.length; index += 1) {
      const bin = Math.min(bins - 1, Math.floor(index * bins / grid.length));
      expected[bin] += weights[index]! / total / grid.length;
    }
  }
  for (let bin = 0; bin < bins; bin += 1) {
    assert.ok(Math.abs(actual[bin]! - expected[bin]!) < 1e-12);
  }
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
    holdingPeriodSteps: 1,
    temperature: 0.01,
  });
  const quarterHourly = strategyExposureTemperatures([100, 101], {
    intervalMs: 900_000,
    holdingPeriodSteps: 4,
    temperature: 0.01,
  });
  assert.ok(Math.abs(hourly[0]! - 0.01) < 1e-8);
  assert.ok(Math.abs(quarterHourly[0]! - 0.02) < 1e-8);

  const exact = strategyExposureTemperatures([100, 110, 99], {
    intervalMs: 60_000,
    holdingPeriodSteps: 2,
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
    holdingPeriodSteps: 10,
    temperature: 0.01,
    scaleByVolatility: true,
  });
  const high = strategyExposureTemperatures(prices(0.01), {
    intervalMs: 60_000,
    holdingPeriodSteps: 10,
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
