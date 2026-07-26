import assert from "node:assert/strict";
import test from "node:test";
import {
  columnarVwKamaCandles,
  evaluateVwKamaOracle,
  prepareVwKamaOracle,
  type VwKamaParameters,
} from "../src/kama-signal-evaluator.js";
import { perfectMarginOracle } from "../src/perfect-margin-oracle.js";
import {
  conditionalExposureProbabilities,
  DEFAULT_EXPOSURE_VALUE_DISTILLATION_LOSS,
  exposureHoldingFeasibleInterval,
  prepareExposureValueOracle,
} from "../src/exposure-value-distillation.js";
import type { TradingCandle } from "../src/trading-api.js";
import {
  directOracleDiagnosticsCuda,
  evaluateVwKamaCudaBatch,
  evaluateVwKamaCudaFitnessCases,
  exposureHoldingCutoffsCuda,
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
} from "../src/vw-kama-cuda.js";

const MINUTE = 60_000;

test("CUDA mandatory-hold cutoffs match the causal CPU bisection", async (context) => {
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    context.skip(status.reason);
    return;
  }
  const prices = Float64Array.from(
    { length: 200 },
    (_, index) => 100 + 35 * Math.sin(index / 5) + 0.04 * index,
  );
  const execution = {
    friction: 0.00175,
    minExposure: -10,
    maxExposure: 10,
    maxEffectiveExposure: 12,
    quoteLendRate: 1e-7,
    quoteBorrowRate: 2e-7,
    assetBorrowRate: 3e-7,
  };
  const count = 180;
  const holdingPeriodSteps = 20;
  const gpu = await exposureHoldingCutoffsCuda(
    prices,
    count,
    holdingPeriodSteps,
    execution,
  );
  for (let index = 0; index < count; index += 1) {
    const cpu = exposureHoldingFeasibleInterval(
      prices,
      index,
      holdingPeriodSteps,
      execution,
    );
    assert.ok(Math.abs(gpu.cutoffLowers[index]! - cpu.lower) < 1e-7);
    assert.ok(Math.abs(gpu.cutoffUppers[index]! - cpu.upper) < 1e-7);
  }
});

test("CUDA direct-oracle diagnostics match the transition-aware CPU surface", async (context) => {
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    context.skip(status.reason);
    return;
  }
  const actions = Float64Array.from({ length: 21 }, (_, index) => index - 10);
  const currents = Float64Array.from({ length: 25 }, (_, index) => index - 12);
  const count = 32;
  const probabilities = new Float32Array(count * actions.length);
  const cutoffLowers = new Float64Array(count);
  const cutoffUppers = new Float64Array(count);
  for (let row = 0; row < count; row += 1) {
    let total = 0;
    for (let action = 0; action < actions.length; action += 1) {
      const value = Math.exp(-Math.abs(action - row % actions.length) / 3)
        + action % 4 * 0.001;
      probabilities[row * actions.length + action] = value;
      total += value;
    }
    for (let action = 0; action < actions.length; action += 1) {
      probabilities[row * actions.length + action] /= total;
    }
    cutoffLowers[row] = -10 + row % 4;
    cutoffUppers[row] = 10 - row % 5;
  }
  const options = {
    visibleLower: -5,
    visibleUpper: 5,
    friction: 0.00175,
    transitionLogScale: 100,
    distanceEpsilon: 1e-6,
  };
  const gpu = await directOracleDiagnosticsCuda(
    probabilities,
    actions,
    currents,
    cutoffLowers,
    cutoffUppers,
    options,
  );
  const visibleActions = Float64Array.from(actions.filter((action) =>
    action >= options.visibleLower && action <= options.visibleUpper));
  const visibleCurrents = currents.filter((current) =>
    current >= options.visibleLower && current <= options.visibleUpper);
  for (let row = 0; row < count; row += 1) {
    const base = Float64Array.from(visibleActions, (action) => {
      const actionIndex = actions.indexOf(action);
      return action >= cutoffLowers[row]! && action <= cutoffUppers[row]!
        ? probabilities[row * actions.length + actionIndex]!
        : 0;
    });
    let expectedEntropy = 0;
    let expectedDisplacement = 0;
    let expectedDistance = 0;
    for (const current of visibleCurrents) {
      const target = conditionalExposureProbabilities(
        base,
        visibleActions,
        current,
        options.friction,
        options.transitionLogScale,
      );
      for (let action = 0; action < target.length; action += 1) {
        const probability = target[action]!;
        if (probability > 0) expectedEntropy -= probability * Math.log(probability);
        const displacement = visibleActions[action]! - current;
        expectedDisplacement += probability * displacement;
        expectedDistance += probability * Math.abs(displacement);
      }
    }
    expectedEntropy /= visibleCurrents.length;
    const expectedAdvice = expectedDisplacement
      / (expectedDistance + options.distanceEpsilon);
    assert.ok(Math.abs(gpu.entropies[row]! - expectedEntropy) < 1e-5);
    assert.ok(Math.abs(gpu.distanceImbalances[row]! - expectedAdvice) < 1e-5);
  }
});

test("CUDA rolling-horizon oracle matches CPU statistics through liquidations", async (context) => {
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    context.skip(status.reason);
    return;
  }
  const prices = Float64Array.from([
    100, 90, 105, 80, 120, 75, 130, 85, 110, 70,
    140, 95, 100, 60, 150, 100, 90, 120, 80, 100,
  ]);
  const options = {
    scoreStartIndex: 1,
    holdingPeriodSteps: 5,
    valueHorizonSteps: 15,
    friction: 0.00175,
    gridSize: 21,
    minExposure: -10,
    maxExposure: 10,
    maxEffectiveExposure: 12,
    temperature: 0.01,
    includeProbabilities: true,
  };
  const cpu = prepareExposureValueOracle(prices, options);
  const gpu = (await prepareExposureValueOracleCuda(prices, options)).oracle;
  for (const field of [
    "means", "secondMoments", "entropies", "policyMeans", "policySecondMoments",
    "policyMeanLogRebalances", "policyEntropies", "averageRegrets", "weights",
    "opportunities",
  ] as const) {
    for (let index = options.scoreStartIndex; index < prices.length; index += 1) {
      assert.ok(
        Math.abs(cpu[field][index]! - gpu[field][index]!) < 1e-5,
        `${field}[${index}] drifted: CPU ${cpu[field][index]}, CUDA ${gpu[field][index]}`,
      );
    }
  }
  assert.ok(Math.abs(cpu.path.logReturn - gpu.path.logReturn) < 1e-10);
});

test("CUDA distribution-only oracle is deterministic and faithful to the full statistics path", async (context) => {
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    context.skip(status.reason);
    return;
  }
  const prices = Float64Array.from({ length: 600 }, (_, index) =>
    100 * Math.exp(index * 0.00001 + Math.sin(index / 17) * 0.002));
  const options = {
    scoreStartIndex: 0,
    holdingPeriodSteps: 5,
    valueHorizonSteps: 60,
    friction: 0.00175,
    gridSize: 21,
    minExposure: -10,
    maxExposure: 10,
    maxEffectiveExposure: 12,
    temperature: 0.01,
    quoteLendRate: 0.000001,
    quoteBorrowRate: 0.000002,
    assetBorrowRate: 0.000003,
    includeProbabilities: true,
    includePath: false,
  };
  const full = (await prepareExposureValueOracleCuda(prices, options)).oracle;
  const first = (await prepareExposureValueOracleCuda(prices, {
    ...options,
    distributionOnly: true,
  })).oracle;
  const second = (await prepareExposureValueOracleCuda(prices, {
    ...options,
    distributionOnly: true,
  })).oracle;
  assert.equal(first.probabilities!.length, full.probabilities!.length);
  let maximumDifference = 0;
  let maximumRowKlDivergence = 0;
  for (let row = 0; row < prices.length; row += 1) {
    let rowKlDivergence = 0;
    for (let action = 0; action < options.gridSize; action += 1) {
      const index = row * options.gridSize + action;
      const expected = full.probabilities![index]!;
      const actual = first.probabilities![index]!;
      maximumDifference = Math.max(maximumDifference, Math.abs(expected - actual));
      if (expected > 0 && actual > 0) {
        rowKlDivergence += expected * Math.log(expected / actual);
      }
    }
    maximumRowKlDivergence = Math.max(maximumRowKlDivergence, rowKlDivergence);
  }
  // The compact production path evaluates the exact recurrence in Float32;
  // the full diagnostic path retains Float64 intermediates.
  assert.ok(maximumDifference < 2e-5, `maximum probability drifted by ${maximumDifference}`);
  assert.ok(
    maximumRowKlDivergence < 1e-6,
    `maximum row KL divergence drifted by ${maximumRowKlDivergence}`,
  );
  assert.deepEqual(second.probabilities, first.probabilities);
  assert.equal(first.path.logReturn, 0);
  assert.ok(first.path.exposures.every((value) => value === 0));

  const unevenOptions = {
    ...options,
    valueHorizonSteps: 62,
  };
  const unevenFull = (await prepareExposureValueOracleCuda(
    prices,
    unevenOptions,
  )).oracle;
  const unevenCompact = (await prepareExposureValueOracleCuda(prices, {
    ...unevenOptions,
    distributionOnly: true,
  })).oracle;
  let unevenMaximumDifference = 0;
  for (let index = 0; index < unevenFull.probabilities!.length; index += 1) {
    unevenMaximumDifference = Math.max(
      unevenMaximumDifference,
      Math.abs(
        unevenFull.probabilities![index]!
          - unevenCompact.probabilities![index]!,
      ),
    );
  }
  assert.ok(
    unevenMaximumDifference < 2e-5,
    `non-fused probability drifted by ${unevenMaximumDifference}`,
  );
});

test("CUDA evaluation tracks the Float64 CPU evaluator", async (context) => {
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    context.skip(status.reason);
    return;
  }
  const candles = syntheticCandles(24_000);
  const scoreStartIndex = 8_000;
  const oracle = perfectMarginOracle(candles, {
    startingQuote: 1,
    leverage: 1,
    friction: 0.00175,
    eventMode: "close",
    maxPathCandles: 1,
  });
  const columns = columnarVwKamaCandles(candles);
  const prepared = prepareVwKamaOracle(columns, scoreStartIndex, oracle);
  const candidates = [baseParameters(), featureParameters()].map((parameters, index) => ({
    ...parameters,
    strategyTemperature: index === 0 ? 0.001 : 0.003,
    strategyQuadraticScale: index === 0 ? 200_000 : 2_000,
    strategyQuadraticVolatilityMs: index === 0 ? 120 * MINUTE : 30 * MINUTE,
    strategyNormalMixture: index === 0 ? 0 : 0.35,
    strategyNormalSigma: 0.4,
  }));
  const valueOracle = prepareExposureValueOracle(candles.map((candle) => candle.close), {
    scoreStartIndex,
    holdingPeriodSteps: 30,
    valueHorizonSteps: 90,
    friction: 0.00175,
    gridSize: 21,
    temperature: 0.01,
    includeProbabilities: true,
  });
  const common = {
    intervalMs: MINUTE,
    scoreStartIndex,
    oracleFriction: 0.00175,
    matchWindowMs: 2 * 60 * MINUTE,
    timingHalfLifeMs: 10 * MINUTE,
    warmupMultiple: 3,
    valueDistillation: {
      oracle: valueOracle,
      strategyVolatilityScaling: true,
      lossConfig: {
        ...DEFAULT_EXPOSURE_VALUE_DISTILLATION_LOSS,
        entropyGapLambda: 0.2,
        stateMutualInformationLambda: 0.1,
        oracleMutualInformationLambda: 0.3,
      },
    },
  };
  const gpu = await evaluateVwKamaCudaBatch(columns, prepared, candidates, common);
  const fitnessOnly = await evaluateVwKamaCudaBatch(columns, prepared, candidates, {
    ...common,
    fitnessOnly: true,
  });
  const baseLossOnly = await evaluateVwKamaCudaBatch(columns, prepared, candidates, {
    ...common,
    fitnessOnly: true,
    valueDistillation: {
      ...common.valueDistillation,
      lossConfig: DEFAULT_EXPOSURE_VALUE_DISTILLATION_LOSS,
    },
  });
  assert.ok(baseLossOnly.every((result) =>
    Number.isFinite(result.distillationWeightedCrossEntropy)));
  const [scheduledFitness] = await evaluateVwKamaCudaFitnessCases([{
    candles: columns,
    options: { ...common, fitnessOnly: true },
  }], candidates);
  assert.equal(gpu.length, candidates.length);
  for (let index = 0; index < candidates.length; index += 1) {
    assert.ok(
      Math.abs(baseLossOnly[index]!.distillationWeightedCrossEntropy
        - gpu[index]!.distillationWeightedCrossEntropy)
        <= Math.max(1e-3, Math.abs(gpu[index]!.distillationWeightedCrossEntropy) * 5e-4),
      `${index}: fast base ${baseLossOnly[index]!.distillationWeightedCrossEntropy}, diagnostic ${gpu[index]!.distillationWeightedCrossEntropy}`,
    );
    assert.ok(Math.abs(
      fitnessOnly[index]!.distillationWeightedCrossEntropy
        - gpu[index]!.distillationWeightedCrossEntropy,
    ) < 1e-6, `${index}: resident ${fitnessOnly[index]!.distillationWeightedCrossEntropy}, direct ${gpu[index]!.distillationWeightedCrossEntropy}`);
    assert.equal(
      scheduledFitness![index]!.distillationWeightedCrossEntropy,
      fitnessOnly[index]!.distillationWeightedCrossEntropy,
    );
    assert.ok(Math.abs(
      fitnessOnly[index]!.distillationWeight - gpu[index]!.distillationWeight,
    ) < 1e-9);
    assert.ok(Math.abs(
      fitnessOnly[index]!.distillationWeightedOracleEntropy
        - gpu[index]!.distillationWeightedOracleEntropy,
    ) < 1e-6);
    const cpu = evaluateVwKamaOracle(columns, {
      ...common,
      scoreStartTime: candles[scoreStartIndex]!.openTime,
      parameters: candidates[index]!,
      preparedOracle: prepared,
      includeTrace: false,
    });
    assert.ok(
      Math.abs(gpu[index]!.stateCredit / gpu[index]!.stateCount - cpu.metrics.exposureAgreement) < 0.025,
      `candidate ${index} exposure agreement drifted`,
    );
    const transitionDrift = Math.abs(gpu[index]!.signalCount - cpu.metrics.signalCount);
    assert.ok(
      transitionDrift <= Math.max(3, cpu.metrics.signalCount * 0.08),
      `candidate ${index} transition count drifted by ${transitionDrift}`,
    );
    const cpuDistillation = cpu.metrics.valueDistillation!;
    const gpuCrossEntropy = gpu[index]!.distillationWeightedCrossEntropy
      / gpu[index]!.distillationWeight;
    assert.ok(
      Math.abs(gpuCrossEntropy - cpuDistillation.crossEntropy)
        <= Math.max(0.02, cpuDistillation.crossEntropy * 0.05),
      `candidate ${index} value-distillation loss drifted: CPU ${cpuDistillation.crossEntropy}, GPU ${gpuCrossEntropy}`,
    );
    assert.ok(Math.abs(
      gpu[index]!.distillationWeightedEntropyGap / gpu[index]!.distillationWeight
        - cpuDistillation.entropyGap,
    ) < 0.02);
    assert.ok(Math.abs(
      gpu[index]!.distillationStateMutualInformation - cpuDistillation.stateMutualInformation,
    ) < 0.02, `candidate ${index} state MI drifted: CPU ${cpuDistillation.stateMutualInformation}, GPU ${gpu[index]!.distillationStateMutualInformation}`);
    assert.ok(Math.abs(
      gpu[index]!.distillationOracleMutualInformation - cpuDistillation.oracleMutualInformation,
    ) < 0.02, `candidate ${index} oracle MI drifted: CPU ${cpuDistillation.oracleMutualInformation}, GPU ${gpu[index]!.distillationOracleMutualInformation}`);
    assert.ok(
      Math.abs(gpu[index]!.distillationMixedLoss - cpuDistillation.mixedLoss)
        <= Math.max(0.03, Math.abs(cpuDistillation.mixedLoss) * 0.05),
      `candidate ${index} mixed loss drifted: CPU ${cpuDistillation.mixedLoss}, GPU ${gpu[index]!.distillationMixedLoss}`,
    );
    assert.ok(
      Math.abs(Math.log(gpu[index]!.strategyFinalEquity)
        - Math.log(cpuDistillation.returns.strategy.equity)) < 0.08,
      `candidate ${index} strategy return drifted`,
    );
    assert.ok(
      Math.abs(Math.log(gpu[index]!.oracleFinalEquity)
        - Math.log(cpuDistillation.returns.oracle.equity)) < 0.02,
      `candidate ${index} oracle return drifted`,
    );
  }

  const preciseOptions = {
    ...common,
    fitnessOnly: true,
    valueDistillation: {
      ...common.valueDistillation,
      lossConfig: {
        ...common.valueDistillation.lossConfig,
        oracleMutualInformationMode: "precise" as const,
        mutualInformationBins: 7,
      },
    },
  };
  const [preciseGpu] = await evaluateVwKamaCudaBatch(
    columns,
    prepared,
    candidates.slice(0, 1),
    preciseOptions,
  );
  const [preciseScheduled] = await evaluateVwKamaCudaFitnessCases([{
    candles: columns,
    options: preciseOptions,
  }], candidates.slice(0, 1));
  const preciseCpu = evaluateVwKamaOracle(columns, {
    ...preciseOptions,
    scoreStartTime: candles[scoreStartIndex]!.openTime,
    parameters: candidates[0]!,
    preparedOracle: prepared,
    includeTrace: false,
  }).metrics.valueDistillation!;
  assert.ok(Math.abs(
    preciseGpu!.distillationOracleMutualInformation - preciseCpu.oracleMutualInformation,
  ) < 0.03);
  assert.equal(
    preciseScheduled![0]!.distillationOracleMutualInformation,
    preciseGpu!.distillationOracleMutualInformation,
  );
  assert.ok(
    Math.abs(preciseGpu!.distillationMixedLoss - preciseCpu.mixedLoss)
      <= Math.max(0.04, Math.abs(preciseCpu.mixedLoss) * 0.05),
  );
});

function baseParameters(): VwKamaParameters {
  return {
    efficiencyMs: 45 * MINUTE,
    efficiencyVolumeEmaMs: 90 * MINUTE,
    efficiencyVolumePower: 1.2,
    fastMs: 5 * MINUTE,
    slowMs: 180 * MINUTE,
    power: 1.7,
    volumeMs: 120 * MINUTE,
    volumeCap: 4,
    volumePower: 1.1,
    deadbandBpsHour: 8,
    deadbandMode: "hysteresis",
    hysteresisReleaseRatio: 0.3,
    thresholdLookbackMs: 90 * MINUTE,
    thresholdNoiseMultiplier: 1.2,
    buyMaxFraction: 0.8,
    sellMaxFraction: 0.7,
    buySizingSigmaBpsHour: 120,
    sellSizingSigmaBpsHour: 100,
    agreementMode: "sizing",
  };
}

function featureParameters(): VwKamaParameters {
  return {
    ...baseParameters(),
    rateMode: "log",
    rateEmaMs: 30 * MINUTE,
    thresholdNoiseResponse: "inverse",
    thresholdInverseMaxBpsHour: 30,
    thresholdInverseNoiseScaleBpsHour: 20,
    agreementMode: "confidence",
    confirmationMix: 0.75,
    confirmationMinQuality: 0.25,
    confirmationAccelerationLookbackMs: 45 * MINUTE,
    confirmationDistanceLookbackMs: 60 * MINUTE,
    confirmationAccelerationWeight: 1.2,
    confirmationDistanceWeight: 0.8,
    confirmationBias: 0.1,
    confirmationEmaMs: 180 * MINUTE,
    confirmationEmaThresholdBpsHour: 5,
    confirmationEmaWeight: 0.9,
    confirmationEmaGateStrength: 0.25,
    confirmationRsiMs: 30 * MINUTE,
    confirmationRsiThreshold: 4,
    confirmationRsiWeight: 0.7,
    confirmationDmiMs: 30 * MINUTE,
    confirmationDmiWeight: 0.6,
    confirmationAdxThreshold: 18,
    signalFrictionFraction: 0.35,
    meanReversionSuppressionThreshold: 1.2,
    meanReversionEfficiencyMs: 120 * MINUTE,
    meanReversionFastMs: 60 * MINUTE,
    meanReversionSlowMs: 240 * MINUTE,
    meanReversionVolatilityMs: 180 * MINUTE,
    meanReversionReversalThreshold: 1.8,
  };
}

function syntheticCandles(count: number): TradingCandle[] {
  let price = 40_000;
  return Array.from({ length: count }, (_, index) => {
    const wave = Math.sin(index / 83) * 0.0008 + Math.sin(index / 1_301) * 0.0012;
    const impulse = index % 701 < 9 ? 0.0025 : index % 997 < 7 ? -0.002 : 0;
    const open = price;
    price *= 1 + wave + impulse;
    const high = Math.max(open, price) * 1.0007;
    const low = Math.min(open, price) * 0.9993;
    const openTime = index * MINUTE;
    return {
      openTime,
      closeTime: openTime + MINUTE - 1,
      open,
      high,
      low,
      close: price,
      volume: 10 + 8 * (1 + Math.sin(index / 37)) + (index % 701 < 9 ? 60 : 0),
    };
  });
}
