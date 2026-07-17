import assert from "node:assert/strict";
import test from "node:test";
import {
  columnarVwKamaCandles,
  evaluateVwKamaOracle,
  prepareVwKamaOracle,
  type VwKamaParameters,
} from "../src/kama-signal-evaluator.js";
import { perfectMarginOracle } from "../src/perfect-margin-oracle.js";
import { prepareExposureValueOracle } from "../src/exposure-value-distillation.js";
import type { TradingCandle } from "../src/trading-api.js";
import {
  evaluateVwKamaCudaBatch,
  vwKamaCudaStatus,
} from "../src/vw-kama-cuda.js";

const MINUTE = 60_000;

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
  const candidates = [baseParameters(), featureParameters()];
  const valueOracle = prepareExposureValueOracle(candles.map((candle) => candle.close), {
    scoreStartIndex,
    friction: 0.00175,
    gridSize: 21,
    temperature: 0.01,
  });
  const common = {
    intervalMs: MINUTE,
    scoreStartIndex,
    oracleFriction: 0.00175,
    matchWindowMs: 2 * 60 * MINUTE,
    timingHalfLifeMs: 10 * MINUTE,
    warmupMultiple: 3,
    valueDistillation: { oracle: valueOracle, strategySigma: 0.15 },
  };
  const gpu = await evaluateVwKamaCudaBatch(columns, prepared, candidates, common);
  assert.equal(gpu.length, candidates.length);
  for (let index = 0; index < candidates.length; index += 1) {
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
      `candidate ${index} value-distillation loss drifted`,
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
