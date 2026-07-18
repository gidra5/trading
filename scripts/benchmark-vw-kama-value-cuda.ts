import { performance } from "node:perf_hooks";
import {
  columnarVwKamaCandles,
  evaluateVwKamaOracle,
  prepareVwKamaOracle,
  type VwKamaParameters,
} from "../packages/bot-algo/src/kama-signal-evaluator.js";
import { prepareExposureValueOracle } from "../packages/bot-algo/src/exposure-value-distillation.js";
import { perfectMarginOracle } from "../packages/bot-algo/src/perfect-margin-oracle.js";
import type { TradingCandle } from "../packages/bot-algo/src/trading-api.js";
import {
  evaluateVwKamaCudaBatch,
  evaluateVwKamaCudaFitnessCases,
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
} from "../packages/bot-algo/src/vw-kama-cuda.js";

const MINUTE = 60_000;

void run().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function run(): Promise<void> {
  const candidateCount = positiveInteger(process.argv[2] ?? "384", "candidate count");
  const candleCount = positiveInteger(process.argv[3] ?? "20000", "candle count");
  const gridSize = positiveInteger(process.argv[4] ?? "101", "grid size");
  const horizonSteps = positiveInteger(process.argv[5] ?? "300", "horizon steps");
  const caseCount = positiveInteger(process.argv[6] ?? "4", "case count");
  const strategyQuadraticScale = nonNegativeNumber(
    process.argv[7] ?? "0",
    "strategy quadratic scale",
  );
  const scoreStartIndex = Math.max(1_000, Math.floor(candleCount * 0.25));
  if (scoreStartIndex >= candleCount) throw new Error("Benchmark requires more than 1,000 candles.");
  const status = await vwKamaCudaStatus();
  if (!status.available) throw new Error(status.reason);

  const candles = syntheticCandles(candleCount);
  const columns = columnarVwKamaCandles(candles);
  const prices = candles.map((candle) => candle.close);
  const oracleOptions = {
    scoreStartIndex,
    horizonSteps,
    friction: 0.00175,
    gridSize,
    temperature: 0.001,
    opportunityEpsilon: 1e-6,
  };
  let started = performance.now();
  const cpuOracle = prepareExposureValueOracle(prices, oracleOptions);
  const cpuOracleMs = performance.now() - started;
  started = performance.now();
  const cudaOracle = await prepareExposureValueOracleCuda(prices, oracleOptions);
  const cudaOracleWallMs = performance.now() - started;

  const signalOracle = perfectMarginOracle(candles, {
    startingQuote: 1,
    leverage: 1,
    friction: 0.00175,
    eventMode: "close",
    maxPathCandles: 1,
  });
  const preparedSignalOracle = prepareVwKamaOracle(columns, scoreStartIndex, signalOracle);
  const candidates = Array.from({ length: candidateCount }, (_, index) => ({
    ...parameters(index),
    strategyTemperature: 0.001,
    strategyQuadraticScale,
    strategyQuadraticVolatilityMs: 60 * MINUTE,
  }));
  const common = {
    intervalMs: MINUTE,
    scoreStartIndex,
    oracleFriction: 0.00175,
    matchWindowMs: 120 * MINUTE,
    timingHalfLifeMs: 10 * MINUTE,
    warmupMultiple: 3,
    valueDistillation: {
      oracle: cpuOracle,
      strategyVolatilityScaling: false,
    },
  };
  await evaluateVwKamaCudaBatch(columns, preparedSignalOracle, candidates.slice(0, 1), {
    ...common,
    fitnessOnly: true,
  });
  started = performance.now();
  const fitness = await evaluateVwKamaCudaBatch(columns, preparedSignalOracle, candidates, {
    ...common,
    fitnessOnly: true,
  });
  const fitnessWallMs = performance.now() - started;
  started = performance.now();
  const repeatedFitness = await evaluateVwKamaCudaBatch(columns, preparedSignalOracle, candidates, {
    ...common,
    fitnessOnly: true,
  });
  const repeatedFitnessWallMs = performance.now() - started;
  started = performance.now();
  const diagnostics = await evaluateVwKamaCudaBatch(columns, preparedSignalOracle, candidates, common);
  const diagnosticsWallMs = performance.now() - started;
  const scheduledCases = Array.from({ length: caseCount }, () => ({
    candles: { ...columns },
    options: { ...common, fitnessOnly: true },
  }));
  await evaluateVwKamaCudaFitnessCases(scheduledCases, candidates.slice(0, 1));
  started = performance.now();
  for (const testCase of scheduledCases) {
    await evaluateVwKamaCudaBatch(
      testCase.candles,
      preparedSignalOracle,
      candidates,
      testCase.options,
    );
  }
  const sequentialCasesWallMs = performance.now() - started;
  started = performance.now();
  const scheduled = await evaluateVwKamaCudaFitnessCases(scheduledCases, candidates);
  const scheduledCasesWallMs = performance.now() - started;
  const maximumLossDrift = Math.max(...fitness.map((result, index) => Math.abs(
    result.distillationWeightedCrossEntropy
      - diagnostics[index]!.distillationWeightedCrossEntropy,
  )));
  let cpuCandidateWallMs: number | null = null;
  if (candidateCount <= 8) {
    started = performance.now();
    for (const candidate of candidates) {
      evaluateVwKamaOracle(columns, {
        ...common,
        scoreStartTime: candles[scoreStartIndex]!.openTime,
        parameters: candidate,
        preparedOracle: preparedSignalOracle,
        includeTrace: false,
      });
    }
    cpuCandidateWallMs = performance.now() - started;
  }

  console.log(JSON.stringify({
    device: status.device,
    workload: {
      candidateCount,
      candleCount,
      gridSize,
      horizonSteps,
      caseCount,
      scoreStartIndex,
      strategyQuadraticScale,
    },
    oracle: {
      cpuWallMs: cpuOracleMs,
      cudaWallMs: cudaOracleWallMs,
      cudaKernelMs: cudaOracle.kernelMs,
    },
    candidateEvaluation: {
      fitnessOnlyWallMs: fitnessWallMs,
      fitnessOnlyKernelMs: fitness[0]?.elapsedMs ?? 0,
      repeatedFitnessWallMs,
      repeatedFitnessKernelMs: repeatedFitness[0]?.elapsedMs ?? 0,
      diagnosticsWallMs,
      diagnosticsKernelMs: diagnostics[0]?.elapsedMs ?? 0,
      maximumSignalCount: Math.max(...diagnostics.map((result) => result.signalCount)),
      fitnessSpeedup: diagnosticsWallMs / fitnessWallMs,
      maximumLossDrift,
      cpuCandidateWallMs,
      sequentialCasesWallMs,
      scheduledCasesWallMs,
      caseSchedulingSpeedup: sequentialCasesWallMs / scheduledCasesWallMs,
      scheduledKernelMs: Math.max(...scheduled.map((results) => results[0]?.elapsedMs ?? 0)),
    },
  }, null, 2));
}

function parameters(index: number): VwKamaParameters {
  const confirmation = index % 3 !== 0;
  const meanReversion = index % 5 === 0;
  return {
    efficiencyMs: (30 + index % 90) * MINUTE,
    efficiencyVolumeEmaMs: (60 + index % 240) * MINUTE,
    efficiencyVolumePower: index % 4 === 0 ? 0 : 0.5 + index % 20 / 10,
    fastMs: (2 + index % 20) * MINUTE,
    slowMs: (120 + index % 600) * MINUTE,
    power: 0.8 + index % 30 / 10,
    volumeMs: (30 + index % 300) * MINUTE,
    volumeCap: 2 + index % 50 / 10,
    volumePower: index % 5 === 0 ? 0 : 0.4 + index % 20 / 10,
    rateEmaMs: (1 + index % 60) * MINUTE,
    deadbandBpsHour: 2 + index % 60,
    deadbandMode: (index % 3 === 0 ? "flat" : index % 3 === 1 ? "hold" : "hysteresis"),
    hysteresisReleaseRatio: index % 10 / 10,
    thresholdLookbackMs: (30 + index % 300) * MINUTE,
    thresholdNoiseMultiplier: index % 2 === 0 ? 0 : 0.5 + index % 20 / 10,
    buyMaxFraction: 0.5 + index % 6 / 10,
    sellMaxFraction: 0.5 + (index * 3) % 6 / 10,
    buySizingSigmaBpsHour: 20 + index % 200,
    sellSizingSigmaBpsHour: 20 + (index * 7) % 200,
    agreementMode: index % 2 === 0 ? "sizing" : "confidence",
    confirmationMix: confirmation ? 0.7 : 0,
    confirmationAccelerationLookbackMs: (30 + index % 120) * MINUTE,
    confirmationDistanceLookbackMs: (30 + index % 120) * MINUTE,
    confirmationAccelerationWeight: confirmation && index % 2 === 0 ? 1 : 0,
    confirmationDistanceWeight: confirmation && index % 4 === 0 ? 1 : 0,
    confirmationEmaMs: (60 + index % 240) * MINUTE,
    confirmationEmaWeight: confirmation && index % 5 === 0 ? 0.8 : 0,
    confirmationRsiMs: (15 + index % 60) * MINUTE,
    confirmationRsiWeight: confirmation && index % 7 === 0 ? 0.7 : 0,
    confirmationDmiMs: (15 + index % 60) * MINUTE,
    confirmationDmiWeight: confirmation && index % 11 === 0 ? 0.6 : 0,
    meanReversionSuppressionThreshold: meanReversion ? 1.2 : 0,
    meanReversionEfficiencyMs: (60 + index % 180) * MINUTE,
    meanReversionFastMs: 30 * MINUTE,
    meanReversionSlowMs: 180 * MINUTE,
    meanReversionVolatilityMs: 120 * MINUTE,
    meanReversionReversalThreshold: meanReversion ? 1.8 : 0,
  };
}

function syntheticCandles(count: number): TradingCandle[] {
  let price = 40_000;
  return Array.from({ length: count }, (_, index) => {
    const movement = Math.sin(index / 83) * 0.0008
      + Math.sin(index / 1_301) * 0.0012
      + (index % 701 < 9 ? 0.0025 : index % 997 < 7 ? -0.002 : 0);
    const open = price;
    price *= 1 + movement;
    const openTime = index * MINUTE;
    return {
      openTime,
      closeTime: openTime + MINUTE - 1,
      open,
      high: Math.max(open, price) * 1.0007,
      low: Math.min(open, price) * 0.9993,
      close: price,
      volume: 10 + 8 * (1 + Math.sin(index / 37)) + (index % 701 < 9 ? 60 : 0),
    };
  });
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${label} must be a positive integer.`);
  return parsed;
}

function finiteNumber(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be finite.`);
  return parsed;
}

function nonNegativeNumber(value: string, label: string): number {
  const parsed = finiteNumber(value, label);
  if (parsed < 0) throw new Error(`${label} must be non-negative.`);
  return parsed;
}
