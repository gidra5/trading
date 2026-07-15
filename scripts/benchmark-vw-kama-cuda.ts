import { performance } from "node:perf_hooks";
import {
  columnarVwKamaCandles,
  evaluateVwKamaOracle,
  prepareVwKamaOracle,
  vwKamaScore,
  type VwKamaParameters,
} from "../packages/bot-algo/src/kama-signal-evaluator.js";
import { perfectMarginOracle } from "../packages/bot-algo/src/perfect-margin-oracle.js";
import type { TradingCandle } from "../packages/bot-algo/src/trading-api.js";
import {
  evaluateVwKamaCudaBatch,
  vwKamaCudaStatus,
} from "../packages/bot-algo/src/vw-kama-cuda.js";

const MINUTE = 60_000;
void run().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function run(): Promise<void> {
  const candidateCount = positiveInteger(process.argv[2] ?? "128", "candidate count");
  const candleCount = positiveInteger(process.argv[3] ?? "60000", "candle count");
  const scoreStartIndex = Math.max(2_000, Math.floor(candleCount * 0.25));
  if (scoreStartIndex >= candleCount) throw new Error("Benchmark requires more than 2,000 candles.");

  const status = await vwKamaCudaStatus();
  if (!status.available) throw new Error(status.reason);
  const candles = syntheticCandles(candleCount);
  const columns = columnarVwKamaCandles(candles);
  const oracleResult = perfectMarginOracle(candles, {
    startingQuote: 1,
    leverage: 1,
    friction: 0.00175,
    eventMode: "close",
    maxPathCandles: 1,
  });
  const preparedOracle = prepareVwKamaOracle(columns, scoreStartIndex, oracleResult);
  const candidates = Array.from({ length: candidateCount }, (_, index) => parameters(index));
  const options = {
    intervalMs: MINUTE,
    scoreStartIndex,
    oracleFriction: 0.00175,
    matchWindowMs: 120 * MINUTE,
    timingHalfLifeMs: 10 * MINUTE,
    warmupMultiple: 3,
  };

  // Exclude one-time CUDA context creation from the measured batch.
  await evaluateVwKamaCudaBatch(columns, preparedOracle, candidates.slice(0, 1), options);
  const gpuStarted = performance.now();
  const gpu = await evaluateVwKamaCudaBatch(columns, preparedOracle, candidates, options);
  const gpuMs = performance.now() - gpuStarted;
  const gpuScores = gpu.map((result) => score(
    result.timingCredit,
    result.signalCount,
    result.oracleCount,
    result.matchedCount,
    result.stateCredit / result.stateCount,
  ));

  const cpuStarted = performance.now();
  const cpuScores = candidates.map((candidate) => evaluateVwKamaOracle(columns, {
    ...options,
    scoreStartTime: candles[scoreStartIndex]!.openTime,
    parameters: candidate,
    preparedOracle,
    includeTrace: false,
  }).metrics.score);
  const cpuMs = performance.now() - cpuStarted;
  const candidateCandles = candidateCount * candleCount;
  const scoreMae = mean(cpuScores.map((value, index) => Math.abs(value - gpuScores[index]!)));

  console.log(`Device: ${status.device}`);
  console.log(`Workload: ${candidateCount} candidates × ${candleCount.toLocaleString()} candles`);
  console.log(`CUDA batch: ${gpuMs.toFixed(1)} ms (${throughput(candidateCandles, gpuMs)} candidate-candles/s)`);
  console.log(`CPU serial exact: ${cpuMs.toFixed(1)} ms (${throughput(candidateCandles, cpuMs)} candidate-candles/s)`);
  console.log(`Serial speedup: ${(cpuMs / gpuMs).toFixed(2)}×`);
  console.log(`Screening score MAE: ${scoreMae.toFixed(6)}; Pearson r: ${correlation(cpuScores, gpuScores).toFixed(6)}`);
}

function score(
  timingCredit: number,
  signals: number,
  oracle: number,
  matches: number,
  exposureAgreement: number,
): number {
  const precision = eventRatio(timingCredit, signals, oracle);
  const recall = eventRatio(timingCredit, oracle, signals);
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  const cleanliness = signals > 0 ? matches / signals : 1;
  return vwKamaScore(f1, exposureAgreement, cleanliness);
}

function eventRatio(value: number, total: number, opposite: number): number {
  return total > 0 ? value / total : opposite === 0 ? 1 : 0;
}

function parameters(index: number): VwKamaParameters {
  const feature = index % 3 !== 0;
  return {
    efficiencyMs: (15 + index % 105) * MINUTE,
    efficiencyVolumeEmaMs: (30 + index % 330) * MINUTE,
    efficiencyVolumePower: index % 4 === 0 ? 0 : 0.4 + index % 30 / 10,
    fastMs: (1 + index % 25) * MINUTE,
    slowMs: (120 + index % 1_200) * MINUTE,
    power: 0.5 + index % 40 / 10,
    volumeMs: (20 + index % 500) * MINUTE,
    volumeCap: 1.5 + index % 70 / 10,
    volumePower: index % 5 === 0 ? 0 : 0.2 + index % 25 / 10,
    deadbandBpsHour: 1 + index % 80,
    deadbandMode: (["flat", "hold", "hysteresis"] as const)[index % 3]!,
    hysteresisReleaseRatio: (index % 10) / 10,
    thresholdMode: index % 2 === 0 ? "static" : "adaptive",
    thresholdLookbackMs: (30 + index % 600) * MINUTE,
    thresholdNoiseMultiplier: index % 2 === 0 ? 0 : 0.5 + index % 30 / 10,
    buyMaxFraction: 0.4 + index % 6 / 10,
    sellMaxFraction: 0.4 + (index * 3) % 6 / 10,
    buySizingSigmaBpsHour: 20 + index % 250,
    sellSizingSigmaBpsHour: 20 + (index * 7) % 250,
    agreementMode: index % 2 === 0 ? "sizing" : "confidence",
    confirmationMix: feature ? 0.2 + index % 8 / 10 : 0,
    confirmationMinQuality: feature ? index % 5 / 10 : 0,
    confirmationAccelerationLookbackMs: (20 + index % 180) * MINUTE,
    confirmationDistanceLookbackMs: (20 + (index * 3) % 180) * MINUTE,
    confirmationAccelerationWeight: feature ? index % 20 / 10 : 0,
    confirmationDistanceWeight: feature ? (index * 5) % 20 / 10 : 0,
    confirmationBias: (index % 20 - 10) / 10,
    confirmationEmaMs: (60 + index % 600) * MINUTE,
    confirmationEmaThresholdBpsHour: 1 + index % 80,
    confirmationEmaWeight: feature && index % 2 === 0 ? 0.8 : 0,
    confirmationEmaGateStrength: feature && index % 7 === 0 ? 0.4 : 0,
    confirmationRsiMs: (10 + index % 120) * MINUTE,
    confirmationRsiThreshold: index % 15,
    confirmationRsiWeight: feature && index % 3 === 0 ? 0.7 : 0,
    confirmationDmiMs: (10 + index % 120) * MINUTE,
    confirmationDmiWeight: feature && index % 5 === 0 ? 0.6 : 0,
    confirmationAdxThreshold: 10 + index % 35,
    meanReversionMix: index % 6 === 0 ? 0.35 : 0,
    meanReversionMeanMs: (60 + index % 600) * MINUTE,
    meanReversionVolatilityMs: (60 + (index * 7) % 600) * MINUTE,
    meanReversionThreshold: 0.75 + index % 30 / 10,
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

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function correlation(left: number[], right: number[]): number {
  const leftMean = mean(left);
  const rightMean = mean(right);
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index]! - leftMean;
    const b = right[index]! - rightMean;
    covariance += a * b;
    leftVariance += a * a;
    rightVariance += b * b;
  }
  return covariance / Math.sqrt(leftVariance * rightVariance);
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${label} must be a positive integer.`);
  return parsed;
}

function throughput(work: number, elapsedMs: number): string {
  return (work / (elapsedMs / 1_000)).toLocaleString(undefined, { maximumFractionDigits: 0 });
}
