import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  VwKamaCandleColumns,
  VwKamaParameters,
  VwKamaPreparedOracle,
} from "./kama-signal-evaluator.js";
import type { ExposureValueOracle } from "./exposure-value-distillation.js";
import {
  createExposureValueOracleStorage,
  shareExposureValueOracle,
  strategyExposureTemperatures,
  type ExposureValueOracleOptions,
} from "./exposure-value-distillation.js";

const PARAMETER_SIZE = 196;
const RESULT_SIZE = 152;
const INT_PARAMETER_COUNT = 20;
export const VW_KAMA_CUDA_VALUE_ORACLE_AUTO_MIN_GRID_SIZE = 129;
const nativeDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../native/cuda/build");
const defaultLibraryPath = path.join(nativeDirectory, "libvw_kama_cuda.so");

interface NativeCuda {
  deviceCount(): number;
  deviceName(device: number): string | null;
  lastError(): string;
  parameterSize(): number;
  resultSize(): number;
  prepareValueOracle(
    prices: Float64Array,
    priceCount: number,
    scoreStart: number,
    horizonSteps: number,
    gridSize: number,
    minimumExposure: number,
    maximumExposure: number,
    temperature: number,
    friction: number,
    opportunityEpsilon: number,
    quoteLendRate: number,
    quoteBorrowRate: number,
    assetBorrowRate: number,
    means: Float32Array,
    secondMoments: Float32Array,
    modalExposures: Float32Array,
    optimalExposures: Float32Array,
    entropies: Float32Array,
    weights: Float32Array,
    opportunities: Float32Array,
    probabilities: Float32Array | null,
    elapsedMs: Float64Array,
  ): number;
  evaluate(
    closeTimes: Float64Array,
    high: Float64Array,
    low: Float64Array,
    close: Float64Array,
    volume: Float64Array,
    oracleCodes: Uint8Array,
    valueMeans: Float32Array | null,
    valueOptimalExposures: Float32Array | null,
    valueSecondMoments: Float32Array | null,
    valueEntropies: Float32Array | null,
    valueWeights: Float32Array | null,
    valueOpportunities: Float32Array | null,
    strategyTemperatures: Float32Array | null,
    candleCount: number,
    scoreStart: number,
    intervalMs: number,
    valueHorizonSteps: number,
    oracleFriction: number,
    quoteLendRate: number,
    quoteBorrowRate: number,
    assetBorrowRate: number,
    matchWindowMs: number,
    timingHalfLifeMs: number,
    valueGridSize: number,
    valueGridMinimum: number,
    valueGridMaximum: number,
    parameters: Buffer,
    candidateCount: number,
    output: Buffer,
  ): number;
}

export interface VwKamaCudaBatchOptions {
  intervalMs: number;
  scoreStartIndex: number;
  oracleFriction: number;
  matchWindowMs: number;
  timingHalfLifeMs: number;
  warmupMultiple: number;
  valueDistillation?: {
    oracle: ExposureValueOracle;
    strategyTemperature: number;
    strategyVolatilityScaling: boolean;
    strategyTemperatures?: Float32Array;
  };
}

export interface VwKamaCudaCaseResult {
  stateCredit: number;
  timingCredit: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
  lagP95Ms: number | null;
  lagMedianSignedMs: number | null;
  elapsedMs: number;
  distillationWeightedCrossEntropy: number;
  distillationWeightedOracleEntropy: number;
  distillationWeight: number;
  distillationOpportunity: number;
  strategyFinalEquity: number;
  oracleFinalEquity: number;
  strategyMaxDrawdown: number;
  oracleMaxDrawdown: number;
  strategyTurnover: number;
  oracleTurnover: number;
  stateCount: number;
  signalCount: number;
  oracleCount: number;
  matchedCount: number;
}

export interface VwKamaCudaStatus {
  available: boolean;
  device?: string;
  reason?: string;
  libraryPath: string;
}

export interface ExposureValueOracleCudaResult {
  oracle: ExposureValueOracle;
  kernelMs: number;
}

let loaded: NativeCuda | null | undefined;
let loadFailure: string | undefined;

export async function vwKamaCudaStatus(): Promise<VwKamaCudaStatus> {
  const libraryPath = cudaLibraryPath();
  try {
    const native = await loadNative();
    const count = native.deviceCount();
    if (count <= 0) {
      return {
        available: false,
        reason: count < 0 ? native.lastError() : "No CUDA device is available.",
        libraryPath,
      };
    }
    return {
      available: true,
      device: native.deviceName(0) ?? "CUDA device 0",
      libraryPath,
    };
  } catch (error) {
    return {
      available: false,
      reason: error instanceof Error ? error.message : String(error),
      libraryPath,
    };
  }
}

export async function evaluateVwKamaCudaBatch(
  candles: VwKamaCandleColumns,
  oracle: VwKamaPreparedOracle,
  candidates: readonly VwKamaParameters[],
  options: VwKamaCudaBatchOptions,
): Promise<VwKamaCudaCaseResult[]> {
  if (candidates.length === 0) return [];
  if (oracle.stateCodes.length < candles.length) {
    throw new Error("VW-KAMA CUDA oracle states do not cover the candle columns.");
  }
  const valueDistillation = options.valueDistillation;
  if (valueDistillation && (
    valueDistillation.oracle.means.length < candles.length
    || valueDistillation.oracle.optimalExposures.length < candles.length
    || valueDistillation.oracle.secondMoments.length < candles.length
    || valueDistillation.oracle.entropies.length < candles.length
    || valueDistillation.oracle.weights.length < candles.length
    || valueDistillation.oracle.opportunities.length < candles.length
  )) {
    throw new Error("VW-KAMA CUDA value oracle does not cover the candle columns.");
  }
  const strategyTemperatures = valueDistillation
    ? valueDistillation.strategyTemperatures ?? strategyExposureTemperatures(candles.close, {
      intervalMs: options.intervalMs,
      horizonSteps: valueDistillation.oracle.horizonSteps,
      temperature: valueDistillation.strategyTemperature,
      scaleByVolatility: valueDistillation.strategyVolatilityScaling,
    })
    : null;
  if (strategyTemperatures && strategyTemperatures.length < candles.length) {
    throw new Error("VW-KAMA CUDA strategy temperatures do not cover the candle columns.");
  }
  const native = await loadNative();
  const parameterBuffer = Buffer.alloc(candidates.length * PARAMETER_SIZE);
  for (let index = 0; index < candidates.length; index += 1) {
    writeParameters(parameterBuffer, index * PARAMETER_SIZE, candidates[index]!, options);
  }
  const output = Buffer.alloc(candidates.length * RESULT_SIZE);
  const status = native.evaluate(
    candles.closeTime,
    candles.high,
    candles.low,
    candles.close,
    candles.volume,
    oracle.stateCodes,
    valueDistillation?.oracle.means ?? null,
    valueDistillation?.oracle.optimalExposures ?? null,
    valueDistillation?.oracle.secondMoments ?? null,
    valueDistillation?.oracle.entropies ?? null,
    valueDistillation?.oracle.weights ?? null,
    valueDistillation?.oracle.opportunities ?? null,
    strategyTemperatures,
    candles.length,
    options.scoreStartIndex,
    options.intervalMs,
    valueDistillation?.oracle.horizonSteps ?? 1,
    options.oracleFriction,
    valueDistillation?.oracle.execution.quoteLendRate ?? 0,
    valueDistillation?.oracle.execution.quoteBorrowRate ?? 0,
    valueDistillation?.oracle.execution.assetBorrowRate ?? 0,
    options.matchWindowMs,
    options.timingHalfLifeMs,
    valueDistillation?.oracle.grid.length ?? 0,
    valueDistillation?.oracle.grid[0] ?? 0,
    valueDistillation?.oracle.grid[valueDistillation.oracle.grid.length - 1] ?? 0,
    parameterBuffer,
    candidates.length,
    output,
  );
  if (status !== 0) throw new Error(`VW-KAMA CUDA evaluation failed: ${native.lastError()}`);
  return Array.from({ length: candidates.length }, (_, index) => readResult(output, index * RESULT_SIZE));
}

export async function prepareExposureValueOracleCuda(
  prices: ArrayLike<number>,
  options: ExposureValueOracleOptions,
  shared = false,
): Promise<ExposureValueOracleCudaResult> {
  if (options.gridSize > 1_024) {
    throw new Error("Exposure-value CUDA oracle supports at most 1,024 grid points.");
  }
  const native = await loadNative();
  const source = prices instanceof Float64Array ? prices : Float64Array.from(prices);
  const oracle = createExposureValueOracleStorage(source, options, false);
  const elapsedMs = new Float64Array(1);
  const status = native.prepareValueOracle(
    source,
    source.length,
    options.scoreStartIndex,
    oracle.horizonSteps,
    options.gridSize,
    oracle.execution.minExposure,
    oracle.execution.maxExposure,
    options.temperature,
    oracle.execution.friction,
    Math.max(0, options.opportunityEpsilon ?? 1e-6),
    oracle.execution.quoteLendRate,
    oracle.execution.quoteBorrowRate,
    oracle.execution.assetBorrowRate,
    oracle.means,
    oracle.secondMoments,
    oracle.modalExposures,
    oracle.optimalExposures,
    oracle.entropies,
    oracle.weights,
    oracle.opportunities,
    oracle.probabilities ?? null,
    elapsedMs,
  );
  if (status !== 0) {
    throw new Error(`Exposure-value CUDA oracle failed: ${native.lastError()}`);
  }
  return {
    oracle: shared ? shareExposureValueOracle(oracle) : oracle,
    kernelMs: elapsedMs[0]!,
  };
}

async function loadNative(): Promise<NativeCuda> {
  if (loaded) return loaded;
  if (loaded === null) throw new Error(loadFailure ?? "VW-KAMA CUDA helper is unavailable.");
  const libraryPath = cudaLibraryPath();
  if (!fs.existsSync(libraryPath)) {
    loaded = null;
    loadFailure = `CUDA helper is not built at ${libraryPath}; run npm run build:cuda.`;
    throw new Error(loadFailure);
  }
  try {
    const { default: koffi } = await import("koffi");
    const library = koffi.load(libraryPath);
    const pointer = koffi.pointer("void");
    const native: NativeCuda = {
      deviceCount: library.func("int vw_kama_cuda_device_count()"),
      deviceName: library.func("str vw_kama_cuda_device_name(int)"),
      lastError: library.func("str vw_kama_cuda_last_error()"),
      parameterSize: library.func("int vw_kama_cuda_params_size()"),
      resultSize: library.func("int vw_kama_cuda_result_size()"),
      prepareValueOracle: library.func("vw_kama_cuda_prepare_value_oracle", "int", [
        pointer, "int", "int", "int", "int",
        "double", "double", "double", "double", "double", "double", "double", "double",
        pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
      ]),
      evaluate: library.func("vw_kama_cuda_evaluate", "int", [
        pointer, pointer, pointer, pointer, pointer, pointer,
        pointer, pointer, pointer, pointer, pointer, pointer, pointer,
        "int", "int", "double", "int", "double", "double", "double", "double", "double", "double",
        "int", "double", "double",
        pointer, "int", pointer,
      ]),
    };
    if (native.parameterSize() !== PARAMETER_SIZE || native.resultSize() !== RESULT_SIZE) {
      throw new Error("VW-KAMA CUDA JavaScript/native ABI size mismatch; rebuild the helper.");
    }
    loaded = native;
    return native;
  } catch (error) {
    loaded = null;
    loadFailure = error instanceof Error ? error.message : String(error);
    throw error;
  }
}

function cudaLibraryPath(): string {
  return path.resolve(process.env.VW_KAMA_CUDA_LIBRARY ?? defaultLibraryPath);
}

function writeParameters(
  buffer: Buffer,
  offset: number,
  parameters: VwKamaParameters,
  options: VwKamaCudaBatchOptions,
): void {
  const samples = (durationMs: number): number => Math.max(1, Math.round(durationMs / options.intervalMs));
  const integers = [
    samples(parameters.efficiencyMs),
    samples(parameters.efficiencyVolumeEmaMs ?? parameters.volumeMs),
    samples(parameters.fastMs),
    samples(parameters.slowMs),
    samples(parameters.volumeMs),
    samples(parameters.rateEmaMs ?? options.intervalMs),
    samples(parameters.thresholdLookbackMs ?? options.intervalMs),
    samples(parameters.confirmationAccelerationLookbackMs ?? options.intervalMs),
    samples(parameters.confirmationDistanceLookbackMs ?? options.intervalMs),
    samples(parameters.confirmationEmaMs ?? options.intervalMs),
    samples(parameters.confirmationRsiMs ?? options.intervalMs),
    samples(parameters.confirmationDmiMs ?? options.intervalMs),
    samples(parameters.meanReversionEfficiencyMs ?? parameters.efficiencyMs),
    samples(parameters.meanReversionFastMs ?? parameters.fastMs),
    samples(parameters.meanReversionSlowMs ?? parameters.slowMs),
    samples(parameters.meanReversionVolatilityMs ?? parameters.slowMs),
    deadbandMode(parameters.deadbandMode),
    parameters.agreementMode === "confidence" ? 1 : 0,
    parameters.rateMode === "log" ? 1 : 0,
    parameters.thresholdNoiseResponse === "inverse" ? 1 : 0,
  ];
  const floats = [
    parameters.power,
    parameters.volumeCap,
    parameters.volumePower,
    parameters.efficiencyVolumePower ?? 0,
    parameters.deadbandBpsHour,
    parameters.hysteresisReleaseRatio ?? 0.25,
    parameters.thresholdNoiseMultiplier ?? 0,
    parameters.thresholdInverseMaxBpsHour ?? 0,
    parameters.thresholdInverseNoiseScaleBpsHour ?? 1,
    parameters.buyMaxFraction ?? 1,
    parameters.sellMaxFraction ?? 1,
    parameters.buySizingSigmaBpsHour ?? Number.MAX_VALUE,
    parameters.sellSizingSigmaBpsHour ?? Number.MAX_VALUE,
    parameters.confirmationMix ?? 0,
    parameters.confirmationMinQuality ?? 0,
    parameters.confirmationAccelerationWeight ?? 1,
    parameters.confirmationDistanceWeight ?? 1,
    parameters.confirmationBias ?? 0,
    parameters.confirmationEmaThresholdBpsHour ?? 0,
    parameters.confirmationEmaWeight ?? 0,
    parameters.confirmationEmaGateStrength ?? 0,
    parameters.confirmationRsiThreshold ?? 0,
    parameters.confirmationRsiWeight ?? 0,
    parameters.confirmationDmiWeight ?? 0,
    parameters.confirmationAdxThreshold ?? 20,
    parameters.meanReversionSuppressionThreshold ?? 1,
    parameters.meanReversionReversalThreshold ?? 0,
    parameters.signalFrictionFraction ?? 1,
    options.warmupMultiple,
  ];
  if (integers.length !== INT_PARAMETER_COUNT || floats.length !== 29) {
    throw new Error("VW-KAMA CUDA parameter layout is inconsistent.");
  }
  for (let index = 0; index < integers.length; index += 1) {
    buffer.writeInt32LE(integers[index]!, offset + index * 4);
  }
  const floatOffset = offset + INT_PARAMETER_COUNT * 4;
  for (let index = 0; index < floats.length; index += 1) {
    buffer.writeFloatLE(finiteFloat(floats[index]!), floatOffset + index * 4);
  }
}

function readResult(buffer: Buffer, offset: number): VwKamaCudaCaseResult {
  const nullable = (value: number): number | null => Number.isNaN(value) ? null : value;
  return {
    stateCredit: buffer.readDoubleLE(offset),
    timingCredit: buffer.readDoubleLE(offset + 8),
    lagP50Ms: nullable(buffer.readDoubleLE(offset + 16)),
    lagP90Ms: nullable(buffer.readDoubleLE(offset + 24)),
    lagP95Ms: nullable(buffer.readDoubleLE(offset + 32)),
    lagMedianSignedMs: nullable(buffer.readDoubleLE(offset + 40)),
    elapsedMs: buffer.readDoubleLE(offset + 48),
    distillationWeightedCrossEntropy: buffer.readDoubleLE(offset + 56),
    distillationWeightedOracleEntropy: buffer.readDoubleLE(offset + 64),
    distillationWeight: buffer.readDoubleLE(offset + 72),
    distillationOpportunity: buffer.readDoubleLE(offset + 80),
    strategyFinalEquity: buffer.readDoubleLE(offset + 88),
    oracleFinalEquity: buffer.readDoubleLE(offset + 96),
    strategyMaxDrawdown: buffer.readDoubleLE(offset + 104),
    oracleMaxDrawdown: buffer.readDoubleLE(offset + 112),
    strategyTurnover: buffer.readDoubleLE(offset + 120),
    oracleTurnover: buffer.readDoubleLE(offset + 128),
    stateCount: buffer.readInt32LE(offset + 136),
    signalCount: buffer.readInt32LE(offset + 140),
    oracleCount: buffer.readInt32LE(offset + 144),
    matchedCount: buffer.readInt32LE(offset + 148),
  };
}

function finiteFloat(value: number): number {
  if (value === Number.MAX_VALUE) return 3.402823466e38;
  if (!Number.isFinite(value)) throw new Error(`VW-KAMA CUDA parameter is not finite: ${value}`);
  return value;
}

function deadbandMode(mode: VwKamaParameters["deadbandMode"]): number {
  return mode === "hold" ? 1 : mode === "hysteresis" ? 2 : 0;
}
