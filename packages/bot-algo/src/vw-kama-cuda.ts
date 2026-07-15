import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  VwKamaCandleColumns,
  VwKamaParameters,
  VwKamaPreparedOracle,
} from "./kama-signal-evaluator.js";

const PARAMETER_SIZE = 168;
const RESULT_SIZE = 72;
const INT_PARAMETER_COUNT = 16;
const nativeDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../native/cuda/build");
const defaultLibraryPath = path.join(nativeDirectory, "libvw_kama_cuda.so");

interface NativeCuda {
  deviceCount(): number;
  deviceName(device: number): string | null;
  lastError(): string;
  parameterSize(): number;
  resultSize(): number;
  evaluate(
    closeTimes: Float64Array,
    high: Float64Array,
    low: Float64Array,
    close: Float64Array,
    volume: Float64Array,
    oracleCodes: Uint8Array,
    candleCount: number,
    scoreStart: number,
    intervalMs: number,
    oracleFriction: number,
    matchWindowMs: number,
    timingHalfLifeMs: number,
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
}

export interface VwKamaCudaCaseResult {
  stateCredit: number;
  timingCredit: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
  lagP95Ms: number | null;
  lagMedianSignedMs: number | null;
  elapsedMs: number;
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
    candles.length,
    options.scoreStartIndex,
    options.intervalMs,
    options.oracleFriction,
    options.matchWindowMs,
    options.timingHalfLifeMs,
    parameterBuffer,
    candidates.length,
    output,
  );
  if (status !== 0) throw new Error(`VW-KAMA CUDA evaluation failed: ${native.lastError()}`);
  return Array.from({ length: candidates.length }, (_, index) => readResult(output, index * RESULT_SIZE));
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
      evaluate: library.func("vw_kama_cuda_evaluate", "int", [
        pointer, pointer, pointer, pointer, pointer, pointer,
        "int", "int", "double", "double", "double", "double",
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
    samples(parameters.thresholdLookbackMs ?? options.intervalMs),
    samples(parameters.confirmationAccelerationLookbackMs ?? options.intervalMs),
    samples(parameters.confirmationDistanceLookbackMs ?? options.intervalMs),
    samples(parameters.confirmationEmaMs ?? options.intervalMs),
    samples(parameters.confirmationRsiMs ?? options.intervalMs),
    samples(parameters.confirmationDmiMs ?? options.intervalMs),
    samples(parameters.meanReversionMeanMs ?? parameters.slowMs),
    samples(parameters.meanReversionVolatilityMs ?? parameters.slowMs),
    deadbandMode(parameters.deadbandMode),
    parameters.thresholdMode === "adaptive" ? 1 : 0,
    parameters.agreementMode === "confidence" ? 1 : 0,
  ];
  const floats = [
    parameters.power,
    parameters.volumeCap,
    parameters.volumePower,
    parameters.efficiencyVolumePower ?? 0,
    parameters.deadbandBpsHour,
    parameters.hysteresisReleaseRatio ?? 0.25,
    parameters.thresholdNoiseMultiplier ?? 0,
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
    parameters.meanReversionMix ?? 0,
    parameters.meanReversionThreshold ?? 2,
    options.warmupMultiple,
  ];
  if (integers.length !== INT_PARAMETER_COUNT || floats.length !== 26) {
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
    stateCount: buffer.readInt32LE(offset + 56),
    signalCount: buffer.readInt32LE(offset + 60),
    oracleCount: buffer.readInt32LE(offset + 64),
    matchedCount: buffer.readInt32LE(offset + 68),
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
