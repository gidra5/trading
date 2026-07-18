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
  normalizeExposureValueDistillationLossConfig,
  shareExposureValueOracle,
  strategyExposureTemperatures,
  type ExposureValueDistillationLossConfig,
  type ExposureValueOracleOptions,
} from "./exposure-value-distillation.js";

const PARAMETER_SIZE = 208;
const RESULT_SIZE = 192;
const INT_PARAMETER_COUNT = 21;
const nativeDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../native/cuda/build");
const defaultLibraryPath = path.join(nativeDirectory, "libvw_kama_cuda.so");

type CudaHandle = number | bigint;

interface NativeCuda {
  deviceCount(): number;
  deviceName(device: number): string | null;
  lastError(): string;
  parameterSize(): number;
  resultSize(): number;
  createFitnessCase(
    high: Float64Array | null,
    low: Float64Array | null,
    close: Float64Array,
    volume: Float64Array,
    valueMeans: Float32Array,
    valueSecondMoments: Float32Array,
    valueEntropies: Float32Array,
    valueWeights: Float32Array,
    strategyTemperatures: Float32Array,
    strategyQuadraticVolatilities: Float32Array,
    oracleBinProbabilities: Float32Array | null,
    candleCount: number,
    scoreStart: number,
    intervalMs: number,
    valueHoldingPeriodSteps: number,
    oracleFriction: number,
    quoteLendRate: number,
    quoteBorrowRate: number,
    assetBorrowRate: number,
    valueGridSize: number,
    valueGridMinimum: number,
    valueGridMaximum: number,
    maximumEffectiveExposure: number,
    strategyQuadraticScale: number,
    entropyGapLambda: number,
    stateMutualInformationLambda: number,
    oracleMutualInformationLambda: number,
    oracleMutualInformationMode: number,
    mutualInformationBins: number,
    includeHighLow: number,
    residentBytes: Float64Array,
  ): CudaHandle;
  evaluateFitnessCase(
    handle: CudaHandle,
    parameters: Buffer,
    candidateCount: number,
    output: Buffer,
  ): number;
  evaluateFitnessCases(
    handles: BigUint64Array,
    caseCount: number,
    parameters: Buffer,
    candidateCount: number,
    output: Buffer,
  ): number;
  fitnessCaseDeviceBytes(handle: CudaHandle): number;
  destroyFitnessCase(handle: CudaHandle): number;
  prepareValueOracle(
    prices: Float64Array,
    priceCount: number,
    scoreStart: number,
    holdingPeriodSteps: number,
    valueHorizonSteps: number,
    gridSize: number,
    minimumExposure: number,
    maximumExposure: number,
    maximumEffectiveExposure: number,
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
    valueSecondMoments: Float32Array | null,
    valueOptimalExposures: Float32Array | null,
    valueEntropies: Float32Array | null,
    valueWeights: Float32Array | null,
    valueOpportunities: Float32Array | null,
    oracleBinProbabilities: Float32Array | null,
    strategyTemperatures: Float32Array | null,
    strategyQuadraticVolatilities: Float32Array | null,
    candleCount: number,
    scoreStart: number,
    intervalMs: number,
    valueHoldingPeriodSteps: number,
    oracleFriction: number,
    quoteLendRate: number,
    quoteBorrowRate: number,
    assetBorrowRate: number,
    matchWindowMs: number,
    timingHalfLifeMs: number,
    valueGridSize: number,
    valueGridMinimum: number,
    valueGridMaximum: number,
    maximumEffectiveExposure: number,
    strategyQuadraticScale: number,
    entropyGapLambda: number,
    stateMutualInformationLambda: number,
    oracleMutualInformationLambda: number,
    oracleMutualInformationMode: number,
    mutualInformationBins: number,
    parameters: Buffer,
    candidateCount: number,
    fitnessOnly: number,
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
  fitnessOnly?: boolean;
  valueDistillation?: {
    oracle: ExposureValueOracle;
    strategyVolatilityScaling: boolean;
    lossConfig: ExposureValueDistillationLossConfig;
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
  distillationWeightedStrategyEntropy: number;
  distillationWeightedEntropyGap: number;
  distillationStateMutualInformation: number;
  distillationOracleMutualInformation: number;
  distillationMixedLoss: number;
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

export interface VwKamaCudaFitnessBatchCase {
  candles: VwKamaCandleColumns;
  options: VwKamaCudaBatchOptions & {
    valueDistillation: NonNullable<VwKamaCudaBatchOptions["valueDistillation"]>;
  };
}

export interface ExposureValueOracleCudaResult {
  oracle: ExposureValueOracle;
  kernelMs: number;
}

let loaded: NativeCuda | null | undefined;
let loadFailure: string | undefined;
let nextObjectId = 1;
const objectIds = new WeakMap<object, number>();
const oracleBinCaches = new WeakMap<ExposureValueOracle, Map<number, Float32Array>>();
const fitnessCaseCache = new Map<string, CachedFitnessCase>();
let fitnessCaseCacheBytes = 0;

interface CachedFitnessCase {
  handle: CudaHandle;
  deviceBytes: number;
  preciseScratchBytes: number;
}

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
  const loss = normalizeExposureValueDistillationLossConfig(
    valueDistillation?.lossConfig,
    valueDistillation?.oracle.grid.length,
  );
  const lossEnabled = loss.entropyGapLambda > 0
    || loss.stateMutualInformationLambda > 0
    || loss.oracleMutualInformationLambda > 0;
  const oracleBinProbabilities = valueDistillation && preciseOracleMutualInformationEnabled(loss)
    ? binnedOracleProbabilities(valueDistillation.oracle, loss.mutualInformationBins)
    : null;
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
  const strategyQuadraticVolatilities = valueDistillation
    ? new Float32Array(candles.length)
    : null;
  const strategyTemperatures = valueDistillation
    ? valueDistillation.strategyTemperatures ?? strategyExposureTemperatures(candles.close, {
      intervalMs: options.intervalMs,
      holdingPeriodSteps: valueDistillation.oracle.holdingPeriodSteps,
      temperature: 1,
      scaleByVolatility: valueDistillation.strategyVolatilityScaling,
    })
    : null;
  if ((strategyTemperatures && strategyTemperatures.length < candles.length)
    || (strategyQuadraticVolatilities && strategyQuadraticVolatilities.length < candles.length)) {
    throw new Error("VW-KAMA CUDA strategy calibration does not cover the candle columns.");
  }
  const native = await loadNative();
  const parameterBuffer = Buffer.alloc(candidates.length * PARAMETER_SIZE);
  for (let index = 0; index < candidates.length; index += 1) {
    writeParameters(parameterBuffer, index * PARAMETER_SIZE, candidates[index]!, options);
  }
  const output = Buffer.alloc(candidates.length * RESULT_SIZE);
  if (options.fitnessOnly && valueDistillation && strategyTemperatures) {
    const includeHighLow = candidates.some(candidateNeedsDmi);
    const cacheKey = fitnessCaseKey(
      candles,
      valueDistillation.oracle,
      options,
      includeHighLow,
      valueDistillation.strategyTemperatures,
      undefined,
    );
    const testCase = getOrCreateFitnessCase(
      native,
      cacheKey,
      candles,
      valueDistillation.oracle,
      strategyTemperatures,
      strategyQuadraticVolatilities!,
      options,
      includeHighLow,
    );
    const preciseScratchBytes = preciseOracleMutualInformationEnabled(loss)
      ? candidates.length * (candles.length - options.scoreStartIndex) * 3
          * Float32Array.BYTES_PER_ELEMENT
        + candidates.length * loss.mutualInformationBins * loss.mutualInformationBins
          * Float64Array.BYTES_PER_ELEMENT
      : 0;
    const additionalScratchBytes = Math.max(0, preciseScratchBytes - testCase.preciseScratchBytes);
    if (testCase.deviceBytes + additionalScratchBytes > fitnessCaseBudgetBytes()) {
      throw new Error(
        `Precise oracle MI needs about ${formatByteCount(testCase.deviceBytes + additionalScratchBytes)} `
        + `for this case; increase VW_KAMA_CUDA_CASE_CACHE_BYTES or use approximate mode.`,
      );
    }
    admitFitnessCaseBytes(native, additionalScratchBytes, cacheKey);
    const status = native.evaluateFitnessCase(
      testCase.handle,
      parameterBuffer,
      candidates.length,
      output,
    );
    if (status !== 0) throw new Error(`VW-KAMA resident CUDA evaluation failed: ${native.lastError()}`);
    testCase.preciseScratchBytes = Math.max(testCase.preciseScratchBytes, preciseScratchBytes);
    refreshFitnessCaseBytes(native, cacheKey, testCase);
    return Array.from(
      { length: candidates.length },
      (_, index) => readResult(output, index * RESULT_SIZE),
    );
  }
  const status = native.evaluate(
    candles.closeTime,
    candles.high,
    candles.low,
    candles.close,
    candles.volume,
    oracle.stateCodes,
    valueDistillation?.oracle.means ?? null,
    valueDistillation?.oracle.secondMoments ?? null,
    options.fitnessOnly ? null : valueDistillation?.oracle.optimalExposures ?? null,
    lossEnabled || !options.fitnessOnly ? valueDistillation?.oracle.entropies ?? null : null,
    valueDistillation?.oracle.weights ?? null,
    options.fitnessOnly ? null : valueDistillation?.oracle.opportunities ?? null,
    oracleBinProbabilities,
    strategyTemperatures,
    strategyQuadraticVolatilities,
    candles.length,
    options.scoreStartIndex,
    options.intervalMs,
    valueDistillation?.oracle.holdingPeriodSteps ?? 1,
    options.oracleFriction,
    valueDistillation?.oracle.execution.quoteLendRate ?? 0,
    valueDistillation?.oracle.execution.quoteBorrowRate ?? 0,
    valueDistillation?.oracle.execution.assetBorrowRate ?? 0,
    options.matchWindowMs,
    options.timingHalfLifeMs,
    valueDistillation?.oracle.grid.length ?? 0,
    valueDistillation?.oracle.grid[0] ?? 0,
    valueDistillation?.oracle.grid[valueDistillation.oracle.grid.length - 1] ?? 0,
    valueDistillation?.oracle.execution.maxEffectiveExposure ?? 250,
    0,
    loss.entropyGapLambda,
    loss.stateMutualInformationLambda,
    loss.oracleMutualInformationLambda,
    loss.oracleMutualInformationMode === "precise" ? 1 : 0,
    loss.mutualInformationBins,
    parameterBuffer,
    candidates.length,
    options.fitnessOnly ? 1 : 0,
    output,
  );
  if (status !== 0) throw new Error(`VW-KAMA CUDA evaluation failed: ${native.lastError()}`);
  return Array.from({ length: candidates.length }, (_, index) => readResult(output, index * RESULT_SIZE));
}

export async function evaluateVwKamaCudaFitnessCases(
  cases: readonly VwKamaCudaFitnessBatchCase[],
  candidates: readonly VwKamaParameters[],
): Promise<VwKamaCudaCaseResult[][]> {
  if (cases.length === 0 || candidates.length === 0) return cases.map(() => []);
  if (cases.some(({ options }) => preciseOracleMutualInformationEnabled(
    normalizeExposureValueDistillationLossConfig(
      options.valueDistillation.lossConfig,
      options.valueDistillation.oracle.grid.length,
    ),
  ))) {
    const sequential: VwKamaCudaCaseResult[][] = [];
    for (const { candles, options } of cases) {
      sequential.push(await evaluateVwKamaCudaBatch(
        candles,
        { stateCodes: new Uint8Array(candles.length), transitions: [] },
        candidates,
        { ...options, fitnessOnly: true },
      ));
    }
    return sequential;
  }
  const native = await loadNative();
  const handles = new BigUint64Array(cases.length);
  const caseKeys: string[] = [];
  const cachedCases: CachedFitnessCase[] = [];
  const parameterBuffer = Buffer.alloc(cases.length * candidates.length * PARAMETER_SIZE);
  const includeHighLow = candidates.some(candidateNeedsDmi);
  const scheduledKeys = new Set<string>();
  for (let caseIndex = 0; caseIndex < cases.length; caseIndex += 1) {
    const request = cases[caseIndex]!;
    const { candles, options } = request;
    const distillation = options.valueDistillation;
    const quadraticVolatilities = new Float32Array(candles.length);
    const temperatures = distillation.strategyTemperatures ?? strategyExposureTemperatures(
      candles.close,
      {
        intervalMs: options.intervalMs,
        holdingPeriodSteps: distillation.oracle.holdingPeriodSteps,
        temperature: 1,
        scaleByVolatility: distillation.strategyVolatilityScaling,
      },
    );
    if (
      distillation.oracle.means.length < candles.length
      || distillation.oracle.secondMoments.length < candles.length
      || distillation.oracle.weights.length < candles.length
      || temperatures.length < candles.length
      || quadraticVolatilities.length < candles.length
    ) {
      throw new Error(`VW-KAMA CUDA fitness case ${caseIndex} does not cover its candle columns.`);
    }
    const key = fitnessCaseKey(
      candles,
      distillation.oracle,
      options,
      includeHighLow,
      distillation.strategyTemperatures,
      undefined,
    );
    if (scheduledKeys.has(key)) {
      throw new Error("VW-KAMA CUDA fitness cases must be distinct within one scheduled batch.");
    }
    scheduledKeys.add(key);
    const cached = getOrCreateFitnessCase(
      native,
      key,
      candles,
      distillation.oracle,
      temperatures,
      quadraticVolatilities,
      options,
      includeHighLow,
    );
    handles[caseIndex] = BigInt(cached.handle);
    caseKeys.push(key);
    cachedCases.push(cached);
    for (let candidate = 0; candidate < candidates.length; candidate += 1) {
      writeParameters(
        parameterBuffer,
        (caseIndex * candidates.length + candidate) * PARAMETER_SIZE,
        candidates[candidate]!,
        options,
      );
    }
  }
  const output = Buffer.alloc(cases.length * candidates.length * RESULT_SIZE);
  const status = native.evaluateFitnessCases(
    handles,
    cases.length,
    parameterBuffer,
    candidates.length,
    output,
  );
  if (status !== 0) throw new Error(`VW-KAMA CUDA fitness scheduler failed: ${native.lastError()}`);
  for (let index = 0; index < cachedCases.length; index += 1) {
    refreshFitnessCaseBytes(native, caseKeys[index]!, cachedCases[index]!, false);
  }
  admitFitnessCaseBytes(native, 0);
  return cases.map((_, caseIndex) => Array.from(
    { length: candidates.length },
    (_, candidate) => readResult(
      output,
      (caseIndex * candidates.length + candidate) * RESULT_SIZE,
    ),
  ));
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
    oracle.holdingPeriodSteps,
    oracle.valueHorizonSteps,
    options.gridSize,
    oracle.execution.minExposure,
    oracle.execution.maxExposure,
    oracle.execution.maxEffectiveExposure,
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
      createFitnessCase: library.func("vw_kama_cuda_create_fitness_case", "uint64_t", [
        pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
        pointer,
        "int", "int", "double", "int", "double", "double", "double", "double",
        "int", "double", "double", "double", "double", "double", "double", "double",
        "int", "int", "int", pointer,
      ]),
      evaluateFitnessCase: library.func(
        "int vw_kama_cuda_evaluate_fitness_case(uint64_t, void*, int, void*)",
      ),
      evaluateFitnessCases: library.func(
        "int vw_kama_cuda_evaluate_fitness_cases(void*, int, void*, int, void*)",
      ),
      fitnessCaseDeviceBytes: library.func(
        "double vw_kama_cuda_fitness_case_device_bytes(uint64_t)",
      ),
      destroyFitnessCase: library.func("int vw_kama_cuda_destroy_fitness_case(uint64_t)"),
      prepareValueOracle: library.func("vw_kama_cuda_prepare_value_oracle", "int", [
        pointer, "int", "int", "int", "int", "int",
        "double", "double", "double", "double", "double", "double", "double", "double", "double",
        pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
      ]),
      evaluate: library.func("vw_kama_cuda_evaluate", "int", [
        pointer, pointer, pointer, pointer, pointer, pointer,
        pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
        "int", "int", "double", "int", "double", "double", "double", "double", "double", "double",
        "int", "double", "double", "double", "double", "double", "double", "double",
        "int", "int",
        pointer, "int", "int", pointer,
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

export async function clearVwKamaCudaFitnessCaseCache(): Promise<void> {
  if (fitnessCaseCache.size === 0) return;
  const native = await loadNative();
  for (const testCase of fitnessCaseCache.values()) {
    native.destroyFitnessCase(testCase.handle);
  }
  fitnessCaseCache.clear();
  fitnessCaseCacheBytes = 0;
}

function getOrCreateFitnessCase(
  native: NativeCuda,
  key: string,
  candles: VwKamaCandleColumns,
  oracle: ExposureValueOracle,
  strategyTemperatures: Float32Array,
  strategyQuadraticVolatilities: Float32Array,
  options: VwKamaCudaBatchOptions,
  includeHighLow: boolean,
): CachedFitnessCase {
  const cached = fitnessCaseCache.get(key);
  if (cached) {
    fitnessCaseCache.delete(key);
    fitnessCaseCache.set(key, cached);
    return cached;
  }
  const loss = normalizeExposureValueDistillationLossConfig(
    options.valueDistillation?.lossConfig,
    oracle.grid.length,
  );
  const oracleBinProbabilities = preciseOracleMutualInformationEnabled(loss)
    ? binnedOracleProbabilities(oracle, loss.mutualInformationBins)
    : null;
  const lossEnabled = loss.entropyGapLambda > 0
    || loss.stateMutualInformationLambda > 0
    || loss.oracleMutualInformationLambda > 0;
  const baseResidentColumns = (includeHighLow ? 9 : 7) + (lossEnabled ? 1 : 0);
  const preciseResidentColumns = oracleBinProbabilities ? loss.mutualInformationBins : 0;
  const estimatedResidentBytes = candles.length
    * (baseResidentColumns + preciseResidentColumns)
    * Float32Array.BYTES_PER_ELEMENT;
  admitFitnessCaseBytes(native, estimatedResidentBytes);
  const residentBytes = new Float64Array(1);
  const handle = native.createFitnessCase(
    includeHighLow ? candles.high : null,
    includeHighLow ? candles.low : null,
    candles.close,
    candles.volume,
    oracle.means,
    oracle.secondMoments,
    oracle.entropies,
    oracle.weights,
    strategyTemperatures,
    strategyQuadraticVolatilities,
    oracleBinProbabilities,
    candles.length,
    options.scoreStartIndex,
    options.intervalMs,
    oracle.holdingPeriodSteps,
    options.oracleFriction,
    oracle.execution.quoteLendRate,
    oracle.execution.quoteBorrowRate,
    oracle.execution.assetBorrowRate,
    oracle.grid.length,
    oracle.grid[0]!,
    oracle.grid[oracle.grid.length - 1]!,
    oracle.execution.maxEffectiveExposure,
    0,
    loss.entropyGapLambda,
    loss.stateMutualInformationLambda,
    loss.oracleMutualInformationLambda,
    loss.oracleMutualInformationMode === "precise" ? 1 : 0,
    loss.mutualInformationBins,
    includeHighLow ? 1 : 0,
    residentBytes,
  );
  if (Number(handle) === 0) throw new Error(`Failed to create resident CUDA fitness case: ${native.lastError()}`);
  const result = { handle, deviceBytes: residentBytes[0]!, preciseScratchBytes: 0 };
  fitnessCaseCache.set(key, result);
  fitnessCaseCacheBytes += result.deviceBytes;
  return result;
}

function refreshFitnessCaseBytes(
  native: NativeCuda,
  key: string,
  testCase: CachedFitnessCase,
  evict = true,
): void {
  const bytes = native.fitnessCaseDeviceBytes(testCase.handle);
  fitnessCaseCacheBytes += bytes - testCase.deviceBytes;
  testCase.deviceBytes = bytes;
  fitnessCaseCache.delete(key);
  fitnessCaseCache.set(key, testCase);
  if (evict) admitFitnessCaseBytes(native, 0, key);
}

function admitFitnessCaseBytes(native: NativeCuda, incomingBytes: number, retainedKey?: string): void {
  const budget = fitnessCaseBudgetBytes();
  for (const [key, testCase] of fitnessCaseCache) {
    if (fitnessCaseCacheBytes + incomingBytes <= budget) break;
    if (key === retainedKey) continue;
    native.destroyFitnessCase(testCase.handle);
    fitnessCaseCache.delete(key);
    fitnessCaseCacheBytes -= testCase.deviceBytes;
  }
}

function fitnessCaseBudgetBytes(): number {
  const configured = Number(process.env.VW_KAMA_CUDA_CASE_CACHE_BYTES ?? 1_500_000_000);
  return Number.isFinite(configured) && configured > 0 ? configured : 1_500_000_000;
}

function formatByteCount(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
}

function fitnessCaseKey(
  candles: VwKamaCandleColumns,
  oracle: ExposureValueOracle,
  options: VwKamaCudaBatchOptions,
  includeHighLow: boolean,
  strategyTemperatures?: Float32Array,
  strategyQuadraticVolatilities?: Float32Array,
): string {
  return [
    objectId(candles),
    objectId(oracle),
    strategyTemperatures ? objectId(strategyTemperatures) : "generated",
    strategyQuadraticVolatilities
      ? objectId(strategyQuadraticVolatilities)
      : "generated",
    options.scoreStartIndex,
    options.intervalMs,
    oracle.holdingPeriodSteps,
    includeHighLow ? 1 : 0,
    options.valueDistillation?.strategyVolatilityScaling ? 1 : 0,
    options.valueDistillation?.lossConfig.entropyGapLambda ?? 0,
    options.valueDistillation?.lossConfig.stateMutualInformationLambda ?? 0,
    options.valueDistillation?.lossConfig.oracleMutualInformationLambda ?? 0,
    options.valueDistillation?.lossConfig.oracleMutualInformationMode ?? "approximate",
    options.valueDistillation?.lossConfig.mutualInformationBins ?? 15,
  ].join(":");
}

function preciseOracleMutualInformationEnabled(config: ExposureValueDistillationLossConfig): boolean {
  return config.oracleMutualInformationLambda > 0
    && config.oracleMutualInformationMode === "precise";
}

function binnedOracleProbabilities(oracle: ExposureValueOracle, bins: number): Float32Array {
  if (!oracle.probabilities) {
    throw new Error("Precise CUDA oracle mutual information requires retained oracle probabilities.");
  }
  let cache = oracleBinCaches.get(oracle);
  if (!cache) {
    cache = new Map();
    oracleBinCaches.set(oracle, cache);
  }
  const cached = cache.get(bins);
  if (cached) return cached;
  const result = new Float32Array(oracle.means.length * bins);
  const gridSize = oracle.grid.length;
  for (let candle = 0; candle < oracle.means.length; candle += 1) {
    const sourceOffset = candle * gridSize;
    const targetOffset = candle * bins;
    for (let gridIndex = 0; gridIndex < gridSize; gridIndex += 1) {
      const bin = Math.min(bins - 1, Math.floor(gridIndex * bins / gridSize));
      result[targetOffset + bin] += oracle.probabilities[sourceOffset + gridIndex]!;
    }
  }
  cache.set(bins, result);
  return result;
}

function objectId(value: object): number {
  const existing = objectIds.get(value);
  if (existing !== undefined) return existing;
  const created = nextObjectId++;
  objectIds.set(value, created);
  return created;
}

function candidateNeedsDmi(candidate: VwKamaParameters): boolean {
  return (candidate.confirmationMix ?? 0) > 0 && (candidate.confirmationDmiWeight ?? 0) > 0;
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
    samples(parameters.strategyQuadraticVolatilityMs ?? 60 * 60_000),
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
    parameters.strategyTemperature ?? 0.001,
    parameters.strategyQuadraticScale ?? 0,
    options.warmupMultiple,
  ];
  if (integers.length !== INT_PARAMETER_COUNT || floats.length !== 31) {
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
    distillationWeightedStrategyEntropy: buffer.readDoubleLE(offset + 152),
    distillationWeightedEntropyGap: buffer.readDoubleLE(offset + 160),
    distillationStateMutualInformation: buffer.readDoubleLE(offset + 168),
    distillationOracleMutualInformation: buffer.readDoubleLE(offset + 176),
    distillationMixedLoss: buffer.readDoubleLE(offset + 184),
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
