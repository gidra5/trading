import fs from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import { gunzipSync } from "node:zlib";
import {
  evaluateVwKamaOracle,
  noiseSignalRatio,
  perfectMarginOracle,
  prepareHandcraftedIndicatorStates,
  DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
  HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS,
  DEFAULT_DIRECT_INDICATOR_PARAMETERS,
  exposureProbabilityTransitionCrossEntropy,
  predictHandcraftedIndicatorRegret,
  prepareHandcraftedIndicatorStateAt,
  prepareExposureValueOracle,
  prepareSparseExposureValueOracle,
  populateSparseExposureValueOracle,
  rebalanceEquityFactor,
  prepareExposureValueOracleCuda,
  normalizeExposureValueDistillationLossConfig,
  resolveVwKamaHoldingPeriodSteps,
  rescoreVwKamaEvaluation,
  truncateExposureValueOracle,
  vwKamaScore,
  vwKamaCudaStatus,
  VW_KAMA_SCORE_VERSION,
  type Candle,
  type BacktestOraclePoint,
  type PerfectMarginOracleResult,
  type VwKamaAccuracyMetrics,
  type VwKamaCandleRangeRequest,
  type VwKamaCandleRangeResponse,
  type VwKamaEvaluation,
  type VwKamaInspectorCatalog,
  type VwKamaInspectorRequest,
  type VwKamaInspectorResponse,
  type VwKamaInspectorWindow,
  type VwKamaHistoricalAveragesRequest,
  type VwKamaHistoricalAveragesResponse,
  type VwKamaIndicatorPoint,
  type VwKamaParameters,
  type VwKamaPreset,
  type VwKamaPredictorFitRequest,
  type VwKamaPredictorFitResponse,
  type VwKamaTransition,
  type ExposureReturnMetrics,
  type ExposureValueOracle,
  type HandcraftedIndicatorPredictorParameters,
  type DirectIndicatorPredictorParameters,
} from "@trading/bot-algo";
import { fetchBinanceSpotDailyShard } from "./binance-history-cache.js";
import { HANDCRAFTED_PREDICTOR_PRESETS } from "./handcrafted-predictor-presets.js";
import { DIRECT_INDICATOR_PREDICTOR_PRESETS } from "./direct-indicator-predictor-presets.js";

const DAY_MS = 86_400_000;
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const MAX_WARMUP_MS = 3 * DAY_MS;
const MAX_CHART_CANDLES = 2_000;
const MAX_VALUE_DISTRIBUTIONS = 250;
const MIN_RANGE_CANDLES = 100;
const MAX_RANGE_CANDLES = 5_000;
const MIN_CUDA_ORACLE_HOLDING_STEPS = 2;
let cudaValueOracleFallbackReported = false;
const SCALES = [1_000, 5_000, 15_000, 60_000, 300_000, 900_000, 3_600_000]
  .map((intervalMs) => ({ label: duration(intervalMs), intervalMs }));
const STATIC_WINDOWS = [
  window("fit-full", "Optimizer fit · full", "Optimizer fit", "2025-03-19", "2025-11-13"),
  window("fit-1", "Optimizer fit 1", "Optimizer fit", "2025-03-19", "2025-05-17"),
  window("fit-2", "Optimizer fit 2", "Optimizer fit", "2025-05-18", "2025-07-16"),
  window("fit-3", "Optimizer fit 3", "Optimizer fit", "2025-07-17", "2025-09-14"),
  window("fit-4", "Optimizer fit 4", "Optimizer fit", "2025-09-15", "2025-11-13"),
  window("sideways-churn-2022-07", "Sideways · highest OHLC churn", "Sideways / choppy", "2022-07-28", "2022-08-03"),
  window("sideways-churn-2022-05", "Sideways · highest close churn", "Sideways / choppy", "2022-05-14", "2022-05-20"),
  window("sideways-churn-2021-12", "Sideways · very choppy (Dec 2021)", "Sideways / choppy", "2021-12-14", "2021-12-20"),
  window("sideways-churn-2021-09", "Sideways · very choppy (Sep 2021)", "Sideways / choppy", "2021-09-08", "2021-09-14"),
  window("sideways-churn-2023-03", "Sideways · choppy near-flat", "Sideways / choppy", "2023-03-18", "2023-03-24"),
  window("regime-up-2023-03", "Uptrend · +35.95%", "Regime", "2023-03-11", "2023-03-17"),
  window("regime-flat-2026-04", "Sideways · +0.009%", "Regime", "2026-04-22", "2026-04-28"),
  window("regime-down-2022-06", "Downtrend · -33.26%", "Regime", "2022-06-12", "2022-06-18"),
  window("shape-up-low-2024-02", "Uptrend · low churn", "3-day shape", "2024-02-24", "2024-02-26"),
  window("shape-up-high-2022-06", "Uptrend · high churn", "3-day shape", "2022-06-19", "2022-06-21"),
  window("shape-down-low-2023-06", "Downtrend · low churn", "3-day shape", "2023-06-03", "2023-06-05"),
  window("shape-down-high-2022-06", "Downtrend · high churn", "3-day shape", "2022-06-13", "2022-06-15"),
  window("shape-flat-high-bias-2021-10", "Sideways · high bias / churn", "3-day shape", "2021-10-19", "2021-10-21"),
  window("shape-flat-high-bias-low-2025-02", "Sideways · high bias / low churn", "3-day shape", "2025-02-14", "2025-02-16"),
  window("shape-flat-low-bias-2024-07", "Sideways · low bias / churn", "3-day shape", "2024-07-07", "2024-07-09"),
  window("shape-flat-low-bias-low-2025-07", "Sideways · low bias / low churn", "3-day shape", "2025-07-04", "2025-07-06"),
  window("shape-flat-mid-bias-2024-01", "Sideways · mid bias / churn", "3-day shape", "2024-01-02", "2024-01-04"),
  window("shape-flat-mid-bias-low-2023-09", "Sideways · mid bias / low churn", "3-day shape", "2023-09-15", "2023-09-17"),
  window("sharpe-up-3d-2024-11", "Uptrend · high Sharpe (3d, Nov 2024)", "Sharpe", "2024-11-09", "2024-11-11"),
  window("sharpe-up-3d-2023-12", "Uptrend · high Sharpe (3d, Dec 2023)", "Sharpe", "2023-12-03", "2023-12-05"),
  window("sharpe-down-3d-2026-06", "Downtrend · high Sharpe (3d, Jun 2026)", "Sharpe", "2026-06-01", "2026-06-03"),
  window("sharpe-down-3d-2023-03", "Downtrend · high Sharpe (3d, Mar 2023)", "Sharpe", "2023-03-07", "2023-03-09"),
  window("sharpe-up-7d-2023-12", "Uptrend · high Sharpe (7d, Dec 2023)", "Sharpe", "2023-11-29", "2023-12-05"),
  window("sharpe-up-7d-2024-11", "Uptrend · high Sharpe (7d, Nov 2024)", "Sharpe", "2024-11-05", "2024-11-11"),
  window("sharpe-down-7d-2023-03", "Downtrend · high Sharpe (7d, Mar 2023)", "Sharpe", "2023-03-03", "2023-03-09"),
  window("sharpe-down-7d-2026-06", "Downtrend · high Sharpe (7d, Jun 2026)", "Sharpe", "2026-05-27", "2026-06-02"),
  window("failure-down-3d-2022-06", "Known miss · -22.7% (3d)", "Known miss", "2022-06-11", "2022-06-13"),
  window("failure-down-7d-2022-06", "Known miss · June selloff (7d)", "Known miss", "2022-06-07", "2022-06-13"),
] satisfies VwKamaInspectorWindow[];

const BASELINE_GLOBAL_PARAMETERS = {
  efficiencyMs: 265_846,
  fastMs: 1_113_418,
  slowMs: 64_800_000,
  power: 0.62547,
  volumeMs: 283_525,
  volumeCap: 5.08117,
  volumePower: 0.48543,
  rateMode: "relative" as const,
  deadbandBpsHour: 30,
  deadbandMode: "hold" as const,
  hysteresisReleaseRatio: 0.25,
  thresholdLookbackMs: 2_457_073,
  thresholdNoiseResponse: "proportional" as const,
  thresholdNoiseMultiplier: 0,
  thresholdInverseMaxBpsHour: 0,
  thresholdInverseNoiseScaleBpsHour: 30,
  signalFrictionFraction: 1,
  strategyTemperature: 0.001,
  strategyQuadraticScale: 0,
  strategyQuadraticVolatilityMs: 60 * 60_000,
  strategyNormalMixture: 0,
  strategyNormalSigma: 25,
  buyMaxFraction: 1,
  sellMaxFraction: 1,
  buySizingSigmaBpsHour: 1e12,
  sellSizingSigmaBpsHour: 1e12,
  agreementMode: "sizing" as const,
  confirmationMix: 0,
  confirmationMinQuality: 0,
  confirmationAccelerationLookbackMs: 60 * 60_000,
  confirmationDistanceLookbackMs: 60 * 60_000,
  confirmationAccelerationWeight: 1,
  confirmationDistanceWeight: 1,
  confirmationEmaMs: 60 * 60_000,
  confirmationEmaThresholdBpsHour: 0,
  confirmationEmaWeight: 0,
  confirmationEmaGateStrength: 0,
  confirmationRsiMs: 14 * 60_000,
  confirmationRsiThreshold: 0,
  confirmationRsiWeight: 0,
  confirmationDmiMs: 14 * 60_000,
  confirmationDmiWeight: 0,
  confirmationAdxThreshold: 20,
  confirmationBias: 0,
  meanReversionEfficiencyMs: 265_846,
  meanReversionFastMs: 1_113_418,
  meanReversionSlowMs: 60 * 60_000,
  meanReversionVolatilityMs: 60 * 60_000,
  meanReversionSuppressionThreshold: 1,
  meanReversionReversalThreshold: 0,
};

const DEFAULT_REQUEST: VwKamaInspectorRequest = {
  windowId: "shape-up-low-2024-02",
  latestDays: 7,
  intervalMs: 1_000,
  parameters: structuredClone(BASELINE_GLOBAL_PARAMETERS),
  predictor: {
    model: "handcrafted",
    handcraftedParameters: { ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS },
    directIndicatorParameters: { ...DEFAULT_DIRECT_INDICATOR_PARAMETERS },
  },
  oracleFriction: 0.00175,
  matchWindowMs: 2 * 3_600_000,
  timingHalfLifeMs: 10 * 60_000,
  warmupMultiple: 3,
  valueDistillation: {
    gridSize: 151,
    minExposure: -100,
    maxExposure: 100,
    maxEffectiveExposure: 250,
    initialExposure: 0,
    holdingPeriodMode: "fixed",
    holdingPeriodMs: 60_000,
    valueHorizonMode: "fixed",
    valueHorizonMs: 60 * 60_000,
    horizonEndMode: "truncate",
    oracleTemperature: 0.01,
    strategyVolatilityScaling: false,
    opportunityEpsilon: 0.000001,
    quoteLendRate: 0,
    quoteBorrowRate: 0,
    assetBorrowRate: 0,
    entropyGapLambda: 0,
    stateMutualInformationLambda: 0,
    oracleMutualInformationLambda: 0,
    oracleMutualInformationMode: "approximate",
    mutualInformationBins: 15,
  },
};

const GLOBAL_PRESETS: VwKamaPreset[] = [
  {
    id: "global-clean-v0182",
    label: "Global · score-v2 v0182",
    scope: "global",
    windowId: null,
    parameters: structuredClone(BASELINE_GLOBAL_PARAMETERS),
    source: "Score-v2 chronological validation winner; full sizing",
  },
  {
    id: "global-clean-k0050",
    label: "Global · score-v2 k0050",
    scope: "global",
    windowId: null,
    parameters: {
      efficiencyMs: 1_420_250,
      fastMs: 976_600,
      slowMs: 976_600,
      power: 0.79612,
      volumeMs: 82_113,
      volumeCap: 4.27354,
      volumePower: 0,
      deadbandBpsHour: 21.21468,
      deadbandMode: "hold",
      thresholdLookbackMs: 900_000,
      thresholdNoiseMultiplier: 0,
      buyMaxFraction: 1,
      sellMaxFraction: 1,
      buySizingSigmaBpsHour: 1e12,
      sellSizingSigmaBpsHour: 1e12,
    },
    source: "Score-v2 chronological canonical finalist; full sizing",
  },
  {
    id: "global-runtime-k0021",
    label: "Global · runtime k0021",
    scope: "global",
    windowId: null,
    parameters: {
      efficiencyMs: 14 * 60_000,
      fastMs: 28 * 60_000,
      slowMs: 153 * 60_000,
      power: 0.49045,
      volumeMs: 130 * 60_000,
      volumeCap: 2.65003,
      volumePower: 0,
      deadbandBpsHour: 67.56654,
      deadbandMode: "flat",
      thresholdLookbackMs: 60 * 60_000,
      thresholdNoiseMultiplier: 0,
    },
    source: "Validation-selected multi-scale baseline",
  },
  {
    id: "global-refined-v0016",
    label: "Global · refined v0016",
    scope: "global",
    windowId: null,
    parameters: {
      efficiencyMs: 1_856_036,
      fastMs: 1_873_481,
      slowMs: 3_701_806,
      power: 0.61357,
      volumeMs: 440_028,
      volumeCap: 1.97271,
      volumePower: 1.72738,
      deadbandBpsHour: 0.44351,
      deadbandMode: "hold",
      thresholdLookbackMs: 60 * 60_000,
      thresholdNoiseMultiplier: 0,
    },
    source: "Refined one-to-one 60/40 validation winner",
  },
];

interface CachedWindow {
  id: string;
  sourceEndTime: number;
  sourceIntervalMs: number;
  source: Promise<Candle[]>;
  scales: Map<number, Promise<Candle[][]>>;
  oracles: Map<string, Promise<PerfectMarginOracleResult>>;
  valueOracles: Map<string, CachedExposureOracle>;
  analyses: Map<string, CachedAnalysis>;
}

interface CachedExposureOracle {
  value: Promise<ExposureValueOracle>;
  populated: Set<number> | null;
  population: Promise<void>;
}

interface EvaluatedSegment {
  result: VwKamaEvaluation;
  candles: Candle[];
  scoreStart: number;
}

interface CachedAnalysis {
  sourceSegmentCount: number;
  evaluated: EvaluatedSegment[];
}

interface MissingDailyShardRequest {
  dataDir: string;
  date: string;
  day: number;
}

export interface KamaInspectorEngineOptions {
  now?: () => number;
  fetchDailyShard?: (request: MissingDailyShardRequest) => Promise<void>;
}

type InspectorResult = VwKamaInspectorResponse
  | VwKamaCandleRangeResponse
  | VwKamaPredictorFitResponse
  | VwKamaHistoricalAveragesResponse;

interface PendingRequest {
  resolve: (result: InspectorResult) => void;
  reject: (error: Error) => void;
  cleanup: () => void;
}

interface InspectorWorkerResponse {
  id: number;
  result?: InspectorResult;
  error?: string;
}

export class KamaInspector {
  private worker: Worker | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, PendingRequest>();

  constructor(private readonly dataDir: string) {}

  catalog(): VwKamaInspectorCatalog {
    return inspectorCatalog(this.dataDir);
  }

  analyze(input: VwKamaInspectorRequest, signal?: AbortSignal): Promise<VwKamaInspectorResponse> {
    return this.request("analyze", input, signal);
  }

  candles(input: VwKamaCandleRangeRequest, signal?: AbortSignal): Promise<VwKamaCandleRangeResponse> {
    return this.request("candles", input, signal);
  }

  fitPredictor(input: VwKamaPredictorFitRequest, signal?: AbortSignal): Promise<VwKamaPredictorFitResponse> {
    return this.request("fit-predictor", input, signal);
  }

  historicalAverages(
    input: VwKamaHistoricalAveragesRequest,
    signal?: AbortSignal,
  ): Promise<VwKamaHistoricalAveragesResponse> {
    return this.request("historical-averages", input, signal);
  }

  private request<T extends InspectorResult>(
    type: "analyze" | "candles" | "fit-predictor" | "historical-averages",
    input: VwKamaInspectorRequest
      | VwKamaCandleRangeRequest
      | VwKamaPredictorFitRequest
      | VwKamaHistoricalAveragesRequest,
    signal?: AbortSignal,
  ): Promise<T> {
    const worker = this.getWorker();
    const id = this.nextId++;
    const cancelFlag = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
    return new Promise<T>((resolve, reject) => {
      const cleanup = (): void => signal?.removeEventListener("abort", cancel);
      const cancel = (): void => {
        Atomics.store(cancelFlag, 0, 1);
        if (!this.pending.delete(id)) return;
        cleanup();
        reject(new Error("VW-KAMA request cancelled."));
      };
      if (signal?.aborted) {
        Atomics.store(cancelFlag, 0, 1);
        reject(new Error("VW-KAMA request cancelled."));
        return;
      }
      signal?.addEventListener("abort", cancel, { once: true });
      this.pending.set(id, {
        resolve: (result) => {
          cleanup();
          resolve(result as T);
        },
        reject: (error) => {
          cleanup();
          reject(error);
        },
        cleanup,
      });
      worker.postMessage({ type, id, input, cancelFlag });
    });
  }

  async close(): Promise<void> {
    const worker = this.worker;
    this.worker = null;
    const error = new Error("VW-KAMA inspector stopped");
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    if (worker) await worker.terminate();
  }

  private getWorker(): Worker {
    if (this.worker) return this.worker;
    const source = import.meta.url.endsWith(".ts")
      ? new URL("./kama-inspector-worker.ts", import.meta.url)
      : new URL("./kama-inspector-worker.js", import.meta.url);
    const worker = import.meta.url.endsWith(".ts")
      ? new Worker(
          `import("tsx/esm/api").then(({tsImport}) => tsImport(${JSON.stringify(source.href)}, {parentURL:${JSON.stringify(import.meta.url)}}));`,
          { eval: true },
        )
      : new Worker(source, { execArgv: [] });
    worker.on("message", (message: InspectorWorkerResponse) => {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.result) pending.resolve(message.result);
      else pending.reject(new Error(message.error ?? "VW-KAMA worker failed"));
    });
    worker.on("error", (error) => this.failWorker(worker, error));
    worker.on("exit", (code) => {
      if (code !== 0) this.failWorker(worker, new Error(`VW-KAMA worker exited with code ${code}`));
      else if (this.worker === worker) this.worker = null;
    });
    worker.postMessage({ type: "init", dataDir: this.dataDir });
    this.worker = worker;
    return worker;
  }

  private failWorker(worker: Worker, error: Error): void {
    if (this.worker !== worker) return;
    this.worker = null;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}

export class KamaInspectorEngine {
  private cache: CachedWindow | null = null;
  private readonly pendingShardFetches = new Map<string, Promise<void>>();
  private readonly now: () => number;
  private readonly fetchDailyShard: (request: MissingDailyShardRequest) => Promise<void>;

  constructor(
    private readonly dataDir: string,
    options: KamaInspectorEngineOptions = {},
  ) {
    this.now = options.now ?? Date.now;
    this.fetchDailyShard = options.fetchDailyShard ?? fetchBinanceSpotDailyShard;
  }

  catalog(): VwKamaInspectorCatalog {
    return inspectorCatalog(this.dataDir, this.now());
  }

  async analyze(input: VwKamaInspectorRequest, cancelFlag?: Int32Array): Promise<VwKamaInspectorResponse> {
    throwIfInspectorCancelled(cancelFlag);
    const startedAt = performance.now();
    const request = normalizeRequest(input);
    const selected = resolveInspectorWindow(request.windowId, request.latestDays, this.now());
    validateWindowScale(selected, request.intervalMs);
    const sourceEndTime = selected.endTime + (request.valueDistillation!.valueHorizonMode === "fixed"
      && request.valueDistillation!.horizonEndMode === "extend"
      ? request.valueDistillation!.valueHorizonMs
      : 0);
    const cached = this.cachedWindow(selected, sourceEndTime);
    const cacheKey = analysisCacheKey(request);
    let base = cached.analyses.get(cacheKey);
    if (!base) {
      base = await (async (): Promise<CachedAnalysis> => {
        const segments = await this.scaledSegments(cached, request.intervalMs);
        const warmupMs = candidateWarmupMs(request);
        const evaluated: EvaluatedSegment[] = [];
        for (const [index, valueCandles] of segments.entries()) {
          throwIfInspectorCancelled(cancelFlag);
          const scoreEndIndex = candleLowerBound(valueCandles, selected.endTime);
          const candles = valueCandles.slice(0, scoreEndIndex);
          if (candles.length === 0) continue;
          const scoreStart = Math.max(
            selected.startTime,
            candles[0]!.openTime + (index > 0 ? warmupMs : 0),
          );
          if (scoreStart >= selected.endTime || candles.at(-1)!.openTime < scoreStart) continue;
          const scoreStartIndex = candleLowerBound(candles, scoreStart);
          const maxPoints = Math.max(100, Math.floor(MAX_CHART_CANDLES / segments.length));
          const oracle = await this.oracle(cached, request, candles);
          const valueOracle = await this.exposureOracle(
            cached,
            request,
            valueCandles,
            scoreStartIndex,
            scoreEndIndex,
            oracle.stateCodes,
            sampledIndexes(candles.length, scoreStartIndex, maxPoints),
            cancelFlag,
          );
          evaluated.push({
            candles,
            scoreStart,
            result: evaluateVwKamaOracle(candles, {
              ...request,
              scoreStartTime: scoreStart,
              scoreStartIndex,
              maxPoints,
              maxDistributionPoints: Math.max(
                25,
                Math.floor(MAX_VALUE_DISTRIBUTIONS / segments.length),
              ),
              cancelFlag,
              oracleResult: oracle,
              valueDistillation: {
                oracle: valueOracle,
                strategyVolatilityScaling: request.valueDistillation!.strategyVolatilityScaling,
                lossConfig: request.valueDistillation!,
                ...handcraftedPredictorOptions(request, candles),
              },
            }),
          });
        }
        if (evaluated.length === 0) {
          throw new Error("No continuous candles remain after VW-KAMA warmup.");
        }
        return { sourceSegmentCount: segments.length, evaluated };
      })();
      cached.analyses.set(cacheKey, base);
      while (cached.analyses.size > 2) {
        const oldest = cached.analyses.keys().next().value as string | undefined;
        if (oldest === undefined) break;
        cached.analyses.delete(oldest);
      }
    }
    const evaluated = base.evaluated.map((item): EvaluatedSegment => ({
      ...item,
      result: rescoreVwKamaEvaluation(item.result, request),
    }));
    const result = combineEvaluations(evaluated, request);
    return {
      window: { ...selected },
      intervalMs: request.intervalMs,
      sourceSegmentCount: base.sourceSegmentCount,
      scoredSegmentCount: evaluated.length,
      elapsedMs: Math.round(performance.now() - startedAt),
      ...result,
    };
  }

  async candles(input: VwKamaCandleRangeRequest, cancelFlag?: Int32Array): Promise<VwKamaCandleRangeResponse> {
    throwIfInspectorCancelled(cancelFlag);
    const { request, selected } = normalizeCandleRangeRequest(input, this.now());
    const sourceEndTime = selected.endTime + (request.valueDistillation!.valueHorizonMode === "fixed"
      && request.valueDistillation!.horizonEndMode === "extend"
      ? request.valueDistillation!.valueHorizonMs
      : 0);
    const cached = this.cachedWindow(selected, sourceEndTime);
    const segments = await this.scaledSegments(cached, request.intervalMs);
    const rendered = renderCandleRange(
      segments,
      request.intervalMs,
      request.startTime,
      request.endTime,
      request.maxCandles,
    );
    const renderedTimes = rendered.candles.map((candle) => candle.closeTime);
    const indicatorPoints: VwKamaIndicatorPoint[] = [];
    const valueDistributions: VwKamaCandleRangeResponse["valueDistributions"] = [];
    const valueOraclePath: VwKamaCandleRangeResponse["valueOraclePath"] = [];
    const valueCandidatePath: VwKamaCandleRangeResponse["valueCandidatePath"] = [];
    const warmupMs = candidateWarmupMs(request);
    for (const [index, segment] of segments.entries()) {
      throwIfInspectorCancelled(cancelFlag);
      const firstTime = segment[0]?.closeTime;
      const lastTime = segment.at(-1)?.closeTime;
      if (firstTime === undefined || lastTime === undefined) continue;
      const traceTimes = renderedTimes.filter((time) => time >= firstTime && time <= lastTime);
      if (traceTimes.length === 0) continue;
      const distributionTraceTimes = sampleEven(traceTimes, MAX_VALUE_DISTRIBUTIONS);
      const endIndex = candleLowerBound(segment, traceTimes.at(-1)! + 1);
      const candles = segment.slice(0, endIndex);
      const scoreStart = Math.min(
        Math.max(selected.startTime, segment[0]!.openTime + (index > 0 ? warmupMs : 0)),
        candles.at(-1)!.openTime,
      );
      const scoreStartIndex = candleLowerBound(segment, scoreStart);
      const oracle = await this.oracle(cached, request, segment);
      const valueOracle = await this.exposureOracle(
        cached,
        request,
        segment,
        scoreStartIndex,
        candleLowerBound(segment, selected.endTime),
        oracle.stateCodes,
        distributionTraceTimes.map((time) => candleIndexAtClose(segment, time)),
        cancelFlag,
      );
      const evaluation = evaluateVwKamaOracle(candles, {
        ...request,
        scoreStartTime: scoreStart,
        scoreStartIndex,
        maxPoints: traceTimes.length,
        maxDistributionPoints: distributionTraceTimes.length,
        traceTimes,
        cancelFlag,
        oracleResult: oracle,
        valueDistillation: {
          oracle: valueOracle,
          score: false,
          strategyVolatilityScaling: request.valueDistillation!.strategyVolatilityScaling,
          lossConfig: request.valueDistillation!,
          ...handcraftedPredictorOptions(request, candles),
        },
      });
      indicatorPoints.push(...evaluation.indicatorPoints);
      valueDistributions.push(...evaluation.valueDistributions);
      valueOraclePath.push(...evaluation.valueOraclePath);
      valueCandidatePath.push(...evaluation.valueCandidatePath);
    }
    indicatorPoints.sort((left, right) => left.time - right.time);
    return {
      windowId: selected.id,
      intervalMs: request.intervalMs,
      renderIntervalMs: rendered.intervalMs,
      startTime: request.startTime,
      endTime: request.endTime,
      sourceCandleCount: rendered.sourceCount,
      candles: rendered.candles,
      kamaSeries: {
        index: -1,
        windowSec: request.parameters.slowMs / 1_000,
        label: "VW-KAMA",
        color: "#f472b6",
        points: indicatorPoints.map((point) => ({ time: point.time, value: point.kama })),
      },
      indicatorPoints,
      valueDistributions: valueDistributions.sort((left, right) => left.time - right.time),
      valueOraclePath: valueOraclePath.sort((left, right) => left.time - right.time),
      valueCandidatePath: valueCandidatePath.sort((left, right) => left.time - right.time),
    };
  }

  async historicalAverages(
    input: VwKamaHistoricalAveragesRequest,
    cancelFlag?: Int32Array,
  ): Promise<VwKamaHistoricalAveragesResponse> {
    throwIfInspectorCancelled(cancelFlag);
    const request = normalizeRequest(input);
    if (![input.time, input.lookbackMs,
      input.holdingPeriodMs, input.valueHorizonMs].every(Number.isFinite)) {
      throw new Error("Historical oracle time and durations must be finite.");
    }
    if (input.lookbackMs <= 0
      || input.holdingPeriodMs <= 0 || input.valueHorizonMs < input.holdingPeriodMs) {
      throw new Error("Historical oracle lookback and H must be positive, with T at least H.");
    }
    const selected = resolveInspectorWindow(request.windowId, request.latestDays, this.now());
    validateWindowScale(selected, request.intervalMs);
    const sourceEndTime = selected.endTime + (request.valueDistillation!.valueHorizonMode === "fixed"
      && request.valueDistillation!.horizonEndMode === "extend"
      ? request.valueDistillation!.valueHorizonMs
      : 0);
    const cached = this.cachedWindow(selected, sourceEndTime);
    const segments = await this.scaledSegments(cached, request.intervalMs);
    for (const segment of segments) {
      const selectedIndex = segment.findIndex((candle) => candle.closeTime === input.time);
      if (selectedIndex < 0) continue;
      const scoreEndIndex = candleLowerBound(segment, selected.endTime);
      const scoreStart = Math.max(selected.startTime, segment[0]!.openTime);
      const scoreStartIndex = candleLowerBound(segment, scoreStart);
      const perfectOracle = await this.oracle(cached, request, segment);
      const mainHoldingSteps = resolveVwKamaHoldingPeriodSteps(
        segment.slice(0, scoreEndIndex),
        scoreStartIndex,
        perfectOracle.stateCodes,
        request.intervalMs,
        request.valueDistillation!,
      );
      const mainHorizonSteps = request.valueDistillation!.valueHorizonMode === "fixed"
        ? Math.max(1, Math.round(request.valueDistillation!.valueHorizonMs / request.intervalMs))
        : Math.max(1, scoreEndIndex - scoreStartIndex - 1);
      const holdingPeriodSteps = Math.max(1, Math.round(input.holdingPeriodMs / request.intervalMs));
      const valueHorizonSteps = Math.max(1, Math.round(input.valueHorizonMs / request.intervalMs));
      if (holdingPeriodSteps > mainHoldingSteps || valueHorizonSteps > mainHorizonSteps) {
        throw new Error("Historical oracle H and T cannot exceed the main oracle's resolved H and T.");
      }
      if (valueHorizonSteps < holdingPeriodSteps) {
        throw new Error("Historical oracle T must be at least H after candle-scale rounding.");
      }
      const latestOrigin = selectedIndex - valueHorizonSteps;
      if (latestOrigin < 0) {
        throw new Error("The selected candle does not have one completed historical oracle horizon.");
      }
      const originsForLookback = (lookbackMs: number): number[] => {
        const lookbackSteps = Math.max(1, Math.ceil(lookbackMs / request.intervalMs));
        const earliestOrigin = Math.max(0, latestOrigin - lookbackSteps + 1);
        return sampleEven(
          Array.from(
            { length: latestOrigin - earliestOrigin + 1 },
            (_, index) => earliestOrigin + index,
          ),
          MAX_VALUE_DISTRIBUTIONS,
        );
      };
      const origins = originsForLookback(input.lookbackMs);
      const historyStartIndex = origins[0]!;
      const historyCandles = segment.slice(historyStartIndex, selectedIndex + 1);
      const localOrigins = origins.map((index) => index - historyStartIndex);
      const prices = historyCandles.map((candle) => candle.close);
      const config = request.valueDistillation!;
      const oracle = prepareSparseExposureValueOracle(prices, {
        scoreStartIndex: 0,
        holdingPeriodSteps,
        valueHorizonSteps,
        friction: request.oracleFriction,
        gridSize: config.gridSize,
        minExposure: config.minExposure,
        maxExposure: config.maxExposure,
        maxEffectiveExposure: config.maxEffectiveExposure,
        initialExposure: config.initialExposure,
        terminalIndex: prices.length - 1,
        temperature: config.oracleTemperature,
        opportunityEpsilon: 0,
        quoteLendRate: hourlyRatePerCandle(config.quoteLendRate, request.intervalMs),
        quoteBorrowRate: hourlyRatePerCandle(config.quoteBorrowRate, request.intervalMs),
        assetBorrowRate: hourlyRatePerCandle(config.assetBorrowRate, request.intervalMs),
        includeActionValues: true,
        cancelFlag,
      }, localOrigins);
      const averages = historicalOracleAverage(oracle, localOrigins);
      return {
        time: input.time,
        holdingPeriodMs: holdingPeriodSteps * request.intervalMs,
        valueHorizonMs: valueHorizonSteps * request.intervalMs,
        targetExposures: Array.from(oracle.grid),
        currentExposures: Array.from(oracle.currentGrid),
        regretProbabilities: Array.from(averages.regretProbabilities),
        maximumReturnGapProbabilities: Array.from(averages.maximumReturnGapProbabilities),
        sampleCount: averages.sampleCount,
        averageRegret: averages.averageRegret,
        averageMaximumReturn: averages.averageMaximumReturn,
      };
    }
    throw new Error("Historical oracle candle is not present in the selected continuous history.");
  }

  async fitPredictor(input: VwKamaPredictorFitRequest, cancelFlag?: Int32Array): Promise<VwKamaPredictorFitResponse> {
    throwIfInspectorCancelled(cancelFlag);
    const startedAt = performance.now();
    const request = normalizeRequest(input);
    if (request.predictor?.model !== "handcrafted") {
      throw new Error("Per-candle fitting is available only for the handcrafted forecast model.");
    }
    if (!Number.isFinite(input.time)) throw new Error("Predictor fit time must be finite.");
    const selected = resolveInspectorWindow(request.windowId, request.latestDays, this.now());
    validateWindowScale(selected, request.intervalMs);
    const sourceEndTime = selected.endTime + (request.valueDistillation!.valueHorizonMode === "fixed"
      && request.valueDistillation!.horizonEndMode === "extend"
      ? request.valueDistillation!.valueHorizonMs
      : 0);
    const cached = this.cachedWindow(selected, sourceEndTime);
    const segments = await this.scaledSegments(cached, request.intervalMs);
    const warmupMs = candidateWarmupMs(request);

    for (const [segmentIndex, candles] of segments.entries()) {
      throwIfInspectorCancelled(cancelFlag);
      const candleIndex = candles.findIndex((candle) => candle.closeTime === input.time);
      if (candleIndex < 0) continue;
      const scoreEndIndex = candleLowerBound(candles, selected.endTime);
      const scoreStart = Math.max(
        selected.startTime,
        candles[0]!.openTime + (segmentIndex > 0 ? warmupMs : 0),
      );
      const scoreStartIndex = candleLowerBound(candles, scoreStart);
      if (candleIndex < scoreStartIndex || candleIndex >= scoreEndIndex) {
        throw new Error("Predictor fit candle is outside the scored inspector range.");
      }
      const oracle = await this.oracle(cached, request, candles);
      const valueOracle = await this.exposureOracle(
        cached,
        request,
        candles,
        scoreStartIndex,
        scoreEndIndex,
        oracle.stateCodes,
        [candleIndex],
        cancelFlag,
      );
      const prices = candles.map((candle) => candle.close);
      const fit = fitHandcraftedPredictorAtCandle(
        prices,
        candleIndex,
        request.intervalMs,
        valueOracle,
        request.predictor.handcraftedParameters,
        input.time,
        cancelFlag,
      );
      const states = prepareHandcraftedIndicatorStates(prices, request.intervalMs, fit.parameters);
      const evaluation = evaluateVwKamaOracle(candles, {
        ...request,
        scoreStartTime: scoreStart,
        scoreStartIndex,
        maxPoints: 1,
        traceTimes: [input.time],
        cancelFlag,
        oracleResult: oracle,
        valueDistillation: {
          oracle: valueOracle,
          strategyVolatilityScaling: request.valueDistillation!.strategyVolatilityScaling,
          lossConfig: request.valueDistillation!,
          handcraftedPredictor: { parameters: fit.parameters, states },
        },
      });
      const point = evaluation.valueDistributions.find((item) => item.time === input.time);
      if (!point) throw new Error("Predictor fit did not produce the requested candle distribution.");
      return {
        time: input.time,
        point,
        baselineCrossEntropy: fit.baselineCrossEntropy,
        fittedCrossEntropy: fit.crossEntropy,
        iterations: fit.iterations,
        elapsedMs: Math.round(performance.now() - startedAt),
      };
    }
    throw new Error("Predictor fit candle is not present in the selected continuous history.");
  }

  private cachedWindow(selected: VwKamaInspectorWindow, sourceEndTime: number): CachedWindow {
    const cacheId = `${selected.id}:${selected.startTime}`;
    if (this.cache?.id === cacheId && this.cache.sourceEndTime === sourceEndTime) return this.cache;
    const source = this.loadSource(selected, sourceEndTime);
    const cached = {
      id: cacheId,
      sourceEndTime,
      sourceIntervalMs: selected.sourceIntervalMs,
      source,
      scales: new Map<number, Promise<Candle[][]>>(),
      oracles: new Map<string, Promise<PerfectMarginOracleResult>>(),
      valueOracles: new Map<string, CachedExposureOracle>(),
      analyses: new Map<string, CachedAnalysis>(),
    };
    source.catch(() => {
      if (this.cache === cached) this.cache = null;
    });
    this.cache = cached;
    return cached;
  }

  private scaledSegments(cached: CachedWindow, intervalMs: number): Promise<Candle[][]> {
    const existing = cached.scales.get(intervalMs);
    if (existing) return existing;
    const pending = cached.source.then((candles) =>
      continuousSegments(aggregateCandles(candles, cached.sourceIntervalMs, intervalMs), intervalMs));
    cached.scales.set(intervalMs, pending);
    return pending;
  }

  private oracle(
    cached: CachedWindow,
    request: VwKamaInspectorRequest,
    candles: Candle[],
  ): Promise<PerfectMarginOracleResult> {
    const key = `${request.intervalMs}:${candles[0]!.openTime}:${request.oracleFriction}`;
    const existing = cached.oracles.get(key);
    if (existing) return existing;
    const pending = Promise.resolve(perfectMarginOracle(candles, {
      startingQuote: 1,
      leverage: 1,
      friction: request.oracleFriction,
      eventMode: "close",
      maxPathCandles: MAX_CHART_CANDLES,
    }));
    cached.oracles.set(key, pending);
    while (cached.oracles.size > 4) {
      const oldest = cached.oracles.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      cached.oracles.delete(oldest);
    }
    return pending;
  }

  private async exposureOracle(
    cached: CachedWindow,
    request: VwKamaInspectorRequest,
    candles: Candle[],
    scoreStartIndex: number,
    scoreEndIndex: number,
    oracleStateCodes: Uint8Array,
    scoreIndexes?: readonly number[],
    cancelFlag?: Int32Array,
  ): Promise<ExposureValueOracle> {
    const config = request.valueDistillation!;
    const holdingPeriodSteps = resolveVwKamaHoldingPeriodSteps(
      candles.slice(0, scoreEndIndex),
      scoreStartIndex,
      oracleStateCodes,
      request.intervalMs,
      config,
    );
    const valueHorizonSteps = config.valueHorizonMode === "fixed"
      ? Math.max(1, Math.round(config.valueHorizonMs / request.intervalMs))
      : Math.max(1, scoreEndIndex - scoreStartIndex - 1);
    if (valueHorizonSteps < holdingPeriodSteps) {
      throw new Error("Resolved holding period exceeds the configured value horizon.");
    }
    if (config.valueHorizonMode === "fixed" && config.horizonEndMode === "extend"
      && candles.length - scoreEndIndex < valueHorizonSteps) {
      throw new Error("Value horizon requires more continuous post-window candles.");
    }
    const valueCandles = config.valueHorizonMode === "fixed" && config.horizonEndMode === "extend"
      ? candles
      : candles.slice(0, scoreEndIndex);
    const prices = valueCandles.map((candle) => candle.close);
    const sparse = scoreIndexes !== undefined
      && valueHorizonSteps < prices.length - 1 - scoreStartIndex;
    const options = {
      scoreStartIndex,
      holdingPeriodSteps,
      valueHorizonSteps,
      friction: request.oracleFriction,
      gridSize: config.gridSize,
      minExposure: config.minExposure,
      maxExposure: config.maxExposure,
      maxEffectiveExposure: config.maxEffectiveExposure,
      initialExposure: config.initialExposure,
      terminalIndex: scoreEndIndex - 1,
      temperature: config.oracleTemperature,
      // Opportunity epsilon only offsets per-row weights. Keep the expensive
      // oracle neutral so epsilon edits can reuse its exact values/statistics.
      opportunityEpsilon: 0,
      quoteLendRate: hourlyRatePerCandle(config.quoteLendRate, request.intervalMs),
      quoteBorrowRate: hourlyRatePerCandle(config.quoteBorrowRate, request.intervalMs),
      assetBorrowRate: hourlyRatePerCandle(config.assetBorrowRate, request.intervalMs),
      includeProbabilities: true,
      cancelFlag,
    };
    const key = [
      request.intervalMs,
      candles[0]!.openTime,
      candles.at(-1)!.closeTime,
      scoreStartIndex,
      request.oracleFriction,
      config.gridSize,
      config.minExposure,
      config.maxExposure,
      config.maxEffectiveExposure,
      config.initialExposure,
      config.holdingPeriodMode,
      config.holdingPeriodMs,
      holdingPeriodSteps,
      config.valueHorizonMode,
      config.valueHorizonMs,
      valueHorizonSteps,
      config.horizonEndMode,
      config.oracleTemperature,
      config.quoteLendRate,
      config.quoteBorrowRate,
      config.assetBorrowRate,
      sparse ? "sparse" : "dense",
    ].join(":");
    const existing = cached.valueOracles.get(key);
    if (existing) {
      if (existing.populated && scoreIndexes) {
        const missing = scoreIndexes.filter((index) => !existing.populated!.has(index));
        if (missing.length > 0) {
          const population = existing.population.catch(() => undefined).then(async () => {
            const remaining = missing.filter((index) => !existing.populated!.has(index));
            if (remaining.length === 0) return;
            populateSparseExposureValueOracle(prices, options, await existing.value, remaining);
            for (const index of remaining) existing.populated!.add(index);
          });
          existing.population = population;
          try {
            await population;
          } catch (error) {
            if (existing.population === population) existing.population = Promise.resolve();
            throw error;
          }
        }
      }
      return exposureOracleWithOpportunityEpsilon(
        await existing.value,
        config.opportunityEpsilon,
      );
    }
    const pending = Promise.resolve().then(async () => {
      const oracleCells = (candles.length - scoreStartIndex) * config.gridSize;
      // The CUDA full-window reconstruction launches one residue chain per holding
      // step. Very small H therefore under-utilizes the device; keep those cases on
      // the checkpointed CPU scan.
      if (!sparse && holdingPeriodSteps >= MIN_CUDA_ORACLE_HOLDING_STEPS
        && oracleCells >= 10_000_000 && config.gridSize <= 1_024) {
        const status = await vwKamaCudaStatus();
        if (status.available) {
          try {
            return truncateExposureValueOracle(
              (await prepareExposureValueOracleCuda(prices, options)).oracle,
              scoreEndIndex,
            );
          } catch (error) {
            if (!cudaValueOracleFallbackReported) {
              console.error(
                `Inspector CUDA value-oracle preparation failed; using CPU: ${error instanceof Error
                  ? error.message
                  : String(error)}`,
              );
              cudaValueOracleFallbackReported = true;
            }
          }
        }
      }
      const prepared = sparse
        ? prepareSparseExposureValueOracle(prices, options, scoreIndexes!)
        : prepareExposureValueOracle(prices, options);
      return truncateExposureValueOracle(prepared, scoreEndIndex);
    });
    const entry: CachedExposureOracle = {
      value: pending,
      populated: sparse ? new Set(scoreIndexes) : null,
      population: Promise.resolve(),
    };
    cached.valueOracles.set(key, entry);
    while (cached.valueOracles.size > 2) {
      const oldest = cached.valueOracles.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      cached.valueOracles.delete(oldest);
    }
    try {
      return exposureOracleWithOpportunityEpsilon(await pending, config.opportunityEpsilon);
    } catch (error) {
      if (cached.valueOracles.get(key) === entry) cached.valueOracles.delete(key);
      throw error;
    }
  }

  private async loadSource(selected: VwKamaInspectorWindow, sourceEndTime: number): Promise<Candle[]> {
    const start = selected.startTime - MAX_WARMUP_MS;
    const sourceLabel = duration(selected.sourceIntervalMs);
    const root = path.join(this.dataDir, "historical", "spot-btcusdt", "btcusdt", sourceLabel);
    const candles: Candle[] = [];
    for (let day = utcDay(start); day < sourceEndTime; day += DAY_MS) {
      const date = new Date(day).toISOString().slice(0, 10);
      let content: string;
      try {
        content = await readDailyShard(root, date);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") {
          if (selected.id !== "latest") {
            throw new Error(`Missing BTCUSDT ${sourceLabel} shard ${date}; representative fetch is not complete.`);
          }
          await this.fetchMissingDailyShard(date, day);
          try {
            content = await readDailyShard(root, date);
          } catch (readError) {
            if ((readError as NodeJS.ErrnoException).code === "ENOENT") {
              throw new Error(
                `Failed to fetch missing BTCUSDT ${sourceLabel} shard ${date}: downloader completed without creating the shard.`,
              );
            }
            throw readError;
          }
        } else {
          throw error;
        }
      }
      for (const line of content.split("\n")) {
        if (!line) continue;
        const candle = JSON.parse(line) as Candle;
        if (candle.openTime >= start && candle.openTime < sourceEndTime) candles.push(candle);
      }
    }
    candles.sort((left, right) => left.openTime - right.openTime);
    if (candles.length === 0) throw new Error(`No BTCUSDT 1s candles for ${selected.label}.`);
    return candles;
  }

  private fetchMissingDailyShard(date: string, day: number): Promise<void> {
    const existing = this.pendingShardFetches.get(date);
    if (existing) return existing;
    const pending = this.fetchDailyShard({ dataDir: this.dataDir, date, day }).catch((error) => {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to fetch missing BTCUSDT 1s shard ${date}: ${reason}`);
    }).finally(() => this.pendingShardFetches.delete(date));
    this.pendingShardFetches.set(date, pending);
    return pending;
  }
}

function inspectorCatalog(dataDir: string, now = Date.now()): VwKamaInspectorCatalog {
  return {
    windows: [latestInspectorWindow(DEFAULT_REQUEST.latestDays, now),
      ...STATIC_WINDOWS.map((item) => ({ ...item }))],
    scales: SCALES.map((item) => ({ ...item })),
    defaults: structuredClone(DEFAULT_REQUEST),
    presets: [
      ...GLOBAL_PRESETS.map((item) => structuredClone(item)),
      ...loadGlobalPresets(dataDir),
      ...loadWindowPresets(dataDir),
    ],
    predictorPresets: [
      ...HANDCRAFTED_PREDICTOR_PRESETS,
      ...DIRECT_INDICATOR_PREDICTOR_PRESETS,
    ].map((preset) => structuredClone(preset)),
  };
}

function loadGlobalPresets(dataDir: string): VwKamaPreset[] {
  return loadPresets(dataDir, "vw-kama-global-presets.json").filter((preset) =>
    preset.scope === "global" && preset.parameters).map((preset) => ({
      ...preset,
      historicalScore: preset.scoreVersion === VW_KAMA_SCORE_VERSION
        ? undefined
        : preset.score,
      score: preset.scoreVersion === VW_KAMA_SCORE_VERSION ? preset.score : undefined,
    }));
}

function loadWindowPresets(dataDir: string): VwKamaPreset[] {
  return loadPresets(dataDir, "vw-kama-window-presets.json").flatMap((preset) => {
    const window = STATIC_WINDOWS.find((item) => item.id === preset.windowId);
    const scale = preset.intervalMs == null
      ? null
      : SCALES.find((item) => item.intervalMs === preset.intervalMs);
    return preset.scope === "window" && window && preset.parameters && (preset.intervalMs == null || scale)
      ? [{
          ...preset,
          historicalScore: preset.scoreVersion === VW_KAMA_SCORE_VERSION
            ? undefined
            : preset.score,
          score: preset.scoreVersion === VW_KAMA_SCORE_VERSION ? preset.score : undefined,
          label: `Window best found · ${window.label}${scale ? ` · ${scale.label}` : ""}`,
        }]
      : [];
  });
}

function loadPresets(dataDir: string, name: string): VwKamaPreset[] {
  const files = [path.join(dataDir, "benchmarks", name), path.join(REPO_ROOT, "data", "benchmarks", name)];
  for (const file of new Set(files)) try {
    const parsed = JSON.parse(readFileSync(file, "utf8"));
    return Array.isArray(parsed) ? parsed as VwKamaPreset[] : [];
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
    throw error;
  }
  return [];
}

function combineEvaluations(
  evaluated: Array<{ result: VwKamaEvaluation; candles: Candle[]; scoreStart: number }>,
  request: VwKamaInspectorRequest,
): Omit<
  VwKamaInspectorResponse,
  "window" | "intervalMs" | "sourceSegmentCount" | "scoredSegmentCount" | "elapsedMs"
> {
  const results = evaluated.map((item) => item.result);
  const candidateTransitions = results.flatMap((item) => item.candidateTransitions);
  const oracleTransitions = results.flatMap((item) => item.oracleTransitions);
  const candleCount = results.reduce((sum, item) => sum + item.candleCount, 0);
  const metrics = combineMetrics(results, candidateTransitions, oracleTransitions, candleCount, request.intervalMs);
  const scored = evaluated.flatMap((item) =>
    item.candles.filter((candle) => candle.openTime >= item.scoreStart));
  const statePoints = results.flatMap((item) => item.statePoints);
  const indicatorPoints = results.flatMap((item) => item.indicatorPoints);
  const valueDistributions = results.flatMap((item) => item.valueDistributions);
  const valueOraclePath = results.flatMap((item) => item.valueOraclePath);
  const valueCandidatePath = results.flatMap((item) => item.valueCandidatePath);
  const rendered = renderCandleRange(
    continuousSegments(scored, request.intervalMs),
    request.intervalMs,
    scored[0]!.openTime,
    scored.at(-1)!.closeTime + 1,
    MAX_CHART_CANDLES,
  );
  return {
    candleCount,
    metrics,
    renderIntervalMs: rendered.intervalMs,
    candles: rendered.candles,
    kamaSeries: {
      index: -1,
      windowSec: request.parameters.slowMs / 1_000,
      label: "VW-KAMA",
      color: "#f472b6",
      points: results.flatMap((item) => item.kamaSeries.points),
    },
    annotations: sampleEven(results.flatMap((item) => item.annotations), 5_000),
    oracle: {
      mode: "fixed-notional",
      eventMode: "close",
      leverage: 1,
      friction: request.oracleFriction,
      points: results.flatMap((item) => item.oracle.points),
    },
    candidatePath: { points: candidatePath(evaluated) },
    statePoints,
    indicatorPoints,
    valueDistributions,
    valueOraclePath,
    valueCandidatePath,
    candidateTransitions,
    oracleTransitions,
  };
}

function candidatePath(
  evaluated: Array<{ result: VwKamaEvaluation; scoreStart: number }>,
): VwKamaInspectorResponse["candidatePath"]["points"] {
  return evaluated.flatMap(({ result, scoreStart }) => {
    const first = result.statePoints[0];
    const firstTransition = result.candidateTransitions[0];
    const initialState = firstTransition?.time === first?.time
      ? firstTransition.fromState
      : first?.candidate;
    const initial: BacktestOraclePoint[] = initialState ? [{
      time: scoreStart,
      price: 0,
      fromState: initialState,
      state: initialState,
      action: "hold" as const,
    }] : [];
    return [...initial, ...result.candidateTransitions.map((point): BacktestOraclePoint => ({
      time: point.time,
      price: point.price,
      fromState: point.fromState,
      state: point.state,
      action: point.fromState === "flat" ? "open" as const
        : point.state === "flat" ? "close" as const
        : "switch" as const,
    }))];
  }).sort((left, right) => left.time - right.time);
}

interface HistoricalOracleAverage {
  regretProbabilities: Float64Array;
  maximumReturnGapProbabilities: Float64Array;
  sampleCount: number;
  averageRegret: number;
  averageMaximumReturn: number;
}

function historicalOracleAverage(
  oracle: ExposureValueOracle,
  indexes: readonly number[],
): HistoricalOracleAverage {
  if (!oracle.actionValues) throw new Error("Historical oracle action values were not retained.");
  const targetSize = oracle.grid.length;
  const stateSize = oracle.currentGrid.length;
  const cellCount = targetSize * stateSize;
  const regretSums = new Float64Array(cellCount);
  const regretCounts = new Uint32Array(cellCount);
  const returnSums = new Float64Array(cellCount);
  const returnCounts = new Uint32Array(cellCount);
  const conditionalValues = new Float64Array(targetSize);
  let maximumReturnSum = 0;
  let maximumReturnCount = 0;
  for (const candleIndex of indexes) {
    const actionOffset = candleIndex * targetSize;
    let maximumActionValue = Number.NEGATIVE_INFINITY;
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      maximumActionValue = Math.max(
        maximumActionValue,
        oracle.actionValues[actionOffset + targetIndex]!,
      );
    }
    if (Number.isFinite(maximumActionValue)) {
      const maximumReturn = Math.expm1(maximumActionValue);
      // Cash is an executable post-action baseline, so the oracle maximum cannot lose money.
      if (maximumReturn < -1e-10) {
        throw new Error("Historical oracle maximum violated its cash-return baseline.");
      }
      maximumReturnSum += Math.max(0, maximumReturn);
      maximumReturnCount += 1;
    }
    for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
      let maximum = Number.NEGATIVE_INFINITY;
      for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
        const actionValue = oracle.actionValues[actionOffset + targetIndex]!;
        const factor = rebalanceEquityFactor(
          oracle.currentGrid[stateIndex]!,
          oracle.grid[targetIndex]!,
          oracle.execution.friction,
        );
        const conditionalValue = Number.isFinite(actionValue) && factor > 0
          ? actionValue + Math.log(factor)
          : Number.NEGATIVE_INFINITY;
        conditionalValues[targetIndex] = conditionalValue;
        maximum = Math.max(maximum, conditionalValue);
      }
      if (!Number.isFinite(maximum)) continue;
      const cellOffset = stateIndex * targetSize;
      for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
        const conditionalValue = conditionalValues[targetIndex]!;
        const cell = cellOffset + targetIndex;
        if (Number.isFinite(conditionalValue)) {
          regretSums[cell] += Math.max(0, maximum - conditionalValue);
          regretCounts[cell] += 1;
          returnSums[cell] += Math.expm1(conditionalValue);
        } else {
          returnSums[cell] -= 1;
        }
        returnCounts[cell] += 1;
      }
    }
  }
  const regretCosts = Float64Array.from(regretSums, (sum, index) =>
    regretCounts[index]! > 0 ? sum / regretCounts[index]! : Number.POSITIVE_INFINITY);
  const returnGapCosts = new Float64Array(cellCount);
  returnGapCosts.fill(Number.POSITIVE_INFINITY);
  const averageMaximumReturn = maximumReturnCount > 0
    ? maximumReturnSum / maximumReturnCount
    : 0;
  for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
    const cellOffset = stateIndex * targetSize;
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      const cell = cellOffset + targetIndex;
      if (returnCounts[cell]! === 0) continue;
      const averageReturn = returnSums[cell]! / returnCounts[cell]!;
      returnGapCosts[cell] = Math.max(0, averageMaximumReturn - averageReturn);
    }
  }
  let averageRegret = 0;
  let averageRegretCount = 0;
  for (let index = 0; index < regretCosts.length; index += 1) {
    if (!Number.isFinite(regretCosts[index])) continue;
    averageRegret += regretCosts[index]!;
    averageRegretCount += 1;
  }
  return {
    regretProbabilities: normalizedHistoricalCosts(
      regretCosts,
      stateSize,
      targetSize,
      oracle.temperature,
    ),
    maximumReturnGapProbabilities: normalizedHistoricalCosts(
      returnGapCosts,
      stateSize,
      targetSize,
      oracle.temperature,
    ),
    sampleCount: indexes.length,
    averageRegret: averageRegretCount > 0 ? averageRegret / averageRegretCount : 0,
    averageMaximumReturn,
  };
}

function normalizedHistoricalCosts(
  costs: Float64Array,
  stateSize: number,
  targetSize: number,
  temperature: number,
): Float64Array {
  const result = new Float64Array(costs.length);
  for (let stateIndex = 0; stateIndex < stateSize; stateIndex += 1) {
    const rowOffset = stateIndex * targetSize;
    let minimum = Number.POSITIVE_INFINITY;
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      minimum = Math.min(minimum, costs[rowOffset + targetIndex]!);
    }
    if (!Number.isFinite(minimum)) continue;
    let total = 0;
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      const cost = costs[rowOffset + targetIndex]!;
      const weight = Number.isFinite(cost)
        ? Math.exp(Math.max(-50, -(cost - minimum) / temperature))
        : 0;
      result[rowOffset + targetIndex] = weight;
      total += weight;
    }
    for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
      result[rowOffset + targetIndex] /= total;
    }
  }
  return result;
}

function sampleEven<T>(values: T[], limit: number): T[] {
  if (values.length <= limit) return values;
  const result: T[] = [];
  const step = (values.length - 1) / (limit - 1);
  for (let index = 0; index < limit; index += 1) {
    result.push(values[Math.round(index * step)]!);
  }
  return result;
}

function combineMetrics(
  results: VwKamaEvaluation[],
  candidates: VwKamaTransition[],
  oracle: VwKamaTransition[],
  candleCount: number,
  intervalMs: number,
): VwKamaAccuracyMetrics {
  const signalCount = candidates.length;
  const oracleCount = oracle.length;
  const matchedCount = results.reduce((sum, item) => sum + item.metrics.matchedCount, 0);
  const extraSignalCount = signalCount - matchedCount;
  const signalCleanliness = signalCount > 0 ? matchedCount / signalCount : 1;
  const timingCredit = candidates.reduce((sum, item) => sum + item.timingCredit, 0);
  const precision = eventRatio(timingCredit, signalCount, oracleCount);
  const recall = eventRatio(timingCredit, oracleCount, signalCount);
  const f1 = harmonic(precision, recall);
  const exposureAgreement = results.reduce(
    (sum, item) => sum + item.metrics.exposureAgreement * item.candleCount,
    0,
  ) / candleCount;
  const lags = candidates.flatMap((item) => item.lagMs === null ? [] : [item.lagMs]);
  const absoluteLags = lags.map(Math.abs);
  const valueParts = results.flatMap((item) => item.metrics.valueDistillation ?? []);
  const distillationWeight = valueParts.reduce((sum, item) => sum + item.weightSum, 0);
  const weightedCrossEntropy = valueParts.reduce(
    (sum, item) => sum + item.weightedCrossEntropy,
    0,
  );
  const weightedOracleEntropy = valueParts.reduce(
    (sum, item) => sum + item.weightedOracleEntropy,
    0,
  );
  const weightedStrategyEntropy = valueParts.reduce(
    (sum, item) => sum + item.weightedStrategyEntropy,
    0,
  );
  const weightedEntropyGap = valueParts.reduce(
    (sum, item) => sum + item.weightedEntropyGap,
    0,
  );
  const crossEntropy = distillationWeight > 0 ? weightedCrossEntropy / distillationWeight : 0;
  const oracleEntropy = distillationWeight > 0 ? weightedOracleEntropy / distillationWeight : 0;
  const strategyEntropy = distillationWeight > 0 ? weightedStrategyEntropy / distillationWeight : 0;
  const entropyGap = distillationWeight > 0 ? weightedEntropyGap / distillationWeight : 0;
  const stateMutualInformation = distillationWeight > 0 ? valueParts.reduce(
    (sum, item) => sum + item.stateMutualInformation * item.weightSum,
    0,
  ) / distillationWeight : 0;
  const oracleMutualInformation = distillationWeight > 0 ? valueParts.reduce(
    (sum, item) => sum + item.oracleMutualInformation * item.weightSum,
    0,
  ) / distillationWeight : 0;
  const mixedLoss = distillationWeight > 0 ? valueParts.reduce(
    (sum, item) => sum + item.mixedLoss * item.weightSum,
    0,
  ) / distillationWeight : 0;
  return {
    score: vwKamaScore(f1, exposureAgreement, signalCleanliness),
    precision,
    recall,
    f1,
    rawPrecision: eventRatio(matchedCount, signalCount, oracleCount),
    rawRecall: eventRatio(matchedCount, oracleCount, signalCount),
    exposureAgreement,
    noiseSignalRatio: noiseSignalRatio(extraSignalCount, matchedCount),
    signalCleanliness,
    signalsPerDay: signalCount / Math.max(intervalMs / DAY_MS, candleCount * intervalMs / DAY_MS),
    signalCount,
    oracleCount,
    matchedCount,
    extraSignalCount,
    missedOracleCount: oracleCount - matchedCount,
    lagP50Ms: percentile(absoluteLags, 0.5),
    lagP90Ms: percentile(absoluteLags, 0.9),
    lagP95Ms: percentile(absoluteLags, 0.95),
    lagMedianSignedMs: percentile(lags, 0.5),
    ...(valueParts.length > 0 ? {
      valueDistillation: {
        holdingPeriodMs: valueParts.reduce(
          (sum, item) => sum + item.holdingPeriodMs * item.sampleCount,
          0,
        ) / Math.max(1, valueParts.reduce((sum, item) => sum + item.sampleCount, 0)),
        valueHorizonMs: valueParts.reduce(
          (sum, item) => sum + item.valueHorizonMs * item.sampleCount,
          0,
        ) / Math.max(1, valueParts.reduce((sum, item) => sum + item.sampleCount, 0)),
        weightedCrossEntropy,
        weightedOracleEntropy,
        weightedStrategyEntropy,
        weightedEntropyGap,
        weightSum: distillationWeight,
        opportunitySum: valueParts.reduce((sum, item) => sum + item.opportunitySum, 0),
        averageRegretSum: valueParts.reduce((sum, item) => sum + item.averageRegretSum, 0),
        sampleCount: valueParts.reduce((sum, item) => sum + item.sampleCount, 0),
        crossEntropy,
        oracleEntropy,
        strategyEntropy,
        entropyGap,
        stateMutualInformation,
        oracleMutualInformation,
        oracleMutualInformationMode: valueParts[0]!.oracleMutualInformationMode,
        mixedLoss,
        mixedScore: Math.exp(-Math.max(0, mixedLoss - oracleEntropy)),
        klDivergence: Math.max(0, crossEntropy - oracleEntropy),
        score: Math.exp(-Math.max(0, crossEntropy - oracleEntropy)),
        meanOpportunity: valueParts.reduce((sum, item) => sum + item.opportunitySum, 0)
          / Math.max(1, valueParts.reduce((sum, item) => sum + item.sampleCount, 0)),
        meanAverageRegret: valueParts.reduce((sum, item) => sum + item.averageRegretSum, 0)
          / Math.max(1, valueParts.reduce((sum, item) => sum + item.sampleCount, 0)),
        returns: {
          oracle: combineExposureReturns(valueParts.map((item) => item.returns.oracle)),
          strategy: combineExposureReturns(valueParts.map((item) => item.returns.strategy)),
        },
      },
    } : {}),
  };
}

function combineExposureReturns(parts: ExposureReturnMetrics[]): ExposureReturnMetrics {
  const logReturn = parts.reduce((sum, item) => sum + item.logReturn, 0);
  const equity = Math.exp(logReturn);
  return {
    equity,
    exposure: parts.at(-1)?.exposure ?? 0,
    peakEquity: Math.max(1, ...parts.map((item) => item.peakEquity)),
    maxDrawdown: Math.max(0, ...parts.map((item) => item.maxDrawdown)),
    turnover: parts.reduce((sum, item) => sum + item.turnover, 0),
    rebalanceCount: parts.reduce((sum, item) => sum + item.rebalanceCount, 0),
    liquidationCount: parts.reduce((sum, item) => sum + item.liquidationCount, 0),
    sampleCount: parts.reduce((sum, item) => sum + item.sampleCount, 0),
    totalReturn: equity - 1,
    logReturn,
  };
}

function hourlyRatePerCandle(hourlyRate: number, intervalMs: number): number {
  return Math.expm1(Math.log1p(hourlyRate) * intervalMs / 3_600_000);
}

function handcraftedPredictorOptions(
  request: VwKamaInspectorRequest,
  candles: Candle[],
): {
  handcraftedPredictor?: {
    parameters: HandcraftedIndicatorPredictorParameters;
    states: ReturnType<typeof prepareHandcraftedIndicatorStates>;
  };
  directIndicatorPredictor?: {
    parameters: DirectIndicatorPredictorParameters;
    states: ReturnType<typeof prepareHandcraftedIndicatorStates>;
  };
} {
  const model = request.predictor?.model;
  if (model !== "handcrafted" && model !== "direct-indicator") return {};
  const predictor = request.predictor!;
  const prices = candles.map((candle) => candle.close);
  if (model === "handcrafted") {
    const parameters = predictor.handcraftedParameters;
    return {
      handcraftedPredictor: {
        parameters,
        states: prepareHandcraftedIndicatorStates(prices, request.intervalMs, parameters),
      },
    };
  }
  const parameters = predictor.directIndicatorParameters;
  return {
    directIndicatorPredictor: {
      parameters,
      states: prepareHandcraftedIndicatorStates(prices, request.intervalMs, parameters),
    },
  };
}

interface PerCandlePredictorFit {
  parameters: HandcraftedIndicatorPredictorParameters;
  baselineCrossEntropy: number;
  crossEntropy: number;
  iterations: number;
}

function fitHandcraftedPredictorAtCandle(
  prices: ArrayLike<number>,
  candleIndex: number,
  intervalMs: number,
  oracle: ExposureValueOracle,
  startingParameters: HandcraftedIndicatorPredictorParameters,
  seed: number,
  cancelFlag?: Int32Array,
): PerCandlePredictorFit {
  const score = (parameters: HandcraftedIndicatorPredictorParameters): number => {
    throwIfInspectorCancelled(cancelFlag);
    const state = prepareHandcraftedIndicatorStateAt(prices, candleIndex, intervalMs, parameters);
    const prediction = predictHandcraftedIndicatorRegret(
      oracle.grid,
      state,
      parameters,
      {
        intervalMs,
        holdingPeriodSteps: oracle.holdingPeriodSteps,
        valueHorizonSteps: oracle.valueHorizonSteps,
        temperature: oracle.temperature,
        execution: oracle.execution,
      },
    );
    return exposureProbabilityTransitionCrossEntropy(
      oracle,
      candleIndex,
      prediction.probabilities,
      1 / oracle.temperature,
    );
  };
  const baselineCrossEntropy = score(startingParameters);
  let best = { parameters: { ...startingParameters }, crossEntropy: baselineCrossEntropy };
  let iterations = 1;
  const random = mulberry32((Math.floor(seed) ^ 0x51f17e5d) >>> 0);
  const keys = Object.keys(startingParameters) as Array<keyof HandcraftedIndicatorPredictorParameters>;
  const candidates: HandcraftedIndicatorPredictorParameters[] = [
    { ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS },
  ];
  while (candidates.length < 48) {
    const candidate = {} as HandcraftedIndicatorPredictorParameters;
    for (const key of keys) {
      const [minimum, maximum] = HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS[key];
      candidate[key] = Math.exp(Math.log(minimum) + random() * (Math.log(maximum) - Math.log(minimum)));
    }
    candidates.push(candidate);
  }
  for (const parameters of candidates) {
    const crossEntropy = score(parameters);
    iterations += 1;
    if (crossEntropy < best.crossEntropy) best = { parameters, crossEntropy };
  }
  for (let round = 0; round < 4; round += 1) {
    const multiplier = 1 + 0.75 / (round + 1);
    const center = best.parameters;
    for (const key of keys) for (const scale of [1 / multiplier, multiplier]) {
      const [minimum, maximum] = HANDCRAFTED_INDICATOR_PARAMETER_BOUNDS[key];
      const parameters = {
        ...center,
        [key]: clamp(center[key] * scale, minimum, maximum),
      };
      const crossEntropy = score(parameters);
      iterations += 1;
      if (crossEntropy < best.crossEntropy) best = { parameters, crossEntropy };
    }
  }
  return {
    parameters: best.parameters,
    baselineCrossEntropy,
    crossEntropy: best.crossEntropy,
    iterations,
  };
}

function throwIfInspectorCancelled(flag: Int32Array | undefined): void {
  if (flag && Atomics.load(flag, 0) !== 0) throw new Error("VW-KAMA request cancelled.");
}

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = seed + 0x6D2B79F5 | 0;
    let value = Math.imul(seed ^ seed >>> 15, 1 | seed);
    value = value + Math.imul(value ^ value >>> 7, 61 | value) ^ value;
    return ((value ^ value >>> 14) >>> 0) / 4_294_967_296;
  };
}

function normalizeRequest(input: VwKamaInspectorRequest): VwKamaInspectorRequest {
  if (!SCALES.some((item) => item.intervalMs === input.intervalMs)) {
    throw new Error(`Unsupported VW-KAMA scale: ${input.intervalMs}`);
  }
  const request = structuredClone(input);
  request.latestDays = normalizeLatestDays(request.latestDays);
  request.predictor = {
    model: request.predictor?.model ?? "handcrafted",
    handcraftedParameters: {
      ...DEFAULT_HANDCRAFTED_INDICATOR_PARAMETERS,
      ...request.predictor?.handcraftedParameters,
    },
    directIndicatorParameters: {
      ...DEFAULT_DIRECT_INDICATOR_PARAMETERS,
      ...request.predictor?.directIndicatorParameters,
    },
  };
  if (request.predictor.model !== "legacy"
    && request.predictor.model !== "handcrafted"
    && request.predictor.model !== "direct-indicator") {
    throw new Error("VW-KAMA predictor model must be legacy, handcrafted, or direct-indicator.");
  }
  request.valueDistillation = {
    ...DEFAULT_REQUEST.valueDistillation!,
    ...request.valueDistillation,
  };
  Object.assign(
    request.valueDistillation,
    normalizeExposureValueDistillationLossConfig(
      request.valueDistillation,
      request.valueDistillation.gridSize,
    ),
  );
  if (!["flat", "hold", "hysteresis"].includes(request.parameters.deadbandMode)) {
    throw new Error("VW-KAMA deadband mode must be flat, hold, or hysteresis.");
  }
  if (request.parameters.rateMode !== undefined
    && request.parameters.rateMode !== "relative"
    && request.parameters.rateMode !== "log") {
    throw new Error("VW-KAMA rate mode must be relative or log.");
  }
  if (request.parameters.thresholdNoiseResponse !== undefined
    && request.parameters.thresholdNoiseResponse !== "proportional"
    && request.parameters.thresholdNoiseResponse !== "inverse") {
    throw new Error("VW-KAMA threshold noise response must be proportional or inverse.");
  }
  if (request.parameters.agreementMode !== undefined
    && request.parameters.agreementMode !== "sizing"
    && request.parameters.agreementMode !== "confidence") {
    throw new Error("VW-KAMA agreement mode must be sizing or confidence.");
  }
  request.parameters.agreementMode ??= "sizing";
  request.parameters.efficiencyVolumeEmaMs ??= request.parameters.volumeMs;
  request.parameters.efficiencyVolumePower ??= 0;
  request.parameters.rateMode ??= "relative";
  request.parameters.rateEmaMs ??= request.intervalMs;
  request.parameters.thresholdLookbackMs ??= 60 * 60_000;
  request.parameters.thresholdNoiseResponse ??= "proportional";
  request.parameters.thresholdNoiseMultiplier ??= 0;
  request.parameters.thresholdInverseMaxBpsHour ??= 0;
  request.parameters.thresholdInverseNoiseScaleBpsHour ??= 30;
  request.parameters.signalFrictionFraction ??= 1;
  request.parameters.strategyTemperature ??= 0.001;
  request.parameters.strategyQuadraticScale ??= 0;
  request.parameters.strategyQuadraticVolatilityMs ??= 60 * 60_000;
  request.parameters.strategyNormalMixture ??= 0;
  request.parameters.strategyNormalSigma ??= 25;
  request.parameters.buyMaxFraction ??= 1;
  request.parameters.sellMaxFraction ??= 1;
  request.parameters.buySizingSigmaBpsHour ??= 1e12;
  request.parameters.sellSizingSigmaBpsHour ??= 1e12;
  request.parameters.hysteresisReleaseRatio ??= 0.25;
  request.parameters.confirmationMix ??= 0;
  request.parameters.confirmationMinQuality ??= 0;
  request.parameters.confirmationAccelerationLookbackMs ??= 60 * 60_000;
  request.parameters.confirmationDistanceLookbackMs ??= 60 * 60_000;
  request.parameters.confirmationAccelerationWeight ??= 1;
  request.parameters.confirmationDistanceWeight ??= 1;
  request.parameters.confirmationEmaMs ??= 60 * 60_000;
  request.parameters.confirmationEmaThresholdBpsHour ??= 0;
  request.parameters.confirmationEmaWeight ??= 0;
  request.parameters.confirmationEmaGateStrength ??= 0;
  request.parameters.confirmationRsiMs ??= 14 * 60_000;
  request.parameters.confirmationRsiThreshold ??= 0;
  request.parameters.confirmationRsiWeight ??= 0;
  request.parameters.confirmationDmiMs ??= 14 * 60_000;
  request.parameters.confirmationDmiWeight ??= 0;
  request.parameters.confirmationAdxThreshold ??= 20;
  request.parameters.confirmationBias ??= 0;
  request.parameters.meanReversionEfficiencyMs ??= request.parameters.efficiencyMs;
  request.parameters.meanReversionFastMs ??= request.parameters.fastMs;
  request.parameters.meanReversionSlowMs ??= 60 * 60_000;
  request.parameters.meanReversionVolatilityMs ??= 60 * 60_000;
  request.parameters.meanReversionSuppressionThreshold ??= 1;
  request.parameters.meanReversionReversalThreshold ??= 0;
  const values = [
    request.parameters.efficiencyMs,
    ...(request.parameters.efficiencyVolumePower > 0
      ? [request.parameters.efficiencyVolumeEmaMs]
      : []),
    request.parameters.fastMs,
    request.parameters.slowMs,
    request.parameters.power,
    request.parameters.volumeMs,
    request.parameters.volumeCap,
    request.parameters.rateEmaMs,
    ...(thresholdNoiseEnabled(request.parameters)
      ? [request.parameters.thresholdLookbackMs]
      : []),
    ...(request.parameters.confirmationMix > 0
      ? [
        request.parameters.confirmationAccelerationLookbackMs,
        request.parameters.confirmationDistanceLookbackMs,
      ]
      : []),
    ...((request.parameters.confirmationEmaWeight > 0
      || request.parameters.confirmationEmaGateStrength > 0)
      ? [request.parameters.confirmationEmaMs]
      : []),
    ...(request.parameters.confirmationRsiWeight > 0
      ? [request.parameters.confirmationRsiMs]
      : []),
    ...(request.parameters.confirmationDmiWeight > 0
      ? [request.parameters.confirmationDmiMs]
      : []),
    ...(meanReversionEnabled(request.parameters)
      ? [
        request.parameters.meanReversionEfficiencyMs,
        request.parameters.meanReversionFastMs,
        request.parameters.meanReversionSlowMs,
        request.parameters.meanReversionVolatilityMs,
      ]
      : []),
    request.timingHalfLifeMs,
    request.warmupMultiple,
  ];
  if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("VW-KAMA durations and powers must be positive.");
  }
  const nonNegative = [
    request.parameters.volumePower,
    request.parameters.efficiencyVolumePower,
    request.parameters.deadbandBpsHour,
    request.parameters.thresholdNoiseMultiplier,
    request.parameters.thresholdInverseMaxBpsHour,
    request.parameters.signalFrictionFraction,
    request.parameters.strategyQuadraticScale,
    request.parameters.strategyNormalMixture,
    request.parameters.buyMaxFraction,
    request.parameters.sellMaxFraction,
    request.parameters.confirmationAccelerationWeight,
    request.parameters.confirmationDistanceWeight,
    request.parameters.confirmationEmaThresholdBpsHour,
    request.parameters.confirmationEmaWeight,
    request.parameters.confirmationRsiThreshold,
    request.parameters.confirmationRsiWeight,
    request.parameters.confirmationDmiWeight,
    request.parameters.confirmationAdxThreshold,
    request.parameters.meanReversionSuppressionThreshold,
    request.parameters.meanReversionReversalThreshold,
    request.oracleFriction,
    request.matchWindowMs,
  ];
  if (nonNegative.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("VW-KAMA thresholds and friction cannot be negative.");
  }
  const valueConfig = request.valueDistillation;
  if (!Number.isInteger(valueConfig.gridSize) || valueConfig.gridSize < 3 || valueConfig.gridSize > 1_024) {
    throw new Error("VW-KAMA value grid size must be an integer from 3 to 1,024.");
  }
  if (!Number.isFinite(valueConfig.minExposure) || valueConfig.minExposure >= 0
    || !Number.isFinite(valueConfig.maxExposure) || valueConfig.maxExposure <= 0) {
    throw new Error("VW-KAMA value exposure bounds must contain zero.");
  }
  if (!Number.isFinite(valueConfig.maxEffectiveExposure)
    || valueConfig.maxEffectiveExposure < Math.max(
      Math.abs(valueConfig.minExposure),
      Math.abs(valueConfig.maxExposure),
    )) {
    throw new Error("VW-KAMA effective exposure must cover the tradable exposure range.");
  }
  const zeroGridPosition = -valueConfig.minExposure
    / (valueConfig.maxExposure - valueConfig.minExposure) * (valueConfig.gridSize - 1);
  if (Math.abs(zeroGridPosition - Math.round(zeroGridPosition)) > 1e-9) {
    throw new Error("VW-KAMA value grid must contain exact zero exposure.");
  }
  if (!Number.isFinite(valueConfig.initialExposure)
    || Math.abs(valueConfig.initialExposure) > valueConfig.maxEffectiveExposure) {
    throw new Error("VW-KAMA initial exposure exceeds the effective-exposure limit.");
  }
  if (!["fixed", "oracle-half-average-trade"].includes(valueConfig.holdingPeriodMode)
    || !Number.isFinite(valueConfig.holdingPeriodMs) || valueConfig.holdingPeriodMs <= 0
    || !["full-window", "fixed"].includes(valueConfig.valueHorizonMode ?? "full-window")
    || !Number.isFinite(valueConfig.valueHorizonMs)
    || (valueConfig.valueHorizonMode === "fixed"
      && valueConfig.valueHorizonMs < valueConfig.holdingPeriodMs)
    || !["truncate", "extend"].includes(valueConfig.horizonEndMode)
    || !Number.isFinite(valueConfig.oracleTemperature) || valueConfig.oracleTemperature <= 0
    || typeof valueConfig.strategyVolatilityScaling !== "boolean") {
    throw new Error("VW-KAMA value and strategy calibration settings are invalid.");
  }
  if ([
    valueConfig.opportunityEpsilon,
    valueConfig.quoteLendRate,
    valueConfig.quoteBorrowRate,
    valueConfig.assetBorrowRate,
  ].some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("VW-KAMA value costs and rates must be finite and non-negative.");
  }
  if (request.oracleFriction >= 1) throw new Error("VW-KAMA oracle friction must be less than one.");
  if (request.parameters.thresholdNoiseResponse === "inverse"
    && request.parameters.thresholdInverseMaxBpsHour > 0
    && (!Number.isFinite(request.parameters.thresholdInverseNoiseScaleBpsHour)
      || request.parameters.thresholdInverseNoiseScaleBpsHour <= 0)) {
    throw new Error("VW-KAMA inverse threshold noise scale must be positive.");
  }
  if (request.parameters.buyMaxFraction > 1 || request.parameters.sellMaxFraction > 1) {
    throw new Error("VW-KAMA sizing fractions cannot exceed one.");
  }
  if (request.parameters.signalFrictionFraction > 1) {
    throw new Error("VW-KAMA signal friction fraction cannot exceed one.");
  }
  if (!Number.isFinite(request.parameters.strategyTemperature)
    || request.parameters.strategyTemperature <= 0
    || !Number.isFinite(request.parameters.strategyQuadraticVolatilityMs)
    || request.parameters.strategyQuadraticVolatilityMs <= 0
    || !Number.isFinite(request.parameters.strategyNormalSigma)
    || request.parameters.strategyNormalSigma <= 0) {
    throw new Error("VW-KAMA strategy distribution temperature, volatility window, and normal sigma must be positive.");
  }
  if (![
    request.parameters.confirmationMix,
    request.parameters.confirmationMinQuality,
    request.parameters.hysteresisReleaseRatio,
    request.parameters.strategyNormalMixture,
    request.parameters.confirmationEmaGateStrength,
  ]
    .every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
    throw new Error("VW-KAMA confirmation and hysteresis fractions must be between zero and one.");
  }
  if (request.parameters.confirmationRsiThreshold > 50
    || request.parameters.confirmationAdxThreshold > 100) {
    throw new Error("VW-KAMA RSI and ADX thresholds exceed their oscillator ranges.");
  }
  if (!Number.isFinite(request.parameters.confirmationBias)) {
    throw new Error("VW-KAMA confirmation bias must be finite.");
  }
  if (meanReversionEnabled(request.parameters)
    && (request.parameters.meanReversionSuppressionThreshold <= 0
      || request.parameters.meanReversionSuppressionThreshold
        > request.parameters.meanReversionReversalThreshold)) {
    throw new Error("VW-KAMA mean-reversion suppression threshold must be positive and at most the reversal threshold.");
  }
  if (meanReversionEnabled(request.parameters)
    && request.parameters.meanReversionFastMs > request.parameters.meanReversionSlowMs) {
    throw new Error("VW-KAMA mean-reversion fast duration cannot exceed its slow duration.");
  }
  if (![request.parameters.buySizingSigmaBpsHour, request.parameters.sellSizingSigmaBpsHour]
    .every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("VW-KAMA sizing sigmas must be positive.");
  }
  if (candidateWarmupMs(request) > MAX_WARMUP_MS) {
    throw new Error("VW-KAMA warmup exceeds the cached three-day lead-in.");
  }
  return request;
}

function normalizeCandleRangeRequest(input: VwKamaCandleRangeRequest, now: number): {
  request: Omit<VwKamaCandleRangeRequest, "maxCandles"> & { maxCandles: number };
  selected: VwKamaInspectorWindow;
} {
  const analysis = normalizeRequest(input);
  const selected = resolveInspectorWindow(analysis.windowId, analysis.latestDays, now);
  validateWindowScale(selected, analysis.intervalMs);
  const requestedStart = Number(input.startTime);
  const requestedEnd = Number(input.endTime);
  if (!Number.isFinite(requestedStart) || !Number.isFinite(requestedEnd)) {
    throw new Error("VW-KAMA candle range timestamps must be finite.");
  }
  const startTime = clamp(Math.floor(requestedStart), selected.startTime, selected.endTime);
  const endTime = clamp(Math.ceil(requestedEnd), selected.startTime, selected.endTime);
  if (endTime <= startTime) throw new Error("VW-KAMA candle range must overlap the selected window.");
  const requestedLimit = Number(input.maxCandles ?? MAX_CHART_CANDLES);
  if (!Number.isFinite(requestedLimit) || requestedLimit <= 0) {
    throw new Error("VW-KAMA candle range limit must be positive.");
  }
  return {
    selected,
    request: {
      ...analysis,
      startTime,
      endTime,
      maxCandles: clamp(Math.round(requestedLimit), MIN_RANGE_CANDLES, MAX_RANGE_CANDLES),
    },
  };
}

function candidateWarmupMs(request: VwKamaInspectorRequest): number {
  const longest = Math.max(
    request.parameters.efficiencyMs,
    request.parameters.efficiencyVolumePower
      ? request.parameters.efficiencyVolumeEmaMs ?? request.parameters.volumeMs
      : 0,
    request.parameters.slowMs,
    request.parameters.volumeMs,
    request.parameters.rateEmaMs ?? request.intervalMs,
    thresholdNoiseEnabled(request.parameters)
      ? request.parameters.thresholdLookbackMs ?? 0
      : 0,
    request.parameters.confirmationMix
      ? request.parameters.confirmationAccelerationLookbackMs ?? 0
      : 0,
    request.parameters.confirmationMix
      ? request.parameters.confirmationDistanceLookbackMs ?? 0
      : 0,
    (request.parameters.confirmationMix && request.parameters.confirmationEmaWeight)
      || request.parameters.confirmationEmaGateStrength
      ? request.parameters.confirmationEmaMs ?? 0
      : 0,
    request.parameters.confirmationMix && request.parameters.confirmationRsiWeight
      ? request.parameters.confirmationRsiMs ?? 0
      : 0,
    request.parameters.confirmationMix && request.parameters.confirmationDmiWeight
      ? (request.parameters.confirmationDmiMs ?? 0) * 2
      : 0,
    meanReversionEnabled(request.parameters)
      ? Math.max(
        request.parameters.meanReversionEfficiencyMs ?? request.parameters.efficiencyMs,
        request.parameters.meanReversionFastMs ?? request.parameters.fastMs,
        request.parameters.meanReversionSlowMs ?? request.parameters.slowMs,
        request.parameters.meanReversionVolatilityMs ?? 0,
      )
      : 0,
  );
  return Math.max(1, Math.round(longest / request.intervalMs))
    * request.intervalMs
    * request.warmupMultiple;
}

function aggregateCandles(candles: Candle[], sourceMs: number, targetMs: number): Candle[] {
  if (sourceMs === targetMs) return candles.slice();
  const result: Candle[] = [];
  const expected = targetMs / sourceMs;
  let current: Candle | undefined;
  let bucket = -1;
  let count = 0;
  let contiguous = false;
  let previous = -1;
  const flush = () => {
    if (current && contiguous && count === expected) result.push(current);
  };
  for (const candle of candles) {
    const nextBucket = Math.floor(candle.openTime / targetMs) * targetMs;
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      count = 1;
      contiguous = candle.openTime === bucket;
      previous = candle.openTime;
      current = { ...candle, interval: duration(targetMs), openTime: bucket, closeTime: bucket + targetMs - 1 };
      continue;
    }
    if (!current) continue;
    contiguous &&= candle.openTime === previous + sourceMs;
    previous = candle.openTime;
    current.high = Math.max(current.high, candle.high);
    current.low = Math.min(current.low, candle.low);
    current.close = candle.close;
    current.volume += candle.volume;
    count += 1;
  }
  flush();
  return result;
}

function continuousSegments(candles: Candle[], intervalMs: number): Candle[][] {
  const result: Candle[][] = [];
  let current: Candle[] = [];
  for (const candle of candles) {
    if (current.length && candle.openTime !== current.at(-1)!.openTime + intervalMs) {
      result.push(current);
      current = [];
    }
    current.push(candle);
  }
  if (current.length) result.push(current);
  return result;
}

function renderCandleRange(
  segments: Candle[][],
  sourceIntervalMs: number,
  startTime: number,
  endTime: number,
  maxCandles: number,
): { intervalMs: number; sourceCount: number; candles: Candle[] } {
  const sourceSlots = Math.max(1, Math.ceil((endTime - startTime) / sourceIntervalMs));
  let multiplier = Math.max(1, Math.ceil(sourceSlots / maxCandles));
  let intervalMs = sourceIntervalMs * multiplier;
  while (Math.ceil(endTime / intervalMs) - Math.floor(startTime / intervalMs) > maxCandles) {
    multiplier += 1;
    intervalMs = sourceIntervalMs * multiplier;
  }
  const candles: Candle[] = [];
  let sourceCount = 0;

  for (const segment of segments) {
    const from = candleLowerBound(segment, startTime);
    const to = candleLowerBound(segment, endTime);
    sourceCount += to - from;
    if (from >= to) continue;
    candles.push(...renderSegmentCandles(segment.slice(from, to), intervalMs));
  }
  candles.sort((left, right) => left.openTime - right.openTime);
  return { intervalMs, sourceCount, candles };
}

function renderSegmentCandles(candles: Candle[], intervalMs: number): Candle[] {
  const result: Candle[] = [];
  let current: Candle | undefined;
  let bucket = -1;
  const flush = () => {
    if (!current) return;
    result.push({
      ...current,
      interval: duration(current.closeTime - current.openTime + 1),
    });
  };
  for (const candle of candles) {
    const nextBucket = Math.floor(candle.openTime / intervalMs) * intervalMs;
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      current = { ...candle };
      continue;
    }
    if (!current) continue;
    current.high = Math.max(current.high, candle.high);
    current.low = Math.min(current.low, candle.low);
    current.close = candle.close;
    current.closeTime = candle.closeTime;
    current.volume += candle.volume;
  }
  flush();
  return result;
}

function candleLowerBound(candles: Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.openTime < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function candleIndexAtClose(candles: Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.closeTime < time) low = middle + 1;
    else high = middle;
  }
  if (candles[low]?.closeTime !== time) {
    throw new Error("Rendered candle does not map to its source VW-KAMA candle.");
  }
  return low;
}

function sampledIndexes(length: number, scoreStartIndex: number, maxPoints: number): number[] {
  const sampleEvery = Math.max(1, Math.ceil((length - scoreStartIndex) / maxPoints));
  const result: number[] = [];
  for (let index = scoreStartIndex; index < length; index += sampleEvery) result.push(index);
  if (result.at(-1) !== length - 1) result.push(length - 1);
  return result;
}

function exposureOracleWithOpportunityEpsilon(
  oracle: ExposureValueOracle,
  opportunityEpsilon: number,
): ExposureValueOracle {
  const epsilon = Math.max(0, opportunityEpsilon);
  return {
    ...oracle,
    weights: Float32Array.from(oracle.averageRegrets, (regret) => regret + epsilon),
  };
}

function analysisCacheKey(request: VwKamaInspectorRequest): string {
  const {
    matchWindowMs: _matchWindowMs,
    timingHalfLifeMs: _timingHalfLifeMs,
    ...modelAndOracleRequest
  } = request;
  return JSON.stringify(modelAndOracleRequest);
}

function window(
  id: string,
  label: string,
  group: string,
  start: string,
  end: string,
  sourceIntervalMs = 1_000,
): VwKamaInspectorWindow {
  return {
    id,
    label,
    group,
    startTime: Date.parse(`${start}T00:00:00.000Z`),
    endTime: Date.parse(`${end}T00:00:00.000Z`) + DAY_MS,
    sourceIntervalMs,
  };
}

function resolveInspectorWindow(
  windowId: string,
  latestDays: number | undefined,
  now: number,
): VwKamaInspectorWindow {
  if (windowId === "latest") return latestInspectorWindow(latestDays, now);
  const selected = STATIC_WINDOWS.find((item) => item.id === windowId);
  if (!selected) throw new Error(`Unknown VW-KAMA window: ${windowId}`);
  return selected;
}

function latestInspectorWindow(requestedDays: number | undefined, now: number): VwKamaInspectorWindow {
  const days = normalizeLatestDays(requestedDays);
  const endTime = utcDay(now);
  const startTime = endTime - days * DAY_MS;
  if (!Number.isFinite(startTime) || Number.isNaN(new Date(startTime).getTime())) {
    throw new Error("Latest inspector history range exceeds the supported timestamp range.");
  }
  return {
    id: "latest",
    label: `Latest history · last ${days} day${days === 1 ? "" : "s"}`,
    group: "Latest",
    startTime,
    endTime,
    sourceIntervalMs: 1_000,
  };
}

function normalizeLatestDays(value: number | undefined): number {
  const days = value ?? 7;
  if (!Number.isSafeInteger(days) || days < 1) {
    throw new Error("Latest inspector history days must be a positive integer.");
  }
  return days;
}

function validateWindowScale(window: VwKamaInspectorWindow, intervalMs: number): void {
  if (intervalMs < window.sourceIntervalMs) {
    throw new Error(
      `${window.label} starts at ${duration(window.sourceIntervalMs)} candles; ${duration(intervalMs)} is unavailable.`,
    );
  }
}

async function readDailyShard(root: string, date: string): Promise<string> {
  try {
    return await fs.readFile(path.join(root, `${date}.jsonl`), "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const compressed = await fs.readFile(path.join(root, `${date}.jsonl.gz`));
  return gunzipSync(compressed).toString("utf8");
}

function duration(ms: number): string {
  if (ms % 3_600_000 === 0) return `${ms / 3_600_000}h`;
  if (ms % 60_000 === 0) return `${ms / 60_000}m`;
  return `${ms / 1_000}s`;
}

function thresholdNoiseEnabled(parameters: VwKamaParameters): boolean {
  return parameters.thresholdNoiseResponse === "inverse"
    ? (parameters.thresholdInverseMaxBpsHour ?? 0) > 0
    : (parameters.thresholdNoiseMultiplier ?? 0) > 0;
}

function meanReversionEnabled(parameters: VwKamaParameters): boolean {
  return (parameters.meanReversionReversalThreshold ?? 0) > 0;
}

function utcDay(time: number): number {
  return Math.floor(time / DAY_MS) * DAY_MS;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function eventRatio(value: number, total: number, opposite: number): number {
  return total > 0 ? value / total : opposite === 0 ? 1 : 0;
}

function harmonic(left: number, right: number): number {
  return left + right > 0 ? 2 * left * right / (left + right) : 0;
}

function percentile(values: number[], quantile: number): number | null {
  if (!values.length) return null;
  const sorted = values.slice().sort((left, right) => left - right);
  const index = (sorted.length - 1) * quantile;
  const lower = Math.floor(index);
  const fraction = index - lower;
  return sorted[lower]! * (1 - fraction) + (sorted[lower + 1] ?? sorted[lower]!) * fraction;
}
