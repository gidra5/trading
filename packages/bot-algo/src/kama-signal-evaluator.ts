import {
  EMAIndicator,
  RSIIndicator,
  VolumeWeightedKAMAIndicator,
  volumeWeightedKamaWarmupSamples,
} from "./indicators.js";
import {
  perfectMarginOracle,
  type PerfectMarginOracleResult,
} from "./perfect-margin-oracle.js";
import type {
  BacktestOraclePath,
  BacktestOraclePoint,
  BacktestOracleState,
} from "./backtest-trace.js";
import type {
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
  Candle,
} from "./legacy/types.js";
import type { TradingApi, TradingCandle } from "./trading-api.js";
import { KamaRateNoise } from "./kama-rate-noise.js";
import {
  clampPeakValleyKamaRate,
  peakValleyKamaRate,
  peakValleyKamaThresholdAdjustment,
  resolvePeakValleyKamaSignal,
} from "./peak-valley-kama-signal.js";
import type { PeakValleyStrategyConfig } from "./peak-valley-strategy.js";
import {
  createExposureReturnAccumulator,
  createExposureValueDistillationAccumulator,
  exposureValueOraclePathMetrics,
  exposureValueOracleProbabilities,
  finalizeExposureReturn,
  finalizeExposureValueDistillation,
  observeExposureReturn,
  observeExposureValueDistillation,
  strategyExposureProbabilities,
  strategyExposureLogSlope,
  strategyExposureQuadraticCoefficient,
  strategyExposureTransitionCrossEntropy,
  strategyExposureTransitionStatistics,
  strategyExposureTemperatures,
  strategyExposureVolatilities,
  type ExposureReturnMetrics,
  type ExposureValueDistillationLossConfig,
  type ExposureValueDistillationMetrics,
  type ExposureValueOracle,
} from "./exposure-value-distillation.js";

const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;
const F1_WEIGHT = 0.2;
const AGREEMENT_WEIGHT = 0.6;
const CLEANLINESS_WEIGHT = 0.2;
export const VW_KAMA_SCORE_VERSION = 12;

export type VwKamaDeadbandMode = "flat" | "hold" | "hysteresis";
export type VwKamaAgreementMode = "sizing" | "confidence";
export type VwKamaRateMode = "relative" | "log";
export type VwKamaThresholdResponse = "proportional" | "inverse";
export type VwKamaHoldingPeriodMode = "fixed" | "oracle-half-average-trade";
export type VwKamaValueHorizonEndMode = "truncate" | "extend";
export type VwKamaValueHorizonMode = "full-window" | "fixed";

export interface VwKamaParameters {
  efficiencyMs: number;
  efficiencyVolumeEmaMs?: number;
  efficiencyVolumePower?: number;
  fastMs: number;
  slowMs: number;
  power: number;
  volumeMs: number;
  volumeCap: number;
  volumePower: number;
  rateMode?: VwKamaRateMode;
  rateEmaMs?: number;
  deadbandBpsHour: number;
  deadbandMode: VwKamaDeadbandMode;
  hysteresisReleaseRatio?: number;
  thresholdLookbackMs?: number;
  thresholdNoiseResponse?: VwKamaThresholdResponse;
  thresholdNoiseMultiplier?: number;
  thresholdInverseMaxBpsHour?: number;
  thresholdInverseNoiseScaleBpsHour?: number;
  signalFrictionFraction?: number;
  strategyTemperature?: number;
  strategyQuadraticScale?: number;
  strategyQuadraticVolatilityMs?: number;
  strategyNormalMixture?: number;
  strategyNormalSigma?: number;
  buyMaxFraction?: number;
  sellMaxFraction?: number;
  buySizingSigmaBpsHour?: number;
  sellSizingSigmaBpsHour?: number;
  agreementMode?: VwKamaAgreementMode;
  confirmationMix?: number;
  confirmationMinQuality?: number;
  confirmationAccelerationLookbackMs?: number;
  confirmationDistanceLookbackMs?: number;
  confirmationAccelerationWeight?: number;
  confirmationDistanceWeight?: number;
  confirmationEmaMs?: number;
  confirmationEmaThresholdBpsHour?: number;
  confirmationEmaWeight?: number;
  confirmationEmaGateStrength?: number;
  confirmationRsiMs?: number;
  confirmationRsiThreshold?: number;
  confirmationRsiWeight?: number;
  confirmationDmiMs?: number;
  confirmationDmiWeight?: number;
  confirmationAdxThreshold?: number;
  confirmationBias?: number;
  meanReversionEfficiencyMs?: number;
  meanReversionFastMs?: number;
  meanReversionSlowMs?: number;
  meanReversionVolatilityMs?: number;
  meanReversionSuppressionThreshold?: number;
  meanReversionReversalThreshold?: number;
}

/**
 * Translate the deployable PeakValleyStrategy KAMA signal into this runner's
 * physical-duration and bps/hour parameterization. Optional experimental
 * confirmation and mean-reversion layers are deliberately disabled.
 */
export function vwKamaParametersFromPeakValleySignal(
  config: PeakValleyStrategyConfig,
  oracleFriction = config.kamaSignalFriction,
): VwKamaParameters {
  if (config.derivativeSource !== "kama") {
    throw new Error("VW-KAMA runner requires PeakValleyStrategy derivativeSource=kama.");
  }
  if (!config.relativeRateEnabled) {
    throw new Error("VW-KAMA runner requires relative PeakValleyStrategy rates.");
  }
  if (Math.abs(config.kamaRateThresholdHigh - config.kamaRateThresholdLow) > Number.EPSILON) {
    throw new Error("VW-KAMA runner currently requires symmetric KAMA low/high thresholds.");
  }
  if (
    config.buyConfirmationOffsets.length > 0
    || config.sellConfirmationOffsets.length > 0
    || config.buyExitConfirmationOffsets.length > 0
    || config.sellExitConfirmationOffsets.length > 0
  ) {
    throw new Error("VW-KAMA production baseline requires empty PeakValleyStrategy confirmation offsets.");
  }
  if (
    config.buyEntrySignalTiming !== "end"
    || config.sellEntrySignalTiming !== "end"
    || config.buyExitSignalTiming !== "start"
    || config.sellExitSignalTiming !== "start"
  ) {
    throw new Error("VW-KAMA production baseline requires end entry and start exit timing.");
  }
  if (!config.longSideEnabled || !config.shortSideEnabled) {
    throw new Error("VW-KAMA production baseline requires both long and short sides.");
  }
  const signalFrictionFraction = oracleFriction > 0
    ? config.kamaSignalFriction / oracleFriction
    : config.kamaSignalFriction === 0 ? 0 : Number.POSITIVE_INFINITY;
  if (!Number.isFinite(signalFrictionFraction) || signalFrictionFraction > 1) {
    throw new Error("PeakValleyStrategy signal friction cannot exceed the runner oracle friction.");
  }
  const duration = (samples: number) => samples * config.sampleIntervalMs;
  const rateScale = 10_000 * HOUR_MS / 1_000;
  return {
    efficiencyMs: duration(config.kamaErLen),
    efficiencyVolumeEmaMs: duration(config.kamaErVolumeLen),
    efficiencyVolumePower: config.kamaErVolumePower,
    fastMs: duration(config.kamaFastLen),
    slowMs: duration(config.kamaSlowLen),
    power: config.kamaPower,
    volumeMs: duration(config.kamaVolumeLen),
    volumeCap: config.kamaVolumeCap,
    volumePower: config.kamaVolumePower,
    rateMode: config.kamaRateMode,
    rateEmaMs: duration(config.kamaRateEmaLen),
    deadbandBpsHour: config.kamaRateThresholdLow * rateScale,
    deadbandMode: config.derivativeClampMode === "deadband"
      ? "flat"
      : config.derivativeClampMode,
    hysteresisReleaseRatio: config.derivativeClampInnerThresholdRatio,
    thresholdLookbackMs: config.kamaThresholdLookbackSec * 1_000,
    thresholdNoiseResponse: config.kamaThresholdNoiseResponse,
    thresholdNoiseMultiplier: config.kamaThresholdNoiseMultiplier,
    thresholdInverseMaxBpsHour: config.kamaThresholdInverseMax * rateScale,
    thresholdInverseNoiseScaleBpsHour: config.kamaThresholdInverseNoiseScale * rateScale,
    signalFrictionFraction,
    strategyTemperature: 0.001,
    strategyQuadraticScale: 0,
    strategyQuadraticVolatilityMs: HOUR_MS,
    strategyNormalMixture: 0,
    strategyNormalSigma: 25,
    buyMaxFraction: Math.min(1, config.buySpendRate),
    sellMaxFraction: Math.min(1, config.sellAmountRate),
    buySizingSigmaBpsHour: Number.MAX_VALUE,
    sellSizingSigmaBpsHour: Number.MAX_VALUE,
    agreementMode: "sizing",
    confirmationMix: 0,
    confirmationMinQuality: 0,
    meanReversionReversalThreshold: 0,
  };
}

export interface VwKamaInspectorWindow {
  id: string;
  label: string;
  group: string;
  startTime: number;
  endTime: number;
  sourceIntervalMs: number;
}

export interface VwKamaInspectorRequest {
  windowId: string;
  intervalMs: number;
  parameters: VwKamaParameters;
  oracleFriction: number;
  matchWindowMs: number;
  timingHalfLifeMs: number;
  warmupMultiple: number;
  valueDistillation?: VwKamaValueDistillationConfig;
}

export interface VwKamaValueDistillationConfig extends ExposureValueDistillationLossConfig {
  gridSize: number;
  minExposure: number;
  maxExposure: number;
  maxEffectiveExposure: number;
  initialExposure: number;
  holdingPeriodMode: VwKamaHoldingPeriodMode;
  holdingPeriodMs: number;
  valueHorizonMode?: VwKamaValueHorizonMode;
  valueHorizonMs: number;
  horizonEndMode: VwKamaValueHorizonEndMode;
  oracleTemperature: number;
  strategyVolatilityScaling: boolean;
  opportunityEpsilon: number;
  quoteLendRate: number;
  quoteBorrowRate: number;
  assetBorrowRate: number;
}

export interface VwKamaCandleRangeRequest extends VwKamaInspectorRequest {
  startTime: number;
  endTime: number;
  maxCandles?: number;
}

export interface VwKamaCandleRangeResponse {
  windowId: string;
  intervalMs: number;
  renderIntervalMs: number;
  startTime: number;
  endTime: number;
  sourceCandleCount: number;
  candles: Candle[];
  kamaSeries: BacktestChartSmaSeries;
  indicatorPoints: VwKamaIndicatorPoint[];
  valueDistributions: VwKamaValueDistributionPoint[];
  valueOraclePath: VwKamaValueOraclePathPoint[];
}

export interface VwKamaInspectorCatalog {
  windows: VwKamaInspectorWindow[];
  scales: Array<{ label: string; intervalMs: number }>;
  defaults: VwKamaInspectorRequest;
  presets: VwKamaPreset[];
}

export interface VwKamaPreset {
  id: string;
  label: string;
  scope: "global" | "window";
  windowId: string | null;
  intervalMs?: number | null;
  parameters: VwKamaParameters;
  score?: number;
  historicalScore?: number;
  scoreVersion?: number;
  source?: string;
  generatedAt?: string;
  incumbentScore?: number;
  optimization?: {
    algorithm: "random" | "de";
    objective?: "signal" | "value-distillation";
    population: number;
    generations: number;
    restarts: number;
    refinementRounds: number;
    elapsedMs: number;
    hindsight: boolean;
    valueDistillation?: VwKamaValueDistillationConfig;
  };
}

export interface VwKamaTransition {
  time: number;
  price: number;
  side: "buy" | "sell";
  fromState: BacktestOracleState;
  state: BacktestOracleState;
  fromExposure: number;
  exposure: number;
  sizeFraction: number;
  quality: number;
  acceleration: number;
  overextension: number;
  emaRate: number;
  rsi: number;
  dmi: number;
  adx: number;
  meanDistance: number;
  matchedTime: number | null;
  lagMs: number | null;
  timingCredit: number;
}

export interface VwKamaAccuracyMetrics {
  score: number;
  precision: number;
  recall: number;
  f1: number;
  rawPrecision: number;
  rawRecall: number;
  exposureAgreement: number;
  noiseSignalRatio: number | null;
  signalCleanliness: number;
  signalsPerDay: number;
  signalCount: number;
  oracleCount: number;
  matchedCount: number;
  extraSignalCount: number;
  missedOracleCount: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
  lagP95Ms: number | null;
  lagMedianSignedMs: number | null;
  valueDistillation?: VwKamaValueDistillationMetrics;
}

export interface VwKamaValueDistillationMetrics extends ExposureValueDistillationMetrics {
  holdingPeriodMs: number;
  valueHorizonMs: number;
  returns: {
    oracle: ExposureReturnMetrics;
    strategy: ExposureReturnMetrics;
  };
}

export interface VwKamaStatePoint {
  time: number;
  candidate: BacktestOracleState;
  oracle: BacktestOracleState;
  candidateExposure: number;
  oracleExposure: number;
}

export interface VwKamaIndicatorPoint {
  time: number;
  kama: number;
  kamaRate: number;
  kamaRateRaw: number;
  threshold: number;
  signalFrictionLower: number | null;
  signalFrictionUpper: number | null;
  efficiencyRatio: number;
  effectiveEfficiencyRatio: number;
  volume: number;
  volumeAverage: number;
  relativeVolume: number;
  alpha: number;
  confirmationEma: number;
  rsi: number;
  dmi: number;
  adx: number;
  meanReversionKama: number;
  meanDistance: number;
  signalIntent: BacktestOracleState | null;
  rejectionReasons: VwKamaSignalRejectionReason[];
}

export interface VwKamaValueDistributionPoint {
  time: number;
  candidateExposure: number;
  oracleMeanExposure: number;
  oracleModalExposure: number;
  oraclePathExposure: number;
  strategyMeanExposure: number;
  oraclePolicyMeanExposure: number;
  strategyPolicyMeanExposure: number;
  strategyRateBpsHour: number;
  oracleTemperature: number;
  strategyTemperature: number;
  friction: number;
  frictionFraction: number;
  strategyLinearCoefficient: number;
  strategyQuadraticVolatility: number;
  strategyQuadraticScale: number;
  strategyQuadraticCoefficient: number;
  strategyNormalMixture: number;
  strategyNormalSigma: number;
  oracleEntropy: number;
  oraclePolicyEntropy: number;
  strategyPolicyEntropy: number;
  currentExposureMinimum: number;
  currentExposureMaximum: number;
  currentExposureGridSize: number;
  opportunity: number;
  averageRegret: number;
  crossEntropy: number;
  postActionCrossEntropy: number;
  values: Array<{
    exposure: number;
    oracleProbability: number;
    strategyProbability: number;
  }>;
}

export interface VwKamaValueOraclePathPoint {
  time: number;
  exposure: number;
  equity: number;
}

export type VwKamaSignalRejectionReason =
  | "mean-reversion"
  | "ema-hard-gate"
  | "zero-quality"
  | "minimum-quality"
  | "signal-friction";

export interface VwKamaInspectorResponse {
  window: VwKamaInspectorWindow;
  intervalMs: number;
  renderIntervalMs: number;
  candleCount: number;
  sourceSegmentCount: number;
  scoredSegmentCount: number;
  elapsedMs: number;
  metrics: VwKamaAccuracyMetrics;
  candles: Candle[];
  kamaSeries: BacktestChartSmaSeries;
  annotations: BacktestChartAnnotation[];
  oracle: BacktestOraclePath;
  candidatePath: { points: BacktestOraclePoint[] };
  statePoints: VwKamaStatePoint[];
  indicatorPoints: VwKamaIndicatorPoint[];
  candidateTransitions: VwKamaTransition[];
  oracleTransitions: VwKamaTransition[];
  valueDistributions: VwKamaValueDistributionPoint[];
  valueOraclePath: VwKamaValueOraclePathPoint[];
}

export interface EvaluateVwKamaOptions extends Omit<VwKamaInspectorRequest, "windowId" | "valueDistillation"> {
  scoreStartTime: number;
  scoreStartIndex?: number;
  maxPoints?: number;
  traceTimes?: readonly number[];
  oracleResult?: PerfectMarginOracleResult;
  preparedOracle?: VwKamaPreparedOracle;
  includeTrace?: boolean;
  valueDistillation?: {
    oracle: ExposureValueOracle;
    strategyVolatilityScaling: boolean;
    lossConfig: ExposureValueDistillationLossConfig;
    strategyTemperatures?: Float32Array;
    strategyQuadraticVolatilities?: Float32Array;
  };
}

export interface VwKamaCandleColumns {
  length: number;
  openTime: Float64Array;
  closeTime: Float64Array;
  open: Float64Array;
  high: Float64Array;
  low: Float64Array;
  close: Float64Array;
  volume: Float64Array;
}

export type VwKamaCandleSeries = readonly TradingCandle[] | VwKamaCandleColumns;

export interface VwKamaPreparedOracle {
  stateCodes: Uint8Array;
  transitions: readonly VwKamaTransition[];
  result?: PerfectMarginOracleResult;
}

export function columnarVwKamaCandles(
  candles: readonly TradingCandle[],
  shared = false,
): VwKamaCandleColumns {
  const column = (): Float64Array => shared
    ? new Float64Array(new SharedArrayBuffer(candles.length * Float64Array.BYTES_PER_ELEMENT))
    : new Float64Array(candles.length);
  const result: VwKamaCandleColumns = {
    length: candles.length,
    openTime: column(),
    closeTime: column(),
    open: column(),
    high: column(),
    low: column(),
    close: column(),
    volume: column(),
  };
  for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index]!;
    result.openTime[index] = candle.openTime;
    result.closeTime[index] = candle.closeTime;
    result.open[index] = candle.open;
    result.high[index] = candle.high;
    result.low[index] = candle.low;
    result.close[index] = candle.close;
    result.volume[index] = candle.volume;
  }
  return result;
}

export function prepareVwKamaOracle(
  candles: VwKamaCandleSeries,
  scoreStartIndex: number,
  result: PerfectMarginOracleResult,
  shared = false,
  includeResult = false,
): VwKamaPreparedOracle {
  const stateCodes = shared
    ? new Uint8Array(new SharedArrayBuffer(result.stateCodes.length))
    : result.stateCodes;
  if (stateCodes !== result.stateCodes) stateCodes.set(result.stateCodes);
  return {
    stateCodes,
    transitions: prepareVwKamaOracleTransitions(candles, scoreStartIndex, stateCodes),
    ...(includeResult ? { result } : {}),
  };
}

/** Mean elapsed time between consecutive executable oracle state changes. */
export function averageVwKamaOracleTradeIntervalMs(
  candles: VwKamaCandleSeries,
  scoreStartIndex: number,
  stateCodes: Uint8Array,
): number | null {
  let previousTime: number | null = null;
  let intervalSum = 0;
  let intervalCount = 0;
  for (let index = Math.max(1, scoreStartIndex); index < candles.length; index += 1) {
    if ((stateCodes[index] ?? 0) === (stateCodes[index - 1] ?? 0)) continue;
    const time = closeTimeAt(candles, index);
    if (previousTime !== null) {
      intervalSum += time - previousTime;
      intervalCount += 1;
    }
    previousTime = time;
  }
  return intervalCount > 0 ? intervalSum / intervalCount : null;
}

/** Resolves fixed H or half the oracle's average inter-trade interval to candle steps. */
export function resolveVwKamaHoldingPeriodSteps(
  candles: VwKamaCandleSeries,
  scoreStartIndex: number,
  stateCodes: Uint8Array,
  intervalMs: number,
  config: Pick<VwKamaValueDistillationConfig, "holdingPeriodMode" | "holdingPeriodMs">,
): number {
  const average = config.holdingPeriodMode === "oracle-half-average-trade"
    ? averageVwKamaOracleTradeIntervalMs(candles, scoreStartIndex, stateCodes)
    : null;
  const holdingPeriodMs = average === null ? config.holdingPeriodMs : average * 0.5;
  return Math.max(1, Math.round(holdingPeriodMs / intervalMs));
}

export type VwKamaEvaluation = Omit<
  VwKamaInspectorResponse,
  | "window"
  | "renderIntervalMs"
  | "sourceSegmentCount"
  | "scoredSegmentCount"
  | "elapsedMs"
  | "candles"
>;

export function evaluateVwKamaOracle(
  candles: VwKamaCandleSeries,
  options: EvaluateVwKamaOptions,
): VwKamaEvaluation {
  if (candles.length === 0) throw new Error("VW-KAMA inspection requires candles.");
  validate(options);
  const scoreStart = options.scoreStartIndex ?? lowerBound(candles, options.scoreStartTime);
  if (scoreStart >= candles.length) throw new Error("VW-KAMA score window has no candles.");
  const includeTrace = options.includeTrace !== false;
  const oracleResult = options.preparedOracle?.result
    ?? options.oracleResult
    ?? (!isCandleColumns(candles) ? perfectMarginOracle(candles, {
      startingQuote: 1,
      leverage: 1,
      friction: options.oracleFriction,
      eventMode: "close",
      maxPathCandles: options.maxPoints ?? 2_000,
    }) : undefined);
  if (!oracleResult && !options.preparedOracle) {
    throw new Error("Columnar VW-KAMA evaluation requires a prepared oracle.");
  }
  if (includeTrace && !oracleResult) {
    throw new Error("VW-KAMA trace evaluation requires the full oracle result.");
  }
  const oracleStateCodes = options.preparedOracle?.stateCodes ?? oracleResult!.stateCodes;
  if (oracleStateCodes.length < candles.length) {
    throw new Error("VW-KAMA oracle state count does not cover the candle series.");
  }
  const oracleTransitions = options.preparedOracle?.transitions
    ?? prepareVwKamaOracleTransitions(candles, scoreStart, oracleStateCodes);

  const periods = {
    efficiencyPeriod: samples(options.parameters.efficiencyMs, options.intervalMs),
    efficiencyVolumePeriod: samples(
      options.parameters.efficiencyVolumeEmaMs ?? options.parameters.volumeMs,
      options.intervalMs,
    ),
    efficiencyVolumePower: options.parameters.efficiencyVolumePower ?? 0,
    fastPeriod: samples(options.parameters.fastMs, options.intervalMs),
    slowPeriod: samples(options.parameters.slowMs, options.intervalMs),
    volumePeriod: samples(options.parameters.volumeMs, options.intervalMs),
  };
  const warmup = volumeWeightedKamaWarmupSamples(periods, options.warmupMultiple);
  const thresholdSamples = samples(options.parameters.thresholdLookbackMs ?? options.intervalMs, options.intervalMs);
  const rateEmaSamples = samples(options.parameters.rateEmaMs ?? options.intervalMs, options.intervalMs);
  const accelerationSamples = samples(
    options.parameters.confirmationAccelerationLookbackMs ?? options.intervalMs,
    options.intervalMs,
  );
  const distanceSamples = samples(
    options.parameters.confirmationDistanceLookbackMs ?? options.intervalMs,
    options.intervalMs,
  );
  const emaSamples = samples(
    options.parameters.confirmationEmaMs ?? options.intervalMs,
    options.intervalMs,
  );
  const rsiSamples = samples(
    options.parameters.confirmationRsiMs ?? options.intervalMs,
    options.intervalMs,
  );
  const dmiSamples = samples(
    options.parameters.confirmationDmiMs ?? options.intervalMs,
    options.intervalMs,
  );
  const meanReversionPeriods = {
    ...periods,
    efficiencyPeriod: samples(
      options.parameters.meanReversionEfficiencyMs ?? options.parameters.efficiencyMs,
      options.intervalMs,
    ),
    fastPeriod: samples(
      options.parameters.meanReversionFastMs ?? options.parameters.fastMs,
      options.intervalMs,
    ),
    slowPeriod: samples(
      options.parameters.meanReversionSlowMs ?? options.parameters.slowMs,
      options.intervalMs,
    ),
  };
  const meanReversionKamaWarmup = volumeWeightedKamaWarmupSamples(
    meanReversionPeriods,
    options.warmupMultiple,
  );
  const meanReversionVolatilitySamples = samples(
    options.parameters.meanReversionVolatilityMs ?? options.parameters.slowMs,
    options.intervalMs,
  );
  const thresholdEnabled = thresholdNoiseEnabled(options.parameters);
  const confirmationEnabled = (options.parameters.confirmationMix ?? 0) > 0;
  const accelerationEnabled = confirmationEnabled
    && (options.parameters.confirmationAccelerationWeight ?? 1) > 0;
  const distanceEnabled = confirmationEnabled
    && (options.parameters.confirmationDistanceWeight ?? 1) > 0;
  const emaEnabled = confirmationEnabled && (options.parameters.confirmationEmaWeight ?? 0) > 0
    || (options.parameters.confirmationEmaGateStrength ?? 0) > 0;
  const rsiEnabled = confirmationEnabled && (options.parameters.confirmationRsiWeight ?? 0) > 0;
  const dmiEnabled = confirmationEnabled && (options.parameters.confirmationDmiWeight ?? 0) > 0;
  const meanReversionEnabled = (options.parameters.meanReversionReversalThreshold ?? 0) > 0;
  const computeAcceleration = includeTrace || accelerationEnabled;
  const computeDistance = includeTrace || distanceEnabled;
  const computeEma = includeTrace || emaEnabled;
  const computeRsi = includeTrace || rsiEnabled;
  const computeDmi = includeTrace || dmiEnabled;
  const feedStart = Math.max(0, scoreStart - Math.max(
    warmup,
    rateEmaSamples * options.warmupMultiple,
    thresholdEnabled ? thresholdSamples * options.warmupMultiple : 0,
    accelerationEnabled ? accelerationSamples * options.warmupMultiple : 0,
    distanceEnabled ? distanceSamples * options.warmupMultiple : 0,
    emaEnabled ? emaSamples * options.warmupMultiple : 0,
    rsiEnabled ? rsiSamples * options.warmupMultiple : 0,
    dmiEnabled ? dmiSamples * options.warmupMultiple * 2 : 0,
    meanReversionEnabled
      ? Math.max(
        meanReversionKamaWarmup,
        meanReversionVolatilitySamples * options.warmupMultiple,
      )
      : 0,
  ));
  const indicator = new VolumeWeightedKAMAIndicator({} as TradingApi, {
    ...periods,
    power: options.parameters.power,
    volumeCap: options.parameters.volumeCap,
    volumePower: options.parameters.volumePower,
  });
  const rateEma = new EMAIndicator(rateEmaSamples);
  const slowEma = new EMAIndicator(emaSamples);
  const rsi = new RSIIndicator(rsiSamples, {} as TradingApi);
  const dmi = new DmiAdx(dmiSamples);
  const meanReversionKama = new VolumeWeightedKAMAIndicator({} as TradingApi, {
    ...meanReversionPeriods,
    power: options.parameters.power,
    volumeCap: options.parameters.volumeCap,
    volumePower: options.parameters.volumePower,
  });
  const meanReversionVariance = new EMAIndicator(meanReversionVolatilitySamples);
  let meanReversionSignedDistance = 0;
  const updateMeanReversion = (nextCandle: TradingCandle): void => {
    meanReversionKama.onTick({ eventTime: nextCandle.closeTime, candle: nextCandle });
    const mean = meanReversionKama.indicator();
    meanReversionSignedDistance = mean > 0 ? nextCandle.close / mean - 1 : 0;
    meanReversionVariance.onTick({
      eventTime: nextCandle.closeTime,
      value: Math.max(Number.EPSILON, meanReversionSignedDistance ** 2),
    });
  };
  const candidateStates = includeTrace ? new Int8Array(candles.length) : null;
  const candidateExposures = includeTrace ? new Float64Array(candles.length) : null;
  const candidateTransitions: VwKamaTransition[] = [];
  const points: BacktestOraclePoint[] = [];
  const indicatorPoints: VwKamaIndicatorPoint[] = [];
  const valueDistributions: VwKamaValueDistributionPoint[] = [];
  const valueOraclePath: VwKamaValueOraclePathPoint[] = [];
  const kamaSeries: BacktestChartSmaSeries = {
    index: -1,
    windowSec: options.parameters.slowMs / 1_000,
    label: "VW-KAMA",
    color: "#f472b6",
    points: [],
  };
  const maxPoints = Math.max(1, Math.round(options.maxPoints ?? 2_000));
  const requestedTraceTimes = options.traceTimes ? new Set(options.traceTimes) : null;
  const sampleEvery = Math.max(1, Math.ceil((candles.length - scoreStart) / maxPoints));
  const agreementMode = options.parameters.agreementMode ?? "sizing";
  const valueExposureMinimum = options.valueDistillation?.oracle.grid[0] ?? -1;
  const valueExposureMaximum = options.valueDistillation?.oracle.grid.at(-1) ?? 1;
  let current = 0;
  let trend = 0;
  let targetFraction = 0;
  const candleBuffers: [TradingCandle, TradingCandle] = [emptyCandle(), emptyCandle()];
  let candle = candleBuffers[0];
  let anchorPrice = closeAt(candles, feedStart);
  let lastSignalPrice: number | null = null;
  const rateNoise = new KamaRateNoise(thresholdSamples);
  const accelerationNoise = new KamaRateNoise(accelerationSamples);
  const accelerationAlpha = 2 / (accelerationSamples + 1);
  const distanceAlpha = 2 / (distanceSamples + 1);
  let previousRate: number | null = null;
  let smoothedRateChange = 0;
  let distanceNoise = 0;
  let stateCredit = 0;
  const valueDistillation = options.valueDistillation
    ? createExposureValueDistillationAccumulator(
        options.valueDistillation.lossConfig,
        options.valueDistillation.oracle.grid.length,
      )
    : null;
  const valueReturns = options.valueDistillation
    ? {
        strategy: createExposureReturnAccumulator(),
      }
    : null;
  const valuePrices = options.valueDistillation
    ? isCandleColumns(candles) ? candles.close : candles.map((item) => item.close)
    : null;
  const strategyQuadraticVolatilities = options.valueDistillation
    ? options.valueDistillation.strategyQuadraticVolatilities ?? strategyExposureVolatilities(
      valuePrices!,
      Math.max(1, Math.round(
        (options.parameters.strategyQuadraticVolatilityMs ?? HOUR_MS) / options.intervalMs,
      )),
    )
    : null;
  const strategyTemperatures = options.valueDistillation
    ? options.valueDistillation.strategyTemperatures ?? strategyExposureTemperatures(
      valuePrices!,
      {
        intervalMs: options.intervalMs,
        holdingPeriodSteps: options.valueDistillation.oracle.holdingPeriodSteps,
        temperature: 1,
        scaleByVolatility: options.valueDistillation.strategyVolatilityScaling,
      },
    )
    : null;
  if ((strategyTemperatures && strategyTemperatures.length < candles.length)
    || (strategyQuadraticVolatilities
      && strategyQuadraticVolatilities.length < candles.length)) {
    throw new Error("VW-KAMA strategy calibration does not cover the candle series.");
  }
  const signalFriction = options.oracleFriction
    * Math.max(0, options.parameters.signalFrictionFraction ?? 1);

  if (includeTrace) {
    const emaFeedStart = Math.max(0, scoreStart - emaSamples * options.warmupMultiple);
    const rsiFeedStart = Math.max(0, scoreStart - rsiSamples * options.warmupMultiple);
    const dmiFeedStart = Math.max(0, scoreStart - dmiSamples * options.warmupMultiple * 2);
    const meanReversionFeedStart = Math.max(
      0,
      scoreStart - Math.max(
        meanReversionKamaWarmup,
        meanReversionVolatilitySamples * options.warmupMultiple,
      ),
    );
    const diagnosticFeedStart = Math.min(
      emaFeedStart,
      rsiFeedStart,
      dmiFeedStart,
      meanReversionFeedStart,
    );
    for (let index = diagnosticFeedStart; index < feedStart; index += 1) {
      const diagnosticCandle = candleAt(
        candles,
        index,
        index % 2 === 0 ? candleBuffers[0] : candleBuffers[1],
      );
      if (index >= emaFeedStart) {
        slowEma.onTick({ eventTime: diagnosticCandle.closeTime, candle: diagnosticCandle });
      }
      if (index >= rsiFeedStart) {
        rsi.onTick({ eventTime: diagnosticCandle.closeTime, candle: diagnosticCandle });
      }
      if (index >= dmiFeedStart) dmi.update(diagnosticCandle);
      if (index >= meanReversionFeedStart) {
        updateMeanReversion(diagnosticCandle);
      }
    }
  }

  for (let index = feedStart; index < candles.length; index += 1) {
    candle = candleAt(candles, index, index % 2 === 0 ? candleBuffers[0] : candleBuffers[1]);
    indicator.onTick({ eventTime: candle.closeTime, candle });
    if (computeEma) slowEma.onTick({ eventTime: candle.closeTime, candle });
    if (computeRsi) rsi.onTick({ eventTime: candle.closeTime, candle });
    const dmiValue = computeDmi ? dmi.update(candle) : { direction: 0, adx: 0 };
    if (includeTrace || meanReversionEnabled) {
      updateMeanReversion(candle);
    }
    const kama = indicator.indicator();
    const rawRate = peakValleyKamaRate(
      indicator.derivative(),
      kama,
      options.intervalMs / 1_000,
      true,
      options.parameters.rateMode === "log" ? "log" : "relative",
    ) * 10_000 * HOUR_MS / 1_000;
    rateEma.onTick({ eventTime: candle.closeTime, value: rawRate });
    const rate = rateEma.indicator();
    const noise = thresholdEnabled ? rateNoise.update(rate) : 0;
    const rateChange = computeAcceleration && previousRate !== null ? rate - previousRate : 0;
    const accelerationScale = computeAcceleration ? accelerationNoise.update(rate) : 0;
    if (computeAcceleration) {
      smoothedRateChange += accelerationAlpha * (rateChange - smoothedRateChange);
      previousRate = rate;
    }
    const distance = kama > 0 ? (candle.close / kama - 1) * 10_000 : 0;
    if (computeDistance) distanceNoise += distanceAlpha * (Math.abs(distance) - distanceNoise);
    const acceleration = computeAcceleration && accelerationScale > Number.EPSILON
      ? smoothedRateChange / accelerationScale
      : 0;
    const overextension = computeDistance && distanceNoise > Number.EPSILON ? distance / distanceNoise : 0;
    const emaRate = computeEma && slowEma.indicator() > 0
      ? slowEma.derivative() / slowEma.indicator() * 10_000 * HOUR_MS / options.intervalMs
      : 0;
    const rsiValue = computeRsi ? rsi.indicator() : 50;
    const threshold = options.parameters.deadbandBpsHour
      + thresholdNoiseAdjustment(noise, options.parameters);
    const previousTrend = trend;
    trend = candidateState(rate, trend, threshold, options.parameters);
    const sourceEdge = trend !== previousTrend;
    const meanReversionVolatility = Math.sqrt(Math.max(0, meanReversionVariance.indicator()));
    const normalizedMeanDistance = Math.abs(meanReversionSignedDistance)
      / Math.max(Number.EPSILON, meanReversionVolatility);
    const proposed = !meanReversionEnabled || trend === 0
      ? trend
      : normalizedMeanDistance >= (options.parameters.meanReversionReversalThreshold ?? 0)
        ? -trend
        : normalizedMeanDistance >= (options.parameters.meanReversionSuppressionThreshold ?? 0)
          ? 0
          : trend;
    const emaAligned = proposed === 0 || emaTrendAligned(proposed, emaRate, options.parameters);
    const desired = proposed !== 0
      && clamp01(options.parameters.confirmationEmaGateStrength ?? 0) === 1
      && !emaAligned
      ? 0
      : proposed;
    const confirmation = desired === 0
      ? 1
      : confirmationQuality(
        desired,
        acceleration,
        overextension,
        emaRate,
        rsiValue,
        dmiValue.direction,
        dmiValue.adx,
        options.parameters,
      );
    const quality = desired === 0 ? 1 : confirmation;
    const minimumQuality = clamp01(options.parameters.confirmationMinQuality ?? 0);
    const positiveQuality = desired === 0 || quality > 0;
    const sufficientQuality = quality >= minimumQuality;
    const signal = resolvePeakValleyKamaSignal(
      desired,
      candle.close,
      { candidate: previousTrend, accepted: current, lastSignalPrice },
      {
        sourceEdge,
        signalFriction,
        transitionAllowed: positiveQuality && sufficientQuality,
      },
    );
    const transitionRequested = signal.transitionRequested;
    const beyondFriction = signal.beyondFriction;
    const accepted = signal.transitionAccepted;
    const signalIntent = includeTrace && sourceEdge && trend !== current
      ? stateName(trend)
      : null;
    const fromAgreementExposure = candidateAgreementValue(
      agreementMode,
      current,
      targetFraction,
      anchorPrice,
      candle.close,
    );
    const fromExposure = options.valueDistillation
      ? index > scoreStart && valueReturns
        ? valueReturns.strategy.exposure
        : candidateValueExposure(
            current,
            targetFraction,
            anchorPrice,
            candle.close,
            valueExposureMinimum,
            valueExposureMaximum,
          )
      : fromAgreementExposure;
    let requestedValueExposure = fromExposure;
    if (accepted) {
      const nextFraction = desired === 0
        ? 0
        : signalFraction(desired, rate, options.parameters) * quality;
      const nextExposure = options.valueDistillation
        ? candidateValueTarget(
            desired,
            nextFraction,
            valueExposureMinimum,
            valueExposureMaximum,
          )
        : desired * nextFraction;
      requestedValueExposure = nextExposure;
      if (index >= scoreStart) {
        candidateTransitions.push(baseTransition(
          candle,
          current,
          desired,
          fromExposure,
          nextExposure,
          nextFraction,
          quality,
          acceleration,
          overextension,
          emaRate,
          rsiValue,
          dmiValue.direction,
          dmiValue.adx,
          meanReversionSignedDistance / Math.max(Number.EPSILON, meanReversionVolatility),
        ));
        if (includeTrace) points.push(statePoint(candle, current, desired));
      }
      current = desired;
      targetFraction = nextFraction;
      anchorPrice = candle.close;
      lastSignalPrice = signal.lastSignalPrice;
    }
    const agreementExposure = candidateAgreementValue(
      agreementMode,
      current,
      targetFraction,
      anchorPrice,
      candle.close,
    );
    const candidateExposure = options.valueDistillation
      ? requestedValueExposure
      : agreementExposure;
    if (index >= scoreStart) {
      stateCredit += agreementCredit(
        agreementMode,
        current,
        agreementExposure,
        exposureFromCode(oracleStateCodes[index] ?? 0),
      );
      if (valueDistillation && options.valueDistillation) {
        const strategyTemperature = (options.parameters.strategyTemperature ?? 0.001)
          * strategyTemperatures![index]!;
        const quadraticCoefficient = strategyExposureQuadraticCoefficient(
          options.parameters.strategyQuadraticScale ?? 0,
          strategyQuadraticVolatilities![index]!,
        );
        observeExposureValueDistillation(
          valueDistillation,
          options.valueDistillation.oracle,
          index,
          rate,
          options.intervalMs,
          strategyTemperature,
          quadraticCoefficient,
          options.parameters.signalFrictionFraction ?? 1,
          candidateExposure,
          options.parameters.strategyNormalMixture ?? 0,
          options.parameters.strategyNormalSigma ?? 25,
        );
        if (valueReturns && index + 1 < candles.length) {
          const price = closeAt(candles, index);
          const nextPrice = closeAt(candles, index + 1);
          observeExposureReturn(
            valueReturns.strategy,
            candidateExposure,
            price,
            nextPrice,
            options.valueDistillation.oracle.execution,
          );
        }
      }
    }
    if (candidateStates) candidateStates[index] = current;
    if (candidateExposures) candidateExposures[index] = candidateExposure;
    const tracePoint = requestedTraceTimes
      ? requestedTraceTimes.has(candle.closeTime)
      : index >= scoreStart
        && ((index - scoreStart) % sampleEvery === 0 || index === candles.length - 1);
    if (includeTrace && tracePoint) {
      kamaSeries.points.push({ time: candle.closeTime, value: kama });
      const kamaDetails = indicator.details();
      const rejectionReasons: VwKamaSignalRejectionReason[] = [];
      if (signalIntent) {
        if (proposed !== trend) rejectionReasons.push("mean-reversion");
        if (desired !== proposed) rejectionReasons.push("ema-hard-gate");
        if (!accepted) {
          if (transitionRequested && !positiveQuality) rejectionReasons.push("zero-quality");
          else if (transitionRequested && !sufficientQuality) rejectionReasons.push("minimum-quality");
          else if (transitionRequested && !beyondFriction) rejectionReasons.push("signal-friction");
        }
      }
      indicatorPoints.push({
        time: candle.closeTime,
        kama,
        kamaRate: rate,
        kamaRateRaw: rawRate,
        threshold,
        signalFrictionLower: lastSignalPrice === null
          ? null
          : lastSignalPrice * (1 - signalFriction),
        signalFrictionUpper: lastSignalPrice === null
          ? null
          : lastSignalPrice * (1 + signalFriction),
        efficiencyRatio: kamaDetails.efficiencyRatio,
        effectiveEfficiencyRatio: kamaDetails.effectiveEfficiencyRatio,
        volume: candle.volume,
        volumeAverage: indicator.volumeAverage(),
        relativeVolume: kamaDetails.relativeVolume,
        alpha: kamaDetails.alpha,
        confirmationEma: slowEma.indicator(),
        rsi: rsiValue,
        dmi: dmiValue.direction,
        adx: dmiValue.adx,
        meanReversionKama: meanReversionKama.indicator(),
        meanDistance: meanReversionSignedDistance
          / Math.max(Number.EPSILON, meanReversionVolatility),
        signalIntent,
        rejectionReasons,
      });
      if (options.valueDistillation?.oracle.probabilities) {
        valueDistributions.push(valueDistributionPoint(
          candle.closeTime,
          index,
          candidateExposure,
          rate,
          options.intervalMs,
          options.valueDistillation.oracle,
          (options.parameters.strategyTemperature ?? 0.001) * strategyTemperatures![index]!,
          options.parameters.strategyQuadraticScale ?? 0,
          strategyQuadraticVolatilities![index]!,
          options.parameters.signalFrictionFraction ?? 1,
          options.parameters.strategyNormalMixture ?? 0,
          options.parameters.strategyNormalSigma ?? 25,
        ));
      }
      if (points.at(-1)?.time !== candle.closeTime) points.push(statePoint(candle, current, current));
    }
  }

  const alignment = alignVwKamaTransitionsInternal(
    candidateTransitions,
    oracleTransitions,
    options,
    !options.preparedOracle,
  );
  const stateCount = candles.length - scoreStart;
  const precision = eventRatio(alignment.credit, candidateTransitions.length, oracleTransitions.length);
  const recall = eventRatio(alignment.credit, oracleTransitions.length, candidateTransitions.length);
  const f1 = harmonic(precision, recall);
  const matchedCount = alignment.matches.length;
  const extraSignalCount = candidateTransitions.length - matchedCount;
  const signalCleanliness = candidateTransitions.length > 0
    ? matchedCount / candidateTransitions.length
    : 1;
  const lags = alignment.matches.map((item) => item.lagMs);
  const absoluteLags = lags.map(Math.abs);
  const metrics: VwKamaAccuracyMetrics = {
    score: vwKamaScore(f1, stateCredit / stateCount, signalCleanliness),
    precision,
    recall,
    f1,
    rawPrecision: eventRatio(alignment.matches.length, candidateTransitions.length, oracleTransitions.length),
    rawRecall: eventRatio(alignment.matches.length, oracleTransitions.length, candidateTransitions.length),
    exposureAgreement: stateCredit / stateCount,
    noiseSignalRatio: noiseSignalRatio(extraSignalCount, matchedCount),
    signalCleanliness,
    signalsPerDay: candidateTransitions.length / Math.max(options.intervalMs / DAY_MS, stateCount * options.intervalMs / DAY_MS),
    signalCount: candidateTransitions.length,
    oracleCount: oracleTransitions.length,
    matchedCount,
    extraSignalCount,
    missedOracleCount: oracleTransitions.length - matchedCount,
    lagP50Ms: percentile(absoluteLags, 0.5),
    lagP90Ms: percentile(absoluteLags, 0.9),
    lagP95Ms: percentile(absoluteLags, 0.95),
    lagMedianSignedMs: percentile(lags, 0.5),
    ...(valueDistillation && valueReturns
      ? { valueDistillation: {
          ...finalizeExposureValueDistillation(valueDistillation),
          holdingPeriodMs: options.valueDistillation!.oracle.holdingPeriodSteps * options.intervalMs,
          valueHorizonMs: options.valueDistillation!.oracle.valueHorizonSteps * options.intervalMs,
          returns: {
            oracle: exposureValueOraclePathMetrics(options.valueDistillation!.oracle),
            strategy: finalizeExposureReturn(valueReturns.strategy),
          },
        } }
      : {}),
  };
  const scoredStartTime = closeTimeAt(candles, scoreStart);
  const oraclePoints = includeTrace ? slicePath(oracleResult!.path.points, scoredStartTime) : [];
  const sampledIndexes = new Set(kamaSeries.points.map((point) => point.time));
  const statePoints: VwKamaStatePoint[] = [];
  if (includeTrace) {
    for (let index = scoreStart; index < candles.length; index += 1) {
      const candle = candleAt(candles, index, candleBuffers[index % 2]!);
      const valuePath = options.valueDistillation?.oracle.path;
      if (!sampledIndexes.has(candle.closeTime)
        && index !== valuePath?.startIndex
        && index !== valuePath?.terminalIndex) continue;
      statePoints.push({
        time: candle.closeTime,
        candidate: stateName(candidateStates![index]!),
        oracle: stateName(exposureFromCode(oracleStateCodes[index] ?? 0)),
        candidateExposure: candidateExposures![index]!,
        oracleExposure: exposureFromCode(oracleStateCodes[index] ?? 0),
      });
      if (valuePath && index >= valuePath.startIndex && index <= valuePath.terminalIndex) {
        valueOraclePath.push({
          time: candle.closeTime,
          exposure: valuePath.exposures[index]!,
          equity: valuePath.equities[index]!,
        });
      }
    }
  }

  return {
    intervalMs: options.intervalMs,
    candleCount: stateCount,
    metrics,
    kamaSeries,
    annotations: includeTrace ? candidateTransitions.map((item) => ({
      time: item.time,
      price: item.price,
      kind: item.side === "buy" ? "buy-signal" : "sell-signal",
      label: `VW-KAMA ${item.fromState} → ${item.state} · ${(item.sizeFraction * 100).toFixed(1)}%`,
      signalState: item.state,
      reason: `Quality ${(item.quality * 100).toFixed(1)}% · mean distance ${item.meanDistance.toFixed(2)}σ · EMA ${item.emaRate.toFixed(2)} bps/h · RSI ${item.rsi.toFixed(1)} · DMI ${item.dmi.toFixed(1)} / ADX ${item.adx.toFixed(1)} · ${item.lagMs === null ? "no one-to-one oracle match" : `oracle timing offset ${item.lagMs} ms`}`,
    })) : [],
    oracle: oracleResult
      ? { ...oracleResult.path, points: oraclePoints }
      : {
        mode: "fixed-notional",
        eventMode: "close",
        leverage: 1,
        friction: options.oracleFriction,
        points: [],
      },
    candidatePath: { points: includeTrace ? points.sort((left, right) => left.time - right.time) : [] },
    statePoints,
    indicatorPoints,
    valueDistributions,
    valueOraclePath,
    candidateTransitions,
    oracleTransitions: oracleTransitions as VwKamaTransition[],
  };
}

function valueDistributionPoint(
  time: number,
  candleIndex: number,
  candidateExposure: number,
  rateBpsPerHour: number,
  intervalMs: number,
  oracle: ExposureValueOracle,
  strategyTemperature: number,
  strategyQuadraticScale: number,
  strategyQuadraticVolatility: number,
  frictionFraction: number,
  strategyNormalMixture: number,
  strategyNormalSigma: number,
): VwKamaValueDistributionPoint {
  const strategyQuadraticCoefficient = strategyExposureQuadraticCoefficient(
    strategyQuadraticScale,
    strategyQuadraticVolatility,
  );
  const oracleProbabilities = exposureValueOracleProbabilities(oracle, candleIndex);
  const strategyProbabilities = strategyExposureProbabilities(
    oracle.grid,
    rateBpsPerHour,
    intervalMs * oracle.holdingPeriodSteps,
    strategyTemperature,
    strategyQuadraticCoefficient,
    candidateExposure,
    strategyNormalMixture,
    strategyNormalSigma,
  );
  let strategyMeanExposure = 0;
  let postActionCrossEntropy = 0;
  const values = Array.from(oracle.grid, (exposure, index) => {
    const oracleProbability = oracleProbabilities[index]!;
    const strategyProbability = strategyProbabilities[index]!;
    strategyMeanExposure += strategyProbability * exposure;
    if (oracleProbability > 0) {
      postActionCrossEntropy -= oracleProbability
        * Math.log(Math.max(Number.MIN_VALUE, strategyProbability));
    }
    return { exposure, oracleProbability, strategyProbability };
  });
  const strategyPolicy = strategyExposureTransitionStatistics(
    oracle.grid,
    oracle.currentGrid,
    rateBpsPerHour,
    intervalMs * oracle.holdingPeriodSteps,
    strategyTemperature,
    strategyQuadraticCoefficient,
    oracle.execution.friction,
    frictionFraction,
    candidateExposure,
    strategyNormalMixture,
    strategyNormalSigma,
  );
  const slope = strategyExposureLogSlope(
    rateBpsPerHour,
    intervalMs * oracle.holdingPeriodSteps,
    strategyTemperature,
  );
  const crossEntropy = strategyExposureTransitionCrossEntropy(
    oracle,
    candleIndex,
    rateBpsPerHour,
    intervalMs * oracle.holdingPeriodSteps,
    strategyTemperature,
    strategyQuadraticCoefficient,
    frictionFraction,
    candidateExposure,
    strategyNormalMixture,
    strategyNormalSigma,
  );
  return {
    time,
    candidateExposure,
    oracleMeanExposure: oracle.means[candleIndex]!,
    oracleModalExposure: oracle.modalExposures[candleIndex]!,
    oraclePathExposure: oracle.path.exposures[candleIndex]!,
    strategyMeanExposure,
    oraclePolicyMeanExposure: oracle.policyMeans[candleIndex]!,
    strategyPolicyMeanExposure: strategyPolicy.mean,
    strategyRateBpsHour: rateBpsPerHour,
    oracleTemperature: oracle.temperature,
    strategyTemperature,
    friction: oracle.execution.friction,
    frictionFraction,
    strategyLinearCoefficient: slope,
    strategyQuadraticVolatility,
    strategyQuadraticScale,
    strategyQuadraticCoefficient,
    strategyNormalMixture,
    strategyNormalSigma,
    oracleEntropy: oracle.entropies[candleIndex]!,
    oraclePolicyEntropy: oracle.policyEntropies[candleIndex]!,
    strategyPolicyEntropy: strategyPolicy.entropy,
    currentExposureMinimum: oracle.currentGrid[0]!,
    currentExposureMaximum: oracle.currentGrid[oracle.currentGrid.length - 1]!,
    currentExposureGridSize: oracle.currentGrid.length,
    opportunity: oracle.opportunities[candleIndex]!,
    averageRegret: oracle.averageRegrets[candleIndex]!,
    crossEntropy,
    postActionCrossEntropy,
    values,
  };
}

export function vwKamaScore(
  f1: number,
  exposureAgreement: number,
  signalCleanliness: number,
): number {
  return f1 * F1_WEIGHT
    + exposureAgreement * AGREEMENT_WEIGHT
    + signalCleanliness * CLEANLINESS_WEIGHT;
}

export function noiseSignalRatio(extra: number, matched: number): number | null {
  return matched > 0 ? extra / matched : extra > 0 ? null : 0;
}

export interface VwKamaTransitionMatch {
  candidateIndex: number;
  oracleIndex: number;
  lagMs: number;
  timingCredit: number;
}

export interface VwKamaTransitionAlignment {
  credit: number;
  matches: VwKamaTransitionMatch[];
}

interface AlignmentNode extends VwKamaTransitionMatch {
  previous: number;
  score: AlignmentScore;
}

interface AlignmentScore {
  credit: number;
  count: number;
  absoluteLagMs: number;
  node: number;
}

const EMPTY_ALIGNMENT: AlignmentScore = {
  credit: 0,
  count: 0,
  absoluteLagMs: 0,
  node: -1,
};

export function alignVwKamaTransitions(
  candidates: VwKamaTransition[],
  oracle: VwKamaTransition[],
  options: Pick<EvaluateVwKamaOptions, "matchWindowMs" | "timingHalfLifeMs">,
): VwKamaTransitionAlignment {
  return alignVwKamaTransitionsInternal(candidates, oracle, options, true);
}

function alignVwKamaTransitionsInternal(
  candidates: VwKamaTransition[],
  oracle: readonly VwKamaTransition[],
  options: Pick<EvaluateVwKamaOptions, "matchWindowMs" | "timingHalfLifeMs">,
  mutateOracle: boolean,
): VwKamaTransitionAlignment {
  if (options.matchWindowMs < 0 || options.timingHalfLifeMs <= 0) {
    throw new Error("VW-KAMA alignment window must be non-negative and half-life positive.");
  }
  if (!chronological(candidates) || !chronological(oracle)) {
    throw new Error("VW-KAMA transitions must be chronological.");
  }
  resetMatches(candidates);
  if (mutateOracle) resetMatches(oracle);
  const byState: Record<BacktestOracleState, Array<{ index: number; time: number }>> = {
    flat: [],
    long: [],
    short: [],
  };
  for (let index = 0; index < oracle.length; index += 1) {
    const transition = oracle[index]!;
    byState[transition.state].push({ index, time: transition.time });
  }
  const tree: Array<AlignmentScore | undefined> = Array(oracle.length + 1);
  const nodes: AlignmentNode[] = [];
  for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
    const candidate = candidates[candidateIndex]!;
    const targets = byState[candidate.state];
    const first = timeLowerBound(targets, candidate.time - options.matchWindowMs);
    const end = timeUpperBound(targets, candidate.time + options.matchWindowMs);
    const pending: Array<Omit<AlignmentNode, "previous" | "score"> & {
      previous: AlignmentScore;
      score: Omit<AlignmentScore, "node">;
    }> = [];
    for (let targetIndex = first; targetIndex < end; targetIndex += 1) {
      const target = targets[targetIndex]!;
      const lagMs = candidate.time - target.time;
      const timingCredit = Math.exp(-Math.LN2 * Math.abs(lagMs) / options.timingHalfLifeMs);
      const previous = queryAlignment(tree, target.index);
      const score = {
        credit: previous.credit + timingCredit,
        count: previous.count + 1,
        absoluteLagMs: previous.absoluteLagMs + Math.abs(lagMs),
      };
      if (compareAlignment(score, queryAlignment(tree, target.index + 1)) <= 0) continue;
      pending.push({ candidateIndex, oracleIndex: target.index, lagMs, timingCredit, previous, score });
    }
    for (const item of pending) {
      const node = nodes.length;
      const score = { ...item.score, node };
      nodes.push({
        candidateIndex: item.candidateIndex,
        oracleIndex: item.oracleIndex,
        lagMs: item.lagMs,
        timingCredit: item.timingCredit,
        previous: item.previous.node,
        score,
      });
      updateAlignment(tree, item.oracleIndex, score);
    }
  }
  const best = queryAlignment(tree, oracle.length);
  const matches: VwKamaTransitionMatch[] = [];
  for (let node = best.node; node >= 0; node = nodes[node]!.previous) {
    const { candidateIndex, oracleIndex, lagMs, timingCredit } = nodes[node]!;
    matches.push({ candidateIndex, oracleIndex, lagMs, timingCredit });
  }
  matches.reverse();
  for (const match of matches) {
    const candidate = candidates[match.candidateIndex]!;
    const target = oracle[match.oracleIndex]!;
    candidate.matchedTime = target.time;
    candidate.lagMs = match.lagMs;
    candidate.timingCredit = match.timingCredit;
    if (mutateOracle) {
      target.matchedTime = candidate.time;
      target.lagMs = -match.lagMs;
      target.timingCredit = match.timingCredit;
    }
  }
  return { credit: best.credit, matches };
}

function queryAlignment(tree: Array<AlignmentScore | undefined>, end: number): AlignmentScore {
  let best = EMPTY_ALIGNMENT;
  for (let index = end; index > 0; index -= index & -index) {
    const candidate = tree[index];
    if (candidate && compareAlignment(candidate, best) > 0) best = candidate;
  }
  return best;
}

function updateAlignment(
  tree: Array<AlignmentScore | undefined>,
  oracleIndex: number,
  score: AlignmentScore,
): void {
  for (let index = oracleIndex + 1; index < tree.length; index += index & -index) {
    const current = tree[index];
    if (!current || compareAlignment(score, current) > 0) tree[index] = score;
  }
}

function compareAlignment(
  left: Omit<AlignmentScore, "node">,
  right: Omit<AlignmentScore, "node">,
): number {
  const tolerance = Number.EPSILON * 32 * Math.max(1, Math.abs(left.credit), Math.abs(right.credit));
  if (left.credit > right.credit + tolerance) return 1;
  if (right.credit > left.credit + tolerance) return -1;
  if (left.count !== right.count) return left.count > right.count ? 1 : -1;
  if (left.absoluteLagMs !== right.absoluteLagMs) return left.absoluteLagMs < right.absoluteLagMs ? 1 : -1;
  return 0;
}

function resetMatches(transitions: readonly VwKamaTransition[]): void {
  for (const transition of transitions) {
    transition.matchedTime = null;
    transition.lagMs = null;
    transition.timingCredit = 0;
  }
}

function chronological(transitions: readonly VwKamaTransition[]): boolean {
  return transitions.every((transition, index) => index === 0 || transition.time >= transitions[index - 1]!.time);
}

function timeLowerBound(values: Array<{ time: number }>, time: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (values[middle]!.time < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function timeUpperBound(values: Array<{ time: number }>, time: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (values[middle]!.time <= time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function baseTransition(
  candle: TradingCandle,
  from: number,
  to: number,
  fromExposure = from,
  exposure = to,
  sizeFraction = Math.abs(to),
  quality = 1,
  acceleration = 0,
  overextension = 0,
  emaRate = 0,
  rsi = 50,
  dmi = 0,
  adx = 0,
  meanDistance = 0,
): VwKamaTransition {
  return {
    time: candle.closeTime,
    price: candle.close,
    side: to - from > 0 ? "buy" : "sell",
    fromState: stateName(from),
    state: stateName(to),
    fromExposure,
    exposure,
    sizeFraction,
    quality,
    acceleration,
    overextension,
    emaRate,
    rsi,
    dmi,
    adx,
    meanDistance,
    matchedTime: null,
    lagMs: null,
    timingCredit: 0,
  };
}

function prepareVwKamaOracleTransitions(
  candles: VwKamaCandleSeries,
  scoreStart: number,
  stateCodes: Uint8Array,
): VwKamaTransition[] {
  const transitions: VwKamaTransition[] = [];
  const buffers: [TradingCandle, TradingCandle] = [emptyCandle(), emptyCandle()];
  for (let index = Math.max(1, scoreStart); index < candles.length; index += 1) {
    const previous = exposureFromCode(stateCodes[index - 1] ?? 0);
    const current = exposureFromCode(stateCodes[index] ?? 0);
    if (current !== previous) {
      transitions.push(baseTransition(candleAt(candles, index, buffers[index % 2]!), previous, current));
    }
  }
  return transitions;
}

function confirmationQuality(
  direction: number,
  acceleration: number,
  overextension: number,
  emaRate: number,
  rsi: number,
  dmi: number,
  adx: number,
  parameters: VwKamaParameters,
): number {
  const mix = clamp01(parameters.confirmationMix ?? 0);
  if (mix === 0) return 1;
  const emaThreshold = Math.max(0, parameters.confirmationEmaThresholdBpsHour ?? 0);
  const emaEvidence = clamp(
    (direction * emaRate + emaThreshold) / Math.max(1, emaThreshold),
    -5,
    5,
  );
  const rsiEvidence = (direction * (rsi - 50) + Math.max(0, parameters.confirmationRsiThreshold ?? 0)) / 10;
  const adxThreshold = Math.max(0, parameters.confirmationAdxThreshold ?? 20);
  const adxTrust = clamp01((adx - adxThreshold) / Math.max(1, 100 - adxThreshold));
  const dmiEvidence = direction * dmi / 100 * adxTrust;
  const score = (parameters.confirmationBias ?? 0)
    + Math.max(0, parameters.confirmationAccelerationWeight ?? 1) * direction * acceleration
    - Math.max(0, parameters.confirmationDistanceWeight ?? 1) * direction * overextension
    + Math.max(0, parameters.confirmationEmaWeight ?? 0) * emaEvidence
    + Math.max(0, parameters.confirmationRsiWeight ?? 0) * rsiEvidence
    + Math.max(0, parameters.confirmationDmiWeight ?? 0) * dmiEvidence;
  const logistic = 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, score))));
  const gate = emaTrendAligned(direction, emaRate, parameters)
    ? 1
    : 1 - clamp01(parameters.confirmationEmaGateStrength ?? 0);
  return (1 - mix + mix * logistic) * gate;
}

function candidateState(
  rate: number,
  current: number,
  threshold: number,
  parameters: VwKamaParameters,
): number {
  return Math.sign(clampPeakValleyKamaRate(
    rate,
    current,
    parameters.deadbandMode === "flat" ? "deadband" : parameters.deadbandMode,
    parameters.hysteresisReleaseRatio ?? 0.25,
    threshold,
    threshold,
  ));
}

function emaTrendAligned(direction: number, emaRate: number, parameters: VwKamaParameters): boolean {
  return direction * emaRate >= -Math.max(0, parameters.confirmationEmaThresholdBpsHour ?? 0);
}

function signalFraction(direction: number, rate: number, parameters: VwKamaParameters): number {
  const buy = direction > 0;
  const maximum = clamp01(buy ? parameters.buyMaxFraction ?? 1 : parameters.sellMaxFraction ?? 1);
  const sigma = buy ? parameters.buySizingSigmaBpsHour : parameters.sellSizingSigmaBpsHour;
  if (sigma === undefined || !Number.isFinite(sigma)) return maximum;
  const normalized = rate / Math.max(Number.EPSILON, sigma);
  return maximum * Math.exp(-0.5 * normalized * normalized);
}

function markedExposure(direction: number, fraction: number, anchorPrice: number, price: number): number {
  if (direction === 0 || fraction <= 0) return 0;
  if (fraction >= 1) return Math.sign(direction);
  const movement = price / anchorPrice;
  const equity = direction > 0
    ? 1 - fraction + fraction * movement
    : 1 + fraction - fraction * movement;
  if (equity <= Number.EPSILON) return Math.sign(direction);
  return direction * fraction * movement / equity;
}

function candidateAgreementValue(
  mode: VwKamaAgreementMode,
  direction: number,
  fraction: number,
  anchorPrice: number,
  price: number,
): number {
  return mode === "confidence"
    ? direction * fraction
    : markedExposure(direction, fraction, anchorPrice, price);
}

function candidateValueTarget(
  direction: number,
  fraction: number,
  minimumExposure: number,
  maximumExposure: number,
): number {
  if (direction > 0) return fraction * Math.max(0, maximumExposure);
  if (direction < 0) return fraction * Math.min(0, minimumExposure);
  return 0;
}

function candidateValueExposure(
  direction: number,
  fraction: number,
  anchorPrice: number,
  price: number,
  minimumExposure: number,
  maximumExposure: number,
): number {
  const target = candidateValueTarget(
    direction,
    fraction,
    minimumExposure,
    maximumExposure,
  );
  if (target === 0) return 0;
  const movement = price / anchorPrice;
  const equity = 1 + target * (movement - 1);
  if (equity <= Number.EPSILON) return Math.sign(target) * Number.MAX_VALUE;
  return target * movement / equity;
}

function agreementCredit(
  mode: VwKamaAgreementMode,
  direction: number,
  candidate: number,
  oracle: number,
): number {
  if (mode === "confidence") {
    const confidence = clamp01(Math.abs(candidate));
    if (direction === oracle) return direction === 0 ? 1 : confidence;
    return oracle === 0 ? 1 - confidence : 0;
  }
  if (oracle > 0) return clamp01(candidate);
  if (oracle < 0) return clamp01(-candidate);
  return 1 - clamp01(Math.abs(candidate));
}

function statePoint(candle: TradingCandle, from: number, to: number): BacktestOraclePoint {
  const fromState = stateName(from);
  const state = stateName(to);
  return {
    time: candle.closeTime,
    price: candle.close,
    fromState,
    state,
    action: fromState === state ? "hold" : fromState === "flat" ? "open" : state === "flat" ? "close" : "switch",
  };
}

function slicePath(points: BacktestOraclePoint[], startTime: number): BacktestOraclePoint[] {
  const index = points.findIndex((point) => point.time >= startTime);
  if (index <= 0) return index < 0 ? points.slice(-1) : points.slice();
  return points.slice(index - 1);
}

function exposureFromCode(code: number): -1 | 0 | 1 {
  return code === 1 ? 1 : code === 2 ? -1 : 0;
}

function stateName(exposure: number): BacktestOracleState {
  return exposure > 0 ? "long" : exposure < 0 ? "short" : "flat";
}

function samples(durationMs: number, intervalMs: number): number {
  return Math.max(1, Math.round(durationMs / intervalMs));
}

function lowerBound(candles: VwKamaCandleSeries, time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (openTimeAt(candles, middle) < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function isCandleColumns(candles: VwKamaCandleSeries): candles is VwKamaCandleColumns {
  return !Array.isArray(candles);
}

function openTimeAt(candles: VwKamaCandleSeries, index: number): number {
  return isCandleColumns(candles) ? candles.openTime[index]! : candles[index]!.openTime;
}

function closeTimeAt(candles: VwKamaCandleSeries, index: number): number {
  return isCandleColumns(candles) ? candles.closeTime[index]! : candles[index]!.closeTime;
}

function closeAt(candles: VwKamaCandleSeries, index: number): number {
  return isCandleColumns(candles) ? candles.close[index]! : candles[index]!.close;
}

function candleAt(
  candles: VwKamaCandleSeries,
  index: number,
  buffer: TradingCandle,
): TradingCandle {
  if (!isCandleColumns(candles)) return candles[index]!;
  buffer.openTime = candles.openTime[index]!;
  buffer.closeTime = candles.closeTime[index]!;
  buffer.open = candles.open[index]!;
  buffer.high = candles.high[index]!;
  buffer.low = candles.low[index]!;
  buffer.close = candles.close[index]!;
  buffer.volume = candles.volume[index]!;
  return buffer;
}

function emptyCandle(): TradingCandle {
  return { openTime: 0, closeTime: 0, open: 0, high: 0, low: 0, close: 0, volume: 0 };
}

function validate(options: EvaluateVwKamaOptions): void {
  if (!["flat", "hold", "hysteresis"].includes(options.parameters.deadbandMode)) {
    throw new Error("VW-KAMA deadband mode must be flat, hold, or hysteresis.");
  }
  if (options.parameters.rateMode !== undefined
    && options.parameters.rateMode !== "relative"
    && options.parameters.rateMode !== "log") {
    throw new Error("VW-KAMA rate mode must be relative or log.");
  }
  if (options.parameters.thresholdNoiseResponse !== undefined
    && options.parameters.thresholdNoiseResponse !== "proportional"
    && options.parameters.thresholdNoiseResponse !== "inverse") {
    throw new Error("VW-KAMA threshold noise response must be proportional or inverse.");
  }
  if (options.parameters.agreementMode !== undefined
    && options.parameters.agreementMode !== "sizing"
    && options.parameters.agreementMode !== "confidence") {
    throw new Error("VW-KAMA agreement mode must be sizing or confidence.");
  }
  const positive = [
    options.intervalMs,
    options.parameters.efficiencyMs,
    ...(options.parameters.efficiencyVolumePower
      ? [options.parameters.efficiencyVolumeEmaMs ?? options.parameters.volumeMs]
      : []),
    options.parameters.fastMs,
    options.parameters.slowMs,
    options.parameters.power,
    options.parameters.volumeMs,
    options.parameters.volumeCap,
    options.parameters.rateEmaMs ?? options.intervalMs,
    options.parameters.strategyTemperature ?? 0.001,
    options.parameters.strategyQuadraticVolatilityMs ?? HOUR_MS,
    options.timingHalfLifeMs,
    options.warmupMultiple,
  ];
  if (positive.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("VW-KAMA durations and powers must be positive.");
  }
  const nonNegative = [
    options.parameters.volumePower,
    options.parameters.efficiencyVolumePower ?? 0,
    options.parameters.deadbandBpsHour,
    options.parameters.thresholdNoiseMultiplier ?? 0,
    options.parameters.thresholdInverseMaxBpsHour ?? 0,
    options.parameters.signalFrictionFraction ?? 1,
    options.parameters.strategyQuadraticScale ?? 0,
    options.parameters.buyMaxFraction ?? 1,
    options.parameters.sellMaxFraction ?? 1,
    options.parameters.confirmationAccelerationWeight ?? 1,
    options.parameters.confirmationDistanceWeight ?? 1,
    options.parameters.confirmationEmaThresholdBpsHour ?? 0,
    options.parameters.confirmationEmaWeight ?? 0,
    options.parameters.confirmationRsiThreshold ?? 0,
    options.parameters.confirmationRsiWeight ?? 0,
    options.parameters.confirmationDmiWeight ?? 0,
    options.parameters.confirmationAdxThreshold ?? 20,
    options.parameters.meanReversionSuppressionThreshold ?? 0,
    options.parameters.meanReversionReversalThreshold ?? 0,
    options.oracleFriction,
    options.matchWindowMs,
  ];
  if (nonNegative.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("VW-KAMA thresholds and friction cannot be negative.");
  }
  if (options.parameters.thresholdNoiseResponse === "inverse"
    && (options.parameters.thresholdInverseMaxBpsHour ?? 0) > 0
    && (!Number.isFinite(options.parameters.thresholdInverseNoiseScaleBpsHour)
      || (options.parameters.thresholdInverseNoiseScaleBpsHour ?? 0) <= 0)) {
    throw new Error("VW-KAMA inverse threshold noise scale must be positive.");
  }
  const sigmas = [
    options.parameters.buySizingSigmaBpsHour,
    options.parameters.sellSizingSigmaBpsHour,
  ].filter((value): value is number => value !== undefined);
  if (sigmas.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("VW-KAMA sizing sigmas must be positive.");
  }
  if ((options.parameters.buyMaxFraction ?? 1) > 1 || (options.parameters.sellMaxFraction ?? 1) > 1) {
    throw new Error("VW-KAMA sizing fractions cannot exceed one.");
  }
  if ((options.parameters.signalFrictionFraction ?? 1) > 1) {
    throw new Error("VW-KAMA signal friction fraction cannot exceed one.");
  }
  const confirmationFractions = [
    options.parameters.confirmationMix ?? 0,
    options.parameters.confirmationMinQuality ?? 0,
    options.parameters.hysteresisReleaseRatio ?? 0.25,
    options.parameters.confirmationEmaGateStrength ?? 0,
  ];
  if (confirmationFractions.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("VW-KAMA confirmation mix and minimum quality must be between zero and one.");
  }
  if (!Number.isFinite(options.parameters.confirmationBias ?? 0)) {
    throw new Error("VW-KAMA confirmation bias must be finite.");
  }
  if ((options.parameters.confirmationRsiThreshold ?? 0) > 50
    || (options.parameters.confirmationAdxThreshold ?? 20) > 100) {
    throw new Error("VW-KAMA RSI and ADX thresholds exceed their oscillator ranges.");
  }
  if ((options.parameters.confirmationMix ?? 0) > 0) {
    const lookbacks = [
      options.parameters.confirmationAccelerationLookbackMs,
      options.parameters.confirmationDistanceLookbackMs,
    ];
    if (lookbacks.some((value) => !Number.isFinite(value) || (value ?? 0) <= 0)) {
      throw new Error("VW-KAMA confirmation lookbacks must be positive.");
    }
  }
  const optionalLookbacks = [
    [(options.parameters.confirmationEmaWeight ?? 0) > 0
      || (options.parameters.confirmationEmaGateStrength ?? 0) > 0, options.parameters.confirmationEmaMs],
    [(options.parameters.confirmationRsiWeight ?? 0) > 0, options.parameters.confirmationRsiMs],
    [(options.parameters.confirmationDmiWeight ?? 0) > 0, options.parameters.confirmationDmiMs],
    [meanReversionParametersEnabled(options.parameters),
      options.parameters.meanReversionEfficiencyMs ?? options.parameters.efficiencyMs],
    [meanReversionParametersEnabled(options.parameters),
      options.parameters.meanReversionFastMs ?? options.parameters.fastMs],
    [meanReversionParametersEnabled(options.parameters),
      options.parameters.meanReversionSlowMs ?? options.parameters.slowMs],
    [meanReversionParametersEnabled(options.parameters),
      options.parameters.meanReversionVolatilityMs ?? options.parameters.slowMs],
  ] as const;
  if (optionalLookbacks.some(([enabled, value]) => enabled
    && (!Number.isFinite(value) || (value ?? 0) <= 0))) {
    throw new Error("VW-KAMA enabled indicator lookbacks must be positive.");
  }
  if (meanReversionParametersEnabled(options.parameters)) {
    const suppression = options.parameters.meanReversionSuppressionThreshold ?? 0;
    const reversal = options.parameters.meanReversionReversalThreshold ?? 0;
    const fast = options.parameters.meanReversionFastMs ?? options.parameters.fastMs;
    const slow = options.parameters.meanReversionSlowMs ?? options.parameters.slowMs;
    if (!Number.isFinite(suppression) || suppression <= 0 || suppression > reversal) {
      throw new Error("VW-KAMA mean-reversion suppression threshold must be positive and at most the reversal threshold.");
    }
    if (fast > slow) {
      throw new Error("VW-KAMA mean-reversion fast duration cannot exceed its slow duration.");
    }
  }
  if (thresholdNoiseEnabled(options.parameters)
    && (!Number.isFinite(options.parameters.thresholdLookbackMs)
      || (options.parameters.thresholdLookbackMs ?? 0) <= 0)) {
    throw new Error("VW-KAMA adaptive threshold lookback must be positive.");
  }
}

function thresholdNoiseEnabled(parameters: VwKamaParameters): boolean {
  return parameters.thresholdNoiseResponse === "inverse"
    ? (parameters.thresholdInverseMaxBpsHour ?? 0) > 0
    : (parameters.thresholdNoiseMultiplier ?? 0) > 0;
}

function meanReversionParametersEnabled(parameters: VwKamaParameters): boolean {
  return (parameters.meanReversionReversalThreshold ?? 0) > 0;
}

function thresholdNoiseAdjustment(noise: number, parameters: VwKamaParameters): number {
  return peakValleyKamaThresholdAdjustment(noise, {
    response: parameters.thresholdNoiseResponse === "inverse" ? "inverse" : "proportional",
    multiplier: parameters.thresholdNoiseMultiplier ?? 0,
    inverseMax: parameters.thresholdInverseMaxBpsHour ?? 0,
    inverseNoiseScale: parameters.thresholdInverseNoiseScaleBpsHour ?? 1,
  });
}

class DmiAdx {
  private previous: TradingCandle | null = null;
  private trueRange = 0;
  private positiveMovement = 0;
  private negativeMovement = 0;
  private adx = 0;
  private readonly alpha: number;

  constructor(period: number) {
    this.alpha = 1 / Math.max(1, period);
  }

  update(candle: TradingCandle): { direction: number; adx: number } {
    const previous = this.previous;
    this.previous = candle;
    if (!previous) return { direction: 0, adx: 0 };
    const up = candle.high - previous.high;
    const down = previous.low - candle.low;
    const trueRange = Math.max(
      candle.high - candle.low,
      Math.abs(candle.high - previous.close),
      Math.abs(candle.low - previous.close),
    );
    this.trueRange = smooth(this.trueRange, trueRange, this.alpha);
    this.positiveMovement = smooth(this.positiveMovement, up > down && up > 0 ? up : 0, this.alpha);
    this.negativeMovement = smooth(this.negativeMovement, down > up && down > 0 ? down : 0, this.alpha);
    const positive = this.trueRange > 0 ? 100 * this.positiveMovement / this.trueRange : 0;
    const negative = this.trueRange > 0 ? 100 * this.negativeMovement / this.trueRange : 0;
    const sum = positive + negative;
    const dx = sum > 0 ? 100 * Math.abs(positive - negative) / sum : 0;
    this.adx = smooth(this.adx, dx, this.alpha);
    return { direction: positive - negative, adx: this.adx };
  }
}

function smooth(previous: number, value: number, alpha: number): number {
  return previous === 0 ? value : previous + alpha * (value - previous);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function eventRatio(value: number, total: number, opposite: number): number {
  return total > 0 ? value / total : opposite === 0 ? 1 : 0;
}

function harmonic(left: number, right: number): number {
  return left + right > 0 ? 2 * left * right / (left + right) : 0;
}

function percentile(values: number[], quantile: number): number | null {
  if (values.length === 0) return null;
  const sorted = values.slice().sort((left, right) => left - right);
  const index = (sorted.length - 1) * quantile;
  const lower = Math.floor(index);
  const fraction = index - lower;
  return sorted[lower]! * (1 - fraction) + (sorted[lower + 1] ?? sorted[lower]!) * fraction;
}
