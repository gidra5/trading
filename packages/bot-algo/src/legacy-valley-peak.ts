import type {
  BotSignal,
  Candle,
  LegacyDerivativeClampMode,
  LegacyExtremaSignalTiming,
  LegacyMovingAverageType,
  LegacyValleyPeakCheckDebug,
  LegacyValleyPeakConfig,
  LegacyValleyPeakDebugSnapshot,
  LegacyValleyPeakMemory,
  LegacyMarketStateDebug,
  PositionLotSide,
  RollingCandleRangeBucket,
  RollingCandleRangeMemory,
  RollingCandleRangeMaxCandidate,
  RollingCandleRangePoint,
  RollingKamaMemory,
  RollingPriceRangeBucket,
  RollingPriceRangeMemory,
  RollingPriceRangePoint,
  RollingPriceRangeWindow,
  RollingAverageMemory,
  RollingAveragePoint,
} from "./types.js";

export interface LegacyValleyPeakInput {
  eventTime: number;
  price: number;
  feeRate: number;
  buyingPowerQuote: number;
  shortSellingPowerQuote?: number;
  baseFree: number;
  shortBaseFree?: number;
  sourceCandle?: Candle;
}

export type LegacyValleyPeakEntrySignal =
  | { signal: "hold" }
  | { signal: "buy"; reason: string; quoteSize: number }
  | { signal: "sell"; reason: string; quoteSize: number };

export type LegacyValleyPeakExitSignal =
  | { signal: "hold" }
  | { signal: "buy"; reason: string; coverQuantity: number }
  | { signal: "sell"; reason: string; quantity: number };

export interface LegacyValleyPeakDecision {
  entrySignal: LegacyValleyPeakEntrySignal;
  exitSignal: LegacyValleyPeakExitSignal;
}

export function legacyValleyPeakDecisionSignal(
  decision: LegacyValleyPeakDecision,
): BotSignal {
  return decision.exitSignal.signal !== "hold"
    ? decision.exitSignal.signal
    : decision.entrySignal.signal;
}

export function legacyValleyPeakDecisionReason(
  decision: LegacyValleyPeakDecision,
): string | undefined {
  const signal = legacyValleyPeakDecisionSignal(decision);
  if (decision.exitSignal.signal === signal && decision.exitSignal.signal !== "hold") {
    return decision.exitSignal.reason;
  }
  if (decision.entrySignal.signal === signal && decision.entrySignal.signal !== "hold") {
    return decision.entrySignal.reason;
  }
  return undefined;
}

export const legacyValleyPeakStrictSymmetricConfig: LegacyValleyPeakConfig = {
  averagingRangesSec: [1, 60, 600, 1800, 3600, 3600 * 4, 3600 * 12],
  movingAverageType: "sma",
  rateRatios: [0.5, 0.5, 0.1, 0.05, 0.01, 0.01, 0.001],
  relativeRateEnabled: true,
  derivativeSource: "price",
  derivativeClampMode: "deadband",
  derivativeClampInnerThresholdRatio: 0,
  kamaErLen: 20,
  kamaFastLen: 5,
  kamaSlowLen: 50,
  kamaPower: 1,
  rateThresholdsLow: [0.25, 0.25, 0.25, 0.25, 0.15, 0.05, 0.05],
  rateThresholdsHigh: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
  buyDataIndex: 1,
  sellDataIndex: 1,
  buyConfirmationOffsets: [1, 2],
  sellConfirmationOffsets: [1, 2],
  buyExitConfirmationOffsets: [1, 2],
  sellExitConfirmationOffsets: [1, 2],
  buyEntrySignalTiming: "start",
  sellEntrySignalTiming: "start",
  buyExitSignalTiming: "start",
  sellExitSignalTiming: "start",
  saturationSec: 3600,
  buySpendRate: 1,
  sellAmountRate: 1,
  sigmaMode: "trend",
  buySigma: 0.3,
  sellSigma: 0.1,
  trendSigmaA: 1,
  trendSigmaSellB1: 1,
  trendSigmaBuyB2: 1,
  trendSigmaWindowSec: 3600,
  sigmoidSigmaLow: 0.05,
  sigmoidSigmaHigh: 0.3,
  minTradeQuote: 25,
  maxTradeQuote: 50_000,
  longSideEnabled: true,
  shortSideEnabled: true,
  exitGridEnabled: true,
  exitGridMarketEntry: true,
  exitGridOrderCount: 200,
  exitGridMaxStepPct: 0.006,
  exitGridPriceDistribution: "uniform",
  exitGridSizeDistribution: "geometric",
  exitGridSellFraction: 0.2,
  exitGridMinProfitBps: 20,
  exitGridResetBps: 10,
  exitGridPositionMode: "per-lot",
  exitGridResetMode: "filled-grid",
  anticipatoryConfirmationMaxMisses: 0,
  anticipatoryConfirmationWindowSec: 30 * 60,
  anticipatoryConfirmationLookaheadFraction: 0.1,
  rangeLeverageEnabled: true,
  leverageLongTermRangeWindow: "1y",
  leverageRangeEdgeFraction: 0.2,
  leverageLongTermRangePaddingPct: 20,
};

export const legacyValleyPeakAsymmetricShortFavoringConfig: LegacyValleyPeakConfig = {
  ...legacyValleyPeakStrictSymmetricConfig,
  sellDataIndex: 0,
  sellConfirmationOffsets: [6],
};

export const legacyValleyPeakReferenceConfigs = {
  strictSymmetric035Anchor: legacyValleyPeakStrictSymmetricConfig,
  asymmetricShortFavoring: legacyValleyPeakAsymmetricShortFavoringConfig,
};

export const defaultLegacyValleyPeakConfig = legacyValleyPeakStrictSymmetricConfig;

const ROLLING_AVERAGE_COMPACT_EXPIRED = 2048;
const CANDLE_RANGE_WINDOW_SIZE = 1_000;
const PRICE_RANGE_BUCKET_SEC = 5 * 60;
const PRICE_RANGE_WINDOWS: { window: RollingPriceRangeWindow; windowSec: number }[] = [
  { window: "1y", windowSec: 365 * 24 * 60 * 60 },
  { window: "3m", windowSec: 90 * 24 * 60 * 60 },
  { window: "2w", windowSec: 14 * 24 * 60 * 60 },
];
const TREND_SIGMA_WINDOW_SEC = 3600;
const ANTICIPATORY_CONFIRMATION_WINDOW_SEC = 30 * 60;
const ANTICIPATORY_CONFIRMATION_LOOKAHEAD_FRACTION = 0.1;
const MIN_ANTICIPATORY_CONFIRMATION_SAMPLES = 8;
const MIN_RELATIVE_SIGMA = 0.00000001;
const MIN_ABSOLUTE_SIGMA = 0.000001;
const MAX_TREND_SIGMA_EXPONENT = 50;
const MAX_EXIT_GRID_ORDER_COUNT = 5_000;
const BTC_SCALE_REFERENCE_PRICE = 100_000;
const MAX_REASONABLE_RELATIVE_RATE_PER_SEC = 0.01;

export function createLegacyValleyPeakConfig(
  overrides: Partial<LegacyValleyPeakConfig> = {},
): LegacyValleyPeakConfig {
  const config: LegacyValleyPeakConfig = {
    ...defaultLegacyValleyPeakConfig,
    ...overrides,
    averagingRangesSec:
      overrides.averagingRangesSec ?? defaultLegacyValleyPeakConfig.averagingRangesSec,
    rateRatios: overrides.rateRatios ?? defaultLegacyValleyPeakConfig.rateRatios,
    rateThresholdsLow:
      overrides.rateThresholdsLow ?? defaultLegacyValleyPeakConfig.rateThresholdsLow,
    rateThresholdsHigh:
      overrides.rateThresholdsHigh ?? defaultLegacyValleyPeakConfig.rateThresholdsHigh,
    buyConfirmationOffsets:
      overrides.buyConfirmationOffsets ??
      defaultLegacyValleyPeakConfig.buyConfirmationOffsets,
    sellConfirmationOffsets:
      overrides.sellConfirmationOffsets ??
      defaultLegacyValleyPeakConfig.sellConfirmationOffsets,
    buyExitConfirmationOffsets:
      overrides.buyExitConfirmationOffsets ??
      defaultLegacyValleyPeakConfig.buyExitConfirmationOffsets,
    sellExitConfirmationOffsets:
      overrides.sellExitConfirmationOffsets ??
      defaultLegacyValleyPeakConfig.sellExitConfirmationOffsets,
  };

  const rangeCount = Math.max(1, config.averagingRangesSec.length);
  config.movingAverageType = normalizeMovingAverageType(config.movingAverageType);
  config.rateRatios = padNumbers(config.rateRatios, rangeCount, 0.5);
  config.relativeRateEnabled = config.relativeRateEnabled === true;
  config.derivativeSource = config.derivativeSource === "kama" ? "kama" : "price";
  config.derivativeClampMode =
    config.derivativeClampMode === "hysteresis" ? "hysteresis" : "deadband";
  config.buyEntrySignalTiming = normalizeExtremaSignalTiming(
    config.buyEntrySignalTiming,
  );
  config.sellEntrySignalTiming = normalizeExtremaSignalTiming(
    config.sellEntrySignalTiming,
  );
  config.buyExitSignalTiming = normalizeExtremaSignalTiming(
    config.buyExitSignalTiming,
  );
  config.sellExitSignalTiming = normalizeExtremaSignalTiming(
    config.sellExitSignalTiming,
  );
  config.derivativeClampInnerThresholdRatio = clamp(
    Number.isFinite(config.derivativeClampInnerThresholdRatio)
      ? config.derivativeClampInnerThresholdRatio
      : defaultLegacyValleyPeakConfig.derivativeClampInnerThresholdRatio,
    0,
    1,
  );
  config.kamaErLen = clampInt(
    config.kamaErLen,
    1,
    Number.MAX_SAFE_INTEGER,
  );
  config.kamaFastLen = clampInt(
    config.kamaFastLen,
    1,
    Number.MAX_SAFE_INTEGER,
  );
  config.kamaSlowLen = clampInt(
    config.kamaSlowLen,
    1,
    Number.MAX_SAFE_INTEGER,
  );
  config.kamaPower = Math.max(
    0.1,
    Number.isFinite(config.kamaPower)
      ? config.kamaPower
      : defaultLegacyValleyPeakConfig.kamaPower,
  );
  config.rateThresholdsLow = config.relativeRateEnabled
    ? normalizeRelativeRates(config.rateThresholdsLow, rangeCount, 0.0000025)
    : padNumbers(config.rateThresholdsLow, rangeCount, 0.25);
  config.rateThresholdsHigh = config.relativeRateEnabled
    ? normalizeRelativeRates(config.rateThresholdsHigh, rangeCount, 0.0000025)
    : padNumbers(config.rateThresholdsHigh, rangeCount, 0.25);
  config.buyDataIndex = clampInt(config.buyDataIndex, 0, rangeCount - 1);
  config.sellDataIndex = clampInt(config.sellDataIndex, 0, rangeCount - 1);
  config.buyConfirmationOffsets = config.buyConfirmationOffsets.map((offset) =>
    Math.max(0, Math.round(offset)),
  );
  config.sellConfirmationOffsets = config.sellConfirmationOffsets.map((offset) =>
    Math.max(0, Math.round(offset)),
  );
  config.buyExitConfirmationOffsets = config.buyExitConfirmationOffsets.map((offset) =>
    Math.max(0, Math.round(offset)),
  );
  config.sellExitConfirmationOffsets = config.sellExitConfirmationOffsets.map((offset) =>
    Math.max(0, Math.round(offset)),
  );
  config.saturationSec = Math.max(0, config.saturationSec);
  config.buySpendRate = Math.max(0, config.buySpendRate);
  config.sellAmountRate = Math.max(0, config.sellAmountRate);
  if (
    config.sigmaMode !== "static" &&
    config.sigmaMode !== "trend" &&
    config.sigmaMode !== "sigmoid-trend"
  ) {
    config.sigmaMode = defaultLegacyValleyPeakConfig.sigmaMode;
  }
  config.buySigma = Math.max(
    MIN_ABSOLUTE_SIGMA,
    Number.isFinite(config.buySigma) ? config.buySigma : defaultLegacyValleyPeakConfig.buySigma,
  );
  config.sellSigma = Math.max(
    MIN_ABSOLUTE_SIGMA,
    Number.isFinite(config.sellSigma) ? config.sellSigma : defaultLegacyValleyPeakConfig.sellSigma,
  );
  const trendSigmaA = Number.isFinite(config.trendSigmaA)
    ? config.trendSigmaA
    : defaultLegacyValleyPeakConfig.trendSigmaA;
  config.trendSigmaA = Math.max(MIN_ABSOLUTE_SIGMA, trendSigmaA);
  config.trendSigmaSellB1 = Number.isFinite(config.trendSigmaSellB1)
    ? Math.max(0, config.trendSigmaSellB1)
    : defaultLegacyValleyPeakConfig.trendSigmaSellB1;
  config.trendSigmaBuyB2 = Number.isFinite(config.trendSigmaBuyB2)
    ? Math.max(0, config.trendSigmaBuyB2)
    : defaultLegacyValleyPeakConfig.trendSigmaBuyB2;
  config.trendSigmaWindowSec = isPositiveFinite(config.trendSigmaWindowSec)
    ? config.trendSigmaWindowSec
    : defaultLegacyValleyPeakConfig.trendSigmaWindowSec;
  config.sigmoidSigmaLow = Math.max(
    MIN_ABSOLUTE_SIGMA,
    Number.isFinite(config.sigmoidSigmaLow)
      ? config.sigmoidSigmaLow
      : defaultLegacyValleyPeakConfig.sigmoidSigmaLow,
  );
  config.sigmoidSigmaHigh = Math.max(
    config.sigmoidSigmaLow,
    Number.isFinite(config.sigmoidSigmaHigh)
      ? config.sigmoidSigmaHigh
      : defaultLegacyValleyPeakConfig.sigmoidSigmaHigh,
  );
  config.minTradeQuote = Math.max(0, config.minTradeQuote);
  config.maxTradeQuote = Math.max(config.minTradeQuote, config.maxTradeQuote);
  config.longSideEnabled = config.longSideEnabled !== false;
  config.shortSideEnabled = config.shortSideEnabled !== false;
  config.exitGridOrderCount = clampInt(
    config.exitGridOrderCount,
    1,
    MAX_EXIT_GRID_ORDER_COUNT,
  );
  config.exitGridMaxStepPct = Math.max(0, config.exitGridMaxStepPct);
  if (
    config.exitGridPriceDistribution !== "uniform" &&
    config.exitGridPriceDistribution !== "geometric"
  ) {
    config.exitGridPriceDistribution =
      defaultLegacyValleyPeakConfig.exitGridPriceDistribution;
  }
  if (
    config.exitGridSizeDistribution !== "geometric" &&
    config.exitGridSizeDistribution !== "linear" &&
    config.exitGridSizeDistribution !== "constant"
  ) {
    config.exitGridSizeDistribution =
      defaultLegacyValleyPeakConfig.exitGridSizeDistribution;
  }
  config.exitGridSellFraction = clamp(config.exitGridSellFraction, 0.01, 1);
  config.exitGridMinProfitBps = Math.max(0, config.exitGridMinProfitBps);
  config.exitGridResetBps = Math.max(0, config.exitGridResetBps);
  config.rangeLeverageEnabled = config.rangeLeverageEnabled !== false;
  if (!PRICE_RANGE_WINDOWS.some((range) => range.window === config.leverageLongTermRangeWindow)) {
    config.leverageLongTermRangeWindow =
      defaultLegacyValleyPeakConfig.leverageLongTermRangeWindow;
  }
  config.leverageRangeEdgeFraction = clamp(config.leverageRangeEdgeFraction, 0, 0.49);
  config.leverageLongTermRangePaddingPct = clamp(
    isPositiveFinite(config.leverageLongTermRangePaddingPct)
      ? config.leverageLongTermRangePaddingPct
      : defaultLegacyValleyPeakConfig.leverageLongTermRangePaddingPct,
    0,
    100,
  );
  if (
    config.exitGridPositionMode !== "aggregate" &&
    config.exitGridPositionMode !== "per-lot"
  ) {
    config.exitGridPositionMode = defaultLegacyValleyPeakConfig.exitGridPositionMode;
  }
  if (
    config.exitGridResetMode !== "higher-peak" &&
    config.exitGridResetMode !== "filled-grid"
  ) {
    config.exitGridResetMode = defaultLegacyValleyPeakConfig.exitGridResetMode;
  }
  delete (config as { anticipatoryConfirmationEnabled?: unknown }).anticipatoryConfirmationEnabled;
  config.anticipatoryConfirmationMaxMisses = clampInt(
    Number.isFinite(config.anticipatoryConfirmationMaxMisses)
      ? config.anticipatoryConfirmationMaxMisses
      : defaultLegacyValleyPeakConfig.anticipatoryConfirmationMaxMisses,
    0,
    1,
  );
  config.anticipatoryConfirmationWindowSec = isPositiveFinite(
    config.anticipatoryConfirmationWindowSec,
  )
    ? Math.max(60, config.anticipatoryConfirmationWindowSec)
    : ANTICIPATORY_CONFIRMATION_WINDOW_SEC;
  config.anticipatoryConfirmationLookaheadFraction = clamp(
    Number.isFinite(config.anticipatoryConfirmationLookaheadFraction)
      ? config.anticipatoryConfirmationLookaheadFraction
      : ANTICIPATORY_CONFIRMATION_LOOKAHEAD_FRACTION,
    0.01,
    1,
  );

  return config;
}

export function legacyValleyPeakHistoricalWarmupSec(
  config: LegacyValleyPeakConfig,
): number {
  return Math.max(
    legacyValleyPeakSignalWarmupSec(config),
    legacyValleyPeakPriceRangeWarmupSec(),
  );
}

export function legacyValleyPeakSignalWarmupSec(
  config: LegacyValleyPeakConfig,
): number {
  return Math.max(
    ...config.averagingRangesSec,
    config.saturationSec,
    hasAnticipatoryConfirmationMissBudget(config)
      ? config.anticipatoryConfirmationWindowSec
      : 0,
  );
}

export function legacyValleyPeakPriceRangeWarmupSec(): number {
  return Math.max(...PRICE_RANGE_WINDOWS.map((range) => range.windowSec));
}

export function legacyValleyPeakObservedSignalWarmupSec(
  memory: LegacyValleyPeakMemory | undefined,
): number {
  if (!memory?.buyAverages?.length) {
    return 0;
  }

  const latestTsSec = Math.max(
    0,
    ...memory.buyAverages
      .map((average) => average.timestamps.at(-1))
      .filter((value): value is number =>
        value !== undefined && isNonNegativeFinite(value),
      ),
  );
  const startedAtSec = memory.startedAt === undefined ? undefined : memory.startedAt / 1000;
  const replaySpanSec =
    startedAtSec !== undefined && isNonNegativeFinite(startedAtSec)
      ? Math.max(0, latestTsSec - startedAtSec)
      : 0;

  return Math.max(
    0,
    replaySpanSec,
    ...memory.buyAverages.map((average) => rollingAverageSampleSpanSec(average)),
  );
}

export function legacyValleyPeakObservedWarmupSec(
  memory: LegacyValleyPeakMemory | undefined,
): number {
  if (!memory?.priceRanges?.length) {
    return 0;
  }

  return Math.max(
    0,
    ...memory.priceRanges.map((range) => priceRangeSampleStats(range).spanMs / 1000),
  );
}

export function legacyValleyPeakObservedPriceRangeWarmupRatio(
  memory: LegacyValleyPeakMemory | undefined,
): number {
  if (!memory?.priceRanges?.length) {
    return 0;
  }

  return Math.min(
    1,
    ...PRICE_RANGE_WINDOWS.map(({ window, windowSec }) => {
      const range = memory.priceRanges.find((item) => item.window === window);
      return windowSec > 0 ? priceRangeSampleStats(range).spanMs / 1000 / windowSec : 0;
    }),
  );
}

export function warmupLegacyValleyPeakPriceRanges(
  memory: LegacyValleyPeakMemory,
  candles: readonly Candle[],
): number {
  const priceRanges = ensurePriceRangeMemories(memory);
  let processed = 0;
  for (const candle of candles) {
    if (
      !candle.closed ||
      !isPositiveFinite(candle.low) ||
      !isPositiveFinite(candle.high) ||
      candle.high < candle.low ||
      !isNonNegativeFinite(candle.openTime) ||
      !isNonNegativeFinite(candle.closeTime)
    ) {
      continue;
    }

    const sampleTsSec = candle.openTime / 1000;
    const nowSec = candle.closeTime / 1000;
    for (const priceRange of priceRanges) {
      observePriceRangeSample(priceRange, sampleTsSec, candle.low, candle.high, nowSec);
    }
    processed += 1;
  }
  return processed;
}

export function createLegacyValleyPeakMemory(
  config: LegacyValleyPeakConfig,
): LegacyValleyPeakMemory {
  const averages = config.averagingRangesSec.map(createRollingAverageMemory);
  return {
    kama: createRollingKamaMemory(),
    kamaBuySignal: createRollingAverageMemory(),
    kamaSellSignal: createRollingAverageMemory(),
    buyAverages: averages,
    sellAverages: averages,
    candleRanges: config.averagingRangesSec.map(createRollingCandleRangeMemory),
    priceRanges: createRollingPriceRangeMemories(),
  };
}

export function normalizeLegacyValleyPeakMemory(
  memory: LegacyValleyPeakMemory | undefined,
  config: LegacyValleyPeakConfig,
): LegacyValleyPeakMemory {
  if (!memory) {
    return createLegacyValleyPeakMemory(config);
  }

  const averages = normalizeAverageList(memory.buyAverages ?? memory.sellAverages, config);
  return {
    startedAt: memory.startedAt,
    kama: normalizeKama(memory.kama),
    kamaBuySignal: normalizeAverage(memory.kamaBuySignal),
    kamaSellSignal: normalizeAverage(memory.kamaSellSignal),
    buyAverages: averages,
    sellAverages: averages,
    candleRanges: normalizeCandleRangeList(memory.candleRanges, config),
    priceRanges: normalizePriceRangeList(memory.priceRanges),
    exitGrids: memory.exitGrids,
  };
}

export function evaluateLegacyValleyPeak(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
): LegacyValleyPeakDecision {
  const tsSec = input.eventTime / 1000;
  memory.startedAt ??= input.eventTime;

  const averages = memory.buyAverages;
  memory.sellAverages = averages;
  const candleRanges = ensureCandleRangeMemories(memory, config);
  const priceRanges = ensurePriceRangeMemories(memory);
  for (const priceRange of priceRanges) {
    updateRollingPriceRange(priceRange, input, tsSec);
  }
  updateSignalSource(memory, config, input.price, tsSec);
  for (let index = 0; index < config.averagingRangesSec.length; index += 1) {
    const rangeSec = config.averagingRangesSec[index];
    updateRollingAverage(
      averages[index],
      input.price,
      tsSec,
      rangeSec,
      config.rateRatios[index],
      config.rateThresholdsLow[index],
      config.rateThresholdsHigh[index],
      config.relativeRateEnabled,
      config.derivativeClampMode,
      config.derivativeClampInnerThresholdRatio,
      config.movingAverageType,
    );
    updateRollingCandleRange(candleRanges[index], rangeSec, input, tsSec);
  }

  if (input.eventTime - memory.startedAt < config.saturationSec * 1000) {
    return holdLegacyValleyPeakDecision();
  }

  const feeAdjustedBuyRate = input.price / (1 - input.feeRate);
  const feeAdjustedSellRate = input.price * (1 - input.feeRate);

  const buyEntryPassed = shouldBuy(
    memory,
    config,
    config.buyConfirmationOffsets,
    config.buyEntrySignalTiming,
    input,
  );
  const buyExitPassed = shouldBuy(
    memory,
    config,
    config.buyExitConfirmationOffsets,
    config.buyExitSignalTiming,
    input,
  );
  if (buyEntryPassed || buyExitPassed) {
    const quoteSize = buyQuoteSize(memory, config, input, feeAdjustedBuyRate);
    const coverQuantity = buyCoverQuantity(memory, config, input, feeAdjustedBuyRate);
    const decision: LegacyValleyPeakDecision = {
      entrySignal:
        buyEntryPassed && quoteSize >= config.minTradeQuote
          ? {
              signal: "buy",
              reason: "legacy valley detected",
              quoteSize,
            }
          : { signal: "hold" },
      exitSignal:
        buyExitPassed && coverQuantity * feeAdjustedBuyRate >= config.minTradeQuote
          ? {
              signal: "buy",
              reason: "legacy valley detected",
              coverQuantity,
            }
          : { signal: "hold" },
    };
    if (hasLegacyValleyPeakSignal(decision)) {
      return decision;
    }
  }

  const sellEntryPassed = shouldSell(
    memory,
    config,
    config.sellConfirmationOffsets,
    config.sellEntrySignalTiming,
    input,
  );
  const sellExitPassed = shouldSell(
    memory,
    config,
    config.sellExitConfirmationOffsets,
    config.sellExitSignalTiming,
    input,
  );
  if (sellEntryPassed || sellExitPassed) {
    const quantity = sellQuantity(memory, config, input, feeAdjustedSellRate);
    const quoteSize = shortSellQuoteSize(memory, config, input, feeAdjustedSellRate);
    const decision: LegacyValleyPeakDecision = {
      entrySignal:
        sellEntryPassed && quoteSize >= config.minTradeQuote
          ? {
              signal: "sell",
              reason: "legacy peak detected",
              quoteSize,
            }
          : { signal: "hold" },
      exitSignal:
        sellExitPassed && quantity * feeAdjustedSellRate >= config.minTradeQuote
          ? {
              signal: "sell",
              reason: "legacy peak detected",
              quantity,
            }
          : { signal: "hold" },
    };
    if (hasLegacyValleyPeakSignal(decision)) {
      return decision;
    }
  }

  return holdLegacyValleyPeakDecision();
}

function holdLegacyValleyPeakDecision(): LegacyValleyPeakDecision {
  return {
    entrySignal: { signal: "hold" },
    exitSignal: { signal: "hold" },
  };
}

function hasLegacyValleyPeakSignal(decision: LegacyValleyPeakDecision): boolean {
  return decision.entrySignal.signal !== "hold" || decision.exitSignal.signal !== "hold";
}

interface AnticipatoryConfirmationPrediction {
  side: PositionLotSide;
  extremaX: number;
  extremaPrice: number;
}

function canAnticipateConfirmationFailure(
  side: PositionLotSide,
  pricePrediction: AnticipatoryConfirmationPrediction | undefined,
  failedConfirmationCount: number,
  missBudget: number,
): boolean {
  return (
    missBudget > 0 &&
    failedConfirmationCount > 0 &&
    failedConfirmationCount <= missBudget &&
    pricePrediction?.side === side
  );
}

function anticipatoryConfirmationMissBudget(
  config: LegacyValleyPeakConfig,
  confirmationOffsets: readonly number[],
): number {
  return Math.min(
    config.anticipatoryConfirmationMaxMisses,
    Math.max(0, confirmationOffsets.length - 1),
  );
}

function hasAnticipatoryConfirmationMissBudget(config: LegacyValleyPeakConfig): boolean {
  return (
    anticipatoryConfirmationMissBudget(config, config.buyConfirmationOffsets) > 0 ||
    anticipatoryConfirmationMissBudget(config, config.sellConfirmationOffsets) > 0 ||
    anticipatoryConfirmationMissBudget(config, config.buyExitConfirmationOffsets) > 0 ||
    anticipatoryConfirmationMissBudget(config, config.sellExitConfirmationOffsets) > 0
  );
}

function predictAnticipatoryConfirmationExtremum(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput | undefined,
): AnticipatoryConfirmationPrediction | undefined {
  if (
    config.anticipatoryConfirmationMaxMisses <= 0 ||
    !input ||
    input.price <= 0
  ) {
    return undefined;
  }

  const windowSec = config.anticipatoryConfirmationWindowSec;
  const trendMemory = anticipatoryTrendMemory(memory, config, windowSec);
  const fit = fitRollingAverageEntries(trendMemory, windowSec, input);
  if (!fit || Math.abs(fit.a) <= Number.EPSILON) {
    return undefined;
  }

  const extremaX = -fit.b / (2 * fit.a);
  const { minExtremaX, maxExtremaX } = anticipatoryExtremaXRange(config);
  if (extremaX < minExtremaX || extremaX > maxExtremaX) {
    return undefined;
  }

  const extremaPrice = fit.a * extremaX ** 2 + fit.b * extremaX + fit.c;
  if (!isPositiveFinite(extremaPrice)) {
    return undefined;
  }

  const side: PositionLotSide = fit.a > 0 ? "long" : "short";
  if (side === "long") {
    if (extremaPrice >= input.price) {
      return undefined;
    }
  } else if (extremaPrice <= input.price) {
    return undefined;
  }

  return {
    side,
    extremaX,
    extremaPrice,
  };
}

function fitRollingAverageEntries(
  memory: RollingAverageMemory | undefined,
  windowSec: number,
  input: LegacyValleyPeakInput | undefined,
): { a: number; b: number; c: number } | undefined {
  if (!memory || !input) {
    return undefined;
  }

  const latestTs = input.eventTime / 1000;
  const windowStartTs = latestTs - windowSec;
  const samples: Array<{ x: number; y: number; ts: number }> = [];
  const startIndex = clampInt(memory.startIndex ?? 0, 0, memory.timestamps.length);
  for (let index = startIndex; index < memory.timestamps.length; index += 1) {
    const ts = memory.timestamps[index];
    const value = memory.entries[index];
    if (
      ts >= windowStartTs &&
      ts <= latestTs &&
      isPositiveFinite(value)
    ) {
      samples.push({
        x: (ts - windowStartTs) / windowSec,
        y: value,
        ts,
      });
    }
  }

  if (samples.length < MIN_ANTICIPATORY_CONFIRMATION_SAMPLES) {
    return undefined;
  }

  const spanSec = samples[samples.length - 1].ts - samples[0].ts;
  if (spanSec < windowSec * 0.8) {
    return undefined;
  }

  return fitQuadratic(samples);
}

function anticipatoryExtremaXRange(
  config: LegacyValleyPeakConfig,
): { minExtremaX: number; maxExtremaX: number } {
  const lookaround = config.anticipatoryConfirmationLookaheadFraction;
  return {
    minExtremaX: 1 - lookaround,
    maxExtremaX: 1 + lookaround,
  };
}

function anticipatoryTrendMemory(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  targetWindowSec: number,
): RollingAverageMemory | undefined {
  const exact = config.averagingRangesSec.indexOf(targetWindowSec);
  if (exact >= 0) {
    return memory.buyAverages[exact] ?? memory.sellAverages[exact];
  }

  let closest = 0;
  let closestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < config.averagingRangesSec.length; index += 1) {
    const distance = Math.abs(config.averagingRangesSec[index] - targetWindowSec);
    if (distance < closestDistance) {
      closest = index;
      closestDistance = distance;
    }
  }
  return memory.buyAverages[closest] ?? memory.sellAverages[closest];
}

function fitQuadratic(
  samples: Array<{ x: number; y: number }>,
): { a: number; b: number; c: number } | undefined {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let sy = 0;
  let sxy = 0;
  let sx2y = 0;

  for (const sample of samples) {
    const x = sample.x;
    const x2 = x * x;
    s0 += 1;
    s1 += x;
    s2 += x2;
    s3 += x2 * x;
    s4 += x2 * x2;
    sy += sample.y;
    sxy += x * sample.y;
    sx2y += x2 * sample.y;
  }

  const solution = solve3x3(
    [
      [s4, s3, s2],
      [s3, s2, s1],
      [s2, s1, s0],
    ],
    [sx2y, sxy, sy],
  );
  if (!solution) {
    return undefined;
  }

  const [a, b, c] = solution;
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) {
    return undefined;
  }
  return { a, b, c };
}

function solve3x3(matrix: number[][], vector: number[]): number[] | undefined {
  const augmented = matrix.map((row, index) => [...row, vector[index]]);

  for (let pivot = 0; pivot < 3; pivot += 1) {
    let selected = pivot;
    for (let row = pivot + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[selected][pivot])) {
        selected = row;
      }
    }
    if (Math.abs(augmented[selected][pivot]) < 1e-12) {
      return undefined;
    }
    if (selected !== pivot) {
      const tmp = augmented[pivot];
      augmented[pivot] = augmented[selected];
      augmented[selected] = tmp;
    }

    const pivotValue = augmented[pivot][pivot];
    for (let column = pivot; column < 4; column += 1) {
      augmented[pivot][column] /= pivotValue;
    }

    for (let row = 0; row < 3; row += 1) {
      if (row === pivot) {
        continue;
      }
      const factor = augmented[row][pivot];
      for (let column = pivot; column < 4; column += 1) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }

  return [augmented[0][3], augmented[1][3], augmented[2][3]];
}

function legacyValleyPeakSignalReason(
  signal: LegacyValleyPeakEntrySignal | LegacyValleyPeakExitSignal,
): string | undefined {
  return signal.signal === "hold" ? undefined : signal.reason;
}

export function createLegacyValleyPeakDebugSnapshot(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  decision: LegacyValleyPeakDecision,
  lastExtrema?: Pick<
    LegacyValleyPeakDebugSnapshot,
    | "lastExtremaSignal"
    | "lastExtremaSignalAt"
    | "lastExtremaSignalPrice"
    | "lastExtremaSignalReason"
  >,
): LegacyValleyPeakDebugSnapshot {
  const saturated =
    input.eventTime - (memory.startedAt ?? input.eventTime) >= config.saturationSec * 1000;
  const feeAdjustedBuyRate = input.price / (1 - input.feeRate);
  const feeAdjustedSellRate = input.price * (1 - input.feeRate);

  return {
    updatedAt: input.eventTime,
    price: input.price,
    movingAverageType: config.movingAverageType,
    derivativeSource: config.derivativeSource,
    derivativeSourceValue: currentDerivativeSourceValue(memory, config, input.price),
    entrySignal: decision.entrySignal.signal,
    exitSignal: decision.exitSignal.signal,
    entryReason: legacyValleyPeakSignalReason(decision.entrySignal),
    exitReason: legacyValleyPeakSignalReason(decision.exitSignal),
    marketState: detectLegacyMarketState(memory, config),
    saturated,
    saturationRemainingMs: saturated
      ? 0
      : Math.max(
          0,
          config.saturationSec * 1000 - (input.eventTime - (memory.startedAt ?? input.eventTime)),
        ),
    ...(lastExtrema ?? {}),
    averages: config.averagingRangesSec.map((windowSec, index) => {
      const point = latestPoint(memory.buyAverages[index] ?? memory.sellAverages[index]);
      return {
        index,
        windowSec,
        avg: point?.avg,
        rate: point?.rate,
        rateClamped: point?.rateClamped,
        thresholdLow: config.rateThresholdsLow[index] ?? 0,
        thresholdHigh: config.rateThresholdsHigh[index] ?? 0,
        buyPrimary: index === config.buyDataIndex,
        sellPrimary: index === config.sellDataIndex,
        buyConfirmation: config.buyConfirmationOffsets.some(
          (offset) => index === config.buyDataIndex + offset,
        ) || config.buyExitConfirmationOffsets.some(
          (offset) => index === config.buyDataIndex + offset,
        ),
        sellConfirmation: config.sellConfirmationOffsets.some(
          (offset) => index === config.sellDataIndex + offset,
        ) || config.sellExitConfirmationOffsets.some(
          (offset) => index === config.sellDataIndex + offset,
        ),
        valley: isValley(memory.buyAverages[index], "start"),
        peak: isPeak(memory.sellAverages[index], "start"),
      };
    }),
    candleRanges: config.averagingRangesSec.map((windowSec, index) => {
      const point = memory.candleRanges[index]?.points.at(-1);
      const stats = candleRangeSampleStats(memory.candleRanges[index], windowSec);
      return {
        index,
        windowSec,
        avgPct: point?.avgPct,
        maxPct: point?.maxPct,
        currentPct: point?.currentPct,
        count: point?.count,
        sampleCount: stats.sampleCount,
        sampleSpanMs: stats.spanMs,
      };
    }),
    priceRanges: memory.priceRanges.map((range) => {
      const point = range.points.at(-1);
      const stats = priceRangeSampleStats(range);
      return {
        window: range.window,
        windowSec: range.windowSec,
        minPrice: point?.minPrice,
        maxPrice: point?.maxPrice,
        rangePct: point?.rangePct,
        updatedAt: point?.updatedAt,
        sampleCount: stats.sampleCount,
        sampleSpanMs: stats.spanMs,
      };
    }),
    buyCheck: buildLegacyValleyPeakCheckDebug(
      "buy",
      memory,
      config,
      input,
      feeAdjustedBuyRate,
      config.buyConfirmationOffsets,
      config.buyEntrySignalTiming,
    ),
    sellCheck: buildLegacyValleyPeakCheckDebug(
      "sell",
      memory,
      config,
      input,
      feeAdjustedSellRate,
      config.sellConfirmationOffsets,
      config.sellEntrySignalTiming,
    ),
    buyExitCheck: buildLegacyValleyPeakCheckDebug(
      "buy",
      memory,
      config,
      input,
      feeAdjustedBuyRate,
      config.buyExitConfirmationOffsets,
      config.buyExitSignalTiming,
    ),
    sellExitCheck: buildLegacyValleyPeakCheckDebug(
      "sell",
      memory,
      config,
      input,
      feeAdjustedSellRate,
      config.sellExitConfirmationOffsets,
      config.sellExitSignalTiming,
    ),
  };
}

function shouldBuy(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  confirmationOffsets: number[],
  timing: LegacyExtremaSignalTiming,
  input?: LegacyValleyPeakInput,
): boolean {
  const primary = primarySignalMemory("buy", memory, config);
  if (!isValley(primary, timing)) {
    return false;
  }

  const missBudget = anticipatoryConfirmationMissBudget(config, confirmationOffsets);
  const pricePrediction =
    missBudget > 0
      ? predictAnticipatoryConfirmationExtremum(memory, config, input)
      : undefined;
  let failedConfirmationCount = 0;
  for (const offset of confirmationOffsets) {
    const confirmation = memory.buyAverages[config.buyDataIndex + offset];
    const confirmationPoint = latestPoint(confirmation);
    if (confirmationPoint && confirmationPoint.rateClamped <= 0) {
      failedConfirmationCount += 1;
      if (
        !canAnticipateConfirmationFailure(
          "long",
          pricePrediction,
          failedConfirmationCount,
          missBudget,
        )
      ) {
        return false;
      }
    }
  }

  return true;
}

function shouldSell(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  confirmationOffsets: number[],
  timing: LegacyExtremaSignalTiming,
  input?: LegacyValleyPeakInput,
): boolean {
  const primary = primarySignalMemory("sell", memory, config);
  if (!isPeak(primary, timing)) {
    return false;
  }

  const missBudget = anticipatoryConfirmationMissBudget(config, confirmationOffsets);
  const pricePrediction =
    missBudget > 0
      ? predictAnticipatoryConfirmationExtremum(memory, config, input)
      : undefined;
  let failedConfirmationCount = 0;
  for (const offset of confirmationOffsets) {
    const confirmation = memory.sellAverages[config.sellDataIndex + offset];
    const confirmationPoint = latestPoint(confirmation);
    if (confirmationPoint && confirmationPoint.rateClamped >= 0) {
      failedConfirmationCount += 1;
      if (
        !canAnticipateConfirmationFailure(
          "short",
          pricePrediction,
          failedConfirmationCount,
          missBudget,
        )
      ) {
        return false;
      }
    }
  }

  return true;
}

function buildLegacyValleyPeakCheckDebug(
  side: "buy" | "sell",
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  rate: number,
  confirmationOffsets: number[],
  timing: LegacyExtremaSignalTiming,
): LegacyValleyPeakCheckDebug {
  const primaryIndex = side === "buy" ? config.buyDataIndex : config.sellDataIndex;
  const primary = primarySignalMemory(side, memory, config);
  const primaryPoint = latestPoint(primary);
  const primaryShape = !primaryPoint
    ? "missing"
    : side === "buy" && isValley(primary, timing)
      ? "valley"
      : side === "sell" && isPeak(primary, timing)
        ? "peak"
        : "flat";
  const missBudget = anticipatoryConfirmationMissBudget(config, confirmationOffsets);
  const pricePrediction =
    missBudget > 0
      ? predictAnticipatoryConfirmationExtremum(memory, config, input)
      : undefined;
  const failedConfirmationCount = confirmationOffsets.reduce((count, offset) => {
    const index = primaryIndex + offset;
    const confirmation = side === "buy"
      ? memory.buyAverages[index]
      : memory.sellAverages[index];
    const point = latestPoint(confirmation);
    if (!point) {
      return count;
    }
    const failed = side === "buy" ? point.rateClamped <= 0 : point.rateClamped >= 0;
    return failed ? count + 1 : count;
  }, 0);
  const confirmations = confirmationOffsets.map((offset) => {
    const index = primaryIndex + offset;
    const confirmation = side === "buy"
      ? memory.buyAverages[index]
      : memory.sellAverages[index];
    const point = latestPoint(confirmation);
    const expected: "positive" | "negative" = side === "buy" ? "positive" : "negative";
    const actualPassed =
      !point ||
      (side === "buy" ? point.rateClamped > 0 : point.rateClamped < 0);
    const anticipated =
      !actualPassed &&
      canAnticipateConfirmationFailure(
        side === "buy" ? "long" : "short",
        pricePrediction,
        failedConfirmationCount,
        missBudget,
      );
    return {
      index,
      windowSec: config.averagingRangesSec[index],
      rateClamped: point?.rateClamped,
      expected,
      passed: actualPassed || anticipated,
      anticipated: anticipated || undefined,
    };
  });

  if (side === "buy") {
    return {
      side,
      passed: shouldBuy(memory, config, confirmationOffsets, timing, input),
      primaryIndex,
      primaryWindowSec: config.averagingRangesSec[primaryIndex],
      primaryRate: primaryPoint?.rate,
      primaryRateClamped: primaryPoint?.rateClamped,
      primaryShape,
      confirmations,
      quoteSize: buyQuoteSize(memory, config, input, rate),
      coverQuantity: buyCoverQuantity(memory, config, input, rate),
      trendRate: trendSigmaRate(memory, config),
      effectiveSigma: buySizingSigma(memory, config),
      minTradeQuote: config.minTradeQuote,
    };
  }

  return {
    side,
    passed: shouldSell(memory, config, confirmationOffsets, timing, input),
    primaryIndex,
    primaryWindowSec: config.averagingRangesSec[primaryIndex],
    primaryRate: primaryPoint?.rate,
    primaryRateClamped: primaryPoint?.rateClamped,
    primaryShape,
    confirmations,
    quantity: sellQuantity(memory, config, input, rate),
    quoteSize: shortSellQuoteSize(memory, config, input, rate),
    trendRate: trendSigmaRate(memory, config),
    effectiveSigma: sellSizingSigma(memory, config),
    minTradeQuote: config.minTradeQuote,
  };
}

function detectLegacyMarketState(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): LegacyMarketStateDebug {
  const candidateIndices = [
    config.buyDataIndex,
    config.sellDataIndex,
    ...config.buyConfirmationOffsets.map((offset) => config.buyDataIndex + offset),
    ...config.sellConfirmationOffsets.map((offset) => config.sellDataIndex + offset),
    ...config.buyExitConfirmationOffsets.map((offset) => config.buyDataIndex + offset),
    ...config.sellExitConfirmationOffsets.map((offset) => config.sellDataIndex + offset),
  ];
  const candidates = [...new Set(candidateIndices)]
    .filter((index) => index >= 0 && index < config.averagingRangesSec.length)
    .map((index) => {
      const point = latestPoint(memory.buyAverages[index] ?? memory.sellAverages[index]);
      return {
        index,
        point,
        thresholdLow: config.rateThresholdsLow[index] ?? 0,
        thresholdHigh: config.rateThresholdsHigh[index] ?? 0,
      };
    })
    .filter((candidate) => candidate.point);

  const active = candidates
    .filter((candidate) => (candidate.point?.rateClamped ?? 0) !== 0)
    .sort(
      (left, right) =>
        Math.abs(right.point?.rateClamped ?? 0) - Math.abs(left.point?.rateClamped ?? 0) ||
        (right.index - left.index),
    )[0];
  const selected =
    active ??
    candidates.sort(
      (left, right) =>
        Math.abs(right.point?.rate ?? 0) - Math.abs(left.point?.rate ?? 0) ||
        (right.index - left.index),
    )[0];
  const rate = selected?.point?.rate;
  const rateClamped = selected?.point?.rateClamped ?? 0;

  return {
    state: rateClamped > 0 ? "rising" : rateClamped < 0 ? "falling" : "sideways",
    sourceIndex: selected?.index,
    windowSec:
      selected?.index === undefined
        ? undefined
        : config.averagingRangesSec[selected.index],
    rate,
    rateClamped,
    thresholdLow: selected?.thresholdLow,
    thresholdHigh: selected?.thresholdHigh,
  };
}

function buySizingSigma(memory: LegacyValleyPeakMemory, config: LegacyValleyPeakConfig): number {
  if (config.sigmaMode === "static") {
    return staticSigma(config, config.buySigma, defaultLegacyValleyPeakConfig.buySigma);
  }
  if (config.sigmaMode === "sigmoid-trend") {
    return sigmoidTrendSigma(memory, config, "buy");
  }
  return buyTrendSigma(memory, config);
}

function sellSizingSigma(memory: LegacyValleyPeakMemory, config: LegacyValleyPeakConfig): number {
  if (config.sigmaMode === "static") {
    return staticSigma(config, config.sellSigma, defaultLegacyValleyPeakConfig.sellSigma);
  }
  if (config.sigmaMode === "sigmoid-trend") {
    return sigmoidTrendSigma(memory, config, "sell");
  }
  return sellTrendSigma(memory, config);
}

function staticSigma(config: LegacyValleyPeakConfig, rawSigma: number, fallbackRawSigma: number): number {
  return config.relativeRateEnabled
    ? Math.max(
        MIN_RELATIVE_SIGMA,
        normalizeRelativeRate(rawSigma, normalizeRelativeRate(fallbackRawSigma, MIN_RELATIVE_SIGMA)),
      )
    : Math.max(MIN_ABSOLUTE_SIGMA, rawSigma);
}

function buyTrendSigma(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): number {
  return trendAdjustedSigma(memory, config, config.trendSigmaBuyB2);
}

function sellTrendSigma(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): number {
  return trendAdjustedSigma(memory, config, -config.trendSigmaSellB1);
}

function sigmoidTrendSigma(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  side: "buy" | "sell",
): number {
  const trend = trendSigmaRate(memory, config);
  const weight =
    side === "buy"
      ? sigmoid(-trend, config.trendSigmaBuyB2)
      : sigmoid(trend, config.trendSigmaSellB1);
  const rawSigma =
    config.sigmoidSigmaLow * weight +
    config.sigmoidSigmaHigh * (1 - weight);
  return staticSigma(config, rawSigma, rawSigma);
}

function trendAdjustedSigma(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  exponentScale: number,
): number {
  const exponent = clamp(
    exponentScale * trendSigmaRate(memory, config),
    -MAX_TREND_SIGMA_EXPONENT,
    MAX_TREND_SIGMA_EXPONENT,
  );
  const sigma = trendSigmaBase(config) * Math.exp(exponent);
  return Math.max(
    config.relativeRateEnabled ? MIN_RELATIVE_SIGMA : MIN_ABSOLUTE_SIGMA,
    sigma,
  );
}

function trendSigmaBase(config: LegacyValleyPeakConfig): number {
  return config.relativeRateEnabled
    ? Math.max(MIN_RELATIVE_SIGMA, normalizeRelativeRate(config.trendSigmaA, 0.00001))
    : Math.max(MIN_ABSOLUTE_SIGMA, config.trendSigmaA);
}

function trendSigmaRate(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): number {
  const index = trendSigmaWindowIndex(config);
  const point = latestPoint(memory.buyAverages[index] ?? memory.sellAverages[index]);
  const rate = point?.rate ?? 0;
  if (!Number.isFinite(rate)) {
    return 0;
  }

  return config.relativeRateEnabled ? rate * BTC_SCALE_REFERENCE_PRICE : rate;
}

function trendSigmaWindowIndex(config: LegacyValleyPeakConfig): number {
  const targetWindowSec = config.trendSigmaWindowSec || TREND_SIGMA_WINDOW_SEC;
  const exact = config.averagingRangesSec.indexOf(targetWindowSec);
  if (exact >= 0) {
    return exact;
  }

  let closest = 0;
  let closestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < config.averagingRangesSec.length; index += 1) {
    const distance = Math.abs(config.averagingRangesSec[index] - targetWindowSec);
    if (distance < closestDistance) {
      closest = index;
      closestDistance = distance;
    }
  }
  return closest;
}

function buyQuoteSize(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  rate: number,
): number {
  if (!config.longSideEnabled) {
    return 0;
  }

  const data = memory.buyAverages[Math.min(3, memory.buyAverages.length - 1)];
  const derivative = latestPoint(data)?.rate ?? 0;
  const buyingPowerQuote = Math.max(0, input.buyingPowerQuote);
  const desired = buyingPowerQuote * config.buySpendRate * gaussian(derivative, 0, buySizingSigma(memory, config));

  return clamp(
    desired,
    config.minTradeQuote,
    Math.min(config.maxTradeQuote, buyingPowerQuote, rate * 10_000),
  );
}

function buyCoverQuantity(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  rate: number,
): number {
  if (!config.shortSideEnabled) {
    return 0;
  }

  const data = memory.buyAverages[Math.min(3, memory.buyAverages.length - 1)];
  const derivative = latestPoint(data)?.rate ?? 0;
  const shortBaseFree = Math.max(0, input.shortBaseFree ?? 0);
  const desired = shortBaseFree * config.buySpendRate * gaussian(derivative, 0, buySizingSigma(memory, config));
  const minQuantity = config.minTradeQuote / rate;
  const maxQuantity = Math.min(config.maxTradeQuote / rate, shortBaseFree);

  return clamp(desired, minQuantity, maxQuantity);
}

function sellQuantity(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  rate: number,
): number {
  if (!config.longSideEnabled) {
    return 0;
  }

  const data = memory.sellAverages[Math.min(3, memory.sellAverages.length - 1)];
  const derivative = latestPoint(data)?.rate ?? 0;
  const desired = input.baseFree * config.sellAmountRate * gaussian(derivative, 0, sellSizingSigma(memory, config));
  const minQuantity = config.minTradeQuote / rate;
  const maxQuantity = Math.min(config.maxTradeQuote / rate, input.baseFree);

  return clamp(desired, minQuantity, maxQuantity);
}

function shortSellQuoteSize(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  input: LegacyValleyPeakInput,
  rate: number,
): number {
  if (!config.shortSideEnabled) {
    return 0;
  }

  const data = memory.sellAverages[Math.min(3, memory.sellAverages.length - 1)];
  const derivative = latestPoint(data)?.rate ?? 0;
  const sellingPowerQuote = Math.max(0, input.shortSellingPowerQuote ?? 0);
  const desired = sellingPowerQuote * config.sellAmountRate * gaussian(derivative, 0, sellSizingSigma(memory, config));

  return clamp(
    desired,
    config.minTradeQuote,
    Math.min(config.maxTradeQuote, sellingPowerQuote, rate * 10_000),
  );
}

function updateSignalSource(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  price: number,
  tsSec: number,
): void {
  if (config.derivativeSource !== "kama") {
    return;
  }

  const value = updateKama(memory.kama, config, price);
  updateSignalDerivative(
    memory.kamaBuySignal,
    value,
    tsSec,
    config.rateThresholdsLow[config.buyDataIndex] ?? 0,
    config.rateThresholdsHigh[config.buyDataIndex] ?? 0,
    config.relativeRateEnabled,
    config.derivativeClampMode,
    config.derivativeClampInnerThresholdRatio,
  );
  updateSignalDerivative(
    memory.kamaSellSignal,
    value,
    tsSec,
    config.rateThresholdsLow[config.sellDataIndex] ?? 0,
    config.rateThresholdsHigh[config.sellDataIndex] ?? 0,
    config.relativeRateEnabled,
    config.derivativeClampMode,
    config.derivativeClampInnerThresholdRatio,
  );
}

function primarySignalMemory(
  side: "buy" | "sell",
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): RollingAverageMemory | undefined {
  if (config.derivativeSource === "kama") {
    return side === "buy" ? memory.kamaBuySignal : memory.kamaSellSignal;
  }

  return side === "buy"
    ? memory.buyAverages[config.buyDataIndex]
    : memory.sellAverages[config.sellDataIndex];
}

function currentDerivativeSourceValue(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
  price: number,
): number {
  if (config.derivativeSource !== "kama") {
    return latestPoint(memory.buyAverages[config.buyDataIndex])?.avg ?? price;
  }
  return memory.kama.ama ?? price;
}

function updateSignalDerivative(
  memory: RollingAverageMemory,
  value: number,
  tsSec: number,
  thresholdLow: number,
  thresholdHigh: number,
  relativeRateEnabled: boolean,
  clampMode: LegacyDerivativeClampMode,
  innerThresholdRatio: number,
): void {
  const previousValue = memory.averages.at(-1);
  const previousTsSec = memory.timestamps.at(-1);
  let derivative = 0;

  if (
    previousValue !== undefined &&
    previousTsSec !== undefined &&
    tsSec > previousTsSec &&
    (!relativeRateEnabled || previousValue > 0)
  ) {
    const delta = tsSec - previousTsSec;
    derivative = relativeRateEnabled
      ? (value - previousValue) / previousValue / delta
      : (value - previousValue) / delta;
  }

  const rateClamped = clampDerivativeRate(
    derivative,
    latestPoint(memory)?.rateClamped ?? 0,
    thresholdLow,
    thresholdHigh,
    clampMode,
    innerThresholdRatio,
  );

  recordPoint(memory, value, derivative, rateClamped);
  memory.entries.push(value);
  memory.averages.push(value);
  memory.timestamps.push(tsSec);
  memory.sum = value;
  if (memory.entries.length > 2) {
    memory.entries.splice(0, memory.entries.length - 2);
    memory.averages.splice(0, memory.averages.length - 2);
    memory.timestamps.splice(0, memory.timestamps.length - 2);
  }
  memory.startIndex = 0;
  memory.previousSampleIndex = undefined;
}

function updateKama(
  memory: RollingKamaMemory,
  config: LegacyValleyPeakConfig,
  source: number,
): number {
  if (!isPositiveFinite(source)) {
    return memory.ama ?? source;
  }

  memory.sources.push(source);
  const erLen = config.kamaErLen;
  let efficiencyRatio = 0;

  if (memory.sources.length > erLen) {
    const latestIndex = memory.sources.length - 1;
    const previousIndex = latestIndex - erLen;
    const change = Math.abs(source - memory.sources[previousIndex]);
    let volatility = 0;
    for (let index = previousIndex + 1; index <= latestIndex; index += 1) {
      volatility += Math.abs(memory.sources[index] - memory.sources[index - 1]);
    }
    efficiencyRatio = volatility !== 0 ? change / volatility : 0;
  }

  const chop = Math.pow(clamp(efficiencyRatio, 0, 1), config.kamaPower);
  const alphaFast = 2 / (config.kamaFastLen + 1);
  const alphaSlow = 2 / (config.kamaSlowLen + 1);
  const alpha = alphaSlow + chop * (alphaFast - alphaSlow);
  memory.ama = memory.ama === undefined
    ? source
    : memory.ama + alpha * (source - memory.ama);
  compactKama(memory, erLen);
  return memory.ama;
}

function updateRollingAverage(
  memory: RollingAverageMemory,
  value: number,
  tsSec: number,
  rangeSec: number,
  derivativeRatio: number,
  thresholdLow: number,
  thresholdHigh: number,
  relativeRateEnabled: boolean,
  clampMode: LegacyDerivativeClampMode,
  innerThresholdRatio: number,
  averageType: LegacyMovingAverageType,
): void {
  if (averageType === "ema") {
    updateRollingEma(
      memory,
      value,
      tsSec,
      rangeSec,
      derivativeRatio,
      thresholdLow,
      thresholdHigh,
      relativeRateEnabled,
      clampMode,
      innerThresholdRatio,
    );
    return;
  }

  let startIndex = memory.startIndex ?? 0;
  startIndex = clampInt(startIndex, 0, memory.timestamps.length);

  while (
    startIndex < memory.timestamps.length &&
    memory.timestamps[startIndex] + rangeSec < tsSec
  ) {
    memory.sum -= memory.entries[startIndex] ?? 0;
    startIndex += 1;
  }
  memory.startIndex = startIndex;

  memory.timestamps.push(tsSec);
  memory.entries.push(value);
  memory.sum += value;
  const activeCount = memory.timestamps.length - startIndex;
  const avg = memory.sum / activeCount;
  memory.averages.push(avg);

  if (activeCount < 2) {
    recordPoint(memory, avg, 0, 0);
    compactRollingAverage(memory);
    return;
  }

  const lastIndex = memory.timestamps.length - 1;
  const windowSize = memory.timestamps[lastIndex] - memory.timestamps[startIndex];
  const pointTs = tsSec - windowSize * derivativeRatio;
  const sampleValue = sampleAverage(memory, pointTs, startIndex);

  const delta = tsSec - pointTs;
  if (delta === 0) {
    recordPoint(memory, avg, 0, 0);
    compactRollingAverage(memory);
    return;
  }

  if (relativeRateEnabled && sampleValue <= 0) {
    recordPoint(memory, avg, 0, 0);
    compactRollingAverage(memory);
    return;
  }

  const derivative = relativeRateEnabled
    ? (avg - sampleValue) / sampleValue / delta
    : (avg - sampleValue) / delta;
  const rateClamped = clampDerivativeRate(
    derivative,
    latestPoint(memory)?.rateClamped ?? 0,
    thresholdLow,
    thresholdHigh,
    clampMode,
    innerThresholdRatio,
  );

  recordPoint(memory, avg, derivative, rateClamped);
  compactRollingAverage(memory);
}

function updateRollingEma(
  memory: RollingAverageMemory,
  value: number,
  tsSec: number,
  rangeSec: number,
  derivativeRatio: number,
  thresholdLow: number,
  thresholdHigh: number,
  relativeRateEnabled: boolean,
  clampMode: LegacyDerivativeClampMode,
  innerThresholdRatio: number,
): void {
  let startIndex = memory.startIndex ?? 0;
  startIndex = clampInt(startIndex, 0, memory.timestamps.length);

  while (
    startIndex < memory.timestamps.length &&
    memory.timestamps[startIndex] + rangeSec < tsSec
  ) {
    startIndex += 1;
  }
  memory.startIndex = startIndex;

  const previousAvg = memory.averages.at(-1);
  const previousTsSec = memory.timestamps.at(-1);
  const avg =
    previousAvg === undefined || previousTsSec === undefined || tsSec <= previousTsSec
      ? value
      : previousAvg + emaAlpha(tsSec - previousTsSec, rangeSec) * (value - previousAvg);

  memory.timestamps.push(tsSec);
  memory.entries.push(value);
  memory.averages.push(avg);
  memory.sum = avg;

  const activeCount = memory.timestamps.length - startIndex;
  if (activeCount < 2) {
    recordPoint(memory, avg, 0, 0);
    compactRollingAverage(memory);
    return;
  }

  const pointTs = Math.max(
    memory.timestamps[startIndex],
    tsSec - rangeSec * derivativeRatio,
  );
  const sampleValue = sampleAverage(memory, pointTs, startIndex);
  const delta = tsSec - pointTs;
  if (delta === 0 || (relativeRateEnabled && sampleValue <= 0)) {
    recordPoint(memory, avg, 0, 0);
    compactRollingAverage(memory);
    return;
  }

  const derivative = relativeRateEnabled
    ? (avg - sampleValue) / sampleValue / delta
    : (avg - sampleValue) / delta;
  const rateClamped = clampDerivativeRate(
    derivative,
    latestPoint(memory)?.rateClamped ?? 0,
    thresholdLow,
    thresholdHigh,
    clampMode,
    innerThresholdRatio,
  );

  recordPoint(memory, avg, derivative, rateClamped);
  compactRollingAverage(memory);
}

function emaAlpha(deltaSec: number, rangeSec: number): number {
  const tauSec = Math.max(Number.EPSILON, rangeSec / 2);
  return clamp(1 - Math.exp(-Math.max(0, deltaSec) / tauSec), 0, 1);
}

function clampDerivativeRate(
  derivative: number,
  previousClamped: number,
  thresholdLow: number,
  thresholdHigh: number,
  mode: LegacyDerivativeClampMode,
  innerThresholdRatio: number,
): number {
  if (mode !== "hysteresis") {
    return deadbandClampDerivativeRate(derivative, thresholdLow, thresholdHigh);
  }

  const positiveExitThreshold = thresholdHigh * innerThresholdRatio;
  const negativeExitThreshold = thresholdLow * innerThresholdRatio;
  if (previousClamped > 0 && derivative > positiveExitThreshold) {
    return derivative;
  }
  if (previousClamped < 0 && derivative < -negativeExitThreshold) {
    return derivative;
  }
  return deadbandClampDerivativeRate(derivative, thresholdLow, thresholdHigh);
}

function deadbandClampDerivativeRate(
  derivative: number,
  thresholdLow: number,
  thresholdHigh: number,
): number {
  if (derivative >= thresholdHigh) {
    return derivative;
  }
  if (derivative <= -thresholdLow) {
    return derivative;
  }
  return 0;
}

function updateRollingCandleRange(
  memory: RollingCandleRangeMemory,
  rangeSec: number,
  input: LegacyValleyPeakInput,
  tsSec: number,
): void {
  const source = input.sourceCandle;
  const sourceDurationSec = source
    ? Math.max(0, (source.closeTime - source.openTime) / 1000)
    : 0;

  if (source && sourceDurationSec <= rangeSec) {
    observeCandleRangeSample(memory, rangeSec, {
      openTimeSec: source.openTime / 1000,
      open: source.open,
      high: source.high,
      low: source.low,
      close: source.close,
    });
    return;
  }

  observeCandleRangeSample(memory, rangeSec, {
    openTimeSec: tsSec,
    open: input.price,
    high: input.price,
    low: input.price,
    close: input.price,
  });
}

function observeCandleRangeSample(
  memory: RollingCandleRangeMemory,
  rangeSec: number,
  sample: {
    openTimeSec: number;
    open: number;
    high: number;
    low: number;
    close: number;
  },
): void {
  if (
    rangeSec <= 0 ||
    !isNonNegativeFinite(sample.openTimeSec) ||
    !isPositiveFinite(sample.open) ||
    !isPositiveFinite(sample.high) ||
    !isPositiveFinite(sample.low) ||
    !isPositiveFinite(sample.close)
  ) {
    return;
  }

  const bucketStartSec = Math.floor(sample.openTimeSec / rangeSec) * rangeSec;
  const current = memory.current;
  if (current && bucketStartSec < current.bucketStartSec) {
    return;
  }

  if (!current || bucketStartSec > current.bucketStartSec) {
    finalizeCandleRangeBucket(memory);
    memory.current = {
      bucketStartSec,
      open: sample.open,
      high: sample.high,
      low: sample.low,
      close: sample.close,
    };
    recordCandleRangePoint(memory);
    return;
  }

  current.high = Math.max(current.high, sample.high);
  current.low = Math.min(current.low, sample.low);
  current.close = sample.close;
  recordCandleRangePoint(memory);
}

function finalizeCandleRangeBucket(memory: RollingCandleRangeMemory): void {
  const bucket = memory.current;
  if (!bucket) {
    return;
  }

  const rangePct = candleRangePct(bucket);
  if (rangePct !== undefined) {
    const entryIndex = nextCandleRangeEntryIndex(memory);
    memory.entries.push(rangePct);
    memory.timestamps.push(bucket.bucketStartSec);
    memory.sum += rangePct;
    pushCandleRangeMaxCandidate(memory, entryIndex, rangePct);
    memory.nextIndex = entryIndex + 1;

    if (memory.entries.length > CANDLE_RANGE_WINDOW_SIZE) {
      const removed = memory.entries.shift() ?? 0;
      memory.timestamps.shift();
      memory.sum -= removed;
      memory.entryOffset += 1;
    }
    expireCandleRangeMaxCandidates(memory);
    memory.max = candleRangeMaxFromCandidates(memory, memory.entryOffset);
  }

  memory.current = undefined;
}

function recordCandleRangePoint(memory: RollingCandleRangeMemory): void {
  const currentPct = memory.current ? candleRangePct(memory.current) : undefined;
  const includeCurrent = currentPct !== undefined;
  const completedLimit = includeCurrent
    ? Math.max(0, CANDLE_RANGE_WINDOW_SIZE - 1)
    : CANDLE_RANGE_WINDOW_SIZE;
  const completedCount = Math.min(memory.entries.length, completedLimit);
  const excludedCount = memory.entries.length - completedCount;
  const completedStartIndex = memory.entryOffset + excludedCount;

  let sum = memory.sum;
  for (let index = 0; index < excludedCount; index += 1) {
    sum -= memory.entries[index] ?? 0;
  }
  let max =
    completedCount === 0
      ? 0
      : completedCount === memory.entries.length
        ? memory.max
        : candleRangeMaxFromCandidates(memory, completedStartIndex);
  let count = completedCount;

  if (currentPct !== undefined) {
    sum += currentPct;
    max = count === 0 ? currentPct : Math.max(max, currentPct);
    count += 1;
  }

  if (count === 0) {
    return;
  }

  recordCandleRangePointValue(memory, {
    avgPct: sum / count,
    maxPct: max,
    currentPct: currentPct ?? 0,
    count,
  });
}

function nextCandleRangeEntryIndex(memory: RollingCandleRangeMemory): number {
  const entryOffset = Number.isFinite(memory.entryOffset)
    ? Math.max(0, Math.floor(memory.entryOffset))
    : 0;
  const minimumNextIndex = entryOffset + memory.entries.length;
  const nextIndex = Number.isFinite(memory.nextIndex)
    ? Math.max(minimumNextIndex, Math.floor(memory.nextIndex))
    : minimumNextIndex;

  memory.entryOffset = entryOffset;
  memory.nextIndex = nextIndex;
  return nextIndex;
}

function pushCandleRangeMaxCandidate(
  memory: RollingCandleRangeMemory,
  index: number,
  value: number,
): void {
  while (
    memory.maxCandidates.length > 0 &&
    memory.maxCandidates[memory.maxCandidates.length - 1].value <= value
  ) {
    memory.maxCandidates.pop();
  }
  memory.maxCandidates.push({ index, value });
}

function expireCandleRangeMaxCandidates(memory: RollingCandleRangeMemory): void {
  while (
    memory.maxCandidates.length > 0 &&
    memory.maxCandidates[0].index < memory.entryOffset
  ) {
    memory.maxCandidates.shift();
  }
}

function candleRangeMaxFromCandidates(
  memory: RollingCandleRangeMemory,
  startIndex: number,
): number {
  for (const candidate of memory.maxCandidates) {
    if (candidate.index >= startIndex) {
      return candidate.value;
    }
  }
  return 0;
}

function recordCandleRangePointValue(
  memory: RollingCandleRangeMemory,
  point: RollingCandleRangePoint,
): void {
  const points = memory.points;
  if (points.length === 0) {
    points.push({ ...point });
    return;
  }
  if (points.length === 1) {
    points.push({ ...point });
    return;
  }
  if (points.length > 2) {
    points[1] = points[points.length - 1];
    points.length = 2;
  }

  const previous = points[0];
  const latest = points[1];
  previous.avgPct = latest.avgPct;
  previous.maxPct = latest.maxPct;
  previous.currentPct = latest.currentPct;
  previous.count = latest.count;
  latest.avgPct = point.avgPct;
  latest.maxPct = point.maxPct;
  latest.currentPct = point.currentPct;
  latest.count = point.count;
}

function candleRangeSampleStats(
  memory: RollingCandleRangeMemory | undefined,
  rangeSec: number,
): { sampleCount: number; spanMs: number } {
  if (!memory || rangeSec <= 0) {
    return { sampleCount: 0, spanMs: 0 };
  }

  const starts = [
    ...memory.timestamps,
    ...(memory.current ? [memory.current.bucketStartSec] : []),
  ]
    .filter(isNonNegativeFinite)
    .sort((left, right) => left - right);
  if (starts.length === 0) {
    return { sampleCount: 0, spanMs: 0 };
  }

  const latestPoint = memory.points.at(-1);
  const sampleCount = Math.min(starts.length, Math.max(0, latestPoint?.count ?? starts.length));
  if (sampleCount <= 0) {
    return { sampleCount: 0, spanMs: 0 };
  }

  const includedStarts = starts.slice(-sampleCount);
  const first = includedStarts[0];
  const last = includedStarts[includedStarts.length - 1];
  return {
    sampleCount,
    spanMs: Math.max(0, last - first + rangeSec) * 1000,
  };
}

function candleRangePct(bucket: RollingCandleRangeBucket): number | undefined {
  if (
    !isPositiveFinite(bucket.open) ||
    !isPositiveFinite(bucket.high) ||
    !isPositiveFinite(bucket.low)
  ) {
    return undefined;
  }

  return Math.max(0, ((bucket.high - bucket.low) / bucket.open) * 100);
}

function updateRollingPriceRange(
  memory: RollingPriceRangeMemory,
  input: LegacyValleyPeakInput,
  tsSec: number,
): void {
  const source = input.sourceCandle;
  const sourceDurationSec = source
    ? Math.max(0, (source.closeTime - source.openTime) / 1000)
    : 0;

  if (source && sourceDurationSec <= memory.bucketSec) {
    observePriceRangeSample(
      memory,
      source.openTime / 1000,
      source.low,
      source.high,
      tsSec,
    );
    return;
  }

  observePriceRangeSample(memory, tsSec, input.price, input.price, tsSec);
}

function observePriceRangeSample(
  memory: RollingPriceRangeMemory,
  sampleTsSec: number,
  low: number,
  high: number,
  nowSec: number,
): void {
  if (
    !isNonNegativeFinite(sampleTsSec) ||
    !isPositiveFinite(low) ||
    !isPositiveFinite(high)
  ) {
    return;
  }

  if (sampleTsSec + memory.bucketSec <= nowSec - memory.windowSec) {
    return;
  }

  const bucketStartSec =
    Math.floor(sampleTsSec / memory.bucketSec) * memory.bucketSec;
  const current = memory.current;
  if (current && bucketStartSec < current.bucketStartSec) {
    expireRollingPriceRange(memory, nowSec);
    recordRollingPriceRangePoint(memory, nowSec);
    return;
  }

  if (!current || bucketStartSec > current.bucketStartSec) {
    finalizePriceRangeBucket(memory);
    rememberPriceRangeBucketStart(memory, bucketStartSec);
    memory.current = {
      bucketStartSec,
      low,
      high,
    };
  } else {
    current.low = Math.min(current.low, low);
    current.high = Math.max(current.high, high);
  }

  expireRollingPriceRange(memory, nowSec);
  recordRollingPriceRangePoint(memory, nowSec);
}

function finalizePriceRangeBucket(memory: RollingPriceRangeMemory): void {
  const bucket = memory.current;
  if (!bucket) {
    return;
  }

  const finalized = { ...bucket };
  while (
    memory.minCandidates.length > 0 &&
    memory.minCandidates[memory.minCandidates.length - 1].low >= finalized.low
  ) {
    memory.minCandidates.pop();
  }
  memory.minCandidates.push(finalized);

  while (
    memory.maxCandidates.length > 0 &&
    memory.maxCandidates[memory.maxCandidates.length - 1].high <= finalized.high
  ) {
    memory.maxCandidates.pop();
  }
  memory.maxCandidates.push(finalized);
  memory.current = undefined;
}

function expireRollingPriceRange(
  memory: RollingPriceRangeMemory,
  nowSec: number,
): void {
  const cutoffSec = nowSec - memory.windowSec;
  while (
    memory.bucketStarts.length > 0 &&
    memory.bucketStarts[0] + memory.bucketSec <= cutoffSec
  ) {
    memory.bucketStarts.shift();
  }
  while (
    memory.minCandidates.length > 0 &&
    memory.minCandidates[0].bucketStartSec + memory.bucketSec <= cutoffSec
  ) {
    memory.minCandidates.shift();
  }
  while (
    memory.maxCandidates.length > 0 &&
    memory.maxCandidates[0].bucketStartSec + memory.bucketSec <= cutoffSec
  ) {
    memory.maxCandidates.shift();
  }
}

function rememberPriceRangeBucketStart(
  memory: RollingPriceRangeMemory,
  bucketStartSec: number,
): void {
  if (memory.bucketStarts[memory.bucketStarts.length - 1] === bucketStartSec) {
    return;
  }
  memory.bucketStarts.push(bucketStartSec);
}

function recordRollingPriceRangePoint(
  memory: RollingPriceRangeMemory,
  nowSec: number,
): void {
  const minCandidate = memory.minCandidates[0];
  const maxCandidate = memory.maxCandidates[0];
  let minPrice = minCandidate?.low;
  let maxPrice = maxCandidate?.high;

  if (memory.current) {
    minPrice =
      minPrice === undefined
        ? memory.current.low
        : Math.min(minPrice, memory.current.low);
    maxPrice =
      maxPrice === undefined
        ? memory.current.high
        : Math.max(maxPrice, memory.current.high);
  }

  if (minPrice === undefined || maxPrice === undefined || minPrice <= 0) {
    return;
  }

  recordRollingPriceRangePointValue(memory, {
    window: memory.window,
    windowSec: memory.windowSec,
    minPrice,
    maxPrice,
    rangePct: ((maxPrice - minPrice) / minPrice) * 100,
    updatedAt: nowSec * 1000,
  });
}

function recordRollingPriceRangePointValue(
  memory: RollingPriceRangeMemory,
  point: RollingPriceRangePoint,
): void {
  const points = memory.points;
  if (points.length === 0) {
    points.push({ ...point });
    return;
  }
  if (points.length === 1) {
    points.push({ ...point });
    return;
  }
  if (points.length > 2) {
    points[1] = points[points.length - 1];
    points.length = 2;
  }

  const previous = points[0];
  const latest = points[1];
  previous.window = latest.window;
  previous.windowSec = latest.windowSec;
  previous.minPrice = latest.minPrice;
  previous.maxPrice = latest.maxPrice;
  previous.rangePct = latest.rangePct;
  previous.updatedAt = latest.updatedAt;
  latest.window = point.window;
  latest.windowSec = point.windowSec;
  latest.minPrice = point.minPrice;
  latest.maxPrice = point.maxPrice;
  latest.rangePct = point.rangePct;
  latest.updatedAt = point.updatedAt;
}

function priceRangeSampleStats(
  memory: RollingPriceRangeMemory | undefined,
): { sampleCount: number; spanMs: number } {
  if (!memory) {
    return { sampleCount: 0, spanMs: 0 };
  }

  const starts = [
    ...(memory.bucketStarts ?? []),
    ...(memory.current ? [memory.current.bucketStartSec] : []),
  ]
    .filter(isNonNegativeFinite)
    .sort((left, right) => left - right);
  const uniqueStarts = [...new Set(starts)];
  if (uniqueStarts.length === 0) {
    return { sampleCount: 0, spanMs: 0 };
  }

  const first = uniqueStarts[0];
  const last = uniqueStarts[uniqueStarts.length - 1];
  return {
    sampleCount: uniqueStarts.length,
    spanMs: Math.max(0, last - first + memory.bucketSec) * 1000,
  };
}

function rollingAverageSampleSpanSec(memory: RollingAverageMemory | undefined): number {
  const timestamps = memory?.timestamps ?? [];
  if (timestamps.length === 0) {
    return 0;
  }

  const startIndex = clampInt(memory?.startIndex ?? 0, 0, timestamps.length - 1);
  return Math.max(0, timestamps[timestamps.length - 1] - timestamps[startIndex]);
}

function sampleAverage(
  memory: RollingAverageMemory,
  ts: number,
  startIndex = 0,
): number {
  const data = memory.averages;
  const timestamps = memory.timestamps;
  const lastIndex = timestamps.length - 1;
  if (lastIndex <= startIndex || ts <= timestamps[startIndex]) {
    memory.previousSampleIndex = startIndex;
    return data[startIndex];
  }

  let index =
    memory.previousSampleIndex ??
    binarySearchLeft(timestamps, ts, startIndex, timestamps.length) - 1;
  index = clampInt(index, startIndex, Math.max(startIndex, lastIndex - 1));

  while (index > startIndex && timestamps[index] >= ts) {
    index -= 1;
  }
  while (index < lastIndex - 1 && timestamps[index + 1] < ts) {
    index += 1;
  }

  const prev = data[index];
  const next = data[index + 1];
  const delta = timestamps[index + 1] - timestamps[index];
  if (delta === 0) {
    memory.previousSampleIndex = index;
    return prev;
  }

  const nextDelta = timestamps[index + 1] - ts;
  const fraction = nextDelta / delta;
  memory.previousSampleIndex = index;
  return interpolate(prev, next, fraction);
}

function isValley(
  memory: RollingAverageMemory | undefined,
  timing: LegacyExtremaSignalTiming,
): boolean {
  return timing === "start" ? isValleyStart(memory) : isValleyEnd(memory);
}

function isPeak(
  memory: RollingAverageMemory | undefined,
  timing: LegacyExtremaSignalTiming,
): boolean {
  return timing === "start" ? isPeakStart(memory) : isPeakEnd(memory);
}

function isValleyStart(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(latest && previous && latest.rateClamped >= 0 && previous.rateClamped < 0);
}

function isValleyEnd(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(
    latest &&
    previous &&
    latest.rateClamped > 0 &&
    previous.rateClamped <= 0,
  );
}

function isPeakStart(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(latest && previous && latest.rateClamped <= 0 && previous.rateClamped > 0);
}

function isPeakEnd(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(
    latest &&
    previous &&
    latest.rateClamped < 0 &&
    previous.rateClamped >= 0,
  );
}

function latestPoint(memory: RollingAverageMemory | undefined): RollingAveragePoint | undefined {
  return memory?.points.at(-1);
}

function previousPoint(memory: RollingAverageMemory | undefined): RollingAveragePoint | undefined {
  return memory?.points.at(-2);
}

function recordPoint(
  memory: RollingAverageMemory,
  avg: number,
  rate: number,
  rateClamped: number,
): void {
  const points = memory.points;
  if (points.length === 0) {
    points.push({ avg, rate, rateClamped });
    return;
  }
  if (points.length === 1) {
    points.push({ avg, rate, rateClamped });
    return;
  }
  if (points.length > 2) {
    points[1] = points[points.length - 1];
    points.length = 2;
  }

  const previous = points[0];
  const latest = points[1];
  previous.avg = latest.avg;
  previous.rate = latest.rate;
  previous.rateClamped = latest.rateClamped;
  latest.avg = avg;
  latest.rate = rate;
  latest.rateClamped = rateClamped;
}

function createRollingAverageMemory(): RollingAverageMemory {
  return {
    entries: [],
    averages: [],
    timestamps: [],
    sum: 0,
    startIndex: 0,
    points: [],
  };
}

function createRollingKamaMemory(): RollingKamaMemory {
  return {
    sources: [],
  };
}

function createRollingCandleRangeMemory(): RollingCandleRangeMemory {
  return {
    entries: [],
    timestamps: [],
    sum: 0,
    max: 0,
    entryOffset: 0,
    nextIndex: 0,
    maxCandidates: [],
    points: [],
  };
}

function createRollingPriceRangeMemories(): RollingPriceRangeMemory[] {
  return PRICE_RANGE_WINDOWS.map(({ window, windowSec }) =>
    createRollingPriceRangeMemory(window, windowSec),
  );
}

function createRollingPriceRangeMemory(
  window: RollingPriceRangeWindow,
  windowSec: number,
): RollingPriceRangeMemory {
  return {
    window,
    windowSec,
    bucketSec: PRICE_RANGE_BUCKET_SEC,
    bucketStarts: [],
    minCandidates: [],
    maxCandidates: [],
    points: [],
  };
}

function normalizeAverageList(
  list: RollingAverageMemory[] | undefined,
  config: LegacyValleyPeakConfig,
): RollingAverageMemory[] {
  return config.averagingRangesSec.map((_, index) => normalizeAverage(list?.[index]));
}

function normalizeCandleRangeList(
  list: RollingCandleRangeMemory[] | undefined,
  config: LegacyValleyPeakConfig,
): RollingCandleRangeMemory[] {
  return config.averagingRangesSec.map((_, index) =>
    normalizeCandleRange(list?.[index]),
  );
}

function normalizePriceRangeList(
  list: RollingPriceRangeMemory[] | undefined,
): RollingPriceRangeMemory[] {
  return PRICE_RANGE_WINDOWS.map(({ window, windowSec }) => {
    const existing = list?.find((memory) => memory.window === window);
    return normalizePriceRange(existing, window, windowSec);
  });
}

function normalizeAverage(memory: RollingAverageMemory | undefined): RollingAverageMemory {
  const entries = memory?.entries ?? [];
  const averages = memory?.averages ?? [];
  const timestamps = memory?.timestamps ?? [];
  const maxStartIndex = Math.min(entries.length, averages.length, timestamps.length);
  const startIndex = clampInt(memory?.startIndex ?? 0, 0, maxStartIndex);

  return {
    entries,
    averages,
    timestamps,
    sum: memory?.sum ?? 0,
    startIndex,
    previousSampleIndex: memory?.previousSampleIndex,
    points: memory?.points ?? [],
  };
}

function normalizeKama(memory: RollingKamaMemory | undefined): RollingKamaMemory {
  return {
    sources: (Array.isArray(memory?.sources) ? memory.sources : []).filter(isPositiveFinite),
    ama: isPositiveFinite(memory?.ama ?? 0) ? memory?.ama : undefined,
  };
}

function normalizeCandleRange(
  memory: RollingCandleRangeMemory | undefined,
): RollingCandleRangeMemory {
  const memoryEntries = Array.isArray(memory?.entries) ? memory.entries : [];
  const memoryTimestamps = Array.isArray(memory?.timestamps) ? memory.timestamps : [];
  const maxLength = Math.min(
    memoryEntries.length,
    memoryTimestamps.length,
    CANDLE_RANGE_WINDOW_SIZE,
  );
  const entries = memoryEntries.slice(-maxLength);
  const timestamps = memoryTimestamps.slice(-maxLength);
  const sum = entries.reduce((total, value) => total + value, 0);
  const storedEntryOffset = memory?.entryOffset;
  const storedNextIndex = memory?.nextIndex;
  const rawEntryOffset =
    typeof storedEntryOffset === "number" && Number.isFinite(storedEntryOffset)
      ? Math.max(0, Math.floor(storedEntryOffset))
      : 0;
  const entryOffset = rawEntryOffset + Math.max(0, memoryEntries.length - maxLength);
  const nextIndex =
    typeof storedNextIndex === "number" && Number.isFinite(storedNextIndex)
      ? Math.max(entryOffset + entries.length, Math.floor(storedNextIndex))
      : entryOffset + entries.length;
  const maxCandidates = buildCandleRangeMaxCandidates(entries, entryOffset);

  return {
    entries,
    timestamps,
    sum,
    max: maxNumber(entries),
    entryOffset,
    nextIndex,
    maxCandidates,
    current: normalizeCandleRangeBucket(memory?.current),
    points: normalizeCandleRangePoints(memory?.points),
  };
}

function buildCandleRangeMaxCandidates(
  entries: number[],
  entryOffset: number,
): RollingCandleRangeMaxCandidate[] {
  const candidates: RollingCandleRangeMaxCandidate[] = [];
  for (let index = 0; index < entries.length; index += 1) {
    const value = entries[index];
    while (
      candidates.length > 0 &&
      candidates[candidates.length - 1].value <= value
    ) {
      candidates.pop();
    }
    candidates.push({ index: entryOffset + index, value });
  }
  return candidates;
}

function normalizeCandleRangeBucket(
  bucket: RollingCandleRangeBucket | undefined,
): RollingCandleRangeBucket | undefined {
  if (
    !bucket ||
    !isNonNegativeFinite(bucket.bucketStartSec) ||
    !isPositiveFinite(bucket.open) ||
    !isPositiveFinite(bucket.high) ||
    !isPositiveFinite(bucket.low) ||
    !isPositiveFinite(bucket.close)
  ) {
    return undefined;
  }

  return { ...bucket };
}

function normalizeCandleRangePoints(
  points: RollingCandleRangePoint[] | undefined,
): RollingCandleRangePoint[] {
  return (points ?? [])
    .filter(
      (point) =>
        Number.isFinite(point.avgPct) &&
        Number.isFinite(point.maxPct) &&
        Number.isFinite(point.currentPct) &&
        Number.isFinite(point.count),
    )
    .slice(-2)
    .map((point) => ({ ...point }));
}

function normalizePriceRange(
  memory: RollingPriceRangeMemory | undefined,
  window: RollingPriceRangeWindow,
  windowSec: number,
): RollingPriceRangeMemory {
  return {
    window,
    windowSec,
    bucketSec: PRICE_RANGE_BUCKET_SEC,
    current: normalizePriceRangeBucket(memory?.current),
    bucketStarts: normalizePriceRangeBucketStarts(memory),
    minCandidates: normalizePriceRangeBuckets(memory?.minCandidates),
    maxCandidates: normalizePriceRangeBuckets(memory?.maxCandidates),
    points: normalizePriceRangePoints(memory?.points, window, windowSec),
  };
}

function normalizePriceRangeBucketStarts(
  memory: RollingPriceRangeMemory | undefined,
): number[] {
  const starts = Array.isArray(memory?.bucketStarts)
    ? memory.bucketStarts
    : [
        ...(memory?.minCandidates ?? []).map((bucket) => bucket.bucketStartSec),
        ...(memory?.maxCandidates ?? []).map((bucket) => bucket.bucketStartSec),
        ...(memory?.current ? [memory.current.bucketStartSec] : []),
      ];

  return [...new Set(starts)]
    .filter(isNonNegativeFinite)
    .sort((left, right) => left - right);
}

function normalizePriceRangeBuckets(
  buckets: RollingPriceRangeBucket[] | undefined,
): RollingPriceRangeBucket[] {
  return (Array.isArray(buckets) ? buckets : [])
    .filter(
      (bucket) =>
        isNonNegativeFinite(bucket.bucketStartSec) &&
        isPositiveFinite(bucket.low) &&
        isPositiveFinite(bucket.high),
    )
    .map((bucket) => ({ ...bucket }));
}

function normalizePriceRangeBucket(
  bucket: RollingPriceRangeBucket | undefined,
): RollingPriceRangeBucket | undefined {
  return normalizePriceRangeBuckets(bucket ? [bucket] : [])[0];
}

function normalizePriceRangePoints(
  points: RollingPriceRangePoint[] | undefined,
  window: RollingPriceRangeWindow,
  windowSec: number,
): RollingPriceRangePoint[] {
  return (Array.isArray(points) ? points : [])
    .filter(
      (point) =>
        isPositiveFinite(point.minPrice) &&
        isPositiveFinite(point.maxPrice) &&
        Number.isFinite(point.rangePct) &&
        Number.isFinite(point.updatedAt),
    )
    .slice(-2)
    .map((point) => ({
      window,
      windowSec,
      minPrice: point.minPrice,
      maxPrice: point.maxPrice,
      rangePct: point.rangePct,
      updatedAt: point.updatedAt,
    }));
}

function ensureCandleRangeMemories(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): RollingCandleRangeMemory[] {
  if (
    memory.candleRanges?.length !== config.averagingRangesSec.length ||
    memory.candleRanges.some((range) => !isNormalizedCandleRangeMemory(range))
  ) {
    memory.candleRanges = normalizeCandleRangeList(memory.candleRanges, config);
  }
  return memory.candleRanges;
}

function isNormalizedCandleRangeMemory(
  memory: RollingCandleRangeMemory | undefined,
): boolean {
  return Boolean(
    memory &&
      Array.isArray(memory.entries) &&
      Array.isArray(memory.timestamps) &&
      Array.isArray(memory.maxCandidates) &&
      Number.isFinite(memory.sum) &&
      Number.isFinite(memory.max) &&
      Number.isFinite(memory.entryOffset) &&
      Number.isFinite(memory.nextIndex),
  );
}

function ensurePriceRangeMemories(
  memory: LegacyValleyPeakMemory,
): RollingPriceRangeMemory[] {
  if (memory.priceRanges?.length !== PRICE_RANGE_WINDOWS.length) {
    memory.priceRanges = normalizePriceRangeList(memory.priceRanges);
  }
  return memory.priceRanges;
}

function gaussian(x: number, mu: number, sigma: number): number {
  const w = (x - mu) / sigma;
  return Math.exp(-(w ** 2) / 2);
}

function sigmoid(value: number, slope: number): number {
  return 1 / (
    1 +
    Math.exp(
      -clamp(value * slope, -MAX_TREND_SIGMA_EXPONENT, MAX_TREND_SIGMA_EXPONENT),
    )
  );
}

function interpolate(nextValue: number, prevValue: number, fraction: number): number {
  return nextValue * (1 - fraction) + prevValue * fraction;
}

function compactRollingAverage(memory: RollingAverageMemory): void {
  const startIndex = memory.startIndex ?? 0;
  if (
    startIndex < ROLLING_AVERAGE_COMPACT_EXPIRED ||
    startIndex < memory.timestamps.length / 2
  ) {
    return;
  }

  memory.entries.splice(0, startIndex);
  memory.averages.splice(0, startIndex);
  memory.timestamps.splice(0, startIndex);
  if (memory.previousSampleIndex !== undefined) {
    memory.previousSampleIndex = Math.max(0, memory.previousSampleIndex - startIndex);
  }
  memory.startIndex = 0;
}

function compactKama(memory: RollingKamaMemory, erLen: number): void {
  const keep = Math.max(2, erLen + 1);
  if (memory.sources.length <= keep * 2) {
    return;
  }
  memory.sources.splice(0, memory.sources.length - keep);
}

function binarySearchLeft(
  values: number[],
  target: number,
  low = 0,
  high = values.length,
): number {
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (values[mid] < target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return 0;
  }

  return Math.min(Math.max(value, min), max);
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }

  return Math.max(min, Math.min(max, Math.round(value)));
}

function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function isNonNegativeFinite(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

function maxNumber(values: number[]): number {
  return values.length > 0 ? Math.max(...values) : 0;
}

function normalizeRelativeRates(
  values: number[],
  length: number,
  fallback: number,
): number[] {
  return padNumbers(values, length, fallback).map((value) =>
    normalizeRelativeRate(value, fallback),
  );
}

// todo: bake into constants
function normalizeRelativeRate(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }

  if (value > MAX_REASONABLE_RELATIVE_RATE_PER_SEC) {
    return value / BTC_SCALE_REFERENCE_PRICE;
  }

  return value;
}

function normalizeExtremaSignalTiming(
  value: LegacyExtremaSignalTiming | undefined,
): LegacyExtremaSignalTiming {
  return value === "end" ? "end" : "start";
}

function normalizeMovingAverageType(
  value: LegacyMovingAverageType | undefined,
): LegacyMovingAverageType {
  return value === "ema" ? "ema" : "sma";
}

function padNumbers(values: number[], length: number, fallback: number): number[] {
  const next = values.slice(0, length);
  while (next.length < length) {
    next.push(next.at(-1) ?? fallback);
  }

  return next;
}
