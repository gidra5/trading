import type {
  Candle,
  LegacyValleyPeakCheckDebug,
  LegacyValleyPeakConfig,
  LegacyValleyPeakDebugSnapshot,
  LegacyValleyPeakMemory,
  LegacyMarketStateDebug,
  RollingCandleRangeBucket,
  RollingCandleRangeMemory,
  RollingCandleRangeMaxCandidate,
  RollingCandleRangePoint,
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

export type LegacyValleyPeakDecision =
  | { signal: "hold" }
  | { signal: "buy"; reason: string; quoteSize: number; coverQuantity: number }
  | { signal: "sell"; reason: string; quantity: number; quoteSize: number };

export const legacyValleyPeakStrictSymmetricConfig: LegacyValleyPeakConfig = {
  averagingRangesSec: [1, 60, 600, 1800, 3600, 3600 * 4, 3600 * 12],
  rateRatios: [0.5, 0.5, 0.1, 0.05, 0.01, 0.01, 0.001],
  relativeRateEnabled: true,
  rateThresholdsLow: [0.25, 0.25, 0.25, 0.25, 0.15, 0.05, 0.05],
  rateThresholdsHigh: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
  buyDataIndex: 1,
  sellDataIndex: 1,
  buyConfirmationOffsets: [2, 1],
  sellConfirmationOffsets: [2, 1],
  saturationSec: 3600,
  buySpendRate: 1,
  sellAmountRate: 1,
  buySigma: 0.3,
  sellSigma: 0.1,
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
  exitGridSellFraction: 0.35,
  exitGridMinProfitBps: 20,
  exitGridResetBps: 10,
  exitGridPositionMode: "per-lot",
  exitGridResetMode: "filled-grid",
  rangeLeverageEnabled: true,
  leverageLongTermRangeWindow: "1y",
  leverageRangeEdgeFraction: 0.2,
  leverageLongTermRangePaddingPct: 3,
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
  };

  const rangeCount = Math.max(1, config.averagingRangesSec.length);
  config.rateRatios = padNumbers(config.rateRatios, rangeCount, 0.5);
  config.relativeRateEnabled = config.relativeRateEnabled === true;
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
  config.saturationSec = Math.max(0, config.saturationSec);
  config.buySpendRate = Math.max(0, config.buySpendRate);
  config.sellAmountRate = Math.max(0, config.sellAmountRate);
  config.buySigma = config.relativeRateEnabled
    ? Math.max(0.00000001, normalizeRelativeRate(config.buySigma, 0.000003))
    : Math.max(0.000001, config.buySigma);
  config.sellSigma = config.relativeRateEnabled
    ? Math.max(0.00000001, normalizeRelativeRate(config.sellSigma, 0.000001))
    : Math.max(0.000001, config.sellSigma);
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

  return config;
}

export function createLegacyValleyPeakMemory(
  config: LegacyValleyPeakConfig,
): LegacyValleyPeakMemory {
  const averages = config.averagingRangesSec.map(createRollingAverageMemory);
  return {
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
    );
    updateRollingCandleRange(candleRanges[index], rangeSec, input, tsSec);
  }

  if (input.eventTime - memory.startedAt < config.saturationSec * 1000) {
    return { signal: "hold" };
  }

  const feeAdjustedBuyRate = input.price / (1 - input.feeRate);
  const feeAdjustedSellRate = input.price * (1 - input.feeRate);

  if (shouldBuy(memory, config)) {
    const quoteSize = buyQuoteSize(memory, config, input, feeAdjustedBuyRate);
    const coverQuantity = buyCoverQuantity(memory, config, input, feeAdjustedBuyRate);
    if (
      quoteSize >= config.minTradeQuote ||
      coverQuantity * feeAdjustedBuyRate >= config.minTradeQuote
    ) {
      return {
        signal: "buy",
        reason: "legacy valley detected",
        quoteSize,
        coverQuantity,
      };
    }
  }

  if (shouldSell(memory, config)) {
    const quantity = sellQuantity(memory, config, input, feeAdjustedSellRate);
    const quoteSize = shortSellQuoteSize(memory, config, input, feeAdjustedSellRate);
    if (
      quantity * feeAdjustedSellRate >= config.minTradeQuote ||
      quoteSize >= config.minTradeQuote
    ) {
      return {
        signal: "sell",
        reason: "legacy peak detected",
        quantity,
        quoteSize,
      };
    }
  }

  return { signal: "hold" };
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
    signal: decision.signal,
    reason: decision.signal === "hold" ? undefined : decision.reason,
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
        ),
        sellConfirmation: config.sellConfirmationOffsets.some(
          (offset) => index === config.sellDataIndex + offset,
        ),
        valley: isValley(memory.buyAverages[index]),
        peak: isPeak(memory.sellAverages[index]),
      };
    }),
    candleRanges: config.averagingRangesSec.map((windowSec, index) => {
      const point = memory.candleRanges[index]?.points.at(-1);
      return {
        index,
        windowSec,
        avgPct: point?.avgPct,
        maxPct: point?.maxPct,
        currentPct: point?.currentPct,
        count: point?.count,
      };
    }),
    priceRanges: memory.priceRanges.map((range) => {
      const point = range.points.at(-1);
      return {
        window: range.window,
        windowSec: range.windowSec,
        minPrice: point?.minPrice,
        maxPrice: point?.maxPrice,
        rangePct: point?.rangePct,
        updatedAt: point?.updatedAt,
      };
    }),
    buyCheck: buildLegacyValleyPeakCheckDebug(
      "buy",
      memory,
      config,
      input,
      feeAdjustedBuyRate,
    ),
    sellCheck: buildLegacyValleyPeakCheckDebug(
      "sell",
      memory,
      config,
      input,
      feeAdjustedSellRate,
    ),
  };
}

function shouldBuy(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): boolean {
  const primary = memory.buyAverages[config.buyDataIndex];
  if (!isValley(primary)) {
    return false;
  }

  for (const offset of config.buyConfirmationOffsets) {
    const confirmation = memory.buyAverages[config.buyDataIndex + offset];
    const confirmationPoint = latestPoint(confirmation);
    if (confirmationPoint && confirmationPoint.rateClamped <= 0) {
      return false;
    }
  }

  return true;
}

function shouldSell(
  memory: LegacyValleyPeakMemory,
  config: LegacyValleyPeakConfig,
): boolean {
  const primary = memory.sellAverages[config.sellDataIndex];
  if (!isPeak(primary)) {
    return false;
  }

  for (const offset of config.sellConfirmationOffsets) {
    const confirmation = memory.sellAverages[config.sellDataIndex + offset];
    const confirmationPoint = latestPoint(confirmation);
    if (confirmationPoint && confirmationPoint.rateClamped >= 0) {
      return false;
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
): LegacyValleyPeakCheckDebug {
  const primaryIndex = side === "buy" ? config.buyDataIndex : config.sellDataIndex;
  const primary = side === "buy"
    ? memory.buyAverages[primaryIndex]
    : memory.sellAverages[primaryIndex];
  const primaryPoint = latestPoint(primary);
  const primaryShape = !primaryPoint
    ? "missing"
    : side === "buy" && isValley(primary)
      ? "valley"
      : side === "sell" && isPeak(primary)
        ? "peak"
        : "flat";
  const offsets =
    side === "buy" ? config.buyConfirmationOffsets : config.sellConfirmationOffsets;
  const confirmations = offsets.map((offset) => {
    const index = primaryIndex + offset;
    const confirmation = side === "buy"
      ? memory.buyAverages[index]
      : memory.sellAverages[index];
    const point = latestPoint(confirmation);
    const expected: "positive" | "negative" = side === "buy" ? "positive" : "negative";
    return {
      index,
      windowSec: config.averagingRangesSec[index],
      rateClamped: point?.rateClamped,
      expected,
      passed:
        !point ||
        (side === "buy" ? point.rateClamped > 0 : point.rateClamped < 0),
    };
  });

  if (side === "buy") {
    return {
      side,
      passed: shouldBuy(memory, config),
      primaryIndex,
      primaryWindowSec: config.averagingRangesSec[primaryIndex],
      primaryRate: primaryPoint?.rate,
      primaryRateClamped: primaryPoint?.rateClamped,
      primaryShape,
      confirmations,
      quoteSize: buyQuoteSize(memory, config, input, rate),
      coverQuantity: buyCoverQuantity(memory, config, input, rate),
      minTradeQuote: config.minTradeQuote,
    };
  }

  return {
    side,
    passed: shouldSell(memory, config),
    primaryIndex,
    primaryWindowSec: config.averagingRangesSec[primaryIndex],
    primaryRate: primaryPoint?.rate,
    primaryRateClamped: primaryPoint?.rateClamped,
    primaryShape,
    confirmations,
    quantity: sellQuantity(memory, config, input, rate),
    quoteSize: shortSellQuoteSize(memory, config, input, rate),
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
  const desired =
    buyingPowerQuote *
    config.buySpendRate *
    gaussian(derivative, 0, config.buySigma);

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
  const desired =
    shortBaseFree * config.buySpendRate * gaussian(derivative, 0, config.buySigma);
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
  const desired =
    input.baseFree * config.sellAmountRate * gaussian(derivative, 0, config.sellSigma);
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
  const desired =
    sellingPowerQuote *
    config.sellAmountRate *
    gaussian(derivative, 0, config.sellSigma);

  return clamp(
    desired,
    config.minTradeQuote,
    Math.min(config.maxTradeQuote, sellingPowerQuote, rate * 10_000),
  );
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
): void {
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
  let rateClamped = 0;
  if (derivative >= thresholdHigh) {
    rateClamped = derivative;
  } else if (derivative <= -thresholdLow) {
    rateClamped = derivative;
  }

  recordPoint(memory, avg, derivative, rateClamped);
  compactRollingAverage(memory);
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

function isValley(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(latest && previous && latest.rateClamped >= 0 && previous.rateClamped < 0);
}

function isPeak(memory: RollingAverageMemory | undefined): boolean {
  const latest = latestPoint(memory);
  const previous = previousPoint(memory);
  return Boolean(latest && previous && latest.rateClamped <= 0 && previous.rateClamped > 0);
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
    minCandidates: normalizePriceRangeBuckets(memory?.minCandidates),
    maxCandidates: normalizePriceRangeBuckets(memory?.maxCandidates),
    points: normalizePriceRangePoints(memory?.points, window, windowSec),
  };
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

function normalizeRelativeRate(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }

  if (value > MAX_REASONABLE_RELATIVE_RATE_PER_SEC) {
    return value / BTC_SCALE_REFERENCE_PRICE;
  }

  return value;
}

function padNumbers(values: number[], length: number, fallback: number): number[] {
  const next = values.slice(0, length);
  while (next.length < length) {
    next.push(next.at(-1) ?? fallback);
  }

  return next;
}
