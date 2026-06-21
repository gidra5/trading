import type {
  BotSignal,
  MasterAdaptiveConfig,
  MeanReversionConfig,
  StrategyConfig,
  TrendFollowingConfig,
  VolatilityBreakoutConfig,
} from "./types.js";

export type DirectionalSide = "long" | "short" | "flat";

export interface DirectionalStrategyInput {
  prices: number[];
  currentSide: DirectionalSide;
  stats?: DirectionalRuntimeStats;
}

export interface DirectionalRuntimeStats {
  priceCount: number;
  previousPrice?: number;
  meanReversionPriceWindow: RollingValueWindow;
  trendVolatilityWindow: RollingValueWindow;
  masterVolatilityWindow: RollingValueWindow;
  breakoutRangeWindow: RollingRangeWindow;
}

interface RollingValueWindow {
  window: number;
  values: number[];
  cursor: number;
  count: number;
  sum: number;
  sumSquares: number;
}

interface RollingRangeWindow {
  window: number;
  index: number;
  maxIndexes: number[];
  maxValues: number[];
  maxHead: number;
  minIndexes: number[];
  minValues: number[];
  minHead: number;
}

export type DirectionalDecision =
  | {
      action: "hold";
      signal: "hold";
      reason: string;
    }
  | {
      action: "rebalance";
      signal: BotSignal;
      targetExposurePct: number;
      reason: string;
    };

export const defaultTrendFollowingConfig: TrendFollowingConfig = {
  fastWindow: 24,
  slowWindow: 144,
  volatilityWindow: 96,
  entryThresholdBps: 24,
  exitThresholdBps: 8,
  targetExposurePct: 0.35,
};

export const defaultVolatilityBreakoutConfig: VolatilityBreakoutConfig = {
  lookbackWindow: 288,
  breakoutThresholdBps: 18,
  exitThresholdBps: 6,
  targetExposurePct: 0.3,
};

export const defaultMeanReversionConfig: MeanReversionConfig = {
  window: 96,
  trendWindow: 288,
  entryZScore: 2.4,
  exitZScore: 0.25,
  maxTrendBps: 45,
  targetExposurePct: 0.25,
};

export const defaultMasterAdaptiveConfig: MasterAdaptiveConfig = {
  trendWeight: 1.2,
  breakoutWeight: 1,
  reversionWeight: 0.7,
  minConsensusScore: 0.25,
  disagreementExposureScale: 0.5,
  targetExposurePct: 0.4,
  volatilityWindow: 720,
  highVolatilityBps: 35,
  highVolatilityExposureScale: 0.65,
};

export function createTrendFollowingConfig(
  overrides: Partial<TrendFollowingConfig> = {},
): TrendFollowingConfig {
  const config = {
    ...defaultTrendFollowingConfig,
    ...overrides,
  };

  config.fastWindow = clampInt(config.fastWindow, 2, 10_000);
  config.slowWindow = clampInt(config.slowWindow, config.fastWindow + 1, 20_000);
  config.volatilityWindow = clampInt(config.volatilityWindow, 2, 20_000);
  config.entryThresholdBps = cleanNonNegative(config.entryThresholdBps);
  config.exitThresholdBps = clamp(
    cleanNonNegative(config.exitThresholdBps),
    0,
    config.entryThresholdBps,
  );
  config.targetExposurePct = clamp(cleanNumber(config.targetExposurePct), 0, 1);

  return config;
}

export function createVolatilityBreakoutConfig(
  overrides: Partial<VolatilityBreakoutConfig> = {},
): VolatilityBreakoutConfig {
  const config = {
    ...defaultVolatilityBreakoutConfig,
    ...overrides,
  };

  config.lookbackWindow = clampInt(config.lookbackWindow, 2, 20_000);
  config.breakoutThresholdBps = cleanNonNegative(config.breakoutThresholdBps);
  config.exitThresholdBps = cleanNonNegative(config.exitThresholdBps);
  config.targetExposurePct = clamp(cleanNumber(config.targetExposurePct), 0, 1);

  return config;
}

export function createMeanReversionConfig(
  overrides: Partial<MeanReversionConfig> = {},
): MeanReversionConfig {
  const config = {
    ...defaultMeanReversionConfig,
    ...overrides,
  };

  config.window = clampInt(config.window, 3, 20_000);
  config.trendWindow = clampInt(config.trendWindow, config.window, 40_000);
  config.entryZScore = Math.max(0.01, cleanNumber(config.entryZScore));
  config.exitZScore = clamp(cleanNonNegative(config.exitZScore), 0, config.entryZScore);
  config.maxTrendBps = cleanNonNegative(config.maxTrendBps);
  config.targetExposurePct = clamp(cleanNumber(config.targetExposurePct), 0, 1);

  return config;
}

export function createMasterAdaptiveConfig(
  overrides: Partial<MasterAdaptiveConfig> = {},
): MasterAdaptiveConfig {
  const config = {
    ...defaultMasterAdaptiveConfig,
    ...overrides,
  };

  config.trendWeight = cleanNonNegative(config.trendWeight);
  config.breakoutWeight = cleanNonNegative(config.breakoutWeight);
  config.reversionWeight = cleanNonNegative(config.reversionWeight);
  config.minConsensusScore = clamp(cleanNumber(config.minConsensusScore), 0, 1);
  config.disagreementExposureScale = clamp(cleanNumber(config.disagreementExposureScale), 0, 1);
  config.targetExposurePct = clamp(cleanNumber(config.targetExposurePct), 0, 1);
  config.volatilityWindow = clampInt(config.volatilityWindow, 2, 40_000);
  config.highVolatilityBps = cleanNonNegative(config.highVolatilityBps);
  config.highVolatilityExposureScale = clamp(cleanNumber(config.highVolatilityExposureScale), 0, 1);

  return config;
}

export function createDirectionalRuntimeStats(
  config: StrategyConfig,
  initialPrices: readonly number[] = [],
): DirectionalRuntimeStats {
  const trendVolatilityWindow = createRollingValueWindow(
    config.trendFollowing.volatilityWindow,
  );
  const masterVolatilityWindow =
    config.masterAdaptive.volatilityWindow === config.trendFollowing.volatilityWindow
      ? trendVolatilityWindow
      : createRollingValueWindow(config.masterAdaptive.volatilityWindow);
  const stats: DirectionalRuntimeStats = {
    priceCount: 0,
    meanReversionPriceWindow: createRollingValueWindow(config.meanReversion.window),
    trendVolatilityWindow,
    masterVolatilityWindow,
    breakoutRangeWindow: createRollingRangeWindow(config.volatilityBreakout.lookbackWindow),
  };

  for (const price of initialPrices) {
    recordDirectionalRuntimePrice(stats, price);
  }

  return stats;
}

export function recordDirectionalRuntimePrice(
  stats: DirectionalRuntimeStats,
  price: number,
): void {
  if (!Number.isFinite(price) || price <= 0) {
    return;
  }

  stats.priceCount += 1;
  recordRollingValue(stats.meanReversionPriceWindow, price);
  recordRollingRange(stats.breakoutRangeWindow, price);

  const previousPrice = stats.previousPrice;
  if (previousPrice !== undefined && previousPrice > 0) {
    const returnBps = ((price - previousPrice) / previousPrice) * 10_000;
    recordRollingValue(stats.trendVolatilityWindow, returnBps);
    if (stats.masterVolatilityWindow !== stats.trendVolatilityWindow) {
      recordRollingValue(stats.masterVolatilityWindow, returnBps);
    }
  }
  stats.previousPrice = price;
}

export function evaluateTrendFollowing(
  config: TrendFollowingConfig,
  input: DirectionalStrategyInput,
): DirectionalDecision {
  const prices = input.prices;
  const requiredPrices = Math.max(config.slowWindow, config.volatilityWindow) + 1;
  if (prices.length < requiredPrices) {
    return hold("trend following warming up");
  }

  const currentPrice = prices[prices.length - 1];
  const fastBps = windowReturnBps(prices, config.fastWindow);
  const slowBps = windowReturnBps(prices, config.slowWindow);
  const volatilityBps = rollingReturnVolatilityBps(
    prices,
    config.volatilityWindow,
    input.stats,
  );
  const entryBps = volatilityAdjustedThreshold(config.entryThresholdBps, volatilityBps);
  const exitBps = config.exitThresholdBps;

  if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
    return hold("trend following invalid price");
  }

  if (fastBps >= entryBps && slowBps >= exitBps) {
    return target(
      "long",
      config.targetExposurePct * confidenceFromMove(fastBps, entryBps),
      `trend following long: fast ${formatBps(fastBps)}bps, slow ${formatBps(slowBps)}bps`,
    );
  }

  if (fastBps <= -entryBps && slowBps <= -exitBps) {
    return target(
      "short",
      config.targetExposurePct * confidenceFromMove(fastBps, entryBps),
      `trend following short: fast ${formatBps(fastBps)}bps, slow ${formatBps(slowBps)}bps`,
    );
  }

  if (input.currentSide === "long" && (fastBps <= exitBps || slowBps < -exitBps)) {
    return target("flat", 0, "trend following long exit");
  }

  if (input.currentSide === "short" && (fastBps >= -exitBps || slowBps > exitBps)) {
    return target("flat", 0, "trend following short exit");
  }

  return hold("trend following hold");
}

export function evaluateVolatilityBreakout(
  config: VolatilityBreakoutConfig,
  input: DirectionalStrategyInput,
): DirectionalDecision {
  const prices = input.prices;
  if (prices.length < config.lookbackWindow + 1) {
    return hold("volatility breakout warming up");
  }

  const currentPrice = prices[prices.length - 1];
  const historyStart = prices.length - config.lookbackWindow - 1;
  const historyEnd = prices.length - 1;
  const rangeHigh =
    previousRangeHigh(input.stats, config.lookbackWindow) ??
    maxRange(prices, historyStart, historyEnd);
  const rangeLow =
    previousRangeLow(input.stats, config.lookbackWindow) ??
    minRange(prices, historyStart, historyEnd);
  const midpoint = (rangeHigh + rangeLow) / 2;
  const breakoutRate = config.breakoutThresholdBps / 10_000;
  const exitRate = config.exitThresholdBps / 10_000;

  if (currentPrice >= rangeHigh * (1 + breakoutRate)) {
    const moveBps = ((currentPrice - rangeHigh) / rangeHigh) * 10_000;
    return target(
      "long",
      config.targetExposurePct * confidenceFromMove(moveBps, config.breakoutThresholdBps),
      `volatility breakout long above ${roundQuote(rangeHigh)}`,
    );
  }

  if (currentPrice <= rangeLow * (1 - breakoutRate)) {
    const moveBps = ((rangeLow - currentPrice) / rangeLow) * 10_000;
    return target(
      "short",
      config.targetExposurePct * confidenceFromMove(moveBps, config.breakoutThresholdBps),
      `volatility breakout short below ${roundQuote(rangeLow)}`,
    );
  }

  if (input.currentSide === "long" && currentPrice <= midpoint * (1 - exitRate)) {
    return target("flat", 0, "volatility breakout long exit");
  }

  if (input.currentSide === "short" && currentPrice >= midpoint * (1 + exitRate)) {
    return target("flat", 0, "volatility breakout short exit");
  }

  return hold("volatility breakout hold");
}

export function evaluateMeanReversion(
  config: MeanReversionConfig,
  input: DirectionalStrategyInput,
): DirectionalDecision {
  const prices = input.prices;
  if (prices.length < config.trendWindow + 1) {
    return hold("mean reversion warming up");
  }

  const currentPrice = prices[prices.length - 1];
  const windowStart = prices.length - config.window;
  const cachedWindow = priceStatsWindow(input.stats, config.window);
  const cachedMean = valueWindowMean(cachedWindow);
  const mean = cachedMean ?? averageRange(prices, windowStart, prices.length);
  const stdDev =
    valueWindowStdDev(cachedWindow) ??
    standardDeviationRange(prices, windowStart, prices.length, mean);
  if (stdDev <= 0) {
    return hold("mean reversion flat volatility");
  }

  const zScore = (currentPrice - mean) / stdDev;
  const trendBps = windowReturnBps(prices, config.trendWindow);

  if (zScore <= -config.entryZScore && trendBps >= -config.maxTrendBps) {
    return target(
      "long",
      config.targetExposurePct * confidenceFromZScore(zScore, config.entryZScore),
      `mean reversion long: z ${formatZ(zScore)}, trend ${formatBps(trendBps)}bps`,
    );
  }

  if (zScore >= config.entryZScore && trendBps <= config.maxTrendBps) {
    return target(
      "short",
      config.targetExposurePct * confidenceFromZScore(zScore, config.entryZScore),
      `mean reversion short: z ${formatZ(zScore)}, trend ${formatBps(trendBps)}bps`,
    );
  }

  if (input.currentSide !== "flat" && Math.abs(zScore) <= config.exitZScore) {
    return target("flat", 0, "mean reversion exit");
  }

  return hold("mean reversion hold");
}

export function evaluateMasterAdaptive(
  config: MasterAdaptiveConfig,
  children: {
    trendFollowing: TrendFollowingConfig;
    volatilityBreakout: VolatilityBreakoutConfig;
    meanReversion: MeanReversionConfig;
  },
  input: DirectionalStrategyInput,
): DirectionalDecision {
  const prices = input.prices;
  let totalWeight = 0;
  let weightedScore = 0;
  let trendScore = 0;
  let breakoutScore = 0;
  let reversionScore = 0;

  if (config.trendWeight > 0) {
    trendScore = trendFollowingConsensusScore(children.trendFollowing, input);
    totalWeight += config.trendWeight;
    weightedScore += config.trendWeight * trendScore;
  }
  if (config.breakoutWeight > 0) {
    breakoutScore = volatilityBreakoutConsensusScore(
      children.volatilityBreakout,
      input,
    );
    totalWeight += config.breakoutWeight;
    weightedScore += config.breakoutWeight * breakoutScore;
  }
  if (config.reversionWeight > 0) {
    reversionScore = meanReversionConsensusScore(children.meanReversion, input);
    totalWeight += config.reversionWeight;
    weightedScore += config.reversionWeight * reversionScore;
  }

  if (totalWeight <= 0) {
    return target("flat", 0, "master adaptive has no enabled child signals");
  }

  weightedScore /= totalWeight;
  const hasLong = trendScore > 0 || breakoutScore > 0 || reversionScore > 0;
  const hasShort = trendScore < 0 || breakoutScore < 0 || reversionScore < 0;
  const disagreementScale = hasLong && hasShort ? config.disagreementExposureScale : 1;
  const volatilityBps = rollingReturnVolatilityBps(
    prices,
    config.volatilityWindow,
    input.stats,
  );
  const volatilityScale =
    config.highVolatilityBps > 0 && volatilityBps >= config.highVolatilityBps
      ? config.highVolatilityExposureScale
      : 1;
  const finalScore = clamp(weightedScore * disagreementScale * volatilityScale, -1, 1);
  const absScore = Math.abs(finalScore);

  if (absScore < config.minConsensusScore) {
    return target(
      "flat",
      0,
      `master adaptive flat: score ${formatScore(finalScore)}, vol ${formatBps(volatilityBps)}bps`,
    );
  }

  const side: DirectionalSide = finalScore > 0 ? "long" : "short";
  const activeSignals = formatActiveMasterSignals(
    trendScore,
    breakoutScore,
    reversionScore,
  );

  return target(
    side,
    config.targetExposurePct * absScore,
    `master adaptive ${side}: score ${formatScore(finalScore)}, vol ${formatBps(volatilityBps)}bps${
      activeSignals ? `, ${activeSignals}` : ""
    }`,
  );
}

function trendFollowingConsensusScore(
  config: TrendFollowingConfig,
  input: DirectionalStrategyInput,
): number {
  const prices = input.prices;
  const requiredPrices = Math.max(config.slowWindow, config.volatilityWindow) + 1;
  if (prices.length < requiredPrices || config.targetExposurePct <= 0) {
    return 0;
  }

  const currentPrice = prices[prices.length - 1];
  if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
    return 0;
  }

  const fastBps = windowReturnBps(prices, config.fastWindow);
  const slowBps = windowReturnBps(prices, config.slowWindow);
  const volatilityBps = rollingReturnVolatilityBps(
    prices,
    config.volatilityWindow,
    input.stats,
  );
  const entryBps = volatilityAdjustedThreshold(config.entryThresholdBps, volatilityBps);
  const exitBps = config.exitThresholdBps;

  if (fastBps >= entryBps && slowBps >= exitBps) {
    return childConsensusScore(
      config.targetExposurePct,
      confidenceFromMove(fastBps, entryBps),
    );
  }

  if (fastBps <= -entryBps && slowBps <= -exitBps) {
    return childConsensusScore(
      config.targetExposurePct,
      -confidenceFromMove(fastBps, entryBps),
    );
  }

  return 0;
}

function volatilityBreakoutConsensusScore(
  config: VolatilityBreakoutConfig,
  input: DirectionalStrategyInput,
): number {
  const prices = input.prices;
  if (prices.length < config.lookbackWindow + 1 || config.targetExposurePct <= 0) {
    return 0;
  }

  const currentPrice = prices[prices.length - 1];
  const historyStart = prices.length - config.lookbackWindow - 1;
  const historyEnd = prices.length - 1;
  const rangeHigh =
    previousRangeHigh(input.stats, config.lookbackWindow) ??
    maxRange(prices, historyStart, historyEnd);
  const rangeLow =
    previousRangeLow(input.stats, config.lookbackWindow) ??
    minRange(prices, historyStart, historyEnd);
  const breakoutRate = config.breakoutThresholdBps / 10_000;

  if (currentPrice >= rangeHigh * (1 + breakoutRate)) {
    const moveBps = ((currentPrice - rangeHigh) / rangeHigh) * 10_000;
    return childConsensusScore(
      config.targetExposurePct,
      confidenceFromMove(moveBps, config.breakoutThresholdBps),
    );
  }

  if (currentPrice <= rangeLow * (1 - breakoutRate)) {
    const moveBps = ((rangeLow - currentPrice) / rangeLow) * 10_000;
    return childConsensusScore(
      config.targetExposurePct,
      -confidenceFromMove(moveBps, config.breakoutThresholdBps),
    );
  }

  return 0;
}

function meanReversionConsensusScore(
  config: MeanReversionConfig,
  input: DirectionalStrategyInput,
): number {
  const prices = input.prices;
  if (prices.length < config.trendWindow + 1 || config.targetExposurePct <= 0) {
    return 0;
  }

  const currentPrice = prices[prices.length - 1];
  const windowStart = prices.length - config.window;
  const cachedWindow = priceStatsWindow(input.stats, config.window);
  const cachedMean = valueWindowMean(cachedWindow);
  const mean = cachedMean ?? averageRange(prices, windowStart, prices.length);
  const stdDev =
    valueWindowStdDev(cachedWindow) ??
    standardDeviationRange(prices, windowStart, prices.length, mean);
  if (stdDev <= 0) {
    return 0;
  }

  const zScore = (currentPrice - mean) / stdDev;
  const trendBps = windowReturnBps(prices, config.trendWindow);

  if (zScore <= -config.entryZScore && trendBps >= -config.maxTrendBps) {
    return childConsensusScore(
      config.targetExposurePct,
      confidenceFromZScore(zScore, config.entryZScore),
    );
  }

  if (zScore >= config.entryZScore && trendBps <= config.maxTrendBps) {
    return childConsensusScore(
      config.targetExposurePct,
      -confidenceFromZScore(zScore, config.entryZScore),
    );
  }

  return 0;
}

function childConsensusScore(maxTargetExposurePct: number, score: number): number {
  if (maxTargetExposurePct <= 0) {
    return 0;
  }

  return clamp(
    (maxTargetExposurePct * score) / Math.max(maxTargetExposurePct, 0.01),
    -1,
    1,
  );
}

function formatActiveMasterSignals(
  trendScore: number,
  breakoutScore: number,
  reversionScore: number,
): string {
  let summary = "";
  if (trendScore !== 0) {
    summary = `trend ${formatScore(trendScore)}`;
  }
  if (breakoutScore !== 0) {
    summary = appendSummary(summary, `breakout ${formatScore(breakoutScore)}`);
  }
  if (reversionScore !== 0) {
    summary = appendSummary(summary, `reversion ${formatScore(reversionScore)}`);
  }
  return summary;
}

function appendSummary(summary: string, next: string): string {
  return summary ? `${summary}, ${next}` : next;
}

function hold(reason: string): DirectionalDecision {
  return {
    action: "hold",
    signal: "hold",
    reason,
  };
}

function target(side: DirectionalSide, exposurePct: number, reason: string): DirectionalDecision {
  const signedExposurePct =
    side === "long" ? exposurePct : side === "short" ? -exposurePct : 0;

  return {
    action: "rebalance",
    signal: side === "long" ? "buy" : side === "short" ? "sell" : "hold",
    targetExposurePct: clamp(signedExposurePct, -1, 1),
    reason,
  };
}

function windowReturnBps(prices: number[], window: number): number {
  const current = prices[prices.length - 1];
  const previous = prices[Math.max(0, prices.length - 1 - window)];
  if (!Number.isFinite(previous) || previous <= 0 || !Number.isFinite(current)) {
    return 0;
  }

  return ((current - previous) / previous) * 10_000;
}

function rollingReturnVolatilityBps(
  prices: number[],
  window: number,
  stats?: DirectionalRuntimeStats,
): number {
  const cachedStdDev = valueWindowStdDev(returnStatsWindow(stats, window));
  if (cachedStdDev !== undefined) {
    return cachedStdDev;
  }

  const start = Math.max(1, prices.length - window);
  let count = 0;
  let sum = 0;

  for (let index = start; index < prices.length; index += 1) {
    const previous = prices[index - 1];
    const current = prices[index];
    if (previous > 0 && current > 0) {
      const value = ((current - previous) / previous) * 10_000;
      count += 1;
      sum += value;
    }
  }

  if (count < 2) {
    return 0;
  }

  const mean = sum / count;
  let variance = 0;
  for (let index = start; index < prices.length; index += 1) {
    const previous = prices[index - 1];
    const current = prices[index];
    if (previous > 0 && current > 0) {
      const value = ((current - previous) / previous) * 10_000;
      variance += (value - mean) ** 2;
    }
  }

  return Math.sqrt(variance / count);
}

function valueWindowMean(window: RollingValueWindow | undefined): number | undefined {
  if (!window || window.count < 2) {
    return undefined;
  }

  return window.sum / window.count;
}

function valueWindowStdDev(window: RollingValueWindow | undefined): number | undefined {
  if (!window || window.count < 2) {
    return undefined;
  }

  const mean = window.sum / window.count;
  const variance = Math.max(0, window.sumSquares / window.count - mean * mean);
  return Math.sqrt(variance);
}

function previousRangeHigh(
  stats: DirectionalRuntimeStats | undefined,
  window: number,
): number | undefined {
  return previousRangeValue(rangeStatsWindow(stats, window), "max");
}

function previousRangeLow(
  stats: DirectionalRuntimeStats | undefined,
  window: number,
): number | undefined {
  return previousRangeValue(rangeStatsWindow(stats, window), "min");
}

function priceStatsWindow(
  stats: DirectionalRuntimeStats | undefined,
  window: number,
): RollingValueWindow | undefined {
  const candidate = stats?.meanReversionPriceWindow;
  return candidate?.window === window ? candidate : undefined;
}

function returnStatsWindow(
  stats: DirectionalRuntimeStats | undefined,
  window: number,
): RollingValueWindow | undefined {
  if (!stats) {
    return undefined;
  }
  if (stats.trendVolatilityWindow.window === window) {
    return stats.trendVolatilityWindow;
  }
  if (stats.masterVolatilityWindow.window === window) {
    return stats.masterVolatilityWindow;
  }
  return undefined;
}

function rangeStatsWindow(
  stats: DirectionalRuntimeStats | undefined,
  window: number,
): RollingRangeWindow | undefined {
  const candidate = stats?.breakoutRangeWindow;
  return candidate?.window === window ? candidate : undefined;
}

function previousRangeValue(
  window: RollingRangeWindow | undefined,
  side: "max" | "min",
): number | undefined {
  if (!window || window.index <= window.window) {
    return undefined;
  }

  const currentIndex = window.index - 1;
  const indexes = side === "max" ? window.maxIndexes : window.minIndexes;
  const values = side === "max" ? window.maxValues : window.minValues;
  let head = side === "max" ? window.maxHead : window.minHead;
  if (indexes[head] === currentIndex) {
    head += 1;
  }
  return head < indexes.length ? values[head] : undefined;
}

function createRollingValueWindow(window: number): RollingValueWindow {
  return {
    window,
    values: new Array<number>(window),
    cursor: 0,
    count: 0,
    sum: 0,
    sumSquares: 0,
  };
}

function recordRollingValue(window: RollingValueWindow, value: number): void {
  if (window.count < window.window) {
    window.values[window.cursor] = value;
    window.cursor = (window.cursor + 1) % window.window;
    window.count += 1;
    window.sum += value;
    window.sumSquares += value * value;
    return;
  }

  const previous = window.values[window.cursor] ?? 0;
  window.values[window.cursor] = value;
  window.cursor = (window.cursor + 1) % window.window;
  window.sum += value - previous;
  window.sumSquares += value * value - previous * previous;
}

function createRollingRangeWindow(window: number): RollingRangeWindow {
  return {
    window,
    index: 0,
    maxIndexes: [],
    maxValues: [],
    maxHead: 0,
    minIndexes: [],
    minValues: [],
    minHead: 0,
  };
}

function recordRollingRange(window: RollingRangeWindow, value: number): void {
  const index = window.index;
  window.index += 1;

  while (
    window.maxValues.length > window.maxHead &&
    window.maxValues[window.maxValues.length - 1] <= value
  ) {
    window.maxValues.pop();
    window.maxIndexes.pop();
  }
  window.maxValues.push(value);
  window.maxIndexes.push(index);

  while (
    window.minValues.length > window.minHead &&
    window.minValues[window.minValues.length - 1] >= value
  ) {
    window.minValues.pop();
    window.minIndexes.pop();
  }
  window.minValues.push(value);
  window.minIndexes.push(index);

  const oldestIndex = index - window.window;
  while (
    window.maxHead < window.maxIndexes.length &&
    window.maxIndexes[window.maxHead] < oldestIndex
  ) {
    window.maxHead += 1;
  }
  while (
    window.minHead < window.minIndexes.length &&
    window.minIndexes[window.minHead] < oldestIndex
  ) {
    window.minHead += 1;
  }

  compactRangeWindow(window);
}

function compactRangeWindow(window: RollingRangeWindow): void {
  if (window.maxHead > 2048) {
    window.maxIndexes = window.maxIndexes.slice(window.maxHead);
    window.maxValues = window.maxValues.slice(window.maxHead);
    window.maxHead = 0;
  }
  if (window.minHead > 2048) {
    window.minIndexes = window.minIndexes.slice(window.minHead);
    window.minValues = window.minValues.slice(window.minHead);
    window.minHead = 0;
  }
}

function volatilityAdjustedThreshold(baseThresholdBps: number, volatilityBps: number): number {
  return Math.max(baseThresholdBps, volatilityBps * 1.25);
}

function confidenceFromMove(moveBps: number, thresholdBps: number): number {
  const denominator = Math.max(1, thresholdBps * 4);
  return clamp(Math.abs(moveBps) / denominator, 0.25, 1);
}

function confidenceFromZScore(zScore: number, entryZScore: number): number {
  return clamp(Math.abs(zScore) / Math.max(entryZScore * 2, 0.01), 0.25, 1);
}

function decisionConsensusScore(
  decision: DirectionalDecision,
  maxTargetExposurePct: number,
): number {
  if (decision.action !== "rebalance") {
    return 0;
  }

  const denominator = Math.max(maxTargetExposurePct, 0.01);
  return clamp(decision.targetExposurePct / denominator, -1, 1);
}

function maxRange(values: number[], start: number, end: number): number {
  let max = -Infinity;
  for (let index = start; index < end; index += 1) {
    if (values[index] > max) {
      max = values[index];
    }
  }
  return max;
}

function minRange(values: number[], start: number, end: number): number {
  let min = Infinity;
  for (let index = start; index < end; index += 1) {
    if (values[index] < min) {
      min = values[index];
    }
  }
  return min;
}

function averageRange(values: number[], start: number, end: number): number {
  const count = end - start;
  if (count <= 0) {
    return 0;
  }

  let sum = 0;
  for (let index = start; index < end; index += 1) {
    sum += values[index];
  }
  return sum / count;
}

function standardDeviationRange(
  values: number[],
  start: number,
  end: number,
  mean: number,
): number {
  const count = end - start;
  if (count < 2) {
    return 0;
  }

  let variance = 0;
  for (let index = start; index < end; index += 1) {
    const value = values[index];
    variance += (value - mean) ** 2;
  }
  return Math.sqrt(variance / count);
}

function clampInt(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.round(cleanNumber(value))));
}

function cleanNonNegative(value: number): number {
  return Math.max(0, cleanNumber(value));
}

function cleanNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function roundQuote(value: number): number {
  return Number((Number.isFinite(value) ? value : 0).toFixed(6));
}

function formatBps(value: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(1);
}

function formatZ(value: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(2);
}

function formatScore(value: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(3);
}
