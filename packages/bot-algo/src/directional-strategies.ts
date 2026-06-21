import type {
  BotSignal,
  MeanReversionConfig,
  TrendFollowingConfig,
  VolatilityBreakoutConfig,
} from "./types.js";

export type DirectionalSide = "long" | "short" | "flat";

export interface DirectionalStrategyInput {
  prices: number[];
  currentSide: DirectionalSide;
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
  const volatilityBps = rollingReturnVolatilityBps(prices, config.volatilityWindow);
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
  const history = prices.slice(prices.length - config.lookbackWindow - 1, prices.length - 1);
  const rangeHigh = Math.max(...history);
  const rangeLow = Math.min(...history);
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
  const windowPrices = prices.slice(prices.length - config.window);
  const mean = average(windowPrices);
  const stdDev = standardDeviation(windowPrices, mean);
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

function rollingReturnVolatilityBps(prices: number[], window: number): number {
  const start = Math.max(1, prices.length - window);
  const returns: number[] = [];
  for (let index = start; index < prices.length; index += 1) {
    const previous = prices[index - 1];
    const current = prices[index];
    if (previous > 0 && current > 0) {
      returns.push(((current - previous) / previous) * 10_000);
    }
  }

  if (returns.length < 2) {
    return 0;
  }

  return standardDeviation(returns, average(returns));
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

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  let sum = 0;
  for (const value of values) {
    sum += value;
  }
  return sum / values.length;
}

function standardDeviation(values: number[], mean: number): number {
  if (values.length < 2) {
    return 0;
  }

  let variance = 0;
  for (const value of values) {
    variance += (value - mean) ** 2;
  }
  return Math.sqrt(variance / values.length);
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
