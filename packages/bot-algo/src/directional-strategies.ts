import type {
  BotSignal,
  MasterAdaptiveConfig,
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
  const decisions = [
    {
      label: "trend",
      weight: config.trendWeight,
      maxTarget: children.trendFollowing.targetExposurePct,
      decision: evaluateTrendFollowing(children.trendFollowing, input),
    },
    {
      label: "breakout",
      weight: config.breakoutWeight,
      maxTarget: children.volatilityBreakout.targetExposurePct,
      decision: evaluateVolatilityBreakout(children.volatilityBreakout, input),
    },
    {
      label: "reversion",
      weight: config.reversionWeight,
      maxTarget: children.meanReversion.targetExposurePct,
      decision: evaluateMeanReversion(children.meanReversion, input),
    },
  ].filter((entry) => entry.weight > 0);

  const totalWeight = decisions.reduce((total, entry) => total + entry.weight, 0);
  if (totalWeight <= 0) {
    return target("flat", 0, "master adaptive has no enabled child signals");
  }

  const scores = decisions.map((entry) => ({
    ...entry,
    score: decisionConsensusScore(entry.decision, entry.maxTarget),
  }));
  const weightedScore =
    scores.reduce((total, entry) => total + entry.weight * entry.score, 0) / totalWeight;
  const hasLong = scores.some((entry) => entry.score > 0);
  const hasShort = scores.some((entry) => entry.score < 0);
  const disagreementScale = hasLong && hasShort ? config.disagreementExposureScale : 1;
  const volatilityBps = rollingReturnVolatilityBps(prices, config.volatilityWindow);
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
  const activeSignals = scores
    .filter((entry) => entry.score !== 0)
    .map((entry) => `${entry.label} ${formatScore(entry.score)}`)
    .join(", ");

  return target(
    side,
    config.targetExposurePct * absScore,
    `master adaptive ${side}: score ${formatScore(finalScore)}, vol ${formatBps(volatilityBps)}bps${
      activeSignals ? `, ${activeSignals}` : ""
    }`,
  );
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

function formatScore(value: number): string {
  return (Number.isFinite(value) ? value : 0).toFixed(3);
}
