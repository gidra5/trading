import { createBotCoreState } from "./legacy/bot-core.js";
import {
  createPositionRiskConfig,
  defaultPositionRiskConfig,
} from "./legacy/position-ledger.js";
import {
  createLegacyValleyPeakConfig,
  defaultLegacyValleyPeakConfig,
} from "./legacy/valley-peak.js";
import type {
  BotMetrics,
  InternalBorrowAccounting,
  PaperBotState,
  StrategyConfig,
} from "./legacy/types.js";

export const defaultStrategyConfig: StrategyConfig = {
  symbol: "BTCUSDT",
  baseAsset: "BTC",
  quoteAsset: "USDT",
  algorithm: "legacy-valley-peak",
  startingQuote: 10_000,
  maxLeverage: 5,
  shortMarginModel: "futures-margin",
  longBorrowDepth: 999,
  shortBorrowDepth: 999,
  internalBorrowAccounting: "inactive",
  borrowerProfitShareToLender: 1,
  feeBps: 7.5,
  maxPositionQuote: Number.POSITIVE_INFINITY,
  limitOffsetBps: 2,
  maxOpenOrders: 1024,
  cooldownMs: 300_000,
  staleOrderMs: 30 * 24 * 60 * 60 * 1000,
  minOrderQuote: 5,
  legacyValleyPeak: defaultLegacyValleyPeakConfig,
  positionRisk: defaultPositionRiskConfig,
};

export type PartialStrategyConfig = Partial<
  Omit<StrategyConfig, "legacyValleyPeak" | "positionRisk">
> & {
  legacyValleyPeak?: Partial<StrategyConfig["legacyValleyPeak"]>;
  positionRisk?: Partial<StrategyConfig["positionRisk"]>;
};

export function createStrategyConfig(
  overrides: PartialStrategyConfig = {},
): StrategyConfig {
  const config: StrategyConfig = {
    symbol: overrides.symbol ?? defaultStrategyConfig.symbol,
    baseAsset: overrides.baseAsset ?? defaultStrategyConfig.baseAsset,
    quoteAsset: overrides.quoteAsset ?? defaultStrategyConfig.quoteAsset,
    algorithm: overrides.algorithm ?? defaultStrategyConfig.algorithm,
    startingQuote: overrides.startingQuote ?? defaultStrategyConfig.startingQuote,
    maxLeverage: overrides.maxLeverage ?? defaultStrategyConfig.maxLeverage,
    shortMarginModel: normalizeShortMarginModel(
      overrides.shortMarginModel ?? defaultStrategyConfig.shortMarginModel,
    ),
    longBorrowDepth: normalizeBorrowDepth(
      overrides.longBorrowDepth ?? defaultStrategyConfig.longBorrowDepth,
    ),
    shortBorrowDepth: normalizeBorrowDepth(
      overrides.shortBorrowDepth ?? defaultStrategyConfig.shortBorrowDepth,
    ),
    internalBorrowAccounting: normalizeInternalBorrowAccounting(
      overrides.internalBorrowAccounting ?? defaultStrategyConfig.internalBorrowAccounting,
    ),
    borrowerProfitShareToLender: clamp(
      cleanFiniteNumber(
        overrides.borrowerProfitShareToLender
          ?? defaultStrategyConfig.borrowerProfitShareToLender,
      ),
      0,
      1,
    ),
    feeBps: overrides.feeBps ?? defaultStrategyConfig.feeBps,
    maxPositionQuote: overrides.maxPositionQuote ?? defaultStrategyConfig.maxPositionQuote,
    limitOffsetBps: overrides.limitOffsetBps ?? defaultStrategyConfig.limitOffsetBps,
    maxOpenOrders: overrides.maxOpenOrders ?? defaultStrategyConfig.maxOpenOrders,
    cooldownMs: overrides.cooldownMs ?? defaultStrategyConfig.cooldownMs,
    staleOrderMs: overrides.staleOrderMs ?? defaultStrategyConfig.staleOrderMs,
    minOrderQuote: overrides.minOrderQuote ?? defaultStrategyConfig.minOrderQuote,
    legacyValleyPeak: createLegacyValleyPeakConfig({
      ...defaultStrategyConfig.legacyValleyPeak,
      ...(overrides.legacyValleyPeak ?? {}),
    }),
    positionRisk: createPositionRiskConfig({
      ...defaultStrategyConfig.positionRisk,
      ...(overrides.positionRisk ?? {}),
    }),
  };

  if (config.algorithm !== "legacy-valley-peak") {
    config.algorithm = defaultStrategyConfig.algorithm;
  }
  config.maxLeverage = clamp(cleanPositive(config.maxLeverage) || 1, 1, 999);
  config.longBorrowDepth = normalizeBorrowDepth(config.longBorrowDepth);
  config.shortBorrowDepth = normalizeBorrowDepth(config.shortBorrowDepth);
  config.internalBorrowAccounting = normalizeInternalBorrowAccounting(
    config.internalBorrowAccounting,
  );
  config.borrowerProfitShareToLender = clamp(
    cleanFiniteNumber(config.borrowerProfitShareToLender),
    0,
    1,
  );
  config.maxPositionQuote = Math.max(config.minOrderQuote, config.maxPositionQuote);
  config.limitOffsetBps = Math.max(0, config.limitOffsetBps);
  config.maxOpenOrders = Math.max(1, Math.round(config.maxOpenOrders));
  config.cooldownMs = Math.max(0, config.cooldownMs);
  config.staleOrderMs = Math.max(1_000, config.staleOrderMs);
  return config;
}

/** Transitional result shape for the UI while BacktestResult still exposes its
 * historical flat account fields. It is not an execution engine. */
export function createInitialBotState(
  overrides: PartialStrategyConfig = {},
): PaperBotState {
  const config = createStrategyConfig(overrides);
  const now = Date.now();
  const core = createBotCoreState(config, {
    id: "paper-bot",
    now,
    status: "running",
  });
  return {
    ...core,
    startingQuote: config.startingQuote,
    quoteFree: config.startingQuote,
    quoteReserved: 0,
    baseFree: 0,
    baseReserved: 0,
    avgEntryPrice: 0,
    avgShortEntryPrice: 0,
    realizedPnl: 0,
    feesPaid: 0,
    exitGridSpanTotal: 0,
    exitGridSpanCount: 0,
    exitGridOrderCountTotal: 0,
    winningTrades: 0,
    losingTrades: 0,
    orders: [],
    fills: [],
    metrics: emptyMetrics(config.startingQuote),
  };
}

function emptyMetrics(startingQuote: number): BotMetrics {
  return {
    equity: startingQuote,
    realizedPnl: 0,
    unrealizedPnl: 0,
    netPnl: 0,
    returnPct: 0,
    feesPaid: 0,
    tradeCount: 0,
    winningTrades: 0,
    losingTrades: 0,
    winRate: 0,
    peakEquity: startingQuote,
    maxInitialBalanceDrawdownPct: 0,
    maxDrawdownPct: 0,
    exposurePct: 0,
    maxEntryLeverage: 1,
    maxEffectiveLeverage: 1,
    avgExitGridSpan: 0,
    avgExitGridOrderCount: 0,
    exitGridSpanCount: 0,
  };
}

function normalizeShortMarginModel(
  value: StrategyConfig["shortMarginModel"],
): StrategyConfig["shortMarginModel"] {
  return value === "futures-margin" ? "futures-margin" : "spot-borrow";
}

function normalizeInternalBorrowAccounting(
  value: InternalBorrowAccounting | undefined,
): InternalBorrowAccounting {
  return value === "inactive" ? "inactive" : "active";
}

function normalizeBorrowDepth(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.round(value)) : 0;
}

function cleanPositive(value: number | undefined): number {
  return Number.isFinite(value) && (value as number) > 0 ? value as number : 0;
}

function cleanFiniteNumber(value: number | undefined): number {
  return Number.isFinite(value) ? value as number : 0;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
