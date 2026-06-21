import type {
  BotEvent,
  BotSignal,
  BotStatus,
  BotMetrics,
  Candle,
  ManualTradeInput,
  PaperBotState,
  PriceTick,
  StrategyConfig,
  TickProcessingOptions,
  TradeFill,
  TradingOrder,
} from "./types.js";
import {
  createLegacyValleyPeakConfig,
  createLegacyValleyPeakMemory,
  defaultLegacyValleyPeakConfig,
  evaluateLegacyValleyPeak,
  normalizeLegacyValleyPeakMemory,
} from "./legacy-valley-peak.js";
import {
  createMeanReversionConfig,
  createTrendFollowingConfig,
  createVolatilityBreakoutConfig,
  defaultMeanReversionConfig,
  defaultTrendFollowingConfig,
  defaultVolatilityBreakoutConfig,
  evaluateMeanReversion,
  evaluateTrendFollowing,
  evaluateVolatilityBreakout,
  type DirectionalDecision,
  type DirectionalSide,
} from "./directional-strategies.js";
import {
  analyzePositions,
  createPositionRiskConfig,
  defaultPositionRiskConfig,
} from "./position-ledger.js";

export const defaultStrategyConfig: StrategyConfig = {
  symbol: "BTCUSDT",
  baseAsset: "BTC",
  quoteAsset: "USDT",
  algorithm: "moving-average",
  startingQuote: 10_000,
  maxLeverage: 1,
  feeBps: 7.5,
  orderQuoteSize: 750,
  maxPositionQuote: 4_500,
  fastWindow: 8,
  slowWindow: 26,
  signalThresholdBps: 1.5,
  limitOffsetBps: 2,
  maxOpenOrders: 3,
  cooldownMs: 30_000,
  staleOrderMs: 180_000,
  takeProfitBps: 45,
  stopLossBps: 35,
  minOrderQuote: 25,
  benchmarkRandomSeed: 1337,
  legacyValleyPeak: defaultLegacyValleyPeakConfig,
  trendFollowing: defaultTrendFollowingConfig,
  volatilityBreakout: defaultVolatilityBreakoutConfig,
  meanReversion: defaultMeanReversionConfig,
  positionRisk: defaultPositionRiskConfig,
};

export type PartialStrategyConfig = Partial<StrategyConfig> &
  Pick<StrategyConfig, "symbol">;

interface ImmediateFillRollback {
  quoteFree: number;
  quoteReserved: number;
  baseFree: number;
  baseReserved: number;
  avgEntryPrice: number;
  avgShortEntryPrice: number;
  lastPrice: number;
  updatedAt: number;
  realizedPnl: number;
  feesPaid: number;
  winningTrades: number;
  losingTrades: number;
  sequence: number;
  ordersLength: number;
  fillsLength: number;
  metrics: BotMetrics;
}

const roundAsset = (value: number) => Number(value.toFixed(8));
const roundQuote = (value: number) => Number(value.toFixed(6));

export function createStrategyConfig(
  overrides: Partial<StrategyConfig> = {},
): StrategyConfig {
  const config = {
    ...defaultStrategyConfig,
    ...overrides,
    legacyValleyPeak: createLegacyValleyPeakConfig({
      ...defaultStrategyConfig.legacyValleyPeak,
      ...(overrides.legacyValleyPeak ?? {}),
    }),
    trendFollowing: createTrendFollowingConfig({
      ...defaultStrategyConfig.trendFollowing,
      ...(overrides.trendFollowing ?? {}),
    }),
    volatilityBreakout: createVolatilityBreakoutConfig({
      ...defaultStrategyConfig.volatilityBreakout,
      ...(overrides.volatilityBreakout ?? {}),
    }),
    meanReversion: createMeanReversionConfig({
      ...defaultStrategyConfig.meanReversion,
      ...(overrides.meanReversion ?? {}),
    }),
    positionRisk: createPositionRiskConfig({
      ...defaultStrategyConfig.positionRisk,
      ...(overrides.positionRisk ?? {}),
    }),
  };

  if (config.fastWindow >= config.slowWindow) {
    config.fastWindow = Math.max(2, Math.floor(config.slowWindow / 2));
  }
  if (
    config.algorithm !== "moving-average" &&
    config.algorithm !== "legacy-valley-peak" &&
    config.algorithm !== "trend-following" &&
    config.algorithm !== "volatility-breakout" &&
    config.algorithm !== "mean-reversion" &&
    config.algorithm !== "benchmark-always-long" &&
    config.algorithm !== "benchmark-always-short" &&
    config.algorithm !== "benchmark-always-flat" &&
    config.algorithm !== "benchmark-random-sign"
  ) {
    config.algorithm = defaultStrategyConfig.algorithm;
  }
  config.maxOpenOrders = Math.max(1, Math.round(config.maxOpenOrders));
  config.maxLeverage = clamp(cleanPositive(config.maxLeverage) || 1, 1, 999);
  config.orderQuoteSize = Math.max(config.minOrderQuote, config.orderQuoteSize);
  config.maxPositionQuote = Math.max(config.minOrderQuote, config.maxPositionQuote);
  config.limitOffsetBps = Math.max(0, config.limitOffsetBps);
  config.cooldownMs = Math.max(0, config.cooldownMs);
  config.staleOrderMs = Math.max(1_000, config.staleOrderMs);

  return config;
}

export function createInitialBotState(
  overrides: Partial<StrategyConfig> = {},
): PaperBotState {
  const config = createStrategyConfig(overrides);
  const now = Date.now();
  const state: PaperBotState = {
    id: "paper-bot",
    status: "running",
    symbol: config.symbol,
    baseAsset: config.baseAsset,
    quoteAsset: config.quoteAsset,
    startingQuote: config.startingQuote,
    quoteFree: config.startingQuote,
    quoteReserved: 0,
    baseFree: 0,
    baseReserved: 0,
    avgEntryPrice: 0,
    avgShortEntryPrice: 0,
    lastPrice: 0,
    sequence: 0,
    createdAt: now,
    updatedAt: now,
    realizedPnl: 0,
    feesPaid: 0,
    winningTrades: 0,
    losingTrades: 0,
    orders: [],
    fills: [],
    memory: {
      prices: [],
      lastSignal: "hold",
      lastActionAt: 0,
      legacyValleyPeak: createLegacyValleyPeakMemory(config.legacyValleyPeak),
    },
    metrics: emptyMetrics(config.startingQuote),
    config,
  };

  return recalculateMetrics(state);
}

export class SimulatedTradingBot {
  private state: PaperBotState;
  private openOrderIndexes = new Set<number>();

  constructor(initialState?: PaperBotState, overrides: Partial<StrategyConfig> = {}) {
    this.state = initialState
      ? normalizeLoadedState(initialState, overrides)
      : createInitialBotState(overrides);
    this.rebuildOpenOrderIndex();
  }

  snapshot(): PaperBotState {
    return structuredClone(this.state);
  }

  view(): Readonly<PaperBotState> {
    return this.state;
  }

  markToMarket(): Readonly<BotMetrics> {
    recalculateMetrics(this.state);
    return this.state.metrics;
  }

  setStatus(status: BotStatus, at = Date.now()): BotEvent[] {
    if (this.state.status === status) {
      return [];
    }

    this.state.status = status;
    this.state.updatedAt = at;
    return [
      {
        type: "status_changed",
        at,
        message: `Bot ${status}`,
        state: this.snapshot(),
      },
    ];
  }

  reset(overrides: Partial<StrategyConfig> = {}, at = Date.now()): BotEvent[] {
    this.state = createInitialBotState({ ...this.state.config, ...overrides });
    this.state.createdAt = at;
    this.state.updatedAt = at;
    this.rebuildOpenOrderIndex();
    return [
      {
        type: "state_reset",
        at,
        message: "Paper bot state reset",
        state: this.snapshot(),
      },
    ];
  }

  recordManualTrade(input: ManualTradeInput, at = Date.now()): BotEvent[] {
    const price = cleanPositive(input.price) || this.state.lastPrice || this.state.avgEntryPrice;
    const quantity = roundAsset(input.quantity);
    const positionEffect = input.targetPositionId
      ? "close"
      : input.positionEffect === "open" || input.positionEffect === "close"
        ? input.positionEffect
        : "auto";

    if (price <= 0) {
      throw new Error("Manual trade price must be positive.");
    }
    if (quantity <= 0) {
      throw new Error("Manual trade quantity must be positive.");
    }
    if (input.side !== "buy" && input.side !== "sell") {
      throw new Error("Manual trade side must be buy or sell.");
    }

    const previousState = structuredClone(this.state);
    try {
      const config = this.state.config;
      const feeRate = config.feeBps / 10_000;
      const quoteQuantity = roundQuote(price * quantity);
      const feeQuote = roundQuote(quoteQuantity * feeRate);
      const reason = input.reason?.trim() || "manual position fill";
      const orderId = `ord_${this.nextSequence().toString().padStart(6, "0")}`;
      let realizedPnl = 0;
      const opensPosition = positionEffect === "open";

      if (input.side === "buy") {
        const spent = roundQuote(quoteQuantity + feeQuote);
        const oldBase = this.state.baseFree + this.state.baseReserved;
        const newBase = oldBase + quantity;
        this.state.quoteFree = roundQuote(this.state.quoteFree - spent);
        this.state.baseFree = roundAsset(this.state.baseFree + quantity);

        if (opensPosition) {
          const oldLongBase = Math.max(0, oldBase);
          const oldCost = this.state.avgEntryPrice * oldLongBase;
          const newLongBase = oldLongBase + quantity;
          this.state.avgEntryPrice =
            newLongBase > 0 ? roundQuote((oldCost + spent) / newLongBase) : 0;
        } else if (oldBase >= 0) {
          const oldCost = this.state.avgEntryPrice * oldBase;
          this.state.avgEntryPrice = newBase > 0 ? roundQuote((oldCost + spent) / newBase) : 0;
        } else if (newBase > 0) {
          const leftoverRatio = Math.min(1, newBase / quantity);
          this.state.avgEntryPrice = roundQuote((spent * leftoverRatio) / newBase);
        } else {
          this.state.avgEntryPrice = 0;
        }
      } else {
        const proceeds = roundQuote(quoteQuantity - feeQuote);
        const oldBase = this.state.baseFree + this.state.baseReserved;
        const closedLongQuantity = Math.max(0, Math.min(quantity, oldBase));
        this.state.quoteFree = roundQuote(this.state.quoteFree + proceeds);
        this.state.baseFree = roundAsset(this.state.baseFree - quantity);

        if (!opensPosition && closedLongQuantity > 0 && this.state.avgEntryPrice > 0) {
          const feeForClosedQuantity = feeQuote * (closedLongQuantity / quantity);
          realizedPnl = roundQuote(
            (price - this.state.avgEntryPrice) * closedLongQuantity - feeForClosedQuantity,
          );
        }

        const remainingBase = this.state.baseFree + this.state.baseReserved;
        if (!opensPosition && remainingBase <= 0.00000001) {
          this.state.avgEntryPrice = 0;
        }
      }

      this.state.feesPaid = roundQuote(this.state.feesPaid + feeQuote);
      this.state.realizedPnl = roundQuote(this.state.realizedPnl + realizedPnl);
      if (realizedPnl > 0) {
        this.state.winningTrades += 1;
      } else if (realizedPnl < 0) {
        this.state.losingTrades += 1;
      }

      const order: TradingOrder = {
        id: orderId,
        side: input.side,
        type: "limit",
        status: "filled",
        price: roundQuote(price),
        quantity,
        filledQuantity: quantity,
        estimatedQuoteCost: input.side === "buy" ? roundQuote(quoteQuantity + feeQuote) : 0,
        createdAt: at,
        updatedAt: at,
        filledAt: at,
        reason,
        realizedPnl,
        feeQuote,
        targetPositionId: input.targetPositionId,
        positionEffect,
        manual: true,
      };
      const fill: TradeFill = {
        id: `fill_${this.nextSequence().toString().padStart(6, "0")}`,
        orderId,
        side: input.side,
        price: roundQuote(price),
        quantity,
        quoteQuantity,
        feeQuote,
        realizedPnl,
        filledAt: at,
        reason,
        targetPositionId: input.targetPositionId,
        positionEffect,
        manual: true,
      };

      this.state.orders.push(order);
      this.state.fills.push(fill);
      this.state.lastPrice = roundQuote(price);
      this.state.updatedAt = at;
      recalculateMetrics(this.state);
      this.assertLeverageLimit();

      return [
        {
          type: "order_filled",
          at,
          message: `Manual ${input.side.toUpperCase()} fill recorded`,
          order: structuredClone(order),
          fill: structuredClone(fill),
          state: this.snapshot(),
        },
      ];
    } catch (error) {
      this.state = previousState;
      this.rebuildOpenOrderIndex();
      throw error;
    }
  }

  onCandle(candle: Candle): BotEvent[] {
    return this.onTick({
      symbol: candle.symbol,
      eventTime: candle.closeTime,
      price: candle.close,
      quantity: candle.volume,
    });
  }

  onTick(tick: PriceTick, options: TickProcessingOptions = {}): BotEvent[] {
    if (tick.symbol !== this.state.symbol || tick.price <= 0) {
      return [];
    }

    const collectEvents = options.collectEvents ?? true;
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;
    this.state.lastPrice = tick.price;
    this.state.updatedAt = tick.eventTime;

    if (events) {
      events.push(...this.cancelStaleOrders(tick.eventTime, collectEvents));
      events.push(...this.fillOpenOrders(tick, collectEvents));
    } else {
      this.cancelStaleOrders(tick.eventTime, collectEvents);
      this.fillOpenOrders(tick, collectEvents);
    }
    this.rememberPrice(tick.price);

    if (this.state.status === "running") {
      if (events) {
        events.push(...this.evaluateStrategy(tick, collectEvents));
      } else {
        this.evaluateStrategy(tick, collectEvents);
      }
    }

    if (options.updateMetrics ?? true) {
      recalculateMetrics(this.state);
    }
    return events ?? [];
  }

  private evaluateStrategy(tick: PriceTick, collectEvents: boolean): BotEvent[] {
    const config = this.state.config;
    if (config.algorithm === "legacy-valley-peak") {
      return this.evaluateLegacyValleyPeakStrategy(tick, collectEvents);
    }
    if (
      config.algorithm === "trend-following" ||
      config.algorithm === "volatility-breakout" ||
      config.algorithm === "mean-reversion"
    ) {
      return this.evaluateDirectionalStrategy(tick, collectEvents);
    }
    if (
      config.algorithm === "benchmark-always-long" ||
      config.algorithm === "benchmark-always-short" ||
      config.algorithm === "benchmark-always-flat" ||
      config.algorithm === "benchmark-random-sign"
    ) {
      return this.evaluateBenchmarkControlStrategy(tick, collectEvents);
    }

    return this.evaluateMovingAverageStrategy(tick, collectEvents);
  }

  private evaluateMovingAverageStrategy(tick: PriceTick, collectEvents: boolean): BotEvent[] {
    const config = this.state.config;
    const prices = this.state.memory.prices;
    if (prices.length < config.slowWindow) {
      return [];
    }

    if (this.openOrderIndexes.size >= config.maxOpenOrders) {
      return [];
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return [];
    }

    const fastAvg = averageLast(prices, config.fastWindow);
    const slowAvg = averageLast(prices, config.slowWindow);
    this.state.memory.previousFastAvg = fastAvg;
    this.state.memory.previousSlowAvg = slowAvg;

    const threshold = config.signalThresholdBps / 10_000;
    const positionQuote = (this.state.baseFree + this.state.baseReserved) * tick.price;
    const hasPosition = positionQuote >= config.minOrderQuote;
    const changeFromEntry =
      this.state.avgEntryPrice > 0
        ? (tick.price - this.state.avgEntryPrice) / this.state.avgEntryPrice
        : 0;

    const takeProfit = hasPosition && changeFromEntry >= config.takeProfitBps / 10_000;
    const stopLoss = hasPosition && changeFromEntry <= -config.stopLossBps / 10_000;

    let signal: BotSignal = "hold";
    let reason = "moving average neutral";

    if ((fastAvg > slowAvg * (1 + threshold)) && positionQuote < config.maxPositionQuote) {
      signal = "buy";
      reason = "fast average above slow average";
    }

    if (hasPosition && (fastAvg < slowAvg * (1 - threshold))) {
      signal = "sell";
      reason = "fast average below slow average";
    }

    if (takeProfit) {
      signal = "sell";
      reason = "take profit";
    }

    if (stopLoss) {
      signal = "sell";
      reason = "stop loss";
    }

    if (signal === "hold") {
      this.state.memory.lastSignal = signal;
      return [];
    }

    const order =
      signal === "buy"
        ? this.createBuyOrder(tick.price, tick.eventTime, reason)
        : this.createSellOrder(tick.price, tick.eventTime, reason);

    if (!order) {
      return [];
    }

    this.state.memory.lastSignal = signal;
    this.state.memory.lastActionAt = tick.eventTime;

    if (!collectEvents) {
      return [];
    }

    return [
      {
        type: "order_created",
        at: tick.eventTime,
        message: `${signal.toUpperCase()} limit order created: ${reason}`,
        order: structuredClone(order),
      },
    ];
  }

  private evaluateLegacyValleyPeakStrategy(tick: PriceTick, collectEvents: boolean): BotEvent[] {
    const config = this.state.config;

    if (this.openOrderIndexes.size >= config.maxOpenOrders) {
      return [];
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return [];
    }

    this.state.memory.legacyValleyPeak = normalizeLegacyValleyPeakMemory(
      this.state.memory.legacyValleyPeak,
      config.legacyValleyPeak,
    );

    const decision = evaluateLegacyValleyPeak(
      this.state.memory.legacyValleyPeak,
      config.legacyValleyPeak,
      {
        eventTime: tick.eventTime,
        price: tick.price,
        feeRate: config.feeBps / 10_000,
        quoteFree: this.state.quoteFree,
        baseFree: this.state.baseFree,
        positionQuote: (this.state.baseFree + this.state.baseReserved) * tick.price,
        maxPositionQuote: config.maxPositionQuote,
      },
    );

    if (decision.signal === "hold") {
      this.state.memory.lastSignal = "hold";
      return [];
    }

    const order =
      decision.signal === "buy"
        ? this.createBuyOrder(tick.price, tick.eventTime, decision.reason, decision.quoteSize)
        : this.createSellOrder(tick.price, tick.eventTime, decision.reason, decision.quantity);

    if (!order) {
      return [];
    }

    this.state.memory.lastSignal = decision.signal;
    this.state.memory.lastActionAt = tick.eventTime;

    if (!collectEvents) {
      return [];
    }

    return [
      {
        type: "order_created",
        at: tick.eventTime,
        message: `${decision.signal.toUpperCase()} limit order created: ${decision.reason}`,
        order: structuredClone(order),
      },
    ];
  }

  private evaluateDirectionalStrategy(tick: PriceTick, collectEvents: boolean): BotEvent[] {
    const config = this.state.config;

    if (this.openOrderIndexes.size > 0) {
      return [];
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return [];
    }

    const currentSide = this.currentDirectionalSide(tick.price);
    const input = {
      prices: this.state.memory.prices,
      currentSide,
    };
    const decision =
      config.algorithm === "trend-following"
        ? evaluateTrendFollowing(config.trendFollowing, input)
        : config.algorithm === "volatility-breakout"
          ? evaluateVolatilityBreakout(config.volatilityBreakout, input)
          : evaluateMeanReversion(config.meanReversion, input);

    if (decision.action === "hold") {
      this.state.memory.lastSignal = "hold";
      return [];
    }

    const result = this.rebalanceToTargetExposure(
      tick.price,
      tick.eventTime,
      decision,
    );

    if (!result) {
      return [];
    }

    this.state.memory.lastSignal = result.order.side === "buy" ? "buy" : "sell";
    this.state.memory.lastActionAt = tick.eventTime;

    if (!collectEvents) {
      return [];
    }

    return [
      {
        type: "order_filled",
        at: tick.eventTime,
        message: `${result.order.side.toUpperCase()} market fill: ${decision.reason}`,
        order: structuredClone(result.order),
        fill: structuredClone(result.fill),
      },
    ];
  }

  private evaluateBenchmarkControlStrategy(
    tick: PriceTick,
    collectEvents: boolean,
  ): BotEvent[] {
    const config = this.state.config;

    if (this.openOrderIndexes.size > 0) {
      return [];
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return [];
    }

    const currentSide = this.currentDirectionalSide(tick.price);
    const targetSide = this.benchmarkControlTargetSide(tick);
    if (currentSide === targetSide) {
      this.state.memory.lastSignal = "hold";
      return [];
    }

    const targetExposurePct =
      targetSide === "long" ? 1 : targetSide === "short" ? -1 : 0;
    const decision: Extract<DirectionalDecision, { action: "rebalance" }> = {
      action: "rebalance",
      signal: targetSide === "long" ? "buy" : targetSide === "short" ? "sell" : "hold",
      targetExposurePct,
      reason: `${config.algorithm} target ${targetSide}`,
    };

    const result = this.rebalanceToTargetExposure(tick.price, tick.eventTime, decision);

    if (!result) {
      return [];
    }

    this.state.memory.lastSignal = result.order.side === "buy" ? "buy" : "sell";
    this.state.memory.lastActionAt = tick.eventTime;

    if (!collectEvents) {
      return [];
    }

    return [
      {
        type: "order_filled",
        at: tick.eventTime,
        message: `${result.order.side.toUpperCase()} benchmark fill: ${decision.reason}`,
        order: structuredClone(result.order),
        fill: structuredClone(result.fill),
      },
    ];
  }

  private benchmarkControlTargetSide(tick: PriceTick): DirectionalSide {
    const algorithm = this.state.config.algorithm;
    if (algorithm === "benchmark-always-long") {
      return "long";
    }
    if (algorithm === "benchmark-always-short") {
      return "short";
    }
    if (algorithm === "benchmark-always-flat") {
      return "flat";
    }

    const cooldownMs = Math.max(1, this.state.config.cooldownMs);
    const decisionBucket = Math.floor(tick.eventTime / cooldownMs);
    const randomUnit = deterministicUnitInterval(
      this.state.config.benchmarkRandomSeed,
      decisionBucket,
    );
    if (randomUnit < 0.45) {
      return "long";
    }
    if (randomUnit > 0.55) {
      return "short";
    }
    return "flat";
  }

  private currentDirectionalSide(price: number): DirectionalSide {
    const exposureQuote = (this.state.baseFree + this.state.baseReserved) * price;
    if (exposureQuote >= this.state.config.minOrderQuote) {
      return "long";
    }
    if (exposureQuote <= -this.state.config.minOrderQuote) {
      return "short";
    }
    return "flat";
  }

  private rebalanceToTargetExposure(
    marketPrice: number,
    at: number,
    decision: Extract<DirectionalDecision, { action: "rebalance" }>,
  ): { order: TradingOrder; fill: TradeFill } | undefined {
    const config = this.state.config;
    const equity = this.equityAt(marketPrice);
    if (equity <= 0 || marketPrice <= 0) {
      return undefined;
    }

    const longLimitQuote = Math.min(
      config.maxPositionQuote,
      equity * config.maxLeverage * 0.98,
    );
    const shortLimitQuote = Math.min(
      config.maxPositionQuote,
      equity * Math.max(0, config.maxLeverage - 1) * 0.98,
    );
    if (longLimitQuote <= 0 && shortLimitQuote <= 0) {
      return undefined;
    }

    const targetExposurePct = clamp(decision.targetExposurePct, -1, 1);
    const targetQuote =
      targetExposurePct >= 0
        ? targetExposurePct * longLimitQuote
        : targetExposurePct * shortLimitQuote;
    const currentBase = this.state.baseFree + this.state.baseReserved;
    const targetBase = targetQuote / marketPrice;
    const deltaBase = roundAsset(targetBase - currentBase);
    const quantity = roundAsset(Math.abs(deltaBase));
    const tradeQuote = quantity * marketPrice;
    const closingToFlat =
      Math.abs(targetQuote) < config.minOrderQuote &&
      Math.abs(currentBase * marketPrice) > 0;

    if (quantity <= 0 || (tradeQuote < config.minOrderQuote && !closingToFlat)) {
      return undefined;
    }

    return this.executeMarketFill(
      deltaBase > 0 ? "buy" : "sell",
      marketPrice,
      quantity,
      at,
      decision.reason,
    );
  }

  private executeMarketFill(
    side: "buy" | "sell",
    marketPrice: number,
    quantity: number,
    at: number,
    reason: string,
  ): { order: TradingOrder; fill: TradeFill } | undefined {
    const rollback = this.captureImmediateFillRollback();

    try {
      const config = this.state.config;
      const feeRate = config.feeBps / 10_000;
      const slippageRate = Math.max(0, config.positionRisk.marketSlippageBps) / 10_000;
      const price = roundQuote(
        side === "buy"
          ? marketPrice * (1 + slippageRate)
          : marketPrice * Math.max(0.000001, 1 - slippageRate),
      );
      const quoteQuantity = roundQuote(price * quantity);
      const feeQuote = roundQuote(quoteQuantity * feeRate);
      const oldBase = this.state.baseFree + this.state.baseReserved;
      let realizedPnl = 0;

      if (side === "buy") {
        const spent = roundQuote(quoteQuantity + feeQuote);
        const closedShortQuantity = Math.min(quantity, Math.max(0, -oldBase));
        if (closedShortQuantity > 0) {
          const feeForClosedQuantity = feeQuote * (closedShortQuantity / quantity);
          const averageShortEntryPrice = this.averageOpenShortEntryPrice();
          realizedPnl = roundQuote(
            averageShortEntryPrice * closedShortQuantity -
              price * closedShortQuantity -
              feeForClosedQuantity,
          );
        }

        this.state.quoteFree = roundQuote(this.state.quoteFree - spent);
        this.state.baseFree = roundAsset(this.state.baseFree + quantity);

        const newBase = oldBase + quantity;
        if (newBase >= -0.00000001) {
          this.state.avgShortEntryPrice = 0;
        }
        if (oldBase >= 0) {
          const oldCost = this.state.avgEntryPrice * oldBase;
          this.state.avgEntryPrice =
            newBase > 0 ? roundQuote((oldCost + spent) / newBase) : 0;
        } else if (newBase > 0) {
          const leftoverRatio = Math.min(1, newBase / quantity);
          this.state.avgEntryPrice = roundQuote((spent * leftoverRatio) / newBase);
        } else {
          this.state.avgEntryPrice = 0;
        }
      } else {
        const proceeds = roundQuote(quoteQuantity - feeQuote);
        const closedLongQuantity = Math.min(quantity, Math.max(0, oldBase));
        const openedShortQuantity = Math.max(0, -Math.min(0, oldBase - quantity));
        const previousShortQuantity = Math.max(0, -oldBase);
        if (closedLongQuantity > 0 && this.state.avgEntryPrice > 0) {
          const feeForClosedQuantity = feeQuote * (closedLongQuantity / quantity);
          realizedPnl = roundQuote(
            price * closedLongQuantity -
              feeForClosedQuantity -
              this.state.avgEntryPrice * closedLongQuantity,
          );
        }

        this.state.quoteFree = roundQuote(this.state.quoteFree + proceeds);
        this.state.baseFree = roundAsset(this.state.baseFree - quantity);

        const newBase = oldBase - quantity;
        if (newBase <= 0.00000001) {
          this.state.avgEntryPrice = 0;
        }
        if (newBase < -0.00000001) {
          const openedInThisFill = Math.max(0, openedShortQuantity - previousShortQuantity);
          const feeForOpenedQuantity = feeQuote * (openedInThisFill / quantity);
          const openedProceeds = Math.max(0, price * openedInThisFill - feeForOpenedQuantity);
          const previousProceeds = this.state.avgShortEntryPrice * previousShortQuantity;
          this.state.avgShortEntryPrice = roundQuote(
            (previousProceeds + openedProceeds) / Math.max(openedShortQuantity, 0.00000001),
          );
        } else {
          this.state.avgShortEntryPrice = 0;
        }
      }

      this.state.feesPaid = roundQuote(this.state.feesPaid + feeQuote);
      this.state.realizedPnl = roundQuote(this.state.realizedPnl + realizedPnl);
      if (realizedPnl > 0) {
        this.state.winningTrades += 1;
      } else if (realizedPnl < 0) {
        this.state.losingTrades += 1;
      }

      const orderId = `ord_${this.nextSequence().toString().padStart(6, "0")}`;
      const order: TradingOrder = {
        id: orderId,
        side,
        type: "market",
        status: "filled",
        price,
        quantity,
        filledQuantity: quantity,
        estimatedQuoteCost: side === "buy" ? roundQuote(quoteQuantity + feeQuote) : 0,
        createdAt: at,
        updatedAt: at,
        filledAt: at,
        reason,
        realizedPnl,
        feeQuote,
        positionEffect: "auto",
      };
      const fill: TradeFill = {
        id: `fill_${this.nextSequence().toString().padStart(6, "0")}`,
        orderId,
        side,
        price,
        quantity,
        quoteQuantity,
        feeQuote,
        realizedPnl,
        filledAt: at,
        reason,
        positionEffect: "auto",
      };

      this.state.orders.push(order);
      this.state.fills.push(fill);
      this.state.lastPrice = roundQuote(marketPrice);
      this.state.updatedAt = at;
      recalculateMetrics(this.state);
      this.assertLeverageLimit();

      return { order, fill };
    } catch (error) {
      this.restoreImmediateFillRollback(rollback);
      this.rebuildOpenOrderIndex();
      if (isLeverageLimitError(error)) {
        return undefined;
      }
      throw error;
    }
  }

  private averageOpenShortEntryPrice(): number {
    if (this.state.avgShortEntryPrice > 0) {
      return this.state.avgShortEntryPrice;
    }

    const positions = analyzePositions(this.state);
    let quantity = 0;
    let proceeds = 0;

    for (const lot of positions.shorts) {
      if (lot.status === "pending" || lot.remainingQuantity <= 0) {
        continue;
      }
      quantity += lot.remainingQuantity;
      proceeds += lot.remainingProceedsQuote;
    }

    if (quantity <= 0) {
      return this.state.lastPrice;
    }

    return proceeds / quantity;
  }

  private equityAt(price: number): number {
    return roundQuote(
      this.state.quoteFree +
        this.state.quoteReserved +
        (this.state.baseFree + this.state.baseReserved) * price,
    );
  }

  private captureImmediateFillRollback(): ImmediateFillRollback {
    return {
      quoteFree: this.state.quoteFree,
      quoteReserved: this.state.quoteReserved,
      baseFree: this.state.baseFree,
      baseReserved: this.state.baseReserved,
      avgEntryPrice: this.state.avgEntryPrice,
      avgShortEntryPrice: this.state.avgShortEntryPrice,
      lastPrice: this.state.lastPrice,
      updatedAt: this.state.updatedAt,
      realizedPnl: this.state.realizedPnl,
      feesPaid: this.state.feesPaid,
      winningTrades: this.state.winningTrades,
      losingTrades: this.state.losingTrades,
      sequence: this.state.sequence,
      ordersLength: this.state.orders.length,
      fillsLength: this.state.fills.length,
      metrics: { ...this.state.metrics },
    };
  }

  private restoreImmediateFillRollback(rollback: ImmediateFillRollback): void {
    this.state.quoteFree = rollback.quoteFree;
    this.state.quoteReserved = rollback.quoteReserved;
    this.state.baseFree = rollback.baseFree;
    this.state.baseReserved = rollback.baseReserved;
    this.state.avgEntryPrice = rollback.avgEntryPrice;
    this.state.avgShortEntryPrice = rollback.avgShortEntryPrice;
    this.state.lastPrice = rollback.lastPrice;
    this.state.updatedAt = rollback.updatedAt;
    this.state.realizedPnl = rollback.realizedPnl;
    this.state.feesPaid = rollback.feesPaid;
    this.state.winningTrades = rollback.winningTrades;
    this.state.losingTrades = rollback.losingTrades;
    this.state.sequence = rollback.sequence;
    this.state.orders.length = rollback.ordersLength;
    this.state.fills.length = rollback.fillsLength;
    this.state.metrics = rollback.metrics;
  }

  private createBuyOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuoteSize = this.state.config.orderQuoteSize,
  ): TradingOrder | undefined {
    const config = this.state.config;
    const positionQuote = (this.state.baseFree + this.state.baseReserved) * marketPrice;
    const remainingPositionQuote = Math.max(0, config.maxPositionQuote - positionQuote);
    const availableQuote = Math.max(0, this.state.quoteFree * 0.98);
    const quoteSize = Math.min(desiredQuoteSize, remainingPositionQuote, availableQuote);

    if (quoteSize < config.minOrderQuote) {
      return undefined;
    }

    const price = roundQuote(marketPrice * (1 - config.limitOffsetBps / 10_000));
    const feeRate = config.feeBps / 10_000;
    const quantity = roundAsset(quoteSize / price);
    const estimatedQuoteCost = roundQuote(quantity * price * (1 + feeRate));

    if (estimatedQuoteCost > this.state.quoteFree || quantity <= 0) {
      return undefined;
    }

    this.state.quoteFree = roundQuote(this.state.quoteFree - estimatedQuoteCost);
    this.state.quoteReserved = roundQuote(this.state.quoteReserved + estimatedQuoteCost);

    const order = this.buildOrder("buy", price, quantity, estimatedQuoteCost, createdAt, reason);
    this.state.orders.push(order);
    this.openOrderIndexes.add(this.state.orders.length - 1);
    return order;
  }

  private createSellOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuantity?: number,
  ): TradingOrder | undefined {
    const config = this.state.config;
    const quantity = roundAsset(
      desiredQuantity ?? Math.min(config.orderQuoteSize, this.state.baseFree * marketPrice) / marketPrice,
    );
    const quoteSize = quantity * marketPrice;

    if (quantity <= 0 || quoteSize < config.minOrderQuote || quantity > this.state.baseFree) {
      return undefined;
    }

    const isStopLoss = reason === "stop loss";
    const priceOffset = config.limitOffsetBps / 10_000;
    const price = roundQuote(
      marketPrice * (isStopLoss ? 1 - priceOffset : 1 + priceOffset),
    );

    this.state.baseFree = roundAsset(this.state.baseFree - quantity);
    this.state.baseReserved = roundAsset(this.state.baseReserved + quantity);

    const order = this.buildOrder("sell", price, quantity, 0, createdAt, reason);
    this.state.orders.push(order);
    this.openOrderIndexes.add(this.state.orders.length - 1);
    return order;
  }

  private buildOrder(
    side: "buy" | "sell",
    price: number,
    quantity: number,
    estimatedQuoteCost: number,
    createdAt: number,
    reason: string,
  ): TradingOrder {
    const id = `ord_${this.nextSequence().toString().padStart(6, "0")}`;
    return {
      id,
      side,
      type: "limit",
      status: "open",
      price,
      quantity,
      filledQuantity: 0,
      estimatedQuoteCost,
      createdAt,
      updatedAt: createdAt,
      reason,
      realizedPnl: 0,
      feeQuote: 0,
    };
  }

  private fillOpenOrders(tick: PriceTick, collectEvents: boolean): BotEvent[] {
    const events: BotEvent[] = [];

    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (order?.status !== "open") {
        this.openOrderIndexes.delete(index);
        continue;
      }

      const canFill =
        order.side === "buy" ? tick.price <= order.price : tick.price >= order.price;

      if (!canFill) {
        continue;
      }

      const result = this.tryFillOrder(order, index, tick.eventTime);
      if (result.cancelled) {
        if (collectEvents) {
          events.push({
            type: "order_cancelled",
            at: tick.eventTime,
            message: `${order.side.toUpperCase()} order cancelled by leverage limit`,
            order: structuredClone(result.cancelled),
          });
        }
        continue;
      }
      if (!result.fill) {
        continue;
      }
      if (!collectEvents) {
        continue;
      }
      events.push({
        type: "order_filled",
        at: tick.eventTime,
        message: `${order.side.toUpperCase()} order filled at ${order.price}`,
        order: structuredClone(order),
        fill: structuredClone(result.fill),
      });
    }

    return events;
  }

  private tryFillOrder(
    order: TradingOrder,
    index: number,
    filledAt: number,
  ): { fill?: TradeFill; cancelled?: TradingOrder } {
    const rollback = this.captureImmediateFillRollback();
    const orderBeforeFill = { ...order };

    try {
      const fill = this.fillOrder(order, index, filledAt);
      this.assertLeverageLimit();
      return { fill };
    } catch (error) {
      if (!isLeverageLimitError(error)) {
        throw error;
      }

      this.restoreImmediateFillRollback(rollback);
      this.state.orders[index] = orderBeforeFill;
      this.rebuildOpenOrderIndex();
      const restoredOrder = this.state.orders[index];
      if (restoredOrder?.status !== "open") {
        return {};
      }

      this.releaseOrderReserve(restoredOrder);
      restoredOrder.status = "cancelled";
      restoredOrder.cancelledAt = filledAt;
      restoredOrder.updatedAt = filledAt;
      restoredOrder.reason = `${restoredOrder.reason}; leverage limit`;
      this.openOrderIndexes.delete(index);
      return { cancelled: structuredClone(restoredOrder) };
    }
  }

  private fillOrder(order: TradingOrder, index: number, filledAt: number): TradeFill {
    const config = this.state.config;
    const feeRate = config.feeBps / 10_000;
    const quoteQuantity = roundQuote(order.price * order.quantity);
    const feeQuote = roundQuote(quoteQuantity * feeRate);
    let realizedPnl = 0;

    if (order.side === "buy") {
      const spent = roundQuote(quoteQuantity + feeQuote);
      this.state.quoteReserved = roundQuote(
        Math.max(0, this.state.quoteReserved - order.estimatedQuoteCost),
      );
      this.state.quoteFree = roundQuote(
        this.state.quoteFree + Math.max(0, order.estimatedQuoteCost - spent),
      );

      const oldBase = this.state.baseFree + this.state.baseReserved;
      const newBase = oldBase + order.quantity;
      const oldCost = this.state.avgEntryPrice * oldBase;
      const newCost = oldCost + spent;
      this.state.avgEntryPrice = newBase > 0 ? roundQuote(newCost / newBase) : 0;
      this.state.baseFree = roundAsset(this.state.baseFree + order.quantity);
    } else {
      this.state.baseReserved = roundAsset(
        Math.max(0, this.state.baseReserved - order.quantity),
      );
      const proceeds = roundQuote(quoteQuantity - feeQuote);
      this.state.quoteFree = roundQuote(this.state.quoteFree + proceeds);
      realizedPnl = roundQuote((order.price - this.state.avgEntryPrice) * order.quantity - feeQuote);
      this.state.realizedPnl = roundQuote(this.state.realizedPnl + realizedPnl);

      if (realizedPnl > 0) {
        this.state.winningTrades += 1;
      } else if (realizedPnl < 0) {
        this.state.losingTrades += 1;
      }

      const remainingBase = this.state.baseFree + this.state.baseReserved;
      if (remainingBase <= 0.00000001) {
        this.state.avgEntryPrice = 0;
        this.state.baseFree = 0;
        this.state.baseReserved = 0;
      }
    }

    this.state.feesPaid = roundQuote(this.state.feesPaid + feeQuote);

    order.status = "filled";
    order.filledQuantity = order.quantity;
    order.filledAt = filledAt;
    order.updatedAt = filledAt;
    order.realizedPnl = realizedPnl;
    order.feeQuote = feeQuote;
    this.openOrderIndexes.delete(index);

    const fill: TradeFill = {
      id: `fill_${this.nextSequence().toString().padStart(6, "0")}`,
      orderId: order.id,
      side: order.side,
      price: order.price,
      quantity: order.quantity,
      quoteQuantity,
      feeQuote,
      realizedPnl,
      filledAt,
      reason: order.reason,
      targetPositionId: order.targetPositionId,
      positionEffect: order.positionEffect,
      manual: order.manual,
    };
    this.state.fills.push(fill);

    return fill;
  }

  private assertLeverageLimit(): void {
    const maxLeverage = cleanPositive(this.state.config.maxLeverage) || 1;
    if (maxLeverage >= 999) {
      return;
    }

    const estimatedLeverage = this.estimateDebtLeverage();
    if (estimatedLeverage <= maxLeverage + 0.0001) {
      return;
    }

    const effectiveLeverage = analyzePositions(this.state).summary.effectiveLeverage;
    if (effectiveLeverage > maxLeverage + 0.0001) {
      throw new Error(
        `Leverage limit exceeded: ${formatLeverageForError(effectiveLeverage)}x > ${formatLeverageForError(maxLeverage)}x.`,
      );
    }
  }

  private estimateDebtLeverage(): number {
    const price =
      cleanPositive(this.state.lastPrice) ||
      cleanPositive(this.state.avgEntryPrice) ||
      cleanPositive(this.state.avgShortEntryPrice);
    if (price <= 0) {
      return 1;
    }

    const quoteBalance = this.state.quoteFree + this.state.quoteReserved;
    const baseQuantity = this.state.baseFree + this.state.baseReserved;
    const equity = this.equityAt(price);
    const borrowedQuote = Math.max(0, -quoteBalance);
    const borrowedBaseValue = Math.max(0, -baseQuantity * price);
    const externalBorrowedQuote = borrowedQuote + borrowedBaseValue;
    if (externalBorrowedQuote <= 0) {
      return 1;
    }
    if (equity <= 0) {
      return 999;
    }

    return clamp(1 + externalBorrowedQuote / equity, 1, 999);
  }

  private cancelStaleOrders(at: number, collectEvents: boolean): BotEvent[] {
    const events: BotEvent[] = [];

    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (order?.status !== "open") {
        this.openOrderIndexes.delete(index);
        continue;
      }
      if (at - order.createdAt < this.state.config.staleOrderMs) {
        continue;
      }

      this.releaseOrderReserve(order);
      order.status = "cancelled";
      order.cancelledAt = at;
      order.updatedAt = at;
      this.openOrderIndexes.delete(index);

      if (collectEvents) {
        events.push({
          type: "order_cancelled",
          at,
          message: `${order.side.toUpperCase()} order cancelled after waiting too long`,
          order: structuredClone(order),
        });
      }
    }

    return events;
  }

  private releaseOrderReserve(order: TradingOrder): void {
    if (order.side === "buy") {
      this.state.quoteReserved = roundQuote(
        Math.max(0, this.state.quoteReserved - order.estimatedQuoteCost),
      );
      this.state.quoteFree = roundQuote(this.state.quoteFree + order.estimatedQuoteCost);
    } else {
      this.state.baseReserved = roundAsset(
        Math.max(0, this.state.baseReserved - order.quantity),
      );
      this.state.baseFree = roundAsset(this.state.baseFree + order.quantity);
    }
  }

  private rememberPrice(price: number): void {
    const maxPrices = priceMemoryLimit(this.state.config);
    this.state.memory.prices.push(price);
    if (this.state.memory.prices.length > maxPrices * 2) {
      this.state.memory.prices.splice(0, this.state.memory.prices.length - maxPrices);
    }
  }

  private openOrders(): TradingOrder[] {
    const orders: TradingOrder[] = [];
    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (order?.status === "open") {
        orders.push(order);
      } else {
        this.openOrderIndexes.delete(index);
      }
    }

    return orders;
  }

  private rebuildOpenOrderIndex(): void {
    this.openOrderIndexes.clear();
    this.state.orders.forEach((order, index) => {
      if (order.status === "open") {
        this.openOrderIndexes.add(index);
      }
    });
  }

  private nextSequence(): number {
    this.state.sequence += 1;
    return this.state.sequence;
  }
}

function normalizeLoadedState(
  state: PaperBotState,
  overrides: Partial<StrategyConfig>,
): PaperBotState {
  const config = createStrategyConfig({ ...state.config, ...overrides });
  const normalized = structuredClone(state);
  normalized.config = config;
  normalized.symbol = config.symbol;
  normalized.baseAsset = config.baseAsset;
  normalized.quoteAsset = config.quoteAsset;
  normalized.startingQuote = config.startingQuote;
  normalized.memory ??= { prices: [], lastSignal: "hold", lastActionAt: 0 };
  normalized.memory.legacyValleyPeak = normalizeLegacyValleyPeakMemory(
    normalized.memory.legacyValleyPeak,
    config.legacyValleyPeak,
  );
  normalized.orders ??= [];
  normalized.fills ??= [];
  normalized.quoteReserved ??= 0;
  normalized.baseReserved ??= 0;
  normalized.avgShortEntryPrice ??= inferAverageShortEntryPrice(normalized);
  normalized.winningTrades ??= 0;
  normalized.losingTrades ??= 0;
  normalized.sequence ??= normalized.orders.length + normalized.fills.length;
  return recalculateMetrics(normalized);
}

function recalculateMetrics(state: PaperBotState): PaperBotState {
  state.metrics = calculateMetrics(state);
  return state;
}

function calculateMetrics(state: PaperBotState): BotMetrics {
  const lastPrice = state.lastPrice || state.avgEntryPrice || 0;
  const baseQuantity = state.baseFree + state.baseReserved;
  const baseValue = baseQuantity * lastPrice;
  const exposureValue = Math.abs(baseValue);
  const equity = roundQuote(state.quoteFree + state.quoteReserved + baseValue);
  const unrealizedPnl = calculateUnrealizedPnl(state, baseQuantity, lastPrice);
  const netPnl = roundQuote(equity - state.startingQuote);
  const peakEquity = Math.max(state.metrics?.peakEquity ?? state.startingQuote, equity);
  const drawdown = peakEquity > 0 ? ((peakEquity - equity) / peakEquity) * 100 : 0;
  const totalClosedTrades = state.winningTrades + state.losingTrades;

  return {
    equity,
    realizedPnl: roundQuote(state.realizedPnl),
    unrealizedPnl,
    netPnl,
    returnPct: state.startingQuote > 0 ? (netPnl / state.startingQuote) * 100 : 0,
    feesPaid: roundQuote(state.feesPaid),
    tradeCount: state.fills.length,
    winningTrades: state.winningTrades,
    losingTrades: state.losingTrades,
    winRate: totalClosedTrades > 0 ? (state.winningTrades / totalClosedTrades) * 100 : 0,
    peakEquity,
    maxDrawdownPct: Math.max(state.metrics?.maxDrawdownPct ?? 0, drawdown),
    exposurePct: equity > 0 ? (exposureValue / equity) * 100 : 0,
  };
}

function calculateUnrealizedPnl(
  state: PaperBotState,
  baseQuantity: number,
  lastPrice: number,
): number {
  if (baseQuantity > 0 && state.avgEntryPrice > 0) {
    return roundQuote((lastPrice - state.avgEntryPrice) * baseQuantity);
  }
  if (baseQuantity < 0 && state.avgShortEntryPrice > 0) {
    return roundQuote((state.avgShortEntryPrice - lastPrice) * Math.abs(baseQuantity));
  }
  return 0;
}

function inferAverageShortEntryPrice(state: PaperBotState): number {
  const positions = analyzePositions(state);
  let quantity = 0;
  let proceeds = 0;

  for (const lot of positions.shorts) {
    if (lot.status === "pending" || lot.remainingQuantity <= 0) {
      continue;
    }
    quantity += lot.remainingQuantity;
    proceeds += lot.remainingProceedsQuote;
  }

  return quantity > 0 ? roundQuote(proceeds / quantity) : 0;
}

function priceMemoryLimit(config: StrategyConfig): number {
  return Math.max(
    50,
    config.slowWindow * 8,
    config.trendFollowing.slowWindow + 2,
    config.trendFollowing.volatilityWindow + 2,
    config.volatilityBreakout.lookbackWindow + 2,
    config.meanReversion.trendWindow + 2,
  );
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
    maxDrawdownPct: 0,
    exposurePct: 0,
  };
}

function averageLast(values: number[], count: number): number {
  const start = Math.max(0, values.length - count);
  let sum = 0;
  for (let index = start; index < values.length; index += 1) {
    sum += values[index];
  }
  return sum / (values.length - start);
}

function cleanPositive(value: number | undefined): number {
  return Number.isFinite(value) && (value as number) > 0 ? (value as number) : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function deterministicUnitInterval(...values: number[]): number {
  let hash = 2166136261;
  for (const value of values) {
    hash ^= Math.trunc(value);
    hash = Math.imul(hash, 16777619);
  }
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 2246822507);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 3266489909);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 4294967296;
}

function isLeverageLimitError(error: unknown): boolean {
  return error instanceof Error && error.message.startsWith("Leverage limit exceeded:");
}

function formatLeverageForError(value: number): string {
  return (Number.isFinite(value) ? value : 999).toFixed(2);
}
