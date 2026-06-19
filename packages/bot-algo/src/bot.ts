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
  createPositionRiskConfig,
  defaultPositionRiskConfig,
} from "./position-ledger.js";

export const defaultStrategyConfig: StrategyConfig = {
  symbol: "BTCUSDT",
  baseAsset: "BTC",
  quoteAsset: "USDT",
  algorithm: "moving-average",
  startingQuote: 10_000,
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
  legacyValleyPeak: defaultLegacyValleyPeakConfig,
  positionRisk: defaultPositionRiskConfig,
};

export type PartialStrategyConfig = Partial<StrategyConfig> &
  Pick<StrategyConfig, "symbol">;

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
    positionRisk: createPositionRiskConfig({
      ...defaultStrategyConfig.positionRisk,
      ...(overrides.positionRisk ?? {}),
    }),
  };

  if (config.fastWindow >= config.slowWindow) {
    config.fastWindow = Math.max(2, Math.floor(config.slowWindow / 2));
  }
  if (config.algorithm !== "moving-average" && config.algorithm !== "legacy-valley-peak") {
    config.algorithm = defaultStrategyConfig.algorithm;
  }
  config.maxOpenOrders = Math.max(1, Math.round(config.maxOpenOrders));
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

    if (price <= 0) {
      throw new Error("Manual trade price must be positive.");
    }
    if (quantity <= 0) {
      throw new Error("Manual trade quantity must be positive.");
    }
    if (input.side !== "buy" && input.side !== "sell") {
      throw new Error("Manual trade side must be buy or sell.");
    }

    const config = this.state.config;
    const feeRate = config.feeBps / 10_000;
    const quoteQuantity = roundQuote(price * quantity);
    const feeQuote = roundQuote(quoteQuantity * feeRate);
    const reason = input.reason?.trim() || "manual position fill";
    const orderId = `ord_${this.nextSequence().toString().padStart(6, "0")}`;
    let realizedPnl = 0;

    if (input.side === "buy") {
      const spent = roundQuote(quoteQuantity + feeQuote);
      const oldBase = this.state.baseFree + this.state.baseReserved;
      const newBase = oldBase + quantity;
      this.state.quoteFree = roundQuote(this.state.quoteFree - spent);
      this.state.baseFree = roundAsset(this.state.baseFree + quantity);

      if (oldBase >= 0) {
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

      if (closedLongQuantity > 0 && this.state.avgEntryPrice > 0) {
        const feeForClosedQuantity = feeQuote * (closedLongQuantity / quantity);
        realizedPnl = roundQuote(
          (price - this.state.avgEntryPrice) * closedLongQuantity - feeForClosedQuantity,
        );
      }

      const remainingBase = this.state.baseFree + this.state.baseReserved;
      if (remainingBase <= 0.00000001) {
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
      manual: true,
    };

    this.state.orders.push(order);
    this.state.fills.push(fill);
    this.state.lastPrice = roundQuote(price);
    this.state.updatedAt = at;
    recalculateMetrics(this.state);

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

      const fill = this.fillOrder(order, index, tick.eventTime);
      if (!collectEvents) {
        continue;
      }
      events.push({
        type: "order_filled",
        at: tick.eventTime,
        message: `${order.side.toUpperCase()} order filled at ${order.price}`,
        order: structuredClone(order),
        fill: structuredClone(fill),
      });
    }

    return events;
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
    };
    this.state.fills.push(fill);

    return fill;
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
    const maxPrices = this.state.config.slowWindow * 8;
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
  const baseValue = (state.baseFree + state.baseReserved) * lastPrice;
  const equity = roundQuote(state.quoteFree + state.quoteReserved + baseValue);
  const unrealizedPnl =
    state.avgEntryPrice > 0
      ? roundQuote((lastPrice - state.avgEntryPrice) * (state.baseFree + state.baseReserved))
      : 0;
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
    exposurePct: equity > 0 ? (baseValue / equity) * 100 : 0,
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
