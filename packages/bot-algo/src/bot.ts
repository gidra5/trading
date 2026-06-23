import type {
  BotEvent,
  BotStatus,
  BotMetrics,
  Candle,
  LegacyExitGridMemory,
  LegacyValleyPeakMemory,
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
  analyzePositions,
  createPositionRiskConfig,
  defaultPositionRiskConfig,
} from "./position-ledger.js";

export const defaultStrategyConfig: StrategyConfig = {
  symbol: "BTCUSDT",
  baseAsset: "BTC",
  quoteAsset: "USDT",
  algorithm: "legacy-valley-peak",
  startingQuote: 10_000,
  maxLeverage: 5,
  feeBps: 7.5,
  maxPositionQuote: 50_000,
  limitOffsetBps: 2,
  maxOpenOrders: 24,
  cooldownMs: 30_000,
  staleOrderMs: 30 * 24 * 60 * 60 * 1000,
  minOrderQuote: 25,
  legacyValleyPeak: defaultLegacyValleyPeakConfig,
  positionRisk: defaultPositionRiskConfig,
};

export type PartialStrategyConfig = Partial<
  Omit<
    StrategyConfig,
    | "legacyValleyPeak"
    | "positionRisk"
  >
> & {
  legacyValleyPeak?: Partial<StrategyConfig["legacyValleyPeak"]>;
  positionRisk?: Partial<StrategyConfig["positionRisk"]>;
};

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
  liquidatedPositionCount: number;
  metrics: BotMetrics;
}

interface ImmediateOrderResult {
  order?: TradingOrder;
  fill?: TradeFill;
  cancelled?: TradingOrder;
}

interface LegacyExitGridLot {
  id: string;
  averagePrice: number;
  originalQuantity: number;
  remainingQuantity: number;
  breakEvenSellPrice: number;
}

interface LegacyLongLotCache {
  processedFillsLength: number;
  lots: Map<string, LegacyTrackedLongLot>;
}

interface LegacyTrackedLongLot extends LegacyExitGridLot {
  remainingCostQuote: number;
}

const roundAsset = (value: number) => Number(value.toFixed(8));
const roundQuote = (value: number) => Number(value.toFixed(6));
const NO_EVENTS: BotEvent[] = [];
const MIN_BASE_QUANTITY = 0.00000001;
const LEGACY_EXIT_GRID_REASON = "legacy exit grid";

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
    feeBps: overrides.feeBps ?? defaultStrategyConfig.feeBps,
    maxPositionQuote:
      overrides.maxPositionQuote ?? defaultStrategyConfig.maxPositionQuote,
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
  config.maxOpenOrders = Math.max(1, Math.round(config.maxOpenOrders));
  config.maxLeverage = clamp(cleanPositive(config.maxLeverage) || 1, 1, 999);
  config.maxPositionQuote = Math.max(config.minOrderQuote, config.maxPositionQuote);
  config.limitOffsetBps = Math.max(0, config.limitOffsetBps);
  config.cooldownMs = Math.max(0, config.cooldownMs);
  config.staleOrderMs = Math.max(1_000, config.staleOrderMs);

  return config;
}

export function createInitialBotState(
  overrides: PartialStrategyConfig = {},
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
  private priceMemoryLimit = 0;
  private legacyLongLotCache?: LegacyLongLotCache;
  private liquidatedPositionCountValue = 0;
  private readonly reusableTick: PriceTick = {
    symbol: "",
    eventTime: 0,
    price: 0,
  };

  constructor(initialState?: PaperBotState, overrides: PartialStrategyConfig = {}) {
    this.state = initialState
      ? normalizeLoadedState(initialState, overrides)
      : createInitialBotState(overrides);
    this.priceMemoryLimit = priceMemoryLimit(this.state.config);
    this.reusableTick.symbol = this.state.symbol;
    this.liquidatedPositionCountValue = countLiquidatedPositions(this.state.fills);
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

  accountLiquidationPrice(): number | undefined {
    const quoteBalance = this.state.quoteFree + this.state.quoteReserved;
    const baseQuantity = this.totalBase();
    if (baseQuantity > MIN_BASE_QUANTITY && quoteBalance < 0) {
      return roundQuote(-quoteBalance / baseQuantity);
    }
    if (baseQuantity < -MIN_BASE_QUANTITY && quoteBalance > 0) {
      return roundQuote(quoteBalance / -baseQuantity);
    }
    return undefined;
  }

  liquidatedPositionCount(): number {
    return this.liquidatedPositionCountValue;
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

  reset(overrides: PartialStrategyConfig = {}, at = Date.now()): BotEvent[] {
    this.state = createInitialBotState(
      mergeStrategyOverrides(this.state.config, overrides),
    );
    this.state.createdAt = at;
    this.state.updatedAt = at;
    this.priceMemoryLimit = priceMemoryLimit(this.state.config);
    this.reusableTick.symbol = this.state.symbol;
    this.legacyLongLotCache = undefined;
    this.liquidatedPositionCountValue = 0;
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
    return this.onPriceTick(tick.symbol, tick.eventTime, tick.price, tick.quantity, options);
  }

  onPriceTick(
    symbol: string,
    eventTime: number,
    price: number,
    quantity?: number,
    options: TickProcessingOptions = {},
  ): BotEvent[] {
    if (symbol !== this.state.symbol || price <= 0) {
      return [];
    }

    return this.processAcceptedPriceTick(eventTime, price, quantity, options);
  }

  onReplayPriceTick(
    eventTime: number,
    price: number,
  ): BotEvent[] {
    if (price <= 0) {
      return [];
    }

    this.state.lastPrice = price;
    this.state.updatedAt = eventTime;

    const tick = this.reusableTick;
    if (tick.symbol !== this.state.symbol) {
      tick.symbol = this.state.symbol;
    }
    tick.eventTime = eventTime;
    tick.price = price;
    tick.quantity = undefined;

    this.cancelStaleOrders(eventTime, false);
    this.fillOpenOrders(tick, false);
    if (this.liquidateAccountIfNeeded(eventTime, price, false)) {
      return NO_EVENTS;
    }
    this.rememberPrice(price);

    if (this.state.status === "running") {
      this.evaluateStrategy(tick, false);
    }

    return NO_EVENTS;
  }

  private processAcceptedPriceTick(
    eventTime: number,
    price: number,
    quantity: number | undefined,
    options: TickProcessingOptions,
  ): BotEvent[] {
    if (this.reusableTick.symbol !== this.state.symbol) {
      this.reusableTick.symbol = this.state.symbol;
    }

    const collectEvents = options.collectEvents ?? true;
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;
    this.state.lastPrice = price;
    this.state.updatedAt = eventTime;

    const tick = this.reusableTick;
    tick.eventTime = eventTime;
    tick.price = price;
    tick.quantity = quantity;

    if (events) {
      events.push(...this.cancelStaleOrders(eventTime, collectEvents));
      events.push(...this.fillOpenOrders(tick, collectEvents));
    } else {
      this.cancelStaleOrders(eventTime, collectEvents);
      this.fillOpenOrders(tick, collectEvents);
    }
    const liquidated = options.simulateLiquidation
      ? this.liquidateAccountIfNeeded(eventTime, price, collectEvents, events)
      : false;
    this.rememberPrice(price);

    if (!liquidated && this.state.status === "running") {
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
    return this.evaluateLegacyValleyPeakStrategy(tick, collectEvents);
  }

  private evaluateLegacyValleyPeakStrategy(tick: PriceTick, collectEvents: boolean): BotEvent[] {
    const config = this.state.config;

    if (config.legacyValleyPeak.exitGridEnabled) {
      return this.evaluateLegacyValleyPeakExitGridStrategy(tick, collectEvents);
    }

    if (this.openOrderIndexes.size >= config.maxOpenOrders) {
      return [];
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return [];
    }

    const memory = this.ensureLegacyValleyPeakMemory();
    const targetEntryLeverage = this.targetLongEntryLeverage(tick.price);
    const decision = evaluateLegacyValleyPeak(
      memory,
      config.legacyValleyPeak,
      {
        eventTime: tick.eventTime,
        price: tick.price,
        feeRate: config.feeBps / 10_000,
        buyingPowerQuote: this.longEntryBuyingPowerQuote(tick.price, targetEntryLeverage),
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

  private evaluateLegacyValleyPeakExitGridStrategy(
    tick: PriceTick,
    collectEvents: boolean,
  ): BotEvent[] {
    const config = this.state.config;
    const legacyConfig = config.legacyValleyPeak;
    const memory = this.ensureLegacyValleyPeakMemory();
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;
    const totalBase = this.totalBase();

    if (
      legacyConfig.exitGridPositionMode === "aggregate" &&
      totalBase <= MIN_BASE_QUANTITY
    ) {
      delete memory.exitGrids?.["aggregate-long"];
    }
    this.updateLegacyExitGridPeaks(memory, tick.price);
    const targetEntryLeverage = this.targetLongEntryLeverage(tick.price);

    const decision = evaluateLegacyValleyPeak(
      memory,
      legacyConfig,
      {
        eventTime: tick.eventTime,
        price: tick.price,
        feeRate: config.feeBps / 10_000,
        buyingPowerQuote: this.longEntryBuyingPowerQuote(tick.price, targetEntryLeverage),
        baseFree: totalBase,
        positionQuote: totalBase * tick.price,
        maxPositionQuote: config.maxPositionQuote,
      },
    );

    if (decision.signal === "buy") {
      if (
        legacyConfig.exitGridPositionMode === "aggregate" &&
        totalBase > MIN_BASE_QUANTITY
      ) {
        this.state.memory.lastSignal = "buy";
        return events ?? NO_EVENTS;
      }
      if (
        this.openOrderIndexes.size >= config.maxOpenOrders ||
        tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs
      ) {
        return events ?? NO_EVENTS;
      }

      if (legacyConfig.exitGridMarketEntry) {
        const result = this.createMarketBuyOrder(
          tick.price,
          tick.eventTime,
          decision.reason,
          decision.quoteSize,
        );
        if (!result.order) {
          return events ?? NO_EVENTS;
        }

        this.state.memory.lastSignal = "buy";
        this.state.memory.lastActionAt = tick.eventTime;
        this.syncLegacyExitGridMemories(
          memory,
          this.activeLegacyLongLots(tick.price),
          tick.price,
        );

        if (events) {
          events.push(...this.immediateOrderEvents("BUY", result, tick.eventTime));
        }
        return events ?? NO_EVENTS;
      }

      const order = this.createBuyOrder(
        tick.price,
        tick.eventTime,
        decision.reason,
        decision.quoteSize,
      );
      if (!order) {
        return events ?? NO_EVENTS;
      }
      order.positionEffect = "open";

      this.state.memory.lastSignal = "buy";
      this.state.memory.lastActionAt = tick.eventTime;
      if (events) {
        events.push({
          type: "order_created",
          at: tick.eventTime,
          message: `BUY limit order created: ${decision.reason}`,
          order: structuredClone(order),
        });
      }
      return events ?? NO_EVENTS;
    }

    if (decision.signal !== "sell") {
      this.state.memory.lastSignal = decision.signal;
      return events ?? NO_EVENTS;
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return events ?? NO_EVENTS;
    }

    const activeLongs = this.activeLegacyLongLots(tick.price);
    this.syncLegacyExitGridMemories(memory, activeLongs, tick.price);

    const orders: TradingOrder[] = [];
    for (const lot of activeLongs) {
      const grid = this.ensureLegacyExitGrid(memory, lot, tick.price);
      grid.peakPrice = Math.max(grid.peakPrice, tick.price);

      const resetPeakPrice = this.legacyExitGridResetPeakPrice(grid, lot, tick.price);
      if (resetPeakPrice === undefined) {
        continue;
      }
      grid.peakPrice = resetPeakPrice;

      if (events) {
        events.push(...this.cancelLegacyExitGridOrders(tick.eventTime, true, lot.id));
      } else {
        this.cancelLegacyExitGridOrders(tick.eventTime, false, lot.id);
      }

      orders.push(...this.createLegacyExitGridOrders(grid, lot, tick.eventTime));
      if (this.openOrderIndexes.size >= config.maxOpenOrders) {
        break;
      }
    }

    if (orders.length === 0) {
      this.state.memory.lastSignal = "sell";
      return events ?? NO_EVENTS;
    }

    this.state.memory.lastSignal = "sell";
    this.state.memory.lastActionAt = tick.eventTime;

    if (events) {
      for (const order of orders) {
        events.push({
          type: "order_created",
          at: tick.eventTime,
          message: `SELL exit grid order created at ${order.price}`,
          order: structuredClone(order),
        });
      }
    }

    return events ?? NO_EVENTS;
  }

  private createBuyOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuoteSize: number,
  ): TradingOrder | undefined {
    const config = this.state.config;
    const availableQuote = this.longEntryBuyingPowerQuote(marketPrice);
    const quoteSize = Math.min(desiredQuoteSize, availableQuote);

    if (quoteSize < config.minOrderQuote) {
      return undefined;
    }

    const price = roundQuote(marketPrice * (1 - config.limitOffsetBps / 10_000));
    const feeRate = config.feeBps / 10_000;
    const quantity = roundAsset(quoteSize / price);
    const estimatedQuoteCost = roundQuote(quantity * price * (1 + feeRate));

    if (quantity <= 0) {
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
    desiredQuantity: number,
  ): TradingOrder | undefined {
    const config = this.state.config;
    const quantity = roundAsset(desiredQuantity);
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

  private createMarketBuyOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuoteSize: number,
  ): ImmediateOrderResult {
    const config = this.state.config;
    const availableQuote = this.longEntryBuyingPowerQuote(marketPrice);
    const quoteSize = Math.min(desiredQuoteSize, availableQuote);

    if (quoteSize < config.minOrderQuote) {
      return {};
    }

    const price = roundQuote(marketPrice);
    const feeRate = config.feeBps / 10_000;
    const quantity = roundAsset(quoteSize / price);
    const estimatedQuoteCost = roundQuote(quantity * price * (1 + feeRate));

    if (quantity <= 0) {
      return {};
    }

    this.state.quoteFree = roundQuote(this.state.quoteFree - estimatedQuoteCost);
    this.state.quoteReserved = roundQuote(this.state.quoteReserved + estimatedQuoteCost);

    const order = this.buildOrder(
      "buy",
      price,
      quantity,
      estimatedQuoteCost,
      createdAt,
      reason,
      "market",
    );
    const index = this.state.orders.length;
    order.positionEffect = "open";
    this.state.orders.push(order);
    this.openOrderIndexes.add(index);

    const result = this.tryFillOrder(order, index, createdAt);
    return {
      order: this.state.orders[index],
      fill: result.fill,
      cancelled: result.cancelled,
    };
  }

  private createTriggeredSellOrder(
    price: number,
    quantity: number,
    createdAt: number,
    reason: string,
    targetPositionId?: string,
  ): TradingOrder | undefined {
    const roundedQuantity = roundAsset(quantity);
    const quoteSize = roundedQuantity * price;

    if (
      roundedQuantity <= 0 ||
      quoteSize < this.state.config.minOrderQuote ||
      roundedQuantity > this.state.baseFree
    ) {
      return undefined;
    }

    this.state.baseFree = roundAsset(this.state.baseFree - roundedQuantity);
    this.state.baseReserved = roundAsset(this.state.baseReserved + roundedQuantity);

    const order = this.buildOrder(
      "sell",
      roundQuote(price),
      roundedQuantity,
      0,
      createdAt,
      reason,
      "limit",
      "below",
    );
    if (targetPositionId) {
      order.targetPositionId = targetPositionId;
      order.positionEffect = "close";
    }
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
    type: "limit" | "market" = "limit",
    trigger?: "above" | "below",
  ): TradingOrder {
    const id = `ord_${this.nextSequence().toString().padStart(6, "0")}`;
    return {
      id,
      side,
      type,
      trigger,
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

  private immediateOrderEvents(
    side: "BUY" | "SELL",
    result: ImmediateOrderResult,
    at: number,
  ): BotEvent[] {
    if (result.fill && result.order) {
      return [
        {
          type: "order_filled",
          at,
          message: `${side} market order filled at ${result.order.price}`,
          order: structuredClone(result.order),
          fill: structuredClone(result.fill),
        },
      ];
    }

    if (result.cancelled) {
      return [
        {
          type: "order_cancelled",
          at,
          message: `${side} market order cancelled by leverage limit`,
          order: structuredClone(result.cancelled),
        },
      ];
    }

    return [];
  }

  private activeLegacyLongLots(price: number): LegacyExitGridLot[] {
    if (this.state.config.legacyValleyPeak.exitGridPositionMode === "aggregate") {
      const totalBase = this.totalBase();
      const averagePrice = cleanPositive(this.state.avgEntryPrice) || price;
      if (totalBase <= MIN_BASE_QUANTITY || averagePrice <= 0) {
        return [];
      }

      const feeRate = this.state.config.feeBps / 10_000;
      return [
        {
          id: "aggregate-long",
          averagePrice,
          originalQuantity: totalBase,
          remainingQuantity: totalBase,
          breakEvenSellPrice: averagePrice / Math.max(0.000001, 1 - feeRate),
        },
      ];
    }

    this.updateLegacyLongLotCache();
    const lots = this.legacyLongLotCache?.lots.values() ?? [];
    return [...lots].filter((lot) => lot.remainingQuantity > MIN_BASE_QUANTITY);
  }

  private updateLegacyLongLotCache(): void {
    const feeAndSlippageRate =
      (this.state.config.feeBps + this.state.config.positionRisk.marketSlippageBps) / 10_000;
    const quantityFloor = Math.max(MIN_BASE_QUANTITY, this.state.config.positionRisk.quantityFloor);

    if (
      !this.legacyLongLotCache ||
      this.legacyLongLotCache.processedFillsLength > this.state.fills.length
    ) {
      this.legacyLongLotCache = {
        processedFillsLength: 0,
        lots: new Map(),
      };
    }

    const cache = this.legacyLongLotCache;
    for (
      let index = cache.processedFillsLength;
      index < this.state.fills.length;
      index += 1
    ) {
      const fill = this.state.fills[index];
      if (fill.side === "buy") {
        if (fill.positionEffect !== "open") {
          continue;
        }

        const costQuote = roundQuote(fill.quoteQuantity + fill.feeQuote);
        const averagePrice = roundQuote(costQuote / fill.quantity);
        cache.lots.set(`long_${fill.id}`, {
          id: `long_${fill.id}`,
          averagePrice,
          originalQuantity: fill.quantity,
          remainingQuantity: fill.quantity,
          remainingCostQuote: costQuote,
          breakEvenSellPrice: roundQuote(
            (costQuote / Math.max(fill.quantity, quantityFloor)) * (1 + feeAndSlippageRate),
          ),
        });
        continue;
      }

      if (fill.side !== "sell" || fill.positionEffect === "open") {
        continue;
      }

      this.closeLegacyTrackedLongLots(cache, fill, feeAndSlippageRate, quantityFloor);
    }

    cache.processedFillsLength = this.state.fills.length;
  }

  private closeLegacyTrackedLongLots(
    cache: LegacyLongLotCache,
    fill: TradeFill,
    feeAndSlippageRate: number,
    quantityFloor: number,
  ): void {
    let quantityLeft = fill.quantity;
    const unitProceeds = (fill.quoteQuantity - fill.feeQuote) / fill.quantity;
    const targets = fill.targetPositionId
      ? [cache.lots.get(fill.targetPositionId)].filter(
          (lot): lot is LegacyTrackedLongLot => lot !== undefined,
        )
      : [...cache.lots.values()];

    for (const lot of targets) {
      if (quantityLeft <= MIN_BASE_QUANTITY) {
        break;
      }

      const quantity = Math.min(quantityLeft, lot.remainingQuantity);
      if (quantity <= MIN_BASE_QUANTITY) {
        continue;
      }

      lot.remainingQuantity = roundAsset(lot.remainingQuantity - quantity);
      lot.remainingCostQuote = roundQuote(lot.remainingCostQuote - quantity * unitProceeds);
      lot.breakEvenSellPrice =
        lot.remainingQuantity > MIN_BASE_QUANTITY
          ? roundQuote(
              (lot.remainingCostQuote / Math.max(lot.remainingQuantity, quantityFloor)) *
                (1 + feeAndSlippageRate),
            )
          : 0;
      quantityLeft = roundAsset(quantityLeft - quantity);

      if (lot.remainingQuantity <= MIN_BASE_QUANTITY) {
        cache.lots.delete(lot.id);
      }
    }
  }

  private updateLegacyExitGridPeaks(
    memory: LegacyValleyPeakMemory,
    price: number,
  ): void {
    for (const grid of Object.values(memory.exitGrids ?? {})) {
      grid.peakPrice = Math.max(grid.peakPrice, price);
    }
  }

  private syncLegacyExitGridMemories(
    memory: LegacyValleyPeakMemory,
    lots: LegacyExitGridLot[],
    price: number,
  ): void {
    const grids = (memory.exitGrids ??= {});
    const activeLotIds = new Set(lots.map((lot) => lot.id));
    const openGridIds = new Map<string, string[]>();
    for (const { order } of this.openLegacyExitGridOrders()) {
      if (!order.targetPositionId) {
        continue;
      }
      const ids = openGridIds.get(order.targetPositionId) ?? [];
      ids.push(order.id);
      openGridIds.set(order.targetPositionId, ids);
    }

    for (const lotId of Object.keys(grids)) {
      if (!activeLotIds.has(lotId)) {
        delete grids[lotId];
        continue;
      }
      grids[lotId].gridOrderIds = grids[lotId].gridOrderIds.filter((id) =>
        (openGridIds.get(lotId) ?? []).includes(id),
      );
    }

    for (const lot of lots) {
      const grid = this.ensureLegacyExitGrid(memory, lot, price);
      grid.peakPrice = Math.max(grid.peakPrice, price);
      grid.gridOrderIds = openGridIds.get(lot.id) ?? grid.gridOrderIds;
    }
  }

  private ensureLegacyExitGrid(
    memory: LegacyValleyPeakMemory,
    lot: LegacyExitGridLot,
    price: number,
  ): LegacyExitGridMemory {
    const grids = (memory.exitGrids ??= {});
    let grid = grids[lot.id];
    if (!grid) {
      const entryPrice = cleanPositive(lot.averagePrice) || price;
      grid = {
        lotId: lot.id,
        entryPrice,
        entryQuantity: lot.originalQuantity,
        peakPrice: Math.max(entryPrice, price),
        gridPeakPrice: 0,
        gridOrderIds: [],
      };
      grids[lot.id] = grid;
    }
    return grid;
  }

  private legacyExitGridResetPeakPrice(
    grid: LegacyExitGridMemory,
    lot: LegacyExitGridLot,
    currentPrice: number,
  ): number | undefined {
    const config = this.state.config.legacyValleyPeak;
    const breakEvenSellPrice = Math.max(lot.breakEvenSellPrice, grid.entryPrice);
    const minimumPeakPrice = breakEvenSellPrice * (1 + config.exitGridMinProfitBps / 10_000);
    const peakPrice = Math.max(grid.peakPrice, currentPrice);

    if (grid.gridOrderIds.length === 0 || grid.gridPeakPrice <= 0) {
      return peakPrice > minimumPeakPrice ? peakPrice : undefined;
    }

    if (
      config.exitGridResetMode === "higher-peak" &&
      peakPrice >= grid.gridPeakPrice * (1 + config.exitGridResetBps / 10_000)
    ) {
      return peakPrice > minimumPeakPrice ? peakPrice : undefined;
    }

    if (
      config.exitGridResetMode === "filled-grid" &&
      this.hasCrossedFilledLegacyExitGridPoint(lot.id, currentPrice)
    ) {
      return currentPrice > minimumPeakPrice ? currentPrice : undefined;
    }

    return undefined;
  }

  private hasCrossedFilledLegacyExitGridPoint(lotId: string, peakPrice: number): boolean {
    return this.state.orders.some(
      (order) =>
        order.status === "filled" &&
        order.targetPositionId === lotId &&
        order.reason.startsWith(LEGACY_EXIT_GRID_REASON) &&
        peakPrice > order.price,
    );
  }

  private createLegacyExitGridOrders(
    grid: LegacyExitGridMemory,
    lot: LegacyExitGridLot,
    createdAt: number,
  ): TradingOrder[] {
    const config = this.state.config;
    const legacyConfig = config.legacyValleyPeak;
    const lowerPrice = roundQuote(Math.max(lot.breakEvenSellPrice, grid.entryPrice));
    const upperPrice = roundQuote(Math.max(grid.peakPrice, lowerPrice));
    const availableSlots = Math.max(0, config.maxOpenOrders - this.openOrderIndexes.size);
    const orderCount = Math.min(legacyConfig.exitGridOrderCount, availableSlots);
    const availableQuantity = Math.min(lot.remainingQuantity, this.state.baseFree);

    if (orderCount <= 0 || upperPrice <= lowerPrice || availableQuantity <= MIN_BASE_QUANTITY) {
      return [];
    }

    const orders: TradingOrder[] = [];
    let remainingQuantity = availableQuantity;

    for (let index = 0; index < orderCount; index += 1) {
      const price = this.legacyExitGridOrderPrice(
        index,
        orderCount,
        lowerPrice,
        upperPrice,
      );
      const desiredQuantity = this.legacyExitGridOrderQuantity(
        index,
        orderCount,
        remainingQuantity,
      );
      const quantity = roundAsset(Math.min(remainingQuantity, desiredQuantity));

      if (quantity <= MIN_BASE_QUANTITY) {
        break;
      }
      if (quantity * price < config.minOrderQuote) {
        if (index === orderCount - 1 && orders.length === 0) {
          break;
        }
        continue;
      }

      const order = this.createTriggeredSellOrder(
        price,
        quantity,
        createdAt,
        `${LEGACY_EXIT_GRID_REASON}; lot ${lot.id}; entry ${roundQuote(grid.entryPrice)}; peak ${upperPrice}`,
        lot.id,
      );
      if (!order) {
        break;
      }

      orders.push(order);
      remainingQuantity = roundAsset(remainingQuantity - quantity);
      if (remainingQuantity <= MIN_BASE_QUANTITY) {
        break;
      }
    }

    grid.gridPeakPrice = upperPrice;
    grid.gridOrderIds = orders.map((order) => order.id);
    return orders;
  }

  private legacyExitGridOrderPrice(
    index: number,
    orderCount: number,
    lowerPrice: number,
    upperPrice: number,
  ): number {
    if (orderCount <= 1) {
      return lowerPrice;
    }

    const progress = index / Math.max(1, orderCount - 1);
    if (
      this.state.config.legacyValleyPeak.exitGridPriceDistribution === "geometric" &&
      lowerPrice > 0 &&
      upperPrice > lowerPrice
    ) {
      return roundQuote(upperPrice * Math.pow(lowerPrice / upperPrice, progress));
    }

    return roundQuote(upperPrice - (upperPrice - lowerPrice) * progress);
  }

  private legacyExitGridOrderQuantity(
    index: number,
    orderCount: number,
    remainingQuantity: number,
  ): number {
    if (index >= orderCount - 1) {
      return remainingQuantity;
    }

    const config = this.state.config.legacyValleyPeak;
    const slotsLeft = Math.max(1, orderCount - index);
    if (config.exitGridSizeDistribution === "constant") {
      return remainingQuantity / slotsLeft;
    }

    if (config.exitGridSizeDistribution === "linear") {
      const remainingWeight = (slotsLeft * (slotsLeft + 1)) / 2;
      return remainingQuantity * (slotsLeft / remainingWeight);
    }

    return remainingQuantity * config.exitGridSellFraction;
  }

  private cancelLegacyExitGridOrders(
    at: number,
    collectEvents: boolean,
    lotId?: string,
  ): BotEvent[] {
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;

    for (const { index, order } of this.openLegacyExitGridOrders(lotId)) {
      this.releaseOrderReserve(order);
      order.status = "cancelled";
      order.cancelledAt = at;
      order.updatedAt = at;
      order.reason = `${order.reason}; reset`;
      this.openOrderIndexes.delete(index);

      if (events) {
        events.push({
          type: "order_cancelled",
          at,
          message: `SELL exit grid order cancelled for reset`,
          order: structuredClone(order),
        });
      }
    }

    return events ?? NO_EVENTS;
  }

  private openLegacyExitGridOrders(lotId?: string): Array<{ index: number; order: TradingOrder }> {
    const orders: Array<{ index: number; order: TradingOrder }> = [];
    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (
        order?.status === "open" &&
        order.reason.startsWith(LEGACY_EXIT_GRID_REASON) &&
        (!lotId || order.targetPositionId === lotId)
      ) {
        orders.push({ index, order });
      }
    }
    return orders;
  }

  private fillOpenOrders(tick: PriceTick, collectEvents: boolean): BotEvent[] {
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;

    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (order?.status !== "open") {
        this.openOrderIndexes.delete(index);
        continue;
      }

      const trigger = order.trigger ?? (order.side === "buy" ? "below" : "above");
      const canFill = trigger === "below" ? tick.price <= order.price : tick.price >= order.price;

      if (!canFill) {
        continue;
      }

      const result = this.tryFillOrder(order, index, tick.eventTime);
      if (result.cancelled) {
        if (events) {
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
      if (!events) {
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

    return events ?? NO_EVENTS;
  }

  private tryFillOrder(
    order: TradingOrder,
    index: number,
    filledAt: number,
    skipLeverageCheck = false,
  ): { fill?: TradeFill; cancelled?: TradingOrder } {
    const rollback = this.captureImmediateFillRollback();
    const orderBeforeFill = { ...order };

    try {
      const fill = this.fillOrder(order, index, filledAt);
      if (!skipLeverageCheck) {
        this.assertLeverageLimit();
      }
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
      const entryPrice = this.longEntryPriceForCloseOrder(order);
      this.state.quoteFree = roundQuote(this.state.quoteFree + proceeds);
      realizedPnl = roundQuote((order.price - entryPrice) * order.quantity - feeQuote);
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
      liquidation: order.liquidation,
      liquidatedPositionCount: order.liquidatedPositionCount,
    };
    this.state.fills.push(fill);
    if (fill.liquidation) {
      this.liquidatedPositionCountValue += Math.max(
        1,
        fill.liquidatedPositionCount ?? 1,
      );
    }

    return fill;
  }

  private liquidateAccountIfNeeded(
    at: number,
    price: number,
    collectEvents: boolean,
    existingEvents?: BotEvent[],
  ): boolean {
    const liquidationPrice = this.accountLiquidationPrice();
    if (!liquidationPrice || !Number.isFinite(liquidationPrice) || liquidationPrice <= 0) {
      return false;
    }

    const baseQuantity = this.totalBase();
    const crossed =
      baseQuantity > MIN_BASE_QUANTITY
        ? price <= liquidationPrice
        : baseQuantity < -MIN_BASE_QUANTITY
          ? price >= liquidationPrice
          : false;
    if (!crossed) {
      return false;
    }

    const events = existingEvents ?? (collectEvents ? [] : undefined);
    const positions = analyzePositions(this.state, { currentPrice: liquidationPrice });
    const activeLongs = positions.longs.filter(
      (lot) => lot.status !== "pending" && lot.remainingQuantity > MIN_BASE_QUANTITY,
    );
    const activeShorts = positions.shorts.filter(
      (lot) => lot.status !== "pending" && lot.remainingQuantity > MIN_BASE_QUANTITY,
    );
    if (activeLongs.length === 0 && activeShorts.length === 0) {
      return false;
    }

    this.cancelAllOpenOrdersForLiquidation(at, events);

    const shortQuantity = roundAsset(
      activeShorts.reduce((quantity, lot) => quantity + lot.remainingQuantity, 0),
    );
    const longQuantity = roundAsset(
      activeLongs.reduce((quantity, lot) => quantity + lot.remainingQuantity, 0),
    );

    if (shortQuantity > MIN_BASE_QUANTITY) {
      const result = this.createLiquidationMarketOrder(
        "buy",
        liquidationPrice,
        shortQuantity,
        activeShorts.length,
        at,
      );
      if (result && events) {
        events.push(result);
      }
    }

    if (longQuantity > MIN_BASE_QUANTITY) {
      const result = this.createLiquidationMarketOrder(
        "sell",
        liquidationPrice,
        longQuantity,
        activeLongs.length,
        at,
      );
      if (result && events) {
        events.push(result);
      }
    }

    const memory = this.state.memory.legacyValleyPeak;
    if (memory?.exitGrids) {
      memory.exitGrids = {};
    }
    recalculateMetrics(this.state);
    return true;
  }

  private cancelAllOpenOrdersForLiquidation(
    at: number,
    events: BotEvent[] | undefined,
  ): void {
    for (const index of [...this.openOrderIndexes]) {
      const order = this.state.orders[index];
      if (order?.status !== "open") {
        this.openOrderIndexes.delete(index);
        continue;
      }

      this.releaseOrderReserve(order);
      order.status = "cancelled";
      order.cancelledAt = at;
      order.updatedAt = at;
      order.reason = `${order.reason}; account liquidation`;
      this.openOrderIndexes.delete(index);

      if (events) {
        events.push({
          type: "order_cancelled",
          at,
          message: `${order.side.toUpperCase()} order cancelled by account liquidation`,
          order: structuredClone(order),
        });
      }
    }
  }

  private createLiquidationMarketOrder(
    side: "buy" | "sell",
    price: number,
    quantity: number,
    liquidatedPositionCount: number,
    at: number,
  ): BotEvent | undefined {
    const roundedQuantity = roundAsset(quantity);
    if (roundedQuantity <= MIN_BASE_QUANTITY || price <= 0) {
      return undefined;
    }

    const feeRate = this.state.config.feeBps / 10_000;
    const quoteQuantity = roundQuote(price * roundedQuantity);
    const estimatedQuoteCost =
      side === "buy" ? roundQuote(quoteQuantity * (1 + feeRate)) : 0;

    if (side === "buy") {
      this.state.quoteFree = roundQuote(this.state.quoteFree - estimatedQuoteCost);
      this.state.quoteReserved = roundQuote(this.state.quoteReserved + estimatedQuoteCost);
    } else {
      this.state.baseFree = roundAsset(this.state.baseFree - roundedQuantity);
      this.state.baseReserved = roundAsset(this.state.baseReserved + roundedQuantity);
    }

    const order = this.buildOrder(
      side,
      price,
      roundedQuantity,
      estimatedQuoteCost,
      at,
      "account liquidation",
      "market",
    );
    order.positionEffect = "close";
    order.liquidation = true;
    order.liquidatedPositionCount = liquidatedPositionCount;

    const index = this.state.orders.length;
    this.state.orders.push(order);
    this.openOrderIndexes.add(index);
    const result = this.tryFillOrder(order, index, at, true);
    if (!result.fill) {
      return undefined;
    }

    return {
      type: "order_filled",
      at,
      message: `${side.toUpperCase()} liquidation order filled at ${price}`,
      order: structuredClone(this.state.orders[index]),
      fill: structuredClone(result.fill),
    };
  }

  private longEntryPriceForCloseOrder(order: TradingOrder): number {
    if (!order.targetPositionId) {
      return this.state.avgEntryPrice;
    }

    const cachedLot = this.legacyLongLotCache?.lots.get(order.targetPositionId);
    if (cachedLot) {
      return cachedLot.averagePrice;
    }

    const target = analyzePositions(this.state, {
      currentPrice: this.state.lastPrice || order.price,
    }).longs.find((lot) => lot.id === order.targetPositionId);

    return cleanPositive(target?.averagePrice) || this.state.avgEntryPrice;
  }

  private cancelStaleOrders(at: number, collectEvents: boolean): BotEvent[] {
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;

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

      if (events) {
        events.push({
          type: "order_cancelled",
          at,
          message: `${order.side.toUpperCase()} order cancelled after waiting too long`,
          order: structuredClone(order),
        });
      }
    }

    return events ?? NO_EVENTS;
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

  private equityAt(price: number): number {
    return roundQuote(
      this.state.quoteFree +
        this.state.quoteReserved +
        (this.state.baseFree + this.state.baseReserved) * price,
    );
  }

  private totalBase(): number {
    return this.state.baseFree + this.state.baseReserved;
  }

  private pendingLongEntryQuote(): number {
    let pendingQuote = 0;
    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (order?.status === "open" && order.side === "buy") {
        pendingQuote += order.price * order.quantity;
      }
    }
    return roundQuote(pendingQuote);
  }

  private targetLongEntryLeverage(_marketPrice: number): number {
    return clamp(cleanPositive(this.state.config.maxLeverage) || 1, 1, 999);
  }

  private longEntryBuyingPowerQuote(
    marketPrice: number,
    targetEntryLeverage = this.targetLongEntryLeverage(marketPrice),
  ): number {
    const config = this.state.config;
    const equity = this.equityAt(marketPrice);
    if (equity <= 0 || marketPrice <= 0) {
      return 0;
    }

    const feeRate = config.feeBps / 10_000;
    const longExposureQuote = Math.max(0, this.totalBase() * marketPrice);
    const pendingLongEntryQuote = this.pendingLongEntryQuote();
    const positionCapacity = Math.max(
      0,
      config.maxPositionQuote - longExposureQuote - pendingLongEntryQuote,
    );
    const leverageCapacity = Math.max(
      0,
      (
        targetEntryLeverage * equity -
        longExposureQuote -
        pendingLongEntryQuote * (1 + targetEntryLeverage * feeRate)
      ) /
        (1 + targetEntryLeverage * feeRate),
    );

    return roundQuote(Math.min(positionCapacity, leverageCapacity));
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
      liquidatedPositionCount: this.liquidatedPositionCountValue,
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
    this.liquidatedPositionCountValue = rollback.liquidatedPositionCount;
    this.state.metrics = rollback.metrics;
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

  private rememberPrice(price: number): void {
    const maxPrices = this.priceMemoryLimit;
    this.state.memory.prices.push(price);
    if (this.state.memory.prices.length > maxPrices * 2) {
      this.state.memory.prices.splice(0, this.state.memory.prices.length - maxPrices);
    }
  }

  private ensureLegacyValleyPeakMemory(): LegacyValleyPeakMemory {
    let memory = this.state.memory.legacyValleyPeak;
    const rangeCount = this.state.config.legacyValleyPeak.averagingRangesSec.length;
    if (
      !memory ||
      memory.buyAverages.length !== rangeCount ||
      memory.sellAverages.length !== rangeCount
    ) {
      memory = normalizeLegacyValleyPeakMemory(
        memory,
        this.state.config.legacyValleyPeak,
      );
      this.state.memory.legacyValleyPeak = memory;
    }

    return memory;
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
  overrides: PartialStrategyConfig,
): PaperBotState {
  const config = createStrategyConfig(mergeStrategyOverrides(state.config, overrides));
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

function mergeStrategyOverrides(
  base: StrategyConfig,
  overrides: PartialStrategyConfig,
): PartialStrategyConfig {
  return {
    symbol: overrides.symbol ?? base.symbol,
    baseAsset: overrides.baseAsset ?? base.baseAsset,
    quoteAsset: overrides.quoteAsset ?? base.quoteAsset,
    algorithm: overrides.algorithm ?? base.algorithm,
    startingQuote: overrides.startingQuote ?? base.startingQuote,
    maxLeverage: overrides.maxLeverage ?? base.maxLeverage,
    feeBps: overrides.feeBps ?? base.feeBps,
    maxPositionQuote: overrides.maxPositionQuote ?? base.maxPositionQuote,
    limitOffsetBps: overrides.limitOffsetBps ?? base.limitOffsetBps,
    maxOpenOrders: overrides.maxOpenOrders ?? base.maxOpenOrders,
    cooldownMs: overrides.cooldownMs ?? base.cooldownMs,
    staleOrderMs: overrides.staleOrderMs ?? base.staleOrderMs,
    minOrderQuote: overrides.minOrderQuote ?? base.minOrderQuote,
    legacyValleyPeak: {
      ...base.legacyValleyPeak,
      ...(overrides.legacyValleyPeak ?? {}),
    },
    positionRisk: {
      ...base.positionRisk,
      ...(overrides.positionRisk ?? {}),
    },
  };
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
    config.legacyValleyPeak.averagingRangesSec.length * 100,
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

function countLiquidatedPositions(fills: readonly TradeFill[]): number {
  return fills.reduce(
    (count, fill) =>
      fill.liquidation ? count + Math.max(1, fill.liquidatedPositionCount ?? 1) : count,
    0,
  );
}

function cleanPositive(value: number | undefined): number {
  return Number.isFinite(value) && (value as number) > 0 ? (value as number) : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function isLeverageLimitError(error: unknown): boolean {
  return error instanceof Error && error.message.startsWith("Leverage limit exceeded:");
}

function formatLeverageForError(value: number): string {
  return (Number.isFinite(value) ? value : 999).toFixed(2);
}
