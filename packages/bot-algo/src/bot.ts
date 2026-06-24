import type {
  BotEvent,
  BotStatus,
  BotMetrics,
  Candle,
  ExchangeOrderUpdate,
  ExchangeReconciliationInput,
  ExchangeTradeFill,
  LegacyExitGridMemory,
  LegacyValleyPeakMemory,
  ManualTradeInput,
  PaperBotState,
  PositionLotSide,
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
  maxLeverage: 1,
  shortMarginModel: "futures-margin",
  longBorrowDepth: 999,
  shortBorrowDepth: 999,
  lockBorrowedLenderCollateral: false,
  borrowerProfitShareToLender: 1,
  feeBps: 7.5,
  maxPositionQuote: 10_000,
  limitOffsetBps: 2,
  maxOpenOrders: 1024,
  cooldownMs: 300_000,
  staleOrderMs: 30 * 24 * 60 * 60 * 1000,
  minOrderQuote: 5,
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
  exitGridSpanTotal: number;
  exitGridSpanCount: number;
  exitGridOrderCountTotal: number;
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
  side: PositionLotSide;
  averagePrice: number;
  originalQuantity: number;
  remainingQuantity: number;
}

interface LegacyLongExitGridLot extends LegacyExitGridLot {
  side: "long";
  breakEvenSellPrice: number;
  lentQuantity?: number;
}

interface LegacyShortExitGridLot extends LegacyExitGridLot {
  side: "short";
  breakEvenBuyPrice: number;
  lentQuote?: number;
}

interface LegacyPositionLotCache {
  processedFillsLength: number;
  longs: Map<string, LegacyTrackedLongLot>;
  shorts: Map<string, LegacyTrackedShortLot>;
}

interface LegacyTrackedLongLot extends LegacyLongExitGridLot {
  openedAt: number;
  remainingCostQuote: number;
  lentQuantity: number;
  borrowDepthRemaining: number;
  borrowAllocations: LegacyLongBorrowAllocation[];
}

interface LegacyTrackedShortLot extends LegacyShortExitGridLot {
  openedAt: number;
  remainingProceedsQuote: number;
  lentQuote: number;
  borrowDepthRemaining: number;
  borrowAllocations: LegacyShortBorrowAllocation[];
}

interface LegacyShortBorrowAllocation {
  longLotId: string;
  quantity: number;
  quote: number;
  depthRemaining: number;
}

interface LegacyLongBorrowAllocation {
  shortLotId: string;
  quantity: number;
  quote: number;
  depthRemaining: number;
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
    shortMarginModel: normalizeShortMarginModel(
      overrides.shortMarginModel ?? defaultStrategyConfig.shortMarginModel,
    ),
    longBorrowDepth: normalizeBorrowDepth(
      overrides.longBorrowDepth ?? defaultStrategyConfig.longBorrowDepth,
    ),
    shortBorrowDepth: normalizeBorrowDepth(
      overrides.shortBorrowDepth ?? defaultStrategyConfig.shortBorrowDepth,
    ),
    lockBorrowedLenderCollateral:
      overrides.lockBorrowedLenderCollateral ??
      defaultStrategyConfig.lockBorrowedLenderCollateral,
    borrowerProfitShareToLender: clamp(
      cleanFiniteNumber(
        overrides.borrowerProfitShareToLender ??
          defaultStrategyConfig.borrowerProfitShareToLender,
      ),
      0,
      1,
    ),
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
  config.maxLeverage = clamp(cleanPositive(config.maxLeverage) || 1, 1, 999);
  config.longBorrowDepth = normalizeBorrowDepth(config.longBorrowDepth);
  config.shortBorrowDepth = normalizeBorrowDepth(config.shortBorrowDepth);
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
    exitGridSpanTotal: 0,
    exitGridSpanCount: 0,
    exitGridOrderCountTotal: 0,
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
  private legacyPositionLotCache?: LegacyPositionLotCache;
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
    this.legacyPositionLotCache = undefined;
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

  setConfig(config: StrategyConfig, at = Date.now()): void {
    this.state.config = createStrategyConfig(config);
    this.state.updatedAt = at;
    this.priceMemoryLimit = priceMemoryLimit(this.state.config);
  }

  cancelOpenOrder(orderId: string, reason: string, at = Date.now()): BotEvent[] {
    const index = this.findOrderIndex(orderId);
    if (index === undefined) {
      return [];
    }
    const order = this.state.orders[index];
    if (!order || order.status !== "open") {
      return [];
    }

    this.releaseOrderReserve(order);
    order.status = "cancelled";
    order.cancelledAt = at;
    order.updatedAt = at;
    order.reason = `${order.reason}; ${reason}`;
    this.openOrderIndexes.delete(index);
    this.state.updatedAt = at;
    recalculateMetrics(this.state);

    return [
      {
        type: "order_cancelled",
        at,
        message: `${order.side.toUpperCase()} order cancelled: ${reason}`,
        order: structuredClone(order),
        state: this.snapshot(),
      },
    ];
  }

  cancelOpenOrders(reason: string, at = Date.now()): BotEvent[] {
    const events: BotEvent[] = [];
    for (const index of [...this.openOrderIndexes]) {
      const order = this.state.orders[index];
      if (!order || order.status !== "open") {
        this.openOrderIndexes.delete(index);
        continue;
      }

      this.releaseOrderReserve(order);
      order.status = "cancelled";
      order.cancelledAt = at;
      order.updatedAt = at;
      order.reason = `${order.reason}; ${reason}`;
      this.openOrderIndexes.delete(index);
      events.push({
        type: "order_cancelled",
        at,
        message: `${order.side.toUpperCase()} order cancelled: ${reason}`,
        order: structuredClone(order),
      });
    }

    if (events.length > 0) {
      this.state.updatedAt = at;
      recalculateMetrics(this.state);
    }
    return events;
  }

  createPositionCloseOrders(
    options: { includeUnprofitable?: boolean } = {},
    at = Date.now(),
  ): BotEvent[] {
    const price = cleanPositive(this.state.lastPrice);
    if (price <= 0) {
      return [];
    }

    const positions = analyzePositions(this.state, { currentPrice: price });
    const events: BotEvent[] = [];
    const longs = positions.longs.filter(
      (lot) =>
        lot.status !== "pending" &&
        lot.remainingQuantity > MIN_BASE_QUANTITY &&
        (options.includeUnprofitable || price >= lot.breakEvenSellPrice),
    );
    const shorts = positions.shorts.filter(
      (lot) =>
        lot.status !== "pending" &&
        lot.remainingQuantity > MIN_BASE_QUANTITY &&
        (options.includeUnprofitable || price <= lot.breakEvenBuyPrice),
    );

    for (const lot of longs) {
      const order = this.createExternalCloseOrder(
        "sell",
        price,
        lot.remainingQuantity,
        at,
        options.includeUnprofitable
          ? "forced close position"
          : "close profitable position",
        lot.id,
      );
      if (order) {
        events.push({
          type: "order_created",
          at,
          message: `SELL close order created for ${lot.id}`,
          order: structuredClone(order),
        });
      }
    }

    for (const lot of shorts) {
      const order = this.createExternalCloseOrder(
        "buy",
        price,
        lot.remainingQuantity,
        at,
        options.includeUnprofitable
          ? "forced close position"
          : "close profitable position",
        lot.id,
      );
      if (order) {
        events.push({
          type: "order_created",
          at,
          message: `BUY close order created for ${lot.id}`,
          order: structuredClone(order),
        });
      }
    }

    if (events.length > 0) {
      this.state.updatedAt = at;
      recalculateMetrics(this.state);
    }
    return events;
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
      this.refreshAverageEntryPrices();
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

  applyExchangeReconciliation(
    input: ExchangeReconciliationInput,
    at = Date.now(),
  ): BotEvent[] {
    const previousState = structuredClone(this.state);
    const events: BotEvent[] = [];

    try {
      const fills = [...(input.fills ?? [])].sort(
        (left, right) =>
          left.filledAt - right.filledAt ||
          left.id.localeCompare(right.id),
      );
      for (const fill of fills) {
        const event = this.applyExchangeFill(fill);
        if (event) {
          events.push(event);
        }
      }

      const orders = [...(input.orders ?? [])].sort(
        (left, right) =>
          (left.updatedAt ?? left.createdAt ?? at) -
          (right.updatedAt ?? right.createdAt ?? at),
      );
      for (const order of orders) {
        events.push(...this.applyExchangeOrderUpdate(order, at));
      }

      if (events.length === 0) {
        return [];
      }

      this.refreshAverageEntryPrices();
      this.state.updatedAt = Math.max(at, ...events.map((event) => event.at));
      recalculateMetrics(this.state);
      this.rebuildOpenOrderIndex();
      return events;
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
      this.evaluateStrategy(tick, false, {});
    }

    return NO_EVENTS;
  }

  warmupFromCandles(candles: readonly Candle[]): number {
    let processed = 0;
    for (const candle of candles) {
      const duration = Math.max(1, candle.closeTime - candle.openTime);
      processed += this.warmupPrice(candle.openTime, candle.open);
      processed += this.warmupPrice(candle.openTime + duration * 0.33, candle.high);
      processed += this.warmupPrice(candle.openTime + duration * 0.66, candle.low);
      processed += this.warmupPrice(candle.closeTime, candle.close);
    }
    if (processed > 0) {
      recalculateMetrics(this.state);
    }
    return processed;
  }

  private warmupPrice(eventTime: number, price: number): number {
    if (!Number.isFinite(price) || price <= 0) {
      return 0;
    }

    this.state.lastPrice = roundQuote(price);
    this.state.updatedAt = Math.max(this.state.updatedAt, eventTime);
    this.rememberPrice(price);
    const config = this.state.config;
    evaluateLegacyValleyPeak(
      this.ensureLegacyValleyPeakMemory(),
      config.legacyValleyPeak,
      {
        eventTime,
        price,
        feeRate: config.feeBps / 10_000,
        buyingPowerQuote: 0,
        shortSellingPowerQuote: 0,
        baseFree: 0,
        shortBaseFree: 0,
      },
    );
    return 1;
  }

  private applyExchangeOrderUpdate(
    update: ExchangeOrderUpdate,
    fallbackAt: number,
  ): BotEvent[] {
    const index = this.findOrderIndex(update.localOrderId);
    if (index === undefined) {
      return [];
    }

    const order = this.state.orders[index];
    if (!order) {
      return [];
    }

    const updatedAt = cleanPositive(update.updatedAt) || cleanPositive(update.createdAt) || fallbackAt;
    const missingFilledQuantity = roundAsset(
      Math.max(0, update.filledQuantity - order.filledQuantity),
    );
    this.adjustOpenOrderToExchange(order, update);

    const events: BotEvent[] = [];
    if (missingFilledQuantity > MIN_BASE_QUANTITY) {
      const filledQuote = cleanPositive(update.quoteQuantity);
      const averagePrice =
        filledQuote && update.filledQuantity > 0
          ? filledQuote / update.filledQuantity
          : update.price;
      const fill = this.applyExchangeFill({
        id: `exchange_order_${update.externalOrderId || order.id}_${update.filledQuantity}`,
        localOrderId: order.id,
        externalOrderId: update.externalOrderId,
        clientOrderId: update.clientOrderId,
        side: update.side,
        price: cleanPositive(averagePrice) || order.price,
        quantity: missingFilledQuantity,
        quoteQuantity: roundQuote((cleanPositive(averagePrice) || order.price) * missingFilledQuantity),
        feeQuote: update.feeQuote ?? 0,
        realizedPnl: undefined,
        filledAt: updatedAt,
        reason: update.reason,
        positionEffect: update.positionEffect ?? order.positionEffect,
      });
      if (fill) {
        events.push(fill);
      }
    }

    if (update.status === "cancelled" && order.status === "open") {
      this.releaseOrderReserve(order);
      order.status = "cancelled";
      order.cancelledAt = updatedAt;
      order.updatedAt = updatedAt;
      this.openOrderIndexes.delete(index);
      events.push({
        type: "order_cancelled",
        at: updatedAt,
        message: `${order.side.toUpperCase()} order cancelled on exchange`,
        order: structuredClone(order),
      });
    } else if (update.status === "filled" && order.status === "open") {
      order.status = "filled";
      order.filledAt = updatedAt;
      order.updatedAt = updatedAt;
      order.filledQuantity = Math.max(order.filledQuantity, order.quantity);
      this.openOrderIndexes.delete(index);
    } else {
      order.updatedAt = Math.max(order.updatedAt, updatedAt);
    }

    return events;
  }

  private applyExchangeFill(input: ExchangeTradeFill): BotEvent | undefined {
    if (!input.id || this.state.fills.some((fill) => fill.id === input.id)) {
      return undefined;
    }

    const index = this.findOrCreateExchangeOrder(input);
    const order = this.state.orders[index];
    if (!order || order.status === "cancelled") {
      return undefined;
    }
    if (order.status === "filled" && order.filledQuantity >= order.quantity - MIN_BASE_QUANTITY) {
      return undefined;
    }

    const remainingQuantity = roundAsset(Math.max(0, order.quantity - order.filledQuantity));
    const quantity = roundAsset(input.quantity);
    const price = roundQuote(input.price);
    if (quantity <= MIN_BASE_QUANTITY || price <= 0) {
      return undefined;
    }
    if (quantity > remainingQuantity + MIN_BASE_QUANTITY) {
      order.quantity = roundAsset(order.filledQuantity + quantity);
      if (order.side === "buy") {
        order.estimatedQuoteCost = Math.max(
          order.estimatedQuoteCost,
          roundQuote((cleanPositive(input.quoteQuantity) || price * quantity) + input.feeQuote),
        );
      }
    }

    const quoteQuantity = roundQuote(
      cleanPositive(input.quoteQuantity) || price * quantity,
    );
    const feeQuote = roundQuote(Math.max(0, input.feeQuote));
    const positionEffect = input.positionEffect ?? order.positionEffect ?? "auto";
    const orderView: TradingOrder = {
      ...order,
      price,
      quantity,
      positionEffect,
    };
    let realizedPnl = 0;

    if (input.side === "buy") {
      const closedShortQuantity = this.shortCloseQuantityForOrder(orderView);
      const spent = roundQuote(quoteQuantity + feeQuote);
      this.consumeQuoteReserveForExchangeFill(order, quantity, spent);
      this.state.baseFree = roundAsset(this.state.baseFree + quantity);
      if (closedShortQuantity > MIN_BASE_QUANTITY) {
        realizedPnl =
          input.realizedPnl !== undefined && Number.isFinite(input.realizedPnl)
            ? roundQuote(input.realizedPnl)
            : roundQuote(
                (this.shortEntryPriceForCloseOrder(orderView) - price) *
                  closedShortQuantity -
                  feeQuote * (closedShortQuantity / quantity),
              );
      }
    } else {
      const closedLongQuantity = this.longCloseQuantityForOrder(orderView);
      this.consumeBaseReserveForExchangeFill(order, quantity);
      const proceeds = roundQuote(quoteQuantity - feeQuote);
      this.state.quoteFree = roundQuote(this.state.quoteFree + proceeds);
      if (closedLongQuantity > MIN_BASE_QUANTITY) {
        realizedPnl =
          input.realizedPnl !== undefined && Number.isFinite(input.realizedPnl)
            ? roundQuote(input.realizedPnl)
            : roundQuote(
                (price - this.longEntryPriceForCloseOrder(orderView)) *
                  closedLongQuantity -
                  feeQuote * (closedLongQuantity / quantity),
              );
      }
    }

    this.state.realizedPnl = roundQuote(this.state.realizedPnl + realizedPnl);
    if (realizedPnl > 0) {
      this.state.winningTrades += 1;
    } else if (realizedPnl < 0) {
      this.state.losingTrades += 1;
    }
    this.state.feesPaid = roundQuote(this.state.feesPaid + feeQuote);

    order.price = price;
    order.filledQuantity = roundAsset(order.filledQuantity + quantity);
    order.updatedAt = input.filledAt;
    order.feeQuote = roundQuote(order.feeQuote + feeQuote);
    order.realizedPnl = roundQuote(order.realizedPnl + realizedPnl);
    order.positionEffect = positionEffect;
    if (order.filledQuantity >= order.quantity - MIN_BASE_QUANTITY) {
      order.status = "filled";
      order.filledAt = input.filledAt;
      this.openOrderIndexes.delete(index);
    }

    const fill: TradeFill = {
      id: input.id,
      orderId: order.id,
      side: input.side,
      price,
      quantity,
      quoteQuantity,
      feeQuote,
      realizedPnl,
      filledAt: input.filledAt,
      reason: input.reason?.trim() || order.reason || "exchange fill",
      targetPositionId: order.targetPositionId,
      positionEffect,
      manual: false,
    };
    this.state.fills.push(fill);
    this.state.lastPrice = price;

    return {
      type: "order_filled",
      at: input.filledAt,
      message: `${input.side.toUpperCase()} exchange fill reconciled at ${price}`,
      order: structuredClone(order),
      fill: structuredClone(fill),
      state: this.snapshot(),
    };
  }

  private findOrCreateExchangeOrder(input: ExchangeTradeFill): number {
    const existingIndex = this.findOrderIndex(input.localOrderId);
    if (existingIndex !== undefined) {
      return existingIndex;
    }

    const quantity = roundAsset(input.quantity);
    const price = roundQuote(input.price);
    const quoteQuantity = roundQuote(cleanPositive(input.quoteQuantity) || price * quantity);
    const order: TradingOrder = {
      id: input.localOrderId || `ex_ord_${this.nextSequence().toString().padStart(6, "0")}`,
      side: input.side,
      type: "limit",
      status: "open",
      price,
      quantity,
      filledQuantity: 0,
      estimatedQuoteCost: input.side === "buy" ? roundQuote(quoteQuantity + input.feeQuote) : 0,
      createdAt: input.filledAt,
      updatedAt: input.filledAt,
      reason: input.reason?.trim() || "exchange fill",
      realizedPnl: 0,
      feeQuote: 0,
      positionEffect: input.positionEffect ?? "auto",
      manual: false,
    };
    const index = this.state.orders.length;
    this.state.orders.push(order);
    this.openOrderIndexes.add(index);
    return index;
  }

  private findOrderIndex(orderId: string | undefined): number | undefined {
    if (!orderId) {
      return undefined;
    }
    const index = this.state.orders.findIndex((order) => order.id === orderId);
    return index >= 0 ? index : undefined;
  }

  private adjustOpenOrderToExchange(
    order: TradingOrder,
    update: ExchangeOrderUpdate,
  ): void {
    if (order.status !== "open") {
      return;
    }

    const nextQuantity = roundAsset(update.quantity);
    const nextPrice = roundQuote(update.price);
    if (nextQuantity <= MIN_BASE_QUANTITY || nextPrice <= 0) {
      return;
    }

    const currentRemainingQuantity = roundAsset(
      Math.max(0, order.quantity - order.filledQuantity),
    );
    const nextRemainingQuantity = roundAsset(
      Math.max(0, nextQuantity - order.filledQuantity),
    );

    if (order.side === "buy") {
      const currentRemainingReserve = remainingOrderReserveQuote(order);
      const nextEstimatedQuoteCost = roundQuote(
        nextQuantity * nextPrice * (1 + this.state.config.feeBps / 10_000),
      );
      const nextRemainingReserve =
        nextQuantity > MIN_BASE_QUANTITY
          ? roundQuote(nextEstimatedQuoteCost * (nextRemainingQuantity / nextQuantity))
          : 0;
      const reserveDelta = roundQuote(nextRemainingReserve - currentRemainingReserve);
      this.state.quoteReserved = roundQuote(this.state.quoteReserved + reserveDelta);
      this.state.quoteFree = roundQuote(this.state.quoteFree - reserveDelta);
      order.estimatedQuoteCost = nextEstimatedQuoteCost;
    } else if (order.positionEffect !== "open") {
      const reserveDelta = roundAsset(nextRemainingQuantity - currentRemainingQuantity);
      this.state.baseReserved = roundAsset(this.state.baseReserved + reserveDelta);
      this.state.baseFree = roundAsset(this.state.baseFree - reserveDelta);
    }

    order.price = nextPrice;
    order.quantity = nextQuantity;
  }

  private consumeQuoteReserveForExchangeFill(
    order: TradingOrder,
    quantity: number,
    spentQuote: number,
  ): void {
    const reservedPortion =
      order.status === "open"
        ? Math.min(this.state.quoteReserved, remainingOrderReserveQuote(order, quantity))
        : 0;
    if (reservedPortion > 0) {
      this.state.quoteReserved = roundQuote(
        Math.max(0, this.state.quoteReserved - reservedPortion),
      );
      this.state.quoteFree = roundQuote(
        this.state.quoteFree + reservedPortion - spentQuote,
      );
    } else {
      this.state.quoteFree = roundQuote(this.state.quoteFree - spentQuote);
    }
  }

  private consumeBaseReserveForExchangeFill(order: TradingOrder, quantity: number): void {
    if (order.status === "open" && order.positionEffect !== "open") {
      const reservedQuantity = Math.min(this.state.baseReserved, quantity);
      this.state.baseReserved = roundAsset(
        Math.max(0, this.state.baseReserved - reservedQuantity),
      );
      if (quantity > reservedQuantity + MIN_BASE_QUANTITY) {
        this.state.baseFree = roundAsset(
          this.state.baseFree - (quantity - reservedQuantity),
        );
      }
      return;
    }

    this.state.baseFree = roundAsset(this.state.baseFree - quantity);
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

    if (options.processOpenOrders ?? true) {
      if (events) {
        events.push(...this.cancelStaleOrders(eventTime, collectEvents));
        events.push(...this.fillOpenOrders(tick, collectEvents));
      } else {
        this.cancelStaleOrders(eventTime, collectEvents);
        this.fillOpenOrders(tick, collectEvents);
      }
    }
    const liquidated = options.simulateLiquidation
      ? this.liquidateAccountIfNeeded(eventTime, price, collectEvents, events)
      : false;
    this.rememberPrice(price);

    if (!liquidated && this.state.status === "running") {
      if (events) {
        events.push(...this.evaluateStrategy(tick, collectEvents, options));
      } else {
        this.evaluateStrategy(tick, collectEvents, options);
      }
    }

    if (options.updateMetrics ?? true) {
      recalculateMetrics(this.state);
    }
    return events ?? [];
  }

  private evaluateStrategy(
    tick: PriceTick,
    collectEvents: boolean,
    options: TickProcessingOptions,
  ): BotEvent[] {
    return this.evaluateLegacyValleyPeakStrategy(tick, collectEvents, options);
  }

  private evaluateLegacyValleyPeakStrategy(
    tick: PriceTick,
    collectEvents: boolean,
    options: TickProcessingOptions,
  ): BotEvent[] {
    const config = this.state.config;

    if (config.legacyValleyPeak.exitGridEnabled) {
      return this.evaluateLegacyValleyPeakExitGridStrategy(tick, collectEvents, options);
    }

    if (this.openOrderIndexes.size >= config.maxOpenOrders) {
      return [];
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return [];
    }

    const memory = this.ensureLegacyValleyPeakMemory();
    const targetEntryLeverage = this.targetLongEntryLeverage(tick.price);
    const targetShortEntryLeverage = this.targetShortEntryLeverage(tick.price);
    const decision = evaluateLegacyValleyPeak(
      memory,
      config.legacyValleyPeak,
      {
        eventTime: tick.eventTime,
        price: tick.price,
        feeRate: config.feeBps / 10_000,
        buyingPowerQuote: this.longEntryBuyingPowerQuote(tick.price, targetEntryLeverage),
        shortSellingPowerQuote: this.shortEntrySellingPowerQuote(
          tick.price,
          targetShortEntryLeverage,
        ),
        baseFree: this.activeLongQuantity(),
        shortBaseFree: this.activeShortQuantity(),
      },
    );

    if (decision.signal === "hold") {
      this.state.memory.lastSignal = "hold";
      return [];
    }

    const order =
      decision.signal === "buy"
        ? decision.coverQuantity * tick.price >= config.minOrderQuote
          ? this.createBuyToCoverOrder(
              tick.price,
              tick.eventTime,
              decision.reason,
              decision.coverQuantity,
            )
          : this.createBuyOrder(tick.price, tick.eventTime, decision.reason, decision.quoteSize)
        : decision.quantity * tick.price >= config.minOrderQuote
          ? this.createSellOrder(tick.price, tick.eventTime, decision.reason, decision.quantity)
          : this.createShortSellOrder(
              tick.price,
              tick.eventTime,
              decision.reason,
              decision.quoteSize,
            );

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
    options: TickProcessingOptions,
  ): BotEvent[] {
    const config = this.state.config;
    const legacyConfig = config.legacyValleyPeak;
    const memory = this.ensureLegacyValleyPeakMemory();
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;
    const totalBase = this.totalBase();
    const activeLongQuantity = this.activeLongQuantity();
    const activeShortQuantity = this.activeShortQuantity();

    if (
      legacyConfig.exitGridPositionMode === "aggregate" &&
      totalBase <= MIN_BASE_QUANTITY
    ) {
      delete memory.exitGrids?.["aggregate-long"];
    }
    if (
      legacyConfig.exitGridPositionMode === "aggregate" &&
      totalBase >= -MIN_BASE_QUANTITY
    ) {
      delete memory.exitGrids?.["aggregate-short"];
    }
    this.updateLegacyExitGridExtremes(memory, tick.price);
    const targetLongEntryLeverage = this.targetLongEntryLeverage(tick.price);
    const targetShortEntryLeverage = this.targetShortEntryLeverage(tick.price);

    const decision = evaluateLegacyValleyPeak(
      memory,
      legacyConfig,
      {
        eventTime: tick.eventTime,
        price: tick.price,
        feeRate: config.feeBps / 10_000,
        buyingPowerQuote: this.longEntryBuyingPowerQuote(
          tick.price,
          targetLongEntryLeverage,
        ),
        shortSellingPowerQuote: this.shortEntrySellingPowerQuote(
          tick.price,
          targetShortEntryLeverage,
        ),
        baseFree: activeLongQuantity,
        shortBaseFree: activeShortQuantity,
      },
    );

    if (decision.signal !== "buy" && decision.signal !== "sell") {
      this.state.memory.lastSignal = decision.signal;
      return events ?? NO_EVENTS;
    }

    if (tick.eventTime - this.state.memory.lastActionAt < config.cooldownMs) {
      return events ?? NO_EVENTS;
    }

    const orders: TradingOrder[] = [];
    let acted = false;

    if (decision.signal === "buy") {
      if (legacyConfig.shortSideEnabled && activeShortQuantity > MIN_BASE_QUANTITY) {
        const activeShorts = this.activeLegacyShortLots(tick.price);
        this.syncLegacyExitGridMemories(memory, activeShorts, tick.price, "short");

        for (const lot of activeShorts) {
          const grid = this.ensureLegacyExitGrid(memory, lot, tick.price);
          grid.troughPrice = Math.min(grid.troughPrice ?? tick.price, tick.price);

          const resetTroughPrice = this.legacyExitGridResetTroughPrice(
            grid,
            lot,
            tick.price,
          );
          if (resetTroughPrice === undefined) {
            continue;
          }
          grid.troughPrice = resetTroughPrice;

          if (events) {
            events.push(
              ...this.cancelLegacyExitGridOrders(tick.eventTime, true, lot.id, "short"),
            );
          } else {
            this.cancelLegacyExitGridOrders(tick.eventTime, false, lot.id, "short");
          }

          orders.push(
            ...this.createLegacyShortExitGridOrders(grid, lot, tick.eventTime, tick.price),
          );
          if (this.openOrderIndexes.size >= config.maxOpenOrders) {
            break;
          }
        }
      }

      if (
        legacyConfig.exitGridPositionMode === "aggregate" &&
        activeLongQuantity > MIN_BASE_QUANTITY
      ) {
        acted = orders.length > 0;
      } else if (
        legacyConfig.longSideEnabled &&
        this.openOrderIndexes.size < config.maxOpenOrders &&
        decision.quoteSize >= config.minOrderQuote &&
        legacyConfig.exitGridMarketEntry
      ) {
        const result = this.createMarketBuyOrder(
          tick.price,
          tick.eventTime,
          decision.reason,
          decision.quoteSize,
          { fillImmediately: !options.deferMarketOrderFills },
        );
        if (result.order) {
          if (options.deferMarketOrderFills) {
            orders.push(result.order);
          } else {
            acted = true;
            this.syncLegacyExitGridMemories(
              memory,
              this.activeLegacyLongLots(tick.price),
              tick.price,
              "long",
            );
          }
        }
        if (events && result.order && !options.deferMarketOrderFills) {
          events.push(...this.immediateOrderEvents("BUY", result, tick.eventTime));
        }
      } else if (
        legacyConfig.longSideEnabled &&
        this.openOrderIndexes.size < config.maxOpenOrders &&
        decision.quoteSize >= config.minOrderQuote
      ) {
        const order = this.createBuyOrder(
          tick.price,
          tick.eventTime,
          decision.reason,
          decision.quoteSize,
        );
        if (order) {
          order.positionEffect = "open";
          orders.push(order);
        }
      }
    } else {
      if (legacyConfig.longSideEnabled && activeLongQuantity > MIN_BASE_QUANTITY) {
        const activeLongs = this.activeLegacyLongLots(tick.price);
        this.syncLegacyExitGridMemories(memory, activeLongs, tick.price, "long");

        for (const lot of activeLongs) {
          const grid = this.ensureLegacyExitGrid(memory, lot, tick.price);
          grid.peakPrice = Math.max(grid.peakPrice, tick.price);

          const resetPeakPrice = this.legacyExitGridResetPeakPrice(grid, lot, tick.price);
          if (resetPeakPrice === undefined) {
            continue;
          }
          grid.peakPrice = resetPeakPrice;

          if (events) {
            events.push(
              ...this.cancelLegacyExitGridOrders(tick.eventTime, true, lot.id, "long"),
            );
          } else {
            this.cancelLegacyExitGridOrders(tick.eventTime, false, lot.id, "long");
          }

          orders.push(
            ...this.createLegacyLongExitGridOrders(grid, lot, tick.eventTime, tick.price),
          );
          if (this.openOrderIndexes.size >= config.maxOpenOrders) {
            break;
          }
        }
      }

      if (
        legacyConfig.shortSideEnabled &&
        this.openOrderIndexes.size < config.maxOpenOrders &&
        decision.quoteSize >= config.minOrderQuote &&
        !(
          legacyConfig.exitGridPositionMode === "aggregate" &&
          activeShortQuantity > MIN_BASE_QUANTITY
        )
      ) {
        if (legacyConfig.exitGridMarketEntry) {
          const result = this.createMarketSellOrder(
            tick.price,
            tick.eventTime,
            decision.reason,
            decision.quoteSize,
            { fillImmediately: !options.deferMarketOrderFills },
          );
          if (result.order) {
            if (options.deferMarketOrderFills) {
              orders.push(result.order);
            } else {
              acted = true;
              this.syncLegacyExitGridMemories(
                memory,
                this.activeLegacyShortLots(tick.price),
                tick.price,
                "short",
              );
            }
          }
          if (events && result.order && !options.deferMarketOrderFills) {
            events.push(...this.immediateOrderEvents("SELL", result, tick.eventTime));
          }
        } else {
          const order = this.createShortSellOrder(
            tick.price,
            tick.eventTime,
            decision.reason,
            decision.quoteSize,
          );
          if (order) {
            orders.push(order);
          }
        }
      }
    }

    if (orders.length === 0 && !acted) {
      this.state.memory.lastSignal = decision.signal;
      return events ?? NO_EVENTS;
    }

    this.state.memory.lastSignal = decision.signal;
    this.state.memory.lastActionAt = tick.eventTime;

    if (events) {
      for (const order of orders) {
        events.push({
          type: "order_created",
          at: tick.eventTime,
          message:
            order.reason.startsWith(LEGACY_EXIT_GRID_REASON)
              ? `${order.side.toUpperCase()} exit grid order created at ${order.price}`
              : `${order.side.toUpperCase()} ${order.type} order created: ${order.reason}`,
          order: structuredClone(order),
        });
      }
      if (!options.deferMarketOrderFills) {
        events.push(...this.fillOpenOrders(tick, true));
      }
    } else {
      if (!options.deferMarketOrderFills) {
        this.fillOpenOrders(tick, false);
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
    if (!config.legacyValleyPeak.longSideEnabled) {
      return undefined;
    }
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
    order.positionEffect = "open";
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
    if (!config.legacyValleyPeak.longSideEnabled) {
      return undefined;
    }
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
    order.positionEffect = "close";
    this.state.orders.push(order);
    this.openOrderIndexes.add(this.state.orders.length - 1);
    return order;
  }

  private createShortSellOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuoteSize: number,
  ): TradingOrder | undefined {
    const config = this.state.config;
    const availableQuote = this.shortEntrySellingPowerQuote(marketPrice);
    const quoteSize = Math.min(desiredQuoteSize, availableQuote);

    if (!config.legacyValleyPeak.shortSideEnabled || quoteSize < config.minOrderQuote) {
      return undefined;
    }

    const price = roundQuote(marketPrice * (1 + config.limitOffsetBps / 10_000));
    const quantity = roundAsset(quoteSize / price);

    if (quantity <= 0) {
      return undefined;
    }

    const order = this.buildOrder("sell", price, quantity, 0, createdAt, reason);
    order.positionEffect = "open";
    this.state.orders.push(order);
    this.openOrderIndexes.add(this.state.orders.length - 1);
    return order;
  }

  private createMarketBuyOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuoteSize: number,
    options: { fillImmediately?: boolean } = {},
  ): ImmediateOrderResult {
    const config = this.state.config;
    if (!config.legacyValleyPeak.longSideEnabled) {
      return {};
    }
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

    if (options.fillImmediately === false) {
      return {
        order: this.state.orders[index],
      };
    }

    const result = this.tryFillOrder(order, index, createdAt);
    return {
      order: this.state.orders[index],
      fill: result.fill,
      cancelled: result.cancelled,
    };
  }

  private createMarketSellOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuoteSize: number,
    options: { fillImmediately?: boolean } = {},
  ): ImmediateOrderResult {
    const config = this.state.config;
    const availableQuote = this.shortEntrySellingPowerQuote(marketPrice);
    const quoteSize = Math.min(desiredQuoteSize, availableQuote);

    if (!config.legacyValleyPeak.shortSideEnabled || quoteSize < config.minOrderQuote) {
      return {};
    }

    const price = roundQuote(marketPrice);
    const quantity = roundAsset(quoteSize / price);

    if (quantity <= 0) {
      return {};
    }

    const order = this.buildOrder("sell", price, quantity, 0, createdAt, reason, "market");
    const index = this.state.orders.length;
    order.positionEffect = "open";
    this.state.orders.push(order);
    this.openOrderIndexes.add(index);

    if (options.fillImmediately === false) {
      return {
        order: this.state.orders[index],
      };
    }

    const result = this.tryFillOrder(order, index, createdAt);
    return {
      order: this.state.orders[index],
      fill: result.fill,
      cancelled: result.cancelled,
    };
  }

  private createBuyToCoverOrder(
    marketPrice: number,
    createdAt: number,
    reason: string,
    desiredQuantity: number,
  ): TradingOrder | undefined {
    const price = roundQuote(marketPrice * (1 - this.state.config.limitOffsetBps / 10_000));
    return this.createTriggeredBuyOrder(price, desiredQuantity, createdAt, reason);
  }

  private createTriggeredBuyOrder(
    price: number,
    quantity: number,
    createdAt: number,
    reason: string,
    targetPositionId?: string,
  ): TradingOrder | undefined {
    const config = this.state.config;
    const roundedQuantity = roundAsset(quantity);
    const quoteSize = roundedQuantity * price;

    if (roundedQuantity <= 0 || quoteSize < config.minOrderQuote) {
      return undefined;
    }

    const feeRate = config.feeBps / 10_000;
    const estimatedQuoteCost = roundQuote(roundedQuantity * price * (1 + feeRate));
    this.state.quoteFree = roundQuote(this.state.quoteFree - estimatedQuoteCost);
    this.state.quoteReserved = roundQuote(this.state.quoteReserved + estimatedQuoteCost);

    const order = this.buildOrder(
      "buy",
      roundQuote(price),
      roundedQuantity,
      estimatedQuoteCost,
      createdAt,
      reason,
      "limit",
      "above",
    );
    order.positionEffect = "close";
    if (targetPositionId) {
      order.targetPositionId = targetPositionId;
    }
    this.state.orders.push(order);
    this.openOrderIndexes.add(this.state.orders.length - 1);
    return order;
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

  private createExternalCloseOrder(
    side: "buy" | "sell",
    price: number,
    quantity: number,
    createdAt: number,
    reason: string,
    targetPositionId: string,
  ): TradingOrder | undefined {
    const roundedQuantity = roundAsset(quantity);
    const roundedPrice = roundQuote(price);
    if (roundedQuantity <= MIN_BASE_QUANTITY || roundedPrice <= 0) {
      return undefined;
    }

    const feeRate = this.state.config.feeBps / 10_000;
    const estimatedQuoteCost =
      side === "buy" ? roundQuote(roundedQuantity * roundedPrice * (1 + feeRate)) : 0;

    if (side === "buy") {
      this.state.quoteFree = roundQuote(this.state.quoteFree - estimatedQuoteCost);
      this.state.quoteReserved = roundQuote(this.state.quoteReserved + estimatedQuoteCost);
    } else {
      this.state.baseFree = roundAsset(this.state.baseFree - roundedQuantity);
      this.state.baseReserved = roundAsset(this.state.baseReserved + roundedQuantity);
    }

    const order = this.buildOrder(
      side,
      roundedPrice,
      roundedQuantity,
      estimatedQuoteCost,
      createdAt,
      reason,
      "market",
    );
    order.positionEffect = "close";
    order.targetPositionId = targetPositionId;
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

  private activeLegacyLongLots(price: number): LegacyLongExitGridLot[] {
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
          side: "long",
          averagePrice,
          originalQuantity: totalBase,
          remainingQuantity: totalBase,
          breakEvenSellPrice: averagePrice / Math.max(0.000001, 1 - feeRate),
        },
      ];
    }

    this.updateLegacyPositionLotCache();
    const lots = this.legacyPositionLotCache?.longs.values() ?? [];
    return [...lots].filter((lot) => lot.remainingQuantity > MIN_BASE_QUANTITY);
  }

  private activeLegacyShortLots(price: number): LegacyShortExitGridLot[] {
    if (this.state.config.legacyValleyPeak.exitGridPositionMode === "aggregate") {
      const totalShortBase = Math.max(0, -this.totalBase());
      const averagePrice = cleanPositive(this.state.avgShortEntryPrice) || price;
      if (totalShortBase <= MIN_BASE_QUANTITY || averagePrice <= 0) {
        return [];
      }

      const feeAndSlippageRate =
        (this.state.config.feeBps + this.state.config.positionRisk.marketSlippageBps) /
        10_000;
      return [
        {
          id: "aggregate-short",
          side: "short",
          averagePrice,
          originalQuantity: totalShortBase,
          remainingQuantity: totalShortBase,
          breakEvenBuyPrice: roundQuote(
            averagePrice / ((1 + feeAndSlippageRate) ** 2),
          ),
        },
      ];
    }

    this.updateLegacyPositionLotCache();
    const lots = this.legacyPositionLotCache?.shorts.values() ?? [];
    return [...lots].filter((lot) => lot.remainingQuantity > MIN_BASE_QUANTITY);
  }

  private updateLegacyPositionLotCache(): void {
    const feeAndSlippageRate =
      (this.state.config.feeBps + this.state.config.positionRisk.marketSlippageBps) / 10_000;
    const quantityFloor = Math.max(MIN_BASE_QUANTITY, this.state.config.positionRisk.quantityFloor);
    const feeFactor = (1 + feeAndSlippageRate) ** 2;

    if (
      !this.legacyPositionLotCache ||
      this.legacyPositionLotCache.processedFillsLength > this.state.fills.length
    ) {
      this.legacyPositionLotCache = {
        processedFillsLength: 0,
        longs: new Map(),
        shorts: new Map(),
      };
    }

    const cache = this.legacyPositionLotCache;
    for (
      let index = cache.processedFillsLength;
      index < this.state.fills.length;
      index += 1
    ) {
      const fill = this.state.fills[index];
      if (fill.side === "buy") {
        this.applyLegacyBuyFillToPositionCache(
          cache,
          fill,
          quantityFloor,
          feeAndSlippageRate,
          feeFactor,
        );
      } else {
        this.applyLegacySellFillToPositionCache(
          cache,
          fill,
          quantityFloor,
          feeAndSlippageRate,
          feeFactor,
        );
      }
    }

    cache.processedFillsLength = this.state.fills.length;
  }

  private applyLegacyBuyFillToPositionCache(
    cache: LegacyPositionLotCache,
    fill: TradeFill,
    quantityFloor: number,
    feeAndSlippageRate: number,
    feeFactor: number,
  ): void {
    const unitCost = (fill.quoteQuantity + fill.feeQuote) / fill.quantity;
    let quantityLeft = fill.quantity;
    let costLeft = roundQuote(fill.quoteQuantity + fill.feeQuote);

    if (fill.targetPositionId) {
      const target = cache.shorts.get(fill.targetPositionId);
      if (target) {
        const closed = this.closeLegacyTrackedShortLot(
          cache,
          target,
          quantityLeft,
          unitCost,
          quantityFloor,
          feeAndSlippageRate,
          feeFactor,
        );
        quantityLeft = roundAsset(quantityLeft - closed.quantity);
        costLeft = roundQuote(costLeft - closed.quote);
      }
    }

    if (fill.positionEffect !== "open") {
      for (const short of [...cache.shorts.values()]) {
        if (quantityLeft <= MIN_BASE_QUANTITY) {
          break;
        }
        if (short.remainingQuantity <= MIN_BASE_QUANTITY) {
          continue;
        }
        const closed = this.closeLegacyTrackedShortLot(
          cache,
          short,
          quantityLeft,
          unitCost,
          quantityFloor,
          feeAndSlippageRate,
          feeFactor,
        );
        quantityLeft = roundAsset(quantityLeft - closed.quantity);
        costLeft = roundQuote(costLeft - closed.quote);
      }
    }

    if (quantityLeft <= MIN_BASE_QUANTITY || costLeft <= 0) {
      return;
    }

    const averagePrice = roundQuote(costLeft / quantityLeft);
    const borrowAllocations = this.allocateLegacyLongBorrowFromShortLots(
      cache,
      quantityLeft,
      averagePrice,
      quantityFloor,
      feeFactor,
    );
    cache.longs.set(`long_${fill.id}`, {
      id: `long_${fill.id}`,
      side: "long",
      openedAt: fill.filledAt,
      averagePrice,
      originalQuantity: roundAsset(quantityLeft),
      remainingQuantity: roundAsset(quantityLeft),
      remainingCostQuote: roundQuote(costLeft),
      lentQuantity: 0,
      borrowDepthRemaining: this.inheritedLegacyLongBorrowDepth(borrowAllocations),
      borrowAllocations,
      breakEvenSellPrice: roundQuote(
        (costLeft / Math.max(quantityLeft, quantityFloor)) * (1 + feeAndSlippageRate),
      ),
    });
  }

  private applyLegacySellFillToPositionCache(
    cache: LegacyPositionLotCache,
    fill: TradeFill,
    quantityFloor: number,
    feeAndSlippageRate: number,
    feeFactor: number,
  ): void {
    const unitProceeds = (fill.quoteQuantity - fill.feeQuote) / fill.quantity;
    let quantityLeft = fill.quantity;
    let proceedsLeft = roundQuote(fill.quoteQuantity - fill.feeQuote);

    if (fill.targetPositionId) {
      const target = cache.longs.get(fill.targetPositionId);
      if (target) {
        const closed = this.closeLegacyTrackedLongLot(
          cache,
          target,
          quantityLeft,
          unitProceeds,
          quantityFloor,
          feeAndSlippageRate,
          feeFactor,
        );
        quantityLeft = roundAsset(quantityLeft - closed.quantity);
        proceedsLeft = roundQuote(proceedsLeft - closed.quote);
      }
    }

    if (fill.positionEffect !== "open") {
      for (const long of [...cache.longs.values()]) {
        if (quantityLeft <= MIN_BASE_QUANTITY) {
          break;
        }
        if (long.remainingQuantity <= MIN_BASE_QUANTITY) {
          continue;
        }
        const closed = this.closeLegacyTrackedLongLot(
          cache,
          long,
          quantityLeft,
          unitProceeds,
          quantityFloor,
          feeAndSlippageRate,
          feeFactor,
        );
        quantityLeft = roundAsset(quantityLeft - closed.quantity);
        proceedsLeft = roundQuote(proceedsLeft - closed.quote);
      }
    }

    if (quantityLeft <= MIN_BASE_QUANTITY || proceedsLeft <= 0) {
      return;
    }

    const averagePrice = roundQuote(proceedsLeft / quantityLeft);
    const borrowAllocations = this.allocateLegacyShortBorrowFromLongLots(
      cache,
      quantityLeft,
      averagePrice,
      quantityFloor,
      feeAndSlippageRate,
    );
    cache.shorts.set(`short_${fill.id}`, {
      id: `short_${fill.id}`,
      side: "short",
      openedAt: fill.filledAt,
      averagePrice,
      originalQuantity: roundAsset(quantityLeft),
      remainingQuantity: roundAsset(quantityLeft),
      remainingProceedsQuote: roundQuote(proceedsLeft),
      lentQuote: 0,
      borrowDepthRemaining: this.inheritedLegacyShortBorrowDepth(borrowAllocations),
      borrowAllocations,
      breakEvenBuyPrice: roundQuote(
        proceedsLeft / Math.max(quantityLeft, quantityFloor) / feeFactor,
      ),
    });
  }

  private closeLegacyTrackedLongLot(
    cache: LegacyPositionLotCache,
    lot: LegacyTrackedLongLot,
    requestedQuantity: number,
    unitProceeds: number,
    quantityFloor: number,
    feeAndSlippageRate: number,
    feeFactor: number,
  ): { quantity: number; quote: number } {
    const quantity = Math.min(requestedQuantity, lot.remainingQuantity);
    if (quantity <= MIN_BASE_QUANTITY) {
      return { quantity: 0, quote: 0 };
    }

    const quote = quantity * unitProceeds;
    this.settleLegacyLongBorrowAllocations(
      cache,
      lot,
      quantity,
      unitProceeds,
      quantityFloor,
      feeFactor,
    );
    lot.remainingQuantity = roundAsset(lot.remainingQuantity - quantity);
    lot.remainingCostQuote = roundQuote(lot.remainingCostQuote - quote);
    lot.lentQuantity = roundAsset(Math.min(lot.lentQuantity, lot.remainingQuantity));
    lot.breakEvenSellPrice = this.legacyLongLotBreakEvenSellPrice(
      lot,
      quantityFloor,
      feeAndSlippageRate,
    );
    this.deleteLegacyLongLotIfInactive(cache, lot);

    return {
      quantity: roundAsset(quantity),
      quote: roundQuote(quote),
    };
  }

  private closeLegacyTrackedShortLot(
    cache: LegacyPositionLotCache,
    lot: LegacyTrackedShortLot,
    requestedQuantity: number,
    unitCost: number,
    quantityFloor: number,
    feeAndSlippageRate: number,
    feeFactor: number,
  ): { quantity: number; quote: number } {
    const quantity = Math.min(requestedQuantity, lot.remainingQuantity);
    if (quantity <= MIN_BASE_QUANTITY) {
      return { quantity: 0, quote: 0 };
    }

    const quote = quantity * unitCost;
    this.settleLegacyShortBorrowAllocations(
      cache,
      lot,
      quantity,
      unitCost,
      quantityFloor,
      feeAndSlippageRate,
    );
    lot.remainingQuantity = roundAsset(lot.remainingQuantity - quantity);
    lot.remainingProceedsQuote = roundQuote(lot.remainingProceedsQuote - quote);
    lot.breakEvenBuyPrice = this.legacyShortLotBreakEvenBuyPrice(
      lot,
      quantityFloor,
      feeFactor,
    );
    this.deleteLegacyShortLotIfInactive(cache, lot);

    return {
      quantity: roundAsset(quantity),
      quote: roundQuote(quote),
    };
  }

  private allocateLegacyShortBorrowFromLongLots(
    cache: LegacyPositionLotCache,
    quantity: number,
    unitProceeds: number,
    quantityFloor: number,
    feeAndSlippageRate: number,
  ): LegacyShortBorrowAllocation[] {
    let quantityLeft = roundAsset(quantity);
    const allocations: LegacyShortBorrowAllocation[] = [];
    const sources = [...cache.longs.values()]
      .filter(
        (lot) =>
          lot.borrowDepthRemaining > 0 &&
          lot.remainingQuantity - lot.lentQuantity > MIN_BASE_QUANTITY,
      )
      .sort((left, right) => {
        const leftBreakEven = this.legacyLongLotBreakEvenBeforeFees(left);
        const rightBreakEven = this.legacyLongLotBreakEvenBeforeFees(right);
        const leftIsBad = unitProceeds < leftBreakEven;
        const rightIsBad = unitProceeds < rightBreakEven;
        if (leftIsBad !== rightIsBad) {
          return leftIsBad ? -1 : 1;
        }
        return rightBreakEven - leftBreakEven || left.openedAt - right.openedAt;
      });

    for (const source of sources) {
      if (quantityLeft <= MIN_BASE_QUANTITY) {
        break;
      }

      const available = Math.max(0, source.remainingQuantity - source.lentQuantity);
      const borrowedQuantity = roundAsset(Math.min(quantityLeft, available));
      if (borrowedQuantity <= MIN_BASE_QUANTITY) {
        continue;
      }

      const borrowedQuote = roundQuote(borrowedQuantity * unitProceeds);
      source.lentQuantity = roundAsset(source.lentQuantity + borrowedQuantity);
      source.remainingCostQuote = roundQuote(
        source.remainingCostQuote - borrowedQuote,
      );
      source.breakEvenSellPrice = this.legacyLongLotBreakEvenSellPrice(
        source,
        quantityFloor,
        feeAndSlippageRate,
      );
      quantityLeft = roundAsset(quantityLeft - borrowedQuantity);
      allocations.push({
        longLotId: source.id,
        quantity: borrowedQuantity,
        quote: borrowedQuote,
        depthRemaining: Math.max(0, source.borrowDepthRemaining - 1),
      });
    }

    return allocations;
  }

  private allocateLegacyLongBorrowFromShortLots(
    cache: LegacyPositionLotCache,
    quantity: number,
    unitCost: number,
    quantityFloor: number,
    feeFactor: number,
  ): LegacyLongBorrowAllocation[] {
    let quantityLeft = roundAsset(quantity);
    const allocations: LegacyLongBorrowAllocation[] = [];
    const sources = [...cache.shorts.values()]
      .filter(
        (lot) =>
          lot.borrowDepthRemaining > 0 &&
          lot.remainingQuantity > MIN_BASE_QUANTITY &&
          lot.remainingProceedsQuote > 0,
      )
      .sort((left, right) => {
        const leftBreakEven = this.legacyShortLotBreakEvenBeforeFees(left);
        const rightBreakEven = this.legacyShortLotBreakEvenBeforeFees(right);
        const leftIsBad = unitCost > leftBreakEven;
        const rightIsBad = unitCost > rightBreakEven;
        if (leftIsBad !== rightIsBad) {
          return leftIsBad ? -1 : 1;
        }
        return leftBreakEven - rightBreakEven || left.openedAt - right.openedAt;
      });

    for (const source of sources) {
      if (quantityLeft <= MIN_BASE_QUANTITY) {
        break;
      }

      const affordableQuantity = source.remainingProceedsQuote / unitCost;
      const borrowedQuantity = roundAsset(Math.min(quantityLeft, affordableQuantity));
      if (borrowedQuantity <= MIN_BASE_QUANTITY) {
        continue;
      }

      const borrowedQuote = roundQuote(borrowedQuantity * unitCost);
      source.lentQuote = roundQuote(source.lentQuote + borrowedQuote);
      source.remainingProceedsQuote = roundQuote(
        source.remainingProceedsQuote - borrowedQuote,
      );
      source.breakEvenBuyPrice = this.legacyShortLotBreakEvenBuyPrice(
        source,
        quantityFloor,
        feeFactor,
      );
      quantityLeft = roundAsset(quantityLeft - borrowedQuantity);
      allocations.push({
        shortLotId: source.id,
        quantity: borrowedQuantity,
        quote: borrowedQuote,
        depthRemaining: Math.max(0, source.borrowDepthRemaining - 1),
      });
    }

    return allocations;
  }

  private settleLegacyShortBorrowAllocations(
    cache: LegacyPositionLotCache,
    short: LegacyTrackedShortLot,
    closedQuantity: number,
    unitCost: number,
    quantityFloor: number,
    feeAndSlippageRate: number,
  ): void {
    let quantityLeft = roundAsset(closedQuantity);

    for (const allocation of short.borrowAllocations) {
      if (quantityLeft <= MIN_BASE_QUANTITY) {
        break;
      }
      if (allocation.quantity <= MIN_BASE_QUANTITY || allocation.quote <= 0) {
        continue;
      }

      const quantity = roundAsset(Math.min(quantityLeft, allocation.quantity));
      const allocationRatio = quantity / allocation.quantity;
      const principalQuote = roundQuote(allocation.quote * allocationRatio);
      const coverQuote = roundQuote(quantity * unitCost);
      const long = cache.longs.get(allocation.longLotId);
      if (long) {
        long.lentQuantity = roundAsset(Math.max(0, long.lentQuantity - quantity));
        const profitQuote = principalQuote - coverQuote;
        const returnedQuote =
          profitQuote > 0
            ? roundQuote(
                principalQuote -
                  profitQuote * this.state.config.borrowerProfitShareToLender,
              )
            : coverQuote;
        long.remainingCostQuote = roundQuote(
          long.remainingCostQuote + returnedQuote,
        );
        long.breakEvenSellPrice = this.legacyLongLotBreakEvenSellPrice(
          long,
          quantityFloor,
          feeAndSlippageRate,
        );
        this.deleteLegacyLongLotIfInactive(cache, long);
      }

      allocation.quantity = roundAsset(allocation.quantity - quantity);
      allocation.quote = roundQuote(allocation.quote - principalQuote);
      quantityLeft = roundAsset(quantityLeft - quantity);
    }

    short.borrowAllocations = short.borrowAllocations.filter(
      (allocation) =>
        allocation.quantity > MIN_BASE_QUANTITY && allocation.quote > MIN_BASE_QUANTITY,
    );
  }

  private settleLegacyLongBorrowAllocations(
    cache: LegacyPositionLotCache,
    long: LegacyTrackedLongLot,
    closedQuantity: number,
    unitProceeds: number,
    quantityFloor: number,
    feeFactor: number,
  ): void {
    let quantityLeft = roundAsset(closedQuantity);

    for (const allocation of long.borrowAllocations) {
      if (quantityLeft <= MIN_BASE_QUANTITY) {
        break;
      }
      if (allocation.quantity <= MIN_BASE_QUANTITY || allocation.quote <= 0) {
        continue;
      }

      const quantity = roundAsset(Math.min(quantityLeft, allocation.quantity));
      const allocationRatio = quantity / allocation.quantity;
      const principalQuote = roundQuote(allocation.quote * allocationRatio);
      const returnedQuote = roundQuote(quantity * unitProceeds);
      const short = cache.shorts.get(allocation.shortLotId);
      if (short) {
        short.lentQuote = roundQuote(Math.max(0, short.lentQuote - principalQuote));
        const profitQuote = returnedQuote - principalQuote;
        const lenderQuote =
          profitQuote > 0
            ? roundQuote(
                principalQuote +
                  profitQuote * this.state.config.borrowerProfitShareToLender,
              )
            : returnedQuote;
        short.remainingProceedsQuote = roundQuote(
          short.remainingProceedsQuote + lenderQuote,
        );
        short.breakEvenBuyPrice = this.legacyShortLotBreakEvenBuyPrice(
          short,
          quantityFloor,
          feeFactor,
        );
        this.deleteLegacyShortLotIfInactive(cache, short);
      }

      allocation.quantity = roundAsset(allocation.quantity - quantity);
      allocation.quote = roundQuote(allocation.quote - principalQuote);
      quantityLeft = roundAsset(quantityLeft - quantity);
    }

    long.borrowAllocations = long.borrowAllocations.filter(
      (allocation) =>
        allocation.quantity > MIN_BASE_QUANTITY && allocation.quote > MIN_BASE_QUANTITY,
    );
  }

  private deleteLegacyLongLotIfInactive(
    cache: LegacyPositionLotCache,
    lot: LegacyTrackedLongLot,
  ): void {
    if (
      lot.remainingQuantity <= MIN_BASE_QUANTITY &&
      lot.lentQuantity <= MIN_BASE_QUANTITY &&
      lot.borrowAllocations.length === 0
    ) {
      cache.longs.delete(lot.id);
    }
  }

  private deleteLegacyShortLotIfInactive(
    cache: LegacyPositionLotCache,
    lot: LegacyTrackedShortLot,
  ): void {
    if (
      lot.remainingQuantity <= MIN_BASE_QUANTITY &&
      lot.lentQuote <= MIN_BASE_QUANTITY &&
      lot.borrowAllocations.length === 0
    ) {
      cache.shorts.delete(lot.id);
    }
  }

  private inheritedLegacyLongBorrowDepth(
    allocations: LegacyLongBorrowAllocation[],
  ): number {
    return allocations.length > 0
      ? Math.max(...allocations.map((allocation) => allocation.depthRemaining))
      : this.state.config.longBorrowDepth;
  }

  private inheritedLegacyShortBorrowDepth(
    allocations: LegacyShortBorrowAllocation[],
  ): number {
    return allocations.length > 0
      ? Math.max(...allocations.map((allocation) => allocation.depthRemaining))
      : this.state.config.shortBorrowDepth;
  }

  private legacyLongLotBreakEvenBeforeFees(lot: LegacyTrackedLongLot): number {
    return lot.remainingQuantity > MIN_BASE_QUANTITY
      ? lot.remainingCostQuote / lot.remainingQuantity
      : 0;
  }

  private legacyShortLotBreakEvenBeforeFees(lot: LegacyTrackedShortLot): number {
    return lot.remainingQuantity > MIN_BASE_QUANTITY
      ? lot.remainingProceedsQuote / lot.remainingQuantity
      : 0;
  }

  private legacyLongLotBreakEvenSellPrice(
    lot: LegacyTrackedLongLot,
    quantityFloor: number,
    feeAndSlippageRate: number,
  ): number {
    return lot.remainingQuantity > MIN_BASE_QUANTITY
      ? roundQuote(
          (lot.remainingCostQuote / Math.max(lot.remainingQuantity, quantityFloor)) *
            (1 + feeAndSlippageRate),
        )
      : 0;
  }

  private legacyShortLotBreakEvenBuyPrice(
    lot: LegacyTrackedShortLot,
    quantityFloor: number,
    feeFactor: number,
  ): number {
    return lot.remainingQuantity > MIN_BASE_QUANTITY
      ? roundQuote(
          lot.remainingProceedsQuote /
            Math.max(lot.remainingQuantity, quantityFloor) /
            feeFactor,
        )
      : 0;
  }

  private updateLegacyExitGridExtremes(
    memory: LegacyValleyPeakMemory,
    price: number,
  ): void {
    for (const grid of Object.values(memory.exitGrids ?? {})) {
      if (this.legacyExitGridSide(grid) === "short") {
        grid.troughPrice = Math.min(grid.troughPrice ?? grid.entryPrice, price);
      } else {
        grid.peakPrice = Math.max(grid.peakPrice, price);
      }
    }
  }

  private legacyExitGridSide(grid: LegacyExitGridMemory): PositionLotSide {
    if (
      grid.side === "short" ||
      grid.lotId.startsWith("short_") ||
      grid.lotId === "aggregate-short"
    ) {
      return "short";
    }
    return "long";
  }

  private syncLegacyExitGridMemories(
    memory: LegacyValleyPeakMemory,
    lots: LegacyExitGridLot[],
    price: number,
    side: PositionLotSide,
  ): void {
    const grids = (memory.exitGrids ??= {});
    const activeLotIds = new Set(lots.map((lot) => lot.id));
    const openGridIds = new Map<string, string[]>();
    for (const { order } of this.openLegacyExitGridOrders(undefined, side)) {
      if (!order.targetPositionId) {
        continue;
      }
      const ids = openGridIds.get(order.targetPositionId) ?? [];
      ids.push(order.id);
      openGridIds.set(order.targetPositionId, ids);
    }

    for (const lotId of Object.keys(grids)) {
      if (this.legacyExitGridSide(grids[lotId]) !== side) {
        continue;
      }
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
      if (side === "short") {
        grid.troughPrice = Math.min(grid.troughPrice ?? price, price);
      } else {
        grid.peakPrice = Math.max(grid.peakPrice, price);
      }
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
        side: lot.side,
        entryPrice,
        entryQuantity: lot.originalQuantity,
        peakPrice: Math.max(entryPrice, price),
        gridPeakPrice: 0,
        troughPrice: Math.min(entryPrice, price),
        gridTroughPrice: 0,
        gridCreatedAt: 0,
        gridOrderIds: [],
      };
      grids[lot.id] = grid;
    } else {
      grid.side = lot.side;
      grid.troughPrice ??= Math.min(grid.entryPrice, price);
      grid.gridTroughPrice ??= 0;
      grid.gridCreatedAt ??= 0;
    }
    return grid;
  }

  private legacyExitGridResetPeakPrice(
    grid: LegacyExitGridMemory,
    lot: LegacyLongExitGridLot,
    currentPrice: number,
  ): number | undefined {
    const config = this.state.config.legacyValleyPeak;
    const breakEvenSellPrice = cleanPositive(lot.breakEvenSellPrice) || grid.entryPrice;
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

    if (config.exitGridResetMode === "filled-grid" && peakPrice > grid.gridPeakPrice) {
      return peakPrice > minimumPeakPrice ? peakPrice : undefined;
    }

    if (
      config.exitGridResetMode === "filled-grid" &&
      this.hasCrossedFilledLegacyExitGridPoint(lot.id, currentPrice, "long", grid)
    ) {
      return currentPrice > minimumPeakPrice ? currentPrice : undefined;
    }

    return undefined;
  }

  private legacyExitGridResetTroughPrice(
    grid: LegacyExitGridMemory,
    lot: LegacyShortExitGridLot,
    currentPrice: number,
  ): number | undefined {
    const config = this.state.config.legacyValleyPeak;
    const breakEvenBuyPrice = cleanPositive(lot.breakEvenBuyPrice) || grid.entryPrice;
    const maximumTroughPrice = breakEvenBuyPrice * (1 - config.exitGridMinProfitBps / 10_000);
    const troughPrice = Math.min(grid.troughPrice ?? currentPrice, currentPrice);

    if (grid.gridOrderIds.length === 0 || !grid.gridTroughPrice || grid.gridTroughPrice <= 0) {
      return troughPrice < maximumTroughPrice ? troughPrice : undefined;
    }

    if (
      config.exitGridResetMode === "higher-peak" &&
      troughPrice <= grid.gridTroughPrice * (1 - config.exitGridResetBps / 10_000)
    ) {
      return troughPrice < maximumTroughPrice ? troughPrice : undefined;
    }

    if (config.exitGridResetMode === "filled-grid" && troughPrice < grid.gridTroughPrice) {
      return troughPrice < maximumTroughPrice ? troughPrice : undefined;
    }

    if (
      config.exitGridResetMode === "filled-grid" &&
      this.hasCrossedFilledLegacyExitGridPoint(lot.id, currentPrice, "short", grid)
    ) {
      return currentPrice < maximumTroughPrice ? currentPrice : undefined;
    }

    return undefined;
  }

  private hasCrossedFilledLegacyExitGridPoint(
    lotId: string,
    price: number,
    side: PositionLotSide,
    grid: LegacyExitGridMemory,
  ): boolean {
    const gridCreatedAt = grid.gridCreatedAt ?? 0;
    return this.state.orders.some(
      (order) =>
        order.status === "filled" &&
        order.targetPositionId === lotId &&
        order.createdAt >= gridCreatedAt &&
        order.reason.startsWith(LEGACY_EXIT_GRID_REASON) &&
        (side === "long" ? price > order.price : price < order.price),
    );
  }

  private createLegacyLongExitGridOrders(
    grid: LegacyExitGridMemory,
    lot: LegacyLongExitGridLot,
    createdAt: number,
    currentPrice: number,
  ): TradingOrder[] {
    const config = this.state.config;
    const lowerPrice = roundQuote(cleanPositive(lot.breakEvenSellPrice) || grid.entryPrice);
    const upperPrice = roundQuote(Math.max(grid.peakPrice, lowerPrice));
    const availableSlots = Math.max(0, config.maxOpenOrders - this.openOrderIndexes.size);
    const lockedQuantity = config.lockBorrowedLenderCollateral
      ? (lot.lentQuantity ?? 0)
      : 0;
    const availableQuantity = Math.min(
      Math.max(0, lot.remainingQuantity - lockedQuantity),
      this.state.baseFree,
    );
    const orderCount = this.legacyExitGridOrderCount({
      lowerPrice,
      upperPrice,
      currentPrice,
      availableQuantity,
      availableSlots,
    });

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
      let quantity = roundAsset(Math.min(remainingQuantity, desiredQuantity));

      if (quantity <= MIN_BASE_QUANTITY) {
        break;
      }
      const remainingAfterFill = roundAsset(remainingQuantity - quantity);
      if (
        remainingAfterFill > MIN_BASE_QUANTITY &&
        remainingAfterFill * price < config.minOrderQuote
      ) {
        quantity = remainingQuantity;
      }
      if (quantity * price < config.minOrderQuote) {
        if (remainingQuantity * price >= config.minOrderQuote) {
          quantity = remainingQuantity;
        } else if (index === orderCount - 1 && orders.length === 0) {
          break;
        } else {
          continue;
        }
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
    grid.gridCreatedAt = createdAt;
    grid.gridOrderIds = orders.map((order) => order.id);
    this.recordLegacyExitGridSpan(upperPrice - lowerPrice, orders.length);
    return orders;
  }

  private createLegacyShortExitGridOrders(
    grid: LegacyExitGridMemory,
    lot: LegacyShortExitGridLot,
    createdAt: number,
    currentPrice: number,
  ): TradingOrder[] {
    const config = this.state.config;
    const upperPrice = roundQuote(cleanPositive(lot.breakEvenBuyPrice) || grid.entryPrice);
    const lowerPrice = roundQuote(Math.min(grid.troughPrice ?? upperPrice, upperPrice));
    const availableSlots = Math.max(0, config.maxOpenOrders - this.openOrderIndexes.size);
    const lockedQuantity = config.lockBorrowedLenderCollateral
      ? (lot.lentQuote ?? 0) / Math.max(upperPrice, MIN_BASE_QUANTITY)
      : 0;
    const availableQuantity = Math.max(0, lot.remainingQuantity - lockedQuantity);
    const orderCount = this.legacyExitGridOrderCount({
      lowerPrice,
      upperPrice,
      currentPrice,
      availableQuantity,
      availableSlots,
    });

    if (orderCount <= 0 || upperPrice <= lowerPrice || availableQuantity <= MIN_BASE_QUANTITY) {
      return [];
    }

    const orders: TradingOrder[] = [];
    let remainingQuantity = availableQuantity;

    for (let index = 0; index < orderCount; index += 1) {
      const price = this.legacyExitGridCoverOrderPrice(
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
      let quantity = roundAsset(Math.min(remainingQuantity, desiredQuantity));

      if (quantity <= MIN_BASE_QUANTITY) {
        break;
      }
      const remainingAfterFill = roundAsset(remainingQuantity - quantity);
      if (
        remainingAfterFill > MIN_BASE_QUANTITY &&
        remainingAfterFill * price < config.minOrderQuote
      ) {
        quantity = remainingQuantity;
      }
      if (quantity * price < config.minOrderQuote) {
        if (remainingQuantity * price >= config.minOrderQuote) {
          quantity = remainingQuantity;
        } else if (index === orderCount - 1 && orders.length === 0) {
          break;
        } else {
          continue;
        }
      }

      const order = this.createTriggeredBuyOrder(
        price,
        quantity,
        createdAt,
        `${LEGACY_EXIT_GRID_REASON}; lot ${lot.id}; entry ${roundQuote(grid.entryPrice)}; trough ${lowerPrice}`,
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

    grid.gridTroughPrice = lowerPrice;
    grid.gridCreatedAt = createdAt;
    grid.gridOrderIds = orders.map((order) => order.id);
    this.recordLegacyExitGridSpan(upperPrice - lowerPrice, orders.length);
    return orders;
  }

  private legacyExitGridOrderCount(input: {
    lowerPrice: number;
    upperPrice: number;
    currentPrice: number;
    availableQuantity: number;
    availableSlots: number;
  }): number {
    const config = this.state.config.legacyValleyPeak;
    const maxOrderCount = Math.min(config.exitGridOrderCount, input.availableSlots);
    if (maxOrderCount <= 0) {
      return 0;
    }

    const span = Math.max(0, input.upperPrice - input.lowerPrice);
    const stepOrderCount =
      span > 0 && config.exitGridMaxStepPct > 0
        ? Math.ceil(
            span /
              (Math.max(input.currentPrice, MIN_BASE_QUANTITY) *
                (config.exitGridMaxStepPct / 100)),
          ) + 1
        : maxOrderCount;
    return Math.min(maxOrderCount, stepOrderCount);
  }

  private recordLegacyExitGridSpan(span: number, orderCount: number): void {
    if (orderCount <= 0 || !Number.isFinite(span) || span <= 0) {
      return;
    }

    this.state.exitGridSpanTotal = roundQuote(this.state.exitGridSpanTotal + span);
    this.state.exitGridSpanCount += 1;
    this.state.exitGridOrderCountTotal += orderCount;
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

  private legacyExitGridCoverOrderPrice(
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
      return roundQuote(lowerPrice * Math.pow(upperPrice / lowerPrice, progress));
    }

    return roundQuote(lowerPrice + (upperPrice - lowerPrice) * progress);
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
    side?: PositionLotSide,
  ): BotEvent[] {
    const events: BotEvent[] | undefined = collectEvents ? [] : undefined;

    for (const { index, order } of this.openLegacyExitGridOrders(lotId, side)) {
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
          message: `${order.side.toUpperCase()} exit grid order cancelled for reset`,
          order: structuredClone(order),
        });
      }
    }

    return events ?? NO_EVENTS;
  }

  private openLegacyExitGridOrders(
    lotId?: string,
    side?: PositionLotSide,
  ): Array<{ index: number; order: TradingOrder }> {
    const orders: Array<{ index: number; order: TradingOrder }> = [];
    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (
        order?.status === "open" &&
        order.reason.startsWith(LEGACY_EXIT_GRID_REASON) &&
        (!side || (side === "long" ? order.side === "sell" : order.side === "buy")) &&
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
      const closedShortQuantity = this.shortCloseQuantityForOrder(order);
      const spent = roundQuote(quoteQuantity + feeQuote);
      this.state.quoteReserved = roundQuote(
        Math.max(0, this.state.quoteReserved - order.estimatedQuoteCost),
      );
      this.state.quoteFree = roundQuote(
        this.state.quoteFree + Math.max(0, order.estimatedQuoteCost - spent),
      );

      this.state.baseFree = roundAsset(this.state.baseFree + order.quantity);
      if (closedShortQuantity > MIN_BASE_QUANTITY) {
        const entryPrice = this.shortEntryPriceForCloseOrder(order);
        const feeForClosedQuantity = feeQuote * (closedShortQuantity / order.quantity);
        realizedPnl = roundQuote(
          (entryPrice - order.price) * closedShortQuantity - feeForClosedQuantity,
        );
      }
    } else {
      const closedLongQuantity = this.longCloseQuantityForOrder(order);
      if (order.positionEffect === "open") {
        this.state.baseFree = roundAsset(this.state.baseFree - order.quantity);
      } else {
        this.state.baseReserved = roundAsset(
          Math.max(0, this.state.baseReserved - order.quantity),
        );
        const openedShortQuantity = roundAsset(
          Math.max(0, order.quantity - closedLongQuantity),
        );
        if (openedShortQuantity > MIN_BASE_QUANTITY) {
          this.state.baseFree = roundAsset(this.state.baseFree - openedShortQuantity);
        }
      }
      const proceeds = roundQuote(quoteQuantity - feeQuote);
      this.state.quoteFree = roundQuote(this.state.quoteFree + proceeds);
      if (closedLongQuantity > MIN_BASE_QUANTITY) {
        const entryPrice = this.longEntryPriceForCloseOrder(order);
        const feeForClosedQuantity = feeQuote * (closedLongQuantity / order.quantity);
        realizedPnl = roundQuote(
          (order.price - entryPrice) * closedLongQuantity - feeForClosedQuantity,
        );
      }
    }

    this.state.realizedPnl = roundQuote(this.state.realizedPnl + realizedPnl);
    if (realizedPnl > 0) {
      this.state.winningTrades += 1;
    } else if (realizedPnl < 0) {
      this.state.losingTrades += 1;
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
    this.refreshAverageEntryPrices();
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

    const cachedLot = this.legacyPositionLotCache?.longs.get(order.targetPositionId);
    if (cachedLot) {
      return cachedLot.averagePrice;
    }

    const target = analyzePositions(this.state, {
      currentPrice: this.state.lastPrice || order.price,
    }).longs.find((lot) => lot.id === order.targetPositionId);

    return cleanPositive(target?.averagePrice) || this.state.avgEntryPrice;
  }

  private shortEntryPriceForCloseOrder(order: TradingOrder): number {
    if (!order.targetPositionId) {
      return this.state.avgShortEntryPrice;
    }

    const cachedLot = this.legacyPositionLotCache?.shorts.get(order.targetPositionId);
    if (cachedLot) {
      return cachedLot.averagePrice;
    }

    const target = analyzePositions(this.state, {
      currentPrice: this.state.lastPrice || order.price,
    }).shorts.find((lot) => lot.id === order.targetPositionId);

    return cleanPositive(target?.averagePrice) || this.state.avgShortEntryPrice;
  }

  private longCloseQuantityForOrder(order: TradingOrder): number {
    if (order.positionEffect === "open") {
      return 0;
    }
    if (order.targetPositionId) {
      const cachedLot = this.legacyPositionLotCache?.longs.get(order.targetPositionId);
      if (cachedLot) {
        return roundAsset(Math.min(order.quantity, cachedLot.remainingQuantity));
      }
      const target = analyzePositions(this.state, {
        currentPrice: this.state.lastPrice || order.price,
      }).longs.find((lot) => lot.id === order.targetPositionId);
      return roundAsset(Math.min(order.quantity, target?.remainingQuantity ?? 0));
    }
    return roundAsset(Math.min(order.quantity, this.activeLongQuantity()));
  }

  private shortCloseQuantityForOrder(order: TradingOrder): number {
    if (order.positionEffect === "open") {
      return 0;
    }
    if (order.targetPositionId) {
      const cachedLot = this.legacyPositionLotCache?.shorts.get(order.targetPositionId);
      if (cachedLot) {
        return roundAsset(Math.min(order.quantity, cachedLot.remainingQuantity));
      }
      const target = analyzePositions(this.state, {
        currentPrice: this.state.lastPrice || order.price,
      }).shorts.find((lot) => lot.id === order.targetPositionId);
      return roundAsset(Math.min(order.quantity, target?.remainingQuantity ?? 0));
    }
    return roundAsset(Math.min(order.quantity, this.activeShortQuantity()));
  }

  private refreshAverageEntryPrices(): void {
    this.updateLegacyPositionLotCache();
    const cache = this.legacyPositionLotCache;
    if (!cache) {
      this.state.avgEntryPrice = inferAverageLongEntryPrice(this.state);
      this.state.avgShortEntryPrice = inferAverageShortEntryPrice(this.state);
      return;
    }

    let longQuantity = 0;
    let longCost = 0;
    for (const lot of cache.longs.values()) {
      if (lot.remainingQuantity <= MIN_BASE_QUANTITY) {
        continue;
      }
      longQuantity += lot.remainingQuantity;
      longCost += lot.remainingCostQuote;
    }

    let shortQuantity = 0;
    let shortProceeds = 0;
    for (const lot of cache.shorts.values()) {
      if (lot.remainingQuantity <= MIN_BASE_QUANTITY) {
        continue;
      }
      shortQuantity += lot.remainingQuantity;
      shortProceeds += lot.remainingProceedsQuote;
    }

    this.state.avgEntryPrice = longQuantity > 0 ? roundQuote(longCost / longQuantity) : 0;
    this.state.avgShortEntryPrice =
      shortQuantity > 0 ? roundQuote(shortProceeds / shortQuantity) : 0;
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
    const remainingQuantity = roundAsset(Math.max(0, order.quantity - order.filledQuantity));
    if (order.side === "buy") {
      const reserveQuote = remainingOrderReserveQuote(order, remainingQuantity);
      this.state.quoteReserved = roundQuote(
        Math.max(0, this.state.quoteReserved - reserveQuote),
      );
      this.state.quoteFree = roundQuote(this.state.quoteFree + reserveQuote);
    } else if (order.positionEffect === "open") {
      return;
    } else {
      this.state.baseReserved = roundAsset(
        Math.max(0, this.state.baseReserved - remainingQuantity),
      );
      this.state.baseFree = roundAsset(this.state.baseFree + remainingQuantity);
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

  private activeLongQuantity(): number {
    return roundAsset(
      this.activeLegacyLongLots(this.state.lastPrice || this.state.avgEntryPrice || 0).reduce(
        (quantity, lot) => quantity + lot.remainingQuantity,
        0,
      ),
    );
  }

  private activeShortQuantity(): number {
    return roundAsset(
      this.activeLegacyShortLots(this.state.lastPrice || this.state.avgShortEntryPrice || 0).reduce(
        (quantity, lot) => quantity + lot.remainingQuantity,
        0,
      ),
    );
  }

  private activeLongExposureQuote(marketPrice: number): number {
    return roundQuote(this.activeLongQuantity() * marketPrice);
  }

  private activeShortExposureQuote(marketPrice: number): number {
    return roundQuote(this.activeShortQuantity() * marketPrice);
  }

  private pendingLongEntryQuote(): number {
    let pendingQuote = 0;
    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (
        order?.status === "open" &&
        order.side === "buy" &&
        order.positionEffect !== "close"
      ) {
        pendingQuote += order.price * order.quantity;
      }
    }
    return roundQuote(pendingQuote);
  }

  private pendingShortEntryQuote(): number {
    let pendingQuote = 0;
    for (const index of this.openOrderIndexes) {
      const order = this.state.orders[index];
      if (
        order?.status === "open" &&
        order.side === "sell" &&
        order.positionEffect === "open"
      ) {
        pendingQuote += order.price * order.quantity;
      }
    }
    return roundQuote(pendingQuote);
  }

  private targetLongEntryLeverage(_marketPrice: number): number {
    return clamp(cleanPositive(this.state.config.maxLeverage) || 1, 1, 999);
  }

  private targetShortEntryLeverage(_marketPrice: number): number {
    return clamp(cleanPositive(this.state.config.maxLeverage) || 1, 1, 999);
  }

  private longEntryBuyingPowerQuote(
    marketPrice: number,
    targetEntryLeverage = this.targetLongEntryLeverage(marketPrice),
  ): number {
    const config = this.state.config;
    const equity = this.equityAt(marketPrice);
    if (!config.legacyValleyPeak.longSideEnabled || equity <= 0 || marketPrice <= 0) {
      return 0;
    }

    const feeRate = config.feeBps / 10_000;
    const longExposureQuote = this.activeLongExposureQuote(marketPrice);
    const pendingLongEntryQuote = this.pendingLongEntryQuote();
    const positionCapacity = Math.max(
      0,
      config.maxPositionQuote - longExposureQuote - pendingLongEntryQuote,
    );
    const internalBorrowCapacity = this.longInternalBorrowCapacityQuote(marketPrice);
    if (config.shortMarginModel === "futures-margin") {
      return roundQuote(
        Math.min(
          positionCapacity,
          this.grossEntryLeverageCapacityQuote(marketPrice, targetEntryLeverage),
        ) + internalBorrowCapacity,
      );
    }

    const leverageCapacity = Math.max(
      0,
      (
        targetEntryLeverage * equity -
        longExposureQuote -
        pendingLongEntryQuote * (1 + targetEntryLeverage * feeRate)
      ) /
        (1 + targetEntryLeverage * feeRate),
    );

    return roundQuote(Math.min(positionCapacity, leverageCapacity) + internalBorrowCapacity);
  }

  private shortEntrySellingPowerQuote(
    marketPrice: number,
    targetEntryLeverage = this.targetShortEntryLeverage(marketPrice),
  ): number {
    const config = this.state.config;
    const equity = this.equityAt(marketPrice);
    if (
      !config.legacyValleyPeak.shortSideEnabled ||
      equity <= 0 ||
      marketPrice <= 0
    ) {
      return 0;
    }

    const feeRate = config.feeBps / 10_000;
    const shortExposureQuote = this.activeShortExposureQuote(marketPrice);
    const pendingShortEntryQuote = this.pendingShortEntryQuote();
    const positionCapacity = Math.max(
      0,
      config.maxPositionQuote - shortExposureQuote - pendingShortEntryQuote,
    );
    const internalBorrowCapacity = this.shortInternalBorrowCapacityQuote(marketPrice);
    if (config.shortMarginModel === "futures-margin") {
      return roundQuote(
        Math.min(
          positionCapacity,
          this.grossEntryLeverageCapacityQuote(marketPrice, targetEntryLeverage),
        ) + internalBorrowCapacity,
      );
    }

    const leverageCapacity = Math.max(
      0,
      (
        targetEntryLeverage * equity -
        shortExposureQuote -
        pendingShortEntryQuote * (1 + targetEntryLeverage * feeRate)
      ) /
        (1 + targetEntryLeverage * feeRate),
    );

    return roundQuote(Math.min(positionCapacity, leverageCapacity) + internalBorrowCapacity);
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
      exitGridSpanTotal: this.state.exitGridSpanTotal,
      exitGridSpanCount: this.state.exitGridSpanCount,
      exitGridOrderCountTotal: this.state.exitGridOrderCountTotal,
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
    this.state.exitGridSpanTotal = rollback.exitGridSpanTotal;
    this.state.exitGridSpanCount = rollback.exitGridSpanCount;
    this.state.exitGridOrderCountTotal = rollback.exitGridOrderCountTotal;
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

    if (this.state.config.shortMarginModel === "futures-margin") {
      const effectiveLeverage = this.grossExposureLeverage();
      if (effectiveLeverage > maxLeverage + 0.0001) {
        throw new Error(
          `Leverage limit exceeded: ${formatLeverageForError(effectiveLeverage)}x > ${formatLeverageForError(maxLeverage)}x.`,
        );
      }
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

  private grossEntryLeverageCapacityQuote(
    marketPrice: number,
    targetEntryLeverage: number,
  ): number {
    const config = this.state.config;
    const equity = this.equityAt(marketPrice);
    if (equity <= 0 || marketPrice <= 0) {
      return 0;
    }

    const feeRate = config.feeBps / 10_000;
    const grossExposureQuote = this.marginGrossExposureQuote(marketPrice);
    const pendingEntryQuote =
      this.pendingLongEntryQuote() + this.pendingShortEntryQuote();

    return Math.max(
      0,
      (
        targetEntryLeverage * equity -
        grossExposureQuote -
        pendingEntryQuote * (1 + targetEntryLeverage * feeRate)
      ) /
        (1 + targetEntryLeverage * feeRate),
    );
  }

  private grossExposureLeverage(): number {
    const price =
      cleanPositive(this.state.lastPrice) ||
      cleanPositive(this.state.avgEntryPrice) ||
      cleanPositive(this.state.avgShortEntryPrice);
    if (price <= 0) {
      return 1;
    }

    this.updateLegacyPositionLotCache();
    const grossExposureQuote = this.marginGrossExposureQuote(price);
    if (grossExposureQuote <= 0) {
      return 1;
    }

    const equity = this.equityAt(price);
    if (equity <= 0) {
      return 999;
    }

    return clamp(grossExposureQuote / equity, 1, 999);
  }

  private marginGrossExposureQuote(marketPrice: number): number {
    if (marketPrice <= 0) {
      return 0;
    }

    const grossExposureQuote =
      this.activeLongExposureQuote(marketPrice) +
      this.activeShortExposureQuote(marketPrice);
    return roundQuote(
      Math.max(0, grossExposureQuote - this.internalBorrowedExposureQuote(marketPrice)),
    );
  }

  private internalBorrowedExposureQuote(marketPrice: number): number {
    this.updateLegacyPositionLotCache();
    const cache = this.legacyPositionLotCache;
    if (!cache || marketPrice <= 0) {
      return 0;
    }

    let quantity = 0;
    for (const short of cache.shorts.values()) {
      for (const allocation of short.borrowAllocations) {
        quantity += Math.max(0, allocation.quantity);
      }
    }
    for (const long of cache.longs.values()) {
      for (const allocation of long.borrowAllocations) {
        quantity += Math.max(0, allocation.quantity);
      }
    }

    return roundQuote(quantity * marketPrice);
  }

  private shortInternalBorrowCapacityQuote(marketPrice: number): number {
    this.updateLegacyPositionLotCache();
    const cache = this.legacyPositionLotCache;
    if (!cache || marketPrice <= 0) {
      return 0;
    }

    let quote = 0;
    for (const lot of cache.longs.values()) {
      if (lot.borrowDepthRemaining <= 0) {
        continue;
      }
      const availableQuantity = Math.max(0, lot.remainingQuantity - lot.lentQuantity);
      if (availableQuantity <= MIN_BASE_QUANTITY) {
        continue;
      }
      quote += availableQuantity * marketPrice;
    }

    return roundQuote(quote);
  }

  private longInternalBorrowCapacityQuote(_marketPrice: number): number {
    this.updateLegacyPositionLotCache();
    const cache = this.legacyPositionLotCache;
    if (!cache) {
      return 0;
    }

    let quote = 0;
    for (const lot of cache.shorts.values()) {
      if (
        lot.borrowDepthRemaining <= 0 ||
        lot.remainingQuantity <= MIN_BASE_QUANTITY ||
        lot.remainingProceedsQuote <= 0
      ) {
        continue;
      }
      quote += Math.max(0, lot.remainingProceedsQuote);
    }

    return roundQuote(quote);
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
  normalized.exitGridSpanTotal ??= 0;
  normalized.exitGridSpanCount ??= 0;
  normalized.exitGridOrderCountTotal ??= 0;
  normalized.avgEntryPrice = inferAverageLongEntryPrice(normalized);
  normalized.avgShortEntryPrice = inferAverageShortEntryPrice(normalized);
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
    shortMarginModel: overrides.shortMarginModel ?? base.shortMarginModel,
    longBorrowDepth: overrides.longBorrowDepth ?? base.longBorrowDepth,
    shortBorrowDepth: overrides.shortBorrowDepth ?? base.shortBorrowDepth,
    lockBorrowedLenderCollateral:
      overrides.lockBorrowedLenderCollateral ?? base.lockBorrowedLenderCollateral,
    borrowerProfitShareToLender:
      overrides.borrowerProfitShareToLender ?? base.borrowerProfitShareToLender,
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

function normalizeShortMarginModel(
  value: StrategyConfig["shortMarginModel"],
): StrategyConfig["shortMarginModel"] {
  return value === "futures-margin" ? "futures-margin" : "spot-borrow";
}

function normalizeBorrowDepth(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.round(value));
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
  const netPnl = roundQuote(equity - state.startingQuote);
  const unrealizedPnl = roundQuote(netPnl - state.realizedPnl);
  const peakEquity = Math.max(state.metrics?.peakEquity ?? state.startingQuote, equity);
  const drawdown = peakEquity > 0 ? ((peakEquity - equity) / peakEquity) * 100 : 0;
  const totalClosedTrades = state.winningTrades + state.losingTrades;
  const exitGridSpanCount = Math.max(0, state.exitGridSpanCount ?? 0);

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
    avgExitGridSpan:
      exitGridSpanCount > 0
        ? roundQuote((state.exitGridSpanTotal ?? 0) / exitGridSpanCount)
        : 0,
    avgExitGridOrderCount:
      exitGridSpanCount > 0
        ? (state.exitGridOrderCountTotal ?? 0) / exitGridSpanCount
        : 0,
    exitGridSpanCount,
  };
}

function inferAverageLongEntryPrice(state: PaperBotState): number {
  const positions = analyzePositions(state);
  let quantity = 0;
  let cost = 0;

  for (const lot of positions.longs) {
    if (lot.status === "pending" || lot.remainingQuantity <= 0) {
      continue;
    }
    quantity += lot.remainingQuantity;
    cost += lot.remainingCostQuote;
  }

  return quantity > 0 ? roundQuote(cost / quantity) : 0;
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
    avgExitGridSpan: 0,
    avgExitGridOrderCount: 0,
    exitGridSpanCount: 0,
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

function remainingOrderReserveQuote(
  order: TradingOrder,
  quantity = Math.max(0, order.quantity - order.filledQuantity),
): number {
  if (order.side !== "buy" || order.quantity <= MIN_BASE_QUANTITY) {
    return 0;
  }
  return roundQuote(order.estimatedQuoteCost * (quantity / order.quantity));
}

function cleanFiniteNumber(value: number | undefined): number {
  return Number.isFinite(value) ? (value as number) : 0;
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
