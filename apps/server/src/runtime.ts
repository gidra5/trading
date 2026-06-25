import {
  SimulatedTradingBot,
  analyzePositions,
  createStrategyConfig,
  runBacktestFromCandles,
  runBacktestFromOrderBook,
  type BacktestPreset,
  type BacktestProgressSnapshot,
  type BacktestResult,
  type BotEvent,
  type Candle,
  type ExchangeReconciliationInput,
  type ManualTradeInput,
  type OrderBookSnapshot,
  type PartialStrategyConfig,
  type PaperBotState,
  type PriceTick,
  type PositionLedger,
  type StrategyMemory,
  type StrategyConfig,
  type TradingOrder,
} from "@trading/bot-algo";
import type { MarketStreamStatus } from "./binance-stream.js";
import type { BinanceMarketListing, StreamVenue } from "./binance-markets.js";
import {
  BacktestCancelledError,
  fetchKlines,
  intervalToMs,
  isBacktestCancelledError,
  runHistoricalCandleBacktest,
  type HistoricalBacktestMarket,
} from "./historical-backtest.js";
import {
  BinancePaperTrading,
  type BinancePaperCancelOrderInput,
  type BinancePaperOrder,
  type BinancePaperPlaceOrderInput,
  type BinancePaperSnapshot,
  type BinancePaperTrade,
} from "./binance-paper.js";
import type { TradingStorage } from "./storage.js";

const PUBLIC_ORDER_LIMIT = 500;
const PUBLIC_FILL_LIMIT = 500;
const PUBLIC_PRICE_MEMORY_LIMIT = 100;

interface BacktestStartOptions {
  preset: BacktestPreset;
  limit: number;
  startingQuote?: number;
  historicalDays?: number;
  randomSampleCount?: number;
  randomWindowDays?: number;
  randomMinWindowDays?: number;
  randomMaxWindowDays?: number;
  randomLookbackDays?: number;
  randomPairCount?: number;
  randomMarkets?: HistoricalBacktestMarket[];
}

export interface RuntimeSnapshot {
  market: {
    id: string;
    group: string;
    venue: string;
    symbol: string;
    displaySymbol: string;
    baseAsset: string;
    quoteAsset: string;
    interval: string;
    maxLeverage?: number;
    connected: boolean;
    statusMessage: string;
    lastEventAt: number;
    lastPrice: number;
    candles: Candle[];
    orderBook?: OrderBookSnapshot;
  };
  bot: PaperBotState;
  positions: PositionLedger;
  recentEvents: BotEvent[];
  backtest: BacktestProgressSnapshot;
  exchange: BinancePaperSnapshot;
}

export class TradingRuntime {
  private bot!: SimulatedTradingBot;
  private candles: Candle[] = [];
  private orderBook?: OrderBookSnapshot;
  private status: MarketStreamStatus = {
    connected: false,
    message: "Starting",
    lastEventAt: Date.now(),
    reconnectAttempt: 0,
  };
  private recentEvents: BotEvent[] = [];
  private saveTimer?: NodeJS.Timeout;
  private stateSaveQueue: Promise<void> = Promise.resolve();
  private lastSavedOrderBookAt = 0;
  private backtest: BacktestProgressSnapshot = createIdleBacktest();
  private backtestAbort?: AbortController;

  constructor(
    private storage: TradingStorage,
    private market: BinanceMarketListing,
    private config: StrategyConfig,
    private readonly interval: string,
    private readonly historicalCache: {
      dataDir: string;
      maxBytes: number;
      minFreeBytes: number;
    },
    private readonly paperTrading?: BinancePaperTrading,
  ) {}

  async init(): Promise<void> {
    await this.storage.ensureReady();
    this.candles = await this.storage.loadCandles(500);
    const savedState = await this.storage.loadBotState();
    this.config = this.createMarketConfig(savedState);
    this.bot = new SimulatedTradingBot(savedState, this.config);
    await this.warmupBotFromHistory(savedState);
    await this.recoverExchangeState();
  }

  async switchMarket(
    market: BinanceMarketListing,
    storage: TradingStorage,
  ): Promise<void> {
    await this.flushState();
    this.market = market;
    this.storage = storage;
    this.candles = [];
    this.orderBook = undefined;
    this.recentEvents = [];
    this.lastSavedOrderBookAt = 0;
    this.backtestAbort?.abort();
    this.backtestAbort = undefined;
    this.backtest = createIdleBacktest();
    this.status = {
      connected: false,
      message: `Switching to ${market.displaySymbol}`,
      lastEventAt: Date.now(),
      reconnectAttempt: 0,
    };

    await this.storage.ensureReady();
    this.candles = await this.storage.loadCandles(500);
    const savedState = await this.storage.loadBotState();
    this.config = this.createMarketConfig(savedState);
    this.bot = new SimulatedTradingBot(savedState, this.config);
    await this.warmupBotFromHistory(savedState);
    await this.recoverExchangeState();
  }

  handleStatus(status: MarketStreamStatus): void {
    this.status = status;
  }

  async handleTick(tick: PriceTick): Promise<BotEvent[]> {
    const exchangeDriven = this.paperTrading?.drivesOrderExecution(this.market) ?? false;
    const events = this.bot.onTick(tick, {
      processOpenOrders: !exchangeDriven,
      deferMarketOrderFills: exchangeDriven,
    });
    await this.submitCreatedOrdersToPaperExchange(events);
    this.recordEvents(events);
    this.scheduleStateSave();
    return events;
  }

  async handleCandle(candle: Candle): Promise<void> {
    upsertCandle(this.candles, candle, 500);

    if (candle.closed) {
      await this.storage.appendCandle(candle);
    }
  }

  async handleOrderBook(snapshot: OrderBookSnapshot): Promise<void> {
    this.orderBook = snapshot;

    if (snapshot.eventTime - this.lastSavedOrderBookAt >= 5_000) {
      this.lastSavedOrderBookAt = snapshot.eventTime;
      await this.storage.appendOrderBookSnapshot(snapshot);
    }
  }

  async handleExchangeUserData(payload: unknown): Promise<BotEvent[]> {
    if (!this.paperTrading) {
      return [];
    }

    const events: BotEvent[] = [];
    const reconciliation = this.paperTrading.reconciliationFromUserDataEvent(
      this.market,
      payload,
    );
    if (hasExchangeReconciliationUpdates(reconciliation)) {
      const directEvents = this.bot.applyExchangeReconciliation(reconciliation);
      this.recordEvents(directEvents);
      events.push(...directEvents);
      await this.flushState();
    }

    try {
      const snapshot = await this.paperTrading.sync(this.market);
      events.push(...(await this.applyExchangeSnapshot(snapshot)));
    } catch {
      return events;
    }

    return events;
  }

  snapshot(): RuntimeSnapshot {
    const bot = this.bot.view();
    const publicBot = compactPublicBotState(bot);
    return {
      market: {
        id: this.market.id,
        group: this.market.group,
        venue: this.market.venue,
        symbol: this.config.symbol,
        displaySymbol: this.market.displaySymbol,
        baseAsset: this.config.baseAsset,
        quoteAsset: this.config.quoteAsset,
        interval: this.interval,
        maxLeverage: this.market.maxLeverage,
        connected: this.status.connected,
        statusMessage: this.status.message,
        lastEventAt: this.status.lastEventAt,
        lastPrice: bot.lastPrice,
        candles: this.candles,
        orderBook: this.orderBook,
      },
      bot: publicBot,
      positions: analyzePositions(bot as PaperBotState),
      recentEvents: compactPublicEvents(this.recentEvents),
      backtest: this.backtest,
      exchange: this.paperTrading?.snapshot(this.market) ?? disabledExchangeSnapshot(),
    };
  }

  async startBot(): Promise<BotEvent[]> {
    const events = this.bot.setStatus("running");
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  async stopBot(): Promise<BotEvent[]> {
    const events = this.bot.setStatus("stopped");
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  async resetBot(): Promise<BotEvent[]> {
    const exchangeCanSubmit = this.paperTrading?.canSubmitOrders(this.market) ?? false;
    if (exchangeCanSubmit) {
      await this.paperTrading!.cancelAllOpenOrders(this.market);
      await this.paperTrading!.closeOpenPositions(this.market, {
        includeUnprofitable: true,
      });
    }

    const events = this.bot.reset(this.config);
    this.recordEvents(events);
    await this.flushState();
    if (exchangeCanSubmit) {
      const snapshot = await this.paperTrading!.sync(this.market);
      await this.applyExchangeSnapshot(snapshot);
    }
    return events;
  }

  async closePositions(
    options: { includeUnprofitable?: boolean } = {},
  ): Promise<BotEvent[]> {
    const at = Date.now();
    const events: BotEvent[] = [];
    events.push(...this.bot.setStatus("stopped", at));

    const exchangeCanSubmit = this.paperTrading?.canSubmitOrders(this.market) ?? false;
    if (exchangeCanSubmit) {
      try {
        const snapshot = await this.paperTrading!.cancelAllOpenOrders(this.market);
        await this.applyExchangeSnapshot(snapshot);
      } catch {
        // Local open orders are cancelled below so the strategy will not place follow-up orders.
      }
    }

    events.push(...this.bot.cancelOpenOrders("close positions requested", at));

    if (exchangeCanSubmit) {
      if (options.includeUnprofitable) {
        this.recordEvents(events);
        const snapshot = await this.paperTrading!.closeOpenPositions(this.market, options);
        await this.applyExchangeSnapshot(snapshot);
        await this.flushState();
        return events;
      }

      const closeEvents = this.bot.createPositionCloseOrders(options, at);
      events.push(...closeEvents);
      this.recordEvents(events);
      await this.submitCreatedOrdersToPaperExchange(closeEvents, {
        force: true,
      });
      const snapshot = await this.paperTrading!.closeOpenPositions(this.market, options);
      await this.applyExchangeSnapshot(snapshot);
      await this.flushState();
      return events;
    }

    events.push(...this.closePositionsLocally(options, at));
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  async updateBotConfig(patch: PartialStrategyConfig): Promise<BotEvent[]> {
    this.config = applyMarketMaxLeverage(
      createStrategyConfig({
        ...this.config,
        ...patch,
        symbol: this.market.symbol,
        baseAsset: this.market.baseAsset,
        quoteAsset: this.market.quoteAsset,
        legacyValleyPeak: {
          ...this.config.legacyValleyPeak,
          ...(patch.legacyValleyPeak ?? {}),
        },
        positionRisk: {
          ...this.config.positionRisk,
          ...(patch.positionRisk ?? {}),
        },
      }),
      this.market.maxLeverage,
    );
    const events = this.bot.reset(this.config);
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  async recordManualTrade(input: ManualTradeInput): Promise<BotEvent[]> {
    const quantity = Number(input.quantity);
    if (!Number.isFinite(quantity) || quantity <= 0) {
      throw new Error("Manual trade quantity must be positive.");
    }
    if (input.price !== undefined && (!Number.isFinite(input.price) || input.price <= 0)) {
      throw new Error("Manual trade price must be positive.");
    }
    if (input.side !== "buy" && input.side !== "sell") {
      throw new Error("Manual trade side must be buy or sell.");
    }
    const lifetimeMs = normalizeOptionalPositiveNumber(input.lifetimeMs, "Lot lifetime");
    const stopLossPrice = normalizeOptionalPositiveNumber(input.stopLossPrice, "Stop loss");
    const takeProfitPrice = normalizeOptionalPositiveNumber(input.takeProfitPrice, "Take profit");

    if (input.targetPositionId) {
      const positions = analyzePositions(this.bot.snapshot());
      const target =
        input.side === "sell"
          ? positions.longs.find((lot) => lot.id === input.targetPositionId)
          : positions.shorts.find((lot) => lot.id === input.targetPositionId);

      if (!target || target.status === "pending") {
        throw new Error("Target position is no longer open.");
      }
      if (quantity > target.remainingQuantity + 0.00000001) {
        throw new Error("Close quantity is larger than the target position.");
      }
    }

    const events = this.bot.recordManualTrade({
      ...input,
      quantity,
      price: input.price === undefined ? undefined : Number(input.price),
      lifetimeMs,
      stopLossPrice,
      takeProfitPrice,
    });
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  async syncExchange(): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance paper trading is not configured.");
    }
    const snapshot = await this.paperTrading.sync(this.market);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async placeExchangeOrder(
    input: BinancePaperPlaceOrderInput,
  ): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance paper trading is not configured.");
    }
    const snapshot = await this.paperTrading.placeOrder(this.market, input);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async cancelExchangeOrder(
    input: BinancePaperCancelOrderInput,
  ): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance paper trading is not configured.");
    }
    const snapshot = await this.paperTrading.cancelOrder(this.market, input);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async cancelAllExchangeOrders(): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance paper trading is not configured.");
    }
    const snapshot = await this.paperTrading.cancelAllOpenOrders(this.market);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async setExchangeLeverage(leverage: number): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance paper trading is not configured.");
    }
    const snapshot = await this.paperTrading.changeLeverage(this.market, leverage);
    this.applyExchangeMaxLeverage(snapshot);
    await this.flushState();
    return snapshot;
  }

  startBacktest(
    options: BacktestStartOptions,
    onUpdate: () => void,
  ): BacktestProgressSnapshot {
    if (this.backtest.status === "running") {
      throw new Error("A backtest is already running.");
    }

    const id = `bt_${Date.now()}`;
    const abortController = new AbortController();
    this.backtestAbort = abortController;
    const source = options.preset === "saved-orderbook" ? "orderbook-mid" : "candles";
    const now = Date.now();
    this.backtest = {
      id,
      preset: options.preset,
      status: "running",
      source,
      startedAt: now,
      updatedAt: now,
      targetStartTime: 0,
      targetEndTime: now,
      processedCandles: 0,
      estimatedCandles: 0,
      requests: 0,
      marketCount: options.randomMarkets ? options.randomMarkets.length + 1 : undefined,
      randomPairCount: options.randomMarkets?.length ?? options.randomPairCount,
      percent: 0,
      equity: this.config.startingQuote,
      returnPct: 0,
      message: "Starting backtest",
    };
    onUpdate();

    void this.executeBacktest(id, options, onUpdate, abortController);
    return this.backtest;
  }

  stopBacktest(onUpdate: () => void): BacktestProgressSnapshot {
    if (this.backtest.status !== "running") {
      return this.backtest;
    }

    this.backtestAbort?.abort();
    this.backtest = {
      ...this.backtest,
      status: "cancelled",
      stopReason: "cancelled",
      updatedAt: Date.now(),
      message: "Backtest cancelled",
    };
    onUpdate();
    return this.backtest;
  }

  async flushState(): Promise<void> {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = undefined;
    }

    const snapshot = this.bot.snapshot();
    this.stateSaveQueue = this.stateSaveQueue.then(
      () => this.storage.saveBotState(snapshot),
      () => this.storage.saveBotState(snapshot),
    );
    await this.stateSaveQueue;
  }

  private scheduleStateSave(): void {
    if (this.saveTimer) {
      return;
    }

    this.saveTimer = setTimeout(() => {
      this.saveTimer = undefined;
      void this.flushState();
    }, 2_000);
  }

  private recordEvents(events: BotEvent[]): void {
    if (events.length === 0) {
      return;
    }

    this.recentEvents = [...events.map(compactPublicEvent), ...this.recentEvents].slice(0, 60);
  }

  private async submitCreatedOrdersToPaperExchange(
    events: BotEvent[],
    options: { force?: boolean; throwOnFailure?: boolean } = {},
  ): Promise<void> {
    if (!this.paperTrading) {
      return;
    }
    for (const event of events) {
      if (event.type !== "order_created" || !event.order) {
        continue;
      }
      try {
        const snapshot = await this.paperTrading.submitBotOrder(
          this.market,
          event.order,
          { force: options.force },
        );
        if (snapshot) {
          await this.applyExchangeSnapshot(snapshot);
        }
      } catch (error) {
        const cancelled = this.bot.cancelOpenOrder(event.order.id, "exchange submit failed");
        this.recordEvents(cancelled);
        await this.flushState();
        if (options.throwOnFailure) {
          throw error;
        }
        return;
      }
    }
  }

  private closePositionsLocally(
    options: { includeUnprofitable?: boolean },
    at: number,
  ): BotEvent[] {
    const state = this.bot.snapshot();
    const price = state.lastPrice;
    if (!Number.isFinite(price) || price <= 0) {
      return [];
    }

    const ledger = analyzePositions(state, { currentPrice: price });
    const reason = options.includeUnprofitable
      ? "forced local position close"
      : "local profitable position close";
    const events: BotEvent[] = [];
    for (const lot of ledger.longs) {
      if (
        lot.status === "pending" ||
        lot.remainingQuantity <= 0 ||
        (!options.includeUnprofitable && price < lot.breakEvenSellPrice)
      ) {
        continue;
      }
      events.push(
        ...this.bot.recordManualTrade({
          side: "sell",
          price,
          quantity: lot.remainingQuantity,
          targetPositionId: lot.id,
          positionEffect: "close",
          reason,
        }, at),
      );
    }
    for (const lot of ledger.shorts) {
      if (
        lot.status === "pending" ||
        lot.remainingQuantity <= 0 ||
        (!options.includeUnprofitable && price > lot.breakEvenBuyPrice)
      ) {
        continue;
      }
      events.push(
        ...this.bot.recordManualTrade({
          side: "buy",
          price,
          quantity: lot.remainingQuantity,
          targetPositionId: lot.id,
          positionEffect: "close",
          reason,
        }, at),
      );
    }
    return events;
  }

  private async recoverExchangeState(): Promise<void> {
    if (!this.paperTrading?.drivesOrderExecution(this.market)) {
      return;
    }
    try {
      const snapshot = await this.paperTrading.sync(this.market);
      await this.applyExchangeSnapshot(snapshot);
    } catch {
      return;
    }
  }

  private async warmupBotFromHistory(savedState?: PaperBotState): Promise<void> {
    if (!this.needsHistoricalWarmup(savedState) || !isHistoricalVenue(this.market.venue)) {
      return;
    }

    const candles = await this.loadWarmupCandles();
    if (candles.length === 0) {
      return;
    }

    const processed = this.bot.warmupFromCandles(candles);
    if (processed <= 0) {
      return;
    }
    for (const candle of candles) {
      upsertCandle(this.candles, candle, 500);
    }
    await this.flushState();
  }

  private needsHistoricalWarmup(savedState?: PaperBotState): boolean {
    const memory = savedState?.memory.legacyValleyPeak;
    if (!memory) {
      return true;
    }
    return memory.buyAverages.every((average) => average.timestamps.length === 0);
  }

  private async loadWarmupCandles(): Promise<Candle[]> {
    const intervalMs = intervalToMs(this.interval);
    const config = this.config.legacyValleyPeak;
    const warmupMs =
      (Math.max(...config.averagingRangesSec, 0) + config.saturationSec) * 1000;
    const requiredCandles = Math.max(10, Math.ceil(warmupMs / intervalMs) + 5);
    const storedCandles = this.candles
      .filter((candle) => candle.closed)
      .slice(-requiredCandles);
    const storedSpan =
      storedCandles.length > 0
        ? storedCandles.at(-1)!.closeTime - storedCandles[0].openTime
        : 0;
    if (storedSpan >= warmupMs) {
      return storedCandles;
    }

    const limit = Math.min(1000, requiredCandles);
    const endTime = Date.now();
    const startTime = endTime - limit * intervalMs;
    if (!isHistoricalVenue(this.market.venue)) {
      return storedCandles;
    }
    try {
      return await fetchKlines({
        venue: this.market.venue,
        symbol: this.market.symbol,
        interval: this.interval,
        startTime,
        endTime,
        limit,
        endpoint: this.paperTrading?.klineEndpointFor(this.market),
      });
    } catch {
      return storedCandles;
    }
  }

  private async applyExchangeSnapshot(snapshot: BinancePaperSnapshot): Promise<BotEvent[]> {
    this.applyExchangeMaxLeverage(snapshot);
    this.applyExchangeTradingRules(snapshot);
    const reconciliation = exchangeReconciliationFromSnapshot(snapshot);
    if (
      (reconciliation.orders?.length ?? 0) === 0 &&
      (reconciliation.fills?.length ?? 0) === 0
    ) {
      await this.flushState();
      return [];
    }

    const events = this.bot.applyExchangeReconciliation(reconciliation);
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  private applyExchangeMaxLeverage(snapshot: BinancePaperSnapshot): void {
    if (!snapshot.maxLeverage || snapshot.maxLeverage < 1) {
      return;
    }
    const capped = applyMarketMaxLeverage(this.config, snapshot.maxLeverage);
    if (capped.maxLeverage === this.config.maxLeverage) {
      return;
    }
    this.config = capped;
    this.bot.setConfig(capped);
  }

  private applyExchangeTradingRules(snapshot: BinancePaperSnapshot): void {
    let next = this.config;
    const minOrderQuote = snapshot.symbolFilters?.minNotional;
    if (minOrderQuote && minOrderQuote > 0) {
      next = createStrategyConfig({
        ...next,
        minOrderQuote: Math.max(next.minOrderQuote, minOrderQuote),
        legacyValleyPeak: {
          ...next.legacyValleyPeak,
          minTradeQuote: Math.max(next.legacyValleyPeak.minTradeQuote, minOrderQuote),
          maxTradeQuote: snapshot.symbolFilters?.maxNotional
            ? Math.min(next.legacyValleyPeak.maxTradeQuote, snapshot.symbolFilters.maxNotional)
            : next.legacyValleyPeak.maxTradeQuote,
        },
      });
    }
    if (snapshot.feeBps !== undefined && Number.isFinite(snapshot.feeBps)) {
      next = createStrategyConfig({
        ...next,
        feeBps: snapshot.feeBps,
      });
    }
    if (
      snapshot.estimatedSlippageBps !== undefined &&
      Number.isFinite(snapshot.estimatedSlippageBps) &&
      snapshot.estimatedSlippageBps >= 0
    ) {
      next = createStrategyConfig({
        ...next,
        positionRisk: {
          ...next.positionRisk,
          marketSlippageBps: snapshot.estimatedSlippageBps,
        },
      });
    }

    if (JSON.stringify(next) === JSON.stringify(this.config)) {
      return;
    }
    this.config = next;
    this.bot.setConfig(next);
  }

  private async executeBacktest(
    id: string,
    options: BacktestStartOptions,
    onUpdate: () => void,
    abortController: AbortController,
  ): Promise<void> {
    try {
      const result = await this.runBacktestNow(
        id,
        options,
        onUpdate,
        abortController.signal,
      );
      if (abortController.signal.aborted) {
        throw new BacktestCancelledError();
      }
      await this.storage.saveBacktest(result);
      if (this.backtest.id !== id) {
        return;
      }
      this.backtest = buildCompletedProgress(
        this.backtest,
        result,
        result.summary.stoppedEarly
          ? result.summary.stopReason === "liquidated"
            ? "Stopped early after account liquidation"
            : "Stopped early after portfolio wipeout"
          : "Backtest completed",
      );
      onUpdate();
    } catch (error) {
      if (this.backtest.id !== id) {
        return;
      }
      if (abortController.signal.aborted || isBacktestCancelledError(error)) {
        this.backtest = {
          ...this.backtest,
          status: "cancelled",
          stopReason: "cancelled",
          updatedAt: Date.now(),
          message: "Backtest cancelled",
        };
        onUpdate();
        return;
      }
      const message = error instanceof Error ? error.message : "Backtest failed";
      this.backtest = {
        ...this.backtest,
        status: "failed",
        stopReason: "error",
        updatedAt: Date.now(),
        error: message,
        message,
      };
      onUpdate();
    } finally {
      if (this.backtestAbort === abortController) {
        this.backtestAbort = undefined;
      }
    }
  }

  private async runBacktestNow(
    id: string,
    options: BacktestStartOptions,
    onUpdate: () => void,
    cancelSignal: AbortSignal,
  ): Promise<BacktestResult> {
    throwIfBacktestCancelled(cancelSignal);
    const config = {
      ...this.config,
      ...(options.startingQuote ? { startingQuote: options.startingQuote } : {}),
    };

    if (isHistoricalBacktestPreset(options.preset)) {
      const venue = this.market.venue;
      if (!this.market.supportsHistoricalCandles || !isHistoricalVenue(venue)) {
        throw new Error(`${this.market.displaySymbol} does not support candle backtests yet.`);
      }

      return runHistoricalCandleBacktest(
        {
          id,
          preset: options.preset,
          marketId: this.market.id,
          marketKey: this.market.id,
          venue,
          symbol: this.config.symbol,
          displaySymbol: this.market.displaySymbol,
          baseAsset: this.market.baseAsset,
          quoteAsset: this.market.quoteAsset,
          maxLeverage: this.market.maxLeverage,
          interval: this.interval,
          config,
          cache: this.historicalCache,
          historicalRangeMs:
            options.historicalDays === undefined
              ? undefined
              : options.historicalDays * 24 * 60 * 60 * 1000,
          randomSampleCount: options.randomSampleCount,
          randomWindowMs:
            options.randomWindowDays === undefined
              ? undefined
              : options.randomWindowDays * 24 * 60 * 60 * 1000,
          randomMinWindowMs:
            options.randomMinWindowDays === undefined
              ? undefined
              : options.randomMinWindowDays * 24 * 60 * 60 * 1000,
          randomMaxWindowMs:
            options.randomMaxWindowDays === undefined
              ? undefined
              : options.randomMaxWindowDays * 24 * 60 * 60 * 1000,
          randomLookbackMs:
            options.randomLookbackDays === undefined
              ? undefined
              : options.randomLookbackDays * 24 * 60 * 60 * 1000,
          randomMarkets: options.randomMarkets,
          randomPairCount: options.randomMarkets?.length ?? options.randomPairCount,
          cancelSignal,
        },
        (progress) => {
          if (cancelSignal.aborted) {
            return;
          }
          this.backtest = progress;
          onUpdate();
        },
      );
    }

    throwIfBacktestCancelled(cancelSignal);
    const startedAt = Date.now();
    const result =
      options.preset === "saved-orderbook"
        ? runBacktestFromOrderBook(await this.storage.loadOrderBookSnapshots(options.limit), {
            config,
          })
        : runBacktestFromCandles(await this.storage.loadCandles(options.limit), {
            config,
          });
    throwIfBacktestCancelled(cancelSignal);
    const durationMs = Date.now() - startedAt;
    result.summary.durationMs = durationMs;
    result.summary.stopReason = "completed";
    result.summary.stoppedEarly = false;
    result.summary.candlesProcessed =
      options.preset === "saved-candles" ? result.summary.eventsProcessed / 4 : undefined;
    result.summary.requests = 0;
    return result;
  }

  private createMarketConfig(savedState?: PaperBotState): StrategyConfig {
    const config = createStrategyConfig({
      ...(savedState?.config ?? this.config),
      symbol: this.market.symbol,
      baseAsset: this.market.baseAsset,
      quoteAsset: this.market.quoteAsset,
    });
    return applyMarketMaxLeverage(config, this.market.maxLeverage);
  }
}

function disabledExchangeSnapshot(): BinancePaperSnapshot {
  return {
    enabled: false,
    configured: false,
    compatible: true,
    mode: "auto",
    autoSubmit: false,
    connected: false,
    message: "Binance paper trading disabled",
    balances: [],
    positions: [],
    openOrders: [],
    recentOrders: [],
    recentTrades: [],
  };
}

function exchangeReconciliationFromSnapshot(
  snapshot: BinancePaperSnapshot,
): ExchangeReconciliationInput {
  const orderById = new Map<string, BinancePaperOrder>();
  for (const order of [...snapshot.recentOrders, ...snapshot.openOrders]) {
    if (order.localOrderId) {
      orderById.set(order.orderId, order);
    }
  }

  const orders = [...orderById.values()].map((order) => ({
    localOrderId: order.localOrderId,
    externalOrderId: order.orderId,
    clientOrderId: order.clientOrderId,
    side: normalizeExchangeSide(order.side),
    type: normalizeExchangeType(order.type),
    status: normalizeExchangeOrderStatus(order.status),
    price: order.avgPrice || order.price,
    quantity: order.originalQuantity,
    filledQuantity: order.executedQuantity,
    quoteQuantity: order.cumulativeQuoteQuantity,
    createdAt: order.createdAt,
    updatedAt: order.updatedAt,
    positionEffect: order.reduceOnly ? "close" as const : undefined,
    reason: `exchange order ${order.status}`,
  }));

  const fills = snapshot.recentTrades
    .filter((trade) => trade.localOrderId)
    .map((trade) => {
      const order = orderById.get(trade.orderId);
      return {
        id: trade.id,
        localOrderId: trade.localOrderId,
        externalOrderId: trade.orderId,
        clientOrderId: trade.clientOrderId,
        side: trade.side,
        price: trade.price,
        quantity: trade.quantity,
        quoteQuantity: trade.quoteQuantity,
        feeQuote: trade.feeQuote,
        realizedPnl: trade.realizedPnl,
        filledAt: trade.time,
        reason: "exchange fill",
        positionEffect: order?.reduceOnly ? "close" as const : undefined,
      };
    });

  return { orders, fills };
}

function hasExchangeReconciliationUpdates(
  input: ExchangeReconciliationInput | undefined,
): input is ExchangeReconciliationInput {
  return (input?.orders?.length ?? 0) > 0 || (input?.fills?.length ?? 0) > 0;
}

function normalizeOptionalPositiveNumber(
  value: number | undefined,
  label: string,
): number | undefined {
  if (value === undefined || value === null || value === 0) {
    return undefined;
  }

  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    throw new Error(`${label} must be positive.`);
  }

  return number > 0 ? number : undefined;
}

function normalizeExchangeSide(side: string): "buy" | "sell" {
  return side.toUpperCase() === "SELL" ? "sell" : "buy";
}

function normalizeExchangeType(type: string): "limit" | "market" {
  return type.toUpperCase() === "MARKET" ? "market" : "limit";
}

function normalizeExchangeOrderStatus(status: string): "open" | "filled" | "cancelled" {
  const upper = status.toUpperCase();
  if (upper === "FILLED") {
    return "filled";
  }
  if (
    upper === "CANCELED" ||
    upper === "CANCELLED" ||
    upper === "EXPIRED" ||
    upper === "REJECTED"
  ) {
    return "cancelled";
  }
  return "open";
}

function applyMarketMaxLeverage(
  config: StrategyConfig,
  marketMaxLeverage: number | undefined,
): StrategyConfig {
  if (!Number.isFinite(marketMaxLeverage) || (marketMaxLeverage as number) < 1) {
    return config;
  }

  return {
    ...config,
    maxLeverage: Math.min(config.maxLeverage, marketMaxLeverage as number),
  };
}

function isHistoricalVenue(venue: string): venue is StreamVenue {
  return (
    venue === "spot" ||
    venue === "usdm-futures" ||
    venue === "coinm-futures" ||
    venue === "options"
  );
}

function throwIfBacktestCancelled(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new BacktestCancelledError();
  }
}

function isHistoricalBacktestPreset(
  preset: BacktestPreset,
): preset is Extract<
  BacktestPreset,
  "last-x" | "week" | "month" | "year" | "random-windows" | "random-length-windows"
> {
  return (
    preset === "last-x" ||
    preset === "week" ||
    preset === "month" ||
    preset === "year" ||
    preset === "random-windows" ||
    preset === "random-length-windows"
  );
}

function compactPublicBotState(state: Readonly<PaperBotState>): PaperBotState {
  return {
    ...state,
    orders: compactPublicOrders(state.orders).map((order) => ({ ...order })),
    fills: state.fills.slice(-PUBLIC_FILL_LIMIT).map((fill) => ({ ...fill })),
    memory: compactPublicMemory(state.memory, state.config),
    metrics: { ...state.metrics },
    config: structuredClone(state.config),
  };
}

function compactPublicEvents(events: readonly BotEvent[]): BotEvent[] {
  return events.map(compactPublicEvent);
}

function compactPublicEvent(event: BotEvent): BotEvent {
  return {
    type: event.type,
    at: event.at,
    message: event.message,
    order: event.order ? { ...event.order } : undefined,
    fill: event.fill ? { ...event.fill } : undefined,
  };
}

function compactPublicMemory(
  memory: Readonly<StrategyMemory>,
  config: Readonly<StrategyConfig>,
): StrategyMemory {
  return {
    prices: memory.prices
      .slice(
        -Math.max(
          PUBLIC_PRICE_MEMORY_LIMIT,
          config.legacyValleyPeak.averagingRangesSec.length * 100,
        ),
      )
      .slice(),
    lastSignal: memory.lastSignal,
    lastActionAt: memory.lastActionAt,
  };
}

function compactPublicOrders(orders: readonly TradingOrder[]): readonly TradingOrder[] {
  const recentStart = Math.max(0, orders.length - PUBLIC_ORDER_LIMIT);
  const openBeforeRecent = orders
    .slice(0, recentStart)
    .filter((order) => order.status === "open");

  return [...openBeforeRecent, ...orders.slice(recentStart)];
}

function createIdleBacktest(): BacktestProgressSnapshot {
  const now = Date.now();
  return {
    id: "idle",
    preset: "saved-candles",
    status: "idle",
    source: "candles",
    startedAt: now,
    updatedAt: now,
    targetStartTime: 0,
    targetEndTime: 0,
    processedCandles: 0,
    estimatedCandles: 0,
    requests: 0,
    percent: 0,
    equity: 0,
    returnPct: 0,
    message: "No backtest has run yet",
  };
}

function buildCompletedProgress(
  progress: BacktestProgressSnapshot,
  result: BacktestResult,
  message: string,
): BacktestProgressSnapshot {
  const processedCandles =
    result.summary.candlesProcessed ??
    (result.summary.source === "candles"
      ? Math.round(result.summary.eventsProcessed / 4)
      : result.summary.eventsProcessed);

  return {
    ...progress,
    status: "completed",
    updatedAt: Date.now(),
    targetStartTime: result.summary.targetStartTime ?? result.summary.startTime,
    targetEndTime: result.summary.targetEndTime ?? result.summary.endTime,
    processedStartTime: result.summary.startTime,
    processedEndTime: result.summary.endTime,
    processedCandles,
    estimatedCandles: progress.estimatedCandles || processedCandles,
    requests: result.summary.requests ?? progress.requests,
    cacheHitCandles: result.summary.cacheHitCandles ?? progress.cacheHitCandles,
    cacheMissCandles: result.summary.cacheMissCandles ?? progress.cacheMissCandles,
    cacheFetchedCandles: result.summary.cacheFetchedCandles ?? progress.cacheFetchedCandles,
    cacheSizeBytes: result.summary.cacheSizeBytes ?? progress.cacheSizeBytes,
    cacheEvictedBytes: result.summary.cacheEvictedBytes ?? progress.cacheEvictedBytes,
    cacheEvictedFiles: result.summary.cacheEvictedFiles ?? progress.cacheEvictedFiles,
    percent: result.summary.stoppedEarly ? progress.percent : 100,
    equity: result.summary.finalEquity,
    returnPct: result.summary.returnPct,
    stopReason: result.summary.stopReason ?? "completed",
    survivedMs: result.summary.survivedMs,
    message,
    result,
  };
}

function upsertCandle(candles: Candle[], candle: Candle, maxCandles: number): void {
  const existingIndex = candles.findIndex(
    (item) => item.openTime === candle.openTime && item.interval === candle.interval,
  );

  if (existingIndex >= 0) {
    candles[existingIndex] = candle;
  } else {
    candles.push(candle);
  }

  candles.sort((a, b) => a.openTime - b.openTime);

  if (candles.length > maxCandles) {
    candles.splice(0, candles.length - maxCandles);
  }
}
