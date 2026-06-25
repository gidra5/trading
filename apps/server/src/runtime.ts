import {
  SimulatedTradingBot,
  analyzePositions,
  createStrategyConfig,
  legacyValleyPeakHistoricalWarmupSec,
  legacyValleyPeakObservedWarmupSec,
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
import { HistoricalCandleCache } from "./historical-candle-cache.js";
import type { TradingStorage } from "./storage.js";

const PUBLIC_ORDER_LIMIT = 500;
const PUBLIC_FILL_LIMIT = 500;
const PUBLIC_PRICE_MEMORY_LIMIT = 100;
const BALANCE_EPSILON = 0.000001;
const ASSET_DRIFT_TOLERANCE_RATE = 0.0001;
const MIN_ASSET_DRIFT_TOLERANCE = 0.00000001;
const QUOTE_DRIFT_TOLERANCE_RATE = 0.001;
const MIN_QUOTE_DRIFT_TOLERANCE = 2;

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

interface ExchangeAccountGuardOptions {
  hardStop: boolean;
  onWarning?: (message: string) => void;
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
  private exchangeAccountWarningMessage?: string;
  private historicalWarmupGeneration = 0;
  private historicalWarmupPromise?: Promise<void>;

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
    private readonly exchangeAccountGuard: ExchangeAccountGuardOptions = {
      hardStop: false,
    },
  ) {}

  async init(): Promise<void> {
    await this.storage.ensureReady();
    this.candles = await this.storage.loadCandles(500);
    const savedState = await this.storage.loadBotState();
    const exchangeSnapshot = await this.syncExecutionExchangeSnapshot();
    const exchangeStartingQuote = exchangeStartingQuoteForInitialState(
      exchangeSnapshot,
      this.market,
      savedState,
    );
    const initialState = rebaseBotStateCapital(savedState, exchangeStartingQuote);
    this.config = this.createMarketConfig(initialState ?? savedState, exchangeStartingQuote);
    this.bot = new SimulatedTradingBot(initialState, this.config);
    this.warmupBotFromRecentCandles(initialState);
    this.startHistoricalWarmup(initialState);
    await this.recoverExchangeState(exchangeSnapshot);
  }

  async switchMarket(
    market: BinanceMarketListing,
    storage: TradingStorage,
  ): Promise<void> {
    this.cancelHistoricalWarmup();
    await this.flushState();
    this.market = market;
    this.storage = storage;
    this.candles = [];
    this.orderBook = undefined;
    this.recentEvents = [];
    this.lastSavedOrderBookAt = 0;
    this.exchangeAccountWarningMessage = undefined;
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
    const exchangeSnapshot = await this.syncExecutionExchangeSnapshot();
    const exchangeStartingQuote = exchangeStartingQuoteForInitialState(
      exchangeSnapshot,
      this.market,
      savedState,
    );
    const initialState = rebaseBotStateCapital(savedState, exchangeStartingQuote);
    this.config = this.createMarketConfig(initialState ?? savedState, exchangeStartingQuote);
    this.bot = new SimulatedTradingBot(initialState, this.config);
    this.warmupBotFromRecentCandles(initialState);
    this.startHistoricalWarmup(initialState);
    await this.recoverExchangeState(exchangeSnapshot);
  }

  handleStatus(status: MarketStreamStatus): void {
    this.status = status;
  }

  async handleTick(tick: PriceTick): Promise<BotEvent[]> {
    const exchangeDriven = this.paperTrading?.drivesOrderExecution(this.market) ?? false;
    if (
      exchangeDriven &&
      this.exchangeAccountGuard.hardStop &&
      this.exchangeAccountWarningMessage
    ) {
      return [];
    }

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
      exchange: this.publicExchangeSnapshot(),
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
    const exchangeDrivesExecution = this.paperTrading?.drivesOrderExecution(this.market) ?? false;
    this.exchangeAccountWarningMessage = undefined;
    let resetSnapshot: BinancePaperSnapshot | undefined;
    if (exchangeCanSubmit) {
      resetSnapshot = await this.paperTrading!.cancelAllOpenOrders(this.market);
      if (resetSnapshot.positions.length > 0) {
        resetSnapshot = await this.paperTrading!.closeOpenPositions(this.market, {
          includeUnprofitable: true,
        });
      }
    }

    const exchangeStartingQuote = exchangeDrivesExecution
      ? exchangeQuoteBalance(resetSnapshot, this.market)
      : undefined;
    if (exchangeStartingQuote !== undefined) {
      this.config = this.createMarketConfig(undefined, exchangeStartingQuote);
    }

    const events = this.bot.reset(this.config);
    this.recordEvents(events);
    await this.flushState();
    if (resetSnapshot) {
      await this.applyExchangeSnapshot(resetSnapshot);
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
        const reason = exchangeSubmitFailureReason(error);
        const cancelled = this.bot.cancelOpenOrder(event.order.id, reason);
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

  private async recoverExchangeState(snapshot?: BinancePaperSnapshot): Promise<void> {
    if (!this.paperTrading?.drivesOrderExecution(this.market)) {
      return;
    }
    try {
      await this.applyExchangeSnapshot(snapshot ?? await this.paperTrading.sync(this.market));
    } catch {
      return;
    }
  }

  private async syncExecutionExchangeSnapshot(): Promise<BinancePaperSnapshot | undefined> {
    if (!this.paperTrading?.drivesOrderExecution(this.market)) {
      return undefined;
    }
    try {
      return await this.paperTrading.sync(this.market);
    } catch {
      return undefined;
    }
  }

  private warmupBotFromRecentCandles(savedState?: PaperBotState): void {
    if (this.hasUsableLegacyWarmup(savedState)) {
      return;
    }

    const candles = this.candles.filter((candle) => candle.closed);
    if (candles.length === 0) {
      return;
    }

    const processed = this.bot.warmupFromCandles(candles);
    if (processed > 0) {
      for (const candle of candles.slice(-500)) {
        upsertCandle(this.candles, candle, 500);
      }
    }
  }

  private startHistoricalWarmup(savedState?: PaperBotState): void {
    const market = this.market;
    const config = this.config;
    if (!this.needsHistoricalWarmup(savedState, config) || !isHistoricalVenue(market.venue)) {
      return;
    }

    const generation = ++this.historicalWarmupGeneration;
    const endpoint = this.paperTrading?.klineEndpointFor(market);
    let task: Promise<void>;
    task = this.warmupBotFromHistory({
      config,
      endpoint,
      generation,
      market,
    })
      .catch(() => undefined)
      .finally(() => {
        if (this.historicalWarmupPromise === task) {
          this.historicalWarmupPromise = undefined;
        }
      });
    this.historicalWarmupPromise = task;
  }

  private cancelHistoricalWarmup(): void {
    this.historicalWarmupGeneration += 1;
    this.historicalWarmupPromise = undefined;
  }

  private async warmupBotFromHistory(options: {
    config: StrategyConfig;
    endpoint?: string;
    generation: number;
    market: BinanceMarketListing;
  }): Promise<void> {
    const candles = await this.loadWarmupCandles(
      options.market,
      options.config,
      options.endpoint,
    );
    if (
      options.generation !== this.historicalWarmupGeneration ||
      options.market.id !== this.market.id ||
      candles.length === 0
    ) {
      return;
    }

    const mergedCandles = mergeWarmupCandles(
      candles,
      this.candles.filter((candle) => candle.closed),
    );
    const processed = this.bot.warmupFromCandles(mergedCandles);
    if (processed <= 0) {
      return;
    }
    for (const candle of mergedCandles.slice(-500)) {
      upsertCandle(this.candles, candle, 500);
    }
    await this.flushState();
  }

  private needsHistoricalWarmup(
    savedState: PaperBotState | undefined,
    config: StrategyConfig,
  ): boolean {
    if (!this.hasUsableLegacyWarmup(savedState)) {
      return true;
    }

    const requiredWarmupSec = legacyValleyPeakHistoricalWarmupSec(
      config.legacyValleyPeak,
    );
    const observedWarmupSec = legacyValleyPeakObservedWarmupSec(
      savedState!.memory.legacyValleyPeak,
    );
    return observedWarmupSec < requiredWarmupSec * 0.95;
  }

  private hasUsableLegacyWarmup(savedState?: PaperBotState): boolean {
    const memory = savedState?.memory.legacyValleyPeak;
    return (
      !!memory &&
      Array.isArray(memory.buyAverages) &&
      memory.buyAverages.some((average) => average.timestamps.length > 0)
    );
  }

  private async loadWarmupCandles(
    market: BinanceMarketListing,
    config: StrategyConfig,
    endpoint?: string,
  ): Promise<Candle[]> {
    const intervalMs = intervalToMs(this.interval);
    const warmupMs = legacyValleyPeakHistoricalWarmupSec(config.legacyValleyPeak) * 1000;
    const requiredCandles = Math.max(10, Math.ceil(warmupMs / intervalMs) + 5);
    const storedCandles =
      market.id === this.market.id
        ? this.candles.filter((candle) => candle.closed).slice(-requiredCandles)
        : [];
    const storedSpan =
      storedCandles.length > 0
        ? storedCandles.at(-1)!.closeTime - storedCandles[0].openTime
        : 0;
    if (storedSpan >= warmupMs) {
      return storedCandles;
    }

    const endTime = Date.now();
    const startTime = endTime - requiredCandles * intervalMs;
    if (!isHistoricalVenue(market.venue)) {
      return storedCandles;
    }
    const venue = market.venue;

    try {
      const cache = new HistoricalCandleCache({
        dataDir: this.historicalCache.dataDir,
        marketKey: market.id,
        symbol: market.symbol,
        interval: this.interval,
        intervalMs,
        maxBytes: this.historicalCache.maxBytes,
        minFreeBytes: this.historicalCache.minFreeBytes,
      });

      await cache.ensureRange(startTime, endTime, (request) =>
        fetchKlines({
          venue,
          symbol: market.symbol,
          interval: this.interval,
          startTime: request.startTime,
          endTime: request.endTime,
          limit: request.limit,
          endpoint,
        }),
      );

      const candles: Candle[] = [];
      for await (const batch of cache.readRangeBatches(startTime, endTime, 5000)) {
        candles.push(...batch.filter((candle) => candle.closed));
      }

      return candles.length > 0 ? candles : storedCandles;
    } catch {
      return storedCandles;
    }
  }

  private async applyExchangeSnapshot(snapshot: BinancePaperSnapshot): Promise<BotEvent[]> {
    this.applyExchangeMaxLeverage(snapshot);
    this.applyExchangeTradingRules(snapshot);
    const reconciliation = exchangeReconciliationFromSnapshot(snapshot);
    const events: BotEvent[] = [];
    if (
      (reconciliation.orders?.length ?? 0) > 0 ||
      (reconciliation.fills?.length ?? 0) > 0
    ) {
      events.push(...this.bot.applyExchangeReconciliation(reconciliation));
    }

    events.push(...this.applyExchangeAccountGuard(snapshot));
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  private applyExchangeAccountGuard(snapshot: BinancePaperSnapshot): BotEvent[] {
    if (!this.paperTrading?.drivesOrderExecution(this.market)) {
      this.exchangeAccountWarningMessage = undefined;
      return [];
    }

    const driftMessage = exchangeAccountDriftMessage(
      snapshot,
      this.market,
      this.bot.snapshot(),
    );
    if (!driftMessage) {
      this.exchangeAccountWarningMessage = undefined;
      return [];
    }

    const message = this.exchangeAccountGuard.hardStop
      ? `${driftMessage} Bot paused; reset or manually reconcile the exchange account before restarting.`
      : `${driftMessage} Exchange account guard hard stop is disabled; recording warning only.`;
    const wasAlreadyReported = this.exchangeAccountWarningMessage === message;
    this.exchangeAccountWarningMessage = message;
    if (!wasAlreadyReported) {
      this.exchangeAccountGuard.onWarning?.(message);
    }

    if (!this.exchangeAccountGuard.hardStop) {
      return [];
    }

    if (wasAlreadyReported && this.bot.view().status === "stopped") {
      return [];
    }

    const at = Date.now();
    const events = this.bot.setStatus("stopped", at);
    if (events.length > 0) {
      return events.map((event) => ({
        ...event,
        message,
        state: this.bot.snapshot(),
      }));
    }

    return [
      {
        type: "status_changed",
        at,
        message,
        state: this.bot.snapshot(),
      },
    ];
  }

  private publicExchangeSnapshot(): BinancePaperSnapshot {
    const snapshot = this.paperTrading?.snapshot(this.market) ?? disabledExchangeSnapshot();
    if (!this.exchangeAccountWarningMessage) {
      return snapshot;
    }

    return {
      ...snapshot,
      message: this.exchangeAccountWarningMessage,
      userDataStreamMessage: this.exchangeAccountWarningMessage,
      ...(this.exchangeAccountGuard.hardStop
        ? { error: this.exchangeAccountWarningMessage }
        : {}),
    };
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

  private createMarketConfig(
    savedState?: PaperBotState,
    startingQuote?: number,
  ): StrategyConfig {
    const config = createStrategyConfig({
      ...(savedState?.config ?? this.config),
      symbol: this.market.symbol,
      baseAsset: this.market.baseAsset,
      quoteAsset: this.market.quoteAsset,
      ...(startingQuote !== undefined ? { startingQuote } : {}),
    });
    return applyMarketMaxLeverage(config, this.market.maxLeverage);
  }
}

function exchangeStartingQuoteForInitialState(
  snapshot: BinancePaperSnapshot | undefined,
  market: BinanceMarketListing,
  savedState: PaperBotState | undefined,
): number | undefined {
  if (!isRebasableBotState(savedState) || hasExchangeMarketExposure(snapshot, market)) {
    return undefined;
  }
  return exchangeQuoteBalance(snapshot, market);
}

function rebaseBotStateCapital(
  savedState: PaperBotState | undefined,
  startingQuote: number | undefined,
): PaperBotState | undefined {
  if (!savedState || startingQuote === undefined) {
    return savedState;
  }

  const next = structuredClone(savedState);
  next.startingQuote = startingQuote;
  next.quoteFree = startingQuote;
  next.quoteReserved = 0;
  next.baseFree = 0;
  next.baseReserved = 0;
  next.avgEntryPrice = 0;
  next.avgShortEntryPrice = 0;
  next.realizedPnl = 0;
  next.feesPaid = 0;
  next.winningTrades = 0;
  next.losingTrades = 0;
  next.exitGridSpanTotal = 0;
  next.exitGridSpanCount = 0;
  next.exitGridOrderCountTotal = 0;
  next.config = createStrategyConfig({
    ...next.config,
    startingQuote,
  });
  return next;
}

function isRebasableBotState(state: PaperBotState | undefined): boolean {
  if (!state) {
    return true;
  }
  return (
    state.orders.length === 0 &&
    state.fills.length === 0 &&
    Math.abs(state.quoteReserved) <= BALANCE_EPSILON &&
    Math.abs(state.baseFree) <= BALANCE_EPSILON &&
    Math.abs(state.baseReserved) <= BALANCE_EPSILON
  );
}

function hasExchangeMarketExposure(
  snapshot: BinancePaperSnapshot | undefined,
  market: BinanceMarketListing,
): boolean {
  return Boolean(
    snapshot &&
      (snapshot.openOrders.length > 0 ||
        snapshot.positions.length > 0 ||
        (exchangeAssetBalance(snapshot, market.baseAsset) ?? 0) > BALANCE_EPSILON),
  );
}

function exchangeQuoteBalance(
  snapshot: BinancePaperSnapshot | undefined,
  market: BinanceMarketListing,
): number | undefined {
  return exchangeAssetBalance(snapshot, market.quoteAsset);
}

function exchangeAssetBalance(
  snapshot: BinancePaperSnapshot | undefined,
  asset: string,
): number | undefined {
  const targetAsset = asset.toUpperCase();
  const balance = snapshot?.balances.find((item) => item.asset.toUpperCase() === targetAsset);
  if (!balance) {
    return undefined;
  }

  return firstPositiveNumber(
    balance.availableBalance,
    balance.free,
    balance.walletBalance,
    balance.free + balance.locked,
  );
}

function firstPositiveNumber(...values: Array<number | undefined>): number | undefined {
  for (const value of values) {
    if (value !== undefined && Number.isFinite(value) && value > BALANCE_EPSILON) {
      return Number(value.toFixed(6));
    }
  }
  return undefined;
}

function exchangeAccountDriftMessage(
  snapshot: BinancePaperSnapshot,
  market: BinanceMarketListing,
  state: PaperBotState,
): string | undefined {
  const localOrderIds = new Set(state.orders.map((order) => order.id));
  const unmanagedOrder = snapshot.openOrders.find((order) => !order.localOrderId);
  if (unmanagedOrder) {
    return `Exchange has unmanaged open ${market.symbol} order ${unmanagedOrder.orderId}.`;
  }

  const unknownBotOrder = snapshot.openOrders.find(
    (order) => order.localOrderId && !localOrderIds.has(order.localOrderId),
  );
  if (unknownBotOrder) {
    return `Exchange open order ${unknownBotOrder.orderId} points to missing local order ${unknownBotOrder.localOrderId}.`;
  }

  if (market.venue === "spot") {
    return spotAccountDriftMessage(snapshot, market, state);
  }
  if (market.venue === "usdm-futures" || market.venue === "coinm-futures") {
    return futuresAccountDriftMessage(snapshot, market, state);
  }

  return undefined;
}

function spotAccountDriftMessage(
  snapshot: BinancePaperSnapshot,
  market: BinanceMarketListing,
  state: PaperBotState,
): string | undefined {
  const exchangeQuote = exchangeSpotAssetTotal(snapshot, market.quoteAsset);
  const localQuote = roundQuoteBalance(state.quoteFree + state.quoteReserved);
  if (hasQuoteDrift(localQuote, exchangeQuote)) {
    return `${market.quoteAsset} balance drift: local ${localQuote}, exchange ${exchangeQuote}.`;
  }

  const exchangeBase = exchangeSpotAssetTotal(snapshot, market.baseAsset);
  const localBase = roundAssetBalance(state.baseFree + state.baseReserved);
  if (hasAssetDrift(localBase, exchangeBase)) {
    return `${market.baseAsset} balance drift: local ${localBase}, exchange ${exchangeBase}.`;
  }

  return undefined;
}

function futuresAccountDriftMessage(
  snapshot: BinancePaperSnapshot,
  market: BinanceMarketListing,
  state: PaperBotState,
): string | undefined {
  const positionProfile = futuresPositionProfile(snapshot);
  if (
    positionProfile.longQuantity > BALANCE_EPSILON &&
    positionProfile.shortQuantity > BALANCE_EPSILON
  ) {
    return "Exchange account has simultaneous long and short hedge-mode positions that the local net-position guard cannot safely reconcile.";
  }

  const localBase = roundAssetBalance(state.baseFree + state.baseReserved);
  const exchangeBase = roundAssetBalance(positionProfile.netQuantity);
  if (hasAssetDrift(localBase, exchangeBase)) {
    return `${market.baseAsset} futures position drift: local ${localBase}, exchange ${exchangeBase}.`;
  }

  const markPrice =
    positionProfile.markPrice ||
    state.lastPrice ||
    state.avgEntryPrice ||
    state.avgShortEntryPrice;
  const exchangeEquity = exchangeFuturesEquity(snapshot, market.quoteAsset);
  if (exchangeEquity !== undefined && markPrice > 0) {
    const localEquity = roundQuoteBalance(
      state.quoteFree + state.quoteReserved + localBase * markPrice,
    );
    if (hasQuoteDrift(localEquity, exchangeEquity)) {
      return `${market.quoteAsset} equity drift: local ${localEquity}, exchange ${exchangeEquity}.`;
    }
  }

  const exchangeEntryPrice = positionProfile.entryPrice;
  if (exchangeEntryPrice && Math.abs(exchangeBase) > MIN_ASSET_DRIFT_TOLERANCE) {
    const localEntryPrice = exchangeBase > 0 ? state.avgEntryPrice : state.avgShortEntryPrice;
    if (localEntryPrice > 0 && hasQuoteDrift(localEntryPrice, exchangeEntryPrice)) {
      return `Futures entry price drift: local ${roundQuoteBalance(localEntryPrice)}, exchange ${roundQuoteBalance(exchangeEntryPrice)}.`;
    }
  }

  return undefined;
}

function exchangeSpotAssetTotal(snapshot: BinancePaperSnapshot, asset: string): number {
  const targetAsset = asset.toUpperCase();
  const balance = snapshot.balances.find((item) => item.asset.toUpperCase() === targetAsset);
  if (!balance) {
    return 0;
  }
  return roundAssetBalance(balance.free + balance.locked);
}

function exchangeFuturesEquity(
  snapshot: BinancePaperSnapshot,
  quoteAsset: string,
): number | undefined {
  const targetAsset = quoteAsset.toUpperCase();
  const balance = snapshot.balances.find((item) => item.asset.toUpperCase() === targetAsset);
  if (!balance) {
    return undefined;
  }

  const walletBalance = firstFiniteNumber(
    balance.walletBalance,
    balance.free + balance.locked,
    balance.availableBalance,
    balance.free,
  );
  if (walletBalance === undefined) {
    return undefined;
  }

  const unrealizedPnl = snapshot.positions.reduce(
    (sum, position) => sum + finiteOrZero(position.unrealizedPnl),
    0,
  );
  return roundQuoteBalance(walletBalance + unrealizedPnl);
}

function futuresPositionProfile(snapshot: BinancePaperSnapshot): {
  netQuantity: number;
  longQuantity: number;
  shortQuantity: number;
  markPrice: number;
  entryPrice?: number;
} {
  let netQuantity = 0;
  let longQuantity = 0;
  let shortQuantity = 0;
  let markNotional = 0;
  let entryNotional = 0;
  let absoluteQuantity = 0;

  for (const position of snapshot.positions) {
    const quantity = finiteOrZero(position.positionAmt);
    if (quantity > 0) {
      longQuantity += quantity;
    } else if (quantity < 0) {
      shortQuantity += -quantity;
    }
    netQuantity += quantity;

    const absQuantity = Math.abs(quantity);
    if (absQuantity <= 0) {
      continue;
    }
    absoluteQuantity += absQuantity;
    const markPrice = finiteOrZero(position.markPrice);
    const entryPrice = finiteOrZero(position.entryPrice);
    if (markPrice > 0) {
      markNotional += markPrice * absQuantity;
    }
    if (entryPrice > 0) {
      entryNotional += entryPrice * absQuantity;
    }
  }

  return {
    netQuantity: roundAssetBalance(netQuantity),
    longQuantity: roundAssetBalance(longQuantity),
    shortQuantity: roundAssetBalance(shortQuantity),
    markPrice: absoluteQuantity > 0 ? markNotional / absoluteQuantity : 0,
    entryPrice: absoluteQuantity > 0 && entryNotional > 0
      ? entryNotional / absoluteQuantity
      : undefined,
  };
}

function hasQuoteDrift(localValue: number, exchangeValue: number): boolean {
  return Math.abs(localValue - exchangeValue) > quoteDriftTolerance(exchangeValue);
}

function hasAssetDrift(localValue: number, exchangeValue: number): boolean {
  return Math.abs(localValue - exchangeValue) > assetDriftTolerance(exchangeValue);
}

function quoteDriftTolerance(reference: number): number {
  return Math.max(MIN_QUOTE_DRIFT_TOLERANCE, Math.abs(reference) * QUOTE_DRIFT_TOLERANCE_RATE);
}

function assetDriftTolerance(reference: number): number {
  return Math.max(MIN_ASSET_DRIFT_TOLERANCE, Math.abs(reference) * ASSET_DRIFT_TOLERANCE_RATE);
}

function firstFiniteNumber(...values: Array<number | undefined>): number | undefined {
  for (const value of values) {
    if (value !== undefined && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function finiteOrZero(value: number | undefined): number {
  return value !== undefined && Number.isFinite(value) ? value : 0;
}

function roundQuoteBalance(value: number): number {
  return Number(value.toFixed(6));
}

function roundAssetBalance(value: number): number {
  return Number(value.toFixed(8));
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

function exchangeSubmitFailureReason(error: unknown): string {
  const message = error instanceof Error ? error.message.trim() : "";
  return message ? `exchange submit failed: ${message}` : "exchange submit failed";
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
    lastExtremaSignal: memory.lastExtremaSignal,
    lastExtremaSignalAt: memory.lastExtremaSignalAt,
    lastExtremaSignalPrice: memory.lastExtremaSignalPrice,
    lastExtremaSignalReason: memory.lastExtremaSignalReason,
    legacyValleyPeakDebug: memory.legacyValleyPeakDebug
      ? structuredClone(memory.legacyValleyPeakDebug)
      : undefined,
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

function mergeWarmupCandles(
  historicalCandles: readonly Candle[],
  recentCandles: readonly Candle[],
): Candle[] {
  const byOpenTime = new Map<number, Candle>();
  for (const candle of historicalCandles) {
    if (candle.closed) {
      byOpenTime.set(candle.openTime, candle);
    }
  }
  for (const candle of recentCandles) {
    if (candle.closed) {
      byOpenTime.set(candle.openTime, candle);
    }
  }
  return [...byOpenTime.values()].sort((left, right) => left.openTime - right.openTime);
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
