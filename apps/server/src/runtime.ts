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
  type ManualTradeInput,
  type OrderBookSnapshot,
  type PaperBotState,
  type PriceTick,
  type PositionLedger,
  type StrategyMemory,
  type StrategyConfig,
  type TradingOrder,
} from "@trading/bot-algo";
import type { MarketStreamStatus } from "./binance-stream.js";
import type { BinanceMarketListing, StreamVenue } from "./binance-markets.js";
import { runHistoricalCandleBacktest } from "./historical-backtest.js";
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
  ) {}

  async init(): Promise<void> {
    await this.storage.ensureReady();
    this.candles = await this.storage.loadCandles(500);
    const savedState = await this.storage.loadBotState();
    this.config = this.createMarketConfig(savedState);
    this.bot = new SimulatedTradingBot(savedState, this.config);
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
  }

  handleStatus(status: MarketStreamStatus): void {
    this.status = status;
  }

  async handleTick(tick: PriceTick): Promise<BotEvent[]> {
    const events = this.bot.onTick(tick);
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
        connected: this.status.connected,
        statusMessage: this.status.message,
        lastEventAt: this.status.lastEventAt,
        lastPrice: bot.lastPrice,
        candles: this.candles,
        orderBook: this.orderBook,
      },
      bot: publicBot,
      positions: analyzePositions(bot as PaperBotState),
      recentEvents: this.recentEvents,
      backtest: this.backtest,
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
    const events = this.bot.reset(this.config);
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  async updateBotConfig(patch: Partial<StrategyConfig>): Promise<BotEvent[]> {
    this.config = createStrategyConfig({
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
    });
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
    });
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  startBacktest(
    options: BacktestStartOptions,
    onUpdate: () => void,
  ): BacktestProgressSnapshot {
    if (this.backtest.status === "running") {
      throw new Error("A backtest is already running.");
    }

    const id = `bt_${Date.now()}`;
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
      percent: 0,
      equity: this.config.startingQuote,
      returnPct: 0,
      message: "Starting backtest",
    };
    onUpdate();

    void this.executeBacktest(id, options, onUpdate);
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

    this.recentEvents = [...events, ...this.recentEvents].slice(0, 60);
  }

  private async executeBacktest(
    id: string,
    options: BacktestStartOptions,
    onUpdate: () => void,
  ): Promise<void> {
    try {
      const result = await this.runBacktestNow(id, options, onUpdate);
      await this.storage.saveBacktest(result);
      this.backtest = buildCompletedProgress(
        this.backtest,
        result,
        result.summary.stoppedEarly
          ? "Stopped early after portfolio wipeout"
          : "Backtest completed",
      );
      onUpdate();
    } catch (error) {
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
    }
  }

  private async runBacktestNow(
    id: string,
    options: BacktestStartOptions,
    onUpdate: () => void,
  ): Promise<BacktestResult> {
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
          marketKey: this.market.id,
          venue,
          symbol: this.config.symbol,
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
        },
        (progress) => {
          this.backtest = progress;
          onUpdate();
        },
      );
    }

    const startedAt = Date.now();
    const result =
      options.preset === "saved-orderbook"
        ? runBacktestFromOrderBook(await this.storage.loadOrderBookSnapshots(options.limit), {
            config,
          })
        : runBacktestFromCandles(await this.storage.loadCandles(options.limit), {
            config,
          });
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
    return createStrategyConfig({
      ...(savedState?.config ?? this.config),
      symbol: this.market.symbol,
      baseAsset: this.market.baseAsset,
      quoteAsset: this.market.quoteAsset,
    });
  }
}

function isHistoricalVenue(venue: string): venue is StreamVenue {
  return (
    venue === "spot" ||
    venue === "usdm-futures" ||
    venue === "coinm-futures" ||
    venue === "options"
  );
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

function compactPublicMemory(
  memory: Readonly<StrategyMemory>,
  config: Readonly<StrategyConfig>,
): StrategyMemory {
  return {
    prices: memory.prices
      .slice(-Math.max(PUBLIC_PRICE_MEMORY_LIMIT, config.slowWindow * 2))
      .slice(),
    lastSignal: memory.lastSignal,
    lastActionAt: memory.lastActionAt,
    previousFastAvg: memory.previousFastAvg,
    previousSlowAvg: memory.previousSlowAvg,
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
