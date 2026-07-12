import {
  GridTradingBot,
  PeakValleyStrategy,
  createPeakValleyBotConfig,
  createPeakValleyStrategyConfig,
  rescalePeakValleyStrategyConfig,
  createStrategyConfig,
  runBacktestFromOrderBook,
  type BacktestPreset,
  type BacktestProgressSnapshot,
  type BacktestResult,
  type BotDiagnostics,
  type BotEntryRiskReport,
  type BotMetricsSnapshot,
  type BotSnapshot,
  type Candle,
  type EquityPoint,
  type ManualTradeInput,
  type OrderBookSnapshot,
  type PartialStrategyConfig,
  type PeakValleyBotConfig,
  type PeakValleyStrategyDiagnostics,
  type PeakValleyStrategySnapshot,
  type PriceTick,
  type StrategyConfig,
  type TradingCandle,
  type TradingOrderEvent,
  type TradingOrderSnapshot,
  type TradingTick,
} from "@trading/bot-algo";
import type { MarketStreamStatus } from "./binance-stream.js";
import { runBotBacktestFromCandles } from "./bot-backtest.js";
import type { BinanceMarketListing, StreamVenue } from "./binance-markets.js";
import {
  BacktestCancelledError,
  intervalToMs,
  isBacktestCancelledError,
  runHistoricalCandleBacktest,
  type HistoricalBacktestMarket,
} from "./historical-backtest.js";
import {
  BinanceExchangeTrading,
  type BinanceExchangeCancelOrderInput,
  type BinanceExchangePlaceOrderInput,
  type BinanceExchangeSnapshot,
} from "./binance-exchange.js";
import {
  BinanceTradingApi,
  type BinanceTradingApiSnapshot,
} from "./trading-api/binance-api.js";
import { BinanceExchangeClient } from "./trading-api/binance-client.js";
import {
  PaperTradingApi,
  type PaperTradingSnapshot,
} from "./trading-api/paper-api.js";
import type { EquitySnapshot, RuntimeTradingApi } from "./trading-api/runtime-api.js";
import type {
  TradingExchangeCredentials,
  TradingExchangeMode,
  TradingExecutionMode,
  TradingRuntimeSettings,
  TradingStorage,
} from "./storage.js";

interface BacktestStartOptions {
  preset: BacktestPreset;
  limit: number;
  historicalStartTime?: number;
  startingQuote?: number;
  historicalDays?: number;
  randomSampleCount?: number;
  randomWindowDays?: number;
  randomMinWindowDays?: number;
  randomMaxWindowDays?: number;
  randomLookbackDays?: number;
  randomPairCount?: number;
  randomMarkets?: HistoricalBacktestMarket[];
  extremaSmaWindowMinutes?: number;
}

type HistoricalPreset = Extract<
  BacktestPreset,
  "last-x" | "week" | "month" | "year" | "random-windows" | "random-length-windows"
>;

interface StoredRuntimeState {
  version: 1;
  status: "running" | "stopped";
  runStartedAt: number;
  bot: BotSnapshot<PeakValleyBotConfig["strategy"], PeakValleyStrategySnapshot>;
  paper?: PaperTradingSnapshot;
  binance?: BinanceTradingApiSnapshot;
}

export interface RuntimeBotEvent {
  type: "open" | "partial-fill" | "fill" | "rejected" | "status" | "reset";
  at: number;
  message: string;
  order?: TradingOrderSnapshot;
  orderId?: string;
  fill?: Extract<TradingOrderEvent, { type: "fill" | "partial-fill" }>["fill"];
}

export interface RuntimeBotSnapshot {
  status: "running" | "stopped";
  runStartedAt: number;
  config: PeakValleyBotConfig;
  state: BotSnapshot<PeakValleyBotConfig["strategy"], PeakValleyStrategySnapshot>;
  metrics: BotMetricsSnapshot & {
    equity: number;
    netPnl: number;
    returnPct: number;
  };
  diagnostics: BotDiagnostics<PeakValleyStrategyDiagnostics> & { entryRisk: readonly BotEntryRiskReport[] };
  equity: EquitySnapshot;
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
  bot: RuntimeBotSnapshot;
  recentEvents: RuntimeBotEvent[];
  backtest: BacktestProgressSnapshot;
  equityCurve: EquityPoint[];
  execution: {
    mode: TradingExecutionMode;
    exchangeDriven: boolean;
    canUseExchange: boolean;
    live: boolean;
    message: string;
  };
  exchange: BinanceExchangeSnapshot;
}

export class TradingRuntime {
  private api!: RuntimeTradingApi;
  private bot!: GridTradingBot<
    PeakValleyBotConfig["strategy"],
    PeakValleyStrategySnapshot,
    PeakValleyStrategyDiagnostics
  >;
  private botState!: RuntimeBotSnapshot["state"];
  private readonly entryRisk = new Map<BotEntryRiskReport["side"], BotEntryRiskReport>();
  private botConfig: PeakValleyBotConfig;
  private equity: EquitySnapshot = emptyEquity();
  private status: "running" | "stopped" = "running";
  private runStartedAt = Date.now();
  private candles: Candle[] = [];
  private orderBook?: OrderBookSnapshot;
  private lastTickPrice = 0;
  private streamStatus: MarketStreamStatus = {
    connected: false,
    message: "Starting",
    lastEventAt: Date.now(),
    reconnectAttempt: 0,
  };
  private recentEvents: RuntimeBotEvent[] = [];
  private backtest = createIdleBacktest();
  private backtestAbort?: AbortController;
  private liveEquityCurve: EquityPoint[] = [];
  private lastOrderBookSaveAt = 0;
  private lastEquityAt = 0;
  private saveTimer?: NodeJS.Timeout;
  private saveQueue: Promise<void> = Promise.resolve();
  private operationQueue: Promise<void> = Promise.resolve();
  private runtimeSettings: TradingRuntimeSettings = {
    executionMode: "simulated",
    updatedAt: 0,
  };
  private executionMode: TradingExecutionMode = "simulated";

  constructor(
    private storage: TradingStorage,
    private market: BinanceMarketListing,
    private legacyConfig: StrategyConfig,
    private readonly interval: string,
    private readonly historicalCache: {
      dataDir: string;
      maxBytes: number;
      minFreeBytes: number;
    },
    private readonly exchangeTrading?: BinanceExchangeTrading,
    _exchangeAccountGuard?: { hardStop: boolean; onWarning?: (message: string) => void },
  ) {
    this.botConfig = createPeakValleyBotConfig(legacyConfig, intervalToMs(interval));
  }

  async init(): Promise<void> {
    await this.storage.ensureReady();
    this.candles = await this.storage.loadCandles(500);
    await this.loadSettings();
    await this.createBot(await this.storage.loadTradingState<StoredRuntimeState>());
  }

  async switchMarket(market: BinanceMarketListing, storage: TradingStorage): Promise<void> {
    await this.flushState();
    this.market = market;
    this.storage = storage;
    this.legacyConfig = createStrategyConfig({
      ...this.legacyConfig,
      symbol: market.symbol,
      baseAsset: market.baseAsset,
      quoteAsset: market.quoteAsset,
      maxLeverage: Math.min(this.legacyConfig.maxLeverage, market.maxLeverage ?? Infinity),
    });
    this.botConfig = createPeakValleyBotConfig(this.legacyConfig, intervalToMs(this.interval));
    this.candles = [];
    this.orderBook = undefined;
    this.recentEvents = [];
    this.liveEquityCurve = [];
    this.backtestAbort?.abort();
    this.backtest = createIdleBacktest();
    await this.storage.ensureReady();
    this.candles = await this.storage.loadCandles(500);
    await this.loadSettings();
    await this.createBot(await this.storage.loadTradingState<StoredRuntimeState>());
  }

  handleStatus(status: MarketStreamStatus): void {
    this.streamStatus = status;
  }

  async handleTick(tick: PriceTick): Promise<RuntimeBotEvent[]> {
    return this.withOperation(async () => {
      const next: TradingTick = {
        timestamp: tick.eventTime,
        price: tick.price,
        quantity: tick.quantity ?? 0,
        candle: null,
      };
      this.lastTickPrice = tick.price;
      await this.api.onTick(next);
      let events = await this.deliverOrderEvents();
      if (this.status === "running") {
        await this.bot.onTick(next);
        events = [...events, ...await this.deliverOrderEvents()];
      }
      await this.refreshState(hasFill(events));
      this.recordEquity(tick.eventTime);
      this.scheduleSave();
      return events;
    });
  }

  async handleCandle(candle: Candle): Promise<RuntimeBotEvent[]> {
    return this.withOperation(async () => {
      if (!this.rememberCandle(candle)) return [];
      if (!candle.closed) return [];
      await this.storage.appendCandle(candle);
      if (this.status === "running") {
        await this.bot.onTick({
          timestamp: candle.closeTime,
          price: candle.close,
          quantity: candle.volume,
          candle: toTradingCandle(candle),
        });
      }
      const events = await this.deliverOrderEvents();
      await this.refreshState(hasFill(events));
      this.recordEquity(candle.closeTime);
      this.scheduleSave();
      return events;
    });
  }

  private rememberCandle(candle: Candle): boolean {
    const existing = this.candles.findIndex((item) => item.openTime === candle.openTime);
    if (existing >= 0) {
      if (this.candles[existing]!.closed) return false;
      this.candles[existing] = candle;
    } else {
      if (candle.openTime < (this.candles.at(-1)?.openTime ?? Number.NEGATIVE_INFINITY)) {
        return false;
      }
      this.candles.push(candle);
      if (this.candles.length > 500) this.candles.shift();
    }
    return true;
  }

  async handleOrderBook(snapshot: OrderBookSnapshot): Promise<void> {
    this.orderBook = snapshot;
    if (snapshot.eventTime - this.lastOrderBookSaveAt >= 1_000) {
      this.lastOrderBookSaveAt = snapshot.eventTime;
      await this.storage.appendOrderBookSnapshot(snapshot);
    }
  }

  async handleExchangeUserData(_payload: unknown): Promise<RuntimeBotEvent[]> {
    if (!(this.api instanceof BinanceTradingApi) || !this.exchangeTrading) {
      return [];
    }
    const api = this.api;
    return this.withOperation(async () => {
      api.reconcile(
        this.exchangeTrading!.reconciliationFromUserDataEvent(this.market, _payload),
      );
      const events = await this.deliverOrderEvents();
      await this.refreshState(hasFill(events));
      this.scheduleSave();
      return events;
    });
  }

  snapshot(): RuntimeSnapshot {
    const lastPrice = this.lastPrice();
    const accountEquity = this.equity.quoteUnleveraged + this.equity.assetUnleveraged * lastPrice;
    const netPnl = accountEquity - this.legacyConfig.startingQuote;
    return {
      market: {
        id: this.market.id,
        group: this.market.group,
        venue: this.market.venue,
        symbol: this.market.symbol,
        displaySymbol: this.market.displaySymbol,
        baseAsset: this.market.baseAsset,
        quoteAsset: this.market.quoteAsset,
        interval: this.interval,
        maxLeverage: this.market.maxLeverage,
        connected: this.streamStatus.connected,
        statusMessage: this.streamStatus.message,
        lastEventAt: this.streamStatus.lastEventAt,
        lastPrice,
        candles: this.candles,
        orderBook: this.orderBook,
      },
      bot: {
        status: this.status,
        runStartedAt: this.runStartedAt,
        config: this.botConfig,
        state: this.botState,
        metrics: {
          ...this.bot.getMetrics(),
          equity: accountEquity,
          netPnl,
          returnPct: this.legacyConfig.startingQuote > 0
            ? netPnl / this.legacyConfig.startingQuote * 100
            : 0,
        },
        diagnostics: {
          ...this.bot.getDiagnostics(),
          entryRisk: [...this.entryRisk.values()],
        },
        equity: this.equity,
      },
      recentEvents: this.recentEvents,
      backtest: this.backtest,
      equityCurve: this.liveEquityCurve,
      execution: this.executionSnapshot(),
      exchange: this.exchangeTrading?.snapshot(this.market) ?? emptyExchange(),
    };
  }

  async startBot(): Promise<RuntimeBotEvent[]> {
    if (this.status === "running") {
      return [];
    }
    this.status = "running";
    this.runStartedAt = Date.now();
    this.liveEquityCurve = [];
    return this.recordStatus("Bot running");
  }

  async stopBot(): Promise<RuntimeBotEvent[]> {
    if (this.status === "stopped") {
      return [];
    }
    this.status = "stopped";
    return this.recordStatus("Bot stopped");
  }

  async resetBot(): Promise<RuntimeBotEvent[]> {
    return this.withOperation(async () => {
      await this.bot.cancelOpenOrders();
      if (this.executionMode === "binance" && this.exchangeTrading?.canSubmitOrders(this.market)) {
        await this.exchangeTrading.cancelAllOpenOrders(this.market);
      }
      await this.createBot();
      const event: RuntimeBotEvent = {
        type: "reset",
        at: Date.now(),
        message: "Bot reset",
      };
      this.recordEvents([event]);
      await this.flushState();
      return [event];
    });
  }

  async closePositions(_options: { includeUnprofitable?: boolean } = {}): Promise<RuntimeBotEvent[]> {
    return this.withOperation(async () => {
      this.status = "stopped";
      await this.bot.closePositions();
      const events = await this.deliverOrderEvents();
      await this.refreshState(hasFill(events));
      await this.flushState();
      return events;
    });
  }

  async updateBotConfig(patch: PartialStrategyConfig | PeakValleyBotConfig): Promise<RuntimeBotEvent[]> {
    return this.withOperation(async () => {
      if (isBotConfig(patch)) {
        this.botConfig = normalizeBotConfig(
          patch,
          this.market.maxLeverage,
          intervalToMs(this.interval),
        );
        this.legacyConfig = createStrategyConfig({
          ...this.legacyConfig,
          maxLeverage: this.botConfig.maxTargetLeverage,
          cooldownMs: this.botConfig.cooldownMs,
          minOrderQuote: this.botConfig.minTradeQuote,
          maxPositionQuote: this.botConfig.maxTradeQuote,
          internalBorrowAccounting: this.botConfig.internalBorrow.enabled ? "active" : "inactive",
          borrowerProfitShareToLender: this.botConfig.internalBorrow.borrowerProfitShare,
          legacyValleyPeak: this.botConfig.strategy,
        });
      } else {
        this.legacyConfig = createStrategyConfig({
          ...this.legacyConfig,
          ...patch,
          legacyValleyPeak: {
            ...this.legacyConfig.legacyValleyPeak,
            ...(patch.legacyValleyPeak ?? {}),
          },
        });
        this.botConfig = createPeakValleyBotConfig(this.legacyConfig, intervalToMs(this.interval));
      }
      await this.bot.updateConfig(this.botConfig);
      await this.refreshState();
      await this.flushState();
      return [];
    });
  }

  async recordManualTrade(input: ManualTradeInput): Promise<RuntimeBotEvent[]> {
    if (input.positionEffect === "close" || input.targetPositionId) {
      await this.bot.closePosition(
        input.side === "sell" ? "long" : "short",
        input.quantity,
        input.targetPositionId,
      );
    } else {
      await this.bot.openPosition(input.side === "buy" ? "long" : "short", input.quantity, {
        lifetimeMs: input.lifetimeMs,
        stopLossPrice: input.stopLossPrice,
        takeProfitPrice: input.takeProfitPrice,
      });
    }
    const events = await this.deliverOrderEvents();
    await this.refreshState(hasFill(events));
    return events;
  }

  async syncExchange(): Promise<BinanceExchangeSnapshot> {
    if (!this.exchangeTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    if (this.api instanceof BinanceTradingApi) {
      await this.api.sync();
      await this.deliverOrderEvents();
      await this.refreshState(true);
    } else {
      await this.exchangeTrading.sync(this.market);
    }
    return this.exchangeTrading.snapshot(this.market);
  }

  async placeExchangeOrder(input: BinanceExchangePlaceOrderInput): Promise<BinanceExchangeSnapshot> {
    if (!this.exchangeTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    return this.exchangeTrading.placeOrder(this.market, input);
  }

  async cancelExchangeOrder(input: BinanceExchangeCancelOrderInput): Promise<BinanceExchangeSnapshot> {
    if (!this.exchangeTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    return this.exchangeTrading.cancelOrder(this.market, input);
  }

  async cancelAllExchangeOrders(): Promise<BinanceExchangeSnapshot> {
    if (!this.exchangeTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    return this.exchangeTrading.cancelAllOpenOrders(this.market);
  }

  async setExchangeLeverage(leverage: number): Promise<BinanceExchangeSnapshot> {
    if (!this.exchangeTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    return this.exchangeTrading.changeLeverage(this.market, leverage);
  }

  async setExchangeCredentials(input: {
    mode?: string;
    apiKey?: string;
    apiSecret?: string;
  }): Promise<BinanceExchangeSnapshot> {
    if (!this.exchangeTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    if (this.status === "running") {
      throw new Error("Stop the bot before changing Binance credentials.");
    }
    const mode = normalizeExchangeMode(input.mode)
      ?? this.runtimeSettings.exchange?.mode
      ?? "live";
    const current = this.runtimeSettings.exchange;
    const next: TradingExchangeCredentials = {
      mode,
      sandboxApiKey: current?.sandboxApiKey,
      sandboxApiSecret: current?.sandboxApiSecret,
      liveApiKey: current?.liveApiKey,
      liveApiSecret: current?.liveApiSecret,
      updatedAt: Date.now(),
    };
    const live = mode === "live" || mode.endsWith("-live");
    if (input.apiKey !== undefined) {
      live ? next.liveApiKey = input.apiKey : next.sandboxApiKey = input.apiKey;
    }
    if (input.apiSecret !== undefined) {
      live ? next.liveApiSecret = input.apiSecret : next.sandboxApiSecret = input.apiSecret;
    }
    this.runtimeSettings.exchange = next;
    this.runtimeSettings.updatedAt = Date.now();
    this.applyExchangeCredentials();
    await this.storage.saveRuntimeSettings(this.runtimeSettings);
    return this.exchangeTrading.snapshot(this.market);
  }

  async setExecutionMode(mode: TradingExecutionMode): Promise<RuntimeBotEvent[]> {
    if (mode === this.executionMode) {
      return [];
    }
    if (this.status === "running") {
      throw new Error("Stop the bot before changing execution mode.");
    }
    if (this.botState.positions.length > 0) {
      throw new Error("Close all positions before changing execution mode.");
    }
    if (mode === "binance" && !this.exchangeTrading?.canSubmitOrders(this.market)) {
      throw new Error("Configure a compatible Binance account before enabling Binance execution.");
    }
    await this.bot.cancelOpenOrders();
    this.executionMode = mode;
    this.runtimeSettings.executionMode = mode;
    this.runtimeSettings.updatedAt = Date.now();
    await this.storage.saveRuntimeSettings(this.runtimeSettings);
    await this.createBot();
    return [];
  }

  startBacktest(options: BacktestStartOptions, onUpdate: () => void): BacktestProgressSnapshot {
    if (this.backtest.status === "running") {
      throw new Error("A backtest is already running.");
    }
    const id = `bt_${Date.now()}`;
    const controller = new AbortController();
    this.backtestAbort = controller;
    const now = Date.now();
    this.backtest = {
      id,
      preset: options.preset,
      status: "running",
      source: options.preset === "saved-orderbook" ? "orderbook-mid" : "candles",
      startedAt: now,
      updatedAt: now,
      targetStartTime: 0,
      targetEndTime: now,
      processedCandles: 0,
      estimatedCandles: 0,
      requests: 0,
      percent: 0,
      equity: this.legacyConfig.startingQuote,
      returnPct: 0,
      message: "Starting backtest",
    };
    onUpdate();
    void this.executeBacktest(id, options, onUpdate, controller);
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
    await this.refreshState();
    const state: StoredRuntimeState = {
      version: 1,
      status: this.status,
      runStartedAt: this.runStartedAt,
      bot: this.botState,
      paper: this.api instanceof PaperTradingApi ? this.api.snapshot() : undefined,
      binance: this.api instanceof BinanceTradingApi ? this.api.state() : undefined,
    };
    const save = () => this.storage.saveTradingState(state);
    this.saveQueue = this.saveQueue.then(save, save);
    await this.saveQueue;
  }

  private async createBot(saved?: StoredRuntimeState): Promise<void> {
    this.entryRisk.clear();
    const restored = currentStoredState(saved);
    this.status = restored?.status ?? this.status;
    this.runStartedAt = restored?.runStartedAt ?? Date.now();
    this.botConfig = restored?.bot.config
      ? normalizeBotConfig(
          restored.bot.config,
          this.market.maxLeverage,
          intervalToMs(this.interval),
        )
      : createPeakValleyBotConfig(this.legacyConfig, intervalToMs(this.interval));
    this.api = this.executionMode === "binance" && this.exchangeTrading
      ? new BinanceTradingApi({
          market: this.market,
          client: new BinanceExchangeClient(this.exchangeTrading),
          getHistory: (request) => this.getHistory(request.count),
          snapshot: restored?.binance,
        })
      : new PaperTradingApi({
          startingQuote: this.legacyConfig.startingQuote,
          friction: (this.legacyConfig.feeBps + this.legacyConfig.positionRisk.marketSlippageBps) / 10_000,
          rules: marketRules(this.market, this.exchangeTrading?.snapshot(this.market)),
          getHistory: (request) => this.getHistory(request.count),
          snapshot: restored?.paper,
        });
    const strategy = new PeakValleyStrategy({
      config: this.botConfig.strategy,
      getHistory: this.api.getHistory.bind(this.api),
    });
    this.bot = new GridTradingBot({
      api: this.api,
      strategy,
      config: this.botConfig,
      onEntryRisk: (report) => this.entryRisk.set(report.side, report),
    });
    if (restored?.bot) {
      const sampleIntervalMs = intervalToMs(this.interval);
      const savedIntervalMs = restored.bot.config.strategy.sampleIntervalMs ?? 60_000;
      await this.bot.restore(
        { ...restored.bot, config: this.botConfig },
        { restoreStrategy: savedIntervalMs === sampleIntervalMs },
      );
    } else {
      await this.bot.warmup();
    }
    if (this.api instanceof BinanceTradingApi) {
      await this.api.sync();
      await this.deliverOrderEvents();
    }
    await this.refreshState(true);
    this.recordEquity(Date.now(), true);
  }

  private async getHistory(count: number): Promise<TradingCandle[]> {
    return (await this.storage.loadCandles(Math.max(1, count))).map(toTradingCandle);
  }

  private async deliverOrderEvents(): Promise<RuntimeBotEvent[]> {
    const events = this.api.drainEvents();
    for (const event of events) {
      await this.bot.onOrder(event);
    }
    const runtimeEvents = events.map(toRuntimeEvent);
    this.recordEvents(runtimeEvents);
    return runtimeEvents;
  }

  private async refreshState(refreshEquity = false): Promise<void> {
    this.botState = await this.bot.snapshot();
    if (refreshEquity) {
      this.equity = await this.api.getEquity();
    }
  }

  private scheduleSave(): void {
    this.saveTimer ??= setTimeout(() => {
      this.saveTimer = undefined;
      void this.flushState();
    }, 2_000);
  }

  private recordEvents(events: RuntimeBotEvent[]): void {
    this.recentEvents = [...events.reverse(), ...this.recentEvents].slice(0, 60);
  }

  private recordStatus(message: string): RuntimeBotEvent[] {
    const event: RuntimeBotEvent = { type: "status", at: Date.now(), message };
    this.recordEvents([event]);
    this.scheduleSave();
    return [event];
  }

  private recordEquity(at: number, force = false): void {
    if (!force && at - this.lastEquityAt < 500) {
      return;
    }
    this.lastEquityAt = at;
    this.liveEquityCurve.push({
      time: at,
      equity: this.equity.quoteUnleveraged + this.equity.assetUnleveraged * this.lastPrice(),
      price: this.lastPrice(),
    });
    if (this.liveEquityCurve.length > 1_200) {
      this.liveEquityCurve.shift();
    }
  }

  private lastPrice(): number {
    return this.lastTickPrice || this.candles.at(-1)?.close || 0;
  }

  private executionSnapshot(): RuntimeSnapshot["execution"] {
    const exchange = this.exchangeTrading?.snapshot(this.market);
    const exchangeDriven = this.executionMode === "binance";
    return {
      mode: this.executionMode,
      exchangeDriven,
      canUseExchange: this.exchangeTrading?.canSubmitOrders(this.market) ?? false,
      live: exchange?.live ?? false,
      message: exchangeDriven
        ? "Bot orders use the Binance trading adapter"
        : "Bot orders use the local paper trading adapter",
    };
  }

  private async loadSettings(): Promise<void> {
    this.runtimeSettings = await this.storage.loadRuntimeSettings() ?? this.runtimeSettings;
    this.applyExchangeCredentials();
    this.executionMode = this.runtimeSettings.executionMode === "binance"
      && this.exchangeTrading?.canSubmitOrders(this.market)
      ? "binance"
      : "simulated";
  }

  private applyExchangeCredentials(): void {
    const exchange = this.runtimeSettings.exchange;
    if (!exchange || !this.exchangeTrading) {
      return;
    }
    this.exchangeTrading.updateConfig({
      mode: exchange.mode,
      apiKey: exchange.sandboxApiKey,
      apiSecret: exchange.sandboxApiSecret,
      liveApiKey: exchange.liveApiKey,
      liveApiSecret: exchange.liveApiSecret,
    });
  }

  private async withOperation<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationQueue;
    let release!: () => void;
    this.operationQueue = new Promise<void>((resolve) => release = resolve);
    await previous.catch(() => undefined);
    try {
      return await operation();
    } finally {
      release();
    }
  }

  private async executeBacktest(
    id: string,
    options: BacktestStartOptions,
    onUpdate: () => void,
    controller: AbortController,
  ): Promise<void> {
    try {
      const result = await this.runBacktest(options, onUpdate, controller.signal);
      throwIfCancelled(controller.signal);
      await this.storage.saveBacktest(result);
      if (this.backtest.id === id) {
        this.backtest = completedBacktest(this.backtest, result);
        onUpdate();
      }
    } catch (error) {
      if (this.backtest.id !== id) {
        return;
      }
      const cancelled = controller.signal.aborted || isBacktestCancelledError(error);
      this.backtest = {
        ...this.backtest,
        status: cancelled ? "cancelled" : "failed",
        stopReason: cancelled ? "cancelled" : "error",
        updatedAt: Date.now(),
        message: cancelled ? "Backtest cancelled" : errorMessage(error),
        error: cancelled ? undefined : errorMessage(error),
      };
      onUpdate();
    } finally {
      if (this.backtestAbort === controller) {
        this.backtestAbort = undefined;
      }
    }
  }

  private async runBacktest(
    options: BacktestStartOptions,
    onUpdate: () => void,
    signal: AbortSignal,
  ): Promise<BacktestResult> {
    throwIfCancelled(signal);
    const config = createStrategyConfig({
      ...this.legacyConfig,
      ...(options.startingQuote ? { startingQuote: options.startingQuote } : {}),
    });
    if (isHistoricalPreset(options.preset)) {
      if (!this.market.supportsHistoricalCandles || !isHistoricalVenue(this.market.venue)) {
        throw new Error(`${this.market.displaySymbol} does not support candle backtests.`);
      }
      return runHistoricalCandleBacktest({
        id: this.backtest.id,
        preset: options.preset,
        marketId: this.market.id,
        marketKey: this.market.id,
        venue: this.market.venue,
        symbol: this.market.symbol,
        displaySymbol: this.market.displaySymbol,
        baseAsset: this.market.baseAsset,
        quoteAsset: this.market.quoteAsset,
        maxLeverage: this.market.maxLeverage,
        interval: this.interval,
        config,
        cache: this.historicalCache,
        historicalStartTime: options.historicalStartTime,
        historicalRangeMs: days(options.historicalDays),
        randomSampleCount: options.randomSampleCount,
        randomWindowMs: days(options.randomWindowDays),
        randomMinWindowMs: days(options.randomMinWindowDays),
        randomMaxWindowMs: days(options.randomMaxWindowDays),
        randomLookbackMs: days(options.randomLookbackDays),
        randomMarkets: options.randomMarkets,
        randomPairCount: options.randomMarkets?.length ?? options.randomPairCount,
        extremaSmaWindowMs: options.extremaSmaWindowMinutes === undefined
          ? undefined
          : options.extremaSmaWindowMinutes * 60_000,
        cancelSignal: signal,
      }, (progress) => {
        this.backtest = progress;
        onUpdate();
      });
    }
    const result = options.preset === "saved-orderbook"
      ? runBacktestFromOrderBook(await this.storage.loadOrderBookSnapshots(options.limit), { config })
      : await runBotBacktestFromCandles(await this.storage.loadCandles(options.limit), {
          config,
          extremaSmaWindowMs: options.extremaSmaWindowMinutes === undefined
            ? undefined
            : options.extremaSmaWindowMinutes * 60_000,
        });
    throwIfCancelled(signal);
    return result;
  }
}

function currentStoredState(state: StoredRuntimeState | undefined): StoredRuntimeState | undefined {
  return state?.bot.version === 2 && state.bot.strategy?.version === 3 ? state : undefined;
}

function normalizeBotConfig(
  config: PeakValleyBotConfig,
  marketMax?: number,
  sampleIntervalMs = config.strategy.sampleIntervalMs ?? 60_000,
): PeakValleyBotConfig {
  const maxLeverage = Math.max(1, Math.min(config.maxTargetLeverage, marketMax ?? Infinity));
  const minTrade = Math.max(0, config.minTradeQuote);
  return {
    ...structuredClone(config),
    strategy: rescalePeakValleyStrategyConfig(
      createPeakValleyStrategyConfig(config.strategy),
      sampleIntervalMs,
    ),
    maxTargetLeverage: maxLeverage,
    minTradeQuote: minTrade,
    maxTradeQuote: Math.max(minTrade, config.maxTradeQuote),
    cooldownMs: Math.max(0, config.cooldownMs),
  };
}

function isBotConfig(value: PartialStrategyConfig | PeakValleyBotConfig): value is PeakValleyBotConfig {
  return "strategy" in value;
}

function toTradingCandle(candle: Candle): TradingCandle {
  return {
    openTime: candle.openTime,
    closeTime: candle.closeTime,
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
  };
}

function toRuntimeEvent(event: TradingOrderEvent): RuntimeBotEvent {
  const at = Date.now();
  if (event.type === "open") {
    return { type: "open", at, message: `${event.order.side} order open`, order: event.order };
  }
  if (event.type === "rejected") {
    return { type: "rejected", at, message: "Order rejected", orderId: event.orderId };
  }
  return {
    type: event.type,
    at,
    message: event.type === "fill" ? "Order filled" : "Order partially filled",
    orderId: event.orderId,
    fill: event.fill,
  };
}

function hasFill(events: RuntimeBotEvent[]): boolean {
  return events.some((event) => event.type === "fill" || event.type === "partial-fill");
}

function marketRules(
  market: BinanceMarketListing,
  snapshot?: BinanceExchangeSnapshot,
) {
  const filters = snapshot?.symbolFilters;
  return {
    price: quantityRules(filters?.minPrice, filters?.maxPrice, filters?.tickSize),
    limitQuantity: quantityRules(filters?.minQuantity, filters?.maxQuantity, filters?.stepSize),
    marketQuantity: quantityRules(
      filters?.minMarketQuantity ?? filters?.minQuantity,
      filters?.maxMarketQuantity ?? filters?.maxQuantity,
      filters?.marketStepSize ?? filters?.stepSize,
    ),
    minNotional: filters?.minNotional ?? 5,
    maxNotional: filters?.maxNotional ?? null,
    maxLeverage: snapshot?.maxLeverage ?? market.maxLeverage ?? 1,
  };
}

function quantityRules(min?: number, max?: number, step?: number) {
  return { min: min ?? null, max: max ?? null, step: step ?? null };
}

function emptyEquity(): EquitySnapshot {
  return {
    quoteAvailable: 0,
    quoteReserved: 0,
    quoteUnleveraged: 0,
    assetAvailable: 0,
    assetReserved: 0,
    assetUnleveraged: 0,
  };
}

function emptyExchange(): BinanceExchangeSnapshot {
  return {
    enabled: false,
    configured: false,
    compatible: true,
    mode: "auto",
    live: false,
    autoSubmit: false,
    connected: false,
    message: "Binance exchange trading is not configured",
    balances: [],
    positions: [],
    openOrders: [],
    recentOrders: [],
    recentTrades: [],
  };
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

function completedBacktest(
  progress: BacktestProgressSnapshot,
  result: BacktestResult,
): BacktestProgressSnapshot {
  return {
    ...progress,
    status: "completed",
    updatedAt: Date.now(),
    percent: 100,
    equity: result.summary.finalEquity,
    returnPct: result.summary.returnPct,
    stopReason: result.summary.stopReason ?? "completed",
    message: "Backtest completed",
    result,
  };
}

function isHistoricalPreset(preset: BacktestPreset): preset is HistoricalPreset {
  return ["last-x", "week", "month", "year", "random-windows", "random-length-windows"].includes(preset);
}

function isHistoricalVenue(venue: string): venue is StreamVenue {
  return venue === "spot" || venue === "usdm-futures" || venue === "coinm-futures";
}

function days(value?: number): number | undefined {
  return value === undefined ? undefined : value * 24 * 60 * 60 * 1000;
}

function throwIfCancelled(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new BacktestCancelledError();
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Backtest failed";
}

function normalizeExchangeMode(value?: string): TradingExchangeMode | undefined {
  return [
    "auto",
    "live",
    "spot-live",
    "usdm-futures-live",
    "coinm-futures-live",
    "spot-testnet",
    "spot-demo",
    "usdm-futures-testnet",
    "coinm-futures-testnet",
  ].includes(value ?? "") ? value as TradingExchangeMode : undefined;
}
