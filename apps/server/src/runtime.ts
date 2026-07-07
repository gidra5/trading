import fs from "node:fs/promises";
import path from "node:path";
import {
  SimulatedExecutionEngine,
  analyzePositions,
  createInitialBotState,
  createStrategyConfig,
  legacyValleyPeakObservedPriceRangeWarmupRatio,
  legacyValleyPeakObservedSignalWarmupSec,
  legacyValleyPeakPriceRangeWarmupSec,
  legacyValleyPeakSignalWarmupSec,
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
  type EquityPoint,
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
  BinancePaperOrderSubmissionSkipped,
  BinancePaperTrading,
  type BinancePaperConfig,
  type BinancePaperCancelOrderInput,
  type BinancePaperOrder,
  type BinancePaperPlaceOrderInput,
  type BinancePaperSnapshot,
  type BinancePaperTrade,
} from "./binance-paper.js";
import { HistoricalCandleCache } from "./historical-candle-cache.js";
import type {
  TradingExchangeCredentials,
  TradingExchangeMode,
  TradingExecutionMode,
  TradingRuntimeSettings,
  TradingStorage,
} from "./storage.js";

const PUBLIC_ORDER_LIMIT = 500;
const PUBLIC_FILL_LIMIT = 500;
const PUBLIC_PRICE_MEMORY_LIMIT = 100;
const BALANCE_EPSILON = 0.000001;
const ASSET_DRIFT_TOLERANCE_RATE = 0.0001;
const MIN_ASSET_DRIFT_TOLERANCE = 0.00000001;
const QUOTE_DRIFT_TOLERANCE_RATE = 0.001;
const MIN_QUOTE_DRIFT_TOLERANCE = 2;
const FUTURES_UNRELIABLE_MARK_EQUITY_DRIFT_TOLERANCE_RATE = 0.001;
const EXCHANGE_ACCOUNT_GUARD_SETTLEMENT_GRACE_MS = 15_000;
const LIVE_EQUITY_CURVE_CAP = 1_200;
const LIVE_EQUITY_SAMPLE_MIN_SPACING_MS = 500;
const HISTORICAL_WARMUP_COMPLETENESS_RATIO = 0.95;
const PRICE_RANGE_WARMUP_INTERVAL = "1d";

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
}

interface ExchangeCredentialsUpdateInput {
  mode?: string;
  apiKey?: string;
  apiSecret?: string;
}

interface HistoricalWarmupCandles {
  signalCandles: Candle[];
  priceRangeCandles: Candle[];
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
  equityCurve: EquityPoint[];
  execution: {
    mode: TradingExecutionMode;
    exchangeDriven: boolean;
    canUseExchange: boolean;
    live: boolean;
    message: string;
  };
  exchange: BinancePaperSnapshot;
}

interface ExchangeAccountGuardOptions {
  hardStop: boolean;
  onWarning?: (message: string) => void;
}

export class TradingRuntime {
  private bot!: SimulatedExecutionEngine;
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
  private lastExchangeExecutionAt = 0;
  private historicalWarmupGeneration = 0;
  private historicalWarmupPromise?: Promise<void>;
  private botOperationQueue: Promise<void> = Promise.resolve();
  private liveEquityCurve: EquityPoint[] = [];
  private lastLiveEquitySampleAt = 0;
  private executionMode: TradingExecutionMode = "simulated";
  private runtimeSettings: TradingRuntimeSettings = {
    executionMode: "simulated",
    updatedAt: 0,
  };

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
    await this.loadRuntimeSettings();
    this.candles = await this.storage.loadCandles(500);
    const exchangeDriven = this.isExchangeDrivenExecution();
    const exchangeSnapshot = await this.syncExecutionExchangeSnapshot();
    const savedState = exchangeDriven
      ? await this.storage.loadLiveBotState()
      : await this.storage.loadBotState();
    const exchangeStartingQuote = exchangeDriven
      ? undefined
      : exchangeStartingQuoteForInitialState(exchangeSnapshot, this.market, savedState);
    const initialState = exchangeDriven
      ? savedState
      : rebaseBotStateCapital(savedState, exchangeStartingQuote);
    this.config = this.createMarketConfig(initialState ?? savedState, exchangeStartingQuote);
    this.bot = new SimulatedExecutionEngine(initialState, this.config);
    this.warmupBotFromRecentCandles(initialState);
    this.startHistoricalWarmup(initialState);
    await this.recoverExchangeState(exchangeSnapshot);
    this.recordLiveEquity(Date.now());
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
    this.liveEquityCurve = [];
    this.lastLiveEquitySampleAt = 0;
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
    await this.loadRuntimeSettings();
    this.candles = await this.storage.loadCandles(500);
    const exchangeDriven = this.isExchangeDrivenExecution();
    const exchangeSnapshot = await this.syncExecutionExchangeSnapshot();
    const savedState = exchangeDriven
      ? await this.storage.loadLiveBotState()
      : await this.storage.loadBotState();
    const exchangeStartingQuote = exchangeDriven
      ? undefined
      : exchangeStartingQuoteForInitialState(exchangeSnapshot, this.market, savedState);
    const initialState = exchangeDriven
      ? savedState
      : rebaseBotStateCapital(savedState, exchangeStartingQuote);
    this.config = this.createMarketConfig(initialState ?? savedState, exchangeStartingQuote);
    this.bot = new SimulatedExecutionEngine(initialState, this.config);
    this.warmupBotFromRecentCandles(initialState);
    this.startHistoricalWarmup(initialState);
    await this.recoverExchangeState(exchangeSnapshot);
    this.recordLiveEquity(Date.now());
  }

  handleStatus(status: MarketStreamStatus): void {
    this.status = status;
  }

  async handleTick(tick: PriceTick): Promise<BotEvent[]> {
    return this.withBotOperation(async () => {
      const exchangeDriven = this.isExchangeDrivenExecution();
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
      await this.submitCreatedOrdersToPaperExchange(events, { force: exchangeDriven });
      this.recordEvents(events);
      this.recordLiveEquity(tick.eventTime);
      this.scheduleStateSave();
      return events;
    });
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
    return this.withBotOperation(async () => {
      if (!this.paperTrading) {
        return [];
      }

      const events: BotEvent[] = [];
      const exchangeDriven = this.isExchangeDrivenExecution();
      const reconciliation = this.paperTrading.reconciliationFromUserDataEvent(
        this.market,
        payload,
      );
      if (hasExchangeReconciliationUpdates(reconciliation)) {
        const directEvents = this.bot.applyExchangeReconciliation(reconciliation);
        this.noteExchangeExecutions(directEvents);
        this.recordEvents(directEvents);
        events.push(...directEvents);
        await this.flushState();
      }

      try {
        const snapshot = await this.paperTrading.sync(this.market);
        events.push(...(await this.applyExchangeSnapshot(snapshot)));
      } catch {
        if (events.length > 0) {
          this.recordLiveEquity(Date.now());
        }
        return events;
      }

      return events;
    });
  }

  snapshot(): RuntimeSnapshot {
    const bot = this.bot.view();
    const exchangeSnapshot = this.publicExchangeSnapshot();
    const exchangeDriven = this.isExchangeDrivenExecution();
    const publicBot = compactPublicBotState(
      exchangeDriven
        ? exchangeDrivenPublicBotState(bot, exchangeSnapshot, this.market)
        : bot,
    );
    const positionState = exchangeDriven ? this.bot.snapshot() : publicBot;
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
      positions: analyzePositions(positionState),
      recentEvents: compactPublicEvents(this.recentEvents),
      backtest: this.backtest,
      equityCurve: this.liveEquityCurve,
      execution: this.publicExecutionSnapshot(exchangeSnapshot),
      exchange: exchangeSnapshot,
    };
  }

  async startBot(): Promise<BotEvent[]> {
    return this.withBotOperation(async () => {
      const events: BotEvent[] = [];
      if (this.isExchangeDrivenExecution()) {
        const snapshot = await this.syncExecutionExchangeSnapshot();
        if (snapshot) {
          events.push(...(await this.applyExchangeSnapshot(snapshot)));
        }
      }

      const wasStopped = this.bot.view().status !== "running";
      if (wasStopped) {
        this.liveEquityCurve = [];
        this.lastLiveEquitySampleAt = 0;
      }
      const statusEvents = this.bot.setStatus("running");
      events.push(...statusEvents);
      if (wasStopped) {
        this.recordLiveEquity(Date.now(), this.bot.snapshot(), true);
      }
      this.recordEvents(statusEvents);
      await this.flushState();
      return events;
    });
  }

  async stopBot(): Promise<BotEvent[]> {
    return this.withBotOperation(async () => {
      const events = this.bot.setStatus("stopped");
      this.recordEvents(events);
      await this.flushState();

      if (this.isExchangeDrivenExecution()) {
        try {
          const snapshot = await this.syncExecutionExchangeSnapshot();
          if (snapshot) {
            events.push(...(await this.applyExchangeSnapshot(snapshot)));
          }
        } catch {
          // The local stop must remain effective even if Binance sync is unavailable.
        }
      }

      return events;
    });
  }

  async resetBot(): Promise<BotEvent[]> {
    return this.withBotOperation(() => this.resetBotUnlocked());
  }

  private async resetBotUnlocked(): Promise<BotEvent[]> {
    this.liveEquityCurve = [];
    this.lastLiveEquitySampleAt = 0;
    const previousStatus = this.bot.view().status;
    const exchangeCanSubmit = this.paperTrading?.canSubmitOrders(this.market) ?? false;
    const exchangeDrivesExecution = this.isExchangeDrivenExecution();
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
      ? undefined
      : exchangeQuoteBalance(resetSnapshot, this.market);
    if (exchangeStartingQuote !== undefined) {
      this.config = this.createMarketConfig(undefined, exchangeStartingQuote);
    }

    const events = this.bot.reset(this.config);
    this.warmupBotFromRecentCandles();
    this.startHistoricalWarmup();
    if (previousStatus === "stopped") {
      events.push(...this.bot.setStatus("stopped"));
    }
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
    return this.withBotOperation(() => this.closePositionsUnlocked(options));
  }

  private async closePositionsUnlocked(
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

    if (this.isExchangeDrivenExecution()) {
      if (
        exchangeCanSubmit &&
        (this.market.venue === "usdm-futures" || this.market.venue === "coinm-futures")
      ) {
        const snapshot = await this.paperTrading!.closeOpenPositions(this.market, options);
        await this.applyExchangeSnapshot(snapshot);
      }
      this.recordEvents(events);
      await this.flushState();
      return events;
    }

    if (exchangeCanSubmit) {
      if (options.includeUnprofitable) {
        const snapshot = await this.paperTrading!.closeOpenPositions(this.market, options);
        events.push(...this.closePositionsLocally(options, at));
        events.push(...this.resetFlatBotFromExchangeSnapshot(snapshot));
        this.recordEvents(events);
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
    return this.withBotOperation(() => this.updateBotConfigUnlocked(patch));
  }

  private async updateBotConfigUnlocked(patch: PartialStrategyConfig): Promise<BotEvent[]> {
    const previousStatus = this.bot.view().status;
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
    this.warmupBotFromRecentCandles();
    this.startHistoricalWarmup();
    if (previousStatus === "stopped") {
      events.push(...this.bot.setStatus("stopped"));
    }
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  async recordManualTrade(input: ManualTradeInput): Promise<BotEvent[]> {
    if (this.isExchangeDrivenExecution()) {
      throw new Error(
        "Manual local trades are simulation-only when exchange drives execution; submit an exchange order instead.",
      );
    }

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
      throw new Error("Binance exchange trading is not configured.");
    }
    const snapshot = await this.paperTrading.sync(this.market);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async placeExchangeOrder(
    input: BinancePaperPlaceOrderInput,
  ): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    const snapshot = await this.paperTrading.placeOrder(this.market, input);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async cancelExchangeOrder(
    input: BinancePaperCancelOrderInput,
  ): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    const snapshot = await this.paperTrading.cancelOrder(this.market, input);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async cancelAllExchangeOrders(): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    const snapshot = await this.paperTrading.cancelAllOpenOrders(this.market);
    await this.applyExchangeSnapshot(snapshot);
    return snapshot;
  }

  async setExchangeLeverage(leverage: number): Promise<BinancePaperSnapshot> {
    if (!this.paperTrading) {
      throw new Error("Binance exchange trading is not configured.");
    }
    const snapshot = await this.paperTrading.changeLeverage(this.market, leverage);
    this.applyExchangeMaxLeverage(snapshot);
    await this.flushState();
    return snapshot;
  }

  async setExchangeCredentials(
    input: ExchangeCredentialsUpdateInput,
  ): Promise<BinancePaperSnapshot> {
    return this.withBotOperation(async () => {
      if (!this.paperTrading) {
        throw new Error("Binance exchange trading is not configured.");
      }
      if (this.bot.view().status === "running") {
        throw new Error("Stop the bot before changing Binance credentials.");
      }

      const now = Date.now();
      const current = this.runtimeSettings.exchange;
      const mode =
        normalizeTradingExchangeMode(input.mode) ??
        current?.mode ??
        normalizeTradingExchangeMode(this.paperTrading.snapshot(this.market).mode) ??
        "live";
      const next: TradingExchangeCredentials = {
        mode,
        sandboxApiKey: current?.sandboxApiKey,
        sandboxApiSecret: current?.sandboxApiSecret,
        liveApiKey: current?.liveApiKey,
        liveApiSecret: current?.liveApiSecret,
        updatedAt: now,
      };
      const apiKey = normalizeCredentialInput(input.apiKey);
      const apiSecret = normalizeCredentialInput(input.apiSecret);
      if (exchangeModeUsesLiveCredentials(mode)) {
        if (apiKey !== undefined) {
          next.liveApiKey = apiKey;
        }
        if (apiSecret !== undefined) {
          next.liveApiSecret = apiSecret;
        }
      } else {
        if (apiKey !== undefined) {
          next.sandboxApiKey = apiKey;
        }
        if (apiSecret !== undefined) {
          next.sandboxApiSecret = apiSecret;
        }
      }

      this.runtimeSettings = {
        ...this.runtimeSettings,
        exchange: next,
        updatedAt: now,
      };
      this.applyRuntimeExchangeCredentials();
      if (this.executionMode === "binance" && !this.canUseExchangeForBotExecution()) {
        this.executionMode = "simulated";
        this.runtimeSettings.executionMode = "simulated";
      }
      await this.saveRuntimeSettings();
      this.exchangeAccountWarningMessage = undefined;
      return this.publicExchangeSnapshot();
    });
  }

  async setExecutionMode(mode: TradingExecutionMode): Promise<BotEvent[]> {
    return this.withBotOperation(() => this.setExecutionModeUnlocked(mode));
  }

  private async setExecutionModeUnlocked(mode: TradingExecutionMode): Promise<BotEvent[]> {
    const nextMode = normalizeTradingExecutionMode(mode) ?? "simulated";
    if (nextMode === this.executionMode) {
      return [];
    }
    if (this.bot.view().status === "running") {
      throw new Error("Stop the bot before changing execution mode.");
    }

    let exchangeSnapshot: BinancePaperSnapshot | undefined;
    if (nextMode === "binance") {
      exchangeSnapshot = await this.prepareExchangeExecutionSwitch();
    } else if (this.executionMode === "binance") {
      exchangeSnapshot = await this.prepareSimulatedExecutionSwitch();
    }

    await this.flushState();
    this.executionMode = nextMode;
    await this.saveExecutionMode();
    this.exchangeAccountWarningMessage = undefined;
    this.liveEquityCurve = [];
    this.lastLiveEquitySampleAt = 0;

    const at = Date.now();
    const events =
      nextMode === "binance"
        ? this.rebuildBotForExchangeExecution(exchangeSnapshot, at)
        : await this.rebuildBotForSimulatedExecution(at);
    if (exchangeSnapshot) {
      events.push(...(await this.applyExchangeSnapshot(exchangeSnapshot)));
    }
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  private async prepareExchangeExecutionSwitch(): Promise<BinancePaperSnapshot> {
    if (!this.canUseExchangeForBotExecution()) {
      throw new Error(this.exchangeExecutionUnavailableMessage());
    }
    const snapshot = await this.paperTrading!.sync(this.market);
    assertExchangeAccountFlatForExecutionSwitch(snapshot, this.market);
    return snapshot;
  }

  private async prepareSimulatedExecutionSwitch(): Promise<BinancePaperSnapshot | undefined> {
    if (!this.paperTrading?.canSubmitOrders(this.market)) {
      return undefined;
    }
    const snapshot = await this.paperTrading.sync(this.market);
    if (hasExchangeMarketExposure(snapshot, this.market)) {
      throw new Error(
        "Cancel Binance open orders and close the selected market position before switching back to simulated execution.",
      );
    }
    return snapshot;
  }

  private rebuildBotForExchangeExecution(
    exchangeSnapshot: BinancePaperSnapshot | undefined,
    at: number,
  ): BotEvent[] {
    const previous = this.bot.snapshot();
    const exchangeStartingQuote =
      exchangeQuoteBalance(exchangeSnapshot, this.market) ?? this.config.startingQuote;
    this.config = this.createMarketConfig(undefined, exchangeStartingQuote);
    const state = createInitialBotState(this.config);
    state.id = previous.id;
    state.status = "stopped";
    state.lastPrice = previous.lastPrice;
    state.sequence = previous.sequence;
    state.createdAt = at;
    state.updatedAt = at;
    state.runStartedAt = previous.runStartedAt;
    state.memory = structuredClone(previous.memory);
    state.config = this.config;
    this.bot = new SimulatedExecutionEngine(state, this.config);
    this.warmupBotFromRecentCandles(state);
    this.startHistoricalWarmup(state);
    return [{
      type: "state_reset",
      at,
      message: "Binance execution enabled; local simulated positions cleared",
      state: this.bot.snapshot(),
    }];
  }

  private async rebuildBotForSimulatedExecution(at: number): Promise<BotEvent[]> {
    const savedState = await this.storage.loadBotState();
    this.config = this.createMarketConfig(savedState);
    this.bot = new SimulatedExecutionEngine(savedState, this.config);
    this.warmupBotFromRecentCandles(savedState);
    this.startHistoricalWarmup(savedState);
    this.bot.setStatus("stopped", at);
    return [{
      type: "state_reset",
      at,
      message: "Simulated execution enabled",
      state: this.bot.snapshot(),
    }];
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
    const saveState = this.isExchangeDrivenExecution()
      ? () => this.storage.saveLiveBotState(snapshot)
      : () => this.storage.saveBotState(snapshot);
    this.stateSaveQueue = this.stateSaveQueue.then(
      saveState,
      saveState,
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

  private async withBotOperation<T>(operation: () => Promise<T> | T): Promise<T> {
    const previous = this.botOperationQueue;
    let release!: () => void;
    this.botOperationQueue = new Promise<void>((resolve) => {
      release = resolve;
    });

    await previous.catch(() => undefined);
    try {
      return await operation();
    } finally {
      release();
    }
  }

  private recordEvents(events: BotEvent[]): void {
    if (events.length === 0) {
      return;
    }

    this.recentEvents = [...events.map(compactPublicEvent), ...this.recentEvents].slice(0, 60);
  }

  private recordLiveEquity(
    time: number,
    state: PaperBotState = this.bot.snapshot(),
    force = false,
    exchangeSnapshot?: BinancePaperSnapshot,
  ): void {
    if (!Number.isFinite(time)) {
      return;
    }

    const exchangeDriven = this.isExchangeDrivenExecution();
    const exchange = exchangeSnapshot ?? this.paperTrading?.snapshot(this.market);
    const exchangeEquity = exchangeDriven
      ? exchangeRuntimeEquity(exchange, this.market, state.lastPrice)
      : undefined;
    const exchangePrice = exchangeDriven
      ? exchangeRuntimePrice(exchange, state.lastPrice)
      : undefined;
    const equity = exchangeEquity ?? state.metrics.equity;
    const price = exchangePrice ?? state.lastPrice;
    if (!Number.isFinite(equity) || !Number.isFinite(price)) {
      return;
    }
    if (state.status !== "running") {
      return;
    }

    if (!force && this.lastLiveEquitySampleAt > 0) {
      if (time - this.lastLiveEquitySampleAt < LIVE_EQUITY_SAMPLE_MIN_SPACING_MS) {
        return;
      }
    }

    const point: EquityPoint = { time, equity, price };
    const lastPoint = this.liveEquityCurve.at(-1);
    if (lastPoint && lastPoint.time === time) {
      this.liveEquityCurve[this.liveEquityCurve.length - 1] = point;
    } else {
      this.liveEquityCurve.push(point);
    }

    if (this.liveEquityCurve.length > LIVE_EQUITY_CURVE_CAP) {
      this.liveEquityCurve = this.liveEquityCurve.slice(-LIVE_EQUITY_CURVE_CAP);
    }

    this.lastLiveEquitySampleAt = time;
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
        if (error instanceof BinancePaperOrderSubmissionSkipped) {
          continue;
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
    if (options.includeUnprofitable) {
      events.push(...this.closeResidualBaseLocally(price, reason, at));
    }
    return events;
  }

  private closeResidualBaseLocally(
    price: number,
    reason: string,
    at: number,
  ): BotEvent[] {
    const state = this.bot.snapshot();
    const residualBase = roundAssetBalance(state.baseFree + state.baseReserved);
    if (Math.abs(residualBase) <= BALANCE_EPSILON) {
      return [];
    }

    return this.bot.recordManualTrade({
      side: residualBase > 0 ? "sell" : "buy",
      price,
      quantity: Math.abs(residualBase),
      positionEffect: "close",
      reason: `${reason} residual settlement`,
    }, at);
  }

  private resetFlatBotFromExchangeSnapshot(snapshot: BinancePaperSnapshot): BotEvent[] {
    if (!this.isExchangeDrivenExecution()) {
      return [];
    }
    if (hasExchangeOpenOrdersOrPositions(snapshot)) {
      return [];
    }

    this.config = this.createMarketConfig();
    const events = this.bot.reset(this.config);
    this.warmupBotFromRecentCandles();
    this.startHistoricalWarmup();
    events.push(...this.bot.setStatus("stopped"));
    return events;
  }

  private async recoverExchangeState(snapshot?: BinancePaperSnapshot): Promise<void> {
    if (!this.isExchangeDrivenExecution()) {
      return;
    }
    try {
      await this.applyExchangeSnapshot(snapshot ?? await this.paperTrading!.sync(this.market));
    } catch {
      return;
    }
  }

  private async syncExecutionExchangeSnapshot(): Promise<BinancePaperSnapshot | undefined> {
    if (!this.isExchangeDrivenExecution()) {
      return undefined;
    }
    try {
      return await this.paperTrading!.sync(this.market);
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
    const warmupCandles = await this.loadWarmupCandles(
      options.market,
      options.config,
      options.endpoint,
    );
    if (
      options.generation !== this.historicalWarmupGeneration ||
      options.market.id !== this.market.id ||
      (warmupCandles.signalCandles.length === 0 &&
        warmupCandles.priceRangeCandles.length === 0)
    ) {
      return;
    }

    const mergedCandles = mergeWarmupCandles(
      warmupCandles.signalCandles,
      this.candles.filter((candle) => candle.closed),
    );
    const processed = await this.bot.warmupFromCandlesCooperative(mergedCandles, {
      priceRangeCandles: warmupCandles.priceRangeCandles,
      shouldContinue: () =>
        options.generation === this.historicalWarmupGeneration &&
        options.market.id === this.market.id,
    });
    if (
      options.generation !== this.historicalWarmupGeneration ||
      options.market.id !== this.market.id
    ) {
      return;
    }
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

    const requiredSignalWarmupSec = legacyValleyPeakSignalWarmupSec(
      config.legacyValleyPeak,
    );
    const memory = savedState!.memory.legacyValleyPeak;
    const observedSignalWarmupSec = legacyValleyPeakObservedSignalWarmupSec(memory);
    const observedPriceRangeWarmupRatio = legacyValleyPeakObservedPriceRangeWarmupRatio(
      memory,
    );
    return (
      observedSignalWarmupSec <
        requiredSignalWarmupSec * HISTORICAL_WARMUP_COMPLETENESS_RATIO ||
      observedPriceRangeWarmupRatio < HISTORICAL_WARMUP_COMPLETENESS_RATIO
    );
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
  ): Promise<HistoricalWarmupCandles> {
    const signalWarmupMs = legacyValleyPeakSignalWarmupSec(
      config.legacyValleyPeak,
    ) * 1000;
    const priceRangeWarmupMs = legacyValleyPeakPriceRangeWarmupSec() * 1000;
    const [signalCandles, priceRangeCandles] = await Promise.all([
      this.loadWarmupCandlesForInterval(
        market,
        endpoint,
        this.interval,
        signalWarmupMs,
        true,
      ),
      this.loadWarmupCandlesForInterval(
        market,
        endpoint,
        PRICE_RANGE_WARMUP_INTERVAL,
        priceRangeWarmupMs,
        this.interval === PRICE_RANGE_WARMUP_INTERVAL,
      ),
    ]);

    return { signalCandles, priceRangeCandles };
  }

  private async loadWarmupCandlesForInterval(
    market: BinanceMarketListing,
    endpoint: string | undefined,
    interval: string,
    warmupMs: number,
    includeStoredCandles: boolean,
  ): Promise<Candle[]> {
    const intervalMs = intervalToMs(interval);
    const requiredCandles = Math.max(10, Math.ceil(warmupMs / intervalMs) + 5);
    const storedCandles =
      includeStoredCandles && market.id === this.market.id
        ? this.candles
            .filter((candle) => candle.closed && candle.interval === interval)
            .slice(-requiredCandles)
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
    const legacyCandles = await readLegacyHistoricalCandles(
      this.historicalCache.dataDir,
      market.symbol,
      interval,
      startTime,
      endTime,
    );
    const legacyMergedCandles = mergeWarmupCandles(legacyCandles, storedCandles);
    const legacySpan = candleSpanMs(legacyMergedCandles);
    if (legacySpan >= warmupMs) {
      return legacyMergedCandles;
    }

    if (!isHistoricalVenue(market.venue)) {
      return legacyMergedCandles.length > 0 ? legacyMergedCandles : storedCandles;
    }
    const venue = market.venue;

    try {
      const cache = new HistoricalCandleCache({
        dataDir: this.historicalCache.dataDir,
        marketKey: market.id,
        symbol: market.symbol,
        interval,
        intervalMs,
        maxBytes: this.historicalCache.maxBytes,
        minFreeBytes: this.historicalCache.minFreeBytes,
      });

      await cache.ensureRange(startTime, endTime, (request) =>
        fetchKlines({
          venue,
          symbol: market.symbol,
          interval,
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

      const merged = mergeWarmupCandles(candles, legacyMergedCandles);
      return merged.length > 0 ? merged : storedCandles;
    } catch {
      return legacyMergedCandles.length > 0 ? legacyMergedCandles : storedCandles;
    }
  }

  private async applyExchangeSnapshot(snapshot: BinancePaperSnapshot): Promise<BotEvent[]> {
    this.applyExchangeMaxLeverage(snapshot);
    this.applyExchangeTradingRules(snapshot);
    const reconciliation = exchangeReconciliationFromSnapshot(snapshot);
    const events: BotEvent[] = [];
    const exchangeDriven = this.isExchangeDrivenExecution();
    if (
      ((reconciliation.orders?.length ?? 0) > 0 ||
        (reconciliation.fills?.length ?? 0) > 0)
    ) {
      const reconciliationEvents = this.bot.applyExchangeReconciliation(reconciliation);
      this.noteExchangeExecutions(reconciliationEvents);
      events.push(...reconciliationEvents);
    }

    if (exchangeDriven) {
      this.exchangeAccountWarningMessage = undefined;
    } else {
      events.push(...this.applyExchangeAccountGuard(snapshot));
    }
    this.recordLiveEquity(Date.now(), this.bot.snapshot(), false, snapshot);
    this.recordEvents(events);
    await this.flushState();
    return events;
  }

  private applyExchangeAccountGuard(snapshot: BinancePaperSnapshot): BotEvent[] {
    if (this.isExchangeDrivenExecution() || !(this.paperTrading?.canSubmitOrders(this.market) ?? false)) {
      this.exchangeAccountWarningMessage = undefined;
      return [];
    }

    const driftMessage = exchangeAccountDriftMessage(
      snapshot,
      this.market,
      this.bot.snapshot(),
      {
        suppressTransientFuturesDrift: this.isExchangeSettlementGraceActive(),
      },
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

  private noteExchangeExecutions(events: readonly BotEvent[]): void {
    if (events.some((event) => event.type === "order_filled")) {
      this.lastExchangeExecutionAt = Date.now();
    }
  }

  private isExchangeSettlementGraceActive(): boolean {
    return Date.now() - this.lastExchangeExecutionAt <= EXCHANGE_ACCOUNT_GUARD_SETTLEMENT_GRACE_MS;
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
          historicalStartTime: options.historicalStartTime,
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

  private async loadRuntimeSettings(): Promise<void> {
    this.runtimeSettings = normalizeTradingRuntimeSettings(
      await this.storage.loadRuntimeSettings(),
    );
    this.applyRuntimeExchangeCredentials();
    this.executionMode = this.resolveExecutionMode(this.runtimeSettings.executionMode);
  }

  private resolveExecutionMode(saved: TradingExecutionMode | undefined): TradingExecutionMode {
    if (saved === "binance") {
      return this.canUseExchangeForBotExecution() ? "binance" : "simulated";
    }
    if (saved === "simulated") {
      return "simulated";
    }
    return this.paperTrading?.drivesOrderExecution(this.market) &&
      this.canUseExchangeForBotExecution()
      ? "binance"
      : "simulated";
  }

  private async saveExecutionMode(): Promise<void> {
    this.runtimeSettings = {
      ...this.runtimeSettings,
      executionMode: this.executionMode,
      updatedAt: Date.now(),
    };
    await this.saveRuntimeSettings();
  }

  private async saveRuntimeSettings(): Promise<void> {
    await this.storage.saveRuntimeSettings(this.runtimeSettings);
  }

  private applyRuntimeExchangeCredentials(): void {
    const exchange = this.runtimeSettings.exchange;
    if (!exchange || !this.paperTrading) {
      return;
    }

    const patch: Partial<BinancePaperConfig> = {
      enabled: true,
      mode: exchange.mode,
    };
    if (exchange.sandboxApiKey !== undefined) {
      patch.apiKey = exchange.sandboxApiKey;
    }
    if (exchange.sandboxApiSecret !== undefined) {
      patch.apiSecret = exchange.sandboxApiSecret;
    }
    if (exchange.liveApiKey !== undefined) {
      patch.liveApiKey = exchange.liveApiKey;
    }
    if (exchange.liveApiSecret !== undefined) {
      patch.liveApiSecret = exchange.liveApiSecret;
    }
    this.paperTrading.updateConfig(patch);
  }

  private isExchangeDrivenExecution(): boolean {
    return this.executionMode === "binance" && this.canUseExchangeForBotExecution();
  }

  private canUseExchangeForBotExecution(): boolean {
    return Boolean(
      this.paperTrading?.canSubmitOrders(this.market) &&
        isAutomatedExchangeExecutionVenue(this.market.venue),
    );
  }

  private exchangeExecutionUnavailableMessage(): string {
    if (!isAutomatedExchangeExecutionVenue(this.market.venue)) {
      return "Automated Binance execution is only enabled for USD-M and COIN-M futures markets.";
    }
    const snapshot = this.paperTrading?.snapshot(this.market);
    if (!snapshot?.enabled) {
      return "Binance exchange trading is disabled.";
    }
    if (!snapshot.compatible) {
      return snapshot.message;
    }
    if (!snapshot.configured) {
      return snapshot.message;
    }
    return "Binance exchange trading is not ready.";
  }

  private publicExecutionSnapshot(exchangeSnapshot: BinancePaperSnapshot): RuntimeSnapshot["execution"] {
    const exchangeDriven = this.isExchangeDrivenExecution();
    const canUseExchange = this.canUseExchangeForBotExecution();
    return {
      mode: this.executionMode,
      exchangeDriven,
      canUseExchange,
      live: exchangeSnapshot.live,
      message: executionModeMessage({
        mode: this.executionMode,
        exchangeDriven,
        canUseExchange,
        exchange: exchangeSnapshot,
        venue: this.market.venue,
      }),
    };
  }

  private createMarketConfig(
    savedState?: { config: StrategyConfig },
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

function exchangeDrivenPublicBotState(
  state: Readonly<PaperBotState>,
  snapshot: BinancePaperSnapshot,
  market: BinanceMarketListing,
): PaperBotState {
  const next = structuredClone(state) as PaperBotState;
  const equity = exchangeRuntimeEquity(snapshot, market, state.lastPrice);
  const exposureQuote = exchangeRuntimeExposureQuote(snapshot, market, state.lastPrice);
  const startingQuote = next.config.startingQuote;
  const netPnl =
    equity !== undefined && Number.isFinite(startingQuote)
      ? equity - startingQuote
      : 0;

  next.quoteFree = 0;
  next.quoteReserved = 0;
  next.baseFree = 0;
  next.baseReserved = 0;
  next.avgEntryPrice = 0;
  next.avgShortEntryPrice = 0;
  next.realizedPnl = state.realizedPnl;
  next.feesPaid = state.feesPaid;
  next.winningTrades = state.winningTrades;
  next.losingTrades = state.losingTrades;
  next.metrics = {
    ...next.metrics,
    equity: equity ?? next.metrics.equity,
    realizedPnl: state.realizedPnl,
    unrealizedPnl: 0,
    netPnl,
    returnPct:
      equity !== undefined && startingQuote > 0
        ? (netPnl / startingQuote) * 100
        : 0,
    feesPaid: state.feesPaid,
    tradeCount: state.fills.length,
    winningTrades: state.winningTrades,
    losingTrades: state.losingTrades,
    winRate:
      state.winningTrades + state.losingTrades > 0
        ? (state.winningTrades / (state.winningTrades + state.losingTrades)) * 100
        : 0,
    peakEquity: equity ?? 0,
    maxDrawdownPct: 0,
    exposurePct:
      equity !== undefined && equity > BALANCE_EPSILON
        ? (exposureQuote / equity) * 100
        : 0,
    maxEffectiveLeverage:
      equity !== undefined && equity > BALANCE_EPSILON
        ? exposureQuote / equity
        : 0,
  };
  return next;
}

function exchangeRuntimeEquity(
  snapshot: BinancePaperSnapshot | undefined,
  market: BinanceMarketListing,
  fallbackPrice: number,
): number | undefined {
  if (!snapshot) {
    return undefined;
  }
  if (market.venue === "usdm-futures" || market.venue === "coinm-futures") {
    return exchangeFuturesEquity(snapshot, market.quoteAsset);
  }

  const quote = exchangeSpotAssetTotal(snapshot, market.quoteAsset);
  const base = exchangeSpotAssetTotal(snapshot, market.baseAsset);
  const price = exchangeRuntimePrice(snapshot, fallbackPrice);
  return roundQuoteBalance(quote + base * price);
}

function exchangeRuntimePrice(
  snapshot: BinancePaperSnapshot | undefined,
  fallbackPrice: number,
): number {
  if (!snapshot) {
    return fallbackPrice;
  }

  const profile = futuresPositionProfile(snapshot);
  if (profile.markPrice > 0) {
    return profile.markPrice;
  }
  return fallbackPrice;
}

function exchangeRuntimeExposureQuote(
  snapshot: BinancePaperSnapshot,
  market: BinanceMarketListing,
  fallbackPrice: number,
): number {
  if (market.venue === "usdm-futures" || market.venue === "coinm-futures") {
    return snapshot.positions.reduce((sum, position) => {
      const notional = finiteOrZero(position.notional);
      if (notional !== 0) {
        return sum + Math.abs(notional);
      }
      const markPrice = finiteOrZero(position.markPrice) || fallbackPrice;
      return sum + Math.abs(finiteOrZero(position.positionAmt) * markPrice);
    }, 0);
  }

  return exchangeSpotAssetTotal(snapshot, market.baseAsset) * fallbackPrice;
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
  const orders = Array.isArray(state.orders) ? state.orders : [];
  const fills = Array.isArray(state.fills) ? state.fills : [];
  return (
    orders.length === 0 &&
    fills.length === 0 &&
    Math.abs(finiteOrZero(state.quoteReserved)) <= BALANCE_EPSILON &&
    Math.abs(finiteOrZero(state.baseFree)) <= BALANCE_EPSILON &&
    Math.abs(finiteOrZero(state.baseReserved)) <= BALANCE_EPSILON
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

function hasExchangeOpenOrdersOrPositions(snapshot: BinancePaperSnapshot): boolean {
  return (
    snapshot.openOrders.length > 0 ||
    snapshot.positions.some((position) => Math.abs(finiteOrZero(position.positionAmt)) > BALANCE_EPSILON)
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
  options: { suppressTransientFuturesDrift?: boolean } = {},
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
    return futuresAccountDriftMessage(snapshot, market, state, options);
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
  options: { suppressTransientFuturesDrift?: boolean } = {},
): string | undefined {
  const positionProfile = futuresPositionProfile(snapshot);
  if (
    positionProfile.longQuantity > BALANCE_EPSILON &&
    positionProfile.shortQuantity > BALANCE_EPSILON
  ) {
    return "Exchange account has simultaneous long and short hedge-mode positions that the local net-position guard cannot safely reconcile.";
  }

  if (
    snapshot.positionMode === "one-way" &&
    state.config.legacyValleyPeak.exitGridPositionMode === "per-lot"
  ) {
    const localLedger = analyzePositions(state);
    if (
      hasActivePositionLots(localLedger.longs) &&
      hasActivePositionLots(localLedger.shorts)
    ) {
      return "Local bot has simultaneous long and short per-lot exposure, but the Binance futures account is in one-way position mode. Enable hedge mode or reset to an aggregate/net strategy before enabling Binance execution.";
    }
  }

  if (options.suppressTransientFuturesDrift) {
    return undefined;
  }

  const localBase = roundAssetBalance(state.baseFree + state.baseReserved);
  const exchangeBase = roundAssetBalance(positionProfile.netQuantity);
  if (Math.abs(localBase - exchangeBase) > futuresAssetDriftTolerance(snapshot, exchangeBase)) {
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
    const tolerance = futuresEquityDriftTolerance(
      exchangeEquity,
      localBase,
      markPrice,
      positionProfile.markPrice > 0,
    );
    if (Math.abs(localEquity - exchangeEquity) > tolerance) {
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

function hasActivePositionLots(
  lots: Array<{ remainingQuantity: number; status?: string }>,
): boolean {
  return lots.some(
    (lot) =>
      lot.remainingQuantity > BALANCE_EPSILON &&
      lot.status !== "closed",
  );
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

function futuresEquityDriftTolerance(
  exchangeEquity: number,
  localBase: number,
  markPrice: number,
  hasReliableExchangeMarkPrice: boolean,
): number {
  const baseTolerance = quoteDriftTolerance(exchangeEquity);
  if (hasReliableExchangeMarkPrice || Math.abs(localBase) <= MIN_ASSET_DRIFT_TOLERANCE) {
    return baseTolerance;
  }

  const openNotional = Math.abs(localBase * markPrice);
  return Math.max(
    baseTolerance,
    openNotional * FUTURES_UNRELIABLE_MARK_EQUITY_DRIFT_TOLERANCE_RATE,
  );
}

function assetDriftTolerance(reference: number): number {
  return Math.max(MIN_ASSET_DRIFT_TOLERANCE, Math.abs(reference) * ASSET_DRIFT_TOLERANCE_RATE);
}

function futuresAssetDriftTolerance(
  snapshot: BinancePaperSnapshot,
  reference: number,
): number {
  return Math.max(
    assetDriftTolerance(reference),
    finiteOrZero(snapshot.symbolFilters?.stepSize),
    finiteOrZero(snapshot.symbolFilters?.marketStepSize),
  );
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
    live: false,
    autoSubmit: false,
    connected: false,
    message: "Binance exchange trading disabled",
    balances: [],
    positions: [],
    openOrders: [],
    recentOrders: [],
    recentTrades: [],
  };
}

function exchangeSubmitFailureReason(error: unknown): string {
  if (error instanceof BinancePaperOrderSubmissionSkipped) {
    return error.message;
  }
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

  const tradeQuantityByOrderId = new Map<string, { quantity: number; quoteQuantity: number }>();
  for (const trade of snapshot.recentTrades) {
    if (!trade.localOrderId) {
      continue;
    }
    const current = tradeQuantityByOrderId.get(trade.orderId) ?? {
      quantity: 0,
      quoteQuantity: 0,
    };
    current.quantity += trade.quantity;
    current.quoteQuantity += trade.quoteQuantity;
    tradeQuantityByOrderId.set(trade.orderId, current);
  }

  const orders = [...orderById.values()].map((order) => {
    const tradeTotals = tradeQuantityByOrderId.get(order.orderId);
    const filledQuantity = Math.min(
      order.executedQuantity,
      tradeTotals?.quantity ?? 0,
    );
    const status =
      order.executedQuantity > filledQuantity + 0.00000001
        ? "open"
        : normalizeExchangeOrderStatus(order.status);
    const averageTradePrice =
      tradeTotals && tradeTotals.quantity > 0
        ? tradeTotals.quoteQuantity / tradeTotals.quantity
        : undefined;

    return {
      localOrderId: order.localOrderId,
      externalOrderId: order.orderId,
      clientOrderId: order.clientOrderId,
      side: normalizeExchangeSide(order.side),
      type: normalizeExchangeType(order.type),
      status,
      price: averageTradePrice || order.avgPrice || order.price,
      quantity: order.originalQuantity,
      filledQuantity,
      quoteQuantity: tradeTotals?.quoteQuantity,
      createdAt: order.createdAt,
      updatedAt: order.updatedAt,
      positionEffect: futuresExchangeOrderPositionEffect(order),
      reason: `exchange order ${order.status}`,
    };
  });

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
        positionEffect: order
          ? futuresExchangeOrderPositionEffect(order)
          : futuresPositionEffectFromFields(trade.side, false, trade.positionSide),
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

function futuresExchangeOrderPositionEffect(
  order: BinancePaperOrder,
): "open" | "close" | undefined {
  return futuresPositionEffectFromFields(
    normalizeExchangeSide(order.side),
    order.reduceOnly === true,
    order.positionSide,
  );
}

function futuresPositionEffectFromFields(
  side: "buy" | "sell",
  reduceOnly: boolean,
  positionSide: string | undefined,
): "open" | "close" | undefined {
  if (reduceOnly) {
    return "close";
  }

  const normalizedPositionSide = (positionSide ?? "").toUpperCase();
  if (normalizedPositionSide === "LONG") {
    return side === "buy" ? "open" : "close";
  }
  if (normalizedPositionSide === "SHORT") {
    return side === "sell" ? "open" : "close";
  }
  return undefined;
}

function normalizeExchangeType(type: string): "limit" | "market" | "stop-market" {
  const upper = type.toUpperCase();
  if (upper === "MARKET") {
    return "market";
  }
  if (upper === "STOP_MARKET" || upper === "STOP_LOSS") {
    return "stop-market";
  }
  return "limit";
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

function normalizeTradingExecutionMode(value: unknown): TradingExecutionMode | undefined {
  return value === "binance" || value === "simulated" ? value : undefined;
}

function normalizeTradingExchangeMode(value: unknown): TradingExchangeMode | undefined {
  return value === "auto" ||
    value === "live" ||
    value === "spot-live" ||
    value === "usdm-futures-live" ||
    value === "coinm-futures-live" ||
    value === "spot-testnet" ||
    value === "spot-demo" ||
    value === "usdm-futures-testnet" ||
    value === "coinm-futures-testnet"
    ? value
    : undefined;
}

function normalizeTradingRuntimeSettings(
  value: TradingRuntimeSettings | undefined,
): TradingRuntimeSettings {
  const exchange = normalizeTradingExchangeCredentials(value?.exchange);
  return {
    executionMode: normalizeTradingExecutionMode(value?.executionMode) ?? "simulated",
    ...(exchange ? { exchange } : {}),
    updatedAt: finiteOrZero(value?.updatedAt) || 0,
  };
}

function normalizeTradingExchangeCredentials(
  value: TradingExchangeCredentials | undefined,
): TradingExchangeCredentials | undefined {
  if (!value) {
    return undefined;
  }
  const mode = normalizeTradingExchangeMode(value.mode);
  if (!mode) {
    return undefined;
  }
  return {
    mode,
    sandboxApiKey: normalizeCredentialInput(value.sandboxApiKey),
    sandboxApiSecret: normalizeCredentialInput(value.sandboxApiSecret),
    liveApiKey: normalizeCredentialInput(value.liveApiKey),
    liveApiSecret: normalizeCredentialInput(value.liveApiSecret),
    updatedAt: finiteOrZero(value.updatedAt) || 0,
  };
}

function normalizeCredentialInput(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

function exchangeModeUsesLiveCredentials(mode: TradingExchangeMode): boolean {
  return mode === "live" || mode.endsWith("-live");
}

function isAutomatedExchangeExecutionVenue(venue: string): boolean {
  return venue === "usdm-futures" || venue === "coinm-futures";
}

function assertExchangeAccountFlatForExecutionSwitch(
  snapshot: BinancePaperSnapshot,
  market: BinanceMarketListing,
): void {
  if (snapshot.openOrders.length > 0) {
    throw new Error(
      `Cancel ${snapshot.openOrders.length} open ${market.symbol} Binance order(s) before enabling Binance execution.`,
    );
  }
  if (hasExchangeOpenOrdersOrPositions(snapshot)) {
    throw new Error(
      `Close the open ${market.symbol} Binance position before enabling Binance execution.`,
    );
  }
  const baseBalance = exchangeAssetBalance(snapshot, market.baseAsset) ?? 0;
  if (Math.abs(baseBalance) > BALANCE_EPSILON) {
    throw new Error(
      `Clear the ${market.baseAsset} Binance balance before enabling Binance execution for ${market.symbol}.`,
    );
  }
}

function executionModeMessage(options: {
  mode: TradingExecutionMode;
  exchangeDriven: boolean;
  canUseExchange: boolean;
  exchange: BinancePaperSnapshot;
  venue: string;
}): string {
  if (options.exchangeDriven) {
    const prefix = options.exchange.live ? "Live" : "Sandbox";
    return `${prefix} Binance execution active`;
  }
  if (options.mode === "binance") {
    if (!isAutomatedExchangeExecutionVenue(options.venue)) {
      return "Binance bot execution requires a futures market.";
    }
    if (!options.canUseExchange) {
      return options.exchange.message;
    }
    return "Binance execution selected but not active.";
  }
  return "Bot orders execute in the local simulator.";
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

async function readLegacyHistoricalCandles(
  dataDir: string,
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number,
): Promise<Candle[]> {
  const root = path.join(
    dataDir,
    "historical",
    safeHistoricalPathPart(symbol),
    safeHistoricalPathPart(interval),
  );
  let files: string[];
  try {
    files = await fs.readdir(root);
  } catch {
    return [];
  }

  const candles: Candle[] = [];
  for (const file of files.sort()) {
    if (!file.endsWith(".jsonl")) {
      continue;
    }
    const filePath = path.join(root, file);
    let content: string;
    try {
      content = await fs.readFile(filePath, "utf8");
    } catch {
      continue;
    }

    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        const candle = JSON.parse(trimmed) as Candle;
        if (
          candle.closed &&
          candle.interval === interval &&
          candle.symbol.toUpperCase() === symbol.toUpperCase() &&
          candle.openTime >= startTime &&
          candle.closeTime <= endTime
        ) {
          candles.push(candle);
        }
      } catch {
        continue;
      }
    }
  }

  return candles.sort((left, right) => left.openTime - right.openTime);
}

function candleSpanMs(candles: readonly Candle[]): number {
  if (candles.length === 0) {
    return 0;
  }
  return candles.at(-1)!.closeTime - candles[0].openTime;
}

function safeHistoricalPathPart(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9._-]+/g, "-");
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
