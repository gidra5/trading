import type { TradingBotConfig } from "./bot.js";
import {
  EMAIndicator,
  KAMAIndicator,
  LookbackIndicator,
  SMAIndicator,
  type EMAIndicatorSnapshot,
  type KAMAIndicatorSnapshot,
  type LookbackIndicatorSnapshot,
  type PriceIndicatorInput,
  type SMAIndicatorSnapshot,
} from "./indicators.js";
import type {
  PositionSide,
  StrategyDiagnostics,
  StrategyOptions,
  TradingStrategy,
  TradingStrategyEntrySignal,
  TradingStrategyExitSignal,
} from "./strategy.js";
import type { TradingApi, TradingCandle, TradingTick } from "./trading-api.js";

const SAMPLE_INTERVAL_MS = 60_000;
const PRICE_SCALE = 100_000;

export type PeakValleyAverageType = "sma" | "ema";
export type PeakValleyDerivativeSource = "price" | "kama";
export type PeakValleyDerivativeClampMode = "deadband" | "hysteresis";
export type PeakValleySignalTiming = "start" | "end";
export type PeakValleySigmaMode = "static" | "trend" | "sigmoid-trend";

export interface PeakValleyStrategyConfig {
  averagingRangesSec: number[];
  movingAverageType: PeakValleyAverageType;
  relativeRateEnabled: boolean;
  derivativeSource: PeakValleyDerivativeSource;
  derivativeClampMode: PeakValleyDerivativeClampMode;
  derivativeClampInnerThresholdRatio: number;
  kamaErLen: number;
  kamaFastLen: number;
  kamaSlowLen: number;
  kamaPower: number;
  rateThresholdsLow: number[];
  rateThresholdsHigh: number[];
  buyDataIndex: number;
  sellDataIndex: number;
  buyConfirmationOffsets: number[];
  sellConfirmationOffsets: number[];
  buyExitConfirmationOffsets: number[];
  sellExitConfirmationOffsets: number[];
  buyEntrySignalTiming: PeakValleySignalTiming;
  sellEntrySignalTiming: PeakValleySignalTiming;
  buyExitSignalTiming: PeakValleySignalTiming;
  sellExitSignalTiming: PeakValleySignalTiming;
  saturationSec: number;
  buySpendRate: number;
  sellAmountRate: number;
  sigmaMode: PeakValleySigmaMode;
  buySigma: number;
  sellSigma: number;
  trendSigmaA: number;
  trendSigmaSellB1: number;
  trendSigmaBuyB2: number;
  trendSigmaWindowSec: number;
  sigmoidSigmaLow: number;
  sigmoidSigmaHigh: number;
  longSideEnabled: boolean;
  shortSideEnabled: boolean;
}

export const defaultPeakValleyStrategyConfig: PeakValleyStrategyConfig = {
  averagingRangesSec: [1, 60, 600, 1_800, 3_600, 14_400, 43_200],
  movingAverageType: "sma",
  relativeRateEnabled: true,
  derivativeSource: "price",
  derivativeClampMode: "deadband",
  derivativeClampInnerThresholdRatio: 0,
  kamaErLen: 20,
  kamaFastLen: 5,
  kamaSlowLen: 50,
  kamaPower: 1,
  rateThresholdsLow: [0.25, 0.25, 0.25, 0.25, 0.15, 0.05, 0.05],
  rateThresholdsHigh: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
  buyDataIndex: 1,
  sellDataIndex: 1,
  buyConfirmationOffsets: [1, 2],
  sellConfirmationOffsets: [1, 2],
  buyExitConfirmationOffsets: [1, 2],
  sellExitConfirmationOffsets: [1, 2],
  buyEntrySignalTiming: "start",
  sellEntrySignalTiming: "start",
  buyExitSignalTiming: "start",
  sellExitSignalTiming: "start",
  saturationSec: 3_600,
  buySpendRate: 1,
  sellAmountRate: 1,
  sigmaMode: "trend",
  buySigma: 0.3,
  sellSigma: 0.1,
  trendSigmaA: 1,
  trendSigmaSellB1: 1,
  trendSigmaBuyB2: 1,
  trendSigmaWindowSec: 3_600,
  sigmoidSigmaLow: 0.05,
  sigmoidSigmaHigh: 0.3,
  longSideEnabled: true,
  shortSideEnabled: true,
};

export function createPeakValleyStrategyConfig(
  overrides: Partial<PeakValleyStrategyConfig> = {},
): PeakValleyStrategyConfig {
  const merged = { ...defaultPeakValleyStrategyConfig, ...overrides };
  const config = Object.fromEntries(
    Object.keys(defaultPeakValleyStrategyConfig).map((key) => [
      key,
      merged[key as keyof PeakValleyStrategyConfig],
    ]),
  ) as unknown as PeakValleyStrategyConfig;
  const count = Math.max(1, config.averagingRangesSec.length);
  config.averagingRangesSec = pad(config.averagingRangesSec, count, 1).map(positiveInt);
  config.rateThresholdsLow = normalizeRates(config.rateThresholdsLow, count, config.relativeRateEnabled);
  config.rateThresholdsHigh = normalizeRates(config.rateThresholdsHigh, count, config.relativeRateEnabled);
  config.buyDataIndex = arrayIndex(config.buyDataIndex, count);
  config.sellDataIndex = arrayIndex(config.sellDataIndex, count);
  config.buyConfirmationOffsets = offsets(config.buyConfirmationOffsets);
  config.sellConfirmationOffsets = offsets(config.sellConfirmationOffsets);
  config.buyExitConfirmationOffsets = offsets(config.buyExitConfirmationOffsets);
  config.sellExitConfirmationOffsets = offsets(config.sellExitConfirmationOffsets);
  config.derivativeClampInnerThresholdRatio = clamp01(config.derivativeClampInnerThresholdRatio);
  config.kamaErLen = positiveInt(config.kamaErLen);
  config.kamaFastLen = positiveInt(config.kamaFastLen);
  config.kamaSlowLen = positiveInt(config.kamaSlowLen);
  config.kamaPower = Math.max(0.1, finite(config.kamaPower, 1));
  config.saturationSec = Math.max(0, finite(config.saturationSec, 0));
  config.buySpendRate = Math.max(0, finite(config.buySpendRate, 0));
  config.sellAmountRate = Math.max(0, finite(config.sellAmountRate, 0));
  config.buySigma = positive(config.buySigma);
  config.sellSigma = positive(config.sellSigma);
  config.trendSigmaA = positive(config.trendSigmaA);
  config.trendSigmaWindowSec = positive(config.trendSigmaWindowSec);
  config.sigmoidSigmaLow = positive(config.sigmoidSigmaLow);
  config.sigmoidSigmaHigh = Math.max(config.sigmoidSigmaLow, positive(config.sigmoidSigmaHigh));
  return config;
}

interface PeakValleyExecutionConfig {
  minTradeQuote?: number;
  maxTradeQuote?: number;
  exitGridOrderCount?: number;
  exitGridMaxStepPct?: number;
  exitGridSizeDistribution?: "geometric" | "linear" | "constant";
  exitGridSellFraction?: number;
  exitGridResetMode?: "higher-peak" | "filled-grid";
  anticipatoryGridOrderCount?: number;
}

export interface PeakValleyBotConfigSource {
  maxLeverage: number;
  minOrderQuote: number;
  maxPositionQuote: number;
  cooldownMs: number;
  internalBorrowAccounting: "active" | "inactive";
  borrowerProfitShareToLender: number;
  legacyValleyPeak: Partial<PeakValleyStrategyConfig & PeakValleyExecutionConfig>;
}

export type PeakValleyBotConfig = TradingBotConfig<PeakValleyStrategyConfig>;

export function createPeakValleyBotConfig(config: PeakValleyBotConfigSource): PeakValleyBotConfig {
  const source = config.legacyValleyPeak;
  const strategy = createPeakValleyStrategyConfig(source);
  const minTrade = Math.max(config.minOrderQuote, source.minTradeQuote ?? 0);
  const maxTrade = Math.max(minTrade, Math.min(config.maxPositionQuote, source.maxTradeQuote ?? Infinity));
  const distribution = source.exitGridSizeDistribution === "linear" ? "linear" : "geometric";
  return {
    strategy,
    maxTargetLeverage: config.maxLeverage,
    minTradeQuote: minTrade,
    maxTradeQuote: maxTrade,
    entryGrid: {
      orderCount: Math.max(1, Math.round(source.anticipatoryGridOrderCount ?? 1)),
      maxPriceStep: Math.max(0, source.exitGridMaxStepPct ?? 0),
      sizeDistribution: distribution,
      sizeFraction: clamp(source.exitGridSellFraction ?? 1, 0.01, 1),
    },
    exitGrid: {
      orderCount: Math.max(1, Math.round(source.exitGridOrderCount ?? 1)),
      maxPriceStep: Math.max(0, source.exitGridMaxStepPct ?? 0),
      sizeDistribution: distribution,
      sizeFraction: clamp(source.exitGridSellFraction ?? 1, 0.01, 1),
      reset: source.exitGridResetMode === "filled-grid" ? "last-filled-order" : "previous-anchor",
    },
    positionLifetimeMs: null,
    stopLossRate: null,
    takeProfitRate: null,
    cooldownMs: config.cooldownMs,
    internalBorrow: {
      enabled: config.internalBorrowAccounting === "active",
      lockLenderAmounts: true,
      borrowerProfitShare: clamp01(config.borrowerProfitShareToLender),
    },
  };
}

type AverageSnapshot =
  | { type: "sma"; indicator: SMAIndicatorSnapshot; rate: number; clamped: LookbackIndicatorSnapshot }
  | { type: "ema"; indicator: EMAIndicatorSnapshot; rate: number; clamped: LookbackIndicatorSnapshot };

type AverageState = ({ type: "sma"; indicator: SMAIndicator } | { type: "ema"; indicator: EMAIndicator }) & {
  rate: number;
  clamped: LookbackIndicator;
};

export interface PeakValleyStrategySnapshot {
  version: 2;
  averages: AverageSnapshot[];
  kama: KAMAIndicatorSnapshot | null;
  kamaRate: number;
  kamaClamped: LookbackIndicatorSnapshot;
  lastTick: TradingTick | null;
  startedAt: number;
  ready: boolean;
  lastSignal: StrategyDiagnostics["lastSignal"];
}

export interface PeakValleyAverageDiagnostics {
  index: number;
  windowSec: number;
  value: number;
  rawRate: number;
  clampedRate: number;
  previousClampedRate: number;
  thresholdLow: number;
  thresholdHigh: number;
  roles: {
    buyPrimary: boolean;
    sellPrimary: boolean;
    buyEntryConfirmation: boolean;
    sellEntryConfirmation: boolean;
    buyExitConfirmation: boolean;
    sellExitConfirmation: boolean;
    sizing: boolean;
    trendSigma: boolean;
  };
}

export interface PeakValleyStrategyDiagnostics extends StrategyDiagnostics {
  ready: boolean;
  warmupRemainingMs: number;
  movingAverageType: PeakValleyAverageType;
  derivativeSource: PeakValleyDerivativeSource;
  latestTick: TradingTick | null;
  averages: readonly PeakValleyAverageDiagnostics[];
  kama: {
    value: number;
    rawRate: number;
    clampedRate: number;
    previousClampedRate: number;
  } | null;
  sizing: {
    buy: { size: number; sigma: number };
    sell: { size: number; sigma: number };
  };
}

export class PeakValleyStrategy
  implements TradingStrategy<PeakValleyStrategyConfig, PeakValleyStrategySnapshot, PeakValleyStrategyDiagnostics>
{
  private config: PeakValleyStrategyConfig;
  private readonly historyApi: TradingApi;
  private averages: AverageState[] = [];
  private kama!: KAMAIndicator;
  private kamaRate = 0;
  private kamaClamped = new LookbackIndicator(1);
  private lastTick: TradingTick | null = null;
  private startedAt = 0;
  private ready = false;
  private diagnostics: PeakValleyStrategyDiagnostics;

  constructor(private readonly options: StrategyOptions<PeakValleyStrategyConfig>) {
    this.config = createPeakValleyStrategyConfig(options.config);
    this.historyApi = { getHistory: options.getHistory } as TradingApi;
    this.rebuildIndicators();
    this.diagnostics = emptyDiagnostics(this.config);
    this.diagnostics = this.buildDiagnostics();
  }

  async warmup(): Promise<void> {
    const count = Math.max(
      ...this.config.averagingRangesSec.map(samplesForWindow),
      this.config.kamaErLen + 1,
      this.config.kamaSlowLen,
    );
    const candles = await this.options.getHistory({ intervalMs: SAMPLE_INTERVAL_MS, count });
    for (const candle of candles) this.updateIndicators(candleTick(candle));
    const span = candles.length > 1 ? (candles.at(-1)!.closeTime - candles[0].openTime) / 1_000 : 0;
    this.ready = span >= this.config.saturationSec || candles.length >= count;
    this.retireTransition();
    this.diagnostics = this.buildDiagnostics();
  }

  async onTick(tick: TradingTick): Promise<void> {
    if (!Number.isFinite(tick.price) || tick.price <= 0) return;
    if (this.lastTick && tick.candle === null && tick.timestamp - this.lastTick.timestamp < SAMPLE_INTERVAL_MS) {
      this.retireTransition();
      return;
    }
    this.updateIndicators(tick);
    this.diagnostics = this.buildDiagnostics();
  }

  async entrySignal(): Promise<TradingStrategyEntrySignal | null> {
    if (!this.lastTick) return null;
    const valley = this.ready && this.check("valley", this.config.buyDataIndex, this.config.buyConfirmationOffsets, this.config.buyEntrySignalTiming);
    const peak = this.ready && this.check("peak", this.config.sellDataIndex, this.config.sellConfirmationOffsets, this.config.sellEntrySignalTiming);
    const signal = valley && this.config.longSideEnabled
      ? entry("long", this.size("buy"), this.lastTick.price)
      : peak && this.config.shortSideEnabled
        ? entry("short", this.size("sell"), this.lastTick.price)
        : null;
    this.recordDecision("entry", signal, valley ? "valley" : peak ? "peak" : null, [
      gate("long.entry", valley, this.primaryRate(this.config.buyDataIndex), this.config.rateThresholdsLow[this.config.buyDataIndex]),
      ...this.decisionGates("long.entry", "valley", this.config.buyDataIndex, this.config.buyConfirmationOffsets, this.config.buyEntrySignalTiming),
      gate("short.entry", peak, this.primaryRate(this.config.sellDataIndex), this.config.rateThresholdsLow[this.config.sellDataIndex]),
      ...this.decisionGates("short.entry", "peak", this.config.sellDataIndex, this.config.sellConfirmationOffsets, this.config.sellEntrySignalTiming),
    ]);
    return signal;
  }

  async exitSignal(): Promise<TradingStrategyExitSignal | null> {
    if (!this.lastTick) return null;
    const valley = this.ready && this.check("valley", this.config.buyDataIndex, this.config.buyExitConfirmationOffsets, this.config.buyExitSignalTiming);
    const peak = this.ready && this.check("peak", this.config.sellDataIndex, this.config.sellExitConfirmationOffsets, this.config.sellExitSignalTiming);
    const signal = valley && this.config.shortSideEnabled
      ? exit("short", this.size("buy"), this.lastTick.price)
      : peak && this.config.longSideEnabled
        ? exit("long", this.size("sell"), this.lastTick.price)
        : null;
    this.recordDecision("exit", signal, valley ? "valley" : peak ? "peak" : null, [
      gate("short.exit", valley, this.primaryRate(this.config.buyDataIndex), this.config.rateThresholdsLow[this.config.buyDataIndex]),
      ...this.decisionGates("short.exit", "valley", this.config.buyDataIndex, this.config.buyExitConfirmationOffsets, this.config.buyExitSignalTiming),
      gate("long.exit", peak, this.primaryRate(this.config.sellDataIndex), this.config.rateThresholdsLow[this.config.sellDataIndex]),
      ...this.decisionGates("long.exit", "peak", this.config.sellDataIndex, this.config.sellExitConfirmationOffsets, this.config.sellExitSignalTiming),
    ]);
    return signal;
  }

  async snapshot(): Promise<PeakValleyStrategySnapshot> {
    return {
      version: 2,
      averages: this.averages.map(snapshotAverage),
      kama: this.config.derivativeSource === "kama" ? this.kama.snapshot() : null,
      kamaRate: this.kamaRate,
      kamaClamped: this.kamaClamped.snapshot(),
      lastTick: structuredClone(this.lastTick),
      startedAt: this.startedAt,
      ready: this.ready,
      lastSignal: structuredClone(this.diagnostics.lastSignal),
    };
  }

  async restore(snapshot: PeakValleyStrategySnapshot): Promise<void> {
    if (snapshot.version !== 2) throw new Error(`Unsupported peak/valley snapshot version: ${snapshot.version}`);
    for (let position = 0; position < this.averages.length; position += 1) {
      const state = this.averages[position];
      const saved = snapshot.averages[position];
      if (!saved || saved.type !== state.type) continue;
      restoreAverage(state, saved);
      state.rate = saved.rate;
    }
    if (snapshot.kama) this.kama.restore(snapshot.kama);
    this.kamaRate = finite(snapshot.kamaRate, 0);
    this.kamaClamped.restore(snapshot.kamaClamped);
    this.lastTick = structuredClone(snapshot.lastTick);
    this.startedAt = finite(snapshot.startedAt, this.lastTick?.timestamp ?? 0);
    this.ready = snapshot.ready;
    this.diagnostics = { ...this.buildDiagnostics(), lastSignal: structuredClone(snapshot.lastSignal) };
  }

  async updateConfig(config: PeakValleyStrategyConfig): Promise<void> {
    this.config = createPeakValleyStrategyConfig(config);
    this.rebuildIndicators();
    this.lastTick = null;
    this.startedAt = 0;
    this.ready = false;
    await this.warmup();
  }

  getDiagnostics(): PeakValleyStrategyDiagnostics {
    return this.diagnostics;
  }

  private rebuildIndicators(): void {
    this.averages = this.config.averagingRangesSec.map((windowSec) => {
      const samples = samplesForWindow(windowSec);
      return this.config.movingAverageType === "ema"
        ? { type: "ema", indicator: new EMAIndicator(samples), rate: 0, clamped: new LookbackIndicator(1) }
        : { type: "sma", indicator: new SMAIndicator(samples, this.historyApi), rate: 0, clamped: new LookbackIndicator(1) };
    });
    this.kama = new KAMAIndicator(
      this.config.kamaErLen,
      this.config.kamaFastLen,
      this.config.kamaSlowLen,
      this.historyApi,
      Math.max(this.config.kamaSlowLen, this.config.kamaErLen + 1),
      this.config.kamaPower,
    );
    this.kamaClamped = new LookbackIndicator(1);
  }

  private updateIndicators(tick: TradingTick): void {
    const seconds = Math.max(1, (tick.timestamp - (this.lastTick?.timestamp ?? tick.timestamp - SAMPLE_INTERVAL_MS)) / 1_000);
    const input = indicatorInput(tick);
    this.startedAt ||= tick.timestamp;
    this.lastTick = tick;
    for (let position = 0; position < this.averages.length; position += 1) {
      const state = this.averages[position];
      state.indicator.onTick(input);
      state.rate = rate(state.indicator.derivative(), state.indicator.indicator(), seconds, this.config.relativeRateEnabled);
      state.clamped.onTick({
        eventTime: tick.timestamp,
        value: clampRate(state.rate, state.clamped.indicator(), this.config, position),
      });
    }
    if (this.config.derivativeSource === "kama") {
      this.kama.onTick(input);
      this.kamaRate = rate(this.kama.derivative(), this.kama.indicator(), seconds, this.config.relativeRateEnabled);
      this.kamaClamped.onTick({
        eventTime: tick.timestamp,
        value: clampRate(this.kamaRate, this.kamaClamped.indicator(), this.config, this.config.buyDataIndex),
      });
    }
    this.ready ||= tick.timestamp - this.startedAt >= this.config.saturationSec * 1_000;
  }

  private retireTransition(): void {
    const eventTime = this.lastTick?.timestamp ?? 0;
    for (const state of this.averages) {
      state.clamped.onTick({ eventTime, value: state.clamped.indicator() });
    }
    this.kamaClamped.onTick({ eventTime, value: this.kamaClamped.indicator() });
  }

  private recordDecision(
    type: "entry" | "exit",
    signal: TradingStrategyEntrySignal | TradingStrategyExitSignal | null,
    reason: string | null,
    gates: StrategyDiagnostics["gates"],
  ): void {
    const codes = new Set(gates.map((item) => item.code));
    this.diagnostics = {
      ...this.buildDiagnostics(),
      gates: [...this.diagnostics.gates.filter((item) => !codes.has(item.code)), ...gates],
      lastSignal: signal && reason
        ? { type, side: signal.side, reason }
        : this.diagnostics.lastSignal,
    };
  }

  private check(
    shape: "valley" | "peak",
    primaryIndex: number,
    confirmationOffsets: number[],
    timing: PeakValleySignalTiming,
  ): boolean {
    return this.decisionGates("signal", shape, primaryIndex, confirmationOffsets, timing)
      .every((item) => item.passed);
  }

  private decisionGates(
    code: string,
    shape: "valley" | "peak",
    primaryIndex: number,
    confirmationOffsets: number[],
    timing: PeakValleySignalTiming,
  ): StrategyDiagnostics["gates"] {
    const primary = this.config.derivativeSource === "kama"
      ? { current: this.kamaClamped.indicator(), previous: this.kamaClamped.previous() }
      : {
          current: this.averages[primaryIndex]?.clamped.indicator() ?? 0,
          previous: this.averages[primaryIndex]?.clamped.previous() ?? 0,
        };
    return [
      gate(`${code}.source.${this.config.derivativeSource}`, extremum(shape, timing, primary.previous, primary.current), primary.current, primary.previous),
      ...confirmationOffsets.flatMap((offset) => {
        const confirmation = this.averages[primaryIndex + offset];
        const value = confirmation?.clamped.indicator();
        return value === undefined
          ? []
          : [gate(`${code}.confirmation.${offset}`, shape === "valley" ? value > 0 : value < 0, value, 0)];
      }),
    ];
  }

  private size(side: "buy" | "sell"): number {
    const sizingRate = this.averages[Math.min(3, this.averages.length - 1)]?.rate ?? 0;
    const sizingFraction = side === "buy" ? this.config.buySpendRate : this.config.sellAmountRate;
    return clamp01(sizingFraction * gaussian(sizingRate, this.sigma(side)));
  }

  private sigma(side: "buy" | "sell"): number {
    if (this.config.sigmaMode === "static") {
      return normalizedSigma(side === "buy" ? this.config.buySigma : this.config.sellSigma, this.config.relativeRateEnabled);
    }
    const trend = this.averages[closestWindow(this.config.averagingRangesSec, this.config.trendSigmaWindowSec)]?.rate ?? 0;
    const scaled = this.config.relativeRateEnabled ? trend * PRICE_SCALE : trend;
    if (this.config.sigmaMode === "sigmoid-trend") {
      const direction = side === "buy" ? -scaled * this.config.trendSigmaBuyB2 : scaled * this.config.trendSigmaSellB1;
      const weight = 1 / (1 + Math.exp(clamp(direction, -50, 50)));
      return normalizedSigma(
        this.config.sigmoidSigmaLow * weight + this.config.sigmoidSigmaHigh * (1 - weight),
        this.config.relativeRateEnabled,
      );
    }
    const exponent = side === "buy" ? scaled * this.config.trendSigmaBuyB2 : -scaled * this.config.trendSigmaSellB1;
    return normalizedSigma(this.config.trendSigmaA, this.config.relativeRateEnabled) * Math.exp(clamp(exponent, -50, 50));
  }

  private buildDiagnostics(): PeakValleyStrategyDiagnostics {
    const sizingIndex = Math.min(3, this.averages.length - 1);
    const trendIndex = closestWindow(this.config.averagingRangesSec, this.config.trendSigmaWindowSec);
    const warmupElapsedMs = this.lastTick ? this.lastTick.timestamp - this.startedAt : 0;
    return {
      indicators: Object.fromEntries([
        ...this.averages.flatMap((state, position) => [
          [`average.${this.config.averagingRangesSec[position]}`, state.indicator.indicator()],
          [`rate.${this.config.averagingRangesSec[position]}`, state.clamped.indicator()],
        ]),
        ...(this.config.derivativeSource === "kama"
          ? [["kama", this.kama.indicator()], ["kama.rate", this.kamaClamped.indicator()]]
          : []),
      ]),
      gates: [],
      blockers: this.ready ? [] : ["warmup"],
      lastSignal: this.diagnostics.lastSignal,
      ready: this.ready,
      warmupRemainingMs: this.ready
        ? 0
        : Math.max(0, this.config.saturationSec * 1_000 - warmupElapsedMs),
      movingAverageType: this.config.movingAverageType,
      derivativeSource: this.config.derivativeSource,
      latestTick: structuredClone(this.lastTick),
      averages: this.averages.map((state, index) => ({
        index,
        windowSec: this.config.averagingRangesSec[index],
        value: state.indicator.indicator(),
        rawRate: state.rate,
        clampedRate: state.clamped.indicator(),
        previousClampedRate: state.clamped.previous(),
        thresholdLow: this.config.rateThresholdsLow[index] ?? 0,
        thresholdHigh: this.config.rateThresholdsHigh[index] ?? 0,
        roles: {
          buyPrimary: index === this.config.buyDataIndex,
          sellPrimary: index === this.config.sellDataIndex,
          buyEntryConfirmation: confirms(index, this.config.buyDataIndex, this.config.buyConfirmationOffsets),
          sellEntryConfirmation: confirms(index, this.config.sellDataIndex, this.config.sellConfirmationOffsets),
          buyExitConfirmation: confirms(index, this.config.buyDataIndex, this.config.buyExitConfirmationOffsets),
          sellExitConfirmation: confirms(index, this.config.sellDataIndex, this.config.sellExitConfirmationOffsets),
          sizing: index === sizingIndex,
          trendSigma: this.config.sigmaMode !== "static" && index === trendIndex,
        },
      })),
      kama: this.config.derivativeSource === "kama"
        ? {
            value: this.kama.indicator(),
            rawRate: this.kamaRate,
            clampedRate: this.kamaClamped.indicator(),
            previousClampedRate: this.kamaClamped.previous(),
          }
        : null,
      sizing: {
        buy: { size: this.size("buy"), sigma: this.sigma("buy") },
        sell: { size: this.size("sell"), sigma: this.sigma("sell") },
      },
    };
  }

  private primaryRate(position: number): number {
    return this.config.derivativeSource === "kama"
      ? this.kamaClamped.indicator()
      : this.averages[position]?.clamped.indicator() ?? 0;
  }
}

function snapshotAverage(state: AverageState): AverageSnapshot {
  const shared = { rate: state.rate, clamped: state.clamped.snapshot() };
  return state.type === "sma"
    ? { type: "sma", indicator: state.indicator.snapshot(), ...shared }
    : { type: "ema", indicator: state.indicator.snapshot(), ...shared };
}

function restoreAverage(state: AverageState, snapshot: AverageSnapshot): void {
  if (state.type === "sma" && snapshot.type === "sma") state.indicator.restore(snapshot.indicator);
  if (state.type === "ema" && snapshot.type === "ema") state.indicator.restore(snapshot.indicator);
  state.clamped.restore(snapshot.clamped);
}

function indicatorInput(tick: TradingTick): PriceIndicatorInput {
  return tick.candle
    ? { eventTime: tick.timestamp, candle: tick.candle }
    : { eventTime: tick.timestamp, price: tick.price, quantity: tick.quantity };
}

function candleTick(candle: TradingCandle): TradingTick {
  return { timestamp: candle.closeTime, price: candle.close, quantity: candle.volume, candle };
}

function entry(side: PositionSide, size: number, price: number | null): TradingStrategyEntrySignal | null {
  return size > 0 ? { side, size, leverage: 999, price, confidence: null } : null;
}

function exit(side: PositionSide, size: number, price: number | null): TradingStrategyExitSignal | null {
  return size > 0 ? { side, size, price, confidence: null } : null;
}

function extremum(shape: "valley" | "peak", timing: PeakValleySignalTiming, previous: number, current: number): boolean {
  if (shape === "valley") return timing === "start" ? current >= 0 && previous < 0 : current > 0 && previous <= 0;
  return timing === "start" ? current <= 0 && previous > 0 : current < 0 && previous >= 0;
}

function clampRate(value: number, previous: number, config: PeakValleyStrategyConfig, position: number): number {
  const low = config.rateThresholdsLow[position] ?? 0;
  const high = config.rateThresholdsHigh[position] ?? low;
  const release = low * config.derivativeClampInnerThresholdRatio;
  if (
    config.derivativeClampMode === "hysteresis" &&
    previous !== 0 &&
    Math.sign(previous) === Math.sign(value) &&
    Math.abs(value) >= release
  ) return value;
  const threshold = config.derivativeClampMode === "hysteresis" && previous === 0 ? high : low;
  return Math.abs(value) >= threshold ? value : 0;
}

function rate(delta: number, value: number, seconds: number, relative: boolean): number {
  const absolute = delta / seconds;
  return relative && value > 0 ? absolute / value : absolute;
}

function normalizedSigma(value: number, relative: boolean): number {
  const result = positive(value);
  return relative && result > 0.01 ? result / PRICE_SCALE : result;
}

function gaussian(value: number, sigma: number): number {
  const normalized = value / Math.max(Number.EPSILON, sigma);
  return Math.exp(-0.5 * normalized * normalized);
}

function gate(code: string, passed: boolean, value: number, threshold: number) {
  return { code, passed, value, threshold };
}

function samplesForWindow(seconds: number): number {
  return Math.max(1, Math.ceil(seconds * 1_000 / SAMPLE_INTERVAL_MS));
}

function closestWindow(windows: number[], target: number): number {
  return windows.reduce((best, value, position) =>
    Math.abs(value - target) < Math.abs((windows[best] ?? value) - target) ? position : best, 0);
}

function normalizeRates(values: number[], count: number, relative: boolean): number[] {
  return pad(values, count, relative ? 0.0000025 : 0.25).map((value) =>
    relative && value > 0.01 ? value / PRICE_SCALE : Math.max(0, finite(value, 0)));
}

function pad(values: number[], count: number, fallback: number): number[] {
  return Array.from({ length: count }, (_, position) => finite(values[position], values.at(-1) ?? fallback));
}

function offsets(values: number[]): number[] {
  return values.map((value) => Math.max(0, Math.round(value))).filter(Number.isFinite);
}

function confirms(index: number, primary: number, offsets: number[]): boolean {
  return offsets.some((offset) => index === primary + offset);
}

function arrayIndex(value: number, count: number): number {
  return Math.max(0, Math.min(count - 1, Math.round(value)));
}

function positiveInt(value: number): number {
  return Math.max(1, Math.round(positive(value)));
}

function positive(value: number): number {
  return Math.max(0.00000001, finite(value, 0.00000001));
}

function finite(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? value as number : fallback;
}

function clamp01(value: number): number {
  return clamp(finite(value, 0), 0, 1);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function emptyDiagnostics(config: PeakValleyStrategyConfig): PeakValleyStrategyDiagnostics {
  return {
    indicators: {},
    gates: [],
    blockers: ["warmup"],
    lastSignal: null,
    ready: false,
    warmupRemainingMs: config.saturationSec * 1_000,
    movingAverageType: config.movingAverageType,
    derivativeSource: config.derivativeSource,
    latestTick: null,
    averages: [],
    kama: null,
    sizing: {
      buy: { size: 0, sigma: 0 },
      sell: { size: 0, sigma: 0 },
    },
  };
}
