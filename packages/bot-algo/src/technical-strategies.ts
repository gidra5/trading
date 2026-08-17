import {
  MACDIndicator,
  RSIIndicator,
  type MACDIndicatorSnapshot,
  type RSIIndicatorSnapshot,
} from "./indicators.js";
import {
  PeakValleyStrategy,
  type PeakValleyStrategyConfig,
} from "./peak-valley-strategy.js";
import type {
  PositionSide,
  StrategyDiagnostics,
  StrategyOptions,
  StrategySnapshot,
  TradingStrategy,
  TradingStrategyEntrySignal,
  TradingStrategyExitSignal,
} from "./strategy.js";
import type { TradingApi, TradingCandle, TradingTick } from "./trading-api.js";

type ExposureState = -1 | 0 | 1;

interface ExposureTransition {
  previous: ExposureState;
  current: ExposureState;
}

interface CandleBucket {
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MacdStrategyConfig {
  /** Physical chart interval on which the MACD is evaluated. */
  signalIntervalMs: number;
  fastPeriod: number;
  slowPeriod: number;
  signalPeriod: number;
  rsiPeriod: number;
  rsiOversold: number;
  rsiOverbought: number;
  /** A MACD crossover can use an RSI extreme from at most this many signal bars ago. */
  rsiConfirmationWindowPeriods: number;
}

/** Conventional 12/26/9 MACD with RSI(14) 30/70 confirmation on one-hour bars. */
export const defaultMacdStrategyConfig: MacdStrategyConfig = {
  signalIntervalMs: 60 * 60_000,
  fastPeriod: 12,
  slowPeriod: 26,
  signalPeriod: 9,
  rsiPeriod: 14,
  rsiOversold: 30,
  rsiOverbought: 70,
  rsiConfirmationWindowPeriods: 6,
};

export interface VolumeImbalanceStrategyConfig {
  /** Number of source candles whose aggressive flow is pooled. */
  lookbackPeriods: number;
  /** Allow a directional entry when aggressive volume agrees by this magnitude. */
  entryThreshold: number;
}

/** Current one-minute BTC flow with a 15% directional-confirmation threshold. */
export const defaultVolumeImbalanceStrategyConfig: VolumeImbalanceStrategyConfig = {
  lookbackPeriods: 1,
  entryThreshold: 0.15,
};

export interface MacdStrategySnapshot {
  version: 2;
  indicator: MACDIndicatorSnapshot;
  rsi: RSIIndicatorSnapshot;
  signalSampleCount: number;
  macdState: ExposureState;
  state: ExposureState;
  lastOversoldSample: number | null;
  lastOverboughtSample: number | null;
  bucket: CandleBucket | null;
  lastTick: TradingTick | null;
  lastSignal: StrategyDiagnostics["lastSignal"];
}

export interface VolumeImbalanceStrategySnapshot {
  version: 2;
  base: StrategySnapshot;
  samples: { buyVolume: number; sellVolume: number }[];
  lastTick: TradingTick | null;
  currentTickHasFlow: boolean;
  lastSignal: StrategyDiagnostics["lastSignal"];
}

/**
 * A MACD crossover strategy whose entries require recent RSI confirmation.
 *
 * The indicator is evaluated on explicit signal bars. A crossover emits one
 * entry/exit event; it does not continuously rebalance leverage while the
 * histogram remains on one side of zero. Crossovers always close an opposing
 * position, even when the RSI entry filter rejects the reversal.
 */
export class MacdStrategy implements TradingStrategy<
  PeakValleyStrategyConfig,
  MacdStrategySnapshot,
  StrategyDiagnostics
> {
  private readonly historyApi: TradingApi;
  private indicator: MACDIndicator;
  private rsi: RSIIndicator;
  private signalSampleCount = 0;
  private macdState: ExposureState = 0;
  private state: ExposureState = 0;
  private lastOversoldSample: number | null = null;
  private lastOverboughtSample: number | null = null;
  private bucket: CandleBucket | null = null;
  private transition: ExposureTransition | null = null;
  private lastTick: TradingTick | null = null;
  private lastSignal: StrategyDiagnostics["lastSignal"] = null;

  constructor(
    private readonly options: StrategyOptions<PeakValleyStrategyConfig>,
    private readonly config: MacdStrategyConfig = defaultMacdStrategyConfig,
  ) {
    validateMacdConfig(config, options.config.sampleIntervalMs);
    this.historyApi = { getHistory: options.getHistory } as TradingApi;
    this.indicator = this.createIndicator();
    this.rsi = this.createRsi();
  }

  staticConfidence(): number {
    return 1;
  }

  async warmup(): Promise<void> {
    this.reset();
    const candles = await this.options.getHistory({
      intervalMs: this.options.config.sampleIntervalMs,
      count: this.requiredInputSamples(),
    });
    for (const candle of candles.slice(-this.requiredInputSamples())) this.consume(candle);
    this.transition = null;
    this.lastSignal = null;
  }

  async onTick(tick: TradingTick): Promise<void> {
    this.transition = null;
    if (!tick.candle || !(tick.price > 0) || !Number.isFinite(tick.price)) return;
    if (this.lastTick && tick.timestamp <= this.lastTick.timestamp) return;
    this.consume(tick.candle, tick);
  }

  async entrySignal(): Promise<TradingStrategyEntrySignal | null> {
    const current = this.transition?.current ?? 0;
    return current === 0 ? null : entry(side(current));
  }

  async exitSignal(): Promise<TradingStrategyExitSignal | null> {
    const transition = this.transition;
    return transition && transition.previous !== 0 && transition.previous !== transition.current
      ? exit(side(transition.previous))
      : null;
  }

  async snapshot(): Promise<MacdStrategySnapshot> {
    return {
      version: 2,
      indicator: this.indicator.snapshot(),
      rsi: this.rsi.snapshot(),
      signalSampleCount: this.signalSampleCount,
      macdState: this.macdState,
      state: this.state,
      lastOversoldSample: this.lastOversoldSample,
      lastOverboughtSample: this.lastOverboughtSample,
      bucket: structuredClone(this.bucket),
      lastTick: structuredClone(this.lastTick),
      lastSignal: structuredClone(this.lastSignal),
    };
  }

  async restore(snapshot: MacdStrategySnapshot): Promise<void> {
    if (snapshot.version !== 2) throw new Error(`Unsupported MACD strategy snapshot: ${snapshot.version}`);
    this.indicator.restore(snapshot.indicator);
    this.rsi.restore(snapshot.rsi);
    this.signalSampleCount = Math.max(0, Math.round(snapshot.signalSampleCount));
    this.macdState = exposureState(snapshot.macdState);
    this.state = exposureState(snapshot.state);
    this.lastOversoldSample = validSampleIndex(snapshot.lastOversoldSample);
    this.lastOverboughtSample = validSampleIndex(snapshot.lastOverboughtSample);
    this.bucket = structuredClone(snapshot.bucket);
    this.transition = null;
    this.lastTick = structuredClone(snapshot.lastTick);
    this.lastSignal = structuredClone(snapshot.lastSignal);
  }

  async updateConfig(config: PeakValleyStrategyConfig): Promise<void> {
    validateMacdConfig(this.config, config.sampleIntervalMs);
    this.options.config = config;
    await this.warmup();
  }

  getDiagnostics(): StrategyDiagnostics {
    const value = this.indicator.indicator();
    const ready = this.signalSampleCount >= this.requiredSignalSamples();
    const rsi = this.rsi.indicator();
    const longConfirmed = this.recent(this.lastOversoldSample);
    const shortConfirmed = this.recent(this.lastOverboughtSample);
    return {
      indicators: {
        "macd.value": value.macd,
        "macd.signal": value.signal,
        "macd.histogram": value.histogram,
        "macd.state": this.macdState,
        "macd.positionState": this.state,
        "macd.signalIntervalMs": this.config.signalIntervalMs,
        "rsi.value": rsi,
        "rsi.lastOversoldAge": sampleAge(this.signalSampleCount, this.lastOversoldSample),
        "rsi.lastOverboughtAge": sampleAge(this.signalSampleCount, this.lastOverboughtSample),
      },
      gates: [
        { code: "macd.ready", passed: ready, value: this.signalSampleCount, threshold: this.requiredSignalSamples() },
        { code: "macd.bullish", passed: ready && value.histogram > 0, value: value.histogram, threshold: 0 },
        { code: "macd.bearish", passed: ready && value.histogram < 0, value: value.histogram, threshold: 0 },
        { code: "rsi.long-confirmed", passed: ready && longConfirmed, value: rsi, threshold: this.config.rsiOversold },
        { code: "rsi.short-confirmed", passed: ready && shortConfirmed, value: rsi, threshold: this.config.rsiOverbought },
      ],
      blockers: ready ? [] : ["warmup"],
      lastSignal: structuredClone(this.lastSignal),
    };
  }

  private createIndicator(): MACDIndicator {
    return new MACDIndicator(
      this.config.fastPeriod,
      this.config.slowPeriod,
      this.config.signalPeriod,
      this.historyApi,
      this.requiredSignalSamples(),
    );
  }

  private createRsi(): RSIIndicator {
    return new RSIIndicator(this.config.rsiPeriod, this.historyApi);
  }

  private requiredSignalSamples(): number {
    return Math.max(
      Math.max(this.config.fastPeriod, this.config.slowPeriod) + this.config.signalPeriod,
      this.config.rsiPeriod + 1,
    );
  }

  private requiredInputSamples(): number {
    return (this.requiredSignalSamples() + 1)
      * (this.config.signalIntervalMs / this.options.config.sampleIntervalMs);
  }

  private reset(): void {
    this.indicator = this.createIndicator();
    this.rsi = this.createRsi();
    this.signalSampleCount = 0;
    this.macdState = 0;
    this.state = 0;
    this.lastOversoldSample = null;
    this.lastOverboughtSample = null;
    this.bucket = null;
    this.transition = null;
    this.lastTick = null;
    this.lastSignal = null;
  }

  private consume(candle: TradingCandle, tick: TradingTick = candleTick(candle)): void {
    const complete = this.aggregate(candle);
    this.lastTick = tick;
    if (!complete) return;
    const previous = this.state;
    this.indicator.onTick({ eventTime: complete.closeTime, candle: complete });
    this.rsi.onTick({ eventTime: complete.closeTime, candle: complete });
    this.signalSampleCount += 1;
    const rsi = this.rsi.indicator();
    if (rsi <= this.config.rsiOversold) this.lastOversoldSample = this.signalSampleCount;
    if (rsi >= this.config.rsiOverbought) this.lastOverboughtSample = this.signalSampleCount;
    if (this.signalSampleCount < this.requiredSignalSamples()) return;
    const histogram = this.indicator.indicator().histogram;
    const candidate = histogram > 0 ? 1 : histogram < 0 ? -1 : 0;
    if (candidate === this.macdState) return;
    this.macdState = candidate;
    const confirmed = candidate > 0
      ? this.recent(this.lastOversoldSample)
      : candidate < 0
        ? this.recent(this.lastOverboughtSample)
        : false;
    this.state = enabledState(confirmed ? candidate : 0, this.options.config);
    this.transition = changed(previous, this.state);
    this.lastSignal = transitionSignal(
      this.transition,
      this.state === 0
        ? "MACD line/signal-line crossover exit"
        : "MACD crossover confirmed by recent RSI extreme",
    );
  }

  private recent(sample: number | null): boolean {
    return sample !== null
      && this.signalSampleCount - sample <= this.config.rsiConfirmationWindowPeriods;
  }

  private aggregate(candle: TradingCandle): TradingCandle | null {
    const interval = this.config.signalIntervalMs;
    const openTime = Math.floor(candle.openTime / interval) * interval;
    if (!this.bucket || this.bucket.openTime !== openTime) {
      this.bucket = {
        openTime,
        closeTime: candle.closeTime,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
      };
    } else {
      this.bucket.closeTime = candle.closeTime;
      this.bucket.high = Math.max(this.bucket.high, candle.high);
      this.bucket.low = Math.min(this.bucket.low, candle.low);
      this.bucket.close = candle.close;
      this.bucket.volume += candle.volume;
    }
    if (candle.closeTime < openTime + interval - 1) return null;
    const complete = this.bucket;
    this.bucket = null;
    return complete;
  }
}

/**
 * Peak/valley entries filtered by aggressor-volume imbalance:
 * (buy volume - sell volume) / (buy volume + sell volume).
 *
 * Imbalance cannot open or maintain a position by itself. It only accepts a
 * same-direction entry from the slower peak/valley strategy. Exits are always
 * delegated unchanged so stale or missing flow can never delay de-risking.
 */
export class VolumeImbalanceStrategy implements TradingStrategy<
  PeakValleyStrategyConfig,
  VolumeImbalanceStrategySnapshot,
  StrategyDiagnostics
> {
  private readonly base: TradingStrategy<
    PeakValleyStrategyConfig,
    StrategySnapshot,
    StrategyDiagnostics
  >;
  private samples: { buyVolume: number; sellVolume: number }[] = [];
  private lastTick: TradingTick | null = null;
  private currentTickHasFlow = false;
  private blockedEntrySide: PositionSide | null = null;
  private lastSignal: StrategyDiagnostics["lastSignal"] = null;

  constructor(
    private readonly options: StrategyOptions<PeakValleyStrategyConfig>,
    private readonly config: VolumeImbalanceStrategyConfig = defaultVolumeImbalanceStrategyConfig,
    base?: TradingStrategy<PeakValleyStrategyConfig, StrategySnapshot, StrategyDiagnostics>,
  ) {
    validateVolumeImbalanceConfig(config);
    this.base = base ?? new PeakValleyStrategy(options);
  }

  staticConfidence(): number {
    return this.base.staticConfidence();
  }

  async warmup(): Promise<void> {
    this.reset();
    await this.base.warmup();
    const candles = await this.options.getHistory({
      intervalMs: this.options.config.sampleIntervalMs,
      count: this.config.lookbackPeriods,
    });
    for (const candle of candles.slice(-this.config.lookbackPeriods)) this.consumeFlow(candle);
    this.currentTickHasFlow = false;
    this.lastSignal = null;
  }

  async onTick(tick: TradingTick): Promise<void> {
    this.blockedEntrySide = null;
    this.currentTickHasFlow = false;
    await this.base.onTick(tick);
    if (!tick.candle || !(tick.price > 0) || !Number.isFinite(tick.price)) return;
    if (this.lastTick && tick.timestamp <= this.lastTick.timestamp) return;
    this.lastTick = tick;
    this.currentTickHasFlow = this.consumeFlow(tick.candle);
  }

  async entrySignal(): Promise<TradingStrategyEntrySignal | null> {
    const candidate = await this.base.entrySignal();
    if (!candidate) return null;
    if (!this.allows(candidate.side)) {
      this.blockedEntrySide = candidate.side;
      return null;
    }
    this.lastSignal = {
      type: "entry",
      side: candidate.side,
      reason: "peak/valley entry confirmed by aggressor-volume imbalance",
    };
    return candidate;
  }

  async exitSignal(): Promise<TradingStrategyExitSignal | null> {
    const signal = await this.base.exitSignal();
    if (signal) {
      this.lastSignal = {
        type: "exit",
        side: signal.side,
        reason: "unfiltered peak/valley exit",
      };
    }
    return signal;
  }

  async snapshot(): Promise<VolumeImbalanceStrategySnapshot> {
    return {
      version: 2,
      base: await this.base.snapshot(),
      samples: structuredClone(this.samples),
      lastTick: structuredClone(this.lastTick),
      currentTickHasFlow: this.currentTickHasFlow,
      lastSignal: structuredClone(this.lastSignal),
    };
  }

  async restore(snapshot: VolumeImbalanceStrategySnapshot): Promise<void> {
    if (snapshot.version !== 2) {
      throw new Error(`Unsupported volume-imbalance strategy snapshot: ${snapshot.version}`);
    }
    await this.base.restore(snapshot.base);
    this.samples = snapshot.samples
      .filter((sample) => validVolume(sample.buyVolume) && validVolume(sample.sellVolume))
      .slice(-this.config.lookbackPeriods);
    this.lastTick = structuredClone(snapshot.lastTick);
    this.currentTickHasFlow = snapshot.currentTickHasFlow === true;
    this.blockedEntrySide = null;
    this.lastSignal = structuredClone(snapshot.lastSignal);
  }

  async updateConfig(config: PeakValleyStrategyConfig): Promise<void> {
    this.options.config = config;
    await this.warmup();
  }

  getDiagnostics(): StrategyDiagnostics {
    const base = this.base.getDiagnostics();
    const imbalance = this.imbalance();
    const ready = this.ready();
    return {
      indicators: {
        ...base.indicators,
        "volumeImbalance.value": imbalance,
        "volumeImbalance.lookbackPeriods": this.config.lookbackPeriods,
      },
      gates: [
        ...base.gates,
        { code: "volume-imbalance.ready", passed: ready, value: this.samples.length, threshold: this.config.lookbackPeriods },
        { code: "volume-imbalance.long", passed: ready && imbalance >= this.config.entryThreshold, value: imbalance, threshold: this.config.entryThreshold },
        { code: "volume-imbalance.short", passed: ready && imbalance <= -this.config.entryThreshold, value: imbalance, threshold: -this.config.entryThreshold },
      ],
      blockers: [
        ...base.blockers,
        ...(!ready ? ["aggressor-volume-warmup"] : []),
        ...(this.blockedEntrySide ? [`aggressor-volume-blocked-${this.blockedEntrySide}`] : []),
      ],
      lastSignal: structuredClone(this.lastSignal),
    };
  }

  private reset(): void {
    this.samples = [];
    this.lastTick = null;
    this.currentTickHasFlow = false;
    this.blockedEntrySide = null;
    this.lastSignal = null;
  }

  private consumeFlow(candle: TradingCandle): boolean {
    const buyVolume = candle.aggressiveBuyVolume;
    const sellVolume = candle.aggressiveSellVolume;
    if (!validVolume(buyVolume) || !validVolume(sellVolume)) {
      this.samples = [];
      return false;
    }
    this.samples.push({ buyVolume, sellVolume });
    while (this.samples.length > this.config.lookbackPeriods) this.samples.shift();
    return this.samples.length >= this.config.lookbackPeriods;
  }

  private ready(): boolean {
    return this.currentTickHasFlow && this.samples.length >= this.config.lookbackPeriods;
  }

  private allows(side: PositionSide): boolean {
    if (!this.ready()) return false;
    const imbalance = this.imbalance();
    return side === "long"
      ? imbalance >= this.config.entryThreshold
      : imbalance <= -this.config.entryThreshold;
  }

  private imbalance(): number {
    const buy = this.samples.reduce((sum, sample) => sum + sample.buyVolume, 0);
    const sell = this.samples.reduce((sum, sample) => sum + sample.sellVolume, 0);
    return buy + sell > 0 ? (buy - sell) / (buy + sell) : 0;
  }
}

function entry(side: PositionSide): TradingStrategyEntrySignal {
  return { side, size: 1, leverage: 999, price: null, confidence: null };
}

function exit(side: PositionSide): TradingStrategyExitSignal {
  return { side, size: 1, price: null, confidence: null };
}

function changed(previous: ExposureState, current: ExposureState): ExposureTransition | null {
  return previous === current ? null : { previous, current };
}

function transitionSignal(
  transition: ExposureTransition | null,
  reason: string,
): StrategyDiagnostics["lastSignal"] {
  if (!transition) return null;
  if (transition.current !== 0) {
    return { type: "entry", side: side(transition.current), reason };
  }
  return transition.previous !== 0
    ? { type: "exit", side: side(transition.previous), reason }
    : null;
}

function enabledState(
  state: ExposureState,
  config: PeakValleyStrategyConfig,
): ExposureState {
  if (state > 0 && !config.longSideEnabled) return 0;
  if (state < 0 && !config.shortSideEnabled) return 0;
  return state;
}

function side(state: Exclude<ExposureState, 0>): PositionSide {
  return state > 0 ? "long" : "short";
}

function exposureState(value: number): ExposureState {
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}

function validSampleIndex(value: number | null): number | null {
  return value !== null && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function sampleAge(current: number, sample: number | null): number | null {
  return sample === null ? null : Math.max(0, current - sample);
}

function candleTick(candle: TradingCandle): TradingTick {
  return {
    timestamp: candle.closeTime,
    price: candle.close,
    quantity: candle.volume,
    candle,
  };
}

function validVolume(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value) && value >= 0;
}

function validateMacdConfig(
  config: MacdStrategyConfig,
  sampleIntervalMs: number,
): void {
  if (!Number.isSafeInteger(config.signalIntervalMs)
    || !Number.isSafeInteger(sampleIntervalMs)
    || config.signalIntervalMs < sampleIntervalMs
    || config.signalIntervalMs % sampleIntervalMs !== 0
    || ![config.fastPeriod, config.slowPeriod, config.signalPeriod, config.rsiPeriod].every(
      (value) => Number.isSafeInteger(value) && value > 0,
    )
    || config.fastPeriod >= config.slowPeriod
    || !(config.rsiOversold > 0 && config.rsiOversold < config.rsiOverbought)
    || !(config.rsiOverbought < 100)
    || !Number.isSafeInteger(config.rsiConfirmationWindowPeriods)
    || config.rsiConfirmationWindowPeriods < 0) {
    throw new Error("MACD periods and signal interval are invalid.");
  }
}

function validateVolumeImbalanceConfig(config: VolumeImbalanceStrategyConfig): void {
  if (!Number.isSafeInteger(config.lookbackPeriods)
    || config.lookbackPeriods < 1
    || !(config.entryThreshold > 0 && config.entryThreshold <= 1)) {
    throw new Error("Volume-imbalance lookback or thresholds are invalid.");
  }
}
