import type {
  TradingCandle as Candle,
  TradingApi,
} from "./trading-api.js";

const DEFAULT_EMA_WARMUP_COUNT = 256;
const MAX_EMA_WARMUP_COUNT = 2_000;

export interface IndicatorInputBase {
  eventTime: number;
}

export interface PriceIndicatorInput extends IndicatorInputBase {
  price?: number;
  quantity?: number;
  candle?: Candle;
}

export interface NumericIndicatorInput extends IndicatorInputBase {
  value: number;
}

export interface WeightedNumericIndicatorInput extends NumericIndicatorInput {
  weight?: number;
}

export type ValueIndicatorInput = PriceIndicatorInput | NumericIndicatorInput;
export type WeightedValueIndicatorInput = PriceIndicatorInput | WeightedNumericIndicatorInput;
export type EMAIndicatorInput = ValueIndicatorInput & { alpha?: number };

export interface TradingIndicator<TSnapshot, TValue, TInput> {
  warmup(initialValue?: TValue): Promise<void>;
  onTick(input: TInput): void;
  indicator(): TValue;
  snapshot(): TSnapshot;
  restore(snapshot: TSnapshot | null): void;
}

export interface NumericTradingIndicator<TSnapshot, TInput> extends TradingIndicator<TSnapshot, number, TInput> {
  derivative(): number;
}

export interface LookbackIndicatorSnapshot {
  version: 1;
  lookback: number;
  values: number[];
}

export class LookbackIndicator
  implements NumericTradingIndicator<LookbackIndicatorSnapshot, NumericIndicatorInput>
{
  private values: number[] = [];

  constructor(private readonly lookback: number) {
    assertPositiveWindow(lookback, "lookback");
  }

  async warmup(initialValue?: number): Promise<void> {
    if (typeof initialValue === "number" && Number.isFinite(initialValue)) {
      this.onTick({ eventTime: 0, value: initialValue });
    }
  }

  onTick(input: NumericIndicatorInput): void {
    if (!Number.isFinite(input.value)) return;
    this.values.push(input.value);
    while (this.values.length > this.lookback + 1) this.values.shift();
  }

  indicator(): number {
    return this.values.at(-1) ?? 0;
  }

  previous(): number {
    return this.values.at(-1 - this.lookback) ?? this.indicator();
  }

  derivative(): number {
    return this.indicator() - this.previous();
  }

  snapshot(): LookbackIndicatorSnapshot {
    return { version: 1, lookback: this.lookback, values: this.values.slice() };
  }

  restore(snapshot: LookbackIndicatorSnapshot | null): void {
    this.values = snapshot?.values?.filter(isFiniteNumber).slice(-(this.lookback + 1)) ?? [];
  }
}

export interface SMAIndicatorSnapshot {
  version: 1;
  windowSize: number;
  values: number[];
  weights: number[];
  sum: number;
  weightedSum: number;
  weightSum: number;
  delta: number;
}

abstract class BaseSMAIndicator<TInput extends ValueIndicatorInput>
  implements NumericTradingIndicator<SMAIndicatorSnapshot, TInput>
{
  private values: number[] = [];
  private weights: number[] = [];
  private sum = 0;
  private weightedSum = 0;
  private weightSum = 0;
  private delta = 0;

  constructor(
    private readonly windowSize: number,
    private readonly tradingApi: TradingApi,
    private readonly label: string,
  ) {
    assertPositiveWindow(windowSize, label);
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.windowSize);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime } as TInput);
    }
    assertWarmupValue(this.values.length > 0, this.label);
  }

  onTick(input: TInput): void {
    const price = indicatorValue(input);
    if (!Number.isFinite(price)) return;
    const weight = this.entryWeight(input);

    const hadPrevious = this.values.length > 0;
    const prev = this.indicator();
    this.values.push(price);
    this.weights.push(weight);
    this.sum += price;
    this.weightedSum += price * weight;
    this.weightSum += weight;
    while (this.values.length > this.windowSize) {
      const removedValue = this.values.shift() ?? 0;
      this.weights.shift();
      this.sum -= removedValue;
    }
    this.recalculateWeightedSums();
    this.delta = hadPrevious ? this.indicator() - prev : 0;
  }

  indicator(): number {
    if (this.values.length === 0) return 0;
    return this.weightSum > 0 ? this.weightedSum / this.weightSum : this.sum / this.values.length;
  }

  derivative(): number {
    return this.delta;
  }

  snapshot(): SMAIndicatorSnapshot {
    return {
      version: 1,
      windowSize: this.windowSize,
      values: this.values.slice(),
      weights: this.weights.slice(),
      sum: this.sum,
      weightedSum: this.weightedSum,
      weightSum: this.weightSum,
      delta: this.delta,
    };
  }

  restore(snapshot: SMAIndicatorSnapshot | null): void {
    this.values = snapshot?.values?.slice(-this.windowSize).filter(isFiniteNumber) ?? [];
    this.weights = normalizeWeights(snapshot?.weights, this.values.length);
    this.sum = this.values.reduce((total, value) => total + value, 0);
    this.recalculateWeightedSums();
    this.delta = Number.isFinite(snapshot?.delta) ? (snapshot?.delta ?? 0) : 0;
  }

  protected abstract entryWeight(input: TInput): number;

  private recalculateWeightedSums(): void {
    this.weightedSum = 0;
    this.weightSum = 0;
    for (let index = 0; index < this.values.length; index += 1) {
      const value = this.values[index] ?? 0;
      const weight = this.weights[index] ?? 1;
      this.weightedSum += value * weight;
      this.weightSum += weight;
    }
  }
}

export class SMAIndicator extends BaseSMAIndicator<ValueIndicatorInput> {
  constructor(windowSize: number, tradingApi: TradingApi) {
    super(windowSize, tradingApi, "SMA");
  }

  protected entryWeight(): number {
    return 1;
  }
}

export class VolumeWeightedSMAIndicator extends BaseSMAIndicator<WeightedValueIndicatorInput> {
  constructor(windowSize: number, tradingApi: TradingApi) {
    super(windowSize, tradingApi, "volume-weighted SMA");
  }

  protected entryWeight(input: WeightedValueIndicatorInput): number {
    return indicatorWeight(input);
  }
}

export interface EMAIndicatorSnapshot {
  version: 1;
  alpha: number;
  value: number;
  delta: number;
}

export class EMAIndicator implements NumericTradingIndicator<EMAIndicatorSnapshot, EMAIndicatorInput> {
  private value = 0;
  private delta = 0;
  private alpha: number;
  private readonly warmupCount: number;

  constructor(
    periodOrAlpha: number,
    private readonly tradingApi?: TradingApi,
    warmupCount?: number,
  ) {
    if (!Number.isFinite(periodOrAlpha) || periodOrAlpha <= 0) {
      throw new Error("EMA period/alpha must be positive.");
    }
    this.alpha = normalizeAlpha(periodOrAlpha);
    this.warmupCount = warmupCount
      ? warmupCount
      : clampInteger(Math.ceil(periodOrAlpha <= 1 ? 1 / periodOrAlpha : periodOrAlpha), 1, MAX_EMA_WARMUP_COUNT);
  }

  async warmup(initialValue?: number): Promise<void> {
    if (Number.isFinite(initialValue)) {
      this.onTick({ eventTime: 0, value: initialValue });
    }
    if (!this.tradingApi) {
      return;
    }
    const candles = await warmupCandles(this.tradingApi, this.warmupCount);
    const prices = candles
      .map((candle) => indicatorPrice({ candle, eventTime: candle.closeTime }))
      .filter(isPositiveFinite);
    for (const price of prices) {
      this.onTick({ eventTime: 0, value: price });
    }
    assertWarmupValue(this.value !== 0, "EMA");
  }

  onTick(input: EMAIndicatorInput): void {
    const value = indicatorValue(input);
    if (!Number.isFinite(value)) return;

    const previous = this.value;
    if (previous === 0) {
      this.value = value;
      this.delta = 0;
      return;
    }
    const alpha = input.alpha ?? this.alpha;
    this.value = previous + alpha * (value - previous);
    this.delta = this.value - previous;
  }

  indicator(): number {
    return this.value;
  }

  derivative(): number {
    return this.delta;
  }

  snapshot(): EMAIndicatorSnapshot {
    return {
      version: 1,
      alpha: this.alpha,
      value: this.value,
      delta: this.delta,
    };
  }

  restore(snapshot: EMAIndicatorSnapshot | null): void {
    this.alpha = isNormalizedAlpha(snapshot?.alpha) ? snapshot.alpha : this.alpha;
    this.value = Number.isFinite(snapshot?.value) ? (snapshot?.value ?? 0) : 0;
    this.delta = Number.isFinite(snapshot?.delta) ? (snapshot?.delta ?? 0) : 0;
  }
}

export class VolumeWeightedEMAIndicator
  implements NumericTradingIndicator<VolumeWeightedEMAIndicatorSnapshot, PriceIndicatorInput>
{
  private readonly priceVolumeEma: EMAIndicator;
  private readonly volumeEma: EMAIndicator;
  private value = 0;
  private delta = 0;
  private readonly warmupCount: number;

  constructor(
    periodOrAlpha: number,
    private readonly tradingApi: TradingApi,
    warmupCount?: number,
  ) {
    this.priceVolumeEma = new EMAIndicator(periodOrAlpha, tradingApi, warmupCount);
    this.volumeEma = new EMAIndicator(periodOrAlpha, tradingApi, warmupCount);
    this.warmupCount = warmupCount ?? DEFAULT_EMA_WARMUP_COUNT;
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.warmupCount);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.value !== 0, "volume-weighted EMA");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    const volume = indicatorVolume(input);
    if (price <= 0 || volume <= 0) return;

    const previous = this.value;
    this.priceVolumeEma.onTick({ eventTime: input.eventTime, value: price * volume });
    this.volumeEma.onTick({ eventTime: input.eventTime, value: volume });
    const denominator = this.volumeEma.indicator();
    this.value = denominator > 0 ? this.priceVolumeEma.indicator() / denominator : previous;
    this.delta = this.value - previous;
  }

  indicator(): number {
    return this.value;
  }

  derivative(): number {
    return this.delta;
  }

  snapshot(): VolumeWeightedEMAIndicatorSnapshot {
    return {
      version: 1,
      priceVolumeEma: this.priceVolumeEma.snapshot(),
      volumeEma: this.volumeEma.snapshot(),
      value: this.value,
      delta: this.delta,
    };
  }

  restore(snapshot: VolumeWeightedEMAIndicatorSnapshot | null): void {
    this.priceVolumeEma.restore(snapshot?.priceVolumeEma ?? null);
    this.volumeEma.restore(snapshot?.volumeEma ?? null);
    this.value = Number.isFinite(snapshot?.value) ? (snapshot?.value ?? 0) : 0;
    this.delta = Number.isFinite(snapshot?.delta) ? (snapshot?.delta ?? 0) : 0;
  }
}

export interface VolumeWeightedEMAIndicatorSnapshot {
  version: 1;
  priceVolumeEma: EMAIndicatorSnapshot;
  volumeEma: EMAIndicatorSnapshot;
  value: number;
  delta: number;
}

export interface EfficiencyRatioIndicatorSnapshot {
  version: 1;
  windowSize: number;
  values: number[];
  value: number;
  delta: number;
}

export class EfficiencyRatioIndicator
  implements NumericTradingIndicator<EfficiencyRatioIndicatorSnapshot, ValueIndicatorInput>
{
  private values: number[] = [];
  private start = 0;
  private noise = 0;
  private value = 0;
  private delta = 0;

  constructor(
    private readonly windowSize: number,
    private readonly tradingApi: TradingApi,
  ) {
    assertPositiveWindow(windowSize, "efficiency ratio");
  }

  async warmup(initialValue?: number): Promise<void> {
    if (Number.isFinite(initialValue)) {
      this.onTick({ eventTime: 0, value: initialValue });
    }
    const candles = await warmupCandles(this.tradingApi, this.windowSize + 1);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.values.length >= this.windowSize + 1, "efficiency ratio");
  }

  onTick(input: ValueIndicatorInput): void {
    const value = indicatorValue(input);
    if (!Number.isFinite(value)) return;

    const previous = this.value;
    const last = this.values.at(-1);
    if (last !== undefined) this.noise += Math.abs(value - last);
    this.values.push(value);
    if (this.values.length - this.start > this.windowSize + 1) {
      this.noise -= Math.abs(this.values[this.start + 1]! - this.values[this.start]!);
      this.start += 1;
      if (this.start >= 4_096 && this.start * 2 >= this.values.length) {
        this.values = this.values.slice(this.start);
        this.start = 0;
      }
    }
    this.value = this.calculate();
    this.delta = this.value - previous;
  }

  indicator(): number {
    return this.value;
  }

  derivative(): number {
    return this.delta;
  }

  snapshot(): EfficiencyRatioIndicatorSnapshot {
    return {
      version: 1,
      windowSize: this.windowSize,
      values: this.values.slice(this.start),
      value: this.value,
      delta: this.delta,
    };
  }

  restore(snapshot: EfficiencyRatioIndicatorSnapshot | null): void {
    this.values = snapshot?.values?.slice(-(this.windowSize + 1)).filter(isFiniteNumber) ?? [];
    this.start = 0;
    this.noise = rollingNoise(this.values);
    this.value = this.calculate();
    this.delta = Number.isFinite(snapshot?.delta) ? (snapshot?.delta ?? 0) : 0;
  }

  private calculate(): number {
    if (this.values.length - this.start < this.windowSize + 1) {
      return 0;
    }

    const first = this.values[this.start] ?? 0;
    const last = this.values.at(-1) ?? 0;
    const signal = Math.abs(last - first);
    return this.noise > 0 ? clampRatio(signal / this.noise) : 0;
  }
}

function rollingNoise(values: number[]): number {
  let noise = 0;
  for (let index = 1; index < values.length; index += 1) {
    noise += Math.abs(values[index]! - values[index - 1]!);
  }
  return noise;
}

export interface KAMAIndicatorSnapshot {
  version: 1;
  efficiencyPeriod: number;
  efficiencyRatio: EfficiencyRatioIndicatorSnapshot;
  ema: EMAIndicatorSnapshot;
  alpha: number;
}

export class KAMAIndicator implements NumericTradingIndicator<KAMAIndicatorSnapshot, PriceIndicatorInput> {
  private readonly ema: EMAIndicator;
  private readonly efficiencyRatio: EfficiencyRatioIndicator;
  private readonly fastAlpha: number;
  private readonly slowAlpha: number;
  private alpha: number;

  constructor(
    private readonly efficiencyPeriod: number,
    fastPeriodOrAlpha: number,
    slowPeriodOrAlpha: number,
    private readonly tradingApi: TradingApi,
    private readonly warmupCount = Math.max(efficiencyPeriod + 1, DEFAULT_EMA_WARMUP_COUNT),
    private readonly power = 2,
  ) {
    assertPositiveWindow(efficiencyPeriod, "KAMA efficiency");
    if (!Number.isFinite(power) || power <= 0) {
      throw new Error("KAMA power must be positive.");
    }
    this.fastAlpha = normalizeAlpha(fastPeriodOrAlpha);
    this.slowAlpha = normalizeAlpha(slowPeriodOrAlpha);
    this.alpha = this.slowAlpha;
    this.ema = new EMAIndicator(this.slowAlpha, tradingApi, warmupCount);
    this.efficiencyRatio = new EfficiencyRatioIndicator(efficiencyPeriod, tradingApi);
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.warmupCount);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(
      this.efficiencyRatio.snapshot().values.length >= this.efficiencyPeriod + 1,
      "KAMA efficiency ratio",
    );
    assertWarmupValue(this.ema.indicator() !== 0, "KAMA");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    if (price <= 0) return;

    this.efficiencyRatio.onTick(input);
    this.alpha = this.adaptiveAlpha();
    this.ema.onTick({ eventTime: input.eventTime, value: price, alpha: this.alpha });
  }

  indicator(): number {
    return this.ema.indicator();
  }

  derivative(): number {
    return this.ema.derivative();
  }

  snapshot(): KAMAIndicatorSnapshot {
    return {
      version: 1,
      efficiencyPeriod: this.efficiencyPeriod,
      efficiencyRatio: this.efficiencyRatio.snapshot(),
      ema: this.ema.snapshot(),
      alpha: this.alpha,
    };
  }

  restore(snapshot: KAMAIndicatorSnapshot | null): void {
    this.efficiencyRatio.restore(snapshot?.efficiencyRatio ?? null);
    this.ema.restore(snapshot?.ema ?? null);
    this.alpha = isNormalizedAlpha(snapshot?.alpha) ? snapshot.alpha : this.slowAlpha;
  }

  private adaptiveAlpha(): number {
    const efficiencyRatio = this.efficiencyRatio.indicator();
    const smoothing = efficiencyRatio * (this.fastAlpha - this.slowAlpha) + this.slowAlpha;
    return clampRatio(Math.pow(smoothing, this.power));
  }
}

export interface VolumeWeightedKAMAIndicatorConfig {
  efficiencyPeriod: number;
  fastPeriod: number;
  slowPeriod: number;
  power: number;
  volumePeriod: number;
  volumeCap: number;
  volumePower: number;
}

export function volumeWeightedKamaWarmupSamples(
  config: Pick<
    VolumeWeightedKAMAIndicatorConfig,
    "efficiencyPeriod" | "slowPeriod" | "volumePeriod"
  >,
  multiple = 1,
): number {
  return Math.ceil(Math.max(
    config.efficiencyPeriod + 1,
    config.slowPeriod,
    config.volumePeriod,
  ) * Math.max(1, multiple));
}

export type VolumeWeightedKAMAIndicatorOptions = Partial<VolumeWeightedKAMAIndicatorConfig> & {
  warmupCount?: number;
  warmupIntervalMs?: number;
};

export interface VolumeWeightedKAMAIndicatorValue {
  kama: number;
  efficiencyRatio: number;
  relativeVolume: number;
  effectiveEfficiencyRatio: number;
  alpha: number;
}

export interface VolumeWeightedKAMAIndicatorSnapshot {
  version: 1;
  config: VolumeWeightedKAMAIndicatorConfig;
  efficiencyRatio: EfficiencyRatioIndicatorSnapshot;
  volumeEma: EMAIndicatorSnapshot;
  kama: EMAIndicatorSnapshot;
  value: VolumeWeightedKAMAIndicatorValue;
  delta: number;
}

const DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG: VolumeWeightedKAMAIndicatorConfig = {
  efficiencyPeriod: 20,
  fastPeriod: 5,
  slowPeriod: 50,
  power: 1,
  volumePeriod: 50,
  volumeCap: 3,
  volumePower: 1,
};

export class VolumeWeightedKAMAIndicator
  implements NumericTradingIndicator<VolumeWeightedKAMAIndicatorSnapshot, PriceIndicatorInput>
{
  private readonly config: VolumeWeightedKAMAIndicatorConfig;
  private readonly warmupCount: number;
  private readonly warmupIntervalMs: number;
  private readonly efficiencyRatio: EfficiencyRatioIndicator;
  private readonly volumeEma: EMAIndicator;
  private readonly kama: EMAIndicator;
  private readonly fastAlpha: number;
  private readonly slowAlpha: number;
  private value = emptyVolumeWeightedKAMAValue();
  private delta = 0;

  constructor(
    private readonly tradingApi: TradingApi,
    options: VolumeWeightedKAMAIndicatorOptions = {},
  ) {
    this.config = normalizeVolumeWeightedKAMAConfig(options);
    this.warmupCount = options.warmupCount ?? volumeWeightedKamaWarmupSamples(this.config);
    this.warmupIntervalMs = Math.max(1, Math.round(options.warmupIntervalMs ?? 1_000));
    this.fastAlpha = normalizeAlpha(this.config.fastPeriod);
    this.slowAlpha = normalizeAlpha(this.config.slowPeriod);
    this.efficiencyRatio = new EfficiencyRatioIndicator(this.config.efficiencyPeriod, tradingApi);
    this.volumeEma = new EMAIndicator(this.config.volumePeriod);
    this.kama = new EMAIndicator(this.slowAlpha);
  }

  async warmup(): Promise<void> {
    const candles = await this.tradingApi.getHistory({
      intervalMs: this.warmupIntervalMs,
      count: this.warmupCount,
    });
    for (const candle of candles) this.onTick({ candle, eventTime: candle.closeTime });
    assertWarmupValue(
      this.efficiencyRatio.snapshot().values.length >= this.config.efficiencyPeriod + 1,
      "volume-weighted KAMA efficiency ratio",
    );
    assertWarmupValue(this.kama.indicator() !== 0, "volume-weighted KAMA");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    if (price <= 0) return;
    this.efficiencyRatio.onTick(input);
    const volume = indicatorVolume(input);
    const volumeAverage = this.volumeEma.indicator();
    const relativeVolume = volume > 0 && volumeAverage > 0
      ? clamp(volume / volumeAverage, 0, this.config.volumeCap)
      : 1;
    const efficiencyRatio = this.efficiencyRatio.indicator();
    // The prior volume EMA keeps the adjustment causal; volumePower=0 is canonical KAMA.
    const effectiveEfficiencyRatio = clampRatio(
      efficiencyRatio * Math.pow(relativeVolume, this.config.volumePower),
    );
    const smoothing = this.slowAlpha
      + effectiveEfficiencyRatio * (this.fastAlpha - this.slowAlpha);
    const alpha = clampRatio(Math.pow(smoothing, this.config.power));
    this.kama.onTick({ eventTime: input.eventTime, value: price, alpha });
    if (volume > 0) this.volumeEma.onTick({ eventTime: input.eventTime, value: volume });
    this.delta = this.kama.derivative();
    this.value = {
      kama: this.kama.indicator(),
      efficiencyRatio,
      relativeVolume,
      effectiveEfficiencyRatio,
      alpha,
    };
  }

  indicator(): number {
    return this.value.kama;
  }

  derivative(): number {
    return this.delta;
  }

  details(): VolumeWeightedKAMAIndicatorValue {
    return { ...this.value };
  }

  snapshot(): VolumeWeightedKAMAIndicatorSnapshot {
    return {
      version: 1,
      config: { ...this.config },
      efficiencyRatio: this.efficiencyRatio.snapshot(),
      volumeEma: this.volumeEma.snapshot(),
      kama: this.kama.snapshot(),
      value: this.details(),
      delta: this.delta,
    };
  }

  restore(snapshot: VolumeWeightedKAMAIndicatorSnapshot | KAMAIndicatorSnapshot | null): void {
    this.efficiencyRatio.restore(snapshot?.efficiencyRatio ?? null);
    const weighted = snapshot && "kama" in snapshot ? snapshot : null;
    const canonical = snapshot && "ema" in snapshot ? snapshot : null;
    this.volumeEma.restore(weighted?.volumeEma ?? null);
    this.kama.restore(weighted?.kama ?? canonical?.ema ?? null);
    this.value = weighted?.value
      ? normalizeVolumeWeightedKAMAValue(weighted.value)
      : {
          kama: this.kama.indicator(),
          efficiencyRatio: this.efficiencyRatio.indicator(),
          relativeVolume: 1,
          effectiveEfficiencyRatio: this.efficiencyRatio.indicator(),
          alpha: isNormalizedAlpha(canonical?.alpha) ? canonical.alpha : 0,
        };
    this.delta = Number.isFinite(weighted?.delta)
      ? (weighted?.delta ?? 0)
      : (canonical?.ema.delta ?? 0);
  }
}

const HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_COMBINE_MODES = ["Either", "Both", "Weighted Average"] as const;

export type HighVolumeChopFastAdaptiveEMACombineMode =
  (typeof HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_COMBINE_MODES)[number];

export interface HighVolumeChopFastAdaptiveEMAIndicatorConfig {
  erLen: number;
  fastLen: number;
  slowLen: number;
  volumeLen: number;
  volumeCap: number;
  chopWeight: number;
  volumeWeight: number;
  chopPower: number;
  volumePower: number;
  combineMode: HighVolumeChopFastAdaptiveEMACombineMode;
}

export type HighVolumeChopFastAdaptiveEMAIndicatorOptions =
  Partial<HighVolumeChopFastAdaptiveEMAIndicatorConfig> & {
    warmupCount?: number;
  };

const DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG: HighVolumeChopFastAdaptiveEMAIndicatorConfig = {
  erLen: 20,
  fastLen: 5,
  slowLen: 50,
  volumeLen: 50,
  volumeCap: 3,
  chopWeight: 1,
  volumeWeight: 1,
  chopPower: 1,
  volumePower: 1,
  combineMode: "Either",
};

export interface HighVolumeChopFastAdaptiveEMAIndicatorValue {
  adaptiveEma: number;
  efficiencyRatio: number;
  chopScore: number;
  relativeVolume: number;
  volumeScore: number;
  speed: number;
  alpha: number;
}

export interface HighVolumeChopFastAdaptiveEMAIndicatorSnapshot {
  version: 1;
  config: HighVolumeChopFastAdaptiveEMAIndicatorConfig;
  efficiencyRatio: EfficiencyRatioIndicatorSnapshot;
  volumeEma: EMAIndicatorSnapshot;
  adaptiveEma: EMAIndicatorSnapshot;
  value: HighVolumeChopFastAdaptiveEMAIndicatorValue;
  delta: number;
}

export class HighVolumeChopFastAdaptiveEMAIndicator
  implements NumericTradingIndicator<HighVolumeChopFastAdaptiveEMAIndicatorSnapshot, PriceIndicatorInput>
{
  private readonly config: HighVolumeChopFastAdaptiveEMAIndicatorConfig;
  private readonly warmupCount: number;
  private readonly efficiencyRatio: EfficiencyRatioIndicator;
  private readonly volumeEma: EMAIndicator;
  private readonly adaptiveEma: EMAIndicator;
  private readonly alphaFast: number;
  private readonly alphaSlow: number;
  private value = emptyHighVolumeChopFastAdaptiveEMAIndicatorValue();
  private delta = 0;

  constructor(
    private readonly tradingApi: TradingApi,
    options: HighVolumeChopFastAdaptiveEMAIndicatorOptions = {},
  ) {
    this.config = normalizeHighVolumeChopFastAdaptiveEMAConfig(options);
    this.warmupCount =
      options.warmupCount ??
      Math.max(this.config.erLen + 1, this.config.fastLen, this.config.slowLen, this.config.volumeLen);
    assertPositiveWindow(this.warmupCount, "high-volume chop-fast adaptive EMA warmup");
    this.alphaFast = normalizeAlpha(this.config.fastLen);
    this.alphaSlow = normalizeAlpha(this.config.slowLen);
    this.efficiencyRatio = new EfficiencyRatioIndicator(this.config.erLen, tradingApi);
    this.volumeEma = new EMAIndicator(this.config.volumeLen, tradingApi, this.warmupCount);
    this.adaptiveEma = new EMAIndicator(this.alphaSlow, tradingApi, this.warmupCount);
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.warmupCount);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(
      this.efficiencyRatio.snapshot().values.length >= this.config.erLen + 1,
      "high-volume chop-fast adaptive EMA efficiency ratio",
    );
    assertWarmupValue(this.volumeEma.indicator() !== 0, "high-volume chop-fast adaptive EMA volume EMA");
    assertWarmupValue(this.adaptiveEma.indicator() !== 0, "high-volume chop-fast adaptive EMA");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    const volume = indicatorVolume(input);
    if (price <= 0 || volume <= 0) return;

    this.efficiencyRatio.onTick(input);
    this.volumeEma.onTick({ eventTime: input.eventTime, value: volume });

    const efficiencyRatio = this.efficiencyRatio.indicator();
    const chopScore = Math.pow(clampRatio(1 - efficiencyRatio), this.config.chopPower);
    const volumeAverage = this.volumeEma.indicator();
    const relativeVolume = clamp(volumeAverage !== 0 ? volume / volumeAverage : 1, 0, this.config.volumeCap);
    const volumeScoreBase =
      this.config.volumeCap > 1 ? (relativeVolume - 1) / (this.config.volumeCap - 1) : 0;
    const volumeScore = Math.pow(clampRatio(volumeScoreBase), this.config.volumePower);
    const speed = this.speed(chopScore, volumeScore);
    const alpha = this.alphaSlow + speed * (this.alphaFast - this.alphaSlow);

    this.adaptiveEma.onTick({ eventTime: input.eventTime, value: price, alpha });
    const adaptiveEma = this.adaptiveEma.indicator();
    this.delta = this.adaptiveEma.derivative();
    this.value = {
      adaptiveEma,
      efficiencyRatio,
      chopScore,
      relativeVolume,
      volumeScore,
      speed,
      alpha,
    };
  }

  indicator(): number {
    return this.value.adaptiveEma;
  }

  derivative(): number {
    return this.delta;
  }

  details(): HighVolumeChopFastAdaptiveEMAIndicatorValue {
    return { ...this.value };
  }

  snapshot(): HighVolumeChopFastAdaptiveEMAIndicatorSnapshot {
    return {
      version: 1,
      config: { ...this.config },
      efficiencyRatio: this.efficiencyRatio.snapshot(),
      volumeEma: this.volumeEma.snapshot(),
      adaptiveEma: this.adaptiveEma.snapshot(),
      value: { ...this.value },
      delta: this.delta,
    };
  }

  restore(snapshot: HighVolumeChopFastAdaptiveEMAIndicatorSnapshot | null): void {
    this.efficiencyRatio.restore(snapshot?.efficiencyRatio ?? null);
    this.volumeEma.restore(snapshot?.volumeEma ?? null);
    this.adaptiveEma.restore(snapshot?.adaptiveEma ?? null);
    this.value = snapshot?.value ? normalizeHighVolumeChopFastAdaptiveEMAValue(snapshot.value) : this.details();
    this.delta = Number.isFinite(snapshot?.delta) ? (snapshot?.delta ?? 0) : 0;
  }

  private speed(chopScore: number, volumeScore: number): number {
    switch (this.config.combineMode) {
      case "Either":
        return clampRatio(1 - (1 - chopScore) * (1 - volumeScore));
      case "Both":
        return clampRatio(chopScore * volumeScore);
      case "Weighted Average": {
        const weightSum = this.config.chopWeight + this.config.volumeWeight;
        return weightSum !== 0
          ? clampRatio((chopScore * this.config.chopWeight + volumeScore * this.config.volumeWeight) / weightSum)
          : 0;
      }
    }
  }
}

export interface DerivativeThresholdIndicatorValue {
  derivative: number;
  thresholded: number;
  signal: -1 | 0 | 1;
  passed: boolean;
}

export interface DerivativeThresholdIndicatorSnapshot<TChildSnapshot = unknown> {
  version: 1;
  child: TChildSnapshot;
  lowerThreshold: number;
  upperThreshold: number;
}

export class DerivativeThresholdIndicator<TChildSnapshot, TInput> implements TradingIndicator<
  DerivativeThresholdIndicatorSnapshot<TChildSnapshot>,
  DerivativeThresholdIndicatorValue,
  TInput
> {
  constructor(
    private readonly child: NumericTradingIndicator<TChildSnapshot, TInput>,
    private readonly lowerThreshold: number,
    private readonly upperThreshold = lowerThreshold,
  ) {}

  async warmup(): Promise<void> {
    await this.child.warmup();
  }

  onTick(input: TInput): void {
    this.child.onTick(input);
  }

  indicator(): DerivativeThresholdIndicatorValue {
    const derivative = this.child.derivative();
    const lowerThreshold = Math.max(0, this.lowerThreshold);
    const upperThreshold = Math.max(0, this.upperThreshold);
    if (derivative >= upperThreshold) {
      return { derivative, thresholded: derivative, signal: 1, passed: true };
    }
    if (derivative <= -lowerThreshold) {
      return { derivative, thresholded: derivative, signal: -1, passed: true };
    }
    return { derivative, thresholded: 0, signal: 0, passed: false };
  }

  snapshot(): DerivativeThresholdIndicatorSnapshot<TChildSnapshot> {
    return {
      version: 1,
      child: this.child.snapshot(),
      lowerThreshold: this.lowerThreshold,
      upperThreshold: this.upperThreshold,
    };
  }

  restore(snapshot: DerivativeThresholdIndicatorSnapshot<TChildSnapshot> | null): void {
    this.child.restore(snapshot?.child ?? null);
  }
}

export interface MACDIndicatorValue {
  macd: number;
  signal: number;
  histogram: number;
}

export interface MACDIndicatorSnapshot {
  version: 1;
  fast: EMAIndicatorSnapshot;
  slow: EMAIndicatorSnapshot;
  signal: EMAIndicatorSnapshot;
  value: MACDIndicatorValue;
}

export class MACDIndicator implements TradingIndicator<MACDIndicatorSnapshot, MACDIndicatorValue, PriceIndicatorInput> {
  private readonly fast: EMAIndicator;
  private readonly slow: EMAIndicator;
  private readonly signal: EMAIndicator;
  private value: MACDIndicatorValue = { macd: 0, signal: 0, histogram: 0 };

  constructor(
    fastPeriod: number,
    slowPeriod: number,
    signalPeriod: number,
    private readonly tradingApi: TradingApi,
    private readonly warmupCount = Math.max(fastPeriod, slowPeriod) + signalPeriod,
  ) {
    this.fast = new EMAIndicator(fastPeriod, tradingApi, warmupCount);
    this.slow = new EMAIndicator(slowPeriod, tradingApi, warmupCount);
    this.signal = new EMAIndicator(signalPeriod, tradingApi, warmupCount);
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.warmupCount);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.fast.indicator() !== 0 && this.slow.indicator() !== 0, "MACD");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    if (price <= 0) return;

    this.fast.onTick({ eventTime: input.eventTime, value: price });
    this.slow.onTick({ eventTime: input.eventTime, value: price });
    const macd = this.fast.indicator() - this.slow.indicator();
    this.signal.onTick({ eventTime: input.eventTime, value: macd });
    const signal = this.signal.indicator();
    this.value = {
      macd,
      signal,
      histogram: macd - signal,
    };
  }

  indicator(): MACDIndicatorValue {
    return { ...this.value };
  }

  snapshot(): MACDIndicatorSnapshot {
    return {
      version: 1,
      fast: this.fast.snapshot(),
      slow: this.slow.snapshot(),
      signal: this.signal.snapshot(),
      value: { ...this.value },
    };
  }

  restore(snapshot: MACDIndicatorSnapshot | null): void {
    this.fast.restore(snapshot?.fast ?? null);
    this.slow.restore(snapshot?.slow ?? null);
    this.signal.restore(snapshot?.signal ?? null);
    this.value = snapshot?.value ? { ...snapshot.value } : { macd: 0, signal: 0, histogram: 0 };
  }
}

export interface RSIIndicatorSnapshot {
  version: 1;
  previousPrice?: number;
  gain: EMAIndicatorSnapshot;
  loss: EMAIndicatorSnapshot;
}

export class RSIIndicator implements TradingIndicator<RSIIndicatorSnapshot, number, PriceIndicatorInput> {
  private previousPrice: number | undefined;
  private readonly gain: EMAIndicator;
  private readonly loss: EMAIndicator;

  constructor(
    private readonly period: number,
    private readonly tradingApi: TradingApi,
  ) {
    assertPositiveWindow(period, "RSI");
    this.gain = new EMAIndicator(1 / period, tradingApi, period + 1);
    this.loss = new EMAIndicator(1 / period, tradingApi, period + 1);
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.period + 1);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.previousPrice !== undefined, "RSI");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    if (price <= 0) return;

    if (this.previousPrice === undefined) {
      this.previousPrice = price;
      this.gain.onTick({ eventTime: input.eventTime, value: 0 });
      this.loss.onTick({ eventTime: input.eventTime, value: 0 });
      return;
    }

    const change = price - this.previousPrice;
    this.previousPrice = price;
    this.gain.onTick({ eventTime: input.eventTime, value: Math.max(0, change) });
    this.loss.onTick({ eventTime: input.eventTime, value: Math.max(0, -change) });
  }

  indicator(): number {
    const gain = this.gain.indicator();
    const loss = this.loss.indicator();
    if (loss <= 0) return gain > 0 ? 100 : 50;
    const rs = gain / loss;
    return 100 - 100 / (1 + rs);
  }

  snapshot(): RSIIndicatorSnapshot {
    return {
      version: 1,
      previousPrice: this.previousPrice,
      gain: this.gain.snapshot(),
      loss: this.loss.snapshot(),
    };
  }

  restore(snapshot: RSIIndicatorSnapshot | null): void {
    this.previousPrice = Number.isFinite(snapshot?.previousPrice) ? snapshot?.previousPrice : undefined;
    this.gain.restore(snapshot?.gain ?? null);
    this.loss.restore(snapshot?.loss ?? null);
  }
}

export interface BollingerBandsIndicatorValue {
  middle: number;
  upper: number;
  lower: number;
  stdDev: number;
  width: number;
  percentB: number;
}

export interface BollingerBandsIndicatorSnapshot {
  version: 1;
  entries: number[];
  sma: SMAIndicatorSnapshot;
}

export class BollingerBandsIndicator implements TradingIndicator<
  BollingerBandsIndicatorSnapshot,
  BollingerBandsIndicatorValue,
  PriceIndicatorInput
> {
  private readonly sma: SMAIndicator;
  private readonly stdDev = new RollingStdDevWindow();

  constructor(
    private readonly windowSize: number,
    private readonly tradingApi: TradingApi,
    private readonly stdDevMultiplier = 2,
  ) {
    assertPositiveWindow(windowSize, "Bollinger Bands");
    this.sma = new SMAIndicator(windowSize, tradingApi);
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.windowSize);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.stdDev.count() > 0, "Bollinger Bands");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    if (price <= 0) return;

    this.stdDev.add(price);
    while (this.stdDev.count() > this.windowSize) {
      this.stdDev.removeOldest();
    }
    this.sma.onTick(input);
  }

  indicator(): BollingerBandsIndicatorValue {
    const middle = this.sma.indicator();
    const stdDev = this.stdDev.absolute(middle);
    const bandOffset = stdDev * this.stdDevMultiplier;
    const upper = middle + bandOffset;
    const lower = middle - bandOffset;
    const width = middle > 0 ? (upper - lower) / middle : 0;
    const lastPrice = this.stdDev.latest() ?? middle;
    const percentB = upper > lower ? (lastPrice - lower) / (upper - lower) : 0;
    return { middle, upper, lower, stdDev, width, percentB };
  }

  snapshot(): BollingerBandsIndicatorSnapshot {
    return {
      version: 1,
      entries: this.stdDev.values(),
      sma: this.sma.snapshot(),
    };
  }

  restore(snapshot: BollingerBandsIndicatorSnapshot | null): void {
    this.stdDev.set(snapshot?.entries?.slice(-this.windowSize).filter(isPositiveFinite) ?? []);
    this.sma.restore(snapshot?.sma ?? null);
  }
}

export interface StdDevIndicatorEntry {
  eventTime: number;
  value: number;
}

export interface StdDevIndicatorSnapshot {
  version: 1;
  windowMs: number;
  entries: StdDevIndicatorEntry[];
  sum: number;
  sumSquares: number;
  value: number;
  delta: number;
}

export class StdDevIndicator implements NumericTradingIndicator<StdDevIndicatorSnapshot, ValueIndicatorInput> {
  private entries: StdDevIndicatorEntry[] = [];
  private readonly stdDev = new RollingStdDevWindow();
  private value = 0;
  private delta = 0;

  constructor(
    private readonly windowMs: number,
    private readonly tradingApi: TradingApi,
  ) {
    assertPositiveTimeWindow(windowMs, "standard deviation");
  }

  async warmup(initialValue?: number): Promise<void> {
    if (Number.isFinite(initialValue)) {
      this.onTick({ eventTime: 0, value: initialValue });
    }
    const candles = await warmupCandles(this.tradingApi, warmupCountForTimeWindow(this.windowMs));
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.entries.length > 0, "standard deviation");
  }

  onTick(input: ValueIndicatorInput): void {
    const value = indicatorValue(input);
    if (!Number.isFinite(value) || !Number.isFinite(input.eventTime)) return;

    const previous = this.value;
    this.entries.push({ eventTime: input.eventTime, value });
    this.stdDev.add(value);
    this.prune(input.eventTime - this.windowMs);
    this.value = this.stdDev.relative();
    this.delta = this.value - previous;
  }

  indicator(): number {
    return this.value;
  }

  derivative(): number {
    return this.delta;
  }

  snapshot(): StdDevIndicatorSnapshot {
    return {
      version: 1,
      windowMs: this.windowMs,
      entries: this.entries.map((entry) => ({ ...entry })),
      sum: this.stdDev.sum(),
      sumSquares: this.stdDev.sumSquares(),
      value: this.value,
      delta: this.delta,
    };
  }

  restore(snapshot: StdDevIndicatorSnapshot | null): void {
    const entries =
      snapshot?.entries
        ?.filter((entry) => Number.isFinite(entry.eventTime) && Number.isFinite(entry.value))
        .sort((left, right) => left.eventTime - right.eventTime) ?? [];
    const latestTime = entries.at(-1)?.eventTime;
    this.entries =
      latestTime === undefined ? [] : entries.filter((entry) => entry.eventTime >= latestTime - this.windowMs);
    this.stdDev.set(this.entries.map((entry) => entry.value));
    this.value = this.stdDev.relative();
    this.delta = Number.isFinite(snapshot?.delta) ? (snapshot?.delta ?? 0) : 0;
  }

  private prune(cutoffTime: number): void {
    while (this.entries.length > 0 && (this.entries[0]?.eventTime ?? 0) < cutoffTime) {
      const removed = this.entries.shift();
      if (removed) {
        this.stdDev.removeOldest();
      }
    }
  }
}

export interface LinearRegressionIndicatorEntry {
  eventTime: number;
  value: number;
  weight: number;
}

export interface LinearRegressionIndicatorValue {
  slope: number;
  intercept: number;
  current: number;
  fitted: number;
  next: number;
  rSquared: number;
  count: number;
}

export interface LinearRegressionIndicatorSnapshot<TChildSnapshot = unknown> {
  version: 1;
  entries: LinearRegressionIndicatorEntry[];
  child?: TChildSnapshot;
}

export interface LinearRegressionIndicatorOptions<TChildSnapshot> {
  child?: NumericTradingIndicator<TChildSnapshot, ValueIndicatorInput>;
  warmupCount?: number;
}

export interface VolumeWeightedLinearRegressionIndicatorOptions<TChildSnapshot> {
  child?: NumericTradingIndicator<TChildSnapshot, WeightedValueIndicatorInput>;
  warmupCount?: number;
}

abstract class BaseLinearRegressionIndicator<TChildSnapshot, TInput extends ValueIndicatorInput>
  implements TradingIndicator<LinearRegressionIndicatorSnapshot<TChildSnapshot>, LinearRegressionIndicatorValue, TInput>
{
  private readonly child?: NumericTradingIndicator<TChildSnapshot, TInput>;
  private readonly warmupCount: number;
  private entries: LinearRegressionIndicatorEntry[] = [];
  private value: LinearRegressionIndicatorValue = emptyLinearRegressionValue();

  constructor(
    private readonly windowSize: number,
    private readonly tradingApi: TradingApi,
    options: {
      child?: NumericTradingIndicator<TChildSnapshot, TInput>;
      warmupCount?: number;
    },
  ) {
    assertPositiveWindow(windowSize, "linear regression");
    this.child = options.child;
    this.warmupCount = options.warmupCount ?? windowSize;
  }

  async warmup(): Promise<void> {
    if (this.child) {
      await this.child.warmup();
    }
    const candles = await warmupCandles(this.tradingApi, this.warmupCount);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime } as TInput);
    }
    assertWarmupValue(this.entries.length > 0, "linear regression");
  }

  onTick(input: TInput): void {
    const sourceValue = this.sourceValue(input);
    if (!Number.isFinite(sourceValue)) {
      return;
    }

    this.entries.push({
      eventTime: input.eventTime,
      value: sourceValue,
      weight: this.entryWeight(input),
    });
    while (this.entries.length > this.windowSize) {
      this.entries.shift();
    }
    this.value = linearRegression(this.entries);
  }

  indicator(): LinearRegressionIndicatorValue {
    return { ...this.value };
  }

  derivative(): number {
    return this.value.slope;
  }

  predict(stepsAhead = 1): number {
    const steps = Math.max(0, Math.floor(stepsAhead));
    if (this.entries.length === 0) {
      return 0;
    }
    return this.value.intercept + this.value.slope * (this.entries.length - 1 + steps);
  }

  snapshot(): LinearRegressionIndicatorSnapshot<TChildSnapshot> {
    return {
      version: 1,
      entries: this.entries.map((entry) => ({ ...entry })),
      child: this.child?.snapshot(),
    };
  }

  restore(snapshot: LinearRegressionIndicatorSnapshot<TChildSnapshot> | null): void {
    this.child?.restore(snapshot?.child ?? null);
    this.entries =
      snapshot?.entries
        ?.filter(
          (entry) => Number.isFinite(entry.eventTime) && Number.isFinite(entry.value) && Number.isFinite(entry.weight),
        )
        .slice(-this.windowSize)
        .map((entry) => ({
          eventTime: entry.eventTime,
          value: entry.value,
          weight: Math.max(0, entry.weight),
        })) ?? [];
    this.value = linearRegression(this.entries);
  }

  protected abstract entryWeight(input: TInput): number;

  private sourceValue(input: TInput): number {
    if (!this.child) {
      return indicatorValue(input);
    }

    this.child.onTick(input);
    return this.child.indicator();
  }
}

export class LinearRegressionIndicator<TChildSnapshot = unknown>
  extends BaseLinearRegressionIndicator<TChildSnapshot, ValueIndicatorInput>
  implements TradingIndicator<
    LinearRegressionIndicatorSnapshot<TChildSnapshot>,
    LinearRegressionIndicatorValue,
    ValueIndicatorInput
  >
{
  constructor(
    windowSize: number,
    tradingApi: TradingApi,
    options: LinearRegressionIndicatorOptions<TChildSnapshot> = {},
  ) {
    super(windowSize, tradingApi, options);
  }

  protected entryWeight(): number {
    return 1;
  }
}

export class VolumeWeightedLinearRegressionIndicator<
  TChildSnapshot = unknown,
> extends BaseLinearRegressionIndicator<TChildSnapshot, WeightedValueIndicatorInput> {
  constructor(
    windowSize: number,
    tradingApi: TradingApi,
    options: VolumeWeightedLinearRegressionIndicatorOptions<TChildSnapshot> = {},
  ) {
    super(windowSize, tradingApi, options);
  }

  protected entryWeight(input: WeightedValueIndicatorInput): number {
    return indicatorWeight(input);
  }
}

export interface AvgCandleIndicatorSnapshot {
  version: 1;
  windowSize: number;
  samples: AvgCandleIndicatorValue[];
  sum: AvgCandleIndicatorValue;
}

// OHLC values are candle-shape returns, not prices: for each candle we use the
// body midpoint `(open + close) / 2` as zero, then store `(price - midpoint) / midpoint`.
// Volume is kept as raw volume and averaged over the same window.
export type AvgCandleIndicatorValue = Pick<Candle, "open" | "high" | "low" | "close" | "volume">;

export class AvgCandleIndicator
  implements TradingIndicator<AvgCandleIndicatorSnapshot, AvgCandleIndicatorValue, PriceIndicatorInput>
{
  private samples: AvgCandleIndicatorValue[] = [];
  private sum = emptyAvgCandleValue();

  constructor(
    private readonly windowSize: number,
    private readonly tradingApi: TradingApi,
  ) {
    assertPositiveWindow(windowSize, "average candle");
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.windowSize);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.samples.length > 0, "average candle");
  }

  onTick(input: PriceIndicatorInput): void {
    const candle = input.candle ? normalizeCandle(input.candle) : tickAsCandle(input.price, input.quantity);
    const sample = candle ? relativeBodyMidpointCandle(candle) : undefined;
    if (!sample) return;

    this.samples.push(sample);
    addAvgCandleValue(this.sum, sample);
    while (this.samples.length > this.windowSize) {
      const removed = this.samples.shift();
      if (removed) {
        subtractAvgCandleValue(this.sum, removed);
      }
    }
  }

  indicator(): AvgCandleIndicatorValue {
    const count = this.samples.length;
    if (count === 0) return emptyAvgCandleValue();

    return {
      open: this.sum.open / count,
      high: this.sum.high / count,
      low: this.sum.low / count,
      close: this.sum.close / count,
      volume: this.sum.volume / count,
    };
  }

  snapshot(): AvgCandleIndicatorSnapshot {
    return {
      version: 1,
      windowSize: this.windowSize,
      samples: this.samples.map((sample) => ({ ...sample })),
      sum: { ...this.sum },
    };
  }

  restore(snapshot: AvgCandleIndicatorSnapshot | null): void {
    this.samples =
      snapshot?.samples
        ?.map((sample) => normalizeAvgCandleSample(sample))
        .filter((sample): sample is AvgCandleIndicatorValue => sample !== undefined)
        .slice(-this.windowSize) ?? [];
    this.sum = this.samples.reduce((total, sample) => addAvgCandleValue(total, sample), emptyAvgCandleValue());
  }
}

export interface PriceRangeIndicatorValue {
  low: number;
  high: number;
}

export interface PriceRangeIndicatorSnapshot {
  version: 1;
  windowSize: number;
  entries: number[];
  values: number[];
}

export class PriceRangeIndicator
  implements TradingIndicator<PriceRangeIndicatorSnapshot, PriceRangeIndicatorValue, PriceIndicatorInput>
{
  // `entries` preserves rolling-window order; `values` is sorted for O(1) low/high reads.
  private entries: number[] = [];
  private values: number[] = [];

  constructor(
    private readonly windowSize: number,
    private readonly tradingApi: TradingApi,
  ) {
    assertPositiveWindow(windowSize, "price range");
  }

  async warmup(): Promise<void> {
    const candles = await warmupCandles(this.tradingApi, this.windowSize);
    for (const candle of candles) {
      this.onTick({ candle, eventTime: candle.closeTime });
    }
    assertWarmupValue(this.values.length > 0, "price range");
  }

  onTick(input: PriceIndicatorInput): void {
    const price = indicatorPrice(input);
    if (price <= 0) return;

    this.entries.push(price);
    insertSorted(this.values, price);
    while (this.entries.length > this.windowSize) {
      const removed = this.entries.shift();
      if (removed !== undefined) {
        removeSorted(this.values, removed);
      }
    }
  }

  indicator(): PriceRangeIndicatorValue {
    if (this.values.length === 0) return { low: 0, high: 0 };

    const low = this.values[0] ?? 0;
    const high = this.values.at(-1) ?? 0;
    return { low, high };
  }

  snapshot(): PriceRangeIndicatorSnapshot {
    return {
      version: 1,
      windowSize: this.windowSize,
      entries: this.entries.slice(),
      values: this.values.slice(),
    };
  }

  restore(snapshot: PriceRangeIndicatorSnapshot | null): void {
    this.entries = (snapshot?.entries ?? snapshot?.values)?.slice(-this.windowSize).filter(isPositiveFinite) ?? [];
    this.values = this.entries.slice().sort((left, right) => left - right);
  }
}

async function warmupCandles(tradingApi: TradingApi, count: number): Promise<Candle[]> {
  return tradingApi.getHistory({
    intervalMs: 1000,
    count,
  });
}

function normalizeVolumeWeightedKAMAConfig(
  options: VolumeWeightedKAMAIndicatorOptions,
): VolumeWeightedKAMAIndicatorConfig {
  const config: VolumeWeightedKAMAIndicatorConfig = {
    efficiencyPeriod: options.efficiencyPeriod ?? DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG.efficiencyPeriod,
    fastPeriod: options.fastPeriod ?? DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG.fastPeriod,
    slowPeriod: options.slowPeriod ?? DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG.slowPeriod,
    power: options.power ?? DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG.power,
    volumePeriod: options.volumePeriod ?? DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG.volumePeriod,
    volumeCap: options.volumeCap ?? DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG.volumeCap,
    volumePower: options.volumePower ?? DEFAULT_VOLUME_WEIGHTED_KAMA_CONFIG.volumePower,
  };
  assertMinimumWindow(config.efficiencyPeriod, 1, "volume-weighted KAMA efficiency ratio");
  assertFiniteMinimum(config.fastPeriod, Number.EPSILON, "volume-weighted KAMA fast EMA");
  assertFiniteMinimum(config.slowPeriod, Number.EPSILON, "volume-weighted KAMA slow EMA");
  assertFiniteMinimum(config.power, 0.1, "volume-weighted KAMA power");
  assertMinimumWindow(config.volumePeriod, 1, "volume-weighted KAMA volume EMA");
  assertFiniteMinimum(config.volumeCap, 1, "volume-weighted KAMA relative volume cap");
  assertFiniteMinimum(config.volumePower, 0, "volume-weighted KAMA volume power");
  return config;
}

function emptyVolumeWeightedKAMAValue(): VolumeWeightedKAMAIndicatorValue {
  return {
    kama: 0,
    efficiencyRatio: 0,
    relativeVolume: 1,
    effectiveEfficiencyRatio: 0,
    alpha: 0,
  };
}

function normalizeVolumeWeightedKAMAValue(
  value: VolumeWeightedKAMAIndicatorValue | undefined,
): VolumeWeightedKAMAIndicatorValue {
  return value
    ? {
        kama: finiteNumber(value.kama),
        efficiencyRatio: clampRatio(value.efficiencyRatio),
        relativeVolume: Math.max(0, finiteNumber(value.relativeVolume)),
        effectiveEfficiencyRatio: clampRatio(value.effectiveEfficiencyRatio),
        alpha: isNormalizedAlpha(value.alpha) ? value.alpha : 0,
      }
    : emptyVolumeWeightedKAMAValue();
}

function normalizeHighVolumeChopFastAdaptiveEMAConfig(
  options: HighVolumeChopFastAdaptiveEMAIndicatorOptions,
): HighVolumeChopFastAdaptiveEMAIndicatorConfig {
  const config: HighVolumeChopFastAdaptiveEMAIndicatorConfig = {
    erLen: options.erLen ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.erLen,
    fastLen: options.fastLen ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.fastLen,
    slowLen: options.slowLen ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.slowLen,
    volumeLen: options.volumeLen ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.volumeLen,
    volumeCap: options.volumeCap ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.volumeCap,
    chopWeight: options.chopWeight ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.chopWeight,
    volumeWeight: options.volumeWeight ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.volumeWeight,
    chopPower: options.chopPower ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.chopPower,
    volumePower: options.volumePower ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.volumePower,
    combineMode: options.combineMode ?? DEFAULT_HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_CONFIG.combineMode,
  };

  assertMinimumWindow(config.erLen, 2, "high-volume chop-fast adaptive EMA efficiency ratio");
  assertMinimumWindow(config.fastLen, 1, "high-volume chop-fast adaptive EMA fast EMA");
  assertMinimumWindow(config.slowLen, 2, "high-volume chop-fast adaptive EMA slow EMA");
  assertMinimumWindow(config.volumeLen, 1, "high-volume chop-fast adaptive EMA volume EMA");
  assertFiniteMinimum(config.volumeCap, 1, "high-volume chop-fast adaptive EMA relative volume cap");
  assertFiniteMinimum(config.chopWeight, 0, "high-volume chop-fast adaptive EMA chop weight");
  assertFiniteMinimum(config.volumeWeight, 0, "high-volume chop-fast adaptive EMA volume weight");
  assertFiniteMinimum(config.chopPower, 0.1, "high-volume chop-fast adaptive EMA chop power");
  assertFiniteMinimum(config.volumePower, 0.1, "high-volume chop-fast adaptive EMA volume power");
  if (!isHighVolumeChopFastAdaptiveEMACombineMode(config.combineMode)) {
    throw new Error("high-volume chop-fast adaptive EMA combine mode is invalid.");
  }

  return config;
}

function emptyHighVolumeChopFastAdaptiveEMAIndicatorValue(): HighVolumeChopFastAdaptiveEMAIndicatorValue {
  return {
    adaptiveEma: 0,
    efficiencyRatio: 0,
    chopScore: 0,
    relativeVolume: 0,
    volumeScore: 0,
    speed: 0,
    alpha: 0,
  };
}

function normalizeHighVolumeChopFastAdaptiveEMAValue(
  value: HighVolumeChopFastAdaptiveEMAIndicatorValue,
): HighVolumeChopFastAdaptiveEMAIndicatorValue {
  return {
    adaptiveEma: finiteNumber(value.adaptiveEma),
    efficiencyRatio: clampRatio(value.efficiencyRatio),
    chopScore: clampRatio(value.chopScore),
    relativeVolume: Math.max(0, finiteNumber(value.relativeVolume)),
    volumeScore: clampRatio(value.volumeScore),
    speed: clampRatio(value.speed),
    alpha: isNormalizedAlpha(value.alpha) ? value.alpha : 0,
  };
}

function isHighVolumeChopFastAdaptiveEMACombineMode(
  value: string,
): value is HighVolumeChopFastAdaptiveEMACombineMode {
  return HIGH_VOLUME_CHOP_FAST_ADAPTIVE_EMA_COMBINE_MODES.some((mode) => mode === value);
}

function indicatorPrice(input: PriceIndicatorInput): number {
  return cleanPositive(input.price ?? input.candle?.close ?? 0);
}

function indicatorValue(input: ValueIndicatorInput): number {
  if (isNumericInput(input)) {
    return Number.isFinite(input.value) ? input.value : Number.NaN;
  }
  const price = indicatorPrice(input);
  return price > 0 ? price : Number.NaN;
}

function indicatorVolume(input: PriceIndicatorInput): number {
  const volume = input.quantity ?? input.candle?.volume ?? 0;
  return Number.isFinite(volume) && volume > 0 ? volume : 0;
}

function indicatorWeight(input: WeightedValueIndicatorInput): number {
  if (isNumericInput(input)) {
    const weight = input.weight;
    return typeof weight === "number" && Number.isFinite(weight) && weight > 0 ? weight : 0;
  }
  return indicatorVolume(input);
}

function isNumericInput(input: ValueIndicatorInput): input is WeightedNumericIndicatorInput {
  return "value" in input;
}

function tickAsCandle(
  price: number | undefined,
  quantity = 0,
): Pick<Candle, "open" | "high" | "low" | "close" | "volume"> | undefined {
  const cleanPrice = cleanPositive(price ?? 0);
  if (cleanPrice <= 0) {
    return undefined;
  }
  return {
    open: cleanPrice,
    high: cleanPrice,
    low: cleanPrice,
    close: cleanPrice,
    volume: Math.max(0, Number.isFinite(quantity) ? quantity : 0),
  };
}

function normalizeCandle(
  candle: Pick<Candle, "open" | "high" | "low" | "close" | "volume">,
): Pick<Candle, "open" | "high" | "low" | "close" | "volume"> | undefined {
  const open = cleanPositive(candle.open);
  const high = cleanPositive(candle.high);
  const low = cleanPositive(candle.low);
  const close = cleanPositive(candle.close);
  if (open <= 0 || high <= 0 || low <= 0 || close <= 0) {
    return undefined;
  }
  return {
    open,
    high: Math.max(open, high, low, close),
    low: Math.min(open, high, low, close),
    close,
    volume: Math.max(0, Number.isFinite(candle.volume) ? candle.volume : 0),
  };
}

function relativeBodyMidpointCandle(
  candle: Pick<Candle, "open" | "high" | "low" | "close" | "volume">,
): AvgCandleIndicatorValue | undefined {
  const midpoint = (candle.open + candle.close) / 2;
  if (!Number.isFinite(midpoint) || midpoint <= 0) {
    return undefined;
  }

  return {
    open: (candle.open - midpoint) / midpoint,
    high: (candle.high - midpoint) / midpoint,
    low: (candle.low - midpoint) / midpoint,
    close: (candle.close - midpoint) / midpoint,
    volume: candle.volume,
  };
}

function normalizeAvgCandleSample(sample: AvgCandleIndicatorValue): AvgCandleIndicatorValue | undefined {
  if (
    !Number.isFinite(sample.open) ||
    !Number.isFinite(sample.high) ||
    !Number.isFinite(sample.low) ||
    !Number.isFinite(sample.close)
  ) {
    return undefined;
  }

  return {
    open: sample.open,
    high: sample.high,
    low: sample.low,
    close: sample.close,
    volume: Math.max(0, Number.isFinite(sample.volume) ? sample.volume : 0),
  };
}

function emptyAvgCandleValue(): AvgCandleIndicatorValue {
  return { open: 0, high: 0, low: 0, close: 0, volume: 0 };
}

function addAvgCandleValue(target: AvgCandleIndicatorValue, value: AvgCandleIndicatorValue): AvgCandleIndicatorValue {
  target.open += value.open;
  target.high += value.high;
  target.low += value.low;
  target.close += value.close;
  target.volume += value.volume;
  return target;
}

function subtractAvgCandleValue(
  target: AvgCandleIndicatorValue,
  value: AvgCandleIndicatorValue,
): AvgCandleIndicatorValue {
  target.open -= value.open;
  target.high -= value.high;
  target.low -= value.low;
  target.close -= value.close;
  target.volume -= value.volume;
  return target;
}

function cleanPositive(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 0;
}

function finiteNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

function normalizeWeights(weights: readonly number[] | undefined, count: number): number[] {
  const normalized = weights?.slice(-count).map((weight) => (Number.isFinite(weight) && weight > 0 ? weight : 0)) ?? [];
  while (normalized.length < count) {
    normalized.unshift(1);
  }
  return normalized;
}

function normalizeAlpha(periodOrAlpha: number): number {
  if (!Number.isFinite(periodOrAlpha) || periodOrAlpha <= 0) {
    return 1;
  }
  return periodOrAlpha <= 1 ? clampRatio(periodOrAlpha) : 2 / (periodOrAlpha + 1);
}

function isNormalizedAlpha(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function clampRatio(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function emptyLinearRegressionValue(): LinearRegressionIndicatorValue {
  return {
    slope: 0,
    intercept: 0,
    current: 0,
    fitted: 0,
    next: 0,
    rSquared: 0,
    count: 0,
  };
}

function linearRegression(entries: readonly LinearRegressionIndicatorEntry[]): LinearRegressionIndicatorValue {
  const count = entries.length;
  if (count === 0) {
    return emptyLinearRegressionValue();
  }

  const weights = regressionWeights(entries);
  const totalWeight = weights.reduce((total, weight) => total + weight, 0);
  if (totalWeight <= 0) {
    return emptyLinearRegressionValue();
  }

  let sumX = 0;
  let sumY = 0;
  for (let index = 0; index < count; index += 1) {
    const weight = weights[index] ?? 1;
    sumX += index * weight;
    sumY += (entries[index]?.value ?? 0) * weight;
  }

  const meanX = sumX / totalWeight;
  const meanY = sumY / totalWeight;
  let varianceX = 0;
  let varianceY = 0;
  let covariance = 0;
  for (let index = 0; index < count; index += 1) {
    const weight = weights[index] ?? 1;
    const x = index - meanX;
    const y = (entries[index]?.value ?? 0) - meanY;
    varianceX += weight * x * x;
    varianceY += weight * y * y;
    covariance += weight * x * y;
  }

  const slope = varianceX > 0 ? covariance / varianceX : 0;
  const intercept = meanY - slope * meanX;
  const fitted = intercept + slope * (count - 1);
  const next = intercept + slope * count;
  const current = entries.at(-1)?.value ?? 0;
  const rSquared = varianceX > 0 && varianceY > 0 ? clampRatio((covariance * covariance) / (varianceX * varianceY)) : 0;

  return {
    slope,
    intercept,
    current,
    fitted,
    next,
    rSquared,
    count,
  };
}

function regressionWeights(entries: readonly LinearRegressionIndicatorEntry[]): number[] {
  const weights = entries.map((entry) => (Number.isFinite(entry.weight) && entry.weight > 0 ? entry.weight : 0));
  return weights.some((weight) => weight > 0) ? weights : entries.map(() => 1);
}

class RollingStdDevWindow {
  private entries: number[] = [];
  private total = 0;
  private totalSquares = 0;

  add(value: number): void {
    this.entries.push(value);
    this.total += value;
    this.totalSquares += value * value;
  }

  removeOldest(): number | undefined {
    const value = this.entries.shift();
    if (value !== undefined) {
      this.total -= value;
      this.totalSquares -= value * value;
    }
    return value;
  }

  set(values: readonly number[]): void {
    this.entries = [];
    this.total = 0;
    this.totalSquares = 0;
    for (const value of values) {
      if (Number.isFinite(value)) {
        this.add(value);
      }
    }
  }

  count(): number {
    return this.entries.length;
  }

  latest(): number | undefined {
    return this.entries.at(-1);
  }

  values(): number[] {
    return this.entries.slice();
  }

  sum(): number {
    return this.total;
  }

  sumSquares(): number {
    return this.totalSquares;
  }

  mean(): number {
    return this.entries.length > 0 ? this.total / this.entries.length : 0;
  }

  absolute(mean = this.mean()): number {
    const count = this.entries.length;
    if (count === 0) return 0;

    const variance = Math.max(0, this.totalSquares / count - mean * mean);
    return Math.sqrt(variance);
  }

  relative(): number {
    const mean = this.mean();
    return mean > 0 ? this.absolute(mean) / mean : 0;
  }
}

function insertSorted(values: number[], value: number): void {
  values.splice(lowerBound(values, value), 0, value);
}

function removeSorted(values: number[], value: number): void {
  const index = lowerBound(values, value);
  if (values[index] === value) {
    values.splice(index, 1);
  }
}

function lowerBound(values: readonly number[], value: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if ((values[mid] ?? 0) < value) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function assertPositiveWindow(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 1 || Math.floor(value) !== value) {
    throw new Error(`${label} indicator window size must be a positive integer.`);
  }
}

function assertMinimumWindow(value: number, min: number, label: string): void {
  assertPositiveWindow(value, label);
  if (value < min) {
    throw new Error(`${label} indicator window size must be at least ${min}.`);
  }
}

function assertFiniteMinimum(value: number, min: number, label: string): void {
  if (!Number.isFinite(value) || value < min) {
    throw new Error(`${label} must be at least ${min}.`);
  }
}

function assertWarmupValue(condition: boolean, label: string): void {
  if (!condition) {
    throw new Error(`${label} indicator warmup did not produce a value.`);
  }
}

function assertPositiveTimeWindow(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 1 || Math.floor(value) !== value) {
    throw new Error(`${label} indicator time window must be a positive integer in milliseconds.`);
  }
}

function warmupCountForTimeWindow(windowMs: number): number {
  return Math.max(1, Math.ceil(windowMs / 1000) + 1);
}

function clampInteger(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, Math.floor(value)));
}
