/** Causal native-second inputs, with exactly 64 completed candles of support. */
export const EVENT_SECOND_DYNAMICS = ["return-1s-bps", "ema2-acceleration-1s-bps", "ema2-slope-2s-bps", "rsi2-centered"] as const;
export const NATIVE_SECOND_EVENT_FEATURES = [...EVENT_SECOND_DYNAMICS,
  "log1p-rv-60s-bps", "range-1s-bps", "close-location-1s", "log1p-volume-ratio-60s", "active-fraction-60s",
  "run-sign-1s", "log1p-run-age-capped-63s", "run-return-capped-63s-bps"] as const;
export const NATIVE_SECOND_WARMUP = 63;
export const NATIVE_SECOND_CONTEXT_LAGS = [5, 15, 60, 300, 900, 3600, 14400] as const;
export const NATIVE_SECOND_CONTEXT_FEATURES = [...NATIVE_SECOND_EVENT_FEATURES,
  ...NATIVE_SECOND_CONTEXT_LAGS.map(seconds => `return-${seconds}s-bps`)] as const;
export const NATIVE_SECOND_CONTEXT_WARMUP = 14400;
export const NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES = [...NATIVE_SECOND_CONTEXT_FEATURES,
  "log1p-rms-return-3600s-bps", "log1p-rms-return-900s-minus-3600s",
  "log1p-rms-return-1800s-minus-3600s", "log1p-rms-return-14400s-minus-3600s"] as const;
export const NATIVE_SECOND_TRADE_FLOW_FEATURES = [
  "spot-aggregate-count-imbalance-1s",
  "spot-last-aggressor-side-1s",
] as const;
export const NATIVE_SECOND_FLOW_CONTEXT_FEATURES = [...NATIVE_SECOND_CONTEXT_FEATURES,
  ...NATIVE_SECOND_TRADE_FLOW_FEATURES] as const;
export const NATIVE_SECOND_TRADE_FLOW_VWAP_FEATURES = [...NATIVE_SECOND_TRADE_FLOW_FEATURES,
  "spot-flow-vwap-gap-1s"] as const;
export const NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES = [...NATIVE_SECOND_CONTEXT_FEATURES,
  ...NATIVE_SECOND_TRADE_FLOW_VWAP_FEATURES] as const;
/** A separate sign-head basis selected from the long-window feature audits.
 * It deliberately leaves the saved event-tree/magnitude basis unchanged. */
export const NATIVE_SECOND_SELECTED_SIGN_FEATURES = [...NATIVE_SECOND_CONTEXT_FEATURES,
  "return-lag-2s-bps", "haar-adjacent-contrast-16s",
  ...NATIVE_SECOND_TRADE_FLOW_FEATURES,
  "spot-flow-last-side-lag-2s", "spot-flow-quote-imbalance-1s",
  "spot-flow-quote-imbalance-ema-2", "spot-flow-maximum-skew-1s"] as const;
export const NATIVE_SECOND_DAY_CONTEXT_LAGS = [...NATIVE_SECOND_CONTEXT_LAGS, 43200, 86400] as const;
export const NATIVE_SECOND_DAY_CONTEXT_FEATURES = [...NATIVE_SECOND_EVENT_FEATURES,
  ...NATIVE_SECOND_DAY_CONTEXT_LAGS.map(seconds => `return-${seconds}s-bps`)] as const;
export const NATIVE_SECOND_DAY_CONTEXT_WARMUP = 86400;
type SecondClose = { openTime: number; close: number };
type SecondCandle = SecondClose & { open: number; high: number; low: number; volume: number;
  nativeTradeFlow?: { availableAt: number; aggregateCountImbalance: number; lastAggressorSide: number;
    buyerSellerVwapGap?: number; aggressiveBuyQuoteVolume?: number; aggressiveSellQuoteVolume?: number;
    aggressiveBuyMaxAggregateQuantity?: number; aggressiveSellMaxAggregateQuantity?: number } };

/** Exactly 64 contiguous completed second closes; no carried recursive state.
 * EMA uses alpha=2/3, RSI Wilder alpha=1/2 and arithmetic price changes. */
export function eventSecondDynamics(c: readonly SecondClose[], i: number): number[] | null {
  if (i < NATIVE_SECOND_WARMUP || !c[i]) return null;
  let ema = c[i - 63].close, gain = 0, loss = 0, lag1 = ema, lag2 = ema;
  for (let j = i - 63; j <= i; j++) {
    if (c[j].openTime !== c[i].openTime - (i - j) * 1000 || !(c[j].close > 0) || !Number.isFinite(c[j].close)) return null;
    if (j === i - 63) continue;
    const change = c[j].close - c[j - 1].close;
    lag2 = lag1; lag1 = ema; ema += (2 / 3) * (c[j].close - ema);
    gain += 0.5 * (Math.max(0, change) - gain); loss += 0.5 * (Math.max(0, -change) - loss);
  }
  const rsi = gain + loss > 0 ? (gain - loss) / (gain + loss) : 0;
  return [Math.log(c[i].close / c[i - 1].close) * 10000,
    (Math.log(ema / lag1) - Math.log(lag1 / lag2)) * 10000, Math.log(ema / lag2) * 5000, rsi];
}

export function nativeSecondEventFeatures(c: readonly SecondCandle[], i: number): number[] {
  const dynamics = eventSecondDynamics(c, i);
  if (!dynamics) throw new Error("Native second features require 64 contiguous completed seconds");
  let variance = 0, volume = 0, active = 0;
  for (let j = i - 59; j <= i; j++) {
    const row = c[j];
    if (![row.open, row.high, row.low, row.volume].every(Number.isFinite) || row.low <= 0 || row.volume < 0
      || row.low > Math.min(row.open, row.close) || row.high < Math.max(row.open, row.close))
      throw new Error("Invalid native second OHLCV");
    const r = Math.log(row.close / c[j - 1].close);
    variance += r * r; volume += row.volume; active += Number(row.volume > 0);
  }
  const row = c[i], sign = Math.sign(row.close - c[i - 1].close);
  let age = 1;
  while (age < 63 && Math.sign(c[i - age].close - c[i - age - 1].close) === sign) age++;
  return [...dynamics, Math.log1p(Math.sqrt(variance) * 10000), Math.log(row.high / row.low) * 10000,
    row.high > row.low ? (2 * row.close - row.high - row.low) / (row.high - row.low) : 0,
    Math.log1p(volume ? row.volume * 60 / volume : 0), active / 60,
    sign, Math.log1p(age), Math.log(row.close / c[i - age].close) * 10000];
}

/** Slow context still uses exact completed second closes. Each return needs
 * only its two endpoints, whose timestamps are checked against its true lag. */
export function nativeSecondContextFeatures(c: readonly SecondCandle[], i: number): number[] {
  if (i < NATIVE_SECOND_CONTEXT_WARMUP) throw new Error("Native context requires four hours of price history");
  return nativeSecondLagFeatures(c, i, NATIVE_SECOND_CONTEXT_LAGS);
}

interface NativeVariancePrefix { sums: Float64Array; segmentStarts: Int32Array; }
const nativeVariancePrefixes = new WeakMap<readonly SecondCandle[], NativeVariancePrefix>();
function nativeVariancePrefix(c: readonly SecondCandle[]): NativeVariancePrefix {
  const cached = nativeVariancePrefixes.get(c);
  if (cached) return cached;
  const sums = new Float64Array(c.length), segmentStarts = new Int32Array(c.length);
  for (let i = 1; i < c.length; i++) {
    const contiguous = c[i].openTime === c[i - 1].openTime + 1000
      && c[i].close > 0 && c[i - 1].close > 0 && Number.isFinite(c[i].close) && Number.isFinite(c[i - 1].close);
    segmentStarts[i] = contiguous ? segmentStarts[i - 1] : i;
    const value = contiguous ? Math.log(c[i].close / c[i - 1].close) : 0;
    sums[i] = contiguous ? sums[i - 1] + value * value : 0;
  }
  const result = { sums, segmentStarts };
  nativeVariancePrefixes.set(c, result);
  return result;
}

/** Long volatility is a separate state from endpoint return. Prefix sums keep
 * repeated fixed-stride event screens linear in source length. */
export function nativeSecondVolatilityContextFeatures(c: readonly SecondCandle[], i: number): number[] {
  if (i < NATIVE_SECOND_CONTEXT_WARMUP) throw new Error("Native volatility context requires four hours of price history");
  const prefix = nativeVariancePrefix(c), windows = [900, 1800, 3600, 14400] as const;
  if (prefix.segmentStarts[i] > i - windows.at(-1)!) throw new Error("Native volatility context crosses missing seconds");
  const levels = windows.map(window => Math.log1p(Math.sqrt((prefix.sums[i] - prefix.sums[i - window]) / window) * 10000));
  const anchor = levels[2];
  return [...nativeSecondContextFeatures(c, i), anchor, levels[0] - anchor, levels[1] - anchor, levels[3] - anchor];
}

/** Completed trade flow from the same second is available with its close. */
export function nativeSecondFlowContextFeatures(c: readonly SecondCandle[], i: number): number[] {
  const flow = checkedNativeTradeFlow(c, i);
  return [...nativeSecondContextFeatures(c, i), flow.aggregateCountImbalance, flow.lastAggressorSide];
}

/** Versioned extension of the original flow contract. Existing saved two-flow
 * models retain their exact feature layout and inference path. */
export function nativeSecondFlowVwapContextFeatures(c: readonly SecondCandle[], i: number): number[] {
  const flow = checkedNativeTradeFlow(c, i);
  if (!Number.isFinite(flow.buyerSellerVwapGap) || Math.abs(flow.buyerSellerVwapGap!) > 2)
    throw new Error("Missing or invalid native trade-flow VWAP gap");
  return [...nativeSecondContextFeatures(c, i), flow.aggregateCountImbalance, flow.lastAggressorSide,
    flow.buyerSellerVwapGap!];
}

/** Price and spot-flow additions that repeatedly transferred in the historical
 * 1s sign audits. All inputs end at the completed second i. The two-second
 * lags therefore read i-1, and the short EMA is reconstructed from 64 causal
 * flow bins so its initialization has no observable effect at double precision. */
export function nativeSecondSelectedSignFeatures(c: readonly SecondCandle[], i: number): number[] {
  const context = nativeSecondContextFeatures(c, i);
  const currentReturn = Math.log(c[i].close / c[i - 1].close);
  const priorReturn = Math.log(c[i - 1].close / c[i - 2].close);
  let square = 0;
  for (let j = i - 15; j <= i; j++) {
    if (c[j].openTime !== c[i].openTime - (i - j) * 1000) throw new Error("Misaligned native sign history");
    const value = Math.log(c[j].close / c[j - 1].close);
    square += value * value;
  }
  const haar = square > 0 ? (priorReturn - currentReturn) / (Math.SQRT2 * Math.sqrt(square)) : 0;
  let buyQuoteEma = 0, sellQuoteEma = 0;
  for (let j = i - 63; j <= i; j++) {
    const flow = checkedRichNativeTradeFlow(c, j);
    if (j === i - 63) {
      buyQuoteEma = flow.aggressiveBuyQuoteVolume;
      sellQuoteEma = flow.aggressiveSellQuoteVolume;
    } else {
      buyQuoteEma += 2 / 3 * (flow.aggressiveBuyQuoteVolume - buyQuoteEma);
      sellQuoteEma += 2 / 3 * (flow.aggressiveSellQuoteVolume - sellQuoteEma);
    }
  }
  const flow = checkedRichNativeTradeFlow(c, i), laggedFlow = checkedRichNativeTradeFlow(c, i - 1);
  const imbalance = (positive: number, negative: number) => positive + negative > 0
    ? (positive - negative) / (positive + negative) : 0;
  return [...context, priorReturn * 10000, haar, flow.aggregateCountImbalance, flow.lastAggressorSide,
    laggedFlow.lastAggressorSide,
    imbalance(flow.aggressiveBuyQuoteVolume, flow.aggressiveSellQuoteVolume),
    imbalance(buyQuoteEma, sellQuoteEma),
    imbalance(flow.aggressiveBuyMaxAggregateQuantity, flow.aggressiveSellMaxAggregateQuantity)];
}

function checkedNativeTradeFlow(c: readonly SecondCandle[], i: number): NonNullable<SecondCandle["nativeTradeFlow"]> {
  const row = c[i], flow = row?.nativeTradeFlow, decisionTime = row?.openTime + 1000;
  if (!flow || flow.availableAt !== decisionTime || !Number.isFinite(flow.aggregateCountImbalance)
    || Math.abs(flow.aggregateCountImbalance) > 1 || !Number.isInteger(flow.lastAggressorSide)
    || Math.abs(flow.lastAggressorSide) > 1) throw new Error("Missing, future or invalid native trade flow");
  return flow;
}

function checkedRichNativeTradeFlow(c: readonly SecondCandle[], i: number): Required<NonNullable<SecondCandle["nativeTradeFlow"]>> {
  const flow = checkedNativeTradeFlow(c, i);
  if (![flow.aggressiveBuyQuoteVolume, flow.aggressiveSellQuoteVolume,
    flow.aggressiveBuyMaxAggregateQuantity, flow.aggressiveSellMaxAggregateQuantity]
    .every(value => Number.isFinite(value) && value! >= 0))
    throw new Error("Missing or invalid rich native trade flow");
  return flow as Required<typeof flow>;
}

export function usesNativeSecondTradeFlowFeatures(names: readonly string[]): boolean {
  return [NATIVE_SECOND_FLOW_CONTEXT_FEATURES, NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES,
    NATIVE_SECOND_SELECTED_SIGN_FEATURES]
    .some(expected => names.length === expected.length && names.every((name, index) => name === expected[index]));
}

export function nativeSecondDayContextFeatures(c: readonly SecondCandle[], i: number): number[] {
  if (i < NATIVE_SECOND_DAY_CONTEXT_WARMUP) throw new Error("Native day context requires one day of price history");
  return nativeSecondLagFeatures(c, i, NATIVE_SECOND_DAY_CONTEXT_LAGS);
}

function nativeSecondLagFeatures(c: readonly SecondCandle[], i: number, lags: readonly number[]): number[] {
  const fast = nativeSecondEventFeatures(c, i);
  const current = c[i], returns = lags.map(lag => {
    const previous = c[i - lag];
    if (!previous || previous.openTime !== current.openTime - lag * 1000
      || !(previous.close > 0) || !Number.isFinite(previous.close)) throw new Error("Misaligned native context endpoint");
    return Math.log(current.close / previous.close) * 10000;
  });
  return [...fast, ...returns];
}
