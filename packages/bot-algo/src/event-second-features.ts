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
export const NATIVE_SECOND_DAY_CONTEXT_LAGS = [...NATIVE_SECOND_CONTEXT_LAGS, 43200, 86400] as const;
export const NATIVE_SECOND_DAY_CONTEXT_FEATURES = [...NATIVE_SECOND_EVENT_FEATURES,
  ...NATIVE_SECOND_DAY_CONTEXT_LAGS.map(seconds => `return-${seconds}s-bps`)] as const;
export const NATIVE_SECOND_DAY_CONTEXT_WARMUP = 86400;
type SecondClose = { openTime: number; close: number };
type SecondCandle = SecondClose & { open: number; high: number; low: number; volume: number };

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
