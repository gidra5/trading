import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { readDerivativesKlinesShardReferenceSync, type SequentialDerivativesKlineRow } from "../packages/storage/src/derivatives-klines.js";
import type { EventCandle } from "../packages/bot-algo/src/event-distribution.js";

export const EVENT_FUTURES_PRICE_INPUTS = ["futures-spot-basis-bps", "futures-relative-return-1m-bps", "futures-relative-return-5m-bps"] as const;
export const EVENT_FUTURES_FLOW_INPUTS = ["futures-log1p-trade-count-1m", "futures-taker-quote-imbalance-1m", "futures-taker-quote-imbalance-5m"] as const;
export const EVENT_FUTURES_DEVIATION_INPUTS = ["futures-basis-minus-ema5-bps", "futures-basis-minus-ema15-bps", "futures-basis-minus-ema60-bps"] as const;
export const EVENT_CANDLE_SHAPE_INPUTS = ["spot-close-location-1m", "futures-close-location-1m"] as const;

/** Close location in the latest completed spot/futures minute. A flat, valid
 * candle is neutral; absent or malformed source data is never a zero value. */
export function eventCandleShapes(candles: readonly EventCandle[], index: number,
  rowAt: (openTime: number) => SequentialDerivativesKlineRow | undefined): number[] | null {
  const candle = candles[index];
  if (!candle || !Number.isSafeInteger(candle.openTime)) return null;
  const row = rowAt(candle.openTime);
  if (!row || row.openTime !== candle.openTime) return null;
  const values: number[] = [];
  for (const r of [candle, row]) {
    const { open, high, low, close } = r;
    if (open === null || high === null || low === null || close === null
      || ![open, high, low, close].every(v => Number.isFinite(v) && v > 0)
      || high < low || open < low || open > high || close < low || close > high) return null;
    values.push(high === low ? 0 : 2 * ((close - low) / (high - low)) - 1);
  }
  return values;
}

/** Shared fitted-value layout for numerical coordinates and their names.
 * Centered removes absolute basis while retaining relative returns and deviations. */
export function eventFittedFuturesInputs<T>(basis: string, values: { price: readonly T[]; flow: readonly T[] }, deviations: readonly T[] = []): T[] {
  if (!["spot", "price", "all", "deviation", "centered"].includes(basis) || values.price.length !== 3 || values.flow.length !== 3
    || (["deviation", "centered"].includes(basis) && deviations.length !== 3)) throw new Error("Invalid fitted futures layout");
  if (basis === "spot") return [];
  if (basis === "centered") return [...values.price.slice(1), ...deviations];
  return [...values.price, ...(basis === "all" ? values.flow : []), ...(basis === "deviation" ? deviations : [])];
}

/** Each EMA starts at the oldest of exactly 240 completed paired minutes.
 * Fixed causal warmup makes the feature independent of the loader's start date. */
export function eventFuturesBasisDeviations(candles: readonly EventCandle[], index: number,
  rowAt: (openTime: number) => SequentialDerivativesKlineRow | undefined): number[] | null {
  if (index < 239 || !candles[index]) return null;
  const means = [0, 0, 0], rates = [5, 15, 60].map(span => 2 / (span + 1));
  let basis = 0;
  for (let i = index - 239; i <= index; i++) {
    const expected = candles[index].openTime - (index - i) * 60000, candle = candles[i], row = rowAt(expected);
    if (!candle || candle.openTime !== expected || !row || row.openTime !== expected || !(candle.close > 0) || !Number.isFinite(candle.close)
      || row.close === null || !(row.close > 0) || !Number.isFinite(row.close)) return null;
    basis = Math.log(row.close / candle.close) * 10000;
    for (let j = 0; j < means.length; j++) means[j] = i === index - 239 ? basis : means[j] + rates[j] * (basis - means[j]);
  }
  return means.map(mean => basis - mean);
}

export function loadEventFuturesRows(start: number, end: number) {
  const root = path.resolve(__dirname, "..", "data/market/immutable/refs"), DAY = 86_400_000;
  const rows = new Map<number, SequentialDerivativesKlineRow>(), references: string[] = [], missing: string[] = [], hash = createHash("sha256");
  for (let day = Math.floor(start / DAY) * DAY; day < end; day += DAY) {
    const date = new Date(day).toISOString().slice(0, 10), suffix = `derivatives-klines/usdm-futures/btcusdt/1m/${date}.json`;
    const files = [path.join(root, "research", suffix), path.join(root, suffix)], file = files.find(f => fs.existsSync(f));
    if (!file) { missing.push(date); continue; }
    references.push(file); hash.update(file).update(fs.readFileSync(file));
    for (const row of readDerivativesKlinesShardReferenceSync(file)) rows.set(row.openTime, row);
  }
  return { rows, references, missing, fingerprint: hash.digest("hex") };
}

/** Completed, contiguous minute observations only. Missing source data is not
 * a zero flow observation and cannot be forward-filled through a gap. */
export function eventFuturesFeatures(candles: readonly EventCandle[], index: number,
  rowAt: (openTime: number) => SequentialDerivativesKlineRow | undefined): { price: number[]; flow: number[] } | null {
  if (index < 5 || !candles[index]) return null;
  const rows: SequentialDerivativesKlineRow[] = [];
  for (let lag = 0; lag <= 5; lag++) {
    const expected = candles[index].openTime - lag * 60_000, candle = candles[index - lag], row = rowAt(expected);
    if (!candle || candle.openTime !== expected || !row || row.openTime !== expected
      || row.close === null || !(row.close > 0) || !Number.isFinite(row.close)
      || row.tradeCount === null || !Number.isSafeInteger(row.tradeCount) || row.tradeCount < 0
      || row.quoteVolume === null || !Number.isFinite(row.quoteVolume) || row.quoteVolume < 0
      || row.takerBuyQuoteVolume === null || !Number.isFinite(row.takerBuyQuoteVolume)
      || row.takerBuyQuoteVolume < 0 || row.takerBuyQuoteVolume > row.quoteVolume) return null;
    rows.push(row);
  }
  const current = rows[0], price = [Math.log(current.close! / candles[index].close) * 1e4,
    (Math.log(current.close! / rows[1].close!) - Math.log(candles[index].close / candles[index - 1].close)) * 1e4,
    (Math.log(current.close! / rows[5].close!) - Math.log(candles[index].close / candles[index - 5].close)) * 1e4];
  const imbalance = (selected: typeof rows) => {
    const volume = selected.reduce((s, r) => s + r.quoteVolume!, 0), buy = selected.reduce((s, r) => s + r.takerBuyQuoteVolume!, 0);
    return volume ? 2 * buy / volume - 1 : 0;
  };
  return { price, flow: [Math.log1p(current.tradeCount!), imbalance(rows.slice(0, 1)), imbalance(rows.slice(0, 5))] };
}
