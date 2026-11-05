import type { Kline, KlineInterval } from "binance";
import type {
  TradingCandle,
  TradingHistoryRequest,
  TradingTick,
} from "@trading/bot-algo/trading-api";
import { validateHistoryRequest } from "./common.js";

const intervals = new Map<number, KlineInterval>([
  [1000, "1s"],
  [60_000, "1m"],
  [180_000, "3m"],
  [300_000, "5m"],
  [900_000, "15m"],
  [1_800_000, "30m"],
  [3_600_000, "1h"],
  [7_200_000, "2h"],
  [14_400_000, "4h"],
  [21_600_000, "6h"],
  [28_800_000, "8h"],
  [43_200_000, "12h"],
  [86_400_000, "1d"],
  [259_200_000, "3d"],
  [604_800_000, "1w"],
]);

interface AggregateTrade {
  a: number;
  p: string | number;
  q: string | number;
  T: number;
  m: boolean;
}
interface TickTransport {
  getAggregateTrades(input: {
    symbol: string;
    startTime?: number;
    endTime?: number;
    fromId?: number;
    limit?: number;
  }): Promise<AggregateTrade[]>;
}
interface HistoryTransport {
  getKlines(input: {
    symbol: string;
    interval: KlineInterval;
    endTime: number;
    limit: number;
  }): Promise<Kline[]>;
}

async function aggregateTrades(
  client: TickTransport,
  symbol: string,
  input: { startTime: number; endTime: number },
): Promise<AggregateTrade[]> {
  if (
    !Number.isSafeInteger(input.startTime) ||
    !Number.isSafeInteger(input.endTime) ||
    input.startTime < 0 ||
    input.endTime < input.startTime ||
    input.endTime - input.startTime >= 3_600_000
  ) {
    throw new Error(
      "Tick history requires an inclusive range shorter than one hour",
    );
  }
  const result: AggregateTrade[] = [];
  let fromId: number | undefined;
  for (let pageNumber = 0; pageNumber < 1000; pageNumber++) {
    const page = await client.getAggregateTrades({
      symbol,
      limit: 1000,
      ...(fromId === undefined ? input : { fromId }),
    });
    if (!page.length) return result;
    for (const row of page) {
      if (row.T > input.endTime) return result;
      if (row.T >= input.startTime) result.push(row);
    }
    const next = page.at(-1)!.a + 1;
    if (!Number.isSafeInteger(next) || (fromId !== undefined && next <= fromId))
      throw new Error("Binance tick pagination did not advance");
    fromId = next;
    if (page.length < 1000) return result;
  }
  throw new Error(
    "Tick history exceeds one million aggregate trades; request a shorter range",
  );
}

export async function getTicks(
  client: TickTransport,
  symbol: string,
  input: { startTime: number; endTime: number },
): Promise<readonly TradingTick[]> {
  return (await aggregateTrades(client, symbol, input)).map((row) => ({
    timestamp: row.T,
    price: Number(row.p),
    quantity: Number(row.q),
    candle: null,
  }));
}

export async function getHistory(
  client: HistoryTransport & Partial<TickTransport>,
  symbol: string,
  input: TradingHistoryRequest,
  product: "spot" | "futures" | "options",
): Promise<readonly TradingCandle[]> {
  validateHistoryRequest(input);
  if (!input.count) return [];
  if (product === "futures" && input.intervalMs === 1000) {
    if (!client.getAggregateTrades || input.count > 3600)
      throw new Error("Futures 1s history is limited to one hour per request");
    const endTime = Math.floor(Date.now() / 1000) * 1000 - 1;
    const rows = await aggregateTrades(client as TickTransport, symbol, {
      startTime: Math.max(0, endTime + 1 - input.count * 1000),
      endTime,
    });
    // Private builders are mutable only while aggregating a response.
    const buckets = new Map<
      number,
      { -readonly [K in keyof TradingCandle]: TradingCandle[K] }
    >();
    for (const row of rows) {
      const openTime = Math.floor(row.T / 1000) * 1000,
        price = Number(row.p),
        volume = Number(row.q);
      const candle = buckets.get(openTime) ?? {
        openTime,
        closeTime: openTime + 999,
        open: price,
        high: price,
        low: price,
        close: price,
        volume: 0,
        aggressiveBuyVolume: 0,
        aggressiveSellVolume: 0,
      };
      candle.high = Math.max(candle.high, price);
      candle.low = Math.min(candle.low, price);
      candle.close = price;
      candle.volume += volume;
      if (row.m) candle.aggressiveSellVolume! += volume;
      else candle.aggressiveBuyVolume! += volume;
      buckets.set(openTime, candle);
    }
    // No invented candles for seconds without trades.
    return [...buckets.values()].sort((a, b) => a.openTime - b.openTime);
  }
  const interval = intervals.get(input.intervalMs);
  if (!interval || (product === "options" && interval === "1s"))
    throw new Error(`Unsupported ${product} candle interval`);
  const now = Date.now(),
    candles = new Map<number, TradingCandle>();
  let endTime = now - 1;
  while (candles.size < input.count && endTime >= 0) {
    const page = await client.getKlines({
      symbol,
      interval,
      endTime,
      limit: Math.min(
        product === "spot" ? 1000 : 1500,
        input.count - candles.size + 1,
      ),
    });
    if (!page.length) break;
    for (const row of page) {
      if (row[6] >= now) continue;
      candles.set(row[0], {
        openTime: row[0],
        closeTime: row[6],
        open: Number(row[1]),
        high: Number(row[2]),
        low: Number(row[3]),
        close: Number(row[4]),
        volume: Number(row[5]),
        aggressiveBuyVolume: Number(row[9]),
        aggressiveSellVolume: Number(row[5]) - Number(row[9]),
      });
    }
    const next = Math.min(...page.map((row) => row[0])) - 1;
    if (next >= endTime)
      throw new Error("Binance history pagination did not advance");
    endTime = next;
  }
  return [...candles.values()]
    .sort((a, b) => a.openTime - b.openTime)
    .slice(-input.count);
}
