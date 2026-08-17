import fs from "node:fs/promises";
import path from "node:path";
import type { Candle } from "@trading/bot-algo";
import {
  readTradeFlowShardReference,
  TradingStorageLayout,
} from "@trading/storage";
import type { StreamVenue } from "./binance-markets.js";

const MINUTE_MS = 60_000;

export interface StoredAggressorVolumeOptions {
  dataDir: string;
  venue: StreamVenue;
  symbol: string;
  /** Missing days before this instant are tolerated as unavailable warmup. */
  requiredFrom?: number;
}

/**
 * Join immutable Binance aggregate-trade flow onto one-minute candles.
 *
 * Buyer aggression means the buyer removed liquidity; seller aggression means
 * the seller removed liquidity. The two fields are kept separate so strategy
 * code never has to infer trade direction from OHLC geometry.
 */
export async function withStoredAggressorVolume(
  candles: readonly Candle[],
  options: StoredAggressorVolumeOptions,
): Promise<Candle[]> {
  if (candles.length === 0) return [];
  if (options.venue !== "spot" || options.symbol.toUpperCase() !== "BTCUSDT") {
    throw new Error(
      "Stored aggressor-volume backtests currently support Binance spot BTCUSDT only.",
    );
  }
  if (candles.some((candle) => candle.openTime % MINUTE_MS !== 0
    || candle.closeTime < candle.openTime
    || candle.closeTime >= candle.openTime + MINUTE_MS)) {
    throw new Error("Stored aggressor-volume backtests currently require one-minute candles.");
  }

  const layout = new TradingStorageLayout(path.resolve(options.dataDir));
  const root = layout.tradeFlowReferences("spot-btcusdt", "btcusdt", "1s");
  const output = candles.map((candle) => ({ ...candle }));
  const byDay = new Map<string, number[]>();
  output.forEach((candle, index) => {
    const day = new Date(candle.openTime).toISOString().slice(0, 10);
    const indexes = byDay.get(day) ?? [];
    indexes.push(index);
    byDay.set(day, indexes);
  });

  for (const [day, indexes] of byDay) {
    const reference = path.join(root, `${day}.json`);
    try {
      await fs.access(reference);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const required = indexes.some(
        (index) => output[index]!.openTime >= (options.requiredFrom ?? Number.NEGATIVE_INFINITY),
      );
      if (required) throw new Error(`Aggressor-volume history is unavailable for ${day}.`);
      continue;
    }
    const seconds = await readTradeFlowShardReference(reference);
    const buyPrefix = new Float64Array(86_401);
    const sellPrefix = new Float64Array(86_401);
    for (let index = 0; index < 86_400; index += 1) {
      const second = seconds[index];
      buyPrefix[index + 1] = buyPrefix[index]! + (second?.aggressiveBuyBaseVolume ?? 0);
      sellPrefix[index + 1] = sellPrefix[index]! + (second?.aggressiveSellBaseVolume ?? 0);
    }
    for (const index of indexes) {
      const candle = output[index]!;
      const dayStart = Date.parse(`${day}T00:00:00.000Z`);
      const startSecond = Math.floor((candle.openTime - dayStart) / 1_000);
      const endSecond = Math.min(
        86_400,
        Math.floor((candle.closeTime - dayStart) / 1_000) + 1,
      );
      candle.aggressiveBuyVolume = buyPrefix[endSecond]! - buyPrefix[startSecond]!;
      candle.aggressiveSellVolume = sellPrefix[endSecond]! - sellPrefix[startSecond]!;
    }
  }
  return output;
}
