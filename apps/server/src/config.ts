import path from "node:path";
import { fileURLToPath } from "node:url";
import { createStrategyConfig } from "@trading/bot-algo";

const sourceDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(sourceDir, "../../..");

const symbol = (process.env.TRADING_SYMBOL ?? "BTCUSDT").toUpperCase();
const interval = process.env.TRADING_INTERVAL ?? "1m";

export const appConfig = {
  host: process.env.HOST ?? "0.0.0.0",
  port: Number(process.env.PORT ?? 3001),
  symbol,
  streamSymbol: symbol.toLowerCase(),
  interval,
  dataDir: process.env.TRADING_DATA_DIR
    ? path.resolve(process.env.TRADING_DATA_DIR)
    : path.join(repoRoot, "data"),
  webDistDir: path.join(repoRoot, "apps/web/dist"),
  historicalCache: {
    maxBytes: parseBytes(process.env.TRADING_HISTORY_CACHE_MAX_BYTES, 512 * 1024 * 1024),
    minFreeBytes: parseBytes(
      process.env.TRADING_HISTORY_CACHE_MIN_FREE_BYTES,
      512 * 1024 * 1024,
    ),
  },
  strategy: createStrategyConfig({
    symbol,
    baseAsset: process.env.TRADING_BASE_ASSET ?? symbol.replace(/USDT$/, "") ?? "BTC",
    quoteAsset: process.env.TRADING_QUOTE_ASSET ?? "USDT",
    startingQuote: Number(process.env.TRADING_STARTING_QUOTE ?? 10_000),
  }),
};

function parseBytes(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const match = /^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb|tb)?$/i.exec(value.trim());
  if (!match) {
    return fallback;
  }

  const amount = Number(match[1]);
  const unit = (match[2] ?? "b").toLowerCase();
  const multipliers: Record<string, number> = {
    b: 1,
    kb: 1024,
    mb: 1024 ** 2,
    gb: 1024 ** 3,
    tb: 1024 ** 4,
  };

  return Math.floor(amount * multipliers[unit]);
}
