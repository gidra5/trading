import path from "node:path";
import { fileURLToPath } from "node:url";
import { createStrategyConfig } from "@trading/bot-algo";
import { createConfiguredMarketListing, type MarketVenue } from "./binance-markets.js";

const sourceDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(sourceDir, "../../..");

const symbol = (process.env.TRADING_SYMBOL ?? "BTCUSDT").toUpperCase();
const interval = process.env.TRADING_INTERVAL ?? "1m";
const market = createConfiguredMarketListing({
  marketId: process.env.TRADING_MARKET_ID,
  venue: parseMarketVenue(process.env.TRADING_MARKET_VENUE),
  symbol,
  baseAsset: process.env.TRADING_BASE_ASSET,
  quoteAsset: process.env.TRADING_QUOTE_ASSET,
});

export const appConfig = {
  host: process.env.HOST ?? "0.0.0.0",
  port: Number(process.env.PORT ?? 3001),
  symbol: market.symbol,
  market,
  interval,
  binanceApiKey: process.env.BINANCE_API_KEY,
  binanceApiSecret: process.env.BINANCE_API_SECRET,
  dataDir: process.env.TRADING_DATA_DIR ? path.resolve(process.env.TRADING_DATA_DIR) : path.join(repoRoot, "data"),
  webDistDir: path.join(repoRoot, "apps/web/dist"),
  historicalCache: {
    maxBytes: parseBytes(process.env.TRADING_HISTORY_CACHE_MAX_BYTES, 1024 * 1024 * 1024),
    minFreeBytes: parseBytes(process.env.TRADING_HISTORY_CACHE_MIN_FREE_BYTES, 1024 * 1024 * 1024),
  },
  correlations: {
    lookbackDays: parseNumber(process.env.TRADING_CORRELATION_LOOKBACK_DAYS, 14),
    maxMarkets: Math.max(2, Math.round(parseNumber(process.env.TRADING_CORRELATION_MAX_MARKETS, 60))),
  },
  strategy: createStrategyConfig({
    symbol: market.symbol,
    baseAsset: market.baseAsset,
    quoteAsset: market.quoteAsset,
    startingQuote: Number(process.env.TRADING_STARTING_QUOTE ?? 10_000),
    maxLeverage: Number(process.env.TRADING_MAX_LEVERAGE ?? 5),
    longBorrowDepth: parseNumber(process.env.TRADING_LONG_BORROW_DEPTH, 7),
    shortBorrowDepth: parseNumber(process.env.TRADING_SHORT_BORROW_DEPTH, 7),
    lockBorrowedLenderCollateral:
      process.env.TRADING_LOCK_BORROWED_LENDER_COLLATERAL !== "false",
    borrowerProfitShareToLender: parseNumber(
      process.env.TRADING_BORROWER_PROFIT_SHARE_TO_LENDER,
      1,
    ),
    shortMarginModel:
      process.env.TRADING_SHORT_MARGIN_MODEL === "futures-margin"
        ? "futures-margin"
        : "spot-borrow",
    maxPositionQuote: Number(
      process.env.TRADING_MAX_POSITION_QUOTE ??
        Number(process.env.TRADING_STARTING_QUOTE ?? 10_000) * 100,
    ),
    maxOpenOrders: parseNumber(process.env.TRADING_MAX_OPEN_ORDERS, 1024),
  }),
};

function parseMarketVenue(value: string | undefined): MarketVenue | undefined {
  if (
    value === "spot" ||
    value === "usdm-futures" ||
    value === "coinm-futures" ||
    value === "options" ||
    value === "predictions"
  ) {
    return value;
  }

  return undefined;
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

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
