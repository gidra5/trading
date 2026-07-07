import path from "node:path";
import { fileURLToPath } from "node:url";
import { createStrategyConfig, type PartialStrategyConfig } from "@trading/bot-algo";
import { createConfiguredMarketListing, type MarketVenue } from "./binance-markets.js";
import type { BinancePaperMode } from "./binance-paper.js";

const sourceDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(sourceDir, "../../..");

const environment = normalizeEnvironment(process.env.TRADING_ENV);
const symbol = (process.env.TRADING_SYMBOL ?? "BTCUSDT").toUpperCase();
const interval = process.env.TRADING_INTERVAL ?? "1m";
const binanceExchangeMode = parseBinancePaperMode(
  process.env.TRADING_BINANCE_EXCHANGE_MODE ??
    (parseBoolean(process.env.TRADING_BINANCE_LIVE_ENABLED, false)
      ? "live"
      : process.env.TRADING_BINANCE_PAPER_MODE),
);
const market = createConfiguredMarketListing({
  marketId: process.env.TRADING_MARKET_ID,
  venue: parseMarketVenue(process.env.TRADING_MARKET_VENUE),
  symbol,
  baseAsset: process.env.TRADING_BASE_ASSET,
  quoteAsset: process.env.TRADING_QUOTE_ASSET,
});
const legacyValleyPeakEnv = parseLegacyValleyPeakEnv();

export const appConfig = {
  environment: environment ?? "local",
  host: process.env.HOST ?? "0.0.0.0",
  port: Number(process.env.PORT ?? 3001),
  symbol: market.symbol,
  market,
  interval,
  binanceApiKey: process.env.BINANCE_API_KEY,
  binanceApiSecret: process.env.BINANCE_API_SECRET,
  binancePaper: {
    enabled:
      parseBoolean(process.env.TRADING_BINANCE_PAPER_ENABLED, false) ||
      parseBoolean(process.env.TRADING_BINANCE_LIVE_ENABLED, false) ||
      isLiveBinanceMode(binanceExchangeMode),
    mode: binanceExchangeMode,
    apiKey: process.env.BINANCE_PAPER_API_KEY,
    apiSecret: process.env.BINANCE_PAPER_API_SECRET,
    liveApiKey: process.env.BINANCE_API_KEY,
    liveApiSecret: process.env.BINANCE_API_SECRET,
    recvWindowMs: Math.max(1_000, parseNumber(process.env.BINANCE_PAPER_RECV_WINDOW_MS, 5_000)),
    autoSubmit: parseBoolean(process.env.TRADING_BINANCE_PAPER_AUTO_SUBMIT, false),
    baseUrlOverride: process.env.BINANCE_PAPER_BASE_URL,
  },
  dataDir: resolveDataDir(environment),
  webDistDir: path.join(repoRoot, "apps/web/dist"),
  historicalCache: {
    maxBytes: parseBytes(process.env.TRADING_HISTORY_CACHE_MAX_BYTES, 1024 * 1024 * 1024),
    minFreeBytes: parseBytes(process.env.TRADING_HISTORY_CACHE_MIN_FREE_BYTES, 1024 * 1024 * 1024),
  },
  correlations: {
    lookbackDays: parseNumber(process.env.TRADING_CORRELATION_LOOKBACK_DAYS, 14),
    maxMarkets: Math.max(2, Math.round(parseNumber(process.env.TRADING_CORRELATION_MAX_MARKETS, 60))),
  },
  exchangeAccountGuard: {
    hardStop: parseBoolean(process.env.TRADING_EXCHANGE_ACCOUNT_GUARD_HARD_STOP, false),
  },
  strategy: createStrategyConfig({
    symbol: market.symbol,
    baseAsset: market.baseAsset,
    quoteAsset: market.quoteAsset,
    startingQuote: Number(process.env.TRADING_STARTING_QUOTE ?? 10_000),
    maxLeverage: Number(process.env.TRADING_MAX_LEVERAGE ?? 5),
    longBorrowDepth: parseNumber(process.env.TRADING_LONG_BORROW_DEPTH, 7),
    shortBorrowDepth: parseNumber(process.env.TRADING_SHORT_BORROW_DEPTH, 7),
    borrowerProfitShareToLender: parseNumber(
      process.env.TRADING_BORROWER_PROFIT_SHARE_TO_LENDER,
      1,
    ),
    shortMarginModel:
      process.env.TRADING_SHORT_MARGIN_MODEL === "futures-margin"
        ? "futures-margin"
        : "spot-borrow",
    ...(process.env.TRADING_MAX_POSITION_QUOTE
      ? { maxPositionQuote: Number(process.env.TRADING_MAX_POSITION_QUOTE) }
      : {}),
    maxOpenOrders: parseNumber(process.env.TRADING_MAX_OPEN_ORDERS, 1024),
    ...(legacyValleyPeakEnv ? { legacyValleyPeak: legacyValleyPeakEnv } : {}),
  }),
};

function resolveDataDir(environmentName: string | undefined): string {
  if (process.env.TRADING_DATA_DIR) {
    return path.resolve(process.env.TRADING_DATA_DIR);
  }

  if (environmentName) {
    return path.join(repoRoot, "data", environmentName);
  }

  return path.join(repoRoot, "data");
}

function normalizeEnvironment(value: string | undefined): string | undefined {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (normalized === "production") {
    return "prod";
  }
  if (normalized === "staging") {
    return "stage";
  }

  const safe = normalized.replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return safe || undefined;
}

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

function parseBinancePaperMode(value: string | undefined): BinancePaperMode {
  if (
    value === "live" ||
    value === "spot-live" ||
    value === "usdm-futures-live" ||
    value === "coinm-futures-live" ||
    value === "spot-testnet" ||
    value === "spot-demo" ||
    value === "usdm-futures-testnet" ||
    value === "coinm-futures-testnet"
  ) {
    return value;
  }

  return "auto";
}

function isLiveBinanceMode(value: BinancePaperMode): boolean {
  return value === "live" || value.endsWith("-live");
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (!value) {
    return fallback;
  }
  if (["1", "true", "yes", "on"].includes(value.toLowerCase())) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(value.toLowerCase())) {
    return false;
  }
  return fallback;
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseOptionalNumber(value: string | undefined): number | undefined {
  if (!value) {
    return undefined;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseLegacyValleyPeakEnv():
  | NonNullable<PartialStrategyConfig["legacyValleyPeak"]>
  | undefined {
  const overrides: NonNullable<PartialStrategyConfig["legacyValleyPeak"]> = {};

  const maxMisses = parseOptionalNumber(
    process.env.TRADING_ANTICIPATORY_CONFIRMATION_MAX_MISSES,
  );
  if (maxMisses !== undefined) {
    overrides.anticipatoryConfirmationMaxMisses = maxMisses;
  }

  const windowSec = parseOptionalNumber(
    process.env.TRADING_ANTICIPATORY_CONFIRMATION_WINDOW_SEC,
  );
  if (windowSec !== undefined) {
    overrides.anticipatoryConfirmationWindowSec = windowSec;
  }

  const lookaheadFraction = parseOptionalNumber(
    process.env.TRADING_ANTICIPATORY_CONFIRMATION_LOOKAHEAD_FRACTION,
  );
  if (lookaheadFraction !== undefined) {
    overrides.anticipatoryConfirmationLookaheadFraction = lookaheadFraction;
  }

  const gridWindowSec = parseOptionalNumber(
    process.env.TRADING_ANTICIPATORY_GRID_WINDOW_SEC,
  );
  if (gridWindowSec !== undefined) {
    overrides.anticipatoryGridWindowSec = gridWindowSec;
  }

  const gridOrderCount = parseOptionalNumber(
    process.env.TRADING_ANTICIPATORY_GRID_ORDER_COUNT,
  );
  if (gridOrderCount !== undefined) {
    overrides.anticipatoryGridOrderCount = gridOrderCount;
  }

  return Object.keys(overrides).length > 0 ? overrides : undefined;
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
