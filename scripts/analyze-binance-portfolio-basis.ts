import fs from "node:fs/promises";
import path from "node:path";
import {
  selectPortfolioBasis,
  type AssetReturnSeries,
  type BasisCorrelationMethod,
  type PortfolioBasisResult,
} from "../apps/server/src/portfolio-basis.js";

const DAY_MS = 24 * 60 * 60 * 1000;
const CACHE_VERSION = 2;
const REPORT_VERSION = 4;
const INTERVAL_MS = {
  "1m": 60 * 1000,
  "3m": 3 * 60 * 1000,
  "5m": 5 * 60 * 1000,
  "15m": 15 * 60 * 1000,
  "30m": 30 * 60 * 1000,
  "1h": 60 * 60 * 1000,
  "2h": 2 * 60 * 60 * 1000,
  "4h": 4 * 60 * 60 * 1000,
  "6h": 6 * 60 * 60 * 1000,
  "8h": 8 * 60 * 60 * 1000,
  "12h": 12 * 60 * 60 * 1000,
  "1d": DAY_MS,
} as const;
const DEFAULT_STABLE_ASSETS = new Set([
  "AEUR",
  "BFUSD",
  "BUSD",
  "DAI",
  "EURI",
  "EUR",
  "FDUSD",
  "PYUSD",
  "RLUSD",
  "TUSD",
  "USD",
  "USD1",
  "USDC",
  "USDE",
  "USDP",
  "USDS",
  "USDT",
  "XUSD",
]);
const LEVERAGED_TOKEN_SUFFIXES = ["BULL", "BEAR", "UP", "DOWN"];

type PriceVenue = "spot" | "usdm-futures" | "coinm-futures";
type BinanceProduct = PriceVenue | "options";
type ProductSelection = "all" | PriceVenue;
type CandleInterval = keyof typeof INTERVAL_MS;
type ExclusionReason =
  | "request-failed"
  | "insufficient-history"
  | "inactive"
  | "zero-variance";

interface CliOptions {
  products: ProductSelection;
  quoteAsset: string;
  lookbackDays: number;
  candleCount: number;
  windowMode: "days" | "candles";
  interval: CandleInterval;
  basisSize?: number;
  minBasisSize: number;
  maxBasisSize: number;
  targetMedianRSquared: number;
  targetP10RSquared: number;
  correlationMethod: BasisCorrelationMethod;
  anchorSymbol?: string;
  residualEquivalenceBand: number;
  minTradedDayRatio: number;
  maxStaleHours: number;
  minMedianQuoteVolume: number;
  includeStableAssets: boolean;
  concurrency: number;
  endDay: number;
  dataDir: string;
  refresh: boolean;
}

interface BinanceSymbol {
  product: BinanceProduct;
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  status?: string;
  contractType?: string;
  isSpotTradingAllowed?: boolean;
  underlying?: string;
  underlyingType?: string;
  underlyingSubType?: string[];
}

interface CanonicalAsset {
  asset: string;
  priceMarket: BinanceSymbol;
  products: BinanceProduct[];
  productSymbols: Partial<Record<BinanceProduct, string[]>>;
  isTradFi: boolean;
}

interface ProductCatalog {
  listings: BinanceSymbol[];
  productCounts: Record<BinanceProduct, number>;
  warnings: string[];
}

interface MarketCandle {
  openTime: number;
  close: number;
  quoteVolume: number;
  trades: number;
}

interface CandleCache {
  version: 2;
  venue: PriceVenue;
  symbol: string;
  interval: CandleInterval;
  updatedAt: number;
  coveredStartTime?: number;
  coveredEndTime?: number;
  candles: MarketCandle[];
}

interface AssetAnalysis extends AssetReturnSeries {
  priceVenue: PriceVenue;
  products: BinanceProduct[];
  productSymbols: Partial<Record<BinanceProduct, string[]>>;
  isTradFi: boolean;
  medianDailyQuoteVolume: number;
  tradedPeriodRatio: number;
  tradedDayRatio: number;
  maxStaleHours: number;
  meanAbsoluteReturn: number;
  annualizedVolatility: number;
}

interface ExcludedAsset {
  symbol: string;
  baseAsset: string;
  reason: ExclusionReason;
  detail: string;
}

interface PortfolioBasisReport {
  version: 4;
  generatedAt: string;
  parameters: {
    products: ProductSelection;
    quoteAsset: string;
    lookbackDays: number;
    windowMode: "days" | "candles";
    interval: CandleInterval;
    sampleCount: number;
    returnStartTime: string;
    returnEndTime: string;
    basisSize?: number;
    minBasisSize: number;
    maxBasisSize: number;
    targetMedianRSquared: number;
    targetP10RSquared: number;
    correlationMethod: BasisCorrelationMethod;
    anchorSymbol?: string;
    residualEquivalenceBand: number;
    minTradedDayRatio: number;
    maxStaleHours: number;
    minMedianQuoteVolume: number;
    stableAssetsExcluded: boolean;
  };
  universe: {
    listedSymbols: number;
    activeListings: number;
    listedEconomicAssets: number;
    activeEconomicAssets: number;
    candidateSymbols: number;
    eligibleSymbols: number;
    excludedSymbols: number;
    exclusionsByReason: Record<ExclusionReason, number>;
    excluded: ExcludedAsset[];
    productListings: Record<BinanceProduct, number>;
    statusListings: Record<string, number>;
    discoveryWarnings: string[];
  };
  basis: PortfolioBasisResult;
  assets: Array<{
    symbol: string;
    baseAsset: string;
    quoteAsset: string;
    isTradFi: boolean;
    medianDailyQuoteVolume: number;
    tradedPeriodRatio: number;
    tradedDayRatio: number;
    maxStaleHours: number;
    meanAbsoluteReturn: number;
    annualizedVolatility: number;
    priceVenue: PriceVenue;
    products: BinanceProduct[];
    productSymbols: Partial<Record<BinanceProduct, string[]>>;
  }>;
  matrixSymbols: string[];
}

const args = process.argv.slice(2);
if (args.includes("--help")) {
  printHelp();
  process.exit(0);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const options = parseOptions(args);
  const intervalMs = INTERVAL_MS[options.interval];
  const sampleCount = options.candleCount;
  const returnEndTime = options.endDay + DAY_MS - intervalMs;
  const returnStartTime = returnEndTime - (sampleCount - 1) * intervalMs;
  const firstCloseTime = returnStartTime - intervalMs;
  const expectedTimes = Array.from(
    { length: sampleCount + 1 },
    (_, index) => firstCloseTime + index * intervalMs,
  );

  console.log(
    `Discovering active ${options.products} product markets on Binance...`,
  );
  const catalog = await listBinanceProducts(options.products);
  const candidates = buildCanonicalAssets(catalog.listings, options);
  if (candidates.length === 0) {
    throw new Error(
      `No continuously priced ${options.quoteAsset}/USD assets found across ${options.products}. ` +
        `Listings: ${JSON.stringify(catalog.productCounts)}. ` +
        `${catalog.warnings.join(" ")}`,
    );
  }
  console.log(
    `Loading ${sampleCount} ${options.interval} returns over ${options.lookbackDays} days ` +
      `for ${candidates.length} candidate assets ` +
      `(${formatTime(returnStartTime, options.interval)} through ` +
      `${formatTime(returnEndTime, options.interval)})...`,
  );

  let completed = 0;
  const analyses = await mapConcurrent(candidates, options.concurrency, async (asset) => {
    try {
      const candles = await loadSymbolCandles(
        asset.priceMarket,
        expectedTimes,
        options,
      );
      return analyzeSymbol(asset, candles, expectedTimes, options);
    } catch (error) {
      return {
        symbol: asset.priceMarket.symbol,
        baseAsset: asset.asset,
        reason: "request-failed" as const,
        detail: error instanceof Error ? error.message : "Unknown request failure",
      };
    } finally {
      completed += 1;
      if (completed % 20 === 0 || completed === candidates.length) {
        console.log(`  ${completed}/${candidates.length} assets loaded`);
      }
    }
  });

  const eligible = analyses.filter(
    (analysis): analysis is AssetAnalysis => "returns" in analysis,
  );
  const excluded = analyses.filter(
    (analysis): analysis is ExcludedAsset => "reason" in analysis,
  );
  if (eligible.length === 0) {
    throw new Error("No assets passed the history and activity requirements.");
  }
  eligible.sort((a, b) => a.symbol.localeCompare(b.symbol));

  const anchorSymbol = resolveAnchorSymbol(eligible, options.anchorSymbol);
  console.log(
    options.basisSize === undefined
      ? `Growing a coverage-driven basis from ${eligible.length} eligible return series` +
          `${anchorSymbol ? ` (anchor ${anchorSymbol})` : ""}...`
      : `Selecting ${Math.min(options.basisSize, eligible.length)} basis assets from ` +
          `${eligible.length} eligible return series${anchorSymbol ? ` (anchor ${anchorSymbol})` : ""}...`,
  );
  const basis = selectPortfolioBasis(eligible, {
    size: options.basisSize,
    minSize: options.minBasisSize,
    maxSize: options.maxBasisSize,
    targetMedianRSquared: options.targetMedianRSquared,
    targetP10RSquared: options.targetP10RSquared,
    correlationMethod: options.correlationMethod,
    anchorSymbol,
    pivotPriorityScores: eligible.map((asset) => asset.meanAbsoluteReturn),
    residualEquivalenceBand: options.residualEquivalenceBand,
  });
  const report = buildReport({
    options: { ...options, anchorSymbol },
    catalog,
    candidates,
    eligible,
    excluded,
    basis,
    returnStartTime,
    returnEndTime,
  });
  const output = await writeReport(report, options.dataDir);

  console.log("");
  console.log(renderConsoleSummary(report));
  console.log("");
  console.log(`JSON: ${output.json}`);
  console.log(`Markdown: ${output.markdown}`);
}

function parseOptions(values: string[]): CliOptions {
  const value = (name: string): string | undefined => {
    const index = values.indexOf(name);
    return index >= 0 ? values[index + 1] : undefined;
  };
  const products = value("--products") ?? value("--venue") ?? "all";
  if (
    products !== "all" &&
    products !== "spot" &&
    products !== "usdm-futures" &&
    products !== "coinm-futures"
  ) {
    throw new Error(
      "--products must be all, spot, usdm-futures, or coinm-futures.",
    );
  }
  const correlationMethod = value("--method") ?? "pearson";
  if (correlationMethod !== "pearson" && correlationMethod !== "spearman") {
    throw new Error("--method must be pearson or spearman.");
  }

  const interval = candleInterval(value("--interval") ?? "1d");
  const requestedCandleCount = value("--candles");
  if (requestedCandleCount !== undefined && value("--days") !== undefined) {
    throw new Error("Use either --candles or --days, not both.");
  }
  const windowMode =
    requestedCandleCount !== undefined ? "candles" : "days";
  const candleCount =
    requestedCandleCount !== undefined
      ? positiveInteger(requestedCandleCount, 0, "--candles")
      : positiveInteger(value("--days"), 365, "--days") *
        (DAY_MS / INTERVAL_MS[interval]);
  const lookbackDays = candleCount * (INTERVAL_MS[interval] / DAY_MS);
  const basisSize = value("--size")
    ? positiveInteger(value("--size"), 0, "--size")
    : undefined;
  const minBasisSize = positiveInteger(value("--min-size"), 8, "--min-size");
  const maxBasisSize = positiveInteger(value("--max-size"), 512, "--max-size");
  if (maxBasisSize < minBasisSize) {
    throw new Error("--max-size must be at least --min-size.");
  }
  const targetMedianRSquared = ratio(
    value("--target-median-r2"),
    0.8,
    "--target-median-r2",
  );
  const targetP10RSquared = ratio(
    value("--target-p10-r2"),
    0.5,
    "--target-p10-r2",
  );
  const concurrency = positiveInteger(value("--concurrency"), 6, "--concurrency");
  const minTradedDayRatio = ratio(
    value("--min-traded-day-ratio"),
    0.5,
    "--min-traded-day-ratio",
  );
  const maxStaleHours = positiveNumber(
    value("--max-stale-hours"),
    48,
    "--max-stale-hours",
  );
  const minMedianQuoteVolume = nonNegativeNumber(
    value("--min-median-quote-volume"),
    0,
    "--min-median-quote-volume",
  );
  const endDay = value("--end")
    ? parseUtcDay(value("--end")!)
    : utcDayStart(Date.now()) - DAY_MS;
  const quoteAsset = (value("--quote") ?? "USDT").toUpperCase();
  const configuredAnchor = value("--anchor");
  const anchorSymbol =
    configuredAnchor === "none"
      ? undefined
      : (configuredAnchor ?? "BTC").toUpperCase();

  return {
    products,
    quoteAsset,
    lookbackDays,
    candleCount,
    windowMode,
    interval,
    basisSize,
    minBasisSize,
    maxBasisSize,
    targetMedianRSquared,
    targetP10RSquared,
    correlationMethod,
    anchorSymbol,
    residualEquivalenceBand: ratio(
      value("--residual-equivalence-band"),
      0.05,
      "--residual-equivalence-band",
    ),
    minTradedDayRatio,
    maxStaleHours,
    minMedianQuoteVolume,
    includeStableAssets: values.includes("--include-stables"),
    concurrency,
    endDay,
    dataDir: path.resolve(value("--data-dir") ?? "data"),
    refresh: values.includes("--refresh"),
  };
}

async function listBinanceProducts(
  selection: ProductSelection,
): Promise<ProductCatalog> {
  const products: BinanceProduct[] =
    selection === "all"
      ? ["spot", "usdm-futures", "coinm-futures", "options"]
      : [selection];
  const results = await Promise.all(
    products.map(async (product) => {
      try {
        const endpoint = exchangeInfoEndpoint(product);
        const payload = await fetchJson<{
          symbols?: unknown[];
          optionSymbols?: unknown[];
          optionContracts?: unknown[];
        }>(endpoint);
        const rows =
          product === "options"
            ? payload.optionSymbols ?? payload.optionContracts
            : payload.symbols;
        const listings = (rows ?? [])
          .map((row) => parseBinanceSymbol(row, product))
          .filter((symbol): symbol is BinanceSymbol => Boolean(symbol));
        return { product, listings };
      } catch (error) {
        return {
          product,
          listings: [] as BinanceSymbol[],
          warning:
            `${product}: ` +
            (error instanceof Error ? error.message : "unknown discovery failure"),
        };
      }
    }),
  );
  const productCounts: Record<BinanceProduct, number> = {
    spot: 0,
    "usdm-futures": 0,
    "coinm-futures": 0,
    options: 0,
  };
  for (const result of results) {
    productCounts[result.product] = result.listings.length;
  }
  return {
    listings: results.flatMap((result) => result.listings),
    productCounts,
    warnings: results
      .map((result) => result.warning)
      .filter((warning): warning is string => Boolean(warning)),
  };
}

function exchangeInfoEndpoint(product: BinanceProduct): string {
  switch (product) {
    case "spot":
      return "https://data-api.binance.vision/api/v3/exchangeInfo?showPermissionSets=false";
    case "usdm-futures":
      return "https://fapi.binance.com/fapi/v1/exchangeInfo";
    case "coinm-futures":
      return "https://dapi.binance.com/dapi/v1/exchangeInfo";
    case "options":
      return "https://eapi.binance.com/eapi/v1/exchangeInfo";
  }
}

function parseBinanceSymbol(
  value: unknown,
  product: BinanceProduct,
): BinanceSymbol | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const row = value as Record<string, unknown>;
  const underlying = stringValue(row.underlying);
  const symbol =
    stringValue(row.symbol) || (product === "options" ? underlying : "");
  const quoteAsset = stringValue(row.quoteAsset) || "USDT";
  const baseAsset =
    stringValue(row.baseAsset) ||
    (product === "options"
      ? optionUnderlyingAsset(underlying || symbol, quoteAsset)
      : "");
  if (!symbol || !baseAsset || !quoteAsset) {
    return undefined;
  }
  return {
    product,
    symbol,
    baseAsset,
    quoteAsset,
    status:
      stringValue(row.status) ||
      stringValue(row.contractStatus) ||
      (product === "options" ? "TRADING" : ""),
    contractType: stringValue(row.contractType),
    isSpotTradingAllowed:
      typeof row.isSpotTradingAllowed === "boolean"
        ? row.isSpotTradingAllowed
        : undefined,
    underlying,
    underlyingType: stringValue(row.underlyingType),
    underlyingSubType: Array.isArray(row.underlyingSubType)
      ? row.underlyingSubType.filter(
          (value): value is string => typeof value === "string",
        )
      : [],
  };
}

function buildCanonicalAssets(
  listed: readonly BinanceSymbol[],
  options: CliOptions,
): CanonicalAsset[] {
  const bStockAliases = new Map(
    listed
      .filter(isTradFiMarket)
      .map(
        (symbol): [string, string] => [
          `${normalizedBaseAsset(symbol.baseAsset)}B`,
          normalizedBaseAsset(symbol.baseAsset),
        ],
      ),
  );
  const byAsset = new Map<string, BinanceSymbol[]>();
  for (const market of listed) {
    const asset = normalizedEconomicAsset(market, options.quoteAsset, bStockAliases);
    if (
      !eligibleProductListing(market, asset, options) ||
      (!options.includeStableAssets && DEFAULT_STABLE_ASSETS.has(asset)) ||
      isLeveragedToken(asset)
    ) {
      continue;
    }
    const markets = byAsset.get(asset) ?? [];
    markets.push(market);
    byAsset.set(asset, markets);
  }

  const assets: CanonicalAsset[] = [];
  for (const [asset, markets] of byAsset) {
    const priceMarkets = markets
      .filter((market): market is BinanceSymbol & { product: PriceVenue } =>
        market.product !== "options",
      )
      .sort(comparePriceMarkets);
    const priceMarket = priceMarkets[0];
    if (!priceMarket) {
      continue;
    }
    const productSymbols: Partial<Record<BinanceProduct, string[]>> = {};
    for (const market of markets) {
      const symbols = productSymbols[market.product] ?? [];
      symbols.push(market.symbol);
      productSymbols[market.product] = symbols;
    }
    assets.push({
      asset,
      priceMarket,
      products: [...new Set(markets.map((market) => market.product))].sort(
        (a, b) => productRank(a) - productRank(b),
      ),
      productSymbols,
      isTradFi: markets.some(isTradFiMarket),
    });
  }
  return assets.sort((a, b) => a.asset.localeCompare(b.asset));
}

function eligibleProductListing(
  market: BinanceSymbol,
  asset: string,
  options: CliOptions,
): boolean {
  if (!asset || market.status !== "TRADING") {
    return false;
  }
  switch (market.product) {
    case "spot":
      return (
        market.quoteAsset === options.quoteAsset &&
        market.isSpotTradingAllowed !== false
      );
    case "usdm-futures":
      return (
        market.quoteAsset === options.quoteAsset &&
        (market.contractType === "PERPETUAL" ||
          market.contractType === "TRADIFI_PERPETUAL")
      );
    case "coinm-futures":
      return (
        options.quoteAsset === "USDT" &&
        market.quoteAsset === "USD" &&
        market.contractType === "PERPETUAL"
      );
    case "options":
      return true;
  }
}

function comparePriceMarkets(left: BinanceSymbol, right: BinanceSymbol): number {
  return (
    productRank(left.product) - productRank(right.product) ||
    Number(left.baseAsset !== normalizedBaseAsset(left.baseAsset)) -
      Number(right.baseAsset !== normalizedBaseAsset(right.baseAsset)) ||
    left.symbol.localeCompare(right.symbol)
  );
}

function productRank(product: BinanceProduct): number {
  return ["spot", "usdm-futures", "coinm-futures", "options"].indexOf(product);
}

function normalizedEconomicAsset(
  market: BinanceSymbol,
  quoteAsset: string,
  bStockAliases: ReadonlyMap<string, string>,
): string {
  const normalized = normalizedBaseAsset(economicAsset(market, quoteAsset));
  if (market.product === "spot") {
    return bStockAliases.get(normalized) ?? normalized;
  }
  return normalized;
}

function isTradFiMarket(market: BinanceSymbol): boolean {
  return (
    market.product === "usdm-futures" &&
    (market.contractType === "TRADIFI_PERPETUAL" ||
      market.underlyingSubType?.includes("TradFi") === true ||
      ["EQUITY", "KR_EQUITY", "PREMARKET", "COMMODITY"].includes(
        market.underlyingType ?? "",
      ))
  );
}

function economicAsset(market: BinanceSymbol, quoteAsset: string): string {
  if (market.product === "options") {
    return optionUnderlyingAsset(
      market.underlying || market.baseAsset,
      market.quoteAsset || quoteAsset,
    );
  }
  return market.baseAsset;
}

function optionUnderlyingAsset(underlying: string, quoteAsset: string): string {
  const normalized = underlying.split("-")[0].toUpperCase();
  const knownQuotes = [quoteAsset, "USDT", "USDC", "USD"];
  const suffix = knownQuotes.find(
    (quote) => normalized.length > quote.length && normalized.endsWith(quote),
  );
  return suffix ? normalized.slice(0, -suffix.length) : normalized;
}

function normalizedBaseAsset(asset: string): string {
  return asset.replace(/^(?:1000|10000|1000000)(?=[A-Z])/, "");
}

function isLeveragedToken(asset: string): boolean {
  return LEVERAGED_TOKEN_SUFFIXES.some(
    (suffix) => asset.length > suffix.length && asset.endsWith(suffix),
  );
}

async function loadSymbolCandles(
  symbol: BinanceSymbol,
  expectedTimes: readonly number[],
  options: CliOptions,
): Promise<MarketCandle[]> {
  const intervalMs = INTERVAL_MS[options.interval];
  const cacheFile = path.join(
    options.dataDir,
    "portfolio-basis",
    "candle-cache",
    options.interval,
    symbol.product,
    `${safePathPart(symbol.symbol)}.json`,
  );
  if (symbol.product === "options") {
    throw new Error("Options do not provide a canonical continuous price series.");
  }
  const cached = options.refresh
    ? undefined
    : await readCandleCache(
        cacheFile,
        symbol.product,
        symbol.symbol,
        options.interval,
      );
  let candles = cached?.candles ?? [];
  const cachedTimes = new Set(candles.map((candle) => candle.openTime));
  const missing = expectedTimes.some((time) => !cachedTimes.has(time));

  if (missing) {
    const requestedStart = expectedTimes[0];
    const requestedEnd = expectedTimes.at(-1)!;
    const ranges: Array<{ start: number; end: number }> = [];
    if (
      cached?.coveredStartTime === undefined ||
      cached.coveredEndTime === undefined
    ) {
      ranges.push({ start: requestedStart, end: requestedEnd });
    } else {
      if (requestedStart < cached.coveredStartTime) {
        ranges.push({
          start: requestedStart,
          end: cached.coveredStartTime - intervalMs,
        });
      }
      if (requestedEnd > cached.coveredEndTime) {
        ranges.push({
          start: cached.coveredEndTime + intervalMs,
          end: requestedEnd,
        });
      }
    }

    for (const range of ranges) {
      const fetched = await fetchCandles(
        symbol.product,
        symbol.symbol,
        options.interval,
        range.start,
        range.end + intervalMs - 1,
      );
      candles = mergeCandles(candles, fetched, intervalMs);
    }
    if (ranges.length > 0) {
      await writeJsonAtomic(cacheFile, {
        version: CACHE_VERSION,
        venue: symbol.product,
        symbol: symbol.symbol,
        interval: options.interval,
        updatedAt: Date.now(),
        coveredStartTime: Math.min(
          requestedStart,
          cached?.coveredStartTime ?? requestedStart,
        ),
        coveredEndTime: Math.max(
          requestedEnd,
          cached?.coveredEndTime ?? requestedEnd,
        ),
        candles,
      } satisfies CandleCache);
    }
  } else if (
    cached &&
    (cached.coveredStartTime === undefined || cached.coveredEndTime === undefined)
  ) {
    await writeJsonAtomic(cacheFile, {
      ...cached,
      updatedAt: Date.now(),
      coveredStartTime: expectedTimes[0],
      coveredEndTime: expectedTimes.at(-1)!,
      candles,
    } satisfies CandleCache);
  }

  const first = expectedTimes[0];
  const last = expectedTimes.at(-1)!;
  return candles.filter(
    (candle) => candle.openTime >= first && candle.openTime <= last,
  );
}

async function readCandleCache(
  file: string,
  venue: PriceVenue,
  symbol: string,
  interval: CandleInterval,
): Promise<CandleCache | undefined> {
  try {
    const parsed = JSON.parse(await fs.readFile(file, "utf8")) as CandleCache;
    if (
      parsed.version !== CACHE_VERSION ||
      parsed.venue !== venue ||
      parsed.symbol !== symbol ||
      parsed.interval !== interval ||
      !Array.isArray(parsed.candles)
    ) {
      return undefined;
    }
    return {
      ...parsed,
      candles: parsed.candles
        .filter((candle) => validCandle(candle, INTERVAL_MS[interval]))
        .sort((a, b) => a.openTime - b.openTime),
    };
  } catch (error) {
    if (isMissingFile(error)) {
      return undefined;
    }
    throw error;
  }
}

async function fetchCandles(
  venue: PriceVenue,
  symbol: string,
  interval: CandleInterval,
  startTime: number,
  endTime: number,
): Promise<MarketCandle[]> {
  const baseUrl =
    venue === "spot"
      ? "https://data-api.binance.vision/api/v3/klines"
      : venue === "usdm-futures"
        ? "https://fapi.binance.com/fapi/v1/klines"
        : "https://dapi.binance.com/dapi/v1/klines";
  const intervalMs = INTERVAL_MS[interval];
  const candles: MarketCandle[] = [];
  let cursor = startTime;

  while (cursor <= endTime) {
    const limit = Math.min(
      1000,
      Math.floor((endTime - cursor) / intervalMs) + 1,
    );
    const url = new URL(baseUrl);
    url.search = new URLSearchParams({
      symbol,
      interval,
      startTime: String(cursor),
      endTime: String(endTime),
      limit: String(limit),
    }).toString();
    const payload = await fetchJson<unknown[]>(url);
    const page = payload
      .map((value) => parseCandle(value, intervalMs))
      .filter((candle): candle is MarketCandle => Boolean(candle))
      .sort((a, b) => a.openTime - b.openTime);
    candles.push(...page);
    const lastOpenTime = page.at(-1)?.openTime;
    if (lastOpenTime === undefined || page.length < limit) {
      break;
    }
    cursor = lastOpenTime + intervalMs;
  }

  return mergeCandles([], candles, intervalMs);
}

function parseCandle(
  value: unknown,
  intervalMs: number,
): MarketCandle | undefined {
  if (!Array.isArray(value) || value.length < 9) {
    return undefined;
  }
  const candle = {
    openTime: milliseconds(value[0]),
    close: Number(value[4]),
    quoteVolume: Number(value[7]),
    trades: Number(value[8]),
  };
  return validCandle(candle, intervalMs) ? candle : undefined;
}

function validCandle(
  value: unknown,
  intervalMs: number,
): value is MarketCandle {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candle = value as MarketCandle;
  return (
    Number.isSafeInteger(candle.openTime) &&
    candle.openTime >= 0 &&
    candle.openTime % intervalMs === 0 &&
    Number.isFinite(candle.close) &&
    candle.close > 0 &&
    Number.isFinite(candle.quoteVolume) &&
    candle.quoteVolume >= 0 &&
    Number.isFinite(candle.trades) &&
    candle.trades >= 0
  );
}

function mergeCandles(
  cached: readonly MarketCandle[],
  fetched: readonly MarketCandle[],
  intervalMs: number,
): MarketCandle[] {
  const byTime = new Map<number, MarketCandle>();
  for (const candle of [...cached, ...fetched]) {
    if (validCandle(candle, intervalMs)) {
      byTime.set(candle.openTime, candle);
    }
  }
  return [...byTime.values()].sort((a, b) => a.openTime - b.openTime);
}

function analyzeSymbol(
  canonical: CanonicalAsset,
  candles: readonly MarketCandle[],
  expectedTimes: readonly number[],
  options: CliOptions,
): AssetAnalysis | ExcludedAsset {
  const symbol = canonical.priceMarket;
  const byTime = new Map(candles.map((candle) => [candle.openTime, candle]));
  const aligned = expectedTimes
    .map((time) => byTime.get(time))
    .filter((candle): candle is MarketCandle => Boolean(candle));
  if (aligned.length !== expectedTimes.length) {
    return {
      symbol: symbol.symbol,
      baseAsset: canonical.asset,
      reason: "insufficient-history",
      detail:
        `${aligned.length}/${expectedTimes.length} required ` +
        `${options.interval} closes`,
    };
  }

  const returns: number[] = [];
  for (let index = 1; index < aligned.length; index += 1) {
    returns.push(Math.log(aligned[index].close / aligned[index - 1].close));
  }
  const measuredCandles = aligned.slice(1);
  const tradedPeriodRatio =
    measuredCandles.filter(isTradedCandle).length / measuredCandles.length;
  const observedDays = new Set(
    measuredCandles.map((candle) => formatDay(candle.openTime)),
  ).size;
  const tradedDayRatio =
    new Set(
      measuredCandles
        .filter(isTradedCandle)
        .map((candle) => formatDay(candle.openTime)),
    ).size / observedDays;
  if (tradedDayRatio < options.minTradedDayRatio) {
    return {
      symbol: symbol.symbol,
      baseAsset: canonical.asset,
      reason: "inactive",
      detail:
        `${formatPercent(tradedDayRatio)} days with trades or quote volume; ` +
        `minimum ${formatPercent(options.minTradedDayRatio)}`,
    };
  }

  const maxStaleHours = maximumStaleHours(
    returns,
    measuredCandles,
    INTERVAL_MS[options.interval],
    canonical.isTradFi,
  );
  if (maxStaleHours > options.maxStaleHours) {
    return {
      symbol: symbol.symbol,
      baseAsset: canonical.asset,
      reason: "inactive",
      detail:
        `${formatCompactNumber(maxStaleHours)} maximum stale hours; ` +
        `maximum ${formatCompactNumber(options.maxStaleHours)}` +
        (canonical.isTradFi ? " during active TradFi candles" : ""),
    };
  }

  const standardDeviation = sampleStandardDeviation(returns);
  if (!Number.isFinite(standardDeviation) || standardDeviation <= 1e-10) {
    return {
      symbol: symbol.symbol,
      baseAsset: canonical.asset,
      reason: "zero-variance",
      detail: `${options.interval} log returns have no usable variance`,
    };
  }

  const medianDailyQuoteVolume = medianDailyVolume(measuredCandles);
  if (medianDailyQuoteVolume < options.minMedianQuoteVolume) {
    return {
      symbol: symbol.symbol,
      baseAsset: canonical.asset,
      reason: "inactive",
      detail:
        `${formatCompactNumber(medianDailyQuoteVolume)} median daily quote volume; ` +
        `minimum ${formatCompactNumber(options.minMedianQuoteVolume)}`,
    };
  }

  return {
    symbol: symbol.symbol,
    baseAsset: canonical.asset,
    quoteAsset: symbol.quoteAsset,
    returns,
    priceVenue: symbol.product as PriceVenue,
    products: canonical.products,
    productSymbols: canonical.productSymbols,
    isTradFi: canonical.isTradFi,
    medianDailyQuoteVolume,
    tradedPeriodRatio,
    tradedDayRatio,
    maxStaleHours,
    meanAbsoluteReturn:
      returns.reduce((total, value) => total + Math.abs(value), 0) /
      returns.length,
    annualizedVolatility:
      standardDeviation * Math.sqrt((365 * DAY_MS) / INTERVAL_MS[options.interval]),
  };
}

function isTradedCandle(candle: MarketCandle): boolean {
  return candle.trades > 0 || candle.quoteVolume > 0;
}

function maximumStaleHours(
  returns: readonly number[],
  candles: readonly MarketCandle[],
  intervalMs: number,
  sessionAware: boolean,
): number {
  let currentPeriods = 0;
  let maximumPeriods = 0;
  for (let index = 0; index < returns.length; index += 1) {
    if (sessionAware && !isTradedCandle(candles[index])) {
      continue;
    }
    if (Math.abs(returns[index]) <= 1e-10) {
      currentPeriods += 1;
      maximumPeriods = Math.max(maximumPeriods, currentPeriods);
    } else {
      currentPeriods = 0;
    }
  }
  return maximumPeriods * (intervalMs / (60 * 60 * 1000));
}

function medianDailyVolume(candles: readonly MarketCandle[]): number {
  const totals = new Map<string, number>();
  for (const candle of candles) {
    const day = formatDay(candle.openTime);
    totals.set(day, (totals.get(day) ?? 0) + candle.quoteVolume);
  }
  return quantile([...totals.values()], 0.5);
}

function resolveAnchorSymbol(
  eligible: readonly AssetAnalysis[],
  requested: string | undefined,
): string | undefined {
  if (!requested) {
    return undefined;
  }
  const normalized = requested.toUpperCase();
  return eligible.find(
    (asset) =>
      asset.symbol.toUpperCase() === normalized ||
      asset.baseAsset.toUpperCase() === normalized,
  )?.symbol;
}

function buildReport(input: {
  options: CliOptions;
  catalog: ProductCatalog;
  candidates: readonly CanonicalAsset[];
  eligible: readonly AssetAnalysis[];
  excluded: readonly ExcludedAsset[];
  basis: PortfolioBasisResult;
  returnStartTime: number;
  returnEndTime: number;
}): PortfolioBasisReport {
  const exclusionsByReason: Record<ExclusionReason, number> = {
    "request-failed": 0,
    "insufficient-history": 0,
    inactive: 0,
    "zero-variance": 0,
  };
  for (const asset of input.excluded) {
    exclusionsByReason[asset.reason] += 1;
  }

  return {
    version: REPORT_VERSION,
    generatedAt: new Date().toISOString(),
    parameters: {
      products: input.options.products,
      quoteAsset: input.options.quoteAsset,
      lookbackDays: input.options.lookbackDays,
      windowMode: input.options.windowMode,
      interval: input.options.interval,
      sampleCount: input.basis.sampleCount,
      returnStartTime: formatTime(
        input.returnStartTime,
        input.options.interval,
      ),
      returnEndTime: formatTime(input.returnEndTime, input.options.interval),
      basisSize: input.options.basisSize,
      minBasisSize: input.options.minBasisSize,
      maxBasisSize: input.options.maxBasisSize,
      targetMedianRSquared: input.options.targetMedianRSquared,
      targetP10RSquared: input.options.targetP10RSquared,
      correlationMethod: input.options.correlationMethod,
      anchorSymbol: input.options.anchorSymbol,
      residualEquivalenceBand: input.options.residualEquivalenceBand,
      minTradedDayRatio: input.options.minTradedDayRatio,
      maxStaleHours: input.options.maxStaleHours,
      minMedianQuoteVolume: input.options.minMedianQuoteVolume,
      stableAssetsExcluded: !input.options.includeStableAssets,
    },
    universe: {
      listedSymbols: input.catalog.listings.length,
      activeListings: input.catalog.listings.filter(
        (listing) => listing.status === "TRADING",
      ).length,
      listedEconomicAssets: new Set(
        input.catalog.listings.map((listing) =>
          normalizedBaseAsset(
            economicAsset(listing, input.options.quoteAsset),
          ),
        ),
      ).size,
      activeEconomicAssets: new Set(
        input.catalog.listings
          .filter((listing) => listing.status === "TRADING")
          .map((listing) =>
            normalizedBaseAsset(
              economicAsset(listing, input.options.quoteAsset),
            ),
          ),
      ).size,
      candidateSymbols: input.candidates.length,
      eligibleSymbols: input.eligible.length,
      excludedSymbols: input.excluded.length,
      exclusionsByReason,
      excluded: [...input.excluded].sort(
        (a, b) => a.reason.localeCompare(b.reason) || a.symbol.localeCompare(b.symbol),
      ),
      productListings: input.catalog.productCounts,
      statusListings: countBy(
        input.catalog.listings,
        (listing) => listing.status || "UNKNOWN",
      ),
      discoveryWarnings: input.catalog.warnings,
    },
    basis: input.basis,
    assets: input.eligible.map((asset) => ({
      symbol: asset.symbol,
      baseAsset: asset.baseAsset,
      quoteAsset: asset.quoteAsset,
      isTradFi: asset.isTradFi,
      medianDailyQuoteVolume: asset.medianDailyQuoteVolume,
      tradedPeriodRatio: asset.tradedPeriodRatio,
      tradedDayRatio: asset.tradedDayRatio,
      maxStaleHours: asset.maxStaleHours,
      meanAbsoluteReturn: asset.meanAbsoluteReturn,
      annualizedVolatility: asset.annualizedVolatility,
      priceVenue: asset.priceVenue,
      products: asset.products,
      productSymbols: asset.productSymbols,
    })),
    matrixSymbols: input.eligible.map((asset) => asset.symbol),
  };
}

async function writeReport(
  report: PortfolioBasisReport,
  dataDir: string,
): Promise<{ json: string; markdown: string }> {
  const root = path.join(dataDir, "portfolio-basis");
  const runStem = [
    report.parameters.returnEndTime.slice(0, 10),
    report.parameters.products,
    report.parameters.quoteAsset.toLowerCase(),
    report.parameters.interval,
    `${report.parameters.sampleCount}c`,
    `q${report.version}`,
    residualBandTag(report.parameters.residualEquivalenceBand),
    report.parameters.correlationMethod,
    `k${report.basis.entries.length}`,
  ].join("-");
  const runDir = path.join(root, "runs");
  const json = path.join(runDir, `${runStem}.json`);
  const markdown = path.join(runDir, `${runStem}.md`);
  const jsonContent = `${JSON.stringify(report, null, 2)}\n`;
  const markdownContent = renderMarkdownReport(report);
  await Promise.all([
    writeTextAtomic(json, jsonContent),
    writeTextAtomic(markdown, markdownContent),
    writeTextAtomic(path.join(root, "latest.json"), jsonContent),
    writeTextAtomic(path.join(root, "latest.md"), markdownContent),
  ]);
  return { json, markdown };
}

function renderConsoleSummary(report: PortfolioBasisReport): string {
  const displayedAssets = report.basis.entries
    .slice(0, 20)
    .map((entry) => entry.baseAsset);
  const remainingAssets = report.basis.entries.length - displayedAssets.length;
  const lines = [
    `Basis size: ${report.basis.entries.length} (${report.basis.sizingMode}; ` +
      `${report.basis.targetReached ? "coverage target reached" : "maximum size reached"})`,
    `Basis: ${displayedAssets.join(", ")}` +
      (remainingAssets > 0 ? `, ... (+${remainingAssets})` : ""),
    `Pairwise mean |r|: ${formatDecimal(report.basis.pairwiseMeanAbsCorrelation)}`,
    `Pairwise max |r|: ${formatDecimal(report.basis.pairwiseMaxAbsCorrelation)}`,
    `Market median R²: ${formatPercent(report.basis.marketMedianRSquared)}`,
    `Market mean R²: ${formatPercent(report.basis.marketMeanRSquared)}`,
  ];
  return lines.join("\n");
}

function renderMarkdownReport(report: PortfolioBasisReport): string {
  const assetBySymbol = new Map(report.assets.map((asset) => [asset.symbol, asset]));
  const lines = [
    "# Binance portfolio basis",
    "",
    `Generated ${report.generatedAt}.`,
    "",
    "## Scope",
    "",
    `- Products: ${report.parameters.products}`,
    `- Product listings discovered: ${Object.entries(report.universe.productListings)
      .map(([product, count]) => `${product} ${count}`)
      .join(", ")}`,
    `- Listing statuses: ${Object.entries(report.universe.statusListings)
      .map(([status, count]) => `${status} ${count}`)
      .join(", ")}`,
    `- Numeraire: ${report.parameters.quoteAsset}`,
    `- Return window: ${report.parameters.returnStartTime} through ${report.parameters.returnEndTime}`,
    `- Sampling: exactly ${report.parameters.sampleCount} ${report.parameters.interval} log returns (${formatCompactNumber(report.parameters.lookbackDays)} days)`,
    `- Correlation: ${report.parameters.correlationMethod}`,
    `- Pivot tie-break: among candidates within ${formatPercent(report.parameters.residualEquivalenceBand)} of the maximum unexplained variance, select the largest mean absolute ${report.parameters.interval} return`,
    report.parameters.basisSize === undefined
      ? `- Sizing: automatic until median R² >= ${formatPercent(report.parameters.targetMedianRSquared)} and 10th-percentile R² >= ${formatPercent(report.parameters.targetP10RSquared)} (maximum ${report.parameters.maxBasisSize})`
      : `- Sizing: fixed at ${report.parameters.basisSize} assets`,
    `- Full catalog: ${report.universe.listedSymbols} listings, ${report.universe.activeListings} active listings, ${report.universe.listedEconomicAssets} deduplicated economic assets`,
    `- Return universe: ${report.universe.eligibleSymbols} eligible assets from ${report.universe.candidateSymbols} active continuously priced candidates`,
    `- Quality filter: complete window, trades or quote volume on at least ${formatPercent(report.parameters.minTradedDayRatio)} of UTC days, no stale-price run longer than ${formatCompactNumber(report.parameters.maxStaleHours)} hours`,
    "- TradFi session handling: zero-volume off-session candles do not extend stale-price runs",
    `- Stable assets excluded: ${report.parameters.stableAssetsExcluded ? "yes" : "no"}`,
    "",
    "## Selected basis",
    "",
    "The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.",
    "",
    "| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |",
    "| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const entry of report.basis.entries) {
    const asset = assetBySymbol.get(entry.symbol);
    lines.push(
      `| ${entry.rank} | ${entry.baseAsset} | ${entry.symbol} (${asset?.priceVenue ?? "-"}) | ` +
        `${asset?.products.join(", ") ?? "-"} | ` +
        `${formatPercent(entry.residualRatio)} | ` +
        `${formatDecimal(entry.maxAbsCorrelationToEarlier)} | ` +
        `${entry.closestEarlierSymbol ?? "-"} | ` +
        `${formatBasisPoints(asset?.meanAbsoluteReturn ?? 0)} | ` +
        `${formatPercent(asset?.tradedDayRatio ?? 0)} | ` +
        `${formatCompactNumber(asset?.maxStaleHours ?? 0)} | ` +
        `${formatCompactNumber(asset?.medianDailyQuoteVolume ?? 0)} | ` +
        `${formatPercent(asset?.annualizedVolatility ?? 0)} |`,
    );
  }

  lines.push(
    "",
    "## Diagnostics",
    "",
    `- Basis size selected: ${report.basis.entries.length}`,
    `- Coverage target reached: ${report.basis.targetReached ? "yes" : "no"}`,
    `- Mean pairwise absolute correlation: ${formatDecimal(report.basis.pairwiseMeanAbsCorrelation)}`,
    `- Maximum pairwise absolute correlation: ${formatDecimal(report.basis.pairwiseMaxAbsCorrelation)}`,
    `- Mean whole-market projection R²: ${formatPercent(report.basis.marketMeanRSquared)}`,
    `- Median whole-market projection R²: ${formatPercent(report.basis.marketMedianRSquared)}`,
    `- 10th-percentile whole-market projection R²: ${formatPercent(report.basis.marketP10RSquared)}`,
    `- Minimum whole-market projection R²: ${formatPercent(report.basis.marketMinRSquared)}`,
    "",
    "### Coverage by basis size",
    "",
    "| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |",
    "| ---: | ---: | ---: | ---: | ---: |",
    ...coverageCheckpoints(report.basis.coverageCurve).map(
      (point) =>
        `| ${point.size} | ${formatPercent(point.meanRSquared)} | ` +
        `${formatPercent(point.medianRSquared)} | ${formatPercent(point.p10RSquared)} | ` +
        `${formatPercent(point.minRSquared)} |`,
    ),
    "",
    report.basis.entries.length > 24
      ? "### Selected-asset correlation matrix (first 24 basis assets)"
      : "### Selected-asset correlation matrix",
    "",
  );
  const displayedBasisEntries = report.basis.entries.slice(0, 24);
  const selectedIndexes = displayedBasisEntries.map((entry) =>
    report.matrixSymbols.indexOf(entry.symbol),
  );
  lines.push(
    `| Asset | ${displayedBasisEntries.map((entry) => entry.baseAsset).join(" | ")} |`,
    `| --- | ${displayedBasisEntries.map(() => "---:").join(" | ")} |`,
  );
  for (let row = 0; row < selectedIndexes.length; row += 1) {
    const values = selectedIndexes.map((column) =>
      formatDecimal(report.basis.correlationMatrix[selectedIndexes[row]][column]),
    );
    lines.push(`| ${displayedBasisEntries[row].baseAsset} | ${values.join(" | ")} |`);
  }
  if (report.basis.entries.length > displayedBasisEntries.length) {
    lines.push(
      "",
      `The complete ${report.basis.entries.length} × ${report.basis.entries.length} matrix is retained in the JSON report.`,
    );
  }

  lines.push(
    "",
    "### Least-covered eligible assets",
    "",
    "A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.",
    "",
    "| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |",
    "| --- | --- | ---: | ---: | --- | ---: |",
  );
  for (const coverage of report.basis.coverage.slice(0, 15)) {
    lines.push(
      `| ${coverage.baseAsset} | ${coverage.symbol} | ` +
        `${formatPercent(coverage.rSquared)} | ${formatPercent(coverage.residualRatio)} | ` +
        `${coverage.closestBasisSymbol} | ${formatDecimal(coverage.closestBasisCorrelation)} |`,
    );
  }

  lines.push(
    "",
    "## Interpretation",
    "",
    "This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.",
    "",
  );
  return `${lines.join("\n")}\n`;
}

function coverageCheckpoints(
  curve: PortfolioBasisResult["coverageCurve"],
): PortfolioBasisResult["coverageCurve"] {
  const lastSize = curve.at(-1)?.size;
  return curve.filter(
    (point) =>
      point.size === 1 ||
      point.size % 5 === 0 ||
      point.size === lastSize,
  );
}

async function fetchJson<T>(url: string | URL): Promise<T> {
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(30_000),
      headers: { accept: "application/json" },
    });
    if (response.ok) {
      return (await response.json()) as T;
    }

    const body = await response.text();
    const retryable =
      response.status === 418 || response.status === 429 || response.status >= 500;
    if (!retryable || attempt === 5) {
      throw new Error(
        `Binance request failed: HTTP ${response.status} ${body.slice(0, 240)}`,
      );
    }
    const retryAfter = Number(response.headers.get("retry-after"));
    await delay(
      Number.isFinite(retryAfter) && retryAfter > 0
        ? retryAfter * 1000
        : 750 * 2 ** (attempt - 1),
    );
  }
  throw new Error("Binance request failed after retries.");
}

async function mapConcurrent<T, R>(
  items: readonly T[],
  concurrency: number,
  map: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from(
    { length: Math.min(concurrency, items.length) },
    async () => {
      while (true) {
        const index = cursor;
        cursor += 1;
        if (index >= items.length) {
          return;
        }
        results[index] = await map(items[index], index);
      }
    },
  );
  await Promise.all(workers);
  return results;
}

async function writeJsonAtomic(file: string, value: unknown): Promise<void> {
  await writeTextAtomic(file, `${JSON.stringify(value)}\n`);
}

async function writeTextAtomic(file: string, content: string): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  await fs.writeFile(temporary, content);
  await fs.rename(temporary, file);
}

function sampleStandardDeviation(values: readonly number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const average = values.reduce((total, value) => total + value, 0) / values.length;
  const variance =
    values.reduce((total, value) => total + (value - average) ** 2, 0) /
    (values.length - 1);
  return Math.sqrt(Math.max(0, variance));
}

function countBy<T>(
  values: readonly T[],
  key: (value: T) => string,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    const name = key(value);
    counts[name] = (counts[name] ?? 0) + 1;
  }
  return counts;
}

function quantile(values: readonly number[], probability: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sorted[lower];
  }
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function positiveInteger(
  raw: string | undefined,
  fallback: number,
  label: string,
): number {
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value;
}

function nonNegativeNumber(
  raw: string | undefined,
  fallback: number,
  label: string,
): number {
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a non-negative number.`);
  }
  return value;
}

function positiveNumber(
  raw: string | undefined,
  fallback: number,
  label: string,
): number {
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive number.`);
  }
  return value;
}

function ratio(raw: string | undefined, fallback: number, label: string): number {
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${label} must be between 0 and 1.`);
  }
  return value;
}

function candleInterval(value: string): CandleInterval {
  if (value in INTERVAL_MS) {
    return value as CandleInterval;
  }
  throw new Error(
    `--interval must be one of ${Object.keys(INTERVAL_MS).join(", ")}.`,
  );
}

function parseUtcDay(value: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error(`Invalid UTC date: ${value}`);
  }
  const time = Date.parse(`${value}T00:00:00.000Z`);
  if (!Number.isFinite(time) || formatDay(time) !== value) {
    throw new Error(`Invalid UTC date: ${value}`);
  }
  return time;
}

function utcDayStart(time: number): number {
  return Math.floor(time / DAY_MS) * DAY_MS;
}

function milliseconds(value: unknown): number {
  let number = Number(value);
  while (number > 100_000_000_000_000) {
    number /= 1000;
  }
  return Math.trunc(number);
}

function safePathPart(value: string): string {
  return Buffer.from(value, "utf8").toString("base64url");
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function formatDay(time: number): string {
  return new Date(time).toISOString().slice(0, 10);
}

function formatTime(time: number, interval: CandleInterval): string {
  const iso = new Date(time).toISOString();
  return interval === "1d" ? iso.slice(0, 10) : `${iso.slice(0, 16)}Z`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatBasisPoints(value: number): string {
  return `${(value * 10_000).toFixed(2)} bp`;
}

function residualBandTag(value: number): string {
  if (value === 0) {
    return "pure";
  }
  return `amp${(value * 100).toFixed(4).replace(/0+$/, "").replace(/\.$/, "").replace(".", "p")}pct`;
}

function formatDecimal(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function isMissingFile(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function printHelp(): void {
  console.log(`Usage: npm run basis:binance -- [options]

Build a near-orthogonal basis of actual Binance asset return vectors.

  --products all|spot|usdm-futures|coinm-futures
                                Product universe (default: all)
  --quote USDT                  Common stable quote / numeraire
  --days 365                    Completed UTC-day lookback window
  --candles N                   Exact return count (mutually exclusive with --days)
  --interval 1d                 Candle interval: 1m, 3m, 5m, 15m, 30m,
                                1h, 2h, 4h, 6h, 8h, 12h, or 1d
  --end YYYY-MM-DD              Last fully included UTC day (default: yesterday)
  --size N                      Fixed basis size; overrides automatic sizing
  --min-size 8                  Minimum automatic basis size
  --max-size 512                Maximum automatic basis size
  --target-median-r2 0.8        Automatic median market-coverage target
  --target-p10-r2 0.5           Automatic lower-decile coverage target
  --method pearson|spearman     Dependence measure (default: pearson)
  --anchor BTC|none             First basis asset (default: BTC)
  --residual-equivalence-band 0.05
                                Prefer largest mean absolute candle return
                                within 5% of the best unexplained variance
  --min-traded-day-ratio 0.5    Required fraction of UTC days with trading
  --max-stale-hours 48          Longest accepted unchanged-price run
  --min-median-quote-volume 0   Optional median daily liquidity floor
  --include-stables             Keep stablecoin base assets
  --concurrency 6               Parallel Binance requests
  --refresh                     Ignore candle caches for the requested window
  --data-dir data               Cache and report root

Reports are written under data/portfolio-basis/.`);
}
