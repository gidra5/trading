import { createHmac } from "node:crypto";

export type MarketGroup =
  | "spot"
  | "bstocks"
  | "futures"
  | "tradfi"
  | "options"
  | "predictions";

export type MarketVenue =
  | "spot"
  | "usdm-futures"
  | "coinm-futures"
  | "options"
  | "predictions";

export type StreamVenue = Exclude<MarketVenue, "predictions">;

export interface BinanceMarketListing {
  id: string;
  group: MarketGroup;
  venue: MarketVenue;
  symbol: string;
  displaySymbol: string;
  baseAsset: string;
  quoteAsset: string;
  status: string;
  searchable: string;
  supportsLiveStream: boolean;
  supportsHistoricalCandles: boolean;
  quoteVolume24h?: number;
  volume24h?: number;
  priceChangePercent24h?: number;
  tradeCount24h?: number;
  maxLeverage?: number;
  unavailableReason?: string;
  pair?: string;
  contractType?: string;
  marginAsset?: string;
  underlying?: string;
  underlyingType?: string;
  underlyingSubType?: string[];
  expiryTime?: number;
  strikePrice?: number;
  optionSide?: "CALL" | "PUT";
  predictionMarketTopicId?: number;
  predictionMarketId?: number;
  predictionTokenId?: string;
}

export interface BinanceMarketSourceStatus {
  source: MarketVenue;
  status: "ok" | "skipped" | "failed";
  count: number;
  message?: string;
}

export interface BinanceMarketCatalogSnapshot {
  markets: BinanceMarketListing[];
  counts: Record<MarketGroup, number>;
  sources: BinanceMarketSourceStatus[];
  warnings: string[];
  refreshedAt: number;
}

interface BinanceMarketCatalogOptions {
  apiKey?: string;
  apiSecret?: string;
  ttlMs?: number;
}

interface SpotSymbol {
  symbol?: unknown;
  status?: unknown;
  baseAsset?: unknown;
  quoteAsset?: unknown;
  isSpotTradingAllowed?: unknown;
  isMarginTradingAllowed?: unknown;
  permissionSets?: unknown;
}

interface FuturesSymbol {
  symbol?: unknown;
  pair?: unknown;
  contractType?: unknown;
  status?: unknown;
  contractStatus?: unknown;
  baseAsset?: unknown;
  quoteAsset?: unknown;
  marginAsset?: unknown;
  underlyingType?: unknown;
  underlyingSubType?: unknown;
}

interface FuturesLeverageBracket {
  symbol?: unknown;
  pair?: unknown;
  brackets?: Array<{
    initialLeverage?: unknown;
  }>;
}

interface OptionSymbol {
  symbol?: unknown;
  status?: unknown;
  underlying?: unknown;
  quoteAsset?: unknown;
  side?: unknown;
  strikePrice?: unknown;
  expiryDate?: unknown;
}

interface PredictionTopic {
  marketTopicId?: unknown;
  title?: unknown;
  question?: unknown;
  symbol?: unknown;
  collateral?: unknown;
  status?: unknown;
  vendor?: unknown;
  markets?: PredictionOutcomeMarket[];
}

interface PredictionOutcomeMarket {
  marketId?: unknown;
  title?: unknown;
  tradingStatus?: unknown;
  status?: unknown;
  outcomes?: Array<{
    name?: unknown;
    tokenId?: unknown;
    price?: unknown;
    chance?: unknown;
  }>;
}

interface Ticker24h {
  symbol?: unknown;
  quoteVolume?: unknown;
  volume?: unknown;
  priceChangePercent?: unknown;
  count?: unknown;
}

type MarketTickerMetrics = Pick<
  BinanceMarketListing,
  "quoteVolume24h" | "volume24h" | "priceChangePercent24h" | "tradeCount24h"
>;

const DEFAULT_TTL_MS = 5 * 60 * 1000;
const KNOWN_BSTOCK_BASE_ASSETS = new Set([
  "CRCLB",
  "MUB",
  "NVDAB",
  "SNDKB",
  "SPCXB",
  "TSLAB",
]);
const COMMON_QUOTES = [
  "USDT",
  "USDC",
  "FDUSD",
  "TUSD",
  "BUSD",
  "BTC",
  "ETH",
  "BNB",
  "TRY",
  "EUR",
  "BRL",
  "USD",
];

export class BinanceMarketCatalog {
  private cache?: {
    expiresAt: number;
    snapshot: BinanceMarketCatalogSnapshot;
  };

  constructor(private readonly options: BinanceMarketCatalogOptions = {}) {}

  async list(forceRefresh = false): Promise<BinanceMarketCatalogSnapshot> {
    const now = Date.now();
    if (!forceRefresh && this.cache && this.cache.expiresAt > now) {
      return this.cache.snapshot;
    }

    const usdmResult = await loadMarketSource("usdm-futures", () =>
      fetchUsdmFuturesMarkets(this.options.apiKey, this.options.apiSecret),
    );
    const bStockUnderlyingBases = new Set(
      usdmResult.markets
        .filter((market) => market.group === "tradfi")
        .map((market) => market.baseAsset),
    );
    const results = await Promise.all([
      loadMarketSource("spot", () => fetchSpotMarkets(bStockUnderlyingBases)),
      Promise.resolve(usdmResult),
      loadMarketSource("coinm-futures", () =>
        fetchCoinmFuturesMarkets(this.options.apiKey, this.options.apiSecret),
      ),
      loadMarketSource("options", () => fetchOptionsMarkets()),
      loadMarketSource("predictions", () => fetchPredictionMarkets(this.options.apiKey)),
    ]);
    const markets: BinanceMarketListing[] = [];
    const sources: BinanceMarketSourceStatus[] = [];
    const warnings: string[] = [];

    for (const result of results) {
      markets.push(...result.markets);
      sources.push(result.source);
      if (result.warning) {
        warnings.push(result.warning);
      }
    }

    const listedMarkets = dedupeMarkets(markets).sort(compareMarkets);
    const snapshot: BinanceMarketCatalogSnapshot = {
      markets: listedMarkets,
      counts: countGroups(listedMarkets),
      sources,
      warnings,
      refreshedAt: now,
    };

    this.cache = {
      expiresAt: now + (this.options.ttlMs ?? DEFAULT_TTL_MS),
      snapshot,
    };
    return snapshot;
  }

  async find(marketId: string): Promise<BinanceMarketListing | undefined> {
    const catalog = await this.list();
    return catalog.markets.find((market) => market.id === marketId);
  }
}

async function loadMarketSource(
  source: MarketVenue,
  load: () => Promise<{
    source: BinanceMarketSourceStatus;
    markets: BinanceMarketListing[];
    warning?: string;
  }>,
): Promise<{
  source: BinanceMarketSourceStatus;
  markets: BinanceMarketListing[];
  warning?: string;
}> {
  try {
    return await load();
  } catch (error) {
    const message =
      error instanceof Error ? error.message : `Unknown ${source} market fetch error`;
    return {
      source: {
        source,
        status: "failed",
        count: 0,
        message,
      },
      markets: [],
      warning: message,
    };
  }
}

export function createConfiguredMarketListing(options: {
  marketId?: string;
  venue?: MarketVenue;
  symbol: string;
  baseAsset?: string;
  quoteAsset?: string;
}): BinanceMarketListing {
  const parsed = parseMarketId(options.marketId);
  const venue = parsed?.venue ?? options.venue ?? "spot";
  const symbol = (parsed?.symbol ?? options.symbol).toUpperCase();
  const quoteAsset = options.quoteAsset ?? inferQuoteAsset(symbol);
  const baseAsset = options.baseAsset ?? inferBaseAsset(symbol, quoteAsset);
  const group =
    venue === "predictions"
      ? "predictions"
      : venue === "options"
        ? "options"
        : venue === "spot"
          ? "spot"
          : "futures";

  return buildListing({
    id: marketIdFor(venue, symbol),
    group,
    venue,
    symbol,
    displaySymbol: symbol,
    baseAsset,
    quoteAsset,
    status: "TRADING",
    supportsLiveStream: venue !== "predictions",
    supportsHistoricalCandles: venue !== "predictions",
  });
}

export function getMarketStorageKey(market: Pick<BinanceMarketListing, "id">): string {
  return market.id.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
}

export function isStreamVenue(venue: MarketVenue): venue is StreamVenue {
  return venue !== "predictions";
}

function parseMarketId(marketId: string | undefined): { venue: MarketVenue; symbol: string } | undefined {
  if (!marketId) {
    return undefined;
  }

  const separator = marketId.indexOf(":");
  if (separator <= 0) {
    return undefined;
  }

  const venue = marketId.slice(0, separator) as MarketVenue;
  const symbol = marketId.slice(separator + 1);
  if (!isMarketVenue(venue) || !symbol) {
    return undefined;
  }

  return { venue, symbol };
}

function isMarketVenue(value: string): value is MarketVenue {
  return (
    value === "spot" ||
    value === "usdm-futures" ||
    value === "coinm-futures" ||
    value === "options" ||
    value === "predictions"
  );
}

async function fetchSpotMarkets(bStockUnderlyingBases: Set<string>): Promise<{
  source: BinanceMarketSourceStatus;
  markets: BinanceMarketListing[];
}> {
  const [payload, tickerMetrics] = await Promise.all([
    fetchJson<{ symbols?: SpotSymbol[] }>("https://api.binance.com/api/v3/exchangeInfo"),
    fetchTickerMetrics("spot"),
  ]);
  const markets = (payload.symbols ?? [])
    .map((symbol) => mapSpotSymbol(symbol, bStockUnderlyingBases, tickerMetrics))
    .filter((market): market is BinanceMarketListing => Boolean(market));

  return {
    source: { source: "spot", status: "ok", count: markets.length },
    markets,
  };
}

async function fetchUsdmFuturesMarkets(
  apiKey: string | undefined,
  apiSecret: string | undefined,
): Promise<{
  source: BinanceMarketSourceStatus;
  markets: BinanceMarketListing[];
  warning?: string;
}> {
  const [payload, leverageResult, tickerMetrics] = await Promise.all([
    fetchJson<{ symbols?: FuturesSymbol[] }>("https://fapi.binance.com/fapi/v1/exchangeInfo"),
    fetchFuturesMaxLeverage("usdm-futures", apiKey, apiSecret),
    fetchTickerMetrics("usdm-futures"),
  ]);
  const markets = (payload.symbols ?? [])
    .map((symbol) =>
      mapFuturesSymbol(
        symbol,
        "usdm-futures",
        leverageResult.maxLeverageBySymbol,
        tickerMetrics,
      ),
    )
    .filter((market): market is BinanceMarketListing => Boolean(market));

  return {
    source: { source: "usdm-futures", status: "ok", count: markets.length },
    markets,
    warning: leverageResult.warning,
  };
}

async function fetchCoinmFuturesMarkets(
  apiKey: string | undefined,
  apiSecret: string | undefined,
): Promise<{
  source: BinanceMarketSourceStatus;
  markets: BinanceMarketListing[];
  warning?: string;
}> {
  const [payload, leverageResult, tickerMetrics] = await Promise.all([
    fetchJson<{ symbols?: FuturesSymbol[] }>("https://dapi.binance.com/dapi/v1/exchangeInfo"),
    fetchFuturesMaxLeverage("coinm-futures", apiKey, apiSecret),
    fetchTickerMetrics("coinm-futures"),
  ]);
  const markets = (payload.symbols ?? [])
    .map((symbol) =>
      mapFuturesSymbol(
        symbol,
        "coinm-futures",
        leverageResult.maxLeverageBySymbol,
        tickerMetrics,
      ),
    )
    .filter((market): market is BinanceMarketListing => Boolean(market));

  return {
    source: { source: "coinm-futures", status: "ok", count: markets.length },
    markets,
    warning: leverageResult.warning,
  };
}

async function fetchOptionsMarkets(): Promise<{
  source: BinanceMarketSourceStatus;
  markets: BinanceMarketListing[];
}> {
  const payload = await fetchJson<{ optionSymbols?: OptionSymbol[] }>(
    "https://eapi.binance.com/eapi/v1/exchangeInfo",
  );
  const markets = (payload.optionSymbols ?? [])
    .map(mapOptionSymbol)
    .filter((market): market is BinanceMarketListing => Boolean(market));

  return {
    source: { source: "options", status: "ok", count: markets.length },
    markets,
  };
}

async function fetchPredictionMarkets(apiKey: string | undefined): Promise<{
  source: BinanceMarketSourceStatus;
  markets: BinanceMarketListing[];
  warning?: string;
}> {
  if (!apiKey) {
    const warning =
      "Prediction markets require BINANCE_API_KEY; skipping /sapi prediction market list.";
    return {
      source: {
        source: "predictions",
        status: "skipped",
        count: 0,
        message: warning,
      },
      markets: [],
      warning,
    };
  }

  const markets: BinanceMarketListing[] = [];
  let offset = 0;
  const limit = 100;
  let hasMore = true;

  while (hasMore) {
    const url = new URL("https://api.binance.com/sapi/v1/w3w/wallet/prediction/market/list");
    url.search = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
    }).toString();
    const payload = await fetchJson<{
      marketTopics?: PredictionTopic[];
      hasMore?: boolean;
    }>(url, {
      headers: {
        "X-MBX-APIKEY": apiKey,
      },
    });
    const pageMarkets = (payload.marketTopics ?? []).flatMap(mapPredictionTopic);
    markets.push(...pageMarkets);
    hasMore = Boolean(payload.hasMore) && pageMarkets.length > 0;
    offset += limit;

    if (offset >= 5_000) {
      break;
    }
  }

  return {
    source: { source: "predictions", status: "ok", count: markets.length },
    markets,
  };
}

function mapSpotSymbol(
  symbol: SpotSymbol,
  bStockUnderlyingBases: Set<string>,
  tickerMetrics: Map<string, MarketTickerMetrics>,
): BinanceMarketListing | undefined {
  const marketSymbol = stringValue(symbol.symbol);
  if (!marketSymbol) {
    return undefined;
  }

  const status = stringValue(symbol.status) || "UNKNOWN";
  const baseAsset = stringValue(symbol.baseAsset) || inferBaseAsset(marketSymbol);
  const quoteAsset = stringValue(symbol.quoteAsset) || inferQuoteAsset(marketSymbol);
  const isTrading = status === "TRADING" && symbol.isSpotTradingAllowed !== false;
  const bStockUnderlying = bStockUnderlyingFor(baseAsset, quoteAsset, bStockUnderlyingBases);
  const isBStock = Boolean(bStockUnderlying);
  return buildListing({
    id: marketIdFor("spot", marketSymbol),
    group: isBStock ? "bstocks" : "spot",
    venue: "spot",
    symbol: marketSymbol,
    displaySymbol: isBStock
      ? `${bStockUnderlying} bStock/${quoteAsset}`
      : `${baseAsset}/${quoteAsset}`,
    baseAsset,
    quoteAsset,
    status,
    supportsLiveStream: isTrading,
    supportsHistoricalCandles: isTrading,
    unavailableReason: isTrading ? undefined : `Spot status is ${status}`,
    ...tickerMetrics.get(marketSymbol),
    underlying: bStockUnderlying,
    underlyingType: isBStock ? "EQUITY" : undefined,
    underlyingSubType: isBStock ? ["bStock"] : undefined,
  });
}

function mapFuturesSymbol(
  symbol: FuturesSymbol,
  venue: "usdm-futures" | "coinm-futures",
  maxLeverageBySymbol: Map<string, number>,
  tickerMetrics: Map<string, MarketTickerMetrics>,
): BinanceMarketListing | undefined {
  const marketSymbol = stringValue(symbol.symbol);
  if (!marketSymbol) {
    return undefined;
  }

  const status = stringValue(symbol.status) || stringValue(symbol.contractStatus) || "UNKNOWN";
  const baseAsset = stringValue(symbol.baseAsset) || inferBaseAsset(marketSymbol);
  const quoteAsset = stringValue(symbol.quoteAsset) || inferQuoteAsset(marketSymbol);
  const contractType = stringValue(symbol.contractType);
  const underlyingType = stringValue(symbol.underlyingType);
  const underlyingSubType = stringArray(symbol.underlyingSubType);
  const isTradFi =
    contractType === "TRADIFI_PERPETUAL" ||
    underlyingSubType.includes("TradFi") ||
    ["EQUITY", "KR_EQUITY", "PREMARKET", "COMMODITY"].includes(underlyingType);
  const isTrading = status === "TRADING";
  const displaySuffix =
    contractType && contractType !== "PERPETUAL" ? ` ${contractType.replace("_", " ")}` : " PERP";

  return buildListing({
    id: marketIdFor(venue, marketSymbol),
    group: isTradFi ? "tradfi" : "futures",
    venue,
    symbol: marketSymbol,
    displaySymbol: `${baseAsset}/${quoteAsset}${displaySuffix}`,
    baseAsset,
    quoteAsset,
    status,
    supportsLiveStream: isTrading,
    supportsHistoricalCandles: isTrading,
    ...tickerMetrics.get(marketSymbol),
    maxLeverage: maxLeverageBySymbol.get(marketSymbol),
    unavailableReason: isTrading ? undefined : `Futures status is ${status}`,
    pair: stringValue(symbol.pair),
    contractType,
    marginAsset: stringValue(symbol.marginAsset),
    underlyingType,
    underlyingSubType,
  });
}

function mapOptionSymbol(symbol: OptionSymbol): BinanceMarketListing | undefined {
  const marketSymbol = stringValue(symbol.symbol);
  if (!marketSymbol) {
    return undefined;
  }

  const status = stringValue(symbol.status) || "UNKNOWN";
  const quoteAsset = stringValue(symbol.quoteAsset) || "USDT";
  const underlying = stringValue(symbol.underlying) || inferBaseAsset(marketSymbol, quoteAsset);
  const side = stringValue(symbol.side);
  const strikePrice = numberValue(symbol.strikePrice);
  const expiryTime = numberValue(symbol.expiryDate);
  const isTrading = status === "TRADING";

  return buildListing({
    id: marketIdFor("options", marketSymbol),
    group: "options",
    venue: "options",
    symbol: marketSymbol,
    displaySymbol: formatOptionDisplay(marketSymbol, underlying, strikePrice, side, expiryTime),
    baseAsset: underlying,
    quoteAsset,
    status,
    supportsLiveStream: isTrading,
    supportsHistoricalCandles: isTrading,
    unavailableReason: isTrading ? undefined : `Option status is ${status}`,
    underlying,
    expiryTime,
    strikePrice,
    optionSide: side === "CALL" || side === "PUT" ? side : undefined,
  });
}

function mapPredictionTopic(topic: PredictionTopic): BinanceMarketListing[] {
  const topicId = numberValue(topic.marketTopicId);
  if (!topicId) {
    return [];
  }

  const title = stringValue(topic.title) || stringValue(topic.question) || `Prediction ${topicId}`;
  const sourceSymbol = stringValue(topic.symbol);
  const quoteAsset = stringValue(topic.collateral) || "USDT";
  const topicStatus = stringValue(topic.status) || "UNKNOWN";
  const markets = topic.markets ?? [];

  if (markets.length === 0) {
    return [
      buildListing({
        id: marketIdFor("predictions", String(topicId)),
        group: "predictions",
        venue: "predictions",
        symbol: String(topicId),
        displaySymbol: title,
        baseAsset: sourceSymbol || "PREDICTION",
        quoteAsset,
        status: topicStatus,
        supportsLiveStream: false,
        supportsHistoricalCandles: false,
        unavailableReason: "Prediction topics use token order books, not Binance candle streams.",
        predictionMarketTopicId: topicId,
      }),
    ];
  }

  return markets.flatMap((market) => {
    const marketId = numberValue(market.marketId);
    const marketTitle = stringValue(market.title);
    const tradingStatus = stringValue(market.tradingStatus) || stringValue(market.status) || topicStatus;
    return (market.outcomes ?? []).map((outcome) => {
      const tokenId = stringValue(outcome.tokenId);
      const outcomeName = stringValue(outcome.name) || "Outcome";
      const symbol = [topicId, marketId, tokenId].filter(Boolean).join("-");
      return buildListing({
        id: marketIdFor("predictions", symbol),
        group: "predictions",
        venue: "predictions",
        symbol,
        displaySymbol: `${title} / ${marketTitle || outcomeName}`,
        baseAsset: outcomeName,
        quoteAsset,
        status: tradingStatus,
        supportsLiveStream: false,
        supportsHistoricalCandles: false,
        unavailableReason: "Prediction outcome books need the prediction market adapter.",
        underlying: sourceSymbol,
        predictionMarketTopicId: topicId,
        predictionMarketId: marketId,
        predictionTokenId: tokenId,
      });
    });
  });
}

function buildListing(
  market: Omit<BinanceMarketListing, "searchable">,
): BinanceMarketListing {
  const searchable = [
    market.symbol,
    market.displaySymbol,
    market.baseAsset,
    market.quoteAsset,
    market.group,
    market.venue,
    market.contractType,
    market.maxLeverage ? `${market.maxLeverage}x` : undefined,
    market.underlying,
    market.underlyingType,
    ...(market.underlyingSubType ?? []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return {
    ...market,
    searchable,
  };
}

function marketIdFor(venue: MarketVenue, symbol: string): string {
  return `${venue}:${symbol.toUpperCase()}`;
}

function dedupeMarkets(markets: BinanceMarketListing[]): BinanceMarketListing[] {
  return [...new Map(markets.map((market) => [market.id, market])).values()];
}

function countGroups(markets: BinanceMarketListing[]): Record<MarketGroup, number> {
  return {
    spot: 0,
    bstocks: 0,
    futures: 0,
    tradfi: 0,
    options: 0,
    predictions: 0,
    ...markets.reduce<Partial<Record<MarketGroup, number>>>((counts, market) => {
      counts[market.group] = (counts[market.group] ?? 0) + 1;
      return counts;
    }, {}),
  };
}

function compareMarkets(a: BinanceMarketListing, b: BinanceMarketListing): number {
  return (
    groupRank(a.group) - groupRank(b.group) ||
    a.quoteAsset.localeCompare(b.quoteAsset) ||
    a.baseAsset.localeCompare(b.baseAsset) ||
    a.symbol.localeCompare(b.symbol)
  );
}

function groupRank(group: MarketGroup): number {
  return ["spot", "bstocks", "futures", "tradfi", "options", "predictions"].indexOf(group);
}

async function fetchTickerMetrics(
  venue: Extract<MarketVenue, "spot" | "usdm-futures" | "coinm-futures">,
): Promise<Map<string, MarketTickerMetrics>> {
  try {
    const payload = await fetchJson<Ticker24h[]>(ticker24hEndpointForVenue(venue));
    const metricsBySymbol = new Map<string, MarketTickerMetrics>();

    for (const row of payload) {
      const symbol = stringValue(row.symbol);
      if (!symbol) {
        continue;
      }

      const metrics: MarketTickerMetrics = {
        quoteVolume24h: numberValue(row.quoteVolume),
        volume24h: numberValue(row.volume),
        priceChangePercent24h: numberValue(row.priceChangePercent),
        tradeCount24h: numberValue(row.count),
      };
      metricsBySymbol.set(symbol, metrics);
    }

    return metricsBySymbol;
  } catch {
    return new Map();
  }
}

function ticker24hEndpointForVenue(
  venue: Extract<MarketVenue, "spot" | "usdm-futures" | "coinm-futures">,
): string {
  if (venue === "spot") {
    return "https://api.binance.com/api/v3/ticker/24hr";
  }
  if (venue === "coinm-futures") {
    return "https://dapi.binance.com/dapi/v1/ticker/24hr";
  }
  return "https://fapi.binance.com/fapi/v1/ticker/24hr";
}

async function fetchFuturesMaxLeverage(
  venue: "usdm-futures" | "coinm-futures",
  apiKey: string | undefined,
  apiSecret: string | undefined,
): Promise<{
  maxLeverageBySymbol: Map<string, number>;
  warning?: string;
}> {
  if (!apiKey || !apiSecret) {
    return {
      maxLeverageBySymbol: new Map(),
      warning: `${venueLabel(venue)} max leverage requires BINANCE_API_KEY and BINANCE_API_SECRET.`,
    };
  }

  try {
    const endpoint =
      venue === "usdm-futures"
        ? "https://fapi.binance.com/fapi/v1/leverageBracket"
        : "https://dapi.binance.com/dapi/v2/leverageBracket";
    const payload = await fetchSignedJson<unknown>(endpoint, apiKey, apiSecret);
    return {
      maxLeverageBySymbol: parseMaxLeverageBySymbol(payload),
    };
  } catch (error) {
    const message =
      error instanceof Error ? error.message : `${venueLabel(venue)} max leverage request failed.`;
    return {
      maxLeverageBySymbol: new Map(),
      warning: message,
    };
  }
}

async function fetchSignedJson<T>(
  endpoint: string,
  apiKey: string,
  apiSecret: string,
): Promise<T> {
  const url = new URL(endpoint);
  const query = new URLSearchParams({
    recvWindow: "5000",
    timestamp: String(Date.now()),
  });
  const signature = createHmac("sha256", apiSecret).update(query.toString()).digest("hex");
  query.set("signature", signature);
  url.search = query.toString();

  return fetchJson<T>(url, {
    headers: {
      "X-MBX-APIKEY": apiKey,
    },
  });
}

function parseMaxLeverageBySymbol(payload: unknown): Map<string, number> {
  const rows = Array.isArray(payload) ? payload : [payload];
  const maxLeverageBySymbol = new Map<string, number>();

  for (const row of rows) {
    if (!row || typeof row !== "object") {
      continue;
    }

    const bracketRow = row as FuturesLeverageBracket;
    const symbol = stringValue(bracketRow.symbol) || stringValue(bracketRow.pair);
    const maxLeverage = Math.max(
      0,
      ...(bracketRow.brackets ?? []).map(
        (bracket) => numberValue(bracket.initialLeverage) ?? 0,
      ),
    );
    if (symbol && maxLeverage > 0) {
      maxLeverageBySymbol.set(symbol, maxLeverage);
    }
  }

  return maxLeverageBySymbol;
}

function venueLabel(venue: "usdm-futures" | "coinm-futures"): string {
  return venue === "usdm-futures" ? "USD-M futures" : "COIN-M futures";
}

async function fetchJson<T>(url: string | URL, init?: RequestInit): Promise<T> {
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    const response = await fetch(url, {
      ...init,
      signal: AbortSignal.timeout(20_000),
    });
    if (response.ok) {
      return (await response.json()) as T;
    }

    const body = await response.text();
    const retryable = response.status === 418 || response.status === 429 || response.status >= 500;
    if (retryable && attempt < 4) {
      await delay(750 * attempt);
      continue;
    }

    throw new Error(
      `Binance market request failed: HTTP ${response.status} ${body.slice(0, 240)}`,
    );
  }

  throw new Error("Binance market request failed.");
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numberValue(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function inferQuoteAsset(symbol: string): string {
  const upperSymbol = symbol.toUpperCase();
  const quote = COMMON_QUOTES.find((asset) => upperSymbol.endsWith(asset));
  return quote ?? "USDT";
}

function inferBaseAsset(symbol: string, quoteAsset = inferQuoteAsset(symbol)): string {
  const upperSymbol = symbol.toUpperCase();
  if (upperSymbol.endsWith("_PERP")) {
    return upperSymbol.replace(/USD_PERP$/, "").replace(/USDT_PERP$/, "");
  }

  return upperSymbol.endsWith(quoteAsset)
    ? upperSymbol.slice(0, -quoteAsset.length)
    : upperSymbol;
}

function bStockUnderlyingFor(
  baseAsset: string,
  quoteAsset: string,
  bStockUnderlyingBases: Set<string>,
): string | undefined {
  if (quoteAsset !== "USDT" || !baseAsset.endsWith("B") || baseAsset.length < 3) {
    return undefined;
  }

  const underlying = baseAsset.slice(0, -1);
  if (KNOWN_BSTOCK_BASE_ASSETS.has(baseAsset) || bStockUnderlyingBases.has(underlying)) {
    return underlying;
  }

  return undefined;
}

function formatOptionDisplay(
  symbol: string,
  underlying: string,
  strikePrice: number | undefined,
  side: string,
  expiryTime: number | undefined,
): string {
  const expiry = expiryTime
    ? new Date(expiryTime).toISOString().slice(2, 10).replace(/-/g, "")
    : "";
  const optionSide = side === "CALL" ? "C" : side === "PUT" ? "P" : side;
  const strike = strikePrice ? strikePrice.toLocaleString("en-US", { maximumFractionDigits: 8 }) : "";
  const details = [expiry, strike, optionSide].filter(Boolean).join(" ");
  return details ? `${underlying} ${details}` : symbol;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
