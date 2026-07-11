import type {
  BacktestProgressSnapshot,
  BacktestPreset,
  BotDiagnostics,
  BotEntryRiskReport,
  BotMetricsSnapshot,
  BotSnapshot,
  Candle,
  EquityPoint,
  OrderBookSnapshot,
  PeakValleyBotConfig,
  PeakValleyStrategySnapshot,
  TradingFill,
  TradingOrderSnapshot,
} from "@trading/bot-algo";

export interface EquitySnapshot {
  quoteAvailable: number;
  quoteReserved: number;
  quoteUnleveraged: number;
  assetAvailable: number;
  assetReserved: number;
  assetUnleveraged: number;
}

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

export interface BinanceMarketCatalog {
  markets: BinanceMarketListing[];
  counts: Record<MarketGroup, number>;
  sources: Array<{
    source: MarketVenue;
    status: "ok" | "skipped" | "failed";
    count: number;
    message?: string;
  }>;
  warnings: string[];
  refreshedAt: number;
}

export interface RuntimeSnapshot {
  snapshotSource: string;
  snapshotSeq: number;
  snapshotAt: number;
  market: {
    id: string;
    group: MarketGroup;
    venue: MarketVenue;
    symbol: string;
    displaySymbol: string;
    baseAsset: string;
    quoteAsset: string;
    interval: string;
    maxLeverage?: number;
    connected: boolean;
    statusMessage: string;
    lastEventAt: number;
    lastPrice: number;
    candles: Candle[];
    orderBook?: OrderBookSnapshot;
  };
  bot: {
    status: "running" | "stopped";
    runStartedAt: number;
    config: PeakValleyBotConfig;
    state: BotSnapshot<PeakValleyBotConfig["strategy"], PeakValleyStrategySnapshot>;
    metrics: BotMetricsSnapshot & {
      equity: number;
      netPnl: number;
      returnPct: number;
    };
    diagnostics: BotDiagnostics & { entryRisk: readonly BotEntryRiskReport[] };
    equity: EquitySnapshot;
  };
  recentEvents: Array<{
    type: "open" | "partial-fill" | "fill" | "rejected" | "status" | "reset";
    at: number;
    message: string;
    order?: TradingOrderSnapshot;
    orderId?: string;
    fill?: TradingFill;
  }>;
  backtest: BacktestProgressSnapshot;
  correlations: CorrelationSnapshot;
  equityCurve: EquityPoint[];
  execution: {
    mode: BotExecutionMode;
    exchangeDriven: boolean;
    canUseExchange: boolean;
    live: boolean;
    message: string;
  };
  exchange: BinanceExchangeSnapshot;
}

export type BotExecutionMode = "simulated" | "binance";

export type BacktestSelection = BacktestPreset;
export type { BacktestProgressSnapshot };

export type CorrelationStatus = "idle" | "running" | "ready" | "failed";

export interface CorrelationEntry {
  marketId: string;
  symbol: string;
  displaySymbol: string;
  baseAsset: string;
  quoteAsset: string;
  venue: string;
  correlation?: number;
  samples: number;
  startTime?: number;
  endTime?: number;
  updatedAt?: number;
}

export interface CorrelationSnapshot {
  status: CorrelationStatus;
  focalMarketId?: string;
  focalSymbol?: string;
  focalDisplaySymbol?: string;
  interval?: string;
  lookbackMs?: number;
  marketCount: number;
  expectedPairs: number;
  calculatedPairs: number;
  processedMarkets: number;
  requests: number;
  cacheHitCandles: number;
  cacheMissCandles: number;
  cacheFetchedCandles: number;
  cacheLoaded: boolean;
  truncated: boolean;
  startedAt?: number;
  updatedAt?: number;
  startTime?: number;
  endTime?: number;
  streamConnected: boolean;
  message: string;
  error?: string;
  entries: CorrelationEntry[];
}

export type BinanceExchangeMode =
  | "auto"
  | "live"
  | "spot-live"
  | "usdm-futures-live"
  | "coinm-futures-live"
  | "spot-testnet"
  | "spot-demo"
  | "usdm-futures-testnet"
  | "coinm-futures-testnet";

export interface BinanceExchangeBalance {
  asset: string;
  free: number;
  locked: number;
  walletBalance?: number;
  availableBalance?: number;
  unrealizedPnl?: number;
}

export interface BinanceExchangePosition {
  symbol: string;
  positionSide?: string;
  positionAmt: number;
  entryPrice?: number;
  markPrice?: number;
  unrealizedPnl?: number;
  notional?: number;
  leverage?: number;
  marginType?: string;
  isolatedMargin?: number;
  updateTime?: number;
}

export interface BinanceExchangeOrder {
  symbol: string;
  orderId: string;
  clientOrderId: string;
  localOrderId?: string;
  algo?: boolean;
  side: string;
  type: string;
  status: string;
  price: number;
  stopPrice?: number;
  originalQuantity: number;
  executedQuantity: number;
  cumulativeQuoteQuantity?: number;
  avgPrice?: number;
  timeInForce?: string;
  reduceOnly?: boolean;
  positionSide?: string;
  createdAt?: number;
  updatedAt?: number;
}

export interface BinanceExchangeTrade {
  id: string;
  symbol: string;
  orderId: string;
  clientOrderId?: string;
  localOrderId?: string;
  side: "buy" | "sell";
  price: number;
  quantity: number;
  quoteQuantity: number;
  commission: number;
  commissionAsset: string;
  feeQuote: number;
  realizedPnl?: number;
  positionSide?: string;
  time: number;
  maker?: boolean;
}

export interface BinanceExchangeSymbolFilters {
  symbol: string;
  pricePrecision?: number;
  quantityPrecision?: number;
  tickSize?: number;
  minPrice?: number;
  maxPrice?: number;
  stepSize?: number;
  marketStepSize?: number;
  minQuantity?: number;
  maxQuantity?: number;
  minMarketQuantity?: number;
  maxMarketQuantity?: number;
  minNotional?: number;
  maxNotional?: number;
}

export interface BinanceExchangeCommission {
  makerFeeBps?: number;
  takerFeeBps?: number;
}

export interface BinanceExchangeSnapshot {
  enabled: boolean;
  configured: boolean;
  compatible: boolean;
  mode: BinanceExchangeMode;
  resolvedMode?: Exclude<BinanceExchangeMode, "auto" | "live">;
  live: boolean;
  baseUrl?: string;
  streamEnvironment?: string;
  autoSubmit: boolean;
  connected: boolean;
  userDataStreamConnected?: boolean;
  lastUserDataStreamAt?: number;
  userDataStreamMessage?: string;
  lastSyncAt?: number;
  lastSubmitAt?: number;
  message: string;
  error?: string;
  maxLeverage?: number;
  symbolFilters?: BinanceExchangeSymbolFilters;
  commission?: BinanceExchangeCommission;
  feeBps?: number;
  estimatedSlippageBps?: number;
  balances: BinanceExchangeBalance[];
  positions: BinanceExchangePosition[];
  openOrders: BinanceExchangeOrder[];
  recentOrders: BinanceExchangeOrder[];
  recentTrades: BinanceExchangeTrade[];
  lastOrder?: BinanceExchangeOrder;
}
