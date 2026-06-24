import type {
  BacktestProgressSnapshot,
  BacktestPreset,
  BotEvent,
  Candle,
  OrderBookSnapshot,
  PaperBotState,
  PositionLedger,
} from "@trading/bot-algo";

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
  bot: PaperBotState;
  positions: PositionLedger;
  recentEvents: BotEvent[];
  backtest: BacktestProgressSnapshot;
  correlations: CorrelationSnapshot;
  exchange: BinancePaperSnapshot;
}

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

export type BinancePaperMode =
  | "auto"
  | "spot-testnet"
  | "spot-demo"
  | "usdm-futures-testnet"
  | "coinm-futures-testnet";

export interface BinancePaperBalance {
  asset: string;
  free: number;
  locked: number;
  walletBalance?: number;
  availableBalance?: number;
  unrealizedPnl?: number;
}

export interface BinancePaperPosition {
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

export interface BinancePaperOrder {
  symbol: string;
  orderId: string;
  clientOrderId: string;
  localOrderId?: string;
  side: string;
  type: string;
  status: string;
  price: number;
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

export interface BinancePaperTrade {
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

export interface BinancePaperSymbolFilters {
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

export interface BinancePaperCommission {
  makerFeeBps?: number;
  takerFeeBps?: number;
}

export interface BinancePaperSnapshot {
  enabled: boolean;
  configured: boolean;
  compatible: boolean;
  mode: BinancePaperMode;
  resolvedMode?: Exclude<BinancePaperMode, "auto">;
  baseUrl?: string;
  streamEnvironment?: string;
  autoSubmit: boolean;
  connected: boolean;
  lastSyncAt?: number;
  lastSubmitAt?: number;
  message: string;
  error?: string;
  maxLeverage?: number;
  symbolFilters?: BinancePaperSymbolFilters;
  commission?: BinancePaperCommission;
  feeBps?: number;
  estimatedSlippageBps?: number;
  balances: BinancePaperBalance[];
  positions: BinancePaperPosition[];
  openOrders: BinancePaperOrder[];
  recentOrders: BinancePaperOrder[];
  recentTrades: BinancePaperTrade[];
  lastOrder?: BinancePaperOrder;
}
