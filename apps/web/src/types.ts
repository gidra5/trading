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
  market: {
    id: string;
    group: MarketGroup;
    venue: MarketVenue;
    symbol: string;
    displaySymbol: string;
    baseAsset: string;
    quoteAsset: string;
    interval: string;
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
}

export type BacktestSelection = BacktestPreset;
export type { BacktestProgressSnapshot };
