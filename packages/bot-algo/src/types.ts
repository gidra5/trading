export type BotStatus = "running" | "stopped";

export type OrderSide = "buy" | "sell";

export type OrderStatus = "open" | "filled" | "cancelled";

export type OrderType = "limit";

export type BotSignal = "buy" | "sell" | "hold";

export type StrategyAlgorithm = "moving-average" | "legacy-valley-peak";

export type BacktestPreset =
  | "saved-candles"
  | "saved-orderbook"
  | "week"
  | "month"
  | "year";

export type BacktestRunStatus = "idle" | "running" | "completed" | "failed";

export type BacktestStopReason = "completed" | "wiped_out" | "error";

export interface PriceTick {
  symbol: string;
  eventTime: number;
  price: number;
  quantity?: number;
}

export interface TickProcessingOptions {
  collectEvents?: boolean;
  updateMetrics?: boolean;
}

export interface Candle {
  symbol: string;
  interval: string;
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closed: boolean;
}

export interface OrderBookLevel {
  price: number;
  quantity: number;
}

export interface OrderBookSnapshot {
  symbol: string;
  eventTime: number;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
}

export interface StrategyConfig {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  algorithm: StrategyAlgorithm;
  startingQuote: number;
  feeBps: number;
  orderQuoteSize: number;
  maxPositionQuote: number;
  fastWindow: number;
  slowWindow: number;
  signalThresholdBps: number;
  limitOffsetBps: number;
  maxOpenOrders: number;
  cooldownMs: number;
  staleOrderMs: number;
  takeProfitBps: number;
  stopLossBps: number;
  minOrderQuote: number;
  legacyValleyPeak: LegacyValleyPeakConfig;
  positionRisk: PositionRiskConfig;
}

export interface PositionRiskConfig {
  lowerPriceExpectation: number;
  lowerBaselinePrice: number;
  upperPriceExpectation: number;
  upperBaselinePrice: number;
  maxLossPct: number;
  marketSlippageBps: number;
  quantityFloor: number;
}

export interface LegacyValleyPeakConfig {
  averagingRangesSec: number[];
  rateRatios: number[];
  rateThresholdsLow: number[];
  rateThresholdsHigh: number[];
  buyDataIndex: number;
  sellDataIndex: number;
  buyConfirmationOffset: number;
  sellConfirmationOffsets: number[];
  saturationSec: number;
  buySpendRate: number;
  sellAmountRate: number;
  buySigma: number;
  sellSigma: number;
  minTradeQuote: number;
  maxTradeQuote: number;
}

export interface RollingAveragePoint {
  avg: number;
  rate: number;
  rateClamped: number;
}

export interface RollingAverageMemory {
  entries: number[];
  averages: number[];
  timestamps: number[];
  sum: number;
  startIndex?: number;
  previousSampleIndex?: number;
  points: RollingAveragePoint[];
}

export interface LegacyValleyPeakMemory {
  startedAt?: number;
  buyAverages: RollingAverageMemory[];
  sellAverages: RollingAverageMemory[];
}

export interface TradingOrder {
  id: string;
  side: OrderSide;
  type: OrderType;
  status: OrderStatus;
  price: number;
  quantity: number;
  filledQuantity: number;
  estimatedQuoteCost: number;
  createdAt: number;
  updatedAt: number;
  filledAt?: number;
  cancelledAt?: number;
  reason: string;
  realizedPnl: number;
  feeQuote: number;
  targetPositionId?: string;
  manual?: boolean;
}

export interface TradeFill {
  id: string;
  orderId: string;
  side: OrderSide;
  price: number;
  quantity: number;
  quoteQuantity: number;
  feeQuote: number;
  realizedPnl: number;
  filledAt: number;
  reason: string;
  targetPositionId?: string;
  manual?: boolean;
}

export interface ManualTradeInput {
  side: OrderSide;
  price?: number;
  quantity: number;
  reason?: string;
  targetPositionId?: string;
}

export interface StrategyMemory {
  prices: number[];
  lastSignal: BotSignal;
  lastActionAt: number;
  previousFastAvg?: number;
  previousSlowAvg?: number;
  legacyValleyPeak?: LegacyValleyPeakMemory;
}

export interface BotMetrics {
  equity: number;
  realizedPnl: number;
  unrealizedPnl: number;
  netPnl: number;
  returnPct: number;
  feesPaid: number;
  tradeCount: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  peakEquity: number;
  maxDrawdownPct: number;
  exposurePct: number;
}

export interface PaperBotState {
  id: string;
  status: BotStatus;
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  startingQuote: number;
  quoteFree: number;
  quoteReserved: number;
  baseFree: number;
  baseReserved: number;
  avgEntryPrice: number;
  lastPrice: number;
  sequence: number;
  createdAt: number;
  updatedAt: number;
  realizedPnl: number;
  feesPaid: number;
  winningTrades: number;
  losingTrades: number;
  orders: TradingOrder[];
  fills: TradeFill[];
  memory: StrategyMemory;
  metrics: BotMetrics;
  config: StrategyConfig;
}

export type PositionLotSide = "long" | "short";

export type PositionLotStatus = "pending" | "open" | "partially-closed" | "closed";

export interface PositionLotBase {
  id: string;
  side: PositionLotSide;
  status: PositionLotStatus;
  sourceOrderId: string;
  openedAt: number;
  originalQuantity: number;
  filledQuantity: number;
  pendingQuantity: number;
  pendingQuote: number;
  pendingLimitPrice: number;
  closedQuantity: number;
  closedQuote: number;
  averagePrice: number;
}

export interface LongPositionLot extends PositionLotBase {
  side: "long";
  costQuote: number;
  remainingQuantity: number;
  remainingCostQuote: number;
  breakEvenSellPrice: number;
  maxLossSellPrice: number;
  recommendedSellQuote: number;
  recommendedSellQuantity: number;
  projectedRemainingQuantity: number;
  projectedRemainingCostQuote: number;
  projectedBreakEvenSellPrice: number;
  canReachLowerBaseline: boolean;
}

export interface ShortPositionLot extends PositionLotBase {
  side: "short";
  proceedsQuote: number;
  remainingQuantity: number;
  remainingProceedsQuote: number;
  breakEvenBuyPrice: number;
  maxLossBuyPrice: number;
  recommendedBuyQuote: number;
  recommendedBuyQuantity: number;
  projectedRemainingQuantity: number;
  projectedRemainingProceedsQuote: number;
  projectedBreakEvenBuyPrice: number;
  canReachUpperBaseline: boolean;
}

export type PositionLot = LongPositionLot | ShortPositionLot;

export interface PositionLedgerSummary {
  currentPrice: number;
  netMarketSellPrice: number;
  grossMarketBuyPrice: number;
  lowerBaselinePrice: number;
  upperBaselinePrice: number;
  lowerPriceExpectation: number;
  upperPriceExpectation: number;
  maxLossPct: number;
  feeAndSlippageRate: number;
  longQuantity: number;
  shortQuantity: number;
  longRemainingCostQuote: number;
  shortRemainingProceedsQuote: number;
  pendingLongQuantity: number;
  pendingShortQuantity: number;
  pendingLongQuote: number;
  pendingShortQuote: number;
  realizedQuotePnl: number;
}

export interface PositionLedger {
  summary: PositionLedgerSummary;
  longs: LongPositionLot[];
  shorts: ShortPositionLot[];
}

export interface BotEvent {
  type:
    | "order_created"
    | "order_filled"
    | "order_cancelled"
    | "status_changed"
    | "state_reset";
  at: number;
  message: string;
  order?: TradingOrder;
  fill?: TradeFill;
  state?: PaperBotState;
}

export interface BacktestSummary {
  symbol: string;
  source: "candles" | "orderbook-mid";
  startTime: number;
  endTime: number;
  targetStartTime?: number;
  targetEndTime?: number;
  eventsProcessed: number;
  candlesProcessed?: number;
  requests?: number;
  cacheHitCandles?: number;
  cacheMissCandles?: number;
  cacheFetchedCandles?: number;
  cacheSizeBytes?: number;
  cacheEvictedBytes?: number;
  cacheEvictedFiles?: number;
  stoppedEarly?: boolean;
  stopReason?: BacktestStopReason;
  survivedMs?: number;
  durationMs?: number;
  candlesPerSecond?: number;
  replayDurationMs?: number;
  finalEquity: number;
  netPnl: number;
  returnPct: number;
  maxDrawdownPct: number;
  tradeCount: number;
  winRate: number;
}

export interface EquityPoint {
  time: number;
  equity: number;
  price: number;
}

export interface BacktestResult {
  summary: BacktestSummary;
  equityCurve: EquityPoint[];
  orders: TradingOrder[];
  fills: TradeFill[];
  finalState: PaperBotState;
}

export interface BacktestProgressSnapshot {
  id: string;
  preset: BacktestPreset;
  status: BacktestRunStatus;
  source: "candles" | "orderbook-mid";
  startedAt: number;
  updatedAt: number;
  targetStartTime: number;
  targetEndTime: number;
  processedStartTime?: number;
  processedEndTime?: number;
  processedCandles: number;
  estimatedCandles: number;
  requests: number;
  cacheHitCandles?: number;
  cacheMissCandles?: number;
  cacheFetchedCandles?: number;
  cacheSizeBytes?: number;
  cacheEvictedBytes?: number;
  cacheEvictedFiles?: number;
  percent: number;
  equity: number;
  returnPct: number;
  stopReason?: BacktestStopReason;
  survivedMs?: number;
  candlesPerSecond?: number;
  message: string;
  error?: string;
  result?: BacktestResult;
}
