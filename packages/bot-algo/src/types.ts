export type BotStatus = "running" | "stopped";

export type OrderSide = "buy" | "sell";

export type OrderStatus = "open" | "filled" | "cancelled";

export type OrderType = "limit" | "market";

export type BotSignal = "buy" | "sell" | "hold";

export type StrategyAlgorithm = "legacy-valley-peak";

export type ShortMarginModel = "spot-borrow" | "futures-margin";

export type BacktestPreset =
  | "saved-candles"
  | "saved-orderbook"
  | "last-x"
  | "week"
  | "month"
  | "year"
  | "random-windows"
  | "random-length-windows";

export type BacktestRunStatus = "idle" | "running" | "completed" | "failed" | "cancelled";

export type BacktestStopReason =
  | "completed"
  | "wiped_out"
  | "liquidated"
  | "error"
  | "cancelled";

export type PositionEffect = "auto" | "open" | "close";

export type ExitGridPriceDistribution = "uniform" | "geometric";

export type ExitGridSizeDistribution = "geometric" | "linear" | "constant";

export interface PriceTick {
  symbol: string;
  eventTime: number;
  price: number;
  quantity?: number;
}

export interface TickProcessingOptions {
  collectEvents?: boolean;
  updateMetrics?: boolean;
  simulateLiquidation?: boolean;
  processOpenOrders?: boolean;
  deferMarketOrderFills?: boolean;
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
  maxLeverage: number;
  shortMarginModel: ShortMarginModel;
  longBorrowDepth: number;
  shortBorrowDepth: number;
  lockBorrowedLenderCollateral: boolean;
  borrowerProfitShareToLender: number;
  feeBps: number;
  maxPositionQuote: number;
  limitOffsetBps: number;
  maxOpenOrders: number;
  cooldownMs: number;
  staleOrderMs: number;
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
  buyConfirmationOffsets: number[];
  sellConfirmationOffsets: number[];
  saturationSec: number;
  buySpendRate: number;
  sellAmountRate: number;
  buySigma: number;
  sellSigma: number;
  minTradeQuote: number;
  maxTradeQuote: number;
  longSideEnabled: boolean;
  shortSideEnabled: boolean;
  exitGridEnabled: boolean;
  exitGridMarketEntry: boolean;
  exitGridOrderCount: number;
  exitGridMaxStepPct: number;
  exitGridPriceDistribution: ExitGridPriceDistribution;
  exitGridSizeDistribution: ExitGridSizeDistribution;
  exitGridSellFraction: number;
  exitGridMinProfitBps: number;
  exitGridResetBps: number;
  exitGridPositionMode: "aggregate" | "per-lot";
  exitGridResetMode: "higher-peak" | "filled-grid";
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
  exitGrids?: Record<string, LegacyExitGridMemory>;
}

export interface LegacyExitGridMemory {
  lotId: string;
  side?: PositionLotSide;
  entryPrice: number;
  entryQuantity: number;
  peakPrice: number;
  gridPeakPrice: number;
  troughPrice?: number;
  gridTroughPrice?: number;
  gridCreatedAt?: number;
  gridOrderIds: string[];
}

export interface TradingOrder {
  id: string;
  side: OrderSide;
  type: OrderType;
  trigger?: "above" | "below";
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
  positionEffect?: PositionEffect;
  lifetimeMs?: number;
  stopLossPrice?: number;
  takeProfitPrice?: number;
  manual?: boolean;
  liquidation?: boolean;
  liquidatedPositionCount?: number;
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
  positionEffect?: PositionEffect;
  lifetimeMs?: number;
  stopLossPrice?: number;
  takeProfitPrice?: number;
  manual?: boolean;
  liquidation?: boolean;
  liquidatedPositionCount?: number;
}

export interface ManualTradeInput {
  side: OrderSide;
  price?: number;
  quantity: number;
  reason?: string;
  targetPositionId?: string;
  positionEffect?: PositionEffect;
  lifetimeMs?: number;
  stopLossPrice?: number;
  takeProfitPrice?: number;
}

export interface ExchangeOrderUpdate {
  localOrderId?: string;
  externalOrderId?: string;
  clientOrderId?: string;
  side: OrderSide;
  type: OrderType;
  status: OrderStatus;
  price: number;
  quantity: number;
  filledQuantity: number;
  quoteQuantity?: number;
  feeQuote?: number;
  createdAt?: number;
  updatedAt?: number;
  reason?: string;
  positionEffect?: PositionEffect;
}

export interface ExchangeTradeFill {
  id: string;
  localOrderId?: string;
  externalOrderId?: string;
  clientOrderId?: string;
  side: OrderSide;
  price: number;
  quantity: number;
  quoteQuantity: number;
  feeQuote: number;
  realizedPnl?: number;
  filledAt: number;
  reason?: string;
  positionEffect?: PositionEffect;
}

export interface ExchangeReconciliationInput {
  orders?: ExchangeOrderUpdate[];
  fills?: ExchangeTradeFill[];
}

export interface StrategyMemory {
  prices: number[];
  lastSignal: BotSignal;
  lastActionAt: number;
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
  avgExitGridSpan: number;
  avgExitGridOrderCount: number;
  exitGridSpanCount: number;
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
  avgShortEntryPrice: number;
  lastPrice: number;
  sequence: number;
  createdAt: number;
  updatedAt: number;
  realizedPnl: number;
  feesPaid: number;
  exitGridSpanTotal: number;
  exitGridSpanCount: number;
  exitGridOrderCountTotal: number;
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
  exposureQuote: number;
  leverage: number;
  borrowedQuantity: number;
  borrowedQuote: number;
  internalBorrowedQuantity: number;
  internalBorrowedQuote: number;
  externalBorrowedQuantity: number;
  externalBorrowedQuote: number;
  borrowedFromPositionCount: number;
  lifetimeMs?: number;
  expiresAt?: number;
  stopLossPrice?: number;
  takeProfitPrice?: number;
  borrowLocked: boolean;
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
  grossExposureQuote: number;
  netExposureQuote: number;
  longExposureQuote: number;
  shortExposureQuote: number;
  internalBorrowedBaseQuantity: number;
  externalBorrowedBaseQuantity: number;
  internalBorrowedQuote: number;
  externalBorrowedQuote: number;
  effectiveLeverage: number;
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
  marketId?: string;
  displaySymbol?: string;
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
  sampleCount?: number;
  samplesProcessed?: number;
  sampleWindowMs?: number;
  sampleMinWindowMs?: number;
  sampleMaxWindowMs?: number;
  sampleLookbackMs?: number;
  marketCount?: number;
  randomPairCount?: number;
  marketSymbols?: string[];
  profitableSamples?: number;
  wipedOutSamples?: number;
  bestReturnPct?: number;
  worstReturnPct?: number;
  netPnlPerDay?: number;
  returnPctPerDay?: number;
  bestNetPnlPerDay?: number;
  worstNetPnlPerDay?: number;
  bestReturnPctPerDay?: number;
  worstReturnPctPerDay?: number;
  perfectMarginLeverage?: number;
  perfectMarginFinalEquity?: number;
  perfectMarginNetPnl?: number;
  perfectMarginReturnPct?: number;
  perfectMarginCapturePct?: number;
  perfectMarginCompoundedFinalEquity?: number;
  perfectMarginCompoundedNetPnl?: number;
  perfectMarginCompoundedReturnPct?: number;
  perfectMarginCompoundedCapturePct?: number;
  stoppedEarly?: boolean;
  stopReason?: BacktestStopReason;
  survivedMs?: number;
  durationMs?: number;
  candlesPerSecond?: number;
  replayDurationMs?: number;
  finalEquity: number;
  netPnl: number;
  returnPct: number;
  riskAdjustedReturn?: number;
  sharpeRatio?: number;
  maxDrawdownPct: number;
  tradeCount: number;
  winRate: number;
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  profitableClosedPositionRate: number;
  liquidatedPositionCount: number;
}

export interface BacktestSampleSummary {
  index: number;
  marketId?: string;
  symbol?: string;
  displaySymbol?: string;
  startTime: number;
  endTime: number;
  durationMs: number;
  eventsProcessed: number;
  candlesProcessed?: number;
  finalEquity: number;
  netPnl: number;
  returnPct: number;
  riskAdjustedReturn?: number;
  sharpeRatio?: number;
  netPnlPerDay: number;
  returnPctPerDay: number;
  perfectMarginLeverage?: number;
  perfectMarginFinalEquity?: number;
  perfectMarginNetPnl?: number;
  perfectMarginReturnPct?: number;
  perfectMarginCapturePct?: number;
  perfectMarginCompoundedFinalEquity?: number;
  perfectMarginCompoundedNetPnl?: number;
  perfectMarginCompoundedReturnPct?: number;
  perfectMarginCompoundedCapturePct?: number;
  maxDrawdownPct: number;
  tradeCount: number;
  winRate: number;
  closedPositionCount: number;
  profitableClosedPositionCount: number;
  profitableClosedPositionRate: number;
  liquidatedPositionCount: number;
  stoppedEarly?: boolean;
  stopReason?: BacktestStopReason;
  survivedMs?: number;
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
  samples?: BacktestSampleSummary[];
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
  currentSample?: number;
  sampleCount?: number;
  sampleWindowMs?: number;
  sampleMinWindowMs?: number;
  sampleMaxWindowMs?: number;
  sampleLookbackMs?: number;
  marketCount?: number;
  randomPairCount?: number;
  netPnlPerDay?: number;
  returnPctPerDay?: number;
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
