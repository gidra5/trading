export {
  SimulatedExecutionEngine,
  createInitialBotState,
  createStrategyConfig,
  defaultStrategyConfig,
  type PartialStrategyConfig,
} from "./legacy/execution-simulator.js";
export * from "./bot.js";
export * from "./peak-valley-strategy.js";
export type * from "./strategy.js";
export type * from "./trading-api.js";
export * from "./indicators.js";
export {
  compactBacktestState,
  createBacktestChartCollector,
  finalizeBacktestCandleChart,
  observeBacktestChartCandle,
  runBacktestFromCandles,
  runBacktestFromOrderBook,
  runBacktestFromTicks,
  type BacktestChartCollector,
} from "./legacy/backtest.js";
export {
  aggregateExtremaOrderMassSummaries,
  createExtremaOrderMassCollector,
  observeExtremaOrderMassCandle,
  summarizeExtremaOrderMass,
  type ExtremaOrderMassCollector,
} from "./legacy/extrema-order-mass.js";
export {
  createLegacyValleyPeakConfig,
  defaultLegacyValleyPeakConfig,
  legacyValleyPeakHistoricalWarmupSec,
  legacyValleyPeakObservedPriceRangeWarmupRatio,
  legacyValleyPeakObservedSignalWarmupSec,
  legacyValleyPeakObservedWarmupSec,
  legacyValleyPeakPriceRangeWarmupSec,
  legacyValleyPeakSignalWarmupSec,
  legacyValleyPeakAsymmetricShortFavoringConfig,
  legacyValleyPeakReferenceConfigs,
  legacyValleyPeakStrictSymmetricConfig,
} from "./legacy/valley-peak.js";
export {
  analyzePositions,
  createPositionRiskConfig,
  defaultPositionRiskConfig,
  summarizeClosedPositions,
} from "./legacy/position-ledger.js";
export {
  FuturesMarginBalanceModel,
  LeveragedBalanceModel,
  SpotBorrowBalanceModel,
  createLeveragedBalanceModel,
  type BalanceEntryCapacityInput,
  type BalanceEntrySide,
  type BalanceLiquidationInput,
  type BalanceProjection,
} from "./legacy/leveraged-balance.js";
export { calculateRiskAdjustedMetrics } from "./legacy/risk-metrics.js";
export type * from "./legacy/types.js";
