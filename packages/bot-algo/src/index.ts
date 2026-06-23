export {
  SimulatedTradingBot,
  createInitialBotState,
  createStrategyConfig,
  defaultStrategyConfig,
  type PartialStrategyConfig,
} from "./bot.js";
export {
  compactBacktestState,
  runBacktestFromCandles,
  runBacktestFromOrderBook,
  runBacktestFromTicks,
} from "./backtest.js";
export {
  createLegacyValleyPeakConfig,
  defaultLegacyValleyPeakConfig,
} from "./legacy-valley-peak.js";
export {
  analyzePositions,
  createPositionRiskConfig,
  defaultPositionRiskConfig,
  summarizeClosedPositions,
} from "./position-ledger.js";
export { calculateRiskAdjustedMetrics } from "./risk-metrics.js";
export type * from "./types.js";
