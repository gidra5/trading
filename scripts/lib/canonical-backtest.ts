import {
  createStrategyConfig,
  type BacktestResult,
  type Candle,
  type PartialStrategyConfig,
} from "../../packages/bot-algo/src/index.js";
import { runBotBacktestFromCandles } from "../../apps/server/src/bot-backtest.js";

export interface CanonicalCandleBacktestOptions {
  config?: PartialStrategyConfig;
  startIndex?: number;
  endIndex?: number;
  maxEquityPoints?: number;
  maxChartCandles?: number;
  maxReturnedOrders?: number;
  maxReturnedFills?: number;
}

/** The single script-facing entry point for the same bot and simulator used by
 * the server backtester. Indices use the normal half-open [start, end) range. */
export async function runCanonicalBacktestFromCandles(
  candles: readonly Candle[],
  options: CanonicalCandleBacktestOptions = {},
): Promise<BacktestResult> {
  const start = clampIndex(options.startIndex, candles.length, 0);
  const end = clampIndex(options.endIndex, candles.length, candles.length);
  const replay = candles.slice(Math.min(start, end), Math.max(start, end));
  const result = await runBotBacktestFromCandles(replay, {
    config: createStrategyConfig(options.config),
    maxEquityPoints: options.maxEquityPoints,
    maxChartCandles: options.maxChartCandles,
  });

  result.orders = tail(result.orders, options.maxReturnedOrders);
  result.fills = tail(result.fills, options.maxReturnedFills);
  result.finalState.orders = result.orders;
  result.finalState.fills = result.fills;
  return result;
}

function clampIndex(value: number | undefined, length: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(0, Math.min(length, Math.floor(value as number)));
}

function tail<T>(items: T[], limit: number | undefined): T[] {
  if (limit === undefined || !Number.isFinite(limit)) return items;
  const count = Math.max(0, Math.floor(limit));
  return count >= items.length ? items : items.slice(items.length - count);
}
