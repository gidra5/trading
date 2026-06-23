import {
  SimulatedTradingBot,
  createInitialBotState,
  createStrategyConfig,
  type PartialStrategyConfig,
} from "./bot.js";
import { summarizeClosedPositions } from "./position-ledger.js";
import { calculateRiskAdjustedMetrics } from "./risk-metrics.js";
import type {
  BacktestSummary,
  BacktestResult,
  BacktestStopReason,
  Candle,
  EquityPoint,
  OrderBookSnapshot,
  PaperBotState,
  PriceTick,
  StrategyConfig,
  StrategyMemory,
} from "./types.js";

export interface RunBacktestOptions {
  source: "candles" | "orderbook-mid";
  config?: PartialStrategyConfig;
  startingQuote?: number;
  maxEquityPoints?: number;
  maxReturnedOrders?: number;
  maxReturnedFills?: number;
  startIndex?: number;
  endIndex?: number;
}

const DEFAULT_MAX_EQUITY_POINTS = 800;
const DEFAULT_MAX_RETURNED_ORDERS = 2_000;
const DEFAULT_MAX_RETURNED_FILLS = 2_000;

interface PerfectMarginBenchmarkAccumulator {
  startingQuote: number;
  leverage: number;
  roundTripFrictionRate: number;
  previousPrice?: number;
  netPnl: number;
  compoundedEquity: number;
}

type PerfectMarginBenchmark = Pick<
  BacktestSummary,
  | "perfectMarginLeverage"
  | "perfectMarginFinalEquity"
  | "perfectMarginNetPnl"
  | "perfectMarginReturnPct"
  | "perfectMarginCapturePct"
  | "perfectMarginCompoundedFinalEquity"
  | "perfectMarginCompoundedNetPnl"
  | "perfectMarginCompoundedReturnPct"
  | "perfectMarginCompoundedCapturePct"
>;

export function runBacktestFromCandles(
  candles: Candle[],
  options: Omit<RunBacktestOptions, "source"> = {},
): BacktestResult {
  const startIndex = clampReplayIndex(options.startIndex, 0, candles.length, 0);
  const endIndex = clampReplayIndex(
    options.endIndex,
    startIndex,
    candles.length,
    candles.length,
  );
  const candleCount = endIndex - startIndex;
  if (candleCount <= 0) {
    throw new Error("Backtest requires at least one candle.");
  }

  const firstCandle = candles[startIndex];
  const lastCandle = candles[endIndex - 1];
  const config = createStrategyConfig({
    symbol: firstCandle?.symbol ?? "BTCUSDT",
    ...(options.config ?? {}),
    ...(options.startingQuote ? { startingQuote: options.startingQuote } : {}),
  });
  const bot = new SimulatedTradingBot(createInitialBotState(config));
  const perfectMargin = createPerfectMarginBenchmark(config);
  const equityCurve: EquityPoint[] = [];
  const startedAt = Date.now();
  let stopReason: BacktestStopReason = "completed";
  let processedCandles = 0;
  let processedEndTime = firstCandle.closeTime;
  const sampleEvery = Math.max(
    1,
    Math.ceil(candleCount / (options.maxEquityPoints ?? DEFAULT_MAX_EQUITY_POINTS)),
  );

  for (let index = startIndex; index < endIndex; index += 1) {
    const candle = candles[index];
    const relativeIndex = index - startIndex;
    let liquidated = false;
    processedCandles += 1;
    processedEndTime = candle.closeTime;
    if (replayCandle(bot, candle, perfectMargin)) {
      stopReason = "liquidated";
      liquidated = true;
    }

    if (relativeIndex % sampleEvery === 0 || index === endIndex - 1 || liquidated) {
      const metrics = bot.markToMarket();
      equityCurve.push({
        time: liquidated ? bot.view().updatedAt : candle.closeTime,
        equity: metrics.equity,
        price: liquidated ? bot.view().lastPrice : candle.close,
      });
    }

    if (stopReason === "liquidated") {
      break;
    }
  }

  bot.markToMarket();
  const finalState = bot.view();
  return buildBacktestResult(
    "candles",
    {
      symbol: finalState.symbol,
      startTime: firstCandle.openTime,
      endTime: stopReason === "liquidated" ? processedEndTime : lastCandle.closeTime,
      eventsProcessed: processedCandles * 4,
      candlesProcessed: processedCandles,
      replayDurationMs: Date.now() - startedAt,
    },
    equityCurve,
    finalState,
    options,
    finalizePerfectMarginBenchmark(perfectMargin, finalState.metrics.netPnl),
    stopReason,
  );
}

export function runBacktestFromOrderBook(
  snapshots: OrderBookSnapshot[],
  options: Omit<RunBacktestOptions, "source"> = {},
): BacktestResult {
  if (snapshots.length === 0) {
    throw new Error("Backtest requires at least one order book snapshot.");
  }

  const firstSnapshot = snapshots[0];
  const config = createStrategyConfig({
    symbol: firstSnapshot?.symbol ?? "BTCUSDT",
    ...(options.config ?? {}),
    ...(options.startingQuote ? { startingQuote: options.startingQuote } : {}),
  });
  const bot = new SimulatedTradingBot(createInitialBotState(config));
  const perfectMargin = createPerfectMarginBenchmark(config);
  const equityCurve: EquityPoint[] = [];
  const startedAt = Date.now();
  let stopReason: BacktestStopReason = "completed";
  const sampleEvery = Math.max(
    1,
    Math.ceil(snapshots.length / (options.maxEquityPoints ?? DEFAULT_MAX_EQUITY_POINTS)),
  );
  let processed = 0;
  let startTime = 0;
  let endTime = 0;

  for (let index = 0; index < snapshots.length; index += 1) {
    const snapshot = snapshots[index];
    const bestBid = snapshot.bids[0]?.price;
    const bestAsk = snapshot.asks[0]?.price;
    if (!bestBid || !bestAsk) {
      continue;
    }

    const price = (bestBid + bestAsk) / 2;
    observePerfectMarginPrice(perfectMargin, price);
    startTime ||= snapshot.eventTime;
    endTime = snapshot.eventTime;
    processed += 1;
    const liquidatedBefore = bot.liquidatedPositionCount();
    bot.onTick(
      {
        symbol: snapshot.symbol,
        eventTime: snapshot.eventTime,
        price,
      },
      {
        collectEvents: false,
        updateMetrics: false,
        simulateLiquidation: true,
      },
    );
    const liquidated = bot.liquidatedPositionCount() > liquidatedBefore;

    if (processed % sampleEvery === 0 || index === snapshots.length - 1 || liquidated) {
      const metrics = bot.markToMarket();
      equityCurve.push({
        time: snapshot.eventTime,
        equity: metrics.equity,
        price,
      });
    }

    if (liquidated) {
      stopReason = "liquidated";
      break;
    }
  }

  if (processed === 0) {
    throw new Error("Backtest requires at least one valid order book price.");
  }

  bot.markToMarket();
  const finalState = bot.view();
  return buildBacktestResult(
    "orderbook-mid",
    {
      symbol: finalState.symbol,
      startTime,
      endTime,
      eventsProcessed: processed,
      replayDurationMs: Date.now() - startedAt,
    },
    equityCurve,
    finalState,
    options,
    finalizePerfectMarginBenchmark(perfectMargin, finalState.metrics.netPnl),
    stopReason,
  );
}

export function runBacktestFromTicks(
  ticks: PriceTick[],
  options: RunBacktestOptions,
): BacktestResult {
  if (ticks.length === 0) {
    throw new Error("Backtest requires at least one price event.");
  }

  const config = createStrategyConfig({
    symbol: ticks[0]?.symbol ?? "BTCUSDT",
    ...(options.config ?? {}),
    ...(options.startingQuote ? { startingQuote: options.startingQuote } : {}),
  });
  const state = createInitialBotState(config);
  const bot = new SimulatedTradingBot(state);
  const perfectMargin = createPerfectMarginBenchmark(config);
  const equityCurve: EquityPoint[] = [];
  const startedAt = Date.now();
  let stopReason: BacktestStopReason = "completed";
  let processedTicks = 0;
  let processedEndTime = ticks[0].eventTime;
  const sampleEvery = Math.max(
    1,
    Math.ceil(ticks.length / (options.maxEquityPoints ?? DEFAULT_MAX_EQUITY_POINTS)),
  );

  for (let index = 0; index < ticks.length; index += 1) {
    const tick = ticks[index];
    observePerfectMarginPrice(perfectMargin, tick.price);
    processedTicks += 1;
    processedEndTime = tick.eventTime;
    const liquidatedBefore = bot.liquidatedPositionCount();
    bot.onTick(tick, {
      collectEvents: false,
      updateMetrics: false,
      simulateLiquidation: true,
    });
    const liquidated = bot.liquidatedPositionCount() > liquidatedBefore;
    if (index % sampleEvery === 0 || index === ticks.length - 1 || liquidated) {
      const metrics = bot.markToMarket();
      equityCurve.push({
        time: tick.eventTime,
        equity: metrics.equity,
        price: tick.price,
      });
    }

    if (liquidated) {
      stopReason = "liquidated";
      break;
    }
  }

  bot.markToMarket();
  const finalState = bot.view();
  return buildBacktestResult(
    options.source,
    {
      symbol: finalState.symbol,
      startTime: ticks[0].eventTime,
      endTime: stopReason === "liquidated" ? processedEndTime : ticks[ticks.length - 1].eventTime,
      eventsProcessed: processedTicks,
      replayDurationMs: Date.now() - startedAt,
    },
    equityCurve,
    finalState,
    options,
    finalizePerfectMarginBenchmark(perfectMargin, finalState.metrics.netPnl),
    stopReason,
  );
}

function replayCandle(
  bot: SimulatedTradingBot,
  candle: Candle,
  perfectMargin?: PerfectMarginBenchmarkAccumulator,
): boolean {
  const duration = Math.max(1, candle.closeTime - candle.openTime);
  const highTime = candle.openTime + duration * 0.33;
  const lowTime = candle.openTime + duration * 0.66;
  const liquidatedBefore = bot.liquidatedPositionCount();

  observePerfectMarginPrice(perfectMargin, candle.open);
  bot.onReplayPriceTick(candle.openTime, candle.open);
  if (bot.liquidatedPositionCount() > liquidatedBefore) {
    return true;
  }
  observePerfectMarginPrice(perfectMargin, candle.high);
  bot.onReplayPriceTick(highTime, candle.high);
  if (bot.liquidatedPositionCount() > liquidatedBefore) {
    return true;
  }
  observePerfectMarginPrice(perfectMargin, candle.low);
  bot.onReplayPriceTick(lowTime, candle.low);
  if (bot.liquidatedPositionCount() > liquidatedBefore) {
    return true;
  }
  observePerfectMarginPrice(perfectMargin, candle.close);
  bot.onReplayPriceTick(candle.closeTime, candle.close);
  return bot.liquidatedPositionCount() > liquidatedBefore;
}

function createPerfectMarginBenchmark(
  config: StrategyConfig,
): PerfectMarginBenchmarkAccumulator {
  return {
    startingQuote: config.startingQuote,
    leverage: config.maxLeverage,
    roundTripFrictionRate:
      2 *
      ((Math.max(0, config.feeBps) + Math.max(0, config.positionRisk.marketSlippageBps)) /
        10_000),
    netPnl: 0,
    compoundedEquity: config.startingQuote,
  };
}

function observePerfectMarginPrice(
  benchmark: PerfectMarginBenchmarkAccumulator | undefined,
  price: number,
): void {
  if (!benchmark || !Number.isFinite(price) || price <= 0) {
    return;
  }

  if (benchmark.previousPrice && benchmark.previousPrice > 0) {
    const changeRate = Math.abs(price - benchmark.previousPrice) / benchmark.previousPrice;
    const netRate = changeRate - benchmark.roundTripFrictionRate;
    if (netRate > 0) {
      benchmark.netPnl += benchmark.startingQuote * benchmark.leverage * netRate;
      benchmark.compoundedEquity *= 1 + benchmark.leverage * netRate;
    }
  }
  benchmark.previousPrice = price;
}

function finalizePerfectMarginBenchmark(
  benchmark: PerfectMarginBenchmarkAccumulator,
  actualNetPnl: number,
): PerfectMarginBenchmark {
  const netPnl = benchmark.netPnl;
  const finalEquity = benchmark.startingQuote + netPnl;
  const returnPct =
    benchmark.startingQuote > 0 ? (netPnl / benchmark.startingQuote) * 100 : 0;
  const compoundedFinalEquity = benchmark.compoundedEquity;
  const compoundedNetPnl = compoundedFinalEquity - benchmark.startingQuote;
  const compoundedReturnPct =
    benchmark.startingQuote > 0
      ? (compoundedNetPnl / benchmark.startingQuote) * 100
      : 0;

  return {
    perfectMarginLeverage: benchmark.leverage,
    perfectMarginFinalEquity: finalEquity,
    perfectMarginNetPnl: netPnl,
    perfectMarginReturnPct: returnPct,
    perfectMarginCapturePct: netPnl > 0 ? (actualNetPnl / netPnl) * 100 : undefined,
    perfectMarginCompoundedFinalEquity: compoundedFinalEquity,
    perfectMarginCompoundedNetPnl: compoundedNetPnl,
    perfectMarginCompoundedReturnPct: compoundedReturnPct,
    perfectMarginCompoundedCapturePct:
      compoundedNetPnl > 0 ? (actualNetPnl / compoundedNetPnl) * 100 : undefined,
  };
}

function buildBacktestResult(
  source: "candles" | "orderbook-mid",
  processed: {
    symbol: string;
    startTime: number;
    endTime: number;
    eventsProcessed: number;
    candlesProcessed?: number;
    replayDurationMs?: number;
  },
  equityCurve: EquityPoint[],
  finalState: Readonly<PaperBotState>,
  options: Pick<RunBacktestOptions, "maxReturnedOrders" | "maxReturnedFills">,
  perfectMargin: PerfectMarginBenchmark,
  stopReason: BacktestStopReason = "completed",
): BacktestResult {
  const resultState = compactBacktestState(finalState, options);
  const riskMetrics = calculateRiskAdjustedMetrics(
    equityCurve,
    finalState.metrics.returnPct,
    finalState.metrics.maxDrawdownPct,
  );
  const closedPositionStats = summarizeClosedPositions(finalState);

  return {
    summary: {
      symbol: processed.symbol,
      source,
      startTime: processed.startTime,
      endTime: processed.endTime,
      eventsProcessed: processed.eventsProcessed,
      candlesProcessed: processed.candlesProcessed,
      replayDurationMs: processed.replayDurationMs,
      candlesPerSecond:
        processed.candlesProcessed && processed.replayDurationMs && processed.replayDurationMs > 0
          ? (processed.candlesProcessed / processed.replayDurationMs) * 1000
          : undefined,
      finalEquity: finalState.metrics.equity,
      netPnl: finalState.metrics.netPnl,
      returnPct: finalState.metrics.returnPct,
      ...riskMetrics,
      maxDrawdownPct: finalState.metrics.maxDrawdownPct,
      tradeCount: finalState.metrics.tradeCount,
      winRate: finalState.metrics.winRate,
      ...closedPositionStats,
      ...perfectMargin,
      stoppedEarly: stopReason !== "completed",
      stopReason,
    },
    equityCurve,
    orders: resultState.orders,
    fills: resultState.fills,
    finalState: resultState,
  };
}

export function compactBacktestState(
  state: Readonly<PaperBotState>,
  options: {
    maxReturnedOrders?: number;
    maxReturnedFills?: number;
  } = {},
): PaperBotState {
  const maxOrders = normalizeResultLimit(
    options.maxReturnedOrders,
    DEFAULT_MAX_RETURNED_ORDERS,
  );
  const maxFills = normalizeResultLimit(
    options.maxReturnedFills,
    DEFAULT_MAX_RETURNED_FILLS,
  );

  return {
    ...state,
    orders: tail(state.orders, maxOrders).map((order) => ({ ...order })),
    fills: tail(state.fills, maxFills).map((fill) => ({ ...fill })),
    memory: compactBacktestMemory(state.memory, state.config),
    metrics: { ...state.metrics },
    config: structuredClone(state.config),
  };
}

function compactBacktestMemory(
  memory: Readonly<StrategyMemory>,
  config: Readonly<StrategyConfig>,
): StrategyMemory {
  const priceLimit = Math.max(
    50,
    config.legacyValleyPeak.averagingRangesSec.length * 100,
  );

  return {
    prices: tail(memory.prices, priceLimit).slice(),
    lastSignal: memory.lastSignal,
    lastActionAt: memory.lastActionAt,
  };
}

function normalizeResultLimit(value: number | undefined, fallback: number): number {
  if (value === undefined) {
    return fallback;
  }
  if (!Number.isFinite(value)) {
    return Number.MAX_SAFE_INTEGER;
  }
  return Math.max(0, Math.round(value));
}

function clampReplayIndex(
  value: number | undefined,
  minValue: number,
  maxValue: number,
  fallback: number,
): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(minValue, Math.min(maxValue, Math.floor(value)));
}

function tail<T>(items: readonly T[], limit: number): readonly T[] {
  if (limit >= items.length) {
    return items;
  }
  return items.slice(items.length - limit);
}
