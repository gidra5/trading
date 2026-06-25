import {
  SimulatedTradingBot,
  createInitialBotState,
  createStrategyConfig,
  type PartialStrategyConfig,
} from "./bot.js";
import { defaultPositionRiskConfig, summarizeClosedPositions } from "./position-ledger.js";
import { calculateRiskAdjustedMetrics } from "./risk-metrics.js";
import type {
  BacktestSummary,
  BacktestCandleChart,
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
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
  maxChartCandles?: number;
  startIndex?: number;
  endIndex?: number;
}

const DEFAULT_MAX_EQUITY_POINTS = 800;
const DEFAULT_MAX_RETURNED_ORDERS = 2_000;
const DEFAULT_MAX_RETURNED_FILLS = 2_000;
const DEFAULT_MAX_CHART_CANDLES = 2_000;
const DEFAULT_MAX_CHART_ANNOTATIONS = 5_000;
const SMA_COLORS = ["#38bdf8", "#f5b84b", "#a78bfa", "#22c55e", "#f472b6", "#eab308"];

export interface BacktestChartCollector {
  sampleEvery: number;
  candles: Candle[];
  smaSeries: BacktestChartSmaSeries[];
  annotations: BacktestChartAnnotation[];
  lastObservedOrderCount: number;
  lastObservedFillCount: number;
  lastObservedSignalAt?: number;
}

interface PerfectMarginBenchmarkAccumulator {
  startingQuote: number;
  leverage: number;
  roundTripFrictionRate: number;
  previousPrice?: number;
  netPnl: number;
  compoundedEquity: number;
}

const PERFECT_MARGIN_MIN_SLIPPAGE_BPS = defaultPositionRiskConfig.marketSlippageBps;

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

export function createBacktestChartCollector(
  config: Readonly<StrategyConfig>,
  totalCandles: number,
  maxChartCandles = DEFAULT_MAX_CHART_CANDLES,
): BacktestChartCollector {
  const selectedIndices = selectedSmaIndices(config);
  return {
    sampleEvery: Math.max(1, Math.ceil(totalCandles / Math.max(1, maxChartCandles))),
    candles: [],
    smaSeries: selectedIndices.map((index, colorIndex) => ({
      index,
      windowSec: config.legacyValleyPeak.averagingRangesSec[index] ?? 0,
      label: `${formatWindowLabel(config.legacyValleyPeak.averagingRangesSec[index] ?? 0)} SMA`,
      color: SMA_COLORS[colorIndex % SMA_COLORS.length] ?? "#38bdf8",
      points: [],
    })),
    annotations: [],
    lastObservedOrderCount: 0,
    lastObservedFillCount: 0,
  };
}

export function observeBacktestChartCandle(
  collector: BacktestChartCollector,
  bot: SimulatedTradingBot,
  candle: Candle,
  processedCandles: number,
  forceSample = false,
): void {
  observeBacktestChartAnnotations(collector, bot.view());
  if (
    !forceSample &&
    processedCandles !== 1 &&
    processedCandles % collector.sampleEvery !== 0
  ) {
    return;
  }

  const previous = collector.candles.at(-1);
  if (previous?.openTime === candle.openTime && previous.interval === candle.interval) {
    collector.candles[collector.candles.length - 1] = { ...candle };
  } else {
    collector.candles.push({ ...candle });
  }

  const averages = bot.view().memory.legacyValleyPeakDebug?.averages ?? [];
  for (const series of collector.smaSeries) {
    const avg = averages.find((item) => item.index === series.index)?.avg;
    if (Number.isFinite(avg)) {
      const lastPoint = series.points.at(-1);
      if (lastPoint?.time === candle.closeTime) {
        lastPoint.value = avg as number;
      } else {
        series.points.push({
          time: candle.closeTime,
          value: avg as number,
        });
      }
    }
  }
}

export function finalizeBacktestCandleChart(
  collector: BacktestChartCollector,
): BacktestCandleChart | undefined {
  if (collector.candles.length === 0) {
    return undefined;
  }

  return {
    candles: collector.candles,
    smaSeries: collector.smaSeries.filter((series) => series.points.length > 0),
    annotations: collector.annotations,
  };
}

function observeBacktestChartAnnotations(
  collector: BacktestChartCollector,
  state: Readonly<PaperBotState>,
): void {
  const memory = state.memory;
  if (
    memory.lastExtremaSignal &&
    memory.lastExtremaSignalAt &&
    memory.lastExtremaSignalAt !== collector.lastObservedSignalAt
  ) {
    collector.lastObservedSignalAt = memory.lastExtremaSignalAt;
    appendBacktestAnnotation(collector, {
      time: memory.lastExtremaSignalAt,
      price: memory.lastExtremaSignalPrice ?? state.lastPrice,
      kind: memory.lastExtremaSignal === "buy" ? "buy-signal" : "sell-signal",
      label: memory.lastExtremaSignal === "buy" ? "Valley signal" : "Peak signal",
      reason: memory.lastExtremaSignalReason,
    });
  }

  for (
    let index = collector.lastObservedOrderCount;
    index < state.orders.length;
    index += 1
  ) {
    const order = state.orders[index];
    if (!order) {
      continue;
    }
    appendBacktestAnnotation(collector, {
      time: order.createdAt,
      price: order.price,
      kind: order.side === "buy" ? "buy-order" : "sell-order",
      label: `${order.side.toUpperCase()} ${order.type} ${order.positionEffect ?? "auto"}`,
      reason: order.reason,
      orderId: order.id,
      targetPositionId: order.targetPositionId,
    });
  }
  collector.lastObservedOrderCount = state.orders.length;

  for (
    let index = collector.lastObservedFillCount;
    index < state.fills.length;
    index += 1
  ) {
    const fill = state.fills[index];
    if (!fill) {
      continue;
    }
    appendBacktestAnnotation(collector, {
      time: fill.filledAt,
      price: fill.price,
      kind: fill.side === "buy" ? "buy-fill" : "sell-fill",
      label: `${fill.side.toUpperCase()} fill ${fill.positionEffect ?? "auto"}`,
      reason: fill.reason,
      orderId: fill.orderId,
      fillId: fill.id,
      targetPositionId: fill.targetPositionId,
    });
  }
  collector.lastObservedFillCount = state.fills.length;
}

function appendBacktestAnnotation(
  collector: BacktestChartCollector,
  annotation: BacktestChartAnnotation,
): void {
  if (!Number.isFinite(annotation.time) || !Number.isFinite(annotation.price)) {
    return;
  }

  collector.annotations.push(annotation);
  if (collector.annotations.length > DEFAULT_MAX_CHART_ANNOTATIONS) {
    collector.annotations.splice(
      0,
      collector.annotations.length - DEFAULT_MAX_CHART_ANNOTATIONS,
    );
  }
}

function selectedSmaIndices(config: Readonly<StrategyConfig>): number[] {
  const legacy = config.legacyValleyPeak;
  const maxIndex = legacy.averagingRangesSec.length - 1;
  const indices = new Set<number>();
  const add = (index: number) => {
    if (Number.isFinite(index) && index >= 0 && index <= maxIndex) {
      indices.add(Math.round(index));
    }
  };

  add(legacy.buyDataIndex);
  add(legacy.sellDataIndex);
  for (const offset of legacy.buyConfirmationOffsets) {
    add(legacy.buyDataIndex + offset);
  }
  for (const offset of legacy.sellConfirmationOffsets) {
    add(legacy.sellDataIndex + offset);
  }

  return [...indices].sort((a, b) => a - b);
}

function formatWindowLabel(seconds: number): string {
  if (seconds >= 86_400 && seconds % 86_400 === 0) {
    return `${seconds / 86_400}d`;
  }
  if (seconds >= 3_600 && seconds % 3_600 === 0) {
    return `${seconds / 3_600}h`;
  }
  if (seconds >= 60 && seconds % 60 === 0) {
    return `${seconds / 60}m`;
  }
  return `${seconds}s`;
}

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
  const chartCollector = createBacktestChartCollector(
    config,
    candleCount,
    options.maxChartCandles,
  );
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
    observeBacktestChartCandle(
      chartCollector,
      bot,
      candle,
      processedCandles,
      index === endIndex - 1 || liquidated,
    );

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
    finalizeBacktestCandleChart(chartCollector),
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
        debug: false,
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
      debug: false,
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
  const slippageBps = Math.max(
    PERFECT_MARGIN_MIN_SLIPPAGE_BPS,
    Math.max(0, config.positionRisk.marketSlippageBps),
  );
  return {
    startingQuote: config.startingQuote,
    leverage: config.maxLeverage,
    roundTripFrictionRate:
      2 * ((Math.max(0, config.feeBps) + slippageBps) / 10_000),
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
  candleChart?: BacktestCandleChart,
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
      maxEntryLeverage: finalState.metrics.maxEntryLeverage,
      maxEffectiveLeverage: finalState.metrics.maxEffectiveLeverage,
      tradeCount: finalState.metrics.tradeCount,
      winRate: finalState.metrics.winRate,
      ...closedPositionStats,
      ...perfectMargin,
      stoppedEarly: stopReason !== "completed",
      stopReason,
    },
    equityCurve,
    candleChart,
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
    lastExtremaSignal: memory.lastExtremaSignal,
    lastExtremaSignalAt: memory.lastExtremaSignalAt,
    lastExtremaSignalPrice: memory.lastExtremaSignalPrice,
    lastExtremaSignalReason: memory.lastExtremaSignalReason,
    legacyValleyPeakDebug: memory.legacyValleyPeakDebug
      ? structuredClone(memory.legacyValleyPeakDebug)
      : undefined,
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
