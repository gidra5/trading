import {
  SimulatedTradingBot,
  createInitialBotState,
  createStrategyConfig,
} from "./bot.js";
import type {
  BacktestResult,
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
  config?: Partial<StrategyConfig>;
  startingQuote?: number;
  maxEquityPoints?: number;
  maxReturnedOrders?: number;
  maxReturnedFills?: number;
}

const DEFAULT_MAX_EQUITY_POINTS = 800;
const DEFAULT_MAX_RETURNED_ORDERS = 2_000;
const DEFAULT_MAX_RETURNED_FILLS = 2_000;

export function runBacktestFromCandles(
  candles: Candle[],
  options: Omit<RunBacktestOptions, "source"> = {},
): BacktestResult {
  if (candles.length === 0) {
    throw new Error("Backtest requires at least one candle.");
  }

  const config = createStrategyConfig({
    symbol: candles[0]?.symbol ?? "BTCUSDT",
    ...(options.config ?? {}),
    ...(options.startingQuote ? { startingQuote: options.startingQuote } : {}),
  });
  const bot = new SimulatedTradingBot(createInitialBotState(config));
  const equityCurve: EquityPoint[] = [];
  const startedAt = Date.now();
  const sampleEvery = Math.max(
    1,
    Math.ceil(candles.length / (options.maxEquityPoints ?? DEFAULT_MAX_EQUITY_POINTS)),
  );

  for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index];
    replayCandle(bot, candle);

    if (index % sampleEvery === 0 || index === candles.length - 1) {
      const metrics = bot.markToMarket();
      equityCurve.push({
        time: candle.closeTime,
        equity: metrics.equity,
        price: candle.close,
      });
    }
  }

  const finalState = bot.view();
  return buildBacktestResult(
    "candles",
    {
      symbol: finalState.symbol,
      startTime: candles[0].openTime,
      endTime: candles[candles.length - 1].closeTime,
      eventsProcessed: candles.length * 4,
      candlesProcessed: candles.length,
      replayDurationMs: Date.now() - startedAt,
    },
    equityCurve,
    finalState,
    options,
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
  const equityCurve: EquityPoint[] = [];
  const startedAt = Date.now();
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
    startTime ||= snapshot.eventTime;
    endTime = snapshot.eventTime;
    processed += 1;
    bot.onTick(
      {
        symbol: snapshot.symbol,
        eventTime: snapshot.eventTime,
        price,
      },
      {
        collectEvents: false,
        updateMetrics: false,
      },
    );

    if (processed % sampleEvery === 0 || index === snapshots.length - 1) {
      const metrics = bot.markToMarket();
      equityCurve.push({
        time: snapshot.eventTime,
        equity: metrics.equity,
        price,
      });
    }
  }

  if (processed === 0) {
    throw new Error("Backtest requires at least one valid order book price.");
  }

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
  const equityCurve: EquityPoint[] = [];
  const startedAt = Date.now();
  const sampleEvery = Math.max(
    1,
    Math.ceil(ticks.length / (options.maxEquityPoints ?? DEFAULT_MAX_EQUITY_POINTS)),
  );

  for (let index = 0; index < ticks.length; index += 1) {
    const tick = ticks[index];
    bot.onTick(tick, {
      collectEvents: false,
      updateMetrics: false,
    });
    if (index % sampleEvery === 0 || index === ticks.length - 1) {
      const metrics = bot.markToMarket();
      equityCurve.push({
        time: tick.eventTime,
        equity: metrics.equity,
        price: tick.price,
      });
    }
  }

  const finalState = bot.view();
  return buildBacktestResult(
    options.source,
    {
      symbol: finalState.symbol,
      startTime: ticks[0].eventTime,
      endTime: ticks[ticks.length - 1].eventTime,
      eventsProcessed: ticks.length,
      replayDurationMs: Date.now() - startedAt,
    },
    equityCurve,
    finalState,
    options,
  );
}

function replayCandle(bot: SimulatedTradingBot, candle: Candle): void {
  const duration = Math.max(1, candle.closeTime - candle.openTime);
  const symbol = candle.symbol;
  const options = {
    collectEvents: false,
    updateMetrics: false,
  };
  bot.onTick(
    {
      symbol,
      eventTime: candle.openTime,
      price: candle.open,
      quantity: candle.volume * 0.2,
    },
    options,
  );
  bot.onTick(
    {
      symbol,
      eventTime: candle.openTime + duration * 0.33,
      price: candle.high,
      quantity: candle.volume * 0.25,
    },
    options,
  );
  bot.onTick(
    {
      symbol,
      eventTime: candle.openTime + duration * 0.66,
      price: candle.low,
      quantity: candle.volume * 0.25,
    },
    options,
  );
  bot.onTick(
    {
      symbol,
      eventTime: candle.closeTime,
      price: candle.close,
      quantity: candle.volume * 0.3,
    },
    options,
  );
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
): BacktestResult {
  const resultState = compactBacktestState(finalState, options);

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
      maxDrawdownPct: finalState.metrics.maxDrawdownPct,
      tradeCount: finalState.metrics.tradeCount,
      winRate: finalState.metrics.winRate,
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
  return {
    prices: tail(memory.prices, Math.max(50, config.slowWindow * 2)).slice(),
    lastSignal: memory.lastSignal,
    lastActionAt: memory.lastActionAt,
    previousFastAvg: memory.previousFastAvg,
    previousSlowAvg: memory.previousSlowAvg,
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

function tail<T>(items: readonly T[], limit: number): readonly T[] {
  if (limit >= items.length) {
    return items;
  }
  return items.slice(items.length - limit);
}
