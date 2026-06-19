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
} from "./types.js";

export interface RunBacktestOptions {
  source: "candles" | "orderbook-mid";
  config?: Partial<StrategyConfig>;
  startingQuote?: number;
}

export function runBacktestFromCandles(
  candles: Candle[],
  options: Omit<RunBacktestOptions, "source"> = {},
): BacktestResult {
  const ticks = candles.flatMap(candleToTicks);
  return runBacktestFromTicks(ticks, {
    ...options,
    source: "candles",
  });
}

export function runBacktestFromOrderBook(
  snapshots: OrderBookSnapshot[],
  options: Omit<RunBacktestOptions, "source"> = {},
): BacktestResult {
  const ticks = snapshots
    .map((snapshot): PriceTick | undefined => {
      const bestBid = snapshot.bids[0]?.price;
      const bestAsk = snapshot.asks[0]?.price;
      if (!bestBid || !bestAsk) {
        return undefined;
      }

      return {
        symbol: snapshot.symbol,
        eventTime: snapshot.eventTime,
        price: (bestBid + bestAsk) / 2,
      };
    })
    .filter((tick): tick is PriceTick => Boolean(tick));

  return runBacktestFromTicks(ticks, {
    ...options,
    source: "orderbook-mid",
  });
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

  for (const tick of ticks) {
    bot.onTick(tick);
    const snapshot = bot.snapshot();
    equityCurve.push({
      time: tick.eventTime,
      equity: snapshot.metrics.equity,
      price: tick.price,
    });
  }

  const finalState = bot.snapshot();
  return buildBacktestResult(options.source, ticks, equityCurve, finalState);
}

function candleToTicks(candle: Candle): PriceTick[] {
  const duration = Math.max(1, candle.closeTime - candle.openTime);
  const symbol = candle.symbol;

  return [
    {
      symbol,
      eventTime: candle.openTime,
      price: candle.open,
      quantity: candle.volume * 0.2,
    },
    {
      symbol,
      eventTime: candle.openTime + duration * 0.33,
      price: candle.high,
      quantity: candle.volume * 0.25,
    },
    {
      symbol,
      eventTime: candle.openTime + duration * 0.66,
      price: candle.low,
      quantity: candle.volume * 0.25,
    },
    {
      symbol,
      eventTime: candle.closeTime,
      price: candle.close,
      quantity: candle.volume * 0.3,
    },
  ];
}

function buildBacktestResult(
  source: "candles" | "orderbook-mid",
  ticks: PriceTick[],
  equityCurve: EquityPoint[],
  finalState: PaperBotState,
): BacktestResult {
  const firstTick = ticks[0];
  const lastTick = ticks[ticks.length - 1];

  return {
    summary: {
      symbol: finalState.symbol,
      source,
      startTime: firstTick.eventTime,
      endTime: lastTick.eventTime,
      eventsProcessed: ticks.length,
      finalEquity: finalState.metrics.equity,
      netPnl: finalState.metrics.netPnl,
      returnPct: finalState.metrics.returnPct,
      maxDrawdownPct: finalState.metrics.maxDrawdownPct,
      tradeCount: finalState.metrics.tradeCount,
      winRate: finalState.metrics.winRate,
    },
    equityCurve,
    orders: finalState.orders,
    fills: finalState.fills,
    finalState,
  };
}
