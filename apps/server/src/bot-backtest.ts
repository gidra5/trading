import {
  GridTradingBot,
  PeakValleyStrategy,
  calculateRiskAdjustedMetrics,
  createInitialBotState,
  createPeakValleyBotConfig,
  type BacktestChartAnnotation,
  type BacktestChartSmaSeries,
  type BacktestResult,
  type BotSnapshot,
  type Candle,
  type EquityPoint,
  type PeakValleyBotConfig,
  type PeakValleyStrategySnapshot,
  type StrategyConfig,
  type TradeFill,
  type TradingOrder,
  type TradingOrderEvent,
  type TradingOrderSnapshot,
  type TradingTick,
  type PositionSide,
} from "@trading/bot-algo";
import { PaperTradingApi } from "./trading-api/paper-api.js";

export interface BotBacktestOptions {
  config: StrategyConfig;
  warmup?: readonly Candle[];
  maxEquityPoints?: number;
  maxChartCandles?: number;
}

interface OrderRecord {
  order: TradingOrderSnapshot;
  positionId: string;
  positionSide: PositionSide;
  entry: boolean;
}

interface PositionAccounting {
  side: PositionSide;
  asset: number;
  quote: number;
  realizedPnl: number;
}

export async function runBotBacktestFromCandles(
  candles: readonly Candle[],
  options: BotBacktestOptions,
): Promise<BacktestResult> {
  if (candles.length === 0) throw new Error("Backtest requires at least one candle.");
  const startedAt = Date.now();
  const config = options.config;
  const botConfig = createPeakValleyBotConfig(config);
  const history = (options.warmup ?? []).map(tradingCandle);
  const api = new PaperTradingApi({
    startingQuote: config.startingQuote,
    friction: (config.feeBps + config.positionRisk.marketSlippageBps) / 10_000,
    rules: marketRules(botConfig),
    getHistory: async ({ count }) => history.slice(-count),
  });
  const strategy = new PeakValleyStrategy({
    config: botConfig.strategy,
    getHistory: api.getHistory.bind(api),
  });
  const bot = new GridTradingBot({ api, strategy, config: botConfig });
  const orders = new Map<string, OrderRecord>();
  const positions = new Map<string, PositionAccounting>();
  const fills: TradeFill[] = [];
  const annotations: BacktestChartAnnotation[] = [];
  const equityCurve: EquityPoint[] = [];
  const chartCandles: Candle[] = [];
  const series = averageSeries(botConfig);
  const equityEvery = Math.max(1, Math.ceil(candles.length / (options.maxEquityPoints ?? 800)));
  const chartEvery = Math.max(1, Math.ceil(candles.length / (options.maxChartCandles ?? 2_000)));
  let peakEquity = config.startingQuote;
  let maxDrawdownPct = 0;
  let maxEffectiveLeverage = 0;
  let currentTime = candles[0].openTime;
  let realizedPnl = 0;
  let closedPositionCount = 0;
  let profitableClosedPositionCount = 0;

  await bot.warmup();
  for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index];
    for (const tick of executionTicks(candle)) {
      currentTime = tick.timestamp;
      await api.onTick(tick);
      await deliver();
    }
    const closeTick = candleTick(candle);
    currentTime = closeTick.timestamp;
    await bot.onTick(closeTick);
    await rememberOrders();
    await deliver();

    const equity = await markedEquity(candle.close);
    peakEquity = Math.max(peakEquity, equity);
    maxDrawdownPct = Math.max(maxDrawdownPct, peakEquity > 0 ? (peakEquity - equity) / peakEquity * 100 : 0);
    maxEffectiveLeverage = Math.max(maxEffectiveLeverage, effectiveLeverage(await bot.snapshot(), candle.close, equity));
    if (index % equityEvery === 0 || index === candles.length - 1) {
      equityCurve.push({ time: candle.closeTime, equity, price: candle.close });
    }
    if (index % chartEvery === 0 || index === candles.length - 1) {
      chartCandles.push({ ...candle });
      const diagnostics = strategy.getDiagnostics();
      for (const item of series) {
        const value = diagnostics.indicators[`average.${item.windowSec}`];
        if (Number.isFinite(value)) item.points.push({ time: candle.closeTime, value: value as number });
      }
    }
  }

  const last = candles.at(-1)!;
  const equity = await api.getEquity();
  const finalEquity = equity.quoteUnleveraged + equity.assetUnleveraged * last.close;
  const netPnl = finalEquity - config.startingQuote;
  const returnPct = config.startingQuote > 0 ? netPnl / config.startingQuote * 100 : 0;
  const risk = calculateRiskAdjustedMetrics(equityCurve, returnPct, maxDrawdownPct);
  const snapshots = [...orders.values()].map(({ order }) => legacyOrder(order));
  const finalState = createInitialBotState(config);
  finalState.lastPrice = last.close;
  finalState.updatedAt = last.closeTime;
  finalState.quoteFree = equity.quoteAvailable;
  finalState.quoteReserved = equity.quoteReserved;
  finalState.baseFree = equity.assetUnleveraged;
  finalState.baseReserved = equity.assetReserved;
  finalState.orders = snapshots;
  finalState.fills = fills;
  finalState.realizedPnl = realizedPnl;
  finalState.winningTrades = profitableClosedPositionCount;
  finalState.losingTrades = closedPositionCount - profitableClosedPositionCount;
  finalState.metrics = {
    ...finalState.metrics,
    equity: finalEquity,
    netPnl,
    returnPct,
    peakEquity,
    maxDrawdownPct,
    tradeCount: fills.length,
    feesPaid: fills.reduce((sum, fill) => sum + fill.feeQuote, 0),
    realizedPnl,
    winningTrades: profitableClosedPositionCount,
    losingTrades: closedPositionCount - profitableClosedPositionCount,
    winRate: closedPositionCount > 0 ? profitableClosedPositionCount / closedPositionCount * 100 : 0,
    maxEntryLeverage: botConfig.maxTargetLeverage,
    maxEffectiveLeverage,
  };

  return {
    summary: {
      symbol: config.symbol,
      source: "candles",
      startTime: candles[0].openTime,
      endTime: last.closeTime,
      eventsProcessed: candles.length * 4,
      candlesProcessed: candles.length,
      stoppedEarly: false,
      stopReason: "completed",
      durationMs: Date.now() - startedAt,
      replayDurationMs: Date.now() - startedAt,
      finalEquity,
      netPnl,
      returnPct,
      ...risk,
      maxDrawdownPct,
      maxEntryLeverage: botConfig.maxTargetLeverage,
      maxEffectiveLeverage,
      tradeCount: fills.length,
      winRate: closedPositionCount > 0 ? profitableClosedPositionCount / closedPositionCount * 100 : 0,
      closedPositionCount,
      profitableClosedPositionCount,
      profitableClosedPositionRate: closedPositionCount > 0
        ? profitableClosedPositionCount / closedPositionCount * 100
        : 0,
      liquidatedPositionCount: 0,
    },
    equityCurve,
    orders: snapshots,
    fills,
    finalState,
    candleChart: {
      candles: chartCandles,
      smaSeries: series.filter((item) => item.points.length > 0),
      annotations,
    },
  };

  async function deliver(): Promise<void> {
    for (const event of api.drainEvents()) {
      recordFill(event);
      await bot.onOrder(event);
    }
  }

  async function rememberOrders(): Promise<void> {
    const snapshot = await bot.snapshot();
    for (const position of snapshot.positions) {
      positions.set(position.id, positions.get(position.id) ?? {
        side: position.side,
        asset: 0,
        quote: 0,
        realizedPnl: 0,
      });
      for (const [grid, entry] of [[position.entryGrid, true], [position.exitGrid, false]] as const) {
        for (const item of grid?.orders ?? []) {
          orders.set(item.order.id, {
            order: structuredClone(item.order),
            positionId: position.id,
            positionSide: position.side,
            entry,
          });
        }
      }
    }
  }

  function recordFill(event: TradingOrderEvent): void {
    if (event.type !== "fill" && event.type !== "partial-fill") return;
    const record = orders.get(event.orderId);
    if (!record) return;
    const { order } = record;
    order.status = event.type === "fill" ? "filled" : "partially-filled";
    const price = event.fill.filledAsset > 0 ? event.fill.filledQuote / event.fill.filledAsset : 0;
    const feeQuote = event.fill.filledQuote * apiFriction(config);
    const fillPnl = accountFill(record, event.fill.filledAsset, event.fill.filledQuote);
    realizedPnl += fillPnl;
    fills.push({
      id: `fill-${fills.length + 1}`,
      orderId: event.orderId,
      side: order.side,
      price,
      quantity: event.fill.filledAsset,
      quoteQuantity: event.fill.filledQuote,
      feeQuote,
      realizedPnl: fillPnl,
      filledAt: currentTime,
      reason: "peak-valley",
    });
    annotations.push({
      time: fills.at(-1)!.filledAt,
      price,
      kind: order.side === "buy" ? "buy-fill" : "sell-fill",
      label: `${order.side.toUpperCase()} fill`,
      orderId: event.orderId,
      fillId: fills.at(-1)!.id,
    });
  }

  function accountFill(record: OrderRecord, asset: number, quote: number): number {
    const position = positions.get(record.positionId);
    if (!position) return 0;
    if (record.entry) {
      position.asset += asset;
      position.quote += quote;
      return 0;
    }
    const closing = Math.min(position.asset, asset);
    const fraction = position.asset > 0 ? closing / position.asset : 0;
    const basis = position.quote * fraction;
    const pnl = position.side === "long" ? quote - basis : basis - quote;
    position.asset -= closing;
    position.quote -= basis;
    position.realizedPnl += pnl;
    if (position.asset <= Number.EPSILON) {
      closedPositionCount += 1;
      if (position.realizedPnl > 0) profitableClosedPositionCount += 1;
      positions.delete(record.positionId);
    }
    return pnl;
  }

  async function markedEquity(price: number): Promise<number> {
    const current = await api.getEquity();
    return current.quoteUnleveraged + current.assetUnleveraged * price;
  }
}

function marketRules(config: PeakValleyBotConfig) {
  const quantity = { min: 0.00000001, max: null, step: 0.00000001 };
  return {
    price: { min: null, max: null, step: 0.01 },
    limitQuantity: quantity,
    marketQuantity: quantity,
    minNotional: config.minTradeQuote,
    maxNotional: config.maxTradeQuote,
    maxLeverage: config.maxTargetLeverage,
  };
}

function executionTicks(candle: Candle): TradingTick[] {
  const span = Math.max(1, candle.closeTime - candle.openTime);
  return [
    tick(candle.openTime, candle.open, candle.volume / 4),
    tick(candle.openTime + span * 0.33, candle.high, candle.volume / 4),
    tick(candle.openTime + span * 0.66, candle.low, candle.volume / 4),
    tick(candle.closeTime, candle.close, candle.volume / 4),
  ];
}

function candleTick(candle: Candle): TradingTick {
  return { ...tick(candle.closeTime, candle.close, candle.volume), candle: tradingCandle(candle) };
}

function tick(timestamp: number, price: number, quantity: number): TradingTick {
  return { timestamp, price, quantity, candle: null };
}

function tradingCandle(candle: Candle) {
  return {
    openTime: candle.openTime,
    closeTime: candle.closeTime,
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
  };
}

function averageSeries(config: PeakValleyBotConfig): BacktestChartSmaSeries[] {
  return config.strategy.averagingRangesSec.map((windowSec, index) => ({
    index,
    windowSec,
    label: `${windowSec}s ${config.strategy.movingAverageType.toUpperCase()}`,
    color: ["#38bdf8", "#f5b84b", "#a78bfa", "#22c55e"][index % 4]!,
    points: [],
  }));
}

function legacyOrder(order: TradingOrderSnapshot): TradingOrder {
  return {
    id: order.id,
    side: order.side,
    type: order.type === "stop-limit" ? "limit" : order.type,
    status: order.status === "filled" ? "filled" : order.status === "rejected" ? "cancelled" : "open",
    price: order.price ?? order.stopPrice ?? 0,
    quantity: order.size,
    filledQuantity: order.status === "filled" ? order.size : 0,
    estimatedQuoteCost: order.size * (order.price ?? order.stopPrice ?? 0),
    createdAt: 0,
    updatedAt: 0,
    reason: "peak-valley",
    realizedPnl: 0,
    feeQuote: 0,
  };
}

function effectiveLeverage(
  snapshot: BotSnapshot<PeakValleyBotConfig["strategy"], PeakValleyStrategySnapshot>,
  price: number,
  equity: number,
): number {
  const exposure = snapshot.positions.reduce((sum, position) => sum + position.asset * price, 0);
  return equity > 0 ? exposure / equity : 0;
}

function apiFriction(config: StrategyConfig): number {
  return (config.feeBps + config.positionRisk.marketSlippageBps) / 10_000;
}
