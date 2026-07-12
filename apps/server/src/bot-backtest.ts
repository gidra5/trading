import {
  GridTradingBot,
  PeakValleyStrategy,
  calculateRiskAdjustedMetrics,
  createExtremaOrderMassCollector,
  createInitialBotState,
  createPeakValleyBotConfig,
  observeExtremaOrderMassCandle,
  perfectMarginOracle,
  summarizeExtremaOrderMass,
  type BacktestExtremumTrace,
  type BacktestExtremaOrderMassSummary,
  type BacktestGridKind,
  type BacktestGridCause,
  type BacktestGridTrace,
  type BacktestOrderTrace,
  type BacktestPositionTrace,
  type BacktestSignalTrace,
  type BacktestTrace,
  type BacktestTraceFrame,
  type BacktestChartAnnotation,
  type BacktestChartSmaSeries,
  type BacktestResult,
  type BotSnapshot,
  type Candle,
  type EquityPoint,
  type PeakValleyBotConfig,
  type PeakValleyStrategyConfig,
  type PeakValleyStrategySnapshot,
  type StrategyOptions,
  type StrategyConfig,
  type TradeFill,
  type TradingOrder,
  type TradingOrderEvent,
  type TradingOrderSnapshot,
  type TradingTick,
  type TradingPosition,
  type TradingStrategyEntrySignal,
  type TradingStrategyExitSignal,
  type PositionSide,
} from "@trading/bot-algo";
import { PaperTradingApi } from "./trading-api/paper-api.js";

export interface BotBacktestOptions {
  config: StrategyConfig;
  warmup?: readonly Candle[];
  maxEquityPoints?: number;
  maxChartCandles?: number;
  extremaSmaWindowMs?: number;
}

interface OrderRecord {
  order: TradingOrderSnapshot;
  positionId: string;
  positionSide: PositionSide;
  entry: boolean;
  trace: BacktestOrderTrace;
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
  const botConfig = createPeakValleyBotConfig(config, candleIntervalMs(candles));
  const history = (options.warmup ?? []).map(tradingCandle);
  const api = new PaperTradingApi({
    startingQuote: config.startingQuote,
    friction: (config.feeBps + config.positionRisk.marketSlippageBps) / 10_000,
    rules: marketRules(botConfig),
    getHistory: async ({ count }) => history.slice(-count),
  });
  const strategy = new TracingPeakValleyStrategy({
    config: botConfig.strategy,
    getHistory: api.getHistory.bind(api),
  });
  const bot = new GridTradingBot({ api, strategy, config: botConfig });
  const orders = new Map<string, OrderRecord>();
  const positions = new Map<string, PositionAccounting>();
  const fills: TradeFill[] = [];
  const annotations: BacktestChartAnnotation[] = [];
  const positionTraces = new Map<string, BacktestPositionTrace>();
  const gridTraces = new Map<string, BacktestGridTrace>();
  const activeGrids = new Map<string, BacktestGridTrace>();
  const signals: BacktestSignalTrace[] = [];
  const frames: BacktestTraceFrame[] = [];
  const extremaCollector = createExtremaOrderMassCollector({ smaWindowMs: options.extremaSmaWindowMs });
  const equityCurve: EquityPoint[] = [];
  const chartCandles: Candle[] = [];
  const series = averageSeries(botConfig);
  const equityEvery = Math.max(1, Math.ceil(candles.length / (options.maxEquityPoints ?? 800)));
  const chartEvery = Math.max(1, Math.ceil(candles.length / (options.maxChartCandles ?? 2_000)));
  let peakEquity = config.startingQuote;
  let maxDrawdownPct = 0;
  let maxEffectiveLeverage = 0;
  let currentTime = candles[0].openTime;
  let captureCause: "tick" | "fill" = "tick";
  let realizedPnl = 0;
  let closedPositionCount = 0;
  let profitableClosedPositionCount = 0;
  let latestDecision: BacktestSignalTrace | null = null;

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
    await capturePositions();
    await deliver();
    captureSignal((index + 1) % chartEvery === 0 || index === candles.length - 1);
    observeExtremaOrderMassCandle(extremaCollector, candle);

    const account = await api.getEquity();
    const equity = account.quoteUnleveraged + account.assetUnleveraged * candle.close;
    const botSnapshot = await bot.snapshot();
    peakEquity = Math.max(peakEquity, equity);
    maxDrawdownPct = Math.max(maxDrawdownPct, peakEquity > 0 ? (peakEquity - equity) / peakEquity * 100 : 0);
    maxEffectiveLeverage = Math.max(maxEffectiveLeverage, effectiveLeverage(botSnapshot, candle.close, equity));
    if (index % equityEvery === 0 || index === candles.length - 1) {
      equityCurve.push({ time: candle.closeTime, equity, price: candle.close });
    }
    if (index % chartEvery === 0) {
      chartCandles.push({ ...candle });
    } else {
      mergeChartCandle(chartCandles.at(-1)!, candle);
    }
    if ((index + 1) % chartEvery === 0 || index === candles.length - 1) {
      frames.push(traceFrame(candle.closeTime, candle.close, account, botSnapshot, equity));
      const diagnostics = strategy.getDiagnostics();
      for (const item of series) {
        const value = diagnostics.indicators[item.index < 0 ? "kama" : `average.${item.windowSec}`];
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
  const oracle = perfectMarginOracle(candles, {
    startingQuote: config.startingQuote,
    leverage: config.maxLeverage,
    friction: apiFriction(config),
    eventMode: "close",
    maxPathCandles: options.maxChartCandles ?? 2_000,
  });
  const snapshots = [...orders.values()].map(({ order, trace }) => legacyOrder(order, trace.createdAt));
  const extremaOrderMass = summarizeExtremaOrderMass(extremaCollector, fills);
  const trace: BacktestTrace = {
    positions: [...positionTraces.values()],
    grids: [...gridTraces.values()],
    orders: [...orders.values()].map(({ trace }) => trace),
    signals,
    extrema: buildExtremaTrace(extremaCollector, [...orders.values()].map(({ trace }) => trace), extremaOrderMass),
    oracle: oracle.path,
    frames,
  };
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
      perfectMarginLeverage: oracle.leverage,
      perfectMarginFinalEquity: oracle.finalEquity,
      perfectMarginNetPnl: oracle.netPnl,
      perfectMarginReturnPct: oracle.returnPct,
      perfectMarginCapturePct: oracle.netPnl > 0 ? netPnl / oracle.netPnl * 100 : undefined,
      perfectMarginCompoundedFinalEquity: oracle.compoundedFinalEquity,
      perfectMarginCompoundedNetPnl: oracle.compoundedNetPnl,
      perfectMarginCompoundedReturnPct: oracle.compoundedReturnPct,
      perfectMarginCompoundedCapturePct: oracle.compoundedNetPnl > 0
        ? netPnl / oracle.compoundedNetPnl * 100
        : undefined,
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
      extremaOrderMass,
    },
    equityCurve,
    orders: snapshots,
    fills,
    finalState,
    candleChart: {
      candles: chartCandles,
      smaSeries: series.filter((item) => item.points.length > 0),
      annotations,
      trace,
    },
  };

  async function deliver(): Promise<void> {
    for (const event of api.drainEvents()) {
      recordFill(event);
      await bot.onOrder(event);
      captureCause = "fill";
      await capturePositions();
      captureCause = "tick";
    }
  }

  async function capturePositions(): Promise<void> {
    const snapshot = await bot.snapshot();
    const currentIds = new Set(snapshot.positions.map((position) => position.id));
    const liveOrderIds = new Set<string>();
    for (const position of snapshot.positions) {
      positions.set(position.id, positions.get(position.id) ?? {
        side: position.side,
        asset: 0,
        quote: 0,
        realizedPnl: 0,
      });
      for (const [grid, entry] of [[position.entryGrid, true], [position.exitGrid, false]] as const) {
        const gridTrace = grid ? observeGrid(position, entry ? "entry" : "exit", grid) : undefined;
        for (const item of grid?.orders ?? []) {
          liveOrderIds.add(item.order.id);
          const existing = orders.get(item.order.id);
          if (existing) {
            existing.order = structuredClone(item.order);
            continue;
          }
          const trace: BacktestOrderTrace = {
            id: item.order.id,
            positionId: position.id,
            positionSide: position.side,
            gridId: gridTrace!.id,
            grid: entry ? "entry" : "exit",
            side: item.order.side,
            type: item.order.type,
            size: item.order.size,
            price: item.order.price,
            stopPrice: item.order.stopPrice,
            createdAt: currentTime,
            endedAt: item.order.status === "filled" || item.order.status === "rejected" ? currentTime : null,
            outcome: item.order.status === "filled"
              ? "filled"
              : item.order.status === "rejected"
                ? "rejected"
                : "open",
            fills: [],
          };
          orders.set(item.order.id, {
            order: structuredClone(item.order),
            positionId: position.id,
            positionSide: position.side,
            entry,
            trace,
          });
          annotations.push({
            time: currentTime,
            price: item.order.price ?? item.order.stopPrice ?? grid!.creationPrice,
            kind: item.order.side === "buy" ? "buy-order" : "sell-order",
            label: `${entry ? "Entry" : "Exit"} ${item.order.type}`,
            orderId: item.order.id,
            targetPositionId: position.id,
            gridId: gridTrace!.id,
            gridKind: entry ? "entry" : "exit",
          });
        }
      }
      observePosition(position, snapshot.positions);
    }
    for (const trace of positionTraces.values()) {
      if (trace.closedAt === null && !currentIds.has(trace.id)) trace.closedAt = currentTime;
    }
    for (const record of orders.values()) {
      if (record.trace.outcome === "open" && !liveOrderIds.has(record.order.id)) {
        record.trace.outcome = "withdrawn";
        record.trace.endedAt = currentTime;
      }
    }
  }

  function observeGrid(
    position: TradingPosition,
    kind: BacktestGridKind,
    grid: NonNullable<TradingPosition["entryGrid"]>,
  ): BacktestGridTrace {
    const key = `${position.id}:${kind}`;
    const orderIds = grid.orders.map(({ order }) => order.id);
    let trace = activeGrids.get(key);
    if (!trace || (trace.orderIds.length > 0 && !trace.orderIds.some((id) => orderIds.includes(id)))) {
      trace = {
        id: `${key}:${gridTraces.size + 1}`,
        positionId: position.id,
        kind,
        cause: gridCause(position, kind),
        createdAt: currentTime,
        creationPrice: grid.creationPrice,
        orderIds: [],
      };
      activeGrids.set(key, trace);
      gridTraces.set(trace.id, trace);
    }
    trace.orderIds = [...new Set([...trace.orderIds, ...orderIds])];
    return trace;
  }

  function gridCause(position: TradingPosition, kind: BacktestGridKind): BacktestGridCause {
    if (kind === "entry") return "strategy-entry";
    if (captureCause === "fill") return "fill-reset";
    if (strategy.hasExitDecision(position.side)) return "strategy-exit";
    if (position.stopLossPrice !== null && (
      position.side === "long" ? currentPrice() <= position.stopLossPrice : currentPrice() >= position.stopLossPrice
    )) return "stop-loss";
    if (position.takeProfitPrice !== null && (
      position.side === "long" ? currentPrice() >= position.takeProfitPrice : currentPrice() <= position.takeProfitPrice
    )) return "take-profit";
    if (position.expiresAt !== null && currentTime >= position.expiresAt) return "expiry";
    return "price-reset";
  }

  function currentPrice(): number {
    return strategy.currentPrice();
  }

  function observePosition(position: TradingPosition, all: TradingPosition[]): void {
    const entryGridId = position.entryGrid
      ? observeGrid(position, "entry", position.entryGrid).id
      : null;
    const exitGridId = position.exitGrid
      ? observeGrid(position, "exit", position.exitGrid).id
      : null;
    const lentTo = all.flatMap((borrower) => borrower.internalBorrow
      .filter((borrow) => borrow.positionId === position.id)
      .map((borrow) => ({ positionId: borrower.id, asset: borrow.asset, quote: borrow.quote })));
    let trace = positionTraces.get(position.id);
    if (!trace) {
      trace = {
        id: position.id,
        side: position.side,
        leverage: position.leverage,
        createdAt: currentTime,
        openedAt: position.asset > Number.EPSILON ? currentTime : null,
        closedAt: null,
        entryOrderIds: [],
        exitOrderIds: [],
        states: [],
      };
      positionTraces.set(position.id, trace);
    }
    if (trace.openedAt === null && position.asset > Number.EPSILON) trace.openedAt = currentTime;
    trace.entryOrderIds = [...new Set([
      ...trace.entryOrderIds,
      ...(position.entryGrid?.orders.map(({ order }) => order.id) ?? []),
    ])];
    trace.exitOrderIds = [...new Set([
      ...trace.exitOrderIds,
      ...(position.exitGrid?.orders.map(({ order }) => order.id) ?? []),
    ])];
    const state = {
      time: currentTime,
      asset: position.asset,
      quote: position.quote,
      externalBorrow: structuredClone(position.externalBorrow),
      internalBorrow: structuredClone(position.internalBorrow),
      lentTo,
      entryGridId,
      exitGridId,
    };
    if (positionStateKey(trace.states.at(-1)) !== positionStateKey(state)) trace.states.push(state);
  }

  function captureSignal(includeConfirmations: boolean): void {
    const decision = strategy.takeDecision(botConfig.strategy.derivativeSource, includeConfirmations);
    if (!decision) return;
    latestDecision = decision;
    signals.push(decision);
    for (const active of decision.active) {
      const buy = active.side === "long" === (active.type === "entry");
      annotations.push({
        time: decision.time,
        price: decision.price,
        kind: buy ? "buy-signal" : "sell-signal",
        label: `${active.side} ${active.type}`,
        reason: `${decision.source}; ${decision.gates.filter((gate) => gate.passed).map((gate) => gate.code).join(", ")}`,
      });
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
    record.trace.fills.push({
      id: fills.at(-1)!.id,
      time: currentTime,
      price,
      asset: event.fill.filledAsset,
      quote: event.fill.filledQuote,
      feeQuote,
      remaining: event.fill.remaining,
    });
    if (event.type === "fill") {
      record.trace.outcome = "filled";
      record.trace.endedAt = currentTime;
    }
    annotations.push({
      time: fills.at(-1)!.filledAt,
      price,
      kind: order.side === "buy" ? "buy-fill" : "sell-fill",
      label: `${order.side.toUpperCase()} fill`,
      orderId: event.orderId,
      fillId: fills.at(-1)!.id,
      targetPositionId: record.positionId,
      gridId: record.trace.gridId,
      gridKind: record.trace.grid,
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

  function traceFrame(
    time: number,
    price: number,
    account: Awaited<ReturnType<PaperTradingApi["getEquity"]>>,
    snapshot: BotSnapshot<PeakValleyBotConfig["strategy"], PeakValleyStrategySnapshot>,
    equity: number,
  ): BacktestTraceFrame {
    const longQuantity = snapshot.positions.reduce(
      (sum, position) => sum + (position.side === "long" ? position.asset : 0),
      0,
    );
    const shortQuantity = snapshot.positions.reduce(
      (sum, position) => sum + (position.side === "short" ? position.asset : 0),
      0,
    );
    const longExposureQuote = longQuantity * price;
    const shortExposureQuote = shortQuantity * price;
    const grossExposureQuote = longExposureQuote + shortExposureQuote;
    const currentNetPnl = equity - config.startingQuote;
    const openOrders = snapshot.positions.flatMap((position) => [
      ...(position.entryGrid?.orders ?? []),
      ...(position.exitGrid?.orders ?? []),
    ]).filter(({ order }) => order.status !== "filled" && order.status !== "rejected");
    const pendingQuote = (side: PositionSide) => snapshot.positions.reduce((total, position) =>
      total + (position.side === side
        ? (position.entryGrid?.orders ?? []).reduce((sum, { order }) =>
            sum + (order.status === "filled" || order.status === "rejected"
              ? 0
              : order.size * (order.price ?? order.stopPrice ?? price)), 0)
        : 0), 0);
    const signal = (type: "entry" | "exit") => {
      const active = latestDecision?.time === time
        ? latestDecision.active.find((item) => item.type === type)
        : undefined;
      if (!active) return undefined;
      return active.side === "long" === (type === "entry") ? "buy" as const : "sell" as const;
    };
    return {
      time,
      price,
      metrics: {
        equity,
        netPnl: currentNetPnl,
        returnPct: config.startingQuote > 0 ? currentNetPnl / config.startingQuote * 100 : 0,
        realizedPnl,
        unrealizedPnl: currentNetPnl - realizedPnl,
        maxDrawdownPct,
        exposurePct: equity > 0 ? grossExposureQuote / equity * 100 : 0,
        maxEffectiveLeverage,
        feesPaid: fills.reduce((sum, fill) => sum + fill.feeQuote, 0),
        tradeCount: fills.length,
        winRate: closedPositionCount > 0 ? profitableClosedPositionCount / closedPositionCount * 100 : 0,
      },
      quoteFree: account.quoteAvailable,
      quoteReserved: account.quoteReserved,
      baseFree: account.assetAvailable,
      baseReserved: account.assetReserved,
      openOrderCount: openOrders.length,
      longLotCount: snapshot.positions.filter((position) => position.side === "long").length,
      shortLotCount: snapshot.positions.filter((position) => position.side === "short").length,
      entrySignal: signal("entry"),
      exitSignal: signal("exit"),
      positions: {
        summary: {
          longQuantity,
          shortQuantity,
          netExposureQuote: longExposureQuote - shortExposureQuote,
          grossExposureQuote,
          effectiveLeverage: equity > 0 ? grossExposureQuote / equity : 0,
          longExposureQuote,
          shortExposureQuote,
          pendingLongQuote: pendingQuote("long"),
          pendingShortQuote: pendingQuote("short"),
        },
      },
    };
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

function candleIntervalMs(candles: readonly Candle[]): number {
  const spans = candles.slice(0, 32)
    .map((candle) => candle.closeTime - candle.openTime + 1)
    .filter((span) => Number.isFinite(span) && span > 0)
    .sort((left, right) => left - right);
  return Math.max(1, Math.round(spans[Math.floor(spans.length / 2)] ?? 60_000));
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
  const averages = config.strategy.averagingRangesSec.map((windowSec, index) => ({
    index,
    windowSec,
    label: `${windowSec}s ${config.strategy.movingAverageType.toUpperCase()}`,
    color: ["#38bdf8", "#f5b84b", "#a78bfa", "#22c55e"][index % 4]!,
    points: [],
  }));
  return config.strategy.derivativeSource === "kama"
    ? [{ index: -1, windowSec: 0, label: "Volume KAMA", color: "#f472b6", points: [] }, ...averages]
    : averages;
}

function legacyOrder(order: TradingOrderSnapshot, createdAt: number): TradingOrder {
  return {
    id: order.id,
    side: order.side,
    type: order.type === "stop-limit" ? "limit" : order.type,
    status: order.status === "filled" ? "filled" : order.status === "rejected" ? "cancelled" : "open",
    price: order.price ?? order.stopPrice ?? 0,
    quantity: order.size,
    filledQuantity: order.status === "filled" ? order.size : 0,
    estimatedQuoteCost: order.size * (order.price ?? order.stopPrice ?? 0),
    createdAt,
    updatedAt: createdAt,
    reason: "peak-valley",
    realizedPnl: 0,
    feeQuote: 0,
  };
}

class TracingPeakValleyStrategy extends PeakValleyStrategy {
  private tick: TradingTick | null = null;
  private entry: TradingStrategyEntrySignal | null = null;
  private exit: TradingStrategyExitSignal | null = null;

  constructor(options: StrategyOptions<PeakValleyStrategyConfig>) {
    super(options);
  }

  override async onTick(tick: TradingTick): Promise<void> {
    this.tick = tick;
    this.entry = null;
    this.exit = null;
    await super.onTick(tick);
  }

  override async entrySignal(): Promise<TradingStrategyEntrySignal | null> {
    return this.entry = await super.entrySignal();
  }

  override async exitSignal(): Promise<TradingStrategyExitSignal | null> {
    return this.exit = await super.exitSignal();
  }

  takeDecision(
    source: "price" | "kama" = "price",
    includeConfirmations = false,
  ): BacktestSignalTrace | null {
    if (!this.tick) return null;
    const diagnostics = this.getDiagnostics();
    const active = [
      ...(this.entry ? [{ type: "entry" as const, side: this.entry.side }] : []),
      ...(this.exit ? [{ type: "exit" as const, side: this.exit.side }] : []),
    ];
    const confirmationActive = diagnostics.gates.some((gate) =>
      gate.passed && (gate.code.includes(".confirmation.") || gate.code.includes(".source.")));
    if (active.length === 0 && (!includeConfirmations || !confirmationActive)) return null;
    return {
      time: this.tick.timestamp,
      price: this.tick.price,
      source,
      active,
      gates: structuredClone(diagnostics.gates),
      blockers: [...diagnostics.blockers],
      indicators: structuredClone(diagnostics.indicators),
    };
  }

  hasExitDecision(side: PositionSide): boolean {
    return this.exit?.side === side;
  }

  currentPrice(): number {
    return this.tick?.price ?? 0;
  }
}

function mergeChartCandle(target: Candle, candle: Candle): void {
  target.closeTime = candle.closeTime;
  target.high = Math.max(target.high, candle.high);
  target.low = Math.min(target.low, candle.low);
  target.close = candle.close;
  target.volume += candle.volume;
}

function positionStateKey(value: unknown): string {
  return value === undefined ? "" : JSON.stringify({ ...(value as object), time: 0 });
}

function buildExtremaTrace(
  collector: ReturnType<typeof createExtremaOrderMassCollector>,
  orders: BacktestOrderTrace[],
  summary: BacktestExtremaOrderMassSummary,
): BacktestExtremumTrace[] {
  const traces = [
    ...collector.peaks.map((point, index) => extremum(point, `peak-${index + 1}`, summary.sell)),
    ...collector.valleys.map((point, index) => extremum(point, `valley-${index + 1}`, summary.buy)),
  ].sort((left, right) => left.time - right.time);
  const byKind = {
    peak: traces.filter((item) => item.kind === "peak"),
    valley: traces.filter((item) => item.kind === "valley"),
  };
  for (const order of orders) {
    const extrema = byKind[order.side === "buy" ? "valley" : "peak"];
    for (const fill of order.fills) {
      const target = nearestExtremum(extrema, fill.time);
      if (!target) continue;
      const timeErrorMs = fill.time - target.time;
      const priceErrorPct = target.price > 0 ? (fill.price - target.price) / target.price * 100 : 0;
      target.orders.push({
        orderId: order.id,
        fillId: fill.id,
        positionId: order.positionId,
        gridId: order.gridId,
        grid: order.grid,
        time: fill.time,
        price: fill.price,
        asset: fill.asset,
        quote: fill.quote,
        timeErrorMs,
        priceErrorPct,
        withinThreshold: Math.abs(timeErrorMs) <= target.thresholdTimeMs
          && Math.abs(priceErrorPct) <= target.thresholdPriceDistancePct,
      });
    }
  }
  for (const trace of traces) {
    if (trace.orders.length === 0) continue;
    trace.errorBox = {
      minTimeErrorMs: Math.min(0, ...trace.orders.map((order) => order.timeErrorMs)),
      maxTimeErrorMs: Math.max(0, ...trace.orders.map((order) => order.timeErrorMs)),
      minPriceErrorPct: Math.min(0, ...trace.orders.map((order) => order.priceErrorPct)),
      maxPriceErrorPct: Math.max(0, ...trace.orders.map((order) => order.priceErrorPct)),
      quote: trace.orders.reduce((sum, order) => sum + order.quote, 0),
      asset: trace.orders.reduce((sum, order) => sum + order.asset, 0),
      withinThresholdQuote: trace.orders.reduce(
        (sum, order) => sum + (order.withinThreshold ? order.quote : 0),
        0,
      ),
    };
  }
  return traces;

  function extremum(
    point: { time: number; price: number; kind: "peak" | "valley" },
    id: string,
    side: BacktestExtremaOrderMassSummary["buy"],
  ): BacktestExtremumTrace {
    return {
      id,
      kind: point.kind,
      time: point.time,
      price: point.price,
      smaWindowMs: collector.smaWindowMs,
      thresholdTimeMs: collector.thresholdTimeMs,
      thresholdPriceDistancePct: collector.thresholdPriceDistancePct,
      p99TimeDistanceMs: side.massP99JointTimeDistanceMs ?? collector.thresholdTimeMs,
      p99PriceDistancePct: side.massP99JointPriceDistancePct ?? collector.thresholdPriceDistancePct,
      orders: [],
      errorBox: null,
    };
  }
}

function nearestExtremum(
  extrema: BacktestExtremumTrace[],
  time: number,
): BacktestExtremumTrace | undefined {
  let nearest: BacktestExtremumTrace | undefined;
  let distance = Infinity;
  for (const item of extrema) {
    const next = Math.abs(item.time - time);
    if (next < distance) {
      nearest = item;
      distance = next;
    }
  }
  return nearest;
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
