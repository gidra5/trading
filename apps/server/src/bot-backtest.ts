import {
  GridTradingBot,
  PeakValleyStrategy,
  SimulatedTradingApi,
  DEFAULT_SIMULATED_BORROW_BPS_HOUR,
  DEFAULT_SIMULATED_MAX_EFFECTIVE_LEVERAGE,
  calculateRiskAdjustedMetrics,
  conditionalExposureProbabilities,
  createExtremaOrderMassCollector,
  createInitialBotState,
  createPeakValleyBotConfig,
  exposureValueOracleActionDistribution,
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
  type BacktestStrategy,
  type BotSnapshot,
  type Candle,
  type EquityPoint,
  type ExposureValueOracleActionDistribution,
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
  type TradingStrategyTargetExposureContext,
  type TradingStrategyTargetExposureSignal,
  type PositionSide,
} from "@trading/bot-algo";

export interface BotBacktestOptions {
  config: StrategyConfig;
  strategy?: BacktestStrategy;
  warmup?: readonly Candle[];
  oracleFuture?: readonly Candle[];
  maxEquityPoints?: number;
  maxChartCandles?: number;
  extremaSmaWindowMs?: number;
  summaryOnly?: boolean;
  /** Optional precomputed base oracle rows, used by long historical suite runs. */
  hindsightOracleDistributionAt?: (
    timestamp: number,
  ) => ExposureValueOracleActionDistribution | null;
  /** Causal learned-policy rows. Unlike the hindsight provider, this must never read future candles. */
  learnedOracleDistributionAt?: (
    timestamp: number,
  ) => ExposureValueOracleActionDistribution | null;
  /** Hard leverage cap for the imperfect causal policy; hindsight keeps its oracle cap. */
  learnedOracleMaximumLeverage?: number;
  /** Optional hindsight cap for apples-to-apples learned/oracle comparisons. */
  hindsightOracleMaximumLeverage?: number;
  /** Receives every policy-minute oracle decision with the simulator's actual marked exposure. */
  onOracleDecision?: (decision: OracleBacktestDecision) => void;
  /** Scales modal target exposure by confidence^power. Zero preserves raw modal exposure. */
  hindsightOracleConfidenceExposurePower?: number;
  /** Minimum fraction of max leverage available to the confidence-conditioned ceiling. */
  /** Bot-level leverage floor for confidence-conditioned target exposure. */
  oracleConfidenceLeverageFloor?: number;
  /** Calibrated predictor quality multiplied into each distribution-derived confidence. */
  oracleStaticConfidenceScale?: number;
  /** Minimum distribution confidence required for an expansion at full predictor quality. */
  oracleMinimumDistributionConfidence?: number;
  /** Expansion confidence threshold approached as static predictor quality falls to zero. */
  oracleMaximumDistributionConfidenceThreshold?: number;
  /** Total same-side distribution-confidence mass required before expansion. */
  oracleExpansionConfirmationMass?: number;
  /** Base expansion-delta cap as a fraction of max leverage, before static-confidence squared. */
  oracleExpansionDeltaCapFraction?: number;
  onProgress?: (progress: {
    candlesProcessed: number;
    totalCandles: number;
    elapsedMs: number;
  }) => void;
}

export interface OracleBacktestDecision {
  timestamp: number;
  currentExposure: number;
  /** Modal exposure before transition costs are conditioned on current exposure. */
  rawModalExposure: number;
  /** Modal exposure after transition-cost conditioning, before leverage/confidence limits. */
  conditionedModalExposure: number;
  /** Strategy-requested exposure before bot-level confidence and transition limits. */
  targetExposure: number;
  /** Confidence of this distribution-derived decision. */
  confidence: number;
  /** Predictor/model reliability exposed separately by the strategy. */
  staticConfidence: number;
  /** Combined confidence retained for diagnostics. */
  effectiveConfidence: number;
  entropy: number;
  /** The strategy returned a signal; this does not imply that an order filled. */
  signalEmitted: boolean;
}

export const HINDSIGHT_ORACLE_INTERVAL_MS = 1_000;
export const HINDSIGHT_ORACLE_HOLDING_PERIOD_MS = 60_000;
export const HINDSIGHT_ORACLE_DECISION_DELAY_MS = 60_000;
export const HINDSIGHT_ORACLE_VALUE_HORIZON_MS = 60 * 60_000;
export const HINDSIGHT_ORACLE_MAX_EXPOSURE = 100;
export const HINDSIGHT_ORACLE_GRID_SIZE = 255;
export const HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE = 250;
export const HINDSIGHT_ORACLE_TEMPERATURE = 0.01;
export const HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER = 0;
export const HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR = 0.75;
export const ORACLE_DEFAULT_STATIC_CONFIDENCE_SCALE = 1;
export const ORACLE_MINIMUM_DISTRIBUTION_CONFIDENCE = 0.05;
export const ORACLE_MAXIMUM_DISTRIBUTION_CONFIDENCE_THRESHOLD = 0.5;
export const HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR = 10;
export const LEARNED_ORACLE_DEFAULT_MAXIMUM_LEVERAGE = 1;

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
  const summaryOnly = options.summaryOnly ?? false;
  const backtestStrategy = options.strategy ?? "peak-valley";
  const intervalMs = candleIntervalMs(candles);
  const oracleStrategy = backtestStrategy === "hindsight-oracle-1s"
    || backtestStrategy === "learned-oracle-1s"
    || backtestStrategy === "learned-oracle-1m";
  const learnedMaximumLeverage = options.learnedOracleMaximumLeverage
    ?? LEARNED_ORACLE_DEFAULT_MAXIMUM_LEVERAGE;
  const hindsightMaximumLeverage = options.hindsightOracleMaximumLeverage
    ?? HINDSIGHT_ORACLE_MAX_EXPOSURE;
  if (!(learnedMaximumLeverage > 0) || !Number.isFinite(learnedMaximumLeverage)) {
    throw new Error("Learned oracle maximum leverage must be finite and positive.");
  }
  if (!(hindsightMaximumLeverage > 0) || !Number.isFinite(hindsightMaximumLeverage)) {
    throw new Error("Hindsight oracle maximum leverage must be finite and positive.");
  }
  const oracleTargetMaximumLeverage = backtestStrategy === "learned-oracle-1s"
    || backtestStrategy === "learned-oracle-1m"
    ? Math.min(HINDSIGHT_ORACLE_MAX_EXPOSURE, learnedMaximumLeverage)
    : Math.min(HINDSIGHT_ORACLE_MAX_EXPOSURE, hindsightMaximumLeverage);
  const minimumDistributionConfidence = options.oracleMinimumDistributionConfidence
    ?? ORACLE_MINIMUM_DISTRIBUTION_CONFIDENCE;
  const maximumDistributionConfidenceThreshold =
    options.oracleMaximumDistributionConfidenceThreshold
    ?? ORACLE_MAXIMUM_DISTRIBUTION_CONFIDENCE_THRESHOLD;
  if (
    !Number.isFinite(minimumDistributionConfidence)
    || !Number.isFinite(maximumDistributionConfidenceThreshold)
    || minimumDistributionConfidence < 0
    || maximumDistributionConfidenceThreshold > 1
    || maximumDistributionConfidenceThreshold < minimumDistributionConfidence
  ) {
    throw new Error(
      "Oracle distribution-confidence thresholds must satisfy 0 <= minimum <= maximum <= 1.",
    );
  }
  const confidenceLeverageFloor = options.oracleConfidenceLeverageFloor
    ?? HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR;
  if (
    !(confidenceLeverageFloor >= 0 && confidenceLeverageFloor <= 1)
    || !Number.isFinite(confidenceLeverageFloor)
  ) {
    throw new Error("Oracle confidence leverage floor must be in [0, 1].");
  }
  if (backtestStrategy === "learned-oracle-1s" && intervalMs !== HINDSIGHT_ORACLE_INTERVAL_MS) {
    throw new Error("The learned one-second oracle strategy requires one-second candles.");
  }
  if (backtestStrategy === "learned-oracle-1m" && intervalMs !== 60_000) {
    throw new Error("The learned one-minute oracle strategy requires one-minute candles.");
  }
  if (
    backtestStrategy === "hindsight-oracle-1s"
    && (
      intervalMs > HINDSIGHT_ORACLE_HOLDING_PERIOD_MS
      || HINDSIGHT_ORACLE_HOLDING_PERIOD_MS % intervalMs !== 0
      || HINDSIGHT_ORACLE_DECISION_DELAY_MS % intervalMs !== 0
      || HINDSIGHT_ORACLE_VALUE_HORIZON_MS % intervalMs !== 0
    )
  ) {
    throw new Error(
      "The hindsight oracle candle interval must evenly divide its holding period, decision delay, and value horizon.",
    );
  }
  const baseBotConfig = createPeakValleyBotConfig(config, intervalMs);
  const botConfig = oracleStrategy
    ? {
        ...baseBotConfig,
        maxTargetLeverage: oracleTargetMaximumLeverage,
        // The exposure target is the risk cap; a second fixed quote cap would
        // prevent 100x targets after account equity changes.
        maxTradeQuote: Number.POSITIVE_INFINITY,
        cooldownMs: HINDSIGHT_ORACLE_HOLDING_PERIOD_MS,
        exposureControl: {
          confidenceLeverageFloor,
          minimumSignalConfidence: minimumDistributionConfidence,
          maximumSignalConfidenceThreshold: maximumDistributionConfidenceThreshold,
          expansionConfirmationMass: options.oracleExpansionConfirmationMass ?? 1,
          expansionDeltaCapFraction: options.oracleExpansionDeltaCapFraction
            ?? HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
        },
      }
    : baseBotConfig;
  const history = (options.warmup ?? []).map(tradingCandle);
  const api = new SimulatedTradingApi({
    startingQuote: config.startingQuote,
    friction: (config.feeBps + config.positionRisk.marketSlippageBps) / 10_000,
    rules: marketRules(botConfig),
    getHistory: async ({ count }) => history.slice(-count),
    quoteBorrowBpsHour: oracleStrategy
      ? HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR
      : DEFAULT_SIMULATED_BORROW_BPS_HOUR,
    assetBorrowBpsHour: oracleStrategy
      ? HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR
      : DEFAULT_SIMULATED_BORROW_BPS_HOUR,
    maxEffectiveLeverage: oracleStrategy
      ? oracleMaximumEffectiveLeverage(oracleTargetMaximumLeverage)
      : DEFAULT_SIMULATED_MAX_EFFECTIVE_LEVERAGE,
  });
  const oracleDistributionAt = backtestStrategy === "hindsight-oracle-1s"
    ? options.hindsightOracleDistributionAt ?? createHindsightOracleDistributionProvider(
      candles,
      options.oracleFuture ?? [],
      config,
      intervalMs,
    )
    : backtestStrategy === "learned-oracle-1s"
      || backtestStrategy === "learned-oracle-1m"
      ? options.learnedOracleDistributionAt
      : undefined;
  if (
    (backtestStrategy === "learned-oracle-1s" || backtestStrategy === "learned-oracle-1m")
    && !oracleDistributionAt
  ) {
    throw new Error("The learned oracle strategy requires causal model distributions.");
  }
  const confidenceExposurePower = options.hindsightOracleConfidenceExposurePower
    ?? HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER;
  if (!(confidenceExposurePower >= 0) || !Number.isFinite(confidenceExposurePower)) {
    throw new Error("Hindsight oracle confidence exposure power must be finite and non-negative.");
  }
  const staticConfidenceScale = options.oracleStaticConfidenceScale
    ?? ORACLE_DEFAULT_STATIC_CONFIDENCE_SCALE;
  validateOracleStaticConfidenceScale(staticConfidenceScale);
  const strategyOptions = {
    config: botConfig.strategy,
    getHistory: api.getHistory.bind(api),
  };
  const strategy = oracleDistributionAt
    ? new TracingOraclePolicyStrategy(
        strategyOptions,
        oracleDistributionAt,
        apiFriction(config),
        confidenceExposurePower,
        staticConfidenceScale,
        options.onOracleDecision,
      )
    : new TracingPeakValleyStrategy(strategyOptions);
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
  const progressEvery = Math.max(1, Math.ceil(candles.length / 100));
  let peakEquity = config.startingQuote;
  let maxInitialBalanceDrawdownPct = 0;
  let maxDrawdownPct = 0;
  let maxEffectiveLeverage = 0;
  let currentTime = candles[0].openTime;
  let captureCause: "tick" | "fill" = "tick";
  let realizedPnl = 0;
  let closedPositionCount = 0;
  let profitableClosedPositionCount = 0;
  let summaryFillCount = 0;
  let processedCandles = 0;
  let processedMarketEvents = 0;
  let lastProcessed = candles[0]!;
  let latestDecision: BacktestSignalTrace | null = null;
  let latestBotSnapshot:
    BotSnapshot<PeakValleyBotConfig["strategy"], PeakValleyStrategySnapshot> | undefined;

  await bot.warmup();
  replay: for (let index = 0; index < candles.length; index += 1) {
    const candle = candles[index];
    processedCandles += 1;
    lastProcessed = candle;
    const balancesBefore = api.getUnleveragedBalances();
    const replayTicks = api.hasOpenOrders() || api.hasExposure()
      ? executionTicks(candle, balancesBefore.asset)
      : [candleTick(candle)];
    for (const tick of replayTicks) {
      currentTime = tick.timestamp;
      await api.onTick(tick);
      processedMarketEvents += 1;
      observeRisk(api.status().equity, api.status().effectiveLeverage);
      if (api.isLiquidated()) break;
    }
    // Apply all fills from the candle as one atomic interval. Orders created in
    // response to an intrabar fill cannot use a later, unknowable OHLC extreme.
    await deliver();
    const closeTick = candleTick(candle);
    if (!api.isLiquidated()) {
      currentTime = closeTick.timestamp;
      await bot.onTick(closeTick);
      if (!summaryOnly) await capturePositions();
      await deliver();
      observeRisk(api.status().equity, api.status().effectiveLeverage);
    }
    if (!summaryOnly) {
      captureSignal((index + 1) % chartEvery === 0 || index === candles.length - 1);
      observeExtremaOrderMassCandle(extremaCollector, candle);
    }

    const simulation = api.status();
    const equity = simulation.equity;
    const markTime = simulation.liquidatedAt ?? candle.closeTime;
    const markPrice = simulation.liquidationPrice ?? candle.close;
    const account = summaryOnly ? undefined : await api.getEquity();
    if (index % equityEvery === 0 || index === candles.length - 1 || simulation.liquidated) {
      equityCurve.push({ time: markTime, equity, price: markPrice });
    }
    if (!summaryOnly) {
      if (index % chartEvery === 0) {
        chartCandles.push({ ...candle });
      } else {
        mergeChartCandle(chartCandles.at(-1)!, candle);
      }
      if ((index + 1) % chartEvery === 0 || index === candles.length - 1) {
        frames.push(traceFrame(markTime, markPrice, account!, latestBotSnapshot!, equity));
        const diagnostics = strategy.getDiagnostics();
        for (const item of series) {
          const value = diagnostics.indicators[item.index < 0 ? "kama" : `average.${item.windowSec}`];
          if (Number.isFinite(value)) item.points.push({ time: candle.closeTime, value: value as number });
        }
      }
    }
    if (
      options.onProgress
      && (processedCandles % progressEvery === 0 || processedCandles === candles.length)
    ) {
      options.onProgress({
        candlesProcessed: processedCandles,
        totalCandles: candles.length,
        elapsedMs: Date.now() - startedAt,
      });
    }
    if (simulation.liquidated) break replay;
  }

  const last = lastProcessed;
  const equity = await api.getEquity();
  const simulation = api.status();
  maxEffectiveLeverage = Math.max(
    maxEffectiveLeverage,
    simulation.maxEffectiveLeverage,
  );
  const finalPrice = simulation.liquidationPrice ?? last.close;
  const finalTime = simulation.liquidatedAt ?? last.closeTime;
  const finalEquity = simulation.equity;
  const netPnl = finalEquity - config.startingQuote;
  const returnPct = config.startingQuote > 0 ? netPnl / config.startingQuote * 100 : 0;
  const risk = calculateRiskAdjustedMetrics(equityCurve, returnPct, maxDrawdownPct);
  const oracle = perfectMarginOracle(candles.slice(0, processedCandles), {
    startingQuote: config.startingQuote,
    leverage: botConfig.maxTargetLeverage,
    friction: apiFriction(config),
    eventMode: "close",
    maxPathCandles: options.maxChartCandles ?? 2_000,
  });
  const snapshots = [...orders.values()].map(({ order, trace }) => legacyOrder(order, trace.createdAt));
  const extremaOrderMass = summaryOnly
    ? undefined
    : summarizeExtremaOrderMass(extremaCollector, fills);
  const trace: BacktestTrace = {
    positions: [...positionTraces.values()],
    grids: [...gridTraces.values()],
    orders: [...orders.values()].map(({ trace }) => trace),
    signals,
    extrema: summaryOnly
      ? []
      : buildExtremaTrace(extremaCollector, [...orders.values()].map(({ trace }) => trace), extremaOrderMass!),
    oracle: oracle.path,
    frames,
  };
  const finalState = createInitialBotState(config);
  finalState.lastPrice = finalPrice;
  finalState.updatedAt = finalTime;
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
    maxInitialBalanceDrawdownPct,
    maxDrawdownPct,
    tradeCount: summaryOnly ? summaryFillCount : fills.length,
    feesPaid: simulation.feesPaid,
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
      strategy: backtestStrategy,
      startTime: candles[0].openTime,
      endTime: finalTime,
      eventsProcessed: processedMarketEvents,
      candlesProcessed: processedCandles,
      stoppedEarly: simulation.liquidated,
      stopReason: simulation.liquidated ? "liquidated" : "completed",
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
      maxInitialBalanceDrawdownPct,
      maxDrawdownPct,
      maxEntryLeverage: botConfig.maxTargetLeverage,
      maxEffectiveLeverage,
      tradeCount: summaryOnly ? summaryFillCount : fills.length,
      feesPaid: simulation.feesPaid,
      maintenancePaid: simulation.maintenancePaid,
      winRate: closedPositionCount > 0 ? profitableClosedPositionCount / closedPositionCount * 100 : 0,
      closedPositionCount,
      profitableClosedPositionCount,
      profitableClosedPositionRate: closedPositionCount > 0
        ? profitableClosedPositionCount / closedPositionCount * 100
        : 0,
      liquidatedPositionCount: simulation.liquidationCount,
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
      if (summaryOnly && (event.type === "fill" || event.type === "partial-fill")) {
        summaryFillCount += 1;
      } else {
        recordFill(event);
      }
      await bot.onOrder(event);
      if (!summaryOnly) {
        captureCause = "fill";
        await capturePositions();
        captureCause = "tick";
      }
    }
  }

  async function capturePositions(): Promise<void> {
    const snapshot = await bot.snapshot();
    latestBotSnapshot = snapshot;
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
    const price = event.fill.price
      ?? (event.fill.filledAsset > 0 ? event.fill.filledQuote / event.fill.filledAsset : 0);
    const feeQuote = event.fill.feeQuote ?? event.fill.filledQuote * apiFriction(config);
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
      reason: backtestStrategy,
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

  function observeRisk(equity: number, effective: number): void {
    peakEquity = Math.max(peakEquity, equity);
    maxInitialBalanceDrawdownPct = Math.max(
      maxInitialBalanceDrawdownPct,
      config.startingQuote > 0
        ? (config.startingQuote - equity) / config.startingQuote * 100
        : 0,
    );
    maxDrawdownPct = Math.max(
      maxDrawdownPct,
      peakEquity > 0 ? (peakEquity - equity) / peakEquity * 100 : 0,
    );
    maxEffectiveLeverage = Math.max(maxEffectiveLeverage, effective);
  }

  function traceFrame(
    time: number,
    price: number,
    account: Awaited<ReturnType<SimulatedTradingApi["getEquity"]>>,
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
        maxInitialBalanceDrawdownPct,
        maxDrawdownPct,
        exposurePct: equity > 0 ? grossExposureQuote / equity * 100 : 0,
        maxEffectiveLeverage,
        feesPaid: api.status().feesPaid,
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

function executionTicks(candle: Candle, signedAsset: number): TradingTick[] {
  const span = Math.max(1, candle.closeTime - candle.openTime);
  const adverse = signedAsset > 0
    ? [[candle.low, 0.33], [candle.high, 0.66]] as const
    : [[candle.high, 0.33], [candle.low, 0.66]] as const;
  return [
    tick(candle.openTime, candle.open, candle.volume / 4),
    tick(candle.openTime + span * adverse[0][1], adverse[0][0], candle.volume / 4),
    tick(candle.openTime + span * adverse[1][1], adverse[1][0], candle.volume / 4),
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
  protected tick: TradingTick | null = null;
  private entry: TradingStrategyEntrySignal | null = null;
  private exit: TradingStrategyExitSignal | null = null;
  protected targetActive: { type: "entry" | "exit"; side: PositionSide }[] = [];
  protected targetIndicators: Record<string, number | null> = {};

  override async onTick(tick: TradingTick): Promise<void> {
    this.tick = tick;
    this.entry = null;
    this.exit = null;
    this.targetActive = [];
    this.targetIndicators = {};
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
      ...this.targetActive,
    ];
    const confirmationActive = diagnostics.gates.some((gate) =>
      gate.passed && (gate.code.includes(".confirmation.") || gate.code.includes(".source.")));
    if (
      active.length === 0
      && Object.keys(this.targetIndicators).length === 0
      && (!includeConfirmations || !confirmationActive)
    ) return null;
    return {
      time: this.tick.timestamp,
      price: this.tick.price,
      source,
      active,
      gates: structuredClone(diagnostics.gates),
      blockers: [...diagnostics.blockers],
      indicators: {
        ...structuredClone(diagnostics.indicators),
        ...this.targetIndicators,
      },
    };
  }

  hasExitDecision(side: PositionSide): boolean {
    return this.exit?.side === side
      || this.targetActive.some((action) => action.type === "exit" && action.side === side);
  }

  currentPrice(): number {
    return this.tick?.price ?? 0;
  }
}

interface HindsightOracleTargetDecision {
  targetExposure: number;
  confidence: number;
  entropy: number;
  feasibleActionCount: number;
}

export function hindsightOracleTargetDecision(
  distribution: ExposureValueOracleActionDistribution,
  currentExposure: number,
  friction: number,
  temperature = HINDSIGHT_ORACLE_TEMPERATURE,
): HindsightOracleTargetDecision {
  let probabilities: Float64Array<ArrayBufferLike>;
  try {
    probabilities = conditionalExposureProbabilities(
      distribution.probabilities,
      distribution.grid,
      currentExposure,
      friction,
      1 / temperature,
    );
  } catch (error) {
    if (error instanceof Error
      && error.message === "Conditional exposure policy has no valid target action.") {
      return {
        targetExposure: currentExposure,
        confidence: 1,
        entropy: 0,
        feasibleActionCount: 0,
      };
    }
    throw error;
  }
  let modalIndex = 0;
  let entropy = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index]!;
    if (probability > probabilities[modalIndex]!) modalIndex = index;
    if (probability > 0) entropy -= probability * Math.log(probability);
  }
  const feasibleActionCount = Math.max(1, distribution.feasibleActionCount);
  const maximumEntropy = Math.log(feasibleActionCount);
  const confidence = maximumEntropy > 0
    ? Math.max(0, Math.min(1, 1 - entropy / maximumEntropy))
    : 1;
  return {
    targetExposure: distribution.grid[modalIndex]!,
    confidence,
    entropy,
    feasibleActionCount,
  };
}

export function confidenceScaledHindsightOracleExposure(
  modalExposure: number,
  confidence: number,
  power = HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
): number {
  if (!(power >= 0) || !Number.isFinite(power)) {
    throw new Error("Hindsight oracle confidence exposure power must be finite and non-negative.");
  }
  return modalExposure * Math.max(0, Math.min(1, confidence)) ** power;
}

export function scaleOracleConfidence(confidence: number, staticScale: number): number {
  validateOracleStaticConfidenceScale(staticScale);
  if (!Number.isFinite(confidence)) throw new Error("Oracle confidence must be finite.");
  return Math.max(0, Math.min(1, confidence)) * staticScale;
}

function validateOracleStaticConfidenceScale(value: number): void {
  if (!(value >= 0 && value <= 1) || !Number.isFinite(value)) {
    throw new Error("Oracle static confidence scale must be in [0, 1].");
  }
}

export function confidenceConditionedHindsightOracleExposure(
  modalExposure: number,
  distributionConfidence: number,
  maximumLeverage: number,
  power = HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
  leverageFloor = HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
  staticConfidenceScale = ORACLE_DEFAULT_STATIC_CONFIDENCE_SCALE,
): number {
  if (!(maximumLeverage >= 0) || !Number.isFinite(maximumLeverage)) {
    throw new Error("Hindsight oracle maximum leverage must be finite and non-negative.");
  }
  if (!(leverageFloor >= 0 && leverageFloor <= 1) || !Number.isFinite(leverageFloor)) {
    throw new Error("Hindsight oracle confidence leverage floor must be in [0, 1].");
  }
  validateOracleStaticConfidenceScale(staticConfidenceScale);
  const boundedConfidence = Math.max(0, Math.min(1, distributionConfidence));
  const scaled = confidenceScaledHindsightOracleExposure(
    modalExposure,
    boundedConfidence,
    power,
  );
  const cap = maximumLeverage * confidenceConditionedOracleLeverageFraction(
    boundedConfidence,
    leverageFloor,
    staticConfidenceScale,
  );
  return Math.max(-cap, Math.min(cap, scaled));
}

export function confidenceConditionedOracleLeverageFraction(
  distributionConfidence: number,
  leverageFloor: number,
  staticConfidenceScale: number,
): number {
  if (!Number.isFinite(distributionConfidence)) {
    throw new Error("Oracle distribution confidence must be finite.");
  }
  if (!(leverageFloor >= 0 && leverageFloor <= 1) || !Number.isFinite(leverageFloor)) {
    throw new Error("Hindsight oracle confidence leverage floor must be in [0, 1].");
  }
  validateOracleStaticConfidenceScale(staticConfidenceScale);
  const boundedConfidence = Math.max(0, Math.min(1, distributionConfidence));
  const effectiveFloor = leverageFloor * staticConfidenceScale;
  return staticConfidenceScale
    * (effectiveFloor + (1 - effectiveFloor) * boundedConfidence);
}

/**
 * Map the native oracle exposure grid onto an execution leverage ceiling.
 *
 * The learned logits describe the native oracle grid (about +/-98.4x).  A
 * lower-risk replay must scale both the policy state and its target by the
 * same factor.  Merely clipping a native target to the execution ceiling
 * makes a filled +1x position look like a nearly-flat +1x native state on the
 * following decision, which changes transition-cost conditioning.
 */
export function oracleExecutionExposureScale(
  distribution: ExposureValueOracleActionDistribution,
  maximumLeverage: number,
): number {
  if (!(maximumLeverage > 0) || !Number.isFinite(maximumLeverage)) {
    throw new Error("Oracle execution maximum leverage must be finite and positive.");
  }
  let nativeMaximum = 0;
  for (const exposure of distribution.grid) {
    if (!Number.isFinite(exposure)) {
      throw new Error("Oracle exposure grid must contain only finite values.");
    }
    nativeMaximum = Math.max(nativeMaximum, Math.abs(exposure));
  }
  if (!(nativeMaximum > 0)) {
    throw new Error("Oracle exposure grid must contain a non-zero action.");
  }
  return Math.min(1, maximumLeverage / nativeMaximum);
}

export function oracleMaximumEffectiveLeverage(maximumLeverage: number): number {
  if (!(maximumLeverage > 0) || !Number.isFinite(maximumLeverage)) {
    throw new Error("Oracle maximum leverage must be finite and positive.");
  }
  const targetMaximum = Math.min(HINDSIGHT_ORACLE_MAX_EXPOSURE, maximumLeverage);
  return HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE
    * targetMaximum / HINDSIGHT_ORACLE_MAX_EXPOSURE;
}

export function hindsightOracleUsableDistribution(
  distribution: ExposureValueOracleActionDistribution,
  maximumExposure = HINDSIGHT_ORACLE_MAX_EXPOSURE,
): ExposureValueOracleActionDistribution {
  const indexes: number[] = [];
  for (let index = 0; index < distribution.grid.length; index += 1) {
    if (Math.abs(distribution.grid[index]!) <= maximumExposure) indexes.push(index);
  }
  if (indexes.length < 2) {
    throw new Error("Hindsight oracle distribution has no usable exposure interval.");
  }
  const grid = Float64Array.from(indexes, (index) => distribution.grid[index]!);
  const probabilities = Float64Array.from(
    indexes,
    (index) => distribution.probabilities[index]!,
  );
  const total = probabilities.reduce((sum, probability) => sum + probability, 0);
  if (!(total > 0)) throw new Error("Hindsight oracle usable interval has no probability mass.");
  let feasibleActionCount = 0;
  let modalIndex = 0;
  let mean = 0;
  let secondMoment = 0;
  let entropy = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] /= total;
    const probability = probabilities[index]!;
    if (probability > probabilities[modalIndex]!) modalIndex = index;
    if (probability > 0) {
      feasibleActionCount += 1;
      entropy -= probability * Math.log(probability);
    }
    mean += probability * grid[index]!;
    secondMoment += probability * grid[index]! ** 2;
  }
  return {
    grid,
    probabilities,
    mean,
    secondMoment,
    modalExposure: grid[modalIndex]!,
    entropy,
    opportunity: distribution.opportunity,
    feasibleActionCount,
  };
}

class TracingOraclePolicyStrategy extends TracingPeakValleyStrategy {
  constructor(
    options: StrategyOptions<PeakValleyStrategyConfig>,
    private readonly distributionAt: (
      timestamp: number,
    ) => ExposureValueOracleActionDistribution | null,
    private readonly friction: number,
    private readonly confidenceExposurePower: number,
    private readonly staticConfidenceScale: number,
    private readonly onOracleDecision?: (decision: OracleBacktestDecision) => void,
  ) {
    super(options);
  }

  override staticConfidence(): number {
    return this.staticConfidenceScale;
  }

  async targetExposureSignal(
    context: TradingStrategyTargetExposureContext,
  ): Promise<TradingStrategyTargetExposureSignal | null> {
    const distribution = this.distributionAt(context.timestamp);
    if (!distribution || !this.tick) return null;
    const executionScale = oracleExecutionExposureScale(
      distribution,
      context.maxLeverage,
    );
    const decision = hindsightOracleTargetDecision(
      distribution,
      context.currentExposure / executionScale,
      this.friction,
    );
    const scaledModalExposure = decision.targetExposure * executionScale;
    const effectiveConfidence = scaleOracleConfidence(
      decision.confidence,
      this.staticConfidenceScale,
    );
    const targetExposure = confidenceScaledHindsightOracleExposure(
      scaledModalExposure,
      decision.confidence,
      this.confidenceExposurePower,
    );
    const confidenceScale = scaledModalExposure !== 0
      ? targetExposure / scaledModalExposure
      : effectiveConfidence ** this.confidenceExposurePower;
    const gridStep = Math.abs(distribution.grid[1]! - distribution.grid[0]!);
    const effectiveGridStep = gridStep * executionScale;
    const delta = targetExposure - context.currentExposure;
    const signalEmitted = Math.abs(delta) >= effectiveGridStep / 2;
    this.onOracleDecision?.({
      timestamp: context.timestamp,
      currentExposure: context.currentExposure,
      rawModalExposure: distribution.modalExposure,
      conditionedModalExposure: decision.targetExposure,
      targetExposure,
      confidence: decision.confidence,
      staticConfidence: this.staticConfidenceScale,
      effectiveConfidence,
      entropy: decision.entropy,
      signalEmitted,
    });
    this.targetIndicators = {
      "oracle.currentExposure": context.currentExposure,
      "oracle.nativeCurrentExposure": context.currentExposure / executionScale,
      "oracle.executionScale": executionScale,
      "oracle.modalExposure": decision.targetExposure,
      "oracle.targetExposure": targetExposure,
      "oracle.deltaExposure": delta,
      "oracle.confidence": decision.confidence,
      "oracle.staticConfidenceScale": this.staticConfidenceScale,
      "oracle.effectiveConfidence": effectiveConfidence,
      "oracle.confidenceExposureScale": confidenceScale,
      "oracle.confidenceExposurePower": this.confidenceExposurePower,
      "oracle.entropy": decision.entropy,
      "oracle.feasibleActions": decision.feasibleActionCount,
    };
    if (!signalEmitted) return null;
    this.targetActive = targetExposureActions(context.currentExposure, targetExposure);
    return {
      targetExposure,
      // A flat target is a time-sensitive risk reduction.  Market-close it at
      // the decision tick so a backtest window (and the live policy) cannot
      // retain stale exposure merely because a passive exit never trades.
      price: targetExposure === 0 ? null : this.tick.price,
      confidence: decision.confidence,
    };
  }
}

function targetExposureActions(
  currentExposure: number,
  targetExposure: number,
): { type: "entry" | "exit"; side: PositionSide }[] {
  if (targetExposure > 0) {
    return currentExposure < 0
      ? [{ type: "exit", side: "short" }, { type: "entry", side: "long" }]
      : [{ type: targetExposure > currentExposure ? "entry" : "exit", side: "long" }];
  }
  if (targetExposure < 0) {
    return currentExposure > 0
      ? [{ type: "exit", side: "long" }, { type: "entry", side: "short" }]
      : [{ type: targetExposure < currentExposure ? "entry" : "exit", side: "short" }];
  }
  return currentExposure > 0
    ? [{ type: "exit", side: "long" }]
    : currentExposure < 0
      ? [{ type: "exit", side: "short" }]
      : [];
}

function createHindsightOracleDistributionProvider(
  candles: readonly Candle[],
  future: readonly Candle[],
  config: StrategyConfig,
  intervalMs: number,
): (timestamp: number) => ExposureValueOracleActionDistribution | null {
  const source = continuousOracleCandles(candles, future, intervalMs);
  const prices = Float64Array.from(source, (candle) => candle.close);
  const scoredLength = candles.length;
  const holdingPeriodSteps = Math.max(1, Math.round(
    HINDSIGHT_ORACLE_HOLDING_PERIOD_MS / intervalMs,
  ));
  const decisionDelaySteps = Math.max(1, Math.round(
    HINDSIGHT_ORACLE_DECISION_DELAY_MS / intervalMs,
  ));
  const valueHorizonSteps = Math.max(holdingPeriodSteps, Math.round(
    HINDSIGHT_ORACLE_VALUE_HORIZON_MS / intervalMs,
  ));
  const quoteBorrowRate = hourlyRatePerCandle(
    HINDSIGHT_ORACLE_MAINTENANCE_BPS_HOUR,
    intervalMs,
  );

  return (timestamp) => {
    const candleIndex = candleIndexAtClose(candles, timestamp);
    if (
      candleIndex < 0
      || candleIndex >= scoredLength - 1
      || candleIndex % holdingPeriodSteps !== 0
    ) return null;
    const terminalIndex = Math.min(prices.length - 1, candleIndex + valueHorizonSteps);
    if (terminalIndex <= candleIndex) return null;
    const horizonPrices = prices.subarray(candleIndex, terminalIndex + 1);
    return hindsightOracleUsableDistribution(exposureValueOracleActionDistribution(horizonPrices, {
      scoreStartIndex: 0,
      holdingPeriodSteps: Math.min(holdingPeriodSteps, horizonPrices.length - 1),
      decisionDelaySteps,
      valueHorizonSteps,
      terminalIndex: horizonPrices.length - 1,
      friction: apiFriction(config),
      gridSize: HINDSIGHT_ORACLE_GRID_SIZE,
      minExposure: -HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE,
      maxExposure: HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE,
      maxEffectiveExposure: HINDSIGHT_ORACLE_MAX_EFFECTIVE_EXPOSURE,
      initialExposure: 0,
      temperature: HINDSIGHT_ORACLE_TEMPERATURE,
      opportunityEpsilon: 0,
      quoteBorrowRate,
      assetBorrowRate: quoteBorrowRate,
      distributionOnly: true,
      includePath: false,
    }));
  };
}

function continuousOracleCandles(
  candles: readonly Candle[],
  future: readonly Candle[],
  intervalMs: number,
): Candle[] {
  const result = [...candles];
  let expected = result.at(-1)!.openTime + intervalMs;
  for (const candle of future) {
    if (candle.openTime < expected) continue;
    if (candle.openTime !== expected) break;
    result.push(candle);
    expected += intervalMs;
  }
  return result;
}

function candleIndexAtClose(candles: readonly Candle[], timestamp: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.closeTime < timestamp) low = middle + 1;
    else high = middle;
  }
  return candles[low]?.closeTime === timestamp ? low : -1;
}

function hourlyRatePerCandle(hourlyRateBps: number, intervalMs: number): number {
  return Math.expm1(Math.log1p(hourlyRateBps / 10_000) * intervalMs / 3_600_000);
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
