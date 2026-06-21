import {
  SimulatedTradingBot,
  calculateRiskAdjustedMetrics,
  compactBacktestState,
  createInitialBotState,
  createStrategyConfig,
  type BacktestSummary,
  type BacktestSampleSummary,
  type BacktestPreset,
  type BacktestProgressSnapshot,
  type BacktestResult,
  type BotMetrics,
  type Candle,
  type EquityPoint,
  type PaperBotState,
  type StrategyConfig,
} from "@trading/bot-algo";
import {
  HistoricalCandleCache,
  type HistoricalCandleCacheStats,
} from "./historical-candle-cache.js";
import type { StreamVenue } from "./binance-markets.js";

const WIPEOUT_EQUITY_FRACTION = 0.01;
const WIPEOUT_CHECK_CANDLES = 100;
const MAX_EQUITY_POINTS = 800;
const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_RANDOM_SAMPLE_COUNT = 40;
const MAX_RANDOM_SAMPLE_COUNT = 200;
const DEFAULT_RANDOM_WINDOW_MS = 7 * DAY_MS;
const DEFAULT_RANDOM_MIN_WINDOW_MS = DAY_MS;
const DEFAULT_RANDOM_MAX_WINDOW_MS = 30 * DAY_MS;
const DEFAULT_RANDOM_LOOKBACK_MS = 365 * DAY_MS;
const DEFAULT_HISTORICAL_RANGE_MS = 30 * DAY_MS;

type FixedHistoricalPreset = Extract<BacktestPreset, "week" | "month" | "year">;
type HistoricalPreset =
  | FixedHistoricalPreset
  | "last-x"
  | "random-windows"
  | "random-length-windows";

const periodDurations: Record<FixedHistoricalPreset, number> = {
  week: 7 * 24 * 60 * 60 * 1000,
  month: 30 * 24 * 60 * 60 * 1000,
  year: 365 * 24 * 60 * 60 * 1000,
};

interface HistoricalCacheOptions {
  dataDir: string;
  maxBytes: number;
  minFreeBytes: number;
}

export interface HistoricalBacktestMarket {
  marketId: string;
  marketKey: string;
  venue: StreamVenue;
  symbol: string;
  displaySymbol: string;
  baseAsset: string;
  quoteAsset: string;
  maxLeverage?: number;
}

export interface HistoricalBacktestOptions {
  id: string;
  preset: HistoricalPreset;
  marketId?: string;
  marketKey: string;
  venue: StreamVenue;
  symbol: string;
  displaySymbol?: string;
  baseAsset?: string;
  quoteAsset?: string;
  maxLeverage?: number;
  interval: string;
  config: StrategyConfig;
  cache: HistoricalCacheOptions;
  historicalRangeMs?: number;
  randomSampleCount?: number;
  randomWindowMs?: number;
  randomMinWindowMs?: number;
  randomMaxWindowMs?: number;
  randomLookbackMs?: number;
  randomMarkets?: HistoricalBacktestMarket[];
  randomPairCount?: number;
  cancelSignal?: AbortSignal;
}

interface HistoricalRangeBacktestOptions extends HistoricalBacktestOptions {
  targetStartTime: number;
  targetEndTime: number;
  currentSample?: number;
  sampleCount?: number;
  sampleWindowMs?: number;
  sampleMinWindowMs?: number;
  sampleMaxWindowMs?: number;
  sampleLookbackMs?: number;
}

interface RandomBacktestWindow {
  startTime: number;
  endTime: number;
  windowMs: number;
}

interface PerfectMarginBenchmarkAccumulator {
  startingQuote: number;
  leverage: number;
  roundTripFrictionRate: number;
  previousPrice?: number;
  netPnl: number;
}

type PerfectMarginBenchmark = Pick<
  BacktestSummary,
  | "perfectMarginLeverage"
  | "perfectMarginFinalEquity"
  | "perfectMarginNetPnl"
  | "perfectMarginReturnPct"
  | "perfectMarginCapturePct"
>;

export class BacktestCancelledError extends Error {
  constructor(message = "Backtest cancelled") {
    super(message);
    this.name = "BacktestCancelledError";
  }
}

export function isBacktestCancelledError(error: unknown): boolean {
  return (
    error instanceof BacktestCancelledError ||
    (error instanceof Error &&
      (error.name === "AbortError" || error.name === "BacktestCancelledError"))
  );
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new BacktestCancelledError();
  }
}

function historicalMarketFromOptions(
  options: HistoricalBacktestOptions,
): HistoricalBacktestMarket {
  return {
    marketId: options.marketId ?? `${options.venue}:${options.symbol}`,
    marketKey: options.marketKey,
    venue: options.venue,
    symbol: options.symbol,
    displaySymbol: options.displaySymbol ?? options.symbol,
    baseAsset: options.baseAsset ?? options.config.baseAsset,
    quoteAsset: options.quoteAsset ?? options.config.quoteAsset,
    maxLeverage: options.maxLeverage,
  };
}

export async function runHistoricalCandleBacktest(
  options: HistoricalBacktestOptions,
  onProgress: (progress: BacktestProgressSnapshot) => void,
): Promise<BacktestResult> {
  throwIfCancelled(options.cancelSignal);
  if (options.preset === "random-windows" || options.preset === "random-length-windows") {
    return runRandomHistoricalCandleBacktest(options, onProgress);
  }

  const targetEndTime = Date.now();
  const intervalMs = intervalToMs(options.interval);
  const durationMs =
    options.preset === "last-x"
      ? normalizeDurationMs(
          options.historicalRangeMs,
          DEFAULT_HISTORICAL_RANGE_MS,
          intervalMs,
        )
      : periodDurations[options.preset];
  const targetStartTime = targetEndTime - durationMs;

  return runHistoricalRangeBacktest(
    {
      ...options,
      targetStartTime,
      targetEndTime,
    },
    onProgress,
  );
}

async function runRandomHistoricalCandleBacktest(
  options: HistoricalBacktestOptions,
  onProgress: (progress: BacktestProgressSnapshot) => void,
): Promise<BacktestResult> {
  throwIfCancelled(options.cancelSignal);
  const intervalMs = intervalToMs(options.interval);
  const variableLengthWindows = options.preset === "random-length-windows";
  const primaryMarket = historicalMarketFromOptions(options);
  const markets = [primaryMarket, ...(options.randomMarkets ?? [])];
  const marketCount = markets.length;
  const randomPairCount = Math.max(0, marketCount - 1);
  const sampleCount = clampInteger(
    options.randomSampleCount,
    DEFAULT_RANDOM_SAMPLE_COUNT,
    1,
    MAX_RANDOM_SAMPLE_COUNT,
  );
  const fixedSampleWindowMs = normalizeDurationMs(
    options.randomWindowMs,
    DEFAULT_RANDOM_WINDOW_MS,
    intervalMs,
  );
  const sampleWindowMs = variableLengthWindows ? undefined : fixedSampleWindowMs;
  const sampleMinWindowMs = variableLengthWindows
    ? normalizeDurationMs(
        options.randomMinWindowMs,
        DEFAULT_RANDOM_MIN_WINDOW_MS,
        intervalMs,
      )
    : fixedSampleWindowMs;
  const sampleMaxWindowMs = variableLengthWindows
    ? normalizeDurationMs(
        options.randomMaxWindowMs,
        DEFAULT_RANDOM_MAX_WINDOW_MS,
        sampleMinWindowMs,
      )
    : fixedSampleWindowMs;
  const sampleLookbackMs = normalizeDurationMs(
    options.randomLookbackMs,
    DEFAULT_RANDOM_LOOKBACK_MS,
    sampleMaxWindowMs,
  );
  const config = createStrategyConfig({
    ...options.config,
    symbol: primaryMarket.symbol,
    baseAsset: primaryMarket.baseAsset,
    quoteAsset: primaryMarket.quoteAsset,
    maxLeverage: cappedMaxLeverage(options.config.maxLeverage, primaryMarket.maxLeverage),
  });
  const targetEndTime = Date.now();
  const targetStartTime = targetEndTime - sampleLookbackMs;
  const windows = buildRandomWindows({
    count: sampleCount,
    targetStartTime,
    targetEndTime,
    minWindowMs: sampleMinWindowMs,
    maxWindowMs: sampleMaxWindowMs,
    intervalMs,
  });
  const totalSampleCount = sampleCount * marketCount;
  const estimatedCandles = sumDefined(
    windows.map((window) =>
      estimateRangeCandles(window.startTime, window.endTime, intervalMs),
    ),
  ) * marketCount;
  const startedAt = Date.now();
  const results: BacktestResult[] = [];
  let completedCandles = 0;
  let completedFinalEquity = 0;
  let completedReturnPct = 0;
  let completedNetPnlPerDay = 0;
  let completedReturnPctPerDay = 0;
  let completedSurvivedMs = 0;
  let completedSurvivedSamples = 0;
  let cacheStats = emptyCacheStats();

  emitAggregateProgress(
    variableLengthWindows
      ? "Starting random-length backtest"
      : "Starting random-window backtest",
  );

  for (let marketIndex = 0; marketIndex < markets.length; marketIndex += 1) {
    const market = markets[marketIndex];
    for (let windowIndex = 0; windowIndex < windows.length; windowIndex += 1) {
      throwIfCancelled(options.cancelSignal);
      const currentSample = marketIndex * sampleCount + windowIndex + 1;
      const currentWindowSample = windowIndex + 1;
      const window = windows[windowIndex];
      const result = await runHistoricalRangeBacktest(
        {
          ...options,
          ...market,
          config,
          randomPairCount,
          targetStartTime: window.startTime,
          targetEndTime: window.endTime,
          currentSample,
          sampleCount: totalSampleCount,
          sampleWindowMs: window.windowMs,
          sampleMinWindowMs,
          sampleMaxWindowMs,
          sampleLookbackMs,
        },
        (progress) => {
          emitAggregateProgress(
            `${market.displaySymbol} sample ${currentWindowSample}/${sampleCount}: ${progress.message}`,
            progress,
          );
        },
      );

      throwIfCancelled(options.cancelSignal);
      results.push(result);
      completedCandles += result.summary.candlesProcessed ?? 0;
      completedFinalEquity += result.summary.finalEquity;
      completedReturnPct += result.summary.returnPct;
      const sampleRate = dailyRateForResult(result);
      completedNetPnlPerDay += sampleRate.netPnlPerDay;
      completedReturnPctPerDay += sampleRate.returnPctPerDay;
      if (result.summary.survivedMs !== undefined) {
        completedSurvivedMs += result.summary.survivedMs;
        completedSurvivedSamples += 1;
      }
      mergeSummaryCacheStats(cacheStats, result.summary);
      emitAggregateProgress(
        `Completed ${market.displaySymbol} sample ${currentWindowSample}/${sampleCount}`,
      );
    }
  }

  const aggregate = buildRandomAggregateResult({
    options,
    config,
    markets,
    results,
    startedAt,
    targetStartTime,
    targetEndTime,
    sampleWindowMs,
    sampleMinWindowMs,
    sampleMaxWindowMs,
    sampleLookbackMs,
  });
  onProgress({
    id: options.id,
    preset: options.preset,
    status: "completed",
    source: "candles",
    startedAt,
    updatedAt: Date.now(),
    targetStartTime,
    targetEndTime,
    processedStartTime: aggregate.summary.startTime,
    processedEndTime: aggregate.summary.endTime,
    processedCandles: aggregate.summary.candlesProcessed ?? completedCandles,
    estimatedCandles,
    requests: aggregate.summary.requests ?? cacheStats.requests,
    cacheHitCandles: aggregate.summary.cacheHitCandles,
    cacheMissCandles: aggregate.summary.cacheMissCandles,
    cacheFetchedCandles: aggregate.summary.cacheFetchedCandles,
    cacheSizeBytes: aggregate.summary.cacheSizeBytes,
    cacheEvictedBytes: aggregate.summary.cacheEvictedBytes,
    cacheEvictedFiles: aggregate.summary.cacheEvictedFiles,
    currentSample: totalSampleCount,
    sampleCount: totalSampleCount,
    sampleWindowMs,
    sampleMinWindowMs,
    sampleMaxWindowMs,
    sampleLookbackMs,
    marketCount,
    randomPairCount,
    netPnlPerDay: aggregate.summary.netPnlPerDay,
    returnPctPerDay: aggregate.summary.returnPctPerDay,
    percent: 100,
    equity: aggregate.summary.finalEquity,
    returnPct: aggregate.summary.returnPct,
    stopReason: "completed",
    survivedMs: aggregate.summary.survivedMs,
    candlesPerSecond: aggregate.summary.candlesPerSecond,
    message:
      marketCount > 1
        ? `Averaged ${totalSampleCount.toLocaleString()} samples across ${marketCount.toLocaleString()} pairs`
        : variableLengthWindows
          ? `Averaged ${sampleCount.toLocaleString()} random-length windows`
          : `Averaged ${sampleCount.toLocaleString()} random windows`,
    result: aggregate,
  });

  return aggregate;

  function emitAggregateProgress(
    message: string,
    currentProgress?: BacktestProgressSnapshot,
  ): void {
    const completedSamples = results.length;
    const currentSample =
      currentProgress?.currentSample ?? Math.min(totalSampleCount, completedSamples + 1);
    const currentSampleFraction =
      currentProgress === undefined ? 0 : currentProgress.percent / 100;
    const percent =
      currentProgress === undefined
        ? (completedSamples / totalSampleCount) * 100
        : ((currentSample - 1 + currentSampleFraction) / totalSampleCount) * 100;
    const processedCandles = completedCandles + (currentProgress?.processedCandles ?? 0);
    const inFlightSamples = completedSamples + (currentProgress ? 1 : 0);
    const equity =
      inFlightSamples > 0
        ? (completedFinalEquity + (currentProgress?.equity ?? 0)) / inFlightSamples
        : config.startingQuote;
    const returnPct =
      inFlightSamples > 0
        ? (completedReturnPct + (currentProgress?.returnPct ?? 0)) / inFlightSamples
        : 0;
    const currentRate = currentProgress
      ? dailyRateForProgress(currentProgress, config.startingQuote)
      : undefined;
    const netPnlPerDay =
      inFlightSamples > 0
        ? (completedNetPnlPerDay + (currentRate?.netPnlPerDay ?? 0)) / inFlightSamples
        : undefined;
    const returnPctPerDay =
      inFlightSamples > 0
        ? (completedReturnPctPerDay + (currentRate?.returnPctPerDay ?? 0)) /
          inFlightSamples
        : undefined;
    const survivedMs =
      currentProgress?.survivedMs !== undefined
        ? (completedSurvivedMs + currentProgress.survivedMs) /
          Math.max(1, completedSurvivedSamples + 1)
        : completedSurvivedSamples > 0
          ? completedSurvivedMs / completedSurvivedSamples
          : undefined;
    const progressCacheStats = mergeProgressCacheStats(cacheStats, currentProgress);
    const elapsedMs = Date.now() - startedAt;

    onProgress({
      id: options.id,
      preset: options.preset,
      status: "running",
      source: "candles",
      startedAt,
      updatedAt: Date.now(),
      targetStartTime,
      targetEndTime,
      processedStartTime: currentProgress?.processedStartTime,
      processedEndTime: currentProgress?.processedEndTime,
      processedCandles,
      estimatedCandles,
      requests: progressCacheStats.requests,
      cacheHitCandles: progressCacheStats.cacheHitCandles,
      cacheMissCandles: progressCacheStats.cacheMissCandles,
      cacheFetchedCandles: progressCacheStats.cacheFetchedCandles,
      cacheSizeBytes: progressCacheStats.cacheSizeBytes,
      cacheEvictedBytes: progressCacheStats.cacheEvictedBytes,
      cacheEvictedFiles: progressCacheStats.cacheEvictedFiles,
      currentSample,
      sampleCount: totalSampleCount,
      sampleWindowMs: currentProgress?.sampleWindowMs ?? sampleWindowMs,
      sampleMinWindowMs,
      sampleMaxWindowMs,
      sampleLookbackMs,
      marketCount,
      randomPairCount,
      netPnlPerDay,
      returnPctPerDay,
      percent: Math.max(0, Math.min(100, percent)),
      equity,
      returnPct,
      survivedMs,
      candlesPerSecond:
        processedCandles > 0 && elapsedMs > 0
          ? (processedCandles / elapsedMs) * 1000
          : undefined,
      message,
    });
  }
}

async function runHistoricalRangeBacktest(
  options: HistoricalRangeBacktestOptions,
  onProgress: (progress: BacktestProgressSnapshot) => void,
): Promise<BacktestResult> {
  throwIfCancelled(options.cancelSignal);
  const { targetStartTime, targetEndTime } = options;
  const intervalMs = intervalToMs(options.interval);
  const estimatedCandles = estimateRangeCandles(
    targetStartTime,
    targetEndTime,
    intervalMs,
  );
  const config = createStrategyConfig({
    ...options.config,
    symbol: options.symbol,
    baseAsset: options.baseAsset ?? options.config.baseAsset,
    quoteAsset: options.quoteAsset ?? options.config.quoteAsset,
    maxLeverage: cappedMaxLeverage(options.config.maxLeverage, options.maxLeverage),
  });
  const bot = new SimulatedTradingBot(createInitialBotState(config));
  const perfectMargin = createPerfectMarginBenchmark(config);
  const cache = new HistoricalCandleCache({
    dataDir: options.cache.dataDir,
    marketKey: options.marketKey,
    symbol: options.symbol,
    interval: options.interval,
    intervalMs,
    maxBytes: options.cache.maxBytes,
    minFreeBytes: options.cache.minFreeBytes,
  });
  const equityCurve: EquityPoint[] = [];
  const sampleEvery = Math.max(1, Math.ceil(estimatedCandles / MAX_EQUITY_POINTS));
  const metricsEvery = Math.max(1, Math.min(sampleEvery, WIPEOUT_CHECK_CANDLES));
  const startedAt = Date.now();
  const wipeoutEquity = config.startingQuote * WIPEOUT_EQUITY_FRACTION;

  let processedCandles = 0;
  let cacheStats = emptyCacheStats();
  let processedStartTime: number | undefined;
  let processedEndTime: number | undefined;
  let stopReason: "completed" | "wiped_out" = "completed";
  let latestMetrics = bot.view().metrics;
  let survivedMs: number | undefined;

  emitProgress("running", "Checking historical candle cache");

  cacheStats = await cache.ensureRange(
    targetStartTime,
    targetEndTime,
    (request) =>
      fetchKlines({
        venue: options.venue,
        symbol: options.symbol,
        interval: options.interval,
        startTime: request.startTime,
        endTime: request.endTime,
        limit: request.limit,
        signal: options.cancelSignal,
      }),
    (stats) => {
      throwIfCancelled(options.cancelSignal);
      cacheStats = stats;
      emitProgress(
        "running",
        stats.cacheMissCandles > stats.cacheFetchedCandles
          ? `Caching ${(
              stats.cacheMissCandles - stats.cacheFetchedCandles
            ).toLocaleString()} missing candles`
          : "Historical cache ready",
      );
    },
  );
  throwIfCancelled(options.cancelSignal);

  const replayStartedAt = Date.now();

  outer: for await (const candles of cache.readRangeBatches(targetStartTime, targetEndTime)) {
    for (const candle of candles) {
      if (processedCandles % 100 === 0) {
        throwIfCancelled(options.cancelSignal);
      }
      if (candle.openTime < targetStartTime || candle.openTime > targetEndTime) {
        continue;
      }

      processedStartTime ??= candle.openTime;
      processedEndTime = candle.closeTime;

      replayCandle(bot, candle, perfectMargin);
      processedCandles += 1;
      const shouldSample = processedCandles % sampleEvery === 0;
      const shouldCheckMetrics =
        shouldSample ||
        processedCandles % metricsEvery === 0 ||
        processedCandles >= estimatedCandles;

      if (shouldCheckMetrics) {
        latestMetrics = bot.markToMarket();
      }

      if (shouldSample) {
        equityCurve.push({
          time: candle.closeTime,
          equity: latestMetrics.equity,
          price: candle.close,
        });
      }

      if (
        shouldCheckMetrics &&
        processedCandles >= config.slowWindow &&
        latestMetrics.equity <= wipeoutEquity
      ) {
        stopReason = "wiped_out";
        break outer;
      }
    }

    throwIfCancelled(options.cancelSignal);
    emitProgress("running", `Processed ${processedCandles.toLocaleString()} candles`);
  }

  latestMetrics = bot.markToMarket();
  const finalState = compactBacktestState(bot.view());
  if (processedEndTime && equityCurve.at(-1)?.time !== processedEndTime) {
    equityCurve.push({
      time: processedEndTime,
      equity: finalState.metrics.equity,
      price: finalState.lastPrice,
    });
  }

  const durationMs = Date.now() - startedAt;
  const replayDurationMs = Date.now() - replayStartedAt;
  survivedMs =
    processedStartTime && processedEndTime
      ? processedEndTime - processedStartTime
      : undefined;
  const riskMetrics = calculateRiskAdjustedMetrics(
    equityCurve,
    finalState.metrics.returnPct,
    finalState.metrics.maxDrawdownPct,
  );
  const result: BacktestResult = {
    summary: {
      symbol: options.symbol,
      marketId: options.marketId,
      displaySymbol: options.displaySymbol,
      source: "candles",
      startTime: processedStartTime ?? targetStartTime,
      endTime: processedEndTime ?? targetStartTime,
      targetStartTime,
      targetEndTime,
      eventsProcessed: processedCandles * 4,
      candlesProcessed: processedCandles,
      requests: cacheStats.requests,
      cacheHitCandles: cacheStats.cacheHitCandles,
      cacheMissCandles: cacheStats.cacheMissCandles,
      cacheFetchedCandles: cacheStats.cacheFetchedCandles,
      cacheSizeBytes: cacheStats.cacheSizeBytes,
      cacheEvictedBytes: cacheStats.cacheEvictedBytes,
      cacheEvictedFiles: cacheStats.cacheEvictedFiles,
      stoppedEarly: stopReason !== "completed",
      stopReason,
      survivedMs,
      durationMs,
      replayDurationMs,
      candlesPerSecond:
        replayDurationMs > 0 ? (processedCandles / replayDurationMs) * 1000 : undefined,
      finalEquity: finalState.metrics.equity,
      netPnl: finalState.metrics.netPnl,
      returnPct: finalState.metrics.returnPct,
      ...riskMetrics,
      maxDrawdownPct: finalState.metrics.maxDrawdownPct,
      tradeCount: finalState.metrics.tradeCount,
      winRate: finalState.metrics.winRate,
      ...finalizePerfectMarginBenchmark(perfectMargin, finalState.metrics.netPnl),
    },
    equityCurve,
    orders: finalState.orders,
    fills: finalState.fills,
    finalState,
  };

  return result;

  function emitProgress(
    status: BacktestProgressSnapshot["status"],
    message: string,
  ): void {
    const currentSurvivedMs =
      processedStartTime && processedEndTime
        ? processedEndTime - processedStartTime
        : undefined;
    onProgress({
      id: options.id,
      preset: options.preset,
      status,
      source: "candles",
      startedAt,
      updatedAt: Date.now(),
      targetStartTime,
      targetEndTime,
      processedStartTime,
      processedEndTime,
      processedCandles,
      estimatedCandles,
      requests: cacheStats.requests,
      cacheHitCandles: cacheStats.cacheHitCandles,
      cacheMissCandles: cacheStats.cacheMissCandles,
      cacheFetchedCandles: cacheStats.cacheFetchedCandles,
      cacheSizeBytes: cacheStats.cacheSizeBytes,
      cacheEvictedBytes: cacheStats.cacheEvictedBytes,
      cacheEvictedFiles: cacheStats.cacheEvictedFiles,
      currentSample: options.currentSample,
      sampleCount: options.sampleCount,
      sampleWindowMs: options.sampleWindowMs,
      sampleMinWindowMs: options.sampleMinWindowMs,
      sampleMaxWindowMs: options.sampleMaxWindowMs,
      sampleLookbackMs: options.sampleLookbackMs,
      marketCount: options.randomMarkets ? options.randomMarkets.length + 1 : undefined,
      randomPairCount: options.randomPairCount,
      percent: Math.min(100, (processedCandles / estimatedCandles) * 100),
      equity: latestMetrics.equity,
      returnPct: latestMetrics.returnPct,
      stopReason: status === "completed" ? stopReason : undefined,
      survivedMs: currentSurvivedMs,
      candlesPerSecond: currentCandlesPerSecond(),
      message,
    });
  }

  function currentCandlesPerSecond(): number | undefined {
    if (processedCandles <= 0) {
      return undefined;
    }

    const replayDurationMs = Date.now() - replayStartedAt;
    return replayDurationMs > 0 ? (processedCandles / replayDurationMs) * 1000 : undefined;
  }
}

function buildRandomAggregateResult(input: {
  options: HistoricalBacktestOptions;
  config: StrategyConfig;
  markets: HistoricalBacktestMarket[];
  results: BacktestResult[];
  startedAt: number;
  targetStartTime: number;
  targetEndTime: number;
  sampleWindowMs?: number;
  sampleMinWindowMs?: number;
  sampleMaxWindowMs?: number;
  sampleLookbackMs: number;
}): BacktestResult {
  const {
    options,
    config,
    markets,
    results,
    startedAt,
    targetStartTime,
    targetEndTime,
    sampleWindowMs,
    sampleMinWindowMs,
    sampleMaxWindowMs,
    sampleLookbackMs,
  } = input;

  if (results.length === 0) {
    throw new Error("Random-window backtest requires at least one completed sample.");
  }

  const samples = results.map<BacktestSampleSummary>((result, index) => {
    const rate = dailyRateForResult(result);
    return {
      index: index + 1,
      marketId: result.summary.marketId,
      symbol: result.summary.symbol,
      displaySymbol: result.summary.displaySymbol,
      startTime: result.summary.startTime,
      endTime: result.summary.endTime,
      durationMs: rate.durationMs,
      eventsProcessed: result.summary.eventsProcessed,
      candlesProcessed: result.summary.candlesProcessed,
      finalEquity: result.summary.finalEquity,
      netPnl: result.summary.netPnl,
      returnPct: result.summary.returnPct,
      riskAdjustedReturn: result.summary.riskAdjustedReturn,
      sharpeRatio: result.summary.sharpeRatio,
      netPnlPerDay: rate.netPnlPerDay,
      returnPctPerDay: rate.returnPctPerDay,
      perfectMarginLeverage: result.summary.perfectMarginLeverage,
      perfectMarginFinalEquity: result.summary.perfectMarginFinalEquity,
      perfectMarginNetPnl: result.summary.perfectMarginNetPnl,
      perfectMarginReturnPct: result.summary.perfectMarginReturnPct,
      perfectMarginCapturePct: result.summary.perfectMarginCapturePct,
      maxDrawdownPct: result.summary.maxDrawdownPct,
      tradeCount: result.summary.tradeCount,
      winRate: result.summary.winRate,
      stoppedEarly: result.summary.stoppedEarly,
      stopReason: result.summary.stopReason,
      survivedMs: result.summary.survivedMs,
    };
  });
  const cacheStats = summarizeCacheStats(results);
  const totalCandles = sumDefined(results.map((result) => result.summary.candlesProcessed));
  const totalReplayDurationMs = sumDefined(
    results.map((result) => result.summary.replayDurationMs),
  );
  const metrics = averageBotMetrics(results);
  const finalState = createAggregateFinalState(config, results, metrics);
  const returnValues = samples.map((sample) => sample.returnPct);
  const riskAdjustedReturnValues = samples
    .map((sample) => sample.riskAdjustedReturn)
    .filter((value): value is number => Number.isFinite(value));
  const sharpeRatioValues = samples
    .map((sample) => sample.sharpeRatio)
    .filter((value): value is number => Number.isFinite(value));
  const netPnlPerDayValues = samples.map((sample) => sample.netPnlPerDay);
  const returnPctPerDayValues = samples.map((sample) => sample.returnPctPerDay);
  const perfectMarginNetPnl = averageDefined(
    samples.map((sample) => sample.perfectMarginNetPnl),
  );

  return {
    summary: {
      symbol: options.symbol,
      marketId: options.marketId,
      displaySymbol: options.displaySymbol,
      source: "candles",
      startTime: Math.min(...samples.map((sample) => sample.startTime)),
      endTime: Math.max(...samples.map((sample) => sample.endTime)),
      targetStartTime,
      targetEndTime,
      eventsProcessed: sumDefined(samples.map((sample) => sample.eventsProcessed)),
      candlesProcessed: totalCandles,
      requests: cacheStats.requests,
      cacheHitCandles: cacheStats.cacheHitCandles,
      cacheMissCandles: cacheStats.cacheMissCandles,
      cacheFetchedCandles: cacheStats.cacheFetchedCandles,
      cacheSizeBytes: cacheStats.cacheSizeBytes,
      cacheEvictedBytes: cacheStats.cacheEvictedBytes,
      cacheEvictedFiles: cacheStats.cacheEvictedFiles,
      sampleCount: results.length,
      samplesProcessed: results.length,
      sampleWindowMs,
      sampleMinWindowMs,
      sampleMaxWindowMs,
      sampleLookbackMs,
      marketCount: markets.length,
      randomPairCount: Math.max(0, markets.length - 1),
      marketSymbols: markets.map((market) => market.symbol),
      profitableSamples: samples.filter((sample) => sample.returnPct > 0).length,
      wipedOutSamples: samples.filter((sample) => sample.stopReason === "wiped_out").length,
      bestReturnPct: Math.max(...returnValues),
      worstReturnPct: Math.min(...returnValues),
      netPnlPerDay: average(netPnlPerDayValues),
      returnPctPerDay: average(returnPctPerDayValues),
      bestNetPnlPerDay: Math.max(...netPnlPerDayValues),
      worstNetPnlPerDay: Math.min(...netPnlPerDayValues),
      bestReturnPctPerDay: Math.max(...returnPctPerDayValues),
      worstReturnPctPerDay: Math.min(...returnPctPerDayValues),
      perfectMarginLeverage: averageDefined(
        samples.map((sample) => sample.perfectMarginLeverage),
      ),
      perfectMarginFinalEquity: averageDefined(
        samples.map((sample) => sample.perfectMarginFinalEquity),
      ),
      perfectMarginNetPnl,
      perfectMarginReturnPct: averageDefined(
        samples.map((sample) => sample.perfectMarginReturnPct),
      ),
      perfectMarginCapturePct:
        perfectMarginNetPnl && perfectMarginNetPnl > 0
          ? (metrics.netPnl / perfectMarginNetPnl) * 100
          : undefined,
      stoppedEarly: false,
      stopReason: "completed",
      survivedMs: averageDefined(samples.map((sample) => sample.survivedMs)),
      durationMs: Date.now() - startedAt,
      replayDurationMs: totalReplayDurationMs,
      candlesPerSecond:
        totalReplayDurationMs > 0 ? (totalCandles / totalReplayDurationMs) * 1000 : undefined,
      finalEquity: metrics.equity,
      netPnl: metrics.netPnl,
      returnPct: metrics.returnPct,
      riskAdjustedReturn:
        riskAdjustedReturnValues.length > 0 ? average(riskAdjustedReturnValues) : undefined,
      sharpeRatio: sharpeRatioValues.length > 0 ? average(sharpeRatioValues) : undefined,
      maxDrawdownPct: metrics.maxDrawdownPct,
      tradeCount: metrics.tradeCount,
      winRate: metrics.winRate,
    },
    equityCurve: averageEquityCurves(results),
    orders: [],
    fills: [],
    finalState,
    samples,
  };
}

function buildRandomWindows(options: {
  count: number;
  targetStartTime: number;
  targetEndTime: number;
  minWindowMs: number;
  maxWindowMs: number;
  intervalMs: number;
}): RandomBacktestWindow[] {
  const earliestStartTime = alignUp(options.targetStartTime, options.intervalMs);
  const minWindowSteps = Math.max(1, Math.ceil(options.minWindowMs / options.intervalMs));
  const maxWindowSteps = Math.max(
    minWindowSteps,
    Math.floor(options.maxWindowMs / options.intervalMs),
  );
  const maxWindowMs = maxWindowSteps * options.intervalMs;
  const latestMaxStartTime = alignDown(
    options.targetEndTime - maxWindowMs + options.intervalMs,
    options.intervalMs,
  );

  if (latestMaxStartTime < earliestStartTime) {
    throw new Error("Random-window lookback must be at least as long as one sample window.");
  }

  const windows = Array.from({ length: options.count }, () => {
    const windowSteps = randomInteger(minWindowSteps, maxWindowSteps);
    const windowMs = windowSteps * options.intervalMs;
    const latestStartTime = alignDown(
      options.targetEndTime - windowMs + options.intervalMs,
      options.intervalMs,
    );
    const startSteps = Math.floor(
      (latestStartTime - earliestStartTime) / options.intervalMs,
    );
    const offsetSteps = Math.floor(Math.random() * (startSteps + 1));
    const startTime = earliestStartTime + offsetSteps * options.intervalMs;
    return {
      startTime,
      endTime: startTime + windowMs - options.intervalMs,
      windowMs,
    };
  });

  return windows.sort((a, b) => a.startTime - b.startTime);
}

function averageEquityCurves(results: BacktestResult[]): EquityPoint[] {
  const curves = results
    .map((result) => result.equityCurve)
    .filter((curve) => curve.length > 0);

  if (curves.length === 0) {
    return [];
  }

  const pointCount = Math.min(
    MAX_EQUITY_POINTS,
    Math.max(2, ...curves.map((curve) => curve.length)),
  );

  return Array.from({ length: pointCount }, (_, index) => {
    const position = pointCount === 1 ? 0 : index / (pointCount - 1);
    const points = curves.map((curve) => {
      const curveIndex = Math.min(
        curve.length - 1,
        Math.round(position * (curve.length - 1)),
      );
      return curve[curveIndex];
    });

    return {
      time: index,
      equity: average(points.map((point) => point.equity)),
      price: average(points.map((point) => point.price)),
    };
  });
}

function createAggregateFinalState(
  config: StrategyConfig,
  results: BacktestResult[],
  metrics: BotMetrics,
): PaperBotState {
  const state = createInitialBotState(config);
  const lastPrices = results.map((result) => result.finalState.lastPrice);
  const winningTrades = results.map((result) => result.finalState.winningTrades);
  const losingTrades = results.map((result) => result.finalState.losingTrades);

  state.quoteFree = metrics.equity;
  state.lastPrice = average(lastPrices);
  state.updatedAt = Date.now();
  state.realizedPnl = metrics.realizedPnl;
  state.feesPaid = metrics.feesPaid;
  state.winningTrades = average(winningTrades);
  state.losingTrades = average(losingTrades);
  state.metrics = metrics;

  return compactBacktestState(state, {
    maxReturnedOrders: 0,
    maxReturnedFills: 0,
  });
}

function averageBotMetrics(results: BacktestResult[]): BotMetrics {
  const metrics = results.map((result) => result.finalState.metrics);

  return {
    equity: average(metrics.map((item) => item.equity)),
    realizedPnl: average(metrics.map((item) => item.realizedPnl)),
    unrealizedPnl: average(metrics.map((item) => item.unrealizedPnl)),
    netPnl: average(metrics.map((item) => item.netPnl)),
    returnPct: average(metrics.map((item) => item.returnPct)),
    feesPaid: average(metrics.map((item) => item.feesPaid)),
    tradeCount: average(metrics.map((item) => item.tradeCount)),
    winningTrades: average(metrics.map((item) => item.winningTrades)),
    losingTrades: average(metrics.map((item) => item.losingTrades)),
    winRate: average(metrics.map((item) => item.winRate)),
    peakEquity: average(metrics.map((item) => item.peakEquity)),
    maxDrawdownPct: average(metrics.map((item) => item.maxDrawdownPct)),
    exposurePct: average(metrics.map((item) => item.exposurePct)),
  };
}

function summarizeCacheStats(results: BacktestResult[]): HistoricalCandleCacheStats {
  const stats = emptyCacheStats();
  for (const result of results) {
    mergeSummaryCacheStats(stats, result.summary);
  }
  return stats;
}

function mergeSummaryCacheStats(
  target: HistoricalCandleCacheStats,
  summary: BacktestResult["summary"],
): void {
  target.cacheHitCandles += summary.cacheHitCandles ?? 0;
  target.cacheMissCandles += summary.cacheMissCandles ?? 0;
  target.cacheFetchedCandles += summary.cacheFetchedCandles ?? 0;
  target.requests += summary.requests ?? 0;
  target.cacheSizeBytes = summary.cacheSizeBytes ?? target.cacheSizeBytes;
  target.cacheEvictedBytes += summary.cacheEvictedBytes ?? 0;
  target.cacheEvictedFiles += summary.cacheEvictedFiles ?? 0;
}

function mergeProgressCacheStats(
  completed: HistoricalCandleCacheStats,
  progress: BacktestProgressSnapshot | undefined,
): HistoricalCandleCacheStats {
  return {
    cacheHitCandles: completed.cacheHitCandles + (progress?.cacheHitCandles ?? 0),
    cacheMissCandles: completed.cacheMissCandles + (progress?.cacheMissCandles ?? 0),
    cacheFetchedCandles:
      completed.cacheFetchedCandles + (progress?.cacheFetchedCandles ?? 0),
    requests: completed.requests + (progress?.requests ?? 0),
    cacheSizeBytes: progress?.cacheSizeBytes ?? completed.cacheSizeBytes,
    cacheEvictedBytes: completed.cacheEvictedBytes + (progress?.cacheEvictedBytes ?? 0),
    cacheEvictedFiles: completed.cacheEvictedFiles + (progress?.cacheEvictedFiles ?? 0),
    freeBytes: progress ? completed.freeBytes : completed.freeBytes,
  };
}

function dailyRateForResult(result: BacktestResult): {
  durationMs: number;
  netPnlPerDay: number;
  returnPctPerDay: number;
} {
  const durationMs = Math.max(
    1,
    result.summary.survivedMs ??
      result.summary.endTime - result.summary.startTime,
  );
  const days = durationMs / DAY_MS;

  return {
    durationMs,
    netPnlPerDay: result.summary.netPnl / days,
    returnPctPerDay: result.summary.returnPct / days,
  };
}

function dailyRateForProgress(
  progress: BacktestProgressSnapshot,
  startingQuote: number,
): {
  netPnlPerDay: number;
  returnPctPerDay: number;
} {
  const durationMs = Math.max(
    1,
    progress.survivedMs ??
      (progress.processedStartTime && progress.processedEndTime
        ? progress.processedEndTime - progress.processedStartTime
        : progress.sampleWindowMs ?? DAY_MS),
  );
  const days = durationMs / DAY_MS;

  return {
    netPnlPerDay: (progress.equity - startingQuote) / days,
    returnPctPerDay: progress.returnPct / days,
  };
}

function clampInteger(
  value: number | undefined,
  fallback: number,
  min: number,
  max: number,
): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(min, Math.min(max, Math.round(value as number)));
}

function cappedMaxLeverage(
  configuredMaxLeverage: number,
  marketMaxLeverage: number | undefined,
): number {
  if (!Number.isFinite(marketMaxLeverage) || (marketMaxLeverage as number) < 1) {
    return configuredMaxLeverage;
  }

  return Math.min(configuredMaxLeverage, marketMaxLeverage as number);
}

function normalizeDurationMs(
  value: number | undefined,
  fallback: number,
  min: number,
): number {
  if (!Number.isFinite(value)) {
    return Math.max(min, fallback);
  }

  return Math.max(min, Math.round(value as number));
}

function randomInteger(min: number, max: number): number {
  if (max <= min) {
    return min;
  }

  return min + Math.floor(Math.random() * (max - min + 1));
}

function estimateRangeCandles(
  startTime: number,
  endTime: number,
  intervalMs: number,
): number {
  const first = alignUp(startTime, intervalMs);
  const last = alignDown(endTime, intervalMs);
  if (last < first) {
    return 1;
  }

  return Math.floor((last - first) / intervalMs) + 1;
}

function average(values: number[]): number {
  return values.length === 0 ? 0 : sumDefined(values) / values.length;
}

function averageDefined(values: Array<number | undefined>): number | undefined {
  const defined = values.filter((value): value is number => Number.isFinite(value));
  return defined.length > 0 ? average(defined) : undefined;
}

function sumDefined(values: Array<number | undefined>): number {
  let sum = 0;
  for (const value of values) {
    if (Number.isFinite(value)) {
      sum += value as number;
    }
  }
  return sum;
}

function alignUp(value: number, intervalMs: number): number {
  return Math.ceil(value / intervalMs) * intervalMs;
}

function alignDown(value: number, intervalMs: number): number {
  return Math.floor(value / intervalMs) * intervalMs;
}

function replayCandle(
  bot: SimulatedTradingBot,
  candle: Candle,
  perfectMargin?: PerfectMarginBenchmarkAccumulator,
): void {
  const options = {
    collectEvents: false,
    updateMetrics: false,
  };

  for (const tick of candleReplayTicks(candle)) {
    observePerfectMarginPrice(perfectMargin, tick.price);
    bot.onTick(tick, options);
  }
}

function candleReplayTicks(candle: Candle): Array<{
  symbol: string;
  eventTime: number;
  price: number;
  quantity: number;
}> {
  const duration = Math.max(1, candle.closeTime - candle.openTime);

  return [
    {
      symbol: candle.symbol,
      eventTime: candle.openTime,
      price: candle.open,
      quantity: candle.volume * 0.2,
    },
    {
      symbol: candle.symbol,
      eventTime: candle.openTime + duration * 0.33,
      price: candle.high,
      quantity: candle.volume * 0.25,
    },
    {
      symbol: candle.symbol,
      eventTime: candle.openTime + duration * 0.66,
      price: candle.low,
      quantity: candle.volume * 0.25,
    },
    {
      symbol: candle.symbol,
      eventTime: candle.closeTime,
      price: candle.close,
      quantity: candle.volume * 0.3,
    },
  ];
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

  return {
    perfectMarginLeverage: benchmark.leverage,
    perfectMarginFinalEquity: finalEquity,
    perfectMarginNetPnl: netPnl,
    perfectMarginReturnPct: returnPct,
    perfectMarginCapturePct: netPnl > 0 ? (actualNetPnl / netPnl) * 100 : undefined,
  };
}

function parseKline(symbol: string, interval: string, row: unknown[]): Candle {
  return {
    symbol,
    interval,
    openTime: Number(row[0]),
    open: Number(row[1]),
    high: Number(row[2]),
    low: Number(row[3]),
    close: Number(row[4]),
    volume: Number(row[5]),
    closeTime: Number(row[6]),
    closed: true,
  };
}

export async function fetchKlines(options: {
  venue: StreamVenue;
  symbol: string;
  interval: string;
  startTime: number;
  endTime: number;
  limit: number;
  signal?: AbortSignal;
}): Promise<Candle[]> {
  const url = new URL(klineEndpointForVenue(options.venue));
  const endTime =
    options.venue === "coinm-futures"
      ? Math.min(options.endTime, options.startTime + 200 * DAY_MS - 1)
      : options.endTime;
  url.search = new URLSearchParams({
    symbol: options.symbol,
    interval: options.interval,
    startTime: String(options.startTime),
    endTime: String(endTime),
    limit: String(options.limit),
  }).toString();

  for (let attempt = 1; attempt <= 5; attempt += 1) {
    const response = await fetch(url, {
      signal: requestSignal(options.signal, 20_000),
    });

    if (response.ok) {
      const rows = (await response.json()) as unknown[][];
      return rows.map((row) => parseKline(options.symbol, options.interval, row));
    }

    const body = await response.text();
    const retryable =
      response.status === 418 || response.status === 429 || response.status >= 500;
    if (retryable && attempt < 5) {
      await delay(1000 * attempt);
      continue;
    }

    throw new Error(
      `Binance klines request failed: HTTP ${response.status} ${body.slice(0, 240)}`,
    );
  }

  return [];
}

function requestSignal(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
}

function klineEndpointForVenue(venue: StreamVenue): string {
  if (venue === "spot") {
    return "https://api.binance.com/api/v3/klines";
  }
  if (venue === "usdm-futures") {
    return "https://fapi.binance.com/fapi/v1/klines";
  }
  if (venue === "coinm-futures") {
    return "https://dapi.binance.com/dapi/v1/klines";
  }

  return "https://eapi.binance.com/eapi/v1/klines";
}

export function intervalToMs(interval: string): number {
  const match = /^(\d+)([mhdw])$/.exec(interval);
  if (!match) {
    return 60_000;
  }

  const value = Number(match[1]);
  const unit = match[2];
  const multipliers: Record<string, number> = {
    m: 60_000,
    h: 60 * 60_000,
    d: 24 * 60 * 60_000,
    w: 7 * 24 * 60 * 60_000,
  };

  return value * multipliers[unit];
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function emptyCacheStats(): HistoricalCandleCacheStats {
  return {
    cacheHitCandles: 0,
    cacheMissCandles: 0,
    cacheFetchedCandles: 0,
    requests: 0,
    cacheSizeBytes: 0,
    cacheEvictedBytes: 0,
    cacheEvictedFiles: 0,
    freeBytes: 0,
  };
}
