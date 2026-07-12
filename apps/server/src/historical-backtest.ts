import {
  aggregateExtremaOrderMassSummaries,
  compactBacktestState,
  createInitialBotState,
  createPeakValleyBotConfig,
  createStrategyConfig,
  peakValleyWarmupSamples,
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
  type CandleTimeRange,
  type HistoricalCandleCacheStats,
} from "./historical-candle-cache.js";
import type { StreamVenue } from "./binance-markets.js";
import { runBotBacktestFromCandles } from "./bot-backtest.js";

const MAX_EQUITY_POINTS = 800;
const REPLAY_PROGRESS_CANDLES = 10_000;
const MAX_RANDOM_PRELOADED_CANDLES = 2_000_000;
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
  historicalStartTime?: number;
  historicalRangeMs?: number;
  randomSampleCount?: number;
  randomWindowMs?: number;
  randomMinWindowMs?: number;
  randomMaxWindowMs?: number;
  randomLookbackMs?: number;
  randomMarkets?: HistoricalBacktestMarket[];
  randomPairCount?: number;
  extremaSmaWindowMs?: number;
  cancelSignal?: AbortSignal;
}

interface HistoricalRangeBacktestOptions extends HistoricalBacktestOptions {
  targetStartTime: number;
  targetEndTime: number;
  cacheAlreadyEnsured?: boolean;
  replayCandles?: readonly Candle[];
  replayEndIndex?: number;
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

interface RandomReplayWindow extends RandomBacktestWindow { replayEndIndex?: number }

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
  const configuredStartTime = Number(options.historicalStartTime);
  const durationMs =
    options.preset === "last-x"
      ? normalizeDurationMs(
          options.historicalRangeMs,
          DEFAULT_HISTORICAL_RANGE_MS,
          intervalMs,
        )
      : periodDurations[options.preset];
  const fallbackStartTime = targetEndTime - durationMs;
  const explicitStartTime =
    Number.isFinite(configuredStartTime) && configuredStartTime > 0
      ? configuredStartTime
      : undefined;
  const targetStartTime =
    explicitStartTime && explicitStartTime < targetEndTime
      ? explicitStartTime
      : fallbackStartTime;

  return runBotHistoricalRangeBacktest(
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
  const warmupMs = historicalWarmupSamples(config, intervalMs) * intervalMs;
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
    const completedMarketCacheStats = { ...cacheStats };
    const marketCacheStats = await ensureRandomWindowCache(
      options,
      market,
      windows,
      intervalMs,
      warmupMs,
      (stats) => {
        cacheStats = mergedCacheStats(completedMarketCacheStats, stats);
        emitAggregateProgress(`Checking ${market.displaySymbol} random-window cache`);
      },
    );
    cacheStats = mergedCacheStats(completedMarketCacheStats, marketCacheStats);
    const preloadedCandles = await preloadRandomWindowCandles(
      options,
      market,
      windows,
      intervalMs,
      warmupMs,
      (loadedCandles, estimatedCandlesToLoad) => {
        const suffix =
          estimatedCandlesToLoad > 0
            ? ` ${loadedCandles.toLocaleString()}/${estimatedCandlesToLoad.toLocaleString()}`
            : "";
        emitAggregateProgress(
          `Loading ${market.displaySymbol} random-window candles${suffix}`,
        );
      },
    );
    const replayWindows: RandomReplayWindow[] = preloadedCandles
      ? windows.map((window) => ({
          ...window,
          replayEndIndex: upperBoundCandleOpenTime(preloadedCandles, window.endTime),
        }))
      : windows;

    for (let windowIndex = 0; windowIndex < replayWindows.length; windowIndex += 1) {
      throwIfCancelled(options.cancelSignal);
      const currentSample = marketIndex * sampleCount + windowIndex + 1;
      const currentWindowSample = windowIndex + 1;
      const window = replayWindows[windowIndex];
      const result = await runBotHistoricalRangeBacktest(
        {
          ...options,
          ...market,
          config,
          randomPairCount,
          targetStartTime: window.startTime,
          targetEndTime: window.endTime,
          cacheAlreadyEnsured: true,
          replayCandles: preloadedCandles,
          replayEndIndex: window.replayEndIndex,
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
      result.candleChart = undefined;
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
    cacheStats,
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

async function ensureRandomWindowCache(
  options: HistoricalBacktestOptions,
  market: HistoricalBacktestMarket,
  windows: RandomBacktestWindow[],
  intervalMs: number,
  warmupMs: number,
  onProgress: (stats: HistoricalCandleCacheStats) => void,
): Promise<HistoricalCandleCacheStats> {
  throwIfCancelled(options.cancelSignal);
  const cache = new HistoricalCandleCache({
    dataDir: options.cache.dataDir,
    marketKey: market.marketKey,
    symbol: market.symbol,
    interval: options.interval,
    intervalMs,
    maxBytes: options.cache.maxBytes,
    minFreeBytes: options.cache.minFreeBytes,
  });
  const ranges: CandleTimeRange[] = windows.map((window) => ({
    startTime: window.startTime - warmupMs,
    endTime: window.endTime,
  }));

  return cache.ensureRanges(
    ranges,
    (request) =>
      fetchKlines({
        venue: market.venue,
        symbol: market.symbol,
        interval: options.interval,
        startTime: request.startTime,
        endTime: request.endTime,
        limit: request.limit,
        signal: options.cancelSignal,
      }),
    (stats) => {
      throwIfCancelled(options.cancelSignal);
      onProgress(stats);
    },
  );
}

async function preloadRandomWindowCandles(
  options: HistoricalBacktestOptions,
  market: HistoricalBacktestMarket,
  windows: RandomBacktestWindow[],
  intervalMs: number,
  warmupMs: number,
  onProgress: (loadedCandles: number, estimatedCandles: number) => void,
): Promise<readonly Candle[] | undefined> {
  const ranges = mergeCandleTimeRanges(windows.map((window) => ({
    startTime: window.startTime - warmupMs,
    endTime: window.endTime,
  })), intervalMs);
  const estimatedCandles = sumDefined(
    ranges.map((range) => estimateRangeCandles(range.startTime, range.endTime, intervalMs)),
  );
  if (estimatedCandles > MAX_RANDOM_PRELOADED_CANDLES) {
    return undefined;
  }

  throwIfCancelled(options.cancelSignal);
  const cache = new HistoricalCandleCache({
    dataDir: options.cache.dataDir,
    marketKey: market.marketKey,
    symbol: market.symbol,
    interval: options.interval,
    intervalMs,
    maxBytes: options.cache.maxBytes,
    minFreeBytes: options.cache.minFreeBytes,
  });
  const candles: Candle[] = [];

  for (const range of ranges) {
    for await (const batch of cache.readRangeBatches(
      range.startTime,
      range.endTime,
      REPLAY_PROGRESS_CANDLES,
    )) {
      for (const candle of batch) {
        if (candle.openTime >= range.startTime && candle.openTime <= range.endTime) {
          candles.push(candle);
        }
      }
      throwIfCancelled(options.cancelSignal);
      onProgress(candles.length, estimatedCandles);
    }
  }

  return candles;
}

async function runBotHistoricalRangeBacktest(
  options: HistoricalRangeBacktestOptions,
  onProgress: (progress: BacktestProgressSnapshot) => void,
): Promise<BacktestResult> {
  throwIfCancelled(options.cancelSignal);
  const { targetStartTime, targetEndTime } = options;
  const intervalMs = intervalToMs(options.interval);
  const estimatedCandles = estimateRangeCandles(targetStartTime, targetEndTime, intervalMs);
  const config = createStrategyConfig({
    ...options.config,
    symbol: options.symbol,
    baseAsset: options.baseAsset ?? options.config.baseAsset,
    quoteAsset: options.quoteAsset ?? options.config.quoteAsset,
    maxLeverage: cappedMaxLeverage(options.config.maxLeverage, options.maxLeverage),
  });
  const warmupSamples = historicalWarmupSamples(config, intervalMs);
  const firstTargetTime = alignUp(targetStartTime, intervalMs);
  const warmupStartTime = firstTargetTime - warmupSamples * intervalMs;
  const cache = new HistoricalCandleCache({
    dataDir: options.cache.dataDir,
    marketKey: options.marketKey,
    symbol: options.symbol,
    interval: options.interval,
    intervalMs,
    maxBytes: options.cache.maxBytes,
    minFreeBytes: options.cache.minFreeBytes,
  });
  let cacheStats = emptyCacheStats();
  const startedAt = Date.now();
  const warmup: Candle[] = [];
  const candles: Candle[] = [];

  emit("Checking historical candle cache");
  if (!options.cacheAlreadyEnsured) {
    cacheStats = await cache.ensureRange(
      warmupStartTime,
      targetEndTime,
      (request) => fetchKlines({
        venue: options.venue,
        symbol: options.symbol,
        interval: options.interval,
        startTime: request.startTime,
        endTime: request.endTime,
        limit: request.limit,
        signal: options.cancelSignal,
      }),
      (stats) => {
        cacheStats = stats;
        emit("Caching historical candles");
      },
    );
  }
  throwIfCancelled(options.cancelSignal);

  if (options.replayCandles) {
    const start = lowerBoundCandleOpenTime(options.replayCandles, warmupStartTime);
    const end = Math.min(options.replayCandles.length, options.replayEndIndex ?? options.replayCandles.length);
    for (let index = start; index < end; index += 1) {
      const candle = options.replayCandles[index];
      collect(candle);
    }
  } else {
    for await (const batch of cache.readRangeBatches(warmupStartTime, targetEndTime, REPLAY_PROGRESS_CANDLES)) {
      for (const candle of batch) collect(candle);
      throwIfCancelled(options.cancelSignal);
      emit(`Loaded ${candles.length.toLocaleString()} candles`);
    }
  }
  if (candles.length === 0) throw new Error("Historical backtest loaded no candles.");

  emit(`Replaying ${candles.length.toLocaleString()} candles`);
  const result = await runBotBacktestFromCandles(candles, {
    config,
    warmup,
    extremaSmaWindowMs: options.extremaSmaWindowMs,
  });
  Object.assign(result.summary, {
    marketId: options.marketId,
    displaySymbol: options.displaySymbol,
    targetStartTime,
    targetEndTime,
    requests: cacheStats.requests,
    cacheHitCandles: cacheStats.cacheHitCandles,
    cacheMissCandles: cacheStats.cacheMissCandles,
    cacheFetchedCandles: cacheStats.cacheFetchedCandles,
    cacheSizeBytes: cacheStats.cacheSizeBytes,
    cacheEvictedBytes: cacheStats.cacheEvictedBytes,
    cacheEvictedFiles: cacheStats.cacheEvictedFiles,
    survivedMs: result.summary.endTime - result.summary.startTime,
  });
  emit("Backtest completed", result);
  return result;

  function collect(candle: Candle): void {
    if (candle.openTime >= warmupStartTime && candle.openTime < firstTargetTime) {
      warmup.push(candle);
    } else if (candle.openTime >= targetStartTime && candle.openTime <= targetEndTime) {
      candles.push(candle);
    }
  }

  function emit(message: string, result?: BacktestResult): void {
    const processed = result?.summary.candlesProcessed ?? candles.length;
    onProgress({
      id: options.id,
      preset: options.preset,
      status: result ? "completed" : "running",
      source: "candles",
      startedAt,
      updatedAt: Date.now(),
      targetStartTime,
      targetEndTime,
      processedStartTime: candles[0]?.openTime,
      processedEndTime: candles.at(-1)?.closeTime,
      processedCandles: processed,
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
      percent: result ? 100 : Math.min(95, estimatedCandles > 0 ? candles.length / estimatedCandles * 90 : 0),
      equity: result?.summary.finalEquity ?? config.startingQuote,
      returnPct: result?.summary.returnPct ?? 0,
      stopReason: result?.summary.stopReason,
      survivedMs: result ? result.summary.endTime - result.summary.startTime : undefined,
      candlesPerSecond: result?.summary.candlesPerSecond,
      message,
      result,
    });
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
  cacheStats?: HistoricalCandleCacheStats;
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
    cacheStats: inputCacheStats,
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
      backtestSharpeRatio: result.summary.backtestSharpeRatio,
      netPnlPerDay: rate.netPnlPerDay,
      returnPctPerDay: rate.returnPctPerDay,
      perfectMarginLeverage: result.summary.perfectMarginLeverage,
      perfectMarginFinalEquity: result.summary.perfectMarginFinalEquity,
      perfectMarginNetPnl: result.summary.perfectMarginNetPnl,
      perfectMarginReturnPct: result.summary.perfectMarginReturnPct,
      perfectMarginCapturePct: result.summary.perfectMarginCapturePct,
      perfectMarginCompoundedFinalEquity:
        result.summary.perfectMarginCompoundedFinalEquity,
      perfectMarginCompoundedNetPnl: result.summary.perfectMarginCompoundedNetPnl,
      perfectMarginCompoundedReturnPct:
        result.summary.perfectMarginCompoundedReturnPct,
      perfectMarginCompoundedCapturePct:
        result.summary.perfectMarginCompoundedCapturePct,
      maxDrawdownPct: result.summary.maxDrawdownPct,
      maxEntryLeverage: result.summary.maxEntryLeverage,
      maxEffectiveLeverage: result.summary.maxEffectiveLeverage,
      tradeCount: result.summary.tradeCount,
      winRate: result.summary.winRate,
      closedPositionCount: result.summary.closedPositionCount,
      profitableClosedPositionCount: result.summary.profitableClosedPositionCount,
      profitableClosedPositionRate: result.summary.profitableClosedPositionRate,
      liquidatedPositionCount: result.summary.liquidatedPositionCount,
      stoppedEarly: result.summary.stoppedEarly,
      stopReason: result.summary.stopReason,
      survivedMs: result.summary.survivedMs,
      extremaOrderMass: result.summary.extremaOrderMass,
    };
  });
  const cacheStats = mergedCacheStats(
    inputCacheStats ?? emptyCacheStats(),
    summarizeCacheStats(results),
  );
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
  const backtestSharpeRatioValues = samples
    .map((sample) => sample.backtestSharpeRatio)
    .filter((value): value is number => Number.isFinite(value));
  const netPnlPerDayValues = samples.map((sample) => sample.netPnlPerDay);
  const returnPctPerDayValues = samples.map((sample) => sample.returnPctPerDay);
  const perfectMarginNetPnl = averageDefined(
    samples.map((sample) => sample.perfectMarginNetPnl),
  );
  const perfectMarginCompoundedNetPnl = averageDefined(
    samples.map((sample) => sample.perfectMarginCompoundedNetPnl),
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
      liquidatedPositionCount: average(
        samples.map((sample) => sample.liquidatedPositionCount),
      ),
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
      perfectMarginCompoundedFinalEquity: averageDefined(
        samples.map((sample) => sample.perfectMarginCompoundedFinalEquity),
      ),
      perfectMarginCompoundedNetPnl,
      perfectMarginCompoundedReturnPct: averageDefined(
        samples.map((sample) => sample.perfectMarginCompoundedReturnPct),
      ),
      perfectMarginCompoundedCapturePct:
        perfectMarginCompoundedNetPnl && perfectMarginCompoundedNetPnl > 0
          ? (metrics.netPnl / perfectMarginCompoundedNetPnl) * 100
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
      backtestSharpeRatio:
        backtestSharpeRatioValues.length > 0
          ? average(backtestSharpeRatioValues)
          : undefined,
      maxDrawdownPct: metrics.maxDrawdownPct,
      maxEntryLeverage: metrics.maxEntryLeverage,
      maxEffectiveLeverage: metrics.maxEffectiveLeverage,
      tradeCount: metrics.tradeCount,
      winRate: metrics.winRate,
      closedPositionCount: average(samples.map((sample) => sample.closedPositionCount)),
      profitableClosedPositionCount: average(
        samples.map((sample) => sample.profitableClosedPositionCount),
      ),
      profitableClosedPositionRate: average(
        samples.map((sample) => sample.profitableClosedPositionRate),
      ),
      extremaOrderMass: aggregateExtremaOrderMassSummaries(
        results.map((result) => result.summary.extremaOrderMass),
      ),
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

function mergeCandleTimeRanges(
  ranges: CandleTimeRange[],
  intervalMs: number,
): CandleTimeRange[] {
  const sorted = ranges
    .filter((range) => range.endTime >= range.startTime)
    .map((range) => ({
      startTime: alignUp(range.startTime, intervalMs),
      endTime: alignDown(range.endTime, intervalMs),
    }))
    .filter((range) => range.endTime >= range.startTime)
    .sort((left, right) => left.startTime - right.startTime || left.endTime - right.endTime);
  const merged: CandleTimeRange[] = [];

  for (const range of sorted) {
    const previous = merged[merged.length - 1];
    if (!previous || range.startTime > previous.endTime + intervalMs) {
      merged.push({ ...range });
      continue;
    }

    previous.endTime = Math.max(previous.endTime, range.endTime);
  }

  return merged;
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
    let equity = 0;
    let price = 0;

    for (const curve of curves) {
      const curveIndex = Math.min(
        curve.length - 1,
        Math.round(position * (curve.length - 1)),
      );
      equity += curve[curveIndex].equity;
      price += curve[curveIndex].price;
    }

    return {
      time: index,
      equity: equity / curves.length,
      price: price / curves.length,
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
    maxEntryLeverage: average(metrics.map((item) => item.maxEntryLeverage)),
    maxEffectiveLeverage: average(metrics.map((item) => item.maxEffectiveLeverage)),
    avgExitGridSpan: average(metrics.map((item) => item.avgExitGridSpan)),
    avgExitGridOrderCount: average(metrics.map((item) => item.avgExitGridOrderCount)),
    exitGridSpanCount: average(metrics.map((item) => item.exitGridSpanCount)),
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

function mergedCacheStats(
  left: HistoricalCandleCacheStats,
  right: HistoricalCandleCacheStats,
): HistoricalCandleCacheStats {
  return {
    cacheHitCandles: left.cacheHitCandles + right.cacheHitCandles,
    cacheMissCandles: left.cacheMissCandles + right.cacheMissCandles,
    cacheFetchedCandles: left.cacheFetchedCandles + right.cacheFetchedCandles,
    requests: left.requests + right.requests,
    cacheSizeBytes: right.cacheSizeBytes || left.cacheSizeBytes,
    cacheEvictedBytes: left.cacheEvictedBytes + right.cacheEvictedBytes,
    cacheEvictedFiles: left.cacheEvictedFiles + right.cacheEvictedFiles,
    freeBytes: right.freeBytes || left.freeBytes,
  };
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
    freeBytes: completed.freeBytes,
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

function lowerBoundCandleOpenTime(candles: readonly Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (candles[mid].openTime < time) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function upperBoundCandleOpenTime(candles: readonly Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (candles[mid].openTime <= time) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
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
  endpoint?: string;
  signal?: AbortSignal;
}): Promise<Candle[]> {
  const url = new URL(options.endpoint ?? klineEndpointForVenue(options.venue));
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

export function historicalWarmupSamples(config: StrategyConfig, intervalMs: number): number {
  return peakValleyWarmupSamples(createPeakValleyBotConfig(config, intervalMs).strategy);
}

export function intervalToMs(interval: string): number {
  const match = /^(\d+)([smhdw])$/.exec(interval);
  if (!match) {
    return 60_000;
  }

  const value = Number(match[1]);
  const unit = match[2];
  const multipliers: Record<string, number> = {
    s: 1_000,
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
