import {
  SimulatedTradingBot,
  createInitialBotState,
  createStrategyConfig,
  type BacktestPreset,
  type BacktestProgressSnapshot,
  type BacktestResult,
  type Candle,
  type EquityPoint,
  type PaperBotState,
  type PriceTick,
  type StrategyConfig,
} from "@trading/bot-algo";
import {
  HistoricalCandleCache,
  type HistoricalCandleCacheStats,
} from "./historical-candle-cache.js";

const KLINE_LIMIT = 1000;
const WIPEOUT_EQUITY_FRACTION = 0.01;
const MAX_EQUITY_POINTS = 800;

const periodDurations: Record<Extract<BacktestPreset, "week" | "month" | "year">, number> = {
  week: 7 * 24 * 60 * 60 * 1000,
  month: 30 * 24 * 60 * 60 * 1000,
  year: 365 * 24 * 60 * 60 * 1000,
};

export interface HistoricalBacktestOptions {
  id: string;
  preset: Extract<BacktestPreset, "week" | "month" | "year">;
  symbol: string;
  interval: string;
  config: StrategyConfig;
  cache: {
    dataDir: string;
    maxBytes: number;
    minFreeBytes: number;
  };
}

export async function runHistoricalCandleBacktest(
  options: HistoricalBacktestOptions,
  onProgress: (progress: BacktestProgressSnapshot) => void,
): Promise<BacktestResult> {
  const targetEndTime = Date.now();
  const targetStartTime = targetEndTime - periodDurations[options.preset];
  const intervalMs = intervalToMs(options.interval);
  const estimatedCandles = Math.max(
    1,
    Math.ceil((targetEndTime - targetStartTime) / intervalMs),
  );
  const config = createStrategyConfig({
    ...options.config,
    symbol: options.symbol,
  });
  const bot = new SimulatedTradingBot(createInitialBotState(config));
  const cache = new HistoricalCandleCache({
    dataDir: options.cache.dataDir,
    symbol: options.symbol,
    interval: options.interval,
    intervalMs,
    maxBytes: options.cache.maxBytes,
    minFreeBytes: options.cache.minFreeBytes,
  });
  const equityCurve: EquityPoint[] = [];
  const sampleEvery = Math.max(1, Math.floor(estimatedCandles / MAX_EQUITY_POINTS));
  const startedAt = Date.now();
  const wipeoutEquity = config.startingQuote * WIPEOUT_EQUITY_FRACTION;

  let processedCandles = 0;
  let cacheStats = emptyCacheStats();
  let processedStartTime: number | undefined;
  let processedEndTime: number | undefined;
  let stopReason: "completed" | "wiped_out" = "completed";
  let finalState = bot.snapshot();
  let survivedMs: number | undefined;

  emitProgress("running", "Checking historical candle cache");

  cacheStats = await cache.ensureRange(
    targetStartTime,
    targetEndTime,
    (request) =>
      fetchKlines({
        symbol: options.symbol,
        interval: options.interval,
        startTime: request.startTime,
        endTime: request.endTime,
        limit: request.limit,
      }),
    (stats) => {
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

  outer: for await (const candles of cache.readRangeBatches(targetStartTime, targetEndTime)) {
    for (const candle of candles) {
      if (candle.openTime < targetStartTime || candle.openTime > targetEndTime) {
        continue;
      }

      processedStartTime ??= candle.openTime;
      processedEndTime = candle.closeTime;

      for (const tick of candleToSyntheticTicks(candle)) {
        bot.onTick(tick);
      }

      processedCandles += 1;
      finalState = bot.snapshot();

      if (processedCandles % sampleEvery === 0) {
        equityCurve.push({
          time: candle.closeTime,
          equity: finalState.metrics.equity,
          price: candle.close,
        });
      }

      if (
        processedCandles >= config.slowWindow &&
        finalState.metrics.equity <= wipeoutEquity
      ) {
        stopReason = "wiped_out";
        break outer;
      }
    }

    emitProgress("running", `Processed ${processedCandles.toLocaleString()} candles`);
  }

  finalState = bot.snapshot();
  if (processedEndTime && equityCurve.at(-1)?.time !== processedEndTime) {
    equityCurve.push({
      time: processedEndTime,
      equity: finalState.metrics.equity,
      price: finalState.lastPrice,
    });
  }

  const durationMs = Date.now() - startedAt;
  survivedMs =
    processedStartTime && processedEndTime
      ? processedEndTime - processedStartTime
      : undefined;
  const result: BacktestResult = {
    summary: {
      symbol: options.symbol,
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

  return result;

  function emitProgress(
    status: BacktestProgressSnapshot["status"],
    message: string,
  ): void {
    const state = finalState;
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
      percent: Math.min(100, (processedCandles / estimatedCandles) * 100),
      equity: state.metrics.equity,
      returnPct: state.metrics.returnPct,
      stopReason: status === "completed" ? stopReason : undefined,
      survivedMs: currentSurvivedMs,
      message,
    });
  }
}

function candleToSyntheticTicks(candle: Candle): PriceTick[] {
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

async function fetchKlines(options: {
  symbol: string;
  interval: string;
  startTime: number;
  endTime: number;
  limit: number;
}): Promise<Candle[]> {
  const url = new URL("https://api.binance.com/api/v3/klines");
  url.search = new URLSearchParams({
    symbol: options.symbol,
    interval: options.interval,
    startTime: String(options.startTime),
    endTime: String(options.endTime),
    limit: String(options.limit),
  }).toString();

  for (let attempt = 1; attempt <= 5; attempt += 1) {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(20_000),
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

function intervalToMs(interval: string): number {
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
