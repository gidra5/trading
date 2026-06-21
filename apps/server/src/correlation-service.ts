import fs from "node:fs/promises";
import path from "node:path";
import WebSocket from "ws";
import type { Candle } from "@trading/bot-algo";
import {
  getMarketStorageKey,
  type BinanceMarketListing,
  type StreamVenue,
} from "./binance-markets.js";
import {
  fetchKlines,
  intervalToMs,
} from "./historical-backtest.js";
import {
  HistoricalCandleCache,
  type HistoricalCandleCacheStats,
} from "./historical-candle-cache.js";

const CORRELATION_CACHE_VERSION = 1;
const STREAM_CHUNK_SIZE = 80;
const PENDING_RETURN_BUCKET_LIMIT = 4_000;
const CATEGORY_MIN_MARKETS = 3;

export type CorrelationStatus = "idle" | "running" | "ready" | "failed";

export interface CorrelationEntry {
  marketId: string;
  symbol: string;
  displaySymbol: string;
  baseAsset: string;
  quoteAsset: string;
  venue: string;
  correlation?: number;
  samples: number;
  startTime?: number;
  endTime?: number;
  updatedAt?: number;
}

export interface CorrelationSnapshot {
  status: CorrelationStatus;
  focalMarketId?: string;
  focalSymbol?: string;
  focalDisplaySymbol?: string;
  interval?: string;
  lookbackMs?: number;
  marketCount: number;
  expectedPairs: number;
  calculatedPairs: number;
  processedMarkets: number;
  requests: number;
  cacheHitCandles: number;
  cacheMissCandles: number;
  cacheFetchedCandles: number;
  cacheLoaded: boolean;
  truncated: boolean;
  startedAt?: number;
  updatedAt?: number;
  startTime?: number;
  endTime?: number;
  streamConnected: boolean;
  message: string;
  error?: string;
  entries: CorrelationEntry[];
}

export interface CorrelationServiceOptions {
  dataDir: string;
  interval: string;
  lookbackMs: number;
  maxMarkets: number;
  historicalCache: {
    maxBytes: number;
    minFreeBytes: number;
  };
}

interface RunningCorrelationStats {
  samples: number;
  sumX: number;
  sumY: number;
  sumXX: number;
  sumYY: number;
  sumXY: number;
}

interface CorrelationPairState {
  key: string;
  marketAId: string;
  marketBId: string;
  stats: RunningCorrelationStats;
  startTime?: number;
  endTime?: number;
  updatedAt?: number;
}

interface CorrelationVectorState {
  focalMarketId: string;
  peerMarketIds: string[];
  pairKeys: string[];
  marketIds: string[];
  startTime: number;
  endTime: number;
  lookbackMs: number;
  createdAt: number;
  updatedAt: number;
  truncated: boolean;
}

interface MarketReturnSeries {
  market: BinanceMarketListing;
  returns: Map<number, number>;
  lastClose?: number;
  lastReturnOpenTime?: number;
  cacheStats: HistoricalCandleCacheStats;
}

interface CachedCorrelationVector {
  version: 1;
  focalMarketId: string;
  interval: string;
  lookbackMs: number;
  marketIds: string[];
  peerMarketIds: string[];
  startTime: number;
  endTime: number;
  createdAt: number;
  updatedAt: number;
  truncated: boolean;
  pairs: Array<{
    key: string;
    marketAId: string;
    marketBId: string;
    stats: RunningCorrelationStats;
    startTime?: number;
    endTime?: number;
    updatedAt?: number;
  }>;
  lastCloses: Array<{
    marketId: string;
    close: number;
    lastReturnOpenTime?: number;
  }>;
}

interface KlineStreamStatus {
  connected: boolean;
  message: string;
}

export class CorrelationService {
  private readonly intervalMs: number;
  private readonly cacheDir: string;
  private snapshots = new Map<string, CorrelationSnapshot>();
  private vectors = new Map<string, CorrelationVectorState>();
  private marketsById = new Map<string, BinanceMarketListing>();
  private marketIdBySymbol = new Map<string, string>();
  private pairStatsByKey = new Map<string, CorrelationPairState>();
  private pairKeysByMarketId = new Map<string, Set<string>>();
  private lastCloseByMarketId = new Map<string, number>();
  private lastReturnOpenTimeByMarketId = new Map<string, number>();
  private pendingReturnsByOpenTime = new Map<number, Map<string, number>>();
  private saveTimers = new Map<string, NodeJS.Timeout>();
  private activeRun?: {
    focalMarketId: string;
    abortController: AbortController;
  };
  private stream?: CorrelationKlineStream;

  constructor(private readonly options: CorrelationServiceOptions) {
    this.intervalMs = intervalToMs(options.interval);
    this.cacheDir = path.join(options.dataDir, "correlations");
  }

  startVector(
    focalMarket: BinanceMarketListing,
    candidateMarkets: BinanceMarketListing[],
    onUpdate: () => void,
    forceRefresh = false,
  ): CorrelationSnapshot {
    const universe = this.normalizeUniverse(focalMarket, candidateMarkets);
    const marketIds = universe.markets.map((market) => market.id);
    const existing = this.snapshots.get(focalMarket.id);
    if (
      existing?.status === "running" &&
      sameStringList(existing.entries.map((entry) => entry.marketId), marketIds.slice(1))
    ) {
      return existing;
    }

    this.activeRun?.abortController.abort();
    this.stream?.stop();
    this.stream = undefined;

    const startedAt = Date.now();
    const snapshot: CorrelationSnapshot = {
      status: "running",
      focalMarketId: focalMarket.id,
      focalSymbol: focalMarket.symbol,
      focalDisplaySymbol: focalMarket.displaySymbol,
      interval: this.options.interval,
      lookbackMs: this.options.lookbackMs,
      marketCount: universe.markets.length,
      expectedPairs: Math.max(0, universe.markets.length - 1),
      calculatedPairs: 0,
      processedMarkets: 0,
      requests: 0,
      cacheHitCandles: 0,
      cacheMissCandles: 0,
      cacheFetchedCandles: 0,
      cacheLoaded: false,
      truncated: universe.truncated,
      startedAt,
      updatedAt: startedAt,
      streamConnected: false,
      message:
        universe.markets.length > 1
          ? `Preparing ${universe.markets.length - 1} correlation pairs`
          : "No eligible correlation peers found",
      entries: [],
    };
    this.snapshots.set(focalMarket.id, snapshot);
    onUpdate();

    const abortController = new AbortController();
    this.activeRun = {
      focalMarketId: focalMarket.id,
      abortController,
    };

    void this.computeVector(
      focalMarket,
      universe.markets,
      universe.truncated,
      forceRefresh,
      abortController,
      onUpdate,
    );
    return snapshot;
  }

  snapshotForMarket(marketId: string | undefined): CorrelationSnapshot {
    if (!marketId) {
      return idleCorrelationSnapshot();
    }

    return this.snapshots.get(marketId) ?? idleCorrelationSnapshot(marketId);
  }

  handleCandle(candle: Candle): boolean {
    if (!candle.closed) {
      return false;
    }

    const marketId = this.marketIdBySymbol.get(candle.symbol.toUpperCase());
    if (!marketId) {
      return false;
    }

    return this.recordClosedCandle(marketId, candle);
  }

  async flush(): Promise<void> {
    const pending = [...this.saveTimers.keys()];
    for (const focalMarketId of pending) {
      const timer = this.saveTimers.get(focalMarketId);
      if (timer) {
        clearTimeout(timer);
      }
      this.saveTimers.delete(focalMarketId);
      await this.writeVectorCache(focalMarketId);
    }
  }

  stop(): void {
    this.activeRun?.abortController.abort();
    this.activeRun = undefined;
    this.stream?.stop();
    this.stream = undefined;
    for (const timer of this.saveTimers.values()) {
      clearTimeout(timer);
    }
    this.saveTimers.clear();
  }

  private async computeVector(
    focalMarket: BinanceMarketListing,
    markets: BinanceMarketListing[],
    truncated: boolean,
    forceRefresh: boolean,
    abortController: AbortController,
    onUpdate: () => void,
  ): Promise<void> {
    const signal = abortController.signal;
    const startedAt = Date.now();

    try {
      this.registerMarkets(markets);

      if (!forceRefresh) {
        const cached = await this.readFreshVectorCache(focalMarket.id, markets);
        throwIfAborted(signal);
        if (cached) {
          this.installCachedVector(focalMarket, markets, cached);
          this.publishReadySnapshot(focalMarket.id, {
            cacheLoaded: true,
            message: `Loaded cached correlations for ${focalMarket.displaySymbol}`,
          });
          this.startStream(markets, onUpdate);
          onUpdate();
          return;
        }
      }

      const targetEndTime = alignDown(Date.now() - this.intervalMs, this.intervalMs);
      const targetStartTime = alignDown(targetEndTime - this.options.lookbackMs, this.intervalMs);
      this.patchSnapshot(focalMarket.id, {
        startTime: targetStartTime,
        endTime: targetEndTime,
        message: `Loading ${focalMarket.displaySymbol} candles`,
      });
      onUpdate();

      const focalSeries = await this.loadMarketReturns(
        focalMarket,
        targetStartTime,
        targetEndTime,
        signal,
        (stats) => {
          this.patchSnapshot(focalMarket.id, {
            requests: stats.requests,
            cacheHitCandles: stats.cacheHitCandles,
            cacheMissCandles: stats.cacheMissCandles,
            cacheFetchedCandles: stats.cacheFetchedCandles,
            message:
              stats.cacheMissCandles > stats.cacheFetchedCandles
                ? `Caching ${focalMarket.displaySymbol} candles`
                : `Loaded ${focalMarket.displaySymbol} candles`,
          });
          onUpdate();
        },
      );
      this.seedLastClose(focalSeries);

      let aggregateRequests = focalSeries.cacheStats.requests;
      let aggregateCacheHitCandles = focalSeries.cacheStats.cacheHitCandles;
      let aggregateCacheMissCandles = focalSeries.cacheStats.cacheMissCandles;
      let aggregateCacheFetchedCandles = focalSeries.cacheStats.cacheFetchedCandles;
      let calculatedPairs = 0;
      let processedMarkets = 1;
      const peerMarketIds: string[] = [];
      const pairKeys: string[] = [];

      for (const peerMarket of markets) {
        throwIfAborted(signal);
        if (peerMarket.id === focalMarket.id) {
          continue;
        }

        this.patchSnapshot(focalMarket.id, {
          processedMarkets,
          calculatedPairs,
          requests: aggregateRequests,
          cacheHitCandles: aggregateCacheHitCandles,
          cacheMissCandles: aggregateCacheMissCandles,
          cacheFetchedCandles: aggregateCacheFetchedCandles,
          message: `Loading ${peerMarket.displaySymbol} candles`,
        });
        onUpdate();

        const peerSeries = await this.loadMarketReturns(
          peerMarket,
          targetStartTime,
          targetEndTime,
          signal,
        );
        this.seedLastClose(peerSeries);

        aggregateRequests += peerSeries.cacheStats.requests;
        aggregateCacheHitCandles += peerSeries.cacheStats.cacheHitCandles;
        aggregateCacheMissCandles += peerSeries.cacheStats.cacheMissCandles;
        aggregateCacheFetchedCandles += peerSeries.cacheStats.cacheFetchedCandles;
        processedMarkets += 1;

        const pairState = computePairState(focalSeries, peerSeries, Date.now());
        this.setPairState(pairState);
        peerMarketIds.push(peerMarket.id);
        pairKeys.push(pairState.key);
        calculatedPairs += 1;

        this.patchSnapshot(focalMarket.id, {
          processedMarkets,
          calculatedPairs,
          requests: aggregateRequests,
          cacheHitCandles: aggregateCacheHitCandles,
          cacheMissCandles: aggregateCacheMissCandles,
          cacheFetchedCandles: aggregateCacheFetchedCandles,
          message: `Computed ${calculatedPairs}/${Math.max(0, markets.length - 1)} pairs`,
        });
        onUpdate();
      }

      const vector: CorrelationVectorState = {
        focalMarketId: focalMarket.id,
        peerMarketIds,
        pairKeys,
        marketIds: markets.map((market) => market.id),
        startTime: targetStartTime,
        endTime: targetEndTime,
        lookbackMs: this.options.lookbackMs,
        createdAt: startedAt,
        updatedAt: Date.now(),
        truncated,
      };
      this.vectors.set(focalMarket.id, vector);
      this.publishReadySnapshot(focalMarket.id, {
        requests: aggregateRequests,
        cacheHitCandles: aggregateCacheHitCandles,
        cacheMissCandles: aggregateCacheMissCandles,
        cacheFetchedCandles: aggregateCacheFetchedCandles,
        cacheLoaded: false,
        message: `Correlations ready for ${focalMarket.displaySymbol}`,
      });
      await this.writeVectorCache(focalMarket.id);
      this.startStream(markets, onUpdate);
      onUpdate();
    } catch (error) {
      if (signal.aborted) {
        return;
      }

      const message = error instanceof Error ? error.message : "Correlation computation failed";
      this.snapshots.set(focalMarket.id, {
        ...this.snapshotForMarket(focalMarket.id),
        status: "failed",
        updatedAt: Date.now(),
        streamConnected: false,
        message,
        error: message,
      });
      onUpdate();
    } finally {
      if (this.activeRun?.abortController === abortController) {
        this.activeRun = undefined;
      }
    }
  }

  private async loadMarketReturns(
    market: BinanceMarketListing,
    startTime: number,
    endTime: number,
    signal: AbortSignal,
    onCacheProgress?: (stats: HistoricalCandleCacheStats) => void,
  ): Promise<MarketReturnSeries> {
    const cache = new HistoricalCandleCache({
      dataDir: this.options.dataDir,
      marketKey: getMarketStorageKey(market),
      symbol: market.symbol,
      interval: this.options.interval,
      intervalMs: this.intervalMs,
      maxBytes: this.options.historicalCache.maxBytes,
      minFreeBytes: this.options.historicalCache.minFreeBytes,
    });

    let cacheStats = emptyCacheStats();
    cacheStats = await cache.ensureRange(
      startTime,
      endTime,
      (request) =>
        fetchKlines({
          venue: market.venue as StreamVenue,
          symbol: market.symbol,
          interval: this.options.interval,
          startTime: request.startTime,
          endTime: request.endTime,
          limit: request.limit,
          signal,
        }),
      (stats) => {
        throwIfAborted(signal);
        cacheStats = stats;
        onCacheProgress?.(stats);
      },
    );
    throwIfAborted(signal);

    const candles: Candle[] = [];
    for await (const batch of cache.readRangeBatches(startTime, endTime)) {
      candles.push(...batch);
    }
    candles.sort((a, b) => a.openTime - b.openTime);

    const returns = new Map<number, number>();
    let previousClose: number | undefined;
    let lastClose: number | undefined;
    let lastReturnOpenTime: number | undefined;

    for (const candle of candles) {
      if (!Number.isFinite(candle.close) || candle.close <= 0) {
        continue;
      }
      if (previousClose !== undefined && previousClose > 0) {
        returns.set(candle.openTime, Math.log(candle.close / previousClose));
        lastReturnOpenTime = candle.openTime;
      }
      previousClose = candle.close;
      lastClose = candle.close;
    }

    return {
      market,
      returns,
      lastClose,
      lastReturnOpenTime,
      cacheStats,
    };
  }

  private normalizeUniverse(
    focalMarket: BinanceMarketListing,
    candidateMarkets: BinanceMarketListing[],
  ): { markets: BinanceMarketListing[]; truncated: boolean } {
    const byId = new Map<string, BinanceMarketListing>();
    byId.set(focalMarket.id, focalMarket);

    for (const market of candidateMarkets) {
      if (market.id === focalMarket.id) {
        byId.set(market.id, focalMarket);
        continue;
      }
      if (
        market.quoteAsset !== focalMarket.quoteAsset ||
        !market.supportsHistoricalCandles ||
        !market.supportsLiveStream ||
        !isCorrelationMarketGroup(market.group) ||
        market.venue === "predictions"
      ) {
        continue;
      }
      byId.set(market.id, market);
    }

    const eligibleMarkets = [...byId.values()];
    const maxMarkets = Math.max(1, Math.floor(this.options.maxMarkets));
    const truncated = eligibleMarkets.length > maxMarkets;
    return {
      markets: rankCorrelationMarkets(focalMarket, eligibleMarkets, maxMarkets),
      truncated,
    };
  }

  private startStream(markets: BinanceMarketListing[], onUpdate: () => void): void {
    this.stream?.stop();
    if (markets.length <= 1) {
      this.stream = undefined;
      return;
    }

    this.stream = new CorrelationKlineStream({
      markets,
      interval: this.options.interval,
      onCandle: (candle) => {
        if (this.handleCandle(candle)) {
          onUpdate();
        }
      },
      onStatus: (status) => {
        for (const vector of this.vectors.values()) {
          if (sameStringList(vector.marketIds, markets.map((market) => market.id))) {
            this.patchSnapshot(vector.focalMarketId, {
              streamConnected: status.connected,
              message:
                this.snapshotForMarket(vector.focalMarketId).status === "ready"
                  ? status.message
                  : this.snapshotForMarket(vector.focalMarketId).message,
            });
          }
        }
        onUpdate();
      },
    });
    this.stream.start();
  }

  private recordClosedCandle(marketId: string, candle: Candle): boolean {
    if (!Number.isFinite(candle.close) || candle.close <= 0) {
      return false;
    }

    const previousOpenTime = this.lastReturnOpenTimeByMarketId.get(marketId);
    if (previousOpenTime !== undefined && candle.openTime <= previousOpenTime) {
      return false;
    }

    const previousClose = this.lastCloseByMarketId.get(marketId);
    this.lastCloseByMarketId.set(marketId, candle.close);
    if (previousClose === undefined || previousClose <= 0) {
      return false;
    }

    const returnValue = Math.log(candle.close / previousClose);
    this.lastReturnOpenTimeByMarketId.set(marketId, candle.openTime);

    const returnsAtTime =
      this.pendingReturnsByOpenTime.get(candle.openTime) ?? new Map<string, number>();
    if (returnsAtTime.has(marketId)) {
      return false;
    }

    const changedPairKeys: string[] = [];
    for (const [otherMarketId, otherReturn] of returnsAtTime) {
      const key = correlationPairKey(marketId, otherMarketId);
      const pairState = this.pairStatsByKey.get(key);
      if (!pairState || (pairState.endTime !== undefined && candle.openTime <= pairState.endTime)) {
        continue;
      }

      addPairSample(
        pairState,
        marketId,
        returnValue,
        otherMarketId,
        otherReturn,
        candle.openTime,
      );
      changedPairKeys.push(key);
    }

    returnsAtTime.set(marketId, returnValue);
    this.pendingReturnsByOpenTime.set(candle.openTime, returnsAtTime);
    this.trimPendingReturns();

    if (changedPairKeys.length === 0) {
      return false;
    }

    const changedVectors = this.vectorIdsForPairKeys(changedPairKeys);
    for (const focalMarketId of changedVectors) {
      this.publishReadySnapshot(focalMarketId, {
        message: `Updated correlations at ${new Date(candle.closeTime).toLocaleTimeString()}`,
      });
      this.scheduleVectorCacheWrite(focalMarketId);
    }

    return changedVectors.size > 0;
  }

  private vectorIdsForPairKeys(pairKeys: string[]): Set<string> {
    const keys = new Set(pairKeys);
    const vectorIds = new Set<string>();
    for (const vector of this.vectors.values()) {
      if (vector.pairKeys.some((key) => keys.has(key))) {
        vectorIds.add(vector.focalMarketId);
      }
    }
    return vectorIds;
  }

  private publishReadySnapshot(
    focalMarketId: string,
    patch: Partial<CorrelationSnapshot> = {},
  ): void {
    const vector = this.vectors.get(focalMarketId);
    const focalMarket = this.marketsById.get(focalMarketId);
    if (!vector || !focalMarket) {
      return;
    }

    const entries = vector.peerMarketIds
      .map((peerMarketId) => this.entryForPair(focalMarketId, peerMarketId))
      .filter((entry): entry is CorrelationEntry => Boolean(entry));
    const now = Date.now();
    vector.updatedAt = now;

    this.snapshots.set(focalMarketId, {
      status: "ready",
      focalMarketId,
      focalSymbol: focalMarket.symbol,
      focalDisplaySymbol: focalMarket.displaySymbol,
      interval: this.options.interval,
      lookbackMs: vector.lookbackMs,
      marketCount: vector.marketIds.length,
      expectedPairs: vector.peerMarketIds.length,
      calculatedPairs: entries.length,
      processedMarkets: vector.marketIds.length,
      requests: 0,
      cacheHitCandles: 0,
      cacheMissCandles: 0,
      cacheFetchedCandles: 0,
      cacheLoaded: false,
      truncated: vector.truncated,
      startedAt: vector.createdAt,
      updatedAt: now,
      startTime: vector.startTime,
      endTime: vector.endTime,
      streamConnected: this.stream?.connected ?? false,
      message: `Correlations ready for ${focalMarket.displaySymbol}`,
      entries,
      ...patch,
    });
  }

  private entryForPair(
    focalMarketId: string,
    peerMarketId: string,
  ): CorrelationEntry | undefined {
    const peerMarket = this.marketsById.get(peerMarketId);
    const pairState = this.pairStatsByKey.get(correlationPairKey(focalMarketId, peerMarketId));
    if (!peerMarket || !pairState) {
      return undefined;
    }

    return {
      marketId: peerMarket.id,
      symbol: peerMarket.symbol,
      displaySymbol: peerMarket.displaySymbol,
      baseAsset: peerMarket.baseAsset,
      quoteAsset: peerMarket.quoteAsset,
      venue: peerMarket.venue,
      correlation: correlationValue(pairState.stats),
      samples: pairState.stats.samples,
      startTime: pairState.startTime,
      endTime: pairState.endTime,
      updatedAt: pairState.updatedAt,
    };
  }

  private patchSnapshot(
    focalMarketId: string,
    patch: Partial<CorrelationSnapshot>,
  ): void {
    this.snapshots.set(focalMarketId, {
      ...this.snapshotForMarket(focalMarketId),
      ...patch,
      updatedAt: Date.now(),
    });
  }

  private registerMarkets(markets: BinanceMarketListing[]): void {
    for (const market of markets) {
      this.marketsById.set(market.id, market);
      this.marketIdBySymbol.set(market.symbol.toUpperCase(), market.id);
    }
  }

  private seedLastClose(series: MarketReturnSeries): void {
    if (series.lastClose !== undefined) {
      this.lastCloseByMarketId.set(series.market.id, series.lastClose);
    }
    if (series.lastReturnOpenTime !== undefined) {
      this.lastReturnOpenTimeByMarketId.set(series.market.id, series.lastReturnOpenTime);
    }
  }

  private setPairState(pairState: CorrelationPairState): void {
    this.pairStatsByKey.set(pairState.key, pairState);
    addToSetMap(this.pairKeysByMarketId, pairState.marketAId, pairState.key);
    addToSetMap(this.pairKeysByMarketId, pairState.marketBId, pairState.key);
  }

  private trimPendingReturns(): void {
    if (this.pendingReturnsByOpenTime.size <= PENDING_RETURN_BUCKET_LIMIT) {
      return;
    }

    const sortedTimes = [...this.pendingReturnsByOpenTime.keys()].sort((a, b) => a - b);
    for (const time of sortedTimes.slice(0, sortedTimes.length - PENDING_RETURN_BUCKET_LIMIT)) {
      this.pendingReturnsByOpenTime.delete(time);
    }
  }

  private cachePath(focalMarketId: string): string {
    const fileName = `${safePathPart(focalMarketId)}-${safePathPart(this.options.interval)}-${Math.round(
      this.options.lookbackMs,
    )}.json`;
    return path.join(this.cacheDir, fileName);
  }

  private async readFreshVectorCache(
    focalMarketId: string,
    markets: BinanceMarketListing[],
  ): Promise<CachedCorrelationVector | undefined> {
    try {
      const content = await fs.readFile(this.cachePath(focalMarketId), "utf8");
      const cached = JSON.parse(content) as CachedCorrelationVector;
      const expectedMarketIds = markets.map((market) => market.id);
      const maxAgeMs = Math.max(this.intervalMs * 2, 60_000);

      if (
        cached.version !== CORRELATION_CACHE_VERSION ||
        cached.focalMarketId !== focalMarketId ||
        cached.interval !== this.options.interval ||
        cached.lookbackMs !== this.options.lookbackMs ||
        Date.now() - cached.updatedAt > maxAgeMs ||
        !sameStringList(cached.marketIds, expectedMarketIds)
      ) {
        return undefined;
      }

      return cached;
    } catch (error) {
      if (isMissingFile(error)) {
        return undefined;
      }
      throw error;
    }
  }

  private installCachedVector(
    focalMarket: BinanceMarketListing,
    markets: BinanceMarketListing[],
    cached: CachedCorrelationVector,
  ): void {
    this.registerMarkets(markets);

    for (const pair of cached.pairs) {
      this.setPairState({
        key: pair.key,
        marketAId: pair.marketAId,
        marketBId: pair.marketBId,
        stats: pair.stats,
        startTime: pair.startTime,
        endTime: pair.endTime,
        updatedAt: pair.updatedAt,
      });
    }

    for (const lastClose of cached.lastCloses) {
      this.lastCloseByMarketId.set(lastClose.marketId, lastClose.close);
      if (lastClose.lastReturnOpenTime !== undefined) {
        this.lastReturnOpenTimeByMarketId.set(
          lastClose.marketId,
          lastClose.lastReturnOpenTime,
        );
      }
    }

    this.vectors.set(focalMarket.id, {
      focalMarketId: focalMarket.id,
      peerMarketIds: cached.peerMarketIds,
      pairKeys: cached.pairs.map((pair) => pair.key),
      marketIds: cached.marketIds,
      startTime: cached.startTime,
      endTime: cached.endTime,
      lookbackMs: cached.lookbackMs,
      createdAt: cached.createdAt,
      updatedAt: cached.updatedAt,
      truncated: cached.truncated,
    });
  }

  private scheduleVectorCacheWrite(focalMarketId: string): void {
    if (this.saveTimers.has(focalMarketId)) {
      return;
    }

    const timer = setTimeout(() => {
      this.saveTimers.delete(focalMarketId);
      void this.writeVectorCache(focalMarketId);
    }, 1_000);
    this.saveTimers.set(focalMarketId, timer);
  }

  private async writeVectorCache(focalMarketId: string): Promise<void> {
    const vector = this.vectors.get(focalMarketId);
    if (!vector) {
      return;
    }

    const lastCloses: CachedCorrelationVector["lastCloses"] = [];
    for (const marketId of vector.marketIds) {
      const close = this.lastCloseByMarketId.get(marketId);
      if (close === undefined) {
        continue;
      }

      const lastReturnOpenTime = this.lastReturnOpenTimeByMarketId.get(marketId);
      lastCloses.push({
        marketId,
        close,
        ...(lastReturnOpenTime === undefined ? {} : { lastReturnOpenTime }),
      });
    }

    const payload: CachedCorrelationVector = {
      version: CORRELATION_CACHE_VERSION,
      focalMarketId,
      interval: this.options.interval,
      lookbackMs: vector.lookbackMs,
      marketIds: vector.marketIds,
      peerMarketIds: vector.peerMarketIds,
      startTime: vector.startTime,
      endTime: vector.endTime,
      createdAt: vector.createdAt,
      updatedAt: vector.updatedAt,
      truncated: vector.truncated,
      pairs: vector.pairKeys
        .map((key) => this.pairStatsByKey.get(key))
        .filter((pair): pair is CorrelationPairState => Boolean(pair))
        .map((pair) => ({
          key: pair.key,
          marketAId: pair.marketAId,
          marketBId: pair.marketBId,
          stats: pair.stats,
          startTime: pair.startTime,
          endTime: pair.endTime,
          updatedAt: pair.updatedAt,
        })),
      lastCloses,
    };

    await fs.mkdir(this.cacheDir, { recursive: true });
    await writeJsonAtomic(this.cachePath(focalMarketId), payload);
  }
}

class CorrelationKlineStream {
  private sockets = new Map<string, WebSocket>();
  private reconnectTimers = new Map<string, NodeJS.Timeout>();
  private reconnectAttempts = new Map<string, number>();
  private openSockets = new Set<string>();
  private stopped = true;
  readonly marketsBySymbol: Map<string, BinanceMarketListing>;

  constructor(
    private readonly options: {
      markets: BinanceMarketListing[];
      interval: string;
      onCandle: (candle: Candle) => void;
      onStatus: (status: KlineStreamStatus) => void;
    },
  ) {
    this.marketsBySymbol = new Map(
      options.markets.map((market) => [market.symbol.toUpperCase(), market]),
    );
  }

  get connected(): boolean {
    return this.openSockets.size > 0 && this.openSockets.size === this.socketSpecs().length;
  }

  start(): void {
    if (!this.stopped) {
      return;
    }

    this.stopped = false;
    for (const spec of this.socketSpecs()) {
      this.connect(spec);
    }
  }

  stop(): void {
    this.stopped = true;
    for (const timer of this.reconnectTimers.values()) {
      clearTimeout(timer);
    }
    this.reconnectTimers.clear();
    this.reconnectAttempts.clear();
    this.openSockets.clear();
    for (const socket of this.sockets.values()) {
      socket.close();
    }
    this.sockets.clear();
  }

  private connect(spec: { label: string; url: string }): void {
    const socket = new WebSocket(spec.url);
    this.sockets.set(spec.label, socket);

    socket.on("open", () => {
      this.openSockets.add(spec.label);
      this.reconnectAttempts.set(spec.label, 0);
      this.emitStatus(`Correlation stream connected (${this.openSockets.size}/${this.socketSpecs().length})`);
    });

    socket.on("message", (raw) => {
      this.handleMessage(raw.toString());
    });

    socket.on("close", () => {
      this.openSockets.delete(spec.label);
      this.emitStatus("Correlation stream closed");
      this.scheduleReconnect(spec);
    });

    socket.on("error", (error) => {
      this.openSockets.delete(spec.label);
      this.emitStatus(`Correlation stream error: ${error.message}`);
    });
  }

  private handleMessage(raw: string): void {
    const payload = JSON.parse(raw) as { data?: Record<string, unknown> };
    const candle = parseKlinePayload(payload.data);
    if (candle?.closed && this.marketsBySymbol.has(candle.symbol.toUpperCase())) {
      this.options.onCandle(candle);
    }
  }

  private scheduleReconnect(spec: { label: string; url: string }): void {
    if (this.stopped) {
      return;
    }

    const reconnectAttempt = (this.reconnectAttempts.get(spec.label) ?? 0) + 1;
    this.reconnectAttempts.set(spec.label, reconnectAttempt);
    const delayMs = Math.min(30_000, 1_000 * 2 ** Math.min(5, reconnectAttempt));
    const timer = setTimeout(() => {
      this.reconnectTimers.delete(spec.label);
      this.connect(spec);
    }, delayMs);
    this.reconnectTimers.set(spec.label, timer);
  }

  private emitStatus(message: string): void {
    this.options.onStatus({
      connected: this.connected,
      message,
    });
  }

  private socketSpecs(): Array<{ label: string; url: string }> {
    const marketsByVenue = new Map<StreamVenue, BinanceMarketListing[]>();
    for (const market of this.options.markets) {
      const venue = market.venue as StreamVenue;
      const list = marketsByVenue.get(venue) ?? [];
      list.push(market);
      marketsByVenue.set(venue, list);
    }

    const specs: Array<{ label: string; url: string }> = [];
    for (const [venue, markets] of marketsByVenue) {
      const streams = markets.map(
        (market) => `${market.symbol.toLowerCase()}@kline_${this.options.interval}`,
      );
      for (let index = 0; index < streams.length; index += STREAM_CHUNK_SIZE) {
        const chunk = streams.slice(index, index + STREAM_CHUNK_SIZE);
        specs.push({
          label: `${venue}-${Math.floor(index / STREAM_CHUNK_SIZE) + 1}`,
          url: combinedStreamUrl(streamBaseUrl(venue), chunk),
        });
      }
    }

    return specs;
  }
}

function computePairState(
  focalSeries: MarketReturnSeries,
  peerSeries: MarketReturnSeries,
  updatedAt: number,
): CorrelationPairState {
  const marketAId =
    focalSeries.market.id < peerSeries.market.id ? focalSeries.market.id : peerSeries.market.id;
  const marketBId =
    focalSeries.market.id < peerSeries.market.id ? peerSeries.market.id : focalSeries.market.id;
  const pairState: CorrelationPairState = {
    key: correlationPairKey(focalSeries.market.id, peerSeries.market.id),
    marketAId,
    marketBId,
    stats: emptyRunningStats(),
  };

  for (const [openTime, focalReturn] of focalSeries.returns) {
    const peerReturn = peerSeries.returns.get(openTime);
    if (peerReturn === undefined) {
      continue;
    }
    addPairSample(
      pairState,
      focalSeries.market.id,
      focalReturn,
      peerSeries.market.id,
      peerReturn,
      openTime,
      updatedAt,
    );
  }

  return pairState;
}

function addPairSample(
  pairState: CorrelationPairState,
  marketId: string,
  returnValue: number,
  otherMarketId: string,
  otherReturn: number,
  openTime: number,
  updatedAt = Date.now(),
): void {
  const x = marketId === pairState.marketAId ? returnValue : otherReturn;
  const y = otherMarketId === pairState.marketBId ? otherReturn : returnValue;
  pairState.stats.samples += 1;
  pairState.stats.sumX += x;
  pairState.stats.sumY += y;
  pairState.stats.sumXX += x * x;
  pairState.stats.sumYY += y * y;
  pairState.stats.sumXY += x * y;
  pairState.startTime = pairState.startTime === undefined ? openTime : Math.min(pairState.startTime, openTime);
  pairState.endTime = pairState.endTime === undefined ? openTime : Math.max(pairState.endTime, openTime);
  pairState.updatedAt = updatedAt;
}

function correlationValue(stats: RunningCorrelationStats): number | undefined {
  if (stats.samples < 2) {
    return undefined;
  }

  const n = stats.samples;
  const covariance = n * stats.sumXY - stats.sumX * stats.sumY;
  const varianceX = n * stats.sumXX - stats.sumX * stats.sumX;
  const varianceY = n * stats.sumYY - stats.sumY * stats.sumY;
  const denominator = Math.sqrt(varianceX * varianceY);

  if (!Number.isFinite(denominator) || denominator <= 0) {
    return undefined;
  }

  return Math.max(-1, Math.min(1, covariance / denominator));
}

function emptyRunningStats(): RunningCorrelationStats {
  return {
    samples: 0,
    sumX: 0,
    sumY: 0,
    sumXX: 0,
    sumYY: 0,
    sumXY: 0,
  };
}

function rankCorrelationMarkets(
  focalMarket: BinanceMarketListing,
  markets: BinanceMarketListing[],
  maxMarkets: number,
): BinanceMarketListing[] {
  const focal = markets.find((market) => market.id === focalMarket.id) ?? focalMarket;
  const peers = markets
    .filter((market) => market.id !== focal.id)
    .sort(compareCorrelationAlphabetically);
  const limit = Math.max(1, maxMarkets);

  if (peers.length + 1 <= limit) {
    return [focal, ...peers];
  }

  const selected = new Map<string, BinanceMarketListing>([[focal.id, focal]]);
  seedCategoryQuotas(selected, peers, focal, limit);
  const rankings = [
    rankByCategory(peers, focal),
    rankByMetric(peers, (market) => market.quoteVolume24h),
    rankByMetric(peers, (market) =>
      Number.isFinite(market.priceChangePercent24h)
        ? Math.abs(market.priceChangePercent24h as number)
        : undefined,
    ),
    rankByMetric(peers, (market) => market.tradeCount24h),
    rankByMetric(peers, (market) => market.volume24h),
    peers,
  ].filter((ranking) => ranking.length > 0);
  const cursors = new Array(rankings.length).fill(0);

  while (selected.size < limit) {
    let added = false;
    for (let rankingIndex = 0; rankingIndex < rankings.length; rankingIndex += 1) {
      const ranking = rankings[rankingIndex];
      while (
        cursors[rankingIndex] < ranking.length &&
        selected.has(ranking[cursors[rankingIndex]].id)
      ) {
        cursors[rankingIndex] += 1;
      }

      const market = ranking[cursors[rankingIndex]];
      if (!market) {
        continue;
      }

      selected.set(market.id, market);
      cursors[rankingIndex] += 1;
      added = true;
      if (selected.size >= limit) {
        break;
      }
    }

    if (!added) {
      break;
    }
  }

  for (const market of peers) {
    if (selected.size >= limit) {
      break;
    }
    selected.set(market.id, market);
  }

  return [...selected.values()];
}

type CorrelationAssetCategory = "spot" | "futures" | "stocks" | "options";

function seedCategoryQuotas(
  selected: Map<string, BinanceMarketListing>,
  peers: BinanceMarketListing[],
  focalMarket: BinanceMarketListing,
  limit: number,
): void {
  const categories = categoryOrderFor(focalMarket).filter((category) =>
    peers.some((market) => correlationAssetCategory(market) === category) ||
    correlationAssetCategory(focalMarket) === category,
  );
  const marketsByCategory = new Map<CorrelationAssetCategory, BinanceMarketListing[]>();
  for (const category of categories) {
    marketsByCategory.set(
      category,
      peers
        .filter((market) => correlationAssetCategory(market) === category)
        .sort(compareCorrelationSelectionScore),
    );
  }

  for (let targetCount = 1; targetCount <= CATEGORY_MIN_MARKETS; targetCount += 1) {
    for (const category of categories) {
      if (selected.size >= limit) {
        return;
      }
      if (selectedCategoryCount(selected, category) >= targetCount) {
        continue;
      }

      const market = nextUnselected(marketsByCategory.get(category) ?? [], selected);
      if (market) {
        selected.set(market.id, market);
      }
    }
  }
}

function rankByCategory(
  markets: BinanceMarketListing[],
  focalMarket: BinanceMarketListing,
): BinanceMarketListing[] {
  const marketsByCategory = new Map<CorrelationAssetCategory, BinanceMarketListing[]>();
  for (const market of markets) {
    const category = correlationAssetCategory(market);
    const bucket = marketsByCategory.get(category) ?? [];
    bucket.push(market);
    marketsByCategory.set(category, bucket);
  }
  for (const bucket of marketsByCategory.values()) {
    bucket.sort(compareCorrelationSelectionScore);
  }

  const ranked: BinanceMarketListing[] = [];
  const categories = categoryOrderFor(focalMarket);
  let added = true;
  while (added) {
    added = false;
    for (const category of categories) {
      const market = marketsByCategory.get(category)?.shift();
      if (!market) {
        continue;
      }
      ranked.push(market);
      added = true;
    }
  }

  return ranked;
}

function rankByMetric(
  markets: BinanceMarketListing[],
  metric: (market: BinanceMarketListing) => number | undefined,
): BinanceMarketListing[] {
  return markets
    .filter((market) => Number.isFinite(metric(market)))
    .sort((a, b) => {
      const metricDiff = (metric(b) as number) - (metric(a) as number);
      return metricDiff || compareCorrelationAlphabetically(a, b);
    });
}

function selectedCategoryCount(
  selected: Map<string, BinanceMarketListing>,
  category: CorrelationAssetCategory,
): number {
  return [...selected.values()].filter(
    (market) => correlationAssetCategory(market) === category,
  ).length;
}

function nextUnselected(
  markets: BinanceMarketListing[],
  selected: Map<string, BinanceMarketListing>,
): BinanceMarketListing | undefined {
  return markets.find((market) => !selected.has(market.id));
}

function categoryOrderFor(focalMarket: BinanceMarketListing): CorrelationAssetCategory[] {
  const focalCategory = correlationAssetCategory(focalMarket);
  const categories: CorrelationAssetCategory[] = ["spot", "futures", "stocks", "options"];
  return [
    focalCategory,
    ...categories.filter((category) => category !== focalCategory),
  ];
}

function correlationAssetCategory(
  market: Pick<BinanceMarketListing, "group" | "venue">,
): CorrelationAssetCategory {
  if (market.group === "options" || market.venue === "options") {
    return "options";
  }
  if (market.group === "bstocks" || market.group === "tradfi") {
    return "stocks";
  }
  if (
    market.group === "futures" ||
    market.venue === "usdm-futures" ||
    market.venue === "coinm-futures"
  ) {
    return "futures";
  }
  return "spot";
}

function isCorrelationMarketGroup(group: BinanceMarketListing["group"]): boolean {
  return (
    group === "spot" ||
    group === "bstocks" ||
    group === "futures" ||
    group === "tradfi" ||
    group === "options"
  );
}

function compareCorrelationSelectionScore(
  a: BinanceMarketListing,
  b: BinanceMarketListing,
): number {
  return (
    marketSelectionScore(b) - marketSelectionScore(a) ||
    compareCorrelationAlphabetically(a, b)
  );
}

function marketSelectionScore(market: BinanceMarketListing): number {
  const quoteVolume = Math.log10((market.quoteVolume24h ?? 0) + 1);
  const tradeCount = Math.log10((market.tradeCount24h ?? 0) + 1);
  const baseVolume = Math.log10((market.volume24h ?? 0) + 1);
  const absMove = Math.abs(market.priceChangePercent24h ?? 0);
  return quoteVolume * 3 + tradeCount * 2 + baseVolume + absMove * 0.25;
}

function compareCorrelationAlphabetically(
  a: BinanceMarketListing,
  b: BinanceMarketListing,
): number {
  return a.displaySymbol.localeCompare(b.displaySymbol) || a.id.localeCompare(b.id);
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

function idleCorrelationSnapshot(focalMarketId?: string): CorrelationSnapshot {
  return {
    status: "idle",
    focalMarketId,
    marketCount: 0,
    expectedPairs: 0,
    calculatedPairs: 0,
    processedMarkets: 0,
    requests: 0,
    cacheHitCandles: 0,
    cacheMissCandles: 0,
    cacheFetchedCandles: 0,
    cacheLoaded: false,
    truncated: false,
    streamConnected: false,
    message: "Correlations have not been computed yet",
    entries: [],
  };
}

function parseKlinePayload(data: Record<string, unknown> | undefined): Candle | undefined {
  const kline = data?.k as Record<string, unknown> | undefined;
  if (!kline) {
    return undefined;
  }

  const symbol = String(kline.s ?? data?.s ?? "");
  const close = Number(kline.c);
  if (!symbol || !Number.isFinite(close)) {
    return undefined;
  }

  return {
    symbol,
    interval: String(kline.i ?? "1m"),
    openTime: Number(kline.t),
    closeTime: Number(kline.T),
    open: Number(kline.o),
    high: Number(kline.h),
    low: Number(kline.l),
    close,
    volume: Number(kline.v),
    closed: Boolean(kline.x),
  };
}

function streamBaseUrl(venue: StreamVenue): string {
  if (venue === "spot") {
    return "wss://stream.binance.com:9443";
  }
  if (venue === "coinm-futures") {
    return "wss://dstream.binance.com";
  }
  if (venue === "options") {
    return "wss://nbstream.binance.com/eoptions";
  }
  return "wss://fstream.binance.com/market";
}

function combinedStreamUrl(baseUrl: string, streams: string[]): string {
  return `${baseUrl}/stream?streams=${streams.join("/")}`;
}

function correlationPairKey(marketAId: string, marketBId: string): string {
  return [marketAId, marketBId].sort().join("__");
}

function addToSetMap<K, V>(map: Map<K, Set<V>>, key: K, value: V): void {
  const set = map.get(key) ?? new Set<V>();
  set.add(value);
  map.set(key, set);
}

function alignDown(value: number, intervalMs: number): number {
  return Math.floor(value / intervalMs) * intervalMs;
}

function sameStringList(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new Error("Correlation computation cancelled");
  }
}

function safePathPart(value: string): string {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
}

async function writeJsonAtomic(filePath: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.${process.pid}.${Date.now()}.${Math.random().toString(36).slice(2)}.tmp`;
  await fs.writeFile(tempPath, `${JSON.stringify(value, null, 2)}\n`);
  await fs.rename(tempPath, filePath);
}

function isMissingFile(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}
