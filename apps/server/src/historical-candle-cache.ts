import fs from "node:fs/promises";
import path from "node:path";
import type { Candle } from "@trading/bot-algo";

const DAY_MS = 24 * 60 * 60 * 1000;

export interface HistoricalCandleCacheOptions {
  dataDir: string;
  marketKey: string;
  symbol: string;
  interval: string;
  intervalMs: number;
  maxBytes: number;
  minFreeBytes: number;
}

export interface HistoricalCandleCacheStats {
  cacheHitCandles: number;
  cacheMissCandles: number;
  cacheFetchedCandles: number;
  requests: number;
  cacheSizeBytes: number;
  cacheEvictedBytes: number;
  cacheEvictedFiles: number;
  freeBytes: number;
}

export interface CandleFetchRequest {
  startTime: number;
  endTime: number;
  limit: number;
}

export type CandleRangeFetcher = (request: CandleFetchRequest) => Promise<Candle[]>;

export class HistoricalCandleCache {
  private readonly rootDir: string;

  constructor(private readonly options: HistoricalCandleCacheOptions) {
    this.rootDir = path.join(
      options.dataDir,
      "historical",
      safePathPart(options.marketKey),
      safePathPart(options.symbol),
      safePathPart(options.interval),
    );
  }

  async ensureRange(
    startTime: number,
    endTime: number,
    fetcher: CandleRangeFetcher,
    onProgress?: (stats: HistoricalCandleCacheStats) => void,
  ): Promise<HistoricalCandleCacheStats> {
    await fs.mkdir(this.rootDir, { recursive: true });
    const protectedFiles = new Set(this.filePathsForRange(startTime, endTime));
    const stats = await this.initialStats(protectedFiles);
    const cachedOpenTimes = await this.collectCachedOpenTimes(startTime, endTime);
    const expectedOpenTimes = expectedCandleOpenTimes(
      startTime,
      endTime,
      this.options.intervalMs,
    );
    const missingRanges = missingRangesFromExpected(
      expectedOpenTimes,
      cachedOpenTimes,
      this.options.intervalMs,
    );

    stats.cacheHitCandles =
      expectedOpenTimes.length - sumRangeCandles(missingRanges, this.options.intervalMs);
    stats.cacheMissCandles = sumRangeCandles(missingRanges, this.options.intervalMs);
    onProgress?.({ ...stats });

    for (const range of missingRanges) {
      let cursor = range.startTime;

      while (cursor <= range.endTime) {
        const candles = await fetcher({
          startTime: cursor,
          endTime: range.endTime,
          limit: 1000,
        });
        stats.requests += 1;

        if (candles.length === 0) {
          break;
        }

        await this.mergeCandles(candles);
        stats.cacheFetchedCandles += candles.length;
        const last = candles[candles.length - 1];
        const nextCursor = last.openTime + this.options.intervalMs;
        if (!Number.isFinite(nextCursor) || nextCursor <= cursor) {
          break;
        }

        cursor = nextCursor;
        mergeBudgetStats(stats, await this.enforceStorageBudget(protectedFiles));
        onProgress?.({ ...stats });
      }
    }

    mergeBudgetStats(stats, await this.enforceStorageBudget(protectedFiles));
    return stats;
  }

  async *readRangeBatches(
    startTime: number,
    endTime: number,
    batchSize = 1000,
  ): AsyncGenerator<Candle[]> {
    let batch: Candle[] = [];

    for (const filePath of this.filePathsForRange(startTime, endTime)) {
      const candles = (await readJsonLines<Candle>(filePath))
        .filter((candle) => candle.openTime >= startTime && candle.openTime <= endTime)
        .sort((a, b) => a.openTime - b.openTime);

      for (const candle of candles) {
        batch.push(candle);
        if (batch.length >= batchSize) {
          yield batch;
          batch = [];
        }
      }
    }

    if (batch.length > 0) {
      yield batch;
    }
  }

  private async collectCachedOpenTimes(
    startTime: number,
    endTime: number,
  ): Promise<Set<number>> {
    const openTimes = new Set<number>();

    for (const filePath of this.filePathsForRange(startTime, endTime)) {
      for (const candle of await readJsonLines<Candle>(filePath)) {
        if (candle.openTime >= startTime && candle.openTime <= endTime) {
          openTimes.add(candle.openTime);
        }
      }
    }

    return openTimes;
  }

  private async mergeCandles(candles: Candle[]): Promise<void> {
    const candlesByFile = new Map<string, Candle[]>();

    for (const candle of candles) {
      const filePath = this.filePathForTime(candle.openTime);
      const existing = candlesByFile.get(filePath) ?? [];
      existing.push(candle);
      candlesByFile.set(filePath, existing);
    }

    for (const [filePath, newCandles] of candlesByFile) {
      const byOpenTime = new Map<number, Candle>();
      for (const candle of await readJsonLines<Candle>(filePath)) {
        byOpenTime.set(candle.openTime, candle);
      }
      for (const candle of newCandles) {
        byOpenTime.set(candle.openTime, candle);
      }

      const sorted = [...byOpenTime.values()].sort((a, b) => a.openTime - b.openTime);
      await writeJsonLinesAtomic(filePath, sorted);
    }
  }

  private async initialStats(protectedFiles: Set<string>): Promise<HistoricalCandleCacheStats> {
    const budget = await this.enforceStorageBudget(protectedFiles);
    return {
      cacheHitCandles: 0,
      cacheMissCandles: 0,
      cacheFetchedCandles: 0,
      requests: 0,
      ...budget,
    };
  }

  private async enforceStorageBudget(protectedFiles = new Set<string>()): Promise<
    Pick<
      HistoricalCandleCacheStats,
      "cacheSizeBytes" | "cacheEvictedBytes" | "cacheEvictedFiles" | "freeBytes"
    >
  > {
    await fs.mkdir(this.rootDir, { recursive: true });
    let entries = await listCacheFiles(this.rootDir);
    let cacheSizeBytes = entries.reduce((sum, entry) => sum + entry.size, 0);
    let freeBytes = await availableBytes(this.options.dataDir);
    let cacheEvictedBytes = 0;
    let cacheEvictedFiles = 0;

    entries.sort((a, b) => a.mtimeMs - b.mtimeMs || a.filePath.localeCompare(b.filePath));
    const evictableEntries = entries.filter((entry) => !protectedFiles.has(entry.filePath));

    while (
      evictableEntries.length > 0 &&
      (cacheSizeBytes > this.options.maxBytes || freeBytes < this.options.minFreeBytes)
    ) {
      const entry = evictableEntries.shift();
      if (!entry) {
        break;
      }

      await fs.unlink(entry.filePath);
      cacheSizeBytes -= entry.size;
      cacheEvictedBytes += entry.size;
      cacheEvictedFiles += 1;
      freeBytes = await availableBytes(this.options.dataDir);
    }

    if (cacheSizeBytes > this.options.maxBytes) {
      throw new Error(
        `Historical cache limit ${formatBytes(
          this.options.maxBytes,
        )} is too small for the requested range.`,
      );
    }

    if (freeBytes < this.options.minFreeBytes) {
      throw new Error(
        `Not enough free disk space for historical cache. Available ${formatBytes(
          freeBytes,
        )}, required ${formatBytes(this.options.minFreeBytes)}.`,
      );
    }

    return {
      cacheSizeBytes,
      cacheEvictedBytes,
      cacheEvictedFiles,
      freeBytes,
    };
  }

  private filePathsForRange(startTime: number, endTime: number): string[] {
    const paths: string[] = [];
    let cursor = utcDayStart(startTime);
    const endDay = utcDayStart(endTime);

    while (cursor <= endDay) {
      paths.push(this.filePathForTime(cursor));
      cursor += DAY_MS;
    }

    return paths;
  }

  private filePathForTime(time: number): string {
    const day = new Date(utcDayStart(time)).toISOString().slice(0, 10);
    return path.join(this.rootDir, `${day}.jsonl`);
  }
}

function expectedCandleOpenTimes(
  startTime: number,
  endTime: number,
  intervalMs: number,
): number[] {
  const openTimes: number[] = [];
  const first = alignUp(startTime, intervalMs);
  const last = alignDown(endTime, intervalMs);

  for (let time = first; time <= last; time += intervalMs) {
    openTimes.push(time);
  }

  return openTimes;
}

function missingRangesFromExpected(
  expectedOpenTimes: number[],
  cachedOpenTimes: Set<number>,
  intervalMs: number,
): Array<{ startTime: number; endTime: number }> {
  const ranges: Array<{ startTime: number; endTime: number }> = [];
  let currentStart: number | undefined;
  let previousMissing: number | undefined;

  for (const openTime of expectedOpenTimes) {
    if (cachedOpenTimes.has(openTime)) {
      if (currentStart !== undefined && previousMissing !== undefined) {
        ranges.push({
          startTime: currentStart,
          endTime: previousMissing + intervalMs - 1,
        });
      }
      currentStart = undefined;
      previousMissing = undefined;
      continue;
    }

    currentStart ??= openTime;
    previousMissing = openTime;
  }

  if (currentStart !== undefined && previousMissing !== undefined) {
    ranges.push({
      startTime: currentStart,
      endTime: previousMissing + intervalMs - 1,
    });
  }

  return ranges;
}

function sumRangeCandles(
  ranges: Array<{ startTime: number; endTime: number }>,
  intervalMs: number,
): number {
  return ranges.reduce((sum, range) => {
    return sum + Math.max(0, Math.floor((range.endTime - range.startTime + 1) / intervalMs));
  }, 0);
}

function mergeBudgetStats(
  target: HistoricalCandleCacheStats,
  budget: Pick<
    HistoricalCandleCacheStats,
    "cacheSizeBytes" | "cacheEvictedBytes" | "cacheEvictedFiles" | "freeBytes"
  >,
): void {
  target.cacheSizeBytes = budget.cacheSizeBytes;
  target.freeBytes = budget.freeBytes;
  target.cacheEvictedBytes += budget.cacheEvictedBytes;
  target.cacheEvictedFiles += budget.cacheEvictedFiles;
}

async function readJsonLines<T>(filePath: string): Promise<T[]> {
  try {
    const content = await fs.readFile(filePath, "utf8");
    return content
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as T);
  } catch (error) {
    if (isMissingFile(error)) {
      return [];
    }

    throw error;
  }
}

async function writeJsonLinesAtomic(filePath: string, values: unknown[]): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.tmp`;
  await fs.writeFile(tempPath, `${values.map((value) => JSON.stringify(value)).join("\n")}\n`);
  await fs.rename(tempPath, filePath);
}

async function listCacheFiles(rootDir: string): Promise<
  Array<{ filePath: string; size: number; mtimeMs: number }>
> {
  const entries: Array<{ filePath: string; size: number; mtimeMs: number }> = [];

  async function walk(dir: string): Promise<void> {
    let dirents: Array<import("node:fs").Dirent>;
    try {
      dirents = await fs.readdir(dir, { withFileTypes: true });
    } catch (error) {
      if (isMissingFile(error)) {
        return;
      }

      throw error;
    }

    for (const dirent of dirents) {
      const filePath = path.join(dir, dirent.name);
      if (dirent.isDirectory()) {
        await walk(filePath);
      } else if (dirent.isFile() && filePath.endsWith(".jsonl")) {
        const stat = await fs.stat(filePath);
        entries.push({ filePath, size: stat.size, mtimeMs: stat.mtimeMs });
      }
    }
  }

  await walk(rootDir);
  return entries;
}

async function availableBytes(dataDir: string): Promise<number> {
  await fs.mkdir(dataDir, { recursive: true });
  const stats = await fs.statfs(dataDir);
  return Number(stats.bavail) * Number(stats.bsize);
}

function safePathPart(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9._-]+/g, "-");
}

function utcDayStart(time: number): number {
  const date = new Date(time);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function alignUp(value: number, intervalMs: number): number {
  return Math.ceil(value / intervalMs) * intervalMs;
}

function alignDown(value: number, intervalMs: number): number {
  return Math.floor(value / intervalMs) * intervalMs;
}

function isMissingFile(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(1)} ${units[unitIndex]}`;
}
