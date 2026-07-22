import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import {
  MLP_CANDLE_FEATURE_COUNT,
  MLP_CANDLE_WINDOWS,
  MLP_INPUT_FEATURE_COUNT,
  MLP_STATE_FEATURES,
  MLP_VOLUME_EMA_WARMUP_MULTIPLE,
  encodeMlpCandleWindow,
  encodeMlpStateInputs,
  type Candle,
  type MlpExposureStateInputs,
} from "@trading/bot-algo";

const MINUTE_MS = 60_000;
const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;

interface MlpAggregateArchive {
  hour: Candle[];
  day: Candle[];
  month: Candle[];
  quarter: Candle[];
}

interface PartialCandleHierarchy {
  minute: Candle;
  hour: Candle;
  day: Candle;
  month: Candle;
  quarter: Candle;
}

export interface PreparedMlpFeatures {
  encode(time: number, output: Float32Array, outputOffset: number): void;
}

/** Lazily prepares causal multi-resolution candle histories for MLP inference. */
export class MlpFeatureStore {
  private aggregateArchive: Promise<MlpAggregateArchive> | undefined;

  constructor(private readonly dataDir: string) {}

  async prepare(
    oneSecondCandles: readonly Candle[],
    times: readonly number[],
    state: MlpExposureStateInputs,
  ): Promise<PreparedMlpFeatures> {
    if (times.length === 0) throw new Error("MLP feature preparation requires at least one timestamp.");
    let firstTime = Number.POSITIVE_INFINITY;
    let lastTime = Number.NEGATIVE_INFINITY;
    for (const time of times) {
      if (!Number.isFinite(time)) throw new Error("MLP feature timestamps must be finite.");
      firstTime = Math.min(firstTime, time);
      lastTime = Math.max(lastTime, time);
    }
    const [minute, aggregate] = await Promise.all([
      this.loadMinuteRange(firstTime - DAY_MS, lastTime + 1),
      this.aggregates(),
    ]);
    const completeSeries = new Map<string, readonly Candle[]>([
      ["1m", minute],
      ["1h", aggregate.hour],
      ["1d", aggregate.day],
      ["1M", aggregate.month],
      ["3M", aggregate.quarter],
    ]);
    const encodedCaches = new Map<string, Map<number, { values: Float32Array; fill?: number }>>();
    const partialCache = new Map<number, PartialCandleHierarchy>();
    const stateVector = encodeMlpStateInputs(state);
    return {
      encode: (time, output, outputOffset) => {
        if (!Number.isFinite(time) || outputOffset < 0
          || outputOffset + MLP_INPUT_FEATURE_COUNT > output.length) {
          throw new Error("MLP feature row output is invalid.");
        }
        let cursor = outputOffset;
        for (const window of MLP_CANDLE_WINDOWS) {
          let cache = encodedCaches.get(window.id);
          if (!cache) {
            cache = new Map();
            encodedCaches.set(window.id, cache);
          }
          let encoded = cache.get(time);
          if (!encoded) {
            const warmupCount = window.candleCount * MLP_VOLUME_EMA_WARMUP_MULTIPLE;
            if (window.id === "1s") {
              const end = candleOpenUpperBound(oneSecondCandles, time);
              encoded = {
                values: encodeMlpCandleWindow(
                  oneSecondCandles.slice(Math.max(0, end - warmupCount), end),
                  window.candleCount,
                ),
              };
            } else {
              let partial = partialCache.get(time);
              if (!partial) {
                partial = buildPartialCandleHierarchy(oneSecondCandles, minute, aggregate, time);
                partialCache.set(time, partial);
              }
              const current = partial[partialKey(window.id)];
              const complete = completeSeries.get(window.id)!;
              const completeEnd = candleOpenLowerBound(complete, current.openTime);
              const history = complete.slice(
                Math.max(0, completeEnd - (warmupCount - 1)),
                completeEnd,
              );
              encoded = {
                values: encodeMlpCandleWindow([...history, current], window.candleCount),
                fill: candleFillFraction(current),
              };
            }
            cache.set(time, encoded);
          }
          output.set(encoded.values, cursor);
          cursor += window.candleCount * MLP_CANDLE_FEATURE_COUNT;
          if (window.id !== "1s") output[cursor++] = encoded.fill!;
        }
        output.set(stateVector, cursor);
        cursor += MLP_STATE_FEATURES.length;
        if (cursor !== outputOffset + MLP_INPUT_FEATURE_COUNT) {
          throw new Error("MLP feature encoder violated its manifest feature count.");
        }
      },
    };
  }

  private aggregates(): Promise<MlpAggregateArchive> {
    this.aggregateArchive ??= this.buildAggregateArchive();
    return this.aggregateArchive;
  }

  private async buildAggregateArchive(): Promise<MlpAggregateArchive> {
    const root = this.minuteRoot();
    const filesByDate = new Map<string, string>();
    for (const file of (await fs.readdir(root)).filter(isDailyCandleShard).sort()) {
      const date = file.slice(0, 10);
      const existing = filesByDate.get(date);
      if (!existing || existing.endsWith(".gz") && !file.endsWith(".gz")) {
        filesByDate.set(date, file);
      }
    }
    const files = [...filesByDate.values()].sort();
    if (files.length === 0) throw new Error("MLP feature archive has no BTCUSDT 1m candles.");
    const hour = new CompleteCandleAggregator(hourBounds, "1h");
    const day = new CompleteCandleAggregator(dayBounds, "1d");
    const month = new CompleteCandleAggregator(monthBounds, "1M");
    const quarter = new CompleteCandleAggregator(quarterBounds, "3M");
    for (const file of files) {
      for (const candle of await readCandleShard(path.join(root, file))) {
        hour.add(candle);
        day.add(candle);
        month.add(candle);
        quarter.add(candle);
      }
    }
    return {
      hour: hour.finish(),
      day: day.finish(),
      month: month.finish(),
      quarter: quarter.finish(),
    };
  }

  private async loadMinuteRange(startTime: number, endTime: number): Promise<Candle[]> {
    const root = this.minuteRoot();
    const result: Candle[] = [];
    for (let day = utcDay(startTime); day < endTime; day += DAY_MS) {
      const date = new Date(day).toISOString().slice(0, 10);
      let candles: Candle[];
      try {
        candles = await readCandleShard(path.join(root, `${date}.jsonl`));
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        try {
          candles = await readCandleShard(path.join(root, `${date}.jsonl.gz`));
        } catch (compressedError) {
          if ((compressedError as NodeJS.ErrnoException).code === "ENOENT") continue;
          throw compressedError;
        }
      }
      for (const candle of candles) {
        if (candle.openTime >= startTime && candle.openTime < endTime) result.push(candle);
      }
    }
    result.sort((left, right) => left.openTime - right.openTime);
    return result;
  }

  private minuteRoot(): string {
    return path.join(
      this.dataDir,
      "historical",
      "spot-btcusdt",
      "btcusdt",
      "1m",
    );
  }
}

function buildPartialCandleHierarchy(
  seconds: readonly Candle[],
  minutes: readonly Candle[],
  aggregate: MlpAggregateArchive,
  time: number,
): PartialCandleHierarchy {
  const minute = partialCandleFromSeries(seconds, undefined, minuteBounds, "1m", time);
  const hour = partialCandleFromSeries(minutes, minute, hourBounds, "1h", time);
  const day = partialCandleFromSeries(aggregate.hour, hour, dayBounds, "1d", time);
  const month = partialCandleFromSeries(aggregate.day, day, monthBounds, "1M", time);
  const quarter = partialCandleFromSeries(aggregate.month, month, quarterBounds, "3M", time);
  return { minute, hour, day, month, quarter };
}

function partialCandleFromSeries(
  completeChildren: readonly Candle[],
  partialChild: Candle | undefined,
  bounds: (time: number) => readonly [number, number],
  interval: string,
  time: number,
): Candle {
  const [start, end] = bounds(time);
  const first = candleOpenLowerBound(completeChildren, start);
  const completeEnd = partialChild
    ? candleOpenLowerBound(completeChildren, partialChild.openTime)
    : candleOpenUpperBound(completeChildren, time);
  const children = completeChildren.slice(first, completeEnd);
  if (partialChild) children.push(partialChild);
  if (children.length === 0 || children[0]!.openTime !== start) {
    throw new Error(`MLP cannot form the causal partial ${interval} candle at ${new Date(time).toISOString()}.`);
  }
  for (let index = 1; index < children.length; index += 1) {
    const previous = children[index - 1]!;
    if (children[index]!.openTime !== intervalBounds(previous.interval, previous.openTime)[1]) {
      throw new Error(`MLP partial ${interval} candle has non-contiguous source history.`);
    }
  }
  const firstChild = children[0]!;
  const lastChild = children.at(-1)!;
  const logicalLastClose = lastChild.closed
    ? intervalBounds(lastChild.interval, lastChild.openTime)[1] - 1
    : lastChild.closeTime;
  let high = firstChild.high;
  let low = firstChild.low;
  let volume = 0;
  for (const child of children) {
    high = Math.max(high, child.high);
    low = Math.min(low, child.low);
    volume += child.volume;
  }
  return {
    symbol: firstChild.symbol,
    interval,
    openTime: start,
    closeTime: Math.min(logicalLastClose, end - 1),
    open: firstChild.open,
    high,
    low,
    close: lastChild.close,
    volume,
    closed: lastChild.closeTime >= end - 1,
  };
}

function partialKey(id: Exclude<typeof MLP_CANDLE_WINDOWS[number]["id"], "1s">): keyof PartialCandleHierarchy {
  if (id === "1m") return "minute";
  if (id === "1h") return "hour";
  if (id === "1d") return "day";
  if (id === "1M") return "month";
  return "quarter";
}

function candleFillFraction(candle: Candle): number {
  const [, end] = intervalBounds(candle.interval, candle.openTime);
  return Math.max(0, Math.min(1, (candle.closeTime + 1 - candle.openTime) / (end - candle.openTime)));
}

function intervalBounds(interval: string, time: number): readonly [number, number] {
  if (interval === "1s") {
    const start = Math.floor(time / 1_000) * 1_000;
    return [start, start + 1_000];
  }
  if (interval === "1m") return minuteBounds(time);
  if (interval === "1h") return hourBounds(time);
  if (interval === "1d") return dayBounds(time);
  if (interval === "1M") return monthBounds(time);
  if (interval === "3M") return quarterBounds(time);
  throw new Error(`Unsupported partial MLP candle interval '${interval}'.`);
}

class CompleteCandleAggregator {
  private result: Candle[] = [];
  private current: Candle | undefined;
  private currentStart = 0;
  private currentEnd = 0;
  private count = 0;
  private contiguous = true;
  private previousOpen = 0;

  constructor(
    private readonly bounds: (time: number) => readonly [number, number],
    private readonly interval: string,
  ) {}

  add(candle: Candle): void {
    const [start, end] = this.bounds(candle.openTime);
    if (!this.current || start !== this.currentStart) {
      this.flush();
      this.currentStart = start;
      this.currentEnd = end;
      this.count = 1;
      this.contiguous = candle.openTime === start;
      this.previousOpen = candle.openTime;
      this.current = {
        ...candle,
        interval: this.interval,
        openTime: start,
        closeTime: end - 1,
      };
      return;
    }
    this.contiguous &&= candle.openTime === this.previousOpen + MINUTE_MS;
    this.previousOpen = candle.openTime;
    this.count += 1;
    this.current.high = Math.max(this.current.high, candle.high);
    this.current.low = Math.min(this.current.low, candle.low);
    this.current.close = candle.close;
    this.current.volume += candle.volume;
  }

  finish(): Candle[] {
    this.flush();
    return this.result;
  }

  private flush(): void {
    if (this.current && this.contiguous
      && this.count === (this.currentEnd - this.currentStart) / MINUTE_MS
      && this.previousOpen === this.currentEnd - MINUTE_MS) {
      this.result.push(this.current);
    }
    this.current = undefined;
  }
}

async function readCandleShard(file: string): Promise<Candle[]> {
  const content = file.endsWith(".gz")
    ? gunzipSync(await fs.readFile(file)).toString("utf8")
    : await fs.readFile(file, "utf8");
  const result: Candle[] = [];
  for (const line of content.split("\n")) {
    if (line) result.push(JSON.parse(line) as Candle);
  }
  return result;
}

function isDailyCandleShard(file: string): boolean {
  return /^\d{4}-\d{2}-\d{2}\.jsonl(?:\.gz)?$/.test(file);
}

function candleOpenLowerBound(candles: readonly Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.openTime < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function candleOpenUpperBound(candles: readonly Candle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.openTime <= time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function minuteBounds(time: number): readonly [number, number] {
  const start = Math.floor(time / MINUTE_MS) * MINUTE_MS;
  return [start, start + MINUTE_MS];
}

function hourBounds(time: number): readonly [number, number] {
  const start = Math.floor(time / HOUR_MS) * HOUR_MS;
  return [start, start + HOUR_MS];
}

function dayBounds(time: number): readonly [number, number] {
  const start = utcDay(time);
  return [start, start + DAY_MS];
}

function monthBounds(time: number): readonly [number, number] {
  const value = new Date(time);
  return [
    Date.UTC(value.getUTCFullYear(), value.getUTCMonth(), 1),
    Date.UTC(value.getUTCFullYear(), value.getUTCMonth() + 1, 1),
  ];
}

function quarterBounds(time: number): readonly [number, number] {
  const value = new Date(time);
  const month = Math.floor(value.getUTCMonth() / 3) * 3;
  return [
    Date.UTC(value.getUTCFullYear(), month, 1),
    Date.UTC(value.getUTCFullYear(), month + 3, 1),
  ];
}

function utcDay(time: number): number {
  const value = new Date(time);
  return Date.UTC(value.getUTCFullYear(), value.getUTCMonth(), value.getUTCDate());
}
