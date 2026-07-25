import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import {
  MLP_CANDLE_FEATURE_COUNT,
  MLP_CANDLE_WINDOWS,
  MLP_INPUT_FEATURE_COUNT,
  MLP_VOLUME_EMA_PERIOD_MULTIPLE,
  prepareMlpHalfFeatureAssembler,
  type MlpHalfFeatureAssembler,
  type Candle,
} from "@trading/bot-algo";

const MINUTE_MS = 60_000;
const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;
interface MlpAggregateArchive {
  hour: Candle[];
  day: Candle[];
  month: Candle[];
  quarter: Candle[];
  minuteVolumeEmaBeforeDay?: Map<number, number>;
}

interface PartialCandleHierarchy {
  minute: Candle;
  hour: Candle;
  day: Candle;
  month: Candle;
  quarter: Candle;
}

interface PartialCandleEnds {
  minute: number;
  hour: number;
  day: number;
  month: number;
  quarter: number;
}

interface HalfWindowCache {
  openTime: number;
  prefix: Uint16Array;
  completeEnd: number;
  volumeEmaBeforePartial: number | undefined;
  logOpen: number;
  high: number;
  logHigh: number;
  low: number;
  logLow: number;
}

export interface PreparedMlpFeatures {
  encode(time: number, output: Float32Array, outputOffset: number): void;
  encodeHalf(time: number, output: Uint16Array, outputOffset: number): void;
  encodeHalfRows?(
    times: readonly number[],
    output: Uint16Array,
  ): void;
}

/** Lazily prepares causal multi-resolution candle histories for MLP inference. */
export class MlpFeatureStore {
  private aggregateArchive: Promise<MlpAggregateArchive> | undefined;

  constructor(private readonly dataDir: string) {}

  async prepare(
    oneSecondCandles: readonly Candle[],
    times: readonly number[],
  ): Promise<PreparedMlpFeatures> {
    if (times.length === 0) throw new Error("MLP feature preparation requires at least one timestamp.");
    let firstTime = Number.POSITIVE_INFINITY;
    let lastTime = Number.NEGATIVE_INFINITY;
    for (const time of times) {
      if (!Number.isFinite(time)) throw new Error("MLP feature timestamps must be finite.");
      firstTime = Math.min(firstTime, time);
      lastTime = Math.max(lastTime, time);
    }
    const historyStart = utcDay(firstTime) - DAY_MS;
    const [archivedMinute, aggregate] = await Promise.all([
      this.loadMinuteRange(historyStart, lastTime + 1),
      this.aggregates(),
    ]);
    let nativeHalfAssembler: MlpHalfFeatureAssembler | undefined;
    try {
      nativeHalfAssembler = await prepareMlpHalfFeatureAssembler();
    } catch {
      // CPU-only deployments retain the exact TypeScript row encoder.
    }
    const seconds = oneSecondCandles.filter((candle) =>
      candle.openTime >= historyStart && candle.openTime <= lastTime);
    // Newly recovered one-second days can legitimately precede their archived
    // one-minute shard. Reconstruct completed minutes from the authoritative
    // seconds so current-hour features remain causal and preparation never
    // waits for a second archive refresh.
    const minute = mergeCandlesByOpenTime(
      archivedMinute,
      completedMinutesFromSeconds(seconds),
    );
    const recentAggregate = aggregateMinuteCandles(minute);
    const completeAggregate: MlpAggregateArchive = {
      hour: mergeCandlesByOpenTime(aggregate.hour, recentAggregate.hour),
      day: mergeCandlesByOpenTime(aggregate.day, recentAggregate.day),
      month: mergeCandlesByOpenTime(aggregate.month, recentAggregate.month),
      quarter: mergeCandlesByOpenTime(
        aggregate.quarter,
        recentAggregate.quarter,
      ),
      minuteVolumeEmaBeforeDay: aggregate.minuteVolumeEmaBeforeDay,
    };
    const completeSeries = new Map<string, readonly Candle[]>([
      ["1m", minute],
      ["1h", completeAggregate.hour],
      ["1d", completeAggregate.day],
      ["1M", completeAggregate.month],
      ["3M", completeAggregate.quarter],
    ]);
    const trees = new Map<string, CumulativeCandleTree>();
    const secondTree = new CumulativeCandleTree(
      seconds,
      MLP_CANDLE_WINDOWS[0].candleCount,
    );
    trees.set("1s", secondTree);
    for (const window of MLP_CANDLE_WINDOWS.slice(1)) {
      trees.set(
        window.id,
        new CumulativeCandleTree(
          completeSeries.get(window.id)!,
          window.candleCount,
          window.id === "1m" && minute.length > 0
            ? aggregate.minuteVolumeEmaBeforeDay?.get(utcDay(minute[0]!.openTime))
            : undefined,
        ),
      );
    }
    const minuteTree = trees.get("1m")!;
    const hourTree = trees.get("1h")!;
    const dayTree = trees.get("1d")!;
    const monthTree = trees.get("1M")!;
    const quarterTree = trees.get("3M")!;
    const [
      secondFeatureCount,
      minuteFeatureCount,
      hourFeatureCount,
      dayFeatureCount,
      monthFeatureCount,
      quarterFeatureCount,
    ] = MLP_CANDLE_WINDOWS.map((window) =>
      window.candleCount * MLP_CANDLE_FEATURE_COUNT);
    const halfWindowCaches: Array<HalfWindowCache | undefined> =
      new Array(5);
    const halfScratch = new Float32Array(Math.max(
      ...MLP_CANDLE_WINDOWS.map((window) =>
        window.candleCount * MLP_CANDLE_FEATURE_COUNT),
    ));
    const halfScratchBits = new Uint32Array(halfScratch.buffer);
    let partialScratch: PartialCandleHierarchy | undefined;
    let partialEnds: PartialCandleEnds | undefined;
    let partialTime: number | undefined;
    let partialSecondEnd = 0;
    let lastHalfOutput: Uint16Array | undefined;
    let lastHalfOutputOffset = 0;
    let lastHalfEncodedTime: number | undefined;
    const partialAt = (time: number): PartialCandleHierarchy => {
      if (partialTime === time && partialScratch) return partialScratch;
      const next = seconds[partialSecondEnd];
      if (partialScratch
        && partialTime !== undefined
        && time === partialTime + 1_000
        && next?.openTime === Math.floor(time / 1_000) * 1_000) {
        updatePartialCandleHierarchy(partialScratch, partialEnds!, next);
        partialSecondEnd += 1;
      } else {
        partialScratch = buildPartialCandleHierarchyFromTrees(
          secondTree,
          minuteTree,
          hourTree,
          dayTree,
          monthTree,
          time,
          partialScratch,
        );
        partialEnds = partialCandleEnds(partialScratch);
        partialSecondEnd = secondTree.upperBound(time);
      }
      partialTime = time;
      return partialScratch;
    };
    const coarseHalfCache = (
      cacheIndex: number,
      tree: CumulativeCandleTree,
      current: Candle,
      featureCount: number,
    ): HalfWindowCache => {
      let cached = halfWindowCaches[cacheIndex];
      if (!cached || cached.openTime !== current.openTime) {
        const completeEnd = tree.lowerBound(current.openTime);
        const volumeEmaBeforePartial = tree.encodeWindow(
          completeEnd,
          current,
          halfScratch,
          0,
        );
        const prefix = new Uint16Array(featureCount - MLP_CANDLE_FEATURE_COUNT);
        for (let index = 0; index < prefix.length; index += 1) {
          prefix[index] = float32BitsToFloat16Bits(halfScratchBits[index]!);
        }
        cached = {
          openTime: current.openTime,
          prefix,
          completeEnd,
          volumeEmaBeforePartial,
          logOpen: Math.log(current.open),
          high: current.high,
          logHigh: Math.log(current.high),
          low: current.low,
          logLow: Math.log(current.low),
        };
        halfWindowCaches[cacheIndex] = cached;
      }
      if (cached.high !== current.high) {
        cached.high = current.high;
        cached.logHigh = Math.log(current.high);
      }
      if (cached.low !== current.low) {
        cached.low = current.low;
        cached.logLow = Math.log(current.low);
      }
      return cached;
    };
    const encodeCoarseHalf = (
      cacheIndex: number,
      tree: CumulativeCandleTree,
      current: Candle,
      featureCount: number,
      output: Uint16Array,
      outputOffset: number,
      logClose: number,
      reusedPreviousRow: boolean,
    ): void => {
      const previous = halfWindowCaches[cacheIndex];
      const cached = coarseHalfCache(
        cacheIndex, tree, current, featureCount,
      );
      if (!reusedPreviousRow || cached !== previous) {
        output.set(cached.prefix, outputOffset);
      }
      tree.encodePartialHalf(
        current,
        cached.volumeEmaBeforePartial,
        output,
        outputOffset + cached.prefix.length,
        logClose,
        cached.logOpen,
        cached.logHigh,
        cached.logLow,
      );
    };
    return {
      encode: (time, output, outputOffset) => {
        if (!Number.isFinite(time) || outputOffset < 0
          || outputOffset + MLP_INPUT_FEATURE_COUNT > output.length) {
          throw new Error("MLP feature row output is invalid.");
        }
        const partial = partialAt(time);
        let cursor = outputOffset;
        for (const window of MLP_CANDLE_WINDOWS) {
          const tree = trees.get(window.id)!;
          if (window.id === "1s") {
            tree.encodeWindow(
              tree.upperBound(time),
              undefined,
              output,
              cursor,
            );
          } else {
            const current = partial[partialKey(window.id)];
            tree.encodeWindow(
              tree.lowerBound(current.openTime),
              current,
              output,
              cursor,
            );
          }
          cursor += window.candleCount * MLP_CANDLE_FEATURE_COUNT;
          if (window.id !== "1s") {
            const key = partialKey(window.id);
            const current = partial[key];
            output[cursor++] = (current.closeTime + 1 - current.openTime)
              / (partialEnds![key] - current.openTime);
          }
        }
        if (cursor !== outputOffset + MLP_INPUT_FEATURE_COUNT) {
          throw new Error("MLP feature encoder violated its manifest feature count.");
        }
      },
      encodeHalf: (time, output, outputOffset) => {
        if (!Number.isFinite(time) || outputOffset < 0
          || outputOffset + MLP_INPUT_FEATURE_COUNT > output.length) {
          throw new Error("MLP half feature row output is invalid.");
        }
        const partial = partialAt(time);
        const partialLogClose = Math.log(partial.minute.close);
        const reusedPreviousRow = lastHalfOutput === output
          && lastHalfEncodedTime !== undefined
          && time === lastHalfEncodedTime + 1_000
          && outputOffset === lastHalfOutputOffset + MLP_INPUT_FEATURE_COUNT;
        if (reusedPreviousRow) {
          output.copyWithin(
            outputOffset,
            lastHalfOutputOffset,
            lastHalfOutputOffset + MLP_INPUT_FEATURE_COUNT,
          );
        }
        let cursor = outputOffset;
        secondTree.encodeWindowHalf(partialSecondEnd, output, cursor);
        cursor += secondFeatureCount!;

        encodeCoarseHalf(
          0, minuteTree, partial.minute, minuteFeatureCount!,
          output, cursor, partialLogClose, reusedPreviousRow,
        );
        cursor += minuteFeatureCount!;
        output[cursor++] = halfFillFraction(
          partial.minute, partialEnds!.minute,
        );

        encodeCoarseHalf(
          1, hourTree, partial.hour, hourFeatureCount!,
          output, cursor, partialLogClose, reusedPreviousRow,
        );
        cursor += hourFeatureCount!;
        output[cursor++] = halfFillFraction(partial.hour, partialEnds!.hour);

        encodeCoarseHalf(
          2, dayTree, partial.day, dayFeatureCount!,
          output, cursor, partialLogClose, reusedPreviousRow,
        );
        cursor += dayFeatureCount!;
        output[cursor++] = halfFillFraction(partial.day, partialEnds!.day);

        encodeCoarseHalf(
          3, monthTree, partial.month, monthFeatureCount!,
          output, cursor, partialLogClose, reusedPreviousRow,
        );
        cursor += monthFeatureCount!;
        output[cursor++] = halfFillFraction(
          partial.month, partialEnds!.month,
        );

        encodeCoarseHalf(
          4, quarterTree, partial.quarter, quarterFeatureCount!,
          output, cursor, partialLogClose, reusedPreviousRow,
        );
        cursor += quarterFeatureCount!;
        output[cursor++] = halfFillFraction(
          partial.quarter, partialEnds!.quarter,
        );
        if (cursor !== outputOffset + MLP_INPUT_FEATURE_COUNT) {
          throw new Error("MLP half feature encoder violated its manifest feature count.");
        }
        lastHalfOutput = output;
        lastHalfOutputOffset = outputOffset;
        lastHalfEncodedTime = time;
      },
      encodeHalfRows: nativeHalfAssembler
        ? (times, output) => {
            if (output.length !== times.length * MLP_INPUT_FEATURE_COUNT
              || times.some((time) => !Number.isFinite(time))) {
              throw new Error("MLP half feature batch storage is invalid.");
            }
            const completeEnds = new Int32Array(times.length * 6);
            const partialFeatures = new Float32Array(times.length * 25);
            for (let row = 0; row < times.length; row += 1) {
              const time = times[row]!;
              const partial = partialAt(time);
              const logClose = Math.log(partial.minute.close);
              const endOffset = row * 6;
              const partialOffset = row * 25;
              completeEnds[endOffset] = partialSecondEnd;

              const minuteCache = coarseHalfCache(
                0, minuteTree, partial.minute, minuteFeatureCount!,
              );
              completeEnds[endOffset + 1] = minuteCache.completeEnd;
              minuteTree.encodePartial(
                partial.minute,
                minuteCache.volumeEmaBeforePartial,
                partialFeatures,
                partialOffset,
                logClose,
                minuteCache.logOpen,
                minuteCache.logHigh,
                minuteCache.logLow,
              );
              partialFeatures[partialOffset + 4] =
                (partial.minute.closeTime + 1 - partial.minute.openTime)
                / (partialEnds!.minute - partial.minute.openTime);

              const hourCache = coarseHalfCache(
                1, hourTree, partial.hour, hourFeatureCount!,
              );
              completeEnds[endOffset + 2] = hourCache.completeEnd;
              hourTree.encodePartial(
                partial.hour,
                hourCache.volumeEmaBeforePartial,
                partialFeatures,
                partialOffset + 5,
                logClose,
                hourCache.logOpen,
                hourCache.logHigh,
                hourCache.logLow,
              );
              partialFeatures[partialOffset + 9] =
                (partial.hour.closeTime + 1 - partial.hour.openTime)
                / (partialEnds!.hour - partial.hour.openTime);

              const dayCache = coarseHalfCache(
                2, dayTree, partial.day, dayFeatureCount!,
              );
              completeEnds[endOffset + 3] = dayCache.completeEnd;
              dayTree.encodePartial(
                partial.day,
                dayCache.volumeEmaBeforePartial,
                partialFeatures,
                partialOffset + 10,
                logClose,
                dayCache.logOpen,
                dayCache.logHigh,
                dayCache.logLow,
              );
              partialFeatures[partialOffset + 14] =
                (partial.day.closeTime + 1 - partial.day.openTime)
                / (partialEnds!.day - partial.day.openTime);

              const monthCache = coarseHalfCache(
                3, monthTree, partial.month, monthFeatureCount!,
              );
              completeEnds[endOffset + 4] = monthCache.completeEnd;
              monthTree.encodePartial(
                partial.month,
                monthCache.volumeEmaBeforePartial,
                partialFeatures,
                partialOffset + 15,
                logClose,
                monthCache.logOpen,
                monthCache.logHigh,
                monthCache.logLow,
              );
              partialFeatures[partialOffset + 19] =
                (partial.month.closeTime + 1 - partial.month.openTime)
                / (partialEnds!.month - partial.month.openTime);

              const quarterCache = coarseHalfCache(
                4, quarterTree, partial.quarter, quarterFeatureCount!,
              );
              completeEnds[endOffset + 5] = quarterCache.completeEnd;
              quarterTree.encodePartial(
                partial.quarter,
                quarterCache.volumeEmaBeforePartial,
                partialFeatures,
                partialOffset + 20,
                logClose,
                quarterCache.logOpen,
                quarterCache.logHigh,
                quarterCache.logLow,
              );
              partialFeatures[partialOffset + 24] =
                (partial.quarter.closeTime + 1 - partial.quarter.openTime)
                / (partialEnds!.quarter - partial.quarter.openTime);
            }
            nativeHalfAssembler({
              sources: [
                secondTree.halfFeatures(),
                minuteTree.halfFeatures(),
                hourTree.halfFeatures(),
                dayTree.halfFeatures(),
                monthTree.halfFeatures(),
                quarterTree.halfFeatures(),
              ],
              completeEnds,
              partialFeatures,
              rowCount: times.length,
              output,
            });
            if (times.length > 0) {
              lastHalfOutput = output;
              lastHalfOutputOffset =
                (times.length - 1) * MLP_INPUT_FEATURE_COUNT;
              lastHalfEncodedTime = times[times.length - 1];
            }
          }
        : undefined,
    };
  }

  async loadCompletedMinuteRange(
    startTime: number,
    endTime: number,
  ): Promise<Candle[]> {
    const archived = await this.loadMinuteRange(startTime, endTime);
    const expected = Math.max(0, Math.ceil((endTime - startTime) / MINUTE_MS));
    if (archived.length >= expected) return archived;
    const seconds = await this.loadSecondRange(startTime, endTime);
    return mergeCandlesByOpenTime(
      archived,
      completedMinutesFromSeconds(seconds),
    ).filter((candle) =>
      candle.openTime >= startTime && candle.openTime < endTime);
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
    const minuteWindow = MLP_CANDLE_WINDOWS.find((window) => window.id === "1m")!;
    const minuteAlpha = 2
      / (minuteWindow.candleCount * MLP_VOLUME_EMA_PERIOD_MULTIPLE + 1);
    const minuteBeta = 1 - minuteAlpha;
    const minuteVolumeEmaBeforeDay = new Map<number, number>();
    let minuteVolumeEma: number | undefined;
    let minuteDay: number | undefined;
    for (const file of files) {
      for (const candle of await readCandleShard(path.join(root, file))) {
        const candleDay = utcDay(candle.openTime);
        if (candleDay !== minuteDay) {
          if (minuteVolumeEma !== undefined) {
            minuteVolumeEmaBeforeDay.set(candleDay, minuteVolumeEma);
          }
          minuteDay = candleDay;
        }
        minuteVolumeEma = minuteVolumeEma === undefined
          ? candle.volume
          : minuteAlpha * candle.volume + minuteBeta * minuteVolumeEma;
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
      minuteVolumeEmaBeforeDay,
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

  async loadSecondRange(startTime: number, endTime: number): Promise<Candle[]> {
    const root = path.join(
      this.dataDir,
      "historical",
      "spot-btcusdt",
      "btcusdt",
      "1s",
    );
    const result: Candle[] = [];
    for (let day = utcDay(startTime); day < endTime; day += DAY_MS) {
      const date = new Date(day).toISOString().slice(0, 10);
      let candles: Candle[];
      try {
        candles = await readCandleShard(path.join(root, `${date}.jsonl`));
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        candles = await readCandleShard(path.join(root, `${date}.jsonl.gz`));
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

function completedMinutesFromSeconds(seconds: readonly Candle[]): Candle[] {
  const result: Candle[] = [];
  let index = 0;
  while (index < seconds.length) {
    const start = minuteBounds(seconds[index]!.openTime)[0];
    const end = start + MINUTE_MS;
    const first = index;
    while (index < seconds.length && seconds[index]!.openTime < end) index += 1;
    if (index - first !== 60
      || seconds[first]!.openTime !== start
      || seconds[index - 1]!.openTime !== end - 1_000) continue;
    let contiguous = true;
    let high = seconds[first]!.high;
    let low = seconds[first]!.low;
    let volume = 0;
    for (let child = first; child < index; child += 1) {
      if (seconds[child]!.openTime !== start + (child - first) * 1_000) {
        contiguous = false;
        break;
      }
      high = Math.max(high, seconds[child]!.high);
      low = Math.min(low, seconds[child]!.low);
      volume += seconds[child]!.volume;
    }
    if (!contiguous) continue;
    result.push({
      symbol: seconds[first]!.symbol,
      interval: "1m",
      openTime: start,
      closeTime: end - 1,
      open: seconds[first]!.open,
      high,
      low,
      close: seconds[index - 1]!.close,
      volume,
      closed: true,
    });
  }
  return result;
}

function mergeCandlesByOpenTime(
  archived: readonly Candle[],
  reconstructed: readonly Candle[],
): Candle[] {
  const byOpenTime = new Map(archived.map((candle) => [candle.openTime, candle]));
  for (const candle of reconstructed) byOpenTime.set(candle.openTime, candle);
  return [...byOpenTime.values()].sort((left, right) => left.openTime - right.openTime);
}

function aggregateMinuteCandles(minutes: readonly Candle[]): MlpAggregateArchive {
  const hour = new CompleteCandleAggregator(hourBounds, "1h");
  const day = new CompleteCandleAggregator(dayBounds, "1d");
  const month = new CompleteCandleAggregator(monthBounds, "1M");
  const quarter = new CompleteCandleAggregator(quarterBounds, "3M");
  for (const candle of minutes) {
    hour.add(candle);
    day.add(candle);
    month.add(candle);
    quarter.add(candle);
  }
  return {
    hour: hour.finish(),
    day: day.finish(),
    month: month.finish(),
    quarter: quarter.finish(),
  };
}

interface CandleTreeAggregate {
  firstIndex: number;
  lastIndex: number;
  high: number;
  low: number;
  volume: number;
  contiguous: boolean;
}

/**
 * A segment tree for causal OHLCV aggregation with canonical complete-candle
 * features precomputed once for bulk row assembly.
 */
class CumulativeCandleTree {
  private readonly leafBase: number;
  private readonly firstIndex: Int32Array;
  private readonly lastIndex: Int32Array;
  private readonly contiguous: Uint8Array;
  private readonly adjacent: Uint8Array;
  private readonly high: Float64Array;
  private readonly low: Float64Array;
  private readonly volume: Float64Array;
  private readonly featureValues: Float32Array;
  private readonly halfFeatureValues: Uint16Array;
  private readonly volumeEma: Float64Array;
  private readonly volumeAlpha: number;
  private readonly volumeBeta: number;
  private readonly leftNodes = new Int32Array(64);
  private readonly rightNodes = new Int32Array(64);
  private readonly rangeNodes = new Int32Array(128);
  private readonly lowerBoundCache = new Map<number, number>();
  private aggregateStart = -1;
  private aggregateEnd = -1;
  private readonly aggregateScratch: CandleTreeAggregate = {
    firstIndex: -1,
    lastIndex: -1,
    high: 0,
    low: 0,
    volume: 0,
    contiguous: false,
  };

  constructor(
    private readonly candles: readonly Candle[],
    private readonly expectedCount: number,
    initialVolumeEma?: number,
  ) {
    if (!Number.isInteger(expectedCount) || expectedCount <= 0) {
      throw new Error("MLP cumulative candle tree requires a positive feature window.");
    }
    this.volumeAlpha = 2
      / (expectedCount * MLP_VOLUME_EMA_PERIOD_MULTIPLE + 1);
    this.volumeBeta = 1 - this.volumeAlpha;
    let leafBase = 1;
    while (leafBase < candles.length) leafBase *= 2;
    this.leafBase = leafBase;
    const nodeCount = leafBase * 2;
    this.firstIndex = new Int32Array(nodeCount);
    this.lastIndex = new Int32Array(nodeCount);
    this.firstIndex.fill(-1);
    this.lastIndex.fill(-1);
    this.contiguous = new Uint8Array(nodeCount);
    this.adjacent = new Uint8Array(candles.length);
    this.high = new Float64Array(nodeCount);
    this.low = new Float64Array(nodeCount);
    this.volume = new Float64Array(nodeCount);
    this.featureValues = new Float32Array(
      candles.length * MLP_CANDLE_FEATURE_COUNT,
    );
    this.halfFeatureValues = new Uint16Array(this.featureValues.length);
    this.volumeEma = new Float64Array(candles.length);
    const featureBits = new Uint32Array(this.featureValues.buffer);

    for (let index = 0; index < candles.length; index += 1) {
      const candle = candles[index]!;
      validateFeatureCandle(candle);
      if (index > 0 && candles[index - 1]!.openTime >= candle.openTime) {
        throw new Error("MLP cumulative candle tree requires strictly ordered candles.");
      }
      if (index > 0) {
        const previous = candles[index - 1]!;
        this.adjacent[index] = Number(
          candle.openTime
            === intervalBounds(previous.interval, previous.openTime)[1],
        );
      }
      const node = leafBase + index;
      this.firstIndex[node] = index;
      this.lastIndex[node] = index;
      this.contiguous[node] = 1;
      this.high[node] = candle.high;
      this.low[node] = candle.low;
      this.volume[node] = candle.volume;
      const destination = index * MLP_CANDLE_FEATURE_COUNT;
      const previousVolumeEma = index > 0
        ? this.volumeEma[index - 1]
        : initialVolumeEma;
      const volumeEma = previousVolumeEma === undefined
        ? candle.volume
        : this.volumeAlpha * candle.volume
          + this.volumeBeta * previousVolumeEma;
      this.volumeEma[index] = volumeEma;
      encodePriceFeatures(candle, this.featureValues, destination);
      this.featureValues[destination + 3] = volumeEma > 0 && candle.volume > 0
        ? Math.log(candle.volume / volumeEma)
        : 0;
      for (let feature = 0; feature < MLP_CANDLE_FEATURE_COUNT; feature += 1) {
        this.halfFeatureValues[destination + feature] =
          float32BitsToFloat16Bits(featureBits[destination + feature]!);
      }
    }
    for (let node = leafBase - 1; node > 0; node -= 1) {
      this.combineNode(node, node * 2, node * 2 + 1);
    }
  }

  lowerBound(time: number): number {
    const cached = this.lowerBoundCache.get(time);
    if (cached !== undefined) return cached;
    const result = candleOpenLowerBound(this.candles, time);
    this.lowerBoundCache.set(time, result);
    return result;
  }

  upperBound(time: number): number {
    return candleOpenUpperBound(this.candles, time);
  }

  candle(index: number): Candle {
    const candle = this.candles[index];
    if (!candle) throw new Error(`MLP cumulative candle index ${index} is invalid.`);
    return candle;
  }

  halfFeatures(): Uint16Array {
    return this.halfFeatureValues;
  }

  aggregateRange(start: number, end: number): CandleTreeAggregate | undefined {
    if (start === this.aggregateStart && end === this.aggregateEnd) {
      return this.aggregateScratch.firstIndex >= 0
        ? this.aggregateScratch
        : undefined;
    }
    this.aggregateStart = start;
    this.aggregateEnd = end;
    const nodeCount = this.collectRangeNodes(start, end);
    if (nodeCount === 0) {
      this.aggregateScratch.firstIndex = -1;
      this.aggregateScratch.lastIndex = -1;
      return undefined;
    }
    let firstIndex = -1;
    let lastIndex = -1;
    let high = Number.NEGATIVE_INFINITY;
    let low = Number.POSITIVE_INFINITY;
    let volume = 0;
    let contiguous = true;
    for (let position = 0; position < nodeCount; position += 1) {
      const node = this.rangeNodes[position]!;
      const nodeFirst = this.firstIndex[node]!;
      if (nodeFirst < 0) continue;
      if (lastIndex >= 0) {
        contiguous &&= nodeFirst === lastIndex + 1
          && this.adjacent[nodeFirst] === 1;
      } else {
        firstIndex = nodeFirst;
      }
      contiguous &&= this.contiguous[node] === 1;
      lastIndex = this.lastIndex[node]!;
      high = Math.max(high, this.high[node]!);
      low = Math.min(low, this.low[node]!);
      volume += this.volume[node]!;
    }
    if (firstIndex < 0 || lastIndex < 0) return undefined;
    this.aggregateScratch.firstIndex = firstIndex;
    this.aggregateScratch.lastIndex = lastIndex;
    this.aggregateScratch.high = high;
    this.aggregateScratch.low = low;
    this.aggregateScratch.volume = volume;
    this.aggregateScratch.contiguous = contiguous;
    return this.aggregateScratch;
  }

  encodeWindow(
    completeEnd: number,
    partial: Candle | undefined,
    output: Float32Array,
    outputOffset: number,
  ): number | undefined {
    const featureCount = this.expectedCount * MLP_CANDLE_FEATURE_COUNT;
    if (!Number.isInteger(completeEnd)
      || completeEnd < 0
      || completeEnd > this.candles.length
      || outputOffset < 0
      || outputOffset + featureCount > output.length) {
      throw new Error("MLP cumulative candle encoding range is invalid.");
    }
    if (partial) validateFeatureCandle(partial);
    const completeSlots = this.expectedCount - (partial ? 1 : 0);
    const emittedStart = Math.max(0, completeEnd - completeSlots);
    const emittedLength = completeEnd - emittedStart;
    const paddingFeatures = (completeSlots - emittedLength)
      * MLP_CANDLE_FEATURE_COUNT;
    if (paddingFeatures > 0) {
      output.fill(0, outputOffset, outputOffset + paddingFeatures);
    }
    let destination = outputOffset + paddingFeatures;
    output.set(
      this.featureValues.subarray(
        emittedStart * MLP_CANDLE_FEATURE_COUNT,
        completeEnd * MLP_CANDLE_FEATURE_COUNT,
      ),
      destination,
    );
    destination += emittedLength * MLP_CANDLE_FEATURE_COUNT;
    const volumeEmaBeforePartial = completeEnd > 0
      ? this.volumeEma[completeEnd - 1]
      : undefined;
    if (partial) {
      const volumeEma = volumeEmaBeforePartial !== undefined
        ? this.volumeAlpha * partial.volume
          + this.volumeBeta * volumeEmaBeforePartial
        : partial.volume;
      encodePriceFeatures(partial, output, destination);
      output[destination + 3] = volumeEma > 0 && partial.volume > 0
        ? Math.log(partial.volume / volumeEma)
        : 0;
      destination += MLP_CANDLE_FEATURE_COUNT;
    }
    if (destination !== outputOffset + featureCount) {
      throw new Error("MLP cumulative candle encoder emitted an invalid feature count.");
    }
    return volumeEmaBeforePartial;
  }

  encodeWindowHalf(
    completeEnd: number,
    output: Uint16Array,
    outputOffset: number,
  ): void {
    const featureCount = this.expectedCount * MLP_CANDLE_FEATURE_COUNT;
    if (!Number.isInteger(completeEnd)
      || completeEnd < 0
      || completeEnd > this.candles.length
      || outputOffset < 0
      || outputOffset + featureCount > output.length) {
      throw new Error("MLP cumulative half candle encoding range is invalid.");
    }
    const emittedStart = Math.max(0, completeEnd - this.expectedCount);
    const emittedLength = completeEnd - emittedStart;
    const paddingFeatures = (this.expectedCount - emittedLength)
      * MLP_CANDLE_FEATURE_COUNT;
    if (paddingFeatures > 0) {
      output.fill(0, outputOffset, outputOffset + paddingFeatures);
    }
    let destination = outputOffset + paddingFeatures;
    output.set(
      this.halfFeatureValues.subarray(
        emittedStart * MLP_CANDLE_FEATURE_COUNT,
        completeEnd * MLP_CANDLE_FEATURE_COUNT,
      ),
      destination,
    );
    destination += emittedLength * MLP_CANDLE_FEATURE_COUNT;
    if (destination !== outputOffset + featureCount) {
      throw new Error("MLP cumulative half candle encoder emitted an invalid feature count.");
    }
  }

  encodePartialHalf(
    partial: Candle,
    volumeEmaBeforePartial: number | undefined,
    output: Uint16Array,
    outputOffset: number,
    logClose = Math.log(partial.close),
    logOpen = Math.log(partial.open),
    logHigh = Math.log(partial.high),
    logLow = Math.log(partial.low),
  ): void {
    if (outputOffset < 0
      || outputOffset + MLP_CANDLE_FEATURE_COUNT > output.length) {
      throw new Error("MLP partial half feature output is invalid.");
    }
    const logMiddle = (logOpen + logClose) / 2;
    const volumeEma = volumeEmaBeforePartial === undefined
      ? partial.volume
      : this.volumeAlpha * partial.volume
        + this.volumeBeta * volumeEmaBeforePartial;
    output[outputOffset] = float32ToFloat16Bits(
      Math.fround(logClose - logOpen),
    );
    output[outputOffset + 1] = float32ToFloat16Bits(
      Math.fround(Math.max(0, logHigh - logMiddle)),
    );
    output[outputOffset + 2] = float32ToFloat16Bits(
      Math.fround(Math.max(0, logMiddle - logLow)),
    );
    output[outputOffset + 3] = float32ToFloat16Bits(Math.fround(
      volumeEma > 0 && partial.volume > 0
        ? Math.log(partial.volume / volumeEma)
        : 0,
    ));
  }

  encodePartial(
    partial: Candle,
    volumeEmaBeforePartial: number | undefined,
    output: Float32Array,
    outputOffset: number,
    logClose = Math.log(partial.close),
    logOpen = Math.log(partial.open),
    logHigh = Math.log(partial.high),
    logLow = Math.log(partial.low),
  ): void {
    if (outputOffset < 0
      || outputOffset + MLP_CANDLE_FEATURE_COUNT > output.length) {
      throw new Error("MLP partial feature output is invalid.");
    }
    const logMiddle = (logOpen + logClose) / 2;
    const volumeEma = volumeEmaBeforePartial === undefined
      ? partial.volume
      : this.volumeAlpha * partial.volume
        + this.volumeBeta * volumeEmaBeforePartial;
    output[outputOffset] = logClose - logOpen;
    output[outputOffset + 1] = Math.max(0, logHigh - logMiddle);
    output[outputOffset + 2] = Math.max(0, logMiddle - logLow);
    output[outputOffset + 3] = volumeEma > 0 && partial.volume > 0
      ? Math.log(partial.volume / volumeEma)
      : 0;
  }

  private combineNode(destination: number, left: number, right: number): void {
    const leftFirst = this.firstIndex[left]!;
    const rightFirst = this.firstIndex[right]!;
    if (leftFirst < 0 && rightFirst < 0) return;
    if (leftFirst < 0 || rightFirst < 0) {
      const source = leftFirst >= 0 ? left : right;
      this.firstIndex[destination] = this.firstIndex[source]!;
      this.lastIndex[destination] = this.lastIndex[source]!;
      this.contiguous[destination] = this.contiguous[source]!;
      this.high[destination] = this.high[source]!;
      this.low[destination] = this.low[source]!;
      this.volume[destination] = this.volume[source]!;
      return;
    }
    const leftLast = this.lastIndex[left]!;
    this.firstIndex[destination] = leftFirst;
    this.lastIndex[destination] = this.lastIndex[right]!;
    this.contiguous[destination] = Number(
      this.contiguous[left] === 1
      && this.contiguous[right] === 1
      && rightFirst === leftLast + 1
      && this.adjacent[rightFirst] === 1,
    );
    this.high[destination] = Math.max(this.high[left]!, this.high[right]!);
    this.low[destination] = Math.min(this.low[left]!, this.low[right]!);
    this.volume[destination] = this.volume[left]! + this.volume[right]!;
  }

  private collectRangeNodes(start: number, end: number): number {
    if (!Number.isInteger(start)
      || !Number.isInteger(end)
      || start < 0
      || start > end
      || end > this.candles.length) {
      throw new Error("MLP cumulative candle query range is invalid.");
    }
    let left = start + this.leafBase;
    let right = end + this.leafBase;
    let leftCount = 0;
    let rightCount = 0;
    while (left < right) {
      if ((left & 1) === 1) this.leftNodes[leftCount++] = left++;
      if ((right & 1) === 1) this.rightNodes[rightCount++] = --right;
      left >>>= 1;
      right >>>= 1;
    }
    let count = 0;
    for (let index = 0; index < leftCount; index += 1) {
      this.rangeNodes[count++] = this.leftNodes[index]!;
    }
    for (let index = rightCount - 1; index >= 0; index -= 1) {
      this.rangeNodes[count++] = this.rightNodes[index]!;
    }
    return count;
  }
}

function encodePriceFeatures(
  candle: Pick<Candle, "open" | "high" | "low" | "close">,
  output: Float32Array,
  outputOffset: number,
): void {
  const logOpen = Math.log(candle.open);
  const logClose = Math.log(candle.close);
  const logMiddle = (logOpen + logClose) / 2;
  output[outputOffset] = logClose - logOpen;
  output[outputOffset + 1] = Math.max(0, Math.log(candle.high) - logMiddle);
  output[outputOffset + 2] = Math.max(0, logMiddle - Math.log(candle.low));
}

function halfFillFraction(candle: Candle, end: number): number {
  return float32ToFloat16Bits(Math.fround(
    (candle.closeTime + 1 - candle.openTime) / (end - candle.openTime),
  ));
}

const FLOAT32_TO_FLOAT16_SCRATCH = new Float32Array(1);
const FLOAT32_TO_FLOAT16_BITS = new Uint32Array(
  FLOAT32_TO_FLOAT16_SCRATCH.buffer,
);

function float32ToFloat16Bits(value: number): number {
  FLOAT32_TO_FLOAT16_SCRATCH[0] = value;
  return float32BitsToFloat16Bits(FLOAT32_TO_FLOAT16_BITS[0]!);
}

function float32BitsToFloat16Bits(raw: number): number {
  const sign = raw >>> 16 & 0x8000;
  let exponent = (raw >>> 23 & 0xff) - 127 + 15;
  let mantissa = raw & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | (mantissa + 0x1000 >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000;
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
  }
  return exponent >= 31
    ? sign | 0x7c00
    : sign | exponent << 10 | mantissa >>> 13;
}

function validateFeatureCandle(
  candle: Pick<Candle, "open" | "high" | "low" | "close" | "volume">,
): void {
  if (![candle.open, candle.high, candle.low, candle.close, candle.volume]
    .every(Number.isFinite)
    || candle.open <= 0
    || candle.high <= 0
    || candle.low <= 0
    || candle.close <= 0
    || candle.volume < 0
    || candle.high < Math.max(candle.open, candle.close)
    || candle.low > Math.min(candle.open, candle.close)) {
    throw new Error("MLP candle encoding received invalid OHLCV values.");
  }
}

function buildPartialCandleHierarchyFromTrees(
  seconds: CumulativeCandleTree,
  minutes: CumulativeCandleTree,
  hours: CumulativeCandleTree,
  days: CumulativeCandleTree,
  months: CumulativeCandleTree,
  time: number,
  target?: PartialCandleHierarchy,
): PartialCandleHierarchy {
  const minute = partialCandleFromTree(
    seconds, undefined, minuteBounds, "1m", time, target?.minute,
  );
  const hour = partialCandleFromTree(
    minutes, minute, hourBounds, "1h", time, target?.hour,
  );
  const day = partialCandleFromTree(
    hours, hour, dayBounds, "1d", time, target?.day,
  );
  const month = partialCandleFromTree(
    days, day, monthBounds, "1M", time, target?.month,
  );
  const quarter = partialCandleFromTree(
    months, month, quarterBounds, "3M", time, target?.quarter,
  );
  if (!target) return { minute, hour, day, month, quarter };
  return target;
}

function updatePartialCandleHierarchy(
  target: PartialCandleHierarchy,
  ends: PartialCandleEnds,
  second: Candle,
): void {
  if (second.openTime >= ends.minute) {
    ends.minute += MINUTE_MS;
    resetPartialCandle(
      target.minute, second, "1m", ends.minute - MINUTE_MS, ends.minute,
    );
  } else {
    extendPartialCandle(target.minute, second, ends.minute);
  }
  if (second.openTime >= ends.hour) {
    ends.hour += HOUR_MS;
    resetPartialCandle(
      target.hour, second, "1h", ends.hour - HOUR_MS, ends.hour,
    );
  } else {
    extendPartialCandle(target.hour, second, ends.hour);
  }
  if (second.openTime >= ends.day) {
    ends.day += DAY_MS;
    resetPartialCandle(
      target.day, second, "1d", ends.day - DAY_MS, ends.day,
    );
  } else {
    extendPartialCandle(target.day, second, ends.day);
  }
  if (second.openTime >= ends.month) {
    const [start, end] = monthBounds(second.openTime);
    ends.month = end;
    resetPartialCandle(target.month, second, "1M", start, end);
  } else {
    extendPartialCandle(target.month, second, ends.month);
  }
  if (second.openTime >= ends.quarter) {
    const [start, end] = quarterBounds(second.openTime);
    ends.quarter = end;
    resetPartialCandle(target.quarter, second, "3M", start, end);
  } else {
    extendPartialCandle(target.quarter, second, ends.quarter);
  }
}

function resetPartialCandle(
  target: Candle,
  second: Candle,
  interval: string,
  start: number,
  end: number,
): void {
  target.symbol = second.symbol;
  target.interval = interval;
  target.openTime = start;
  target.closeTime = Math.min(second.closeTime, end - 1);
  target.open = second.open;
  target.high = second.high;
  target.low = second.low;
  target.close = second.close;
  target.volume = second.volume;
  target.closed = target.closeTime >= end - 1;
}

function extendPartialCandle(
  target: Candle,
  second: Candle,
  end: number,
): void {
  if (second.openTime !== target.closeTime + 1) {
    throw new Error(`MLP incremental partial ${target.interval} candle is non-contiguous.`);
  }
  target.closeTime = Math.min(second.closeTime, end - 1);
  target.high = Math.max(target.high, second.high);
  target.low = Math.min(target.low, second.low);
  target.close = second.close;
  target.volume += second.volume;
  target.closed = target.closeTime >= end - 1;
}

function partialCandleEnds(partial: PartialCandleHierarchy): PartialCandleEnds {
  return {
    minute: partial.minute.openTime + MINUTE_MS,
    hour: partial.hour.openTime + HOUR_MS,
    day: partial.day.openTime + DAY_MS,
    month: monthBounds(partial.month.openTime)[1],
    quarter: quarterBounds(partial.quarter.openTime)[1],
  };
}

function partialCandleFromTree(
  completeChildren: CumulativeCandleTree,
  partialChild: Candle | undefined,
  bounds: (time: number) => readonly [number, number],
  interval: string,
  time: number,
  target?: Candle,
): Candle {
  const [start, end] = bounds(time);
  const first = completeChildren.lowerBound(start);
  const completeEnd = partialChild
    ? completeChildren.lowerBound(partialChild.openTime)
    : completeChildren.upperBound(time);
  const aggregate = completeChildren.aggregateRange(first, completeEnd);
  const firstChild = aggregate
    ? completeChildren.candle(aggregate.firstIndex)
    : partialChild;
  const lastComplete = aggregate
    ? completeChildren.candle(aggregate.lastIndex)
    : undefined;
  const lastChild = partialChild ?? lastComplete;
  if (!firstChild || !lastChild || firstChild.openTime !== start) {
    throw new Error(`MLP cannot form the causal partial ${interval} candle at ${new Date(time).toISOString()}.`);
  }
  if (aggregate && !aggregate.contiguous
    || partialChild && lastComplete
      && partialChild.openTime
        !== intervalBounds(lastComplete.interval, lastComplete.openTime)[1]) {
    throw new Error(`MLP partial ${interval} candle has non-contiguous source history.`);
  }
  const logicalLastClose = lastChild.closed
    ? intervalBounds(lastChild.interval, lastChild.openTime)[1] - 1
    : lastChild.closeTime;
  const high = partialChild
    ? Math.max(aggregate?.high ?? partialChild.high, partialChild.high)
    : aggregate!.high;
  const low = partialChild
    ? Math.min(aggregate?.low ?? partialChild.low, partialChild.low)
    : aggregate!.low;
  const volume = (aggregate?.volume ?? 0) + (partialChild?.volume ?? 0);
  if (target) {
    target.symbol = firstChild.symbol;
    target.interval = interval;
    target.openTime = start;
    target.closeTime = Math.min(logicalLastClose, end - 1);
    target.open = firstChild.open;
    target.high = high;
    target.low = low;
    target.close = lastChild.close;
    target.volume = volume;
    target.closed = lastChild.closeTime >= end - 1;
    return target;
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

const MONTH_BOUNDS_BY_DAY = new Map<number, readonly [number, number]>();
const QUARTER_BOUNDS_BY_DAY = new Map<number, readonly [number, number]>();

function monthBounds(time: number): readonly [number, number] {
  const day = utcDay(time);
  const cached = MONTH_BOUNDS_BY_DAY.get(day);
  if (cached) return cached;
  const value = new Date(day);
  const result = [
    Date.UTC(value.getUTCFullYear(), value.getUTCMonth(), 1),
    Date.UTC(value.getUTCFullYear(), value.getUTCMonth() + 1, 1),
  ] as const;
  MONTH_BOUNDS_BY_DAY.set(day, result);
  return result;
}

function quarterBounds(time: number): readonly [number, number] {
  const day = utcDay(time);
  const cached = QUARTER_BOUNDS_BY_DAY.get(day);
  if (cached) return cached;
  const value = new Date(day);
  const month = Math.floor(value.getUTCMonth() / 3) * 3;
  const result = [
    Date.UTC(value.getUTCFullYear(), month, 1),
    Date.UTC(value.getUTCFullYear(), month + 3, 1),
  ] as const;
  QUARTER_BOUNDS_BY_DAY.set(day, result);
  return result;
}

function utcDay(time: number): number {
  return Math.floor(time / DAY_MS) * DAY_MS;
}
