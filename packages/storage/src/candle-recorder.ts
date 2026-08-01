import fs from "node:fs/promises";
import path from "node:path";
import {
  putCandleShard,
  readCandleShardReference,
  type SequentialCandle,
} from "./candles.js";
import { SequentialShardStore } from "./store.js";

const DAY_MS = 86_400_000;
const DATE_FILE = /^(\d{4}-\d{2}-\d{2})\.jsonl$/;
const REFERENCE_FILE = /^(\d{4}-\d{2}-\d{2})\.json$/;

export interface DailyCandleRecorderOptions {
  store: SequentialShardStore;
  namespace: string;
  stagingDirectory: string;
  symbol: string;
  interval: string;
  stepMs: number;
  metadata?: Record<string, unknown>;
}

export interface DailyCandleAppendResult {
  appended: boolean;
  sealed: boolean;
  referenceFile?: string;
}

interface MutableDayState {
  count: number;
  firstOpenTime?: number;
  last?: SequentialCandle;
}

/**
 * Records an active candle date as mutable JSONL, then replaces it with one
 * immutable content-addressed shard once the UTC day (or one multi-day candle)
 * is complete.
 */
export class DailyCandleRecorder {
  readonly stagingDirectory: string;
  private readonly expectedRows: number;
  private readonly states = new Map<string, MutableDayState>();
  private writeTail = Promise.resolve();

  constructor(private readonly options: DailyCandleRecorderOptions) {
    this.stagingDirectory = path.resolve(options.stagingDirectory);
    if (!Number.isSafeInteger(options.stepMs)
      || options.stepMs < 1
      || (DAY_MS % options.stepMs !== 0 && options.stepMs % DAY_MS !== 0)) {
      throw new Error("Candle recorder interval must align to whole UTC days.");
    }
    if (!options.symbol || !options.interval) {
      throw new Error("Daily candle recorder requires a symbol and interval.");
    }
    this.expectedRows = options.stepMs >= DAY_MS ? 1 : DAY_MS / options.stepMs;
  }

  async ensureReady(): Promise<void> {
    await fs.mkdir(this.stagingDirectory, { recursive: true });
  }

  append(candle: SequentialCandle): Promise<DailyCandleAppendResult> {
    const operation = this.writeTail.then(() => this.appendSerial(candle));
    this.writeTail = operation.then(() => undefined, () => undefined);
    return operation;
  }

  async readRecent(limit: number): Promise<SequentialCandle[]> {
    if (!Number.isSafeInteger(limit) || limit < 1) {
      throw new Error("Recent candle limit must be a positive integer.");
    }
    await this.writeTail;
    const referenceDirectory = this.referenceDirectory();
    const [referenceDates, stagingDates] = await Promise.all([
      matchingDates(referenceDirectory, REFERENCE_FILE),
      matchingDates(this.stagingDirectory, DATE_FILE),
    ]);
    const dates = [...new Set([...referenceDates, ...stagingDates])]
      .sort((left, right) => right.localeCompare(left));
    const selected: SequentialCandle[] = [];
    for (const date of dates) {
      const referenceFile = this.referenceFile(date);
      const candles = await exists(referenceFile)
        ? await readCandleShardReference(referenceFile)
        : await this.readStagingDay(date);
      selected.unshift(...candles.slice(Math.max(0, candles.length - (limit - selected.length))));
      if (selected.length >= limit) break;
    }
    return selected.slice(Math.max(0, selected.length - limit));
  }

  private async appendSerial(candle: SequentialCandle): Promise<DailyCandleAppendResult> {
    validateCandle(candle, this.options);
    const date = isoDay(candle.openTime);
    const referenceFile = this.referenceFile(date);
    const stagingFile = this.stagingFile(date);
    if (await exists(referenceFile)) {
      await fs.rm(stagingFile, { force: true });
      this.states.delete(date);
      return { appended: false, sealed: true, referenceFile };
    }

    const state = await this.mutableState(date);
    if (state.last && candle.openTime <= state.last.openTime) {
      if (candle.openTime === state.last.openTime && equalCandle(candle, state.last)) {
        return { appended: false, sealed: false };
      }
      const existing = (await this.readStagingDay(date))
        .find((value) => value.openTime === candle.openTime);
      if (existing && equalCandle(candle, existing)) return { appended: false, sealed: false };
      throw new Error(`Live candle ${candle.openTime} is out of order or changed after persistence.`);
    }

    await fs.mkdir(this.stagingDirectory, { recursive: true });
    await fs.appendFile(stagingFile, `${JSON.stringify(candle)}\n`, "utf8");
    state.count += 1;
    state.firstOpenTime ??= candle.openTime;
    state.last = candle;

    const dayStart = Date.parse(`${date}T00:00:00.000Z`);
    const expectedLastOpenTime = this.expectedRows === 1
      ? dayStart
      : dayStart + DAY_MS - this.options.stepMs;
    if (state.count !== this.expectedRows
      || state.firstOpenTime !== dayStart
      || state.last.openTime !== expectedLastOpenTime) {
      return { appended: true, sealed: false };
    }

    const candles = await this.readStagingDay(date);
    const stored = await putCandleShard(this.options.store, {
      namespace: this.options.namespace,
      key: date,
      candles,
      stepMs: this.options.stepMs,
      metadata: {
        source: "live-recorder",
        completeUtcDay: this.options.stepMs <= DAY_MS,
        completeCandlePeriod: true,
        ...(this.options.metadata ?? {}),
      },
    });
    await fs.rm(stagingFile);
    this.states.delete(date);
    return { appended: true, sealed: true, referenceFile: stored.referenceFile };
  }

  private async mutableState(date: string): Promise<MutableDayState> {
    const existing = this.states.get(date);
    if (existing) return existing;
    const candles = await this.readStagingDay(date);
    const state: MutableDayState = {
      count: candles.length,
      firstOpenTime: candles[0]?.openTime,
      last: candles.at(-1),
    };
    this.states.set(date, state);
    return state;
  }

  private async readStagingDay(date: string): Promise<SequentialCandle[]> {
    let content: string;
    try {
      content = await fs.readFile(this.stagingFile(date), "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
    const candles = content.split(/\r?\n/).filter(Boolean).map((line) =>
      JSON.parse(line) as SequentialCandle);
    for (let index = 0; index < candles.length; index += 1) {
      validateCandle(candles[index]!, this.options);
      if (isoDay(candles[index]!.openTime) !== date
        || (index > 0 && candles[index]!.openTime <= candles[index - 1]!.openTime)) {
        throw new Error(`Mutable candle day is invalid or unordered: ${this.stagingFile(date)}.`);
      }
    }
    return candles;
  }

  private referenceDirectory(): string {
    return path.dirname(this.options.store.referenceFile(this.options.namespace, "placeholder"));
  }

  private referenceFile(date: string): string {
    return this.options.store.referenceFile(this.options.namespace, date);
  }

  private stagingFile(date: string): string {
    return path.join(this.stagingDirectory, `${date}.jsonl`);
  }
}

function validateCandle(
  candle: SequentialCandle,
  options: Pick<DailyCandleRecorderOptions, "symbol" | "interval" | "stepMs">,
): void {
  if (candle.symbol !== options.symbol
    || candle.interval !== options.interval
    || !candle.closed
    || !Number.isSafeInteger(candle.openTime)
    || candle.openTime % Math.min(options.stepMs, DAY_MS) !== 0
    || candle.closeTime !== candle.openTime + options.stepMs - 1
    || ![candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)) {
    throw new Error(`Invalid live ${options.symbol} ${options.interval} candle at ${candle.openTime}.`);
  }
}

function equalCandle(left: SequentialCandle, right: SequentialCandle): boolean {
  return left.symbol === right.symbol
    && left.interval === right.interval
    && left.openTime === right.openTime
    && left.closeTime === right.closeTime
    && left.open === right.open
    && left.high === right.high
    && left.low === right.low
    && left.close === right.close
    && left.volume === right.volume
    && left.closed === right.closed;
}

async function matchingDates(directory: string, pattern: RegExp): Promise<string[]> {
  try {
    return (await fs.readdir(directory))
      .map((file) => pattern.exec(file)?.[1])
      .filter((date): date is string => Boolean(date));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function exists(file: string): Promise<boolean> {
  try {
    return (await fs.stat(file)).isFile();
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function isoDay(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}
