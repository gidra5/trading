import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import https from "node:https";
import path from "node:path";
import readline from "node:readline";
import { Readable } from "node:stream";
import AdmZip from "adm-zip";
import {
  putCandleShard,
  readCandleShardReference,
  SequentialShardStore,
  TradingStorageLayout,
} from "@trading/storage";

const DAY_MS = 24 * 60 * 60 * 1000;
const args = process.argv.slice(2);
if (args.includes("--help")) {
  console.log(`Usage: npm run fetch:candles -- [options]

  --symbol BTCUSDT
  --interval 1s
  --days 30 --end YYYY-MM-DD       Rolling UTC day range; end is inclusive
  --ranges START..END,...          Sparse UTC date ranges; ends are inclusive
  --warmup-days 0                  Prepend this many days to every range
  --fill-gaps                     Fill no-trade intervals with zero-volume carry candles
  --quiet                          Only print the final total
  --data-dir data`);
  process.exit(0);
}
const value = (name: string): string | undefined => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
};
const symbol = (value("--symbol") ?? "BTCUSDT").toUpperCase();
const interval = value("--interval") ?? "1s";
const fillGaps = args.includes("--fill-gaps");
const quiet = args.includes("--quiet");
const days = Number(value("--days") ?? 30);
const warmupDays = Number(value("--warmup-days") ?? 0);
const intervalMs = parseInterval(interval);
const defaultEnd = new Date(Math.floor(Date.now() / DAY_MS) * DAY_MS - DAY_MS);
const end = parseDay(value("--end") ?? defaultEnd.toISOString().slice(0, 10));
const start = end - (days - 1) * DAY_MS;
const ranges = mergeRanges(
  (value("--ranges") ? parseRanges(value("--ranges")!) : [{ start, end }])
    .map((range) => ({ ...range, start: range.start - warmupDays * DAY_MS })),
);
const dataDir = path.resolve(value("--data-dir") ?? "data");
const storageLayout = new TradingStorageLayout(dataDir);
const shardStore = new SequentialShardStore(storageLayout.marketStore);
const namespace = [
  "candles",
  `spot-${symbol.toLowerCase()}`,
  symbol.toLowerCase(),
  interval,
].join("/");
const outputDir = path.dirname(shardStore.referenceFile(namespace, "placeholder"));
const temporaryDir = new TradingStorageLayout(dataDir).marketTmp;

if (!Number.isInteger(days) || days <= 0) throw new Error("--days must be a positive integer");
if (!Number.isInteger(warmupDays) || warmupDays < 0) {
  throw new Error("--warmup-days must be a non-negative integer");
}
if (DAY_MS % intervalMs !== 0) throw new Error(`${interval} does not divide a UTC day`);

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  await fs.mkdir(outputDir, { recursive: true });
  await fs.mkdir(temporaryDir, { recursive: true });
  if (!quiet) console.log(`Ranges: ${ranges.map(formatRange).join(", ")}`);
  let total = 0;
  for (const { start, end } of ranges) for (let day = start; day <= end; day += DAY_MS) {
    const date = new Date(day).toISOString().slice(0, 10);
    const target = shardStore.referenceFile(namespace, date);
    const expected = DAY_MS / intervalMs;
    const cached = await validatedCount(target, day, expected);
    if (cached > 0) {
      total += cached;
      if (!quiet) console.log(
        `${date}: cached (${cached.toLocaleString()} candles${fillGaps ? "; dense" : ""})`,
      );
      continue;
    }

    const archive = path.join(temporaryDir, `.${date}.zip`);
    try {
      const url = [
        "https://data.binance.vision/data/spot/daily/klines",
        symbol,
        interval,
        `${symbol}-${interval}-${date}.zip`,
      ].map((part, index) => index === 0 ? part : encodeURIComponent(part)).join("/");
      await download(url, archive);
      await verifyChecksum(url, archive);

      const zip = new AdmZip(archive);
      const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
      if (entries.length !== 1) {
        throw new Error(`${date}: expected one file in archive, found ${entries.length}`);
      }
      const lines = readline.createInterface({
        input: Readable.from([entries[0]!.getData()]),
        crlfDelay: Infinity,
      });
      const candles = new Map<number, Candle>();
      let duplicates = 0;
      for await (const line of lines) {
        if (!line) continue;
        const row = line.split(",");
        const openTime = milliseconds(row[0]);
        const closeTime = milliseconds(row[6]);
        const candle = {
          symbol,
          interval,
          openTime,
          open: Number(row[1]),
          high: Number(row[2]),
          low: Number(row[3]),
          close: Number(row[4]),
          volume: Number(row[5]),
          closeTime,
          closed: true,
        };
        if (!validCandle(candle, day)) {
          throw new Error(`${date}: invalid candle at ${openTime}`);
        }
        if (candles.has(openTime)) duplicates += 1;
        candles.set(openTime, candle);
      }

      let gaps = 0;
      let previousTime = day - intervalMs;
      const ordered = [...candles.values()].sort((a, b) => a.openTime - b.openTime);
      const storedCandles: Candle[] = [];
      let previous: Candle | undefined;
      let written = 0;
      if (fillGaps && ordered[0] && ordered[0].openTime > day) {
        previous = await loadPreviousCandle(day);
        if (!previous) {
          throw new Error(`${date}: cannot fill leading no-trade seconds without the prior day`);
        }
      }
      for (const candle of ordered) {
        gaps += (candle.openTime - previousTime) / intervalMs - 1;
        if (fillGaps && previous) {
          for (
            let time = Math.max(day, previous.openTime + intervalMs);
            time < candle.openTime;
            time += intervalMs
          ) {
            const filled = carryCandle(previous, time);
            storedCandles.push(filled);
            previous = filled;
            written += 1;
          }
        }
        storedCandles.push(candle);
        written += 1;
        previous = candle;
        previousTime = candle.openTime;
      }
      if (fillGaps && previous) {
        for (
          let time = Math.max(day, previous.openTime + intervalMs);
          time < day + DAY_MS;
          time += intervalMs
        ) {
          const filled = carryCandle(previous, time);
          storedCandles.push(filled);
          previous = filled;
          written += 1;
        }
      }
      gaps += (day + DAY_MS - intervalMs - previousTime) / intervalMs;
      const count = fillGaps ? written : ordered.length;
      if (count === 0 || ordered.length + gaps !== expected || (fillGaps && count !== expected)) {
        throw new Error(`${date}: invalid archive coverage`);
      }
      if (count !== expected) {
        throw new Error(
          `${date}: sequential storage requires a dense UTC day; re-run with --fill-gaps`,
        );
      }
      await putCandleShard(shardStore, {
          namespace,
          key: date,
          candles: storedCandles,
          stepMs: intervalMs,
          metadata: {
            source: "data.binance.vision",
            market: "spot",
            symbol,
            interval,
            completeUtcDay: count === expected,
            gapFilled: fillGaps,
          },
        });
      total += count;
      const notes = [gaps && `${gaps.toLocaleString()} missing intervals`, duplicates && `${duplicates} duplicate rows`]
        .filter(Boolean).join(", ");
      if (!quiet) console.log(`${date}: fetched (${count.toLocaleString()} candles${notes ? `; ${notes}` : ""})`);
    } finally {
      await fs.rm(archive, { force: true });
    }
  }

  console.log(`Stored ${total.toLocaleString()} candles in ${outputDir}`);
}

async function loadPreviousCandle(day: number): Promise<Candle | undefined> {
  const date = new Date(day - DAY_MS).toISOString().slice(0, 10);
  try {
    const candles = await readCandleShardReference(
      shardStore.referenceFile(namespace, date),
    );
    const last = candles.at(-1);
    return last && validCandle(last, day - DAY_MS)
      && last.openTime === day - intervalMs
      ? last
      : undefined;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

function carryCandle(previous: Candle, openTime: number): Candle {
  return {
    symbol,
    interval,
    openTime,
    open: previous.close,
    high: previous.close,
    low: previous.close,
    close: previous.close,
    volume: 0,
    closeTime: openTime + intervalMs - 1,
    closed: true,
  };
}

function parseDay(day: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`Invalid UTC date: ${day}`);
  const time = Date.parse(`${day}T00:00:00.000Z`);
  if (!Number.isFinite(time) || new Date(time).toISOString().slice(0, 10) !== day) {
    throw new Error(`Invalid UTC date: ${day}`);
  }
  return time;
}

function parseRanges(input: string): Array<{ start: number; end: number }> {
  return input.split(",").filter(Boolean).map((part) => {
    const pieces = part.split("..");
    if (pieces.length !== 2) throw new Error(`Invalid UTC range: ${part}`);
    const range = { start: parseDay(pieces[0]!), end: parseDay(pieces[1]!) };
    if (range.end < range.start) throw new Error(`Invalid UTC range: ${part}`);
    return range;
  });
}

function mergeRanges(input: Array<{ start: number; end: number }>): Array<{ start: number; end: number }> {
  const ranges = input.slice().sort((a, b) => a.start - b.start);
  const merged: Array<{ start: number; end: number }> = [];
  for (const range of ranges) {
    const previous = merged.at(-1);
    if (previous && range.start <= previous.end + DAY_MS) previous.end = Math.max(previous.end, range.end);
    else merged.push({ ...range });
  }
  return merged;
}

function parseInterval(input: string): number {
  const match = /^(\d+)([smhd])$/.exec(input);
  if (!match) throw new Error(`Unsupported interval: ${input}`);
  const units = { s: 1_000, m: 60_000, h: 3_600_000, d: DAY_MS };
  return Number(match[1]) * units[match[2] as keyof typeof units];
}

function milliseconds(input: string): number {
  const value = Number(input);
  return value >= 1e15 ? Math.floor(value / 1_000) : value;
}

async function validatedCount(file: string, day: number, expected: number): Promise<number> {
  try {
    const candles = await readCandleShardReference(file);
    if (candles.length === 0 || candles.length > expected) return 0;
    let previousTime = day - intervalMs;
    for (const candle of candles) {
      if (!validCandle(candle, day) || candle.openTime <= previousTime) return 0;
      previousTime = candle.openTime;
    }
    return candles[0]!.openTime === day
      && previousTime === day + DAY_MS - intervalMs
      ? candles.length
      : 0;
  } catch {
    return 0;
  }
}

async function verifyChecksum(url: string, file: string): Promise<void> {
  const expected = (await requestBuffer(`${url}.CHECKSUM`, 20_000))
    .toString("utf8").trim().split(/\s+/, 1)[0]?.toLowerCase();
  const actual = createHash("sha256").update(await fs.readFile(file)).digest("hex");
  if (!expected || expected !== actual) throw new Error(`${path.basename(file)}: checksum mismatch`);
}

async function download(url: string, file: string): Promise<void> {
  let failure: unknown;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      await fs.writeFile(file, await requestBuffer(url, 60_000));
      return;
    } catch (error) {
      failure = error;
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
    }
  }
  throw failure;
}

function requestBuffer(url: string, timeoutMs: number, redirects = 5): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const request = https.get(url, { headers: { "user-agent": "trading-history-fetcher/1.0" } }, (response) => {
      const status = response.statusCode ?? 0;
      if (status >= 300 && status < 400 && response.headers.location) {
        response.resume();
        if (redirects <= 0) {
          reject(new Error(`${url}: too many redirects`));
          return;
        }
        requestBuffer(new URL(response.headers.location, url).toString(), timeoutMs, redirects - 1)
          .then(resolve, reject);
        return;
      }
      if (status < 200 || status >= 300) {
        response.resume();
        reject(new Error(`${url}: HTTP ${status}`));
        return;
      }
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => chunks.push(chunk));
      response.once("end", () => resolve(Buffer.concat(chunks)));
      response.once("error", reject);
    });
    request.setTimeout(timeoutMs, () => request.destroy(new Error(`${url}: timed out`)));
    request.once("error", reject);
  });
}

function formatRange(range: { start: number; end: number }): string {
  return `${new Date(range.start).toISOString().slice(0, 10)}..${new Date(range.end).toISOString().slice(0, 10)}`;
}

interface Candle {
  symbol: string;
  interval: string;
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closed: boolean;
}

function validCandle(candle: Candle, day: number): boolean {
  return candle.symbol === symbol
    && candle.interval === interval
    && candle.closed === true
    && Number.isSafeInteger(candle.openTime)
    && candle.openTime >= day && candle.openTime < day + DAY_MS
    && (candle.openTime - day) % intervalMs === 0
    && Number.isSafeInteger(candle.closeTime)
    && candle.closeTime >= candle.openTime && candle.closeTime < candle.openTime + intervalMs
    && [candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)
    && candle.volume >= 0
    && candle.high >= Math.max(candle.open, candle.low, candle.close)
    && candle.low <= Math.min(candle.open, candle.high, candle.close);
}
