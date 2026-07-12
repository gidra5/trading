import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { once } from "node:events";
import fs from "node:fs/promises";
import { createReadStream, createWriteStream } from "node:fs";
import path from "node:path";
import readline from "node:readline";

const DAY_MS = 24 * 60 * 60 * 1000;
const args = process.argv.slice(2);
if (args.includes("--help")) {
  console.log(`Usage: npm run fetch:candles -- [options]

  --symbol BTCUSDT
  --interval 1s
  --days 30 --end YYYY-MM-DD       Rolling UTC day range; end is inclusive
  --ranges START..END,...          Sparse UTC date ranges; ends are inclusive
  --warmup-days 0                  Prepend this many days to every range
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
const outputDir = path.resolve(
  value("--data-dir") ?? "data",
  "historical",
  `spot-${symbol.toLowerCase()}`,
  symbol.toLowerCase(),
  interval,
);

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
  if (!quiet) console.log(`Ranges: ${ranges.map(formatRange).join(", ")}`);
  let total = 0;
  for (const { start, end } of ranges) for (let day = start; day <= end; day += DAY_MS) {
    const date = new Date(day).toISOString().slice(0, 10);
    const target = path.join(outputDir, `${date}.jsonl`);
    const expected = DAY_MS / intervalMs;
    const cached = await validatedCount(target, day, expected);
    if (cached > 0) {
      total += cached;
      if (!quiet) console.log(`${date}: cached (${cached.toLocaleString()} candles)`);
      continue;
    }

    const archive = path.join(outputDir, `.${date}.zip`);
    const temporary = `${target}.tmp`;
    try {
      const url = [
        "https://data.binance.vision/data/spot/daily/klines",
        symbol,
        interval,
        `${symbol}-${interval}-${date}.zip`,
      ].map((part, index) => index === 0 ? part : encodeURIComponent(part)).join("/");
      await download(url, archive);
      await verifyChecksum(url, archive);

      const unzip = spawn("unzip", ["-p", archive], { stdio: ["ignore", "pipe", "pipe"] });
      const exited = new Promise<number | null>((resolve, reject) => {
        unzip.once("error", reject);
        unzip.once("close", resolve);
      });
      let stderr = "";
      unzip.stderr.setEncoding("utf8");
      unzip.stderr.on("data", (chunk: string) => stderr += chunk);

      const lines = readline.createInterface({ input: unzip.stdout, crlfDelay: Infinity });
      const candles = new Map<number, Candle>();
      let duplicates = 0;
      try {
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
      } catch (error) {
        unzip.kill();
        await exited.catch(() => undefined);
        throw error;
      }
      const exitCode = await exited;
      if (exitCode !== 0) throw new Error(`unzip failed (${exitCode}): ${stderr.trim()}`);

      let gaps = 0;
      let previousTime = day - intervalMs;
      const ordered = [...candles.values()].sort((a, b) => a.openTime - b.openTime);
      const output = createWriteStream(temporary, { encoding: "utf8" });
      const outputError = new Promise<never>((_, reject) => output.once("error", reject));
      for (const candle of ordered) {
        gaps += (candle.openTime - previousTime) / intervalMs - 1;
        if (!output.write(`${JSON.stringify(candle)}\n`)) {
          await Promise.race([once(output, "drain"), outputError]);
        }
        previousTime = candle.openTime;
      }
      output.end();
      await Promise.race([once(output, "finish"), outputError]);
      gaps += (day + DAY_MS - intervalMs - previousTime) / intervalMs;
      const count = ordered.length;
      if (count === 0 || count + gaps !== expected) throw new Error(`${date}: invalid archive coverage`);
      await fs.rename(temporary, target);
      total += count;
      const notes = [gaps && `${gaps.toLocaleString()} missing intervals`, duplicates && `${duplicates} duplicate rows`]
        .filter(Boolean).join(", ");
      if (!quiet) console.log(`${date}: fetched (${count.toLocaleString()} candles${notes ? `; ${notes}` : ""})`);
    } finally {
      await Promise.all([fs.rm(archive, { force: true }), fs.rm(temporary, { force: true })]);
    }
  }

  console.log(`Stored ${total.toLocaleString()} candles in ${outputDir}`);
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
    const lines = readline.createInterface({ input: createReadStream(file), crlfDelay: Infinity });
    let count = 0;
    let previousTime = day - intervalMs;
    let firstTime = -1;
    for await (const line of lines) {
      if (!line) continue;
      const candle = JSON.parse(line) as Candle;
      if (!validCandle(candle, day) || candle.openTime <= previousTime) return 0;
      if (count === 0) firstTime = candle.openTime;
      previousTime = candle.openTime;
      count += 1;
    }
    const lastTime = day + DAY_MS - intervalMs;
    return count > 0 && count <= expected && firstTime === day && previousTime === lastTime
      ? count
      : 0;
  } catch {
    return 0;
  }
}

async function verifyChecksum(url: string, file: string): Promise<void> {
  const response = await fetch(`${url}.CHECKSUM`, { signal: AbortSignal.timeout(20_000) });
  if (!response.ok) throw new Error(`${url}.CHECKSUM: HTTP ${response.status}`);
  const expected = (await response.text()).trim().split(/\s+/, 1)[0]?.toLowerCase();
  const actual = createHash("sha256").update(await fs.readFile(file)).digest("hex");
  if (!expected || expected !== actual) throw new Error(`${path.basename(file)}: checksum mismatch`);
}

async function download(url: string, file: string): Promise<void> {
  let failure: unknown;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      await fs.writeFile(file, Buffer.from(await response.arrayBuffer()));
      return;
    } catch (error) {
      failure = error;
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
    }
  }
  throw failure;
}

function formatRange(range: { start: number; end: number }): string {
  return `${new Date(range.start).toISOString().slice(0, 10)}..${new Date(range.end).toISOString().slice(0, 10)}`;
}

interface Candle {
  symbol?: string;
  interval?: string;
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closed?: boolean;
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
