import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { once } from "node:events";
import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { pipeline } from "node:stream/promises";
import { createGzip } from "node:zlib";
import type { Candle } from "@trading/bot-algo";

const DAY_MS = 86_400_000;
const BINANCE_SPOT_ARCHIVE_ROOT = "https://data.binance.vision/data/spot/daily/klines";

export interface BinanceSpotDailyShardRequest {
  dataDir: string;
  date: string;
  day: number;
  symbol?: string;
  interval?: string;
  intervalMs?: number;
  archiveRoot?: string;
}

/**
 * Downloads one complete Binance spot UTC-day kline archive and installs it as
 * a compressed JSONL shard. The target is renamed into place only after the
 * checksum, archive extraction, ordering, and full-day boundary checks pass.
 */
export async function fetchBinanceSpotDailyShard(
  request: BinanceSpotDailyShardRequest,
): Promise<void> {
  const symbol = (request.symbol ?? "BTCUSDT").toUpperCase();
  const interval = request.interval ?? "1s";
  const intervalMs = request.intervalMs ?? 1_000;
  if (!Number.isInteger(intervalMs) || intervalMs <= 0 || DAY_MS % intervalMs !== 0) {
    throw new Error(`Daily shard interval must divide one UTC day exactly; received ${intervalMs}ms.`);
  }
  const archiveRoot = request.archiveRoot ?? BINANCE_SPOT_ARCHIVE_ROOT;
  const outputDir = path.join(
    request.dataDir,
    "historical",
    `spot-${symbol.toLowerCase()}`,
    symbol.toLowerCase(),
    interval,
  );
  const target = path.join(outputDir, `${request.date}.jsonl.gz`);
  const temporarySuffix = `${process.pid}-${randomUUID()}`;
  const archive = path.join(outputDir, `.${request.date}.${temporarySuffix}.zip`);
  const temporary = path.join(outputDir, `.${request.date}.${temporarySuffix}.jsonl.gz`);
  const archiveUrl = [archiveRoot, symbol, interval, `${symbol}-${interval}-${request.date}.zip`]
    .map((part, index) => index === 0 ? part.replace(/\/$/, "") : encodeURIComponent(part))
    .join("/");

  await fs.mkdir(outputDir, { recursive: true });
  try {
    const bytes = await requestBuffer(archiveUrl);
    await fs.writeFile(archive, bytes);
    await verifyChecksum(archiveUrl, bytes);
    await extractDailyShard({
      archive,
      temporary,
      symbol,
      interval,
      intervalMs,
      day: request.day,
      date: request.date,
    });
    await fs.rename(temporary, target);
  } finally {
    await Promise.all([
      fs.rm(archive, { force: true }),
      fs.rm(temporary, { force: true }),
    ]);
  }
}

async function extractDailyShard(options: {
  archive: string;
  temporary: string;
  symbol: string;
  interval: string;
  intervalMs: number;
  day: number;
  date: string;
}): Promise<void> {
  const unzip = spawn("unzip", ["-p", options.archive], { stdio: ["ignore", "pipe", "pipe"] });
  const exited = new Promise<number | null>((resolve, reject) => {
    unzip.once("error", reject);
    unzip.once("close", resolve);
  });
  let stderr = "";
  unzip.stderr.setEncoding("utf8");
  unzip.stderr.on("data", (chunk: string) => stderr += chunk);

  const lines = readline.createInterface({ input: unzip.stdout, crlfDelay: Infinity });
  const gzip = createGzip({ level: 6 });
  const outputDone = pipeline(gzip, createWriteStream(options.temporary));
  let count = 0;
  let firstTime: number | undefined;
  let previousTime: number | undefined;
  try {
    for await (const line of lines) {
      if (!line) continue;
      const row = line.split(",");
      const candle = parseArchiveCandle(options.symbol, options.interval, row);
      if (!validCandle(candle, options.day, options.intervalMs)) {
        throw new Error(`${options.date}: invalid Binance candle at ${candle.openTime}`);
      }
      if (previousTime !== undefined && candle.openTime !== previousTime + options.intervalMs) {
        throw new Error(`${options.date}: Binance candles are missing, duplicated, or out of order`);
      }
      firstTime ??= candle.openTime;
      previousTime = candle.openTime;
      count += 1;
      if (!gzip.write(`${JSON.stringify(candle)}\n`)) await once(gzip, "drain");
    }
    gzip.end();
    await outputDone;
    const expectedLastTime = options.day + DAY_MS - options.intervalMs;
    const expectedCount = DAY_MS / options.intervalMs;
    if (count !== expectedCount || firstTime !== options.day || previousTime !== expectedLastTime) {
      throw new Error(`${options.date}: Binance archive does not cover the complete UTC day`);
    }
  } catch (error) {
    unzip.kill();
    gzip.destroy();
    await Promise.allSettled([exited, outputDone]);
    throw error;
  }

  const exitCode = await exited;
  if (exitCode !== 0) {
    throw new Error(`unzip failed (${exitCode}): ${stderr.trim() || "no diagnostic output"}`);
  }
}

function parseArchiveCandle(symbol: string, interval: string, row: string[]): Candle {
  return {
    symbol,
    interval,
    openTime: milliseconds(row[0]),
    open: Number(row[1]),
    high: Number(row[2]),
    low: Number(row[3]),
    close: Number(row[4]),
    volume: Number(row[5]),
    closeTime: milliseconds(row[6]),
    closed: true,
  };
}

function validCandle(candle: Candle, day: number, intervalMs: number): boolean {
  return Number.isSafeInteger(candle.openTime)
    && candle.openTime >= day && candle.openTime < day + DAY_MS
    && (candle.openTime - day) % intervalMs === 0
    && Number.isSafeInteger(candle.closeTime)
    && candle.closeTime >= candle.openTime && candle.closeTime < candle.openTime + intervalMs
    && [candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite)
    && candle.volume >= 0
    && candle.high >= Math.max(candle.open, candle.low, candle.close)
    && candle.low <= Math.min(candle.open, candle.high, candle.close);
}

function milliseconds(input: string | undefined): number {
  const value = Number(input);
  return value >= 1e15 ? Math.floor(value / 1_000) : value;
}

async function verifyChecksum(url: string, bytes: Buffer): Promise<void> {
  const checksum = (await requestBuffer(`${url}.CHECKSUM`)).toString("utf8").trim().split(/\s+/, 1)[0];
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (!checksum || checksum.toLowerCase() !== actual) {
    throw new Error(`${path.basename(url)}: Binance checksum mismatch`);
  }
}

async function requestBuffer(url: string): Promise<Buffer> {
  let failure: unknown;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { "user-agent": "trading-history-fetcher/1.0" },
        signal: AbortSignal.timeout(60_000),
      });
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      failure = error;
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
    }
  }
  throw failure;
}
