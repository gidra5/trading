import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import {
  DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS,
  putDerivativesKlinesShard,
  putDerivativesMetricsShard,
  putTradeFlowShard,
  SequentialShardStore,
  TradingStorageLayout,
} from "@trading/storage";
import { parseArchive } from "./fetch-binance-spot-agg-trades.js";
import {
  parseKlinesArchive,
} from "./fetch-binance-usdm-klines.js";
import { parseMetricsArchive } from "./fetch-binance-usdm-metrics.js";

const DAY_MS = 86_400_000;
const SOURCES = ["spot-flow", "futures-klines", "futures-metrics"] as const;
type Source = typeof SOURCES[number];

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:research-forward-market -- [options]

  --symbol BTCUSDT                  Binance symbol
  --ranges START..END,...           Inclusive UTC date ranges
  --only YYYY-MM-DD,...             Explicit UTC dates
  --sources spot-flow,futures-klines,futures-metrics
  --data-dir data
  --quiet

This command writes to research-only namespaces and never changes the sealed oracle corpus.`);
    return;
  }
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const dates = value("--only")
    ? parseDays(value("--only")!.split(","))
    : expandRanges(value("--ranges") ?? "2026-07-18..2026-08-16");
  const sources = (value("--sources") ?? SOURCES.join(",")).split(",") as Source[];
  for (const source of sources) {
    if (!SOURCES.includes(source)) throw new Error(`Unsupported source: ${source}`);
  }
  const quiet = args.includes("--quiet");
  const symbol = (value("--symbol") ?? "BTCUSDT").toUpperCase();
  if (!/^[A-Z0-9]{4,30}$/.test(symbol)) throw new Error(`Invalid symbol: ${symbol}`);
  const namespaces = namespacesFor(symbol);
  const layout = new TradingStorageLayout(path.resolve(value("--data-dir") ?? "data"));
  const store = new SequentialShardStore(layout.marketStore);
  const temporary = path.join(layout.marketTmp, "research-forward-market", symbol.toLowerCase());
  await fs.mkdir(temporary, { recursive: true });
  const result: Record<Source, { stored: number; cached: number }> = Object.fromEntries(
    SOURCES.map((source) => [source, { stored: 0, cached: 0 }]),
  ) as Record<Source, { stored: number; cached: number }>;
  for (const date of dates) {
    for (const source of sources) {
      if (await cached(store, namespaces[source], date, source)) {
        result[source].cached += 1;
        continue;
      }
      if (source === "spot-flow") await ingestSpotFlow(store, temporary, date, symbol, namespaces);
      if (source === "futures-klines") await ingestFuturesKlines(store, date, symbol, namespaces);
      if (source === "futures-metrics") await ingestFuturesMetrics(store, date, symbol, namespaces);
      result[source].stored += 1;
      if (!quiet) console.log(`${date}: stored ${source}`);
    }
  }
  for (const source of sources) {
    console.log(`${source}: ${result[source].stored} stored, ${result[source].cached} cached`);
  }
}

async function ingestSpotFlow(
  store: SequentialShardStore,
  temporary: string,
  date: string,
  symbol: string,
  namespaces: Record<Source, string>,
): Promise<void> {
  const archiveName = `${symbol}-aggTrades-${date}.zip`;
  const csvName = `${symbol}-aggTrades-${date}.csv`;
  const url = `https://data.binance.vision/data/spot/daily/aggTrades/${symbol}/${archiveName}`;
  const checksum = await fetchChecksum(url);
  const archiveFile = path.join(temporary, archiveName);
  try {
    const bytes = await downloadVerified(url, archiveFile, checksum);
    const parsed = await parseArchive(archiveFile, csvName, date);
    await putTradeFlowShard(store, {
      namespace: namespaces["spot-flow"],
      key: date,
      seconds: parsed.seconds,
      stepMs: 1_000,
      metadata: researchMetadata("spot-flow", url, checksum, bytes, symbol, {
        sourceCsvRows: parsed.csvRows,
        sourceCsvBytes: parsed.csvBytes,
        sourceTimestampUnit: parsed.timestampUnit,
      }),
    });
  } finally {
    await fs.rm(archiveFile, { force: true });
  }
}

async function ingestFuturesKlines(
  store: SequentialShardStore,
  date: string,
  symbol: string,
  namespaces: Record<Source, string>,
): Promise<void> {
  const archiveName = `${symbol}-1m-${date}.zip`;
  const csvName = `${symbol}-1m-${date}.csv`;
  const url = `https://data.binance.vision/data/futures/um/daily/klines/${symbol}/1m/${archiveName}`;
  const checksum = await fetchChecksum(url);
  const archive = await requestBuffer(url, 8_000_000);
  verifyChecksum(url, archive, checksum);
  const parsed = parseKlinesArchive(archive, csvName, date);
  await putDerivativesKlinesShard(store, {
    namespace: namespaces["futures-klines"],
    key: date,
    rows: parsed.rows,
    metadata: researchMetadata("futures-klines", url, checksum, archive.byteLength, symbol, {
      sourceCsvRows: parsed.sourceCsvRows,
      observedGridRows: parsed.observedGridRows,
      missingGridRows: parsed.missingGridRows,
      closeAvailability: "openTime+59999ms",
      closeTimeOffsetMs: DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS,
    }),
  });
}

async function ingestFuturesMetrics(
  store: SequentialShardStore,
  date: string,
  symbol: string,
  namespaces: Record<Source, string>,
): Promise<void> {
  const archiveName = `${symbol}-metrics-${date}.zip`;
  const csvName = `${symbol}-metrics-${date}.csv`;
  const url = `https://data.binance.vision/data/futures/um/daily/metrics/${symbol}/${archiveName}`;
  const checksum = await fetchChecksum(url);
  const archive = await requestBuffer(url, 2_000_000);
  verifyChecksum(url, archive, checksum);
  const parsed = parseMetricsArchive(archive, csvName, date, symbol);
  await putDerivativesMetricsShard(store, {
    namespace: namespaces["futures-metrics"],
    key: date,
    rows: parsed.rows,
    stepMs: 300_000,
    metadata: researchMetadata("futures-metrics", url, checksum, archive.byteLength, symbol, {
      sourceCsvRows: parsed.sourceCsvRows,
      observedGridRows: parsed.observedGridRows,
      missingGridRows: parsed.missingGridRows,
      availabilityLagMs: 300_000,
    }),
  });
}

function researchMetadata(
  source: Source,
  url: string,
  checksum: string,
  bytes: number,
  symbol: string,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  return {
    featureSchema: `research-${source}-v1`,
    source: "data.binance.vision",
    sourceArchiveUrl: url,
    sourceArchiveSha256: checksum,
    sourceArchiveBytes: bytes,
    researchOnly: true,
    sealedOracleCorpus: false,
    symbol,
    ...extra,
  };
}

function namespacesFor(symbol: string): Record<Source, string> {
  const normalized = symbol.toLowerCase();
  return {
    "spot-flow": `research/trade-flow/spot-${normalized}/${normalized}/1s`,
    "futures-klines": `research/derivatives-klines/usdm-futures/${normalized}/1m`,
    "futures-metrics": `research/derivatives-metrics/usdm-futures/${normalized}/5m`,
  };
}

async function cached(
  store: SequentialShardStore,
  namespace: string,
  date: string,
  source: Source,
): Promise<boolean> {
  try {
    const reference = await store.readReference(namespace, date);
    if (reference.metadata?.researchOnly !== true
      || reference.metadata?.featureSchema !== `research-${source}-v1`
      || typeof reference.metadata?.sourceArchiveSha256 !== "string"
      || !/^[a-f0-9]{64}$/.test(reference.metadata.sourceArchiveSha256)) return false;
    await store.readPayload(reference);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function expandRanges(raw: string): string[] {
  const dates: string[] = [];
  for (const range of raw.split(",")) {
    const [start, end, extra] = range.split("..");
    if (!start || !end || extra !== undefined) throw new Error(`Invalid range: ${range}`);
    const startMs = parseDay(start);
    const endMs = parseDay(end);
    if (endMs < startMs) throw new Error(`Range ends before it starts: ${range}`);
    for (let time = startMs; time <= endMs; time += DAY_MS) {
      dates.push(new Date(time).toISOString().slice(0, 10));
    }
  }
  return [...new Set(dates)].sort();
}

function parseDays(days: string[]): string[] {
  return [...new Set(days.map((day) => {
    parseDay(day);
    return day;
  }))].sort();
}

function parseDay(day: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`Invalid UTC day: ${day}`);
  const time = Date.parse(`${day}T00:00:00.000Z`);
  if (!Number.isFinite(time) || new Date(time).toISOString().slice(0, 10) !== day) {
    throw new Error(`Invalid UTC day: ${day}`);
  }
  return time;
}

async function fetchChecksum(url: string): Promise<string> {
  const text = (await requestBuffer(`${url}.CHECKSUM`, 20_000)).toString("utf8");
  const checksum = /^([a-fA-F0-9]{64})(?:\s|$)/.exec(text.trim())?.[1]?.toLowerCase();
  if (!checksum) throw new Error(`${path.basename(url)}: invalid checksum response.`);
  return checksum;
}

async function downloadVerified(url: string, file: string, checksum: string): Promise<number> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${url}: HTTP ${response.status}`);
  const digest = createHash("sha256");
  let bytes = 0;
  const hasher = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      digest.update(chunk);
      bytes += chunk.byteLength;
      callback(null, chunk);
    },
  });
  await pipeline(Readable.fromWeb(response.body as any), hasher, createWriteStream(file));
  const actual = digest.digest("hex");
  if (actual !== checksum) {
    await fs.rm(file, { force: true });
    throw new Error(`${path.basename(url)}: checksum mismatch.`);
  }
  return bytes;
}

async function requestBuffer(url: string, maximumBytes: number): Promise<Buffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const length = Number(response.headers.get("content-length") ?? 0);
  if (length > maximumBytes) throw new Error(`${url}: response exceeds ${maximumBytes} bytes.`);
  const buffer = Buffer.from(await response.arrayBuffer());
  if (buffer.byteLength > maximumBytes) throw new Error(`${url}: response exceeds ${maximumBytes} bytes.`);
  return buffer;
}

function verifyChecksum(url: string, buffer: Buffer, expected: string): void {
  const actual = createHash("sha256").update(buffer).digest("hex");
  if (actual !== expected) throw new Error(`${path.basename(url)}: checksum mismatch.`);
}

const invokedFile = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invokedFile === path.resolve(import.meta.filename)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
