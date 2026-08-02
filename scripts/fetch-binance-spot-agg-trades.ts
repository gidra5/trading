import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import type { IncomingMessage } from "node:http";
import https from "node:https";
import path from "node:path";
import readline from "node:readline";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import {
  DailyTradeFlowAccumulator,
  putTradeFlowShard,
  SequentialShardStore,
  TradingStorageLayout,
} from "@trading/storage";
import {
  oracleScopedTradeFlowDates,
  parseBinanceAggregateTradeCsvRow,
  parseDateList,
  readOracleCorpusSplitContract,
  targetContractId,
} from "./lib/binance-agg-trades.js";
import { openSingleZipEntry } from "./lib/single-entry-zip.js";

const DAY_MS = 86_400_000;
const SYMBOL = "BTCUSDT";
const INTERVAL = "1s";
const FEATURE_SCHEMA = "binance-spot-agg-trade-flow-v1";
const SOURCE_ROOT = "https://data.binance.vision/data/spot/daily/aggTrades";
const DEFAULT_TARGET_REFS = path.join(
  "data",
  "training",
  "immutable",
  "refs",
  "oracle",
  "1s",
  "hindsight-bot-71391c44b323e044e6ab",
);
const ORACLE_SPLIT_CONTRACT = path.join(
  "ml",
  "corpus-contracts",
  "oracle-hindsight-bot-71391c44-split-v1.json",
);

interface ParsedArchive {
  seconds: ReturnType<DailyTradeFlowAccumulator["finish"]>;
  csvEntry: string;
  csvBytes: number;
  csvRows: number;
  invalidSentinelRows: number;
  falseBestPriceMatchRows: number;
  timestampUnit: "millisecond" | "microsecond";
  firstAggregateTradeId: string;
  lastAggregateTradeId: string;
  aggregateIdGapCount: string;
  firstTradeId: string;
  lastTradeId: string;
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:agg-trades -- [options]

  --target-ref-dir PATH             Oracle target references that define the sealed split
  --only YYYY-MM-DD,...             Fetch a subset of allowed target/context dates
  --data-dir data
  --dry-run                         Print the leakage-safe date scope only
  --quiet                           Print only the final summary

The immutable split contract's test dates are sealed and cannot be requested.`);
    return;
  }
  const value = (name: string): string | undefined => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const targetReferenceDirectory = path.resolve(value("--target-ref-dir") ?? DEFAULT_TARGET_REFS);
  const requestedDates = value("--only") ? parseDateList(value("--only")!) : undefined;
  const splitContract = await readOracleCorpusSplitContract(
    path.resolve(ORACLE_SPLIT_CONTRACT),
  );
  const scope = await oracleScopedTradeFlowDates(
    targetReferenceDirectory,
    requestedDates,
    splitContract,
  );
  const quiet = args.includes("--quiet");
  if (!quiet) {
    console.log(
      `Scope: ${scope.dates.length} allowed days; sealed test `
      + `${scope.sealedTestStart}..${scope.sealedTestEnd} excluded`,
    );
  }
  if (args.includes("--dry-run")) {
    console.log(scope.dates.join(","));
    return;
  }

  const dataDir = path.resolve(value("--data-dir") ?? "data");
  const layout = new TradingStorageLayout(dataDir);
  const store = new SequentialShardStore(layout.marketStore);
  const namespace = `trade-flow/spot-${SYMBOL.toLowerCase()}/${SYMBOL.toLowerCase()}/${INTERVAL}`;
  const temporaryDir = path.join(layout.marketTmp, "agg-trades", SYMBOL.toLowerCase());
  await fs.mkdir(temporaryDir, { recursive: true });
  let stored = 0;
  let cached = 0;
  let aggregateRows = 0;
  for (const date of scope.dates) {
    const referenceFile = store.referenceFile(namespace, date);
    if (await validCachedReference(store, namespace, date, scope.sealedTestStart)) {
      cached += 1;
      if (!quiet) console.log(`${date}: cached`);
      continue;
    }
    if (await exists(referenceFile)) {
      throw new Error(`${date}: existing immutable trade-flow reference is invalid.`);
    }
    const archiveName = `${SYMBOL}-aggTrades-${date}.zip`;
    const csvName = `${SYMBOL}-aggTrades-${date}.csv`;
    const url = `${SOURCE_ROOT}/${SYMBOL}/${archiveName}`;
    const archiveFile = path.join(temporaryDir, `.${archiveName}`);
    const partialFile = `${archiveFile}.partial`;
    try {
      const expectedSha256 = await fetchChecksum(url);
      const archive = await downloadVerified(url, archiveFile, partialFile, expectedSha256);
      const parsed = await parseArchive(archiveFile, csvName, date);
      await putTradeFlowShard(store, {
        namespace,
        key: date,
        seconds: parsed.seconds,
        stepMs: 1_000,
        metadata: {
          featureSchema: FEATURE_SCHEMA,
          source: "data.binance.vision",
          sourceDataset: "spot/daily/aggTrades",
          sourceArchiveUrl: url,
          sourceArchiveSha256: expectedSha256,
          sourceArchiveBytes: archive.bytes,
          sourceCsvEntry: parsed.csvEntry,
          sourceCsvBytes: parsed.csvBytes,
          sourceCsvRows: parsed.csvRows,
          invalidSentinelRows: parsed.invalidSentinelRows,
          falseBestPriceMatchRows: parsed.falseBestPriceMatchRows,
          sourceTimestampUnit: parsed.timestampUnit,
          firstAggregateTradeId: parsed.firstAggregateTradeId,
          lastAggregateTradeId: parsed.lastAggregateTradeId,
          aggregateIdGapCount: parsed.aggregateIdGapCount,
          firstTradeId: parsed.firstTradeId,
          lastTradeId: parsed.lastTradeId,
          market: "spot",
          symbol: SYMBOL,
          interval: INTERVAL,
          completeUtcDay: true,
          oracleTargetContract: targetContractId(targetReferenceDirectory),
          oracleScope: scope.targetDates.has(date)
            ? "train-or-validation-target"
            : "predecessor-context",
          sealedTestStart: scope.sealedTestStart,
          sealedTestEnd: scope.sealedTestEnd,
        },
      });
      stored += 1;
      aggregateRows += parsed.csvRows - parsed.invalidSentinelRows;
      if (!quiet) {
        console.log(
          `${date}: stored ${parsed.csvRows.toLocaleString()} rows `
          + `(${archive.bytes.toLocaleString()} archive bytes)`,
        );
      }
    } finally {
      await fs.rm(archiveFile, { force: true });
      await fs.rm(partialFile, { force: true });
    }
  }
  console.log(
    `Trade-flow ingestion complete: ${stored} stored, ${cached} cached, `
    + `${aggregateRows.toLocaleString()} aggregate rows parsed.`,
  );
}

export async function parseArchive(
  archiveFile: string,
  expectedCsvName: string,
  date: string,
): Promise<ParsedArchive> {
  const dayStart = Date.parse(`${date}T00:00:00.000Z`);
  const accumulator = new DailyTradeFlowAccumulator(dayStart);
  const entry = await openSingleZipEntry(archiveFile, expectedCsvName);
  const lines = readline.createInterface({ input: entry.stream, crlfDelay: Infinity });
  let csvRows = 0;
  let invalidSentinelRows = 0;
  let falseBestPriceMatchRows = 0;
  let headerSeen = false;
  let timestampUnit: ParsedArchive["timestampUnit"] | undefined;
  let previousAggregateId: bigint | undefined;
  let aggregateIdGapCount = 0n;
  let firstAggregateTradeId: bigint | undefined;
  let lastAggregateTradeId: bigint | undefined;
  let firstTradeId: bigint | undefined;
  let lastTradeId: bigint | undefined;
  let previousLastTradeId: bigint | undefined;
  for await (const line of lines) {
    if (!line) continue;
    const parsed = parseBinanceAggregateTradeCsvRow(line);
    if (parsed.kind === "header") {
      if (headerSeen || csvRows > 0) throw new Error(`${date}: misplaced or duplicate CSV header.`);
      headerSeen = true;
      continue;
    }
    csvRows += 1;
    if (parsed.kind === "invalid-sentinel") {
      invalidSentinelRows += 1;
      continue;
    }
    const trade = parsed.value;
    if (timestampUnit && timestampUnit !== trade.timestampUnit) {
      throw new Error(`${date}: mixed timestamp units in aggTrade archive.`);
    }
    timestampUnit = trade.timestampUnit;
    if (previousAggregateId !== undefined) {
      if (trade.aggregateTradeId <= previousAggregateId) {
        throw new Error(`${date}: aggregate trade IDs are not strictly increasing.`);
      }
      const gap = trade.aggregateTradeId - previousAggregateId - 1n;
      if (gap !== 0n) throw new Error(`${date}: aggregate trade ID gap detected.`);
      aggregateIdGapCount += gap;
    }
    if (previousLastTradeId !== undefined
      && trade.firstTradeId !== previousLastTradeId + 1n) {
      throw new Error(`${date}: constituent raw-trade IDs have a gap or overlap.`);
    }
    previousAggregateId = trade.aggregateTradeId;
    firstAggregateTradeId ??= trade.aggregateTradeId;
    lastAggregateTradeId = trade.aggregateTradeId;
    firstTradeId ??= trade.firstTradeId;
    lastTradeId = trade.lastTradeId;
    previousLastTradeId = trade.lastTradeId;
    if (!trade.bestPriceMatch) falseBestPriceMatchRows += 1;
    accumulator.append(trade.aggregate);
  }
  const csvBytes = await entry.completed;
  if (csvRows === 0 || !timestampUnit
    || firstAggregateTradeId === undefined || lastAggregateTradeId === undefined
    || firstTradeId === undefined || lastTradeId === undefined) {
    throw new Error(`${date}: aggTrade archive contains no valid trades.`);
  }
  return {
    seconds: accumulator.finish(),
    csvEntry: entry.name,
    csvBytes,
    csvRows,
    invalidSentinelRows,
    falseBestPriceMatchRows,
    timestampUnit,
    firstAggregateTradeId: firstAggregateTradeId.toString(),
    lastAggregateTradeId: lastAggregateTradeId.toString(),
    aggregateIdGapCount: aggregateIdGapCount.toString(),
    firstTradeId: firstTradeId.toString(),
    lastTradeId: lastTradeId.toString(),
  };
}

async function validCachedReference(
  store: SequentialShardStore,
  namespace: string,
  date: string,
  sealedTestStart: string,
): Promise<boolean> {
  try {
    const reference = await store.readReference(namespace, date);
    const expectedStart = Date.parse(`${date}T00:00:00.000Z`);
    if (reference.sequence.start !== expectedStart
      || reference.sequence.step !== 1_000
      || reference.sequence.count !== DAY_MS / 1_000
      || reference.layout.encoding !== "trade-flow-columnar-v1"
      || reference.metadata?.featureSchema !== FEATURE_SCHEMA
      || reference.metadata?.sealedTestStart !== sealedTestStart) {
      return false;
    }
    await store.readPayload(reference);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function fetchChecksum(url: string): Promise<string> {
  const text = await requestText(`${url}.CHECKSUM`, 20_000);
  const checksum = /^([a-fA-F0-9]{64})(?:\s|$)/.exec(text.trim())?.[1]?.toLowerCase();
  if (!checksum) throw new Error(`${path.basename(url)}: invalid checksum response.`);
  return checksum;
}

async function downloadVerified(
  url: string,
  archiveFile: string,
  partialFile: string,
  expectedSha256: string,
): Promise<{ bytes: number }> {
  if (await exists(archiveFile)) {
    const existing = await hashFile(archiveFile);
    if (existing.sha256 === expectedSha256) return { bytes: existing.bytes };
    await fs.rm(archiveFile);
  }
  let failure: unknown;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    await fs.rm(partialFile, { force: true });
    try {
      const response = await responseFor(url, 60_000);
      const digest = createHash("sha256");
      let bytes = 0;
      const hasher = new Transform({
        transform(chunk: Buffer, _encoding, callback) {
          digest.update(chunk);
          bytes += chunk.byteLength;
          callback(null, chunk);
        },
      });
      await pipeline(response, hasher, createWriteStream(partialFile, { flags: "wx" }));
      const actual = digest.digest("hex");
      if (actual !== expectedSha256) throw new Error(`${path.basename(url)}: checksum mismatch.`);
      await fs.rename(partialFile, archiveFile);
      return { bytes };
    } catch (error) {
      failure = error;
      await fs.rm(partialFile, { force: true });
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
    }
  }
  throw failure;
}

async function hashFile(file: string): Promise<{ sha256: string; bytes: number }> {
  const digest = createHash("sha256");
  let bytes = 0;
  for await (const chunk of createReadStream(file)) {
    digest.update(chunk as Buffer);
    bytes += (chunk as Buffer).byteLength;
  }
  return { sha256: digest.digest("hex"), bytes };
}

async function requestText(url: string, maximumBytes: number): Promise<string> {
  const response = await responseFor(url, 20_000);
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of response) {
    bytes += (chunk as Buffer).byteLength;
    if (bytes > maximumBytes) throw new Error(`${url}: response is too large.`);
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function responseFor(url: string, timeoutMs: number, redirects = 5): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    const request = https.get(
      url,
      { headers: { "user-agent": "trading-agg-trade-fetcher/1.0" } },
      (response) => {
        const status = response.statusCode ?? 0;
        if (status >= 300 && status < 400 && response.headers.location) {
          response.resume();
          if (redirects <= 0) {
            reject(new Error(`${url}: too many redirects.`));
            return;
          }
          responseFor(
            new URL(response.headers.location, url).toString(),
            timeoutMs,
            redirects - 1,
          ).then(resolve, reject);
          return;
        }
        if (status < 200 || status >= 300) {
          response.resume();
          reject(new Error(`${url}: HTTP ${status}.`));
          return;
        }
        resolve(response);
      },
    );
    request.setTimeout(timeoutMs, () => request.destroy(new Error(`${url}: timed out.`)));
    request.once("error", reject);
  });
}

async function exists(file: string): Promise<boolean> {
  try {
    return (await fs.stat(file)).isFile();
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

const invokedFile = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invokedFile === path.resolve(import.meta.filename)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
