import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import AdmZip from "adm-zip";
import {
  decodeDerivativesKlines,
  DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS,
  DERIVATIVES_KLINE_ENCODING,
  DERIVATIVES_KLINE_STEP_MS,
  putDerivativesKlinesShard,
  SequentialShardStore,
  TradingStorageLayout,
  type SequentialDerivativesKlineRow,
} from "@trading/storage";
import {
  oracleScopedTradeFlowDates,
  parseDateList,
  readOracleCorpusSplitContract,
  targetContractId,
} from "./lib/binance-agg-trades.js";

const SYMBOL = "BTCUSDT";
const INTERVAL = "1m";
const DAY_MS = 86_400_000;
const DAY_ROWS = 1_440;
const EXPECTED_FULL_SCOPE_DAYS = 420;
const FEATURE_SCHEMA = "binance-usdm-futures-klines-v1";
const NAMESPACE = "derivatives-klines/usdm-futures/btcusdt/1m";
const SOURCE_ROOT = "https://data.binance.vision/data/futures/um/daily/klines";
const HEADER = [
  "open_time",
  "open",
  "high",
  "low",
  "close",
  "volume",
  "close_time",
  "quote_volume",
  "count",
  "taker_buy_volume",
  "taker_buy_quote_volume",
  "ignore",
].join(",");
const DEFAULT_TARGET_REFS = path.join(
  "data", "training", "immutable", "refs", "oracle", "1s",
  "hindsight-bot-71391c44b323e044e6ab",
);
const ORACLE_SPLIT_CONTRACT = path.join(
  "ml", "corpus-contracts", "oracle-hindsight-bot-71391c44-split-v1.json",
);

export interface ParsedKlinesArchive {
  rows: SequentialDerivativesKlineRow[];
  csvBytes: number;
  sourceCsvRows: number;
  headerRows: 0 | 1;
  observedGridRows: number;
  liveGridRows: number;
  noTradeGridRows: number;
  missingGridRows: number;
  outsideUtcDayRows: number;
  offGridRows: number;
  timestampAdjustedRows: 0;
  sourceTimestampUnit: "millisecond" | "microsecond";
}

export function officialKlineSource(day: string): {
  archiveName: string;
  csvName: string;
  url: string;
  checksumUrl: string;
} {
  parseUtcDay(day);
  const archiveName = `${SYMBOL}-${INTERVAL}-${day}.zip`;
  const csvName = `${SYMBOL}-${INTERVAL}-${day}.csv`;
  const url = `${SOURCE_ROOT}/${SYMBOL}/${INTERVAL}/${archiveName}`;
  return { archiveName, csvName, url, checksumUrl: `${url}.CHECKSUM` };
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:futures-klines -- [options]

  --target-ref-dir PATH             Oracle references defining the fixed split
  --only YYYY-MM-DD,...             Fetch a subset of allowed target/context dates
  --data-dir data
  --dry-run                         Print the leakage-safe date scope only
  --quiet                           Print only the final summary

The immutable split contract's test dates are sealed and cannot be requested.
No missing source kline is synthesized; close data is available at open+59,999ms.`);
    return;
  }
  const value = (name: string): string | undefined => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const targetReferenceDirectory = path.resolve(
    value("--target-ref-dir") ?? DEFAULT_TARGET_REFS,
  );
  const requestedDates = value("--only")
    ? parseDateList(value("--only")!)
    : undefined;
  const splitContract = await readOracleCorpusSplitContract(
    path.resolve(ORACLE_SPLIT_CONTRACT),
  );
  const scope = await oracleScopedTradeFlowDates(
    targetReferenceDirectory,
    requestedDates,
    splitContract,
  );
  if (requestedDates === undefined && scope.dates.length !== EXPECTED_FULL_SCOPE_DAYS) {
    throw new Error(
      `Fixed oracle train/validation/predecessor scope changed: expected `
      + `${EXPECTED_FULL_SCOPE_DAYS} days, found ${scope.dates.length}.`,
    );
  }
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

  const layout = new TradingStorageLayout(path.resolve(value("--data-dir") ?? "data"));
  const store = new SequentialShardStore(layout.marketStore);
  const targetContract = targetContractId(targetReferenceDirectory);
  let stored = 0;
  let cached = 0;
  let sourceBytes = 0;
  for (const day of scope.dates) {
    const oracleScope = scope.targetDates.has(day)
      ? "train-or-validation-target"
      : "predecessor-context";
    if (await validCachedReference(
      store,
      day,
      targetContract,
      oracleScope,
      scope.sealedTestStart,
      scope.sealedTestEnd,
    )) {
      cached += 1;
      if (!quiet) console.log(`${day}: cached`);
      continue;
    }
    const referenceFile = store.referenceFile(NAMESPACE, day);
    if (await exists(referenceFile)) {
      throw new Error(`${day}: existing immutable USD-M kline reference is invalid.`);
    }
    const { archiveName, csvName, url, checksumUrl } = officialKlineSource(day);
    const checksum = await fetchChecksum(url);
    const archive = await requestBuffer(url, 5_000_000);
    const actualSha256 = createHash("sha256").update(archive).digest("hex");
    if (actualSha256 !== checksum.sha256) {
      throw new Error(`${archiveName}: checksum mismatch.`);
    }
    const parsed = parseKlinesArchive(archive, csvName, day);
    await putDerivativesKlinesShard(store, {
      namespace: NAMESPACE,
      key: day,
      rows: parsed.rows,
      metadata: {
        featureSchema: FEATURE_SCHEMA,
        source: "data.binance.vision",
        sourceDataset: "futures/um/daily/klines",
        sourceArchiveUrl: url,
        sourceArchiveChecksumUrl: checksumUrl,
        sourceArchiveChecksumAlgorithm: "sha256",
        sourceArchiveChecksumFilename: checksum.filename,
        sourceArchiveSha256: checksum.sha256,
        sourceArchiveBytes: archive.byteLength,
        sourceCsvEntry: csvName,
        sourceCsvBytes: parsed.csvBytes,
        sourceCsvRows: parsed.sourceCsvRows,
        sourceCsvHeaderRows: parsed.headerRows,
        sourceTimestampUnit: parsed.sourceTimestampUnit,
        observedGridRows: parsed.observedGridRows,
        liveGridRows: parsed.liveGridRows,
        noTradeGridRows: parsed.noTradeGridRows,
        missingGridRows: parsed.missingGridRows,
        outsideUtcDayRows: parsed.outsideUtcDayRows,
        offGridRows: parsed.offGridRows,
        timestampAdjustedRows: parsed.timestampAdjustedRows,
        filledGridRows: 0,
        market: "usdm-futures",
        symbol: SYMBOL,
        interval: INTERVAL,
        denseUtcDayAxis: true,
        rowValidity: "official-source-row-present",
        liveObservationRule: "validMask && tradeCount > 0",
        closeAvailability: "openTime+59999ms",
        closeTimeOffsetMs: DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS,
        oracleTargetContract: targetContract,
        oracleScope,
        sealedTestStart: scope.sealedTestStart,
        sealedTestEnd: scope.sealedTestEnd,
      },
    });
    stored += 1;
    sourceBytes += archive.byteLength;
    if (!quiet) {
      console.log(
        `${day}: stored ${parsed.observedGridRows}/${DAY_ROWS} source rows `
        + `(${parsed.noTradeGridRows} no-trade; ${archive.byteLength} bytes)`,
      );
    }
  }
  console.log(
    `USD-M kline ingestion complete: ${stored} stored, ${cached} cached, `
    + `${sourceBytes.toLocaleString()} source bytes downloaded.`,
  );
}

export function parseKlinesArchive(
  archive: Buffer,
  expectedCsvName: string,
  day: string,
): ParsedKlinesArchive {
  const zip = new AdmZip(archive);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1 || entries[0]!.entryName !== expectedCsvName) {
    throw new Error(`${day}: expected one ZIP entry named ${expectedCsvName}.`);
  }
  const payload = entries[0]!.getData();
  const text = payload.toString("utf8").replace(/^\ufeff/, "").trimEnd();
  if (!text) throw new Error(`${day}: USD-M kline CSV has no rows.`);
  const lines = text.split(/\r?\n/);
  const firstColumns = csvColumns(lines[0]!, day, 0);
  let headerRows: 0 | 1 = 0;
  if (!/^\d+$/.test(firstColumns[0]!)) {
    if (firstColumns.join(",") !== HEADER) {
      throw new Error(`${day}: USD-M kline CSV header differs from Binance's schema.`);
    }
    headerRows = 1;
  }
  const sourceLines = lines.slice(headerRows);
  if (sourceLines.length === 0) throw new Error(`${day}: USD-M kline CSV has no data rows.`);

  const dayStart = parseUtcDay(day);
  const rows = Array.from({ length: DAY_ROWS }, (_, index) => missingRow(
    dayStart + index * DERIVATIVES_KLINE_STEP_MS,
  ));
  const seen = new Set<number>();
  const timestampUnits = new Set<"millisecond" | "microsecond">();
  let outsideUtcDayRows = 0;
  let offGridRows = 0;
  let noTradeGridRows = 0;
  sourceLines.forEach((line, sourceIndex) => {
    const columns = csvColumns(line, day, sourceIndex + headerRows);
    const openTime = timestamp(columns[0]!, "open time", day, sourceIndex);
    const closeTime = timestamp(columns[6]!, "close time", day, sourceIndex);
    if (openTime.unit !== closeTime.unit) {
      throw new Error(`${day}: mixed timestamp units in kline row ${sourceIndex}.`);
    }
    timestampUnits.add(openTime.unit);
    if (closeTime.milliseconds
      !== openTime.milliseconds + DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS) {
      throw new Error(
        `${day}: close time is not open+${DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS}ms `
        + `at row ${sourceIndex}.`,
      );
    }
    const parsed: Omit<SequentialDerivativesKlineRow, "openTime"> = {
      open: positive(columns[1]!, "open", day, sourceIndex),
      high: positive(columns[2]!, "high", day, sourceIndex),
      low: positive(columns[3]!, "low", day, sourceIndex),
      close: positive(columns[4]!, "close", day, sourceIndex),
      baseVolume: nonnegative(columns[5]!, "base volume", day, sourceIndex),
      quoteVolume: nonnegative(columns[7]!, "quote volume", day, sourceIndex),
      tradeCount: count(columns[8]!, day, sourceIndex),
      takerBuyBaseVolume: nonnegative(
        columns[9]!, "taker-buy base volume", day, sourceIndex,
      ),
      takerBuyQuoteVolume: nonnegative(
        columns[10]!, "taker-buy quote volume", day, sourceIndex,
      ),
    };
    if (nonnegative(columns[11]!, "ignore", day, sourceIndex) !== 0) {
      throw new Error(`${day}: Binance ignore column is nonzero at row ${sourceIndex}.`);
    }
    validateMarketInvariants(parsed, day, sourceIndex);

    if (openTime.milliseconds < dayStart || openTime.milliseconds >= dayStart + DAY_MS) {
      outsideUtcDayRows += 1;
      return;
    }
    const elapsed = openTime.milliseconds - dayStart;
    if (elapsed % DERIVATIVES_KLINE_STEP_MS !== 0) {
      offGridRows += 1;
      return;
    }
    const index = elapsed / DERIVATIVES_KLINE_STEP_MS;
    if (seen.has(index)) {
      throw new Error(`${day}: duplicate USD-M kline grid row ${index}.`);
    }
    seen.add(index);
    if (parsed.tradeCount === 0) noTradeGridRows += 1;
    rows[index] = { openTime: openTime.milliseconds, ...parsed };
  });
  if (timestampUnits.size !== 1) {
    throw new Error(`${day}: USD-M kline archive mixes timestamp units.`);
  }
  return {
    rows,
    csvBytes: payload.byteLength,
    sourceCsvRows: sourceLines.length,
    headerRows,
    observedGridRows: seen.size,
    liveGridRows: seen.size - noTradeGridRows,
    noTradeGridRows,
    missingGridRows: DAY_ROWS - seen.size,
    outsideUtcDayRows,
    offGridRows,
    timestampAdjustedRows: 0,
    sourceTimestampUnit: [...timestampUnits][0]!,
  };
}

async function validCachedReference(
  store: SequentialShardStore,
  day: string,
  targetContract: string,
  oracleScope: "train-or-validation-target" | "predecessor-context",
  sealedTestStart: string,
  sealedTestEnd: string,
): Promise<boolean> {
  try {
    const reference = await store.readReference(NAMESPACE, day);
    const metadata = reference.metadata ?? {};
    const integerMetadata = [
      "sourceArchiveBytes",
      "sourceCsvBytes",
      "sourceCsvRows",
      "sourceCsvHeaderRows",
      "observedGridRows",
      "liveGridRows",
      "noTradeGridRows",
      "missingGridRows",
      "outsideUtcDayRows",
      "offGridRows",
    ] as const;
    const counts = Object.fromEntries(integerMetadata.map((name) => [
      name,
      Number(metadata[name]),
    ])) as Record<typeof integerMetadata[number], number>;
    const { archiveName, csvName, url, checksumUrl } = officialKlineSource(day);
    if (reference.sequence.start !== parseUtcDay(day)
      || reference.sequence.step !== DERIVATIVES_KLINE_STEP_MS
      || reference.sequence.count !== DAY_ROWS
      || reference.sequence.unit !== "unix-ms"
      || reference.layout.encoding !== DERIVATIVES_KLINE_ENCODING
      || reference.layout.closeTimeOffsetMs !== DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS
      || reference.layout.closed !== true
      || metadata.featureSchema !== FEATURE_SCHEMA
      || metadata.source !== "data.binance.vision"
      || metadata.sourceDataset !== "futures/um/daily/klines"
      || metadata.sourceArchiveUrl !== url
      || metadata.sourceArchiveChecksumUrl !== checksumUrl
      || metadata.sourceArchiveChecksumAlgorithm !== "sha256"
      || metadata.sourceArchiveChecksumFilename !== archiveName
      || typeof metadata.sourceArchiveSha256 !== "string"
      || !/^[a-f0-9]{64}$/.test(metadata.sourceArchiveSha256)
      || metadata.sourceCsvEntry !== csvName
      || integerMetadata.some((name) => !Number.isSafeInteger(counts[name]))
      || counts.sourceArchiveBytes < 1
      || counts.sourceCsvBytes < 1
      || counts.sourceCsvRows < 1
      || (counts.sourceCsvHeaderRows !== 0 && counts.sourceCsvHeaderRows !== 1)
      || counts.observedGridRows < 0
      || counts.liveGridRows < 0
      || counts.noTradeGridRows < 0
      || counts.missingGridRows < 0
      || counts.outsideUtcDayRows < 0
      || counts.offGridRows < 0
      || counts.observedGridRows + counts.missingGridRows !== DAY_ROWS
      || counts.liveGridRows + counts.noTradeGridRows !== counts.observedGridRows
      || counts.observedGridRows + counts.outsideUtcDayRows + counts.offGridRows
        !== counts.sourceCsvRows
      || (metadata.sourceTimestampUnit !== "millisecond"
        && metadata.sourceTimestampUnit !== "microsecond")
      || metadata.timestampAdjustedRows !== 0
      || metadata.filledGridRows !== 0
      || metadata.market !== "usdm-futures"
      || metadata.symbol !== SYMBOL
      || metadata.interval !== INTERVAL
      || metadata.denseUtcDayAxis !== true
      || metadata.rowValidity !== "official-source-row-present"
      || metadata.liveObservationRule !== "validMask && tradeCount > 0"
      || metadata.closeAvailability !== "openTime+59999ms"
      || metadata.closeTimeOffsetMs !== DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS
      || metadata.oracleTargetContract !== targetContract
      || metadata.oracleScope !== oracleScope
      || metadata.sealedTestStart !== sealedTestStart
      || metadata.sealedTestEnd !== sealedTestEnd) {
      return false;
    }
    const decoded = decodeDerivativesKlines(
      reference,
      await store.readPayload(reference),
    );
    const observed = decoded.filter((row) => row.tradeCount !== null);
    const noTrade = observed.filter((row) => row.tradeCount === 0);
    if (observed.length !== counts.observedGridRows
      || noTrade.length !== counts.noTradeGridRows) {
      return false;
    }
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function missingRow(openTime: number): SequentialDerivativesKlineRow {
  return {
    openTime,
    open: null,
    high: null,
    low: null,
    close: null,
    baseVolume: null,
    quoteVolume: null,
    tradeCount: null,
    takerBuyBaseVolume: null,
    takerBuyQuoteVolume: null,
  };
}

function csvColumns(line: string, day: string, row: number): string[] {
  const columns = line.split(",").map((value) => value.trim());
  if (columns.length !== 12) {
    throw new Error(`${day}: expected 12 USD-M kline columns at row ${row}.`);
  }
  return columns;
}

function timestamp(
  raw: string,
  label: string,
  day: string,
  row: number,
): { milliseconds: number; unit: "millisecond" | "microsecond" } {
  if (!/^\d+$/.test(raw)) throw new Error(`${day}: invalid ${label} at row ${row}.`);
  let value = BigInt(raw);
  const unit = value >= 1_000_000_000_000_000n ? "microsecond" : "millisecond";
  if (unit === "microsecond") {
    if (value % 1_000n !== 0n) {
      throw new Error(`${day}: ${label} has sub-millisecond precision at row ${row}.`);
    }
    value /= 1_000n;
  }
  const milliseconds = Number(value);
  if (!Number.isSafeInteger(milliseconds)) {
    throw new Error(`${day}: ${label} exceeds safe integers at row ${row}.`);
  }
  return { milliseconds, unit };
}

function positive(raw: string, label: string, day: string, row: number): number {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${day}: invalid ${label} at row ${row}.`);
  }
  return value;
}

function nonnegative(raw: string, label: string, day: string, row: number): number {
  const value = Number(raw);
  if (raw === "" || !Number.isFinite(value) || value < 0) {
    throw new Error(`${day}: invalid ${label} at row ${row}.`);
  }
  return value;
}

function count(raw: string, day: string, row: number): number {
  if (!/^\d+$/.test(raw)) throw new Error(`${day}: invalid trade count at row ${row}.`);
  const value = Number(BigInt(raw));
  if (!Number.isSafeInteger(value)) {
    throw new Error(`${day}: trade count exceeds safe integers at row ${row}.`);
  }
  return value;
}

function validateMarketInvariants(
  row: Omit<SequentialDerivativesKlineRow, "openTime">,
  day: string,
  sourceIndex: number,
): void {
  const open = row.open!;
  const high = row.high!;
  const low = row.low!;
  const close = row.close!;
  const baseVolume = row.baseVolume!;
  const quoteVolume = row.quoteVolume!;
  const tradeCount = row.tradeCount!;
  const takerBuyBaseVolume = row.takerBuyBaseVolume!;
  const takerBuyQuoteVolume = row.takerBuyQuoteVolume!;
  if (high < Math.max(open, close)
    || low > Math.min(open, close)
    || low > high
    || takerBuyBaseVolume > baseVolume
    || takerBuyQuoteVolume > quoteVolume
    || (tradeCount === 0 && (
      open !== high
      || open !== low
      || open !== close
      ||
      baseVolume !== 0
      || quoteVolume !== 0
      || takerBuyBaseVolume !== 0
      || takerBuyQuoteVolume !== 0
    ))
    || (tradeCount > 0 && (baseVolume === 0 || quoteVolume === 0))) {
    throw new Error(`${day}: inconsistent USD-M kline row ${sourceIndex}.`);
  }
}

function parseUtcDay(day: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`Invalid UTC date: ${day}.`);
  const value = Date.parse(`${day}T00:00:00.000Z`);
  if (!Number.isSafeInteger(value) || new Date(value).toISOString().slice(0, 10) !== day) {
    throw new Error(`Invalid UTC date: ${day}.`);
  }
  return value;
}

async function fetchChecksum(url: string): Promise<{ sha256: string; filename: string }> {
  const text = (await requestBuffer(`${url}.CHECKSUM`, 20_000)).toString("utf8");
  return parseOfficialChecksum(text, url);
}

export function parseOfficialChecksum(
  raw: string,
  archiveUrl: string,
): { sha256: string; filename: string } {
  const text = raw.trim();
  const match = /^([a-fA-F0-9]{64})\s+\*?([^\s]+)$/.exec(text);
  const filename = path.basename(archiveUrl);
  if (!match || match[2] !== filename) {
    throw new Error(`${filename}: invalid checksum response.`);
  }
  return { sha256: match[1]!.toLowerCase(), filename };
}

async function requestBuffer(url: string, maximumBytes: number): Promise<Buffer> {
  let failure: unknown;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}.`);
      const length = Number(response.headers.get("content-length") ?? 0);
      if (length > maximumBytes) throw new Error(`${url}: response is too large.`);
      const result = Buffer.from(await response.arrayBuffer());
      if (result.byteLength > maximumBytes) throw new Error(`${url}: response is too large.`);
      return result;
    } catch (error) {
      failure = error;
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, attempt * 500));
    }
  }
  throw failure;
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
