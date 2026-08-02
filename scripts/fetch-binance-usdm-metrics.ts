import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import AdmZip from "adm-zip";
import {
  DERIVATIVES_METRIC_COLUMNS,
  decodeDerivativesMetrics,
  putDerivativesMetricsShard,
  SequentialShardStore,
  TradingStorageLayout,
  type SequentialDerivativesMetricRow,
} from "@trading/storage";
import {
  oracleScopedTradeFlowDates,
  parseDateList,
  readOracleCorpusSplitContract,
  targetContractId,
} from "./lib/binance-agg-trades.js";

const SYMBOL = "BTCUSDT";
const INTERVAL = "5m";
const STEP_MS = 300_000;
const DAY_ROWS = 288;
const FEATURE_SCHEMA = "binance-usdm-futures-metrics-v1";
const SOURCE_ROOT = "https://data.binance.vision/data/futures/um/daily/metrics";
const HEADER = [
  "create_time",
  "symbol",
  "sum_open_interest",
  "sum_open_interest_value",
  "count_toptrader_long_short_ratio",
  "sum_toptrader_long_short_ratio",
  "count_long_short_ratio",
  "sum_taker_long_short_vol_ratio",
].join(",");
const DEFAULT_TARGET_REFS = path.join(
  "data", "training", "immutable", "refs", "oracle", "1s",
  "hindsight-bot-71391c44b323e044e6ab",
);
const ORACLE_SPLIT_CONTRACT = path.join(
  "ml", "corpus-contracts", "oracle-hindsight-bot-71391c44-split-v1.json",
);

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:futures-metrics -- [options]

  --target-ref-dir PATH             Oracle references defining the fixed split
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
  const namespace = `derivatives-metrics/usdm-futures/${SYMBOL.toLowerCase()}/${INTERVAL}`;
  let stored = 0;
  let cached = 0;
  let sourceBytes = 0;
  for (const day of scope.dates) {
    if (await validCachedReference(
      store,
      namespace,
      day,
      targetContractId(targetReferenceDirectory),
      scope.sealedTestStart,
      scope.sealedTestEnd,
    )) {
      cached += 1;
      if (!quiet) console.log(`${day}: cached`);
      continue;
    }
    const referenceFile = store.referenceFile(namespace, day);
    if (await exists(referenceFile)) {
      throw new Error(`${day}: existing immutable futures-metrics reference is invalid.`);
    }
    const archiveName = `${SYMBOL}-metrics-${day}.zip`;
    const csvName = `${SYMBOL}-metrics-${day}.csv`;
    const url = `${SOURCE_ROOT}/${SYMBOL}/${archiveName}`;
    const expectedSha256 = await fetchChecksum(url);
    const archive = await requestBuffer(url, 1_000_000);
    const actualSha256 = createHash("sha256").update(archive).digest("hex");
    if (actualSha256 !== expectedSha256) {
      throw new Error(`${archiveName}: checksum mismatch.`);
    }
    const parsed = parseMetricsArchive(archive, csvName, day);
    await putDerivativesMetricsShard(store, {
      namespace,
      key: day,
      rows: parsed.rows,
      stepMs: STEP_MS,
      metadata: {
        featureSchema: FEATURE_SCHEMA,
        source: "data.binance.vision",
        sourceDataset: "futures/um/daily/metrics",
        sourceArchiveUrl: url,
        sourceArchiveSha256: expectedSha256,
        sourceArchiveBytes: archive.byteLength,
        sourceCsvEntry: csvName,
        sourceCsvBytes: parsed.csvBytes,
        sourceCsvRows: parsed.sourceCsvRows,
        observedGridRows: parsed.observedGridRows,
        missingGridRows: parsed.missingGridRows,
        outsideUtcDayRows: parsed.outsideUtcDayRows,
        offGridRows: parsed.offGridRows,
        timestampAdjustedRows: parsed.timestampAdjustedRows,
        missingValueCounts: parsed.missingValueCounts,
        market: "usdm-futures",
        symbol: SYMBOL,
        interval: INTERVAL,
        denseUtcDayAxis: true,
        availabilityLagMs: STEP_MS,
        oracleTargetContract: targetContractId(targetReferenceDirectory),
        oracleScope: scope.targetDates.has(day)
          ? "train-or-validation-target"
          : "predecessor-context",
        sealedTestStart: scope.sealedTestStart,
        sealedTestEnd: scope.sealedTestEnd,
      },
    });
    stored += 1;
    sourceBytes += archive.byteLength;
    if (!quiet) {
      console.log(`${day}: stored ${parsed.rows.length} rows (${archive.byteLength} bytes)`);
    }
  }
  console.log(
    `Futures-metrics ingestion complete: ${stored} stored, ${cached} cached, `
    + `${sourceBytes.toLocaleString()} source bytes downloaded.`,
  );
}

export function parseMetricsArchive(
  archive: Buffer,
  expectedCsvName: string,
  day: string,
): {
  rows: SequentialDerivativesMetricRow[];
  csvBytes: number;
  sourceCsvRows: number;
  observedGridRows: number;
  missingGridRows: number;
  outsideUtcDayRows: number;
  offGridRows: number;
  timestampAdjustedRows: number;
  missingValueCounts: Record<string, number>;
} {
  const zip = new AdmZip(archive);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1 || entries[0]!.entryName !== expectedCsvName) {
    throw new Error(`${day}: expected one ZIP entry named ${expectedCsvName}.`);
  }
  const payload = entries[0]!.getData();
  const lines = payload.toString("utf8").replace(/^\ufeff/, "").trimEnd().split(/\r?\n/);
  if (lines[0]?.trim().toLowerCase() !== HEADER || lines.length < 2) {
    throw new Error(`${day}: futures-metrics CSV schema differs or has no rows.`);
  }
  const dayStart = Date.parse(`${day}T00:00:00.000Z`);
  const empty = (index: number): SequentialDerivativesMetricRow => ({
    openTime: dayStart + index * STEP_MS,
    sumOpenInterest: null,
    sumOpenInterestValue: null,
    topTraderAccountLongShortRatio: null,
    topTraderPositionLongShortRatio: null,
    globalLongShortRatio: null,
    takerBuySellVolumeRatio: null,
  });
  const rows = Array.from({ length: DAY_ROWS }, (_, index) => empty(index));
  const seen = new Set<number>();
  let outsideUtcDayRows = 0;
  let offGridRows = 0;
  let timestampAdjustedRows = 0;
  const missingValueCounts = Object.fromEntries(
    DERIVATIVES_METRIC_COLUMNS.map((name) => [name, 0]),
  ) as Record<string, number>;
  lines.slice(1).forEach((line, sourceIndex) => {
    const columns = line.split(",").map((item) => item.trim());
    if (columns.length !== 8 || columns[1] !== SYMBOL) {
      throw new Error(`${day}: malformed futures-metrics row ${sourceIndex}.`);
    }
    const openTime = Date.parse(`${columns[0]!.replace(" ", "T")}Z`);
    if (!Number.isFinite(openTime)) {
      throw new Error(`${day}: invalid futures-metrics timestamp at row ${sourceIndex}.`);
    }
    if (openTime < dayStart || openTime >= dayStart + 86_400_000) {
      outsideUtcDayRows += 1;
      return;
    }
    const elapsed = openTime - dayStart;
    const offset = elapsed % STEP_MS;
    if (offset !== 0) {
      // A few official archives contain isolated wall-clock timestamps rather
      // than a trustworthy five-minute bucket. Guessing a bucket would turn
      // corrupted source timing into false precision. Preserve the dense axis
      // as missing and let the causal feature layer carry the last known value
      // together with its age/missingness signal.
      offGridRows += 1;
      return;
    }
    const index = Math.floor(elapsed / STEP_MS);
    if (seen.has(index)) {
      throw new Error(`${day}: duplicate futures-metrics grid row ${index}.`);
    }
    seen.add(index);
    const values = columns.slice(2).map((item, valueIndex) => {
      const normalized = item.replace(/^"|"$/g, "");
      if (normalized === "") {
        missingValueCounts[DERIVATIVES_METRIC_COLUMNS[valueIndex]!] += 1;
        return null;
      }
      const number = Number(normalized);
      if (!Number.isFinite(number) || number < 0) {
        throw new Error(`${day}: invalid futures-metrics value at row ${sourceIndex}.`);
      }
      if (number === 0) {
        missingValueCounts[DERIVATIVES_METRIC_COLUMNS[valueIndex]!] += 1;
        return null;
      }
      return number;
    });
    rows[index] = {
      openTime: dayStart + index * STEP_MS,
      sumOpenInterest: values[0]!,
      sumOpenInterestValue: values[1]!,
      topTraderAccountLongShortRatio: values[2]!,
      topTraderPositionLongShortRatio: values[3]!,
      globalLongShortRatio: values[4]!,
      takerBuySellVolumeRatio: values[5]!,
    };
  });
  for (const name of DERIVATIVES_METRIC_COLUMNS) {
    missingValueCounts[name] += DAY_ROWS - seen.size;
  }
  return {
    rows,
    csvBytes: payload.byteLength,
    sourceCsvRows: lines.length - 1,
    observedGridRows: seen.size,
    missingGridRows: DAY_ROWS - seen.size,
    outsideUtcDayRows,
    offGridRows,
    timestampAdjustedRows,
    missingValueCounts,
  };
}

async function validCachedReference(
  store: SequentialShardStore,
  namespace: string,
  day: string,
  targetContract: string,
  sealedTestStart: string,
  sealedTestEnd: string,
): Promise<boolean> {
  try {
    const reference = await store.readReference(namespace, day);
    const metadata = reference.metadata ?? {};
    const offGridRows = metadata.offGridRows === undefined
      ? 0
      : Number(metadata.offGridRows);
    const observedGridRows = Number(metadata.observedGridRows);
    const missingGridRows = Number(metadata.missingGridRows);
    const outsideUtcDayRows = Number(metadata.outsideUtcDayRows);
    const sourceCsvRows = Number(metadata.sourceCsvRows);
    const expectedArchive = `${SYMBOL}-metrics-${day}.zip`;
    const expectedCsv = `${SYMBOL}-metrics-${day}.csv`;
    if (reference.sequence.start !== Date.parse(`${day}T00:00:00.000Z`)
      || reference.sequence.step !== STEP_MS
      || reference.sequence.count !== DAY_ROWS
      || reference.layout.encoding !== "derivatives-metrics-columnar-v1"
      || metadata.featureSchema !== FEATURE_SCHEMA
      || metadata.source !== "data.binance.vision"
      || metadata.sourceDataset !== "futures/um/daily/metrics"
      || metadata.sourceArchiveUrl
        !== `${SOURCE_ROOT}/${SYMBOL}/${expectedArchive}`
      || typeof metadata.sourceArchiveSha256 !== "string"
      || !/^[a-f0-9]{64}$/.test(metadata.sourceArchiveSha256)
      || metadata.sourceCsvEntry !== expectedCsv
      || !Number.isSafeInteger(sourceCsvRows)
      || !Number.isSafeInteger(observedGridRows)
      || !Number.isSafeInteger(missingGridRows)
      || !Number.isSafeInteger(outsideUtcDayRows)
      || !Number.isSafeInteger(offGridRows)
      || observedGridRows < 0
      || missingGridRows < 0
      || outsideUtcDayRows < 0
      || offGridRows < 0
      || observedGridRows + missingGridRows !== DAY_ROWS
      || observedGridRows + outsideUtcDayRows + offGridRows !== sourceCsvRows
      || metadata.timestampAdjustedRows !== 0
      || metadata.market !== "usdm-futures"
      || metadata.symbol !== SYMBOL
      || metadata.interval !== INTERVAL
      || metadata.availabilityLagMs !== STEP_MS
      || metadata.oracleTargetContract !== targetContract
      || metadata.sealedTestStart !== sealedTestStart
      || metadata.sealedTestEnd !== sealedTestEnd
      || metadata.denseUtcDayAxis !== true) {
      return false;
    }
    decodeDerivativesMetrics(reference, await store.readPayload(reference));
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function fetchChecksum(url: string): Promise<string> {
  const text = (await requestBuffer(`${url}.CHECKSUM`, 20_000)).toString("utf8");
  const checksum = /^([a-fA-F0-9]{64})(?:\s|$)/.exec(text.trim())?.[1]?.toLowerCase();
  if (!checksum) throw new Error(`${path.basename(url)}: invalid checksum response.`);
  return checksum;
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
