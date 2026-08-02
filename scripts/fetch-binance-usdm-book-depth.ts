import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import AdmZip from "adm-zip";
import {
  decodeDerivativesBookDepth,
  DERIVATIVES_BOOK_DEPTH_BANDS,
  DERIVATIVES_BOOK_DEPTH_ENCODING,
  DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS,
  putDerivativesBookDepthShard,
  SequentialShardStore,
  TradingStorageLayout,
  type DerivativesBookDepthAvailabilityVector,
  type DerivativesBookDepthValueVector,
  type SequentialDerivativesBookDepthSnapshot,
  type StorageMetadata,
} from "@trading/storage";
import {
  oracleScopedTradeFlowDates,
  parseDateList,
  readOracleCorpusSplitContract,
  targetContractId,
  type OracleCorpusSplitContract,
} from "./lib/binance-agg-trades.js";

export const BOOK_DEPTH_SYMBOL = "BTCUSDT";
export const BOOK_DEPTH_NAMESPACE = "derivatives-book-depth/usdm-futures/btcusdt";
export const BOOK_DEPTH_FEATURE_SCHEMA = "binance-usdm-futures-book-depth-v1";
export const BOOK_DEPTH_EXPECTED_ALLOWLIST_DAYS = 420;
export const BOOK_DEPTH_EXPECTED_AVAILABLE_DAYS = 368;
export const BOOK_DEPTH_EXPECTED_UNAVAILABLE_DAYS = 52;
export const BOOK_DEPTH_SEALED_TEST_START = "2026-06-24";
export const BOOK_DEPTH_SEALED_TEST_END = "2026-07-23";
export const BOOK_DEPTH_SOURCE_ROOT =
  "https://data.binance.vision/data/futures/um/daily/bookDepth";

const HEADER = "timestamp,percentage,depth,notional";
const DAY_MS = 86_400_000;
const MAXIMUM_ARCHIVE_BYTES = 10_000_000;
const DEFAULT_TARGET_REFS = path.join(
  "data", "training", "immutable", "refs", "oracle", "1s",
  "hindsight-bot-71391c44b323e044e6ab",
);
const ORACLE_SPLIT_CONTRACT = path.join(
  "ml", "corpus-contracts", "oracle-hindsight-bot-71391c44-split-v1.json",
);
const ALLOWED_PERCENTAGES = new Set([-5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5]);
const TEN_BAND_PERCENTAGES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5] as const;
const TWELVE_BAND_PERCENTAGES = [
  -5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5,
] as const;

const OFFICIAL_UNAVAILABLE_RANGES = [
  ["2021-09-07", "2021-09-14"],
  ["2021-10-18", "2021-10-21"],
  ["2021-12-13", "2021-12-20"],
  ["2022-05-13", "2022-05-20"],
  ["2022-06-06", "2022-06-21"],
  ["2022-07-27", "2022-08-03"],
] as const;

export const BOOK_DEPTH_OFFICIAL_UNAVAILABLE_DATES = Object.freeze(
  OFFICIAL_UNAVAILABLE_RANGES.flatMap(([first, last]) => utcDateRange(first, last)),
);

if (BOOK_DEPTH_OFFICIAL_UNAVAILABLE_DATES.length
  !== BOOK_DEPTH_EXPECTED_UNAVAILABLE_DAYS) {
  throw new Error("USD-M book-depth unavailable-date contract has changed.");
}

const OFFICIAL_UNAVAILABLE_SET = new Set(BOOK_DEPTH_OFFICIAL_UNAVAILABLE_DATES);
const OFFICIAL_UNAVAILABLE_DATES_SHA256 = createHash("sha256")
  .update(`${BOOK_DEPTH_OFFICIAL_UNAVAILABLE_DATES.join("\n")}\n`)
  .digest("hex");

export interface ParsedBookDepthArchive {
  snapshots: SequentialDerivativesBookDepthSnapshot[];
  csvBytes: number;
  csvSha256: string;
  sourceCsvHeaderRows: 1;
  sourceCsvRows: number;
  sourceBandRows: number;
  sourceSnapshotCount: number;
  tenBandSnapshotCount: number;
  twelveBandSnapshotCount: number;
  firstTimestampOffsetSeconds: number;
  lastTimestampOffsetSeconds: number;
  minimumSnapshotGapSeconds: number;
  maximumSnapshotGapSeconds: number;
  timestampAdjustedRows: 0;
  timestampFlooredRows: 0;
  timestampShiftedRows: 0;
  filledSnapshotCount: 0;
  repairedBandRows: 0;
}

export interface OfficialBookDepthSource {
  archiveName: string;
  csvName: string;
  url: string;
  checksumUrl: string;
}

export interface BookDepthMetadataRequest {
  day: string;
  parsed: ParsedBookDepthArchive;
  source: OfficialBookDepthSource;
  archiveSha256: string;
  archiveBytes: number;
  checksumResponseSha256: string;
  checksumResponseBytes: number;
  targetContract: string;
  oracleScope: "train-or-validation-target" | "predecessor-context";
  splitContract: OracleCorpusSplitContract;
}

export function officialBookDepthSource(day: string): OfficialBookDepthSource {
  parseUtcDay(day);
  const archiveName = `${BOOK_DEPTH_SYMBOL}-bookDepth-${day}.zip`;
  const csvName = `${BOOK_DEPTH_SYMBOL}-bookDepth-${day}.csv`;
  const url = `${BOOK_DEPTH_SOURCE_ROOT}/${BOOK_DEPTH_SYMBOL}/${archiveName}`;
  return { archiveName, csvName, url, checksumUrl: `${url}.CHECKSUM` };
}

export function isOfficialBookDepthUnavailable(day: string): boolean {
  parseUtcDay(day);
  return OFFICIAL_UNAVAILABLE_SET.has(day);
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Usage: npm run fetch:futures-book-depth -- [options]

  --target-ref-dir PATH             Oracle references defining the fixed split
  --only YYYY-MM-DD,...             Fetch a subset of allowed target/context dates
  --data-dir data
  --dry-run                         Print available and officially absent dates only
  --quiet                           Print only missing-source and final summaries

The fixed scope has 420 train/validation/context days: 368 official archives and
52 explicit early source absences. Sealed ${BOOK_DEPTH_SEALED_TEST_START}..
${BOOK_DEPTH_SEALED_TEST_END} dates are categorically refused and never probed.`);
    return;
  }
  const value = (name: string): string | undefined => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const targetReferenceDirectory = path.resolve(
    value("--target-ref-dir") ?? DEFAULT_TARGET_REFS,
  );
  const splitContract = await readOracleCorpusSplitContract(
    path.resolve(ORACLE_SPLIT_CONTRACT),
  );
  validateSealedContract(splitContract);
  const fullScope = await oracleScopedTradeFlowDates(
    targetReferenceDirectory,
    undefined,
    splitContract,
  );
  const fullAvailableDates = validateAvailabilityContract(fullScope.dates);
  const selectedScope = value("--only")
    ? await oracleScopedTradeFlowDates(
        targetReferenceDirectory,
        parseDateList(value("--only")!),
        splitContract,
      )
    : fullScope;
  const missingSourceDates = selectedScope.dates.filter(
    (day) => OFFICIAL_UNAVAILABLE_SET.has(day),
  );
  const selectedAvailableDates = selectedScope.dates.filter(
    (day) => !OFFICIAL_UNAVAILABLE_SET.has(day),
  );
  const layout = new TradingStorageLayout(path.resolve(value("--data-dir") ?? "data"));
  await refuseUnexpectedNamespaceReferences(layout, new Set(fullAvailableDates));
  const quiet = args.includes("--quiet");
  if (!quiet) {
    console.log(
      `Scope: ${selectedAvailableDates.length} official archives; `
      + `${missingSourceDates.length} deterministic source absences; sealed test `
      + `${fullScope.sealedTestStart}..${fullScope.sealedTestEnd} excluded`,
    );
  }
  if (missingSourceDates.length > 0) {
    console.log(`Official source unavailable (no payload created): ${missingSourceDates.join(",")}`);
  }
  if (args.includes("--dry-run")) {
    console.log(`available:${selectedAvailableDates.join(",")}`);
    return;
  }

  const store = new SequentialShardStore(layout.marketStore);
  const targetContract = targetContractId(targetReferenceDirectory);
  let stored = 0;
  let cached = 0;
  let sourceBytes = 0;
  for (const day of selectedAvailableDates) {
    const oracleScope = fullScope.targetDates.has(day)
      ? "train-or-validation-target"
      : "predecessor-context";
    if (await validCachedBookDepthReference(
      store,
      day,
      targetContract,
      oracleScope,
      splitContract,
    )) {
      cached += 1;
      if (!quiet) console.log(`${day}: cached`);
      continue;
    }
    const referenceFile = store.referenceFile(BOOK_DEPTH_NAMESPACE, day);
    if (await exists(referenceFile)) {
      throw new Error(`${day}: existing immutable USD-M book-depth reference is invalid.`);
    }
    const source = officialBookDepthSource(day);
    const checksumPayload = await requestBuffer(source.checksumUrl, 20_000);
    const checksum = parseOfficialBookDepthChecksum(
      checksumPayload.toString("utf8"),
      source.url,
    );
    const archive = await requestBuffer(source.url, MAXIMUM_ARCHIVE_BYTES);
    const archiveSha256 = sha256(archive);
    if (archiveSha256 !== checksum.sha256) {
      throw new Error(`${source.archiveName}: checksum mismatch.`);
    }
    const parsed = parseBookDepthArchive(archive, source.csvName, day);
    await putDerivativesBookDepthShard(store, {
      namespace: BOOK_DEPTH_NAMESPACE,
      key: day,
      utcDayStartMs: parseUtcDay(day),
      snapshots: parsed.snapshots,
      metadata: createBookDepthMetadata({
        day,
        parsed,
        source,
        archiveSha256,
        archiveBytes: archive.byteLength,
        checksumResponseSha256: sha256(checksumPayload),
        checksumResponseBytes: checksumPayload.byteLength,
        targetContract,
        oracleScope,
        splitContract,
      }),
    });
    stored += 1;
    sourceBytes += archive.byteLength;
    if (!quiet) {
      console.log(
        `${day}: stored ${parsed.sourceSnapshotCount} exact snapshots `
        + `(${parsed.tenBandSnapshotCount} ten-band, `
        + `${parsed.twelveBandSnapshotCount} twelve-band)`,
      );
    }
  }
  console.log(
    `USD-M book-depth ingestion complete: ${stored} stored, ${cached} cached, `
    + `${missingSourceDates.length} officially unavailable, `
    + `${sourceBytes.toLocaleString()} source bytes downloaded.`,
  );
}

export function parseBookDepthArchive(
  archive: Buffer,
  expectedCsvName: string,
  day: string,
): ParsedBookDepthArchive {
  const dayStart = parseUtcDay(day);
  const zip = new AdmZip(archive);
  const entries = zip.getEntries().filter((entry) => !entry.isDirectory);
  if (entries.length !== 1 || entries[0]!.entryName !== expectedCsvName) {
    throw new Error(`${day}: expected one ZIP entry named ${expectedCsvName}.`);
  }
  const payload = entries[0]!.getData();
  const text = payload.toString("utf8").replace(/^\ufeff/, "").trimEnd();
  if (!text) throw new Error(`${day}: USD-M book-depth CSV has no rows.`);
  const lines = text.split(/\r?\n/);
  if (lines[0]!.trim() !== HEADER || lines.length < 2) {
    throw new Error(`${day}: USD-M book-depth CSV schema differs or has no data rows.`);
  }

  const snapshots: SequentialDerivativesBookDepthSnapshot[] = [];
  let currentTimestamp: number | undefined;
  let currentBands = new Map<number, { depth: number; notional: number }>();
  const finishSnapshot = (): void => {
    if (currentTimestamp === undefined) return;
    snapshots.push(snapshotFromBands(currentTimestamp, currentBands, dayStart, day));
    currentBands = new Map();
  };

  lines.slice(1).forEach((line, sourceIndex) => {
    const columns = line.split(",").map((column) => column.trim());
    if (columns.length !== 4) {
      throw new Error(`${day}: expected four book-depth columns at row ${sourceIndex}.`);
    }
    const timestamp = parseSourceTimestamp(columns[0]!, day, sourceIndex);
    if (timestamp < dayStart || timestamp >= dayStart + DAY_MS) {
      throw new Error(`${day}: book-depth timestamp is outside its UTC day at row ${sourceIndex}.`);
    }
    if (currentTimestamp !== undefined && timestamp < currentTimestamp) {
      throw new Error(`${day}: book-depth snapshot timestamps are non-increasing.`);
    }
    if (currentTimestamp !== undefined && timestamp > currentTimestamp) finishSnapshot();
    if (currentTimestamp === undefined || timestamp > currentTimestamp) {
      currentTimestamp = timestamp;
    }
    const percentage = exactPercentage(columns[1]!, day, sourceIndex);
    if (currentBands.has(percentage)) {
      throw new Error(
        `${day}: duplicate ${percentage}% book-depth band at row ${sourceIndex}.`,
      );
    }
    currentBands.set(percentage, {
      depth: positive(columns[2]!, "depth", day, sourceIndex),
      notional: positive(columns[3]!, "notional", day, sourceIndex),
    });
  });
  finishSnapshot();
  if (snapshots.length === 0) throw new Error(`${day}: book-depth archive has no snapshots.`);

  let tenBandSnapshotCount = 0;
  let twelveBandSnapshotCount = 0;
  snapshots.forEach((snapshot) => {
    if (snapshot.schemaBandCount === 10) tenBandSnapshotCount += 1;
    else twelveBandSnapshotCount += 1;
  });
  const gaps = snapshots.slice(1).map((snapshot, index) => (
    snapshot.timestampOffsetSeconds - snapshots[index]!.timestampOffsetSeconds
  ));
  return {
    snapshots,
    csvBytes: payload.byteLength,
    csvSha256: sha256(payload),
    sourceCsvHeaderRows: 1,
    sourceCsvRows: lines.length - 1,
    sourceBandRows: lines.length - 1,
    sourceSnapshotCount: snapshots.length,
    tenBandSnapshotCount,
    twelveBandSnapshotCount,
    firstTimestampOffsetSeconds: snapshots[0]!.timestampOffsetSeconds,
    lastTimestampOffsetSeconds: snapshots.at(-1)!.timestampOffsetSeconds,
    minimumSnapshotGapSeconds: gaps.length === 0 ? 0 : Math.min(...gaps),
    maximumSnapshotGapSeconds: gaps.length === 0 ? 0 : Math.max(...gaps),
    timestampAdjustedRows: 0,
    timestampFlooredRows: 0,
    timestampShiftedRows: 0,
    filledSnapshotCount: 0,
    repairedBandRows: 0,
  };
}

export function createBookDepthMetadata(
  request: BookDepthMetadataRequest,
): StorageMetadata {
  const { parsed, source, splitContract } = request;
  return {
    featureSchema: BOOK_DEPTH_FEATURE_SCHEMA,
    source: "data.binance.vision",
    sourceDataset: "futures/um/daily/bookDepth",
    sourceArchiveUrl: source.url,
    sourceArchiveChecksumUrl: source.checksumUrl,
    sourceArchiveChecksumAlgorithm: "sha256",
    sourceArchiveChecksumFilename: source.archiveName,
    sourceArchiveSha256: request.archiveSha256,
    sourceArchiveBytes: request.archiveBytes,
    sourceChecksumResponseSha256: request.checksumResponseSha256,
    sourceChecksumResponseBytes: request.checksumResponseBytes,
    sourceCsvEntry: source.csvName,
    sourceCsvBytes: parsed.csvBytes,
    sourceCsvSha256: parsed.csvSha256,
    sourceCsvHeaderRows: parsed.sourceCsvHeaderRows,
    sourceCsvRows: parsed.sourceCsvRows,
    sourceBandRows: parsed.sourceBandRows,
    sourceSnapshotCount: parsed.sourceSnapshotCount,
    tenBandSnapshotCount: parsed.tenBandSnapshotCount,
    twelveBandSnapshotCount: parsed.twelveBandSnapshotCount,
    firstTimestampOffsetSeconds: parsed.firstTimestampOffsetSeconds,
    lastTimestampOffsetSeconds: parsed.lastTimestampOffsetSeconds,
    minimumSnapshotGapSeconds: parsed.minimumSnapshotGapSeconds,
    maximumSnapshotGapSeconds: parsed.maximumSnapshotGapSeconds,
    timestampAdjustedRows: parsed.timestampAdjustedRows,
    timestampFlooredRows: parsed.timestampFlooredRows,
    timestampShiftedRows: parsed.timestampShiftedRows,
    filledSnapshotCount: parsed.filledSnapshotCount,
    repairedBandRows: parsed.repairedBandRows,
    market: "usdm-futures",
    symbol: BOOK_DEPTH_SYMBOL,
    timestampResolutionMs: DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS,
    irregularSnapshotAxis: true,
    valuesAreCumulative: true,
    bandPercentages: [...DERIVATIVES_BOOK_DEPTH_BANDS],
    schemaBandCounts: [10, 12],
    sourceAvailability: "official-archive-present",
    officialUnavailableDateCount: BOOK_DEPTH_EXPECTED_UNAVAILABLE_DAYS,
    officialUnavailableDatesSha256: OFFICIAL_UNAVAILABLE_DATES_SHA256,
    oracleAllowlistDays: BOOK_DEPTH_EXPECTED_ALLOWLIST_DAYS,
    oracleAvailableIntersectionDays: BOOK_DEPTH_EXPECTED_AVAILABLE_DAYS,
    oracleTargetContract: request.targetContract,
    oracleReferenceCount: splitContract.referenceCount,
    oracleReferenceFilenameSha256: splitContract.referenceFilenameSha256,
    oracleScope: request.oracleScope,
    sealedTestStart: splitContract.test.first,
    sealedTestEnd: splitContract.test.last,
  };
}

export async function validCachedBookDepthReference(
  store: SequentialShardStore,
  day: string,
  targetContract: string,
  oracleScope: "train-or-validation-target" | "predecessor-context",
  splitContract: OracleCorpusSplitContract,
): Promise<boolean> {
  try {
    const reference = await store.readReference(BOOK_DEPTH_NAMESPACE, day);
    const metadata = reference.metadata ?? {};
    const source = officialBookDepthSource(day);
    const integerNames = [
      "sourceArchiveBytes",
      "sourceChecksumResponseBytes",
      "sourceCsvBytes",
      "sourceCsvHeaderRows",
      "sourceCsvRows",
      "sourceBandRows",
      "sourceSnapshotCount",
      "tenBandSnapshotCount",
      "twelveBandSnapshotCount",
      "firstTimestampOffsetSeconds",
      "lastTimestampOffsetSeconds",
      "minimumSnapshotGapSeconds",
      "maximumSnapshotGapSeconds",
    ] as const;
    const counts = Object.fromEntries(integerNames.map((name) => [
      name,
      Number(metadata[name]),
    ])) as Record<typeof integerNames[number], number>;
    if (reference.namespace !== BOOK_DEPTH_NAMESPACE
      || reference.key !== day
      || reference.sequence.start !== 0
      || reference.sequence.step !== 1
      || reference.sequence.unit !== "index"
      || reference.sequence.count < 1
      || reference.layout.encoding !== DERIVATIVES_BOOK_DEPTH_ENCODING
      || reference.layout.utcDayStartMs !== parseUtcDay(day)
      || metadata.featureSchema !== BOOK_DEPTH_FEATURE_SCHEMA
      || metadata.source !== "data.binance.vision"
      || metadata.sourceDataset !== "futures/um/daily/bookDepth"
      || metadata.sourceArchiveUrl !== source.url
      || metadata.sourceArchiveChecksumUrl !== source.checksumUrl
      || metadata.sourceArchiveChecksumAlgorithm !== "sha256"
      || metadata.sourceArchiveChecksumFilename !== source.archiveName
      || !sha256Metadata(metadata.sourceArchiveSha256)
      || !sha256Metadata(metadata.sourceChecksumResponseSha256)
      || metadata.sourceCsvEntry !== source.csvName
      || !sha256Metadata(metadata.sourceCsvSha256)
      || integerNames.some((name) => !Number.isSafeInteger(counts[name]))
      || counts.sourceArchiveBytes < 1
      || counts.sourceChecksumResponseBytes < 1
      || counts.sourceCsvBytes < 1
      || counts.sourceCsvHeaderRows !== 1
      || counts.sourceCsvRows < 10
      || counts.sourceBandRows !== counts.sourceCsvRows
      || counts.sourceSnapshotCount !== reference.sequence.count
      || counts.tenBandSnapshotCount + counts.twelveBandSnapshotCount
        !== counts.sourceSnapshotCount
      || counts.tenBandSnapshotCount * 10 + counts.twelveBandSnapshotCount * 12
        !== counts.sourceBandRows
      || counts.firstTimestampOffsetSeconds < 0
      || counts.lastTimestampOffsetSeconds < counts.firstTimestampOffsetSeconds
      || counts.lastTimestampOffsetSeconds >= 86_400
      || counts.minimumSnapshotGapSeconds < 0
      || counts.maximumSnapshotGapSeconds < counts.minimumSnapshotGapSeconds
      || metadata.timestampAdjustedRows !== 0
      || metadata.timestampFlooredRows !== 0
      || metadata.timestampShiftedRows !== 0
      || metadata.filledSnapshotCount !== 0
      || metadata.repairedBandRows !== 0
      || metadata.market !== "usdm-futures"
      || metadata.symbol !== BOOK_DEPTH_SYMBOL
      || metadata.timestampResolutionMs !== DERIVATIVES_BOOK_DEPTH_TIMESTAMP_RESOLUTION_MS
      || metadata.irregularSnapshotAxis !== true
      || metadata.valuesAreCumulative !== true
      || !sameNumberArray(metadata.bandPercentages, DERIVATIVES_BOOK_DEPTH_BANDS)
      || !sameNumberArray(metadata.schemaBandCounts, [10, 12])
      || metadata.sourceAvailability !== "official-archive-present"
      || metadata.officialUnavailableDateCount !== BOOK_DEPTH_EXPECTED_UNAVAILABLE_DAYS
      || metadata.officialUnavailableDatesSha256 !== OFFICIAL_UNAVAILABLE_DATES_SHA256
      || metadata.oracleAllowlistDays !== BOOK_DEPTH_EXPECTED_ALLOWLIST_DAYS
      || metadata.oracleAvailableIntersectionDays !== BOOK_DEPTH_EXPECTED_AVAILABLE_DAYS
      || metadata.oracleTargetContract !== targetContract
      || metadata.oracleReferenceCount !== splitContract.referenceCount
      || metadata.oracleReferenceFilenameSha256
        !== splitContract.referenceFilenameSha256
      || metadata.oracleScope !== oracleScope
      || metadata.sealedTestStart !== splitContract.test.first
      || metadata.sealedTestEnd !== splitContract.test.last) {
      return false;
    }

    // readPayload verifies the canonical object's SHA-256 content hash before
    // the codec validates layout, masks, timing, positivity, and monotonicity.
    const decoded = decodeDerivativesBookDepth(
      reference,
      await store.readPayload(reference),
    );
    const tenBandCount = decoded.filter((row) => row.schemaBandCount === 10).length;
    const gaps = decoded.slice(1).map((snapshot, index) => (
      snapshot.timestampOffsetSeconds - decoded[index]!.timestampOffsetSeconds
    ));
    return decoded.length === counts.sourceSnapshotCount
      && tenBandCount === counts.tenBandSnapshotCount
      && decoded.length - tenBandCount === counts.twelveBandSnapshotCount
      && decoded[0]!.timestampOffsetSeconds === counts.firstTimestampOffsetSeconds
      && decoded.at(-1)!.timestampOffsetSeconds === counts.lastTimestampOffsetSeconds
      && (gaps.length === 0 ? 0 : Math.min(...gaps))
        === counts.minimumSnapshotGapSeconds
      && (gaps.length === 0 ? 0 : Math.max(...gaps))
        === counts.maximumSnapshotGapSeconds;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

export function parseOfficialBookDepthChecksum(
  raw: string,
  archiveUrl: string,
): { sha256: string; filename: string } {
  const match = /^([a-fA-F0-9]{64})\s+\*?([^\s]+)$/.exec(raw.trim());
  const filename = path.basename(archiveUrl);
  if (!match || match[2] !== filename) {
    throw new Error(`${filename}: invalid checksum response.`);
  }
  return { sha256: match[1]!.toLowerCase(), filename };
}

async function refuseUnexpectedNamespaceReferences(
  layout: TradingStorageLayout,
  expectedAvailableDates: ReadonlySet<string>,
): Promise<void> {
  const directory = layout.derivativesBookDepthReferences(
    "usdm-futures",
    BOOK_DEPTH_SYMBOL.toLowerCase(),
  );
  let entries: Array<import("node:fs").Dirent>;
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  for (const entry of entries) {
    const match = /^(\d{4}-\d{2}-\d{2})\.json$/.exec(entry.name);
    if (!entry.isFile() || !match) {
      throw new Error(`Unexpected USD-M book-depth namespace entry: ${entry.name}.`);
    }
    const day = match[1]!;
    if (day >= BOOK_DEPTH_SEALED_TEST_START && day <= BOOK_DEPTH_SEALED_TEST_END) {
      throw new Error(`Sealed USD-M book-depth reference is categorically forbidden: ${day}.`);
    }
    if (!expectedAvailableDates.has(day)) {
      throw new Error(`Unexpected USD-M book-depth reference outside the fixed scope: ${day}.`);
    }
  }
}

function validateAvailabilityContract(allDates: readonly string[]): string[] {
  if (allDates.length !== BOOK_DEPTH_EXPECTED_ALLOWLIST_DAYS) {
    throw new Error(
      `Fixed book-depth allowlist changed: expected ${BOOK_DEPTH_EXPECTED_ALLOWLIST_DAYS}, `
      + `found ${allDates.length}.`,
    );
  }
  const allowed = new Set(allDates);
  const missing = BOOK_DEPTH_OFFICIAL_UNAVAILABLE_DATES.filter((day) => allowed.has(day));
  const available = allDates.filter((day) => !OFFICIAL_UNAVAILABLE_SET.has(day));
  if (missing.length !== BOOK_DEPTH_EXPECTED_UNAVAILABLE_DAYS
    || available.length !== BOOK_DEPTH_EXPECTED_AVAILABLE_DAYS) {
    throw new Error(
      "USD-M book-depth official-availability intersection differs from the fixed contract.",
    );
  }
  return available;
}

function validateSealedContract(splitContract: OracleCorpusSplitContract): void {
  if (splitContract.test.first !== BOOK_DEPTH_SEALED_TEST_START
    || splitContract.test.last !== BOOK_DEPTH_SEALED_TEST_END
    || splitContract.test.policy !== "sealed-never-load") {
    throw new Error("Book-depth sealed-test boundary differs from the immutable contract.");
  }
}

function snapshotFromBands(
  timestamp: number,
  bands: ReadonlyMap<number, { depth: number; notional: number }>,
  dayStart: number,
  day: string,
): SequentialDerivativesBookDepthSnapshot {
  const schemaBandCount = bands.size === 10
    && TEN_BAND_PERCENTAGES.every((percentage) => bands.has(percentage))
    ? 10
    : bands.size === 12
      && TWELVE_BAND_PERCENTAGES.every((percentage) => bands.has(percentage))
      ? 12
      : undefined;
  if (schemaBandCount === undefined) {
    throw new Error(
      `${day}: book-depth snapshot has a partial or mixed band schema at `
      + `${new Date(timestamp).toISOString()}.`,
    );
  }
  const available = DERIVATIVES_BOOK_DEPTH_BANDS.map((band) => (
    band !== 0.2 || schemaBandCount === 12
  )) as unknown as DerivativesBookDepthAvailabilityVector;
  const matrix = (
    side: -1 | 1,
    field: "depth" | "notional",
  ): DerivativesBookDepthValueVector => DERIVATIVES_BOOK_DEPTH_BANDS.map(
    (band, index) => available[index] ? bands.get(side * band)![field] : null,
  ) as unknown as DerivativesBookDepthValueVector;
  const snapshot: SequentialDerivativesBookDepthSnapshot = {
    timestampOffsetSeconds: (timestamp - dayStart) / 1_000,
    schemaBandCount,
    bidDepth: matrix(-1, "depth"),
    askDepth: matrix(1, "depth"),
    bidNotional: matrix(-1, "notional"),
    askNotional: matrix(1, "notional"),
    bandAvailable: available,
  };
  for (const name of ["bidDepth", "askDepth", "bidNotional", "askNotional"] as const) {
    let previous = -Infinity;
    snapshot[name].forEach((value, bandIndex) => {
      if (!available[bandIndex]) return;
      if (value! < previous) {
        throw new Error(
          `${day}: cumulative ${name} is non-monotone at `
          + `${new Date(timestamp).toISOString()}.`,
        );
      }
      previous = value!;
    });
  }
  return snapshot;
}

function parseSourceTimestamp(raw: string, day: string, row: number): number {
  if (!/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(raw)) {
    throw new Error(`${day}: invalid one-second UTC timestamp at row ${row}.`);
  }
  const timestamp = Date.parse(`${raw.replace(" ", "T")}.000Z`);
  const canonical = Number.isFinite(timestamp)
    ? new Date(timestamp).toISOString().replace("T", " ").slice(0, 19)
    : "";
  if (canonical !== raw) {
    throw new Error(`${day}: invalid one-second UTC timestamp at row ${row}.`);
  }
  return timestamp;
}

function exactPercentage(raw: string, day: string, row: number): number {
  if (raw === "" || !/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$/.test(raw)) {
    throw new Error(`${day}: invalid book-depth percentage at row ${row}.`);
  }
  const value = Number(raw);
  if (!ALLOWED_PERCENTAGES.has(value)) {
    throw new Error(`${day}: unknown ${raw}% book-depth band at row ${row}.`);
  }
  return value;
}

function positive(raw: string, label: string, day: string, row: number): number {
  const value = Number(raw);
  if (raw === "" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${day}: nonpositive or invalid book-depth ${label} at row ${row}.`);
  }
  return value;
}

function parseUtcDay(day: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`Invalid UTC date: ${day}.`);
  const value = Date.parse(`${day}T00:00:00.000Z`);
  if (!Number.isSafeInteger(value) || new Date(value).toISOString().slice(0, 10) !== day) {
    throw new Error(`Invalid UTC date: ${day}.`);
  }
  return value;
}

function utcDateRange(first: string, last: string): string[] {
  const dates: string[] = [];
  for (let timestamp = parseUtcDay(first); timestamp <= parseUtcDay(last); timestamp += DAY_MS) {
    dates.push(new Date(timestamp).toISOString().slice(0, 10));
  }
  return dates;
}

function sha256(payload: Uint8Array): string {
  return createHash("sha256").update(payload).digest("hex");
}

function sha256Metadata(value: unknown): boolean {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function sameNumberArray(actual: unknown, expected: readonly number[]): boolean {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
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
