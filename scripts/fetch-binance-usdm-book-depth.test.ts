import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import AdmZip from "adm-zip";
import {
  putDerivativesBookDepthShard,
  SequentialShardStore,
} from "@trading/storage";
import type { OracleCorpusSplitContract } from "./lib/binance-agg-trades.js";
import {
  BOOK_DEPTH_EXPECTED_AVAILABLE_DAYS,
  BOOK_DEPTH_EXPECTED_ALLOWLIST_DAYS,
  BOOK_DEPTH_EXPECTED_UNAVAILABLE_DAYS,
  BOOK_DEPTH_FEATURE_SCHEMA,
  BOOK_DEPTH_NAMESPACE,
  BOOK_DEPTH_OFFICIAL_UNAVAILABLE_DATES,
  createBookDepthMetadata,
  isOfficialBookDepthUnavailable,
  officialBookDepthSource,
  parseBookDepthArchive,
  parseOfficialBookDepthChecksum,
  validCachedBookDepthReference,
} from "./fetch-binance-usdm-book-depth.js";

const DAY = "2026-01-01";
const CSV_NAME = `BTCUSDT-bookDepth-${DAY}.csv`;
const HEADER = "timestamp,percentage,depth,notional";
const HASH = "ab".repeat(32);
const SPLIT_CONTRACT: OracleCorpusSplitContract = {
  schemaVersion: 1,
  targetContract: "hindsight-bot-71391c44b323e044e6ab",
  referenceCount: 432,
  referenceFilenameSha256: "cd".repeat(32),
  filenameHashEncoding: "utf8-lf-with-trailing-lf",
  train: { count: 372, first: "2021-09-08", last: "2026-05-24" },
  validation: { count: 30, first: "2026-05-25", last: "2026-06-23" },
  test: {
    count: 30,
    first: "2026-06-24",
    last: "2026-07-23",
    policy: "sealed-never-load",
  },
};

function timestamp(seconds: number): string {
  return `2026-01-01 00:${String(Math.floor(seconds / 60)).padStart(2, "0")}:`
    + String(seconds % 60).padStart(2, "0");
}

function snapshotRows(
  seconds: number,
  twelveBand: boolean,
  mutate?: (percentage: number, depth: number, notional: number) => [number, number],
): string[] {
  const percentages = twelveBand
    ? [-5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5]
    : [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5];
  return percentages.map((percentage) => {
    const distance = Math.abs(percentage);
    const initialDepth = distance * 10;
    const initialNotional = distance * 1_000;
    const [depth, notional] = mutate?.(
      percentage,
      initialDepth,
      initialNotional,
    ) ?? [initialDepth, initialNotional];
    return [timestamp(seconds), percentage, depth, notional].join(",");
  });
}

function archive(
  rows: string[],
  options: { name?: string; header?: string } = {},
): Buffer {
  const zip = new AdmZip();
  zip.addFile(
    options.name ?? CSV_NAME,
    Buffer.from([options.header ?? HEADER, ...rows, ""].join("\n")),
  );
  return zip.toBuffer();
}

test("pins the official bookDepth source, checksum, and 368/420 availability contract", () => {
  assert.deepEqual(officialBookDepthSource(DAY), {
    archiveName: "BTCUSDT-bookDepth-2026-01-01.zip",
    csvName: CSV_NAME,
    url: "https://data.binance.vision/data/futures/um/daily/bookDepth/"
      + "BTCUSDT/BTCUSDT-bookDepth-2026-01-01.zip",
    checksumUrl: "https://data.binance.vision/data/futures/um/daily/bookDepth/"
      + "BTCUSDT/BTCUSDT-bookDepth-2026-01-01.zip.CHECKSUM",
  });
  assert.deepEqual(
    parseOfficialBookDepthChecksum(
      `${HASH.toUpperCase()}  BTCUSDT-bookDepth-2026-01-01.zip\n`,
      officialBookDepthSource(DAY).url,
    ),
    { sha256: HASH, filename: "BTCUSDT-bookDepth-2026-01-01.zip" },
  );
  assert.throws(
    () => parseOfficialBookDepthChecksum(`${HASH} wrong.zip`, officialBookDepthSource(DAY).url),
    /invalid checksum response/,
  );
  assert.equal(BOOK_DEPTH_EXPECTED_ALLOWLIST_DAYS, 420);
  assert.equal(BOOK_DEPTH_EXPECTED_AVAILABLE_DAYS, 368);
  assert.equal(BOOK_DEPTH_EXPECTED_UNAVAILABLE_DAYS, 52);
  assert.equal(BOOK_DEPTH_OFFICIAL_UNAVAILABLE_DATES.length, 52);
  assert.equal(isOfficialBookDepthUnavailable("2021-09-07"), true);
  assert.equal(isOfficialBookDepthUnavailable("2022-08-03"), true);
  assert.equal(isOfficialBookDepthUnavailable("2023-03-02"), false);
  assert.equal(isOfficialBookDepthUnavailable("2026-06-24"), false);
});

test("parses exact ten/twelve-band snapshots without flooring, filling, or repair", () => {
  const parsed = parseBookDepthArchive(
    archive([
      ...snapshotRows(7, false),
      ...snapshotRows(37, true),
    ]),
    CSV_NAME,
    DAY,
  );
  assert.equal(parsed.sourceCsvHeaderRows, 1);
  assert.equal(parsed.sourceCsvRows, 22);
  assert.equal(parsed.sourceBandRows, 22);
  assert.equal(parsed.sourceSnapshotCount, 2);
  assert.equal(parsed.tenBandSnapshotCount, 1);
  assert.equal(parsed.twelveBandSnapshotCount, 1);
  assert.equal(parsed.firstTimestampOffsetSeconds, 7);
  assert.equal(parsed.lastTimestampOffsetSeconds, 37);
  assert.equal(parsed.minimumSnapshotGapSeconds, 30);
  assert.equal(parsed.maximumSnapshotGapSeconds, 30);
  assert.equal(parsed.timestampAdjustedRows, 0);
  assert.equal(parsed.timestampFlooredRows, 0);
  assert.equal(parsed.timestampShiftedRows, 0);
  assert.equal(parsed.filledSnapshotCount, 0);
  assert.equal(parsed.repairedBandRows, 0);
  assert.deepEqual(parsed.snapshots[0], {
    timestampOffsetSeconds: 7,
    schemaBandCount: 10,
    bidDepth: [null, 10, 20, 30, 40, 50],
    askDepth: [null, 10, 20, 30, 40, 50],
    bidNotional: [null, 1_000, 2_000, 3_000, 4_000, 5_000],
    askNotional: [null, 1_000, 2_000, 3_000, 4_000, 5_000],
    bandAvailable: [false, true, true, true, true, true],
  });
  assert.deepEqual(parsed.snapshots[1]!.bidDepth, [2, 10, 20, 30, 40, 50]);
  assert.deepEqual(parsed.snapshots[1]!.bandAvailable, [true, true, true, true, true, true]);
});

test("rejects ZIP/schema/band/timestamp/value/cumulative source corruption", () => {
  const goodTen = snapshotRows(7, false);
  const goodTwelve = snapshotRows(37, true);
  const cases: Array<{ payload: Buffer; pattern: RegExp }> = [
    {
      payload: archive(goodTen, { name: "wrong.csv" }),
      pattern: /expected one ZIP entry/,
    },
    {
      payload: archive(goodTen, { header: "time,percentage,depth,notional" }),
      pattern: /CSV schema differs/,
    },
    {
      payload: archive(goodTen.slice(1)),
      pattern: /partial or mixed band schema/,
    },
    {
      payload: archive([...goodTen, goodTen[0]!]),
      pattern: /duplicate -5% book-depth band/,
    },
    {
      payload: archive([
        ...goodTen.slice(0, -1),
        `${timestamp(7)},6,60,6000`,
      ]),
      pattern: /unknown 6% book-depth band/,
    },
    {
      payload: archive([...goodTwelve, ...goodTen]),
      pattern: /snapshot timestamps are non-increasing/,
    },
    {
      payload: archive(snapshotRows(7, false, (percentage, depth, notional) => (
        percentage === -1 ? [0, notional] : [depth, notional]
      ))),
      pattern: /nonpositive or invalid book-depth depth/,
    },
    {
      payload: archive(snapshotRows(7, false, (percentage, depth, notional) => (
        percentage === -2 ? [5, notional] : [depth, notional]
      ))),
      pattern: /cumulative bidDepth is non-monotone/,
    },
    {
      payload: archive([
        ...snapshotRows(7, false).map((line) => line.replace(timestamp(7), "2026-01-01 00:00:07.000")),
      ]),
      pattern: /invalid one-second UTC timestamp/,
    },
  ];
  cases.forEach(({ payload, pattern }) => {
    assert.throws(() => parseBookDepthArchive(payload, CSV_NAME, DAY), pattern);
  });
});

test("validates cached object hashes, metadata, decoded counters, and schema", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-book-depth-cache-"));
  try {
    const sourceArchive = archive([
      ...snapshotRows(7, false),
      ...snapshotRows(37, true),
    ]);
    const parsed = parseBookDepthArchive(sourceArchive, CSV_NAME, DAY);
    const store = new SequentialShardStore(path.join(root, "market", "immutable"));
    const source = officialBookDepthSource(DAY);
    const metadata = createBookDepthMetadata({
      day: DAY,
      parsed,
      source,
      archiveSha256: HASH,
      archiveBytes: sourceArchive.byteLength,
      checksumResponseSha256: "ef".repeat(32),
      checksumResponseBytes: 96,
      targetContract: SPLIT_CONTRACT.targetContract,
      oracleScope: "train-or-validation-target",
      splitContract: SPLIT_CONTRACT,
    });
    assert.equal(metadata.featureSchema, BOOK_DEPTH_FEATURE_SCHEMA);
    const stored = await putDerivativesBookDepthShard(store, {
      namespace: BOOK_DEPTH_NAMESPACE,
      key: DAY,
      utcDayStartMs: Date.parse(`${DAY}T00:00:00.000Z`),
      snapshots: parsed.snapshots,
      metadata,
    });
    assert.equal(await validCachedBookDepthReference(
      store,
      DAY,
      SPLIT_CONTRACT.targetContract,
      "train-or-validation-target",
      SPLIT_CONTRACT,
    ), true);

    const storedObject = await fs.readFile(stored.objectFile);
    const corruptedObject = Buffer.from(storedObject);
    corruptedObject[Math.floor(corruptedObject.length / 2)]! ^= 0xff;
    await fs.writeFile(stored.objectFile, corruptedObject);
    await assert.rejects(() => validCachedBookDepthReference(
      store,
      DAY,
      SPLIT_CONTRACT.targetContract,
      "train-or-validation-target",
      SPLIT_CONTRACT,
    ));
    await fs.writeFile(stored.objectFile, storedObject);

    const reference = JSON.parse(await fs.readFile(stored.referenceFile, "utf8")) as {
      metadata: Record<string, unknown>;
    };
    reference.metadata.sourceSnapshotCount = 3;
    await fs.writeFile(stored.referenceFile, `${JSON.stringify(reference, null, 2)}\n`);
    assert.equal(await validCachedBookDepthReference(
      store,
      DAY,
      SPLIT_CONTRACT.targetContract,
      "train-or-validation-target",
      SPLIT_CONTRACT,
    ), false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
