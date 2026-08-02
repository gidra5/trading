import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import {
  DailyCandleRecorder,
  DailyTradeFlowAccumulator,
  decodeCandles,
  decodeDerivativesMetrics,
  decodeTradeFlow,
  deduplicateFilesWithHardLinks,
  encodeCandles,
  encodeTradeFlow,
  putCandleShard,
  putDerivativesMetricsShard,
  putTradeFlowShard,
  readCandleShardReference,
  readDerivativesMetricsShardReference,
  readTradeFlowShardReference,
  SequentialShardStore,
  TradingStorageLayout,
  type SequentialCandle,
  type SequentialDerivativesMetricRow,
} from "../src/index.js";

const execFileAsync = promisify(execFile);

function candle(
  openTime: number,
  interval = "12h",
  stepMs = 43_200_000,
): SequentialCandle {
  const close = 100 + openTime / stepMs;
  return {
    symbol: "BTCUSDT",
    interval,
    openTime,
    closeTime: openTime + stepMs - 1,
    open: close - 1,
    high: close + 2,
    low: close - 3,
    close,
    volume: 4.5,
    closed: true,
  };
}

test("stores equal sequential payloads once without compatibility copies", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-storage-"));
  try {
    const store = new SequentialShardStore(path.join(root, "storage"));
    const payload = Buffer.from(Array.from({ length: 256 }, (_, index) => index));
    const first = await store.put({
      namespace: "features/schema-1",
      key: "dataset-a/2026-01-01",
      payload,
      sequence: { start: 1_767_225_600_000, step: 1_000, count: 128, unit: "unix-ms" },
      layout: { encoding: "row-major", dtype: "float16", columns: 1 },
    });
    const second = await store.put({
      namespace: "features/schema-1",
      key: "dataset-b/2026-01-01",
      payload,
      sequence: { start: 1_767_225_600_000, step: 1_000, count: 128, unit: "unix-ms" },
      layout: { encoding: "row-major", dtype: "float16", columns: 1 },
    });
    assert.equal(first.objectCreated, true);
    assert.equal(second.objectCreated, false);
    assert.equal(first.reference.object.contentHash, second.reference.object.contentHash);
    await assert.rejects(fs.stat(path.join(root, "dataset-a")), { code: "ENOENT" });
    assert.deepEqual(await store.readPayload(second.reference), payload);
    const audit = await store.audit();
    assert.equal(audit.references, 2);
    assert.equal(audit.objects, 1);
    assert.equal(audit.orphanObjects.length, 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("hash-verified deduplication consolidates independent files without touching mismatches", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-file-dedupe-"));
  try {
    const directories = ["a", "b", "c", "linked"];
    await Promise.all(directories.map((directory) =>
      fs.mkdir(path.join(root, directory), { recursive: true })));
    const payload = Buffer.alloc(128 * 1_024, 17);
    const different = Buffer.alloc(payload.length, 23);
    const canonical = path.join(root, "a", "tensor.bin");
    const duplicate = path.join(root, "b", "tensor.bin");
    const mismatch = path.join(root, "c", "tensor.bin");
    const existingLink = path.join(root, "linked", "tensor.bin");
    await fs.writeFile(canonical, payload);
    await fs.writeFile(duplicate, payload);
    await fs.writeFile(mismatch, different);
    await fs.link(canonical, existingLink);

    const dryRun = await deduplicateFilesWithHardLinks({ root, minimumBytes: 1 });
    assert.equal(dryRun.mode, "dry-run");
    assert.equal(dryRun.duplicateObjects, 1);
    assert.equal(dryRun.duplicatePaths, 1);
    assert.equal(dryRun.duplicateBytes, payload.length);

    const applied = await deduplicateFilesWithHardLinks({ root, minimumBytes: 1, apply: true });
    assert.equal(applied.linkedPaths, 1);
    assert.equal(applied.reclaimedBytes, payload.length);
    assert.deepEqual(await fs.readFile(duplicate), payload);
    assert.deepEqual(await fs.readFile(mismatch), different);
    const [canonicalStat, duplicateStat, linkStat, mismatchStat] = await Promise.all([
      fs.stat(canonical, { bigint: true }),
      fs.stat(duplicate, { bigint: true }),
      fs.stat(existingLink, { bigint: true }),
      fs.stat(mismatch, { bigint: true }),
    ]);
    assert.equal(duplicateStat.ino, canonicalStat.ino);
    assert.equal(linkStat.ino, canonicalStat.ino);
    assert.notEqual(mismatchStat.ino, canonicalStat.ino);
    assert.equal(
      (await deduplicateFilesWithHardLinks({ root, minimumBytes: 1 })).duplicateObjects,
      0,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("candle codec uses an implicit axis with sparse gaps and exact wick values", async () => {
  const start = 1_767_225_600_000;
  const candles: SequentialCandle[] = Array.from({ length: 121 }, (_, index) => {
    const close = 91_234.12345678 + (index % 7) * 0.00000001;
    return {
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: start + index * 1_000,
      closeTime: start + index * 1_000 + 999,
      open: close - 0.01,
      high: close + 0.02,
      low: close - 0.03,
      close,
      volume: index % 5 === 0 ? 0 : 0.00012345 + index * 0.00000001,
      closed: true,
    };
  }).filter((_, index) => index !== 60);
  candles[30]!.closeTime = candles[30]!.openTime;
  candles[90]!.closed = false;
  const encoded = encodeCandles(candles);
  assert.equal(encoded.sequence.start, start);
  assert.equal(encoded.sequence.step, 1_000);
  assert.equal(encoded.sequence.count, candles.length);
  assert.deepEqual(encoded.layout.timeJumps, [{ index: 60, deltaMs: 1_000 }]);
  assert.deepEqual(encoded.layout.closeTimeOffsetOverrides, [{ index: 30, offsetMs: 0 }]);
  assert.deepEqual(encoded.layout.closedOverrides, [{ index: 90, closed: false }]);
  assert.ok(encoded.payload.byteLength < candles.length * 5 * 8);
  assert.deepEqual(
    decodeCandles({
      version: 1,
      kind: "trading-sequential-shard",
      namespace: "candles/spot-btcusdt/btcusdt/1s",
      key: "2026-01-01",
      createdAt: new Date().toISOString(),
      object: {
        algorithm: "sha256",
        contentHash: "0".repeat(64),
        file: "objects/sha256/00/fake.zst",
        compression: "zstd",
        compressionLevel: 9,
        uncompressedBytes: encoded.payload.byteLength,
        compressedBytes: 0,
      },
      sequence: encoded.sequence,
      layout: encoded.layout,
    }, encoded.payload),
    candles,
  );
});

test("canonical candle references are read directly", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-candles-"));
  try {
    const store = new SequentialShardStore(path.join(root, "storage"));
    const candles: SequentialCandle[] = [{
      symbol: "BTCUSDT",
      interval: "1m",
      openTime: 1_767_225_600_000,
      closeTime: 1_767_225_659_999,
      open: 1,
      high: 2,
      low: 0.5,
      close: 1.5,
      volume: 3,
      closed: true,
    }];
    const stored = await putCandleShard(store, {
      namespace: "candles/spot-btcusdt/btcusdt/1m",
      key: "2026-01-01",
      candles,
      stepMs: 60_000,
    });
    assert.deepEqual(await readCandleShardReference(stored.referenceFile), candles);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("trade-flow accumulator preserves aggressor direction, timing, and raw-trade counts", () => {
  const day = Date.parse("2026-01-01T00:00:00.000Z");
  const accumulator = new DailyTradeFlowAccumulator(day);
  accumulator.append({
    timeMicros: day * 1_000 + 100_000,
    price: 100,
    quantity: 2,
    tradeCount: 3,
    aggressorSide: 1,
  });
  accumulator.append({
    timeMicros: day * 1_000 + 300_000,
    price: 101,
    quantity: 1,
    tradeCount: 2,
    aggressorSide: -1,
  });
  accumulator.append({
    timeMicros: day * 1_000 + 900_000,
    price: 102,
    quantity: 3,
    tradeCount: 1,
    aggressorSide: 1,
  });
  const first = accumulator.finish()[0]!;
  assert.equal(first.aggressiveBuyBaseVolume, 5);
  assert.equal(first.aggressiveSellBaseVolume, 1);
  assert.equal(first.aggressiveBuyQuoteVolume, 506);
  assert.equal(first.aggressiveSellQuoteVolume, 101);
  assert.equal(first.aggressiveBuyAggregateQuantitySquared, 13);
  assert.equal(first.aggressiveSellAggregateQuantitySquared, 1);
  assert.equal(first.aggressiveBuyMaxAggregateQuantity, 3);
  assert.equal(first.aggressiveSellMaxAggregateQuantity, 1);
  assert.ok(Math.abs(first.aggressiveBuyBaseVolumeTimeMoment - 2.9) < 1e-12);
  assert.ok(Math.abs(first.aggressiveSellBaseVolumeTimeMoment - 0.3) < 1e-12);
  assert.equal(first.aggressiveBuyAggregateTradeCount, 2);
  assert.equal(first.aggressiveSellAggregateTradeCount, 1);
  assert.equal(first.aggressiveBuyTradeCount, 4);
  assert.equal(first.aggressiveSellTradeCount, 2);
  assert.equal(first.aggressorSideFlipCount, 2);
  assert.equal(first.firstAggressorSide, 1);
  assert.equal(first.lastAggressorSide, 1);
  assert.equal(first.firstTradeOffsetMicros, 100_000);
  assert.equal(first.lastTradeOffsetMicros, 900_000);
  assert.equal(accumulator.finish()[1]!.firstAggressorSide, 0);
});

test("trade-flow codec round-trips a dense UTC day through immutable storage", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-trade-flow-"));
  try {
    const day = Date.parse("2026-01-01T00:00:00.000Z");
    const accumulator = new DailyTradeFlowAccumulator(day, 43_200_000);
    accumulator.append({
      timeMicros: day * 1_000 + 123_456,
      price: 100.25,
      quantity: 0.125,
      tradeCount: 4,
      aggressorSide: -1,
    });
    const encoded = encodeTradeFlow(accumulator.finish(), 43_200_000);
    const fakeReference = {
      version: 1,
      kind: "trading-sequential-shard",
      namespace: "trade-flow/spot-btcusdt/btcusdt/12h",
      key: "2026-01-01",
      createdAt: new Date().toISOString(),
      object: {
        algorithm: "sha256",
        contentHash: "0".repeat(64),
        file: "objects/sha256/00/fake.zst",
        compression: "zstd",
        compressionLevel: 9,
        uncompressedBytes: encoded.payload.byteLength,
        compressedBytes: 0,
      },
      sequence: encoded.sequence,
      layout: encoded.layout,
    } as const;
    assert.deepEqual(decodeTradeFlow(fakeReference, encoded.payload), accumulator.finish());

    const store = new SequentialShardStore(path.join(root, "immutable"));
    const stored = await putTradeFlowShard(store, {
      namespace: "trade-flow/spot-btcusdt/btcusdt/12h",
      key: "2026-01-01",
      seconds: accumulator.finish(),
      stepMs: 43_200_000,
    });
    assert.deepEqual(
      await readTradeFlowShardReference(stored.referenceFile),
      accumulator.finish(),
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("derivatives-metrics codec round-trips dense positive 5m rows", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-derivatives-metrics-"));
  try {
    const start = Date.parse("2026-01-01T00:00:00.000Z");
    const rows: SequentialDerivativesMetricRow[] = Array.from(
      { length: 288 },
      (_, index) => ({
        openTime: start + index * 300_000,
        sumOpenInterest: 100_000 + index,
        sumOpenInterestValue: 6_000_000_000 + index * 100,
        topTraderAccountLongShortRatio: 1.8 + index / 10_000,
        topTraderPositionLongShortRatio: 1.1 + index / 20_000,
        globalLongShortRatio: 1.7 + index / 30_000,
        takerBuySellVolumeRatio: 0.9 + index / 40_000,
      }),
    );
    const store = new SequentialShardStore(path.join(root, "storage"));
    const stored = await putDerivativesMetricsShard(store, {
      namespace: "derivatives-metrics/usdm-futures/btcusdt/5m",
      key: "2026-01-01",
      rows,
      stepMs: 300_000,
    });
    assert.deepEqual(
      decodeDerivativesMetrics(
        stored.reference,
        await store.readPayload(stored.reference),
      ),
      rows,
    );
    assert.deepEqual(
      await readDerivativesMetricsShardReference(stored.referenceFile),
      rows,
    );
    const invalid = rows.slice();
    invalid[2] = { ...invalid[2]!, globalLongShortRatio: 0 };
    await assert.rejects(
      putDerivativesMetricsShard(store, {
        namespace: "derivatives-metrics/usdm-futures/btcusdt/5m",
        key: "invalid",
        rows: invalid,
      }),
      /invalid value/,
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("daily recorder resumes mutable days and seals them into canonical storage", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-live-candles-"));
  try {
    const store = new SequentialShardStore(path.join(root, "immutable"));
    const stagingDirectory = path.join(root, "mutable", "candles", "spot-btcusdt", "btcusdt", "12h");
    const options = {
      store,
      namespace: "candles/spot-btcusdt/btcusdt/12h",
      stagingDirectory,
      symbol: "BTCUSDT",
      interval: "12h",
      stepMs: 43_200_000,
    };
    const day = Date.parse("2026-01-01T00:00:00.000Z");
    const first = candle(day);
    const second = candle(day + 43_200_000);

    const initial = new DailyCandleRecorder(options);
    assert.deepEqual(await initial.append(first), { appended: true, sealed: false });
    assert.deepEqual(await initial.append(first), { appended: false, sealed: false });
    assert.deepEqual(await initial.readRecent(10), [first]);

    const resumed = new DailyCandleRecorder(options);
    const sealed = await resumed.append(second);
    assert.equal(sealed.appended, true);
    assert.equal(sealed.sealed, true);
    assert.ok(sealed.referenceFile);
    await assert.rejects(
      fs.stat(path.join(stagingDirectory, "2026-01-01.jsonl")),
      { code: "ENOENT" },
    );
    assert.deepEqual(await resumed.readRecent(10), [first, second]);

    const nextDay = candle(day + 86_400_000);
    await resumed.append(nextDay);
    assert.deepEqual(await resumed.readRecent(2), [second, nextDay]);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("daily recorder stores each closed multi-day candle as one immutable shard", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-weekly-candles-"));
  try {
    const recorder = new DailyCandleRecorder({
      store: new SequentialShardStore(path.join(root, "immutable")),
      namespace: "candles/spot-btcusdt/btcusdt/1w",
      stagingDirectory: path.join(root, "mutable"),
      symbol: "BTCUSDT",
      interval: "1w",
      stepMs: 604_800_000,
    });
    const weekly = candle(Date.parse("2026-01-05T00:00:00.000Z"), "1w", 604_800_000);
    const result = await recorder.append(weekly);
    assert.equal(result.sealed, true);
    assert.deepEqual(await recorder.readRecent(1), [weekly]);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("live candle migration splits the unbounded stream into immutable and mutable days", async () => {
  const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "trading-live-migration-"));
  try {
    const legacyDirectory = path.join(dataDir, "market", "spot-btcusdt");
    const legacyFile = path.join(legacyDirectory, "btcusdt-12h-candles.jsonl");
    const firstDay = Date.parse("2026-01-01T00:00:00.000Z");
    const values = [
      candle(firstDay),
      candle(firstDay + 43_200_000),
      candle(firstDay + 86_400_000),
    ];
    await fs.mkdir(legacyDirectory, { recursive: true });
    await fs.writeFile(legacyFile, `${values.map((value) => JSON.stringify(value)).join("\n")}\n`);

    const repoRoot = path.resolve(import.meta.dirname, "../../..");
    await execFileAsync(process.execPath, [
      "--conditions=development",
      "--import",
      "tsx",
      path.join(repoRoot, "scripts", "migrate-ml-storage.ts"),
      "--live-candles",
      "--apply",
      "--confirm-stopped",
      "--data-dir",
      dataDir,
    ], { cwd: repoRoot });

    const layout = new TradingStorageLayout(dataDir);
    const reference = path.join(
      layout.candleReferences("spot-btcusdt", "btcusdt", "12h"),
      "2026-01-01.json",
    );
    assert.deepEqual(await readCandleShardReference(reference), values.slice(0, 2));
    const staging = path.join(
      layout.marketMutable,
      "candles",
      "spot-btcusdt",
      "btcusdt",
      "12h",
      "2026-01-02.jsonl",
    );
    assert.deepEqual(
      (await fs.readFile(staging, "utf8")).trim().split(/\r?\n/).map(JSON.parse),
      values.slice(2),
    );
    await assert.rejects(fs.stat(legacyFile), { code: "ENOENT" });
  } finally {
    await fs.rm(dataDir, { recursive: true, force: true });
  }
});

test("audit and grace-period GC identify only unreachable canonical objects", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-storage-gc-"));
  try {
    const store = new SequentialShardStore(path.join(root, "storage"));
    const stored = await store.put({
      namespace: "oracle/1s",
      key: "dataset/day",
      payload: Buffer.from("immutable payload"),
      sequence: { start: 0, step: 1_000, count: 1, unit: "unix-ms" },
      layout: { encoding: "row-major", dtype: "float32-le", columns: 1 },
    });
    await fs.rm(stored.referenceFile);
    const audit = await store.audit();
    assert.equal(audit.orphanObjects.length, 1);
    assert.equal(audit.orphanObjects[0]!.file, stored.objectFile);
    const dryRun = await store.collectGarbage({ minimumAgeMs: 0 });
    assert.equal(dryRun.removedObjects, 0);
    assert.ok((await fs.stat(stored.objectFile)).isFile());
    const applied = await store.collectGarbage({ apply: true, minimumAgeMs: 0 });
    assert.equal(applied.removedObjects, 1);
    await assert.rejects(fs.stat(stored.objectFile), { code: "ENOENT" });
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
