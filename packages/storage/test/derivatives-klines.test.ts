import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  decodeDerivativesKlines,
  DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS,
  DERIVATIVES_KLINE_COLUMNS,
  DERIVATIVES_KLINE_ENCODING,
  encodeDerivativesKlines,
  putDerivativesKlinesShard,
  readDerivativesKlinesShardReference,
  SequentialShardStore,
  TradingStorageLayout,
  type SequentialDerivativesKlineRow,
} from "../src/index.js";

const START = Date.parse("2026-01-01T00:00:00.000Z");

function row(index: number): SequentialDerivativesKlineRow {
  const open = 100 + index;
  const close = open + 0.25;
  return {
    openTime: START + index * 60_000,
    open,
    high: close + 0.5,
    low: open - 0.5,
    close,
    baseVolume: 10 + index,
    quoteVolume: 1_010 + index,
    tradeCount: 20 + index,
    takerBuyBaseVolume: 4 + index / 10,
    takerBuyQuoteVolume: 404 + index,
  };
}

function missing(index: number): SequentialDerivativesKlineRow {
  return {
    openTime: START + index * 60_000,
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

test("USD-M kline codec exports and round-trips complete or fully missing rows", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-derivatives-klines-"));
  try {
    const rows = [row(0), missing(1), row(2)];
    const store = new SequentialShardStore(path.join(root, "market", "immutable"));
    const stored = await putDerivativesKlinesShard(store, {
      namespace: "derivatives-klines/usdm-futures/btcusdt/1m",
      key: "2026-01-01",
      rows,
    });
    assert.equal(stored.reference.layout.encoding, DERIVATIVES_KLINE_ENCODING);
    assert.equal(stored.reference.layout.closeTimeOffsetMs, 59_999);
    assert.equal(DERIVATIVES_KLINE_CLOSE_TIME_OFFSET_MS, 59_999);
    assert.deepEqual(
      (stored.reference.layout.columns as Array<{ name: string }>).map((column) => column.name),
      [...DERIVATIVES_KLINE_COLUMNS, "validMask"],
    );
    assert.deepEqual(
      decodeDerivativesKlines(stored.reference, await store.readPayload(stored.reference)),
      rows,
    );
    assert.deepEqual(await readDerivativesKlinesShardReference(stored.referenceFile), rows);

    const layout = new TradingStorageLayout(root);
    assert.equal(
      layout.derivativesKlinesReferences("usdm-futures", "btcusdt", "1m"),
      path.join(
        root,
        "market",
        "immutable",
        "refs",
        "derivatives-klines",
        "usdm-futures",
        "btcusdt",
        "1m",
      ),
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("USD-M kline codec reconciles row-level and market invariants", () => {
  const partial = missing(0);
  partial.close = 100;
  assert.throws(() => encodeDerivativesKlines([partial]), /partially missing/);
  assert.throws(
    () => encodeDerivativesKlines([{ ...row(0), high: 99 }]),
    /invalid row/,
  );
  assert.throws(
    () => encodeDerivativesKlines([{ ...row(0), takerBuyBaseVolume: 11 }]),
    /invalid row/,
  );
  assert.throws(
    () => encodeDerivativesKlines([{ ...row(0), tradeCount: 1.5 }]),
    /invalid row/,
  );
  assert.throws(
    () => encodeDerivativesKlines([{
      ...row(0),
      high: row(0).open,
      low: row(0).open,
      close: row(0).open,
      tradeCount: 0,
    }]),
    /invalid row/,
  );
  assert.throws(
    () => encodeDerivativesKlines([row(0), { ...row(1), openTime: START + 60_001 }]),
    /time axis/,
  );
});

test("USD-M kline decoder rejects fabricated values in missing source rows", () => {
  const encoded = encodeDerivativesKlines([row(0)]);
  const reference = {
    version: 1,
    kind: "trading-sequential-shard",
    namespace: "derivatives-klines/usdm-futures/btcusdt/1m",
    key: "2026-01-01",
    createdAt: "2026-01-02T00:00:00.000Z",
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
  const missingMask = Buffer.from(encoded.payload);
  missingMask[missingMask.length - 1] = 0;
  assert.throws(
    () => decodeDerivativesKlines(reference, missingMask),
    /missing row 0 contains nonzero data/,
  );
  const invalidMask = Buffer.from(encoded.payload);
  invalidMask[invalidMask.length - 1] = 2;
  assert.throws(
    () => decodeDerivativesKlines(reference, invalidMask),
    /validity mask is invalid/,
  );
});
