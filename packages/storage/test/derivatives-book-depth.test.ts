import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  decodeDerivativesBookDepth,
  DERIVATIVES_BOOK_DEPTH_BANDS,
  DERIVATIVES_BOOK_DEPTH_ENCODING,
  DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS,
  encodeDerivativesBookDepth,
  putDerivativesBookDepthShard,
  readDerivativesBookDepthShardReference,
  SequentialShardStore,
  TradingStorageLayout,
  type DerivativesBookDepthValueVector,
  type SequentialDerivativesBookDepthSnapshot,
} from "../src/index.js";

const DAY = "2026-01-01";
const START = Date.parse(`${DAY}T00:00:00.000Z`);

function vector(
  base: number,
  twelveBand = true,
): DerivativesBookDepthValueVector {
  return [
    twelveBand ? base : null,
    base + 1,
    base + 2,
    base + 3,
    base + 4,
    base + 5,
  ];
}

function snapshot(
  timestampOffsetSeconds: number,
  schemaBandCount: 10 | 12 = 12,
): SequentialDerivativesBookDepthSnapshot {
  const twelveBand = schemaBandCount === 12;
  return {
    timestampOffsetSeconds,
    schemaBandCount,
    bidDepth: vector(10, twelveBand),
    askDepth: vector(20, twelveBand),
    bidNotional: vector(1_000, twelveBand),
    askNotional: vector(2_000, twelveBand),
    bandAvailable: [twelveBand, true, true, true, true, true],
  };
}

test("book-depth codec preserves irregular second offsets and both exact schemas", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-book-depth-"));
  try {
    const snapshots = [snapshot(7, 10), snapshot(38, 12), snapshot(65, 10)];
    const store = new SequentialShardStore(path.join(root, "market", "immutable"));
    const stored = await putDerivativesBookDepthShard(store, {
      namespace: "derivatives-book-depth/usdm-futures/btcusdt",
      key: DAY,
      utcDayStartMs: START,
      snapshots,
    });
    assert.equal(stored.reference.layout.encoding, DERIVATIVES_BOOK_DEPTH_ENCODING);
    assert.deepEqual(stored.reference.sequence, {
      start: 0,
      step: 1,
      count: 3,
      unit: "index",
    });
    assert.deepEqual(stored.reference.layout.bandPercentages, DERIVATIVES_BOOK_DEPTH_BANDS);
    assert.deepEqual(
      (stored.reference.layout.columns as Array<{ name: string }>).map(({ name }) => name),
      [
        "timestampOffsetSeconds",
        "schemaBandCount",
        ...DERIVATIVES_BOOK_DEPTH_MATRIX_COLUMNS,
        "bandAvailable",
      ],
    );
    assert.deepEqual(
      decodeDerivativesBookDepth(
        stored.reference,
        await store.readPayload(stored.reference),
      ),
      snapshots,
    );
    assert.deepEqual(
      await readDerivativesBookDepthShardReference(stored.referenceFile),
      snapshots,
    );

    const layout = new TradingStorageLayout(root);
    assert.equal(
      layout.derivativesBookDepthReferences("usdm-futures", "btcusdt"),
      path.join(
        root,
        "market",
        "immutable",
        "refs",
        "derivatives-book-depth",
        "usdm-futures",
        "btcusdt",
      ),
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("book-depth encoder rejects timing, schema, availability, and value corruption", () => {
  assert.throws(
    () => encodeDerivativesBookDepth([snapshot(1), snapshot(1)], START),
    /timestamps are not increasing/,
  );
  assert.throws(
    () => encodeDerivativesBookDepth([snapshot(86_400)], START),
    /timestamp offset is invalid/,
  );
  assert.throws(
    () => encodeDerivativesBookDepth([{
      ...snapshot(1, 10),
      schemaBandCount: 11 as 10,
    }], START),
    /schema is invalid/,
  );
  assert.throws(
    () => encodeDerivativesBookDepth([{
      ...snapshot(1, 10),
      bandAvailable: [true, true, true, true, true, true],
    }], START),
    /availability schema differs/,
  );
  assert.throws(
    () => encodeDerivativesBookDepth([{
      ...snapshot(1, 10),
      bidDepth: [1, 11, 12, 13, 14, 15],
    }], START),
    /unavailable bidDepth has data/,
  );
  assert.throws(
    () => encodeDerivativesBookDepth([{
      ...snapshot(1),
      askDepth: [0, 21, 22, 23, 24, 25],
    }], START),
    /missing or nonpositive/,
  );
  assert.throws(
    () => encodeDerivativesBookDepth([{
      ...snapshot(1),
      bidNotional: [1_000, 999, 1_002, 1_003, 1_004, 1_005],
    }], START),
    /non-monotone/,
  );
  assert.throws(
    () => encodeDerivativesBookDepth([snapshot(1)], START + 1_000),
    /UTC day start is invalid/,
  );
});

test("book-depth decoder rejects masks, schema codes, and absent-band payload values", () => {
  const encoded = encodeDerivativesBookDepth([snapshot(1, 10)], START);
  const reference = {
    version: 1,
    kind: "trading-sequential-shard",
    namespace: "derivatives-book-depth/usdm-futures/btcusdt",
    key: DAY,
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
  const columns = new Map(encoded.layout.columns.map((column) => [column.name, column]));

  const invalidAvailability = Buffer.from(encoded.payload);
  invalidAvailability.writeUInt8(1, columns.get("bandAvailable")!.offset);
  assert.throws(
    () => decodeDerivativesBookDepth(reference, invalidAvailability),
    /availability schema differs/,
  );

  const invalidSchema = Buffer.from(encoded.payload);
  invalidSchema.writeUInt8(11, columns.get("schemaBandCount")!.offset);
  assert.throws(
    () => decodeDerivativesBookDepth(reference, invalidSchema),
    /schema code is invalid/,
  );

  const fabricatedAbsentValue = Buffer.from(encoded.payload);
  fabricatedAbsentValue.writeDoubleLE(1, columns.get("bidDepth")!.offset);
  assert.throws(
    () => decodeDerivativesBookDepth(reference, fabricatedAbsentValue),
    /unavailable bidDepth is nonzero/,
  );
});
