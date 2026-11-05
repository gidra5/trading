import assert from "node:assert/strict";
import test from "node:test";
import { budgetStreams, gapClippedMilliseconds, MONTH_SECONDS, rateClass, unionMilliseconds } from "./collector-storage.ts";
import { buildEstimate } from "../estimate-collector-storage.ts";

test("a 30-day month and simultaneous spot/futures budgets retain their units", () => {
  assert.equal(MONTH_SECONDS, 2_592_000);
  const streams = ["spot", "futures"].map((source) => ({ source, recordsPerSecond: 2,
    uncompressedBytesPerRecord: 200, compressedBytesPerRecord: 50, evidence: "test" }));
  const monthly = budgetStreams(streams);
  assert.equal(monthly.compressedBytes, 518_400_000);
  assert.equal(monthly.uncompressedBytes, 2_073_600_000);
  assert.equal(budgetStreams(streams, MONTH_SECONDS, 3).compressedBytes, 3 * monthly.compressedBytes);
});

test("coverage unions overlapping sessions and excludes capture gaps", () => {
  assert.equal(unionMilliseconds([{ startMs: 1000, endMs: 3000 }, { startMs: 2000, endMs: 4000 }, { startMs: 9000, endMs: 10_000 }]), 4000);
  assert.equal(gapClippedMilliseconds([0, 1000, 2000, 100_000, 101_000], 2000, 1000), 4000);
  assert.throws(() => gapClippedMilliseconds([3, 2], 1000));
  assert.throws(() => unionMilliseconds([{ startMs: 10, endMs: 2 }]));
});

test("sparse markets are not assigned the BTC traffic class", () => {
  assert.deepEqual([0, .099, .1, 1, 10].map(rateClass), ["quiet", "quiet", "medium", "busy", "very-busy"]);
  assert.throws(() => rateClass(-1));
  assert.throws(() => budgetStreams([{ source: "bad", recordsPerSecond: Number.NaN, uncompressedBytesPerRecord: 1, compressedBytesPerRecord: 1, evidence: "test" }]));
});

test("universe route identity survives ticker collisions and repeated references", () => {
  const entries: Parameters<typeof buildEstimate>[0] = [
    { assetId: "crypto:CAT", baseAsset: "CAT", productSymbols: { spot: ["1000CATUSDT"] } },
    { assetId: "tradfi:CAT", baseAsset: "CAT", productSymbols: { "usdm-futures": ["CATUSDT"] } },
    { assetId: "crypto:duplicate-reference", baseAsset: "CAT", productSymbols: { spot: ["1000CATUSDT"] } },
  ];
  const inputs = { generatedAt: "test", measurements: [], samples: [], snapshots: [], measuredAt: {}, tickers: {
    spot: [{ symbol: "1000CATUSDT", count: 86, openTime: 0, closeTime: 86_400_000, quoteVolume: 1 }],
    "usdm-futures": [{ symbol: "CATUSDT", count: 0, openTime: 0, closeTime: 86_400_000, quoteVolume: 0 }],
  } };
  const result = buildEstimate(entries, inputs);
  assert.equal(result.routeCount, 2);
  assert.equal(result.routeCounts.tradfi, 1);
  assert.equal(result.routes[1]!.planningFillsPerSecond, .5); // Sunday zero is not a zero-storage month.
  assert.equal(result.routes[1]!.depthRate, 5);
  assert.ok(result.profiles["all-routes-full-public-replay"].base.compressedBytes > result.profiles["all-routes-feature-only-lossy"].base.compressedBytes);
  const fullRows = result.profiles["all-routes-full-public-replay"].base.rows;
  assert.ok(!fullRows.some((row) => row.source === "spot:1000CATUSDT/depth-bbo-derived"));
  assert.ok(result.profiles["all-routes-full-plus-derived"].base.rows.some((row) => row.source === "spot:1000CATUSDT/depth-bbo-derived"));
  assert.ok(fullRows.some((row) => row.source === "usdm-futures:CATUSDT/mark-price-funding"));
});

test("actual full-recorder calibration replaces fallback envelope and derived-row sizes", () => {
  const inputs = { generatedAt: "test", measurements: [], snapshots: [], measuredAt: {}, tickers: {
    spot: [{ symbol: "BTCUSDT", count: 864_000, openTime: 0, closeTime: 86_400_000, quoteVolume: 1 }],
  }, samples: [{ venue: "spot", symbol: "BTCUSDT", stream: "btcusdt@depth@100ms", rateClass: "very-busy" as const,
    messages: 10, observedSeconds: 1, rawBytes: 1000, gzipOneSecondBytes: 500, gzipSixtySecondBytes: 400, individualFills: 0, failure: null }],
  fullCapture: { directory: "fixture", wireMessages: 10, actualGzipBytesIncludingBootstrapAndPeripherals: 9999, groups: [
    { venue: "spot", kind: "depth@100ms", rows: 10, gzip1s: 1000, rawBytes: 3000,
      legacyGzip1s: 500, legacyRawBytes: 1000, compressedRatio: 2, rawRatio: 3 },
    { venue: "spot", kind: "depth_bbo", rows: 10, gzip1s: 800, rawBytes: 5000,
      legacyGzip1s: null, legacyRawBytes: null, compressedRatio: null, rawRatio: null },
  ] } };
  const result = buildEstimate([{ assetId: "crypto:BTC", baseAsset: "BTC", productSymbols: { spot: ["BTCUSDT"] } }], inputs);
  const raw = result.profiles["all-routes-full-public-replay"].base.rows.find((row) => row.source.endsWith("/depth-diff"))!;
  assert.equal(raw.compressedBytesPerRecord, 100); // 50 measured legacy bytes * actual ratio 2; not fallback 1.5.
  assert.equal(raw.uncompressedBytesPerRecord, 300);
  const derived = result.profiles["all-routes-full-plus-derived"].base.rows.find((row) => row.source.endsWith("/depth-bbo-derived"))!;
  assert.equal(derived.compressedBytesPerRecord, 80); // Actual 800/10; not the fallback 250 bytes.
  assert.equal(derived.uncompressedBytesPerRecord, 500);
});
