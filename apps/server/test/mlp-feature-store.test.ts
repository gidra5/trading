import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { MLP_INPUT_FEATURE_COUNT, type Candle } from "@trading/bot-algo";
import { MlpFeatureStore } from "../src/mlp-feature-store.js";

const MINUTE_MS = 60_000;

test("MLP features end every coarse window in a causal partial candle", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "mlp-features-"));
  const start = Date.parse("2026-01-01T00:00:00.000Z");
  const minuteRoot = path.join(
    dataDir,
    "historical",
    "spot-btcusdt",
    "btcusdt",
    "1m",
  );
  await mkdir(minuteRoot, { recursive: true });
  await writeFile(path.join(minuteRoot, "2026-01-01.jsonl"), `${JSON.stringify({
    symbol: "BTCUSDT",
    interval: "1m",
    openTime: start,
    closeTime: start + MINUTE_MS - 1,
    open: 100,
    high: 160,
    low: 100,
    close: 160,
    volume: 60,
    closed: true,
  } satisfies Candle)}\n`);
  const seconds = Array.from({ length: 30 }, (_, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: start + index * 1_000,
    closeTime: start + (index + 1) * 1_000 - 1,
    open: 100 + index,
    high: 101 + index,
    low: 99 + index,
    close: 101 + index,
    volume: 1,
    closed: true,
  }));
  try {
    const time = seconds.at(-1)!.closeTime;
    const prepared = await new MlpFeatureStore(dataDir).prepare(seconds, [time], {
      feeRate: 1,
      minimumUsableExposure: 2,
      maximumUsableExposure: 3,
      minimumEffectiveExposure: 4,
      maximumEffectiveExposure: 5,
      quoteLendRate: 6,
      quoteBorrowRate: 7,
      assetBorrowRate: 8,
    });
    const row = new Float32Array(MLP_INPUT_FEATURE_COUNT);
    prepared.encode(time, row, 0);

    assert.equal(row.length, 909);
    assert.ok(Math.abs(row[508]! - Math.log(130 / 100)) < 1e-6);
    assert.ok(Math.abs(row[512]! - 30 / 60) < 1e-7);
    assert.ok(Math.abs(row[641]! - 30 / 3_600) < 1e-7);
    assert.ok(Math.abs(row[770]! - 30 / 86_400) < 1e-7);
    assert.ok(Math.abs(row[835]! - 30 / (31 * 86_400)) < 1e-7);
    assert.ok(Math.abs(row[900]! - 30 / (90 * 86_400)) < 1e-7);
    assert.deepEqual(Array.from(row.slice(901)), [1, 2, 3, 4, 5, 6, 7, 8]);
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});
