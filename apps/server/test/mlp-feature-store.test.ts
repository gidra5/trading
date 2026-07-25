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
    const store = new MlpFeatureStore(dataDir);
    const prepared = await store.prepare(seconds, [time]);
    const row = new Float32Array(MLP_INPUT_FEATURE_COUNT);
    prepared.encode(time, row, 0);

    assert.equal(row.length, 901);
    assert.ok(Math.abs(row[508]! - Math.log(130 / 100)) < 1e-6);
    assert.ok(Math.abs(row[512]! - 30 / 60) < 1e-7);
    assert.ok(Math.abs(row[641]! - 30 / 3_600) < 1e-7);
    assert.ok(Math.abs(row[770]! - 30 / 86_400) < 1e-7);
    assert.ok(Math.abs(row[835]! - 30 / (31 * 86_400)) < 1e-7);
    assert.ok(Math.abs(row[900]! - 30 / (90 * 86_400)) < 1e-7);
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

test("completed minute loading reconstructs a missing archive day from seconds", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "mlp-minute-recovery-"));
  const start = Date.parse("2026-01-02T00:00:00.000Z");
  const historicalRoot = path.join(
    dataDir,
    "historical",
    "spot-btcusdt",
    "btcusdt",
  );
  const minuteRoot = path.join(historicalRoot, "1m");
  const secondRoot = path.join(historicalRoot, "1s");
  await Promise.all([
    mkdir(minuteRoot, { recursive: true }),
    mkdir(secondRoot, { recursive: true }),
  ]);
  const seconds = Array.from({ length: 60 }, (_, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: start + index * 1_000,
    closeTime: start + (index + 1) * 1_000 - 1,
    open: 100 + index,
    high: 102 + index,
    low: 99 + index,
    close: 101 + index,
    volume: index + 1,
    closed: true,
  }));
  await writeFile(
    path.join(secondRoot, "2026-01-02.jsonl"),
    `${seconds.map((candle) => JSON.stringify(candle)).join("\n")}\n`,
  );
  try {
    const store = new MlpFeatureStore(dataDir);
    const [minute] = await store.loadCompletedMinuteRange(
      start,
      start + MINUTE_MS,
    );
    assert.ok(minute);
    assert.equal(minute.openTime, start);
    assert.equal(minute.closeTime, start + MINUTE_MS - 1);
    assert.equal(minute.open, 100);
    assert.equal(minute.close, 160);
    assert.equal(minute.high, 161);
    assert.equal(minute.low, 99);
    assert.equal(minute.volume, 1_830);
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});
