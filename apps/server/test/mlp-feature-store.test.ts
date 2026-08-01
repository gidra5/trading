import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import {
  MLP_INPUT_FEATURE_COUNT,
  encodeMlpCandleWindow,
  type Candle,
} from "@trading/bot-algo";
import { MlpFeatureStore } from "../src/mlp-feature-store.js";
import { writeCanonicalCandleDay } from "./canonical-storage-fixture.js";

const MINUTE_MS = 60_000;

test("MLP features end every coarse window in a causal partial candle", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "mlp-features-"));
  const start = Date.parse("2026-01-01T00:00:00.000Z");
  await writeCanonicalCandleDay(dataDir, [{
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
  } satisfies Candle], { stepMs: MINUTE_MS });
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
  await writeCanonicalCandleDay(dataDir, seconds, { stepMs: 1_000 });
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

test("deployment warmup depth does not change canonical MLP features", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "mlp-deployment-warmup-"));
  const currentDay = Date.parse("2026-01-01T00:00:00.000Z");
  const previousDay = currentDay - 86_400_000;
  const extraDay = previousDay - 86_400_000;
  const minute = (openTime: number, price: number): Candle => ({
    symbol: "BTCUSDT",
    interval: "1m",
    openTime,
    closeTime: openTime + MINUTE_MS - 1,
    open: price,
    high: price + 1,
    low: price - 1,
    close: price + 0.5,
    volume: 60,
    closed: true,
  });
  await Promise.all([
    writeCanonicalCandleDay(dataDir, [minute(extraDay, 90)], { stepMs: MINUTE_MS }),
    writeCanonicalCandleDay(dataDir, [minute(previousDay, 95)], { stepMs: MINUTE_MS }),
  ]);
  const seconds = (start: number, count: number, price: number) =>
    Array.from({ length: count }, (_, index): Candle => ({
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: start + index * 1_000,
      closeTime: start + (index + 1) * 1_000 - 1,
      open: price + index * 0.01,
      high: price + index * 0.01 + 0.02,
      low: price + index * 0.01 - 0.02,
      close: price + index * 0.01 + 0.01,
      volume: 1 + index % 5,
      closed: true,
    }));
  const previous = seconds(previousDay, 60, 95);
  const current = seconds(currentDay, 30, 100);
  const extra = seconds(extraDay, 60, 90);
  const time = current.at(-1)!.closeTime;
  try {
    const store = new MlpFeatureStore(dataDir);
    const [shortWarmup, deploymentWarmup] = await Promise.all([
      store.prepare([...previous, ...current], [time]),
      store.prepare([...extra, ...previous, ...current], [time]),
    ]);
    const shortRow = new Float32Array(MLP_INPUT_FEATURE_COUNT);
    const deploymentRow = new Float32Array(MLP_INPUT_FEATURE_COUNT);
    shortWarmup.encode(time, shortRow, 0);
    deploymentWarmup.encode(time, deploymentRow, 0);
    assert.deepEqual(deploymentRow, shortRow);
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

test("cumulative feature queries preserve the reference EMA encoding", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "mlp-cumulative-features-"));
  const start = Date.parse("2026-01-01T00:00:00.000Z");
  const archivedMinute: Candle = {
    symbol: "BTCUSDT",
    interval: "1m",
    openTime: start,
    closeTime: start + MINUTE_MS - 1,
    open: 100,
    high: 101,
    low: 99,
    close: 100.5,
    volume: 60,
    closed: true,
  };
  await writeCanonicalCandleDay(dataDir, [archivedMinute], { stepMs: MINUTE_MS });
  const seconds = Array.from({ length: 300 }, (_, index): Candle => {
    const open = 100 + index * 0.01;
    const close = open + (index % 7 - 3) * 0.001;
    return {
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: start + index * 1_000,
      closeTime: start + (index + 1) * 1_000 - 1,
      open,
      high: Math.max(open, close) + 0.02,
      low: Math.min(open, close) - 0.02,
      close,
      volume: 0.1 + (index * 17 % 43) / 10,
      closed: true,
    };
  });
  try {
    const time = seconds.at(-1)!.closeTime;
    const prepared = await new MlpFeatureStore(dataDir).prepare(seconds, [time]);
    const row = new Float32Array(MLP_INPUT_FEATURE_COUNT);
    prepared.encode(time, row, 0);
    const reference = encodeMlpCandleWindow(seconds, 64);
    assert.deepEqual(row.slice(0, reference.length), reference);
    const half = new Uint16Array(MLP_INPUT_FEATURE_COUNT);
    prepared.encodeHalf(time, half, 0);
    assert.deepEqual(
      half,
      Uint16Array.from(row, float32ToFloat16Bits),
    );
    const batchTimes = seconds.slice(-60).map((candle) => candle.closeTime);
    const batchPrepared = await new MlpFeatureStore(dataDir).prepare(
      seconds,
      batchTimes,
    );
    if (batchPrepared.encodeHalfRows) {
      const batch = new Uint16Array(
        batchTimes.length * MLP_INPUT_FEATURE_COUNT,
      );
      const rowWise = new Uint16Array(batch.length);
      batchPrepared.encodeHalfRows(batchTimes, batch);
      const rowPrepared = await new MlpFeatureStore(dataDir).prepare(
        seconds,
        batchTimes,
      );
      for (let index = 0; index < batchTimes.length; index += 1) {
        rowPrepared.encodeHalf(
          batchTimes[index]!,
          rowWise,
          index * MLP_INPUT_FEATURE_COUNT,
        );
      }
      assert.deepEqual(batch, rowWise);
    }
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

const FLOAT32_TO_FLOAT16_SCRATCH = new Float32Array(1);
const FLOAT32_TO_FLOAT16_BITS = new Uint32Array(
  FLOAT32_TO_FLOAT16_SCRATCH.buffer,
);

function float32ToFloat16Bits(value: number): number {
  FLOAT32_TO_FLOAT16_SCRATCH[0] = value;
  const raw = FLOAT32_TO_FLOAT16_BITS[0]!;
  const sign = raw >>> 16 & 0x8000;
  let exponent = (raw >>> 23 & 0xff) - 127 + 15;
  let mantissa = raw & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | (mantissa + 0x1000 >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000;
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
  }
  return exponent >= 31
    ? sign | 0x7c00
    : sign | exponent << 10 | mantissa >>> 13;
}
