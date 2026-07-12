import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import type { Candle, VwKamaInspectorRequest } from "@trading/bot-algo";
import { KamaInspectorEngine } from "../src/kama-inspector.js";

const WINDOW_ID = "shape-up-low-2024-02";
const START = Date.parse("2024-02-24T00:00:00.000Z");

test("KAMA inspector serves truthful viewport candle resolutions", async () => {
  const dataDir = await fixture();
  try {
    const engine = new KamaInspectorEngine(dataDir);
    const raw = await engine.candles({
      ...analysisRequest(),
      startTime: START,
      endTime: START + 5_000,
      maxCandles: 100,
    });
    assert.equal(raw.renderIntervalMs, 1_000);
    assert.equal(raw.sourceCandleCount, 5);
    assert.equal(raw.candles.length, 5);
    assert.equal(raw.kamaSeries.points.length, raw.candles.length);
    assert.deepEqual(
      raw.kamaSeries.points.map((point) => point.time),
      raw.candles.map((candle) => candle.closeTime),
    );
    assert.ok(raw.candles.every((candle) =>
      candle.interval === "1s" && candle.closeTime - candle.openTime === 999));

    const wide = await engine.candles({
      ...analysisRequest(),
      startTime: START,
      endTime: START + 200_000,
      maxCandles: 100,
    });
    assert.equal(wide.renderIntervalMs, 2_000);
    assert.equal(wide.sourceCandleCount, 199);
    assert.equal(wide.candles.length, 100);
    assert.equal(wide.kamaSeries.points.length, wide.candles.length);
    assert.deepEqual(
      wide.kamaSeries.points.map((point) => point.time),
      wide.candles.map((candle) => candle.closeTime),
    );
    assert.ok(wide.candles.every((candle) =>
      candle.closeTime - candle.openTime + 1 <= wide.renderIntervalMs
      && candle.interval === intervalLabel(candle.closeTime - candle.openTime + 1)));
    assert.equal(wide.candles.some((candle) =>
      candle.openTime === START + 121_000
      && candle.closeTime === START + 121_999
      && candle.interval === "1s"), true);
    assert.deepEqual(
      pick(wide.candles[0]!),
      { open: 99.75, high: 102, low: 99, close: 101.25, volume: 3 },
    );
    assert.equal(wide.candles.some((candle) => candle.openTime === START + 120_000), false);

    const unaligned = await engine.candles({
      ...analysisRequest(),
      startTime: START + 1_000,
      endTime: START + 200_000,
      maxCandles: 100,
    });
    assert.equal(unaligned.sourceCandleCount, 198);
    assert.equal(unaligned.candles[0]!.openTime, START + 1_000);
    assert.equal(unaligned.candles.at(-1)!.closeTime, START + 199_999);

    const overlap = await engine.candles({
      ...analysisRequest(),
      startTime: START + 10_000,
      endTime: START + 210_000,
      maxCandles: 100,
    });
    const overlapping = overlap.candles.filter((candle) =>
      candle.openTime >= START + 10_000 && candle.openTime < START + 200_000);
    assert.deepEqual(
      overlapping,
      wide.candles.filter((candle) =>
        candle.openTime >= START + 10_000 && candle.openTime < START + 200_000),
    );
    const overlapKama = new Map(overlap.kamaSeries.points.map((point) => [point.time, point.value]));
    for (const point of wide.kamaSeries.points) {
      if (point.time >= START + 10_000 && point.time < START + 200_000) {
        assert.equal(overlapKama.get(point.time), point.value);
      }
    }

    const clamped = await engine.candles({
      ...analysisRequest(),
      startTime: START - 5_000,
      endTime: START + 2_000,
      maxCandles: 100,
    });
    assert.equal(clamped.startTime, START);
    assert.equal(clamped.sourceCandleCount, 2);
    assert.ok(clamped.candles.every((candle) => candle.openTime >= START));

    const analysis = await engine.analyze(analysisRequest());
    assert.equal(analysis.renderIntervalMs, 2_000);
    assert.ok(analysis.candles.length <= 2_000);
    assert.ok(analysis.candles.every((candle) =>
      candle.closeTime - candle.openTime + 1 <= analysis.renderIntervalMs
      && candle.interval === intervalLabel(candle.closeTime - candle.openTime + 1)));
    const wideKama = new Map(wide.kamaSeries.points.map((point) => [point.time, point.value]));
    let compared = 0;
    for (const point of analysis.kamaSeries.points) {
      if (point.time < START + 200_000 && wideKama.has(point.time)) {
        assert.equal(wideKama.get(point.time), point.value);
        compared += 1;
      }
    }
    assert.ok(compared > 50);
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

test("KAMA inspector catalogs global and generated per-window presets", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "kama-presets-"));
  try {
    await mkdir(path.join(dataDir, "benchmarks"), { recursive: true });
    await writeFile(path.join(dataDir, "benchmarks", "vw-kama-window-presets.json"), JSON.stringify([{
      id: "window-shape-up-low-2024-02",
      label: "generated label",
      scope: "window",
      windowId: WINDOW_ID,
      intervalMs: 1_000,
      parameters: analysisRequest().parameters,
      score: 0.5,
    }]));
    const catalog = new KamaInspectorEngine(dataDir).catalog();
    assert.ok(catalog.presets.some((preset) => preset.scope === "global"));
    const local = catalog.presets.find((preset) => preset.windowId === WINDOW_ID);
    assert.equal(local?.label, "Window best found · Uptrend · low churn · 1s");
    assert.equal(local?.score, 0.5);
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

async function fixture(): Promise<string> {
  const dataDir = await mkdtemp(path.join(tmpdir(), "kama-inspector-"));
  const root = path.join(dataDir, "historical", "spot-btcusdt", "btcusdt", "1s");
  await mkdir(root, { recursive: true });
  const dates = ["2024-02-21", "2024-02-22", "2024-02-23", "2024-02-24", "2024-02-25", "2024-02-26"];
  const byDate = new Map(dates.map((date) => [date, [] as Candle[]]));
  byDate.get("2024-02-23")!.push(candle(-1));
  for (let index = 0; index < 2_400; index += 1) {
    if (index !== 120) byDate.get("2024-02-24")!.push(candle(index));
  }
  await Promise.all(dates.map((date) => writeFile(
    path.join(root, `${date}.jsonl`),
    byDate.get(date)!.map((value) => JSON.stringify(value)).join("\n"),
  )));
  return dataDir;
}

function candle(index: number): Candle {
  const price = 100 + index;
  const openTime = START + index * 1_000;
  return {
    symbol: "BTCUSDT",
    interval: "1s",
    openTime,
    closeTime: openTime + 999,
    open: price - 0.25,
    high: price + 1,
    low: price - 1,
    close: price + 0.25,
    volume: index + 1,
    closed: true,
  };
}

function pick(value: Candle): Pick<Candle, "open" | "high" | "low" | "close" | "volume"> {
  return {
    open: value.open,
    high: value.high,
    low: value.low,
    close: value.close,
    volume: value.volume,
  };
}

function intervalLabel(intervalMs: number): string {
  return intervalMs % 60_000 === 0 ? `${intervalMs / 60_000}m` : `${intervalMs / 1_000}s`;
}

function analysisRequest(): VwKamaInspectorRequest {
  return {
    windowId: WINDOW_ID,
    intervalMs: 1_000,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 2_000,
      slowMs: 5_000,
      power: 1,
      volumeMs: 2_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
    },
    oracleFriction: 0.001,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  };
}
