import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { gzipSync } from "node:zlib";
import { VW_KAMA_SCORE_VERSION, type Candle, type VwKamaInspectorRequest } from "@trading/bot-algo";
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
    assert.equal(raw.indicatorPoints.length, raw.candles.length);
    assert.equal(raw.valueDistributions.length, raw.candles.length);
    assert.ok(raw.valueDistributions.every((point) =>
      Math.abs(point.values.reduce((sum, value) => sum + value.oracleProbability, 0) - 1) < 1e-6
      && Math.abs(point.values.reduce((sum, value) => sum + value.strategyProbability, 0) - 1) < 1e-9));
    assert.deepEqual(
      raw.kamaSeries.points.map((point) => point.time),
      raw.candles.map((candle) => candle.closeTime),
    );
    assert.deepEqual(
      raw.indicatorPoints.map((point) => point.time),
      raw.candles.map((candle) => candle.closeTime),
    );
    assert.ok(raw.indicatorPoints.every((point) =>
      Number.isFinite(point.threshold)
      && Number.isFinite(point.kamaRate)
      && Number.isFinite(point.kamaRateRaw)
      && Number.isFinite(point.volume)
      && Number.isFinite(point.volumeAverage)
      && Array.isArray(point.rejectionReasons)));
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
    assert.equal(wide.indicatorPoints.length, wide.candles.length);
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
    assert.ok(analysis.valueDistributions.length > 0);
    assert.ok(analysis.metrics.valueDistillation);
    assert.equal(analysis.metrics.valueDistillation!.holdingPeriodMs, 1_000);
    assert.ok(analysis.metrics.valueDistillation!.valueHorizonMs > 1_000);
    assert.ok(Number.isFinite(analysis.metrics.valueDistillation!.returns.strategy.totalReturn));
    assert.ok(Number.isFinite(analysis.metrics.valueDistillation!.returns.oracle.totalReturn));
    assert.equal(analysis.indicatorPoints.length, analysis.kamaSeries.points.length);
    assert.deepEqual(
      analysis.indicatorPoints.map((point) => point.time),
      analysis.kamaSeries.points.map((point) => point.time),
    );
    assert.ok(analysis.indicatorPoints.every((point) =>
      Number.isFinite(point.confirmationEma)
      && Number.isFinite(point.threshold)
      && Number.isFinite(point.kamaRate)
      && Number.isFinite(point.kamaRateRaw)
      && Number.isFinite(point.volume)
      && Number.isFinite(point.volumeAverage)
      && Array.isArray(point.rejectionReasons)
      && Number.isFinite(point.rsi)
      && Number.isFinite(point.dmi)
      && Number.isFinite(point.adx)));
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

test("KAMA inspector catalogs generated global and per-window presets", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "kama-presets-"));
  try {
    await mkdir(path.join(dataDir, "benchmarks"), { recursive: true });
    await writeFile(path.join(dataDir, "benchmarks", "vw-kama-global-presets.json"), JSON.stringify([{
      id: "global-score-v3-volume",
      label: "Global · score-v3 volume finalist",
      scope: "global",
      windowId: null,
      parameters: analysisRequest().parameters,
      score: 0.6,
      scoreVersion: VW_KAMA_SCORE_VERSION,
      source: "Validation-selected global finalist",
    }]));
    await writeFile(path.join(dataDir, "benchmarks", "vw-kama-window-presets.json"), JSON.stringify([{
      id: "window-shape-up-low-2024-02",
      label: "generated label",
      scope: "window",
      windowId: WINDOW_ID,
      intervalMs: 1_000,
      parameters: analysisRequest().parameters,
      score: 0.5,
      scoreVersion: VW_KAMA_SCORE_VERSION,
      generatedAt: "2026-07-13T17:15:04.339Z",
      incumbentScore: 0.45,
      optimization: {
        algorithm: "de",
        population: 2_048,
        generations: 64,
        restarts: 2,
        refinementRounds: 5,
        elapsedMs: 1_000,
        hindsight: true,
      },
    }, {
      id: "window-shape-up-low-2024-02-old-score",
      label: "historical label",
      scope: "window",
      windowId: WINDOW_ID,
      intervalMs: 1_000,
      parameters: analysisRequest().parameters,
      score: 0.4,
      scoreVersion: VW_KAMA_SCORE_VERSION - 1,
    }, {
      id: "window-fit-1-1m",
      label: "optimizer fit label",
      scope: "window",
      windowId: "fit-1",
      intervalMs: 60_000,
      parameters: analysisRequest().parameters,
      score: -5.01,
      scoreVersion: VW_KAMA_SCORE_VERSION,
      optimization: {
        algorithm: "de",
        objective: "value-distillation",
        population: 384,
        generations: 12,
        restarts: 2,
        refinementRounds: 3,
        elapsedMs: 1_000,
        hindsight: true,
      },
    }]));
    const engine = new KamaInspectorEngine(dataDir);
    const catalog = engine.catalog();
    assert.equal(catalog.defaults.intervalMs, 1_000);
    assert.equal(catalog.windows.find((window) => window.id === "fit-1")?.label, "Optimizer fit 1");
    const fullFit = catalog.windows.find((window) => window.id === "fit-full");
    assert.equal(fullFit?.label, "Optimizer fit · full");
    assert.equal(fullFit?.startTime, Date.parse("2025-03-19T00:00:00.000Z"));
    assert.equal(fullFit?.endTime, Date.parse("2025-11-14T00:00:00.000Z"));
    assert.equal(fullFit?.sourceIntervalMs, 1_000);
    const global = catalog.presets.find((preset) => preset.id === "global-score-v3-volume");
    assert.equal(global?.score, 0.6);
    assert.equal(global?.source, "Validation-selected global finalist");
    const local = catalog.presets.find((preset) => preset.windowId === WINDOW_ID);
    assert.equal(local?.label, "Window best found · Uptrend · low churn · 1s");
    assert.equal(local?.score, 0.5);
    assert.equal(local?.incumbentScore, 0.45);
    assert.equal(local?.optimization?.population, 2_048);
    const historical = catalog.presets.find((preset) => preset.id.endsWith("old-score"));
    assert.equal(historical?.score, undefined);
    assert.equal(historical?.historicalScore, 0.4);
    const fit = catalog.presets.find((preset) => preset.id === "window-fit-1-1m");
    assert.equal(fit?.label, "Window best found · Optimizer fit 1 · 1m");
    assert.equal(fit?.optimization?.objective, "value-distillation");
    await assert.rejects(
      engine.analyze({
        ...analysisRequest(),
        windowId: "fit-1",
        intervalMs: 60_000,
        parameters: {
          ...analysisRequest().parameters,
          slowMs: 86_400_000.00000006,
        },
        warmupMultiple: 3,
      }),
      /Missing BTCUSDT 1s shard/,
    );
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
  await Promise.all(dates.map((date, index) => {
    const content = byDate.get(date)!.map((value) => JSON.stringify(value)).join("\n");
    return index === 0
      ? writeFile(path.join(root, `${date}.jsonl.gz`), gzipSync(content))
      : writeFile(path.join(root, `${date}.jsonl`), content);
  }));
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
