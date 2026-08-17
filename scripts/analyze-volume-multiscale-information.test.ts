import assert from "node:assert/strict";
import test from "node:test";
import type { SequentialCandle } from "@trading/storage";
import {
  buildMarketFeatureDefinitions,
  CausalMarketFeatureEngine,
} from "./analyze-volume-multiscale-information.ts";

function candle(index: number, close = 100 + index, volume = 1): SequentialCandle {
  return {
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: close - 0.5,
    high: close + 1,
    low: close - 1,
    close,
    volume,
    closed: true,
  };
}

test("market feature definitions are unique and aligned to the engine output", () => {
  const definitions = buildMarketFeatureDefinitions();
  assert.equal(definitions.length, 95);
  assert.equal(new Set(definitions.map((definition) => definition.id)).size, definitions.length);
  definitions.forEach((definition, index) => assert.equal(definition.sourceIndex, index));
  const engine = new CausalMarketFeatureEngine();
  assert.equal(engine.values().length, definitions.length);
  assert.equal(engine.valid().length, definitions.length);
});

test("a higher-timeframe feature changes only after the full aligned bar closes", () => {
  const definitions = buildMarketFeatureDefinitions();
  const returnIndex = definitions.find((definition) => definition.id === "5s-return")!.sourceIndex;
  const volumeIndex = definitions.find(
    (definition) => definition.id === "5s-log-volume",
  )!.sourceIndex;
  const engine = new CausalMarketFeatureEngine();
  for (let index = 0; index < 4; index += 1) engine.update(candle(index));
  assert.equal(engine.values()[returnIndex], 0);
  assert.equal(engine.values()[volumeIndex], 0);
  engine.update(candle(4));
  const firstVolume = engine.values()[volumeIndex]!;
  assert.ok(firstVolume > 0);
  assert.equal(engine.values()[returnIndex], 0);
  for (let index = 5; index < 9; index += 1) engine.update(candle(index, 200 + index, 10));
  assert.equal(engine.values()[volumeIndex], firstVolume);
  assert.equal(engine.values()[returnIndex], 0);
  engine.update(candle(9, 209, 10));
  assert.ok(engine.values()[volumeIndex]! > firstVolume);
  assert.ok(engine.values()[returnIndex]! > 0);
});

test("completed-bar inputs become valid only after the common causal warmup", () => {
  const definitions = buildMarketFeatureDefinitions();
  const fiveSecond = definitions.filter((definition) => definition.resolution === "5s");
  const engine = new CausalMarketFeatureEngine();
  for (let index = 0; index < 64 * 5; index += 1) engine.update(candle(index));
  for (const definition of fiveSecond) assert.equal(engine.valid()[definition.sourceIndex], 0);
  for (let index = 64 * 5; index < 65 * 5; index += 1) engine.update(candle(index));
  for (const definition of fiveSecond) assert.equal(engine.valid()[definition.sourceIndex], 1);
});

test("one-second features contain only the latest completed candle", () => {
  const definitions = buildMarketFeatureDefinitions();
  const volumeIndex = definitions.find(
    (definition) => definition.id === "1s-log-volume",
  )!.sourceIndex;
  const locationIndex = definitions.find(
    (definition) => definition.id === "1s-close-location",
  )!.sourceIndex;
  const engine = new CausalMarketFeatureEngine();
  engine.update(candle(0, 100, 3));
  assert.equal(engine.values()[volumeIndex], Math.log1p(3));
  assert.equal(engine.values()[locationIndex], 0);
  assert.equal(engine.valid()[volumeIndex], 1);
});
