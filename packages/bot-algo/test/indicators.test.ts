import assert from "node:assert/strict";
import test from "node:test";
import {
  EMAIndicator,
  KAMAIndicator,
  LinearRegressionIndicator,
  LookbackIndicator,
  SMAIndicator,
  VolumeWeightedKAMAIndicator,
} from "../src/indicators.js";
import type { TradingApi } from "../src/trading-api.js";

const unusedApi = {} as TradingApi;

test("lookback exposes the value at the configured distance", () => {
  const indicator = new LookbackIndicator(2);
  for (const [eventTime, value] of [1, 2, 4, 8].map((value, index) => [index, value])) {
    indicator.onTick({ eventTime, value });
  }

  assert.equal(indicator.indicator(), 8);
  assert.equal(indicator.previous(), 2);
  assert.equal(indicator.derivative(), 6);

  const restored = new LookbackIndicator(2);
  restored.restore(indicator.snapshot());
  assert.deepEqual(restored.snapshot(), indicator.snapshot());
});

test("numeric moving averages preserve zero and negative values", () => {
  const sma = new SMAIndicator(3, unusedApi);
  sma.onTick({ eventTime: 1, value: -1 });
  sma.onTick({ eventTime: 2, value: 0 });
  sma.onTick({ eventTime: 3, value: 2 });

  assert.equal(sma.indicator(), 1 / 3);
  assert.deepEqual(sma.snapshot().values, [-1, 0, 2]);

  const restored = new SMAIndicator(3, unusedApi);
  restored.restore(structuredClone(sma.snapshot()));
  assert.equal(restored.indicator(), sma.indicator());
});

test("EMA accepts signed child-indicator series", () => {
  const ema = new EMAIndicator(1);
  ema.onTick({ eventTime: 1, value: -2 });
  ema.onTick({ eventTime: 2, value: -4 });

  assert.equal(ema.indicator(), -4);
  assert.equal(ema.derivative(), -2);
});

test("linear regression retains a negative series across snapshot restore", () => {
  const regression = new LinearRegressionIndicator(4, unusedApi);
  for (let index = 0; index < 4; index += 1) {
    regression.onTick({ eventTime: index + 1, value: -2 * index });
  }

  assert.equal(regression.indicator().slope, -2);
  assert.equal(regression.predict(1), -8);

  const restored = new LinearRegressionIndicator(4, unusedApi);
  restored.restore(structuredClone(regression.snapshot()));
  assert.deepEqual(restored.indicator(), regression.indicator());
});

test("volume-weighted KAMA with both volume powers zero matches canonical KAMA", () => {
  const canonical = new KAMAIndicator(3, 2, 10, unusedApi, 10, 2);
  const weighted = new VolumeWeightedKAMAIndicator(unusedApi, {
    efficiencyPeriod: 3,
    efficiencyVolumePower: 0,
    fastPeriod: 2,
    slowPeriod: 10,
    power: 2,
    volumePeriod: 4,
    volumePower: 0,
  });
  for (const [index, price] of [100, 101, 99, 103, 102, 106].entries()) {
    const input = { eventTime: index, price, quantity: index + 1 };
    canonical.onTick(input);
    weighted.onTick(input);
    assert.ok(Math.abs(weighted.indicator() - canonical.indicator()) < 1e-12);
  }
});

test("volume-weighted KAMA speeds up on high relative volume", () => {
  const options = {
    efficiencyPeriod: 2,
    fastPeriod: 2,
    slowPeriod: 20,
    power: 1,
    volumePeriod: 3,
    volumeCap: 4,
    volumePower: 1,
  };
  const base = new VolumeWeightedKAMAIndicator(unusedApi, options);
  for (const [index, price] of [100, 101, 102].entries()) {
    base.onTick({ eventTime: index, price, quantity: 10 });
  }
  const high = new VolumeWeightedKAMAIndicator(unusedApi, options);
  const low = new VolumeWeightedKAMAIndicator(unusedApi, options);
  high.restore(base.snapshot());
  low.restore(base.snapshot());
  high.onTick({ eventTime: 4, price: 104, quantity: 40 });
  low.onTick({ eventTime: 4, price: 104, quantity: 2 });

  assert.equal(high.volumeAverage(), 25);
  assert.equal(low.volumeAverage(), 6);
  assert.ok(high.details().alpha > low.details().alpha);
  assert.ok(high.derivative() > low.derivative());
});

test("volume-weighted KAMA weights each efficiency-ratio move by relative volume", () => {
  const indicator = new VolumeWeightedKAMAIndicator(unusedApi, {
    efficiencyPeriod: 2,
    efficiencyVolumePeriod: 3,
    efficiencyVolumePower: 2,
    fastPeriod: 2,
    slowPeriod: 10,
    power: 1,
    volumePeriod: 3,
    volumePower: 0,
  });
  indicator.onTick({ eventTime: 1, price: 100, quantity: 10 });
  indicator.onTick({ eventTime: 2, price: 102, quantity: 20 });
  indicator.onTick({ eventTime: 3, price: 101, quantity: 5 });

  const up = 2 * Math.pow(20 / 15, 2);
  const down = -1 * Math.pow(5 / 10, 2);
  assert.ok(Math.abs(
    indicator.details().efficiencyRatio - Math.abs(up + down) / (Math.abs(up) + Math.abs(down))
  ) < 1e-12);
});

test("volume-weighted KAMA restores exactly and treats missing volume as neutral", () => {
  const options = {
    efficiencyPeriod: 2,
    efficiencyVolumePeriod: 3,
    efficiencyVolumePower: 1.5,
    fastPeriod: 2,
    slowPeriod: 10,
    volumePeriod: 3,
    volumePower: 1,
  };
  const original = new VolumeWeightedKAMAIndicator(unusedApi, options);
  for (const [index, price] of [100, 99, 101].entries()) {
    original.onTick({ eventTime: index, price, quantity: 10 });
  }
  const restored = new VolumeWeightedKAMAIndicator(unusedApi, options);
  restored.restore(structuredClone(original.snapshot()));
  original.onTick({ eventTime: 4, price: 103, quantity: 0 });
  restored.onTick({ eventTime: 4, price: 103 });

  assert.deepEqual(restored.snapshot(), original.snapshot());
  assert.equal(restored.details().relativeVolume, 1);
  assert.ok(restored.derivative() > 0);
});

test("volume-weighted KAMA restores canonical KAMA snapshots without losing price state", () => {
  const canonical = new KAMAIndicator(2, 2, 10, unusedApi, 10, 1);
  for (const [index, price] of [100, 99, 102].entries()) {
    canonical.onTick({ eventTime: index, price, quantity: 10 });
  }
  const weighted = new VolumeWeightedKAMAIndicator(unusedApi, {
    efficiencyPeriod: 2,
    fastPeriod: 2,
    slowPeriod: 10,
    power: 1,
    volumePeriod: 3,
  });
  weighted.restore(canonical.snapshot());

  assert.equal(weighted.indicator(), canonical.indicator());
  assert.equal(weighted.derivative(), canonical.derivative());
});
