import assert from "node:assert/strict";
import test from "node:test";
import {
  EMAIndicator,
  LinearRegressionIndicator,
  SMAIndicator,
} from "../src/indicators.js";
import type { TradingApi } from "../src/trading-api.js";

const unusedApi = {} as TradingApi;

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
