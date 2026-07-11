import assert from "node:assert/strict";
import test from "node:test";
import {
  PeakValleyStrategy,
  createPeakValleyStrategyConfig,
  type TradingCandle,
} from "../src/index.js";

test("peak/valley warms indicators without emitting and signals a confirmed reversal", async () => {
  const history = [candle(0, 3), candle(60_000, 2)];
  const strategy = new PeakValleyStrategy({
    config: createPeakValleyStrategyConfig({
      averagingRangesSec: [60, 120],
      relativeRateEnabled: false,
      rateThresholdsLow: [0, 0],
      rateThresholdsHigh: [0, 0],
      buyDataIndex: 0,
      sellDataIndex: 0,
      buyConfirmationOffsets: [1],
      sellConfirmationOffsets: [1],
      buyExitConfirmationOffsets: [1],
      sellExitConfirmationOffsets: [1],
      saturationSec: 0,
      sigmaMode: "static",
      buySigma: 1,
      sellSigma: 1,
      kamaErLen: 1,
      kamaSlowLen: 2,
    }),
    getHistory: async ({ count }) => history.slice(-count),
  });

  await strategy.warmup();
  assert.equal(await strategy.entrySignal(), null);
  assert.equal(await strategy.exitSignal(), null);

  await strategy.onTick({
    timestamp: 180_000,
    price: 4,
    quantity: 1,
    candle: null,
  });
  assert.equal((await strategy.entrySignal())?.side, "long");
  assert.equal((await strategy.exitSignal())?.side, "short");
  assert.equal(strategy.getDiagnostics().blockers.length, 0);
  assert.equal(strategy.getDiagnostics().gates[0]?.passed, true);

  await strategy.onTick({
    timestamp: 181_000,
    price: 4,
    quantity: 1,
    candle: null,
  });
  assert.equal(await strategy.entrySignal(), null);
  assert.equal(await strategy.exitSignal(), null);
});

test("peak/valley restores indicator snapshots", async () => {
  const config = createPeakValleyStrategyConfig({
    averagingRangesSec: [60],
    saturationSec: 0,
    kamaErLen: 1,
    kamaSlowLen: 2,
  });
  const getHistory = async () => [candle(0, 100), candle(60_000, 99)];
  const strategy = new PeakValleyStrategy({ config, getHistory });
  await strategy.warmup();
  const snapshot = await strategy.snapshot();
  const restored = new PeakValleyStrategy({ config, getHistory });
  await restored.restore(structuredClone(snapshot));

  assert.deepEqual(await restored.snapshot(), snapshot);
});

function candle(openTime: number, close: number): TradingCandle {
  return {
    openTime,
    closeTime: openTime + 59_999,
    open: close,
    high: close,
    low: close,
    close,
    volume: 1,
  };
}
