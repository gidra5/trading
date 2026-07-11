import assert from "node:assert/strict";
import test from "node:test";
import { createStrategyConfig, type Candle } from "@trading/bot-algo";
import { runBotBacktestFromCandles } from "../src/bot-backtest.js";

test("new-bot candle replay returns finite account metrics", async () => {
  const prices = [100, 99, 98, 99, 100, 101, 100, 99, 100, 101];
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [60],
      relativeRateEnabled: false,
      rateThresholdsLow: [0],
      rateThresholdsHigh: [0],
      buyDataIndex: 0,
      sellDataIndex: 0,
      buyConfirmationOffsets: [],
      sellConfirmationOffsets: [],
      buyExitConfirmationOffsets: [],
      sellExitConfirmationOffsets: [],
      saturationSec: 0,
      sigmaMode: "static",
      buySigma: 1,
      sellSigma: 1,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const result = await runBotBacktestFromCandles(prices.map(candle), { config });

  assert.equal(Number.isFinite(result.summary.finalEquity), true);
  assert.equal(Number.isFinite(result.summary.returnPct), true);
  assert.equal(result.summary.tradeCount > 0, true);
  assert.equal(result.finalState.metrics.equity, result.summary.finalEquity);
  assert.equal(Number.isFinite(result.summary.perfectMarginReturnPct), true);
  const trace = result.candleChart?.trace;
  assert.ok(trace);
  assert.equal(trace.positions.length > 0, true);
  assert.equal(trace.orders.length > 0, true);
  assert.equal(trace.orders.every((order) => order.positionId && order.gridId), true);
  assert.equal(trace.grids.every((grid) => grid.cause.length > 0), true);
  assert.equal(trace.signals.length > 0, true);
  assert.equal(trace.oracle.points.length > 0, true);
  assert.equal(trace.frames.length > 0, true);
  assert.equal(trace.frames.every((frame) => Number.isFinite(frame.metrics.equity)), true);
  assert.equal(trace.frames.at(-1)?.metrics.equity, result.summary.finalEquity);
  assert.equal(trace.positions.some((position) => position.states.length > 1), true);
  assert.equal(
    result.candleChart?.annotations
      .filter((annotation) => annotation.orderId)
      .every((annotation) => annotation.targetPositionId && annotation.gridId),
    true,
  );
});

test("new-bot replay exposes centered-SMA extrema and their order errors", async () => {
  const prices = Array.from({ length: 100 }, (_, index) => 100 + Math.sin(index / 8) * 10);
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [60],
      relativeRateEnabled: false,
      rateThresholdsLow: [0],
      rateThresholdsHigh: [0],
      buyDataIndex: 0,
      sellDataIndex: 0,
      buyConfirmationOffsets: [],
      sellConfirmationOffsets: [],
      buyExitConfirmationOffsets: [],
      sellExitConfirmationOffsets: [],
      saturationSec: 0,
      sigmaMode: "static",
      buySigma: 1,
      sellSigma: 1,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const result = await runBotBacktestFromCandles(prices.map(candle), {
    config,
    extremaSmaWindowMs: 10 * 60_000,
  });
  const extrema = result.candleChart?.trace?.extrema ?? [];

  assert.equal(extrema.some((item) => item.kind === "peak"), true);
  assert.equal(extrema.some((item) => item.kind === "valley"), true);
  assert.equal(extrema.some((item) => item.orders.length > 0), true);
  assert.equal((result.summary.extremaOrderMass?.peakCount ?? 0) > 0, true);
  assert.equal(result.summary.extremaOrderMass?.smaWindowMs, 10 * 60_000);
});

function candle(close: number, index: number): Candle {
  const openTime = index * 60_000;
  return {
    symbol: "BTCUSDT",
    interval: "1m",
    openTime,
    closeTime: openTime + 59_999,
    open: close,
    high: close,
    low: close,
    close,
    volume: 10,
    closed: true,
  };
}
