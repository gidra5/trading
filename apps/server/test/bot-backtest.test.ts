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
