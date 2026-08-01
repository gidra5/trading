import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import {
  createStrategyConfig,
  type Candle,
  type ExposureValueOracleActionDistribution,
} from "@trading/bot-algo";
import {
  confidenceConditionedHindsightOracleExposure,
  confidenceScaledHindsightOracleExposure,
  hindsightOracleTargetDecision,
  hindsightOracleUsableDistribution,
  runBotBacktestFromCandles,
} from "../src/bot-backtest.js";
import { HistoricalCandleCache } from "../src/historical-candle-cache.js";
import { intervalToMs, runHistoricalCandleBacktest } from "../src/historical-backtest.js";
import { logitsDistribution } from "../src/joint-price-oracle-runtime.js";

test("historical intervals include one-second candles", () => {
  assert.equal(intervalToMs("1s"), 1_000);
  assert.equal(intervalToMs("5m"), 300_000);
});

test("hindsight oracle derives its target mode and confidence from the distribution", () => {
  const concentrated = oracleDistribution([0.01, 0.09, 0.9]);
  const decision = hindsightOracleTargetDecision(concentrated, 0, 0);
  assert.equal(decision.targetExposure, 1);
  assert.ok(decision.confidence > 0.6);

  const uniform = hindsightOracleTargetDecision(
    oracleDistribution([1 / 3, 1 / 3, 1 / 3]),
    0,
    0,
  );
  assert.ok(Math.abs(uniform.confidence) < 1e-12);
});

test("hindsight oracle confidence scales deployed exposure without shrinking its modal span", () => {
  assert.equal(confidenceScaledHindsightOracleExposure(100, 0.5, 0), 100);
  assert.equal(confidenceScaledHindsightOracleExposure(100, 0.5, 1), 50);
  assert.equal(confidenceScaledHindsightOracleExposure(-100, 0.25, 1), -25);
  assert.equal(confidenceScaledHindsightOracleExposure(100, 0.5, 2), 25);
  assert.equal(confidenceConditionedHindsightOracleExposure(100, 0.5, 100, 0, 0.5), 75);
  assert.equal(confidenceConditionedHindsightOracleExposure(50, 0.5, 100, 0, 0.5), 50);
  assert.equal(confidenceConditionedHindsightOracleExposure(-100, 0.25, 100, 0, 0.5), -62.5);
});

test("hindsight oracle conditions the target distribution on actual exposure", () => {
  const distribution = oracleDistribution([0.5, 0, 0.5]);
  assert.equal(hindsightOracleTargetDecision(distribution, 1, 0.1, 1).targetExposure, 1);
  assert.equal(hindsightOracleTargetDecision(distribution, -1, 0.1, 1).targetExposure, -1);
});

test("hindsight oracle truncates a latent distribution to the usable exposure interval", () => {
  const distribution: ExposureValueOracleActionDistribution = {
    grid: Float64Array.from([-250, -100, 0, 100, 250]),
    probabilities: Float64Array.from([0.1, 0.2, 0.3, 0.4, 0]),
    mean: 0,
    secondMoment: 0,
    modalExposure: 100,
    entropy: 0,
    opportunity: 7,
    feasibleActionCount: 4,
  };

  const usable = hindsightOracleUsableDistribution(distribution, 100);

  assert.deepEqual(Array.from(usable.grid), [-100, 0, 100]);
  assert.deepEqual(
    Array.from(usable.probabilities).map((value) => Number(value.toFixed(12))),
    [2 / 9, 1 / 3, 4 / 9].map((value) => Number(value.toFixed(12))),
  );
  assert.equal(usable.modalExposure, 100);
  assert.equal(usable.feasibleActionCount, 3);
  assert.equal(usable.opportunity, 7);
});

test("joint price-oracle logits become a normalized bot distribution", () => {
  const distribution = logitsDistribution([-4, 0, 4], [-1, 0, 1]);
  assert.ok(Math.abs(distribution.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
  assert.equal(distribution.modalExposure, 1);
  assert.ok(distribution.mean > 0.9);
  assert.equal(distribution.feasibleActionCount, 3);
});

test("historical backtests warm the strategy before the measured window", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "historical-warmup-"));
  const intervalMs = 60_000;
  const targetStartTime = Math.floor(Date.now() / intervalMs) * intervalMs - 5 * intervalMs;
  const source = Array.from({ length: 10 }, (_, index) =>
    timedCandle(targetStartTime - intervalMs + index * intervalMs, 100 - index));
  const cacheOptions = {
    dataDir,
    marketKey: "spot:test",
    symbol: "BTCUSDT",
    interval: "1m",
    intervalMs,
    maxBytes: 100_000_000,
    minFreeBytes: 0,
  };
  try {
    const cache = new HistoricalCandleCache(cacheOptions);
    await cache.ensureRange(source[0]!.openTime, source.at(-1)!.openTime, async (request) =>
      source.filter((item) =>
        item.openTime >= request.startTime && item.openTime <= request.endTime).slice(0, request.limit));
    const config = createStrategyConfig({
      startingQuote: 1_000,
      maxLeverage: 1,
      cooldownMs: 0,
      legacyValleyPeak: {
        averagingRangesSec: [60],
        derivativeSource: "price",
        derivativeClampMode: "deadband",
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
    const result = await runHistoricalCandleBacktest({
      id: "warmup",
      preset: "last-x",
      marketKey: cacheOptions.marketKey,
      venue: "spot",
      symbol: cacheOptions.symbol,
      interval: cacheOptions.interval,
      config,
      cache: cacheOptions,
      historicalStartTime: targetStartTime,
    }, () => {});

    assert.equal(result.candleChart?.trace?.signals[0]?.time, targetStartTime + intervalMs - 1);
    assert.equal(result.summary.startTime, targetStartTime);
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

test("new-bot candle replay returns finite account metrics", async () => {
  const prices = [100, 99, 98, 99, 100, 101, 100, 99, 100, 101];
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [60],
      derivativeSource: "price",
      derivativeClampMode: "deadband",
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
  assert.equal(trace.oracle.eventMode, "close");
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

test("one-second hindsight oracle drives the regular bot execution path", async () => {
  const intervalMs = 1_000;
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [2],
      trendSigmaWindowSec: 2,
      derivativeClampMode: "deadband",
      relativeRateEnabled: false,
      rateThresholdsLow: [0],
      rateThresholdsHigh: [0],
      saturationSec: 0,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const warmup = [103, 102, 101].map((price, index) =>
    timedCandle((index - 3) * intervalMs, price, intervalMs));
  const prices = [100, 99, 101, 101, 150];
  const result = await runBotBacktestFromCandles(
    prices.map((price, index) => timedCandle(index * intervalMs, price, intervalMs)),
    { config, strategy: "hindsight-oracle-1s", warmup },
  );

  assert.equal(result.summary.strategy, "hindsight-oracle-1s");
  assert.equal(result.summary.tradeCount > 0, true);
  const oracleSignal = result.candleChart?.trace?.signals.find((signal) =>
    signal.indicators["oracle.targetExposure"] !== undefined);
  assert.ok(oracleSignal);
  assert.ok((oracleSignal.indicators["oracle.targetExposure"] ?? 0) > 0);
  assert.ok((oracleSignal.indicators["oracle.confidence"] ?? 0) > 0);
  assert.equal(result.summary.maxEntryLeverage, 100);
  assert.equal(result.summary.perfectMarginLeverage, 100);
});

test("causal learned oracle drives the regular bot without future candles", async () => {
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [2],
      trendSigmaWindowSec: 2,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const candles = Array.from({ length: 62 }, (_, index) =>
    timedCandle(
      index * 1_000,
      [100, 99, 101, 101, 150][index] ?? 150 + index * 0.01,
      1_000,
    ));
  let predictions = 0;
  const result = await runBotBacktestFromCandles(candles, {
    config,
    strategy: "learned-oracle-1s",
    learnedOracleDistributionAt: (timestamp) => {
      if (timestamp !== candles[0]!.closeTime && timestamp !== candles[60]!.closeTime) return null;
      predictions += 1;
      return oracleDistribution([0, 0, 1]);
    },
  });

  assert.equal(result.summary.strategy, "learned-oracle-1s");
  assert.equal(predictions, 2);
  assert.ok(result.summary.tradeCount > 0);
});

test("one-second hindsight oracle evaluates once per 60-second holding block", async () => {
  const intervalMs = 1_000;
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    legacyValleyPeak: {
      averagingRangesSec: [2],
      trendSigmaWindowSec: 2,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const candles = Array.from({ length: 122 }, (_, index) =>
    timedCandle(index * intervalMs, 100 + index * 0.01, intervalMs));
  const result = await runBotBacktestFromCandles(candles, {
    config,
    strategy: "hindsight-oracle-1s",
  });
  const oracleSignals = result.candleChart?.trace?.signals.filter((signal) =>
    signal.indicators["oracle.targetExposure"] !== undefined) ?? [];

  assert.deepEqual(
    oracleSignals.map((signal) => signal.time),
    [candles[0]!.closeTime, candles[60]!.closeTime, candles[120]!.closeTime],
  );
});

test("one-second hindsight replay stops at an intrabar liquidation boundary", async () => {
  const config = createStrategyConfig({
    startingQuote: 1_000,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [2],
      trendSigmaWindowSec: 2,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const first = timedCandle(0, 100, 1_000);
  const adverse: Candle = {
    ...timedCandle(1_000, 100, 1_000),
    open: 100,
    high: 100,
    low: 98,
    volume: 1_000_000,
  };
  const distribution: ExposureValueOracleActionDistribution = {
    grid: Float64Array.from([-100, 0, 100]),
    probabilities: Float64Array.from([0, 0, 1]),
    mean: 100,
    secondMoment: 10_000,
    modalExposure: 100,
    entropy: 0,
    opportunity: 1,
    feasibleActionCount: 1,
  };

  const result = await runBotBacktestFromCandles([first, adverse], {
    config,
    strategy: "hindsight-oracle-1s",
    hindsightOracleDistributionAt: () => distribution,
    summaryOnly: true,
  });

  assert.equal(result.summary.stoppedEarly, true);
  assert.equal(result.summary.stopReason, "liquidated");
  assert.equal(result.summary.liquidatedPositionCount, 1);
  assert.equal(result.summary.candlesProcessed, 2);
  assert.ok(result.summary.finalEquity > 0 && result.summary.finalEquity < 1_000);
  assert.ok((result.summary.maxEffectiveLeverage ?? 0) >= 249.999);
});

test("summary-only replay preserves trading and risk results", async () => {
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [2],
      trendSigmaWindowSec: 2,
      derivativeClampMode: "deadband",
      relativeRateEnabled: false,
      rateThresholdsLow: [0],
      rateThresholdsHigh: [0],
      saturationSec: 0,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const warmup = [103, 102, 101]
    .map((price, index) => timedCandle((index - 3) * 1_000, price, 1_000));
  const candles = [100, 99, 101, 101, 150]
    .map((price, index) => timedCandle(index * 1_000, price, 1_000));
  const full = await runBotBacktestFromCandles(candles, {
    config,
    strategy: "hindsight-oracle-1s",
    warmup,
  });
  const summary = await runBotBacktestFromCandles(candles, {
    config,
    strategy: "hindsight-oracle-1s",
    warmup,
    summaryOnly: true,
  });

  assert.equal(summary.summary.finalEquity, full.summary.finalEquity);
  assert.equal(summary.summary.returnPct, full.summary.returnPct);
  assert.equal(summary.summary.maxDrawdownPct, full.summary.maxDrawdownPct);
  assert.equal(summary.summary.maxEffectiveLeverage, full.summary.maxEffectiveLeverage);
  assert.equal(summary.summary.tradeCount, full.summary.tradeCount);
});

test("new-bot replay exposes centered-SMA extrema and their order errors", async () => {
  const prices = Array.from({ length: 100 }, (_, index) => 100 + Math.sin(index / 8) * 10);
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      averagingRangesSec: [60],
      derivativeSource: "price",
      derivativeClampMode: "deadband",
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
  return timedCandle(index * 60_000, close);
}

function timedCandle(openTime: number, close: number, intervalMs = 60_000): Candle {
  return {
    symbol: "BTCUSDT",
    interval: intervalMs === 1_000 ? "1s" : "1m",
    openTime,
    closeTime: openTime + intervalMs - 1,
    open: close,
    high: close,
    low: close,
    close,
    volume: 10,
    closed: true,
  };
}

function oracleDistribution(probabilities: readonly number[]): ExposureValueOracleActionDistribution {
  const grid = Float64Array.from([-1, 0, 1]);
  const values = Float64Array.from(probabilities);
  const entropy = values.reduce(
    (total, probability) => total - (probability > 0 ? probability * Math.log(probability) : 0),
    0,
  );
  const modalIndex = values.reduce(
    (best, probability, index) => probability > values[best]! ? index : best,
    0,
  );
  return {
    grid,
    probabilities: values,
    mean: values.reduce((sum, probability, index) => sum + probability * grid[index]!, 0),
    secondMoment: values.reduce(
      (sum, probability, index) => sum + probability * grid[index]! ** 2,
      0,
    ),
    modalExposure: grid[modalIndex]!,
    entropy,
    opportunity: 1,
    feasibleActionCount: values.filter((probability) => probability > 0).length,
  };
}
