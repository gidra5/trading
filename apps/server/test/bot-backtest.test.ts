import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import {
  createPeakValleyBotConfig,
  createStrategyConfig,
  type Candle,
  type ExposureValueOracleActionDistribution,
  type TradingCandle,
} from "@trading/bot-algo";
import {
  confidenceConditionedHindsightOracleExposure,
  confidenceConditionedOracleLeverageFraction,
  confidenceScaledHindsightOracleExposure,
  hindsightOracleTargetDecision,
  hindsightOracleUsableDistribution,
  oracleExecutionExposureScale,
  oracleMaximumEffectiveLeverage,
  runBotBacktestFromCandles,
  scaleOracleConfidence,
  type OracleBacktestDecision,
} from "../src/bot-backtest.js";
import { HistoricalCandleCache } from "../src/historical-candle-cache.js";
import {
  historicalStrategyWarmupSamples,
  intervalToMs,
  runHistoricalCandleBacktest,
} from "../src/historical-backtest.js";
import { logitsDistribution } from "../src/joint-price-oracle-runtime.js";
import { LearnedOracleStrategy } from "../src/learned-oracle-strategy.js";

test("historical intervals include one-second candles", () => {
  assert.equal(intervalToMs("1s"), 1_000);
  assert.equal(intervalToMs("5m"), 300_000);
});

test("historical learned-oracle replay always loads a full model context", () => {
  const config = createStrategyConfig({
    legacyValleyPeak: {
      averagingRangesSec: [2],
      derivativeSource: "price",
    },
  });
  assert.equal(
    historicalStrategyWarmupSamples(config, 1_000, "learned-oracle-1s"),
    3_600,
  );
  assert.equal(historicalStrategyWarmupSamples(config, 1_000, "peak-valley"), 2);
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
  assert.equal(confidenceConditionedHindsightOracleExposure(100, 0.5, 100, 0, 0.75, 0.2), 11.5);
  assert.equal(confidenceConditionedHindsightOracleExposure(100, 1, 100, 0, 0.75, 0.05), 5);
  assert.equal(confidenceConditionedHindsightOracleExposure(100, 0, 100, 0, 0.75, 0.5), 18.75);
  assert.equal(confidenceConditionedOracleLeverageFraction(0, 0.75, 0.5), 0.1875);
  assert.equal(confidenceConditionedOracleLeverageFraction(1, 0.75, 0.5), 0.5);
  assert.equal(scaleOracleConfidence(0.8, 0.5), 0.4);
  assert.equal(scaleOracleConfidence(1, 0.05), 0.05);
  assert.throws(() => scaleOracleConfidence(0.5, 1.01), /must be in \[0, 1\]/);
});

test("hindsight oracle conditions the target distribution on actual exposure", () => {
  const distribution = oracleDistribution([0.5, 0, 0.5]);
  assert.equal(hindsightOracleTargetDecision(distribution, 1, 0.1, 1).targetExposure, 1);
  assert.equal(hindsightOracleTargetDecision(distribution, -1, 0.1, 1).targetExposure, -1);
});

test("hindsight oracle holds when transition conditioning has no solvent action", () => {
  const decision = hindsightOracleTargetDecision(
    oracleDistribution([1, 0, 0]),
    100,
    0.02,
  );

  assert.equal(decision.targetExposure, 100);
  assert.equal(decision.confidence, 1);
  assert.equal(decision.feasibleActionCount, 0);
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
  const candles = prices.map((price, index) =>
    timedCandle(index * intervalMs, price, intervalMs));
  const result = await runBotBacktestFromCandles(
    candles,
    {
      config,
      strategy: "hindsight-oracle-1s",
      warmup,
      oracleExpansionConfirmationMass: 0,
    },
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

  const capped = await runBotBacktestFromCandles(candles, {
    config,
    strategy: "hindsight-oracle-1s",
    warmup,
    hindsightOracleMaximumLeverage: 1,
    oracleExpansionConfirmationMass: 0,
  });
  assert.equal(capped.summary.maxEntryLeverage, 1);
  assert.equal(capped.summary.perfectMarginLeverage, 1);
  assert.equal(capped.summary.tradeCount > 0, true);
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
  const oracleDecisions: { currentExposure: number; targetExposure: number }[] = [];
  const result = await runBotBacktestFromCandles(candles, {
    config,
    strategy: "learned-oracle-1s",
    learnedOracleDistributionAt: (timestamp) => {
      if (timestamp !== candles[0]!.closeTime && timestamp !== candles[60]!.closeTime) return null;
      predictions += 1;
      return oracleDistribution([0, 0, 1]);
    },
    onOracleDecision: ({ currentExposure, targetExposure }) => {
      oracleDecisions.push({ currentExposure, targetExposure });
    },
  });

  assert.equal(result.summary.strategy, "learned-oracle-1s");
  assert.equal(predictions, 2);
  assert.equal(oracleDecisions.length, 2);
  assert.equal(oracleDecisions[0]?.currentExposure, 0);
  assert.ok((oracleDecisions[0]?.targetExposure ?? 0) > 0);
  assert.ok(result.summary.tradeCount > 0);
  assert.equal(result.summary.maxEntryLeverage, 1);
  assert.equal(result.summary.perfectMarginLeverage, 1);
});

test("a learned-oracle flat target market-closes exposure on its decision tick", async () => {
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
    timedCandle(index * 1_000, index === 1 ? 99 : 100, 1_000));
  const enterTime = candles[0]!.closeTime;
  const closeTime = candles[60]!.closeTime;
  const result = await runBotBacktestFromCandles(candles, {
    config,
    strategy: "learned-oracle-1s",
    learnedOracleMaximumLeverage: 1,
    oracleExpansionConfirmationMass: 0,
    oracleExpansionDeltaCapFraction: 1,
    learnedOracleDistributionAt: (timestamp) => timestamp === enterTime
      ? oracleDistribution([0, 0, 1])
      : timestamp === closeTime
        ? oracleDistribution([0, 1, 0])
        : null,
  });

  assert.ok(result.summary.tradeCount >= 2);
  assert.equal(result.summary.closedPositionCount, 1);
  assert.equal(result.fills.at(-1)?.filledAt, closeTime);
  assert.equal(result.fills.at(-1)?.side, "sell");
});

test("capped oracle replay scales filled exposure back to the native policy state", async () => {
  const config = createStrategyConfig({
    startingQuote: 10_000,
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
    timedCandle(index * 1_000, 100, 1_000));
  const firstTime = candles[0]!.closeTime;
  const secondTime = candles[60]!.closeTime;
  const enterLong = nativeOracleDistribution([0, 0, 1]);
  const preferShortWithoutNativeState = nativeOracleDistribution([0.9, 0, 0.1]);
  const decisions: OracleBacktestDecision[] = [];

  await runBotBacktestFromCandles(candles, {
    config,
    strategy: "learned-oracle-1s",
    learnedOracleMaximumLeverage: 1,
    oracleExpansionDeltaCapFraction: 1,
    learnedOracleDistributionAt: (timestamp) => timestamp === firstTime
      ? enterLong
      : timestamp === secondTime
        ? preferShortWithoutNativeState
        : null,
    onOracleDecision: (decision) => decisions.push(decision),
  });

  assert.equal(oracleExecutionExposureScale(enterLong, 1), 0.01);
  assert.equal(oracleMaximumEffectiveLeverage(1), 2.5);
  assert.equal(decisions.length, 2);
  assert.ok((decisions[1]?.currentExposure ?? 0) > 0.9);
  // A filled +1x execution position is native +100x policy state.  Native
  // transition friction therefore keeps the existing long despite a 9:1 base
  // preference for short; conditioning on the unscaled +1 would reverse it.
  assert.equal(decisions[1]?.conditionedModalExposure, 100);
  assert.ok(Math.abs((decisions[1]?.targetExposure ?? 0) - 1) < 1e-6);
  assert.equal(decisions[1]?.signalEmitted, false);
});

test("live learned-oracle strategy uses the scaled native-grid deadband", async () => {
  const config = createStrategyConfig({
    maxLeverage: 1,
    legacyValleyPeak: {
      averagingRangesSec: [2],
      derivativeSource: "price",
    },
  });
  const history: TradingCandle[] = Array.from({ length: 3_600 }, (_, index) => ({
    openTime: -3_599_000 + index * 1_000,
    closeTime: -3_598_001 + index * 1_000,
    open: 100,
    high: 100,
    low: 100,
    close: 100,
    volume: 1,
  }));
  const distribution = nativeOracleDistribution([0, 0, 1]);
  const runtime = {
    predictLatest: async () => distribution,
  } as unknown as ConstructorParameters<typeof LearnedOracleStrategy>[1];
  const strategy = new LearnedOracleStrategy(
    {
      config: createPeakValleyBotConfig(config, 1_000).strategy,
      getHistory: async () => history,
    },
    runtime,
    async () => history,
    0.00175,
  );
  await strategy.onTick({
    timestamp: 999,
    price: 100,
    quantity: 1,
    candle: history.at(-1)!,
  });

  const signal = await strategy.targetExposureSignal({
    timestamp: 999,
    price: 100,
    equity: 10_000,
    currentExposure: 0,
    maxLeverage: 1,
  });
  assert.ok(signal);
  assert.equal(signal.targetExposure, 1);
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

test("hindsight oracle accepts one-minute candles and evaluates every candle", async () => {
  const intervalMs = 60_000;
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    legacyValleyPeak: {
      averagingRangesSec: [60],
      trendSigmaWindowSec: 60,
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const candles = Array.from({ length: 3 }, (_, index) =>
    timedCandle(index * intervalMs, 100 + index, intervalMs));
  const timestamps: number[] = [];

  await runBotBacktestFromCandles(candles, {
    config,
    strategy: "hindsight-oracle-1s",
    hindsightOracleDistributionAt: (timestamp) => {
      timestamps.push(timestamp);
      return oracleDistribution([0, 1, 0]);
    },
  });

  assert.deepEqual(timestamps, candles.map((candle) => candle.closeTime));
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
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const candles = Array.from({ length: 62 }, (_, index) =>
    timedCandle(index * 1_000, index === 1 ? 99 : 100, 1_000));
  const enterTime = candles[0]!.closeTime;
  const closeTime = candles[60]!.closeTime;
  const distributionAt = (timestamp: number) => timestamp === enterTime
    ? oracleDistribution([0, 0, 1])
    : timestamp === closeTime
      ? oracleDistribution([0, 1, 0])
      : null;
  const backtestOptions = {
    config,
    strategy: "learned-oracle-1s" as const,
    learnedOracleMaximumLeverage: 1,
    oracleExpansionConfirmationMass: 0,
    oracleExpansionDeltaCapFraction: 1,
    learnedOracleDistributionAt: distributionAt,
  };
  const full = await runBotBacktestFromCandles(candles, {
    ...backtestOptions,
  });
  const summary = await runBotBacktestFromCandles(candles, {
    ...backtestOptions,
    summaryOnly: true,
  });

  assert.equal(summary.summary.finalEquity, full.summary.finalEquity);
  assert.equal(summary.summary.returnPct, full.summary.returnPct);
  assert.equal(
    summary.summary.maxInitialBalanceDrawdownPct,
    full.summary.maxInitialBalanceDrawdownPct,
  );
  assert.equal(summary.summary.maxDrawdownPct, full.summary.maxDrawdownPct);
  assert.equal(summary.summary.maxEffectiveLeverage, full.summary.maxEffectiveLeverage);
  assert.equal(summary.summary.tradeCount, full.summary.tradeCount);
  assert.ok(full.summary.closedPositionCount > 0);
  assert.equal(summary.summary.closedPositionCount, full.summary.closedPositionCount);
  assert.equal(
    summary.summary.profitableClosedPositionCount,
    full.summary.profitableClosedPositionCount,
  );
  assert.equal(summary.summary.winRate, full.summary.winRate);
  assert.equal(summary.finalState.realizedPnl, full.finalState.realizedPnl);
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

test("MACD and aggressor-volume strategies execute through the regular bot", async () => {
  const config = createStrategyConfig({
    startingQuote: 1_000,
    maxLeverage: 1,
    cooldownMs: 0,
    legacyValleyPeak: {
      anticipatoryGridOrderCount: 1,
      exitGridOrderCount: 1,
    },
  });
  const candles = Array.from({ length: 6_000 }, (_, index) => {
    const previous = technicalStrategyTestPrice(index - 1);
    const close = technicalStrategyTestPrice(index);
    return {
      ...timedCandle(index * 60_000, close),
      open: previous,
      high: Math.max(previous, close) + 0.1,
      low: Math.min(previous, close) - 0.1,
      volume: 10 + index % 7,
      aggressiveBuyVolume: close >= previous ? 9 : 1,
      aggressiveSellVolume: close >= previous ? 1 : 9,
    };
  });

  for (const strategy of ["macd", "volume-imbalance"] as const) {
    const result = await runBotBacktestFromCandles(candles, {
      config,
      strategy,
      summaryOnly: true,
    });
    assert.equal(result.summary.strategy, strategy);
    assert.ok(result.summary.tradeCount > 0, `${strategy} should trade the matched synthetic cycle`);
    assert.ok(Number.isFinite(result.summary.returnPct));
    if (strategy === "macd") {
      assert.ok(result.summary.closedPositionCount > 0, "MACD should close a synthetic-cycle position");
    }
  }
});

function technicalStrategyTestPrice(index: number): number {
  const hour = Math.max(0, index) / 60;
  const phase = hour % 72;
  if (phase < 30) return 120 - phase * 0.5;
  if (phase < 42) return 105 + (phase - 30) * 1.5;
  return 123 - (phase - 42) * 0.6;
}

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

function nativeOracleDistribution(
  probabilities: readonly number[],
): ExposureValueOracleActionDistribution {
  const distribution = oracleDistribution(probabilities);
  const grid = Float64Array.from([-100, 0, 100]);
  return {
    ...distribution,
    grid,
    mean: probabilities.reduce(
      (sum, probability, index) => sum + probability * grid[index]!,
      0,
    ),
    secondMoment: probabilities.reduce(
      (sum, probability, index) => sum + probability * grid[index]! ** 2,
      0,
    ),
    modalExposure: grid[probabilities.reduce(
      (best, probability, index) => probability > probabilities[best]! ? index : best,
      0,
    )]!,
  };
}
