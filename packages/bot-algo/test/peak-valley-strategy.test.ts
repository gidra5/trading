import assert from "node:assert/strict";
import test from "node:test";
import {
  PeakValleyStrategy,
  createLegacyValleyPeakConfig,
  createPeakValleyBotConfig,
  createPeakValleyStrategyConfig,
  peakValleyKamaWarmupSamples,
  rescalePeakValleyStrategyConfig,
  type TradingCandle,
} from "../src/index.js";
import {
  createLegacyValleyPeakMemory,
  evaluateLegacyValleyPeak,
} from "../src/legacy/valley-peak.js";

test("peak/valley warms indicators without emitting and signals a confirmed reversal", async () => {
  const history = [candle(0, 3), candle(60_000, 2)];
  const strategy = new PeakValleyStrategy({
    config: createPeakValleyStrategyConfig({
      averagingRangesSec: [60, 120],
      derivativeSource: "price",
      derivativeClampMode: "deadband",
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

  const warmed = strategy.getDiagnostics();
  assert.equal(warmed.ready, true);
  assert.equal(warmed.warmupRemainingMs, 0);
  assert.equal(warmed.movingAverageType, "sma");
  assert.equal(warmed.derivativeSource, "price");
  assert.equal(warmed.latestTick?.price, 2);
  assert.equal(warmed.averages[0]?.roles.buyPrimary, true);
  assert.equal(warmed.averages[1]?.roles.buyEntryConfirmation, true);
  assert.equal(warmed.averages[1]?.roles.sellExitConfirmation, true);
  assert.equal(warmed.kama, null);
  assert.equal(warmed.sizing.buy.sigma, 1);

  const reversal = candle(120_000, 4);
  await strategy.onTick({
    timestamp: reversal.closeTime,
    price: reversal.close,
    quantity: reversal.volume,
    candle: reversal,
  });
  assert.equal((await strategy.entrySignal())?.side, "long");
  assert.equal((await strategy.exitSignal())?.side, "short");
  assert.equal(strategy.getDiagnostics().blockers.length, 0);
  assert.equal(strategy.getDiagnostics().gates[0]?.passed, true);

  await strategy.onTick({
    timestamp: reversal.closeTime + 1_000,
    price: 4,
    quantity: 100,
    candle: null,
  });
  assert.equal(await strategy.entrySignal(), null);
  assert.equal(await strategy.exitSignal(), null);
  assert.equal(strategy.getDiagnostics().latestTick?.timestamp, reversal.closeTime);
});

test("peak/valley restores indicator snapshots", async () => {
  const config = createPeakValleyStrategyConfig({
    averagingRangesSec: [60],
    derivativeSource: "kama",
    saturationSec: 0,
    kamaErLen: 1,
    kamaSlowLen: 2,
    kamaVolumeLen: 2,
  });
  const getHistory = async () => [candle(0, 100), candle(60_000, 99)];
  const strategy = new PeakValleyStrategy({ config, getHistory });
  await strategy.warmup();
  const snapshot = await strategy.snapshot();
  assert.equal(snapshot.version, 3);
  assert.equal(snapshot.lastKamaSignalPrice, 99);
  const restored = new PeakValleyStrategy({ config, getHistory });
  await restored.restore(structuredClone(snapshot));

  assert.deepEqual(await restored.snapshot(), snapshot);
});

test("peak/valley converts duration windows at the configured candle scale", async () => {
  const history = [candle(0, 1, 1_000), candle(1_000, 3, 1_000)];
  let requestedInterval = 0;
  const strategy = new PeakValleyStrategy({
    config: createPeakValleyStrategyConfig({
      sampleIntervalMs: 1_000,
      averagingRangesSec: [2],
      derivativeSource: "price",
      saturationSec: 0,
      kamaErLen: 1,
      kamaSlowLen: 2,
      kamaVolumeLen: 2,
    }),
    getHistory: async ({ intervalMs }) => {
      requestedInterval = intervalMs;
      return history;
    },
  });
  await strategy.warmup();

  assert.equal(requestedInterval, 1_000);
  assert.equal(strategy.getDiagnostics().averages[0]?.value, 2);
  assert.equal(strategy.getDiagnostics().ready, true);
});

test("peak/valley hold clamp retains its state until a strict opposite threshold crossing", async () => {
  const strategy = new PeakValleyStrategy({
    config: createPeakValleyStrategyConfig({
      sampleIntervalMs: 1_000,
      averagingRangesSec: [1],
      relativeRateEnabled: false,
      derivativeClampMode: "hold",
      rateThresholdsLow: [1],
      rateThresholdsHigh: [1],
      saturationSec: 0,
    }),
    getHistory: async () => [],
  });
  const update = async (time: number, price: number) => {
    const item = candle(time, price, 1_000);
    await strategy.onTick({
      timestamp: item.closeTime,
      price: item.close,
      quantity: item.volume,
      candle: item,
    });
    return strategy.getDiagnostics().averages[0]?.clampedRate;
  };

  assert.equal(await update(0, 100), 0);
  assert.equal(await update(1_000, 102), 2);
  assert.equal(await update(2_000, 102.5), 2);
  assert.equal(await update(3_000, 101.5), 2);
  assert.equal(await update(4_000, 100), -1.5);
  assert.equal(await update(5_000, 100.5), -1.5);
  assert.equal(await update(6_000, 101.5), -1.5);
  assert.equal(await update(7_000, 103), 1.5);
});

test("peak/valley deadband requires a strict threshold crossing", async () => {
  const strategy = new PeakValleyStrategy({
    config: createPeakValleyStrategyConfig({
      sampleIntervalMs: 1_000,
      averagingRangesSec: [1],
      derivativeSource: "price",
      relativeRateEnabled: false,
      derivativeClampMode: "deadband",
      rateThresholdsLow: [1],
      rateThresholdsHigh: [1],
      saturationSec: 0,
    }),
    getHistory: async () => [],
  });
  for (const [time, price] of [[0, 100], [1_000, 101]] as const) {
    await strategy.onTick(candleTick(candle(time, price, 1_000)));
  }
  assert.equal(strategy.getDiagnostics().averages[0]?.clampedRate, 0);
  await strategy.onTick(candleTick(candle(2_000, 102.1, 1_000)));
  assert.ok((strategy.getDiagnostics().averages[0]?.clampedRate ?? 0) > 1);
});

test("peak/valley config normalization preserves hold clamp mode", () => {
  assert.equal(createPeakValleyStrategyConfig({ derivativeClampMode: "hold" }).derivativeClampMode, "hold");
  assert.equal(createLegacyValleyPeakConfig({ derivativeClampMode: "hold" }).derivativeClampMode, "hold");
});

test("legacy execution applies the shared KAMA volume configuration", () => {
  const config = createLegacyValleyPeakConfig({
    derivativeSource: "kama",
    kamaVolumeLen: 2,
    saturationSec: 1_000,
  });
  const memory = createLegacyValleyPeakMemory(config);
  for (const [index, volume] of [10, 30].entries()) {
    const sourceCandle = {
      ...candle(index * 60_000, 100 + index),
      symbol: "BTCUSDT",
      interval: "1m",
      volume,
      closed: true,
    };
    evaluateLegacyValleyPeak(memory, config, {
      eventTime: sourceCandle.closeTime,
      price: sourceCandle.close,
      feeRate: 0,
      buyingPowerQuote: 1_000,
      shortSellingPowerQuote: 1_000,
      baseFree: 1,
      shortBaseFree: 1,
      sourceCandle,
    });
  }
  assert.ok(Math.abs((memory.kama.volumeEma ?? 0) - 70 / 3) < 1e-9);
});

test("defaults use the validation-selected multi-scale oracle candidate", () => {
  const config = createPeakValleyStrategyConfig();
  assert.deepEqual(
    [config.kamaErLen, config.kamaFastLen, config.kamaSlowLen, config.kamaVolumeLen],
    [14, 28, 153, 130],
  );
  assert.equal(config.derivativeSource, "kama");
  assert.equal(config.derivativeClampMode, "deadband");
  assert.deepEqual(config.buyConfirmationOffsets, []);
  assert.deepEqual(config.sellConfirmationOffsets, []);
  assert.equal(config.buyEntrySignalTiming, "end");
  assert.equal(config.buyExitSignalTiming, "start");
  assert.equal(config.kamaSignalFriction, 0.00175);
  assert.equal(config.kamaVolumePower, 0);
  assert.ok(Math.abs(config.kamaRateThresholdLow * 36_000_000 - 67.56654) < 1e-9);
});

test("bot config derives KAMA signal memory from execution friction", () => {
  const source = {
    maxLeverage: 5,
    minOrderQuote: 5,
    maxPositionQuote: 10_000,
    cooldownMs: 0,
    feeBps: 7.5,
    positionRisk: { marketSlippageBps: 10 },
    internalBorrowAccounting: "inactive" as const,
    borrowerProfitShareToLender: 1,
    legacyValleyPeak: { kamaSignalFriction: 1 },
  };
  assert.equal(createPeakValleyBotConfig(source).strategy.kamaSignalFriction, 0.00175);
  assert.equal(createPeakValleyBotConfig({
    ...source,
    feeBps: -20,
    positionRisk: { marketSlippageBps: 10 },
  }).strategy.kamaSignalFriction, 0);
});

test("KAMA sample counts preserve physical durations across candle scales", () => {
  const config = createPeakValleyStrategyConfig();
  const oneSecond = rescalePeakValleyStrategyConfig(config, 1_000);
  const fiveMinute = rescalePeakValleyStrategyConfig(config, 300_000);
  assert.deepEqual(
    [oneSecond.kamaErLen, oneSecond.kamaFastLen, oneSecond.kamaSlowLen, oneSecond.kamaVolumeLen],
    [840, 1_680, 9_180, 7_800],
  );
  assert.deepEqual(
    [fiveMinute.kamaErLen, fiveMinute.kamaFastLen, fiveMinute.kamaSlowLen, fiveMinute.kamaVolumeLen],
    [3, 6, 31, 26],
  );
  assert.equal(oneSecond.kamaRateThresholdLow, config.kamaRateThresholdLow);
  assert.equal(oneSecond.kamaSignalFriction, config.kamaSignalFriction);
});

test("KAMA warmup uses the same three-period context as the search", async () => {
  let count = 0;
  const config = createPeakValleyStrategyConfig({ averagingRangesSec: [60] });
  const strategy = new PeakValleyStrategy({
    config,
    getHistory: async (request) => {
      count = request.count;
      return [];
    },
  });
  await strategy.warmup();
  assert.equal(count, peakValleyKamaWarmupSamples(config));
});

test("unrelated average warmup does not change KAMA signal memory", async () => {
  const history = Array.from({ length: 100 }, (_, index) =>
    candle(index * 1_000, 100 + Math.sin(index) * 2, 1_000));
  const create = async (averagingRangesSec: number[]) => {
    const strategy = new PeakValleyStrategy({
      config: createPeakValleyStrategyConfig({
        sampleIntervalMs: 1_000,
        averagingRangesSec,
        saturationSec: 0,
        kamaErLen: 1,
        kamaFastLen: 1,
        kamaSlowLen: 1,
        kamaVolumeLen: 1,
        kamaSignalFriction: 0.01,
      }),
      getHistory: async ({ count }) => history.slice(-count),
    });
    await strategy.warmup();
    return strategy;
  };
  const short = await create([1]);
  const long = await create([100]);

  assert.deepEqual((await long.snapshot()).kama, (await short.snapshot()).kama);
  assert.equal(
    long.getDiagnostics().kama?.clampedRate,
    short.getDiagnostics().kama?.clampedRate,
  );
  assert.equal(
    long.getDiagnostics().kama?.lastSignalPrice,
    short.getDiagnostics().kama?.lastSignalPrice,
  );
});

test("peak/valley rewarms from the latest continuous segment after a candle gap", async () => {
  let history = Array.from({ length: 6 }, (_, index) => candle(index * 1_000, 100 + index, 1_000));
  const strategy = new PeakValleyStrategy({
    config: createPeakValleyStrategyConfig({
      sampleIntervalMs: 1_000,
      averagingRangesSec: [1],
      saturationSec: 0,
      kamaErLen: 1,
      kamaFastLen: 1,
      kamaSlowLen: 1,
      kamaVolumeLen: 1,
    }),
    getHistory: async () => history,
  });
  await strategy.warmup();
  assert.deepEqual(strategy.getDiagnostics().blockers, []);

  const afterGap = candle(10_000, 120, 1_000);
  history = [afterGap];
  await strategy.onTick({
    timestamp: afterGap.closeTime,
    price: afterGap.close,
    quantity: afterGap.volume,
    candle: afterGap,
  });
  assert.deepEqual(strategy.getDiagnostics().blockers, ["warmup"]);
  assert.equal(strategy.getDiagnostics().latestTick?.timestamp, afterGap.closeTime);
  const value = strategy.getDiagnostics().kama?.value;
  await strategy.onTick({
    timestamp: afterGap.closeTime,
    price: 999,
    quantity: 1,
    candle: { ...afterGap, close: 999 },
  });
  assert.equal(strategy.getDiagnostics().kama?.value, value);
});

test("KAMA signal memory suppresses reversals inside friction", async () => {
  const history = [100, 100.1, 100, 100.1, 100, 100.1]
    .map((price, index) => candle(index * 1_000, price, 1_000));
  const config = createPeakValleyStrategyConfig({
    sampleIntervalMs: 1_000,
    averagingRangesSec: [1],
    relativeRateEnabled: false,
    derivativeClampMode: "hold",
    kamaErLen: 1,
    kamaFastLen: 1,
    kamaSlowLen: 1,
    kamaPower: 1,
    kamaVolumeLen: 1,
    kamaVolumeCap: 1,
    kamaVolumePower: 0,
    kamaRateThresholdLow: 0.00001,
    kamaRateThresholdHigh: 0.00001,
    kamaSignalFriction: 0.01,
    saturationSec: 0,
    sigmaMode: "static",
    buySigma: 1,
    sellSigma: 1,
  });
  const strategy = new PeakValleyStrategy({ config, getHistory: async () => history });
  await strategy.warmup();

  assert.ok((strategy.getDiagnostics().kama?.clampedRate ?? 0) > 0);
  assert.equal(strategy.getDiagnostics().kama?.lastSignalPrice, 100.1);

  const near = candle(6_000, 100, 1_000);
  await strategy.onTick(candleTick(near));
  assert.ok((strategy.getDiagnostics().kama?.clampedRate ?? 0) > 0);
  assert.equal(strategy.getDiagnostics().kama?.lastSignalPrice, 100.1);
  assert.equal(await strategy.entrySignal(), null);
  assert.equal(await strategy.exitSignal(), null);

  const far = candle(7_000, 98, 1_000);
  await strategy.onTick(candleTick(far));
  assert.ok((strategy.getDiagnostics().kama?.clampedRate ?? 0) < 0);
  assert.equal(strategy.getDiagnostics().kama?.lastSignalPrice, 98);
  assert.equal((await strategy.entrySignal())?.side, "short");
  assert.equal((await strategy.exitSignal())?.side, "long");
});

test("KAMA signal memory also spaces flat transitions", async () => {
  const prices = [100, 102, 102, 103, 104, 104];
  const config = createPeakValleyStrategyConfig({
    sampleIntervalMs: 1_000,
    averagingRangesSec: [1],
    relativeRateEnabled: false,
    derivativeClampMode: "deadband",
    kamaErLen: 1,
    kamaFastLen: 1,
    kamaSlowLen: 1,
    kamaPower: 1,
    kamaVolumeLen: 1,
    kamaVolumeCap: 1,
    kamaVolumePower: 0,
    kamaRateThresholdLow: 0.00001,
    kamaRateThresholdHigh: 0.00001,
    kamaSignalFriction: 0.015,
    saturationSec: 0,
    sigmaMode: "static",
    buySigma: 1,
    sellSigma: 1,
  });
  const strategy = new PeakValleyStrategy({
    config,
    getHistory: async () => prices.slice(0, 2)
      .map((price, index) => candle(index * 1_000, price, 1_000)),
  });
  await strategy.warmup();

  for (let index = 2; index < prices.length - 1; index += 1) {
    await strategy.onTick(candleTick(candle(index * 1_000, prices[index]!, 1_000)));
    assert.ok((strategy.getDiagnostics().kama?.clampedRate ?? 0) > 0);
  }
  await strategy.onTick(candleTick(candle(5_000, prices[5]!, 1_000)));

  assert.equal(strategy.getDiagnostics().kama?.clampedRate, 0);
  assert.equal(strategy.getDiagnostics().kama?.lastSignalPrice, 104);
  assert.equal((await strategy.exitSignal())?.side, "long");
  assert.equal(await strategy.entrySignal(), null);
});

test("KAMA signal friction is nonnegative", () => {
  assert.equal(createPeakValleyStrategyConfig({ kamaSignalFriction: -1 }).kamaSignalFriction, 0);
});

test("selected strategy emits the searched exposure transitions", async () => {
  const config = createPeakValleyStrategyConfig({
    averagingRangesSec: [60],
    saturationSec: 0,
    sigmaMode: "static",
    buySigma: 0.01,
    sellSigma: 0.01,
  });
  const series = (index: number) => {
    const item = candle(index * 60_000, 100 + Math.sin(index / 12) * 4);
    item.volume = 10 + index % 11;
    return item;
  };
  const history = Array.from({ length: 720 }, (_, index) => series(index));
  const strategy = new PeakValleyStrategy({ config, getHistory: async () => history });
  const reference = referenceVolumeKama(config);
  let state = 0;
  let lastSignalPrice: number | null = null;
  for (const item of history.slice(-peakValleyKamaWarmupSamples(config))) {
    const rate = reference(item);
    const desired = rate > config.kamaRateThresholdHigh ? 1
      : rate < -config.kamaRateThresholdLow ? -1
      : config.derivativeClampMode === "hold" ? state : 0;
    if (
      desired !== state
      && (lastSignalPrice === null
        || Math.abs(item.close - lastSignalPrice) > lastSignalPrice * config.kamaSignalFriction)
    ) {
      state = desired;
      lastSignalPrice = item.close;
    }
  }
  await strategy.warmup();
  assert.equal(Math.sign(strategy.getDiagnostics().kama?.clampedRate ?? 0), state);
  assert.equal(strategy.getDiagnostics().kama?.lastSignalPrice, lastSignalPrice);
  let transitions = 0;

  for (let index = history.length; index < history.length + 300; index += 1) {
    const item = series(index);
    const rate = reference(item);
    await strategy.onTick({
      timestamp: item.closeTime,
      price: item.close,
      quantity: item.volume,
      candle: item,
    });
    assert.ok(Math.abs((strategy.getDiagnostics().kama?.rawRate ?? 0) - rate) < 1e-15);
    const desired = rate > config.kamaRateThresholdHigh ? 1
      : rate < -config.kamaRateThresholdLow ? -1
      : config.derivativeClampMode === "hold" ? state : 0;
    const next = desired !== state
      && (lastSignalPrice === null
        || Math.abs(item.close - lastSignalPrice) > lastSignalPrice * config.kamaSignalFriction)
      ? desired : state;
    if (next !== state) lastSignalPrice = item.close;
    const exitSignal = await strategy.exitSignal();
    const entrySignal = await strategy.entrySignal();
    assert.equal(exitSignal?.side ?? null, state === 1 && next !== 1 ? "long"
      : state === -1 && next !== -1 ? "short" : null);
    assert.equal(entrySignal?.side ?? null, next === 1 && state !== 1 ? "long"
      : next === -1 && state !== -1 ? "short" : null);
    if (next !== state) transitions += 1;
    state = next;
  }
  assert.ok(transitions > 0);
});

function candle(openTime: number, close: number, intervalMs = 60_000): TradingCandle {
  return {
    openTime,
    closeTime: openTime + intervalMs - 1,
    open: close,
    high: close,
    low: close,
    close,
    volume: 1,
  };
}

function candleTick(item: TradingCandle) {
  return {
    timestamp: item.closeTime,
    price: item.close,
    quantity: item.volume,
    candle: item,
  };
}

function referenceVolumeKama(config: ReturnType<typeof createPeakValleyStrategyConfig>) {
  const prices: number[] = [];
  const fastAlpha = config.kamaFastLen <= 1 ? config.kamaFastLen : 2 / (config.kamaFastLen + 1);
  const slowAlpha = config.kamaSlowLen <= 1 ? config.kamaSlowLen : 2 / (config.kamaSlowLen + 1);
  const volumeAlpha = 2 / (config.kamaVolumeLen + 1);
  let kama = 0;
  let volumeEma = 0;
  return (item: TradingCandle): number => {
    prices.push(item.close);
    if (prices.length > config.kamaErLen + 1) prices.shift();
    let efficiency = 0;
    if (prices.length === config.kamaErLen + 1) {
      const signal = Math.abs(prices.at(-1)! - prices[0]!);
      const noise = prices.slice(1).reduce((sum, price, index) =>
        sum + Math.abs(price - prices[index]!), 0);
      efficiency = noise > 0 ? Math.min(1, signal / noise) : 0;
    }
    const relativeVolume = item.volume > 0 && volumeEma > 0
      ? Math.min(config.kamaVolumeCap, item.volume / volumeEma)
      : 1;
    const effectiveEfficiency = Math.min(
      1,
      efficiency * Math.pow(relativeVolume, config.kamaVolumePower),
    );
    const alpha = Math.pow(
      slowAlpha + effectiveEfficiency * (fastAlpha - slowAlpha),
      config.kamaPower,
    );
    const previous = kama;
    kama = previous === 0 ? item.close : previous + alpha * (item.close - previous);
    if (item.volume > 0) {
      volumeEma = volumeEma === 0
        ? item.volume
        : volumeEma + volumeAlpha * (item.volume - volumeEma);
    }
    return previous > 0 && kama > 0 ? (kama - previous) / kama / 60 : 0;
  };
}
