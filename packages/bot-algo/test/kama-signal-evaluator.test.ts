import assert from "node:assert/strict";
import test from "node:test";
import {
  alignVwKamaTransitions,
  columnarVwKamaCandles,
  evaluateVwKamaOracle,
  perfectMarginOracle,
  prepareVwKamaOracle,
  signalBeyondFriction,
  type Candle,
  type VwKamaTransition,
} from "../src/index.js";

function transition(
  time: number,
  fromState: VwKamaTransition["fromState"],
  state: VwKamaTransition["state"],
): VwKamaTransition {
  const exposure = { short: -1, flat: 0, long: 1 };
  return {
    time,
    price: 100,
    side: exposure[state] > exposure[fromState] ? "buy" : "sell",
    fromState,
    state,
    fromExposure: exposure[fromState],
    exposure: exposure[state],
    sizeFraction: Math.abs(exposure[state]),
    quality: 1,
    acceleration: 0,
    overextension: 0,
    emaRate: 0,
    rsi: 50,
    dmi: 0,
    adx: 0,
    meanDistance: 0,
    matchedTime: null,
    lagMs: null,
    timingCredit: 0,
  };
}

test("VW-KAMA evaluator produces causal transitions and bounded chart traces", () => {
  const closes = [100, 101, 102, 101, 100, 101];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: index === 0 ? close : closes[index - 1]!,
    high: Math.max(close, index === 0 ? close : closes[index - 1]!),
    low: Math.min(close, index === 0 ? close : closes[index - 1]!),
    close,
    volume: 1,
    closed: true,
  }));

  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[1]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
    maxPoints: 2,
  });

  assert.equal(result.candleCount, 5);
  assert.deepEqual(
    result.candidateTransitions.map(({ time, side, fromState, state }) => ({
      time,
      side,
      fromState,
      state,
    })),
    [
      { time: 1_999, side: "buy", fromState: "flat", state: "long" },
      { time: 3_999, side: "sell", fromState: "long", state: "short" },
      { time: 5_999, side: "buy", fromState: "short", state: "long" },
    ],
  );
  assert.equal(result.annotations.length, 3);
  assert.deepEqual(result.annotations.map((annotation) => annotation.signalState), ["long", "short", "long"]);
  assert.ok(result.kamaSeries.points.length <= 3);
  assert.ok(result.statePoints.length <= 3);
  assert.equal(result.indicatorPoints.length, result.kamaSeries.points.length);
  assert.ok(result.indicatorPoints.every((point) =>
    Number.isFinite(point.kama)
    && Number.isFinite(point.kamaRate)
    && Number.isFinite(point.kamaRateRaw)
    && Number.isFinite(point.threshold)
    && Array.isArray(point.rejectionReasons)
    && point.efficiencyRatio >= 0
    && point.efficiencyRatio <= 1
    && point.effectiveEfficiencyRatio >= 0
    && point.effectiveEfficiencyRatio <= 1
    && Number.isFinite(point.volume)
    && Number.isFinite(point.volumeAverage)
    && point.volume >= 0
    && point.volumeAverage >= 0
    && point.rsi >= 0
    && point.rsi <= 100
    && point.adx >= 0
    && point.adx <= 100));
  assert.ok(result.metrics.score >= 0 && result.metrics.score <= 1);
  assert.equal(
    result.metrics.score,
    result.metrics.f1 * 0.2
      + result.metrics.exposureAgreement * 0.6
      + result.metrics.signalCleanliness * 0.2,
  );
  assert.ok(result.metrics.exposureAgreement >= 0 && result.metrics.exposureAgreement <= 1);
  assert.equal(result.metrics.matchedCount + result.metrics.extraSignalCount, result.metrics.signalCount);
  assert.equal(
    result.metrics.noiseSignalRatio,
    result.metrics.matchedCount > 0
      ? result.metrics.extraSignalCount / result.metrics.matchedCount
      : result.metrics.extraSignalCount > 0 ? null : 0,
  );
  assert.equal(result.metrics.matchedCount + result.metrics.missedOracleCount, result.metrics.oracleCount);
  assert.ok(result.candidateTransitions.every((item) =>
    item.matchedTime === null || item.lagMs === item.time - item.matchedTime));
  assert.ok(result.candidateTransitions.every((item) =>
    [item.quality, item.acceleration, item.overextension, item.emaRate, item.rsi, item.dmi, item.adx]
      .every(Number.isFinite)
    && item.rsi >= 0 && item.rsi <= 100
    && item.adx >= 0 && item.adx <= 100));
});

test("VW-KAMA log rate is invariant to the asset price scale", () => {
  const evaluate = (scale: number) => {
    const candles = [100, 102, 101].map((value, index): Candle => ({
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: index * 1_000,
      closeTime: index * 1_000 + 999,
      open: value * scale,
      high: value * scale,
      low: value * scale,
      close: value * scale,
      volume: 1,
      closed: true,
    }));
    return evaluateVwKamaOracle(candles, {
      intervalMs: 1_000,
      scoreStartTime: candles[0]!.openTime,
      parameters: {
        efficiencyMs: 1_000,
        fastMs: 1_000,
        slowMs: 1_000,
        power: 1,
        volumeMs: 1_000,
        volumeCap: 1,
        volumePower: 0,
        rateMode: "log",
        rateEmaMs: 2_000,
        deadbandBpsHour: 0,
        deadbandMode: "hold",
      },
      oracleFriction: 0,
      matchWindowMs: 10_000,
      timingHalfLifeMs: 1_000,
      warmupMultiple: 1,
      maxPoints: 10,
    }).indicatorPoints;
  };

  const original = evaluate(1);
  const rescaled = evaluate(1_000);
  assert.equal(original.length, 3);
  assert.ok(Math.abs(original[1]!.kamaRateRaw - Math.log(1.02) * 36_000_000) < 1e-6);
  assert.ok(original.every((point, index) =>
    Math.abs(point.kamaRateRaw - rescaled[index]!.kamaRateRaw) < 1e-6
    && Math.abs(point.kamaRate - rescaled[index]!.kamaRate) < 1e-6));
});

test("VW-KAMA mean-reversion KAMA uses volume-weighted efficiency", () => {
  const evaluate = (lastVolume: number) => {
    const candles = [
      { close: 100, volume: 10 },
      { close: 102, volume: 20 },
      { close: 101, volume: lastVolume },
    ].map((item, index): Candle => ({
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: index * 1_000,
      closeTime: index * 1_000 + 999,
      open: item.close,
      high: item.close,
      low: item.close,
      close: item.close,
      volume: item.volume,
      closed: true,
    }));
    return evaluateVwKamaOracle(candles, {
      intervalMs: 1_000,
      scoreStartTime: candles[0]!.openTime,
      parameters: {
        efficiencyMs: 2_000,
        efficiencyVolumeEmaMs: 3_000,
        efficiencyVolumePower: 2,
        fastMs: 2_000,
        slowMs: 20_000,
        power: 1,
        volumeMs: 3_000,
        volumeCap: 4,
        volumePower: 0,
        deadbandBpsHour: 0,
        deadbandMode: "hold",
        meanReversionEfficiencyMs: 2_000,
        meanReversionFastMs: 2_000,
        meanReversionSlowMs: 20_000,
        meanReversionVolatilityMs: 20_000,
        meanReversionSuppressionThreshold: 1,
        meanReversionReversalThreshold: 0,
      },
      oracleFriction: 0,
      matchWindowMs: 10_000,
      timingHalfLifeMs: 1_000,
      warmupMultiple: 1,
    }).indicatorPoints.at(-1)!.meanReversionKama;
  };

  assert.notEqual(evaluate(40), evaluate(5));
});

test("VW-KAMA rate EMA drives the same band-transition pipeline", () => {
  const candles = [100, 110, 100].map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: close,
    high: close,
    low: close,
    close,
    volume: 1,
    closed: true,
  }));
  const evaluate = (rateEmaMs: number) => evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[0]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      rateMode: "log",
      rateEmaMs,
      deadbandBpsHour: 1,
      deadbandMode: "hold",
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
    maxPoints: 10,
  });

  assert.deepEqual(evaluate(1_000).candidateTransitions.map((item) => item.state), ["long", "short"]);
  assert.deepEqual(evaluate(10_000).candidateTransitions.map((item) => item.state), ["long"]);
});

test("VW-KAMA streaming and shared-columnar evaluation are exactly equivalent", () => {
  const candles = Array.from({ length: 500 }, (_, index): Candle => {
    const close = 100 + Math.sin(index / 13) * 4 + Math.cos(index / 37) * 2 + index * 0.003;
    return {
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: index * 1_000,
      closeTime: index * 1_000 + 999,
      open: close - Math.sin(index) * 0.2,
      high: close + 0.5,
      low: close - 0.5,
      close,
      volume: 1 + index % 17,
      closed: true,
    };
  });
  const scoreStartIndex = 200;
  const oracle = perfectMarginOracle(candles, {
    startingQuote: 1,
    leverage: 1,
    friction: 0.001,
    eventMode: "close",
    maxPathCandles: 1,
  });
  const options = {
    intervalMs: 1_000,
    scoreStartTime: candles[scoreStartIndex]!.openTime,
    scoreStartIndex,
    parameters: {
      efficiencyMs: 8_000,
      efficiencyVolumeEmaMs: 21_000,
      efficiencyVolumePower: 0.4,
      fastMs: 3_000,
      slowMs: 40_000,
      power: 1.7,
      volumeMs: 12_000,
      volumeCap: 4,
      volumePower: 0.6,
      deadbandBpsHour: 20,
      deadbandMode: "hysteresis" as const,
      hysteresisReleaseRatio: 0.3,
      confirmationMix: 0.7,
      confirmationMinQuality: 0.2,
      confirmationAccelerationLookbackMs: 15_000,
      confirmationDistanceLookbackMs: 30_000,
      confirmationAccelerationWeight: 0.4,
      confirmationDistanceWeight: 0.5,
      confirmationEmaMs: 45_000,
      confirmationEmaWeight: 0.3,
      confirmationRsiMs: 14_000,
      confirmationRsiWeight: 0.2,
      confirmationDmiMs: 14_000,
      confirmationDmiWeight: 0.2,
      meanReversionSuppressionThreshold: 1,
      meanReversionEfficiencyMs: 20_000,
      meanReversionFastMs: 30_000,
      meanReversionSlowMs: 60_000,
      meanReversionVolatilityMs: 60_000,
      meanReversionReversalThreshold: 1.5,
    },
    oracleFriction: 0.001,
    matchWindowMs: 60_000,
    timingHalfLifeMs: 10_000,
    warmupMultiple: 2,
    oracleResult: oracle,
  };
  const traced = evaluateVwKamaOracle(candles, options);
  const streaming = evaluateVwKamaOracle(candles, { ...options, includeTrace: false });
  const columns = columnarVwKamaCandles(candles, true);
  const preparedOracle = prepareVwKamaOracle(columns, scoreStartIndex, oracle, true);
  const columnar = evaluateVwKamaOracle(columns, {
    ...options,
    oracleResult: undefined,
    preparedOracle,
    includeTrace: false,
  });

  assert.deepEqual(streaming.metrics, traced.metrics);
  assert.deepEqual(streaming.candidateTransitions, traced.candidateTransitions);
  assert.deepEqual(columnar.metrics, traced.metrics);
  assert.deepEqual(columnar.candidateTransitions, traced.candidateTransitions);
  assert.ok(columns.close.buffer instanceof SharedArrayBuffer);
  assert.ok(preparedOracle.stateCodes.buffer instanceof SharedArrayBuffer);
});

test("VW-KAMA mean-reversion regime reverses a sufficiently extended local trend", () => {
  const closes = [...Array.from({ length: 40 }, () => 100), 130];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: index === 0 ? close : closes[index - 1]!,
    high: Math.max(close, index === 0 ? close : closes[index - 1]!),
    low: Math.min(close, index === 0 ? close : closes[index - 1]!),
    close,
    volume: 1,
    closed: true,
  }));
  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[20]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
      meanReversionSuppressionThreshold: 0.5,
      meanReversionEfficiencyMs: 20_000,
      meanReversionFastMs: 20_000,
      meanReversionSlowMs: 20_000,
      meanReversionVolatilityMs: 20_000,
      meanReversionReversalThreshold: 1,
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });

  const signal = result.candidateTransitions.at(-1)!;
  assert.equal(signal.state, "short");
  assert.ok(signal.meanDistance > 1);
});

test("VW-KAMA mean-reversion suppression zone consumes an extended trend edge", () => {
  const closes = [...Array.from({ length: 40 }, () => 100), 130];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: index === 0 ? close : closes[index - 1]!,
    high: Math.max(close, index === 0 ? close : closes[index - 1]!),
    low: Math.min(close, index === 0 ? close : closes[index - 1]!),
    close,
    volume: 1,
    closed: true,
  }));
  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[20]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
      meanReversionSuppressionThreshold: 0.5,
      meanReversionEfficiencyMs: 20_000,
      meanReversionFastMs: 20_000,
      meanReversionSlowMs: 20_000,
      meanReversionVolatilityMs: 20_000,
      meanReversionReversalThreshold: 100,
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });

  assert.equal(result.candidateTransitions.length, 0);
  const distance = Math.abs(result.indicatorPoints.at(-1)!.meanDistance);
  assert.ok(distance >= 0.5 && distance < 100);
});

test("adaptive VW-KAMA threshold suppresses noisy rate reversals causally", () => {
  const candles = Array.from({ length: 40 }, (_, index): Candle => {
    const close = index % 2 === 0 ? 100 : 101;
    return {
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: index * 1_000,
      closeTime: index * 1_000 + 999,
      open: close,
      high: close,
      low: close,
      close,
      volume: 1,
      closed: true,
    };
  });
  const options = {
    intervalMs: 1_000,
    scoreStartTime: 10_000,
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold" as const,
      thresholdLookbackMs: 5_000,
      thresholdNoiseMultiplier: 4,
    },
  };
  const fixed = evaluateVwKamaOracle(candles, {
    ...options,
    parameters: { ...options.parameters, thresholdNoiseMultiplier: 0 },
  });
  const adaptive = evaluateVwKamaOracle(candles, {
    ...options,
    parameters: { ...options.parameters, thresholdNoiseMultiplier: 4 },
  });

  assert.ok(adaptive.metrics.signalCount < fixed.metrics.signalCount);
  assert.ok(adaptive.indicatorPoints.some((point) => point.threshold > 0));

  const inverse = evaluateVwKamaOracle(candles, {
    ...options,
    scoreStartTime: 0,
    parameters: {
      ...options.parameters,
      thresholdNoiseResponse: "inverse",
      thresholdNoiseMultiplier: 0,
      thresholdInverseMaxBpsHour: 100,
      thresholdInverseNoiseScaleBpsHour: 100,
    },
  });
  assert.equal(inverse.indicatorPoints[0]!.threshold, 100);
  assert.ok(inverse.indicatorPoints.slice(1).some((point) => point.threshold > 0 && point.threshold < 100));
});

test("inactive threshold and confirmation lookbacks do not affect search scores", () => {
  const candles = Array.from({ length: 240 }, (_, index): Candle => {
    const close = 100 + Math.sin(index / 7) * 3 + index * 0.01;
    return {
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: index * 1_000,
      closeTime: index * 1_000 + 999,
      open: close,
      high: close,
      low: close,
      close,
      volume: 1,
      closed: true,
    };
  });
  const parameters = {
    efficiencyMs: 5_000,
    fastMs: 2_000,
    slowMs: 10_000,
    power: 1,
    volumeMs: 5_000,
    volumeCap: 1,
    volumePower: 0,
    deadbandBpsHour: 0,
    deadbandMode: "hold" as const,
    thresholdNoiseMultiplier: 0,
    confirmationMix: 0,
    confirmationAccelerationWeight: 5,
    confirmationDistanceWeight: 5,
    confirmationEmaWeight: 5,
    confirmationEmaGateStrength: 0,
    confirmationRsiWeight: 5,
    confirmationDmiWeight: 5,
  };
  const evaluate = (lookbackMs: number, includeTrace = false) => evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: 120_000,
    parameters: {
      ...parameters,
      thresholdLookbackMs: lookbackMs,
      confirmationAccelerationLookbackMs: lookbackMs,
      confirmationDistanceLookbackMs: lookbackMs,
      confirmationEmaMs: lookbackMs,
      confirmationRsiMs: lookbackMs,
      confirmationDmiMs: lookbackMs,
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 3,
    includeTrace,
  });
  const short = evaluate(1_000);
  const long = evaluate(80_000);
  const tracedLong = evaluate(80_000, true);

  assert.deepEqual(long.metrics, short.metrics);
  assert.deepEqual(long.candidateTransitions, short.candidateTransitions);
  assert.deepEqual(tracedLong.metrics, long.metrics);
  assert.deepEqual(
    tracedLong.candidateTransitions.map(({ time, fromState, state, sizeFraction, quality }) =>
      ({ time, fromState, state, sizeFraction, quality })),
    long.candidateTransitions.map(({ time, fromState, state, sizeFraction, quality }) =>
      ({ time, fromState, state, sizeFraction, quality })),
  );
});

test("VW-KAMA confirmation combines quality with an optional hard floor", () => {
  const candles = Array.from({ length: 40 }, (_, index): Candle => {
    const close = index % 2 === 0 ? 100 : 101;
    return {
      symbol: "BTCUSDT",
      interval: "1s",
      openTime: index * 1_000,
      closeTime: index * 1_000 + 999,
      open: close,
      high: close,
      low: close,
      close,
      volume: 1,
      closed: true,
    };
  });
  const options = {
    intervalMs: 1_000,
    scoreStartTime: 10_000,
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold" as const,
      confirmationMinQuality: 1,
      confirmationAccelerationLookbackMs: 5_000,
      confirmationDistanceLookbackMs: 5_000,
      confirmationAccelerationWeight: 0,
      confirmationDistanceWeight: 0,
      confirmationBias: 0,
    },
  };
  const disabled = evaluateVwKamaOracle(candles, {
    ...options,
    parameters: { ...options.parameters, confirmationMix: 0 },
  });
  const filtered = evaluateVwKamaOracle(candles, {
    ...options,
    parameters: { ...options.parameters, confirmationMix: 1 },
  });

  assert.ok(disabled.metrics.signalCount > 0);
  assert.ok(disabled.candidateTransitions.every((item) => item.quality === 1));
  assert.equal(filtered.metrics.signalCount, 0);
});

test("VW-KAMA hysteresis retains direction inside its outer threshold", () => {
  const closes = [100, 101, 101.0001, 101.0002];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: close,
    high: close,
    low: close,
    close,
    volume: 1,
    closed: true,
  }));
  const parameters = {
    efficiencyMs: 1_000,
    fastMs: 1_000,
    slowMs: 1_000,
    power: 1,
    volumeMs: 1_000,
    volumeCap: 1,
    volumePower: 0,
    deadbandBpsHour: 10_000,
    hysteresisReleaseRatio: 0.001,
  };
  const evaluate = (deadbandMode: "flat" | "hysteresis") => evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[1]!.openTime,
    parameters: { ...parameters, deadbandMode },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });

  assert.deepEqual(evaluate("flat").candidateTransitions.map((item) => item.state), ["long", "flat"]);
  assert.deepEqual(evaluate("hysteresis").candidateTransitions.map((item) => item.state), ["long"]);
});

test("VW-KAMA hard EMA gate closes a countertrend flip to flat", () => {
  const closes = [100, 110, 120, 119, 118];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: close,
    high: close,
    low: close,
    close,
    volume: 1,
    closed: true,
  }));
  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[1]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
      confirmationEmaMs: 10_000,
      confirmationEmaThresholdBpsHour: 0,
      confirmationEmaGateStrength: 1,
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });

  assert.deepEqual(result.candidateTransitions.map((item) => item.state), ["long", "flat"]);
  assert.ok(result.candidateTransitions.at(-1)!.emaRate > 0);
  assert.ok(result.indicatorPoints.some((point) =>
    point.signalIntent === "short" && point.rejectionReasons.includes("ema-hard-gate")));
});

test("VW-KAMA partial sizing is marked against the current price", () => {
  const closes = [100, 110, 120, 130];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: close,
    high: close,
    low: close,
    close,
    volume: 1,
    closed: true,
  }));
  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[1]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
      buyMaxFraction: 0.5,
      sellMaxFraction: 0.5,
      buySizingSigmaBpsHour: 1e12,
      sellSizingSigmaBpsHour: 1e12,
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });

  const exposures = result.statePoints.map((point) => point.candidateExposure);
  assert.ok(Math.abs(exposures[0]! - 0.5) < 1e-6);
  assert.ok(exposures[1]! > exposures[0]!);
  assert.ok(exposures[2]! > exposures[1]!);
  assert.ok(result.candidateTransitions.every((item) => item.sizeFraction <= 0.5));
});

test("VW-KAMA confidence agreement stays fixed between signals", () => {
  const closes = [100, 110, 120, 130];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: close,
    high: close,
    low: close,
    close,
    volume: 1,
    closed: true,
  }));
  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[1]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
      agreementMode: "confidence",
      buyMaxFraction: 0.5,
      sellMaxFraction: 0.5,
      buySizingSigmaBpsHour: 1e12,
      sellSizingSigmaBpsHour: 1e12,
    },
    oracleFriction: 0,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });

  const exposures = result.statePoints.map((point) => point.candidateExposure);
  assert.ok(exposures.every((value) => Math.abs(value - 0.5) < 1e-6));
  assert.ok(result.metrics.exposureAgreement >= 0.49 && result.metrics.exposureAgreement <= 0.51);
});

test("VW-KAMA alignment is chronological and one-to-one", () => {
  const candidates = [
    transition(8, "flat", "long"),
    transition(9, "short", "long"),
  ];
  const oracle = [
    transition(0, "flat", "long"),
    transition(10, "short", "long"),
  ];
  const alignment = alignVwKamaTransitions(candidates, oracle, {
    matchWindowMs: 20,
    timingHalfLifeMs: 10,
  });

  assert.deepEqual(
    alignment.matches.map(({ candidateIndex, oracleIndex }) => ({ candidateIndex, oracleIndex })),
    [
      { candidateIndex: 0, oracleIndex: 0 },
      { candidateIndex: 1, oracleIndex: 1 },
    ],
  );
  assert.deepEqual(candidates.map(({ matchedTime, lagMs }) => ({ matchedTime, lagMs })), [
    { matchedTime: 0, lagMs: 8 },
    { matchedTime: 10, lagMs: -1 },
  ]);
  assert.deepEqual(oracle.map(({ matchedTime, lagMs }) => ({ matchedTime, lagMs })), [
    { matchedTime: 8, lagMs: -8 },
    { matchedTime: 9, lagMs: 1 },
  ]);
});

test("VW-KAMA alignment cannot reuse one oracle transition", () => {
  const candidates = [
    transition(90, "flat", "long"),
    transition(99, "short", "long"),
  ];
  const oracle = [transition(100, "flat", "long")];
  const alignment = alignVwKamaTransitions(candidates, oracle, {
    matchWindowMs: 20,
    timingHalfLifeMs: 10,
  });

  assert.deepEqual(
    alignment.matches.map(({ candidateIndex, oracleIndex }) => ({ candidateIndex, oracleIndex })),
    [{ candidateIndex: 1, oracleIndex: 0 }],
  );
  assert.equal(candidates[0]!.matchedTime, null);
  assert.equal(candidates[1]!.matchedTime, 100);
  assert.equal(oracle[0]!.matchedTime, 99);
});

test("VW-KAMA alignment matches the resulting state rather than order side", () => {
  const candidate = transition(100, "long", "flat");
  const oracle = transition(100, "long", "short");
  assert.equal(candidate.side, oracle.side);

  const alignment = alignVwKamaTransitions([candidate], [oracle], {
    matchWindowMs: 1_000,
    timingHalfLifeMs: 100,
  });

  assert.equal(alignment.matches.length, 0);
  assert.equal(candidate.matchedTime, null);
  assert.equal(oracle.matchedTime, null);
});

test("VW-KAMA alignment breaks zero-credit ties by pair count then absolute lag", () => {
  const candidates = [
    transition(3_000, "flat", "long"),
    transition(4_000, "long", "short"),
  ];
  const oracle = [
    transition(0, "flat", "long"),
    transition(1_000, "short", "long"),
    transition(2_000, "long", "short"),
  ];
  const alignment = alignVwKamaTransitions(candidates, oracle, {
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1,
  });

  assert.equal(alignment.credit, 0);
  assert.deepEqual(
    alignment.matches.map(({ candidateIndex, oracleIndex }) => ({ candidateIndex, oracleIndex })),
    [
      { candidateIndex: 0, oracleIndex: 1 },
      { candidateIndex: 1, oracleIndex: 2 },
    ],
  );
});

test("VW-KAMA consumes a rejected band edge instead of emitting it late", () => {
  const closes = [100, 102, 101, 100.5, 100, 101, 102, 103];
  const candles = closes.map((close, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: index === 0 ? close : closes[index - 1]!,
    high: Math.max(close, index === 0 ? close : closes[index - 1]!),
    low: Math.min(close, index === 0 ? close : closes[index - 1]!),
    close,
    volume: 1,
    closed: true,
  }));

  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[1]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
    },
    oracleFriction: 0.015,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });

  assert.deepEqual(
    result.candidateTransitions.map(({ time, price, state }) => ({ time, price, state })),
    [
      { time: 1_999, price: 102, state: "long" },
    ],
  );

  const withoutSignalFriction = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: candles[1]!.openTime,
    parameters: {
      efficiencyMs: 1_000,
      fastMs: 1_000,
      slowMs: 1_000,
      power: 1,
      volumeMs: 1_000,
      volumeCap: 1,
      volumePower: 0,
      deadbandBpsHour: 0,
      deadbandMode: "hold",
      signalFrictionFraction: 0,
    },
    oracleFriction: 0.015,
    matchWindowMs: 10_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
  });
  assert.deepEqual(
    withoutSignalFriction.candidateTransitions.map(({ time, state }) => ({ time, state })),
    [
      { time: 1_999, state: "long" },
      { time: 2_999, state: "short" },
      { time: 5_999, state: "long" },
    ],
  );
  assert.ok(result.indicatorPoints.some((point) =>
    point.signalIntent !== null && point.rejectionReasons.includes("signal-friction")));
  assert.equal(result.indicatorPoints[0]!.signalFrictionLower, 102 * (1 - 0.015));
  assert.equal(result.indicatorPoints[0]!.signalFrictionUpper, 102 * (1 + 0.015));
  assert.ok(withoutSignalFriction.indicatorPoints.every((point) =>
    !point.rejectionReasons.includes("signal-friction")));
});

test("price signal memory requires movement strictly past friction", () => {
  assert.equal(signalBeyondFriction(101, 100, 0.01), false);
  assert.equal(signalBeyondFriction(99, 100, 0.01), false);
  assert.equal(signalBeyondFriction(101.01, 100, 0.01), true);
  assert.equal(signalBeyondFriction(100, 100, 0), true);
});

test("VW-KAMA evaluator can omit visualization data during parameter search", () => {
  const candles = Array.from({ length: 20 }, (_, index): Candle => ({
    symbol: "BTCUSDT",
    interval: "1s",
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: 100 + index,
    high: 101 + index,
    low: 100 + index,
    close: 101 + index,
    volume: 1 + index,
    closed: true,
  }));
  const result = evaluateVwKamaOracle(candles, {
    intervalMs: 1_000,
    scoreStartTime: 5_000,
    parameters: {
      efficiencyMs: 2_000,
      fastMs: 1_000,
      slowMs: 3_000,
      power: 2,
      volumeMs: 3_000,
      volumeCap: 3,
      volumePower: 1,
      deadbandBpsHour: 1,
      deadbandMode: "hold",
    },
    oracleFriction: 0.001,
    matchWindowMs: 5_000,
    timingHalfLifeMs: 1_000,
    warmupMultiple: 1,
    includeTrace: false,
  });

  assert.equal(result.kamaSeries.points.length, 0);
  assert.equal(result.indicatorPoints.length, 0);
  assert.equal(result.annotations.length, 0);
  assert.equal(result.candidatePath.points.length, 0);
  assert.equal(result.statePoints.length, 0);
  assert.ok(result.metrics.signalCount >= 0);
});
