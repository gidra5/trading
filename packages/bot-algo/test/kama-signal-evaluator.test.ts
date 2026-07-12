import assert from "node:assert/strict";
import test from "node:test";
import {
  alignVwKamaTransitions,
  evaluateVwKamaOracle,
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
  assert.ok(result.metrics.score >= 0 && result.metrics.score <= 1);
  assert.equal(
    result.metrics.score,
    result.metrics.f1 * 0.6
      + result.metrics.exposureAgreement * 0.3
      + result.metrics.signalCleanliness * 0.1,
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
    parameters: { ...options.parameters, thresholdMode: "static" },
  });
  const adaptive = evaluateVwKamaOracle(candles, {
    ...options,
    parameters: { ...options.parameters, thresholdMode: "adaptive" },
  });

  assert.ok(adaptive.metrics.signalCount < fixed.metrics.signalCount);
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

test("VW-KAMA signal memory spaces transitions from the last accepted signal price", () => {
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
      { time: 4_999, price: 100, state: "short" },
      { time: 6_999, price: 102, state: "long" },
    ],
  );
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
  assert.equal(result.annotations.length, 0);
  assert.equal(result.candidatePath.points.length, 0);
  assert.equal(result.statePoints.length, 0);
  assert.ok(result.metrics.signalCount >= 0);
});
