import assert from "node:assert/strict";
import test from "node:test";
import { perfectMarginOracle } from "../src/perfect-margin-oracle.js";

test("perfect-margin oracle reconstructs its optimal three-state path", () => {
  const prices = [100, 110, 100];
  const candles = prices.map((price, index) => ({
    openTime: index * 60_000,
    closeTime: index * 60_000 + 59_999,
    open: price,
    high: price,
    low: price,
    close: price,
  }));
  const options = {
    startingQuote: 1_000,
    leverage: 1,
    friction: 0,
  };
  const result = perfectMarginOracle(candles, options);
  const sampled = perfectMarginOracle(candles, { ...options, maxPathCandles: 1 });
  const transitions = (points: typeof result.path.points) => points
    .filter((point) => point.action !== "hold")
    .map(({ time, price, fromState, state, action }) => ({ time, price, fromState, state, action }));

  assert.equal(result.netPnl > 190, true);
  assert.equal(result.path.points.some((point) => point.state === "long"), true);
  assert.equal(result.path.points.some((point) => point.state === "short"), true);
  assert.equal(result.path.points.some((point) => point.action === "switch"), true);
  assert.equal(result.path.eventMode, "ohlc");
  assert.deepEqual(transitions(sampled.path.points), transitions(result.path.points));
  assert.equal(sampled.path.points[0]?.fromState, "flat");
  for (const [index, point] of sampled.path.points.entries()) {
    if (index > 0) assert.equal(point.fromState, sampled.path.points[index - 1]?.state);
    const expected = point.fromState === point.state
      ? "hold"
      : point.fromState === "flat"
        ? "open"
        : point.state === "flat"
          ? "close"
          : "switch";
    assert.equal(point.action, expected);
  }
});

test("perfect-margin oracle can use causal candle-close observations", () => {
  const candles = [100, 110, 90, 105].map((price, index) => ({
    openTime: index * 60_000,
    closeTime: index * 60_000 + 59_999,
    open: price - 1,
    high: price + 5,
    low: price - 5,
    close: price,
  }));
  const result = perfectMarginOracle(candles, {
    startingQuote: 1_000,
    leverage: 1,
    friction: 0,
    eventMode: "close",
    maxPathCandles: 1,
  });

  assert.equal(result.path.eventMode, "close");
  assert.ok(result.path.points.every((point) =>
    candles.some((candle) => candle.closeTime === point.time && candle.close === point.price)));
  assert.ok(result.path.points.some((point) => point.action !== "hold"));
});

test("close-only oracle ignores unknowable intrabar paths", () => {
  const closeOnly = (spread: number) => perfectMarginOracle(
    [100, 110, 90, 105].map((close, index) => ({
      openTime: index * 60_000,
      closeTime: index * 60_000 + 59_999,
      open: close + spread / 2,
      high: close + spread,
      low: close - spread,
      close,
    })),
    { startingQuote: 1_000, leverage: 1, friction: 0.001, eventMode: "close" },
  );

  assert.deepEqual(closeOnly(5), closeOnly(500));
});

test("oracle prefers a direct switch over a rounding-equivalent flat bridge", () => {
  const candles = [100, 99, 99, 100].map((price, index) => ({
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: price,
    high: price,
    low: price,
    close: price,
  }));
  const result = perfectMarginOracle(candles, {
    startingQuote: 1,
    leverage: 1,
    friction: 0.0001,
    eventMode: "close",
  });

  assert.deepEqual([...result.stateCodes], [2, 1, 1, 1]);
  assert.deepEqual(
    result.path.points.filter((point) => point.action !== "hold").map((point) => point.action),
    ["open", "switch"],
  );
  assert.ok(Math.abs(result.netPnl - (0.01 + (100 / 99 - 1) - 0.0003)) < 1e-15);
});

test("oracle keeps economically useful initial flat states", () => {
  const candles = [100, 99.95, 101].map((price, index) => ({
    openTime: index * 1_000,
    closeTime: index * 1_000 + 999,
    open: price,
    high: price,
    low: price,
    close: price,
  }));
  const result = perfectMarginOracle(candles, {
    startingQuote: 1,
    leverage: 1,
    friction: 0.001,
    eventMode: "close",
  });

  assert.deepEqual([...result.stateCodes], [0, 1, 1]);
});
