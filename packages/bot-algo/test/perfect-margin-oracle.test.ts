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
