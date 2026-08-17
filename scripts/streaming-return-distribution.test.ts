import assert from "node:assert/strict";
import test from "node:test";
import { summarizeReturnDistribution } from "./lib/log-return-distribution.js";
import {
  buildReturnDistributionChunk,
  StreamingReturnDistribution,
} from "./lib/streaming-return-distribution.js";

test("streamed moments and correlations match the exact in-memory calculation", () => {
  const values = Float64Array.from({ length: 10_000 }, (_, index) =>
    0.0002 * Math.sin(index * 0.31) + 0.0001 * Math.cos(index * 0.017));
  const exact = summarizeReturnDistribution(Array.from(values));
  const streamed = new StreamingReturnDistribution(500);
  streamed.merge(buildReturnDistributionChunk(values.slice(0, 4_000), 0, 4_000, 200));
  streamed.merge(buildReturnDistributionChunk(values.slice(4_000), 4_000, 10_000, 200));
  const result = streamed.summarize();

  assert.equal(result.observations, exact.observations);
  for (const key of [
    "meanBps",
    "meanAbsoluteBps",
    "standardDeviationBps",
    "skewness",
    "excessKurtosis",
    "positiveFraction",
    "zeroFraction",
    "lag1ReturnCorrelation",
    "lag1AbsoluteReturnCorrelation",
  ] as const) {
    const actual = result[key];
    const expected = exact[key];
    assert.ok(actual !== null && expected !== null);
    assert.ok(Math.abs(actual - expected) < 1e-10, `${key}: ${actual} != ${expected}`);
  }
});

test("streamed quantile sketch preserves central and tail shape", () => {
  const values = Float64Array.from({ length: 100_000 }, (_, index) => {
    const sign = index % 2 === 0 ? -1 : 1;
    return sign * ((index % 10_000) / 10_000) ** 3;
  });
  const exact = summarizeReturnDistribution(Array.from(values));
  const streamed = new StreamingReturnDistribution(800);
  for (let start = 0; start < values.length; start += 10_000) {
    streamed.merge(buildReturnDistributionChunk(
      values.slice(start, start + 10_000),
      start,
      start + 10_000,
      300,
    ));
  }
  const result = streamed.summarize();
  assert.ok(Math.abs(result.quantilesBps.p99 - exact.quantilesBps.p99) < 5);
  assert.ok(Math.abs(result.quantilesBps.p01 - exact.quantilesBps.p01) < 5);
  assert.ok(Math.abs(
    (result.tailMass3SigmaVsGaussian ?? 0) - (exact.tailMass3SigmaVsGaussian ?? 0),
  ) < 0.05);
});
