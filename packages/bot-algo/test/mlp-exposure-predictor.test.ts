import assert from "node:assert/strict";
import test from "node:test";
import {
  MLP_CANDLE_FEATURE_COUNT,
  MLP_INPUT_FEATURE_COUNT,
  MLP_OUTPUT_PARAMETER_COUNT,
  encodeMlpCandleWindow,
  encodeMlpStateInputs,
  predictMlpConditionalDistribution,
  validateMlpModelManifest,
} from "../src/mlp-exposure-predictor.js";

test("MLP feature schema has the documented shape", () => {
  assert.equal(MLP_INPUT_FEATURE_COUNT, 909);
  assert.equal(MLP_OUTPUT_PARAMETER_COUNT, 8);
});

test("MLP candle features are causal, padded, and scale invariant", () => {
  const candles = Array.from({ length: 8 }, (_, index) => ({
    open: 100 + index,
    high: 102 + index,
    low: 99 + index,
    close: 101 + index,
    volume: 10 + index,
  }));
  const first = encodeMlpCandleWindow(candles.slice(0, 7), 4);
  const withFuture = encodeMlpCandleWindow(candles, 4);
  const scaled = encodeMlpCandleWindow(candles.slice(0, 7).map((candle) => ({
    ...candle,
    open: candle.open * 3,
    high: candle.high * 3,
    low: candle.low * 3,
    close: candle.close * 3,
    volume: candle.volume * 5,
  })), 4);
  assert.deepEqual(Array.from(first), Array.from(scaled));
  assert.notDeepEqual(Array.from(first), Array.from(withFuture));

  const padded = encodeMlpCandleWindow(candles.slice(0, 1), 4);
  assert.deepEqual(Array.from(padded.slice(0, 3 * MLP_CANDLE_FEATURE_COUNT)), Array(12).fill(0));
});

test("MLP state inputs use a stable finite ordering", () => {
  const encoded = encodeMlpStateInputs({
    feeRate: 1,
    minimumUsableExposure: 2,
    maximumUsableExposure: 3,
    minimumEffectiveExposure: 4,
    maximumEffectiveExposure: 5,
    quoteLendRate: 6,
    quoteBorrowRate: 7,
    assetBorrowRate: 8,
  });
  assert.deepEqual(Array.from(encoded), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test("MLP raw outputs always decode to a normalized valid policy", () => {
  const grid = Float64Array.from({ length: 21 }, (_, index) => -10 + index);
  const prediction = predictMlpConditionalDistribution(
    grid,
    -3,
    new Float64Array(8),
    { latentLower: -25, latentUpper: 25, friction: 0.001, temperature: 0.01 },
  );
  assert.ok(prediction.conditionalParameters.c1 < prediction.conditionalParameters.c2);
  assert.ok(prediction.conditionalParameters.cutoffLower < 0);
  assert.ok(prediction.conditionalParameters.cutoffUpper > 0);
  assert.ok(Math.abs(prediction.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(Number.isFinite(prediction.meanExposure));
  assert.ok(grid.includes(prediction.optimalExposure));
});

test("MLP manifest validation rejects architecture drift", () => {
  const manifest = {
    id: "test",
    label: "Test",
    createdAt: new Date(0).toISOString(),
    featureSchemaVersion: 4,
    inputFeatureCount: 909,
    outputParameterCount: 8,
    hiddenLayerCount: 16,
    hiddenWidth: 1_024,
    modelFile: "model.onnx",
  };
  assert.doesNotThrow(() => validateMlpModelManifest(manifest));
  assert.doesNotThrow(() => validateMlpModelManifest({ ...manifest, outputParameterCount: 6 }));
  assert.throws(() => validateMlpModelManifest({ ...manifest, hiddenWidth: 512 }));
});
