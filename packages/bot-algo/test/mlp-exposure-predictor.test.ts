import assert from "node:assert/strict";
import test from "node:test";
import {
  MLP_CANDLE_FEATURE_COUNT,
  MLP_FEATURE_SCHEMA_VERSION,
  MLP_INPUT_FEATURE_COUNT,
  MLP_OUTPUT_ACTION_COUNT,
  encodeMlpCandleWindow,
  predictMlpDistribution,
  validateMlpModelManifest,
} from "../src/mlp-exposure-predictor.js";

test("MLP feature schema has the documented shape", () => {
  assert.equal(MLP_FEATURE_SCHEMA_VERSION, 5);
  assert.equal(MLP_INPUT_FEATURE_COUNT, 901);
  assert.equal(MLP_OUTPUT_ACTION_COUNT, 255);
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

test("MLP action logits interpolate to a normalized valid policy", () => {
  const grid = Float64Array.from({ length: 21 }, (_, index) => -10 + index);
  const modelGrid = Float64Array.from({ length: 255 }, (_, index) => -25 + index * 50 / 254);
  const logits = Float64Array.from(modelGrid, (action) => -(((action - 3) / 2) ** 2));
  const prediction = predictMlpDistribution(
    grid,
    logits,
    modelGrid,
  );
  assert.ok(Math.abs(prediction.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(Number.isFinite(prediction.meanExposure));
  assert.ok(grid.includes(prediction.optimalExposure));
});

test("MLP manifest validation rejects architecture drift", () => {
  const manifest = {
    id: "test",
    label: "Test",
    createdAt: new Date(0).toISOString(),
    featureSchemaVersion: 5,
    inputFeatureCount: 901,
    outputRepresentation: "base-action-logits" as const,
    outputActionCount: 255,
    actionGrid: Array.from({ length: 255 }, (_, index) => -250 + index * 500 / 254),
    hiddenLayerCount: 16,
    hiddenWidth: 1_024,
    predictionDelayMs: 60_000,
    modelFile: "model.onnx",
  };
  assert.doesNotThrow(() => validateMlpModelManifest(manifest));
  assert.throws(() => validateMlpModelManifest({ ...manifest, outputActionCount: 127 }));
  assert.throws(() => validateMlpModelManifest({
    ...manifest,
    featureSchemaVersion: 4,
    inputFeatureCount: 909,
  }));
  assert.throws(() => validateMlpModelManifest({ ...manifest, inputFeatureCount: 909 }));
  assert.throws(() => validateMlpModelManifest({ ...manifest, hiddenWidth: 512 }));
});
