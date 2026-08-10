import assert from "node:assert/strict";
import test from "node:test";
import {
  crossValidateKronosReturnCalibrations,
  fitKronosReturnCalibration,
  predictKronosReturn,
  validateKronosReturnCalibration,
  type KronosReturnCalibrationExample,
} from "./kronos-return-calibrator.js";

function example(
  episodeId: string,
  decisionTime: number,
  signal: number,
): KronosReturnCalibrationExample {
  const grid = Float64Array.from([-1, 0, 1]);
  const probabilities = signal < 0
    ? Float32Array.from([0.7, 0.2, 0.1])
    : Float32Array.from([0.1, 0.2, 0.7]);
  const row = {
    horizonLogReturnMean: signal,
    horizonLogReturnMedian: signal * 0.9,
    horizonLogReturnStd: 0.002 + Math.abs(signal) * 0.1,
    horizonUpProbability: signal < 0 ? 0.3 : 0.7,
    horizonLogReturnP10: signal - 0.002,
    horizonLogReturnP90: signal + 0.002,
    meanCloseLogPath: Array.from({ length: 15 }, (_, index) =>
      signal * (index + 1) / 15),
    medianCloseLogPath: Array.from({ length: 15 }, (_, index) =>
      signal * 0.9 * (index + 1) / 15),
  };
  return {
    episodeId,
    decisionTime,
    forecast: {
      row,
      distribution: { grid, probabilities, mean: signal < 0 ? -0.6 : 0.6 },
      utilityDistribution: { grid, probabilities, mean: signal < 0 ? -0.6 : 0.6 },
    },
    actualLogReturn: 0.0004 + 1.5 * signal,
  };
}

test("ridge return calibration learns a deterministic affine signal", () => {
  const examples = Array.from({ length: 60 }, (_, index) =>
    example(index < 30 ? "a" : "b", index, (index - 29.5) / 20_000));
  const calibration = fitKronosReturnCalibration(examples, 0.01);
  const prediction = predictKronosReturn(calibration, example("x", 100, 0.001).forecast);
  assert.ok(Math.abs(prediction - 0.0019) < 1e-4);
  assert.equal(calibration.trainingExamples, 60);
  assert.throws(
    () => validateKronosReturnCalibration({
      ...calibration,
      coefficients: calibration.coefficients.map((value, index) =>
        index === 0 ? value + 1e-6 : value),
    }),
    /fingerprint/,
  );
});

test("episode cross-validation emits one strictly held-out prediction per row", () => {
  const examples = Array.from({ length: 80 }, (_, index) =>
    example(index < 40 ? "a" : "b", index, (index % 40 - 19.5) / 10_000));
  const [result] = crossValidateKronosReturnCalibrations(examples, [0.1]);
  assert.equal(result!.predictions.size, examples.length);
  assert.ok(result!.oofCorrelation > 0.99);
  assert.ok(result!.oofDirectionAccuracy > 0.8);
});
