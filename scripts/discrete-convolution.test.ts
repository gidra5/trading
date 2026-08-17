import assert from "node:assert/strict";
import test from "node:test";
import {
  circularConvolutionPower,
  compoundCircularConvolution,
} from "./lib/discrete-convolution.js";

test("raises a discrete distribution by circular convolution", () => {
  const probabilities = new Float64Array(8);
  probabilities[0] = 0.5;
  probabilities[1] = 0.5;
  const convolved = circularConvolutionPower(probabilities, 2);
  assert.ok(Math.abs(convolved[0]! - 0.25) < 1e-12);
  assert.ok(Math.abs(convolved[1]! - 0.5) < 1e-12);
  assert.ok(Math.abs(convolved[2]! - 0.25) < 1e-12);
  assert.ok(Math.abs(convolved.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("preserves negative lattice offsets in circular indexing", () => {
  const probabilities = new Float64Array(8);
  probabilities[7] = 0.5;
  probabilities[1] = 0.5;
  const convolved = circularConvolutionPower(probabilities, 2);
  assert.ok(Math.abs(convolved[6]! - 0.25) < 1e-12);
  assert.ok(Math.abs(convolved[0]! - 0.5) < 1e-12);
  assert.ok(Math.abs(convolved[2]! - 0.25) < 1e-12);
});

test("matches a binomial law for repeated convolution", () => {
  const probabilities = new Float64Array(32);
  probabilities[0] = 0.5;
  probabilities[1] = 0.5;
  const convolved = circularConvolutionPower(probabilities, 5);
  const expected = [1, 5, 10, 10, 5, 1].map((coefficient) => coefficient / 32);
  expected.forEach((probability, index) => {
    assert.ok(Math.abs(convolved[index]! - probability) < 1e-12);
  });
  assert.ok(Math.abs(convolved.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("mixes convolution powers using a count distribution", () => {
  const marks = new Float64Array(8);
  marks[1] = 1;
  const compound = compoundCircularConvolution(marks, [0.25, 0.5, 0.25]);
  assert.ok(Math.abs(compound[0]! - 0.25) < 1e-12);
  assert.ok(Math.abs(compound[1]! - 0.5) < 1e-12);
  assert.ok(Math.abs(compound[2]! - 0.25) < 1e-12);
});
