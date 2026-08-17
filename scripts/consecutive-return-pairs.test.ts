import assert from "node:assert/strict";
import test from "node:test";
import {
  fitEllipticalGeneralizedGaussianPower,
  fitRadialCandidates,
  jensenShannonBits,
  PairMoments,
  summarizePairShape,
} from "./lib/consecutive-return-pairs.js";

test("PairMoments measures signed, magnitude, sign, and zero dependence", () => {
  const moments = new PairMoments();
  moments.add(1, 2);
  moments.add(2, 4);
  moments.add(3, 6);
  moments.add(0, 0);
  const summary = moments.snapshot();
  assert.equal(summary.observations, 4);
  assert.equal(summary.zeroZeroFraction, 0.25);
  assert.equal(summary.anyZeroFraction, 0.25);
  assert.equal(summary.nonzeroSameSignFraction, 1);
  assert.ok((summary.correlation ?? 0) > 0.99);
  assert.ok((summary.absoluteCorrelation ?? 0) > 0.99);
  assert.equal(summary.continuous.observations, 3);
  assert.ok((summary.continuous.correlation ?? 0) > 0.99);
});

test("radial-lognormal candidate recovers an elliptical lognormal radius", () => {
  const moments = new PairMoments();
  const sample = { x: [] as number[], y: [] as number[] };
  let state = 0x3456_789a;
  const uniform = () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return (state + 0.5) / 0x1_0000_0000;
  };
  const targetLogStandardDeviation = 0.55;
  for (let index = 0; index < 40_000; index += 1) {
    const normal = Math.sqrt(-2 * Math.log(uniform()))
      * Math.cos(2 * Math.PI * uniform());
    const radius = Math.exp(0.2 + targetLogStandardDeviation * normal);
    const angle = 2 * Math.PI * uniform();
    const x = radius * Math.cos(angle);
    const y = radius * Math.sin(angle);
    moments.add(x, y);
    sample.x.push(x);
    sample.y.push(y);
  }
  const fits = fitRadialCandidates(sample, moments.snapshot(), 40_000);
  const fit = fits.find((item) => item.family === "elliptical-radial-lognormal");
  assert.ok(fit);
  assert.equal(fits[0]!.family, "elliptical-radial-lognormal");
  assert.ok(Math.abs(
    fit.parameters.logRadiusStandardDeviation! - targetLogStandardDeviation
  ) < 0.03);
});

test("shape histograms normalize and Jensen-Shannon divergence is symmetric", () => {
  const moments = new PairMoments();
  const sample = { x: [] as number[], y: [] as number[] };
  for (let index = 0; index < 1_000; index += 1) {
    const x = Math.sin(index * 0.31);
    const y = Math.cos(index * 0.17);
    moments.add(x, y);
    sample.x.push(x);
    sample.y.push(y);
  }
  const shape = summarizePairShape(sample, moments.snapshot());
  const unconditionalTotal = shape.unconditionalHistogram.reduce((sum, value) => sum + value, 0);
  const continuousTotal = shape.continuousHistogram.reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(unconditionalTotal - 1) < 1e-12);
  assert.ok(Math.abs(continuousTotal - 1) < 1e-12);
  assert.equal(jensenShannonBits(shape.continuousHistogram, shape.continuousHistogram), 0);
  const reversed = shape.continuousHistogram.slice().reverse();
  assert.ok(Math.abs(
    jensenShannonBits(shape.continuousHistogram, reversed)
      - jensenShannonBits(reversed, shape.continuousHistogram),
  ) < 1e-12);
});

test("elliptical generalized-Gaussian fit recovers a bivariate Laplace power", () => {
  const moments = new PairMoments();
  const sample = { x: [] as number[], y: [] as number[] };
  let state = 0x1234_5678;
  const uniform = () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return (state + 0.5) / 0x1_0000_0000;
  };
  for (let index = 0; index < 30_000; index += 1) {
    // For d=2 and p=1, R has a Gamma(shape=2, scale=1) law.
    const radius = -Math.log(uniform() * uniform());
    const angle = 2 * Math.PI * uniform();
    const x = radius * Math.cos(angle);
    const y = radius * Math.sin(angle);
    moments.add(x, y);
    sample.x.push(x);
    sample.y.push(y);
  }
  const fit = fitEllipticalGeneralizedGaussianPower(sample, moments.snapshot(), 30_000);
  assert.ok(Math.abs(fit.power - 1) < 0.08, `estimated power was ${fit.power}`);
});
