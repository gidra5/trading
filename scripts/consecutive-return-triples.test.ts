import assert from "node:assert/strict";
import test from "node:test";
import {
  fitTripleCandidates,
  fitTripleGeneralizedGaussianPower,
  projectionJensenShannonBits,
  summarizeTripleShape,
  TripleMoments,
} from "./lib/consecutive-return-triples.js";

test("TripleMoments measures lag-one, lag-two, sign, and zero structure", () => {
  const moments = new TripleMoments();
  moments.add(1, 2, 3);
  moments.add(2, 4, 6);
  moments.add(3, 6, 9);
  moments.add(0, 0, 0);
  const summary = moments.snapshot();
  assert.equal(summary.observations, 4);
  assert.equal(summary.allZeroFraction, 0.25);
  assert.equal(summary.anyZeroFraction, 0.25);
  assert.equal(summary.nonzeroAllSameSignFraction, 1);
  assert.ok(summary.correlations.every((value) => value > 0.99));
  assert.ok(summary.absoluteCorrelations.every((value) => value > 0.99));
  assert.equal(summary.continuous.observations, 3);
});

test("trivariate radial-lognormal candidate recovers log-radius spread", () => {
  const moments = new TripleMoments();
  const sample = { x: [] as number[], y: [] as number[], z: [] as number[] };
  let state = 0x4567_89ab;
  const uniform = () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return (state + 0.5) / 0x1_0000_0000;
  };
  const targetLogStandardDeviation = 0.5;
  for (let index = 0; index < 50_000; index += 1) {
    const normal = Math.sqrt(-2 * Math.log(uniform()))
      * Math.cos(2 * Math.PI * uniform());
    const radius = Math.exp(0.1 + targetLogStandardDeviation * normal);
    const cosine = 2 * uniform() - 1;
    const sine = Math.sqrt(1 - cosine * cosine);
    const azimuth = 2 * Math.PI * uniform();
    const x = radius * sine * Math.cos(azimuth);
    const y = radius * sine * Math.sin(azimuth);
    const z = radius * cosine;
    moments.add(x, y, z);
    sample.x.push(x);
    sample.y.push(y);
    sample.z.push(z);
  }
  const fits = fitTripleCandidates(sample, moments.snapshot(), 50_000);
  const fit = fits.find((item) => item.family === "trivariate-radial-lognormal");
  assert.ok(fit);
  assert.equal(fits[0]!.family, "trivariate-radial-lognormal");
  assert.ok(Math.abs(
    fit.parameters.logRadiusStandardDeviation! - targetLogStandardDeviation
  ) < 0.03);
});

test("triple projection histograms normalize and self-divergence is zero", () => {
  const moments = new TripleMoments();
  const sample = { x: [] as number[], y: [] as number[], z: [] as number[] };
  for (let index = 0; index < 2_000; index += 1) {
    const x = Math.sin(index * 0.31);
    const y = Math.cos(index * 0.17);
    const z = Math.sin(index * 0.11 + 0.4);
    moments.add(x, y, z);
    sample.x.push(x);
    sample.y.push(y);
    sample.z.push(z);
  }
  const shape = summarizeTripleShape(sample, moments.snapshot(), 100);
  for (const histogram of Object.values(shape.projectionHistograms)) {
    assert.ok(Math.abs(histogram.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  }
  assert.equal(
    projectionJensenShannonBits(shape.projectionHistograms, shape.projectionHistograms),
    0,
  );
  assert.equal(shape.visualizationSample.length, 100);
});

test("trivariate generalized-Gaussian fit recovers Laplace radial power", () => {
  const moments = new TripleMoments();
  const sample = { x: [] as number[], y: [] as number[], z: [] as number[] };
  let state = 0x2345_6789;
  const uniform = () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return (state + 0.5) / 0x1_0000_0000;
  };
  for (let index = 0; index < 40_000; index += 1) {
    // In d=3 with p=1 and s=1, R is Gamma(shape=3, scale=1).
    const radius = -Math.log(uniform() * uniform() * uniform());
    const cosine = 2 * uniform() - 1;
    const sine = Math.sqrt(1 - cosine * cosine);
    const azimuth = 2 * Math.PI * uniform();
    const x = radius * sine * Math.cos(azimuth);
    const y = radius * sine * Math.sin(azimuth);
    const z = radius * cosine;
    moments.add(x, y, z);
    sample.x.push(x);
    sample.y.push(y);
    sample.z.push(z);
  }
  const fit = fitTripleGeneralizedGaussianPower(sample, moments.snapshot(), 40_000);
  assert.ok(Math.abs(fit.power - 1) < 0.08, `estimated power was ${fit.power}`);
});
