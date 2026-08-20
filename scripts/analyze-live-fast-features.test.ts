import assert from "node:assert/strict";
import test from "node:test";
import { bookFeatures, componentSpecs } from "./analyze-live-fast-features.ts";

test("book feature extraction excludes stale venues and normalizes depth pressure", () => {
  const values = bookFeatures({
    venues: {
      binanceSpot: { valid: true, ageMs: 10, spreadBps: 1, l1Imbalance: 0.2, top5Imbalance: 0.1 },
      krakenSpot: { valid: true, ageMs: 6_000, spreadBps: 2, l1Imbalance: -0.2, top5Imbalance: -0.1 },
    },
    binanceSpotFlow: { bidAddedQuote: 10, bidRemovedQuote: 2, askAddedQuote: 3, askRemovedQuote: 5 },
    crossVenue: { spotMidDispersionBps: 2.5, binancePerpetualBasisBps: -1 },
  });
  assert.equal(values.binance_spot_l1_imbalance, 0.2);
  assert.equal(values.kraken_spot_l1_imbalance, undefined);
  assert.equal(values.binance_depth_pressure, 10 / 20);
  assert.equal(values.spot_mid_dispersion_bps, 2.5);
});

test("component targets separate inactivity, active sign, magnitude, and conditional combinations", () => {
  const specs = componentSpecs();
  assert.equal(specs.length, 19);
  const sample = { second: 1, feature: 0, target: -0.0002, previous: 0, volatility: 1 };
  const thresholds = [0.0001, 0.0002, 0.0003, 0.0004];
  assert.equal(specs.find((row) => row.id === "inactive")!.target(sample, thresholds), 0);
  assert.equal(specs.find((row) => row.id === "sign_given_active")!.condition(sample, thresholds), true);
  assert.equal(specs.find((row) => row.id === "sign_given_active")!.target(sample, thresholds), 0);
  assert.equal(specs.find((row) => row.id === "large_q50_given_active")!.target(sample, thresholds), 1);
  assert.equal(specs.find((row) => row.id === "large_q90_given_positive")!.condition(sample, thresholds), false);
});
