import assert from "node:assert/strict";
import test from "node:test";
import { stationaryRenewalCountDistribution } from "./lib/renewal-process.js";

test("counts an event every second for unit interarrivals", () => {
  const counts = stationaryRenewalCountDistribution([0, 1], 5);
  assert.ok(Math.abs(counts[5]! - 1) < 1e-12);
});

test("uses the stationary phase for deterministic two-second interarrivals", () => {
  const oneSecond = stationaryRenewalCountDistribution([0, 0, 1], 1);
  assert.ok(Math.abs(oneSecond[0]! - 0.5) < 1e-12);
  assert.ok(Math.abs(oneSecond[1]! - 0.5) < 1e-12);
  const twoSeconds = stationaryRenewalCountDistribution([0, 0, 1], 2);
  assert.ok(Math.abs(twoSeconds[1]! - 1) < 1e-12);
});
