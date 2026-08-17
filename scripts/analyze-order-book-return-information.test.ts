import assert from "node:assert/strict";
import test from "node:test";
import type { SequentialDerivativesBookDepthSnapshot } from "@trading/storage";
import {
  buildBookFeatureDefinitions,
  snapshotValues,
} from "./analyze-order-book-return-information.ts";

function snapshot(bidMultiplier: number, askMultiplier: number) {
  const bid = [null, 10, 20, 30, 40, 50].map((value) =>
    value === null ? null : value * bidMultiplier) as any;
  const ask = [null, 10, 20, 30, 40, 50].map((value) =>
    value === null ? null : value * askMultiplier) as any;
  return {
    timestampOffsetSeconds: 30,
    schemaBandCount: 10,
    bidDepth: bid,
    askDepth: ask,
    bidNotional: bid.map((value: number | null) => value === null ? null : value * 100),
    askNotional: ask.map((value: number | null) => value === null ? null : value * 100),
    bandAvailable: [false, true, true, true, true, true],
  } as SequentialDerivativesBookDepthSnapshot;
}

test("book feature definitions align with the computed vector", () => {
  const definitions = buildBookFeatureDefinitions();
  const values = snapshotValues(snapshot(1, 1), undefined);
  assert.equal(definitions.length, 25);
  assert.equal(values.length, definitions.length);
  assert.equal(new Set(definitions.map((definition) => definition.id)).size, definitions.length);
});

test("balanced books have zero imbalance and first-snapshot deltas", () => {
  const values = snapshotValues(snapshot(1, 1), undefined);
  for (const index of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17]) {
    assert.equal(values[index], 0);
  }
  for (let index = 20; index < 25; index += 1) assert.equal(values[index], 0);
});

test("bid-heavy books produce positive imbalance and causal changes", () => {
  const balanced = snapshotValues(snapshot(1, 1), undefined);
  const bidHeavy = snapshotValues(snapshot(2, 1), balanced);
  assert.ok(bidHeavy[0]! > 0);
  assert.ok(bidHeavy[8]! > 0);
  assert.ok(bidHeavy[14]! > 0);
  assert.ok(bidHeavy[20]! > 0);
  assert.ok(bidHeavy[22]! > 0);
});
