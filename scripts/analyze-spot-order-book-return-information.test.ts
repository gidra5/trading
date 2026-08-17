import assert from "node:assert/strict";
import test from "node:test";
import {
  buildSpotBookFeatureDefinitions,
  spotSnapshotValues,
  type SpotBookSnapshot,
  validSnapshot,
} from "./analyze-spot-order-book-return-information.ts";

function snapshot(
  eventTime: number,
  bidQuantity = 1,
  askQuantity = 1,
): SpotBookSnapshot {
  return {
    symbol: "BTCUSDT",
    eventTime,
    bids: Array.from({ length: 10 }, (_, index) => ({
      price: 99.99 - index * 0.01,
      quantity: bidQuantity,
    })),
    asks: Array.from({ length: 10 }, (_, index) => ({
      price: 100.01 + index * 0.01,
      quantity: askQuantity,
    })),
  };
}

test("spot feature definitions align with the computed top-10 vector", () => {
  const book = snapshot(1_000);
  const definitions = buildSpotBookFeatureDefinitions();
  const values = spotSnapshotValues(book, undefined);
  assert.equal(definitions.length, 27);
  assert.equal(values.length, definitions.length);
  assert.equal(new Set(definitions.map((definition) => definition.id)).size, definitions.length);
  assert.equal(validSnapshot(book), true);
});

test("a symmetric book has zero queue pressure and first-snapshot changes", () => {
  const values = spotSnapshotValues(snapshot(1_000), undefined);
  assert.ok(Math.abs(values[1]!) < 1e-10);
  for (const index of [2, 3, 4, 5, 16, 19, 20, 21, 22, 23, 24, 25, 26]) {
    assert.ok(Math.abs(values[index]!) < 1e-10, `feature ${index} was ${values[index]}`);
  }
});

test("bid-heavy updates produce positive microprice, imbalance, and OFI", () => {
  const firstSnapshot = snapshot(1_000);
  const firstValues = spotSnapshotValues(firstSnapshot, undefined);
  const nextSnapshot = snapshot(2_000, 2, 1);
  const values = spotSnapshotValues(nextSnapshot, { snapshot: firstSnapshot, values: firstValues });
  assert.ok(values[1]! > 0);
  assert.ok(values[2]! > 0);
  assert.ok(values[19]! > 0);
  assert.ok(values[26]! > 0);
});

test("change features reset across stale gaps", () => {
  const firstSnapshot = snapshot(1_000);
  const firstValues = spotSnapshotValues(firstSnapshot, undefined);
  const laterSnapshot = snapshot(7_000, 2, 1);
  const values = spotSnapshotValues(laterSnapshot, { snapshot: firstSnapshot, values: firstValues });
  for (let index = 19; index < 27; index += 1) assert.equal(values[index], 0);
});

test("crossed or incorrectly sorted books are rejected", () => {
  const crossed = snapshot(1_000);
  crossed.bids[0]!.price = crossed.asks[0]!.price;
  assert.equal(validSnapshot(crossed), false);
  const unsorted = snapshot(1_000);
  unsorted.bids[1]!.price = unsorted.bids[0]!.price + 0.01;
  assert.equal(validSnapshot(unsorted), false);
});
