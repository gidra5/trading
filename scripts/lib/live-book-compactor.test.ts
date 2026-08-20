import assert from "node:assert/strict";
import test from "node:test";
import { LiveBookCompactor } from "./live-book-compactor.ts";

test("compacts synchronized Binance depth and resets flow counters", () => {
  const compactor = new LiveBookCompactor();
  compactor.consume("binance-spot-depth-diff", { U: 11, u: 11, b: [["99", "3"]], a: [] }, 1_100);
  compactor.consume("binance-spot-depth-snapshot", {
    lastUpdateId: 10,
    bids: [["99", "2"]],
    asks: [["101", "4"]],
  }, 1_000);
  const first = compactor.summarize(2_000);
  assert.equal(first.venues.binanceSpot.valid, true);
  assert.equal(first.venues.binanceSpot.bidQty, 3);
  assert.equal(first.binanceSpotFlow.bidAddedQuote, 99);
  assert.equal(compactor.summarize(3_000).binanceSpotFlow.bidAddedQuote, 0);
});

test("summarizes cross-venue dispersion and futures basis", () => {
  const compactor = new LiveBookCompactor();
  compactor.consume("coinbase-btcusd-level2", { type: "snapshot", bids: [["99", "1"]], asks: [["101", "1"]] }, 1_000);
  compactor.consume("kraken-btcusd-book", {
    channel: "book",
    type: "snapshot",
    data: [{ bids: [{ price: 100, qty: 2 }], asks: [{ price: 102, qty: 2 }] }],
  }, 1_000);
  compactor.consume("binance-usdm-book-ticker", { b: "102", B: "3", a: "104", A: "3" }, 1_000);
  const row = compactor.summarize(1_500);
  assert.equal(row.crossVenue.freshSpotVenueCount, 2);
  assert.ok(row.crossVenue.spotMidDispersionBps! > 0);
  assert.ok(row.crossVenue.binancePerpetualBasisBps! > 0);
});

test("invalidates Binance depth on a sequence gap", () => {
  const compactor = new LiveBookCompactor();
  compactor.consume("binance-spot-depth-snapshot", { lastUpdateId: 10, bids: [[99, 1]], asks: [[101, 1]] }, 1_000);
  compactor.consume("binance-spot-depth-diff", { U: 12, u: 12, b: [], a: [] }, 1_100);
  assert.equal(compactor.summarize(1_200).venues.binanceSpot.valid, false);
});

test("flags a crossed Kraken book for resubscription", () => {
  const compactor = new LiveBookCompactor();
  compactor.consume("kraken-btcusd-book", {
    channel: "book",
    type: "snapshot",
    data: [{ bids: [{ price: 100, qty: 1 }], asks: [{ price: 101, qty: 1 }] }],
  }, 1_000);
  assert.equal(compactor.needsKrakenSnapshot(), false);
  compactor.consume("kraken-btcusd-book", {
    channel: "book",
    type: "update",
    data: [{ bids: [{ price: 102, qty: 1 }], asks: [] }],
  }, 1_100);
  assert.equal(compactor.needsKrakenSnapshot(), true);
});
