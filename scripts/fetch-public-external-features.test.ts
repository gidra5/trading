import assert from "node:assert/strict";
import test from "node:test";
import {
  deriveDeribitOptionSummary,
  mergeMempoolMiningSeries,
  normalizeCommunityCryptoDaily,
  normalizeCoinMetricsRows,
  normalizeDvolRows,
  normalizeFredVixCsv,
} from "./lib/external-public-data.ts";

test("DVOL rows receive a causal one-hour availability lag", () => {
  const rows = normalizeDvolRows([[1_000, 50, 51, 49, 50.5]]);
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.availableAt, 3_601_000);
});

test("FRED VIX rows skip missing values and become available at the next UTC day", () => {
  const rows = normalizeFredVixCsv([
    "observation_date,VIXCLS",
    "2026-01-02,17.24",
    "2026-01-05,.",
    "2026-01-06,18.50",
  ].join("\n"));
  assert.equal(rows.length, 2);
  assert.equal(rows[0]!.close, 17.24);
  assert.equal(rows[0]!.availableAt, Date.parse("2026-01-03T00:00:00.000Z"));
});

test("Coin Metrics rows separate assumed next-day availability from latest revision time", () => {
  const rows = normalizeCoinMetricsRows([{
    time: "2026-01-01T00:00:00Z",
    FlowInExUSD: "10",
    "FlowInExUSD-status": "flash",
    "FlowInExUSD-status-time": "2026-01-02T03:00:00Z",
  }], ["FlowInExUSD"]);
  assert.equal(rows[0]!.values.FlowInExUSD, 10);
  assert.equal(rows[0]!.availableAt, Date.parse("2026-01-02T00:00:00Z"));
  assert.equal(rows[0]!.latestRevisionAt, Date.parse("2026-01-02T03:00:00Z"));
});

test("mempool mining series merges fields at a shared timestamp", () => {
  const rows = mergeMempoolMiningSeries({
    fees: [{ timestamp: 100, avgHeight: 5, avgFees: 7 }, { timestamp: 200, avgHeight: 6, avgFees: 8 }],
    feeRates: [{ timestamp: 100, avgHeight: 5, avgFee_50: 2 }, { timestamp: 200, avgHeight: 6, avgFee_50: 3 }],
  });
  assert.equal(rows.length, 2);
  assert.equal(rows[0]!.values["fees.avgFees"], 7);
  assert.equal(rows[0]!.values["feeRates.avgFee_50"], 2);
  assert.equal(rows[0]!.availableAt, 200_000);
});

test("Deribit option surface produces ATM, skew, OI, and strike features", () => {
  const observedAt = Date.UTC(2026, 0, 1, 8);
  const expiry = "08JAN26";
  const books = [
    { instrument_name: `BTC-${expiry}-90000-P`, mark_iv: 60, underlying_price: 100000, open_interest: 3 },
    { instrument_name: `BTC-${expiry}-100000-P`, mark_iv: 50, underlying_price: 100000, open_interest: 4 },
    { instrument_name: `BTC-${expiry}-100000-C`, mark_iv: 49, underlying_price: 100000, open_interest: 5 },
    { instrument_name: `BTC-${expiry}-110000-C`, mark_iv: 55, underlying_price: 100000, open_interest: 2 },
  ];
  const summary = deriveDeribitOptionSummary(books, observedAt);
  assert.equal(summary.instrumentCount, 4);
  assert.equal(summary.expiries.length, 1);
  assert.ok(summary.expiries[0]!.atmIv !== null);
  assert.ok(summary.expiries[0]!.putCall25Skew !== null);
  assert.equal(summary.totalCallOpenInterest, 7);
  assert.equal(summary.totalPutOpenInterest, 7);
  assert.equal(summary.nearestMajorStrike, 100000);
});

test("community daily metrics retain revisions separately from assumed availability", () => {
  const rows = normalizeCommunityCryptoDaily({
    whale: { data: [{ timestamp: 1_000, value: 0.5, last_modified: 9_000 }] },
    miner: { data: [{ timestamp: 1_000, value: -2, last_modified: 10_000 }] },
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.values.whale, 0.5);
  assert.equal(rows[0]!.values.miner, -2);
  assert.equal(rows[0]!.availableAt, 86_401_000);
  assert.equal(rows[0]!.latestRevisionAt, 10_000);
});
