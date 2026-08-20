import assert from "node:assert/strict";
import test from "node:test";
import {
  deriveDeribitOptionSummary,
  mergeMempoolMiningSeries,
  normalizeCommunityCryptoDaily,
  normalizeCoinMetricsRows,
  normalizeDvolRows,
  normalizeBinanceFundingRows,
  normalizeMacroCsv,
  normalizeMacroHtmlTable,
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

test("macro CSV rows retain provider metadata and use the declared publication lag", () => {
  const rows = normalizeMacroCsv([
    "observation_date,CPIAUCSL",
    "2026-01-01,325.252",
    "2026-02-01,.",
  ].join("\n"), {
    id: "CPIAUCSL",
    label: "US consumer price index",
    economy: "United States",
    frequency: "monthly",
    availabilityLagDays: 45,
    provider: "FRED",
    sourceUrl: "https://fred.stlouisfed.org/series/CPIAUCSL",
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.value, 325.252);
  assert.equal(rows[0]!.availableAt, Date.parse("2026-02-15T00:00:00.000Z"));
  assert.equal(rows[0]!.frequency, "monthly");
  assert.equal(rows[0]!.provider, "FRED");
});

test("macro CSV normalization supports quoted SDMX fields, filters, and quarterly periods", () => {
  const rows = normalizeMacroCsv([
    'TIME_PERIOD,OBS_VALUE,MEASURE,NOTE',
    '2025-Q4,1.2,CPI,"quoted, note"',
    '2026-Q1,2.3,OTHER,"ignored"',
  ].join("\n"), {
    id: "TEST",
    label: "Test series",
    economy: "Test",
    frequency: "quarterly",
    availabilityLagDays: 90,
    provider: "Official test provider",
    sourceUrl: "https://example.test",
  }, { period: "TIME_PERIOD", value: "OBS_VALUE", filter: { MEASURE: "CPI" } });
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.time, Date.parse("2025-10-01T00:00:00Z"));
  assert.equal(rows[0]!.availableAt, Date.parse("2025-12-30T00:00:00Z"));
});

test("official HTML macro tables support Russian and Bank of England date formats", () => {
  const definition = {
    id: "RATE",
    label: "Policy rate",
    economy: "Test",
    frequency: "event" as const,
    availabilityLagDays: 1,
    provider: "Official provider",
    sourceUrl: "https://example.test",
  };
  const russian = normalizeMacroHtmlTable("<table><tr><th>Date</th><th>Rate</th></tr><tr><td>01.08.2026</td><td>18,00</td></tr></table>", definition, { period: 0, value: 1 });
  const british = normalizeMacroHtmlTable("<table><tr><td>18 Dec 25</td><td>3.75</td></tr></table>", definition, { period: 0, value: 1 });
  assert.equal(russian[0]!.value, 18);
  assert.equal(russian[0]!.time, Date.parse("2026-08-01T00:00:00Z"));
  assert.equal(british[0]!.time, Date.parse("2025-12-18T00:00:00Z"));
});

test("Binance funding rows become available after settlement and tolerate missing mark prices", () => {
  const rows = normalizeBinanceFundingRows([{
    symbol: "BTCUSDT",
    fundingTime: 1_700_000_000_000,
    fundingRate: "0.0001",
    markPrice: "",
  }]);
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.fundingRate, 0.0001);
  assert.equal(rows[0]!.availableAt, 1_700_000_060_000);
  assert.equal(rows[0]!.markPrice, null);
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
