import assert from "node:assert/strict";
import test from "node:test";
import {
  buildCausalTradeStates,
  buildCausalTradeStatesForOrigins,
  normalizeKalshiMinuteCandles,
  normalizeKalshiTrades,
  statesAvailableAt,
} from "./lib/prediction-market-data.ts";
import { classifyRelevantSeries } from "./fetch-prediction-market-history.ts";

test("trade states are strictly causal and aggregate each preceding interval", () => {
  const trades = normalizeKalshiTrades([
    { trade_id: "a", ticker: "TEST", count_fp: "2", yes_price_dollars: "0.40", created_time: "2026-01-01T00:00:00.200Z", taker_outcome_side: "yes" },
    { trade_id: "b", ticker: "TEST", count_fp: "1", yes_price_dollars: "0.70", created_time: "2026-01-01T00:00:00.800Z", taker_outcome_side: "no" },
    { trade_id: "c", ticker: "TEST", count_fp: "4", yes_price_dollars: "0.90", created_time: "2026-01-01T00:00:01.000Z", taker_outcome_side: "yes" },
  ]);
  const base = Date.parse("2026-01-01T00:00:00Z");
  const rows = buildCausalTradeStates(trades, base + 1_000, base + 2_000, 1_000);
  assert.equal(rows[0]!.lastProbability, 0.7);
  assert.equal(rows[0]!.intervalTradeCount, 2);
  assert.equal(rows[0]!.intervalVwapProbability, 0.5);
  assert.equal(rows[0]!.intervalYesTakerFraction, 2 / 3);
  assert.equal(rows[1]!.lastProbability, 0.9);
  assert.equal(rows[1]!.intervalTradeCount, 1);
});

test("duplicate and invalid trades are removed", () => {
  const rows = normalizeKalshiTrades([
    { trade_id: "same", ticker: "TEST", count_fp: "1", yes_price_dollars: "0.4", created_time: "2026-01-01T00:00:00Z" },
    { trade_id: "same", ticker: "TEST", count_fp: "1", yes_price_dollars: "0.4", created_time: "2026-01-01T00:00:00Z" },
    { trade_id: "bad", ticker: "TEST", count_fp: "1", yes_price_dollars: "1.4", created_time: "2026-01-01T00:00:00Z" },
  ]);
  assert.equal(rows.length, 1);
});

test("sparse trade origins preserve strict causality", () => {
  const trades = normalizeKalshiTrades([
    { trade_id: "a", ticker: "TEST", count_fp: "1", yes_price_dollars: "0.4", created_time: "2026-01-01T00:00:10Z" },
    { trade_id: "b", ticker: "TEST", count_fp: "2", yes_price_dollars: "0.7", created_time: "2026-01-01T00:01:00Z" },
  ]);
  const origins = [Date.parse("2026-01-01T00:01:00Z"), Date.parse("2026-01-01T00:02:00Z")];
  const rows = buildCausalTradeStatesForOrigins(trades, origins, 60_000);
  assert.equal(rows[0]!.lastProbability, 0.4);
  assert.equal(rows[0]!.intervalTradeCount, 1);
  assert.equal(rows[1]!.lastProbability, 0.7);
  assert.equal(rows[1]!.intervalTradeCount, 1);
});

test("minute quote state only becomes usable at the completed candle end", () => {
  const rows = normalizeKalshiMinuteCandles([{
    end_period_ts: 1_700_000_040,
    yes_bid: { close_dollars: "0.44" },
    yes_ask: { close_dollars: "0.48" },
    price: { open_dollars: "0.43", low_dollars: "0.42", high_dollars: "0.49", close_dollars: "0.47", mean_dollars: "0.46", previous_dollars: "0.41" },
    volume_fp: "12.5",
    open_interest_fp: "88",
  }]);
  assert.ok(Math.abs(rows[0]!.quoteMidProbability! - 0.46) < 1e-12);
  assert.ok(Math.abs(rows[0]!.quotedSpread! - 0.04) < 1e-12);
  assert.equal(statesAvailableAt(rows, 1_700_000_039_999).length, 0);
  assert.equal(statesAvailableAt(rows, 1_700_000_040_000).length, 1);
});

test("relevance discovery keeps canonical crypto and systemic event series", () => {
  assert.deepEqual(classifyRelevantSeries({
    ticker: "KXETH15M", title: "ETH 15M price up down", category: "Crypto", frequency: "fifteen_min",
  }), {
    ticker: "KXETH15M", title: "ETH 15M price up down", category: "Crypto",
    frequency: "fifteen_min", scope: "asset", fast: true,
  });
  assert.equal(classifyRelevantSeries({
    ticker: "KXFEDDECISION", title: "Fed decision", category: "Economics", frequency: "custom",
  })?.scope, "global");
  assert.equal(classifyRelevantSeries({
    ticker: "KXEARTHQUAKE", title: "Earthquake in Japan", category: "Climate and Weather", frequency: "custom",
  })?.scope, "global");
  assert.equal(classifyRelevantSeries({
    ticker: "KXNFL", title: "NFL winner", category: "Sports", frequency: "weekly",
  }), null);
  assert.equal(classifyRelevantSeries({
    ticker: "BTC", title: "Bitcoin range", category: "Crypto", frequency: "daily",
  }), null);
});
