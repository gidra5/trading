import assert from "node:assert/strict";
import test from "node:test";
import type {
  SequentialDerivativesKlineRow,
  SequentialDerivativesMetricRow,
  SequentialTradeFlowSecond,
} from "@trading/storage";
import {
  buildForwardFeatureDefinitions,
  FuturesMetricsFeatureEngine,
  FuturesMinuteFeatureEngine,
  TradeFlowFeatureEngine,
} from "./analyze-forward-market-return-information.ts";

function flow(overrides: Partial<SequentialTradeFlowSecond> = {}): SequentialTradeFlowSecond {
  return {
    openTime: 0,
    aggressiveBuyBaseVolume: 1,
    aggressiveSellBaseVolume: 1,
    aggressiveBuyQuoteVolume: 100,
    aggressiveSellQuoteVolume: 100,
    aggressiveBuyAggregateQuantitySquared: 1,
    aggressiveSellAggregateQuantitySquared: 1,
    aggressiveBuyMaxAggregateQuantity: 1,
    aggressiveSellMaxAggregateQuantity: 1,
    aggressiveBuyBaseVolumeTimeMoment: 0.4,
    aggressiveSellBaseVolumeTimeMoment: 0.6,
    aggressiveBuyAggregateTradeCount: 1,
    aggressiveSellAggregateTradeCount: 1,
    aggressiveBuyTradeCount: 1,
    aggressiveSellTradeCount: 1,
    aggressorSideFlipCount: 1,
    firstAggressorSide: 1,
    lastAggressorSide: -1,
    firstTradeOffsetMicros: 100_000,
    lastTradeOffsetMicros: 900_000,
    ...overrides,
  };
}

function future(overrides: Partial<SequentialDerivativesKlineRow> = {}): SequentialDerivativesKlineRow {
  return {
    openTime: 0,
    open: 100,
    high: 102,
    low: 99,
    close: 101,
    baseVolume: 10,
    quoteVolume: 1_000,
    tradeCount: 10,
    takerBuyBaseVolume: 7,
    takerBuyQuoteVolume: 700,
    ...overrides,
  };
}

function metrics(scale = 1): SequentialDerivativesMetricRow {
  return {
    openTime: 0,
    sumOpenInterest: 10 * scale,
    sumOpenInterestValue: 1_000 * scale,
    topTraderAccountLongShortRatio: 1.1 * scale,
    topTraderPositionLongShortRatio: 1.2 * scale,
    globalLongShortRatio: 0.9 * scale,
    takerBuySellVolumeRatio: 1.05 * scale,
  };
}

test("forward feature definitions match engine vector lengths", () => {
  const definitions = buildForwardFeatureDefinitions();
  assert.equal(definitions.length, 87);
  assert.equal(new Set(definitions.map((definition) => definition.id)).size, definitions.length);
  assert.equal(TradeFlowFeatureEngine.definitions().length, new TradeFlowFeatureEngine().values().length);
  assert.equal(FuturesMinuteFeatureEngine.definitions().length, new FuturesMinuteFeatureEngine().values().length);
  assert.equal(FuturesMetricsFeatureEngine.definitions().length, new FuturesMetricsFeatureEngine().values().length);
});

test("spot buyer aggression produces positive directional flow coordinates", () => {
  const engine = new TradeFlowFeatureEngine();
  engine.update(flow({
    aggressiveBuyBaseVolume: 3,
    aggressiveBuyQuoteVolume: 300,
    aggressiveBuyTradeCount: 3,
    aggressiveBuyAggregateTradeCount: 3,
  }));
  assert.equal(engine.valid, true);
  assert.ok(engine.values()[0]! > 0);
  assert.ok(engine.values()[1]! > 0);
  assert.ok(engine.values()[2]! > 0);
});

test("spot flow lag coordinates retain only the requested completed second", () => {
  const engine = new TradeFlowFeatureEngine();
  engine.update(flow({ lastAggressorSide: -1 }));
  engine.update(flow({ lastAggressorSide: 1 }));
  assert.equal(engine.values()[29], -1);
  engine.update(flow({ lastAggressorSide: 1 }));
  assert.equal(engine.values()[32], -1);
});

test("completed futures rows expose basis and taker pressure", () => {
  const engine = new FuturesMinuteFeatureEngine();
  engine.update(future(), 100, 5);
  assert.equal(engine.valid, true);
  assert.ok(engine.values()[0]! > 0);
  assert.ok(engine.values()[9]! > 0);
  assert.ok(engine.values()[10]! > 0);
});

test("futures positioning engine computes finite changes after lagged updates", () => {
  const engine = new FuturesMetricsFeatureEngine();
  engine.update(metrics(1), 100);
  engine.update(metrics(1.01), 101);
  assert.equal(engine.valid, true);
  assert.equal(engine.values().length, 29);
  assert.ok([...engine.values()].every(Number.isFinite));
  assert.ok(engine.values()[0]! > 0);
});
