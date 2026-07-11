import assert from "node:assert/strict";
import test from "node:test";
import { PaperTradingApi } from "../src/trading-api/paper-api.js";

const rules = {
  price: { min: null, max: null, step: null },
  limitQuantity: { min: null, max: null, step: null },
  marketQuantity: { min: null, max: null, step: null },
  minNotional: null,
  maxNotional: null,
  maxLeverage: 5,
};

test("paper adapter reserves, triggers, fills, and accounts through the trading API", async () => {
  const api = new PaperTradingApi({
    startingQuote: 1_000,
    friction: 0.01,
    rules,
    getHistory: async () => [],
  });
  await api.onTick(tick(100));
  const buy = await api.createLimitOrder({ side: "buy", size: 1, price: 90 });
  assert.equal(buy.accepted, true);
  assert.equal((await api.getEquity()).quoteReserved, 90.9);

  await api.onTick(tick(89));
  const buyEvents = api.drainEvents();
  assert.equal(buyEvents.length, 1);
  assert.equal(buyEvents[0].type, "fill");
  assert.deepEqual(await api.getEquity(), {
    quoteAvailable: 909.1,
    quoteReserved: 0,
    quoteUnleveraged: 909.1,
    assetAvailable: 1,
    assetReserved: 0,
    assetUnleveraged: 1,
  });

  const stop = await api.createStopMarketOrder({ side: "sell", size: 1, price: 80 });
  assert.equal(stop.order.status, "pending");
  await api.onTick(tick(81));
  assert.equal(api.drainEvents().length, 0);
  await api.onTick(tick(80));
  assert.deepEqual(api.drainEvents().map((event) => event.type), ["open", "fill"]);
  assert.equal((await api.getEquity()).assetUnleveraged, 0);
});

test("paper adapter emits incremental partial fills from tick liquidity", async () => {
  const api = new PaperTradingApi({
    startingQuote: 1_000,
    friction: 0,
    rules,
    getHistory: async () => [],
  });
  await api.onTick(tick(100));
  const order = await api.createLimitOrder({ side: "buy", size: 2, price: 100 });
  await api.onTick(tick(99));
  assert.deepEqual(api.drainEvents(), [{
    type: "partial-fill",
    orderId: order.order.id,
    fill: { filledAsset: 1, filledQuote: 100, remaining: 1 },
  }]);
  await api.onTick(tick(98));
  assert.deepEqual(api.drainEvents(), [{
    type: "fill",
    orderId: order.order.id,
    fill: { filledAsset: 1, filledQuote: 100, remaining: 0 },
  }]);
});

test("paper adapter does not expose short-sale proceeds as free collateral", async () => {
  const api = new PaperTradingApi({
    startingQuote: 1_000,
    friction: 0,
    rules: { ...rules, maxLeverage: 1 },
    getHistory: async () => [],
  });
  await api.onTick(tick(100));
  await api.createMarketOrder({ side: "sell", size: 10 });

  assert.deepEqual(await api.getEquity(), {
    quoteAvailable: 0,
    quoteReserved: 0,
    quoteUnleveraged: 2_000,
    assetAvailable: 0,
    assetReserved: 0,
    assetUnleveraged: -10,
  });

  await api.onTick(tick(80));
  assert.equal((await api.getEquity()).quoteAvailable, 400);
});

test("paper capacity reserves friction on the full order notional", async () => {
  const api = new PaperTradingApi({
    startingQuote: 1_000,
    friction: 0.001,
    rules: { ...rules, maxLeverage: 20 },
    getHistory: async () => [],
  });

  assert.deepEqual(await api.getOrderCapacity({ side: "sell", price: 100, leverage: 20 }), {
    quote: 1_000 / (1 / 20 + 0.001),
    leverage: 20,
  });
});

function tick(price: number) {
  return { timestamp: price * 1_000, price, quantity: 1, candle: null };
}
