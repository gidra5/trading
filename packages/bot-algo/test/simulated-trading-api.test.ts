import assert from "node:assert/strict";
import test from "node:test";
import { SimulatedTradingApi } from "../src/simulated-trading-api.js";

const rules = {
  price: { min: null, max: null, step: null },
  limitQuantity: { min: null, max: null, step: null },
  marketQuantity: { min: null, max: null, step: null },
  minNotional: null,
  maxNotional: null,
  maxLeverage: 100,
};

test("simulated trading reserves, fills, and accounts for execution friction", async () => {
  const api = simulated({ friction: 0.01 });
  await api.onTick(tick(0, 100));
  const buy = await api.createLimitOrder({ side: "buy", size: 1, price: 90, leverage: 1 });
  assert.equal(buy.accepted, true);
  assert.equal((await api.getEquity()).quoteReserved, 90.9);

  await api.onTick(tick(1_000, 89));
  assert.deepEqual(api.drainEvents(), [{
    type: "fill",
    orderId: buy.order.id,
    fill: { filledAsset: 1, filledQuote: 90.9, price: 90, feeQuote: 0.9, remaining: 0 },
  }]);
  assert.equal(api.status().feesPaid, 0.9);
  assert.deepEqual(await api.getEquity(), {
    quoteAvailable: 909.1,
    quoteReserved: 0,
    quoteUnleveraged: 909.1,
    assetAvailable: 1,
    assetReserved: 0,
    assetUnleveraged: 1,
  });
});

test("simulated stop orders trigger before filling and liquidity produces partial fills", async () => {
  const api = simulated();
  await api.onTick(tick(0, 100));
  const stop = await api.createStopMarketOrder({ side: "sell", size: 1, price: 80 });
  assert.equal(stop.order.status, "pending");
  await api.onTick(tick(1_000, 81, 1));
  assert.deepEqual(api.drainEvents(), []);
  await api.onTick(tick(2_000, 80, 1));
  assert.deepEqual(api.drainEvents().map((event) => event.type), ["open", "fill"]);

  const partialApi = simulated();
  await partialApi.onTick(tick(0, 100));
  const order = await partialApi.createLimitOrder({ side: "buy", size: 2, price: 100 });
  await partialApi.onTick(tick(1_000, 99, 1));
  const partial = partialApi.drainEvents()[0];
  assert.equal(partial?.type, "partial-fill");
  if (partial?.type !== "partial-fill") throw new Error("Expected a partial fill.");
  assert.deepEqual(partial.fill, {
    filledAsset: 1,
    filledQuote: 100,
    price: 100,
    feeQuote: 0,
    remaining: 1,
  });
  await partialApi.onTick(tick(2_000, 98, 1));
  assert.equal(partialApi.drainEvents()[0]?.type, "fill");
});

test("short-sale proceeds are not exposed as free collateral", async () => {
  const api = simulated({ rules: { ...rules, maxLeverage: 1 } });
  await api.onTick(tick(0, 100));
  await api.createMarketOrder({ side: "sell", size: 10, leverage: 1 });

  assert.deepEqual(await api.getEquity(), {
    quoteAvailable: 0,
    quoteReserved: 0,
    quoteUnleveraged: 2_000,
    assetAvailable: 0,
    assetReserved: 0,
    assetUnleveraged: -10,
  });

  await api.onTick(tick(1_000, 80));
  assert.equal((await api.getEquity()).quoteAvailable, 400);
});

test("simulated capacity is the exact fee-aware leverage capacity", async () => {
  const api = simulated({ friction: 0.001 });
  await api.onTick(tick(0, 100));

  const capacity = await api.getOrderCapacity({ side: "buy", price: 100, leverage: 20 });

  assert.ok(Math.abs(capacity.quote - 1_000 / (1 / 20 + 0.001)) < 1e-6);
  assert.equal(capacity.leverage, 20);
});

test("short closeout uses the same buy-side friction as an actual cover fill", async () => {
  const friction = 0.01;
  const api = simulated({ friction });
  await api.onTick(tick(0, 100));
  const capacity = await api.getOrderCapacity({ side: "sell", price: 100, leverage: 2 });
  await api.createMarketOrder({
    side: "sell",
    size: capacity.quote / 100,
    leverage: 2,
  });
  const entry = api.drainEvents()[0];
  assert.equal(entry?.type, "fill");
  if (entry?.type !== "fill") throw new Error("Expected short entry fill.");
  assert.equal(entry.fill.price, 100);
  assert.ok(Math.abs((entry.fill.feeQuote ?? 0) - capacity.quote * friction) < 1e-8);

  const beforeCover = api.status().closeoutEquity;
  await api.createMarketOrder({
    side: "buy",
    size: entry.fill.filledAsset,
    leverage: 1,
    reduceOnly: true,
  });

  assert.ok(Math.abs(api.status().equity - beforeCover) < 1e-8);
});

test("simulated trading liquidates at the configured effective-leverage boundary", async () => {
  const api = simulated({ maxEffectiveLeverage: 250 });
  await api.onTick(tick(0, 100));
  const capacity = await api.getOrderCapacity({ side: "buy", price: 100, leverage: 100 });
  await api.createMarketOrder({
    side: "buy",
    size: capacity.quote / 100,
    leverage: 100,
  });
  api.drainEvents();

  await api.onTick(tick(1_000, 99));

  const status = api.status();
  assert.equal(status.liquidated, true);
  assert.equal(status.liquidationCount, 1);
  assert.equal(status.liquidationReason, "effective-leverage");
  assert.ok(Math.abs((status.liquidationPrice ?? 0) - 99.39759036144578) < 1e-9);
  assert.ok(Math.abs(status.equity - 397.59036144577817) < 1e-6);
  assert.deepEqual(api.drainEvents(), [{
    type: "liquidation",
    at: 1_000,
    price: status.liquidationPrice,
    equity: status.equity,
    reason: "effective-leverage",
  }]);
});

test("simulated trading stops an insolvent account instead of allowing recovery", async () => {
  const api = simulated();
  await api.onTick(tick(0, 100));
  const capacity = await api.getOrderCapacity({ side: "buy", price: 100, leverage: 100 });
  await api.createMarketOrder({ side: "buy", size: capacity.quote / 100, leverage: 100 });
  api.drainEvents();

  await api.onTick(tick(1_000, 98));
  await api.onTick(tick(2_000, 110));

  assert.equal(api.status().liquidationReason, "insolvent");
  assert.equal(api.status().equity, 0);
  assert.equal((await api.getEquity()).assetUnleveraged, 0);
});

test("simulated trading liquidates negative cash even after exposure is fully closed", async () => {
  const api = simulated({
    snapshot: {
      version: 1,
      quote: -1,
      asset: 0,
      price: 100,
      updatedAt: 0,
      orders: [],
      feesPaid: 0,
      maintenancePaid: 0,
      liquidationCount: 0,
      liquidatedAt: null,
      liquidationPrice: null,
      liquidationReason: null,
      maxEffectiveLeverage: 0,
    },
  });

  await api.onTick(tick(1_000, 100));

  assert.equal(api.status().liquidationReason, "insolvent");
  assert.equal(api.status().liquidationCount, 1);
  assert.equal(api.status().equity, 0);
});

test("simulated trading accrues quote and asset borrow maintenance", async () => {
  const long = simulated({ quoteBorrowBpsHour: 10 });
  await long.onTick(tick(1, 100));
  await long.createMarketOrder({ side: "buy", size: 20, leverage: 2 });
  long.drainEvents();
  await long.onTick(tick(3_600_001, 100));
  assert.ok(Math.abs(long.status().equity - 999) < 1e-9);
  assert.deepEqual(long.drainEvents(), [{
    type: "maintenance",
    elapsedMs: 3_600_000,
    quoteCharge: 1,
    assetCharge: 0,
  }]);

  const short = simulated({ assetBorrowBpsHour: 10 });
  await short.onTick(tick(1, 100));
  await short.createMarketOrder({ side: "sell", size: 20, leverage: 2 });
  short.drainEvents();
  await short.onTick(tick(3_600_001, 100));
  assert.ok(Math.abs(short.status().equity - 998) < 1e-9);
  assert.deepEqual(short.drainEvents(), [{
    type: "maintenance",
    elapsedMs: 3_600_000,
    quoteCharge: 0,
    assetCharge: 0.02,
  }]);
});

function simulated(overrides: Partial<ConstructorParameters<typeof SimulatedTradingApi>[0]> = {}) {
  return new SimulatedTradingApi({
    startingQuote: 1_000,
    friction: 0,
    rules,
    getHistory: async () => [],
    ...overrides,
  });
}

function tick(timestamp: number, price: number, quantity = Number.POSITIVE_INFINITY) {
  return { timestamp, price, quantity, candle: null };
}
