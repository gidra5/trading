import assert from "node:assert/strict";
import test from "node:test";
import {
  GridTradingBot,
  type StrategyDiagnostics,
  type StrategySnapshot,
  type TradingApi,
  type TradingBotConfig,
  type TradingOrderResult,
  type TradingStrategy,
  type TradingStrategyEntrySignal,
  type TradingStrategyExitSignal,
  type TradingTick,
} from "../src/index.js";

const tick: TradingTick = {
  timestamp: 1_000,
  price: 100,
  quantity: 1,
  candle: null,
};

test("positions own their entry and exit orders from creation through fills", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  const bot = new GridTradingBot({ api, strategy, config: config() });
  strategy.entry = { side: "long", size: 0.5, leverage: 1, price: null, confidence: null };

  await bot.onTick(tick);
  let snapshot = await bot.snapshot();
  assert.equal(snapshot.positions.length, 1);
  assert.equal(snapshot.positions[0].asset, 0);
  assert.equal(snapshot.positions[0].entryGrid?.orders.length, 1);
  assert.equal("orders" in snapshot, false);

  const entry = api.orders[0];
  await bot.onOrder({
    type: "fill",
    orderId: entry.order.id,
    fill: { filledAsset: 5, filledQuote: 500, remaining: 0 },
  });
  snapshot = await bot.snapshot();
  assert.equal(snapshot.positions[0].asset, 5);
  assert.equal(snapshot.positions[0].quote, 500);
  assert.equal(snapshot.positions[0].entryGrid?.orders[0].filled, 500);

  strategy.exit = { side: "long", size: 0.5, price: null, confidence: null };
  await bot.onTick({ ...tick, timestamp: 2_000 });
  const exit = api.orders[1];
  assert.equal(exit.order.type, "market");
  assert.equal(exit.order.side, "sell");
  assert.equal(exit.order.size, 2.5);
  await bot.onOrder({
    type: "fill",
    orderId: exit.order.id,
    fill: { filledAsset: 2.5, filledQuote: 260, remaining: 0 },
  });
  snapshot = await bot.snapshot();
  assert.equal(snapshot.positions[0].asset, 2.5);
  assert.equal(snapshot.positions[0].quote, 240);
  assert.equal(snapshot.positions[0].exitGrid?.orders[0].filled, 2.5);
});

test("exits repay external then internal debt and preserve signed profit", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  const nextConfig = config();
  nextConfig.internalBorrow.enabled = true;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.entry = { side: "short", size: 1, leverage: 2, price: null, confidence: null };
  await bot.onTick(tick);
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 2, filledQuote: 200, remaining: 0 },
  });
  strategy.entry = { side: "long", size: 1, leverage: 2, price: null, confidence: null };
  await bot.onTick({ ...tick, timestamp: 2_000 });
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[1].order.id,
    fill: { filledAsset: 10, filledQuote: 1_000, remaining: 0 },
  });

  strategy.exit = { side: "long", size: 1, price: null, confidence: null };
  await bot.onTick({ ...tick, timestamp: 3_000 });
  const exit = api.orders[2].order;
  await bot.onOrder({
    type: "partial-fill",
    orderId: exit.id,
    fill: { filledAsset: 4, filledQuote: 350, remaining: 6 },
  });
  let long = (await bot.snapshot()).positions.find((position) => position.side === "long")!;
  assert.equal(long.quote, 650);
  assert.equal(long.externalBorrow.quote, 0);
  assert.equal(long.internalBorrow[0].quote, 150);

  await bot.onOrder({
    type: "partial-fill",
    orderId: exit.id,
    fill: { filledAsset: 1, filledQuote: 700, remaining: 5 },
  });
  long = (await bot.snapshot()).positions.find((position) => position.side === "long")!;
  assert.equal(long.asset, 5);
  assert.equal(long.quote, -50);
  assert.equal(long.externalBorrow.quote, 0);
  assert.deepEqual(long.internalBorrow, []);
});

test("a rejected unfilled entry removes its position", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  const bot = new GridTradingBot({ api, strategy, config: config() });
  strategy.entry = { side: "short", size: 0.25, leverage: 1, price: null, confidence: null };
  await bot.onTick(tick);
  await bot.onOrder({ type: "rejected", orderId: api.orders[0].order.id });
  assert.equal((await bot.snapshot()).positions.length, 0);
});

test("bot restore can preserve positions while rewarming incompatible strategy state", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  const bot = new GridTradingBot({ api, strategy, config: config() });
  const snapshot = await bot.snapshot();
  await bot.restore(snapshot, { restoreStrategy: false });
  assert.equal(strategy.restores, 0);
  assert.equal(strategy.updates, 1);
});

test("entry sizing uses provider capacity and effective leverage", async () => {
  const api = new FakeApi();
  api.capacity = { quote: 200, leverage: 2 };
  const strategy = new FakeStrategy();
  const reports: Array<{ side: string; size: number; leverage: number; blocker: string | null }> = [];
  const bot = new GridTradingBot({
    api,
    strategy,
    config: config(),
    onEntryRisk: (report) => reports.push(report),
  });
  strategy.entry = { side: "short", size: 0.5, leverage: 5, price: null, confidence: null };

  await bot.onTick(tick);

  assert.equal(api.orders[0].order.size, 1);
  assert.deepEqual(reports, [{
    side: "short",
    size: 100,
    leverage: 2,
    blocker: null,
  }]);
  assert.equal((await bot.snapshot()).positions[0].leverage, 2);
});

test("manual open and close use market orders without a tick", async () => {
  const api = new FakeApi();
  const bot = new GridTradingBot({ api, strategy: new FakeStrategy(), config: config() });

  await bot.openPosition("long", 2);
  assert.equal(api.orders[0].order.type, "market");
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 2, filledQuote: 220, remaining: 0 },
  });
  assert.equal((await bot.snapshot()).positions[0].entryGrid?.creationPrice, 110);

  await bot.closePositions();
  assert.equal(api.orders[1].order.type, "market");
});

test("entry grids stay within provider quote capacity", async () => {
  const api = new FakeApi();
  api.capacity = { quote: 200, leverage: 1 };
  const strategy = new FakeStrategy();
  const nextConfig = config();
  nextConfig.entryGrid = {
    orderCount: 2,
    maxPriceStep: 0.1,
    sizeDistribution: "linear",
    sizeFraction: 1,
  };
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });
  strategy.entry = { side: "short", size: 1, leverage: 1, price: 100, confidence: null };

  await bot.onTick(tick);

  const notional = api.orders.reduce(
    (sum, result) => sum + result.order.size * (result.order.price ?? tick.price),
    0,
  );
  assert.ok(notional <= 200);
  assert.ok(notional > 199.99);
});

test("entry sizing absorbs provider dust before applying the trade cap", async () => {
  const api = new FakeApi();
  api.capacity = { quote: 100, leverage: 1 };
  const strategy = new FakeStrategy();
  const nextConfig = config();
  nextConfig.minTradeQuote = 10;
  nextConfig.maxTradeQuote = 98;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });
  strategy.entry = { side: "long", size: 0.95, leverage: 1, price: null, confidence: null };

  await bot.onTick(tick);

  assert.equal(api.orders[0].order.size, 0.98);
});

test("internal borrowing locks only the amount borrowed from the lender", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  const nextConfig = config();
  nextConfig.internalBorrow.enabled = true;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.entry = { side: "long", size: 0.5, leverage: 2, price: null, confidence: null };
  await bot.onTick(tick);
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 10, filledQuote: 1_000, remaining: 0 },
  });
  strategy.entry = { side: "short", size: 0.5, leverage: 2, price: null, confidence: null };
  await bot.onTick({ ...tick, timestamp: 2_000 });
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[1].order.id,
    fill: { filledAsset: 10, filledQuote: 1_000, remaining: 0 },
  });

  const snapshot = await bot.snapshot();
  const long = snapshot.positions.find((position) => position.side === "long")!;
  const short = snapshot.positions.find((position) => position.side === "short")!;
  assert.deepEqual(short.internalBorrow, [{
    positionId: long.id,
    asset: 5,
    quote: 0,
  }]);
  assert.equal(short.externalBorrow.asset, 0);

  strategy.exit = { side: "long", size: 1, price: null, confidence: null };
  await bot.onTick({ ...tick, timestamp: 3_000 });
  assert.equal(api.orders[2].order.size, 5);
});

class FakeApi implements TradingApi {
  orders: TradingOrderResult[] = [];
  capacity?: { quote: number; leverage: number };

  createStopMarketOrder = this.create.bind(this, "stop-market");
  createStopLimitOrder = this.create.bind(this, "stop-limit");
  createLimitOrder = this.create.bind(this, "limit");
  createMarketOrder = this.create.bind(this, "market");

  async cancelOrder(): Promise<boolean> {
    return true;
  }

  async getHistory() {
    return [];
  }

  async getMarketRules() {
    const quantity = { min: null, max: null, step: null };
    return {
      price: quantity,
      limitQuantity: quantity,
      marketQuantity: quantity,
      minNotional: null,
      maxNotional: null,
      maxLeverage: 10,
    };
  }

  async getOrderCapacity(input: { leverage: number }) {
    return this.capacity ?? { quote: 1_000 * input.leverage, leverage: input.leverage };
  }

  async getFriction() {
    return 0;
  }

  private async create(type: "market" | "limit" | "stop-market" | "stop-limit", input: {
    side: "buy" | "sell";
    size: number;
    price?: number;
    stopPrice?: number;
    limitPrice?: number;
  }): Promise<TradingOrderResult> {
    const result: TradingOrderResult = {
      accepted: true,
      order: {
        id: `order-${this.orders.length + 1}`,
        type,
        side: input.side,
        status: "open",
        size: input.size,
        price: input.price ?? input.limitPrice ?? null,
        stopPrice: input.stopPrice ?? null,
      },
    };
    this.orders.push(result);
    return result;
  }
}

class FakeStrategy implements TradingStrategy<unknown, StrategySnapshot, StrategyDiagnostics> {
  entry: TradingStrategyEntrySignal | null = null;
  exit: TradingStrategyExitSignal | null = null;
  restores = 0;
  updates = 0;

  async warmup() {}
  async onTick() {}
  async entrySignal() {
    const signal = this.entry;
    this.entry = null;
    return signal;
  }
  async exitSignal() {
    const signal = this.exit;
    this.exit = null;
    return signal;
  }
  async snapshot() {
    return { version: 1 };
  }
  async restore() { this.restores += 1; }
  async updateConfig() { this.updates += 1; }
  getDiagnostics() {
    return { indicators: {}, gates: [], blockers: [], lastSignal: null };
  }
}

function config(): TradingBotConfig {
  return {
    strategy: {},
    maxTargetLeverage: 5,
    minTradeQuote: 1,
    maxTradeQuote: 1_000,
    entryGrid: { orderCount: 1, maxPriceStep: 0, sizeDistribution: "linear", sizeFraction: 1 },
    exitGrid: {
      orderCount: 1,
      maxPriceStep: 0,
      sizeDistribution: "linear",
      sizeFraction: 1,
      reset: "previous-anchor",
    },
    positionLifetimeMs: null,
    stopLossRate: null,
    takeProfitRate: null,
    cooldownMs: 0,
    internalBorrow: { enabled: false, lockLenderAmounts: true, borrowerProfitShare: 1 },
  };
}
