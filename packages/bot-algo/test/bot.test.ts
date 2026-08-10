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
  type TradingStrategyTargetExposureContext,
  type TradingStrategyTargetExposureSignal,
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
  strategy.entry = { side: "long", size: 0.5, leverage: 1, price: null, confidence: 1 };

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

  strategy.entry = { side: "short", size: 1, leverage: 2, price: null, confidence: 1 };
  await bot.onTick(tick);
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 2, filledQuote: 200, remaining: 0 },
  });
  strategy.entry = { side: "long", size: 1, leverage: 2, price: null, confidence: 1 };
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
  strategy.entry = { side: "short", size: 0.25, leverage: 1, price: null, confidence: 1 };
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
  strategy.entry = { side: "short", size: 0.5, leverage: 5, price: null, confidence: 1 };

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
  strategy.entry = { side: "short", size: 1, leverage: 1, price: 100, confidence: 1 };

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
  strategy.entry = { side: "long", size: 0.95, leverage: 1, price: null, confidence: 1 };

  await bot.onTick(tick);

  assert.equal(api.orders[0].order.size, 0.98);
});

test("conventional entries use strategy and signal confidence exposure controls", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  strategy.modelConfidence = 0.5;
  const nextConfig = config();
  useOracleExposureControls(nextConfig);
  nextConfig.maxTargetLeverage = 10;
  nextConfig.maxTradeQuote = 10_000;
  nextConfig.exposureControl.expansionConfirmationMass = 0;
  nextConfig.exposureControl.expansionDeltaCapFraction = 1;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  // Static confidence 0.5 raises the signal-confidence threshold to 0.275.
  strategy.entry = { side: "long", size: 1, leverage: 8, price: null, confidence: 0.27 };
  await bot.onTick(tick);
  assert.equal(api.orders.length, 0);

  strategy.entry = { side: "long", size: 1, leverage: 8, price: null, confidence: 0.28 };
  await bot.onTick({ ...tick, timestamp: 2_000 });
  assert.equal(api.orders.length, 1);
  // Confidence permits 2.75x, while the quadratic per-decision delta cap permits 2.5x.
  assert.equal((await bot.snapshot()).positions[0].leverage, 2.5);
  assert.equal(api.orders[0].order.size, 25);
});

test("conventional entry confirmations accumulate signal confidence", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  const nextConfig = config();
  nextConfig.exposureControl.expansionConfirmationMass = 1.5;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  for (let observation = 1; observation <= 2; observation += 1) {
    strategy.entry = { side: "long", size: 1, leverage: 2, price: null, confidence: 0.5 };
    await bot.onTick({ ...tick, timestamp: observation * 1_000 });
  }
  assert.equal(api.orders.length, 0);
  assert.equal((await bot.snapshot()).signalConfirmation?.confirmationMass, 1);

  strategy.entry = { side: "long", size: 1, leverage: 2, price: null, confidence: 0.5 };
  await bot.onTick({ ...tick, timestamp: 3_000 });
  assert.equal(api.orders.length, 1);
  assert.equal((await bot.snapshot()).positions[0].leverage, 2);
});

test("target exposure expands and reduces positions through the bot contract", async () => {
  const api = new FakeApi();
  const strategy = new FakeTargetStrategy();
  const nextConfig = config();
  nextConfig.maxTargetLeverage = 10;
  nextConfig.maxTradeQuote = 10_000;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.target = {
    targetExposure: 3, price: 100, confidence: 1,
  };
  await bot.onTick(tick);
  assert.equal(api.orders[0].order.side, "buy");
  assert.equal(api.orders[0].order.size, 30);
  assert.equal((await bot.snapshot()).positions[0].leverage, 3);
  assert.equal(strategy.contexts[0]?.currentExposure, 0);

  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 30, filledQuote: 3_000, remaining: 0 },
  });
  strategy.target = {
    targetExposure: 1, price: 100, confidence: 1,
  };
  await bot.onTick({ ...tick, timestamp: 2_000 });

  assert.equal(strategy.contexts[1]?.currentExposure, 3);
  assert.equal(api.orders[1].order.side, "sell");
  assert.equal(api.orders[1].order.size, 20);
});

test("target exposure reversals exit before entering the opposite side", async () => {
  const api = new FakeApi();
  const strategy = new FakeTargetStrategy();
  const nextConfig = config();
  nextConfig.maxTargetLeverage = 10;
  nextConfig.maxTradeQuote = 10_000;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.target = {
    targetExposure: 2, price: 100, confidence: 1,
  };
  await bot.onTick(tick);
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 20, filledQuote: 2_000, remaining: 0 },
  });
  strategy.target = {
    targetExposure: -2, price: 100, confidence: 1,
  };
  await bot.onTick({ ...tick, timestamp: 2_000 });

  assert.equal(api.orders[1].order.type, "market");
  assert.equal(api.orders[1].order.side, "sell");
  assert.equal(api.orders[1].order.size, 20);
  assert.equal(api.orders.length, 2);
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[1].order.id,
    fill: { filledAsset: 20, filledQuote: 2_000, remaining: 0 },
  });
  strategy.target = {
    targetExposure: -2, price: 100, confidence: 1,
  };
  await bot.onTick({ ...tick, timestamp: 3_000 });
  assert.equal(api.orders[2].order.type, "limit");
  assert.equal(api.orders[2].order.side, "sell");
  assert.equal(api.orders[2].order.size, 20);
});

test("target exposure rate-limits expansions quadratically by static confidence", async () => {
  const api = new FakeApi();
  const strategy = new FakeTargetStrategy();
  const nextConfig = config();
  useOracleExposureControls(nextConfig);
  nextConfig.maxTargetLeverage = 10;
  nextConfig.maxTradeQuote = 10_000;
  strategy.modelConfidence = 0.5;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.target = {
    targetExposure: 8, price: 100, confidence: 1,
  };
  await bot.onTick(tick);

  assert.equal((await bot.snapshot()).positions[0].leverage, 1.875);
  assert.equal(api.orders[0].order.size, 18.75);
});

test("target exposure applies the bot's configurable confidence leverage floor", async () => {
  const api = new FakeApi();
  const strategy = new FakeTargetStrategy();
  const nextConfig = config();
  nextConfig.maxTargetLeverage = 10;
  nextConfig.maxTradeQuote = 10_000;
  nextConfig.exposureControl.confidenceLeverageFloor = 0.75;
  nextConfig.exposureControl.minimumSignalConfidence = 0;
  nextConfig.exposureControl.maximumSignalConfidenceThreshold = 0;
  nextConfig.exposureControl.expansionConfirmationMass = 0;
  nextConfig.exposureControl.expansionDeltaCapFraction = 1;
  strategy.modelConfidence = 0.5;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.target = { targetExposure: 8, price: 100, confidence: 0 };
  await bot.onTick(tick);

  // 10 * 0.5 * (0.75 * 0.5) = 1.875x.
  assert.equal((await bot.snapshot()).positions[0].leverage, 1.875);
  assert.equal(api.orders[0].order.size, 18.75);
});

test("target exposure gates expansions on quality-scaled distribution confidence", async () => {
  const api = new FakeApi();
  const strategy = new FakeTargetStrategy();
  const nextConfig = config();
  useOracleExposureControls(nextConfig);
  nextConfig.maxTargetLeverage = 10;
  nextConfig.maxTradeQuote = 10_000;
  nextConfig.exposureControl.expansionConfirmationMass = 0;
  strategy.modelConfidence = 0.5;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  // At static confidence 0.5, the threshold is 0.05 + (0.5 - 0.05) * 0.5 = 0.275.
  strategy.target = {
    targetExposure: 3, price: 100, confidence: 0.27,
  };
  await bot.onTick(tick);
  assert.equal(api.orders.length, 0);

  strategy.target = {
    targetExposure: 3, price: 100, confidence: 0.28,
  };
  await bot.onTick({ ...tick, timestamp: 2_000 });
  assert.equal(api.orders.length, 1);
  assert.equal((await bot.snapshot()).positions[0].leverage, 1.875);
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 18.75, filledQuote: 1_875, remaining: 0 },
  });

  // Risk reductions bypass the confidence gate.
  strategy.target = {
    targetExposure: 1, price: 100, confidence: 0,
  };
  await bot.onTick({ ...tick, timestamp: 3_000 });
  assert.equal(api.orders[1].order.side, "sell");
  assert.equal(api.orders[1].order.size, 8.75);
});

test("target exposure confirmations accumulate distribution confidence and interpolate leverage", async () => {
  const api = new FakeApi();
  const strategy = new FakeTargetStrategy();
  const nextConfig = config();
  useOracleExposureControls(nextConfig);
  nextConfig.maxTargetLeverage = 100;
  nextConfig.maxTradeQuote = 100_000;
  nextConfig.exposureControl.expansionConfirmationMass = 1.5;
  strategy.modelConfidence = 0.5;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.target = {
    targetExposure: 3, price: 100, confidence: 0.5,
  };
  await bot.onTick(tick);
  strategy.target = {
    targetExposure: 5, price: 100, confidence: 0.5,
  };
  await bot.onTick({ ...tick, timestamp: 2_000 });
  assert.equal(api.orders.length, 0);
  assert.equal((await bot.snapshot()).signalConfirmation?.confirmationMass, 1);
  strategy.target = {
    targetExposure: 4, price: 100, confidence: 0.5,
  };
  await bot.onTick({ ...tick, timestamp: 3_000 });

  assert.equal((await bot.snapshot()).positions[0].leverage, 4);
  assert.equal(api.orders[0].order.size, 40);
});

test("internal borrowing locks only the amount borrowed from the lender", async () => {
  const api = new FakeApi();
  const strategy = new FakeStrategy();
  const nextConfig = config();
  nextConfig.internalBorrow.enabled = true;
  const bot = new GridTradingBot({ api, strategy, config: nextConfig });

  strategy.entry = { side: "long", size: 0.5, leverage: 2, price: null, confidence: 1 };
  await bot.onTick(tick);
  await bot.onOrder({
    type: "fill",
    orderId: api.orders[0].order.id,
    fill: { filledAsset: 10, filledQuote: 1_000, remaining: 0 },
  });
  strategy.entry = { side: "short", size: 0.5, leverage: 2, price: null, confidence: 1 };
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
  equity = {
    quoteAvailable: 1_000,
    quoteReserved: 0,
    quoteUnleveraged: 1_000,
    assetAvailable: 0,
    assetReserved: 0,
    assetUnleveraged: 0,
  };

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

  async getEquity() {
    return this.equity;
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
  modelConfidence = 1;

  async warmup() {}
  async onTick() {}
  staticConfidence() { return this.modelConfidence; }
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

class FakeTargetStrategy extends FakeStrategy {
  target: TradingStrategyTargetExposureSignal | null = null;
  contexts: TradingStrategyTargetExposureContext[] = [];

  async targetExposureSignal(context: TradingStrategyTargetExposureContext) {
    this.contexts.push(context);
    const signal = this.target;
    this.target = null;
    return signal;
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
    exposureControl: {
      confidenceLeverageFloor: 1,
      minimumSignalConfidence: 0,
      maximumSignalConfidenceThreshold: 0,
      expansionConfirmationMass: 0,
      expansionDeltaCapFraction: 1,
    },
    internalBorrow: { enabled: false, lockLenderAmounts: true, borrowerProfitShare: 1 },
  };
}

function useOracleExposureControls(config: TradingBotConfig): void {
  config.exposureControl = {
    confidenceLeverageFloor: 0.75,
    minimumSignalConfidence: 0.05,
    maximumSignalConfidenceThreshold: 0.5,
    expansionConfirmationMass: 1,
    expansionDeltaCapFraction: 0.75,
  };
}
