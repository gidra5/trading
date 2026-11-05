import assert from "node:assert/strict";
import test from "node:test";
import type { BinanceClientConfig } from "../src/clients/binance-types.js";
import type { PredictionQuote } from "../src/clients/binance-predictions.js";
import type { BotConfigureRequest } from "../src/schemas.js";
import type {
  TradingCandle,
  TradingMarketRules,
} from "@trading/bot-algo/trading-api";
import {
  BacktestTradingApi,
  type BacktestTradingApiConfig,
} from "../src/clients/backtest.js";

const candle = (
  index: number,
  close = 100,
  open = close,
  high = Math.max(open, close),
  low = Math.min(open, close),
): TradingCandle => ({
  openTime: index * 60_000,
  closeTime: (index + 1) * 60_000 - 1,
  open,
  high,
  low,
  close,
  volume: 10,
});
const config = (
  overrides: Partial<BacktestTradingApiConfig> = {},
): BacktestTradingApiConfig => ({
  initialQuote: 1000,
  leverage: 5,
  feeRate: 0,
  slippageRate: 0,
  maintenanceMarginRate: 0.005,
  liquidationFeeRate: 0.002,
  marketRules: {
    price: { min: null, max: null, step: null },
    limitQuantity: { min: null, max: null, step: null },
    marketQuantity: { min: null, max: null, step: null },
    minNotional: null,
    maxNotional: null,
    maxLeverage: 125,
  },
  candlePath: "open-high-low-close",
  startIndex: 1,
  ...overrides,
});
const near = (actual: number, expected: number) =>
  assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} != ${expected}`);

// Compile-only contract checks; deliberately never executed.
function immutableContracts(
  api: BacktestTradingApi,
  settings: BacktestTradingApiConfig,
  history: readonly TradingCandle[],
  connection: BinanceClientConfig,
  bot: BotConfigureRequest,
  quote: PredictionQuote,
) {
  // @ts-expect-error Config values are readonly.
  settings.feeRate = 0;
  // @ts-expect-error Nested market rules are readonly.
  settings.marketRules.price.step = 1;
  // @ts-expect-error Funding arrays are readonly.
  settings.funding!.push({ timestamp: 0, rate: 0, markPrice: 1 });
  // @ts-expect-error Candle fields are readonly.
  history[0].close = 1;
  // @ts-expect-error History arrays are readonly.
  history.push(candle(1));
  // @ts-expect-error Returned order snapshots are readonly.
  api.getOrder("order")!.status = "filled";
  // @ts-expect-error Account snapshots are readonly.
  api.snapshot().walletBalance = 0;
  // @ts-expect-error Binance connection configuration is readonly.
  connection.environment = "live";
  // @ts-expect-error Zod-derived nested credentials are readonly.
  bot.credentials.apiSecret = "changed";
  // @ts-expect-error Prediction quotes are readonly.
  quote.amountIn = "0";
}

test("history exposes only completed replay data, aggregates full buckets, and shares immutable candles", async () => {
  const history = [candle(0), candle(1, 110), candle(2, 120), candle(3, 130)];
  const api = new BacktestTradingApi(history, config());
  assert.equal(
    (await api.getHistory({ intervalMs: 60_000, count: 10 }))[0].close,
    100,
  );
  assert.deepEqual(
    await api.getHistory({ intervalMs: 120_000, count: 10 }),
    [],
  );
  api.advance();
  const aggregated = await api.getHistory({ intervalMs: 120_000, count: 10 });
  assert.equal(aggregated.length, 1);
  assert.deepEqual(
    [aggregated[0].open, aggregated[0].close, aggregated[0].volume],
    [100, 110, 20],
  );
  assert.equal(history[0].close, 100);
  assert.equal(
    (await api.getHistory({ intervalMs: 120_000, count: 10 }))[0].close,
    110,
  );
  assert.deepEqual(await api.getHistory({ intervalMs: 60_000, count: 0 }), []);
  await assert.rejects(api.getHistory({ intervalMs: 30_000, count: 1 }));
});

test("missing source candles are never presented as complete aggregated history", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(2)],
    config({ startIndex: 2 }),
  );
  assert.deepEqual(await api.getHistory({ intervalMs: 180_000, count: 1 }), []);
});

test("markets fill at the next open; short reversal realizes P&L and keeps synthetic equity consistent", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1, 120, 110), candle(2, 90, 120), candle(3)],
    config(),
  );
  const buy = await api.createMarketOrder({ side: "buy", size: 2 });
  assert.equal(buy.order.status, "pending");
  assert.equal(api.snapshot().positionSize, 0);
  api.advance();
  assert.equal(api.getOrder(buy.order.id)?.status, "filled");
  assert.equal(api.snapshot().entryPrice, 110);
  assert.equal(api.snapshot().equity, 1020);
  await api.createMarketOrder({ side: "sell", size: 3 });
  api.advance();
  const account = api.snapshot();
  assert.equal(account.positionSize, -1);
  assert.equal(account.entryPrice, 120);
  assert.equal(account.walletBalance, 1020);
  assert.equal(account.equity, 1050);
  const equity = await api.getEquity();
  assert.equal(equity.quoteUnleveraged + equity.assetUnleveraged * 90, 1050);
});

test("fees and market slippage affect wallet, equity and fill events exactly once", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1), candle(2)],
    config({ feeRate: 0.001, slippageRate: 0.01 }),
  );
  await api.createMarketOrder({ side: "buy", size: 2 });
  api.drainOrderEvents();
  api.advance();
  const event = api.drainOrderEvents()[0];
  assert.equal(event.type, "fill");
  if (event.type !== "fill") return;
  near(event.fill.filledQuote, 202.202);
  near(event.fill.feeQuote!, 2.202);
  near(api.snapshot().walletBalance, 999.798);
  near(api.snapshot().equity, 997.798);
  assert.deepEqual(api.drainOrderEvents(), []);
});

test("resting limits fill at their crossing; gaps receive improvement and cancellation releases reservations", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1, 100, 80, 110, 70), candle(2)],
    config(),
  );
  const first = await api.createLimitOrder({ side: "buy", size: 1, price: 90 });
  const canceled = await api.createLimitOrder({
    side: "buy",
    size: 1,
    price: 75,
  });
  near((await api.getEquity()).quoteReserved, 33);
  assert.equal(await api.cancelOrder(canceled.order.id), true);
  assert.equal(await api.cancelOrder(canceled.order.id), false);
  near((await api.getEquity()).quoteReserved, 18);
  api.advance();
  assert.equal(api.snapshot().entryPrice, 80);
  assert.equal(api.getOrder(first.order.id)?.status, "filled");
  assert.equal(api.getOrder(canceled.order.id)?.status, "canceled");
});

test("stop-limit never fills a low that occurred before its trigger", async () => {
  const api = new BacktestTradingApi(
    [
      candle(0),
      candle(1, 115, 100, 120, 90),
      candle(2, 105, 115, 116, 104),
      candle(3),
    ],
    config({ candlePath: "open-low-high-close" }),
  );
  const order = await api.createStopLimitOrder({
    side: "buy",
    size: 1,
    stopPrice: 110,
    limitPrice: 105,
  });
  api.advance();
  assert.equal(api.getOrder(order.order.id)?.status, "open");
  assert.equal(api.snapshot().positionSize, 0);
  api.advance();
  assert.equal(api.getOrder(order.order.id)?.status, "filled");
  assert.equal(api.snapshot().entryPrice, 105);
});

test("stop-market fills at a gap price and reduce-only cannot reverse a position", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1), candle(2, 80), candle(3)],
    config(),
  );
  await api.createMarketOrder({ side: "buy", size: 1 });
  api.advance();
  const invalid = await api.createMarketOrder({
    side: "sell",
    size: 2,
    reduceOnly: true,
  });
  assert.equal(invalid.accepted, false);
  const stop = await api.createStopMarketOrder({
    side: "sell",
    size: 1,
    price: 90,
    reduceOnly: true,
  });
  api.advance();
  assert.equal(api.getOrder(stop.order.id)?.status, "filled");
  assert.equal(api.snapshot().positionSize, 0);
  assert.equal(api.snapshot().walletBalance, 980);
});

test("margin prevents simultaneous overcommitment and rechecks orders after price gaps", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1, 200), candle(2)],
    config({ initialQuote: 100, leverage: 2 }),
  );
  const first = await api.createMarketOrder({ side: "buy", size: 1.5 });
  const second = await api.createMarketOrder({ side: "buy", size: 1 });
  assert.equal(first.accepted, true);
  assert.equal(second.accepted, false);
  api.advance();
  assert.equal(api.getOrder(first.order.id)?.status, "rejected");
  assert.equal(api.snapshot().positionSize, 0);
});

test("short capacity exists without spot inventory; price and quantity rules are enforced", async () => {
  const rules: TradingMarketRules = {
    price: { min: 1, max: 1000, step: 0.5 },
    limitQuantity: { min: 0.1, max: 10, step: 0.1 },
    marketQuantity: { min: 0.1, max: 10, step: 0.1 },
    minNotional: 5,
    maxNotional: 1000,
    maxLeverage: 10,
  };
  const api = new BacktestTradingApi(
    [candle(0), candle(1)],
    config({ marketRules: rules }),
  );
  assert.equal(
    (await api.getOrderCapacity({ side: "sell", price: 100, leverage: 3 }))
      .quote,
    1000,
  );
  assert.equal(
    (await api.createMarketOrder({ side: "sell", size: 0.15 })).accepted,
    false,
  );
  assert.equal(
    (await api.createLimitOrder({ side: "buy", size: 1, price: 99.9 }))
      .accepted,
    false,
  );
  assert.equal(
    (await api.createMarketOrder({ side: "sell", size: 1 })).accepted,
    true,
  );
});

test("funding debits longs and credits shorts using supplied historical rates", async () => {
  for (const side of ["buy", "sell"] as const) {
    const api = new BacktestTradingApi(
      [candle(0), candle(1), candle(2)],
      config({
        funding: [{ timestamp: 90_000, rate: 0.01, markPrice: 100 }],
      }),
    );
    await api.createMarketOrder({ side, size: 2 });
    api.advance();
    const charge = side === "buy" ? 2 : -2;
    assert.equal(api.snapshot().walletBalance, 1000 - charge);
    assert.equal(api.snapshot().fundingPaid, charge);
    assert.ok(
      api
        .drainOrderEvents()
        .some(
          (event) =>
            event.type === "maintenance" && event.quoteCharge === charge,
        ),
    );
  }
});

test("liquidation occurs at a crossed threshold before a later stop and prevents new orders", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1), candle(2, 100, 100, 101, 70), candle(3)],
    config({ initialQuote: 100, leverage: 10 }),
  );
  await api.createMarketOrder({ side: "buy", size: 9 });
  api.advance();
  const stop = await api.createStopMarketOrder({
    side: "sell",
    size: 9,
    price: 80,
    reduceOnly: true,
  });
  assert.equal(stop.accepted, true);
  api.advance();
  assert.equal(api.snapshot().liquidated, true);
  assert.equal(api.snapshot().positionSize, 0);
  const liquidation = api
    .drainOrderEvents()
    .find((event) => event.type === "liquidation");
  assert.ok(liquidation && liquidation.price > 80 && liquidation.price < 100);
  assert.equal(api.getOrder(stop.order.id)?.status, "canceled");
  assert.equal(
    (await api.createMarketOrder({ side: "buy", size: 1 })).accepted,
    false,
  );
});

test("replay finishes explicitly and malformed histories/configuration fail early", async () => {
  const api = new BacktestTradingApi([candle(0)], config({ startIndex: 0 }));
  assert.deepEqual(await api.getHistory({ intervalMs: 60_000, count: 5 }), []);
  assert.ok(api.advance());
  assert.equal(api.advance(), null);
  assert.equal(
    (await api.createMarketOrder({ side: "buy", size: 1 })).accepted,
    false,
  );
  assert.throws(() => new BacktestTradingApi([candle(1), candle(0)], config()));
  assert.throws(
    () => new BacktestTradingApi([candle(0)], config({ leverage: 0 })),
  );
  await assert.rejects(
    api.createLimitOrder({ side: "buy", size: 1, price: NaN }),
  );
});

test("the selected candle path determines which protective order executes first", async () => {
  for (const candlePath of [
    "open-high-low-close",
    "open-low-high-close",
  ] as const) {
    const api = new BacktestTradingApi(
      [candle(0), candle(1), candle(2, 100, 100, 120, 80), candle(3)],
      config({ candlePath }),
    );
    await api.createMarketOrder({ side: "buy", size: 1 });
    api.advance();
    const profit = await api.createLimitOrder({
      side: "sell",
      size: 1,
      price: 110,
      reduceOnly: true,
    });
    const stop = await api.createStopMarketOrder({
      side: "sell",
      size: 1,
      price: 90,
      reduceOnly: true,
    });
    api.advance();
    const highFirst = candlePath === "open-high-low-close";
    assert.equal(api.snapshot().walletBalance, highFirst ? 1010 : 990);
    assert.equal(
      api.getOrder(highFirst ? profit.order.id : stop.order.id)?.status,
      "filled",
    );
    assert.equal(
      api.getOrder(highFirst ? stop.order.id : profit.order.id)?.status,
      "rejected",
    );
    assert.equal(api.snapshot().positionSize, 0);
  }
});

test("price crossings execute chronologically rather than in submission order", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1, 100, 100, 100, 80), candle(2)],
    config(),
  );
  const lower = await api.createLimitOrder({ side: "buy", size: 1, price: 90 });
  const higher = await api.createLimitOrder({
    side: "buy",
    size: 1,
    price: 95,
  });
  api.drainOrderEvents();
  api.advance();
  const fills = api
    .drainOrderEvents()
    .flatMap((event) => (event.type === "fill" ? [event] : []));
  assert.deepEqual(
    fills.map((event) => event.orderId),
    [higher.order.id, lower.order.id],
  );
  assert.deepEqual(
    fills.map((event) => event.fill.price),
    [95, 90],
  );
  assert.equal(api.snapshot().entryPrice, 92.5);
});

test("fees, slippage and synthetic equity reconcile across a complete round trip", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1), candle(2), candle(3)],
    config({ feeRate: 0.001, slippageRate: 0.01 }),
  );
  await api.createMarketOrder({ side: "buy", size: 2 });
  api.advance();
  await api.createMarketOrder({ side: "sell", size: 2 });
  api.advance();
  const snapshot = api.snapshot();
  near(snapshot.realizedPnl, -4);
  near(snapshot.feesPaid, 0.4);
  near(snapshot.equity, 995.6);
  assert.equal(snapshot.entryPrice, 0);
  const fills = api
    .drainOrderEvents()
    .flatMap((event) => (event.type === "fill" ? [event] : []));
  near(fills[1].fill.filledQuote, 197.802);
  near(
    fills.reduce(
      (cash, event) =>
        cash - Math.sign(event.fill.filledAsset) * event.fill.filledQuote,
      1000,
    ),
    snapshot.equity,
  );
});

test("size capacity includes costs, rounds to the lot grid, and reserves pending exposure", async () => {
  const settings = config({
    initialQuote: 100,
    leverage: 2,
    feeRate: 0.001,
    slippageRate: 0.01,
  });
  const api = new BacktestTradingApi([candle(0), candle(1), candle(2)], {
    ...settings,
    marketRules: {
      ...settings.marketRules,
      marketQuantity: { ...settings.marketRules.marketQuantity, step: 0.1 },
    },
  });
  const capacity = await api.getOrderCapacity({
    side: "buy",
    price: 100,
    leverage: 10,
  });
  assert.equal(capacity.leverage, 2);
  near(capacity.quote, 190);
  assert.equal(
    (await api.createMarketOrder({ side: "buy", size: capacity.quote / 100 }))
      .accepted,
    true,
  );
  assert.equal(
    (await api.createMarketOrder({ side: "buy", size: 0.1 })).accepted,
    false,
  );
  api.advance();
  near(api.snapshot().positionSize, 1.9);
  assert.ok(api.snapshot().quoteAvailable >= 0);
});

test("funding is applied at its replay timestamp and can liquidate an open position", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1), candle(2)],
    config({
      initialQuote: 100,
      leverage: 10,
      funding: [{ timestamp: 90000, rate: 0.11, markPrice: 100 }],
    }),
  );
  await api.createMarketOrder({ side: "buy", size: 9 });
  api.advance();
  const events = api.drainOrderEvents();
  const liquidation = events.find((event) => event.type === "liquidation");
  assert.ok(liquidation);
  assert.equal(liquidation.at, 90000);
  near(api.snapshot().fundingPaid, 99);
  assert.equal(api.snapshot().liquidated, true);
});

test("funding during missing candles applies before orders at the next open", async () => {
  const api = new BacktestTradingApi(
    [candle(0), candle(1), candle(4), candle(5)],
    config({
      funding: [{ timestamp: 180000, rate: 0.01, markPrice: 120 }],
    }),
  );
  await api.createMarketOrder({ side: "buy", size: 2 });
  api.advance();
  await api.createMarketOrder({ side: "sell", size: 2, reduceOnly: true });
  api.advance();
  near(api.snapshot().fundingPaid, 2.4);
  near(api.snapshot().walletBalance, 997.6);
});

test("short liquidation uses the upward crossing and a gap uses the observed open", async () => {
  for (const gap of [false, true]) {
    const api = new BacktestTradingApi(
      [
        candle(0),
        candle(1),
        gap ? candle(2, 130) : candle(2, 100, 100, 130, 100),
        candle(3),
      ],
      config({ initialQuote: 100, leverage: 10 }),
    );
    await api.createMarketOrder({ side: "sell", size: 9 });
    api.advance();
    api.advance();
    const liquidation = api
      .drainOrderEvents()
      .find((event) => event.type === "liquidation");
    assert.ok(liquidation);
    if (gap) assert.equal(liquidation.price, 130);
    else near(liquidation.price, 1000 / (9 * 1.005));
    assert.equal(api.snapshot().positionSize, 0);
  }
});

test("immutable inputs are shared and prior order snapshots stay stable", async () => {
  const history = Object.freeze(
    [candle(0), candle(1), candle(2)].map((candle) => Object.freeze(candle)),
  );
  const settings = config();
  const api = new BacktestTradingApi(history, settings);
  const previousHistory = await api.getHistory({
    intervalMs: 60000,
    count: 10,
  });
  assert.equal(previousHistory[0], history[0]);
  assert.equal(await api.getMarketRules(), settings.marketRules);
  const order = await api.createMarketOrder({ side: "buy", size: 1 });
  assert.equal(api.getOrder(order.order.id), order.order);
  const events = api.drainOrderEvents();
  assert.equal(events[0].type, "open");
  if (events[0].type === "open") assert.equal(events[0].order, order.order);
  const tick = api.advance()!;
  assert.equal(tick.candle, history[1]);
  assert.equal(previousHistory.length, 1);
  assert.equal(order.order.status, "pending");
  assert.equal(api.getOrder(order.order.id)?.status, "filled");
  assert.notEqual(api.getOrder(order.order.id), order.order);
  assert.equal(api.snapshot().positionSize, 1);
  assert.equal(api.snapshot().equity, 1000);
});

test("exhaustion releases unfilled orders and empty histories are explicit", async () => {
  const api = new BacktestTradingApi([candle(0), candle(1)], config());
  const order = await api.createLimitOrder({ side: "buy", size: 1, price: 50 });
  api.advance();
  assert.equal(api.getOrder(order.order.id)?.status, "canceled");
  assert.equal((await api.getEquity()).quoteReserved, 0);
  assert.equal(api.advance(), null);
  const empty = new BacktestTradingApi([], config({ startIndex: 0 }));
  assert.equal(empty.advance(), null);
  assert.deepEqual(await empty.getHistory({ intervalMs: 60000, count: 1 }), []);
});

test("invalid rates, rules, funding records, and source intervals fail before replay", () => {
  for (const override of [
    { feeRate: -1 },
    { slippageRate: NaN },
    { maintenanceMarginRate: 1 },
    { liquidationFeeRate: Infinity },
    { startIndex: 3 },
    {
      funding: [
        { timestamp: 2, rate: 0, markPrice: 100 },
        { timestamp: 1, rate: 0, markPrice: 100 },
      ],
    },
  ])
    assert.throws(
      () => new BacktestTradingApi([candle(0), candle(1)], config(override)),
    );
  const valid = config();
  const invalid = {
    ...valid,
    marketRules: {
      ...valid.marketRules,
      marketQuantity: { ...valid.marketRules.marketQuantity, step: 0 },
    },
  };
  assert.throws(() => new BacktestTradingApi([candle(0)], invalid));
  assert.throws(
    () =>
      new BacktestTradingApi(
        [candle(0), { ...candle(1), closeTime: 70000 }],
        config(),
      ),
  );
});
