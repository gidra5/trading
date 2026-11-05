import assert from "node:assert/strict";
import test from "node:test";
import type { Kline } from "binance";
import {
  BinanceClient,
  getBinanceTradingApi,
  BinanceOrderSubmissionError,
  type FuturesTransport,
  type BinanceTradingApiConfig,
} from "../src/clients/binance.js";

const config: BinanceTradingApiConfig & { product: "futures" } = {
  environment: "demo",
  symbol: "BTCUSDT",
  apiKey: "test-key",
  apiSecret: "test-secret",
  product: "futures",
};
const symbol = {
  symbol: "BTCUSDT",
  status: "TRADING",
  quoteAsset: "USDT",
  marginAsset: "USDT",
  baseAsset: "BTC",
  filters: [
    {
      filterType: "PRICE_FILTER",
      minPrice: "0.1",
      maxPrice: "1000000",
      tickSize: "0.1",
    },
    {
      filterType: "LOT_SIZE",
      minQty: "0.001",
      maxQty: "1000",
      stepSize: "0.001",
    },
    {
      filterType: "MARKET_LOT_SIZE",
      minQty: "0",
      maxQty: "100",
      stepSize: "0",
    },
    { filterType: "MIN_NOTIONAL", notional: "5" },
  ],
};
const account = {
  multiAssetsMargin: false,
  assets: [
    {
      asset: "USDT",
      walletBalance: "1000",
      marginBalance: "1100",
      availableBalance: "1060",
      initialMargin: "40",
    },
  ],
  positions: [
    {
      symbol: "BTCUSDT",
      positionSide: "BOTH",
      positionAmt: "-2",
      notional: "-200",
    },
  ],
};

async function fixture(overrides: Partial<FuturesTransport> = {}) {
  const calls: { method: string; params: unknown }[] = [];
  // Fixtures intentionally omit SDK response fields that the adapter does not consume.
  const client = {
    getExchangeInfo: async () => ({ symbols: [symbol] }),
    getFuturesAccountConfig: async () => ({
      dualSidePosition: false,
      multiAssetsMargin: false,
      canTrade: true,
    }),
    getFuturesSymbolConfig: async () => [
      { symbol: "BTCUSDT", marginType: "CROSSED", leverage: 10 },
    ],
    getNotionalAndLeverageBrackets: async () => [
      {
        symbol: "BTCUSDT",
        brackets: [
          { initialLeverage: 125, notionalCap: 1000 },
          { initialLeverage: 10, notionalCap: 10000 },
        ],
      },
    ],
    getAccountInformationV3: async () => account,
    getAccountCommissionRate: async () => ({
      makerCommissionRate: "0.0002",
      takerCommissionRate: "0.0005",
    }),
    setLeverage: async (params: unknown) => {
      calls.push({ method: "setLeverage", params });
      return { leverage: 125, maxNotionalValue: "1000" };
    },
    getKlines: async () => [],
    getAggregateTrades: async () => [],
    getOrderBook: async () => ({
      bids: [["100", "100"]],
      asks: [["100", "100"]],
    }),
    submitNewOrder: async (params: unknown) => {
      calls.push({ method: "submitNewOrder", params });
      return { orderId: 42, status: "NEW" };
    },
    submitNewAlgoOrder: async (params: unknown) => {
      calls.push({ method: "submitNewAlgoOrder", params });
      return { algoId: 42, algoStatus: "NEW" };
    },
    cancelOrder: async (params: unknown) => {
      calls.push({ method: "cancelOrder", params });
      return { status: "CANCELED" };
    },
    cancelAlgoOrder: async (params: unknown) => {
      calls.push({ method: "cancelAlgoOrder", params });
      return { code: "200" };
    },
    ...overrides,
  } as unknown as FuturesTransport;
  return {
    client,
    calls,
    api: await getBinanceTradingApi(config, { futures: client }),
  };
}

test("orders map to USD-M regular and algo APIs, with namespaced IDs and maximum available leverage", async () => {
  const { api, calls } = await fixture();
  const market = await api.createMarketOrder({
    side: "sell",
    size: 0.1,
    reduceOnly: true,
  });
  await api.createLimitOrder({ side: "buy", size: 0.2, price: 100 });
  const stop = await api.createStopMarketOrder({
    side: "buy",
    size: 0.3,
    price: 110,
  });
  await api.createStopLimitOrder({
    side: "sell",
    size: 0.4,
    stopPrice: 90,
    limitPrice: 89,
  });
  assert.equal(market.order.id, "order:42");
  assert.equal(stop.order.id, "algo:42");
  assert.equal(market.order.leverage, 125);
  assert.equal(calls.filter((call) => call.method === "setLeverage").length, 1);
  const regular = calls
    .filter((call) => call.method === "submitNewOrder")
    .map((call) => call.params as Record<string, unknown>);
  assert.deepEqual(
    [
      regular[0].type,
      regular[0].positionSide,
      regular[0].side,
      regular[0].reduceOnly,
    ],
    ["MARKET", "BOTH", "SELL", "true"],
  );
  assert.equal(regular[0].price, undefined);
  assert.equal(regular[1].price, 100);
  assert.equal(regular[1].timeInForce, "GTC");
  assert.notEqual(regular[0].newClientOrderId, regular[1].newClientOrderId);
  const algo = calls
    .filter((call) => call.method === "submitNewAlgoOrder")
    .map((call) => call.params as Record<string, unknown>);
  assert.deepEqual(
    [algo[0].type, algo[0].triggerPrice, algo[0].quantity, algo[0].workingType],
    ["STOP_MARKET", "110", "0.3", "CONTRACT_PRICE"],
  );
  assert.equal(algo[0].price, undefined);
  assert.deepEqual(
    [algo[1].type, algo[1].triggerPrice, algo[1].price],
    ["STOP", "90", "89"],
  );
});

test("ordinary and conditional cancellation use separate endpoints even with identical numeric IDs", async () => {
  const { api, calls } = await fixture();
  assert.equal(await api.cancelOrder("order:42"), true);
  assert.equal(await api.cancelOrder("algo:42"), true);
  assert.deepEqual(calls, [
    { method: "cancelOrder", params: { symbol: "BTCUSDT", orderId: 42 } },
    { method: "cancelAlgoOrder", params: { algoId: 42 } },
  ]);
  await assert.rejects(api.cancelOrder("42"));
  const missing = await fixture({
    cancelOrder: async () => {
      throw { code: -2011 };
    },
  });
  assert.equal(await missing.api.cancelOrder("order:42"), false);
  const outage = await fixture({
    cancelOrder: async () => {
      throw new Error("timeout");
    },
  });
  await assert.rejects(outage.api.cancelOrder("order:42"), /timeout/);
});

test("definite rejections are distinguished from uncertain submissions; neither is retried", async () => {
  let submissions = 0;
  const { api } = await fixture({
    submitNewOrder: async () => {
      submissions++;
      throw { code: -2019 };
    },
  });
  assert.equal(
    (await api.createMarketOrder({ side: "buy", size: 1 })).accepted,
    false,
  );
  assert.equal(submissions, 1);
  const uncertain = await fixture({
    submitNewOrder: async () => {
      submissions++;
      throw { code: -1007, message: "Timeout" };
    },
  });
  await assert.rejects(
    uncertain.api.createMarketOrder({ side: "buy", size: 1 }),
    (error: unknown) => {
      assert.ok(error instanceof BinanceOrderSubmissionError);
      assert.ok(error.clientOrderId.length > 0);
      assert.equal(error.conditional, false);
      return true;
    },
  );
  assert.equal(submissions, 2);
});

test("partial fills, fills and expired responses retain their actual status", async () => {
  for (const [status, expected, accepted] of [
    ["PARTIALLY_FILLED", "partially-filled", true],
    ["FILLED", "filled", true],
    ["EXPIRED", "expired", false],
  ] as const) {
    const { api } = await fixture({
      submitNewOrder: async () =>
        ({ orderId: 1, status }) as Awaited<
          ReturnType<FuturesTransport["submitNewOrder"]>
        >,
    });
    const result = await api.createMarketOrder({ side: "buy", size: 1 });
    assert.equal(result.order.status, expected);
    assert.equal(result.accepted, accepted);
  }
});

test("balances use futures collateral and signed exposure; short capacity is not tied to spot assets", async () => {
  const { api, calls } = await fixture();
  const equity = await api.getEquity();
  assert.equal(equity.assetUnleveraged, -2);
  assert.equal(equity.quoteUnleveraged + equity.assetUnleveraged * 100, 1100);
  assert.equal(equity.quoteReserved, 40);
  const capacity = await api.getOrderCapacity({
    side: "sell",
    price: 100,
    leverage: 20,
  });
  assert.equal(capacity.leverage, 10);
  assert.ok(capacity.quote > 9700 && capacity.quote <= 9800);
  assert.equal(
    calls.length,
    0,
    "read methods must not change leverage or account modes",
  );
  assert.ok(Math.abs((await api.getFriction()) - 0.0005) < 1e-12);
});

test("exchange filters and leverage brackets map to market rules with disabled filters handled", async () => {
  const { api } = await fixture();
  const rules = await api.getMarketRules();
  assert.deepEqual(rules.marketQuantity, { min: 0.001, max: 100, step: 0.001 });
  assert.equal(rules.maxLeverage, 125);
  assert.equal(rules.maxNotional, 10000);
  assert.equal(rules.minNotional, 5);
  assert.equal(await api.getMarketRules(), rules);
  await api.initialize();
  assert.equal(
    rules.maxNotional,
    10000,
    "previous rule snapshots remain stable",
  );
  assert.equal((await api.getMarketRules()).maxNotional, 1000);
  assert.notEqual(await api.getMarketRules(), rules);
});

test("history paginates beyond exchange page limits and excludes the currently forming candle", async (t) => {
  t.mock.method(Date, "now", () => 1600 * 60_000 + 30_000);
  const rows: Kline[] = Array.from({ length: 1601 }, (_, index) => [
    index * 60_000,
    "100",
    "110",
    "90",
    String(index + 1),
    "10",
    (index + 1) * 60_000 - 1,
    "1000",
    10,
    "4",
    "400",
    "0",
  ]);
  let pages = 0;
  const { api } = await fixture({
    getKlines: async (params) => {
      pages++;
      assert.equal(params.interval, "1m");
      assert.ok(params.limit! <= 1500);
      return rows
        .filter((row) => row[0] <= params.endTime!)
        .slice(-params.limit!);
    },
  });
  const history = await api.getHistory({ intervalMs: 60_000, count: 1501 });
  assert.equal(pages, 2);
  assert.equal(history.length, 1501);
  assert.equal(history[0].openTime, 99 * 60_000);
  assert.equal(history.at(-1)?.openTime, 1599 * 60_000);
  assert.equal(history[0].aggressiveSellVolume, 6);
  assert.deepEqual(await api.getHistory({ intervalMs: 60_000, count: 0 }), []);
  await assert.rejects(
    api.getHistory({ intervalMs: 2000, count: 1 }),
    /Unsupported/,
  );
});

test("market quantity steps satisfy both applicable exchange grids", async () => {
  const altered = structuredClone(symbol);
  altered.filters.find((filter) => filter.filterType === "LOT_SIZE")!.stepSize =
    "0.002";
  altered.filters.find(
    (filter) => filter.filterType === "MARKET_LOT_SIZE",
  )!.stepSize = "0.003";
  const { api } = await fixture({
    getExchangeInfo: async () =>
      ({ symbols: [altered] }) as Awaited<
        ReturnType<FuturesTransport["getExchangeInfo"]>
      >,
  });
  assert.equal((await api.getMarketRules()).marketQuantity.step, 0.006);
});

test("unsupported futures account modes fail before an order or leverage mutation", async () => {
  const { client, calls } = await fixture();
  await assert.rejects(
    getBinanceTradingApi(config, {
      futures: {
        ...client,
        getFuturesAccountConfig: async () =>
          ({
            dualSidePosition: true,
            multiAssetsMargin: false,
            canTrade: true,
          }) as Awaited<
            ReturnType<FuturesTransport["getFuturesAccountConfig"]>
          >,
      },
    }),
    /one-way/,
  );
  assert.deepEqual(calls, []);
  await assert.rejects(
    getBinanceTradingApi(config, {
      futures: {
        ...client,
        getFuturesSymbolConfig: async () =>
          [
            {
              symbol: "BTCUSDT",
              marginType: "ISOLATED",
              leverage: 5,
            },
          ] as Awaited<ReturnType<FuturesTransport["getFuturesSymbolConfig"]>>,
      },
    }),
    /cross margin/,
  );
  assert.deepEqual(calls, []);
});

test("configuration and order validation reject invalid input without network calls", async () => {
  const { api, client, calls } = await fixture();
  assert.throws(() =>
    getBinanceTradingApi(
      { ...config, environment: "paper" as "demo" },
      { futures: client },
    ),
  );
  assert.throws(() =>
    getBinanceTradingApi({ ...config, timeoutMs: 0 }, { futures: client }),
  );
  await assert.rejects(api.createMarketOrder({ side: "buy", size: NaN }));
  await assert.rejects(
    api.createStopMarketOrder({ side: "sell", size: 1, price: 0 }),
  );
  assert.deepEqual(calls, []);
});

test("maximum leverage accounts for existing positions and open orders", async () => {
  let selected = 0;
  const { api } = await fixture({
    getAccountInformationV3: async () =>
      ({
        ...account,
        positions: [
          { ...account.positions[0], askNotional: "900", bidNotional: "0" },
        ],
      }) as Awaited<ReturnType<FuturesTransport["getAccountInformationV3"]>>,
    setLeverage: async (params) => {
      selected = params.leverage;
      return {
        leverage: params.leverage,
        maxNotionalValue: "10000",
        symbol: "BTCUSDT",
      };
    },
  });
  await api.initialize();
  assert.equal(selected, 10, "1100 exposure does not fit the 125x tier");
  assert.equal((await api.getMarketRules()).maxNotional, 10000);
});

test("one shared client reuses symbol initialization and transport across configurations", async () => {
  const { client, calls } = await fixture();
  const shared = new BinanceClient(config, { futures: client });
  const first = shared.market({ product: "futures", symbol: "BTCUSDT" });
  const second = shared.market({ product: "futures", symbol: "BTCUSDT" });
  assert.equal(first, second);
  const api = await first;
  assert.equal(api, await second);
  assert.equal(
    calls.length,
    0,
    "creation must not change leverage or submit orders",
  );
  await Promise.all([api.initialize(), (await second).initialize()]);
  assert.equal(calls.filter((c) => c.method === "setLeverage").length, 1);
  assert.equal(
    typeof (
      await shared.market({
        product: "futures",
        symbol: "BTCUSDT",
        frictionDecayRate: 0,
      })
    ).getTicks,
    "function",
  );
});

test("friction averages spread and depth over sizes without a configured quantity", async () => {
  const { client } = await fixture({
    getOrderBook: async () =>
      ({
        bids: [
          ["99", "1"],
          ["98", "1"],
        ],
        asks: [
          ["101", "1"],
          ["103", "1"],
        ],
      }) as Awaited<ReturnType<FuturesTransport["getOrderBook"]>>,
  });
  const api = await getBinanceTradingApi(config, { futures: client });
  const friction = await api.getFriction();
  assert.ok(friction > 0.0105 && friction < 0.0205);
  assert.equal(await api.getFriction(), friction);
  const shared = new BinanceClient(config, { futures: client });
  const market = { product: "futures" as const, symbol: config.symbol };
  const defaultApi = await shared.market(market);
  assert.equal(
    defaultApi,
    await shared.market({ ...market, frictionDecayRate: 5 }),
  );
  const uniform = await shared.market({ ...market, frictionDecayRate: 0 });
  const concentrated = await shared.market({
    ...market,
    frictionDecayRate: 20,
  });
  assert.notEqual(defaultApi, uniform);
  assert.ok((await uniform.getFriction()) > friction);
  assert.ok((await concentrated.getFriction()) < friction);
  assert.equal(await uniform.getFriction(2), await concentrated.getFriction(2));
  for (const frictionDecayRate of [-1, NaN, Infinity]) {
    assert.throws(
      () => shared.market({ ...market, frictionDecayRate }),
      /frictionDecayRate/,
    );
  }
  assert.ok(Math.abs((await api.getFriction(1)) - 0.0105) < 1e-12);
  assert.ok(Math.abs((await api.getFriction(2)) - 0.0205) < 1e-12);
  await assert.rejects(api.getFriction(3), /Insufficient/);
  await assert.rejects(api.getFriction(0), /positive/);
});

test("one-second futures candles exclude the current second and leave empty seconds absent", async (t) => {
  t.mock.method(Date, "now", () => 4500);
  const requests: unknown[] = [];
  const { api } = await fixture({
    getAggregateTrades: async (params) => {
      requests.push(params);
      return [
        { a: 1, p: "100", q: "2", T: 1100, m: false },
        { a: 2, p: "105", q: "1", T: 1600, m: true },
        { a: 3, p: "103", q: "3", T: 3100, m: false },
        { a: 4, p: "999", q: "1", T: 4100, m: false },
      ] as Awaited<ReturnType<FuturesTransport["getAggregateTrades"]>>;
    },
  });
  const history = await api.getHistory({ intervalMs: 1000, count: 3 });
  assert.deepEqual(
    history.map((c) => c.openTime),
    [1000, 3000],
  );
  assert.deepEqual(history[0], {
    openTime: 1000,
    closeTime: 1999,
    open: 100,
    high: 105,
    low: 100,
    close: 105,
    volume: 3,
    aggressiveBuyVolume: 2,
    aggressiveSellVolume: 1,
  });
  assert.deepEqual(requests[0], {
    symbol: "BTCUSDT",
    limit: 1000,
    startTime: 1000,
    endTime: 3999,
  });
  await assert.rejects(
    api.getHistory({ intervalMs: 1000, count: 3601 }),
    /one hour/,
  );
});

test("tick history paginates by aggregate ID when more than 1000 trades share a timestamp", async () => {
  const calls: unknown[] = [];
  const { api } = await fixture({
    getAggregateTrades: async (params) => {
      calls.push(params);
      const start = params.fromId ?? 1;
      return Array.from({ length: start === 1 ? 1000 : 2 }, (_, i) => ({
        a: start + i,
        p: "1",
        q: "2",
        T: start + i === 1002 ? 2001 : 1500,
        m: false,
      })) as Awaited<ReturnType<FuturesTransport["getAggregateTrades"]>>;
    },
  });
  const ticks = await api.getTicks({ startTime: 1000, endTime: 2000 });
  assert.equal(ticks.length, 1001);
  assert.deepEqual(calls[1], { symbol: "BTCUSDT", limit: 1000, fromId: 1001 });
  assert.deepEqual(ticks[0], {
    timestamp: 1500,
    price: 1,
    quantity: 2,
    candle: null,
  });
});
