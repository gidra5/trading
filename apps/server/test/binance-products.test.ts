import assert from "node:assert/strict";
import test from "node:test";
import { MainClient, USDMClient } from "binance";
import {
  BinanceClient,
  getBinanceTradingApi,
  BinanceOrderSubmissionError,
  type SpotTransport,
  type SignedTransport,
  type BinancePredictionConfig,
} from "../src/clients/binance.js";
import { predictionAmountToWei } from "../src/clients/binance-predictions.js";

const connection = {
  environment: "live" as const,
  apiKey: "key",
  apiSecret: "secret",
};
const book = { bids: [["99", "100"]], asks: [["101", "100"]] };
const filters = [
  {
    filterType: "PRICE_FILTER",
    minPrice: "0.01",
    maxPrice: "1000000",
    tickSize: "0.01",
  },
  { filterType: "LOT_SIZE", minQty: "0.01", maxQty: "100", stepSize: "0.01" },
  { filterType: "NOTIONAL", minNotional: "5", maxNotional: "10000" },
];

async function spotFixture() {
  const calls: Record<string, unknown>[] = [];
  const transport = {
    getExchangeInfo: async () => ({
      symbols: [
        {
          symbol: "BTCUSDT",
          status: "TRADING",
          isSpotTradingAllowed: true,
          baseAsset: "BTC",
          quoteAsset: "USDT",
          filters,
          orderTypes: ["MARKET", "LIMIT", "STOP_LOSS", "STOP_LOSS_LIMIT"],
        },
      ],
    }),
    getAccountInformation: async () => ({
      canTrade: true,
      balances: [
        { asset: "BTC", free: "2", locked: "1" },
        { asset: "USDT", free: "1000", locked: "200" },
      ],
    }),
    getCommissionRates: async () => ({
      standardCommission: { taker: "0.001", buyer: "0", seller: "0.0002" },
      taxCommission: { taker: "0.0001", buyer: "0", seller: "0" },
      specialCommission: { taker: "0.0003", buyer: "0", seller: "0" },
    }),
    getOrderBook: async () => book,
    submitNewOrder: async (params: Record<string, unknown>) => {
      calls.push(params);
      return { status: "FILLED" };
    },
    cancelOrder: async (params: Record<string, unknown>) => {
      calls.push(params);
      return { status: "CANCELED" };
    },
    getKlines: async (params: Record<string, unknown>) => {
      calls.push(params);
      return [];
    },
  } as unknown as SpotTransport;
  return {
    calls,
    transport,
    api: await getBinanceTradingApi(
      { ...connection, product: "spot", symbol: "BTCUSDT" },
      { spot: transport },
    ),
  };
}

test("Spot uses native order types, base quantities and client IDs for cancellation", async () => {
  const { api, calls } = await spotFixture();
  const result = await api.createMarketOrder({ side: "buy", size: 1 });
  await api.createLimitOrder({ side: "sell", size: 1, price: 100 });
  await api.createStopMarketOrder({ side: "sell", size: 1, price: 90 });
  await api.createStopLimitOrder({
    side: "buy",
    size: 1,
    stopPrice: 110,
    limitPrice: 111,
  });
  assert.deepEqual(
    calls.map((c) => c.type),
    ["MARKET", "LIMIT", "STOP_LOSS", "STOP_LOSS_LIMIT"],
  );
  assert.equal(result.order.status, "filled");
  assert.equal(calls[0].quantity, 1);
  assert.equal(calls[2].stopPrice, 90);
  assert.equal(calls[3].price, 111);
  assert.equal(calls[0].positionSide, undefined);
  assert.equal(calls[0].reduceOnly, undefined);
  assert.equal(await api.cancelOrder(result.order.id), true);
  assert.equal(calls.at(-1)?.origClientOrderId, calls[0].newClientOrderId);
  await assert.rejects(
    api.createMarketOrder({ side: "sell", size: 1, reduceOnly: true }),
    /reduceOnly/,
  );
});

test("Spot equity uses free/locked balances, cash capacity cannot short, and friction includes fees", async () => {
  const { api } = await spotFixture();
  assert.deepEqual(await api.getEquity(), {
    quoteAvailable: 1000,
    quoteReserved: 200,
    quoteUnleveraged: 1200,
    assetAvailable: 2,
    assetReserved: 1,
    assetUnleveraged: 3,
  });
  assert.deepEqual(
    await api.getOrderCapacity({ side: "sell", price: 100, leverage: 125 }),
    { quote: 200, leverage: 1 },
  );
  assert.ok(Math.abs((await api.getFriction()) - 0.0116) < 1e-12);
  assert.ok(Math.abs((await api.getFriction(1)) - 0.0116) < 1e-12);
  await assert.rejects(api.getFriction(0.015), /quantity limits and step/);
  assert.equal((await api.getMarketRules()).minNotional, 5);
  assert.equal((await api.getMarketRules()).maxNotional, 10000);
});

test("market creation loads metadata once and returns a ready Spot adapter", async () => {
  const { transport, calls } = await spotFixture();
  const info = await transport.getExchangeInfo();
  let resolveInfo!: (value: typeof info) => void;
  const pending = new Promise<typeof info>((resolve) => {
    resolveInfo = resolve;
  });
  let requests = 0;
  const client = new BinanceClient(connection, {
    spot: {
      ...transport,
      getExchangeInfo: () => {
        requests++;
        return pending;
      },
    },
  });
  const market = { product: "spot" as const, symbol: "BTCUSDT" };
  const first = client.market(market);
  const second = client.market(market);
  assert.equal(first, second);
  assert.equal(requests, 1);
  resolveInfo(info);
  const api = await first;
  assert.equal(api, await second);
  assert.equal(api, await client.market(market));
  await api.getMarketRules();
  await api.getEquity();
  await api.getFriction();
  assert.equal(requests, 1);
  assert.equal(calls.length, 0);
});

test("failed metadata and inactive symbols reject creation and allow retry", async () => {
  const { transport } = await spotFixture();
  const info = await transport.getExchangeInfo();
  let requests = 0;
  const client = new BinanceClient(connection, {
    spot: {
      ...transport,
      getExchangeInfo: async () => {
        requests++;
        if (requests === 1) throw new Error("metadata unavailable");
        if (requests === 2) return { ...info, symbols: [] };
        return info;
      },
    },
  });
  const market = { product: "spot" as const, symbol: "BTCUSDT" };
  await assert.rejects(client.market(market), /metadata unavailable/);
  await assert.rejects(client.market(market), /inactive Spot symbol/);
  const api = await client.market(market);
  assert.equal((await api.getMarketRules()).minNotional, 5);
  assert.equal(requests, 3);
});

test("Spot supports native one-second history", async () => {
  const { api, calls } = await spotFixture();
  await api.getHistory({ intervalMs: 1000, count: 1200 });
  assert.equal(calls[0].interval, "1s");
  assert.equal(calls[0].limit, 1000);
});

const optionSymbol = {
  symbol: "BTC-991231-50000-C",
  underlying: "BTCUSDT",
  quoteAsset: "USDT",
  status: "TRADING",
  expiryDate: 4102358400000,
  unit: 1,
  filters,
  minQty: "0.01",
  maxQty: "100",
  quantityScale: 2,
};
async function optionsFixture() {
  const calls: { endpoint: string; params: Record<string, unknown> }[] = [];
  const transport: SignedTransport = {
    get: async (endpoint) => {
      if (endpoint.endsWith("exchangeInfo"))
        return { optionSymbols: [optionSymbol] };
      if (endpoint.endsWith("index")) return { indexPrice: "50000" };
      if (endpoint.endsWith("depth")) return book;
      return [];
    },
    getPrivate: async (endpoint) => {
      if (endpoint.endsWith("marginAccount"))
        return {
          canTrade: true,
          asset: [
            {
              asset: "USDT",
              equity: "1300",
              available: "900",
              initialMargin: "100",
            },
          ],
        };
      if (endpoint.endsWith("position"))
        return [
          {
            symbol: optionSymbol.symbol,
            quantity: "3",
            markPrice: "100",
            askQuantity: "1",
          },
        ];
      return { commissions: [{ underlying: "BTCUSDT", takerFee: "0.0002" }] };
    },
    postPrivate: async (endpoint, params) => {
      calls.push({ endpoint, params });
      return { orderId: 4611875134427365000, status: "ACCEPTED" };
    },
    deletePrivate: async (endpoint, params) => {
      calls.push({ endpoint, params });
      return { status: "CANCELED" };
    },
  };
  return {
    calls,
    transport,
    api: await getBinanceTradingApi(
      { ...connection, product: "options", symbol: optionSymbol.symbol },
      { options: transport },
    ),
  };
}

test("options submit native limits and cancel by client ID without rounding int64 IDs", async () => {
  const { api, calls } = await optionsFixture();
  const order = await api.createLimitOrder({
    side: "sell",
    size: 1,
    price: 100,
    reduceOnly: true,
  });
  assert.equal(order.order.status, "open");
  assert.equal(calls[0].endpoint, "eapi/v1/order");
  assert.equal(calls[0].params.type, "LIMIT");
  assert.equal(calls[0].params.reduceOnly, true);
  assert.equal(await api.cancelOrder(order.order.id), true);
  assert.equal(calls[1].params.clientOrderId, calls[0].params.clientOrderId);
  assert.equal(calls[1].params.orderId, undefined);
  await assert.rejects(
    api.createMarketOrder({ side: "buy", size: 1 }),
    /limit orders/,
  );
  await assert.rejects(
    api.createStopLimitOrder({
      side: "buy",
      size: 1,
      stopPrice: 100,
      limitPrice: 101,
    }),
    /limit orders/,
  );
  await assert.rejects(
    api.createLimitOrder({ side: "sell", size: 1, price: 100 }),
    /uncovered/,
  );
  assert.equal(calls.length, 2);
});

test("options use premium equity, reserved sell quantity and underlying-based capped fees", async () => {
  const { api } = await optionsFixture();
  assert.deepEqual(await api.getEquity(), {
    quoteAvailable: 900,
    quoteReserved: 100,
    quoteUnleveraged: 1000,
    assetAvailable: 2,
    assetReserved: 1,
    assetUnleveraged: 3,
  });
  assert.deepEqual(
    await api.getOrderCapacity({ side: "sell", price: 100, leverage: 20 }),
    { quote: 200, leverage: 1 },
  );
  assert.ok(Math.abs((await api.getFriction()) - 0.11) < 1e-12);
  assert.ok(Math.abs((await api.getFriction(1)) - 0.11) < 1e-12);
  await assert.rejects(api.getFriction(0.015), /quantity limits and step/);
});

const prediction: BinancePredictionConfig = {
  product: "predictions",
  walletAddress: "0x123",
  walletId: "wallet",
  tokenId: "112233",
  marketId: 123,
  vendor: "predict_fun",
  accountType: "SPOT",
  fundingSource: "CEX",
};
async function predictionFixture(overrides: Partial<SignedTransport> = {}) {
  const calls: { endpoint: string; params: Record<string, unknown> }[] = [];
  const transport: SignedTransport = {
    get: async () => {
      throw new Error("Prediction market data must be signed");
    },
    getPrivate: async (endpoint) => {
      if (endpoint.endsWith("wallet/list"))
        return { wallets: [{ walletAddress: "0x123", walletId: "wallet" }] };
      if (endpoint.endsWith("order-book"))
        return {
          tokenId: "112233",
          timestamp: 1000,
          bids: [{ price: "0.4", size: "20" }],
          asks: [{ price: "0.5", size: "30" }],
        };
      if (endpoint.endsWith("position/token"))
        return {
          position: {
            tokenId: "112233",
            shares: "2.5",
            avgPrice: "0.4",
            value: "1.25",
            currentPrice: "0.5",
            positionStatus: "OPEN",
          },
        };
      return {
        items: [
          {
            accountType: "SPOT",
            availableBalanceDisplay: "100",
            enabled: true,
          },
        ],
      };
    },
    postPrivate: async (endpoint, params) => {
      calls.push({ endpoint, params });
      if (endpoint.endsWith("get-quote"))
        return {
          ...params,
          quoteId: "quote-1",
          amountOut: "4000000000000000000",
          feeAmount: "10000000000000000",
          minReceive: "3900000000000000000",
          averagePrice: 0.5,
          priceImpact: 0.001,
          expireAt: Date.now() + 60000,
          priceLimit: params.priceLimit ?? null,
        };
      if (endpoint.endsWith("batch-cancel"))
        return { canceled: ["9007199254740993123"], failed: [] };
      return { orderId: "9007199254740993123" };
    },
    deletePrivate: async () => {
      throw new Error("Wrong cancellation method");
    },
    ...overrides,
  };
  return {
    calls,
    api: await new BinanceClient(connection, { predictions: transport }).market(
      prediction,
    ),
  };
}

test("prediction market buys use exact collateral wei, quote-bound submission and FOK", async () => {
  const { api, calls } = await predictionFixture();
  await api.initialize();
  const quote = await api.getQuote({
    side: "buy",
    type: "market",
    amount: "2.000000000000000001",
  });
  assert.equal(calls[0].params.amountIn, "2000000000000000001");
  assert.equal(quote.amountIn, "2000000000000000001");
  const order = await api.placeQuote(quote.quoteId);
  assert.deepEqual(order, {
    id: "prediction:9007199254740993123",
    status: "pending",
  });
  assert.equal(calls[1].params.timeInForce, "FOK");
  assert.equal(calls[1].params.fundingSource, "CEX");
  assert.equal(calls[1].params.fundTransferAmount, undefined);
  await assert.rejects(api.placeQuote(quote.quoteId), /already submitted/);
  assert.equal(await api.cancelOrder(order.id), true);
  assert.equal(
    calls[2].params["cancelInfoList[0].orderId"],
    "9007199254740993123",
  );
});

test("prediction limit sells spend shares and preserve price through quote and GTC placement", async () => {
  const { api, calls } = await predictionFixture();
  const quote = await api.getQuote({
    side: "sell",
    type: "limit",
    amount: "3.5",
    price: "0.6",
  });
  await api.placeQuote(quote.quoteId);
  assert.equal(calls[0].params.amountIn, "3500000000000000000");
  assert.equal(calls[0].params.priceLimit, "0.6");
  assert.equal(calls[1].params.priceLimit, "0.6");
  assert.equal(calls[1].params.timeInForce, "GTC");
  assert.equal((await api.getOrderBook()).tokenId, "112233");
  assert.equal((await api.getPosition())?.shares, "2.5");
  assert.equal(
    (await api.getPaymentBalances())[0].availableBalanceDisplay,
    "100",
  );
});

test("prediction quote expiry and ambiguous submissions prevent reuse", async (t) => {
  t.mock.method(Date, "now", () => 1000);
  const { api } = await predictionFixture();
  const quote = await api.getQuote({
    side: "buy",
    type: "market",
    amount: "2",
  });
  t.mock.method(Date, "now", () => 62000);
  await assert.rejects(api.placeQuote(quote.quoteId), /expired/);
  let submissions = 0;
  const uncertain = await predictionFixture({
    postPrivate: async (endpoint, params) => {
      if (endpoint.endsWith("get-quote"))
        return {
          ...params,
          quoteId: "uncertain",
          amountOut: "4",
          feeAmount: "1",
          minReceive: "3",
          averagePrice: 0.5,
          priceImpact: 0,
          expireAt: 100000,
          priceLimit: null,
        };
      submissions++;
      throw new Error("timeout");
    },
  });
  await uncertain.api.getQuote({ side: "buy", type: "market", amount: "2" });
  await assert.rejects(
    uncertain.api.placeQuote("uncertain"),
    BinanceOrderSubmissionError,
  );
  await assert.rejects(
    uncertain.api.placeQuote("uncertain"),
    /already submitted/,
  );
  assert.equal(submissions, 1);
});

test("prediction quantities reject exponent notation, excess precision and zero", () => {
  for (const value of ["0", "-1", "NaN", "1e3", "0.0000000000000000001"])
    assert.throws(() => predictionAmountToWei(value));
  assert.equal(predictionAmountToWei("0.000000000000000001"), "1");
});

test("unsupported environments never fall back to live options or predictions", () => {
  for (const environment of ["demo", "testnet"] as const) {
    const client = new BinanceClient({ ...connection, environment });
    assert.throws(
      () => client.market({ product: "options", symbol: optionSymbol.symbol }),
      /environment/,
    );
    assert.throws(() => client.market(prediction), /environment/);
  }
});

test("SDK instances select separate live, demo and testnet hosts", async (t) => {
  const hosts: string[] = [];
  const { transport } = await spotFixture();
  const info = await transport.getExchangeInfo();
  const capture = async function (this: unknown) {
    hosts.push((this as { baseUrl: string }).baseUrl);
    return info;
  };
  t.mock.method(MainClient.prototype, "getExchangeInfo", capture);
  t.mock.method(
    USDMClient.prototype,
    "getExchangeInfo",
    async function (this: unknown) {
      hosts.push((this as { baseUrl: string }).baseUrl);
      return { symbols: [{ ...info.symbols[0], marginAsset: "USDT" }] };
    },
  );
  t.mock.method(USDMClient.prototype, "getFuturesAccountConfig", async () => ({
    dualSidePosition: false,
    multiAssetsMargin: false,
    canTrade: true,
  }));
  t.mock.method(USDMClient.prototype, "getFuturesSymbolConfig", async () => [
    { symbol: "BTCUSDT", marginType: "CROSSED", leverage: 10 },
  ]);
  t.mock.method(
    USDMClient.prototype,
    "getNotionalAndLeverageBrackets",
    async () => [
      {
        symbol: "BTCUSDT",
        brackets: [{ initialLeverage: 10, notionalCap: 10000 }],
      },
    ],
  );
  for (const environment of ["live", "demo", "testnet"] as const) {
    const client = new BinanceClient({ ...connection, environment });
    await client.market({ product: "spot", symbol: "BTCUSDT" });
    await client.market({ product: "futures", symbol: "BTCUSDT" });
  }
  assert.deepEqual(hosts, [
    "https://api.binance.com",
    "https://fapi.binance.com",
    "https://demo-api.binance.com",
    "https://demo-fapi.binance.com",
    "https://testnet.binance.vision",
    "https://testnet.binancefuture.com",
  ]);
});
