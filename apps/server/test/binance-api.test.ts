import assert from "node:assert/strict";
import test from "node:test";
import type { BinanceMarketListing } from "../src/binance-markets.js";
import { BinanceExchangeTrading } from "../src/binance-exchange.js";
import { BinanceTradingApi } from "../src/trading-api/binance-api.js";
import { BinanceExchangeClient } from "../src/trading-api/binance-client.js";
import type {
  BinanceCancelOrderInput,
  BinanceOrderUpdates,
  BinancePlaceOrderInput,
  BinancePositionContext,
  BinanceProviderOrder,
  BinanceTradingClient,
} from "../src/trading-api/binance-client.js";

test("Binance adapter uses the focused exchange client for account, submit, and cancel", async () => {
  const client = new FakeClient();
  const api = new BinanceTradingApi({
    market,
    client,
    getHistory: async () => [],
  });

  assert.equal((await api.getEquity()).quoteAvailable, 900);
  assert.equal((await api.getMarketRules()).maxLeverage, 20);
  assert.equal(await api.getFriction(), 0.001);

  const result = await api.createLimitOrder({ side: "buy", size: 0.1, price: 100 });
  assert.equal(result.accepted, true);
  assert.equal(client.placed?.clientOrderId, result.order.id);
  assert.equal(await api.cancelOrder(result.order.id), true);
  assert.equal(client.cancelled?.orderId, "provider-1");
});

test("Binance user-data fills preserve new bot order ids", () => {
  const exchange = new BinanceExchangeTrading({
    enabled: true,
    mode: "usdm-futures-testnet",
    apiKey: "key",
    apiSecret: "secret",
    recvWindowMs: 5_000,
    autoSubmit: true,
  });
  const update = exchange.reconciliationFromUserDataEvent(futuresMarket, {
    e: "ORDER_TRADE_UPDATE",
    T: 123,
    o: {
      s: "BTCUSDT",
      c: "bot-123",
      S: "BUY",
      o: "MARKET",
      x: "TRADE",
      X: "FILLED",
      i: 42,
      q: "0.1",
      z: "0.1",
      l: "0.1",
      L: "100",
      ap: "100",
      t: 7,
      n: "0.01",
      N: "USDT",
      ps: "BOTH",
    },
  });

  assert.equal(update?.orders?.[0]?.localOrderId, "bot-123");
  assert.equal(update?.fills?.[0]?.localOrderId, "bot-123");
});

test("Binance capacity is capped by remaining provider notional", async () => {
  const exchange = {
    fetchFriction: async () => ({ feeBps: 4, estimatedSlippageBps: 1 }),
    fetchOrderCapacityState: async () => ({
      availableBalanceQuote: 200,
      positionMode: "one-way" as const,
      positions: [{
        symbol: "BTCUSDT",
        positionSide: "BOTH",
        positionAmt: 2,
        notional: 200,
        maxNotional: 2_500,
        leverage: 20,
      }],
      openOrders: [{ ...providerOrder("existing"), originalQuantity: 1 }],
    }),
  } as unknown as BinanceExchangeTrading;
  const client = new BinanceExchangeClient(exchange);

  assert.deepEqual(
    await client.getOrderCapacity(futuresMarket, { side: "buy", price: 100, leverage: 20 }),
    { quote: 2_200, leverage: 20 },
  );
});

class FakeClient implements BinanceTradingClient {
  placed?: BinancePlaceOrderInput;
  cancelled?: BinanceCancelOrderInput;

  async getEquity() {
    return {
      quoteAvailable: 900,
      quoteReserved: 100,
      quoteUnleveraged: 1_000,
      assetAvailable: 0,
      assetReserved: 0,
      assetUnleveraged: 0,
    };
  }

  async getPositionContext(): Promise<BinancePositionContext> {
    return { positions: [], positionMode: "one-way" };
  }

  async getOrderUpdates(): Promise<BinanceOrderUpdates> {
    return { openOrders: [], recentTrades: [] };
  }

  async getMarketRules() {
    const quantity = { min: null, max: null, step: null };
    return {
      price: quantity,
      limitQuantity: quantity,
      marketQuantity: quantity,
      minNotional: 5,
      maxNotional: null,
      maxLeverage: 20,
    };
  }

  async getOrderCapacity(_market: BinanceMarketListing, input: { leverage: number }) {
    return { quote: 1_800, leverage: input.leverage };
  }

  async getFriction(): Promise<number> {
    return 0.001;
  }

  async placeOrder(
    _market: BinanceMarketListing,
    input: BinancePlaceOrderInput,
  ): Promise<BinanceProviderOrder> {
    this.placed = input;
    return providerOrder(input.clientOrderId ?? "");
  }

  async cancelOrder(
    _market: BinanceMarketListing,
    input: BinanceCancelOrderInput,
  ): Promise<void> {
    this.cancelled = input;
  }
}

function providerOrder(clientOrderId: string): BinanceProviderOrder {
  return {
    symbol: "BTCUSDT",
    orderId: "provider-1",
    clientOrderId,
    localOrderId: clientOrderId,
    side: "BUY",
    type: "LIMIT",
    status: "NEW",
    price: 100,
    originalQuantity: 0.1,
    executedQuantity: 0,
  };
}

const market: BinanceMarketListing = {
  id: "spot:BTCUSDT",
  group: "spot",
  venue: "spot",
  symbol: "BTCUSDT",
  displaySymbol: "BTC/USDT",
  baseAsset: "BTC",
  quoteAsset: "USDT",
  status: "TRADING",
  searchable: "BTC USDT",
  supportsLiveStream: true,
  supportsHistoricalCandles: true,
};

const futuresMarket: BinanceMarketListing = {
  ...market,
  id: "usdm-futures:BTCUSDT",
  group: "futures",
  venue: "usdm-futures",
};
