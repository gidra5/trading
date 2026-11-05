import assert from "node:assert/strict";
import { z } from "zod";
import type { Kline } from "binance";
import type {
  TradingHistoryRequest,
  TradingEquitySnapshot,
  TradingOrderSnapshot,
  TradingMarketRules,
} from "@trading/bot-algo/trading-api";
import type { BinanceMarketConfig, SignedTransport } from "./binance-types.js";
import { BinanceCashApi } from "./binance-cash.js";
import {
  cashMarketRules,
  estimateBookCost,
  errorCode,
  type Book,
  type ExchangeFilter,
} from "./binance-common.js";
import { getHistory } from "./binance-history.js";
import { positive } from "./common.js";

interface OptionSymbol {
  symbol: string;
  quoteAsset: string;
  underlying: string;
  unit: number;
  expiryDate: number;
  status: string;
  filters: ExchangeFilter[];
  minQty: string;
  maxQty: string;
  quantityScale: number;
}
const decimal = z
  .string()
  .refine(
    (v) => v.trim() !== "" && Number.isFinite(Number(v)),
    "Invalid exchange decimal",
  );
const accountSchema = z.object({
  canTrade: z.boolean(),
  asset: z.array(
    z.object({
      asset: z.string(),
      equity: decimal,
      available: decimal,
      initialMargin: decimal,
    }),
  ),
});
const positionsSchema = z.array(
  z.object({
    symbol: z.string(),
    quantity: decimal,
    markPrice: decimal,
    askQuantity: decimal,
  }),
);

export class BinanceOptionsApi extends BinanceCashApi {
  protected readonly product = "options";
  private constructor(
    private readonly config: BinanceMarketConfig,
    private readonly client: SignedTransport,
    private readonly context: {
      symbol: OptionSymbol;
      rules: TradingMarketRules;
    },
  ) {
    super();
  }

  static async create(
    config: BinanceMarketConfig,
    client: SignedTransport,
  ): Promise<BinanceOptionsApi> {
    const marketConfig = config;
    const info: { optionSymbols: OptionSymbol[] } = await client.get(
      "eapi/v1/exchangeInfo",
    );
    const symbol = info.optionSymbols.find(
      (s) => s.symbol === marketConfig.symbol,
    );
    assert(
      symbol && symbol.status === "TRADING" && symbol.expiryDate > Date.now(),
      "Unsupported, expired or inactive option symbol",
    );
    // TradingApi measures quantity in underlying units and price in quote per unit.
    // Multi-unit commodity contracts need a separate normalization convention.
    assert(symbol.unit === 1, "Options adapter requires contract unit 1");
    const filters = symbol.filters.some((f) => f.filterType === "LOT_SIZE")
      ? symbol.filters
      : [
          ...symbol.filters,
          {
            filterType: "LOT_SIZE",
            minQty: symbol.minQty,
            maxQty: symbol.maxQty,
            stepSize: String(10 ** -symbol.quantityScale),
          },
        ];
    return new BinanceOptionsApi(marketConfig, client, {
      symbol,
      rules: cashMarketRules(filters),
    });
  }
  async initialize() {
    const { symbol } = this.context;
    assert(symbol.expiryDate > Date.now(), "Option has expired");
    const account = accountSchema.parse(
      await this.client.getPrivate("eapi/v1/marginAccount"),
    );
    assert(account.canTrade, "Options trading is disabled for this account");
  }
  async getMarketRules() {
    return this.context.rules;
  }
  getHistory(input: TradingHistoryRequest) {
    return getHistory(
      {
        getKlines: (params) =>
          this.client.get("eapi/v1/klines", params) as Promise<Kline[]>,
      },
      this.config.symbol,
      input,
      "options",
    );
  }
  async getEquity(): Promise<TradingEquitySnapshot> {
    const { symbol } = this.context;
    const [account, positions] = await Promise.all([
      this.client
        .getPrivate("eapi/v1/marginAccount")
        .then((v) => accountSchema.parse(v)),
      this.client
        .getPrivate("eapi/v1/position", { symbol: this.config.symbol })
        .then((v) => positionsSchema.parse(v)),
    ]);
    const asset = account.asset.find((a) => a.asset === symbol.quoteAsset);
    assert(asset, `No ${symbol.quoteAsset} options balance`);
    const position = positions.find((p) => p.symbol === symbol.symbol);
    const quantity = Number(position?.quantity ?? 0);
    assert(
      quantity >= 0,
      "Uncovered short options are not supported by cash capacity",
    );
    const reserved = Math.min(
      quantity,
      Math.max(0, Number(position?.askQuantity ?? 0)),
    );
    return {
      quoteAvailable: Number(asset.available),
      quoteReserved: Number(asset.initialMargin),
      quoteUnleveraged:
        Number(asset.equity) - quantity * Number(position?.markPrice ?? 0),
      assetAvailable: quantity - reserved,
      assetReserved: reserved,
      assetUnleveraged: quantity,
    };
  }
  async getFriction(size?: number) {
    if (size !== undefined) positive(size, "size");
    const { symbol, rules } = this.context;
    const [fees, index, book]: [
      { commissions: { underlying: string; takerFee: string }[] },
      { indexPrice: string },
      Book,
    ] = await Promise.all([
      this.client.getPrivate("eapi/v1/commission"),
      this.client.get("eapi/v1/index", { underlying: symbol.underlying }),
      this.client.get("eapi/v1/depth", { symbol: symbol.symbol, limit: 1000 }),
    ]);
    const fee = fees.commissions.find(
      (f) => f.underlying === symbol.underlying,
    );
    assert(fee, "Options commission not available");
    const slippage = estimateBookCost(
      book,
      rules.limitQuantity,
      size,
      this.config.frictionDecayRate,
    );
    const mid = (Number(book.bids[0][0]) + Number(book.asks[0][0])) / 2;
    // Fee is based on the underlying index, capped at 10% of option premium.
    const feeRate = Math.min(
      (Number(index.indexPrice) * Number(fee.takerFee)) / mid,
      0.1 * (1 + slippage),
    );
    return feeRate + slippage;
  }
  protected async send(
    order: TradingOrderSnapshot,
    clientOrderId: string,
  ): Promise<{ status: string }> {
    return z.object({ status: z.string() }).parse(
      await this.client.postPrivate("eapi/v1/order", {
        symbol: this.config.symbol,
        side: order.side === "buy" ? "BUY" : "SELL",
        type: "LIMIT",
        quantity: String(order.size),
        price: String(order.price),
        timeInForce: "GTC",
        reduceOnly: order.reduceOnly ?? false,
        clientOrderId,
        newOrderRespType: "RESULT",
      }),
    );
  }
  async cancelOrder(id: string) {
    const clientOrderId = this.clientId(id);
    try {
      const result = await this.client.deletePrivate("eapi/v1/order", {
        symbol: this.config.symbol,
        clientOrderId,
      });
      return result.status === "CANCELED";
    } catch (error) {
      if ([-2011, -2013].includes(errorCode(error) ?? 0)) return false;
      throw error;
    }
  }
}
