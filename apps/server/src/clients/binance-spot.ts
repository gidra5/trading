import assert from "node:assert/strict";
import type { SymbolExchangeInfo } from "binance";
import type {
  TradingHistoryRequest,
  TradingEquitySnapshot,
  TradingOrderSnapshot,
  TradingMarketRules,
} from "@trading/bot-algo/trading-api";
import type { BinanceMarketConfig, SpotTransport } from "./binance-types.js";
import { BinanceCashApi } from "./binance-cash.js";
import {
  cashMarketRules,
  errorCode,
  estimateBookCost,
} from "./binance-common.js";
import { getHistory, getTicks } from "./binance-history.js";
import { positive } from "./common.js";

export class BinanceSpotApi extends BinanceCashApi {
  protected readonly product = "spot";
  private constructor(
    private readonly config: BinanceMarketConfig,
    private readonly client: SpotTransport,
    private readonly context: {
      symbol: SymbolExchangeInfo;
      rules: TradingMarketRules;
    },
  ) {
    super();
  }

  static async create(
    config: BinanceMarketConfig,
    client: SpotTransport,
  ): Promise<BinanceSpotApi> {
    const marketConfig = config;
    const info = await client.getExchangeInfo();
    const symbol = info.symbols.find((s) => s.symbol === marketConfig.symbol);
    assert(
      symbol && symbol.status === "TRADING" && symbol.isSpotTradingAllowed,
      "Unsupported or inactive Spot symbol",
    );
    return new BinanceSpotApi(marketConfig, client, {
      symbol,
      rules: cashMarketRules(symbol.filters),
    });
  }
  async initialize() {
    const account = await this.client.getAccountInformation();
    assert(account.canTrade, "Spot trading is disabled for this account");
  }
  async getMarketRules() {
    return this.context.rules;
  }
  getHistory(input: TradingHistoryRequest) {
    return getHistory(this.client, this.config.symbol, input, "spot");
  }
  getTicks(input: { startTime: number; endTime: number }) {
    return getTicks(this.client, this.config.symbol, input);
  }

  async getEquity(): Promise<TradingEquitySnapshot> {
    const { symbol } = this.context;
    const account = await this.client.getAccountInformation();
    const base = account.balances.find((b) => b.asset === symbol.baseAsset);
    const quote = account.balances.find((b) => b.asset === symbol.quoteAsset);
    return {
      quoteAvailable: Number(quote?.free ?? 0),
      quoteReserved: Number(quote?.locked ?? 0),
      quoteUnleveraged: Number(quote?.free ?? 0) + Number(quote?.locked ?? 0),
      assetAvailable: Number(base?.free ?? 0),
      assetReserved: Number(base?.locked ?? 0),
      assetUnleveraged: Number(base?.free ?? 0) + Number(base?.locked ?? 0),
    };
  }
  async getFriction(size?: number) {
    if (size !== undefined) positive(size, "size");
    const { rules } = this.context;
    const [rates, book] = await Promise.all([
      this.client.getCommissionRates({ symbol: this.config.symbol }),
      this.client.getOrderBook({ symbol: this.config.symbol, limit: 1000 }),
    ]);
    // Conservative: ignore conditional BNB discounts, include both buyer/seller surcharges.
    const worst = (rate: { taker: string; buyer: string; seller: string }) =>
      Number(rate.taker) + Math.max(Number(rate.buyer), Number(rate.seller));
    const special = (
      rates as typeof rates & {
        specialCommission?: typeof rates.standardCommission;
      }
    ).specialCommission;
    const fee =
      worst(rates.standardCommission) +
      worst(rates.taxCommission) +
      (special ? worst(special) : 0);
    return (
      fee +
      estimateBookCost(
        book,
        rules.marketQuantity,
        size,
        this.config.frictionDecayRate,
      )
    );
  }
  protected async send(order: TradingOrderSnapshot, clientId: string) {
    const type = (
      {
        market: "MARKET",
        limit: "LIMIT",
        "stop-market": "STOP_LOSS",
        "stop-limit": "STOP_LOSS_LIMIT",
      } as const
    )[order.type];
    const { symbol } = this.context;
    assert(
      symbol.orderTypes.includes(type),
      `Spot symbol does not allow ${type}`,
    );
    return this.client.submitNewOrder({
      symbol: this.config.symbol,
      side: order.side === "buy" ? "BUY" : "SELL",
      type,
      quantity: order.size,
      newClientOrderId: clientId,
      newOrderRespType: "FULL",
      ...(order.price === null
        ? {}
        : { price: order.price, timeInForce: "GTC" as const }),
      ...(order.stopPrice === null ? {} : { stopPrice: order.stopPrice }),
    });
  }
  async cancelOrder(id: string) {
    const origClientOrderId = this.clientId(id);
    try {
      const result = await this.client.cancelOrder({
        symbol: this.config.symbol,
        origClientOrderId,
      });
      return result.status === "CANCELED";
    } catch (error) {
      if ([-2011, -2013].includes(errorCode(error) ?? 0)) return false;
      throw error;
    }
  }
}
