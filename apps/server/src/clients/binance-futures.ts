import assert from "node:assert/strict";
import { BaseTradingApi } from "./base-trading-api.js";
import { randomUUID } from "node:crypto";
import type { FuturesSymbolExchangeInfo } from "binance";
import type {
  TradingCandle,
  TradingEquitySnapshot,
  TradingHistoryRequest,
  TradingMarketRules,
  TradingOrderCapacity,
  TradingOrderCapacityRequest,
  TradingOrderResult,
  TradingOrderSnapshot,
} from "@trading/bot-algo/trading-api";
import {
  combineSteps,
  floorToStep,
  positive,
  validateOrder,
} from "./common.js";

import type { BinanceMarketConfig, FuturesTransport } from "./binance-types.js";
import {
  BinanceOrderSubmissionError,
  errorCode,
  orderStatus,
  estimateBookCost,
} from "./binance-common.js";
import { getHistory, getTicks } from "./binance-history.js";
interface Context {
  symbol: FuturesSymbolExchangeInfo;
  venueLeverage: number;
  rules: TradingMarketRules;
  tiers: { initialLeverage: number; notionalCap: number }[];
}

export class BinanceFuturesApi extends BaseTradingApi {
  private readonly config: BinanceMarketConfig;
  private readonly client: FuturesTransport;
  private initialization?: Promise<void>;

  private constructor(
    config: BinanceMarketConfig,
    client: FuturesTransport,
    private readonly context: Context,
  ) {
    super();
    this.config = config;
    this.client = client;
  }

  /** Selects the highest tier allowed by existing position and pending order notional. */
  async initialize(): Promise<void> {
    this.initialization ??= (async () => {
      const context = this.context;

      const account = await this.client.getAccountInformationV3();
      const position = account.positions.find(
        (p) => p.symbol === this.config.symbol,
      );
      const notional = Number(position?.notional ?? 0);
      const exposure = Math.max(
        Math.abs(notional),
        Math.abs(notional + Number(position?.bidNotional ?? 0)),
        Math.abs(notional - Number(position?.askNotional ?? 0)),
      );
      const eligible = context.tiers.filter(
        (tier) => tier.notionalCap > exposure,
      );
      assert(
        eligible.length > 0,
        "Position exceeds available leverage brackets",
      );
      const leverage = Math.max(
        ...eligible.map((tier) => tier.initialLeverage),
      );
      const result = await this.client.setLeverage({
        symbol: this.config.symbol,
        leverage,
      });
      context.venueLeverage = result.leverage;
      context.rules = {
        ...context.rules,
        maxNotional: Math.min(
          Number(result.maxNotionalValue) || Infinity,
          Math.max(
            ...context.tiers
              .filter((tier) => tier.initialLeverage >= result.leverage)
              .map((tier) => tier.notionalCap),
          ),
        ),
      };
    })().catch((error) => {
      this.initialization = undefined;
      throw error;
    });
    return this.initialization;
  }

  async cancelOrder(orderId: string): Promise<boolean> {
    const match = /^(order|algo):(\d+)$/.exec(orderId);
    assert(
      match && Number.isSafeInteger(Number(match[2])),
      "Invalid Binance order ID",
    );
    try {
      if (match[1] === "algo") {
        const result = await this.client.cancelAlgoOrder({
          algoId: Number(match[2]),
        });
        return Number(result.code) === 200;
      }
      const result = await this.client.cancelOrder({
        symbol: this.config.symbol,
        orderId: Number(match[2]),
      });
      return result.status === "CANCELED";
    } catch (error) {
      if (errorCode(error) === -2011 || errorCode(error) === -2013)
        return false;
      throw error;
    }
  }

  getHistory(input: TradingHistoryRequest): Promise<readonly TradingCandle[]> {
    return getHistory(this.client, this.config.symbol, input, "futures");
  }

  getTicks(input: { startTime: number; endTime: number }) {
    return getTicks(this.client, this.config.symbol, input);
  }
  async getMarketRules(): Promise<TradingMarketRules> {
    return this.context.rules;
  }

  async getEquity(): Promise<TradingEquitySnapshot> {
    const { symbol } = this.context;
    const account = await this.client.getAccountInformationV3();
    assert(!account.multiAssetsMargin, "Multi-assets margin is not supported");
    const asset = account.assets.find(
      (item) => item.asset === symbol.marginAsset,
    );
    assert(asset, `No ${symbol.marginAsset} futures balance`);
    const positions = account.positions.filter(
      (item) => item.symbol === symbol.symbol,
    );
    assert(
      positions.every((item) => item.positionSide === "BOTH"),
      "Hedge mode is not supported",
    );
    const position = positions[0];
    const size = Number(position?.positionAmt ?? 0);
    return {
      quoteAvailable: Number(asset.availableBalance),
      quoteReserved: Number(asset.initialMargin),
      // Synthetic cash + signed base exposure preserves equity = cash + base * mark.
      quoteUnleveraged:
        Number(asset.marginBalance) - Number(position?.notional ?? 0),
      assetAvailable: size,
      assetReserved: 0,
      assetUnleveraged: size,
    };
  }

  async getFriction(size?: number): Promise<number> {
    if (size !== undefined) positive(size, "size");
    const context = this.context;
    const [rates, book] = await Promise.all([
      this.client.getAccountCommissionRate({ symbol: this.config.symbol }),
      this.client.getOrderBook({ symbol: this.config.symbol, limit: 1000 }),
    ]);
    return (
      Number(rates.takerCommissionRate) +
      estimateBookCost(
        book,
        context.rules.marketQuantity,
        size,
        this.config.frictionDecayRate,
      )
    );
  }

  async getOrderCapacity(
    input: TradingOrderCapacityRequest,
  ): Promise<TradingOrderCapacity> {
    positive(input.price, "price");
    positive(input.leverage, "leverage");
    assert(input.side === "buy" || input.side === "sell", "Invalid order side");
    const context = this.context;
    const [equity, fee] = await Promise.all([
      this.getEquity(),
      this.getFriction(),
    ]);
    const leverage = Math.max(
      1,
      Math.min(input.leverage, context.venueLeverage),
    );
    const closing =
      Math.max(0, (input.side === "buy" ? -1 : 1) * equity.assetUnleveraged) *
      input.price;
    let quote =
      closing +
      Math.max(0, equity.quoteAvailable - closing * fee) / (1 / leverage + fee);
    const rules = context.rules;
    quote = Math.min(
      quote,
      Math.max(
        0,
        (rules.maxNotional ?? Infinity) -
          (input.side === "buy" ? 1 : -1) *
            equity.assetUnleveraged *
            input.price,
      ),
    );
    const size = floorToStep(
      Math.min(quote / input.price, rules.marketQuantity.max ?? Infinity),
      rules.marketQuantity.step,
    );
    quote = size * input.price;
    if (
      size < (rules.marketQuantity.min ?? 0) ||
      (quote < (rules.minNotional ?? 0) && closing === 0)
    )
      quote = 0;
    return { quote, leverage };
  }

  static async create(
    input: BinanceMarketConfig,
    client: FuturesTransport,
  ): Promise<BinanceFuturesApi> {
    const marketConfig = input;
    const [info, account, configs, brackets] = await Promise.all([
      client.getExchangeInfo(),
      client.getFuturesAccountConfig(),
      client.getFuturesSymbolConfig({ symbol: marketConfig.symbol }),
      client.getNotionalAndLeverageBrackets({
        symbol: marketConfig.symbol,
      }),
    ]);
    assert(
      !account.dualSidePosition && !account.multiAssetsMargin,
      "Requires one-way, single-asset futures mode",
    );
    assert(account.canTrade, "Futures trading is disabled for this account");
    const symbol = info.symbols.find(
      (item) => item.symbol === marketConfig.symbol,
    );
    const config = configs.find((item) => item.symbol === marketConfig.symbol);
    assert(
      symbol &&
        symbol.status === "TRADING" &&
        symbol.marginAsset === symbol.quoteAsset,
      "Unsupported or inactive USD-M symbol",
    );
    assert(
      config && config.marginType.toUpperCase() === "CROSSED",
      "Requires cross margin for the configured symbol",
    );
    const tiers =
      brackets.find((item) => item.symbol === symbol.symbol)?.brackets ?? [];
    const eligible = tiers.filter(
      (tier) => tier.initialLeverage >= config.leverage,
    );
    assert(
      eligible.length > 0,
      "Requested leverage exceeds this symbol's limits",
    );
    const price = symbol.filters.find(
      (filter) => filter.filterType === "PRICE_FILTER",
    );
    const lot = symbol.filters.find(
      (filter) => filter.filterType === "LOT_SIZE",
    );
    const market = symbol.filters.find(
      (filter) => filter.filterType === "MARKET_LOT_SIZE",
    );
    const notional = symbol.filters.find(
      (filter) => filter.filterType === "MIN_NOTIONAL",
    );
    const enabled = (value: string | number | undefined): number | null =>
      Number(value) > 0 ? Number(value) : null;
    const limitQuantity = {
      min: enabled(lot?.minQty),
      max: enabled(lot?.maxQty),
      step: enabled(lot?.stepSize),
    };
    const context: Context = {
      symbol,
      venueLeverage: config.leverage,
      tiers,
      rules: {
        price: {
          min: enabled(price?.minPrice),
          max: enabled(price?.maxPrice),
          step: enabled(price?.tickSize),
        },
        limitQuantity,
        marketQuantity: {
          min:
            Math.max(enabled(market?.minQty) ?? 0, limitQuantity.min ?? 0) ||
            null,
          max:
            Math.min(
              enabled(market?.maxQty) ?? Infinity,
              limitQuantity.max ?? Infinity,
            ) === Infinity
              ? null
              : Math.min(
                  enabled(market?.maxQty) ?? Infinity,
                  limitQuantity.max ?? Infinity,
                ),
          step: combineSteps(enabled(market?.stepSize), limitQuantity.step),
        },
        minNotional: enabled(notional?.notional),
        maxNotional: Math.max(...eligible.map((tier) => tier.notionalCap)),
        maxLeverage: Math.max(...tiers.map((tier) => tier.initialLeverage)),
      },
    };
    return new BinanceFuturesApi(marketConfig, client, context);
  }

  protected async submit(
    order: TradingOrderSnapshot,
  ): Promise<TradingOrderResult> {
    validateOrder(order);
    await this.initialize();
    const clientOrderId = randomUUID();
    const conditional = order.stopPrice !== null;
    try {
      const completed: TradingOrderSnapshot = await (async () => {
        if (conditional) {
          const result = await this.client.submitNewAlgoOrder({
            symbol: this.config.symbol,
            algoType: "CONDITIONAL",
            positionSide: "BOTH",
            side: order.side === "buy" ? "BUY" : "SELL",
            type: order.type === "stop-market" ? "STOP_MARKET" : "STOP",
            quantity: String(order.size),
            triggerPrice: String(order.stopPrice),
            ...(order.price === null
              ? {}
              : { price: String(order.price), timeInForce: "GTC" as const }),
            reduceOnly: order.reduceOnly ? "true" : "false",
            workingType: "CONTRACT_PRICE",
            clientAlgoId: clientOrderId,
          });
          return {
            ...order,
            leverage: this.context.venueLeverage,
            id: `algo:${result.algoId}`,
            status:
              result.algoStatus === "NEW"
                ? "open"
                : result.algoStatus === "REJECTED"
                  ? "rejected"
                  : result.algoStatus === "CANCELED"
                    ? "canceled"
                    : result.algoStatus === "EXPIRED"
                      ? "expired"
                      : "pending",
          };
        } else {
          const result = await this.client.submitNewOrder({
            symbol: this.config.symbol,
            positionSide: "BOTH",
            side: order.side === "buy" ? "BUY" : "SELL",
            type: order.type === "market" ? "MARKET" : "LIMIT",
            quantity: order.size,
            ...(order.price === null
              ? {}
              : { price: order.price, timeInForce: "GTC" as const }),
            reduceOnly: order.reduceOnly ? "true" : "false",
            newOrderRespType: "RESULT",
            newClientOrderId: clientOrderId,
          });
          return {
            ...order,
            leverage: this.context.venueLeverage,
            id: `order:${result.orderId}`,
            status: orderStatus(result.status),
          };
        }
      })();
      return {
        accepted: !["rejected", "canceled", "expired"].includes(
          completed.status,
        ),
        order: completed,
      };
    } catch (error) {
      // Only definite matching-engine rejections become rejected orders. Transport failures remain uncertain.
      if ([-2010, -2019, -2021, -2022, -1013].includes(errorCode(error) ?? 0)) {
        return {
          accepted: false,
          order: {
            ...order,
            leverage: this.context.venueLeverage,
            id: `rejected:${clientOrderId}`,
            status: "rejected",
          },
        };
      }
      throw new BinanceOrderSubmissionError(clientOrderId, conditional, error);
    }
  }
}
