import type {
  TradingMarketRules,
  TradingOrderCapacity,
  TradingOrderCapacityRequest,
  TradingSide,
} from "@trading/bot-algo";
import type { BinanceMarketListing } from "../binance-markets.js";
import type { EquitySnapshot } from "./runtime-api.js";
import {
  BinanceExchangeTrading,
  type BinanceExchangeOrder,
  type BinanceExchangePlaceOrderInput,
  type BinanceExchangePosition,
} from "../binance-exchange.js";

export interface BinanceProviderOrder extends BinanceExchangeOrder {}

export interface BinanceProviderTrade {
  id: string;
  orderId: string;
  clientOrderId?: string;
  localOrderId?: string;
  side: TradingSide;
  quantity: number;
  quoteQuantity: number;
  feeQuote: number;
}

export interface BinancePositionContext {
  positions: BinanceExchangePosition[];
  positionMode?: "one-way" | "hedge";
}

export interface BinanceOrderUpdates {
  openOrders: BinanceProviderOrder[];
  recentTrades: BinanceProviderTrade[];
}

export interface BinancePlaceOrderInput extends BinanceExchangePlaceOrderInput {}

export interface BinanceCancelOrderInput {
  orderId?: string;
  clientOrderId?: string;
  algo?: boolean;
}

export interface BinanceTradingClient {
  getEquity(market: BinanceMarketListing): Promise<EquitySnapshot>;
  getPositionContext(market: BinanceMarketListing): Promise<BinancePositionContext>;
  getOrderUpdates(market: BinanceMarketListing): Promise<BinanceOrderUpdates>;
  getMarketRules(market: BinanceMarketListing): Promise<TradingMarketRules>;
  getOrderCapacity(
    market: BinanceMarketListing,
    input: TradingOrderCapacityRequest,
  ): Promise<TradingOrderCapacity>;
  getFriction(market: BinanceMarketListing): Promise<number>;
  placeOrder(market: BinanceMarketListing, input: BinancePlaceOrderInput): Promise<BinanceProviderOrder>;
  cancelOrder(market: BinanceMarketListing, input: BinanceCancelOrderInput): Promise<void>;
}

export class BinanceExchangeClient implements BinanceTradingClient {
  private readonly rules = new Map<string, TradingMarketRules>();
  private readonly friction = new Map<string, number>();

  constructor(private readonly exchange: BinanceExchangeTrading) {}

  async getEquity(market: BinanceMarketListing): Promise<EquitySnapshot> {
    const balances = await this.exchange.fetchAccountBalances(market);
    const quote = balances.find((balance) => balance.asset === market.quoteAsset);
    const base = balances.find((balance) => balance.asset === market.baseAsset);
    const futures = market.venue === "usdm-futures" || market.venue === "coinm-futures";
    const quoteTotal = quote?.walletBalance ?? ((quote?.free ?? 0) + (quote?.locked ?? 0));
    const quoteAvailable = quote?.availableBalance ?? quote?.free ?? 0;
    return {
      quoteAvailable,
      quoteReserved: Math.max(0, quoteTotal - quoteAvailable),
      quoteUnleveraged: quoteTotal + (futures ? quote?.unrealizedPnl ?? 0 : 0),
      assetAvailable: futures ? 0 : base?.free ?? 0,
      assetReserved: futures ? 0 : base?.locked ?? 0,
      assetUnleveraged: futures ? 0 : base ? base.free + base.locked : 0,
    };
  }

  async getPositionContext(market: BinanceMarketListing): Promise<BinancePositionContext> {
    const { positions, positionMode } = await this.exchange.fetchPositionContext(market);
    return { positions, positionMode };
  }

  async getOrderUpdates(market: BinanceMarketListing): Promise<BinanceOrderUpdates> {
    const updates = await this.exchange.fetchOrderUpdates(market);
    return {
      openOrders: updates.openOrders,
      recentTrades: updates.recentTrades.map((trade) => ({
        id: trade.id,
        orderId: trade.orderId,
        clientOrderId: trade.clientOrderId,
        localOrderId: trade.localOrderId,
        side: trade.side,
        quantity: trade.quantity,
        quoteQuantity: trade.quoteQuantity,
        feeQuote: trade.feeQuote,
      })),
    };
  }

  async getMarketRules(market: BinanceMarketListing): Promise<TradingMarketRules> {
    const cached = this.rules.get(market.id);
    if (cached) return cached;
    const { filters, maxLeverage } = await this.exchange.fetchMarketInfo(market);
    const rules: TradingMarketRules = {
      price: { min: filters?.minPrice ?? null, max: filters?.maxPrice ?? null, step: filters?.tickSize ?? null },
      limitQuantity: {
        min: filters?.minQuantity ?? null,
        max: filters?.maxQuantity ?? null,
        step: filters?.stepSize ?? null,
      },
      marketQuantity: {
        min: filters?.minMarketQuantity ?? filters?.minQuantity ?? null,
        max: filters?.maxMarketQuantity ?? filters?.maxQuantity ?? null,
        step: filters?.marketStepSize ?? filters?.stepSize ?? null,
      },
      minNotional: filters?.minNotional ?? null,
      maxNotional: filters?.maxNotional ?? null,
      maxLeverage: maxLeverage ?? market.maxLeverage ?? 1,
    };
    this.rules.set(market.id, rules);
    return rules;
  }

  async getOrderCapacity(
    market: BinanceMarketListing,
    input: TradingOrderCapacityRequest,
  ): Promise<TradingOrderCapacity> {
    const friction = await this.getFriction(market);
    if (market.venue === "spot") {
      const balances = await this.exchange.fetchAccountBalances(market);
      const quote = balances.find((balance) => balance.asset === market.quoteAsset)?.free ?? 0;
      const asset = balances.find((balance) => balance.asset === market.baseAsset)?.free ?? 0;
      return {
        quote: Math.max(0, input.side === "buy" ? quote / (1 + friction) : asset * input.price),
        leverage: 1,
      };
    }

    const state = await this.exchange.fetchOrderCapacityState(market, input.price);
    const positionSide = state.positionMode === "hedge"
      ? input.side === "buy" ? "LONG" : "SHORT"
      : "BOTH";
    const position = state.positions.find((item) => item.positionSide === positionSide)
      ?? state.positions[0];
    const configuredLeverage = position?.leverage && position.leverage > 0
      ? position.leverage
      : input.leverage;
    const leverage = Math.max(1, Math.min(input.leverage, configuredLeverage));
    const marginCapacity = state.availableBalanceQuote / (1 / leverage + friction);
    const currentNotional = Math.abs(position?.notional ?? (position?.positionAmt ?? 0) * input.price);
    const openOrderNotional = state.openOrders
      .filter((order) => orderIncreasesPosition(order, input.side, positionSide))
      .reduce((total, order) => total + remainingOrderNotional(order, input.price), 0);
    const maxNotional = position?.maxNotional
      ?? (position?.maxQuantity === undefined ? Infinity : position.maxQuantity * input.price);
    return {
      quote: Math.max(0, Math.min(marginCapacity, maxNotional - currentNotional - openOrderNotional)),
      leverage,
    };
  }

  async getFriction(market: BinanceMarketListing): Promise<number> {
    const cached = this.friction.get(market.id);
    if (cached !== undefined) return cached;
    const value = await this.exchange.fetchFriction(market);
    const friction = ((value.feeBps ?? 0) + (value.estimatedSlippageBps ?? 0)) / 10_000;
    this.friction.set(market.id, friction);
    return friction;
  }

  placeOrder(market: BinanceMarketListing, input: BinancePlaceOrderInput): Promise<BinanceProviderOrder> {
    return this.exchange.submitOrderDirect(market, input);
  }

  async cancelOrder(market: BinanceMarketListing, input: BinanceCancelOrderInput): Promise<void> {
    await this.exchange.cancelOrderDirect(market, input);
  }
}

function orderIncreasesPosition(
  order: BinanceProviderOrder,
  side: TradingSide,
  positionSide: "BOTH" | "LONG" | "SHORT",
): boolean {
  if (order.side.toLowerCase() !== side) return false;
  const orderPositionSide = order.positionSide?.toUpperCase();
  return positionSide === "BOTH" || !orderPositionSide || orderPositionSide === positionSide;
}

function remainingOrderNotional(order: BinanceProviderOrder, fallbackPrice: number): number {
  const quantity = Math.max(0, order.originalQuantity - order.executedQuantity);
  const price = order.price || order.stopPrice || fallbackPrice;
  return quantity * price;
}
