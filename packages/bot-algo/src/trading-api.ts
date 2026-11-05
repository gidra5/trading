/** Data contracts are immutable; mutable builders stay private to their owner. */
export type Immutable<T> = { readonly [K in keyof T]: Immutable<T[K]> };

export type TradingSide = "buy" | "sell";
export type TradingOrderType =
  "market" | "limit" | "stop-market" | "stop-limit";
export type TradingOrderStatus =
  | "pending"
  | "open"
  | "partially-filled"
  | "filled"
  | "canceled"
  | "expired"
  | "rejected";

export interface TradingCandle {
  readonly openTime: number;
  readonly closeTime: number;
  readonly open: number;
  readonly high: number;
  readonly low: number;
  readonly close: number;
  readonly volume: number;
  /** Buyer-initiated base volume, when the market-data source supplies aggressor flow. */
  readonly aggressiveBuyVolume?: number;
  /** Seller-initiated base volume, when the market-data source supplies aggressor flow. */
  readonly aggressiveSellVolume?: number;
}

export interface TradingTick {
  readonly timestamp: number;
  readonly price: number;
  readonly quantity: number;
  /** Present when this tick represents a candle update/replay step. */
  readonly candle: TradingCandle | null;
}

export interface TradingHistoryRequest {
  readonly intervalMs: number;
  readonly count: number;
}

export interface TradingQuantityRules {
  readonly min: number | null;
  readonly max: number | null;
  readonly step: number | null;
}

export interface TradingMarketRules {
  readonly price: TradingQuantityRules;
  readonly limitQuantity: TradingQuantityRules;
  readonly marketQuantity: TradingQuantityRules;
  readonly minNotional: number | null;
  readonly maxNotional: number | null;
  readonly maxLeverage: number;
}

export interface TradingOrderCapacityRequest {
  readonly side: TradingSide;
  readonly price: number;
  readonly leverage: number;
}

export interface TradingOrderCapacity {
  /** Maximum additional quote notional accepted by the provider. */
  readonly quote: number;
  /** Leverage actually used to derive the capacity. */
  readonly leverage: number;
}

export interface TradingEquitySnapshot {
  readonly quoteAvailable: number;
  readonly quoteReserved: number;
  readonly quoteUnleveraged: number;
  readonly assetAvailable: number;
  readonly assetReserved: number;
  readonly assetUnleveraged: number;
}

interface TradingOrderInput {
  readonly side: TradingSide;
  /** Asset/base quantity. */
  readonly size: number;
  /** Futures: only reduce an existing position, never open or reverse one. */
  readonly reduceOnly?: boolean;
}

export interface CreateMarketOrderInput extends TradingOrderInput {}

export interface CreateLimitOrderInput extends TradingOrderInput {
  readonly price: number;
}

export interface CreateStopMarketOrderInput extends TradingOrderInput {
  /** Buy stops trigger above this price; sell stops trigger below it. */
  readonly price: number;
}

export interface CreateStopLimitOrderInput extends TradingOrderInput {
  /** Buy stops trigger above this price; sell stops trigger below it. */
  readonly stopPrice: number;
  readonly limitPrice: number;
}

export interface TradingOrderSnapshot {
  readonly id: string;
  readonly type: TradingOrderType;
  readonly side: TradingSide;
  readonly status: TradingOrderStatus;
  readonly size: number;
  readonly price: number | null;
  readonly stopPrice: number | null;
  readonly leverage?: number;
  readonly reduceOnly?: boolean;
}

export interface TradingOrderResult {
  readonly accepted: boolean;
  readonly order: TradingOrderSnapshot;
}

export interface TradingFill {
  /** Net asset amount received or disposed. */
  readonly filledAsset: number;
  /** Absolute net quote amount, including fees and execution friction. */
  readonly filledQuote: number;
  /** Gross execution price before fees/friction, when supplied by the venue. */
  readonly price?: number;
  /** Fee or modeled execution friction charged in quote units. */
  readonly feeQuote?: number;
  readonly remaining: number;
}

export type TradingOrderEvent =
  | { readonly type: "open"; readonly order: TradingOrderSnapshot }
  | { readonly type: "rejected"; readonly orderId: string }
  | {
      readonly type: "partial-fill" | "fill";
      readonly orderId: string;
      readonly fill: TradingFill;
    }
  | {
      readonly type: "maintenance";
      readonly elapsedMs: number;
      readonly quoteCharge: number;
      readonly assetCharge: number;
    }
  | {
      readonly type: "liquidation";
      readonly at: number;
      readonly price: number;
      readonly equity: number;
      readonly reason: "insolvent" | "effective-leverage";
    };

export interface TradingApi {
  createStopMarketOrder(
    input: CreateStopMarketOrderInput,
  ): Promise<TradingOrderResult>;
  createStopLimitOrder(
    input: CreateStopLimitOrderInput,
  ): Promise<TradingOrderResult>;
  createLimitOrder(input: CreateLimitOrderInput): Promise<TradingOrderResult>;
  createMarketOrder(input: CreateMarketOrderInput): Promise<TradingOrderResult>;
  cancelOrder(orderId: string): Promise<boolean>;
  getHistory(input: TradingHistoryRequest): Promise<readonly TradingCandle[]>;
  getMarketRules(): Promise<TradingMarketRules>;
  getOrderCapacity(
    input: TradingOrderCapacityRequest,
  ): Promise<TradingOrderCapacity>;
  getEquity(): Promise<TradingEquitySnapshot>;
  /**
   * Expected proportional execution cost, e.g. 0.001 means 0.1%.
   * Optional base quantity selects a size-specific estimate; omitted uses the provider's average.
   */
  getFriction(size?: number): Promise<number>;
}
