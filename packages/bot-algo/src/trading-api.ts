export type TradingSide = "buy" | "sell";
export type TradingOrderType = "market" | "limit" | "stop-market" | "stop-limit";
export type TradingOrderStatus =
  | "pending"
  | "open"
  | "partially-filled"
  | "filled"
  | "rejected";

export interface TradingCandle {
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  /** Buyer-initiated base volume, when the market-data source supplies aggressor flow. */
  aggressiveBuyVolume?: number;
  /** Seller-initiated base volume, when the market-data source supplies aggressor flow. */
  aggressiveSellVolume?: number;
}

export interface TradingTick {
  timestamp: number;
  price: number;
  quantity: number;
  /** Present when this tick represents a candle update/replay step. */
  candle: TradingCandle | null;
}

export interface TradingHistoryRequest {
  intervalMs: number;
  count: number;
}

export interface TradingQuantityRules {
  min: number | null;
  max: number | null;
  step: number | null;
}

export interface TradingMarketRules {
  price: TradingQuantityRules;
  limitQuantity: TradingQuantityRules;
  marketQuantity: TradingQuantityRules;
  minNotional: number | null;
  maxNotional: number | null;
  maxLeverage: number;
}

export interface TradingOrderCapacityRequest {
  side: TradingSide;
  price: number;
  leverage: number;
}

export interface TradingOrderCapacity {
  /** Maximum additional quote notional accepted by the provider. */
  quote: number;
  /** Leverage actually used to derive the capacity. */
  leverage: number;
}

export interface TradingEquitySnapshot {
  quoteAvailable: number;
  quoteReserved: number;
  quoteUnleveraged: number;
  assetAvailable: number;
  assetReserved: number;
  assetUnleveraged: number;
}

interface TradingOrderInput {
  side: TradingSide;
  /** Asset/base quantity. */
  size: number;
  /** Requested leverage for exposure-increasing fills. */
  leverage?: number;
  /** Closing orders may reduce exposure even when no entry capacity remains. */
  reduceOnly?: boolean;
}

export interface CreateMarketOrderInput extends TradingOrderInput {}

export interface CreateLimitOrderInput extends TradingOrderInput {
  price: number;
}

export interface CreateStopMarketOrderInput extends TradingOrderInput {
  /** Buy stops trigger above this price; sell stops trigger below it. */
  price: number;
}

export interface CreateStopLimitOrderInput extends TradingOrderInput {
  /** Buy stops trigger above this price; sell stops trigger below it. */
  stopPrice: number;
  limitPrice: number;
}

export interface TradingOrderSnapshot {
  id: string;
  type: TradingOrderType;
  side: TradingSide;
  status: TradingOrderStatus;
  size: number;
  price: number | null;
  stopPrice: number | null;
  leverage?: number;
  reduceOnly?: boolean;
}

export interface TradingOrderResult {
  accepted: boolean;
  order: TradingOrderSnapshot;
}

export interface TradingFill {
  /** Net asset amount received or disposed. */
  filledAsset: number;
  /** Absolute net quote amount, including fees and execution friction. */
  filledQuote: number;
  /** Gross execution price before fees/friction, when supplied by the venue. */
  price?: number;
  /** Fee or modeled execution friction charged in quote units. */
  feeQuote?: number;
  remaining: number;
}

export type TradingOrderEvent =
  | { type: "open"; order: TradingOrderSnapshot }
  | { type: "rejected"; orderId: string }
  | { type: "partial-fill" | "fill"; orderId: string; fill: TradingFill }
  | {
      type: "maintenance";
      elapsedMs: number;
      quoteCharge: number;
      assetCharge: number;
    }
  | {
      type: "liquidation";
      at: number;
      price: number;
      equity: number;
      reason: "insolvent" | "effective-leverage";
    };

export interface TradingApi {
  createStopMarketOrder(input: CreateStopMarketOrderInput): Promise<TradingOrderResult>;
  createStopLimitOrder(input: CreateStopLimitOrderInput): Promise<TradingOrderResult>;
  createLimitOrder(input: CreateLimitOrderInput): Promise<TradingOrderResult>;
  createMarketOrder(input: CreateMarketOrderInput): Promise<TradingOrderResult>;
  cancelOrder(orderId: string): Promise<boolean>;
  getHistory(input: TradingHistoryRequest): Promise<TradingCandle[]>;
  getMarketRules(): Promise<TradingMarketRules>;
  getOrderCapacity(input: TradingOrderCapacityRequest): Promise<TradingOrderCapacity>;
  getEquity(): Promise<TradingEquitySnapshot>;
  /** Expected proportional execution cost, e.g. 0.001 means 0.1%. */
  getFriction(): Promise<number>;
}
