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

interface TradingOrderInput {
  side: TradingSide;
  /** Asset/base quantity. */
  size: number;
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
  remaining: number;
}

export type TradingOrderEvent =
  | { type: "open"; order: TradingOrderSnapshot }
  | { type: "rejected"; orderId: string }
  | { type: "partial-fill" | "fill"; orderId: string; fill: TradingFill };

export interface TradingApi {
  createStopMarketOrder(input: CreateStopMarketOrderInput): Promise<TradingOrderResult>;
  createStopLimitOrder(input: CreateStopLimitOrderInput): Promise<TradingOrderResult>;
  createLimitOrder(input: CreateLimitOrderInput): Promise<TradingOrderResult>;
  createMarketOrder(input: CreateMarketOrderInput): Promise<TradingOrderResult>;
  cancelOrder(orderId: string): Promise<boolean>;
  getHistory(input: TradingHistoryRequest): Promise<TradingCandle[]>;
  getMarketRules(): Promise<TradingMarketRules>;
  getOrderCapacity(input: TradingOrderCapacityRequest): Promise<TradingOrderCapacity>;
  /** Expected proportional execution cost, e.g. 0.001 means 0.1%. */
  getFriction(): Promise<number>;
}
