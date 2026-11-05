import type {
  TradingApi,
  TradingCandle,
  TradingEquitySnapshot,
  TradingHistoryRequest,
  TradingMarketRules,
  TradingOrderCapacity,
  TradingOrderCapacityRequest,
  TradingOrderResult,
  TradingOrderSnapshot,
  CreateMarketOrderInput,
  CreateLimitOrderInput,
  CreateStopMarketOrderInput,
  CreateStopLimitOrderInput,
} from "@trading/bot-algo/trading-api";

/** Constructs pending orders; adapters own validation and execution through submit(). */
export abstract class BaseTradingApi implements TradingApi {
  protected abstract submit(
    order: TradingOrderSnapshot,
  ): Promise<TradingOrderResult>;
  abstract cancelOrder(orderId: string): Promise<boolean>;
  abstract getHistory(
    input: TradingHistoryRequest,
  ): Promise<readonly TradingCandle[]>;
  abstract getMarketRules(): Promise<TradingMarketRules>;
  abstract getOrderCapacity(
    input: TradingOrderCapacityRequest,
  ): Promise<TradingOrderCapacity>;
  abstract getEquity(): Promise<TradingEquitySnapshot>;
  abstract getFriction(size?: number): Promise<number>;

  createMarketOrder(
    input: CreateMarketOrderInput,
  ): Promise<TradingOrderResult> {
    return this.submit({
      ...input,
      id: "",
      type: "market",
      status: "pending",
      price: null,
      stopPrice: null,
    });
  }

  createLimitOrder(input: CreateLimitOrderInput): Promise<TradingOrderResult> {
    return this.submit({
      ...input,
      id: "",
      type: "limit",
      status: "pending",
      stopPrice: null,
    });
  }

  createStopMarketOrder(
    input: CreateStopMarketOrderInput,
  ): Promise<TradingOrderResult> {
    return this.submit({
      ...input,
      id: "",
      type: "stop-market",
      status: "pending",
      price: null,
      stopPrice: input.price,
    });
  }

  createStopLimitOrder(
    input: CreateStopLimitOrderInput,
  ): Promise<TradingOrderResult> {
    return this.submit({
      side: input.side,
      size: input.size,
      reduceOnly: input.reduceOnly,
      id: "",
      type: "stop-limit",
      status: "pending",
      price: input.limitPrice,
      stopPrice: input.stopPrice,
    });
  }
}
