import type { OrderSide, TradingOrder } from "./types.js";

type FillableOrderFields = Pick<TradingOrder, "side" | "type" | "price" | "trigger">;

export function canFillOrderAtTick(order: FillableOrderFields, tickPrice: number): boolean {
  if (!Number.isFinite(tickPrice) || tickPrice <= 0) {
    return false;
  }

  if (order.type === "market") {
    return true;
  }

  const trigger = order.trigger ?? defaultTriggerForLimitSide(order.side);
  return trigger === "below" ? tickPrice <= order.price : tickPrice >= order.price;
}

export function orderExecutionPrice(
  order: FillableOrderFields,
  tickPrice: number,
  marketSlippageBps: number,
): number {
  const basePrice = marketExecutedOrder(order) ? tickPrice : order.price;
  return marketExecutedOrder(order)
    ? applyMarketSlippage(order.side, basePrice, marketSlippageBps)
    : basePrice;
}

export function applyMarketSlippage(
  side: OrderSide,
  price: number,
  marketSlippageBps: number,
): number {
  if (!Number.isFinite(price) || price <= 0) {
    return price;
  }

  const rate = Math.max(0, marketSlippageBps) / 10_000;
  return side === "buy"
    ? price * (1 + rate)
    : price * Math.max(0.000001, 1 - rate);
}

function marketExecutedOrder(order: FillableOrderFields): boolean {
  return order.type === "market" || order.type === "stop-market" || order.trigger !== undefined;
}

function defaultTriggerForLimitSide(side: OrderSide): "above" | "below" {
  return side === "buy" ? "below" : "above";
}
