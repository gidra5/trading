import type {
  CreateLimitOrderInput,
  CreateMarketOrderInput,
  CreateStopLimitOrderInput,
  CreateStopMarketOrderInput,
  TradingApi,
  TradingCandle,
  TradingMarketRules,
  TradingOrderCapacity,
  TradingOrderCapacityRequest,
  TradingOrderEvent,
  TradingOrderResult,
  TradingOrderSnapshot,
  TradingOrderType,
  TradingTick,
} from "@trading/bot-algo";
import type { EquitySnapshot, RuntimeTradingApi } from "./runtime-api.js";

interface PaperOrder {
  order: TradingOrderSnapshot;
  triggered: boolean;
  filled: number;
}

export interface PaperTradingSnapshot {
  version: 1;
  quote: number;
  asset: number;
  orders: PaperOrder[];
}

export interface PaperTradingApiOptions {
  startingQuote: number;
  friction: number;
  rules: TradingMarketRules;
  getHistory: TradingApi["getHistory"];
  snapshot?: PaperTradingSnapshot;
}

export class PaperTradingApi implements RuntimeTradingApi {
  private quote: number;
  private asset: number;
  private price = 0;
  private readonly orders = new Map<string, PaperOrder>();
  private events: TradingOrderEvent[] = [];

  constructor(private readonly options: PaperTradingApiOptions) {
    this.quote = options.snapshot?.quote ?? options.startingQuote;
    this.asset = options.snapshot?.asset ?? 0;
    for (const item of options.snapshot?.orders ?? []) {
      this.orders.set(item.order.id, structuredClone(item));
    }
  }

  createStopMarketOrder(input: CreateStopMarketOrderInput): Promise<TradingOrderResult> {
    return this.create("stop-market", input, input.price, input.price);
  }

  createStopLimitOrder(input: CreateStopLimitOrderInput): Promise<TradingOrderResult> {
    return this.create("stop-limit", input, input.limitPrice, input.stopPrice);
  }

  createLimitOrder(input: CreateLimitOrderInput): Promise<TradingOrderResult> {
    return this.create("limit", input, input.price, null);
  }

  createMarketOrder(input: CreateMarketOrderInput): Promise<TradingOrderResult> {
    return this.create("market", input, null, null);
  }

  async cancelOrder(orderId: string): Promise<boolean> {
    return this.orders.delete(orderId);
  }

  getHistory(input: { intervalMs: number; count: number }): Promise<TradingCandle[]> {
    return this.options.getHistory(input);
  }

  async getEquity(): Promise<EquitySnapshot> {
    const reserved = [...this.orders.values()].reduce(
      (total, { order, filled }) => {
        const remaining = Math.max(0, order.size - filled);
        if (order.side === "buy") {
          total.quote += remaining * (order.price ?? order.stopPrice ?? this.price) * (1 + this.options.friction);
        } else {
          total.asset += Math.min(Math.max(0, this.asset), remaining);
          total.quote += Math.max(0, remaining - Math.max(0, this.asset))
            * (order.price ?? order.stopPrice ?? this.price) / this.options.rules.maxLeverage;
        }
        return total;
      },
      { quote: 0, asset: 0 },
    );
    const shortAsset = Math.max(0, -this.asset);
    const shortLiability = shortAsset * this.price;
    const shortMargin = shortLiability / this.options.rules.maxLeverage;
    const freeCollateral = this.quote - shortLiability - shortMargin;
    return {
      quoteAvailable: Math.max(0, freeCollateral - reserved.quote),
      quoteReserved: reserved.quote,
      quoteUnleveraged: this.quote,
      assetAvailable: Math.max(0, this.asset - reserved.asset),
      assetReserved: reserved.asset,
      assetUnleveraged: this.asset,
    };
  }

  async getMarketRules(): Promise<TradingMarketRules> {
    return this.options.rules;
  }

  async getOrderCapacity(input: TradingOrderCapacityRequest): Promise<TradingOrderCapacity> {
    const leverage = Math.max(1, Math.min(input.leverage, this.options.rules.maxLeverage));
    const available = (await this.getEquity()).quoteAvailable;
    const quote = available / (1 / leverage + Math.max(0, this.options.friction));
    return {
      quote: Math.max(0, quote),
      leverage,
    };
  }

  async getFriction(): Promise<number> {
    return this.options.friction;
  }

  async onTick(tick: TradingTick): Promise<void> {
    this.price = tick.price;
    let liquidity = tick.quantity > 0 ? tick.quantity : Infinity;
    for (const item of [...this.orders.values()]) {
      if (item.order.status === "pending" && stopTriggered(item.order, tick.price)) {
        item.triggered = true;
        item.order.status = "open";
        this.events.push({ type: "open", order: structuredClone(item.order) });
      }
      if (item.triggered && canFill(item.order, tick.price) && liquidity > 0) {
        liquidity -= this.fill(item, executionPrice(item.order, tick.price), liquidity);
      }
    }
  }

  drainEvents(): TradingOrderEvent[] {
    const events = this.events;
    this.events = [];
    return events;
  }

  snapshot(): PaperTradingSnapshot {
    return {
      version: 1,
      quote: this.quote,
      asset: this.asset,
      orders: [...this.orders.values()].map((item) => structuredClone(item)),
    };
  }

  private async create(
    type: TradingOrderType,
    input: CreateMarketOrderInput,
    price: number | null,
    stopPrice: number | null,
  ): Promise<TradingOrderResult> {
    const order: TradingOrderSnapshot = {
      id: crypto.randomUUID(),
      type,
      side: input.side,
      status: type.startsWith("stop-") ? "pending" : "open",
      size: input.size,
      price,
      stopPrice,
    };
    if (!this.valid(order)) {
      order.status = "rejected";
      return { accepted: false, order };
    }
    const item = { order, triggered: order.status === "open", filled: 0 };
    this.orders.set(order.id, item);
    if (type === "market") {
      if (this.price <= 0) {
        this.orders.delete(order.id);
        order.status = "rejected";
        return { accepted: false, order };
      }
      this.fill(item, this.price, Infinity);
    }
    return { accepted: true, order: structuredClone(order) };
  }

  private valid(order: TradingOrderSnapshot): boolean {
    if (!Number.isFinite(order.size) || order.size <= 0) {
      return false;
    }
    const rules = order.type === "market"
      ? this.options.rules.marketQuantity
      : this.options.rules.limitQuantity;
    if ((rules.min !== null && order.size < rules.min) || (rules.max !== null && order.size > rules.max)) {
      return false;
    }
    const price = order.price ?? order.stopPrice ?? this.price;
    const notional = order.size * price;
    return price > 0
      && (this.options.rules.minNotional === null || notional >= this.options.rules.minNotional)
      && (this.options.rules.maxNotional === null || notional <= this.options.rules.maxNotional);
  }

  private fill(item: PaperOrder, price: number, liquidity: number): number {
    const { order } = item;
    const quantity = Math.min(order.size - item.filled, liquidity);
    const quote = quantity * price;
    const filledQuote = order.side === "buy"
      ? quote * (1 + this.options.friction)
      : quote * (1 - this.options.friction);
    if (order.side === "buy") {
      this.quote -= filledQuote;
      this.asset += quantity;
    } else {
      this.quote += filledQuote;
      this.asset -= quantity;
    }
    item.filled += quantity;
    const remaining = Math.max(0, order.size - item.filled);
    order.status = remaining > 0 ? "partially-filled" : "filled";
    if (remaining === 0) {
      this.orders.delete(order.id);
    }
    this.events.push({
      type: remaining > 0 ? "partial-fill" : "fill",
      orderId: order.id,
      fill: { filledAsset: quantity, filledQuote, remaining },
    });
    return quantity;
  }
}

function stopTriggered(order: TradingOrderSnapshot, price: number): boolean {
  const stop = order.stopPrice ?? 0;
  return order.side === "buy" ? price >= stop : price <= stop;
}

function canFill(order: TradingOrderSnapshot, price: number): boolean {
  if (order.type === "market" || order.type === "stop-market") {
    return true;
  }
  const limit = order.price ?? 0;
  return order.side === "buy" ? price <= limit : price >= limit;
}

function executionPrice(order: TradingOrderSnapshot, price: number): number {
  if (order.type === "market" || order.type === "stop-market") {
    return price;
  }
  return order.price ?? price;
}
