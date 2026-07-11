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
import type { BinanceMarketListing } from "../binance-markets.js";
import type {
  BinanceProviderOrder,
  BinancePositionContext,
  BinanceTradingClient,
} from "./binance-client.js";
import type { EquitySnapshot, RuntimeTradingApi } from "./runtime-api.js";

interface LiveOrder {
  order: TradingOrderSnapshot;
  provider?: BinanceProviderOrder;
  filled: number;
}

export interface BinanceReconciliationInput {
  orders?: Array<{
    status: string;
    clientOrderId?: string;
    localOrderId?: string;
  }>;
  fills?: Array<{
    id: string;
    clientOrderId?: string;
    localOrderId?: string;
    side: "buy" | "sell";
    quantity: number;
    quoteQuantity: number;
    feeQuote: number;
  }>;
}

export interface BinanceTradingApiOptions {
  market: BinanceMarketListing;
  client: BinanceTradingClient;
  getHistory: TradingApi["getHistory"];
  snapshot?: BinanceTradingApiSnapshot;
}

export interface BinanceTradingApiSnapshot {
  version: 1;
  orders: LiveOrder[];
  trades: string[];
}

export class BinanceTradingApi implements RuntimeTradingApi {
  private readonly orders = new Map<string, LiveOrder>();
  private readonly trades = new Set<string>();
  private events: TradingOrderEvent[] = [];

  constructor(private readonly options: BinanceTradingApiOptions) {
    for (const item of options.snapshot?.orders ?? []) {
      this.orders.set(item.order.id, structuredClone(item));
    }
    for (const id of options.snapshot?.trades ?? []) {
      this.trades.add(id);
    }
  }

  createStopMarketOrder(input: CreateStopMarketOrderInput): Promise<TradingOrderResult> {
    return this.create("stop-market", input, null, input.price);
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
    const item = this.orders.get(orderId);
    if (!item) {
      return false;
    }
    if (!item.provider) {
      this.orders.delete(orderId);
      return true;
    }
    try {
      await this.options.client.cancelOrder(this.options.market, {
        orderId: item.provider.orderId,
        clientOrderId: item.provider.clientOrderId,
        algo: item.provider.algo,
      });
      this.orders.delete(orderId);
      return true;
    } catch {
      return false;
    }
  }

  getHistory(input: { intervalMs: number; count: number }): Promise<TradingCandle[]> {
    return this.options.getHistory(input);
  }

  async getEquity(): Promise<EquitySnapshot> {
    return this.options.client.getEquity(this.options.market);
  }

  async getMarketRules(): Promise<TradingMarketRules> {
    return this.options.client.getMarketRules(this.options.market);
  }

  async getOrderCapacity(input: TradingOrderCapacityRequest): Promise<TradingOrderCapacity> {
    return this.options.client.getOrderCapacity(this.options.market, input);
  }

  async getFriction(): Promise<number> {
    return this.options.client.getFriction(this.options.market);
  }

  async onTick(tick: TradingTick): Promise<void> {
    for (const item of this.orders.values()) {
      if (
        item.order.type === "stop-limit" &&
        item.order.status === "pending" &&
        stopTriggered(item.order, tick.price)
      ) {
        item.order.status = "open";
        const result = await this.submit(item);
        if (!result.accepted) {
          this.events.push({ type: "rejected", orderId: item.order.id });
        } else {
          this.events.push({ type: "open", order: structuredClone(item.order) });
        }
      }
    }
  }

  drainEvents(): TradingOrderEvent[] {
    const events = this.events;
    this.events = [];
    return events;
  }

  async sync(): Promise<void> {
    const updates = await this.options.client.getOrderUpdates(this.options.market);
    for (const trade of updates.recentTrades) {
      if (this.trades.has(trade.id)) {
        continue;
      }
      const item = this.orders.get(trade.localOrderId ?? trade.clientOrderId ?? "");
      if (!item) {
        continue;
      }
      this.trades.add(trade.id);
      item.filled += trade.quantity;
      const remaining = Math.max(0, item.order.size - item.filled);
      item.order.status = remaining > 0 ? "partially-filled" : "filled";
      const filledQuote = trade.side === "buy"
        ? trade.quoteQuantity + trade.feeQuote
        : trade.quoteQuantity - trade.feeQuote;
      this.events.push({
        type: remaining > 0 ? "partial-fill" : "fill",
        orderId: item.order.id,
        fill: { filledAsset: trade.quantity, filledQuote, remaining },
      });
      if (remaining === 0) {
        this.orders.delete(item.order.id);
      }
    }
    for (const item of this.orders.values()) {
      const provider = updates.openOrders.find(
        (order) => order.localOrderId === item.order.id || order.clientOrderId === item.order.id,
      );
      if (provider) {
        item.provider = provider;
      }
    }
  }

  reconcile(input: BinanceReconciliationInput | undefined): void {
    for (const update of input?.orders ?? []) {
      const item = this.orders.get(update.localOrderId ?? update.clientOrderId ?? "");
      if (!item || update.status !== "open") {
        continue;
      }
      item.order.status = "open";
      this.events.push({ type: "open", order: structuredClone(item.order) });
    }
    for (const fill of input?.fills ?? []) {
      if (this.trades.has(fill.id)) {
        continue;
      }
      const item = this.orders.get(fill.localOrderId ?? fill.clientOrderId ?? "");
      if (!item) {
        continue;
      }
      this.trades.add(fill.id);
      item.filled += fill.quantity;
      const remaining = Math.max(0, item.order.size - item.filled);
      item.order.status = remaining > 0 ? "partially-filled" : "filled";
      this.events.push({
        type: remaining > 0 ? "partial-fill" : "fill",
        orderId: item.order.id,
        fill: {
          filledAsset: fill.quantity,
          filledQuote: fill.side === "buy"
            ? fill.quoteQuantity + fill.feeQuote
            : fill.quoteQuantity - fill.feeQuote,
          remaining,
        },
      });
      if (remaining === 0) {
        this.orders.delete(item.order.id);
      }
    }
  }

  state(): BinanceTradingApiSnapshot {
    return {
      version: 1,
      orders: [...this.orders.values()].map((item) => structuredClone(item)),
      trades: [...this.trades],
    };
  }

  private async create(
    type: TradingOrderType,
    input: CreateMarketOrderInput,
    price: number | null,
    stopPrice: number | null,
  ): Promise<TradingOrderResult> {
    const order: TradingOrderSnapshot = {
      id: liveOrderId(),
      type,
      side: input.side,
      status: type === "stop-limit" ? "pending" : "open",
      size: input.size,
      price,
      stopPrice,
    };
    const item: LiveOrder = { order, filled: 0 };
    this.orders.set(order.id, item);
    if (type === "stop-limit") {
      return { accepted: true, order: structuredClone(order) };
    }
    return this.submit(item);
  }

  private async submit(item: LiveOrder): Promise<TradingOrderResult> {
    const { order } = item;
    try {
      const intent = isFutures(this.options.market)
        ? positionIntent(
            await this.options.client.getPositionContext(this.options.market),
            this.options.market,
            order,
          )
        : {};
      const provider = await this.options.client.placeOrder(this.options.market, {
        side: order.side,
        type: order.type === "stop-market" ? "stop-market" : order.type === "market" ? "market" : "limit",
        quantity: order.size,
        price: order.price ?? undefined,
        stopPrice: order.stopPrice ?? undefined,
        timeInForce: order.type === "limit" || order.type === "stop-limit" ? "GTC" : undefined,
        clientOrderId: order.id,
        ...intent,
      });
      item.provider = provider;
      return { accepted: true, order: structuredClone(order) };
    } catch {
      order.status = "rejected";
      this.orders.delete(order.id);
      return { accepted: false, order: structuredClone(order) };
    }
  }

}

function stopTriggered(order: TradingOrderSnapshot, price: number): boolean {
  return order.side === "buy"
    ? price >= (order.stopPrice ?? 0)
    : price <= (order.stopPrice ?? 0);
}

function liveOrderId(): string {
  return `bot-${crypto.randomUUID().replaceAll("-", "").slice(0, 28)}`;
}

function positionIntent(
  snapshot: BinancePositionContext,
  market: BinanceMarketListing,
  order: TradingOrderSnapshot,
): { reduceOnly?: boolean; positionSide?: "BOTH" | "LONG" | "SHORT" } {
  if (market.venue !== "usdm-futures" && market.venue !== "coinm-futures") {
    return {};
  }
  const positions = snapshot.positions.filter((position) => position.symbol === market.symbol);
  const long = positions.some((position) => position.positionAmt > 0);
  const short = positions.some((position) => position.positionAmt < 0);
  const closing = order.side === "sell" ? long : short;
  if (snapshot.positionMode === "hedge") {
    return {
      positionSide: order.side === "sell"
        ? long ? "LONG" : "SHORT"
        : short ? "SHORT" : "LONG",
    };
  }
  return { positionSide: "BOTH", reduceOnly: closing || undefined };
}

function isFutures(market: BinanceMarketListing): boolean {
  return market.venue === "usdm-futures" || market.venue === "coinm-futures";
}
