import assert from "node:assert/strict";
import type {
  TradingEquitySnapshot,
  TradingOrderCapacityRequest,
  TradingOrderCapacity,
  TradingOrderEvent,
  TradingOrderResult,
  TradingOrderSnapshot,
} from "@trading/bot-algo/trading-api";
import type { BacktestTradingApiConfig } from "./backtest.js";
import {
  floorToStep,
  positive,
  validateLeverage,
  validateOrder,
  withinRules,
} from "./common.js";

interface OrderState {
  order: TradingOrderSnapshot;
  triggered: boolean;
  reserve: number;
}

/** Account arithmetic and execution along a piecewise linear mark-price path. */
export class BacktestAccount {
  private walletBalance: number;
  private positionSize = 0;
  private entryPrice = 0;
  private realizedPnl = 0;
  private feesPaid = 0;
  private fundingPaid = 0;
  private liquidated = false;
  private readonly orders = new Map<string, OrderState>();
  private events: TradingOrderEvent[] = [];
  private nextId = 1;
  private fundingIndex = 0;
  private lastFundingTime: number;

  constructor(
    private readonly config: BacktestTradingApiConfig,
    public price: number | null,
    private time: number,
  ) {
    this.walletBalance = config.initialQuote;
    this.lastFundingTime = time;
    while ((config.funding?.[this.fundingIndex]?.timestamp ?? Infinity) <= time)
      this.fundingIndex++;
  }

  private get reserved() {
    let total = 0;
    for (const state of this.orders.values()) total += state.reserve;
    return total;
  }
  private equityAt(price: number) {
    return this.walletBalance + this.positionSize * (price - this.entryPrice);
  }
  get equity(): TradingEquitySnapshot {
    const price = this.price ?? 0;
    const equity = this.equityAt(price);
    const reserved =
      (Math.abs(this.positionSize) * price) / this.config.leverage +
      this.reserved;
    return {
      quoteAvailable: Math.max(0, equity - reserved),
      quoteReserved: reserved,
      quoteUnleveraged: equity - this.positionSize * price,
      assetAvailable: this.positionSize,
      assetReserved: 0,
      assetUnleveraged: this.positionSize,
    };
  }
  snapshot() {
    const equity = this.equityAt(this.price ?? 0);
    return {
      walletBalance: this.walletBalance,
      positionSize: this.positionSize,
      entryPrice: this.entryPrice,
      equity,
      unrealizedPnl: equity - this.walletBalance,
      realizedPnl: this.realizedPnl,
      feesPaid: this.feesPaid,
      fundingPaid: this.fundingPaid,
      liquidated: this.liquidated,
      price: this.price,
      timestamp: this.time,
      ...this.equity,
    } as const;
  }
  getOrder(id: string) {
    const state = this.orders.get(id);
    return state?.order;
  }
  drainOrderEvents(): readonly TradingOrderEvent[] {
    const events = this.events;
    this.events = [];
    return events;
  }
  cancelOrder(id: string) {
    const state = this.orders.get(id);
    if (!state || !isActive(state)) return false;
    state.order = { ...state.order, status: "canceled" };
    state.reserve = 0;
    return true;
  }
  cancelAll() {
    for (const id of this.orders.keys()) this.cancelOrder(id);
  }

  getOrderCapacity(
    input: TradingOrderCapacityRequest,
    finished = false,
  ): TradingOrderCapacity {
    positive(input.price, "price");
    validateLeverage(input.leverage);
    assert(input.side === "buy" || input.side === "sell", "Invalid order side");
    const leverage = Math.min(input.leverage, this.config.leverage);
    if (finished || this.liquidated || this.price === null)
      return { quote: 0, leverage };
    const sign = input.side === "buy" ? 1 : -1;
    const price = input.price;
    const cost = this.costRate(sign, true);
    const equity = this.equityAt(price) - this.reserved;
    const exposure = sign * this.positionSize;
    const closing = Math.max(0, -exposure);
    const openingBound =
      (equity / price - exposure / leverage) / (cost + 1 / leverage);
    const bound =
      openingBound >= closing
        ? openingBound
        : cost > 1 / leverage
          ? (equity / price - Math.abs(this.positionSize) / leverage) /
            (cost - 1 / leverage)
          : 0;
    const rules = this.config.marketRules;
    const pending = this.pendingSize(input.side);
    const size = Math.max(
      0,
      floorToStep(
        Math.min(
          bound,
          rules.marketQuantity.max ?? Infinity,
          (rules.maxNotional ?? Infinity) / price - exposure - pending,
        ),
        rules.marketQuantity.step,
      ),
    );
    const quote = size * price;
    return {
      quote:
        size < (rules.marketQuantity.min ?? 0) ||
        (size > closing && quote < (rules.minNotional ?? 0))
          ? 0
          : quote,
      leverage,
    };
  }

  submit(input: TradingOrderSnapshot, finished = false): TradingOrderResult {
    validateOrder(input);
    assert(
      ["market", "limit", "stop-market", "stop-limit"].includes(input.type),
      "Invalid order type",
    );
    assert(
      (input.type === "limit" || input.type === "stop-limit") ===
        (input.price !== null),
      "Invalid limit price",
    );
    assert(
      (input.type === "stop-market" || input.type === "stop-limit") ===
        (input.stopPrice !== null),
      "Invalid stop price",
    );
    const order: TradingOrderSnapshot = {
      ...input,
      id: `backtest:${this.nextId++}`,
      status: input.type === "market" ? "pending" : "open",
      leverage: this.config.leverage,
    };
    const state: OrderState = {
      order,
      triggered: order.stopPrice === null,
      reserve: 0,
    };
    this.orders.set(order.id, state);
    const price = order.price ?? order.stopPrice ?? this.price;
    const valid =
      !finished &&
      !this.liquidated &&
      this.price !== null &&
      price !== null &&
      this.validRules(order, price) &&
      this.canExecute(order, price, this.reserved, this.price) &&
      this.withinPositionLimit(
        order,
        price,
        this.pendingSize(order.side, order.id),
      );
    if (!valid) this.reject(state);
    else {
      const sign = order.side === "buy" ? 1 : -1;
      const marginChange =
        ((Math.abs(this.positionSize + sign * order.size) -
          Math.abs(this.positionSize)) *
          price) /
        this.config.leverage;
      state.reserve = Math.max(
        0,
        marginChange +
          order.size * price * this.costRate(sign, isMarket(order)),
      );
      this.events.push({ type: "open", order });
    }
    return { accepted: valid, order: state.order };
  }

  private pendingSize(side: TradingOrderSnapshot["side"], except?: string) {
    let total = 0;
    for (const state of this.orders.values())
      if (
        state.order.id !== except &&
        isActive(state) &&
        state.order.side === side
      )
        total += state.order.size;
    return total;
  }
  private costRate(sign: number, market: boolean) {
    const slip = market ? this.config.slippageRate : 0;
    return slip + this.config.feeRate * (1 + sign * slip);
  }
  private reducing(order: TradingOrderSnapshot) {
    return (
      (order.side === "buy" ? 1 : -1) * this.positionSize < 0 &&
      order.size <= Math.abs(this.positionSize) + 1e-12
    );
  }
  private validRules(order: TradingOrderSnapshot, price: number) {
    const rules = this.config.marketRules;
    return (
      withinRules(
        order.size,
        isMarket(order) ? rules.marketQuantity : rules.limitQuantity,
      ) &&
      (order.price === null || withinRules(order.price, rules.price)) &&
      (order.stopPrice === null || withinRules(order.stopPrice, rules.price)) &&
      (this.reducing(order) ||
        order.size * price >= (rules.minNotional ?? 0) - 1e-10)
    );
  }
  private withinPositionLimit(
    order: TradingOrderSnapshot,
    price: number,
    pending = 0,
  ) {
    const sign = order.side === "buy" ? 1 : -1;
    return (
      this.reducing(order) ||
      Math.abs(this.positionSize + sign * (order.size + pending)) * price <=
        (this.config.marketRules.maxNotional ?? Infinity) + 1e-10
    );
  }
  private canExecute(
    order: TradingOrderSnapshot,
    price: number,
    reserved: number,
    mark = price,
  ) {
    if (order.reduceOnly && !this.reducing(order)) return false;
    const sign = order.side === "buy" ? 1 : -1;
    const equity =
      this.equityAt(mark) -
      order.size * price * this.costRate(sign, isMarket(order));
    if (this.reducing(order)) return equity >= -1e-10;
    const margin =
      (Math.abs(this.positionSize + sign * order.size) * price) /
      this.config.leverage;
    return equity >= margin + reserved - 1e-10;
  }

  /** Gaps match at the new open; continuous segments match each intervening crossing. */
  advance(target: number, at: number, gap = false) {
    assert(at >= this.time, "Account time cannot go backwards");
    if (gap || this.price === null) {
      // Missing candles do not invent a price path. Funding still uses its supplied mark.
      this.applyFundingBefore(at);
      this.price = target;
      this.time = at;
      this.match();
      this.applyFundingAt(at);
      return;
    }
    while (this.time < at || this.price !== target) {
      const from: number = this.price!;
      const start = this.time;
      let fraction = 1;
      let nextPrice = target;
      let nextTime = at;
      const crossing = (price: number) => {
        const f = (price - from) / (target - from);
        if (f > 0 && f < fraction) {
          fraction = f;
          nextPrice = price;
          nextTime = start + (at - start) * f;
        }
      };
      if (target !== from && !this.liquidated) {
        const liquidation = this.liquidationPrice();
        if (liquidation !== null) crossing(liquidation);
        for (const state of this.orders.values()) {
          if (!isActive(state)) continue;
          const threshold = state.triggered
            ? state.order.price
            : state.order.stopPrice;
          if (threshold !== null) crossing(threshold);
        }
      }
      const funding = this.config.funding?.[this.fundingIndex];
      if (funding && funding.timestamp <= nextTime && at > start) {
        if (funding.timestamp < nextTime)
          nextPrice =
            from +
            ((target - from) * (funding.timestamp - start)) / (at - start);
        nextTime = funding.timestamp;
      }
      this.price = nextPrice;
      this.time = nextTime;
      this.match();
      this.applyFundingAt(this.time);
    }
    this.match();
  }

  private match() {
    if (this.checkLiquidation()) return;
    const price = this.price!;
    for (const state of this.orders.values()) {
      if (!isActive(state)) continue;
      const order = state.order;
      if (
        !state.triggered &&
        (order.side === "buy"
          ? price >= order.stopPrice!
          : price <= order.stopPrice!)
      )
        state.triggered = true;
      if (!state.triggered) continue;
      if (
        order.price !== null &&
        !(order.side === "buy" ? price <= order.price : price >= order.price)
      )
        continue;
      this.fill(state, price);
      if (this.checkLiquidation()) return;
    }
  }
  private fill(state: OrderState, price: number) {
    const order = state.order;
    state.reserve = 0;
    if (
      !this.validRules(order, price) ||
      !this.withinPositionLimit(order, price) ||
      !this.canExecute(order, price, this.reserved)
    ) {
      this.reject(state);
      return;
    }
    const sign = order.side === "buy" ? 1 : -1;
    const execution =
      price * (1 + (isMarket(order) ? sign * this.config.slippageRate : 0));
    const fee = order.size * execution * this.config.feeRate;
    this.trade(sign * order.size, execution, fee);
    state.order = { ...order, status: "filled" };
    const friction = order.size * Math.abs(execution - price) + fee;
    this.events.push({
      type: "fill",
      orderId: order.id,
      fill: {
        filledAsset: sign * order.size,
        filledQuote: order.size * execution + sign * fee,
        price,
        feeQuote: friction,
        remaining: 0,
      },
    });
  }
  private trade(delta: number, execution: number, fee: number) {
    const previous = this.positionSize;
    const next = previous + delta;
    const closing =
      previous * delta < 0 ? Math.min(Math.abs(previous), Math.abs(delta)) : 0;
    const pnl = closing * Math.sign(previous) * (execution - this.entryPrice);
    this.realizedPnl += pnl;
    this.walletBalance += pnl - fee;
    this.feesPaid += fee;
    if (Math.abs(next) < 1e-12) this.entryPrice = 0;
    else if (previous === 0 || previous * next < 0) this.entryPrice = execution;
    else if (previous * delta > 0)
      this.entryPrice =
        (Math.abs(previous) * this.entryPrice + Math.abs(delta) * execution) /
        Math.abs(next);
    this.positionSize = Math.abs(next) < 1e-12 ? 0 : next;
  }
  private reject(state: OrderState) {
    state.order = { ...state.order, status: "rejected" };
    state.reserve = 0;
    this.events.push({ type: "rejected", orderId: state.order.id });
  }
  private liquidationPrice() {
    if (this.positionSize === 0) return null;
    const rate = this.config.maintenanceMarginRate;
    const price =
      (this.positionSize * this.entryPrice - this.walletBalance) /
      (this.positionSize - Math.abs(this.positionSize) * rate);
    return price > 0 ? price : null;
  }
  private checkLiquidation() {
    if (this.liquidated) return true;
    if (this.positionSize === 0 || this.price === null) return false;
    const equity = this.equityAt(this.price);
    const maintenance =
      Math.abs(this.positionSize) *
      this.price *
      this.config.maintenanceMarginRate;
    if (equity > maintenance + 1e-10) return false;
    const fee =
      Math.abs(this.positionSize) * this.price * this.config.liquidationFeeRate;
    this.trade(-this.positionSize, this.price, fee);
    this.liquidated = true;
    this.cancelAll();
    this.events.push({
      type: "liquidation",
      at: Math.round(this.time),
      price: this.price,
      equity: this.walletBalance,
      reason: equity <= 0 ? "insolvent" : "effective-leverage",
    });
    return true;
  }
  private applyFundingBefore(at: number) {
    while (
      (this.config.funding?.[this.fundingIndex]?.timestamp ?? Infinity) < at
    )
      this.applyFunding();
  }
  private applyFundingAt(at: number) {
    while (
      (this.config.funding?.[this.fundingIndex]?.timestamp ?? Infinity) <= at
    )
      this.applyFunding();
  }
  private applyFunding() {
    const funding = this.config.funding![this.fundingIndex++];
    const charge = this.positionSize * funding.markPrice * funding.rate;
    this.walletBalance -= charge;
    this.fundingPaid += charge;
    this.events.push({
      type: "maintenance",
      elapsedMs: Math.max(0, funding.timestamp - this.lastFundingTime),
      quoteCharge: charge,
      assetCharge: 0,
    });
    this.lastFundingTime = funding.timestamp;
    // Between observed candles, the funding mark is the only supplied valuation.
    const previousPrice = this.price;
    const previousTime = this.time;
    if (funding.timestamp > this.time) {
      this.price = funding.markPrice;
      this.time = funding.timestamp;
    }
    this.checkLiquidation();
    this.price = previousPrice;
    this.time = previousTime;
  }
}

function isActive(state: OrderState) {
  return state.order.status === "open" || state.order.status === "pending";
}
function isMarket(order: TradingOrderSnapshot) {
  return order.type === "market" || order.type === "stop-market";
}
