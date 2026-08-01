import type {
  CreateLimitOrderInput,
  CreateMarketOrderInput,
  CreateStopLimitOrderInput,
  CreateStopMarketOrderInput,
  TradingApi,
  TradingCandle,
  TradingEquitySnapshot,
  TradingMarketRules,
  TradingOrderCapacity,
  TradingOrderCapacityRequest,
  TradingOrderEvent,
  TradingOrderResult,
  TradingOrderSnapshot,
  TradingOrderType,
  TradingTick,
} from "./trading-api.js";

interface SimulatedOrder {
  order: TradingOrderSnapshot;
  triggered: boolean;
  filled: number;
}

export interface SimulatedTradingSnapshot {
  version: 1;
  quote: number;
  asset: number;
  price: number;
  updatedAt: number;
  orders: SimulatedOrder[];
  feesPaid: number;
  maintenancePaid: number;
  liquidationCount: number;
  liquidatedAt: number | null;
  liquidationPrice: number | null;
  liquidationReason: "insolvent" | "effective-leverage" | null;
  maxEffectiveLeverage: number;
}

export interface SimulatedTradingApiOptions {
  startingQuote: number;
  friction: number;
  rules: TradingMarketRules;
  getHistory: TradingApi["getHistory"];
  /** Hourly rates expressed in basis points and charged only on negative balances. */
  quoteBorrowBpsHour?: number;
  assetBorrowBpsHour?: number;
  /** Drift boundary. Infinity means liquidation only when closeout equity is exhausted. */
  maxEffectiveLeverage?: number;
  snapshot?: SimulatedTradingSnapshot;
}

export interface SimulatedTradingStatus {
  liquidated: boolean;
  liquidationCount: number;
  liquidatedAt: number | null;
  liquidationPrice: number | null;
  liquidationReason: "insolvent" | "effective-leverage" | null;
  equity: number;
  closeoutEquity: number;
  effectiveLeverage: number;
  maxEffectiveLeverage: number;
  feesPaid: number;
  maintenancePaid: number;
}

const HOUR_MS = 3_600_000;
const EPSILON = 1e-12;
export const DEFAULT_SIMULATED_BORROW_BPS_HOUR = 5;
export const DEFAULT_SIMULATED_MAX_EFFECTIVE_LEVERAGE = 250;

/**
 * Event-driven exchange/account simulator used by both paper trading and backtests.
 * Negative quote and asset balances are explicit margin debts. Risk is checked
 * before and after every fill and every observed market price.
 */
export class SimulatedTradingApi implements TradingApi {
  private quote: number;
  private asset: number;
  private price: number;
  private updatedAt: number;
  private readonly orders = new Map<string, SimulatedOrder>();
  private events: TradingOrderEvent[] = [];
  private feesPaidValue: number;
  private maintenancePaidValue: number;
  private liquidationCountValue: number;
  private liquidatedAtValue: number | null;
  private liquidationPriceValue: number | null;
  private liquidationReasonValue: "insolvent" | "effective-leverage" | null;
  private maxEffectiveLeverageValue: number;

  constructor(private readonly options: SimulatedTradingApiOptions) {
    const snapshot = options.snapshot;
    this.quote = snapshot?.quote ?? options.startingQuote;
    this.asset = snapshot?.asset ?? 0;
    this.price = snapshot?.price ?? 0;
    this.updatedAt = snapshot?.updatedAt ?? 0;
    this.feesPaidValue = snapshot?.feesPaid ?? 0;
    this.maintenancePaidValue = snapshot?.maintenancePaid ?? 0;
    this.liquidationCountValue = snapshot?.liquidationCount ?? 0;
    this.liquidatedAtValue = snapshot?.liquidatedAt ?? null;
    this.liquidationPriceValue = snapshot?.liquidationPrice ?? null;
    this.liquidationReasonValue = snapshot?.liquidationReason ?? null;
    this.maxEffectiveLeverageValue = snapshot?.maxEffectiveLeverage ?? 0;
    for (const item of snapshot?.orders ?? []) {
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

  async getEquity(): Promise<TradingEquitySnapshot> {
    const reserved = this.reservedBalances();
    const shortAsset = Math.max(0, -this.asset);
    const shortLiability = shortAsset * this.price;
    const shortMargin = shortLiability / Math.max(1, this.options.rules.maxLeverage);
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

  getUnleveragedBalances(): { quote: number; asset: number } {
    return { quote: this.quote, asset: this.asset };
  }

  hasOpenOrders(): boolean {
    return this.orders.size > 0;
  }

  hasExposure(): boolean {
    return Math.abs(this.asset) > EPSILON;
  }

  isLiquidated(): boolean {
    return this.liquidatedAtValue !== null;
  }

  async getMarketRules(): Promise<TradingMarketRules> {
    return this.options.rules;
  }

  async getOrderCapacity(input: TradingOrderCapacityRequest): Promise<TradingOrderCapacity> {
    const leverage = Math.max(1, Math.min(input.leverage, this.options.rules.maxLeverage));
    return {
      quote: this.entryCapacity(input.side, input.price, leverage, true),
      leverage,
    };
  }

  async getFriction(): Promise<number> {
    return this.friction();
  }

  async onTick(tick: TradingTick): Promise<void> {
    if (!(tick.price > 0) || !Number.isFinite(tick.price)) return;
    if (this.isLiquidated()) {
      this.price = tick.price;
      this.updatedAt = Math.max(this.updatedAt, tick.timestamp);
      return;
    }
    const previousPrice = this.price;
    this.accrueMaintenance(tick.timestamp);
    this.price = tick.price;
    this.updatedAt = Math.max(this.updatedAt, tick.timestamp);
    if (this.assessLiquidation(previousPrice)) return;

    let liquidity = tick.quantity > 0 ? tick.quantity : Infinity;
    for (const item of this.fillPriority(tick.price)) {
      if (item.order.status === "pending" && stopTriggered(item.order, tick.price)) {
        item.triggered = true;
        item.order.status = "open";
        this.events.push({ type: "open", order: structuredClone(item.order) });
      }
      if (item.triggered && canFill(item.order, tick.price) && liquidity > 0) {
        liquidity -= this.fill(item, executionPrice(item.order, tick.price), liquidity);
        if (this.assessLiquidation(this.price)) break;
      }
    }
    this.observeRisk();
  }

  drainEvents(): TradingOrderEvent[] {
    const events = this.events;
    this.events = [];
    return events;
  }

  status(): SimulatedTradingStatus {
    const closeoutEquity = this.closeoutEquity(this.price);
    return {
      liquidated: this.isLiquidated(),
      liquidationCount: this.liquidationCountValue,
      liquidatedAt: this.liquidatedAtValue,
      liquidationPrice: this.liquidationPriceValue,
      liquidationReason: this.liquidationReasonValue,
      equity: this.markedEquity(this.price),
      closeoutEquity,
      effectiveLeverage: this.effectiveLeverage(this.price),
      maxEffectiveLeverage: this.maxEffectiveLeverageValue,
      feesPaid: this.feesPaidValue,
      maintenancePaid: this.maintenancePaidValue,
    };
  }

  snapshot(): SimulatedTradingSnapshot {
    return {
      version: 1,
      quote: this.quote,
      asset: this.asset,
      price: this.price,
      updatedAt: this.updatedAt,
      orders: [...this.orders.values()].map((item) => structuredClone(item)),
      feesPaid: this.feesPaidValue,
      maintenancePaid: this.maintenancePaidValue,
      liquidationCount: this.liquidationCountValue,
      liquidatedAt: this.liquidatedAtValue,
      liquidationPrice: this.liquidationPriceValue,
      liquidationReason: this.liquidationReasonValue,
      maxEffectiveLeverage: this.maxEffectiveLeverageValue,
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
      leverage: input.leverage,
      reduceOnly: input.reduceOnly,
    };
    if (this.isLiquidated() || !this.valid(order)) {
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
      this.assessLiquidation(this.price);
    }
    return { accepted: true, order: structuredClone(order) };
  }

  private valid(order: TradingOrderSnapshot): boolean {
    if (!Number.isFinite(order.size) || order.size <= 0) return false;
    const rules = order.type === "market"
      ? this.options.rules.marketQuantity
      : this.options.rules.limitQuantity;
    if ((rules.min !== null && order.size < rules.min)
      || (rules.max !== null && order.size > rules.max)) return false;
    const price = order.price ?? order.stopPrice ?? this.price;
    const notional = order.size * price;
    return price > 0
      && (this.options.rules.minNotional === null || notional >= this.options.rules.minNotional)
      && (this.options.rules.maxNotional === null || notional <= this.options.rules.maxNotional);
  }

  private fill(item: SimulatedOrder, price: number, liquidity: number): number {
    const { order } = item;
    const remaining = Math.max(0, order.size - item.filled);
    let quantity = Math.min(remaining, liquidity);
    if (order.reduceOnly) {
      quantity = Math.min(quantity, order.side === "buy"
        ? Math.max(0, -this.asset)
        : Math.max(0, this.asset));
    } else {
      const reducing = order.side === "buy"
        ? Math.max(0, Math.min(quantity, -this.asset))
        : Math.max(0, Math.min(quantity, this.asset));
      const requestedLeverage = Math.max(
        1,
        Math.min(order.leverage ?? this.options.rules.maxLeverage, this.options.rules.maxLeverage),
      );
      const entryCapacity = this.entryCapacity(order.side, price, requestedLeverage, false);
      quantity = Math.min(quantity, reducing + entryCapacity / price);
    }
    if (!(quantity > EPSILON)) return 0;

    const grossQuote = quantity * price;
    const fee = grossQuote * this.friction();
    const filledQuote = order.side === "buy" ? grossQuote + fee : grossQuote - fee;
    if (order.side === "buy") {
      this.quote -= filledQuote;
      this.asset += quantity;
    } else {
      this.quote += filledQuote;
      this.asset -= quantity;
    }
    this.feesPaidValue += fee;
    item.filled += quantity;
    const orderRemaining = Math.max(0, order.size - item.filled);
    order.status = orderRemaining > EPSILON ? "partially-filled" : "filled";
    if (order.status === "filled") this.orders.delete(order.id);
    this.events.push({
      type: order.status === "filled" ? "fill" : "partial-fill",
      orderId: order.id,
      fill: {
        filledAsset: quantity,
        filledQuote,
        price,
        feeQuote: fee,
        remaining: orderRemaining,
      },
    });
    return quantity;
  }

  private accrueMaintenance(timestamp: number): void {
    if (this.updatedAt <= 0 || timestamp <= this.updatedAt) return;
    const elapsedMs = timestamp - this.updatedAt;
    const quoteGrowth = hourlyGrowth(this.options.quoteBorrowBpsHour ?? 0, elapsedMs);
    const assetGrowth = hourlyGrowth(this.options.assetBorrowBpsHour ?? 0, elapsedMs);
    const quoteCharge = this.quote < 0 ? -this.quote * quoteGrowth : 0;
    const assetCharge = this.asset < 0 ? -this.asset * assetGrowth : 0;
    if (!(quoteCharge > 0) && !(assetCharge > 0)) return;
    this.quote -= quoteCharge;
    this.asset -= assetCharge;
    this.maintenancePaidValue += quoteCharge + assetCharge * Math.max(0, this.price);
    this.events.push({ type: "maintenance", elapsedMs, quoteCharge, assetCharge });
  }

  private assessLiquidation(previousPrice: number): boolean {
    if (this.isLiquidated() || Math.abs(this.asset) <= EPSILON || !(this.price > 0)) return false;
    const boundary = this.liquidationBoundaryPrice();
    const crossed = boundary !== null && previousPrice > 0 && (
      this.asset > 0
        ? previousPrice > boundary && this.price <= boundary
        : previousPrice < boundary && this.price >= boundary
    );
    const closeoutEquity = this.closeoutEquity(this.price);
    const effectiveLeverage = this.effectiveLeverage(this.price);
    const maximum = this.maximumEffectiveLeverage();
    if (crossed) {
      const boundaryEffective = this.effectiveLeverage(boundary!);
      this.liquidate(
        boundary!,
        Number.isFinite(maximum) && boundaryEffective >= maximum - 1e-6
          ? "effective-leverage"
          : "insolvent",
      );
      return true;
    }
    if (!(closeoutEquity > 0) || !Number.isFinite(closeoutEquity)) {
      this.liquidate(this.price, "insolvent");
      return true;
    }
    if (Number.isFinite(maximum) && effectiveLeverage > maximum + 1e-6) {
      this.liquidate(this.price, "effective-leverage");
      return true;
    }
    this.observeRisk();
    return false;
  }

  private liquidate(price: number, reason: "insolvent" | "effective-leverage"): void {
    const boundaryLeverage = this.effectiveLeverage(price);
    if (Number.isFinite(boundaryLeverage)) {
      this.maxEffectiveLeverageValue = Math.max(
        this.maxEffectiveLeverageValue,
        boundaryLeverage,
      );
    }
    const assetValue = this.asset * price;
    const closeout = this.closeoutEquity(price);
    const closeFee = Math.abs(assetValue) * this.friction();
    this.feesPaidValue += closeFee;
    this.quote = Math.max(0, Number.isFinite(closeout) ? closeout : 0);
    this.asset = 0;
    this.price = price;
    this.orders.clear();
    this.liquidationCountValue += 1;
    this.liquidatedAtValue = this.updatedAt;
    this.liquidationPriceValue = price;
    this.liquidationReasonValue = reason;
    this.events.push({
      type: "liquidation",
      at: this.updatedAt,
      price,
      equity: this.quote,
      reason,
    });
  }

  private entryCapacity(
    side: "buy" | "sell",
    price: number,
    leverage: number,
    includePending: boolean,
  ): number {
    if (this.isLiquidated() || !(price > 0)) return 0;
    let quote = this.quote;
    let asset = this.asset;
    if (includePending) {
      for (const { order, filled } of this.orders.values()) {
        if (order.reduceOnly || order.status === "pending") continue;
        const remaining = Math.max(0, order.size - filled);
        const orderPrice = order.price ?? price;
        ({ quote, asset } = projectFill(quote, asset, order.side, remaining * orderPrice, orderPrice, this.friction()));
      }
    }
    const valid = (candidate: number) => {
      const projected = projectFill(quote, asset, side, candidate, price, this.friction());
      const closeout = closeoutEquity(projected.quote, projected.asset, price, this.friction());
      if (!(closeout > 0)) return false;
      const targetExposure = markedEffectiveLeverage(projected.quote, projected.asset, price);
      const liquidationExposure = effectiveLeverage(
        projected.quote,
        projected.asset,
        price,
        this.friction(),
      );
      return targetExposure <= leverage + 1e-10
        && liquidationExposure <= this.maximumEffectiveLeverage() + 1e-10;
    };
    let high = Math.max(1, leverage * Math.max(0, closeoutEquity(quote, asset, price, this.friction()))
      + Math.abs(asset * price));
    if (this.options.rules.maxNotional !== null) high = Math.min(high, this.options.rules.maxNotional);
    if (valid(high) && this.options.rules.maxNotional === null) {
      for (let index = 0; index < 12 && valid(high); index += 1) high *= 2;
    }
    if (valid(high)) return high;
    let low = 0;
    for (let index = 0; index < 64; index += 1) {
      const middle = (low + high) / 2;
      if (valid(middle)) low = middle;
      else high = middle;
    }
    return low;
  }

  private liquidationBoundaryPrice(): number | null {
    if (Math.abs(this.asset) <= EPSILON) return null;
    const fee = this.friction();
    const maximum = this.maximumEffectiveLeverage();
    if (this.asset > 0 && this.quote < 0) {
      const insolvency = -this.quote / (this.asset * Math.max(EPSILON, 1 - fee));
      if (!Number.isFinite(maximum)) return insolvency;
      const effective = maximum * this.quote / (1 - maximum);
      return Math.max(0, effective / (this.asset * Math.max(EPSILON, 1 - fee)));
    }
    if (this.asset < 0 && this.quote > 0) {
      const insolvency = -this.quote / (this.asset * (1 + fee));
      if (!Number.isFinite(maximum)) return insolvency;
      const effective = -maximum * this.quote / (1 + maximum);
      return Math.max(0, effective / (this.asset * (1 + fee)));
    }
    return null;
  }

  private reservedBalances(): { quote: number; asset: number } {
    return [...this.orders.values()].reduce((total, { order, filled }) => {
      const remaining = Math.max(0, order.size - filled);
      if (order.side === "buy") {
        total.quote += remaining * (order.price ?? order.stopPrice ?? this.price) * (1 + this.friction());
      } else {
        total.asset += Math.min(Math.max(0, this.asset), remaining);
        total.quote += Math.max(0, remaining - Math.max(0, this.asset))
          * (order.price ?? order.stopPrice ?? this.price)
          / Math.max(1, order.leverage ?? this.options.rules.maxLeverage);
      }
      return total;
    }, { quote: 0, asset: 0 });
  }

  private fillPriority(price: number): SimulatedOrder[] {
    return [...this.orders.values()].sort((left, right) => {
      const leftExecutable = executionPrice(left.order, price);
      const rightExecutable = executionPrice(right.order, price);
      if (left.order.side === right.order.side) {
        return left.order.side === "buy"
          ? rightExecutable - leftExecutable
          : leftExecutable - rightExecutable;
      }
      return left.order.side.localeCompare(right.order.side);
    });
  }

  private markedEquity(price: number): number {
    return this.quote + this.asset * Math.max(0, price);
  }

  private closeoutEquity(price: number): number {
    return closeoutEquity(this.quote, this.asset, price, this.friction());
  }

  private effectiveLeverage(price: number): number {
    return effectiveLeverage(this.quote, this.asset, price, this.friction());
  }

  private observeRisk(): void {
    const effective = this.effectiveLeverage(this.price);
    if (Number.isFinite(effective)) {
      this.maxEffectiveLeverageValue = Math.max(this.maxEffectiveLeverageValue, effective);
    }
  }

  private maximumEffectiveLeverage(): number {
    const maximum = this.options.maxEffectiveLeverage ?? Number.POSITIVE_INFINITY;
    return maximum > 0 ? maximum : Number.POSITIVE_INFINITY;
  }

  private friction(): number {
    return Math.max(0, Math.min(0.999999, this.options.friction));
  }
}

function projectFill(
  quote: number,
  asset: number,
  side: "buy" | "sell",
  grossQuote: number,
  price: number,
  friction: number,
): { quote: number; asset: number } {
  return side === "buy"
    ? { quote: quote - grossQuote * (1 + friction), asset: asset + grossQuote / price }
    : { quote: quote + grossQuote * (1 - friction), asset: asset - grossQuote / price };
}

function closeoutEquity(
  quote: number,
  asset: number,
  price: number,
  friction: number,
): number {
  const assetValue = asset * Math.max(0, price);
  const liquidatedAssetValue = assetValue >= 0
    ? assetValue * (1 - friction)
    : assetValue * (1 + friction);
  return quote + liquidatedAssetValue;
}

function effectiveLeverage(
  quote: number,
  asset: number,
  price: number,
  friction: number,
): number {
  const equity = closeoutEquity(quote, asset, price, friction);
  if (Math.abs(asset) <= EPSILON) return 0;
  if (!(equity > 0)) return Number.POSITIVE_INFINITY;
  const assetValue = asset * price;
  const liquidatedAssetValue = assetValue >= 0
    ? assetValue * (1 - friction)
    : assetValue * (1 + friction);
  return Math.abs(liquidatedAssetValue / equity);
}

function markedEffectiveLeverage(quote: number, asset: number, price: number): number {
  const equity = quote + asset * price;
  if (Math.abs(asset) <= EPSILON) return 0;
  return equity > 0 ? Math.abs(asset * price / equity) : Number.POSITIVE_INFINITY;
}

function hourlyGrowth(bpsHour: number, elapsedMs: number): number {
  const hourly = Math.max(0, bpsHour) / 10_000;
  return hourly > 0 ? Math.expm1(Math.log1p(hourly) * elapsedMs / HOUR_MS) : 0;
}

function stopTriggered(order: TradingOrderSnapshot, price: number): boolean {
  const stop = order.stopPrice ?? 0;
  return order.side === "buy" ? price >= stop : price <= stop;
}

function canFill(order: TradingOrderSnapshot, price: number): boolean {
  if (order.type === "market" || order.type === "stop-market") return true;
  const limit = order.price ?? 0;
  return order.side === "buy" ? price <= limit : price >= limit;
}

function executionPrice(order: TradingOrderSnapshot, price: number): number {
  return order.type === "market" || order.type === "stop-market"
    ? price
    : order.price ?? price;
}
