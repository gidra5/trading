import type {
  TradingApi,
  TradingOrderEvent,
  TradingOrderSnapshot,
  TradingTick,
} from "./trading-api.js";
import type {
  PositionSide,
  StrategyDiagnostics,
  StrategySnapshot,
  TradingStrategy,
} from "./strategy.js";

export type GridSizeDistribution = "linear" | "geometric";

export interface GridConfig {
  orderCount: number;
  maxPriceStep: number;
  sizeDistribution: GridSizeDistribution;
  /** Used by geometric distribution. */
  sizeFraction: number;
}

export interface ExitGridConfig extends GridConfig {
  reset: "previous-anchor" | "last-filled-order";
}

export interface TradingBotConfig<TStrategyConfig = unknown> {
  strategy: TStrategyConfig;
  maxTargetLeverage: number;
  minTradeQuote: number;
  maxTradeQuote: number;
  entryGrid: GridConfig;
  exitGrid: ExitGridConfig;
  positionLifetimeMs: number | null;
  stopLossRate: number | null;
  takeProfitRate: number | null;
  cooldownMs: number;
  internalBorrow: {
    enabled: boolean;
    /** Locks exactly the borrowed asset and quote amounts when enabled. */
    lockLenderAmounts: boolean;
    borrowerProfitShare: number;
  };
}

export interface PositionBorrow {
  positionId: string;
  asset: number;
  quote: number;
}

export interface PositionGridOrder {
  order: TradingOrderSnapshot;
  /** Unit is implied by position side and whether this is an entry or exit grid. */
  filled: number;
}

export interface PositionGrid {
  orders: PositionGridOrder[];
  creationPrice: number;
}

/** Created with its entry grid; asset and quote remain zero until the first fill. */
export interface TradingPosition {
  id: string;
  side: PositionSide;
  asset: number;
  quote: number;
  leverage: number;
  internalBorrow: PositionBorrow[];
  externalBorrow: {
    asset: number;
    quote: number;
  };
  entryGrid: PositionGrid | null;
  exitGrid: PositionGrid | null;
  stopLossPrice: number | null;
  takeProfitPrice: number | null;
  expiresAt: number | null;
}

export interface BotSnapshot<
  TStrategyConfig = unknown,
  TStrategySnapshot extends StrategySnapshot = StrategySnapshot,
> {
  version: number;
  config: TradingBotConfig<TStrategyConfig>;
  strategy: TStrategySnapshot;
  positions: TradingPosition[];
  lastEntryAt: number;
}

export interface BotMetricsSnapshot {
  positionCount: number;
  orderCount: number;
  longAsset: number;
  shortAsset: number;
}

export interface BotEntryRiskReport {
  side: PositionSide;
  size: number;
  leverage: number;
  blocker: string | null;
}

export interface BotDiagnostics<TDiagnostics extends StrategyDiagnostics = StrategyDiagnostics> {
  strategy: TDiagnostics;
  positions: readonly TradingPosition[];
  /** Derived by flattening position entry/exit grids. */
  plannedOrders: readonly TradingOrderSnapshot[];
  saturated: boolean;
}

export interface BotOptions<
  TStrategyConfig = unknown,
  TStrategySnapshot extends StrategySnapshot = StrategySnapshot,
  TDiagnostics extends StrategyDiagnostics = StrategyDiagnostics,
> {
  api: TradingApi;
  strategy: TradingStrategy<TStrategyConfig, TStrategySnapshot, TDiagnostics>;
  config: TradingBotConfig<TStrategyConfig>;
  onEntryRisk?: (report: BotEntryRiskReport) => void;
}

export interface TradingBot<
  TStrategyConfig = unknown,
  TStrategySnapshot extends StrategySnapshot = StrategySnapshot,
  TDiagnostics extends StrategyDiagnostics = StrategyDiagnostics,
> {
  warmup(): Promise<void>;
  onTick(tick: TradingTick): Promise<void>;
  onOrder(event: TradingOrderEvent): Promise<void>;
  snapshot(): Promise<BotSnapshot<TStrategyConfig, TStrategySnapshot>>;
  restore(
    snapshot: BotSnapshot<TStrategyConfig, TStrategySnapshot>,
    options?: { restoreStrategy?: boolean },
  ): Promise<void>;
  getMetrics(): BotMetricsSnapshot;
  getDiagnostics(): BotDiagnostics<TDiagnostics>;
  updateConfig(config: TradingBotConfig<TStrategyConfig>): Promise<void>;
}

export class GridTradingBot<
  TStrategyConfig = unknown,
  TStrategySnapshot extends StrategySnapshot = StrategySnapshot,
  TDiagnostics extends StrategyDiagnostics = StrategyDiagnostics,
> implements TradingBot<TStrategyConfig, TStrategySnapshot, TDiagnostics> {
  private config: TradingBotConfig<TStrategyConfig>;
  private positions: TradingPosition[] = [];
  private lastTickAt = 0;
  private lastEntryAt = 0;

  constructor(private readonly options: BotOptions<TStrategyConfig, TStrategySnapshot, TDiagnostics>) {
    this.config = options.config;
  }

  async warmup(): Promise<void> {
    await this.options.strategy.warmup();
  }

  async onTick(tick: TradingTick): Promise<void> {
    this.lastTickAt = tick.timestamp;
    await this.options.strategy.onTick(tick);
    await this.applyLifecycle(tick);
    const exit = await this.options.strategy.exitSignal();
    if (exit) {
      await this.createExit(exit.side, exit.size, exit.price, exit.confidence);
    }
    const entry = await this.options.strategy.entrySignal();
    if (entry && tick.timestamp - this.lastEntryAt >= this.config.cooldownMs) {
      await this.createEntry(
        entry.side,
        entry.size,
        entry.leverage,
        entry.price,
        entry.confidence,
        tick,
      );
    } else if (entry) {
      this.reportEntryRisk({
        side: entry.side,
        size: entry.size,
        leverage: entry.leverage,
        blocker: "cooldown",
      });
    }
  }

  async onOrder(event: TradingOrderEvent): Promise<void> {
    const found = this.findOrder(event.type === "open" ? event.order.id : event.orderId);
    if (!found) {
      return;
    }
    const { position, grid, entry, item } = found;
    if (event.type === "open") {
      item.order = event.order;
      return;
    }
    if (event.type === "rejected") {
      grid.orders.splice(grid.orders.indexOf(item), 1);
      this.removeEmptyGrid(position, grid, entry);
      return;
    }

    item.order.status = event.type === "fill" ? "filled" : "partially-filled";
    if (grid.creationPrice <= 0 && event.fill.filledAsset > 0) {
      grid.creationPrice = event.fill.filledQuote / event.fill.filledAsset;
    }
    item.filled += filledGridAmount(position.side, entry, event.fill);
    if (entry) {
      position.asset += event.fill.filledAsset;
      position.quote += event.fill.filledQuote;
      this.updateBorrow(position, event.fill);
      this.setLifecyclePrices(position);
    } else {
      this.reducePositionAmount(position, "asset", event.fill.filledAsset);
      this.reducePositionAmount(position, "quote", event.fill.filledQuote);
    }

    if (position.asset <= Number.EPSILON) {
      await this.removePosition(position);
    } else if (
      !entry &&
      event.type === "fill" &&
      this.config.exitGrid.reset === "last-filled-order"
    ) {
      await this.replaceExitGrid(
        position,
        position.asset,
        event.fill.filledQuote / event.fill.filledAsset,
      );
    }
  }

  async snapshot(): Promise<BotSnapshot<TStrategyConfig, TStrategySnapshot>> {
    return {
      version: 2,
      config: structuredClone(this.config),
      strategy: await this.options.strategy.snapshot(),
      positions: structuredClone(this.positions),
      lastEntryAt: this.lastEntryAt,
    };
  }

  async restore(
    snapshot: BotSnapshot<TStrategyConfig, TStrategySnapshot>,
    options: { restoreStrategy?: boolean } = {},
  ): Promise<void> {
    if (snapshot.version !== 2) {
      throw new Error(`Unsupported bot snapshot version: ${snapshot.version}`);
    }
    this.config = structuredClone(snapshot.config);
    this.positions = structuredClone(snapshot.positions);
    this.lastEntryAt = snapshot.lastEntryAt ?? 0;
    if (options.restoreStrategy === false) {
      await this.options.strategy.updateConfig(this.config.strategy);
    } else {
      await this.options.strategy.restore(snapshot.strategy);
    }
  }

  getMetrics(): BotMetricsSnapshot {
    return {
      positionCount: this.positions.length,
      orderCount: plannedOrdersOf(this.positions).length,
      longAsset: sumSide(this.positions, "long"),
      shortAsset: sumSide(this.positions, "short"),
    };
  }

  getDiagnostics(): BotDiagnostics<TDiagnostics> {
    return {
      strategy: this.options.strategy.getDiagnostics(),
      positions: this.positions,
      plannedOrders: plannedOrdersOf(this.positions),
      saturated: Boolean(
        this.lastTickAt && this.lastTickAt - this.lastEntryAt < this.config.cooldownMs
      ),
    };
  }

  async updateConfig(config: TradingBotConfig<TStrategyConfig>): Promise<void> {
    this.config = structuredClone(config);
    await this.options.strategy.updateConfig(config.strategy);
  }

  async closePositions(): Promise<void> {
    for (const position of this.positions) {
      const quantity = this.closableAsset(position);
      if (quantity > 0) {
        await this.replaceExitGrid(position, quantity, null);
      }
    }
  }

  async cancelOpenOrders(): Promise<void> {
    for (const position of this.positions) {
      if (position.entryGrid) {
        await this.cancelGrid(position.entryGrid);
        position.entryGrid = position.asset > 0 ? null : position.entryGrid;
      }
      if (position.exitGrid) {
        await this.cancelGrid(position.exitGrid);
        position.exitGrid = null;
      }
    }
    this.positions = this.positions.filter((position) => position.asset > 0);
  }

  async openPosition(
    side: PositionSide,
    quantity: number,
    lifecycle: {
      lifetimeMs?: number;
      stopLossPrice?: number;
      takeProfitPrice?: number;
    } = {},
  ): Promise<void> {
    if (quantity <= 0) {
      throw new Error("A positive quantity is required.");
    }
    const entryGrid: PositionGrid = { orders: [], creationPrice: 0 };
    const position: TradingPosition = {
      id: crypto.randomUUID(),
      side,
      asset: 0,
      quote: 0,
      leverage: 1,
      internalBorrow: [],
      externalBorrow: { asset: 0, quote: 0 },
      entryGrid,
      exitGrid: null,
      stopLossPrice: lifecycle.stopLossPrice ?? null,
      takeProfitPrice: lifecycle.takeProfitPrice ?? null,
      expiresAt: lifecycle.lifetimeMs
        ? Date.now() + lifecycle.lifetimeMs
        : null,
    };
    this.positions.push(position);
    await this.placeGrid(
      position,
      entryGrid,
      side === "long" ? "buy" : "sell",
      quantity,
      null,
      singleOrderGrid(),
      null,
    );
  }

  async closePosition(
    side: PositionSide,
    quantity: number,
    positionId?: string,
  ): Promise<void> {
    let remaining = quantity;
    for (const position of this.positions) {
      if (
        remaining <= 0 ||
        position.side !== side ||
        (positionId && position.id !== positionId)
      ) {
        continue;
      }
      const close = Math.min(remaining, this.closableAsset(position));
      if (close > 0) {
        await this.replaceExitGrid(position, close, null);
        remaining -= close;
      }
    }
  }

  private async createEntry(
    side: PositionSide,
    size: number,
    requestedLeverage: number,
    signalPrice: number | null,
    confidence: number | null,
    tick: TradingTick,
  ): Promise<void> {
    const currentPrice = tick.price;
    if (currentPrice <= 0 || size <= 0) {
      this.reportEntryRisk({
        side,
        size,
        leverage: requestedLeverage,
        blocker: currentPrice <= 0 ? "price" : "size",
      });
      return;
    }
    const rules = await this.options.api.getMarketRules();
    const requested = Math.max(
      1,
      Math.min(requestedLeverage, this.config.maxTargetLeverage, rules.maxLeverage),
    );
    const capacity = await this.options.api.getOrderCapacity({
      side: side === "long" ? "buy" : "sell",
      price: currentPrice,
      leverage: requested,
    });
    const leverage = Math.max(1, Math.min(requested, capacity.leverage));
    const count = signalPrice === null ? 1 : Math.max(1, Math.round(this.config.entryGrid.orderCount));
    const quantityRules = signalPrice === null ? rules.marketQuantity : rules.limitQuantity;
    const quantityCap = quantityRules.max === null ? Infinity : quantityRules.max * currentPrice;
    const notionalCap = rules.maxNotional ?? Infinity;
    const providerCapacity = Math.max(
      0,
      Math.min(capacity.quote, quantityCap * count, notionalCap * count),
    );
    const minimum = Math.max(this.config.minTradeQuote, rules.minNotional ?? 0);
    let desiredQuote = providerCapacity * clamp01(size);
    const remainingCapacity = providerCapacity - desiredQuote;
    if (remainingCapacity > 0 && remainingCapacity < minimum) {
      desiredQuote = providerCapacity;
    }
    const quote = Math.min(this.config.maxTradeQuote, desiredQuote);
    this.reportEntryRisk({
      side,
      size: quote,
      leverage,
      blocker: quote < minimum ? "min-trade" : null,
    });
    if (quote < minimum) {
      return;
    }

    const entryGrid: PositionGrid = { orders: [], creationPrice: currentPrice };
    const position: TradingPosition = {
      id: crypto.randomUUID(),
      side,
      asset: 0,
      quote: 0,
      leverage,
      internalBorrow: [],
      externalBorrow: { asset: 0, quote: 0 },
      entryGrid,
      exitGrid: null,
      stopLossPrice: null,
      takeProfitPrice: null,
      expiresAt: this.config.positionLifetimeMs === null
        ? null
        : tick.timestamp + this.config.positionLifetimeMs,
    };
    this.positions.push(position);
    await this.placeGrid(
      position,
      entryGrid,
      side === "long" ? "buy" : "sell",
      quote / currentPrice,
      signalPrice,
      this.config.entryGrid,
      confidence,
      quote,
    );
    if (entryGrid.orders.length === 0) {
      this.positions.splice(this.positions.indexOf(position), 1);
      this.reportEntryRisk({ side, size: quote, leverage, blocker: "provider" });
      return;
    }
    this.lastEntryAt = tick.timestamp;
  }

  private async createExit(
    side: PositionSide,
    size: number,
    price: number | null,
    confidence: number | null,
  ): Promise<void> {
    const positions = this.positions.filter((position) => position.side === side && position.asset > 0);
    const total = positions.reduce((sum, position) => sum + this.closableAsset(position), 0);
    let remaining = total * clamp01(size);
    for (const position of positions) {
      if (remaining <= 0) {
        break;
      }
      const quantity = Math.min(this.closableAsset(position), remaining);
      if (quantity <= 0) {
        continue;
      }
      remaining -= quantity;
      await this.replaceExitGrid(position, quantity, price, confidence);
    }
  }

  private async replaceExitGrid(
    position: TradingPosition,
    quantity: number,
    signalPrice: number | null,
    confidence: number | null = null,
  ): Promise<void> {
    if (position.exitGrid) {
      await this.cancelGrid(position.exitGrid);
    }
    const currentPrice = signalPrice ?? 0;
    const grid: PositionGrid = { orders: [], creationPrice: currentPrice };
    position.exitGrid = grid;
    await this.placeGrid(
      position,
      grid,
      position.side === "long" ? "sell" : "buy",
      quantity,
      signalPrice,
      this.config.exitGrid,
      confidence,
    );
    if (grid.orders.length === 0) {
      position.exitGrid = null;
    }
  }

  private async placeGrid(
    position: TradingPosition,
    grid: PositionGrid,
    side: "buy" | "sell",
    quantity: number,
    signalPrice: number | null,
    config: GridConfig,
    confidence: number | null = null,
    quoteLimit: number | null = null,
  ): Promise<void> {
    const market = signalPrice === null;
    const count = market ? 1 : Math.max(1, Math.round(config.orderCount));
    const weights = gridWeights(count, config, confidence);
    const rules = await this.options.api.getMarketRules();
    const quantityRules = market ? rules.marketQuantity : rules.limitQuantity;
    const prices = weights.map((_, index) => signalPrice === null
      ? null
      : normalizePrice(
          gridPrice(signalPrice, side, index, config.maxPriceStep),
          rules.price,
        ));
    const sizes = weights.map((weight) => quantity * weight);
    const totalQuote = sizes.reduce(
      (sum, size, index) => sum + size * (prices[index] ?? grid.creationPrice),
      0,
    );
    if (quoteLimit !== null && totalQuote > quoteLimit) {
      const scale = quoteLimit / totalQuote;
      for (let index = 0; index < sizes.length; index += 1) sizes[index] *= scale;
    }
    let carry = 0;
    for (let index = sizes.length - 1; index >= 0; index -= 1) {
      sizes[index] += carry;
      carry = 0;
      const notional = sizes[index] * (prices[index] ?? grid.creationPrice);
      if (index > 0 && rules.minNotional !== null && notional < rules.minNotional) {
        carry = sizes[index];
        sizes[index] = 0;
      }
    }
    for (let index = 0; index < count; index += 1) {
      const size = roundDown(sizes[index], quantityRules.step);
      if (
        size <= 0 ||
        (quantityRules.min !== null && size < quantityRules.min) ||
        (quantityRules.max !== null && size > quantityRules.max)
      ) {
        continue;
      }
      const price = prices[index];
      const result = price === null
        ? await this.options.api.createMarketOrder({ side, size })
        : await this.options.api.createLimitOrder({ side, size, price });
      if (result.accepted) {
        grid.orders.push({ order: result.order, filled: 0 });
      }
    }
    if (grid.orders.length === 0 && position.asset <= 0) {
      this.positions.splice(this.positions.indexOf(position), 1);
    }
  }

  private async applyLifecycle(tick: TradingTick): Promise<void> {
    for (const position of [...this.positions]) {
      if (position.asset <= 0) {
        continue;
      }
      if (position.exitGrid) {
        const reset = this.config.exitGrid.reset === "previous-anchor" && (
          position.side === "long"
            ? tick.price > position.exitGrid.creationPrice * (1 + this.config.exitGrid.maxPriceStep)
            : tick.price < position.exitGrid.creationPrice * (1 - this.config.exitGrid.maxPriceStep)
        );
        if (reset) {
          await this.replaceExitGrid(position, position.asset, tick.price);
        }
        continue;
      }
      const stop = position.stopLossPrice !== null && (
        position.side === "long" ? tick.price <= position.stopLossPrice : tick.price >= position.stopLossPrice
      );
      const take = position.takeProfitPrice !== null && (
        position.side === "long" ? tick.price >= position.takeProfitPrice : tick.price <= position.takeProfitPrice
      );
      if (stop || take || (position.expiresAt !== null && tick.timestamp >= position.expiresAt)) {
        await this.replaceExitGrid(position, position.asset, null);
      }
    }
  }

  private setLifecyclePrices(position: TradingPosition): void {
    const entryPrice = position.asset > 0 ? position.quote / position.asset : 0;
    if (entryPrice <= 0) {
      return;
    }
    if (this.config.stopLossRate !== null && position.stopLossPrice === null) {
      position.stopLossPrice = entryPrice
        * (position.side === "long" ? 1 - this.config.stopLossRate : 1 + this.config.stopLossRate);
    }
    if (this.config.takeProfitRate !== null && position.takeProfitPrice === null) {
      position.takeProfitPrice = entryPrice
        * (position.side === "long" ? 1 + this.config.takeProfitRate : 1 - this.config.takeProfitRate);
    }
  }

  private updateBorrow(position: TradingPosition, fill: { filledAsset: number; filledQuote: number }): void {
    const borrowedRate = Math.max(0, 1 - 1 / position.leverage);
    if (position.side === "long") {
      const borrowed = fill.filledQuote * borrowedRate;
      position.externalBorrow.quote += borrowed - this.allocateInternal(position, 0, borrowed);
    } else {
      const borrowed = fill.filledAsset * borrowedRate;
      position.externalBorrow.asset += borrowed - this.allocateInternal(position, borrowed, 0);
    }
  }

  private reducePositionAmount(
    position: TradingPosition,
    unit: "asset" | "quote",
    amount: number,
  ): void {
    position[unit] -= amount;
    let remaining = Math.max(0, amount);
    const external = Math.min(remaining, position.externalBorrow[unit]);
    position.externalBorrow[unit] -= external;
    remaining -= external;
    for (const borrow of position.internalBorrow) {
      const internal = Math.min(remaining, borrow[unit]);
      borrow[unit] -= internal;
      remaining -= internal;
      if (remaining <= 0) break;
    }
    position.internalBorrow = position.internalBorrow.filter(
      (borrow) => borrow.asset > Number.EPSILON || borrow.quote > Number.EPSILON,
    );
  }

  private allocateInternal(
    borrower: TradingPosition,
    asset: number,
    quote: number,
  ): number {
    if (!this.config.internalBorrow.enabled) {
      return 0;
    }
    let remaining = asset || quote;
    for (const lender of this.positions) {
      if (lender.id === borrower.id || lender.side === borrower.side || remaining <= 0) {
        continue;
      }
      const lent = this.lentAmounts(lender.id);
      const available = asset > 0
        ? Math.max(0, lender.asset - lent.asset)
        : Math.max(0, lender.quote - lent.quote);
      const amount = Math.min(remaining, available);
      if (amount <= 0) {
        continue;
      }
      const existing = borrower.internalBorrow.find((borrow) => borrow.positionId === lender.id);
      const borrow = existing ?? { positionId: lender.id, asset: 0, quote: 0 };
      borrow.asset += asset > 0 ? amount : 0;
      borrow.quote += quote > 0 ? amount : 0;
      if (!existing) {
        borrower.internalBorrow.push(borrow);
      }
      remaining -= amount;
    }
    return (asset || quote) - remaining;
  }

  private closableAsset(position: TradingPosition): number {
    if (!this.config.internalBorrow.lockLenderAmounts) {
      return position.asset;
    }
    const lent = this.lentAmounts(position.id);
    if (position.side === "long") {
      return Math.max(0, position.asset - lent.asset);
    }
    const availableQuoteRate = position.quote > 0
      ? Math.max(0, position.quote - lent.quote) / position.quote
      : 1;
    return position.asset * Math.min(1, availableQuoteRate);
  }

  private lentAmounts(positionId: string): { asset: number; quote: number } {
    return this.positions.reduce((total, position) => {
      for (const borrow of position.internalBorrow) {
        if (borrow.positionId === positionId) {
          total.asset += borrow.asset;
          total.quote += borrow.quote;
        }
      }
      return total;
    }, { asset: 0, quote: 0 });
  }

  private findOrder(id: string) {
    for (const position of this.positions) {
      for (const [grid, entry] of [[position.entryGrid, true], [position.exitGrid, false]] as const) {
        const item = grid?.orders.find(({ order }) => order.id === id);
        if (grid && item) {
          return { position, grid, entry, item };
        }
      }
    }
  }

  private removeEmptyGrid(position: TradingPosition, grid: PositionGrid, entry: boolean): void {
    if (grid.orders.length > 0) {
      return;
    }
    if (entry && position.asset <= 0) {
      this.positions.splice(this.positions.indexOf(position), 1);
    } else if (entry) {
      position.entryGrid = null;
    } else {
      position.exitGrid = null;
    }
  }

  private async removePosition(position: TradingPosition): Promise<void> {
    await Promise.all([
      position.entryGrid && this.cancelGrid(position.entryGrid),
      position.exitGrid && this.cancelGrid(position.exitGrid),
    ]);
    this.positions.splice(this.positions.indexOf(position), 1);
    for (const borrower of this.positions) {
      borrower.internalBorrow = borrower.internalBorrow.filter(
        (borrow) => borrow.positionId !== position.id,
      );
    }
  }

  private async cancelGrid(grid: PositionGrid): Promise<void> {
    await Promise.all(grid.orders
      .filter(({ order }) => order.status !== "filled" && order.status !== "rejected")
      .map(({ order }) => this.options.api.cancelOrder(order.id)));
    grid.orders.length = 0;
  }

  private reportEntryRisk(report: BotEntryRiskReport): void {
    this.options.onEntryRisk?.(structuredClone(report));
  }
}

function filledGridAmount(
  side: PositionSide,
  entry: boolean,
  fill: { filledAsset: number; filledQuote: number },
): number {
  return side === "long" === entry ? fill.filledQuote : fill.filledAsset;
}

function ordersOf(positions: readonly TradingPosition[]): TradingOrderSnapshot[] {
  return positions.flatMap((position) => [
    ...(position.entryGrid?.orders ?? []),
    ...(position.exitGrid?.orders ?? []),
  ].map(({ order }) => order));
}

function plannedOrdersOf(positions: readonly TradingPosition[]): TradingOrderSnapshot[] {
  return ordersOf(positions).filter(
    (order) => order.status !== "filled" && order.status !== "rejected",
  );
}

function sumSide(positions: readonly TradingPosition[], side: PositionSide): number {
  return positions.reduce((sum, position) => sum + (position.side === side ? position.asset : 0), 0);
}

function gridWeights(
  count: number,
  config: GridConfig,
  confidence: number | null,
): number[] {
  const ratio = confidence === null
    ? config.sizeFraction
    : 1 - clamp01(confidence) * (1 - config.sizeFraction);
  const raw = Array.from({ length: count }, (_, index) => config.sizeDistribution === "linear"
    ? confidence === null ? count - index : 1 + clamp01(confidence) * (count - index - 1)
    : Math.max(Number.EPSILON, ratio) ** index);
  const total = raw.reduce((sum, weight) => sum + weight, 0);
  return raw.map((weight) => weight / total);
}

function gridPrice(anchor: number, side: "buy" | "sell", index: number, step: number): number {
  const direction = side === "buy" ? -1 : 1;
  return anchor * (1 + direction * Math.max(0, step) * index);
}

function roundDown(value: number, step: number | null): number {
  return step && step > 0 ? Math.floor(value / step) * step : value;
}

function normalizePrice(
  value: number,
  rules: { min: number | null; max: number | null; step: number | null },
): number {
  const rounded = roundDown(value, rules.step);
  return Math.max(rules.min ?? 0, Math.min(rules.max ?? Infinity, rounded));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function singleOrderGrid(): GridConfig {
  return {
    orderCount: 1,
    maxPriceStep: 0,
    sizeDistribution: "linear",
    sizeFraction: 1,
  };
}
