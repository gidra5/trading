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
  TradingStrategyEntrySignal,
  TradingStrategyTargetExposureSignal,
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
  exposureControl: {
    /** Minimum leverage fraction before distribution confidence, scaled by model confidence. */
    confidenceLeverageFloor: number;
    /** Minimum signal confidence required at full predictor confidence. */
    minimumSignalConfidence: number;
    /** Signal-confidence threshold approached as predictor confidence falls to zero. */
    maximumSignalConfidenceThreshold: number;
    /** Total same-side signal-confidence mass required before expansion. */
    expansionConfirmationMass: number;
    /** Fraction of max leverage forming the expansion-delta cap before quality scaling. */
    expansionDeltaCapFraction: number;
  };
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
  signalConfirmation: SignalConfirmation | null;
}

export interface SignalConfirmation {
  side: PositionSide;
  confirmationMass: number;
  minimumLeverage: number;
  maximumLeverage: number;
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

export interface BotDiagnostics<
  TDiagnostics extends StrategyDiagnostics = StrategyDiagnostics,
> {
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
  private signalConfirmation: SignalConfirmation | null =
    null;

  constructor(
    private readonly options: BotOptions<
      TStrategyConfig,
      TStrategySnapshot,
      TDiagnostics
    >,
  ) {
    this.config = options.config;
  }

  async warmup(): Promise<void> {
    await this.options.strategy.warmup();
  }

  async onTick(tick: TradingTick): Promise<void> {
    this.lastTickAt = tick.timestamp;
    await this.options.strategy.onTick(tick);
    await this.applyLifecycle(tick);
    // todo: should be decomposed and decoupled better
    if (this.options.strategy.targetExposureSignal) {
      const account = await this.options.api.getEquity();
      const equity =
        account.quoteUnleveraged + account.assetUnleveraged * tick.price;
      const currentExposure =
        equity > 0 ? this.signedPositionNotional(tick.price) / equity : 0;
      const target = await this.options.strategy.targetExposureSignal({
        timestamp: tick.timestamp,
        price: tick.price,
        equity,
        currentExposure,
        maxLeverage: this.config.maxTargetLeverage,
      });
      if (target) await this.rebalanceToTargetExposure(target, tick);
      return;
    }
    const exit = await this.options.strategy.exitSignal();
    if (exit) {
      this.cancelSignalConfirmation(exit.side);
      await this.createExit(exit.side, exit.size, exit.price, exit.confidence);
    }
    const entry = await this.options.strategy.entrySignal();
    if (entry && tick.timestamp - this.lastEntryAt >= this.config.cooldownMs) {
      const controlled = await this.controlEntry(entry, tick);
      if (controlled) {
        await this.createEntry(
          controlled.signal.side,
          controlled.signal.size,
          controlled.signal.leverage,
          controlled.signal.price,
          controlled.signal.confidence,
          tick,
          controlled.maximumQuote,
        );
      }
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
    if (event.type === "liquidation") {
      this.positions = [];
      this.signalConfirmation = null;
      return;
    }
    if (event.type === "maintenance") {
      this.applyMaintenance(event.quoteCharge, event.assetCharge);
      return;
    }
    const found = this.findOrder(
      event.type === "open" ? event.order.id : event.orderId,
    );
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
      version: 6,
      config: structuredClone(this.config),
      strategy: await this.options.strategy.snapshot(),
      positions: structuredClone(this.positions),
      lastEntryAt: this.lastEntryAt,
      signalConfirmation: structuredClone(this.signalConfirmation),
    };
  }

  async restore(
    snapshot: BotSnapshot<TStrategyConfig, TStrategySnapshot>,
    options: { restoreStrategy?: boolean } = {},
  ): Promise<void> {
    if (snapshot.version !== 6) {
      throw new Error(`Unsupported bot snapshot version: ${snapshot.version}`);
    }
    this.config = structuredClone(snapshot.config);
    this.positions = structuredClone(snapshot.positions);
    this.lastEntryAt = snapshot.lastEntryAt ?? 0;
    this.signalConfirmation = structuredClone(
      snapshot.signalConfirmation,
    );
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
        this.lastTickAt &&
        this.lastTickAt - this.lastEntryAt < this.config.cooldownMs,
      ),
    };
  }

  async updateConfig(config: TradingBotConfig<TStrategyConfig>): Promise<void> {
    this.config = structuredClone(config);
    this.signalConfirmation = null;
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
    maximumQuote = Number.POSITIVE_INFINITY,
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
      Math.min(
        requestedLeverage,
        this.config.maxTargetLeverage,
        rules.maxLeverage,
      ),
    );
    const capacity = await this.options.api.getOrderCapacity({
      side: side === "long" ? "buy" : "sell",
      price: currentPrice,
      leverage: requested,
    });
    const leverage = Math.max(1, Math.min(requested, capacity.leverage));
    const count =
      signalPrice === null
        ? 1
        : Math.max(1, Math.round(this.config.entryGrid.orderCount));
    const quantityRules =
      signalPrice === null ? rules.marketQuantity : rules.limitQuantity;
    const quantityCap =
      quantityRules.max === null ? Infinity : quantityRules.max * currentPrice;
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
    const quote = Math.min(this.config.maxTradeQuote, maximumQuote, desiredQuote);
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
      expiresAt:
        this.config.positionLifetimeMs === null
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
      this.reportEntryRisk({
        side,
        size: quote,
        leverage,
        blocker: "provider",
      });
      return;
    }
    this.lastEntryAt = tick.timestamp;
  }

  private async rebalanceToTargetExposure(
    signal: TradingStrategyTargetExposureSignal,
    tick: TradingTick,
  ): Promise<void> {
    if (!(tick.price > 0) || !Number.isFinite(signal.targetExposure)) return;
    const accountBeforeControl = await this.options.api.getEquity();
    const equityBeforeControl =
      accountBeforeControl.quoteUnleveraged +
      accountBeforeControl.assetUnleveraged * tick.price;
    if (!(equityBeforeControl > 0)) return;
    const currentExposure =
      this.signedPositionNotional(tick.price) / equityBeforeControl;
    const staticConfidence = this.options.strategy.staticConfidence();
    const controlled = this.controlTargetExposure(
      signal,
      currentExposure,
      staticConfidence,
    );
    if (!controlled) return;
    await this.cancelOpenOrders();
    const account = await this.options.api.getEquity();
    const equity =
      account.quoteUnleveraged + account.assetUnleveraged * tick.price;
    if (!(equity > 0)) return;

    const target = Math.max(
      -this.config.maxTargetLeverage,
      Math.min(this.config.maxTargetLeverage, controlled.targetExposure),
    );
    const targetSide: PositionSide | null =
      target > 0 ? "long" : target < 0 ? "short" : null;
    const oppositeSide: PositionSide | null =
      targetSide === "long" ? "short" : targetSide === "short" ? "long" : null;

    if (
      oppositeSide &&
      this.sidePositionNotional(oppositeSide, tick.price) > 0
    ) {
      // A reversal must release the old side before the new entry consumes capacity.
      await this.createExit(oppositeSide, 1, null, controlled.confidence);
    }
    if (targetSide === null) {
      await this.createExit("long", 1, controlled.price, controlled.confidence);
      await this.createExit(
        "short",
        1,
        controlled.price,
        controlled.confidence,
      );
      return;
    }

    const refreshed = await this.options.api.getEquity();
    const refreshedEquity =
      refreshed.quoteUnleveraged + refreshed.assetUnleveraged * tick.price;
    if (!(refreshedEquity > 0)) return;
    const currentSideQuote = this.sidePositionNotional(targetSide, tick.price);
    const desiredSideQuote = Math.abs(target) * refreshedEquity;
    const difference = desiredSideQuote - currentSideQuote;
    const minimum = Math.max(this.config.minTradeQuote, Number.EPSILON);
    if (difference < -minimum) {
      await this.createExit(
        targetSide,
        Math.min(1, -difference / currentSideQuote),
        controlled.price,
        controlled.confidence,
      );
      return;
    }
    if (difference < minimum) return;

    const requestedLeverage = Math.max(1, Math.abs(target));
    const capacity = await this.options.api.getOrderCapacity({
      side: targetSide === "long" ? "buy" : "sell",
      price: tick.price,
      leverage: requestedLeverage,
    });
    if (!(capacity.quote > 0)) return;
    await this.createEntry(
      targetSide,
      Math.min(1, difference / capacity.quote),
      requestedLeverage,
      controlled.price,
      controlled.confidence,
      tick,
    );
  }

  private controlTargetExposure(
    signal: TradingStrategyTargetExposureSignal,
    currentExposure: number,
    strategyStaticConfidence: number,
  ): TradingStrategyTargetExposureSignal | null {
    const desired = Math.max(
      -this.config.maxTargetLeverage,
      Math.min(this.config.maxTargetLeverage, signal.targetExposure),
    );
    const epsilon = 1e-10;
    const currentSide: PositionSide | null =
      currentExposure > epsilon
        ? "long"
        : currentExposure < -epsilon
          ? "short"
          : null;
    const desiredSide: PositionSide | null =
      desired > epsilon ? "long" : desired < -epsilon ? "short" : null;

    // Risk-reducing targets, including complete closure, are never delayed or capped.
    if (
      desiredSide === null ||
      (currentSide === desiredSide &&
        Math.abs(desired) <= Math.abs(currentExposure) + epsilon)
    ) {
      this.signalConfirmation = null;
      return { ...signal, targetExposure: desired };
    }
    // A reversal always crosses flat first. A later observation may enter the new side.
    if (currentSide !== null && desiredSide !== currentSide) {
      this.signalConfirmation = null;
      return { ...signal, targetExposure: 0, price: null };
    }
    const controlledLeverage = this.controlExposureExpansion({
      side: desiredSide!,
      desiredLeverage: Math.abs(desired),
      currentLeverage: Math.abs(currentExposure),
      confidence: signal.confidence,
      staticConfidence: strategyStaticConfidence,
    });
    if (controlledLeverage === null) return null;
    return {
      ...signal,
      targetExposure:
        desiredSide === "long" ? controlledLeverage : -controlledLeverage,
    };
  }

  private async controlEntry(
    signal: TradingStrategyEntrySignal,
    tick: TradingTick,
  ): Promise<{ signal: TradingStrategyEntrySignal; maximumQuote: number } | null> {
    if (this.exposureControlsAreNeutral()) {
      return { signal, maximumQuote: Number.POSITIVE_INFINITY };
    }
    if (!(tick.price > 0)) return null;
    const account = await this.options.api.getEquity();
    const equity = account.quoteUnleveraged + account.assetUnleveraged * tick.price;
    if (!(equity > 0)) return null;
    const currentLeverage = this.sidePositionNotional(signal.side, tick.price) / equity;
    const desiredLeverage = Math.min(
      this.config.maxTargetLeverage,
      currentLeverage + Math.max(0, signal.leverage),
    );
    const controlledLeverage = this.controlExposureExpansion({
      side: signal.side,
      desiredLeverage,
      currentLeverage,
      confidence: signal.confidence,
      staticConfidence: this.options.strategy.staticConfidence(),
    });
    if (controlledLeverage === null || controlledLeverage <= currentLeverage) return null;
    return {
      signal: {
        ...signal,
        leverage: Math.min(signal.leverage, Math.max(1, controlledLeverage)),
      },
      maximumQuote: (controlledLeverage - currentLeverage) * equity,
    };
  }

  private controlExposureExpansion(input: {
    side: PositionSide;
    desiredLeverage: number;
    currentLeverage: number;
    confidence: number | null;
    staticConfidence: number;
  }): number | null {
    if (this.exposureControlsAreNeutral()) return input.desiredLeverage;
    const epsilon = 1e-10;
    if (!Number.isFinite(input.staticConfidence)) return null;
    const staticConfidence = Math.max(0, Math.min(1, input.staticConfidence));
    const signalConfidence = input.confidence === null
      ? 0
      : Math.max(0, Math.min(1, input.confidence));
    if (!Number.isFinite(signalConfidence)) return null;
    const configuredLeverageFloor = this.config.exposureControl.confidenceLeverageFloor;
    if (!Number.isFinite(configuredLeverageFloor)) return null;
    const leverageFloor = Math.max(0, Math.min(1, configuredLeverageFloor));
    const effectiveFloor = leverageFloor * staticConfidence;
    const leverageFraction = staticConfidence * (
      effectiveFloor + (1 - effectiveFloor) * signalConfidence
    );
    const confidenceLimitedLeverage = Math.min(
      input.desiredLeverage,
      this.config.maxTargetLeverage * leverageFraction,
    );
    if (!(confidenceLimitedLeverage > input.currentLeverage + epsilon)) {
      this.signalConfirmation = null;
      return confidenceLimitedLeverage;
    }

    const configuredMinimumConfidence =
      this.config.exposureControl.minimumSignalConfidence;
    const configuredMaximumThreshold =
      this.config.exposureControl.maximumSignalConfidenceThreshold;
    if (
      !Number.isFinite(configuredMinimumConfidence)
      || !Number.isFinite(configuredMaximumThreshold)
    ) return null;
    const minimumConfidence = Math.max(0, Math.min(1, configuredMinimumConfidence));
    const maximumThreshold = Math.max(
      minimumConfidence,
      Math.min(1, configuredMaximumThreshold),
    );
    const requiredSignalConfidence = minimumConfidence
      + (maximumThreshold - minimumConfidence) * (1 - staticConfidence);
    if (signalConfidence < requiredSignalConfidence) return null;

    const configuredConfirmationMass = this.config.exposureControl.expansionConfirmationMass;
    if (!Number.isFinite(configuredConfirmationMass)) return null;
    const requiredConfirmationMass = Math.max(0, configuredConfirmationMass);
    if (this.signalConfirmation?.side === input.side) {
      this.signalConfirmation.confirmationMass += signalConfidence;
      this.signalConfirmation.minimumLeverage = Math.min(
        this.signalConfirmation.minimumLeverage,
        confidenceLimitedLeverage,
      );
      this.signalConfirmation.maximumLeverage = Math.max(
        this.signalConfirmation.maximumLeverage,
        confidenceLimitedLeverage,
      );
    } else {
      this.signalConfirmation = {
        side: input.side,
        confirmationMass: signalConfidence,
        minimumLeverage: confidenceLimitedLeverage,
        maximumLeverage: confidenceLimitedLeverage,
      };
    }
    if (this.signalConfirmation.confirmationMass < requiredConfirmationMass) return null;

    const confirmed = this.signalConfirmation;
    this.signalConfirmation = null;
    const selectedLeverage = confirmed.minimumLeverage
      + staticConfidence * (
        confirmed.maximumLeverage - confirmed.minimumLeverage
      );
    const capFraction = Math.max(0, this.config.exposureControl.expansionDeltaCapFraction);
    const maximumDelta = this.config.maxTargetLeverage
      * capFraction
      * staticConfidence ** 2;
    const controlledLeverage = Math.min(
      selectedLeverage,
      input.currentLeverage + maximumDelta,
    );
    return controlledLeverage > input.currentLeverage + epsilon
      ? controlledLeverage
      : null;
  }

  private cancelSignalConfirmation(
    side: PositionSide,
  ): void {
    if (this.signalConfirmation?.side === side) this.signalConfirmation = null;
  }

  private exposureControlsAreNeutral(): boolean {
    const control = this.config.exposureControl;
    return control.confidenceLeverageFloor === 1
      && control.minimumSignalConfidence === 0
      && control.maximumSignalConfidenceThreshold === 0
      && control.expansionConfirmationMass === 0
      && control.expansionDeltaCapFraction >= 1;
  }

  private signedPositionNotional(price: number): number {
    return (
      this.sidePositionNotional("long", price) -
      this.sidePositionNotional("short", price)
    );
  }

  private sidePositionNotional(side: PositionSide, price: number): number {
    return this.positions.reduce(
      (sum, position) =>
        sum + (position.side === side ? position.asset * price : 0),
      0,
    );
  }

  private async createExit(
    side: PositionSide,
    size: number,
    price: number | null,
    confidence: number | null,
  ): Promise<void> {
    const positions = this.positions.filter(
      (position) => position.side === side && position.asset > 0,
    );
    const total = positions.reduce(
      (sum, position) => sum + this.closableAsset(position),
      0,
    );
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
      null,
      true,
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
    reduceOnly = false,
  ): Promise<void> {
    const market = signalPrice === null;
    const count = market ? 1 : Math.max(1, Math.round(config.orderCount));
    const weights = gridWeights(count, config, confidence);
    const rules = await this.options.api.getMarketRules();
    const quantityRules = market ? rules.marketQuantity : rules.limitQuantity;
    const prices = weights.map((_, index) =>
      signalPrice === null
        ? null
        : normalizePrice(
            gridPrice(signalPrice, side, index, config.maxPriceStep),
            rules.price,
          ),
    );
    const sizes = weights.map((weight) => quantity * weight);
    const totalQuote = sizes.reduce(
      (sum, size, index) => sum + size * (prices[index] ?? grid.creationPrice),
      0,
    );
    if (quoteLimit !== null && totalQuote > quoteLimit) {
      const scale = quoteLimit / totalQuote;
      for (let index = 0; index < sizes.length; index += 1)
        sizes[index] *= scale;
    }
    let carry = 0;
    for (let index = sizes.length - 1; index >= 0; index -= 1) {
      sizes[index] += carry;
      carry = 0;
      const notional = sizes[index] * (prices[index] ?? grid.creationPrice);
      if (
        index > 0 &&
        rules.minNotional !== null &&
        notional < rules.minNotional
      ) {
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
      const orderInput = {
        side,
        size,
        leverage: position.leverage,
        reduceOnly,
      };
      const result =
        price === null
          ? await this.options.api.createMarketOrder(orderInput)
          : await this.options.api.createLimitOrder({ ...orderInput, price });
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
        const reset =
          this.config.exitGrid.reset === "previous-anchor" &&
          (position.side === "long"
            ? tick.price >
              position.exitGrid.creationPrice *
                (1 + this.config.exitGrid.maxPriceStep)
            : tick.price <
              position.exitGrid.creationPrice *
                (1 - this.config.exitGrid.maxPriceStep));
        if (reset) {
          await this.replaceExitGrid(position, position.asset, tick.price);
        }
        continue;
      }
      const stop =
        position.stopLossPrice !== null &&
        (position.side === "long"
          ? tick.price <= position.stopLossPrice
          : tick.price >= position.stopLossPrice);
      const take =
        position.takeProfitPrice !== null &&
        (position.side === "long"
          ? tick.price >= position.takeProfitPrice
          : tick.price <= position.takeProfitPrice);
      if (
        stop ||
        take ||
        (position.expiresAt !== null && tick.timestamp >= position.expiresAt)
      ) {
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
      position.stopLossPrice =
        entryPrice *
        (position.side === "long"
          ? 1 - this.config.stopLossRate
          : 1 + this.config.stopLossRate);
    }
    if (
      this.config.takeProfitRate !== null &&
      position.takeProfitPrice === null
    ) {
      position.takeProfitPrice =
        entryPrice *
        (position.side === "long"
          ? 1 + this.config.takeProfitRate
          : 1 - this.config.takeProfitRate);
    }
  }

  private updateBorrow(
    position: TradingPosition,
    fill: { filledAsset: number; filledQuote: number },
  ): void {
    const borrowedRate = Math.max(0, 1 - 1 / position.leverage);
    if (position.side === "long") {
      const borrowed = fill.filledQuote * borrowedRate;
      position.externalBorrow.quote +=
        borrowed - this.allocateInternal(position, 0, borrowed);
    } else {
      const borrowed = fill.filledAsset * borrowedRate;
      position.externalBorrow.asset +=
        borrowed - this.allocateInternal(position, borrowed, 0);
    }
  }

  private applyMaintenance(quoteCharge: number, assetCharge: number): void {
    distributeMaintenance(
      this.positions.filter((position) => position.side === "long"),
      quoteCharge,
      "quote",
    );
    distributeMaintenance(
      this.positions.filter((position) => position.side === "short"),
      assetCharge,
      "asset",
    );
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
      (borrow) =>
        borrow.asset > Number.EPSILON || borrow.quote > Number.EPSILON,
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
      if (
        lender.id === borrower.id ||
        lender.side === borrower.side ||
        remaining <= 0
      ) {
        continue;
      }
      const lent = this.lentAmounts(lender.id);
      const available =
        asset > 0
          ? Math.max(0, lender.asset - lent.asset)
          : Math.max(0, lender.quote - lent.quote);
      const amount = Math.min(remaining, available);
      if (amount <= 0) {
        continue;
      }
      const existing = borrower.internalBorrow.find(
        (borrow) => borrow.positionId === lender.id,
      );
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
    const availableQuoteRate =
      position.quote > 0
        ? Math.max(0, position.quote - lent.quote) / position.quote
        : 1;
    return position.asset * Math.min(1, availableQuoteRate);
  }

  private lentAmounts(positionId: string): { asset: number; quote: number } {
    return this.positions.reduce(
      (total, position) => {
        for (const borrow of position.internalBorrow) {
          if (borrow.positionId === positionId) {
            total.asset += borrow.asset;
            total.quote += borrow.quote;
          }
        }
        return total;
      },
      { asset: 0, quote: 0 },
    );
  }

  private findOrder(id: string) {
    for (const position of this.positions) {
      for (const [grid, entry] of [
        [position.entryGrid, true],
        [position.exitGrid, false],
      ] as const) {
        const item = grid?.orders.find(({ order }) => order.id === id);
        if (grid && item) {
          return { position, grid, entry, item };
        }
      }
    }
  }

  private removeEmptyGrid(
    position: TradingPosition,
    grid: PositionGrid,
    entry: boolean,
  ): void {
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
    await Promise.all(
      grid.orders
        .filter(
          ({ order }) =>
            order.status !== "filled" && order.status !== "rejected",
        )
        .map(({ order }) => this.options.api.cancelOrder(order.id)),
    );
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
  return (side === "long") === entry ? fill.filledQuote : fill.filledAsset;
}

function ordersOf(
  positions: readonly TradingPosition[],
): TradingOrderSnapshot[] {
  return positions.flatMap((position) =>
    [
      ...(position.entryGrid?.orders ?? []),
      ...(position.exitGrid?.orders ?? []),
    ].map(({ order }) => order),
  );
}

function plannedOrdersOf(
  positions: readonly TradingPosition[],
): TradingOrderSnapshot[] {
  return ordersOf(positions).filter(
    (order) => order.status !== "filled" && order.status !== "rejected",
  );
}

function sumSide(
  positions: readonly TradingPosition[],
  side: PositionSide,
): number {
  return positions.reduce(
    (sum, position) => sum + (position.side === side ? position.asset : 0),
    0,
  );
}

function distributeMaintenance(
  positions: readonly TradingPosition[],
  charge: number,
  unit: "asset" | "quote",
): void {
  if (!(charge > 0) || positions.length === 0) return;
  const externalTotal = positions.reduce(
    (sum, position) => sum + Math.max(0, position.externalBorrow[unit]),
    0,
  );
  const amountTotal = positions.reduce(
    (sum, position) => sum + Math.max(0, position[unit]),
    0,
  );
  let remaining = charge;
  for (let index = 0; index < positions.length; index += 1) {
    const position = positions[index]!;
    const weight =
      externalTotal > 0
        ? Math.max(0, position.externalBorrow[unit]) / externalTotal
        : amountTotal > 0
          ? Math.max(0, position[unit]) / amountTotal
          : 1 / positions.length;
    const allocated =
      index === positions.length - 1 ? remaining : charge * weight;
    position[unit] += allocated;
    position.externalBorrow[unit] += allocated;
    remaining -= allocated;
  }
}

function gridWeights(
  count: number,
  config: GridConfig,
  confidence: number | null,
): number[] {
  const ratio =
    confidence === null
      ? config.sizeFraction
      : 1 - clamp01(confidence) * (1 - config.sizeFraction);
  const raw = Array.from({ length: count }, (_, index) =>
    config.sizeDistribution === "linear"
      ? confidence === null
        ? count - index
        : 1 + clamp01(confidence) * (count - index - 1)
      : Math.max(Number.EPSILON, ratio) ** index,
  );
  const total = raw.reduce((sum, weight) => sum + weight, 0);
  return raw.map((weight) => weight / total);
}

function gridPrice(
  anchor: number,
  side: "buy" | "sell",
  index: number,
  step: number,
): number {
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
