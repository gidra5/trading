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
}

export interface BotMetricsSnapshot {
  positionCount: number;
  orderCount: number;
  longAsset: number;
  shortAsset: number;
}

export interface BotDiagnostics<TDiagnostics extends StrategyDiagnostics = StrategyDiagnostics> {
  strategy: TDiagnostics;
  positions: readonly TradingPosition[];
  /** Derived by flattening position entry/exit grids. */
  plannedOrders: readonly TradingOrderSnapshot[];
  entryRisk: readonly {
    side: PositionSide;
    size: number;
    leverage: number;
    blocker: string | null;
  }[];
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
  restore(snapshot: BotSnapshot<TStrategyConfig, TStrategySnapshot>): Promise<void>;
  getMetrics(): BotMetricsSnapshot;
  getDiagnostics(): BotDiagnostics<TDiagnostics>;
  updateConfig(config: TradingBotConfig<TStrategyConfig>): Promise<void>;
}
