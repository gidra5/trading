import type { PositionBorrow } from "./bot.js";
import type { PositionSide, StrategyDiagnostics } from "./strategy.js";
import type { TradingOrderType } from "./trading-api.js";

export type BacktestGridKind = "entry" | "exit";
export type BacktestGridCause =
  | "strategy-entry"
  | "strategy-exit"
  | "stop-loss"
  | "take-profit"
  | "expiry"
  | "price-reset"
  | "fill-reset";

export interface BacktestGridTrace {
  id: string;
  positionId: string;
  kind: BacktestGridKind;
  cause: BacktestGridCause;
  createdAt: number;
  creationPrice: number;
  orderIds: string[];
}

export interface BacktestOrderFillTrace {
  id: string;
  time: number;
  price: number;
  asset: number;
  quote: number;
  feeQuote: number;
  remaining: number;
}

export interface BacktestOrderTrace {
  id: string;
  positionId: string;
  positionSide: PositionSide;
  gridId: string;
  grid: BacktestGridKind;
  side: "buy" | "sell";
  type: TradingOrderType;
  size: number;
  price: number | null;
  stopPrice: number | null;
  createdAt: number;
  endedAt: number | null;
  outcome: "open" | "filled" | "rejected" | "withdrawn";
  fills: BacktestOrderFillTrace[];
}

export interface BacktestPositionStateTrace {
  time: number;
  asset: number;
  quote: number;
  externalBorrow: { asset: number; quote: number };
  internalBorrow: PositionBorrow[];
  lentTo: PositionBorrow[];
  entryGridId: string | null;
  exitGridId: string | null;
}

export interface BacktestPositionTrace {
  id: string;
  side: PositionSide;
  leverage: number;
  createdAt: number;
  openedAt: number | null;
  closedAt: number | null;
  entryOrderIds: string[];
  exitOrderIds: string[];
  states: BacktestPositionStateTrace[];
}

export interface BacktestSignalTrace {
  time: number;
  price: number;
  source: "price" | "kama";
  active: Array<{ type: "entry" | "exit"; side: PositionSide }>;
  gates: StrategyDiagnostics["gates"];
  blockers: readonly string[];
  indicators: Readonly<Record<string, number | null>>;
}

export interface BacktestExtremumOrderTrace {
  orderId: string;
  fillId: string;
  positionId: string;
  gridId: string;
  grid: BacktestGridKind;
  time: number;
  price: number;
  asset: number;
  quote: number;
  timeErrorMs: number;
  priceErrorPct: number;
  withinThreshold: boolean;
}

export interface BacktestExtremumErrorBox {
  minTimeErrorMs: number;
  maxTimeErrorMs: number;
  minPriceErrorPct: number;
  maxPriceErrorPct: number;
  quote: number;
  asset: number;
  withinThresholdQuote: number;
}

export interface BacktestExtremumTrace {
  id: string;
  kind: "peak" | "valley";
  time: number;
  price: number;
  smaWindowMs: number;
  thresholdTimeMs: number;
  thresholdPriceDistancePct: number;
  p99TimeDistanceMs: number;
  p99PriceDistancePct: number;
  orders: BacktestExtremumOrderTrace[];
  errorBox: BacktestExtremumErrorBox | null;
}

export type BacktestOracleState = "flat" | "long" | "short";

export interface BacktestOraclePoint {
  time: number;
  price: number;
  fromState: BacktestOracleState;
  state: BacktestOracleState;
  action: "hold" | "open" | "close" | "switch";
}

export interface BacktestOraclePath {
  mode: "fixed-notional";
  leverage: number;
  friction: number;
  points: BacktestOraclePoint[];
}

export interface BacktestTraceFrame {
  time: number;
  price: number;
  metrics: {
    equity: number;
    netPnl: number;
    returnPct: number;
    realizedPnl: number;
    unrealizedPnl: number;
    maxDrawdownPct: number;
    exposurePct: number;
    maxEffectiveLeverage: number;
    feesPaid: number;
    tradeCount: number;
    winRate: number;
  };
  quoteFree: number;
  quoteReserved: number;
  baseFree: number;
  baseReserved: number;
  openOrderCount: number;
  longLotCount: number;
  shortLotCount: number;
  entrySignal?: "buy" | "sell";
  exitSignal?: "buy" | "sell";
  positions: {
    summary: {
      longQuantity: number;
      shortQuantity: number;
      netExposureQuote: number;
      grossExposureQuote: number;
      effectiveLeverage: number;
      longExposureQuote: number;
      shortExposureQuote: number;
      pendingLongQuote: number;
      pendingShortQuote: number;
    };
  };
}

export interface BacktestTrace {
  positions: BacktestPositionTrace[];
  grids: BacktestGridTrace[];
  orders: BacktestOrderTrace[];
  signals: BacktestSignalTrace[];
  extrema: BacktestExtremumTrace[];
  oracle: BacktestOraclePath;
  frames: BacktestTraceFrame[];
}
