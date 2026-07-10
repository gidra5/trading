import type { TradingApi, TradingTick } from "./trading-api.js";

export type PositionSide = "long" | "short";

export interface StrategyOptions<TConfig = unknown> {
  config: TConfig;
  getHistory: TradingApi["getHistory"];
}

export interface TradingStrategyEntrySignal {
  side: PositionSide;
  /** Fraction [0, 1] of the available asset/quote capacity to use. */
  size: number;
  leverage: number;
  /** Null means execute at the current market price. */
  price: number | null;
  /** 0 is uniform grid sizing; 1 concentrates sizing at price. */
  confidence: number | null;
}

export interface TradingStrategyExitSignal {
  side: PositionSide;
  /** Fraction [0, 1] of total matching-side exposure to close. */
  size: number;
  /** Null means execute at the current market price. */
  price: number | null;
  confidence: number | null;
}

export interface StrategyDiagnostics {
  indicators: Readonly<Record<string, number | null>>;
  gates: readonly {
    code: string;
    passed: boolean;
    value?: number;
    threshold?: number;
  }[];
  blockers: readonly string[];
  lastSignal: {
    type: "entry" | "exit";
    side: PositionSide;
    reason: string;
  } | null;
}

export interface StrategySnapshot {
  version: number;
}

export interface TradingStrategy<
  TConfig = unknown,
  TSnapshot extends StrategySnapshot = StrategySnapshot,
  TDiagnostics extends StrategyDiagnostics = StrategyDiagnostics,
> {
  /** The strategy fetches the history required by its indicators. */
  warmup(): Promise<void>;
  onTick(tick: TradingTick): Promise<void>;
  entrySignal(): Promise<TradingStrategyEntrySignal | null>;
  exitSignal(): Promise<TradingStrategyExitSignal | null>;
  snapshot(): Promise<TSnapshot>;
  restore(snapshot: TSnapshot): Promise<void>;
  updateConfig(config: TConfig): Promise<void>;
  getDiagnostics(): TDiagnostics;
}
