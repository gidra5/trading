import type {
  BotCoreState,
  BotStatus,
  LegacyValleyPeakConfig,
  LegacyValleyPeakMemory,
  StrategyConfig,
  StrategyMemory,
} from "./types.js";
import {
  createLegacyValleyPeakMemory,
  evaluateLegacyValleyPeak,
  normalizeLegacyValleyPeakMemory,
  type LegacyValleyPeakDecision,
  type LegacyValleyPeakInput,
} from "./legacy-valley-peak.js";

export type BotConfig = LegacyValleyPeakConfig;
export type BotMemory = LegacyValleyPeakMemory;
export type BotInput = LegacyValleyPeakInput;
export type BotDecision = LegacyValleyPeakDecision;

export function evaluateBot(
  memory: BotMemory,
  config: BotConfig,
  input: BotInput,
): BotDecision {
  return evaluateLegacyValleyPeak(memory, config, input);
}

export function createBotCoreMemory(config: StrategyConfig): StrategyMemory {
  return {
    prices: [],
    lastSignal: "hold",
    lastActionAt: 0,
    legacyValleyPeak: createLegacyValleyPeakMemory(config.legacyValleyPeak),
  };
}

export function createBotCoreState(
  config: StrategyConfig,
  options: {
    id?: string;
    status?: BotStatus;
    now?: number;
  } = {},
): BotCoreState {
  const now = options.now ?? Date.now();
  return {
    id: options.id ?? "bot-core",
    status: options.status ?? "running",
    symbol: config.symbol,
    baseAsset: config.baseAsset,
    quoteAsset: config.quoteAsset,
    lastPrice: 0,
    sequence: 0,
    createdAt: now,
    updatedAt: now,
    runStartedAt: now,
    memory: createBotCoreMemory(config),
    config,
  };
}

export class PeakValleyBotCore {
  constructor(private readonly state: BotCoreState) {}

  view(): Readonly<BotCoreState> {
    return this.state;
  }

  snapshot(): BotCoreState {
    return structuredClone(this.state);
  }

  memory(): LegacyValleyPeakMemory {
    return this.ensureLegacyValleyPeakMemory();
  }

  evaluate(input: BotInput): BotDecision {
    return evaluateBot(
      this.ensureLegacyValleyPeakMemory(),
      this.state.config.legacyValleyPeak,
      input,
    );
  }

  private ensureLegacyValleyPeakMemory(): LegacyValleyPeakMemory {
    const normalized = normalizeLegacyValleyPeakMemory(
      this.state.memory.legacyValleyPeak,
      this.state.config.legacyValleyPeak,
    );
    this.state.memory.legacyValleyPeak = normalized;
    return normalized;
  }
}
