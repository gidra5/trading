import type {
  PaperBotState,
  StrategyConfig,
  StrategyMemory,
} from "./legacy/types.js";

const DEFAULT_MAX_RETURNED_ORDERS = 2_000;
const DEFAULT_MAX_RETURNED_FILLS = 2_000;

export function compactBacktestState(
  state: Readonly<PaperBotState>,
  options: {
    maxReturnedOrders?: number;
    maxReturnedFills?: number;
  } = {},
): PaperBotState {
  const maxOrders = normalizeResultLimit(
    options.maxReturnedOrders,
    DEFAULT_MAX_RETURNED_ORDERS,
  );
  const maxFills = normalizeResultLimit(
    options.maxReturnedFills,
    DEFAULT_MAX_RETURNED_FILLS,
  );

  return {
    ...state,
    orders: tail(state.orders, maxOrders).map((order) => ({ ...order })),
    fills: tail(state.fills, maxFills).map((fill) => ({ ...fill })),
    memory: compactBacktestMemory(state.memory, state.config),
    metrics: { ...state.metrics },
    config: structuredClone(state.config),
  };
}

function compactBacktestMemory(
  memory: Readonly<StrategyMemory>,
  config: Readonly<StrategyConfig>,
): StrategyMemory {
  const priceLimit = Math.max(
    50,
    config.legacyValleyPeak.averagingRangesSec.length * 100,
  );

  return {
    prices: tail(memory.prices, priceLimit).slice(),
    lastSignal: memory.lastSignal,
    lastActionAt: memory.lastActionAt,
    lastExtremaSignal: memory.lastExtremaSignal,
    lastExtremaSignalAt: memory.lastExtremaSignalAt,
    lastExtremaSignalPrice: memory.lastExtremaSignalPrice,
    lastExtremaSignalReason: memory.lastExtremaSignalReason,
    legacyValleyPeakDebug: memory.legacyValleyPeakDebug
      ? structuredClone(memory.legacyValleyPeakDebug)
      : undefined,
  };
}

function normalizeResultLimit(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value)) return Number.MAX_SAFE_INTEGER;
  return Math.max(0, Math.round(value));
}

function tail<T>(items: readonly T[], limit: number): readonly T[] {
  if (limit >= items.length) return items;
  return items.slice(items.length - limit);
}
