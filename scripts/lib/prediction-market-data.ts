export interface KalshiTradeRaw {
  trade_id: string;
  ticker: string;
  count_fp?: string;
  count?: number;
  yes_price_dollars?: string;
  yes_price?: number;
  no_price_dollars?: string;
  no_price?: number;
  created_time: string;
  taker_side?: string;
  taker_outcome_side?: string;
  is_block_trade?: boolean;
}

export interface PredictionTrade {
  id: string;
  ticker: string;
  timeMs: number;
  yesPrice: number;
  count: number;
  takerSide: string | null;
  isBlockTrade: boolean;
}

export interface CausalTradeState {
  availableAt: number;
  lastProbability: number | null;
  lastTradeAgeMs: number | null;
  intervalTradeCount: number;
  intervalContractVolume: number;
  intervalVwapProbability: number | null;
  intervalMinProbability: number | null;
  intervalMaxProbability: number | null;
  intervalProbabilityChange: number | null;
  intervalYesTakerFraction: number | null;
}

export interface KalshiCandlestickRaw {
  end_period_ts: number;
  yes_bid?: Record<string, string | number | null> | null;
  yes_ask?: Record<string, string | number | null> | null;
  price?: Record<string, string | number | null> | null;
  volume_fp?: string;
  volume?: string | number;
  open_interest_fp?: string;
  open_interest?: string | number;
}

export interface CausalMinuteState {
  availableAt: number;
  yesBidClose: number | null;
  yesAskClose: number | null;
  quoteMidProbability: number | null;
  quotedSpread: number | null;
  tradeOpenProbability: number | null;
  tradeLowProbability: number | null;
  tradeHighProbability: number | null;
  tradeCloseProbability: number | null;
  tradeMeanProbability: number | null;
  previousTradeProbability: number | null;
  contractVolume: number;
  openInterest: number | null;
}

function finite(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function probability(dollars: unknown, cents: unknown): number | null {
  const dollarValue = finite(dollars);
  if (dollarValue !== null) return dollarValue;
  const centValue = finite(cents);
  return centValue === null ? null : centValue / 100;
}

function candleField(
  field: Record<string, string | number | null> | null | undefined,
  name: string,
): number | null {
  if (!field) return null;
  return probability(field[`${name}_dollars`], field[name]);
}

export function normalizeKalshiTrades(rows: KalshiTradeRaw[]): PredictionTrade[] {
  const seen = new Set<string>();
  const normalized: PredictionTrade[] = [];
  for (const row of rows) {
    if (!row.trade_id || seen.has(row.trade_id)) continue;
    const timeMs = Date.parse(row.created_time);
    const yesPrice = probability(row.yes_price_dollars, row.yes_price);
    const count = finite(row.count_fp ?? row.count);
    if (!Number.isFinite(timeMs) || yesPrice === null || yesPrice < 0 || yesPrice > 1 || count === null || count < 0) continue;
    seen.add(row.trade_id);
    normalized.push({
      id: row.trade_id,
      ticker: row.ticker,
      timeMs,
      yesPrice,
      count,
      takerSide: row.taker_outcome_side ?? row.taker_side ?? null,
      isBlockTrade: row.is_block_trade === true,
    });
  }
  normalized.sort((left, right) => left.timeMs - right.timeMs || left.id.localeCompare(right.id));
  return normalized;
}

/**
 * Produces state at candle origins. A trade stamped exactly at an origin is not
 * included: the model could not have observed it before beginning that candle.
 */
export function buildCausalTradeStates(
  trades: PredictionTrade[],
  startMs: number,
  endMs: number,
  stepMs: number,
): CausalTradeState[] {
  if (!(stepMs > 0) || endMs < startMs) throw new Error("Invalid causal-state range");
  const sorted = [...trades].sort((left, right) => left.timeMs - right.timeMs || left.id.localeCompare(right.id));
  const firstOrigin = Math.ceil(startMs / stepMs) * stepMs;
  const states: CausalTradeState[] = [];
  let cursor = 0;
  let last: PredictionTrade | null = null;
  for (let origin = firstOrigin; origin <= endMs; origin += stepMs) {
    const interval: PredictionTrade[] = [];
    while (cursor < sorted.length && sorted[cursor]!.timeMs < origin) {
      const trade = sorted[cursor++]!;
      last = trade;
      if (trade.timeMs >= origin - stepMs) interval.push(trade);
    }
    const intervalVolume = interval.reduce((sum, trade) => sum + trade.count, 0);
    const vwap = intervalVolume > 0
      ? interval.reduce((sum, trade) => sum + trade.yesPrice * trade.count, 0) / intervalVolume
      : null;
    const yesTakerVolume = interval.reduce((sum, trade) => sum + (trade.takerSide === "yes" ? trade.count : 0), 0);
    states.push({
      availableAt: origin,
      lastProbability: last?.yesPrice ?? null,
      lastTradeAgeMs: last ? origin - last.timeMs : null,
      intervalTradeCount: interval.length,
      intervalContractVolume: intervalVolume,
      intervalVwapProbability: vwap,
      intervalMinProbability: interval.length ? Math.min(...interval.map((trade) => trade.yesPrice)) : null,
      intervalMaxProbability: interval.length ? Math.max(...interval.map((trade) => trade.yesPrice)) : null,
      intervalProbabilityChange: interval.length > 1
        ? interval.at(-1)!.yesPrice - interval[0]!.yesPrice
        : null,
      intervalYesTakerFraction: intervalVolume > 0 ? yesTakerVolume / intervalVolume : null,
    });
  }
  return states;
}

/** Build the same strict-pre-origin state for a sparse set of model origins. */
export function buildCausalTradeStatesForOrigins(
  trades: PredictionTrade[],
  originsMs: number[],
  intervalMs: number,
): CausalTradeState[] {
  if (!(intervalMs > 0)) throw new Error("Invalid causal-state interval");
  const sorted = [...trades].sort((left, right) => left.timeMs - right.timeMs || left.id.localeCompare(right.id));
  const origins = [...new Set(originsMs)].sort((left, right) => left - right);
  const states: CausalTradeState[] = [];
  let cursor = 0;
  let last: PredictionTrade | null = null;
  const interval: PredictionTrade[] = [];
  for (const origin of origins) {
    while (cursor < sorted.length && sorted[cursor]!.timeMs < origin) {
      last = sorted[cursor++]!;
    }
    interval.length = 0;
    for (let index = cursor - 1; index >= 0 && sorted[index]!.timeMs >= origin - intervalMs; index -= 1) {
      interval.push(sorted[index]!);
    }
    interval.reverse();
    const intervalVolume = interval.reduce((sum, trade) => sum + trade.count, 0);
    const vwap = intervalVolume > 0
      ? interval.reduce((sum, trade) => sum + trade.yesPrice * trade.count, 0) / intervalVolume
      : null;
    const yesTakerVolume = interval.reduce((sum, trade) => sum + (trade.takerSide === "yes" ? trade.count : 0), 0);
    states.push({
      availableAt: origin,
      lastProbability: last?.yesPrice ?? null,
      lastTradeAgeMs: last ? origin - last.timeMs : null,
      intervalTradeCount: interval.length,
      intervalContractVolume: intervalVolume,
      intervalVwapProbability: vwap,
      intervalMinProbability: interval.length ? Math.min(...interval.map((trade) => trade.yesPrice)) : null,
      intervalMaxProbability: interval.length ? Math.max(...interval.map((trade) => trade.yesPrice)) : null,
      intervalProbabilityChange: interval.length > 1 ? interval.at(-1)!.yesPrice - interval[0]!.yesPrice : null,
      intervalYesTakerFraction: intervalVolume > 0 ? yesTakerVolume / intervalVolume : null,
    });
  }
  return states;
}

export function normalizeKalshiMinuteCandles(rows: KalshiCandlestickRaw[]): CausalMinuteState[] {
  const byEnd = new Map<number, CausalMinuteState>();
  for (const row of rows) {
    const availableAt = finite(row.end_period_ts);
    if (availableAt === null) continue;
    const yesBidClose = candleField(row.yes_bid, "close");
    const yesAskClose = candleField(row.yes_ask, "close");
    const contractVolume = finite(row.volume_fp ?? row.volume) ?? 0;
    const openInterest = finite(row.open_interest_fp ?? row.open_interest);
    byEnd.set(availableAt * 1000, {
      availableAt: availableAt * 1000,
      yesBidClose,
      yesAskClose,
      quoteMidProbability: yesBidClose !== null && yesAskClose !== null ? (yesBidClose + yesAskClose) / 2 : null,
      quotedSpread: yesBidClose !== null && yesAskClose !== null ? yesAskClose - yesBidClose : null,
      tradeOpenProbability: candleField(row.price, "open"),
      tradeLowProbability: candleField(row.price, "low"),
      tradeHighProbability: candleField(row.price, "high"),
      tradeCloseProbability: candleField(row.price, "close"),
      tradeMeanProbability: candleField(row.price, "mean"),
      previousTradeProbability: candleField(row.price, "previous"),
      contractVolume,
      openInterest,
    });
  }
  return [...byEnd.values()].sort((left, right) => left.availableAt - right.availableAt);
}

export function statesAvailableAt<T extends { availableAt: number }>(rows: T[], originMs: number): T[] {
  return rows.filter((row) => row.availableAt <= originMs);
}
