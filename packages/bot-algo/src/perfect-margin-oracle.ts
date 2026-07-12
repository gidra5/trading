import type {
  BacktestOraclePath,
  BacktestOracleEventMode,
  BacktestOraclePoint,
  BacktestOracleState,
} from "./backtest-trace.js";

interface OracleCandle {
  openTime: number;
  closeTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface PerfectMarginOracleOptions {
  startingQuote: number;
  leverage: number;
  friction: number;
  eventMode?: BacktestOracleEventMode;
  maxPathCandles?: number;
}

export interface PerfectMarginOracleResult {
  leverage: number;
  finalEquity: number;
  netPnl: number;
  returnPct: number;
  compoundedFinalEquity: number;
  compoundedNetPnl: number;
  compoundedReturnPct: number;
  stateCodes: Uint8Array;
  path: BacktestOraclePath;
}

const FLAT = 0;
const LONG = 1;
const SHORT = 2;
type OracleStateCode = typeof FLAT | typeof LONG | typeof SHORT;

export function perfectMarginOracle(
  candles: readonly OracleCandle[],
  options: PerfectMarginOracleOptions,
): PerfectMarginOracleResult {
  const startingQuote = Math.max(0, options.startingQuote);
  const leverage = Math.max(0, options.leverage);
  const friction = Math.max(0, options.friction);
  const eventMode = options.eventMode ?? "ohlc";
  const eventsPerCandle = eventMode === "close" ? 1 : 4;
  const eventCount = candles.length * eventsPerCandle;
  const parents = new Uint8Array(eventCount);
  const states = new Uint8Array(eventCount);
  const openCost = startingQuote * leverage * friction;
  const transitionFactor = (ways: number) => 1 - leverage * friction * ways;
  let additive = [0, -openCost, -openCost];
  // Canonicalize numerically equal paths to the one with fewer state changes.
  let transitions = [0, 1, 1];
  let compounded = [startingQuote, multiply(startingQuote, transitionFactor(1)), multiply(startingQuote, transitionFactor(1))];
  let previousPrice = eventCount > 0 ? event(candles, 0, eventMode).price : 0;

  if (eventCount > 0) parents[0] = encodeParents(FLAT, FLAT, FLAT);
  for (let index = 1; index < eventCount; index += 1) {
    const price = event(candles, index, eventMode).price;
    const move = previousPrice > 0 ? price / previousPrice - 1 : 0;
    const marked = [
      additive[FLAT],
      add(additive[LONG], startingQuote * leverage * move),
      add(additive[SHORT], -startingQuote * leverage * move),
    ];
    const closeCost = openCost;
    const nextFlat = best(
      [marked[FLAT], marked[LONG] - closeCost, marked[SHORT] - closeCost],
      [FLAT, LONG, SHORT],
      [transitions[FLAT], transitions[LONG] + 1, transitions[SHORT] + 1],
    );
    const nextLong = best(
      [marked[LONG], marked[FLAT] - openCost, marked[SHORT] - closeCost * 2],
      [LONG, FLAT, SHORT],
      [transitions[LONG], transitions[FLAT] + 1, transitions[SHORT] + 1],
    );
    const nextShort = best(
      [marked[SHORT], marked[FLAT] - openCost, marked[LONG] - closeCost * 2],
      [SHORT, FLAT, LONG],
      [transitions[SHORT], transitions[FLAT] + 1, transitions[LONG] + 1],
    );
    additive = [nextFlat.value, nextLong.value, nextShort.value];
    transitions = [nextFlat.transitions, nextLong.transitions, nextShort.transitions];
    parents[index] = encodeParents(nextFlat.state, nextLong.state, nextShort.state);

    const markedEquity = [
      compounded[FLAT],
      multiply(compounded[LONG], 1 + leverage * move),
      multiply(compounded[SHORT], 1 - leverage * move),
    ];
    compounded = [
      Math.max(markedEquity[FLAT], multiply(markedEquity[LONG], transitionFactor(1)), multiply(markedEquity[SHORT], transitionFactor(1))),
      Math.max(markedEquity[LONG], multiply(markedEquity[FLAT], transitionFactor(1)), multiply(markedEquity[SHORT], transitionFactor(2))),
      Math.max(markedEquity[SHORT], multiply(markedEquity[FLAT], transitionFactor(1)), multiply(markedEquity[LONG], transitionFactor(2))),
    ];
    previousPrice = price;
  }

  let state = best(additive, [FLAT, LONG, SHORT], transitions).state;
  for (let index = eventCount - 1; index >= 0; index -= 1) {
    states[index] = state;
    state = decodeParent(parents[index] ?? 0, state);
  }

  const netPnl = Math.max(...additive);
  const compoundedFinalEquity = Math.max(...compounded);
  const compoundedNetPnl = compoundedFinalEquity - startingQuote;
  return {
    leverage,
    finalEquity: startingQuote + netPnl,
    netPnl,
    returnPct: startingQuote > 0 ? netPnl / startingQuote * 100 : 0,
    compoundedFinalEquity,
    compoundedNetPnl,
    compoundedReturnPct: startingQuote > 0 ? compoundedNetPnl / startingQuote * 100 : 0,
    stateCodes: states,
    path: {
      mode: "fixed-notional",
      eventMode,
      leverage,
      friction,
      points: tracedPath(candles, states, eventsPerCandle, eventMode, options.maxPathCandles ?? 2_000),
    },
  };
}

function tracedPath(
  candles: readonly OracleCandle[],
  states: Uint8Array,
  eventsPerCandle: number,
  eventMode: BacktestOracleEventMode,
  maxCandles: number,
): BacktestOraclePoint[] {
  const limit = Number.isFinite(maxCandles) ? Math.max(1, Math.floor(maxCandles)) : 2_000;
  const sampleEvery = Math.max(1, Math.ceil(candles.length / limit));
  const included = new Uint8Array(states.length);
  for (let candleIndex = 0; candleIndex < candles.length; candleIndex += 1) {
    if (candleIndex % sampleEvery !== 0 && candleIndex !== candles.length - 1) continue;
    for (let offset = 0; offset < eventsPerCandle; offset += 1) {
      included[candleIndex * eventsPerCandle + offset] = 1;
    }
  }
  for (let index = 1; index < states.length; index += 1) {
    if (states[index] === states[index - 1]) continue;
    included[index - 1] = 1;
    included[index] = 1;
  }
  const points: BacktestOraclePoint[] = [];
  for (let index = 0; index < states.length; index += 1) {
    if (!included[index]) continue;
    const current = stateName(states[index] as OracleStateCode);
    const previous = index > 0 ? stateName(states[index - 1] as OracleStateCode) : "flat";
    const point = event(candles, index, eventMode);
    points.push({ ...point, fromState: previous, state: current, action: action(previous, current) });
  }
  return points;
}

function event(
  candles: readonly OracleCandle[],
  index: number,
  mode: BacktestOracleEventMode,
): { time: number; price: number } {
  if (mode === "close") {
    const candle = candles[index]!;
    return { time: candle.closeTime, price: candle.close };
  }
  const candle = candles[Math.floor(index / 4)]!;
  const offset = index % 4;
  const span = Math.max(1, candle.closeTime - candle.openTime);
  if (offset === 0) return { time: candle.openTime, price: candle.open };
  if (offset === 1) return { time: candle.openTime + span * 0.33, price: candle.high };
  if (offset === 2) return { time: candle.openTime + span * 0.66, price: candle.low };
  return { time: candle.closeTime, price: candle.close };
}

function best(
  values: number[],
  states: OracleStateCode[] = [FLAT, LONG, SHORT],
  transitions: number[] = [0, 0, 0],
): { value: number; state: OracleStateCode; transitions: number } {
  let value = values[0] ?? Number.NEGATIVE_INFINITY;
  let state = states[0] ?? FLAT;
  let transitionCount = transitions[0] ?? 0;
  for (let index = 1; index < values.length; index += 1) {
    const candidate = values[index] ?? Number.NEGATIVE_INFINITY;
    const candidateTransitions = transitions[index] ?? 0;
    const tolerance = Number.EPSILON * 16 * Math.max(1, Math.abs(value), Math.abs(candidate));
    if (
      candidate > value + tolerance
      || (Math.abs(candidate - value) <= tolerance && candidateTransitions < transitionCount)
    ) {
      value = candidate;
      state = states[index] ?? FLAT;
      transitionCount = candidateTransitions;
    }
  }
  return { value, state, transitions: transitionCount };
}

function encodeParents(flat: OracleStateCode, long: OracleStateCode, short: OracleStateCode): number {
  return flat | (long << 2) | (short << 4);
}

function decodeParent(encoded: number, state: OracleStateCode): OracleStateCode {
  return ((encoded >> (state * 2)) & 3) as OracleStateCode;
}

function stateName(state: OracleStateCode): BacktestOracleState {
  return state === LONG ? "long" : state === SHORT ? "short" : "flat";
}

function action(previous: BacktestOracleState, current: BacktestOracleState): BacktestOraclePoint["action"] {
  if (previous === current) return "hold";
  if (previous === "flat") return "open";
  if (current === "flat") return "close";
  return "switch";
}

function add(value: number, delta: number): number {
  return Number.isFinite(value) ? value + delta : Number.NEGATIVE_INFINITY;
}

function multiply(value: number, factor: number): number {
  return Number.isFinite(value) && Number.isFinite(factor) && factor > 0
    ? value * factor
    : Number.NEGATIVE_INFINITY;
}
