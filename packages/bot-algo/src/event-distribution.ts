/** A causal event clock and a small distribution tree. All returns are arithmetic. */
import { NATIVE_SECOND_EVENT_FEATURES, NATIVE_SECOND_WARMUP, nativeSecondEventFeatures,
  NATIVE_SECOND_CONTEXT_FEATURES, NATIVE_SECOND_CONTEXT_WARMUP, nativeSecondContextFeatures,
  NATIVE_SECOND_FLOW_CONTEXT_FEATURES, nativeSecondFlowContextFeatures,
  NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES, nativeSecondFlowVwapContextFeatures,
  NATIVE_SECOND_SELECTED_SIGN_FEATURES, nativeSecondSelectedSignFeatures,
  NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES, nativeSecondVolatilityContextFeatures,
  NATIVE_SECOND_DAY_CONTEXT_FEATURES, NATIVE_SECOND_DAY_CONTEXT_WARMUP, nativeSecondDayContextFeatures } from "./event-second-features.js";
export interface EventCandle {
  openTime: number; open: number; high: number; low: number; close: number; volume: number;
  secondBasis?: { availableAt: number; values: number[] };
  nativeTradeFlow?: { availableAt: number; aggregateCountImbalance: number; lastAggressorSide: number;
    buyerSellerVwapGap?: number; aggressiveBuyQuoteVolume?: number; aggressiveSellQuoteVolume?: number;
    aggressiveBuyMaxAggregateQuantity?: number; aggressiveSellMaxAggregateQuantity?: number };
  /** Explicit replay-only carry during declared unavailability; never an observed trade price. */
  carriedMark?: true;
}
export interface EventClock {
  /** Minute observations are the default; native seconds must declare 1000. */
  candleIntervalMs?: 1000 | 60000;
  /** Physical-minute boundaries for the three duration classes. */
  durationBinsMinutes?: readonly [number, number];
  thresholdBps: number; maxCandles: number; runClock?: boolean; reversalClock?: boolean; progressBps?: number;
}
export const eventCandleIntervalMs = (clock: EventClock): number => clock.candleIntervalMs ?? 60000;
export const eventBaseFeatures = (clock: EventClock): readonly string[] => eventCandleIntervalMs(clock) === 1000 ? NATIVE_SECOND_EVENT_FEATURES : EVENT_FEATURES;
export function eventFeatureWarmup(clock: EventClock, names: readonly string[]): number {
  const dayContext = names.length === NATIVE_SECOND_DAY_CONTEXT_FEATURES.length && names.every((name, i) => name === NATIVE_SECOND_DAY_CONTEXT_FEATURES[i]);
  const context = names.length === NATIVE_SECOND_CONTEXT_FEATURES.length && names.every((name, i) => name === NATIVE_SECOND_CONTEXT_FEATURES[i]);
  const flowContext = names.length === NATIVE_SECOND_FLOW_CONTEXT_FEATURES.length && names.every((name, i) => name === NATIVE_SECOND_FLOW_CONTEXT_FEATURES[i]);
  const flowVwapContext = names.length === NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES.length && names.every((name, i) => name === NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES[i]);
  const selectedSign = names.length === NATIVE_SECOND_SELECTED_SIGN_FEATURES.length && names.every((name, i) => name === NATIVE_SECOND_SELECTED_SIGN_FEATURES[i]);
  const volatilityContext = names.length === NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES.length
    && names.every((name, i) => name === NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES[i]);
  const native = dayContext || volatilityContext || selectedSign || flowVwapContext || flowContext || context || names.length === NATIVE_SECOND_EVENT_FEATURES.length && names.every((name, i) => name === NATIVE_SECOND_EVENT_FEATURES[i]);
  if (native !== (eventCandleIntervalMs(clock) === 1000)) throw new Error("Incompatible candle interval and feature contract");
  return dayContext ? NATIVE_SECOND_DAY_CONTEXT_WARMUP : volatilityContext || selectedSign || flowVwapContext || flowContext || context ? NATIVE_SECOND_CONTEXT_WARMUP : native ? NATIVE_SECOND_WARMUP : 1440;
}
export function validateEventClock(clock: EventClock): void {
  if (!Number.isFinite(clock.thresholdBps) || clock.thresholdBps <= 0 || !Number.isInteger(clock.maxCandles) || clock.maxCandles < 1
    || (clock.runClock && clock.reversalClock) || (clock.progressBps !== undefined
      && (!clock.reversalClock || !Number.isFinite(clock.progressBps) || clock.progressBps <= 0))) throw new Error("Invalid event clock");
  if (![1000, 60000].includes(eventCandleIntervalMs(clock)) || eventCandleIntervalMs(clock) === 1000 && clock.reversalClock)
    throw new Error("Unsupported candle interval or directional-change clock");
  if (clock.durationBinsMinutes && (clock.durationBinsMinutes.length !== 2
    || clock.durationBinsMinutes.some(v => !Number.isFinite(v) || v <= 0)
    || clock.durationBinsMinutes[0] >= clock.durationBinsMinutes[1])) throw new Error("Invalid event duration bins");
}
export interface MoveSample {
  start: number; end: number; features: number[]; nextFeatures: number[];
  /** Duration in physical minutes, including fractional minutes for seconds. */
  return: number; low: number; high: number; duration: number; label: number;
  series?: number;
}
export const EVENT_FEATURES = [
  "return1", "return5", "return15", "return60", "return240", "return1440",
  "volatility60", "volatilityRatio", "runSign", "runLength", "efficiency60", "volumeRatio",
] as const;
/** Bounded-lookback coordinates from the existing one-second feature dataset. */
export const EVENT_SECOND_INPUTS = ["range-1s", "active-count-60s", "previous-return-1s",
  "realized-volatility-60s", "close-location-1s"] as const;
export const EVENT_SECOND_FEATURES = [...EVENT_FEATURES, ...EVENT_SECOND_INPUTS] as const;
export const EVENT_PATH_INPUTS = ["runReturn", "distanceHigh60", "distanceLow60", "distanceHigh240", "distanceLow240",
  "distanceHigh1440", "distanceLow1440", "ageHigh1440", "ageLow1440"] as const;
export const EVENT_PATH_FEATURES = [...EVENT_FEATURES, ...EVENT_PATH_INPUTS] as const;
export const EVENT_RUN_INPUTS = ["dcDirection", "dcMagnitude", "dcDuration", "dcOvershoot", "dcPullback",
  "dcConfirmationDuration", "dcPreviousOvershoot", "dcPreviousDuration"] as const;
export const EVENT_RUN_FEATURES = [...EVENT_FEATURES, ...EVENT_RUN_INPUTS] as const;
export const EVENT_RUN_VOLATILITY_FEATURES = [...EVENT_RUN_FEATURES, "log1p-rv-5m-bps"] as const;
const RUN_SIGNED_COORDINATES = [0, 1, 2, 3, 4, 5, 8, 12, 13, 15, 16, 18];
/** All orientation choices are observable and odd under reciprocal prices. */
export function eventRunOrientation(features: readonly number[]): number {
  if (features.length !== EVENT_RUN_FEATURES.length || features.some(v => !Number.isFinite(v))) throw new Error("Invalid run symmetry features");
  for (const index of [12, 8, 0, 1, 2, 3, 4, 5, 13, 15, 16, 18]) if (features[index]) return Math.sign(features[index]);
  return 0;
}
export function canonicalEventRunFeatures(features: readonly number[]): number[] {
  const orientation = eventRunOrientation(features), out = [...features];
  if (orientation < 0) for (const i of RUN_SIGNED_COORDINATES) out[i] *= -1;
  return out;
}
/** Duration is physical minutes; defaults are five and twenty observed candles. */
export function eventMoveLabel(value: number, duration: number, clock: EventClock): number {
  const z = value * 1e4 / clock.thresholdBps;
  const direction = z < -1 ? 0 : z < -0.25 ? 1 : z < 0.25 ? 2 : z < 1 ? 3 : 4;
  const bins = clock.durationBinsMinutes ?? [5 * eventCandleIntervalMs(clock) / 60000, 20 * eventCandleIntervalMs(clock) / 60000];
  return direction * 3 + (duration <= bins[0] ? 0 : duration <= bins[1] ? 1 : 2);
}
const usesSecondBasis = (names: readonly string[]) => names[EVENT_FEATURES.length] === EVENT_SECOND_INPUTS[0];
const validFeatureNames = (names: readonly string[]) => [EVENT_FEATURES, EVENT_SECOND_FEATURES, EVENT_PATH_FEATURES, EVENT_RUN_FEATURES, EVENT_RUN_VOLATILITY_FEATURES, NATIVE_SECOND_EVENT_FEATURES, NATIVE_SECOND_CONTEXT_FEATURES, NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES, NATIVE_SECOND_FLOW_CONTEXT_FEATURES, NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES, NATIVE_SECOND_SELECTED_SIGN_FEATURES, NATIVE_SECOND_DAY_CONTEXT_FEATURES]
  .some(expected => names.length === expected.length && names.every((name, i) => name === expected[i]));
export function hasEventSecondBasis(candle: EventCandle): boolean {
  const basis = candle.secondBasis, decisionTime = candle.openTime + 60_000;
  return !!basis && basis.availableAt <= decisionTime && decisionTime - basis.availableAt < 60_000
    && basis.values.length === EVENT_SECOND_INPUTS.length && basis.values.every(Number.isFinite);
}

export interface EventRunState {
  direction: number; anchor: number; anchorAt: number; extreme: number; extremeAt: number;
  confirmation: number; confirmationAt: number; previousOvershoot: number; previousDuration: number;
  low: number; lowAt: number; high: number; highAt: number;
}

/** Uses completed closes and symmetric log-price thresholds. Confirmation is
 * observable now; the earlier extreme is history, never a retroactive fill. */
function advanceEventRun(s: EventRunState, price: number, at: number, threshold: number): boolean {
  if (!s.direction) {
    if (price < s.low) { s.low = price; s.lowAt = at; }
    if (price > s.high) { s.high = price; s.highAt = at; }
    if (Math.log(price / s.low) >= threshold) { s.direction = 1; s.anchor = s.low; s.anchorAt = s.lowAt; }
    else if (Math.log(price / s.high) <= -threshold) { s.direction = -1; s.anchor = s.high; s.anchorAt = s.highAt; }
    else return false;
    s.confirmation = price; s.confirmationAt = at; s.extreme = price; s.extremeAt = at;
    return true;
  }
  if (s.direction * Math.log(price / s.extreme) > 0) { s.extreme = price; s.extremeAt = at; return false; }
  if (s.direction * Math.log(price / s.extreme) > -threshold) return false;
  s.previousOvershoot = Math.log(s.extreme / s.confirmation);
  s.previousDuration = s.extremeAt - s.confirmationAt;
  s.direction *= -1; s.anchor = s.extreme; s.anchorAt = s.extremeAt;
  s.confirmation = price; s.confirmationAt = at; s.extreme = price; s.extremeAt = at;
  return true;
}

/** One-day bounded state keeps train purging and frozen replay reproducible.
 * It deliberately does not claim an unlimited-history directional-change state. */
export function eventRunState(c: readonly EventCandle[], i: number, thresholdBps: number): EventRunState {
  if (i < 1440 || !Number.isFinite(thresholdBps) || thresholdBps <= 0) throw new Error("Invalid directional-change state");
  const start = i - 1440, price = c[start].close;
  const state: EventRunState = { direction: 0, anchor: price, anchorAt: start, extreme: price, extremeAt: start,
    confirmation: price, confirmationAt: start, previousOvershoot: 0, previousDuration: 0,
    low: price, lowAt: start, high: price, highAt: start };
  for (let j = start + 1; j <= i; j++) {
    if (c[j].openTime - c[j - 1].openTime !== 60_000) throw new Error("Gap in directional-change history");
    advanceEventRun(state, c[j].close, j, thresholdBps / 1e4);
  }
  return state;
}

export function eventFeatures(c: readonly EventCandle[], i: number, names: readonly string[] = EVENT_FEATURES, clock?: EventClock): number[] {
  if (!validFeatureNames(names)) throw new Error("Incompatible event distribution feature contract");
  if (clock) eventFeatureWarmup(clock, names);
  if (names[0] === NATIVE_SECOND_EVENT_FEATURES[0]) {
    if (!clock || eventCandleIntervalMs(clock) !== 1000) throw new Error("Native features require an explicit second clock");
    if (names.length === NATIVE_SECOND_DAY_CONTEXT_FEATURES.length
      && names.every((name, index) => name === NATIVE_SECOND_DAY_CONTEXT_FEATURES[index])) return nativeSecondDayContextFeatures(c, i);
    if (names.length === NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES.length
      && names.every((name, index) => name === NATIVE_SECOND_FLOW_VWAP_CONTEXT_FEATURES[index])) return nativeSecondFlowVwapContextFeatures(c, i);
    if (names.length === NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES.length
      && names.every((name, index) => name === NATIVE_SECOND_VOLATILITY_CONTEXT_FEATURES[index])) return nativeSecondVolatilityContextFeatures(c, i);
    if (names.length === NATIVE_SECOND_SELECTED_SIGN_FEATURES.length
      && names.every((name, index) => name === NATIVE_SECOND_SELECTED_SIGN_FEATURES[index])) return nativeSecondSelectedSignFeatures(c, i);
    if (names.length === NATIVE_SECOND_FLOW_CONTEXT_FEATURES.length
      && names.every((name, index) => name === NATIVE_SECOND_FLOW_CONTEXT_FEATURES[index])) return nativeSecondFlowContextFeatures(c, i);
    return names.length === NATIVE_SECOND_CONTEXT_FEATURES.length
      && names.every((name, index) => name === NATIVE_SECOND_CONTEXT_FEATURES[index])
      ? nativeSecondContextFeatures(c, i) : nativeSecondEventFeatures(c, i);
  }
  if (i < 1440) throw new Error("Event features require one day of causal warmup");
  if (usesSecondBasis(names) && !hasEventSecondBasis(c[i])) throw new Error("Missing, stale or future one-second features");
  let variance = 0, slowVariance = 0, fastVariance = 0, distance = 0, volume = 0;
  for (let j = i - 239; j <= i; j++) {
    const r = Math.log(c[j].close / c[j - 1].close);
    slowVariance += r * r;
    if (j > i - 5) fastVariance += r * r;
    if (j > i - 60) { variance += r * r; distance += Math.abs(r); volume += c[j].volume; }
  }
  const sigma = Math.max(0.00001, Math.sqrt(variance / 60));
  const sign = Math.sign(c[i].close - c[i - 1].close);
  let length = 1;
  while (length < 60 && Math.sign(c[i - length].close - c[i - length - 1].close) === sign) length++;
  const pathFeatures: number[] = [];
  if (names[EVENT_FEATURES.length] === EVENT_RUN_INPUTS[0]) {
    if (!clock) throw new Error("Directional-change features require the model clock");
    const s = eventRunState(c, i, clock.thresholdBps), threshold = clock.thresholdBps / 1e4;
    pathFeatures.push(s.direction, Math.log(c[i].close / s.anchor) / threshold, Math.log1p(i - s.anchorAt),
      Math.log(s.extreme / s.confirmation) / threshold, Math.log(c[i].close / s.extreme) / threshold,
      Math.log1p(s.confirmationAt - s.anchorAt), s.previousOvershoot / threshold, Math.log1p(s.previousDuration));
  }
  if (names[EVENT_FEATURES.length] === EVENT_PATH_INPUTS[0]) {
    pathFeatures.push(Math.log(c[i].close / c[i - length].close) / (sigma * Math.sqrt(length)));
    for (const lookback of [60, 240, 1440]) {
      let high = -Infinity, low = Infinity, highAt = i, lowAt = i;
      for (let j = i - lookback + 1; j <= i; j++) {
        if (c[j].high >= high) { high = c[j].high; highAt = j; }
        if (c[j].low <= low) { low = c[j].low; lowAt = j; }
      }
      pathFeatures.push(Math.log(c[i].close / high) / (sigma * Math.sqrt(lookback)),
        Math.log(c[i].close / low) / (sigma * Math.sqrt(lookback)));
      if (lookback === 1440) pathFeatures.push(Math.log1p(i - highAt) / Math.log1p(1440), Math.log1p(i - lowAt) / Math.log1p(1440));
    }
  }
  return [
    ...[1, 5, 15, 60, 240, 1440].map(lag => Math.log(c[i].close / c[i - lag].close) / (sigma * Math.sqrt(lag))),
    Math.log(sigma * 1e4), Math.log(sigma / Math.max(0.00001, Math.sqrt(slowVariance / 240))),
    sign, Math.log1p(length), Math.abs(Math.log(c[i].close / c[i - 60].close)) / Math.max(1e-12, distance),
    Math.log1p(c[i].volume / Math.max(1e-12, volume / 60)),
    ...(usesSecondBasis(names) ? c[i].secondBasis!.values : pathFeatures),
    ...(names.length === EVENT_RUN_VOLATILITY_FEATURES.length && names[EVENT_FEATURES.length] === EVENT_RUN_INPUTS[0]
      ? [Math.log1p(Math.sqrt(fastVariance) * 1e4)] : []),
  ];
}

/** Ends only when an observed close crosses a barrier, or the timeout expires.
 * No extrema backdating, partial last labels, or crossing missing candles. */
export function observeMove(c: readonly EventCandle[], start: number, clock: EventClock,
  names: readonly string[] = EVENT_FEATURES): MoveSample | null {
  validateEventClock(clock);
  const interval = eventCandleIntervalMs(clock);
  eventFeatureWarmup(clock, names);
  const p = c[start].close;
  if (usesSecondBasis(names) && !hasEventSecondBasis(c[start])) return null;
  const runSign = Math.sign(p - c[start - 1].close);
  const reversal = clock.reversalClock ? eventRunState(c, start, clock.thresholdBps) : undefined;
  let low = 0, high = 0;
  for (let end = start + 1; end <= start + clock.maxCandles && end < c.length; end++) {
    if (c[end].openTime - c[end - 1].openTime !== interval) return null;
    low = Math.min(low, c[end].low / p - 1);
    high = Math.max(high, c[end].high / p - 1);
    const r = c[end].close / p - 1, bars = end - start, duration = bars * interval / 60000;
    const observedBoundary = reversal ? advanceEventRun(reversal, c[end].close, end, clock.thresholdBps / 1e4)
      || (clock.progressBps !== undefined && Math.abs(Math.log(c[end].close / p)) * 1e4 >= clock.progressBps) : clock.runClock
      ? Math.sign(c[end].close - c[end - 1].close) !== runSign
      : Math.abs(r) * 1e4 >= clock.thresholdBps;
    if (observedBoundary || bars === clock.maxCandles) {
      if (usesSecondBasis(names) && !hasEventSecondBasis(c[end])) return null;
      return { start, end, return: r, low, high, duration, label: eventMoveLabel(r, duration, clock),
        features: eventFeatures(c, start, names, clock), nextFeatures: eventFeatures(c, end, names, clock) };
    }
  }
  return null;
}

export interface DistributionNode { feature: number; cut: number; left: number; right: number; leaf: number; }
/** duration is always physical minutes, independent of candle resolution. */
export interface MoveAtom { probability: number; return: number; low: number; high: number; duration: number; next: number; }
export interface EventProjection {
  means: number[]; scales: number[]; coefficients: number[]; intercept: number; cuts: number[]; penalty: number;
}
export interface EventBoost {
  intercept: number; rate: number; trees: Array<{ nodes: DistributionNode[]; values: number[] }>; cuts: number[];
}
export interface EventHidden {
  transition: number[][]; emission: number[][]; beliefs: number[][];
  initial: number; nextByClass: number[][]; iterations: number; sequences: number;
  logLikelihood: number; resolution: number;
}
export interface EventDistribution {
  version: 1; clock: EventClock; featureNames: readonly string[]; nodes: DistributionNode[];
  kernels: MoveAtom[][]; counts: number[]; classProbabilities: number[][]; priorClasses: number[];
  trainingSamples: number;
  meanCalibration?: { scale: number; samples: number };
  forest?: { nodes: DistributionNode[]; leafCount: number }[];
  projection?: EventProjection;
  boost?: EventBoost;
  hidden?: EventHidden;
  runSymmetry?: boolean;
  runConditioned?: { directionPrior: number };
  /** Shared canonical partition, physical direction, then observed RV5 band.
   * Kernels are empirical joint paths with observed successor bands. */
  runVolatility?: { cut: number; prior: number; globalPriorShare: number; recentSamples: number; pooledHighSamples: number; quietBase?: boolean };
}

/** Only an observed event class updates the belief; latent states are never exposed. */
export function eventHiddenNext(model: EventDistribution, state: number, label: number): number {
  if (!model.hidden || !Number.isInteger(state) || !model.hidden.nextByClass[state]
    || !Number.isInteger(label) || label < 0 || label >= 15) throw new Error("Invalid event belief update");
  return model.hidden.nextByClass[state][label];
}

export function validateEventDistribution(model: EventDistribution): void {
  validateEventClock(model.clock);
  eventFeatureWarmup(model.clock, model.featureNames);
  if (model.version !== 1 || !validFeatureNames(model.featureNames) || !model.kernels.length)
    throw new Error("Incompatible event distribution feature contract");
  if (model.runVolatility) {
    const v = model.runVolatility, leaves = model.nodes.reduce((n, node) => Math.max(n, node.leaf + 1), 0);
    if (model.runSymmetry || model.runConditioned || model.hidden || model.forest || model.projection || model.boost || model.meanCalibration
      || !model.clock.reversalClock || !Number.isFinite(v.cut) || v.cut < 0 || !Number.isFinite(v.prior) || v.prior < 0
      || !(v.globalPriorShare >= 0 && v.globalPriorShare <= 1)
      || !Number.isInteger(v.recentSamples) || v.recentSamples < 1 || !Number.isInteger(v.pooledHighSamples) || v.pooledHighSamples < 0
      || model.featureNames.length !== EVENT_RUN_VOLATILITY_FEATURES.length || !model.featureNames.every((name, i) => name === EVENT_RUN_VOLATILITY_FEATURES[i])
      || model.kernels.length !== 6 * leaves) throw new Error("Invalid run-volatility distribution");
  }
  if (model.runSymmetry || model.runConditioned) {
    const leaves = model.nodes.reduce((n, node) => Math.max(n, node.leaf + 1), 0);
    if (model.hidden || model.forest || model.projection || model.boost || model.meanCalibration
      || !model.clock.reversalClock || (model.runSymmetry && model.runConditioned)
      || (model.runConditioned && (!Number.isFinite(model.runConditioned.directionPrior) || model.runConditioned.directionPrior < 0))
      || !EVENT_RUN_FEATURES.every((name, i) => name === model.featureNames[i])
      || model.featureNames.length !== EVENT_RUN_FEATURES.length || model.kernels.length !== 3 * leaves)
      throw new Error("Invalid canonical run distribution");
    for (let leaf = 0; model.runSymmetry && leaf < leaves; leaf++) {
      const up = model.kernels[3 * leaf], down = model.kernels[3 * leaf + 1];
      if (up.length !== down.length || up.some((a, i) => {
        const b = down[i], side = a.next % 3;
        return Math.abs(a.probability - b.probability) > 1e-12 || Math.abs(b.return + a.return / (1 + a.return)) > 1e-10
          || Math.abs(b.low + a.high / (1 + a.high)) > 1e-10 || Math.abs(b.high + a.low / (1 + a.low)) > 1e-10
          || b.duration !== a.duration || b.next !== a.next - side + (side === 2 ? 2 : 1 - side);
      })) throw new Error("Broken run-reflection law");
      const neutral = model.kernels[3 * leaf + 2];
      if (neutral.length !== up.length + down.length || neutral.some((a, i) => {
        const b = i < up.length ? up[i] : down[i - up.length];
        return Math.abs(a.probability - b.probability / 2) > 1e-12 || a.return !== b.return
          || a.low !== b.low || a.high !== b.high || a.duration !== b.duration || a.next !== b.next;
      })) throw new Error("Broken neutral run-reflection mixture");
    }
  }
  if (model.hidden) {
    const h = model.hidden, n = h.transition.length;
    const probability = (row: number[], size: number) => row.length === size && row.every(v => Number.isFinite(v) && v >= 0)
      && Math.abs(row.reduce((s, v) => s + v, 0) - 1) < 1e-8;
    if (model.forest || model.projection || model.boost || n < 2 || n > 3
      || h.transition.some(row => !probability(row, n)) || h.emission.length !== n
      || h.emission.some(row => !probability(row, 15)) || h.beliefs.length !== model.kernels.length
      || h.beliefs.some(row => !probability(row, n)) || !Number.isInteger(h.initial) || !h.beliefs[h.initial]
      || h.nextByClass.length !== h.beliefs.length || h.nextByClass.some(row => row.length !== 15
        || row.some(v => !Number.isInteger(v) || !h.beliefs[v]))) throw new Error("Invalid hidden event distribution");
  }
  if (model.projection) {
    const p = model.projection;
    if (model.forest || [p.means, p.scales, p.coefficients].some(a => a.length !== model.featureNames.length || a.some(v => !Number.isFinite(v)))
      || p.scales.some(v => v <= 0) || !Number.isFinite(p.intercept) || !Number.isFinite(p.penalty) || p.penalty <= 0
      || p.cuts.length + 1 !== model.kernels.length || p.cuts.some((v, i) => !Number.isFinite(v) || (i > 0 && v <= p.cuts[i - 1])))
      throw new Error("Invalid event projection");
  }
  if (model.boost && (model.projection || model.forest || !Number.isFinite(model.boost.intercept)
    || !(model.boost.rate > 0 && model.boost.rate <= 1) || model.boost.cuts.length + 1 !== model.kernels.length
    || model.boost.cuts.some((v, i) => !Number.isFinite(v) || (i > 0 && v <= model.boost!.cuts[i - 1]))
    || model.boost.trees.some(t => !t.nodes.length || t.values.some(v => !Number.isFinite(v))))) throw new Error("Invalid event boosted forecast");
  for (const kernel of model.kernels) {
    let probability = 0;
    for (const atom of kernel) {
      if (!Number.isFinite(atom.probability) || atom.probability < 0 || !Number.isFinite(atom.return)
        || !Number.isFinite(atom.low) || !Number.isFinite(atom.high) || atom.low <= -1
        || atom.low > Math.min(0, atom.return) + 1e-12 || atom.high < Math.max(0, atom.return) - 1e-12
        || !Number.isFinite(atom.duration) || atom.duration <= 0 || !Number.isInteger(atom.next)
        || atom.next < 0 || atom.next >= model.kernels.length) throw new Error("Invalid joint event transition");
      probability += atom.probability;
    }
    if (Math.abs(probability - 1) > 1e-8) throw new Error("Event transition probabilities do not sum to one");
  }
}

/** Scale-only calibration of the conditional expectation. Exponential tilting
 * changes probabilities on observed JOINT outcomes, preserving duration, tail
 * excursions and successor-state dependence instead of shifting price paths. */
export function calibrateEventMean(model: EventDistribution, samples: readonly MoveSample[]): EventDistribution {
  if (!samples.length) throw new Error("Calibration requires resolved historical events");
  const means = model.kernels.map(k => k.reduce((s, a) => s + a.probability * a.return, 0));
  let xy = 0, xx = 0;
  for (const s of samples) { const x = means[eventLeaf(model, s.features)]; xy += x * s.return; xx += x * x; }
  const scale = xx > 1e-15 ? Math.max(0, Math.min(2, xy / xx)) : 0;
  return scaleEventMean(model, scale, samples.length);
}

export function scaleEventMean(model: EventDistribution, scale: number, samples = 0): EventDistribution {
  if (model.hidden) throw new Error("Hidden event laws require a consistent posterior update, not an independent mean tilt");
  if (model.runSymmetry || model.runConditioned) throw new Error("Canonical run laws cannot receive independent mean tilts");
  if (!Number.isFinite(scale) || scale < -2 || scale > 2 || !Number.isInteger(samples) || samples < 0)
    throw new Error("Invalid event mean scale");
  const means = model.kernels.map(k => k.reduce((s, a) => s + a.probability * a.return, 0));
  return { ...withEventMeans(model, means.map(mean => mean * scale)), meanCalibration: { scale, samples } };
}

function withEventMeans(model: EventDistribution, means: readonly number[]): EventDistribution {
  const kernels = model.kernels.map((kernel, leaf) => {
    const minimum = Math.min(...kernel.map(a => a.return)), maximum = Math.max(...kernel.map(a => a.return));
    const target = Math.max(minimum + 1e-12, Math.min(maximum - 1e-12, means[leaf]));
    if (maximum - minimum < 1e-10) return kernel.map(a => ({ ...a }));
    const weights = (theta: number) => {
      const logs = kernel.map(a => Math.log(a.probability) + theta * a.return * 1e4);
      const high = Math.max(...logs), raw = logs.map(v => Math.exp(v - high)), total = raw.reduce((a, b) => a + b, 0);
      return raw.map(v => v / total);
    };
    let lo = -10, hi = 10;
    for (let j = 0; j < 50; j++) {
      const mid = (lo + hi) / 2, p = weights(mid);
      const mean = p.reduce((s, v, i) => s + v * kernel[i].return, 0);
      if (mean < target) lo = mid; else hi = mid;
    }
    const p = weights((lo + hi) / 2);
    return kernel.map((a, i) => ({ ...a, probability: p[i] }));
  });
  const classProbabilities = kernels.map((kernel, leaf) => {
    const n = model.counts[leaf], probs = new Array<number>(15).fill(0.5 / (n + 7.5));
    for (const a of kernel) {
      probs[eventMoveLabel(a.return, a.duration, model.clock)] += a.probability * n / (n + 7.5);
    }
    return probs;
  });
  return { ...model, kernels, classProbabilities };
}
export function eventProjectionScore(p: EventProjection, features: readonly number[]): number {
  return p.intercept + p.coefficients.reduce((sum, coefficient, i) =>
    sum + coefficient * Math.max(-5, Math.min(5, (features[i] - p.means[i]) / p.scales[i])), 0);
}
export function eventBoostScore(boost: EventBoost, features: readonly number[]): number {
  return boost.intercept + boost.rate * boost.trees.reduce((sum, tree) => sum + tree.values[eventLeaf(tree, features)], 0);
}
/** A quantile can fall inside a tied score mass. Choose a nearby distinct
 * boundary instead of discarding that quantile and collapsing the state space. */
function eventScoreCuts(scores: readonly number[], cells: number, minLeaf: number): number[] {
  const boundaries: number[] = [];
  for (let i = minLeaf; i <= scores.length - minLeaf; i++) if (scores[i] - scores[i - 1] > 1e-12) boundaries.push(i);
  if (!boundaries.length) return [];
  const selected = new Set<number>();
  for (let cell = 1; cell < cells; cell++) {
    const target = cell * scores.length / cells;
    selected.add(boundaries.reduce((best, at) => Math.abs(at - target) < Math.abs(best - target) ? at : best, boundaries[0]));
  }
  let previous = 0;
  return [...selected].sort((a, b) => a - b).filter(at => {
    if (at - previous < minLeaf) return false;
    previous = at; return true;
  }).map(at => (scores[at - 1] + scores[at]) / 2);
}
export function eventLeaf(model: Pick<EventDistribution, "nodes" | "forest" | "projection" | "boost" | "hidden" | "runSymmetry" | "runConditioned" | "runVolatility"> & { featureNames?: readonly string[] }, features: readonly number[]): number {
  if (model.hidden) throw new Error("Event sequence model requires an observed belief state, not snapshot features");
  if ((model.featureNames && features.length !== model.featureNames.length) || features.some(v => !Number.isFinite(v))) throw new Error("Invalid event features");
  if (model.runVolatility) {
    const base = features.slice(0, EVENT_RUN_FEATURES.length), orientation = eventRunOrientation(base);
    const physical = 3 * eventLeaf({ nodes: model.nodes }, canonicalEventRunFeatures(base)) + (orientation > 0 ? 0 : orientation < 0 ? 1 : 2);
    return 2 * physical + Number(features[EVENT_RUN_FEATURES.length] > model.runVolatility.cut);
  }
  if (model.runSymmetry || model.runConditioned) {
    const orientation = eventRunOrientation(features);
    return 3 * eventLeaf({ nodes: model.nodes }, canonicalEventRunFeatures(features)) + (orientation > 0 ? 0 : orientation < 0 ? 1 : 2);
  }
  if (model.projection) {
    const score = eventProjectionScore(model.projection, features);
    const index = model.projection.cuts.findIndex(cut => score <= cut);
    return index < 0 ? model.projection.cuts.length : index;
  }
  if (model.boost) {
    const score = eventBoostScore(model.boost, features), index = model.boost.cuts.findIndex(cut => score <= cut);
    return index < 0 ? model.boost.cuts.length : index;
  }
  if (model.forest) {
    let state = 0, radix = 1;
    for (const tree of model.forest) { state += radix * eventLeaf(tree, features); radix *= tree.leafCount; }
    return state;
  }
  let n = model.nodes[0];
  while (n.leaf < 0) n = model.nodes[features[n.feature] <= n.cut ? n.left : n.right];
  return n.leaf;
}

/** Multinomial CART learns conditional move/duration distributions, followed by
 * Dirichlet-style shrinkage toward the unconditional empirical transition law.
 * Outcome atoms retain the joint return, duration, extrema, and successor leaf. */
export function trainEventDistribution(partitionSamples: readonly MoveSample[], clock: EventClock,
  options: { maxDepth: number; minLeaf: number; prior: number; criterion?: "distribution" | "mean"; honestyFraction?: number;
    /** Explicit separately purged outcomes; callers own temporal splitting. */
    estimationSamples?: readonly MoveSample[];
    /** Optional covariate-support guard for honest trees. Candidate splits still
     * use only partition outcomes, but both children must contain this many
     * separate estimation feature rows before either child can be created. */
    minimumEstimationLeaf?: number; featureNames?: readonly string[] }): EventDistribution {
  const samples = options.estimationSamples ? [...partitionSamples, ...options.estimationSamples] : partitionSamples;
  if (!samples.length || options.prior < 0 || options.minLeaf < 1 || options.maxDepth < 0) throw new Error("Invalid distribution training inputs");
  const featureNames = options.featureNames ?? EVENT_FEATURES;
  validateEventClock(clock); eventFeatureWarmup(clock, featureNames);
  if (!validFeatureNames(featureNames) || samples.some(s => s.features.length !== featureNames.length
    || s.nextFeatures.length !== featureNames.length)) throw new Error("Incompatible training feature contract");
  const honesty = options.honestyFraction ?? 0;
  if (!(honesty >= 0 && honesty < 1)) throw new Error("Invalid honest estimation fraction");
  if (options.minimumEstimationLeaf !== undefined && (!Number.isInteger(options.minimumEstimationLeaf)
    || options.minimumEstimationLeaf < 1 || !options.estimationSamples))
    throw new Error("Minimum estimation leaf requires explicit estimation samples and a positive integer");
  if (options.estimationSamples && (honesty || !partitionSamples.length || !options.estimationSamples.length))
    throw new Error("Explicit estimation requires nonempty separate populations and no fractional split");
  // A chronological honest split prevents a return observation from both
  // choosing a profitable-looking partition and estimating its trading edge.
  const separate = !!options.estimationSamples || !!honesty;
  const split = options.estimationSamples ? partitionSamples.length : honesty ? Math.floor(samples.length * (1 - honesty)) : samples.length;
  const partitionRows = samples.slice(0, split).map((_, i) => i);
  const estimationRows = separate ? samples.slice(split).map((_, i) => i + split) : partitionRows;
  if (!partitionRows.length || !estimationRows.length) throw new Error("Empty honest partition or estimation split");
  const nodes: DistributionNode[] = [], groups: number[][] = [];
  const impurity = (counts: number[], n: number) => n ? n - counts.reduce((s, v) => s + v * v, 0) / n : 0;
  const build = (rows: number[], supportRows: number[], depth: number): number => {
    const index = nodes.length;
    const node: DistributionNode = { feature: -1, cut: 0, left: -1, right: -1, leaf: -1 };
    nodes.push(node);
    const counts = new Array<number>(15).fill(0);
    for (const row of rows) counts[samples[row].label]++;
    const sum = rows.reduce((s, i) => s + samples[i].return, 0);
    let best = options.criterion === "mean" ? -sum * sum / rows.length : impurity(counts, rows.length), feature = -1, cut = 0;
    if (depth < options.maxDepth && rows.length >= options.minLeaf * 2
      && (!options.minimumEstimationLeaf || supportRows.length >= options.minimumEstimationLeaf * 2)) {
      for (let f = 0; f < featureNames.length; f++) {
        const ordered = rows.slice().sort((a, b) => samples[a].features[f] - samples[b].features[f]);
        const support = options.minimumEstimationLeaf
          ? supportRows.slice().sort((a, b) => samples[a].features[f] - samples[b].features[f]) : [];
        const left = new Array<number>(15).fill(0), right = counts.slice();
        let leftSum = 0, supportLeft = 0;
        for (let j = 0; j < ordered.length - 1; j++) {
          left[samples[ordered[j]].label]++; right[samples[ordered[j]].label]--;
          leftSum += samples[ordered[j]].return;
          const n = j + 1;
          if (n < options.minLeaf || rows.length - n < options.minLeaf) continue;
          const a = samples[ordered[j]].features[f], b = samples[ordered[j + 1]].features[f];
          if (a === b) continue;
          const candidateCut = (a + b) / 2;
          if (options.minimumEstimationLeaf) {
            while (supportLeft < support.length
              && samples[support[supportLeft]].features[f] <= candidateCut) supportLeft++;
            if (supportLeft < options.minimumEstimationLeaf
              || support.length - supportLeft < options.minimumEstimationLeaf) continue;
          }
          const score = options.criterion === "mean"
            ? -leftSum * leftSum / n - (sum - leftSum) ** 2 / (rows.length - n)
            : impurity(left, n) + impurity(right, rows.length - n);
          if (score < best - 1e-12) { best = score; feature = f; cut = candidateCut; }
        }
      }
    }
    if (feature < 0) { node.leaf = groups.length; groups.push(rows); }
    else {
      node.feature = feature; node.cut = cut;
      node.left = build(rows.filter(r => samples[r].features[feature] <= cut),
        supportRows.filter(r => samples[r].features[feature] <= cut), depth + 1);
      node.right = build(rows.filter(r => samples[r].features[feature] > cut),
        supportRows.filter(r => samples[r].features[feature] > cut), depth + 1);
    }
    return index;
  };
  build(partitionRows, estimationRows, 0);
  if (separate) {
    for (const group of groups) group.length = 0;
    for (const i of estimationRows) groups[eventLeaf({ nodes }, samples[i].features)].push(i);
  }
  const model: EventDistribution = { version: 1, clock, featureNames, nodes, kernels: [],
    counts: groups.map(g => g.length), classProbabilities: [], priorClasses: new Array<number>(15).fill(0), trainingSamples: estimationRows.length };
  populateEventKernels(model, samples, estimationRows, groups, options.prior);
  return model;
}

/** Re-estimate a plain tree's joint law without changing its state partition.
 * Callers own the split/availability contract; every supplied outcome is past
 * data at the deployment origin, even when it predates partition training.
 * Optional positive observation weights also weight the joint shrinkage prior.
 * Counts/trainingSamples remain raw observation counts, not effective mass. */
export function reestimateEventTree(model: EventDistribution, samples: readonly MoveSample[], prior: number,
  weights?: readonly number[]): EventDistribution {
  return reestimateEventTreeWithSources(model, samples, prior, weights).model;
}

/** Retain each joint atom's actual estimation row, including prior representatives.
 * This lets richer controlled transitions preserve the exact fitted mixture. */
export function reestimateEventTreeWithSources(model: EventDistribution, samples: readonly MoveSample[], prior: number,
  weights?: readonly number[]): { model: EventDistribution; sources: number[][] } {
  validateEventDistribution(model);
  if (model.forest || model.projection || model.boost || model.hidden || model.runSymmetry || model.runConditioned || model.runVolatility)
    throw new Error("Joint re-estimation requires a plain distribution tree");
  if (!samples.length || !Number.isFinite(prior) || prior < 0 || samples.some(s =>
    s.features.length !== model.featureNames.length || s.nextFeatures.length !== model.featureNames.length
    || [...s.features, ...s.nextFeatures, s.return, s.duration, s.low, s.high].some(v => !Number.isFinite(v))
    || s.duration <= 0 || s.low <= -1 || s.low > Math.min(0, s.return) || s.high < Math.max(0, s.return)
    || s.label !== eventMoveLabel(s.return, s.duration, model.clock))) throw new Error("Invalid joint re-estimation population");
  if (weights && (weights.length !== samples.length || weights.some(w => !Number.isFinite(w) || w <= 0)
    || !Number.isFinite(weights.reduce((sum, w) => sum + w, 0)))) throw new Error("Invalid joint estimation weights");
  const groups = model.kernels.map(() => [] as number[]), rows = samples.map((_, i) => i);
  for (const i of rows) groups[eventLeaf(model, samples[i].features)].push(i);
  const result: EventDistribution = { version: 1, clock: structuredClone(model.clock), featureNames: [...model.featureNames],
    nodes: structuredClone(model.nodes), kernels: [], counts: groups.map(g => g.length), classProbabilities: [],
    priorClasses: new Array<number>(15).fill(0), trainingSamples: samples.length };
  const sources = populateEventKernels(result, samples, rows, groups, prior, weights);
  validateEventDistribution(result);
  return { model: result, sources };
}

function populateEventKernels(model: EventDistribution, samples: readonly MoveSample[], estimationRows: readonly number[],
  groups: readonly number[][], priorStrength: number, weights?: readonly number[]): number[][] {
  const weight = (i: number) => weights?.[i] ?? 1, mass = estimationRows.reduce((sum, i) => sum + weight(i), 0);
  for (const i of estimationRows) model.priorClasses[samples[i].label] += weight(i) / mass;
  // Quantile quadrature for the shrinkage prior only; leaf observations are exact.
  const sorted = estimationRows.slice().sort((a, b) => samples[a].return - samples[b].return);
  const priorRows: number[] = [], priorWeights: number[] = [];
  for (let j = 0; j < sorted.length; j += Math.max(1, Math.ceil(sorted.length / 128))) {
    const end = Math.min(sorted.length, j + Math.max(1, Math.ceil(sorted.length / 128)));
    let binMass = 0;
    for (let k = j; k < end; k++) binMass += weight(sorted[k]);
    let representative = j, cumulative = weight(sorted[j]);
    while (representative + 1 < end && cumulative < binMass / 2) cumulative += weight(sorted[++representative]);
    priorRows.push(sorted[representative]); priorWeights.push(binMass / mass);
  }
  const sources: number[][] = [];
  for (const group of groups) {
    const prior = group.length ? priorStrength : Math.max(1, priorStrength);
    const total = group.reduce((sum, i) => sum + weight(i), 0) + prior;
    const atom = (i: number, probability: number): MoveAtom => {
      const s = samples[i];
      return { probability, return: s.return, low: s.low, high: s.high, duration: s.duration, next: eventLeaf(model, s.nextFeatures) };
    };
    model.kernels.push([...group.map(i => atom(i, weight(i) / total)),
      ...(prior ? priorRows.map((i, j) => atom(i, prior * priorWeights[j] / total)) : [])]);
    sources.push([...group, ...(prior ? priorRows : [])]);
    const probs = model.priorClasses.map(p => (p * prior + 0.5) / (total + 7.5));
    for (const i of group) probs[samples[i].label] += weight(i) / (total + 7.5);
    model.classProbabilities.push(probs);
  }
  return sources;
}

/** A regularized continuous directional forecast defines ordered finite states.
 * Empirical JOINT laws supply tails/duration/successors; a probability tilt makes
 * their means match the ridge forecast, so regularization also reaches utility. */
export function trainEventProjection(samples: readonly MoveSample[], clock: EventClock,
  options: { penalty: number; cells: number; minLeaf: number; prior: number; honestyFraction?: number; featureNames?: readonly string[] }): EventDistribution {
  const featureNames = options.featureNames ?? EVENT_FEATURES, width = featureNames.length;
  validateEventClock(clock); eventFeatureWarmup(clock, featureNames);
  if (!samples.length || !validFeatureNames(featureNames) || !Number.isFinite(options.penalty) || !(options.penalty > 0) || !Number.isInteger(options.cells)
    || options.cells < 1 || !Number.isInteger(options.minLeaf) || options.minLeaf < 1 || !Number.isFinite(options.prior) || options.prior < 0 || samples.some(s => s.features.length !== width
      || s.nextFeatures.length !== width || [...s.features, ...s.nextFeatures, s.return].some(v => !Number.isFinite(v))))
    throw new Error("Invalid projection training inputs");
  const honesty = options.honestyFraction ?? 0;
  if (!(honesty >= 0 && honesty < 1)) throw new Error("Invalid honest estimation fraction");
  const split = honesty ? Math.floor(samples.length * (1 - honesty)) : samples.length;
  const partition = samples.slice(0, split), estimationRows = Array.from({ length: samples.length - (honesty ? split : 0) }, (_, i) => i + (honesty ? split : 0));
  if (partition.length < 2 || !estimationRows.length) throw new Error("Empty projection partition/estimation data");
  const means = Array.from({ length: width }, (_, f) => partition.reduce((s, row) => s + row.features[f], 0) / partition.length);
  const scales = means.map((mean, f) => Math.max(1e-8, Math.sqrt(partition.reduce((s, row) => s + (row.features[f] - mean) ** 2, 0) / partition.length)));
  const intercept = partition.reduce((s, row) => s + row.return, 0) / partition.length;
  const matrix = Array.from({ length: width }, () => new Array<number>(width + 1).fill(0));
  for (const row of partition) {
    const x = row.features.map((v, i) => Math.max(-5, Math.min(5, (v - means[i]) / scales[i])));
    for (let i = 0; i < width; i++) {
      matrix[i][width] += x[i] * (row.return - intercept);
      for (let j = 0; j < width; j++) matrix[i][j] += x[i] * x[j];
    }
  }
  for (let i = 0; i < width; i++) matrix[i][i] += options.penalty * partition.length;
  // Pivoted Gaussian elimination is tiny for these feature sets.
  for (let column = 0; column < width; column++) {
    let pivot = column;
    for (let i = column + 1; i < width; i++) if (Math.abs(matrix[i][column]) > Math.abs(matrix[pivot][column])) pivot = i;
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    const divisor = matrix[column][column];
    if (!(Math.abs(divisor) > 1e-14)) throw new Error("Singular regularized projection");
    for (let j = column; j <= width; j++) matrix[column][j] /= divisor;
    for (let i = 0; i < width; i++) if (i !== column) {
      const factor = matrix[i][column];
      for (let j = column; j <= width; j++) matrix[i][j] -= factor * matrix[column][j];
    }
  }
  const projection: EventProjection = { means, scales, coefficients: matrix.map(row => row[width]), intercept, cuts: [], penalty: options.penalty };
  const scores = partition.map(s => eventProjectionScore(projection, s.features)).sort((a, b) => a - b);
  const cells = Math.max(1, Math.min(options.cells, Math.floor(partition.length / options.minLeaf)));
  projection.cuts = eventScoreCuts(scores, cells, options.minLeaf);
  const model: EventDistribution = { version: 1, clock, featureNames, nodes: [], projection, kernels: [],
    counts: [], classProbabilities: [], priorClasses: new Array<number>(15).fill(0), trainingSamples: estimationRows.length };
  const groups = Array.from({ length: projection.cuts.length + 1 }, () => [] as number[]);
  for (const i of estimationRows) groups[eventLeaf(model, samples[i].features)].push(i);
  model.counts = groups.map(group => group.length);
  populateEventKernels(model, samples, estimationRows, groups, options.prior);
  const targetMeans = groups.map(group => group.length
    ? group.reduce((sum, i) => sum + eventProjectionScore(projection, samples[i].features), 0) / group.length : intercept);
  return withEventMeans(model, targetMeans);
}

/** Small histogram gradient boosting for conditional means, followed by the
 * same joint empirical transition law as the linear projection. */
export function trainEventBoost(samples: readonly MoveSample[], clock: EventClock,
  options: { iterations: number; rate: number; cells: number; minLeaf: number; prior: number; featureNames?: readonly string[] }): EventDistribution {
  const featureNames = options.featureNames ?? EVENT_FEATURES, width = featureNames.length;
  validateEventClock(clock); eventFeatureWarmup(clock, featureNames);
  if (!samples.length || !validFeatureNames(featureNames) || !Number.isInteger(options.iterations) || options.iterations < 1
    || !(options.rate > 0 && options.rate <= 1) || !Number.isInteger(options.minLeaf) || options.minLeaf < 1
    || !Number.isFinite(options.prior) || options.prior < 0 || !Number.isInteger(options.cells) || options.cells < 1
    || samples.some(s => s.features.length !== width || s.nextFeatures.length !== width
      || [...s.features, ...s.nextFeatures, s.return].some(v => !Number.isFinite(v)))) throw new Error("Invalid boosted event inputs");
  const cuts = Array.from({ length: width }, (_, feature) => {
    const ordered = samples.map(s => s.features[feature]).sort((a, b) => a - b), out: number[] = [];
    for (let i = 1; i < 32; i++) {
      const at = Math.floor(i * ordered.length / 32), cut = (ordered[at - 1] + ordered[at]) / 2;
      if (at > 0 && ordered[at] > ordered[at - 1] && (!out.length || cut > out.at(-1)!)) out.push(cut);
    }
    return out;
  });
  const bins = samples.map(s => cuts.map((cs, f) => { const bin = cs.findIndex(c => s.features[f] <= c); return bin < 0 ? cs.length : bin; }));
  const intercept = samples.reduce((s, row) => s + row.return, 0) / samples.length;
  const predictions = new Float64Array(samples.length).fill(intercept);
  const boost: EventBoost = { intercept, rate: options.rate, trees: [], cuts: [] };
  const all = samples.map((_, i) => i);
  for (let iteration = 0; iteration < options.iterations; iteration++) {
    const residual = samples.map((row, i) => row.return - predictions[i]);
    const nodes: DistributionNode[] = [], values: number[] = [];
    const build = (rows: number[], depth: number): number => {
      const index = nodes.length, total = rows.reduce((s, i) => s + residual[i], 0);
      const node: DistributionNode = { feature: -1, cut: 0, left: -1, right: -1, leaf: -1 }; nodes.push(node);
      let best = total * total / rows.length, bestFeature = -1, bestBin = -1;
      if (depth < 2 && rows.length >= options.minLeaf * 2) for (let f = 0; f < width; f++) {
        const count = new Uint32Array(cuts[f].length + 1), sum = new Float64Array(count.length);
        for (const i of rows) { count[bins[i][f]]++; sum[bins[i][f]] += residual[i]; }
        let leftCount = 0, leftSum = 0;
        for (let b = 0; b < cuts[f].length; b++) {
          leftCount += count[b]; leftSum += sum[b];
          if (leftCount < options.minLeaf || rows.length - leftCount < options.minLeaf) continue;
          const score = leftSum ** 2 / leftCount + (total - leftSum) ** 2 / (rows.length - leftCount);
          if (score > best + 1e-12) { best = score; bestFeature = f; bestBin = b; }
        }
      }
      if (bestFeature < 0) { node.leaf = values.length; values.push(total / rows.length); }
      else {
        node.feature = bestFeature; node.cut = cuts[bestFeature][bestBin];
        node.left = build(rows.filter(i => bins[i][bestFeature] <= bestBin), depth + 1);
        node.right = build(rows.filter(i => bins[i][bestFeature] > bestBin), depth + 1);
      }
      return index;
    };
    build(all, 0);
    const tree = { nodes, values }; boost.trees.push(tree);
    for (const i of all) predictions[i] += options.rate * values[eventLeaf(tree, samples[i].features)];
  }
  const scores = Array.from(predictions).sort((a, b) => a - b), cells = Math.max(1, Math.min(options.cells, Math.floor(samples.length / options.minLeaf)));
  boost.cuts = eventScoreCuts(scores, cells, options.minLeaf);
  const model: EventDistribution = { version: 1, clock, featureNames, nodes: [], boost, kernels: [], counts: [],
    classProbabilities: [], priorClasses: new Array(15).fill(0), trainingSamples: samples.length };
  const groups = Array.from({ length: boost.cuts.length + 1 }, () => [] as number[]);
  for (const i of all) groups[eventLeaf(model, samples[i].features)].push(i);
  model.counts = groups.map(group => group.length);
  populateEventKernels(model, samples, all, groups, options.prior);
  return withEventMeans(model, groups.map(group => group.reduce((s, i) => s + predictions[i], 0) / group.length));
}

/** Block-bagged shallow partitions. The Bellman state is the JOINT leaf
 * signature, so the future action never gets to know which ensemble member
 * generated an outcome. Ensemble predictive laws are averaged before max. */
export function trainEventForest(samples: readonly MoveSample[], clock: EventClock,
  options: { trees: number; maxDepth: number; minLeaf: number; prior: number; seed: number; featureNames?: readonly string[] }): EventDistribution {
  if (options.trees < 1 || options.trees > 4) throw new Error("Use 1–4 shallow forest trees");
  const days = new Map<number, MoveSample[]>();
  for (const s of samples) {
    const day = Math.floor(s.start * eventCandleIntervalMs(clock) / 86400000);
    if (!days.has(day)) days.set(day, []);
    days.get(day)!.push(s);
  }
  const blocks = [...days.values()];
  let randomState = options.seed >>> 0;
  const random = () => {
    randomState = (Math.imul(1664525, randomState) + 1013904223) >>> 0;
    return randomState / 0x100000000;
  };
  const forest = Array.from({ length: options.trees }, () => {
    const bootstrap = Array.from({ length: blocks.length }, () => blocks[Math.floor(random() * blocks.length)]).flat();
    const tree = trainEventDistribution(bootstrap, clock, { maxDepth: options.maxDepth,
      minLeaf: options.minLeaf, prior: 0, criterion: "mean", featureNames: options.featureNames });
    return { nodes: tree.nodes, leafCount: tree.kernels.length };
  });
  const states = forest.reduce((n, t) => n * t.leafCount, 1);
  if (states > 128) throw new Error("Forest joint state exceeds the cheap 128-state screen budget");
  const model: EventDistribution = { version: 1, clock, featureNames: options.featureNames ?? EVENT_FEATURES, nodes: [], forest,
    kernels: [], counts: [], classProbabilities: [], priorClasses: new Array(15).fill(0), trainingSamples: samples.length };
  const groups = forest.map(t => Array.from({ length: t.leafCount }, () => [] as number[]));
  const next = samples.map(s => eventLeaf(model, s.nextFeatures));
  for (let i = 0; i < samples.length; i++) {
    for (let t = 0; t < forest.length; t++) groups[t][eventLeaf(forest[t], samples[i].features)].push(i);
    model.priorClasses[samples[i].label] += 1 / samples.length;
  }
  for (let state = 0; state < states; state++) {
    let residual = state;
    const selected = forest.map((t, index) => { const leaf = residual % t.leafCount; residual = Math.floor(residual / t.leafCount); return groups[index][leaf]; });
    const count = Math.min(...selected.map(g => g.length));
    const mixture = count / (count + options.prior);
    const weights = new Float64Array(samples.length).fill((1 - mixture) / samples.length);
    for (const group of selected) for (const i of group) weights[i] += mixture / (forest.length * group.length);
    // Preserve next-state and class probabilities exactly. Within each stratum,
    // a two-endpoint quadrature preserves mean return and duration. The endpoint
    // spread is conservative for one-period concave utility at that mean.
    const strata = new Map<number, { mass: number; sum: number; duration: number; min: number; max: number; low: number; high: number; next: number }>();
    const probs = new Array<number>(15).fill(0.5 / (count + 7.5));
    for (let i = 0; i < samples.length; i++) {
      const weight = weights[i]; if (!weight) continue;
      const s = samples[i], key = next[i] * 15 + s.label;
      const g = strata.get(key) ?? { mass: 0, sum: 0, duration: 0, min: Infinity, max: -Infinity, low: 0, high: 0, next: next[i] };
      g.mass += weight; g.sum += weight * s.return; g.duration += weight * s.duration;
      g.min = Math.min(g.min, s.return); g.max = Math.max(g.max, s.return);
      g.low = Math.min(g.low, s.low); g.high = Math.max(g.high, s.high); strata.set(key, g);
      probs[s.label] += weight * count / (count + 7.5);
    }
    const atoms: MoveAtom[] = [];
    for (const g of strata.values()) {
      const base = { low: g.low, high: g.high, duration: g.duration / g.mass, next: g.next };
      if (g.max - g.min < 1e-12) atoms.push({ ...base, return: g.min, probability: g.mass });
      else {
        const upper = Math.max(0, Math.min(1, (g.sum / g.mass - g.min) / (g.max - g.min)));
        if (upper < 1) atoms.push({ ...base, return: g.min, probability: g.mass * (1 - upper) });
        if (upper > 0) atoms.push({ ...base, return: g.max, probability: g.mass * upper });
      }
    }
    model.kernels.push(atoms); model.counts.push(count); model.classProbabilities.push(probs);
  }
  return model;
}

export function distributionMetrics(model: EventDistribution, samples: readonly MoveSample[]) {
  let nll = 0, baselineNll = 0, brier = 0, mse = 0, zeroMse = 0, mean = 0;
  const expectations = model.kernels.map(k => k.reduce((s, a) => s + a.probability * a.return, 0));
  let belief = model.hidden?.initial ?? 0, previousEnd = -1, previousSeries = samples[0]?.series;
  for (const s of samples) {
    if (model.hidden && (s.start !== previousEnd || s.series !== previousSeries)) belief = model.hidden.initial;
    const leaf = model.hidden ? belief : eventLeaf(model, s.features), probs = model.classProbabilities[leaf];
    nll -= Math.log(Math.max(1e-12, probs[s.label]));
    baselineNll -= Math.log((model.priorClasses[s.label] * model.trainingSamples + 0.5) / (model.trainingSamples + 7.5));
    brier += probs.reduce((v, p, j) => v + (p - Number(j === s.label)) ** 2, 0);
    mse += (s.return - expectations[leaf]) ** 2; zeroMse += s.return ** 2; mean += expectations[leaf];
    if (model.hidden) belief = eventHiddenNext(model, belief, s.label);
    previousEnd = s.end; previousSeries = s.series;
  }
  const n = samples.length;
  return { samples: n, nll: nll / n, baselineNll: baselineNll / n, brier: brier / n,
    mseSkill: 1 - mse / zeroMse, predictedMeanBps: mean / n * 1e4 };
}
